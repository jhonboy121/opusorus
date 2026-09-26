//! Port of dnn/dred_encoder.c and dred_encoder.h: the DRED encoder (`DREDEnc`), i.e. the
//! 16 kHz conversion, LPCNet feature extraction + RDOVAE latent computation per 20 ms double
//! frame, and the entropy coding of the DRED payload for a SILK/hybrid/CELT packet.
//!
//! Entry points for the Opus encoder unit (src/opus_encoder.c):
//! * [`DredEnc::new`] (`dred_encoder_init`), [`DredEnc::dred_encoder_reset`],
//!   [`DredEnc::load_model`] (`dred_encoder_load_model`, OPUS_SET_DNN_BLOB), field `loaded`;
//! * [`dred_compute_latents`] (called before SILK, with `&pcm_buf[total_buffer*channels..]`);
//! * [`dred_encode_silk_frame`] (writes the payload after the experimental header); the
//!   encoder also clears `latents_buffer_fill` when DRED is off.
//!
//! The model is never compiled in (PLAN D-015): a new encoder is not `loaded` until a weight
//! blob holding the RDOVAE encoder and pitch DNN arrays is loaded.

use alloc::boxed::Box;
use alloc::sync::Arc;

use crate::celt::arch::VERY_SMALL;
use crate::celt::entenc::EcEnc;
use crate::celt::laplace::ec_laplace_encode_p0;
use crate::celt::mathops::float2int16;
use crate::math;
use crate::{Error, Result};

use super::dred_coding::{
    DRED_DFRAME_SIZE, DRED_FRAME_SIZE, DRED_MAX_FRAMES, DRED_NUM_REDUNDANCY_FRAMES,
    DRED_SILK_ENCODER_DELAY, compute_quantizer,
};
use super::dred_rdovae::{
    DRED_LATENT_DEAD_ZONE_Q8, DRED_LATENT_DIM, DRED_LATENT_P0_Q8, DRED_LATENT_QUANT_SCALES_Q8,
    DRED_LATENT_R_Q8, DRED_NUM_FEATURES, DRED_STATE_DEAD_ZONE_Q8, DRED_STATE_DIM, DRED_STATE_P0_Q8,
    DRED_STATE_QUANT_SCALES_Q8, DRED_STATE_R_Q8, RdovaeEnc, RdovaeEncState,
    dred_rdovae_enc_load_shared, dred_rdovae_encode_dframe,
};
use super::lpcnet_enc::{
    LpcnetEncState, NB_TOTAL_FEATURES, lpcnet_compute_single_frame_features_float,
};
use super::nnet::ACTIVATION_TANH;
use super::nnet_arch::compute_activation_inplace;

/// `RESAMPLING_ORDER`.
pub const RESAMPLING_ORDER: usize = 8;

/// `MAX_DOWNMIX_BUFFER` (dred_encoder.c).
#[cfg(feature = "qext")]
pub const MAX_DOWNMIX_BUFFER: usize = 1920 * 2;
/// `MAX_DOWNMIX_BUFFER` (dred_encoder.c).
#[cfg(not(feature = "qext"))]
pub const MAX_DOWNMIX_BUFFER: usize = 960 * 2;

/// Size of the `activity_mem` array of the C `OpusEncoder` (2.5 ms resolution).
pub const DRED_ACTIVITY_MEM_SIZE: usize = DRED_MAX_FRAMES * 4;

/// `DREDEnc` (dred_encoder.h). Fields after `resample_mem`'s C position are Rust-only scratch.
#[derive(Debug, Clone, PartialEq)]
pub struct DredEnc {
    pub model: Arc<RdovaeEnc>,
    pub lpcnet_enc_state: LpcnetEncState,
    pub rdovae_enc: RdovaeEncState,
    pub loaded: bool,
    pub fs: i32,
    pub channels: i32,

    // DREDENC_RESET_START
    pub input_buffer: [f32; 2 * DRED_DFRAME_SIZE],
    pub input_buffer_fill: i32,
    pub dred_offset: i32,
    pub latent_offset: i32,
    pub last_extra_dred_offset: i32,
    pub latents_buffer: [f32; DRED_MAX_FRAMES * DRED_LATENT_DIM],
    pub latents_buffer_fill: i32,
    pub state_buffer: [f32; DRED_MAX_FRAMES * DRED_STATE_DIM],
    pub resample_mem: [f32; RESAMPLING_ORDER + 1],

    /// Scratch for the C stack array `downmix[MAX_DOWNMIX_BUFFER]` of `dred_convert_to_16k`
    /// (not part of the C state).
    downmix: [f32; MAX_DOWNMIX_BUFFER],
}

impl DredEnc {
    /// Port of dnn/dred_encoder.c:dred_encoder_init on a new heap-allocated encoder. Upstream
    /// binds the compiled-in model; here the encoder is not `loaded` until
    /// [`Self::load_model`] (the upstream `USE_WEIGHTS_FILE` behaviour).
    #[must_use]
    pub fn new(fs: i32, channels: i32) -> Box<Self> {
        let mut enc = Box::new(Self {
            model: Arc::default(),
            lpcnet_enc_state: LpcnetEncState::default(),
            rdovae_enc: RdovaeEncState::new(),
            loaded: false,
            fs,
            channels,
            input_buffer: [0.0; 2 * DRED_DFRAME_SIZE],
            input_buffer_fill: 0,
            dred_offset: 0,
            latent_offset: 0,
            last_extra_dred_offset: 0,
            latents_buffer: [0.0; DRED_MAX_FRAMES * DRED_LATENT_DIM],
            latents_buffer_fill: 0,
            state_buffer: [0.0; DRED_MAX_FRAMES * DRED_STATE_DIM],
            resample_mem: [0.0; RESAMPLING_ORDER + 1],
            downmix: [0.0; MAX_DOWNMIX_BUFFER],
        });
        enc.dred_encoder_init(fs, channels);
        enc
    }

    /// Port of dnn/dred_encoder.c:dred_encoder_init. Clears `loaded` like the upstream
    /// `USE_WEIGHTS_FILE` build (no compiled-in model); a previously loaded model stays in
    /// memory but must be loaded again to be used.
    pub fn dred_encoder_init(&mut self, fs: i32, channels: i32) {
        self.fs = fs;
        self.channels = channels;
        self.loaded = false;
        self.dred_encoder_reset();
    }

    /// Port of dnn/dred_encoder.c:dred_encoder_load_model. `data` is a weight blob holding the
    /// RDOVAE encoder arrays and the pitch DNN arrays (used by the LPCNet feature extraction).
    ///
    /// On an RDOVAE binding failure nothing changes (C may leave a partially overwritten
    /// model); when the pitch DNN part fails the RDOVAE model is already replaced, as in C.
    pub fn load_model(&mut self, data: &[u8]) -> Result<()> {
        let model = dred_rdovae_enc_load_shared(data).map_err(|_| Error::BadArg)?;
        self.model = model;
        self.lpcnet_enc_state
            .load_model(data)
            .map_err(|_| Error::BadArg)?;
        self.loaded = true;
        Ok(())
    }

    /// Port of dnn/dred_encoder.c:dred_encoder_reset: clears everything from
    /// `DREDENC_RESET_START` (`input_buffer`) on, then re-inits the LPCNet and RDOVAE states.
    pub fn dred_encoder_reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.input_buffer_fill = 0;
        self.dred_offset = 0;
        self.latent_offset = 0;
        self.last_extra_dred_offset = 0;
        self.latents_buffer.fill(0.0);
        self.latents_buffer_fill = 0;
        self.state_buffer.fill(0.0);
        self.resample_mem.fill(0.0);
        self.input_buffer_fill = DRED_SILK_ENCODER_DELAY;
        self.lpcnet_enc_state.lpcnet_encoder_init();
        dred_rdovae_init_encoder(&mut self.rdovae_enc);
    }

    /// Port of dnn/dred_encoder.c:dred_convert_to_16k (static) with the C signature: converts
    /// `in_len` samples per channel of `input` at `self.fs` to `out_len` samples at 16 kHz.
    pub fn dred_convert_to_16k(
        &mut self,
        input: &[f32],
        in_len: usize,
        out: &mut [f32],
        out_len: usize,
    ) {
        dred_convert_to_16k(
            self.fs,
            self.channels,
            &mut self.resample_mem,
            &mut self.downmix,
            input,
            in_len,
            out,
            out_len,
        );
    }
}

/// Port of dnn/dred_encoder.c:DRED_rdovae_init_encoder (static): `memset(0)`.
const fn dred_rdovae_init_encoder(enc_state: &mut RdovaeEncState) {
    *enc_state = RdovaeEncState::new();
}

/// Port of dnn/dred_encoder.c:dred_process_frame (static): computes the features of the 20 ms
/// double frame at the start of `input_buffer` and runs the RDOVAE encoder on them.
pub fn dred_process_frame(enc: &mut DredEnc) {
    let mut feature_buffer = [0f32; 2 * 36];
    let mut input_buffer = [0f32; 2 * DRED_NUM_FEATURES];

    celt_assert!(enc.loaded);
    // shift latents buffer
    enc.latents_buffer
        .copy_within(0..(DRED_MAX_FRAMES - 1) * DRED_LATENT_DIM, DRED_LATENT_DIM);
    enc.state_buffer
        .copy_within(0..(DRED_MAX_FRAMES - 1) * DRED_STATE_DIM, DRED_STATE_DIM);

    // calculate LPCNet features
    lpcnet_compute_single_frame_features_float(
        &mut enc.lpcnet_enc_state,
        &enc.input_buffer,
        &mut feature_buffer,
    );
    lpcnet_compute_single_frame_features_float(
        &mut enc.lpcnet_enc_state,
        &enc.input_buffer[DRED_FRAME_SIZE..],
        &mut feature_buffer[NB_TOTAL_FEATURES..],
    );

    // prepare input buffer (discard LPC coefficients)
    input_buffer[..DRED_NUM_FEATURES].copy_from_slice(&feature_buffer[..DRED_NUM_FEATURES]);
    input_buffer[DRED_NUM_FEATURES..].copy_from_slice(&feature_buffer[36..36 + DRED_NUM_FEATURES]);

    // run RDOVAE encoder
    dred_rdovae_encode_dframe(
        &mut enc.rdovae_enc,
        &enc.model,
        &mut enc.latents_buffer,
        &mut enc.state_buffer,
        &input_buffer,
    );
    enc.latents_buffer_fill = (enc.latents_buffer_fill + 1).min(DRED_NUM_REDUNDANCY_FRAMES as i32);
}

/// Port of dnn/dred_encoder.c:filter_df2t for `out != in`: a direct-form II transposed IIR
/// filter of order `order` (`b`/`a` hold `order` taps after `b0`; `mem` holds `order+1`).
#[expect(clippy::too_many_arguments, reason = "mirrors C signature")]
pub fn filter_df2t(
    input: &[f32],
    out: &mut [f32],
    len: usize,
    b0: f32,
    b: &[f32],
    a: &[f32],
    order: usize,
    mem: &mut [f32],
) {
    for (o, &xi) in out[..len].iter_mut().zip(&input[..len]) {
        *o = filter_df2t_sample(xi, b0, b, a, order, mem);
    }
}

/// Port of dnn/dred_encoder.c:filter_df2t for the in-place call (`out == in`).
pub fn filter_df2t_inplace(
    x: &mut [f32],
    len: usize,
    b0: f32,
    b: &[f32],
    a: &[f32],
    order: usize,
    mem: &mut [f32],
) {
    for v in &mut x[..len] {
        *v = filter_df2t_sample(*v, b0, b, a, order, mem);
    }
}

/// One iteration of the filter_df2t loop.
#[inline(always)]
fn filter_df2t_sample(
    xi: f32,
    b0: f32,
    b: &[f32],
    a: &[f32],
    order: usize,
    mem: &mut [f32],
) -> f32 {
    let yi = xi * b0 + mem[0];
    let nyi = -yi;
    for j in 0..order {
        mem[j] = mem[j + 1] + b[j] * xi + a[j] * nyi;
    }
    yi
}

/// Filter coefficients of dred_convert_to_16k: `(b0, filter_b, filter_a)`.
type Coefs = (f32, [f32; 8], [f32; 8]);

/// `ellip(7, .2, 70, 7750/24000)` (48 kHz and 24 kHz).
const COEFS_48K: Coefs = (
    0.004523418224,
    [
        0.005873358047,
        0.012980854831,
        0.014531340042,
        0.014531340042,
        0.012980854831,
        0.005873358047,
        0.004523418224,
        0.,
    ],
    [
        -3.878718597768,
        7.748834257468,
        -9.653651699533,
        8.007342726666,
        -4.379450178552,
        1.463182111810,
        -0.231720677804,
        0.,
    ],
);
/// `ellip(7, .2, 70, 5800/24000)` (12 kHz).
const COEFS_12K: Coefs = (
    0.002033596776,
    [
        -0.001017101081,
        0.003673127243,
        0.001009165267,
        0.001009165267,
        0.003673127243,
        -0.001017101081,
        0.002033596776,
        0.,
    ],
    [
        -4.930414411612,
        11.291643096504,
        -15.322037343815,
        13.216403930898,
        -7.220409219553,
        2.310550142771,
        -0.334338618782,
        0.,
    ],
);
/// `ellip(7, .2, 70, 3900/8000)` (8 kHz).
const COEFS_8K: Coefs = (
    0.020109185709,
    [
        0.081670120929,
        0.180401598565,
        0.259391051971,
        0.259391051971,
        0.180401598565,
        0.081670120929,
        0.020109185709,
        0.,
    ],
    [
        -1.393651933659,
        2.609789872676,
        -2.403541968806,
        2.056814957331,
        -1.148908574570,
        0.473001413788,
        -0.110359852412,
        0.,
    ],
);
/// `ellip(7, .2, 70, 7750/48000)` (96 kHz).
#[cfg(feature = "qext")]
const COEFS_96K: Coefs = (
    0.000880286074,
    [
        -0.002160290245,
        0.002887088080,
        -0.001214921271,
        -0.001214921271,
        0.002887088080,
        -0.002160290245,
        0.000880286074,
        0.,
    ],
    [
        -5.813483928050,
        14.932091805554,
        -21.900933283269,
        19.774128964756,
        -10.978028462771,
        3.467650469467,
        -0.480641240411,
        0.,
    ],
);

/// Port of dnn/dred_encoder.c:dred_convert_to_16k (static): downmixes `in_len` samples of
/// `input` (interleaved, `enc.channels`) at `enc.fs` and resamples them to `out_len` samples at
/// 16 kHz (int16 scale). Unsupported rates are a (debug) assertion failure, as in C.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors C signature (plus split state)"
)]
fn dred_convert_to_16k(
    enc_fs: i32,
    enc_channels: i32,
    resample_mem: &mut [f32; RESAMPLING_ORDER + 1],
    downmix: &mut [f32; MAX_DOWNMIX_BUFFER],
    input: &[f32],
    in_len: usize,
    out: &mut [f32],
    out_len: usize,
) {
    celt_assert!(enc_channels as usize * in_len <= MAX_DOWNMIX_BUFFER);
    celt_assert!(in_len as i64 * 16000 == out_len as i64 * i64::from(enc_fs));
    let up: usize = match enc_fs {
        8000 => 2,
        12000 => 4,
        16000 => 1,
        24000 => 2,
        48000 => 1,
        #[cfg(feature = "qext")]
        96000 => 1,
        _ => {
            celt_assert!(false, "unsupported DRED rate {enc_fs}");
            // C leaves `up` uninitialized and later asserts again; bail out.
            return;
        }
    };
    celt_assert!(up * in_len <= MAX_DOWNMIX_BUFFER);
    downmix[..up * in_len].fill(0.0);
    if enc_channels == 1 {
        for i in 0..in_len {
            downmix[up * i] = f32::from(float2int16(up as f32 * input[i])) + VERY_SMALL;
        }
    } else {
        for i in 0..in_len {
            // `.5*up*(...)` is a double expression converted to float by FLOAT2INT16.
            let v = 0.5 * up as f64 * f64::from(input[2 * i] + input[2 * i + 1]);
            downmix[up * i] = f32::from(float2int16(v as f32)) + VERY_SMALL;
        }
    }
    if enc_fs == 16000 {
        out[..out_len].copy_from_slice(&downmix[..out_len]);
    } else if enc_fs == 48000 || enc_fs == 24000 {
        let (b0, fb, fa) = &COEFS_48K;
        filter_df2t_inplace(
            downmix,
            up * in_len,
            *b0,
            fb,
            fa,
            RESAMPLING_ORDER,
            resample_mem,
        );
        for i in 0..out_len {
            out[i] = downmix[3 * i];
        }
    } else if enc_fs == 12000 {
        let (b0, fb, fa) = &COEFS_12K;
        filter_df2t_inplace(
            downmix,
            up * in_len,
            *b0,
            fb,
            fa,
            RESAMPLING_ORDER,
            resample_mem,
        );
        for i in 0..out_len {
            out[i] = downmix[3 * i];
        }
    } else if enc_fs == 8000 {
        let (b0, fb, fa) = &COEFS_8K;
        filter_df2t(
            &downmix[..],
            out,
            out_len,
            *b0,
            fb,
            fa,
            RESAMPLING_ORDER,
            resample_mem,
        );
    } else {
        #[cfg(feature = "qext")]
        if enc_fs == 96000 {
            let (b0, fb, fa) = &COEFS_96K;
            filter_df2t_inplace(
                downmix,
                up * in_len,
                *b0,
                fb,
                fa,
                RESAMPLING_ORDER,
                resample_mem,
            );
            for i in 0..out_len {
                out[i] = downmix[6 * i];
            }
            return;
        }
        celt_assert!(false, "unsupported DRED rate {enc_fs}");
    }
}

/// Port of dnn/dred_encoder.c:dred_compute_latents.
///
/// `pcm` holds `frame_size` interleaved samples per channel at `enc.fs` (float, ±1 scale);
/// `extra_delay` is the encoder look-ahead in samples at `enc.fs`. Must only be called on a
/// loaded encoder (C asserts).
///
/// Quirk (ported as-is): for frames longer than 20 ms the input pointer advances by
/// `process_size` samples, not `process_size*channels`, so stereo input is re-read.
pub fn dred_compute_latents(enc: &mut DredEnc, pcm: &[f32], frame_size: i32, extra_delay: i32) {
    let mut frame_size16k = frame_size * 16000 / enc.fs;
    celt_assert!(enc.loaded);
    let curr_offset16k = 40 + extra_delay * 16000 / enc.fs - enc.input_buffer_fill;
    enc.dred_offset = math::floor(f64::from((curr_offset16k as f32 + 20.0f32) / 40.0f32)) as i32;
    enc.latent_offset = 0;
    let mut pcm_off = 0usize;
    while frame_size16k > 0 {
        let process_size16k = (2 * DRED_FRAME_SIZE as i32).min(frame_size16k);
        let process_size = process_size16k * enc.fs / 16000;
        let fill = enc.input_buffer_fill as usize;
        dred_convert_to_16k(
            enc.fs,
            enc.channels,
            &mut enc.resample_mem,
            &mut enc.downmix,
            &pcm[pcm_off..],
            process_size as usize,
            &mut enc.input_buffer[fill..],
            process_size16k as usize,
        );
        enc.input_buffer_fill += process_size16k;
        if enc.input_buffer_fill >= 2 * DRED_FRAME_SIZE as i32 {
            // C: `curr_offset16k += 320;` (never read again).
            dred_process_frame(enc);
            enc.input_buffer_fill -= 2 * DRED_FRAME_SIZE as i32;
            let fill = enc.input_buffer_fill as usize;
            enc.input_buffer
                .copy_within(2 * DRED_FRAME_SIZE..2 * DRED_FRAME_SIZE + fill, 0);
            // 15 ms (6*2.5 ms) is the ideal offset for DRED because it corresponds to our
            // vocoder look-ahead.
            if enc.dred_offset < 6 {
                enc.dred_offset += 8;
            } else {
                enc.latent_offset += 1;
            }
        }

        pcm_off += process_size as usize;
        frame_size16k -= process_size16k;
    }
}

/// Largest of `DRED_LATENT_DIM` / `DRED_STATE_DIM` (`IMAX(...)`).
const MAX_LATENT_OR_STATE_DIM: usize = if DRED_LATENT_DIM > DRED_STATE_DIM {
    DRED_LATENT_DIM
} else {
    DRED_STATE_DIM
};

/// Port of dnn/dred_encoder.c:dred_encode_latents (static): dead-zone quantizes `dim` values
/// of `x` and codes them with the Laplace `_p0` coder.
pub fn dred_encode_latents(
    enc: &mut EcEnc<'_>,
    x: &[f32],
    scale: &[u8],
    dzone: &[u8],
    r: &[u8],
    p0: &[u8],
    dim: usize,
) {
    let mut q = [0i32; MAX_LATENT_OR_STATE_DIM];
    let mut xq = [0f32; MAX_LATENT_OR_STATE_DIM];
    let mut delta = [0f32; MAX_LATENT_OR_STATE_DIM];
    let mut deadzone = [0f32; MAX_LATENT_OR_STATE_DIM];
    let eps = 0.1f32;
    let (x, scale, dzone, r, p0) = (
        &x[..dim],
        &scale[..dim],
        &dzone[..dim],
        &r[..dim],
        &p0[..dim],
    );
    // This is split into multiple loops (with temporary arrays) so that the compiler can
    // vectorize all of it, and so we can call the vector tanh().
    for i in 0..dim {
        delta[i] = f32::from(dzone[i]) * (1.0f32 / 256.0f32);
        xq[i] = x[i] * f32::from(scale[i]) * (1.0f32 / 256.0f32);
        deadzone[i] = xq[i] / (delta[i] + eps);
    }
    compute_activation_inplace(&mut deadzone, dim, ACTIVATION_TANH);
    for i in 0..dim {
        xq[i] -= delta[i] * deadzone[i];
        q[i] = math::floor(f64::from(0.5f32 + xq[i])) as i32;
    }
    for i in 0..dim {
        // Make the impossible actually impossible.
        if r[i] == 0 || p0[i] == 255 {
            q[i] = 0;
        } else {
            ec_laplace_encode_p0(enc, q[i], u16::from(p0[i]) << 7, u16::from(r[i]) << 7);
        }
    }
}

/// Port of dnn/dred_encoder.c:dred_voice_active (static): whether any of the 16 activity
/// entries (2.5 ms each) of the 40 ms chunk at latent `offset` is 1.
///
/// C reads `activity_mem[8*offset + i]` for `i < 16`, which for `offset == 51` runs 8 bytes past
/// the 416-byte `OpusEncoder.activity_mem` (into `nonfinal_frame` and `rangeFinal`). Entries
/// past the end of the slice are treated as inactive; a caller that needs bit-exactness in that
/// corner case passes those trailing bytes in the slice.
#[must_use]
pub fn dred_voice_active(activity_mem: &[u8], offset: usize) -> bool {
    (0..16).any(|i| activity_mem.get(8 * offset + i) == Some(&1))
}

/// Port of dnn/dred_encoder.c:dred_encode_silk_frame.
///
/// Codes the DRED payload into `buf` (at least `max_bytes` long; `max_bytes` is the C
/// `ec_enc_init` size) and returns the number of bytes used, or 0 when no DRED is sent.
/// `activity_mem` is the encoder's voice-activity history (newest first, 2.5 ms resolution, see
/// [`dred_voice_active`]). `dq` must be in `0..8` and `q0 <= qmax` (C asserts).
#[expect(clippy::too_many_arguments, reason = "mirrors C signature")]
pub fn dred_encode_silk_frame(
    enc: &mut DredEnc,
    buf: &mut [u8],
    max_chunks: i32,
    max_bytes: i32,
    q0: i32,
    dq: i32,
    qmax: i32,
    activity_mem: &[u8],
) -> i32 {
    let mut prev_active = false;
    let mut extra_dred_offset: i32 = 0;
    let mut dred_encoded: i32 = 0;
    let mut delayed_dred = false;

    let mut latent_offset = enc.latent_offset;
    // Delaying new DRED data when just out of silence because we already have the main Opus
    // payload for that frame.
    if activity_mem[0] != 0 && enc.last_extra_dred_offset > 0 {
        latent_offset = enc.last_extra_dred_offset;
        delayed_dred = true;
        enc.last_extra_dred_offset = 0;
    }
    while latent_offset < enc.latents_buffer_fill
        && !dred_voice_active(activity_mem, latent_offset as usize)
    {
        latent_offset += 1;
        extra_dred_offset += 1;
    }
    if !delayed_dred {
        enc.last_extra_dred_offset = extra_dred_offset;
    }

    // entropy coding of state and latents
    let max_bytes_u = max_bytes as usize;
    let mut ec_encoder = EcEnc::new(&mut buf[..max_bytes_u]);
    ec_encoder.enc_uint(q0 as u32, 16);
    ec_encoder.enc_uint(dq as u32, 8);
    let total_offset = 16 - (enc.dred_offset - extra_dred_offset * 8);
    celt_assert!(total_offset >= 0);
    if total_offset > 31 {
        ec_encoder.enc_uint(1, 2);
        ec_encoder.enc_uint((total_offset >> 5) as u32, 256);
        ec_encoder.enc_uint((total_offset & 31) as u32, 32);
    } else {
        ec_encoder.enc_uint(0, 2);
        ec_encoder.enc_uint(total_offset as u32, 32);
    }
    celt_assert!(qmax >= q0);
    if q0 < 14 && dq > 0 {
        // If you want to use qmax == q0, you should have set dQ = 0.
        celt_assert!(qmax > q0);
        let nvals = 15 - (q0 + 1);
        let (fl, fh) = if qmax >= 15 {
            (0, nvals)
        } else {
            (nvals + qmax - (q0 + 1), nvals + qmax - q0)
        };
        ec_encoder.encode(fl as u32, fh as u32, (2 * nvals) as u32);
    }
    let state_qoffset = q0 as usize * DRED_STATE_DIM;
    let lo = latent_offset as usize;
    dred_encode_latents(
        &mut ec_encoder,
        &enc.state_buffer[lo * DRED_STATE_DIM..],
        &DRED_STATE_QUANT_SCALES_Q8[state_qoffset..],
        &DRED_STATE_DEAD_ZONE_Q8[state_qoffset..],
        &DRED_STATE_R_Q8[state_qoffset..],
        &DRED_STATE_P0_Q8[state_qoffset..],
        DRED_STATE_DIM,
    );
    if ec_encoder.tell() > 8 * max_bytes {
        return 0;
    }
    let mut ec_bak = ec_encoder.snapshot();
    let limit = (2 * max_chunks).min(enc.latents_buffer_fill - latent_offset - 1);
    let mut i: i32 = 0;
    while i < limit {
        let q_level = compute_quantizer(q0, dq, qmax, i / 2);
        let offset = q_level as usize * DRED_LATENT_DIM;

        dred_encode_latents(
            &mut ec_encoder,
            &enc.latents_buffer[(i + latent_offset) as usize * DRED_LATENT_DIM..],
            &DRED_LATENT_QUANT_SCALES_Q8[offset..],
            &DRED_LATENT_DEAD_ZONE_Q8[offset..],
            &DRED_LATENT_R_Q8[offset..],
            &DRED_LATENT_P0_Q8[offset..],
            DRED_LATENT_DIM,
        );
        if ec_encoder.tell() > 8 * max_bytes {
            // If we haven't been able to code one chunk, give up on DRED completely.
            if i == 0 {
                return 0;
            }
            break;
        }
        let active = dred_voice_active(activity_mem, (i + latent_offset) as usize);
        if active || prev_active {
            ec_bak = ec_encoder.snapshot();
            dred_encoded = i + 2;
        }
        prev_active = active;
        i += 2;
    }
    // Avoid sending empty DRED packets.
    if dred_encoded == 0 || (dred_encoded <= 2 && extra_dred_offset != 0) {
        return 0;
    }
    ec_encoder.restore(&ec_bak);

    let ec_buffer_fill = (ec_encoder.tell() + 7) / 8;
    ec_encoder.shrink(ec_buffer_fill as u32);
    ec_encoder.done();
    ec_buffer_fill
}
