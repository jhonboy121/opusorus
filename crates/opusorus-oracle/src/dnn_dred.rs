//! Oracle bindings for unit `dnn_dred`: dnn/dred_rdovae_enc.c, dred_rdovae_dec.c,
//! dred_rdovae_stats_data.c, dred_coding.c, dred_encoder.c (incl. its statics) and
//! dred_decoder.c.
//!
//! Only available when the oracle is built with the `dred` feature. Models are the C
//! compiled-in tables; states live on the C heap behind RAII handles and every wrapper checks
//! the sizes C will read or write before calling it.
#![cfg(feature = "dred")]

use core::ffi::{c_int, c_uint, c_void};

use crate::dnn_core::LPCNET_ENC_STATE_LEN;

/// `DRED_LATENT_DIM`.
pub const LATENT_DIM: usize = 25;
/// `DRED_STATE_DIM`.
pub const STATE_DIM: usize = 50;
/// `DRED_NUM_FEATURES`.
pub const NUM_FEATURES: usize = 20;
/// `DRED_MAX_FRAMES`.
pub const MAX_FRAMES: usize = 104;
/// `2*DRED_DFRAME_SIZE`.
pub const INPUT_BUFFER_SIZE: usize = 640;
/// `RESAMPLING_ORDER + 1`.
pub const RESAMPLE_MEM_SIZE: usize = 9;
/// Floats of the RDOVAE encoder state (gru1..5, conv1, conv2..5 with dilation 2).
pub const RDOVAE_ENC_STATE_LEN: usize = 5 * 32 + 64 + 4 * 128;
/// Floats of the RDOVAE decoder state (gru1..5, conv1..5).
pub const RDOVAE_DEC_STATE_LEN: usize = 5 * 64 + 5 * 32;
/// Floats of [`DredEnc::floats`].
pub const DRED_ENC_FLOATS_LEN: usize = INPUT_BUFFER_SIZE
    + MAX_FRAMES * LATENT_DIM
    + MAX_FRAMES * STATE_DIM
    + RESAMPLE_MEM_SIZE
    + RDOVAE_ENC_STATE_LEN
    + LPCNET_ENC_STATE_LEN;
/// `(DRED_NUM_REDUNDANCY_FRAMES/2)*(DRED_LATENT_DIM+1)`.
pub const LATENTS_SIZE: usize = 26 * (LATENT_DIM + 1);
/// Number of values of [`constants`].
pub const NUM_CONSTANTS: usize = 28;

unsafe extern "C" {
    fn oracle_dd_stats(which: c_int, out: *mut u8) -> c_int;
    fn oracle_dd_compute_quantizer(q0: c_int, dq: c_int, qmax: c_int, i: c_int) -> c_int;
    fn oracle_dd_constants(out: *mut c_int);

    fn oracle_dd_rdovae_enc_new() -> *mut c_void;
    fn oracle_dd_rdovae_enc_free(p: *mut c_void);
    fn oracle_dd_rdovae_encode_dframe(
        p: *mut c_void,
        latents: *mut f32,
        initial_state: *mut f32,
        input: *const f32,
    );
    fn oracle_dd_rdovae_enc_state(p: *mut c_void, out: *mut f32, len: *mut c_int) -> c_int;

    fn oracle_dd_rdovae_dec_new() -> *mut c_void;
    fn oracle_dd_rdovae_dec_free(p: *mut c_void);
    fn oracle_dd_rdovae_dec_init_states(p: *mut c_void, initial_state: *const f32);
    fn oracle_dd_rdovae_decode_qframe(p: *mut c_void, qframe: *mut f32, z: *const f32);
    fn oracle_dd_rdovae_decode_all(
        p: *mut c_void,
        features: *mut f32,
        state: *const f32,
        latents: *const f32,
        nb_latents: c_int,
    );
    fn oracle_dd_rdovae_dec_state(p: *mut c_void, out: *mut f32, len: *mut c_int) -> c_int;

    fn oracle_dd_enc_new(fs: c_int, channels: c_int) -> *mut c_void;
    fn oracle_dd_enc_free(p: *mut c_void);
    fn oracle_dd_enc_reset(p: *mut c_void);
    fn oracle_dd_enc_loaded(p: *mut c_void) -> c_int;
    fn oracle_dd_enc_load_model(p: *mut c_void, data: *const u8, len: c_int) -> c_int;
    fn oracle_dd_compute_latents(
        p: *mut c_void,
        pcm: *const f32,
        frame_size: c_int,
        extra_delay: c_int,
    );
    fn oracle_dd_encode_silk_frame(
        p: *mut c_void,
        buf: *mut u8,
        max_chunks: c_int,
        max_bytes: c_int,
        q0: c_int,
        dq: c_int,
        qmax: c_int,
        activity_mem: *mut u8,
    ) -> c_int;
    fn oracle_dd_convert_to_16k(
        p: *mut c_void,
        input: *const f32,
        in_len: c_int,
        out: *mut f32,
        out_len: c_int,
    );
    fn oracle_dd_process_frame(p: *mut c_void);
    fn oracle_dd_enc_ints(p: *mut c_void, out: *mut c_int);
    fn oracle_dd_enc_floats(p: *mut c_void, out: *mut f32) -> c_int;
    fn oracle_dd_enc_set(
        p: *mut c_void,
        latents: *const f32,
        state: *const f32,
        latents_fill: c_int,
        dred_offset: c_int,
        latent_offset: c_int,
        last_extra_dred_offset: c_int,
    );
    fn oracle_dd_enc_set_input(p: *mut c_void, input: *const f32, fill: c_int);

    fn oracle_dd_filter_df2t(
        input: *const f32,
        out: *mut f32,
        len: c_int,
        b0: f32,
        b: *const f32,
        a: *const f32,
        order: c_int,
        mem: *mut f32,
        inplace: c_int,
    );
    fn oracle_dd_voice_active(activity_mem: *const u8, offset: c_int) -> c_int;
    fn oracle_dd_encode_latents(
        buf: *mut u8,
        max_bytes: c_int,
        x: *const f32,
        scale: *const u8,
        dzone: *const u8,
        r: *const u8,
        p0: *const u8,
        dim: c_int,
        rng: *mut c_uint,
        err: *mut c_int,
    ) -> c_int;
    fn oracle_dd_decode_latents(
        bytes: *const u8,
        len: c_int,
        x: *mut f32,
        scale: *const u8,
        r: *const u8,
        p0: *const u8,
        dim: c_int,
        rng: *mut c_uint,
    ) -> c_int;
    fn oracle_dd_ec_decode(
        bytes: *const u8,
        num_bytes: c_int,
        min_feature_frames: c_int,
        dred_frame_offset: c_int,
        state: *mut f32,
        latents: *mut f32,
        ints: *mut c_int,
    ) -> c_int;
}

fn len_i32(n: usize) -> c_int {
    c_int::try_from(n).expect("length fits in int")
}

/// One of the 8 quantization statistics tables (0..4: latent scales/dead zone/r/p0, 4..8: state
/// scales/dead zone/r/p0).
#[must_use]
pub fn stats(which: i32) -> Vec<u8> {
    assert!((0..8).contains(&which));
    // SAFETY: NULL only returns the size.
    let n = unsafe { oracle_dd_stats(which, core::ptr::null_mut()) };
    let mut v = vec![0u8; usize::try_from(n).expect("valid table")];
    // SAFETY: v has the size C writes.
    unsafe { oracle_dd_stats(which, v.as_mut_ptr()) };
    v
}

/// C `compute_quantizer` (`dq` in 0..8).
#[must_use]
pub fn compute_quantizer(q0: i32, dq: i32, qmax: i32, i: i32) -> i32 {
    assert!((0..8).contains(&dq));
    // SAFETY: pure function on ints; dq indexes an 8-entry table.
    unsafe { oracle_dd_compute_quantizer(q0, dq, qmax, i) }
}

/// The DRED constants in the order of the C `oracle_dd_constants`.
#[must_use]
pub fn constants() -> Vec<i32> {
    let mut v = vec![0; NUM_CONSTANTS];
    // SAFETY: v holds NUM_CONSTANTS ints.
    unsafe { oracle_dd_constants(v.as_mut_ptr()) };
    v
}

/// C `RDOVAEEnc` (compiled-in model) + `RDOVAEEncState` (zeroed).
#[derive(Debug)]
pub struct RdovaeEnc(*mut c_void);

impl RdovaeEnc {
    #[must_use]
    #[expect(clippy::new_without_default, reason = "allocates C state")]
    pub fn new() -> Self {
        // SAFETY: allocates and binds the compiled-in model.
        let p = unsafe { oracle_dd_rdovae_enc_new() };
        assert!(!p.is_null(), "init_rdovaeenc failed");
        Self(p)
    }

    /// C `dred_rdovae_encode_dframe`: 40 input features → (25 latents, 50 state).
    #[must_use]
    pub fn encode_dframe(&mut self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert!(input.len() >= 2 * NUM_FEATURES);
        let mut lat = vec![0f32; LATENT_DIM];
        let mut st = vec![0f32; STATE_DIM];
        // SAFETY: sizes checked/allocated above.
        unsafe {
            oracle_dd_rdovae_encode_dframe(
                self.0,
                lat.as_mut_ptr(),
                st.as_mut_ptr(),
                input.as_ptr(),
            );
        };
        (lat, st)
    }

    /// (initialized, flattened float state).
    #[must_use]
    pub fn state(&self) -> (i32, Vec<f32>) {
        let mut v = vec![0f32; RDOVAE_ENC_STATE_LEN];
        let mut n = 0;
        // SAFETY: v has the size C writes (asserted below).
        let init = unsafe { oracle_dd_rdovae_enc_state(self.0, v.as_mut_ptr(), &mut n) };
        assert_eq!(n as usize, RDOVAE_ENC_STATE_LEN);
        (init, v)
    }
}

impl Drop for RdovaeEnc {
    fn drop(&mut self) {
        // SAFETY: allocated by oracle_dd_rdovae_enc_new.
        unsafe { oracle_dd_rdovae_enc_free(self.0) }
    }
}

/// C `RDOVAEDec` (compiled-in model) + `RDOVAEDecState` (zeroed).
#[derive(Debug)]
pub struct RdovaeDec(*mut c_void);

impl RdovaeDec {
    #[must_use]
    #[expect(clippy::new_without_default, reason = "allocates C state")]
    pub fn new() -> Self {
        // SAFETY: allocates and binds the compiled-in model.
        let p = unsafe { oracle_dd_rdovae_dec_new() };
        assert!(!p.is_null(), "init_rdovaedec failed");
        Self(p)
    }

    /// C `dred_rdovae_dec_init_states`.
    pub fn init_states(&mut self, initial_state: &[f32]) {
        assert!(initial_state.len() >= STATE_DIM);
        // SAFETY: reads STATE_DIM floats.
        unsafe { oracle_dd_rdovae_dec_init_states(self.0, initial_state.as_ptr()) }
    }

    /// C `dred_rdovae_decode_qframe`: 26 floats → 80 features.
    #[must_use]
    pub fn decode_qframe(&mut self, z: &[f32]) -> Vec<f32> {
        assert!(z.len() > LATENT_DIM);
        let mut q = vec![0f32; 4 * NUM_FEATURES];
        // SAFETY: reads 26 floats, writes 80.
        unsafe { oracle_dd_rdovae_decode_qframe(self.0, q.as_mut_ptr(), z.as_ptr()) };
        q
    }

    /// C `DRED_rdovae_decode_all` (own zeroed state; the handle state is unchanged).
    #[must_use]
    pub fn decode_all(&self, state: &[f32], latents: &[f32], nb_latents: usize) -> Vec<f32> {
        assert!(state.len() >= STATE_DIM);
        assert!(latents.len() >= nb_latents * (LATENT_DIM + 1));
        let mut f = vec![0f32; 4 * NUM_FEATURES * nb_latents];
        // SAFETY: sizes checked above.
        unsafe {
            oracle_dd_rdovae_decode_all(
                self.0,
                f.as_mut_ptr(),
                state.as_ptr(),
                latents.as_ptr(),
                len_i32(nb_latents),
            );
        };
        f
    }

    /// (initialized, flattened float state).
    #[must_use]
    pub fn state(&self) -> (i32, Vec<f32>) {
        let mut v = vec![0f32; RDOVAE_DEC_STATE_LEN];
        let mut n = 0;
        // SAFETY: v has the size C writes (asserted below).
        let init = unsafe { oracle_dd_rdovae_dec_state(self.0, v.as_mut_ptr(), &mut n) };
        assert_eq!(n as usize, RDOVAE_DEC_STATE_LEN);
        (init, v)
    }
}

impl Drop for RdovaeDec {
    fn drop(&mut self) {
        // SAFETY: allocated by oracle_dd_rdovae_dec_new.
        unsafe { oracle_dd_rdovae_dec_free(self.0) }
    }
}

/// Integer fields of [`DredEnc::ints`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DredEncInts {
    pub input_buffer_fill: i32,
    pub dred_offset: i32,
    pub latent_offset: i32,
    pub last_extra_dred_offset: i32,
    pub latents_buffer_fill: i32,
    pub loaded: i32,
    pub fs: i32,
    pub channels: i32,
    pub rdovae_initialized: i32,
}

/// C `DREDEnc` from `dred_encoder_init` (compiled-in model, `loaded == 1`).
#[derive(Debug)]
pub struct DredEnc {
    p: *mut c_void,
    fs: i32,
    channels: i32,
}

impl DredEnc {
    #[must_use]
    pub fn new(fs: i32, channels: i32) -> Self {
        assert!(channels == 1 || channels == 2);
        // SAFETY: allocates and initialises a C DREDEnc.
        let p = unsafe { oracle_dd_enc_new(fs, channels) };
        Self { p, fs, channels }
    }

    /// C `dred_encoder_reset`.
    pub fn reset(&mut self) {
        // SAFETY: valid handle.
        unsafe { oracle_dd_enc_reset(self.p) }
    }

    #[must_use]
    pub fn loaded(&self) -> bool {
        // SAFETY: valid handle.
        unsafe { oracle_dd_enc_loaded(self.p) != 0 }
    }

    /// C `dred_encoder_load_model` (returns the OPUS_* code). C crashes on an unparsable blob,
    /// so only pass well-formed blobs.
    pub fn load_model(&mut self, data: &[u8]) -> i32 {
        // C reads the headers as ints: give it an aligned copy.
        let mut aligned = vec![0u32; data.len().div_ceil(4)];
        let bytes: Vec<u8> = data.to_vec();
        for (i, chunk) in bytes.chunks(4).enumerate() {
            let mut w = [0u8; 4];
            w[..chunk.len()].copy_from_slice(chunk);
            aligned[i] = u32::from_ne_bytes(w);
        }
        // SAFETY: `aligned` holds data.len() bytes.
        unsafe { oracle_dd_enc_load_model(self.p, aligned.as_ptr().cast(), len_i32(data.len())) }
    }

    /// C `dred_compute_latents`. `pcm` must hold what C reads (see the Rust port's quirk note:
    /// `frame_size*channels` samples, or more for > 20 ms stereo frames).
    pub fn compute_latents(&mut self, pcm: &[f32], frame_size: i32, extra_delay: i32) {
        // Exactly what C reads: each chunk reads process_size*channels samples at an offset
        // that advances by process_size.
        let mut fs16k = frame_size * 16000 / self.fs;
        let (mut off, mut need) = (0i32, 0i32);
        while fs16k > 0 {
            let p16 = fs16k.min(320);
            let p = p16 * self.fs / 16000;
            need = need.max(off + p * self.channels);
            off += p;
            fs16k -= p16;
        }
        assert!(pcm.len() >= need as usize);
        // SAFETY: size checked above.
        unsafe { oracle_dd_compute_latents(self.p, pcm.as_ptr(), frame_size, extra_delay) }
    }

    /// C `dred_encode_silk_frame` into a zeroed buffer of `buf_len >= max_bytes` bytes;
    /// returns (ret, whole buffer). `activity_mem` must hold at least 424 bytes (C may read
    /// 8 bytes past the 416-byte array).
    #[expect(clippy::too_many_arguments, reason = "mirrors C signature")]
    pub fn encode_silk_frame(
        &mut self,
        buf_len: usize,
        max_chunks: i32,
        max_bytes: i32,
        q0: i32,
        dq: i32,
        qmax: i32,
        activity_mem: &[u8],
    ) -> (i32, Vec<u8>) {
        assert!(activity_mem.len() >= 424);
        assert!(max_bytes >= 0 && max_bytes as usize <= buf_len);
        assert!((0..16).contains(&q0) && (0..8).contains(&dq));
        let mut act = activity_mem.to_vec();
        let mut buf = vec![0u8; buf_len];
        // SAFETY: buffer sizes checked above.
        let ret = unsafe {
            oracle_dd_encode_silk_frame(
                self.p,
                buf.as_mut_ptr(),
                max_chunks,
                max_bytes,
                q0,
                dq,
                qmax,
                act.as_mut_ptr(),
            )
        };
        (ret, buf)
    }

    /// Static `dred_convert_to_16k` (updates `resample_mem`).
    #[must_use]
    pub fn convert_to_16k(&mut self, input: &[f32], in_len: usize, out_len: usize) -> Vec<f32> {
        assert!(input.len() >= in_len * self.channels as usize);
        assert!(
            in_len * self.channels as usize <= 1920 * if cfg!(feature = "qext") { 2 } else { 1 }
        );
        let mut out = vec![0f32; out_len];
        // SAFETY: sizes checked above.
        unsafe {
            oracle_dd_convert_to_16k(
                self.p,
                input.as_ptr(),
                len_i32(in_len),
                out.as_mut_ptr(),
                len_i32(out_len),
            );
        };
        out
    }

    /// Static `dred_process_frame` on the current input buffer.
    pub fn process_frame(&mut self) {
        // SAFETY: valid handle.
        unsafe { oracle_dd_process_frame(self.p) }
    }

    #[must_use]
    pub fn ints(&self) -> DredEncInts {
        let mut v = [0; 9];
        // SAFETY: v holds the 9 ints C writes.
        unsafe { oracle_dd_enc_ints(self.p, v.as_mut_ptr()) };
        DredEncInts {
            input_buffer_fill: v[0],
            dred_offset: v[1],
            latent_offset: v[2],
            last_extra_dred_offset: v[3],
            latents_buffer_fill: v[4],
            loaded: v[5],
            fs: v[6],
            channels: v[7],
            rdovae_initialized: v[8],
        }
    }

    /// Flattened float state (see [`DRED_ENC_FLOATS_LEN`]).
    #[must_use]
    pub fn floats(&self) -> Vec<f32> {
        let mut v = vec![0f32; DRED_ENC_FLOATS_LEN];
        // SAFETY: v has the size C writes (asserted below).
        let n = unsafe { oracle_dd_enc_floats(self.p, v.as_mut_ptr()) };
        assert_eq!(n as usize, DRED_ENC_FLOATS_LEN);
        v
    }

    /// Overwrites latents/state buffers and offsets.
    pub fn set(
        &mut self,
        latents: &[f32],
        state: &[f32],
        latents_fill: i32,
        dred_offset: i32,
        latent_offset: i32,
        last_extra_dred_offset: i32,
    ) {
        assert_eq!(latents.len(), MAX_FRAMES * LATENT_DIM);
        assert_eq!(state.len(), MAX_FRAMES * STATE_DIM);
        // SAFETY: sizes checked above.
        unsafe {
            oracle_dd_enc_set(
                self.p,
                latents.as_ptr(),
                state.as_ptr(),
                latents_fill,
                dred_offset,
                latent_offset,
                last_extra_dred_offset,
            );
        };
    }

    /// Overwrites the 16 kHz input buffer and its fill.
    pub fn set_input(&mut self, input: &[f32], fill: i32) {
        assert_eq!(input.len(), INPUT_BUFFER_SIZE);
        // SAFETY: size checked above.
        unsafe { oracle_dd_enc_set_input(self.p, input.as_ptr(), fill) }
    }
}

impl Drop for DredEnc {
    fn drop(&mut self) {
        // SAFETY: allocated by oracle_dd_enc_new.
        unsafe { oracle_dd_enc_free(self.p) }
    }
}

/// C `filter_df2t` (`inplace`: out == in). `b`/`a` hold `order` taps, `mem` `order+1`.
#[expect(clippy::too_many_arguments, reason = "mirrors C signature")]
#[must_use]
pub fn filter_df2t(
    input: &[f32],
    b0: f32,
    b: &[f32],
    a: &[f32],
    order: usize,
    mem: &mut [f32],
    inplace: bool,
    len: usize,
) -> Vec<f32> {
    assert!(input.len() >= len && b.len() >= order && a.len() >= order && mem.len() > order);
    let mut out = vec![0f32; len];
    // SAFETY: sizes checked above.
    unsafe {
        oracle_dd_filter_df2t(
            input.as_ptr(),
            out.as_mut_ptr(),
            len_i32(len),
            b0,
            b.as_ptr(),
            a.as_ptr(),
            len_i32(order),
            mem.as_mut_ptr(),
            c_int::from(inplace),
        );
    };
    out
}

/// Static `dred_voice_active`.
#[must_use]
pub fn voice_active(activity_mem: &[u8], offset: usize) -> bool {
    assert!(activity_mem.len() >= 8 * offset + 16);
    // SAFETY: reads 16 bytes at 8*offset (checked above).
    unsafe { oracle_dd_voice_active(activity_mem.as_ptr(), len_i32(offset)) != 0 }
}

/// Result of [`encode_latents`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedLatents {
    pub tell: i32,
    pub rng: u32,
    pub error: i32,
    pub buf: Vec<u8>,
}

/// Static `dred_encode_latents` on a fresh `ec_enc` of `max_bytes`, then `ec_enc_done`.
#[must_use]
pub fn encode_latents(
    max_bytes: usize,
    x: &[f32],
    scale: &[u8],
    dzone: &[u8],
    r: &[u8],
    p0: &[u8],
    dim: usize,
) -> EncodedLatents {
    assert!(dim <= STATE_DIM);
    assert!(x.len() >= dim && scale.len() >= dim && dzone.len() >= dim);
    assert!(r.len() >= dim && p0.len() >= dim);
    let mut buf = vec![0u8; max_bytes];
    let mut rng = 0;
    let mut err = 0;
    // SAFETY: sizes checked above.
    let tell = unsafe {
        oracle_dd_encode_latents(
            buf.as_mut_ptr(),
            len_i32(max_bytes),
            x.as_ptr(),
            scale.as_ptr(),
            dzone.as_ptr(),
            r.as_ptr(),
            p0.as_ptr(),
            len_i32(dim),
            &mut rng,
            &mut err,
        )
    };
    EncodedLatents {
        tell,
        rng,
        error: err,
        buf,
    }
}

/// C `dred_decode_latents` on a fresh decoder: (values, tell, rng).
#[must_use]
pub fn decode_latents(
    bytes: &[u8],
    scale: &[u8],
    r: &[u8],
    p0: &[u8],
    dim: usize,
) -> (Vec<f32>, i32, u32) {
    assert!(scale.len() >= dim && r.len() >= dim && p0.len() >= dim);
    let mut x = vec![0f32; dim];
    let mut rng = 0;
    // SAFETY: sizes checked above.
    let tell = unsafe {
        oracle_dd_decode_latents(
            bytes.as_ptr(),
            len_i32(bytes.len()),
            x.as_mut_ptr(),
            scale.as_ptr(),
            r.as_ptr(),
            p0.as_ptr(),
            len_i32(dim),
            &mut rng,
        )
    };
    (x, tell, rng)
}

/// Result of [`ec_decode`] (an `OpusDRED` after `dred_ec_decode`).
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedDred {
    pub ret: i32,
    pub state: Vec<f32>,
    pub latents: Vec<f32>,
    pub nb_latents: i32,
    pub process_stage: i32,
    pub dred_offset: i32,
}

/// C `dred_ec_decode` on a zeroed `OpusDRED`.
#[must_use]
pub fn ec_decode(bytes: &[u8], min_feature_frames: i32, dred_frame_offset: i32) -> DecodedDred {
    let mut state = vec![0f32; STATE_DIM];
    let mut latents = vec![0f32; LATENTS_SIZE];
    let mut ints = [0; 3];
    // SAFETY: output sizes match the OpusDRED fields.
    let ret = unsafe {
        oracle_dd_ec_decode(
            bytes.as_ptr(),
            len_i32(bytes.len()),
            min_feature_frames,
            dred_frame_offset,
            state.as_mut_ptr(),
            latents.as_mut_ptr(),
            ints.as_mut_ptr(),
        )
    };
    DecodedDred {
        ret,
        state,
        latents,
        nb_latents: ints[0],
        process_stage: ints[1],
        dred_offset: ints[2],
    }
}
