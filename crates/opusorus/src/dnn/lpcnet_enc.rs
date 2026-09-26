//! Port of `dnn/lpcnet_enc.c` (LPCNet feature extraction used by deep PLC and DRED) and the
//! `LPCNetEncState` / constants of `dnn/lpcnet_private.h` / `dnn/lpcnet.h`.
//!
//! `lpcnet_encoder_create` / `lpcnet_encoder_destroy` / `lpcnet_encoder_get_size` are C memory
//! management and map to [`LpcnetEncState::new`] and `Drop`.

use alloc::boxed::Box;

use crate::Result;
use crate::celt::arch::{max16, min16};
use crate::celt::celt_lpc::celt_fir;
use crate::celt::mathops::celt_log2;
use crate::celt::pitch::{celt_inner_prod, celt_pitch_xcorr};
use crate::celt::static_modes::KissFftCpx;
use crate::math;

use super::freq::{
    FRAME_SIZE, FREQ_SIZE, LPC_ORDER, NB_BANDS, OVERLAP_SIZE, PREEMPHASIS, TRAINING_OFFSET,
    WINDOW_SIZE, apply_window, dct, forward_transform, lpc_from_cepstrum, lpcn_compute_band_energy,
};
use super::pitchdnn::{PITCH_MAX_PERIOD, PITCH_MIN_PERIOD, PitchDnnState, compute_pitchdnn};

/// `NB_FEATURES` (lpcnet.h).
pub const NB_FEATURES: usize = 20;
/// `NB_TOTAL_FEATURES` (lpcnet.h).
pub const NB_TOTAL_FEATURES: usize = 36;
/// `LPCNET_FRAME_SIZE` (lpcnet.h).
pub const LPCNET_FRAME_SIZE: usize = 160;

/// `PITCH_FRAME_SIZE` (lpcnet_private.h).
pub const PITCH_FRAME_SIZE: usize = 320;
/// `PITCH_BUF_SIZE`.
pub const PITCH_BUF_SIZE: usize = PITCH_MAX_PERIOD + PITCH_FRAME_SIZE;
/// `PLC_MAX_FEC`.
pub const PLC_MAX_FEC: usize = 104;
/// `MAX_FEATURE_BUFFER_SIZE`.
pub const MAX_FEATURE_BUFFER_SIZE: usize = 4;
/// `PITCH_IF_MAX_FREQ`.
pub const PITCH_IF_MAX_FREQ: usize = 30;
/// `PITCH_IF_FEATURES`.
pub const PITCH_IF_FEATURES: usize = 3 * PITCH_IF_MAX_FREQ - 2;
/// `CONT_VECTORS`.
pub const CONT_VECTORS: usize = 5;
/// `FEATURES_DELAY`.
pub const FEATURES_DELAY: usize = 1;

/// `struct LPCNetEncState` (lpcnet_private.h).
#[derive(Debug, Clone, PartialEq)]
pub struct LpcnetEncState {
    pub pitchdnn: PitchDnnState,
    pub analysis_mem: [f32; OVERLAP_SIZE],
    pub mem_preemph: f32,
    pub prev_if: [KissFftCpx; PITCH_IF_MAX_FREQ],
    pub if_features: [f32; PITCH_IF_FEATURES],
    pub xcorr_features: [f32; PITCH_MAX_PERIOD - PITCH_MIN_PERIOD],
    pub dnn_pitch: f32,
    pub pitch_mem: [f32; LPC_ORDER],
    pub pitch_filt: f32,
    pub exc_buf: [f32; PITCH_BUF_SIZE],
    pub lp_buf: [f32; PITCH_BUF_SIZE],
    pub lp_mem: [f32; 4],
    pub lpc: [f32; LPC_ORDER],
    pub features: [f32; NB_TOTAL_FEATURES],
    pub sig_mem: [f32; LPC_ORDER],
    pub burg_cepstrum: [f32; 2 * NB_BANDS],
}

impl Default for LpcnetEncState {
    fn default() -> Self {
        Self::new_inline()
    }
}

impl LpcnetEncState {
    fn new_inline() -> Self {
        Self {
            pitchdnn: PitchDnnState::new(),
            analysis_mem: [0.0; OVERLAP_SIZE],
            mem_preemph: 0.0,
            prev_if: [KissFftCpx::default(); PITCH_IF_MAX_FREQ],
            if_features: [0.0; PITCH_IF_FEATURES],
            xcorr_features: [0.0; PITCH_MAX_PERIOD - PITCH_MIN_PERIOD],
            dnn_pitch: 0.0,
            pitch_mem: [0.0; LPC_ORDER],
            pitch_filt: 0.0,
            exc_buf: [0.0; PITCH_BUF_SIZE],
            lp_buf: [0.0; PITCH_BUF_SIZE],
            lp_mem: [0.0; 4],
            lpc: [0.0; LPC_ORDER],
            features: [0.0; NB_TOTAL_FEATURES],
            sig_mem: [0.0; LPC_ORDER],
            burg_cepstrum: [0.0; 2 * NB_BANDS],
        }
    }

    /// Port of dnn/lpcnet_enc.c:lpcnet_encoder_create (+ `lpcnet_encoder_init`): a cleared,
    /// heap-allocated state without a model (see [`Self::load_model`]).
    #[must_use]
    pub fn new() -> Box<Self> {
        Box::new(Self::new_inline())
    }

    /// Port of dnn/lpcnet_enc.c:lpcnet_encoder_init: clears the state (the loaded pitch model is
    /// kept; upstream re-binds the compiled-in model here).
    pub fn lpcnet_encoder_init(&mut self) {
        let model = core::mem::take(&mut self.pitchdnn.model);
        *self = Self::new_inline();
        self.pitchdnn.model = model;
    }

    /// Port of dnn/lpcnet_enc.c:lpcnet_encoder_load_model.
    pub fn load_model(&mut self, data: &[u8]) -> Result<()> {
        self.pitchdnn.load_model(data)
    }
}

/// Port of dnn/lpcnet_enc.c:frame_analysis (static).
pub fn frame_analysis(
    st: &mut LpcnetEncState,
    x_out: &mut [KissFftCpx],
    ex: &mut [f32],
    input: &[f32],
) {
    let mut x = [0f32; WINDOW_SIZE];
    x[..OVERLAP_SIZE].copy_from_slice(&st.analysis_mem);
    x[OVERLAP_SIZE..].copy_from_slice(&input[..FRAME_SIZE]);
    st.analysis_mem
        .copy_from_slice(&input[FRAME_SIZE - OVERLAP_SIZE..FRAME_SIZE]);
    apply_window(&mut x);
    forward_transform(x_out, &x);
    lpcn_compute_band_energy(ex, x_out);
}

/// Port of dnn/lpcnet_enc.c:biquad (static) for `y != x`.
pub fn biquad(y: &mut [f32], mem: &mut [f32], x: &[f32], b: &[f32; 2], a: &[f32; 2], n: usize) {
    let mut mem0 = mem[0];
    let mut mem1 = mem[1];
    for (yi, &xi) in y[..n].iter_mut().zip(&x[..n]) {
        let yi_ = xi + mem0;
        let mem00 = mem0;
        // Original: mem0 = mem1 + (b[0]*xi - a[0]*yi); mem1 = (b[1]*xi - a[1]*yi).
        // Rewritten upstream to reduce dependency chains (the +1e-30f forces the ordering).
        mem0 = (b[0] - a[0]) * xi + mem1 - a[0] * mem0;
        mem1 = (b[1] - a[1]) * xi + 1e-30f32 - a[1] * mem00;
        *yi = yi_;
    }
    mem[0] = mem0;
    mem[1] = mem1;
}

/// Port of dnn/lpcnet_enc.c:biquad (static) for the in-place call (`y == x`).
pub fn biquad_inplace(y: &mut [f32], mem: &mut [f32], b: &[f32; 2], a: &[f32; 2], n: usize) {
    let mut mem0 = mem[0];
    let mut mem1 = mem[1];
    for v in &mut y[..n] {
        let xi = *v;
        let yi = xi + mem0;
        let mem00 = mem0;
        mem0 = (b[0] - a[0]) * xi + mem1 - a[0] * mem0;
        mem1 = (b[1] - a[1]) * xi + 1e-30f32 - a[1] * mem00;
        *v = yi;
    }
    mem[0] = mem0;
    mem[1] = mem1;
}

/// `celt_log10(x)` (lpcnet_enc.c): `0.3010299957f*celt_log2(x)`.
#[inline(always)]
fn celt_log10(x: f32) -> f32 {
    0.3010299957f32 * celt_log2(x)
}

/// `[b,a]=ellip(2, 2, 20, 1200/8000)` (compute_frame_features).
const LP_B: [f32; 2] = [-0.84946, 1.0];
const LP_A: [f32; 2] = [-1.54220, 0.70781];

/// Port of dnn/lpcnet_enc.c:compute_frame_features on one `FRAME_SIZE` frame (already
/// pre-emphasized).
#[expect(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]
pub fn compute_frame_features(st: &mut LpcnetEncState, input: &[f32]) {
    let mut aligned_in = [0f32; FRAME_SIZE];
    let mut ly = [0f32; NB_BANDS];
    let mut x_f = [KissFftCpx::default(); FREQ_SIZE];
    let mut ex = [0f32; NB_BANDS];
    let mut xcorr = [0f32; PITCH_MAX_PERIOD];
    let mut x = [0f32; FRAME_SIZE + LPC_ORDER];
    let mut ener_norm = [0f32; PITCH_MAX_PERIOD - PITCH_MIN_PERIOD];
    aligned_in[..TRAINING_OFFSET]
        .copy_from_slice(&st.analysis_mem[OVERLAP_SIZE - TRAINING_OFFSET..]);
    frame_analysis(st, &mut x_f, &mut ex, input);
    st.if_features[0] = max16(
        -1.0,
        min16(
            1.0,
            (1.0f32 / 64.0) * (10.0f32 * celt_log10(1e-15f32 + x_f[0].r * x_f[0].r) - 6.0),
        ),
    );
    for i in 1..PITCH_IF_MAX_FREQ {
        let a = x_f[i];
        let b = st.prev_if[i];
        // C_MULC(prod, X[i], prev_if[i])
        let mut prod = KissFftCpx {
            r: a.r * b.r + a.i * b.i,
            i: a.i * b.r - a.r * b.i,
        };
        let norm_1 =
            (1.0f64 / math::sqrt((1e-15f32 + prod.r * prod.r + prod.i * prod.i) as f64)) as f32;
        // C_MULBYSCALAR(prod, norm_1)
        prod.r *= norm_1;
        prod.i *= norm_1;
        st.if_features[3 * i - 2] = prod.r;
        st.if_features[3 * i - 1] = prod.i;
        st.if_features[3 * i] = max16(
            -1.0,
            min16(
                1.0,
                (1.0f32 / 64.0) * (10.0f32 * celt_log10(1e-15f32 + a.r * a.r + a.i * a.i) - 6.0),
            ),
        );
    }
    st.prev_if.copy_from_slice(&x_f[..PITCH_IF_MAX_FREQ]);
    let mut log_max = -2f32;
    let mut follow = -2f32;
    for i in 0..NB_BANDS {
        ly[i] = celt_log10(1e-2f32 + ex[i]);
        ly[i] = max16(log_max - 8.0, max16(follow - 2.5f32, ly[i]));
        log_max = max16(log_max, ly[i]);
        follow = max16(follow - 2.5f32, ly[i]);
    }
    dct(&mut st.features, &ly);
    st.features[0] -= 4.0;
    lpc_from_cepstrum(&mut st.lpc, &st.features);
    for i in 0..LPC_ORDER {
        st.features[NB_BANDS + 2 + i] = st.lpc[i];
    }
    st.exc_buf
        .copy_within(FRAME_SIZE..FRAME_SIZE + PITCH_MAX_PERIOD, 0);
    st.lp_buf
        .copy_within(FRAME_SIZE..FRAME_SIZE + PITCH_MAX_PERIOD, 0);
    aligned_in[TRAINING_OFFSET..].copy_from_slice(&input[..FRAME_SIZE - TRAINING_OFFSET]);
    x[..LPC_ORDER].copy_from_slice(&st.pitch_mem);
    x[LPC_ORDER..].copy_from_slice(&aligned_in);
    st.pitch_mem
        .copy_from_slice(&aligned_in[FRAME_SIZE - LPC_ORDER..]);
    // C: celt_fir(&x[LPC_ORDER], lpc, &lp_buf[PITCH_MAX_PERIOD], ...) — the port's celt_fir
    // takes the `LPC_ORDER` history samples in front of the input.
    celt_fir(
        &x,
        &st.lpc,
        &mut st.lp_buf[PITCH_MAX_PERIOD..],
        FRAME_SIZE,
        LPC_ORDER,
    );
    for i in 0..FRAME_SIZE {
        st.exc_buf[PITCH_MAX_PERIOD + i] = st.lp_buf[PITCH_MAX_PERIOD + i] + 0.7f32 * st.pitch_filt;
        st.pitch_filt = st.lp_buf[PITCH_MAX_PERIOD + i];
    }
    biquad_inplace(
        &mut st.lp_buf[PITCH_MAX_PERIOD..],
        &mut st.lp_mem,
        &LP_B,
        &LP_A,
        FRAME_SIZE,
    );
    {
        let buf = &st.exc_buf;
        celt_pitch_xcorr(
            &buf[PITCH_MAX_PERIOD..],
            buf,
            &mut xcorr,
            FRAME_SIZE,
            PITCH_MAX_PERIOD - PITCH_MIN_PERIOD,
        );
        let ener0 = celt_inner_prod(
            &buf[PITCH_MAX_PERIOD..],
            &buf[PITCH_MAX_PERIOD..],
            FRAME_SIZE,
        );
        let mut ener1 = celt_inner_prod(buf, buf, FRAME_SIZE) as f64;
        for i in 0..PITCH_MAX_PERIOD - PITCH_MIN_PERIOD {
            // C: ener = 1 + ener0 + ener1 ((1 + ener0) in float, then + double).
            let ener = ((1.0f32 + ener0) as f64 + ener1) as f32;
            st.xcorr_features[i] = 2.0 * xcorr[i];
            ener_norm[i] = ener;
            ener1 += buf[i + FRAME_SIZE] as f64 * buf[i + FRAME_SIZE] as f64
                - buf[i] as f64 * buf[i] as f64;
        }
        // Split in a separate loop so the compiler can vectorize it.
        for (xf, &e) in st.xcorr_features.iter_mut().zip(&ener_norm) {
            *xf /= e;
        }
    }
    st.dnn_pitch = compute_pitchdnn(&mut st.pitchdnn, &st.if_features, &st.xcorr_features);
    let pitch = math::floor(
        0.5 + 256.0 / math::pow(2.0, (1.0 / 60.0) * ((st.dnn_pitch as f64 + 1.5) * 60.0)),
    ) as i32 as usize;
    let lp = &st.lp_buf;
    let xx = celt_inner_prod(&lp[PITCH_MAX_PERIOD..], &lp[PITCH_MAX_PERIOD..], FRAME_SIZE);
    let yy = celt_inner_prod(
        &lp[PITCH_MAX_PERIOD - pitch..],
        &lp[PITCH_MAX_PERIOD - pitch..],
        FRAME_SIZE,
    );
    let xy = celt_inner_prod(
        &lp[PITCH_MAX_PERIOD..],
        &lp[PITCH_MAX_PERIOD - pitch..],
        FRAME_SIZE,
    );
    let mut frame_corr = (xy as f64 / math::sqrt((1.0f32 + xx * yy) as f64)) as f32;
    frame_corr = (math::log(1.0f32 as f64 + math::exp((5.0f32 * frame_corr) as f64))
        / math::log(1.0 + math::exp(5.0f32 as f64))) as f32;
    st.features[NB_BANDS] = st.dnn_pitch;
    st.features[NB_BANDS + 1] = frame_corr - 0.5f32;
}

/// Port of dnn/lpcnet_enc.c:preemphasis for `y != x`.
pub fn preemphasis(y: &mut [f32], mem: &mut f32, x: &[f32], coef: f32, n: usize) {
    for (yi, &xi) in y[..n].iter_mut().zip(&x[..n]) {
        let v = xi + *mem;
        *mem = -coef * xi;
        *yi = v;
    }
}

/// Port of dnn/lpcnet_enc.c:preemphasis for the in-place call (`y == x`).
pub fn preemphasis_inplace(x: &mut [f32], mem: &mut f32, coef: f32, n: usize) {
    for v in &mut x[..n] {
        let xi = *v;
        *v = xi + *mem;
        *mem = -coef * xi;
    }
}

/// Port of dnn/lpcnet_enc.c:lpcnet_compute_single_frame_features_impl (static). `x` is
/// pre-emphasized in place.
fn lpcnet_compute_single_frame_features_impl(
    st: &mut LpcnetEncState,
    x: &mut [f32; FRAME_SIZE],
    features: &mut [f32],
) {
    preemphasis_inplace(x, &mut st.mem_preemph, PREEMPHASIS, FRAME_SIZE);
    compute_frame_features(st, x);
    features[..NB_TOTAL_FEATURES].copy_from_slice(&st.features);
}

/// Port of dnn/lpcnet_enc.c:lpcnet_compute_single_frame_features (C always returns 0).
/// `pcm` holds `FRAME_SIZE` samples at 16 kHz; `features` receives `NB_TOTAL_FEATURES`.
pub fn lpcnet_compute_single_frame_features(
    st: &mut LpcnetEncState,
    pcm: &[i16],
    features: &mut [f32],
) {
    let mut x = [0f32; FRAME_SIZE];
    for (xi, &p) in x.iter_mut().zip(&pcm[..FRAME_SIZE]) {
        *xi = f32::from(p);
    }
    lpcnet_compute_single_frame_features_impl(st, &mut x, features);
}

/// Port of dnn/lpcnet_enc.c:lpcnet_compute_single_frame_features_float (C always returns 0).
pub fn lpcnet_compute_single_frame_features_float(
    st: &mut LpcnetEncState,
    pcm: &[f32],
    features: &mut [f32],
) {
    let mut x = [0f32; FRAME_SIZE];
    x.copy_from_slice(&pcm[..FRAME_SIZE]);
    lpcnet_compute_single_frame_features_impl(st, &mut x, features);
}
