//! Port of `dnn/fargan.c` / `dnn/fargan.h` (FARGAN neural vocoder used by deep PLC and DRED)
//! plus the model binding (`init_fargan`) and layer sizes of the generated
//! `dnn/fargan_data.c` / `fargan_data.h` (libopus 1.6.1 model).
//!
//! Upstream binds the compiled-in model arrays in `fargan_init`; the port always loads them
//! from a weight blob ([`FarganState::load_model`], upstream `fargan_load_model`), per PLAN
//! D-015. [`FarganState::fargan_init`] therefore clears the state but keeps the loaded model
//! (the equivalent of upstream re-binding the compiled-in tables).
//!
//! Entry points: [`fargan_cont`] (prime the vocoder from 320 past samples and 5 feature
//! vectors), then [`fargan_synthesize`] / [`fargan_synthesize_int`] (one 10 ms frame of
//! 16 kHz audio per 20-dim feature vector).

use alloc::boxed::Box;

use crate::Result;
use crate::celt::arch::{imax, imin, max32, min16, min32};
use crate::math;

use super::freq::NB_BANDS;
use super::lpcnet_enc::{LPCNET_FRAME_SIZE, NB_FEATURES};
use super::nnet::{
    ACTIVATION_LINEAR, ACTIVATION_SIGMOID, ACTIVATION_TANH, LinearLayer, compute_generic_conv1d,
    compute_generic_dense, compute_generic_gru, compute_glu, compute_glu_inplace,
};
use super::parse_lpcnet_weights::{WeightArray, linear_init, parse_weights};
use super::pitchdnn::PITCH_MAX_PERIOD;

// ---- fargan_data.h (generated) -------------------------------------------------------------

/// `COND_NET_PEMBED_OUT_SIZE`.
pub const COND_NET_PEMBED_OUT_SIZE: usize = 12;
/// `COND_NET_FDENSE1_OUT_SIZE`.
pub const COND_NET_FDENSE1_OUT_SIZE: usize = 64;
/// `COND_NET_FCONV1_OUT_SIZE`.
pub const COND_NET_FCONV1_OUT_SIZE: usize = 128;
/// `COND_NET_FCONV1_IN_SIZE`.
pub const COND_NET_FCONV1_IN_SIZE: usize = 64;
/// `COND_NET_FCONV1_STATE_SIZE`.
pub const COND_NET_FCONV1_STATE_SIZE: usize = 64 * 2;
/// `COND_NET_FCONV1_DELAY`.
pub const COND_NET_FCONV1_DELAY: usize = 1;
/// `COND_NET_FDENSE2_OUT_SIZE`.
pub const COND_NET_FDENSE2_OUT_SIZE: usize = 320;
/// `SIG_NET_COND_GAIN_DENSE_OUT_SIZE`.
pub const SIG_NET_COND_GAIN_DENSE_OUT_SIZE: usize = 1;
/// `SIG_NET_FWC0_CONV_OUT_SIZE`.
pub const SIG_NET_FWC0_CONV_OUT_SIZE: usize = 192;
/// `SIG_NET_FWC0_GLU_GATE_OUT_SIZE`.
pub const SIG_NET_FWC0_GLU_GATE_OUT_SIZE: usize = 192;
/// `SIG_NET_GRU1_OUT_SIZE`.
pub const SIG_NET_GRU1_OUT_SIZE: usize = 160;
/// `SIG_NET_GRU1_STATE_SIZE`.
pub const SIG_NET_GRU1_STATE_SIZE: usize = 160;
/// `SIG_NET_GRU2_OUT_SIZE`.
pub const SIG_NET_GRU2_OUT_SIZE: usize = 128;
/// `SIG_NET_GRU2_STATE_SIZE`.
pub const SIG_NET_GRU2_STATE_SIZE: usize = 128;
/// `SIG_NET_GRU3_OUT_SIZE`.
pub const SIG_NET_GRU3_OUT_SIZE: usize = 128;
/// `SIG_NET_GRU3_STATE_SIZE`.
pub const SIG_NET_GRU3_STATE_SIZE: usize = 128;
/// `SIG_NET_GRU1_GLU_GATE_OUT_SIZE`.
pub const SIG_NET_GRU1_GLU_GATE_OUT_SIZE: usize = 160;
/// `SIG_NET_GRU2_GLU_GATE_OUT_SIZE`.
pub const SIG_NET_GRU2_GLU_GATE_OUT_SIZE: usize = 128;
/// `SIG_NET_GRU3_GLU_GATE_OUT_SIZE`.
pub const SIG_NET_GRU3_GLU_GATE_OUT_SIZE: usize = 128;
/// `SIG_NET_SKIP_GLU_GATE_OUT_SIZE`.
pub const SIG_NET_SKIP_GLU_GATE_OUT_SIZE: usize = 128;
/// `SIG_NET_SKIP_DENSE_OUT_SIZE`.
pub const SIG_NET_SKIP_DENSE_OUT_SIZE: usize = 128;
/// `SIG_NET_SIG_DENSE_OUT_OUT_SIZE`.
pub const SIG_NET_SIG_DENSE_OUT_OUT_SIZE: usize = 40;
/// `SIG_NET_GAIN_DENSE_OUT_OUT_SIZE`.
pub const SIG_NET_GAIN_DENSE_OUT_OUT_SIZE: usize = 4;

// ---- fargan.h ------------------------------------------------------------------------------

/// `FARGAN_CONT_SAMPLES`.
pub const FARGAN_CONT_SAMPLES: usize = 320;
/// `FARGAN_NB_SUBFRAMES`.
pub const FARGAN_NB_SUBFRAMES: usize = 4;
/// `FARGAN_SUBFRAME_SIZE`.
pub const FARGAN_SUBFRAME_SIZE: usize = 40;
/// `FARGAN_FRAME_SIZE`.
pub const FARGAN_FRAME_SIZE: usize = FARGAN_NB_SUBFRAMES * FARGAN_SUBFRAME_SIZE;
/// `FARGAN_COND_SIZE`.
pub const FARGAN_COND_SIZE: usize = COND_NET_FDENSE2_OUT_SIZE / FARGAN_NB_SUBFRAMES;
/// `FARGAN_DEEMPHASIS`.
pub const FARGAN_DEEMPHASIS: f32 = 0.85;
/// `SIG_NET_INPUT_SIZE`.
pub const SIG_NET_INPUT_SIZE: usize = FARGAN_COND_SIZE + 2 * FARGAN_SUBFRAME_SIZE + 4;
/// `SIG_NET_FWC0_STATE_SIZE`.
pub const SIG_NET_FWC0_STATE_SIZE: usize = 2 * SIG_NET_INPUT_SIZE;
/// `FARGAN_MAX_RNN_NEURONS`.
pub const FARGAN_MAX_RNN_NEURONS: usize = SIG_NET_GRU1_OUT_SIZE;

/// `FARGAN_FEATURES` (fargan.c).
const FARGAN_FEATURES: usize = NB_FEATURES;

/// Size of the `skip_cat` input of `sig_net_skip_dense` (C declares `skip_cat[10000]` and
/// uses the first 688 entries).
const SKIP_CAT_SIZE: usize = SIG_NET_GRU1_OUT_SIZE
    + SIG_NET_GRU2_OUT_SIZE
    + SIG_NET_GRU3_OUT_SIZE
    + SIG_NET_FWC0_CONV_OUT_SIZE
    + 2 * FARGAN_SUBFRAME_SIZE;

/// `FARGAN` (fargan_data.h): the model layers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Fargan {
    pub cond_net_pembed: LinearLayer,
    pub cond_net_fdense1: LinearLayer,
    pub cond_net_fconv1: LinearLayer,
    pub cond_net_fdense2: LinearLayer,
    pub sig_net_cond_gain_dense: LinearLayer,
    pub sig_net_fwc0_conv: LinearLayer,
    pub sig_net_fwc0_glu_gate: LinearLayer,
    pub sig_net_gru1_input: LinearLayer,
    pub sig_net_gru1_recurrent: LinearLayer,
    pub sig_net_gru2_input: LinearLayer,
    pub sig_net_gru2_recurrent: LinearLayer,
    pub sig_net_gru3_input: LinearLayer,
    pub sig_net_gru3_recurrent: LinearLayer,
    pub sig_net_gru1_glu_gate: LinearLayer,
    pub sig_net_gru2_glu_gate: LinearLayer,
    pub sig_net_gru3_glu_gate: LinearLayer,
    pub sig_net_skip_glu_gate: LinearLayer,
    pub sig_net_skip_dense: LinearLayer,
    pub sig_net_sig_dense_out: LinearLayer,
    pub sig_net_gain_dense_out: LinearLayer,
}

/// `linear_init` with the generated naming scheme: `<name>_bias`, `<name>_subias`,
/// `<name>_weights_int8`, `<name>_weights_float`, `<name>_scale` (no sparse index, no diag).
/// `bias` / `int8` select which of the optional names the generated line passes.
fn bind(
    arrays: &[WeightArray<'_>],
    names: [&str; 5],
    bias: bool,
    int8: bool,
    nb_inputs: usize,
    nb_outputs: usize,
) -> Result<LinearLayer> {
    let [b, sb, w8, wf, sc] = names;
    linear_init(
        arrays,
        bias.then_some(b),
        int8.then_some(sb),
        int8.then_some(w8),
        Some(wf),
        None,
        None,
        int8.then_some(sc),
        nb_inputs,
        nb_outputs,
    )
}

/// Expands a layer name into the five generated array names.
macro_rules! names {
    ($n:literal) => {
        [
            concat!($n, "_bias"),
            concat!($n, "_subias"),
            concat!($n, "_weights_int8"),
            concat!($n, "_weights_float"),
            concat!($n, "_scale"),
        ]
    };
}

/// Port of the generated dnn/fargan_data.c:init_fargan (fails with `BadArg` where C returns 1).
/// Each call mirrors one generated `linear_init` line (same names, sizes and order).
pub fn init_fargan(arrays: &[WeightArray<'_>]) -> Result<Fargan> {
    Ok(Fargan {
        cond_net_pembed: bind(arrays, names!("cond_net_pembed"), true, false, 224, 12)?,
        cond_net_fdense1: bind(arrays, names!("cond_net_fdense1"), true, false, 32, 64)?,
        cond_net_fconv1: bind(arrays, names!("cond_net_fconv1"), true, true, 192, 128)?,
        cond_net_fdense2: bind(arrays, names!("cond_net_fdense2"), true, true, 128, 320)?,
        sig_net_cond_gain_dense: bind(
            arrays,
            names!("sig_net_cond_gain_dense"),
            true,
            false,
            80,
            1,
        )?,
        sig_net_fwc0_conv: bind(arrays, names!("sig_net_fwc0_conv"), true, true, 328, 192)?,
        sig_net_fwc0_glu_gate: bind(
            arrays,
            names!("sig_net_fwc0_glu_gate"),
            true,
            true,
            192,
            192,
        )?,
        sig_net_gru1_input: bind(arrays, names!("sig_net_gru1_input"), false, true, 272, 480)?,
        sig_net_gru1_recurrent: bind(
            arrays,
            names!("sig_net_gru1_recurrent"),
            false,
            true,
            160,
            480,
        )?,
        sig_net_gru2_input: bind(arrays, names!("sig_net_gru2_input"), false, true, 240, 384)?,
        sig_net_gru2_recurrent: bind(
            arrays,
            names!("sig_net_gru2_recurrent"),
            false,
            true,
            128,
            384,
        )?,
        sig_net_gru3_input: bind(arrays, names!("sig_net_gru3_input"), false, true, 208, 384)?,
        sig_net_gru3_recurrent: bind(
            arrays,
            names!("sig_net_gru3_recurrent"),
            false,
            true,
            128,
            384,
        )?,
        sig_net_gru1_glu_gate: bind(
            arrays,
            names!("sig_net_gru1_glu_gate"),
            true,
            true,
            160,
            160,
        )?,
        sig_net_gru2_glu_gate: bind(
            arrays,
            names!("sig_net_gru2_glu_gate"),
            true,
            true,
            128,
            128,
        )?,
        sig_net_gru3_glu_gate: bind(
            arrays,
            names!("sig_net_gru3_glu_gate"),
            true,
            true,
            128,
            128,
        )?,
        sig_net_skip_glu_gate: bind(
            arrays,
            names!("sig_net_skip_glu_gate"),
            true,
            true,
            128,
            128,
        )?,
        sig_net_skip_dense: bind(arrays, names!("sig_net_skip_dense"), true, true, 688, 128)?,
        sig_net_sig_dense_out: bind(arrays, names!("sig_net_sig_dense_out"), true, true, 128, 40)?,
        sig_net_gain_dense_out: bind(
            arrays,
            names!("sig_net_gain_dense_out"),
            true,
            false,
            192,
            4,
        )?,
    })
}

/// `FARGANState` (fargan.h). The C `arch` field is dropped.
#[derive(Debug, Clone, PartialEq)]
pub struct FarganState {
    pub model: Fargan,
    pub cont_initialized: bool,
    pub deemph_mem: f32,
    pub pitch_buf: [f32; PITCH_MAX_PERIOD],
    pub cond_conv1_state: [f32; COND_NET_FCONV1_STATE_SIZE],
    pub fwc0_mem: [f32; SIG_NET_FWC0_STATE_SIZE],
    pub gru1_state: [f32; SIG_NET_GRU1_STATE_SIZE],
    pub gru2_state: [f32; SIG_NET_GRU2_STATE_SIZE],
    pub gru3_state: [f32; SIG_NET_GRU3_STATE_SIZE],
    pub last_period: i32,
}

impl Default for FarganState {
    fn default() -> Self {
        Self::new_inline()
    }
}

impl FarganState {
    /// A cleared state without a model (C `OPUS_CLEAR` in `fargan_init`).
    #[must_use]
    pub fn new_inline() -> Self {
        Self {
            model: Fargan::default(),
            cont_initialized: false,
            deemph_mem: 0.0,
            pitch_buf: [0.0; PITCH_MAX_PERIOD],
            cond_conv1_state: [0.0; COND_NET_FCONV1_STATE_SIZE],
            fwc0_mem: [0.0; SIG_NET_FWC0_STATE_SIZE],
            gru1_state: [0.0; SIG_NET_GRU1_STATE_SIZE],
            gru2_state: [0.0; SIG_NET_GRU2_STATE_SIZE],
            gru3_state: [0.0; SIG_NET_GRU3_STATE_SIZE],
            last_period: 0,
        }
    }

    /// Port of dnn/fargan.c:fargan_init for a new heap state (no model until
    /// [`Self::load_model`]).
    #[must_use]
    pub fn new() -> Box<Self> {
        Box::new(Self::new_inline())
    }

    /// Port of dnn/fargan.c:fargan_init: clears the state. Upstream then re-binds the
    /// compiled-in model; here the loaded model (if any) is kept.
    pub fn fargan_init(&mut self) {
        let model = core::mem::take(&mut self.model);
        *self = Self::new_inline();
        self.model = model;
    }

    /// Port of dnn/fargan.c:fargan_load_model: parses a weight blob and binds the model
    /// (C returns -1 on failure). On failure the previous model is kept (C would have
    /// partially overwritten it, or crashed on an unparsable blob).
    pub fn load_model(&mut self, data: &[u8]) -> Result<()> {
        let list = parse_weights(data)?;
        self.model = init_fargan(&list)?;
        Ok(())
    }
}

/// `period = (int)floor(.5+256./pow(2.f,((1./60.)*((features[NB_BANDS]+1.5)*60))))`
/// (all in double).
///
/// The `(int)` cast of an out-of-range or NaN value is undefined in C. Rust's saturating cast
/// (NaN → 0) matches the AArch64 `fcvtzs` the oracle runs on (x86-64 `cvttsd2si` would give
/// `INT_MIN`); it keeps the period in `0..=i32::MAX`, so the period arithmetic below cannot
/// overflow. Only non-finite or absurd pitch features (never produced by the models) get there.
#[inline]
fn feature_period(pitch_feature: f32) -> i32 {
    math::floor(0.5 + 256.0 / math::pow(2.0, (1.0 / 60.0) * ((pitch_feature as f64 + 1.5) * 60.0)))
        as i32
}

/// Port of dnn/fargan.c:compute_fargan_cond (static). `cond` receives
/// `COND_NET_FDENSE2_OUT_SIZE` values; `features` holds `NB_FEATURES`.
pub fn compute_fargan_cond(st: &mut FarganState, cond: &mut [f32], features: &[f32], period: i32) {
    let mut dense_in = [0f32; NB_FEATURES + COND_NET_PEMBED_OUT_SIZE];
    let mut conv1_in = [0f32; COND_NET_FCONV1_IN_SIZE];
    let mut fdense2_in = [0f32; COND_NET_FCONV1_OUT_SIZE];
    let model = &st.model;
    debug_assert!(FARGAN_FEATURES + COND_NET_PEMBED_OUT_SIZE == model.cond_net_fdense1.nb_inputs);
    debug_assert!(COND_NET_FCONV1_IN_SIZE == model.cond_net_fdense1.nb_outputs);
    debug_assert!(COND_NET_FCONV1_OUT_SIZE == model.cond_net_fconv1.nb_outputs);
    let row = imax(0, imin(period - 32, 223)) as usize;
    // C dereferences the float weights unconditionally (crashes without a model); an unloaded
    // model contributes zeros here, like every unbound layer in `compute_linear`.
    debug_assert!(model.cond_net_pembed.float_weights.is_some());
    if let Some(pembed) = model.cond_net_pembed.float_weights.as_deref() {
        dense_in[NB_FEATURES..].copy_from_slice(
            &pembed[row * COND_NET_PEMBED_OUT_SIZE..(row + 1) * COND_NET_PEMBED_OUT_SIZE],
        );
    }
    dense_in[..NB_FEATURES].copy_from_slice(&features[..NB_FEATURES]);

    compute_generic_dense(
        &model.cond_net_fdense1,
        &mut conv1_in,
        &dense_in,
        ACTIVATION_TANH,
    );
    compute_generic_conv1d(
        &model.cond_net_fconv1,
        &mut fdense2_in,
        &mut st.cond_conv1_state,
        &conv1_in,
        COND_NET_FCONV1_IN_SIZE,
        ACTIVATION_TANH,
    );
    compute_generic_dense(&model.cond_net_fdense2, cond, &fdense2_in, ACTIVATION_TANH);
}

/// Port of dnn/fargan.c:fargan_deemphasis (static): `FARGAN_SUBFRAME_SIZE` samples in place.
pub fn fargan_deemphasis(pcm: &mut [f32], deemph_mem: &mut f32) {
    for p in &mut pcm[..FARGAN_SUBFRAME_SIZE] {
        *p += FARGAN_DEEMPHASIS * *deemph_mem;
        *deemph_mem = *p;
    }
}

/// `st->pitch_buf[idx]` as C reads it. Indices past the end only occur for a period of 0
/// (pitch feature above ~7.5 or NaN, never produced by the models): C then reads on into the
/// next struct field, `cond_conv1_state` (at most 42 entries), which is reproduced here instead
/// of panicking. (Negative periods, which could read further, cannot come from a feature.)
#[inline]
const fn pitch_buf_at(st: &FarganState, idx: usize) -> f32 {
    if idx < PITCH_MAX_PERIOD {
        st.pitch_buf[idx]
    } else {
        st.cond_conv1_state[idx - PITCH_MAX_PERIOD]
    }
}

/// Port of dnn/fargan.c:run_fargan_subframe (static). Writes `FARGAN_SUBFRAME_SIZE` samples
/// to `pcm`; `cond` holds `FARGAN_COND_SIZE` values. `period` is a feature-derived period
/// (`0..=i32::MAX`); a negative one would make C read further past `pitch_buf`.
pub fn run_fargan_subframe(st: &mut FarganState, pcm: &mut [f32], cond: &[f32], period: i32) {
    let mut fwc0_in = [0f32; SIG_NET_INPUT_SIZE];
    let mut gru1_in = [0f32; SIG_NET_FWC0_CONV_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE];
    let mut gru2_in = [0f32; SIG_NET_GRU1_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE];
    let mut gru3_in = [0f32; SIG_NET_GRU2_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE];
    let mut pred = [0f32; FARGAN_SUBFRAME_SIZE + 4];
    let mut prev = [0f32; FARGAN_SUBFRAME_SIZE];
    let mut pitch_gate = [0f32; 4];
    let mut gain = [0f32; SIG_NET_COND_GAIN_DENSE_OUT_SIZE];
    let mut skip_cat = [0f32; SKIP_CAT_SIZE];
    let mut skip_out = [0f32; SIG_NET_SKIP_DENSE_OUT_SIZE];
    let pcm = &mut pcm[..FARGAN_SUBFRAME_SIZE];

    debug_assert!(st.cont_initialized);

    compute_generic_dense(
        &st.model.sig_net_cond_gain_dense,
        &mut gain,
        cond,
        ACTIVATION_LINEAR,
    );
    let gain = math::exp(gain[0] as f64) as f32;
    let gain_1 = 1.0f32 / (1e-5f32 + gain);

    let mut pos = PITCH_MAX_PERIOD as i32 - period - 2;
    for p in &mut pred {
        *p = min32(
            1.0,
            max32(-1.0, gain_1 * pitch_buf_at(st, imax(0, pos) as usize)),
        );
        pos += 1;
        if pos == PITCH_MAX_PERIOD as i32 {
            pos -= period;
        }
    }
    for (i, p) in prev.iter_mut().enumerate() {
        *p = max32(
            -1.0,
            min16(
                1.0,
                gain_1 * st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE + i],
            ),
        );
    }

    fwc0_in[..FARGAN_COND_SIZE].copy_from_slice(&cond[..FARGAN_COND_SIZE]);
    fwc0_in[FARGAN_COND_SIZE..FARGAN_COND_SIZE + FARGAN_SUBFRAME_SIZE + 4].copy_from_slice(&pred);
    fwc0_in[FARGAN_COND_SIZE + FARGAN_SUBFRAME_SIZE + 4..].copy_from_slice(&prev);

    let model = &st.model;
    compute_generic_conv1d(
        &model.sig_net_fwc0_conv,
        &mut gru1_in,
        &mut st.fwc0_mem,
        &fwc0_in,
        SIG_NET_INPUT_SIZE,
        ACTIVATION_TANH,
    );
    debug_assert!(SIG_NET_FWC0_GLU_GATE_OUT_SIZE == model.sig_net_fwc0_glu_gate.nb_outputs);
    compute_glu_inplace(&model.sig_net_fwc0_glu_gate, &mut gru1_in);

    compute_generic_dense(
        &model.sig_net_gain_dense_out,
        &mut pitch_gate,
        &gru1_in,
        ACTIVATION_SIGMOID,
    );

    for i in 0..FARGAN_SUBFRAME_SIZE {
        gru1_in[SIG_NET_FWC0_GLU_GATE_OUT_SIZE + i] = pitch_gate[0] * pred[i + 2];
    }
    gru1_in[SIG_NET_FWC0_GLU_GATE_OUT_SIZE + FARGAN_SUBFRAME_SIZE..].copy_from_slice(&prev);
    compute_generic_gru(
        &model.sig_net_gru1_input,
        &model.sig_net_gru1_recurrent,
        &mut st.gru1_state,
        &gru1_in,
    );
    compute_glu(&model.sig_net_gru1_glu_gate, &mut gru2_in, &st.gru1_state);

    for i in 0..FARGAN_SUBFRAME_SIZE {
        gru2_in[SIG_NET_GRU1_OUT_SIZE + i] = pitch_gate[1] * pred[i + 2];
    }
    gru2_in[SIG_NET_GRU1_OUT_SIZE + FARGAN_SUBFRAME_SIZE..].copy_from_slice(&prev);
    compute_generic_gru(
        &model.sig_net_gru2_input,
        &model.sig_net_gru2_recurrent,
        &mut st.gru2_state,
        &gru2_in,
    );
    compute_glu(&model.sig_net_gru2_glu_gate, &mut gru3_in, &st.gru2_state);

    for i in 0..FARGAN_SUBFRAME_SIZE {
        gru3_in[SIG_NET_GRU2_OUT_SIZE + i] = pitch_gate[2] * pred[i + 2];
    }
    gru3_in[SIG_NET_GRU2_OUT_SIZE + FARGAN_SUBFRAME_SIZE..].copy_from_slice(&prev);
    compute_generic_gru(
        &model.sig_net_gru3_input,
        &model.sig_net_gru3_recurrent,
        &mut st.gru3_state,
        &gru3_in,
    );
    const G12: usize = SIG_NET_GRU1_OUT_SIZE + SIG_NET_GRU2_OUT_SIZE;
    const G123: usize = G12 + SIG_NET_GRU3_OUT_SIZE;
    const G123F: usize = G123 + SIG_NET_FWC0_CONV_OUT_SIZE;
    compute_glu(
        &model.sig_net_gru3_glu_gate,
        &mut skip_cat[G12..G123],
        &st.gru3_state,
    );

    skip_cat[..SIG_NET_GRU1_OUT_SIZE].copy_from_slice(&gru2_in[..SIG_NET_GRU1_OUT_SIZE]);
    skip_cat[SIG_NET_GRU1_OUT_SIZE..G12].copy_from_slice(&gru3_in[..SIG_NET_GRU2_OUT_SIZE]);
    skip_cat[G123..G123F].copy_from_slice(&gru1_in[..SIG_NET_FWC0_CONV_OUT_SIZE]);
    for i in 0..FARGAN_SUBFRAME_SIZE {
        skip_cat[G123F + i] = pitch_gate[3] * pred[i + 2];
    }
    skip_cat[G123F + FARGAN_SUBFRAME_SIZE..].copy_from_slice(&prev);

    compute_generic_dense(
        &model.sig_net_skip_dense,
        &mut skip_out,
        &skip_cat,
        ACTIVATION_TANH,
    );
    compute_glu_inplace(&model.sig_net_skip_glu_gate, &mut skip_out);

    compute_generic_dense(
        &model.sig_net_sig_dense_out,
        pcm,
        &skip_out,
        ACTIVATION_TANH,
    );
    for p in pcm.iter_mut() {
        *p *= gain;
    }

    st.pitch_buf.copy_within(FARGAN_SUBFRAME_SIZE.., 0);
    st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE..].copy_from_slice(pcm);
    fargan_deemphasis(pcm, &mut st.deemph_mem);
}

/// Port of dnn/fargan.c:fargan_cont: primes the vocoder with the last
/// `FARGAN_CONT_SAMPLES` output samples `pcm0` (float, ±1 scale) and the `5*NB_FEATURES`
/// features `features0` of the frames that produced them.
pub fn fargan_cont(st: &mut FarganState, pcm0: &[f32], features0: &[f32]) {
    let mut cond = [0f32; COND_NET_FDENSE2_OUT_SIZE];
    let mut x0 = [0f32; FARGAN_CONT_SAMPLES];
    let mut dummy = [0f32; FARGAN_SUBFRAME_SIZE];
    let mut period = 0i32;
    let pcm0 = &pcm0[..FARGAN_CONT_SAMPLES];

    // Pre-load features.
    for i in 0..5 {
        let features = &features0[i * NB_FEATURES..(i + 1) * NB_FEATURES];
        st.last_period = period;
        period = feature_period(features[NB_BANDS]);
        compute_fargan_cond(st, &mut cond, features, period);
    }

    x0[0] = 0.0;
    for i in 1..FARGAN_CONT_SAMPLES {
        x0[i] = pcm0[i] - FARGAN_DEEMPHASIS * pcm0[i - 1];
    }

    st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_FRAME_SIZE..].copy_from_slice(&x0[..FARGAN_FRAME_SIZE]);
    st.cont_initialized = true;

    for i in 0..FARGAN_NB_SUBFRAMES {
        let last_period = st.last_period;
        run_fargan_subframe(
            st,
            &mut dummy,
            &cond[i * FARGAN_COND_SIZE..(i + 1) * FARGAN_COND_SIZE],
            last_period,
        );
        st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE..].copy_from_slice(
            &x0[FARGAN_FRAME_SIZE + i * FARGAN_SUBFRAME_SIZE
                ..FARGAN_FRAME_SIZE + (i + 1) * FARGAN_SUBFRAME_SIZE],
        );
    }
    st.deemph_mem = pcm0[FARGAN_CONT_SAMPLES - 1];
}

/// Port of dnn/fargan.c:fargan_synthesize_impl (static).
fn fargan_synthesize_impl(st: &mut FarganState, pcm: &mut [f32], features: &[f32]) {
    let mut cond = [0f32; COND_NET_FDENSE2_OUT_SIZE];
    debug_assert!(st.cont_initialized);

    let period = feature_period(features[NB_BANDS]);
    compute_fargan_cond(st, &mut cond, features, period);
    for subframe in 0..FARGAN_NB_SUBFRAMES {
        let sub_cond = &cond[subframe * FARGAN_COND_SIZE..(subframe + 1) * FARGAN_COND_SIZE];
        let last_period = st.last_period;
        run_fargan_subframe(
            st,
            &mut pcm[subframe * FARGAN_SUBFRAME_SIZE..(subframe + 1) * FARGAN_SUBFRAME_SIZE],
            sub_cond,
            last_period,
        );
    }
    st.last_period = period;
}

/// Port of dnn/fargan.c:fargan_synthesize: one frame (`FARGAN_FRAME_SIZE` float samples,
/// ±1 scale) from `NB_FEATURES` features. [`fargan_cont`] must have been called first.
pub fn fargan_synthesize(st: &mut FarganState, pcm: &mut [f32], features: &[f32]) {
    fargan_synthesize_impl(st, pcm, features);
}

/// Port of dnn/fargan.c:fargan_synthesize_int: like [`fargan_synthesize`] with 16-bit output
/// (`LPCNET_FRAME_SIZE` samples).
pub fn fargan_synthesize_int(st: &mut FarganState, pcm: &mut [i16], features: &[f32]) {
    let mut fpcm = [0f32; FARGAN_FRAME_SIZE];
    fargan_synthesize(st, &mut fpcm, features);
    for (p, &f) in pcm[..LPCNET_FRAME_SIZE].iter_mut().zip(&fpcm) {
        // C: (int)floor(.5 + MIN32(32767, MAX32(-32767, 32768.f*fpcm[i]))) stored to int16
        // (a NaN sample gives 0 with the saturating cast, as on the AArch64 oracle).
        *p =
            math::floor(0.5 + min32(32767.0, max32(-32767.0, 32768.0f32 * f)) as f64) as i32 as i16;
    }
}
