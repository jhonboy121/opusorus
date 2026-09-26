//! Port of `dnn/lpcnet_plc.c` (neural packet-loss concealment), the `LPCNetPLCState` /
//! `PLCNetState` definitions of `dnn/lpcnet_private.h`, the PLC entry points of
//! `dnn/lpcnet.h`, and the model binding (`init_plcmodel`) and layer sizes of the generated
//! `dnn/plc_data.c` / `plc_data.h` (libopus 1.6.1 model, checkpoint `plc4ar_16.pth`).
//!
//! Weights are always loaded from a blob (PLAN D-015): [`LpcnetPlcState::load_model`]
//! (upstream `lpcnet_plc_load_model`) binds the PLC model, the pitch DNN of the feature
//! extractor and FARGAN from one blob holding all three. Until then `loaded` is false and the
//! decoders must not call [`lpcnet_plc_conceal`] (upstream checks `lpcnet->loaded` too).
//!
//! # Decoder integration (not wired yet)
//!
//! The Opus/SILK/CELT decoders drive one [`LpcnetPlcState`] per Opus decoder (C
//! `OpusDecoder::lpcnet`):
//! * `opus_decoder_init` → [`LpcnetPlcState::new`] / [`LpcnetPlcState::lpcnet_plc_init`]
//!   (src/opus_decoder.c:180), `OPUS_RESET_STATE` → [`LpcnetPlcState::lpcnet_plc_reset`]
//!   (:1120), `OPUS_SET_DNN_BLOB` → [`LpcnetPlcState::load_model`] (:1226).
//! * DRED: [`lpcnet_plc_fec_clear`] then [`lpcnet_plc_fec_add`] per feature vector, `None`
//!   for a skipped one (src/opus_decoder.c:741-754).
//! * Every good 10 ms frame at 16 kHz: [`lpcnet_plc_update`] (celt/celt_decoder.c:668,
//!   silk/PLC.c:109 and :412).
//! * Every lost 10 ms frame at 16 kHz: [`lpcnet_plc_conceal`] (celt/celt_decoder.c:1039,
//!   silk/PLC.c:404), guarded by `loaded`.

use alloc::boxed::Box;

use crate::Result;
use crate::celt::arch::max16;

use super::fargan::{FARGAN_CONT_SAMPLES, FarganState, fargan_cont, fargan_synthesize_int};
use super::freq::{FRAME_SIZE, NB_BANDS, burg_cepstral_analysis};
use super::lpcnet_enc::{
    CONT_VECTORS, LpcnetEncState, NB_FEATURES, NB_TOTAL_FEATURES, PLC_MAX_FEC,
    lpcnet_compute_single_frame_features_float,
};
use super::nnet::{
    ACTIVATION_LINEAR, ACTIVATION_TANH, LinearLayer, compute_generic_dense, compute_generic_gru,
};
use super::parse_lpcnet_weights::{WeightArray, linear_init, parse_weights};

// ---- plc_data.h (generated) ----------------------------------------------------------------

/// `PLC_DENSE_IN_OUT_SIZE`.
pub const PLC_DENSE_IN_OUT_SIZE: usize = 128;
/// `PLC_DENSE_OUT_OUT_SIZE`.
pub const PLC_DENSE_OUT_OUT_SIZE: usize = 20;
/// `PLC_GRU1_OUT_SIZE`.
pub const PLC_GRU1_OUT_SIZE: usize = 192;
/// `PLC_GRU1_STATE_SIZE`.
pub const PLC_GRU1_STATE_SIZE: usize = 192;
/// `PLC_GRU2_OUT_SIZE`.
pub const PLC_GRU2_OUT_SIZE: usize = 192;
/// `PLC_GRU2_STATE_SIZE`.
pub const PLC_GRU2_STATE_SIZE: usize = 192;
/// `PLC_MAX_RNN_UNITS`.
pub const PLC_MAX_RNN_UNITS: usize = 192;

// ---- lpcnet_private.h ----------------------------------------------------------------------

/// `PLC_BUF_SIZE`.
pub const PLC_BUF_SIZE: usize = (CONT_VECTORS + 10) * FRAME_SIZE;

/// Size of the PLC network input: Burg cepstrum (`2*NB_BANDS`), features, and a flag.
pub const PLC_INPUT_SIZE: usize = 2 * NB_BANDS + NB_FEATURES + 1;

/// `PLCModel` (plc_data.h): the model layers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlcModel {
    pub plc_dense_in: LinearLayer,
    pub plc_dense_out: LinearLayer,
    pub plc_gru1_input: LinearLayer,
    pub plc_gru1_recurrent: LinearLayer,
    pub plc_gru2_input: LinearLayer,
    pub plc_gru2_recurrent: LinearLayer,
}

/// Port of the generated dnn/plc_data.c:init_plcmodel (fails with `BadArg` where C returns 1).
pub fn init_plcmodel(arrays: &[WeightArray<'_>]) -> Result<PlcModel> {
    Ok(PlcModel {
        plc_dense_in: linear_init(
            arrays,
            Some("plc_dense_in_bias"),
            None,
            None,
            Some("plc_dense_in_weights_float"),
            None,
            None,
            None,
            57,
            128,
        )?,
        plc_dense_out: linear_init(
            arrays,
            Some("plc_dense_out_bias"),
            None,
            None,
            Some("plc_dense_out_weights_float"),
            None,
            None,
            None,
            192,
            20,
        )?,
        plc_gru1_input: linear_init(
            arrays,
            Some("plc_gru1_input_bias"),
            Some("plc_gru1_input_subias"),
            Some("plc_gru1_input_weights_int8"),
            Some("plc_gru1_input_weights_float"),
            None,
            None,
            Some("plc_gru1_input_scale"),
            128,
            576,
        )?,
        plc_gru1_recurrent: linear_init(
            arrays,
            Some("plc_gru1_recurrent_bias"),
            Some("plc_gru1_recurrent_subias"),
            Some("plc_gru1_recurrent_weights_int8"),
            Some("plc_gru1_recurrent_weights_float"),
            None,
            None,
            Some("plc_gru1_recurrent_scale"),
            192,
            576,
        )?,
        plc_gru2_input: linear_init(
            arrays,
            Some("plc_gru2_input_bias"),
            Some("plc_gru2_input_subias"),
            Some("plc_gru2_input_weights_int8"),
            Some("plc_gru2_input_weights_float"),
            None,
            None,
            Some("plc_gru2_input_scale"),
            192,
            576,
        )?,
        plc_gru2_recurrent: linear_init(
            arrays,
            Some("plc_gru2_recurrent_bias"),
            Some("plc_gru2_recurrent_subias"),
            Some("plc_gru2_recurrent_weights_int8"),
            Some("plc_gru2_recurrent_weights_float"),
            None,
            None,
            Some("plc_gru2_recurrent_scale"),
            192,
            576,
        )?,
    })
}

/// `PLCNetState` (lpcnet_private.h).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlcNetState {
    pub gru1_state: [f32; PLC_GRU1_STATE_SIZE],
    pub gru2_state: [f32; PLC_GRU2_STATE_SIZE],
}

impl Default for PlcNetState {
    fn default() -> Self {
        Self {
            gru1_state: [0.0; PLC_GRU1_STATE_SIZE],
            gru2_state: [0.0; PLC_GRU2_STATE_SIZE],
        }
    }
}

/// `struct LPCNetPLCState` (lpcnet_private.h). The C `arch` field is dropped; the fields from
/// `fec` on are the ones `lpcnet_plc_reset` clears (`LPCNET_PLC_RESET_START`).
#[derive(Debug, Clone, PartialEq)]
pub struct LpcnetPlcState {
    pub model: PlcModel,
    pub fargan: FarganState,
    pub enc: LpcnetEncState,
    pub loaded: bool,

    // LPCNET_PLC_RESET_START
    pub fec: [[f32; NB_FEATURES]; PLC_MAX_FEC],
    pub analysis_gap: i32,
    pub fec_read_pos: i32,
    pub fec_fill_pos: i32,
    pub fec_skip: i32,
    pub analysis_pos: i32,
    pub predict_pos: i32,
    pub pcm: [f32; PLC_BUF_SIZE],
    pub blend: i32,
    pub features: [f32; NB_TOTAL_FEATURES],
    pub cont_features: [f32; CONT_VECTORS * NB_FEATURES],
    pub loss_count: i32,
    pub plc_net: PlcNetState,
    pub plc_bak: [PlcNetState; 2],
}

impl Default for LpcnetPlcState {
    fn default() -> Self {
        let mut st = Self::new_inline();
        st.lpcnet_plc_init();
        st
    }
}

impl LpcnetPlcState {
    /// All-zero state (no model).
    fn new_inline() -> Self {
        Self {
            model: PlcModel::default(),
            fargan: FarganState::new_inline(),
            enc: LpcnetEncState::default(),
            loaded: false,
            fec: [[0.0; NB_FEATURES]; PLC_MAX_FEC],
            analysis_gap: 0,
            fec_read_pos: 0,
            fec_fill_pos: 0,
            fec_skip: 0,
            analysis_pos: 0,
            predict_pos: 0,
            pcm: [0.0; PLC_BUF_SIZE],
            blend: 0,
            features: [0.0; NB_TOTAL_FEATURES],
            cont_features: [0.0; CONT_VECTORS * NB_FEATURES],
            loss_count: 0,
            plc_net: PlcNetState::default(),
            plc_bak: [PlcNetState::default(); 2],
        }
    }

    /// A new heap state after [`Self::lpcnet_plc_init`] (no model loaded yet).
    #[must_use]
    pub fn new() -> Box<Self> {
        let mut st = Box::new(Self::new_inline());
        st.lpcnet_plc_init();
        st
    }

    /// Port of dnn/lpcnet_plc.c:lpcnet_plc_reset.
    pub fn lpcnet_plc_reset(&mut self) {
        // OPUS_CLEAR from LPCNET_PLC_RESET_START (`fec`) to the end of the struct.
        self.fec = [[0.0; NB_FEATURES]; PLC_MAX_FEC];
        self.analysis_gap = 0;
        self.fec_read_pos = 0;
        self.fec_fill_pos = 0;
        self.fec_skip = 0;
        self.analysis_pos = 0;
        self.predict_pos = 0;
        self.pcm.fill(0.0);
        self.blend = 0;
        self.features.fill(0.0);
        self.cont_features.fill(0.0);
        self.loss_count = 0;
        self.plc_net = PlcNetState::default();
        self.plc_bak = [PlcNetState::default(); 2];
        self.enc.lpcnet_encoder_init();
        self.pcm.fill(0.0);
        self.blend = 0;
        self.loss_count = 0;
        self.analysis_gap = 1;
        self.analysis_pos = PLC_BUF_SIZE as i32;
        self.predict_pos = PLC_BUF_SIZE as i32;
    }

    /// Port of dnn/lpcnet_plc.c:lpcnet_plc_init (C always returns 0 here).
    ///
    /// Upstream binds the compiled-in PLC model and sets `loaded`; with weights from a blob
    /// (upstream `USE_WEIGHTS_FILE`) nothing is bound. The port keeps whatever models were
    /// loaded (and therefore `loaded`), which is the equivalent of the compiled-in tables:
    /// `fargan_init` / `lpcnet_encoder_init` likewise keep their models.
    pub fn lpcnet_plc_init(&mut self) {
        self.fargan.fargan_init();
        self.enc.lpcnet_encoder_init();
        self.lpcnet_plc_reset();
    }

    /// Port of dnn/lpcnet_plc.c:lpcnet_plc_load_model: binds the PLC model, the pitch DNN
    /// (feature extraction) and FARGAN from one weight blob (C returns -1 on failure).
    ///
    /// As in C, each model that binds successfully is replaced even when a later one fails,
    /// and `loaded` is only set when all three succeed. (C also partially overwrites the
    /// failing model; the port keeps it, and rejects an unparsable blob instead of crashing.)
    pub fn load_model(&mut self, data: &[u8]) -> Result<()> {
        let list = parse_weights(data)?;
        self.model = init_plcmodel(&list)?;
        self.enc.load_model(data)?;
        self.fargan.load_model(data)?;
        self.loaded = true;
        Ok(())
    }
}

/// Port of dnn/lpcnet_plc.c:lpcnet_plc_reset.
pub fn lpcnet_plc_reset(st: &mut LpcnetPlcState) {
    st.lpcnet_plc_reset();
}

/// Port of dnn/lpcnet_plc.c:lpcnet_plc_init (see [`LpcnetPlcState::lpcnet_plc_init`]).
pub fn lpcnet_plc_init(st: &mut LpcnetPlcState) {
    st.lpcnet_plc_init();
}

/// Port of dnn/lpcnet_plc.c:lpcnet_plc_load_model.
pub fn lpcnet_plc_load_model(st: &mut LpcnetPlcState, data: &[u8]) -> Result<()> {
    st.load_model(data)
}

/// Port of dnn/lpcnet_plc.c:lpcnet_plc_fec_add. `None` (C `NULL`) records a skipped vector;
/// otherwise the first `NB_FEATURES` values are queued.
///
/// C asserts `fec_fill_pos < PLC_MAX_FEC` and would write out of bounds past it; the port
/// drops the vector instead (the decoder never queues more than `PLC_MAX_FEC`).
pub fn lpcnet_plc_fec_add(st: &mut LpcnetPlcState, features: Option<&[f32]>) {
    let Some(features) = features else {
        st.fec_skip += 1;
        return;
    };
    debug_assert!((st.fec_fill_pos as usize) < PLC_MAX_FEC);
    if let Some(slot) = st.fec.get_mut(st.fec_fill_pos as usize) {
        slot.copy_from_slice(&features[..NB_FEATURES]);
        st.fec_fill_pos += 1;
    }
}

/// Port of dnn/lpcnet_plc.c:lpcnet_plc_fec_clear.
pub const fn lpcnet_plc_fec_clear(st: &mut LpcnetPlcState) {
    st.fec_read_pos = 0;
    st.fec_fill_pos = 0;
    st.fec_skip = 0;
}

/// Port of dnn/lpcnet_plc.c:compute_plc_pred (static). `input` holds [`PLC_INPUT_SIZE`]
/// values; `out` receives `PLC_DENSE_OUT_OUT_SIZE` (= `NB_FEATURES`).
pub fn compute_plc_pred(st: &mut LpcnetPlcState, out: &mut [f32], input: &[f32]) {
    let mut tmp = [0f32; PLC_DENSE_IN_OUT_SIZE];
    let model = &st.model;
    let net = &mut st.plc_net;
    debug_assert!(st.loaded);
    compute_generic_dense(&model.plc_dense_in, &mut tmp, input, ACTIVATION_TANH);
    compute_generic_gru(
        &model.plc_gru1_input,
        &model.plc_gru1_recurrent,
        &mut net.gru1_state,
        &tmp,
    );
    compute_generic_gru(
        &model.plc_gru2_input,
        &model.plc_gru2_recurrent,
        &mut net.gru2_state,
        &net.gru1_state,
    );
    compute_generic_dense(
        &model.plc_dense_out,
        out,
        &net.gru2_state,
        ACTIVATION_LINEAR,
    );
}

/// Port of dnn/lpcnet_plc.c:get_fec_or_pred (static): the next queued FEC vector (returns
/// true) or a prediction (false), written to `out[..NB_FEATURES]`.
pub fn get_fec_or_pred(st: &mut LpcnetPlcState, out: &mut [f32]) -> bool {
    if st.fec_read_pos != st.fec_fill_pos && st.fec_skip == 0 {
        let mut plc_features = [0f32; PLC_INPUT_SIZE];
        let mut discard = [0f32; NB_FEATURES];
        out[..NB_FEATURES].copy_from_slice(&st.fec[st.fec_read_pos as usize]);
        st.fec_read_pos += 1;
        // Update PLC state using FEC, so without Burg features.
        plc_features[2 * NB_BANDS..2 * NB_BANDS + NB_FEATURES].copy_from_slice(&out[..NB_FEATURES]);
        plc_features[2 * NB_BANDS + NB_FEATURES] = -1.0;
        compute_plc_pred(st, &mut discard, &plc_features);
        true
    } else {
        let zeros = [0f32; PLC_INPUT_SIZE];
        compute_plc_pred(st, out, &zeros);
        if st.fec_skip > 0 {
            st.fec_skip -= 1;
        }
        false
    }
}

/// [`get_fec_or_pred`] into `st.features` (the C calls pass `st->features` as `out`).
fn get_fec_or_pred_features(st: &mut LpcnetPlcState) -> bool {
    let mut out = [0f32; NB_FEATURES];
    out.copy_from_slice(&st.features[..NB_FEATURES]);
    let ret = get_fec_or_pred(st, &mut out);
    st.features[..NB_FEATURES].copy_from_slice(&out);
    ret
}

/// Port of dnn/lpcnet_plc.c:queue_features (static): appends `NB_FEATURES` values to the
/// `CONT_VECTORS`-deep history used by `fargan_cont`.
pub fn queue_features(st: &mut LpcnetPlcState, features: &[f32]) {
    st.cont_features
        .copy_within(NB_FEATURES..CONT_VECTORS * NB_FEATURES, 0);
    st.cont_features[(CONT_VECTORS - 1) * NB_FEATURES..].copy_from_slice(&features[..NB_FEATURES]);
}

/// [`queue_features`] of `st.features`.
fn queue_own_features(st: &mut LpcnetPlcState) {
    let mut f = [0f32; NB_FEATURES];
    f.copy_from_slice(&st.features[..NB_FEATURES]);
    queue_features(st, &f);
}

/// Port of dnn/lpcnet_plc.c:lpcnet_plc_update: records one good 10 ms frame
/// (`FRAME_SIZE` samples at 16 kHz). C always returns 0.
pub fn lpcnet_plc_update(st: &mut LpcnetPlcState, pcm: &[i16]) {
    if st.analysis_pos - FRAME_SIZE as i32 >= 0 {
        st.analysis_pos -= FRAME_SIZE as i32;
    } else {
        st.analysis_gap = 1;
    }
    if st.predict_pos - FRAME_SIZE as i32 >= 0 {
        st.predict_pos -= FRAME_SIZE as i32;
    }
    st.pcm.copy_within(FRAME_SIZE.., 0);
    for (d, &s) in st.pcm[PLC_BUF_SIZE - FRAME_SIZE..]
        .iter_mut()
        .zip(&pcm[..FRAME_SIZE])
    {
        *d = (1.0f32 / 32768.0f32) * f32::from(s);
    }
    st.loss_count = 0;
    st.blend = 0;
}

/// `att_table` (lpcnet_plc.c): double literals stored as float.
const ATT_TABLE: [f32; 10] = [
    0.0,
    0.0,
    -0.2f64 as f32,
    -0.2f64 as f32,
    -0.4f64 as f32,
    -0.4f64 as f32,
    -0.8f64 as f32,
    -0.8f64 as f32,
    -1.6f64 as f32,
    -1.6f64 as f32,
];

/// Port of dnn/lpcnet_plc.c:lpcnet_plc_conceal: synthesizes one lost 10 ms frame
/// (`FRAME_SIZE` samples at 16 kHz) into `pcm`. C always returns 0.
///
/// The model must be loaded (C asserts `st->loaded`; callers check it). Without a model
/// every unbound layer outputs zeros instead of C's NULL dereference.
pub fn lpcnet_plc_conceal(st: &mut LpcnetPlcState, pcm: &mut [i16]) {
    debug_assert!(st.loaded);
    if st.blend == 0 {
        let mut count = 0;
        st.plc_net = st.plc_bak[0];
        while st.analysis_pos + FRAME_SIZE as i32 <= PLC_BUF_SIZE as i32 {
            let mut x = [0f32; FRAME_SIZE];
            let mut plc_features = [0f32; PLC_INPUT_SIZE];
            debug_assert!(st.analysis_pos >= 0);
            let apos = st.analysis_pos as usize;
            for (xi, &p) in x.iter_mut().zip(&st.pcm[apos..apos + FRAME_SIZE]) {
                *xi = 32768.0f32 * p;
            }
            burg_cepstral_analysis(&mut plc_features, &x);
            lpcnet_compute_single_frame_features_float(&mut st.enc, &x, &mut st.features);
            if (st.analysis_gap == 0 || count > 0) && st.analysis_pos >= st.predict_pos {
                queue_own_features(st);
                plc_features[2 * NB_BANDS..2 * NB_BANDS + NB_FEATURES]
                    .copy_from_slice(&st.features[..NB_FEATURES]);
                plc_features[2 * NB_BANDS + NB_FEATURES] = 1.0;
                st.plc_bak[0] = st.plc_bak[1];
                st.plc_bak[1] = st.plc_net;
                let mut out = [0f32; NB_FEATURES];
                compute_plc_pred(st, &mut out, &plc_features);
                st.features[..NB_FEATURES].copy_from_slice(&out);
            }
            st.analysis_pos += FRAME_SIZE as i32;
            count += 1;
        }
        st.plc_bak[0] = st.plc_bak[1];
        st.plc_bak[1] = st.plc_net;
        get_fec_or_pred_features(st);
        queue_own_features(st);
        st.plc_bak[0] = st.plc_bak[1];
        st.plc_bak[1] = st.plc_net;
        get_fec_or_pred_features(st);
        queue_own_features(st);
        fargan_cont(
            &mut st.fargan,
            &st.pcm[PLC_BUF_SIZE - FARGAN_CONT_SAMPLES..],
            &st.cont_features,
        );
        st.analysis_gap = 0;
    }
    st.plc_bak[0] = st.plc_bak[1];
    st.plc_bak[1] = st.plc_net;
    if get_fec_or_pred_features(st) {
        st.loss_count = 0;
    } else {
        st.loss_count += 1;
    }
    if st.loss_count >= 10 {
        st.features[0] = max16(
            -15.0,
            st.features[0] + ATT_TABLE[9] - (2 * (st.loss_count - 9)) as f32,
        );
    } else {
        st.features[0] = max16(-15.0, st.features[0] + ATT_TABLE[st.loss_count as usize]);
    }
    fargan_synthesize_int(&mut st.fargan, pcm, &st.features);
    queue_own_features(st);
    if st.analysis_pos - FRAME_SIZE as i32 >= 0 {
        st.analysis_pos -= FRAME_SIZE as i32;
    } else {
        st.analysis_gap = 1;
    }
    st.predict_pos = PLC_BUF_SIZE as i32;
    st.pcm.copy_within(FRAME_SIZE.., 0);
    for (d, &s) in st.pcm[PLC_BUF_SIZE - FRAME_SIZE..]
        .iter_mut()
        .zip(&pcm[..FRAME_SIZE])
    {
        *d = (1.0f32 / 32768.0f32) * f32::from(s);
    }
    st.blend = 1;
}
