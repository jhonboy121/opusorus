//! Port of dnn/osce.c, osce.h, osce_config.h and osce_structs.h: the SILK speech enhancers
//! LACE and NoLACE, and the BBWENet bandwidth extension (`ENABLE_OSCE_BWE`, always on with the
//! `osce` feature as in upstream `--enable-osce`). The layer sizes and model bindings of the
//! generated `lace_data.h`, `nolace_data.h` and `bbwenet_data.h` live in private submodules and
//! are re-exported here.
//!
//! ## Entry points (for the SILK / Opus decoder integration)
//! * [`OsceModel`] + [`osce_load_models`]: upstream `silk_decoder.osce_model` and
//!   `silk_LoadOSCEModels` (the caller sets [`OsceModel::loaded`] to `result.is_ok()`). Weights
//!   always come from a blob (PLAN D-015): `osce_load_models(model, None)` fails like upstream
//!   `USE_WEIGHTS_FILE` builds, so `silk_InitDecoder` leaves the model unloaded.
//! * [`SilkOsceStruct`] (`silk_decoder_state.osce`), [`osce_reset`] (init_decoder.c,
//!   decode_frame.c on loss, dec_API.c on a method change) and [`osce_enhance_frame`]
//!   (decode_frame.c after the output-buffer update, `num_bits = ec_tell - ec_start`). The
//!   `silk_decoder_state` fields it reads are passed as an [`OsceDecInfo`] so that the OSCE
//!   state can live inside the decoder state without borrow conflicts.
//! * [`SilkOsceBweStruct`] (`silk_decoder_state.osce_bwe`), [`osce_bwe_reset`], [`osce_bwe`] and
//!   [`osce_bwe_cross_fade_10ms`](super::osce_features::osce_bwe_cross_fade_10ms) (dec_API.c).
//! * `silk_init_decoder` = both structs `Default` (C memset) followed by
//!   `osce_reset(.., OSCE_DEFAULT_METHOD)`; `silk_reset_decoder` only calls `osce_reset` (the
//!   OSCE fields precede `SILK_DECODER_STATE_RESET_START`).
//! * Like C, [`osce_bwe`] runs BBWENet even when the model is not loaded (all layers empty), so
//!   the Opus decoder should only select `OSCE_MODE_SILK_BBWE` with a loaded model.
//!
//! ## Deviations
//! * `OSCEState` is a C union of `LACEState` / `NoLACEState`; [`OsceState`] keeps the member
//!   of the current method on the heap, allocated when it first runs with a loaded model
//!   (until then it is in its reset state). The union is never observed through the wrong
//!   member: the method only changes in [`osce_reset`], which resets the matching member, so
//!   the other member is dropped there. Likewise [`OsceBweState`] allocates the BBWENet state
//!   when BWE first runs (until then it is the all-zero state of `silk_init_decoder`).
//! * `osce_load_models` returns an error for an unparsable blob (C passes a NULL list on).
//! * An invalid method in [`osce_enhance_frame`] (C: `celt_assert(0)` and an uninitialized
//!   output buffer) passes the input through.
//! * BBWENet's two large C stack buffers (`x_buffer1/2`, 2 x 34 KB) are stack arrays of the
//!   7.5 KB / 11.25 KB actually used.

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::sync::Arc;

use crate::celt::mathops::{celt_log, celt_sin, float2int};
use crate::math;
use crate::{Error, Result};

use super::nndsp::{
    AdaCombState, AdaConvState, AdaShapeState, adacomb_process_frame, adaconv_process_frame,
    adashape_process_frame, compute_overlap_window, init_adacomb_state, init_adaconv_state,
    init_adashape_state,
};
use super::nnet::{
    ACTIVATION_TANH, compute_generic_conv1d, compute_generic_dense, compute_generic_gru,
};
use super::osce_features::{
    OsceDecInfo, osce_bwe_calculate_features, osce_calculate_features, osce_cross_fade_10ms,
};
use super::parse_lpcnet_weights::{ModelCache, WeightArray, load_shared};
use crate::silk::structs::SilkDecoderControl;

mod bbwenet_data;
mod lace_data;
mod nolace_data;

pub use bbwenet_data::*;
pub use lace_data::*;
pub use nolace_data::*;

// ---- osce_config.h ----

/// `OSCE_FEATURES_MAX_HISTORY`.
pub const OSCE_FEATURES_MAX_HISTORY: usize = 350;
/// `OSCE_FEATURE_DIM`.
pub const OSCE_FEATURE_DIM: usize = 93;
/// `OSCE_MAX_FEATURE_FRAMES`.
pub const OSCE_MAX_FEATURE_FRAMES: usize = 4;
/// `OSCE_CLEAN_SPEC_NUM_BANDS`.
pub const OSCE_CLEAN_SPEC_NUM_BANDS: usize = 64;
/// `OSCE_NOISY_SPEC_NUM_BANDS`.
pub const OSCE_NOISY_SPEC_NUM_BANDS: usize = 18;
/// `OSCE_NO_PITCH_VALUE`.
pub const OSCE_NO_PITCH_VALUE: i32 = 7;
/// `OSCE_PREEMPH`.
pub const OSCE_PREEMPH: f32 = 0.85;
/// `OSCE_PITCH_HANGOVER`.
pub const OSCE_PITCH_HANGOVER: i32 = 0;
/// `OSCE_CLEAN_SPEC_START`.
pub const OSCE_CLEAN_SPEC_START: usize = 0;
/// `OSCE_CLEAN_SPEC_LENGTH`.
pub const OSCE_CLEAN_SPEC_LENGTH: usize = 64;
/// `OSCE_NOISY_CEPSTRUM_START`.
pub const OSCE_NOISY_CEPSTRUM_START: usize = 64;
/// `OSCE_NOISY_CEPSTRUM_LENGTH`.
pub const OSCE_NOISY_CEPSTRUM_LENGTH: usize = 18;
/// `OSCE_ACORR_START`.
pub const OSCE_ACORR_START: usize = 82;
/// `OSCE_ACORR_LENGTH`.
pub const OSCE_ACORR_LENGTH: usize = 5;
/// `OSCE_LTP_START`.
pub const OSCE_LTP_START: usize = 87;
/// `OSCE_LTP_LENGTH`.
pub const OSCE_LTP_LENGTH: usize = 5;
/// `OSCE_LOG_GAIN_START`.
pub const OSCE_LOG_GAIN_START: usize = 92;
/// `OSCE_LOG_GAIN_LENGTH`.
pub const OSCE_LOG_GAIN_LENGTH: usize = 1;
/// `OSCE_BWE_MAX_INSTAFREQ_BIN`.
pub const OSCE_BWE_MAX_INSTAFREQ_BIN: usize = 40;
/// `OSCE_BWE_HALF_WINDOW_SIZE`.
pub const OSCE_BWE_HALF_WINDOW_SIZE: usize = 160;
/// `OSCE_BWE_WINDOW_SIZE`.
pub const OSCE_BWE_WINDOW_SIZE: usize = 2 * OSCE_BWE_HALF_WINDOW_SIZE;
/// `OSCE_BWE_NUM_BANDS`.
pub const OSCE_BWE_NUM_BANDS: usize = 32;
/// `OSCE_BWE_FEATURE_DIM`.
pub const OSCE_BWE_FEATURE_DIM: usize = 114;
/// `OSCE_BWE_OUTPUT_DELAY`.
pub const OSCE_BWE_OUTPUT_DELAY: usize = 21;

// ---- osce.h ----

/// `OSCE_MODE_SILK_ONLY`.
pub const OSCE_MODE_SILK_ONLY: i32 = 1000;
/// `OSCE_MODE_HYBRID`.
pub const OSCE_MODE_HYBRID: i32 = 1001;
/// `OSCE_MODE_CELT_ONLY`.
pub const OSCE_MODE_CELT_ONLY: i32 = 1002;
/// `OSCE_MODE_SILK_BBWE`.
pub const OSCE_MODE_SILK_BBWE: i32 = 1003;

/// `OSCE_METHOD_NONE`.
pub const OSCE_METHOD_NONE: i32 = 0;
/// `OSCE_METHOD_LACE`.
pub const OSCE_METHOD_LACE: i32 = 1;
/// `OSCE_METHOD_NOLACE`.
pub const OSCE_METHOD_NOLACE: i32 = 2;
/// `OSCE_DEFAULT_METHOD` (NoLACE: neither `DISABLE_LACE` nor `DISABLE_NOLACE` is defined).
pub const OSCE_DEFAULT_METHOD: i32 = OSCE_METHOD_NOLACE;
/// `OSCE_MAX_RNN_NEURONS`.
pub const OSCE_MAX_RNN_NEURONS: usize = NOLACE_FNET_GRU_STATE_SIZE;
/// `OSCE_BWE_MAX_RNN_NEURONS`.
pub const OSCE_BWE_MAX_RNN_NEURONS: usize = BBWENET_FNET_GRU_STATE_SIZE;

// ---- osce_structs.h ----

/// `OSCEFeatureState`.
#[derive(Debug, Clone, PartialEq)]
pub struct OsceFeatureState {
    pub numbits_smooth: f32,
    pub pitch_hangover_count: i32,
    pub last_lag: i32,
    pub last_type: i32,
    pub signal_history: [f32; OSCE_FEATURES_MAX_HISTORY],
    pub reset: i32,
}

impl Default for OsceFeatureState {
    fn default() -> Self {
        Self {
            numbits_smooth: 0.0,
            pitch_hangover_count: 0,
            last_lag: 0,
            last_type: 0,
            signal_history: [0.0; OSCE_FEATURES_MAX_HISTORY],
            reset: 0,
        }
    }
}

/// `OSCEBWEFeatureState`.
#[derive(Debug, Clone, PartialEq)]
pub struct OsceBweFeatureState {
    pub signal_history: [f32; OSCE_BWE_HALF_WINDOW_SIZE],
    pub last_spec: [f32; 2 * OSCE_BWE_MAX_INSTAFREQ_BIN + 2],
}

impl Default for OsceBweFeatureState {
    fn default() -> Self {
        Self {
            signal_history: [0.0; OSCE_BWE_HALF_WINDOW_SIZE],
            last_spec: [0.0; 2 * OSCE_BWE_MAX_INSTAFREQ_BIN + 2],
        }
    }
}

/// `resamp_state` (BBWENet 2x upsampler / 3/2 interpolator memory of one channel).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ResampState {
    pub upsamp_buffer: [[f32; 3]; 2],
    pub interpol_buffer: [f32; 8],
}

/// Floats of BBWENet scratch `x_buffer1` actually used (af2 output: 4 subframes x 3 channels x
/// 160 samples).
const BBWENET_XBUF1_SIZE: usize = 4 * BBWENET_AF2_OUT_CHANNELS * BBWENET_AF2_FRAME_SIZE;
/// Floats of BBWENet scratch `x_buffer2` actually used (interpolator output: 4 subframes x 3
/// channels x 240 samples).
const BBWENET_XBUF2_SIZE: usize = 4 * BBWENET_AF2_OUT_CHANNELS * BBWENET_TDSHAPE2_FRAME_SIZE;

/// `BBWENetState`.
#[derive(Debug, Clone, PartialEq)]
pub struct BbwenetState {
    pub feature_net_conv1_state: [f32; BBWENET_FNET_CONV1_STATE_SIZE],
    pub feature_net_conv2_state: [f32; BBWENET_FNET_CONV2_STATE_SIZE],
    pub feature_net_gru_state: [f32; BBWENET_FNET_GRU_STATE_SIZE],
    /// C `outbut_buffer` (sic).
    pub outbut_buffer: [i16; OSCE_BWE_OUTPUT_DELAY],
    pub af1_state: AdaConvState,
    pub af2_state: AdaConvState,
    pub af3_state: AdaConvState,
    pub tdshape1_state: AdaShapeState,
    pub tdshape2_state: AdaShapeState,
    pub resampler_state: [ResampState; 3],
}

impl Default for BbwenetState {
    fn default() -> Self {
        Self {
            feature_net_conv1_state: [0.0; BBWENET_FNET_CONV1_STATE_SIZE],
            feature_net_conv2_state: [0.0; BBWENET_FNET_CONV2_STATE_SIZE],
            feature_net_gru_state: [0.0; BBWENET_FNET_GRU_STATE_SIZE],
            outbut_buffer: [0; OSCE_BWE_OUTPUT_DELAY],
            af1_state: AdaConvState::default(),
            af2_state: AdaConvState::default(),
            af3_state: AdaConvState::default(),
            tdshape1_state: AdaShapeState::default(),
            tdshape2_state: AdaShapeState::default(),
            resampler_state: [ResampState::default(); 3],
        }
    }
}

impl BbwenetState {
    /// `OPUS_CLEAR(state, 1)`: zeroes every state field.
    fn clear(&mut self) {
        self.feature_net_conv1_state.fill(0.0);
        self.feature_net_conv2_state.fill(0.0);
        self.feature_net_gru_state.fill(0.0);
        self.outbut_buffer.fill(0);
        self.af1_state = AdaConvState::default();
        self.af2_state = AdaConvState::default();
        self.af3_state = AdaConvState::default();
        self.tdshape1_state = AdaShapeState::default();
        self.tdshape2_state = AdaShapeState::default();
        self.resampler_state = [ResampState::default(); 3];
    }
}

/// `BBWENet`: the model layers plus the overlap windows.
#[derive(Debug, Clone, PartialEq)]
pub struct Bbwenet {
    pub layers: BbwenetLayers,
    pub window16: [f32; BBWENET_AF1_OVERLAP_SIZE],
    pub window32: [f32; BBWENET_AF2_OVERLAP_SIZE],
    pub window48: [f32; BBWENET_AF3_OVERLAP_SIZE],
}

impl Default for Bbwenet {
    fn default() -> Self {
        Self {
            layers: BbwenetLayers::default(),
            window16: [0.0; BBWENET_AF1_OVERLAP_SIZE],
            window32: [0.0; BBWENET_AF2_OVERLAP_SIZE],
            window48: [0.0; BBWENET_AF3_OVERLAP_SIZE],
        }
    }
}

/// `LACEState`.
#[derive(Debug, Clone, PartialEq)]
pub struct LaceState {
    pub feature_net_conv2_state: [f32; LACE_FNET_CONV2_STATE_SIZE],
    pub feature_net_gru_state: [f32; LACE_COND_DIM],
    pub cf1_state: AdaCombState,
    pub cf2_state: AdaCombState,
    pub af1_state: AdaConvState,
    pub preemph_mem: f32,
    pub deemph_mem: f32,
}

impl Default for LaceState {
    fn default() -> Self {
        Self {
            feature_net_conv2_state: [0.0; LACE_FNET_CONV2_STATE_SIZE],
            feature_net_gru_state: [0.0; LACE_COND_DIM],
            cf1_state: AdaCombState::default(),
            cf2_state: AdaCombState::default(),
            af1_state: AdaConvState::default(),
            preemph_mem: 0.0,
            deemph_mem: 0.0,
        }
    }
}

/// `LACE`: the model layers plus the overlap window.
#[derive(Debug, Clone, PartialEq)]
pub struct Lace {
    pub layers: LaceLayers,
    pub window: [f32; LACE_OVERLAP_SIZE],
}

impl Default for Lace {
    fn default() -> Self {
        Self {
            layers: LaceLayers::default(),
            window: [0.0; LACE_OVERLAP_SIZE],
        }
    }
}

/// `NoLACEState`.
#[derive(Debug, Clone, PartialEq)]
pub struct NoLaceState {
    pub feature_net_conv2_state: [f32; NOLACE_FNET_CONV2_STATE_SIZE],
    pub feature_net_gru_state: [f32; NOLACE_COND_DIM],
    pub post_cf1_state: [f32; NOLACE_COND_DIM],
    pub post_cf2_state: [f32; NOLACE_COND_DIM],
    pub post_af1_state: [f32; NOLACE_COND_DIM],
    pub post_af2_state: [f32; NOLACE_COND_DIM],
    pub post_af3_state: [f32; NOLACE_COND_DIM],
    pub cf1_state: AdaCombState,
    pub cf2_state: AdaCombState,
    pub af1_state: AdaConvState,
    pub af2_state: AdaConvState,
    pub af3_state: AdaConvState,
    pub af4_state: AdaConvState,
    pub tdshape1_state: AdaShapeState,
    pub tdshape2_state: AdaShapeState,
    pub tdshape3_state: AdaShapeState,
    pub preemph_mem: f32,
    pub deemph_mem: f32,
}

impl Default for NoLaceState {
    fn default() -> Self {
        Self {
            feature_net_conv2_state: [0.0; NOLACE_FNET_CONV2_STATE_SIZE],
            feature_net_gru_state: [0.0; NOLACE_COND_DIM],
            post_cf1_state: [0.0; NOLACE_COND_DIM],
            post_cf2_state: [0.0; NOLACE_COND_DIM],
            post_af1_state: [0.0; NOLACE_COND_DIM],
            post_af2_state: [0.0; NOLACE_COND_DIM],
            post_af3_state: [0.0; NOLACE_COND_DIM],
            cf1_state: AdaCombState::default(),
            cf2_state: AdaCombState::default(),
            af1_state: AdaConvState::default(),
            af2_state: AdaConvState::default(),
            af3_state: AdaConvState::default(),
            af4_state: AdaConvState::default(),
            tdshape1_state: AdaShapeState::default(),
            tdshape2_state: AdaShapeState::default(),
            tdshape3_state: AdaShapeState::default(),
            preemph_mem: 0.0,
            deemph_mem: 0.0,
        }
    }
}

/// `NoLACE`: the model layers plus the overlap window (C sizes it `LACE_OVERLAP_SIZE`).
#[derive(Debug, Clone, PartialEq)]
pub struct NoLace {
    pub layers: NoLaceLayers,
    pub window: [f32; LACE_OVERLAP_SIZE],
}

impl Default for NoLace {
    fn default() -> Self {
        Self {
            layers: NoLaceLayers::default(),
            window: [0.0; LACE_OVERLAP_SIZE],
        }
    }
}

/// The models of the embedded weight blob (see [`load_shared`]).
static LACE_CACHE: ModelCache<Lace> = ModelCache::new();
static NOLACE_CACHE: ModelCache<NoLace> = ModelCache::new();
static BBWENET_CACHE: ModelCache<Bbwenet> = ModelCache::new();

/// `OSCEModel`. The models are allocated when they are loaded and shared ([`Arc`]) by the
/// decoders they were loaded into; before that they read as unbound (empty) models, like C's
/// zeroed struct.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OsceModel {
    /// C `int loaded`; set by the caller (`silk_LoadOSCEModels`) from the load result.
    pub loaded: bool,
    lace: Option<Arc<Lace>>,
    nolace: Option<Arc<NoLace>>,
    bbwenet: Option<Arc<Bbwenet>>,
}

impl OsceModel {
    /// `lace` (an unbound model before it is loaded).
    #[must_use]
    pub fn lace(&self) -> Cow<'_, Lace> {
        match &self.lace {
            Some(m) => Cow::Borrowed(&**m),
            None => Cow::Owned(Lace::default()),
        }
    }

    /// `nolace` (an unbound model before it is loaded).
    #[must_use]
    pub fn nolace(&self) -> Cow<'_, NoLace> {
        match &self.nolace {
            Some(m) => Cow::Borrowed(&**m),
            None => Cow::Owned(NoLace::default()),
        }
    }

    /// `bbwenet` (an unbound model before it is loaded).
    #[must_use]
    pub fn bbwenet(&self) -> Cow<'_, Bbwenet> {
        match &self.bbwenet {
            Some(m) => Cow::Borrowed(&**m),
            None => Cow::Owned(Bbwenet::default()),
        }
    }
}

/// `OSCEState`: the C union of the LACE and NoLACE states (see the module deviations).
///
/// Each member is allocated when first used (`None` is its reset state, the only state it can
/// have before it runs), and dropped when [`osce_reset`] selects another method. The accessors
/// give the state of a member as C would see it after `osce_reset` selected its method.
#[derive(Debug, Clone, Default)]
pub struct OsceState {
    lace: Option<Box<LaceState>>,
    nolace: Option<Box<NoLaceState>>,
}

impl OsceState {
    /// The LACE state (its reset state when not allocated).
    #[must_use]
    pub fn lace(&self) -> Cow<'_, LaceState> {
        match &self.lace {
            Some(b) => Cow::Borrowed(&**b),
            None => Cow::Owned(new_lace_state()),
        }
    }

    /// The LACE state, allocated in its reset state if needed.
    pub fn lace_mut(&mut self) -> &mut LaceState {
        self.lace.get_or_insert_with(|| Box::new(new_lace_state()))
    }

    /// The NoLACE state (its reset state when not allocated).
    #[must_use]
    pub fn nolace(&self) -> Cow<'_, NoLaceState> {
        match &self.nolace {
            Some(b) => Cow::Borrowed(&**b),
            None => Cow::Owned(new_nolace_state()),
        }
    }

    /// The NoLACE state, allocated in its reset state if needed.
    pub fn nolace_mut(&mut self) -> &mut NoLaceState {
        self.nolace
            .get_or_insert_with(|| Box::new(new_nolace_state()))
    }

    /// Heap bytes of the state when a method runs (the larger member of the C union).
    #[must_use]
    pub const fn max_heap_size() -> usize {
        if size_of::<LaceState>() > size_of::<NoLaceState>() {
            size_of::<LaceState>()
        } else {
            size_of::<NoLaceState>()
        }
    }
}

impl PartialEq for OsceState {
    fn eq(&self, other: &Self) -> bool {
        self.lace() == other.lace() && self.nolace() == other.nolace()
    }
}

/// `OSCEBWEState`. The BBWENet state is allocated when first used (`None` is the all-zero
/// state of `silk_init_decoder`'s memset).
#[derive(Debug, Clone, Default)]
pub struct OsceBweState {
    bbwenet: Option<Box<BbwenetState>>,
}

impl OsceBweState {
    /// The BBWENet state (all zero when not allocated).
    #[must_use]
    pub fn bbwenet(&self) -> Cow<'_, BbwenetState> {
        match &self.bbwenet {
            Some(b) => Cow::Borrowed(&**b),
            None => Cow::Owned(BbwenetState::default()),
        }
    }

    /// The BBWENet state, allocated (all zero) if needed.
    pub fn bbwenet_mut(&mut self) -> &mut BbwenetState {
        self.bbwenet.get_or_insert_with(Box::default)
    }

    /// Heap bytes of the state once BWE ran.
    #[must_use]
    pub const fn max_heap_size() -> usize {
        size_of::<BbwenetState>()
    }
}

impl PartialEq for OsceBweState {
    fn eq(&self, other: &Self) -> bool {
        self.bbwenet() == other.bbwenet()
    }
}

/// `silk_OSCE_struct` (silk/structs.h): the per-channel enhancer state of the SILK decoder.
/// `Default` is the all-zero state of `silk_init_decoder`'s memset.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SilkOsceStruct {
    pub features: OsceFeatureState,
    pub state: OsceState,
    pub method: i32,
}

/// `silk_OSCE_BWE_struct` (silk/structs.h): the per-channel BWE state of the SILK decoder.
/// `Default` is the all-zero state of `silk_init_decoder`'s memset.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SilkOsceBweStruct {
    pub features: OsceBweFeatureState,
    pub state: OsceBweState,
}

/// C `CLIP(a, min, max)`: `(((a) < (min) ? (min) : (a)) > (max) ? (max) : (a))`. Note that
/// it returns `a` (not `min`) when `a < min <= max`, which the port keeps.
#[inline]
fn clip(a: f32, min: f32, max: f32) -> f32 {
    let t = if a < min { min } else { a };
    if t > max { max } else { a }
}

// ---- LACE ----

/// Port of dnn/osce.c:compute_lace_numbits_embedding (static). `emb` receives 8 values.
pub fn compute_lace_numbits_embedding(
    emb: &mut [f32],
    numbits: f32,
    _dim: usize,
    min_val: f32,
    max_val: f32,
    logscale: bool,
) {
    // C: numbits = logscale ? log(numbits) : numbits (double ternary, stored to float).
    let numbits = if logscale {
        math::log(numbits as f64) as f32
    } else {
        numbits
    };
    let x = clip(numbits, min_val, max_val) - (max_val + min_val) / 2.0;
    let scales = [
        LACE_NUMBITS_SCALE_0,
        LACE_NUMBITS_SCALE_1,
        LACE_NUMBITS_SCALE_2,
        LACE_NUMBITS_SCALE_3,
        LACE_NUMBITS_SCALE_4,
        LACE_NUMBITS_SCALE_5,
        LACE_NUMBITS_SCALE_6,
        LACE_NUMBITS_SCALE_7,
    ];
    for (e, &s) in emb[..8].iter_mut().zip(&scales) {
        *e = math::sin((x * s - 0.5f32) as f64) as f32;
    }
}

/// Port of dnn/osce.c:init_lace (static): binds the layers and computes the window.
pub fn init_lace(weights: &[WeightArray<'_>]) -> Result<Lace> {
    let mut h_lace = Lace {
        layers: init_lacelayers(weights)?,
        window: [0.0; LACE_OVERLAP_SIZE],
    };
    compute_overlap_window(&mut h_lace.window, LACE_OVERLAP_SIZE);
    Ok(h_lace)
}

/// Port of dnn/osce.c:reset_lace_state (static).
pub fn reset_lace_state(state: &mut LaceState) {
    *state = new_lace_state();
}

/// A LACE state after [`reset_lace_state`].
fn new_lace_state() -> LaceState {
    let mut state = LaceState::default();
    init_adacomb_state(&mut state.cf1_state);
    init_adacomb_state(&mut state.cf2_state);
    init_adaconv_state(&mut state.af1_state);
    state
}

/// Port of dnn/osce.c:lace_feature_net (static). `output` receives `4*LACE_COND_DIM` floats,
/// `features` holds `4*LACE_NUM_FEATURES`.
pub fn lace_feature_net(
    h_lace: &Lace,
    state: &mut LaceState,
    output: &mut [f32],
    features: &[f32],
    numbits: &[f32; 2],
    periods: &[i32; 4],
) {
    const IN_SIZE: usize =
        LACE_NUM_FEATURES + LACE_PITCH_EMBEDDING_DIM + 2 * LACE_NUMBITS_EMBEDDING_DIM;
    const HID: usize = if LACE_COND_DIM > LACE_HIDDEN_FEATURE_DIM {
        LACE_COND_DIM
    } else {
        LACE_HIDDEN_FEATURE_DIM
    };
    const BUF: usize = if 4 * HID > IN_SIZE { 4 * HID } else { IN_SIZE };
    let mut input_buffer = [0f32; BUF];
    let mut output_buffer = [0f32; 4 * HID];
    let mut numbits_embedded = [0f32; 2 * LACE_NUMBITS_EMBEDDING_DIM];
    let layers = &h_lace.layers;

    let lo = math::log(LACE_NUMBITS_RANGE_LOW as f64) as f32;
    let hi = math::log(LACE_NUMBITS_RANGE_HIGH as f64) as f32;
    compute_lace_numbits_embedding(
        &mut numbits_embedded[..LACE_NUMBITS_EMBEDDING_DIM],
        numbits[0],
        LACE_NUMBITS_EMBEDDING_DIM,
        lo,
        hi,
        true,
    );
    compute_lace_numbits_embedding(
        &mut numbits_embedded[LACE_NUMBITS_EMBEDDING_DIM..],
        numbits[1],
        LACE_NUMBITS_EMBEDDING_DIM,
        lo,
        hi,
        true,
    );

    // Scaling and dimensionality reduction.
    let pitch_emb = pitch_embedding_weights(&layers.lace_pitch_embedding);
    for i_subframe in 0..4 {
        input_buffer[..LACE_NUM_FEATURES].copy_from_slice(
            &features[i_subframe * LACE_NUM_FEATURES..(i_subframe + 1) * LACE_NUM_FEATURES],
        );
        let p = periods[i_subframe] as usize * LACE_PITCH_EMBEDDING_DIM;
        input_buffer[LACE_NUM_FEATURES..LACE_NUM_FEATURES + LACE_PITCH_EMBEDDING_DIM]
            .copy_from_slice(&pitch_emb[p..p + LACE_PITCH_EMBEDDING_DIM]);
        input_buffer[LACE_NUM_FEATURES + LACE_PITCH_EMBEDDING_DIM..IN_SIZE]
            .copy_from_slice(&numbits_embedded);

        compute_generic_conv1d(
            &layers.lace_fnet_conv1,
            &mut output_buffer[i_subframe * LACE_HIDDEN_FEATURE_DIM..],
            &mut [],
            &input_buffer,
            IN_SIZE,
            ACTIVATION_TANH,
        );
    }

    // Subframe accumulation.
    input_buffer[..4 * LACE_HIDDEN_FEATURE_DIM]
        .copy_from_slice(&output_buffer[..4 * LACE_HIDDEN_FEATURE_DIM]);
    compute_generic_conv1d(
        &layers.lace_fnet_conv2,
        &mut output_buffer,
        &mut state.feature_net_conv2_state,
        &input_buffer,
        4 * LACE_HIDDEN_FEATURE_DIM,
        ACTIVATION_TANH,
    );

    // tconv upsampling.
    input_buffer[..4 * LACE_COND_DIM].copy_from_slice(&output_buffer[..4 * LACE_COND_DIM]);
    compute_generic_dense(
        &layers.lace_fnet_tconv,
        &mut output_buffer,
        &input_buffer,
        ACTIVATION_TANH,
    );

    // GRU.
    input_buffer[..4 * LACE_COND_DIM].copy_from_slice(&output_buffer[..4 * LACE_COND_DIM]);
    for i_subframe in 0..4 {
        compute_generic_gru(
            &layers.lace_fnet_gru_input,
            &layers.lace_fnet_gru_recurrent,
            &mut state.feature_net_gru_state,
            &input_buffer[i_subframe * LACE_COND_DIM..],
        );
        output[i_subframe * LACE_COND_DIM..(i_subframe + 1) * LACE_COND_DIM]
            .copy_from_slice(&state.feature_net_gru_state);
    }
}

/// The float weights of a pitch-embedding layer (C reads `float_weights` directly; an unbound
/// layer only happens for an unloaded model, which the callers never run).
fn pitch_embedding_weights(layer: &super::nnet::LinearLayer) -> &[f32] {
    match layer.float_weights.as_deref() {
        Some(w) => w,
        None => &[],
    }
}

/// Port of dnn/osce.c:lace_process_20ms_frame (static): enhances 320 samples (4 subframes).
pub fn lace_process_20ms_frame(
    h_lace: &Lace,
    state: &mut LaceState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    numbits: &[f32; 2],
    periods: &[i32; 4],
) {
    let mut feature_buffer = [0f32; 4 * LACE_COND_DIM];
    let mut output_buffer = [0f32; 4 * LACE_FRAME_SIZE];
    let layers = &h_lace.layers;

    // Pre-emphasis.
    for (o, &x) in output_buffer.iter_mut().zip(&x_in[..4 * LACE_FRAME_SIZE]) {
        *o = x - LACE_PREEMPH * state.preemph_mem;
        state.preemph_mem = x;
    }

    // Run feature encoder.
    lace_feature_net(
        h_lace,
        state,
        &mut feature_buffer,
        features,
        numbits,
        periods,
    );

    // 1st comb filtering stage.
    for i_subframe in 0..4 {
        adacomb_process_frame(
            &mut state.cf1_state,
            &mut output_buffer[i_subframe * LACE_FRAME_SIZE..(i_subframe + 1) * LACE_FRAME_SIZE],
            None,
            &feature_buffer[i_subframe * LACE_COND_DIM..],
            &layers.lace_cf1_kernel,
            &layers.lace_cf1_gain,
            &layers.lace_cf1_global_gain,
            periods[i_subframe],
            LACE_COND_DIM,
            LACE_FRAME_SIZE,
            LACE_OVERLAP_SIZE,
            LACE_CF1_KERNEL_SIZE,
            LACE_CF1_LEFT_PADDING,
            LACE_CF1_FILTER_GAIN_A,
            LACE_CF1_FILTER_GAIN_B,
            LACE_CF1_LOG_GAIN_LIMIT,
            &h_lace.window,
        );
    }

    // 2nd comb filtering stage.
    for i_subframe in 0..4 {
        adacomb_process_frame(
            &mut state.cf2_state,
            &mut output_buffer[i_subframe * LACE_FRAME_SIZE..(i_subframe + 1) * LACE_FRAME_SIZE],
            None,
            &feature_buffer[i_subframe * LACE_COND_DIM..],
            &layers.lace_cf2_kernel,
            &layers.lace_cf2_gain,
            &layers.lace_cf2_global_gain,
            periods[i_subframe],
            LACE_COND_DIM,
            LACE_FRAME_SIZE,
            LACE_OVERLAP_SIZE,
            LACE_CF2_KERNEL_SIZE,
            LACE_CF2_LEFT_PADDING,
            LACE_CF2_FILTER_GAIN_A,
            LACE_CF2_FILTER_GAIN_B,
            LACE_CF2_LOG_GAIN_LIMIT,
            &h_lace.window,
        );
    }

    // Final adaptive filtering stage.
    for i_subframe in 0..4 {
        adaconv_process_frame(
            &mut state.af1_state,
            &mut output_buffer[i_subframe * LACE_FRAME_SIZE..(i_subframe + 1) * LACE_FRAME_SIZE],
            None,
            &feature_buffer[i_subframe * LACE_COND_DIM..],
            &layers.lace_af1_kernel,
            &layers.lace_af1_gain,
            LACE_COND_DIM,
            LACE_FRAME_SIZE,
            LACE_OVERLAP_SIZE,
            LACE_AF1_IN_CHANNELS,
            LACE_AF1_OUT_CHANNELS,
            LACE_AF1_KERNEL_SIZE,
            LACE_AF1_LEFT_PADDING,
            LACE_AF1_FILTER_GAIN_A,
            LACE_AF1_FILTER_GAIN_B,
            LACE_AF1_SHAPE_GAIN,
            &h_lace.window,
        );
    }

    // De-emphasis.
    for (o, &x) in x_out[..4 * LACE_FRAME_SIZE].iter_mut().zip(&output_buffer) {
        *o = x + LACE_PREEMPH * state.deemph_mem;
        state.deemph_mem = *o;
    }
}

// ---- NoLACE ----

/// Port of dnn/osce.c:compute_nolace_numbits_embedding (static). `emb` receives 8 values.
pub fn compute_nolace_numbits_embedding(
    emb: &mut [f32],
    numbits: f32,
    _dim: usize,
    min_val: f32,
    max_val: f32,
    logscale: bool,
) {
    let numbits = if logscale {
        math::log(numbits as f64) as f32
    } else {
        numbits
    };
    let x = clip(numbits, min_val, max_val) - (max_val + min_val) / 2.0;
    let scales = [
        NOLACE_NUMBITS_SCALE_0,
        NOLACE_NUMBITS_SCALE_1,
        NOLACE_NUMBITS_SCALE_2,
        NOLACE_NUMBITS_SCALE_3,
        NOLACE_NUMBITS_SCALE_4,
        NOLACE_NUMBITS_SCALE_5,
        NOLACE_NUMBITS_SCALE_6,
        NOLACE_NUMBITS_SCALE_7,
    ];
    for (e, &s) in emb[..8].iter_mut().zip(&scales) {
        *e = math::sin((x * s - 0.5f32) as f64) as f32;
    }
}

/// Port of dnn/osce.c:init_nolace (static): binds the layers and computes the window.
pub fn init_nolace(weights: &[WeightArray<'_>]) -> Result<NoLace> {
    let mut h_nolace = NoLace {
        layers: init_nolacelayers(weights)?,
        window: [0.0; LACE_OVERLAP_SIZE],
    };
    compute_overlap_window(&mut h_nolace.window, NOLACE_OVERLAP_SIZE);
    Ok(h_nolace)
}

/// Port of dnn/osce.c:reset_nolace_state (static).
pub fn reset_nolace_state(state: &mut NoLaceState) {
    *state = new_nolace_state();
}

/// A NoLACE state after [`reset_nolace_state`].
fn new_nolace_state() -> NoLaceState {
    let mut state = NoLaceState::default();
    init_adacomb_state(&mut state.cf1_state);
    init_adacomb_state(&mut state.cf2_state);
    init_adaconv_state(&mut state.af1_state);
    init_adaconv_state(&mut state.af2_state);
    init_adaconv_state(&mut state.af3_state);
    init_adaconv_state(&mut state.af4_state);
    init_adashape_state(&mut state.tdshape1_state);
    init_adashape_state(&mut state.tdshape2_state);
    init_adashape_state(&mut state.tdshape3_state);
    state
}

/// Port of dnn/osce.c:nolace_feature_net (static). `output` receives `4*NOLACE_COND_DIM`
/// floats, `features` holds `4*NOLACE_NUM_FEATURES`.
pub fn nolace_feature_net(
    h_nolace: &NoLace,
    state: &mut NoLaceState,
    output: &mut [f32],
    features: &[f32],
    numbits: &[f32; 2],
    periods: &[i32; 4],
) {
    const IN_SIZE: usize =
        NOLACE_NUM_FEATURES + NOLACE_PITCH_EMBEDDING_DIM + 2 * NOLACE_NUMBITS_EMBEDDING_DIM;
    const HID: usize = if NOLACE_COND_DIM > NOLACE_HIDDEN_FEATURE_DIM {
        NOLACE_COND_DIM
    } else {
        NOLACE_HIDDEN_FEATURE_DIM
    };
    let mut input_buffer = [0f32; 4 * HID];
    let mut output_buffer = [0f32; 4 * HID];
    let mut numbits_embedded = [0f32; 2 * NOLACE_NUMBITS_EMBEDDING_DIM];
    let layers = &h_nolace.layers;

    let lo = math::log(NOLACE_NUMBITS_RANGE_LOW as f64) as f32;
    let hi = math::log(NOLACE_NUMBITS_RANGE_HIGH as f64) as f32;
    compute_nolace_numbits_embedding(
        &mut numbits_embedded[..NOLACE_NUMBITS_EMBEDDING_DIM],
        numbits[0],
        NOLACE_NUMBITS_EMBEDDING_DIM,
        lo,
        hi,
        true,
    );
    compute_nolace_numbits_embedding(
        &mut numbits_embedded[NOLACE_NUMBITS_EMBEDDING_DIM..],
        numbits[1],
        NOLACE_NUMBITS_EMBEDDING_DIM,
        lo,
        hi,
        true,
    );

    // Scaling and dimensionality reduction.
    let pitch_emb = pitch_embedding_weights(&layers.nolace_pitch_embedding);
    for i_subframe in 0..4 {
        input_buffer[..NOLACE_NUM_FEATURES].copy_from_slice(
            &features[i_subframe * NOLACE_NUM_FEATURES..(i_subframe + 1) * NOLACE_NUM_FEATURES],
        );
        let p = periods[i_subframe] as usize * NOLACE_PITCH_EMBEDDING_DIM;
        input_buffer[NOLACE_NUM_FEATURES..NOLACE_NUM_FEATURES + NOLACE_PITCH_EMBEDDING_DIM]
            .copy_from_slice(&pitch_emb[p..p + NOLACE_PITCH_EMBEDDING_DIM]);
        input_buffer[NOLACE_NUM_FEATURES + NOLACE_PITCH_EMBEDDING_DIM..IN_SIZE]
            .copy_from_slice(&numbits_embedded);

        compute_generic_conv1d(
            &layers.nolace_fnet_conv1,
            &mut output_buffer[i_subframe * NOLACE_HIDDEN_FEATURE_DIM..],
            &mut [],
            &input_buffer,
            IN_SIZE,
            ACTIVATION_TANH,
        );
    }

    // Subframe accumulation.
    input_buffer[..4 * NOLACE_HIDDEN_FEATURE_DIM]
        .copy_from_slice(&output_buffer[..4 * NOLACE_HIDDEN_FEATURE_DIM]);
    compute_generic_conv1d(
        &layers.nolace_fnet_conv2,
        &mut output_buffer,
        &mut state.feature_net_conv2_state,
        &input_buffer,
        4 * NOLACE_HIDDEN_FEATURE_DIM,
        ACTIVATION_TANH,
    );

    // tconv upsampling.
    input_buffer[..4 * NOLACE_COND_DIM].copy_from_slice(&output_buffer[..4 * NOLACE_COND_DIM]);
    compute_generic_dense(
        &layers.nolace_fnet_tconv,
        &mut output_buffer,
        &input_buffer,
        ACTIVATION_TANH,
    );

    // GRU.
    input_buffer[..4 * NOLACE_COND_DIM].copy_from_slice(&output_buffer[..4 * NOLACE_COND_DIM]);
    for i_subframe in 0..4 {
        compute_generic_gru(
            &layers.nolace_fnet_gru_input,
            &layers.nolace_fnet_gru_recurrent,
            &mut state.feature_net_gru_state,
            &input_buffer[i_subframe * NOLACE_COND_DIM..],
        );
        output[i_subframe * NOLACE_COND_DIM..(i_subframe + 1) * NOLACE_COND_DIM]
            .copy_from_slice(&state.feature_net_gru_state);
    }
}

/// Port of dnn/osce.c:nolace_process_20ms_frame (static): enhances 320 samples (4 subframes).
pub fn nolace_process_20ms_frame(
    h_nolace: &NoLace,
    state: &mut NoLaceState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    numbits: &[f32; 2],
    periods: &[i32; 4],
) {
    const FS: usize = NOLACE_FRAME_SIZE;
    const CD: usize = NOLACE_COND_DIM;
    let mut feature_buffer = [0f32; 4 * CD];
    let mut feature_transform_buffer = [0f32; 4 * CD];
    let mut x_buffer1 = [0f32; 8 * FS];
    let mut x_buffer2 = [0f32; 8 * FS];
    let layers = &h_nolace.layers;

    // Pre-emphasis.
    for (o, &x) in x_buffer1[..4 * FS].iter_mut().zip(&x_in[..4 * FS]) {
        *o = x - NOLACE_PREEMPH * state.preemph_mem;
        state.preemph_mem = x;
    }

    // Run feature encoder.
    nolace_feature_net(
        h_nolace,
        state,
        &mut feature_buffer,
        features,
        numbits,
        periods,
    );

    // 1st comb filtering stage.
    for i_subframe in 0..4 {
        // Modifies signal in place.
        adacomb_process_frame(
            &mut state.cf1_state,
            &mut x_buffer1[i_subframe * FS..(i_subframe + 1) * FS],
            None,
            &feature_buffer[i_subframe * CD..],
            &layers.nolace_cf1_kernel,
            &layers.nolace_cf1_gain,
            &layers.nolace_cf1_global_gain,
            periods[i_subframe],
            NOLACE_COND_DIM,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            NOLACE_CF1_KERNEL_SIZE,
            NOLACE_CF1_LEFT_PADDING,
            NOLACE_CF1_FILTER_GAIN_A,
            NOLACE_CF1_FILTER_GAIN_B,
            NOLACE_CF1_LOG_GAIN_LIMIT,
            &h_nolace.window,
        );

        compute_generic_conv1d(
            &layers.nolace_post_cf1,
            &mut feature_transform_buffer[i_subframe * CD..],
            &mut state.post_cf1_state,
            &feature_buffer[i_subframe * CD..],
            NOLACE_COND_DIM,
            ACTIVATION_TANH,
        );
    }

    // Update feature buffer.
    feature_buffer.copy_from_slice(&feature_transform_buffer);

    // 2nd comb filtering stage.
    for i_subframe in 0..4 {
        // Modifies signal in place.
        adacomb_process_frame(
            &mut state.cf2_state,
            &mut x_buffer1[i_subframe * FS..(i_subframe + 1) * FS],
            None,
            &feature_buffer[i_subframe * CD..],
            &layers.nolace_cf2_kernel,
            &layers.nolace_cf2_gain,
            &layers.nolace_cf2_global_gain,
            periods[i_subframe],
            NOLACE_COND_DIM,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            NOLACE_CF2_KERNEL_SIZE,
            NOLACE_CF2_LEFT_PADDING,
            NOLACE_CF2_FILTER_GAIN_A,
            NOLACE_CF2_FILTER_GAIN_B,
            NOLACE_CF2_LOG_GAIN_LIMIT,
            &h_nolace.window,
        );

        compute_generic_conv1d(
            &layers.nolace_post_cf2,
            &mut feature_transform_buffer[i_subframe * CD..],
            &mut state.post_cf2_state,
            &feature_buffer[i_subframe * CD..],
            NOLACE_COND_DIM,
            ACTIVATION_TANH,
        );
    }

    // Update feature buffer.
    feature_buffer.copy_from_slice(&feature_transform_buffer);

    // Final adaptive filtering stage.
    for i_subframe in 0..4 {
        let o = i_subframe * FS * NOLACE_AF1_OUT_CHANNELS;
        adaconv_process_frame(
            &mut state.af1_state,
            &mut x_buffer2[o..o + FS * NOLACE_AF1_OUT_CHANNELS],
            Some(&x_buffer1[i_subframe * FS..(i_subframe + 1) * FS]),
            &feature_buffer[i_subframe * CD..],
            &layers.nolace_af1_kernel,
            &layers.nolace_af1_gain,
            NOLACE_COND_DIM,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            NOLACE_AF1_IN_CHANNELS,
            NOLACE_AF1_OUT_CHANNELS,
            NOLACE_AF1_KERNEL_SIZE,
            NOLACE_AF1_LEFT_PADDING,
            NOLACE_AF1_FILTER_GAIN_A,
            NOLACE_AF1_FILTER_GAIN_B,
            NOLACE_AF1_SHAPE_GAIN,
            &h_nolace.window,
        );

        compute_generic_conv1d(
            &layers.nolace_post_af1,
            &mut feature_transform_buffer[i_subframe * CD..],
            &mut state.post_af1_state,
            &feature_buffer[i_subframe * CD..],
            NOLACE_COND_DIM,
            ACTIVATION_TANH,
        );
    }

    // Update feature buffer.
    feature_buffer.copy_from_slice(&feature_transform_buffer);

    // First shape-mix round.
    for i_subframe in 0..4 {
        const { assert!(NOLACE_AF1_OUT_CHANNELS == 2) };
        // Modifies second channel in place.
        let s = i_subframe * NOLACE_AF1_OUT_CHANNELS * FS + FS;
        adashape_process_frame(
            &mut state.tdshape1_state,
            &mut x_buffer2[s..s + FS],
            None,
            &feature_buffer[i_subframe * CD..],
            &layers.nolace_tdshape1_alpha1_f,
            &layers.nolace_tdshape1_alpha1_t,
            &layers.nolace_tdshape1_alpha2,
            NOLACE_TDSHAPE1_FEATURE_DIM,
            NOLACE_TDSHAPE1_FRAME_SIZE,
            NOLACE_TDSHAPE1_AVG_POOL_K,
            1,
        );

        let o = i_subframe * FS * NOLACE_AF2_OUT_CHANNELS;
        let i = i_subframe * FS * NOLACE_AF2_IN_CHANNELS;
        adaconv_process_frame(
            &mut state.af2_state,
            &mut x_buffer1[o..o + FS * NOLACE_AF2_OUT_CHANNELS],
            Some(&x_buffer2[i..i + FS * NOLACE_AF2_IN_CHANNELS]),
            &feature_buffer[i_subframe * CD..],
            &layers.nolace_af2_kernel,
            &layers.nolace_af2_gain,
            NOLACE_COND_DIM,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            NOLACE_AF2_IN_CHANNELS,
            NOLACE_AF2_OUT_CHANNELS,
            NOLACE_AF2_KERNEL_SIZE,
            NOLACE_AF2_LEFT_PADDING,
            NOLACE_AF2_FILTER_GAIN_A,
            NOLACE_AF2_FILTER_GAIN_B,
            NOLACE_AF2_SHAPE_GAIN,
            &h_nolace.window,
        );

        compute_generic_conv1d(
            &layers.nolace_post_af2,
            &mut feature_transform_buffer[i_subframe * CD..],
            &mut state.post_af2_state,
            &feature_buffer[i_subframe * CD..],
            NOLACE_COND_DIM,
            ACTIVATION_TANH,
        );
    }

    // Update feature buffer.
    feature_buffer.copy_from_slice(&feature_transform_buffer);

    // Second shape-mix round.
    for i_subframe in 0..4 {
        const { assert!(NOLACE_AF2_OUT_CHANNELS == 2) };
        // Modifies second channel in place.
        let s = i_subframe * NOLACE_AF2_OUT_CHANNELS * FS + FS;
        adashape_process_frame(
            &mut state.tdshape2_state,
            &mut x_buffer1[s..s + FS],
            None,
            &feature_buffer[i_subframe * CD..],
            &layers.nolace_tdshape2_alpha1_f,
            &layers.nolace_tdshape2_alpha1_t,
            &layers.nolace_tdshape2_alpha2,
            NOLACE_TDSHAPE2_FEATURE_DIM,
            NOLACE_TDSHAPE2_FRAME_SIZE,
            NOLACE_TDSHAPE2_AVG_POOL_K,
            1,
        );

        let o = i_subframe * FS * NOLACE_AF3_OUT_CHANNELS;
        let i = i_subframe * FS * NOLACE_AF3_IN_CHANNELS;
        adaconv_process_frame(
            &mut state.af3_state,
            &mut x_buffer2[o..o + FS * NOLACE_AF3_OUT_CHANNELS],
            Some(&x_buffer1[i..i + FS * NOLACE_AF3_IN_CHANNELS]),
            &feature_buffer[i_subframe * CD..],
            &layers.nolace_af3_kernel,
            &layers.nolace_af3_gain,
            NOLACE_COND_DIM,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            NOLACE_AF3_IN_CHANNELS,
            NOLACE_AF3_OUT_CHANNELS,
            NOLACE_AF3_KERNEL_SIZE,
            NOLACE_AF3_LEFT_PADDING,
            NOLACE_AF3_FILTER_GAIN_A,
            NOLACE_AF3_FILTER_GAIN_B,
            NOLACE_AF3_SHAPE_GAIN,
            &h_nolace.window,
        );

        compute_generic_conv1d(
            &layers.nolace_post_af3,
            &mut feature_transform_buffer[i_subframe * CD..],
            &mut state.post_af3_state,
            &feature_buffer[i_subframe * CD..],
            NOLACE_COND_DIM,
            ACTIVATION_TANH,
        );
    }

    // Update feature buffer.
    feature_buffer.copy_from_slice(&feature_transform_buffer);

    // Third shape-mix round.
    for i_subframe in 0..4 {
        const { assert!(NOLACE_AF3_OUT_CHANNELS == 2) };
        // Modifies second channel in place.
        let s = i_subframe * NOLACE_AF3_OUT_CHANNELS * FS + FS;
        adashape_process_frame(
            &mut state.tdshape3_state,
            &mut x_buffer2[s..s + FS],
            None,
            &feature_buffer[i_subframe * CD..],
            &layers.nolace_tdshape3_alpha1_f,
            &layers.nolace_tdshape3_alpha1_t,
            &layers.nolace_tdshape3_alpha2,
            NOLACE_TDSHAPE3_FEATURE_DIM,
            NOLACE_TDSHAPE3_FRAME_SIZE,
            NOLACE_TDSHAPE3_AVG_POOL_K,
            1,
        );

        let o = i_subframe * FS * NOLACE_AF4_OUT_CHANNELS;
        let i = i_subframe * FS * NOLACE_AF4_IN_CHANNELS;
        adaconv_process_frame(
            &mut state.af4_state,
            &mut x_buffer1[o..o + FS * NOLACE_AF4_OUT_CHANNELS],
            Some(&x_buffer2[i..i + FS * NOLACE_AF4_IN_CHANNELS]),
            &feature_buffer[i_subframe * CD..],
            &layers.nolace_af4_kernel,
            &layers.nolace_af4_gain,
            NOLACE_COND_DIM,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            NOLACE_AF4_IN_CHANNELS,
            NOLACE_AF4_OUT_CHANNELS,
            NOLACE_AF4_KERNEL_SIZE,
            NOLACE_AF4_LEFT_PADDING,
            NOLACE_AF4_FILTER_GAIN_A,
            NOLACE_AF4_FILTER_GAIN_B,
            NOLACE_AF4_SHAPE_GAIN,
            &h_nolace.window,
        );
    }

    // De-emphasis.
    for (o, &x) in x_out[..4 * FS].iter_mut().zip(&x_buffer1[..4 * FS]) {
        *o = x + NOLACE_PREEMPH * state.deemph_mem;
        state.deemph_mem = *o;
    }
}

// ---- BBWENet (ENABLE_OSCE_BWE) ----

/// Port of dnn/osce.c:bbwe_feature_net (static). `features` holds
/// `num_frames*BBWENET_FEATURE_DIM` floats, `output` receives
/// `2*num_frames*BBWENET_FNET_GRU_STATE_SIZE`.
pub fn bbwe_feature_net(
    h_bbwenet: &Bbwenet,
    state: &mut BbwenetState,
    output: &mut [f32],
    features: &[f32],
    num_frames: usize,
) {
    let mut input_buffer = [0f32; 4 * BBWENET_FNET_GRU_STATE_SIZE];
    let mut output_buffer = [0f32; 4 * BBWENET_FNET_GRU_STATE_SIZE];
    let layers = &h_bbwenet.layers;

    // Adjust buffer sizes if any of this breaks.
    const { assert!(BBWENET_FNET_GRU_STATE_SIZE == BBWENET_FNET_TCONV_OUT_CHANNELS) };
    const { assert!(BBWENET_FNET_TCONV_OUT_CHANNELS == BBWENET_FNET_CONV2_OUT_SIZE) };
    const { assert!(BBWENET_FNET_CONV2_OUT_SIZE == BBWENET_FNET_CONV1_OUT_SIZE) };
    debug_assert!(num_frames <= 2);

    // First conv layer.
    for i_frame in 0..num_frames {
        compute_generic_conv1d(
            &layers.bbwenet_fnet_conv1,
            &mut output_buffer[i_frame * BBWENET_FNET_CONV1_OUT_SIZE..],
            &mut state.feature_net_conv1_state,
            &features[i_frame * BBWENET_FEATURE_DIM..],
            BBWENET_FEATURE_DIM,
            ACTIVATION_TANH,
        );
    }
    let n = num_frames * BBWENET_FNET_CONV1_OUT_SIZE;
    input_buffer[..n].copy_from_slice(&output_buffer[..n]);

    // Second conv layer.
    for i_frame in 0..num_frames {
        compute_generic_conv1d(
            &layers.bbwenet_fnet_conv2,
            &mut output_buffer[i_frame * BBWENET_FNET_CONV2_OUT_SIZE..],
            &mut state.feature_net_conv2_state,
            &input_buffer[i_frame * BBWENET_FNET_CONV1_OUT_SIZE..],
            BBWENET_FNET_CONV1_OUT_SIZE,
            ACTIVATION_TANH,
        );
    }
    let n = num_frames * BBWENET_FNET_CONV2_OUT_SIZE;
    input_buffer[..n].copy_from_slice(&output_buffer[..n]);

    // tconv upsampling.
    const TC: usize = BBWENET_FNET_TCONV_OUT_CHANNELS * BBWENET_FNET_TCONV_STRIDE;
    for i_frame in 0..num_frames {
        compute_generic_dense(
            &layers.bbwenet_fnet_tconv,
            &mut output_buffer[i_frame * TC..],
            &input_buffer[i_frame * BBWENET_FNET_CONV2_OUT_SIZE..],
            ACTIVATION_TANH,
        );
    }
    let n = num_frames * TC;
    input_buffer[..n].copy_from_slice(&output_buffer[..n]);

    // GRU.
    const { assert!(BBWENET_FNET_TCONV_STRIDE == 2) };
    for i_subframe in 0..BBWENET_FNET_TCONV_STRIDE * num_frames {
        compute_generic_gru(
            &layers.bbwenet_fnet_gru_input,
            &layers.bbwenet_fnet_gru_recurrent,
            &mut state.feature_net_gru_state,
            &input_buffer[i_subframe * BBWENET_FNET_TCONV_OUT_CHANNELS..],
        );
        output[i_subframe * BBWENET_FNET_GRU_STATE_SIZE
            ..(i_subframe + 1) * BBWENET_FNET_GRU_STATE_SIZE]
            .copy_from_slice(&state.feature_net_gru_state);
    }
}

// The C tables are double literals stored in float arrays: convert through f64 so the rounding
// matches the C compiler's.
/// `hq_2x_even`.
const HQ_2X_EVEN: [f32; 3] = [
    0.026641845703125_f64 as f32,
    0.228668212890625_f64 as f32,
    -0.4036407470703125_f64 as f32,
];
/// `hq_2x_odd`.
const HQ_2X_ODD: [f32; 3] = [
    0.104583740234375_f64 as f32,
    0.3932037353515625_f64 as f32,
    -0.152496337890625_f64 as f32,
];
/// `frac_01_24`.
const FRAC_01_24: [f32; 8] = [
    0.00576782_f64 as f32,
    -0.01831055_f64 as f32,
    0.01882935_f64 as f32,
    0.9328308_f64 as f32,
    0.09143066_f64 as f32,
    -0.04196167_f64 as f32,
    0.01296997_f64 as f32,
    -0.00140381_f64 as f32,
];
/// `frac_17_24`.
const FRAC_17_24: [f32; 8] = [
    -3.14331055e-03_f64 as f32,
    2.73437500e-02_f64 as f32,
    -1.06414795e-01_f64 as f32,
    3.64685059e-01_f64 as f32,
    8.03863525e-01_f64 as f32,
    -1.02233887e-01_f64 as f32,
    1.61437988e-02_f64 as f32,
    -1.22070312e-04_f64 as f32,
];
/// `frac_09_24`.
const FRAC_09_24: [f32; 8] = [
    -0.00146484_f64 as f32,
    0.02313232_f64 as f32,
    -0.12072754_f64 as f32,
    0.7315979_f64 as f32,
    0.4621277_f64 as f32,
    -0.12075806_f64 as f32,
    0.0295105_f64 as f32,
    -0.00326538_f64 as f32,
];

/// Port of dnn/osce.c:apply_valin_activation (static), in place on `x[..len]`.
pub fn apply_valin_activation(x: &mut [f32], len: usize) {
    let mut y = [0f32; 2 * BBWENET_TDSHAPE2_FRAME_SIZE];
    debug_assert!(len <= 2 * BBWENET_TDSHAPE2_FRAME_SIZE);
    let x = &mut x[..len];
    let y = &mut y[..len];
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        // C: fabs(x[i]) + 1e-6f is a double addition.
        *yi = (math::fabs(xi as f64) + 1e-6f32 as f64) as f32;
    }
    for yi in y.iter_mut() {
        *yi = celt_log(*yi);
    }
    for (xi, &yi) in x.iter_mut().zip(y.iter()) {
        *xi *= celt_sin(yi);
    }
}

/// `DELAY_SAMPLES` ("this probably should be 7, bug in python code?").
const DELAY_SAMPLES: usize = 8;

/// 8-tap dot product in C order (`b[0]*f[0] + b[1]*f[1] + ...`).
#[inline(always)]
fn dot8(b: &[f32], f: &[f32; 8]) -> f32 {
    b[0] * f[0]
        + b[1] * f[1]
        + b[2] * f[2]
        + b[3] * f[3]
        + b[4] * f[4]
        + b[5] * f[5]
        + b[6] * f[6]
        + b[7] * f[7]
}

/// Port of dnn/osce.c:interpol_3_2 (static): 3/2 interpolation of `num_samples` input samples
/// into `3*num_samples/2` output samples.
pub fn interpol_3_2(state: &mut ResampState, x_out: &mut [f32], x_in: &[f32], num_samples: usize) {
    let mut buffer = [0f32; 8 * BBWENET_FRAME_SIZE16 + DELAY_SAMPLES];
    let mut i_out = 0;

    debug_assert!(num_samples > 1);
    debug_assert!(num_samples < 8 * BBWENET_FRAME_SIZE16);
    debug_assert!(num_samples.is_multiple_of(2));

    buffer[..DELAY_SAMPLES].copy_from_slice(&state.interpol_buffer);
    buffer[DELAY_SAMPLES..DELAY_SAMPLES + num_samples].copy_from_slice(&x_in[..num_samples]);

    let x_out = &mut x_out[..3 * num_samples / 2];
    for i_sample in (0..num_samples).step_by(2) {
        x_out[i_out] = dot8(&buffer[i_sample..], &FRAC_01_24);
        i_out += 1;
        x_out[i_out] = dot8(&buffer[i_sample..], &FRAC_17_24);
        i_out += 1;
        x_out[i_out] = dot8(&buffer[i_sample + 1..], &FRAC_09_24);
        i_out += 1;
    }

    // Copy last samples to buffer.
    state
        .interpol_buffer
        .copy_from_slice(&buffer[num_samples..num_samples + DELAY_SAMPLES]);
}

/// Port of dnn/osce.c:upsamp_2x (static): 2x upsampling of `num_samples` samples (allpass
/// polyphase, the SILK `hq_2x` coefficients) into `2*num_samples`.
pub fn upsamp_2x(state: &mut ResampState, x_out: &mut [f32], x_in: &[f32], num_samples: usize) {
    debug_assert!(num_samples > 1);
    debug_assert!(num_samples < 4 * BBWENET_FRAME_SIZE16);
    let [s_even, s_odd] = &mut state.upsamp_buffer;
    let x_out = &mut x_out[..2 * num_samples];

    for (k, &x) in x_in[..num_samples].iter().enumerate() {
        // Even sample, first pass,
        let y = x - s_even[0];
        let xx = y * HQ_2X_EVEN[0];
        let tmp1 = s_even[0] + xx;
        s_even[0] = x + xx;

        // ...second pass,
        let y = tmp1 - s_even[1];
        let xx = y * HQ_2X_EVEN[1];
        let tmp2 = s_even[1] + xx;
        s_even[1] = tmp1 + xx;

        // ...third pass.
        let y = tmp2 - s_even[2];
        let xx = y * (1.0 + HQ_2X_EVEN[2]);
        let tmp3 = s_even[2] + xx;
        s_even[2] = tmp2 + xx;

        x_out[2 * k] = tmp3;

        // Odd sample, first pass,
        let y = x - s_odd[0];
        let xx = y * HQ_2X_ODD[0];
        let tmp1 = s_odd[0] + xx;
        s_odd[0] = x + xx;

        // ...second pass,
        let y = tmp1 - s_odd[1];
        let xx = y * HQ_2X_ODD[1];
        let tmp2 = s_odd[1] + xx;
        s_odd[1] = tmp1 + xx;

        // ...third pass.
        let y = tmp2 - s_odd[2];
        let xx = y * (1.0 + HQ_2X_ODD[2]);
        let tmp3 = s_odd[2] + xx;
        s_odd[2] = tmp2 + xx;

        x_out[2 * k + 1] = tmp3;
    }
}

/// Port of dnn/osce.c:bbwenet_process_frames (static): extends `num_frames` 10 ms frames of
/// 16 kHz input (`x_in`, 160 samples each) to 48 kHz (`x_out`, 480 samples each).
pub fn bbwenet_process_frames(
    h_bbwenet: &Bbwenet,
    state: &mut BbwenetState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    num_frames: usize,
) {
    let mut latent_features = [0f32; 4 * BBWENET_COND_DIM];
    let num_subframes = 2 * num_frames;
    let layers = &h_bbwenet.layers;
    debug_assert!(num_frames <= 2);

    // Feature net.
    bbwe_feature_net(h_bbwenet, state, &mut latent_features, features, num_frames);

    // The C buffers are zero-initialized on the stack (only the used part here).
    let mut x_buffer1 = [0f32; BBWENET_XBUF1_SIZE];
    let mut x_buffer2 = [0f32; BBWENET_XBUF2_SIZE];
    let (x_buffer1, x_buffer2) = (&mut x_buffer1[..], &mut x_buffer2[..]);

    // Signal net: first adaptive filtering stage, three output channels.
    for i_subframe in 0..num_subframes {
        let o = i_subframe * BBWENET_AF1_FRAME_SIZE * BBWENET_AF1_OUT_CHANNELS;
        adaconv_process_frame(
            &mut state.af1_state,
            &mut x_buffer1[o..o + BBWENET_AF1_FRAME_SIZE * BBWENET_AF1_OUT_CHANNELS],
            Some(&x_in[i_subframe * BBWENET_AF1_FRAME_SIZE..]),
            &latent_features[i_subframe * BBWENET_COND_DIM..],
            &layers.bbwenet_af1_kernel,
            &layers.bbwenet_af1_gain,
            BBWENET_COND_DIM,
            BBWENET_AF1_FRAME_SIZE,
            BBWENET_AF1_OVERLAP_SIZE,
            BBWENET_AF1_IN_CHANNELS,
            BBWENET_AF1_OUT_CHANNELS,
            BBWENET_AF1_KERNEL_SIZE,
            BBWENET_AF1_LEFT_PADDING,
            BBWENET_AF1_FILTER_GAIN_A,
            BBWENET_AF1_FILTER_GAIN_B,
            BBWENET_AF1_SHAPE_GAIN,
            &h_bbwenet.window16,
        );
    }

    // 1st round of non-linear extension.
    for i_subframe in 0..num_subframes {
        // 2x upsampling on individual channels.
        const { assert!(BBWENET_AF1_OUT_CHANNELS == 3) };
        const { assert!(2 * BBWENET_AF1_FRAME_SIZE == BBWENET_TDSHAPE1_FRAME_SIZE) };
        for i_channel in 0..3 {
            let o = i_subframe * BBWENET_TDSHAPE1_FRAME_SIZE * BBWENET_AF1_OUT_CHANNELS
                + i_channel * BBWENET_TDSHAPE1_FRAME_SIZE;
            let i = i_subframe * BBWENET_AF1_FRAME_SIZE * BBWENET_AF1_OUT_CHANNELS
                + i_channel * BBWENET_AF1_FRAME_SIZE;
            upsamp_2x(
                &mut state.resampler_state[i_channel],
                &mut x_buffer2[o..],
                &x_buffer1[i..],
                BBWENET_AF1_FRAME_SIZE,
            );
        }

        // tdshape on second channel (in place).
        let s = i_subframe * BBWENET_AF1_OUT_CHANNELS * BBWENET_TDSHAPE1_FRAME_SIZE
            + BBWENET_TDSHAPE1_FRAME_SIZE;
        adashape_process_frame(
            &mut state.tdshape1_state,
            &mut x_buffer2[s..s + BBWENET_TDSHAPE1_FRAME_SIZE],
            None,
            &latent_features[i_subframe * BBWENET_COND_DIM..],
            &layers.bbwenet_tdshape1_alpha1_f,
            &layers.bbwenet_tdshape1_alpha1_t,
            &layers.bbwenet_tdshape1_alpha2,
            BBWENET_TDSHAPE1_FEATURE_DIM,
            BBWENET_TDSHAPE1_FRAME_SIZE,
            BBWENET_TDSHAPE1_AVG_POOL_K,
            BBWENET_TDSHAPE1_INTERPOLATE_K,
        );

        // Non-linear activation of third channel (in place).
        let s = i_subframe * BBWENET_AF1_OUT_CHANNELS * BBWENET_TDSHAPE1_FRAME_SIZE
            + 2 * BBWENET_TDSHAPE1_FRAME_SIZE;
        apply_valin_activation(&mut x_buffer2[s..], BBWENET_TDSHAPE1_FRAME_SIZE);
    }

    // Mixing.
    for i_subframe in 0..num_subframes {
        let o = i_subframe * BBWENET_AF2_FRAME_SIZE * BBWENET_AF2_OUT_CHANNELS;
        let i = i_subframe * BBWENET_AF2_FRAME_SIZE * BBWENET_AF1_OUT_CHANNELS;
        adaconv_process_frame(
            &mut state.af2_state,
            &mut x_buffer1[o..o + BBWENET_AF2_FRAME_SIZE * BBWENET_AF2_OUT_CHANNELS],
            Some(&x_buffer2[i..i + BBWENET_AF2_FRAME_SIZE * BBWENET_AF2_IN_CHANNELS]),
            &latent_features[i_subframe * BBWENET_COND_DIM..],
            &layers.bbwenet_af2_kernel,
            &layers.bbwenet_af2_gain,
            BBWENET_COND_DIM,
            BBWENET_AF2_FRAME_SIZE,
            BBWENET_AF2_OVERLAP_SIZE,
            BBWENET_AF2_IN_CHANNELS,
            BBWENET_AF2_OUT_CHANNELS,
            BBWENET_AF2_KERNEL_SIZE,
            BBWENET_AF2_LEFT_PADDING,
            BBWENET_AF2_FILTER_GAIN_A,
            BBWENET_AF2_FILTER_GAIN_B,
            BBWENET_AF2_SHAPE_GAIN,
            &h_bbwenet.window32,
        );
    }

    // Second round of extension.
    for i_subframe in 0..num_subframes {
        // 1.5x interpolation on individual channels.
        const { assert!(BBWENET_AF2_OUT_CHANNELS == 3) };
        const { assert!(3 * BBWENET_AF2_FRAME_SIZE == 2 * BBWENET_TDSHAPE2_FRAME_SIZE) };
        for i_channel in 0..3 {
            let o = i_subframe * BBWENET_AF3_FRAME_SIZE * BBWENET_AF2_OUT_CHANNELS
                + i_channel * BBWENET_TDSHAPE2_FRAME_SIZE;
            let i = i_subframe * BBWENET_TDSHAPE1_FRAME_SIZE * BBWENET_AF2_OUT_CHANNELS
                + i_channel * BBWENET_TDSHAPE1_FRAME_SIZE;
            interpol_3_2(
                &mut state.resampler_state[i_channel],
                &mut x_buffer2[o..],
                &x_buffer1[i..],
                BBWENET_TDSHAPE1_FRAME_SIZE,
            );
        }

        // tdshape on second channel (in place).
        let s = i_subframe * BBWENET_AF2_OUT_CHANNELS * BBWENET_TDSHAPE2_FRAME_SIZE
            + BBWENET_TDSHAPE2_FRAME_SIZE;
        adashape_process_frame(
            &mut state.tdshape2_state,
            &mut x_buffer2[s..s + BBWENET_TDSHAPE2_FRAME_SIZE],
            None,
            &latent_features[i_subframe * BBWENET_COND_DIM..],
            &layers.bbwenet_tdshape2_alpha1_f,
            &layers.bbwenet_tdshape2_alpha1_t,
            &layers.bbwenet_tdshape2_alpha2,
            BBWENET_TDSHAPE2_FEATURE_DIM,
            BBWENET_TDSHAPE2_FRAME_SIZE,
            BBWENET_TDSHAPE2_AVG_POOL_K,
            BBWENET_TDSHAPE2_INTERPOLATE_K,
        );

        // Non-linear activation of third channel (in place).
        let s = i_subframe * BBWENET_AF2_OUT_CHANNELS * BBWENET_TDSHAPE2_FRAME_SIZE
            + 2 * BBWENET_TDSHAPE2_FRAME_SIZE;
        apply_valin_activation(&mut x_buffer2[s..], BBWENET_TDSHAPE2_FRAME_SIZE);
    }

    // Final mixing.
    const { assert!(BBWENET_AF3_OUT_CHANNELS == 1) };
    for i_subframe in 0..num_subframes {
        let o = i_subframe * BBWENET_AF3_FRAME_SIZE;
        let i = i_subframe * BBWENET_TDSHAPE2_FRAME_SIZE * BBWENET_AF2_OUT_CHANNELS;
        adaconv_process_frame(
            &mut state.af3_state,
            &mut x_out[o..o + BBWENET_AF3_FRAME_SIZE],
            Some(&x_buffer2[i..i + BBWENET_TDSHAPE2_FRAME_SIZE * BBWENET_AF3_IN_CHANNELS]),
            &latent_features[i_subframe * BBWENET_COND_DIM..],
            &layers.bbwenet_af3_kernel,
            &layers.bbwenet_af3_gain,
            BBWENET_COND_DIM,
            BBWENET_AF3_FRAME_SIZE,
            BBWENET_AF3_OVERLAP_SIZE,
            BBWENET_AF3_IN_CHANNELS,
            BBWENET_AF3_OUT_CHANNELS,
            BBWENET_AF3_KERNEL_SIZE,
            BBWENET_AF3_LEFT_PADDING,
            BBWENET_AF3_FILTER_GAIN_A,
            BBWENET_AF3_FILTER_GAIN_B,
            BBWENET_AF3_SHAPE_GAIN,
            &h_bbwenet.window48,
        );
    }
}

/// Port of dnn/osce.c:reset_bbwenet_state (static).
pub fn reset_bbwenet_state(state: &mut BbwenetState) {
    state.clear();
    init_adaconv_state(&mut state.af1_state);
    init_adaconv_state(&mut state.af2_state);
    init_adaconv_state(&mut state.af3_state);
    init_adashape_state(&mut state.tdshape1_state);
    init_adashape_state(&mut state.tdshape2_state);
}

/// Port of dnn/osce.c:init_bbwenet (static): binds the layers and computes the windows.
pub fn init_bbwenet(weights: &[WeightArray<'_>]) -> Result<Bbwenet> {
    let mut h = Bbwenet {
        layers: init_bbwenetlayers(weights)?,
        ..Bbwenet::default()
    };
    compute_overlap_window(&mut h.window16, BBWENET_AF1_OVERLAP_SIZE);
    compute_overlap_window(&mut h.window32, BBWENET_AF2_OVERLAP_SIZE);
    compute_overlap_window(&mut h.window48, BBWENET_AF3_OVERLAP_SIZE);
    Ok(h)
}

// ---- API ----

/// Port of dnn/osce.c:osce_reset: clears the feature state, resets the state of `method` and
/// selects it (with a pass-through for the first frame and a cross-fade on the second).
pub fn osce_reset(h_osce: &mut SilkOsceStruct, method: i32) {
    h_osce.features = OsceFeatureState::default();

    // The members not selected here are dead until `osce_reset` selects their method again
    // (a C union), so they are dropped; the selected one is reset in place if allocated.
    let state = &mut h_osce.state;
    match method {
        OSCE_METHOD_NONE => {
            state.lace = None;
            state.nolace = None;
        }
        OSCE_METHOD_LACE => {
            state.nolace = None;
            if let Some(l) = state.lace.as_deref_mut() {
                reset_lace_state(l);
            }
        }
        OSCE_METHOD_NOLACE => {
            state.lace = None;
            if let Some(n) = state.nolace.as_deref_mut() {
                reset_nolace_state(n);
            }
        }
        // C: celt_assert(0 && "method not defined").
        _ => debug_assert!(false, "OSCE method {method} not defined"),
    }
    h_osce.method = method;
    h_osce.features.reset = 2;
}

/// Port of dnn/osce.c:osce_bwe_reset.
pub fn osce_bwe_reset(h_osce_bwe: &mut SilkOsceBweStruct) {
    h_osce_bwe.features = OsceBweFeatureState::default();
    // "weird python initialization: Fix eventually!"
    for k in 0..=OSCE_BWE_MAX_INSTAFREQ_BIN {
        h_osce_bwe.features.last_spec[2 * k] = 1e-9_f64 as f32;
    }
    reset_bbwenet_state(h_osce_bwe.state.bbwenet_mut());
}

/// Port of dnn/osce.c:osce_load_models: binds LACE, NoLACE and BBWENet from a weight blob.
///
/// `None` / an empty blob fails like upstream `USE_WEIGHTS_FILE` builds (the compiled-in weight
/// tables are not part of the port, PLAN D-015), as does a blob that cannot be parsed or lacks
/// a layer of any of the three models (note that upstream `write_lpcnet_weights` does not write
/// BBWENet). C returns -1 for all of these; the caller sets [`OsceModel::loaded`] from the
/// result.
pub fn osce_load_models(model: &mut OsceModel, data: Option<&[u8]>) -> Result<()> {
    match data {
        Some(data) if !data.is_empty() => {
            // Init from buffer (the models of the embedded blob are shared).
            model.lace = Some(load_shared(data, init_lace, &LACE_CACHE)?);
            model.nolace = Some(load_shared(data, init_nolace, &NOLACE_CACHE)?);
            model.bbwenet = Some(load_shared(data, init_bbwenet, &BBWENET_CACHE)?);
            Ok(())
        }
        // USE_WEIGHTS_FILE: return -1.
        _ => Err(Error::BadArg),
    }
}

/// Port of dnn/osce.c:osce_bwe: extends `xq16_len` (160 or 320) samples of 16 kHz speech to
/// `3*xq16_len` samples at 48 kHz in `xq48` (delayed by `OSCE_BWE_OUTPUT_DELAY`).
///
/// Like C, this runs BBWENet whether or not `model.loaded` is set; callers only enable BWE with
/// a loaded model.
pub fn osce_bwe(
    model: &OsceModel,
    ps_osce_bwe: &mut SilkOsceBweStruct,
    xq48: &mut [i16],
    xq16: &[i16],
    xq16_len: usize,
) {
    let mut in_buffer = [0f32; 320];
    let mut out_buffer = [0f32; 3 * 320];
    let mut features = [0f32; 2 * OSCE_BWE_FEATURE_DIM];

    // Currently restricting to 10 or 20 ms frames.
    debug_assert!(xq16_len == 160 || xq16_len == 320);

    let num_frames = xq16_len / 160;

    // Scale input.
    for (b, &x) in in_buffer[..xq16_len].iter_mut().zip(&xq16[..xq16_len]) {
        *b = x as f32 * (1.0f32 / 32768.0f32);
    }

    osce_bwe_calculate_features(&mut ps_osce_bwe.features, &mut features, xq16, xq16_len);

    // Process frames.
    bbwenet_process_frames(
        &model.bbwenet(),
        ps_osce_bwe.state.bbwenet_mut(),
        &mut out_buffer,
        &in_buffer,
        &features,
        num_frames,
    );

    // Scale and delay output.
    let bb = ps_osce_bwe.state.bbwenet_mut();
    let n_out = 3 * xq16_len;
    xq48[..OSCE_BWE_OUTPUT_DELAY].copy_from_slice(&bb.outbut_buffer);
    for (o, &v) in xq48[OSCE_BWE_OUTPUT_DELAY..n_out]
        .iter_mut()
        .zip(&out_buffer[..n_out - OSCE_BWE_OUTPUT_DELAY])
    {
        *o = scale_to_i16(v);
    }
    for (o, &v) in bb
        .outbut_buffer
        .iter_mut()
        .zip(&out_buffer[n_out - OSCE_BWE_OUTPUT_DELAY..n_out])
    {
        *o = scale_to_i16(v);
    }
}

/// C: `tmp = 32768.f * x; clamp to [-32767, 32767]; float2int(tmp)`.
#[inline]
#[expect(clippy::manual_clamp, reason = "mirrors the C comparisons")]
fn scale_to_i16(x: f32) -> i16 {
    let mut tmp = 32768.0f32 * x;
    if tmp > 32767.0 {
        tmp = 32767.0;
    }
    if tmp < -32767.0 {
        tmp = -32767.0;
    }
    float2int(tmp) as i16
}

/// Port of dnn/osce.c:osce_enhance_frame: enhances one decoded SILK frame `xq` in place.
///
/// Only 20 ms frames at 16 kHz (320 samples) are enhanced; otherwise the state is reset.
/// `dec` carries the `silk_decoder_state` fields read here, `num_bits` is the size of the SILK
/// payload of the frame in bits (`ec_tell(psRangeDec) - ec_start`). The ENABLE_OSCE_TRAINING_DATA
/// dump is not ported.
pub fn osce_enhance_frame(
    model: &OsceModel,
    ps_dec_osce: &mut SilkOsceStruct,
    dec: OsceDecInfo,
    ps_dec_ctrl: &SilkDecoderControl,
    xq: &mut [i16],
    num_bits: i32,
) {
    let mut in_buffer = [0f32; 320];
    let mut out_buffer = [0f32; 320];
    let mut features = [0f32; 4 * OSCE_FEATURE_DIM];
    let mut numbits = [0f32; 2];
    let mut periods = [0i32; 4];

    // Enhancement only implemented for 20 ms frame at 16 kHz.
    if dec.fs_khz != 16 || dec.nb_subfr != 4 {
        let method = ps_dec_osce.method;
        osce_reset(ps_dec_osce, method);
        return;
    }

    osce_calculate_features(
        &mut ps_dec_osce.features,
        dec,
        ps_dec_ctrl,
        &mut features,
        &mut numbits,
        &mut periods,
        xq,
        num_bits,
    );

    // Scale input.
    let xq = &mut xq[..320];
    for (b, &x) in in_buffer.iter_mut().zip(xq.iter()) {
        *b = x as f32 * (1.0f32 / 32768.0f32);
    }

    let method = if model.loaded {
        ps_dec_osce.method
    } else {
        OSCE_METHOD_NONE
    };
    match method {
        OSCE_METHOD_NONE => out_buffer.copy_from_slice(&in_buffer),
        OSCE_METHOD_LACE => lace_process_20ms_frame(
            &model.lace(),
            ps_dec_osce.state.lace_mut(),
            &mut out_buffer,
            &in_buffer,
            &features,
            &numbits,
            &periods,
        ),
        OSCE_METHOD_NOLACE => nolace_process_20ms_frame(
            &model.nolace(),
            ps_dec_osce.state.nolace_mut(),
            &mut out_buffer,
            &in_buffer,
            &features,
            &numbits,
            &periods,
        ),
        _ => {
            // C: celt_assert(0 && "method not defined") and an uninitialized out_buffer.
            debug_assert!(false, "OSCE method {method} not defined");
            out_buffer.copy_from_slice(&in_buffer);
        }
    }

    if ps_dec_osce.features.reset > 1 {
        out_buffer.copy_from_slice(&in_buffer);
        ps_dec_osce.features.reset -= 1;
    } else if ps_dec_osce.features.reset != 0 {
        osce_cross_fade_10ms(&mut out_buffer, &in_buffer, 320);
        ps_dec_osce.features.reset = 0;
    }

    // Scale output.
    for (x, &v) in xq.iter_mut().zip(&out_buffer) {
        *x = scale_to_i16(v);
    }
}
