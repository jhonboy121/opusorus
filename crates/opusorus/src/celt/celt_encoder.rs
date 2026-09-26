//! Port of celt/celt_encoder.c: the CELT encoder (`OpusCustomEncoder` / `CELTEncoder`).
//!
//! * [`CeltEncoder`] is `struct OpusCustomEncoder`. The C trailing arrays (`in_mem`,
//!   `prefilter_mem`, `oldBandE`, `oldLogE`, `oldLogE2`, `energyError` and, with QEXT, the QEXT
//!   band energies stored after `energyError`) are owned `Vec`s sized at init. The per-frame C
//!   VLAs live in a private scratch area allocated at init, so encoding never allocates.
//! * `celt_encoder_init` / `opus_custom_encoder_init` are [`CeltEncoder::celt_encoder_init`] /
//!   [`CeltEncoder::opus_custom_encoder_init`]; every `celt_encoder_ctl` request is a typed method
//!   (`set_complexity`, `set_start_band`, ..., [`CeltEncoder::reset`] for `OPUS_RESET_STATE`).
//! * [`CeltEncoder::celt_encode_with_ec`] is the entry point. It takes an optional existing range
//!   encoder (hybrid mode) exactly like C. With QEXT, the extension is signalled by setting the
//!   code-3 bits in the byte *before* the CELT payload (the Opus TOC, C `compressed[-1]`), which
//!   Rust passes explicitly as `toc`.
//!
//! Deviations (none observable in the output):
//! * `energy_mask`: C keeps a pointer to a caller-owned array; the port copies the values when
//!   the mask is set ([`CeltEncoder::set_energy_mask`]). Callers that update the mask (the
//!   multistream surround encoder) set it again before each frame, as the C code does.
//! * `AnalysisInfo` is [`crate::analysis::AnalysisInfo`] (what the Opus encoder produces).
//! * `RESYNTH` (debug-only re-synthesis, never enabled in libopus builds) is not ported.
//! * The QEXT mode (`compute_qext_mode(mode)`, a pure function of the mode) is computed once at
//!   init instead of every frame.
//! * `FUZZING` branches are not ported. `FIXED_POINT` branches are skipped with markers.
//! * There are no DNN (deep PLC / DRED / OSCE) hooks in this file.

#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]
#![allow(clippy::too_many_arguments, reason = "mirrors the C signatures")]

use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;

use crate::analysis::{AnalysisInfo, LEAK_BANDS};
#[cfg(feature = "qext")]
use crate::celt::arch::Q15ONE;
use crate::celt::arch::{
    CeltEner, CeltGlog, CeltNorm, CeltSig, EPSILON, OpusRes, OpusVal16, OpusVal32, abs16, abs32,
    half16, half32, imax, imin, max16, max32, maxg, min16, min32, ming, res2sig,
};
use crate::celt::bands::{
    BandsScratch, SPREAD_AGGRESSIVE, SPREAD_NONE, SPREAD_NORMAL, compute_band_energies, haar1,
    hysteresis_decision, normalise_bands, quant_all_bands, spreading_decision,
};
use crate::celt::celt::{
    COMBFILTER_MAXPERIOD, COMBFILTER_MINPERIOD, SPREAD_ICDF, SilkInfo, TAPSET_ICDF,
    TF_SELECT_TABLE, TRIM_ICDF, bitrate_to_bits, comb_filter, init_caps, resampling_factor,
};
use crate::celt::entcode::{BITRES, EcCoder, ec_ilog};
use crate::celt::entenc::EcEnc;
use crate::celt::mathops::{PI, celt_exp2_db, celt_log2, celt_maxabs_res, celt_rcp};
use crate::celt::mdct::clt_mdct_forward;
use crate::celt::modes::opus_custom_mode_create;
#[cfg(feature = "qext")]
use crate::celt::modes::{NB_QEXT_BANDS, QEXT_PACKET_SIZE_CAP, compute_qext_mode};
use crate::celt::pitch::{celt_inner_prod, pitch_downsample, pitch_search, remove_doubling};
use crate::celt::quant_bands::{
    E_MEANS, amp2_log2, quant_coarse_energy, quant_energy_finalise, quant_fine_energy,
};
use crate::celt::rate::clt_compute_allocation;
#[cfg(feature = "qext")]
use crate::celt::rate::clt_compute_extra_allocation;
use crate::celt::static_modes::CeltMode;
use crate::constants::raw::OPUS_BITRATE_MAX;
use crate::math;
use crate::{Error, Result};

/// `OPUS_BAD_ARG`.
const OPUS_BAD_ARG: i32 = -1;
/// `OPUS_INTERNAL_ERROR`.
const OPUS_INTERNAL_ERROR: i32 = -3;

/// Number of QEXT bands (0 without QEXT; sizes the QEXT energy state).
#[cfg(feature = "qext")]
const QEXT_BANDS: usize = NB_QEXT_BANDS as usize;
#[cfg(not(feature = "qext"))]
const QEXT_BANDS: usize = 0;

/// C `acos` (double). Private copy: `crate::math` has no `acos` yet (dedupe candidate).
#[inline(always)]
fn acos(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.acos()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::acos(x)
    }
}

/// C `floor` applied to a double expression, converted with C `(int)` (truncation).
#[inline(always)]
fn floor_i32(x: f64) -> i32 {
    math::floor(x) as i32
}

/// Scratch buffers replacing the C VLAs of the encoder (allocated once at init).
#[derive(Debug, Clone, Default)]
struct EncScratch {
    /// `in`: `CC*(N+overlap)`.
    input: Vec<CeltSig>,
    /// `freq`: `CC*N`.
    freq: Vec<CeltSig>,
    /// `X`: `CC*N` (C allocates `C*N`).
    x: Vec<CeltNorm>,
    /// `bandE`, `bandLogE`, `bandLogE2`, `error`, `surround_dynalloc`: `CC*nbEBands`.
    band_e: Vec<CeltEner>,
    band_log_e: Vec<CeltGlog>,
    band_log_e2: Vec<CeltGlog>,
    error: Vec<CeltGlog>,
    surround_dynalloc: Vec<CeltGlog>,
    /// Per-band integer arrays (`nbEBands`, `+NB_QEXT_BANDS` for the QEXT allocation).
    offsets: Vec<i32>,
    importance: Vec<i32>,
    spread_weight: Vec<i32>,
    tf_res: Vec<i32>,
    cap: Vec<i32>,
    fine_quant: Vec<i32>,
    pulses: Vec<i32>,
    fine_priority: Vec<i32>,
    collapse_masks: Vec<u8>,
    /// `transient_analysis` / `tone_detect` temporaries: `N+overlap`.
    tmp: Vec<OpusVal16>,
    /// `run_prefilter`: `_pre` (`CC*(N+max_period)`) and `pitch_buf` (`(max_period+N)>>1`).
    pre: Vec<CeltSig>,
    pitch_buf: Vec<OpusVal16>,
    /// `tf_analysis` temporaries.
    tf_metric: Vec<i32>,
    tf_path0: Vec<i32>,
    tf_path1: Vec<i32>,
    tf_tmp: Vec<CeltNorm>,
    tf_tmp1: Vec<CeltNorm>,
    /// `dynalloc_analysis` temporaries.
    dyn_bufs: DynallocScratch,
    /// `quant_all_bands` scratch.
    bands: BandsScratch,
    /// Custom-mode API sample conversion (`opus_custom_encode` / `encode24`).
    #[cfg(feature = "custom-modes")]
    pcm_conv: Vec<OpusRes>,
    #[cfg(feature = "qext")]
    extra_quant: Vec<i32>,
    #[cfg(feature = "qext")]
    extra_pulses: Vec<i32>,
    #[cfg(feature = "qext")]
    error_bak: Vec<CeltGlog>,
    #[cfg(feature = "qext")]
    zeros: Vec<i32>,
}

/// Scratch buffers of [`dynalloc_analysis`] (C VLAs `follower`, `noise_floor`, `bandLogE3`,
/// `mask`, `sig`).
#[derive(Debug, Clone, Default)]
pub struct DynallocScratch {
    follower: Vec<CeltGlog>,
    noise_floor: Vec<CeltGlog>,
    band_log_e3: Vec<CeltGlog>,
    mask: Vec<CeltGlog>,
    sig: Vec<CeltGlog>,
}

impl DynallocScratch {
    /// Scratch for a mode with `nb_ebands` bands and up to `c` channels.
    #[must_use]
    pub fn new(nb_ebands: usize, c: usize) -> Self {
        Self {
            follower: vec![0.0; c * nb_ebands],
            noise_floor: vec![0.0; c * nb_ebands],
            band_log_e3: vec![0.0; nb_ebands],
            mask: vec![0.0; nb_ebands],
            sig: vec![0.0; nb_ebands],
        }
    }
}

/// Port of celt/celt_encoder.c:`struct OpusCustomEncoder` (`CELTEncoder`).
///
/// Field names follow C (snake_case). C `int` flags stay `i32` so the CTL semantics (any
/// non-zero value) are preserved.
#[derive(Debug, Clone)]
pub struct CeltEncoder {
    /// Mode used by the encoder.
    pub mode: Cow<'static, CeltMode>,
    pub channels: i32,
    pub stream_channels: i32,

    pub force_intra: i32,
    pub clip: i32,
    pub disable_pf: i32,
    pub complexity: i32,
    pub upsample: i32,
    pub start: i32,
    pub end: i32,

    pub bitrate: i32,
    pub vbr: i32,
    pub signalling: i32,
    /// If zero, VBR can do whatever it likes with the rate.
    pub constrained_vbr: i32,
    pub loss_rate: i32,
    pub lsb_depth: i32,
    pub lfe: i32,
    pub disable_inv: i32,
    #[cfg(feature = "qext")]
    pub enable_qext: i32,
    #[cfg(feature = "qext")]
    pub qext_scale: i32,

    // Everything beyond this point gets cleared on a reset (ENCODER_RESET_START).
    pub rng: u32,
    pub spread_decision: i32,
    pub delayed_intra: OpusVal32,
    pub tonal_average: i32,
    pub last_coded_bands: i32,
    pub hf_average: i32,
    pub tapset_decision: i32,

    pub prefilter_period: i32,
    pub prefilter_gain: OpusVal16,
    pub prefilter_tapset: i32,
    // RESYNTH: prefilter_*_old not ported (RESYNTH is never enabled).
    pub consec_transient: i32,
    pub analysis: AnalysisInfo,
    pub silk_info: SilkInfo,

    pub preemph_mem_e: [OpusVal32; 2],
    pub preemph_mem_d: [OpusVal32; 2],

    // VBR-related parameters
    pub vbr_reservoir: i32,
    pub vbr_drift: i32,
    pub vbr_offset: i32,
    pub vbr_count: i32,
    pub overlap_max: OpusVal32,
    pub stereo_saving: OpusVal16,
    pub intensity: i32,
    /// `energy_mask != NULL`.
    pub has_energy_mask: bool,
    /// Copy of the `energy_mask` values (`channels*nbEBands`, zero padded).
    pub energy_mask: Vec<CeltGlog>,
    pub spec_avg: CeltGlog,

    // RESYNTH: syn_mem not ported.
    /// `in_mem`: `channels*overlap`.
    pub in_mem: Vec<CeltSig>,
    /// `prefilter_mem`: `channels*QEXT_SCALE(COMBFILTER_MAXPERIOD)`.
    pub prefilter_mem: Vec<CeltSig>,
    /// `oldBandE`: `channels*nbEBands`.
    pub old_band_e: Vec<CeltGlog>,
    /// `oldLogE`: `channels*nbEBands`.
    pub old_log_e: Vec<CeltGlog>,
    /// `oldLogE2`: `channels*nbEBands`.
    pub old_log_e2: Vec<CeltGlog>,
    /// `energyError`: `channels*nbEBands`.
    pub energy_error: Vec<CeltGlog>,
    /// QEXT band energies (`qext_oldBandE`, stored after `energyError` in C):
    /// `channels*NB_QEXT_BANDS`.
    #[cfg(feature = "qext")]
    pub qext_old_band_e: Vec<CeltGlog>,

    /// `compute_qext_mode(mode)`, when the mode supports QEXT.
    #[cfg(feature = "qext")]
    qext_mode: Option<CeltMode>,
    scratch: EncScratch,
}

/// Returns `QEXT_SCALE` for a mode (2 for the 96 kHz QEXT modes, else 1).
#[cfg(feature = "qext")]
const fn mode_qext_scale(mode: &CeltMode) -> i32 {
    if mode.fs == 96000 && (mode.short_mdct_size == 240 || mode.short_mdct_size == 180) {
        2
    } else {
        1
    }
}

/// Port of celt/celt_encoder.c:celt_encoder_get_size: memory footprint of an encoder for the
/// default mode (the 96 kHz mode with QEXT). The value is the Rust footprint (struct plus owned
/// state arrays), not the C `sizeof`.
#[must_use]
pub fn celt_encoder_get_size(channels: i32) -> i32 {
    #[cfg(feature = "qext")]
    let mode = opus_custom_mode_create(96000, 1920);
    #[cfg(not(feature = "qext"))]
    let mode = opus_custom_mode_create(48000, 960);
    match mode {
        Ok(m) => opus_custom_encoder_get_size(m, channels),
        Err(_) => 0,
    }
}

/// Port of celt/celt_encoder.c:opus_custom_encoder_get_size (see [`celt_encoder_get_size`]).
#[must_use]
pub fn opus_custom_encoder_get_size(mode: &CeltMode, channels: i32) -> i32 {
    #[cfg(not(feature = "qext"))]
    let extra = 0usize;
    #[cfg(feature = "qext")]
    let qext_scale = mode_qext_scale(mode) as usize;
    #[cfg(not(feature = "qext"))]
    let qext_scale = 1usize;
    #[cfg(feature = "qext")]
    let extra = channels.max(0) as usize * QEXT_BANDS * size_of::<CeltGlog>();
    let ch = channels.max(0) as usize;
    let size = size_of::<CeltEncoder>()
        + ch * mode.overlap as usize * size_of::<CeltSig>()
        + ch * qext_scale * COMBFILTER_MAXPERIOD as usize * size_of::<CeltSig>()
        + 4 * ch * mode.nb_ebands as usize * size_of::<CeltGlog>()
        + extra;
    size as i32
}

impl CeltEncoder {
    /// Port of celt/celt_encoder.c:opus_custom_encoder_init_arch: allocates and initialises an
    /// encoder for `mode` (C `OPUS_CLEAR` + defaults + `OPUS_RESET_STATE`).
    ///
    /// # Errors
    /// [`Error::BadArg`] if `channels` is not 1 or 2 (C accepts 0, which cannot encode; the
    /// port rejects it too).
    pub fn opus_custom_encoder_init_arch(
        mode: Cow<'static, CeltMode>,
        channels: i32,
    ) -> Result<Self> {
        if !(1..=2).contains(&channels) {
            return Err(Error::BadArg);
        }
        let ch = channels as usize;
        let nb = mode.nb_ebands as usize;
        let overlap = mode.overlap as usize;
        #[cfg(feature = "qext")]
        let qext_scale = mode_qext_scale(&mode);
        #[cfg(not(feature = "qext"))]
        let qext_scale = 1;
        let max_period = (qext_scale * COMBFILTER_MAXPERIOD) as usize;
        let n_max = (mode.short_mdct_size << mode.max_lm) as usize;
        let nbq = nb + QEXT_BANDS;
        let last_band = ((i32::from(mode.e_bands[nb]) - i32::from(mode.e_bands[nb - 1]))
            << mode.max_lm) as usize;
        let x_tail = ((i32::from(mode.e_bands[nb]) << mode.max_lm) as usize).saturating_sub(n_max);
        #[cfg(feature = "qext")]
        let qext_mode = if (mode.fs == 48000 || mode.fs == 96000)
            && (mode.short_mdct_size == 120 * qext_scale || mode.short_mdct_size == 90 * qext_scale)
        {
            Some(compute_qext_mode(&mode))
        } else {
            None
        };
        let scratch = EncScratch {
            input: vec![0.0; ch * (n_max + overlap)],
            freq: vec![0.0; ch * n_max],
            // Custom modes whose last bands lie above the MDCT size (end > effEBands) make C
            // read past `X` (undefined behaviour); the tail keeps the port in bounds.
            x: vec![0.0; ch * n_max + x_tail],
            band_e: vec![0.0; ch * nb],
            band_log_e: vec![0.0; ch * nb],
            band_log_e2: vec![0.0; ch * nb],
            error: vec![0.0; ch * nb],
            surround_dynalloc: vec![0.0; ch * nb],
            offsets: vec![0; nb],
            importance: vec![0; nb],
            spread_weight: vec![0; nb],
            tf_res: vec![0; nb],
            cap: vec![0; nb],
            fine_quant: vec![0; nbq],
            pulses: vec![0; nbq],
            fine_priority: vec![0; nbq],
            collapse_masks: vec![0; ch * nb],
            tmp: vec![0.0; n_max + overlap],
            pre: vec![0.0; ch * (n_max + max_period)],
            pitch_buf: vec![0.0; (max_period + n_max) >> 1],
            tf_metric: vec![0; nb],
            tf_path0: vec![0; nb],
            tf_path1: vec![0; nb],
            tf_tmp: vec![0.0; last_band],
            tf_tmp1: vec![0.0; last_band],
            dyn_bufs: DynallocScratch::new(nb, ch),
            bands: BandsScratch::new(),
            #[cfg(feature = "custom-modes")]
            pcm_conv: Vec::new(),
            #[cfg(feature = "qext")]
            extra_quant: vec![0; nbq],
            #[cfg(feature = "qext")]
            extra_pulses: vec![0; nbq],
            #[cfg(feature = "qext")]
            error_bak: vec![0.0; ch * nb],
            #[cfg(feature = "qext")]
            zeros: vec![0; nb.max(QEXT_BANDS)],
        };
        let mut st = Self {
            channels,
            stream_channels: channels,
            force_intra: 0,
            clip: 1,
            disable_pf: 0,
            complexity: 5,
            upsample: 1,
            start: 0,
            end: mode.eff_ebands,
            bitrate: OPUS_BITRATE_MAX,
            vbr: 0,
            signalling: 1,
            constrained_vbr: 1,
            loss_rate: 0,
            lsb_depth: 24,
            lfe: 0,
            disable_inv: 0,
            #[cfg(feature = "qext")]
            enable_qext: 0,
            #[cfg(feature = "qext")]
            qext_scale,
            rng: 0,
            spread_decision: 0,
            delayed_intra: 0.0,
            tonal_average: 0,
            last_coded_bands: 0,
            hf_average: 0,
            tapset_decision: 0,
            prefilter_period: 0,
            prefilter_gain: 0.0,
            prefilter_tapset: 0,
            consec_transient: 0,
            analysis: AnalysisInfo::default(),
            silk_info: SilkInfo::default(),
            preemph_mem_e: [0.0; 2],
            preemph_mem_d: [0.0; 2],
            vbr_reservoir: 0,
            vbr_drift: 0,
            vbr_offset: 0,
            vbr_count: 0,
            overlap_max: 0.0,
            stereo_saving: 0.0,
            intensity: 0,
            has_energy_mask: false,
            energy_mask: vec![0.0; ch * nb],
            spec_avg: 0.0,
            in_mem: vec![0.0; ch * overlap],
            prefilter_mem: vec![0.0; ch * max_period],
            old_band_e: vec![0.0; ch * nb],
            old_log_e: vec![0.0; ch * nb],
            old_log_e2: vec![0.0; ch * nb],
            energy_error: vec![0.0; ch * nb],
            #[cfg(feature = "qext")]
            qext_old_band_e: vec![0.0; ch * QEXT_BANDS],
            #[cfg(feature = "qext")]
            qext_mode,
            scratch,
            mode,
        };
        st.reset();
        Ok(st)
    }

    /// Port of celt/celt_encoder.c:opus_custom_encoder_init (custom modes / Opus custom API).
    ///
    /// # Errors
    /// See [`Self::opus_custom_encoder_init_arch`].
    #[cfg(feature = "custom-modes")]
    pub fn opus_custom_encoder_init(mode: Cow<'static, CeltMode>, channels: i32) -> Result<Self> {
        Self::opus_custom_encoder_init_arch(mode, channels)
    }

    /// Port of celt/celt_encoder.c:opus_custom_encoder_create (custom modes / Opus custom API).
    ///
    /// # Errors
    /// See [`Self::opus_custom_encoder_init_arch`].
    #[cfg(feature = "custom-modes")]
    pub fn opus_custom_encoder_create(mode: Cow<'static, CeltMode>, channels: i32) -> Result<Self> {
        Self::opus_custom_encoder_init_arch(mode, channels)
    }

    /// Port of celt/celt_encoder.c:celt_encoder_init: an encoder for the Opus 48 kHz mode (or
    /// the 96 kHz mode with QEXT) running at `sampling_rate`.
    ///
    /// # Errors
    /// [`Error::BadArg`] for bad channel counts or unsupported rates (C asserts on the latter).
    pub fn celt_encoder_init(sampling_rate: i32, channels: i32) -> Result<Self> {
        #[cfg(feature = "qext")]
        if sampling_rate == 96000 {
            let mut st = Self::opus_custom_encoder_init_arch(
                Cow::Borrowed(opus_custom_mode_create(96000, 1920)?),
                channels,
            )?;
            st.upsample = 1;
            return Ok(st);
        }
        let upsample = match sampling_rate {
            48000 | 24000 | 16000 | 12000 | 8000 => resampling_factor(sampling_rate),
            _ => return Err(Error::BadArg),
        };
        let mut st = Self::opus_custom_encoder_init_arch(
            Cow::Borrowed(opus_custom_mode_create(48000, 960)?),
            channels,
        )?;
        st.upsample = upsample;
        Ok(st)
    }

    /// `QEXT_SCALE(1)` for this encoder.
    #[inline(always)]
    const fn qext_scale(&self) -> i32 {
        #[cfg(feature = "qext")]
        {
            self.qext_scale
        }
        #[cfg(not(feature = "qext"))]
        {
            1
        }
    }

    // ---------------------------------------------------------------------------------------
    // CTLs (celt/celt_encoder.c:opus_custom_encoder_ctl)
    // ---------------------------------------------------------------------------------------

    /// `OPUS_SET_COMPLEXITY`.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `0..=10`.
    pub const fn set_complexity(&mut self, value: i32) -> Result<()> {
        if value < 0 || value > 10 {
            return Err(Error::BadArg);
        }
        self.complexity = value;
        Ok(())
    }

    /// `CELT_SET_START_BAND`.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `0..nbEBands`.
    pub fn set_start_band(&mut self, value: i32) -> Result<()> {
        if value < 0 || value >= self.mode.nb_ebands {
            return Err(Error::BadArg);
        }
        self.start = value;
        Ok(())
    }

    /// `CELT_SET_END_BAND`.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `1..=nbEBands`.
    pub fn set_end_band(&mut self, value: i32) -> Result<()> {
        if value < 1 || value > self.mode.nb_ebands {
            return Err(Error::BadArg);
        }
        self.end = value;
        Ok(())
    }

    /// `CELT_SET_PREDICTION` (0: independent frames, 1: short-term prediction only,
    /// 2: long-term prediction allowed).
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `0..=2`.
    pub const fn set_prediction(&mut self, value: i32) -> Result<()> {
        if value < 0 || value > 2 {
            return Err(Error::BadArg);
        }
        self.disable_pf = (value <= 1) as i32;
        self.force_intra = (value == 0) as i32;
        Ok(())
    }

    /// `OPUS_SET_PACKET_LOSS_PERC`.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `0..=100`.
    pub const fn set_packet_loss_perc(&mut self, value: i32) -> Result<()> {
        if value < 0 || value > 100 {
            return Err(Error::BadArg);
        }
        self.loss_rate = value;
        Ok(())
    }

    /// `OPUS_SET_VBR_CONSTRAINT`.
    pub const fn set_vbr_constraint(&mut self, value: i32) {
        self.constrained_vbr = value;
    }

    /// `OPUS_SET_VBR`.
    pub const fn set_vbr(&mut self, value: i32) {
        self.vbr = value;
    }

    /// `OPUS_SET_BITRATE` (`OPUS_BITRATE_MAX` = -1 allowed).
    ///
    /// # Errors
    /// [`Error::BadArg`] for values `<= 500` other than `OPUS_BITRATE_MAX`.
    pub const fn set_bitrate(&mut self, value: i32) -> Result<()> {
        if value <= 500 && value != OPUS_BITRATE_MAX {
            return Err(Error::BadArg);
        }
        self.bitrate = imin(value, 750_000 * self.channels);
        Ok(())
    }

    /// `CELT_SET_CHANNELS` (stream channels).
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `1..=2`.
    pub const fn set_channels(&mut self, value: i32) -> Result<()> {
        if value < 1 || value > 2 {
            return Err(Error::BadArg);
        }
        self.stream_channels = value;
        Ok(())
    }

    /// `OPUS_SET_LSB_DEPTH`.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `8..=24`.
    pub const fn set_lsb_depth(&mut self, value: i32) -> Result<()> {
        if value < 8 || value > 24 {
            return Err(Error::BadArg);
        }
        self.lsb_depth = value;
        Ok(())
    }

    /// `OPUS_GET_LSB_DEPTH`.
    #[must_use]
    pub const fn lsb_depth(&self) -> i32 {
        self.lsb_depth
    }

    /// `OPUS_SET_PHASE_INVERSION_DISABLED`.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `0..=1`.
    pub const fn set_phase_inversion_disabled(&mut self, value: i32) -> Result<()> {
        if value < 0 || value > 1 {
            return Err(Error::BadArg);
        }
        self.disable_inv = value;
        Ok(())
    }

    /// `OPUS_GET_PHASE_INVERSION_DISABLED`.
    #[must_use]
    pub const fn phase_inversion_disabled(&self) -> i32 {
        self.disable_inv
    }

    /// `OPUS_SET_QEXT`.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `0..=1`.
    #[cfg(feature = "qext")]
    pub const fn set_qext(&mut self, value: i32) -> Result<()> {
        if value < 0 || value > 1 {
            return Err(Error::BadArg);
        }
        self.enable_qext = value;
        Ok(())
    }

    /// `OPUS_GET_QEXT`.
    #[cfg(feature = "qext")]
    #[must_use]
    pub const fn qext(&self) -> i32 {
        self.enable_qext
    }

    /// `OPUS_RESET_STATE`: clears everything from `rng` on (C `ENCODER_RESET_START`), including
    /// the trailing state arrays and the energy mask pointer, then sets the reset defaults.
    pub fn reset(&mut self) {
        self.rng = 0;
        self.spread_decision = 0;
        self.delayed_intra = 0.0;
        self.tonal_average = 0;
        self.last_coded_bands = 0;
        self.hf_average = 0;
        self.tapset_decision = 0;
        self.prefilter_period = 0;
        self.prefilter_gain = 0.0;
        self.prefilter_tapset = 0;
        self.consec_transient = 0;
        self.analysis = AnalysisInfo::default();
        self.silk_info = SilkInfo::default();
        self.preemph_mem_e = [0.0; 2];
        self.preemph_mem_d = [0.0; 2];
        self.vbr_reservoir = 0;
        self.vbr_drift = 0;
        self.vbr_offset = 0;
        self.vbr_count = 0;
        self.overlap_max = 0.0;
        self.stereo_saving = 0.0;
        self.intensity = 0;
        self.has_energy_mask = false;
        self.energy_mask.fill(0.0);
        self.spec_avg = 0.0;
        self.in_mem.fill(0.0);
        self.prefilter_mem.fill(0.0);
        self.old_band_e.fill(0.0);
        self.energy_error.fill(0.0);
        #[cfg(feature = "qext")]
        self.qext_old_band_e.fill(0.0);
        self.old_log_e.fill(-28.0);
        self.old_log_e2.fill(-28.0);
        self.vbr_offset = 0;
        self.delayed_intra = 1.0;
        self.spread_decision = SPREAD_NORMAL;
        self.tonal_average = 256;
        self.hf_average = 0;
        self.tapset_decision = 0;
    }

    /// `CELT_SET_INPUT_CLIPPING` (custom modes / Opus custom API).
    #[cfg(feature = "custom-modes")]
    pub const fn set_input_clipping(&mut self, value: i32) {
        self.clip = value;
    }

    /// `CELT_SET_SIGNALLING`. Without custom modes only 0 is valid when encoding (C asserts).
    pub const fn set_signalling(&mut self, value: i32) {
        self.signalling = value;
    }

    /// `CELT_SET_ANALYSIS` (`None` = C `NULL`, ignored).
    pub const fn set_analysis(&mut self, info: Option<&AnalysisInfo>) {
        if let Some(i) = info {
            self.analysis = *i;
        }
    }

    /// `CELT_SET_SILK_INFO` (`None` = C `NULL`, ignored).
    pub const fn set_silk_info(&mut self, info: Option<&SilkInfo>) {
        if let Some(i) = info {
            self.silk_info = *i;
        }
    }

    /// `CELT_GET_MODE`.
    #[must_use]
    pub fn mode(&self) -> &CeltMode {
        &self.mode
    }

    /// `OPUS_GET_FINAL_RANGE`.
    #[must_use]
    pub const fn final_range(&self) -> u32 {
        self.rng
    }

    /// `OPUS_SET_LFE`.
    pub const fn set_lfe(&mut self, value: i32) {
        self.lfe = value;
    }

    /// `OPUS_SET_ENERGY_MASK`: `None` is C `NULL`. The values are copied (C stores the
    /// pointer; see the module docs). Entries beyond `mask.len()` read as zero.
    pub fn set_energy_mask(&mut self, mask: Option<&[CeltGlog]>) {
        match mask {
            None => self.has_energy_mask = false,
            Some(m) => {
                let n = m.len().min(self.energy_mask.len());
                self.energy_mask[..n].copy_from_slice(&m[..n]);
                self.energy_mask[n..].fill(0.0);
                self.has_energy_mask = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Analysis helpers
// ---------------------------------------------------------------------------------------------

/// Table of 6*64/x, trained on real data to minimize the average error (`inv_table`).
static INV_TABLE: [u8; 128] = [
    255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, 23, 22, 21, 20, 19, 18, 17,
    16, 16, 15, 15, 14, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8, 8,
    8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
];

/// Port of celt/celt_encoder.c:transient_analysis.
///
/// `input` holds `c` channels of `len` samples. `tmp` is scratch of at least `len` samples.
/// Returns `is_transient`; writes `tf_estimate`, `tf_chan` and `weak_transient`.
pub fn transient_analysis(
    input: &[OpusVal32],
    len: i32,
    c: i32,
    tf_estimate: &mut OpusVal16,
    tf_chan: &mut i32,
    allow_weak_transients: bool,
    weak_transient: &mut bool,
    tone_freq: OpusVal16,
    toneishness: OpusVal32,
    tmp: &mut [OpusVal16],
) -> bool {
    let mut mask_metric: i32 = 0;
    // Forward masking: 6.7 dB/ms.
    // FIXED_POINT: forward_shift not ported (float build).
    let mut forward_decay: OpusVal16 = 0.0625f32;
    let lenu = len as usize;
    let tmp = &mut tmp[..lenu];

    *weak_transient = false;
    // For lower bitrates, let's be more conservative and have a forward masking decay of
    // 3.3 dB/ms. This avoids having to code transients at very low bitrate (mostly for
    // hybrid), which can result in unstable energy and/or partial collapse.
    if allow_weak_transients {
        forward_decay = 0.03125f32;
    }
    let len2 = len / 2;
    let len2u = len2 as usize;
    for ch in 0..c {
        let mut unmask: i32 = 0;
        let mut mem0: OpusVal32 = 0.0;
        let mut mem1: OpusVal32 = 0.0;
        let x_in = &input[(ch * len) as usize..(ch * len) as usize + lenu];
        // High-pass filter: (1 - 2*z^-1 + z^-2) / (1 - z^-1 + .5*z^-2)
        for i in 0..lenu {
            let x: OpusVal32 = x_in[i];
            let y: OpusVal32 = mem0 + x;
            // Modified code to shorten dependency chains.
            let mem00 = mem0;
            mem0 = mem0 - x + 0.5f32 * mem1;
            mem1 = x - mem00;
            tmp[i] = y;
        }
        // First few samples are bad because we don't propagate the memory
        tmp[..12].fill(0.0);
        // FIXED_POINT: normalisation of tmp not ported (float build).

        let mut mean: OpusVal32 = 0.0;
        mem0 = 0.0;
        // Grouping by two to reduce complexity
        // Forward pass to compute the post-echo threshold
        for i in 0..len2u {
            let x2: OpusVal32 = tmp[2 * i] * tmp[2 * i] + tmp[2 * i + 1] * tmp[2 * i + 1];
            mean += x2;
            mem0 = x2 + (1.0f32 - forward_decay) * mem0;
            tmp[i] = forward_decay * mem0;
        }

        mem0 = 0.0;
        let mut max_e: OpusVal16 = 0.0;
        // Backward pass to compute the pre-echo threshold
        for i in (0..len2u).rev() {
            // Backward masking: 13.9 dB/ms.
            mem0 = tmp[i] + 0.875f32 * mem0;
            tmp[i] = 0.125f32 * mem0;
            max_e = max16(max_e, 0.125f32 * mem0);
        }

        // Compute the ratio of the "frame energy" over the harmonic mean of the energy. As a
        // compromise with the old transient detector, frame energy is the geometric mean of
        // the energy and half the max.
        // C: celt_sqrt(mean * maxE*.5*len2) (double arithmetic after the first product).
        mean = math::sqrt(f64::from(mean * max_e) * 0.5 * f64::from(len2)) as f32;
        // Inverse of the mean energy in Q15+6
        let norm: OpusVal32 = len2 as f32 / (EPSILON + mean);
        // We should never see NaNs here (C aborts with hardening).
        debug_assert!(!tmp[0].is_nan());
        debug_assert!(!norm.is_nan());
        let mut i = 12usize;
        while (i as i32) < len2 - 5 {
            // C: (int)MAX32(0,MIN32(127,floor(64*norm*(tmp[i]+EPSILON)))) in double.
            let f = math::floor(f64::from(64.0f32 * norm * (tmp[i] + EPSILON)));
            let f = if 127.0 < f { 127.0 } else { f };
            let f = if 0.0 > f { 0.0 } else { f };
            let id = f as i32;
            unmask += i32::from(INV_TABLE[id as usize]);
            i += 4;
        }
        // Normalize, compensate for the 1/4th of the sample and the factor of 6 in the inverse
        // table
        unmask = 64 * unmask * 4 / (6 * (len2 - 17));
        if unmask > mask_metric {
            *tf_chan = ch;
            mask_metric = unmask;
        }
    }
    let mut is_transient = mask_metric > 200;
    // Prevent the transient detector from confusing the partial cycle of a very low frequency
    // tone with a transient.
    if toneishness > 0.98f32 && tone_freq < 0.026f32 {
        is_transient = false;
        mask_metric = 0;
    }
    // For low bitrates, define "weak transients" that need to be handled differently to avoid
    // partial collapse.
    if allow_weak_transients && is_transient && mask_metric < 600 {
        is_transient = false;
        *weak_transient = true;
    }
    // Arbitrary metric for VBR boost
    let tf_max: OpusVal16 = max16(0.0, math::sqrt(f64::from(27 * mask_metric)) as f32 - 42.0);
    // *tf_estimate = 1 + MIN16(1, sqrt(MAX16(0, tf_max-30))/20);
    // C: celt_sqrt(MAX32(0, (float)0.0069*MIN16(163,tf_max) - 0.139)) (double subtraction).
    let t = f64::from(0.0069f64 as f32 * min16(163.0, tf_max)) - 0.139;
    let t = if 0.0 > t { 0.0 } else { t };
    *tf_estimate = math::sqrt(t) as f32;
    is_transient
}

/// Port of celt/celt_encoder.c:patch_transient_decision: looks for sudden increases of energy
/// to decide whether we need to patch the transient decision.
#[must_use]
pub fn patch_transient_decision(
    new_e: &[CeltGlog],
    old_e: &[CeltGlog],
    nb_ebands: i32,
    start: i32,
    end: i32,
    c: i32,
) -> bool {
    // C: celt_glog spread_old[26] (sized larger here for custom modes).
    let mut spread_old = [0.0f32; 64];
    let nb = nb_ebands as usize;
    let (start, end) = (start as usize, end as usize);
    // Apply an aggressive (-6 dB/Bark) spreading function to the old frame to avoid false
    // detection caused by irrelevant bands
    if c == 1 {
        spread_old[start] = old_e[start];
        for i in start + 1..end {
            spread_old[i] = maxg(spread_old[i - 1] - 1.0f32, old_e[i]);
        }
    } else {
        spread_old[start] = maxg(old_e[start], old_e[start + nb]);
        for i in start + 1..end {
            spread_old[i] = maxg(spread_old[i - 1] - 1.0f32, maxg(old_e[i], old_e[i + nb]));
        }
    }
    for i in (start..end.saturating_sub(1)).rev() {
        spread_old[i] = maxg(spread_old[i], spread_old[i + 1] - 1.0f32);
    }
    // Compute mean increase
    let mut mean_diff: OpusVal32 = 0.0;
    let i0 = imax(2, start as i32);
    for ch in 0..c as usize {
        for i in i0..end as i32 - 1 {
            let iu = i as usize;
            let x1: OpusVal16 = maxg(0.0, new_e[iu + ch * nb]);
            let x2: OpusVal16 = maxg(0.0, spread_old[iu]);
            mean_diff += maxg(0.0, x1 - x2);
        }
    }
    mean_diff /= (c * (end as i32 - 1 - i0)) as f32;
    mean_diff > 1.0f32
}

/// Port of celt/celt_encoder.c:compute_mdcts: applies the window and computes the MDCT for all
/// sub-frames and all channels of a frame. `short_blocks` is the C `shortBlocks` (0 or `M`).
pub fn compute_mdcts(
    mode: &CeltMode,
    short_blocks: i32,
    input: &[CeltSig],
    out: &mut [CeltSig],
    c: i32,
    cc: i32,
    lm: i32,
    upsample: i32,
) {
    let overlap = mode.overlap as usize;
    let (b_cnt, n, shift) = if short_blocks != 0 {
        (short_blocks, mode.short_mdct_size, mode.max_lm)
    } else {
        (1, mode.short_mdct_size << lm, mode.max_lm - lm)
    };
    let (bu, nu) = (b_cnt as usize, n as usize);
    for ch in 0..cc as usize {
        for b in 0..bu {
            // Interleaving the sub-frames while doing the MDCTs
            clt_mdct_forward(
                &mode.mdct,
                &input[ch * (bu * nu + overlap) + b * nu..],
                &mut out[b + ch * nu * bu..],
                &mode.window,
                overlap,
                shift as usize,
                bu,
            );
        }
    }
    let bn = bu * nu;
    if cc == 2 && c == 1 {
        for i in 0..bn {
            out[i] = half32(out[i]) + half32(out[bn + i]);
        }
    }
    if upsample != 1 {
        for ch in 0..c as usize {
            let bound = (b_cnt * n / upsample) as usize;
            let o = &mut out[ch * bn..(ch + 1) * bn];
            for v in &mut o[..bound] {
                *v *= upsample as f32;
            }
            o[bound..].fill(0.0);
        }
    }
}

/// Port of celt/celt_encoder.c:celt_preemphasis.
///
/// `pcmp` starts at the channel's first sample (C `pcm+c`) and is read with stride `cc`;
/// `inp` receives `n` pre-emphasised samples. `coef` is `mode->preemph`.
pub fn celt_preemphasis(
    pcmp: &[OpusRes],
    inp: &mut [CeltSig],
    n: i32,
    cc: i32,
    upsample: i32,
    coef: &[OpusVal16; 4],
    mem: &mut CeltSig,
    clip: bool,
) {
    let coef0 = coef[0];
    let mut m = *mem;
    let nn = n as usize;
    let ccu = cc as usize;
    let inp = &mut inp[..nn];

    // Fast path for the normal 48kHz case and no clipping
    if coef[1] == 0.0 && upsample == 1 && !clip {
        for i in 0..nn {
            let x: CeltSig = res2sig(pcmp[ccu * i]);
            // Apply pre-emphasis
            inp[i] = x - m;
            m = coef0 * x;
        }
        *mem = m;
        return;
    }

    let nu = (n / upsample) as usize;
    let up = upsample as usize;
    if upsample != 1 {
        inp.fill(0.0);
    }
    for i in 0..nu {
        inp[i * up] = res2sig(pcmp[ccu * i]);
    }

    if clip {
        // Clip input to avoid encoding non-portable files
        for i in 0..nu {
            inp[i * up] = max32(-65536.0f32, min32(65536.0f32, inp[i * up]));
        }
    }
    #[cfg(any(feature = "custom-modes", feature = "qext"))]
    if coef[1] != 0.0 {
        let coef1 = coef[1];
        // FIXED_POINT && ENABLE_QEXT: coef2_q30 not ported (float build).
        let coef2 = coef[2];
        for i in 0..nn {
            let x: CeltSig = inp[i];
            // Apply pre-emphasis
            let tmp: CeltSig = coef2 * x;
            inp[i] = tmp + m;
            m = coef1 * inp[i] - coef0 * tmp;
        }
        *mem = m;
        return;
    }
    for i in 0..nn {
        let x: CeltSig = inp[i];
        // Apply pre-emphasis
        inp[i] = x - m;
        m = coef0 * x;
    }
    *mem = m;
}

/// Port of celt/celt_encoder.c:l1_metric.
#[must_use]
pub fn l1_metric(tmp: &[CeltNorm], n: i32, lm: i32, bias: OpusVal16) -> OpusVal32 {
    let mut l1: OpusVal32 = 0.0;
    for &v in &tmp[..n as usize] {
        l1 += abs16(v);
    }
    // When in doubt, prefer good freq resolution
    l1 + (lm as f32 * bias) * l1
}

/// Port of celt/celt_encoder.c:tf_analysis. Returns `tf_select`; writes `tf_res[..len]`.
///
/// `metric`, `path0`, `path1` need `len` entries; `tmp`, `tmp_1` the width of the last band
/// (`(eBands[len]-eBands[len-1])<<LM`).
pub fn tf_analysis(
    m: &CeltMode,
    len: i32,
    is_transient: bool,
    tf_res: &mut [i32],
    lambda: i32,
    x: &[CeltNorm],
    n0: i32,
    lm: i32,
    tf_estimate: OpusVal16,
    tf_chan: i32,
    importance: &[i32],
    metric: &mut [i32],
    path0: &mut [i32],
    path1: &mut [i32],
    tmp: &mut [CeltNorm],
    tmp_1: &mut [CeltNorm],
) -> i32 {
    let e_bands = &m.e_bands;
    let eb = |i: i32| i32::from(e_bands[i as usize]);
    let it = is_transient as i32;
    let bias: OpusVal16 = 0.04f32 * max16(-0.25f32, 0.5f32 - tf_estimate);
    let lenu = len as usize;

    for i in 0..len {
        let iu = i as usize;
        let mut best_level: i32 = 0;
        let n = (eb(i + 1) - eb(i)) << lm;
        let nu = n as usize;
        // band is too narrow to be split down to LM=-1
        let narrow = (eb(i + 1) - eb(i)) == 1;
        let off = (tf_chan * n0 + (eb(i) << lm)) as usize;
        tmp[..nu].copy_from_slice(&x[off..off + nu]);
        // Just add the right channel if we're in stereo
        let mut l1 = l1_metric(tmp, n, if is_transient { lm } else { 0 }, bias);
        let mut best_l1 = l1;
        // Check the -1 case for transients
        if is_transient && !narrow {
            tmp_1[..nu].copy_from_slice(&tmp[..nu]);
            haar1(tmp_1, n >> lm, 1 << lm);
            l1 = l1_metric(tmp_1, n, lm + 1, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = -1;
            }
        }
        let kmax = lm + i32::from(!(is_transient || narrow));
        for k in 0..kmax {
            let b = if is_transient { lm - k - 1 } else { k + 1 };
            haar1(tmp, n >> k, 1 << k);
            l1 = l1_metric(tmp, n, b, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = k + 1;
            }
        }
        // metric is in Q1 to be able to select the mid-point (-0.5) for narrower bands
        metric[iu] = if is_transient {
            2 * best_level
        } else {
            -2 * best_level
        };
        // For bands that can't be split to -1, set the metric to the half-way point to avoid
        // biasing the decision
        if narrow && (metric[iu] == 0 || metric[iu] == -2 * lm) {
            metric[iu] -= 1;
        }
    }
    let table = &TF_SELECT_TABLE[lm as usize];
    let tsel = |idx: i32| 2 * i32::from(table[idx as usize]);
    // Search for the optimal tf resolution, including tf_select
    let mut selcost = [0i32; 2];
    for sel in 0..2 {
        let mut cost0 = importance[0] * (metric[0] - tsel(4 * it + 2 * sel)).abs();
        let mut cost1 = importance[0] * (metric[0] - tsel(4 * it + 2 * sel + 1)).abs()
            + if is_transient { 0 } else { lambda };
        for i in 1..lenu {
            let curr0 = imin(cost0, cost1 + lambda);
            let curr1 = imin(cost0 + lambda, cost1);
            cost0 = curr0 + importance[i] * (metric[i] - tsel(4 * it + 2 * sel)).abs();
            cost1 = curr1 + importance[i] * (metric[i] - tsel(4 * it + 2 * sel + 1)).abs();
        }
        cost0 = imin(cost0, cost1);
        selcost[sel as usize] = cost0;
    }
    // For now, we're conservative and only allow tf_select=1 for transients. If tests confirm
    // it's useful for non-transients, we could allow it.
    let mut tf_select: i32 = 0;
    if selcost[1] < selcost[0] && is_transient {
        tf_select = 1;
    }
    let mut cost0 = importance[0] * (metric[0] - tsel(4 * it + 2 * tf_select)).abs();
    let mut cost1 = importance[0] * (metric[0] - tsel(4 * it + 2 * tf_select + 1)).abs()
        + if is_transient { 0 } else { lambda };
    // Viterbi forward pass
    for i in 1..lenu {
        let mut from0 = cost0;
        let mut from1 = cost1 + lambda;
        let curr0 = if from0 < from1 {
            path0[i] = 0;
            from0
        } else {
            path0[i] = 1;
            from1
        };

        from0 = cost0 + lambda;
        from1 = cost1;
        let curr1 = if from0 < from1 {
            path1[i] = 0;
            from0
        } else {
            path1[i] = 1;
            from1
        };
        cost0 = curr0 + importance[i] * (metric[i] - tsel(4 * it + 2 * tf_select)).abs();
        cost1 = curr1 + importance[i] * (metric[i] - tsel(4 * it + 2 * tf_select + 1)).abs();
    }
    tf_res[lenu - 1] = if cost0 < cost1 { 0 } else { 1 };
    // Viterbi backward pass to check the decisions
    for i in (0..lenu - 1).rev() {
        tf_res[i] = if tf_res[i + 1] == 1 {
            path1[i + 1]
        } else {
            path0[i + 1]
        };
    }
    tf_select
}

/// Port of celt/celt_encoder.c:tf_encode.
pub fn tf_encode(
    start: i32,
    end: i32,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: i32,
    mut tf_select: i32,
    enc: &mut EcEnc<'_>,
) {
    let it = is_transient as usize;
    let mut budget: u32 = enc.storage * 8;
    let mut tell: u32 = enc.tell() as u32;
    let mut logp: u32 = if is_transient { 2 } else { 4 };
    // Reserve space to code the tf_select decision.
    // C: tell+logp+1 <= budget.
    let tf_select_rsv = lm > 0 && tell + logp < budget;
    budget -= u32::from(tf_select_rsv);
    let mut curr: i32 = 0;
    let mut tf_changed: i32 = 0;
    for i in start as usize..end as usize {
        if tell + logp <= budget {
            enc.enc_bit_logp((tf_res[i] ^ curr) != 0, logp);
            tell = enc.tell() as u32;
            curr = tf_res[i];
            tf_changed |= curr;
        } else {
            tf_res[i] = curr;
        }
        logp = if is_transient { 4 } else { 5 };
    }
    let table = &TF_SELECT_TABLE[lm as usize];
    let tc = tf_changed as usize;
    // Only code tf_select if it would actually make a difference.
    if tf_select_rsv && table[4 * it + tc] != table[4 * it + 2 + tc] {
        enc.enc_bit_logp(tf_select != 0, 1);
    } else {
        tf_select = 0;
    }
    for i in start as usize..end as usize {
        tf_res[i] = i32::from(table[4 * it + 2 * tf_select as usize + tf_res[i] as usize]);
    }
}

/// Port of celt/celt_encoder.c:alloc_trim_analysis. Returns the allocation trim index.
#[must_use]
pub fn alloc_trim_analysis(
    m: &CeltMode,
    x: &[CeltNorm],
    band_log_e: &[CeltGlog],
    end: i32,
    lm: i32,
    c: i32,
    n0: i32,
    analysis: &AnalysisInfo,
    stereo_saving: &mut OpusVal16,
    tf_estimate: OpusVal16,
    intensity: i32,
    surround_trim: CeltGlog,
    equiv_rate: i32,
) -> i32 {
    let e_bands = &m.e_bands;
    let eb = |i: i32| i32::from(e_bands[i as usize]);
    let mut diff: OpusVal32 = 0.0;
    let mut trim: OpusVal16 = 5.0f32;
    // At low bitrate, reducing the trim seems to help. At higher bitrates, it's less clear
    // what's best, so we're keeping it as it was before, at least for now.
    if equiv_rate < 64000 {
        trim = 4.0f32;
    } else if equiv_rate < 80000 {
        let frac = (equiv_rate - 64000) >> 10;
        trim = 4.0f32 + (1.0f32 / 16.0f32) * frac as f32;
    }
    if c == 2 {
        let mut sum: OpusVal16 = 0.0; // Q10
        // Compute inter-channel correlation for low frequencies
        for i in 0..8 {
            let off = (eb(i) << lm) as usize;
            let partial = celt_inner_prod(
                &x[off..],
                &x[n0 as usize + off..],
                ((eb(i + 1) - eb(i)) << lm) as usize,
            );
            sum += partial;
        }
        sum *= 1.0f32 / 8.0;
        sum = min16(1.0f32, abs16(sum));
        let mut min_xc: OpusVal16 = sum; // Q10
        for i in 8..intensity {
            let off = (eb(i) << lm) as usize;
            let partial = celt_inner_prod(
                &x[off..],
                &x[n0 as usize + off..],
                ((eb(i + 1) - eb(i)) << lm) as usize,
            );
            min_xc = min16(min_xc, abs16(partial));
        }
        min_xc = min16(1.0f32, abs16(min_xc));
        // mid-side savings estimations based on the LF average
        let log_xc: OpusVal16 = celt_log2(1.001f32 - sum * sum);
        // mid-side savings estimations based on min correlation
        let log_xc2: OpusVal16 = max16(half16(log_xc), celt_log2(1.001f32 - min_xc * min_xc));
        // FIXED_POINT: Q20 compensation not ported (float build).

        trim += max16(-4.0f32, 0.75f32 * log_xc);
        *stereo_saving = min16(*stereo_saving + 0.25f32, -half16(log_xc2));
    }

    // Estimate spectral tilt
    let nb = m.nb_ebands as usize;
    for ch in 0..c as usize {
        for i in 0..(end - 1) as usize {
            diff += band_log_e[i + ch * nb] * (2 + 2 * i as i32 - end) as f32;
        }
    }
    diff /= (c * (end - 1)) as f32;
    trim -= max32(-2.0f32, min32(2.0f32, (diff + 1.0f32) / 6.0));
    trim -= surround_trim;
    trim -= 2.0 * tf_estimate;
    if analysis.valid != 0 {
        trim -= max16(
            -2.0f32,
            min16(2.0f32, 2.0f32 * (analysis.tonality_slope + 0.05f32)),
        );
    }

    // FIXED_POINT: PSHR32 rounding not ported (float build).
    let trim_index = floor_i32(f64::from(0.5f32 + trim));
    imax(0, imin(10, trim_index))
}

/// Port of celt/celt_encoder.c:stereo_analysis: whether dual (L/R) stereo is cheaper than M/S.
#[must_use]
pub fn stereo_analysis(m: &CeltMode, x: &[CeltNorm], lm: i32, n0: i32) -> bool {
    let e_bands = &m.e_bands;
    let eb = |i: i32| i32::from(e_bands[i as usize]);
    let mut sum_lr: OpusVal32 = EPSILON;
    let mut sum_ms: OpusVal32 = EPSILON;

    // Use the L1 norm to model the entropy of the L/R signal vs the M/S signal
    for i in 0..13 {
        for j in (eb(i) << lm) as usize..(eb(i + 1) << lm) as usize {
            // We cast to 32-bit first because of the -32768 case
            let l: OpusVal32 = x[j];
            let r: OpusVal32 = x[n0 as usize + j];
            let mm: OpusVal32 = l + r;
            let s: OpusVal32 = l - r;
            sum_lr += abs32(l) + abs32(r);
            sum_ms += abs32(mm) + abs32(s);
        }
    }
    sum_ms *= 0.707107f32;
    let mut thetas = 13;
    // We don't need thetas for lower bands with LM<=1
    if lm <= 1 {
        thetas -= 8;
    }
    (((eb(13) << (lm + 1)) + thetas) as f32 * sum_ms) > ((eb(13) << (lm + 1)) as f32 * sum_lr)
}

/// Port of celt/celt_encoder.c:median_of_5 (`x[0..5]`).
#[must_use]
pub fn median_of_5(x: &[CeltGlog]) -> CeltGlog {
    let (mut t0, mut t1);
    let (mut t3, mut t4);
    let t2 = x[2];
    if x[0] > x[1] {
        t0 = x[1];
        t1 = x[0];
    } else {
        t0 = x[0];
        t1 = x[1];
    }
    if x[3] > x[4] {
        t3 = x[4];
        t4 = x[3];
    } else {
        t3 = x[3];
        t4 = x[4];
    }
    if t0 > t3 {
        core::mem::swap(&mut t0, &mut t3);
        core::mem::swap(&mut t1, &mut t4);
    }
    if t2 > t1 {
        if t1 < t3 { ming(t2, t3) } else { ming(t4, t1) }
    } else if t2 < t3 {
        ming(t1, t3)
    } else {
        ming(t2, t4)
    }
}

/// Port of celt/celt_encoder.c:median_of_3 (`x[0..3]`).
#[must_use]
pub fn median_of_3(x: &[CeltGlog]) -> CeltGlog {
    let (t0, t1) = if x[0] > x[1] {
        (x[1], x[0])
    } else {
        (x[0], x[1])
    };
    let t2 = x[2];
    if t1 < t2 {
        t1
    } else if t0 < t2 {
        t2
    } else {
        t0
    }
}

/// Port of celt/celt_encoder.c:dynalloc_analysis. Returns `maxDepth`; writes `offsets`
/// (`nbEBands`), `importance`, `spread_weight` and `tot_boost_`.
pub fn dynalloc_analysis(
    band_log_e: &[CeltGlog],
    band_log_e2: &[CeltGlog],
    old_band_e: &[CeltGlog],
    nb_ebands: i32,
    start: i32,
    end: i32,
    c: i32,
    offsets: &mut [i32],
    lsb_depth: i32,
    log_n: &[i16],
    is_transient: bool,
    vbr: bool,
    constrained_vbr: bool,
    e_bands: &[i16],
    lm: i32,
    effective_bytes: i32,
    tot_boost_: &mut i32,
    lfe: bool,
    surround_dynalloc: &[CeltGlog],
    analysis: &AnalysisInfo,
    importance: &mut [i32],
    spread_weight: &mut [i32],
    tone_freq: OpusVal16,
    toneishness: OpusVal32,
    qext_scale: i32,
    scratch: &mut DynallocScratch,
) -> CeltGlog {
    let nb = nb_ebands as usize;
    let (startu, endu) = (start as usize, end as usize);
    let eb = |i: usize| i32::from(e_bands[i]);
    let mut tot_boost: i32 = 0;
    let DynallocScratch {
        follower,
        noise_floor,
        band_log_e3,
        mask,
        sig,
    } = scratch;
    offsets[..nb].fill(0);
    // Dynamic allocation code
    let mut max_depth: CeltGlog = -31.9f32;
    for i in 0..endu {
        // Noise floor must take into account eMeans, the depth, the width of the bands and the
        // preemphasis filter (approx. square of bark band ID)
        noise_floor[i] = 0.0625f32 * f32::from(log_n[i]) + 0.5f32 + (9 - lsb_depth) as f32
            - E_MEANS[i]
            + 0.0062f32 * (i as i32 + 5) as f32 * (i as i32 + 5) as f32;
    }
    for ch in 0..c as usize {
        for i in 0..endu {
            max_depth = maxg(max_depth, band_log_e[ch * nb + i] - noise_floor[i]);
        }
    }
    {
        // Compute a really simple masking model to avoid taking into account completely masked
        // bands when computing the spreading decision.
        for i in 0..endu {
            mask[i] = band_log_e[i] - noise_floor[i];
        }
        if c == 2 {
            for i in 0..endu {
                mask[i] = maxg(mask[i], band_log_e[nb + i] - noise_floor[i]);
            }
        }
        sig[..endu].copy_from_slice(&mask[..endu]);
        for i in 1..endu {
            mask[i] = maxg(mask[i], mask[i - 1] - 2.0f32);
        }
        for i in (0..endu.saturating_sub(1)).rev() {
            mask[i] = maxg(mask[i], mask[i + 1] - 3.0f32);
        }
        for i in 0..endu {
            // Compute SMR: Mask is never more than 72 dB below the peak and never below the
            // noise floor.
            let smr: CeltGlog = sig[i] - maxg(maxg(0.0, max_depth - 12.0f32), mask[i]);
            // Clamp SMR to make sure we're not shifting by something negative or too large.
            // FIXED_POINT: PSHR32 variant not ported (float build).
            let shift = imin(5, imax(0, -floor_i32(f64::from(0.5f32 + smr))));
            spread_weight[i] = 32 >> shift;
        }
    }
    // Make sure that dynamic allocation can't make us bust the budget. We enable the feature
    // starting at 24 kb/s for 20-ms frames and 96 kb/s for 2.5 ms frames.
    if effective_bytes >= (30 + 5 * lm) && !lfe {
        let mut last: usize = 0;
        for ch in 0..c as usize {
            band_log_e3[..endu].copy_from_slice(&band_log_e2[ch * nb..ch * nb + endu]);
            if lm == 0 {
                // For 2.5 ms frames, the first 8 bands have just one bin, so the energy is
                // highly unreliable (high variance). For that reason, we take the max with the
                // previous energy so that at least 2 bins are getting used.
                for i in 0..endu.min(8) {
                    band_log_e3[i] = maxg(band_log_e2[ch * nb + i], old_band_e[ch * nb + i]);
                }
            }
            let f = &mut follower[ch * nb..ch * nb + nb];
            f[0] = band_log_e3[0];
            for i in 1..endu {
                // The last band to be at least 3 dB higher than the previous one is the last
                // we'll consider. Otherwise, we run into problems on bandlimited signals.
                if band_log_e3[i] > band_log_e3[i - 1] + 0.5f32 {
                    last = i;
                }
                f[i] = ming(f[i - 1] + 1.5f32, band_log_e3[i]);
            }
            for i in (0..last).rev() {
                f[i] = ming(f[i], ming(f[i + 1] + 2.0f32, band_log_e3[i]));
            }

            // Combine with a median filter to avoid dynalloc triggering unnecessarily. The
            // "offset" value controls how conservative we are -- a higher offset reduces the
            // impact of the median filter and makes dynalloc use more bits.
            let offset: CeltGlog = 1.0f32;
            for i in 2..endu.saturating_sub(2) {
                f[i] = maxg(f[i], median_of_5(&band_log_e3[i - 2..]) - offset);
            }
            let mut tmp = median_of_3(&band_log_e3[0..]) - offset;
            f[0] = maxg(f[0], tmp);
            f[1] = maxg(f[1], tmp);
            tmp = median_of_3(&band_log_e3[endu - 3..]) - offset;
            f[endu - 2] = maxg(f[endu - 2], tmp);
            f[endu - 1] = maxg(f[endu - 1], tmp);

            for i in 0..endu {
                f[i] = maxg(f[i], noise_floor[i]);
            }
        }
        if c == 2 {
            for i in startu..endu {
                // Consider 24 dB "cross-talk"
                follower[nb + i] = maxg(follower[nb + i], follower[i] - 4.0f32);
                follower[i] = maxg(follower[i], follower[nb + i] - 4.0f32);
                follower[i] = half32(
                    maxg(0.0, band_log_e[i] - follower[i])
                        + maxg(0.0, band_log_e[nb + i] - follower[nb + i]),
                );
            }
        } else {
            for i in startu..endu {
                follower[i] = maxg(0.0, band_log_e[i] - follower[i]);
            }
        }
        for i in startu..endu {
            follower[i] = maxg(follower[i], surround_dynalloc[i]);
        }
        for i in startu..endu {
            // FIXED_POINT: PSHR32 variant not ported (float build).
            importance[i] = floor_i32(f64::from(
                0.5f32 + 13.0 * celt_exp2_db(ming(follower[i], 4.0f32)),
            ));
        }
        // For non-transient CBR/CVBR frames, halve the dynalloc contribution
        if (!vbr || constrained_vbr) && !is_transient {
            for i in startu..endu {
                follower[i] = half32(follower[i]);
            }
        }
        for i in startu..endu {
            if i < 8 {
                follower[i] *= 2.0;
            }
            if i >= 12 {
                follower[i] = half32(follower[i]);
            }
        }
        // Compensate for Opus' under-allocation on tones.
        if toneishness > 0.98f32 {
            // FIXED_POINT: integer freq_bin not ported (float build).
            let freq_bin = floor_i32(0.5 + f64::from(qext_scale as f32 * tone_freq * 120.0) / PI);
            for i in startu..endu {
                if freq_bin >= eb(i) && freq_bin <= eb(i + 1) {
                    follower[i] += 2.0f32;
                }
                if freq_bin >= eb(i) - 1 && freq_bin <= eb(i + 1) + 1 {
                    follower[i] += 1.0f32;
                }
                if freq_bin >= eb(i) - 2 && freq_bin <= eb(i + 1) + 2 {
                    follower[i] += 1.0f32;
                }
                if freq_bin >= eb(i) - 3 && freq_bin <= eb(i + 1) + 3 {
                    follower[i] += 0.5f32;
                }
            }
            if freq_bin >= eb(endu) {
                follower[endu - 1] += 2.0f32;
                follower[endu - 2] += 1.0f32;
            }
        }
        if analysis.valid != 0 {
            for i in startu..LEAK_BANDS.min(endu) {
                follower[i] += (1.0f32 / 64.0f32) * f32::from(analysis.leak_boost[i]);
            }
        }
        for i in startu..endu {
            follower[i] = ming(follower[i], 4.0);

            let width = (c * (eb(i + 1) - eb(i))) << lm;
            let (boost, boost_bits) = if width < 6 {
                let boost = follower[i] as i32;
                (boost, (boost * width) << BITRES)
            } else if width > 48 {
                let boost = (follower[i] * 8.0) as i32;
                (boost, ((boost * width) << BITRES) / 8)
            } else {
                let boost = (follower[i] * width as f32 / 6.0) as i32;
                (boost, (boost * 6) << BITRES)
            };
            // For CBR and non-transient CVBR frames, limit dynalloc to 2/3 of the bits
            if (!vbr || (constrained_vbr && !is_transient))
                && ((tot_boost + boost_bits) >> BITRES >> 3) > 2 * effective_bytes / 3
            {
                let cap = (2 * effective_bytes / 3) << BITRES << 3;
                offsets[i] = cap - tot_boost;
                tot_boost = cap;
                break;
            }
            offsets[i] = boost;
            tot_boost += boost_bits;
        }
    } else {
        for i in startu..endu {
            importance[i] = 13;
        }
    }
    *tot_boost_ = tot_boost;
    max_depth
}

// FIXED_POINT: normalize_tone_input and acos_approx not ported (float build).

/// Port of celt/celt_encoder.c:tone_lpc: computes the LPC coefficients using a least-squares
/// fit for both forward and backward prediction. Returns `true` on failure (C returns 1).
pub fn tone_lpc(x: &[OpusVal16], len: i32, delay: i32, lpc: &mut [OpusVal32; 2]) -> bool {
    let (lenu, d) = (len as usize, delay as usize);
    let x = &x[..lenu];
    let mut r00: OpusVal32 = 0.0;
    let mut r01: OpusVal32 = 0.0;
    let mut r02: OpusVal32 = 0.0;
    debug_assert!(len > 2 * delay);
    // Compute correlations as if using the forward prediction covariance method.
    for i in 0..lenu - 2 * d {
        r00 += x[i] * x[i];
        r01 += x[i] * x[i + d];
        r02 += x[i] * x[i + 2 * d];
    }
    let mut edges: OpusVal32 = 0.0;
    for i in 0..d {
        edges += x[lenu + i - 2 * d] * x[lenu + i - 2 * d] - x[i] * x[i];
    }
    let r11 = r00 + edges;
    edges = 0.0;
    for i in 0..d {
        edges += x[lenu + i - d] * x[lenu + i - d] - x[i + d] * x[i + d];
    }
    let r22 = r11 + edges;
    edges = 0.0;
    for i in 0..d {
        edges += x[lenu + i - 2 * d] * x[lenu + i - d] - x[i] * x[i + d];
    }
    let r12 = r01 + edges;
    // Reverse and sum to get the backward contribution.
    let r00b = r00 + r22;
    let r01b = r01 + r12;
    let r11b = 2.0 * r11;
    let r02b = 2.0 * r02;
    let r12b = r12 + r01;
    let (r00, r01, r11, r02, r12) = (r00b, r01b, r11b, r02b, r12b);
    // Solve A*x=b, where A=[r00, r01; r01, r11] and b=[r02; r12].
    let den: OpusVal32 = r00 * r11 - r01 * r01;
    // FIXED_POINT: integer threshold not ported (float build).
    if den < 0.001f32 * (r00 * r11) {
        return true;
    }
    let num1: OpusVal32 = r02 * r11 - r01 * r12;
    lpc[1] = if num1 >= den {
        1.0f32
    } else if num1 <= -den {
        -1.0f32
    } else {
        num1 / den
    };
    let num0: OpusVal32 = r00 * r12 - r02 * r01;
    lpc[0] = if half32(num0) >= den {
        1.999999f32
    } else if half32(num0) <= -den {
        -1.999999f32
    } else {
        num0 / den
    };
    false
}

/// Port of celt/celt_encoder.c:tone_detect: detects pure or nearly pure tones so we can prevent
/// them from causing problems with the encoder. Returns the tone frequency (radians/sample, -1
/// if none) and writes `toneishness`. `x` is scratch of at least `n` samples.
pub fn tone_detect(
    input: &[CeltSig],
    cc: i32,
    n: i32,
    toneishness: &mut OpusVal32,
    fs: i32,
    x: &mut [OpusVal16],
) -> OpusVal16 {
    let nu = n as usize;
    let mut delay: i32 = 1;
    let mut lpc: [OpusVal32; 2] = [0.0; 2];
    let x = &mut x[..nu];
    // Shift by SIG_SHIFT+2 (+3 for stereo) to account for HF gain of the preemphasis filter.
    if cc == 2 {
        for i in 0..nu {
            x[i] = input[i] + input[i + nu];
        }
    } else {
        x.copy_from_slice(&input[..nu]);
    }
    // FIXED_POINT: normalize_tone_input not ported (float build).
    let mut fail = tone_lpc(x, n, delay, &mut lpc);
    // If our LPC filter resonates too close to DC, retry the analysis with down-sampling.
    while delay <= fs / 3000 && (fail || (lpc[0] > 1.0f32 && lpc[1] < 0.0)) {
        delay *= 2;
        fail = tone_lpc(x, n, delay, &mut lpc);
    }
    // Check that our filter has complex roots.
    // C: lpc[0]*lpc[0] + 3.999999*lpc[1] < 0 (the second product is in double).
    if !fail && f64::from(lpc[0] * lpc[0]) + 3.999999 * f64::from(lpc[1]) < 0.0 {
        // Squared radius of the poles.
        *toneishness = -lpc[1];
        // FIXED_POINT: acos_approx not ported (float build).
        (acos(f64::from(0.5f32 * lpc[0])) / f64::from(delay)) as f32
    } else {
        *toneishness = 0.0;
        -1.0
    }
}

/// Encoder state read/written by [`run_prefilter`] (the `st->` fields of the C function).
#[derive(Debug)]
pub struct PrefilterState<'a> {
    /// `st->prefilter_period` (raised to `COMBFILTER_MINPERIOD`).
    pub prefilter_period: &'a mut i32,
    /// `st->prefilter_gain`.
    pub prefilter_gain: OpusVal16,
    /// `st->prefilter_tapset`.
    pub prefilter_tapset: i32,
    /// `st->loss_rate`.
    pub loss_rate: i32,
    /// `st->in_mem` (`CC*overlap`).
    pub in_mem: &'a mut [CeltSig],
}

/// Outputs of [`run_prefilter`] (C `*pitch`, `*gain`, `*qgain`, return value).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefilterOut {
    pub pitch: i32,
    pub gain: OpusVal16,
    pub qgain: i32,
    pub pf_on: bool,
}

/// Port of celt/celt_encoder.c:run_prefilter: pitch pre-filter (comb filter) of `input`
/// (`CC*(N+overlap)`), updating `prefilter_mem` (`CC*QEXT_SCALE(COMBFILTER_MAXPERIOD)`).
///
/// `pre` needs `CC*(N+max_period)` samples and `pitch_buf` `(max_period+N)>>1`.
pub fn run_prefilter(
    mode: &CeltMode,
    st: PrefilterState<'_>,
    input: &mut [CeltSig],
    prefilter_mem: &mut [CeltSig],
    cc: i32,
    n: i32,
    prefilter_tapset: i32,
    enabled: bool,
    complexity: i32,
    tf_estimate: OpusVal16,
    nb_available_bytes: i32,
    analysis: &AnalysisInfo,
    mut tone_freq: OpusVal16,
    toneishness: OpusVal32,
    qext_scale: i32,
    pre: &mut [CeltSig],
    pitch_buf: &mut [OpusVal16],
) -> PrefilterOut {
    let PrefilterState {
        prefilter_period,
        prefilter_gain: st_gain,
        prefilter_tapset: st_tapset,
        loss_rate,
        in_mem,
    } = st;
    let mut before = [0.0f32; 2];
    let mut after = [0.0f32; 2];
    let mut cancel_pitch = false;
    let max_period = qext_scale * COMBFILTER_MAXPERIOD;
    let min_period = qext_scale * COMBFILTER_MINPERIOD;
    let overlap = mode.overlap as usize;
    let (nu, mp) = (n as usize, max_period as usize);
    let ccu = cc as usize;
    let stride = nu + mp;
    let pre = &mut pre[..ccu * stride];

    for c in 0..ccu {
        pre[c * stride..c * stride + mp].copy_from_slice(&prefilter_mem[c * mp..c * mp + mp]);
        let src = c * (nu + overlap) + overlap;
        pre[c * stride + mp..c * stride + mp + nu].copy_from_slice(&input[src..src + nu]);
    }

    let mut gain1: OpusVal16;
    let mut pitch_index: i32;
    // If we detect that the signal is dominated by a single tone, don't rely on the standard
    // pitch estimator, as it can become unreliable.
    if enabled && toneishness > 0.99f32 {
        let mut multiple: i32 = 1;
        // Using aliased version of the postfilter above 24 kHz. First value is purposely
        // slightly above pi to avoid triggering for Fs=48kHz.
        if qext_scale as f32 * tone_freq >= 3.1416f32 {
            tone_freq = 3.141593f32 - tone_freq;
        }
        // If the pitch is too high for our post-filter, apply pitch doubling until we can get
        // something that fits (not ideal, but better than nothing).
        while qext_scale as f32 * tone_freq >= multiple as f32 * 0.39f32 {
            multiple += 1;
        }
        if qext_scale as f32 * tone_freq > 0.006148f32 {
            // FIXED_POINT: integer variant not ported (float build).
            pitch_index = imin(
                floor_i32(
                    0.5 + 2.0f32 as f64 * PI * f64::from(multiple)
                        / f64::from(qext_scale as f32 * tone_freq),
                ),
                COMBFILTER_MAXPERIOD - 2,
            );
        } else {
            // If the pitch is too low, using a very high pitch will actually give us an
            // improvement due to the DC component of the filter that will be close to our
            // tone. Again, not ideal, but if we only have a single tone, it's better than
            // nothing.
            pitch_index = COMBFILTER_MINPERIOD;
        }
        gain1 = 0.75f32;
    } else if enabled && complexity >= 5 {
        let half = ((max_period + n) >> 1) as usize;
        let pitch_buf = &mut pitch_buf[..half];
        {
            let (p0, p1) = pre.split_at(stride);
            let chans: [&[CeltSig]; 2] = [p0, if ccu == 2 { p1 } else { p0 }];
            pitch_downsample(&chans[..ccu], pitch_buf, half, ccu, 2);
        }
        // Don't search for the fir last 1.5 octave of the range because there's too many
        // false-positives due to short-term correlation
        pitch_index = pitch_search(
            &pitch_buf[(max_period >> 1) as usize..],
            pitch_buf,
            nu,
            (max_period - 3 * min_period) as usize,
        );
        pitch_index = max_period - pitch_index;

        gain1 = remove_doubling(
            pitch_buf,
            max_period,
            min_period,
            n,
            &mut pitch_index,
            *prefilter_period,
            st_gain,
        );
        if pitch_index > max_period - 2 * qext_scale {
            pitch_index = max_period - 2 * qext_scale;
        }
        #[cfg(feature = "qext")]
        {
            pitch_index /= qext_scale;
        }
        gain1 *= 0.7f32;
        if loss_rate > 2 {
            gain1 = half32(gain1);
        }
        if loss_rate > 4 {
            gain1 = half32(gain1);
        }
        if loss_rate > 8 {
            gain1 = 0.0;
        }
    } else {
        gain1 = 0.0;
        pitch_index = COMBFILTER_MINPERIOD;
    }
    if analysis.valid != 0 {
        gain1 *= analysis.max_pitch_ratio;
    }
    // Gain threshold for enabling the prefilter/postfilter
    let mut pf_threshold: OpusVal16 = 0.2f32;

    // Adjusting the threshold based on rate and continuity
    if (pitch_index - *prefilter_period).abs() * 10 > pitch_index {
        pf_threshold += 0.2f32;
        // Completely disable the prefilter on strong transients without continuity.
        if tf_estimate > 0.98f32 {
            gain1 = 0.0;
        }
    }
    if nb_available_bytes < 25 {
        pf_threshold += 0.1f32;
    }
    if nb_available_bytes < 35 {
        pf_threshold += 0.1f32;
    }
    if st_gain > 0.4f32 {
        pf_threshold -= 0.1f32;
    }
    if st_gain > 0.55f32 {
        pf_threshold -= 0.1f32;
    }

    // Hard threshold at 0.2
    pf_threshold = max16(pf_threshold, 0.2f32);
    let mut pf_on: bool;
    let mut qg: i32;
    if gain1 < pf_threshold {
        gain1 = 0.0;
        pf_on = false;
        qg = 0;
    } else {
        // This block is not gated by a total bits check only because of the nbAvailableBytes
        // check above.
        if abs16(gain1 - st_gain) < 0.1f32 {
            gain1 = st_gain;
        }
        // FIXED_POINT: integer variant not ported (float build).
        qg = floor_i32(f64::from(0.5f32 + gain1 * 32.0 / 3.0)) - 1;
        qg = imax(0, imin(7, qg));
        gain1 = 0.09375f32 * (qg + 1) as f32;
        pf_on = true;
    }

    let offset = mode.short_mdct_size - mode.overlap;
    let window: &[f32] = &mode.window;
    for c in 0..ccu {
        let base = c * (nu + overlap);
        *prefilter_period = imax(*prefilter_period, COMBFILTER_MINPERIOD);
        input[base..base + overlap].copy_from_slice(&in_mem[c * overlap..c * overlap + overlap]);
        for i in 0..nu {
            before[c] += abs32(input[base + overlap + i]);
        }
        let pre_c = &pre[c * stride..c * stride + stride];
        if offset != 0 {
            comb_filter(
                &mut input[base + overlap..],
                pre_c,
                mp,
                *prefilter_period,
                *prefilter_period,
                offset,
                -st_gain,
                -st_gain,
                st_tapset,
                st_tapset,
                &[],
                0,
            );
        }

        comb_filter(
            &mut input[base + overlap + offset as usize..],
            pre_c,
            mp + offset as usize,
            *prefilter_period,
            pitch_index,
            n - offset,
            -st_gain,
            -gain1,
            st_tapset,
            prefilter_tapset,
            window,
            mode.overlap,
        );
        for i in 0..nu {
            after[c] += abs32(input[base + overlap + i]);
        }
    }

    if cc == 2 {
        let thresh: [OpusVal16; 2] = [
            (0.25f32 * gain1) * before[0] + 0.01f32 * before[1],
            (0.25f32 * gain1) * before[1] + 0.01f32 * before[0],
        ];
        // Don't use the filter if one channel gets significantly worse.
        if after[0] - before[0] > thresh[0] || after[1] - before[1] > thresh[1] {
            cancel_pitch = true;
        }
        // Use the filter only if at least one channel gets significantly better.
        if before[0] - after[0] < thresh[0] && before[1] - after[1] < thresh[1] {
            cancel_pitch = true;
        }
    } else {
        // Check that the mono channel actually got better.
        if after[0] > before[0] {
            cancel_pitch = true;
        }
    }
    // If needed, revert to a gain of zero.
    if cancel_pitch {
        for c in 0..ccu {
            let base = c * (nu + overlap);
            let pre_c = &pre[c * stride..c * stride + stride];
            input[base + overlap..base + overlap + nu].copy_from_slice(&pre_c[mp..mp + nu]);
            comb_filter(
                &mut input[base + overlap + offset as usize..],
                pre_c,
                mp + offset as usize,
                *prefilter_period,
                pitch_index,
                mode.overlap,
                -st_gain,
                0.0,
                st_tapset,
                prefilter_tapset,
                window,
                mode.overlap,
            );
        }
        gain1 = 0.0;
        pf_on = false;
        qg = 0;
    }

    for c in 0..ccu {
        let base = c * (nu + overlap);
        in_mem[c * overlap..c * overlap + overlap]
            .copy_from_slice(&input[base + nu..base + nu + overlap]);
        let pre_c = &pre[c * stride..c * stride + stride];
        let pm = &mut prefilter_mem[c * mp..c * mp + mp];
        if nu > mp {
            pm.copy_from_slice(&pre_c[nu..nu + mp]);
        } else {
            pm.copy_within(nu..mp, 0);
            pm[mp - nu..].copy_from_slice(&pre_c[mp..mp + nu]);
        }
    }

    PrefilterOut {
        pitch: pitch_index,
        gain: gain1,
        qgain: qg,
        pf_on,
    }
}

/// Port of celt/celt_encoder.c:compute_vbr: the VBR target (in 1/8 bits) for this frame.
#[must_use]
pub fn compute_vbr(
    mode: &CeltMode,
    analysis: &AnalysisInfo,
    base_target: i32,
    lm: i32,
    bitrate: i32,
    last_coded_bands: i32,
    c: i32,
    intensity: i32,
    constrained_vbr: bool,
    mut stereo_saving: OpusVal16,
    tot_boost: i32,
    tf_estimate: OpusVal16,
    pitch_change: bool,
    max_depth: CeltGlog,
    lfe: bool,
    has_surround_mask: bool,
    surround_masking: CeltGlog,
    temporal_vbr: CeltGlog,
    #[cfg(feature = "qext")] enable_qext: bool,
) -> i32 {
    let nb_ebands = mode.nb_ebands;
    let e_bands = &mode.e_bands;
    let eb = |i: i32| i32::from(e_bands[i as usize]);

    let coded_bands = if last_coded_bands != 0 {
        last_coded_bands
    } else {
        nb_ebands
    };
    let mut coded_bins = eb(coded_bands) << lm;
    if c == 2 {
        coded_bins += eb(imin(intensity, coded_bands)) << lm;
    }

    let mut target = base_target;

    if analysis.valid != 0 && f64::from(analysis.activity) < 0.4 {
        target -= ((coded_bins << BITRES) as f32 * (0.4f32 - analysis.activity)) as i32;
    }
    // Stereo savings
    if c == 2 {
        let coded_stereo_bands = imin(intensity, coded_bands);
        let coded_stereo_dof = (eb(coded_stereo_bands) << lm) - coded_stereo_bands;
        // Maximum fraction of the bits we can save if the signal is mono.
        let max_frac: OpusVal16 = (0.8f32 * coded_stereo_dof as f32) / coded_bins as f32;
        stereo_saving = min16(stereo_saving, 1.0f32);
        target -= min32(
            max_frac * target as f32,
            (stereo_saving - 0.1f32) * (coded_stereo_dof << BITRES) as f32,
        ) as i32;
    }
    // Boost the rate according to dynalloc (minus the dynalloc average for calibration).
    target += tot_boost - (19 << lm);
    // Apply transient boost, compensating for average boost.
    let tf_calibration: OpusVal16 = 0.044f32;
    target += ((tf_estimate - tf_calibration) * target as f32) as i32;

    // Apply tonality boost
    if analysis.valid != 0 && !lfe {
        // Tonality boost (compensating for the average).
        let tonal: f32 = max16(0.0f32, analysis.tonality - 0.15f32) - 0.12f32;
        let mut tonal_target = target + ((coded_bins << BITRES) as f32 * 1.2f32 * tonal) as i32;
        if pitch_change {
            tonal_target += ((coded_bins << BITRES) as f32 * 0.8f32) as i32;
        }
        target = tonal_target;
    }

    if has_surround_mask && !lfe {
        let surround_target = target + (surround_masking * (coded_bins << BITRES) as f32) as i32;
        target = imax(target / 4, surround_target);
    }

    {
        #[allow(unused_mut, reason = "only reassigned with qext")]
        let mut bins = eb(nb_ebands - 2) << lm;
        #[cfg(feature = "qext")]
        if enable_qext {
            bins = mode.short_mdct_size << lm;
        }
        let mut floor_depth = (((c * bins) << BITRES) as f32 * max_depth) as i32;
        floor_depth = imax(floor_depth, target >> 2);
        target = imin(target, floor_depth);
    }

    // Make VBR less aggressive for constrained VBR because we can't keep a higher bitrate for
    // long. Needs tuning.
    if (!has_surround_mask || lfe) && constrained_vbr {
        target = base_target + (0.67f32 * (target - base_target) as f32) as i32;
    }

    if !has_surround_mask && tf_estimate < 0.2f32 {
        let amount: OpusVal16 = 0.0000031f32 * imax(0, imin(32000, 96000 - bitrate)) as f32;
        let tvbr_factor: OpusVal16 = temporal_vbr * amount;
        target += (tvbr_factor * target as f32) as i32;
    }

    // Don't allow more than doubling the rate
    imin(2 * base_target, target)
}

/// Port of celt/celt_encoder.c:encode_qext_stereo_params.
#[cfg(feature = "qext")]
pub fn encode_qext_stereo_params(
    ec: &mut EcEnc<'_>,
    qext_end: i32,
    qext_intensity: i32,
    qext_dual_stereo: bool,
) {
    ec.enc_uint(qext_intensity as u32, (qext_end + 1) as u32);
    if qext_intensity != 0 {
        ec.enc_bit_logp(qext_dual_stereo, 1);
    }
}

/// Values computed by `celt_encode_with_ec` before the range encoder is set up.
#[derive(Debug, Clone, Copy)]
struct FrameSetup {
    lm: i32,
    end: i32,
    #[cfg(feature = "qext")]
    frame_size: i32,
    nb_compressed_bytes: i32,
    nb_filled_bytes: i32,
    nb_available_bytes: i32,
    effective_bytes: i32,
    vbr_rate: i32,
    equiv_rate: i32,
    tell: i32,
    tell0_frac: i32,
    packet_size_cap: i32,
}

/// `intensity_thresholds` of celt_encode_with_ec.
static INTENSITY_THRESHOLDS: [OpusVal16; 21] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 16.0, 24.0, 36.0, 44.0, 50.0, 56.0, 62.0, 67.0, 72.0,
    79.0, 88.0, 106.0, 134.0,
];
/// `intensity_histeresis` of celt_encode_with_ec.
static INTENSITY_HISTERESIS: [OpusVal16; 21] = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0,
    8.0, 8.0,
];

impl CeltEncoder {
    /// Port of celt/celt_encoder.c:celt_encode_with_ec: encodes one frame.
    ///
    /// * `pcm`: `channels*frame_size` interleaved samples (float `opus_res`, nominal +-1.0).
    /// * `compressed` / `nb_compressed_bytes`: output buffer and maximum size when `enc` is
    ///   `None` (C `enc == NULL`). With an existing range encoder `compressed` is ignored, as
    ///   when the Opus encoder passes `NULL` (C would otherwise use it only on the QEXT path).
    /// * `enc`: an existing range encoder (hybrid mode: SILK data already written). CELT then
    ///   writes after it and finalises it (`ec_enc_done`).
    /// * `toc` (QEXT): the byte before the payload (the Opus TOC, C `compressed[-1]`), whose
    ///   code bits are set to 3 when a QEXT extension is appended. With custom-mode signalling,
    ///   the CELT header byte is used instead.
    ///
    /// Returns the number of bytes written (C semantics), or a negative `OPUS_*` error code.
    /// Rust-only checks: `OPUS_BAD_ARG` if `pcm` is too short or the output buffer is missing
    /// or shorter than `nb_compressed_bytes` (C reads/writes out of bounds).
    pub fn celt_encode_with_ec(
        &mut self,
        pcm: &[OpusRes],
        frame_size: i32,
        compressed: Option<&mut [u8]>,
        nb_compressed_bytes: i32,
        enc: Option<&mut EcEnc<'_>>,
        #[cfg(feature = "qext")] toc: Option<&mut u8>,
    ) -> i32 {
        let st = self;
        let mode: &CeltMode = &st.mode;
        let cc = st.channels;
        let c = st.stream_channels;
        #[allow(unused_mut, reason = "only changed with custom modes")]
        let mut end = st.end;
        let mut nb_compressed_bytes = nb_compressed_bytes;
        if nb_compressed_bytes < 2 {
            return OPUS_BAD_ARG;
        }

        let frame_size = frame_size * st.upsample;
        let mut lm = 0;
        while lm <= mode.max_lm {
            if mode.short_mdct_size << lm == frame_size {
                break;
            }
            lm += 1;
        }
        if lm > mode.max_lm {
            return OPUS_BAD_ARG;
        }
        if pcm.len() < (cc * (frame_size / st.upsample)) as usize {
            return OPUS_BAD_ARG;
        }
        #[allow(unused_mut, reason = "only changed with qext")]
        let mut packet_size_cap = 1275;
        #[cfg(feature = "qext")]
        if st.enable_qext != 0 {
            packet_size_cap = QEXT_PACKET_SIZE_CAP;
        }

        let (tell0_frac, tell, nb_filled_bytes) = match &enc {
            None => (1, 1, 0),
            Some(e) => (e.tell_frac() as i32, e.tell(), (e.tell() + 4) >> 3),
        };
        let mut buf: Option<&mut [u8]> = None;
        if enc.is_none() {
            match compressed {
                Some(b) if b.len() >= nb_compressed_bytes as usize => buf = Some(b),
                _ => return OPUS_BAD_ARG,
            }
        }
        #[cfg(all(feature = "custom-modes", feature = "qext"))]
        let mut header: Option<&mut u8> = None;
        #[cfg(feature = "custom-modes")]
        if st.signalling != 0 && enc.is_none() {
            let tmp = (mode.eff_ebands - end) >> 1;
            end = imax(1, mode.eff_ebands - tmp);
            st.end = end;
            let Some(b) = buf.take() else {
                return OPUS_BAD_ARG;
            };
            let (h0, rest) = b.split_at_mut(1);
            // C int -> unsigned char stores truncate.
            h0[0] = (tmp << 5) as u8;
            h0[0] |= (lm << 3) as u8;
            h0[0] |= u8::from(c == 2) << 2;
            // Convert "standard mode" to Opus header
            #[cfg(not(feature = "qext"))]
            let convert = mode.fs == 48000 && mode.short_mdct_size == 120;
            #[cfg(feature = "qext")]
            let convert = true;
            if convert {
                let c0 = crate::celt::celt::to_opus(h0[0]);
                if c0 < 0 {
                    return OPUS_BAD_ARG;
                }
                h0[0] = c0 as u8;
            }
            #[cfg(feature = "qext")]
            {
                header = Some(&mut h0[0]);
            }
            buf = Some(rest);
            nb_compressed_bytes -= 1;
        }
        #[cfg(not(feature = "custom-modes"))]
        debug_assert!(st.signalling == 0);

        // Can't produce more than 1275 output bytes for the main payload, plus any QEXT extra
        // data.
        nb_compressed_bytes = imin(nb_compressed_bytes, packet_size_cap);

        let mut enc = enc;
        let (vbr_rate, effective_bytes) = if st.vbr != 0 && st.bitrate != OPUS_BITRATE_MAX {
            #[allow(unused_mut, reason = "only changed with custom modes")]
            let mut vr = bitrate_to_bits(st.bitrate, mode.fs, frame_size) << BITRES;
            #[cfg(feature = "custom-modes")]
            if st.signalling != 0 {
                vr -= 8 << BITRES;
            }
            (vr, vr >> (3 + BITRES))
        } else {
            // C int arithmetic; wraps like the C build for absurd bitrate*frame_size products.
            let mut tmp = st.bitrate.wrapping_mul(frame_size);
            if tell > 1 {
                tmp = tmp.wrapping_add(tell.wrapping_mul(mode.fs));
            }
            if st.bitrate != OPUS_BITRATE_MAX {
                nb_compressed_bytes = imax(
                    2,
                    imin(
                        nb_compressed_bytes,
                        tmp.wrapping_add(4 * mode.fs) / (8 * mode.fs)
                            - i32::from(st.signalling != 0),
                    ),
                );
                if let Some(e) = enc.as_deref_mut() {
                    e.shrink(nb_compressed_bytes as u32);
                }
            }
            (0, nb_compressed_bytes - nb_filled_bytes)
        };
        let nb_available_bytes = nb_compressed_bytes - nb_filled_bytes;
        let mut equiv_rate =
            ((nb_compressed_bytes * 8 * 50) << (3 - lm)) - (40 * c + 20) * ((400 >> lm) - 50);
        if st.bitrate != OPUS_BITRATE_MAX {
            equiv_rate = imin(equiv_rate, st.bitrate - (40 * c + 20) * ((400 >> lm) - 50));
        }
        let setup = FrameSetup {
            lm,
            end,
            #[cfg(feature = "qext")]
            frame_size,
            nb_compressed_bytes,
            nb_filled_bytes,
            nb_available_bytes,
            effective_bytes,
            vbr_rate,
            equiv_rate,
            tell,
            tell0_frac,
            packet_size_cap,
        };

        match enc {
            Some(e) => st.encode_frame(
                pcm,
                e,
                setup,
                #[cfg(feature = "qext")]
                toc,
            ),
            None => {
                let Some(b) = buf else {
                    return OPUS_BAD_ARG;
                };
                let mut e = EcEnc::with_size(b, nb_compressed_bytes as u32);
                #[cfg(all(feature = "qext", feature = "custom-modes"))]
                let toc = match header {
                    Some(h) => Some(h),
                    None => toc,
                };
                st.encode_frame(
                    pcm,
                    &mut e,
                    setup,
                    #[cfg(feature = "qext")]
                    toc,
                )
            }
        }
    }

    /// Body of `celt_encode_with_ec` after the range encoder exists.
    fn encode_frame(
        &mut self,
        pcm: &[OpusRes],
        enc: &mut EcEnc<'_>,
        s: FrameSetup,
        #[cfg(feature = "qext")] toc: Option<&mut u8>,
    ) -> i32 {
        let st = self;
        let mode: &CeltMode = &st.mode;
        let nb_ebands = mode.nb_ebands;
        let nb = nb_ebands as usize;
        let overlap = mode.overlap;
        let ov = overlap as usize;
        let e_bands: &[i16] = &mode.e_bands;
        let eb = |i: i32| i32::from(e_bands[i as usize]);
        let cc = st.channels;
        let c = st.stream_channels;
        let (ccu, cu) = (cc as usize, c as usize);
        let start = st.start;
        let end = s.end;
        let hybrid = start != 0;
        let lm = s.lm;
        let m_ = 1 << lm;
        let n = m_ * mode.short_mdct_size;
        let nu = n as usize;
        #[cfg(feature = "qext")]
        let frame_size = s.frame_size;
        let qext_scale = st.qext_scale();
        let mut nb_compressed_bytes = s.nb_compressed_bytes;
        let nb_filled_bytes = s.nb_filled_bytes;
        let mut nb_available_bytes = s.nb_available_bytes;
        let mut effective_bytes = s.effective_bytes;
        let vbr_rate = s.vbr_rate;
        let equiv_rate = s.equiv_rate;
        let mut tell = s.tell;
        let tell0_frac = s.tell0_frac;
        let packet_size_cap = s.packet_size_cap;
        let mut tf_estimate: OpusVal16 = 0.0;
        let mut tf_chan: i32 = 0;
        let mut pitch_change = false;
        let mut transient_got_disabled = false;
        let mut surround_masking: CeltGlog = 0.0;
        let mut temporal_vbr: CeltGlog = 0.0;
        let mut surround_trim: CeltGlog = 0.0;
        let mut weak_transient = false;
        let mut toneishness: OpusVal32 = 0.0;
        let mut dual_stereo: i32 = 0;
        #[allow(unused_mut, reason = "only changed with qext")]
        let mut qext_bytes: i32 = 0;

        let EncScratch {
            input,
            freq,
            x,
            band_e,
            band_log_e,
            band_log_e2,
            error,
            surround_dynalloc,
            offsets,
            importance,
            spread_weight,
            tf_res,
            cap,
            fine_quant,
            pulses,
            fine_priority,
            collapse_masks,
            tmp: tmp_buf,
            pre,
            pitch_buf,
            tf_metric,
            tf_path0,
            tf_path1,
            tf_tmp,
            tf_tmp1,
            dyn_bufs,
            bands,
            #[cfg(feature = "custom-modes")]
                pcm_conv: _,
            #[cfg(feature = "qext")]
            extra_quant,
            #[cfg(feature = "qext")]
            extra_pulses,
            #[cfg(feature = "qext")]
            error_bak,
            #[cfg(feature = "qext")]
            zeros,
        } = &mut st.scratch;

        if vbr_rate > 0 {
            // Computes the max bit-rate allowed in VBR mode to avoid violating the target rate
            // and buffering. We must do this up front so that bust-prevention logic triggers
            // correctly if we don't have enough bits.
            if st.constrained_vbr != 0 {
                // We could use any multiple of vbr_rate as bound (depending on the delay). This
                // is clamped to ensure we use at least two bytes if the encoder was entirely
                // empty, but to allow 0 in hybrid mode.
                let vbr_bound = vbr_rate;
                let max_allowed = imin(
                    imax(
                        if tell == 1 { 2 } else { 0 },
                        (vbr_rate + vbr_bound - st.vbr_reservoir) >> (BITRES + 3),
                    ),
                    nb_available_bytes,
                );
                if max_allowed < nb_available_bytes {
                    nb_compressed_bytes = nb_filled_bytes + max_allowed;
                    nb_available_bytes = max_allowed;
                    enc.shrink(nb_compressed_bytes as u32);
                }
            }
        }
        let mut total_bits = nb_compressed_bytes * 8;

        let mut eff_end = end;
        if eff_end > mode.eff_ebands {
            eff_end = mode.eff_ebands;
        }

        let input = &mut input[..ccu * (nu + ov)];

        let up = st.upsample;
        let l0 = (c * (n - overlap) / up) as usize;
        let mut sample_max: OpusVal32 = max32(st.overlap_max, celt_maxabs_res(&pcm[..l0]));
        st.overlap_max = celt_maxabs_res(&pcm[l0..l0 + (c * overlap / up) as usize]);
        sample_max = max32(sample_max, st.overlap_max);
        // FIXED_POINT: silence = (sample_max==0) not ported (float build).
        let mut silence = sample_max <= 1.0f32 / (1i32 << st.lsb_depth) as f32;
        if tell == 1 {
            enc.enc_bit_logp(silence, 15);
        } else {
            silence = false;
        }
        if silence {
            // In VBR mode there is no need to send more than the minimum.
            if vbr_rate > 0 {
                nb_compressed_bytes = imin(nb_compressed_bytes, nb_filled_bytes + 2);
                effective_bytes = nb_compressed_bytes;
                total_bits = nb_compressed_bytes * 8;
                nb_available_bytes = 2;
                enc.shrink(nb_compressed_bytes as u32);
            }
            #[cfg(feature = "qext")]
            if vbr_rate <= 0 && st.enable_qext != 0 {
                nb_compressed_bytes = imin(nb_compressed_bytes, 1275);
                nb_available_bytes = nb_compressed_bytes - nb_filled_bytes;
                total_bits = nb_compressed_bytes * 8;
                enc.shrink(nb_compressed_bytes as u32);
            }
            // Pretend we've filled all the remaining bits with zeros (that's what the
            // initialiser did anyway)
            tell = nb_compressed_bytes * 8;
            enc.nbits_total += tell - enc.tell();
        }
        let max_period = (qext_scale * COMBFILTER_MAXPERIOD) as usize;
        for ch in 0..ccu {
            // FIXED_POINT: need_clip threshold in fixed point not ported (float build).
            let need_clip = st.clip != 0 && sample_max > 65536.0f32;
            celt_preemphasis(
                &pcm[ch..],
                &mut input[ch * (nu + ov) + ov..],
                n,
                cc,
                up,
                &mode.preemph,
                &mut st.preemph_mem_e[ch],
                need_clip,
            );
            let src = (1 + ch) * max_period - ov;
            input[ch * (nu + ov)..ch * (nu + ov) + ov]
                .copy_from_slice(&st.prefilter_mem[src..src + ov]);
        }

        let tone_freq = tone_detect(input, cc, n + overlap, &mut toneishness, mode.fs, tmp_buf);
        let mut is_transient = false;
        let mut short_blocks: i32 = 0;
        if st.complexity >= 1 && st.lfe == 0 {
            // Reduces the likelihood of energy instability on fricatives at low bitrate in
            // hybrid mode. It seems like we still want to have real transients on vowels
            // though (small SILK quantization offset value).
            let allow_weak_transients =
                hybrid && effective_bytes < 15 && st.silk_info.signal_type != 2;
            is_transient = transient_analysis(
                input,
                n + overlap,
                cc,
                &mut tf_estimate,
                &mut tf_chan,
                allow_weak_transients,
                &mut weak_transient,
                tone_freq,
                toneishness,
                tmp_buf,
            );
        }
        toneishness = min32(toneishness, 1.0f32 - tf_estimate);
        // Find pitch period and gain
        let prefilter_tapset = st.tapset_decision;
        let pf_on;
        let pitch_index;
        let gain1;
        {
            let enabled = ((st.lfe != 0 && nb_available_bytes > 3) || nb_available_bytes > 12 * c)
                && !hybrid
                && !silence
                && tell + 16 <= total_bits
                && st.disable_pf == 0;

            let pf = run_prefilter(
                mode,
                PrefilterState {
                    prefilter_period: &mut st.prefilter_period,
                    prefilter_gain: st.prefilter_gain,
                    prefilter_tapset: st.prefilter_tapset,
                    loss_rate: st.loss_rate,
                    in_mem: &mut st.in_mem,
                },
                input,
                &mut st.prefilter_mem,
                cc,
                n,
                prefilter_tapset,
                enabled,
                st.complexity,
                tf_estimate,
                nb_available_bytes,
                &st.analysis,
                tone_freq,
                toneishness,
                qext_scale,
                pre,
                pitch_buf,
            );
            pf_on = pf.pf_on;
            gain1 = pf.gain;
            let qg = pf.qgain;
            let mut pi = pf.pitch;
            if (gain1 > 0.4f32 || st.prefilter_gain > 0.4f32)
                && (st.analysis.valid == 0 || f64::from(st.analysis.tonality) > 0.3)
                && (f64::from(pi) > 1.26 * f64::from(st.prefilter_period)
                    || f64::from(pi) < 0.79 * f64::from(st.prefilter_period))
            {
                pitch_change = true;
            }
            if !pf_on {
                if !hybrid && tell + 16 <= total_bits {
                    enc.enc_bit_logp(false, 1);
                }
            } else {
                // This block is not gated by a total bits check only because of the
                // nbAvailableBytes check above.
                enc.enc_bit_logp(true, 1);
                pi += 1;
                let octave = ec_ilog(pi as u32) - 5;
                enc.enc_uint(octave as u32, 6);
                enc.enc_bits((pi - (16 << octave)) as u32, (4 + octave) as u32);
                pi -= 1;
                enc.enc_bits(qg as u32, 3);
                enc.enc_icdf(prefilter_tapset as usize, &TAPSET_ICDF, 2);
            }
            pitch_index = pi;
        }
        if lm > 0 && enc.tell() + 3 <= total_bits {
            if is_transient {
                short_blocks = m_;
            }
        } else {
            is_transient = false;
            transient_got_disabled = true;
        }

        let freq = &mut freq[..ccu * nu];
        let band_e = &mut band_e[..ccu * nb];
        let band_log_e = &mut band_log_e[..ccu * nb];

        let second_mdct = short_blocks != 0 && st.complexity >= 8;
        let band_log_e2 = &mut band_log_e2[..cu * nb];
        if second_mdct {
            compute_mdcts(mode, 0, input, freq, c, cc, lm, up);
            compute_band_energies(mode, freq, band_e, eff_end, c, lm);
            amp2_log2(mode, eff_end, end, band_e, band_log_e2, c);
            for ch in 0..cu {
                for i in 0..end as usize {
                    band_log_e2[nb * ch + i] += half32(lm as f32);
                }
            }
        }

        compute_mdcts(mode, short_blocks, input, freq, c, cc, lm, up);
        // This should catch any NaN in the CELT input. Since we're not supposed to see any
        // (they're filtered at the Opus layer), just abort (C asserts).
        debug_assert!(!freq[0].is_nan() && (c == 1 || !freq[nu].is_nan()));
        if cc == 2 && c == 1 {
            tf_chan = 0;
        }
        compute_band_energies(mode, freq, band_e, eff_end, c, lm);

        if st.lfe != 0 {
            for i in 2..end as usize {
                band_e[i] = min32(band_e[i], 1e-4f32 * band_e[0]);
                band_e[i] = max32(band_e[i], EPSILON);
            }
        }
        amp2_log2(mode, eff_end, end, band_e, band_log_e, c);

        let surround_dynalloc = &mut surround_dynalloc[..cu * nb];
        surround_dynalloc[..end as usize].fill(0.0);
        // This computes how much masking takes place between surround channels
        if !hybrid && st.has_energy_mask && st.lfe == 0 {
            let energy_mask = &st.energy_mask;
            let mut mask_avg: OpusVal32 = 0.0;
            let mut diff: OpusVal32 = 0.0;
            let mut count: i32 = 0;
            let mask_end = imax(2, st.last_coded_bands);
            for ch in 0..cu {
                for i in 0..mask_end {
                    let mut mask: CeltGlog =
                        maxg(ming(energy_mask[nb * ch + i as usize], 0.25f32), -2.0f32);
                    if mask > 0.0 {
                        mask = half32(mask);
                    }
                    let mask16: OpusVal16 = mask;
                    mask_avg += mask16 * (eb(i + 1) - eb(i)) as f32;
                    count += eb(i + 1) - eb(i);
                    diff += mask16 * (1 + 2 * i - mask_end) as f32;
                }
            }
            debug_assert!(count > 0);
            mask_avg /= count as f32;
            mask_avg += 0.2f32;
            diff = diff * 6.0 / (c * (mask_end - 1) * (mask_end + 1) * mask_end) as f32;
            // Again, being conservative
            diff = half32(diff);
            diff = max32(min32(diff, 0.031f32), -0.031f32);
            // Find the band that's in the middle of the coded spectrum
            let mut midband: i32 = 0;
            while eb(midband + 1) < eb(mask_end) / 2 {
                midband += 1;
            }
            let mut count_dynalloc = 0;
            for i in 0..mask_end {
                let iu = i as usize;
                let lin: OpusVal32 = mask_avg + diff * (i - midband) as f32;
                let mut unmask: CeltGlog = if c == 2 {
                    maxg(energy_mask[iu], energy_mask[nb + iu])
                } else {
                    energy_mask[iu]
                };
                unmask = ming(unmask, 0.0f32);
                unmask -= lin;
                if unmask > 0.25f32 {
                    surround_dynalloc[iu] = unmask - 0.25f32;
                    count_dynalloc += 1;
                }
            }
            if count_dynalloc >= 3 {
                // If we need dynalloc in many bands, it's probably because our initial masking
                // rate was too low.
                mask_avg += 0.25f32;
                if mask_avg > 0.0 {
                    // Something went really wrong in the original calculations, disabling
                    // masking.
                    mask_avg = 0.0;
                    diff = 0.0;
                    surround_dynalloc[..mask_end as usize].fill(0.0);
                } else {
                    for i in 0..mask_end as usize {
                        surround_dynalloc[i] = maxg(0.0, surround_dynalloc[i] - 0.25f32);
                    }
                }
            }
            mask_avg += 0.2f32;
            // Convert to 1/64th units used for the trim
            surround_trim = 64.0 * diff;
            surround_masking = mask_avg;
        }
        // Temporal VBR (but not for LFE)
        if st.lfe == 0 {
            let mut follow: CeltGlog = -10.0f32;
            let mut frame_avg: OpusVal32 = 0.0;
            let offset: CeltGlog = if short_blocks != 0 {
                half32(lm as f32)
            } else {
                0.0
            };
            for i in start as usize..end as usize {
                follow = maxg(follow - 1.0f32, band_log_e[i] - offset);
                if c == 2 {
                    follow = maxg(follow, band_log_e[i + nb] - offset);
                }
                frame_avg += follow;
            }
            frame_avg /= (end - start) as f32;
            temporal_vbr = frame_avg - st.spec_avg;
            temporal_vbr = ming(3.0f32, maxg(-1.5f32, temporal_vbr));
            st.spec_avg += 0.02f32 * temporal_vbr;
        }

        if !second_mdct {
            band_log_e2.copy_from_slice(&band_log_e[..cu * nb]);
        }

        // Last chance to catch any transient we might have missed in the time-domain analysis
        if lm > 0
            && enc.tell() + 3 <= total_bits
            && !is_transient
            && st.complexity >= 5
            && st.lfe == 0
            && !hybrid
            && patch_transient_decision(band_log_e, &st.old_band_e, nb_ebands, start, end, c)
        {
            is_transient = true;
            short_blocks = m_;
            compute_mdcts(mode, short_blocks, input, freq, c, cc, lm, up);
            compute_band_energies(mode, freq, band_e, eff_end, c, lm);
            amp2_log2(mode, eff_end, end, band_e, band_log_e, c);
            // Compensate for the scaling of short vs long mdcts
            for ch in 0..cu {
                for i in 0..end as usize {
                    band_log_e2[nb * ch + i] += half32(lm as f32);
                }
            }
            tf_estimate = 0.2f32;
        }

        if lm > 0 && enc.tell() + 3 <= total_bits {
            enc.enc_bit_logp(is_transient, 3);
        }

        // `C*N` (plus the out-of-range tail for custom modes, see `EncScratch::x`).
        let x_tail = ((eb(nb_ebands) << lm) as usize).saturating_sub(nu);
        let x = &mut x[..cu * nu + x_tail];

        // Band normalisation
        normalise_bands(mode, freq, x, band_e, eff_end, c, m_);

        let enable_tf_analysis = effective_bytes >= 15 * c
            && !hybrid
            && st.complexity >= 2
            && st.lfe == 0
            && toneishness < 0.98f32;

        let mut tot_boost: i32 = 0;
        let max_depth = dynalloc_analysis(
            band_log_e,
            band_log_e2,
            &st.old_band_e,
            nb_ebands,
            start,
            end,
            c,
            offsets,
            st.lsb_depth,
            &mode.log_n,
            is_transient,
            st.vbr != 0,
            st.constrained_vbr != 0,
            e_bands,
            lm,
            effective_bytes,
            &mut tot_boost,
            st.lfe != 0,
            surround_dynalloc,
            &st.analysis,
            importance,
            spread_weight,
            tone_freq,
            toneishness,
            qext_scale,
            dyn_bufs,
        );

        let tf_select: i32;
        // Disable variable tf resolution for hybrid and at very low bitrate
        if enable_tf_analysis {
            let lambda = imax(80, 20480 / effective_bytes + 2);
            tf_select = tf_analysis(
                mode,
                eff_end,
                is_transient,
                tf_res,
                lambda,
                x,
                n,
                lm,
                tf_estimate,
                tf_chan,
                importance,
                tf_metric,
                tf_path0,
                tf_path1,
                tf_tmp,
                tf_tmp1,
            );
            for i in eff_end as usize..end as usize {
                tf_res[i] = tf_res[eff_end as usize - 1];
            }
        } else if hybrid && weak_transient {
            // For weak transients, we rely on the fact that improving time resolution using
            // TF on a long window is imperfect and will not result in an energy collapse at
            // low bitrate.
            tf_res[..end as usize].fill(1);
            tf_select = 0;
        } else if hybrid && effective_bytes < 15 && st.silk_info.signal_type != 2 {
            // For low bitrate hybrid, we force temporal resolution to 5 ms rather than 2.5 ms.
            tf_res[..end as usize].fill(0);
            tf_select = i32::from(is_transient);
        } else {
            tf_res[..end as usize].fill(i32::from(is_transient));
            tf_select = 0;
        }

        let error = &mut error[..cu * nb];
        for ch in 0..cu {
            for i in start as usize..end as usize {
                let k = i + ch * nb;
                // When the energy is stable, slightly bias energy quantization towards the
                // previous error to make the gain more stable (a constant offset is better than
                // fluctuations).
                if abs32(band_log_e[k] - st.old_band_e[k]) < 2.0f32 {
                    band_log_e[k] -= 0.25f32 * st.energy_error[k];
                }
            }
        }
        quant_coarse_energy(
            mode,
            start,
            end,
            eff_end,
            band_log_e,
            &mut st.old_band_e,
            total_bits as u32,
            error,
            enc,
            c,
            lm,
            nb_available_bytes,
            st.force_intra != 0,
            &mut st.delayed_intra,
            st.complexity >= 4,
            st.loss_rate,
            st.lfe != 0,
        );

        tf_encode(start, end, is_transient, tf_res, lm, tf_select, enc);

        if enc.tell() + 4 <= total_bits {
            if st.lfe != 0 {
                st.tapset_decision = 0;
                st.spread_decision = SPREAD_NORMAL;
            } else if hybrid {
                st.spread_decision = if st.complexity == 0 {
                    SPREAD_NONE
                } else if is_transient {
                    SPREAD_NORMAL
                } else {
                    SPREAD_AGGRESSIVE
                };
            } else if short_blocks != 0 || st.complexity < 3 || nb_available_bytes < 10 * c {
                st.spread_decision = if st.complexity == 0 {
                    SPREAD_NONE
                } else {
                    SPREAD_NORMAL
                };
            } else {
                // Disable new spreading+tapset estimator until we can show it works better than
                // the old one. So far it seems like spreading_decision() works best.
                // (The `#if 0` hysteresis_decision variant is not ported.)
                st.spread_decision = spreading_decision(
                    mode,
                    x,
                    &mut st.tonal_average,
                    st.spread_decision,
                    &mut st.hf_average,
                    &mut st.tapset_decision,
                    pf_on && short_blocks == 0,
                    eff_end,
                    c,
                    m_,
                    spread_weight,
                );
            }
            enc.enc_icdf(st.spread_decision as usize, &SPREAD_ICDF, 5);
        } else {
            st.spread_decision = SPREAD_NORMAL;
        }

        // For LFE, everything interesting is in the first band
        if st.lfe != 0 {
            offsets[0] = imin(8, effective_bytes / 3);
        }
        init_caps(mode, cap, lm, c);

        let mut dynalloc_logp = 6;
        total_bits <<= BITRES;
        let mut total_boost: i32 = 0;
        tell = enc.tell_frac() as i32;
        for i in start..end {
            let iu = i as usize;
            let width = (c * (eb(i + 1) - eb(i))) << lm;
            // quanta is 6 bits, but no more than 1 bit/sample and no less than 1/8 bit/sample
            let quanta = imin(width << BITRES, imax(6 << BITRES, width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0;
            let mut j = 0;
            while tell + (dynalloc_loop_logp << BITRES) < total_bits - total_boost
                && boost < cap[iu]
            {
                let flag = j < offsets[iu];
                enc.enc_bit_logp(flag, dynalloc_loop_logp as u32);
                tell = enc.tell_frac() as i32;
                if !flag {
                    break;
                }
                boost += quanta;
                total_boost += quanta;
                dynalloc_loop_logp = 1;
                j += 1;
            }
            // Making dynalloc more likely
            if j != 0 {
                dynalloc_logp = imax(2, dynalloc_logp - 1);
            }
            offsets[iu] = boost;
        }

        if c == 2 {
            // Always use MS for 2.5 ms frames until we can do a better analysis
            if lm != 0 {
                dual_stereo = i32::from(stereo_analysis(mode, x, lm, n));
            }

            st.intensity = hysteresis_decision(
                (equiv_rate / 1000) as f32,
                &INTENSITY_THRESHOLDS,
                &INTENSITY_HISTERESIS,
                21,
                st.intensity,
            );
            st.intensity = imin(end, imax(start, st.intensity));
        }

        let mut alloc_trim = 5;
        if tell + (6 << BITRES) <= total_bits - total_boost {
            if start > 0 || st.lfe != 0 {
                st.stereo_saving = 0.0;
                alloc_trim = 5;
            } else {
                alloc_trim = alloc_trim_analysis(
                    mode,
                    x,
                    band_log_e,
                    end,
                    lm,
                    c,
                    n,
                    &st.analysis,
                    &mut st.stereo_saving,
                    tf_estimate,
                    st.intensity,
                    surround_trim,
                    equiv_rate,
                );
            }
            enc.enc_icdf(alloc_trim as usize, &TRIM_ICDF, 7);
            tell = enc.tell_frac() as i32;
        }

        // In VBR mode the frame size must not be reduced so much that it would result in the
        // encoder running out of bits. The margin of 2 bytes ensures that none of the
        // bust-prevention logic in the decoder will have triggered so far.
        let mut min_allowed = ((tell + total_boost + (1 << (BITRES + 3)) - 1) >> (BITRES + 3)) + 2;
        // Take into account the 37 bits we need to have left in the packet to signal a
        // redundant frame in hybrid mode. Creating a shorter packet would create an entropy
        // coder desync.
        if hybrid {
            min_allowed = imax(
                min_allowed,
                (tell0_frac + (37 << BITRES) + total_boost + (1 << (BITRES + 3)) - 1)
                    >> (BITRES + 3),
            );
        }
        // Variable bitrate
        if vbr_rate > 0 {
            // The target rate in 8th bits per frame
            let mut target: i32;
            let lm_diff = mode.max_lm - lm;

            // Don't attempt to use more than 510 kb/s, even for frames smaller than 20 ms. The
            // CELT allocator will just not be able to use more than that anyway.
            nb_compressed_bytes = imin(nb_compressed_bytes, packet_size_cap >> (3 - lm));
            let mut base_target = if !hybrid {
                vbr_rate - ((40 * c + 20) << BITRES)
            } else {
                imax(0, vbr_rate - ((9 * c + 4) << BITRES))
            };

            if st.constrained_vbr != 0 {
                base_target += st.vbr_offset >> lm_diff;
            }

            if !hybrid {
                target = compute_vbr(
                    mode,
                    &st.analysis,
                    base_target,
                    lm,
                    equiv_rate,
                    st.last_coded_bands,
                    c,
                    st.intensity,
                    st.constrained_vbr != 0,
                    st.stereo_saving,
                    tot_boost,
                    tf_estimate,
                    pitch_change,
                    max_depth,
                    st.lfe != 0,
                    st.has_energy_mask,
                    surround_masking,
                    temporal_vbr,
                    #[cfg(feature = "qext")]
                    (st.enable_qext != 0),
                );
            } else {
                target = base_target;
                // Tonal frames (offset<100) need more bits than noisy (offset>100) ones.
                if st.silk_info.offset < 100 {
                    target += 12 << BITRES >> (3 - lm);
                }
                if st.silk_info.offset > 100 {
                    target -= 18 << BITRES >> (3 - lm);
                }
                // Boosting bitrate on transients and vowels with significant temporal spikes.
                target += ((tf_estimate - 0.25f32) * (50 << BITRES) as f32) as i32;
                // If we have a strong transient, let's make sure it has enough bits to code the
                // first two bands, so that it can use folding rather than noise.
                if tf_estimate > 0.7f32 {
                    target = imax(target, 50 << BITRES);
                }
            }
            // The current offset is removed from the target and the space used so far is added
            target += tell;

            nb_available_bytes = (target + (1 << (BITRES + 2))) >> (BITRES + 3);
            nb_available_bytes = imax(min_allowed, nb_available_bytes);
            nb_available_bytes = imin(nb_compressed_bytes, nb_available_bytes);

            // By how much did we "miss" the target on that frame
            let mut delta = target - vbr_rate;

            target = nb_available_bytes << (BITRES + 3);

            // If the frame is silent we don't adjust our drift, otherwise the encoder will
            // shoot to very high rates after hitting a span of silence, but we do allow the
            // bitres to refill. This means that we'll undershoot our target in CVBR/VBR modes
            // on files with lots of silence.
            if silence {
                nb_available_bytes = 2;
                target = (2 * 8) << BITRES;
                delta = 0;
            }

            let alpha: OpusVal16 = if st.vbr_count < 970 {
                st.vbr_count += 1;
                celt_rcp((st.vbr_count + 20) as f32)
            } else {
                0.001f32
            };
            // How many bits have we used in excess of what we're allowed
            if st.constrained_vbr != 0 {
                st.vbr_reservoir += target - vbr_rate;
            }

            // Compute the offset we need to apply in order to reach the target
            if st.constrained_vbr != 0 {
                st.vbr_drift += (alpha
                    * ((delta * (1 << lm_diff)) - st.vbr_offset - st.vbr_drift) as f32)
                    as i32;
                st.vbr_offset = -st.vbr_drift;
            }

            if st.constrained_vbr != 0 && st.vbr_reservoir < 0 {
                // We're under the min value -- increase rate
                let adjust = (-st.vbr_reservoir) / (8 << BITRES);
                // Unless we're just coding silence
                nb_available_bytes += if silence { 0 } else { adjust };
                st.vbr_reservoir = 0;
            }
            nb_compressed_bytes = imin(nb_compressed_bytes, nb_available_bytes);
            // This moves the raw bits to take into account the new compressed size
            enc.shrink(nb_compressed_bytes as u32);
        }

        #[cfg(feature = "qext")]
        let mut qext_end: i32 = 0;
        #[cfg(feature = "qext")]
        let mut padding_len_bytes: i32 = 0;
        #[cfg(feature = "qext")]
        let mut qext_active_mode = false;
        #[cfg(feature = "qext")]
        let mut ext_storage: &mut [u8] = &mut [];
        #[cfg(feature = "qext")]
        if st.enable_qext != 0 {
            // Don't give any bits for the first 80 kb/s per channel. Then 80% of the excess.
            let offset = bitrate_to_bits(c * 80000, mode.fs, frame_size) / 8;
            qext_bytes = imax(
                nb_compressed_bytes - 1275,
                imax(0, (nb_compressed_bytes - offset) * 4 / 5),
            );
            if qext_bytes > 20 {
                let mut target: i32 = ((nb_compressed_bytes - qext_bytes / 3) * 8) << BITRES;
                if vbr_rate == 0 {
                    target -= (40 * c + 20) << BITRES;
                    let tf_estimate2: OpusVal16 = min32(1.0f32, 2.0 * tf_estimate);
                    target = compute_vbr(
                        mode,
                        &st.analysis,
                        target,
                        lm,
                        equiv_rate,
                        st.last_coded_bands,
                        c,
                        st.intensity,
                        st.constrained_vbr != 0,
                        st.stereo_saving,
                        tot_boost,
                        tf_estimate2,
                        pitch_change,
                        max_depth,
                        st.lfe != 0,
                        st.has_energy_mask,
                        surround_masking,
                        temporal_vbr,
                        st.enable_qext != 0,
                    );
                    target += tell;
                }
                let mut scale: OpusVal16 = toneishness;
                scale = Q15ONE - scale * scale;
                // C: qext_bytes += (float) → float addition, truncated back to int.
                qext_bytes = (qext_bytes as f32
                    + scale
                        * ((nb_compressed_bytes - (target / (8 << BITRES))) - qext_bytes) as f32)
                    as i32;
                qext_bytes = imax(nb_compressed_bytes - 1275, imax(21, qext_bytes));
            }
            padding_len_bytes = (qext_bytes + 253) / 254;
            qext_bytes = imin(
                qext_bytes,
                nb_compressed_bytes - min_allowed - padding_len_bytes - 1,
            );
            padding_len_bytes = (qext_bytes + 253) / 254;
            if qext_bytes > 20 {
                let new_compressed_bytes = nb_compressed_bytes - qext_bytes - padding_len_bytes - 1;
                enc.shrink(new_compressed_bytes as u32);
                // Code 3 packet. C writes compressed[-1] (the byte before the payload).
                if let Some(t) = toc {
                    *t |= 0x03;
                }
                let pad = padding_len_bytes as usize;
                let ncb = new_compressed_bytes as usize;
                let whole: &mut [u8] = core::mem::take(&mut enc.buf);
                whole.copy_within(0..ncb, 1 + pad);
                whole[0] = 0x41; // Set padding
                for i in 0..pad - 1 {
                    whole[i + 1] = 255;
                }
                whole[pad] = if qext_bytes % 254 == 0 {
                    254
                } else {
                    (qext_bytes % 254) as u8
                };
                let (_, rest) = whole.split_at_mut(1 + pad);
                let (main, ext_payload) = rest.split_at_mut(ncb);
                enc.buf = main;
                ext_payload[0] = (QEXT_EXTENSION_ID_BYTE) as u8;
                qext_bytes -= 1;
                let ext = &mut ext_payload[1..1 + qext_bytes as usize];
                ext.fill(0);
                ext_storage = ext;
                nb_compressed_bytes = new_compressed_bytes;
                if end == nb_ebands
                    && (mode.fs == 48000 || mode.fs == 96000)
                    && (mode.short_mdct_size == 120 * qext_scale
                        || mode.short_mdct_size == 90 * qext_scale)
                {
                    qext_active_mode = true;
                    qext_end = if qext_scale == 2 { NB_QEXT_BANDS } else { 2 };
                }
            } else {
                qext_bytes = 0;
            }
        }
        #[cfg(feature = "qext")]
        let mut ext_enc = EcEnc::new(ext_storage);
        #[cfg(feature = "qext")]
        let qext_mode: Option<&CeltMode> = if qext_active_mode {
            st.qext_mode.as_ref()
        } else {
            None
        };
        #[cfg(feature = "qext")]
        if qext_mode.is_some() {
            ext_enc.enc_bit_logp(qext_end == NB_QEXT_BANDS, 1);
        }

        // Bit allocation
        // bits =           packet size                    - where we are - safety
        let mut bits: i32 = ((nb_compressed_bytes * 8) << BITRES) - enc.tell_frac() as i32 - 1;
        let anti_collapse_rsv = if is_transient && lm >= 2 && bits >= ((lm + 2) << BITRES) {
            1 << BITRES
        } else {
            0
        };
        bits -= anti_collapse_rsv;
        let mut signal_bandwidth = end - 1;
        if st.analysis.valid != 0 {
            let min_bandwidth = if equiv_rate < 32000 * c {
                13
            } else if equiv_rate < 48000 * c {
                16
            } else if equiv_rate < 60000 * c {
                18
            } else if equiv_rate < 80000 * c {
                19
            } else {
                20
            };
            signal_bandwidth = imax(st.analysis.bandwidth, min_bandwidth);
        }
        if st.lfe != 0 {
            signal_bandwidth = 1;
        }
        let mut balance: i32 = 0;
        let coded_bands = clt_compute_allocation(
            mode,
            start,
            end,
            offsets,
            cap,
            alloc_trim,
            &mut st.intensity,
            &mut dual_stereo,
            bits,
            &mut balance,
            pulses,
            fine_quant,
            fine_priority,
            c,
            lm,
            &mut EcCoder::Enc(&mut *enc),
            st.last_coded_bands,
            signal_bandwidth,
        );
        if st.last_coded_bands != 0 {
            st.last_coded_bands = imin(
                st.last_coded_bands + 1,
                imax(st.last_coded_bands - 1, coded_bands),
            );
        } else {
            st.last_coded_bands = coded_bands;
        }

        quant_fine_energy(
            mode,
            start,
            end,
            &mut st.old_band_e,
            error,
            None,
            fine_quant,
            enc,
            c,
        );
        st.energy_error[..nb * ccu].fill(0.0);

        #[cfg(feature = "qext")]
        let mut qext_band_e = [0.0f32; 2 * QEXT_BANDS];
        #[cfg(feature = "qext")]
        let mut qext_band_log_e = [0.0f32; 2 * QEXT_BANDS];
        #[cfg(feature = "qext")]
        let mut qext_error = [0.0f32; 2 * QEXT_BANDS];
        #[cfg(feature = "qext")]
        let mut qext_intensity: i32 = 0;
        #[cfg(feature = "qext")]
        let mut qext_dual_stereo = false;
        #[cfg(feature = "qext")]
        if let Some(qm) = qext_mode {
            // Don't bias for intra.
            let mut qext_delayed_intra: OpusVal32 = 0.0;
            compute_band_energies(qm, freq, &mut qext_band_e, qext_end, c, lm);
            normalise_bands(qm, freq, x, &qext_band_e, qext_end, c, m_);
            amp2_log2(
                qm,
                qext_end,
                qext_end,
                &qext_band_e,
                &mut qext_band_log_e,
                c,
            );
            if c == 2 {
                qext_intensity = qext_end;
                qext_dual_stereo = dual_stereo != 0;
                encode_qext_stereo_params(&mut ext_enc, qext_end, qext_intensity, qext_dual_stereo);
            }
            quant_coarse_energy(
                qm,
                0,
                qext_end,
                qext_end,
                &qext_band_log_e,
                &mut st.qext_old_band_e,
                (qext_bytes * 8) as u32,
                &mut qext_error,
                &mut ext_enc,
                c,
                lm,
                qext_bytes,
                st.force_intra != 0,
                &mut qext_delayed_intra,
                st.complexity >= 4,
                st.loss_rate,
                st.lfe != 0,
            );
        }
        #[cfg(feature = "qext")]
        {
            let qext_bits: i32 = ((qext_bytes * 8) << BITRES) - enc.tell_frac() as i32 - 1;
            clt_compute_extra_allocation(
                mode,
                qext_mode,
                start,
                end,
                qext_end,
                band_log_e,
                &qext_band_log_e,
                qext_bits,
                extra_pulses,
                extra_quant,
                c,
                lm,
                &mut EcCoder::Enc(&mut ext_enc),
                tone_freq,
                toneishness,
            );
            error_bak[..cu * nb].copy_from_slice(error);
            if qext_bytes > 0 {
                quant_fine_energy(
                    mode,
                    start,
                    end,
                    &mut st.old_band_e,
                    error,
                    Some(fine_quant),
                    extra_quant,
                    &mut ext_enc,
                    c,
                );
            }
        }

        // Residual quantisation
        {
            let (x0, x1) = x.split_at_mut(nu);
            quant_all_bands(
                mode,
                start,
                end,
                x0,
                if c == 2 { Some(x1) } else { None },
                collapse_masks,
                band_e,
                pulses,
                short_blocks != 0,
                st.spread_decision,
                dual_stereo != 0,
                st.intensity,
                tf_res,
                nb_compressed_bytes * (8 << BITRES) - anti_collapse_rsv,
                balance,
                &mut EcCoder::Enc(&mut *enc),
                lm,
                coded_bands,
                &mut st.rng,
                st.complexity,
                st.disable_inv != 0,
                #[cfg(feature = "qext")]
                &mut EcCoder::Enc(&mut ext_enc),
                #[cfg(feature = "qext")]
                extra_pulses,
                #[cfg(feature = "qext")]
                (qext_bytes * (8 << BITRES)),
                #[cfg(feature = "qext")]
                Some(&cap[..]),
                bands,
            );
        }

        #[cfg(feature = "qext")]
        if let Some(qm) = qext_mode {
            let mut dummy_buf: [u8; 0] = [];
            let mut dummy_enc = EcEnc::new(&mut dummy_buf);
            let mut qext_collapse_masks = [0u8; 2 * QEXT_BANDS];
            zeros[..end as usize].fill(0);
            let mut ext_balance: i32 = qext_bytes * (8 << BITRES) - ext_enc.tell_frac() as i32;
            for i in 0..qext_end as usize {
                // C quirk: extra_quant[nbEBands+1], not [nbEBands+i].
                ext_balance -= extra_pulses[nb + i] + c * (extra_quant[nb + 1] << BITRES);
            }
            quant_fine_energy(
                qm,
                0,
                qext_end,
                &mut st.qext_old_band_e,
                &mut qext_error,
                None,
                &extra_quant[nb..],
                &mut ext_enc,
                c,
            );
            let (x0, x1) = x.split_at_mut(nu);
            quant_all_bands(
                qm,
                0,
                qext_end,
                x0,
                if c == 2 { Some(x1) } else { None },
                &mut qext_collapse_masks,
                &qext_band_e,
                &extra_pulses[nb..],
                short_blocks != 0,
                st.spread_decision,
                qext_dual_stereo,
                qext_intensity,
                zeros,
                qext_bytes * (8 << BITRES),
                ext_balance,
                &mut EcCoder::Enc(&mut ext_enc),
                lm,
                qext_end,
                &mut st.rng,
                st.complexity,
                st.disable_inv != 0,
                &mut EcCoder::Enc(&mut dummy_enc),
                zeros,
                0,
                None,
                bands,
            );
        }

        if anti_collapse_rsv > 0 {
            let anti_collapse_on = st.consec_transient < 2;
            enc.enc_bits(u32::from(anti_collapse_on), 1);
        }
        if qext_bytes == 0 {
            quant_energy_finalise(
                mode,
                start,
                end,
                Some(&mut st.old_band_e),
                error,
                fine_quant,
                fine_priority,
                nb_compressed_bytes * 8 - enc.tell(),
                enc,
                c,
            );
        }
        for ch in 0..cu {
            for i in start as usize..end as usize {
                let k = i + ch * nb;
                st.energy_error[k] = maxg(-0.5f32, ming(0.5f32, error[k]));
            }
        }
        #[cfg(feature = "qext")]
        if qext_bytes > 0 {
            quant_energy_finalise(
                mode,
                start,
                end,
                None,
                &mut error_bak[..cu * nb],
                fine_quant,
                fine_priority,
                nb_compressed_bytes * 8 - enc.tell(),
                enc,
                c,
            );
        }
        if silence {
            st.old_band_e[..cu * nb].fill(-28.0f32);
        }

        // RESYNTH: re-synthesis of the coded audio (using anti_collapse_on) not ported.

        st.prefilter_period = pitch_index;
        st.prefilter_gain = gain1;
        st.prefilter_tapset = prefilter_tapset;

        if cc == 2 && c == 1 {
            st.old_band_e.copy_within(0..nb, nb);
        }

        if !is_transient {
            st.old_log_e2[..ccu * nb].copy_from_slice(&st.old_log_e[..ccu * nb]);
            st.old_log_e[..ccu * nb].copy_from_slice(&st.old_band_e[..ccu * nb]);
        } else {
            for i in 0..ccu * nb {
                st.old_log_e[i] = ming(st.old_log_e[i], st.old_band_e[i]);
            }
        }
        // In case start or end were to change
        for ch in 0..ccu {
            for i in (0..start as usize).chain(end as usize..nb) {
                st.old_band_e[ch * nb + i] = 0.0;
                st.old_log_e[ch * nb + i] = -28.0f32;
                st.old_log_e2[ch * nb + i] = -28.0f32;
            }
        }

        if is_transient || transient_got_disabled {
            st.consec_transient += 1;
        } else {
            st.consec_transient = 0;
        }
        st.rng = enc.rng;

        // If there's any room left (can only happen for very high rates), it's already filled
        // with zeros
        enc.done();
        #[cfg(feature = "qext")]
        {
            ext_enc.done();
            if qext_bytes > 0 {
                nb_compressed_bytes += padding_len_bytes + 2 + qext_bytes;
            }
            if qext_bytes != 0 {
                st.rng ^= ext_enc.rng;
            }
            if ext_enc.get_error() != 0 {
                return OPUS_INTERNAL_ERROR;
            }
        }
        #[cfg(feature = "custom-modes")]
        if st.signalling != 0 {
            nb_compressed_bytes += 1;
        }

        if enc.get_error() != 0 {
            OPUS_INTERNAL_ERROR
        } else {
            nb_compressed_bytes
        }
    }
}

/// `QEXT_EXTENSION_ID<<1`: first byte of the QEXT padding extension.
#[cfg(feature = "qext")]
const QEXT_EXTENSION_ID_BYTE: i32 = crate::celt::celt::QEXT_EXTENSION_ID << 1;

// ---------------------------------------------------------------------------------------------
// Opus custom API (CUSTOM_MODES)
// ---------------------------------------------------------------------------------------------

/// Opus custom encoder (`OpusCustomEncoder`); the same type as [`CeltEncoder`].
#[cfg(feature = "custom-modes")]
pub type CustomEncoder = CeltEncoder;

#[cfg(feature = "custom-modes")]
impl CeltEncoder {
    /// Port of celt/celt_encoder.c:opus_custom_encode (float build): encodes 16-bit PCM.
    /// Returns the packet size or a negative error code.
    pub fn opus_custom_encode(
        &mut self,
        pcm: &[i16],
        frame_size: i32,
        compressed: &mut [u8],
        nb_compressed_bytes: i32,
    ) -> i32 {
        let n = (self.channels * frame_size).max(0) as usize;
        if pcm.len() < n {
            return OPUS_BAD_ARG;
        }
        let mut conv = core::mem::take(&mut self.scratch.pcm_conv);
        conv.clear();
        conv.extend(pcm[..n].iter().map(|&v| crate::celt::arch::int16tores(v)));
        let ret = self.celt_encode_with_ec(
            &conv,
            frame_size,
            Some(compressed),
            nb_compressed_bytes,
            None,
            #[cfg(feature = "qext")]
            None,
        );
        // RESYNTH: writing the re-synthesised signal back is not ported.
        self.scratch.pcm_conv = conv;
        ret
    }

    /// Port of celt/celt_encoder.c:opus_custom_encode24 (float build): encodes 24-bit PCM.
    /// Returns the packet size or a negative error code.
    pub fn opus_custom_encode24(
        &mut self,
        pcm: &[i32],
        frame_size: i32,
        compressed: &mut [u8],
        nb_compressed_bytes: i32,
    ) -> i32 {
        let n = (self.channels * frame_size).max(0) as usize;
        if pcm.len() < n {
            return OPUS_BAD_ARG;
        }
        let mut conv = core::mem::take(&mut self.scratch.pcm_conv);
        conv.clear();
        conv.extend(pcm[..n].iter().map(|&v| crate::celt::arch::int24tores(v)));
        let ret = self.celt_encode_with_ec(
            &conv,
            frame_size,
            Some(compressed),
            nb_compressed_bytes,
            None,
            #[cfg(feature = "qext")]
            None,
        );
        self.scratch.pcm_conv = conv;
        ret
    }

    /// Port of celt/celt_encoder.c:opus_custom_encode_float (float build): encodes float PCM.
    /// Returns the packet size or a negative error code.
    pub fn opus_custom_encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: i32,
        compressed: &mut [u8],
        nb_compressed_bytes: i32,
    ) -> i32 {
        self.celt_encode_with_ec(
            pcm,
            frame_size,
            Some(compressed),
            nb_compressed_bytes,
            None,
            #[cfg(feature = "qext")]
            None,
        )
    }
}
