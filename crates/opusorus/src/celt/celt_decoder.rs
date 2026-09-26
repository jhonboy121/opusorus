//! Port of celt/celt_decoder.c (+ the `CELTDecoder` declarations of celt/celt.h): the CELT
//! decoder (float build).
//!
//! * [`CeltDecoder`] is the C `struct OpusCustomDecoder`. The C trailing arrays (`_decode_mem`,
//!   `oldEBands`, `oldLogE`, `oldLogE2`, `backgroundLogE`, `lpc`) are owned `Vec`s sized at init;
//!   [`CeltDecoder::reset`] mirrors `OPUS_RESET_STATE` (the memset of everything after
//!   `DECODER_RESET_START`). The C VLAs are stack arrays (the frame-sized ones heap buffers
//!   kept by the state for large custom modes) and the `quant_all_bands` scratch is owned by
//!   the state, so decoding never allocates after the first frames.
//! * The decoder borrows its mode (`&'m CeltMode`), like the C pointer: the Opus decoder uses
//!   `CeltDecoder<'static>` with a static mode; custom modes are borrowed from the caller.
//! * [`CeltDecoder::celt_decode_with_ec`] / [`CeltDecoder::celt_decode_with_ec_dred`] take the
//!   packet as `Option<&[u8]>` + `len` (C `NULL` = packet loss), an optional shared range decoder
//!   (hybrid mode) and write `opus_res` (float) samples, accumulating when `accum` is set.
//! * The CTLs of `opus_custom_decoder_ctl` are typed methods ([`CeltDecoder::set_start_band`], …)
//!   plus a numeric dispatcher ([`CeltDecoder::ctl_set`] / [`CeltDecoder::ctl_get`]).
//! * With `custom-modes`, [`CustomDecoder`] is the `opus_custom_decoder_*` public API.
//!
//! Deviations (Rust-only guards where C would read/write out of bounds): `pcm` shorter than the
//! decoded frame, or `len` larger than the data slice, return [`Error::BadArg`]; a decoder with 0
//! channels is rejected at init. C's `celt_assert`s (`validate_celt_decoder`) are
//! `debug_assert!`s.
//!
//! DNN (`deep-plc` / `dred` features, C `ENABLE_DEEP_PLC` / `ENABLE_DRED`): the neural PLC of
//! `celt_decode_lost` (`FRAME_PLC_NEURAL` / `FRAME_DRED`, `update_plc_state`, the 16 → 48 kHz
//! resampling of the LPCNet concealment) runs when an [`LpcnetPlcState`] is passed through
//! [`CeltDecoder::celt_decode_with_ec_lpcnet`] (the C `lpcnet` argument of
//! `celt_decode_with_ec_dred`); [`CeltDecoder::celt_decode_with_ec`] and
//! [`CeltDecoder::celt_decode_with_ec_dred`] pass C's `NULL`.

#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]

use alloc::vec;
use alloc::vec::Vec;

use crate::celt::arch::{
    CeltGlog, CeltNorm, CeltSig, OpusRes, OpusVal16, OpusVal32, Q15ONE, Q31ONE, VERY_SMALL,
    add_res, add32, coef2val16, extend32, gconst, half32, imax, imin, max32, maxg, min32, ming,
    mult16_16, mult16_16_q15, mult16_32_q15, qconst16, saturate, shl32, shr32, sig2res, sround16,
};
use crate::celt::bands::{
    BandsScratch, SPREAD_NORMAL, anti_collapse, celt_lcg_rand, denormalise_bands, quant_all_bands,
};
#[cfg(all(feature = "custom-modes", feature = "qext"))]
use crate::celt::celt::QEXT_EXTENSION_ID;
#[cfg(feature = "custom-modes")]
use crate::celt::celt::from_opus;
use crate::celt::celt::{
    COMBFILTER_MINPERIOD, SPREAD_ICDF, TAPSET_ICDF, TF_SELECT_TABLE, TRIM_ICDF, comb_filter,
    comb_filter_inplace, init_caps, resampling_factor,
};
use crate::celt::celt_lpc::{
    _celt_autocorr, _celt_lpc, CELT_LPC_ORDER, celt_fir, celt_iir_inplace,
};
use crate::celt::entcode::{BITRES, EcCoder};
use crate::celt::entdec::EcDec;
use crate::celt::mathops::{celt_sqrt, frac_div32};
use crate::celt::mdct::clt_mdct_backward;
use crate::celt::modes::{DEC_PITCH_BUF_SIZE, MAX_PERIOD, opus_custom_mode_create};
#[cfg(feature = "qext")]
use crate::celt::modes::{NB_QEXT_BANDS, compute_qext_mode};
use crate::celt::pitch::{pitch_downsample, pitch_search};
use crate::celt::quant_bands::{
    unquant_coarse_energy, unquant_energy_finalise, unquant_fine_energy,
};
use crate::celt::rate::clt_compute_allocation;
#[cfg(feature = "qext")]
use crate::celt::rate::clt_compute_extra_allocation;
use crate::celt::static_modes::CeltMode;
use crate::celt::vq::renormalise_vector;
#[cfg(feature = "deep-plc")]
use crate::dnn::freq::{FRAME_SIZE as LPCNET_FRAME_SIZE, PREEMPHASIS as LPCNET_PREEMPHASIS};
#[cfg(feature = "deep-plc")]
use crate::dnn::lpcnet_plc::{LpcnetPlcState, lpcnet_plc_conceal, lpcnet_plc_update};
use crate::{Error, Result};

/// The maximum pitch lag to allow in the pitch-based PLC. It's possible to save CPU time in the
/// PLC pitch search by making this smaller than `MAX_PERIOD`. The current value corresponds to a
/// pitch of 66.67 Hz.
pub const PLC_PITCH_LAG_MAX: i32 = 720;
/// The minimum pitch lag to allow in the pitch-based PLC. This corresponds to a pitch of 480 Hz.
pub const PLC_PITCH_LAG_MIN: i32 = 100;

/// `FRAME_NONE`.
pub const FRAME_NONE: i32 = 0;
/// `FRAME_NORMAL`.
pub const FRAME_NORMAL: i32 = 1;
/// `FRAME_PLC_NOISE`.
pub const FRAME_PLC_NOISE: i32 = 2;
/// `FRAME_PLC_PERIODIC`.
pub const FRAME_PLC_PERIODIC: i32 = 3;
/// `FRAME_PLC_NEURAL`.
pub const FRAME_PLC_NEURAL: i32 = 4;
/// `FRAME_DRED`.
pub const FRAME_DRED: i32 = 5;

/// `DECODE_BUFFER_SIZE` (= `DEC_PITCH_BUF_SIZE`).
pub const DECODE_BUFFER_SIZE: i32 = DEC_PITCH_BUF_SIZE;

/// `PLC_UPDATE_FRAMES`.
pub const PLC_UPDATE_FRAMES: usize = 4;
/// `PLC_UPDATE_SAMPLES` (`PLC_UPDATE_FRAMES*FRAME_SIZE`, with the LPCNet `FRAME_SIZE` of 160).
#[cfg(feature = "deep-plc")]
pub const PLC_UPDATE_SAMPLES: usize = PLC_UPDATE_FRAMES * 160;

/// `SIG_SHIFT` (celt/arch.h); a no-op shift amount in the float build.
const SIG_SHIFT: i32 = 12;
/// `NORM_SHIFT` (celt/arch.h); a no-op shift amount in the float build.
const NORM_SHIFT: i32 = 24;
/// `SIG_SAT` (celt/arch.h); `SATURATE` is a no-op in the float build.
const SIG_SAT: i32 = 536_870_911;

/// `OPUS_SET_COMPLEXITY_REQUEST`.
pub const OPUS_SET_COMPLEXITY_REQUEST: i32 = 4010;
/// `OPUS_GET_COMPLEXITY_REQUEST`.
pub const OPUS_GET_COMPLEXITY_REQUEST: i32 = 4011;
/// `OPUS_GET_LOOKAHEAD_REQUEST`.
pub const OPUS_GET_LOOKAHEAD_REQUEST: i32 = 4027;
/// `OPUS_RESET_STATE`.
pub const OPUS_RESET_STATE: i32 = 4028;
/// `OPUS_GET_FINAL_RANGE_REQUEST`.
pub const OPUS_GET_FINAL_RANGE_REQUEST: i32 = 4031;
/// `OPUS_GET_PITCH_REQUEST`.
pub const OPUS_GET_PITCH_REQUEST: i32 = 4033;
/// `OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST`.
pub const OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST: i32 = 4046;
/// `OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST`.
pub const OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST: i32 = 4047;
pub use crate::celt::celt::{
    CELT_GET_AND_CLEAR_ERROR_REQUEST, CELT_GET_MODE_REQUEST, CELT_SET_CHANNELS_REQUEST,
    CELT_SET_END_BAND_REQUEST, CELT_SET_SIGNALLING_REQUEST, CELT_SET_START_BAND_REQUEST,
};

/// `N` up to which the frame-sized C VLAs of the decoder (`X`, `freq`, the `deemphasis`
/// scratch) are stack arrays sized for it: the 20 ms frames of the standard modes up to
/// 48 kHz (QEXT adds a second size for 96 kHz, see [`with_frame_bufs`]).
const STACK_N: usize = 960;

/// Largest `overlap` of any mode (`(shortMdctSize>>2)<<2` with shorts of at most 3.3 ms, i.e.
/// 320 samples at 96 kHz).
const MAX_OVERLAP: usize = 320;

/// Largest `max_period` (`QEXT_SCALE(MAX_PERIOD)`).
#[cfg(feature = "qext")]
const MAX_MAX_PERIOD: usize = 2 * MAX_PERIOD as usize;
/// Largest `max_period` (`QEXT_SCALE(MAX_PERIOD)`).
#[cfg(not(feature = "qext"))]
const MAX_MAX_PERIOD: usize = MAX_PERIOD as usize;

/// Replacements for the C VLAs of the decoder that are not stack arrays.
///
/// The frame-sized buffers are stack arrays (see [`with_frame_bufs`]) except for large
/// custom modes, which use these heap buffers (grown on first use, then kept so decoding
/// does not allocate). The PLC buffers (`exc`, `fir_tmp`, `lp_pitch_buf`, `etmp`, `buf_copy`)
/// are stack arrays.
#[derive(Debug, Clone, Default)]
struct DecoderScratch {
    /// `X` (`2*N` normalised MDCT coefficients) for `N > MAX_STACK_N`.
    x_heap: Vec<CeltNorm>,
    /// `freq` of `celt_synthesis`, reused as the `deemphasis` scratch (`N`), for
    /// `N > MAX_STACK_N`.
    freq_heap: Vec<CeltSig>,
    /// `tf_res`, `cap`, `offsets`, `fine_quant`, `pulses`, `fine_priority` (`nbEBands` each).
    tf_res: Vec<i32>,
    cap: Vec<i32>,
    offsets: Vec<i32>,
    fine_quant: Vec<i32>,
    pulses: Vec<i32>,
    fine_priority: Vec<i32>,
    /// `collapse_masks` (`2*nbEBands`).
    collapse_masks: Vec<u8>,
    /// QEXT: `extra_quant`, `extra_pulses` (`nbEBands+NB_QEXT_BANDS`), `zeros` (`nbEBands`),
    /// `qext_collapse_masks` (`2*NB_QEXT_BANDS`).
    #[cfg(feature = "qext")]
    extra_quant: Vec<i32>,
    #[cfg(feature = "qext")]
    extra_pulses: Vec<i32>,
    #[cfg(feature = "qext")]
    zeros: Vec<i32>,
    #[cfg(feature = "qext")]
    qext_collapse_masks: Vec<u8>,
    /// `quant_all_bands` scratch.
    bands: BandsScratch,
}

impl DecoderScratch {
    fn new(mode: &CeltMode) -> Self {
        let nb = mode.nb_ebands as usize;
        Self {
            x_heap: Vec::new(),
            freq_heap: Vec::new(),
            tf_res: vec![0; nb],
            cap: vec![0; nb],
            offsets: vec![0; nb],
            fine_quant: vec![0; nb],
            pulses: vec![0; nb],
            fine_priority: vec![0; nb],
            collapse_masks: vec![0; 2 * nb],
            #[cfg(feature = "qext")]
            extra_quant: vec![0; nb + NB_QEXT_BANDS as usize],
            #[cfg(feature = "qext")]
            extra_pulses: vec![0; nb + NB_QEXT_BANDS as usize],
            #[cfg(feature = "qext")]
            zeros: vec![0; nb],
            #[cfg(feature = "qext")]
            qext_collapse_masks: vec![0; 2 * NB_QEXT_BANDS as usize],
            bands: BandsScratch::new(),
        }
    }
}

/// The frame-sized C VLAs of one `celt_decode_with_ec_dred` call: `X` (`2*N`) and `freq` (`N`,
/// also the `deemphasis` scratch, which C allocates separately after `freq` is dead).
struct FrameBufs<'a> {
    x: &'a mut [CeltNorm],
    freq: &'a mut [CeltSig],
}

/// Runs `f` with the frame-sized VLAs for frames of `n` samples: zeroed stack arrays for
/// `n <= STACK_N` (and, with QEXT, `n <= 2*STACK_N`: 96 kHz frames, so 48 kHz frames only
/// clear the smaller arrays), else `x_heap` / `freq_heap`, grown on first use and kept.
fn with_frame_bufs<R>(
    n: usize,
    x_heap: &mut Vec<CeltNorm>,
    freq_heap: &mut Vec<CeltSig>,
    f: impl FnOnce(&mut FrameBufs<'_>) -> R,
) -> R {
    if n <= STACK_N {
        let mut x: [CeltNorm; 2 * STACK_N] = [0.0; 2 * STACK_N];
        let mut freq: [CeltSig; STACK_N] = [0.0; STACK_N];
        return f(&mut FrameBufs {
            x: &mut x[..2 * n],
            freq: &mut freq[..n],
        });
    }
    #[cfg(feature = "qext")]
    if n <= 2 * STACK_N {
        let mut x: [CeltNorm; 4 * STACK_N] = [0.0; 4 * STACK_N];
        let mut freq: [CeltSig; 2 * STACK_N] = [0.0; 2 * STACK_N];
        return f(&mut FrameBufs {
            x: &mut x[..2 * n],
            freq: &mut freq[..n],
        });
    }
    if x_heap.len() < 2 * n {
        x_heap.resize(2 * n, 0.0);
        freq_heap.resize(n, 0.0);
    }
    f(&mut FrameBufs {
        x: &mut x_heap[..2 * n],
        freq: &mut freq_heap[..n],
    })
}

/// Largest `N` whose frame-sized VLAs are stack arrays (see [`with_frame_bufs`]).
#[cfg(feature = "qext")]
const MAX_STACK_N: usize = 2 * STACK_N;
/// Largest `N` whose frame-sized VLAs are stack arrays (see [`with_frame_bufs`]).
#[cfg(not(feature = "qext"))]
const MAX_STACK_N: usize = STACK_N;

/// Port of celt/celt_decoder.c:struct OpusCustomDecoder (`CELTDecoder`): the CELT decoder state.
#[derive(Debug, Clone)]
pub struct CeltDecoder<'m> {
    /// `mode`.
    pub mode: &'m CeltMode,
    /// `overlap`.
    pub overlap: i32,
    /// `channels`: output channels.
    pub channels: i32,
    /// `stream_channels`: coded channels.
    pub stream_channels: i32,
    /// `downsample`.
    pub downsample: i32,
    /// `start`.
    pub start: i32,
    /// `end`.
    pub end: i32,
    /// `signalling` (only used with `custom-modes`).
    pub signalling: i32,
    /// `disable_inv`.
    pub disable_inv: i32,
    /// `complexity`.
    pub complexity: i32,
    /// `qext_scale`: 2 for the 96 kHz modes, else 1.
    #[cfg(feature = "qext")]
    pub qext_scale: i32,

    // Everything beyond this point gets cleared on a reset (`DECODER_RESET_START` = `rng`).
    /// `rng`.
    pub rng: u32,
    /// `error`.
    pub error: i32,
    /// `last_pitch_index`.
    pub last_pitch_index: i32,
    /// `loss_duration`.
    pub loss_duration: i32,
    /// `plc_duration`.
    pub plc_duration: i32,
    /// `last_frame_type`.
    pub last_frame_type: i32,
    /// `skip_plc`.
    pub skip_plc: i32,
    /// `postfilter_period`.
    pub postfilter_period: i32,
    /// `postfilter_period_old`.
    pub postfilter_period_old: i32,
    /// `postfilter_gain`.
    pub postfilter_gain: OpusVal16,
    /// `postfilter_gain_old`.
    pub postfilter_gain_old: OpusVal16,
    /// `postfilter_tapset`.
    pub postfilter_tapset: i32,
    /// `postfilter_tapset_old`.
    pub postfilter_tapset_old: i32,
    /// `prefilter_and_fold`.
    pub prefilter_and_fold: i32,
    /// `preemph_memD`.
    pub preemph_mem_d: [CeltSig; 2],
    /// `plc_pcm` (deep PLC): LPCNet concealment output at 16 kHz waiting to be resampled.
    #[cfg(feature = "deep-plc")]
    pub plc_pcm: [i16; PLC_UPDATE_SAMPLES],
    /// `plc_fill` (deep PLC).
    #[cfg(feature = "deep-plc")]
    pub plc_fill: i32,
    /// `plc_preemphasis_mem` (deep PLC).
    #[cfg(feature = "deep-plc")]
    pub plc_preemphasis_mem: f32,
    /// `qext_oldBandE`.
    #[cfg(feature = "qext")]
    pub qext_old_band_e: [CeltGlog; 2 * NB_QEXT_BANDS as usize],
    /// `_decode_mem`: `channels*(DECODE_BUFFER_SIZE*qext_scale+overlap)`.
    pub decode_mem: Vec<CeltSig>,
    /// `oldEBands` (`oldBandE`): `2*nbEBands`.
    pub old_ebands: Vec<CeltGlog>,
    /// `oldLogE`: `2*nbEBands`.
    pub old_log_e: Vec<CeltGlog>,
    /// `oldLogE2`: `2*nbEBands`.
    pub old_log_e2: Vec<CeltGlog>,
    /// `backgroundLogE`: `2*nbEBands`.
    pub background_log_e: Vec<CeltGlog>,
    /// `lpc`: `channels*CELT_LPC_ORDER`.
    pub lpc: Vec<OpusVal16>,

    scratch: DecoderScratch,
}

/// `celt_decoder_get_size` equivalent: bytes used by a decoder (struct + owned buffers) for the
/// default mode (96 kHz with `qext`, else 48 kHz). The C value is the size of the C struct, so
/// the two differ; this exists for API parity (`opus_decoder_get_size`).
#[must_use]
pub fn celt_decoder_get_size(channels: i32) -> i32 {
    celt_decoder_get_size_for_rate(if cfg!(feature = "qext") { 96000 } else { 48000 }, channels)
}

/// [`celt_decoder_get_size`] for the mode of an Opus decoder at `fs` Hz (the 48 kHz mode, or
/// the 96 kHz one with QEXT).
#[must_use]
pub fn celt_decoder_get_size_for_rate(fs: i32, channels: i32) -> i32 {
    let mode = opus_custom_mode_create(if fs > 48000 { 96000 } else { 48000 }, 960);
    #[expect(clippy::expect_used, reason = "the static default modes always exist")]
    opus_custom_decoder_get_size(mode.expect("static mode"), channels)
}

/// Port of celt/celt_decoder.c:opus_custom_decoder_get_size (Rust memory footprint, see
/// [`celt_decoder_get_size`]): the struct, its buffers (`_decode_mem`, the energy histories,
/// `lpc`, the small allocation scratch), the `quant_all_bands` scratch once grown by decoding
/// (up to two coded channels, any frame size) and the heap `X` / `freq`
/// buffers (`N > MAX_STACK_N`: large custom modes). The other C VLAs are stack arrays.
#[must_use]
pub fn opus_custom_decoder_get_size(mode: &CeltMode, channels: i32) -> i32 {
    #[cfg(feature = "qext")]
    let qext_scale =
        if mode.fs == 96000 && (mode.short_mdct_size == 240 || mode.short_mdct_size == 180) {
            2
        } else {
            1
        };
    #[cfg(not(feature = "qext"))]
    let qext_scale = 1;
    let c = channels as usize;
    let n_max = (mode.short_mdct_size * mode.nb_short_mdcts) as usize;
    let nb = mode.nb_ebands as usize;
    let state = c
        * ((qext_scale * DECODE_BUFFER_SIZE) as usize + mode.overlap as usize)
        * size_of::<CeltSig>()
        + 4 * 2 * nb * size_of::<CeltGlog>()
        + c * CELT_LPC_ORDER * size_of::<OpusVal16>();
    // tf_res, cap, offsets, fine_quant, pulses, fine_priority, collapse_masks.
    let alloc = 6 * nb * size_of::<i32>() + 2 * nb;
    #[cfg(feature = "qext")]
    let alloc = alloc
        + (2 * (nb + NB_QEXT_BANDS as usize) + nb) * size_of::<i32>()
        + 2 * NB_QEXT_BANDS as usize;
    let frame = if n_max > MAX_STACK_N {
        2 * n_max * size_of::<CeltNorm>() + n_max * size_of::<CeltSig>()
    } else {
        0
    };
    let size = size_of::<CeltDecoder<'static>>()
        + state
        + alloc
        + bands_scratch_size(mode, mode.max_lm)
        + frame;
    size as i32
}

/// Bytes of the `quant_all_bands` scratch ([`BandsScratch`]) after decoding frames of up to
/// `2^max_lm` short MDCTs of `mode` with up to two coded channels: `norm`
/// (`C*M*(eBands[nbEBands-1]-eBands[start])`) and ten buffers of the widest band, for the
/// mode's bands and (QEXT) the extension bands.
fn bands_scratch_size(mode: &CeltMode, max_lm: i32) -> usize {
    let m = 1usize << max_lm;
    let bands = |e: &[i16], nb: usize| {
        let norm = 2 * m * (e[nb - 1] - e[0]) as usize;
        let widest = e[..=nb]
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .map_or(0, |w| w * m);
        (norm, widest)
    };
    #[allow(unused_mut, reason = "only updated with QEXT")]
    let (mut norm, mut widest) = bands(&mode.e_bands, mode.nb_ebands as usize);
    // The modes `decode_frame` runs the QEXT bands for.
    #[cfg(feature = "qext")]
    if (mode.fs == 48000 && (mode.short_mdct_size == 120 || mode.short_mdct_size == 90))
        || (mode.fs == 96000 && (mode.short_mdct_size == 240 || mode.short_mdct_size == 180))
    {
        let q = compute_qext_mode(mode);
        let (n, w) = bands(&q.e_bands, q.nb_ebands as usize);
        norm = norm.max(n);
        widest = widest.max(w);
    }
    (norm + 10 * widest) * size_of::<CeltNorm>()
}

/// Port of celt/celt_decoder.c:deemphasis_stereo_simple: special case for stereo with no
/// downsampling and no accumulation (only compiled without custom modes / QEXT, as in C).
#[cfg(not(any(feature = "custom-modes", feature = "qext")))]
fn deemphasis_stereo_simple(
    in0: &[CeltSig],
    in1: &[CeltSig],
    pcm: &mut [OpusRes],
    n: usize,
    coef0: OpusVal16,
    mem: &mut [CeltSig; 2],
) {
    let x0 = &in0[..n];
    let x1 = &in1[..n];
    let pcm = &mut pcm[..2 * n];
    let mut m0 = mem[0];
    let mut m1 = mem[1];
    for j in 0..n {
        // Add VERY_SMALL to x[] first to reduce dependency chain.
        let tmp0: CeltSig = saturate(x0[j] + VERY_SMALL + m0, SIG_SAT);
        let tmp1: CeltSig = saturate(x1[j] + VERY_SMALL + m1, SIG_SAT);
        m0 = mult16_32_q15(coef0, tmp0);
        m1 = mult16_32_q15(coef0, tmp1);
        pcm[2 * j] = sig2res(tmp0);
        pcm[2 * j + 1] = sig2res(tmp1);
    }
    mem[0] = m0;
    mem[1] = m1;
}

/// Port of celt/celt_decoder.c:deemphasis: de-emphasis filter, downsampling and interleaving of
/// `c` channels (`input[ch][..n]`) into `pcm` (accumulating when `accum`).
///
/// Writes `pcm[..(n/downsample)*c]`. `scratch` needs `n` samples (C VLA).
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub fn deemphasis(
    input: &[&[CeltSig]],
    pcm: &mut [OpusRes],
    n: usize,
    c: usize,
    downsample: i32,
    coef: &[OpusVal16; 4],
    mem: &mut [CeltSig; 2],
    accum: bool,
    scratch: &mut [CeltSig],
) {
    #[cfg(not(any(feature = "custom-modes", feature = "qext")))]
    {
        // Short version for common case.
        if downsample == 1 && c == 2 && !accum {
            deemphasis_stereo_simple(input[0], input[1], pcm, n, coef[0], mem);
            return;
        }
    }
    let mut apply_downsampling = false;
    let scratch = &mut scratch[..n];
    let coef0 = coef[0];
    let ds = downsample as usize;
    let nd = n / ds;
    // C: `#if defined(CUSTOM_MODES) || ... || defined(ENABLE_QEXT)` around this branch.
    let custom = cfg!(any(feature = "custom-modes", feature = "qext")) && coef[1] != 0.0;
    if c == 2 && downsample == 1 {
        // Perf: stereo without downsampling with both channels in one loop, so that their
        // recursions overlap; the same operations per sample as the per-channel loops below
        // (with `downsample == 1` the custom filter output goes straight to `pcm`).
        let (x0, x1) = (&input[0][..n], &input[1][..n]);
        let pcm = &mut pcm[..2 * n];
        let [mut m0, mut m1] = *mem;
        if custom {
            let coef1 = coef[1];
            let coef3 = coef[3];
            for ((y, &a), &b) in pcm.as_chunks_mut::<2>().0.iter_mut().zip(x0).zip(x1) {
                let mut tmp0: CeltSig = saturate(a + m0 + VERY_SMALL, SIG_SAT);
                let mut tmp1: CeltSig = saturate(b + m1 + VERY_SMALL, SIG_SAT);
                m0 = mult16_32_q15(coef0, tmp0) - mult16_32_q15(coef1, a);
                m1 = mult16_32_q15(coef0, tmp1) - mult16_32_q15(coef1, b);
                tmp0 = shl32(mult16_32_q15(coef3, tmp0), 2);
                tmp1 = shl32(mult16_32_q15(coef3, tmp1), 2);
                if accum {
                    y[0] = add_res(y[0], sig2res(tmp0));
                    y[1] = add_res(y[1], sig2res(tmp1));
                } else {
                    y[0] = sig2res(tmp0);
                    y[1] = sig2res(tmp1);
                }
            }
        } else if accum {
            for ((y, &a), &b) in pcm.as_chunks_mut::<2>().0.iter_mut().zip(x0).zip(x1) {
                let tmp0: CeltSig = saturate(a + m0 + VERY_SMALL, SIG_SAT);
                let tmp1: CeltSig = saturate(b + m1 + VERY_SMALL, SIG_SAT);
                m0 = mult16_32_q15(coef0, tmp0);
                m1 = mult16_32_q15(coef0, tmp1);
                y[0] = add_res(y[0], sig2res(tmp0));
                y[1] = add_res(y[1], sig2res(tmp1));
            }
        } else {
            for ((y, &a), &b) in pcm.as_chunks_mut::<2>().0.iter_mut().zip(x0).zip(x1) {
                let tmp0: CeltSig = saturate(a + VERY_SMALL + m0, SIG_SAT);
                let tmp1: CeltSig = saturate(b + VERY_SMALL + m1, SIG_SAT);
                m0 = mult16_32_q15(coef0, tmp0);
                m1 = mult16_32_q15(coef0, tmp1);
                y[0] = sig2res(tmp0);
                y[1] = sig2res(tmp1);
            }
        }
        *mem = [m0, m1];
        return;
    }
    for ch in 0..c {
        let mut m: CeltSig = mem[ch];
        let x = &input[ch][..n];
        let y = &mut pcm[ch..];
        if custom {
            let coef1 = coef[1];
            let coef3 = coef[3];
            for j in 0..n {
                let mut tmp: CeltSig = saturate(x[j] + m + VERY_SMALL, SIG_SAT);
                m = mult16_32_q15(coef0, tmp) - mult16_32_q15(coef1, x[j]);
                tmp = shl32(mult16_32_q15(coef3, tmp), 2);
                scratch[j] = tmp;
            }
            apply_downsampling = true;
        } else if downsample > 1 {
            // Shortcut for the standard (non-custom modes) case
            for j in 0..n {
                let tmp: CeltSig = saturate(x[j] + VERY_SMALL + m, SIG_SAT);
                m = mult16_32_q15(coef0, tmp);
                scratch[j] = tmp;
            }
            apply_downsampling = true;
        } else {
            // Shortcut for the standard (non-custom modes) case
            if accum {
                for j in 0..n {
                    let tmp: CeltSig = saturate(x[j] + m + VERY_SMALL, SIG_SAT);
                    m = mult16_32_q15(coef0, tmp);
                    y[j * c] = add_res(y[j * c], sig2res(tmp));
                }
            } else {
                for j in 0..n {
                    let tmp: CeltSig = saturate(x[j] + VERY_SMALL + m, SIG_SAT);
                    m = mult16_32_q15(coef0, tmp);
                    y[j * c] = sig2res(tmp);
                }
            }
        }
        mem[ch] = m;

        if apply_downsampling {
            // Perform down-sampling
            if accum {
                for j in 0..nd {
                    y[j * c] = add_res(y[j * c], sig2res(scratch[j * ds]));
                }
            } else {
                for j in 0..nd {
                    y[j * c] = sig2res(scratch[j * ds]);
                }
            }
        }
    }
}

/// Port of celt/celt_decoder.c:celt_synthesis: denormalises the decoded spectrum `x` (`c`
/// stream channels of `N = shortMdctSize<<lm`) and runs the inverse MDCT into the `cc` output
/// channels.
///
/// `out_syn[ch]` starts at the C `out_syn[ch]` pointer and must hold `N + overlap` samples
/// (the IMDCT writes `N + overlap/2` and the mono→stereo path uses `out_syn[1]` as temporary
/// storage). `freq` needs `N` samples (C VLA).
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub fn celt_synthesis(
    mode: &CeltMode,
    x: &[CeltNorm],
    out_syn: &mut [&mut [CeltSig]],
    old_band_e: &[CeltGlog],
    start: i32,
    eff_end: i32,
    c: i32,
    cc: i32,
    is_transient: bool,
    lm: i32,
    downsample: i32,
    silence: bool,
    #[cfg(feature = "qext")] qext_mode: Option<&CeltMode>,
    #[cfg(feature = "qext")] qext_band_log_e: &[CeltGlog],
    #[cfg(feature = "qext")] mut qext_end: i32,
    freq: &mut [CeltSig],
) {
    let overlap = mode.overlap as usize;
    let nb_ebands = mode.nb_ebands as usize;
    let n = (mode.short_mdct_size << lm) as usize;
    let freq = &mut freq[..n];
    let m = 1 << lm;
    #[cfg(feature = "qext")]
    if mode.fs != 96000 {
        qext_end = 2;
    }
    let window = &mode.window[..];

    let (b_count, nb, shift) = if is_transient {
        (
            m as usize,
            mode.short_mdct_size as usize,
            mode.max_lm as usize,
        )
    } else {
        (
            1usize,
            (mode.short_mdct_size << lm) as usize,
            (mode.max_lm - lm) as usize,
        )
    };

    if cc == 2 && c == 1 {
        // Copying a mono streams to two channels
        denormalise_bands(
            mode, x, freq, old_band_e, start, eff_end, m, downsample, silence,
        );
        #[cfg(feature = "qext")]
        if let Some(qm) = qext_mode {
            denormalise_bands(
                qm,
                x,
                freq,
                qext_band_log_e,
                0,
                qext_end,
                m,
                downsample,
                silence,
            );
        }
        // Store a temporary copy in the output buffer because the IMDCT destroys its input.
        let (o0, o1) = out_syn.split_at_mut(1);
        let out0 = &mut *o0[0];
        let out1 = &mut *o1[0];
        let ov2 = overlap / 2;
        out1[ov2..ov2 + n].copy_from_slice(freq);
        for b in 0..b_count {
            clt_mdct_backward(
                &mode.mdct,
                &out1[ov2 + b..],
                &mut out0[nb * b..],
                window,
                overlap,
                shift,
                b_count,
            );
        }
        for b in 0..b_count {
            clt_mdct_backward(
                &mode.mdct,
                &freq[b..],
                &mut out1[nb * b..],
                window,
                overlap,
                shift,
                b_count,
            );
        }
    } else if cc == 1 && c == 2 {
        // Downmixing a stereo stream to mono
        let out0 = &mut *out_syn[0];
        let ov2 = overlap / 2;
        denormalise_bands(
            mode, x, freq, old_band_e, start, eff_end, m, downsample, silence,
        );
        // Use the output buffer as temp array before downmixing.
        {
            let freq2 = &mut out0[ov2..ov2 + n];
            denormalise_bands(
                mode,
                &x[n..],
                freq2,
                &old_band_e[nb_ebands..],
                start,
                eff_end,
                m,
                downsample,
                silence,
            );
        }
        #[cfg(feature = "qext")]
        if let Some(qm) = qext_mode {
            denormalise_bands(
                qm,
                x,
                freq,
                qext_band_log_e,
                0,
                qext_end,
                m,
                downsample,
                silence,
            );
            let freq2 = &mut out0[ov2..ov2 + n];
            denormalise_bands(
                qm,
                &x[n..],
                freq2,
                &qext_band_log_e[NB_QEXT_BANDS as usize..],
                0,
                qext_end,
                m,
                downsample,
                silence,
            );
        }
        for i in 0..n {
            freq[i] = add32(half32(freq[i]), half32(out0[ov2 + i]));
        }
        for b in 0..b_count {
            clt_mdct_backward(
                &mode.mdct,
                &freq[b..],
                &mut out0[nb * b..],
                window,
                overlap,
                shift,
                b_count,
            );
        }
    } else {
        // Normal case (mono or stereo)
        for ch in 0..cc as usize {
            denormalise_bands(
                mode,
                &x[ch * n..],
                freq,
                &old_band_e[ch * nb_ebands..],
                start,
                eff_end,
                m,
                downsample,
                silence,
            );
            #[cfg(feature = "qext")]
            if let Some(qm) = qext_mode {
                denormalise_bands(
                    qm,
                    &x[ch * n..],
                    freq,
                    &qext_band_log_e[ch * NB_QEXT_BANDS as usize..],
                    0,
                    qext_end,
                    m,
                    downsample,
                    silence,
                );
            }
            let out = &mut *out_syn[ch];
            for b in 0..b_count {
                clt_mdct_backward(
                    &mode.mdct,
                    &freq[b..],
                    &mut out[nb * b..],
                    window,
                    overlap,
                    shift,
                    b_count,
                );
            }
        }
    }
    // Saturate IMDCT output so that we can't overflow in the pitch postfilter or in the
    // deemphasis.
    for ch in 0..cc as usize {
        for v in out_syn[ch][..n].iter_mut() {
            *v = saturate(*v, SIG_SAT);
        }
    }
}

/// Port of celt/celt_decoder.c:tf_decode: decodes the per-band time-frequency resolution
/// changes into `tf_res[start..end]`.
pub fn tf_decode(
    start: i32,
    end: i32,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: i32,
    dec: &mut EcDec<'_>,
) {
    let mut budget: u32 = dec.storage * 8;
    let mut tell: u32 = dec.tell() as u32;
    let mut logp: u32 = if is_transient { 2 } else { 4 };
    #[expect(clippy::int_plus_one, reason = "mirrors the C expression")]
    let tf_select_rsv = lm > 0 && tell + logp + 1 <= budget;
    budget -= u32::from(tf_select_rsv);
    let mut tf_changed: i32 = 0;
    let mut curr: i32 = 0;
    for i in start as usize..end as usize {
        if tell + logp <= budget {
            curr ^= i32::from(dec.dec_bit_logp(logp));
            tell = dec.tell() as u32;
            tf_changed |= curr;
        }
        tf_res[i] = curr;
        logp = if is_transient { 4 } else { 5 };
    }
    let mut tf_select: i32 = 0;
    let it = 4 * i32::from(is_transient);
    let row = &TF_SELECT_TABLE[lm as usize];
    if tf_select_rsv && row[(it + tf_changed) as usize] != row[(it + 2 + tf_changed) as usize] {
        tf_select = i32::from(dec.dec_bit_logp(1));
    }
    for i in start as usize..end as usize {
        tf_res[i] = i32::from(row[(it + 2 * tf_select + tf_res[i]) as usize]);
    }
}

/// Port of celt/celt_decoder.c:celt_plc_pitch_search: pitch search on the past decoded signal
/// `decode_mem[..c]` (each channel `DECODE_BUFFER_SIZE*qext_scale` samples) for the pitch PLC.
/// `lp_pitch_buf` needs `DECODE_BUFFER_SIZE>>1` samples (C VLA).
#[must_use]
pub fn celt_plc_pitch_search(
    decode_mem: &[&[CeltSig]],
    c: usize,
    qext_scale: i32,
    lp_pitch_buf: &mut [OpusVal16],
) -> i32 {
    let half = (DECODE_BUFFER_SIZE >> 1) as usize;
    let lp = &mut lp_pitch_buf[..half];
    pitch_downsample(decode_mem, lp, half, c, (qext_scale * 2) as usize);
    let mut pitch_index = pitch_search(
        &lp[(PLC_PITCH_LAG_MAX >> 1) as usize..],
        lp,
        (DECODE_BUFFER_SIZE - PLC_PITCH_LAG_MAX) as usize,
        (PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN) as usize,
    );
    pitch_index = PLC_PITCH_LAG_MAX - pitch_index;
    qext_scale * pitch_index
}

/// Port of celt/celt_decoder.c:decode_qext_stereo_params.
#[cfg(feature = "qext")]
fn decode_qext_stereo_params(
    ec: &mut EcDec<'_>,
    qext_end: i32,
    qext_intensity: &mut i32,
    qext_dual_stereo: &mut i32,
) {
    *qext_intensity = ec.dec_uint((qext_end + 1) as u32) as i32;
    if *qext_intensity != 0 {
        *qext_dual_stereo = i32::from(ec.dec_bit_logp(1));
    } else {
        *qext_dual_stereo = 0;
    }
}

/// `validate_celt_decoder` checks (hardened builds): basic checks on the CELT state to ensure
/// we don't end up writing all over memory. These are `debug_assert!`s.
pub fn validate_celt_decoder(st: &CeltDecoder<'_>) {
    #[cfg(not(any(feature = "custom-modes", feature = "qext")))]
    {
        debug_assert!(opus_custom_mode_create(48000, 960).is_ok_and(|m| core::ptr::eq(m, st.mode)));
        debug_assert!(st.overlap == 120);
        debug_assert!(st.end <= 21);
    }
    #[cfg(any(feature = "custom-modes", feature = "qext"))]
    {
        // From Section 4.3 in the spec: "The normal CELT layer uses 21 of those bands, though
        // Opus Custom (see Section 6.2) may use a different number of bands". Check if it's
        // within the maximum number of Bark frequency bands instead.
        debug_assert!(st.end <= 25);
    }
    debug_assert!(st.channels == 1 || st.channels == 2);
    debug_assert!(st.stream_channels == 1 || st.stream_channels == 2);
    debug_assert!(st.downsample > 0);
    debug_assert!(st.start == 0 || st.start == 17);
    debug_assert!(st.start < st.end);
    #[cfg(not(feature = "qext"))]
    {
        debug_assert!(st.last_pitch_index <= PLC_PITCH_LAG_MAX);
        debug_assert!(st.last_pitch_index >= PLC_PITCH_LAG_MIN || st.last_pitch_index == 0);
    }
    debug_assert!(st.postfilter_period < MAX_PERIOD);
    debug_assert!(st.postfilter_period >= COMBFILTER_MINPERIOD || st.postfilter_period == 0);
    debug_assert!(st.postfilter_period_old < MAX_PERIOD);
    debug_assert!(
        st.postfilter_period_old >= COMBFILTER_MINPERIOD || st.postfilter_period_old == 0
    );
    debug_assert!(st.postfilter_tapset <= 2);
    debug_assert!(st.postfilter_tapset >= 0);
    debug_assert!(st.postfilter_tapset_old <= 2);
    debug_assert!(st.postfilter_tapset_old >= 0);
}

impl CeltDecoder<'static> {
    /// Port of celt/celt_decoder.c:celt_decoder_init: a decoder for the standard 48 kHz mode
    /// (96 kHz mode with `qext` when `sampling_rate == 96000`), downsampling to
    /// `sampling_rate` (48/24/16/12/8 kHz).
    pub fn celt_decoder_init(sampling_rate: i32, channels: i32) -> Result<Self> {
        #[cfg(feature = "qext")]
        if sampling_rate == 96000 {
            return CeltDecoder::opus_custom_decoder_init(
                opus_custom_mode_create(96000, 960)?,
                channels,
            );
        }
        let mut st =
            CeltDecoder::opus_custom_decoder_init(opus_custom_mode_create(48000, 960)?, channels)?;
        st.downsample = resampling_factor(sampling_rate);
        if st.downsample == 0 {
            Err(Error::BadArg)
        } else {
            Ok(st)
        }
    }
}

impl<'m> CeltDecoder<'m> {
    /// Port of celt/celt_decoder.c:opus_custom_decoder_init: a decoder for `mode` with
    /// `channels` (1 or 2) output channels.
    pub fn opus_custom_decoder_init(mode: &'m CeltMode, channels: i32) -> Result<Self> {
        // C accepts channels == 0 (and then misbehaves); Rust rejects it.
        if !(1..=2).contains(&channels) {
            return Err(Error::BadArg);
        }
        #[cfg(feature = "qext")]
        let qext_scale =
            if mode.fs == 96000 && (mode.short_mdct_size == 240 || mode.short_mdct_size == 180) {
                2
            } else {
                1
            };
        #[cfg(not(feature = "qext"))]
        let qext_scale = 1;
        let overlap = mode.overlap;
        let nb = mode.nb_ebands as usize;
        let dbs = (qext_scale * DECODE_BUFFER_SIZE) as usize;
        let mut st = Self {
            mode,
            overlap,
            channels,
            stream_channels: channels,
            downsample: 1,
            start: 0,
            end: mode.eff_ebands,
            signalling: 1,
            // DISABLE_UPDATE_DRAFT is not defined.
            disable_inv: i32::from(channels == 1),
            complexity: 0,
            #[cfg(feature = "qext")]
            qext_scale,
            rng: 0,
            error: 0,
            last_pitch_index: 0,
            loss_duration: 0,
            plc_duration: 0,
            last_frame_type: 0,
            skip_plc: 0,
            postfilter_period: 0,
            postfilter_period_old: 0,
            postfilter_gain: 0.0,
            postfilter_gain_old: 0.0,
            postfilter_tapset: 0,
            postfilter_tapset_old: 0,
            prefilter_and_fold: 0,
            preemph_mem_d: [0.0; 2],
            #[cfg(feature = "deep-plc")]
            plc_pcm: [0; PLC_UPDATE_SAMPLES],
            #[cfg(feature = "deep-plc")]
            plc_fill: 0,
            #[cfg(feature = "deep-plc")]
            plc_preemphasis_mem: 0.0,
            #[cfg(feature = "qext")]
            qext_old_band_e: [0.0; 2 * NB_QEXT_BANDS as usize],
            decode_mem: vec![0.0; channels as usize * (dbs + overlap as usize)],
            old_ebands: vec![0.0; 2 * nb],
            old_log_e: vec![0.0; 2 * nb],
            old_log_e2: vec![0.0; 2 * nb],
            background_log_e: vec![0.0; 2 * nb],
            lpc: vec![0.0; channels as usize * CELT_LPC_ORDER],
            scratch: DecoderScratch::new(mode),
        };
        st.reset();
        Ok(st)
    }

    /// `QEXT_SCALE(1)` for this decoder.
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

    /// `QEXT_SCALE(DECODE_BUFFER_SIZE)`.
    #[inline(always)]
    const fn decode_buffer_size(&self) -> usize {
        (self.qext_scale() * DECODE_BUFFER_SIZE) as usize
    }

    /// `OPUS_RESET_STATE`: clears everything after `DECODER_RESET_START` and re-initialises the
    /// energy history.
    pub fn reset(&mut self) {
        self.rng = 0;
        self.error = 0;
        self.last_pitch_index = 0;
        self.loss_duration = 0;
        self.plc_duration = 0;
        self.last_frame_type = 0;
        self.skip_plc = 0;
        self.postfilter_period = 0;
        self.postfilter_period_old = 0;
        self.postfilter_gain = 0.0;
        self.postfilter_gain_old = 0.0;
        self.postfilter_tapset = 0;
        self.postfilter_tapset_old = 0;
        self.prefilter_and_fold = 0;
        self.preemph_mem_d = [0.0; 2];
        #[cfg(feature = "deep-plc")]
        {
            self.plc_pcm = [0; PLC_UPDATE_SAMPLES];
            self.plc_fill = 0;
            self.plc_preemphasis_mem = 0.0;
        }
        #[cfg(feature = "qext")]
        {
            self.qext_old_band_e = [0.0; 2 * NB_QEXT_BANDS as usize];
        }
        self.decode_mem.fill(0.0);
        self.old_ebands.fill(0.0);
        self.background_log_e.fill(0.0);
        self.lpc.fill(0.0);
        self.old_log_e.fill(-gconst(28.0));
        self.old_log_e2.fill(-gconst(28.0));
        self.skip_plc = 1;
        self.last_frame_type = FRAME_NONE;
    }

    /// Port of celt/celt_decoder.c:prefilter_and_fold: applies the pre-filter to the MDCT
    /// overlap of the concealed signal and simulates TDAC so it blends with the next frame.
    fn prefilter_and_fold(&mut self, n: usize) {
        let dbs = self.decode_buffer_size();
        let overlap = self.overlap as usize;
        let stride = dbs + overlap;
        let window = &self.mode.window[..overlap];
        let mut etmp_buf: [OpusVal32; MAX_OVERLAP] = [0.0; MAX_OVERLAP];
        let etmp = &mut etmp_buf[..overlap];
        for c in 0..self.channels as usize {
            let chan = &mut self.decode_mem[c * stride..(c + 1) * stride];
            // Apply the pre-filter to the MDCT overlap for the next frame because the
            // post-filter will be re-applied in the decoder after the MDCT overlap.
            comb_filter(
                etmp,
                chan,
                dbs - n,
                self.postfilter_period_old,
                self.postfilter_period,
                overlap as i32,
                -self.postfilter_gain_old,
                -self.postfilter_gain,
                self.postfilter_tapset_old,
                self.postfilter_tapset,
                &[],
                0,
            );
            // Simulate TDAC on the concealed audio so that it blends with the MDCT of the next
            // frame.
            for i in 0..overlap / 2 {
                chan[dbs - n + i] = mult16_32_q15(coef2val16(window[i]), etmp[overlap - 1 - i])
                    + mult16_32_q15(coef2val16(window[overlap - i - 1]), etmp[i]);
            }
        }
    }

    /// Runs the post-filter (`comb_filter` in place) on the `N` output samples of each of the
    /// `cc` channels, as the decoder and the noise PLC do. `next` is the `(period, gain, tapset)`
    /// the second (`LM != 0`) filter moves to; `None` keeps the current (updated) parameters,
    /// as in the noise PLC.
    fn run_postfilter(
        &mut self,
        cc: usize,
        n: usize,
        lm: i32,
        next: Option<(i32, OpusVal16, i32)>,
    ) {
        let dbs = self.decode_buffer_size();
        let overlap = self.overlap as usize;
        let stride = dbs + overlap;
        let mode = self.mode;
        let short = mode.short_mdct_size as usize;
        let window = &mode.window[..];
        for c in 0..cc {
            self.postfilter_period = imax(self.postfilter_period, COMBFILTER_MINPERIOD);
            self.postfilter_period_old = imax(self.postfilter_period_old, COMBFILTER_MINPERIOD);
            let chan = &mut self.decode_mem[c * stride..(c + 1) * stride];
            comb_filter_inplace(
                chan,
                dbs - n,
                self.postfilter_period_old,
                self.postfilter_period,
                short as i32,
                self.postfilter_gain_old,
                self.postfilter_gain,
                self.postfilter_tapset_old,
                self.postfilter_tapset,
                window,
                overlap as i32,
            );
            if lm != 0 {
                let (t1, g1, tapset1) = match next {
                    Some(v) => v,
                    None => (
                        self.postfilter_period,
                        self.postfilter_gain,
                        self.postfilter_tapset,
                    ),
                };
                comb_filter_inplace(
                    chan,
                    dbs - n + short,
                    self.postfilter_period,
                    t1,
                    (n - short) as i32,
                    self.postfilter_gain,
                    g1,
                    self.postfilter_tapset,
                    tapset1,
                    window,
                    overlap as i32,
                );
            }
        }
    }

    /// Port of celt/celt_decoder.c:celt_decode_lost: packet loss concealment (noise-based PLC
    /// or pitch-based PLC with LPC extrapolation) of one frame of `n` samples.
    fn celt_decode_lost(
        &mut self,
        n: usize,
        lm: i32,
        bufs: &mut FrameBufs<'_>,
        #[cfg(feature = "deep-plc")] mut lpcnet: Option<&mut LpcnetPlcState>,
    ) {
        let cn = self.channels as usize;
        let dbs = self.decode_buffer_size();
        let max_period = (self.qext_scale() * MAX_PERIOD) as usize;
        let mode = self.mode;
        let nb = mode.nb_ebands as usize;
        let overlap = mode.overlap as usize;
        let stride = dbs + overlap;
        let e_bands = &mode.e_bands[..];

        let loss_duration = self.loss_duration;
        let start = self.start;
        let mut curr_frame_type = FRAME_PLC_PERIODIC;
        if self.plc_duration >= 40 || start != 0 || self.skip_plc != 0 {
            curr_frame_type = FRAME_PLC_NOISE;
        }
        #[cfg(feature = "deep-plc")]
        if let Some(l) = lpcnet.as_deref()
            && start == 0
            && mode.fs != 96000
            && l.loaded
        {
            if self.complexity >= 5 && self.plc_duration < 80 && self.skip_plc == 0 {
                curr_frame_type = FRAME_PLC_NEURAL;
            }
            #[cfg(feature = "dred")]
            if l.fec_fill_pos > l.fec_read_pos {
                curr_frame_type = FRAME_DRED;
            }
        }

        if curr_frame_type == FRAME_PLC_NOISE {
            // Noise-based PLC/CNG
            let end = self.end;
            let eff_end = imax(start, imin(end, mode.eff_ebands));

            for c in 0..cn {
                let base = c * stride;
                self.decode_mem
                    .copy_within(base + n..base + dbs + overlap, base);
            }

            if self.prefilter_and_fold != 0 {
                self.prefilter_and_fold(n);
            }

            // Energy decay
            let decay: CeltGlog = if loss_duration == 0 {
                gconst(1.5)
            } else {
                gconst(0.5)
            };
            for c in 0..cn {
                for i in start as usize..end as usize {
                    let idx = c * nb + i;
                    self.old_ebands[idx] =
                        maxg(self.background_log_e[idx], self.old_ebands[idx] - decay);
                }
            }
            let mut seed: u32 = self.rng;
            let x = &mut *bufs.x;
            for c in 0..cn {
                for i in start as usize..eff_end as usize {
                    let boffs = n * c + ((e_bands[i] as usize) << lm);
                    let blen = ((e_bands[i + 1] - e_bands[i]) as usize) << lm;
                    for j in 0..blen {
                        seed = celt_lcg_rand(seed);
                        x[boffs + j] = shl32(((seed as i32) >> 20) as CeltNorm, NORM_SHIFT - 14);
                    }
                    renormalise_vector(&mut x[boffs..], blen as i32, Q31ONE);
                }
            }
            self.rng = seed;

            {
                let (d0, d1) = self.decode_mem.split_at_mut(stride);
                let mut out_syn: [&mut [CeltSig]; 2] = [&mut d0[dbs - n..], &mut []];
                if cn == 2 {
                    out_syn[1] = &mut d1[dbs - n..stride];
                }
                celt_synthesis(
                    mode,
                    bufs.x,
                    &mut out_syn[..cn],
                    &self.old_ebands,
                    start,
                    eff_end,
                    cn as i32,
                    cn as i32,
                    false,
                    lm,
                    self.downsample,
                    false,
                    #[cfg(feature = "qext")]
                    None,
                    #[cfg(feature = "qext")]
                    &[],
                    #[cfg(feature = "qext")]
                    0,
                    bufs.freq,
                );
            }

            // Run the postfilter with the last parameters.
            self.run_postfilter(cn, n, lm, None);
            self.postfilter_period_old = self.postfilter_period;
            self.postfilter_gain_old = self.postfilter_gain;
            self.postfilter_tapset_old = self.postfilter_tapset;

            self.prefilter_and_fold = 0;
            // Skip regular PLC until we get two consecutive packets.
            self.skip_plc = 1;
        } else {
            // Pitch-based PLC
            let mut fade: OpusVal16 = Q15ONE;
            let pitch_index: i32;

            let curr_neural = curr_frame_type == FRAME_PLC_NEURAL || curr_frame_type == FRAME_DRED;
            let last_neural =
                self.last_frame_type == FRAME_PLC_NEURAL || self.last_frame_type == FRAME_DRED;
            let search =
                self.last_frame_type != FRAME_PLC_PERIODIC && !(last_neural && curr_neural);
            if search {
                let qext_scale = self.qext_scale();
                let (d0, d1) = self.decode_mem.split_at(stride);
                let chans: [&[CeltSig]; 2] = [d0, if cn == 2 { d1 } else { &[] }];
                let mut lp_pitch_buf = [0.0; (DECODE_BUFFER_SIZE >> 1) as usize];
                pitch_index =
                    celt_plc_pitch_search(&chans[..cn], cn, qext_scale, &mut lp_pitch_buf);
                self.last_pitch_index = pitch_index;
            } else {
                pitch_index = self.last_pitch_index;
                fade = qconst16(0.8, 15);
            }
            #[cfg(feature = "deep-plc")]
            if curr_neural
                && !last_neural
                && let Some(l) = lpcnet.as_deref_mut()
            {
                update_plc_state(
                    l,
                    &self.decode_mem,
                    stride,
                    &mut self.plc_preemphasis_mem,
                    cn,
                );
            }

            // We want the excitation for 2 pitch periods in order to look for a decaying
            // signal, but we can't get more than MAX_PERIOD.
            let exc_length = imin(2 * pitch_index, max_period as i32) as usize;
            let pitch_index_u = pitch_index as usize;

            let window = &mode.window[..];
            let ord = CELT_LPC_ORDER;
            let mut exc_stack: [OpusVal16; MAX_MAX_PERIOD + CELT_LPC_ORDER] =
                [0.0; MAX_MAX_PERIOD + CELT_LPC_ORDER];
            let mut fir_tmp_stack: [OpusVal16; MAX_MAX_PERIOD] = [0.0; MAX_MAX_PERIOD];
            let exc_buf = &mut exc_stack[..max_period + ord];
            let fir_tmp = &mut fir_tmp_stack[..exc_length];
            for c in 0..cn {
                let mut s1: OpusVal32 = 0.0;
                let buf = &mut self.decode_mem[c * stride..(c + 1) * stride];
                let lpc_c = &mut self.lpc[c * ord..(c + 1) * ord];

                // exc[i] is exc_buf[ord + i].
                for i in 0..max_period + ord {
                    exc_buf[i] = sround16(buf[dbs - max_period - ord + i], SIG_SHIFT);
                }

                if search {
                    let mut ac: [OpusVal32; CELT_LPC_ORDER + 1] = [0.0; CELT_LPC_ORDER + 1];
                    // Compute LPC coefficients for the last MAX_PERIOD samples before the
                    // first loss so we can work in the excitation-filter domain.
                    _celt_autocorr(&exc_buf[ord..], &mut ac, window, overlap, ord, max_period);
                    // Add a noise floor of -40 dB.
                    // FIXED_POINT: not ported (float build).
                    ac[0] *= 1.0001f32;
                    // Use lag windowing to stabilize the Levinson-Durbin recursion.
                    for i in 1..=ord {
                        // ac[i] *= exp(-.5*(2*M_PI*.002*i)*(2*M_PI*.002*i));
                        // FIXED_POINT: not ported (float build).
                        ac[i] -= ac[i] * (0.008f32 * 0.008f32) * i as f32 * i as f32;
                    }
                    _celt_lpc(lpc_c, &ac, ord);
                    // FIXED_POINT: bandwidth expansion loop not ported (float build).
                }
                // Initialize the LPC history with the samples just before the start of the
                // region for which we're computing the excitation.
                {
                    // Compute the excitation for exc_length samples before the loss. We need
                    // the copy because celt_fir() cannot filter in-place.
                    celt_fir(
                        &exc_buf[max_period - exc_length..],
                        lpc_c,
                        fir_tmp,
                        exc_length,
                        ord,
                    );
                    exc_buf[ord + max_period - exc_length..ord + max_period]
                        .copy_from_slice(fir_tmp);
                }

                // Check if the waveform is decaying, and if so how fast. We do this to avoid
                // adding energy when concealing in a segment with decaying energy.
                let decay: OpusVal16;
                {
                    let mut e1: OpusVal32 = 1.0;
                    let mut e2: OpusVal32 = 1.0;
                    // FIXED_POINT: shift computation not ported (float build).
                    let decay_length = exc_length >> 1;
                    let exc = &exc_buf[ord..];
                    for i in 0..decay_length {
                        let e: OpusVal16 = exc[max_period - decay_length + i];
                        e1 += shr32(mult16_16(e, e), 0);
                        let e: OpusVal16 = exc[max_period - 2 * decay_length + i];
                        e2 += shr32(mult16_16(e, e), 0);
                    }
                    e1 = min32(e1, e2);
                    decay = celt_sqrt(frac_div32(shr32(e1, 1), e2));
                }

                // Move the decoder memory one frame to the left to give us room to add the
                // data for the new frame. We ignore the overlap that extends past the end of
                // the buffer, because we aren't going to use it.
                buf.copy_within(n..dbs, 0);

                // Extrapolate from the end of the excitation with a period of "pitch_index",
                // scaling down each period by an additional factor of "decay".
                let extrapolation_offset = max_period - pitch_index_u;
                // We need to extrapolate enough samples to cover a complete MDCT window
                // (including overlap/2 samples on both sides).
                let extrapolation_len = n + overlap;
                // We also apply fading if this is not the first loss.
                let mut attenuation: OpusVal16 = mult16_16_q15(fade, decay);
                let mut j = 0usize;
                for i in 0..extrapolation_len {
                    if j >= pitch_index_u {
                        j -= pitch_index_u;
                        attenuation = mult16_16_q15(attenuation, decay);
                    }
                    buf[dbs - n + i] = shl32(
                        extend32(mult16_16_q15(
                            attenuation,
                            exc_buf[ord + extrapolation_offset + j],
                        )),
                        SIG_SHIFT,
                    );
                    // Compute the energy of the previously decoded signal whose excitation
                    // we're copying.
                    let tmp: OpusVal16 = sround16(
                        buf[dbs - max_period - n + extrapolation_offset + j],
                        SIG_SHIFT,
                    );
                    s1 += shr32(mult16_16(tmp, tmp), 11);
                    j += 1;
                }
                {
                    let mut lpc_mem: [OpusVal16; CELT_LPC_ORDER] = [0.0; CELT_LPC_ORDER];
                    // Copy the last decoded samples (prior to the overlap region) to synthesis
                    // filter memory so we can have a continuous signal.
                    for i in 0..ord {
                        lpc_mem[i] = sround16(buf[dbs - n - 1 - i], SIG_SHIFT);
                    }
                    // Apply the synthesis filter to convert the excitation back into the
                    // signal domain.
                    celt_iir_inplace(
                        &mut buf[dbs - n..],
                        lpc_c,
                        extrapolation_len,
                        ord,
                        &mut lpc_mem,
                    );
                    // FIXED_POINT: saturation not ported (float build).
                }

                // Check if the synthesis energy is higher than expected, which can happen with
                // the signal changes during our window. If so, attenuate.
                {
                    let mut s2: OpusVal32 = 0.0;
                    let out = &mut buf[dbs - n..dbs - n + extrapolation_len];
                    for &v in out.iter() {
                        let tmp: OpusVal16 = sround16(v, SIG_SHIFT);
                        s2 += shr32(mult16_16(tmp, tmp), 11);
                    }
                    // This checks for an "explosion" in the synthesis.
                    // FIXED_POINT: not ported (float build).
                    // The float test is written this way to catch NaNs in the output of the IIR
                    // filter at the same time.
                    #[expect(
                        clippy::neg_cmp_op_on_partial_ord,
                        reason = "C writes the test this way to also catch NaNs"
                    )]
                    let explosion = !(s1 > 0.2f32 * s2);
                    if explosion {
                        out.fill(0.0);
                    } else if s1 < s2 {
                        let ratio: OpusVal16 = celt_sqrt(frac_div32(shr32(s1, 1) + 1.0, s2 + 1.0));
                        for i in 0..overlap {
                            let tmp_g: OpusVal16 =
                                Q15ONE - mult16_16_q15(coef2val16(window[i]), Q15ONE - ratio);
                            out[i] = mult16_32_q15(tmp_g, out[i]);
                        }
                        for v in out[overlap..].iter_mut() {
                            *v = mult16_32_q15(ratio, *v);
                        }
                    }
                }
            }

            #[cfg(feature = "deep-plc")]
            if curr_neural && let Some(l) = lpcnet {
                self.neural_plc_synthesis(l, n, dbs, stride, last_neural);
            }
            self.prefilter_and_fold = 1;
        }

        // Saturate to something large to avoid wrap-around.
        self.loss_duration = imin(10000, loss_duration + (1 << lm));
        self.plc_duration = imin(10000, self.plc_duration + (1 << lm));
        #[cfg(feature = "dred")]
        if curr_frame_type == FRAME_DRED {
            self.plc_duration = 0;
            self.skip_plc = 0;
        }
        self.last_frame_type = curr_frame_type;
    }

    /// The `ENABLE_DEEP_PLC` block at the end of the pitch-based branch of
    /// celt/celt_decoder.c:celt_decode_lost: replaces the extrapolated frame (plus overlap) by
    /// the LPCNet concealment resampled from 16 to 48 kHz, re-applies the pre-emphasis, copies
    /// channel 0 to channel 1 ("for now, we just do mono PLC") and cross-fades with the
    /// pitch-based extrapolation when entering neural PLC.
    #[cfg(feature = "deep-plc")]
    fn neural_plc_synthesis(
        &mut self,
        lpcnet: &mut LpcnetPlcState,
        n: usize,
        dbs: usize,
        stride: usize,
        last_neural: bool,
    ) {
        let cn = self.channels as usize;
        let overlap = self.mode.overlap as usize;
        let window = &self.mode.window[..];
        let mut buf_copy_stack: [CeltSig; 2 * MAX_OVERLAP] = [0.0; 2 * MAX_OVERLAP];
        let buf_copy = &mut buf_copy_stack[..];
        for c in 0..cn {
            buf_copy[c * overlap..(c + 1) * overlap]
                .copy_from_slice(&self.decode_mem[c * stride + dbs - n..][..overlap]);
        }

        // Need enough samples from the PLC to cover the frame size, resampling delay, and the
        // overlap at the end.
        let samples_needed16k = (n + SINC_ORDER + overlap) / 3;
        if !last_neural {
            self.plc_fill = 0;
        }
        while (self.plc_fill as usize) < samples_needed16k {
            lpcnet_plc_conceal(lpcnet, &mut self.plc_pcm[self.plc_fill as usize..]);
            self.plc_fill += LPCNET_FRAME_SIZE as i32;
        }
        // Resample to 48 kHz.
        let plc_pcm = &self.plc_pcm;
        let buf = &mut self.decode_mem[..stride];
        for i in 0..(n + overlap) / 3 {
            let mut sum: f32 = 0.0;
            for j in 0..17 {
                sum += (3 * i32::from(plc_pcm[i + j])) as f32 * SINC_FILTER[3 * j];
            }
            buf[dbs - n + 3 * i] = sum;
            let mut sum: f32 = 0.0;
            for j in 0..16 {
                sum += (3 * i32::from(plc_pcm[i + j + 1])) as f32 * SINC_FILTER[3 * j + 2];
            }
            buf[dbs - n + 3 * i + 1] = sum;
            let mut sum: f32 = 0.0;
            for j in 0..16 {
                sum += (3 * i32::from(plc_pcm[i + j + 1])) as f32 * SINC_FILTER[3 * j + 1];
            }
            buf[dbs - n + 3 * i + 2] = sum;
        }
        let n3 = n / 3;
        self.plc_pcm.copy_within(n3..self.plc_fill as usize, 0);
        self.plc_fill -= n3 as i32;
        for v in &mut buf[dbs - n..dbs] {
            let tmp = *v;
            *v -= LPCNET_PREEMPHASIS * self.plc_preemphasis_mem;
            self.plc_preemphasis_mem = tmp;
        }
        let mut overlap_mem = self.plc_preemphasis_mem;
        for v in &mut buf[dbs..dbs + overlap] {
            let tmp = *v;
            *v -= LPCNET_PREEMPHASIS * overlap_mem;
            overlap_mem = tmp;
        }
        // For now, we just do mono PLC.
        if cn == 2 {
            self.decode_mem.copy_within(0..dbs + overlap, stride);
        }
        // Cross-fade with 48-kHz non-neural PLC for the first 2.5 ms to avoid a discontinuity.
        if !last_neural {
            for c in 0..cn {
                let d = &mut self.decode_mem[c * stride + dbs - n..][..overlap];
                let bc = &buf_copy[c * overlap..(c + 1) * overlap];
                for i in 0..overlap {
                    d[i] = (1.0 - window[i]) * bc[i] + window[i] * d[i];
                }
            }
        }
    }

    /// De-emphasis of the `n` output samples of every channel into `pcm`.
    fn deemphasis_out(
        &mut self,
        pcm: &mut [OpusRes],
        n: usize,
        accum: bool,
        scratch: &mut [CeltSig],
    ) {
        let dbs = self.decode_buffer_size();
        let stride = dbs + self.overlap as usize;
        let cc = self.channels as usize;
        let (d0, d1) = self.decode_mem.split_at(stride);
        let out_syn: [&[CeltSig]; 2] = [&d0[dbs - n..], if cc == 2 { &d1[dbs - n..] } else { &[] }];
        deemphasis(
            &out_syn[..cc],
            pcm,
            n,
            cc,
            self.downsample,
            &self.mode.preemph,
            &mut self.preemph_mem_d,
            accum,
            scratch,
        );
    }

    /// Port of celt/celt.h:celt_decode_with_ec: decodes one CELT frame.
    ///
    /// * `data`/`len`: the packet (`None` or `len <= 1` = packet lost → concealment). `data`
    ///   must hold at least `len` bytes.
    /// * `pcm`: output (`frame_size*channels` interleaved samples at the API rate), accumulated
    ///   into when `accum`.
    /// * `frame_size`: samples per channel at the API rate.
    /// * `dec`: a range decoder shared with SILK (hybrid) or `None` to decode `data` alone.
    ///
    /// Returns the number of decoded samples per channel.
    pub fn celt_decode_with_ec(
        &mut self,
        data: Option<&[u8]>,
        len: i32,
        pcm: &mut [OpusRes],
        frame_size: i32,
        dec: Option<&mut EcDec<'_>>,
        accum: bool,
    ) -> Result<i32> {
        self.celt_decode_with_ec_dred(
            data,
            len,
            pcm,
            frame_size,
            dec,
            accum,
            // C passes lpcnet = NULL here (celt_decoder.c:1619-1621).
            #[cfg(feature = "qext")]
            None,
        )
    }

    /// Port of celt/celt_decoder.c:celt_decode_with_ec_dred without the deep-PLC state (C
    /// `lpcnet == NULL`): [`Self::celt_decode_with_ec`] plus (QEXT) the extension payload
    /// `qext_payload`. See [`Self::celt_decode_with_ec_lpcnet`] for the neural PLC.
    #[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
    pub fn celt_decode_with_ec_dred(
        &mut self,
        data: Option<&[u8]>,
        len: i32,
        pcm: &mut [OpusRes],
        frame_size: i32,
        dec: Option<&mut EcDec<'_>>,
        accum: bool,
        #[cfg(feature = "qext")] qext_payload: Option<&[u8]>,
    ) -> Result<i32> {
        self.decode_impl(
            data,
            len,
            pcm,
            frame_size,
            dec,
            accum,
            #[cfg(feature = "deep-plc")]
            None,
            #[cfg(feature = "qext")]
            qext_payload,
        )
    }

    /// Port of celt/celt_decoder.c:celt_decode_with_ec_dred with its `ENABLE_DEEP_PLC`
    /// `LPCNetPLCState *lpcnet` argument: lost frames use the neural PLC (`FRAME_PLC_NEURAL`, at
    /// complexity >= 5) or the DRED features queued in `lpcnet` (`FRAME_DRED`) when `lpcnet` has
    /// a loaded model; a received frame clears `lpcnet.blend`.
    #[cfg(feature = "deep-plc")]
    #[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
    pub fn celt_decode_with_ec_lpcnet(
        &mut self,
        data: Option<&[u8]>,
        len: i32,
        pcm: &mut [OpusRes],
        frame_size: i32,
        dec: Option<&mut EcDec<'_>>,
        accum: bool,
        lpcnet: Option<&mut LpcnetPlcState>,
        #[cfg(feature = "qext")] qext_payload: Option<&[u8]>,
    ) -> Result<i32> {
        self.decode_impl(
            data,
            len,
            pcm,
            frame_size,
            dec,
            accum,
            lpcnet,
            #[cfg(feature = "qext")]
            qext_payload,
        )
    }

    /// The body of `celt_decode_with_ec_dred`.
    #[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
    fn decode_impl(
        &mut self,
        data: Option<&[u8]>,
        len: i32,
        pcm: &mut [OpusRes],
        frame_size: i32,
        dec: Option<&mut EcDec<'_>>,
        accum: bool,
        #[cfg(feature = "deep-plc")] lpcnet: Option<&mut LpcnetPlcState>,
        #[cfg(feature = "qext")] qext_payload: Option<&[u8]>,
    ) -> Result<i32> {
        validate_celt_decoder(self);
        let mode = self.mode;

        // Rust-only guard: C would read past the buffer.
        let data: Option<&[u8]> = match data {
            Some(d) if len >= 0 && len as usize > d.len() => return Err(Error::BadArg),
            Some(d) if len >= 0 => Some(&d[..len as usize]),
            other => other,
        };
        let hdr = Header {
            data,
            len,
            lm: 0,
            c: self.stream_channels,
            end: self.end,
            frame_size: frame_size * self.downsample,
            #[cfg(feature = "qext")]
            ext: qext_payload,
        };
        #[cfg(feature = "custom-modes")]
        let hdr = match hdr.data {
            Some(d) if self.signalling != 0 => self.parse_custom_header(d, hdr)?,
            _ => Self::find_lm(mode, hdr)?,
        };
        #[cfg(not(feature = "custom-modes"))]
        let hdr = Self::find_lm(mode, hdr)?;
        let Header {
            data,
            len,
            lm,
            c: c_stream,
            end,
            frame_size,
            #[cfg(feature = "qext")]
            ext,
        } = hdr;
        let m = 1 << lm;

        if !(0..=1275).contains(&len) {
            return Err(Error::BadArg);
        }

        let n = (m * mode.short_mdct_size) as usize;
        // Rust-only guard (C: pcm == NULL, or writing past the buffer).
        if pcm.len() < (n / self.downsample as usize) * self.channels as usize {
            return Err(Error::BadArg);
        }

        let mut eff_end = end;
        if eff_end > mode.eff_ebands {
            eff_end = mode.eff_ebands;
        }

        let mut x_heap = core::mem::take(&mut self.scratch.x_heap);
        let mut freq_heap = core::mem::take(&mut self.scratch.freq_heap);
        let ret = with_frame_bufs(n, &mut x_heap, &mut freq_heap, |bufs| {
            self.decode_with_bufs(
                data,
                len,
                pcm,
                dec,
                accum,
                n,
                lm,
                c_stream,
                end,
                eff_end,
                frame_size,
                bufs,
                #[cfg(feature = "deep-plc")]
                lpcnet,
                #[cfg(feature = "qext")]
                ext,
            )
        });
        self.scratch.x_heap = x_heap;
        self.scratch.freq_heap = freq_heap;
        ret
    }

    /// The part of `celt_decode_with_ec_dred` after the header parsing, with the frame-sized
    /// VLAs in `bufs`.
    #[expect(clippy::too_many_arguments, reason = "the C function's locals")]
    fn decode_with_bufs(
        &mut self,
        data: Option<&[u8]>,
        len: i32,
        pcm: &mut [OpusRes],
        dec: Option<&mut EcDec<'_>>,
        accum: bool,
        n: usize,
        lm: i32,
        c_stream: i32,
        end: i32,
        eff_end: i32,
        frame_size: i32,
        bufs: &mut FrameBufs<'_>,
        #[cfg(feature = "deep-plc")] lpcnet: Option<&mut LpcnetPlcState>,
        #[cfg(feature = "qext")] ext: Option<&[u8]>,
    ) -> Result<i32> {
        let data = match data {
            Some(d) if len > 1 => &d[..len as usize],
            _ => {
                self.celt_decode_lost(
                    n,
                    lm,
                    bufs,
                    #[cfg(feature = "deep-plc")]
                    lpcnet,
                );
                self.deemphasis_out(pcm, n, accum, bufs.freq);
                return Ok(frame_size / self.downsample);
            }
        };
        // FIXME (C): This is a bit of a hack just to make sure opus_decode_native() knows we're
        // no longer in PLC.
        #[cfg(feature = "deep-plc")]
        if let Some(l) = lpcnet {
            l.blend = 0;
        }

        // Check if there are at least two packets received consecutively before turning on
        // the pitch-based PLC
        if self.loss_duration == 0 {
            self.skip_plc = 0;
        }

        #[cfg(feature = "qext")]
        let ext_payload: &[u8] = match ext {
            Some(p) => p,
            None => &[],
        };
        let fr = FrameParams {
            len,
            lm,
            n,
            c: c_stream,
            end,
            eff_end,
            frame_size,
        };
        match dec {
            Some(d) => self.decode_frame(
                &fr,
                pcm,
                d,
                accum,
                bufs,
                #[cfg(feature = "qext")]
                ext_payload,
            ),
            None => {
                let mut local = EcDec::new(data);
                self.decode_frame(
                    &fr,
                    pcm,
                    &mut local,
                    accum,
                    bufs,
                    #[cfg(feature = "qext")]
                    ext_payload,
                )
            }
        }
    }

    /// The frame size search of `celt_decode_with_ec_dred` (no in-band signalling).
    const fn find_lm<'a>(mode: &CeltMode, mut hdr: Header<'a>) -> Result<Header<'a>> {
        let mut lm = 0;
        while lm <= mode.max_lm {
            if mode.short_mdct_size << lm == hdr.frame_size {
                break;
            }
            lm += 1;
        }
        if lm > mode.max_lm {
            return Err(Error::BadArg);
        }
        hdr.lm = lm;
        Ok(hdr)
    }

    /// The `st->signalling` header parsing of `celt_decode_with_ec_dred` (custom modes): the
    /// TOC byte gives the end band, frame size and channel count, and a code-3 header may
    /// carry padding (with QEXT, a QEXT extension payload).
    #[cfg(feature = "custom-modes")]
    fn parse_custom_header<'a>(&mut self, d: &'a [u8], mut hdr: Header<'a>) -> Result<Header<'a>> {
        let mode = self.mode;
        let mut len = hdr.len;
        // Rust-only guard: C reads data[0] even for len <= 0.
        let Some(&first) = d.first() else {
            return Err(Error::BadArg);
        };
        let mut pos = 0usize;
        let mut data0 = i32::from(first);
        // Convert "standard mode" to Opus header
        #[cfg(not(feature = "qext"))]
        let convert = mode.fs == 48000 && mode.short_mdct_size == 120;
        #[cfg(feature = "qext")]
        let convert = true;
        if convert {
            data0 = from_opus(data0 as u8);
            if data0 < 0 {
                return Err(Error::InvalidPacket);
            }
        }
        hdr.end = imax(1, mode.eff_ebands - 2 * (data0 >> 5));
        self.end = hdr.end;
        let lm = (data0 >> 3) & 0x3;
        hdr.c = 1 + ((data0 >> 2) & 0x1);
        if (d[pos] & 0x03) == 0x03 {
            pos += 1;
            len -= 1;
            if len <= 0 {
                return Err(Error::InvalidPacket);
            }
            if d[pos] & 0x40 != 0 {
                let mut padding: i32 = 0;
                pos += 1;
                len -= 1;
                loop {
                    if len <= 0 {
                        return Err(Error::InvalidPacket);
                    }
                    let p = i32::from(d[pos]);
                    pos += 1;
                    len -= 1;
                    let tmp = if p == 255 { 254 } else { p };
                    len -= tmp;
                    padding += tmp;
                    if p != 255 {
                        break;
                    }
                }
                padding -= 1;
                if len <= 0 || padding < 0 {
                    return Err(Error::InvalidPacket);
                }
                #[cfg(feature = "qext")]
                {
                    let at = pos + len as usize;
                    let mut qext_bytes = padding as usize;
                    if i32::from(d[at]) != QEXT_EXTENSION_ID << 1 {
                        qext_bytes = 0;
                    }
                    hdr.ext = Some(&d[at + 1..at + 1 + qext_bytes]);
                }
            }
        } else {
            pos += 1;
            len -= 1;
        }
        hdr.data = Some(&d[pos..]);
        hdr.len = len;
        if lm > mode.max_lm {
            return Err(Error::InvalidPacket);
        }
        if hdr.frame_size < mode.short_mdct_size << lm {
            return Err(Error::BufferTooSmall);
        }
        hdr.frame_size = mode.short_mdct_size << lm;
        hdr.lm = lm;
        Ok(hdr)
    }

    /// The part of `celt_decode_with_ec_dred` after the range decoder is set up.
    #[allow(clippy::too_many_lines, reason = "mirrors the C function")]
    fn decode_frame(
        &mut self,
        fr: &FrameParams,
        pcm: &mut [OpusRes],
        dec: &mut EcDec<'_>,
        accum: bool,
        bufs: &mut FrameBufs<'_>,
        #[cfg(feature = "qext")] ext_payload: &[u8],
    ) -> Result<i32> {
        let FrameParams {
            len,
            lm,
            n,
            c,
            end,
            eff_end,
            frame_size,
        } = *fr;
        let mode = self.mode;
        let nb = mode.nb_ebands as usize;
        let e_bands = &mode.e_bands[..];
        let start = self.start;
        let cc = self.channels as usize;
        let dbs = self.decode_buffer_size();
        let overlap = self.overlap as usize;
        let stride = dbs + overlap;
        let m = 1 << lm;

        #[cfg(feature = "qext")]
        let mut ext_dec = EcDec::new(ext_payload);
        #[cfg(feature = "qext")]
        let qext_bytes = ext_payload.len() as i32;
        #[cfg(not(feature = "qext"))]
        let qext_bytes = 0i32;

        if c == 1 {
            for i in 0..nb {
                self.old_ebands[i] = maxg(self.old_ebands[i], self.old_ebands[nb + i]);
            }
        }

        let mut total_bits: i32 = len * 8;
        let mut tell: i32 = dec.tell();

        let silence = if tell >= total_bits {
            true
        } else if tell == 1 {
            dec.dec_bit_logp(15)
        } else {
            false
        };
        if silence {
            // Pretend we've read all the remaining bits
            tell = len * 8;
            dec.nbits_total += tell - dec.tell();
        }

        let mut postfilter_gain: OpusVal16 = 0.0;
        let mut postfilter_pitch: i32 = 0;
        let mut postfilter_tapset: i32 = 0;
        if start == 0 && tell + 16 <= total_bits {
            if dec.dec_bit_logp(1) {
                let octave = dec.dec_uint(6) as i32;
                postfilter_pitch = (16 << octave) + dec.dec_bits((4 + octave) as u32) as i32 - 1;
                let qg = dec.dec_bits(3) as i32;
                if dec.tell() + 2 <= total_bits {
                    postfilter_tapset = dec.dec_icdf(&TAPSET_ICDF, 2) as i32;
                }
                postfilter_gain = qconst16(0.09375, 15) * (qg + 1) as f32;
            }
            tell = dec.tell();
        }

        let is_transient = if lm > 0 && tell + 3 <= total_bits {
            let t = dec.dec_bit_logp(3);
            tell = dec.tell();
            t
        } else {
            false
        };

        let short_blocks = is_transient;

        // Decode the global flags (first symbols in the stream)
        let intra_ener = if tell + 3 <= total_bits {
            dec.dec_bit_logp(3)
        } else {
            false
        };
        // If recovering from packet loss, make sure we make the energy prediction safe to
        // reduce the risk of getting loud artifacts.
        if !intra_ener && self.loss_duration != 0 {
            for ch in 0..2usize {
                let mut safety: CeltGlog = 0.0;
                let missing = imin(10, self.loss_duration >> lm);
                if lm == 0 {
                    safety = gconst(1.5);
                } else if lm == 1 {
                    safety = gconst(0.5);
                }
                for i in start as usize..end as usize {
                    let idx = ch * nb + i;
                    if self.old_ebands[idx] < maxg(self.old_log_e[idx], self.old_log_e2[idx]) {
                        // If energy is going down already, continue the trend.
                        let mut e0: OpusVal32 = self.old_ebands[idx];
                        let e1: OpusVal32 = self.old_log_e[idx];
                        let e2: OpusVal32 = self.old_log_e2[idx];
                        let mut slope: OpusVal32 = max32(e1 - e0, half32(e2 - e0));
                        slope = ming(slope, gconst(2.0));
                        e0 -= max32(0.0, (1 + missing) as f32 * slope);
                        self.old_ebands[idx] = max32(-gconst(20.0), e0);
                    } else {
                        // Otherwise take the min of the last frames.
                        self.old_ebands[idx] = ming(
                            ming(self.old_ebands[idx], self.old_log_e[idx]),
                            self.old_log_e2[idx],
                        );
                    }
                    // Shorter frames have more natural fluctuations -- play it safe.
                    self.old_ebands[idx] -= safety;
                }
            }
        }
        // Get band energies
        unquant_coarse_energy(
            mode,
            start,
            end,
            &mut self.old_ebands,
            intra_ener,
            dec,
            c,
            lm,
        );

        let sc = &mut self.scratch;
        tf_decode(start, end, is_transient, &mut sc.tf_res, lm, dec);

        tell = dec.tell();
        let mut spread_decision = SPREAD_NORMAL;
        if tell + 4 <= total_bits {
            spread_decision = dec.dec_icdf(&SPREAD_ICDF, 5) as i32;
        }

        init_caps(mode, &mut sc.cap, lm, c);

        let mut dynalloc_logp: i32 = 6;
        total_bits <<= BITRES;
        tell = dec.tell_frac() as i32;
        for i in start as usize..end as usize {
            let width = (c * i32::from(e_bands[i + 1] - e_bands[i])) << lm;
            // quanta is 6 bits, but no more than 1 bit/sample and no less than 1/8 bit/sample
            let quanta = imin(width << BITRES, imax(6 << BITRES, width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost: i32 = 0;
            while tell + (dynalloc_loop_logp << BITRES) < total_bits && boost < sc.cap[i] {
                let flag = dec.dec_bit_logp(dynalloc_loop_logp as u32);
                tell = dec.tell_frac() as i32;
                if !flag {
                    break;
                }
                boost += quanta;
                total_bits -= quanta;
                dynalloc_loop_logp = 1;
            }
            sc.offsets[i] = boost;
            // Making dynalloc more likely
            if boost > 0 {
                dynalloc_logp = imax(2, dynalloc_logp - 1);
            }
        }

        let alloc_trim = if tell + (6 << BITRES) <= total_bits {
            dec.dec_icdf(&TRIM_ICDF, 7) as i32
        } else {
            5
        };

        let mut bits: i32 = ((len * 8) << BITRES) - dec.tell_frac() as i32 - 1;
        let anti_collapse_rsv = if is_transient && lm >= 2 && bits >= ((lm + 2) << BITRES) {
            1 << BITRES
        } else {
            0
        };
        bits -= anti_collapse_rsv;

        let mut intensity: i32 = 0;
        let mut dual_stereo: i32 = 0;
        let mut balance: i32 = 0;
        let coded_bands = clt_compute_allocation(
            mode,
            start,
            end,
            &sc.offsets,
            &sc.cap,
            alloc_trim,
            &mut intensity,
            &mut dual_stereo,
            bits,
            &mut balance,
            &mut sc.pulses,
            &mut sc.fine_quant,
            &mut sc.fine_priority,
            c,
            lm,
            &mut EcCoder::Dec(dec),
            0,
            0,
        );

        unquant_fine_energy(
            mode,
            start,
            end,
            &mut self.old_ebands,
            None,
            &sc.fine_quant,
            dec,
            c,
        );

        #[cfg(feature = "qext")]
        let mut qext_mode_struct: Option<CeltMode> = None;
        #[cfg(feature = "qext")]
        let mut qext_end: i32 = 0;
        #[cfg(feature = "qext")]
        let mut qext_intensity: i32 = 0;
        #[cfg(feature = "qext")]
        let mut qext_dual_stereo: i32 = 0;
        #[cfg(feature = "qext")]
        {
            if qext_bytes != 0
                && end == mode.nb_ebands
                && ((mode.fs == 48000
                    && (mode.short_mdct_size == 120 || mode.short_mdct_size == 90))
                    || (mode.fs == 96000
                        && (mode.short_mdct_size == 240 || mode.short_mdct_size == 180)))
            {
                let qm = compute_qext_mode(mode);
                qext_end = if ext_dec.dec_bit_logp(1) {
                    NB_QEXT_BANDS
                } else {
                    2
                };
                if c == 2 {
                    decode_qext_stereo_params(
                        &mut ext_dec,
                        qext_end,
                        &mut qext_intensity,
                        &mut qext_dual_stereo,
                    );
                }
                let qext_intra_ener = if ext_dec.tell() + 3 <= qext_bytes * 8 {
                    ext_dec.dec_bit_logp(3)
                } else {
                    false
                };
                unquant_coarse_energy(
                    &qm,
                    0,
                    qext_end,
                    &mut self.qext_old_band_e,
                    qext_intra_ener,
                    &mut ext_dec,
                    c,
                    lm,
                );
                qext_mode_struct = Some(qm);
            }
            let qext_bits: i32 = ((qext_bytes * 8) << BITRES) - dec.tell_frac() as i32 - 1;
            clt_compute_extra_allocation(
                mode,
                qext_mode_struct.as_ref(),
                start,
                end,
                qext_end,
                &[],
                &[],
                qext_bits,
                &mut sc.extra_pulses,
                &mut sc.extra_quant,
                c,
                lm,
                &mut EcCoder::Dec(&mut ext_dec),
                0.0,
                0.0,
            );
            if qext_bytes > 0 {
                unquant_fine_energy(
                    mode,
                    start,
                    end,
                    &mut self.old_ebands,
                    Some(&sc.fine_quant),
                    &sc.extra_quant,
                    &mut ext_dec,
                    c,
                );
            }
        }

        for ch in 0..cc {
            let base = ch * stride;
            self.decode_mem
                .copy_within(base + n..base + dbs + overlap, base);
        }

        // Decode fixed codebook
        {
            let (x0, x1) = bufs.x.split_at_mut(n);
            let y = if c == 2 { Some(&mut x1[..n]) } else { None };
            #[cfg(feature = "qext")]
            let mut ext_ec = EcCoder::Dec(&mut ext_dec);
            quant_all_bands(
                mode,
                start,
                end,
                x0,
                y,
                &mut sc.collapse_masks,
                &[],
                &sc.pulses,
                short_blocks,
                spread_decision,
                dual_stereo != 0,
                intensity,
                &sc.tf_res,
                len * (8 << BITRES) - anti_collapse_rsv,
                balance,
                &mut EcCoder::Dec(dec),
                lm,
                coded_bands,
                &mut self.rng,
                0,
                self.disable_inv != 0,
                #[cfg(feature = "qext")]
                &mut ext_ec,
                #[cfg(feature = "qext")]
                &sc.extra_pulses,
                #[cfg(feature = "qext")]
                (qext_bytes * (8 << BITRES)),
                #[cfg(feature = "qext")]
                Some(&sc.cap),
                &mut sc.bands,
            );
        }

        #[cfg(feature = "qext")]
        if let Some(qm) = qext_mode_struct.as_ref() {
            let mut dummy_dec = EcDec::new(&[]);
            sc.zeros[..end as usize].fill(0);
            let mut ext_balance: i32 = qext_bytes * (8 << BITRES) - ext_dec.tell_frac() as i32;
            for i in 0..qext_end as usize {
                // C quirk ported faithfully: extra_quant[nbEBands+1] (not +i).
                ext_balance -= sc.extra_pulses[nb + i] + c * (sc.extra_quant[nb + 1] << BITRES);
            }
            unquant_fine_energy(
                qm,
                0,
                qext_end,
                &mut self.qext_old_band_e,
                None,
                &sc.extra_quant[nb..],
                &mut ext_dec,
                c,
            );
            let (x0, x1) = bufs.x.split_at_mut(n);
            let y = if c == 2 { Some(&mut x1[..n]) } else { None };
            quant_all_bands(
                qm,
                0,
                qext_end,
                x0,
                y,
                &mut sc.qext_collapse_masks,
                &[],
                &sc.extra_pulses[nb..],
                short_blocks,
                spread_decision,
                qext_dual_stereo != 0,
                qext_intensity,
                &sc.zeros,
                qext_bytes * (8 << BITRES),
                ext_balance,
                &mut EcCoder::Dec(&mut ext_dec),
                lm,
                qext_end,
                &mut self.rng,
                0,
                self.disable_inv != 0,
                &mut EcCoder::Dec(&mut dummy_dec),
                &sc.zeros,
                0,
                None,
                &mut sc.bands,
            );
        }

        let mut anti_collapse_on = false;
        if anti_collapse_rsv > 0 {
            anti_collapse_on = dec.dec_bits(1) != 0;
        }
        let bits_left = len * 8 - dec.tell();
        unquant_energy_finalise(
            mode,
            start,
            end,
            if qext_bytes > 0 {
                None
            } else {
                Some(&mut self.old_ebands[..])
            },
            &sc.fine_quant,
            &sc.fine_priority,
            bits_left,
            dec,
            c,
        );
        if anti_collapse_on {
            anti_collapse(
                mode,
                bufs.x,
                &sc.collapse_masks,
                lm,
                c,
                n as i32,
                start,
                end,
                &self.old_ebands,
                &self.old_log_e,
                &self.old_log_e2,
                &sc.pulses,
                self.rng,
                false,
            );
        }

        if silence {
            for v in self.old_ebands[..c as usize * nb].iter_mut() {
                *v = -gconst(28.0);
            }
        }
        if self.prefilter_and_fold != 0 {
            self.prefilter_and_fold(n);
        }
        {
            let (d0, d1) = self.decode_mem.split_at_mut(stride);
            let mut out_syn: [&mut [CeltSig]; 2] = [&mut d0[dbs - n..], &mut []];
            if cc == 2 {
                out_syn[1] = &mut d1[dbs - n..stride];
            }
            celt_synthesis(
                mode,
                bufs.x,
                &mut out_syn[..cc],
                &self.old_ebands,
                start,
                eff_end,
                c,
                cc as i32,
                is_transient,
                lm,
                self.downsample,
                silence,
                #[cfg(feature = "qext")]
                qext_mode_struct.as_ref(),
                #[cfg(feature = "qext")]
                &self.qext_old_band_e,
                #[cfg(feature = "qext")]
                qext_end,
                bufs.freq,
            );
        }

        self.run_postfilter(
            cc,
            n,
            lm,
            Some((postfilter_pitch, postfilter_gain, postfilter_tapset)),
        );
        self.postfilter_period_old = self.postfilter_period;
        self.postfilter_gain_old = self.postfilter_gain;
        self.postfilter_tapset_old = self.postfilter_tapset;
        self.postfilter_period = postfilter_pitch;
        self.postfilter_gain = postfilter_gain;
        self.postfilter_tapset = postfilter_tapset;
        if lm != 0 {
            self.postfilter_period_old = self.postfilter_period;
            self.postfilter_gain_old = self.postfilter_gain;
            self.postfilter_tapset_old = self.postfilter_tapset;
        }

        if c == 1 {
            self.old_ebands.copy_within(0..nb, nb);
        }

        if !is_transient {
            self.old_log_e2.copy_from_slice(&self.old_log_e);
            self.old_log_e.copy_from_slice(&self.old_ebands);
        } else {
            for i in 0..2 * nb {
                self.old_log_e[i] = ming(self.old_log_e[i], self.old_ebands[i]);
            }
        }
        // In normal circumstances, we only allow the noise floor to increase by up to 2.4
        // dB/second, but when we're in DTX we give the weight of all missing packets to the
        // update packet.
        let max_background_increase: CeltGlog =
            imin(160, self.loss_duration + m) as f32 * gconst(0.001);
        for i in 0..2 * nb {
            self.background_log_e[i] = ming(
                self.background_log_e[i] + max_background_increase,
                self.old_ebands[i],
            );
        }
        // In case start or end were to change
        for ch in 0..2usize {
            for i in 0..start as usize {
                self.old_ebands[ch * nb + i] = 0.0;
                self.old_log_e[ch * nb + i] = -gconst(28.0);
                self.old_log_e2[ch * nb + i] = -gconst(28.0);
            }
            for i in end as usize..nb {
                self.old_ebands[ch * nb + i] = 0.0;
                self.old_log_e[ch * nb + i] = -gconst(28.0);
                self.old_log_e2[ch * nb + i] = -gconst(28.0);
            }
        }
        self.rng = dec.rng;
        #[cfg(feature = "qext")]
        if qext_bytes != 0 {
            self.rng ^= ext_dec.rng;
        }

        self.deemphasis_out(pcm, n, accum, bufs.freq);
        self.loss_duration = 0;
        self.plc_duration = 0;
        self.last_frame_type = FRAME_NORMAL;
        self.prefilter_and_fold = 0;
        if dec.tell() > 8 * len {
            return Err(Error::InternalError);
        }
        #[cfg(feature = "qext")]
        if qext_bytes != 0 && ext_dec.tell() > 8 * qext_bytes {
            return Err(Error::InternalError);
        }
        if dec.get_error() != 0 {
            self.error = 1;
        }
        Ok(frame_size / self.downsample)
    }

    // ---------------------------------------------------------------------------------------
    // CTLs (opus_custom_decoder_ctl)
    // ---------------------------------------------------------------------------------------

    /// `OPUS_SET_COMPLEXITY` (0..=10).
    pub const fn set_complexity(&mut self, value: i32) -> Result<()> {
        if value < 0 || value > 10 {
            return Err(Error::BadArg);
        }
        self.complexity = value;
        Ok(())
    }

    /// `OPUS_GET_COMPLEXITY`.
    #[must_use]
    pub const fn complexity(&self) -> i32 {
        self.complexity
    }

    /// `CELT_SET_START_BAND` (`0..nbEBands`).
    pub const fn set_start_band(&mut self, value: i32) -> Result<()> {
        if value < 0 || value >= self.mode.nb_ebands {
            return Err(Error::BadArg);
        }
        self.start = value;
        Ok(())
    }

    /// `CELT_SET_END_BAND` (`1..=nbEBands`).
    pub const fn set_end_band(&mut self, value: i32) -> Result<()> {
        if value < 1 || value > self.mode.nb_ebands {
            return Err(Error::BadArg);
        }
        self.end = value;
        Ok(())
    }

    /// `CELT_SET_CHANNELS`: number of coded (stream) channels (1 or 2).
    pub const fn set_channels(&mut self, value: i32) -> Result<()> {
        if value < 1 || value > 2 {
            return Err(Error::BadArg);
        }
        self.stream_channels = value;
        Ok(())
    }

    /// `CELT_GET_AND_CLEAR_ERROR`.
    pub const fn get_and_clear_error(&mut self) -> i32 {
        let v = self.error;
        self.error = 0;
        v
    }

    /// `OPUS_GET_LOOKAHEAD`.
    #[must_use]
    pub const fn lookahead(&self) -> i32 {
        self.overlap / self.downsample
    }

    /// `OPUS_GET_PITCH`.
    #[must_use]
    pub const fn pitch(&self) -> i32 {
        self.postfilter_period
    }

    /// `CELT_GET_MODE`.
    #[must_use]
    pub const fn mode(&self) -> &'m CeltMode {
        self.mode
    }

    /// `CELT_SET_SIGNALLING`.
    pub const fn set_signalling(&mut self, value: i32) {
        self.signalling = value;
    }

    /// `OPUS_GET_FINAL_RANGE`.
    #[must_use]
    pub const fn final_range(&self) -> u32 {
        self.rng
    }

    /// `OPUS_SET_PHASE_INVERSION_DISABLED` (0 or 1).
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

    /// Numeric `opus_custom_decoder_ctl` for requests taking an `opus_int32` value (SET
    /// requests and `OPUS_RESET_STATE`, whose value is ignored). Requests that are not SET-style
    /// return [`Error::BadArg`] (C would read a missing argument); unknown requests return
    /// [`Error::Unimplemented`].
    pub fn ctl_set(&mut self, request: i32, value: i32) -> Result<()> {
        match request {
            OPUS_SET_COMPLEXITY_REQUEST => self.set_complexity(value),
            CELT_SET_START_BAND_REQUEST => self.set_start_band(value),
            CELT_SET_END_BAND_REQUEST => self.set_end_band(value),
            CELT_SET_CHANNELS_REQUEST => self.set_channels(value),
            OPUS_RESET_STATE => {
                self.reset();
                Ok(())
            }
            CELT_SET_SIGNALLING_REQUEST => {
                self.set_signalling(value);
                Ok(())
            }
            OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST => self.set_phase_inversion_disabled(value),
            OPUS_GET_COMPLEXITY_REQUEST
            | CELT_GET_AND_CLEAR_ERROR_REQUEST
            | OPUS_GET_LOOKAHEAD_REQUEST
            | OPUS_GET_PITCH_REQUEST
            | CELT_GET_MODE_REQUEST
            | OPUS_GET_FINAL_RANGE_REQUEST
            | OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST => Err(Error::BadArg),
            _ => Err(Error::Unimplemented),
        }
    }

    /// Numeric `opus_custom_decoder_ctl` for GET requests writing one `opus_int32` (or
    /// `opus_uint32`, returned bit-cast for `OPUS_GET_FINAL_RANGE`). `CELT_GET_MODE` is only
    /// available as [`Self::mode`] and returns [`Error::BadArg`] here, as do SET requests;
    /// unknown requests return [`Error::Unimplemented`].
    pub const fn ctl_get(&mut self, request: i32) -> Result<i32> {
        match request {
            OPUS_GET_COMPLEXITY_REQUEST => Ok(self.complexity()),
            CELT_GET_AND_CLEAR_ERROR_REQUEST => Ok(self.get_and_clear_error()),
            OPUS_GET_LOOKAHEAD_REQUEST => Ok(self.lookahead()),
            OPUS_GET_PITCH_REQUEST => Ok(self.pitch()),
            OPUS_GET_FINAL_RANGE_REQUEST => Ok(self.final_range() as i32),
            OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST => Ok(self.phase_inversion_disabled()),
            OPUS_SET_COMPLEXITY_REQUEST
            | CELT_SET_START_BAND_REQUEST
            | CELT_SET_END_BAND_REQUEST
            | CELT_SET_CHANNELS_REQUEST
            | OPUS_RESET_STATE
            | CELT_GET_MODE_REQUEST
            | CELT_SET_SIGNALLING_REQUEST
            | OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST => Err(Error::BadArg),
            _ => Err(Error::Unimplemented),
        }
    }
}

/// `SINC_ORDER` (celt_decoder.c, deep PLC).
#[cfg(feature = "deep-plc")]
const SINC_ORDER: usize = 48;

/// `sinc_filter` (celt_decoder.c, deep PLC): the 16 ↔ 48 kHz resampling filter
/// (`h=cos(pi/2*abs(sin([-24:24]/48*pi*23./24)).^2); b=sinc([-24:24]/3*1.02).*h; b=b/sum(b);`).
#[cfg(feature = "deep-plc")]
const SINC_FILTER: [f32; SINC_ORDER + 1] = [
    4.2931e-05,
    -0.000190293,
    -0.000816132,
    -0.000637162,
    0.00141662,
    0.00354764,
    0.00184368,
    -0.00428274,
    -0.00856105,
    -0.0034003,
    0.00930201,
    0.0159616,
    0.00489785,
    -0.0169649,
    -0.0259484,
    -0.00596856,
    0.0286551,
    0.0405872,
    0.00649994,
    -0.0509284,
    -0.0716655,
    -0.00665212,
    0.134336,
    0.278927,
    0.339995,
    0.278927,
    0.134336,
    -0.00665212,
    -0.0716655,
    -0.0509284,
    0.00649994,
    0.0405872,
    0.0286551,
    -0.00596856,
    -0.0259484,
    -0.0169649,
    0.00489785,
    0.0159616,
    0.00930201,
    -0.0034003,
    -0.00856105,
    -0.00428274,
    0.00184368,
    0.00354764,
    0.00141662,
    -0.000637162,
    -0.000816132,
    -0.000190293,
    4.2931e-05,
];

/// Port of celt/celt_decoder.c:update_plc_state: feeds the last 40 ms of decoded audio
/// (`decode_mem`, channel `c` at `c*stride`; downmixed when `cc == 2`), downsampled to 16 kHz,
/// to the LPCNet PLC state before the first neural concealment, keeping its FEC read position.
#[cfg(feature = "deep-plc")]
fn update_plc_state(
    lpcnet: &mut LpcnetPlcState,
    decode_mem: &[CeltSig],
    stride: usize,
    plc_preemphasis_mem: &mut f32,
    cc: usize,
) {
    const DBS: usize = DECODE_BUFFER_SIZE as usize;
    let mut buf48k = [0f32; DBS];
    let mut buf16k = [0i16; PLC_UPDATE_SAMPLES];
    if cc == 1 {
        buf48k.copy_from_slice(&decode_mem[..DBS]);
    } else {
        let (d0, d1) = (&decode_mem[..DBS], &decode_mem[stride..stride + DBS]);
        for i in 0..DBS {
            // C: `.5*(a + b)` is a double multiply.
            buf48k[i] = (0.5f64 * f64::from(d0[i] + d1[i])) as f32;
        }
    }
    // Down-sample the last 40 ms.
    for i in 1..DBS {
        buf48k[i] += LPCNET_PREEMPHASIS * buf48k[i - 1];
    }
    *plc_preemphasis_mem = buf48k[DBS - 1];
    let offset = DBS - SINC_ORDER - 1 - 3 * (PLC_UPDATE_SAMPLES - 1);
    debug_assert!(3 * (PLC_UPDATE_SAMPLES - 1) + SINC_ORDER + offset == DBS - 1);
    for i in 0..PLC_UPDATE_SAMPLES {
        let mut sum: f32 = 0.0;
        for j in 0..SINC_ORDER + 1 {
            sum += buf48k[3 * i + j + offset] * SINC_FILTER[j];
        }
        buf16k[i] = crate::celt::mathops::float2int(min32(32767.0, max32(-32767.0, sum))) as i16;
    }
    let tmp_read_post = lpcnet.fec_read_pos;
    let tmp_fec_skip = lpcnet.fec_skip;
    for i in 0..PLC_UPDATE_FRAMES {
        lpcnet_plc_update(lpcnet, &buf16k[LPCNET_FRAME_SIZE * i..]);
    }
    lpcnet.fec_read_pos = tmp_read_post;
    lpcnet.fec_skip = tmp_fec_skip;
}

/// Packet header values of `celt_decode_with_ec_dred` (possibly updated by the custom-mode
/// in-band signalling).
#[derive(Debug, Clone, Copy)]
struct Header<'a> {
    data: Option<&'a [u8]>,
    len: i32,
    lm: i32,
    /// Stream channels (`C`).
    c: i32,
    end: i32,
    frame_size: i32,
    /// QEXT extension payload.
    #[cfg(feature = "qext")]
    ext: Option<&'a [u8]>,
}

/// Per-frame values passed from `celt_decode_with_ec_dred` to the frame decoder.
#[derive(Debug, Clone, Copy)]
struct FrameParams {
    len: i32,
    lm: i32,
    n: usize,
    /// Stream channels (`C`).
    c: i32,
    end: i32,
    eff_end: i32,
    frame_size: i32,
}

// -------------------------------------------------------------------------------------------
// Opus Custom API (CUSTOM_MODES)
// -------------------------------------------------------------------------------------------

/// The `opus_custom_decoder_*` public API (`OpusCustomDecoder` handle): a [`CeltDecoder`] for a
/// (possibly custom) mode with in-band signalling of the frame configuration.
#[cfg(feature = "custom-modes")]
#[derive(Debug, Clone)]
pub struct CustomDecoder<'m> {
    /// The decoder state (use it for the CTLs).
    pub st: CeltDecoder<'m>,
    /// `out` of `opus_custom_decode` / `opus_custom_decode24` (C VLA of `C*frame_size`).
    out: Vec<OpusRes>,
}

#[cfg(feature = "custom-modes")]
impl<'m> CustomDecoder<'m> {
    /// Port of celt/celt_decoder.c:opus_custom_decoder_create.
    pub fn opus_custom_decoder_create(mode: &'m CeltMode, channels: i32) -> Result<Self> {
        let st = CeltDecoder::opus_custom_decoder_init(mode, channels)?;
        let n_max = (mode.short_mdct_size * mode.nb_short_mdcts) as usize;
        Ok(Self {
            st,
            out: vec![0.0; channels as usize * n_max],
        })
    }

    /// Port of celt/celt_decoder.c:opus_custom_decode_float.
    pub fn opus_custom_decode_float(
        &mut self,
        data: Option<&[u8]>,
        len: i32,
        pcm: &mut [f32],
        frame_size: i32,
    ) -> Result<i32> {
        self.st
            .celt_decode_with_ec(data, len, pcm, frame_size, None, false)
    }

    /// Decodes into the internal `opus_res` buffer (`C*frame_size` samples).
    fn decode_to_out(&mut self, data: Option<&[u8]>, len: i32, frame_size: i32) -> Result<i32> {
        let c = self.st.channels as usize;
        let need = c * frame_size.max(0) as usize;
        if self.out.len() < need {
            self.out.resize(need, 0.0);
        }
        self.st
            .celt_decode_with_ec(data, len, &mut self.out[..need], frame_size, None, false)
    }

    /// Port of celt/celt_decoder.c:opus_custom_decode (float build: decodes to `opus_res` and
    /// converts with `RES2INT16`).
    pub fn opus_custom_decode(
        &mut self,
        data: Option<&[u8]>,
        len: i32,
        pcm: &mut [i16],
        frame_size: i32,
    ) -> Result<i32> {
        let ret = self.decode_to_out(data, len, frame_size)?;
        let cnt = self.st.channels as usize * ret as usize;
        if pcm.len() < cnt {
            return Err(Error::BadArg);
        }
        for (p, &o) in pcm[..cnt].iter_mut().zip(&self.out[..cnt]) {
            *p = crate::celt::arch::res2int16(o);
        }
        Ok(ret)
    }

    /// Port of celt/celt_decoder.c:opus_custom_decode24 (float build: decodes to `opus_res` and
    /// converts with `RES2INT24`).
    pub fn opus_custom_decode24(
        &mut self,
        data: Option<&[u8]>,
        len: i32,
        pcm: &mut [i32],
        frame_size: i32,
    ) -> Result<i32> {
        let ret = self.decode_to_out(data, len, frame_size)?;
        let cnt = self.st.channels as usize * ret as usize;
        if pcm.len() < cnt {
            return Err(Error::BadArg);
        }
        for (p, &o) in pcm[..cnt].iter_mut().zip(&self.out[..cnt]) {
            *p = crate::celt::arch::res2int24(o);
        }
        Ok(ret)
    }
}

#[cfg(feature = "custom-modes")]
impl<'m> core::ops::Deref for CustomDecoder<'m> {
    type Target = CeltDecoder<'m>;
    fn deref(&self) -> &Self::Target {
        &self.st
    }
}

#[cfg(feature = "custom-modes")]
impl core::ops::DerefMut for CustomDecoder<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.st
    }
}
