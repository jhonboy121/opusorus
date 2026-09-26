//! The Opus decoder: port of `src/opus_decoder.c` (float build).
//!
//! [`Decoder`] is the C `OpusDecoder`: it owns a SILK decoder and a CELT decoder and decodes
//! Opus packets of any mode (SILK-only, hybrid, CELT-only), including mode transitions with
//! redundant CELT frames, in-band FEC (LBRR), packet-loss concealment, DTX, output gain and the
//! float→int16 soft clipper.
//!
//! # Example
//!
//! ```
//! use opusorus::decoder::Decoder;
//!
//! # fn main() -> opusorus::Result<()> {
//! let mut dec = Decoder::new(48000, 2)?;
//! let mut pcm = vec![0i16; 960 * 2];
//! // A lost packet (`None`) runs the packet-loss concealment for `frame_size` samples.
//! let n = dec.decode(None, &mut pcm, 960, false)?;
//! assert_eq!(n, 960);
//! // A CELT-only 20 ms mono silence frame (TOC 0xF8 + 2 payload bytes).
//! let n = dec.decode(Some(&[0xF8, 0xFF, 0xFE]), &mut pcm, 960, false)?;
//! assert_eq!(n, 960);
//! # Ok(())
//! # }
//! ```
//!
//! # Mapping to the C API
//!
//! | C | Rust |
//! |---|---|
//! | `opus_decoder_get_size` | [`Decoder::get_size`] (Rust footprint) |
//! | `opus_decoder_create` / `opus_decoder_init` | [`Decoder::new`] / [`Decoder::init`] |
//! | `opus_decode` / `opus_decode24` / `opus_decode_float` | [`Decoder::decode`] / [`Decoder::decode24`] / [`Decoder::decode_float`] |
//! | `opus_decoder_get_nb_samples` | [`Decoder::nb_samples`] |
//! | `opus_decoder_ctl` | typed methods ([`Decoder::set_gain`], [`Decoder::final_range`], …) and [`Decoder::ctl_set`] / [`Decoder::ctl_get`] |
//! | `opus_decode_native` | [`Decoder::opus_decode_native`] (hidden, used by the multistream layer) |
//! | `opus_decoder_destroy` | `Drop` |
//!
//! The packet inspection helpers that live in `opus_decoder.c` (`opus_packet_get_bandwidth`,
//! `opus_packet_get_nb_frames`, `opus_packet_has_lbrr`, …) are in [`crate::packet`].
//!
//! # Deviations from C
//!
//! * Output buffers are slices; a buffer shorter than the samples the call may write returns
//!   [`Error::BadArg`] (C would write out of bounds). The required length is
//!   `min(frame_size, packet duration) * channels` for a normal decode and
//!   `frame_size * channels` for PLC/FEC.
//! * The packet is `Option<&[u8]>`: `None` or an empty slice is C's `data == NULL` / `len == 0`
//!   (packet loss). A negative C `len` (always `OPUS_BAD_ARG` in C) cannot be expressed.
//! * `validate_opus_decoder` / `celt_assert` are `debug_assert!`s; `MUST_SUCCEED` failures return
//!   [`Error::InternalError`] (the non-hardening C behaviour) instead of aborting.
//!
//! # DNN features
//!
//! * `deep-plc` (C `ENABLE_DEEP_PLC`): the decoder owns an LPCNet PLC state (`lpcnet`) that
//!   the SILK and CELT decoders update on good frames and use to conceal lost frames at
//!   complexity >= 5 ([`Decoder::set_complexity`]).
//! * `osce` (C `ENABLE_OSCE` + `ENABLE_OSCE_BWE`): SILK speech enhancement (LACE at complexity
//!   6, NoLACE at >= 7) and, with [`Decoder::set_osce_bwe`], the wideband → fullband extension
//!   of 16 kHz SILK at 48 kHz output (complexity >= 4).
//! * `dred` (C `ENABLE_DRED`): [`Decoder::dred_decode`] / [`Decoder::dred_decode24`] /
//!   [`Decoder::dred_decode_float`] conceal with the features of a [`crate::dred::Dred`]
//!   (`opus_decoder_dred_decode*`).
//!
//! The model weights are not compiled in (upstream `USE_WEIGHTS_FILE` semantics, PLAN D-015):
//! load a libopus weight blob with [`Decoder::set_dnn_blob`] (`OPUS_SET_DNN_BLOB`). Until then
//! the DNN paths stay off exactly as in an upstream build without loaded weights, except that
//! the OSCE bandwidth extension is only selected with a loaded model (a `USE_WEIGHTS_FILE`
//! build would run BBWENet without weights). Upstream's default build has the weights compiled
//! in and therefore behaves like this decoder after `set_dnn_blob` with a blob holding all
//! models. The blob stays loaded across [`Decoder::reset`] and [`Decoder::init`].

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::celt::arch::{
    OpusRes, Q15ONE, coef2val16, imin, mac16_16, mult16_16, mult16_16_p15, mult16_16_q15,
    mult16_32_p16, qconst16, res2int24, saturate, shr32,
};
#[cfg(feature = "qext")]
use crate::celt::celt::QEXT_EXTENSION_ID;
use crate::celt::celt_decoder::CeltDecoder;
use crate::celt::entdec::EcDec;
use crate::celt::mathops::{celt_exp2, celt_float2int16};
use crate::celt::static_modes::CeltMode;
#[cfg(feature = "dred")]
use crate::dnn::dred_decoder::OpusDred;
#[cfg(feature = "dred")]
use crate::dnn::dred_rdovae::DRED_NUM_FEATURES;
#[cfg(feature = "deep-plc")]
use crate::dnn::lpcnet_plc::LpcnetPlcState;
#[cfg(feature = "dred")]
use crate::dnn::lpcnet_plc::{lpcnet_plc_fec_add, lpcnet_plc_fec_clear};
#[cfg(feature = "osce")]
use crate::dnn::osce::{
    OSCE_METHOD_LACE, OSCE_METHOD_NOLACE, OSCE_METHOD_NONE, OSCE_MODE_CELT_ONLY, OSCE_MODE_HYBRID,
    OSCE_MODE_SILK_BBWE, OSCE_MODE_SILK_ONLY,
};
#[cfg(feature = "qext")]
use crate::extensions::ExtensionIterator;
use crate::packet::{
    MODE_CELT_ONLY, MODE_HYBRID, MODE_SILK_ONLY, get_nb_samples, opus_pcm_soft_clip_impl,
    parse_impl, toc_bandwidth, toc_mode, toc_nb_channels, toc_samples_per_frame,
};
use crate::silk::decoder::SilkDecoder;
#[cfg(feature = "osce")]
use crate::silk::decoder::SilkOsceControl;
use crate::silk::structs::SilkDecControlStruct;
use crate::{Bandwidth, Error, Result};

// ---------------------------------------------------------------------------------------------
// CTL request numbers (opus_defines.h) handled by `opus_decoder_ctl`
// ---------------------------------------------------------------------------------------------

/// `OPUS_GET_BANDWIDTH_REQUEST`.
pub const OPUS_GET_BANDWIDTH_REQUEST: i32 = 4009;
/// `OPUS_SET_COMPLEXITY_REQUEST`.
pub const OPUS_SET_COMPLEXITY_REQUEST: i32 = 4010;
/// `OPUS_GET_COMPLEXITY_REQUEST`.
pub const OPUS_GET_COMPLEXITY_REQUEST: i32 = 4011;
/// `OPUS_RESET_STATE`.
pub const OPUS_RESET_STATE: i32 = 4028;
/// `OPUS_GET_SAMPLE_RATE_REQUEST`.
pub const OPUS_GET_SAMPLE_RATE_REQUEST: i32 = 4029;
/// `OPUS_GET_FINAL_RANGE_REQUEST`.
pub const OPUS_GET_FINAL_RANGE_REQUEST: i32 = 4031;
/// `OPUS_GET_PITCH_REQUEST`.
pub const OPUS_GET_PITCH_REQUEST: i32 = 4033;
/// `OPUS_SET_GAIN_REQUEST`.
pub const OPUS_SET_GAIN_REQUEST: i32 = 4034;
/// `OPUS_GET_LAST_PACKET_DURATION_REQUEST`.
pub const OPUS_GET_LAST_PACKET_DURATION_REQUEST: i32 = 4039;
/// `OPUS_GET_GAIN_REQUEST`.
pub const OPUS_GET_GAIN_REQUEST: i32 = 4045;
/// `OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST`.
pub const OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST: i32 = 4046;
/// `OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST`.
pub const OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST: i32 = 4047;
/// `OPUS_SET_DNN_BLOB_REQUEST` (takes a pointer + length; use `Decoder::set_dnn_blob` with the DNN features).
pub const OPUS_SET_DNN_BLOB_REQUEST: i32 = 4052;
/// `OPUS_SET_OSCE_BWE_REQUEST`.
pub const OPUS_SET_OSCE_BWE_REQUEST: i32 = 4054;
/// `OPUS_GET_OSCE_BWE_REQUEST`.
pub const OPUS_GET_OSCE_BWE_REQUEST: i32 = 4055;
/// `OPUS_SET_IGNORE_EXTENSIONS_REQUEST`.
pub const OPUS_SET_IGNORE_EXTENSIONS_REQUEST: i32 = 4058;
/// `OPUS_GET_IGNORE_EXTENSIONS_REQUEST`.
pub const OPUS_GET_IGNORE_EXTENSIONS_REQUEST: i32 = 4059;

/// Largest supported API sampling rate.
#[cfg(feature = "qext")]
const MAX_FS: i32 = 96000;
/// Largest supported API sampling rate.
#[cfg(not(feature = "qext"))]
const MAX_FS: i32 = 48000;

/// Samples per channel of a 5 ms frame at [`MAX_FS`] (the `pcm_transition` stack buffer).
const MAX_F5: usize = (MAX_FS / 200) as usize;

/// Whether `fs` is a sampling rate accepted by `opus_decoder_init`.
const fn valid_fs(fs: i32) -> bool {
    matches!(fs, 48000 | 24000 | 16000 | 12000 | 8000) || (cfg!(feature = "qext") && fs == 96000)
}

/// Converts a C `st->bandwidth` value (0 = none yet) to a [`Bandwidth`].
const fn bandwidth_from_raw(bw: i32) -> Option<Bandwidth> {
    match Bandwidth::from_raw(bw) {
        Ok(b) => Some(b),
        Err(_) => None,
    }
}

/// Scratch buffers replacing the C VLAs (allocated at init, so decoding does not allocate).
#[derive(Debug, Clone)]
struct Scratch {
    /// `pcm_silk` (10 ms, SILK output when the caller's frame is shorter).
    pcm_silk: Vec<OpusRes>,
    /// `redundant_audio` (5 ms redundant CELT frame).
    redundant_audio: Vec<OpusRes>,
    /// `out` of `opus_decode` / `opus_decode24` (grown only for PLC/FEC requests longer than
    /// 120 ms).
    out: Vec<OpusRes>,
}

/// Where a `smooth_fade` input comes from: a separate buffer, or the output buffer itself (the
/// C calls pass the same pointer as an input and as `out`; each output sample only depends on
/// the inputs at the same index, so reading before writing is equivalent).
#[derive(Debug, Clone, Copy)]
enum FadeIn<'a> {
    Buf(&'a [OpusRes]),
    Out,
}

/// Port of `src/opus_decoder.c:smooth_fade` (float build, not `ENABLE_RES24`): cross-fades from
/// `in1` to `in2` over `overlap` samples using the squared CELT window.
fn smooth_fade(
    in1: FadeIn<'_>,
    in2: FadeIn<'_>,
    out: &mut [OpusRes],
    overlap: usize,
    channels: usize,
    window: &[f32],
    fs: i32,
) {
    // Note: 48000/Fs is 0 at 96 kHz (QEXT), so C then uses window[0] throughout.
    let inc = (48000 / fs) as usize;
    for c in 0..channels {
        for i in 0..overlap {
            let k = i * channels + c;
            let mut w = coef2val16(window[i * inc]);
            w = mult16_16_q15(w, w);
            let a = match in1 {
                FadeIn::Buf(b) => b[k],
                FadeIn::Out => out[k],
            };
            let b = match in2 {
                FadeIn::Buf(b) => b[k],
                FadeIn::Out => out[k],
            };
            out[k] = shr32(mac16_16(mult16_16(w, b), Q15ONE - w, a), 15);
        }
    }
}

/// The C code discards the return value of these calls (concealment into scratch buffers and
/// redundant CELT frames); kept as a named helper so the intent is explicit.
#[inline]
const fn c_ignores_result<T: Copy>(_r: Result<T>) {}

/// `MUST_SUCCEED` for CTL-like calls: a failure is `OPUS_INTERNAL_ERROR` (C aborts with
/// assertions enabled).
#[inline]
const fn must_succeed(r: Result<()>) -> Result<()> {
    match r {
        Ok(()) => Ok(()),
        Err(_) => Err(Error::InternalError),
    }
}

/// An Opus decoder (C `OpusDecoder`).
///
/// Decodes packets for one Opus stream (mono or stereo) at 8, 12, 16, 24 or 48 kHz (and 96 kHz
/// with the `qext` feature). The decoder is stateful: feed it the packets of one stream in order,
/// and signal lost packets with `None` to run the packet-loss concealment.
///
/// ```
/// use opusorus::decoder::Decoder;
///
/// # fn main() -> opusorus::Result<()> {
/// let mut dec = Decoder::new(16000, 1)?;
/// dec.set_gain(-256)?; // -1 dB (Q8)
/// let mut pcm = vec![0f32; 320];
/// // A SILK wideband 20 ms mono DTX frame (TOC only) is concealed.
/// let n = dec.decode_float(Some(&[0x48]), &mut pcm, 320, false)?;
/// assert_eq!(n, 320);
/// assert_eq!(dec.last_packet_duration(), 320);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Decoder {
    /// The CELT decoder (C: at `celt_dec_offset`).
    celt_dec: CeltDecoder<'static>,
    /// The SILK decoder (C: at `silk_dec_offset`; boxed, it is ~9 KB).
    silk_dec: Box<SilkDecoder>,
    channels: i32,
    /// Sampling rate at the API level.
    fs: i32,
    dec_control: SilkDecControlStruct,
    /// The `ENABLE_OSCE` / `ENABLE_OSCE_BWE` fields of `DecControl`.
    #[cfg(feature = "osce")]
    osce_ctl: SilkOsceControl,
    decode_gain: i32,
    complexity: i32,
    ignore_extensions: i32,
    /// `lpcnet`: the deep PLC state (boxed, it holds the PLC/FARGAN/pitch models).
    #[cfg(feature = "deep-plc")]
    lpcnet: Box<LpcnetPlcState>,

    // Everything beyond this point gets cleared on a reset (OPUS_DECODER_RESET_START).
    stream_channels: i32,
    bandwidth: i32,
    mode: i32,
    prev_mode: i32,
    frame_size: i32,
    prev_redundancy: bool,
    last_packet_duration: i32,
    softclip_mem: [f32; 2],
    range_final: u32,

    scratch: Scratch,
}

/// Snapshot of the Opus-level decoder state (for differential tests).
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecoderSnapshot {
    pub channels: i32,
    pub fs: i32,
    pub dec_control: SilkDecControlStruct,
    pub decode_gain: i32,
    pub complexity: i32,
    pub ignore_extensions: i32,
    pub stream_channels: i32,
    pub bandwidth: i32,
    pub mode: i32,
    pub prev_mode: i32,
    pub frame_size: i32,
    pub prev_redundancy: i32,
    pub last_packet_duration: i32,
    pub softclip_mem: [f32; 2],
    pub range_final: u32,
}

impl Decoder {
    /// Port of `src/opus_decoder.c:opus_decoder_create` / `opus_decoder_init`: creates a
    /// decoder for `channels` (1 or 2) output channels at `fs` Hz (8000, 12000, 16000, 24000,
    /// 48000, or 96000 with the `qext` feature).
    ///
    /// # Errors
    /// [`Error::BadArg`] for an unsupported rate or channel count; [`Error::InternalError`] if a
    /// sub-decoder fails to initialize.
    pub fn new(fs: i32, channels: i32) -> Result<Self> {
        if !valid_fs(fs) || (channels != 1 && channels != 2) {
            return Err(Error::BadArg);
        }
        // Initialize SILK decoder (silk_InitDecoder on cleared memory).
        let mut silk_dec = Box::new(SilkDecoder::new());
        if silk_dec.init() != 0 {
            return Err(Error::InternalError);
        }
        // Initialize CELT decoder
        let mut celt_dec = match CeltDecoder::celt_decoder_init(fs, channels) {
            Ok(d) => d,
            Err(_) => return Err(Error::InternalError),
        };
        celt_dec.set_signalling(0);

        let dec_control = SilkDecControlStruct {
            api_sample_rate: fs,
            n_channels_api: channels,
            ..SilkDecControlStruct::default()
        };
        let ch = channels as usize;
        let f10 = (fs / 100) as usize;
        let f5 = (fs / 200) as usize;
        let max_frame = (fs / 25 * 3) as usize;
        Ok(Self {
            celt_dec,
            silk_dec,
            channels,
            fs,
            dec_control,
            #[cfg(feature = "osce")]
            osce_ctl: SilkOsceControl::default(),
            decode_gain: 0,
            complexity: 0,
            ignore_extensions: 0,
            // lpcnet_plc_init (no model is compiled in: loaded by `set_dnn_blob`).
            #[cfg(feature = "deep-plc")]
            lpcnet: LpcnetPlcState::new(),
            stream_channels: channels,
            bandwidth: 0,
            mode: 0,
            prev_mode: 0,
            frame_size: fs / 400,
            prev_redundancy: false,
            last_packet_duration: 0,
            softclip_mem: [0.0; 2],
            range_final: 0,
            scratch: Scratch {
                pcm_silk: vec![0.0; f10 * ch],
                redundant_audio: vec![0.0; f5 * ch],
                out: vec![0.0; max_frame * ch],
            },
        })
    }

    /// Port of `src/opus_decoder.c:opus_decoder_init`: re-initializes this decoder in place for
    /// a (possibly different) rate and channel count. On error the decoder is left unchanged.
    ///
    /// # Errors
    /// As [`Decoder::new`].
    pub fn init(&mut self, fs: i32, channels: i32) -> Result<()> {
        #[allow(unused_mut, reason = "only mutated with DNN features")]
        let mut new = Self::new(fs, channels)?;
        // Keep the DNN models loaded by `set_dnn_blob` (they play the role of upstream's
        // compiled-in weights, which `opus_decoder_init` binds again).
        #[cfg(feature = "deep-plc")]
        {
            core::mem::swap(&mut new.lpcnet, &mut self.lpcnet);
            new.lpcnet.lpcnet_plc_init();
        }
        #[cfg(feature = "osce")]
        core::mem::swap(&mut new.silk_dec.osce_model, &mut self.silk_dec.osce_model);
        *self = new;
        Ok(())
    }

    /// Port of `src/opus_decoder.c:opus_decoder_get_size`: approximate memory footprint in bytes
    /// of a decoder with `channels` channels at 48 kHz (the Rust layout, not the C struct size),
    /// or 0 for an invalid channel count.
    #[must_use]
    pub fn get_size(channels: i32) -> usize {
        if !(1..=2).contains(&channels) {
            return 0;
        }
        let ch = channels as usize;
        let scratch = (480 + 240 + 5760) * ch * size_of::<OpusRes>();
        // The deep PLC state (without the heap-allocated model weights of a loaded blob).
        #[cfg(feature = "deep-plc")]
        let scratch = scratch + size_of::<LpcnetPlcState>();
        size_of::<Self>()
            + size_of::<SilkDecoder>()
            + crate::celt::celt_decoder::celt_decoder_get_size(channels) as usize
            + scratch
    }

    /// Number of output channels.
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels as usize
    }

    /// Port of `validate_opus_decoder` (assertions only).
    fn validate(&self) {
        debug_assert!(self.channels == 1 || self.channels == 2);
        debug_assert!(valid_fs(self.fs));
        debug_assert!(self.dec_control.api_sample_rate == self.fs);
        debug_assert!(matches!(
            self.dec_control.internal_sample_rate,
            0 | 16000 | 12000 | 8000
        ));
        debug_assert!(self.dec_control.n_channels_api == self.channels);
        debug_assert!(matches!(self.dec_control.n_channels_internal, 0..=2));
        debug_assert!(matches!(
            self.dec_control.payload_size_ms,
            0 | 10 | 20 | 40 | 60
        ));
        debug_assert!(self.stream_channels == 1 || self.stream_channels == 2);
    }

    // -----------------------------------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------------------------------

    /// Port of `src/opus_decoder.c:opus_decode_frame`: decodes one Opus frame (`data` is the
    /// frame payload without TOC; `None` or a length <= 1 runs the PLC/DTX) into `pcm`.
    fn opus_decode_frame(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [OpusRes],
        mut frame_size: i32,
        decode_fec: i32,
        #[cfg(feature = "qext")] ext: Option<&[u8]>,
    ) -> Result<i32> {
        let ch = self.channels as usize;
        let f20 = self.fs / 50;
        let f10 = f20 >> 1;
        let f5 = f10 >> 1;
        let f2_5 = f5 >> 1;
        if frame_size < f2_5 {
            return Err(Error::BufferTooSmall);
        }
        // Limit frame_size to avoid excessive stack allocations.
        frame_size = imin(frame_size, self.fs / 25 * 3);
        let mut len: i32 = match data {
            Some(d) => crate::packet::len_i32(d.len()),
            None => 0,
        };
        let mut data = data;
        // Payloads of 1 (2 including ToC) or 0 trigger the PLC/DTX
        if len <= 1 {
            data = None;
            // In that case, don't conceal more than what the ToC says
            frame_size = imin(frame_size, self.frame_size);
        }
        let mut audiosize: i32;
        let mode: i32;
        let bandwidth: i32;
        // C leaves `dec` uninitialized without data; it is then never read.
        let mut dec = EcDec::new(match data {
            Some(d) => d,
            None => &[],
        });
        if data.is_some() {
            audiosize = self.frame_size;
            mode = self.mode;
            bandwidth = self.bandwidth;
        } else {
            audiosize = frame_size;
            // Run PLC using last used mode (CELT if we ended with CELT redundancy)
            mode = if self.prev_redundancy {
                MODE_CELT_ONLY
            } else {
                self.prev_mode
            };
            bandwidth = 0;

            if mode == 0 {
                // If we haven't got any packet yet, all we can do is return zeros
                let n = audiosize as usize * ch;
                pcm[..n].fill(0.0);
                return Ok(audiosize);
            }

            // Avoids trying to run the PLC on sizes other than 2.5 (CELT), 5 (CELT),
            // 10, or 20 (e.g. 12.5 or 30 ms).
            if audiosize > f20 {
                let mut off = 0usize;
                loop {
                    let ret = self.opus_decode_frame(
                        None,
                        &mut pcm[off..],
                        imin(audiosize, f20),
                        0,
                        #[cfg(feature = "qext")]
                        None,
                    )?;
                    off += ret as usize * ch;
                    audiosize -= ret;
                    if audiosize <= 0 {
                        break;
                    }
                }
                return Ok(frame_size);
            } else if audiosize < f20 {
                if audiosize > f10 {
                    audiosize = f10;
                } else if mode != MODE_SILK_ONLY && audiosize > f5 && audiosize < f10 {
                    audiosize = f5;
                }
            }
        }

        // In fixed-point, we can tell CELT to do the accumulation on top of the
        // SILK PCM buffer. This saves some stack space.
        let celt_accum = mode != MODE_CELT_ONLY;

        let mut transition = false;
        if data.is_some()
            && self.prev_mode > 0
            && ((mode == MODE_CELT_ONLY
                && self.prev_mode != MODE_CELT_ONLY
                && !self.prev_redundancy)
                || (mode != MODE_CELT_ONLY && self.prev_mode == MODE_CELT_ONLY))
        {
            transition = true;
        }
        // `pcm_transition_celt` / `pcm_transition_silk` (F5*channels), only materialized when a
        // transition needs it.
        let mut pcm_transition: Option<[OpusRes; 2 * MAX_F5]> = None;
        if transition && mode == MODE_CELT_ONLY {
            let buf = pcm_transition.insert([0.0; 2 * MAX_F5]);
            c_ignores_result(self.opus_decode_frame(
                None,
                &mut buf[..],
                imin(f5, audiosize),
                0,
                #[cfg(feature = "qext")]
                None,
            ));
        }
        if audiosize > frame_size {
            return Err(Error::BadArg);
        }
        frame_size = audiosize;

        // SILK processing
        if mode != MODE_CELT_ONLY {
            let pcm_too_small = frame_size < f10;
            if self.prev_mode == MODE_CELT_ONLY {
                self.silk_dec.reset();
            }

            // The SILK PLC cannot produce frames of less than 10 ms
            self.dec_control.payload_size_ms = (1000 * audiosize / self.fs).max(10);

            if data.is_some() {
                self.dec_control.n_channels_internal = self.stream_channels;
                if mode == MODE_SILK_ONLY {
                    self.dec_control.internal_sample_rate = match bandwidth {
                        crate::constants::raw::OPUS_BANDWIDTH_NARROWBAND => 8000,
                        crate::constants::raw::OPUS_BANDWIDTH_MEDIUMBAND => 12000,
                        crate::constants::raw::OPUS_BANDWIDTH_WIDEBAND => 16000,
                        _ => {
                            debug_assert!(false, "SILK-only bandwidth above wideband");
                            16000
                        }
                    };
                } else {
                    // Hybrid mode
                    self.dec_control.internal_sample_rate = 16000;
                }
            }
            self.dec_control.enable_deep_plc = i32::from(self.complexity >= 5);
            #[cfg(feature = "osce")]
            {
                self.osce_ctl.osce_method = OSCE_METHOD_NONE;
                if self.complexity >= 6 {
                    self.osce_ctl.osce_method = OSCE_METHOD_LACE;
                }
                if self.complexity >= 7 {
                    self.osce_ctl.osce_method = OSCE_METHOD_NOLACE;
                }
                // Rust-only: BWE also needs a loaded model (see the module docs).
                if self.complexity >= 4
                    && self.osce_ctl.enable_osce_bwe != 0
                    && self.silk_dec.osce_model.loaded
                    && self.fs == 48000
                    && self.dec_control.internal_sample_rate == 16000
                    && (mode == MODE_SILK_ONLY || data.is_none())
                {
                    // request WB -> FB signal extension
                    self.osce_ctl.osce_extended_mode = OSCE_MODE_SILK_BBWE;
                } else {
                    // at this point, mode can only be MODE_SILK_ONLY or MODE_HYBRID
                    self.osce_ctl.osce_extended_mode = if mode == MODE_SILK_ONLY {
                        OSCE_MODE_SILK_ONLY
                    } else {
                        OSCE_MODE_HYBRID
                    };
                }
                if self.prev_mode == MODE_CELT_ONLY {
                    // Update extended mode for CELT->SILK transition
                    self.osce_ctl.prev_osce_extended_mode = OSCE_MODE_CELT_ONLY;
                }
            }

            let lost_flag = if data.is_none() {
                1
            } else {
                2 * i32::from(decode_fec != 0)
            };
            let mut decoded_samples = 0;
            let pcm_ptr: &mut [OpusRes] = if pcm_too_small {
                &mut self.scratch.pcm_silk[..]
            } else {
                &mut pcm[..]
            };
            let mut off = 0usize;
            loop {
                // Call SILK decoder
                let first_frame = i32::from(decoded_samples == 0);
                let mut silk_frame_size: i32 = 0;
                let silk_ret = self.silk_dec.silk_decode_dnn(
                    &mut self.dec_control,
                    #[cfg(feature = "osce")]
                    &mut self.osce_ctl,
                    lost_flag,
                    first_frame,
                    &mut dec,
                    &mut pcm_ptr[off..],
                    &mut silk_frame_size,
                    #[cfg(feature = "deep-plc")]
                    Some(&mut self.lpcnet),
                );
                if silk_ret != 0 {
                    if lost_flag != 0 {
                        // PLC failure should not be fatal
                        silk_frame_size = frame_size;
                        let end = (off + frame_size as usize * ch).min(pcm_ptr.len());
                        pcm_ptr[off..end].fill(0.0);
                    } else {
                        return Err(Error::InternalError);
                    }
                }
                off += silk_frame_size as usize * ch;
                decoded_samples += silk_frame_size;
                if decoded_samples >= frame_size {
                    break;
                }
            }
            if pcm_too_small {
                let n = frame_size as usize * ch;
                pcm[..n].copy_from_slice(&self.scratch.pcm_silk[..n]);
            }
        }

        let mut start_band = 0;
        let mut redundancy = false;
        let mut redundancy_bytes: i32 = 0;
        let mut celt_to_silk = false;
        if decode_fec == 0
            && mode != MODE_CELT_ONLY
            && data.is_some()
            && dec.tell() + 17 + 20 * i32::from(mode == MODE_HYBRID) <= 8 * len
        {
            // Check if we have a redundant 0-8 kHz band
            redundancy = if mode == MODE_HYBRID {
                dec.dec_bit_logp(12)
            } else {
                true
            };
            if redundancy {
                celt_to_silk = dec.dec_bit_logp(1);
                // redundancy_bytes will be at least two, in the non-hybrid
                // case due to the ec_tell() check above
                redundancy_bytes = if mode == MODE_HYBRID {
                    dec.dec_uint(256) as i32 + 2
                } else {
                    len - ((dec.tell() + 7) >> 3)
                };
                len -= redundancy_bytes;
                // This is a sanity check. It should never happen for a valid
                // packet, so the exact behaviour is not normative.
                if len * 8 < dec.tell() {
                    len = 0;
                    redundancy_bytes = 0;
                    redundancy = false;
                }
                // Shrink decoder because of raw bits
                dec.storage -= redundancy_bytes as u32;
            }
        }
        if mode != MODE_CELT_ONLY {
            start_band = 17;
        }

        if redundancy {
            transition = false;
        }

        if transition && mode != MODE_CELT_ONLY {
            let buf = pcm_transition.insert([0.0; 2 * MAX_F5]);
            c_ignores_result(self.opus_decode_frame(
                None,
                &mut buf[..],
                imin(f5, audiosize),
                0,
                #[cfg(feature = "qext")]
                None,
            ));
        }

        if bandwidth != 0 {
            let endband = match bandwidth {
                crate::constants::raw::OPUS_BANDWIDTH_NARROWBAND => 13,
                crate::constants::raw::OPUS_BANDWIDTH_MEDIUMBAND
                | crate::constants::raw::OPUS_BANDWIDTH_WIDEBAND => 17,
                crate::constants::raw::OPUS_BANDWIDTH_SUPERWIDEBAND => 19,
                crate::constants::raw::OPUS_BANDWIDTH_FULLBAND => 21,
                _ => {
                    debug_assert!(false, "invalid bandwidth");
                    21
                }
            };
            must_succeed(self.celt_dec.set_end_band(endband))?;
        }
        must_succeed(self.celt_dec.set_channels(self.stream_channels))?;

        let f5u = f5 as usize;
        let f2_5u = f2_5 as usize;
        let mut redundant_rng: u32 = 0;

        // 5 ms redundant frame for CELT->SILK
        if redundancy && celt_to_silk {
            // If the previous frame did not use CELT (the first redundancy frame in
            // a transition from SILK may have been lost) then the CELT decoder is
            // stale at this point and the redundancy audio is not useful, however
            // the final range is still needed (for testing), so the redundancy is
            // always decoded but the decoded audio may not be used
            must_succeed(self.celt_dec.set_start_band(0))?;
            if let Some(d) = data {
                c_ignores_result(self.celt_dec.celt_decode_with_ec(
                    Some(&d[len as usize..]),
                    redundancy_bytes,
                    &mut self.scratch.redundant_audio[..f5u * ch],
                    f5,
                    None,
                    false,
                ));
            }
            redundant_rng = self.celt_dec.final_range();
        }

        // MUST be after PLC
        must_succeed(self.celt_dec.set_start_band(start_band))?;

        let mut celt_err: Option<Error> = None;
        #[cfg(feature = "osce")]
        let decode_celt =
            mode != MODE_SILK_ONLY && self.osce_ctl.osce_extended_mode != OSCE_MODE_SILK_BBWE;
        #[cfg(not(feature = "osce"))]
        let decode_celt = mode != MODE_SILK_ONLY;
        if decode_celt {
            let celt_frame_size = imin(f20, frame_size);
            // Make sure to discard any previous CELT state
            if mode != self.prev_mode && self.prev_mode > 0 && !self.prev_redundancy {
                self.celt_dec.reset();
            }
            // Decode CELT
            #[cfg(feature = "deep-plc")]
            let r = self.celt_dec.celt_decode_with_ec_lpcnet(
                if decode_fec != 0 { None } else { data },
                len,
                pcm,
                celt_frame_size,
                Some(&mut dec),
                celt_accum,
                Some(&mut self.lpcnet),
                #[cfg(feature = "qext")]
                ext,
            );
            #[cfg(not(feature = "deep-plc"))]
            let r = self.celt_dec.celt_decode_with_ec_dred(
                if decode_fec != 0 { None } else { data },
                len,
                pcm,
                celt_frame_size,
                Some(&mut dec),
                celt_accum,
                #[cfg(feature = "qext")]
                ext,
            );
            if let Err(e) = r {
                celt_err = Some(e);
            }
            self.range_final = self.celt_dec.final_range();
        } else {
            let silence: [u8; 2] = [0xFF, 0xFF];
            if !celt_accum {
                pcm[..frame_size as usize * ch].fill(0.0);
            }
            // For hybrid -> SILK transitions, we let the CELT MDCT
            // do a fade-out by decoding a silence frame
            if self.prev_mode == MODE_HYBRID
                && !(redundancy && celt_to_silk && self.prev_redundancy)
            {
                must_succeed(self.celt_dec.set_start_band(0))?;
                c_ignores_result(self.celt_dec.celt_decode_with_ec(
                    Some(&silence),
                    2,
                    pcm,
                    f2_5,
                    None,
                    celt_accum,
                ));
            }
            self.range_final = dec.rng;
        }

        let celt_mode: &'static CeltMode = self.celt_dec.mode();
        let window: &[f32] = &celt_mode.window;

        // 5 ms redundant frame for SILK->CELT
        if redundancy && !celt_to_silk {
            self.celt_dec.reset();
            must_succeed(self.celt_dec.set_start_band(0))?;

            if let Some(d) = data {
                c_ignores_result(self.celt_dec.celt_decode_with_ec(
                    Some(&d[len as usize..]),
                    redundancy_bytes,
                    &mut self.scratch.redundant_audio[..f5u * ch],
                    f5,
                    None,
                    false,
                ));
            }
            redundant_rng = self.celt_dec.final_range();
            let o = ch * (frame_size as usize - f2_5u);
            smooth_fade(
                FadeIn::Out,
                FadeIn::Buf(&self.scratch.redundant_audio[ch * f2_5u..]),
                &mut pcm[o..],
                f2_5u,
                ch,
                window,
                self.fs,
            );
        }
        // 5ms redundant frame for CELT->SILK; ignore if the previous frame did not
        // use CELT (the first redundancy frame in a transition from SILK may have
        // been lost)
        if redundancy && celt_to_silk && (self.prev_mode != MODE_SILK_ONLY || self.prev_redundancy)
        {
            let n = ch * f2_5u;
            pcm[..n].copy_from_slice(&self.scratch.redundant_audio[..n]);
            smooth_fade(
                FadeIn::Buf(&self.scratch.redundant_audio[n..]),
                FadeIn::Out,
                &mut pcm[n..],
                f2_5u,
                ch,
                window,
                self.fs,
            );
        }
        if transition {
            // Allocated above whenever `transition` is still set here.
            if let Some(pt) = &pcm_transition {
                if audiosize >= f5 {
                    let n = ch * f2_5u;
                    pcm[..n].copy_from_slice(&pt[..n]);
                    smooth_fade(
                        FadeIn::Buf(&pt[n..]),
                        FadeIn::Out,
                        &mut pcm[n..],
                        f2_5u,
                        ch,
                        window,
                        self.fs,
                    );
                } else {
                    // Not enough time to do a clean transition, but we do it anyway
                    // This will not preserve amplitude perfectly and may introduce
                    // a bit of temporal aliasing, but it shouldn't be too bad and
                    // that's pretty much the best we can do. In any case, generating this
                    // transition it pretty silly in the first place
                    smooth_fade(
                        FadeIn::Buf(&pt[..]),
                        FadeIn::Out,
                        pcm,
                        f2_5u,
                        ch,
                        window,
                        self.fs,
                    );
                }
            }
        }

        if self.decode_gain != 0 {
            let gain = celt_exp2(mult16_16_p15(
                qconst16(6.48814081e-4f32, 25),
                self.decode_gain as f32,
            ));
            for x in &mut pcm[..frame_size as usize * ch] {
                let v = mult16_32_p16(*x, gain);
                *x = saturate(v, 32767);
            }
        }

        if len <= 1 {
            self.range_final = 0;
        } else {
            self.range_final ^= redundant_rng;
        }

        self.prev_mode = mode;
        self.prev_redundancy = redundancy && !celt_to_silk;

        match celt_err {
            Some(e) => Err(e),
            None => Ok(audiosize),
        }
    }

    /// Port of `src/opus_decoder.c:opus_decode_native`: decodes a packet (or conceals a lost
    /// one) into `opus_res` (float) samples. This is the entry point used by the multistream
    /// layer and the C ABI.
    ///
    /// * `data`: the packet; `None` or empty = lost packet (PLC).
    /// * `pcm`: output, `frame_size * channels` interleaved samples.
    /// * `decode_fec`: 0 or 1 (other values return [`Error::BadArg`] as in C).
    /// * `self_delimited`: parse the packet with self-delimiting framing (multistream);
    ///   `packet_offset` then receives the offset of the next packet (it is left untouched for
    ///   PLC, as in C).
    /// * `soft_clip`: apply the soft clipper (`opus_decode` int16 path).
    ///
    /// The C `dred` / `dred_offset` arguments are `NULL` / 0 here; see
    /// [`Decoder::dred_decode_float`].
    ///
    /// Returns the number of decoded samples per channel.
    ///
    /// # Errors
    /// The C error codes: [`Error::BadArg`], [`Error::BufferTooSmall`],
    /// [`Error::InvalidPacket`], [`Error::InternalError`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
    pub fn opus_decode_native(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [OpusRes],
        frame_size: i32,
        decode_fec: i32,
        self_delimited: bool,
        packet_offset: Option<&mut usize>,
        soft_clip: bool,
    ) -> Result<i32> {
        self.decode_native_impl(
            data,
            pcm,
            frame_size,
            decode_fec,
            self_delimited,
            packet_offset,
            soft_clip,
            #[cfg(feature = "dred")]
            None,
        )
    }

    /// `opus_decode_native` with the `ENABLE_DRED` arguments (`dred`: the processed DRED data
    /// and the C `dred_offset`, in samples).
    #[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
    fn decode_native_impl(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [OpusRes],
        frame_size: i32,
        decode_fec: i32,
        self_delimited: bool,
        packet_offset: Option<&mut usize>,
        soft_clip: bool,
        #[cfg(feature = "dred")] dred: Option<(&OpusDred, i32)>,
    ) -> Result<i32> {
        self.validate();
        if !(0..=1).contains(&decode_fec) {
            return Err(Error::BadArg);
        }
        let data = match data {
            Some(d) if !d.is_empty() => Some(d),
            _ => None,
        };
        // For FEC/PLC, frame_size has to be to have a multiple of 2.5 ms
        if (decode_fec != 0 || data.is_none()) && frame_size % (self.fs / 400) != 0 {
            return Err(Error::BadArg);
        }
        #[cfg(feature = "dred")]
        if let Some((dred, dred_offset)) = dred
            && dred.process_stage == 2
        {
            self.feed_dred(dred, dred_offset, frame_size);
        }
        let ch = self.channels as usize;
        let Some(data) = data else {
            let mut pcm_count = 0;
            loop {
                let ret = self.opus_decode_frame(
                    None,
                    &mut pcm[pcm_count as usize * ch..],
                    frame_size - pcm_count,
                    0,
                    #[cfg(feature = "qext")]
                    None,
                )?;
                pcm_count += ret;
                if pcm_count >= frame_size {
                    break;
                }
            }
            debug_assert!(pcm_count == frame_size);
            self.last_packet_duration = pcm_count;
            return Ok(pcm_count);
        };

        let toc = data[0];
        let packet_mode = toc_mode(toc);
        let packet_bandwidth = toc_bandwidth(toc).to_raw();
        let packet_frame_size = toc_samples_per_frame(toc, self.fs);
        let packet_stream_channels = toc_nb_channels(toc);

        let parsed = parse_impl(data, self_delimited)?;
        if let Some(po) = packet_offset {
            *po = parsed.packet_offset;
        }
        let padding: &[u8] = if self.ignore_extensions != 0 {
            &[]
        } else {
            parsed.padding
        };
        let count = parsed.nb_frames as i32;
        // C always initializes the iterator; only QEXT reads it (no side effects otherwise).
        #[cfg(feature = "qext")]
        let mut iter = ExtensionIterator::new(padding, count);
        #[cfg(not(feature = "qext"))]
        let _ = padding;

        if decode_fec != 0 {
            // If no FEC can be present, run the PLC (recursive call)
            if frame_size < packet_frame_size
                || packet_mode == MODE_CELT_ONLY
                || self.mode == MODE_CELT_ONLY
            {
                return self.opus_decode_native(None, pcm, frame_size, 0, false, None, soft_clip);
            }
            // Otherwise, run the PLC on everything except the size for which we might have FEC
            let duration_copy = self.last_packet_duration;
            if frame_size - packet_frame_size != 0 {
                let ret = self.opus_decode_native(
                    None,
                    pcm,
                    frame_size - packet_frame_size,
                    0,
                    false,
                    None,
                    soft_clip,
                );
                match ret {
                    Err(e) => {
                        self.last_packet_duration = duration_copy;
                        return Err(e);
                    }
                    Ok(r) => debug_assert!(r == frame_size - packet_frame_size),
                }
            }
            // Complete with FEC
            self.mode = packet_mode;
            self.bandwidth = packet_bandwidth;
            self.frame_size = packet_frame_size;
            self.stream_channels = packet_stream_channels;
            self.opus_decode_frame(
                Some(parsed.frames[0]),
                &mut pcm[ch * (frame_size - packet_frame_size) as usize..],
                packet_frame_size,
                1,
                #[cfg(feature = "qext")]
                None,
            )?;
            self.last_packet_duration = frame_size;
            return Ok(frame_size);
        }

        if count * packet_frame_size > frame_size {
            return Err(Error::BufferTooSmall);
        }

        // Update the state as the last step to avoid updating it on an invalid packet
        self.mode = packet_mode;
        self.bandwidth = packet_bandwidth;
        self.frame_size = packet_frame_size;
        self.stream_channels = packet_stream_channels;

        let mut nb_samples: i32 = 0;
        #[cfg_attr(
            not(feature = "qext"),
            expect(
                clippy::unused_enumerate_index,
                reason = "the frame index is only used by QEXT"
            )
        )]
        for (_i, &frame) in parsed.frames().iter().enumerate() {
            #[cfg(feature = "qext")]
            let ext_data: Option<&[u8]> = {
                let i = _i as i32;
                let mut ext_frame: i32 = -1;
                let mut ext_payload: Option<&[u8]> = None;
                while ext_frame < i {
                    let iter_copy = iter.clone();
                    match iter.find(QEXT_EXTENSION_ID) {
                        Ok(Some(e)) => {
                            ext_frame = e.frame;
                            ext_payload = Some(e.data);
                            if e.frame > i {
                                iter = iter_copy;
                            }
                        }
                        // C: `if (ret <= 0) break;`
                        Ok(None) | Err(_) => break,
                    }
                }
                if ext_frame == i { ext_payload } else { None }
            };
            let ret = self.opus_decode_frame(
                Some(frame),
                &mut pcm[nb_samples as usize * ch..],
                frame_size - nb_samples,
                0,
                #[cfg(feature = "qext")]
                ext_data,
            )?;
            debug_assert!(ret == packet_frame_size);
            nb_samples += ret;
        }
        self.last_packet_duration = nb_samples;
        if soft_clip {
            opus_pcm_soft_clip_impl(pcm, nb_samples, self.channels, &mut self.softclip_mem);
        } else {
            self.softclip_mem = [0.0; 2];
        }
        // FIXED_POINT: not ported (float build) — the fixed build has no soft clip.
        Ok(nb_samples)
    }

    /// Output length the decode functions may write, per channel: `frame_size` limited to the
    /// packet duration for a normal decode (C `opus_decode` computes the same bound).
    fn required_len(&self, data: Option<&[u8]>, frame_size: usize, decode_fec: bool) -> usize {
        let n = match data {
            Some(d) if !d.is_empty() && !decode_fec => match get_nb_samples(d, self.fs) {
                Ok(n) if n > 0 => frame_size.min(n as usize),
                _ => frame_size,
            },
            _ => frame_size,
        };
        n * self.channels as usize
    }

    /// Shared body of `opus_decode` / `opus_decode24`: C's `frame_size` checks, then
    /// `opus_decode_native` into the scratch `out` buffer. Returns the sample count per channel
    /// with the samples in `self.scratch.out` (taken out and handed to `convert`).
    fn decode_via_out(
        &mut self,
        data: Option<&[u8]>,
        pcm_len: usize,
        frame_size: i32,
        decode_fec: i32,
        soft_clip: bool,
        convert: &mut dyn FnMut(&[OpusRes]),
    ) -> Result<i32> {
        let mut frame_size = frame_size;
        if frame_size <= 0 {
            return Err(Error::BadArg);
        }
        if let Some(d) = data
            && !d.is_empty()
            && decode_fec == 0
        {
            match get_nb_samples(d, self.fs) {
                Ok(nb) if nb > 0 => frame_size = imin(frame_size, nb),
                _ => return Err(Error::InvalidPacket),
            }
        }
        debug_assert!(self.channels == 1 || self.channels == 2);
        let n = frame_size as usize * self.channels as usize;
        // Rust-only guard: C would write past the caller's buffer.
        if n > pcm_len {
            return Err(Error::BadArg);
        }
        let mut out = core::mem::take(&mut self.scratch.out);
        if out.len() < n {
            // Only PLC/FEC requests longer than 120 ms get here.
            out.resize(n, 0.0);
        }
        let ret = self.opus_decode_native(
            data,
            &mut out[..n],
            frame_size,
            decode_fec,
            false,
            None,
            soft_clip,
        );
        if let Ok(r) = ret
            && r > 0
        {
            convert(&out[..r as usize * self.channels as usize]);
        }
        self.scratch.out = out;
        ret
    }

    /// Port of `src/opus_decoder.c:opus_decode` with C argument types (`frame_size` and
    /// `decode_fec` as `int`), for the C ABI. See [`Decoder::decode`].
    ///
    /// # Errors
    /// As [`Decoder::decode`].
    #[doc(hidden)]
    pub fn opus_decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: i32,
    ) -> Result<i32> {
        let pcm_len = pcm.len();
        self.decode_via_out(data, pcm_len, frame_size, decode_fec, true, &mut |out| {
            celt_float2int16(out, &mut pcm[..out.len()]);
        })
    }

    /// Port of `src/opus_decoder.c:opus_decode24` with C argument types. See
    /// [`Decoder::decode24`].
    ///
    /// # Errors
    /// As [`Decoder::decode24`].
    #[doc(hidden)]
    pub fn opus_decode24(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i32],
        frame_size: i32,
        decode_fec: i32,
    ) -> Result<i32> {
        let pcm_len = pcm.len();
        self.decode_via_out(data, pcm_len, frame_size, decode_fec, false, &mut |out| {
            for (o, &x) in pcm.iter_mut().zip(out) {
                *o = res2int24(x);
            }
        })
    }

    /// Port of `src/opus_decoder.c:opus_decode_float` (float build) with C argument types. See
    /// [`Decoder::decode_float`].
    ///
    /// # Errors
    /// As [`Decoder::decode_float`].
    #[doc(hidden)]
    pub fn opus_decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: i32,
        decode_fec: i32,
    ) -> Result<i32> {
        if frame_size <= 0 {
            return Err(Error::BadArg);
        }
        if self.required_len(data, frame_size as usize, decode_fec != 0) > pcm.len() {
            return Err(Error::BadArg);
        }
        self.opus_decode_native(data, pcm, frame_size, decode_fec, false, None, false)
    }

    /// Decodes an Opus packet to interleaved 16-bit PCM (port of `opus_decode`, which applies
    /// the soft clipper before the conversion).
    ///
    /// * `data`: the packet, or `None` (or an empty slice) to signal a lost packet; the
    ///   decoder then runs packet-loss concealment for `frame_size` samples.
    /// * `pcm`: output, interleaved, `channels` samples per frame. It must hold
    ///   `min(frame_size, packet duration) * channels` samples (`frame_size * channels` for
    ///   PLC/FEC).
    /// * `frame_size`: the maximum number of samples per channel to decode. For PLC and FEC
    ///   it must be a multiple of 2.5 ms and is exactly the concealed duration; otherwise it
    ///   must be at least the packet duration (up to 120 ms = `fs * 3 / 25` samples).
    /// * `decode_fec`: decode the in-band FEC (LBRR) data of `data` for the *previous*,
    ///   lost packet instead of `data` itself.
    ///
    /// Returns the number of decoded samples per channel.
    ///
    /// # Errors
    /// [`Error::BadArg`] (invalid `frame_size`, `pcm` too short),
    /// [`Error::BufferTooSmall`] (`frame_size` shorter than the packet),
    /// [`Error::InvalidPacket`] (corrupted packet), [`Error::InternalError`].
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: usize,
        decode_fec: bool,
    ) -> Result<usize> {
        let fs = frame_size_i32(frame_size)?;
        self.opus_decode(data, pcm, fs, i32::from(decode_fec))
            .map(|n| n as usize)
    }

    /// Decodes an Opus packet to interleaved 24-bit PCM in `i32` (port of `opus_decode24`; no
    /// soft clipping). Arguments and errors as [`Decoder::decode`].
    ///
    /// # Errors
    /// As [`Decoder::decode`].
    pub fn decode24(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i32],
        frame_size: usize,
        decode_fec: bool,
    ) -> Result<usize> {
        let fs = frame_size_i32(frame_size)?;
        self.opus_decode24(data, pcm, fs, i32::from(decode_fec))
            .map(|n| n as usize)
    }

    /// Decodes an Opus packet to interleaved float PCM in `[-1, 1]` nominal range (port of
    /// `opus_decode_float`; no soft clipping, so samples may exceed ±1). Arguments and errors
    /// as [`Decoder::decode`].
    ///
    /// # Errors
    /// As [`Decoder::decode`].
    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: usize,
        decode_fec: bool,
    ) -> Result<usize> {
        let fs = frame_size_i32(frame_size)?;
        self.opus_decode_float(data, pcm, fs, i32::from(decode_fec))
            .map(|n| n as usize)
    }

    /// Port of `src/opus_decoder.c:opus_decoder_get_nb_samples`: the number of samples per
    /// channel of `packet` at this decoder's sampling rate.
    ///
    /// # Errors
    /// [`Error::BadArg`] for an empty packet, [`Error::InvalidPacket`] for a malformed or
    /// too long (> 120 ms) packet.
    pub fn nb_samples(&self, packet: &[u8]) -> Result<usize> {
        get_nb_samples(packet, self.fs).map(|n| n as usize)
    }

    // -----------------------------------------------------------------------------------------
    // CTLs (opus_decoder_ctl)
    // -----------------------------------------------------------------------------------------

    /// `OPUS_RESET_STATE`: resets the decoder to the state after [`Decoder::new`] (keeping the
    /// gain, complexity and extension settings), as when starting a new stream.
    pub fn reset(&mut self) {
        self.stream_channels = 0;
        self.bandwidth = 0;
        self.mode = 0;
        self.prev_mode = 0;
        self.frame_size = 0;
        self.prev_redundancy = false;
        self.last_packet_duration = 0;
        self.softclip_mem = [0.0; 2];
        self.range_final = 0;

        self.celt_dec.reset();
        self.silk_dec.reset();
        self.stream_channels = self.channels;
        self.frame_size = self.fs / 400;
        #[cfg(feature = "deep-plc")]
        self.lpcnet.lpcnet_plc_reset();
    }

    /// `OPUS_GET_BANDWIDTH`: the bandwidth of the last decoded packet (`None` before the first
    /// packet).
    #[must_use]
    pub const fn bandwidth(&self) -> Option<Bandwidth> {
        bandwidth_from_raw(self.bandwidth)
    }

    /// `OPUS_SET_COMPLEXITY` (0..=10): decoder complexity. Values >= 5 enable the deep PLC and
    /// >= 6/7 the OSCE enhancement when those features are compiled in.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside 0..=10.
    pub fn set_complexity(&mut self, value: i32) -> Result<()> {
        if !(0..=10).contains(&value) {
            return Err(Error::BadArg);
        }
        self.complexity = value;
        // C ignores the CELT CTL's return value (it cannot fail for 0..=10).
        must_succeed(self.celt_dec.set_complexity(value))
    }

    /// `OPUS_GET_COMPLEXITY`.
    #[must_use]
    pub const fn complexity(&self) -> i32 {
        self.complexity
    }

    /// `OPUS_SET_OSCE_BWE` (0 or 1): enables the OSCE wideband→fullband extension.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside 0..=1.
    #[cfg(feature = "osce")]
    pub const fn set_osce_bwe(&mut self, value: i32) -> Result<()> {
        if value < 0 || value > 1 {
            return Err(Error::BadArg);
        }
        self.osce_ctl.enable_osce_bwe = value;
        Ok(())
    }

    /// `OPUS_GET_OSCE_BWE`.
    #[cfg(feature = "osce")]
    #[must_use]
    pub const fn osce_bwe(&self) -> i32 {
        self.osce_ctl.enable_osce_bwe
    }

    /// `OPUS_GET_FINAL_RANGE`: the final state of the range coder for the last decoded packet
    /// (0 for concealed packets), for conformance testing against the encoder's value.
    #[must_use]
    pub const fn final_range(&self) -> u32 {
        self.range_final
    }

    /// `OPUS_GET_SAMPLE_RATE`.
    #[must_use]
    pub const fn sample_rate(&self) -> i32 {
        self.fs
    }

    /// `OPUS_GET_PITCH`: the pitch period of the last decoded frame (48 kHz samples), or 0 if
    /// unvoiced / unknown.
    #[must_use]
    pub const fn pitch(&self) -> i32 {
        if self.prev_mode == MODE_CELT_ONLY {
            self.celt_dec.pitch()
        } else {
            self.dec_control.prev_pitch_lag
        }
    }

    /// `OPUS_GET_GAIN`: the output gain in Q8 dB.
    #[must_use]
    pub const fn gain(&self) -> i32 {
        self.decode_gain
    }

    /// `OPUS_SET_GAIN`: output gain in Q8 dB (-32768..=32767, i.e. about ±128 dB).
    ///
    /// # Errors
    /// [`Error::BadArg`] outside the `i16` range.
    pub const fn set_gain(&mut self, value: i32) -> Result<()> {
        if value < -32768 || value > 32767 {
            return Err(Error::BadArg);
        }
        self.decode_gain = value;
        Ok(())
    }

    /// `OPUS_GET_LAST_PACKET_DURATION`: samples per channel of the last packet or concealed
    /// segment.
    #[must_use]
    pub const fn last_packet_duration(&self) -> usize {
        self.last_packet_duration as usize
    }

    /// `OPUS_SET_PHASE_INVERSION_DISABLED`: disables the CELT stereo phase inversion (for
    /// better mono downmixes).
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        // Cannot fail for 0/1.
        c_ignores_result(
            self.celt_dec
                .set_phase_inversion_disabled(i32::from(disabled)),
        );
    }

    /// `OPUS_GET_PHASE_INVERSION_DISABLED`.
    #[must_use]
    pub const fn phase_inversion_disabled(&self) -> bool {
        self.celt_dec.phase_inversion_disabled() != 0
    }

    /// `OPUS_SET_IGNORE_EXTENSIONS`: ignore the packet padding extensions (e.g. QEXT).
    pub const fn set_ignore_extensions(&mut self, ignore: bool) {
        self.ignore_extensions = ignore as i32;
    }

    /// `OPUS_GET_IGNORE_EXTENSIONS`.
    #[must_use]
    pub const fn ignore_extensions(&self) -> bool {
        self.ignore_extensions != 0
    }

    /// `OPUS_SET_DNN_BLOB` (upstream `USE_WEIGHTS_FILE` builds): loads the DNN weights from a
    /// libopus weight blob (as written by upstream `write_lpcnet_weights`, i.e. a concatenation
    /// of weight records). With `deep-plc` the blob must hold the PLC, FARGAN and pitch DNN
    /// models; with `osce` also LACE, NoLACE and BBWENet. Every model is attempted (C:
    /// `lpcnet_plc_load_model` then `silk_LoadOSCEModels`); the ones that bind are used.
    ///
    /// # Errors
    /// [`Error::BadArg`] if a model cannot be loaded from the blob (C returns `1`, not an error
    /// code, in that case).
    #[cfg(any(feature = "deep-plc", feature = "osce"))]
    pub fn set_dnn_blob(&mut self, data: &[u8]) -> Result<()> {
        #[cfg(feature = "deep-plc")]
        let plc_ok = self.lpcnet.load_model(data).is_ok();
        #[cfg(not(feature = "deep-plc"))]
        let plc_ok = true;
        let osce_ok = self.silk_dec.load_osce_models(Some(data)) == 0;
        if plc_ok && osce_ok {
            Ok(())
        } else {
            Err(Error::BadArg)
        }
    }

    /// Whether DNN models are loaded (C `lpcnet.loaded`, `osce_model.loaded`): the deep PLC
    /// (`deep-plc`) and the OSCE models (`osce`; always `false` without the feature).
    #[cfg(any(feature = "deep-plc", feature = "osce"))]
    #[must_use]
    pub fn dnn_loaded(&self) -> (bool, bool) {
        #[cfg(feature = "deep-plc")]
        let plc = self.lpcnet.loaded;
        #[cfg(not(feature = "deep-plc"))]
        let plc = false;
        #[cfg(feature = "osce")]
        let osce = self.silk_dec.osce_model.loaded;
        #[cfg(not(feature = "osce"))]
        let osce = false;
        (plc, osce)
    }

    /// Numeric `opus_decoder_ctl` for requests taking an `opus_int32` value (SET requests and
    /// `OPUS_RESET_STATE`, whose value is ignored). GET requests return [`Error::BadArg`];
    /// unknown requests return [`Error::Unimplemented`], like C. The pointer-taking
    /// `OPUS_SET_DNN_BLOB` is not available numerically (use [`Decoder::set_dnn_blob`]) and
    /// returns [`Error::Unimplemented`], as the C library built with compiled-in weights does.
    ///
    /// # Errors
    /// [`Error::BadArg`] for an out-of-range value, [`Error::Unimplemented`] for an unknown
    /// request.
    pub fn ctl_set(&mut self, request: i32, value: i32) -> Result<()> {
        match request {
            OPUS_SET_COMPLEXITY_REQUEST => self.set_complexity(value),
            OPUS_RESET_STATE => {
                self.reset();
                Ok(())
            }
            OPUS_SET_GAIN_REQUEST => self.set_gain(value),
            OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST => {
                if !(0..=1).contains(&value) {
                    return Err(Error::BadArg);
                }
                self.set_phase_inversion_disabled(value != 0);
                Ok(())
            }
            OPUS_SET_IGNORE_EXTENSIONS_REQUEST => {
                if !(0..=1).contains(&value) {
                    return Err(Error::BadArg);
                }
                self.set_ignore_extensions(value != 0);
                Ok(())
            }
            #[cfg(feature = "osce")]
            OPUS_SET_OSCE_BWE_REQUEST => self.set_osce_bwe(value),
            #[cfg(feature = "osce")]
            OPUS_GET_OSCE_BWE_REQUEST => Err(Error::BadArg),
            OPUS_GET_BANDWIDTH_REQUEST
            | OPUS_GET_COMPLEXITY_REQUEST
            | OPUS_GET_FINAL_RANGE_REQUEST
            | OPUS_GET_SAMPLE_RATE_REQUEST
            | OPUS_GET_PITCH_REQUEST
            | OPUS_GET_GAIN_REQUEST
            | OPUS_GET_LAST_PACKET_DURATION_REQUEST
            | OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST
            | OPUS_GET_IGNORE_EXTENSIONS_REQUEST => Err(Error::BadArg),
            _ => Err(Error::Unimplemented),
        }
    }

    /// Numeric `opus_decoder_ctl` for GET requests writing one `opus_int32` (or `opus_uint32`,
    /// returned bit-cast for `OPUS_GET_FINAL_RANGE`). SET requests return [`Error::BadArg`];
    /// unknown requests return [`Error::Unimplemented`], like C.
    ///
    /// # Errors
    /// See above.
    pub const fn ctl_get(&mut self, request: i32) -> Result<i32> {
        match request {
            OPUS_GET_BANDWIDTH_REQUEST => Ok(self.bandwidth),
            OPUS_GET_COMPLEXITY_REQUEST => Ok(self.complexity),
            OPUS_GET_FINAL_RANGE_REQUEST => Ok(self.range_final as i32),
            OPUS_GET_SAMPLE_RATE_REQUEST => Ok(self.fs),
            OPUS_GET_PITCH_REQUEST => Ok(self.pitch()),
            OPUS_GET_GAIN_REQUEST => Ok(self.decode_gain),
            OPUS_GET_LAST_PACKET_DURATION_REQUEST => Ok(self.last_packet_duration),
            OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST => {
                Ok(self.celt_dec.phase_inversion_disabled())
            }
            OPUS_GET_IGNORE_EXTENSIONS_REQUEST => Ok(self.ignore_extensions),
            #[cfg(feature = "osce")]
            OPUS_GET_OSCE_BWE_REQUEST => Ok(self.osce_ctl.enable_osce_bwe),
            #[cfg(feature = "osce")]
            OPUS_SET_OSCE_BWE_REQUEST => Err(Error::BadArg),
            OPUS_SET_COMPLEXITY_REQUEST
            | OPUS_RESET_STATE
            | OPUS_SET_GAIN_REQUEST
            | OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST
            | OPUS_SET_IGNORE_EXTENSIONS_REQUEST => Err(Error::BadArg),
            _ => Err(Error::Unimplemented),
        }
    }

    // -----------------------------------------------------------------------------------------
    // DRED (ENABLE_DRED)
    // -----------------------------------------------------------------------------------------

    /// The `ENABLE_DRED` block of `opus_decode_native`: queues the DRED features covering the
    /// concealed `frame_size` samples (starting `dred_offset` samples before the end of the
    /// DRED data) in the deep PLC.
    #[cfg(feature = "dred")]
    fn feed_dred(&mut self, dred: &OpusDred, dred_offset: i32, frame_size: i32) {
        lpcnet_plc_fec_clear(&mut self.lpcnet);
        let f10 = self.fs / 100;
        // if blend==0, the last PLC call was "update" and we need to feed two extra 10-ms
        // frames.
        let init_frames = if self.lpcnet.blend == 0 { 2 } else { 0 };
        let features_per_frame = imax_i32(1, frame_size / f10);
        let needed_feature_frames = init_frames + features_per_frame;
        for i in 0..needed_feature_frames {
            // We floor instead of rounding because 5-ms overlap compensates for the missing 0.5
            // rounding offset.
            // C: `(int)floor(((float)dred_offset + dred->dred_offset*F10/4)/F10)`; the int
            // product is computed with wrapping (C: UB on overflow of absurd offsets).
            let num = dred_offset as f32 + (dred.dred_offset.wrapping_mul(f10) / 4) as f32;
            let feature_offset =
                init_frames - i - 2 + crate::math::floor(f64::from(num / f10 as f32)) as i32;
            // C: `feature_offset <= 4*dred->nb_latents-1`.
            if feature_offset < 4 * dred.nb_latents && feature_offset >= 0 {
                let off = feature_offset as usize * DRED_NUM_FEATURES;
                lpcnet_plc_fec_add(&mut self.lpcnet, Some(&dred.fec_features[off..]));
            } else if feature_offset >= 0 {
                lpcnet_plc_fec_add(&mut self.lpcnet, None);
            }
        }
    }

    /// Port of `src/opus_decoder.c:opus_decoder_dred_decode_float` with C argument types (for
    /// the C ABI). See [`Decoder::dred_decode_float`].
    ///
    /// # Errors
    /// As [`Decoder::dred_decode_float`].
    #[cfg(feature = "dred")]
    #[doc(hidden)]
    pub fn opus_decoder_dred_decode_float(
        &mut self,
        dred: &crate::dred::Dred,
        dred_offset: i32,
        pcm: &mut [f32],
        frame_size: i32,
    ) -> Result<i32> {
        if frame_size <= 0 {
            return Err(Error::BadArg);
        }
        // Rust-only guard: C would write past the caller's buffer.
        if frame_size as usize * self.channels as usize > pcm.len() {
            return Err(Error::BadArg);
        }
        self.decode_native_impl(
            None,
            pcm,
            frame_size,
            0,
            false,
            None,
            false,
            Some((dred.inner(), dred_offset)),
        )
    }

    /// `opus_decoder_dred_decode` / `opus_decoder_dred_decode24`: `opus_decode_native` into
    /// the scratch buffer (soft clipping for the int16 path), then `convert`.
    #[cfg(feature = "dred")]
    fn dred_decode_via_out(
        &mut self,
        dred: &OpusDred,
        dred_offset: i32,
        pcm_len: usize,
        frame_size: i32,
        soft_clip: bool,
        convert: &mut dyn FnMut(&[OpusRes]),
    ) -> Result<i32> {
        if frame_size <= 0 {
            return Err(Error::BadArg);
        }
        debug_assert!(self.channels == 1 || self.channels == 2);
        let n = frame_size as usize * self.channels as usize;
        // Rust-only guard: C would write past the caller's buffer.
        if n > pcm_len {
            return Err(Error::BadArg);
        }
        let mut out = core::mem::take(&mut self.scratch.out);
        if out.len() < n {
            out.resize(n, 0.0);
        }
        let ret = self.decode_native_impl(
            None,
            &mut out[..n],
            frame_size,
            0,
            false,
            None,
            soft_clip,
            Some((dred, dred_offset)),
        );
        if let Ok(r) = ret
            && r > 0
        {
            convert(&out[..r as usize * self.channels as usize]);
        }
        self.scratch.out = out;
        ret
    }

    /// Port of `src/opus_decoder.c:opus_decoder_dred_decode` with C argument types (for the C
    /// ABI). See [`Decoder::dred_decode`].
    ///
    /// # Errors
    /// As [`Decoder::dred_decode`].
    #[cfg(feature = "dred")]
    #[doc(hidden)]
    pub fn opus_decoder_dred_decode(
        &mut self,
        dred: &crate::dred::Dred,
        dred_offset: i32,
        pcm: &mut [i16],
        frame_size: i32,
    ) -> Result<i32> {
        let pcm_len = pcm.len();
        self.dred_decode_via_out(
            dred.inner(),
            dred_offset,
            pcm_len,
            frame_size,
            true,
            &mut |out| {
                celt_float2int16(out, &mut pcm[..out.len()]);
            },
        )
    }

    /// Port of `src/opus_decoder.c:opus_decoder_dred_decode24` with C argument types (for the C
    /// ABI). See [`Decoder::dred_decode24`].
    ///
    /// # Errors
    /// As [`Decoder::dred_decode24`].
    #[cfg(feature = "dred")]
    #[doc(hidden)]
    pub fn opus_decoder_dred_decode24(
        &mut self,
        dred: &crate::dred::Dred,
        dred_offset: i32,
        pcm: &mut [i32],
        frame_size: i32,
    ) -> Result<i32> {
        let pcm_len = pcm.len();
        self.dred_decode_via_out(
            dred.inner(),
            dred_offset,
            pcm_len,
            frame_size,
            false,
            &mut |out| {
                for (o, &x) in pcm.iter_mut().zip(out) {
                    *o = res2int24(x);
                }
            },
        )
    }

    /// Conceals `frame_size` samples per channel using the redundancy of a processed
    /// [`crate::dred::Dred`] (port of `opus_decoder_dred_decode`; int16 output with soft
    /// clipping).
    ///
    /// `dred_offset` is the position, in samples before the end of the DRED data's newest
    /// frame, of the first concealed sample (as returned by
    /// [`crate::dred::DredDecoder::parse`]). Without a processed `dred` (or without a deep PLC
    /// model, see [`Decoder::set_dnn_blob`]) this is ordinary packet-loss concealment.
    /// `frame_size` must be a multiple of 2.5 ms and `pcm` must hold `frame_size * channels`
    /// samples.
    ///
    /// # Errors
    /// [`Error::BadArg`] for an invalid `frame_size` or a short `pcm`.
    #[cfg(feature = "dred")]
    pub fn dred_decode(
        &mut self,
        dred: &crate::dred::Dred,
        dred_offset: i32,
        pcm: &mut [i16],
        frame_size: usize,
    ) -> Result<usize> {
        let fs = frame_size_i32(frame_size)?;
        self.opus_decoder_dred_decode(dred, dred_offset, pcm, fs)
            .map(|n| n as usize)
    }

    /// [`Decoder::dred_decode`] with 24-bit output in `i32` (port of
    /// `opus_decoder_dred_decode24`; no soft clipping).
    ///
    /// # Errors
    /// As [`Decoder::dred_decode`].
    #[cfg(feature = "dred")]
    pub fn dred_decode24(
        &mut self,
        dred: &crate::dred::Dred,
        dred_offset: i32,
        pcm: &mut [i32],
        frame_size: usize,
    ) -> Result<usize> {
        let fs = frame_size_i32(frame_size)?;
        self.opus_decoder_dred_decode24(dred, dred_offset, pcm, fs)
            .map(|n| n as usize)
    }

    /// [`Decoder::dred_decode`] with float output (port of `opus_decoder_dred_decode_float`;
    /// no soft clipping).
    ///
    /// # Errors
    /// As [`Decoder::dred_decode`].
    #[cfg(feature = "dred")]
    pub fn dred_decode_float(
        &mut self,
        dred: &crate::dred::Dred,
        dred_offset: i32,
        pcm: &mut [f32],
        frame_size: usize,
    ) -> Result<usize> {
        let fs = frame_size_i32(frame_size)?;
        self.opus_decoder_dred_decode_float(dred, dred_offset, pcm, fs)
            .map(|n| n as usize)
    }

    /// The DNN state in the layout of the oracle's `oracle_di_dec_dump` (for differential
    /// tests): lpcnet `{loaded, analysis_gap, fec_read_pos, fec_fill_pos, fec_skip,
    /// analysis_pos, predict_pos, blend, loss_count}`, `DecControl` `{osce_method,
    /// enable_osce_bwe, osce_extended_mode, prev_osce_extended_mode}`, and lpcnet `pcm` +
    /// `features`.
    #[cfg(all(feature = "deep-plc", feature = "internals"))]
    #[doc(hidden)]
    #[must_use]
    pub fn dnn_debug_state(&self) -> ([i32; 13], Vec<f32>) {
        let l = &self.lpcnet;
        #[cfg(feature = "osce")]
        let o = [
            self.osce_ctl.osce_method,
            self.osce_ctl.enable_osce_bwe,
            self.osce_ctl.osce_extended_mode,
            self.osce_ctl.prev_osce_extended_mode,
        ];
        #[cfg(not(feature = "osce"))]
        let o = [0; 4];
        let ints = [
            i32::from(l.loaded),
            l.analysis_gap,
            l.fec_read_pos,
            l.fec_fill_pos,
            l.fec_skip,
            l.analysis_pos,
            l.predict_pos,
            l.blend,
            l.loss_count,
            o[0],
            o[1],
            o[2],
            o[3],
        ];
        let mut floats = Vec::with_capacity(l.pcm.len() + l.features.len());
        floats.extend_from_slice(&l.pcm);
        floats.extend_from_slice(&l.features);
        (ints, floats)
    }

    /// The Opus-level state (for differential tests).
    #[doc(hidden)]
    #[must_use]
    pub const fn snapshot(&self) -> DecoderSnapshot {
        DecoderSnapshot {
            channels: self.channels,
            fs: self.fs,
            dec_control: self.dec_control,
            decode_gain: self.decode_gain,
            complexity: self.complexity,
            ignore_extensions: self.ignore_extensions,
            stream_channels: self.stream_channels,
            bandwidth: self.bandwidth,
            mode: self.mode,
            prev_mode: self.prev_mode,
            frame_size: self.frame_size,
            prev_redundancy: self.prev_redundancy as i32,
            last_packet_duration: self.last_packet_duration,
            softclip_mem: self.softclip_mem,
            range_final: self.range_final,
        }
    }
}

/// Converts a caller `frame_size` to the C `int` (values beyond `i32::MAX` are
/// [`Error::BadArg`]: they could never fit a buffer).
fn frame_size_i32(frame_size: usize) -> Result<i32> {
    i32::try_from(frame_size).map_err(|_| Error::BadArg)
}

// The OpusDREDDecoder / OpusDRED API of opus_decoder.c (opus_dred_decoder_*, opus_dred_parse,
// opus_dred_process, ...) is in `crate::dred`; `opus_decoder_dred_decode*` are the
// `Decoder::dred_decode*` methods (feature `dred`). Without ENABLE_DRED every one of them returns
// OPUS_UNIMPLEMENTED (or 0 for the sizes and NULL for opus_dred_alloc), which the C ABI layer can
// reproduce directly.

/// C `IMAX` on `int`.
#[cfg(feature = "dred")]
#[inline]
const fn imax_i32(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}
