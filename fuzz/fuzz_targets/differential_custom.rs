//! Differential Opus Custom API (`custom-modes`): `opus_custom_mode_create` (static and
//! custom modes, any sampling rate / frame size, valid or not), `opus_custom_encoder_create` +
//! `opus_custom_encode` / `encode24` / `encode_float`, `opus_custom_decoder_create` +
//! `opus_custom_decode` / `decode24` / `decode_float` (including PLC and arbitrary packets) and
//! the numeric `opus_custom_{en,de}coder_ctl` requests, opusorus against C libopus. Creation
//! errors, every return code, packet byte, output sample (bit-exact), final range and GET CTL
//! must match. Encoder output is decoded by both decoders (or dropped as a lost packet).
//!
//! Input: `fs_sel:u8 frame_sel:u8 ch:u8` (bit 0: encoder channels - 1, bit 1: decoder
//! channels - 1, bit 2: invalid channel count 3 for the encoder; C accepts 0, which the port
//! rejects by design, and allocates a negative size for -1), then repeated:
//!
//! ```text
//! tag:u8   tag % 8:
//!   0..=2  encode  flags:u8 fsel:u8 nb:u16 kind:u8 amp:u8 param:u16 [raw samples]
//!   3      decode  flags:u8 fsel:u8 len:u16 packet[len]
//!   4      PLC     flags:u8 fsel:u8
//!   5      encoder ctl  sel:u8 vsel:u8 [value:i32 if vsel >= 128]
//!   6      decoder ctl  sel:u8 vsel:u8 [value:i32 if vsel >= 128]
//!   7      bands   which:u8 start_sel:u8 end_sel:u8
//! flags: bits 0-1 input format (0 i16, 1 i24, 2/3 float), bits 2-3 output format, bit 4 drop
//!        the encoded packet (PLC instead of decoding it).
//! fsel:  < 0x80 the mode's frame size >> (fsel & 3); else a raw size (see `frame_size`).
//! nb:    `nb % 4002 - 1` bytes (`nbCompressedBytes`; also the buffer size).
//! ```
//!
//! Input classes that are undefined behaviour in C (out-of-bounds accesses, failed
//! `celt_assert`s) are not generated, so the two sides can be compared:
//! * `CELT_SET_CHANNELS` above the encoder's channel count;
//! * QEXT with signalling off (C writes the header bits to `compressed[-1]`): `OPUS_SET_QEXT(1)`
//!   only while signalling is on, and signalling stays on while QEXT is on;
//! * an end band beyond `effEBands` (past the MDCT size: C reads past its `X` buffer; mono
//!   encoders without signalling happen to match, with signalling they diverge), and start
//!   band >= end band;
//! * encoder: start band 17 while signalling is on (signalling rounds `end` up, see
//!   `celt_encode_with_ec`, which can bring back the out-of-bounds hybrid folding below);
//! * decoder: an empty (non-NULL) packet while signalling is on (C reads `data[0]`), an end
//!   band beyond `effEBands` (past the MDCT size), start band 17 while signalling is on (a
//!   signalled packet sets `end` from its TOC, possibly below the start: C asserts
//!   `start < end`);
//! * start bands other than 0 and 17 (`validate_celt_decoder` asserts it; the port's debug
//!   assertion fires), and a start band 17 whose `special_hybrid_folding` (`bands.c`, at band
//!   `start + 1`) would copy out of `_norm` or with a negative length: only static-mode hybrid
//!   layouts (the Opus encoder's start band 17 at 48 kHz) are safe in general, custom modes
//!   often are not;
//! * frame sizes <= 0 (C VLAs of `C*frame_size` samples);
//! * encoder: an end band below 3 (`dynalloc_analysis` reads `bandLogE3[end-3]`), and stereo
//!   encoding (`CELT_SET_CHANNELS` 2) with an end band below 13 or in a mode with fewer than 13
//!   effective bands: `stereo_analysis` always sums the first 13 bands of `X`, but
//!   `normalise_bands` only fills `X` up to `effEnd`, so C reads uninitialized stack memory
//!   (packets then differ at random; the Opus encoder never goes below 13 bands);
//! * with `qext`, a main-band end band of 2 or 14 (set by CTL, by the encoder's signalling
//!   round-up, or by a signalled packet's TOC in the decoder): `quant_all_bands` sets
//!   `extra_bands = end == NB_QEXT_BANDS || end == 2`, meant to detect the QEXT pass, so such a
//!   main pass runs `cubic_quant_partition`, whose `/(N-1)` divides by zero on a 1-bin band
//!   (custom modes, LM 0) - upstream undefined behaviour; the Opus encoder / decoder never use
//!   these end bands;
//! * modes whose MDCT size kiss_fft cannot factor (e.g. 8000 / 44, radix 11): the port returns
//!   `AllocFail`, but C's `opus_fft_alloc_twiddles` failure path calls `opus_fft_free` on a
//!   `malloc`ed state whose `bitrev` / twiddle pointers are uninitialized (upstream bug; a crash
//!   under ASan), so C is not called;
//! * encoding into a 2-byte buffer with signalling on: C takes the TOC byte, then (CBR with a
//!   bitrate set) `nbCompressedBytes = IMAX(2, IMIN(1, ...))` = 2 and `ec_enc_init` gets 2 bytes
//!   at `compressed + 1`, one past the caller's buffer (upstream out-of-bounds write; the port's
//!   debug assertion in `EcEnc::with_size` fires);
//! * `OPUS_SET_QEXT(1)` on a mode without a QEXT band layout (96000 / 120): C
//!   `celt_assert(0)`s in `compute_qext_mode`, then uses the main bands;
//! * non-finite float input (NaN reaches `transient_analysis`, which `celt_assert`s it is not
//!   NaN; the Opus-level encoders are fuzzed with NaN input by `differential_encode`): raw float
//!   samples are made finite and limited to +-65536 (the input-clipping level);
//! * with `qext`, modes whose short MDCT is 90 samples at 48 kHz (48000 / 720, 96000 / 1440 and
//!   their shorter frame sizes; skipped entirely, since any packet may carry QEXT data): their
//!   QEXT layer uses `qext_eBands_180`, whose last bands (6 bins) are narrower than the earlier
//!   ones (8 bins). When decoding, `quant_all_bands` aliases `lowband_scratch` to
//!   `X_ + M*eBands[effEBands-1]` (`bands.c`) and copies a whole (wider) band there, writing
//!   `2*M` values past the end of `X` (into `Y`, or past the buffer for mono) - an upstream
//!   out-of-bounds write; the port stops with a slice-bounds panic. Opus' own modes
//!   (120-sample short MDCT, `qext_eBands_240` of equal widths) are not affected;
//! * with `qext`, modes with a 240-sample overlap (240-sample short MDCT) at a rate other than
//!   96 kHz (e.g. 88200 / 960): `comb_filter` (`celt.c`) dispatches on `overlap == 240` to
//!   `comb_filter_qext`, which reads `2 * COMBFILTER_MAXPERIOD` samples of history, but the
//!   pre/post-filter memories only hold `COMBFILTER_MAXPERIOD` (`qext_scale` is 2 only for
//!   96 kHz modes) - an upstream out-of-bounds read; the port panics on the negative index;
//! * with `qext`, custom modes with frames above 1024 samples (QEXT allows up to 2048) other
//!   than the 96 kHz QEXT layouts: the decoder history is `DECODE_BUFFER_SIZE` (2048) with
//!   `MAX_PERIOD` (1024) pitch lags (`qext_scale` 1), so `celt_decode_lost` reads
//!   `decode_mem[DECODE_BUFFER_SIZE - MAX_PERIOD - N + ...]` before the buffer (upstream
//!   out-of-bounds read; the port's index underflows);
//! * fixed-point: `OPUS_SET_LFE(1)` on a stereo encoder (overflow in `stereo_itheta`, see
//!   `opusorus_fuzz::enc::avoid_known_ub`).

#![no_main]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "fuzz harness: a failed check must panic so libFuzzer records the input"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "byte-level input decoding deliberately truncates and reinterprets"
)]

use std::borrow::Cow;

use libfuzzer_sys::fuzz_target;
use opusorus::celt::celt::{
    CELT_GET_AND_CLEAR_ERROR_REQUEST, CELT_SET_CHANNELS_REQUEST, CELT_SET_END_BAND_REQUEST,
    CELT_SET_INPUT_CLIPPING_REQUEST, CELT_SET_PREDICTION_REQUEST, CELT_SET_SIGNALLING_REQUEST,
    CELT_SET_START_BAND_REQUEST, OPUS_SET_LFE_REQUEST, from_opus,
};
use opusorus::celt::celt_decoder::CustomDecoder;
use opusorus::celt::celt_encoder::CeltEncoder;
use opusorus::celt::modes::opus_custom_mode_create_custom;
use opusorus::celt::static_modes::CeltMode;
use opusorus::decoder::{
    OPUS_GET_COMPLEXITY_REQUEST, OPUS_GET_FINAL_RANGE_REQUEST,
    OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST, OPUS_GET_PITCH_REQUEST, OPUS_RESET_STATE,
    OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
};
use opusorus::encoder::request::{
    OPUS_GET_LOOKAHEAD_REQUEST, OPUS_GET_LSB_DEPTH_REQUEST, OPUS_GET_QEXT_REQUEST,
    OPUS_SET_BITRATE_REQUEST, OPUS_SET_LSB_DEPTH_REQUEST, OPUS_SET_PACKET_LOSS_PERC_REQUEST,
    OPUS_SET_QEXT_REQUEST, OPUS_SET_VBR_CONSTRAINT_REQUEST, OPUS_SET_VBR_REQUEST,
};
use opusorus::{Error, Result};
use opusorus_fuzz::custom::{DEC_SETS, ENC_SETS, FRAME, FS, FS_ODD, RAW_FRAME, UNIMPLEMENTED};
use opusorus_fuzz::dec::read_ctl;
use opusorus_fuzz::enc::{Synth, raw_i24};
use opusorus_fuzz::{Reader, assert_f32_bits_eq, assert_slice_eq};
use opusorus_oracle::celt_decoder::CeltDec;
use opusorus_oracle::celt_encoder::CeltEnc;

/// Encoder GET requests compared after every operation.
const ENC_GETS: [i32; 4] = [
    OPUS_GET_LSB_DEPTH_REQUEST,
    OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
    OPUS_GET_FINAL_RANGE_REQUEST,
    OPUS_GET_QEXT_REQUEST,
];

/// Decoder GET requests.
const DEC_GETS: [i32; 6] = [
    OPUS_GET_COMPLEXITY_REQUEST,
    CELT_GET_AND_CLEAR_ERROR_REQUEST,
    OPUS_GET_LOOKAHEAD_REQUEST,
    OPUS_GET_PITCH_REQUEST,
    OPUS_GET_FINAL_RANGE_REQUEST,
    OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
];

/// A Rust decode result in the C convention.
const fn ret(r: Result<i32>) -> i32 {
    match r {
        Ok(n) => n,
        Err(e) => e.code(),
    }
}

const fn rc(r: Result<()>) -> i32 {
    match r {
        Ok(()) => 0,
        Err(e) => e.code(),
    }
}

/// `opus_custom_encoder_ctl` for an int-valued request on the Rust encoder.
fn rust_enc_ctl(e: &mut CeltEncoder, req: i32, v: i32) -> i32 {
    match req {
        OPUS_SET_COMPLEXITY_REQUEST => rc(e.set_complexity(v)),
        OPUS_SET_BITRATE_REQUEST => rc(e.set_bitrate(v)),
        OPUS_SET_VBR_REQUEST => {
            e.set_vbr(v);
            0
        }
        OPUS_SET_VBR_CONSTRAINT_REQUEST => {
            e.set_vbr_constraint(v);
            0
        }
        CELT_SET_PREDICTION_REQUEST => rc(e.set_prediction(v)),
        OPUS_SET_PACKET_LOSS_PERC_REQUEST => rc(e.set_packet_loss_perc(v)),
        CELT_SET_CHANNELS_REQUEST => rc(e.set_channels(v)),
        OPUS_SET_LSB_DEPTH_REQUEST => rc(e.set_lsb_depth(v)),
        OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST => rc(e.set_phase_inversion_disabled(v)),
        CELT_SET_SIGNALLING_REQUEST => {
            e.set_signalling(v);
            0
        }
        CELT_SET_INPUT_CLIPPING_REQUEST => {
            e.set_input_clipping(v);
            0
        }
        OPUS_SET_LFE_REQUEST => {
            e.set_lfe(v);
            0
        }
        #[cfg(feature = "qext")]
        OPUS_SET_QEXT_REQUEST => rc(e.set_qext(v)),
        CELT_SET_START_BAND_REQUEST => rc(e.set_start_band(v)),
        CELT_SET_END_BAND_REQUEST => rc(e.set_end_band(v)),
        OPUS_RESET_STATE => {
            e.reset();
            0
        }
        _ => UNIMPLEMENTED,
    }
}

/// A GET request on the Rust encoder, in the `(ret, value)` form of the C wrapper.
const fn rust_enc_get(e: &CeltEncoder, req: i32) -> (i32, i32) {
    match req {
        OPUS_GET_LSB_DEPTH_REQUEST => (0, e.lsb_depth()),
        OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST => (0, e.phase_inversion_disabled()),
        OPUS_GET_FINAL_RANGE_REQUEST => (0, e.final_range() as i32),
        #[cfg(feature = "qext")]
        OPUS_GET_QEXT_REQUEST => (0, e.qext()),
        _ => (UNIMPLEMENTED, 0),
    }
}

/// Whether `quant_all_bands` with bands `start..end` keeps `special_hybrid_folding` (called at
/// band `start + 1` when `start > 0`) within C's `_norm` buffer: it copies `n2 - n1` values
/// from `norm[2*n1 - n2]` to `norm[n1]` (`n1`, `n2`: widths of bands `start`, `start + 1`), and
/// `_norm` holds `eBands[nbEBands-1] - eBands[start]` values per channel (all scaled by `M`).
fn hybrid_folding_in_bounds(m: &CeltMode, start: i32, end: i32) -> bool {
    if start == 0 || end < start + 2 {
        return true;
    }
    let eb = |i: i32| i32::from(m.e_bands[i as usize]);
    let n1 = eb(start + 1) - eb(start);
    let n2 = eb(start + 2) - eb(start + 1);
    n1 <= n2 && n2 <= 2 * n1 && n2 <= eb(m.nb_ebands - 1) - eb(start)
}

/// Whether a main-band `quant_all_bands` pass may use end band `end`. With QEXT, C tells the
/// QEXT pass apart by `end == NB_QEXT_BANDS || end == 2` (`bands.c`), so a main pass ending
/// there uses cubic quantization, which divides by `N - 1` = 0 on a 1-bin band (see the header).
const fn main_end_ok(end: i32) -> bool {
    !cfg!(feature = "qext") || (end != 2 && end != 14)
}

/// Whether the mode has a QEXT band layout (`compute_qext_mode`).
#[cfg(feature = "qext")]
const fn qext_supported(m: &CeltMode) -> bool {
    opusorus::celt::modes::qext_mode_supported(m)
}

/// Whether the mode has a QEXT band layout (without `qext`, `OPUS_SET_QEXT` is rejected).
#[cfg(not(feature = "qext"))]
const fn qext_supported(_: &CeltMode) -> bool {
    true
}

/// The `frame_size` argument for selector `fsel`.
fn frame_size(fsel: u8, mode_frame: i32) -> i32 {
    if fsel < 0x80 {
        (mode_frame >> (fsel & 3)).max(1)
    } else {
        RAW_FRAME[usize::from(fsel & 0x7F) % RAW_FRAME.len()]
    }
}

/// Output format from two flag bits.
#[derive(Debug, Clone, Copy)]
enum Fmt {
    I16,
    I24,
    F32,
}

const fn fmt(bits: u8) -> Fmt {
    match bits & 3 {
        0 => Fmt::I16,
        1 => Fmt::I24,
        _ => Fmt::F32,
    }
}

/// The encoder / decoder pairs plus the harness-side view of the settings that gate C UB.
struct Pairs<'m> {
    mode: &'m CeltMode,
    mode_frame: i32,
    re: Option<(CeltEncoder, CeltEnc)>,
    enc_ch: i32,
    /// `CELT_SET_CHANNELS` of the encoder (`stream_channels`).
    enc_stream_ch: i32,
    enc_signalling: bool,
    enc_qext: bool,
    enc_band: (i32, i32),
    rd: Option<(CustomDecoder<'m>, CeltDec)>,
    dec_ch: usize,
    dec_band: (i32, i32),
    dec_signalling: bool,
    synth: Synth,
    i: usize,
}

impl Pairs<'_> {
    #[track_caller]
    fn check(&mut self, what: &str) {
        if let Some((r, c)) = &mut self.re {
            for req in ENC_GETS {
                assert_eq!(
                    rust_enc_get(r, req),
                    c.ctl_get(req),
                    "{what}: enc get {req}"
                );
            }
        }
        if let Some((r, c)) = &mut self.rd {
            for req in DEC_GETS {
                let rr = match r.st.ctl_get(req) {
                    Ok(v) => (0, v),
                    Err(e) => (e.code(), 0),
                };
                assert_eq!(rr, c.ctl_get(req), "{what}: dec get {req}");
            }
        }
    }

    /// Decodes `data` (`None`: PLC) on both decoders.
    #[track_caller]
    fn decode(&mut self, data: Option<&[u8]>, n: i32, f: Fmt, what: &str) {
        // With signalling, the packet's TOC sets the decoder's end band (C
        // `st->end = IMAX(1, effEBands - 2*(data0>>5))`, after `fromOpus` with QEXT).
        if let Some(&b0) = data.and_then(<[u8]>::first)
            && self.dec_signalling
        {
            let data0 = if cfg!(feature = "qext") {
                from_opus(b0)
            } else {
                i32::from(b0)
            };
            if data0 >= 0 && !main_end_ok((self.mode.eff_ebands - 2 * (data0 >> 5)).max(1)) {
                return;
            }
        }
        let Some((r, c)) = &mut self.rd else {
            return;
        };
        let len = data.map_or(0, |d| d.len() as i32);
        let buf = n.max(1) as usize * self.dec_ch + 3;
        match f {
            Fmt::I16 => {
                let mut a = vec![0x5A5A_i16; buf];
                let mut b = a.clone();
                let rr = ret(r.opus_custom_decode(data, len, &mut a, n));
                let cr = c.custom_decode(data, len, &mut b, n);
                assert_eq!(rr, cr, "{what}: decode return");
                assert_slice_eq(what, &a, &b);
            }
            Fmt::I24 => {
                let mut a = vec![0x5A5A_5A5A_i32; buf];
                let mut b = a.clone();
                let rr = ret(r.opus_custom_decode24(data, len, &mut a, n));
                let cr = c.custom_decode24(data, len, &mut b, n);
                assert_eq!(rr, cr, "{what}: decode24 return");
                assert_slice_eq(what, &a, &b);
            }
            Fmt::F32 => {
                let mut a = vec![1234.5_f32; buf];
                let mut b = a.clone();
                let rr = ret(r.opus_custom_decode_float(data, len, &mut a, n));
                let cr = c.custom_decode_float(data, len, &mut b, n);
                assert_eq!(rr, cr, "{what}: decode_float return");
                assert_f32_bits_eq(what, &a, &b);
            }
        }
    }

    fn encode(&mut self, r: &mut Reader<'_>, what: &str) {
        let flags = r.u8();
        let n = frame_size(r.u8(), self.mode_frame);
        let nb = i32::from(r.u16() % 4002) - 1;
        let (kind, amp, param) = (r.u8(), r.u8(), r.u16());
        let ch = self.enc_ch.clamp(1, 2) as usize;
        let total = n as usize * ch;
        let raw = if kind & 0x80 != 0 { r.blob() } else { &[] };
        if nb == 2 && self.enc_signalling {
            // C writes one byte past a 2-byte buffer here (see the header).
            return;
        }
        if self.enc_stream_ch == 2 && (self.enc_band.1 < 13 || self.mode.eff_ebands < 13) {
            // `stereo_analysis` would read uninitialized bands (see the header).
            return;
        }
        let Some((re, ce)) = &mut self.re else {
            return;
        };
        let x = if kind & 0x80 != 0 {
            let s: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect();
            if s.is_empty() {
                vec![0.0; total]
            } else {
                s.iter().copied().cycle().take(total).collect()
            }
        } else {
            self.synth
                .fill(kind, amp, param, n as usize, ch, self.mode.fs)
        };
        // The end band this call encodes with (signalling rounds it up and stores it, once the
        // byte count and frame size are accepted).
        let valid_lm = (0..=self.mode.max_lm).any(|lm| self.mode.short_mdct_size << lm == n);
        let eff = self.mode.eff_ebands;
        let end = if self.enc_signalling {
            (eff - ((eff - self.enc_band.1) >> 1)).max(1)
        } else {
            self.enc_band.1
        };
        if !main_end_ok(end) {
            return;
        }
        if self.enc_signalling && nb >= 2 && valid_lm {
            self.enc_band.1 = end;
        }
        let mut a = vec![0xA5_u8; nb.max(0) as usize];
        let mut b = a.clone();
        let (rr, cr) = match fmt(flags) {
            Fmt::I16 => {
                let p: Vec<i16> = x
                    .iter()
                    .map(|&v| (v * 32768.0).clamp(-32768.0, 32767.0) as i16)
                    .collect();
                (
                    re.opus_custom_encode(&p, n, &mut a, nb),
                    ce.custom_encode(&p, ch, n, &mut b, nb),
                )
            }
            Fmt::I24 => {
                // Raw input keeps its 32-bit pattern (out-of-range 24-bit values) in float
                // builds, 24 bits in fixed-point builds (`raw_i24`).
                let p: Vec<i32> = if kind & 0x80 != 0 {
                    x.iter().map(|v| raw_i24(v.to_bits() as i32)).collect()
                } else {
                    x.iter().map(|&v| (v * 8_388_608.0) as i32).collect()
                };
                (
                    re.opus_custom_encode24(&p, n, &mut a, nb),
                    ce.custom_encode24(&p, ch, n, &mut b, nb),
                )
            }
            Fmt::F32 => {
                // NaN / infinite input trips `celt_assert(!celt_isnan(...))` in C (see header).
                let x: Vec<f32> = x
                    .iter()
                    .map(|&v| {
                        if v.is_finite() {
                            v.clamp(-65536.0, 65536.0)
                        } else {
                            0.0
                        }
                    })
                    .collect();
                (
                    re.opus_custom_encode_float(&x, n, &mut a, nb),
                    ce.custom_encode_float(&x, ch, n, &mut b, nb),
                )
            }
        };
        assert_eq!(rr, cr, "{what}: encode(n {n}, nb {nb}) return");
        if rr <= 0 {
            return;
        }
        let pkt = a[..rr as usize].to_vec();
        assert_slice_eq(&format!("{what}: packet"), &pkt, &b[..rr as usize]);
        let out = fmt(flags >> 2);
        if flags & 0x10 != 0 {
            self.decode(None, n, out, &format!("{what}: PLC for the packet"));
        } else {
            self.decode(Some(&pkt), n, out, &format!("{what}: decode of the packet"));
        }
    }

    fn enc_ctl(&mut self, req: i32, mut v: i32) -> Option<(i32, i32)> {
        let (re, ce) = self.re.as_mut()?;
        match req {
            CELT_SET_CHANNELS_REQUEST if v > self.enc_ch => v = self.enc_ch,
            OPUS_SET_QEXT_REQUEST if v != 0 && !self.enc_signalling => return None,
            OPUS_SET_QEXT_REQUEST if v != 0 && !qext_supported(self.mode) => return None,
            CELT_SET_SIGNALLING_REQUEST if v == 0 && self.enc_qext => return None,
            CELT_SET_SIGNALLING_REQUEST if v != 0 && self.enc_band.0 != 0 => return None,
            OPUS_SET_LFE_REQUEST if cfg!(feature = "fixed-point") && self.enc_ch == 2 => v = 0,
            _ => {}
        }
        let rr = rust_enc_ctl(re, req, v);
        let cr = ce.ctl(req, v);
        if rr == 0 {
            match req {
                OPUS_SET_QEXT_REQUEST => self.enc_qext = v != 0,
                CELT_SET_CHANNELS_REQUEST => self.enc_stream_ch = v,
                CELT_SET_SIGNALLING_REQUEST => self.enc_signalling = v != 0,
                _ => {}
            }
        }
        Some((rr, cr))
    }

    fn bands(&mut self, which: u8, start_sel: u8, end_sel: u8) -> Option<[i32; 4]> {
        let nb = self.mode.nb_ebands;
        let eff = self.mode.eff_ebands;
        // C asserts (`validate_celt_decoder`) that the start band is 0 or 17 (hybrid).
        let start = [0, 17][usize::from(start_sel & 1)];
        if start >= nb {
            return None;
        }
        let end = [eff, nb, 13, 17, 19, 21, 1, eff - 1][usize::from(end_sel & 7)].clamp(1, nb);
        if start >= end || !hybrid_folding_in_bounds(self.mode, start, end) {
            return None;
        }
        if which & 1 == 0 {
            // Encoder: no bands beyond effEBands (C reads past `X`).
            // Start band 17 only with signalling off: signalling rounds `end` up to
            // `effEBands - (effEBands - end) / 2`, which can bring back the unsafe folding.
            // End bands below 3 make `dynalloc_analysis` read `bandLogE3[end - 3]`.
            if end > eff || (start != 0 && self.enc_signalling) || end < 3 {
                return None;
            }
            let (re, ce) = self.re.as_mut()?;
            // Order the two calls so start < end holds in between as well.
            let reqs = if start >= self.enc_band.1 {
                [
                    (CELT_SET_END_BAND_REQUEST, end),
                    (CELT_SET_START_BAND_REQUEST, start),
                ]
            } else {
                [
                    (CELT_SET_START_BAND_REQUEST, start),
                    (CELT_SET_END_BAND_REQUEST, end),
                ]
            };
            let mut out = [0; 4];
            for (k, (req, v)) in reqs.into_iter().enumerate() {
                out[2 * k] = rust_enc_ctl(re, req, v);
                out[2 * k + 1] = ce.ctl(req, v);
            }
            self.enc_band = (start, end);
            Some(out)
        } else {
            // Decoder: no bands beyond effEBands (past the MDCT size), and start band 17 only
            // with signalling off (a signalled packet sets `end` from its TOC, possibly <= 17).
            if end > eff || (start != 0 && self.dec_signalling) || !main_end_ok(end) {
                return None;
            }
            let (rd, cd) = self.rd.as_mut()?;
            let reqs = if start >= self.dec_band.1 {
                [
                    (CELT_SET_END_BAND_REQUEST, end),
                    (CELT_SET_START_BAND_REQUEST, start),
                ]
            } else {
                [
                    (CELT_SET_START_BAND_REQUEST, start),
                    (CELT_SET_END_BAND_REQUEST, end),
                ]
            };
            let mut out = [0; 4];
            for (k, (req, v)) in reqs.into_iter().enumerate() {
                out[2 * k] = rc(rd.st.ctl_set(req, v));
                out[2 * k + 1] = cd.ctl_set(req, v);
            }
            self.dec_band = (start, end);
            Some(out)
        }
    }

    fn step(&mut self, r: &mut Reader<'_>) {
        let what = format!("op {}", self.i);
        self.i += 1;
        match r.u8() % 8 {
            0..=2 => self.encode(r, &what),
            t @ (3 | 4) => {
                let flags = r.u8();
                let n = frame_size(r.u8(), self.mode_frame);
                let data = if t == 3 { Some(r.blob()) } else { None };
                // With signalling, C reads `data[0]` even for an empty packet.
                if data.is_some_and(<[u8]>::is_empty) && self.dec_signalling {
                    return;
                }
                self.decode(data, n, fmt(flags >> 2), &what);
            }
            5 => {
                let (req, v) = read_ctl(r, &ENC_SETS);
                if let Some((rr, cr)) = self.enc_ctl(req, v) {
                    assert_eq!(rr, cr, "{what}: enc ctl({req}, {v})");
                }
            }
            6 => {
                let (req, v) = read_ctl(r, &DEC_SETS);
                let signalling = req == CELT_SET_SIGNALLING_REQUEST;
                if signalling && v != 0 && self.dec_band.0 != 0 {
                    return;
                }
                if let Some((rd, cd)) = &mut self.rd {
                    let rr = rc(rd.st.ctl_set(req, v));
                    assert_eq!(rr, cd.ctl_set(req, v), "{what}: dec ctl({req}, {v})");
                    if signalling {
                        self.dec_signalling = v != 0;
                    }
                }
            }
            _ => {
                let (w, s, e) = (r.u8(), r.u8(), r.u8());
                if let Some(o) = self.bands(w, s, e) {
                    assert_eq!([o[0], o[2]], [o[1], o[3]], "{what}: bands");
                }
            }
        }
        self.check(&what);
    }
}

fuzz_target!(|data: &[u8]| {
    let mut r = Reader::new(data);
    let fs_sel = r.u8();
    let fs = if fs_sel < 0xF0 {
        FS[usize::from(fs_sel) % FS.len()]
    } else {
        FS_ODD[usize::from(fs_sel & 3)]
    };
    let frame_sel = r.u8();
    let frame = if frame_sel < 0xF0 {
        FRAME[usize::from(frame_sel) % FRAME.len()]
    } else {
        2 * i32::from(frame_sel & 15) + 41
    };
    let chb = r.u8();
    let enc_ch = if chb & 4 != 0 {
        3
    } else {
        1 + i32::from(chb & 1)
    };
    let dec_ch = 1 + i32::from((chb >> 1) & 1);

    // Mode creation: the C wrappers create the mode themselves; compare success.
    let rmode: Result<Cow<'static, CeltMode>> = opus_custom_mode_create_custom(fs, frame);
    if matches!(rmode, Err(Error::AllocFail)) {
        // An FFT size kiss_fft cannot factor: C's failure path frees uninitialized pointers.
        return;
    }
    let cdec = CeltDec::new_custom(fs, frame, dec_ch);
    let cenc = CeltEnc::new_custom(fs, frame, enc_ch);
    let Ok(mode) = rmode else {
        let e = rmode.err().map(Error::code);
        assert_eq!(
            cdec.err(),
            e,
            "mode ({fs}, {frame}): C accepts, Rust rejects"
        );
        return;
    };
    let cdec = cdec.expect("C accepts the mode Rust accepts");
    if cfg!(feature = "qext")
        && (mode.short_mdct_size * 48000 == 90 * mode.fs
            || (mode.overlap == 240 && mode.fs != 96000))
    {
        // Upstream out-of-bounds accesses with QEXT compiled in, see the header.
        return;
    }
    let qext_scale = if mode.fs == 96000 && matches!(mode.short_mdct_size, 240 | 180) {
        2
    } else {
        1
    };
    if cfg!(feature = "qext") && (mode.short_mdct_size << mode.max_lm) > 1024 * qext_scale {
        // Frames longer than the decoder history supports, see the header.
        return;
    }
    let rdec = CustomDecoder::opus_custom_decoder_create(&mode, dec_ch).expect("valid channels");
    let re = match (
        CeltEncoder::opus_custom_encoder_create(mode.clone(), enc_ch),
        cenc,
    ) {
        (Ok(a), Ok(b)) => Some((a, b)),
        (Err(a), Err(b)) => {
            assert_eq!(a.code(), b, "encoder create error");
            None
        }
        (a, b) => panic!("encoder create mismatch: {:?} vs {:?}", a.err(), b.err()),
    };
    let eff = mode.eff_ebands;
    let mut p = Pairs {
        mode: &mode,
        mode_frame: frame,
        re,
        enc_ch,
        enc_stream_ch: enc_ch,
        enc_signalling: true,
        enc_qext: false,
        enc_band: (0, eff),
        rd: Some((rdec, cdec)),
        dec_ch: dec_ch as usize,
        dec_band: (0, eff),
        dec_signalling: true,
        synth: Synth::new(),
        i: 0,
    };
    p.check("create");
    while !r.is_empty() {
        p.step(&mut r);
    }
});
