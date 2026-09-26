//! Encoder operation stream: configuration, CTL changes and synthesized PCM frames.
//!
//! Input format:
//!
//! ```text
//! header:  fs_sel:u8 ch_sel:u8 app_sel:u8
//! repeated until the input ends:
//! tag:u8   tag % 8:
//!   0..=4, 7  frame  flags:u8 fsel:u8 max_bytes:u16 kind:u8 amp:u8 param:u16 [raw samples]
//!   5         ctl    sel:u8 vsel:u8 [value:i32 if vsel >= 128]
//!   6         reset
//! flags: bits 0-1 input format (0 i16, 1 i24, 2/3 float).
//! fsel:  < 0xF0: one of the nine valid durations (2.5..120 ms); else an odd raw value.
//! kind:  signal generator, see [`Synth::fill`]; raw kinds read `len:u16 bytes[len]`.
//! ```

use crate::dec::read_ctl;
use crate::{Reader, Rng, Writer, code};
use opusorus::Encoder;
use opusorus::encoder::request::{
    OPUS_GET_APPLICATION_REQUEST, OPUS_GET_BANDWIDTH_REQUEST, OPUS_GET_BITRATE_REQUEST,
    OPUS_GET_COMPLEXITY_REQUEST, OPUS_GET_DRED_DURATION_REQUEST, OPUS_GET_DTX_REQUEST,
    OPUS_GET_EXPERT_FRAME_DURATION_REQUEST, OPUS_GET_FINAL_RANGE_REQUEST,
    OPUS_GET_FORCE_CHANNELS_REQUEST, OPUS_GET_IN_DTX_REQUEST, OPUS_GET_INBAND_FEC_REQUEST,
    OPUS_GET_LOOKAHEAD_REQUEST, OPUS_GET_LSB_DEPTH_REQUEST, OPUS_GET_MAX_BANDWIDTH_REQUEST,
    OPUS_GET_PACKET_LOSS_PERC_REQUEST, OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
    OPUS_GET_PREDICTION_DISABLED_REQUEST, OPUS_GET_QEXT_REQUEST, OPUS_GET_SAMPLE_RATE_REQUEST,
    OPUS_GET_SIGNAL_REQUEST, OPUS_GET_VBR_CONSTRAINT_REQUEST, OPUS_GET_VBR_REQUEST,
    OPUS_GET_VOICE_RATIO_REQUEST, OPUS_RESET_STATE, OPUS_SET_APPLICATION_REQUEST,
    OPUS_SET_BANDWIDTH_REQUEST, OPUS_SET_BITRATE_REQUEST, OPUS_SET_COMPLEXITY_REQUEST,
    OPUS_SET_DRED_DURATION_REQUEST, OPUS_SET_DTX_REQUEST, OPUS_SET_EXPERT_FRAME_DURATION_REQUEST,
    OPUS_SET_FORCE_CHANNELS_REQUEST, OPUS_SET_FORCE_MODE_REQUEST, OPUS_SET_INBAND_FEC_REQUEST,
    OPUS_SET_LFE_REQUEST, OPUS_SET_LSB_DEPTH_REQUEST, OPUS_SET_MAX_BANDWIDTH_REQUEST,
    OPUS_SET_PACKET_LOSS_PERC_REQUEST, OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
    OPUS_SET_PREDICTION_DISABLED_REQUEST, OPUS_SET_QEXT_REQUEST, OPUS_SET_SIGNAL_REQUEST,
    OPUS_SET_VBR_CONSTRAINT_REQUEST, OPUS_SET_VBR_REQUEST, OPUS_SET_VOICE_RATIO_REQUEST,
};
use opusorus_oracle::api as capi;

const AUTO: i32 = -1000;

/// Applications (plus one invalid value).
pub const APPS: [i32; 6] = [2048, 2049, 2051, 2052, 2053, 2050];

/// SET requests the harness issues, with the values seeds pick from (`vsel < 128`).
pub const ENC_SETS: [(i32, &[i32]); 23] = [
    (
        OPUS_SET_BITRATE_REQUEST,
        &[
            AUTO, -1, 500, 6000, 8000, 12000, 16000, 24000, 32000, 48000, 64000, 96000, 128000,
            256000, 510000, 1_000_000, 0, 499, -5,
        ],
    ),
    (
        OPUS_SET_MAX_BANDWIDTH_REQUEST,
        &[1101, 1102, 1103, 1104, 1105, 1100, 1106],
    ),
    (
        OPUS_SET_BANDWIDTH_REQUEST,
        &[AUTO, 1101, 1102, 1103, 1104, 1105, 1106],
    ),
    (OPUS_SET_VBR_REQUEST, &[0, 1, 2]),
    (OPUS_SET_VBR_CONSTRAINT_REQUEST, &[0, 1, -1]),
    (
        OPUS_SET_COMPLEXITY_REQUEST,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1],
    ),
    (OPUS_SET_INBAND_FEC_REQUEST, &[0, 1, 2, 3, -1]),
    (
        OPUS_SET_PACKET_LOSS_PERC_REQUEST,
        &[0, 1, 5, 10, 20, 50, 100, 101, -1],
    ),
    (OPUS_SET_DTX_REQUEST, &[0, 1, 2]),
    (OPUS_SET_FORCE_CHANNELS_REQUEST, &[AUTO, 1, 2, 0, 3]),
    (OPUS_SET_SIGNAL_REQUEST, &[AUTO, 3001, 3002, 3000]),
    (OPUS_SET_LSB_DEPTH_REQUEST, &[8, 16, 24, 7, 25, 12]),
    (
        OPUS_SET_EXPERT_FRAME_DURATION_REQUEST,
        &[
            5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 4999, 5010,
        ],
    ),
    (OPUS_SET_PREDICTION_DISABLED_REQUEST, &[0, 1, 2]),
    (OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, &[0, 1, 2]),
    (OPUS_SET_APPLICATION_REQUEST, &APPS),
    (
        OPUS_SET_FORCE_MODE_REQUEST,
        &[AUTO, 1000, 1001, 1002, 999, 1003],
    ),
    (OPUS_SET_VOICE_RATIO_REQUEST, &[-1, 0, 50, 100, 101, -2]),
    (OPUS_SET_LFE_REQUEST, &[0, 1]),
    (OPUS_SET_QEXT_REQUEST, &[0, 1, 2]),
    (OPUS_SET_DRED_DURATION_REQUEST, &[0, 10, 100]),
    (OPUS_RESET_STATE, &[0]),
    // Unknown request: rejected identically (OPUS_UNIMPLEMENTED).
    (9999, &[1]),
];

/// Index of a request in [`ENC_SETS`] (for the seed generator).
#[must_use]
pub fn ctl_sel(req: i32) -> usize {
    ENC_SETS
        .iter()
        .position(|&(r, _)| r == req)
        .expect("known request")
}

/// Every GET request of `opus_encoder_ctl` taking an `opus_int32*` (plus rejected ones).
pub const ENC_GETS: [i32; 25] = [
    OPUS_GET_APPLICATION_REQUEST,
    OPUS_GET_BITRATE_REQUEST,
    OPUS_GET_FORCE_CHANNELS_REQUEST,
    OPUS_GET_MAX_BANDWIDTH_REQUEST,
    OPUS_GET_BANDWIDTH_REQUEST,
    OPUS_GET_DTX_REQUEST,
    OPUS_GET_COMPLEXITY_REQUEST,
    OPUS_GET_INBAND_FEC_REQUEST,
    OPUS_GET_PACKET_LOSS_PERC_REQUEST,
    OPUS_GET_VBR_REQUEST,
    OPUS_GET_VOICE_RATIO_REQUEST,
    OPUS_GET_VBR_CONSTRAINT_REQUEST,
    OPUS_GET_SIGNAL_REQUEST,
    OPUS_GET_LOOKAHEAD_REQUEST,
    OPUS_GET_SAMPLE_RATE_REQUEST,
    OPUS_GET_FINAL_RANGE_REQUEST,
    OPUS_GET_LSB_DEPTH_REQUEST,
    OPUS_GET_EXPERT_FRAME_DURATION_REQUEST,
    OPUS_GET_PREDICTION_DISABLED_REQUEST,
    OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
    OPUS_GET_IN_DTX_REQUEST,
    OPUS_GET_QEXT_REQUEST,
    OPUS_GET_DRED_DURATION_REQUEST,
    4033,
    12345,
];

/// Tag values for the [`Writer`] helpers.
pub mod tag {
    /// Encode one frame.
    pub const FRAME: u8 = 0;
    /// A SET CTL.
    pub const CTL: u8 = 5;
    /// `OPUS_RESET_STATE`.
    pub const RESET: u8 = 6;
}

/// Encoder configuration from the header.
#[derive(Debug, Clone, Copy)]
pub struct EncHeader {
    /// Sampling rate.
    pub fs: i32,
    /// Channels (1 or 2).
    pub channels: i32,
    /// Application (may be invalid).
    pub app: i32,
}

/// Reads the header.
pub fn read_header(r: &mut Reader<'_>) -> EncHeader {
    let fs = crate::pick_fs(r.u8());
    let channels = 1 + i32::from(r.u8() & 1);
    let app = APPS[r.u8() as usize % APPS.len()];
    EncHeader { fs, channels, app }
}

/// Writes a header.
pub fn write_header(w: &mut Writer, fs: i32, channels: i32, app: i32) {
    let a = APPS.iter().position(|&x| x == app).expect("known app") as u8;
    w.u8(crate::fs_sel(fs)).u8((channels - 1) as u8).u8(a);
}

/// Input sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InFmt {
    /// `opus_encode`.
    I16,
    /// `opus_encode24`.
    I24,
    /// `opus_encode_float`.
    F32,
}

/// One frame's input in every format (the harness converts once).
#[derive(Debug, Clone)]
pub enum Pcm {
    /// 16-bit.
    I16(Vec<i16>),
    /// 24-bit in `i32`.
    I24(Vec<i32>),
    /// Float.
    F32(Vec<f32>),
}

/// One encoder operation.
#[derive(Debug, Clone)]
pub enum EncOp {
    /// Encode a frame.
    Frame {
        /// The input (`frame_size * channels` samples).
        pcm: Pcm,
        /// The `frame_size` argument.
        frame_size: usize,
        /// Output buffer size (`max_data_bytes`).
        max_bytes: usize,
    },
    /// A SET CTL (or reset).
    Ctl {
        /// Request.
        req: i32,
        /// Value.
        value: i32,
    },
}

/// Valid frame durations in units of 0.1 ms.
const DUR_TENTH_MS: [i32; 9] = [25, 50, 100, 200, 400, 600, 800, 1000, 1200];

/// The frame size for selector `fsel` (see the module docs).
#[must_use]
pub fn frame_size(fsel: u8, fs: i32) -> usize {
    let n = if fsel < 0xF0 {
        fs / 10 * DUR_TENTH_MS[usize::from(fsel) % DUR_TENTH_MS.len()] / 1000
    } else {
        let raw = [
            0,
            1,
            7,
            fs / 400 - 1,
            fs / 400 + 1,
            fs / 50 + 1,
            fs / 25 * 3 + fs / 400,
            480,
            960,
            2880,
            5760,
            11520,
            fs / 100 * 3,
            fs / 25 * 3 * 2,
            fs / 50 * 3,
            fs / 200,
        ];
        raw[usize::from(fsel & 15)]
    };
    n.clamp(0, crate::dec::MAX_FRAME) as usize
}

/// Frame selector for `k`-th valid duration (0 = 2.5 ms ... 8 = 120 ms).
#[must_use]
pub const fn fsel_dur(k: u8) -> u8 {
    k
}

/// Continuous signal generator (phase and noise state persist across frames). Any channel
/// count; channels beyond the first pair are detuned, quieter copies.
#[derive(Debug, Clone)]
pub struct Synth {
    phase: Vec<f64>,
    rng: Rng,
    t: u64,
}

impl Default for Synth {
    fn default() -> Self {
        Self::new()
    }
}

impl Synth {
    /// Fresh state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            phase: Vec::new(),
            rng: Rng::new(0x1234_5678),
            t: 0,
        }
    }

    /// Fills `n * ch` interleaved float samples with signal `kind`:
    /// 0 silence, 1 sine (`param` Hz), 2 white noise, 3 harmonic speech-like buzz with
    /// syllable-rate envelope (f0 = 80 + `param` % 300 Hz), 4 sine + noise, 5 full-scale square
    /// wave (clipping), 6 sweep, 7 near-silence (tiny dither). Amplitude `amp / 255`.
    pub fn fill(
        &mut self,
        kind: u8,
        amp: u8,
        param: u16,
        n: usize,
        ch: usize,
        fs: i32,
    ) -> Vec<f32> {
        let a = f64::from(amp) / 255.0;
        let fsf = f64::from(fs);
        let mut out = Vec::with_capacity(n * ch);
        if self.phase.len() < ch {
            self.phase.resize(ch, 0.0);
        }
        for _ in 0..n {
            let t = self.t as f64 / fsf;
            self.t += 1;
            for c in 0..ch {
                // Channels past the first pair (multistream) get detuned, quieter copies.
                let pair = (c / 2) as f64;
                let a = a / (1.0 + pair);
                let fsf = fsf / (1.0 + 0.013 * pair);
                let v = match kind & 7 {
                    0 => 0.0,
                    1 => {
                        self.phase[c] += core::f64::consts::TAU * f64::from(param) / fsf;
                        a * self.phase[c].sin()
                    }
                    2 => a * f64::from(self.rng.f32_sym()),
                    3 => {
                        let f0 = 80.0 + f64::from(param % 300);
                        self.phase[c] += core::f64::consts::TAU * f0 / fsf;
                        let env = 0.5 * (1.0 + (t * 4.0 * core::f64::consts::TAU).sin());
                        let mut s = 0.0;
                        for h in 1..12 {
                            s += (self.phase[c] * f64::from(h)).sin() / f64::from(h);
                        }
                        a * env * s * 0.4 + 0.01 * a * f64::from(self.rng.f32_sym())
                    }
                    4 => {
                        self.phase[c] += core::f64::consts::TAU * f64::from(param) / fsf;
                        a * (0.7 * self.phase[c].sin() + 0.3 * f64::from(self.rng.f32_sym()))
                    }
                    5 => {
                        self.phase[c] += core::f64::consts::TAU * f64::from(param % 2000) / fsf;
                        if self.phase[c].sin() >= 0.0 {
                            a * 1.5
                        } else {
                            -a * 1.5
                        }
                    }
                    6 => {
                        let f = 50.0 + (t * f64::from(param)).rem_euclid(fsf / 2.0);
                        self.phase[c] += core::f64::consts::TAU * f / fsf;
                        a * self.phase[c].sin()
                    }
                    _ => 1e-5 * f64::from(self.rng.f32_sym()),
                };
                out.push(v as f32);
            }
        }
        out
    }
}

fn to_pcm(x: &[f32], fmt: InFmt) -> Pcm {
    match fmt {
        InFmt::I16 => Pcm::I16(
            x.iter()
                .map(|&v| (f64::from(v) * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
                .collect(),
        ),
        InFmt::I24 => Pcm::I24(
            x.iter()
                .map(|&v| (f64::from(v) * 8_388_608.0).round() as i32)
                .collect(),
        ),
        InFmt::F32 => Pcm::F32(x.to_vec()),
    }
}

/// Raw input: the samples come straight from the fuzz input (any bit pattern, including
/// non-finite floats and, in float builds, out-of-range 24-bit values; see [`raw_i24`]),
/// repeated to fill the frame.
fn raw_pcm(raw: &[u8], fmt: InFmt, total: usize) -> Pcm {
    match fmt {
        InFmt::I16 => {
            let s: Vec<i16> = raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| i16::from_le_bytes(b))
                .collect();
            Pcm::I16(cycle(&s, total))
        }
        InFmt::I24 => {
            let s: Vec<i32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| raw_i24(i32::from_le_bytes(b)))
                .collect();
            Pcm::I24(cycle(&s, total))
        }
        InFmt::F32 => {
            let s: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect();
            Pcm::F32(cycle(&s, total))
        }
    }
}

fn cycle<T: Copy + Default>(s: &[T], total: usize) -> Vec<T> {
    if s.is_empty() {
        return vec![T::default(); total];
    }
    s.iter().copied().cycle().take(total).collect()
}

/// Input classes that are undefined behaviour in C and panic in debug builds of the port (the
/// overflow checks cargo-fuzz enables), avoided until the port hardens them:
///
/// * fixed-point: a stereo encoder with `OPUS_SET_LFE(1)` overflows in CELT's `stereo_itheta`
///   (C wraps; docs/FIXED_POINT.md, FX4). Multistream LFE streams are mono, and
///   `opus_multistream_encoder_ctl` does not forward `OPUS_SET_LFE`, so only single-stream
///   stereo encoders are affected: their `OPUS_SET_LFE` value is forced to 0.
#[must_use]
pub const fn avoid_known_ub(req: i32, value: i32, h: &EncHeader) -> i32 {
    if cfg!(feature = "fixed-point") && req == OPUS_SET_LFE_REQUEST && h.channels == 2 {
        0
    } else {
        value
    }
}

/// A raw 24-bit input sample. Float builds take any 32-bit pattern (out-of-range values are
/// scaled like any other). Fixed-point builds keep the sample within 24 bits (sign-extending
/// its low 24 bits): larger values are undefined behaviour in C there (`INT24TORES` =
/// `SAT16(PSHR32(x, 8))` overflows near `i32::MAX` in 16-bit builds; `INT24TOSIG` = `x << 4`
/// wraps and the analysis downmix / `silk_resampler_down2_hp` sums then overflow), and
/// `opus_encode24` documents its input as 24-bit.
#[must_use]
pub const fn raw_i24(v: i32) -> i32 {
    if cfg!(feature = "fixed-point") {
        (v << 8) >> 8
    } else {
        v
    }
}

/// Reads the next operation, or `None` at the end of the input.
pub fn read_op(r: &mut Reader<'_>, h: &EncHeader, synth: &mut Synth) -> Option<EncOp> {
    if r.is_empty() {
        return None;
    }
    Some(match r.u8() % 8 {
        5 => {
            let (req, value) = read_ctl(r, &ENC_SETS);
            EncOp::Ctl {
                req,
                value: avoid_known_ub(req, value, h),
            }
        }
        6 => EncOp::Ctl {
            req: OPUS_RESET_STATE,
            value: 0,
        },
        _ => {
            let fl = r.u8();
            let fmt = match fl & 3 {
                0 => InFmt::I16,
                1 => InFmt::I24,
                _ => InFmt::F32,
            };
            let frame_size = frame_size(r.u8(), h.fs);
            let max_bytes = usize::from(r.u16() % 4001);
            let kind = r.u8();
            let amp = r.u8();
            let param = r.u16();
            let total = frame_size * h.channels as usize;
            let pcm = if kind & 0x80 != 0 {
                raw_pcm(r.blob(), fmt, total)
            } else {
                let x = synth.fill(kind, amp, param, frame_size, h.channels as usize, h.fs);
                to_pcm(&x, fmt)
            };
            EncOp::Frame {
                pcm,
                frame_size,
                max_bytes,
            }
        }
    })
}

/// Writes a synthesized frame op.
#[allow(
    clippy::too_many_arguments,
    reason = "one argument per field of the serialized record"
)]
pub fn write_frame(
    w: &mut Writer,
    fmt: InFmt,
    fsel: u8,
    max_bytes: u16,
    kind: u8,
    amp: u8,
    param: u16,
) {
    let f = match fmt {
        InFmt::I16 => 0,
        InFmt::I24 => 1,
        InFmt::F32 => 2,
    };
    w.u8(tag::FRAME)
        .u8(f)
        .u8(fsel)
        .u16(max_bytes)
        .u8(kind & 0x7F)
        .u8(amp)
        .u16(param);
}

/// Writes a CTL op picking `ENC_SETS[sel].1[vidx]`.
pub fn write_ctl(w: &mut Writer, req: i32, value: i32) {
    let sel = ctl_sel(req);
    match ENC_SETS[sel].1.iter().position(|&v| v == value) {
        Some(vidx) => w.u8(tag::CTL).u8(sel as u8).u8(vidx as u8),
        None => w.u8(tag::CTL).u8(sel as u8).u8(0x80).i32(value),
    };
}

/// Encodes with the Rust encoder (C-convention result).
pub fn rust_encode(e: &mut Encoder, pcm: &Pcm, n: usize, out: &mut [u8]) -> Result<usize, i32> {
    code(match pcm {
        Pcm::I16(x) => e.encode(x, n, out),
        Pcm::I24(x) => e.encode24(x, n, out),
        Pcm::F32(x) => e.encode_float(x, n, out),
    })
}

/// Encodes with the C encoder.
pub fn c_encode(e: &mut capi::Encoder, pcm: &Pcm, n: usize, out: &mut [u8]) -> Result<usize, i32> {
    match pcm {
        Pcm::I16(x) => e.encode(x, n, out),
        Pcm::I24(x) => e.encode24(x, n, out),
        Pcm::F32(x) => e.encode_float(x, n, out),
    }
}
