//! Differential tests for unit `opus_encoder` (src/opus_encoder.c,
//! src/opus_multistream_encoder.c, src/opus_projection_encoder.c) vs the C oracle.
//!
//! * Static helpers (TOC, high-pass / DC filters, stereo/gain fades, frame size selection,
//!   stereo width, FEC / DTX / redundancy decisions, rate helpers, surround analysis, logSum)
//!   on randomized inputs — bit-exact.
//! * Long realistic streams (speech-like, music-like, noise, silence, transitions, stereo width
//!   changes) through [`Encoder`] and the library `OpusEncoder` over a large configuration
//!   matrix (application, rate, channels, frame size, bitrate, VBR/CVBR/CBR, complexity, FEC,
//!   DTX, forced modes/bandwidths/channels, signal hints, LSB depth, prediction, phase
//!   inversion, expert frame duration, i16/i24/float input) with random CTL changes mid-stream.
//!   Every packet (bytes, length, error code), the final range and every GET CTL are compared.
//! * A private copy of the C encoder whose struct is dumped: the full Rust encoder state is
//!   compared after every frame on a subset of the streams.
//! * Multistream (families 0/1/2/255, random mappings, surround with LFE) and projection
//!   (family 3, orders 1-3) encoders: bytes, final ranges, per-stream CTLs, demixing matrices.
//! * Error paths (bad arguments, too-small buffers) return the same codes.
//!
//! `OPUSORUS_OE_DUP=1` runs every single-stream test against the dumpable C copy, which
//! reports the first diverging state field (debugging aid).

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "test drivers mirror the C shim signatures and index several arrays"
)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_const_for_fn,
    reason = "test code: failures should panic"
)]

use opusorus::encoder::request::*;
use opusorus::encoder::{self as oe, Encoder};
use opusorus::ms_encoder::{self as me, MsEncoder};
use opusorus::projection_encoder::ProjectionEncoder;
use opusorus::{Application, Bandwidth, Bitrate, Error, FrameSize, Signal};
use opusorus_conformance::{Rng, assert_bits_eq_f32, assert_slice_eq, signals};
use opusorus_oracle::opus_encoder as oc;

const APP_VOIP: i32 = 2048;
const APP_AUDIO: i32 = 2049;
const APP_LOWDELAY: i32 = 2051;
const APP_RSILK: i32 = 2052;
const APP_RCELT: i32 = 2053;
const APPS: [i32; 5] = [APP_VOIP, APP_AUDIO, APP_LOWDELAY, APP_RSILK, APP_RCELT];
const AUTO: i32 = -1000;
const BITRATE_MAX: i32 = -1;

const MODE_SILK: i32 = 1000;
const MODE_HYBRID: i32 = 1001;
const MODE_CELT: i32 = 1002;

fn code<T>(r: Result<T, Error>) -> Result<T, i32> {
    r.map_err(Error::code)
}

/// All GET requests of `opus_encoder_ctl` taking an `opus_int32*` (plus some it rejects).
const GET_REQUESTS: [i32; 25] = [
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
    4033, // OPUS_GET_PITCH (decoder only): unimplemented
    12345,
];

// ---------------------------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------------------------

#[test]
fn gen_toc_exhaustive() {
    for mode in [MODE_SILK, MODE_HYBRID, MODE_CELT] {
        for framerate in [400, 200, 100, 50, 25, 16, 12, 10, 8] {
            for bw in 1101..=1105 {
                for ch in 1..=2 {
                    assert_eq!(
                        oe::gen_toc(mode, framerate, bw, ch),
                        oc::gen_toc(mode, framerate, bw, ch),
                        "mode {mode} rate {framerate} bw {bw} ch {ch}"
                    );
                }
            }
        }
    }
}

#[test]
fn filters_and_fades() {
    let mut rng = Rng::new(0x0E01);
    for iter in 0..400 {
        let fs: i32 = [8000, 12000, 16000, 24000, 48000][iter % 5];
        let ch: i32 = 1 + (iter as i32 / 5) % 2;
        let len = fs / 400 * [1, 2, 4, 8, 24][rng.range_i32(0, 4) as usize];
        let n = (len * ch) as usize;
        let amp = [1e-4f32, 0.3, 1.0, 3.0][rng.range_i32(0, 3) as usize];
        let input: Vec<f32> = (0..n).map(|_| amp * rng.f32_sym()).collect();

        // hp_cutoff / dc_reject over several frames with state.
        let mut mem_r = [0f32; 4];
        let mut mem_c = [0f32; 4];
        for _ in 0..3 {
            let cutoff = rng.range_i32(20, 200);
            let mut out_r = vec![0f32; n];
            let mut out_c = vec![0f32; n];
            oe::hp_cutoff(&input, cutoff, &mut out_r, &mut mem_r, len, ch, fs);
            oc::hp_cutoff(&input, cutoff, &mut out_c, &mut mem_c, len, ch, fs);
            assert_bits_eq_f32("hp_cutoff", &out_r, &out_c);
            assert_bits_eq_f32("hp_cutoff mem", &mem_r, &mem_c);
            oe::dc_reject(&input, 3, &mut out_r, &mut mem_r, len, ch, fs);
            oc::dc_reject(&input, 3, &mut out_c, &mut mem_c, len, ch, fs);
            assert_bits_eq_f32("dc_reject", &out_r, &out_c);
            assert_bits_eq_f32("dc_reject mem", &mem_r, &mem_c);
        }

        // gain_fade / stereo_fade in place (48 kHz mode window).
        let window: Vec<f32> = window48();
        let g1 = rng.f32_sym().abs();
        let g2 = rng.f32_sym().abs();
        let mut br = input.clone();
        let mut bc = input.clone();
        oe::gain_fade(&mut br, g1, g2, 120, len, ch, &window, fs);
        oc::gain_fade(&mut bc, g1, g2, false, len, ch, fs);
        assert_bits_eq_f32("gain_fade", &br, &bc);
        if ch == 2 {
            let mut br = input.clone();
            let mut bc = input.clone();
            oe::stereo_fade(&mut br, g1, g2, 120, len, ch, &window, fs);
            oc::stereo_fade(&mut bc, g1, g2, false, len, ch, fs);
            assert_bits_eq_f32("stereo_fade", &br, &bc);
        }

        // compute_frame_energy
        assert_eq!(
            oe::compute_frame_energy(&input, len, ch).to_bits(),
            oc::compute_frame_energy(&input, len, ch).to_bits()
        );
    }
}

/// The 48 kHz CELT mode window (via a fresh encoder's CELT mode).
fn window48() -> Vec<f32> {
    let m = opusorus::celt::modes::opus_custom_mode_create(48000, 960).unwrap();
    m.window.to_vec()
}

#[cfg(feature = "qext")]
#[test]
fn fades_96k() {
    let mut rng = Rng::new(0x0E96);
    let m = opusorus::celt::modes::opus_custom_mode_create(96000, 1920).unwrap();
    let window = m.window.to_vec();
    for _ in 0..50 {
        let len: i32 = 1920;
        let input: Vec<f32> = (0..2 * len).map(|_| rng.f32_sym()).collect();
        let (g1, g2) = (rng.f32_sym().abs(), rng.f32_sym().abs());
        let mut br = input.clone();
        let mut bc = input.clone();
        oe::gain_fade(&mut br, g1, g2, m.overlap, len, 2, &window, 96000);
        oc::gain_fade(&mut bc, g1, g2, true, len, 2, 96000);
        assert_bits_eq_f32("gain_fade 96k", &br, &bc);
        let mut br = input.clone();
        let mut bc = input.clone();
        oe::stereo_fade(&mut br, g1, g2, m.overlap, len, 2, &window, 96000);
        oc::stereo_fade(&mut bc, g1, g2, true, len, 2, 96000);
        assert_bits_eq_f32("stereo_fade 96k", &br, &bc);
    }
}

#[test]
fn frame_size_select_matrix() {
    let mut rates = vec![8000, 12000, 16000, 24000, 48000];
    if cfg!(feature = "qext") {
        rates.push(96000);
    }
    for fs in rates {
        for app in APPS {
            for vd in 4998..=5011 {
                let mut sizes: Vec<i32> =
                    (0..=12 * fs / 100).step_by((fs / 400) as usize).collect();
                sizes.extend([-1, 0, 1, 7, fs / 400 - 1, fs / 400 + 1, 12 * fs / 100 + 1]);
                sizes.extend([i32::MAX, 1 << 24, 5_000_000]);
                for n in sizes {
                    assert_eq!(
                        oe::frame_size_select(app, n, vd, fs),
                        oc::frame_size_select(app, n, vd, fs),
                        "app {app} n {n} vd {vd} fs {fs}"
                    );
                }
            }
        }
    }
}

#[test]
fn stereo_width_streams() {
    for (k, fs) in [8000, 12000, 16000, 24000, 48000].into_iter().enumerate() {
        for ms in [25, 50, 100, 200, 400, 600] {
            let frame = fs * ms / 10000;
            let sig = signals::music_like(frame as usize * 40, 2, fs as u32, 7 + k as u64);
            let mut mem_r = oe::StereoWidthState::default();
            let mut mem_c = [0f32; 5];
            for f in 0..40 {
                let mut x = sig[f * 2 * frame as usize..(f + 1) * 2 * frame as usize].to_vec();
                // Vary the width: identical, inverted, scaled, huge.
                for i in 0..frame as usize {
                    match f % 5 {
                        1 => x[2 * i + 1] = x[2 * i],
                        2 => x[2 * i + 1] = -x[2 * i],
                        3 => x[2 * i + 1] *= 0.1,
                        4 if f == 39 => x[2 * i] = 1e6,
                        _ => {}
                    }
                }
                let r = oe::compute_stereo_width(&x, frame, fs, &mut mem_r);
                let c = oc::compute_stereo_width(&x, frame, fs, &mut mem_c);
                assert_eq!(r.to_bits(), c.to_bits(), "fs {fs} ms {ms} frame {f}");
                let m = [
                    mem_r.xx,
                    mem_r.xy,
                    mem_r.yy,
                    mem_r.smoothed_width,
                    mem_r.max_follower,
                ];
                assert_bits_eq_f32("width mem", &m, &mem_c);
            }
        }
    }
}

#[test]
fn decision_helpers() {
    let mut rng = Rng::new(0x0E02);
    for _ in 0..20000 {
        let use_fec = rng.range_i32(0, 1);
        let loss = rng.range_i32(0, 100);
        let last = rng.range_i32(0, 1);
        let mode = [MODE_SILK, MODE_HYBRID, MODE_CELT][rng.range_i32(0, 2) as usize];
        let rate = rng.range_i32(0, 80000);
        let mut bw_r = rng.range_i32(1101, 1105);
        let mut bw_c = bw_r;
        assert_eq!(
            oe::decide_fec(use_fec, loss, last, mode, &mut bw_r, rate),
            oc::decide_fec(use_fec, loss, last, mode, &mut bw_c, rate)
        );
        assert_eq!(bw_r, bw_c);

        let ch = rng.range_i32(1, 2);
        let bw = rng.range_i32(1101, 1105);
        let r = rng.range_i32(-10000, 1_000_000);
        let (f20, vbr, fec) = (
            rng.range_i32(0, 1),
            rng.range_i32(0, 1),
            rng.range_i32(0, 1),
        );
        assert_eq!(
            oe::compute_silk_rate_for_hybrid(r, bw, f20, vbr, fec, ch),
            oc::compute_silk_rate_for_hybrid(r, bw, f20, vbr, fec, ch)
        );

        let fr = [8, 10, 12, 16, 25, 50, 100, 200, 400][rng.range_i32(0, 8) as usize];
        let cx = rng.range_i32(0, 10);
        let m = [0, MODE_SILK, MODE_HYBRID, MODE_CELT][rng.range_i32(0, 3) as usize];
        let br = rng.range_i32(0, 1_500_000);
        assert_eq!(
            oe::compute_equiv_rate(br, ch, fr, vbr, m, cx, loss),
            oc::compute_equiv_rate(br, ch, fr, vbr, m, cx, loss)
        );

        let mdb = rng.range_i32(1, 1276);
        assert_eq!(
            oe::compute_redundancy_bytes(mdb, br.min(600000), fr, ch),
            oc::compute_redundancy_bytes(mdb, br.min(600000), fr, ch)
        );
    }
    // decide_dtx_mode sequences
    for fsq in [5, 10, 20, 40, 80, 120, 240] {
        let mut nr = 0;
        let mut nc = 0;
        let mut rng = Rng::new(fsq as u64);
        for _ in 0..2000 {
            let act = i32::from(rng.range_i32(0, 9) == 0);
            assert_eq!(
                oe::decide_dtx_mode(act, &mut nr, fsq),
                oc::decide_dtx_mode(act, &mut nc, fsq)
            );
            assert_eq!(nr, nc);
        }
    }
}

#[test]
fn surround_helpers() {
    let mut rng = Rng::new(0x0E03);
    for _ in 0..20000 {
        let a = rng.f32_sym() * 40.0;
        let b = rng.f32_sym() * 40.0;
        assert_eq!(
            me::log_sum(a, b).to_bits(),
            oc::log_sum(a, b).to_bits(),
            "{a} {b}"
        );
    }
    assert_eq!(
        me::log_sum(f32::NAN, 1.0).to_bits(),
        oc::log_sum(f32::NAN, 1.0).to_bits()
    );
    for ch in 0..=9 {
        let mut pos = [0i32; 8];
        me::channel_pos(ch, &mut pos);
        assert_eq!(pos, oc::channel_pos(ch));
    }
}

// ---------------------------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------------------------

/// A long test signal: random segments of speech, music, noise, silence, near-silence and
/// loud/clipped content, with stereo width changes.
fn program(total: usize, ch: usize, fs: i32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(total * ch);
    let mut k = 0u64;
    while out.len() < total * ch {
        let n = (fs as usize) * rng.range_i32(15, 90) as usize / 100;
        k += 1;
        let s = seed.wrapping_mul(31).wrapping_add(k);
        let mut seg = match rng.range_i32(0, 9) {
            0..=2 => signals::speech_like(n, ch, fs as u32, s),
            3..=4 => signals::music_like(n, ch, fs as u32, s),
            5 => signals::noise(n, ch, 0.2, s),
            6 => vec![0.0; n * ch],
            7 => signals::noise(n, ch, 2e-5, s),
            8 => signals::noise(n, ch, 1.5, s),
            _ => {
                // Tone sweep.
                let mut v = Vec::with_capacity(n * ch);
                for i in 0..n {
                    let t = i as f64 / fs as f64;
                    let x = (0.5 * (2.0 * core::f64::consts::PI * (200.0 + 3000.0 * t) * t).sin())
                        as f32;
                    for _ in 0..ch {
                        v.push(x);
                    }
                }
                v
            }
        };
        if ch == 2 {
            match rng.range_i32(0, 4) {
                0 => {
                    for i in 0..n {
                        seg[2 * i + 1] = seg[2 * i];
                    }
                }
                1 => {
                    for i in 0..n {
                        seg[2 * i + 1] = -seg[2 * i];
                    }
                }
                2 => {
                    for i in 0..n {
                        seg[2 * i + 1] *= 0.05;
                    }
                }
                _ => {}
            }
        }
        out.extend_from_slice(&seg);
    }
    out.truncate(total * ch);
    out
}

fn to_i16(x: &[f32]) -> Vec<i16> {
    signals::to_i16(x)
}

fn to_i24(x: &[f32]) -> Vec<i32> {
    x.iter()
        .map(|&v| {
            (f64::from(v) * 8_388_608.0)
                .round()
                .clamp(-16_777_216.0, 16_777_215.0) as i32
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Single-stream streams
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Input {
    I16,
    I24,
    F32,
}

/// The C side of a stream comparison: the library encoder or the dumpable private copy.
enum CEnc {
    Lib(oc::Enc),
    Dup(oc::DupEnc),
}

impl CEnc {
    fn new(fs: i32, ch: i32, app: i32, dup: bool) -> Result<Self, i32> {
        Ok(if dup {
            Self::Dup(oc::DupEnc::new(fs, ch, app)?)
        } else {
            Self::Lib(oc::Enc::new(fs, ch, app)?)
        })
    }
    fn ctl_set(&mut self, req: i32, val: i32) -> Result<(), i32> {
        match self {
            Self::Lib(e) => {
                if req == OPUS_RESET_STATE {
                    e.reset()
                } else {
                    e.ctl_set(req, val)
                }
            }
            Self::Dup(e) => e.ctl_set(req, val),
        }
    }
    fn ctl_get(&mut self, req: i32) -> Result<i32, i32> {
        match self {
            Self::Lib(e) => {
                if req == OPUS_GET_FINAL_RANGE_REQUEST {
                    e.final_range().map(|v| v as i32)
                } else {
                    e.ctl_get(req)
                }
            }
            Self::Dup(e) => e.ctl_get(req),
        }
    }
    fn encode(&mut self, input: Input, x: &[f32], n: usize, out: &mut [u8]) -> Result<usize, i32> {
        match (self, input) {
            (Self::Lib(e), Input::I16) => e.encode(&to_i16(x), n, out),
            (Self::Lib(e), Input::I24) => e.encode24(&to_i24(x), n, out),
            (Self::Lib(e), Input::F32) => e.encode_float(x, n, out),
            (Self::Dup(e), Input::I16) => e.encode(&to_i16(x), n, out),
            (Self::Dup(e), Input::I24) => e.encode24(&to_i24(x), n, out),
            (Self::Dup(e), Input::F32) => e.encode_float(x, n, out),
        }
    }
    fn dump(&self) -> Option<(Vec<u32>, Vec<f32>)> {
        match self {
            Self::Lib(_) => None,
            Self::Dup(e) => Some(e.dump()),
        }
    }
}

const DUMP_NAMES: [&str; 63] = [
    "application",
    "channels",
    "delay_compensation",
    "force_channels",
    "signal_type",
    "user_bandwidth",
    "max_bandwidth",
    "user_forced_mode",
    "voice_ratio",
    "Fs",
    "use_vbr",
    "vbr_constraint",
    "variable_duration",
    "bitrate_bps",
    "user_bitrate_bps",
    "lsb_depth",
    "encoder_buffer",
    "lfe",
    "use_dtx",
    "fec_config",
    "stream_channels",
    "hybrid_stereo_width_Q14",
    "variable_HP_smth2_Q15",
    "prev_HB_gain",
    "hp_mem[0]",
    "hp_mem[1]",
    "hp_mem[2]",
    "hp_mem[3]",
    "mode",
    "prev_mode",
    "prev_channels",
    "prev_framesize",
    "bandwidth",
    "auto_bandwidth",
    "silk_bw_switch",
    "first",
    "width.XX",
    "width.XY",
    "width.YY",
    "width.smoothed_width",
    "width.max_follower",
    "detected_bandwidth",
    "nb_no_activity_ms_Q1",
    "peak_signal_energy",
    "nonfinal_frame",
    "rangeFinal",
    "silk.bitRate",
    "silk.maxBits",
    "silk.toMono",
    "silk.LBRR_coded",
    "silk.useDTX",
    "silk.useCBR",
    "silk.stereoWidth_Q14",
    "silk.internalSampleRate",
    "silk.opusCanSwitch",
    "silk.desiredInternalSampleRate",
    "silk.maxInternalSampleRate",
    "silk.minInternalSampleRate",
    "silk.payloadSize_ms",
    "silk.nChannelsInternal",
    "silk.allowBandwidthSwitch",
    "silk.inWBmodeWithoutVariableLP",
    "silk.switchReady",
];

fn compare_dump(r: &Encoder, c: &CEnc, ctx: &str) {
    let Some((cv, cd)) = c.dump() else { return };
    let (rv, rd) = r.debug_state_dump();
    assert_eq!(rv.len(), cv.len());
    for i in 0..rv.len() {
        assert_eq!(
            rv[i], cv[i],
            "{ctx}: state field {} differs: rust {:#x} c {:#x}",
            DUMP_NAMES[i], rv[i], cv[i]
        );
    }
    assert_bits_eq_f32(&format!("{ctx}: delay_buffer"), &rd, &cd);
}

fn rust_encode(
    r: &mut Encoder,
    input: Input,
    x: &[f32],
    n: usize,
    out: &mut [u8],
) -> Result<usize, i32> {
    code(match input {
        Input::I16 => r.encode(&to_i16(x), n, out),
        Input::I24 => r.encode24(&to_i24(x), n, out),
        Input::F32 => r.encode_float(x, n, out),
    })
}

/// A random CTL (mostly valid values, some invalid).
fn random_ctl(rng: &mut Rng, fs: i32) -> (i32, i32) {
    let pick = |rng: &mut Rng, v: &[i32]| v[rng.range_i32(0, v.len() as i32 - 1) as usize];
    match rng.range_i32(0, 24) {
        0..=3 => (
            OPUS_SET_BITRATE_REQUEST,
            if rng.range_i32(0, 9) == 0 {
                pick(rng, &[AUTO, BITRATE_MAX, 0, -5, 100, 500, 501, 2_000_000])
            } else {
                pick(
                    rng,
                    &[
                        6000, 8000, 10000, 12000, 16000, 20000, 24000, 32000, 48000, 64000, 96000,
                        128000, 256000, 510000,
                    ],
                ) + rng.range_i32(-500, 500)
            },
        ),
        4 => (
            OPUS_SET_MAX_BANDWIDTH_REQUEST,
            pick(rng, &[1100, 1101, 1102, 1103, 1104, 1105, 1105, 1106]),
        ),
        5 => (
            OPUS_SET_BANDWIDTH_REQUEST,
            pick(rng, &[AUTO, AUTO, 1101, 1102, 1103, 1104, 1105, 1106]),
        ),
        6 => (OPUS_SET_VBR_REQUEST, pick(rng, &[0, 1, 1, 2])),
        7 => (OPUS_SET_VBR_CONSTRAINT_REQUEST, pick(rng, &[0, 1, -1])),
        8 => (OPUS_SET_COMPLEXITY_REQUEST, rng.range_i32(-1, 11)),
        9 => (OPUS_SET_INBAND_FEC_REQUEST, rng.range_i32(-1, 3)),
        10 => (
            OPUS_SET_PACKET_LOSS_PERC_REQUEST,
            pick(rng, &[0, 0, 1, 5, 6, 10, 20, 30, 50, 100, 101, -1]),
        ),
        11 => (OPUS_SET_DTX_REQUEST, pick(rng, &[0, 1, 1, 2])),
        12 => (
            OPUS_SET_FORCE_CHANNELS_REQUEST,
            pick(rng, &[AUTO, AUTO, 1, 2, 0, 3]),
        ),
        13 => (
            OPUS_SET_SIGNAL_REQUEST,
            pick(rng, &[AUTO, 3001, 3002, 3000]),
        ),
        14 => (OPUS_SET_LSB_DEPTH_REQUEST, rng.range_i32(7, 25)),
        15 => (
            OPUS_SET_EXPERT_FRAME_DURATION_REQUEST,
            rng.range_i32(4999, 5010),
        ),
        16 => (OPUS_SET_PREDICTION_DISABLED_REQUEST, pick(rng, &[0, 1, 2])),
        17 => (
            OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
            pick(rng, &[0, 1, 2]),
        ),
        18 => (OPUS_SET_APPLICATION_REQUEST, pick(rng, &APPS)),
        19 => (
            OPUS_SET_FORCE_MODE_REQUEST,
            pick(
                rng,
                &[AUTO, AUTO, MODE_SILK, MODE_HYBRID, MODE_CELT, 999, 1003],
            ),
        ),
        20 => (OPUS_SET_VOICE_RATIO_REQUEST, rng.range_i32(-2, 101)),
        21 => (OPUS_RESET_STATE, 0),
        22 => (OPUS_SET_LFE_REQUEST, pick(rng, &[0, 0, 0, 1])),
        23 => (
            OPUS_SET_QEXT_REQUEST,
            if fs >= 48000 {
                pick(rng, &[0, 1, 2])
            } else {
                0
            },
        ),
        _ => (9999, 1),
    }
}

#[derive(Debug, Clone)]
struct Stream {
    fs: i32,
    ch: i32,
    app: i32,
    frame_size: usize,
    input: Input,
    frames: usize,
    seed: u64,
    init: Vec<(i32, i32)>,
    /// Probability (1/n) of a random CTL before each frame (0 = never).
    ctl_rate: i32,
    /// Output buffer sizes to pick from.
    max_bytes: Vec<usize>,
}

fn dup_mode() -> bool {
    std::env::var_os("OPUSORUS_OE_DUP").is_some()
}

fn run_stream(s: &Stream, dup: bool) {
    let dup = dup || dup_mode();
    let ctx0 = format!("{s:?}");
    let r0 = code(Encoder::new_raw(s.fs, s.ch, s.app));
    let c0 = CEnc::new(s.fs, s.ch, s.app, dup);
    let (mut r, mut c) = match (r0, c0) {
        (Ok(r), Ok(c)) => (r, c),
        (Err(a), Err(b)) => {
            assert_eq!(a, b, "{ctx0}: create");
            return;
        }
        (a, b) => panic!("{ctx0}: create mismatch {:?} vs {:?}", a.err(), b.err()),
    };
    for &(req, val) in &s.init {
        assert_eq!(
            code(r.ctl_set(req, val)),
            c.ctl_set(req, val),
            "{ctx0}: init ctl {req} {val}"
        );
    }
    let ch = s.ch as usize;
    // Enough input for any expert frame duration change (analysis reads the whole argument).
    let total = s.frame_size * (s.frames + 1);
    let sig = program(total, ch, s.fs, s.seed);
    let mut rng = Rng::new(s.seed ^ 0xABCD);
    let mut out_r = vec![0u8; 24000];
    let mut out_c = vec![0u8; 24000];
    for f in 0..s.frames {
        let ctx = format!("{ctx0} frame {f}");
        if s.ctl_rate > 0 && rng.range_i32(0, s.ctl_rate - 1) == 0 {
            let (req, val) = random_ctl(&mut rng, s.fs);
            assert_eq!(
                code(r.ctl_set(req, val)),
                c.ctl_set(req, val),
                "{ctx}: ctl {req} {val}"
            );
        }
        let max = s.max_bytes[rng.range_i32(0, s.max_bytes.len() as i32 - 1) as usize];
        let x = &sig[f * s.frame_size * ch..(f + 1) * s.frame_size * ch];
        out_r[..max].fill(0);
        out_c[..max].fill(0);
        let rr = rust_encode(&mut r, s.input, x, s.frame_size, &mut out_r[..max]);
        let cr = c.encode(s.input, x, s.frame_size, &mut out_c[..max]);
        assert_eq!(rr, cr, "{ctx}: return value (max {max})");
        if let Ok(n) = rr {
            assert_slice_eq(&format!("{ctx}: packet"), &out_r[..n], &out_c[..n]);
        }
        for req in GET_REQUESTS {
            assert_eq!(code(r.ctl_get(req)), c.ctl_get(req), "{ctx}: get {req}");
        }
        compare_dump(&r, &c, &ctx);
    }
}

fn frame_sizes(fs: i32) -> Vec<usize> {
    [25, 50, 100, 200, 400, 600, 800, 1000, 1200]
        .iter()
        .map(|&ms10| (fs as usize) * ms10 / 10000)
        .collect()
}

const MAXB_NORMAL: [usize; 3] = [1500, 4000, 1276];

/// Base matrix: every application, rate, channel count and frame size with defaults, then a
/// handful of bitrates / VBR modes / complexities.
#[test]
fn streams_default_matrix() {
    let mut rng = Rng::new(0x5100);
    for fs in [8000, 12000, 16000, 24000, 48000] {
        for ch in 1..=2 {
            for app in APPS {
                for fsz in frame_sizes(fs) {
                    let input = [Input::I16, Input::I24, Input::F32][rng.range_i32(0, 2) as usize];
                    let bitrate = [AUTO, 12000, 24000, 64000, 160000, BITRATE_MAX]
                        [rng.range_i32(0, 5) as usize];
                    let s = Stream {
                        fs,
                        ch,
                        app,
                        frame_size: fsz,
                        input,
                        frames: (48000 * 3 / 2 / (fsz * 48000 / fs as usize)).clamp(6, 30),
                        seed: rng.next_u64(),
                        init: vec![
                            (OPUS_SET_BITRATE_REQUEST, bitrate),
                            (OPUS_SET_VBR_REQUEST, rng.range_i32(0, 1)),
                            (OPUS_SET_COMPLEXITY_REQUEST, rng.range_i32(0, 10)),
                        ],
                        ctl_rate: 0,
                        max_bytes: MAXB_NORMAL.to_vec(),
                    };
                    run_stream(&s, false);
                }
            }
        }
    }
}

/// Long streams with random CTL changes (mode switches, bandwidth changes, FEC, DTX, resets,
/// forced modes/channels, application changes...).
#[test]
fn streams_random_ctls() {
    let mut rng = Rng::new(0x5200);
    for i in 0..300 {
        let fs = [8000, 12000, 16000, 24000, 48000, 48000, 16000][rng.range_i32(0, 6) as usize];
        let ch = rng.range_i32(1, 2);
        let app = [
            APP_VOIP,
            APP_AUDIO,
            APP_AUDIO,
            APP_VOIP,
            APP_LOWDELAY,
            APP_RSILK,
            APP_RCELT,
        ][rng.range_i32(0, 6) as usize];
        let sizes = frame_sizes(fs);
        let fsz = sizes[[3, 3, 2, 4, 5, 1, 0, 6, 8][rng.range_i32(0, 8) as usize]];
        let s = Stream {
            fs,
            ch,
            app,
            frame_size: fsz,
            input: [Input::I16, Input::I24, Input::F32][i % 3],
            frames: (48000 * 4 / (fsz * 48000 / fs as usize)).clamp(10, 100),
            seed: rng.next_u64(),
            init: vec![
                (OPUS_SET_COMPLEXITY_REQUEST, rng.range_i32(0, 10)),
                (OPUS_SET_BITRATE_REQUEST, rng.range_i32(6000, 100000)),
            ],
            ctl_rate: 4,
            max_bytes: vec![1500, 1500, 4000, 400, 100],
        };
        run_stream(&s, i % 10 == 0);
    }
}

/// Exhaustive configuration sweep: every rate, channel count, application, frame size,
/// bitrate, VBR mode and three complexities (about 32k short streams).
#[test]
#[ignore = "exhaustive sweep (minutes); run with --ignored"]
fn exhaustive_config_sweep() {
    let mut rng = Rng::new(0x5F00);
    for fs in [8000, 12000, 16000, 24000, 48000] {
        for ch in 1..=2 {
            for app in APPS {
                for fsz in frame_sizes(fs) {
                    for br in [AUTO, BITRATE_MAX, 6000, 12000, 24000, 48000, 128000, 510000] {
                        for vbr_mode in 0..3 {
                            for cx in [0, 5, 10] {
                                let s = Stream {
                                    fs,
                                    ch,
                                    app,
                                    frame_size: fsz,
                                    input: [Input::I16, Input::I24, Input::F32]
                                        [rng.range_i32(0, 2) as usize],
                                    frames: 8,
                                    seed: rng.next_u64(),
                                    init: vec![
                                        (OPUS_SET_BITRATE_REQUEST, br),
                                        (OPUS_SET_VBR_REQUEST, i32::from(vbr_mode != 2)),
                                        (OPUS_SET_VBR_CONSTRAINT_REQUEST, i32::from(vbr_mode == 1)),
                                        (OPUS_SET_COMPLEXITY_REQUEST, cx),
                                    ],
                                    ctl_rate: 0,
                                    max_bytes: MAXB_NORMAL.to_vec(),
                                };
                                run_stream(&s, false);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Long random-CTL streams (thousands of frames each).
#[test]
#[ignore = "long streams (minutes); run with --ignored"]
fn long_random_streams() {
    let mut rng = Rng::new(0x6000);
    for i in 0..60 {
        let fs = [8000, 12000, 16000, 24000, 48000][i % 5];
        let sizes = frame_sizes(fs);
        let s = Stream {
            fs,
            ch: 1 + (i % 2) as i32,
            app: [APP_VOIP, APP_AUDIO, APP_LOWDELAY][i % 3],
            frame_size: sizes[[2, 3, 4, 5, 8][i % 5]],
            input: [Input::I16, Input::I24, Input::F32][i % 3],
            frames: 1500,
            seed: rng.next_u64(),
            init: vec![],
            ctl_rate: 10,
            max_bytes: vec![1500, 4000, 200],
        };
        run_stream(&s, false);
    }
}

/// Bitrate sweep 6 kb/s .. 510 kb/s with VBR / CVBR / CBR at 20 ms, all rates and channels.
#[test]
fn streams_bitrate_sweep() {
    let mut rng = Rng::new(0x5300);
    let bitrates = [
        6000, 8000, 10000, 12000, 16000, 20000, 24000, 32000, 40000, 48000, 64000, 80000, 96000,
        128000, 192000, 256000, 384000, 510000,
    ];
    for (k, &br) in bitrates.iter().enumerate() {
        for vbr_mode in 0..3 {
            let fs = [8000, 12000, 16000, 24000, 48000][(k + vbr_mode) % 5];
            let ch = 1 + (k % 2) as i32;
            let app = [APP_VOIP, APP_AUDIO, APP_LOWDELAY][(k + 2 * vbr_mode) % 3];
            let s = Stream {
                fs,
                ch,
                app,
                frame_size: fs as usize / 50,
                input: Input::F32,
                frames: 40,
                seed: rng.next_u64(),
                init: vec![
                    (OPUS_SET_BITRATE_REQUEST, br),
                    (OPUS_SET_VBR_REQUEST, i32::from(vbr_mode != 2)),
                    (OPUS_SET_VBR_CONSTRAINT_REQUEST, i32::from(vbr_mode == 1)),
                    (OPUS_SET_COMPLEXITY_REQUEST, 10 - (k as i32 % 11)),
                ],
                ctl_rate: 0,
                max_bytes: MAXB_NORMAL.to_vec(),
            };
            run_stream(&s, false);
        }
    }
}

/// Feature combinations: FEC with loss, DTX on silence/noise, forced bandwidth / max bandwidth,
/// forced channels, signal hints, LSB depth, prediction / phase inversion disabled, forced
/// modes, expert frame durations.
#[test]
fn streams_features() {
    let mut rng = Rng::new(0x5400);
    let features: Vec<Vec<(i32, i32)>> = vec![
        vec![
            (OPUS_SET_INBAND_FEC_REQUEST, 1),
            (OPUS_SET_PACKET_LOSS_PERC_REQUEST, 10),
        ],
        vec![
            (OPUS_SET_INBAND_FEC_REQUEST, 2),
            (OPUS_SET_PACKET_LOSS_PERC_REQUEST, 30),
        ],
        vec![
            (OPUS_SET_INBAND_FEC_REQUEST, 1),
            (OPUS_SET_PACKET_LOSS_PERC_REQUEST, 3),
        ],
        vec![(OPUS_SET_DTX_REQUEST, 1)],
        vec![(OPUS_SET_DTX_REQUEST, 1), (OPUS_SET_COMPLEXITY_REQUEST, 3)],
        vec![(OPUS_SET_DTX_REQUEST, 1), (OPUS_SET_SIGNAL_REQUEST, 3001)],
        vec![(OPUS_SET_BANDWIDTH_REQUEST, 1101)],
        vec![(OPUS_SET_BANDWIDTH_REQUEST, 1102)],
        vec![(OPUS_SET_BANDWIDTH_REQUEST, 1103)],
        vec![(OPUS_SET_BANDWIDTH_REQUEST, 1104)],
        vec![(OPUS_SET_MAX_BANDWIDTH_REQUEST, 1103)],
        vec![(OPUS_SET_MAX_BANDWIDTH_REQUEST, 1101)],
        vec![(OPUS_SET_FORCE_CHANNELS_REQUEST, 1)],
        vec![(OPUS_SET_FORCE_CHANNELS_REQUEST, 2)],
        vec![(OPUS_SET_SIGNAL_REQUEST, 3001)],
        vec![(OPUS_SET_SIGNAL_REQUEST, 3002)],
        vec![(OPUS_SET_LSB_DEPTH_REQUEST, 8)],
        vec![(OPUS_SET_LSB_DEPTH_REQUEST, 16)],
        vec![(OPUS_SET_PREDICTION_DISABLED_REQUEST, 1)],
        vec![(OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, 1)],
        vec![(OPUS_SET_FORCE_MODE_REQUEST, MODE_SILK)],
        vec![(OPUS_SET_FORCE_MODE_REQUEST, MODE_HYBRID)],
        vec![(OPUS_SET_FORCE_MODE_REQUEST, MODE_CELT)],
        vec![(OPUS_SET_VOICE_RATIO_REQUEST, 90)],
        vec![(OPUS_SET_LFE_REQUEST, 1)],
        vec![(OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, 5001)],
        vec![(OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, 5003)],
        vec![(OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, 5005)],
        vec![(OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, 5009)],
    ];
    for (k, feat) in features.iter().enumerate() {
        for variant in 0..4 {
            let fs = [16000, 48000, 24000, 8000][variant];
            let ch = 1 + ((k + variant) % 2) as i32;
            let app = [APP_VOIP, APP_AUDIO, APP_AUDIO, APP_VOIP][variant];
            let is_expert = feat[0].0 == OPUS_SET_EXPERT_FRAME_DURATION_REQUEST;
            let fsz = if is_expert {
                fs as usize * 12 / 100
            } else {
                frame_sizes(fs)[[3, 2, 4, 5][variant]]
            };
            let mut init = feat.clone();
            init.push((
                OPUS_SET_BITRATE_REQUEST,
                [12000, 20000, 32000, 64000][(k + variant) % 4],
            ));
            init.push((OPUS_SET_VBR_REQUEST, i32::from(variant != 3)));
            let s = Stream {
                fs,
                ch,
                app,
                frame_size: fsz,
                input: [Input::F32, Input::I16, Input::I24][(k + variant) % 3],
                frames: if is_expert { 12 } else { 40 },
                seed: rng.next_u64(),
                init,
                ctl_rate: 0,
                max_bytes: MAXB_NORMAL.to_vec(),
            };
            run_stream(&s, variant == 0);
        }
    }
}

/// Mode transitions: bitrate / force-mode / bandwidth switching every few frames so the
/// SILK<->CELT redundancy and prefill paths run, at every frame size.
#[test]
fn streams_mode_transitions() {
    let mut rng = Rng::new(0x5500);
    for fs in [16000, 24000, 48000] {
        for ch in 1..=2 {
            for fsz in frame_sizes(fs) {
                let mut r = Encoder::new_raw(fs, ch, APP_AUDIO).unwrap();
                let mut c = oc::Enc::new(fs, ch, APP_AUDIO).unwrap();
                let frames = 60.min(48000 * 6 / (fsz * 48000 / fs as usize)).max(12);
                let sig = program(fsz * frames, ch as usize, fs, rng.next_u64());
                let mut out_r = vec![0u8; 8000];
                let mut out_c = vec![0u8; 8000];
                for f in 0..frames {
                    if f % 3 == 0 {
                        let (req, val) = match rng.range_i32(0, 3) {
                            0 => (
                                OPUS_SET_FORCE_MODE_REQUEST,
                                [AUTO, MODE_SILK, MODE_HYBRID, MODE_CELT]
                                    [rng.range_i32(0, 3) as usize],
                            ),
                            1 => (
                                OPUS_SET_BITRATE_REQUEST,
                                [8000, 16000, 32000, 96000][rng.range_i32(0, 3) as usize],
                            ),
                            2 => (
                                OPUS_SET_BANDWIDTH_REQUEST,
                                [AUTO, 1101, 1102, 1103, 1104, 1105][rng.range_i32(0, 5) as usize],
                            ),
                            _ => (
                                OPUS_SET_SIGNAL_REQUEST,
                                [AUTO, 3001, 3002][rng.range_i32(0, 2) as usize],
                            ),
                        };
                        assert_eq!(code(r.ctl_set(req, val)), c.ctl_set(req, val));
                    }
                    let x = &sig[f * fsz * ch as usize..(f + 1) * fsz * ch as usize];
                    let rr = code(r.encode_float(x, fsz, &mut out_r));
                    let cr = c.encode_float(x, fsz, &mut out_c);
                    assert_eq!(rr, cr, "fs {fs} ch {ch} fsz {fsz} frame {f}");
                    let n = rr.unwrap();
                    assert_slice_eq("packet", &out_r[..n], &out_c[..n]);
                    assert_eq!(r.final_range(), c.final_range().unwrap());
                }
            }
        }
    }
}

/// Small buffers: PLC frames, buffer-too-small errors, CBR padding with tiny budgets.
#[test]
fn streams_small_buffers() {
    let mut rng = Rng::new(0x5600);
    for i in 0..80 {
        let fs = [8000, 16000, 48000, 24000, 12000][i % 5];
        let sizes = frame_sizes(fs);
        let s = Stream {
            fs,
            ch: 1 + (i % 2) as i32,
            app: APPS[i % 5],
            frame_size: sizes[rng.range_i32(0, 8) as usize],
            input: Input::F32,
            frames: 10,
            seed: rng.next_u64(),
            init: vec![
                (OPUS_SET_VBR_REQUEST, (i / 2 % 2) as i32),
                (
                    OPUS_SET_BITRATE_REQUEST,
                    [AUTO, 6000, 64000, BITRATE_MAX][i / 4 % 4],
                ),
            ],
            ctl_rate: 0,
            max_bytes: vec![0, 1, 2, 3, 4, 5, 8, 12, 20, 30, 60],
        };
        run_stream(&s, i % 8 == 0);
    }
}

/// Float API robustness: NaN / huge values in the input.
#[test]
fn float_non_finite_input() {
    for (fs, ch, app) in [
        (48000, 2, APP_AUDIO),
        (16000, 1, APP_VOIP),
        (48000, 1, APP_RCELT),
    ] {
        let mut r = Encoder::new_raw(fs, ch, app).unwrap();
        let mut c = oc::Enc::new(fs, ch, app).unwrap();
        let fsz = fs as usize / 50;
        let mut sig = program(fsz * 20, ch as usize, fs, 99);
        sig[fsz * ch as usize * 3 + 5] = f32::NAN;
        sig[fsz * ch as usize * 7 + 1] = 1e30;
        sig[fsz * ch as usize * 9 + 2] = f32::INFINITY;
        for v in &mut sig[fsz * ch as usize * 12..fsz * ch as usize * 13] {
            *v *= 1e5;
        }
        let mut out_r = vec![0u8; 1500];
        let mut out_c = vec![0u8; 1500];
        for f in 0..20 {
            let x = &sig[f * fsz * ch as usize..(f + 1) * fsz * ch as usize];
            let rr = code(r.encode_float(x, fsz, &mut out_r));
            let cr = c.encode_float(x, fsz, &mut out_c);
            assert_eq!(rr, cr, "frame {f}");
            let n = rr.unwrap();
            assert_slice_eq("packet", &out_r[..n], &out_c[..n]);
            assert_eq!(r.final_range(), c.final_range().unwrap());
        }
    }
}

/// The energy mask (as set by the surround encoder) on a single encoder.
#[test]
fn energy_mask_streams() {
    let mut rng = Rng::new(0x5700);
    for i in 0..24 {
        let fs = [48000, 24000, 16000][i % 3];
        let ch = 1 + (i % 2) as i32;
        let mut r = Encoder::new_raw(fs, ch, APP_AUDIO).unwrap();
        let mut c = oc::Enc::new(fs, ch, APP_AUDIO).unwrap();
        let br = [16000, 32000, 64000, 128000][i / 6 % 4];
        r.set_bitrate(Bitrate::Bits(br)).unwrap();
        c.ctl_set(OPUS_SET_BITRATE_REQUEST, br).unwrap();
        if i % 4 == 3 {
            r.set_vbr(false);
            c.ctl_set(OPUS_SET_VBR_REQUEST, 0).unwrap();
        }
        let fsz = fs as usize / 50;
        let sig = program(fsz * 30, ch as usize, fs, rng.next_u64());
        let mut out_r = vec![0u8; 1500];
        let mut out_c = vec![0u8; 1500];
        for f in 0..30 {
            if f % 7 == 6 {
                r.set_energy_mask(None);
                c.set_energy_mask(None).unwrap();
            } else {
                let mask: Vec<f32> = (0..21 * ch as usize).map(|_| 3.0 * rng.f32_sym()).collect();
                r.set_energy_mask(Some(&mask));
                c.set_energy_mask(Some(&mask)).unwrap();
            }
            let x = &sig[f * fsz * ch as usize..(f + 1) * fsz * ch as usize];
            let rr = code(r.encode_float(x, fsz, &mut out_r));
            let cr = c.encode_float(x, fsz, &mut out_c);
            assert_eq!(rr, cr, "cfg {i} frame {f}");
            let n = rr.unwrap();
            assert_slice_eq("packet", &out_r[..n], &out_c[..n]);
            assert_eq!(r.final_range(), c.final_range().unwrap());
        }
    }
}

/// Every encode entry point on the same stream (i16 / i24 / float), interleaved.
#[test]
fn mixed_entry_points() {
    let mut rng = Rng::new(0x5800);
    for (fs, ch) in [(48000, 2), (16000, 1), (24000, 2), (8000, 1), (12000, 2)] {
        let mut r = Encoder::new_raw(fs, ch, APP_AUDIO).unwrap();
        let mut c = oc::Enc::new(fs, ch, APP_AUDIO).unwrap();
        let fsz = fs as usize / 50;
        let sig = program(fsz * 60, ch as usize, fs, rng.next_u64());
        let mut out_r = vec![0u8; 1500];
        let mut out_c = vec![0u8; 1500];
        for f in 0..60 {
            let x = &sig[f * fsz * ch as usize..(f + 1) * fsz * ch as usize];
            let input = [Input::I16, Input::I24, Input::F32][rng.range_i32(0, 2) as usize];
            let rr = rust_encode(&mut r, input, x, fsz, &mut out_r);
            let cr = match input {
                Input::I16 => c.encode(&to_i16(x), fsz, &mut out_c),
                Input::I24 => c.encode24(&to_i24(x), fsz, &mut out_c),
                Input::F32 => c.encode_float(x, fsz, &mut out_c),
            };
            assert_eq!(rr, cr);
            let n = rr.unwrap();
            assert_slice_eq("packet", &out_r[..n], &out_c[..n]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Error paths and the typed API
// ---------------------------------------------------------------------------------------------

#[test]
fn create_errors() {
    for fs in [
        0, 7999, 8000, 11025, 12000, 16000, 22050, 24000, 44100, 48000, 96000, 192000,
    ] {
        for ch in -1..=3 {
            for app in [
                0,
                2047,
                APP_VOIP,
                APP_AUDIO,
                2050,
                APP_LOWDELAY,
                APP_RSILK,
                APP_RCELT,
                2054,
            ] {
                let r = code(Encoder::new_raw(fs, ch, app)).map(drop);
                let c = oc::Enc::new(fs, ch, app).map(drop);
                assert_eq!(r, c, "fs {fs} ch {ch} app {app}");
            }
        }
    }
    assert_eq!(oe::encoder_get_size(0), 0);
    assert_eq!(oe::encoder_get_size(3), 0);
    assert!(oe::encoder_get_size(1) > 0 && oe::encoder_get_size(2) > oe::encoder_get_size(1));
}

#[test]
fn encode_errors() {
    for app in APPS {
        for (fs, ch) in [(48000, 2), (8000, 1), (16000, 2)] {
            let mut r = Encoder::new_raw(fs, ch, app).unwrap();
            let mut c = oc::Enc::new(fs, ch, app).unwrap();
            let big = vec![0.1f32; fs as usize * 2 * 2];
            for n in [
                0,
                1,
                7,
                fs as usize / 400 - 1,
                fs as usize / 300,
                fs as usize / 400 * 3,
                fs as usize / 10 + 1,
            ] {
                for maxb in [0, 1, 100] {
                    let mut o1 = vec![0u8; maxb];
                    let mut o2 = vec![0u8; maxb];
                    assert_eq!(
                        code(r.encode_float(&big, n, &mut o1)),
                        c.encode_float(&big, n, &mut o2),
                        "float n {n} maxb {maxb}"
                    );
                    assert_eq!(r.final_range(), c.final_range().unwrap());
                    assert_eq!(
                        code(r.encode(&to_i16(&big), n, &mut o1)),
                        c.encode(&to_i16(&big), n, &mut o2),
                        "i16 n {n} maxb {maxb}"
                    );
                    assert_eq!(r.final_range(), c.final_range().unwrap());
                }
            }
            // 100 ms in one byte.
            let n = fs as usize / 10;
            let mut o1 = [0u8; 1];
            let mut o2 = [0u8; 1];
            assert_eq!(
                code(r.encode_float(&big, n, &mut o1)),
                c.encode_float(&big, n, &mut o2)
            );
            assert_eq!(o1, o2);
            // Rust-only: short PCM.
            assert_eq!(
                r.encode_float(&big[..10], 960, &mut [0u8; 100]),
                Err(Error::BadArg)
            );
            // Every CTL request number in a range, set and get.
            for req in (3990..4070).chain(10000..10030).chain(11000..11025) {
                if req == OPUS_SET_ENERGY_MASK_REQUEST || req == CELT_GET_MODE_REQUEST {
                    continue;
                }
                if req % 2 == 0 || req == OPUS_RESET_STATE {
                    if req == OPUS_SET_DNN_BLOB_REQUEST {
                        continue;
                    }
                    for v in [-1000, -1, 0, 1, 2, 5000, 1101, 3001] {
                        assert_eq!(code(r.ctl_set(req, v)), c.ctl_set(req, v), "set {req} {v}");
                    }
                } else {
                    let cv = if req == OPUS_GET_FINAL_RANGE_REQUEST {
                        c.final_range().map(|v| v as i32)
                    } else {
                        c.ctl_get(req)
                    };
                    assert_eq!(code(r.ctl_get(req)), cv, "get {req}");
                }
            }
        }
    }
}

#[test]
fn typed_ctls() {
    for app in APPS {
        for ch in 1..=2 {
            let mut r = Encoder::new(48000, ch, Application::from_raw(app).unwrap()).unwrap();
            let mut c = oc::Enc::new(48000, ch, app).unwrap();
            // Setters
            assert_eq!(
                code(r.set_bitrate(Bitrate::Bits(0))),
                c.ctl_set(OPUS_SET_BITRATE_REQUEST, 0)
            );
            assert_eq!(
                code(r.set_bitrate(Bitrate::Bits(300))),
                c.ctl_set(OPUS_SET_BITRATE_REQUEST, 300)
            );
            assert_eq!(r.bitrate(), c.ctl_get(OPUS_GET_BITRATE_REQUEST).unwrap());
            assert_eq!(
                code(r.set_bitrate(Bitrate::Max)),
                c.ctl_set(OPUS_SET_BITRATE_REQUEST, BITRATE_MAX)
            );
            assert_eq!(r.bitrate(), c.ctl_get(OPUS_GET_BITRATE_REQUEST).unwrap());
            assert_eq!(
                code(r.set_bitrate(Bitrate::Auto)),
                c.ctl_set(OPUS_SET_BITRATE_REQUEST, AUTO)
            );
            assert_eq!(r.bitrate(), c.ctl_get(OPUS_GET_BITRATE_REQUEST).unwrap());
            assert_eq!(
                code(r.set_force_channels(Some(3))),
                c.ctl_set(OPUS_SET_FORCE_CHANNELS_REQUEST, 3)
            );
            assert_eq!(
                code(r.set_force_channels(Some(2))),
                c.ctl_set(OPUS_SET_FORCE_CHANNELS_REQUEST, 2)
            );
            assert_eq!(
                r.force_channels().map_or(AUTO, i32::from),
                c.ctl_get(OPUS_GET_FORCE_CHANNELS_REQUEST).unwrap()
            );
            r.set_max_bandwidth(Bandwidth::Wideband);
            c.ctl_set(OPUS_SET_MAX_BANDWIDTH_REQUEST, 1103).unwrap();
            assert_eq!(
                r.max_bandwidth().to_raw(),
                c.ctl_get(OPUS_GET_MAX_BANDWIDTH_REQUEST).unwrap()
            );
            r.set_bandwidth(Some(Bandwidth::Mediumband));
            c.ctl_set(OPUS_SET_BANDWIDTH_REQUEST, 1102).unwrap();
            r.set_bandwidth(None);
            c.ctl_set(OPUS_SET_BANDWIDTH_REQUEST, AUTO).unwrap();
            assert_eq!(
                r.bandwidth().to_raw(),
                c.ctl_get(OPUS_GET_BANDWIDTH_REQUEST).unwrap()
            );
            r.set_dtx(true);
            c.ctl_set(OPUS_SET_DTX_REQUEST, 1).unwrap();
            assert_eq!(i32::from(r.dtx()), c.ctl_get(OPUS_GET_DTX_REQUEST).unwrap());
            assert_eq!(
                code(r.set_complexity(11)),
                c.ctl_set(OPUS_SET_COMPLEXITY_REQUEST, 11)
            );
            assert_eq!(
                code(r.set_complexity(4)),
                c.ctl_set(OPUS_SET_COMPLEXITY_REQUEST, 4)
            );
            assert_eq!(
                r.complexity(),
                c.ctl_get(OPUS_GET_COMPLEXITY_REQUEST).unwrap()
            );
            assert_eq!(
                code(r.set_inband_fec(3)),
                c.ctl_set(OPUS_SET_INBAND_FEC_REQUEST, 3)
            );
            assert_eq!(
                code(r.set_inband_fec(2)),
                c.ctl_set(OPUS_SET_INBAND_FEC_REQUEST, 2)
            );
            assert_eq!(
                r.inband_fec(),
                c.ctl_get(OPUS_GET_INBAND_FEC_REQUEST).unwrap()
            );
            assert_eq!(
                code(r.set_packet_loss_perc(101)),
                c.ctl_set(OPUS_SET_PACKET_LOSS_PERC_REQUEST, 101)
            );
            assert_eq!(
                code(r.set_packet_loss_perc(7)),
                c.ctl_set(OPUS_SET_PACKET_LOSS_PERC_REQUEST, 7)
            );
            assert_eq!(
                r.packet_loss_perc(),
                c.ctl_get(OPUS_GET_PACKET_LOSS_PERC_REQUEST).unwrap()
            );
            r.set_vbr(false);
            c.ctl_set(OPUS_SET_VBR_REQUEST, 0).unwrap();
            assert_eq!(i32::from(r.vbr()), c.ctl_get(OPUS_GET_VBR_REQUEST).unwrap());
            r.set_vbr_constraint(false);
            c.ctl_set(OPUS_SET_VBR_CONSTRAINT_REQUEST, 0).unwrap();
            assert_eq!(
                i32::from(r.vbr_constraint()),
                c.ctl_get(OPUS_GET_VBR_CONSTRAINT_REQUEST).unwrap()
            );
            r.set_signal(Signal::Music);
            c.ctl_set(OPUS_SET_SIGNAL_REQUEST, 3002).unwrap();
            assert_eq!(
                r.signal().to_raw(),
                c.ctl_get(OPUS_GET_SIGNAL_REQUEST).unwrap()
            );
            assert_eq!(
                r.lookahead() as i32,
                c.ctl_get(OPUS_GET_LOOKAHEAD_REQUEST).unwrap()
            );
            assert_eq!(
                r.sample_rate(),
                c.ctl_get(OPUS_GET_SAMPLE_RATE_REQUEST).unwrap()
            );
            assert_eq!(
                code(r.set_lsb_depth(7)),
                c.ctl_set(OPUS_SET_LSB_DEPTH_REQUEST, 7)
            );
            assert_eq!(
                code(r.set_lsb_depth(12)),
                c.ctl_set(OPUS_SET_LSB_DEPTH_REQUEST, 12)
            );
            assert_eq!(
                r.lsb_depth(),
                c.ctl_get(OPUS_GET_LSB_DEPTH_REQUEST).unwrap()
            );
            r.set_expert_frame_duration(FrameSize::Ms40);
            c.ctl_set(OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, 5005)
                .unwrap();
            assert_eq!(
                r.expert_frame_duration().to_raw(),
                c.ctl_get(OPUS_GET_EXPERT_FRAME_DURATION_REQUEST).unwrap()
            );
            r.set_prediction_disabled(true);
            c.ctl_set(OPUS_SET_PREDICTION_DISABLED_REQUEST, 1).unwrap();
            assert_eq!(
                i32::from(r.prediction_disabled()),
                c.ctl_get(OPUS_GET_PREDICTION_DISABLED_REQUEST).unwrap()
            );
            r.set_phase_inversion_disabled(true);
            c.ctl_set(OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, 1)
                .unwrap();
            assert_eq!(
                i32::from(r.phase_inversion_disabled()),
                c.ctl_get(OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST)
                    .unwrap()
            );
            assert_eq!(
                code(r.set_voice_ratio(101)),
                c.ctl_set(OPUS_SET_VOICE_RATIO_REQUEST, 101)
            );
            assert_eq!(
                code(r.set_voice_ratio(40)),
                c.ctl_set(OPUS_SET_VOICE_RATIO_REQUEST, 40)
            );
            assert_eq!(
                r.voice_ratio(),
                c.ctl_get(OPUS_GET_VOICE_RATIO_REQUEST).unwrap()
            );
            assert_eq!(
                code(r.set_force_mode(1003)),
                c.ctl_set(OPUS_SET_FORCE_MODE_REQUEST, 1003)
            );
            assert_eq!(
                code(r.set_force_mode(MODE_CELT)),
                c.ctl_set(OPUS_SET_FORCE_MODE_REQUEST, MODE_CELT)
            );
            assert_eq!(
                i32::from(r.in_dtx()),
                c.ctl_get(OPUS_GET_IN_DTX_REQUEST).unwrap()
            );
            #[cfg(feature = "qext")]
            {
                r.set_qext(true);
                c.ctl_set(OPUS_SET_QEXT_REQUEST, 1).unwrap();
                assert_eq!(
                    i32::from(r.qext()),
                    c.ctl_get(OPUS_GET_QEXT_REQUEST).unwrap()
                );
            }
            // Application: allowed before the first frame only.
            for a in [
                Application::Voip,
                Application::RestrictedLowDelay,
                Application::Audio,
            ] {
                assert_eq!(
                    code(r.set_application(a)),
                    c.ctl_set(OPUS_SET_APPLICATION_REQUEST, a.to_raw())
                );
                assert_eq!(
                    r.application().to_raw(),
                    c.ctl_get(OPUS_GET_APPLICATION_REQUEST).unwrap()
                );
            }
            let pcm = vec![0i16; 960 * ch as usize];
            let mut o1 = [0u8; 400];
            let mut o2 = [0u8; 400];
            assert_eq!(
                code(r.encode(&pcm, 960, &mut o1)),
                c.encode(&pcm, 960, &mut o2)
            );
            assert_eq!(o1, o2);
            for a in [Application::Voip, Application::Audio] {
                assert_eq!(
                    code(r.set_application(a)),
                    c.ctl_set(OPUS_SET_APPLICATION_REQUEST, a.to_raw())
                );
            }
            r.reset();
            c.reset().unwrap();
            assert_eq!(
                r.bandwidth().to_raw(),
                c.ctl_get(OPUS_GET_BANDWIDTH_REQUEST).unwrap()
            );
            assert_eq!(r.final_range(), c.final_range().unwrap());
            // init() resets everything.
            r.init(16000, 1, Application::Voip).unwrap();
            assert_eq!(r.sample_rate(), 16000);
            assert_eq!(r.channels(), 1);
            assert_eq!(r.complexity(), 9);
        }
    }
}

/// Full state comparison against the dumpable C copy on assorted configurations.
#[test]
fn state_dump_streams() {
    let mut rng = Rng::new(0x5900);
    for i in 0..30 {
        let fs = [8000, 12000, 16000, 24000, 48000][i % 5];
        let sizes = frame_sizes(fs);
        let s = Stream {
            fs,
            ch: 1 + (i / 5 % 2) as i32,
            app: APPS[i % 5],
            frame_size: sizes[[1, 2, 3, 4, 5, 7, 8][i % 7]],
            input: [Input::F32, Input::I16, Input::I24][i % 3],
            frames: 30,
            seed: rng.next_u64(),
            init: vec![(OPUS_SET_BITRATE_REQUEST, rng.range_i32(8000, 128000))],
            ctl_rate: 3,
            max_bytes: vec![1500, 4000, 60],
        };
        run_stream(&s, true);
    }
}

// ---------------------------------------------------------------------------------------------
// QEXT
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "qext")]
#[test]
fn qext_streams() {
    let mut rng = Rng::new(0x5A00);
    for i in 0..40 {
        let fs = [48000, 96000][i % 2];
        let ch = 1 + (i / 2 % 2) as i32;
        let app = [APP_AUDIO, APP_RCELT, APP_LOWDELAY, APP_VOIP][i / 4 % 4];
        let sizes = frame_sizes(fs);
        let s = Stream {
            fs,
            ch,
            app,
            frame_size: sizes[[3, 2, 1, 0, 4, 5, 8][i % 7]],
            input: [Input::F32, Input::I24, Input::I16][i % 3],
            frames: 20,
            seed: rng.next_u64(),
            init: vec![
                (OPUS_SET_QEXT_REQUEST, 1),
                (
                    OPUS_SET_BITRATE_REQUEST,
                    [256000, 510000, BITRATE_MAX, 128000, 64000][i % 5],
                ),
                (OPUS_SET_VBR_REQUEST, i32::from(i % 5 != 4)),
                (OPUS_SET_COMPLEXITY_REQUEST, [10, 5, 0][i % 3]),
            ],
            ctl_rate: if i % 4 == 0 { 5 } else { 0 },
            max_bytes: vec![24000, 4000, 1500],
        };
        run_stream(&s, i % 6 == 0);
    }
    // 96 kHz without QEXT enabled (SILK at 96 kHz input) with transitions.
    for i in 0..10 {
        let s = Stream {
            fs: 96000,
            ch: 1 + (i % 2),
            app: [APP_VOIP, APP_AUDIO, APP_RSILK][i as usize % 3],
            frame_size: 1920,
            input: Input::F32,
            frames: 40,
            seed: rng.next_u64(),
            init: vec![(
                OPUS_SET_BITRATE_REQUEST,
                [12000, 32000, 96000][i as usize % 3],
            )],
            ctl_rate: 4,
            max_bytes: vec![1500],
        };
        run_stream(&s, i % 3 == 0);
    }
}

// ---------------------------------------------------------------------------------------------
// Multistream
// ---------------------------------------------------------------------------------------------

fn ms_code<T>(r: Result<T, Error>) -> Result<T, i32> {
    code(r)
}

/// Compares a multistream stream: bytes, length, final range, per-stream CTLs.
fn run_ms(
    mut r: MsEncoder,
    mut c: oc::MsEnc,
    fs: i32,
    fsz: usize,
    frames: usize,
    seed: u64,
    input: Input,
    maxb: &[usize],
    ctls: &[(i32, i32)],
    ctx: &str,
) {
    for &(req, val) in ctls {
        assert_eq!(
            ms_code(r.ctl_set(req, val)),
            c.ctl_set(req, val),
            "{ctx}: ctl {req} {val}"
        );
    }
    let ch = r.channels() as usize;
    let sig = program(fsz * frames, ch, fs, seed);
    let mut rng = Rng::new(seed);
    let mut out_r = vec![0u8; 60000];
    let mut out_c = vec![0u8; 60000];
    for f in 0..frames {
        let x = &sig[f * fsz * ch..(f + 1) * fsz * ch];
        let max = maxb[rng.range_i32(0, maxb.len() as i32 - 1) as usize];
        let rr = ms_code(match input {
            Input::I16 => r.encode(&to_i16(x), fsz, &mut out_r[..max]),
            Input::I24 => r.encode24(&to_i24(x), fsz, &mut out_r[..max]),
            Input::F32 => r.encode_float(x, fsz, &mut out_r[..max]),
        });
        let cr = match input {
            Input::I16 => c.encode(&to_i16(x), fsz, &mut out_c[..max]),
            Input::I24 => c.encode24(&to_i24(x), fsz, &mut out_c[..max]),
            Input::F32 => c.encode_float(x, fsz, &mut out_c[..max]),
        };
        assert_eq!(rr, cr, "{ctx}: frame {f} max {max}");
        if let Ok(n) = rr {
            assert_slice_eq(
                &format!("{ctx}: frame {f} packet"),
                &out_r[..n],
                &out_c[..n],
            );
        }
        assert_eq!(
            r.final_range(),
            c.final_range().unwrap(),
            "{ctx}: frame {f} rng"
        );
        for req in [
            OPUS_GET_BITRATE_REQUEST,
            OPUS_GET_BANDWIDTH_REQUEST,
            OPUS_GET_VBR_REQUEST,
            OPUS_GET_LOOKAHEAD_REQUEST,
            OPUS_GET_EXPERT_FRAME_DURATION_REQUEST,
            OPUS_GET_MAX_BANDWIDTH_REQUEST,
            OPUS_GET_IN_DTX_REQUEST,
            OPUS_GET_QEXT_REQUEST,
            OPUS_GET_SIGNAL_REQUEST,
        ] {
            assert_eq!(ms_code(r.ctl_get(req)), c.ctl_get(req), "{ctx}: get {req}");
        }
        for s in 0..r.streams() {
            let enc = r.encoder_state(s).unwrap();
            for req in [
                OPUS_GET_BANDWIDTH_REQUEST,
                OPUS_GET_BITRATE_REQUEST,
                OPUS_GET_IN_DTX_REQUEST,
            ] {
                assert_eq!(
                    code(enc.ctl_get(req)),
                    c.stream_ctl_get(s, req),
                    "{ctx}: stream {s} get {req}"
                );
            }
            assert_eq!(enc.final_range(), c.stream_final_range(s).unwrap());
        }
    }
    for s in [-1, r.streams(), 300] {
        assert_eq!(code(r.encoder_state(s).map(drop)), c.encoder_state_check(s));
    }
}

#[test]
fn ms_surround_families() {
    let mut rng = Rng::new(0x5B00);
    for family in [0, 1, 2, 255] {
        let chans: Vec<i32> = match family {
            0 => vec![1, 2, 3],
            1 => (1..=9).collect(),
            2 => vec![1, 2, 3, 4, 6, 9, 11, 16],
            _ => vec![1, 2, 3, 5, 8, 12],
        };
        for ch in chans {
            for app in [APP_AUDIO, APP_VOIP, APP_LOWDELAY, APP_RSILK, APP_RCELT] {
                let fs = [48000, 24000, 16000, 48000, 8000][rng.range_i32(0, 4) as usize];
                let r = code(MsEncoder::new_surround_raw(fs, ch, family, app));
                let c = oc::MsEnc::new_surround(fs, ch, family, app);
                let (r, c) = match (r, c) {
                    (Ok(r), Ok(c)) => (r, c),
                    (Err(a), Err(b)) => {
                        assert_eq!(a, b, "family {family} ch {ch}");
                        continue;
                    }
                    (a, b) => panic!("family {family} ch {ch}: {:?} vs {:?}", a.err(), b.err()),
                };
                assert_eq!(
                    (r.streams(), r.coupled_streams()),
                    (c.streams, c.coupled_streams)
                );
                assert_eq!(r.mapping(), &c.mapping[..]);
                let fsz = frame_sizes(fs)[[3, 2, 4, 5, 1, 7][rng.range_i32(0, 5) as usize]];
                let br =
                    [AUTO, BITRATE_MAX, 64000 * ch, 12000 * ch, 500][rng.range_i32(0, 4) as usize];
                let vbr = rng.range_i32(0, 1);
                let ctx = format!(
                    "family {family} ch {ch} app {app} fs {fs} fsz {fsz} br {br} vbr {vbr}"
                );
                run_ms(
                    r,
                    c,
                    fs,
                    fsz,
                    12,
                    rng.next_u64(),
                    [Input::I16, Input::I24, Input::F32][rng.range_i32(0, 2) as usize],
                    &[60000, 60000, 1500 * ch as usize, 20 * ch as usize],
                    &[
                        (OPUS_SET_BITRATE_REQUEST, br),
                        (OPUS_SET_VBR_REQUEST, vbr),
                        (OPUS_SET_COMPLEXITY_REQUEST, rng.range_i32(0, 10)),
                    ],
                    &ctx,
                );
            }
        }
    }
}

#[test]
fn ms_random_mappings() {
    let mut rng = Rng::new(0x5C00);
    for i in 0..150 {
        let ch = [1, 2, 3, 4, 6, 8, 12, 20][rng.range_i32(0, 7) as usize];
        let streams = rng.range_i32(0, ch + 1);
        let coupled = rng.range_i32(-1, streams.max(0));
        let mut mapping: Vec<u8> = (0..ch)
            .map(|_| rng.range_i32(0, (streams + coupled).max(1)) as u8)
            .collect();
        if rng.range_i32(0, 3) == 0 {
            // A valid layout: every coded channel once, the rest silent.
            let nc = (streams + coupled).max(0);
            for (k, m) in mapping.iter_mut().enumerate() {
                *m = if (k as i32) < nc { k as u8 } else { 255 };
            }
        }
        if rng.range_i32(0, 7) == 0 {
            mapping[0] = 255;
        }
        let fs = [48000, 16000, 24000, 8000, 12000][i % 5];
        let app = APPS[rng.range_i32(0, 4) as usize];
        let r = code(MsEncoder::new_raw(fs, ch, streams, coupled, &mapping, app));
        let c = oc::MsEnc::new(fs, ch, streams, coupled, &mapping, app);
        let (r, c) = match (r, c) {
            (Ok(r), Ok(c)) => (r, c),
            (Err(a), Err(b)) => {
                assert_eq!(a, b, "ch {ch} s {streams} c {coupled} map {mapping:?}");
                continue;
            }
            (a, b) => panic!(
                "ch {ch} s {streams} c {coupled} map {mapping:?}: {:?} vs {:?}",
                a.err(),
                b.err()
            ),
        };
        let fsz = frame_sizes(fs)[[3, 2, 4, 5, 8][rng.range_i32(0, 4) as usize]];
        let ctx = format!("ch {ch} s {streams} c {coupled} map {mapping:?} fs {fs} fsz {fsz}");
        run_ms(
            r,
            c,
            fs,
            fsz,
            8,
            rng.next_u64(),
            Input::F32,
            &[60000, 4000, 100],
            &[
                (
                    OPUS_SET_BITRATE_REQUEST,
                    [AUTO, 32000 * ch, 5000][rng.range_i32(0, 2) as usize],
                ),
                (OPUS_SET_VBR_REQUEST, rng.range_i32(0, 1)),
                (
                    OPUS_SET_FORCE_MODE_REQUEST,
                    [AUTO, MODE_CELT, MODE_SILK][rng.range_i32(0, 2) as usize],
                ),
                (
                    OPUS_SET_EXPERT_FRAME_DURATION_REQUEST,
                    [5000, 5000, 5004, 5010][rng.range_i32(0, 3) as usize],
                ),
            ],
            &ctx,
        );
    }
}

#[test]
fn ms_ctls_and_errors() {
    let mapping = [0u8, 1, 2, 3, 4, 5];
    let mut r = MsEncoder::new(48000, 6, 4, 2, &mapping, Application::Audio).unwrap();
    let mut c = oc::MsEnc::new(48000, 6, 4, 2, &mapping, APP_AUDIO).unwrap();
    for req in (3990..4070)
        .chain(5118..5125)
        .chain(10000..10030)
        .chain(11000..11025)
    {
        if req == OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST
            || req == OPUS_SET_ENERGY_MASK_REQUEST
            || req == CELT_GET_MODE_REQUEST
            || req == OPUS_SET_DNN_BLOB_REQUEST
        {
            continue;
        }
        if req % 2 == 0 || req == OPUS_RESET_STATE {
            for v in [-1000, -1, 0, 1, 2, 5000, 1101, 3001, 2049] {
                assert_eq!(code(r.ctl_set(req, v)), c.ctl_set(req, v), "set {req} {v}");
            }
        } else {
            let cv = if req == OPUS_GET_FINAL_RANGE_REQUEST {
                c.final_range().map(|v| v as i32)
            } else {
                c.ctl_get(req)
            };
            assert_eq!(code(r.ctl_get(req)), cv, "get {req}");
        }
    }
    // Typed CTL wrappers.
    assert_eq!(
        code(r.set_bitrate(Bitrate::Bits(100))),
        c.ctl_set(OPUS_SET_BITRATE_REQUEST, 100)
    );
    assert_eq!(r.bitrate(), c.ctl_get(OPUS_GET_BITRATE_REQUEST).unwrap());
    assert_eq!(
        code(r.set_complexity(3)),
        c.ctl_set(OPUS_SET_COMPLEXITY_REQUEST, 3)
    );
    assert_eq!(
        r.complexity(),
        c.ctl_get(OPUS_GET_COMPLEXITY_REQUEST).unwrap()
    );
    assert_eq!(
        code(r.set_force_channels(Some(2))),
        c.ctl_set(OPUS_SET_FORCE_CHANNELS_REQUEST, 2)
    );
    assert_eq!(
        code(r.set_lsb_depth(30)),
        c.ctl_set(OPUS_SET_LSB_DEPTH_REQUEST, 30)
    );
    assert_eq!(
        code(r.set_packet_loss_perc(5)),
        c.ctl_set(OPUS_SET_PACKET_LOSS_PERC_REQUEST, 5)
    );
    assert_eq!(
        code(r.set_inband_fec(1)),
        c.ctl_set(OPUS_SET_INBAND_FEC_REQUEST, 1)
    );
    assert_eq!(
        code(r.set_application(Application::Voip)),
        c.ctl_set(OPUS_SET_APPLICATION_REQUEST, APP_VOIP)
    );
    r.set_expert_frame_duration(FrameSize::Ms10);
    c.ctl_set(OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, 5003)
        .unwrap();
    r.reset();
    c.reset().unwrap();
    run_ms(
        r,
        c,
        48000,
        960,
        6,
        3,
        Input::I16,
        &[4000],
        &[],
        "ctl stream",
    );

    // Creation errors.
    for (ch, s, cp) in [
        (0, 1, 0),
        (256, 1, 0),
        (2, 0, 0),
        (2, 1, 2),
        (2, 2, 1),
        (2, 1, -1),
        (3, 2, 2),
        (255, 255, 1),
    ] {
        let m = vec![0u8; 256];
        assert_eq!(
            code(MsEncoder::new_raw(48000, ch, s, cp, &m, APP_AUDIO)).map(drop),
            oc::MsEnc::new(48000, ch, s, cp, &m, APP_AUDIO).map(drop),
            "{ch} {s} {cp}"
        );
    }
    for fs in [44100, 48000] {
        for app in [0, APP_AUDIO] {
            assert_eq!(
                code(MsEncoder::new_raw(fs, 2, 1, 1, &[0, 1], app)).map(drop),
                oc::MsEnc::new(fs, 2, 1, 1, &[0, 1], app).map(drop)
            );
            for family in [0, 1, 2, 3, 255] {
                for ch in [0, 1, 2, 5, 256] {
                    assert_eq!(
                        code(MsEncoder::new_surround_raw(fs, ch, family, app)).map(drop),
                        oc::MsEnc::new_surround(fs, ch, family, app).map(drop),
                        "fs {fs} app {app} family {family} ch {ch}"
                    );
                }
            }
        }
    }
    assert_eq!(me::ms_encoder_get_size(0, 0), 0);
    assert!(me::ms_encoder_get_size(2, 1) > 0);
    assert_eq!(me::ms_surround_encoder_get_size(3, 0), 0);
    assert!(me::ms_surround_encoder_get_size(6, 1) > 0);
}

#[test]
fn ms_surround_analysis_direct() {
    // surround_analysis through the public surround encoder is covered above; here compare the
    // static helper on its own over long streams (48 kHz mode).
    let mut rng = Rng::new(0x5D00);
    for ch in 3..=8 {
        for fs in [48000, 24000, 16000, 12000, 8000] {
            for ms10 in [25, 50, 100, 200, 400, 600, 1200] {
                let len = fs as usize * ms10 / 10000;
                let sig = program(len * 4, ch, fs, rng.next_u64());
                let mut mem_c = vec![0f32; ch * 120];
                let mut pre_c = vec![0f32; ch];
                let mut ble_c = vec![0f32; 21 * ch];
                let mut mem_r = mem_c.clone();
                let mut pre_r = pre_c.clone();
                let mut ble_r = ble_c.clone();
                for f in 0..4 {
                    let x = &sig[f * len * ch..(f + 1) * len * ch];
                    oc::surround_analysis(
                        x, &mut ble_c, &mut mem_c, &mut pre_c, len as i32, ch as i32, fs, 120,
                    );
                    me::surround_analysis_float(
                        x, &mut ble_r, &mut mem_r, &mut pre_r, len as i32, ch as i32, fs,
                    )
                    .unwrap();
                    assert_bits_eq_f32("bandLogE", &ble_r, &ble_c);
                    assert_bits_eq_f32("mem", &mem_r, &mem_c);
                    assert_bits_eq_f32("preemph", &pre_r, &pre_c);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------------------------

#[test]
fn projection_streams() {
    let mut rng = Rng::new(0x5E00);
    for family in [2, 3] {
        for ch in [1, 2, 3, 4, 5, 6, 8, 9, 11, 16, 18, 25, 27, 36, 38, 49] {
            let fs = [48000, 16000, 24000][rng.range_i32(0, 2) as usize];
            let app = [APP_AUDIO, APP_VOIP, APP_LOWDELAY][rng.range_i32(0, 2) as usize];
            let r = code(ProjectionEncoder::new_ambisonics_raw(fs, ch, family, app));
            let c = oc::ProjEnc::new(fs, ch, family, app);
            let (mut r, mut c) = match (r, c) {
                (Ok(r), Ok(c)) => (r, c),
                (Err(a), Err(b)) => {
                    assert_eq!(a, b, "family {family} ch {ch}");
                    continue;
                }
                (a, b) => panic!("family {family} ch {ch}: {:?} vs {:?}", a.err(), b.err()),
            };
            assert_eq!(
                (r.streams(), r.coupled_streams()),
                (c.streams, c.coupled_streams)
            );
            let size = c
                .ctl_get(OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST)
                .unwrap();
            assert_eq!(r.demixing_matrix_size() as i32, size);
            assert_eq!(
                code(r.ctl_get(OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST)),
                c.ctl_get(OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST)
            );
            assert_eq!(
                r.demixing_matrix(),
                c.demixing_matrix_sized(size as usize).unwrap()
            );
            let mut wrong = vec![0u8; size as usize + 2];
            assert_eq!(
                code(r.write_demixing_matrix(&mut wrong)),
                c.demixing_matrix_sized(size as usize + 2).map(drop)
            );
            if ch > 18 {
                // Orders 4-5: creation and matrices only (encoding is slow).
                continue;
            }
            let br = [AUTO, 32000 * ch, BITRATE_MAX][rng.range_i32(0, 2) as usize];
            assert_eq!(
                code(r.ctl_set(OPUS_SET_BITRATE_REQUEST, br)),
                c.ctl_set(OPUS_SET_BITRATE_REQUEST, br)
            );
            let vbr = rng.range_i32(0, 1);
            assert_eq!(
                code(r.ctl_set(OPUS_SET_VBR_REQUEST, vbr)),
                c.ctl_set(OPUS_SET_VBR_REQUEST, vbr)
            );
            let fsz = frame_sizes(fs)[[3, 2, 4, 5][rng.range_i32(0, 3) as usize]];
            let frames = 8;
            let sig = program(fsz * frames, ch as usize, fs, rng.next_u64());
            let mut out_r = vec![0u8; 30000];
            let mut out_c = vec![0u8; 30000];
            for f in 0..frames {
                let x = &sig[f * fsz * ch as usize..(f + 1) * fsz * ch as usize];
                let input = [Input::I16, Input::I24, Input::F32][f % 3];
                let rr = code(match input {
                    Input::I16 => r.encode(&to_i16(x), fsz, &mut out_r),
                    Input::I24 => r.encode24(&to_i24(x), fsz, &mut out_r),
                    Input::F32 => r.encode_float(x, fsz, &mut out_r),
                });
                let cr = match input {
                    Input::I16 => c.encode(&to_i16(x), fsz, &mut out_c),
                    Input::I24 => c.encode24(&to_i24(x), fsz, &mut out_c),
                    Input::F32 => c.encode_float(x, fsz, &mut out_c),
                };
                assert_eq!(rr, cr, "family {family} ch {ch} frame {f} {input:?}");
                let n = rr.unwrap();
                assert_slice_eq("projection packet", &out_r[..n], &out_c[..n]);
                assert_eq!(
                    code(r.ctl_get(OPUS_GET_FINAL_RANGE_REQUEST)),
                    c.final_range().map(|v| v as i32)
                );
                assert_eq!(r.ms().final_range(), c.final_range().unwrap());
            }
            r.ms_mut().reset();
            c.reset().unwrap();
        }
    }
    assert_eq!(
        opusorus::projection_encoder::projection_ambisonics_encoder_get_size(4, 2),
        0
    );
    assert!(opusorus::projection_encoder::projection_ambisonics_encoder_get_size(4, 3) > 0);
}
