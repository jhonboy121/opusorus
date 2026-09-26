//! Differential tests for unit `analysis` (`src/analysis.c` + the downmix helpers and
//! `is_digital_silence` of `src/opus_encoder.c`) vs the C oracle.
//!
//! Every function is compared bit-for-bit, including the complete `TonalityAnalysisState` after
//! stateful calls.
//!
//! Shared by the float and the fixed-point builds (`fixed-point`, `fixed-res24`, with or
//! without `qext`): the analysis is float in both, but in fixed-point builds its input signal
//! (`opus_val32`: downmix output, `inmem`, the resampler state and energy) is `celt_sig` (`i32`)
//! and `is_digital_silence` takes `opus_res` (`i16`, or `i32` with `fixed-res24`).

use opusorus::analysis::{
    self as a, AnalysisInfo, DETECT_SIZE, DownmixFunc, TonalityAnalysisState,
};
use opusorus::celt::modes::opus_custom_mode_create;
use opusorus::celt::static_modes::CeltMode;
use opusorus_conformance::{Rng, assert_bits_eq_f32, assert_slice_eq, signals};
use opusorus_oracle::analysis::{
    self as c, CAnalysisInfo, CTonalityAnalysisState, Pcm, Res, Val32,
};

/// Bit-exact comparison of `opus_val32` buffers (float bits, or integers in fixed point).
#[cfg(not(feature = "fixed-point"))]
#[track_caller]
fn assert_val32_eq(what: &str, r: &[Val32], c: &[Val32]) {
    assert_bits_eq_f32(what, r, c);
}

/// Bit-exact comparison of `opus_val32` buffers (float bits, or integers in fixed point).
#[cfg(feature = "fixed-point")]
#[track_caller]
fn assert_val32_eq(what: &str, r: &[Val32], c: &[Val32]) {
    assert_slice_eq(what, r, c);
}

/// An `opus_val32` signal sample `x * amp` (float build), with `amp` the float amplitude.
#[cfg(not(feature = "fixed-point"))]
const fn val32(x: f32, amp: f32) -> Val32 {
    x * amp
}

/// An `opus_val32` signal sample (fixed point: `celt_sig`, where 16-bit full scale is
/// `2^(15+SIG_SHIFT)` = `2^27`); `amp` is the float-build amplitude (full scale 32768).
#[cfg(feature = "fixed-point")]
const fn val32(x: f32, amp: f32) -> Val32 {
    (x as f64 * amp as f64 * 4096.0) as i32
}

/// An `opus_res` sample from a float sample (±1.0 full scale).
#[cfg(not(feature = "fixed-point"))]
const fn to_res(x: f32) -> Res {
    x
}

/// An `opus_res` sample from a float sample (±1.0 full scale): 16-bit.
#[cfg(all(feature = "fixed-point", not(feature = "fixed-res24")))]
fn to_res(x: f32) -> Res {
    (x as f64 * 32768.0).round().clamp(-32768.0, 32767.0) as i16
}

/// An `opus_res` sample from a float sample (±1.0 full scale): 24-bit.
#[cfg(feature = "fixed-res24")]
fn to_res(x: f32) -> Res {
    (x as f64 * 8_388_608.0)
        .round()
        .clamp(-8_388_608.0, 8_388_607.0) as i32
}

#[expect(
    clippy::unwrap_used,
    reason = "the 48 kHz / 960 static mode always exists"
)]
fn mode() -> &'static CeltMode {
    opus_custom_mode_create(48000, 960).unwrap()
}

// ---------------------------------------------------------------------------------------------
// State conversion / comparison
// ---------------------------------------------------------------------------------------------

const fn info_to_c(i: &AnalysisInfo) -> CAnalysisInfo {
    CAnalysisInfo {
        valid: i.valid,
        tonality: i.tonality,
        tonality_slope: i.tonality_slope,
        noisiness: i.noisiness,
        activity: i.activity,
        music_prob: i.music_prob,
        music_prob_min: i.music_prob_min,
        music_prob_max: i.music_prob_max,
        bandwidth: i.bandwidth,
        activity_probability: i.activity_probability,
        max_pitch_ratio: i.max_pitch_ratio,
        leak_boost: i.leak_boost,
    }
}

const fn info_from_c(i: &CAnalysisInfo) -> AnalysisInfo {
    AnalysisInfo {
        valid: i.valid,
        tonality: i.tonality,
        tonality_slope: i.tonality_slope,
        noisiness: i.noisiness,
        activity: i.activity,
        music_prob: i.music_prob,
        music_prob_min: i.music_prob_min,
        music_prob_max: i.music_prob_max,
        bandwidth: i.bandwidth,
        activity_probability: i.activity_probability,
        max_pitch_ratio: i.max_pitch_ratio,
        leak_boost: i.leak_boost,
    }
}

const fn info_floats(i: &AnalysisInfo) -> [f32; 9] {
    [
        i.tonality,
        i.tonality_slope,
        i.noisiness,
        i.activity,
        i.music_prob,
        i.music_prob_min,
        i.music_prob_max,
        i.activity_probability,
        i.max_pitch_ratio,
    ]
}

#[track_caller]
fn assert_info_eq(what: &str, r: &AnalysisInfo, c: &CAnalysisInfo) {
    let c = info_from_c(c);
    assert_eq!(r.valid, c.valid, "{what}: valid");
    assert_eq!(r.bandwidth, c.bandwidth, "{what}: bandwidth");
    assert_bits_eq_f32(
        &format!("{what}: floats"),
        &info_floats(r),
        &info_floats(&c),
    );
    assert_slice_eq(&format!("{what}: leak_boost"), &r.leak_boost, &c.leak_boost);
}

fn flat2(x: &[[f32; a::NB_TBANDS]; a::NB_FRAMES]) -> Vec<f32> {
    x.iter().flatten().copied().collect()
}

#[track_caller]
fn assert_state_eq(what: &str, r: &TonalityAnalysisState, c: &CTonalityAnalysisState) {
    let ints = [
        r.application,
        r.fs,
        r.mem_fill,
        r.prev_bandwidth,
        r.e_count,
        r.count,
        r.analysis_offset,
        r.write_pos,
        r.read_pos,
        r.read_subframe,
        r.initialized,
    ];
    let cints = [
        c.application,
        c.fs,
        c.mem_fill,
        c.prev_bandwidth,
        c.e_count,
        c.count,
        c.analysis_offset,
        c.write_pos,
        c.read_pos,
        c.read_subframe,
        c.initialized,
    ];
    assert_eq!(
        ints, cints,
        "{what}: ints [application, fs, mem_fill, prev_bandwidth, E_count, count, \
         analysis_offset, write_pos, read_pos, read_subframe, initialized]"
    );
    let f = |n: &str, x: &[f32], y: &[f32]| assert_bits_eq_f32(&format!("{what}: {n}"), x, y);
    f("angle", &r.angle, &c.angle);
    f("d_angle", &r.d_angle, &c.d_angle);
    f("d2_angle", &r.d2_angle, &c.d2_angle);
    assert_val32_eq(&format!("{what}: inmem"), &r.inmem, &c.inmem);
    f(
        "prev_band_tonality",
        &r.prev_band_tonality,
        &c.prev_band_tonality,
    );
    f("E", &flat2(&r.e), &flat2(&c.e));
    f("logE", &flat2(&r.log_e), &flat2(&c.log_e));
    f("lowE", &r.low_e, &c.low_e);
    f("highE", &r.high_e, &c.high_e);
    f("meanE", &r.mean_e, &c.mean_e);
    f("mem", &r.mem, &c.mem);
    f("cmean", &r.cmean, &c.cmean);
    f("std", &r.std, &c.std);
    f(
        "scalars [prev_tonality, Etracker, lowECount, hp_ener_accum]",
        &[r.prev_tonality, r.e_tracker, r.low_e_count, r.hp_ener_accum],
        &[c.prev_tonality, c.e_tracker, c.low_e_count, c.hp_ener_accum],
    );
    f("rnn_state", &r.rnn_state, &c.rnn_state);
    assert_val32_eq(
        &format!("{what}: downmix_state"),
        &r.downmix_state,
        &c.downmix_state,
    );
    for k in 0..DETECT_SIZE {
        assert_info_eq(&format!("{what}: info[{k}]"), &r.info[k], &c.info[k]);
    }
}

/// Builds a Rust state from a C state.
fn state_from_c(c: &CTonalityAnalysisState) -> TonalityAnalysisState {
    let mut r = TonalityAnalysisState::new(c.fs);
    r.application = c.application;
    r.fs = c.fs;
    r.angle = c.angle;
    r.d_angle = c.d_angle;
    r.d2_angle = c.d2_angle;
    r.inmem = c.inmem;
    r.mem_fill = c.mem_fill;
    r.prev_band_tonality = c.prev_band_tonality;
    r.prev_tonality = c.prev_tonality;
    r.prev_bandwidth = c.prev_bandwidth;
    r.e = c.e;
    r.log_e = c.log_e;
    r.low_e = c.low_e;
    r.high_e = c.high_e;
    r.mean_e = c.mean_e;
    r.mem = c.mem;
    r.cmean = c.cmean;
    r.std = c.std;
    r.e_tracker = c.e_tracker;
    r.low_e_count = c.low_e_count;
    r.e_count = c.e_count;
    r.count = c.count;
    r.analysis_offset = c.analysis_offset;
    r.write_pos = c.write_pos;
    r.read_pos = c.read_pos;
    r.read_subframe = c.read_subframe;
    r.hp_ener_accum = c.hp_ener_accum;
    r.initialized = c.initialized;
    r.rnn_state = c.rnn_state;
    r.downmix_state = c.downmix_state;
    for k in 0..DETECT_SIZE {
        r.info[k] = info_from_c(&c.info[k]);
    }
    r
}

// ---------------------------------------------------------------------------------------------
// PCM helpers
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum PcmBuf {
    Float(Vec<f32>),
    Int16(Vec<i16>),
    Int24(Vec<i32>),
}

impl PcmBuf {
    fn from_float(x: &[f32], ty: u32) -> Self {
        match ty {
            0 => Self::Float(x.to_vec()),
            1 => Self::Int16(signals::to_i16(x)),
            _ => Self::Int24(
                x.iter()
                    .map(|&v| {
                        (v as f64 * 8_388_608.0)
                            .round()
                            .clamp(-8_388_608.0, 8_388_607.0) as i32
                    })
                    .collect(),
            ),
        }
    }
    fn as_c(&self) -> Pcm<'_> {
        match self {
            Self::Float(x) => Pcm::Float(x),
            Self::Int16(x) => Pcm::Int16(x),
            Self::Int24(x) => Pcm::Int24(x),
        }
    }
    const fn len(&self) -> usize {
        match self {
            Self::Float(x) => x.len(),
            Self::Int16(x) => x.len(),
            Self::Int24(x) => x.len(),
        }
    }
    /// Sub-buffer of interleaved samples `[start, end)`.
    fn slice(&self, start: usize, end: usize) -> Self {
        match self {
            Self::Float(x) => Self::Float(x[start..end].to_vec()),
            Self::Int16(x) => Self::Int16(x[start..end].to_vec()),
            Self::Int24(x) => Self::Int24(x[start..end].to_vec()),
        }
    }
}

/// Calls a generic Rust function with the right sample type and downmix callback.
macro_rules! with_pcm {
    ($buf:expr, |$x:ident, $d:ident| $body:expr) => {
        match $buf {
            PcmBuf::Float(v) => {
                let $x: &[f32] = v;
                let $d: DownmixFunc<f32> = a::downmix_float;
                $body
            }
            PcmBuf::Int16(v) => {
                let $x: &[i16] = v;
                let $d: DownmixFunc<i16> = a::downmix_int;
                $body
            }
            PcmBuf::Int24(v) => {
                let $x: &[i32] = v;
                let $d: DownmixFunc<i32> = a::downmix_int24;
                $body
            }
        }
    };
}

/// Linear chirp from `f0` to `f1` Hz over `n` samples.
fn sweep(n: usize, channels: usize, fs: u32, f0: f64, f1: f64, amp: f64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * channels);
    let dur = n as f64 / fs as f64;
    for i in 0..n {
        let t = i as f64 / fs as f64;
        let ph = 2.0 * core::f64::consts::PI * (f0 * t + 0.5 * (f1 - f0) / dur * t * t);
        for ch in 0..channels {
            out.push((amp * (ph + ch as f64 * 0.3).sin()) as f32);
        }
    }
    out
}

/// A mix of several signal types, each lasting `seg` samples, including digital silence,
/// very quiet noise, loud clipping noise, sweeps, music and speech.
fn mixed_signal(n: usize, channels: usize, fs: u32, seed: u64) -> Vec<f32> {
    let seg = fs as usize / 2;
    let mut out = Vec::with_capacity(n * channels);
    let mut k = 0usize;
    let mut rng = Rng::new(seed);
    while out.len() < n * channels {
        let kind = (k + seed as usize) % 8;
        let chunk = match kind {
            0 => signals::music_like(seg, channels, fs, seed + k as u64),
            1 => signals::speech_like(seg, channels, fs, seed + k as u64),
            2 => vec![0.0; seg * channels],
            3 => signals::noise(seg, channels, 0.3, seed + k as u64),
            4 => sweep(seg, channels, fs, 50.0, fs as f64 / 2.0, 0.5),
            5 => signals::noise(seg, channels, 1e-5, seed + k as u64),
            6 => signals::noise(seg, channels, 1.9, seed + k as u64),
            _ => {
                let f = 200.0 + 3000.0 * (rng.f32_sym().abs() as f64);
                sweep(seg, channels, fs, f, f, 0.7)
            }
        };
        out.extend_from_slice(&chunk);
        k += 1;
    }
    out.truncate(n * channels);
    out
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[test]
fn layout_and_constants() {
    assert!(
        c::layout_matches(),
        "repr(C) mirror does not match the C layout"
    );
    assert_eq!(c::m_pi().to_bits(), opusorus::celt::mathops::PI.to_bits());
    let r = TonalityAnalysisState::new(48000);
    let cs = CTonalityAnalysisState::new(48000);
    assert_state_eq("init", &r, &cs);
}

#[test]
fn downmix_functions() {
    let mut rng = Rng::new(0xD0);
    for iter in 0..3000 {
        let ch = rng.range_i32(1, 8);
        let subframe = rng.range_i32(0, 300);
        let offset = rng.range_i32(0, 50);
        let c1 = rng.range_i32(0, ch - 1);
        let c2 = match rng.range_i32(0, 3) {
            0 => -1,
            1 => -2,
            _ => rng.range_i32(0, ch - 1),
        };
        let n = ((subframe + offset) * ch) as usize;
        let scale = [1.0f32, 0.01, 2.5, 1e-6][rng.range_i32(0, 3) as usize];
        let mut x: Vec<f32> = (0..n).map(|_| scale * rng.f32_sym()).collect();
        if iter % 7 == 0 && n > 0 {
            for _ in 0..3 {
                let k = rng.range_i32(0, n as i32 - 1) as usize;
                x[k] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e30]
                    [rng.range_i32(0, 3) as usize];
            }
        }
        let xi16: Vec<i16> = (0..n).map(|_| rng.i16()).collect();
        let xi24: Vec<i32> = (0..n)
            .map(|_| rng.range_i32(-8_388_608, 8_388_607))
            .collect();
        let what = format!("iter {iter} C={ch} sub={subframe} off={offset} c1={c1} c2={c2}");

        let mut yr = vec![Val32::default(); subframe as usize];
        let mut yc = vec![Val32::default(); subframe as usize];
        a::downmix_float(&x, &mut yr, subframe, offset, c1, c2, ch);
        c::downmix(Pcm::Float(&x), &mut yc, subframe, offset, c1, c2, ch);
        assert_val32_eq(&format!("{what} float"), &yr, &yc);

        a::downmix_int(&xi16, &mut yr, subframe, offset, c1, c2, ch);
        c::downmix(Pcm::Int16(&xi16), &mut yc, subframe, offset, c1, c2, ch);
        assert_val32_eq(&format!("{what} int16"), &yr, &yc);

        a::downmix_int24(&xi24, &mut yr, subframe, offset, c1, c2, ch);
        c::downmix(Pcm::Int24(&xi24), &mut yc, subframe, offset, c1, c2, ch);
        assert_val32_eq(&format!("{what} int24"), &yr, &yc);
    }
}

#[test]
fn is_digital_silence() {
    let mut rng = Rng::new(0x51);
    for iter in 0..20000 {
        let ch = rng.range_i32(1, 3);
        let fsz = rng.range_i32(1, 200);
        let lsb = rng.range_i32(8, 24);
        let amp = match rng.range_i32(0, 4) {
            0 => 0.0,
            1 => 1.0 / (1i32 << lsb) as f32,
            2 => 1.0 / (1i32 << rng.range_i32(6, 26)) as f32,
            3 => 1.0,
            _ => 2.0 / (1i32 << lsb) as f32,
        };
        let mut x: Vec<Res> = (0..(fsz * ch) as usize)
            .map(|_| to_res(amp * rng.f32_sym()))
            .collect();
        if rng.range_i32(0, 3) == 0 {
            let k = rng.range_i32(0, x.len() as i32 - 1) as usize;
            x[k] = to_res(if rng.range_i32(0, 1) == 0 { amp } else { -amp });
        }
        assert_eq!(
            a::is_digital_silence(&x, fsz, ch, lsb),
            c::is_digital_silence(&x, fsz, ch, lsb),
            "iter {iter}"
        );
    }
}

#[test]
fn silk_resampler_down2_hp() {
    let mut rng = Rng::new(0x2D);
    #[cfg(feature = "fixed-point")]
    let mut saturated = false;
    for stream in 0..200 {
        let mut sr = [Val32::default(); 3];
        let mut sc = [Val32::default(); 3];
        let amp = [32768.0f32, 1.0, 65536.0, 1e-3][stream % 4];
        for call in 0..20 {
            let len = rng.range_i32(0, 961);
            let x: Vec<Val32> = (0..len).map(|_| val32(rng.f32_sym(), amp)).collect();
            let mut or = vec![Val32::default(); (len / 2) as usize];
            let mut oc = vec![Val32::default(); (len / 2) as usize];
            let er = a::silk_resampler_down2_hp(&mut sr, &mut or, &x, len);
            let ec = c::silk_resampler_down2_hp(&mut sc, &mut oc, &x);
            let what = format!("stream {stream} call {call} len {len}");
            assert_val32_eq(&format!("{what}: out"), &or, &oc);
            assert_val32_eq(&format!("{what}: state"), &sr, &sc);
            assert_val32_eq(&format!("{what}: hp_ener"), &[er], &[ec]);
            #[cfg(feature = "fixed-point")]
            {
                // The 32-bit saturation of the fixed-point energy.
                saturated |= ec == i32::MAX;
            }
        }
    }
    #[cfg(feature = "fixed-point")]
    assert!(saturated, "hp_ener saturation not exercised");
}

#[test]
fn downmix_and_resample() {
    let mut rng = Rng::new(0xDA);
    for stream in 0..300 {
        let fs = [16000, 24000, 48000][stream % 3];
        let ty = (stream / 3 % 3) as u32;
        let ch = rng.range_i32(1, 3);
        let (c1, c2) = match rng.range_i32(0, 2) {
            0 => (0, -1),
            1 => (0, -2),
            _ => (0, ch - 1),
        };
        let mut sr = [Val32::default(); 3];
        let mut sc = [Val32::default(); 3];
        let n_in = fs as usize / 50 * 2;
        let sig = signals::music_like(n_in, ch as usize, fs as u32, stream as u64);
        let pcm = PcmBuf::from_float(&sig, ty);
        for call in 0..10 {
            // subframe / offset are at the 24 kHz analysis rate.
            let subframe = rng.range_i32(0, 480);
            let max_off24 = (n_in as i32 * 24000 / fs) - subframe - 2;
            let offset = rng.range_i32(0, max_off24.max(0));
            let n_out = match fs {
                16000 => (subframe * 2 / 3 * 3 / 2) as usize,
                48000 => subframe as usize,
                _ => subframe as usize,
            };
            let mut yr = vec![Val32::default(); n_out.max(subframe as usize)];
            let mut yc = yr.clone();
            let er = with_pcm!(&pcm, |x, d| a::downmix_and_resample(
                d, x, &mut yr, &mut sr, subframe, offset, c1, c2, ch, fs
            ));
            let ec = c::downmix_and_resample(
                pcm.as_c(),
                &mut yc,
                &mut sc,
                subframe,
                offset,
                c1,
                c2,
                ch,
                fs,
            );
            let what = format!("stream {stream} Fs {fs} ty {ty} call {call} sub {subframe}");
            assert_val32_eq(&format!("{what}: y"), &yr, &yc);
            assert_val32_eq(&format!("{what}: S"), &sr, &sc);
            assert_val32_eq(&format!("{what}: ret"), &[er], &[ec]);
        }
    }
}

#[test]
fn init_and_reset() {
    let mut rng = Rng::new(0x1A);
    let fs = 48000;
    let n = fs as usize;
    let sig = signals::music_like(n, 1, fs as u32, 3);
    let mut r = TonalityAnalysisState::new(fs);
    let mut cs = CTonalityAnalysisState::new(fs);
    r.application = 2049;
    cs.application = 2049;
    for f in 0..(n / 960) {
        let pcm = &sig[f * 960..(f + 1) * 960];
        let mut ir = AnalysisInfo::default();
        a::run_analysis(
            &mut r,
            mode(),
            Some(pcm),
            960,
            960,
            0,
            -2,
            1,
            fs,
            24,
            a::downmix_float,
            &mut ir,
        );
        let ic = c::run_analysis(&mut cs, Some(Pcm::Float(pcm)), 960, 960, 0, -2, 1, fs, 24);
        assert_info_eq(&format!("frame {f}"), &ir, &ic);
        if rng.range_i32(0, 9) == 0 {
            a::tonality_analysis_reset(&mut r);
            c::tonality_analysis_reset(&mut cs);
        }
        assert_state_eq(&format!("frame {f}"), &r, &cs);
    }
    a::tonality_analysis_init(&mut r, 16000);
    c::tonality_analysis_init(&mut cs, 16000);
    assert_state_eq("re-init", &r, &cs);
}

/// Streams a signal through `run_analysis` + (implicitly) `tonality_get_info` frame by frame,
/// like the Opus encoder does, comparing the info every frame and the full state periodically.
#[allow(clippy::too_many_arguments, reason = "test harness with many knobs")]
fn stream_case(
    what: &str,
    sig: &[f32],
    ch: i32,
    fs: i32,
    frame_size: i32,
    lookahead: i32,
    ty: u32,
    c1: i32,
    c2: i32,
    lsb_depth: i32,
    with_null: bool,
    rng: &mut Rng,
) -> Vec<AnalysisInfo> {
    let mut infos = Vec::new();
    let pcm_all = PcmBuf::from_float(sig, ty);
    let total = sig.len() / ch as usize;
    let mut r = TonalityAnalysisState::new(fs);
    let mut cs = CTonalityAnalysisState::new(fs);
    let mut pos = 0usize;
    let mut frame = 0;
    while pos + (frame_size + lookahead) as usize <= total {
        let afs = frame_size + lookahead;
        let buf = pcm_all.slice(pos * ch as usize, (pos + afs as usize) * ch as usize);
        assert_eq!(buf.len(), (afs * ch) as usize);
        // Occasionally call without PCM (NULL analysis_pcm). Each such call permanently moves
        // the read position ahead of the analysis, so only do it once in some streams.
        let no_pcm = with_null && frame == 10 + rng.range_i32(0, 3);
        let mut ir = AnalysisInfo::default();
        let ic = if no_pcm {
            let d: DownmixFunc<f32> = a::downmix_float;
            a::run_analysis::<f32>(
                &mut r,
                mode(),
                None,
                afs,
                frame_size,
                c1,
                c2,
                ch,
                fs,
                lsb_depth,
                d,
                &mut ir,
            );
            c::run_analysis(&mut cs, None, afs, frame_size, c1, c2, ch, fs, lsb_depth)
        } else {
            with_pcm!(&buf, |x, d| a::run_analysis(
                &mut r,
                mode(),
                Some(x),
                afs,
                frame_size,
                c1,
                c2,
                ch,
                fs,
                lsb_depth,
                d,
                &mut ir
            ));
            c::run_analysis(
                &mut cs,
                Some(buf.as_c()),
                afs,
                frame_size,
                c1,
                c2,
                ch,
                fs,
                lsb_depth,
            )
        };
        let w = format!("{what} frame {frame}");
        assert_info_eq(&w, &ir, &ic);
        infos.push(ir);
        if frame % 7 == 0 {
            assert_state_eq(&w, &r, &cs);
        }
        pos += frame_size as usize;
        frame += 1;
    }
    assert_state_eq(&format!("{what} end"), &r, &cs);
    assert!(frame > 0);
    infos
}

#[test]
fn run_analysis_streams() {
    let mut rng = Rng::new(0xA11);
    let mut case = 0u64;
    let mut all = Vec::new();
    for &fs in &[16000, 24000, 48000] {
        // Frame sizes 2.5, 5, 10, 20, 40, 60 ms (the encoder's analysis frames), plus 100/120 ms.
        for &ms10 in &[25, 50, 100, 200, 400, 600, 1000, 1200] {
            let frame_size = fs / 400 * ms10 / 25;
            let secs = if ms10 >= 400 { 3 } else { 2 };
            let n = fs as usize * secs;
            let kind = case % 6;
            let ch = if case.is_multiple_of(2) { 1 } else { 2 };
            let sig = match kind {
                0 => signals::music_like(n, ch, fs as u32, case),
                1 => signals::speech_like(n, ch, fs as u32, case),
                2 => signals::noise(n, ch, 0.2, case),
                3 => sweep(n, ch, fs as u32, 20.0, fs as f64 / 2.0, 0.6),
                4 => mixed_signal(n, ch, fs as u32, case),
                _ => {
                    let mut s = signals::music_like(n, ch, fs as u32, case);
                    // Leading silence then music.
                    for v in s.iter_mut().take(n * ch / 3) {
                        *v = 0.0;
                    }
                    s
                }
            };
            let ty = (case % 3) as u32;
            let (c1, c2) = if ch == 2 {
                [(0, -2), (0, 1), (1, -1)][(case % 3) as usize]
            } else {
                (0, -2)
            };
            // Encoder lookahead: analysis_frame_size may exceed frame_size.
            let lookahead = [0, fs / 400, fs / 100, fs / 50][(case % 4) as usize];
            let lsb = [16, 24, 8, 12][(case % 4) as usize];
            let what = format!(
                "Fs {fs} fsz {frame_size} ch {ch} kind {kind} ty {ty} c1 {c1} c2 {c2} la {lookahead} lsb {lsb}"
            );
            let got = stream_case(
                &what,
                &sig,
                ch as i32,
                fs,
                frame_size,
                lookahead,
                ty,
                c1,
                c2,
                lsb,
                case.is_multiple_of(3),
                &mut rng,
            );
            all.extend(got);
            case += 1;
        }
    }
    // Coverage sanity: the streams must exercise valid analyses with varied decisions.
    let valid = all.iter().filter(|i| i.valid != 0).count();
    assert!(
        valid * 10 > all.len() * 8,
        "too few valid analyses: {valid}/{}",
        all.len()
    );
    let mut bws: Vec<i32> = all.iter().map(|i| i.bandwidth).collect();
    bws.sort_unstable();
    bws.dedup();
    assert!(bws.len() >= 5, "bandwidths seen: {bws:?}");
    assert!(all.iter().any(|i| i.music_prob > 0.8));
    assert!(all.iter().any(|i| i.music_prob < 0.2));
    assert!(all.iter().any(|i| i.leak_boost.iter().any(|&b| b > 0)));
    eprintln!("frames {} valid {valid} bandwidths {bws:?}", all.len());
}

#[test]
fn run_analysis_variable_frames() {
    // Variable frame sizes per call and a multichannel downmix.
    let mut rng = Rng::new(0x7A);
    for (k, &fs) in [48000, 16000, 24000, 48000, 16000, 24000]
        .iter()
        .enumerate()
    {
        let ch: i32 = [1, 2, 3, 6, 2, 4][k];
        let n = fs as usize * 4;
        let sig = mixed_signal(n, ch as usize, fs as u32, 100 + k as u64);
        let ty = (k % 3) as u32;
        let pcm_all = PcmBuf::from_float(&sig, ty);
        let mut r = TonalityAnalysisState::new(fs);
        let mut cs = CTonalityAnalysisState::new(fs);
        let mut pos = 0usize;
        let mut frame = 0;
        let (c1, c2) = if k % 2 == 0 { (0, -2) } else { (ch - 1, 0) };
        loop {
            let frame_size = fs / 400 * [1, 2, 4, 8, 16, 24][rng.range_i32(0, 5) as usize];
            let afs = frame_size + rng.range_i32(0, fs / 50) + 1;
            if pos + afs as usize > n {
                break;
            }
            let buf = pcm_all.slice(pos * ch as usize, (pos + afs as usize) * ch as usize);
            let mut ir = AnalysisInfo::default();
            with_pcm!(&buf, |x, d| a::run_analysis(
                &mut r,
                mode(),
                Some(x),
                afs,
                frame_size,
                c1,
                c2,
                ch,
                fs,
                16,
                d,
                &mut ir
            ));
            let ic = c::run_analysis(
                &mut cs,
                Some(buf.as_c()),
                afs,
                frame_size,
                c1,
                c2,
                ch,
                fs,
                16,
            );
            let w = format!("Fs {fs} ch {ch} frame {frame} fsz {frame_size} afs {afs}");
            assert_info_eq(&w, &ir, &ic);
            if frame % 5 == 0 {
                assert_state_eq(&w, &r, &cs);
            }
            pos += frame_size as usize;
            frame += 1;
        }
        assert_state_eq(&format!("Fs {fs} end"), &r, &cs);
    }
}

#[test]
fn tonality_analysis_direct() {
    // Direct calls with arbitrary len/offset (tonality_analysis is static in C).
    let mut rng = Rng::new(0x70);
    for (k, &fs) in [48000, 24000, 16000, 48000, 24000, 16000]
        .iter()
        .enumerate()
    {
        let ch = 1 + (k % 2) as i32;
        let n = fs as usize * 3;
        let sig = mixed_signal(n, ch as usize, fs as u32, 7 + k as u64);
        let ty = (k % 3) as u32;
        let pcm = PcmBuf::from_float(&sig, ty);
        let mut r = TonalityAnalysisState::new(fs);
        let mut cs = CTonalityAnalysisState::new(fs);
        let mut pos = 0i32;
        let mut call = 0;
        loop {
            let len = rng.range_i32(1, fs / 50);
            if (pos + len) as usize > n {
                break;
            }
            let lsb = rng.range_i32(8, 24);
            with_pcm!(&pcm, |x, d| a::tonality_analysis(
                &mut r,
                mode(),
                x,
                len,
                pos,
                0,
                -2,
                ch,
                lsb,
                d
            ));
            c::tonality_analysis(&mut cs, pcm.as_c(), len, pos, 0, -2, ch, lsb);
            assert_state_eq(&format!("Fs {fs} call {call} len {len} pos {pos}"), &r, &cs);
            pos += len;
            call += 1;
        }
    }
}

#[test]
fn tonality_get_info_random_states() {
    let mut rng = Rng::new(0x6E7);
    for iter in 0..20000 {
        let fs = [8000, 12000, 16000, 24000, 48000][rng.range_i32(0, 4) as usize];
        let mut cs = CTonalityAnalysisState::new(fs);
        for inf in cs.info.iter_mut() {
            inf.valid = i32::from(rng.range_i32(0, 9) != 0);
            inf.tonality = rng.f32_sym().abs();
            inf.tonality_slope = rng.f32_sym();
            inf.noisiness = rng.f32_sym().abs();
            inf.activity = rng.f32_sym().abs();
            inf.music_prob = match rng.range_i32(0, 5) {
                0 => 0.0,
                1 => 1.0,
                _ => rng.f32_sym().abs(),
            };
            inf.bandwidth = rng.range_i32(0, 20);
            inf.activity_probability = match rng.range_i32(0, 5) {
                0 => 0.0,
                1 => 0.05,
                _ => rng.f32_sym().abs(),
            };
            inf.max_pitch_ratio = rng.f32_sym().abs();
            for b in inf.leak_boost.iter_mut() {
                *b = rng.next_u32() as u8;
            }
        }
        cs.write_pos = rng.range_i32(0, 99);
        cs.read_pos = if rng.range_i32(0, 3) == 0 {
            cs.write_pos
        } else {
            rng.range_i32(0, 99)
        };
        cs.read_subframe = rng.range_i32(0, 7);
        cs.count = rng.range_i32(0, 30);
        let mut r = state_from_c(&cs);
        assert_state_eq("setup", &r, &cs);
        for call in 0..4 {
            let len = fs / 400 * rng.range_i32(1, 48);
            let mut ir = AnalysisInfo::default();
            a::tonality_get_info(&mut r, &mut ir, len);
            let ic = c::tonality_get_info(&mut cs, len);
            let w = format!("iter {iter} call {call} Fs {fs} len {len}");
            assert_info_eq(&w, &ir, &ic);
            assert_eq!(
                (r.read_pos, r.read_subframe),
                (cs.read_pos, cs.read_subframe),
                "{w}: read pos"
            );
        }
        // Round-trip conversion sanity.
        let back = info_to_c(&r.info[0]);
        assert_eq!(back, cs.info[0]);
    }
}

#[test]
fn run_analysis_random_state_perturbation() {
    // Start from a warmed-up state, then perturb it randomly (extreme history energies,
    // counts, bandwidths) and check a few more analysis steps.
    let mut rng = Rng::new(0xBEEF);
    let fs = 48000;
    let n = fs as usize * 2;
    let sig = signals::music_like(n, 1, fs as u32, 9);
    let mut base = CTonalityAnalysisState::new(fs);
    for f in 0..50 {
        let pcm = &sig[f * 960..(f + 1) * 960];
        c::run_analysis(&mut base, Some(Pcm::Float(pcm)), 960, 960, 0, -2, 1, fs, 24);
    }
    for iter in 0..200 {
        let mut cs = base.clone();
        cs.count = [0, 1, 2, 3, 5, 6, 9, 10, 99, 10000][rng.range_i32(0, 9) as usize];
        cs.prev_bandwidth = rng.range_i32(0, 20);
        cs.e_tracker = 100.0 * rng.f32_sym();
        cs.low_e_count = rng.f32_sym().abs();
        for b in 0..a::NB_TBANDS {
            cs.low_e[b] = 10.0 * rng.f32_sym();
            cs.high_e[b] = cs.low_e[b] + 20.0 * rng.f32_sym().abs();
            cs.mean_e[b] = rng.f32_sym().abs() * 1e3;
        }
        for v in cs.std.iter_mut() {
            *v = 10.0 * rng.f32_sym().abs();
        }
        for v in cs.rnn_state.iter_mut() {
            *v = rng.f32_sym();
        }
        let mut r = state_from_c(&cs);
        for f in 50..56 {
            let start = f * 960 + rng.range_i32(0, 400) as usize;
            let pcm = &sig[start..start + 960];
            let lsb = rng.range_i32(8, 24);
            let mut ir = AnalysisInfo::default();
            a::run_analysis(
                &mut r,
                mode(),
                Some(pcm),
                960,
                960,
                0,
                -2,
                1,
                fs,
                lsb,
                a::downmix_float,
                &mut ir,
            );
            let ic = c::run_analysis(&mut cs, Some(Pcm::Float(pcm)), 960, 960, 0, -2, 1, fs, lsb);
            let w = format!("iter {iter} frame {f}");
            assert_info_eq(&w, &ir, &ic);
            assert_state_eq(&w, &r, &cs);
        }
    }
}
