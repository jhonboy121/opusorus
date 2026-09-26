//! Public-API overflow hardening (both builds, overflow-checked test profile).
//!
//! A Rust library must not panic in debug builds on valid API input. Where libopus relies on
//! signed 32-bit arithmetic that overflows (undefined behaviour in C, two's-complement wrap in
//! practice), the port makes exactly those operations wrapping, so debug == release == C. This
//! file holds the regression tests of those sites, each compared bit-exactly with the C oracle
//! (which wraps), and a fuzz-like sweep of public-API configurations that looks for more:
//!
//! * `lfe_stereo_encoder_matches_c`: a *stereo* `Encoder` with `OPUS_SET_LFE(1)` (a
//!   configuration the multistream encoder never creates). In the fixed-point build the CELT
//!   LFE band-energy clamp makes `normalise_bands` scale a band far above unit energy
//!   (`SHL32` wraps), so the mid/side energy sums of `stereo_itheta` overflow `opus_val32`.
//! * `celt_upsampled_full_band_matches_c`: the fixed-point CELT encoder fed up-sampled input
//!   (8–24 kHz) with the end band left at 21: the silent bands above the input bandwidth have
//!   log energies near -14, and the spectral tilt sum of `alloc_trim_analysis` overflows. The
//!   Opus encoder limits the end band to the input bandwidth, so this needs direct CELT use.
//! * `decoder_max_gain_matches_c` / `mult32_32_q16_ovflw_forms_agree`: `OPUS_SET_GAIN` near
//!   ±128 dB in the 24-bit fixed-point build: the decoder gain product does not fit the 32 bits
//!   `MULT32_32_Q16` promises (the 64-bit form wraps in its conversion, the 32-bit form used on
//!   32-bit targets overflows its adds; found by `crates/opusorus/tests/overflow.rs` under
//!   wasmtime).
//! * `public_api_sweep`: random encoders (every rate, channel count, application, CTL, frame
//!   duration, input format and extreme signal: full-scale noise and square waves, DC,
//!   impulses, clipped float input), decoders (random gain, complexity, output format and
//!   frame size, corrupted and random packets, PLC, FEC), surround multistream encoders (with
//!   LFE streams) and projection encoders/decoders, all against the C oracle. Every case runs
//!   under `catch_unwind`, so one run reports all panicking or diverging cases with their
//!   case number. `OPUSORUS_API_SWEEP=<n>` scales the number of cases (default 1),
//!   `OPUSORUS_API_SWEEP_CASE=<kind>:<i>` reruns one case (`enc`, `dec`, `ms`, `proj`).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    clippy::too_many_lines,
    reason = "test code: failures should panic; the sweep reports its failures on stderr"
)]

use opusorus::encoder::request as req;
use opusorus::{Application, Decoder, Encoder, Error, MsEncoder, ProjectionDecoder};
use opusorus::{MsDecoder, ProjectionEncoder};
use opusorus_conformance::{Rng, signals};
use opusorus_oracle::api as c;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Packets encoded and frames decoded by the sweep (both implementations).
static PACKETS: AtomicUsize = AtomicUsize::new(0);
static DECODED: AtomicUsize = AtomicUsize::new(0);

/// Whether a DNN feature (deep PLC, DRED, OSCE) is on.
const DNN: bool = cfg!(any(
    feature = "deep-plc",
    feature = "dred",
    feature = "osce"
));

/// `OPUS_SET_GAIN` (decoder).
const OPUS_SET_GAIN: i32 = 4034;
/// `OPUS_SET_COMPLEXITY` (decoder).
const OPUS_SET_COMPLEXITY: i32 = 4010;
/// `OPUS_SET_PHASE_INVERSION_DISABLED` (decoder).
const OPUS_SET_PHASE_INVERSION_DISABLED: i32 = 4046;
/// `OPUS_SET_IGNORE_EXTENSIONS` (decoder).
const OPUS_SET_IGNORE_EXTENSIONS: i32 = 4058;

fn code<T>(r: Result<T, Error>) -> Result<T, i32> {
    r.map_err(Error::code)
}

/// PCM input format of an encode call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Input {
    I16,
    I24,
    F32,
}

/// `$x.encode_float(pcm, n, out)`, or with `disable-float-api` (no float API in libopus nor
/// the port) `$x.encode24` of the same signal in 24-bit.
macro_rules! encode_f32 {
    ($x:expr, $pcm:expr, $n:expr, $out:expr) => {{
        #[cfg(not(feature = "disable-float-api"))]
        let r = $x.encode_float($pcm, $n, $out);
        #[cfg(feature = "disable-float-api")]
        let r = $x.encode24(&to_i24($pcm), $n, $out);
        r
    }};
}

fn to_i24(x: &[f32]) -> Vec<i32> {
    x.iter()
        .map(|&v| {
            (f64::from(v) * 8_388_608.0)
                .round()
                .clamp(-8_388_608.0, 8_388_607.0) as i32
        })
        .collect()
}

/// Number of signal kinds of [`gen_signal`].
const KINDS: u32 = 13;

/// Test signal `kind` (interleaved, `n` samples per channel). Kinds 2, 3, 6, 8, 9 and 12 are
/// at or beyond full scale; kind 7 exceeds it (clipped by the integer conversions, passed as is
/// to the float API).
fn gen_signal(kind: u32, n: usize, ch: usize, fs: u32, rng: &mut Rng) -> Vec<f32> {
    let seed = rng.next_u64();
    let t = |i: usize| i as f64 / f64::from(fs);
    let tau = 2.0 * core::f64::consts::PI;
    match kind {
        0 => signals::music_like(n, ch, fs, seed),
        1 => signals::speech_like(n, ch, fs, seed),
        2 => signals::noise(n, ch, 1.0, seed),
        3 => {
            // Full-scale square wave, a different period per channel.
            let p = 2 + (seed % 200) as usize;
            (0..n * ch)
                .map(|k| {
                    let (i, c) = (k / ch, k % ch);
                    if (i / (p + c)).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect()
        }
        4 => vec![0.0; n * ch],
        5 => {
            // Sparse full-scale impulses.
            let mut v = vec![0.0; n * ch];
            let mut r = Rng::new(seed);
            for _ in 0..(n / 50).max(1) {
                let k = (r.next_u32() as usize) % (n * ch);
                v[k] = if r.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
            }
            v
        }
        6 => {
            // DC (full scale or a random level).
            let lvl = if seed & 1 == 0 {
                1.0
            } else {
                Rng::new(seed).f32_sym()
            };
            vec![lvl; n * ch]
        }
        7 => signals::music_like(n, ch, fs, seed)
            .iter()
            .map(|v| v * 3.5)
            .collect(),
        8 => {
            // Full-scale sine sweep.
            (0..n * ch)
                .map(|k| {
                    let s = t(k / ch);
                    (tau * (20.0 + 4000.0 * s) * s).sin() as f32
                })
                .collect()
        }
        9 => {
            // Full-scale anti-phase stereo (side only).
            let x = signals::noise(n, 1, 1.0, seed);
            (0..n * ch)
                .map(|k| if k % ch == 0 { x[k / ch] } else { -x[k / ch] })
                .collect()
        }
        10 => signals::noise(n, ch, 1e-4, seed),
        11 => {
            // Low-frequency full-scale sine (LFE-like).
            (0..n * ch)
                .map(|k| (tau * 40.0 * t(k / ch)).sin() as f32)
                .collect()
        }
        _ => {
            // Alternating full-scale samples (energy at Nyquist).
            (0..n * ch)
                .map(|k| {
                    if (k / ch).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect()
        }
    }
}

/// A Rust/C encoder pair.
struct EncPair {
    r: Encoder,
    c: c::Encoder,
    ch: usize,
}

impl EncPair {
    fn new(fs: i32, ch: i32, app: i32) -> Self {
        let r = Encoder::new(fs, ch, Application::from_raw(app).unwrap()).unwrap();
        let c = c::Encoder::new(fs, ch, app).unwrap();
        Self {
            r,
            c,
            ch: ch as usize,
        }
    }

    fn ctl(&mut self, request: i32, value: i32, what: &str) {
        let rr = code(self.r.ctl_set(request, value));
        let cr = self.c.ctl_set(request, value);
        assert_eq!(rr, cr, "{what}: ctl_set({request}, {value})");
    }

    /// Encodes one frame with both encoders; returns the packet (if any).
    fn encode(&mut self, input: Input, pcm: &[f32], n: usize, max: usize, what: &str) -> Vec<u8> {
        let mut out_r = vec![0u8; max];
        let mut out_c = vec![0u8; max];
        let (rr, cr) = match input {
            Input::I16 => {
                let x = signals::to_i16(pcm);
                (
                    code(self.r.encode(&x, n, &mut out_r)),
                    self.c.encode(&x, n, &mut out_c),
                )
            }
            Input::I24 => {
                let x = to_i24(pcm);
                (
                    code(self.r.encode24(&x, n, &mut out_r)),
                    self.c.encode24(&x, n, &mut out_c),
                )
            }
            Input::F32 => (
                code(encode_f32!(self.r, pcm, n, &mut out_r)),
                encode_f32!(self.c, pcm, n, &mut out_c),
            ),
        };
        assert_eq!(rr, cr, "{what}: encode result");
        PACKETS.fetch_add(1, Ordering::Relaxed);
        assert_eq!(
            self.r.final_range(),
            self.c.final_range().unwrap(),
            "{what}: final range"
        );
        match rr {
            Ok(len) => {
                assert_eq!(out_r[..len], out_c[..len], "{what}: packet bytes");
                out_r.truncate(len);
                out_r
            }
            Err(_) => Vec::new(),
        }
    }
}

/// A Rust/C decoder pair.
struct DecPair {
    r: Decoder,
    c: c::Decoder,
    ch: usize,
}

impl DecPair {
    fn new(fs: i32, ch: i32) -> Self {
        Self {
            r: Decoder::new(fs, ch).unwrap(),
            c: c::Decoder::new(fs, ch).unwrap(),
            ch: ch as usize,
        }
    }

    fn ctl(&mut self, request: i32, value: i32, what: &str) {
        let rr = code(self.r.ctl_set(request, value));
        let cr = self.c.ctl_set(request, value);
        assert_eq!(rr, cr, "{what}: decoder ctl_set({request}, {value})");
    }

    /// Decodes with both decoders in `fmt` (0: int16, 1: int24, 2: float) and compares.
    fn decode(&mut self, data: Option<&[u8]>, n: usize, fec: bool, fmt: u32, what: &str) {
        DECODED.fetch_add(1, Ordering::Relaxed);
        let len = n * self.ch;
        match fmt {
            0 => {
                let (mut a, mut b) = (vec![0i16; len], vec![0i16; len]);
                let rr = code(self.r.decode(data, &mut a, n, fec));
                let cr = self.c.decode(data, &mut b, n, fec);
                assert_eq!(rr, cr, "{what}: decode result");
                assert_eq!(a, b, "{what}: int16 output");
            }
            // DISABLE_FLOAT_API: 24-bit output instead of float.
            #[cfg(feature = "disable-float-api")]
            _ => {
                let (mut a, mut b) = (vec![0i32; len], vec![0i32; len]);
                let rr = code(self.r.decode24(data, &mut a, n, fec));
                let cr = self.c.decode24(data, &mut b, n, fec);
                assert_eq!(rr, cr, "{what}: decode24 result");
                assert_eq!(a, b, "{what}: int24 output");
            }
            #[cfg(not(feature = "disable-float-api"))]
            1 => {
                let (mut a, mut b) = (vec![0i32; len], vec![0i32; len]);
                let rr = code(self.r.decode24(data, &mut a, n, fec));
                let cr = self.c.decode24(data, &mut b, n, fec);
                assert_eq!(rr, cr, "{what}: decode24 result");
                assert_eq!(a, b, "{what}: int24 output");
            }
            #[cfg(not(feature = "disable-float-api"))]
            _ => {
                let (mut a, mut b) = (vec![0f32; len], vec![0f32; len]);
                let rr = code(self.r.decode_float(data, &mut a, n, fec));
                let cr = self.c.decode_float(data, &mut b, n, fec);
                assert_eq!(rr, cr, "{what}: decode_float result");
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&a), bits(&b), "{what}: float output");
            }
        }
        assert_eq!(
            self.r.final_range(),
            self.c.final_range().unwrap(),
            "{what}: decoder final range"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Regression tests of the known sites
// ---------------------------------------------------------------------------------------------

/// Stereo `Encoder` with `OPUS_SET_LFE(1)`: bit-exact with C (see the module docs).
#[test]
fn lfe_stereo_encoder_matches_c() {
    let mut rng = Rng::new(0x1fe5_7e2e);
    let mut packets = 0;
    for (k, &kind) in [0u32, 2, 3, 8, 11, 12].iter().enumerate() {
        for &bitrate in &[16_000, 64_000, 256_000] {
            for input in [Input::I16, Input::I24, Input::F32] {
                let what = format!("LFE stereo kind {kind} {bitrate} b/s {input:?}");
                let mut p = EncPair::new(48000, 2, 2049);
                p.ctl(req::OPUS_SET_LFE_REQUEST, 1, &what);
                p.ctl(req::OPUS_SET_BITRATE_REQUEST, bitrate, &what);
                p.ctl(req::OPUS_SET_COMPLEXITY_REQUEST, (k as i32 * 2) % 11, &what);
                let pcm = gen_signal(kind, 960 * 25, 2, 48000, &mut rng);
                let mut dec = DecPair::new(48000, 2);
                for (f, fr) in pcm.chunks(960 * 2).enumerate() {
                    let what = format!("{what} frame {f}");
                    let pkt = p.encode(input, fr, 960, 1500, &what);
                    dec.decode(Some(&pkt), 960, false, f as u32 % 3, &what);
                    packets += 1;
                }
            }
        }
    }
    eprintln!("LFE stereo: {packets} packets bit-exact with C");
}

/// Direct CELT use: up-sampled input with the full end band (see the module docs).
#[test]
fn celt_upsampled_full_band_matches_c() {
    use opusorus::celt::arch::OpusRes;
    use opusorus::celt::celt_encoder::CeltEncoder;
    use opusorus_conformance::float2res;
    use opusorus_oracle::celt_encoder::CeltEnc;
    let mut rng = Rng::new(0xce17_0f11);
    let mut packets = 0;
    for &fs in &[8000, 12000, 16000, 24000] {
        for ch in 1..=2 {
            for (&kind, lm) in [0u32, 2, 3, 7, 10, 12, 8, 7].iter().zip((0..4).cycle()) {
                let frame = (120 << lm) / (48000 / fs) as usize;
                let mut r = CeltEncoder::celt_encoder_init(fs, ch).unwrap();
                r.set_signalling(0);
                let mut c = CeltEnc::new(fs, ch).unwrap();
                // CBR at up to the maximum packet size (the overflow needs a high rate).
                let nb = 200 + (rng.next_u32() % 1076) as i32;
                r.set_bitrate(-1).unwrap();
                assert_eq!(c.ctl(4002, -1), 0);
                let cx = (rng.next_u32() % 11) as i32;
                r.set_complexity(cx).unwrap();
                assert_eq!(c.ctl(4010, cx), 0);
                let pcm: Vec<OpusRes> =
                    gen_signal(kind, frame * 20, ch as usize, fs as u32, &mut rng)
                        .iter()
                        .map(|&v| float2res(v))
                        .collect();
                for (f, fr) in pcm.chunks(frame * ch as usize).enumerate() {
                    let what = format!("CELT fs={fs} ch={ch} kind={kind} frame {f}");
                    let mut out_r = vec![0u8; 1275];
                    let mut out_c = vec![0u8; 1275];
                    #[cfg(feature = "qext")]
                    let rr =
                        r.celt_encode_with_ec(fr, frame as i32, Some(&mut out_r), nb, None, None);
                    #[cfg(not(feature = "qext"))]
                    let rr = r.celt_encode_with_ec(fr, frame as i32, Some(&mut out_r), nb, None);
                    let cr = c.encode(fr, ch as usize, frame as i32, &mut out_c, nb);
                    assert_eq!(rr, cr, "{what}: result");
                    assert!(rr > 0, "{what}: {rr}");
                    assert_eq!(out_r[..rr as usize], out_c[..rr as usize], "{what}: packet");
                    packets += 1;
                }
            }
        }
    }
    eprintln!("CELT up-sampled full band: {packets} packets bit-exact with C");
}

/// Decoders at the extreme `OPUS_SET_GAIN` values (±128 dB): in the 24-bit fixed-point build
/// the gain product `MULT32_32_Q16(pcm, gain)` does not fit 32 bits (see `decoder.rs`).
/// Bit-exact with C for loud CELT, hybrid and SILK packets in every output format.
#[test]
fn decoder_max_gain_matches_c() {
    let mut rng = Rng::new(0x9a1_4a1e);
    let mut frames = 0;
    for (app, bitrate) in [(2049, 128_000), (2048, 32000), (2048, 12000)] {
        let mut e = Encoder::new(48000, 2, Application::from_raw(app).unwrap()).unwrap();
        e.ctl_set(req::OPUS_SET_BITRATE_REQUEST, bitrate).unwrap();
        let pcm = gen_signal(2, 960 * 12, 2, 48000, &mut rng);
        let packets: Vec<Vec<u8>> = pcm
            .chunks(960 * 2)
            .map(|f| {
                let mut buf = vec![0u8; 1500];
                let n = encode_f32!(e, f, 960, &mut buf).unwrap();
                buf.truncate(n);
                buf
            })
            .collect();
        for &gain in &[32767, 30000, -32768] {
            for &(fs, ch) in &[(48000, 2), (48000, 1), (16000, 2)] {
                for fmt in 0..3 {
                    let what = format!("gain {gain} app {app} fs {fs} ch {ch} fmt {fmt}");
                    let mut d = DecPair::new(fs, ch);
                    d.ctl(OPUS_SET_GAIN, gain, &what);
                    let n = fs as usize / 50;
                    for (i, p) in packets.iter().enumerate() {
                        let data = if i % 5 == 4 { None } else { Some(&p[..]) };
                        d.decode(data, n, false, fmt, &format!("{what} packet {i}"));
                        frames += 1;
                    }
                }
            }
        }
    }
    eprintln!("max gain: {frames} frames bit-exact with C");
}

/// The 32-bit (`OPUS_FAST_INT64 = 0`) form of the wrapping `MULT32_32_Q16` used by the decoder
/// gain equals the 64-bit one for all inputs, including products that do not fit 32 bits (the
/// 32-bit form cannot be compared with C on a 64-bit host).
#[cfg(feature = "fixed-point")]
#[test]
fn mult32_32_q16_ovflw_forms_agree() {
    use opusorus::celt::arch::{int32, int64};
    let mut rng = Rng::new(0x3232_1616);
    let edge = [
        0,
        1,
        -1,
        32767,
        -32768,
        65535,
        65536,
        i32::MAX,
        i32::MIN,
        0x7f00_0000,
    ];
    for i in 0..2_000_000u32 {
        let (a, b) = if i < 100 {
            (edge[i as usize % 10], edge[i as usize / 10])
        } else {
            (rng.next_u32() as i32, rng.next_u32() as i32)
        };
        assert_eq!(
            int32::mult32_32_q16_ovflw(a, b),
            int64::mult32_32_q16_ovflw(a, b),
            "MULT32_32_Q16({a}, {b})"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Sweep
// ---------------------------------------------------------------------------------------------

const fn pick<T: Copy>(rng: &mut Rng, v: &[T]) -> T {
    v[rng.range_i32(0, v.len() as i32 - 1) as usize]
}

fn rates() -> Vec<i32> {
    let mut v = vec![8000, 12000, 16000, 24000, 48000, 48000];
    if cfg!(feature = "qext") {
        v.push(96000);
    }
    v
}

const APPS: [i32; 5] = [2048, 2049, 2051, 2052, 2053];

/// A random encoder CTL (valid values, occasionally out of range).
const fn random_ctl(rng: &mut Rng) -> (i32, i32) {
    match rng.range_i32(0, 16) {
        0 => (
            req::OPUS_SET_BITRATE_REQUEST,
            pick(
                rng,
                &[
                    500, 2400, 6000, 9000, 12000, 20000, 32000, 48000, 64000, 96000, 128_000,
                    256_000, 400_000, 510_000, 750_000, -1000, -1,
                ],
            ),
        ),
        1 => (req::OPUS_SET_COMPLEXITY_REQUEST, rng.range_i32(0, 10)),
        2 => (req::OPUS_SET_VBR_REQUEST, rng.range_i32(0, 1)),
        3 => (req::OPUS_SET_VBR_CONSTRAINT_REQUEST, rng.range_i32(0, 1)),
        4 => (
            req::OPUS_SET_FORCE_CHANNELS_REQUEST,
            pick(rng, &[-1000, 1, 2]),
        ),
        5 => (
            req::OPUS_SET_BANDWIDTH_REQUEST,
            pick(rng, &[-1000, 1101, 1102, 1103, 1104, 1105]),
        ),
        6 => (
            req::OPUS_SET_MAX_BANDWIDTH_REQUEST,
            pick(rng, &[1101, 1102, 1103, 1104, 1105]),
        ),
        7 => (
            req::OPUS_SET_SIGNAL_REQUEST,
            pick(rng, &[-1000, 3001, 3002]),
        ),
        8 => (req::OPUS_SET_INBAND_FEC_REQUEST, rng.range_i32(0, 2)),
        9 => (
            req::OPUS_SET_PACKET_LOSS_PERC_REQUEST,
            rng.range_i32(0, 100),
        ),
        10 => (req::OPUS_SET_DTX_REQUEST, rng.range_i32(0, 1)),
        11 => (req::OPUS_SET_LSB_DEPTH_REQUEST, rng.range_i32(8, 24)),
        12 => (
            req::OPUS_SET_PREDICTION_DISABLED_REQUEST,
            rng.range_i32(0, 1),
        ),
        13 => (
            req::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
            rng.range_i32(0, 1),
        ),
        14 => (req::OPUS_SET_LFE_REQUEST, rng.range_i32(0, 1)),
        15 => (
            req::OPUS_SET_FORCE_MODE_REQUEST,
            pick(rng, &[-1000, 1000, 1001, 1002]),
        ),
        _ => (req::OPUS_SET_QEXT_REQUEST, rng.range_i32(0, 1)),
    }
}

/// Frame sizes (samples at `fs`) of 2.5..120 ms and the matching expert frame durations.
const fn random_frame(rng: &mut Rng, fs: i32) -> (usize, i32) {
    let (num, den, dur) = pick(
        rng,
        &[
            (1, 400, 5001),
            (1, 200, 5002),
            (1, 100, 5003),
            (1, 50, 5004),
            (1, 50, 5004),
            (1, 25, 5005),
            (3, 50, 5006),
            (2, 25, 5007),
            (1, 10, 5008),
            (3, 25, 5009),
        ],
    );
    ((fs * num / den) as usize, dur)
}

/// One random encoder case (+ decoding of its packets by a random decoder pair).
fn encoder_case(rng: &mut Rng) -> String {
    let fs = pick(rng, &rates());
    let ch = rng.range_i32(1, 2);
    let app = pick(rng, &APPS);
    let input = pick(rng, &[Input::I16, Input::I24, Input::F32]);
    let kind = rng.range_i32(0, KINDS as i32 - 1) as u32;
    let mut what = format!("enc fs={fs} ch={ch} app={app} {input:?} kind={kind}");
    let mut p = EncPair::new(fs, ch, app);
    for _ in 0..rng.range_i32(0, 8) {
        let (r, v) = random_ctl(rng);
        what += &format!(" {r}={v}");
        p.ctl(r, v, &what);
    }
    let (mut n, dur) = random_frame(rng, fs);
    if rng.range_i32(0, 3) == 0 {
        p.ctl(req::OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, dur, &what);
    }
    let frames = rng.range_i32(4, 14) as usize;
    let maxn = fs as usize * 3 / 25;
    let pcm = gen_signal(kind, maxn * 4, p.ch, fs as u32, rng);
    let dfs = pick(rng, &rates());
    let dch = rng.range_i32(1, 2);
    let mut d = DecPair::new(dfs, dch);
    let mut off = 0;
    for f in 0..frames {
        if rng.range_i32(0, 6) == 0 {
            let (r, v) = random_ctl(rng);
            p.ctl(r, v, &format!("{what} frame {f}"));
        }
        if rng.range_i32(0, 8) == 0 {
            n = random_frame(rng, fs).0;
        }
        let w = format!("{what} frame {f} n={n}");
        let max = pick(rng, &[1500, 1500, 4000, 100, 20]);
        if off + n * p.ch > pcm.len() {
            off = 0;
        }
        let pkt = p.encode(input, &pcm[off..off + n * p.ch], n, max, &w);
        off += n * p.ch;
        let dn = 5760 * dfs as usize / 48000;
        let fmt = rng.range_i32(0, 2) as u32;
        if pkt.is_empty() || rng.range_i32(0, 8) == 0 {
            d.decode(None, dn / 12, false, fmt, &w);
        } else {
            d.decode(Some(&pkt), dn, rng.range_i32(0, 6) == 0, fmt, &w);
        }
    }
    what
}

/// One random decoder case: packets from a Rust encoder, corrupted, random, lost.
fn decoder_case(rng: &mut Rng) -> String {
    let fs = pick(rng, &rates());
    let ch = rng.range_i32(1, 2);
    let gain = pick(rng, &[0, 0, -32768, 32767, 1536, -1536])
        + if rng.range_i32(0, 1) == 0 {
            0
        } else {
            rng.range_i32(-200, 200)
        };
    let gain = gain.clamp(-32768, 32767);
    // With the DNN features, complexity >= 5 runs the deep PLC / OSCE, whose models the C
    // oracle has compiled in and the Rust decoder here does not load (covered by
    // `dnn_integration.rs`): stay below.
    let cx = rng.range_i32(0, 10) % if DNN { 5 } else { 11 };
    let mut what = format!("dec fs={fs} ch={ch} gain={gain} complexity={cx}");
    let mut d = DecPair::new(fs, ch);
    d.ctl(OPUS_SET_GAIN, gain, &what);
    d.ctl(OPUS_SET_COMPLEXITY, cx, &what);
    d.ctl(
        OPUS_SET_PHASE_INVERSION_DISABLED,
        rng.range_i32(0, 1),
        &what,
    );
    d.ctl(OPUS_SET_IGNORE_EXTENSIONS, rng.range_i32(0, 1), &what);
    // Source packets.
    let efs = pick(rng, &rates());
    let ech = rng.range_i32(1, 2);
    let app = pick(rng, &APPS);
    let kind = rng.range_i32(0, KINDS as i32 - 1) as u32;
    what += &format!(" src fs={efs} ch={ech} app={app} kind={kind}");
    let mut e = Encoder::new(efs, ech, Application::from_raw(app).unwrap()).unwrap();
    for _ in 0..rng.range_i32(0, 5) {
        let (r, v) = random_ctl(rng);
        // Invalid values are rejected; the source settings just have to be valid.
        if e.ctl_set(r, v).is_ok() {
            what += &format!(" {r}={v}");
        }
    }
    let (n, _) = random_frame(rng, efs);
    let frames = rng.range_i32(4, 14) as usize;
    let pcm = gen_signal(kind, n * frames, ech as usize, efs as u32, rng);
    let max_n = 5760 * fs as usize / 48000;
    for (f, fr) in pcm.chunks(n * ech as usize).enumerate() {
        let w = format!("{what} packet {f}");
        let mut buf = vec![0u8; 1500];
        let Ok(len) = encode_f32!(e, fr, n, &mut buf) else {
            continue;
        };
        buf.truncate(len);
        match rng.range_i32(0, 9) {
            0 => {
                // Bit flips.
                for _ in 0..rng.range_i32(1, 8) {
                    if buf.is_empty() {
                        break;
                    }
                    let k = (rng.next_u32() as usize) % buf.len();
                    buf[k] ^= 1 << rng.range_i32(0, 7);
                }
            }
            1 => {
                // Random garbage.
                let mut g = vec![0u8; rng.range_i32(1, 1500) as usize];
                rng.fill_bytes(&mut g);
                buf = g;
            }
            2 if buf.len() > 2 => buf.truncate(rng.range_i32(1, buf.len() as i32 - 1) as usize),
            _ => {}
        }
        let fmt = rng.range_i32(0, 2) as u32;
        match rng.range_i32(0, 7) {
            0 => {
                let plc = pick(rng, &[1usize, 2, 4, 8, 12, 24]) * fs as usize / 400;
                d.decode(None, plc.min(max_n), false, fmt, &w);
            }
            1 => d.decode(Some(&buf), max_n, true, fmt, &w),
            _ => d.decode(Some(&buf), max_n, false, fmt, &w),
        }
    }
    what
}

/// One random surround multistream encoder case (family 1 with LFE streams, or 255).
fn ms_case(rng: &mut Rng) -> String {
    let fs = pick(rng, &[16000, 24000, 48000]);
    let family = pick(rng, &[1, 1, 1, 255]);
    let ch = if family == 1 {
        rng.range_i32(1, 8)
    } else {
        rng.range_i32(1, 6)
    };
    let app = pick(rng, &[2048, 2049, 2051]);
    let kind = rng.range_i32(0, KINDS as i32 - 1) as u32;
    let input = pick(rng, &[Input::I16, Input::I24, Input::F32]);
    let mut what = format!("ms fs={fs} family={family} ch={ch} app={app} kind={kind} {input:?}");
    let mut r =
        MsEncoder::new_surround(fs, ch, family, Application::from_raw(app).unwrap()).unwrap();
    let mut c = c::MsEncoder::new_surround(fs, ch, family, app).unwrap();
    assert_eq!(r.mapping(), c.mapping.as_slice(), "{what}: mapping");
    for _ in 0..rng.range_i32(0, 5) {
        let (q, v) = random_ctl(rng);
        if q == req::OPUS_SET_LFE_REQUEST || q == req::OPUS_SET_FORCE_MODE_REQUEST {
            continue;
        }
        what += &format!(" {q}={v}");
        assert_eq!(code(r.ctl_set(q, v)), c.ctl_set(q, v), "{what}: ctl");
    }
    let n = fs as usize / 50;
    let chu = ch as usize;
    let pcm = gen_signal(kind, n * 8, chu, fs as u32, rng);
    let mut rd = MsDecoder::new(fs, ch, r.streams(), r.coupled_streams(), r.mapping()).unwrap();
    for (f, fr) in pcm.chunks(n * chu).enumerate() {
        let w = format!("{what} frame {f}");
        let mut out_r = vec![0u8; 4000];
        let mut out_c = vec![0u8; 4000];
        let (rr, cr) = match input {
            Input::I16 => {
                let x = signals::to_i16(fr);
                (
                    code(r.encode(&x, n, &mut out_r)),
                    c.encode(&x, n, &mut out_c),
                )
            }
            Input::I24 => {
                let x = to_i24(fr);
                (
                    code(r.encode24(&x, n, &mut out_r)),
                    c.encode24(&x, n, &mut out_c),
                )
            }
            Input::F32 => (
                code(encode_f32!(r, fr, n, &mut out_r)),
                encode_f32!(c, fr, n, &mut out_c),
            ),
        };
        assert_eq!(rr, cr, "{w}: result");
        if let Ok(len) = rr {
            assert_eq!(out_r[..len], out_c[..len], "{w}: packet");
            let mut pcm_out = vec![0i16; n * chu];
            // Decoding is checked against C in the decoder cases; here it must not panic.
            code(rd.decode(Some(&out_r[..len]), &mut pcm_out, n, false)).unwrap();
        }
        assert_eq!(r.final_range(), c.final_range().unwrap(), "{w}: range");
    }
    what
}

/// One random projection (ambisonics, family 3) encoder + decoder case.
fn proj_case(rng: &mut Rng) -> String {
    let fs = pick(rng, &[24000, 48000]);
    let ch = pick(rng, &[4, 6, 9, 11, 16]);
    let app = pick(rng, &[2048, 2049]);
    let kind = rng.range_i32(0, KINDS as i32 - 1) as u32;
    let float = rng.range_i32(0, 1) == 1;
    let mut what = format!("proj fs={fs} ch={ch} app={app} kind={kind} float={float}");
    let mut r =
        ProjectionEncoder::new_ambisonics(fs, ch, 3, Application::from_raw(app).unwrap()).unwrap();
    let mut c = c::ProjectionEncoder::new(fs, ch, 3, app).unwrap();
    let bitrate = pick(rng, &[-1000, 32000, 128_000, 512_000]);
    what += &format!(" bitrate={bitrate}");
    assert_eq!(
        code(r.ctl_set(req::OPUS_SET_BITRATE_REQUEST, bitrate)),
        c.ctl_set(req::OPUS_SET_BITRATE_REQUEST, bitrate)
    );
    let matrix = r.demixing_matrix();
    assert_eq!(matrix, c.demixing_matrix().unwrap(), "{what}: matrix");
    let (s, cs) = (c.streams, c.coupled_streams);
    let mut rd = ProjectionDecoder::new(fs, ch, s, cs, &matrix).unwrap();
    let mut cd = c::ProjectionDecoder::new(fs, ch, s, cs, &matrix).unwrap();
    let n = fs as usize / 50;
    let chu = ch as usize;
    let pcm = gen_signal(kind, n * 6, chu, fs as u32, rng);
    for (f, fr) in pcm.chunks(n * chu).enumerate() {
        let w = format!("{what} frame {f}");
        let mut out_r = vec![0u8; 8000];
        let mut out_c = vec![0u8; 8000];
        let (rr, cr) = if float {
            (
                code(encode_f32!(r, fr, n, &mut out_r)),
                encode_f32!(c, fr, n, &mut out_c),
            )
        } else {
            let x = signals::to_i16(fr);
            (
                code(r.encode(&x, n, &mut out_r)),
                c.encode(&x, n, &mut out_c),
            )
        };
        assert_eq!(rr, cr, "{w}: result");
        let Ok(len) = rr else { continue };
        assert_eq!(out_r[..len], out_c[..len], "{w}: packet");
        let (mut a, mut b) = (vec![0i16; n * chu], vec![0i16; n * chu]);
        let dr = code(rd.decode(Some(&out_r[..len]), &mut a, n, false));
        let dc = cd.decode(Some(&out_c[..len]), &mut b, n, false);
        assert_eq!(dr, dc, "{w}: decode result");
        assert_eq!(a, b, "{w}: decoded");
    }
    what
}

/// `OPUSORUS_API_SWEEP` scale factor (default 1).
fn scale() -> usize {
    match std::env::var("OPUSORUS_API_SWEEP") {
        Ok(s) => s.trim().parse().expect("OPUSORUS_API_SWEEP: number"),
        Err(_) => 1,
    }
}

type CaseFn = fn(&mut Rng) -> String;

/// Runs case `i` of `kind` under `catch_unwind`; `Err` describes a panic or divergence.
fn run_case(kind: &str, f: CaseFn, i: usize) -> Result<(), String> {
    let seed = 0xa91_0000_0000 + (i as u64) * 7919 + kind.len() as u64;
    let mut rng = Rng::new(seed);
    match catch_unwind(AssertUnwindSafe(|| f(&mut rng))) {
        Ok(_) => Ok(()),
        Err(e) => {
            let msg = match (e.downcast_ref::<String>(), e.downcast_ref::<&str>()) {
                (Some(s), _) => s.clone(),
                (None, Some(s)) => (*s).to_owned(),
                _ => "panic".to_owned(),
            };
            Err(format!("{kind}:{i}: {msg}"))
        }
    }
}

#[test]
fn public_api_sweep() {
    let kinds: [(&str, CaseFn, usize); 4] = [
        ("enc", encoder_case, 300),
        ("dec", decoder_case, 150),
        ("ms", ms_case, 40),
        ("proj", proj_case, 16),
    ];
    if let Ok(one) = std::env::var("OPUSORUS_API_SWEEP_CASE") {
        let (k, i) = one.split_once(':').expect("<kind>:<index>");
        let (_, f, _) = kinds.iter().find(|(n, ..)| *n == k).expect("kind");
        let i: usize = i.parse().unwrap();
        let mut rng = Rng::new(0xa91_0000_0000 + (i as u64) * 7919 + k.len() as u64);
        eprintln!("{}", f(&mut rng));
        return;
    }
    let scale = scale();
    let mut failures = Vec::new();
    let mut total = 0;
    for (name, f, n) in kinds {
        let n = n * scale;
        let results: Vec<Result<(), String>> = {
            let next = AtomicUsize::new(0);
            let out = std::sync::Mutex::new(Vec::new());
            let workers = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap()
                .min(n.max(1));
            std::thread::scope(|s| {
                for _ in 0..workers {
                    s.spawn(|| {
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= n {
                                break;
                            }
                            let r = run_case(name, f, i);
                            out.lock().unwrap().push(r);
                        }
                    });
                }
            });
            out.into_inner().unwrap()
        };
        total += results.len();
        failures.extend(results.into_iter().filter_map(Result::err));
    }
    for f in &failures {
        eprintln!("FAILED {f}");
    }
    assert!(
        failures.is_empty(),
        "{} of {total} public-API cases panicked or diverged from C (rerun one with \
         OPUSORUS_API_SWEEP_CASE=<kind>:<i>)",
        failures.len()
    );
    eprintln!(
        "public-API sweep: {total} cases ({} encode calls, {} decode calls), no panic, \
         bit-exact with C",
        PACKETS.load(Ordering::Relaxed),
        DECODED.load(Ordering::Relaxed)
    );
}

/// Extreme input: float samples far beyond full scale, infinities and NaN (libopus' float to
/// int conversions of such values are implementation-defined, so this is Rust-only), and
/// random or full-scale 24-bit samples. Nothing may panic, in any build.
#[test]
fn extreme_input_no_panic() {
    let specials = [
        1e30f32,
        -1e30,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
        f32::MIN,
        65536.0,
        -65536.0,
        f32::MIN_POSITIVE,
    ];
    let mut rng = Rng::new(0xe7_7e3e);
    let mut calls = 0;
    for &fs in &rates() {
        for ch in 1..=2 {
            for &app in &APPS {
                let mut e = Encoder::new(fs, ch, Application::from_raw(app).unwrap()).unwrap();
                let mut d = Decoder::new(fs, ch).unwrap();
                let n = fs as usize / 50;
                let chu = ch as usize;
                for f in 0..6 {
                    let (r, v) = random_ctl(&mut rng);
                    // Any result: only panics matter.
                    let mut accepted = e.ctl_set(r, v).is_ok();
                    let x: Vec<f32> = (0..n * chu)
                        .map(|_| match rng.range_i32(0, 3) {
                            0 => specials[rng.range_i32(0, specials.len() as i32 - 1) as usize],
                            1 => rng.f32_sym() * 1e6,
                            _ => rng.f32_sym(),
                        })
                        .collect();
                    let mut buf = vec![0u8; 1500];
                    let len = if f % 2 == 0 {
                        encode_f32!(e, &x, n, &mut buf)
                    } else {
                        // The nominal 24-bit range, at its limits.
                        let y: Vec<i32> = (0..n * chu)
                            .map(|_| match rng.range_i32(0, 2) {
                                0 => pick(&mut rng, &[8_388_607, -8_388_608]),
                                _ => rng.range_i32(-8_388_608, 8_388_607),
                            })
                            .collect();
                        e.encode24(&y, n, &mut buf)
                    };
                    if let Ok(len) = len {
                        #[cfg(not(feature = "disable-float-api"))]
                        let mut out = vec![0f32; n * chu];
                        #[cfg(not(feature = "disable-float-api"))]
                        let r = d.decode_float(Some(&buf[..len]), &mut out, n, false);
                        #[cfg(feature = "disable-float-api")]
                        let mut out = vec![0i32; n * chu];
                        #[cfg(feature = "disable-float-api")]
                        let r = d.decode24(Some(&buf[..len]), &mut out, n, false);
                        accepted &= r.is_ok();
                    }
                    calls += usize::from(accepted);
                }
            }
        }
    }
    eprintln!("extreme input: {calls} fully accepted encode/decode rounds, no panic");
}
