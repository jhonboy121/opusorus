//! Differential tests for unit `opus_decoder`: the Opus decoder (`src/opus_decoder.c`), the
//! multistream decoder (`src/opus_multistream_decoder.c`) and the projection decoder
//! (`src/opus_projection_decoder.c`) vs the C library.
//!
//! Every decode call is made on the Rust port and on the C oracle with identical inputs and the
//! complete output buffers (bit patterns for floats), return values, final range values, the
//! private Opus-level decoder state (`OpusDecoder` fields incl. `DecControl` and the soft-clip
//! memory; per stream for multistream/projection) and every CTL getter are compared after every
//! call.
//!
//! Streams come from the C encoder over a large configuration matrix (applications, rates,
//! channels, bitrates, bandwidths, 2.5..120 ms frames incl. code 1/2/3 multi-frame packets,
//! forced mode switches with redundancy frames, in-band FEC, DTX) and are decoded to every output
//! rate/channel count with losses (PLC, FEC), odd PLC sizes, gain/complexity/phase-inversion
//! changes, resets, all three output formats, corrupted/truncated/garbage packets, padding and
//! extensions.
//!
//! The RFC 8251 vectors (`testdata/vectors/rfc8251`, searched upwards from this crate or at
//! `$OPUSORUS_VECTORS`) are decoded like `opus_demo -d` at every rate, mono and stereo:
//! bit-exact vs C, final ranges checked against the `.bit` files and `opus_compare` run against
//! the `.dec` references. With `--features qext`, 96 kHz decoding and the Opus HD vectors
//! (`testdata/vectors/opushd`) are checked the same way with `qext_compare`. Vector tests print a
//! note and pass if the vectors are absent.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "test code: failures should panic; notes about skipped vectors go to stderr"
)]

use opusorus::Error;
use opusorus::decoder::{self as rd, Decoder};
use opusorus::extensions::Extension;
use opusorus::ms_decoder::MsDecoder;
use opusorus::projection_decoder::ProjectionDecoder;
use opusorus::repacketizer::Repacketizer;
use opusorus_conformance::{Rng, signals};
use opusorus_oracle::api;
use opusorus_oracle::opus_decoder as c;
use opusorus_oracle::sys;
use std::path::{Path, PathBuf};

/// `OPUS_SET_FORCE_MODE_REQUEST` (opus_private.h).
const OPUS_SET_FORCE_MODE_REQUEST: i32 = 11002;
const MODE_SILK_ONLY: i32 = 1000;
const MODE_HYBRID: i32 = 1001;
const MODE_CELT_ONLY: i32 = 1002;

#[cfg(feature = "qext")]
const RATES: [i32; 6] = [8000, 12000, 16000, 24000, 48000, 96000];
#[cfg(not(feature = "qext"))]
const RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];

const APPS: [i32; 5] = [
    sys::OPUS_APPLICATION_VOIP,
    sys::OPUS_APPLICATION_AUDIO,
    sys::OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    sys::OPUS_APPLICATION_RESTRICTED_SILK,
    sys::OPUS_APPLICATION_RESTRICTED_CELT,
];

const fn pick<T: Copy>(rng: &mut Rng, v: &[T]) -> T {
    v[rng.range_i32(0, v.len() as i32 - 1) as usize]
}

/// Rust result in C form (count or negative code).
fn rc<T: TryInto<usize>>(r: opusorus::Result<T>) -> Result<usize, i32>
where
    T::Error: core::fmt::Debug,
{
    r.map(|v| v.try_into().unwrap()).map_err(Error::code)
}

// ---------------------------------------------------------------------------------------------
// Single-stream decoder pair
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fmt {
    I16,
    I24,
    F32,
}

const fn rand_fmt(rng: &mut Rng) -> Fmt {
    pick(rng, &[Fmt::I16, Fmt::I16, Fmt::I24, Fmt::F32, Fmt::F32])
}

/// Every GET request `opus_decoder_ctl` implements (default build).
const DEC_GETS: [i32; 9] = [
    sys::OPUS_GET_BANDWIDTH_REQUEST,
    sys::OPUS_GET_COMPLEXITY_REQUEST,
    sys::OPUS_GET_FINAL_RANGE_REQUEST,
    sys::OPUS_GET_SAMPLE_RATE_REQUEST,
    sys::OPUS_GET_PITCH_REQUEST,
    sys::OPUS_GET_GAIN_REQUEST,
    sys::OPUS_GET_LAST_PACKET_DURATION_REQUEST,
    sys::OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
    sys::OPUS_GET_IGNORE_EXTENSIONS_REQUEST,
];

const fn snapshot_ints(s: &rd::DecoderSnapshot) -> [i32; 21] {
    [
        s.channels,
        s.fs,
        s.dec_control.n_channels_api,
        s.dec_control.n_channels_internal,
        s.dec_control.api_sample_rate,
        s.dec_control.internal_sample_rate,
        s.dec_control.payload_size_ms,
        s.dec_control.prev_pitch_lag,
        s.dec_control.enable_deep_plc,
        s.decode_gain,
        s.complexity,
        s.ignore_extensions,
        s.stream_channels,
        s.bandwidth,
        s.mode,
        s.prev_mode,
        s.frame_size,
        s.prev_redundancy,
        s.last_packet_duration,
        s.range_final as i32,
        0,
    ]
}

#[track_caller]
fn assert_state(r: &Decoder, c: &c::DecState, what: &str) {
    let s = r.snapshot();
    assert_eq!(snapshot_ints(&s), c.ints, "{what}: decoder state");
    assert_eq!(
        s.softclip_mem.map(f32::to_bits),
        c.softclip_mem.map(f32::to_bits),
        "{what}: softclip_mem"
    );
}

/// Coverage counters (to make sure the scenarios reach the interesting paths).
#[derive(Debug, Default, Clone, Copy)]
struct Stats {
    calls: usize,
    errors: usize,
    /// Frames ending with a SILK->CELT redundant frame (`prev_redundancy`).
    redundancy: usize,
    /// Decodes where the coding mode changed.
    transitions: usize,
    /// Successful FEC decodes of packets carrying LBRR.
    fec_lbrr: usize,
    /// Calls leaving a non-zero soft-clip memory.
    softclip: usize,
}

impl core::ops::AddAssign for Stats {
    fn add_assign(&mut self, o: Self) {
        self.calls += o.calls;
        self.errors += o.errors;
        self.redundancy += o.redundancy;
        self.transitions += o.transitions;
        self.fec_lbrr += o.fec_lbrr;
        self.softclip += o.softclip;
    }
}

/// A Rust decoder and a C decoder fed identically.
struct Pair {
    r: Decoder,
    c: c::Dec,
    fs: i32,
    ch: usize,
    stats: Stats,
}

impl Pair {
    fn new(fs: i32, ch: i32) -> Self {
        Self {
            r: Decoder::new(fs, ch).unwrap(),
            c: c::Dec::new(fs, ch).unwrap(),
            fs,
            ch: ch as usize,
            stats: Stats::default(),
        }
    }

    /// One decode call on both; compares everything.
    #[track_caller]
    fn step(&mut self, data: Option<&[u8]>, frame_size: i32, fec: i32, fmt: Fmt, what: &str) {
        let before = self.r.snapshot();
        let n = frame_size.max(1) as usize * self.ch + 3;
        let ret = match fmt {
            Fmt::I16 => {
                let mut a = vec![0x5A5Au16 as i16; n];
                let mut b = a.clone();
                let rr = rc(self.r.opus_decode(data, &mut a, frame_size, fec));
                let cr = self.c.decode(data, &mut b, frame_size, fec);
                assert_eq!(rr, cr, "{what}: opus_decode return");
                if a != b {
                    let i = a.iter().zip(&b).position(|(x, y)| x != y).unwrap();
                    panic!("{what}: opus_decode pcm[{i}] rust={} c={}", a[i], b[i]);
                }
                cr
            }
            Fmt::I24 => {
                let mut a = vec![0x5A5A_5A5A; n];
                let mut b = a.clone();
                let rr = rc(self.r.opus_decode24(data, &mut a, frame_size, fec));
                let cr = self.c.decode24(data, &mut b, frame_size, fec);
                assert_eq!(rr, cr, "{what}: opus_decode24 return");
                if a != b {
                    let i = a.iter().zip(&b).position(|(x, y)| x != y).unwrap();
                    panic!("{what}: opus_decode24 pcm[{i}] rust={} c={}", a[i], b[i]);
                }
                cr
            }
            Fmt::F32 => {
                let mut a = vec![1234.5f32; n];
                let mut b = a.clone();
                let rr = rc(self.r.opus_decode_float(data, &mut a, frame_size, fec));
                let cr = self.c.decode_float(data, &mut b, frame_size, fec);
                assert_eq!(rr, cr, "{what}: opus_decode_float return");
                if let Some(i) = a
                    .iter()
                    .zip(&b)
                    .position(|(x, y)| x.to_bits() != y.to_bits())
                {
                    panic!(
                        "{what}: opus_decode_float pcm[{i}] rust={} c={}",
                        a[i], b[i]
                    );
                }
                cr
            }
        };
        self.check(what);
        let after = self.r.snapshot();
        let st = &mut self.stats;
        st.calls += 1;
        match ret {
            Err(_) => st.errors += 1,
            Ok(_) => {
                st.redundancy += usize::from(after.prev_redundancy != 0);
                st.transitions +=
                    usize::from(before.prev_mode > 0 && after.prev_mode != before.prev_mode);
                st.softclip += usize::from(after.softclip_mem != [0.0; 2]);
                if fec == 1
                    && let Some(d) = data
                    && opusorus::packet::has_lbrr(d) == Ok(true)
                {
                    st.fec_lbrr += 1;
                }
            }
        }
    }

    #[track_caller]
    fn check(&mut self, what: &str) {
        assert_state(&self.r, &self.c.dump(), what);
        for req in DEC_GETS {
            assert_eq!(
                rc(self.r.ctl_get(req).map(|v| v as u32 as usize)),
                self.c.ctl_get(req).map(|v| v as u32 as usize),
                "{what}: ctl_get({req})"
            );
        }
        assert_eq!(
            self.r.final_range(),
            self.c.final_range().unwrap(),
            "{what}: final range"
        );
    }

    #[track_caller]
    fn set(&mut self, req: i32, value: i32, what: &str) {
        let rr = self.r.ctl_set(req, value).map_err(Error::code);
        let cr = self.c.ctl_set(req, value);
        assert_eq!(rr, cr, "{what}: ctl_set({req}, {value})");
        self.check(what);
    }

    #[track_caller]
    fn reset(&mut self, what: &str) {
        self.r.reset();
        self.c.reset().unwrap();
        self.check(what);
    }

    #[track_caller]
    fn nb_samples(&self, pkt: &[u8], what: &str) -> Result<usize, i32> {
        let rr = rc(self.r.nb_samples(pkt));
        let cr = if pkt.is_empty() {
            // C reads nothing for len < 1.
            Err(sys::OPUS_BAD_ARG)
        } else {
            self.c.nb_samples(pkt)
        };
        assert_eq!(rr, cr, "{what}: nb_samples");
        cr
    }
}

// ---------------------------------------------------------------------------------------------
// Stream generation (C encoder)
// ---------------------------------------------------------------------------------------------

/// A test signal: speech-like, music-like, noise and silence segments.
fn test_signal(fs: i32, ch: usize, secs: f32, seed: u64) -> Vec<i16> {
    let n = (fs as f32 * secs) as usize;
    let seg = (fs / 2) as usize;
    let mut out = Vec::with_capacity(n * ch);
    let mut rng = Rng::new(seed);
    let mut k = 0;
    while out.len() < n * ch {
        let len = seg.min(n - out.len() / ch);
        let kind = (rng.next_u32() % 5 + k) % 5;
        let x: Vec<f32> = match kind {
            0 => signals::speech_like(len, ch, fs as u32, rng.next_u64()),
            1 => signals::music_like(len, ch, fs as u32, rng.next_u64()),
            2 => signals::noise(len, ch, 0.3, rng.next_u64()),
            3 => vec![0.0; len * ch],
            _ => signals::music_like(len, ch, fs as u32, rng.next_u64())
                .iter()
                .map(|v| (v * 3.0).clamp(-1.0, 1.0))
                .collect(),
        };
        out.extend(signals::to_i16(&x));
        k += 1;
    }
    out
}

#[derive(Debug, Clone, Copy)]
struct EncCfg {
    fs: i32,
    ch: i32,
    app: i32,
    nframes: usize,
    /// Randomize CTLs frame by frame (bitrate, forced mode, bandwidth, frame size, FEC, DTX…).
    dynamic: bool,
    /// Initial frame size in samples at `fs`.
    frame: i32,
    bitrate: i32,
    fec: bool,
    dtx: bool,
    qext: bool,
}

fn frame_sizes(fs: i32) -> Vec<i32> {
    [1, 2, 4, 8, 16, 24, 32, 40, 48]
        .iter()
        .map(|k| fs / 400 * k)
        .collect()
}

/// Encodes a stream with the C encoder; returns the packets.
fn encode_stream(cfg: &EncCfg, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = Rng::new(seed);
    let mut enc = api::Encoder::new(cfg.fs, cfg.ch, cfg.app).unwrap();
    enc.ctl_set(sys::OPUS_SET_BITRATE_REQUEST, cfg.bitrate)
        .unwrap();
    enc.ctl_set(sys::OPUS_SET_COMPLEXITY_REQUEST, rng.range_i32(0, 10))
        .unwrap();
    if cfg.fec {
        enc.ctl_set(sys::OPUS_SET_INBAND_FEC_REQUEST, rng.range_i32(1, 2))
            .unwrap();
        enc.ctl_set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, rng.range_i32(5, 40))
            .unwrap();
    }
    if cfg.dtx {
        enc.ctl_set(sys::OPUS_SET_DTX_REQUEST, 1).unwrap();
    }
    #[cfg(feature = "qext")]
    if cfg.qext {
        enc.ctl_set(sys::OPUS_SET_QEXT_REQUEST, 1).unwrap();
    }
    let _ = cfg.qext;
    let sig = test_signal(cfg.fs, cfg.ch as usize, 4.0, seed ^ 0xABCD);
    let total = sig.len() / cfg.ch as usize;
    let sizes = frame_sizes(cfg.fs);
    let mut frame = cfg.frame;
    let mut pos = 0usize;
    let mut out = Vec::with_capacity(cfg.nframes);
    let mut buf = vec![0u8; 20000];
    let restricted = cfg.app == sys::OPUS_APPLICATION_RESTRICTED_SILK
        || cfg.app == sys::OPUS_APPLICATION_RESTRICTED_CELT;
    for _ in 0..cfg.nframes {
        if cfg.dynamic {
            let r = rng.range_i32(0, 99);
            if r < 12 {
                let br = pick(
                    &mut rng,
                    &[
                        6000,
                        8000,
                        12000,
                        16000,
                        20000,
                        24000,
                        32000,
                        48000,
                        64000,
                        96000,
                        128000,
                        256000,
                        510000,
                        sys::OPUS_AUTO,
                        sys::OPUS_BITRATE_MAX,
                    ],
                );
                enc.ctl_set(sys::OPUS_SET_BITRATE_REQUEST, br).unwrap();
            } else if r < 22 && !restricted {
                let m = pick(
                    &mut rng,
                    &[MODE_SILK_ONLY, MODE_HYBRID, MODE_CELT_ONLY, sys::OPUS_AUTO],
                );
                enc.ctl_set(OPUS_SET_FORCE_MODE_REQUEST, m).unwrap();
            } else if r < 30 {
                let bw = pick(
                    &mut rng,
                    &[
                        sys::OPUS_AUTO,
                        sys::OPUS_BANDWIDTH_NARROWBAND,
                        sys::OPUS_BANDWIDTH_MEDIUMBAND,
                        sys::OPUS_BANDWIDTH_WIDEBAND,
                        sys::OPUS_BANDWIDTH_SUPERWIDEBAND,
                        sys::OPUS_BANDWIDTH_FULLBAND,
                    ],
                );
                enc.ctl_set(sys::OPUS_SET_BANDWIDTH_REQUEST, bw).unwrap();
            } else if r < 38 {
                frame = pick(&mut rng, &sizes);
            } else if r < 42 {
                enc.ctl_set(sys::OPUS_SET_INBAND_FEC_REQUEST, rng.range_i32(0, 2))
                    .unwrap();
                enc.ctl_set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, rng.range_i32(0, 50))
                    .unwrap();
            } else if r < 45 {
                enc.ctl_set(sys::OPUS_SET_DTX_REQUEST, rng.range_i32(0, 1))
                    .unwrap();
            } else if r < 49 {
                let s = pick(
                    &mut rng,
                    &[
                        sys::OPUS_AUTO,
                        sys::OPUS_SIGNAL_VOICE,
                        sys::OPUS_SIGNAL_MUSIC,
                    ],
                );
                enc.ctl_set(sys::OPUS_SET_SIGNAL_REQUEST, s).unwrap();
            } else if r < 52 && cfg.ch == 2 {
                let fc = pick(&mut rng, &[sys::OPUS_AUTO, 1, 2]);
                enc.ctl_set(sys::OPUS_SET_FORCE_CHANNELS_REQUEST, fc)
                    .unwrap();
            } else if r < 55 {
                enc.ctl_set(sys::OPUS_SET_VBR_REQUEST, rng.range_i32(0, 1))
                    .unwrap();
            }
        }
        let n = frame as usize;
        if pos + n > total {
            pos = 0;
        }
        let pcm = &sig[pos * cfg.ch as usize..(pos + n) * cfg.ch as usize];
        pos += n;
        match enc.encode(pcm, n, &mut buf) {
            Ok(len) => out.push(buf[..len].to_vec()),
            // Frame sizes the application cannot use (e.g. < 10 ms with RESTRICTED_SILK).
            Err(e) => {
                assert_eq!(e, sys::OPUS_BAD_ARG);
                frame = cfg.fs / 50;
            }
        }
    }
    out
}

/// Loss/decode scenario knobs.
#[derive(Debug, Clone, Copy)]
struct Scen {
    loss_pct: i32,
    ctl_changes: bool,
}

/// Decodes `packets` on a pair with a random loss/FEC/PLC scenario.
fn decode_stream(p: &mut Pair, packets: &[Vec<u8>], rng: &mut Rng, scen: Scen, tag: &str) {
    let max_fs = p.fs / 25 * 3;
    let f2_5 = p.fs / 400;
    let mut i = 0;
    while i < packets.len() {
        let pkt = &packets[i];
        let what = format!("{tag} pkt {i} (fs {} ch {})", p.fs, p.ch);
        let dur = p.nb_samples(pkt, &what).unwrap() as i32;
        let fmt = rand_fmt(rng);
        let roll = rng.range_i32(0, 99);
        if scen.ctl_changes && rng.range_i32(0, 99) < 4 {
            match rng.range_i32(0, 5) {
                0 => {
                    let g = pick(rng, &[0, 0, -2000, -256, 256, 1500, 6000, -32768, 32767]);
                    p.set(sys::OPUS_SET_GAIN_REQUEST, g, &what);
                }
                1 => p.set(
                    sys::OPUS_SET_COMPLEXITY_REQUEST,
                    rng.range_i32(0, 10),
                    &what,
                ),
                2 => p.set(
                    sys::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
                    rng.range_i32(0, 1),
                    &what,
                ),
                3 => p.set(
                    sys::OPUS_SET_IGNORE_EXTENSIONS_REQUEST,
                    rng.range_i32(0, 1),
                    &what,
                ),
                4 if rng.range_i32(0, 3) == 0 => p.reset(&what),
                _ => {}
            }
        }
        if roll < scen.loss_pct {
            // Lost packet: FEC from the next packet, or PLC.
            if i + 1 < packets.len() && rng.range_i32(0, 1) == 0 {
                p.step(Some(&packets[i + 1]), dur, 1, fmt, &what);
            } else {
                let fsz = if rng.range_i32(0, 3) == 0 {
                    // Conceal more than one packet's worth at once.
                    (dur * rng.range_i32(1, 3)).min(max_fs * 2)
                } else {
                    dur
                };
                p.step(None, fsz, 0, fmt, &what);
            }
        } else if roll < scen.loss_pct + 3 {
            // Odd concealment sizes (multiples of 2.5 ms), then the packet.
            p.step(None, f2_5 * rng.range_i32(1, 60), 0, fmt, &what);
            p.step(Some(pkt), max_fs, 0, fmt, &what);
        } else if roll < scen.loss_pct + 5 {
            // Too small / odd frame sizes: BUFFER_TOO_SMALL / BAD_ARG paths.
            let fsz = rng.range_i32(-1, dur);
            p.step(Some(pkt), fsz, 0, fmt, &what);
            p.step(Some(pkt), dur, 0, fmt, &what);
        } else {
            let fsz = match rng.range_i32(0, 3) {
                0 => dur,
                1 => max_fs,
                _ => dur + rng.range_i32(0, max_fs - dur.min(max_fs)),
            };
            p.step(Some(pkt), fsz, 0, fmt, &what);
        }
        i += 1;
    }
}

/// Decoder output configurations for a stream: the encoder's own plus random ones.
fn dec_configs(rng: &mut Rng, enc_fs: i32, enc_ch: i32, n: usize) -> Vec<(i32, i32)> {
    let mut v = vec![(enc_fs.min(48000), enc_ch)];
    for _ in 0..n {
        v.push((pick(rng, &RATES), rng.range_i32(1, 2)));
    }
    v
}

fn run_matrix(seed: u64, n_streams: usize, dynamic: bool, tag: &str) -> Stats {
    let mut rng = Rng::new(seed);
    let mut total = Stats::default();
    for s in 0..n_streams {
        let fs = pick(&mut rng, &[8000, 12000, 16000, 24000, 48000]);
        let ch = rng.range_i32(1, 2);
        let app = APPS[s % APPS.len()];
        let sizes = frame_sizes(fs);
        let cfg = EncCfg {
            fs,
            ch,
            app,
            nframes: rng.range_i32(20, 60) as usize,
            dynamic,
            frame: pick(&mut rng, &sizes),
            bitrate: pick(
                &mut rng,
                &[
                    6000, 10000, 16000, 24000, 32000, 48000, 64000, 128000, 300000,
                ],
            ),
            fec: rng.range_i32(0, 2) == 0,
            dtx: rng.range_i32(0, 4) == 0,
            qext: false,
        };
        let packets = encode_stream(&cfg, rng.next_u64());
        for (dfs, dch) in dec_configs(&mut rng, fs, ch, 2) {
            let mut p = Pair::new(dfs, dch);
            let scen = Scen {
                loss_pct: pick(&mut rng, &[0, 5, 15, 30]),
                ctl_changes: true,
            };
            decode_stream(
                &mut p,
                &packets,
                &mut rng,
                scen,
                &format!("{tag} stream {s} {cfg:?}"),
            );
            total += p.stats;
        }
    }
    eprintln!("{tag}: {total:?}");
    assert!(total.errors > 0 && total.transitions > 0 && total.fec_lbrr > 0);
    total
}

#[test]
fn static_configs_matrix() {
    run_matrix(1, 90, false, "static");
}

#[test]
fn dynamic_configs_matrix() {
    let st = run_matrix(2, 90, true, "dynamic");
    assert!(st.redundancy > 0);
}

/// Forced SILK <-> hybrid <-> CELT switches every few frames (redundancy frames in both
/// directions, transitions with and without redundancy, hybrid->SILK CELT fade-out), at every
/// bandwidth, with occasional losses around the switches.
#[test]
fn mode_transitions() {
    let mut rng = Rng::new(3);
    let mut total = Stats::default();
    for s in 0..48 {
        let fs = pick(&mut rng, &[16000, 24000, 48000]);
        let ch = rng.range_i32(1, 2);
        let mut enc = api::Encoder::new(fs, ch, pick(&mut rng, &APPS[..3])).unwrap();
        enc.ctl_set(
            sys::OPUS_SET_BITRATE_REQUEST,
            pick(&mut rng, &[12000, 24000, 40000, 64000]),
        )
        .unwrap();
        if rng.range_i32(0, 1) == 0 {
            enc.ctl_set(sys::OPUS_SET_INBAND_FEC_REQUEST, 1).unwrap();
            enc.ctl_set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, 20)
                .unwrap();
        }
        let sig = test_signal(fs, ch as usize, 3.0, rng.next_u64());
        let mut packets = Vec::new();
        let mut buf = vec![0u8; 8000];
        let mut pos = 0usize;
        let frame_opts = [fs / 100, fs / 50, fs / 25, fs / 50 * 3];
        let mut frame = fs / 50;
        for k in 0..70 {
            if k % rng.range_i32(1, 4) as usize == 0 {
                let m = pick(&mut rng, &[MODE_SILK_ONLY, MODE_HYBRID, MODE_CELT_ONLY]);
                enc.ctl_set(OPUS_SET_FORCE_MODE_REQUEST, m).unwrap();
                let bw = if m == MODE_SILK_ONLY {
                    pick(
                        &mut rng,
                        &[
                            sys::OPUS_BANDWIDTH_NARROWBAND,
                            sys::OPUS_BANDWIDTH_MEDIUMBAND,
                            sys::OPUS_BANDWIDTH_WIDEBAND,
                        ],
                    )
                } else if m == MODE_HYBRID {
                    pick(
                        &mut rng,
                        &[
                            sys::OPUS_BANDWIDTH_SUPERWIDEBAND,
                            sys::OPUS_BANDWIDTH_FULLBAND,
                        ],
                    )
                } else {
                    pick(
                        &mut rng,
                        &[
                            sys::OPUS_AUTO,
                            sys::OPUS_BANDWIDTH_NARROWBAND,
                            sys::OPUS_BANDWIDTH_WIDEBAND,
                            sys::OPUS_BANDWIDTH_FULLBAND,
                        ],
                    )
                };
                enc.ctl_set(sys::OPUS_SET_BANDWIDTH_REQUEST, bw).unwrap();
                frame = if m == MODE_CELT_ONLY && rng.range_i32(0, 2) == 0 {
                    pick(&mut rng, &[fs / 400, fs / 200, fs / 100])
                } else {
                    pick(&mut rng, &frame_opts)
                };
            }
            let n = frame as usize;
            if (pos + n) * ch as usize > sig.len() {
                pos = 0;
            }
            let len = enc
                .encode(
                    &sig[pos * ch as usize..(pos + n) * ch as usize],
                    n,
                    &mut buf,
                )
                .unwrap();
            pos += n;
            packets.push(buf[..len].to_vec());
        }
        for (dfs, dch) in dec_configs(&mut rng, fs, ch, 2) {
            let mut p = Pair::new(dfs, dch);
            let scen = Scen {
                loss_pct: pick(&mut rng, &[0, 0, 8, 20]),
                ctl_changes: rng.range_i32(0, 1) == 0,
            };
            decode_stream(
                &mut p,
                &packets,
                &mut rng,
                scen,
                &format!("transitions {s}"),
            );
            total += p.stats;
        }
    }
    eprintln!("transitions: {total:?}");
    assert!(total.transitions > 500 && total.redundancy > 100);
}

/// In-band FEC: SILK/hybrid streams with LBRR, decoded with systematic losses recovered from the
/// next packet (FEC frame sizes equal to, larger than and smaller than the packet), incl.
/// multi-frame (40/60 ms) SILK packets.
#[test]
fn fec_and_plc() {
    let mut rng = Rng::new(4);
    let mut total = Stats::default();
    for s in 0..40 {
        let fs = pick(&mut rng, &[8000, 12000, 16000, 24000, 48000]);
        let ch = rng.range_i32(1, 2);
        let cfg = EncCfg {
            fs,
            ch,
            app: pick(
                &mut rng,
                &[
                    sys::OPUS_APPLICATION_VOIP,
                    sys::OPUS_APPLICATION_RESTRICTED_SILK,
                ],
            ),
            nframes: 50,
            dynamic: false,
            frame: pick(&mut rng, &[fs / 100, fs / 50, fs / 25, fs / 50 * 3]),
            bitrate: pick(&mut rng, &[12000, 20000, 32000]),
            fec: true,
            dtx: false,
            qext: false,
        };
        let packets = encode_stream(&cfg, rng.next_u64());
        for (dfs, dch) in dec_configs(&mut rng, fs, ch, 1) {
            let mut p = Pair::new(dfs, dch);
            let what = format!("fec {s} {cfg:?} dec {dfs}/{dch}");
            let max_fs = dfs / 25 * 3;
            for i in 0..packets.len() {
                let dur = p.nb_samples(&packets[i], &what).unwrap() as i32;
                let fmt = rand_fmt(&mut rng);
                match rng.range_i32(0, 9) {
                    0..=2 if i + 1 < packets.len() => {
                        let fsz = match rng.range_i32(0, 3) {
                            0 => dur,
                            1 => (dur + dfs / 400 * rng.range_i32(1, 8)).min(max_fs),
                            2 => (dur - dfs / 400).max(dfs / 400),
                            _ => dfs / 400 * rng.range_i32(1, 48),
                        };
                        p.step(Some(&packets[i + 1]), fsz, 1, fmt, &what);
                    }
                    3 => {
                        p.step(None, dur, 0, fmt, &what);
                    }
                    _ => {
                        p.step(Some(&packets[i]), max_fs, 0, fmt, &what);
                    }
                }
            }
            total += p.stats;
        }
    }
    eprintln!("fec: {total:?}");
    assert!(total.fec_lbrr > 300);
}

/// DTX streams (1-byte TOC-only packets during silence) and runs of losses.
#[test]
fn dtx_and_long_losses() {
    let mut rng = Rng::new(5);
    for s in 0..24 {
        let fs = pick(&mut rng, &[8000, 16000, 48000]);
        let ch = rng.range_i32(1, 2);
        let cfg = EncCfg {
            fs,
            ch,
            app: pick(&mut rng, &APPS),
            nframes: 80,
            dynamic: false,
            frame: pick(&mut rng, &[fs / 100, fs / 50, fs / 25]),
            bitrate: pick(&mut rng, &[10000, 24000, 64000]),
            fec: false,
            dtx: true,
            qext: false,
        };
        let packets = encode_stream(&cfg, rng.next_u64());
        for (dfs, dch) in dec_configs(&mut rng, fs, ch, 1) {
            let mut p = Pair::new(dfs, dch);
            let what = format!("dtx {s} {cfg:?} dec {dfs}/{dch}");
            let mut i = 0;
            while i < packets.len() {
                let fmt = rand_fmt(&mut rng);
                if rng.range_i32(0, 19) == 0 {
                    // A burst of losses (PLC runs of up to ~1 s).
                    for _ in 0..rng.range_i32(1, 8) {
                        p.step(None, dfs / 400 * rng.range_i32(1, 60), 0, fmt, &what);
                    }
                }
                p.step(Some(&packets[i]), dfs / 25 * 3, 0, fmt, &what);
                i += 1;
            }
        }
    }
}

/// PLC before any packet, decode after reset, gain extremes with soft clipping.
#[test]
fn plc_start_gain_softclip() {
    let mut rng = Rng::new(6);
    let mut total = Stats::default();
    for &fs in &RATES {
        for ch in 1..=2 {
            let mut p = Pair::new(fs, ch);
            let what = format!("start {fs}/{ch}");
            // Before any packet: zeros.
            for k in 1..6 {
                p.step(None, fs / 400 * k, 0, rand_fmt(&mut rng), &what);
            }
            p.step(Some(&[]), fs / 50, 0, Fmt::I16, &what);
            // Loud music at full gain: exercises the int16 soft clipper and its memory.
            let enc_fs = fs.min(48000);
            let cfg = EncCfg {
                fs: enc_fs,
                ch,
                app: sys::OPUS_APPLICATION_AUDIO,
                nframes: 40,
                dynamic: true,
                frame: enc_fs / 50,
                bitrate: 96000,
                fec: false,
                dtx: false,
                qext: false,
            };
            let packets = encode_stream(&cfg, rng.next_u64());
            p.set(sys::OPUS_SET_GAIN_REQUEST, 32767, &what);
            for (i, pkt) in packets.iter().enumerate() {
                if i % 10 == 5 {
                    p.set(
                        sys::OPUS_SET_GAIN_REQUEST,
                        pick(&mut rng, &[-32768, 0, 3000, 32767]),
                        &what,
                    );
                }
                p.step(Some(pkt), fs / 25 * 3, 0, rand_fmt(&mut rng), &what);
                if i == 20 {
                    p.reset(&what);
                    p.step(None, fs / 50, 0, Fmt::I16, &what);
                }
            }
            total += p.stats;
        }
    }
    eprintln!("softclip: {total:?}");
    assert!(total.softclip > 50);
}

/// Corrupted, truncated and garbage packets; every return code must match C.
#[test]
fn corrupt_and_garbage_packets() {
    let mut rng = Rng::new(7);
    let mut total = Stats::default();
    // Real packets to mutate.
    let mut pool: Vec<Vec<u8>> = Vec::new();
    for s in 0..12 {
        let fs = pick(&mut rng, &[16000, 48000]);
        let cfg = EncCfg {
            fs,
            ch: rng.range_i32(1, 2),
            app: APPS[s % 5],
            nframes: 30,
            dynamic: true,
            frame: fs / 50,
            bitrate: 32000,
            fec: s % 2 == 0,
            dtx: false,
            qext: false,
        };
        pool.extend(encode_stream(&cfg, rng.next_u64()));
    }
    for d in 0..24 {
        let fs = RATES[d % RATES.len()];
        let ch = (d % 2 + 1) as i32;
        let mut p = Pair::new(fs, ch);
        let max_fs = fs / 25 * 3;
        for k in 0..400 {
            let what = format!("garbage dec {fs}/{ch} #{k}");
            let base = pick(&mut rng, &pool.iter().collect::<Vec<_>>()).clone();
            let pkt: Vec<u8> = match rng.range_i32(0, 7) {
                0 => {
                    // Random bytes.
                    let mut v = vec![0u8; rng.range_i32(1, 300) as usize];
                    rng.fill_bytes(&mut v);
                    v
                }
                1 => {
                    // Truncated.
                    let n = rng.range_i32(1, base.len() as i32) as usize;
                    base[..n].to_vec()
                }
                2 => {
                    // Bit flips in the payload.
                    let mut v = base.clone();
                    for _ in 0..rng.range_i32(1, 8) {
                        let i = rng.range_i32(0, v.len() as i32 - 1) as usize;
                        v[i] ^= 1 << rng.range_i32(0, 7);
                    }
                    v
                }
                3 => {
                    // Random TOC with a real payload.
                    let mut v = base.clone();
                    v[0] = rng.next_u32() as u8;
                    v
                }
                4 => {
                    // Code 3 headers with random counts/padding flags.
                    let mut v = vec![(rng.next_u32() as u8) | 3, rng.next_u32() as u8];
                    let mut tail = vec![0u8; rng.range_i32(0, 400) as usize];
                    rng.fill_bytes(&mut tail);
                    v.extend(tail);
                    v
                }
                5 => {
                    // Extended with random bytes.
                    let mut v = base.clone();
                    let mut tail = vec![0u8; rng.range_i32(1, 50) as usize];
                    rng.fill_bytes(&mut tail);
                    v.extend(tail);
                    v
                }
                6 => vec![rng.next_u32() as u8],
                _ => base,
            };
            let fmt = rand_fmt(&mut rng);
            if let Err(e) = p.nb_samples(&pkt, &what) {
                assert!(e < 0);
            }
            let fsz = if rng.range_i32(0, 5) == 0 {
                rng.range_i32(-5, max_fs)
            } else {
                max_fs
            };
            let fec = if rng.range_i32(0, 9) == 0 {
                rng.range_i32(-1, 2)
            } else {
                0
            };
            p.step(Some(&pkt), fsz, fec, fmt, &what);
            if rng.range_i32(0, 9) == 0 {
                p.step(None, fs / 400 * rng.range_i32(0, 10), 0, fmt, &what);
            }
        }
        total += p.stats;
    }
    eprintln!("garbage: {total:?}");
    assert!(total.errors > 1000 && total.calls - total.errors > 1000);
}

/// Packets re-emitted with padding and random extensions (incl. repeated extensions and, with
/// QEXT, garbage QEXT payloads), decoded with `ignore_extensions` on and off.
#[test]
fn padding_and_extensions() {
    let mut rng = Rng::new(8);
    for s in 0..16 {
        let fs = pick(&mut rng, &[16000, 48000]);
        let ch = rng.range_i32(1, 2);
        let cfg = EncCfg {
            fs,
            ch,
            app: APPS[s % 5],
            nframes: 30,
            dynamic: true,
            frame: pick(&mut rng, &[fs / 100, fs / 50, fs / 25]),
            bitrate: pick(&mut rng, &[16000, 64000]),
            fec: false,
            dtx: false,
            qext: false,
        };
        let packets = encode_stream(&cfg, rng.next_u64());
        for &ignore in &[0, 1] {
            let (dfs, dch) = (pick(&mut rng, &RATES), rng.range_i32(1, 2));
            let mut p = Pair::new(dfs, dch);
            p.set(sys::OPUS_SET_IGNORE_EXTENSIONS_REQUEST, ignore, "ext");
            for (i, pkt) in packets.iter().enumerate() {
                let what = format!("ext {s} pkt {i} ignore {ignore}");
                let nb_frames = opusorus::packet::get_nb_frames(pkt).unwrap();
                let mut payloads: Vec<Vec<u8>> = Vec::new();
                let n_ext = rng.range_i32(0, 4);
                for _ in 0..n_ext {
                    let mut v = vec![0u8; rng.range_i32(0, 40) as usize];
                    rng.fill_bytes(&mut v);
                    payloads.push(v);
                }
                let exts: Vec<Extension<'_>> = payloads
                    .iter()
                    .map(|d| {
                        let id = pick(&mut rng, &[3, 5, 31, 33, 100, 124, 124, 127]);
                        Extension {
                            id,
                            frame: rng.range_i32(0, nb_frames - 1),
                            data: if id < 32 { &d[..d.len().min(1)] } else { d },
                        }
                    })
                    .collect();
                let mut rp = Repacketizer::new();
                rp.cat(pkt).unwrap();
                let mut out = vec![0u8; pkt.len() + 400];
                let pad = rng.range_i32(0, 1) == 1;
                let Ok(n) = rp.out_range_impl(0, nb_frames, &mut out, false, pad, &exts) else {
                    continue;
                };
                let data = &out[..n as usize];
                p.step(Some(data), dfs / 25 * 3, 0, rand_fmt(&mut rng), &what);
            }
        }
    }
}

/// Numeric CTL dispatch: every implemented request, invalid values and unknown requests.
#[test]
fn ctl_requests() {
    let mut p = Pair::new(48000, 2);
    let sets = [
        (sys::OPUS_SET_COMPLEXITY_REQUEST, [-1, 0, 5, 10, 11]),
        (
            sys::OPUS_SET_GAIN_REQUEST,
            [-32769, -32768, 0, 32767, 32768],
        ),
        (
            sys::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
            [-1, 0, 1, 2, 0],
        ),
        (sys::OPUS_SET_IGNORE_EXTENSIONS_REQUEST, [-1, 0, 1, 2, 0]),
    ];
    for (req, vals) in sets {
        for v in vals {
            p.set(req, v, "ctl");
        }
    }
    p.set(sys::OPUS_RESET_STATE, 0, "ctl reset");
    // Unknown requests (not handled by the default decoder build) return UNIMPLEMENTED
    // without reading their argument.
    let unknown: Vec<i32> = (3990..4070)
        .chain([5120, 5122, 6001, 11002, 0, -1, 1 << 20])
        .filter(|r| {
            !DEC_GETS.contains(r)
                && ![
                    sys::OPUS_SET_COMPLEXITY_REQUEST,
                    sys::OPUS_SET_GAIN_REQUEST,
                    sys::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
                    sys::OPUS_SET_IGNORE_EXTENSIONS_REQUEST,
                    sys::OPUS_RESET_STATE,
                ]
                .contains(r)
        })
        .collect();
    for &req in &unknown {
        let rr = p.r.ctl_set(req, 0).map_err(Error::code);
        assert_eq!(rr, p.c.ctl_set(req, 0), "ctl_set unknown {req}");
        let rr = p.r.ctl_get(req).map_err(Error::code);
        assert_eq!(rr, p.c.ctl_get(req), "ctl_get unknown {req}");
    }
    // Typed setters/getters agree with the numeric ones.
    let mut d = Decoder::new(24000, 1).unwrap();
    d.set_gain(-300).unwrap();
    assert_eq!(d.gain(), -300);
    assert_eq!(d.set_gain(40000), Err(Error::BadArg));
    d.set_complexity(7).unwrap();
    assert_eq!(d.ctl_get(rd::OPUS_GET_COMPLEXITY_REQUEST), Ok(7));
    d.set_phase_inversion_disabled(true);
    assert!(d.phase_inversion_disabled());
    d.set_ignore_extensions(true);
    assert!(d.ignore_extensions());
    assert_eq!(d.sample_rate(), 24000);
    assert_eq!(d.bandwidth(), None);
    assert_eq!(d.channels(), 1);
    assert_eq!(d.ctl_set(rd::OPUS_GET_GAIN_REQUEST, 0), Err(Error::BadArg));
    assert_eq!(d.ctl_get(rd::OPUS_SET_GAIN_REQUEST), Err(Error::BadArg));
    // Creation errors.
    for fs in [0, 8000, 11025, 44100, 96000, 192000, -1] {
        for ch in [-1, 0, 1, 2, 3] {
            let r = Decoder::new(fs, ch).map(drop).map_err(Error::code);
            let c = c::Dec::new(fs, ch).map(drop);
            assert_eq!(r, c, "create {fs}/{ch}");
        }
    }
    // Typed API on a real stream (usize/bool wrappers).
    let cfg = EncCfg {
        fs: 48000,
        ch: 2,
        app: sys::OPUS_APPLICATION_AUDIO,
        nframes: 10,
        dynamic: false,
        frame: 960,
        bitrate: 64000,
        fec: false,
        dtx: false,
        qext: false,
    };
    let packets = encode_stream(&cfg, 99);
    let mut d = Decoder::new(48000, 2).unwrap();
    let mut cd = c::Dec::new(48000, 2).unwrap();
    for pkt in &packets {
        let mut a = vec![0f32; 1920];
        let mut b = vec![0f32; 1920];
        assert_eq!(d.decode_float(Some(pkt), &mut a, 960, false), Ok(960));
        assert_eq!(cd.decode_float(Some(pkt), &mut b, 960, 0), Ok(960));
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(d.nb_samples(pkt), Ok(960));
        assert_eq!(d.last_packet_duration(), 960);
        assert!(d.bandwidth().is_some());
    }
    // A too-short output buffer is BadArg (C would overflow).
    let mut small = vec![0i16; 100];
    assert_eq!(
        d.decode(Some(&packets[0]), &mut small, 960, false),
        Err(Error::BadArg)
    );
    assert_eq!(d.decode(None, &mut small, 960, false), Err(Error::BadArg));
    // Re-init.
    d.init(16000, 1).unwrap();
    assert_eq!(d.sample_rate(), 16000);
    assert_eq!(d.init(44100, 1), Err(Error::BadArg));
    assert!(Decoder::get_size(1) > 0 && Decoder::get_size(3) == 0);
}

// ---------------------------------------------------------------------------------------------
// Multistream
// ---------------------------------------------------------------------------------------------

/// GET requests handled by `opus_multistream_decoder_ctl`.
const MS_GETS: [i32; 7] = [
    sys::OPUS_GET_BANDWIDTH_REQUEST,
    sys::OPUS_GET_SAMPLE_RATE_REQUEST,
    sys::OPUS_GET_GAIN_REQUEST,
    sys::OPUS_GET_LAST_PACKET_DURATION_REQUEST,
    sys::OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
    sys::OPUS_GET_COMPLEXITY_REQUEST,
    sys::OPUS_GET_FINAL_RANGE_REQUEST,
];

struct MsPair {
    r: MsDecoder,
    c: c::MsDec,
    ch: usize,
    streams: i32,
}

impl MsPair {
    #[track_caller]
    fn check(&mut self, what: &str) {
        for s in 0..self.streams {
            let d = self.c.stream_dump(s);
            assert_state(
                self.r.decoder_state(s).unwrap(),
                &d,
                &format!("{what} stream {s}"),
            );
            let rr = rc(self
                .r
                .decoder_state(s)
                .unwrap()
                .ctl_get(sys::OPUS_GET_PITCH_REQUEST));
            assert_eq!(
                rr,
                self.c
                    .stream_ctl_get(s, sys::OPUS_GET_PITCH_REQUEST)
                    .map(|v| v as usize)
            );
        }
        for req in MS_GETS {
            assert_eq!(
                rc(self.r.ctl_get(req).map(|v| v as u32 as usize)),
                self.c.ctl_get(req).map(|v| v as u32 as usize),
                "{what}: ms ctl_get({req})"
            );
        }
        assert_eq!(self.r.final_range(), self.c.final_range().unwrap());
    }

    #[track_caller]
    fn step(&mut self, data: Option<&[u8]>, frame_size: i32, fec: i32, fmt: Fmt, what: &str) {
        let n = frame_size.clamp(1, 11520) as usize * self.ch + 5;
        match fmt {
            Fmt::I16 => {
                let mut a = vec![0x5A5Au16 as i16; n];
                let mut b = a.clone();
                let rr = rc(self
                    .r
                    .opus_multistream_decode(data, &mut a, frame_size, fec));
                let cr = self.c.decode(data, &mut b, frame_size, fec);
                assert_eq!(rr, cr, "{what}: ms decode return");
                assert!(a == b, "{what}: ms decode pcm");
            }
            Fmt::I24 => {
                let mut a = vec![0x5A5A_5A5A; n];
                let mut b = a.clone();
                let rr = rc(self
                    .r
                    .opus_multistream_decode24(data, &mut a, frame_size, fec));
                let cr = self.c.decode24(data, &mut b, frame_size, fec);
                assert_eq!(rr, cr, "{what}: ms decode24 return");
                assert!(a == b, "{what}: ms decode24 pcm");
            }
            Fmt::F32 => {
                let mut a = vec![1234.5f32; n];
                let mut b = a.clone();
                let rr = rc(self
                    .r
                    .opus_multistream_decode_float(data, &mut a, frame_size, fec));
                let cr = self.c.decode_float(data, &mut b, frame_size, fec);
                assert_eq!(rr, cr, "{what}: ms decode_float return");
                assert!(
                    a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{what}: ms decode_float pcm"
                );
            }
        }
        self.check(what);
    }

    #[track_caller]
    fn set(&mut self, req: i32, v: i32, what: &str) {
        assert_eq!(
            self.r.ctl_set(req, v).map_err(Error::code),
            self.c.ctl_set(req, v),
            "{what}: ms ctl_set({req}, {v})"
        );
        self.check(what);
    }
}

/// Random decoder mapping for `streams`/`coupled` with `channels` outputs (incl. silent 255
/// channels and duplicated coded channels).
fn random_mapping(rng: &mut Rng, channels: i32, streams: i32, coupled: i32) -> Vec<u8> {
    let nc = streams + coupled;
    (0..channels)
        .map(|i| {
            if rng.range_i32(0, 9) == 0 {
                255
            } else if rng.range_i32(0, 3) == 0 {
                rng.range_i32(0, nc - 1) as u8
            } else {
                (i % nc) as u8
            }
        })
        .collect()
}

fn ms_decode_stream(p: &mut MsPair, packets: &[Vec<u8>], fs: i32, rng: &mut Rng, tag: &str) {
    let max_fs = fs / 25 * 3;
    for (i, pkt) in packets.iter().enumerate() {
        let what = format!("{tag} pkt {i}");
        let fmt = rand_fmt(rng);
        match rng.range_i32(0, 19) {
            0 => p.step(None, fs / 50, 0, fmt, &what),
            1 if i + 1 < packets.len() => p.step(Some(&packets[i + 1]), fs / 50, 1, fmt, &what),
            2 => {
                let n = rng.range_i32(1, pkt.len() as i32) as usize;
                p.step(Some(&pkt[..n]), max_fs, 0, fmt, &what);
            }
            3 => {
                let mut v = pkt.clone();
                let k = rng.range_i32(0, v.len() as i32 - 1) as usize;
                v[k] ^= 1 << rng.range_i32(0, 7);
                p.step(Some(&v), max_fs, 0, fmt, &what);
            }
            4 => p.step(Some(pkt), rng.range_i32(-1, fs / 50), 0, fmt, &what),
            5 => p.set(
                sys::OPUS_SET_GAIN_REQUEST,
                rng.range_i32(-3000, 3000),
                &what,
            ),
            6 => p.set(
                sys::OPUS_SET_COMPLEXITY_REQUEST,
                rng.range_i32(-1, 11),
                &what,
            ),
            7 => p.set(
                sys::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
                rng.range_i32(0, 2),
                &what,
            ),
            8 if rng.range_i32(0, 4) == 0 => p.set(sys::OPUS_RESET_STATE, 0, &what),
            _ => p.step(Some(pkt), max_fs, 0, fmt, &what),
        }
    }
}

/// Multistream: C multistream (families 0, 1, 255 and random layouts) encoder packets decoded
/// with the encoder's mapping and with random mappings (silent channels, duplicates), with
/// losses, FEC, truncation, bit flips and CTLs.
#[test]
fn multistream_decode() {
    let mut rng = Rng::new(9);
    for s in 0..40 {
        let fs = pick(&mut rng, &[8000, 16000, 24000, 48000]);
        let app = pick(&mut rng, &APPS[..3]);
        let mut enc = match s % 4 {
            0 => api::MsEncoder::new_surround(fs, rng.range_i32(1, 2), 0, app).unwrap(),
            1 => api::MsEncoder::new_surround(fs, rng.range_i32(1, 8), 1, app).unwrap(),
            2 => api::MsEncoder::new_surround(fs, rng.range_i32(1, 10), 255, app).unwrap(),
            _ => {
                let streams = rng.range_i32(1, 5);
                let coupled = rng.range_i32(0, streams);
                // The encoder needs every coded channel mapped at least once.
                let nc = streams + coupled;
                let channels = nc + rng.range_i32(0, 3);
                let mut mapping: Vec<u8> = (0..channels)
                    .map(|i| {
                        if i < nc {
                            i as u8
                        } else if rng.range_i32(0, 1) == 0 {
                            255
                        } else {
                            rng.range_i32(0, nc - 1) as u8
                        }
                    })
                    .collect();
                for i in (1..mapping.len()).rev() {
                    let j = rng.range_i32(0, i as i32) as usize;
                    mapping.swap(i, j);
                }
                api::MsEncoder::new(fs, channels, streams, coupled, &mapping, app).unwrap()
            }
        };
        let (streams, coupled) = (enc.streams, enc.coupled_streams);
        let enc_ch = enc.mapping.len();
        enc.ctl_set(
            sys::OPUS_SET_BITRATE_REQUEST,
            pick(&mut rng, &[16000, 48000, 128000, 256000]) * streams,
        )
        .unwrap();
        if rng.range_i32(0, 1) == 0 {
            enc.ctl_set(sys::OPUS_SET_INBAND_FEC_REQUEST, 1).unwrap();
            enc.ctl_set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, 20)
                .unwrap();
        }
        let frame = pick(&mut rng, &[fs / 100, fs / 50, fs / 25, fs / 50 * 3]);
        let sig = test_signal(fs, enc_ch, 2.0, rng.next_u64());
        let mut packets = Vec::new();
        let mut buf = vec![0u8; 40000];
        let mut pos = 0;
        for _ in 0..25 {
            let n = frame as usize;
            if (pos + n) * enc_ch > sig.len() {
                pos = 0;
            }
            let len = enc
                .encode(&sig[pos * enc_ch..(pos + n) * enc_ch], n, &mut buf)
                .unwrap();
            pos += n;
            packets.push(buf[..len].to_vec());
        }
        for k in 0..3 {
            let dfs = if k == 0 { fs } else { pick(&mut rng, &RATES) };
            let (channels, mapping) = if k == 0 {
                (enc_ch as i32, enc.mapping.clone())
            } else {
                let c = rng.range_i32(1, 12);
                (c, random_mapping(&mut rng, c, streams, coupled))
            };
            let r = MsDecoder::new(dfs, channels, streams, coupled, &mapping).unwrap();
            let c = c::MsDec::new(dfs, channels, streams, coupled, &mapping).unwrap();
            let mut p = MsPair {
                r,
                c,
                ch: channels as usize,
                streams,
            };
            p.check("init");
            ms_decode_stream(
                &mut p,
                &packets,
                dfs,
                &mut rng,
                &format!("ms {s}/{k} fs {dfs} {streams}+{coupled} map {mapping:?}"),
            );
        }
    }
}

/// Multistream creation/validation errors, decoder_state bounds and unknown CTLs.
#[test]
fn multistream_api_errors() {
    let mut rng = Rng::new(10);
    for _ in 0..500 {
        let fs = pick(&mut rng, &[8000, 44100, 48000, 96000, 0]);
        let channels = rng.range_i32(-1, 257);
        let streams = rng.range_i32(-1, 256);
        let coupled = rng.range_i32(-1, 256);
        let mut mapping = vec![0u8; 260];
        for m in &mut mapping {
            *m = match rng.range_i32(0, 3) {
                0 => 255,
                1 => rng.next_u32() as u8,
                _ => rng.range_i32(0, (streams + coupled).clamp(1, 255) - 1) as u8,
            };
        }
        let r = MsDecoder::new(fs, channels, streams, coupled, &mapping)
            .map(drop)
            .map_err(Error::code);
        let c = c::MsDec::new(fs, channels, streams, coupled, &mapping).map(drop);
        assert_eq!(
            r, c,
            "ms create fs {fs} ch {channels} s {streams} c {coupled}"
        );
        assert_eq!(
            MsDecoder::get_size(streams, coupled) == 0,
            streams < 1 || coupled > streams || coupled < 0
        );
    }
    let mut r = MsDecoder::new(48000, 3, 2, 1, &[0, 1, 2]).unwrap();
    let mut c = c::MsDec::new(48000, 3, 2, 1, &[0, 1, 2]).unwrap();
    for id in [-1, 0, 1, 2, 100] {
        assert_eq!(
            r.decoder_state(id).map(drop).map_err(Error::code),
            match c.stream_state_ret(id) {
                0 => Ok(()),
                e => Err(e),
            }
        );
    }
    for req in (3990..4070).chain([5120, 11002, -1]) {
        if MS_GETS.contains(&req)
            || [
                sys::OPUS_SET_GAIN_REQUEST,
                sys::OPUS_SET_COMPLEXITY_REQUEST,
                sys::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
                sys::OPUS_RESET_STATE,
            ]
            .contains(&req)
        {
            continue;
        }
        assert_eq!(
            r.ctl_set(req, 0).map_err(Error::code),
            c.ctl_set(req, 0),
            "{req}"
        );
        assert_eq!(r.ctl_get(req).map_err(Error::code), c.ctl_get(req), "{req}");
    }
    // Short mapping / output buffer guards.
    assert_eq!(
        MsDecoder::new(48000, 3, 2, 1, &[0, 1]).map(drop),
        Err(Error::BadArg)
    );
    let mut small = vec![0f32; 10];
    assert_eq!(
        r.decode_float(None, &mut small, 960, false),
        Err(Error::BadArg)
    );
    assert_eq!(r.channels(), 3);
    assert_eq!(r.streams(), 2);
    assert_eq!(r.coupled_streams(), 1);
}

// ---------------------------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------------------------

/// Projection (family 3) packets from the C ambisonics encoder, orders 1..3 (with and without
/// the non-diegetic stereo pair), decoded with the encoder's demixing matrix; plus family 2
/// (ambisonics through the plain multistream surround encoder/decoder).
#[test]
fn projection_decode() {
    let mut rng = Rng::new(11);
    for (s, &channels) in [4, 6, 9, 11, 16, 18, 4, 9, 16].iter().enumerate() {
        let fs = pick(&mut rng, &[16000, 24000, 48000]);
        let app = pick(&mut rng, &APPS[..3]);
        let mut enc = api::ProjectionEncoder::new(fs, channels, 3, app).unwrap();
        let (streams, coupled) = (enc.streams, enc.coupled_streams);
        let matrix = enc.demixing_matrix().unwrap();
        enc.ctl_set(
            sys::OPUS_SET_BITRATE_REQUEST,
            pick(&mut rng, &[64000, 256000, 512000]),
        )
        .unwrap();
        let frame = pick(&mut rng, &[fs / 100, fs / 50, fs / 25]);
        let sig = test_signal(fs, channels as usize, 1.5, rng.next_u64());
        let mut packets = Vec::new();
        let mut buf = vec![0u8; 60000];
        let ch = channels as usize;
        let mut pos = 0;
        for _ in 0..20 {
            let n = frame as usize;
            if (pos + n) * ch > sig.len() {
                pos = 0;
            }
            let len = enc
                .encode(&sig[pos * ch..(pos + n) * ch], n, &mut buf)
                .unwrap();
            pos += n;
            packets.push(buf[..len].to_vec());
        }
        for &dfs in &[fs, pick(&mut rng, &RATES)] {
            let mut r = ProjectionDecoder::new(dfs, channels, streams, coupled, &matrix).unwrap();
            let mut c = c::ProjDec::new(dfs, channels, streams, coupled, &matrix).unwrap();
            let max_fs = dfs / 25 * 3;
            for (i, pkt) in packets.iter().enumerate() {
                let what = format!("proj {s} ch {channels} fs {dfs} pkt {i}");
                let fmt = rand_fmt(&mut rng);
                let (data, fsz, fec): (Option<&[u8]>, i32, i32) = match rng.range_i32(0, 9) {
                    0 => (None, dfs / 50, 0),
                    1 if i + 1 < packets.len() => (Some(&packets[i + 1]), dfs / 50, 1),
                    2 => (Some(&pkt[..pkt.len() / 2]), max_fs, 0),
                    _ => (Some(pkt), max_fs, 0),
                };
                let n = fsz as usize * ch + 3;
                match fmt {
                    Fmt::I16 => {
                        let (mut a, mut b) = (vec![7i16; n], vec![7i16; n]);
                        assert_eq!(
                            rc(r.opus_projection_decode(data, &mut a, fsz, fec)),
                            c.decode(data, &mut b, fsz, fec),
                            "{what}"
                        );
                        assert!(a == b, "{what}: pcm");
                    }
                    Fmt::I24 => {
                        let (mut a, mut b) = (vec![7i32; n], vec![7i32; n]);
                        assert_eq!(
                            rc(r.opus_projection_decode24(data, &mut a, fsz, fec)),
                            c.decode24(data, &mut b, fsz, fec),
                            "{what}"
                        );
                        assert!(a == b, "{what}: pcm24");
                    }
                    Fmt::F32 => {
                        let (mut a, mut b) = (vec![7f32; n], vec![7f32; n]);
                        assert_eq!(
                            rc(r.opus_projection_decode_float(data, &mut a, fsz, fec)),
                            c.decode_float(data, &mut b, fsz, fec),
                            "{what}"
                        );
                        assert!(
                            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                            "{what}: pcm float"
                        );
                    }
                }
                for st in 0..streams {
                    assert_state(
                        r.ms_decoder().decoder_state(st).unwrap(),
                        &c.stream_dump(st),
                        &format!("{what} stream {st}"),
                    );
                }
                assert_eq!(r.final_range(), c.final_range().unwrap(), "{what}: rng");
                for req in MS_GETS {
                    assert_eq!(
                        rc(r.ctl_get(req).map(|v| v as u32 as usize)),
                        c.ctl_get(req).map(|v| v as u32 as usize),
                        "{what}: ctl_get({req})"
                    );
                }
                if i == 10 {
                    r.reset();
                    c.reset().unwrap();
                    let g = rng.range_i32(-2000, 2000);
                    assert_eq!(
                        r.ctl_set(sys::OPUS_SET_GAIN_REQUEST, g)
                            .map_err(Error::code),
                        c.ctl_set(sys::OPUS_SET_GAIN_REQUEST, g)
                    );
                }
            }
        }
    }
    // Family 2: ambisonics carried by the plain multistream API.
    for (s, &channels) in [4, 6, 9, 11].iter().enumerate() {
        let fs = 48000;
        let mut enc =
            api::MsEncoder::new_surround(fs, channels, 2, sys::OPUS_APPLICATION_AUDIO).unwrap();
        let (streams, coupled) = (enc.streams, enc.coupled_streams);
        let mapping = enc.mapping.clone();
        let sig = test_signal(fs, channels as usize, 1.0, rng.next_u64());
        let mut buf = vec![0u8; 40000];
        let mut p = MsPair {
            r: MsDecoder::new(fs, channels, streams, coupled, &mapping).unwrap(),
            c: c::MsDec::new(fs, channels, streams, coupled, &mapping).unwrap(),
            ch: channels as usize,
            streams,
        };
        let ch = channels as usize;
        let mut packets = Vec::new();
        for k in 0..20 {
            let len = enc
                .encode(&sig[k * 960 * ch..(k + 1) * 960 * ch], 960, &mut buf)
                .unwrap();
            packets.push(buf[..len].to_vec());
        }
        ms_decode_stream(&mut p, &packets, fs, &mut rng, &format!("family2 {s}"));
    }
}

/// Projection creation errors vs C (incl. OPUS_ALLOC_FAIL from the size computation).
#[test]
fn projection_api_errors() {
    let mut rng = Rng::new(12);
    let mut enc = api::ProjectionEncoder::new(48000, 9, 3, sys::OPUS_APPLICATION_AUDIO).unwrap();
    let matrix = enc.demixing_matrix().unwrap();
    for _ in 0..400 {
        let fs = pick(&mut rng, &[48000, 44100, 16000]);
        let channels = rng.range_i32(0, 260);
        let streams = rng.range_i32(-1, 12);
        let coupled = rng.range_i32(-1, 12);
        let m: Vec<u8> = match rng.range_i32(0, 2) {
            0 => matrix.clone(),
            1 => {
                let n = ((streams + coupled).max(0) * channels.max(0) * 2) as usize;
                let mut v = vec![0u8; n.min(200000)];
                rng.fill_bytes(&mut v);
                v
            }
            _ => matrix[..matrix.len() / 2].to_vec(),
        };
        let r = ProjectionDecoder::new(fs, channels, streams, coupled, &m)
            .map(drop)
            .map_err(Error::code);
        let c = c::ProjDec::new(fs, channels, streams, coupled, &m).map(drop);
        assert_eq!(
            r,
            c,
            "proj create fs {fs} ch {channels} s {streams} c {coupled} m {}",
            m.len()
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Test vectors
// ---------------------------------------------------------------------------------------------

/// Locates `testdata/vectors/<sub>` if present.
fn vectors_dir(sub: &str) -> Option<PathBuf> {
    if let Some(v) = std::env::var_os("OPUSORUS_VECTORS") {
        let p = PathBuf::from(v).join(sub);
        return p.is_dir().then_some(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .map(|a| a.join("testdata/vectors").join(sub))
        .find(|p| p.is_dir())
}

/// An `opus_demo` bitstream packet: payload and the encoder's final range.
struct BitPacket {
    data: Vec<u8>,
    rng: u32,
}

/// Parses an `opus_demo` bitstream (4-byte BE length, 4-byte BE final range, payload).
fn read_bit(path: &Path) -> Vec<BitPacket> {
    let b = std::fs::read(path).unwrap();
    let mut v = Vec::new();
    let mut i = 0;
    while i + 8 <= b.len() {
        let len = u32::from_be_bytes(b[i..i + 4].try_into().unwrap()) as usize;
        let rng = u32::from_be_bytes(b[i + 4..i + 8].try_into().unwrap());
        i += 8;
        if i + len > b.len() {
            break;
        }
        v.push(BitPacket {
            data: b[i..i + len].to_vec(),
            rng,
        });
        i += len;
    }
    v
}

/// Decodes a bitstream exactly like `opus_demo -d fs ch [-ignore_extensions]` (int24 output,
/// PLC/FEC for zero-length packets, no DRED) with the Rust decoder and the C decoder in
/// lock-step, asserting bit-exact output, identical return values and final ranges, and the
/// encoder final range recorded in the file (if `check_enc_range`; returns the number of
/// mismatches otherwise). Returns the Rust int24 output.
fn demo_decode(
    packets: &[BitPacket],
    fs: i32,
    ch: i32,
    ignore_ext: bool,
    check_enc_range: bool,
    what: &str,
) -> (Vec<i32>, usize) {
    let mut r = Decoder::new(fs, ch).unwrap();
    let mut c = c::Dec::new(fs, ch).unwrap();
    r.set_ignore_extensions(ignore_ext);
    c.ctl_set(
        sys::OPUS_SET_IGNORE_EXTENSIONS_REQUEST,
        i32::from(ignore_ext),
    )
    .unwrap();
    let max_frame_size = if cfg!(feature = "qext") {
        96000 * 2
    } else {
        48000 * 2
    };
    let chu = ch as usize;
    let mut out_r = vec![0i32; 5760 * 2 * chu];
    let mut out_c = vec![0i32; max_frame_size as usize * chu];
    let mut all = Vec::new();
    let mut lost_count = 0;
    let mut lost_prev = true;
    let mut range_checks = 0;
    let mut range_mismatches = 0;
    for (k, pkt) in packets.iter().enumerate() {
        let data = &pkt.data;
        let lost = data.is_empty();
        let run_decoder = if lost {
            lost_count += 1;
            0
        } else {
            1 + lost_count
        };
        for fr in 0..run_decoder {
            let (rr, cr) = if fr == lost_count - 1 && opusorus::packet::has_lbrr(data).unwrap() {
                let d = r.last_packet_duration() as i32;
                let cd = c
                    .ctl_get(sys::OPUS_GET_LAST_PACKET_DURATION_REQUEST)
                    .unwrap();
                assert_eq!(d, cd);
                if out_r.len() < d.max(1) as usize * chu {
                    out_r.resize(d as usize * chu, 0);
                }
                (
                    rc(r.opus_decode24(Some(data), &mut out_r, d, 1)),
                    c.decode24(Some(data), &mut out_c, d, 1),
                )
            } else if fr < lost_count {
                let d = r.last_packet_duration() as i32;
                if out_r.len() < d.max(1) as usize * chu {
                    out_r.resize(d as usize * chu, 0);
                }
                (
                    rc(r.opus_decode24(None, &mut out_r, d, 0)),
                    c.decode24(None, &mut out_c, d, 0),
                )
            } else {
                (
                    rc(r.opus_decode24(Some(data), &mut out_r, max_frame_size, 0)),
                    c.decode24(Some(data), &mut out_c, max_frame_size, 0),
                )
            };
            assert_eq!(rr, cr, "{what}: packet {k} return");
            if let Ok(n) = cr {
                let n = n * chu;
                assert!(out_r[..n] == out_c[..n], "{what}: packet {k} pcm");
                all.extend_from_slice(&out_r[..n]);
            }
        }
        let fr = r.final_range();
        assert_eq!(
            fr,
            c.final_range().unwrap(),
            "{what}: packet {k} final range"
        );
        if !lost && !lost_prev {
            if check_enc_range {
                assert_eq!(fr, pkt.rng, "{what}: packet {k} range vs encoder");
            } else if fr != pkt.rng {
                range_mismatches += 1;
            }
            range_checks += 1;
        }
        lost_prev = lost;
        if !lost {
            lost_count = 0;
        }
    }
    assert!(range_checks > 0);
    (all, range_mismatches)
}

/// `opus_demo` s16 output conversion of int24 samples.
fn demo_s16(x: &[i32]) -> Vec<f32> {
    x.iter()
        .map(|&s| {
            let s = s.clamp(-0x007fff00, 0x007fff00);
            ((s + 128) >> 8) as i16 as f32
        })
        .collect()
}

fn rfc_vector(dir: &Path, n: usize) -> bool {
    use opusorus_tools::compare::{OpusCompareOptions, opus_compare, read_pcm16};
    let bit = dir.join(format!("testvector{n:02}.bit"));
    let dec = dir.join(format!("testvector{n:02}.dec"));
    let decm = dir.join(format!("testvector{n:02}m.dec"));
    if !bit.is_file() || !dec.is_file() || !decm.is_file() {
        eprintln!("NOTE: testvector{n:02} missing; skipping");
        return false;
    }
    let packets = read_bit(&bit);
    let x1 = read_pcm16(&std::fs::read(&dec).unwrap(), 2);
    let x2 = read_pcm16(&std::fs::read(&decm).unwrap(), 2);
    for &fs in &[8000, 12000, 16000, 24000, 48000] {
        for ch in 1..=2 {
            let what = format!("testvector{n:02} {fs}/{ch}");
            let (out, _) = demo_decode(&packets, fs, ch, true, true, &what);
            // opus_compare against the references (run_vectors.sh accepts either).
            let check_rate = fs == 48000 || fs == [8000, 12000, 16000, 24000][n % 4];
            if check_rate {
                let y = demo_s16(&out);
                let opts = OpusCompareOptions {
                    nchannels: ch as usize,
                    rate: fs as u32,
                };
                let a = opus_compare(&x1, &y, &opts).unwrap();
                let pass = a.passes || opus_compare(&x2, &y, &opts).unwrap().passes;
                assert!(pass, "{what}: opus_compare fails (Q = {})", a.q);
            }
        }
    }
    true
}

/// RFC 8251 vectors at every rate, mono and stereo (in parallel).
#[test]
fn rfc8251_vectors() {
    let Some(dir) = vectors_dir("rfc8251") else {
        eprintln!(
            "NOTE: testdata/vectors/rfc8251 not found; skipping (run scripts/fetch_vectors.sh)"
        );
        return;
    };
    let ran = std::thread::scope(|s| {
        let handles: Vec<_> = (1..=12)
            .map(|n| {
                let dir = &dir;
                s.spawn(move || rfc_vector(dir, n))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .filter(|&ok| ok)
            .count()
    });
    eprintln!("rfc8251: {ran} vectors decoded bit-exactly and compared");
}

#[cfg(feature = "qext")]
mod qext {
    use super::*;
    use opusorus_tools::compare::{QextCompareOptions, SampleFormat, qext_compare, read_pcm};

    /// Regression test for a `celt/bands.rs` aliasing bug: a 96 kHz QEXT stream (all 14 QEXT
    /// bands) decoded by a 48 kHz decoder has `effEBands = 2 < qext_end` in the QEXT mode, so
    /// C's `lowband_scratch` is the `X` of QEXT band 1; with dual stereo, quant_band(Y) copies
    /// its lowband over the `X` band decoded just before. Uncorrelated stereo with transients
    /// at high rate triggers it.
    #[test]
    fn qext_96k_stream_at_48k_dual_stereo() {
        let mut rng = Rng::new(4896);
        let fs = 96000;
        let n = 96000 * 3;
        let mut sig = Vec::with_capacity(2 * n);
        let l = signals::music_like(n, 1, fs as u32, 1);
        let r = signals::noise(n, 1, 0.4, 2);
        for i in 0..n {
            // Clicks every 50 ms make transients (short blocks).
            let click = if i % 4800 < 48 { 0.8 } else { 0.0 };
            sig.push(l[i] + click);
            sig.push(r[i] * if (i / 9600) % 2 == 0 { 1.0 } else { 0.1 });
        }
        let sig = signals::to_i16(&sig);
        let mut total = Stats::default();
        for &frame in &[fs / 200, fs / 100, fs / 50] {
            let mut enc = api::Encoder::new(fs, 2, sys::OPUS_APPLICATION_RESTRICTED_CELT).unwrap();
            enc.ctl_set(sys::OPUS_SET_QEXT_REQUEST, 1).unwrap();
            enc.ctl_set(sys::OPUS_SET_BITRATE_REQUEST, 510000).unwrap();
            enc.ctl_set(sys::OPUS_SET_COMPLEXITY_REQUEST, 10).unwrap();
            let mut buf = vec![0u8; 20000];
            let packets: Vec<Vec<u8>> = (0..n / frame as usize)
                .map(|k| {
                    let f = frame as usize;
                    let len = enc
                        .encode(&sig[2 * k * f..2 * (k + 1) * f], f, &mut buf)
                        .unwrap();
                    buf[..len].to_vec()
                })
                .collect();
            for (dfs, dch) in [(48000, 2), (48000, 1), (24000, 2), (96000, 2)] {
                let mut p = Pair::new(dfs, dch);
                for (i, pkt) in packets.iter().enumerate() {
                    let what = format!("qext alias frame {frame} dec {dfs}/{dch} pkt {i}");
                    p.step(Some(pkt), dfs / 25 * 3, 0, rand_fmt(&mut rng), &what);
                }
                total += p.stats;
            }
        }
        assert_eq!(total.errors, 0);
    }

    /// 96 kHz decoding of C-encoded streams (with and without the QEXT extension).
    #[test]
    fn decode_96k_streams() {
        assert!(c::has_qext());
        let mut rng = Rng::new(96);
        for s in 0..24 {
            let fs = pick(&mut rng, &[48000, 96000]);
            let ch = rng.range_i32(1, 2);
            let cfg = EncCfg {
                fs,
                ch,
                app: pick(&mut rng, &APPS),
                nframes: 30,
                dynamic: s % 2 == 0,
                frame: pick(&mut rng, &[fs / 400, fs / 200, fs / 100, fs / 50]),
                bitrate: pick(&mut rng, &[64000, 128000, 256000, 510000]),
                fec: false,
                dtx: false,
                qext: rng.range_i32(0, 3) != 0,
            };
            let packets = encode_stream(&cfg, rng.next_u64());
            for (dfs, dch) in [(96000, ch), (pick(&mut rng, &RATES), 3 - ch)] {
                for &ignore in &[0, 1] {
                    let mut p = Pair::new(dfs, dch);
                    p.set(sys::OPUS_SET_IGNORE_EXTENSIONS_REQUEST, ignore, "qext");
                    let scen = Scen {
                        loss_pct: 5,
                        ctl_changes: false,
                    };
                    decode_stream(
                        &mut p,
                        &packets,
                        &mut rng,
                        scen,
                        &format!("qext {s} {cfg:?} ign {ignore}"),
                    );
                }
            }
        }
    }

    fn f32_of(x: &[i32]) -> Vec<f32> {
        // opus_demo -f32: int24 * (1/8388608), then the tools read f32 * 32768.
        x.iter()
            .map(|&v| v as f32 * (1.0 / 8_388_608.0) * 32768.0)
            .collect()
    }

    /// `qext_compare -s -r 96000 -f32 -thresholds 0.05 .1 .1 reference out`.
    fn check(reference: &Path, out: &[i32], what: &str, must_pass: bool) {
        let x = read_pcm(&std::fs::read(reference).unwrap(), 2, SampleFormat::F32Le);
        let opts = QextCompareOptions {
            nchannels: 2,
            base_rate: 96000,
            rate: 96000,
            skip: 0,
        };
        let res = qext_compare(&x, &f32_of(out), &opts).unwrap();
        let pass = res.passes(0.05, 0.1, 0.1);
        if must_pass {
            assert!(pass, "{what}: qext_compare fails {res:?}");
        } else if !pass {
            eprintln!(
                "NOTE: {what}: libopus 1.6.1 output (C and Rust alike) does not match the reference: {res:?}"
            );
        }
    }

    /// Decodes `name.bit` like `opus_demo -d 96000 2 -f32` (bit-exact vs C, also mono and at
    /// 48 kHz) and compares with `reference`.
    ///
    /// The Opus HD `qext_vector*` files do not correspond to the libopus 1.6.1 QEXT bitstream:
    /// the 1.6.1 decoder (the oracle, and an independently built upstream `opus_demo`) reports a
    /// range coder mismatch from the first QEXT packet on, and its output fails `qext_compare`.
    /// For those only Rust == C is asserted; the RFC vectors decoded at 96 kHz must pass.
    fn hd_vector(dir: &Path, name: &str, reference: &str, ignore: bool, is_qext: bool) -> bool {
        let bit = dir.join(format!("{name}.bit"));
        let refp = dir.join(reference);
        if !bit.is_file() || !refp.is_file() {
            eprintln!("NOTE: {name} missing; skipping");
            return false;
        }
        let packets = read_bit(&bit);
        let (out, mism) = demo_decode(&packets, 96000, 2, ignore, !is_qext, name);
        if mism > 0 {
            eprintln!(
                "NOTE: {name}: final range (C and Rust) differs from the file for {mism} packets"
            );
        }
        check(&refp, &out, name, !is_qext);
        // Other output configurations: bit-exact vs C only.
        demo_decode(
            &packets,
            96000,
            1,
            ignore,
            !is_qext,
            &format!("{name} mono"),
        );
        demo_decode(&packets, 48000, 2, ignore, !is_qext, &format!("{name} 48k"));
        true
    }

    /// Opus HD vectors: `qext_vectorNN` (and the fuzzed variants) and the RFC vectors decoded at
    /// 96 kHz, as `run_opushd_vectors.sh` does.
    #[test]
    fn opushd_vectors() {
        let Some(dir) = vectors_dir("opushd") else {
            eprintln!(
                "NOTE: testdata/vectors/opushd not found; skipping (run scripts/fetch_vectors.sh)"
            );
            return;
        };
        let ran = std::thread::scope(|s| {
            let dir = &dir;
            let mut hs = Vec::new();
            for n in 1..=6 {
                hs.push(s.spawn(move || {
                    let name = format!("qext_vector{n:02}");
                    hd_vector(dir, &name, &format!("{name}dec.f32"), false, true)
                }));
                hs.push(s.spawn(move || {
                    let name = format!("qext_vector{n:02}fuzz");
                    hd_vector(dir, &name, &format!("{name}dec.f32"), false, true)
                }));
            }
            for n in 1..=12 {
                hs.push(s.spawn(move || {
                    let name = format!("testvector{n:02}");
                    hd_vector(dir, &name, &format!("{name}_96k.f32"), true, false)
                }));
            }
            hs.into_iter()
                .map(|h| h.join().unwrap())
                .filter(|&ok| ok)
                .count()
        });
        eprintln!("opushd: {ran} vectors decoded bit-exactly and compared");
    }
}
