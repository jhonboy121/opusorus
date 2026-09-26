//! Differential tests for unit `dnn_integration`: the deep PLC, DRED and OSCE wired into the
//! Opus decoder/encoder (libopus `ENABLE_DEEP_PLC`, `ENABLE_DRED`, `ENABLE_OSCE` +
//! `ENABLE_OSCE_BWE`) and the public `opus_dred_*` API, vs the C library built with the same
//! features.
//!
//! The C oracle has the model weights compiled in; the Rust port has none, so every Rust
//! decoder/encoder/DRED decoder loads the weight blob serialized from the oracle's C tables
//! (`OPUS_SET_DNN_BLOB`), which gives both sides the same models.
//!
//! Every call is made on both sides with identical inputs; outputs (all three PCM formats, bit
//! patterns for floats), return values, final ranges, the DNN decoder state (LPCNet PLC
//! positions/buffers/features, OSCE control) and the parsed `OpusDRED` contents are compared
//! after every call. Coverage counters make sure the neural PLC, DRED concealment, LACE,
//! NoLACE and the BWE (with its cross-fades) are actually reached.
//!
//! Run with `--features deep-plc,dred,osce` (or any subset containing `deep-plc`).

#![cfg(feature = "deep-plc")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "test code: failures should panic; coverage notes go to stderr"
)]

use opusorus::Error;
use opusorus::decoder::Decoder;
use opusorus_conformance::{Rng, signals};
use opusorus_oracle::api;
use opusorus_oracle::dnn_integration as di;
use opusorus_oracle::sys;

const APP_VOIP: i32 = sys::OPUS_APPLICATION_VOIP;
const APP_AUDIO: i32 = sys::OPUS_APPLICATION_AUDIO;
#[cfg(feature = "dred")]
const APP_LOWDELAY: i32 = sys::OPUS_APPLICATION_RESTRICTED_LOWDELAY;
/// `OPUS_SET_FORCE_MODE_REQUEST` (opus_private.h).
const OPUS_SET_FORCE_MODE_REQUEST: i32 = 11002;
const MODE_SILK_ONLY: i32 = 1000;
const MODE_HYBRID: i32 = 1001;
const MODE_CELT_ONLY: i32 = 1002;
const OPUS_SET_OSCE_BWE_REQUEST: i32 = 4054;
const OPUS_GET_OSCE_BWE_REQUEST: i32 = 4055;
const RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];

fn rc<T: TryInto<i64>>(r: opusorus::Result<T>) -> Result<i64, i32> {
    r.map(|v| v.try_into().ok().unwrap()).map_err(Error::code)
}

fn cc(r: api::OResult<usize>) -> Result<i64, i32> {
    r.map(|v| v as i64)
}

const fn pick<T: Copy>(rng: &mut Rng, v: &[T]) -> T {
    v[rng.range_i32(0, v.len() as i32 - 1) as usize]
}

// ---------------------------------------------------------------------------------------------
// Stream generation (C encoder)
// ---------------------------------------------------------------------------------------------

/// Encoder configuration for a C-encoded test stream.
#[derive(Debug, Clone, Copy)]
struct EncCfg {
    fs: i32,
    ch: i32,
    app: i32,
    /// Frame size in samples at `fs`.
    frame: i32,
    nframes: usize,
    bitrate: i32,
    vbr: i32,
    fec: i32,
    plp: i32,
    dtx: i32,
    /// Forced mode (0 = auto).
    force_mode: i32,
    /// Forced bandwidth (0 = auto).
    bandwidth: i32,
    complexity: i32,
    dred: i32,
    /// Switch mode / bandwidth / bitrate mid-stream.
    dynamic: bool,
    speech: bool,
}

impl EncCfg {
    const fn silk(fs: i32, ch: i32, frame_ms: i32, bandwidth: i32) -> Self {
        Self {
            fs,
            ch,
            app: APP_VOIP,
            frame: fs * frame_ms / 1000,
            nframes: 60,
            bitrate: 16000,
            vbr: 1,
            fec: 0,
            plp: 0,
            dtx: 0,
            force_mode: MODE_SILK_ONLY,
            bandwidth,
            complexity: 9,
            dred: 0,
            dynamic: false,
            speech: true,
        }
    }
}

/// Configures a C encoder (every CTL must succeed).
fn configure_c(e: &mut api::Encoder, c: &EncCfg) {
    e.ctl_set(sys::OPUS_SET_BITRATE_REQUEST, c.bitrate).unwrap();
    e.ctl_set(sys::OPUS_SET_VBR_REQUEST, c.vbr).unwrap();
    e.ctl_set(sys::OPUS_SET_INBAND_FEC_REQUEST, c.fec).unwrap();
    e.ctl_set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, c.plp)
        .unwrap();
    e.ctl_set(sys::OPUS_SET_DTX_REQUEST, c.dtx).unwrap();
    e.ctl_set(sys::OPUS_SET_COMPLEXITY_REQUEST, c.complexity)
        .unwrap();
    if c.force_mode != 0 {
        e.ctl_set(OPUS_SET_FORCE_MODE_REQUEST, c.force_mode)
            .unwrap();
    }
    if c.bandwidth != 0 {
        e.ctl_set(sys::OPUS_SET_BANDWIDTH_REQUEST, c.bandwidth)
            .unwrap();
    }
    if c.speech {
        e.ctl_set(sys::OPUS_SET_SIGNAL_REQUEST, sys::OPUS_SIGNAL_VOICE)
            .unwrap();
    }
    if c.dred != 0 {
        e.ctl_set(sys::OPUS_SET_DRED_DURATION_REQUEST, c.dred)
            .unwrap();
    }
}

fn test_signal(c: &EncCfg, seed: u64) -> Vec<f32> {
    let n = c.frame as usize * c.nframes;
    if c.speech {
        signals::speech_like(n, c.ch as usize, c.fs as u32, seed)
    } else {
        signals::music_like(n, c.ch as usize, c.fs as u32, seed)
    }
}

/// Mid-stream switches of a dynamic stream (applied at frame `i`).
fn dynamic_switch(e: &mut api::Encoder, c: &EncCfg, i: usize, rng: &mut Rng) {
    if !c.dynamic || i == 0 || !i.is_multiple_of(12) {
        return;
    }
    let modes = [MODE_SILK_ONLY, MODE_HYBRID, MODE_CELT_ONLY, sys::OPUS_AUTO];
    let m = pick(rng, &modes);
    e.ctl_set(OPUS_SET_FORCE_MODE_REQUEST, m).unwrap();
    let bws = [
        sys::OPUS_AUTO,
        sys::OPUS_BANDWIDTH_NARROWBAND,
        sys::OPUS_BANDWIDTH_WIDEBAND,
        sys::OPUS_BANDWIDTH_SUPERWIDEBAND,
        sys::OPUS_BANDWIDTH_FULLBAND,
    ];
    e.ctl_set(sys::OPUS_SET_BANDWIDTH_REQUEST, pick(rng, &bws))
        .unwrap();
    e.ctl_set(
        sys::OPUS_SET_BITRATE_REQUEST,
        pick(rng, &[8000, 12000, 20000, 32000, 64000]),
    )
    .unwrap();
}

/// Encodes a stream with the C encoder.
fn c_stream(c: &EncCfg, seed: u64) -> Vec<Vec<u8>> {
    let mut e = api::Encoder::new(c.fs, c.ch, c.app).unwrap();
    configure_c(&mut e, c);
    let sig = test_signal(c, seed);
    let mut rng = Rng::new(seed ^ 0x5151);
    let n = c.frame as usize * c.ch as usize;
    let mut out = vec![0u8; 4000];
    (0..c.nframes)
        .map(|i| {
            dynamic_switch(&mut e, c, i, &mut rng);
            let len = e
                .encode_float(&sig[i * n..(i + 1) * n], c.frame as usize, &mut out)
                .unwrap();
            out[..len].to_vec()
        })
        .collect()
}

/// Whether `pkt` carries a DRED extension.
#[cfg(feature = "dred")]
fn has_dred_ext(pkt: &[u8]) -> bool {
    let Ok(p) = opusorus::packet::parse(pkt) else {
        return false;
    };
    let mut it = opusorus::extensions::ExtensionIterator::new(p.padding, p.nb_frames as i32);
    matches!(it.find(126), Ok(Some(_)))
}

// ---------------------------------------------------------------------------------------------
// Decoder pair
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fmt {
    I16,
    I24,
    F32,
}

const fn rand_fmt(rng: &mut Rng) -> Fmt {
    pick(rng, &[Fmt::I16, Fmt::I24, Fmt::F32])
}

/// Coverage counters.
#[derive(Debug, Default, Clone, Copy)]
struct Cov {
    calls: usize,
    /// Concealed calls with the neural PLC running (LPCNet `blend == 1` afterwards).
    neural: usize,
    /// DRED concealment calls with queued DRED features.
    dred_fec: usize,
    /// Calls in OSCE BWE mode (`osce_extended_mode == OSCE_MODE_SILK_BBWE`).
    bwe: usize,
    /// Calls with LACE / NoLACE selected.
    lace: usize,
    nolace: usize,
}

impl core::ops::AddAssign for Cov {
    fn add_assign(&mut self, o: Self) {
        self.calls += o.calls;
        self.neural += o.neural;
        self.dred_fec += o.dred_fec;
        self.bwe += o.bwe;
        self.lace += o.lace;
        self.nolace += o.nolace;
    }
}

/// A Rust decoder (with the oracle's models) and a C decoder fed identically.
struct Pair {
    r: Decoder,
    c: di::Dec,
    ch: usize,
    cov: Cov,
}

impl Pair {
    fn new(fs: i32, ch: i32) -> Self {
        let mut r = Decoder::new(fs, ch).unwrap();
        r.set_dnn_blob(di::decoder_blob()).unwrap();
        let c = di::Dec::new(fs, ch).unwrap();
        let mut p = Self {
            r,
            c,
            ch: ch as usize,
            cov: Cov::default(),
        };
        p.check("init");
        p
    }

    fn set(&mut self, req: i32, v: i32, what: &str) {
        let rr = self.r.ctl_set(req, v).map_err(Error::code);
        let cr = self.c.ctl_set(req, v);
        assert_eq!(rr, cr, "{what}: ctl_set {req} {v}");
    }

    #[track_caller]
    fn check(&mut self, what: &str) {
        let (ri, rf) = self.r.dnn_debug_state();
        let cs = self.c.dump();
        assert_eq!(ri, cs.ints, "{what}: DNN decoder state");
        if let Some(i) = rf
            .iter()
            .zip(&cs.floats)
            .position(|(a, b)| a.to_bits() != b.to_bits())
        {
            panic!(
                "{what}: lpcnet float state [{i}] rust={} c={}",
                rf[i], cs.floats[i]
            );
        }
        assert_eq!(
            self.r.final_range(),
            self.c.final_range(),
            "{what}: final range"
        );
        for req in [
            sys::OPUS_GET_BANDWIDTH_REQUEST,
            sys::OPUS_GET_PITCH_REQUEST,
            sys::OPUS_GET_LAST_PACKET_DURATION_REQUEST,
            sys::OPUS_GET_COMPLEXITY_REQUEST,
        ] {
            assert_eq!(
                self.r.ctl_get(req).map_err(Error::code),
                self.c.ctl_get(req),
                "{what}: ctl_get {req}"
            );
        }
        let s = cs.ints;
        self.cov.calls += 1;
        self.cov.bwe += usize::from(s[11] == 1003);
        self.cov.lace += usize::from(s[9] == 1);
        self.cov.nolace += usize::from(s[9] == 2);
    }

    /// Compares two output buffers.
    #[track_caller]
    fn cmp<T: PartialEq + core::fmt::Debug + Copy>(
        a: &[T],
        b: &[T],
        bits: impl Fn(T) -> u64,
        what: &str,
    ) {
        if let Some(i) = a.iter().zip(b).position(|(x, y)| bits(*x) != bits(*y)) {
            panic!("{what}: pcm[{i}] rust={:?} c={:?}", a[i], b[i]);
        }
    }

    /// One decode call on both sides.
    #[track_caller]
    fn step(&mut self, data: Option<&[u8]>, frame_size: i32, fec: i32, fmt: Fmt, what: &str) {
        let n = frame_size.max(1) as usize * self.ch + 3;
        let ret = match fmt {
            Fmt::I16 => {
                let mut a = vec![0x5A5Au16 as i16; n];
                let mut b = a.clone();
                let rr = rc(self.r.opus_decode(data, &mut a, frame_size, fec));
                let cr = cc(self.c.decode(data, &mut b, frame_size, fec));
                assert_eq!(rr, cr, "{what}: opus_decode return");
                Self::cmp(&a, &b, |x| x as u16 as u64, what);
                cr
            }
            Fmt::I24 => {
                let mut a = vec![0x5A5A_5A5A; n];
                let mut b = a.clone();
                let rr = rc(self.r.opus_decode24(data, &mut a, frame_size, fec));
                let cr = cc(self.c.decode24(data, &mut b, frame_size, fec));
                assert_eq!(rr, cr, "{what}: opus_decode24 return");
                Self::cmp(&a, &b, |x| x as u32 as u64, what);
                cr
            }
            Fmt::F32 => {
                let mut a = vec![1234.5f32; n];
                let mut b = a.clone();
                let rr = rc(self.r.opus_decode_float(data, &mut a, frame_size, fec));
                let cr = cc(self.c.decode_float(data, &mut b, frame_size, fec));
                assert_eq!(rr, cr, "{what}: opus_decode_float return");
                Self::cmp(&a, &b, |x: f32| u64::from(x.to_bits()), what);
                cr
            }
        };
        self.check(what);
        if data.is_none() && ret.is_ok() && self.c.dump().ints[7] == 1 {
            self.cov.neural += 1;
        }
    }

    /// One `opus_decoder_dred_decode*` call on both sides.
    #[cfg(feature = "dred")]
    #[track_caller]
    fn dred_step(
        &mut self,
        rd: &opusorus::dred::Dred,
        cd: &di::Dred,
        dred_offset: i32,
        frame_size: i32,
        fmt: Fmt,
        what: &str,
    ) {
        let n = frame_size.max(1) as usize * self.ch + 3;
        let queued = rd.process_stage() == 2;
        match fmt {
            Fmt::I16 => {
                let mut a = vec![0x5A5Au16 as i16; n];
                let mut b = a.clone();
                let rr = rc(self
                    .r
                    .opus_decoder_dred_decode(rd, dred_offset, &mut a, frame_size));
                let cr = cc(self.c.dred_decode(cd, dred_offset, &mut b, frame_size));
                assert_eq!(rr, cr, "{what}: dred_decode return");
                Self::cmp(&a, &b, |x| x as u16 as u64, what);
            }
            Fmt::I24 => {
                let mut a = vec![0x5A5A_5A5A; n];
                let mut b = a.clone();
                let rr = rc(self
                    .r
                    .opus_decoder_dred_decode24(rd, dred_offset, &mut a, frame_size));
                let cr = cc(self.c.dred_decode24(cd, dred_offset, &mut b, frame_size));
                assert_eq!(rr, cr, "{what}: dred_decode24 return");
                Self::cmp(&a, &b, |x| x as u32 as u64, what);
            }
            Fmt::F32 => {
                let mut a = vec![1234.5f32; n];
                let mut b = a.clone();
                let rr =
                    rc(self
                        .r
                        .opus_decoder_dred_decode_float(rd, dred_offset, &mut a, frame_size));
                let cr = cc(self
                    .c
                    .dred_decode_float(cd, dred_offset, &mut b, frame_size));
                assert_eq!(rr, cr, "{what}: dred_decode_float return");
                Self::cmp(&a, &b, |x: f32| u64::from(x.to_bits()), what);
            }
        }
        let fill = self.c.dump().ints[3];
        self.check(what);
        if queued && fill > 0 {
            self.cov.dred_fec += 1;
        }
    }
}

/// Decodes `packets` on a pair with random losses (bursts), FEC, odd PLC sizes, complexity
/// changes and output formats.
fn decode_lossy(
    p: &mut Pair,
    packets: &[Vec<u8>],
    rng: &mut Rng,
    loss_pct: i32,
    cplx: Option<i32>,
    tag: &str,
) {
    let fs = p.r.sample_rate();
    let max_fs = fs / 25 * 3;
    let mut burst = 0;
    for (i, pkt) in packets.iter().enumerate() {
        let what = format!("{tag} pkt {i}");
        let dur = opusorus::packet::get_nb_samples(pkt, fs).unwrap();
        if rng.range_i32(0, 19) == 0 {
            let c = match cplx {
                Some(c) => c,
                None => rng.range_i32(0, 10),
            };
            p.set(sys::OPUS_SET_COMPLEXITY_REQUEST, c, &what);
        }
        if rng.range_i32(0, 199) == 0 {
            p.set(sys::OPUS_RESET_STATE, 0, &what);
        }
        let fmt = rand_fmt(rng);
        if burst > 0 || rng.range_i32(0, 99) < loss_pct {
            if burst == 0 {
                burst = pick(rng, &[1, 1, 2, 3, 5, 9]);
            }
            burst -= 1;
            if burst == 0 && i + 1 < packets.len() && rng.range_i32(0, 2) == 0 {
                // Last lost packet: FEC from the next one.
                p.step(Some(&packets[i + 1]), dur, 1, fmt, &what);
            } else if rng.range_i32(0, 5) == 0 {
                // Odd concealment size (multiple of 2.5 ms).
                let f2_5 = fs / 400;
                p.step(None, f2_5 * rng.range_i32(1, 16), 0, fmt, &what);
            } else {
                p.step(None, dur, 0, fmt, &what);
            }
        } else {
            p.step(Some(pkt), max_fs, 0, fmt, &what);
        }
    }
}

fn note(test: &str, cov: &Cov) {
    eprintln!("{test}: {cov:?}");
}

// ---------------------------------------------------------------------------------------------
// Deep PLC
// ---------------------------------------------------------------------------------------------

/// SILK / hybrid / CELT streams with losses decoded at every rate/channel count with
/// complexity 0..10: the neural PLC (complexity >= 5) runs in the SILK decoder (16 kHz
/// internal rate) and in the CELT decoder (`FRAME_PLC_NEURAL`, `update_plc_state`, the
/// 16 → 48 kHz resampling, stereo copy and cross-fade).
#[test]
fn deep_plc_lossy_streams() {
    let mut rng = Rng::new(0xD0_0001);
    let mut cfgs = vec![
        EncCfg::silk(16000, 1, 20, sys::OPUS_BANDWIDTH_WIDEBAND),
        EncCfg::silk(8000, 1, 10, sys::OPUS_BANDWIDTH_NARROWBAND),
        EncCfg::silk(12000, 2, 20, sys::OPUS_BANDWIDTH_MEDIUMBAND),
        EncCfg::silk(16000, 2, 40, sys::OPUS_BANDWIDTH_WIDEBAND),
        EncCfg::silk(48000, 1, 60, sys::OPUS_BANDWIDTH_WIDEBAND),
    ];
    // Hybrid.
    let mut h = EncCfg::silk(48000, 2, 20, sys::OPUS_BANDWIDTH_FULLBAND);
    h.force_mode = MODE_HYBRID;
    h.bitrate = 32000;
    cfgs.push(h);
    let mut h = EncCfg::silk(24000, 1, 10, sys::OPUS_BANDWIDTH_SUPERWIDEBAND);
    h.force_mode = MODE_HYBRID;
    h.bitrate = 24000;
    cfgs.push(h);
    // CELT.
    for (fs, ch, ms10) in [
        (48000, 1, 200),
        (48000, 2, 100),
        (24000, 2, 50),
        (16000, 1, 25),
    ] {
        let mut c = EncCfg::silk(fs, ch, 20, 0);
        c.frame = fs * ms10 / 10000;
        c.nframes = 60 * 200 / ms10 as usize;
        c.app = APP_AUDIO;
        c.force_mode = MODE_CELT_ONLY;
        c.bitrate = 48000;
        c.speech = ms10 != 100;
        cfgs.push(c);
    }
    // Mode switching with FEC.
    let mut d = EncCfg::silk(48000, 2, 20, 0);
    d.force_mode = 0;
    d.dynamic = true;
    d.fec = 1;
    d.plp = 20;
    d.nframes = 90;
    cfgs.push(d);

    let mut total = Cov::default();
    for (k, cfg) in cfgs.iter().enumerate() {
        let packets = c_stream(cfg, 100 + k as u64);
        let mut decs = vec![(cfg.fs, cfg.ch)];
        decs.push((pick(&mut rng, &RATES), rng.range_i32(1, 2)));
        for (dfs, dch) in decs {
            let mut p = Pair::new(dfs, dch);
            p.set(
                sys::OPUS_SET_COMPLEXITY_REQUEST,
                rng.range_i32(5, 10),
                "init",
            );
            decode_lossy(
                &mut p,
                &packets,
                &mut rng,
                25,
                None,
                &format!("plc {k} {cfg:?} dec {dfs}/{dch}"),
            );
            total += p.cov;
        }
    }
    note("deep_plc_lossy_streams", &total);
    assert!(total.neural > 100, "neural PLC coverage: {total:?}");
}

/// Long losses (the neural PLC fades out after 80 x 2.5 ms in CELT, the SILK attenuation) and
/// very small concealment sizes, plus the complexity boundary 4/5.
#[test]
fn deep_plc_long_losses() {
    let mut rng = Rng::new(0xD0_0002);
    let mut total = Cov::default();
    for (k, (fs, ch, mode)) in [
        (16000, 1, MODE_SILK_ONLY),
        (48000, 2, MODE_CELT_ONLY),
        (48000, 1, MODE_HYBRID),
    ]
    .into_iter()
    .enumerate()
    {
        let mut c = EncCfg::silk(fs, ch, 20, 0);
        c.force_mode = mode;
        c.app = if mode == MODE_CELT_ONLY {
            APP_AUDIO
        } else {
            APP_VOIP
        };
        c.bitrate = 32000;
        c.nframes = 40;
        let packets = c_stream(&c, 7 + k as u64);
        for cplx in [4, 5, 10] {
            let mut p = Pair::new(fs, ch);
            p.set(sys::OPUS_SET_COMPLEXITY_REQUEST, cplx, "init");
            let what = format!("long {k} cplx {cplx}");
            for (i, pkt) in packets.iter().enumerate() {
                if i % 13 == 7 {
                    // ~1.5 s of loss in mixed sizes.
                    for j in 0..40 {
                        let f = if j % 3 == 0 { fs / 400 } else { fs / 50 };
                        p.step(
                            None,
                            f,
                            0,
                            rand_fmt(&mut rng),
                            &format!("{what} loss {i}/{j}"),
                        );
                    }
                }
                p.step(
                    Some(pkt),
                    fs / 50,
                    0,
                    rand_fmt(&mut rng),
                    &format!("{what} pkt {i}"),
                );
            }
            total += p.cov;
        }
    }
    note("deep_plc_long_losses", &total);
    assert!(total.neural > 100, "neural PLC coverage: {total:?}");
}

// ---------------------------------------------------------------------------------------------
// OSCE
// ---------------------------------------------------------------------------------------------

/// SILK enhancement: LACE (complexity 6) and NoLACE (>= 7) on NB/MB/WB SILK at 10/20 ms, with
/// and without losses, every output rate, mono/stereo; complexity changes switch the method
/// (`osce_reset` on a method change).
#[cfg(feature = "osce")]
#[test]
fn osce_enhancement() {
    let mut rng = Rng::new(0x05CE_0001);
    let mut total = Cov::default();
    let cfgs = [
        EncCfg::silk(16000, 1, 20, sys::OPUS_BANDWIDTH_WIDEBAND),
        EncCfg::silk(16000, 2, 20, sys::OPUS_BANDWIDTH_WIDEBAND),
        EncCfg::silk(8000, 1, 20, sys::OPUS_BANDWIDTH_NARROWBAND),
        EncCfg::silk(12000, 1, 20, sys::OPUS_BANDWIDTH_MEDIUMBAND),
        EncCfg::silk(16000, 1, 10, sys::OPUS_BANDWIDTH_WIDEBAND),
        EncCfg::silk(48000, 1, 60, sys::OPUS_BANDWIDTH_WIDEBAND),
    ];
    for (k, cfg) in cfgs.iter().enumerate() {
        let mut cfg = *cfg;
        cfg.nframes = 50;
        cfg.bitrate = pick(&mut rng, &[8000, 12000, 20000]);
        let packets = c_stream(&cfg, 300 + k as u64);
        for (dfs, dch, cplx, loss) in [
            (cfg.fs, cfg.ch, Some(6), 0),
            (cfg.fs, cfg.ch, Some(7), 0),
            (48000, 1, Some(10), 10),
            (pick(&mut rng, &RATES), rng.range_i32(1, 2), None, 10),
        ] {
            let mut p = Pair::new(dfs, dch);
            p.set(sys::OPUS_SET_COMPLEXITY_REQUEST, cplx.unwrap_or(6), "init");
            decode_lossy(
                &mut p,
                &packets,
                &mut rng,
                loss,
                cplx,
                &format!("osce {k} {cfg:?} dec {dfs}/{dch}"),
            );
            total += p.cov;
        }
    }
    note("osce_enhancement", &total);
    assert!(
        total.lace > 50 && total.nolace > 50,
        "OSCE coverage: {total:?}"
    );
}

/// OSCE bandwidth extension (`OPUS_SET_OSCE_BWE`) of 16 kHz SILK at 48 kHz: BWE on/off,
/// complexity around the threshold 4, mono/stereo, losses (BBWE also for PLC), and mode
/// transitions SILK <-> hybrid <-> CELT and bandwidth changes that exercise the cross-fades
/// (`prev_osce_extended_mode`).
#[cfg(feature = "osce")]
#[test]
fn osce_bwe() {
    let mut rng = Rng::new(0x05CE_0002);
    let mut total = Cov::default();
    let mut cfgs = vec![
        EncCfg::silk(16000, 1, 20, sys::OPUS_BANDWIDTH_WIDEBAND),
        EncCfg::silk(48000, 2, 20, sys::OPUS_BANDWIDTH_WIDEBAND),
        EncCfg::silk(16000, 1, 10, sys::OPUS_BANDWIDTH_WIDEBAND),
    ];
    let mut d = EncCfg::silk(48000, 1, 20, 0);
    d.force_mode = 0;
    d.dynamic = true;
    d.nframes = 96;
    cfgs.push(d);
    let mut d = EncCfg::silk(48000, 2, 20, 0);
    d.force_mode = 0;
    d.dynamic = true;
    d.nframes = 96;
    d.fec = 1;
    d.plp = 10;
    cfgs.push(d);
    for (k, cfg) in cfgs.iter().enumerate() {
        let packets = c_stream(cfg, 400 + k as u64);
        for (dch, cplx, bwe, loss) in [
            (cfg.ch, Some(4), 1, 0),
            (cfg.ch, Some(7), 1, 10),
            (3 - cfg.ch, None, 1, 10),
            (cfg.ch, Some(10), 0, 0),
        ] {
            let mut p = Pair::new(48000, dch);
            p.set(OPUS_SET_OSCE_BWE_REQUEST, bwe, "init");
            p.set(sys::OPUS_SET_COMPLEXITY_REQUEST, cplx.unwrap_or(5), "init");
            let what = format!("bwe {k} {cfg:?} ch {dch} bwe {bwe}");
            // Toggle BWE mid-stream too.
            let half = packets.len() / 2;
            decode_lossy(&mut p, &packets[..half], &mut rng, loss, cplx, &what);
            p.set(OPUS_SET_OSCE_BWE_REQUEST, 1 - bwe, &what);
            decode_lossy(&mut p, &packets[half..], &mut rng, loss, cplx, &what);
            total += p.cov;
        }
    }
    note("osce_bwe", &total);
    assert!(total.bwe > 100, "BWE coverage: {total:?}");
}

// ---------------------------------------------------------------------------------------------
// CTLs and API behaviour
// ---------------------------------------------------------------------------------------------

#[test]
fn dnn_ctls() {
    for (fs, ch) in [(48000, 2), (16000, 1)] {
        let mut p = Pair::new(fs, ch);
        // C has compiled-in weights, i.e. no OPUS_SET_DNN_BLOB (UNIMPLEMENTED); the numeric
        // Rust dispatcher cannot carry a pointer and returns the same.
        for v in [-1, 0, 1] {
            p.set(sys::OPUS_SET_DNN_BLOB_REQUEST, v, "dnn blob");
            assert_eq!(
                p.r.ctl_get(sys::OPUS_SET_DNN_BLOB_REQUEST)
                    .map_err(Error::code),
                p.c.ctl_get(sys::OPUS_SET_DNN_BLOB_REQUEST),
            );
        }
        #[cfg(feature = "osce")]
        {
            for v in [-1, 0, 1, 2, 1] {
                p.set(OPUS_SET_OSCE_BWE_REQUEST, v, "osce bwe");
                assert_eq!(
                    p.r.ctl_get(OPUS_GET_OSCE_BWE_REQUEST).map_err(Error::code),
                    p.c.ctl_get(OPUS_GET_OSCE_BWE_REQUEST),
                );
                assert_eq!(
                    p.r.osce_bwe(),
                    p.c.ctl_get(OPUS_GET_OSCE_BWE_REQUEST).unwrap()
                );
            }
        }
        #[cfg(not(feature = "osce"))]
        {
            let _ = (OPUS_SET_OSCE_BWE_REQUEST, OPUS_GET_OSCE_BWE_REQUEST);
        }
        // The loaded models survive a reset and re-init (Rust: the blob plays the role of the
        // compiled-in weights).
        p.set(sys::OPUS_RESET_STATE, 0, "reset");
        p.check("after reset");
        p.r.init(fs, ch).unwrap();
        assert!(p.r.dnn_loaded().0);
    }
    // Rust-only: a decoder without a blob has no models; a bad blob is rejected.
    let mut d = Decoder::new(48000, 1).unwrap();
    assert!(!d.dnn_loaded().0);
    assert_eq!(d.set_dnn_blob(&[1, 2, 3]), Err(Error::BadArg));
    assert_eq!(d.set_dnn_blob(&[]), Err(Error::BadArg));
    assert!(!d.dnn_loaded().0);
    // A blob with only some of the models loads those and reports the error.
    #[cfg(feature = "osce")]
    {
        let plc_only: Vec<u8> = [
            opusorus_oracle::dnn_core::MODEL_PITCHDNN,
            opusorus_oracle::dnn_core::MODEL_PLC,
            opusorus_oracle::dnn_core::MODEL_FARGAN,
        ]
        .iter()
        .flat_map(|&m| opusorus_oracle::dnn_core::write_blob(m))
        .collect();
        assert_eq!(d.set_dnn_blob(&plc_only), Err(Error::BadArg));
        assert_eq!(d.dnn_loaded(), (true, false));
        d.set_dnn_blob(di::decoder_blob()).unwrap();
        assert_eq!(d.dnn_loaded(), (true, true));
    }
    // Without models the DNN paths stay off: at complexity 10 a Rust decoder without a blob
    // conceals like one with the DNN features compiled out (no neural state change).
    let mut d = Decoder::new(16000, 1).unwrap();
    d.set_complexity(10).unwrap();
    let mut pcm = vec![0i16; 320];
    let pkts = c_stream(&EncCfg::silk(16000, 1, 20, sys::OPUS_BANDWIDTH_WIDEBAND), 5);
    for pkt in &pkts[..10] {
        d.decode(Some(pkt), &mut pcm, 320, false).unwrap();
        d.decode(None, &mut pcm, 320, false).unwrap();
    }
    let (ints, _) = d.dnn_debug_state();
    assert_eq!(ints[0], 0, "not loaded");
    assert_eq!(ints[7], 0, "neural PLC never ran");
}

// ---------------------------------------------------------------------------------------------
// DRED
// ---------------------------------------------------------------------------------------------

/// Rust encoder with the oracle's DRED encoder models.
#[cfg(feature = "dred")]
fn rust_encoder(c: &EncCfg) -> opusorus::Encoder {
    let mut r = opusorus::Encoder::new_raw(c.fs, c.ch, c.app).unwrap();
    r.set_dnn_blob(di::encoder_blob()).unwrap();
    r
}

#[cfg(feature = "dred")]
fn configure_both(r: &mut opusorus::Encoder, e: &mut api::Encoder, req: i32, v: i32, what: &str) {
    let rr = r.ctl_set(req, v).map_err(Error::code);
    let cr = e.ctl_set(req, v);
    assert_eq!(rr, cr, "{what}: ctl {req} {v}");
}

/// Encodes a stream on both encoders (DRED on), comparing bytes and final ranges; returns
/// the packets and the number with a DRED extension.
#[cfg(feature = "dred")]
fn encode_both(c: &EncCfg, seed: u64, rng: &mut Rng) -> (Vec<Vec<u8>>, usize) {
    let mut e = api::Encoder::new(c.fs, c.ch, c.app).unwrap();
    let mut r = rust_encoder(c);
    let what = format!("{c:?}");
    let ctls = [
        (sys::OPUS_SET_BITRATE_REQUEST, c.bitrate),
        (sys::OPUS_SET_VBR_REQUEST, c.vbr),
        (sys::OPUS_SET_INBAND_FEC_REQUEST, c.fec),
        (sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, c.plp),
        (sys::OPUS_SET_DTX_REQUEST, c.dtx),
        (sys::OPUS_SET_COMPLEXITY_REQUEST, c.complexity),
        (sys::OPUS_SET_DRED_DURATION_REQUEST, c.dred),
    ];
    for (req, v) in ctls {
        configure_both(&mut r, &mut e, req, v, &what);
    }
    if c.force_mode != 0 {
        configure_both(
            &mut r,
            &mut e,
            OPUS_SET_FORCE_MODE_REQUEST,
            c.force_mode,
            &what,
        );
    }
    if c.bandwidth != 0 {
        configure_both(
            &mut r,
            &mut e,
            sys::OPUS_SET_BANDWIDTH_REQUEST,
            c.bandwidth,
            &what,
        );
    }
    if c.speech {
        configure_both(
            &mut r,
            &mut e,
            sys::OPUS_SET_SIGNAL_REQUEST,
            sys::OPUS_SIGNAL_VOICE,
            &what,
        );
    }
    let sig = test_signal(c, seed);
    let n = c.frame as usize * c.ch as usize;
    let mut out_r = vec![0u8; 1500];
    let mut out_c = vec![0u8; 1500];
    let mut packets = Vec::new();
    let mut with_dred = 0;
    for i in 0..c.nframes {
        let what = format!("{what} frame {i}");
        if c.dynamic && i > 0 && i % 10 == 0 {
            let m = pick(
                rng,
                &[MODE_SILK_ONLY, MODE_HYBRID, MODE_CELT_ONLY, sys::OPUS_AUTO],
            );
            configure_both(&mut r, &mut e, OPUS_SET_FORCE_MODE_REQUEST, m, &what);
            let b = pick(rng, &[8000, 16000, 24000, 40000, 64000, 96000]);
            configure_both(&mut r, &mut e, sys::OPUS_SET_BITRATE_REQUEST, b, &what);
            let d = pick(rng, &[0, 1, 4, 10, 40, 104]);
            configure_both(
                &mut r,
                &mut e,
                sys::OPUS_SET_DRED_DURATION_REQUEST,
                d,
                &what,
            );
            let l = pick(rng, &[0, 3, 5, 6, 20, 50]);
            configure_both(
                &mut r,
                &mut e,
                sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST,
                l,
                &what,
            );
            if rng.range_i32(0, 5) == 0 {
                r.reset();
                e.reset().unwrap();
            }
        }
        let max = if c.vbr == 0 && rng.range_i32(0, 7) == 0 {
            rng.range_i32(3, 1500) as usize
        } else {
            1500
        };
        let pcm = &sig[i * n..(i + 1) * n];
        let rr = rc(r.encode_float(pcm, c.frame as usize, &mut out_r[..max]));
        let cr = cc(e.encode_float(pcm, c.frame as usize, &mut out_c[..max]));
        assert_eq!(rr, cr, "{what}: encode return");
        if let Ok(len) = cr {
            let len = len as usize;
            assert_eq!(out_r[..len], out_c[..len], "{what}: packet bytes");
            assert_eq!(
                r.final_range(),
                e.final_range().unwrap(),
                "{what}: final range"
            );
            with_dred += usize::from(has_dred_ext(&out_c[..len]));
            packets.push(out_c[..len].to_vec());
        }
    }
    (packets, with_dred)
}

/// Encoder with DRED durations 1..104 over modes, rates, bitrates, VBR/CVBR/CBR, FEC, DTX,
/// packet loss percentages, frame sizes (10..120 ms) and mid-stream changes: bytes identical.
#[cfg(feature = "dred")]
#[test]
fn dred_encoder_streams() {
    let mut rng = Rng::new(0xD4ED_0001);
    let mut n_dred = 0;
    let mut n_pkts = 0;
    let mut cfgs = Vec::new();
    for (fs, ch, ms, mode, bitrate) in [
        (16000, 1, 20, MODE_SILK_ONLY, 16000),
        (48000, 1, 20, 0, 32000),
        (48000, 2, 20, MODE_HYBRID, 48000),
        (48000, 1, 10, MODE_CELT_ONLY, 64000),
        (8000, 1, 40, MODE_SILK_ONLY, 12000),
        (24000, 2, 60, 0, 24000),
        (12000, 1, 20, 0, 20000),
        (48000, 1, 120, 0, 24000),
        (48000, 2, 80, MODE_SILK_ONLY, 24000),
    ] {
        let mut c = EncCfg::silk(fs, ch, ms, 0);
        c.force_mode = mode;
        c.app = if mode == MODE_CELT_ONLY {
            APP_AUDIO
        } else {
            APP_VOIP
        };
        c.bitrate = bitrate;
        c.nframes = (1600 / ms).max(12) as usize;
        c.dred = pick(&mut rng, &[4, 10, 50, 104]);
        c.plp = pick(&mut rng, &[3, 5, 10, 25]);
        cfgs.push(c);
    }
    // VBR modes, FEC, DTX, dynamic.
    let base = cfgs[1];
    for (vbr, fec, dtx, dynamic) in [
        (0, 0, 0, false),
        (1, 1, 0, false),
        (1, 0, 1, false),
        (0, 1, 1, true),
        (1, 2, 0, true),
    ] {
        let mut c = base;
        c.vbr = vbr;
        c.fec = fec;
        c.dtx = dtx;
        c.dynamic = dynamic;
        c.force_mode = 0;
        c.nframes = 120;
        c.dred = 20;
        c.plp = 15;
        c.app = pick(&mut rng, &[APP_VOIP, APP_AUDIO]);
        cfgs.push(c);
    }
    let mut lowdelay = base;
    lowdelay.app = APP_LOWDELAY;
    lowdelay.force_mode = 0;
    lowdelay.dred = 30;
    lowdelay.plp = 20;
    cfgs.push(lowdelay);
    for (k, c) in cfgs.iter().enumerate() {
        let (packets, with) = encode_both(c, 500 + k as u64, &mut rng);
        n_dred += with;
        n_pkts += packets.len();
    }
    eprintln!("dred_encoder_streams: {n_pkts} packets, {n_dred} with DRED");
    assert!(n_dred > 300, "DRED coverage: {n_dred}");
}

/// Compares a Rust `Dred` with a C `OpusDRED`.
#[cfg(feature = "dred")]
#[track_caller]
fn cmp_dred(r: &opusorus::dred::Dred, c: &di::Dred, what: &str) {
    let cs = c.dump();
    let ri = r.inner();
    assert_eq!(
        [ri.nb_latents, ri.process_stage, ri.dred_offset],
        cs.ints,
        "{what}: OpusDRED ints"
    );
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&ri.state), bits(&cs.state), "{what}: OpusDRED state");
    assert_eq!(
        bits(&ri.latents),
        bits(&cs.latents),
        "{what}: OpusDRED latents"
    );
    assert_eq!(
        bits(&ri.fec_features),
        bits(&cs.fec_features),
        "{what}: OpusDRED fec_features"
    );
}

/// `opus_dred_parse` (immediate and deferred processing, various `max_dred_samples` and
/// sampling rates), `opus_dred_process` (copy and in place), and DRED concealment of loss
/// bursts with `opus_decoder_dred_decode*` (like `opus_demo`), on C-encoded DRED streams.
#[cfg(feature = "dred")]
#[test]
fn dred_parse_process_decode() {
    use opusorus::dred::{Dred, DredDecoder};
    let mut rng = Rng::new(0xD4ED_0002);
    let mut rdd = DredDecoder::new();
    rdd.set_dnn_blob(di::dred_decoder_blob()).unwrap();
    let mut cdd = di::DredDec::new().unwrap();
    let mut total = Cov::default();
    let mut parsed = 0;
    let mut cfgs = Vec::new();
    for (fs, ch, ms, mode, dred) in [
        (16000, 1, 20, MODE_SILK_ONLY, 104),
        (48000, 1, 20, 0, 50),
        (48000, 2, 20, MODE_HYBRID, 20),
        (48000, 1, 10, MODE_CELT_ONLY, 10),
        (16000, 1, 60, MODE_SILK_ONLY, 30),
        (48000, 2, 40, 0, 104),
    ] {
        let mut c = EncCfg::silk(fs, ch, ms, 0);
        c.force_mode = mode;
        c.app = if mode == MODE_CELT_ONLY {
            APP_AUDIO
        } else {
            APP_VOIP
        };
        c.bitrate = 32000;
        c.plp = 20;
        c.dred = dred;
        c.nframes = (2400 / ms) as usize;
        cfgs.push(c);
    }
    for (k, c) in cfgs.iter().enumerate() {
        let packets = c_stream(c, 700 + k as u64);
        // Parse-only checks on every packet.
        for (i, pkt) in packets.iter().enumerate() {
            let what = format!("parse {k} pkt {i}");
            let mut rd = Dred::new();
            let mut cd = di::Dred::new().unwrap();
            let max = pick(&mut rng, &[0, 480, 960, 4800, 48000, 100_000]);
            let sr = pick(&mut rng, &[8000, 16000, 48000]);
            let defer = rng.range_i32(0, 1) == 1;
            let rr = rdd.parse(&mut rd, pkt, max, sr, defer).map_err(Error::code);
            let cr = cdd.parse(&mut cd, pkt, max, sr, defer);
            assert_eq!(rr, cr, "{what}: parse return (max {max} sr {sr})");
            cmp_dred(&rd, &cd, &what);
            parsed += usize::from(matches!(cr, Ok((n, _)) if n > 0));
            // Process (copy, then in place on the copy: no-op).
            let mut rd2 = Dred::new();
            let mut cd2 = di::Dred::new().unwrap();
            let rr = rdd.process(&rd, &mut rd2).map_err(Error::code);
            let cr = cdd.process(&cd, &mut cd2);
            assert_eq!(rr, cr, "{what}: process return");
            cmp_dred(&rd2, &cd2, &format!("{what} processed"));
            let rr = rdd.process_in_place(&mut rd).map_err(Error::code);
            let cr = cdd.process_in_place(&mut cd);
            assert_eq!(rr, cr, "{what}: process in place return");
            cmp_dred(&rd, &cd, &format!("{what} processed in place"));
        }
        // Loss bursts concealed with DRED.
        for (dfs, dch) in [(c.fs, c.ch), (pick(&mut rng, &RATES), rng.range_i32(1, 2))] {
            let mut p = Pair::new(dfs, dch);
            p.set(
                sys::OPUS_SET_COMPLEXITY_REQUEST,
                pick(&mut rng, &[0, 5, 10]),
                "init",
            );
            let mut rd = Dred::new();
            let mut cd = di::Dred::new().unwrap();
            let mut lost_count = 0;
            let max_fs = dfs / 25 * 3;
            for (i, pkt) in packets.iter().enumerate() {
                let what = format!("dred {k} dec {dfs}/{dch} pkt {i}");
                let lost = i > 2
                    && (lost_count > 0 && lost_count < 6 && rng.range_i32(0, 2) > 0
                        || rng.range_i32(0, 9) == 0);
                if lost {
                    lost_count += 1;
                    continue;
                }
                let fmt = rand_fmt(&mut rng);
                if lost_count > 0 {
                    let output_samples = p.r.last_packet_duration() as i32;
                    assert_eq!(
                        output_samples,
                        p.c.ctl_get(sys::OPUS_GET_LAST_PACKET_DURATION_REQUEST)
                            .unwrap()
                    );
                    let dred_input = lost_count * output_samples;
                    let max = dred_input.clamp(0, dfs);
                    let rr = rdd
                        .parse(&mut rd, pkt, max, dfs, false)
                        .map_err(Error::code);
                    let cr = cdd.parse(&mut cd, pkt, max, dfs, false);
                    assert_eq!(rr, cr, "{what}: parse return");
                    cmp_dred(&rd, &cd, &what);
                    let dred_input = match cr {
                        Ok((n, _)) if n > 0 => n,
                        _ => 0,
                    };
                    for fr in 0..lost_count {
                        let w = format!("{what} lost {fr}/{lost_count}");
                        if fr == lost_count - 1 && opusorus::packet::has_lbrr(pkt) == Ok(true) {
                            p.step(Some(pkt), output_samples, 1, fmt, &w);
                        } else if dred_input > 0 {
                            p.dred_step(
                                &rd,
                                &cd,
                                (lost_count - fr) * output_samples,
                                output_samples,
                                fmt,
                                &w,
                            );
                        } else {
                            p.step(None, output_samples, 0, fmt, &w);
                        }
                    }
                    lost_count = 0;
                }
                p.step(Some(pkt), max_fs, 0, fmt, &what);
            }
            total += p.cov;
        }
    }
    note("dred_parse_process_decode", &total);
    eprintln!("dred_parse_process_decode: {parsed} packets parsed with DRED");
    assert!(total.dred_fec > 50, "DRED concealment coverage: {total:?}");
    assert!(parsed > 100);
}

/// DRED API edge cases vs C: non-DRED and malformed packets, unprocessed data, odd frame sizes
/// and offsets for `opus_decoder_dred_decode*`, sizes.
#[cfg(feature = "dred")]
#[test]
fn dred_api_edges() {
    use opusorus::dred::{Dred, DredDecoder};
    let mut rng = Rng::new(0xD4ED_0003);
    assert_eq!(Dred::get_size() as i32, di::Dred::get_size());
    let mut rdd = DredDecoder::new();
    // Rust-only: no model before the blob (C has it compiled in).
    let mut rd = Dred::new();
    assert_eq!(
        rdd.parse(&mut rd, &[0xF8, 1, 2], 960, 48000, false),
        Err(Error::Unimplemented)
    );
    assert_eq!(rdd.process_in_place(&mut rd), Err(Error::BadArg));
    assert_eq!(rdd.set_dnn_blob(&[0; 7]), Err(Error::BadArg));
    assert!(!rdd.loaded());
    rdd.set_dnn_blob(di::dred_decoder_blob()).unwrap();
    assert!(rdd.loaded());
    let mut cdd = di::DredDec::new().unwrap();
    for req in [sys::OPUS_SET_DNN_BLOB_REQUEST, 4000, 0] {
        assert_eq!(
            rdd.ctl_set(req, 0).map_err(Error::code),
            cdd.ctl_set(req, 0)
        );
    }
    // Unparsed data cannot be processed.
    let mut cd = di::Dred::new().unwrap();
    let mut rd2 = Dred::new();
    let mut cd2 = di::Dred::new().unwrap();
    assert_eq!(
        rdd.process(&rd, &mut rd2).map_err(Error::code),
        cdd.process(&cd, &mut cd2)
    );
    // Garbage / truncated / non-DRED packets.
    let mut c = EncCfg::silk(48000, 1, 20, 0);
    c.force_mode = 0;
    c.dred = 40;
    c.plp = 30;
    c.nframes = 60;
    let packets = c_stream(&c, 42);
    for i in 0..400 {
        let mut pkt = packets[i % packets.len()].clone();
        match i % 4 {
            0 => {}
            1 => {
                let l = rng.range_i32(1, pkt.len() as i32) as usize;
                pkt.truncate(l);
            }
            2 => {
                let j = rng.range_i32(0, pkt.len() as i32 - 1) as usize;
                pkt[j] ^= 1 << rng.range_i32(0, 7);
            }
            _ => {
                let l = rng.range_i32(1, 300) as usize;
                pkt = vec![0; l];
                rng.fill_bytes(&mut pkt);
            }
        }
        let what = format!("edge {i}");
        let max = rng.range_i32(-100, 60000);
        let sr = pick(&mut rng, &[8000, 12000, 16000, 24000, 48000]);
        let defer = rng.range_i32(0, 1) == 1;
        let rr = rdd
            .parse(&mut rd, &pkt, max, sr, defer)
            .map_err(Error::code);
        let cr = cdd.parse(&mut cd, &pkt, max, sr, defer);
        assert_eq!(rr, cr, "{what}: parse return");
        cmp_dred(&rd, &cd, &what);
    }
    // dred_decode argument errors and odd offsets/sizes on a processed DRED.
    let mut p = Pair::new(48000, 2);
    p.set(sys::OPUS_SET_COMPLEXITY_REQUEST, 6, "init");
    for pkt in &packets[..20] {
        p.step(Some(pkt), 5760, 0, Fmt::F32, "prime");
    }
    let pkt = &packets[25];
    let rr = rdd
        .parse(&mut rd, pkt, 48000, 48000, false)
        .map_err(Error::code);
    let cr = cdd.parse(&mut cd, pkt, 48000, 48000, false);
    assert_eq!(rr, cr);
    for i in 0..60 {
        let fsz = pick(&mut rng, &[-1, 0, 120, 240, 480, 960, 1000, 1920, 2880]);
        let off = rng.range_i32(-2000, 60000);
        let fmt = rand_fmt(&mut rng);
        p.dred_step(&rd, &cd, off, fsz, fmt, &format!("dred args {i}"));
    }
    // A Dred parsed with deferred processing (stage 1) is not used for concealment.
    let rr = rdd
        .parse(&mut rd, pkt, 48000, 48000, true)
        .map_err(Error::code);
    let cr = cdd.parse(&mut cd, pkt, 48000, 48000, true);
    assert_eq!(rr, cr);
    p.dred_step(&rd, &cd, 960, 960, Fmt::I16, "deferred");
}

// ---------------------------------------------------------------------------------------------
// Timing (run with `--release -- --ignored --nocapture`)
// ---------------------------------------------------------------------------------------------

/// Decode time Rust vs C with the DNN paths on: 20 s of WB SILK speech and 20 s of 48 kHz CELT
/// at complexity 10 (NoLACE, neural PLC) with 20 % loss.
#[test]
#[ignore = "timing only; run in release"]
fn dnn_decode_timing() {
    use std::time::Instant;
    let mut celt = EncCfg::silk(48000, 1, 20, 0);
    celt.force_mode = MODE_CELT_ONLY;
    celt.app = APP_AUDIO;
    celt.bitrate = 48000;
    for (name, mut cfg) in [
        (
            "SILK WB 20 ms, cplx 10, 20% loss",
            EncCfg::silk(16000, 1, 20, sys::OPUS_BANDWIDTH_WIDEBAND),
        ),
        ("CELT 48k 20 ms, cplx 10, 20% loss", celt),
    ] {
        cfg.nframes = 1000;
        let packets = c_stream(&cfg, 99);
        let mut rng = Rng::new(1);
        let lost: Vec<bool> = packets.iter().map(|_| rng.range_i32(0, 4) == 0).collect();
        let mut pcm = vec![0f32; 5760];
        let mut r = Decoder::new(cfg.fs, 1).unwrap();
        r.set_dnn_blob(di::decoder_blob()).unwrap();
        r.set_complexity(10).unwrap();
        let mut c = di::Dec::new(cfg.fs, 1).unwrap();
        c.ctl_set(sys::OPUS_SET_COMPLEXITY_REQUEST, 10).unwrap();
        let t = Instant::now();
        for (p, &l) in packets.iter().zip(&lost) {
            let d = if l { None } else { Some(&p[..]) };
            r.decode_float(d, &mut pcm, cfg.frame as usize, false)
                .unwrap();
        }
        let tr = t.elapsed().as_secs_f64();
        let t = Instant::now();
        for (p, &l) in packets.iter().zip(&lost) {
            let d = if l { None } else { Some(&p[..]) };
            c.decode_float(d, &mut pcm, cfg.frame, 0).unwrap();
        }
        let tc = t.elapsed().as_secs_f64();
        eprintln!(
            "{name}: rust {:.1} ms, C {:.1} ms, rust/C {:.2}",
            tr * 1e3,
            tc * 1e3,
            tr / tc
        );
    }
}
