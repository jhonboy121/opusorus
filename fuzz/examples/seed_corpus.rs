//! Reproducible seed-corpus generator for the fuzz targets.
//!
//! Real packets come from the C libopus encoder (the oracle) across many configurations:
//! every sampling rate, mono/stereo, the VoIP / audio / low-delay applications, SILK / hybrid /
//! CELT (forced and automatic, with mode switches mid-stream), 2.5-120 ms frames, CBR / VBR,
//! low to high bitrates, in-band FEC, DTX, QEXT (with the `qext` feature), surround
//! multistream (families 0 and 1) and ambisonics projection (orders 1-5). They are wrapped in
//! each target's input format (see `opusorus_fuzz::{dec, enc}` and the target docs).
//!
//! Usage (from `fuzz/`): `cargo run --release --example seed_corpus [-- <corpus dir>]`, or
//! `./seed_corpus.sh`. Output: `<corpus dir>/<target>/seed-NNNN` (default `fuzz/corpus`).
//! The output is deterministic; existing files of the same name are overwritten and other
//! corpus entries are left alone.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    reason = "offline generator: failures abort, progress goes to stdout"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "values are small by construction"
)]

use opusorus::encoder::request::{
    OPUS_SET_BANDWIDTH_REQUEST, OPUS_SET_BITRATE_REQUEST, OPUS_SET_COMPLEXITY_REQUEST,
    OPUS_SET_DTX_REQUEST, OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, OPUS_SET_FORCE_CHANNELS_REQUEST,
    OPUS_SET_FORCE_MODE_REQUEST, OPUS_SET_INBAND_FEC_REQUEST, OPUS_SET_LSB_DEPTH_REQUEST,
    OPUS_SET_MAX_BANDWIDTH_REQUEST, OPUS_SET_PACKET_LOSS_PERC_REQUEST,
    OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, OPUS_SET_PREDICTION_DISABLED_REQUEST,
    OPUS_SET_QEXT_REQUEST, OPUS_SET_SIGNAL_REQUEST, OPUS_SET_VBR_CONSTRAINT_REQUEST,
    OPUS_SET_VBR_REQUEST,
};
use opusorus::extensions::Extension;
use opusorus::{Repacketizer, packet};
use opusorus_fuzz::dec::{self, FSEL_MAX, FSEL_PACKET, Fmt, fsel_2_5ms};
use opusorus_fuzz::enc::{self, InFmt, Synth};
use opusorus_fuzz::{FS_LIST, Rng, Writer, fs_sel};
use opusorus_oracle::api as capi;
use std::path::{Path, PathBuf};

const VOIP: i32 = 2048;
const AUDIO: i32 = 2049;
const LOWDELAY: i32 = 2051;
const AUTO: i32 = -1000;

/// One encoded stream configuration.
#[derive(Debug, Clone)]
struct Cfg {
    fs: i32,
    ch: i32,
    app: i32,
    /// Frame duration in 0.1 ms units.
    dur: i32,
    /// CTLs applied before the first frame.
    init: Vec<(i32, i32)>,
    /// CTLs applied before frame `k`: `(k, request, value)`.
    sched: Vec<(usize, i32, i32)>,
    frames: usize,
    /// Signal kind per frame (cycled), see `Synth::fill`.
    kinds: Vec<u8>,
}

impl Cfg {
    const fn frame_size(&self) -> usize {
        (self.fs / 10 * self.dur / 1000) as usize
    }
}

fn synth_frame(s: &mut Synth, kind: u8, n: usize, ch: usize, fs: i32) -> Vec<f32> {
    s.fill(kind, 180, 440, n, ch, fs)
}

/// Encodes a stream with the C encoder; returns the packets (errors skip the frame).
fn encode_stream(cfg: &Cfg) -> Vec<Vec<u8>> {
    let mut e = capi::Encoder::new(cfg.fs, cfg.ch, cfg.app).expect("valid config");
    for &(req, val) in &cfg.init {
        e.ctl_set(req, val).expect("valid ctl");
    }
    let n = cfg.frame_size();
    let mut synth = Synth::new();
    let mut out = Vec::new();
    for k in 0..cfg.frames {
        for &(f, req, val) in &cfg.sched {
            if f == k {
                e.ctl_set(req, val).expect("valid ctl");
            }
        }
        let kind = cfg.kinds[k % cfg.kinds.len()];
        let x = synth_frame(&mut synth, kind, n, cfg.ch as usize, cfg.fs);
        let mut buf = vec![0u8; 1500];
        if let Ok(len) = e.encode_float(&x, n, &mut buf) {
            buf.truncate(len);
            out.push(buf);
        }
    }
    out
}

fn stream_configs() -> Vec<Cfg> {
    let mut rng = Rng::new(0x5EED);
    let mut v = Vec::new();
    let durs = [25, 50, 100, 200, 400, 600, 800, 1000, 1200];
    let rates = [
        6000, 10000, 16000, 24000, 32000, 48000, 64000, 96000, 128000, 256000,
    ];
    // Systematic: every rate x channels x application at 20 ms.
    for &fs in &FS_LIST[..5] {
        for ch in 1..=2 {
            for app in [VOIP, AUDIO, LOWDELAY] {
                v.push(Cfg {
                    fs,
                    ch,
                    app,
                    dur: 200,
                    init: vec![(OPUS_SET_BITRATE_REQUEST, 16000 * ch)],
                    sched: vec![],
                    frames: 8,
                    kinds: vec![3, 4],
                });
            }
        }
    }
    // Forced modes / bandwidths / durations.
    for mode in [1000, 1001, 1002] {
        for &dur in &durs {
            if mode != 1002 && dur < 100 {
                continue;
            }
            let ch = 1 + (dur / 100 % 2);
            v.push(Cfg {
                fs: 48000,
                ch,
                app: AUDIO,
                dur,
                init: vec![
                    (OPUS_SET_FORCE_MODE_REQUEST, mode),
                    (
                        OPUS_SET_BITRATE_REQUEST,
                        rates[(dur as usize) % rates.len()],
                    ),
                ],
                sched: vec![],
                frames: 6,
                kinds: vec![3, 1, 2],
            });
        }
    }
    // FEC, DTX, CBR / CVBR, complexity, mode switches.
    for i in 0..40 {
        let fs = FS_LIST[rng.below(5) as usize];
        let ch = 1 + rng.below(2) as i32;
        let app = [VOIP, AUDIO, LOWDELAY][rng.below(3) as usize];
        let dur = [100, 200, 400, 600][rng.below(4) as usize];
        let mut init = vec![
            (
                OPUS_SET_BITRATE_REQUEST,
                rates[rng.below(rates.len() as u64) as usize],
            ),
            (OPUS_SET_COMPLEXITY_REQUEST, rng.below(11) as i32),
        ];
        match i % 6 {
            0 => init.extend([
                (OPUS_SET_INBAND_FEC_REQUEST, 1),
                (OPUS_SET_PACKET_LOSS_PERC_REQUEST, 20),
                (OPUS_SET_SIGNAL_REQUEST, 3001),
            ]),
            1 => init.extend([(OPUS_SET_DTX_REQUEST, 1), (OPUS_SET_SIGNAL_REQUEST, 3001)]),
            2 => init.push((OPUS_SET_VBR_REQUEST, 0)),
            3 => init.extend([
                (OPUS_SET_VBR_CONSTRAINT_REQUEST, 1),
                (OPUS_SET_MAX_BANDWIDTH_REQUEST, 1101 + rng.below(5) as i32),
            ]),
            4 => init.extend([
                (OPUS_SET_PREDICTION_DISABLED_REQUEST, 1),
                (OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, 1),
                (OPUS_SET_LSB_DEPTH_REQUEST, 16),
            ]),
            _ => init.extend([
                (OPUS_SET_FORCE_CHANNELS_REQUEST, 1),
                (OPUS_SET_BANDWIDTH_REQUEST, 1101 + rng.below(5) as i32),
            ]),
        }
        let sched = if app == LOWDELAY {
            vec![]
        } else {
            vec![
                (3, OPUS_SET_FORCE_MODE_REQUEST, 1002),
                (5, OPUS_SET_FORCE_MODE_REQUEST, 1000),
                (7, OPUS_SET_FORCE_MODE_REQUEST, 1001),
                (9, OPUS_SET_FORCE_MODE_REQUEST, AUTO),
                (10, OPUS_SET_BITRATE_REQUEST, 64000),
            ]
        };
        v.push(Cfg {
            fs,
            ch,
            app,
            dur,
            init,
            sched,
            frames: 12,
            kinds: vec![3, 3, 0, 0, 4, 1, 3, 7],
        });
    }
    // Expert frame durations.
    for fd in 5001..=5009 {
        v.push(Cfg {
            fs: 48000,
            ch: 2,
            app: AUDIO,
            dur: 1200,
            init: vec![(OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, fd)],
            sched: vec![],
            frames: 3,
            kinds: vec![4],
        });
    }
    if cfg!(feature = "qext") {
        for (fs, ch, br) in [(48000, 2, 256000), (96000, 2, 320000), (96000, 1, 160000)] {
            v.push(Cfg {
                fs,
                ch,
                app: AUDIO,
                dur: 200,
                init: vec![(OPUS_SET_QEXT_REQUEST, 1), (OPUS_SET_BITRATE_REQUEST, br)],
                sched: vec![],
                frames: 6,
                kinds: vec![1, 4, 6],
            });
        }
    }
    v
}

struct Out {
    root: PathBuf,
    counts: std::collections::BTreeMap<&'static str, usize>,
}

impl Out {
    fn put(&mut self, target: &'static str, data: &[u8]) {
        let dir = self.root.join(target);
        std::fs::create_dir_all(&dir).expect("create corpus dir");
        let n = self.counts.entry(target).or_insert(0);
        std::fs::write(dir.join(format!("seed-{n:04}")), data).expect("write seed");
        *n += 1;
    }
}

/// Decoder op stream for a packet sequence: normal decodes with a few losses, FEC recoveries,
/// a gain change and an nb_samples query.
fn decode_ops(w: &mut Writer, packets: &[Vec<u8>], fs: i32, seed: u64) {
    let mut rng = Rng::new(seed);
    let fmts = [Fmt::I16, Fmt::I24, Fmt::F32];
    let mut i = 0;
    while i < packets.len() {
        let p = &packets[i];
        let fmt = fmts[rng.below(3) as usize];
        let dur = match packet::get_nb_samples(p, fs) {
            Ok(d) => d,
            Err(_) => fs / 50,
        };
        let k = (dur / (fs / 400)).clamp(1, 64) as u8;
        match rng.below(12) {
            0 if i + 1 < packets.len() => {
                // Lost packet recovered by FEC from the next one, then the next one.
                dec::write_decode(w, &packets[i + 1], fmt, true, fsel_2_5ms(k));
                i += 1;
                dec::write_decode(w, &packets[i], fmt, false, FSEL_PACKET);
            }
            1 => dec::write_lost(w, fmt, fsel_2_5ms(k)),
            2 => {
                w.u8(dec::tag::CTL);
                dec::write_ctl(w, 1, rng.below(5) as usize);
                dec::write_decode(w, p, fmt, false, FSEL_MAX);
            }
            3 => {
                w.u8(dec::tag::NB_SAMPLES).blob(p);
                dec::write_decode(w, p, fmt, false, FSEL_PACKET);
            }
            _ => dec::write_decode(w, p, fmt, false, FSEL_PACKET),
        }
        i += 1;
    }
}

fn decoder_seeds(out: &mut Out, streams: &[(Cfg, Vec<Vec<u8>>)]) {
    let mut rng = Rng::new(0xDEC0);
    for (idx, (cfg, packets)) in streams.iter().enumerate() {
        // Decode at the encoder's configuration and at a random other one.
        for variant in 0..2 {
            let (fs, ch) = if variant == 0 {
                (cfg.fs, cfg.ch)
            } else {
                (
                    FS_LIST[rng.below(FS_LIST.len() as u64) as usize],
                    1 + rng.below(2) as i32,
                )
            };
            let mut w = Writer::new();
            w.u8(fs_sel(fs)).u8((ch - 1) as u8);
            decode_ops(&mut w, packets, fs, idx as u64 * 2 + variant);
            out.put("decode", &w.0);
            out.put("differential_decode", &w.0);
        }
    }
}

/// Encoder op stream for a configuration (same content as `encode_stream`, in the fuzz
/// format).
fn encoder_input(cfg: &Cfg, fmt_seed: u64) -> Writer {
    let mut w = Writer::new();
    enc::write_header(&mut w, cfg.fs, cfg.ch, cfg.app);
    for &(req, val) in &cfg.init {
        enc::write_ctl(&mut w, req, val);
    }
    let fsel = [25, 50, 100, 200, 400, 600, 800, 1000, 1200]
        .iter()
        .position(|&d| d == cfg.dur)
        .expect("valid duration") as u8;
    let fmts = [InFmt::I16, InFmt::I24, InFmt::F32];
    let fmt = fmts[(fmt_seed % 3) as usize];
    for k in 0..cfg.frames {
        for &(f, req, val) in &cfg.sched {
            if f == k {
                enc::write_ctl(&mut w, req, val);
            }
        }
        let kind = cfg.kinds[k % cfg.kinds.len()];
        enc::write_frame(&mut w, fmt, enc::fsel_dur(fsel), 1500, kind, 180, 440);
    }
    w
}

fn encoder_seeds(out: &mut Out, cfgs: &[Cfg]) {
    for (i, cfg) in cfgs.iter().enumerate() {
        // Encoding is slow: keep the encoder seeds short.
        let mut c = cfg.clone();
        c.frames = c.frames.min(6);
        let w = encoder_input(&c, i as u64);
        out.put("encode", &w.0);
        out.put("differential_encode", &w.0);
        let mut rt = Writer::new();
        rt.u8((i % 256) as u8).bytes(&w.0);
        out.put("roundtrip_invariants", &rt.0);
    }
    // A few raw-sample inputs (non-finite floats, extreme 24-bit values).
    for (fs, ch) in [(48000, 2), (16000, 1)] {
        for f in 0..3u8 {
            let mut w = Writer::new();
            enc::write_header(&mut w, fs, ch, AUDIO);
            let raw: Vec<u8> = match f {
                0 => [i16::MAX, i16::MIN, 0, 1, -1]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                1 => [0x7FFF_FFFF_i32, -0x8000_0000, 8_388_607, -8_388_608, 0]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                _ => [f32::NAN, f32::INFINITY, 1e30, -1.0, 0.5, f32::NEG_INFINITY]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            };
            for _ in 0..3 {
                w.u8(enc::tag::FRAME)
                    .u8(f)
                    .u8(3)
                    .u16(1500)
                    .u8(0x80)
                    .u8(0)
                    .u16(0)
                    .blob(&raw);
                enc::write_frame(&mut w, InFmt::F32, 3, 1500, 3, 200, 200);
            }
            out.put("encode", &w.0);
            out.put("differential_encode", &w.0);
        }
    }
}

fn repacketizer_seeds(out: &mut Out, streams: &[(Cfg, Vec<Vec<u8>>)]) {
    for (cfg, packets) in streams.iter().filter(|(c, _)| c.dur <= 200).take(40) {
        let mut w = Writer::new();
        // Cat consecutive packets that share a TOC configuration.
        let Some(first) = packets.first() else {
            continue;
        };
        let toc = first[0] & 0xFC;
        let same: Vec<&Vec<u8>> = packets
            .iter()
            .filter(|p| p[0] & 0xFC == toc)
            .take(6)
            .collect();
        for p in &same {
            w.u8(0).blob(p);
        }
        w.u8(3).u8(0).u8(2).u16(1500);
        w.u8(3).u8(1).u8(same.len() as u8).u16(4000);
        w.u8(4).u16(4000);
        w.u8(4).u16(10);
        w.u8(5);
        if let Some(p) = same.first() {
            w.u8(6).u16(100).blob(p);
            w.u8(6).u16(300).blob(p);
        }
        if cfg.ch == 2 {
            w.u8(0).blob(&packets[0]);
        }
        out.put("repacketizer", &w.0);
    }
}

fn padded_with_extensions(p: &[u8], rng: &mut Rng) -> Option<Vec<u8>> {
    let mut rp = Repacketizer::new();
    if rp.cat(p).is_err() {
        return None;
    }
    let nb = rp.nb_frames();
    let payload: Vec<u8> = (0..64).map(|_| rng.below(256) as u8).collect();
    let mut exts = Vec::new();
    for f in 0..nb {
        exts.push(Extension {
            id: 3 + rng.below(29) as i32,
            frame: f,
            data: &payload[..rng.below(2) as usize],
        });
        exts.push(Extension {
            id: 32 + rng.below(96) as i32,
            frame: f,
            data: &payload[..rng.below(40) as usize],
        });
    }
    let mut buf = vec![0u8; p.len() + 400];
    match rp.out_range_impl(0, nb, &mut buf, false, rng.below(2) == 0, &exts) {
        Ok(n) => {
            buf.truncate(n as usize);
            Some(buf)
        }
        // Too many extensions for this packet: no seed.
        Err(_) => None,
    }
}

fn packet_seeds(out: &mut Out, streams: &[(Cfg, Vec<Vec<u8>>)]) {
    let mut rng = Rng::new(0xE47);
    for (_, packets) in streams.iter().take(60) {
        for p in packets.iter().take(2) {
            let mut w = Writer::new();
            w.u8(0).bytes(p);
            out.put("packet_parse_extensions", &w.0);
            if let Some(q) = padded_with_extensions(p, &mut rng) {
                let mut w = Writer::new();
                w.u8(0).bytes(&q);
                out.put("packet_parse_extensions", &w.0);
                // The same packet in the decoder targets (extensions in the padding).
                let mut w = Writer::new();
                w.u8(4).u8(1);
                dec::write_decode(&mut w, &q, Fmt::F32, false, FSEL_PACKET);
                out.put("differential_decode", &w.0);
                // The padding alone as extension data.
                if let Ok(parsed) = packet::parse(&q)
                    && !parsed.padding.is_empty()
                {
                    let mut w = Writer::new();
                    w.u8(1).u8(parsed.nb_frames as u8).u8(4);
                    for op in [3u8, 3, 0, 1] {
                        w.u8(op).u8(40);
                    }
                    w.bytes(parsed.padding);
                    out.put("packet_parse_extensions", &w.0);
                }
            }
        }
    }
    // Extension-generation records.
    for i in 0..20u8 {
        let mut w = Writer::new();
        let nbf = 1 + i % 6;
        w.u8(2)
            .u8(nbf)
            .u8(i & 1)
            .u16(200 + u16::from(i) * 30)
            .u8(2 * nbf);
        for f in 0..nbf {
            w.u8(3 + i).u8(f).u8(1).u8(0xAA);
            w.u8(40 + i).u8(f).u8(5).bytes(&[1, 2, 3, 4, 5]);
        }
        out.put("packet_parse_extensions", &w.0);
    }
    // Soft clip.
    for ch in 1..=2u8 {
        let mut w = Writer::new();
        w.u8(3).u8(ch).u8(64);
        w.bytes(&0.0f32.to_le_bytes()).bytes(&0.0f32.to_le_bytes());
        for i in 0..64 * u32::from(ch) {
            let v = (i as f32 * 0.37).sin() * 2.5;
            w.bytes(&v.to_le_bytes());
        }
        out.put("packet_parse_extensions", &w.0);
    }
}

fn multistream_seeds(out: &mut Out) {
    let mut rng = Rng::new(0x3570);
    for family in [0, 1] {
        for channels in 1..=8 {
            if family == 0 && channels > 2 {
                continue;
            }
            for (fs, app) in [(48000, AUDIO), (24000, VOIP), (16000, LOWDELAY)] {
                let Ok(mut e) = capi::MsEncoder::new_surround(fs, channels, family, app) else {
                    continue;
                };
                let n = (fs / 50) as usize;
                let mut synth = Synth::new();
                let mut w = Writer::new();
                w.u8(fs_sel(fs))
                    .u8(channels as u8)
                    .u8(e.streams as u8)
                    .u8(e.coupled_streams as u8)
                    .bytes(&e.mapping);
                for k in 0..6 {
                    // Synthesize per channel pair from the 2-channel generator.
                    let mut x = Vec::with_capacity(n * channels as usize);
                    let pairs: Vec<Vec<f32>> = (0..(channels + 1) / 2)
                        .map(|_| synth.fill(3 + (k % 2) as u8, 150, 300, n, 2, fs))
                        .collect();
                    for s in 0..n {
                        for c in 0..channels as usize {
                            x.push(pairs[c / 2][2 * s + c % 2]);
                        }
                    }
                    let mut buf = vec![0u8; 4000];
                    if let Ok(len) = e.encode_float(&x, n, &mut buf) {
                        let fmt = [Fmt::I16, Fmt::I24, Fmt::F32][rng.below(3) as usize];
                        if rng.below(6) == 0 {
                            dec::write_lost(&mut w, fmt, fsel_2_5ms(8));
                        }
                        dec::write_decode(&mut w, &buf[..len], fmt, false, FSEL_PACKET);
                    }
                }
                out.put("multistream_decode", &w.0);
                // Repacketizer: multistream pad/unpad of one packet.
                let mut buf = vec![0u8; 4000];
                let x = vec![0.1f32; n * channels as usize];
                if let Ok(len) = e.encode_float(&x, n, &mut buf) {
                    let mut w = Writer::new();
                    w.u8(7).u8(e.streams as u8 + 1).u16(77).blob(&buf[..len]);
                    out.put("repacketizer", &w.0);
                }
            }
        }
    }
}

fn projection_seeds(out: &mut Out) {
    for channels in [4, 6, 9, 11, 16, 18, 25, 27, 36, 38] {
        for (fs, app) in [(48000, AUDIO), (16000, VOIP)] {
            let Ok(mut e) = capi::ProjectionEncoder::new(fs, channels, 3, app) else {
                continue;
            };
            let matrix = e.demixing_matrix().expect("matrix");
            let n = (fs / 50) as usize;
            let mut synth = Synth::new();
            let mut w = Writer::new();
            w.u8(fs_sel(fs))
                .u8(channels as u8)
                .u8(e.streams as u8)
                .u8(e.coupled_streams as u8)
                .blob(&matrix);
            for k in 0..5 {
                let mono = synth.fill(3 + (k % 2) as u8, 150, 220, n, 1, fs);
                let mut x = Vec::with_capacity(n * channels as usize);
                for s in &mono {
                    for c in 0..channels {
                        x.push(s * (1.0 / (1.0 + c as f32)));
                    }
                }
                let mut buf = vec![0u8; 8000];
                if let Ok(len) = e.encode_float(&x, n, &mut buf) {
                    let fmt = [Fmt::I16, Fmt::I24, Fmt::F32][k % 3];
                    dec::write_decode(&mut w, &buf[..len], fmt, false, FSEL_PACKET);
                }
            }
            out.put("projection_decode", &w.0);
        }
    }
}

fn ms_encoder_seeds(out: &mut Out) {
    const FAMILIES: [i32; 6] = [0, 1, 2, 3, 255, 254];
    let layouts = [
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 6),
        (1, 8),
        (2, 4),
        (2, 11),
        (3, 4),
        (3, 6),
        (3, 9),
        (3, 16),
        (255, 3),
        (255, 5),
    ];
    for (i, &(family, channels)) in layouts.iter().enumerate() {
        for (fs, app) in [(48000, AUDIO), (16000, VOIP)] {
            let mut w = Writer::new();
            let fam = FAMILIES.iter().position(|&f| f == family).expect("listed") as u8;
            let a = enc::APPS.iter().position(|&x| x == app).expect("listed") as u8;
            w.u8(fs_sel(fs)).u8(channels as u8).u8(fam).u8(a);
            enc::write_ctl(&mut w, OPUS_SET_BITRATE_REQUEST, 24000 * channels);
            if i % 3 == 0 {
                enc::write_ctl(&mut w, OPUS_SET_COMPLEXITY_REQUEST, 5);
            }
            for k in 0..4u8 {
                let fmt = [InFmt::I16, InFmt::F32, InFmt::I24][usize::from(k % 3)];
                enc::write_frame(&mut w, fmt, enc::fsel_dur(3), 1500, 3 + k % 2, 170, 300);
            }
            out.put("differential_ms_encode", &w.0);
        }
    }
}

/// Seeds for `differential_dnn_decode`: real streams decoded with deep PLC / OSCE active
/// (complexity 5-10, BWE on or off) and frequent losses, including bursts.
#[cfg(feature = "deep-plc")]
fn dnn_seeds(out: &mut Out, streams: &[(Cfg, Vec<Vec<u8>>)]) {
    let mut rng = Rng::new(0xD1FF);
    let fmts = [Fmt::I16, Fmt::I24, Fmt::F32];
    for (idx, (cfg, packets)) in streams.iter().enumerate() {
        // Every other stream is decoded at 48 kHz (OSCE BWE works on 16 kHz SILK at 48 kHz).
        let fs = if idx % 2 == 0 { 48000 } else { cfg.fs };
        let mut w = Writer::new();
        // Complexity selector (5, 6, 7, 10, 8, 9, 4, 0) and BWE bit.
        let dnn = (idx % 6) as u8 | if idx % 3 == 0 { 8 } else { 0 };
        w.u8(fs_sel(fs)).u8((cfg.ch - 1) as u8).u8(dnn);
        let mut i = 0;
        while i < packets.len() {
            let p = &packets[i];
            let fmt = fmts[rng.below(3) as usize];
            let dur = match packet::get_nb_samples(p, fs) {
                Ok(d) => d.max(fs / 400),
                Err(_) => fs / 50,
            };
            let k = (dur / (fs / 400)).clamp(1, 64) as u8;
            match rng.below(10) {
                0..=2 => {
                    // A burst of 1-4 losses, then the packet.
                    for _ in 0..=rng.below(4) {
                        dec::write_lost(&mut w, fmt, fsel_2_5ms(k));
                    }
                    dec::write_decode(&mut w, p, fmt, false, FSEL_PACKET);
                }
                3 if i + 1 < packets.len() => {
                    dec::write_decode(&mut w, &packets[i + 1], fmt, true, fsel_2_5ms(k));
                    i += 1;
                    dec::write_decode(&mut w, &packets[i], fmt, false, FSEL_PACKET);
                }
                _ => dec::write_decode(&mut w, p, fmt, false, FSEL_PACKET),
            }
            i += 1;
        }
        out.put("differential_dnn_decode", &w.0);
    }
}

/// Seeds for `differential_custom`: static and custom modes, mono / stereo, every PCM format,
/// CBR / VBR, packets decoded (some dropped: PLC), band limits and decoder CTLs.
#[cfg(feature = "custom-modes")]
fn custom_seeds(out: &mut Out) {
    use opusorus::celt::celt::CELT_SET_SIGNALLING_REQUEST;
    use opusorus_fuzz::custom::{self, DEC_SETS, ENC_SETS, FRAME, FS, index_of, set_index};
    let mut rng = Rng::new(0xC057);
    let mut modes = vec![
        (48000, 960),
        (48000, 480),
        (48000, 240),
        (48000, 120),
        (44100, 882),
        (44100, 1024),
        (32000, 640),
        (48000, 256),
        (24000, 400),
        (16000, 320),
        (48000, 720),
        (8000, 160),
        (22050, 512),
    ];
    if cfg!(feature = "qext") {
        modes.extend([(96000, 1920), (48000, 2048), (96000, 1440)]);
    }
    for (mi, &(fs, frame)) in modes.iter().enumerate() {
        for ch in 0..4u8 {
            let mut w = Writer::new();
            w.u8(index_of(&FS, fs)).u8(index_of(&FRAME, frame)).u8(ch);
            let enc_ctl = |w: &mut Writer, req: i32, v: i32| {
                w.u8(custom::tag::ENC_CTL)
                    .u8(set_index(&ENC_SETS, req))
                    .u8(0x80)
                    .i32(v);
            };
            enc_ctl(
                &mut w,
                OPUS_SET_BITRATE_REQUEST,
                [32000, 64000, 128000][mi % 3] * (1 + i32::from(ch & 1)),
            );
            enc_ctl(&mut w, OPUS_SET_VBR_REQUEST, i32::from(mi % 2 == 0));
            enc_ctl(&mut w, OPUS_SET_COMPLEXITY_REQUEST, (mi % 11) as i32);
            if mi % 4 == 3 {
                enc_ctl(&mut w, CELT_SET_SIGNALLING_REQUEST, 0);
                w.u8(custom::tag::DEC_CTL)
                    .u8(set_index(&DEC_SETS, CELT_SET_SIGNALLING_REQUEST))
                    .u8(0x80)
                    .i32(0);
            }
            if cfg!(feature = "qext") && mi % 4 == 1 {
                enc_ctl(&mut w, OPUS_SET_QEXT_REQUEST, 1);
            }
            if mi % 5 == 2 {
                // Hybrid-like band range on both sides.
                w.u8(custom::tag::BANDS).u8(0).u8(2).u8(0);
                w.u8(custom::tag::BANDS).u8(1).u8(2).u8(0);
            }
            for k in 0..8u8 {
                let fin = rng.below(3) as u8;
                let fout = rng.below(3) as u8;
                let drop = u8::from(rng.below(6) == 0);
                let fsel = if k % 4 == 3 { 1 } else { 0 };
                w.u8(custom::tag::ENCODE)
                    .u8(fin | (fout << 2) | (drop << 4))
                    .u8(fsel)
                    .u16(1 + [120, 200, 400, 1275][usize::from(k % 4)])
                    .u8([3, 4, 1, 2, 6][usize::from(k % 5)])
                    .u8(160)
                    .u16(300 + 50 * u16::from(k));
            }
            w.u8(custom::tag::PLC).u8(0).u8(0);
            out.put("differential_custom", &w.0);
        }
    }
}

fn main() {
    let root = std::env::args().nth(1).map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus"),
        PathBuf::from,
    );
    let mut out = Out {
        root,
        counts: std::collections::BTreeMap::new(),
    };
    let cfgs = stream_configs();
    let streams: Vec<(Cfg, Vec<Vec<u8>>)> = cfgs
        .iter()
        .map(|c| (c.clone(), encode_stream(c)))
        .filter(|(_, p)| !p.is_empty())
        .collect();
    decoder_seeds(&mut out, &streams);
    encoder_seeds(&mut out, &cfgs);
    repacketizer_seeds(&mut out, &streams);
    packet_seeds(&mut out, &streams);
    multistream_seeds(&mut out);
    projection_seeds(&mut out);
    ms_encoder_seeds(&mut out);
    #[cfg(feature = "deep-plc")]
    dnn_seeds(&mut out, &streams);
    #[cfg(feature = "custom-modes")]
    custom_seeds(&mut out);
    for (t, n) in &out.counts {
        println!("{t}: {n} seeds");
    }
    println!("corpus written to {}", out.root.display());
}
