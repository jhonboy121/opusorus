//! Feature `fuzzing` (libopus `FUZZING`) vs the oracle built with the same define.
//!
//! The fuzzing encoder draws random decisions from the C library's process-wide `rand()`; the
//! port draws them from its own process-wide glibc-compatible generator
//! (`opusorus::glibc_rand`). Every test here seeds both generators with the same seed and then
//! runs the same sequence of encoder/decoder calls on both sides, so the two generators stay in
//! step and the outputs must be bit-exact. The tests of this file hold one lock while they use
//! the generators (the other test files must not run the encoder of a fuzzing build in
//! parallel with a C encoder; `just test-options` runs only this file and the decoder suites
//! with the feature).
//!
//! * `glibc_generator_matches_c`: `srand`/`rand` sequences for several seeds (also on hosts
//!   without the feature: the generator is used by `opus_demo`). Needs a glibc host.
//! * `opus_encoder_matches_c`: single-stream encoders over applications, rates, channel
//!   counts, bitrates, frame sizes, complexities and VBR modes (random mode, mono/stereo,
//!   transient, TF, spreading, tapset, trim, band skip, silence and anti-collapse decisions;
//!   with `qext`, the random extra-allocation depths): packets, final ranges and the decoded
//!   PCM of both decoders (whose QEXT allocation also draws from the generator).
//! * `multistream_encoder_matches_c`: surround encoders (several encoders per call).
//! * The `opus_demo` comparison (`tests/vectors.rs`, which serializes its runs in fuzzing builds)
//!   covers the interleaving of the tool's loss simulation draws with the encoder's.
#![allow(
    clippy::unwrap_used,
    clippy::print_stderr,
    reason = "test code: failures should panic; statistics go to stderr"
)]

use std::sync::Mutex;

use opusorus::glibc_rand;
use opusorus_oracle::api;
use opusorus_oracle::build_options as c;

/// Serializes the tests of this file (they reseed the process-wide generators).
static LOCK: Mutex<()> = Mutex::new(());

fn seed_both(seed: u32) {
    glibc_rand::srand(seed);
    c::srand(seed);
}

#[test]
fn oracle_has_the_same_option() {
    let on = cfg!(feature = "fuzzing");
    assert_eq!(c::options().fuzzing, on);
    assert_eq!(glibc_rand::FUZZING, on);
    assert_eq!(
        opusorus::celt::celt::opus_get_version_string().ends_with("-fuzzing"),
        on
    );
    assert_eq!(api::version_string().ends_with("-fuzzing"), on);
}

#[test]
fn glibc_generator_matches_c() {
    let _g = LOCK.lock().unwrap();
    for seed in [
        0u32,
        1,
        2,
        42,
        12345,
        0x7fff_ffff,
        0x8000_0000,
        0x8000_0001,
        u32::MAX,
    ] {
        seed_both(seed);
        for i in 0..20_000 {
            assert_eq!(glibc_rand::rand(), c::rand(), "seed {seed:#x}, draw {i}");
        }
    }
}

#[cfg(feature = "fuzzing")]
mod encoders {
    use super::*;
    use opusorus::{Decoder, Encoder, MsDecoder, MsEncoder};
    use opusorus_conformance::{Rng, signals};
    use opusorus_oracle::sys;

    #[derive(Debug, Clone, Copy)]
    struct Cfg {
        fs: i32,
        ch: i32,
        app: i32,
        bitrate: i32,
        /// Frame size in units of 2.5 ms.
        frame: i32,
        complexity: i32,
        vbr: i32,
        constrained: i32,
        qext: bool,
        fec: bool,
        dtx: bool,
        frames: usize,
    }

    /// Runs one configuration on both sides; returns the TOC bytes seen (for coverage).
    fn run_single(cfg: &Cfg, seed: u32) -> Vec<u8> {
        let what = format!("{cfg:?} seed {seed}");
        let n = (cfg.fs as usize / 400) * cfg.frame as usize;
        let ch = cfg.ch as usize;
        let len = n * cfg.frames;
        let x = if seed.is_multiple_of(2) {
            signals::music_like(len, ch, cfg.fs as u32, u64::from(seed))
        } else {
            signals::speech_like(len, ch, cfg.fs as u32, u64::from(seed))
        };
        let pcm = signals::to_i16(&x);
        let mut re = Encoder::new_raw(cfg.fs, cfg.ch, cfg.app).unwrap();
        let mut ce = api::Encoder::new(cfg.fs, cfg.ch, cfg.app).unwrap();
        let mut rd = Decoder::new(cfg.fs, cfg.ch).unwrap();
        let mut cd = api::Decoder::new(cfg.fs, cfg.ch).unwrap();
        let mut ctls = vec![
            (sys::OPUS_SET_BITRATE_REQUEST, cfg.bitrate),
            (sys::OPUS_SET_COMPLEXITY_REQUEST, cfg.complexity),
            (sys::OPUS_SET_VBR_REQUEST, cfg.vbr),
            (sys::OPUS_SET_VBR_CONSTRAINT_REQUEST, cfg.constrained),
            (sys::OPUS_SET_INBAND_FEC_REQUEST, i32::from(cfg.fec)),
            (
                sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST,
                if cfg.fec { 20 } else { 0 },
            ),
            (sys::OPUS_SET_DTX_REQUEST, i32::from(cfg.dtx)),
        ];
        if cfg.qext {
            ctls.push((sys::OPUS_SET_QEXT_REQUEST, 1));
        }
        for (r, v) in ctls {
            re.ctl_set(r, v).unwrap();
            ce.ctl_set(r, v).unwrap();
        }
        seed_both(seed);
        let mut rbuf = vec![0u8; 4000];
        let mut cbuf = vec![0u8; 4000];
        let mut rout = vec![0i16; n * ch];
        let mut cout = vec![0i16; n * ch];
        let mut tocs = Vec::new();
        for f in 0..cfg.frames {
            let frame = &pcm[f * n * ch..(f + 1) * n * ch];
            let rl = re
                .encode(frame, n, &mut rbuf)
                .map_err(opusorus::Error::code);
            let cl = ce.encode(frame, n, &mut cbuf);
            assert_eq!(rl, cl, "{what}: frame {f}: length");
            let Ok(l) = rl else { continue };
            assert_eq!(rbuf[..l], cbuf[..l], "{what}: frame {f}: packet bytes");
            assert_eq!(
                re.final_range(),
                ce.final_range().unwrap(),
                "{what}: frame {f}: final range"
            );
            tocs.push(rbuf[0]);
            let rn = rd.decode(Some(&rbuf[..l]), &mut rout, n, false);
            let cn = cd.decode(Some(&cbuf[..l]), &mut cout, n, false);
            assert_eq!(
                rn.map_err(opusorus::Error::code),
                cn,
                "{what}: frame {f}: decode"
            );
            assert_eq!(rout, cout, "{what}: frame {f}: decoded PCM");
            assert_eq!(
                rd.final_range(),
                cd.final_range().unwrap(),
                "{what}: frame {f}"
            );
        }
        // The generators were used in step.
        assert_eq!(glibc_rand::rand(), c::rand(), "{what}: generator state");
        tocs
    }

    const VOIP: i32 = sys::OPUS_APPLICATION_VOIP;
    const AUDIO: i32 = sys::OPUS_APPLICATION_AUDIO;
    const LOWDELAY: i32 = sys::OPUS_APPLICATION_RESTRICTED_LOWDELAY;

    #[test]
    fn opus_encoder_matches_c() {
        let _g = LOCK.lock().unwrap();
        let base = Cfg {
            fs: 48000,
            ch: 2,
            app: AUDIO,
            bitrate: 64000,
            frame: 8,
            complexity: 10,
            vbr: 1,
            constrained: 1,
            qext: false,
            fec: false,
            dtx: false,
            frames: 60,
        };
        let mut cfgs = vec![
            base,
            Cfg {
                app: VOIP,
                ch: 1,
                bitrate: 24000,
                ..base
            },
            Cfg {
                fs: 16000,
                app: VOIP,
                ch: 1,
                bitrate: 16000,
                ..base
            },
            Cfg {
                fs: 16000,
                app: VOIP,
                ch: 2,
                bitrate: 32000,
                fec: true,
                ..base
            },
            Cfg {
                app: LOWDELAY,
                bitrate: 128_000,
                frame: 4,
                ..base
            },
            Cfg {
                fs: 24000,
                bitrate: 48000,
                frame: 2,
                ..base
            },
            Cfg {
                frame: 1,
                bitrate: 96000,
                complexity: 5,
                vbr: 0,
                ..base
            },
            Cfg {
                frame: 24,
                bitrate: 32000,
                complexity: 3,
                constrained: 0,
                ..base
            },
            Cfg {
                frame: 48,
                bitrate: 20000,
                ch: 1,
                dtx: true,
                app: VOIP,
                frames: 12,
                ..base
            },
            Cfg {
                fs: 12000,
                ch: 1,
                bitrate: 12000,
                complexity: 0,
                ..base
            },
            Cfg {
                fs: 8000,
                ch: 2,
                bitrate: 10000,
                ..base
            },
            Cfg {
                bitrate: 510_000,
                frame: 8,
                ..base
            },
        ];
        if cfg!(feature = "qext") {
            cfgs.push(Cfg {
                fs: 96000,
                bitrate: 256_000,
                qext: true,
                ..base
            });
            cfgs.push(Cfg {
                fs: 96000,
                bitrate: 510_000,
                frame: 4,
                qext: true,
                ..base
            });
            cfgs.push(Cfg {
                bitrate: 256_000,
                qext: true,
                app: LOWDELAY,
                ..base
            });
        }
        let mut rng = Rng::new(0xF022);
        let mut tocs = std::collections::BTreeSet::new();
        for (i, cfg) in cfgs.iter().enumerate() {
            for s in 0..3 {
                let seed = rng.next_u32() >> (i % 3);
                tocs.extend(run_single(cfg, seed + s));
            }
        }
        // The random mode decisions reach SILK, hybrid and CELT configurations.
        let modes: std::collections::BTreeSet<u8> = tocs
            .iter()
            .map(|t| match t >> 3 {
                0..=11 => 0,
                12..=15 => 1,
                _ => 2,
            })
            .collect();
        assert_eq!(modes.len(), 3, "TOC configurations seen: {tocs:?}");
        eprintln!("{} TOC bytes seen", tocs.len());
    }

    #[test]
    fn multistream_encoder_matches_c() {
        let _g = LOCK.lock().unwrap();
        for (fs, ch, family, bitrate, frame, seed) in [
            (48000, 6, 1, 256_000, 960, 7u32),
            (48000, 3, 1, 96000, 480, 8),
            (24000, 4, 255, 64000, 480, 9),
            (48000, 2, 0, 48000, 1920, 10),
        ] {
            if cfg!(all(feature = "fixed-point", feature = "assertions")) && family == 1 {
                // The LFE stream's band energy wraps under the random decisions and fails a
                // celt_sig_assert (C aborts): see `celt_ilog2_release`.
                continue;
            }
            let what = format!("{fs} Hz {ch} ch family {family}");
            let mut re = MsEncoder::new_surround_raw(fs, ch, family, AUDIO).unwrap();
            let mut ce = api::MsEncoder::new_surround(fs, ch, family, AUDIO).unwrap();
            assert_eq!(
                (re.streams(), re.coupled_streams(), re.mapping()),
                (ce.streams, ce.coupled_streams, &ce.mapping[..])
            );
            re.ctl_set(sys::OPUS_SET_BITRATE_REQUEST, bitrate).unwrap();
            ce.ctl_set(sys::OPUS_SET_BITRATE_REQUEST, bitrate).unwrap();
            let mut rd =
                MsDecoder::new(fs, ch, re.streams(), re.coupled_streams(), re.mapping()).unwrap();
            let mut cd =
                api::MsDecoder::new(fs, ch, ce.streams, ce.coupled_streams, &ce.mapping).unwrap();
            let chu = ch as usize;
            let frames = 40;
            let x = signals::music_like(frame * frames, chu, fs as u32, u64::from(seed));
            let pcm = signals::to_i16(&x);
            seed_both(seed);
            let mut rbuf = vec![0u8; 8000];
            let mut cbuf = vec![0u8; 8000];
            let mut rout = vec![0i16; frame * chu];
            let mut cout = vec![0i16; frame * chu];
            for f in 0..frames {
                let input = &pcm[f * frame * chu..(f + 1) * frame * chu];
                let rl = re
                    .encode(input, frame, &mut rbuf)
                    .map_err(opusorus::Error::code);
                let cl = ce.encode(input, frame, &mut cbuf);
                assert_eq!(rl, cl, "{what}: frame {f}: length");
                let l = rl.unwrap();
                assert_eq!(rbuf[..l], cbuf[..l], "{what}: frame {f}: packet");
                assert_eq!(
                    re.final_range(),
                    ce.final_range().unwrap(),
                    "{what}: frame {f}"
                );
                let rn = rd.decode(Some(&rbuf[..l]), &mut rout, frame, false);
                let cn = cd.decode(Some(&cbuf[..l]), &mut cout, frame, false);
                assert_eq!(rn.map_err(opusorus::Error::code), cn, "{what}: frame {f}");
                assert_eq!(rout, cout, "{what}: frame {f}: decoded PCM");
            }
            assert_eq!(glibc_rand::rand(), c::rand(), "{what}: generator state");
        }
    }
}
