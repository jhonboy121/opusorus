//! Differential tests for the tools_compare unit: `opusorus_tools::compare` (ports of
//! `src/opus_compare.c` and, with `--features qext`, `src/qext_compare.c`) vs the unmodified C
//! tools run in-process by the oracle.
//!
//! Every case writes the input files to a temporary directory and runs both command-line front
//! ends with the same argv: exit codes and the complete stderr text must be identical, and every
//! floating-point value the C tool prints (captured at full precision) must be bit-identical to
//! the corresponding field returned by the Rust analysis function.
//!
//! The RFC 8251 and Opus HD vector tests use `testdata/vectors` (see `scripts/fetch_vectors.sh`),
//! searched upwards from this crate (so git worktrees find the main checkout's copy) or at
//! `$OPUSORUS_VECTORS`; they print a note and pass if the vectors are absent.

// Float-only: not compiled in fixed-point builds until this unit is converted
// (docs/FIXED_POINT.md).
#![cfg(not(feature = "fixed-point"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    reason = "test code: failures should panic; notes about skipped vectors go to stderr"
)]

use opusorus_conformance::{Rng, signals};
use opusorus_oracle::tools_compare as c;
use opusorus_tools::compare::{self as rs, SampleFormat};
use std::path::{Path, PathBuf};

/// Per-test scratch directory.
fn tmp_dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("tools_compare")
        .join(name);
    std::fs::create_dir_all(&d).unwrap();
    d
}

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

fn i16_bytes(s: &[i16]) -> Vec<u8> {
    s.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn to_i16(x: &[f32]) -> Vec<i16> {
    signals::to_i16(x)
}

fn bytes_i16(b: &[u8]) -> Vec<i16> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Runs the Rust command-line front end, returning `(exit code, stderr text)`.
fn rust_cli(
    f: fn(&[String], &mut dyn std::io::Write) -> std::io::Result<i32>,
    args: &[&str],
) -> (i32, String) {
    let args: Vec<String> = args.iter().map(|s| (*s).to_owned()).collect();
    let mut out = Vec::new();
    let code = f(&args, &mut out).unwrap();
    (code, String::from_utf8(out).unwrap())
}

fn assert_values(what: &str, rust: &[f64], c: &[f64]) {
    let rb: Vec<u64> = rust.iter().map(|v| v.to_bits()).collect();
    let cb: Vec<u64> = c.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        rb, cb,
        "{what}: printed values differ (rust {rust:?} vs C {c:?})"
    );
}

// ---------------------------------------------------------------------------------------------
// opus_compare
// ---------------------------------------------------------------------------------------------

/// Parses the opus_compare options the Rust library needs from an argv, returning them with
/// the two file names (`None` if the argv is not a valid comparison).
fn opus_opts<'a>(args: &[&'a str]) -> Option<(rs::OpusCompareOptions, &'a str, &'a str)> {
    let mut o = rs::OpusCompareOptions::default();
    let mut i = 1;
    if args.get(i) == Some(&"-s") {
        o.nchannels = 2;
        i += 1;
    }
    if args.get(i) == Some(&"-r") {
        o.rate = rs::c_atoi(args.get(i + 1)?) as u32;
        i += 2;
    }
    Some((o, args.get(i)?, args.get(i + 1)?))
}

/// The Rust library analysis for `args` (`None` if the argv or files do not get that far).
fn opus_lib(args: &[&str]) -> Option<Result<rs::OpusCompareResult, rs::CompareError>> {
    if !(3..=6).contains(&args.len()) {
        return None;
    }
    let (opts, f1, f2) = opus_opts(args)?;
    let x = rs::read_pcm(&std::fs::read(f1).ok()?, 2, SampleFormat::S16Le);
    let y = rs::read_pcm(
        &std::fs::read(f2).ok()?,
        opts.nchannels,
        SampleFormat::S16Le,
    );
    Some(rs::opus_compare(&x, &y, &opts))
}

/// Runs C and Rust `opus_compare` with `args` (in parallel) and checks they agree: stderr text,
/// exit code, and the full-precision printed values against the library result.
fn check_opus(what: &str, args: &[&str]) -> c::ToolRun {
    let (cr, (code, text), lib) = std::thread::scope(|s| {
        let ct = s.spawn(|| c::opus_compare_main(args));
        let rt = s.spawn(|| rust_cli(rs::opus_compare_main, args));
        let lib = opus_lib(args);
        (ct.join().unwrap(), rt.join().unwrap(), lib)
    });
    assert_eq!(text, cr.stderr, "{what}: stderr text ({args:?})");
    assert_eq!(code, cr.exit_code, "{what}: exit code ({args:?})");
    match lib {
        Some(Ok(r)) => {
            let vals = if r.passes {
                vec![f64::from(r.q), r.err]
            } else {
                vec![r.err]
            };
            assert_values(what, &vals, &cr.values);
        }
        _ => assert!(
            cr.values.is_empty(),
            "{what}: C analysed, Rust library did not"
        ),
    }
    cr
}

/// Writes `x` (stereo, file 1) and `y` (file 2) and compares them with `flags`.
fn opus_case(dir: &Path, tag: &str, x: &[i16], y: &[i16], flags: &[&str]) -> c::ToolRun {
    let p1 = dir.join(format!("{tag}_x.sw"));
    let p2 = dir.join(format!("{tag}_y.sw"));
    std::fs::write(&p1, i16_bytes(x)).unwrap();
    std::fs::write(&p2, i16_bytes(y)).unwrap();
    let mut args = vec!["opus_compare"];
    args.extend_from_slice(flags);
    let (s1, s2) = (p1.to_str().unwrap(), p2.to_str().unwrap());
    args.push(s1);
    args.push(s2);
    check_opus(tag, &args)
}

/// Stereo → mono (average), as a decoder asked for mono output would roughly produce.
fn downmix(x: &[i16]) -> Vec<i16> {
    x.as_chunks::<2>()
        .0
        .iter()
        .map(|c| ((i32::from(c[0]) + i32::from(c[1])) / 2) as i16)
        .collect()
}

/// Keeps every `ds`-th frame of an interleaved signal.
fn decimate(x: &[i16], channels: usize, ds: usize) -> Vec<i16> {
    x.chunks_exact(channels)
        .step_by(ds)
        .flat_map(|c| c.iter().copied())
        .collect()
}

/// Adds uniform noise of amplitude `amp` (saturating).
fn perturb(x: &[i16], amp: i32, rng: &mut Rng) -> Vec<i16> {
    x.iter()
        .map(|&v| (i32::from(v) + rng.range_i32(-amp, amp)).clamp(-32768, 32767) as i16)
        .collect()
}

#[test]
fn opus_compare_random_signals() {
    let dir = tmp_dir("opus_random");
    let mut rng = Rng::new(0x0C0_4A7E);
    let rates = [48000u32, 24000, 16000, 12000, 8000];
    let (mut passes, mut fails) = (0, 0);
    for case in 0..48 {
        let rate = rates[rng.range_i32(0, 4) as usize];
        let ds = (48000 / rate) as usize;
        let stereo = rng.range_i32(0, 1) == 1;
        let nch = if stereo { 2 } else { 1 };
        // Frames (a multiple of ds so the counts match).
        let frames = ds * rng.range_i32((480 / ds) as i32, (9000 / ds) as i32) as usize;
        let src = if case % 3 == 0 {
            signals::speech_like(frames, 2, 48000, case)
        } else {
            signals::music_like(frames, 2, 48000, case)
        };
        let gain = [1.0f32, 0.3, 0.01, 1.4][rng.range_i32(0, 3) as usize];
        let x = to_i16(&src.iter().map(|v| v * gain).collect::<Vec<_>>());
        let mut y = if stereo { x.clone() } else { downmix(&x) };
        y = decimate(&y, nch, ds);
        let amp = [0, 1, 30, 300, 3000, 30000][rng.range_i32(0, 5) as usize];
        y = perturb(&y, amp, &mut rng);
        if rng.range_i32(0, 7) == 0 {
            // Independent signal: fails.
            y = to_i16(&signals::noise(y.len() / nch, nch, 0.5, case + 1000));
        }
        let mut flags = Vec::new();
        if stereo {
            flags.push("-s");
        }
        let rs_str = rate.to_string();
        if rate != 48000 || rng.range_i32(0, 1) == 1 {
            flags.push("-r");
            flags.push(&rs_str);
        }
        let run = opus_case(&dir, &format!("c{case}"), &x, &y, &flags);
        if run.exit_code == 0 {
            passes += 1;
        } else {
            fails += 1;
        }
    }
    assert!(passes > 5 && fails > 5, "passes {passes} fails {fails}");
}

#[test]
fn opus_compare_edge_inputs() {
    let dir = tmp_dir("opus_edge");
    let mut rng = Rng::new(77);
    // Silence, full-scale square wave, DC, and alternating extremes.
    let n = 2400;
    let silence = vec![0i16; 2 * n];
    let square: Vec<i16> = (0..2 * n)
        .map(|i| if (i / 2 / 37) % 2 == 0 { 32767 } else { -32768 })
        .collect();
    let dc = vec![-32768i16; 2 * n];
    let alt: Vec<i16> = (0..2 * n)
        .map(|i| if i % 4 < 2 { 32767 } else { -32768 })
        .collect();
    let sigs = [&silence, &square, &dc, &alt];
    for (a, xa) in sigs.iter().enumerate() {
        for (b, yb) in sigs.iter().enumerate() {
            opus_case(&dir, &format!("s{a}{b}"), xa, yb, &["-s"]);
            opus_case(&dir, &format!("m{a}{b}"), xa, &downmix(yb), &[]);
        }
    }
    // Window-size boundaries: 479 frames (insufficient), 480 (one frame), 599/600/601.
    for frames in [0usize, 1, 479, 480, 481, 599, 600, 601] {
        let x = to_i16(&signals::music_like(frames, 2, 48000, frames as u64));
        let y = perturb(&x, 100, &mut rng);
        opus_case(&dir, &format!("len{frames}"), &x, &y, &["-s"]);
        opus_case(&dir, &format!("mlen{frames}"), &x, &downmix(&y), &[]);
    }
    // Downsampled boundary: 8 kHz with exactly one window.
    let x = to_i16(&signals::music_like(480, 2, 48000, 5));
    opus_case(
        &dir,
        "r8k_one",
        &x,
        &decimate(&downmix(&x), 1, 6),
        &["-r", "8000"],
    );
    // Sample count mismatches (and trailing partial frames that are ignored).
    let x = to_i16(&signals::music_like(4000, 2, 48000, 9));
    opus_case(&dir, "mis1", &x, &x[..x.len() - 2], &["-s"]);
    opus_case(&dir, "mis2", &x, &x, &[]);
    opus_case(
        &dir,
        "mis3",
        &x,
        &decimate(&x, 2, 2),
        &["-s", "-r", "16000"],
    );
    let mut xodd = i16_bytes(&x);
    xodd.push(0x55);
    let p1 = dir.join("odd_x.sw");
    let p2 = dir.join("odd_y.sw");
    std::fs::write(&p1, &xodd).unwrap();
    let mut yodd = i16_bytes(&x);
    yodd.extend_from_slice(&[1, 2, 3]);
    std::fs::write(&p2, &yodd).unwrap();
    check_opus(
        "odd",
        &[
            "opus_compare",
            "-s",
            p1.to_str().unwrap(),
            p2.to_str().unwrap(),
        ],
    );
    // Extra arguments after the two files are ignored by C.
    let p = p1.to_str().unwrap();
    check_opus("extra", &["opus_compare", p, p, p]);
    check_opus("extra2", &["opus_compare", "-s", p, p, "junk", "more"]);
}

#[test]
fn opus_compare_command_line_errors() {
    let dir = tmp_dir("opus_cli");
    let x = to_i16(&signals::music_like(1000, 2, 48000, 1));
    let p = dir.join("x.sw");
    std::fs::write(&p, i16_bytes(&x)).unwrap();
    let p = p.to_str().unwrap();
    let missing = dir.join("does_not_exist.sw");
    let missing = missing.to_str().unwrap();
    let cases: Vec<Vec<&str>> = vec![
        vec!["opus_compare"],
        vec!["opus_compare", p],
        vec!["opus_compare", "-s", "-r", "48000", p, p, "x"],
        vec!["opus_compare", "-r", "44100", p, p],
        vec!["opus_compare", "-r", "0", p, p],
        vec!["opus_compare", "-r", "-8000", p, p],
        vec!["opus_compare", "-r", " 16000xyz", p, p],
        vec!["opus_compare", "-r", "+48000", p, p],
        vec!["opus_compare", "-s", "-r", "96000", p, p],
        vec!["opus_compare", missing, p],
        vec!["opus_compare", p, missing],
        vec!["opus_compare", "-s", missing, missing],
        vec!["opus_compare", "-r", "12000", p, missing],
        // `-r` before `-s`: `-s` is then taken as the first file name.
        vec!["opus_compare", "-r", "48000", "-s", p],
    ];
    for (i, args) in cases.iter().enumerate() {
        let run = check_opus(&format!("cli{i}"), args);
        assert_eq!(run.exit_code, 1, "{args:?}");
    }
}

/// Compares segments of RFC 8251 vector `n` (reference vs the alternative `m` reference, mono,
/// perturbed, reduced rates). Returns false if the vector is missing.
fn rfc_segment(dir: &Path, tmp: &Path, n: usize) -> bool {
    let a = dir.join(format!("testvector{n:02}.dec"));
    let b = dir.join(format!("testvector{n:02}m.dec"));
    if !a.is_file() || !b.is_file() {
        eprintln!("NOTE: {} missing; skipping", a.display());
        return false;
    }
    let mut rng = Rng::new(8251 + n as u64);
    let xa = bytes_i16(&std::fs::read(&a).unwrap());
    let xb = bytes_i16(&std::fs::read(&b).unwrap());
    // A 1.2 s segment at a vector-dependent offset (multiple of 48 frames).
    let frames = 57600;
    let total = xa.len() / 2;
    let start = (total - frames) / 13 * n / 48 * 48;
    let seg = |v: &[i16]| v[2 * start..2 * (start + frames)].to_vec();
    let (x, y) = (seg(&xa), seg(&xb));
    let t = format!("v{n:02}");
    // Reference vs the alternative (`m`) reference, stereo and mono.
    opus_case(tmp, &format!("{t}_s"), &x, &y, &["-s", "-r", "48000"]);
    opus_case(tmp, &format!("{t}_m"), &x, &downmix(&y), &["-r", "48000"]);
    // Slightly and strongly perturbed.
    opus_case(
        tmp,
        &format!("{t}_p1"),
        &x,
        &perturb(&y, 8, &mut rng),
        &["-s"],
    );
    opus_case(
        tmp,
        &format!("{t}_p2"),
        &x,
        &perturb(&y, 2000, &mut rng),
        &["-s"],
    );
    // Reduced rates.
    let rate = [8000u32, 12000, 16000, 24000][n % 4];
    let ds = (48000 / rate) as usize;
    let r = rate.to_string();
    opus_case(
        tmp,
        &format!("{t}_r"),
        &x,
        &decimate(&downmix(&y), 1, ds),
        &["-r", &r],
    );
    opus_case(
        tmp,
        &format!("{t}_rs"),
        &x,
        &decimate(&y, 2, ds),
        &["-s", "-r", &r],
    );
    true
}

#[test]
fn opus_compare_rfc8251_segments() {
    let Some(dir) = vectors_dir("rfc8251") else {
        eprintln!(
            "NOTE: testdata/vectors/rfc8251 not found; skipping (run scripts/fetch_vectors.sh)"
        );
        return;
    };
    let tmp = tmp_dir("opus_rfc_seg");
    let ran = std::thread::scope(|s| {
        let handles: Vec<_> = (1..=12)
            .map(|n| {
                let (dir, tmp) = (&dir, &tmp);
                s.spawn(move || rfc_segment(dir, tmp, n))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .filter(|&ok| ok)
            .count()
    });
    eprintln!("rfc8251 segments: {ran} vectors compared");
}

#[test]
fn opus_compare_rfc8251_full_vector() {
    let Some(dir) = vectors_dir("rfc8251") else {
        eprintln!(
            "NOTE: testdata/vectors/rfc8251 not found; skipping (run scripts/fetch_vectors.sh)"
        );
        return;
    };
    // The smallest vector, compared in full exactly as tests/run_vectors.sh does (stereo).
    let a = dir.join("testvector03.dec");
    let b = dir.join("testvector03m.dec");
    if !a.is_file() || !b.is_file() {
        eprintln!("NOTE: testvector03 missing; skipping");
        return;
    }
    let run = check_opus(
        "tv03",
        &[
            "opus_compare",
            "-s",
            "-r",
            "48000",
            a.to_str().unwrap(),
            b.to_str().unwrap(),
        ],
    );
    assert_eq!(run.exit_code, 0, "{}", run.stderr);
}

// ---------------------------------------------------------------------------------------------
// qext_compare
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "qext")]
mod qext {
    use super::*;

    /// Parses the qext_compare options the Rust library needs from an argv like C does,
    /// returning them with the format and the two file names (`None` for an invalid argv).
    fn qext_opts<'a>(
        args: &[&'a str],
    ) -> Option<(rs::QextCompareOptions, SampleFormat, &'a str, &'a str)> {
        let mut o = rs::QextCompareOptions::default();
        let mut fmt = SampleFormat::S16Le;
        let mut i = 1;
        let mut argc = args.len();
        if argc < 3 {
            return None;
        }
        while argc > 3 {
            let step = match args[i] {
                "-s" => {
                    o.nchannels = 2;
                    1
                }
                "-48k" => {
                    o.base_rate = 48000;
                    1
                }
                "-s16" => {
                    fmt = SampleFormat::S16Le;
                    1
                }
                "-s24" => {
                    fmt = SampleFormat::S24Le;
                    1
                }
                "-f32" => {
                    fmt = SampleFormat::F32Le;
                    1
                }
                "-skip" => {
                    o.skip = rs::c_atoi(args[i + 1]);
                    2
                }
                "-thresholds" if argc >= 7 => 4,
                "-r" => {
                    o.rate = rs::c_atoi(args[i + 1]) as u32;
                    2
                }
                _ => return None,
            };
            i += step;
            argc -= step;
        }
        (argc == 3).then(|| (o, fmt, args[i], args[i + 1]))
    }

    /// The Rust library analysis for `args` (`None` if the argv or files do not get that far).
    fn qext_lib(args: &[&str]) -> Option<Result<rs::QextCompareResult, rs::CompareError>> {
        let (opts, fmt, f1, f2) = qext_opts(args)?;
        let x = rs::read_pcm(&std::fs::read(f1).ok()?, 2, fmt);
        let y = rs::read_pcm(&std::fs::read(f2).ok()?, opts.nchannels, fmt);
        Some(rs::qext_compare(&x, &y, &opts))
    }

    /// Runs C and Rust `qext_compare` with `args` (in parallel) and checks they agree.
    pub(super) fn check_qext(what: &str, args: &[&str]) -> c::ToolRun {
        let (cr, (code, text), lib) = std::thread::scope(|s| {
            let ct = s.spawn(|| c::qext_compare_main(args));
            let rt = s.spawn(|| rust_cli(rs::qext_compare_main, args));
            let lib = qext_lib(args);
            (ct.join().unwrap(), rt.join().unwrap(), lib)
        });
        assert_eq!(text, cr.stderr, "{what}: stderr text ({args:?})");
        assert_eq!(code, cr.exit_code, "{what}: exit code ({args:?})");
        match lib {
            Some(Ok(r)) => {
                // The C tool prints err4, err16, rms (then the thresholds on failure).
                assert!(cr.values.len() >= 3, "{what}: C did not analyse");
                assert_values(what, &[r.err4, r.err16, r.rms], &cr.values[..3]);
            }
            _ => assert!(
                cr.values.is_empty(),
                "{what}: C analysed, Rust library did not"
            ),
        }
        cr
    }

    /// Encodes 16-bit-range float samples in `fmt`.
    fn encode(x: &[f32], fmt: SampleFormat) -> Vec<u8> {
        match fmt {
            SampleFormat::S16Le => x
                .iter()
                .flat_map(|&v| (v.round().clamp(-32768.0, 32767.0) as i16).to_le_bytes())
                .collect(),
            SampleFormat::S24Le => x
                .iter()
                .flat_map(|&v| {
                    let s = (v * 256.0).round().clamp(-8388608.0, 8388607.0) as i32;
                    let b = s.to_le_bytes();
                    [b[0], b[1], b[2]]
                })
                .collect(),
            SampleFormat::F32Le => x
                .iter()
                .flat_map(|&v| (v / 32768.0).to_le_bytes())
                .collect(),
        }
    }

    const fn fmt_flag(fmt: SampleFormat) -> &'static str {
        match fmt {
            SampleFormat::S16Le => "-s16",
            SampleFormat::S24Le => "-s24",
            SampleFormat::F32Le => "-f32",
        }
    }

    /// Writes raw file images and compares them with `flags`.
    fn qext_case_bytes(dir: &Path, tag: &str, x: &[u8], y: &[u8], flags: &[&str]) -> c::ToolRun {
        let p1 = dir.join(format!("{tag}_x.raw"));
        let p2 = dir.join(format!("{tag}_y.raw"));
        std::fs::write(&p1, x).unwrap();
        std::fs::write(&p2, y).unwrap();
        let mut args = vec!["qext_compare"];
        args.extend_from_slice(flags);
        let (s1, s2) = (p1.to_str().unwrap(), p2.to_str().unwrap());
        args.push(s1);
        args.push(s2);
        check_qext(tag, &args)
    }

    /// Float frames → mono average.
    fn downmix_f(x: &[f32]) -> Vec<f32> {
        x.as_chunks::<2>()
            .0
            .iter()
            .map(|c| 0.5 * (c[0] + c[1]))
            .collect()
    }

    fn decimate_f(x: &[f32], channels: usize, ds: usize) -> Vec<f32> {
        x.chunks_exact(channels)
            .step_by(ds)
            .flat_map(|c| c.iter().copied())
            .collect()
    }

    #[test]
    fn qext_compare_random_signals() {
        let dir = tmp_dir("qext_random");
        let mut rng = Rng::new(0x9E47);
        let rates = [96000u32, 48000, 24000, 16000, 12000, 8000];
        let fmts = [
            SampleFormat::S16Le,
            SampleFormat::S24Le,
            SampleFormat::F32Le,
        ];
        let (mut passes, mut fails) = (0, 0);
        for case in 0..60u64 {
            let base: u32 = if rng.range_i32(0, 2) == 0 {
                48000
            } else {
                96000
            };
            let rate = loop {
                let r = rates[rng.range_i32(0, 5) as usize];
                if r <= base {
                    break r;
                }
            };
            let ds = (base / rate) as usize;
            let stereo = rng.range_i32(0, 1) == 1;
            let nch = if stereo { 2 } else { 1 };
            let fmt = fmts[rng.range_i32(0, 2) as usize];
            let win = if base == 48000 { 480 } else { 960 };
            // At least two analysis frames (C reads out of bounds with one).
            let frames =
                ds * rng.range_i32(((win + win / 4) / ds) as i32, (12000 / ds) as i32) as usize;
            let gain = [32768.0f32, 8000.0, 30.0, 45000.0][rng.range_i32(0, 3) as usize];
            let src = if case % 2 == 0 {
                signals::music_like(frames, 2, base, case)
            } else {
                signals::speech_like(frames, 2, base, case)
            };
            let x: Vec<f32> = src.iter().map(|v| v * gain).collect();
            let mut y = if stereo { x.clone() } else { downmix_f(&x) };
            y = decimate_f(&y, nch, ds);
            let amp = [0.0f32, 0.5, 5.0, 100.0, 3000.0][rng.range_i32(0, 4) as usize];
            for v in &mut y {
                *v += amp * rng.f32_sym();
            }
            // Optional skip: prepend `skip` frames of junk to y (mono: exact; stereo: the C
            // quirk makes the lengths disagree unless y is long enough).
            let mut skip = 0usize;
            if rng.range_i32(0, 3) == 0 {
                skip = rng.range_i32(1, 300) as usize;
                let mut junk: Vec<f32> = (0..skip * nch).map(|_| 1000.0 * rng.f32_sym()).collect();
                junk.extend_from_slice(&y);
                y = junk;
            }
            let mut flags: Vec<String> = Vec::new();
            if stereo {
                flags.push("-s".into());
            }
            if base == 48000 {
                flags.push("-48k".into());
            }
            flags.push(fmt_flag(fmt).into());
            if rate != base || rng.range_i32(0, 1) == 1 {
                flags.push("-r".into());
                flags.push(rate.to_string());
            }
            if skip != 0 {
                flags.push("-skip".into());
                flags.push((skip * ds).to_string());
            }
            if rng.range_i32(0, 1) == 1 {
                flags.push("-thresholds".into());
                for t in [0.05, 0.1, 0.1] {
                    let t: f64 = t * [0.1, 1.0, 10.0][rng.range_i32(0, 2) as usize];
                    flags.push(format!("{t}"));
                }
            }
            // C accepts the options in any order.
            if rng.range_i32(0, 1) == 1 {
                let k = flags
                    .iter()
                    .take_while(|f| {
                        !f.starts_with("-r") && !f.starts_with("-sk") && !f.starts_with("-t")
                    })
                    .count();
                flags[..k].reverse();
            }
            let flag_refs: Vec<&str> = flags.iter().map(String::as_str).collect();
            let run = qext_case_bytes(
                &dir,
                &format!("c{case}"),
                &encode(&x, fmt),
                &encode(&y, fmt),
                &flag_refs,
            );
            if run.exit_code == 0 {
                passes += 1;
            } else {
                fails += 1;
            }
        }
        assert!(passes > 5 && fails > 5, "passes {passes} fails {fails}");
    }

    #[test]
    fn qext_compare_edge_inputs() {
        let dir = tmp_dir("qext_edge");
        let n = 3000;
        let silence = vec![0f32; 2 * n];
        let square: Vec<f32> = (0..2 * n)
            .map(|i| {
                if (i / 2 / 41) % 2 == 0 {
                    32767.0
                } else {
                    -32768.0
                }
            })
            .collect();
        let tiny: Vec<f32> = (0..2 * n).map(|i| ((i * 7919) % 13) as f32 - 6.0).collect();
        let sigs = [&silence, &square, &tiny];
        for (a, xa) in sigs.iter().enumerate() {
            for (b, yb) in sigs.iter().enumerate() {
                for fmt in [SampleFormat::S16Le, SampleFormat::F32Le] {
                    let f = fmt_flag(fmt);
                    qext_case_bytes(
                        &dir,
                        &format!("s{a}{b}{f}"),
                        &encode(xa, fmt),
                        &encode(yb, fmt),
                        &["-s", f],
                    );
                    qext_case_bytes(
                        &dir,
                        &format!("m{a}{b}{f}"),
                        &encode(xa, fmt),
                        &encode(&downmix_f(yb), fmt),
                        &["-48k", f],
                    );
                }
            }
        }
        // Special float values in f32 input.
        let mut special = signals::music_like(2000, 2, 96000, 3);
        special[100] = 0.0;
        special[101] = -0.0;
        special[102] = f32::MIN_POSITIVE;
        special[103] = 1.0e-40;
        let sp: Vec<f32> = special.iter().map(|v| v * 32768.0).collect();
        qext_case_bytes(
            &dir,
            "special",
            &encode(&sp, SampleFormat::F32Le),
            &encode(&sp, SampleFormat::F32Le),
            &["-s", "-f32"],
        );
        // Length boundaries around one/two windows (both bases) and count mismatches.
        for (base_flag, win) in [("-48k", 480usize), ("-s16", 960)] {
            for frames in [
                0usize,
                1,
                win - 1,
                win,
                win + win / 4 - 1,
                win + win / 4,
                3 * win,
            ] {
                if frames >= win && frames < win + win / 4 {
                    // Exactly one analysis frame: C's backward masking loop wraps (UB).
                    continue;
                }
                let x: Vec<f32> = signals::music_like(frames, 2, 96000, frames as u64)
                    .iter()
                    .map(|v| v * 20000.0)
                    .collect();
                let e = encode(&x, SampleFormat::S16Le);
                qext_case_bytes(
                    &dir,
                    &format!("len{win}_{frames}"),
                    &e,
                    &e,
                    &["-s", base_flag],
                );
            }
        }
        let x: Vec<f32> = signals::music_like(4000, 2, 96000, 11)
            .iter()
            .map(|v| v * 20000.0)
            .collect();
        let e = encode(&x, SampleFormat::S16Le);
        qext_case_bytes(&dir, "mis1", &e, &e[..e.len() - 4], &["-s", "-s16"]);
        qext_case_bytes(&dir, "mis2", &e, &e, &["-s16", "-r", "48000"]);
        // Skip quirks: stereo (length subtracts twice the frames), mono, larger than the file.
        let mut ys = vec![0u8; 4 * 200];
        ys.extend_from_slice(&e);
        qext_case_bytes(&dir, "skip_s", &e, &ys, &["-s", "-skip", "100"]);
        qext_case_bytes(&dir, "skip_s0", &e, &e, &["-s", "-skip", "0"]);
        let m = encode(&downmix_f(&x), SampleFormat::S16Le);
        let mut ym = vec![0u8; 2 * 100];
        ym.extend_from_slice(&m);
        qext_case_bytes(&dir, "skip_m", &e, &ym, &["-skip", "100"]);
        qext_case_bytes(&dir, "skip_m2", &e, &ym, &["-skip", "50"]);
        qext_case_bytes(&dir, "skip_m3", &e, &ym, &["-skip", "150"]);
        // Thresholds exactly at the computed values, and parsing variants.
        let r = qext_case_bytes(&dir, "thr0", &e, &ys, &["-s", "-skip", "100"]);
        assert_eq!(r.exit_code, 0);
        let p = dir.join("thr0_x.raw");
        let p = p.to_str().unwrap();
        for t in [
            ["0", "0", "0"],
            ["1e3", "1E-1", ".5"],
            ["inf", "nan", "-INFINITY"],
            ["  0.1x", "+.2", "-0"],
            ["abc", "1.", "7e"],
        ] {
            check_qext(
                &format!("thr {t:?}"),
                &["qext_compare", "-s", "-thresholds", t[0], t[1], t[2], p, p],
            );
        }
    }

    #[test]
    fn qext_compare_command_line_errors() {
        let dir = tmp_dir("qext_cli");
        let x: Vec<f32> = signals::music_like(3000, 2, 96000, 1)
            .iter()
            .map(|v| v * 20000.0)
            .collect();
        let p = dir.join("x.raw");
        std::fs::write(&p, encode(&x, SampleFormat::S16Le)).unwrap();
        let p = p.to_str().unwrap();
        let missing = dir.join("does_not_exist.raw");
        let missing = missing.to_str().unwrap();
        let cases: Vec<Vec<&str>> = vec![
            vec!["qext_compare"],
            vec!["qext_compare", p],
            vec!["qext_compare", "-s", p],
            vec!["qext_compare", "-x", p, p],
            vec!["qext_compare", "-r", "44100", p, p],
            vec!["qext_compare", "-r", "0", p, p],
            vec!["qext_compare", "-thresholds", "1", "1", p, p],
            vec!["qext_compare", "-skip", p, p],
            vec!["qext_compare", "-s", "-s", "-s", p, p, p],
            vec!["qext_compare", missing, p],
            vec!["qext_compare", p, missing],
            vec!["qext_compare", "-f32", missing, missing],
            vec!["qext_compare", "-r", "96000", "-skip", "3", p, missing],
        ];
        for (i, args) in cases.iter().enumerate() {
            let run = check_qext(&format!("cli{i}"), args);
            assert_eq!(run.exit_code, 1, "{args:?}");
        }
        // Valid: repeated flags, all accepted.
        let run = check_qext(
            "flags",
            &[
                "qext_compare",
                "-s",
                "-s16",
                "-s",
                "-r",
                "96000",
                "-f32",
                "-s16",
                p,
                p,
            ],
        );
        assert_eq!(run.exit_code, 0);
        // `-48k` with a 96 kHz test rate divides by zero in C: Rust-only check.
        let (code, text) = rust_cli(
            rs::qext_compare_main,
            &["qext_compare", "-48k", "-r", "96000", p, p],
        );
        assert_eq!(code, 1);
        assert_eq!(text, "Sampling rate 96000 exceeds the base rate 48000\n");
    }

    /// Opus HD original-vector checks for `testvectorNN`: the upstream
    /// `tests/run_opushd_vectors.sh` comparison plus variants. False if the vector is missing.
    fn opushd_testvector(dir: &Path, tmp: &Path, n: usize) -> bool {
        let a = dir.join(format!("testvector{n:02}_96k.f32"));
        let b = dir.join(format!("testvector{n:02}m_96k.f32"));
        if !a.is_file() || !b.is_file() {
            eprintln!("NOTE: {} missing; skipping", a.display());
            return false;
        }
        let (a, b) = (a.to_str().unwrap(), b.to_str().unwrap());
        let run = check_qext(
            &format!("tv{n:02}"),
            &[
                "qext_compare",
                "-s",
                "-r",
                "96000",
                "-f32",
                "-thresholds",
                "0.05",
                ".1",
                ".1",
                a,
                b,
            ],
        );
        eprint!("tv{n:02}: {}", run.stderr);
        if n % 4 == 1 {
            check_qext(
                &format!("tv{n:02}_self"),
                &["qext_compare", "-s", "-f32", a, a],
            );
            check_qext(&format!("tv{n:02}_mono"), &["qext_compare", "-f32", a, b]);
            // Decimated and mono/24-bit test signals derived from the `m` reference.
            let y = rs::read_pcm(&std::fs::read(b).unwrap(), 2, SampleFormat::F32Le);
            let yd = decimate_f(&y, 2, 2);
            let p = tmp.join(format!("tv{n:02}_48k.f32"));
            std::fs::write(&p, encode(&yd, SampleFormat::F32Le)).unwrap();
            check_qext(
                &format!("tv{n:02}_r48"),
                &[
                    "qext_compare",
                    "-s",
                    "-f32",
                    "-r",
                    "48000",
                    a,
                    p.to_str().unwrap(),
                ],
            );
            let ym = downmix_f(&y);
            let p = tmp.join(format!("tv{n:02}_mono.s24"));
            std::fs::write(&p, encode(&ym, SampleFormat::S24Le)).unwrap();
            let xs = rs::read_pcm(&std::fs::read(a).unwrap(), 2, SampleFormat::F32Le);
            let px = tmp.join(format!("tv{n:02}_ref.s24"));
            std::fs::write(&px, encode(&xs, SampleFormat::S24Le)).unwrap();
            check_qext(
                &format!("tv{n:02}_s24"),
                &[
                    "qext_compare",
                    "-s24",
                    px.to_str().unwrap(),
                    p.to_str().unwrap(),
                ],
            );
        }
        true
    }

    /// Opus HD `qext_vectorNN` checks. False if the vector is missing.
    fn opushd_qext_vector(dir: &Path, tmp: &Path, n: usize) -> bool {
        let a = dir.join(format!("qext_vector{n:02}.f32"));
        let b = dir.join(format!("qext_vector{n:02}dec.f32"));
        let bf = dir.join(format!("qext_vector{n:02}fuzzdec.f32"));
        if !a.is_file() || !b.is_file() || !bf.is_file() {
            eprintln!("NOTE: qext_vector{n:02} missing; skipping");
            return false;
        }
        let (a, b, bf) = (
            a.to_str().unwrap(),
            b.to_str().unwrap(),
            bf.to_str().unwrap(),
        );
        // Reference decode vs a slightly perturbed copy, with the upstream thresholds (the metric
        // is very sensitive in near-silent passages, so this fails).
        let mut rng = Rng::new(96000 + n as u64);
        let mut y = rs::read_pcm(&std::fs::read(b).unwrap(), 2, SampleFormat::F32Le);
        for v in &mut y {
            *v += 0.25 * rng.f32_sym();
        }
        let p = tmp.join(format!("qv{n:02}_perturbed.f32"));
        std::fs::write(&p, encode(&y, SampleFormat::F32Le)).unwrap();
        let run = check_qext(
            &format!("qv{n:02}_p"),
            &[
                "qext_compare",
                "-s",
                "-r",
                "96000",
                "-f32",
                "-thresholds",
                "0.05",
                ".1",
                ".1",
                b,
                p.to_str().unwrap(),
            ],
        );
        eprint!("qv{n:02} perturbed: {}", run.stderr);
        // Reference decode vs the fuzzed stream's decode (fails the thresholds).
        let run = check_qext(
            &format!("qv{n:02}"),
            &[
                "qext_compare",
                "-s",
                "-r",
                "96000",
                "-f32",
                "-thresholds",
                "0.1",
                ".5",
                "1",
                b,
                bf,
            ],
        );
        eprint!("qv{n:02} fuzz: {}", run.stderr);
        // Source vs decoded: lengths differ by the decoder delay; `-skip` (stereo hits the C
        // length quirk and reports a mismatch).
        let skip = ((std::fs::metadata(b).unwrap().len() - std::fs::metadata(a).unwrap().len())
            / 8)
        .to_string();
        let run = check_qext(
            &format!("qv{n:02}_skip"),
            &["qext_compare", "-f32", "-skip", &skip, a, b],
        );
        eprint!("qv{n:02} skip: {}", run.stderr);
        check_qext(
            &format!("qv{n:02}_skip_s"),
            &["qext_compare", "-s", "-f32", "-skip", &skip, a, b],
        );
        true
    }

    /// Opus HD vectors (all compared in full, in parallel).
    #[test]
    fn qext_compare_opushd_vectors() {
        let Some(dir) = vectors_dir("opushd") else {
            eprintln!(
                "NOTE: testdata/vectors/opushd not found; skipping (run scripts/fetch_vectors.sh)"
            );
            return;
        };
        let tmp = tmp_dir("qext_hd");
        let ran = std::thread::scope(|s| {
            let (dir, tmp) = (&dir, &tmp);
            let mut handles: Vec<_> = (1..=12)
                .map(|n| s.spawn(move || opushd_testvector(dir, tmp, n)))
                .collect();
            handles.extend((1..=6).map(|n| s.spawn(move || opushd_qext_vector(dir, tmp, n))));
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .filter(|&ok| ok)
                .count()
        });
        eprintln!("opushd: {ran} vectors compared");
    }
}
