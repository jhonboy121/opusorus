//! Conformance vectors and the `opus_demo` port.
//!
//! * `rfc8251_vectors`: the official decoder conformance procedure of `tests/run_vectors.sh`
//!   (RFC 6716 as updated by RFC 8251): all 12 test vectors decoded by the Rust `opus_demo`
//!   (`-d <rate> <channels> -ignore_extensions`) at 48, 24, 16, 12 and 8 kHz, mono and stereo,
//!   each checked with the Rust `opus_compare` against `testvectorNN.dec` or `testvectorNNm.dec`;
//!   prints the average quality like the script.
//! * `opushd_vectors` (feature `qext`): `tests/run_opushd_vectors.sh`: 96 kHz float decoding of
//!   the RFC vectors (vs `testvectorNN_96k.f32`), the Opus HD vectors and the Opus HD fuzzing
//!   vectors, checked with the Rust `qext_compare` and the script's thresholds.
//! * `opus_demo_matches_c` (unix): the Rust `opus_demo` against the unmodified C
//!   `src/opus_demo.c`, compiled with the system C compiler (`$CC`, default `cc`) and linked
//!   with the oracle's `libopus.a` (the one built for this test binary's configuration). A matrix
//!   of encode (`-e`), decode (`-d`) and encode+decode runs over every application, rate, PCM
//!   format and option (including simulated loss, which uses glibc `rand()`), plus argument and
//!   input errors, must give identical output files, stdout/stderr text and exit status. The only
//!   expected difference is the version line: the oracle is built without `PACKAGE_VERSION`
//!   (`libopus unknown`).
//!
//!   With the DNN features (`deep-plc`, `dred`, `osce`) both tools are DNN builds (the oracle
//!   with compiled-in weights; the Rust tool with the oracle's weights as a blob, or compiled in
//!   when `opusorus-tools/dnn-weights-embedded` is on), and the matrix adds DRED encoding and
//!   decoding after losses (`opus_dred_parse` + `opus_decoder_dred_decode24`), deep PLC and
//!   OSCE cases. The Rust `opus_demo` must be built with the same DNN features, which
//!   `opusorus-conformance` does not forward: run e.g.
//!   `cargo test -p opusorus-conformance --features dred,osce,opusorus-tools/dred --test vectors`
//!   (otherwise the comparison is skipped with a note).
//! * `dnn_blob_matches_oracle` (features `dred` + `osce`): the blob written by
//!   `scripts/gen_dnn_blob.sh` (`$OPUSORUS_DNN_BLOB` or `target/dnn/weights_blob.bin`; skipped
//!   if absent) is byte-identical to the oracle's compiled-in tables serialized in the same
//!   order, so embedding it (`dnn-weights-embedded`) gives exactly the oracle's models. With
//!   `dnn-debug-float` both are the debug-float tables (`gen_dnn_blob.sh --debug-float`,
//!   `target/dnn/weights_blob_debug_float.bin`).
//!
//! With `lossgen` the C `opus_demo` is built with `ENABLE_LOSSGEN` (+ `dnn/lossgen.c`,
//! `lossgen_data.c`, as upstream's Makefile) and the matrix adds `-sim_loss` cases. With
//! `osce-training-data` the comparison is skipped: the library then writes training files into
//! the working directory from every run (`tests/osce_training_data.rs` covers that build).
//!
//! Fixed-point builds (`fixed-point`, `fixed-res24`, with `qext` / `custom-modes`) run the same
//! procedure with the fixed-point decoder, and `opus_demo_matches_c` compares against the C
//! `opus_demo` linked with the fixed-point oracle (its version line ends in `-fixed`; a 16-bit
//! and a 24-bit fixed-point `libopus.a` have the same objects, so the right one is picked by a
//! small C probe of `opus_decode24`'s resolution). In the 16-bit fixed-point build with `qext`
//! the RFC vectors at 96 kHz are decoded with 16-bit resolution, which cannot meet
//! `qext_compare`'s `rms` threshold (0.1 LSB, below the 1/sqrt(12) LSB rounding floor): those
//! are accepted when the error is within that floor, and their output is asserted identical to
//! the 16-bit fixed-point C `opus_demo`'s (`d_tvNN_96s` cases).
//!
//! The vectors are read from `testdata/vectors` (see `scripts/fetch_vectors.sh`), searched
//! upwards from this crate (so git worktrees find the main checkout's copy) or at
//! `$OPUSORUS_VECTORS`; the vector tests print a note and pass if they are absent. The C
//! comparison is skipped (with a note) without a C compiler.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "test code: failures should panic; the procedure log goes to stdout, notes to stderr"
)]

use opusorus_conformance::signals;
use opusorus_tools::compare::opus_compare_main;
use opusorus_tools::demo::opus_demo_main_with_weights;
use std::path::{Path, PathBuf};

/// Per-test scratch directory.
fn tmp_dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("vectors")
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

fn strings(args: &[&str]) -> Vec<String> {
    args.iter().map(|s| (*s).to_owned()).collect()
}

fn path_str(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// Result of one tool run.
#[derive(Debug, PartialEq, Eq)]
struct Run {
    /// Process exit status (low 8 bits, as the OS reports it).
    status: u8,
    stdout: String,
    stderr: String,
}

/// Whether a DNN feature is on (the oracle then has compiled-in weights).
const DNN: bool = cfg!(any(
    feature = "deep-plc",
    feature = "dred",
    feature = "osce"
));

/// The oracle's compiled-in weights as a blob, in `scripts/gen_dnn_blob.sh` order: the models
/// of the enabled DNN features.
#[cfg(any(feature = "deep-plc", feature = "dred", feature = "osce"))]
fn oracle_blob() -> Vec<u8> {
    use opusorus_oracle::dnn_core::{
        MODEL_BBWENET, MODEL_FARGAN, MODEL_LACE, MODEL_NOLACE, MODEL_PITCHDNN, MODEL_PLC,
        MODEL_RDOVAE_DEC, MODEL_RDOVAE_ENC, write_blob,
    };
    let mut models = vec![MODEL_PITCHDNN, MODEL_FARGAN, MODEL_PLC];
    if cfg!(feature = "dred") {
        models.extend([MODEL_RDOVAE_ENC, MODEL_RDOVAE_DEC]);
    }
    if cfg!(feature = "osce") {
        models.extend([MODEL_LACE, MODEL_NOLACE, MODEL_BBWENET]);
    }
    models.iter().flat_map(|&m| write_blob(m)).collect()
}

/// The weights the Rust `opus_demo` gets: the oracle's (so both tools run the same models),
/// unless they are compiled into the tool (`opusorus-tools/dnn-weights-embedded`).
#[cfg(any(feature = "deep-plc", feature = "dred", feature = "osce"))]
fn demo_weights() -> Option<&'static [u8]> {
    static W: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    if opusorus_tools::demo::WEIGHTS_EMBEDDED {
        None
    } else {
        Some(W.get_or_init(oracle_blob))
    }
}

/// Without DNN features there are no weights.
#[cfg(not(any(feature = "deep-plc", feature = "dred", feature = "osce")))]
const fn demo_weights() -> Option<&'static [u8]> {
    None
}

/// Runs the Rust `opus_demo` in-process.
fn rust_demo(args: &[String]) -> Run {
    let mut out = Vec::new();
    let mut err = Vec::new();
    let code = opus_demo_main_with_weights(args, &mut out, &mut err, demo_weights()).unwrap();
    Run {
        status: code as u8,
        stdout: String::from_utf8(out).unwrap(),
        stderr: String::from_utf8(err).unwrap(),
    }
}

/// Runs a Rust compare front end in-process, returning `(exit code, stderr)`.
fn rust_compare(
    f: fn(&[String], &mut dyn std::io::Write) -> std::io::Result<i32>,
    args: &[&str],
) -> (i32, String) {
    let mut err = Vec::new();
    let code = f(&strings(args), &mut err).unwrap();
    (code, String::from_utf8(err).unwrap())
}

/// awk's default number output (`OFMT` = `%.6g`, integral values as integers).
fn awk_num(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        return format!("{}", v as i64);
    }
    let sci = format!("{v:.5e}");
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let trim = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            s.to_owned()
        }
    };
    if !(-4..6).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mant), exp.abs())
    } else {
        trim(&format!("{v:.*}", (5 - exp) as usize))
    }
}

/// The script's average: `grep quality log | awk '{sum+=$4} END {if (NR == 12) sum /= 12; else
/// sum = 0; print sum}'`.
fn log_average(log: &str) -> f64 {
    let vals: Vec<f64> = log
        .lines()
        .filter(|l| l.contains("quality"))
        .map(|l| l.split_whitespace().nth(3).unwrap().parse::<f64>().unwrap())
        .collect();
    if vals.len() == 12 {
        vals.iter().sum::<f64>() / 12.0
    } else {
        0.0
    }
}

#[test]
fn awk_number_format() {
    assert_eq!(awk_num(0.0), "0");
    assert_eq!(awk_num(100.0), "100");
    assert_eq!(awk_num(99.575), "99.575");
    assert_eq!(awk_num(98.7666666666), "98.7667");
    assert_eq!(awk_num(1.0 / 3.0), "0.333333");
    assert_eq!(awk_num(1234567.5), "1.23457e+06");
}

// ---------------------------------------------------------------------------------------------
// tests/run_vectors.sh
// ---------------------------------------------------------------------------------------------

/// Maps `f` over `items` on all available cores; results in item order.
fn parallel_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<R>>> = Mutex::new((0..items.len()).map(|_| None).collect());
    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap()
        .min(items.len().max(1));
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= items.len() {
                        break;
                    }
                    let r = f(&items[i]);
                    results.lock().unwrap()[i] = Some(r);
                }
            });
        }
    });
    results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(Option::unwrap)
        .collect()
}

/// One vector of `tests/run_vectors.sh`: `opus_demo -d <rate> <channels> -ignore_extensions
/// testvectorNN.bit tmp.out`, then `opus_compare [-s] -r <rate>` against `testvectorNN.dec`
/// (log 1) and `testvectorNNm.dec` (log 2).
#[derive(Debug)]
struct VectorStep {
    exists: bool,
    decoded: bool,
    log1: String,
    log2: String,
    ok1: bool,
    ok2: bool,
}

fn vector_step(vectors: &Path, rate: u32, stereo: bool, n: u32) -> VectorStep {
    let channels = if stereo { "2" } else { "1" };
    let dir = tmp_dir(&format!("rfc8251_{rate}_{channels}"));
    let tmp_out = dir.join(format!("tmp{n:02}.out"));
    let rate_s = rate.to_string();
    let bit = vectors.join(format!("testvector{n:02}.bit"));
    let r = rust_demo(&strings(&[
        "opus_demo",
        "-d",
        &rate_s,
        channels,
        "-ignore_extensions",
        path_str(&bit),
        path_str(&tmp_out),
    ]));
    let mut step = VectorStep {
        exists: bit.exists(),
        decoded: r.status == 0,
        log1: r.stdout + &r.stderr,
        log2: String::new(),
        ok1: false,
        ok2: false,
    };
    if !step.decoded {
        return step;
    }
    let compare = |dec: &str, log: &mut String| -> bool {
        let reference = vectors.join(dec);
        let mut args = vec!["opus_compare"];
        if stereo {
            args.push("-s");
        }
        args.extend(["-r", &rate_s, path_str(&reference), path_str(&tmp_out)]);
        let (code, text) = rust_compare(opus_compare_main, &args);
        *log += &text;
        code == 0
    };
    step.ok1 = compare(&format!("testvector{n:02}.dec"), &mut step.log1);
    step.ok2 = compare(&format!("testvector{n:02}m.dec"), &mut step.log2);
    std::fs::remove_file(&tmp_out).unwrap();
    step
}

/// The output of `tests/run_vectors.sh <exec path> <vector path> <rate>` from the steps of one
/// rate (mono vectors 1-12, then stereo), and whether it passed.
fn run_vectors_report(rate: u32, steps: &[&VectorStep]) -> (String, bool) {
    let mut report = format!("---------- rate {rate} ----------\n");
    let mut averages = Vec::new();
    for (label, steps) in [("mono", &steps[..12]), ("stereo", &steps[12..])] {
        let mut log1 = String::new();
        let mut log2 = String::new();
        report += &format!("==============\nTesting {label}\n==============\n\n");
        for (i, step) in steps.iter().enumerate() {
            let n = i + 1;
            if step.exists {
                report += &format!("Testing testvector{n:02}\n");
            } else {
                report += &format!("Bitstream file not found: testvector{n:02}.bit\n");
            }
            log1 += &step.log1;
            log2 += &step.log2;
            if !step.decoded {
                report += "ERROR: decoding failed\n";
                return (report, false);
            }
            report += "successfully decoded\n";
            if step.ok1 || step.ok2 {
                report += "output matches reference\n\n";
            } else {
                report += "ERROR: output does not match reference\n";
                return (report, false);
            }
        }
        averages.push((label, log_average(&log1), log_average(&log2)));
    }
    report += "All tests have passed successfully\n";
    for (label, a1, a2) in averages {
        let avg = if a2 > a1 { a2 } else { a1 };
        report += &format!("Average {label} quality is {} %\n", awk_num(avg));
    }
    (report, true)
}

#[test]
fn rfc8251_vectors() {
    let Some(vectors) = vectors_dir("rfc8251") else {
        eprintln!("No test vectors found (run scripts/fetch_vectors.sh); skipping");
        return;
    };
    println!("Test vectors found in {}", vectors.display());
    println!("Decoding with opusorus opus_demo");
    let rates = [48000u32, 24000, 16000, 12000, 8000];
    let jobs: Vec<(u32, bool, u32)> = rates
        .iter()
        .flat_map(|&r| [false, true].map(|st| (1..=12).map(move |n| (r, st, n))))
        .flatten()
        .collect();
    let steps = parallel_map(&jobs, |&(rate, stereo, n)| {
        vector_step(&vectors, rate, stereo, n)
    });
    let mut all_ok = true;
    for &rate in &rates {
        let rate_steps: Vec<&VectorStep> = jobs
            .iter()
            .zip(&steps)
            .filter(|((r, ..), _)| *r == rate)
            .map(|(_, s)| s)
            .collect();
        let (report, ok) = run_vectors_report(rate, &rate_steps);
        println!("{report}");
        all_ok &= ok;
    }
    assert!(all_ok, "RFC 8251 conformance failed (see the log above)");
}

// ---------------------------------------------------------------------------------------------
// tests/run_opushd_vectors.sh
// ---------------------------------------------------------------------------------------------

/// Decodes `bit` at 96 kHz stereo float with the Rust `opus_demo` (`extra` flags first) and
/// checks it with `qext_compare -s -r 96000 -f32 -thresholds t...`. Returns the log text.
#[cfg(feature = "qext")]
fn opushd_case(
    dir: &Path,
    bit: &Path,
    reference: &Path,
    extra: &[&str],
    thresholds: [&str; 3],
) -> Result<String, String> {
    let name = bit.file_stem().unwrap().to_str().unwrap();
    let out = dir.join(format!("{name}.out"));
    let mut args = vec!["opus_demo", "-d", "96000", "2"];
    args.extend_from_slice(extra);
    args.extend(["-f32", path_str(bit), path_str(&out)]);
    let r = rust_demo(&strings(&args));
    let mut log = format!("Testing {name}\n{}", r.stderr);
    if r.status != 0 {
        return Err(format!("{log}ERROR: decoding failed\n"));
    }
    log += "successfully decoded\n";
    let [t1, t2, t3] = thresholds;
    let (code, text) = rust_compare(
        opusorus_tools::compare::qext_compare_main,
        &[
            "qext_compare",
            "-s",
            "-r",
            "96000",
            "-f32",
            "-thresholds",
            t1,
            t2,
            t3,
            path_str(reference),
            path_str(&out),
        ],
    );
    log += &text;
    std::fs::remove_file(&out).unwrap();
    if code == 0 {
        Ok(log + "output matches reference\n")
    } else {
        Err(log + "ERROR: output does not match reference\n")
    }
}

/// The 16-bit fixed-point build (`fixed-point` without `fixed-res24`): `opus_res` is 16-bit, so
/// the 96 kHz float output is `RES2FLOAT` of 16-bit samples.
#[cfg(feature = "qext")]
const RES16: bool = cfg!(all(feature = "fixed-point", not(feature = "fixed-res24")));

/// Whether a failed `qext_compare` log shows an error within the quantisation floor of a 16-bit
/// output: `qext_compare` measures in 16-bit LSBs, and rounding a signal to 16 bits leaves a
/// uniform error of rms `1/sqrt(12)` = 0.2887 LSB (less where the signal is silent), above the
/// script's `rms` threshold of 0.1.
#[cfg(feature = "qext")]
fn at_16bit_floor(log: &str) -> bool {
    let Some(rms) = log
        .lines()
        .find_map(|l| l.split_once("rms = ").map(|(_, r)| r.trim()))
    else {
        return false;
    };
    let rms: f64 = rms.parse().unwrap();
    rms < 0.30
}

/// `(section title, bitstream, reference, extra opus_demo flags, qext_compare thresholds)`.
#[cfg(feature = "qext")]
type HdCase = (
    &'static str,
    PathBuf,
    PathBuf,
    &'static [&'static str],
    [&'static str; 3],
);

#[cfg(feature = "qext")]
#[test]
fn opushd_vectors() {
    let Some(vectors) = vectors_dir("opushd") else {
        eprintln!("No Opus HD test vectors found (run scripts/fetch_vectors.sh); skipping");
        return;
    };
    let dir = tmp_dir("opushd");
    println!("Test vectors found in {}", vectors.display());
    // (bitstream, reference, extra opus_demo flags, thresholds) in the script's order.
    let mut cases: Vec<HdCase> = Vec::new();
    for n in 1..=12 {
        cases.push((
            "Testing original testvectors",
            vectors.join(format!("testvector{n:02}.bit")),
            vectors.join(format!("testvector{n:02}_96k.f32")),
            &["-ignore_extensions"],
            ["0.05", ".1", ".1"],
        ));
    }
    for n in 1..=6 {
        cases.push((
            "Testing Opus HD testvectors",
            vectors.join(format!("qext_vector{n:02}.bit")),
            vectors.join(format!("qext_vector{n:02}dec.f32")),
            &[],
            ["0.05", ".1", ".1"],
        ));
    }
    for n in 1..=6 {
        cases.push((
            "Testing Opus HD fuzzing testvectors",
            vectors.join(format!("qext_vector{n:02}fuzz.bit")),
            vectors.join(format!("qext_vector{n:02}fuzzdec.f32")),
            &[],
            ["0.1", ".5", "1"],
        ));
    }
    let results = parallel_map(&cases, |(_, bit, reference, extra, th)| {
        opushd_case(&dir, bit, reference, extra, *th)
    });
    let mut section = "";
    let mut failed = Vec::new();
    let mut known = Vec::new();
    let mut res16 = Vec::new();
    for ((title, bit, ..), res) in cases.iter().zip(&results) {
        if *title != section {
            section = title;
            println!("============================\n{title}\n============================\n");
        }
        let name = bit.file_name().unwrap().to_string_lossy().into_owned();
        match res {
            Ok(log) => println!("{log}"),
            Err(log) => {
                println!("{log}");
                // The published qext_vector files do not match the libopus 1.6.1 QEXT
                // bitstream: the unmodified 1.6.1 opus_demo stops with a range coder mismatch
                // on them too (asserted byte-for-byte in `opus_demo_matches_c`).
                if name.starts_with("qext_vector")
                    && log.contains("Range coder state mismatch between encoder and decoder")
                {
                    known.push(name);
                } else if RES16 && name.starts_with("testvector") && at_16bit_floor(log) {
                    res16.push(name);
                } else {
                    failed.push(name);
                }
            }
        }
    }
    assert!(
        failed.is_empty(),
        "Opus HD vector(s) failed (see the log above): {failed:?}"
    );
    if !res16.is_empty() {
        println!(
            "16-bit fixed-point build: {} RFC vectors at 96 kHz are limited by the 16-bit output \
             resolution (rms within the 1/sqrt(12) LSB quantisation floor; the C fixed-point \
             opus_demo gives the identical output, see opus_demo_matches_c): {res16:?}",
            res16.len()
        );
    }
    if known.is_empty() && res16.is_empty() {
        println!("All tests have passed successfully");
    } else if known.is_empty() {
        println!("All other tests have passed successfully");
    } else {
        let rfc = if res16.is_empty() {
            "All RFC vectors at 96 kHz passed"
        } else {
            "The other RFC vectors at 96 kHz passed"
        };
        println!(
            "{rfc}; {} Opus HD vectors stop with the range coder mismatch that libopus 1.6.1's \
             own opus_demo reports on them: {known:?}",
            known.len()
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Rust opus_demo vs C opus_demo
// ---------------------------------------------------------------------------------------------

/// Archive member / symbol markers of the optional libopus configurations.
#[cfg(unix)]
fn lib_config_matches(lib: &[u8]) -> bool {
    let has = |needle: &[u8]| lib.windows(needle.len()).any(|w| w == needle);
    // Fixed-point oracles compile silk/fixed/* (e.g. burg_modified_FIX.c) instead of silk/float.
    has(b"-burg_modified_FIX.o") == cfg!(feature = "fixed-point")
        && has(b"-mini_kfft.o") == cfg!(feature = "qext")
        && has(b"opus_custom_encoder_create") == cfg!(feature = "custom-modes")
        && has(b"-lpcnet_plc.o")
            == cfg!(any(
                feature = "deep-plc",
                feature = "dred",
                feature = "osce"
            ))
        && has(b"-dred_decoder.o") == cfg!(feature = "dred")
        && has(b"-osce.o") == cfg!(feature = "osce")
        // csrc/dnn_debug_float.c (no DISABLE_DEBUG_FLOAT) and the training file names.
        && has(b"opusorus-oracle: dnn-debug-float build") == (DNN && cfg!(feature = "dnn-debug-float"))
        && has(b"features_lpc.f32") == cfg!(feature = "osce-training-data")
}

/// The oracle `libopus.a` files whose optional components match this crate's features
/// (`<profile>/build/opusorus-oracle-*/out/libopus.a`), newest first.
#[cfg(unix)]
fn find_oracle_libs() -> Vec<PathBuf> {
    let exe = std::env::current_exe().unwrap();
    // <target>/<profile>/deps/vectors-<hash>
    let profile = exe.parent().unwrap().parent().unwrap();
    let Ok(dir) = std::fs::read_dir(profile.join("build")) else {
        return Vec::new();
    };
    let mut libs: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    for entry in dir {
        let entry = entry.unwrap();
        if !entry
            .file_name()
            .to_string_lossy()
            .starts_with("opusorus-oracle-")
        {
            continue;
        }
        let lib = entry.path().join("out/libopus.a");
        let Ok(meta) = std::fs::metadata(&lib) else {
            continue;
        };
        if lib_config_matches(&std::fs::read(&lib).unwrap()) {
            libs.push((meta.modified().unwrap(), lib));
        }
    }
    libs.sort_by_key(|l| core::cmp::Reverse(l.0));
    libs.into_iter().map(|(_, p)| p).collect()
}

/// A C program that tells a 16-bit from a 24-bit (`ENABLE_RES24`) fixed-point libopus, which
/// have the same objects and symbols: the 16-bit build's `opus_decode24` output is
/// `RES2INT24` of 16-bit samples (low 8 bits zero). Exit status 24 or 16.
#[cfg(unix)]
const RES_PROBE_C: &str = r#"#include <math.h>
#include "opus.h"
int main(void) {
   int err, f, i, low = 0;
   opus_int32 pcm[960], out[960];
   unsigned char pkt[1500];
   OpusEncoder *enc = opus_encoder_create(48000, 1, OPUS_APPLICATION_AUDIO, &err);
   OpusDecoder *dec = opus_decoder_create(48000, 1, &err);
   for (f = 0; f < 10; f++) {
      int n;
      for (i = 0; i < 960; i++) pcm[i] = (opus_int32)(4000000*sin((f*960 + i)*0.05));
      n = opus_encode24(enc, pcm, 960, pkt, 1500);
      if (n < 0 || opus_decode24(dec, pkt, n, out, 960, 0) != 960) return 1;
      for (i = 0; i < 960; i++) low |= out[i] & 255;
   }
   return low ? 24 : 16;
}
"#;

/// The oracle library for this test binary: the newest matching `libopus.a`, and in fixed-point
/// builds the newest one with the right resolution (checked with [`RES_PROBE_C`]). `Err` if the
/// C compiler cannot be run.
#[cfg(unix)]
fn find_oracle_lib(cc: &str, root: &Path) -> std::io::Result<PathBuf> {
    let libs = find_oracle_libs();
    if !cfg!(feature = "fixed-point") {
        return Ok(libs
            .into_iter()
            .next()
            .expect("oracle libopus.a for this configuration not found"));
    }
    let dir = tmp_dir("c");
    let src = dir.join("res_probe.c");
    std::fs::write(&src, RES_PROBE_C).unwrap();
    let want = if cfg!(feature = "fixed-res24") {
        24
    } else {
        16
    };
    for (i, lib) in libs.iter().enumerate() {
        let exe = dir.join(format!("res_probe{i}"));
        let out = std::process::Command::new(cc)
            .arg("-w")
            .arg(format!("-I{}", root.join("include").display()))
            .arg(&src)
            .arg(lib)
            .args(["-lm", "-o"])
            .arg(&exe)
            .output()?;
        assert!(
            out.status.success(),
            "compiling the resolution probe failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let status = std::process::Command::new(&exe).status().unwrap();
        if status.code() == Some(want) {
            return Ok(lib.clone());
        }
    }
    panic!("no oracle libopus.a with {want}-bit resolution found among {libs:?}");
}

/// Compiles the unmodified `src/opus_demo.c` against the oracle library. `None` (with a note)
/// if no C compiler can be run.
#[cfg(unix)]
fn build_c_opus_demo() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/libopus");
    let cc = match std::env::var_os("CC") {
        Some(cc) => cc.to_string_lossy().into_owned(),
        None => "cc".to_owned(),
    };
    let lib = match find_oracle_lib(&cc, &root) {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!(
                "cannot run the C compiler `{cc}` ({e}); skipping the C opus_demo comparison"
            );
            return None;
        }
    };
    let mut tag = String::new();
    for (on, t) in [
        (cfg!(feature = "fixed-point"), "_fixed"),
        (cfg!(feature = "fixed-res24"), "_res24"),
        (cfg!(feature = "qext"), "_qext"),
        (cfg!(feature = "custom-modes"), "_custom"),
        (DNN, "_deepplc"),
        (cfg!(feature = "dred"), "_dred"),
        (cfg!(feature = "osce"), "_osce"),
        (DNN && cfg!(feature = "dnn-debug-float"), "_dbgfloat"),
        (cfg!(feature = "lossgen"), "_lossgen"),
    ] {
        if on {
            tag.push_str(t);
        }
    }
    let exe = tmp_dir("c").join(format!("opus_demo{tag}"));
    let mut cmd = std::process::Command::new(&cc);
    cmd.args(["-O2", "-ffp-contract=off", "-fno-fast-math", "-w"])
        .args([
            "-DOPUS_BUILD",
            "-DVAR_ARRAYS",
            "-DHAVE_LRINT",
            "-DHAVE_LRINTF",
        ]);
    // As the oracle's build (opus_demo.c itself has no FIXED_POINT code).
    if cfg!(feature = "fixed-point") {
        cmd.arg("-DFIXED_POINT=1");
    }
    if cfg!(feature = "fixed-res24") {
        cmd.arg("-DENABLE_RES24");
    }
    if cfg!(feature = "qext") {
        cmd.arg("-DENABLE_QEXT");
    }
    if cfg!(feature = "custom-modes") {
        cmd.arg("-DCUSTOM_MODES");
    }
    // As upstream's config.h for the DNN builds (opus_demo.c only tests ENABLE_OSCE_BWE).
    if DNN {
        cmd.arg("-DENABLE_DEEP_PLC");
    }
    if cfg!(feature = "dred") {
        cmd.arg("-DENABLE_DRED");
    }
    if cfg!(feature = "osce") {
        cmd.args(["-DENABLE_OSCE", "-DENABLE_OSCE_BWE"]);
    }
    // Upstream links opus_demo with LOSSGEN_SOURCES (generic C DNN kernels, as the oracle).
    if cfg!(feature = "lossgen") {
        cmd.args([
            "-DENABLE_LOSSGEN",
            "-DDISABLE_NEON",
            "-U__SSE2__",
            "-U__AVX__",
        ])
        .arg(root.join("dnn/lossgen.c"))
        .arg(root.join("dnn/lossgen_data.c"));
    }
    for inc in ["include", "celt", "silk", "src", "dnn"] {
        cmd.arg(format!("-I{}", root.join(inc).display()));
    }
    cmd.arg(root.join("src/opus_demo.c"))
        .arg(&lib)
        .arg("-lm")
        .arg("-o")
        .arg(&exe);
    let out = match cmd.output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!(
                "cannot run the C compiler `{cc}` ({e}); skipping the C opus_demo comparison"
            );
            return None;
        }
    };
    assert!(
        out.status.success(),
        "compiling opus_demo.c failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    Some(exe)
}

/// One invocation, run by both tools. `OUT` in `args` is replaced by the side's output path.
#[cfg(unix)]
struct Case {
    name: String,
    args: Vec<String>,
}

#[cfg(unix)]
fn case(name: &str, args: &[&str]) -> Case {
    Case {
        name: name.to_owned(),
        args: strings(args),
    }
}

/// Replaces the version line (the oracle is built without `PACKAGE_VERSION`; fixed-point builds
/// append `-fixed`).
#[cfg(unix)]
fn normalize_version(stderr: &str) -> String {
    let mut lines: Vec<&str> = stderr.split_inclusive('\n').collect();
    if let Some(first) = lines.first_mut()
        && (*first == "libopus unknown\n"
            || *first == "libopus unknown-fixed\n"
            || *first == format!("{}\n", opusorus::celt::celt::opus_get_version_string()))
    {
        *first = "libopus <version>\n";
    }
    lines.concat()
}

/// Runs `c` with both tools and asserts identical results. Returns the (common) output bytes.
#[cfg(unix)]
fn check_case(c_demo: &Path, dir: &Path, c: &Case) -> Option<Vec<u8>> {
    use std::os::unix::process::CommandExt;
    let out_c = dir.join(format!("{}.c.out", c.name));
    let out_rs = dir.join(format!("{}.rs.out", c.name));
    for p in [&out_c, &out_rs] {
        if p.exists() {
            std::fs::remove_file(p).unwrap();
        }
    }
    let with_out = |p: &Path| -> Vec<String> {
        c.args
            .iter()
            .map(|a| {
                if a == "OUT" {
                    path_str(p).to_owned()
                } else {
                    a.clone()
                }
            })
            .collect()
    };
    let c_args = with_out(&out_c);
    let mut rs_args = vec!["opus_demo".to_owned()];
    rs_args.extend(with_out(&out_rs));
    let (c_out, rs_run) = std::thread::scope(|s| {
        let ct = s.spawn(|| {
            std::process::Command::new(c_demo)
                .arg0("opus_demo")
                .args(&c_args)
                .output()
                .unwrap()
        });
        let rs_run = rust_demo(&rs_args);
        (ct.join().unwrap(), rs_run)
    });
    let c_run = Run {
        status: u8::try_from(c_out.status.code().expect("C opus_demo was killed")).unwrap(),
        stdout: String::from_utf8(c_out.stdout).unwrap(),
        stderr: normalize_version(&String::from_utf8(c_out.stderr).unwrap()),
    };
    let rs_run = Run {
        stderr: normalize_version(&rs_run.stderr),
        ..rs_run
    };
    assert_eq!(rs_run, c_run, "{}: output text/status differ", c.name);
    let file_c = std::fs::read(&out_c).ok();
    let file_rs = std::fs::read(&out_rs).ok();
    assert_eq!(
        file_rs.is_some(),
        file_c.is_some(),
        "{}: output file presence differs",
        c.name
    );
    if let (Some(a), Some(b)) = (&file_rs, &file_c) {
        assert!(
            a == b,
            "{}: output files differ ({} vs {} bytes, first difference at byte {:?})",
            c.name,
            a.len(),
            b.len(),
            a.iter().zip(b).position(|(x, y)| x != y)
        );
    }
    file_c
}

/// Runs `cases` in parallel; returns each case's output file.
#[cfg(unix)]
fn check_cases(c_demo: &Path, dir: &Path, cases: &[Case]) -> Vec<Option<Vec<u8>>> {
    parallel_map(cases, |c| check_case(c_demo, dir, c))
}

#[cfg(unix)]
fn write_s16(path: &Path, x: &[f32]) {
    let b: Vec<u8> = signals::to_i16(x)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(path, b).unwrap();
}

#[cfg(unix)]
fn write_s24(path: &Path, x: &[f32]) {
    let b: Vec<u8> = x
        .iter()
        .flat_map(|&v| {
            let s = (f64::from(v) * 8_388_607.0).round() as i32;
            let [b0, b1, b2, _] = s.to_le_bytes();
            [b0, b1, b2]
        })
        .collect();
    std::fs::write(path, b).unwrap();
}

#[cfg(unix)]
fn write_f32(path: &Path, x: &[f32]) {
    let b: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, b).unwrap();
}

/// Asserts that most packets of an `opus_demo -e` bitstream carry DRED (so the DRED cases
/// exercise the redundancy decoding, not just the PLC).
#[cfg(all(unix, feature = "dred"))]
fn assert_dred_present(bit: &Path) {
    let b = std::fs::read(bit).unwrap();
    let recs = records(&b);
    let mut dd = opusorus::dred::DredDecoder::new();
    if let Some(w) = demo_weights() {
        dd.set_dnn_blob(w).unwrap();
    }
    let mut dred = opusorus::dred::Dred::new();
    let with_dred = recs
        .iter()
        .filter(|(_, pkt)| {
            !pkt.is_empty() && dd.parse(&mut dred, pkt, 48000, 48000, false).unwrap().0 > 0
        })
        .count();
    assert!(
        with_dred * 2 > recs.len(),
        "{}: only {with_dred} of {} packets carry DRED",
        bit.display(),
        recs.len()
    );
}

#[cfg(all(unix, not(feature = "dred")))]
fn assert_dred_present(_bit: &Path) {
    unreachable!("DRED cases need the dred feature")
}

/// A bitstream record (`opus_demo -e` format).
#[cfg(unix)]
fn record(len: u32, rng: u32, payload: &[u8]) -> Vec<u8> {
    let mut v = len.to_be_bytes().to_vec();
    v.extend(rng.to_be_bytes());
    v.extend_from_slice(payload);
    v
}

/// Splits a bitstream file into `(final range, packet)` records.
#[cfg(unix)]
fn records(b: &[u8]) -> Vec<(u32, &[u8])> {
    let mut v = Vec::new();
    let mut i = 0;
    while i + 8 <= b.len() {
        let len = u32::from_be_bytes(b[i..i + 4].try_into().unwrap()) as usize;
        let rng = u32::from_be_bytes(b[i + 4..i + 8].try_into().unwrap());
        v.push((rng, &b[i + 8..i + 8 + len]));
        i += 8 + len;
    }
    v
}

#[cfg(unix)]
#[test]
fn opus_demo_matches_c() {
    use opusorus_tools::demo as rd;
    if (rd::DEEP_PLC, rd::DRED, rd::OSCE) != (DNN, cfg!(feature = "dred"), cfg!(feature = "osce")) {
        eprintln!(
            "NOTE: the Rust opus_demo is built with DNN features (deep-plc, dred, osce) = {:?}, \
             the oracle with {:?}; skipping the C opus_demo comparison. Enable the same \
             features on opusorus-tools (e.g. --features dred,opusorus-tools/dred).",
            (rd::DEEP_PLC, rd::DRED, rd::OSCE),
            (DNN, cfg!(feature = "dred"), cfg!(feature = "osce"))
        );
        return;
    }
    if cfg!(feature = "osce-training-data") {
        eprintln!(
            "NOTE: osce-training-data: every run writes training files into the working \
             directory; skipping (see tests/osce_training_data.rs)"
        );
        return;
    }
    let Some(c_demo) = build_c_opus_demo() else {
        return;
    };
    let dir = tmp_dir("demo");
    let p = |name: &str| path_str(&dir.join(name)).to_owned();

    // Inputs: ~1.5 s signals at every rate and format.
    let secs = |fs: u32| (fs as usize * 3) / 2;
    write_s16(
        &dir.join("m48s.sw"),
        &signals::music_like(secs(48000), 2, 48000, 1),
    );
    write_s16(
        &dir.join("s48m.sw"),
        &signals::speech_like(secs(48000), 1, 48000, 2),
    );
    write_s16(
        &dir.join("s24s.sw"),
        &signals::speech_like(secs(24000), 2, 24000, 3),
    );
    write_s16(
        &dir.join("s16m.sw"),
        &signals::speech_like(secs(16000), 1, 16000, 4),
    );
    write_s16(
        &dir.join("s12m.sw"),
        &signals::speech_like(secs(12000), 1, 12000, 5),
    );
    write_s16(
        &dir.join("s8m.sw"),
        &signals::speech_like(secs(8000), 1, 8000, 6),
    );
    write_s16(
        &dir.join("m16s.sw"),
        &signals::music_like(secs(16000), 2, 16000, 7),
    );
    let mut noisy = signals::music_like(secs(48000), 2, 48000, 8);
    for (v, n) in noisy.iter_mut().zip(signals::noise(secs(48000), 2, 0.3, 9)) {
        *v = (*v + n).clamp(-1.0, 1.0);
    }
    write_s24(&dir.join("m48s.s24"), &noisy);
    // Float input with overs (> 1.0) to exercise the conversion.
    let loud: Vec<f32> = signals::music_like(secs(48000), 2, 48000, 10)
        .iter()
        .map(|v| v * 1.6)
        .collect();
    write_f32(&dir.join("m48s.f32"), &loud);
    // A short file ending mid-frame (partial last frame, and a partial sample at the end).
    let mut short = signals::to_i16(&signals::music_like(12_345, 2, 48000, 11))
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>();
    short.push(0x55);
    std::fs::write(dir.join("short.sw"), short).unwrap();
    std::fs::write(
        dir.join("loss.txt"),
        "0 0 1 0 1 1 0 0 0 1 1 1 0 2 0 -1 0 0\n1 0 x 1",
    )
    .unwrap();
    std::fs::write(dir.join("empty.sw"), []).unwrap();

    let m48s = p("m48s.sw");
    let s48m = p("s48m.sw");
    let s24s = p("s24s.sw");
    let s16m = p("s16m.sw");
    let s12m = p("s12m.sw");
    let s8m = p("s8m.sw");
    let m16s = p("m16s.sw");
    let m48s24 = p("m48s.s24");
    let m48sf = p("m48s.f32");
    let short = p("short.sw");
    let loss = p("loss.txt");
    let empty = p("empty.sw");

    // Phase 1: encode-only (bitstreams reused below), encode+decode, and error cases.
    #[rustfmt::skip]
    let mut phase1 = vec![
        // -e: bitstreams
        case("e_audio48s", &["-e", "audio", "48000", "2", "64000", &m48s, "OUT"]),
        case("e_voip16m_fec", &["-e", "voip", "16000", "1", "20000", "-inbandfec", "-loss", "20", &s16m, "OUT"]),
        case("e_voip16m_encloss", &["-e", "voip", "16000", "1", "20000", "-inbandfec", "-loss", "20", "-enc_loss", &s16m, "OUT"]),
        case("e_voip48m_dtx", &["-e", "voip", "48000", "1", "16000", "-dtx", &s48m, "OUT"]),
        case("e_lowdelay_f32", &["-e", "restricted-lowdelay", "48000", "2", "96000", "-framesize", "2.5", "-f32", &m48sf, "OUT"]),
        case("e_hybrid24s_60ms", &["-e", "voip", "24000", "2", "32000", "-framesize", "60", &s24s, "OUT"]),
        case("e_s24_120ms", &["-e", "audio", "48000", "2", "80000", "-framesize", "120", "-24", &m48s24, "OUT"]),
        case("e_lossfile", &["-e", "audio", "48000", "2", "64000", "-enc_loss", "-lossfile", &loss, &m48s, "OUT"]),
        case("e_short", &["-e", "audio", "48000", "2", "64000", &short, "OUT"]),
        case("e_empty", &["-e", "voip", "8000", "1", "8000", &empty, "OUT"]),
        // encode + decode
        case("ed_audio48s", &["audio", "48000", "2", "64000", &m48s, "OUT"]),
        case("ed_voip16m_fec_loss", &["voip", "16000", "1", "16000", "-inbandfec", "-loss", "10", &s16m, "OUT"]),
        case("ed_cbr_10ms", &["audio", "48000", "2", "32000", "-cbr", "-framesize", "10", "-complexity", "5", &m48s, "OUT"]),
        case("ed_lowdelay_cvbr", &["restricted-lowdelay", "48000", "1", "48000", "-framesize", "2.5", "-cvbr", &s48m, "OUT"]),
        case("ed_voip8m_dtx", &["voip", "8000", "1", "8000", "-dtx", &s8m, "OUT"]),
        case("ed_forcemono_swb", &["audio", "24000", "2", "24000", "-forcemono", "-bandwidth", "SWB", &s24s, "OUT"]),
        case("ed_60ms_maxpayload", &["audio", "48000", "2", "96000", "-framesize", "60", "-max_payload", "200", &m48s, "OUT"]),
        case("ed_voip12m_40ms_loss", &["voip", "12000", "1", "12000", "-framesize", "40", "-loss", "25", "-inbandfec", "-dec_complexity", "10", &s12m, "OUT"]),
        case("ed_delayed_decision", &["audio", "48000", "2", "50000", "-delayed-decision", &m48s, "OUT"]),
        case("ed_delayed_decision_5ms", &["voip", "16000", "2", "40000", "-delayed-decision", "-framesize", "5", &m16s, "OUT"]),
        case("ed_random", &["audio", "48000", "1", "20000", "-random_framesize", "-random_fec", "-sweep", "1000", "-sweep_max", "40000", &s48m, "OUT"]),
        case("ed_sweep_down", &["voip", "16000", "1", "12000", "-sweep", "-500", &s16m, "OUT"]),
        case("ed_silk16k_test", &["restricted-silk", "16000", "1", "20000", "-silk16k_test", &s16m, "OUT"]),
        case("ed_silk8k_test", &["voip", "8000", "1", "12000", "-silk8k_test", &s8m, "OUT"]),
        case("ed_silk12k_test", &["voip", "12000", "1", "12000", "-silk12k_test", &s12m, "OUT"]),
        case("ed_silk_bw_switch", &["voip", "48000", "2", "24000", "-silk_bw_switch_test", &m48s, "OUT"]),
        case("ed_hybrid24k_test", &["audio", "24000", "2", "32000", "-hybrid24k_test", &s24s, "OUT"]),
        case("ed_hybrid48k_test", &["audio", "48000", "2", "40000", "-hybrid48k_test", &m48s, "OUT"]),
        case("ed_celt_test", &["audio", "48000", "2", "64000", "-celt_test", &m48s, "OUT"]),
        case("ed_celt_hq_test", &["audio", "48000", "2", "128000", "-celt_hq_test", "-24", &m48s24, "OUT"]),
        case("ed_s24", &["audio", "48000", "2", "128000", "-24", &m48s24, "OUT"]),
        case("ed_f32", &["audio", "48000", "2", "128000", "-f32", &m48sf, "OUT"]),
        case("ed_restricted_celt", &["restricted-celt", "48000", "2", "64000", "-framesize", "5", &m48s, "OUT"]),
        case("ed_lossfile", &["audio", "48000", "2", "64000", "-lossfile", &loss, &m48s, "OUT"]),
        case("ed_short", &["voip", "48000", "2", "32000", "-loss", "5", &short, "OUT"]),
        case("ed_bitrate0", &["audio", "48000", "1", "0", "-complexity", "11", &s48m, "OUT"]),
        case("ed_max_payload3", &["audio", "48000", "2", "64000", "-max_payload", "3", &m48s, "OUT"]),
        case("ed_max_payload0", &["audio", "48000", "2", "64000", "-max_payload", "0", &m48s, "OUT"]),
        case("ed_empty", &["audio", "48000", "1", "64000", &empty, "OUT"]),
        // argument and file errors
        case("x_noargs", &[]),
        case("x_argc3", &["-d", "48000"]),
        case("x_argc6_encode", &["audio", "48000", "1", "64000", "OUT"]),
        case("x_rate", &["-d", "44100", "1", &m48s, "OUT"]),
        case("x_channels", &["-d", "48000", "3", &m48s, "OUT"]),
        case("x_application", &["bogus", "48000", "1", "64000", &m48s, "OUT"]),
        case("x_cbr_decode", &["-d", "48000", "1", "-cbr", &m48s, "OUT"]),
        case("x_ignext_encode", &["-e", "audio", "48000", "1", "64000", "-ignore_extensions", &m48s, "OUT"]),
        case("x_deccomplexity_encode", &["-e", "audio", "48000", "1", "64000", "-dec_complexity", "3", &m48s, "OUT"]),
        case("x_unknown_option", &["-d", "48000", "1", "-foo", &m48s, "OUT"]),
        case("x_missing_input", &["-d", "48000", "1", &p("missing.bit"), "OUT"]),
        case("x_bad_output", &["-d", "48000", "1", &m48s, &p("no/such/dir/out")]),
        case("x_max_payload", &["audio", "48000", "1", "64000", "-max_payload", "20000", &m48s, "OUT"]),
        case("x_bandwidth", &["audio", "48000", "1", "64000", "-bandwidth", "XB", &m48s, "OUT"]),
        case("x_framesize", &["audio", "48000", "1", "64000", "-framesize", "7", &m48s, "OUT"]),
        case("x_lossfile", &["audio", "48000", "1", "64000", "-lossfile", &p("missing.txt"), &m48s, "OUT"]),
        case("x_qext_decode", &["-d", "48000", "1", "-qext", &m48s, "OUT"]),
        case("x_rate96k", &["-d", "96000", "2", &m48s, "OUT"]),
        case("x_dred", &["voip", "16000", "1", "24000", "-dred", "10", "-loss", "20", &s16m, "OUT"]),
    ];
    if cfg!(feature = "qext") {
        let hd = p("m96s.f32");
        write_f32(
            Path::new(&hd),
            &signals::music_like(secs(96000), 2, 96000, 12),
        );
        #[rustfmt::skip]
        phase1.extend([
            case("e_qext96", &["-e", "audio", "96000", "2", "256000", "-qext", "-f32", &hd, "OUT"]),
            case("ed_qext96", &["audio", "96000", "2", "192000", "-qext", "-framesize", "10", "-f32", &hd, "OUT"]),
            case("ed_qext48", &["audio", "48000", "2", "160000", "-qext", &m48s, "OUT"]),
            case("x_qext_rate", &["-d", "44100", "1", &m48s, "OUT"]),
        ]);
    }
    if DNN {
        // Deep PLC (dec_complexity >= 5), DRED (encoder side, and decoder side after losses),
        // OSCE (LACE at complexity 6, NoLACE at 7+).
        #[rustfmt::skip]
        phase1.extend([
            case("ed_deepplc16m", &["voip", "16000", "1", "16000", "-loss", "20", "-dec_complexity", "5", &s16m, "OUT"]),
            case("ed_deepplc48s_fec", &["audio", "48000", "2", "32000", "-inbandfec", "-loss", "15", "-dec_complexity", "10", &m48s, "OUT"]),
            case("ed_lace16m", &["voip", "16000", "1", "12000", "-dec_complexity", "6", &s16m, "OUT"]),
            case("ed_nolace8m_loss", &["voip", "8000", "1", "8000", "-loss", "10", "-dec_complexity", "7", &s8m, "OUT"]),
        ]);
    }
    if cfg!(feature = "dred") {
        #[rustfmt::skip]
        phase1.extend([
            case("e_dred16m", &["-e", "voip", "16000", "1", "24000", "-dred", "60", "-loss", "10", &s16m, "OUT"]),
            case("e_dred48m_60ms", &["-e", "voip", "48000", "1", "32000", "-dred", "30", "-loss", "15", "-framesize", "60", &s48m, "OUT"]),
            case("ed_dred48m", &["voip", "48000", "1", "32000", "-dred", "50", "-loss", "25", "-dec_complexity", "7", &s48m, "OUT"]),
            case("ed_dred24s_40ms", &["voip", "24000", "2", "48000", "-dred", "100", "-loss", "15", "-framesize", "40", &s24s, "OUT"]),
            case("ed_dred12m_fec", &["voip", "12000", "1", "20000", "-dred", "20", "-inbandfec", "-loss", "30", &s12m, "OUT"]),
        ]);
    }
    if cfg!(feature = "osce") {
        #[rustfmt::skip]
        phase1.extend([
            case("ed_osce_bwe", &["voip", "48000", "1", "16000", "-bandwidth", "WB", "-enable_osce_bwe", &s48m, "OUT"]),
            case("ed_osce_bwe_loss", &["voip", "48000", "2", "24000", "-bandwidth", "WB", "-enable_osce_bwe", "-loss", "10", "-dec_complexity", "7", &m48s, "OUT"]),
        ]);
    }
    if cfg!(feature = "lossgen") {
        // -sim_loss: the generative loss model, drawing from the same rand() as -random_*;
        // -lossfile takes precedence, a negative percentage falls back to -loss.
        #[rustfmt::skip]
        phase1.extend([
            case("ed_simloss16m_fec", &["voip", "16000", "1", "16000", "-sim_loss", "10", "-inbandfec", &s16m, "OUT"]),
            case("ed_simloss48s_random", &["audio", "48000", "2", "64000", "-sim_loss", "25", "-random_framesize", "-random_fec", &m48s, "OUT"]),
            case("e_simloss_encloss", &["-e", "voip", "16000", "1", "20000", "-sim_loss", "30", "-enc_loss", &s16m, "OUT"]),
            case("ed_simloss_twice", &["voip", "48000", "1", "24000", "-sim_loss", "50", "-sim_loss", "3.5", &s48m, "OUT"]),
            case("ed_simloss_negative", &["voip", "8000", "1", "8000", "-sim_loss", "-1", "-loss", "20", &s8m, "OUT"]),
            case("ed_simloss_lossfile", &["audio", "48000", "2", "32000", "-sim_loss", "40", "-lossfile", &loss, &m48s, "OUT"]),
            case("ed_simloss_and_loss", &["voip", "12000", "1", "12000", "-loss", "20", "-sim_loss", "15", &s12m, "OUT"]),
            case("x_simloss_usage", &["-d", "48000", "1", "-bogus", &m48s, "OUT"]),
        ]);
        if DNN {
            #[rustfmt::skip]
            phase1.push(case("ed_simloss_deepplc", &["voip", "16000", "1", "24000", "-sim_loss", "20", "-dec_complexity", "5", "-dred", "40", &s16m, "OUT"]));
        }
    }
    let outs1 = check_cases(&c_demo, &dir, &phase1);
    let bitstream = |name: &str| -> PathBuf {
        let i = phase1.iter().position(|c| c.name == name).unwrap();
        assert!(outs1[i].is_some(), "{name} wrote no bitstream");
        dir.join(format!("{name}.c.out"))
    };

    // Crafted bitstreams: truncated, invalid length, corrupted packets, range mismatch.
    let e48 = std::fs::read(bitstream("e_audio48s")).unwrap();
    let recs = records(&e48);
    assert!(recs.len() > 20);
    std::fs::write(dir.join("truncated.bit"), &e48[..e48.len() - 7]).unwrap();
    let mut bad_len = record(recs[0].1.len() as u32, recs[0].0, recs[0].1);
    bad_len.extend(record(20_000, 0, &[]));
    std::fs::write(dir.join("bad_len.bit"), bad_len).unwrap();
    let mut rng = opusorus_conformance::Rng::new(0xDE_40);
    let mut garbage = Vec::new();
    for (i, (r, pkt)) in recs.iter().enumerate() {
        if i % 4 == 1 {
            let mut junk = vec![0u8; 1 + (rng.next_u32() % 300) as usize];
            rng.fill_bytes(&mut junk);
            garbage.extend(record(junk.len() as u32, 0, &junk));
        } else {
            garbage.extend(record(pkt.len() as u32, *r, pkt));
        }
    }
    std::fs::write(dir.join("garbage.bit"), garbage).unwrap();
    let mut mismatch = Vec::new();
    for (i, (r, pkt)) in recs.iter().enumerate() {
        let r = if i == 5 { 0x123 } else { *r };
        mismatch.extend(record(pkt.len() as u32, r, pkt));
    }
    std::fs::write(dir.join("mismatch.bit"), mismatch).unwrap();
    // Mismatch at packet 7, after packets 4 and 5 were lost (loss.txt): C returns the value of
    // the DRED parse attempt made on packet 6 (OPUS_UNIMPLEMENTED) as the exit status.
    let mut mismatch7 = Vec::new();
    for (i, (r, pkt)) in recs.iter().enumerate() {
        let r = if i == 7 { !*r } else { *r };
        mismatch7.extend(record(pkt.len() as u32, r, pkt));
    }
    std::fs::write(dir.join("mismatch7.bit"), mismatch7).unwrap();

    let b48 = path_str(&bitstream("e_audio48s")).to_owned();
    let b16 = path_str(&bitstream("e_voip16m_fec")).to_owned();
    let b16l = path_str(&bitstream("e_voip16m_encloss")).to_owned();
    let b48dtx = path_str(&bitstream("e_voip48m_dtx")).to_owned();
    let bld = path_str(&bitstream("e_lowdelay_f32")).to_owned();
    let b24 = path_str(&bitstream("e_hybrid24s_60ms")).to_owned();
    let b120 = path_str(&bitstream("e_s24_120ms")).to_owned();
    let blf = path_str(&bitstream("e_lossfile")).to_owned();
    let truncated = p("truncated.bit");
    let bad_len = p("bad_len.bit");
    let garbage = p("garbage.bit");
    let mismatch = p("mismatch.bit");
    let mismatch7 = p("mismatch7.bit");

    // Phase 2: decoding.
    #[rustfmt::skip]
    let mut phase2 = vec![
        case("d_48s", &["-d", "48000", "2", &b48, "OUT"]),
        case("d_24m_s24", &["-d", "24000", "1", "-24", &b48, "OUT"]),
        case("d_16s_f32", &["-d", "16000", "2", "-f32", &b48, "OUT"]),
        case("d_8m_ignext", &["-d", "8000", "1", "-ignore_extensions", &b48, "OUT"]),
        case("d_16m_fec_loss", &["-d", "16000", "1", "-inbandfec", "-loss", "30", &b16, "OUT"]),
        case("d_16m_encloss", &["-d", "16000", "1", &b16l, "OUT"]),
        case("d_48m_encloss_fec", &["-d", "48000", "1", "-inbandfec", &b16l, "OUT"]),
        case("d_48m_dtx_lossfile", &["-d", "48000", "1", "-lossfile", &loss, &b48dtx, "OUT"]),
        case("d_48s_f32_deccomp", &["-d", "48000", "2", "-dec_complexity", "10", "-f32", &bld, "OUT"]),
        case("d_12s_60ms_loss", &["-d", "12000", "2", "-loss", "15", &b24, "OUT"]),
        case("d_48s_120ms", &["-d", "48000", "2", "-24", &b120, "OUT"]),
        case("d_lossfile_stream", &["-d", "48000", "2", &blf, "OUT"]),
        case("d_truncated", &["-d", "48000", "2", &truncated, "OUT"]),
        case("d_bad_len", &["-d", "48000", "2", &bad_len, "OUT"]),
        case("d_garbage", &["-d", "48000", "2", &garbage, "OUT"]),
        case("d_garbage_fec", &["-d", "16000", "1", "-inbandfec", "-loss", "20", &garbage, "OUT"]),
        case("d_mismatch", &["-d", "48000", "2", &mismatch, "OUT"]),
        case("d_mismatch_after_loss", &["-d", "48000", "2", "-lossfile", &loss, &mismatch7, "OUT"]),
        case("d_empty", &["-d", "48000", "2", &empty, "OUT"]),
        case("d_pcm_as_bitstream", &["-d", "48000", "2", &m48s, "OUT"]),
    ];
    if cfg!(feature = "lossgen") {
        #[rustfmt::skip]
        phase2.extend([
            case("d_simloss_48s", &["-d", "48000", "2", "-sim_loss", "15", &b48, "OUT"]),
            case("d_simloss_16m_fec", &["-d", "16000", "1", "-inbandfec", "-sim_loss", "33.3", &b16, "OUT"]),
        ]);
    }
    if cfg!(feature = "qext") {
        let b96 = path_str(&bitstream("e_qext96")).to_owned();
        #[rustfmt::skip]
        phase2.extend([
            case("d_qext96", &["-d", "96000", "2", "-f32", &b96, "OUT"]),
            case("d_qext96_ignext", &["-d", "96000", "1", "-ignore_extensions", "-24", &b96, "OUT"]),
            case("d_qext48", &["-d", "48000", "2", &b96, "OUT"]),
        ]);
    }
    if cfg!(feature = "dred") {
        let d16 = path_str(&bitstream("e_dred16m")).to_owned();
        let d48 = path_str(&bitstream("e_dred48m_60ms")).to_owned();
        assert_dred_present(&bitstream("e_dred16m"));
        assert_dred_present(&bitstream("e_dred48m_60ms"));
        #[rustfmt::skip]
        phase2.extend([
            case("d_dred16m_loss", &["-d", "16000", "1", "-loss", "30", &d16, "OUT"]),
            case("d_dred16m_48k_lossfile", &["-d", "48000", "1", "-lossfile", &loss, &d16, "OUT"]),
            case("d_dred16m_8k_deep", &["-d", "8000", "1", "-loss", "40", "-dec_complexity", "10", "-f32", &d16, "OUT"]),
            case("d_dred16m_fec", &["-d", "16000", "2", "-inbandfec", "-loss", "25", "-24", &d16, "OUT"]),
            case("d_dred48s_loss", &["-d", "48000", "2", "-loss", "20", &d48, "OUT"]),
            case("d_dred48s_24k_deep", &["-d", "24000", "1", "-loss", "35", "-dec_complexity", "6", &d48, "OUT"]),
        ]);
    }
    // The conformance vectors themselves, through both front ends.
    if let Some(v) = vectors_dir("rfc8251") {
        for n in 1..=12 {
            let bit = path_str(&v.join(format!("testvector{n:02}.bit"))).to_owned();
            phase2.push(case(
                &format!("d_tv{n:02}_48s"),
                &["-d", "48000", "2", "-ignore_extensions", &bit, "OUT"],
            ));
            phase2.push(case(
                &format!("d_tv{n:02}_8m_loss"),
                &["-d", "8000", "1", "-inbandfec", "-loss", "7", &bit, "OUT"],
            ));
        }
    }
    if cfg!(feature = "qext")
        && let Some(v) = vectors_dir("opushd")
    {
        // The RFC vectors at 96 kHz as `run_opushd_vectors.sh` decodes them.
        for n in 1..=12 {
            let bit = path_str(&v.join(format!("testvector{n:02}.bit"))).to_owned();
            phase2.push(case(
                &format!("d_tv{n:02}_96s"),
                &[
                    "-d",
                    "96000",
                    "2",
                    "-ignore_extensions",
                    "-f32",
                    &bit,
                    "OUT",
                ],
            ));
        }
        for n in 1..=6 {
            let bit = path_str(&v.join(format!("qext_vector{n:02}.bit"))).to_owned();
            phase2.push(case(
                &format!("d_hd{n:02}"),
                &["-d", "96000", "2", "-f32", &bit, "OUT"],
            ));
            let fuzz = path_str(&v.join(format!("qext_vector{n:02}fuzz.bit"))).to_owned();
            phase2.push(case(
                &format!("d_hd{n:02}_fuzz"),
                &["-d", "96000", "2", "-f32", &fuzz, "OUT"],
            ));
        }
    }
    let outs2 = check_cases(&c_demo, &dir, &phase2);

    // Sanity: the matrix exercises what it claims to.
    let text = |name: &str, phase: &[Case], outs: &[Option<Vec<u8>>]| {
        let i = phase.iter().position(|c| c.name == name).unwrap();
        outs[i].as_ref().map_or(0, Vec::len)
    };
    assert!(text("ed_audio48s", &phase1, &outs1) > 100_000);
    assert!(text("d_48s", &phase2, &outs2) > 100_000);
    println!(
        "opus_demo: {} C/Rust invocations identical (outputs, text, exit status)",
        phase1.len() + phase2.len()
    );
    for c in phase1.iter().chain(&phase2) {
        for side in ["c", "rs"] {
            let f = dir.join(format!("{}.{side}.out", c.name));
            if f.exists() && !c.name.starts_with("e_") {
                std::fs::remove_file(f).unwrap();
            }
        }
    }
}

/// `scripts/gen_dnn_blob.sh` output == the oracle's compiled-in tables serialized in the same
/// order (pitchdnn, fargan, plcmodel, rdovaeenc, rdovaedec, lace, nolace, bbwenet).
#[cfg(all(feature = "dred", feature = "osce"))]
#[test]
fn dnn_blob_matches_oracle() {
    let default = if cfg!(feature = "dnn-debug-float") {
        "../../target/dnn/weights_blob_debug_float.bin"
    } else {
        "../../target/dnn/weights_blob.bin"
    };
    let path = match std::env::var_os("OPUSORUS_DNN_BLOB") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join(default),
    };
    let Ok(blob) = std::fs::read(&path) else {
        eprintln!(
            "NOTE: {} not found (run scripts/gen_dnn_blob.sh); skipping",
            path.display()
        );
        return;
    };
    let oracle = oracle_blob();
    assert_eq!(blob.len(), oracle.len(), "blob size");
    assert!(
        blob == oracle,
        "{} differs from the oracle's tables",
        path.display()
    );
    println!(
        "{}: {} bytes, identical to the oracle's compiled-in weights",
        path.display(),
        blob.len()
    );
}
