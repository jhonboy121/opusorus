//! Runs libopus' own C test programs against `libopusorus`.
//!
//! The test builds the C ABI library with a nested `cargo build` (separate target directory
//! `<target dir>/capi-c-suite/<config>`, `test` profile: optimized, with overflow checks and debug
//! assertions, and unwinding so a Rust panic becomes `OPUS_INTERNAL_ERROR`), compiles the
//! upstream programs from `vendor/libopus/tests` with the system C compiler (`$CC`, default
//! `cc`) against the shipped headers, links them to `libopusorus.so` and asserts they exit
//! with status 0. The configuration follows this package's features (`qext`, `custom-modes`
//! add `-DENABLE_QEXT` / `-DCUSTOM_MODES` and the matching library features); the library
//! is always built with `internal-api` for `test_opus_extensions.c`.
//!
//! `rfc8251_vectors` builds `opus_demo` and `opus_compare` and runs the equivalent of
//! `tests/run_vectors.sh` on `testdata/vectors/rfc8251` (skipped if absent; fetch it with
//! `scripts/fetch_vectors.sh`).
//!
//! The seed of the randomized C tests is fixed (`SEED=42`) unless `OPUSORUS_C_SUITE_SEED` is
//! set.

// Float-only: not compiled in fixed-point builds until this unit is converted
// (docs/FIXED_POINT.md).
#![cfg(not(feature = "fixed-point"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    reason = "test harness: failures are panics, progress goes to stderr"
)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

/// Repository root.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn vendor() -> PathBuf {
    root().join("vendor/libopus")
}

/// Library features for the nested build, and a short name for the configuration.
fn config() -> (Vec<&'static str>, String) {
    let mut features = vec!["internal-api"];
    let mut name = String::from("base");
    if cfg!(feature = "qext") {
        features.push("qext");
        name.push_str("-qext");
    }
    if cfg!(feature = "custom-modes") {
        features.push("custom-modes");
        name.push_str("-custom");
    }
    (features, name)
}

/// Working directory of this configuration (`<target dir>/capi-c-suite/<config>`).
fn work_dir() -> PathBuf {
    let target = match std::env::var_os("CARGO_TARGET_DIR") {
        Some(t) => PathBuf::from(t),
        None => root().join("target"),
    };
    target.join("capi-c-suite").join(config().1)
}

/// Builds `libopusorus` once per test process and returns the directory holding it.
fn lib_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let (features, _) = config();
        let target = work_dir().join("cargo");
        let status = Command::new(env!("CARGO"))
            .current_dir(root())
            .args(["build", "-p", "opusorus-capi", "--profile", "test"])
            .arg("--features")
            .arg(features.join(","))
            .env("CARGO_TARGET_DIR", &target)
            // Keep the nested build small: no incremental state, line tables only.
            .env("CARGO_INCREMENTAL", "0")
            .env("CARGO_PROFILE_TEST_DEBUG", "line-tables-only")
            .status()
            .expect("run cargo");
        assert!(status.success(), "building libopusorus failed");
        let dir = target.join("debug");
        assert!(
            dir.join(shared_lib_name()).exists(),
            "{} not found in {}",
            shared_lib_name(),
            dir.display()
        );
        dir
    })
}

const fn shared_lib_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "libopusorus.dylib"
    } else {
        "libopusorus.so"
    }
}

/// Preprocessor flags matching the library configuration.
fn config_defines() -> Vec<&'static str> {
    let mut d = vec![
        "-DOPUS_BUILD",
        "-DHAVE_LRINT",
        "-DHAVE_LRINTF",
        "-DVAR_ARRAYS",
    ];
    if cfg!(feature = "qext") {
        d.push("-DENABLE_QEXT");
    }
    if cfg!(feature = "custom-modes") {
        d.push("-DCUSTOM_MODES");
    }
    d
}

/// How to link a C program to the library.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Link {
    Shared,
    Static,
}

/// Compiles `sources` (paths relative to `vendor/libopus`, plus absolute paths) into
/// `name` and returns the executable path.
fn compile(name: &str, sources: &[PathBuf], link: Link) -> PathBuf {
    let lib = lib_dir();
    let v = vendor();
    let bin_dir = work_dir().join("bin");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let exe = bin_dir.join(name);
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let mut cmd = Command::new(cc);
    cmd.args(["-O2", "-g", "-std=gnu99", "-w"])
        .args(config_defines())
        .arg(format!(
            "-I{}",
            root().join("crates/opusorus-capi/include").display()
        ))
        .arg(format!("-I{}", v.join("celt").display()))
        .arg(format!("-I{}", v.join("silk").display()))
        .arg(format!("-I{}", v.join("src").display()))
        .args(sources)
        .arg("-o")
        .arg(&exe);
    match link {
        Link::Shared => {
            cmd.arg(format!("-L{}", lib.display()))
                .arg("-lopusorus")
                .arg(format!("-Wl,-rpath,{}", lib.display()));
        }
        Link::Static => {
            cmd.arg(lib.join("libopusorus.a"))
                .args(["-lpthread", "-ldl", "-lrt", "-lutil"]);
        }
    }
    cmd.arg("-lm");
    let out = cmd.output().expect("run the C compiler");
    assert!(
        out.status.success(),
        "compiling {name} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    exe
}

fn seed() -> String {
    std::env::var("OPUSORUS_C_SUITE_SEED").unwrap_or_else(|_| "42".into())
}

/// Runs `exe` (in its own scratch directory) and asserts exit status 0.
fn run(exe: &Path, args: &[&str]) -> Output {
    let dir = exe.with_extension("run");
    std::fs::create_dir_all(&dir).unwrap();
    let out = Command::new(exe)
        .args(args)
        .current_dir(&dir)
        .env("SEED", seed())
        // `cargo test` points the loader at the outer target directory, which may hold a
        // `libopusorus` built with other features; the rpath would come second.
        .env("LD_LIBRARY_PATH", lib_dir())
        .env("DYLD_LIBRARY_PATH", lib_dir())
        .output()
        .expect("run the test program");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "{} failed ({}):\n--- stdout (tail)\n{}\n--- stderr\n{}",
        exe.display(),
        out.status,
        tail(&stdout, 40),
        stderr
    );
    out
}

/// The last `n` non-empty lines of `s`.
fn tail(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// Compiles and runs one upstream test program (`vendor/libopus/tests/<name>.c`).
fn upstream_test(name: &str, extra: &[&str]) {
    let v = vendor();
    let mut sources = vec![v.join(format!("tests/{name}.c"))];
    sources.extend(extra.iter().map(|e| v.join(e)));
    let exe = compile(name, &sources, Link::Shared);
    let out = run(&exe, &[]);
    eprintln!("{name}: {}", tail(&String::from_utf8_lossy(&out.stdout), 1));
}

#[test]
fn test_opus_api() {
    upstream_test("test_opus_api", &[]);
}

#[test]
fn test_opus_decode() {
    upstream_test("test_opus_decode", &[]);
}

#[test]
fn test_opus_encode() {
    // Includes the encoder regression tests (`regression_test()`).
    upstream_test("test_opus_encode", &["tests/opus_encode_regressions.c"]);
}

#[test]
fn test_opus_padding() {
    upstream_test("test_opus_padding", &[]);
}

#[test]
fn test_opus_projection() {
    // The test checks the mapping matrix code itself through the internal
    // `mapping_matrix_*` functions (not part of the libopus ABI), so upstream's C
    // implementation of those is compiled into the test; the projection encoder/decoder calls
    // go to libopusorus.
    upstream_test("test_opus_projection", &["src/mapping_matrix.c"]);
}

#[test]
fn test_opus_extensions() {
    // Uses internal functions (`opus_packet_extensions_*`, `opus_packet_parse_impl`,
    // `opus_repacketizer_out_range_impl`), exported by the `internal-api` feature.
    upstream_test("test_opus_extensions", &[]);
}

#[cfg(feature = "custom-modes")]
#[test]
fn test_opus_custom() {
    upstream_test("test_opus_custom", &[]);
}

/// Extra checks of this crate's design (byte copies of states, stream handles, NULL
/// handling, custom modes at non-48 kHz rates) in `tests/csrc/capi_extra.c`.
#[test]
fn capi_extra() {
    let src = root().join("crates/opusorus-capi/tests/csrc/capi_extra.c");
    let exe = compile("capi_extra", &[src], Link::Shared);
    run(&exe, &[]);
}

/// The static library works too (C code calling the variadic CTLs through the trampolines).
#[cfg(target_os = "linux")]
#[test]
fn static_link() {
    let v = vendor();
    let exe = compile(
        "test_opus_padding_static",
        &[v.join("tests/test_opus_padding.c")],
        Link::Static,
    );
    run(&exe, &[]);
    let src = root().join("crates/opusorus-capi/tests/csrc/capi_extra.c");
    let exe = compile("capi_extra_static", &[src], Link::Static);
    run(&exe, &[]);
}

/// `tests/run_vectors.sh`: all 12 RFC 8251 vectors, mono and stereo, at every output rate;
/// each output must match `testvectorNN.dec` or `testvectorNNm.dec` per `opus_compare`.
#[test]
fn rfc8251_vectors() {
    let vectors = root().join("testdata/vectors/rfc8251");
    if !vectors.join("testvector01.bit").exists() {
        eprintln!(
            "rfc8251_vectors: SKIPPED ({} not found; run scripts/fetch_vectors.sh)",
            vectors.display()
        );
        return;
    }
    let v = vendor();
    let demo = compile("opus_demo", &[v.join("src/opus_demo.c")], Link::Shared);
    let compare = compile(
        "opus_compare",
        &[v.join("src/opus_compare.c")],
        Link::Shared,
    );
    let rates = [8000, 12000, 16000, 24000, 48000];
    let results: Vec<(i32, u8, Vec<String>)> = std::thread::scope(|s| {
        let handles: Vec<_> = rates
            .iter()
            .flat_map(|&rate| [1u8, 2].map(|ch| (rate, ch)))
            .map(|(rate, ch)| {
                let (demo, compare, vectors) = (&demo, &compare, &vectors);
                s.spawn(move || (rate, ch, run_vector_set(demo, compare, vectors, rate, ch)))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut failures = Vec::new();
    for (rate, ch, f) in &results {
        eprintln!(
            "rfc8251 {rate} Hz {}: {}/12 vectors match",
            if *ch == 1 { "mono" } else { "stereo" },
            12 - f.len()
        );
        failures.extend(f.iter().cloned());
    }
    assert!(
        failures.is_empty(),
        "vector failures:\n{}",
        failures.join("\n")
    );
}

/// Decodes the 12 vectors at `rate` with `channels` and compares them; returns the failures.
fn run_vector_set(
    demo: &Path,
    compare: &Path,
    vectors: &Path,
    rate: i32,
    channels: u8,
) -> Vec<String> {
    let dir = work_dir().join(format!("vectors-{rate}-{channels}"));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("tmp.out");
    let mut failures = Vec::new();
    for n in 1..=12 {
        let bit = vectors.join(format!("testvector{n:02}.bit"));
        let dec = Command::new(demo)
            .env("LD_LIBRARY_PATH", lib_dir())
            .env("DYLD_LIBRARY_PATH", lib_dir())
            .args([
                "-d",
                &rate.to_string(),
                &channels.to_string(),
                "-ignore_extensions",
            ])
            .arg(&bit)
            .arg(&out)
            .output()
            .unwrap();
        if !dec.status.success() {
            failures.push(format!(
                "testvector{n:02} {rate} Hz {channels} ch: decoding failed: {}",
                String::from_utf8_lossy(&dec.stderr)
            ));
            continue;
        }
        let matches = ["", "m"].iter().any(|suffix| {
            let reference = vectors.join(format!("testvector{n:02}{suffix}.dec"));
            let mut cmd = Command::new(compare);
            if channels == 2 {
                cmd.arg("-s");
            }
            cmd.args(["-r", &rate.to_string()])
                .arg(&reference)
                .arg(&out)
                .output()
                .unwrap()
                .status
                .success()
        });
        if !matches {
            failures.push(format!(
                "testvector{n:02} {rate} Hz {channels} ch: output does not match the reference"
            ));
        }
    }
    failures
}
