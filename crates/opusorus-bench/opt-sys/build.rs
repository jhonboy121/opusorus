//! Builds the two C variants the micro benchmarks (and the optimized codec benchmarks) run.
//!
//! 1. **C-optimized** (`libopus_opt.a`, prefix `optopus_`): vendored libopus configured with
//!    its default CMake `Release` build (`-O3`, intrinsics on, RTCD where the platform uses it,
//!    hardening on), plus the micro shims in `csrc/bench_shim.c` compiled with the same flags.
//! 2. **C-scalar** (`libopus_scalar.a`, prefix `scalopus_`): the same sources and flags as
//!    `crates/opusorus-oracle/build.rs` (no intrinsics, `-O2 -ffp-contract=off`, hardening),
//!    plus the shims. The codec-level scalar benchmarks use the oracle crate itself; this
//!    identical copy only exists so the shims do not depend on how the linker orders the
//!    oracle's archive relative to this crate's (with LTO it may come first).
//!
//! With the `fixed-point` feature both are fixed-point builds (CMake `OPUS_FIXED_POINT=ON`; the
//! oracle's `FIXED_POINT` sources and defines), with `fixed-res24` also `ENABLE_RES24`. On
//! AArch64 the fixed CMake build needs one workaround (see `build_optimized`).
//!
//! Each library and its shim objects are archived together, then every defined global symbol
//! is renamed `<prefix><name>` with `objcopy --redefine-syms`, so both link next to the
//! unprefixed oracle.
//!
//! Requires `cmake`, `ar`, `nm` and `objcopy` (GNU binutils or LLVM equivalents) on `PATH`;
//! override with the `CMAKE`, `AR`, `NM` and `OBJCOPY` environment variables.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

fn tool(env: &str, default: &str) -> String {
    println!("cargo:rerun-if-env-changed={env}");
    match std::env::var(env) {
        Ok(v) => v,
        Err(std::env::VarError::NotPresent) => default.to_owned(),
        Err(e) => panic!("{env}: {e}"),
    }
}

fn run(cmd: &mut Command) -> Vec<u8> {
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("cannot run {cmd:?}: {e}"));
    assert!(
        out.status.success(),
        "{cmd:?} failed ({}):\n{}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// Source list `var` from an automake fragment (same parser as the oracle's build script).
fn mk_sources(root: &Path, mk: &str, var: &str) -> Vec<PathBuf> {
    let text =
        std::fs::read_to_string(root.join(mk)).unwrap_or_else(|e| panic!("cannot read {mk}: {e}"));
    let mut out = Vec::new();
    let mut in_var = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if !in_var {
            if trimmed.starts_with(var) && trimmed[var.len()..].trim_start().starts_with('=') {
                in_var = true;
            }
            continue;
        }
        let entry = trimmed.trim_end_matches('\\').trim();
        if !entry.is_empty() {
            out.push(root.join(entry));
        }
        if !trimmed.ends_with('\\') {
            break;
        }
    }
    assert!(!out.is_empty(), "no sources for {var} in {mk}");
    out
}

/// Extracts the `-D`/`-I`/`-std` flags CMake used for `celt/mdct.c` from
/// `compile_commands.json`, so the shim sees exactly the library's configuration.
fn cmake_flags(build: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(build.join("compile_commands.json"))
        .expect("CMake writes compile_commands.json (CMAKE_EXPORT_COMPILE_COMMANDS=ON)");
    let line = text
        .lines()
        .find(|l| l.contains("\"command\"") && l.contains("celt/mdct.c"))
        .expect("celt/mdct.c in compile_commands.json");
    let start = line.find(": \"").expect("command value") + 3;
    let cmd = line[start..].trim_end_matches(',').trim_end_matches('"');
    cmd.split_whitespace()
        .filter(|t| t.starts_with("-D") || t.starts_with("-I") || t.starts_with("-std="))
        .map(|t| t.replace("\\\"", "\""))
        .collect()
}

/// Writes `lib` from an optional base archive plus `objs`, renames every defined global
/// symbol to `<prefix><name>` and emits the link directives.
fn archive_prefixed(out: &Path, name: &str, base: Option<&Path>, objs: &[PathBuf], prefix: &str) {
    let lib = out.join(format!("lib{name}.a"));
    if lib.exists() {
        std::fs::remove_file(&lib).expect("remove stale archive");
    }
    if let Some(base) = base {
        std::fs::copy(base, &lib).expect("copy base archive");
    }
    let ar = tool("AR", "ar");
    run(Command::new(&ar).arg("crs").arg(&lib).args(objs));

    let nm = tool("NM", "nm");
    let listing = run(Command::new(&nm)
        .arg("-g")
        .arg("--defined-only")
        .arg("--format=posix")
        .arg(&lib));
    let listing = String::from_utf8(listing).expect("nm output is UTF-8");
    let syms: BTreeSet<&str> = listing
        .lines()
        .filter(|l| !l.ends_with(':'))
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let sym = it.next()?;
            let kind = it.next()?;
            (kind.len() == 1).then_some(sym)
        })
        .collect();
    assert!(
        syms.contains("opus_encoder_create") && syms.contains("bench_mdct_forward"),
        "unexpected nm output for {}",
        lib.display()
    );
    let mut map = String::new();
    for s in &syms {
        writeln!(map, "{s} {prefix}{s}").expect("writing to a String cannot fail");
    }
    let map_path = out.join(format!("{name}.syms"));
    std::fs::write(&map_path, map).expect("write symbol map");
    let objcopy = tool("OBJCOPY", "objcopy");
    run(Command::new(&objcopy)
        .arg(format!("--redefine-syms={}", map_path.display()))
        .arg(&lib));

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static={name}");
}

/// Build options shared by both C variants (cargo features of this crate).
#[derive(Debug, Clone, Copy)]
struct Config {
    qext: bool,
    fixed: bool,
    res24: bool,
}

fn build_optimized(root: &Path, out: &Path, csrc: &Path, cfg: Config) {
    let build = out.join("cmake-build");
    let cmake = tool("CMAKE", "cmake");
    let mut cmd = Command::new(&cmake);
    cmd.arg("-S")
        .arg(root)
        .arg("-B")
        .arg(&build)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
        .arg("-DOPUS_INSTALL_PKG_CONFIG_MODULE=OFF")
        .arg("-DOPUS_INSTALL_CMAKE_CONFIG_MODULE=OFF")
        .arg(format!(
            "-DOPUS_FIXED_POINT={}",
            if cfg.fixed { "ON" } else { "OFF" }
        ));
    // Upstream CMake has no QEXT or RES24 option; configure's --enable-qext only defines
    // ENABLE_QEXT, and ENABLE_RES24 is a plain define too.
    let mut cflags = Vec::new();
    if cfg.qext {
        cflags.push("-DENABLE_QEXT");
    }
    if cfg.res24 {
        cflags.push("-DENABLE_RES24");
    }
    // On AArch64, upstream CMake defines the ARMv7 *assembly* switches OPUS_ARM_MAY_HAVE_NEON /
    // OPUS_ARM_PRESUME_NEON next to the intrinsics ones, but builds no assembly there. The
    // fixed-point `celt_pitch_xcorr` then resolves to `celt_pitch_xcorr_neon`, which only
    // exists in the ARMv7 `celt_pitch_xcorr_arm.s`, and the library does not link. autotools
    // and meson only define the `*_NEON_INTR` switches on AArch64; undo the two assembly
    // defines the same way (the flags come after CMake's -D options on the command line), so
    // the build keeps every NEON intrinsics kernel (`xcorr_kernel_neon_fixed`, the SILK NEON
    // kernels, ...).
    if cfg.fixed && std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64") {
        cflags.extend(["-UOPUS_ARM_MAY_HAVE_NEON", "-UOPUS_ARM_PRESUME_NEON"]);
    }
    cmd.arg(format!("-DCMAKE_C_FLAGS={}", cflags.join(" ")));
    run(&mut cmd);
    let jobs = tool("NUM_JOBS", "4");
    run(Command::new(&cmake)
        .arg("--build")
        .arg(&build)
        .arg("--config")
        .arg("Release")
        .arg("-j")
        .arg(jobs));

    // Shim objects with the library's own defines and include paths.
    let mut b = cc::Build::new();
    b.file(csrc.join("bench_shim.c"))
        .opt_level(3)
        .warnings(false)
        .cargo_metadata(false);
    for f in cmake_flags(&build) {
        b.flag(&f);
    }
    let objs = b.compile_intermediates();
    archive_prefixed(
        out,
        "opus_opt",
        Some(&build.join("libopus.a")),
        &objs,
        "optopus_",
    );
}

/// Scalar copy: same sources, defines and flags as `crates/opusorus-oracle/build.rs` (without
/// its test shims and DNN code).
fn build_scalar(root: &Path, out: &Path, csrc: &Path, cfg: Config) {
    let mut srcs = mk_sources(root, "celt_sources.mk", "CELT_SOURCES");
    srcs.extend(mk_sources(root, "silk_sources.mk", "SILK_SOURCES"));
    srcs.extend(mk_sources(root, "opus_sources.mk", "OPUS_SOURCES"));
    // As the oracle: SILK_SOURCES_FIXED or SILK_SOURCES_FLOAT; OPUS_SOURCES_FLOAT (analysis)
    // in both, since the float API is on.
    srcs.extend(mk_sources(
        root,
        "silk_sources.mk",
        if cfg.fixed {
            "SILK_SOURCES_FIXED"
        } else {
            "SILK_SOURCES_FLOAT"
        },
    ));
    srcs.extend(mk_sources(root, "opus_sources.mk", "OPUS_SOURCES_FLOAT"));
    srcs.push(csrc.join("bench_shim.c"));

    let mut b = cc::Build::new();
    b.files(&srcs)
        .include(root.join("include"))
        .include(root.join("celt"))
        .include(root.join("silk"))
        .include(root.join("src"))
        .include(root.join(if cfg.fixed {
            "silk/fixed"
        } else {
            "silk/float"
        }))
        .define("OPUS_BUILD", None)
        .define("VAR_ARRAYS", None)
        .define("HAVE_LRINT", None)
        .define("HAVE_LRINTF", None)
        .define("ENABLE_HARDENING", None)
        .opt_level(2)
        .flag_if_supported("-ffp-contract=off")
        .flag_if_supported("-fno-fast-math")
        .flag_if_supported("-fvisibility=default")
        .warnings(false)
        .cargo_metadata(false);
    if cfg.qext {
        b.define("ENABLE_QEXT", None);
    }
    if cfg.fixed {
        b.define("FIXED_POINT", "1");
    }
    if cfg.res24 {
        b.define("ENABLE_RES24", None);
    }
    let objs = b.compile_intermediates();
    archive_prefixed(out, "opus_scalar", None, &objs, "scalopus_");
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets this"));
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets this"));
    let root = manifest
        .join("../../../vendor/libopus")
        .canonicalize()
        .expect("vendor/libopus exists");
    let csrc = manifest.join("csrc");
    let feature = |f: &str| std::env::var_os(format!("CARGO_FEATURE_{f}")).is_some();
    let cfg = Config {
        qext: feature("QEXT"),
        fixed: feature("FIXED_POINT"),
        res24: feature("FIXED_RES24"),
    };

    build_optimized(&root, &out, &csrc, cfg);
    build_scalar(&root, &out, &csrc, cfg);

    println!("cargo:rustc-link-lib=m");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=../../../vendor/libopus");
}
