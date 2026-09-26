//! Builds vendored libopus v1.6.1 as a static library for use as a test oracle.
//!
//! Configuration mirrors a default upstream build (float, hardening on) except:
//! * intrinsics/RTCD are disabled so the scalar C reference paths are used, and
//! * `-ffp-contract=off` so the C compiler does not fuse multiply-adds.
//!
//! Both are required for the Rust port to be bit-exact with the oracle.

use std::path::{Path, PathBuf};

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

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets this"));
    let root = manifest.join("../../vendor/libopus");
    let fixed = false; // fixed-point oracle arrives with the fixed-point phase (PLAN D-011)
    let qext = std::env::var_os("CARGO_FEATURE_QEXT").is_some();
    let custom = std::env::var_os("CARGO_FEATURE_CUSTOM_MODES").is_some();
    // DNN features mirror upstream configure: --enable-dred and --enable-osce both imply the
    // deep PLC sources and ENABLE_DEEP_PLC; --enable-osce also defines ENABLE_OSCE_BWE.
    let dred = std::env::var_os("CARGO_FEATURE_DRED").is_some();
    let osce = std::env::var_os("CARGO_FEATURE_OSCE").is_some();
    let deep_plc = std::env::var_os("CARGO_FEATURE_DEEP_PLC").is_some() || dred || osce;

    let mut srcs = mk_sources(&root, "celt_sources.mk", "CELT_SOURCES");
    srcs.extend(mk_sources(&root, "silk_sources.mk", "SILK_SOURCES"));
    srcs.extend(mk_sources(&root, "opus_sources.mk", "OPUS_SOURCES"));
    if fixed {
        srcs.extend(mk_sources(&root, "silk_sources.mk", "SILK_SOURCES_FIXED"));
    } else {
        srcs.extend(mk_sources(&root, "silk_sources.mk", "SILK_SOURCES_FLOAT"));
        srcs.extend(mk_sources(&root, "opus_sources.mk", "OPUS_SOURCES_FLOAT"));
    }
    if qext {
        srcs.push(root.join("celt/mini_kfft.c"));
    }
    if deep_plc {
        let mut dnn = mk_sources(&root, "lpcnet_sources.mk", "DEEP_PLC_SOURCES");
        if dred {
            dnn.extend(mk_sources(&root, "lpcnet_sources.mk", "DRED_SOURCES"));
        }
        if osce {
            dnn.extend(mk_sources(&root, "lpcnet_sources.mk", "OSCE_SOURCES"));
        }
        // The model data (*_data.c/h, dred_rdovae_constants.h) is not vendored in git.
        let mut needed: Vec<PathBuf> = dnn
            .iter()
            .filter(|p| p.to_string_lossy().ends_with("_data.c"))
            .cloned()
            .collect();
        needed.push(root.join("dnn/dred_rdovae_constants.h"));
        let missing: Vec<String> = needed
            .iter()
            .filter(|p| !p.exists())
            .map(|p| p.display().to_string())
            .collect();
        assert!(
            missing.is_empty(),
            "DNN model data missing (a deep-plc/dred/osce feature is enabled): {}\n\
             Run scripts/fetch_dnn_models.sh from the repository root to download and extract it.",
            missing.join(", ")
        );
        srcs.extend(dnn);
    }

    let mut b = cc::Build::new();
    b.files(&srcs)
        .include(root.join("include"))
        .include(root.join("celt"))
        .include(root.join("silk"))
        .include(root.join("src"))
        .include(root.join(if fixed { "silk/fixed" } else { "silk/float" }))
        .define("OPUS_BUILD", None)
        .define("VAR_ARRAYS", None)
        .define("HAVE_LRINT", None)
        .define("HAVE_LRINTF", None)
        .define("ENABLE_HARDENING", None)
        .opt_level(2)
        .flag_if_supported("-ffp-contract=off")
        .flag_if_supported("-fno-fast-math")
        .flag_if_supported("-fvisibility=default")
        .warnings(false);
    if fixed {
        b.define("FIXED_POINT", "1")
            .define("DISABLE_FLOAT_API", None);
    }
    if qext {
        b.define("ENABLE_QEXT", None);
    }
    if custom {
        b.define("CUSTOM_MODES", None);
    }
    if deep_plc {
        // Upstream also adds the source root (dnn/dred_*.c include "celt/entenc.h").
        b.include(&root)
            .include(root.join("dnn"))
            .define("ENABLE_DEEP_PLC", None);
        // Default upstream build: int8 weights for the quantized layers (no float copies).
        b.define("DISABLE_DEBUG_FLOAT", None);
        // Force the generic C path of dnn/vec.h (the port targets it): no NEON, no SSE/AVX
        // emulation. `__SSE2__` is implied on x86_64 and only tested by dnn/vec*.h.
        b.define("DISABLE_NEON", None)
            .flag_if_supported("-U__AVX__")
            .flag_if_supported("-U__SSE2__");
        if dred {
            b.define("ENABLE_DRED", None);
        }
        if osce {
            b.define("ENABLE_OSCE", None)
                .define("ENABLE_OSCE_BWE", None);
        }
    }
    // Per-unit C shims exposing internal functions with flat, FFI-friendly signatures.
    let csrc = manifest.join("csrc");
    let mut shims: Vec<PathBuf> = std::fs::read_dir(&csrc)
        .expect("csrc dir exists")
        .map(|e| e.expect("readable dir entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "c"))
        .collect();
    shims.sort();
    b.files(&shims).include(&csrc);
    b.compile("opus");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=../../vendor/libopus");
}
