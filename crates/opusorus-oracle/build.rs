//! Builds vendored libopus v1.6.1 as a static library for use as a test oracle.
//!
//! Configuration mirrors a default upstream build (float, hardening on) except:
//! * intrinsics/RTCD are disabled so the scalar C reference paths are used, and
//! * `-ffp-contract=off` so the C compiler does not fuse multiply-adds.
//!
//! Both are required for the Rust port to be bit-exact with the oracle.
//!
//! Features `fixed-point` / `fixed-res24` build the fixed-point library instead (`FIXED_POINT`,
//! `silk/fixed` sources and include path, `ENABLE_RES24`), keeping the float API on as upstream
//! does by default (so `src/analysis.c` + `src/mlp*.c` are still compiled, as in upstream's
//! Makefile.am/meson/CMake). Shims in `csrc/` declare which builds they support with a marker
//! line `// oracle-build: float|fixed|any` (no marker = float only, see docs/FIXED_POINT.md).

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

/// The oracle configurations a shim compiles in: the value of its `// oracle-build: <x>` marker
/// line (`float`, `fixed` or `any`); shims without a marker are float-only.
fn shim_build(path: &Path) -> String {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    for line in text.lines() {
        if let Some(v) = line.trim().strip_prefix("// oracle-build:") {
            let v = v.trim();
            assert!(
                matches!(v, "float" | "fixed" | "any"),
                "{}: bad oracle-build marker {v:?}",
                path.display()
            );
            return v.to_string();
        }
    }
    "float".to_string()
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets this"));
    let root = manifest.join("../../vendor/libopus");
    let fixed = std::env::var_os("CARGO_FEATURE_FIXED_POINT").is_some();
    let res24 = std::env::var_os("CARGO_FEATURE_FIXED_RES24").is_some();
    let qext = std::env::var_os("CARGO_FEATURE_QEXT").is_some();
    let custom = std::env::var_os("CARGO_FEATURE_CUSTOM_MODES").is_some();
    // DNN features mirror upstream configure: --enable-dred and --enable-osce both imply the
    // deep PLC sources and ENABLE_DEEP_PLC; --enable-osce also defines ENABLE_OSCE_BWE.
    let dred = std::env::var_os("CARGO_FEATURE_DRED").is_some();
    let osce = std::env::var_os("CARGO_FEATURE_OSCE").is_some();
    let deep_plc = std::env::var_os("CARGO_FEATURE_DEEP_PLC").is_some() || dred || osce;
    // --enable-dnn-debug-float (no DISABLE_DEBUG_FLOAT), --enable-osce-training-data (implies
    // OSCE through the Cargo feature) and --enable-lossgen (sources for the csrc/dnn_lossgen.c shim).
    let dnn_debug_float = std::env::var_os("CARGO_FEATURE_DNN_DEBUG_FLOAT").is_some();
    let osce_training = std::env::var_os("CARGO_FEATURE_OSCE_TRAINING_DATA").is_some();
    let lossgen = std::env::var_os("CARGO_FEATURE_LOSSGEN").is_some();
    // Upstream configure refuses this combination too.
    assert!(
        !(fixed && deep_plc),
        "--enable-fixed-point cannot be used with --enable-deep-plc, --enable-dred, and \
         --enable-osce (features fixed-point + deep-plc/dred/osce)"
    );

    let mut srcs = mk_sources(&root, "celt_sources.mk", "CELT_SOURCES");
    srcs.extend(mk_sources(&root, "silk_sources.mk", "SILK_SOURCES"));
    srcs.extend(mk_sources(&root, "opus_sources.mk", "OPUS_SOURCES"));
    // Upstream: SILK_SOURCES_FIXED or SILK_SOURCES_FLOAT by FIXED_POINT; OPUS_SOURCES_FLOAT
    // (analysis + MLP) whenever the float API is enabled, which is the default in both builds.
    if fixed {
        srcs.extend(mk_sources(&root, "silk_sources.mk", "SILK_SOURCES_FIXED"));
    } else {
        srcs.extend(mk_sources(&root, "silk_sources.mk", "SILK_SOURCES_FLOAT"));
    }
    srcs.extend(mk_sources(&root, "opus_sources.mk", "OPUS_SOURCES_FLOAT"));
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
        b.define("FIXED_POINT", "1");
    }
    if res24 {
        b.define("ENABLE_RES24", None);
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
        if !dnn_debug_float {
            b.define("DISABLE_DEBUG_FLOAT", None);
        }
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
        if osce_training {
            b.define("ENABLE_OSCE_TRAINING_DATA", None);
        }
    }
    if lossgen {
        // The model data is part of the DNN model tarball (not vendored in git).
        let data = root.join("dnn/lossgen_data.c");
        assert!(
            data.exists(),
            "{} missing (feature lossgen); run scripts/fetch_dnn_models.sh",
            data.display()
        );
        // lossgen.c runs the DNN kernels of dnn/vec.h: the generic C path, as for the DNN
        // features (see below).
        b.include(root.join("dnn"))
            .define("ENABLE_LOSSGEN", None)
            .define("DISABLE_NEON", None)
            .flag_if_supported("-U__AVX__")
            .flag_if_supported("-U__SSE2__");
    }
    // Per-unit C shims exposing internal functions with flat, FFI-friendly signatures.
    let csrc = manifest.join("csrc");
    let wanted = if fixed { "fixed" } else { "float" };
    let mut shims: Vec<PathBuf> = std::fs::read_dir(&csrc)
        .expect("csrc dir exists")
        .map(|e| e.expect("readable dir entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "c"))
        .filter(|p| {
            let build = shim_build(p);
            build == "any" || build == wanted
        })
        .collect();
    shims.sort();
    b.files(&shims).include(&csrc);
    b.compile("opus");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=../../vendor/libopus");
}
