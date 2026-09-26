//! Compiles the variadic `*_ctl` glue (`csrc/ctl.c`) against the shipped libopus headers.

use std::error::Error;
use std::path::PathBuf;

/// Architectures with a naked-function trampoline in `src/ctl.rs` (keep in sync with the
/// `trampolines` module there).
const TRAMPOLINE_ARCHS: &[&str] = &["x86_64", "x86", "aarch64", "arm", "riscv64"];

fn main() -> Result<(), Box<dyn Error>> {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH")?;
    let mut b = cc::Build::new();
    b.file(manifest.join("csrc/ctl.c"))
        .include(manifest.join("include"))
        .warnings(true);
    if TRAMPOLINE_ARCHS.contains(&arch.as_str()) {
        b.define("OPUSORUS_CTL_TRAMPOLINE", None);
    }
    if std::env::var_os("CARGO_FEATURE_CUSTOM_MODES").is_some() {
        b.define("OPUSORUS_CUSTOM_MODES", None);
    }
    b.try_compile("opusorus_ctl")?;
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=include");
    Ok(())
}
