//! Feature `assertions` (libopus `ENABLE_ASSERTIONS`) vs the oracle built with the same define.
//!
//! * One check of each kind (`celt_assert`, `celt_sig_assert`, `silk_assert`) is failed on
//!   purpose on both sides: the C oracle aborts (checked in a child process) and the port
//!   panics. With the feature this holds in every build profile (`just test-options` also runs
//!   this file with `--release`, where `debug_assert!` is compiled out); without it the oracle
//!   (built with `ENABLE_HARDENING`, like upstream's default) still aborts on `celt_assert` only.
//! * That no check fails on valid input, and that enabling them changes no output, is covered
//!   by running the other differential suites against the `ENABLE_ASSERTIONS` oracle with
//!   `--features assertions` (`just test-options`).
#![allow(
    clippy::unwrap_used,
    clippy::print_stderr,
    reason = "test code: failures should panic; notes go to stderr"
)]

use std::process::Command;

use opusorus_oracle::build_options as c;

#[test]
fn oracle_has_the_same_option() {
    assert_eq!(c::options().assertions, cfg!(feature = "assertions"));
}

/// The environment variable that makes [`c_child`] fail a C check.
const CHILD_ENV: &str = "OPUSORUS_C_ASSERTION";

/// Child process body: fails the C check named by [`CHILD_ENV`] (no-op otherwise).
#[test]
fn c_child() {
    match std::env::var(CHILD_ENV).as_deref() {
        Ok("celt") => c::fail_celt_assert(),
        Ok("sig") => {
            let v = c::fail_celt_sig_assert();
            eprintln!("celt_sig_assert not checked (value {v})");
        }
        Ok("silk") => c::fail_silk_assert(),
        Ok(other) => panic!("unknown {CHILD_ENV} value {other:?}"),
        Err(_) => return,
    }
    eprintln!("C check did not abort");
}

/// Runs [`c_child`] in a child process for `kind`; returns whether it aborted with libopus'
/// fatal-error message.
fn c_aborts(kind: &str) -> bool {
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "c_child", "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, kind)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    let fatal = stderr.contains("Fatal (internal) error");
    assert_eq!(
        fatal,
        !out.status.success(),
        "{kind}: unexpected child result {:?}:\n{stderr}",
        out.status
    );
    if fatal {
        let msg = if kind == "sig" {
            "signal assertion failed"
        } else {
            "assertion failed"
        };
        assert!(stderr.contains(msg), "{kind}: {stderr}");
    }
    fatal
}

#[cfg(unix)]
#[test]
fn c_checks_abort() {
    // celt_assert is also a hard check in ENABLE_HARDENING builds (the oracle's default).
    assert!(c_aborts("celt"));
    assert_eq!(c_aborts("sig"), cfg!(feature = "assertions"));
    assert_eq!(c_aborts("silk"), cfg!(feature = "assertions"));
}

/// Runs `f` and returns whether it panicked.
fn panics(f: impl FnOnce() + std::panic::UnwindSafe) -> bool {
    static HOOK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = HOOK.lock().unwrap();
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(f).is_err();
    std::panic::set_hook(hook);
    r
}

fn rust_fail_celt_assert() {
    let mut buf = [0u8; 16];
    let mut enc = opusorus::celt::entenc::EcEnc::new(&mut buf);
    // celt_assert!(ft > 1)
    enc.enc_uint(0, 1);
}

const fn rust_fail_celt_sig_assert() {
    // Same functions as the C shim: float celt_atan2p_norm(-1, 1), fixed celt_ilog2(0).
    #[cfg(not(feature = "fixed-point"))]
    let v = opusorus::celt::mathops::celt_atan2p_norm(-1.0, 1.0) as i32;
    #[cfg(feature = "fixed-point")]
    let v = opusorus::celt::mathops::celt_ilog2(0);
    std::hint::black_box(v);
}

fn rust_fail_silk_assert() {
    let mut nlsf: Vec<i16> = (1..=16).map(|i| 1000 * i).collect();
    let mut delta = [100i16; 17];
    delta[16] = 0;
    // silk_assert!(ndelta_min_q15[l] >= 1)
    opusorus::silk::nlsf::silk_nlsf_stabilize(&mut nlsf, &delta, 16);
}

/// The port's checks: hard with the feature (any profile), debug assertions otherwise.
#[test]
fn rust_checks_panic() {
    let expected = cfg!(any(feature = "assertions", debug_assertions));
    eprintln!(
        "assertions feature: {}, debug assertions: {}",
        cfg!(feature = "assertions"),
        cfg!(debug_assertions)
    );
    assert_eq!(panics(rust_fail_celt_assert), expected, "celt_assert");
    assert_eq!(
        panics(rust_fail_celt_sig_assert),
        expected,
        "celt_sig_assert"
    );
    assert_eq!(panics(rust_fail_silk_assert), expected, "silk_assert");
}

/// `celt/modes.c:compute_ebands` checks its band layout with `celt_assert`; the port only
/// compiles those checks with the feature (see `celt/modes.rs`): a mode whose layout fails them
/// is rejected with a panic like C's abort.
#[cfg(all(feature = "assertions", feature = "custom-modes"))]
#[test]
fn custom_mode_band_layout_checks() {
    use opusorus::celt::modes::opus_custom_mode_create_custom;
    assert!(!panics(|| {
        opus_custom_mode_create_custom(48000, 960).unwrap();
    }));
    // A layout that violates "every band must be smaller than the last band" (bands
    // `..., 16, 20, 22`).
    assert!(panics(|| {
        let _mode = opus_custom_mode_create_custom(12000, 208);
    }));
}
