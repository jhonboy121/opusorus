//! Differential tests for the packet loss generator of libopus `--enable-lossgen`
//! (dnn/lossgen.c, lossgen_data.c, lossgen_demo.c) vs the C code (oracle feature `lossgen`).
//!
//! Run with `cargo test -p opusorus-conformance --features lossgen --test lossgen` (float or
//! fixed-point: upstream builds lossgen in either; needs the model data from
//! `scripts/fetch_dnn_models.sh`). `opus_demo -sim_loss` is compared with the C `opus_demo`
//! built with `ENABLE_LOSSGEN` in `vectors.rs` (`opus_demo_matches_c`).
//!
//! * the compiled-in Rust tables equal the oracle's `lossgen_arrays` byte for byte;
//! * loss decisions *and* the GRU states are bit-exact with C after every packet, for many
//!   `rand()` seeds and loss percentages (constant and time-varying), with the compiled-in model
//!   and with models loaded from weight blobs (`lossgen_load_model`);
//! * `lossgen_demo` prints exactly what the C `lossgen_demo` prints (compiled with the system C
//!   compiler, generic C DNN kernels as in the oracle; skipped with a note without one).

#![cfg(feature = "lossgen")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    reason = "test code: failures should panic; notes go to stderr"
)]

use opusorus::dnn::lossgen_data::{as_weight_arrays, lossgen_arrays};
use opusorus::dnn::parse_lpcnet_weights::write_weights;
use opusorus::glibc_rand::GlibcRand;
use opusorus::lossgen::{LossGenState, sample_loss};
use opusorus_oracle::dnn_lossgen as c;
use std::path::{Path, PathBuf};

/// Asserts that the Rust state equals the C snapshot bit for bit.
fn assert_state(what: &str, r: &LossGenState, h: &c::LossGenSnapshot) {
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&r.gru1_state), bits(&h.gru1), "{what}: gru1");
    assert_eq!(bits(&r.gru2_state), bits(&h.gru2), "{what}: gru2");
    assert_eq!((r.last_loss, r.used), (h.last_loss, h.used), "{what}");
}

/// Runs both generators over `percents` (one packet each) after `srand(seed)`, comparing every
/// decision and state; returns the number of losses.
fn run_both(
    what: &str,
    r: &mut LossGenState,
    h: &mut c::LossGen,
    seed: u32,
    percents: &[f32],
) -> usize {
    let _rand = c::with_rand();
    c::srand(seed);
    let mut rng = GlibcRand::new(seed);
    let mut lost = 0;
    for (i, &p) in percents.iter().enumerate() {
        let lr = sample_loss(r, p, &mut || rng.next_value());
        let lc = h.sample(p);
        assert_eq!(lr, lc, "{what}: packet {i} (percent {p})");
        assert_state(&format!("{what}: packet {i}"), r, &h.snapshot());
        lost += usize::from(lr != 0);
    }
    lost
}

fn model_blob() -> Vec<u8> {
    let owned = lossgen_arrays();
    let mut blob = Vec::new();
    write_weights(&as_weight_arrays(&owned), &mut blob);
    blob
}

#[test]
fn tables_match_oracle() {
    let owned = lossgen_arrays();
    let oracle = c::arrays();
    assert_eq!(owned.len(), oracle.len(), "array count");
    for ((name, ty, data), o) in owned.iter().zip(&oracle) {
        assert_eq!(*name, o.name);
        assert_eq!((*ty, data.len()), (o.type_, o.size as usize), "{name}");
        assert!(*data == o.data, "{name}: data differs");
    }
    // The initial states agree too (cleared, not used).
    let r = LossGenState::lossgen_init();
    assert_state("init", &r, &c::LossGen::new().snapshot());
}

#[test]
fn glibc_rand_matches_c() {
    let _rand = c::with_rand();
    for seed in [0u32, 1, 2, 42, 0x8000_0000, u32::MAX] {
        c::srand(seed);
        let mut r = GlibcRand::new(seed);
        for i in 0..2000 {
            assert_eq!(r.next_value(), c::rand(), "seed {seed}: draw {i}");
        }
    }
}

#[test]
fn loss_patterns_match_c() {
    let percents = [
        0.0f32, 0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, -0.1, 1.5,
    ];
    for seed in [1u32, 0, 2, 42, 12345, 0xDEAD_BEEF, u32::MAX] {
        for &p in &percents {
            let mut r = LossGenState::lossgen_init();
            let mut h = c::LossGen::new();
            let lost = run_both(
                &format!("seed {seed} p {p}"),
                &mut r,
                &mut h,
                seed,
                &[p; 1500],
            );
            if (0.05..=0.75).contains(&p) {
                // Sanity: the generator produces losses at roughly the requested rate.
                let rate = lost as f32 / 1500.0;
                assert!(
                    rate > p / 3.0 && rate < p * 3.0,
                    "seed {seed} p {p}: rate {rate}"
                );
            }
        }
    }
}

#[test]
fn time_varying_percentages_match_c() {
    let mut rng = opusorus_conformance::Rng::new(0x1055_6E17);
    for seed in [7u32, 99, 31337] {
        let percents: Vec<f32> = (0..4000)
            .map(|i| {
                if i % 500 < 250 {
                    (i % 97) as f32 / 97.0
                } else {
                    rng.range_i32(0, 1000) as f32 * 1e-3
                }
            })
            .collect();
        let mut r = LossGenState::lossgen_init();
        let mut h = c::LossGen::new();
        run_both(
            &format!("varying seed {seed}"),
            &mut r,
            &mut h,
            seed,
            &percents,
        );
        // Continuing without reseeding (`used` is set: no second warm-up).
        run_both(
            &format!("varying seed {seed} (again)"),
            &mut r,
            &mut h,
            seed + 1,
            &percents[..500],
        );
    }
}

#[test]
fn blob_loaded_models_match_c() {
    let blob = model_blob();
    // The compiled-in model through the blob path.
    let mut r = LossGenState::lossgen_init();
    r.lossgen_load_model(&blob).unwrap();
    let (mut h, ret) = c::LossGen::from_blob(&blob);
    assert_eq!(ret, 0);
    run_both("blob", &mut r, &mut h, 5, &[0.15; 2000]);

    // A modified model (every float weight/bias scaled): both sides use the loaded weights.
    let owned: Vec<_> = lossgen_arrays()
        .into_iter()
        .map(|(name, ty, data)| {
            if ty == opusorus::dnn::nnet::WEIGHT_TYPE_FLOAT {
                let d = data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|&b| (f32::from_le_bytes(b) * 1.25).to_le_bytes())
                    .collect();
                (name, ty, d)
            } else {
                (name, ty, data)
            }
        })
        .collect();
    let mut modified = Vec::new();
    write_weights(&as_weight_arrays(&owned), &mut modified);
    let mut r = LossGenState::lossgen_init();
    r.lossgen_load_model(&modified).unwrap();
    assert_ne!(r.model, LossGenState::lossgen_init().model);
    let (mut h, ret) = c::LossGen::from_blob(&modified);
    assert_eq!(ret, 0);
    run_both("modified blob", &mut r, &mut h, 11, &[0.3; 2000]);

    // A well-formed blob missing an array: both refuse it.
    let owned: Vec<_> = lossgen_arrays()
        .into_iter()
        .filter(|(name, _, _)| *name != "lossgen_gru2_recurrent_scale")
        .collect();
    let mut missing = Vec::new();
    write_weights(&as_weight_arrays(&owned), &mut missing);
    let mut r = LossGenState::lossgen_init();
    assert!(r.lossgen_load_model(&missing).is_err());
    assert_eq!(c::LossGen::from_blob(&missing).1, -1);
    // (A malformed blob makes C dereference a NULL list; Rust returns an error.)
    assert!(r.lossgen_load_model(&blob[..blob.len() - 3]).is_err());
}

// ---------------------------------------------------------------------------------------------
// lossgen_demo vs the C lossgen_demo
// ---------------------------------------------------------------------------------------------

/// Compiles upstream's `dnn/lossgen_demo.c` + `lossgen.c` + `lossgen_data.c` (standalone, as
/// Makefile.am's `lossgen_demo`) with the oracle's flags (generic C DNN kernels, no FMA
/// contraction). `None` (with a note) if no C compiler can be run.
#[cfg(unix)]
fn build_c_lossgen_demo() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/libopus");
    let cc = match std::env::var_os("CC") {
        Some(cc) => cc.to_string_lossy().into_owned(),
        None => "cc".to_owned(),
    };
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("lossgen");
    std::fs::create_dir_all(&dir).unwrap();
    let exe = dir.join("lossgen_demo");
    let mut cmd = std::process::Command::new(&cc);
    cmd.args([
        "-O2",
        "-ffp-contract=off",
        "-fno-fast-math",
        "-w",
        "-DDISABLE_NEON",
    ])
    .args(["-U__SSE2__", "-U__AVX__"]);
    for inc in ["dnn", "celt", "include"] {
        cmd.arg(format!("-I{}", root.join(inc).display()));
    }
    for src in ["dnn/lossgen_demo.c", "dnn/lossgen.c", "dnn/lossgen_data.c"] {
        cmd.arg(root.join(src));
    }
    cmd.args(["-lm", "-o"]).arg(&exe);
    match cmd.output() {
        Ok(out) => {
            assert!(
                out.status.success(),
                "compiling lossgen_demo failed:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            Some(exe)
        }
        Err(e) => {
            eprintln!("NOTE: cannot run the C compiler `{cc}` ({e}); skipping lossgen_demo vs C");
            None
        }
    }
}

#[cfg(unix)]
#[test]
fn lossgen_demo_matches_c() {
    use std::os::unix::process::CommandExt;
    let Some(exe) = build_c_lossgen_demo() else {
        return;
    };
    let cases: &[&[&str]] = &[
        &["10", "3000"],
        &["0", "500"],
        &["2.5", "4000"],
        &["35", "2500"],
        &["100", "300"],
        &["50.75", "1000"],
        &["-5", "200"],
        &["abc", "10"],
        &["20", "0"],
        &["20", "-3"],
        &["20", " 17xyz"],
        &["20"],
        &[],
        &["1", "2", "3"],
    ];
    for args in cases {
        let c_out = std::process::Command::new(&exe)
            .arg0("lossgen_demo")
            .args(*args)
            .output()
            .unwrap();
        let mut argv = vec!["lossgen_demo".to_owned()];
        argv.extend(args.iter().map(|s| (*s).to_owned()));
        let mut out = Vec::new();
        let mut err = Vec::new();
        let code =
            opusorus_tools::lossgen_demo::lossgen_demo_main(&argv, &mut out, &mut err).unwrap();
        assert_eq!(Some(code), c_out.status.code(), "{args:?}: exit status");
        assert!(out == c_out.stdout, "{args:?}: stdout differs");
        assert_eq!(
            String::from_utf8(err).unwrap(),
            String::from_utf8(c_out.stderr).unwrap(),
            "{args:?}: stderr"
        );
    }
}
