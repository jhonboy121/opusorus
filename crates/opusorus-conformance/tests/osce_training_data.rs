//! OSCE training data output (libopus `--enable-osce-training-data`,
//! `ENABLE_OSCE_TRAINING_DATA`): the Rust `opus_demo` with the `osce-training-data` feature vs
//! the unmodified C `src/opus_demo.c` compiled with the define and linked with the oracle
//! built with it. Every run must write the same training files (`clean_hp.s16`,
//! `features_*.f32/s16/s32`, `noisy_16k.s16`) byte for byte, besides the same output file,
//! stdout/stderr text and exit status.
//!
//! Run with `cargo test -p opusorus-conformance --features osce-training-data --test
//! osce_training_data` (optionally `+ dred`, `qext`). The training files go to the working
//! directory, which is process-wide: the cases run one after the other in a single test, each
//! in its own directory (the C tool as a child process started there, the Rust tool in-process
//! after `set_current_dir`). Do not run the other conformance tests in this configuration:
//! every SILK `voip` encode and 16 kHz SILK decode in them would write training files too.

#![cfg(all(unix, feature = "osce-training-data"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "test code: failures should panic; notes go to stdout/stderr"
)]

use opusorus_conformance::signals;
use opusorus_tools::demo::opus_demo_main_with_weights;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The training files of a build with `ENABLE_OSCE_TRAINING_DATA`.
const TRAINING_FILES: [&str; 8] = [
    "clean_hp.s16",
    "features_lpc.f32",
    "features_gain.f32",
    "features_ltp.f32",
    "features_period.s16",
    "noisy_16k.s16",
    "features_num_bits.s32",
    "features_num_bits_smooth.f32",
];

/// The oracle's compiled-in weights as a blob (the models of the enabled features), for the
/// Rust tool (the C tool has them compiled in).
fn oracle_blob() -> Vec<u8> {
    use opusorus_oracle::dnn_core::{
        MODEL_BBWENET, MODEL_FARGAN, MODEL_LACE, MODEL_NOLACE, MODEL_PITCHDNN, MODEL_PLC,
        MODEL_RDOVAE_DEC, MODEL_RDOVAE_ENC, write_blob,
    };
    let mut models = vec![MODEL_PITCHDNN, MODEL_FARGAN, MODEL_PLC];
    if cfg!(feature = "dred") {
        models.extend([MODEL_RDOVAE_ENC, MODEL_RDOVAE_DEC]);
    }
    models.extend([MODEL_LACE, MODEL_NOLACE, MODEL_BBWENET]);
    models.iter().flat_map(|&m| write_blob(m)).collect()
}

/// The newest oracle `libopus.a` built with this test's configuration (training data, OSCE,
/// and the optional components of the enabled features).
fn find_oracle_lib() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let profile = exe.parent().unwrap().parent().unwrap();
    let mut libs: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(profile.join("build")).unwrap() {
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
        let b = std::fs::read(&lib).unwrap();
        let has = |needle: &[u8]| b.windows(needle.len()).any(|w| w == needle);
        if has(b"features_lpc.f32")
            && has(b"-osce.o")
            && has(b"-dred_decoder.o") == cfg!(feature = "dred")
            && has(b"-mini_kfft.o") == cfg!(feature = "qext")
            && has(b"opus_custom_encoder_create") == cfg!(feature = "custom-modes")
            && has(b"opusorus-oracle: dnn-debug-float build") == cfg!(feature = "dnn-debug-float")
        {
            libs.push((meta.modified().unwrap(), lib));
        }
    }
    libs.sort_by_key(|l| core::cmp::Reverse(l.0));
    libs.into_iter()
        .map(|(_, p)| p)
        .next()
        .expect("oracle libopus.a with ENABLE_OSCE_TRAINING_DATA not found")
}

/// Compiles `src/opus_demo.c` with `ENABLE_OSCE_TRAINING_DATA` (upstream's config.h for
/// `--enable-osce-training-data`: also `ENABLE_OSCE` + `ENABLE_OSCE_BWE`, and the deep PLC the
/// Rust `osce` feature implies). `None` (with a note) without a C compiler.
fn build_c_opus_demo(dir: &Path) -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/libopus");
    let cc = match std::env::var_os("CC") {
        Some(cc) => cc.to_string_lossy().into_owned(),
        None => "cc".to_owned(),
    };
    let exe = dir.join("opus_demo_training");
    let mut cmd = std::process::Command::new(&cc);
    cmd.args(["-O2", "-ffp-contract=off", "-fno-fast-math", "-w"])
        .args([
            "-DOPUS_BUILD",
            "-DVAR_ARRAYS",
            "-DHAVE_LRINT",
            "-DHAVE_LRINTF",
        ])
        .args(["-DENABLE_DEEP_PLC", "-DENABLE_OSCE", "-DENABLE_OSCE_BWE"])
        .arg("-DENABLE_OSCE_TRAINING_DATA");
    if cfg!(feature = "dred") {
        cmd.arg("-DENABLE_DRED");
    }
    if cfg!(feature = "qext") {
        cmd.arg("-DENABLE_QEXT");
    }
    if cfg!(feature = "custom-modes") {
        cmd.arg("-DCUSTOM_MODES");
    }
    for inc in ["include", "celt", "silk", "src", "dnn"] {
        cmd.arg(format!("-I{}", root.join(inc).display()));
    }
    cmd.arg(root.join("src/opus_demo.c"))
        .arg(find_oracle_lib())
        .args(["-lm", "-o"])
        .arg(&exe);
    match cmd.output() {
        Ok(out) => {
            assert!(
                out.status.success(),
                "compiling opus_demo.c failed:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            Some(exe)
        }
        Err(e) => {
            eprintln!("NOTE: cannot run the C compiler `{cc}` ({e}); skipping");
            None
        }
    }
}

/// Replaces the version line (the oracle is built without `PACKAGE_VERSION`).
fn normalize_version(stderr: &str) -> String {
    let mut lines: Vec<&str> = stderr.split_inclusive('\n').collect();
    if let Some(first) = lines.first_mut()
        && (*first == "libopus unknown\n"
            || *first == format!("{}\n", opusorus::celt::celt::opus_get_version_string()))
    {
        *first = "libopus <version>\n";
    }
    lines.concat()
}

/// The files of `dir` (name → contents).
fn dir_files(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            (
                e.file_name().to_string_lossy().into_owned(),
                std::fs::read(e.path()).unwrap(),
            )
        })
        .collect()
}

fn fresh_dir(p: &Path) {
    if p.exists() {
        std::fs::remove_dir_all(p).unwrap();
    }
    std::fs::create_dir_all(p).unwrap();
}

/// Runs `args` (`OUT` = `out.bin` in the case directory) with both tools, each in its own
/// empty working directory, and compares everything they print and write. Returns the files.
fn check_case(
    c_demo: &Path,
    base: &Path,
    weights: Option<&[u8]>,
    name: &str,
    args: &[&str],
) -> BTreeMap<String, Vec<u8>> {
    use std::os::unix::process::CommandExt;
    let c_dir = base.join("c").join(name);
    let rs_dir = base.join("rs").join(name);
    fresh_dir(&c_dir);
    fresh_dir(&rs_dir);
    let with_out = |d: &Path| -> Vec<String> {
        args.iter()
            .map(|a| {
                if *a == "OUT" {
                    d.join("out.bin").to_str().unwrap().to_owned()
                } else {
                    (*a).to_owned()
                }
            })
            .collect()
    };
    let c_out = std::process::Command::new(c_demo)
        .arg0("opus_demo")
        .args(with_out(&c_dir))
        .current_dir(&c_dir)
        .output()
        .unwrap();

    let mut rs_args = vec!["opus_demo".to_owned()];
    rs_args.extend(with_out(&rs_dir));
    let (mut out, mut err) = (Vec::new(), Vec::new());
    std::env::set_current_dir(&rs_dir).unwrap();
    let code = opus_demo_main_with_weights(&rs_args, &mut out, &mut err, weights).unwrap();
    std::env::set_current_dir(base).unwrap();

    assert_eq!(
        Some(code & 0xFF),
        c_out.status.code(),
        "{name}: exit status"
    );
    assert_eq!(
        String::from_utf8(out).unwrap(),
        String::from_utf8(c_out.stdout).unwrap(),
        "{name}: stdout"
    );
    assert_eq!(
        normalize_version(&String::from_utf8(err).unwrap()),
        normalize_version(&String::from_utf8(c_out.stderr).unwrap()),
        "{name}: stderr"
    );
    let (rs_files, c_files) = (dir_files(&rs_dir), dir_files(&c_dir));
    assert_eq!(
        rs_files.keys().collect::<Vec<_>>(),
        c_files.keys().collect::<Vec<_>>(),
        "{name}: files written"
    );
    for (f, rs) in &rs_files {
        let c = &c_files[f];
        assert!(
            rs == c,
            "{name}: {f} differs ({} vs {} bytes, first difference at byte {:?})",
            rs.len(),
            c.len(),
            rs.iter().zip(c).position(|(a, b)| a != b)
        );
    }
    println!("{name}: identical ({} files)", rs_files.len());
    rs_files
}

fn write_s16(path: &Path, x: &[f32]) {
    let b: Vec<u8> = signals::to_i16(x)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(path, b).unwrap();
}

#[test]
fn training_files_match_c() {
    let base = Path::new(env!("CARGO_TARGET_TMPDIR")).join("osce_training_data");
    fresh_dir(&base);
    let Some(c_demo) = build_c_opus_demo(&base) else {
        return;
    };
    let blob = oracle_blob();
    let weights = if opusorus_tools::demo::WEIGHTS_EMBEDDED {
        None
    } else {
        Some(&blob[..])
    };
    let inp = |n: &str| base.join(n).to_str().unwrap().to_owned();
    write_s16(
        &base.join("s16m.sw"),
        &signals::speech_like(16000 * 3, 1, 16000, 21),
    );
    write_s16(
        &base.join("s48s.sw"),
        &signals::speech_like(48000 * 2, 2, 48000, 22),
    );
    write_s16(
        &base.join("m16s.sw"),
        &signals::music_like(16000 * 2, 2, 16000, 23),
    );
    write_s16(
        &base.join("s8m.sw"),
        &signals::speech_like(8000 * 2, 1, 8000, 24),
    );
    let (s16m, s48s, m16s, s8m) = (
        inp("s16m.sw"),
        inp("s48s.sw"),
        inp("m16s.sw"),
        inp("s8m.sw"),
    );
    let run = |name: &str, args: &[&str]| check_case(&c_demo, &base, weights, name, args);

    // Encode + decode, NoLACE (complexity 7) and LACE (6): all training files.
    let f = run(
        "ed_voip16m_nolace",
        &[
            "voip",
            "16000",
            "1",
            "16000",
            "-dec_complexity",
            "7",
            &s16m,
            "OUT",
        ],
    );
    for t in TRAINING_FILES {
        assert!(f.get(t).is_some_and(|b| !b.is_empty()), "{t} written");
    }
    // 50 frames of 20 ms per second (in 16-bit samples: 320 per frame).
    assert!(f["noisy_16k.s16"].len() >= 2 * 320 * 140);
    assert_eq!(
        f["features_lpc.f32"].len(),
        16 * f["features_gain.f32"].len()
    );
    run(
        "ed_voip16m_lace_loss",
        &[
            "voip",
            "16000",
            "1",
            "12000",
            "-dec_complexity",
            "6",
            "-loss",
            "10",
            &s16m,
            "OUT",
        ],
    );
    // -silk_random_switching after srand(0), with the random-setting printouts on stdout.
    run(
        "ed_random_switching",
        &[
            "voip",
            "16000",
            "1",
            "16000",
            "-silk_random_switching",
            "7",
            "-loss",
            "5",
            &s16m,
            "OUT",
        ],
    );
    run(
        "ed_random_switching_every_frame",
        &[
            "voip",
            "48000",
            "1",
            "24000",
            "-silk_random_switching",
            "1",
            "-random_fec",
            &s48s,
            "OUT",
        ],
    );
    // Stereo: C writes `frame_size` samples from the interleaved buffer (clean_hp) and one set
    // of features per decoded channel.
    run(
        "ed_voip48s_wb",
        &[
            "voip",
            "48000",
            "2",
            "24000",
            "-bandwidth",
            "WB",
            "-dec_complexity",
            "7",
            &s48s,
            "OUT",
        ],
    );
    run(
        "ed_voip16s_music",
        &[
            "voip",
            "16000",
            "2",
            "32000",
            "-framesize",
            "10",
            &m16s,
            "OUT",
        ],
    );
    // Non-voip applications: no clean_hp.s16 (forced SILK-only all the same).
    run(
        "ed_audio16m",
        &[
            "audio",
            "16000",
            "1",
            "20000",
            "-framesize",
            "40",
            &s16m,
            "OUT",
        ],
    );
    // 8 kHz: no OSCE features (enhancement only runs at 16 kHz).
    run("ed_voip8m", &["voip", "8000", "1", "8000", &s8m, "OUT"]);
    // Encode only (clean_hp only) and decode only (features only).
    let e = run(
        "e_voip16m",
        &[
            "-e",
            "voip",
            "16000",
            "1",
            "20000",
            "-inbandfec",
            "-loss",
            "15",
            &s16m,
            "OUT",
        ],
    );
    let bit = base.join("e_voip16m.bit");
    std::fs::write(&bit, &e["out.bin"]).unwrap();
    let bit = bit.to_str().unwrap().to_owned();
    run(
        "d_16m_nolace",
        &[
            "-d",
            "16000",
            "1",
            "-dec_complexity",
            "7",
            "-inbandfec",
            "-loss",
            "20",
            &bit,
            "OUT",
        ],
    );
    run(
        "d_48s_lace",
        &["-d", "48000", "2", "-dec_complexity", "6", &bit, "OUT"],
    );
    run(
        "d_16m_bwe",
        &["-d", "48000", "1", "-enable_osce_bwe", &bit, "OUT"],
    );
    if cfg!(feature = "dred") {
        run(
            "ed_dred",
            &[
                "voip",
                "16000",
                "1",
                "24000",
                "-dred",
                "40",
                "-loss",
                "20",
                "-dec_complexity",
                "5",
                &s16m,
                "OUT",
            ],
        );
    }
    // Accepted (and announced) in decode-only mode too, as in C.
    run(
        "d_random_switching_ignored",
        &[
            "-d",
            "16000",
            "1",
            "-silk_random_switching",
            "3",
            &bit,
            "OUT",
        ],
    );

    // Library level: an I/O error is kept and reported by close_all (C would crash on the NULL
    // FILE*). The working directory no longer exists, so the files cannot be created.
    let gone = base.join("gone");
    fresh_dir(&gone);
    std::env::set_current_dir(&gone).unwrap();
    std::fs::remove_dir(&gone).unwrap();
    let mut enc = opusorus::Encoder::new(16000, 1, opusorus::Application::Voip).unwrap();
    let mut pkt = [0u8; 1500];
    enc.encode(&[1000i16; 320], 320, &mut pkt).unwrap();
    std::env::set_current_dir(&base).unwrap();
    assert!(opusorus::osce_training_data::close_all().is_err());
    assert!(opusorus::osce_training_data::close_all().is_ok());
}
