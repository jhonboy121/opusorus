//! `qext_compare`: port of libopus `src/qext_compare.c` (Opus HD quality metric).
//!
//! Usage: `qext_compare [-s] [-48k] [-s16|-s24|-f32] [-r rate2] [-thresholds err4 err16 rms]
//! <file1.sw> <file2.sw>`.

use std::process::ExitCode;

#[cfg(not(feature = "fixed-point"))]
fn main() -> std::io::Result<ExitCode> {
    let args: Vec<String> = std::env::args().collect();
    let code = opusorus_tools::compare::qext_compare_main(&args, &mut std::io::stderr().lock())?;
    Ok(if code == opusorus_tools::compare::EXIT_SUCCESS {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

#[cfg(feature = "fixed-point")]
fn main() -> std::io::Result<ExitCode> {
    opusorus_tools::fixed_point_unavailable("qext_compare")
}
