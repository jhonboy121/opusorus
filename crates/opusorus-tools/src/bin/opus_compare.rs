//! `opus_compare`: port of libopus `src/opus_compare.c` (RFC 6716 test vector quality metric).
//!
//! Usage: `opus_compare [-s] [-r rate2] <file1.sw> <file2.sw>` (16-bit little-endian PCM).

use std::process::ExitCode;

fn main() -> std::io::Result<ExitCode> {
    let args: Vec<String> = std::env::args().collect();
    let code = opusorus_tools::compare::opus_compare_main(&args, &mut std::io::stderr().lock())?;
    Ok(if code == opusorus_tools::compare::EXIT_SUCCESS {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}
