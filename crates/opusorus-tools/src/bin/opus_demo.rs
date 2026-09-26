//! `opus_demo`: port of libopus `src/opus_demo.c` (reference encoder/decoder front end).
//!
//! Usage: `opus_demo [-e] <application> <sampling rate (Hz)> <channels (1/2)> <bits per second>
//! [options] <input> <output>` or `opus_demo -d <sampling rate (Hz)> <channels (1/2)> [options]
//! <input> <output>`; run without arguments for the option list.

use std::io::Write;
use std::process::ExitCode;

fn main() -> std::io::Result<ExitCode> {
    let mut args = Vec::new();
    for arg in std::env::args_os() {
        match arg.into_string() {
            Ok(a) => args.push(a),
            Err(bad) => {
                writeln!(
                    std::io::stderr().lock(),
                    "opus_demo: argument is not valid UTF-8: {}",
                    bad.to_string_lossy()
                )?;
                return Ok(ExitCode::FAILURE);
            }
        }
    }
    let mut stdout = std::io::stdout().lock();
    let code =
        opusorus_tools::demo::opus_demo_main(&args, &mut stdout, &mut std::io::stderr().lock())?;
    stdout.flush()?;
    // The OS keeps the low 8 bits of the exit status, as for C's `return ret` from main.
    Ok(ExitCode::from(code as u8))
}
