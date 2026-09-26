//! `lossgen_demo`: port of libopus `dnn/lossgen_demo.c` (packet loss pattern from the
//! generative loss model; feature `lossgen`).
//!
//! Usage: `lossgen_demo <percent_loss> <nb packets>`.

use std::io::Write;
use std::process::ExitCode;

fn main() -> std::io::Result<ExitCode> {
    let args: Vec<String> = std::env::args().collect();
    let mut stdout = std::io::stdout().lock();
    let code = opusorus_tools::lossgen_demo::lossgen_demo_main(
        &args,
        &mut stdout,
        &mut std::io::stderr().lock(),
    )?;
    stdout.flush()?;
    // The OS keeps the low 8 bits of the exit status.
    Ok(ExitCode::from(code as u8))
}
