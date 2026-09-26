//! `lossgen_demo`: port of libopus `dnn/lossgen_demo.c` (feature `lossgen`, built by upstream
//! with `--enable-lossgen`), which prints a packet loss pattern from the generative loss model.
//!
//! ```text
//! lossgen_demo <percent_loss> <nb packets>
//! ```
//!
//! Prints one line per packet, `1` (lost) or `0`, exactly like the C tool: the model draws from
//! C `rand()` without seeding it, reproduced by [`GlibcRand`] (glibc hosts).

use std::io::{self, Write};

use opusorus::lossgen::{LossGenState, sample_loss};

use crate::compare::{EXIT_SUCCESS, argv0, c_atof, c_atol};
use opusorus::glibc_rand::GlibcRand;

/// `main` of `lossgen_demo.c`: returns the exit status (0, or 1 on a usage error).
///
/// # Errors
/// I/O errors writing to `stdout` / `stderr`.
pub fn lossgen_demo_main(
    args: &[String],
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> io::Result<i32> {
    if args.len() != 3 {
        writeln!(stderr, "usage: {} <percent_loss> <nb packets>", argv0(args))?;
        return Ok(1);
    }
    let mut st = LossGenState::lossgen_init();
    // `float percent = atof(argv[1])`, `long num_packets = atol(argv[2])`.
    let percent = c_atof(&args[1]) as f32;
    let num_packets = c_atol(&args[2]);
    let mut rng = GlibcRand::default();
    let mut out = io::BufWriter::new(stdout);
    for _ in 0..num_packets.max(0) {
        let lost = sample_loss(&mut st, percent * 0.01f32, &mut || rng.next_value());
        writeln!(out, "{lost}")?;
    }
    out.flush()?;
    Ok(EXIT_SUCCESS)
}
