//! Differential decoding: the same operation stream (arbitrary packets, PLC, FEC, frame sizes,
//! output formats, gain / complexity / phase-inversion / ignore-extensions CTLs, resets) is
//! applied to an opusorus `Decoder` and a C libopus `OpusDecoder`. Return codes, every output
//! sample (bit-exact, plus the untouched tail of the buffer), all GET CTLs (including the final
//! range) and the Opus-level decoder state must be identical after every operation.
//!
//! Input: `fs_sel:u8 ch_sel:u8` then the operation stream of `opusorus_fuzz::dec`.

#![no_main]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "fuzz harness: a failed check must panic so libFuzzer records the input"
)]

use libfuzzer_sys::fuzz_target;
use opusorus::Decoder;
use opusorus_fuzz::dec::{self, diff_step};
use opusorus_fuzz::{Reader, code, pick_fs};
use opusorus_oracle::opus_decoder as od;

fuzz_target!(|data: &[u8]| {
    let mut r = Reader::new(data);
    let fs = pick_fs(r.u8());
    let channels = 1 + i32::from(r.u8() & 1);
    let (mut rd, mut cd) = match (code(Decoder::new(fs, channels)), od::Dec::new(fs, channels)) {
        (Ok(a), Ok(b)) => (a, b),
        (a, b) => panic!("create mismatch: {:?} vs {:?}", a.err(), b.err()),
    };
    let mut i = 0;
    while let Some(op) = dec::read_op(&mut r) {
        diff_step(&mut rd, &mut cd, &op, fs, i);
        i += 1;
    }
});
