//! Projection (ambisonics, mapping family 3) decoding, differential against C libopus:
//! arbitrary channel / stream counts and demixing matrices (any size and content), then the
//! decoder operation stream of `opusorus_fuzz::dec` with multistream packets. Creation errors,
//! return codes, every output sample (bit-exact, after the demixing matrix), GET CTLs and the
//! state of every stream decoder must match.
//!
//! Input: `fs_sel:u8 channels:u8 streams:u8 coupled:u8 len:u16 matrix[len]` then the operation
//! stream. Counts below 248 are reduced modulo 40 (up to fifth-order ambisonics with
//! non-diegetic stereo = 38 channels); 248..=255 are taken raw.

#![no_main]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "fuzz harness: a failed check must panic so libFuzzer records the input"
)]

use libfuzzer_sys::fuzz_target;
use opusorus::ProjectionDecoder;
use opusorus_fuzz::dec::{self, CProj, check_state, diff_step};
use opusorus_fuzz::{Reader, code, pick_fs};
use opusorus_oracle::opus_decoder as od;

fn count(b: u8) -> i32 {
    if b >= 248 {
        i32::from(b)
    } else {
        i32::from(b % 40)
    }
}

fuzz_target!(|data: &[u8]| {
    let mut r = Reader::new(data);
    let fs = pick_fs(r.u8());
    let channels = count(r.u8());
    let streams = count(r.u8());
    let coupled = count(r.u8());
    let matrix = r.blob();
    let rd = code(ProjectionDecoder::new(
        fs, channels, streams, coupled, matrix,
    ));
    let cd = od::ProjDec::new(fs, channels, streams, coupled, matrix);
    let (mut rd, mut cd) = match (rd, cd) {
        (Ok(a), Ok(b)) => (a, CProj { d: b, streams }),
        (Err(a), Err(b)) => {
            assert_eq!(a, b, "create error ({channels}, {streams}, {coupled})");
            return;
        }
        (a, b) => panic!(
            "create mismatch ({channels}, {streams}, {coupled}, matrix {} bytes): {:?} vs {:?}",
            matrix.len(),
            a.err(),
            b.err()
        ),
    };
    check_state(&mut rd, &mut cd, "create");
    let mut i = 0;
    while let Some(op) = dec::read_op(&mut r) {
        diff_step(&mut rd, &mut cd, &op, fs, i);
        i += 1;
    }
});
