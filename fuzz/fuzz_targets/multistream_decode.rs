//! Multistream decoding, differential against C libopus: arbitrary (valid or invalid) layouts
//! — channel count, stream / coupled-stream counts, mapping (including silent 255 entries) —
//! then the decoder operation stream of `opusorus_fuzz::dec` with multistream packets.
//! Creation errors, return codes, every output sample (bit-exact), GET CTLs and the state of
//! every stream decoder must match.
//!
//! Input: `fs_sel:u8 channels:u8 streams:u8 coupled:u8 mapping[channels]` then the operation
//! stream. Counts below 248 are reduced modulo 25 (small layouts); 248..=255 are taken raw.

#![no_main]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "fuzz harness: a failed check must panic so libFuzzer records the input"
)]

use libfuzzer_sys::fuzz_target;
use opusorus::MsDecoder;
use opusorus_fuzz::dec::{self, CMs, check_state, diff_step};
use opusorus_fuzz::{Reader, code, pick_fs};
use opusorus_oracle::opus_decoder as od;

fn count(b: u8) -> i32 {
    if b >= 248 {
        i32::from(b)
    } else {
        i32::from(b % 25)
    }
}

fuzz_target!(|data: &[u8]| {
    let mut r = Reader::new(data);
    let fs = pick_fs(r.u8());
    let channels = count(r.u8());
    let streams = count(r.u8());
    let coupled = count(r.u8());
    let mut mapping = r.bytes(channels as usize).to_vec();
    mapping.resize(channels as usize, 0);
    let rd = code(MsDecoder::new(fs, channels, streams, coupled, &mapping));
    let cd = od::MsDec::new(fs, channels, streams, coupled, &mapping);
    let (mut rd, mut cd) = match (rd, cd) {
        (Ok(a), Ok(b)) => (a, CMs { d: b, streams }),
        (Err(a), Err(b)) => {
            assert_eq!(
                a, b,
                "create error ({channels}, {streams}, {coupled}, {mapping:?})"
            );
            return;
        }
        (a, b) => panic!(
            "create mismatch ({channels}, {streams}, {coupled}, {mapping:?}): {:?} vs {:?}",
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
