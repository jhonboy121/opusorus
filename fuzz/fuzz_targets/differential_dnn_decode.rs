//! Differential decoding with the DNN decoder features (`deep-plc`, optionally `osce`): an
//! opusorus `Decoder` with the oracle's weights loaded (the blob the oracle serializes from its
//! compiled-in model tables, `set_dnn_blob`) against a C `OpusDecoder` of the DNN oracle. The
//! operation stream of `opusorus_fuzz::dec` (packets, PLC, FEC, frame sizes, output formats,
//! CTLs including complexity and `OPUS_SET_OSCE_BWE`, resets) runs on both; return codes,
//! every output sample (bit-exact), every GET CTL and the DNN decoder state (LPCNet PLC
//! counters, FEC queue, `pcm` / `features` history, OSCE method / BWE mode) must match after
//! every operation.
//!
//! Deep PLC conceals lost frames at complexity >= 5; OSCE enhances SILK frames with LACE at
//! complexity 6 and NoLACE at >= 7, and with BWE on extends 16 kHz SILK to 48 kHz.
//!
//! Input: `fs_sel:u8 ch_sel:u8 dnn:u8` then the operation stream of `opusorus_fuzz::dec`.
//! `dnn`: bits 0-2 the initial complexity (5, 6, 7, 10, 8, 9, 4, 0), bit 3 `OPUS_SET_OSCE_BWE`.

#![no_main]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "fuzz harness: a failed check must panic so libFuzzer records the input"
)]

use libfuzzer_sys::fuzz_target;
use opusorus::decoder::{OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_OSCE_BWE_REQUEST};
use opusorus_fuzz::dec::{self, DecApi, check_state, diff_step};
use opusorus_fuzz::dnn::{CDnn, RustDnn};
use opusorus_fuzz::{Reader, pick_fs};
use opusorus_oracle::dnn_integration as di;

/// Initial complexities (selector bits 0-2 of the `dnn` byte).
const COMPLEXITY: [i32; 8] = [5, 6, 7, 10, 8, 9, 4, 0];

fuzz_target!(|data: &[u8]| {
    let mut r = Reader::new(data);
    let fs = pick_fs(r.u8());
    let channels = 1 + i32::from(r.u8() & 1);
    let dnn = r.u8();
    let (mut rd, mut cd) = match (RustDnn::new(fs, channels), di::Dec::new(fs, channels)) {
        (Ok(a), Ok(b)) => (a, CDnn(b)),
        (a, b) => panic!("create mismatch: {:?} vs {:?}", a.err(), b.err()),
    };
    let init = [
        (
            OPUS_SET_COMPLEXITY_REQUEST,
            COMPLEXITY[usize::from(dnn & 7)],
        ),
        (OPUS_SET_OSCE_BWE_REQUEST, i32::from((dnn >> 3) & 1)),
    ];
    for (req, v) in init {
        assert_eq!(
            rd.ctl_set(req, v),
            cd.ctl_set(req, v),
            "init ctl_set({req}, {v})"
        );
    }
    check_state(&mut rd, &mut cd, "init");
    let mut i = 0;
    while let Some(op) = dec::read_op(&mut r) {
        diff_step(&mut rd, &mut cd, &op, fs, i);
        i += 1;
    }
});
