//! Port of libopus `tests/test_opus_padding.c`: checks for overflow in reading the padding
//! length (<http://lists.xiph.org/pipermail/opus/2012-November/001834.html>).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failed call is a test failure"
)]

use opusorus::{Decoder, Error};

const PACKETSIZE: usize = 16_909_318;
const CHANNELS: i32 = 2;
const FRAMESIZE: usize = 5760;

/// Port of `test_overflow`.
#[test]
fn padding_overflow() {
    let mut input = vec![0xffu8; PACKETSIZE];
    let mut out = vec![0i16; FRAMESIZE * CHANNELS as usize];
    input[0] = 0xff;
    input[1] = 0x41;
    input[PACKETSIZE - 1] = 0x0b;

    let mut decoder = Decoder::new(48000, CHANNELS).unwrap();
    let result = decoder.decode(Some(&input), &mut out, FRAMESIZE, false);
    assert_eq!(result, Err(Error::InvalidPacket));
}
