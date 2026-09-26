//! Tables of the `differential_custom` target (Opus Custom API), shared with the seed-corpus
//! generator. The input format is documented in the target.

use opusorus::celt::celt::{
    CELT_SET_CHANNELS_REQUEST, CELT_SET_INPUT_CLIPPING_REQUEST, CELT_SET_PREDICTION_REQUEST,
    CELT_SET_SIGNALLING_REQUEST, OPUS_SET_LFE_REQUEST,
};
use opusorus::decoder::{
    OPUS_RESET_STATE, OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
};
use opusorus::encoder::request::{
    OPUS_SET_BITRATE_REQUEST, OPUS_SET_LSB_DEPTH_REQUEST, OPUS_SET_PACKET_LOSS_PERC_REQUEST,
    OPUS_SET_QEXT_REQUEST, OPUS_SET_VBR_CONSTRAINT_REQUEST, OPUS_SET_VBR_REQUEST,
};

/// Operation tags (`tag % 8`).
pub mod tag {
    /// Encode a frame (and decode the packet).
    pub const ENCODE: u8 = 0;
    /// Decode an arbitrary packet.
    pub const DECODE: u8 = 3;
    /// PLC.
    pub const PLC: u8 = 4;
    /// Encoder CTL.
    pub const ENC_CTL: u8 = 5;
    /// Decoder CTL.
    pub const DEC_CTL: u8 = 6;
    /// Start / end bands.
    pub const BANDS: u8 = 7;
}

/// Index of `v` in `table` (seed generator).
#[must_use]
pub fn index_of(table: &[i32], v: i32) -> u8 {
    table.iter().position(|&x| x == v).expect("listed value") as u8
}

/// Index of request `req` in a SET table (seed generator).
#[must_use]
pub fn set_index(table: &[(i32, &[i32])], req: i32) -> u8 {
    table
        .iter()
        .position(|&(r, _)| r == req)
        .expect("listed request") as u8
}

/// Mode sampling rates (selector < 0xF0); 0xF0.. pick odd values.
pub const FS: [i32; 12] = [
    48000, 44100, 32000, 24000, 16000, 8000, 22050, 12000, 11025, 96000, 88200, 64000,
];
/// Odd mode sampling rates (selector >= 0xF0).
pub const FS_ODD: [i32; 4] = [7999, 96001, 0, 47999];
/// Mode frame sizes (selector < 0xF0); 0xF0.. is `2 * sel_low + 1` (odd: rejected).
pub const FRAME: [i32; 20] = [
    960, 480, 240, 120, 882, 1024, 640, 256, 400, 320, 720, 512, 2048, 1920, 1440, 40, 44, 1000,
    160, 80,
];
/// Raw frame sizes for `fsel >= 0x80`.
pub const RAW_FRAME: [i32; 12] = [1, 2, 39, 40, 41, 100, 120, 240, 480, 960, 1920, 4096];

/// `OPUS_UNIMPLEMENTED`.
pub const UNIMPLEMENTED: i32 = -5;

/// Encoder SET requests and their seed values.
pub const ENC_SETS: [(i32, &[i32]); 15] = [
    (OPUS_SET_COMPLEXITY_REQUEST, &[0, 1, 5, 9, 10, 11, -1]),
    (
        OPUS_SET_BITRATE_REQUEST,
        &[
            -1, 6000, 16000, 32000, 64000, 128000, 256000, 510000, 500, 501, 0,
        ],
    ),
    (OPUS_SET_VBR_REQUEST, &[1, 0, 2]),
    (OPUS_SET_VBR_CONSTRAINT_REQUEST, &[1, 0]),
    (CELT_SET_PREDICTION_REQUEST, &[0, 1, 2, 3, -1]),
    (OPUS_SET_PACKET_LOSS_PERC_REQUEST, &[0, 10, 50, 100, 101]),
    (CELT_SET_CHANNELS_REQUEST, &[1, 2, 0, 3]),
    (OPUS_SET_LSB_DEPTH_REQUEST, &[16, 24, 8, 7, 25]),
    (OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, &[1, 0, 2]),
    (CELT_SET_SIGNALLING_REQUEST, &[0, 1]),
    (CELT_SET_INPUT_CLIPPING_REQUEST, &[0, 1]),
    (OPUS_SET_LFE_REQUEST, &[1, 0]),
    (OPUS_SET_QEXT_REQUEST, &[1, 0, 2]),
    (OPUS_RESET_STATE, &[0]),
    (9999, &[1]),
];

/// Decoder SET requests (numeric `ctl_set` on both sides).
pub const DEC_SETS: [(i32, &[i32]); 6] = [
    (OPUS_SET_COMPLEXITY_REQUEST, &[0, 5, 10, 11, -1]),
    (CELT_SET_CHANNELS_REQUEST, &[1, 2, 0, 3]),
    (CELT_SET_SIGNALLING_REQUEST, &[0, 1]),
    (OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, &[1, 0, 2]),
    (OPUS_RESET_STATE, &[0]),
    (OPUS_SET_BITRATE_REQUEST, &[64000]),
];
