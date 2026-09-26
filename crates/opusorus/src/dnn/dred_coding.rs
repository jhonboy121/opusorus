//! Port of dnn/dred_coding.c, dred_coding.h and dred_config.h (DRED configuration constants).
//!
//! The latent entropy coders live next to their C counterparts:
//! `dred_encode_latents` in [`super::dred_encoder`] and `dred_decode_latents` in
//! [`super::dred_decoder`] (both use the Laplace `_p0` coder of `celt::laplace`).

// ---- dred_config.h ----

/// `DRED_EXTENSION_ID`: the packet-extension ID carrying DRED (experimental until assigned).
pub const DRED_EXTENSION_ID: i32 = 126;
/// `DRED_EXPERIMENTAL_VERSION`.
pub const DRED_EXPERIMENTAL_VERSION: i32 = 12;
/// `DRED_EXPERIMENTAL_BYTES`.
pub const DRED_EXPERIMENTAL_BYTES: i32 = 2;
/// `DRED_MIN_BYTES`.
pub const DRED_MIN_BYTES: i32 = 8;
/// `DRED_SILK_ENCODER_DELAY` (`79+12-80`).
pub const DRED_SILK_ENCODER_DELAY: i32 = 79 + 12 - 80;
/// `DRED_FRAME_SIZE`.
pub const DRED_FRAME_SIZE: usize = 160;
/// `DRED_DFRAME_SIZE`.
pub const DRED_DFRAME_SIZE: usize = 2 * DRED_FRAME_SIZE;
/// `DRED_MAX_DATA_SIZE`.
pub const DRED_MAX_DATA_SIZE: usize = 1000;
/// `DRED_ENC_Q0`.
pub const DRED_ENC_Q0: i32 = 6;
/// `DRED_ENC_Q1`.
pub const DRED_ENC_Q1: i32 = 15;
/// `DRED_MAX_LATENTS`: covers 1.04 second so we can cover one second, after the lookahead.
pub const DRED_MAX_LATENTS: usize = 26;
/// `DRED_NUM_REDUNDANCY_FRAMES`.
pub const DRED_NUM_REDUNDANCY_FRAMES: usize = 2 * DRED_MAX_LATENTS;
/// `DRED_MAX_FRAMES`.
pub const DRED_MAX_FRAMES: usize = 4 * DRED_MAX_LATENTS;

/// `dQ_table` (static in compute_quantizer).
const DQ_TABLE: [i32; 8] = [0, 2, 3, 4, 6, 8, 12, 16];

/// Port of dnn/dred_coding.c:compute_quantizer: quantizer level of latent chunk `i` for the
/// initial quantizer `q0`, slope index `dq` (0..8) and maximum `qmax`.
#[must_use]
pub const fn compute_quantizer(q0: i32, dq: i32, qmax: i32, i: i32) -> i32 {
    let quant = q0 + (DQ_TABLE[dq as usize] * i + 8) / 16;
    if quant > qmax { qmax } else { quant }
}
