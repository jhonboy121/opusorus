//! Port of dnn/dred_decoder.c and dred_decoder.h: the `OpusDRED` payload state and the DRED
//! entropy decoder (`dred_ec_decode`).
//!
//! The public `opus_dred_*` API (OpusDREDDecoder, `opus_dred_parse`, `opus_dred_process`,
//! `dred_find_payload`, ...) lives in src/opus_decoder.c and is integrated by the Opus decoder
//! unit on top of:
//! * [`OpusDred`] (all C fields public) and [`dred_ec_decode`] (payload → latents, stage 1),
//! * [`super::dred_rdovae::dred_rdovae_decode_all`] (latents → `fec_features`, stage 2) with a
//!   model from [`super::dred_rdovae::dred_rdovae_dec_load_model`].

use crate::celt::entdec::EcDec;
use crate::celt::laplace::ec_laplace_decode_p0;

use super::dred_coding::{DRED_NUM_REDUNDANCY_FRAMES, compute_quantizer};
use super::dred_rdovae::{
    DRED_LATENT_DIM, DRED_LATENT_P0_Q8, DRED_LATENT_QUANT_SCALES_Q8, DRED_LATENT_R_Q8,
    DRED_NUM_FEATURES, DRED_STATE_DIM, DRED_STATE_P0_Q8, DRED_STATE_QUANT_SCALES_Q8,
    DRED_STATE_R_Q8,
};

/// Size of [`OpusDred::fec_features`].
pub const DRED_FEC_FEATURES_SIZE: usize = 2 * DRED_NUM_REDUNDANCY_FRAMES * DRED_NUM_FEATURES;
/// Size of [`OpusDred::latents`].
pub const DRED_LATENTS_SIZE: usize = (DRED_NUM_REDUNDANCY_FRAMES / 2) * (DRED_LATENT_DIM + 1);

/// `struct OpusDRED` (dred_decoder.h). `Default`/[`OpusDred::new`] is a zeroed struct.
#[derive(Debug, Clone, PartialEq)]
pub struct OpusDred {
    pub fec_features: [f32; DRED_FEC_FEATURES_SIZE],
    pub state: [f32; DRED_STATE_DIM],
    pub latents: [f32; DRED_LATENTS_SIZE],
    pub nb_latents: i32,
    pub process_stage: i32,
    pub dred_offset: i32,
}

impl Default for OpusDred {
    fn default() -> Self {
        Self::new()
    }
}

impl OpusDred {
    /// A zeroed `OpusDRED`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            fec_features: [0.0; DRED_FEC_FEATURES_SIZE],
            state: [0.0; DRED_STATE_DIM],
            latents: [0.0; DRED_LATENTS_SIZE],
            nb_latents: 0,
            process_stage: 0,
            dred_offset: 0,
        }
    }
}

/// Port of dnn/dred_decoder.c:dred_decode_latents: decodes `dim` quantized values into `x`.
pub fn dred_decode_latents(
    dec: &mut EcDec<'_>,
    x: &mut [f32],
    scale: &[u8],
    r: &[u8],
    p0: &[u8],
    dim: usize,
) {
    let x = &mut x[..dim];
    let (scale, r, p0) = (&scale[..dim], &r[..dim], &p0[..dim]);
    for i in 0..dim {
        let q = if r[i] == 0 || p0[i] == 255 {
            0
        } else {
            ec_laplace_decode_p0(dec, u16::from(p0[i]) << 7, u16::from(r[i]) << 7)
        };
        let s = if scale[i] == 0 { 1 } else { scale[i] };
        x[i] = q as f32 * 256.0f32 / f32::from(s);
    }
}

/// Port of dnn/dred_decoder.c:dred_ec_decode.
///
/// Decodes the DRED payload `bytes` (C `num_bytes = bytes.len()`) into `dec` (`state`,
/// `latents`, `nb_latents`, `dred_offset`; sets `process_stage = 1`) and returns the number of
/// decoded latents. Any input is accepted (garbage decodes to something, as in C).
pub fn dred_ec_decode(
    dec: &mut OpusDred,
    bytes: &[u8],
    min_feature_frames: i32,
    dred_frame_offset: i32,
) -> i32 {
    let num_bytes = bytes.len() as i32;
    // since features are decoded in quadruples, it makes no sense to go with an uneven number
    // of redundancy frames
    const _: () = assert!(DRED_NUM_REDUNDANCY_FRAMES.is_multiple_of(2));

    // decode initial state and initialize RDOVAE decoder
    let mut ec = EcDec::new(bytes);
    let q0 = ec.dec_uint(16) as i32;
    let dq = ec.dec_uint(8) as i32;
    let extra_offset: u32 = if ec.dec_uint(2) != 0 {
        32 * ec.dec_uint(256)
    } else {
        0
    };
    // Compute total offset, including DRED position in a multiframe packet. C evaluates this
    // in unsigned arithmetic (ec_dec_uint returns opus_uint32) and stores it in an int.
    dec.dred_offset = 16u32
        .wrapping_sub(ec.dec_uint(32))
        .wrapping_sub(extra_offset)
        .wrapping_add(dred_frame_offset as u32) as i32;
    let mut qmax = 15;
    if q0 < 14 && dq > 0 {
        // The distribution for the dQmax symbol is split evenly between zero (which implies
        // qmax == 15) and larger values, with the probability of all larger values being
        // uniform. This is equivalent to coding 1 bit to decide if the maximum is less than 15
        // followed by a uint to decide the actual value if it is less than 15, but combined
        // into a single symbol.
        let nvals = 15 - (q0 + 1);
        let ft = (2 * nvals) as u32;
        let s = ec.decode(ft) as i32;
        if s >= nvals {
            qmax = q0 + (s - nvals) + 1;
            ec.update(s as u32, (s + 1) as u32, ft);
        } else {
            ec.update(0, nvals as u32, ft);
        }
    }
    let state_qoffset = q0 as usize * DRED_STATE_DIM;
    dred_decode_latents(
        &mut ec,
        &mut dec.state,
        &DRED_STATE_QUANT_SCALES_Q8[state_qoffset..],
        &DRED_STATE_R_Q8[state_qoffset..],
        &DRED_STATE_P0_Q8[state_qoffset..],
        DRED_STATE_DIM,
    );

    // decode newest to oldest and store oldest to newest
    let limit = (DRED_NUM_REDUNDANCY_FRAMES as i32).min((min_feature_frames + 1) / 2);
    let mut i: i32 = 0;
    while i < limit {
        // FIXME: Figure out how to avoid missing a last frame that would take up < 8 bits.
        if 8 * num_bytes - ec.tell() <= 7 {
            break;
        }
        let q_level = compute_quantizer(q0, dq, qmax, i / 2);
        let offset = q_level as usize * DRED_LATENT_DIM;
        let base = (i / 2) as usize * (DRED_LATENT_DIM + 1);
        dred_decode_latents(
            &mut ec,
            &mut dec.latents[base..],
            &DRED_LATENT_QUANT_SCALES_Q8[offset..],
            &DRED_LATENT_R_Q8[offset..],
            &DRED_LATENT_P0_Q8[offset..],
            DRED_LATENT_DIM,
        );
        dec.latents[base + DRED_LATENT_DIM] = (f64::from(q_level) * 0.125 - 1.0) as f32;
        // C also computes an unused `offset = 2 * i * DRED_NUM_FEATURES` here.
        i += 2;
    }
    dec.process_stage = 1;
    dec.nb_latents = i / 2;
    i / 2
}
