//! Port of the SILK entropy-coding helpers: `silk/code_signs.c`, `silk/shell_coder.c`,
//! `silk/gain_quant.c`, `silk/decode_pulses.c`, `silk/encode_pulses.c`, `silk/decode_pitch.c`,
//! `silk/stereo_decode_pred.c` and `silk/stereo_encode_pred.c`.

#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]

use crate::celt::entdec::EcDec;
use crate::celt::entenc::EcEnc;
use crate::silk::define::{
    LOG2_SHELL_CODEC_FRAME_LENGTH, MAX_DELTA_GAIN_QUANT, MAX_NB_SHELL_BLOCKS, MAX_QGAIN_DB,
    MIN_DELTA_GAIN_QUANT, MIN_QGAIN_DB, N_LEVELS_QGAIN, N_RATE_LEVELS, SHELL_CODEC_FRAME_LENGTH,
    SILK_MAX_PULSES, STEREO_QUANT_SUB_STEPS,
};
use crate::silk::macros::{
    silk_abs, silk_add_lshift, silk_add_lshift32, silk_div32_16, silk_fix_const, silk_limit,
    silk_limit_int, silk_lshift, silk_max_int, silk_min, silk_min_32, silk_min_int, silk_rshift,
    silk_smlabb, silk_smulbb, silk_smulwb,
};
use crate::silk::sigproc::{silk_lin2log, silk_log2lin};
use crate::silk::tables::{
    PE_MAX_LAG_MS, PE_MAX_NB_SUBFR, PE_MIN_LAG_MS, PE_NB_CBKS_STAGE2_10MS, PE_NB_CBKS_STAGE2_EXT,
    PE_NB_CBKS_STAGE3_10MS, PE_NB_CBKS_STAGE3_MAX, SILK_CB_LAGS_STAGE2, SILK_CB_LAGS_STAGE2_10_MS,
    SILK_CB_LAGS_STAGE3, SILK_CB_LAGS_STAGE3_10_MS, SILK_LSB_ICDF, SILK_MAX_PULSES_TABLE,
    SILK_PULSES_PER_BLOCK_BITS_Q5, SILK_PULSES_PER_BLOCK_ICDF, SILK_RATE_LEVELS_BITS_Q5,
    SILK_RATE_LEVELS_ICDF, SILK_SHELL_CODE_TABLE_OFFSETS, SILK_SHELL_CODE_TABLE0,
    SILK_SHELL_CODE_TABLE1, SILK_SHELL_CODE_TABLE2, SILK_SHELL_CODE_TABLE3, SILK_SIGN_ICDF,
    SILK_STEREO_ONLY_CODE_MID_ICDF, SILK_STEREO_PRED_JOINT_ICDF, SILK_STEREO_PRED_QUANT_Q13,
    SILK_UNIFORM3_ICDF, SILK_UNIFORM5_ICDF,
};

const SCFL: usize = SHELL_CODEC_FRAME_LENGTH as usize;
const MNSB: usize = MAX_NB_SHELL_BLOCKS as usize;

// ---------------------------------------------------------------------------------------------
// code_signs.c
// ---------------------------------------------------------------------------------------------

const_unless_fixed_debug! {
/// `silk_enc_map` (silk/code_signs.c): shifting avoids if-statement.
#[inline(always)]
const fn silk_enc_map(a: i32) -> i32 {
    silk_rshift(a, 15) + 1
}
}

const_unless_fixed_debug! {
/// `silk_dec_map` (silk/code_signs.c).
#[inline(always)]
const fn silk_dec_map(a: i32) -> i32 {
    silk_lshift(a, 1) - 1
}
}

/// Port of silk/code_signs.c:silk_encode_signs — encodes signs of excitation.
pub fn silk_encode_signs(
    ps_range_enc: &mut EcEnc<'_>,
    pulses: &[i8],
    length: i32,
    signal_type: i32,
    quant_offset_type: i32,
    sum_pulses: &[i32],
) {
    let mut icdf = [0u8; 2];
    icdf[1] = 0;
    let i = silk_smulbb(7, silk_add_lshift(quant_offset_type, signal_type, 1));
    let icdf_ptr = &SILK_SIGN_ICDF[i as usize..];
    let length = silk_rshift(
        length + SHELL_CODEC_FRAME_LENGTH / 2,
        LOG2_SHELL_CODEC_FRAME_LENGTH,
    );
    for i in 0..length as usize {
        let q_ptr = &pulses[i * SCFL..(i + 1) * SCFL];
        let p = sum_pulses[i];
        if p > 0 {
            icdf[0] = icdf_ptr[silk_min(p & 0x1F, 6) as usize];
            for j in 0..SCFL {
                if q_ptr[j] != 0 {
                    ps_range_enc.enc_icdf(silk_enc_map(q_ptr[j] as i32) as usize, &icdf, 8);
                }
            }
        }
    }
}

/// Port of silk/code_signs.c:silk_decode_signs — decodes signs of excitation.
pub fn silk_decode_signs(
    ps_range_dec: &mut EcDec<'_>,
    pulses: &mut [i16],
    length: i32,
    signal_type: i32,
    quant_offset_type: i32,
    sum_pulses: &[i32],
) {
    let mut icdf = [0u8; 2];
    icdf[1] = 0;
    let i = silk_smulbb(7, silk_add_lshift(quant_offset_type, signal_type, 1));
    let icdf_ptr = &SILK_SIGN_ICDF[i as usize..];
    let length = silk_rshift(
        length + SHELL_CODEC_FRAME_LENGTH / 2,
        LOG2_SHELL_CODEC_FRAME_LENGTH,
    );
    for i in 0..length as usize {
        let q_ptr = &mut pulses[i * SCFL..(i + 1) * SCFL];
        let p = sum_pulses[i];
        if p > 0 {
            icdf[0] = icdf_ptr[silk_min(p & 0x1F, 6) as usize];
            for j in 0..SCFL {
                if q_ptr[j] > 0 {
                    // attach sign
                    // implementation with shift, subtraction, multiplication
                    q_ptr[j] = (q_ptr[j] as i32
                        * silk_dec_map(ps_range_dec.dec_icdf(&icdf, 8) as i32))
                        as i16;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// shell_coder.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/shell_coder.c:combine_pulses (static helper). `input` has `2 * len` entries.
#[inline(always)]
fn combine_pulses(out: &mut [i32], input: &[i32], len: usize) {
    for k in 0..len {
        out[k] = input[2 * k] + input[2 * k + 1];
    }
}

/// Port of silk/shell_coder.c:encode_split (static helper).
#[inline(always)]
fn encode_split(ps_range_enc: &mut EcEnc<'_>, p_child1: i32, p: i32, shell_table: &[u8]) {
    if p > 0 {
        ps_range_enc.enc_icdf(
            p_child1 as usize,
            &shell_table[SILK_SHELL_CODE_TABLE_OFFSETS[p as usize] as usize..],
            8,
        );
    }
}

/// Port of silk/shell_coder.c:decode_split (static helper): writes `p_child1` / `p_child2` to
/// `dst[ia]` / `dst[ia + 1]` (always adjacent in C).
#[inline(always)]
fn decode_split_into(
    dst: &mut [i16],
    ia: usize,
    ps_range_dec: &mut EcDec<'_>,
    p: i32,
    shell_table: &[u8],
) {
    if p > 0 {
        dst[ia] = ps_range_dec.dec_icdf(
            &shell_table[SILK_SHELL_CODE_TABLE_OFFSETS[p as usize] as usize..],
            8,
        ) as i16;
        dst[ia + 1] = (p - dst[ia] as i32) as i16;
    } else {
        dst[ia] = 0;
        dst[ia + 1] = 0;
    }
}

/// Port of silk/shell_coder.c:silk_shell_encoder — shell encoder, operates on one shell code
/// frame of 16 pulses (`pulses0`: nonnegative pulse amplitudes).
pub fn silk_shell_encoder(ps_range_enc: &mut EcEnc<'_>, pulses0: &[i32]) {
    let mut pulses1 = [0i32; 8];
    let mut pulses2 = [0i32; 4];
    let mut pulses3 = [0i32; 2];
    let mut pulses4 = [0i32; 1];

    // this function operates on one shell code frame of 16 pulses
    const { assert!(SHELL_CODEC_FRAME_LENGTH == 16) };

    // tree representation per pulse-subframe
    combine_pulses(&mut pulses1, pulses0, 8);
    combine_pulses(&mut pulses2, &pulses1, 4);
    combine_pulses(&mut pulses3, &pulses2, 2);
    combine_pulses(&mut pulses4, &pulses3, 1);

    let e = ps_range_enc;
    encode_split(e, pulses3[0], pulses4[0], &SILK_SHELL_CODE_TABLE3);

    encode_split(e, pulses2[0], pulses3[0], &SILK_SHELL_CODE_TABLE2);

    encode_split(e, pulses1[0], pulses2[0], &SILK_SHELL_CODE_TABLE1);
    encode_split(e, pulses0[0], pulses1[0], &SILK_SHELL_CODE_TABLE0);
    encode_split(e, pulses0[2], pulses1[1], &SILK_SHELL_CODE_TABLE0);

    encode_split(e, pulses1[2], pulses2[1], &SILK_SHELL_CODE_TABLE1);
    encode_split(e, pulses0[4], pulses1[2], &SILK_SHELL_CODE_TABLE0);
    encode_split(e, pulses0[6], pulses1[3], &SILK_SHELL_CODE_TABLE0);

    encode_split(e, pulses2[2], pulses3[1], &SILK_SHELL_CODE_TABLE2);

    encode_split(e, pulses1[4], pulses2[2], &SILK_SHELL_CODE_TABLE1);
    encode_split(e, pulses0[8], pulses1[4], &SILK_SHELL_CODE_TABLE0);
    encode_split(e, pulses0[10], pulses1[5], &SILK_SHELL_CODE_TABLE0);

    encode_split(e, pulses1[6], pulses2[3], &SILK_SHELL_CODE_TABLE1);
    encode_split(e, pulses0[12], pulses1[6], &SILK_SHELL_CODE_TABLE0);
    encode_split(e, pulses0[14], pulses1[7], &SILK_SHELL_CODE_TABLE0);
}

/// Port of silk/shell_coder.c:silk_shell_decoder — shell decoder, operates on one shell code
/// frame of 16 pulses (`pulses4`: number of pulses per pulse-subframe).
pub fn silk_shell_decoder(pulses0: &mut [i16], ps_range_dec: &mut EcDec<'_>, pulses4: i32) {
    let mut pulses3 = [0i16; 2];
    let mut pulses2 = [0i16; 4];
    let mut pulses1 = [0i16; 8];

    // this function operates on one shell code frame of 16 pulses
    const { assert!(SHELL_CODEC_FRAME_LENGTH == 16) };

    let d = ps_range_dec;
    decode_split_into(&mut pulses3, 0, d, pulses4, &SILK_SHELL_CODE_TABLE3);

    decode_split_into(
        &mut pulses2,
        0,
        d,
        pulses3[0] as i32,
        &SILK_SHELL_CODE_TABLE2,
    );

    decode_split_into(
        &mut pulses1,
        0,
        d,
        pulses2[0] as i32,
        &SILK_SHELL_CODE_TABLE1,
    );
    decode_split_into(pulses0, 0, d, pulses1[0] as i32, &SILK_SHELL_CODE_TABLE0);
    decode_split_into(pulses0, 2, d, pulses1[1] as i32, &SILK_SHELL_CODE_TABLE0);

    decode_split_into(
        &mut pulses1,
        2,
        d,
        pulses2[1] as i32,
        &SILK_SHELL_CODE_TABLE1,
    );
    decode_split_into(pulses0, 4, d, pulses1[2] as i32, &SILK_SHELL_CODE_TABLE0);
    decode_split_into(pulses0, 6, d, pulses1[3] as i32, &SILK_SHELL_CODE_TABLE0);

    decode_split_into(
        &mut pulses2,
        2,
        d,
        pulses3[1] as i32,
        &SILK_SHELL_CODE_TABLE2,
    );

    decode_split_into(
        &mut pulses1,
        4,
        d,
        pulses2[2] as i32,
        &SILK_SHELL_CODE_TABLE1,
    );
    decode_split_into(pulses0, 8, d, pulses1[4] as i32, &SILK_SHELL_CODE_TABLE0);
    decode_split_into(pulses0, 10, d, pulses1[5] as i32, &SILK_SHELL_CODE_TABLE0);

    decode_split_into(
        &mut pulses1,
        6,
        d,
        pulses2[3] as i32,
        &SILK_SHELL_CODE_TABLE1,
    );
    decode_split_into(pulses0, 12, d, pulses1[6] as i32, &SILK_SHELL_CODE_TABLE0);
    decode_split_into(pulses0, 14, d, pulses1[7] as i32, &SILK_SHELL_CODE_TABLE0);
}

// ---------------------------------------------------------------------------------------------
// gain_quant.c
// ---------------------------------------------------------------------------------------------

/// `OFFSET` (silk/gain_quant.c).
const OFFSET: i32 = (MIN_QGAIN_DB * 128) / 6 + 16 * 128;
/// `SCALE_Q16` (silk/gain_quant.c).
const SCALE_Q16: i32 = (65536 * (N_LEVELS_QGAIN - 1)) / (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6);
/// `INV_SCALE_Q16` (silk/gain_quant.c).
const INV_SCALE_Q16: i32 =
    (65536 * (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6)) / (N_LEVELS_QGAIN - 1);

/// Port of silk/gain_quant.c:silk_gains_quant — gain scalar quantization with hysteresis,
/// uniform on log scale.
///
/// `ind` (gain indices) and `gain_q16` (gains, quantized on output) have `MAX_NB_SUBFR`
/// entries; `prev_ind` is the last index in the previous frame (updated). The first gain is
/// delta coded if `conditional` is 1.
pub fn silk_gains_quant(
    ind: &mut [i8],
    gain_q16: &mut [i32],
    prev_ind: &mut i8,
    conditional: i32,
    nb_subfr: usize,
) {
    // `ind[k]` and `*prev_ind` are `opus_int8` in C: every store truncates.
    for k in 0..nb_subfr {
        // Convert to log scale, scale, floor()
        ind[k] = silk_smulwb(SCALE_Q16, silk_lin2log(gain_q16[k]) - OFFSET) as i8;

        // Round towards previous quantized gain (hysteresis)
        if ind[k] < *prev_ind {
            ind[k] = (ind[k] as i32 + 1) as i8;
        }
        ind[k] = silk_limit_int(ind[k] as i32, 0, N_LEVELS_QGAIN - 1) as i8;

        // Compute delta indices and limit
        if k == 0 && conditional == 0 {
            // Full index
            ind[k] = silk_limit_int(
                ind[k] as i32,
                *prev_ind as i32 + MIN_DELTA_GAIN_QUANT,
                N_LEVELS_QGAIN - 1,
            ) as i8;
            *prev_ind = ind[k];
        } else {
            // Delta index
            ind[k] = (ind[k] as i32 - *prev_ind as i32) as i8;

            // Double the quantization step size for large gain increases, so that the max gain
            // level can be reached
            let double_step_size_threshold =
                2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + *prev_ind as i32;
            if ind[k] as i32 > double_step_size_threshold {
                ind[k] = (double_step_size_threshold
                    + silk_rshift(ind[k] as i32 - double_step_size_threshold + 1, 1))
                    as i8;
            }

            ind[k] =
                silk_limit_int(ind[k] as i32, MIN_DELTA_GAIN_QUANT, MAX_DELTA_GAIN_QUANT) as i8;

            // Accumulate deltas
            if ind[k] as i32 > double_step_size_threshold {
                *prev_ind = (*prev_ind as i32 + silk_lshift(ind[k] as i32, 1)
                    - double_step_size_threshold) as i8;
                *prev_ind = silk_min_int(*prev_ind as i32, N_LEVELS_QGAIN - 1) as i8;
            } else {
                *prev_ind = (*prev_ind as i32 + ind[k] as i32) as i8;
            }

            // Shift to make non-negative
            ind[k] = (ind[k] as i32 - MIN_DELTA_GAIN_QUANT) as i8;
        }

        // Scale and convert to linear scale
        gain_q16[k] = silk_log2lin(silk_min_32(
            silk_smulwb(INV_SCALE_Q16, *prev_ind as i32) + OFFSET,
            3967,
        )); // 3967 = 31 in Q7
    }
}

/// Port of silk/gain_quant.c:silk_gains_dequant — gains scalar dequantization, uniform on log
/// scale.
pub fn silk_gains_dequant(
    gain_q16: &mut [i32],
    ind: &[i8],
    prev_ind: &mut i8,
    conditional: i32,
    nb_subfr: usize,
) {
    for k in 0..nb_subfr {
        if k == 0 && conditional == 0 {
            // Gain index is not allowed to go down more than 16 steps (~21.8 dB)
            *prev_ind = silk_max_int(ind[k] as i32, *prev_ind as i32 - 16) as i8;
        } else {
            // Delta index
            let ind_tmp = ind[k] as i32 + MIN_DELTA_GAIN_QUANT;

            // Accumulate deltas
            let double_step_size_threshold =
                2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + *prev_ind as i32;
            if ind_tmp > double_step_size_threshold {
                *prev_ind =
                    (*prev_ind as i32 + silk_lshift(ind_tmp, 1) - double_step_size_threshold) as i8;
            } else {
                *prev_ind = (*prev_ind as i32 + ind_tmp) as i8;
            }
        }
        *prev_ind = silk_limit_int(*prev_ind as i32, 0, N_LEVELS_QGAIN - 1) as i8;

        // Scale and convert to linear scale
        gain_q16[k] = silk_log2lin(silk_min_32(
            silk_smulwb(INV_SCALE_Q16, *prev_ind as i32) + OFFSET,
            3967,
        )); // 3967 = 31 in Q7
    }
}

/// Port of silk/gain_quant.c:silk_gains_ID — compute unique identifier of gain indices vector.
#[must_use]
pub fn silk_gains_id(ind: &[i8], nb_subfr: usize) -> i32 {
    let mut gains_id: i32 = 0;
    for k in 0..nb_subfr {
        gains_id = silk_add_lshift32(ind[k] as i32, gains_id, 8);
    }
    gains_id
}

// ---------------------------------------------------------------------------------------------
// decode_pulses.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/decode_pulses.c:silk_decode_pulses — decode quantization indices of excitation.
///
/// `pulses` must hold `frame_length` rounded up to a multiple of `SHELL_CODEC_FRAME_LENGTH`.
pub fn silk_decode_pulses(
    ps_range_dec: &mut EcDec<'_>,
    pulses: &mut [i16],
    signal_type: i32,
    quant_offset_type: i32,
    frame_length: i32,
) {
    let mut sum_pulses = [0i32; MNSB];
    let mut n_lshifts = [0i32; MNSB];

    // Decode rate level
    let rate_level_index =
        ps_range_dec.dec_icdf(&SILK_RATE_LEVELS_ICDF[(signal_type >> 1) as usize], 8);

    // Calculate number of shell blocks
    const { assert!(1 << LOG2_SHELL_CODEC_FRAME_LENGTH == SHELL_CODEC_FRAME_LENGTH) };
    let mut iter = silk_rshift(frame_length, LOG2_SHELL_CODEC_FRAME_LENGTH) as usize;
    if (iter as i32) * SHELL_CODEC_FRAME_LENGTH < frame_length {
        debug_assert!(frame_length == 12 * 10); // Make sure only happens for 10 ms @ 12 kHz
        iter += 1;
    }

    // Sum-Weighted-Pulses Decoding
    let cdf_ptr = &SILK_PULSES_PER_BLOCK_ICDF[rate_level_index];
    for i in 0..iter {
        n_lshifts[i] = 0;
        sum_pulses[i] = ps_range_dec.dec_icdf(cdf_ptr, 8) as i32;

        // LSB indication
        while sum_pulses[i] == SILK_MAX_PULSES + 1 {
            n_lshifts[i] += 1;
            // When we've already got 10 LSBs, we shift the table to not allow
            // (SILK_MAX_PULSES + 1)
            let off = usize::from(n_lshifts[i] == 10);
            sum_pulses[i] = ps_range_dec.dec_icdf(
                &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS as usize - 1][off..],
                8,
            ) as i32;
        }
    }

    // Shell decoding
    for i in 0..iter {
        let blk = &mut pulses[i * SCFL..(i + 1) * SCFL];
        if sum_pulses[i] > 0 {
            silk_shell_decoder(blk, ps_range_dec, sum_pulses[i]);
        } else {
            blk.fill(0);
        }
    }

    // LSB Decoding
    for i in 0..iter {
        if n_lshifts[i] > 0 {
            let n_ls = n_lshifts[i];
            let pulses_ptr = &mut pulses[i * SCFL..(i + 1) * SCFL];
            for k in 0..SCFL {
                let mut abs_q = pulses_ptr[k] as i32;
                for _ in 0..n_ls {
                    abs_q = silk_lshift(abs_q, 1);
                    abs_q += ps_range_dec.dec_icdf(&SILK_LSB_ICDF, 8) as i32;
                }
                pulses_ptr[k] = abs_q as i16;
            }
            // Mark the number of pulses non-zero for sign decoding.
            sum_pulses[i] |= n_ls << 5;
        }
    }

    // Decode and add signs to pulse signal
    silk_decode_signs(
        ps_range_dec,
        pulses,
        frame_length,
        signal_type,
        quant_offset_type,
        &sum_pulses,
    );
}

// ---------------------------------------------------------------------------------------------
// encode_pulses.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/encode_pulses.c:combine_and_check (static helper). Returns 1 if a combined sum
/// exceeds `max_pulses`. `pulses_in` may alias `pulses_comb` in C (in-place halving); it is
/// passed by value here.
#[inline(always)]
fn combine_and_check(
    pulses_comb: &mut [i32],
    pulses_in: &[i32],
    max_pulses: i32,
    len: usize,
) -> i32 {
    for k in 0..len {
        let sum = pulses_in[2 * k] + pulses_in[2 * k + 1];
        if sum > max_pulses {
            return 1;
        }
        pulses_comb[k] = sum;
    }
    0
}

/// In-place form of [`combine_and_check`] (C calls it with `pulses_comb == pulses_in`; each
/// output `k` is written after inputs `2k`, `2k + 1` have been read, and `k <= 2k`).
#[inline(always)]
fn combine_and_check_inplace(pulses_comb: &mut [i32], max_pulses: i32, len: usize) -> i32 {
    for k in 0..len {
        let sum = pulses_comb[2 * k] + pulses_comb[2 * k + 1];
        if sum > max_pulses {
            return 1;
        }
        pulses_comb[k] = sum;
    }
    0
}

/// Port of silk/encode_pulses.c:silk_encode_pulses — encode quantization indices of excitation.
///
/// `pulses` must hold `frame_length` rounded up to a multiple of `SHELL_CODEC_FRAME_LENGTH`
/// (for 10 ms @ 12 kHz the padding is zeroed here, as in C).
pub fn silk_encode_pulses(
    ps_range_enc: &mut EcEnc<'_>,
    signal_type: i32,
    quant_offset_type: i32,
    pulses: &mut [i8],
    frame_length: i32,
) {
    let mut rate_level_index = 0usize;
    let mut pulses_comb = [0i32; 8];
    let mut abs_pulses = [0i32; MNSB * SCFL];
    let mut sum_pulses = [0i32; MNSB];
    let mut n_rshifts = [0i32; MNSB];

    // Prepare for shell coding
    // Calculate number of shell blocks
    const { assert!(1 << LOG2_SHELL_CODEC_FRAME_LENGTH == SHELL_CODEC_FRAME_LENGTH) };
    let mut iter = silk_rshift(frame_length, LOG2_SHELL_CODEC_FRAME_LENGTH) as usize;
    if (iter as i32) * SHELL_CODEC_FRAME_LENGTH < frame_length {
        debug_assert!(frame_length == 12 * 10); // Make sure only happens for 10 ms @ 12 kHz
        iter += 1;
        let fl = frame_length as usize;
        pulses[fl..fl + SCFL].fill(0);
    }

    // Take the absolute value of the pulses
    const { assert!(SHELL_CODEC_FRAME_LENGTH & 3 == 0) };
    for i in 0..iter * SCFL {
        abs_pulses[i] = silk_abs(pulses[i] as i32);
    }

    // Calc sum pulses per shell code frame
    for i in 0..iter {
        let abs_pulses_ptr = &mut abs_pulses[i * SCFL..(i + 1) * SCFL];
        n_rshifts[i] = 0;
        loop {
            // 1+1 -> 2
            let mut scale_down = combine_and_check(
                &mut pulses_comb,
                abs_pulses_ptr,
                SILK_MAX_PULSES_TABLE[0] as i32,
                8,
            );
            // 2+2 -> 4
            scale_down +=
                combine_and_check_inplace(&mut pulses_comb, SILK_MAX_PULSES_TABLE[1] as i32, 4);
            // 4+4 -> 8
            scale_down +=
                combine_and_check_inplace(&mut pulses_comb, SILK_MAX_PULSES_TABLE[2] as i32, 2);
            // 8+8 -> 16
            scale_down += combine_and_check(
                &mut sum_pulses[i..],
                &pulses_comb,
                SILK_MAX_PULSES_TABLE[3] as i32,
                1,
            );

            if scale_down != 0 {
                // We need to downscale the quantization signal
                n_rshifts[i] += 1;
                for k in 0..SCFL {
                    abs_pulses_ptr[k] = silk_rshift(abs_pulses_ptr[k], 1);
                }
            } else {
                // Jump out of while(1) loop and go to next shell coding frame
                break;
            }
        }
    }

    // Rate level
    // find rate level that leads to fewest bits for coding of pulses per block info
    let mut min_sum_bits_q5 = i32::MAX;
    for k in 0..N_RATE_LEVELS as usize - 1 {
        let n_bits_ptr = &SILK_PULSES_PER_BLOCK_BITS_Q5[k];
        let mut sum_bits_q5 = SILK_RATE_LEVELS_BITS_Q5[(signal_type >> 1) as usize][k] as i32;
        for i in 0..iter {
            if n_rshifts[i] > 0 {
                sum_bits_q5 += n_bits_ptr[SILK_MAX_PULSES as usize + 1] as i32;
            } else {
                sum_bits_q5 += n_bits_ptr[sum_pulses[i] as usize] as i32;
            }
        }
        if sum_bits_q5 < min_sum_bits_q5 {
            min_sum_bits_q5 = sum_bits_q5;
            rate_level_index = k;
        }
    }
    ps_range_enc.enc_icdf(
        rate_level_index,
        &SILK_RATE_LEVELS_ICDF[(signal_type >> 1) as usize],
        8,
    );

    // Sum-Weighted-Pulses Encoding
    let cdf_ptr = &SILK_PULSES_PER_BLOCK_ICDF[rate_level_index];
    let last_cdf = &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS as usize - 1];
    for i in 0..iter {
        if n_rshifts[i] == 0 {
            ps_range_enc.enc_icdf(sum_pulses[i] as usize, cdf_ptr, 8);
        } else {
            ps_range_enc.enc_icdf(SILK_MAX_PULSES as usize + 1, cdf_ptr, 8);
            for _ in 0..n_rshifts[i] - 1 {
                ps_range_enc.enc_icdf(SILK_MAX_PULSES as usize + 1, last_cdf, 8);
            }
            ps_range_enc.enc_icdf(sum_pulses[i] as usize, last_cdf, 8);
        }
    }

    // Shell Encoding
    for i in 0..iter {
        if sum_pulses[i] > 0 {
            silk_shell_encoder(ps_range_enc, &abs_pulses[i * SCFL..(i + 1) * SCFL]);
        }
    }

    // LSB Encoding
    for i in 0..iter {
        if n_rshifts[i] > 0 {
            let pulses_ptr = &pulses[i * SCFL..(i + 1) * SCFL];
            let n_ls = n_rshifts[i] - 1;
            for k in 0..SCFL {
                // C: `abs_q = (opus_int8)silk_abs( pulses_ptr[ k ] )` (|-128| wraps to -128).
                let abs_q = silk_abs(pulses_ptr[k] as i32) as i8 as i32;
                for j in (1..=n_ls).rev() {
                    let bit = silk_rshift(abs_q, j) & 1;
                    ps_range_enc.enc_icdf(bit as usize, &SILK_LSB_ICDF, 8);
                }
                let bit = abs_q & 1;
                ps_range_enc.enc_icdf(bit as usize, &SILK_LSB_ICDF, 8);
            }
        }
    }

    // Encode signs
    silk_encode_signs(
        ps_range_enc,
        pulses,
        frame_length,
        signal_type,
        quant_offset_type,
        &sum_pulses,
    );
}

// ---------------------------------------------------------------------------------------------
// decode_pitch.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/decode_pitch.c:silk_decode_pitch — writes `nb_subfr` pitch lags.
pub fn silk_decode_pitch(
    lag_index: i16,
    contour_index: i8,
    pitch_lags: &mut [i32],
    fs_khz: i32,
    nb_subfr: i32,
) {
    let (lag_cb_ptr, cbk_size): (&[i8], i32) = if fs_khz == 8 {
        if nb_subfr == PE_MAX_NB_SUBFR {
            (SILK_CB_LAGS_STAGE2.as_flattened(), PE_NB_CBKS_STAGE2_EXT)
        } else {
            debug_assert!(nb_subfr == PE_MAX_NB_SUBFR >> 1);
            (
                SILK_CB_LAGS_STAGE2_10_MS.as_flattened(),
                PE_NB_CBKS_STAGE2_10MS,
            )
        }
    } else if nb_subfr == PE_MAX_NB_SUBFR {
        (SILK_CB_LAGS_STAGE3.as_flattened(), PE_NB_CBKS_STAGE3_MAX)
    } else {
        debug_assert!(nb_subfr == PE_MAX_NB_SUBFR >> 1);
        (
            SILK_CB_LAGS_STAGE3_10_MS.as_flattened(),
            PE_NB_CBKS_STAGE3_10MS,
        )
    };

    let min_lag = silk_smulbb(PE_MIN_LAG_MS, fs_khz);
    let max_lag = silk_smulbb(PE_MAX_LAG_MS, fs_khz);
    let lag = min_lag + lag_index as i32;

    for k in 0..nb_subfr as usize {
        // matrix_ptr( Lag_CB_ptr, k, contourIndex, cbk_size )
        pitch_lags[k] =
            lag + lag_cb_ptr[k * cbk_size as usize + contour_index as i32 as usize] as i32;
        pitch_lags[k] = silk_limit(pitch_lags[k], min_lag, max_lag);
    }
}

// ---------------------------------------------------------------------------------------------
// stereo_decode_pred.c / stereo_encode_pred.c
// ---------------------------------------------------------------------------------------------

/// Port of silk/stereo_decode_pred.c:silk_stereo_decode_pred — decode mid/side predictors.
pub fn silk_stereo_decode_pred(ps_range_dec: &mut EcDec<'_>, pred_q13: &mut [i32]) {
    let mut ix = [[0i32; 3]; 2];

    // Entropy decoding
    let n = ps_range_dec.dec_icdf(&SILK_STEREO_PRED_JOINT_ICDF, 8) as i32;
    ix[0][2] = silk_div32_16(n, 5);
    ix[1][2] = n - 5 * ix[0][2];
    for n in 0..2 {
        ix[n][0] = ps_range_dec.dec_icdf(&SILK_UNIFORM3_ICDF, 8) as i32;
        ix[n][1] = ps_range_dec.dec_icdf(&SILK_UNIFORM5_ICDF, 8) as i32;
    }

    // Dequantize
    for n in 0..2 {
        ix[n][0] += 3 * ix[n][2];
        let low_q13 = SILK_STEREO_PRED_QUANT_Q13[ix[n][0] as usize] as i32;
        let step_q13 = silk_smulwb(
            SILK_STEREO_PRED_QUANT_Q13[ix[n][0] as usize + 1] as i32 - low_q13,
            silk_fix_const(0.5 / STEREO_QUANT_SUB_STEPS as f64, 16),
        );
        pred_q13[n] = silk_smlabb(low_q13, step_q13, 2 * ix[n][1] + 1);
    }

    // Subtract second from first predictor (helps when actually applying these)
    pred_q13[0] -= pred_q13[1];
}

/// Port of silk/stereo_decode_pred.c:silk_stereo_decode_mid_only — decode the flag that only
/// the mid channel has been coded (the C out-parameter is the return value).
#[must_use]
pub fn silk_stereo_decode_mid_only(ps_range_dec: &mut EcDec<'_>) -> i32 {
    // Decode flag that only mid channel is coded
    ps_range_dec.dec_icdf(&SILK_STEREO_ONLY_CODE_MID_ICDF, 8) as i32
}

/// Port of silk/stereo_encode_pred.c:silk_stereo_encode_pred — entropy code the mid/side
/// quantization indices.
pub fn silk_stereo_encode_pred(ps_range_enc: &mut EcEnc<'_>, ix: &[[i8; 3]; 2]) {
    // Entropy coding
    let n = 5 * ix[0][2] as i32 + ix[1][2] as i32;
    debug_assert!(n < 25);
    ps_range_enc.enc_icdf(n as usize, &SILK_STEREO_PRED_JOINT_ICDF, 8);
    for n in 0..2 {
        debug_assert!(ix[n][0] < 3);
        debug_assert!((ix[n][1] as i32) < STEREO_QUANT_SUB_STEPS);
        ps_range_enc.enc_icdf(ix[n][0] as usize, &SILK_UNIFORM3_ICDF, 8);
        ps_range_enc.enc_icdf(ix[n][1] as usize, &SILK_UNIFORM5_ICDF, 8);
    }
}

/// Port of silk/stereo_encode_pred.c:silk_stereo_encode_mid_only — entropy code the mid-only
/// flag.
pub fn silk_stereo_encode_mid_only(ps_range_enc: &mut EcEnc<'_>, mid_only_flag: i8) {
    // Encode flag that only mid channel is coded
    ps_range_enc.enc_icdf(mid_only_flag as usize, &SILK_STEREO_ONLY_CODE_MID_ICDF, 8);
}
