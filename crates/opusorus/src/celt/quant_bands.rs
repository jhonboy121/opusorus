//! Port of celt/quant_bands.c, celt/quant_bands.h: coarse/fine band energy quantisation.
//!
//! Float build. `log2Amp` is declared in `quant_bands.h` but has no definition in libopus, so
//! it has no port.

#![allow(
    clippy::too_many_arguments,
    reason = "energy quantisation functions mirror the C signatures"
)]

use alloc::vec;

use crate::celt::arch::{
    CeltEner, CeltGlog, OpusVal16, OpusVal32, add32, extend32, gconst, imax, imin, mac16_16, maxg,
    min32, mult16_16_q15, mult16_32_q15, pshr32, shl32, shr32, sub32,
};
use crate::celt::entdec::EcDec;
use crate::celt::entenc::EcEnc;
use crate::celt::laplace::{ec_laplace_decode, ec_laplace_encode};
use crate::celt::mathops::celt_log2_db;
use crate::celt::rate::MAX_FINE_BITS;
use crate::celt::static_modes::CeltMode;
use crate::celt::vq::Scratch;
use crate::math;

/// `eMeans`: mean energy in each band quantized in Q4 and converted back to float.
#[rustfmt::skip]
pub static E_MEANS: [OpusVal16; 25] = [
    6.437500, 6.250000, 5.750000, 5.312500, 5.062500,
    4.812500, 4.500000, 4.375000, 4.875000, 4.687500,
    4.562500, 4.437500, 4.875000, 4.625000, 4.312500,
    4.500000, 4.375000, 4.625000, 4.750000, 4.437500,
    3.750000, 3.750000, 3.750000, 3.750000, 3.750000,
];

/// `pred_coef`: prediction coefficients 0.9, 0.8, 0.65, 0.5 (`29440/32768.` ... in double,
/// stored as float; all exactly representable).
pub static PRED_COEF: [OpusVal16; 4] = [
    (29440.0 / 32768.0) as f32,
    (26112.0 / 32768.0) as f32,
    (21248.0 / 32768.0) as f32,
    (16384.0 / 32768.0) as f32,
];
/// `beta_coef`.
pub static BETA_COEF: [OpusVal16; 4] = [
    (30147.0 / 32768.0) as f32,
    (22282.0 / 32768.0) as f32,
    (12124.0 / 32768.0) as f32,
    (6554.0 / 32768.0) as f32,
];
/// `beta_intra`.
pub const BETA_INTRA: OpusVal16 = (4915.0 / 32768.0) as f32;

/// `e_prob_model`: parameters of the Laplace-like probability models used for the coarse
/// energy. There is one pair of parameters for each frame size, prediction type (inter/intra),
/// and band number. The first number of each pair is the probability of 0, and the second is
/// the decay rate, both in Q8 precision.
#[rustfmt::skip]
pub static E_PROB_MODEL: [[[u8; 42]; 2]; 4] = [
    // 120 sample frames.
    [
        // Inter
        [
             72, 127,  65, 129,  66, 128,  65, 128,  64, 128,  62, 128,  64, 128,
             64, 128,  92,  78,  92,  79,  92,  78,  90,  79, 116,  41, 115,  40,
            114,  40, 132,  26, 132,  26, 145,  17, 161,  12, 176,  10, 177,  11,
        ],
        // Intra
        [
             24, 179,  48, 138,  54, 135,  54, 132,  53, 134,  56, 133,  55, 132,
             55, 132,  61, 114,  70,  96,  74,  88,  75,  88,  87,  74,  89,  66,
             91,  67, 100,  59, 108,  50, 120,  40, 122,  37,  97,  43,  78,  50,
        ],
    ],
    // 240 sample frames.
    [
        // Inter
        [
             83,  78,  84,  81,  88,  75,  86,  74,  87,  71,  90,  73,  93,  74,
             93,  74, 109,  40, 114,  36, 117,  34, 117,  34, 143,  17, 145,  18,
            146,  19, 162,  12, 165,  10, 178,   7, 189,   6, 190,   8, 177,   9,
        ],
        // Intra
        [
             23, 178,  54, 115,  63, 102,  66,  98,  69,  99,  74,  89,  71,  91,
             73,  91,  78,  89,  86,  80,  92,  66,  93,  64, 102,  59, 103,  60,
            104,  60, 117,  52, 123,  44, 138,  35, 133,  31,  97,  38,  77,  45,
        ],
    ],
    // 480 sample frames.
    [
        // Inter
        [
             61,  90,  93,  60, 105,  42, 107,  41, 110,  45, 116,  38, 113,  38,
            112,  38, 124,  26, 132,  27, 136,  19, 140,  20, 155,  14, 159,  16,
            158,  18, 170,  13, 177,  10, 187,   8, 192,   6, 175,   9, 159,  10,
        ],
        // Intra
        [
             21, 178,  59, 110,  71,  86,  75,  85,  84,  83,  91,  66,  88,  73,
             87,  72,  92,  75,  98,  72, 105,  58, 107,  54, 115,  52, 114,  55,
            112,  56, 129,  51, 132,  40, 150,  33, 140,  29,  98,  35,  77,  42,
        ],
    ],
    // 960 sample frames.
    [
        // Inter
        [
             42, 121,  96,  66, 108,  43, 111,  40, 117,  44, 123,  32, 120,  36,
            119,  33, 127,  33, 134,  34, 139,  21, 147,  23, 152,  20, 158,  25,
            154,  26, 166,  21, 173,  16, 184,  13, 184,  10, 150,  13, 139,  15,
        ],
        // Intra
        [
             22, 178,  63, 114,  74,  82,  84,  83,  92,  82, 103,  62,  96,  72,
             96,  67, 101,  73, 107,  72, 113,  55, 118,  52, 125,  52, 118,  52,
            117,  55, 135,  49, 137,  39, 157,  32, 145,  29,  97,  33,  77,  40,
        ],
    ],
];

/// `small_energy_icdf`.
pub static SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];

/// Largest range-coder byte span saved on the stack by the two-pass intra decision (the CELT
/// packet cap; QEXT extension payloads can be larger).
const INTRA_SAVE_MAX: usize = if cfg!(feature = "qext") { 3825 } else { 1275 };

/// Port of celt/quant_bands.c:loss_distortion.
#[must_use]
pub fn loss_distortion(
    e_bands: &[CeltGlog],
    old_e_bands: &[CeltGlog],
    start: i32,
    end: i32,
    len: i32,
    c: i32,
) -> OpusVal32 {
    let mut dist: OpusVal32 = 0.0;
    for ch in 0..c {
        for i in start..end {
            let idx = (i + ch * len) as usize;
            let d: CeltGlog = pshr32(sub32(e_bands[idx], old_e_bands[idx]), 0);
            dist = mac16_16(dist, d, d);
        }
    }
    // C: `MIN32(200,SHR32(dist,14))` (the int 200 is compared as float).
    min32(200.0, shr32(dist, 14))
}

/// Port of celt/quant_bands.c:quant_coarse_energy_impl. Returns the "badness" (0 for LFE).
fn quant_coarse_energy_impl(
    m: &CeltMode,
    start: i32,
    end: i32,
    e_bands: &[CeltGlog],
    old_e_bands: &mut [CeltGlog],
    budget: i32,
    mut tell: i32,
    prob_model: &[u8; 42],
    error: &mut [CeltGlog],
    enc: &mut EcEnc<'_>,
    c: i32,
    lm: i32,
    intra: bool,
    max_decay: CeltGlog,
    lfe: bool,
) -> i32 {
    let nb = m.nb_ebands;
    let mut badness: i32 = 0;
    let mut prev: [OpusVal32; 2] = [0.0, 0.0];
    let coef: OpusVal16;
    let beta: OpusVal16;

    if tell + 3 <= budget {
        enc.enc_bit_logp(intra, 3);
    }
    if intra {
        coef = 0.0;
        beta = BETA_INTRA;
    } else {
        beta = BETA_COEF[lm as usize];
        coef = PRED_COEF[lm as usize];
    }

    // Encode at a fixed coarse resolution
    for i in start..end {
        for ch in 0..c {
            let idx = (i + ch * nb) as usize;
            let x: CeltGlog = e_bands[idx];
            let old_e: CeltGlog = maxg(-gconst(9.0), old_e_bands[idx]);
            // FIXED_POINT: not ported (float build).
            let f: OpusVal32 = x - coef * old_e - prev[ch as usize];
            // Rounding to nearest integer here is really important!
            // C: `(int)floor(.5f+f)`.
            let mut qi: i32 = math::floor((0.5f32 + f) as f64) as i32;
            let decay_bound: CeltGlog = maxg(-gconst(28.0), old_e_bands[idx]) - max_decay;
            // Prevent the energy from going down too quickly (e.g. for bands that have just
            // one bin)
            if qi < 0 && x < decay_bound {
                qi += shr32(sub32(decay_bound, x), 24) as i32;
                if qi > 0 {
                    qi = 0;
                }
            }
            let qi0 = qi;
            // If we don't have enough bits to encode all the energy, just assume something
            // safe.
            tell = enc.tell();
            let bits_left = budget - tell - 3 * c * (end - i);
            if i != start && bits_left < 30 {
                if bits_left < 24 {
                    qi = imin(1, qi);
                }
                if bits_left < 16 {
                    qi = imax(-1, qi);
                }
            }
            if lfe && i >= 2 {
                qi = imin(qi, 0);
            }
            if budget - tell >= 15 {
                let pi = (2 * imin(i, 20)) as usize;
                ec_laplace_encode(
                    enc,
                    &mut qi,
                    u32::from(prob_model[pi]) << 7,
                    i32::from(prob_model[pi + 1]) << 6,
                );
            } else if budget - tell >= 2 {
                qi = imax(-1, imin(qi, 1));
                enc.enc_icdf(
                    ((2 * qi) ^ -i32::from(qi < 0)) as usize,
                    &SMALL_ENERGY_ICDF,
                    2,
                );
            } else if budget - tell >= 1 {
                qi = imin(0, qi);
                enc.enc_bit_logp(-qi != 0, 1);
            } else {
                qi = -1;
            }
            error[idx] = f - shl32(qi as f32, 24);
            badness += (qi0 - qi).abs();
            let q: OpusVal32 = shl32(extend32(qi as f32), 24);

            let tmp: OpusVal32 = mult16_32_q15(coef, old_e) + prev[ch as usize] + q;
            // FIXED_POINT: clamp to -28 not ported (float build).
            old_e_bands[idx] = tmp;
            prev[ch as usize] = prev[ch as usize] + q - mult16_32_q15(beta, q);
        }
    }
    if lfe { 0 } else { badness }
}

/// Port of celt/quant_bands.c:quant_coarse_energy.
///
/// Chooses between intra and inter (predicted) coding of the coarse energies of bands
/// `start..end` (trying both when `two_pass`, rolling the encoder back with a state snapshot
/// plus a copy of the overwritten bytes), and updates `old_e_bands`, `error` and
/// `delayed_intra`.
pub fn quant_coarse_energy(
    m: &CeltMode,
    start: i32,
    end: i32,
    eff_end: i32,
    e_bands: &[CeltGlog],
    old_e_bands: &mut [CeltGlog],
    budget: u32,
    error: &mut [CeltGlog],
    enc: &mut EcEnc<'_>,
    c: i32,
    lm: i32,
    nb_available_bytes: i32,
    force_intra: bool,
    delayed_intra: &mut OpusVal32,
    mut two_pass: bool,
    loss_rate: i32,
    lfe: bool,
) {
    let nb = m.nb_ebands;
    let len = (c * nb) as usize;
    let mut badness1: i32 = 0;

    let mut intra = force_intra
        || (!two_pass
            && *delayed_intra > (2 * c * (end - start)) as f32
            && nb_available_bytes > (end - start) * c);
    // C: `(opus_int32)((budget**delayedIntra*loss_rate)/(C*512))` in float.
    let intra_bias: i32 =
        ((budget as f32 * *delayed_intra * loss_rate as f32) / (c * 512) as f32) as i32;
    let new_distortion: OpusVal32 = loss_distortion(e_bands, old_e_bands, start, eff_end, nb, c);

    let tell: u32 = enc.tell() as u32;
    if tell.wrapping_add(3) > budget {
        two_pass = false;
        intra = false;
    }

    let mut max_decay: CeltGlog = gconst(16.0);
    if end - start > 10 {
        // FIXED_POINT: not ported (float build).
        max_decay = min32(max_decay, 0.125f32 * nb_available_bytes as f32);
    }
    if lfe {
        max_decay = gconst(3.0);
    }
    let enc_start_state = enc.snapshot();
    let nstart_bytes = enc.range_bytes();

    let mut old_intra_buf = Scratch::<CeltGlog, 64>::new();
    let mut error_intra_buf = Scratch::<CeltGlog, 64>::new();
    let old_e_bands_intra = old_intra_buf.get(len);
    let error_intra = error_intra_buf.get(len);
    old_e_bands_intra.copy_from_slice(&old_e_bands[..len]);

    if two_pass || intra {
        badness1 = quant_coarse_energy_impl(
            m,
            start,
            end,
            e_bands,
            old_e_bands_intra,
            budget as i32,
            tell as i32,
            &E_PROB_MODEL[lm as usize][1],
            error_intra,
            enc,
            c,
            lm,
            true,
            max_decay,
            lfe,
        );
    }

    if !intra {
        let tell_intra: i32 = enc.tell_frac() as i32;

        let enc_intra_state = enc.snapshot();

        let nintra_bytes = enc.range_bytes();
        let save_bytes = (nintra_bytes - nstart_bytes) as usize;
        let mut stack_bits = [0u8; INTRA_SAVE_MAX];
        let mut heap_bits;
        let intra_bits: &mut [u8] = if save_bytes <= INTRA_SAVE_MAX {
            &mut stack_bits[..save_bytes]
        } else {
            heap_bits = vec![0u8; save_bytes];
            &mut heap_bits
        };
        let (s0, s1) = (nstart_bytes as usize, nintra_bytes as usize);
        // Copy bits from intra bit-stream
        intra_bits.copy_from_slice(&enc.buffer()[s0..s1]);

        enc.restore(&enc_start_state);

        let badness2 = quant_coarse_energy_impl(
            m,
            start,
            end,
            e_bands,
            old_e_bands,
            budget as i32,
            tell as i32,
            &E_PROB_MODEL[lm as usize][usize::from(intra)],
            error,
            enc,
            c,
            lm,
            false,
            max_decay,
            lfe,
        );

        if two_pass
            && (badness1 < badness2
                || (badness1 == badness2 && (enc.tell_frac() as i32) + intra_bias > tell_intra))
        {
            enc.restore(&enc_intra_state);
            // Copy intra bits to bit-stream
            enc.buffer_mut()[s0..s1].copy_from_slice(intra_bits);
            old_e_bands[..len].copy_from_slice(old_e_bands_intra);
            error[..len].copy_from_slice(error_intra);
            intra = true;
        }
    } else {
        old_e_bands[..len].copy_from_slice(old_e_bands_intra);
        error[..len].copy_from_slice(error_intra);
    }

    if intra {
        *delayed_intra = new_distortion;
    } else {
        *delayed_intra = add32(
            mult16_32_q15(
                mult16_16_q15(PRED_COEF[lm as usize], PRED_COEF[lm as usize]),
                *delayed_intra,
            ),
            new_distortion,
        );
    }
}

/// Port of celt/quant_bands.c:quant_fine_energy.
///
/// `prev_quant` is the C `prev_quant` pointer (`None` for `NULL`); `extra_quant[i]` is the
/// number of fine bits for band `i`.
pub fn quant_fine_energy(
    m: &CeltMode,
    start: i32,
    end: i32,
    old_e_bands: &mut [CeltGlog],
    error: &mut [CeltGlog],
    prev_quant: Option<&[i32]>,
    extra_quant: &[i32],
    enc: &mut EcEnc<'_>,
    c: i32,
) {
    let nb = m.nb_ebands;
    // Encode finer resolution
    for i in start..end {
        let iu = i as usize;
        let eq = extra_quant[iu];
        // C computes `extra = 1<<extra_quant[i]` before this test; it is only used after it.
        if eq <= 0 {
            continue;
        }
        let extra: i16 = (1i32 << eq) as i16;
        if enc.tell() + c * eq > (enc.storage as i32) * 8 {
            continue;
        }
        let prev: i16 = match prev_quant {
            Some(p) => p[iu] as i16,
            None => 0,
        };
        for ch in 0..c {
            let idx = (i + ch * nb) as usize;
            // FIXED_POINT: not ported (float build).
            // C: `(int)floor((error*(1<<prev)+.5f)*extra)`.
            let mut q2: i32 = math::floor(
                ((error[idx] * (1i32 << prev) as f32 + 0.5f32) * f32::from(extra)) as f64,
            ) as i32;
            if q2 > i32::from(extra) - 1 {
                q2 = i32::from(extra) - 1;
            }
            if q2 < 0 {
                q2 = 0;
            }
            enc.enc_bits(q2 as u32, eq as u32);
            let mut offset: CeltGlog =
                (q2 as f32 + 0.5f32) * (1i32 << (14 - eq)) as f32 * (1.0f32 / 16384.0) - 0.5f32;
            offset *= (1i32 << (14 - i32::from(prev))) as f32 * (1.0f32 / 16384.0);
            old_e_bands[idx] += offset;
            error[idx] -= offset;
        }
    }
}

/// Port of celt/quant_bands.c:quant_energy_finalise: uses up the remaining `bits_left` bits
/// for one more bit of fine energy in priority order. `old_e_bands` is `None` for the C
/// `NULL`.
pub fn quant_energy_finalise(
    m: &CeltMode,
    start: i32,
    end: i32,
    mut old_e_bands: Option<&mut [CeltGlog]>,
    error: &mut [CeltGlog],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    enc: &mut EcEnc<'_>,
    c: i32,
) {
    let nb = m.nb_ebands;
    // Use up the remaining bits
    for prio in 0..2 {
        let mut i = start;
        while i < end && bits_left >= c {
            let iu = i as usize;
            if fine_quant[iu] >= MAX_FINE_BITS || fine_priority[iu] != prio {
                i += 1;
                continue;
            }
            for ch in 0..c {
                let idx = (i + ch * nb) as usize;
                let q2: i32 = if error[idx] < 0.0 { 0 } else { 1 };
                enc.enc_bits(q2 as u32, 1);
                // FIXED_POINT: not ported (float build).
                let offset: CeltGlog = (q2 as f32 - 0.5f32)
                    * (1i32 << (14 - fine_quant[iu] - 1)) as f32
                    * (1.0f32 / 16384.0);
                if let Some(old) = old_e_bands.as_deref_mut() {
                    old[idx] += offset;
                }
                error[idx] -= offset;
                bits_left -= 1;
            }
            i += 1;
        }
    }
}

/// Port of celt/quant_bands.c:unquant_coarse_energy.
pub fn unquant_coarse_energy(
    m: &CeltMode,
    start: i32,
    end: i32,
    old_e_bands: &mut [CeltGlog],
    intra: bool,
    dec: &mut EcDec<'_>,
    c: i32,
    lm: i32,
) {
    let nb = m.nb_ebands;
    let prob_model = &E_PROB_MODEL[lm as usize][usize::from(intra)];
    // C: `opus_val64 prev[2]` (float in the float build).
    let mut prev: [f32; 2] = [0.0, 0.0];
    let coef: OpusVal16;
    let beta: OpusVal16;

    if intra {
        coef = 0.0;
        beta = BETA_INTRA;
    } else {
        beta = BETA_COEF[lm as usize];
        coef = PRED_COEF[lm as usize];
    }

    let budget: i32 = dec.storage.wrapping_mul(8) as i32;

    // Decode at a fixed coarse resolution
    for i in start..end {
        for ch in 0..c {
            debug_assert!(ch < 2);
            let idx = (i + ch * nb) as usize;
            let tell = dec.tell();
            let qi: i32 = if budget - tell >= 15 {
                let pi = (2 * imin(i, 20)) as usize;
                ec_laplace_decode(
                    dec,
                    u32::from(prob_model[pi]) << 7,
                    i32::from(prob_model[pi + 1]) << 6,
                )
            } else if budget - tell >= 2 {
                let qi = dec.dec_icdf(&SMALL_ENERGY_ICDF, 2) as i32;
                (qi >> 1) ^ -(qi & 1)
            } else if budget - tell >= 1 {
                -i32::from(dec.dec_bit_logp(1))
            } else {
                -1
            };
            let q: OpusVal32 = shl32(extend32(qi as f32), 24);

            old_e_bands[idx] = maxg(-gconst(9.0), old_e_bands[idx]);
            let tmp: OpusVal32 = mult16_32_q15(coef, old_e_bands[idx]) + prev[ch as usize] + q;
            // FIXED_POINT: clamp to +-28 not ported (float build).
            old_e_bands[idx] = tmp;
            prev[ch as usize] = prev[ch as usize] + q - mult16_32_q15(beta, q);
        }
    }
}

/// Port of celt/quant_bands.c:unquant_fine_energy.
pub fn unquant_fine_energy(
    m: &CeltMode,
    start: i32,
    end: i32,
    old_e_bands: &mut [CeltGlog],
    prev_quant: Option<&[i32]>,
    extra_quant: &[i32],
    dec: &mut EcDec<'_>,
    c: i32,
) {
    let nb = m.nb_ebands;
    // Decode finer resolution
    for i in start..end {
        let iu = i as usize;
        let extra: i16 = extra_quant[iu] as i16;
        if extra_quant[iu] <= 0 {
            continue;
        }
        if dec.tell() + c * extra_quant[iu] > (dec.storage as i32) * 8 {
            continue;
        }
        let prev: i16 = match prev_quant {
            Some(p) => p[iu] as i16,
            None => 0,
        };
        for ch in 0..c {
            let idx = (i + ch * nb) as usize;
            let q2: i32 = dec.dec_bits(extra as u32) as i32;
            // FIXED_POINT: not ported (float build).
            let mut offset: CeltGlog = (q2 as f32 + 0.5f32)
                * (1i32 << (14 - i32::from(extra))) as f32
                * (1.0f32 / 16384.0)
                - 0.5f32;
            offset *= (1i32 << (14 - i32::from(prev))) as f32 * (1.0f32 / 16384.0);
            old_e_bands[idx] += offset;
        }
    }
}

/// Port of celt/quant_bands.c:unquant_energy_finalise. `old_e_bands` is `None` for the C
/// `NULL`.
pub fn unquant_energy_finalise(
    m: &CeltMode,
    start: i32,
    end: i32,
    mut old_e_bands: Option<&mut [CeltGlog]>,
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    dec: &mut EcDec<'_>,
    c: i32,
) {
    let nb = m.nb_ebands;
    // Use up the remaining bits
    for prio in 0..2 {
        let mut i = start;
        while i < end && bits_left >= c {
            let iu = i as usize;
            if fine_quant[iu] >= MAX_FINE_BITS || fine_priority[iu] != prio {
                i += 1;
                continue;
            }
            for ch in 0..c {
                let idx = (i + ch * nb) as usize;
                let q2: i32 = dec.dec_bits(1) as i32;
                // FIXED_POINT: not ported (float build).
                let offset: CeltGlog = (q2 as f32 - 0.5f32)
                    * (1i32 << (14 - fine_quant[iu] - 1)) as f32
                    * (1.0f32 / 16384.0);
                if let Some(old) = old_e_bands.as_deref_mut() {
                    old[idx] += offset;
                }
                bits_left -= 1;
            }
            i += 1;
        }
    }
}

/// Port of celt/quant_bands.c:amp2Log2: converts band amplitudes to log2 energies relative to
/// [`E_MEANS`]; bands `eff_end..end` are set to -14.
pub fn amp2_log2(
    m: &CeltMode,
    eff_end: i32,
    end: i32,
    band_e: &[CeltEner],
    band_log_e: &mut [CeltGlog],
    c: i32,
) {
    let nb = m.nb_ebands;
    for ch in 0..c {
        for i in 0..eff_end {
            let idx = (i + ch * nb) as usize;
            band_log_e[idx] = celt_log2_db(band_e[idx]) - shl32(E_MEANS[i as usize], 24 - 4);
            // FIXED_POINT: Q12 compensation not ported (float build).
        }
        for i in eff_end..end {
            band_log_e[(ch * nb + i) as usize] = -gconst(14.0);
        }
    }
}
