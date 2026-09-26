//! Port of the fixed-point (`#ifdef FIXED_POINT`) parts of `celt/mathops.h` and
//! `celt/mathops.c` (feature `fixed-point`).
//!
//! Every function mirrors the C integer arithmetic exactly, including the implicit C narrowing
//! conversions (an `int` assigned to an `opus_val16` / passed to an `opus_val16` parameter is
//! truncated with `as i16`); see `crate::celt::arch` (fixed) for the macro typing rules.

use crate::celt::arch::{
    DB_SHIFT, OpusRes, OpusVal16, OpusVal32, abs32, add16, add32, extend32, max16, max32, min16,
    min32, mult16_16_p15, mult16_16_q15, mult16_32_q15, mult32_32_q31, pshr32, round16, shl16,
    shl32, shr16, shr32, sub16, sub32, vshr32,
};

use super::celt_ilog2;

/// Port of `celt_maxabs16` (fixed): `MAX32(EXTEND32(maxval),-EXTEND32(minval))`.
#[inline]
#[must_use]
pub fn celt_maxabs16(x: &[OpusVal16]) -> OpusVal32 {
    let mut maxval: OpusVal16 = 0;
    let mut minval: OpusVal16 = 0;
    for &v in x {
        maxval = max16(maxval, v);
        minval = min16(minval, v);
    }
    max32(extend32(maxval), -extend32(minval))
}

/// Port of `celt_maxabs_res` (`ENABLE_RES24` fixed-point version).
#[cfg(feature = "fixed-res24")]
#[inline]
#[must_use]
pub fn celt_maxabs_res(x: &[OpusRes]) -> OpusRes {
    let mut maxval: OpusRes = 0;
    let mut minval: OpusRes = 0;
    for &v in x {
        maxval = max32(maxval, v);
        minval = min32(minval, v);
    }
    // opus_res should never reach such amplitude, so we should be safe.
    debug_assert!(minval != i32::MIN);
    max32(maxval, -minval)
}

/// `celt_maxabs_res` (16-bit `opus_res`: alias of [`celt_maxabs16`]).
#[cfg(not(feature = "fixed-res24"))]
#[inline]
#[must_use]
pub fn celt_maxabs_res(x: &[OpusRes]) -> OpusVal32 {
    celt_maxabs16(x)
}

/// Port of `celt_maxabs32` (fixed).
#[inline]
#[must_use]
pub fn celt_maxabs32(x: &[OpusVal32]) -> OpusVal32 {
    let mut maxval: OpusVal32 = 0;
    let mut minval: OpusVal32 = 0;
    for &v in x {
        maxval = max32(maxval, v);
        minval = min32(minval, v);
    }
    max32(maxval, -minval)
}

/// Port of `celt/mathops.c:frac_div32_q29`: `a/b` in Q29.
#[must_use]
pub fn frac_div32_q29(mut a: OpusVal32, mut b: OpusVal32) -> OpusVal32 {
    let shift = celt_ilog2(b) - 29;
    a = vshr32(a, shift);
    b = vshr32(b, shift);
    // 16-bit reciprocal. C: `ROUND16(celt_rcp(ROUND16(b,16)),3)`.
    let rcp: OpusVal16 = round16(celt_rcp(i32::from(round16(b, 16))), 3);
    let mut result = mult16_32_q15(rcp, a);
    let rem = pshr32(a, 2) - mult32_32_q31(result, b);
    result = add32(result, shl32(mult16_32_q15(rcp, rem), 2));
    result
}

/// Port of `celt/mathops.c:frac_div32`: `a/b` in Q31, saturated to ±(2^31-1).
#[must_use]
pub fn frac_div32(a: OpusVal32, b: OpusVal32) -> OpusVal32 {
    let result = frac_div32_q29(a, b);
    if result >= 536_870_912 {
        // 2^29
        2_147_483_647 // 2^31 - 1
    } else if result <= -536_870_912 {
        // -2^29
        -2_147_483_647 // -2^31
    } else {
        shl32(result, 2)
    }
}

/// Port of `celt/mathops.c:celt_rsqrt_norm`: reciprocal sqrt approximation in the range
/// `[0.25,1)` (Q16 in, Q14 out).
#[must_use]
pub fn celt_rsqrt_norm(x: OpusVal32) -> OpusVal16 {
    // Range of n is [-16384,32767] ([-0.5,1) in Q15).
    let n: OpusVal16 = (x - 32768) as i16;
    // Get a rough initial guess for the root. The optimal minimax quadratic approximation (using
    // relative error) is r = 1.437799046117536+n*(-0.823394375837328+n*0.4096419668459485).
    // Coefficients here, and the final result r, are Q14.
    let r: OpusVal16 = add16(
        23557,
        mult16_16_q15(n, add16(-13490, mult16_16_q15(n, 6713))),
    );
    // We want y = x*r*r-1 in Q15, but x is 32-bit Q16 and r is Q14. We can compute the result
    // from n and r using Q15 multiplies with some adjustment, carefully done to avoid overflow.
    // Range of y is [-1564,1594].
    let r2: OpusVal16 = mult16_16_q15(r, r) as i16;
    let y: OpusVal16 = shl16(sub16(add16(mult16_16_q15(r2, n), r2), 16384), 1);
    // Apply a 2nd-order Householder iteration: r += r*y*(y*0.375-0.5). This yields the Q14
    // reciprocal square root of the Q16 x, with a maximum relative error of 1.04956E-4, a
    // (relative) RMSE of 2.80979E-5, and a peak absolute error of 2.26591/16384.
    add16(
        r,
        mult16_16_q15(r, mult16_16_q15(y, sub16(mult16_16_q15(y, 12288), 16384))),
    )
}

/// Port of `celt/mathops.c:celt_rsqrt_norm32`: reciprocal sqrt approximation in the range
/// `[0.25,1)` (Q31 in, Q29 out).
#[must_use]
pub fn celt_rsqrt_norm32(x: OpusVal32) -> OpusVal32 {
    // Use the first-order Newton-Raphson method to refine the root estimate.
    // r = r * (1.5 - 0.5*x*r*r)
    let r_q29 = shl32(celt_rsqrt_norm(shr32(x, 31 - 16)), 15);
    // Split evaluation in steps to avoid exploding macro expansion.
    let mut tmp = mult32_32_q31(r_q29, r_q29);
    tmp = mult32_32_q31(1_073_741_824 /* Q31 */, tmp);
    tmp = mult32_32_q31(x, tmp);
    shl32(mult32_32_q31(r_q29, sub32(201_326_592 /* Q27 */, tmp)), 4)
}

/// Port of `celt/mathops.c:celt_sqrt`: sqrt approximation (QX input, QX/2 output).
#[must_use]
pub fn celt_sqrt(mut x: OpusVal32) -> OpusVal32 {
    // These coeffs are optimized in fixed-point to minimize both RMS and max error of sqrt(x)
    // over .25<x<1 without exceeding 32767. The RMS error is 3.4e-5 and the max is 8.2e-5.
    const C: [OpusVal16; 6] = [23171, 11574, -2901, 1592, -1002, 336];
    if x == 0 {
        return 0;
    } else if x >= 1_073_741_824 {
        return 32767;
    }
    let k = (celt_ilog2(x) >> 1) - 7;
    x = vshr32(x, 2 * k);
    let n: OpusVal16 = (x - 32768) as i16;
    let mut rt = add32(
        C[0],
        mult16_16_q15(
            n,
            add16(
                C[1],
                mult16_16_q15(
                    n,
                    add16(
                        C[2],
                        mult16_16_q15(
                            n,
                            add16(C[3], mult16_16_q15(n, add16(C[4], mult16_16_q15(n, C[5])))),
                        ),
                    ),
                ),
            ),
        ),
    );
    rt = vshr32(rt, 7 - k);
    rt
}

/// Port of `celt/mathops.c:celt_sqrt32`: sqrt approximation; for a Qx input the output is in
/// Q(x/2 + 16).
#[must_use]
pub fn celt_sqrt32(x: OpusVal32) -> OpusVal32 {
    if x == 0 {
        return 0;
    } else if x >= 1_073_741_824 {
        return 2_147_483_647; // 2^31 -1
    }
    let k = celt_ilog2(x) >> 1;
    let mut x_frac = vshr32(x, 2 * (k - 14) - 1);
    x_frac = mult32_32_q31(celt_rsqrt_norm32(x_frac), x_frac);
    if k < 12 {
        pshr32(x_frac, 12 - k)
    } else {
        shl32(x_frac, k - 12)
    }
}

/// Port of `celt/mathops.c:_celt_cos_pi_2`.
#[must_use]
fn celt_cos_pi_2(x: OpusVal16) -> OpusVal16 {
    const L1: i32 = 32767;
    const L2: i32 = -7651;
    const L3: i32 = 8277;
    const L4: i32 = -626;
    let x2: OpusVal16 = mult16_16_p15(x, x) as i16;
    add16(
        1,
        min16(
            32766,
            add32(
                sub16(L1, x2),
                mult16_16_p15(
                    x2,
                    add32(L2, mult16_16_p15(x2, add32(L3, mult16_16_p15(L4, x2)))),
                ),
            ),
        ),
    )
}

/// Port of `celt/mathops.c:celt_cos_norm`: `cos(PI/2 * x)` for a Q16 `x` (Q15 output).
#[must_use]
pub fn celt_cos_norm(mut x: OpusVal32) -> OpusVal16 {
    x &= 0x0001ffff;
    if x > shl32(1, 16) {
        x = sub32(shl32(1, 17), x);
    }
    if x & 0x00007fff != 0 {
        if x < shl32(1, 15) {
            celt_cos_pi_2(x as i16)
        } else {
            // C: `NEG16(_celt_cos_pi_2(EXTRACT16(65536-x)))`, returned as `opus_val16`.
            (-i32::from(celt_cos_pi_2((65536 - x) as i16))) as i16
        }
    } else if x & 0x0000ffff != 0 {
        0
    } else if x & 0x0001ffff != 0 {
        -32767
    } else {
        32767
    }
}

/// Port of `celt/mathops.c:celt_cos_norm32`: `cos(PI*0.5*x)` for a Q30 `x` in `[-1, 1]` (Q31
/// output).
#[must_use]
pub fn celt_cos_norm32(x: OpusVal32) -> OpusVal32 {
    const COS_NORM_COEFF_A0: OpusVal32 = 134_217_720; // Q27
    const COS_NORM_COEFF_A1: OpusVal32 = -662_336_704; // Q29
    const COS_NORM_COEFF_A2: OpusVal32 = 544_710_848; // Q31
    const COS_NORM_COEFF_A3: OpusVal32 = -178_761_936; // Q33
    const COS_NORM_COEFF_A4: OpusVal32 = 29_487_206; // Q35
    // The expected x is in the range of [-1.0f, 1.0f].
    debug_assert!((-1_073_741_824..=1_073_741_824).contains(&x));
    // Make cos(+/- pi/2) exactly zero.
    if abs32(x) == 1 << 30 {
        return 0;
    }
    let x_sq_q29 = mult32_32_q31(x, x);
    // Split evaluation in steps to avoid exploding macro expansion.
    let mut tmp = add32(
        COS_NORM_COEFF_A3,
        mult32_32_q31(x_sq_q29, COS_NORM_COEFF_A4),
    );
    tmp = add32(COS_NORM_COEFF_A2, mult32_32_q31(x_sq_q29, tmp));
    tmp = add32(COS_NORM_COEFF_A1, mult32_32_q31(x_sq_q29, tmp));
    shl32(add32(COS_NORM_COEFF_A0, mult32_32_q31(x_sq_q29, tmp)), 4)
}

/// Port of `celt/mathops.c:celt_rcp_norm16`: 16-bit approximate reciprocal `1/x` of a normalized
/// Q15 input (Q15 output).
#[must_use]
pub fn celt_rcp_norm16(x: OpusVal16) -> OpusVal16 {
    // Start with a linear approximation: r = 1.8823529411764706-0.9411764705882353*n. The
    // coefficients and the result are Q14 in the range [15420,30840].
    let mut r: OpusVal16 = add16(30840, mult16_16_q15(-15420, x));
    // Perform two Newton iterations: r -= r*((r*n)+(r-1.Q15)) = r*((r*n)+(r-1.Q15)).
    r = sub16(
        r,
        mult16_16_q15(r, add16(mult16_16_q15(r, x), add16(r, -32768))),
    ) as i16;
    // We subtract an extra 1 in the second iteration to avoid overflow; it also neatly
    // compensates for truncation error in the rest of the process.
    sub16(
        r,
        add16(
            1,
            mult16_16_q15(r, add16(mult16_16_q15(r, x), add16(r, -32768))),
        ),
    ) as i16
}

/// Port of `celt/mathops.c:celt_rcp_norm32`: 32-bit approximate reciprocal of a normalized Q31
/// input in `[0.5, 1)` (Q30 output in `[1, 2)`).
#[must_use]
pub fn celt_rcp_norm32(x: OpusVal32) -> OpusVal32 {
    debug_assert!(x >= 1_073_741_824);
    let r_q30 = shl32(celt_rcp_norm16((shr32(x, 15) - 32768) as i16), 16);
    // Solving f(y) = a - 1/y using the Newton Method
    // Note: f(y)' = 1/y^2
    // r = r - f(r)/f(r)' = r - (x * r*r - r)
    //   = r - r*(r*x - 1)
    // where
    //   - r means 1/y's approximation.
    //   - x means a, the input of function.
    // Please note that:
    //   - It adds 1 to avoid overflow
    //   - -1.0f in Q30 is -1073741824.
    sub32(
        r_q30,
        add32(
            shl32(
                mult32_32_q31(add32(mult32_32_q31(r_q30, x), -1_073_741_824), r_q30),
                1,
            ),
            1,
        ),
    )
}

/// Port of `celt/mathops.c:celt_rcp`: reciprocal approximation (Q15 input, Q16 output).
#[must_use]
pub fn celt_rcp(x: OpusVal32) -> OpusVal32 {
    debug_assert!(x > 0);
    let i = celt_ilog2(x);
    // Compute the reciprocal of a Q15 number in the range [0, 1).
    let r: OpusVal16 = celt_rcp_norm16((vshr32(x, i - 15) - 32768) as i16);
    // r is now the Q15 solution to 2/(n+1), with a maximum relative error of 7.05346E-5, a
    // (relative) RMSE of 2.14418E-5, and a peak absolute error of 1.24665/32768.
    vshr32(extend32(r), i - 16)
}

/// `celt_div(a,b)`: `MULT32_32_Q31((opus_val32)(a),celt_rcp(b))`.
#[inline]
#[must_use]
pub fn celt_div(a: OpusVal32, b: OpusVal32) -> OpusVal32 {
    mult32_32_q31(a, celt_rcp(b))
}

/// Port of `celt_log2` (fixed): base-2 logarithm approximation (Q14 input, Q10 output).
#[must_use]
pub fn celt_log2(x: OpusVal32) -> OpusVal16 {
    // -0.41509302963303146, 0.9609890551383969, -0.31836011537636605,
    //  0.15530808010959576, -0.08556153059057618
    const C: [OpusVal16; 5] = [-6801 + (1 << (13 - 10)), 15746, -5217, 2545, -1401];
    if x == 0 {
        return -32767;
    }
    let i = celt_ilog2(x);
    let n: OpusVal16 = (vshr32(x, i - 15) - 32768 - 16384) as i16;
    let frac: OpusVal16 = add16(
        C[0],
        mult16_16_q15(
            n,
            add16(
                C[1],
                mult16_16_q15(
                    n,
                    add16(C[2], mult16_16_q15(n, add16(C[3], mult16_16_q15(n, C[4])))),
                ),
            ),
        ),
    );
    (shl32(i - 13, 10) + shr32(frac, 14 - 10)) as i16
}

/// Port of `celt_exp2_frac` (fixed). `K0 = 1, K1 = log(2), K2 = 3-4*log(2), K3 = 3*log(2) - 2`.
#[must_use]
pub fn celt_exp2_frac(x: OpusVal16) -> OpusVal32 {
    const D0: i32 = 16383;
    const D1: i32 = 22804;
    const D2: i32 = 14819;
    const D3: i32 = 10204;
    let frac: OpusVal16 = shl16(x, 4);
    i32::from(add16(
        D0,
        mult16_16_q15(
            frac,
            add16(D1, mult16_16_q15(frac, add16(D2, mult16_16_q15(D3, frac)))),
        ),
    ))
}

/// Port of `celt_exp2` (fixed): base-2 exponential approximation (Q10 input, Q16 output).
#[must_use]
pub fn celt_exp2(x: OpusVal16) -> OpusVal32 {
    let integer = shr16(x, 10);
    if integer > 14 {
        return 0x7f000000;
    } else if integer < -15 {
        return 0;
    }
    // C: `frac = celt_exp2_frac(x-SHL16(integer,10));` (both conversions to opus_val16).
    let frac: OpusVal16 =
        celt_exp2_frac((i32::from(x) - i32::from(shl16(integer, 10))) as i16) as i16;
    vshr32(extend32(frac), -integer - 2)
}

/// Port of `celt_log2_db` (fixed, QEXT): base-2 logarithm of a Q14 input, in Q`DB_SHIFT`
/// (`-32.0` for 0).
#[cfg(feature = "qext")]
#[must_use]
pub fn celt_log2_db(x: OpusVal32) -> OpusVal32 {
    // Q30
    const LOG2_X_NORM_COEFF: [OpusVal32; 8] = [
        1_073_741_824,
        954_437_184,
        858_993_472,
        780_903_168,
        715_827_904,
        660_764_224,
        613_566_784,
        572_662_336,
    ];
    // Q24
    const LOG2_Y_NORM_COEFF: [OpusVal32; 8] = [
        0, 2_850_868, 5_401_057, 7_707_983, 9_814_042, 11_751_428, 13_545_168, 15_215_099,
    ];
    const LOG2_COEFF_A0: OpusVal32 = 1_467_383; // Q24
    const LOG2_COEFF_A1: OpusVal32 = 182_244_800; // Q27
    const LOG2_COEFF_A2: OpusVal32 = -21_440_512; // Q25
    const LOG2_COEFF_A3: OpusVal32 = 107_903_336; // Q28
    const LOG2_COEFF_A4: OpusVal32 = -610_217_024; // Q31
    if x == 0 {
        return -536_870_912; // -32.0f
    }
    let integer = sub32(celt_ilog2(x), 14); // Q0
    let mut mantissa = vshr32(x, integer + 14 - 29); // Q29
    let norm_coeff_idx = (shr32(mantissa, 29 - 3) & 0x7) as usize;
    // mantissa is in Q28 (29 + Q_NORM_CONST - 31 where Q_NORM_CONST is Q30)
    // 285212672 (Q28) is 1.0625f.
    mantissa = sub32(
        mult32_32_q31(mantissa, LOG2_X_NORM_COEFF[norm_coeff_idx]),
        285_212_672,
    );
    // q_a3(Q28): q_mantissa + q_a4 - 31
    // q_a2(Q25): q_mantissa + q_a3 - 31
    // q_a1(Q27): q_mantissa + q_a2 - 31 + 5
    // q_a0(Q24): q_mantissa + q_a1 - 31
    // where  q_mantissa is Q28
    // Split evaluation in steps to avoid exploding macro expansion.
    let mut tmp = mult32_32_q31(mantissa, LOG2_COEFF_A4);
    tmp = mult32_32_q31(mantissa, add32(LOG2_COEFF_A3, tmp));
    tmp = shl32(mult32_32_q31(mantissa, add32(LOG2_COEFF_A2, tmp)), 5);
    tmp = mult32_32_q31(mantissa, add32(LOG2_COEFF_A1, tmp));
    add32(
        LOG2_Y_NORM_COEFF[norm_coeff_idx],
        add32(shl32(integer, DB_SHIFT), add32(LOG2_COEFF_A0, tmp)),
    )
}

/// Port of `celt_exp2_db_frac` (fixed, QEXT): `exp2` of a Q`DB_SHIFT` fraction in `[0, 1)`
/// (Q28 output).
#[cfg(feature = "qext")]
#[must_use]
pub fn celt_exp2_db_frac(x: OpusVal32) -> OpusVal32 {
    // Approximation constants.
    const EXP2_COEFF_A0: i32 = 268_435_440; // Q28
    const EXP2_COEFF_A1: i32 = 744_267_456; // Q30
    const EXP2_COEFF_A2: i32 = 1_031_451_904; // Q32
    const EXP2_COEFF_A3: i32 = 959_088_832; // Q34
    const EXP2_COEFF_A4: i32 = 617_742_720; // Q36
    const EXP2_COEFF_A5: i32 = 516_104_352; // Q38
    // Converts input value from Q24 to Q29.
    let x_q29 = shl32(x, 29 - 24);
    // Split evaluation in steps to avoid exploding macro expansion.
    let mut tmp = add32(EXP2_COEFF_A4, mult32_32_q31(x_q29, EXP2_COEFF_A5));
    tmp = add32(EXP2_COEFF_A3, mult32_32_q31(x_q29, tmp));
    tmp = add32(EXP2_COEFF_A2, mult32_32_q31(x_q29, tmp));
    tmp = add32(EXP2_COEFF_A1, mult32_32_q31(x_q29, tmp));
    add32(EXP2_COEFF_A0, mult32_32_q31(x_q29, tmp))
}

/// Port of `celt_exp2_db` (fixed, QEXT): `exp2` of a Q`DB_SHIFT` input (Q16 output).
#[cfg(feature = "qext")]
#[must_use]
pub fn celt_exp2_db(x: OpusVal32) -> OpusVal32 {
    let integer = shr32(x, DB_SHIFT);
    if integer > 14 {
        return 0x7f000000;
    } else if integer <= -17 {
        return 0;
    }
    let frac = celt_exp2_db_frac(x - shl32(integer, DB_SHIFT)); // Q28
    vshr32(frac, -integer + 28 - 16) // Q16
}

/// `celt_log2_db` (fixed, no QEXT): `SHL32(EXTEND32(celt_log2(x)), DB_SHIFT-10)`.
#[cfg(not(feature = "qext"))]
#[inline]
#[must_use]
pub fn celt_log2_db(x: OpusVal32) -> OpusVal32 {
    shl32(celt_log2(x), DB_SHIFT - 10)
}

/// `celt_exp2_db_frac` (fixed, no QEXT): `SHL32(celt_exp2_frac(PSHR32(x, DB_SHIFT-10)), 14)`
/// (the `PSHR32` result is converted to the `opus_val16` parameter).
#[cfg(not(feature = "qext"))]
#[inline]
#[must_use]
pub fn celt_exp2_db_frac(x: OpusVal32) -> OpusVal32 {
    shl32(celt_exp2_frac(pshr32(x, DB_SHIFT - 10) as i16), 14)
}

/// `celt_exp2_db` (fixed, no QEXT): `celt_exp2(PSHR32(x, DB_SHIFT-10))` (the `PSHR32` result is
/// converted to the `opus_val16` parameter).
#[cfg(not(feature = "qext"))]
#[inline]
#[must_use]
pub fn celt_exp2_db(x: OpusVal32) -> OpusVal32 {
    celt_exp2(pshr32(x, DB_SHIFT - 10) as i16)
}

/// Port of `celt_atan_norm` (fixed): `atan(x)*2/pi` for a Q30 `x` in `[-1, 1]` (Q30 output).
#[must_use]
pub fn celt_atan_norm(x: OpusVal32) -> OpusVal32 {
    // Approximation constants.
    const ATAN_2_OVER_PI: i32 = 1_367_130_551; // Q31
    const ATAN_COEFF_A03: i32 = -715_791_936; // Q31
    const ATAN_COEFF_A05: i32 = 857_391_616; // Q32
    const ATAN_COEFF_A07: i32 = -1_200_579_328; // Q33
    const ATAN_COEFF_A09: i32 = 1_682_636_672; // Q34
    const ATAN_COEFF_A11: i32 = -1_985_085_440; // Q35
    const ATAN_COEFF_A13: i32 = 1_583_306_112; // Q36
    const ATAN_COEFF_A15: i32 = -598_602_432; // Q37
    // The expected x is in the range of [-1.0f, 1.0f].
    debug_assert!((-1_073_741_824..=1_073_741_824).contains(&x));
    // If x = 1.0f, returns 0.5f.
    if x == 1_073_741_824 {
        return 536_870_912; // 0.5f (Q30)
    }
    // If x = -1.0f, returns -0.5f.
    if x == -1_073_741_824 {
        return -536_870_912; // -0.5f (Q30)
    }
    let x_q31 = shl32(x, 1);
    let x_sq_q30 = mult32_32_q31(x_q31, x);
    // Split evaluation in steps to avoid exploding macro expansion.
    let mut tmp = mult32_32_q31(x_sq_q30, ATAN_COEFF_A15);
    tmp = mult32_32_q31(x_sq_q30, add32(ATAN_COEFF_A13, tmp));
    tmp = mult32_32_q31(x_sq_q30, add32(ATAN_COEFF_A11, tmp));
    tmp = mult32_32_q31(x_sq_q30, add32(ATAN_COEFF_A09, tmp));
    tmp = mult32_32_q31(x_sq_q30, add32(ATAN_COEFF_A07, tmp));
    tmp = mult32_32_q31(x_sq_q30, add32(ATAN_COEFF_A05, tmp));
    tmp = mult32_32_q31(x_sq_q30, add32(ATAN_COEFF_A03, tmp));
    tmp = add32(x, mult32_32_q31(x_q31, tmp));
    mult32_32_q31(ATAN_2_OVER_PI, tmp)
}

/// Port of `celt_atan2p_norm` (fixed): `atan2(y,x)*2/pi` in Q30 for Q30 inputs `x, y >= 0`.
#[must_use]
pub fn celt_atan2p_norm(y: OpusVal32, x: OpusVal32) -> OpusVal32 {
    debug_assert!(x >= 0 && y >= 0);
    if y == 0 && x == 0 {
        0
    } else if y < x {
        celt_atan_norm(shr32(frac_div32(y, x), 1))
    } else {
        debug_assert!(y > 0);
        1_073_741_824 /* 1.0f Q30 */ - celt_atan_norm(shr32(frac_div32(x, y), 1))
    }
}

/// Port of `celt_atan01`: atan approximation using a 4th order polynomial. Input is in Q15
/// format and normalized by pi/4. Output is in Q15 format.
#[must_use]
pub fn celt_atan01(x: OpusVal16) -> OpusVal16 {
    const M1: i32 = 32767;
    const M2: i32 = -21;
    const M3: i32 = -11943;
    const M4: i32 = 4936;
    mult16_16_p15(
        x,
        add32(
            M1,
            mult16_16_p15(
                x,
                add32(M2, mult16_16_p15(x, add32(M3, mult16_16_p15(M4, x)))),
            ),
        ),
    ) as i16
}

/// Port of `celt_atan2p`: `atan2()` approximation valid for positive input values.
#[must_use]
pub fn celt_atan2p(y: OpusVal16, x: OpusVal16) -> OpusVal16 {
    if x == 0 && y == 0 {
        0
    } else if y < x {
        let mut arg = celt_div(shl32(y, 15), i32::from(x));
        if arg >= 32767 {
            arg = 32767;
        }
        shr16(celt_atan01(arg as i16), 1) as i16
    } else {
        let mut arg = celt_div(shl32(x, 15), i32::from(y));
        if arg >= 32767 {
            arg = 32767;
        }
        (25736 - shr16(celt_atan01(arg as i16), 1)) as i16
    }
}
