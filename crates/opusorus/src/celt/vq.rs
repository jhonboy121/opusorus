//! Port of celt/vq.c, celt/vq.h: pyramid vector quantisation (PVQ) of the normalised band
//! shapes, spreading rotations and (QEXT) the extra-resolution / cubic quantisers.
//!
//! The float build is ported. `norm_scaleup`/`norm_scaledown` and `celt_inner_prod_norm*` are
//! no-ops / aliases of `celt_inner_prod` in the float build, so they have no separate port.
//! The SSE override of `op_pvq_search` is not used by the oracle and is dropped.

#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]

use alloc::vec::Vec;

use crate::celt::arch::{
    CeltNorm, EPSILON, OpusVal16, OpusVal32, Q15_ONE, Q15ONE, abs16, add16, add32, extend32,
    extract16, half16, mac16_16, mult16_16, mult16_16_q15, mult16_32_q15, mult16_32_q16,
    mult32_32_q31, neg16, pshr32, qconst16, shr32, sub16, sub32, vshr32,
};
#[cfg(feature = "qext")]
use crate::celt::arch::{abs32, imax, imin};
use crate::celt::bands::SPREAD_NONE;
use crate::celt::cwrs::{decode_pulses, encode_pulses};
use crate::celt::entcode::celt_udiv;
use crate::celt::entdec::EcDec;
use crate::celt::entenc::EcEnc;
use crate::celt::mathops::{
    celt_atan2p_norm, celt_cos_norm, celt_div, celt_rcp, celt_rsqrt_norm, celt_rsqrt_norm32,
    celt_sqrt32,
};
use crate::celt::pitch::celt_inner_prod;
use crate::math;

/// Largest band size of the static modes (`M*(eBands[21]-eBands[20])` at LM=3). Scratch arrays
/// of this size live on the stack; larger (custom mode) bands fall back to the heap.
pub(crate) const MAX_BAND_SIZE: usize = 176;

/// Scratch storage replacing a C VLA: on the stack up to `N` elements, heap beyond (only
/// reachable with custom modes), so no input size can make the port panic where C would not.
pub(crate) struct Scratch<T, const N: usize> {
    stack: [T; N],
    heap: Vec<T>,
}

impl<T: Copy + Default, const N: usize> Scratch<T, N> {
    /// Creates empty scratch storage (no allocation).
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            stack: [T::default(); N],
            heap: Vec::new(),
        }
    }

    /// Returns a buffer of exactly `len` elements. Contents are stale (C VLAs are
    /// uninitialised); callers write before reading, exactly like the C code.
    #[inline(always)]
    pub(crate) fn get(&mut self, len: usize) -> &mut [T] {
        if len <= N {
            &mut self.stack[..len]
        } else {
            self.heap.resize(len, T::default());
            &mut self.heap[..len]
        }
    }
}

/// Port of celt/vq.c:exp_rotation1.
pub const fn exp_rotation1(x: &mut [CeltNorm], len: i32, stride: i32, c: OpusVal16, s: OpusVal16) {
    let ms = neg16(s);
    let st = stride as usize;
    // FIXED_POINT: norm_scaledown is a no-op in the float build.
    let mut i: i32 = 0;
    while i < len - stride {
        let p = i as usize;
        let x1 = x[p];
        let x2 = x[p + st];
        x[p + st] = extract16(pshr32(mac16_16(mult16_16(c, x2), s, x1), 15));
        x[p] = extract16(pshr32(mac16_16(mult16_16(c, x1), ms, x2), 15));
        i += 1;
    }
    let mut i: i32 = len - 2 * stride - 1;
    while i >= 0 {
        let p = i as usize;
        let x1 = x[p];
        let x2 = x[p + st];
        x[p + st] = extract16(pshr32(mac16_16(mult16_16(c, x2), s, x1), 15));
        x[p] = extract16(pshr32(mac16_16(mult16_16(c, x1), ms, x2), 15));
        i -= 1;
    }
    // FIXED_POINT: norm_scaleup is a no-op in the float build.
}

/// Port of celt/vq.c:exp_rotation.
///
/// Applies (`dir = 1`) or undoes (`dir = -1`) the spreading rotation on `x[..len]`, which is
/// made of `stride` interleaved blocks.
pub fn exp_rotation(x: &mut [CeltNorm], mut len: i32, dir: i32, stride: i32, k: i32, spread: i32) {
    const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
    let mut stride2: i32 = 0;

    if 2 * k >= len || spread == SPREAD_NONE {
        return;
    }
    let factor = SPREAD_FACTOR[(spread - 1) as usize];

    let gain: OpusVal16 = celt_div(mult16_16(Q15_ONE, len as f32), (len + factor * k) as f32);
    let theta: OpusVal16 = half16(mult16_16_q15(gain, gain));

    let c: OpusVal16 = celt_cos_norm(extend32(theta));
    let s: OpusVal16 = celt_cos_norm(extend32(sub16(Q15ONE, theta))); // sin(theta)

    if len >= 8 * stride {
        stride2 = 1;
        // This is just a simple (equivalent) way of computing sqrt(len/stride) with rounding.
        // It's basically incrementing long as (stride2+0.5)^2 < len/stride.
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
            stride2 += 1;
        }
    }
    len = celt_udiv(len as u32, stride as u32) as i32;
    let l = len as usize;
    for i in 0..stride as usize {
        let xi = &mut x[i * l..(i + 1) * l];
        if dir < 0 {
            if stride2 != 0 {
                exp_rotation1(xi, len, stride2, s, c);
            }
            exp_rotation1(xi, len, 1, c, s);
        } else {
            exp_rotation1(xi, len, 1, c, -s);
            if stride2 != 0 {
                exp_rotation1(xi, len, stride2, s, -c);
            }
        }
    }
}

/// Port of celt/vq.c:normalise_residual: normalises the decoded integer PVQ codeword to unit
/// norm (times `gain`). `_shift` is only used by the fixed-point QEXT build.
pub fn normalise_residual(
    iy: &[i32],
    x: &mut [CeltNorm],
    n: i32,
    ryy: OpusVal32,
    gain: OpusVal32,
    _shift: i32,
) {
    // FIXED_POINT: k = celt_ilog2(Ryy)>>1 (not ported, float build).
    let t: OpusVal32 = vshr32(ryy, 0);
    let g: OpusVal32 = mult32_32_q31(celt_rsqrt_norm32(t), gain);
    let n = n as usize;
    for (xi, &yi) in x[..n].iter_mut().zip(&iy[..n]) {
        *xi = vshr32(mult16_32_q15(yi as f32, g), 0);
    }
}

/// Port of celt/vq.c:extract_collapse_mask.
pub fn extract_collapse_mask(iy: &[i32], n: i32, b: i32) -> u32 {
    if b <= 1 {
        return 1;
    }
    // NOTE: As a minor optimization, we could be passing around log2(B), not B, for both this
    // and for exp_rotation().
    let n0 = celt_udiv(n as u32, b as u32) as usize;
    let mut collapse_mask: u32 = 0;
    for i in 0..b as usize {
        let mut tmp: u32 = 0;
        for &v in &iy[i * n0..(i + 1) * n0] {
            tmp |= v as u32;
        }
        collapse_mask |= u32::from(tmp != 0) << i;
    }
    collapse_mask
}

/// Port of celt/vq.c:op_pvq_search_c.
///
/// Finds the `k`-pulse integer vector `iy[..n]` closest (in angle) to `x[..n]`. `x` is replaced
/// by its absolute value. Returns the squared norm of `iy`.
pub fn op_pvq_search_c(x: &mut [CeltNorm], iy: &mut [i32], k: i32, n: i32) -> OpusVal16 {
    let nu = n as usize;
    let x = &mut x[..nu];
    let iy = &mut iy[..nu];
    let mut y_buf = Scratch::<CeltNorm, MAX_BAND_SIZE>::new();
    let mut signx_buf = Scratch::<i32, MAX_BAND_SIZE>::new();
    let y = y_buf.get(nu);
    let signx = signx_buf.get(nu);
    // FIXED_POINT: input scaling not ported (float build).

    // Get rid of the sign
    let mut sum: OpusVal32 = 0.0;
    for j in 0..nu {
        signx[j] = i32::from(x[j] < 0.0);
        // OPT: Make sure the compiler doesn't use a branch on ABS16().
        x[j] = abs16(x[j]);
        iy[j] = 0;
        y[j] = 0.0;
    }

    let mut xy: OpusVal32 = 0.0;
    let mut yy: OpusVal16 = 0.0;

    let mut pulses_left = k;

    // Do a pre-search by projecting on the pyramid
    if k > (n >> 1) {
        for &v in x.iter() {
            sum += v;
        }

        // If X is too small, just replace it with a pulse at 0.
        // Prevents infinities and NaNs from causing too many pulses to be allocated. 64 is an
        // approximation of infinity here.
        if !(sum > EPSILON && sum < 64.0) {
            x[0] = qconst16(1.0, 14);
            for v in x[1..].iter_mut() {
                *v = 0.0;
            }
            sum = qconst16(1.0, 14);
        }
        // Using K+e with e < 1 guarantees we cannot get more than K pulses.
        let rcp: OpusVal16 = extract16(mult16_32_q16(k as f32 + 0.8, celt_rcp(sum)));
        for j in 0..nu {
            // C: `(int)floor(rcp*X[j])` (double floor of a float product).
            iy[j] = math::floor((rcp * x[j]) as f64) as i32;
            y[j] = iy[j] as CeltNorm;
            yy = mac16_16(yy, y[j], y[j]);
            xy = mac16_16(xy, x[j], y[j]);
            y[j] *= 2.0;
            pulses_left -= iy[j];
        }
    }
    debug_assert!(pulses_left >= 0);

    // This should never happen, but just in case it does (e.g. on silence) we fill the first
    // bin with pulses.
    if pulses_left > n + 3 {
        let tmp = pulses_left as OpusVal16;
        yy = mac16_16(yy, tmp, tmp);
        yy = mac16_16(yy, tmp, y[0]);
        iy[0] += pulses_left;
        pulses_left = 0;
    }

    for _ in 0..pulses_left {
        let mut best_id: usize = 0;
        // The squared magnitude term gets added anyway, so we might as well add it outside the
        // loop.
        yy = add16(yy, 1.0);

        // Calculations for position 0 are out of the loop, in part to reduce mispredicted
        // branches (since the if condition is usually false) in the loop.
        // Temporary sums of the new pulse(s)
        let mut rxy: OpusVal16 = extract16(shr32(add32(xy, extend32(x[0])), 0));
        // We're multiplying y[j] by two so we don't have to do it here
        let mut ryy: OpusVal16 = add16(yy, y[0]);

        // Approximate score: we maximise Rxy/sqrt(Ryy) (we're guaranteed that Rxy is positive
        // because the sign is pre-computed)
        rxy = mult16_16_q15(rxy, rxy);
        let mut best_den: OpusVal16 = ryy;
        let mut best_num: OpusVal32 = rxy;
        for j in 1..nu {
            // Temporary sums of the new pulse(s)
            rxy = extract16(shr32(add32(xy, extend32(x[j])), 0));
            // We're multiplying y[j] by two so we don't have to do it here
            ryy = add16(yy, y[j]);

            // Approximate score: we maximise Rxy/sqrt(Ryy) (we're guaranteed that Rxy is
            // positive because the sign is pre-computed)
            rxy = mult16_16_q15(rxy, rxy);
            // The idea is to check for num/den >= best_num/best_den, but that way we can do it
            // without any division
            if mult16_16(best_den, rxy) > mult16_16(ryy, best_num) {
                best_den = ryy;
                best_num = rxy;
                best_id = j;
            }
        }

        // Updating the sums of the new pulse(s)
        xy = add32(xy, extend32(x[best_id]));
        // We're multiplying y[j] by two so we don't have to do it here
        yy = add16(yy, y[best_id]);

        // Only now that we've made the final choice, update y/iy
        // Multiplying y[j] by 2 so we don't have to do it everywhere else
        y[best_id] += 2.0;
        iy[best_id] += 1;
    }

    // Put the original sign back
    for j in 0..nu {
        // OPT: The is more likely to be compiled without a branch than the code above but has
        // the same performance otherwise.
        iy[j] = (iy[j] ^ -signx[j]) + signx[j];
    }
    yy
}

/// `op_pvq_search` (no SIMD override in the oracle build).
#[inline(always)]
pub fn op_pvq_search(x: &mut [CeltNorm], iy: &mut [i32], k: i32, n: i32) -> OpusVal16 {
    op_pvq_search_c(x, iy, k, n)
}

/// Port of celt/vq.c:op_pvq_search_N2 (QEXT): PVQ search with extra resolution for `N = 2`.
#[cfg(feature = "qext")]
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub fn op_pvq_search_n2(
    x: &[CeltNorm],
    iy: &mut [i32],
    up_iy: &mut [i32],
    k: i32,
    up: i32,
    refine: &mut i32,
    _shift: i32,
) -> OpusVal32 {
    let sum: OpusVal32 = abs32(x[0]) + abs32(x[1]);
    if sum < EPSILON {
        iy[0] = k;
        up_iy[0] = up * k;
        iy[1] = 0;
        up_iy[1] = 0;
        *refine = 0;
        // C: `K*(float)K*up*up`.
        return k as f32 * k as f32 * up as f32 * up as f32;
    }
    // FIXED_POINT: not ported (float build).
    let rcp_sum: OpusVal32 = 1.0 / sum;
    iy[0] = math::floor((0.5f32 + k as f32 * x[0] * rcp_sum) as f64) as i32;
    up_iy[0] = math::floor((0.5f32 + (up * k) as f32 * x[0] * rcp_sum) as f64) as i32;
    up_iy[0] = imax(
        up * iy[0] - (up - 1) / 2,
        imin(up * iy[0] + (up - 1) / 2, up_iy[0]),
    );
    let mut offset = up_iy[0] - up * iy[0];
    iy[1] = k - iy[0].abs();
    up_iy[1] = up * k - up_iy[0].abs();
    if x[1] < 0.0 {
        iy[1] = -iy[1];
        up_iy[1] = -up_iy[1];
        offset = -offset;
    }
    *refine = offset;
    // C: `up_iy[0]*(opus_val64)up_iy[0] + up_iy[1]*(opus_val64)up_iy[1]` (opus_val64 = float).
    up_iy[0] as f32 * up_iy[0] as f32 + up_iy[1] as f32 * up_iy[1] as f32
}

/// Port of celt/vq.c:op_pvq_refine (QEXT). `iy0 = None` means the C call aliases `iy0` with
/// `iy` (`op_pvq_refine(Xn, iy, iy, ...)`). Returns true on failure.
#[cfg(feature = "qext")]
pub fn op_pvq_refine(
    xn: &[OpusVal32],
    iy: &mut [i32],
    iy0: Option<&[i32]>,
    k: i32,
    up: i32,
    margin: i32,
    n: i32,
) -> bool {
    let nu = n as usize;
    let mut rounding_buf = Scratch::<OpusVal32, MAX_BAND_SIZE>::new();
    let rounding = rounding_buf.get(nu);
    let mut iysum: i32 = 0;
    for i in 0..nu {
        let tmp: OpusVal32 = mult32_32_q31(k as f32, xn[i]);
        // C: `(int)floor(.5+tmp)` (double arithmetic).
        iy[i] = math::floor(0.5 + tmp as f64) as i32;
        rounding[i] = tmp - iy[i] as f32;
    }
    if let Some(iy0) = iy0 {
        for i in 0..nu {
            iy[i] = imin(up * iy0[i] + up - 1, imax(up * iy0[i] - up + 1, iy[i]));
        }
    }
    for &v in iy[..nu].iter() {
        iysum += v;
    }
    if (iysum - k).abs() > 32 {
        return true;
    }
    let dir: i32 = if iysum < k { 1 } else { -1 };
    while iysum != k {
        let mut roundval: OpusVal32 = (-1000000 * dir) as f32;
        let mut roundpos: usize = 0;
        for i in 0..nu {
            let y0 = match iy0 {
                Some(iy0) => iy0[i],
                None => iy[i],
            };
            if (rounding[i] - roundval) * dir as f32 > 0.0
                && (iy[i] - up * y0).abs() < (margin - 1)
                && !(dir == -1 && iy[i] == 0)
            {
                roundval = rounding[i];
                roundpos = i;
            }
        }
        iy[roundpos] += dir;
        rounding[roundpos] -= dir as f32;
        iysum += dir;
    }
    false
}

/// Port of celt/vq.c:op_pvq_search_extra (QEXT): PVQ search with extra resolution.
#[cfg(feature = "qext")]
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub fn op_pvq_search_extra(
    x: &[CeltNorm],
    iy: &mut [i32],
    up_iy: &mut [i32],
    k: i32,
    up: i32,
    refine: &mut [i32],
    n: i32,
    _shift: i32,
) -> OpusVal32 {
    let nu = n as usize;
    let mut sum: OpusVal32 = 0.0;
    let mut failed = false;
    let mut yy: f32 = 0.0;
    for &v in &x[..nu] {
        sum += abs32(v);
    }
    let mut xn_buf = Scratch::<OpusVal32, MAX_BAND_SIZE>::new();
    let xn = xn_buf.get(nu);
    if sum < EPSILON {
        failed = true;
    } else {
        // FIXED_POINT: not ported (float build).
        let rcp_sum: OpusVal32 = celt_rcp(sum);
        for i in 0..nu {
            xn[i] = abs32(x[i]) * rcp_sum;
        }
    }
    failed = failed || op_pvq_refine(xn, iy, None, k, 1, k + 1, n);
    failed = failed || op_pvq_refine(xn, up_iy, Some(&*iy), up * k, up, up, n);
    if failed {
        iy[0] = k;
        for v in iy[1..nu].iter_mut() {
            *v = 0;
        }
        up_iy[0] = up * k;
        for v in up_iy[1..nu].iter_mut() {
            *v = 0;
        }
    }
    for i in 0..nu {
        yy += up_iy[i] as f32 * up_iy[i] as f32;
        if x[i] < 0.0 {
            iy[i] = -iy[i];
            up_iy[i] = -up_iy[i];
        }
        refine[i] = up_iy[i] - up * iy[i];
    }
    yy
}

/// Port of celt/vq.c:ec_enc_refine (QEXT). Takes advantage of the fact that "large" refine
/// values are much less likely than smaller ones.
#[cfg(feature = "qext")]
fn ec_enc_refine(enc: &mut EcEnc<'_>, refine: i32, up: i32, extra_bits: i32, use_entropy: bool) {
    let large = refine.abs() > up / 2;
    enc.enc_bit_logp(large, if use_entropy { 3 } else { 1 });
    if large {
        enc.enc_bits(u32::from(refine < 0), 1);
        enc.enc_bits((refine.abs() - up / 2 - 1) as u32, (extra_bits - 1) as u32);
    } else {
        enc.enc_bits((refine + up / 2) as u32, extra_bits as u32);
    }
}

/// Port of celt/vq.c:ec_dec_refine (QEXT).
#[cfg(feature = "qext")]
const fn ec_dec_refine(dec: &mut EcDec<'_>, up: i32, extra_bits: i32, use_entropy: bool) -> i32 {
    let large = dec.dec_bit_logp(if use_entropy { 3 } else { 1 });
    if large {
        let sign = dec.dec_bits(1);
        let mut refine = (dec.dec_bits((extra_bits - 1) as u32) as i32) + up / 2 + 1;
        if sign != 0 {
            refine = -refine;
        }
        refine
    } else {
        dec.dec_bits(extra_bits as u32) as i32 - up / 2
    }
}

/// `use_entropy` test shared by the QEXT extra-resolution encoder/decoder:
/// `(storage*8 - ec_tell(ext)) > (unsigned)(N-1)*(extra_bits+3)+1` in unsigned arithmetic.
#[cfg(feature = "qext")]
#[inline]
const fn qext_use_entropy(storage: u32, tell: i32, n: i32, extra_bits: i32) -> bool {
    storage.wrapping_mul(8).wrapping_sub(tell as u32)
        > ((n - 1) as u32)
            .wrapping_mul((extra_bits + 3) as u32)
            .wrapping_add(1)
}

/// Port of celt/vq.c:alg_quant: algebraic pulse-vector quantiser.
///
/// The signal `x[..n]` is replaced by the sum of the pitch and a combination of `k` pulses such
/// that its norm is still equal to 1 (times `gain`, only when `resynth`). Returns a mask
/// indicating which of the `b` blocks in the band received pulses. With QEXT, `extra_bits >= 2`
/// codes extra resolution into `ext_enc`.
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub fn alg_quant(
    x: &mut [CeltNorm],
    n: i32,
    k: i32,
    spread: i32,
    b: i32,
    enc: &mut EcEnc<'_>,
    gain: OpusVal32,
    resynth: bool,
    #[cfg(feature = "qext")] ext_enc: &mut EcEnc<'_>,
    #[cfg(feature = "qext")] extra_bits: i32,
) -> u32 {
    debug_assert!(k > 0, "alg_quant() needs at least one pulse");
    debug_assert!(n > 1, "alg_quant() needs at least two dimensions");
    let nu = n as usize;

    // Covers vectorization by up to 4.
    let mut iy_buf = Scratch::<i32, { MAX_BAND_SIZE + 3 }>::new();
    let iy = iy_buf.get(nu + 3);

    exp_rotation(x, n, 1, b, k, spread);

    #[allow(
        unused_labels,
        reason = "the label is only targeted by the qext branches"
    )]
    let collapse_mask: u32 = 'quant: {
        #[cfg(feature = "qext")]
        {
            if n == 2 && extra_bits >= 2 {
                let mut refine: i32 = 0;
                let mut up_iy = [0i32; 2];
                let yy_shift = imax(0, extra_bits - 7);
                let up = (1 << extra_bits) - 1;
                let yy = op_pvq_search_n2(x, iy, &mut up_iy, k, up, &mut refine, yy_shift);
                let cm = extract_collapse_mask(&up_iy, n, b);
                encode_pulses(iy, n, k, enc);
                ext_enc.enc_uint((refine + (up - 1) / 2) as u32, up as u32);
                if resynth {
                    normalise_residual(&up_iy, x, n, yy, gain, yy_shift);
                }
                break 'quant cm;
            } else if extra_bits >= 2 {
                let mut up_iy_buf = Scratch::<i32, MAX_BAND_SIZE>::new();
                let mut refine_buf = Scratch::<i32, MAX_BAND_SIZE>::new();
                let up_iy = up_iy_buf.get(nu);
                let refine = refine_buf.get(nu);
                let yy_shift = imax(0, extra_bits - 7);
                let up = (1 << extra_bits) - 1;
                let yy = op_pvq_search_extra(x, iy, up_iy, k, up, refine, n, yy_shift);
                let cm = extract_collapse_mask(up_iy, n, b);
                encode_pulses(iy, n, k, enc);
                let use_entropy = qext_use_entropy(ext_enc.storage, ext_enc.tell(), n, extra_bits);
                for &r in &refine[..nu - 1] {
                    ec_enc_refine(ext_enc, r, up, extra_bits, use_entropy);
                }
                if iy[nu - 1] == 0 {
                    ext_enc.enc_bits(u32::from(up_iy[nu - 1] < 0), 1);
                }
                if resynth {
                    normalise_residual(up_iy, x, n, yy, gain, yy_shift);
                }
                break 'quant cm;
            }
        }
        let yy = op_pvq_search(x, iy, k, n);
        let cm = extract_collapse_mask(iy, n, b);
        encode_pulses(iy, n, k, enc);
        if resynth {
            normalise_residual(iy, x, n, yy, gain, 0);
        }
        cm
    };

    if resynth {
        exp_rotation(x, n, -1, b, k, spread);
    }
    collapse_mask
}

/// Port of celt/vq.c:alg_unquant: decodes the pulse vector and produces the final normalised
/// signal (times `gain`) in `x[..n]`. Returns the collapse mask of the `b` blocks.
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub fn alg_unquant(
    x: &mut [CeltNorm],
    n: i32,
    k: i32,
    spread: i32,
    b: i32,
    dec: &mut EcDec<'_>,
    gain: OpusVal32,
    #[cfg(feature = "qext")] ext_dec: &mut EcDec<'_>,
    #[cfg(feature = "qext")] extra_bits: i32,
) -> u32 {
    debug_assert!(k > 0, "alg_unquant() needs at least one pulse");
    debug_assert!(n > 1, "alg_unquant() needs at least two dimensions");
    let nu = n as usize;
    let mut iy_buf = Scratch::<i32, MAX_BAND_SIZE>::new();
    let iy = iy_buf.get(nu);
    #[allow(unused_mut, reason = "only reassigned with qext")]
    let mut yy_shift: i32 = 0;
    #[allow(unused_mut, reason = "only reassigned with qext")]
    let mut ryy: OpusVal32 = decode_pulses(iy, n, k, dec);
    #[cfg(feature = "qext")]
    {
        if n == 2 && extra_bits >= 2 {
            yy_shift = imax(0, extra_bits - 7);
            let up = (1 << extra_bits) - 1;
            let refine = ext_dec.dec_uint(up as u32) as i32 - (up - 1) / 2;
            iy[0] *= up;
            iy[1] *= up;
            if iy[1] == 0 {
                iy[1] = if iy[0] > 0 { -refine } else { refine };
                iy[0] += if i64::from(refine) * i64::from(iy[0]) > 0 {
                    -refine
                } else {
                    refine
                };
            } else if iy[1] > 0 {
                iy[0] += refine;
                iy[1] -= refine * if iy[0] > 0 { 1 } else { -1 };
            } else {
                iy[0] -= refine;
                iy[1] -= refine * if iy[0] > 0 { 1 } else { -1 };
            }
            ryy = iy[0] as f32 * iy[0] as f32 + iy[1] as f32 * iy[1] as f32;
        } else if extra_bits >= 2 {
            let mut refine_buf = Scratch::<i32, MAX_BAND_SIZE>::new();
            let refine = refine_buf.get(nu);
            yy_shift = imax(0, extra_bits - 7);
            let up = (1 << extra_bits) - 1;
            let use_entropy = qext_use_entropy(ext_dec.storage, ext_dec.tell(), n, extra_bits);
            for r in refine[..nu - 1].iter_mut() {
                *r = ec_dec_refine(ext_dec, up, extra_bits, use_entropy);
            }
            let sign: bool = if iy[nu - 1] == 0 {
                ext_dec.dec_bits(1) != 0
            } else {
                iy[nu - 1] < 0
            };
            for i in 0..nu - 1 {
                iy[i] = iy[i] * up + refine[i];
            }
            iy[nu - 1] = up * k;
            for i in 0..nu - 1 {
                iy[nu - 1] -= iy[i].abs();
            }
            if sign {
                iy[nu - 1] = -iy[nu - 1];
            }
            let mut yy64: f32 = 0.0;
            for &v in iy[..nu].iter() {
                yy64 += v as f32 * v as f32;
            }
            ryy = yy64;
        }
    }
    normalise_residual(iy, x, n, ryy, gain, yy_shift);
    exp_rotation(x, n, -1, b, k, spread);
    extract_collapse_mask(iy, n, b)
}

/// Port of celt/vq.c:renormalise_vector: scales `x[..n]` to have norm `gain`.
pub fn renormalise_vector(x: &mut [CeltNorm], n: i32, gain: OpusVal32) {
    let nu = n as usize;
    // FIXED_POINT: norm_scaledown / k = celt_ilog2(E)>>1 not ported (float build).
    let e: OpusVal32 = EPSILON + celt_inner_prod(x, x, nu);
    let t: OpusVal32 = vshr32(e, 0);
    let g: OpusVal16 = mult32_32_q31(celt_rsqrt_norm(t), gain);
    for v in x[..nu].iter_mut() {
        *v = extract16(pshr32(mult16_16(g, *v), 0));
    }
}

/// Port of celt/vq.c:stereo_itheta: the angle (Q30, `[0, 2^30]`) between the mid and side
/// energies (`stereo`) or between the energies of `x` and `y`.
#[must_use]
pub fn stereo_itheta(x: &[CeltNorm], y: &[CeltNorm], stereo: bool, n: i32) -> i32 {
    let nu = n as usize;
    let mut emid: OpusVal32 = 0.0;
    let mut eside: OpusVal32 = 0.0;
    if stereo {
        for i in 0..nu {
            let m: CeltNorm = pshr32(add32(x[i], y[i]), 0);
            let s: CeltNorm = pshr32(sub32(x[i], y[i]), 0);
            emid = mac16_16(emid, m, m);
            eside = mac16_16(eside, s, s);
        }
    } else {
        emid += celt_inner_prod(x, x, nu);
        eside += celt_inner_prod(y, y, nu);
    }
    let mid: OpusVal32 = celt_sqrt32(emid);
    let side: OpusVal32 = celt_sqrt32(eside);
    // C: `(int)floor(.5f+65536.f*16384*celt_atan2p_norm(side,mid))`.
    math::floor((0.5f32 + 65536.0f32 * 16384.0 * celt_atan2p_norm(side, mid)) as f64) as i32
}

/// Port of celt/vq.c:cubic_synthesis (QEXT).
#[cfg(feature = "qext")]
pub fn cubic_synthesis(
    x: &mut [CeltNorm],
    iy: &[i32],
    n: i32,
    k: i32,
    face: usize,
    sign: bool,
    gain: OpusVal32,
) {
    let nu = n as usize;
    let mut sum: OpusVal32 = 0.0;
    for i in 0..nu {
        x[i] = ((1 + 2 * iy[i]) - k) as f32;
    }
    x[face] = (if sign { -k } else { k }) as f32;
    for &v in &x[..nu] {
        sum += pshr32(mult16_16(v, v), 0);
    }
    // FIXED_POINT: not ported (float build).
    // C: `mag = 1.f/sqrt(sum);` (double division, stored to float).
    let mag: OpusVal32 = (1.0 / math::sqrt(sum as f64)) as f32;
    for v in x[..nu].iter_mut() {
        *v *= mag * gain;
    }
}

/// Port of celt/vq.c:cubic_quant (QEXT): quantises `x[..n]` on the surface of a cube with
/// `res` bits of resolution per dimension. Returns the collapse mask.
#[cfg(feature = "qext")]
pub fn cubic_quant(
    x: &mut [CeltNorm],
    n: i32,
    res: i32,
    b: i32,
    enc: &mut EcEnc<'_>,
    gain: OpusVal32,
    resynth: bool,
) -> u32 {
    let nu = n as usize;
    let mut face: usize = 0;
    let mut faceval: CeltNorm = -1.0;
    let mut iy_buf = Scratch::<i32, MAX_BAND_SIZE>::new();
    let iy = iy_buf.get(nu);
    let mut k: i32 = 1 << res;
    // Using odd K on transients to avoid adding pre-echo.
    if b != 1 {
        k = imax(1, k - 1);
    }
    if k == 1 {
        if resynth {
            x[..nu].fill(0.0);
        }
        return 0;
    }
    for i in 0..nu {
        if abs32(x[i]) > faceval {
            faceval = abs32(x[i]);
            face = i;
        }
    }
    let sign = x[face] < 0.0;
    enc.enc_uint(face as u32, n as u32);
    enc.enc_bits(u32::from(sign), 1);
    // FIXED_POINT: not ported (float build).
    let norm: OpusVal32 = 0.5f32 * k as f32 / (faceval + EPSILON);
    for i in 0..nu {
        iy[i] = imin(k - 1, math::floor(((x[i] + faceval) * norm) as f64) as i32);
    }
    for i in 0..nu {
        if i != face {
            enc.enc_bits(iy[i] as u32, res as u32);
        }
    }
    if resynth {
        cubic_synthesis(x, iy, n, k, face, sign, gain);
    }
    (1u32 << b) - 1
}

/// Port of celt/vq.c:cubic_unquant (QEXT).
#[cfg(feature = "qext")]
pub fn cubic_unquant(
    x: &mut [CeltNorm],
    n: i32,
    res: i32,
    b: i32,
    dec: &mut EcDec<'_>,
    gain: OpusVal32,
) -> u32 {
    let nu = n as usize;
    let mut iy_buf = Scratch::<i32, MAX_BAND_SIZE>::new();
    let iy = iy_buf.get(nu);
    let mut k: i32 = 1 << res;
    // Using odd K on transients to avoid adding pre-echo.
    if b != 1 {
        k = imax(1, k - 1);
    }
    if k == 1 {
        x[..nu].fill(0.0);
        return 0;
    }
    let face = dec.dec_uint(n as u32) as usize;
    let sign = dec.dec_bits(1) != 0;
    for i in 0..nu {
        if i != face {
            iy[i] = dec.dec_bits(res as u32) as i32;
        }
    }
    iy[face] = 0;
    cubic_synthesis(x, iy, n, k, face, sign, gain);
    (1u32 << b) - 1
}
