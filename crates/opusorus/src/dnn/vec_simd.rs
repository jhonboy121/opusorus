//! Vertical SIMD (`fearless_simd`, see [`crate::simd`]) versions of the `dnn/vec.h` int8
//! products, bit-identical to the scalar ports in [`super::vec`].
//!
//! Both `cgemv8x4` and `sparse_cgemv8x4` add, per 8x4 weight block, the four-term dot product
//! `w[4k]*x[0] + ... + w[4k+3]*x[3]` of each row `k` to the float output `y[k]`. The products
//! of two int8 values and their sums are integers of magnitude at most `4*128*128 = 65536`, so
//! the scalar code computes them exactly whether in `int` (`sparse_cgemv8x4`) or in float
//! (`cgemv8x4`: every partial sum is an integer below 2^24). Here the eight row sums of a block
//! are computed exactly in integer lanes, converted to float (exact) and added to `y` in the
//! same block order as the scalar code.
//!
//! On aarch64 the block uses the widening NEON instructions directly (`smull` i8*i8 -> i16,
//! `saddlp` pairwise i16 -> i32, `addp`; through `fearless_simd::kernel!`, safe code); other SIMD
//! targets use the portable version (i8 -> i16 products, exact since `|w*x| <= 16384`, widened
//! to i32 and summed pairwise).

use fearless_simd::{Bytes, Simd, SimdBase, f32x4, i8x16, i16x8, i32x4, u32x4};
#[cfg(target_arch = "aarch64")]
use fearless_simd::{Neon, SimdInto};

/// The four quantized inputs `x[0..4]` as one native-endian word (their memory layout).
#[inline(always)]
const fn x_word(x: &[i8; 4]) -> u32 {
    u32::from_ne_bytes([x[0] as u8, x[1] as u8, x[2] as u8, x[3] as u8])
}

/// `[a0+a1, a2+a3, b0+b1, b2+b3]`.
#[inline(always)]
fn pair_add<S: Simd>(s: S, a: i32x4<S>, b: i32x4<S>) -> i32x4<S> {
    let (even, odd) = s.deinterleave_i32x4(a, b);
    even + odd
}

/// Row sums of four rows held two per i16x8 (`ab` = products of rows 0,1 and `cd` = of rows
/// 2,3, four each): `[sum0, sum1, sum2, sum3]`.
#[inline(always)]
fn row_sums4<S: Simd>(s: S, ab: i16x8<S>, cd: i16x8<S>) -> i32x4<S> {
    let (r0, r1) = s.widen_i16x8(ab);
    let (r2, r3) = s.widen_i16x8(cd);
    pair_add(s, pair_add(s, r0, r1), pair_add(s, r2, r3))
}

/// Portable 8x4 block: returns `y0 + sums(rows 0..4)`, `y1 + sums(rows 4..8)` of the block `wb`
/// (32 weights, 4 per row) with the inputs `x[0..4]`.
#[inline(always)]
fn block<S: Simd>(
    s: S,
    wb: &[i8; 32],
    x: &[i8; 4],
    y0: f32x4<S>,
    y1: f32x4<S>,
) -> (f32x4<S>, f32x4<S>) {
    // [x0, x1, x2, x3, x0, x1, x2, x3] as i16.
    let bytes: i8x16<S> = u32x4::splat(s, x_word(x)).bitcast();
    let xpat = s.widen_i8x16(bytes).0;
    let (w01, w23) = s.widen_i8x16(i8x16::from_slice(s, &wb[..16]));
    let (w45, w67) = s.widen_i8x16(i8x16::from_slice(s, &wb[16..32]));
    let lo = row_sums4(s, w01 * xpat, w23 * xpat);
    let hi = row_sums4(s, w45 * xpat, w67 * xpat);
    (y0 + s.cvt_f32_i32x4(lo), y1 + s.cvt_f32_i32x4(hi))
}

#[cfg(target_arch = "aarch64")]
fearless_simd::kernel!(
    /// NEON 8x4 block, same results as [`block`].
    #[inline(always)]
    fn block_neon(
        neon: Neon,
        wb: &[i8; 32],
        x: &[i8; 4],
        y0: f32x4<Neon>,
        y1: f32x4<Neon>,
    ) -> (f32x4<Neon>, f32x4<Neon>) {
        use core::arch::aarch64::*;
        // [x0, x1, x2, x3, x0, x1, x2, x3]
        let xv = vreinterpret_s8_u32(vdup_n_u32(x_word(x)));
        let w0: int8x16_t = i8x16::from_slice(neon, &wb[..16]).into();
        let w1: int8x16_t = i8x16::from_slice(neon, &wb[16..32]).into();
        // Products of rows (0,1), (2,3), (4,5), (6,7), pairwise sums in i32, then row sums.
        let p01 = vpaddlq_s16(vmull_s8(vget_low_s8(w0), xv));
        let p23 = vpaddlq_s16(vmull_s8(vget_high_s8(w0), xv));
        let p45 = vpaddlq_s16(vmull_s8(vget_low_s8(w1), xv));
        let p67 = vpaddlq_s16(vmull_s8(vget_high_s8(w1), xv));
        let lo = vcvtq_f32_s32(vpaddq_s32(p01, p23));
        let hi = vcvtq_f32_s32(vpaddq_s32(p45, p67));
        let y0: float32x4_t = y0.into();
        let y1: float32x4_t = y1.into();
        (
            vaddq_f32(y0, lo).simd_into(neon),
            vaddq_f32(y1, hi).simd_into(neon),
        )
    }
);

/// Where the 4-column blocks of a row group come from.
#[derive(Clone, Copy)]
enum Cols<'a> {
    /// Dense: all `cols / 4` blocks in order.
    Dense(usize),
    /// Sparse: per 8-row group, a count followed by the starting columns (`idx`).
    Sparse(&'a [i32]),
}

/// The row-group/block loops shared by both products; `blk` adds one block to `(y0, y1)`.
#[inline(always)]
fn gemv<S: Simd>(
    s: S,
    out: &mut [f32],
    w: &[i8],
    rows: usize,
    cols: Cols<'_>,
    xq: &[i8],
    blk: impl Fn(&[i8; 32], &[i8; 4], f32x4<S>, f32x4<S>) -> (f32x4<S>, f32x4<S>),
) {
    let ys = out[..rows].as_chunks_mut::<8>().0;
    let wblocks = w.as_chunks::<32>().0;
    match cols {
        Cols::Dense(cols) => {
            let xs = xq[..cols].as_chunks::<4>().0;
            let nb = xs.len();
            if nb == 0 {
                return;
            }
            // The weights of each group of 8 rows (`nb` blocks each).
            let mut wgroups = wblocks[..ys.len() * nb].chunks_exact(nb);
            // Four row groups at a time: independent float accumulation chains (each row still
            // adds its blocks in order), which hides the add latency.
            let (ys4, ys_rest) = ys.as_chunks_mut::<4>();
            for y in ys4 {
                let (Some(w0), Some(w1), Some(w2), Some(w3)) = (
                    wgroups.next(),
                    wgroups.next(),
                    wgroups.next(),
                    wgroups.next(),
                ) else {
                    return;
                };
                let mut acc: [(f32x4<S>, f32x4<S>); 4] = core::array::from_fn(|g| {
                    (
                        f32x4::from_slice(s, &y[g][..4]),
                        f32x4::from_slice(s, &y[g][4..]),
                    )
                });
                for (jb, xj) in xs.iter().enumerate() {
                    acc[0] = blk(&w0[jb], xj, acc[0].0, acc[0].1);
                    acc[1] = blk(&w1[jb], xj, acc[1].0, acc[1].1);
                    acc[2] = blk(&w2[jb], xj, acc[2].0, acc[2].1);
                    acc[3] = blk(&w3[jb], xj, acc[3].0, acc[3].1);
                }
                for (yg, (a0, a1)) in y.iter_mut().zip(acc) {
                    a0.store_slice(&mut yg[..4]);
                    a1.store_slice(&mut yg[4..]);
                }
            }
            for (y, w) in ys_rest.iter_mut().zip(wgroups) {
                let (mut y0, mut y1) =
                    (f32x4::from_slice(s, &y[..4]), f32x4::from_slice(s, &y[4..]));
                for (xj, wb) in xs.iter().zip(w) {
                    (y0, y1) = blk(wb, xj, y0, y1);
                }
                y0.store_slice(&mut y[..4]);
                y1.store_slice(&mut y[4..]);
            }
        }
        Cols::Sparse(idx) => {
            let mut ip = 0usize;
            let mut wblocks = wblocks.iter();
            for y in ys {
                let (mut y0, mut y1) =
                    (f32x4::from_slice(s, &y[..4]), f32x4::from_slice(s, &y[4..]));
                let colblocks = idx[ip] as usize;
                let pos = &idx[ip + 1..ip + 1 + colblocks];
                ip += 1 + colblocks;
                for (&p, wb) in pos.iter().zip(wblocks.by_ref().take(colblocks)) {
                    let p = p as usize;
                    let xj = &xq[p..p + 4].as_chunks::<4>().0[0];
                    (y0, y1) = blk(wb, xj, y0, y1);
                }
                y0.store_slice(&mut y[..4]);
                y1.store_slice(&mut y[4..]);
            }
        }
    }
}

/// Runs [`gemv`] with the best block kernel of the target. Returns `false` without SIMD, and
/// also for inconsistent sizes (too few weights), which the scalar code then reports by
/// panicking on the out-of-bounds access as before.
#[inline(always)]
fn gemv_any(out: &mut [f32], w: &[i8], rows: usize, cols: Cols<'_>, xq: &[i8]) -> bool {
    let blocks = match cols {
        Cols::Dense(cols) => (rows / 8) * (cols / 4),
        Cols::Sparse(idx) => {
            let mut ip = 0usize;
            let mut n = 0usize;
            for _ in 0..rows / 8 {
                let Some(&c) = idx.get(ip) else { return false };
                n += c as usize;
                ip += 1 + c as usize;
            }
            n
        }
    };
    if w.len() < 32 * blocks {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = neon_token!() {
        neon.vectorize(
            #[inline(always)]
            || {
                gemv(neon, out, w, rows, cols, xq, |wb, x, y0, y1| {
                    block_neon(neon, wb, x, y0, y1)
                })
            },
        );
        return true;
    }
    with_simd!(s => gemv(s, out, w, rows, cols, xq, |wb, x, y0, y1| block(s, wb, x, y0, y1)))
        .is_some()
}

/// Block loop of `cgemv8x4` (output rows `0..rows` start at zero, `xq` holds the `cols`
/// quantized inputs). Returns `false` without SIMD (nothing written).
#[inline]
pub(crate) fn cgemv8x4(out: &mut [f32], w: &[i8], rows: usize, cols: usize, xq: &[i8]) -> bool {
    gemv_any(out, w, rows, Cols::Dense(cols), xq)
}

/// Block loop of `sparse_cgemv8x4` (output rows start at zero). Returns `false` without SIMD
/// (nothing written).
#[inline]
pub(crate) fn sparse_cgemv8x4(
    out: &mut [f32],
    w: &[i8],
    idx: &[i32],
    rows: usize,
    xq: &[i8],
) -> bool {
    gemv_any(out, w, rows, Cols::Sparse(idx), xq)
}

/// `sgemv16x1` / `sgemv8x1` (`N` vectors = `4*N` rows per block): for each block of rows,
/// `y[k] += w[j*col_stride + i + k] * x[j]` for `j = 0..cols` in order, one multiply and one
/// add per lane as in the scalar code. `out[..rows]` must be zero. Returns `false` without SIMD
/// (nothing written).
#[inline]
pub(crate) fn sgemv_blocks<const N: usize>(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) -> bool {
    with_simd!(s => sgemv_simd::<_, N>(s, out, weights, rows, cols, col_stride, x)).is_some()
}

#[inline(always)]
fn sgemv_simd<S: Simd, const N: usize>(
    s: S,
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    let b = 4 * N;
    let x = &x[..cols];
    for (blk, y) in out[..rows].chunks_exact_mut(b).enumerate() {
        let i = blk * b;
        let mut acc: [f32x4<S>; N] =
            core::array::from_fn(|v| f32x4::from_slice(s, &y[4 * v..4 * v + 4]));
        for (j, &xj) in x.iter().enumerate() {
            let w = &weights[j * col_stride + i..][..b];
            let xv = f32x4::splat(s, xj);
            for (v, a) in acc.iter_mut().enumerate() {
                *a += f32x4::from_slice(s, &w[4 * v..4 * v + 4]) * xv;
            }
        }
        for (v, a) in acc.iter().enumerate() {
            a.store_slice(&mut y[4 * v..4 * v + 4]);
        }
    }
}

#[cfg(test)]
mod tests {
    //! SIMD == scalar for both int8 products on random blocks (all targets).

    use alloc::{format, vec::Vec};

    use crate::dnn::vec::{cgemv8x4, sgemv, sparse_cgemv8x4};
    use crate::simd::assert_simd_eq_scalar;

    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            self.0 >> 8
        }
        fn f(&mut self) -> f32 {
            // Includes values that quantize to -128 and wrap (|x| slightly above 1).
            (self.next() % 2200) as f32 / 1000.0 - 1.1
        }
    }

    #[test]
    fn sgemv_matches_scalar() {
        let mut rng = Lcg(9);
        for (rows, cols, stride) in [
            (16, 3, 16),
            (32, 40, 48),
            (8, 17, 8),
            (24, 5, 32),
            (7, 4, 7),
        ] {
            let w: Vec<f32> = (0..cols * stride).map(|_| rng.f() * 3.0).collect();
            let x: Vec<f32> = (0..cols).map(|_| rng.f() * 100.0).collect();
            assert_simd_eq_scalar(&format!("sgemv {rows}x{cols}"), || {
                let mut out = alloc::vec![1f32; rows];
                sgemv(&mut out, &w, rows, cols, stride, &x);
                out.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            });
        }
    }

    #[test]
    fn int8_products_match_scalar() {
        let mut rng = Lcg(7);
        for (rows, cols) in [(8, 4), (16, 32), (64, 128), (24, 260)] {
            let w: Vec<i8> = (0..rows * cols).map(|_| rng.next() as i8).collect();
            let scale: Vec<f32> = (0..rows).map(|_| rng.f()).collect();
            let x: Vec<f32> = (0..cols).map(|_| rng.f()).collect();
            assert_simd_eq_scalar(&format!("cgemv8x4 {rows}x{cols}"), || {
                let mut out = alloc::vec![0f32; rows];
                cgemv8x4(&mut out, &w, &scale, rows, cols, &x);
                out.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            });
            // Sparse: a random subset of 4-column blocks per 8-row group.
            let mut idx = Vec::new();
            let mut nblocks = 0;
            for _ in 0..rows / 8 {
                let blocks: Vec<i32> = (0..cols / 4)
                    .filter(|_| !rng.next().is_multiple_of(3))
                    .map(|b| 4 * b as i32)
                    .collect();
                idx.push(blocks.len() as i32);
                nblocks += blocks.len();
                idx.extend(blocks);
            }
            let ws: Vec<i8> = (0..32 * nblocks).map(|_| rng.next() as i8).collect();
            assert_simd_eq_scalar(&format!("sparse_cgemv8x4 {rows}x{cols}"), || {
                let mut out = alloc::vec![0f32; rows];
                sparse_cgemv8x4(&mut out, &ws, &idx, &scale, rows, cols, &x);
                out.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            });
        }
    }
}
