//! Differential tests for unit `celt_pitch_lpc` (celt/pitch.c, celt/celt_lpc.c) in the
//! fixed-point build (features `fixed-point` / `fixed-res24`, with or without `qext`) vs the
//! fixed-point C oracle. Every comparison is exact.
//!
//! Inputs stay inside the domain where the C code has no signed overflow (undefined behaviour in
//! C, a panic in the overflow-checked Rust test build): the low-level kernels get amplitudes
//! bounded by their length, the analysis functions get the signals the codec feeds them.
//! `_celt_lpc` is also checked in its 32-bit `OPUS_FAST_INT64 == 0` form (32-bit targets)
//! against a copy of `celt_lpc.c` compiled that way.
//!
//! Run with `cargo test -p opusorus-conformance --features fixed-point --test
//! celt_pitch_lpc_fixed` (also `fixed-res24`, and either with `qext`).

#![cfg(feature = "fixed-point")]

use opusorus::celt::arch::{CeltCoef, CeltSig, OpusVal16, OpusVal32};
use opusorus::celt::{celt_lpc as rl, pitch as rp};
use opusorus_conformance::{Rng, assert_slice_eq, signals};
use opusorus_oracle::celt_pitch_lpc as c;

/// Frame sizes the CELT encoder uses (48 kHz, 2.5..20 ms), plus QEXT 96 kHz sizes.
const FRAME_SIZES: [usize; 6] = [120, 240, 480, 960, 1440, 1920];

/// A test signal in `[-1, 1]` of the given kind.
fn shape(rng: &mut Rng, n: usize, kind: u32) -> Vec<f32> {
    let seed = rng.next_u64();
    match kind % 9 {
        0 => signals::noise(n, 1, 1.0, seed),
        1 => signals::music_like(n, 1, 48000, seed),
        2 => signals::speech_like(n, 1, 48000, seed),
        3 => vec![0.0; n],
        4 => {
            // impulses
            let mut v = vec![0.0; n];
            if n == 0 {
                return v;
            }
            let count = rng.range_i32(1, 6);
            for _ in 0..count {
                let p = rng.range_i32(0, n as i32 - 1) as usize;
                v[p] = if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
            }
            v
        }
        5 => {
            // full-scale square wave
            let period = rng.range_i32(2, 400) as usize;
            (0..n)
                .map(|i| {
                    if (i / period).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect()
        }
        6 => {
            // periodic pulse train with noise (strong pitch)
            let period = rng.range_i32(15, 1000) as usize;
            let mut v = signals::noise(n, 1, 0.05, seed);
            let start = rng.range_i32(0, period as i32 - 1) as usize;
            let mut i = start;
            while i < n {
                v[i] = 0.9;
                i += period;
            }
            v
        }
        7 => {
            // sine
            let f = 0.001 + 0.2 * rng.f32_sym().abs() as f64;
            (0..n).map(|i| (f * i as f64).sin() as f32).collect()
        }
        _ => {
            // very quiet noise
            let a = 10f32.powf(rng.range_i32(-5, -1) as f32);
            signals::noise(n, 1, a, seed)
        }
    }
}

/// Scales a `[-1, 1]` signal to integers of magnitude at most `amp` (`-amp-1` allowed, like
/// two's complement full scale).
fn to_int(v: &[f32], amp: i64) -> Vec<i64> {
    v.iter()
        .map(|&s| {
            ((s as f64) * (amp as f64 + 0.5))
                .round()
                .clamp(-(amp as f64) - 1.0, amp as f64) as i64
        })
        .collect()
}

/// `opus_val16` test signal with magnitude at most `amp` (<= 32767).
fn sig16(rng: &mut Rng, n: usize, kind: u32, amp: i32) -> Vec<OpusVal16> {
    let amp = amp.min(32767);
    let v = shape(rng, n, kind);
    to_int(&v, amp as i64)
        .into_iter()
        .map(|x| x as i16)
        .collect()
}

/// `celt_sig`/`opus_val32` test signal with magnitude at most `amp`.
fn sig32(rng: &mut Rng, n: usize, kind: u32, amp: i32) -> Vec<i32> {
    let v = shape(rng, n, kind);
    to_int(&v, amp as i64)
        .into_iter()
        .map(|x| x as i32)
        .collect()
}

/// Random amplitude of `1..=2^bits - 1`, biased towards all bit widths.
fn rand_amp(rng: &mut Rng, bits: i32) -> i32 {
    let b = rng.range_i32(0, bits);
    ((1i64 << b) - 1).max(1) as i32
}

/// Largest amplitude `a` (at most 32767) with `terms * a^2 <= 2^30` (no overflow of a 32-bit
/// MAC accumulator started near 0).
fn mac_amp(terms: usize) -> i32 {
    let t = terms.max(1) as f64;
    ((1u64 << 30) as f64 / t).sqrt().floor().min(32767.0) as i32
}

fn rand16(rng: &mut Rng, n: usize, amp: i32) -> Vec<OpusVal16> {
    (0..n).map(|_| rng.range_i32(-amp, amp) as i16).collect()
}

#[test]
fn xcorr_kernel_matches() {
    let mut rng = Rng::new(0x2001);
    for it in 0..20000 {
        let len = if it < 64 {
            3 + it as usize
        } else {
            rng.range_i32(3, 400) as usize
        };
        let amp = rand_amp(&mut rng, 15).min(mac_amp(len + 1));
        let (x, y) = if it % 3 == 0 {
            (
                sig16(&mut rng, len, it, amp),
                sig16(&mut rng, len + 3, it / 3, amp),
            )
        } else {
            (rand16(&mut rng, len, amp), rand16(&mut rng, len + 3, amp))
        };
        let init: [OpusVal32; 4] = core::array::from_fn(|_| rng.range_i32(-(1 << 29), 1 << 29));
        let mut sr = init;
        let mut sc = init;
        rp::xcorr_kernel(&x, &y, &mut sr, len);
        c::xcorr_kernel(&x, &y, &mut sc, len);
        assert_eq!(sr, sc, "xcorr_kernel len={len}");
    }
}

#[test]
fn inner_prods_match() {
    let mut rng = Rng::new(0x2002);
    for it in 0..20000u32 {
        let n = if it < 50 {
            it as usize
        } else {
            rng.range_i32(0, 2000) as usize
        };
        let amp = rand_amp(&mut rng, 15).min(mac_amp(n));
        let x = rand16(&mut rng, n, amp);
        let y1 = rand16(&mut rng, n, amp);
        let y2 = sig16(&mut rng, n, it, amp);
        let r = rp::celt_inner_prod(&x, &y1, n);
        let cc = c::celt_inner_prod(&x, &y1, n);
        assert_eq!(r, cc, "celt_inner_prod n={n}");
        let r = rp::dual_inner_prod(&x, &y1, &y2, n);
        let cc = c::dual_inner_prod(&x, &y1, &y2, n);
        assert_eq!(r, cc, "dual_inner_prod n={n}");
    }
}

#[test]
fn celt_pitch_xcorr_matches() {
    let mut rng = Rng::new(0x2003);
    let mut maxcorrs = std::collections::HashSet::new();
    for it in 0..12500u32 {
        let len = rng.range_i32(3, 600) as usize;
        let max_pitch = if it < 40 {
            1 + it as usize
        } else {
            rng.range_i32(1, 700) as usize
        };
        let amp = rand_amp(&mut rng, 15).min(mac_amp(len));
        let x = sig16(&mut rng, len, it, amp);
        let y = sig16(&mut rng, len + max_pitch + 3, it / 9, amp);
        let mut xr = vec![0; max_pitch];
        let mut xc = vec![0; max_pitch];
        let mr = rp::celt_pitch_xcorr(&x, &y, &mut xr, len, max_pitch);
        let mc = c::celt_pitch_xcorr(&x, &y, &mut xc, len, max_pitch);
        assert_slice_eq(
            &format!("celt_pitch_xcorr len={len} mp={max_pitch}"),
            &xr,
            &xc,
        );
        assert_eq!(mr, mc, "celt_pitch_xcorr maxcorr len={len} mp={max_pitch}");
        maxcorrs.insert(mr);
    }
    assert!(
        maxcorrs.len() > 5000,
        "maxcorr not diverse: {}",
        maxcorrs.len()
    );
}

#[test]
fn find_best_pitch_matches() {
    let mut rng = Rng::new(0x2004);
    for it in 0..15000u32 {
        let len = rng.range_i32(0, 600) as usize;
        let max_pitch = rng.range_i32(0, 600) as usize;
        let yshift = rng.range_i32(0, 8);
        // Syy accumulates (len + max_pitch) squares >> yshift.
        let amp = rand_amp(&mut rng, 15)
            .min(((mac_amp(len + max_pitch) as i64) << (yshift / 2)).min(32767) as i32);
        let y = sig16(&mut rng, len + max_pitch, it, amp);
        let xcorr: Vec<OpusVal32> = match it % 4 {
            0 => {
                let a = rand_amp(&mut rng, 31);
                (0..max_pitch).map(|_| rng.range_i32(-a, a)).collect()
            }
            1 => {
                // realistic: actual correlations
                let xl = len.max(3);
                let a = amp.min(mac_amp(xl));
                let x = sig16(&mut rng, xl, it + 1, a);
                let yy: Vec<OpusVal16> = y.iter().map(|&v| v.clamp(-a as i16, a as i16)).collect();
                let mut yy = yy;
                yy.resize(xl + max_pitch + 3, 0);
                let mut xc = vec![0; max_pitch];
                if max_pitch > 0 {
                    c::celt_pitch_xcorr(&x, &yy, &mut xc, xl, max_pitch);
                }
                xc
            }
            2 => vec![0; max_pitch],
            _ => (0..max_pitch)
                .map(|_| {
                    if rng.next_u32() & 3 == 0 {
                        -1
                    } else {
                        rng.next_u32() as i32
                    }
                })
                .collect(),
        };
        // C passes the maximum correlation (at least 1); also try other values.
        let maxcorr = if it % 5 == 4 {
            rng.range_i32(1, i32::MAX)
        } else {
            xcorr.iter().copied().fold(1, i32::max)
        };
        let mut br = [7, 7];
        let mut bc = [7, 7];
        rp::find_best_pitch(&xcorr, &y, len, max_pitch, &mut br, yshift, maxcorr);
        c::find_best_pitch(&xcorr, &y, len, max_pitch, &mut bc, yshift, maxcorr);
        assert_eq!(
            br, bc,
            "find_best_pitch len={len} mp={max_pitch} yshift={yshift}"
        );
    }
}

#[test]
fn celt_fir5_matches() {
    let mut rng = Rng::new(0x2005);
    for it in 0..10000u32 {
        let n = rng.range_i32(0, 1100) as usize;
        let amp = rand_amp(&mut rng, 15);
        let mut xr = sig16(&mut rng, n, it, amp);
        let mut xc = xr.clone();
        // pitch_downsample's lpc2 taps (Q12, |tap| < 2): no accumulator overflow.
        let num: [OpusVal16; 5] = core::array::from_fn(|_| rng.range_i32(-8191, 8191) as i16);
        rp::celt_fir5(&mut xr, &num, n);
        c::celt_fir5(&mut xc, &num, n);
        assert_slice_eq(&format!("celt_fir5 n={n}"), &xr, &xc);
    }
}

/// Lengths passed to `pitch_downsample` by the codec: encoder `(1024+N)>>1`, decoder 1024.
const fn downsample_len(rng: &mut Rng, it: u32) -> usize {
    match it % 3 {
        0 => (1024 + FRAME_SIZES[(it as usize / 3) % 4]) >> 1,
        1 => 1024,
        _ => rng.range_i32(8, 1500) as usize,
    }
}

/// A `celt_sig` amplitude: 16-bit PCM in Q12 (`SIG_SHIFT`), up to `SIG_SAT`, or quiet.
fn sig_amp(rng: &mut Rng) -> i32 {
    match rng.range_i32(0, 3) {
        0 => 32767 << 12,
        1 => 536_870_911,
        2 => rand_amp(rng, 29),
        _ => rand_amp(rng, 12),
    }
}

#[test]
fn pitch_downsample_matches() {
    let mut rng = Rng::new(0x2006);
    for it in 0..7500u32 {
        let len = downsample_len(&mut rng, it);
        let factor = match rng.range_i32(0, 4) {
            0 => 1,
            1 => 4,
            2 => rng.range_i32(1, 6) as usize,
            _ => 2,
        };
        let cc = rng.range_i32(1, 2) as usize;
        let n = len * factor + factor;
        let amp = sig_amp(&mut rng);
        let x0 = sig32(&mut rng, n, it, amp);
        let k1 = it / 9 + rng.next_u32() % 3;
        let amp1 = if rng.next_u32() & 1 == 0 {
            amp
        } else {
            sig_amp(&mut rng)
        };
        let x1 = sig32(&mut rng, n, k1, amp1);
        let x: [&[CeltSig]; 2] = [&x0, &x1];
        let mut lr = vec![0; len];
        let mut lc = vec![0; len];
        rp::pitch_downsample(&x[..cc], &mut lr, len, cc, factor);
        c::pitch_downsample(&x[..cc], &mut lc, len, cc, factor);
        assert_slice_eq(
            &format!("pitch_downsample it={it} len={len} C={cc} factor={factor}"),
            &lr,
            &lc,
        );
    }
}

/// Builds a realistic half-rate pitch buffer via `pitch_downsample` (checked against C).
fn pitch_buf(rng: &mut Rng, len: usize, it: u32) -> Vec<OpusVal16> {
    let cc = rng.range_i32(1, 2) as usize;
    let n = 2 * len + 2;
    let amp = sig_amp(rng);
    let x0 = sig32(rng, n, it, amp);
    let x1 = sig32(rng, n, it, amp);
    let x: [&[CeltSig]; 2] = [&x0, &x1];
    let mut lr = vec![0; len];
    let mut lc = vec![0; len];
    rp::pitch_downsample(&x[..cc], &mut lr, len, cc, 2);
    c::pitch_downsample(&x[..cc], &mut lc, len, cc, 2);
    assert_slice_eq("pitch_buf", &lr, &lc);
    lr
}

#[test]
fn pitch_search_matches() {
    let mut rng = Rng::new(0x2007);
    let mut seen = std::collections::HashSet::new();
    for it in 0..6000u32 {
        let (len, max_pitch, off, buf_len) = match it % 3 {
            0 => {
                // encoder: pitch_search(pitch_buf+(max_period>>1), pitch_buf, N, max_period-3*min_period)
                let q = if it % 2 == 0 { 1 } else { 2 };
                let max_period = 1024 * q;
                let min_period = 15 * q;
                let n = FRAME_SIZES[(it as usize / 3) % FRAME_SIZES.len()];
                let n = if q == 1 { n.min(960) } else { n };
                (
                    n,
                    max_period - 3 * min_period,
                    max_period >> 1,
                    (max_period + n) >> 1,
                )
            }
            1 => {
                // decoder PLC: DECODE_BUFFER_SIZE=2048, LAG_MAX=720, LAG_MIN=100
                (2048 - 720, 720 - 100, 720 >> 1, 2048 >> 1)
            }
            _ => {
                let len = rng.range_i32(12, 1500) as usize;
                let max_pitch = rng.range_i32(4, 1200) as usize;
                let off = max_pitch >> 1;
                (len, max_pitch, off, off + (len >> 1) + 2)
            }
        };
        let buf = if it % 2 == 0 {
            pitch_buf(&mut rng, buf_len, it)
        } else {
            // Arbitrary 16-bit input (pitch_search rescales it from the maximum of the even
            // samples, so the peak goes to sample 0: an odd-sample peak far above every even
            // sample overflows the C finer search).
            let amp = rand_amp(&mut rng, 15);
            let mut b = sig16(&mut rng, buf_len, it, amp);
            b[0] = b.iter().fold(0, |m: i16, &v| m.max(v.saturating_abs()));
            b
        };
        let x_lp = &buf[off..];
        let pr = rp::pitch_search(x_lp, &buf, len, max_pitch);
        let pc = c::pitch_search(x_lp, &buf, len, max_pitch);
        assert_eq!(pr, pc, "pitch_search it={it} len={len} mp={max_pitch}");
        seen.insert(pr);
    }
    assert!(
        seen.len() > 200,
        "pitch_search outputs not diverse: {}",
        seen.len()
    );
}

#[test]
fn compute_pitch_gain_matches() {
    let mut rng = Rng::new(0x2008);
    let mut gains = std::collections::HashSet::new();
    for it in 0..100000u32 {
        let xx = if it % 97 == 0 {
            0
        } else {
            rand_amp(&mut rng, 31) & rng.next_u32() as i32 & i32::MAX
        };
        let yy = if it % 89 == 0 {
            0
        } else {
            rand_amp(&mut rng, 31) & rng.next_u32() as i32 & i32::MAX
        };
        let xy = match it % 3 {
            // |xy| <= sqrt(xx*yy), as for real correlations
            0 => {
                let m = ((xx as f64) * (yy as f64)).sqrt().min(i32::MAX as f64) as i32;
                rng.range_i32(-m, m)
            }
            1 => rand_amp(&mut rng, 31) & rng.next_u32() as i32,
            _ => rng.next_u32() as i32,
        };
        let r = rp::compute_pitch_gain(xy, xx, yy);
        let cc = c::compute_pitch_gain(xy, xx, yy);
        assert_eq!(r, cc, "compute_pitch_gain({xy},{xx},{yy})");
        gains.insert(r);
    }
    assert!(gains.len() > 10000, "gains not diverse: {}", gains.len());
}

#[test]
fn remove_doubling_matches() {
    let mut rng = Rng::new(0x2009);
    let mut changed = 0;
    let mut gains = std::collections::HashSet::new();
    for it in 0..10000u32 {
        let q = if it % 4 == 3 { 2 } else { 1 };
        let max_period = 1024 * q;
        let min_period = 15 * q;
        let n = if it % 5 == 4 {
            rng.range_i32(12, 1920)
        } else {
            FRAME_SIZES[(it as usize) % if q == 1 { 4 } else { 6 }] as i32
        };
        let buf_len = ((max_period + n) >> 1) as usize;
        let buf = if it % 2 == 0 {
            pitch_buf(&mut rng, buf_len, it)
        } else {
            // Energies over up to maxperiod/2 + N/2 samples must fit in 32 bits.
            let amp = rand_amp(&mut rng, 15).min(mac_amp(buf_len));
            sig16(&mut rng, buf_len, it, amp)
        };
        let t0 = if it % 3 == 0 {
            // realistic: from pitch_search, as the encoder does
            let p = rp::pitch_search(
                &buf[(max_period >> 1) as usize..],
                &buf,
                n as usize,
                (max_period - 3 * min_period) as usize,
            );
            max_period - p
        } else {
            rng.range_i32(2, max_period + 50)
        };
        let prev_period = match rng.range_i32(0, 2) {
            0 => 0,
            1 => t0 + rng.range_i32(-6, 6),
            _ => rng.range_i32(0, max_period),
        };
        let prev_gain: OpusVal16 = match rng.range_i32(0, 2) {
            0 => 0,
            _ => rng.range_i32(0, 26214) as i16,
        };
        let mut tr = t0;
        let mut tc = t0;
        let gr = rp::remove_doubling(
            &buf,
            max_period,
            min_period,
            n,
            &mut tr,
            prev_period,
            prev_gain,
        );
        let gc = c::remove_doubling(
            &buf,
            max_period,
            min_period,
            n,
            &mut tc,
            prev_period,
            prev_gain,
        );
        assert_eq!(
            (tr, gr),
            (tc, gc),
            "remove_doubling it={it} n={n} t0={t0} prev=({prev_period},{prev_gain})"
        );
        if (tr - t0).abs() > 2 {
            changed += 1;
        }
        gains.insert(gr);
    }
    assert!(
        changed > 100,
        "remove_doubling rarely changes the period: {changed}"
    );
    assert!(
        gains.len() > 1000,
        "remove_doubling gains not diverse: {}",
        gains.len()
    );
}

/// Normalised autocorrelation of a test signal, via the oracle `_celt_autocorr`.
fn real_ac(rng: &mut Rng, p: usize, it: u32) -> Vec<OpusVal32> {
    let n = rng.range_i32((p + 4) as i32, 1024) as usize;
    let amp = rand_amp(rng, 15);
    let x = sig16(rng, n, it, amp);
    let mut ac = vec![0; p + 1];
    let _shift = c::celt_autocorr(&x, &mut ac, &[], 0, p, n);
    ac
}

/// `pitch_downsample`'s conditioning of an autocorrelation: -40 dB noise floor and lag window.
fn condition_ac(ac: &mut [OpusVal32]) {
    ac[0] += ac[0] >> 13;
    for (i, a) in ac.iter_mut().enumerate().skip(1) {
        // MULT16_32_Q15(2*i*i, ac[i]) with |2*i*i| < 2^15
        let w = (2 * i * i) as i64;
        *a -= ((w * *a as i64) >> 15) as i32;
    }
}

#[test]
fn celt_lpc_matches() {
    let mut rng = Rng::new(0x200a);
    for it in 0..20000u32 {
        let p = match it % 4 {
            0 => 4,
            1 => 24,
            _ => rng.range_i32(1, 24) as usize,
        };
        // `conditioned`: noise floor + lag window, as both codec callers (pitch_downsample and
        // the decoder PLC) do. Only then is the 32-bit form free of signed overflow (the
        // `MULT32_32_Q31` of a saturated reflection coefficient overflows its 32-bit partial
        // sums: undefined in C, a panic here).
        let (ac, conditioned): (Vec<OpusVal32>, bool) = match (it / 4) % 6 {
            0 => {
                // positive-definite from a random signal, arbitrary scale
                let mut ac = real_ac(&mut rng, p, it);
                let s = rng.range_i32(0, 20);
                for a in &mut ac {
                    *a >>= s;
                }
                (ac, false)
            }
            1 => {
                let mut ac = real_ac(&mut rng, p, it);
                condition_ac(&mut ac);
                (ac, true)
            }
            2 => {
                let mut ac = vec![0; p + 1];
                ac[0] = rng.range_i32(0, 3);
                (ac, true)
            }
            3 => {
                // strongly peaked spectrum (sine), with or without a noise floor
                let n = 1024;
                let f = 0.01 + 0.3 * rng.f32_sym().abs() as f64;
                let x: Vec<OpusVal16> = (0..n)
                    .map(|i| ((f * i as f64).sin() * 30000.0).round() as i16)
                    .collect();
                let mut ac = vec![0; p + 1];
                let _shift = c::celt_autocorr(&x, &mut ac, &[], 0, p, n);
                if it % 2 == 0 {
                    condition_ac(&mut ac);
                }
                (ac, it % 2 == 0)
            }
            4 => {
                // Noise through (1 + z^-1)^m (or (1 - z^-1)^m): zeros on the unit circle keep
                // the prediction gain below 30 dB up to high orders while the predictor
                // coefficients grow past 16 bits (the bandwidth-expansion / fallback paths).
                let n = 1024;
                let m = rng.range_i32(1, 6) as usize;
                let sign = if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
                let mut taps = vec![1.0f64];
                for _ in 0..m {
                    let mut t = vec![0.0; taps.len() + 1];
                    for (i, &v) in taps.iter().enumerate() {
                        t[i] += v;
                        t[i + 1] += sign * v;
                    }
                    taps = t;
                }
                let norm: f64 = taps.iter().map(|v| v.abs()).sum();
                let w: Vec<f64> = (0..n + m).map(|_| rng.f32_sym() as f64).collect();
                let x: Vec<OpusVal16> = (0..n)
                    .map(|i| {
                        let s: f64 = taps.iter().enumerate().map(|(j, t)| t * w[i + j]).sum();
                        (s / norm * 32000.0).round() as i16
                    })
                    .collect();
                let mut ac = vec![0; p + 1];
                if it % 3 == 0 {
                    // the exact autocorrelation of the filter (no noise at all)
                    let e: f64 = taps.iter().map(|v| v * v).sum();
                    let scale = (1u32 << rng.range_i32(10, 29)) as f64 / e;
                    for (k, a) in ac.iter_mut().enumerate() {
                        let r: f64 = taps
                            .iter()
                            .zip(taps.iter().skip(k))
                            .map(|(u, v)| u * v)
                            .sum();
                        *a = (r * scale).round() as i32;
                    }
                } else {
                    let _shift = c::celt_autocorr(&x, &mut ac, &[], 0, p, n);
                }
                if it % 2 == 0 {
                    condition_ac(&mut ac);
                }
                // (the noise-free exact one overflows the 32-bit form even when conditioned)
                (ac, it % 2 == 0 && it % 3 != 0)
            }
            _ => (real_ac(&mut rng, p, it), false),
        };
        let mut lr = vec![9; p];
        let mut lc = vec![9; p];
        rl::_celt_lpc(&mut lr, &ac, p);
        c::celt_lpc(&mut lc, &ac, p);
        assert_slice_eq(&format!("_celt_lpc it={it} p={p}"), &lr, &lc);
        // Both OPUS_FAST_INT64 forms against C (the 32-bit one is a separate C copy).
        let mut l64 = vec![9; p];
        rl::_celt_lpc_fast_int64::<true>(&mut l64, &ac, p);
        assert_slice_eq(&format!("_celt_lpc(int64) it={it} p={p}"), &l64, &lc);
        if conditioned {
            let mut l32r = vec![9; p];
            let mut l32c = vec![9; p];
            rl::_celt_lpc_fast_int64::<false>(&mut l32r, &ac, p);
            c::celt_lpc_int32(&mut l32c, &ac, p);
            assert_slice_eq(&format!("_celt_lpc(int32) it={it} p={p}"), &l32r, &l32c);
        }
    }
}

/// Q12 LPC filter of order `ord`: from a real signal (via the oracle) or random small taps.
fn filter_taps(rng: &mut Rng, ord: usize, it: u32) -> Vec<OpusVal16> {
    // The fixed-point `_celt_lpc` supports orders up to CELT_LPC_ORDER (24).
    if it.is_multiple_of(3) || ord > 24 {
        let a = (4096 / ord as i32).max(1);
        rand16(rng, ord, a)
    } else {
        let mut ac = real_ac(rng, ord, it);
        condition_ac(&mut ac);
        let mut lpc = vec![0; ord];
        c::celt_lpc(&mut lpc, &ac, ord);
        lpc
    }
}

fn sum_abs(v: &[OpusVal16]) -> i64 {
    v.iter().map(|&x| (x as i64).abs()).sum()
}

#[test]
fn celt_fir_matches() {
    let mut rng = Rng::new(0x200b);
    for it in 0..15000u32 {
        let ord = match it % 4 {
            0 => 24,
            1 => 16,
            _ => rng.range_i32(3, 32) as usize,
        };
        let n = match it % 3 {
            0 => rng.range_i32(0, 12) as usize,
            _ => rng.range_i32(0, 1100) as usize,
        };
        let num = filter_taps(&mut rng, ord, it);
        // |x<<12 + sum num*x| < 2^31
        let bound = ((i32::MAX as i64) / (4096 + sum_abs(&num))).min(32767) as i32;
        let amp = rand_amp(&mut rng, 15).min(bound);
        let x = sig16(&mut rng, n + ord, it, amp);
        let mut yr = vec![0; n];
        let mut yc = vec![0; n];
        rl::celt_fir(&x, &num, &mut yr, n, ord);
        c::celt_fir(&x, &num, &mut yc, n, ord);
        assert_slice_eq(&format!("celt_fir it={it} n={n} ord={ord}"), &yr, &yc);
    }
}

#[test]
fn celt_iir_matches() {
    let mut rng = Rng::new(0x200c);
    let mut done = 0;
    let mut it = 0u32;
    while done < 15000 {
        it += 1;
        let ord = match it % 3 {
            0 => 24,
            1 => 4 * rng.range_i32(1, 8) as usize,
            _ => 16,
        };
        let n = match it % 4 {
            // decoder PLC: extrapolation_len = N + overlap
            0 => FRAME_SIZES[(it as usize / 4) % 4] + 120,
            1 => ord + rng.range_i32(0, 7) as usize,
            _ => rng.range_i32(ord as i32, 2200) as usize,
        };
        let den = filter_taps(&mut rng, ord, it);
        // The filter state is 16-bit (SROUND16), so |sum| <= |x| + sum|den| * 32767.
        let headroom = i32::MAX as i64 - sum_abs(&den) * 32767;
        if headroom < 1 << 16 {
            continue;
        }
        let amp = (rand_amp(&mut rng, 30) as i64).min(headroom) as i32;
        let x = sig32(&mut rng, n, it, amp);
        let mamp = rand_amp(&mut rng, 15);
        let mem0 = rand16(&mut rng, ord, mamp);
        let (mut mr, mut mc) = (mem0.clone(), mem0.clone());
        let mut yr = vec![0; n];
        let mut yc = vec![0; n];
        if it.is_multiple_of(2) {
            rl::celt_iir(&x, &den, &mut yr, n, ord, &mut mr);
            c::celt_iir(&x, &den, &mut yc, n, ord, &mut mc);
        } else {
            yr.copy_from_slice(&x);
            yc.copy_from_slice(&x);
            rl::celt_iir_inplace(&mut yr, &den, n, ord, &mut mr);
            c::celt_iir_inplace(&mut yc, &den, n, ord, &mut mc);
        }
        assert_slice_eq(&format!("celt_iir y it={it} n={n} ord={ord}"), &yr, &yc);
        assert_slice_eq(&format!("celt_iir mem it={it} n={n} ord={ord}"), &mr, &mc);
        done += 1;
    }
}

/// The C `celt_coef` of a window value in `[0, 1]` (Q15, or Q31 with QEXT).
fn coef(w: f64) -> CeltCoef {
    #[cfg(feature = "qext")]
    {
        (w * 2147483647.0).round() as i32
    }
    #[cfg(not(feature = "qext"))]
    {
        (w * 32767.0).round() as i16
    }
}

#[test]
fn celt_autocorr_matches() {
    let mut rng = Rng::new(0x200d);
    let mut shifts = std::collections::HashSet::new();
    for it in 0..12500u32 {
        let (n, lag, overlap) = match it % 4 {
            // pitch_downsample: lag 4, no window
            0 => (downsample_len(&mut rng, it), 4, 0),
            // decoder PLC: MAX_PERIOD samples, lag 24, window of the mode overlap (120)
            1 => (1024 * if it % 8 == 1 { 1 } else { 2 }, 24, 120),
            2 => {
                let n = rng.range_i32(8, 2100) as usize;
                let lag = rng.range_i32(0, (n as i32 - 3).min(40)) as usize;
                let overlap = rng.range_i32(0, (n / 2) as i32) as usize;
                (n, lag, overlap)
            }
            _ => {
                let n = rng.range_i32(8, 300) as usize;
                let lag = rng.range_i32(0, n as i32 - 3) as usize;
                (n, lag, 0)
            }
        };
        let amp = rand_amp(&mut rng, 15);
        let x = sig16(&mut rng, n, it, amp);
        let window: Vec<CeltCoef> = (0..overlap)
            .map(|i| {
                let t = (i as f64 + 0.5) / overlap as f64;
                coef((0.5 * core::f64::consts::PI * t).sin())
            })
            .collect();
        let mut ar = vec![0; lag + 1];
        let mut ac = vec![0; lag + 1];
        let sr = rl::_celt_autocorr(&x, &mut ar, &window, overlap, lag, n);
        let sc = c::celt_autocorr(&x, &mut ac, &window, overlap, lag, n);
        assert_eq!(sr, sc, "_celt_autocorr shift it={it} n={n} lag={lag}");
        assert_slice_eq(
            &format!("_celt_autocorr it={it} n={n} lag={lag} overlap={overlap}"),
            &ar,
            &ac,
        );
        shifts.insert(sr);
    }
    assert!(shifts.len() > 20, "autocorr shifts not diverse: {shifts:?}");
}

/// End-to-end encoder pitch pre-analysis chain (downsample -> search -> remove_doubling), stereo
/// and mono, on music/speech-like 16-bit PCM in Q12 (`celt_sig`), at every frame size.
#[test]
fn encoder_pitch_chain_matches() {
    let mut rng = Rng::new(0x200e);
    for it in 0..2000u32 {
        let n = FRAME_SIZES[(it as usize) % 4];
        let max_period = 1024usize;
        let min_period = 15usize;
        let cc = 1 + (it as usize / 4) % 2;
        let total = max_period + n;
        let seed = rng.next_u64();
        let pcm = if it % 2 == 0 {
            signals::speech_like(total, cc, 48000, seed)
        } else {
            signals::music_like(total, cc, 48000, seed)
        };
        let gain = 32767.0 * 10f32.powf(-(rng.range_i32(0, 4) as f32));
        let chans: Vec<Vec<CeltSig>> = (0..cc)
            .map(|ch| {
                (0..total)
                    .map(|i| ((pcm[i * cc + ch] * gain).round() as i32) << 12)
                    .collect()
            })
            .collect();
        let x: Vec<&[CeltSig]> = chans.iter().map(|v| v.as_slice()).collect();
        let len = total >> 1;
        let mut br = vec![0; len];
        let mut bc = vec![0; len];
        rp::pitch_downsample(&x, &mut br, len, cc, 2);
        c::pitch_downsample(&x, &mut bc, len, cc, 2);
        assert_slice_eq("chain downsample", &br, &bc);
        let mp = max_period - 3 * min_period;
        let pr = rp::pitch_search(&br[max_period >> 1..], &br, n, mp);
        let pc = c::pitch_search(&bc[max_period >> 1..], &bc, n, mp);
        assert_eq!(pr, pc, "chain pitch_search");
        let mut tr = max_period as i32 - pr;
        let mut tc = tr;
        let prev = rng.range_i32(0, 1023);
        let gr = rp::remove_doubling(
            &br,
            max_period as i32,
            min_period as i32,
            n as i32,
            &mut tr,
            prev,
            16384,
        );
        let gc = c::remove_doubling(
            &bc,
            max_period as i32,
            min_period as i32,
            n as i32,
            &mut tc,
            prev,
            16384,
        );
        assert_eq!((tr, gr), (tc, gc), "chain remove_doubling");
    }
}
