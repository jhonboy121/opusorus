//! Differential tests for unit `celt_pitch_lpc` (celt/pitch.c, celt/celt_lpc.c) vs the C oracle.
//! Every comparison is bit-exact (`to_bits`).

// Float-only: not compiled in fixed-point builds until this unit is converted
// (docs/FIXED_POINT.md).
#![cfg(not(feature = "fixed-point"))]

use opusorus::celt::{celt_lpc as rl, pitch as rp};
use opusorus_conformance::{Rng, assert_bits_eq_f32, signals};
use opusorus_oracle::celt_pitch_lpc as c;

/// Frame sizes the CELT encoder uses (48 kHz, 2.5..20 ms), plus QEXT 96 kHz sizes.
const FRAME_SIZES: [usize; 6] = [120, 240, 480, 960, 1440, 1920];

/// Kinds of test signal (CELT signal scale, i.e. +-32768 full scale).
fn signal(rng: &mut Rng, n: usize, kind: u32) -> Vec<f32> {
    let seed = rng.next_u64();
    match kind % 9 {
        0 => signals::noise(n, 1, 32767.0, seed),
        1 => signals::noise(n, 1, 1e-3, seed),
        2 => signals::music_like(n, 1, 48000, seed)
            .iter()
            .map(|v| v * 32768.0)
            .collect(),
        3 => signals::speech_like(n, 1, 48000, seed)
            .iter()
            .map(|v| v * 32768.0)
            .collect(),
        4 => vec![0.0; n],
        5 => {
            // impulses
            let mut v = vec![0.0; n];
            if n == 0 {
                return v;
            }
            let count = rng.range_i32(1, 6);
            for _ in 0..count {
                let p = rng.range_i32(0, n as i32 - 1) as usize;
                v[p] = if rng.next_u32() & 1 == 0 {
                    32767.0
                } else {
                    -32768.0
                };
            }
            v
        }
        6 => {
            // full-scale square wave
            let period = rng.range_i32(2, 400) as usize;
            (0..n)
                .map(|i| {
                    if (i / period).is_multiple_of(2) {
                        32767.0
                    } else {
                        -32768.0
                    }
                })
                .collect()
        }
        7 => {
            // periodic pulse train with noise (strong pitch)
            let period = rng.range_i32(15, 1000) as usize;
            let amp = 10f32.powf(rng.range_i32(-2, 4) as f32);
            let mut v = signals::noise(n, 1, amp * 0.05, seed);
            let start = rng.range_i32(0, period as i32 - 1) as usize;
            let mut i = start;
            while i < n {
                v[i] += amp;
                i += period;
            }
            v
        }
        _ => {
            let amp = 10f32.powf(rng.range_i32(-6, 4) as f32);
            signals::noise(n, 1, amp, seed)
        }
    }
}

fn rand_vec(rng: &mut Rng, n: usize, amp: f32) -> Vec<f32> {
    (0..n).map(|_| amp * rng.f32_sym()).collect()
}

fn rand_amp(rng: &mut Rng) -> f32 {
    10f32.powf(rng.range_i32(-8, 5) as f32)
}

#[test]
fn xcorr_kernel_matches() {
    let mut rng = Rng::new(0x1001);
    for it in 0..20000 {
        let len = if it < 64 {
            3 + it as usize
        } else {
            rng.range_i32(3, 400) as usize
        };
        let amp = rand_amp(&mut rng);
        let x = rand_vec(&mut rng, len, amp);
        let y = rand_vec(&mut rng, len + 3, amp);
        let init: [f32; 4] = core::array::from_fn(|_| rng.f32_sym() * amp);
        let mut sr = init;
        let mut sc = init;
        rp::xcorr_kernel(&x, &y, &mut sr, len);
        c::xcorr_kernel(&x, &y, &mut sc, len);
        assert_bits_eq_f32(&format!("xcorr_kernel len={len}"), &sr, &sc);
    }
}

#[test]
fn inner_prods_match() {
    let mut rng = Rng::new(0x1002);
    for it in 0..20000 {
        let n = if it < 50 {
            it as usize
        } else {
            rng.range_i32(0, 2000) as usize
        };
        let amp = rand_amp(&mut rng);
        let x = rand_vec(&mut rng, n, amp);
        let y1 = rand_vec(&mut rng, n, amp);
        let y2 = signal(&mut rng, n, it);
        let r = rp::celt_inner_prod(&x, &y1, n);
        let cc = c::celt_inner_prod(&x, &y1, n);
        assert_eq!(r.to_bits(), cc.to_bits(), "celt_inner_prod n={n}");
        let r = rp::dual_inner_prod(&x, &y1, &y2, n);
        let cc = c::dual_inner_prod(&x, &y1, &y2, n);
        assert_bits_eq_f32(
            &format!("dual_inner_prod n={n}"),
            &[r.0, r.1],
            &[cc.0, cc.1],
        );
    }
}

#[test]
fn celt_pitch_xcorr_matches() {
    let mut rng = Rng::new(0x1003);
    for it in 0..12500u32 {
        let len = rng.range_i32(3, 600) as usize;
        let max_pitch = if it < 40 {
            1 + it as usize
        } else {
            rng.range_i32(1, 700) as usize
        };
        let x = signal(&mut rng, len, it);
        let y = signal(&mut rng, len + max_pitch + 3, it / 9);
        let mut xr = vec![0f32; max_pitch];
        let mut xc = vec![0f32; max_pitch];
        rp::celt_pitch_xcorr(&x, &y, &mut xr, len, max_pitch);
        c::celt_pitch_xcorr(&x, &y, &mut xc, len, max_pitch);
        assert_bits_eq_f32(
            &format!("celt_pitch_xcorr len={len} mp={max_pitch}"),
            &xr,
            &xc,
        );
    }
}

#[test]
fn find_best_pitch_matches() {
    let mut rng = Rng::new(0x1004);
    for it in 0..15000u32 {
        let len = rng.range_i32(0, 600) as usize;
        let max_pitch = rng.range_i32(0, 600) as usize;
        let y = signal(&mut rng, len + max_pitch, it);
        let xcorr: Vec<f32> = match it % 4 {
            0 => {
                let a = rand_amp(&mut rng) * 1e6;
                rand_vec(&mut rng, max_pitch, a)
            }
            1 => {
                // realistic: actual correlations
                let x = signal(&mut rng, len.max(3), it + 1);
                let mut xc = vec![0f32; max_pitch];
                if max_pitch > 0 && len >= 3 {
                    c::celt_pitch_xcorr(&x, &y, &mut xc, len, max_pitch);
                }
                xc
            }
            2 => vec![0.0; max_pitch],
            _ => (0..max_pitch)
                .map(|_| {
                    if rng.next_u32() & 3 == 0 {
                        -1.0
                    } else {
                        rng.f32_sym() * 1e12
                    }
                })
                .collect(),
        };
        let mut br = [7, 7];
        let mut bc = [7, 7];
        rp::find_best_pitch(&xcorr, &y, len, max_pitch, &mut br);
        c::find_best_pitch(&xcorr, &y, len, max_pitch, &mut bc);
        assert_eq!(br, bc, "find_best_pitch len={len} mp={max_pitch}");
    }
}

#[test]
fn celt_fir5_matches() {
    let mut rng = Rng::new(0x1005);
    for it in 0..10000u32 {
        let n = rng.range_i32(0, 1100) as usize;
        let mut xr = signal(&mut rng, n, it);
        let mut xc = xr.clone();
        let num: [f32; 5] = core::array::from_fn(|_| rng.f32_sym() * 2.0);
        rp::celt_fir5(&mut xr, &num, n);
        c::celt_fir5(&mut xc, &num, n);
        assert_bits_eq_f32(&format!("celt_fir5 n={n}"), &xr, &xc);
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

#[test]
fn pitch_downsample_matches() {
    let mut rng = Rng::new(0x1006);
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
        let x0 = signal(&mut rng, n, it);
        let k1 = it / 9 + rng.next_u32() % 3;
        let x1 = signal(&mut rng, n, k1);
        let x: [&[f32]; 2] = [&x0, &x1];
        let mut lr = vec![0f32; len];
        let mut lc = vec![0f32; len];
        rp::pitch_downsample(&x[..cc], &mut lr, len, cc, factor);
        c::pitch_downsample(&x[..cc], &mut lc, len, cc, factor);
        assert_bits_eq_f32(
            &format!("pitch_downsample it={it} len={len} C={cc} factor={factor}"),
            &lr,
            &lc,
        );
    }
}

/// Builds a realistic half-rate pitch buffer via `pitch_downsample` (checked against C).
fn pitch_buf(rng: &mut Rng, len: usize, it: u32) -> Vec<f32> {
    let cc = rng.range_i32(1, 2) as usize;
    let n = 2 * len + 2;
    let x0 = signal(rng, n, it);
    let x1 = signal(rng, n, it);
    let x: [&[f32]; 2] = [&x0, &x1];
    let mut lr = vec![0f32; len];
    let mut lc = vec![0f32; len];
    rp::pitch_downsample(&x[..cc], &mut lr, len, cc, 2);
    c::pitch_downsample(&x[..cc], &mut lc, len, cc, 2);
    assert_bits_eq_f32("pitch_buf", &lr, &lc);
    lr
}

#[test]
fn pitch_search_matches() {
    let mut rng = Rng::new(0x1007);
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
            signal(&mut rng, buf_len, it)
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
    let mut rng = Rng::new(0x1008);
    for _ in 0..100000 {
        let xy = rng.f32_sym() * rand_amp(&mut rng) * 1e4;
        let xx = rng.f32_sym().abs() * rand_amp(&mut rng) * 1e4;
        let yy = rng.f32_sym().abs() * rand_amp(&mut rng) * 1e4;
        let r = rp::compute_pitch_gain(xy, xx, yy);
        let cc = c::compute_pitch_gain(xy, xx, yy);
        assert_eq!(
            r.to_bits(),
            cc.to_bits(),
            "compute_pitch_gain({xy},{xx},{yy})"
        );
    }
}

#[test]
fn remove_doubling_matches() {
    let mut rng = Rng::new(0x1009);
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
            signal(&mut rng, buf_len, it)
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
        let prev_gain = match rng.range_i32(0, 2) {
            0 => 0.0,
            _ => rng.f32_sym().abs() * 0.8,
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
            (tr, gr.to_bits()),
            (tc, gc.to_bits()),
            "remove_doubling it={it} n={n} t0={t0} prev=({prev_period},{prev_gain}) gains r={gr} c={gc}"
        );
        if (tr - t0).abs() > 2 {
            changed += 1;
        }
        gains.insert(gr.to_bits());
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

/// Real autocorrelation of a random signal (positive definite), via the oracle.
fn real_ac(rng: &mut Rng, p: usize, it: u32) -> Vec<f32> {
    let n = rng.range_i32((p + 4) as i32, 1024) as usize;
    let x = signal(rng, n, it);
    let mut ac = vec![0f32; p + 1];
    c::celt_autocorr(&x, &mut ac, &[], 0, p, n);
    ac
}

#[test]
fn celt_lpc_matches() {
    let mut rng = Rng::new(0x100a);
    for it in 0..20000u32 {
        let p = match it % 4 {
            0 => 4,
            1 => 24,
            _ => rng.range_i32(1, 24) as usize,
        };
        let ac: Vec<f32> = match (it / 4) % 5 {
            0 => {
                let a = rand_amp(&mut rng);
                let mut ac = rand_vec(&mut rng, p + 1, a);
                ac[0] = ac[0].abs();
                ac
            }
            1 => {
                let mut ac = real_ac(&mut rng, p, it);
                ac[0] *= 1.0001;
                ac
            }
            2 => vec![1e-11; p + 1],
            3 => {
                let mut ac = vec![0f32; p + 1];
                ac[0] = if rng.next_u32() & 1 == 0 {
                    1e-10
                } else {
                    1.1e-10
                };
                ac
            }
            _ => real_ac(&mut rng, p, it),
        };
        let mut lr = vec![9f32; p];
        let mut lc = vec![9f32; p];
        rl::_celt_lpc(&mut lr, &ac, p);
        c::celt_lpc(&mut lc, &ac, p);
        assert_bits_eq_f32(&format!("_celt_lpc it={it} p={p}"), &lr, &lc);
    }
}

/// Stable LPC filter of order `ord` (from a real signal) or random taps with `sum |a| < 1`.
fn filter_taps(rng: &mut Rng, ord: usize, it: u32) -> Vec<f32> {
    if it.is_multiple_of(3) {
        rand_vec(rng, ord, 0.99 / ord as f32)
    } else {
        let mut ac = real_ac(rng, ord, it);
        ac[0] *= 1.0001;
        for (i, a) in ac.iter_mut().enumerate().skip(1) {
            *a -= *a * (0.008f32 * 0.008f32) * i as f32 * i as f32;
        }
        let mut lpc = vec![0f32; ord];
        c::celt_lpc(&mut lpc, &ac, ord);
        lpc
    }
}

#[test]
fn celt_fir_matches() {
    let mut rng = Rng::new(0x100b);
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
        let x = signal(&mut rng, n + ord, it);
        let num = filter_taps(&mut rng, ord, it);
        let mut yr = vec![0f32; n];
        let mut yc = vec![0f32; n];
        rl::celt_fir(&x, &num, &mut yr, n, ord);
        c::celt_fir(&x, &num, &mut yc, n, ord);
        assert_bits_eq_f32(&format!("celt_fir it={it} n={n} ord={ord}"), &yr, &yc);
    }
}

#[test]
fn celt_iir_matches() {
    let mut rng = Rng::new(0x100c);
    for it in 0..15000u32 {
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
        let x = signal(&mut rng, n, it);
        let den = filter_taps(&mut rng, ord, it);
        let a = rand_amp(&mut rng);
        let mem0 = rand_vec(&mut rng, ord, a);
        let (mut mr, mut mc) = (mem0.clone(), mem0.clone());
        let mut yr = vec![0f32; n];
        let mut yc = vec![0f32; n];
        if it % 2 == 0 {
            rl::celt_iir(&x, &den, &mut yr, n, ord, &mut mr);
            c::celt_iir(&x, &den, &mut yc, n, ord, &mut mc);
        } else {
            yr.copy_from_slice(&x);
            yc.copy_from_slice(&x);
            rl::celt_iir_inplace(&mut yr, &den, n, ord, &mut mr);
            c::celt_iir_inplace(&mut yc, &den, n, ord, &mut mc);
        }
        assert_bits_eq_f32(&format!("celt_iir y it={it} n={n} ord={ord}"), &yr, &yc);
        assert_bits_eq_f32(&format!("celt_iir mem it={it} n={n} ord={ord}"), &mr, &mc);
    }
}

#[test]
fn celt_autocorr_matches() {
    let mut rng = Rng::new(0x100d);
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
        let x = signal(&mut rng, n, it);
        let window: Vec<f32> = (0..overlap)
            .map(|i| {
                let t = (i as f64 + 0.5) / overlap as f64;
                (0.5 * core::f64::consts::PI * t).sin() as f32
            })
            .collect();
        let mut ar = vec![0f32; lag + 1];
        let mut ac = vec![0f32; lag + 1];
        let sr = rl::_celt_autocorr(&x, &mut ar, &window, overlap, lag, n);
        let sc = c::celt_autocorr(&x, &mut ac, &window, overlap, lag, n);
        assert_eq!(sr, sc, "_celt_autocorr shift");
        assert_bits_eq_f32(
            &format!("_celt_autocorr it={it} n={n} lag={lag} overlap={overlap}"),
            &ar,
            &ac,
        );
    }
}

/// End-to-end encoder pitch pre-analysis chain (downsample -> search -> remove_doubling), stereo
/// and mono, on music/speech-like input at every frame size.
#[test]
fn encoder_pitch_chain_matches() {
    let mut rng = Rng::new(0x100e);
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
        let chans: Vec<Vec<f32>> = (0..cc)
            .map(|ch| (0..total).map(|i| pcm[i * cc + ch] * 32768.0).collect())
            .collect();
        let x: Vec<&[f32]> = chans.iter().map(|v| v.as_slice()).collect();
        let len = total >> 1;
        let mut br = vec![0f32; len];
        let mut bc = vec![0f32; len];
        rp::pitch_downsample(&x, &mut br, len, cc, 2);
        c::pitch_downsample(&x, &mut bc, len, cc, 2);
        assert_bits_eq_f32("chain downsample", &br, &bc);
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
            0.5,
        );
        let gc = c::remove_doubling(
            &bc,
            max_period as i32,
            min_period as i32,
            n as i32,
            &mut tc,
            prev,
            0.5,
        );
        assert_eq!(
            (tr, gr.to_bits()),
            (tc, gc.to_bits()),
            "chain remove_doubling"
        );
    }
}
