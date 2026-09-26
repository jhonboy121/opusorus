//! Differential tests for unit `silk_resampler` (`silk/resampler*.c`) vs the C oracle.
//!
//! Every function is compared bit-for-bit: outputs and the complete resampler state after every
//! call.

use opusorus::silk::resampler::{self as r, ResamplerFunction, SilkResamplerState};
use opusorus_conformance::{Rng, assert_slice_eq};
use opusorus_oracle::silk_resampler::{self as c, StateDump};

const API_RATES: &[i32] = if cfg!(feature = "qext") {
    &[8000, 12000, 16000, 24000, 48000, 96000]
} else {
    &[8000, 12000, 16000, 24000, 48000]
};
const INTERNAL_RATES: &[i32] = &[8000, 12000, 16000];
/// Rates that are never valid (plus 96 kHz without `qext`).
const BAD_RATES: &[i32] = &[
    0, -8000, 1000, 4000, 11025, 22050, 32000, 44100, 64000, 96000, 192000, 7999, 8001,
];

fn coefs_id(t: &[i16]) -> i32 {
    let tables: [&[i16]; 6] = [
        &r::SILK_RESAMPLER_3_4_COEFS,
        &r::SILK_RESAMPLER_2_3_COEFS,
        &r::SILK_RESAMPLER_1_2_COEFS,
        &r::SILK_RESAMPLER_1_3_COEFS,
        &r::SILK_RESAMPLER_1_4_COEFS,
        &r::SILK_RESAMPLER_1_6_COEFS,
    ];
    if t.is_empty() {
        return 0;
    }
    match tables
        .iter()
        .position(|x| x.as_ptr() == t.as_ptr() && x.len() == t.len())
    {
        Some(i) => i as i32 + 1,
        None => -1,
    }
}

/// Compares the Rust state with a C dump. The `sFIR` union view only matters for the function
/// that uses it; the other view is compared only when it is not aliased (both are zero then).
#[track_caller]
fn assert_state_eq(what: &str, s: &SilkResamplerState, d: &StateDump) {
    assert_eq!(s.s_iir, d.s_iir, "{what}: sIIR");
    assert_slice_eq(&format!("{what}: delayBuf"), &s.delay_buf, &d.delay_buf);
    let ints = [
        s.resampler_function.to_c(),
        s.batch_size,
        s.inv_ratio_q16,
        s.fir_order,
        s.fir_fracs,
        s.fs_in_khz,
        s.fs_out_khz,
        s.input_delay,
        coefs_id(s.coefs),
    ];
    let cints = [
        d.resampler_function,
        d.batch_size,
        d.inv_ratio_q16,
        d.fir_order,
        d.fir_fracs,
        d.fs_in_khz,
        d.fs_out_khz,
        d.input_delay,
        d.coefs_id,
    ];
    assert_eq!(
        ints, cints,
        "{what}: [function, batchSize, invRatio_Q16, FIR_Order, FIR_Fracs, Fs_in_kHz, \
         Fs_out_kHz, inputDelay, Coefs]"
    );
    if s.resampler_function != ResamplerFunction::IirFir {
        assert_slice_eq(&format!("{what}: sFIR.i32"), &s.s_fir_i32, &d.s_fir_i32);
    }
    if s.resampler_function != ResamplerFunction::DownFir {
        assert_slice_eq(&format!("{what}: sFIR.i16"), &s.s_fir_i16, &d.s_fir_i16);
    }
}

/// Test signal of `n` samples; `kind` selects the waveform.
fn signal(rng: &mut Rng, n: usize, kind: i32, phase: &mut f64, fs: i32) -> Vec<i16> {
    let mut out = Vec::with_capacity(n);
    let f = 50.0 + 20000.0 * f64::from(rng.next_u32() % 1000) / 1000.0;
    for _ in 0..n {
        *phase += f / f64::from(fs);
        let v = match kind {
            0 => rng.i16(),
            1 => (32767.0 * (2.0 * core::f64::consts::PI * *phase).sin()) as i16,
            2 => {
                if (*phase).fract() < 0.5 {
                    i16::MAX
                } else {
                    i16::MIN
                }
            }
            3 => 0,
            4 => (rng.i16() >> 8) + (1000.0 * (2.0 * core::f64::consts::PI * *phase).sin()) as i16,
            5 => {
                if rng.next_u32().is_multiple_of(2) {
                    i16::MAX
                } else {
                    i16::MIN
                }
            }
            _ => (rng.i16() >> 4) + (8000.0 * (0.37 * *phase).sin()) as i16,
        };
        out.push(v);
    }
    out
}

/// Every rate pair `silk_resampler_init` accepts (encoder and decoder tables).
fn valid_pairs() -> Vec<(i32, i32, i32)> {
    let mut v = Vec::new();
    for &fin in API_RATES {
        for &fout in INTERNAL_RATES {
            if c_accepts(fin, fout, 1) {
                v.push((fin, fout, 1));
            }
        }
    }
    for &fin in INTERNAL_RATES {
        for &fout in API_RATES {
            v.push((fin, fout, 0));
        }
    }
    v
}

#[test]
fn rom_tables_match() {
    let c = c::rom();
    let mut rust: Vec<i16> = vec![r::SILK_RESAMPLER_DOWN2_0, r::SILK_RESAMPLER_DOWN2_1];
    rust.extend_from_slice(&r::SILK_RESAMPLER_UP2_HQ_0);
    rust.extend_from_slice(&r::SILK_RESAMPLER_UP2_HQ_1);
    rust.extend_from_slice(&r::SILK_RESAMPLER_3_4_COEFS);
    rust.extend_from_slice(&r::SILK_RESAMPLER_2_3_COEFS);
    rust.extend_from_slice(&r::SILK_RESAMPLER_1_2_COEFS);
    rust.extend_from_slice(&r::SILK_RESAMPLER_1_3_COEFS);
    rust.extend_from_slice(&r::SILK_RESAMPLER_1_4_COEFS);
    rust.extend_from_slice(&r::SILK_RESAMPLER_1_6_COEFS);
    rust.extend_from_slice(&r::SILK_RESAMPLER_2_3_COEFS_LQ);
    for row in &r::SILK_RESAMPLER_FRAC_FIR_12 {
        rust.extend_from_slice(row);
    }
    assert_slice_eq("resampler ROM", &rust, &c);
}

#[test]
fn rate_id_values() {
    let expect = [
        (8000, 0),
        (12000, 1),
        (16000, 2),
        (24000, 3),
        (48000, 4),
        (96000, 5),
    ];
    for (rate, id) in expect {
        assert_eq!(r::rate_id(rate), id, "rateID({rate})");
    }
}

/// Whether C `silk_resampler_init` accepts the pair. The hardened oracle aborts
/// (`celt_assert( 0 )`) instead of returning `-1` on the others, so it is only called for these.
fn c_accepts(fin: i32, fout: i32, for_enc: i32) -> bool {
    if for_enc != 0 {
        API_RATES.contains(&fin)
            && INTERNAL_RATES.contains(&fout)
            // 96 kHz -> 8/12 kHz (1:12, 1:8) has no filter.
            && !(fin == 96000 && fout != 16000)
    } else {
        INTERNAL_RATES.contains(&fin) && API_RATES.contains(&fout)
    }
}

#[test]
fn init_all_rate_pairs() {
    let mut rates: Vec<i32> = API_RATES.to_vec();
    rates.extend_from_slice(BAD_RATES);
    rates.sort_unstable();
    rates.dedup();
    let mut cs = c::Resampler::new();
    let mut n_ok = 0;
    for for_enc in [0, 1, 2, -1] {
        for &fin in &rates {
            for &fout in &rates {
                // Start from a dirty state to check that init clears everything.
                let mut s = SilkResamplerState {
                    s_iir: [7; 6],
                    delay_buf: [3; 96],
                    batch_size: 99,
                    ..SilkResamplerState::default()
                };
                let ret = r::silk_resampler_init(&mut s, fin, fout, for_enc);
                let what = format!("init({fin}, {fout}, {for_enc})");
                assert_eq!(
                    SilkResamplerState::new(fin, fout, for_enc != 0).is_ok(),
                    ret == 0,
                    "{what}: new()"
                );
                if !c_accepts(fin, fout, for_enc) {
                    assert_eq!(ret, -1, "{what}: must be rejected");
                    continue;
                }
                let cret = cs.init(fin, fout, for_enc);
                assert_eq!(ret, cret, "{what}: return value");
                assert_eq!(ret, 0, "{what}");
                assert_state_eq(&what, &s, &cs.dump());
                n_ok += 1;
            }
        }
    }
    // Decoder: 3 x 5 pairs (3 x 6 with qext); encoder (for_enc = 1, 2, -1): 5 x 3 pairs
    // (6 x 3 - 2 with qext).
    let expect_ok = if cfg!(feature = "qext") {
        18 + 3 * 16
    } else {
        15 + 3 * 15
    };
    assert_eq!(n_ok, expect_ok);
}

/// Output buffer size used for a call with `n` input samples (generous; the tail is compared
/// too, as a canary).
const fn out_len(s: &SilkResamplerState, n: usize) -> usize {
    n * s.fs_out_khz as usize / s.fs_in_khz as usize + s.fs_out_khz as usize + 64
}

#[test]
fn stream_all_pairs() {
    let mut rng = Rng::new(0x5EED_0001);
    let mut total_calls = 0usize;
    for (fin, fout, for_enc) in valid_pairs() {
        for run in 0..10 {
            let mut s = SilkResamplerState::default();
            let mut cs = c::Resampler::new();
            let ret = r::silk_resampler_init(&mut s, fin, fout, for_enc);
            assert_eq!(ret, cs.init(fin, fout, for_enc));
            if ret != 0 {
                continue;
            }
            let fs_in_khz = s.fs_in_khz as usize;
            let mut phase = 0.0f64;
            let calls = 60;
            for call in 0..calls {
                let n = match (run + call) % 4 {
                    // Typical SILK/Opus frame sizes (multiples of 1 ms, up to 60 ms).
                    0 => fs_in_khz * [1, 2, 5, 10, 20, 40, 60][rng.range_i32(0, 6) as usize],
                    // Random lengths >= 1 ms (exercise batch splitting and odd lengths).
                    1 => rng.range_i32(fs_in_khz as i32, 45 * fs_in_khz as i32) as usize,
                    // Around the 10 ms batch boundary (+/- 2 samples).
                    2 => {
                        let base = fs_in_khz * (10 * rng.range_i32(1, 3) as usize + 1);
                        (base as i32 + rng.range_i32(-2, 2)) as usize
                    }
                    // Exactly 1 ms (second private call gets 0 samples) or 1 ms + 1.
                    _ => fs_in_khz + rng.range_i32(0, 1) as usize,
                };
                let kind = (run * 3 + call) % 7;
                let input = signal(&mut rng, n, kind, &mut phase, fin);
                let len = out_len(&s, n);
                let mut out = vec![0x5A5Au16 as i16; len];
                let mut cout = out.clone();
                let ret = r::silk_resampler(&mut s, &mut out, &input, n as i32);
                let cret = cs.resample(&mut cout, &input);
                let what = format!("{fin}->{fout} enc={for_enc} run={run} call={call} n={n}");
                assert_eq!(ret, cret, "{what}: ret");
                assert_slice_eq(&format!("{what}: out"), &out, &cout);
                assert_state_eq(&what, &s, &cs.dump());
                total_calls += 1;
            }
        }
    }
    assert!(total_calls > 15000, "only {total_calls} calls");
}

#[test]
fn private_resamplers_random_state() {
    let mut rng = Rng::new(0x5EED_0002);
    for (fin, fout, for_enc) in valid_pairs() {
        for iter in 0..40 {
            let mut s = SilkResamplerState::default();
            let mut cs = c::Resampler::new();
            assert_eq!(r::silk_resampler_init(&mut s, fin, fout, for_enc), 0);
            assert_eq!(cs.init(fin, fout, for_enc), 0);
            // Random but realistic filter memories (Q8/Q10 magnitudes of 16-bit audio).
            let mut s_iir = [0i32; 6];
            for v in &mut s_iir {
                *v = rng.range_i32(-(1 << 24), 1 << 24);
            }
            let mut s_fir_i32 = [0i32; 36];
            for v in &mut s_fir_i32 {
                *v = rng.range_i32(-(1 << 24), 1 << 24);
            }
            let mut s_fir_i16 = [0i16; 36];
            for v in &mut s_fir_i16 {
                *v = rng.i16();
            }
            let mut delay_buf = [0i16; 96];
            for v in &mut delay_buf {
                *v = rng.i16();
            }
            let use_i16 = s.resampler_function == ResamplerFunction::IirFir;
            if !matches!(
                s.resampler_function,
                ResamplerFunction::IirFir | ResamplerFunction::DownFir
            ) {
                // sFIR is unused by these; keep it zero so both union views stay comparable.
                s_fir_i32 = [0; 36];
            }
            cs.set_mem(&s_iir, &s_fir_i32, &s_fir_i16, use_i16, &delay_buf);
            s.s_iir = s_iir;
            if use_i16 {
                s.s_fir_i16 = s_fir_i16;
            } else {
                s.s_fir_i32 = s_fir_i32;
            }
            s.delay_buf = delay_buf;
            assert_state_eq("set_mem", &s, &cs.dump());

            let batch = s.batch_size as usize;
            for call in 0..8 {
                // Includes 0, 1, batch +/- 1 (the down_FIR `inLen > 1` quirk) and multi-batch.
                let n = match (iter + call) % 5 {
                    0 => rng.range_i32(0, 3) as usize,
                    1 => (batch as i32 + rng.range_i32(-1, 1)) as usize,
                    2 => 2 * batch + rng.range_i32(0, 2) as usize,
                    3 => rng.range_i32(0, 4 * batch as i32) as usize,
                    _ => batch * rng.range_i32(1, 4) as usize + 1,
                };
                let mut phase = 0.0;
                let input = signal(&mut rng, n, (iter + call) % 7, &mut phase, fin);
                let len = 2 * n * s.fs_out_khz as usize / s.fs_in_khz as usize + 64;
                let mut out = vec![0x1234i16; len];
                let mut cout = out.clone();
                match s.resampler_function {
                    ResamplerFunction::IirFir => {
                        r::silk_resampler_private_iir_fir(&mut s, &mut out, &input, n as i32);
                        cs.private_iir_fir(&mut cout, &input);
                    }
                    ResamplerFunction::DownFir => {
                        r::silk_resampler_private_down_fir(&mut s, &mut out, &input, n as i32);
                        cs.private_down_fir(&mut cout, &input);
                    }
                    ResamplerFunction::Up2Hq | ResamplerFunction::Copy => {
                        r::silk_resampler_private_up2_hq_wrapper(
                            &mut s, &mut out, &input, n as i32,
                        );
                        cs.private_up2_hq_wrapper(&mut cout, &input);
                    }
                }
                let what = format!("{fin}->{fout} enc={for_enc} iter={iter} call={call} n={n}");
                assert_slice_eq(&format!("{what}: out"), &out, &cout);
                assert_state_eq(&what, &s, &cs.dump());
            }
        }
    }
}

#[test]
fn down2_matches() {
    let mut rng = Rng::new(0x5EED_0003);
    for iter in 0..3000 {
        let mut s = [
            rng.range_i32(-(1 << 25), 1 << 25),
            rng.range_i32(-(1 << 25), 1 << 25),
        ];
        if iter % 5 == 0 {
            s = [0, 0];
        }
        let mut cst = s;
        let mut phase = 0.0;
        for _ in 0..3 {
            let n = rng.range_i32(0, 1000) as usize;
            let input = signal(&mut rng, n, iter % 7, &mut phase, 16000);
            let mut out = vec![0i16; n / 2];
            r::silk_resampler_down2(&mut s, &mut out, &input, n as i32);
            let cout = c::down2(&mut cst, &input);
            assert_slice_eq(&format!("down2 iter={iter} n={n}"), &out, &cout);
            assert_eq!(s, cst, "down2 state iter={iter} n={n}");
        }
    }
}

#[test]
fn down2_3_matches() {
    let mut rng = Rng::new(0x5EED_0004);
    for iter in 0..2000 {
        let mut s = [0i32; 6];
        if iter % 4 != 0 {
            for v in &mut s {
                *v = rng.range_i32(-(1 << 24), 1 << 24);
            }
        }
        let mut cst = s;
        let mut phase = 0.0;
        for _ in 0..3 {
            let n = match iter % 3 {
                0 => rng.range_i32(0, 12) as usize,
                1 => rng.range_i32(0, 1500) as usize,
                _ => (480 * rng.range_i32(1, 3) + rng.range_i32(-3, 3)) as usize,
            };
            let input = signal(&mut rng, n, iter % 7, &mut phase, 48000);
            let len = 2 * n / 3 + 8;
            let mut out = vec![0x7777i16; len];
            let mut cout = out.clone();
            r::silk_resampler_down2_3(&mut s, &mut out, &input, n as i32);
            c::down2_3(&mut cst, &mut cout, &input);
            assert_slice_eq(&format!("down2_3 iter={iter} n={n}"), &out, &cout);
            assert_eq!(s, cst, "down2_3 state iter={iter} n={n}");
        }
    }
}

#[test]
fn ar2_matches() {
    let mut rng = Rng::new(0x5EED_0005);
    let rom: [&[i16]; 7] = [
        &r::SILK_RESAMPLER_3_4_COEFS,
        &r::SILK_RESAMPLER_2_3_COEFS,
        &r::SILK_RESAMPLER_1_2_COEFS,
        &r::SILK_RESAMPLER_1_3_COEFS,
        &r::SILK_RESAMPLER_1_4_COEFS,
        &r::SILK_RESAMPLER_1_6_COEFS,
        &r::SILK_RESAMPLER_2_3_COEFS_LQ,
    ];
    for iter in 0..4000 {
        let a: [i16; 2] = if iter % 2 == 0 {
            let t = rom[(iter / 2) % rom.len()];
            [t[0], t[1]]
        } else {
            // Random stable AR2 with bounded gain: |a0| + |a1| < 0.9 * 2^14.
            let a1 = rng.range_i32(-14000, 14000);
            let lim = 14745 - a1.abs();
            [rng.range_i32(-lim, lim) as i16, a1 as i16]
        };
        let mut s = [
            rng.range_i32(-(1 << 22), 1 << 22),
            rng.range_i32(-(1 << 22), 1 << 22),
        ];
        let mut cst = s;
        let n = rng.range_i32(0, 700) as usize;
        let mut phase = 0.0;
        let input = signal(&mut rng, n, (iter % 7) as i32, &mut phase, 24000);
        let mut out = vec![0i32; n];
        r::silk_resampler_private_ar2(&mut s, &mut out, &input, &a, n as i32);
        let cout = c::private_ar2(&mut cst, &input, &a);
        assert_slice_eq(&format!("AR2 iter={iter} a={a:?}"), &out, &cout);
        assert_eq!(s, cst, "AR2 state iter={iter}");
    }
}

#[test]
fn up2_hq_matches() {
    let mut rng = Rng::new(0x5EED_0006);
    for iter in 0..3000 {
        let mut s = [0i32; 6];
        if iter % 4 != 0 {
            for v in &mut s {
                *v = rng.range_i32(-(1 << 25), 1 << 25);
            }
        }
        let mut cst = s;
        let mut phase = 0.0;
        for _ in 0..3 {
            let n = rng.range_i32(0, 700) as usize;
            let input = signal(&mut rng, n, iter % 7, &mut phase, 16000);
            let mut out = vec![0i16; 2 * n];
            r::silk_resampler_private_up2_hq(&mut s, &mut out, &input, n as i32);
            let cout = c::private_up2_hq(&mut cst, &input);
            assert_slice_eq(&format!("up2_HQ iter={iter} n={n}"), &out, &cout);
            assert_eq!(s, cst, "up2_HQ state iter={iter} n={n}");
        }
    }
}

#[test]
fn iir_fir_interpol_matches() {
    let mut rng = Rng::new(0x5EED_0007);
    for iter in 0..4000 {
        let max_index = match iter % 4 {
            0 => 0,
            1 => rng.range_i32(1, 1 << 16),
            _ => rng.range_i32(1, 400 << 16),
        };
        let inc = match iter % 3 {
            0 => rng.range_i32(1 << 14, 1 << 18),
            1 => rng.range_i32(1000, 1 << 16),
            // Ratios actually produced by silk_resampler_init for IIR_FIR states.
            _ => [
                65536 / 3 * 2 + 2,
                43691,
                32768,
                21846,
                16384,
                10923,
                16386,
                32770,
            ][rng.range_i32(0, 7) as usize],
        };
        let n_out = if max_index > 0 {
            ((max_index as i64 + inc as i64 - 1) / inc as i64) as usize
        } else {
            0
        };
        let buf_len = (max_index.max(1) as usize >> 16) + 8 + rng.range_i32(0, 4) as usize;
        let buf: Vec<i16> = (0..buf_len)
            .map(|_| {
                if iter % 5 == 0 {
                    if rng.next_u32().is_multiple_of(2) {
                        i16::MAX
                    } else {
                        i16::MIN
                    }
                } else {
                    rng.i16()
                }
            })
            .collect();
        let mut out = vec![0i16; n_out + 4];
        let mut cout = out.clone();
        let n = r::silk_resampler_private_iir_fir_interpol(&mut out, &buf, max_index, inc);
        let cn = c::private_iir_fir_interpol(&mut cout, &buf, max_index, inc);
        assert_eq!(n, cn, "IIR_FIR_INTERPOL count iter={iter}");
        assert_slice_eq(&format!("IIR_FIR_INTERPOL iter={iter}"), &out, &cout);
    }
}

#[test]
fn down_fir_interpol_matches() {
    let mut rng = Rng::new(0x5EED_0008);
    // (coefs id, table, FIR_Order, FIR_Fracs) as set up by silk_resampler_init.
    let cfgs: [(i32, &[i16], i32, i32); 6] = [
        (1, &r::SILK_RESAMPLER_3_4_COEFS, 18, 3),
        (2, &r::SILK_RESAMPLER_2_3_COEFS, 18, 2),
        (3, &r::SILK_RESAMPLER_1_2_COEFS, 24, 1),
        (4, &r::SILK_RESAMPLER_1_3_COEFS, 36, 1),
        (5, &r::SILK_RESAMPLER_1_4_COEFS, 36, 1),
        (6, &r::SILK_RESAMPLER_1_6_COEFS, 36, 1),
    ];
    for iter in 0..6000 {
        let (id, table, order, fracs) = cfgs[iter % cfgs.len()];
        let max_index = match iter % 4 {
            0 => 0,
            1 => rng.range_i32(1, 1 << 16),
            _ => rng.range_i32(1, 1000 << 16),
        };
        let inc = match iter % 3 {
            0 => rng.range_i32(1 << 16, 7 << 16),
            1 => rng.range_i32(1 << 14, 1 << 18),
            _ => [87382, 98304, 131072, 196608, 262144, 393216][rng.range_i32(0, 5) as usize],
        };
        let n_out = if max_index > 0 {
            ((max_index as i64 + inc as i64 - 1) / inc as i64) as usize
        } else {
            0
        };
        let buf_len = (max_index.max(1) as usize >> 16) + order as usize + 2;
        // Q8 AR2 output magnitudes; pairwise sums must not overflow i32.
        let mag = if iter % 7 == 0 { 1 << 29 } else { 1 << 25 };
        let buf: Vec<i32> = (0..buf_len).map(|_| rng.range_i32(-mag, mag)).collect();
        let mut out = vec![0i16; n_out + 4];
        let mut cout = out.clone();
        let n = r::silk_resampler_private_down_fir_interpol(
            &mut out,
            &buf,
            &table[2..],
            order,
            fracs,
            max_index,
            inc,
        );
        let cn = c::private_down_fir_interpol(&mut cout, &buf, id, order, fracs, max_index, inc);
        assert_eq!(n, cn, "down_FIR_INTERPOL count iter={iter}");
        assert_slice_eq(
            &format!("down_FIR_INTERPOL iter={iter} cfg={id}"),
            &out,
            &cout,
        );
    }
}

#[test]
fn reinit_mid_stream() {
    // Re-initialising (as control_codec.c does on rate changes) must fully reset the state.
    let mut rng = Rng::new(0x5EED_0009);
    let pairs = valid_pairs();
    let mut s = SilkResamplerState::default();
    let mut cs = c::Resampler::new();
    let mut phase = 0.0;
    for step in 0..400 {
        let (fin, fout, for_enc) = pairs[rng.range_i32(0, pairs.len() as i32 - 1) as usize];
        let ret = r::silk_resampler_init(&mut s, fin, fout, for_enc);
        assert_eq!(ret, cs.init(fin, fout, for_enc), "step {step}");
        assert_state_eq(&format!("reinit step {step}"), &s, &cs.dump());
        if ret != 0 {
            continue;
        }
        for _ in 0..3 {
            let n = s.fs_in_khz as usize * rng.range_i32(1, 25) as usize;
            let input = signal(&mut rng, n, step % 7, &mut phase, fin);
            let len = out_len(&s, n);
            let mut out = vec![0i16; len];
            let mut cout = out.clone();
            r::silk_resampler(&mut s, &mut out, &input, n as i32);
            cs.resample(&mut cout, &input);
            assert_slice_eq(&format!("reinit step {step}: out"), &out, &cout);
            assert_state_eq(&format!("reinit step {step}"), &s, &cs.dump());
        }
    }
}
