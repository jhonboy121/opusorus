//! Differential tests for unit `celt_modes`: static CELT modes, mode lookup/creation
//! (`modes.c`), Laplace coding (`laplace.c`), PVQ codeword coding (`cwrs.c`) and the bit
//! allocation (`rate.c`) vs the C oracle.

use opusorus::celt::entcode::EcCoder;
use opusorus::celt::entdec::EcDec;
use opusorus::celt::entenc::EcEnc;
use opusorus::celt::static_modes::{CeltMode, MODE48000_960_120, PulseCache};
use opusorus::celt::{cwrs, laplace, modes, rate};
use opusorus_conformance::{Rng, assert_bits_eq_f32, assert_slice_eq};
use opusorus_oracle::celt_modes as c;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn cache_data(p: &PulseCache) -> c::PulseCache {
    c::PulseCache {
        size: p.size,
        index: p.index.to_vec(),
        bits: p.bits.to_vec(),
        caps: p.caps.to_vec(),
    }
}

#[track_caller]
fn assert_cache_eq(what: &str, r: &c::PulseCache, o: &c::PulseCache) {
    assert_eq!(r.size, o.size, "{what}: size");
    assert_slice_eq(&format!("{what}: index"), &r.index, &o.index);
    assert_slice_eq(&format!("{what}: bits"), &r.bits, &o.bits);
    assert_slice_eq(&format!("{what}: caps"), &r.caps, &o.caps);
}

/// Compares every field of a Rust mode against the oracle's copy.
#[track_caller]
fn assert_mode_eq(what: &str, m: &CeltMode, o: &c::ModeData, with_mdct: bool) {
    assert_eq!(m.fs, o.fs, "{what}: Fs");
    assert_eq!(m.overlap, o.overlap, "{what}: overlap");
    assert_eq!(m.nb_ebands, o.nb_ebands, "{what}: nbEBands");
    assert_eq!(m.eff_ebands, o.eff_ebands, "{what}: effEBands");
    assert_bits_eq_f32(&format!("{what}: preemph"), &m.preemph, &o.preemph);
    let nb = m.nb_ebands as usize;
    assert_slice_eq(&format!("{what}: eBands"), &m.e_bands[..=nb], &o.e_bands);
    assert_eq!(m.max_lm, o.max_lm, "{what}: maxLM");
    assert_eq!(m.nb_short_mdcts, o.nb_short_mdcts, "{what}: nbShortMdcts");
    assert_eq!(
        m.short_mdct_size, o.short_mdct_size,
        "{what}: shortMdctSize"
    );
    assert_eq!(
        m.nb_alloc_vectors, o.nb_alloc_vectors,
        "{what}: nbAllocVectors"
    );
    assert_slice_eq(
        &format!("{what}: allocVectors"),
        &m.alloc_vectors[..m.nb_alloc_vectors as usize * nb],
        &o.alloc_vectors,
    );
    assert_slice_eq(&format!("{what}: logN"), &m.log_n[..nb], &o.log_n);
    assert_bits_eq_f32(&format!("{what}: window"), &m.window, &o.window);
    if with_mdct {
        assert_eq!(m.mdct.n, o.mdct_n, "{what}: mdct.n");
        assert_eq!(m.mdct.maxshift, o.mdct_maxshift, "{what}: mdct.maxshift");
        assert_bits_eq_f32(&format!("{what}: mdct.trig"), &m.mdct.trig, &o.trig);
        assert_eq!(o.kfft.len(), m.mdct.maxshift as usize + 1);
        for (i, of) in o.kfft.iter().enumerate() {
            let rf = &m.mdct.kfft[i];
            let w = format!("{what}: kfft[{i}]");
            assert_eq!(rf.nfft, of.nfft, "{w}.nfft");
            assert_eq!(rf.scale.to_bits(), of.scale.to_bits(), "{w}.scale");
            assert_eq!(rf.shift, of.shift, "{w}.shift");
            assert_slice_eq(&format!("{w}.factors"), &rf.factors, &of.factors);
            assert_slice_eq(&format!("{w}.bitrev"), &rf.bitrev, &of.bitrev);
            let rr: Vec<f32> = rf.twiddles.iter().map(|t| t.r).collect();
            let ri: Vec<f32> = rf.twiddles.iter().map(|t| t.i).collect();
            let or: Vec<f32> = of.twiddles.iter().map(|t| t.0).collect();
            let oi: Vec<f32> = of.twiddles.iter().map(|t| t.1).collect();
            assert_bits_eq_f32(&format!("{w}.twiddles.r"), &rr, &or);
            assert_bits_eq_f32(&format!("{w}.twiddles.i"), &ri, &oi);
        }
    }
    assert_cache_eq(&format!("{what}: cache"), &cache_data(&m.cache), &o.cache);
    #[cfg(feature = "qext")]
    {
        match &o.qext_cache {
            Some(q) => assert_cache_eq(
                &format!("{what}: qext_cache"),
                &cache_data(&m.qext_cache),
                q,
            ),
            None => assert!(
                m.qext_cache.index.is_empty() && m.qext_cache.size == 0,
                "{what}: qext_cache should be empty"
            ),
        }
    }
    #[cfg(not(feature = "qext"))]
    assert!(o.qext_cache.is_none());
}

/// Standard CELT `init_caps` (celt.c), used to build realistic `cap` inputs.
fn init_caps(m: &CeltMode, lm: i32, cc: i32) -> Vec<i32> {
    let nb = m.nb_ebands;
    (0..nb)
        .map(|i| {
            let n = (i32::from(m.e_bands[i as usize + 1]) - i32::from(m.e_bands[i as usize])) << lm;
            ((i32::from(m.cache.caps[(nb * (2 * lm + cc - 1) + i) as usize]) + 64) * cc * n) >> 2
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------------------------

#[test]
fn static_mode_48000_960_matches_oracle() {
    let o = c::Mode::create(48000, 960).expect("static mode").data(true);
    assert_mode_eq("mode48000_960_120", &MODE48000_960_120, &o, true);
}

#[cfg(feature = "qext")]
#[test]
fn static_mode_96000_1920_matches_oracle() {
    use opusorus::celt::static_modes::MODE96000_1920_240;
    let o = c::Mode::create(96000, 1920)
        .expect("static mode")
        .data(true);
    assert_mode_eq("mode96000_1920_240", &MODE96000_1920_240, &o, true);
}

const RATES: [i32; 14] = [
    0, 7999, 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 96001,
];

/// Static lookup path: same result (and same static mode) as C for all frame sizes. Without
/// custom modes, anything else is `OPUS_BAD_ARG` in both.
#[test]
fn mode_create_static_lookup() {
    for &fs in &RATES {
        for frame_size in -4..=2100 {
            let r = modes::opus_custom_mode_create(fs, frame_size);
            #[cfg(feature = "custom-modes")]
            if r.is_err()
                && rust_custom_create(fs, frame_size).err() == Some(opusorus::Error::AllocFail)
            {
                // C would take its (buggy) failure path; see custom_mode_create_matches_oracle.
                continue;
            }
            let o = c::Mode::create(fs, frame_size);
            match (&r, &o) {
                (Ok(m), Ok(om)) => {
                    let od = om.data(false);
                    assert_eq!((m.fs, m.short_mdct_size), (od.fs, od.short_mdct_size));
                }
                (Err(e), Err(code)) => assert_eq!(e.code(), *code, "fs={fs} n={frame_size}"),
                #[cfg(feature = "custom-modes")]
                (Err(e), Ok(_)) => assert_eq!(*e, opusorus::Error::BadArg),
                _ => panic!("fs={fs} frame_size={frame_size}: rust {r:?} vs C {o:?}"),
            }
        }
    }
}

#[cfg(feature = "qext")]
#[test]
fn compute_qext_mode_matches_oracle() {
    use opusorus::celt::static_modes::MODE96000_1920_240;
    for (m, fs, n) in [
        (&MODE48000_960_120, 48000, 960),
        (&MODE96000_1920_240, 96000, 1920),
    ] {
        let o = c::Mode::create(fs, n).unwrap();
        let (ints, eb, ln) = o.compute_qext_mode();
        let q = modes::compute_qext_mode(m);
        assert_eq!(
            [q.nb_ebands, q.eff_ebands, q.nb_alloc_vectors, q.cache.size],
            ints
        );
        assert_slice_eq("qext eBands", &q.e_bands[..15], &eb);
        assert_slice_eq("qext logN", &q.log_n[..14], &ln);
        assert_eq!(q.cache, m.qext_cache);
        assert!(q.alloc_vectors.is_empty());
    }
}

/// Whether `kf_factor` accepts `n` (only radices 2, 3, 4, 5).
#[cfg(feature = "custom-modes")]
fn fft_size_ok(mut n: i32) -> bool {
    for p in [2, 3, 5] {
        while n % p == 0 {
            n /= p;
        }
    }
    n == 1
}

/// Custom-mode creation with a placeholder MDCT that fails exactly when C's clt_mdct_init does
/// (an FFT size with a prime factor above 5).
#[cfg(feature = "custom-modes")]
fn rust_custom_create(fs: i32, n: i32) -> opusorus::Result<std::borrow::Cow<'static, CeltMode>> {
    modes::opus_custom_mode_create_with(fs, n, |len, maxshift| {
        (0..=maxshift)
            .all(|i| fft_size_ok(len >> 2 >> i))
            .then(|| MODE48000_960_120.mdct.clone())
    })
}

/// Dynamic custom-mode creation (everything except the MDCT, which belongs to the `celt_fft`
/// unit: a placeholder MDCT is supplied and not compared).
#[cfg(feature = "custom-modes")]
#[test]
fn custom_mode_create_matches_oracle() {
    let mut rng = Rng::new(0xC0DE);
    let mut cases: Vec<(i32, i32)> = Vec::new();
    for &fs in &RATES {
        for n in [
            40, 60, 64, 80, 90, 96, 100, 120, 128, 160, 180, 240, 256, 320, 360, 480, 512,
        ] {
            for mul in [1, 2, 4, 8] {
                cases.push((fs, n * mul));
            }
        }
    }
    for _ in 0..1500 {
        let fs = rng.range_i32(7900, 96100);
        let n = rng.range_i32(30, 2100);
        cases.push((fs, n));
    }
    let mut created = 0;
    let mut skipped = 0;
    for (fs, n) in cases {
        let r = rust_custom_create(fs, n);
        if r == Err(opusorus::Error::AllocFail) {
            // The C `failure:` path frees uninitialised fields of the half-built mode (libopus
            // bug: double free / invalid free), so it cannot be exercised in the oracle.
            skipped += 1;
            continue;
        }
        let o = c::Mode::create(fs, n);
        match (&r, &o) {
            (Ok(m), Ok(om)) => {
                let od = om.data(false);
                assert_mode_eq(&format!("custom fs={fs} n={n}"), m, &od, false);
                created += 1;
                // The pulse cache and allocation must also work on custom modes.
                if created % 8 == 0 {
                    check_bits2pulses(m, om, false);
                    check_alloc(&mut rng, m, om, false, 10);
                }
            }
            (Err(e), Err(code)) => assert_eq!(e.code(), *code, "fs={fs} n={n}"),
            _ => panic!("fs={fs} n={n}: rust {r:?} vs C {o:?}"),
        }
    }
    assert!(
        created > 300,
        "only {created} modes created ({skipped} skipped)"
    );
    // Without an MDCT the public constructor fails for non-static modes, like an allocation
    // failure in C.
    assert_eq!(
        modes::opus_custom_mode_create_custom(44100, 896).err(),
        Some(opusorus::Error::AllocFail)
    );
    assert!(modes::opus_custom_mode_create_custom(48000, 480).is_ok());
}

// ---------------------------------------------------------------------------------------------
// rate.h helpers
// ---------------------------------------------------------------------------------------------

#[test]
fn get_pulses_matches_oracle() {
    for i in 0..=48 {
        assert_eq!(rate::get_pulses(i), c::get_pulses(i), "i={i}");
    }
}

fn check_bits2pulses(m: &CeltMode, o: &c::Mode, use_qext: bool) {
    for lm in -1..=m.max_lm {
        for band in 0..m.nb_ebands {
            let row = m.cache.index[((lm + 1) * m.nb_ebands + band) as usize];
            if row < 0 {
                continue;
            }
            let maxp = i32::from(m.cache.bits[row as usize]);
            for bits in -2..=300 {
                assert_eq!(
                    rate::bits2pulses(m, band, lm, bits),
                    o.bits2pulses(use_qext, band, lm, bits),
                    "bits2pulses band={band} lm={lm} bits={bits} qext={use_qext}"
                );
            }
            for p in 0..=maxp {
                assert_eq!(
                    rate::pulses2bits(m, band, lm, p),
                    o.pulses2bits(use_qext, band, lm, p),
                    "pulses2bits band={band} lm={lm} p={p} qext={use_qext}"
                );
            }
        }
    }
}

#[test]
fn bits2pulses_pulses2bits_match_oracle() {
    let o = c::Mode::create(48000, 960).unwrap();
    check_bits2pulses(&MODE48000_960_120, &o, false);
    #[cfg(feature = "qext")]
    {
        use opusorus::celt::static_modes::MODE96000_1920_240;
        check_bits2pulses(&modes::compute_qext_mode(&MODE48000_960_120), &o, true);
        let o96 = c::Mode::create(96000, 1920).unwrap();
        check_bits2pulses(&MODE96000_1920_240, &o96, false);
        check_bits2pulses(&modes::compute_qext_mode(&MODE96000_1920_240), &o96, true);
    }
}

// ---------------------------------------------------------------------------------------------
// Laplace
// ---------------------------------------------------------------------------------------------

const fn random_laplace_value(rng: &mut Rng) -> i32 {
    let mag = match rng.range_i32(0, 9) {
        0..=3 => rng.range_i32(0, 2),
        4..=6 => rng.range_i32(0, 8),
        7 => rng.range_i32(0, 40),
        8 => rng.range_i32(0, 300),
        _ => rng.range_i32(0, 5000),
    };
    if rng.range_i32(0, 1) == 0 { mag } else { -mag }
}

fn random_laplace_ops(rng: &mut Rng, n: usize, p0_ratio: i32) -> Vec<[i32; 4]> {
    (0..n)
        .map(|_| {
            if rng.range_i32(0, 99) < p0_ratio {
                // ec_laplace_encode_p0: p0 in [1, 32766], decay < 32768.
                let p0 = match rng.range_i32(0, 3) {
                    0 => rng.range_i32(1, 50),
                    1 => rng.range_i32(32700, 32766),
                    _ => rng.range_i32(1, 32766),
                };
                let decay = match rng.range_i32(0, 3) {
                    0 => rng.range_i32(0, 10),
                    1 => rng.range_i32(32000, 32767),
                    _ => rng.range_i32(0, 32767),
                };
                let v = random_laplace_value(rng).clamp(-200, 200);
                [1, v, p0, decay]
            } else {
                // ec_laplace_encode: fs in (0, 32736], decay in (0, 11456].
                let fs = match rng.range_i32(0, 4) {
                    0 => rng.range_i32(1, 64),
                    1 => rng.range_i32(32600, 32736),
                    _ => rng.range_i32(1, 32736),
                };
                let decay = match rng.range_i32(0, 3) {
                    0 => rng.range_i32(1, 16),
                    1 => rng.range_i32(11000, 11456),
                    _ => rng.range_i32(1, 11456),
                };
                [0, random_laplace_value(rng), fs, decay]
            }
        })
        .collect()
}

fn rust_laplace_encode(ops: &[[i32; 4]], size: usize) -> c::LaplaceEnc {
    let mut buf = vec![0u8; size];
    let mut values = Vec::with_capacity(ops.len());
    let mut tells = Vec::with_capacity(ops.len());
    let (rng, error) = {
        let mut e = EcEnc::new(&mut buf);
        for o in ops {
            let mut v = o[1];
            if o[0] == 0 {
                laplace::ec_laplace_encode(&mut e, &mut v, o[2] as u32, o[3]);
            } else {
                laplace::ec_laplace_encode_p0(&mut e, v, o[2] as u16, o[3] as u16);
            }
            values.push(v);
            tells.push(e.tell_frac());
        }
        e.done();
        (e.rng, e.error)
    };
    c::LaplaceEnc {
        buf,
        values,
        tells,
        rng,
        error,
    }
}

fn rust_laplace_decode(ops: &[[i32; 3]], buf: &[u8]) -> c::LaplaceDec {
    let mut d = EcDec::new(buf);
    let mut values = Vec::with_capacity(ops.len());
    let mut tells = Vec::with_capacity(ops.len());
    for o in ops {
        values.push(if o[0] == 0 {
            laplace::ec_laplace_decode(&mut d, o[1] as u32, o[2])
        } else {
            laplace::ec_laplace_decode_p0(&mut d, o[1] as u16, o[2] as u16)
        });
        tells.push(d.tell_frac());
    }
    c::LaplaceDec {
        values,
        tells,
        rng: d.rng,
        error: d.error,
    }
}

#[test]
fn laplace_matches_oracle() {
    let mut rng = Rng::new(0x1A91);
    for iter in 0..3000 {
        let n = rng.range_i32(1, 120) as usize;
        let p0_ratio = [0, 30, 100][iter % 3];
        let ops = random_laplace_ops(&mut rng, n, p0_ratio);
        let size = if iter % 7 == 0 {
            rng.range_i32(1, 40) as usize
        } else {
            4000
        };
        let co = c::laplace_encode(&ops, size);
        let ro = rust_laplace_encode(&ops, size);
        assert_eq!(ro, co, "laplace encode mismatch at iter {iter}");
        let dops: Vec<[i32; 3]> = ops.iter().map(|o| [o[0], o[2], o[3]]).collect();
        let cd = c::laplace_decode(&dops, &co.buf);
        let rd = rust_laplace_decode(&dops, &co.buf);
        assert_eq!(rd, cd, "laplace decode mismatch at iter {iter}");
        if co.error == 0 {
            assert_eq!(rd.values, co.values, "laplace round trip at iter {iter}");
        }
        // Decoding random bytes exercises arbitrary symbol paths.
        let mut junk = vec![0u8; rng.range_i32(0, 64) as usize];
        rng.fill_bytes(&mut junk);
        assert_eq!(
            rust_laplace_decode(&dops, &junk),
            c::laplace_decode(&dops, &junk),
            "laplace junk decode at iter {iter}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// cwrs
// ---------------------------------------------------------------------------------------------

/// Largest column of each row of `CELT_PVQ_U_ROW` in this build.
#[cfg(any(feature = "custom-modes", feature = "qext"))]
const PVQ_ROW_MAX: [i32; 15] = [
    208, 208, 208, 208, 208, 208, 109, 60, 40, 29, 24, 20, 18, 16, 14,
];
#[cfg(not(any(feature = "custom-modes", feature = "qext")))]
const PVQ_ROW_MAX: [i32; 15] = [
    176, 176, 176, 176, 176, 176, 96, 54, 37, 28, 24, 19, 18, 16, 14,
];

/// Whether `V(n,k)` is covered by the table (it may still overflow 32 bits).
fn pvq_in_table(n: i32, k: i32) -> bool {
    let (lo, hi) = (n.min(k + 1), n.max(k + 1));
    lo <= 14 && hi <= PVQ_ROW_MAX[lo as usize]
}

/// Whether encode/decode of `(n,k)` is valid: `V(n,k)` is in the table and fits in 32 bits.
fn pvq_ok(n: i32, k: i32) -> bool {
    pvq_in_table(n, k) && u64::from(c::pvq_u(n, k)) + u64::from(c::pvq_u(n, k + 1)) < 1 << 32
}

fn random_pulses(rng: &mut Rng, n: i32, k: i32) -> Vec<i32> {
    let mut y = vec![0i32; n as usize];
    match rng.range_i32(0, 5) {
        0 => y[rng.range_i32(0, n - 1) as usize] = k,
        1 => y[(n - 1) as usize] = k,
        2 => y[0] = k,
        _ => {
            for _ in 0..k {
                y[rng.range_i32(0, n - 1) as usize] += 1;
            }
        }
    }
    for v in &mut y {
        if rng.range_i32(0, 1) == 1 {
            *v = -*v;
        }
    }
    y
}

#[test]
fn pvq_tables_match_oracle() {
    for n in 0..=210 {
        for k in 0..=210 {
            let (lo, hi) = (n.min(k), n.max(k));
            if lo <= 14 && hi <= PVQ_ROW_MAX[lo as usize] {
                assert_eq!(cwrs::celt_pvq_u(n, k), c::pvq_u(n, k), "U({n},{k})");
            }
            if pvq_in_table(n, k) {
                assert_eq!(cwrs::celt_pvq_v(n, k), c::pvq_v(n, k), "V({n},{k})");
            }
        }
    }
}

fn rust_cwrs_encode(ys: &[i32], ns: &[i32], ks: &[i32], size: usize) -> (Vec<u8>, u32, i32) {
    let mut buf = vec![0u8; size];
    let (rng, err) = {
        let mut e = EcEnc::new(&mut buf);
        let mut off = 0usize;
        for (&n, &k) in ns.iter().zip(ks) {
            cwrs::encode_pulses(&ys[off..off + n as usize], n, k, &mut e);
            off += n as usize;
        }
        e.done();
        (e.rng, e.error)
    };
    (buf, rng, err)
}

fn rust_cwrs_decode(ns: &[i32], ks: &[i32], buf: &[u8]) -> (Vec<i32>, Vec<f32>, u32, i32) {
    let mut d = EcDec::new(buf);
    let mut ys = vec![0i32; ns.iter().map(|&n| n as usize).sum()];
    let mut yy = Vec::with_capacity(ns.len());
    let mut off = 0usize;
    for (&n, &k) in ns.iter().zip(ks) {
        yy.push(cwrs::decode_pulses(
            &mut ys[off..off + n as usize],
            n,
            k,
            &mut d,
        ));
        off += n as usize;
    }
    (ys, yy, d.rng, d.error)
}

fn check_cwrs_batch(rng: &mut Rng, pairs: &[(i32, i32)], what: &str) {
    let ns: Vec<i32> = pairs.iter().map(|p| p.0).collect();
    let ks: Vec<i32> = pairs.iter().map(|p| p.1).collect();
    let ys: Vec<i32> = pairs
        .iter()
        .flat_map(|&(n, k)| random_pulses(rng, n, k))
        .collect();
    let size = 4 * pairs.len() + 16;
    let co = c::cwrs_encode(&ys, &ns, &ks, size);
    let ro = rust_cwrs_encode(&ys, &ns, &ks, size);
    assert_eq!(ro, co, "{what}: encode");
    assert_eq!(co.2, 0, "{what}: encoder error");
    let cd = c::cwrs_decode(&ns, &ks, &co.0);
    let rd = rust_cwrs_decode(&ns, &ks, &co.0);
    assert_eq!(rd.0, cd.0, "{what}: decoded pulses");
    assert_bits_eq_f32(&format!("{what}: yy"), &rd.1, &cd.1);
    assert_eq!((rd.2, rd.3), (cd.2, cd.3), "{what}: decoder state");
    assert_eq!(rd.0, ys, "{what}: round trip");
    // Random bytes decode to arbitrary codewords.
    let mut junk = vec![0u8; size];
    rng.fill_bytes(&mut junk);
    let cd = c::cwrs_decode(&ns, &ks, &junk);
    let rd = rust_cwrs_decode(&ns, &ks, &junk);
    assert_eq!(rd.0, cd.0, "{what}: junk decoded pulses");
    assert_bits_eq_f32(&format!("{what}: junk yy"), &rd.1, &cd.1);
    assert_eq!((rd.2, rd.3), (cd.2, cd.3), "{what}: junk decoder state");
}

/// Every `(n, k)` the table supports (which includes every size/pulse count CELT can use),
/// several random vectors each, plus random batches.
#[test]
fn cwrs_matches_oracle() {
    let mut rng = Rng::new(0xC3A5);
    let mut all = Vec::new();
    for n in 2..=208 {
        for k in 1..=208 {
            if pvq_ok(n, k) {
                all.push((n, k));
            }
        }
    }
    assert!(all.len() > 1500, "{}", all.len());
    for rep in 0..4 {
        for (ci, chunk) in all.chunks(64).enumerate() {
            check_cwrs_batch(&mut rng, chunk, &format!("all pairs rep {rep} chunk {ci}"));
        }
    }
    for iter in 0..2000 {
        let cnt = rng.range_i32(1, 30) as usize;
        let pairs: Vec<(i32, i32)> = (0..cnt)
            .map(|_| all[rng.range_i32(0, all.len() as i32 - 1) as usize])
            .collect();
        check_cwrs_batch(&mut rng, &pairs, &format!("random iter {iter}"));
    }
}

#[cfg(feature = "custom-modes")]
#[test]
fn log2_frac_and_required_bits_match_oracle() {
    let mut rng = Rng::new(0x10C2);
    let mut vals: Vec<u32> = (0..=70000).collect();
    for s in 0..32 {
        let p = 1u32 << s;
        vals.extend([p.wrapping_sub(1), p, p.wrapping_add(1)]);
    }
    vals.extend([
        u32::MAX,
        u32::MAX - 1,
        0xFFFF_0000,
        0xFFFF_8000,
        0x8000_0001,
    ]);
    for _ in 0..100_000 {
        vals.push(rng.next_u32() >> rng.range_i32(0, 31));
    }
    for &v in &vals {
        for frac in 0..=6 {
            assert_eq!(
                cwrs::log2_frac(v, frac),
                c::log2_frac(v, frac),
                "v={v} frac={frac}"
            );
        }
    }
    for n in 1..=208 {
        let maxk = (1..=128).filter(|&k| pvq_ok(n, k)).max();
        let Some(maxk) = maxk else { continue };
        for frac in [0, 3, 5] {
            let mut r = vec![0i16; maxk as usize + 1];
            cwrs::get_required_bits(&mut r, n, maxk, frac);
            assert_slice_eq(
                &format!("get_required_bits n={n} maxk={maxk} frac={frac}"),
                &r,
                &c::get_required_bits(n, maxk, frac),
            );
        }
    }
    // fits_in32 against the table: V(n,k) fits iff it is below 2^32 (checked via the table
    // wherever the table covers it).
    for n in 2..=208 {
        for k in 1..=128 {
            if rate::fits_in32(n, k) {
                assert!(pvq_ok(n, k) || n > 14, "fits_in32({n},{k}) outside table");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------------------------

fn rust_alloc(m: &CeltMode, a: &c::AllocIn, buf: &mut [u8], encode: bool) -> c::AllocOut {
    let nb = m.nb_ebands as usize;
    let mut pulses = vec![0i32; nb];
    let mut ebits = vec![0i32; nb];
    let mut fine_priority = vec![0i32; nb];
    let mut intensity = a.intensity;
    let mut dual_stereo = a.dual_stereo;
    let mut balance = 0;
    let (coded, tell_frac, rng, error);
    if encode {
        let mut e = EcEnc::new(buf);
        coded = rate::clt_compute_allocation(
            m,
            a.start,
            a.end,
            &a.offsets,
            &a.cap,
            a.alloc_trim,
            &mut intensity,
            &mut dual_stereo,
            a.total,
            &mut balance,
            &mut pulses,
            &mut ebits,
            &mut fine_priority,
            a.c,
            a.lm,
            &mut EcCoder::Enc(&mut e),
            a.prev,
            a.signal_bandwidth,
        );
        tell_frac = e.tell_frac();
        e.done();
        rng = e.rng;
        error = e.error;
    } else {
        let mut d = EcDec::new(buf);
        coded = rate::clt_compute_allocation(
            m,
            a.start,
            a.end,
            &a.offsets,
            &a.cap,
            a.alloc_trim,
            &mut intensity,
            &mut dual_stereo,
            a.total,
            &mut balance,
            &mut pulses,
            &mut ebits,
            &mut fine_priority,
            a.c,
            a.lm,
            &mut EcCoder::Dec(&mut d),
            a.prev,
            a.signal_bandwidth,
        );
        tell_frac = d.tell_frac();
        rng = d.rng;
        error = d.error;
    }
    c::AllocOut {
        intensity,
        dual_stereo,
        balance,
        coded_bands: coded,
        tell_frac,
        rng,
        error,
        pulses,
        ebits,
        fine_priority,
    }
}

fn random_alloc_in(rng: &mut Rng, m: &CeltMode, use_qext: bool, lm: i32, cc: i32) -> c::AllocIn {
    let nb = m.nb_ebands;
    let start = if !use_qext && nb == 21 && rng.range_i32(0, 4) == 0 {
        17
    } else {
        0
    };
    let end = match rng.range_i32(0, 3) {
        0 => nb,
        1 => [13, 15, 17, 19, 21][rng.range_i32(0, 4) as usize].min(nb),
        _ => rng.range_i32(start + 1, nb),
    }
    .max(start + 1);
    let mut offsets = vec![0i32; nb as usize];
    if rng.range_i32(0, 2) != 0 {
        for j in start..end {
            if rng.range_i32(0, 5) == 0 {
                offsets[j as usize] = (rng.range_i32(1, 6) * (8 * cc)) << lm;
            }
        }
    }
    let total = match rng.range_i32(0, 5) {
        0 => rng.range_i32(-100, 200),
        1 => rng.range_i32(0, 2000),
        2 => rng.range_i32(0, 20000),
        3 => rng.range_i32(0, 81600),
        _ => rng.range_i32(0, 400_000),
    };
    c::AllocIn {
        use_qext,
        start,
        end,
        offsets,
        cap: init_caps(m, lm, cc),
        alloc_trim: rng.range_i32(0, 10),
        intensity: rng.range_i32(start, end),
        dual_stereo: rng.range_i32(0, 1),
        total,
        c: cc,
        lm,
        prev: rng.range_i32(0, nb),
        signal_bandwidth: rng.range_i32(0, nb - 1),
    }
}

/// Returns `(allocations with skipped bands, stereo allocations using intensity)`.
fn check_alloc(
    rng: &mut Rng,
    m: &CeltMode,
    o: &c::Mode,
    use_qext: bool,
    iters: usize,
) -> (usize, usize) {
    let (mut skipped, mut intensity) = (0, 0);
    for lm in 0..=m.max_lm {
        for cc in 1..=2 {
            for iter in 0..iters {
                let a = random_alloc_in(rng, m, use_qext, lm, cc);
                let size = if iter % 5 == 0 {
                    rng.range_i32(0, 8) as usize
                } else {
                    1275
                };
                let what = format!("lm={lm} c={cc} iter={iter} qext={use_qext} {a:?}");
                let mut cbuf = vec![0u8; size];
                let co = o.clt_compute_allocation(&a, &mut cbuf, true);
                let mut rbuf = vec![0u8; size];
                let ro = rust_alloc(m, &a, &mut rbuf, true);
                assert_eq!(ro, co, "encode {what}");
                assert_eq!(rbuf, cbuf, "encode bytes {what}");
                // Decode the encoder's bytes: decisions must round-trip.
                let mut cb = cbuf.clone();
                let cd = o.clt_compute_allocation(&a, &mut cb, false);
                let mut rb = cbuf.clone();
                let rd = rust_alloc(m, &a, &mut rb, false);
                assert_eq!(rd, cd, "decode {what}");
                if co.error == 0 {
                    assert_eq!(
                        (
                            rd.coded_bands,
                            rd.intensity,
                            rd.dual_stereo,
                            &rd.pulses,
                            &rd.ebits
                        ),
                        (
                            co.coded_bands,
                            co.intensity,
                            co.dual_stereo,
                            &co.pulses,
                            &co.ebits
                        ),
                        "round trip {what}"
                    );
                }
                // Decode random bytes.
                let mut junk = vec![0u8; rng.range_i32(0, 16) as usize];
                rng.fill_bytes(&mut junk);
                let mut cj = junk.clone();
                let cd = o.clt_compute_allocation(&a, &mut cj, false);
                let rd = rust_alloc(m, &a, &mut junk, false);
                assert_eq!(rd, cd, "junk decode {what}");
                if co.coded_bands < a.end {
                    skipped += 1;
                }
                if cc == 2 && co.intensity > a.start && co.intensity < co.coded_bands {
                    intensity += 1;
                }
            }
        }
    }
    (skipped, intensity)
}

#[track_caller]
fn assert_alloc_coverage(iters: usize, (skipped, intensity): (usize, usize)) {
    assert!(
        skipped > iters,
        "only {skipped} allocations with skipped bands"
    );
    assert!(
        intensity > iters / 4,
        "only {intensity} allocations with intensity"
    );
}

#[test]
fn clt_compute_allocation_matches_oracle() {
    let mut rng = Rng::new(0xA110C);
    let o = c::Mode::create(48000, 960).unwrap();
    assert_alloc_coverage(
        1500,
        check_alloc(&mut rng, &MODE48000_960_120, &o, false, 1500),
    );
    #[cfg(feature = "qext")]
    {
        use opusorus::celt::static_modes::MODE96000_1920_240;
        let o96 = c::Mode::create(96000, 1920).unwrap();
        assert_alloc_coverage(
            300,
            check_alloc(&mut rng, &MODE96000_1920_240, &o96, false, 300),
        );
    }
}

/// Exhaustive sweep over the parameters that drive the reservations and band skipping.
#[test]
fn clt_compute_allocation_sweep() {
    let o = c::Mode::create(48000, 960).unwrap();
    let m = &MODE48000_960_120;
    let nb = m.nb_ebands;
    for lm in 0..=3 {
        for cc in 1..=2 {
            let cap = init_caps(m, lm, cc);
            for end in [13, 17, 19, 21] {
                for intensity in [0, 5, end] {
                    for dual_stereo in 0..=1 {
                        for total in (0..60000).step_by(1499).chain([1, 7, 8, 9, 23, 24, 25]) {
                            let a = c::AllocIn {
                                use_qext: false,
                                start: 0,
                                end,
                                offsets: vec![0; nb as usize],
                                cap: cap.clone(),
                                alloc_trim: 5,
                                intensity,
                                dual_stereo,
                                total,
                                c: cc,
                                lm,
                                prev: 17,
                                signal_bandwidth: 20,
                            };
                            let mut cbuf = vec![0u8; 1275];
                            let co = o.clt_compute_allocation(&a, &mut cbuf, true);
                            let mut rbuf = vec![0u8; 1275];
                            let ro = rust_alloc(m, &a, &mut rbuf, true);
                            assert_eq!(ro, co, "sweep encode {a:?}");
                            assert_eq!(rbuf, cbuf, "sweep bytes {a:?}");
                            let rd = rust_alloc(m, &a, &mut cbuf.clone(), false);
                            let cd = o.clt_compute_allocation(&a, &mut cbuf, false);
                            assert_eq!(rd, cd, "sweep decode {a:?}");
                        }
                    }
                }
            }
        }
    }
}

#[cfg(feature = "qext")]
fn rust_extra_alloc(
    m: &CeltMode,
    a: &c::ExtraAllocIn,
    buf: &mut [u8],
    encode: bool,
) -> c::ExtraAllocOut {
    let q = modes::compute_qext_mode(m);
    let qm = a.with_qext.then_some(&q);
    let mut extra_pulses = vec![0i32; a.n_out];
    let mut extra_equant = vec![0i32; a.n_out];
    let (tell_frac, rng, error);
    if encode {
        let mut e = EcEnc::new(buf);
        rate::clt_compute_extra_allocation(
            m,
            qm,
            a.start,
            a.end,
            a.qext_end,
            &a.band_log_e,
            &a.qext_band_log_e,
            a.total,
            &mut extra_pulses,
            &mut extra_equant,
            a.c,
            a.lm,
            &mut EcCoder::Enc(&mut e),
            a.tone_freq,
            a.toneishness,
        );
        tell_frac = e.tell_frac();
        e.done();
        rng = e.rng;
        error = e.error;
    } else {
        let mut d = EcDec::new(buf);
        rate::clt_compute_extra_allocation(
            m,
            qm,
            a.start,
            a.end,
            a.qext_end,
            &[],
            &[],
            a.total,
            &mut extra_pulses,
            &mut extra_equant,
            a.c,
            a.lm,
            &mut EcCoder::Dec(&mut d),
            a.tone_freq,
            a.toneishness,
        );
        tell_frac = d.tell_frac();
        rng = d.rng;
        error = d.error;
    }
    c::ExtraAllocOut {
        extra_pulses,
        extra_equant,
        tell_frac,
        rng,
        error,
    }
}

#[cfg(feature = "qext")]
#[test]
fn clt_compute_extra_allocation_matches_oracle() {
    use opusorus::celt::static_modes::MODE96000_1920_240;
    let mut rng = Rng::new(0xE77A);
    let modes_list = [
        (&MODE48000_960_120, c::Mode::create(48000, 960).unwrap()),
        (&MODE96000_1920_240, c::Mode::create(96000, 1920).unwrap()),
    ];
    let nbq = modes::NB_QEXT_BANDS as usize;
    for (m, o) in &modes_list {
        let nb = m.nb_ebands;
        let mut nonzero = 0;
        for iter in 0..6000 {
            let with_qext = iter % 3 != 0;
            let lm = rng.range_i32(0, 3);
            let cc = rng.range_i32(1, 2);
            // With a QEXT mode, end must be nbEBands; without one, keep at least 5 bands so the
            // median follower is fully initialised (C reads uninitialised stack otherwise).
            let start = if with_qext || rng.range_i32(0, 1) == 0 {
                0
            } else {
                rng.range_i32(0, 12)
            };
            let end = if with_qext {
                nb
            } else {
                rng.range_i32(start + 5, nb)
            };
            let qext_end = if with_qext {
                if rng.range_i32(0, 1) == 0 {
                    2
                } else {
                    modes::NB_QEXT_BANDS
                }
            } else {
                0
            };
            let mut band_log_e = vec![0f32; (cc * nb) as usize];
            let mut qext_band_log_e = vec![0f32; 2 * nbq];
            let scale = [1.0f32, 5.0, 20.0][rng.range_i32(0, 2) as usize];
            for v in band_log_e.iter_mut().chain(qext_band_log_e.iter_mut()) {
                *v = scale * rng.f32_sym() + rng.range_i32(-5, 10) as f32;
            }
            let total = match rng.range_i32(0, 4) {
                0 => rng.range_i32(-100, 100),
                1 => rng.range_i32(0, 5000),
                2 => rng.range_i32(0, 60000),
                _ => rng.range_i32(0, 3825 * 64),
            };
            let toneishness = match rng.range_i32(0, 3) {
                0 => 0.0,
                1 => 0.99,
                _ => 0.5 * (rng.f32_sym() + 1.0),
            };
            let tone_freq = match rng.range_i32(0, 2) {
                0 => -1.0,
                _ => 1.6 * (rng.f32_sym() + 1.0),
            };
            let a = c::ExtraAllocIn {
                with_qext,
                start,
                end,
                qext_end,
                band_log_e,
                qext_band_log_e,
                total,
                c: cc,
                lm,
                tone_freq,
                toneishness,
                n_out: nb as usize + nbq,
            };
            let size = match rng.range_i32(0, 3) {
                0 => rng.range_i32(0, 20) as usize,
                1 => rng.range_i32(0, 200) as usize,
                _ => rng.range_i32(0, 3825) as usize,
            };
            let what = format!("fs={} iter={iter} {a:?} size={size}", m.fs);
            let mut cbuf = vec![0u8; size];
            let co = o.clt_compute_extra_allocation(&a, &mut cbuf, true);
            let mut rbuf = vec![0u8; size];
            let ro = rust_extra_alloc(m, &a, &mut rbuf, true);
            assert_eq!(ro, co, "extra encode {what}");
            assert_eq!(rbuf, cbuf, "extra bytes {what}");
            let rd = rust_extra_alloc(m, &a, &mut cbuf.clone(), false);
            let cd = o.clt_compute_extra_allocation(&a, &mut cbuf, false);
            assert_eq!(rd, cd, "extra decode {what}");
            if co.error == 0 && a.total > 0 {
                assert_eq!(rd.extra_pulses, co.extra_pulses, "extra round trip {what}");
            }
            if co.extra_pulses.iter().any(|&p| p > 0) {
                nonzero += 1;
            }
            let mut junk = vec![0u8; rng.range_i32(0, 64) as usize];
            rng.fill_bytes(&mut junk);
            let rd = rust_extra_alloc(m, &a, &mut junk.clone(), false);
            let cd = o.clt_compute_extra_allocation(&a, &mut junk, false);
            assert_eq!(rd, cd, "extra junk decode {what}");
        }
        assert!(nonzero > 1500, "only {nonzero} cases with extra pulses");
    }
}
