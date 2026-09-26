//! Differential tests for the foundation unit: range coder (entenc/entdec/entcode) and float
//! mathops vs the C oracle.

use opusorus::celt::entcode::EcState;
use opusorus::celt::entdec::EcDec;
use opusorus::celt::entenc::EcEnc;
use opusorus::celt::mathops;
use opusorus_conformance::{Rng, assert_slice_eq};
use opusorus_oracle::foundation as c;

/// Generates a random but valid op sequence for the encoder, plus the matching decoder ops.
fn random_ops(rng: &mut Rng, n: usize) -> (Vec<[u32; 4]>, Vec<[u32; 2]>) {
    let mut enc = Vec::with_capacity(n);
    let mut dec = Vec::with_capacity(n);
    for _ in 0..n {
        match rng.range_i32(0, 5) {
            0 => {
                let ft = rng.range_i32(2, 60000) as u32;
                let fl = rng.next_u32() % ft;
                enc.push([0, fl, fl + 1, ft]);
                dec.push([0, ft]);
            }
            1 => {
                let bits = rng.range_i32(1, 15) as u32;
                let fl = rng.next_u32() % (1 << bits);
                enc.push([1, fl, fl + 1, bits]);
                dec.push([1, bits]);
            }
            2 => {
                let logp = rng.range_i32(1, 15) as u32;
                let v = u32::from(rng.range_i32(0, 3) == 0);
                enc.push([2, v, logp, 0]);
                dec.push([2, logp]);
            }
            3 => {
                let s = rng.range_i32(0, 7) as u32;
                enc.push([3, s, 8, 0]);
                dec.push([3, 8]);
            }
            4 => {
                let ft = rng.range_i32(2, i32::MAX) as u32;
                let fl = rng.next_u32() % ft;
                enc.push([4, fl, ft, 0]);
                dec.push([4, ft]);
            }
            _ => {
                let bits = rng.range_i32(1, 25) as u32;
                let fl = rng.next_u32() & ((1 << bits) - 1);
                enc.push([5, fl, bits, 0]);
                dec.push([5, bits]);
            }
        }
    }
    (enc, dec)
}

fn rust_encode(ops: &[[u32; 4]], size: usize, shrink_to: usize) -> c::EncOut {
    let mut buf = vec![0u8; size];
    let mut tells = Vec::with_capacity(ops.len());
    let (rng, error) = {
        let mut e = EcEnc::new(&mut buf);
        for o in ops {
            match o[0] {
                0 => e.encode(o[1], o[2], o[3]),
                1 => e.encode_bin(o[1], o[2], o[3]),
                2 => e.enc_bit_logp(o[1] != 0, o[2]),
                3 => e.enc_icdf(o[1] as usize, &c::ORACLE_ICDF, o[2]),
                4 => e.enc_uint(o[1], o[2]),
                5 => e.enc_bits(o[1], o[2]),
                6 => e.patch_initial_bits(o[1], o[2]),
                _ => unreachable!(),
            }
            tells.push((e.tell(), e.tell_frac()));
        }
        if shrink_to > 0 {
            e.shrink(shrink_to as u32);
        }
        e.done();
        (e.rng, e.error)
    };
    c::EncOut {
        buf,
        tells,
        rng,
        error,
    }
}

fn rust_decode(ops: &[[u32; 2]], data: &[u8]) -> c::DecOut {
    let mut d = EcDec::new(data);
    let mut values = Vec::with_capacity(ops.len());
    let mut tells = Vec::with_capacity(ops.len());
    for o in ops {
        let v = match o[0] {
            0 => {
                let v = d.decode(o[1]);
                d.update(v, v + 1, o[1]);
                v
            }
            1 => {
                let v = d.decode_bin(o[1]);
                d.update(v, v + 1, 1 << o[1]);
                v
            }
            2 => u32::from(d.dec_bit_logp(o[1])),
            3 => d.dec_icdf(&c::ORACLE_ICDF, o[1]) as u32,
            4 => d.dec_uint(o[1]),
            5 => d.dec_bits(o[1]),
            _ => unreachable!(),
        };
        values.push(v);
        tells.push((d.tell(), EcState::tell_frac(&d)));
    }
    c::DecOut {
        values,
        tells,
        rng: d.rng,
        error: d.error,
    }
}

#[test]
fn range_coder_matches_oracle() {
    let mut rng = Rng::new(1);
    for iter in 0..3000 {
        let n = rng.range_i32(1, 300) as usize;
        let (eops, dops) = random_ops(&mut rng, n);
        // Mix of roomy and too-small buffers (the latter exercise the error paths).
        let size = if iter % 5 == 0 {
            rng.range_i32(1, 64) as usize
        } else {
            2048
        };
        let c_out = c::ec_encode_ops(&eops, size, 0);
        let r_out = rust_encode(&eops, size, 0);
        assert_eq!(r_out, c_out, "encoder mismatch at iter {iter}");
        if c_out.error == 0 {
            let c_dec = c::ec_decode_ops(&dops, &c_out.buf);
            let r_dec = rust_decode(&dops, &c_out.buf);
            assert_eq!(r_dec, c_dec, "decoder mismatch at iter {iter}");
            // Round trip: decoded values equal encoded symbols.
            for (k, (e, v)) in eops.iter().zip(&r_dec.values).enumerate() {
                let want = match e[0] {
                    2..=5 => e[1],
                    _ => e[1],
                };
                assert_eq!(*v, want, "roundtrip value {k} at iter {iter}");
            }
        }
    }
}

#[test]
fn range_decoder_on_garbage_matches_oracle() {
    let mut rng = Rng::new(7);
    for iter in 0..3000 {
        let n = rng.range_i32(1, 200) as usize;
        let (_, dops) = random_ops(&mut rng, n);
        let mut data = vec![0u8; rng.range_i32(0, 100) as usize];
        rng.fill_bytes(&mut data);
        let c_dec = c::ec_decode_ops(&dops, &data);
        let r_dec = rust_decode(&dops, &data);
        assert_eq!(r_dec, c_dec, "garbage decode mismatch at iter {iter}");
    }
}

#[test]
fn shrink_and_patch_match_oracle() {
    let mut rng = Rng::new(3);
    for iter in 0..1000 {
        let n = rng.range_i32(1, 40) as usize;
        let (mut eops, _) = random_ops(&mut rng, n);
        if iter % 2 == 0 {
            eops.push([6, rng.range_i32(0, 3) as u32, 2, 0]);
        }
        let c_out = c::ec_encode_ops(&eops, 1275, 400);
        let r_out = rust_encode(&eops, 1275, 400);
        assert_eq!(r_out, c_out, "shrink/patch mismatch at iter {iter}");
    }
}

#[test]
fn mathops_match_oracle() {
    let mut rng = Rng::new(11);
    for _ in 0..200_000 {
        let x = rng.f32_sym() * 40.0;
        let p = x.abs() + 1e-6;
        assert_eq!(
            mathops::celt_log2(p).to_bits(),
            c::celt_log2(p).to_bits(),
            "log2({p})"
        );
        assert_eq!(
            mathops::celt_exp2(x).to_bits(),
            c::celt_exp2(x).to_bits(),
            "exp2({x})"
        );
        let y = rng.f32_sym() * 4.0;
        assert_eq!(
            mathops::celt_cos_norm(y).to_bits(),
            c::celt_cos_norm(y).to_bits(),
            "cos_norm({y})"
        );
        assert_eq!(
            mathops::celt_cos_norm2(y).to_bits(),
            c::celt_cos_norm2(y).to_bits(),
            "cos_norm2({y})"
        );
        assert_eq!(
            mathops::celt_sin(y).to_bits(),
            c::celt_sin(y).to_bits(),
            "sin({y})"
        );
        assert_eq!(mathops::celt_sqrt(p).to_bits(), c::celt_sqrt(p).to_bits());
        assert_eq!(mathops::celt_rsqrt(p).to_bits(), c::celt_rsqrt(p).to_bits());
        let (a, b) = (rng.f32_sym(), rng.f32_sym());
        assert_eq!(
            mathops::fast_atan2f(a, b).to_bits(),
            c::fast_atan2f(a, b).to_bits()
        );
        assert_eq!(
            mathops::celt_atan2p_norm(a.abs(), b.abs()).to_bits(),
            c::celt_atan2p_norm(a.abs(), b.abs()).to_bits()
        );
        let big = rng.f32_sym() * 70000.0;
        assert_eq!(
            mathops::float2int(big),
            c::float2int(big),
            "float2int({big})"
        );
        let s = rng.f32_sym() * 1.2;
        assert_eq!(
            mathops::float2int16(s),
            c::float2int16(s),
            "float2int16({s})"
        );
        let u = rng.next_u32() | 1;
        assert_eq!(mathops::isqrt32(u), c::isqrt32(u));
    }
    let halves: Vec<f32> = (-20..20).map(|k| k as f32 + 0.5).collect();
    let r: Vec<i32> = halves.iter().map(|&h| mathops::float2int(h)).collect();
    let cc: Vec<i32> = halves.iter().map(|&h| c::float2int(h)).collect();
    assert_slice_eq("float2int ties", &r, &cc);
}
