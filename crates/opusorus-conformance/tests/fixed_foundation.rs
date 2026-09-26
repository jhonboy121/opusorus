//! Differential tests for unit `fixed_foundation` (features `fixed-point` / `fixed-res24`, with
//! or without `qext`) vs the fixed-point C oracle:
//! * every fixed-point macro of `celt/arch.h` + `celt/fixed_generic.h` (both the 64-bit and the
//!   32-bit `OPUS_FAST_INT64` forms of the multiplies), on exhaustive 16-bit sweeps, edge values
//!   and random inputs within each macro's defined domain (C undefined behaviour excluded),
//! * the `opus_res` / `celt_sig` / `celt_coef` conversions and the build constants,
//! * every fixed-point mathops function (`celt/mathops.{h,c}`) and the float API conversions of
//!   `celt/float_cast.h`,
//! * the fixed-point static modes (`celt/static_modes_fixed.h`) field by field,
//! * the range coder, CWRS and Laplace coding (shared integer code) in a fixed-point build.
//!
//! Run with `cargo test -p opusorus-conformance --features fixed-point --test fixed_foundation`
//! (also `fixed-res24`, and either with `qext`). Every comparison is bit-exact.

#![cfg(feature = "fixed-point")]
#![expect(
    clippy::expect_used,
    reason = "test helpers: a failed conversion or a missing oracle mode is a test failure"
)]

use opusorus::celt::arch::{self as a, CeltCoef, CeltSig, OpusRes, OpusVal16, OpusVal32};
use opusorus::celt::entdec::EcDec;
use opusorus::celt::entenc::EcEnc;
use opusorus::celt::mathops as m;
use opusorus::celt::static_modes::{self as sm, CeltMode, KissFftState};
use opusorus::celt::{cwrs, laplace};
use opusorus_conformance::Rng;
use opusorus_oracle::fixed_foundation::{self as c, id1, id2, id3, mid1, mid2, mode_arr};

// ---------------------------------------------------------------------------------------------
// Input generators
// ---------------------------------------------------------------------------------------------

/// Widens any integer result for comparison (the C shims return `int`).
fn w<T: Into<i64>>(x: T) -> i64 {
    x.into()
}

const EDGES16: [i32; 17] = [
    -32768, -32767, -32766, -16385, -16384, -16383, -256, -2, -1, 0, 1, 2, 255, 16383, 16384,
    32766, 32767,
];
const EDGES32: [i32; 24] = [
    i32::MIN,
    i32::MIN + 1,
    -(1 << 30) - 1,
    -(1 << 30),
    -(1 << 29),
    -65537,
    -65536,
    -65535,
    -32769,
    -32768,
    -32767,
    -1,
    0,
    1,
    32767,
    32768,
    32769,
    65535,
    65536,
    1 << 29,
    (1 << 30) - 1,
    1 << 30,
    i32::MAX - 1,
    i32::MAX,
];

/// A random signed value with magnitude below `2^bits` (`bits` in 0..=31), biased towards
/// "interesting" magnitudes by also randomizing the bit width.
fn rbits(rng: &mut Rng, bits: u32) -> i32 {
    let b = rng.range_i32(0, bits as i32) as u32;
    let v = if b == 0 {
        0
    } else {
        (rng.next_u64() % (1u64 << b)) as i64
    };
    let v = if rng.next_u32() & 1 == 1 { -v } else { v };
    v.clamp(i32::MIN as i64 + 1, i32::MAX as i64) as i32
}

/// Any `i32`: full-range random, 16-bit random, 32-bit edges.
fn rany(rng: &mut Rng) -> i32 {
    match rng.range_i32(0, 5) {
        0 => rng.next_u32() as i32,
        1 => i32::from(rng.i16()),
        2 => EDGES32[rng.range_i32(0, EDGES32.len() as i32 - 1) as usize],
        3 => EDGES16[rng.range_i32(0, EDGES16.len() as i32 - 1) as usize],
        _ => rbits(rng, 31),
    }
}

/// Any 16-bit value (as the `int` a caller passes).
fn r16(rng: &mut Rng) -> i32 {
    if rng.range_i32(0, 7) == 0 {
        EDGES16[rng.range_i32(0, EDGES16.len() as i32 - 1) as usize]
    } else {
        i32::from(rng.i16())
    }
}

/// A random `opus_res` value (16-bit PCM, or 24-bit-ish with `fixed-res24`).
fn rres(rng: &mut Rng) -> i32 {
    if cfg!(feature = "fixed-res24") {
        rbits(rng, 26)
    } else {
        r16(rng)
    }
}

/// A random `celt_coef` value.
fn rcoef(rng: &mut Rng) -> i32 {
    if cfg!(feature = "qext") {
        rany(rng)
    } else {
        r16(rng)
    }
}

fn res(x: i32) -> OpusRes {
    OpusRes::try_from(x).expect("value in opus_res range")
}

fn coef(x: i32) -> CeltCoef {
    CeltCoef::try_from(x).expect("value in celt_coef range")
}

/// Number of random cases per macro.
const N: usize = 200_000;

// ---------------------------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------------------------

type F1 = fn(i32) -> i64;
type F2 = fn(i32, i32) -> i64;
type F3 = fn(i32, i32, i32) -> i64;

fn check1(name: &str, op: i32, f: F1, mut gen_a: impl FnMut(&mut Rng) -> i32, seed: u64) {
    let mut rng = Rng::new(seed);
    for i in 0..N {
        let x = gen_a(&mut rng);
        assert_eq!(f(x), i64::from(c::op1(op, x)), "{name}({x}) case {i}");
    }
}

fn check2(
    name: &str,
    f: F2,
    c_f: impl Fn(i32, i32) -> i32,
    mut gen_ab: impl FnMut(&mut Rng) -> (i32, i32),
    seed: u64,
) {
    let mut rng = Rng::new(seed);
    for i in 0..N {
        let (x, y) = gen_ab(&mut rng);
        assert_eq!(f(x, y), i64::from(c_f(x, y)), "{name}({x}, {y}) case {i}");
    }
}

fn check3(
    name: &str,
    op: i32,
    f: F3,
    mut gen_ab: impl FnMut(&mut Rng) -> (i32, i32, i32),
    seed: u64,
) {
    let mut rng = Rng::new(seed);
    for i in 0..N {
        let (z, x, y) = gen_ab(&mut rng);
        assert_eq!(
            f(z, x, y),
            i64::from(c::op3(op, z, x, y)),
            "{name}({z}, {x}, {y}) case {i}"
        );
    }
}

/// Exhaustive sweep of the first argument over every 16-bit value (second random).
fn sweep16(name: &str, op: i32, f: F2, mut gen_b: impl FnMut(&mut Rng) -> i32) {
    let mut rng = Rng::new(99);
    for x in -32768..=32767 {
        let y = gen_b(&mut rng);
        assert_eq!(
            f(x, y),
            i64::from(c::op2(op, x, y)),
            "{name}({x}, {y}) sweep"
        );
    }
}

/// All pairs of 16-bit edge values.
fn edges_pairs(name: &str, op: i32, f: F2) {
    for &x in &EDGES16 {
        for &y in &EDGES16 {
            assert_eq!(
                f(x, y),
                i64::from(c::op2(op, x, y)),
                "{name}({x}, {y}) edges"
            );
        }
    }
}

#[test]
fn one_argument_macros_match_oracle() {
    let not_min = |r: &mut Rng| {
        let x = rany(r);
        if x == i32::MIN { 0 } else { x }
    };
    let cases: [(&str, i32, F1); 12] = [
        ("NEG16", id1::NEG16, |x| w(a::neg16(x))),
        ("NEG32", id1::NEG32, |x| w(a::neg32(x))),
        ("NEG32_ovflw", id1::NEG32_OVFLW, |x| w(a::neg32_ovflw(x))),
        ("EXTRACT16", id1::EXTRACT16, |x| w(a::extract16(x))),
        ("EXTEND32", id1::EXTEND32, |x| w(a::extend32(x))),
        ("SATURATE16", id1::SATURATE16, |x| w(a::saturate16(x))),
        ("HALF16", id1::HALF16, |x| w(a::half16(x))),
        ("HALF32", id1::HALF32, |x| w(a::half32(x))),
        ("ABS16", id1::ABS16, |x| w(a::abs16(x))),
        ("ABS32", id1::ABS32, |x| w(a::abs32(x))),
        ("SAT16", id1::SAT16, |x| w(a::sat16(x))),
        ("celt_isnan", id1::CELT_ISNAN, |x| {
            w(i32::from(a::celt_isnan(x)))
        }),
    ];
    for (k, (name, op, f)) in cases.into_iter().enumerate() {
        check1(name, op, f, not_min, 100 + k as u64);
    }
    // SIG2WORD16: PSHR32(x, SIG_SHIFT) must not overflow.
    check1(
        "SIG2WORD16",
        id1::SIG2WORD16,
        |x| w(a::sig2word16(x)),
        |r| rany(r).min(i32::MAX - 2048),
        120,
    );
    check1(
        "COEF2VAL16",
        id1::COEF2VAL16,
        |x| w(a::coef2val16(coef(x))),
        rcoef,
        121,
    );
}

#[test]
fn res_and_sig_conversions_match_oracle() {
    let sig = |r: &mut Rng| rany(r).min(i32::MAX - 2048);
    check1("SIG2RES", id1::SIG2RES, |x| w(a::sig2res(x)), sig, 200);
    check1(
        "RES2INT16",
        id1::RES2INT16,
        |x| w(a::res2int16(res(x))),
        rres,
        201,
    );
    check1(
        "RES2INT24",
        id1::RES2INT24,
        |x| w(a::res2int24(res(x))),
        rres,
        202,
    );
    check1(
        "INT16TORES",
        id1::INT16TORES,
        |x| w(a::int16tores(x as i16)),
        r16,
        203,
    );
    check1(
        "INT24TORES",
        id1::INT24TORES,
        |x| w(a::int24tores(x)),
        |r| rany(r).min(i32::MAX - 128),
        204,
    );
    check1(
        "RES2SIG",
        id1::RES2SIG,
        |x| w(a::res2sig(res(x))),
        rres,
        205,
    );
    check1(
        "RES2VAL16",
        id1::RES2VAL16,
        |x| w(a::res2val16(res(x))),
        rres,
        206,
    );
    check1(
        "INT16TOSIG",
        id1::INT16TOSIG,
        |x| w(a::int16tosig(x as i16)),
        r16,
        207,
    );
    check1(
        "INT24TOSIG",
        id1::INT24TOSIG,
        |x| w(a::int24tosig(x)),
        rany,
        208,
    );
    check2(
        "ADD_RES",
        |x, y| w(a::add_res(res(x), res(y))),
        |x, y| c::op2(id2::ADD_RES, x, y),
        |r| (rres(r), rres(r)),
        209,
    );
    check2(
        "MULT16_RES_Q15",
        |x, y| w(a::mult16_res_q15(x, res(y))),
        |x, y| c::op2(id2::MULT16_RES_Q15, x, y),
        |r| (rany(r), rres(r)),
        210,
    );
    let mut rng = Rng::new(211);
    for i in 0..N {
        let x = rres(&mut rng);
        assert_eq!(
            a::res2float(res(x)).to_bits(),
            c::res2float(x).to_bits(),
            "RES2FLOAT({x}) case {i}"
        );
        let f = rng.f32_sym() * 1.5;
        assert_eq!(
            w(a::float2res(f)),
            i64::from(c::float2res(f)),
            "FLOAT2RES({f})"
        );
        let g = rng.f32_sym() * 3.0;
        assert_eq!(
            w(a::float2sig(g)),
            i64::from(c::float2sig(g)),
            "FLOAT2SIG({g})"
        );
    }
}

/// Domains of the two-argument macros (C undefined behaviour excluded).
#[derive(Clone, Copy)]
enum D2 {
    /// Both arguments arbitrary (the macro truncates or cannot overflow).
    Any,
    /// Sum/difference must fit: both below 2^30.
    Add,
    /// Product must fit: bit widths summing to at most 30.
    Mul32,
    /// `a` arbitrary, `shift` in 0..=31.
    Shr,
    /// `a` arbitrary, `shift` in 0..=15 (`SHL16`).
    Shl16,
    /// `a + 2^(shift-1)` must not overflow, `shift` in 0..=30.
    Pshr,
    /// `a` arbitrary, `shift` in -31..=31.
    Vshr,
    /// `a` arbitrary, `shift` in 0..=30 (`1<<31` is undefined).
    Shr30,
    /// `x` arbitrary, `a` in 0..=i32::MAX.
    Saturate,
    /// `b` truncated to a non-zero 16-bit value; no `MIN / -1`.
    Div16,
    /// `b != 0`; no `MIN / -1`.
    Div32,
}

fn gen2(d: D2, r: &mut Rng) -> (i32, i32) {
    match d {
        D2::Any => (rany(r), rany(r)),
        D2::Add => (rbits(r, 30), rbits(r, 30)),
        D2::Mul32 => {
            let wa = r.range_i32(0, 30) as u32;
            (rbits(r, wa), rbits(r, 30 - wa))
        }
        D2::Shr => (rany(r), r.range_i32(0, 31)),
        D2::Shl16 => (rany(r), r.range_i32(0, 15)),
        D2::Pshr => (rany(r).min(i32::MAX - (1 << 29)), r.range_i32(0, 30)),
        D2::Vshr => (rany(r), r.range_i32(-31, 31)),
        D2::Shr30 => (rany(r), r.range_i32(0, 30)),
        D2::Saturate => (rany(r), rany(r).max(0)),
        D2::Div16 => {
            let mut b = rany(r);
            if b as i16 == 0 {
                b = 7;
            }
            let x = rany(r);
            (
                if x == i32::MIN && b as i16 == -1 {
                    0
                } else {
                    x
                },
                b,
            )
        }
        D2::Div32 => {
            let mut b = rany(r);
            if b == 0 {
                b = -3;
            }
            let x = rany(r);
            (if x == i32::MIN && b == -1 { 0 } else { x }, b)
        }
    }
}

#[test]
fn two_argument_macros_match_oracle() {
    let cases: Vec<(&str, i32, F2, D2)> = vec![
        (
            "MULT16_16SU",
            id2::MULT16_16SU,
            |x, y| w(a::mult16_16su(x, y)),
            D2::Any,
        ),
        (
            "MULT16_32_Q16",
            id2::MULT16_32_Q16,
            |x, y| w(a::mult16_32_q16(x, y)),
            D2::Any,
        ),
        (
            "MULT16_32_P16",
            id2::MULT16_32_P16,
            |x, y| w(a::mult16_32_p16(x, y)),
            D2::Any,
        ),
        (
            "MULT16_32_Q15",
            id2::MULT16_32_Q15,
            |x, y| w(a::mult16_32_q15(x, y)),
            D2::Any,
        ),
        (
            "MULT32_32_Q16",
            id2::MULT32_32_Q16,
            |x, y| w(a::mult32_32_q16(x, y)),
            D2::Any,
        ),
        (
            "MULT32_32_Q31",
            id2::MULT32_32_Q31,
            |x, y| w(a::mult32_32_q31(x, y)),
            D2::Any,
        ),
        (
            "MULT32_32_P31",
            id2::MULT32_32_P31,
            |x, y| w(a::mult32_32_p31(x, y)),
            D2::Any,
        ),
        (
            "MULT32_32_P31_ovflw",
            id2::MULT32_32_P31_OVFLW,
            |x, y| w(a::mult32_32_p31_ovflw(x, y)),
            D2::Any,
        ),
        (
            "MULT32_32_Q32",
            id2::MULT32_32_Q32,
            |x, y| w(a::mult32_32_q32(x, y)),
            D2::Any,
        ),
        ("SHR16", id2::SHR16, |x, s| w(a::shr16(x, s)), D2::Shr),
        ("SHL16", id2::SHL16, |x, s| w(a::shl16(x, s)), D2::Shl16),
        ("SHR32", id2::SHR32, |x, s| w(a::shr32(x, s)), D2::Shr),
        ("SHL32", id2::SHL32, |x, s| w(a::shl32(x, s)), D2::Shr),
        ("PSHR32", id2::PSHR32, |x, s| w(a::pshr32(x, s)), D2::Pshr),
        ("VSHR32", id2::VSHR32, |x, s| w(a::vshr32(x, s)), D2::Vshr),
        ("SHR", id2::SHR, |x, s| w(a::shr(x, s)), D2::Shr),
        ("SHL", id2::SHL, |x, s| w(a::shl(x, s)), D2::Shr),
        ("PSHR", id2::PSHR, |x, s| w(a::pshr(x, s)), D2::Pshr),
        (
            "SATURATE",
            id2::SATURATE,
            |x, y| w(a::saturate(x, y)),
            D2::Saturate,
        ),
        (
            "ROUND16",
            id2::ROUND16,
            |x, s| w(a::round16(x, s)),
            D2::Pshr,
        ),
        (
            "SROUND16",
            id2::SROUND16,
            |x, s| w(a::sround16(x, s)),
            D2::Pshr,
        ),
        ("ADD16", id2::ADD16, |x, y| w(a::add16(x, y)), D2::Any),
        ("SUB16", id2::SUB16, |x, y| w(a::sub16(x, y)), D2::Any),
        ("ADD32", id2::ADD32, |x, y| w(a::add32(x, y)), D2::Add),
        ("SUB32", id2::SUB32, |x, y| w(a::sub32(x, y)), D2::Add),
        (
            "ADD32_ovflw",
            id2::ADD32_OVFLW,
            |x, y| w(a::add32_ovflw(x, y)),
            D2::Any,
        ),
        (
            "SUB32_ovflw",
            id2::SUB32_OVFLW,
            |x, y| w(a::sub32_ovflw(x, y)),
            D2::Any,
        ),
        (
            "SHL32_ovflw",
            id2::SHL32_OVFLW,
            |x, s| w(a::shl32_ovflw(x, s)),
            D2::Shr,
        ),
        (
            "PSHR32_ovflw",
            id2::PSHR32_OVFLW,
            |x, s| w(a::pshr32_ovflw(x, s)),
            D2::Shr30,
        ),
        (
            "MULT16_16_16",
            id2::MULT16_16_16,
            |x, y| w(a::mult16_16_16(x, y)),
            D2::Any,
        ),
        (
            "MULT32_32_32",
            id2::MULT32_32_32,
            |x, y| w(a::mult32_32_32(x, y)),
            D2::Mul32,
        ),
        (
            "MULT16_16",
            id2::MULT16_16,
            |x, y| w(a::mult16_16(x, y)),
            D2::Any,
        ),
        (
            "MULT16_16_Q11_32",
            id2::MULT16_16_Q11_32,
            |x, y| w(a::mult16_16_q11_32(x, y)),
            D2::Any,
        ),
        (
            "MULT16_16_Q11",
            id2::MULT16_16_Q11,
            |x, y| w(a::mult16_16_q11(x, y)),
            D2::Any,
        ),
        (
            "MULT16_16_Q13",
            id2::MULT16_16_Q13,
            |x, y| w(a::mult16_16_q13(x, y)),
            D2::Any,
        ),
        (
            "MULT16_16_Q14",
            id2::MULT16_16_Q14,
            |x, y| w(a::mult16_16_q14(x, y)),
            D2::Any,
        ),
        (
            "MULT16_16_Q15",
            id2::MULT16_16_Q15,
            |x, y| w(a::mult16_16_q15(x, y)),
            D2::Any,
        ),
        (
            "MULT16_16_P13",
            id2::MULT16_16_P13,
            |x, y| w(a::mult16_16_p13(x, y)),
            D2::Any,
        ),
        (
            "MULT16_16_P14",
            id2::MULT16_16_P14,
            |x, y| w(a::mult16_16_p14(x, y)),
            D2::Any,
        ),
        (
            "MULT16_16_P15",
            id2::MULT16_16_P15,
            |x, y| w(a::mult16_16_p15(x, y)),
            D2::Any,
        ),
        (
            "DIV32_16",
            id2::DIV32_16,
            |x, y| w(a::div32_16(x, y)),
            D2::Div16,
        ),
        ("DIV32", id2::DIV32, |x, y| w(a::div32(x, y)), D2::Div32),
        ("MIN16", id2::MIN16, |x, y| w(a::min16(x, y)), D2::Any),
        ("MAX16", id2::MAX16, |x, y| w(a::max16(x, y)), D2::Any),
        ("MIN32", id2::MIN32, |x, y| w(a::min32(x, y)), D2::Any),
        ("MAX32", id2::MAX32, |x, y| w(a::max32(x, y)), D2::Any),
        ("IMIN", id2::IMIN, |x, y| w(a::imin(x, y)), D2::Any),
        ("IMAX", id2::IMAX, |x, y| w(a::imax(x, y)), D2::Any),
        (
            "MULT_COEF_32",
            id2::MULT_COEF_32,
            |x, y| w(a::mult_coef_32(x, y)),
            D2::Any,
        ),
        (
            "MULT_COEF",
            id2::MULT_COEF,
            |x, y| w(a::mult_coef(x, y)),
            D2::Any,
        ),
        (
            "MULT_COEF_TAPS",
            id2::MULT_COEF_TAPS,
            |x, y| w(a::mult_coef_taps(x, y)),
            D2::Any,
        ),
        ("IMUL32", id2::IMUL32, |x, y| w(a::imul32(x, y)), D2::Mul32),
        ("MING", id2::MING, |x, y| w(a::ming(x, y)), D2::Any),
        ("MAXG", id2::MAXG, |x, y| w(a::maxg(x, y)), D2::Any),
    ];
    for (k, &(name, op, f, d)) in cases.iter().enumerate() {
        check2(
            name,
            f,
            |x, y| c::op2(op, x, y),
            |r| gen2(d, r),
            300 + k as u64,
        );
        if matches!(d, D2::Any) {
            sweep16(name, op, f, rany);
            edges_pairs(name, op, f);
        }
    }
}

#[test]
fn three_argument_macros_match_oracle() {
    check3(
        "MAC16_16",
        id3::MAC16_16,
        |z, x, y| w(a::mac16_16(z, x, y)),
        |r| (rbits(r, 30), rany(r), rany(r)),
        400,
    );
    check3(
        "MAC16_32_Q15",
        id3::MAC16_32_Q15,
        |z, x, y| w(a::mac16_32_q15(z, x, y)),
        |r| (rbits(r, 29), rany(r), rbits(r, 30)),
        401,
    );
    check3(
        "MAC16_32_Q16",
        id3::MAC16_32_Q16,
        |z, x, y| w(a::mac16_32_q16(z, x, y)),
        |r| (rbits(r, 29), rany(r), rany(r)),
        402,
    );
    check3(
        "MAC_COEF_32_ARM",
        id3::MAC_COEF_32_ARM,
        |z, x, y| w(a::mac_coef_32_arm(z, x, y)),
        |r| (rbits(r, 29), rany(r), rany(r)),
        403,
    );
}

/// The 64-bit and 32-bit forms of the 32-bit multiplies, each against the C expansion of that
/// form (the 32-bit one from a copy of `fixed_generic.h` compiled with `OPUS_FAST_INT64 == 0`).
#[test]
fn multiply_forms_match_oracle() {
    let int64: [(&str, i32, F2); 8] = [
        ("MULT16_32_Q16", id2::MULT16_32_Q16, |x, y| {
            w(a::int64::mult16_32_q16(x, y))
        }),
        ("MULT16_32_P16", id2::MULT16_32_P16, |x, y| {
            w(a::int64::mult16_32_p16(x, y))
        }),
        ("MULT16_32_Q15", id2::MULT16_32_Q15, |x, y| {
            w(a::int64::mult16_32_q15(x, y))
        }),
        ("MULT32_32_Q16", id2::MULT32_32_Q16, |x, y| {
            w(a::int64::mult32_32_q16(x, y))
        }),
        ("MULT32_32_Q31", id2::MULT32_32_Q31, |x, y| {
            w(a::int64::mult32_32_q31(x, y))
        }),
        ("MULT32_32_P31", id2::MULT32_32_P31, |x, y| {
            w(a::int64::mult32_32_p31(x, y))
        }),
        ("MULT32_32_P31_ovflw", id2::MULT32_32_P31_OVFLW, |x, y| {
            w(a::int64::mult32_32_p31_ovflw(x, y))
        }),
        ("MULT32_32_Q32", id2::MULT32_32_Q32, |x, y| {
            w(a::int64::mult32_32_q32(x, y))
        }),
    ];
    let int32: [(&str, i32, F2); 8] = [
        ("MULT16_32_Q16", id2::MULT16_32_Q16, |x, y| {
            w(a::int32::mult16_32_q16(x, y))
        }),
        ("MULT16_32_P16", id2::MULT16_32_P16, |x, y| {
            w(a::int32::mult16_32_p16(x, y))
        }),
        ("MULT16_32_Q15", id2::MULT16_32_Q15, |x, y| {
            w(a::int32::mult16_32_q15(x, y))
        }),
        ("MULT32_32_Q16", id2::MULT32_32_Q16, |x, y| {
            w(a::int32::mult32_32_q16(x, y))
        }),
        ("MULT32_32_Q31", id2::MULT32_32_Q31, |x, y| {
            w(a::int32::mult32_32_q31(x, y))
        }),
        ("MULT32_32_P31", id2::MULT32_32_P31, |x, y| {
            w(a::int32::mult32_32_p31(x, y))
        }),
        ("MULT32_32_P31_ovflw", id2::MULT32_32_P31_OVFLW, |x, y| {
            w(a::int32::mult32_32_p31_ovflw(x, y))
        }),
        ("MULT32_32_Q32", id2::MULT32_32_Q32, |x, y| {
            w(a::int32::mult32_32_q32(x, y))
        }),
    ];
    // The oracle's own `OPUS_FAST_INT64` matches the port's; on a 64-bit host its macros are
    // the 64-bit forms.
    assert_eq!(c::constants()[14] != 0, a::OPUS_FAST_INT64);
    for (k, &(name, op, f)) in int64.iter().enumerate() {
        if !a::OPUS_FAST_INT64 {
            break;
        }
        check2(
            name,
            f,
            |x, y| c::op2(op, x, y),
            |r| (rany(r), rany(r)),
            500 + k as u64,
        );
    }
    // The 32-bit forms add partial products with (overflow-checked) ADD32: keep the exact
    // product within range, as every libopus caller does.
    let mut differs = [0usize; 8];
    for (k, &(name, op, f)) in int32.iter().enumerate() {
        let is16 = op <= id2::MULT16_32_Q15;
        let gen_ab = |r: &mut Rng| {
            if is16 {
                // `SHL(MULT16_16(a, SHR(b,16)), 1)` must not reach 2^31.
                (rany(r), rany(r).clamp(i32::MIN + 65536, i32::MAX - 65536))
            } else {
                // The exact result (and so every partial sum) must fit in 32 bits.
                let total = match op {
                    id2::MULT32_32_Q16 => 45,
                    id2::MULT32_32_Q32 => 61,
                    _ => 60,
                };
                let wa = r.range_i32(total - 31, 31) as u32;
                (rbits(r, wa), rbits(r, total as u32 - wa))
            }
        };
        check2(
            name,
            f,
            |x, y| c::op2_int32(op, x, y),
            gen_ab,
            600 + k as u64,
        );
        let mut rng = Rng::new(700 + k as u64);
        for _ in 0..10_000 {
            let (x, y) = gen_ab(&mut rng);
            if c::op2_int32(op, x, y) != c::op2(op, x, y) {
                differs[k] += 1;
            }
        }
    }
    // The 16x32 forms and MULT32_32_Q16 are exact rewrites; the other 32x32 ones drop partial
    // products.
    assert_eq!(
        &differs[..4],
        &[0, 0, 0, 0],
        "16x32 and MULT32_32_Q16 forms must be identical"
    );
    assert!(
        differs[4] > 0,
        "MULT32_32_Q31 forms are expected to differ sometimes"
    );
    assert_eq!(c::mult16_16u(65535, 65535), a::mult16_16u(65535, 65535));
    let mut rng = Rng::new(701);
    for _ in 0..N {
        let (x, y) = (rng.next_u32(), rng.next_u32());
        assert_eq!(
            a::mult16_16u(x & 0xffff, y & 0xffff),
            c::mult16_16u(x & 0xffff, y & 0xffff)
        );
        assert_eq!(a::uadd32(x, y), c::uadd32(x, y));
        assert_eq!(a::usub32(x, y), c::usub32(x, y));
        let z = rng.next_u64() as i64;
        let s = rng.range_i32(0, 63);
        assert_eq!(a::shr64(z, s), c::shr64(z, s), "SHR64({z}, {s})");
    }
}

#[test]
fn constants_and_qconst_match_oracle() {
    let k = c::constants();
    let size = |bytes: usize| bytes as i32;
    let want: [i32; c::N_CONSTANTS] = [
        i32::from(a::Q15ONE),
        a::Q31ONE,
        w(a::COEF_ONE) as i32,
        a::SIG_SHIFT,
        a::SIG_SAT,
        a::NORM_SHIFT,
        a::NORM_SCALING,
        a::DB_SHIFT,
        a::EPSILON,
        a::VERY_SMALL,
        i32::from(a::VERY_LARGE16),
        i32::from(a::Q15_ONE),
        a::RES_SHIFT,
        a::MAX_ENCODING_DEPTH,
        i32::from(a::OPUS_FAST_INT64),
        size(size_of::<OpusVal16>()),
        size(size_of::<OpusVal32>()),
        size(size_of::<a::OpusVal64>()),
        size(size_of::<CeltSig>()),
        size(size_of::<a::CeltNorm>()),
        size(size_of::<a::CeltEner>()),
        size(size_of::<a::CeltGlog>()),
        size(size_of::<OpusRes>()),
        size(size_of::<CeltCoef>()),
        size(size_of::<sm::KissFftScalar>()),
        size(size_of::<sm::KissTwiddleScalar>()),
        a::GLOBAL_STACK_SIZE as i32,
    ];
    assert_eq!(want, k);

    let mut rng = Rng::new(800);
    for _ in 0..N {
        // QCONST16: |x * 2^bits| < 32767.
        let bits = rng.range_i32(0, 15);
        let x = f64::from(rng.f32_sym()) * 32766.0 / f64::from(1 << bits);
        assert_eq!(
            i32::from(a::qconst16(x, bits)),
            c::qconst16(x, bits),
            "QCONST16({x}, {bits})"
        );
        let xf = x as f32;
        assert_eq!(
            i32::from(a::qconst16(f64::from(xf), bits)),
            c::qconst16f(xf, bits),
            "QCONST16({xf}f, {bits})"
        );
        let bits = rng.range_i32(0, 31);
        let x = f64::from(rng.f32_sym()) * 2147483000.0 / (1u64 << bits) as f64;
        assert_eq!(
            a::qconst32(x, bits),
            c::qconst32(x, bits),
            "QCONST32({x}, {bits})"
        );
        let xf = x as f32;
        assert_eq!(
            a::qconst32(f64::from(xf), bits),
            c::qconst32f(xf, bits),
            "QCONST32({xf}f, {bits})"
        );
        let g = f64::from(rng.f32_sym()) * 127.0;
        assert_eq!(a::gconst(g), c::gconst(g), "GCONST({g})");
        let bits = rng.range_i32(0, 24);
        assert_eq!(
            a::gconst2(g, bits),
            c::gconst2(g, bits),
            "GCONST2({g}, {bits})"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Mathops
// ---------------------------------------------------------------------------------------------

type M1 = fn(i32) -> i64;

fn check_math1(name: &str, op: i32, f: M1, xs: impl IntoIterator<Item = i32>) -> usize {
    let mut n = 0;
    for x in xs {
        assert_eq!(f(x), i64::from(c::math1(op, x)), "{name}({x})");
        n += 1;
    }
    n
}

fn randoms(seed: u64, n: usize, mut g: impl FnMut(&mut Rng) -> i32) -> Vec<i32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| g(&mut rng)).collect()
}

fn positive(r: &mut Rng) -> i32 {
    rany(r).checked_abs().unwrap_or(i32::MAX).max(1)
}

#[test]
fn mathops_match_oracle() {
    let all16 = || -32768..=32767;
    // celt_rsqrt_norm: Q16 in [0.25, 1), exhaustive.
    check_math1(
        "celt_rsqrt_norm",
        mid1::CELT_RSQRT_NORM,
        |x| w(m::celt_rsqrt_norm(x)),
        16384..=65535,
    );
    let q31_quarter = |r: &mut Rng| r.range_i32(1 << 29, i32::MAX);
    check_math1(
        "celt_rsqrt_norm32",
        mid1::CELT_RSQRT_NORM32,
        |x| w(m::celt_rsqrt_norm32(x)),
        randoms(900, N, q31_quarter)
            .into_iter()
            .chain([1 << 29, i32::MAX]),
    );
    for (name, op, f) in [
        ("celt_sqrt", mid1::CELT_SQRT, (|x| w(m::celt_sqrt(x))) as M1),
        ("celt_sqrt32", mid1::CELT_SQRT32, |x| w(m::celt_sqrt32(x))),
        ("celt_log2", mid1::CELT_LOG2, |x| w(m::celt_log2(x))),
        ("celt_log2_db", mid1::CELT_LOG2_DB, |x| {
            w(m::celt_log2_db(x))
        }),
    ] {
        check_math1(name, op, f, 0..=(1 << 20));
        check_math1(
            name,
            op,
            f,
            randoms(901, N, positive).into_iter().chain([
                (1 << 30) - 1,
                1 << 30,
                i32::MAX - 1,
                i32::MAX,
            ]),
        );
    }
    check_math1(
        "celt_cos_norm",
        mid1::CELT_COS_NORM,
        |x| w(m::celt_cos_norm(x)),
        (0..=(1 << 17)).chain(randoms(902, N, rany)),
    );
    let q30 = |r: &mut Rng| r.range_i32(-(1 << 30), 1 << 30);
    let q30_edges = [-(1 << 30), -(1 << 30) + 1, -1, 0, 1, (1 << 30) - 1, 1 << 30];
    check_math1(
        "celt_cos_norm32",
        mid1::CELT_COS_NORM32,
        |x| w(m::celt_cos_norm32(x)),
        randoms(903, N, q30).into_iter().chain(q30_edges),
    );
    check_math1(
        "celt_atan_norm",
        mid1::CELT_ATAN_NORM,
        |x| w(m::celt_atan_norm(x)),
        randoms(904, N, q30).into_iter().chain(q30_edges),
    );
    for (name, op, f) in [
        (
            "celt_exp2_frac",
            mid1::CELT_EXP2_FRAC,
            (|x| w(m::celt_exp2_frac(x as i16))) as M1,
        ),
        ("celt_exp2", mid1::CELT_EXP2, |x| w(m::celt_exp2(x as i16))),
        ("celt_rcp_norm16", mid1::CELT_RCP_NORM16, |x| {
            w(m::celt_rcp_norm16(x as i16))
        }),
        ("celt_atan01", mid1::CELT_ATAN01, |x| {
            w(m::celt_atan01(x as i16))
        }),
    ] {
        check_math1(name, op, f, all16());
    }
    // celt_exp2_db_frac: a Q24 fraction with QEXT; any (non-overflowing) value otherwise.
    let frac_gen = |r: &mut Rng| {
        if cfg!(feature = "qext") {
            r.range_i32(0, (1 << 24) - 1)
        } else {
            rany(r).min(i32::MAX - 8192)
        }
    };
    check_math1(
        "celt_exp2_db_frac",
        mid1::CELT_EXP2_DB_FRAC,
        |x| w(m::celt_exp2_db_frac(x)),
        randoms(905, N, frac_gen).into_iter().chain(0..(1 << 16)),
    );
    check_math1(
        "celt_exp2_db",
        mid1::CELT_EXP2_DB,
        |x| w(m::celt_exp2_db(x)),
        randoms(906, N, |r| rany(r).min(i32::MAX - 8192))
            .into_iter()
            .chain((-40 << 24..=20 << 24).step_by(4099)),
    );
    check_math1(
        "celt_rcp",
        mid1::CELT_RCP,
        |x| w(m::celt_rcp(x)),
        (1..=(1 << 17)).chain(randoms(907, N, positive)),
    );
    check_math1(
        "celt_rcp_norm32",
        mid1::CELT_RCP_NORM32,
        |x| w(m::celt_rcp_norm32(x)),
        randoms(908, N, |r| r.range_i32(1 << 30, i32::MAX))
            .into_iter()
            .chain([1 << 30, i32::MAX]),
    );
    check_math1(
        "celt_ilog2",
        mid1::CELT_ILOG2,
        |x| w(m::celt_ilog2(x)),
        (1..=(1 << 16)).chain(randoms(909, N, positive)),
    );
    check_math1(
        "celt_zlog2",
        mid1::CELT_ZLOG2,
        |x| w(m::celt_zlog2(x)),
        randoms(910, N, rany),
    );

    // Two-argument functions.
    let mut rng = Rng::new(911);
    for i in 0..N {
        let b = positive(&mut rng);
        let x = rany(&mut rng);
        assert_eq!(
            w(m::celt_div(x, b)),
            i64::from(c::math2(mid2::CELT_DIV, x, b)),
            "celt_div({x}, {b}) case {i}"
        );
        // frac_div32: |a| < 2b (every libopus call divides values of similar magnitude).
        let a_ = ((rng.f32_sym() * 1.99) as f64 * f64::from(b)) as i32;
        assert_eq!(
            w(m::frac_div32_q29(a_, b)),
            i64::from(c::math2(mid2::FRAC_DIV32_Q29, a_, b)),
            "frac_div32_q29({a_}, {b})"
        );
        assert_eq!(
            w(m::frac_div32(a_, b)),
            i64::from(c::math2(mid2::FRAC_DIV32, a_, b)),
            "frac_div32({a_}, {b})"
        );
        let (y, x) = (rng.range_i32(0, 1 << 30), rng.range_i32(0, 1 << 30));
        assert_eq!(
            w(m::celt_atan2p_norm(y, x)),
            i64::from(c::math2(mid2::CELT_ATAN2P_NORM, y, x)),
            "celt_atan2p_norm({y}, {x})"
        );
        let (y, x) = (rng.range_i32(0, 32767), rng.range_i32(0, 32767));
        assert_eq!(
            w(m::celt_atan2p(y as i16, x as i16)),
            i64::from(c::math2(mid2::CELT_ATAN2P, y, x)),
            "celt_atan2p({y}, {x})"
        );
    }
    for y in (0..=32767).step_by(97).chain([32767]) {
        for x in (0..=32767).step_by(89).chain([32767]) {
            assert_eq!(
                w(m::celt_atan2p(y as i16, x as i16)),
                i64::from(c::math2(mid2::CELT_ATAN2P, y, x)),
                "celt_atan2p({y}, {x}) grid"
            );
        }
    }
    for (y, x) in [
        (0, 0),
        (0, 1 << 30),
        (1 << 30, 0),
        (1 << 30, 1 << 30),
        (1, 1),
    ] {
        assert_eq!(
            w(m::celt_atan2p_norm(y, x)),
            i64::from(c::math2(mid2::CELT_ATAN2P_NORM, y, x))
        );
    }
}

#[test]
fn integer_helpers_and_maxabs_match_oracle() {
    let mut rng = Rng::new(1000);
    for _ in 0..N {
        let (x, y) = (rany(&mut rng), rany(&mut rng));
        assert_eq!(
            m::frac_mul16(x, y),
            c::frac_mul16(x, y),
            "FRAC_MUL16({x}, {y})"
        );
        let u = rng.next_u32() | 1;
        assert_eq!(m::isqrt32(u), c::isqrt32(u));
    }
    for _ in 0..2000 {
        let n = rng.range_i32(0, 64) as usize;
        let v16: Vec<i16> = (0..n).map(|_| r16(&mut rng) as i16).collect();
        assert_eq!(m::celt_maxabs16(&v16), c::maxabs16(&v16));
        let vres: Vec<OpusRes> = (0..n).map(|_| res(rres(&mut rng))).collect();
        assert_eq!(
            w(m::celt_maxabs_res(&vres)),
            i64::from(c::maxabs_res(&vres))
        );
        let v32: Vec<i32> = (0..n)
            .map(|_| {
                let x = rany(&mut rng);
                if x == i32::MIN { 0 } else { x }
            })
            .collect();
        assert_eq!(m::celt_maxabs32(&v32), c::maxabs32(&v32));
    }
}

#[test]
fn float_api_matches_oracle() {
    let mut rng = Rng::new(1100);
    for _ in 0..N {
        let big = rng.f32_sym() * 70000.0;
        assert_eq!(m::float2int(big), c::float2int(big), "float2int({big})");
        let s = rng.f32_sym() * 1.2;
        assert_eq!(
            i32::from(m::float2int16(s)),
            c::float2int16(s),
            "FLOAT2INT16({s})"
        );
        assert_eq!(m::float2int24(s), c::float2int24(s), "FLOAT2INT24({s})");
        let (y, x) = (rng.f32_sym(), rng.f32_sym());
        assert_eq!(
            m::fast_atan2f(y, x).to_bits(),
            c::fast_atan2f(y, x).to_bits()
        );
        #[cfg(feature = "qext")]
        {
            let t = rng.f32_sym() * 4.0;
            assert_eq!(
                m::celt_cos_norm2(t).to_bits(),
                c::celt_cos_norm2(t).to_bits(),
                "celt_cos_norm2({t})"
            );
        }
    }
    for k in -40..40 {
        let h = k as f32 + 0.5;
        assert_eq!(m::float2int(h), c::float2int(h), "float2int tie {h}");
    }
    for _ in 0..200 {
        let n = rng.range_i32(0, 300) as usize;
        let x: Vec<f32> = (0..n).map(|_| rng.f32_sym() * 2.5).collect();
        let mut out = vec![0i16; n];
        m::celt_float2int16(&x, &mut out);
        assert_eq!(out, c::celt_float2int16(&x));
        let mut rx = x.clone();
        let mut cx = x;
        let r = m::opus_limit2_checkwithin1(&mut rx);
        let cr = c::opus_limit2_checkwithin1(&mut cx);
        assert_eq!(i32::from(r), cr);
        assert_eq!(
            rx.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            cx.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Static modes
// ---------------------------------------------------------------------------------------------

fn ints<T: Copy + Into<i64>>(v: &[T]) -> Vec<i64> {
    v.iter().map(|&x| x.into()).collect()
}

fn check_mode(mode: &CeltMode, fs: i32, frame_size: i32) {
    let s = c::mode_scalars(fs, frame_size).expect("static mode exists in C");
    let mut want = vec![mode.fs, mode.overlap, mode.nb_ebands, mode.eff_ebands];
    want.extend(mode.preemph.iter().map(|&p| i32::from(p)));
    want.extend([
        mode.max_lm,
        mode.nb_short_mdcts,
        mode.short_mdct_size,
        mode.nb_alloc_vectors,
        mode.mdct.n,
        mode.mdct.maxshift,
        mode.cache.size,
    ]);
    #[cfg(feature = "qext")]
    want.push(mode.qext_cache.size);
    #[cfg(not(feature = "qext"))]
    want.push(-1);
    assert_eq!(want, s, "mode {fs}/{frame_size} scalars");

    let arr = |which: i32, n: usize| -> Vec<i64> {
        c::mode_array(fs, frame_size, which, n)
            .expect("mode")
            .into_iter()
            .map(i64::from)
            .collect()
    };
    assert_eq!(
        ints(&mode.e_bands),
        arr(mode_arr::E_BANDS, mode.e_bands.len())
    );
    assert_eq!(
        mode.e_bands.len(),
        mode.nb_ebands as usize + 1,
        "eBands length"
    );
    let n_alloc = (mode.nb_alloc_vectors * mode.nb_ebands) as usize;
    assert_eq!(
        ints(&mode.alloc_vectors[..n_alloc]),
        arr(mode_arr::ALLOC_VECTORS, n_alloc)
    );
    assert_eq!(
        ints(&mode.log_n),
        arr(mode_arr::LOG_N, mode.nb_ebands as usize)
    );
    assert_eq!(mode.window.len(), mode.overlap as usize);
    assert_eq!(ints(&mode.window), arr(mode_arr::WINDOW, mode.window.len()));
    let n = mode.mdct.n as usize;
    assert_eq!(mode.mdct.trig.len(), n - ((n >> 1) >> mode.mdct.maxshift));
    assert_eq!(
        ints(&mode.mdct.trig),
        arr(mode_arr::MDCT_TRIG, mode.mdct.trig.len())
    );
    assert_eq!(
        ints(&mode.cache.index),
        arr(mode_arr::CACHE_INDEX, mode.cache.index.len())
    );
    assert_eq!(mode.cache.bits.len(), mode.cache.size as usize);
    assert_eq!(
        ints(&mode.cache.bits),
        arr(mode_arr::CACHE_BITS, mode.cache.bits.len())
    );
    assert_eq!(
        ints(&mode.cache.caps),
        arr(mode_arr::CACHE_CAPS, mode.cache.caps.len())
    );
    #[cfg(feature = "qext")]
    {
        let q = &mode.qext_cache;
        assert_eq!(
            ints(&q.index),
            arr(mode_arr::QEXT_CACHE_INDEX, q.index.len())
        );
        assert_eq!(ints(&q.bits), arr(mode_arr::QEXT_CACHE_BITS, q.bits.len()));
        assert_eq!(ints(&q.caps), arr(mode_arr::QEXT_CACHE_CAPS, q.caps.len()));
    }
    for (k, st) in mode.mdct.kfft.iter().enumerate() {
        let st: &KissFftState = st;
        let cst = c::mode_fft(fs, frame_size, k as i32, st.twiddles.len()).expect("fft state");
        let mut want = vec![st.nfft, w(st.scale) as i32, st.scale_shift, st.shift];
        want.extend(st.factors.iter().map(|&f| i32::from(f)));
        assert_eq!(want, cst.scalars, "mode {fs}/{frame_size} fft {k} scalars");
        assert_eq!(ints(&st.bitrev), ints(&cst.bitrev), "fft {k} bitrev");
        let tw: Vec<i32> = st
            .twiddles
            .iter()
            .flat_map(|t| [w(t.r) as i32, w(t.i) as i32])
            .collect();
        assert_eq!(tw, cst.twiddles, "fft {k} twiddles");
    }
}

#[test]
fn static_modes_match_oracle() {
    check_mode(&sm::MODE48000_960_120, 48000, 960);
    assert!(std::ptr::eq(
        sm::STATIC_MODE_LIST[0],
        &sm::MODE48000_960_120
    ));
    #[cfg(feature = "qext")]
    {
        check_mode(&sm::MODE96000_1920_240, 96000, 1920);
        assert!(std::ptr::eq(
            sm::STATIC_MODE_LIST[1],
            &sm::MODE96000_1920_240
        ));
    }
    assert_eq!(
        sm::STATIC_MODE_LIST.len(),
        if cfg!(feature = "qext") { 2 } else { 1 }
    );
    // The fixed-point window reaches (almost) full scale, the float one 1.0.
    let last = i64::from(*sm::MODE48000_960_120.window.last().expect("window"));
    assert!(last > i64::from(a::COEF_ONE) - 16);
}

// ---------------------------------------------------------------------------------------------
// Range coder, CWRS, Laplace in a fixed-point build
// ---------------------------------------------------------------------------------------------

#[test]
fn range_coder_matches_oracle() {
    use opusorus_oracle::foundation as cf;
    let mut rng = Rng::new(1200);
    for iter in 0..1000 {
        let n = rng.range_i32(1, 200) as usize;
        let mut ops = Vec::with_capacity(n);
        for _ in 0..n {
            ops.push(match rng.range_i32(0, 2) {
                0 => {
                    let ft = rng.range_i32(2, i32::MAX) as u32;
                    [4, rng.next_u32() % ft, ft, 0]
                }
                1 => {
                    let bits = rng.range_i32(1, 25) as u32;
                    [5, rng.next_u32() & ((1 << bits) - 1), bits, 0]
                }
                _ => [3, rng.range_i32(0, 7) as u32, 8, 0],
            });
        }
        let size = if iter % 4 == 0 {
            rng.range_i32(1, 64) as usize
        } else {
            1024
        };
        let c_out = cf::ec_encode_ops(&ops, size, 0);
        let mut buf = vec![0u8; size];
        let mut tells = Vec::with_capacity(n);
        let (rng_state, error) = {
            let mut e = EcEnc::new(&mut buf);
            for o in &ops {
                match o[0] {
                    3 => e.enc_icdf(o[1] as usize, &cf::ORACLE_ICDF, o[2]),
                    4 => e.enc_uint(o[1], o[2]),
                    _ => e.enc_bits(o[1], o[2]),
                }
                tells.push((e.tell(), e.tell_frac()));
            }
            e.done();
            (e.rng, e.error)
        };
        assert_eq!(buf, c_out.buf, "iter {iter}");
        assert_eq!(tells, c_out.tells, "iter {iter}");
        assert_eq!((rng_state, error), (c_out.rng, c_out.error), "iter {iter}");
    }
}

/// Largest column of each row of `CELT_PVQ_U_ROW` (the non-custom, non-QEXT table, a subset of
/// the others).
const PVQ_ROW_MAX: [i32; 15] = [
    176, 176, 176, 176, 176, 176, 96, 54, 37, 28, 24, 19, 18, 16, 14,
];

/// Whether `(n, k)` is codable: `V(n,k)` is in the table and fits in 32 bits.
fn pvq_ok(n: i32, k: i32) -> bool {
    let (lo, hi) = (n.min(k + 1), n.max(k + 1));
    lo <= 14
        && hi <= PVQ_ROW_MAX[lo as usize]
        && u64::from(cwrs::celt_pvq_u(n, k)) + u64::from(cwrs::celt_pvq_u(n, k + 1)) < 1 << 32
}

#[test]
fn cwrs_matches_oracle() {
    let mut rng = Rng::new(1300);
    let mut tested = 0;
    for iter in 0..20_000 {
        let n = rng.range_i32(2, 64);
        let k = rng.range_i32(1, 64);
        if !pvq_ok(n, k) {
            continue;
        }
        let mut y = vec![0i32; n as usize];
        for _ in 0..k {
            y[rng.range_i32(0, n - 1) as usize] += 1;
        }
        for v in &mut y {
            if rng.next_u32() & 1 == 1 {
                *v = -*v;
            }
        }
        let c_out = c::cwrs_roundtrip(&y, k, 256);
        let mut buf = vec![0u8; 256];
        let enc_rng = {
            let mut e = EcEnc::new(&mut buf);
            cwrs::encode_pulses(&y, n, k, &mut e);
            e.done();
            e.rng
        };
        assert_eq!(buf, c_out.buf, "iter {iter} n={n} k={k}");
        assert_eq!(enc_rng, c_out.rng);
        let mut y_out = vec![0i32; n as usize];
        let mut d = EcDec::new(&buf);
        let yy = cwrs::decode_pulses(&mut y_out, n, k, &mut d);
        assert_eq!(y_out, c_out.y);
        assert_eq!(y_out, y);
        assert_eq!(yy, c_out.yy, "decode_pulses squared norm");
        tested += 1;
    }
    assert!(tested > 4000, "{tested}");
}

#[test]
fn laplace_matches_oracle() {
    let mut rng = Rng::new(1400);
    for iter in 0..2000 {
        let count = rng.range_i32(1, 60) as usize;
        let values: Vec<i32> = (0..count).map(|_| rng.range_i32(-300, 300)).collect();
        // ec_laplace_encode: fs in (0, 32736], decay in (0, 11456].
        let fs: Vec<u32> = (0..count).map(|_| rng.range_i32(1, 32736) as u32).collect();
        let decay: Vec<i32> = (0..count).map(|_| rng.range_i32(1, 11456)).collect();
        let c_out = c::laplace_roundtrip(&values, &fs, &decay, 1024);
        let mut buf = vec![0u8; 1024];
        let mut encoded = values.clone();
        let enc_rng = {
            let mut e = EcEnc::new(&mut buf);
            for i in 0..count {
                laplace::ec_laplace_encode(&mut e, &mut encoded[i], fs[i], decay[i]);
            }
            e.done();
            e.rng
        };
        assert_eq!(encoded, c_out.encoded, "iter {iter}");
        assert_eq!(buf, c_out.buf, "iter {iter}");
        assert_eq!(enc_rng, c_out.rng);
        let mut d = EcDec::new(&buf);
        let decoded: Vec<i32> = (0..count)
            .map(|i| laplace::ec_laplace_decode(&mut d, fs[i], decay[i]))
            .collect();
        assert_eq!(decoded, c_out.decoded, "iter {iter}");
        assert_eq!(decoded, encoded);
    }
}
