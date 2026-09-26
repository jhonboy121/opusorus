//! Differential tests of the `fixed-point-debug` feature (libopus `FIXED_DEBUG`) against the
//! oracle built with `-DFIXED_DEBUG`:
//!
//! * every checking macro of `celt/fixed_debug.h` (and the `arch.h` macros built on them) and of
//!   `silk/MacroDebug.h`, on random and edge operands inside and outside their documented
//!   ranges: the value, the diagnostics (the C `fprintf` text, without the `in <file>: line
//!   <n>` location, which names the C and the Rust source respectively) and, for CELT, the
//!   `celt_mips` operation count;
//! * whole encode/decode streams (normal and extreme signals, every build option combined with
//!   the feature): packets, output, and the number of diagnostics.
//!
//! The codec differential suites (`celt_*.rs`, `opus_*.rs`, `silk_*.rs`, `api_overflow.rs`, ...)
//! also run in this configuration and compare every output bit-exactly with the `FIXED_DEBUG`
//! oracle (whose results differ from a release build where the debug macros compute
//! differently, e.g. `MULT32_32_Q31` is always the 16-bit partial-product form).
//!
//! `celt_mips` is a plain global in C, so the tests of this file run one at a time ([`LOCK`]).

#![cfg(feature = "fixed-point-debug")]
#![allow(
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::type_complexity,
    reason = "test code: tables of (name, C op id, Rust function) rows"
)]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Mutex, MutexGuard};

use opusorus::celt::arch as a;
use opusorus::celt::arch::{CeltCoef, OpusRes};
use opusorus::fixed_debug as fd;
use opusorus::silk::macros as s;
use opusorus_conformance::{Rng, signals};
use opusorus_oracle::api as capi;
use opusorus_oracle::fixed_debug as c;

/// Serializes the tests (C's `celt_mips` is a process global).
static LOCK: Mutex<()> = Mutex::new(());

/// Takes [`LOCK`] (a test that failed while holding it does not stop the others).
fn lock() -> MutexGuard<'static, ()> {
    match LOCK.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Random cases per macro.
const N: usize = 20_000;

/// A C or Rust diagnostic without its `in <file>: line <n>` location.
fn strip_location(msg: &str) -> String {
    let mut out = String::new();
    for line in msg.split_inclusive('\n') {
        match line.rfind(" in ") {
            Some(i) if line[i..].contains(": line ") => {
                out.push_str(&line[..i]);
                if line.ends_with('\n') {
                    out.push('\n');
                }
            }
            _ => out.push_str(line),
        }
    }
    out
}

/// Runs `f`, returning its result, the Rust diagnostics it reported (on this thread) and the
/// `celt_mips` it counted.
fn rust_run<R>(f: impl FnOnce() -> R) -> (R, Vec<String>, i64) {
    let msgs = Rc::new(RefCell::new(Vec::new()));
    let sink = Rc::clone(&msgs);
    let prev = fd::set_handler(Some(Box::new(move |m: &str| {
        sink.borrow_mut().push(m.to_owned());
    })));
    fd::set_celt_mips(0);
    let r = f();
    let mips = fd::celt_mips();
    fd::set_handler(prev);
    let v = msgs.borrow().clone();
    (r, v, mips)
}

/// Runs `f` on the C side, returning its result, the C diagnostics and `celt_mips`.
fn c_run<R>(f: impl FnOnce() -> R) -> (R, Vec<String>, i64) {
    c::clear();
    c::set_celt_mips(0);
    let r = f();
    let mips = c::celt_mips();
    let n = c::count();
    let msgs = c::messages();
    assert!(n <= c::KEEP, "more C messages than captured: {n}");
    (r, msgs, mips)
}

/// Compares the diagnostics of a codec call as sets of distinct messages: C evaluates the
/// operands of its macros in an unspecified order, and a macro that uses an operand several
/// times evaluates its side effects (the operand's own checks) several times (e.g. the debug
/// `MULT32_32_Q31(a,b)` expands `b` three times), where the Rust functions evaluate each operand
/// once.
#[track_caller]
fn same_diagnostics(what: &str, r: &[String], c: &[String]) {
    let norm = |v: &[String]| {
        let mut v: Vec<String> = v.iter().map(|m| strip_location(m)).collect();
        v.sort();
        v.dedup();
        v
    };
    assert_eq!(norm(r), norm(c), "{what}: diagnostics");
}

/// Compares one macro evaluation on both sides.
#[track_caller]
fn check(what: &str, r: (i64, Vec<String>, i64), c: (i64, Vec<String>, i64), mips: bool) {
    assert_eq!(r.0, c.0, "{what}: value");
    let rs: Vec<String> = r.1.iter().map(|m| strip_location(m)).collect();
    let cs: Vec<String> = c.1.iter().map(|m| strip_location(m)).collect();
    assert_eq!(rs, cs, "{what}: diagnostics");
    if mips {
        assert_eq!(r.2, c.2, "{what}: celt_mips");
    }
}

/// Operand generator: in-range values, values just outside the 16-bit range and the full
/// 32-bit range (the checks), and edges.
fn operand(rng: &mut Rng, k: u32) -> i32 {
    match k % 6 {
        0 => i32::from(rng.i16()),
        1 => rng.range_i32(-40000, 40000),
        2 => rng.next_u32() as i32,
        3 => [
            0,
            1,
            -1,
            32767,
            -32768,
            32768,
            -32769,
            i32::MAX,
            i32::MIN + 1,
            65535,
        ][rng.range_i32(0, 9) as usize],
        4 => rng.range_i32(-(1 << 24), 1 << 24),
        _ => rng.range_i32(-1000, 1000),
    }
}

/// `x` without `i32::MIN` (whose negation is undefined in C).
const fn no_min(x: i32) -> i32 {
    if x == i32::MIN { i32::MIN + 1 } else { x }
}

type R1 = fn(i32) -> i64;
type R2 = fn(i32, i32) -> i64;
type R3 = fn(i32, i32, i32) -> i64;

fn w(x: impl Into<i64>) -> i64 {
    x.into()
}

/// One-operand CELT macros: `(name, C op id, Rust, operand filter)`.
fn celt_unary() -> Vec<(&'static str, i32, R1, fn(i32) -> i32)> {
    vec![
        ("NEG16", 0, |x| w(a::neg16(x)), no_min),
        ("NEG32", 1, |x| w(a::neg32(x)), no_min),
        ("EXTRACT16", 2, |x| w(a::extract16(x)), |x| x),
        ("EXTEND32", 3, |x| w(a::extend32(x)), |x| x),
        ("HALF16", 13, |x| w(a::half16(x)), |x| x),
        ("HALF32", 14, |x| w(a::half32(x)), |x| x),
        ("SATURATE16", 31, |x| w(a::saturate16(x)), |x| x),
        ("NEG32_ovflw", 49, |x| w(a::neg32_ovflw(x)), |x| x),
        (
            "SIG2WORD16",
            41,
            |x| w(a::sig2word16(x)),
            |x| x.clamp(-(1 << 30), 1 << 30),
        ),
        (
            "SIG2RES",
            60,
            |x| w(a::sig2res(x)),
            |x| x.clamp(-(1 << 30), 1 << 30),
        ),
        (
            "INT24TORES",
            64,
            |x| w(a::int24tores(x)),
            |x| x.clamp(-(1 << 30), 1 << 30),
        ),
        ("INT24TOSIG", 69, |x| w(a::int24tosig(x)), |x| x),
    ]
}

/// Shift macros: `(name, C op id, Rust, shift range)`.
fn celt_shifts() -> Vec<(&'static str, i32, R2, (i32, i32))> {
    vec![
        ("SHR16", 4, |x, s| w(a::shr16(x, s)), (0, 31)),
        ("SHL16", 5, |x, s| w(a::shl16(x, s)), (0, 31)),
        ("SHR32", 6, |x, s| w(a::shr32(x, s)), (0, 31)),
        ("SHL32", 7, |x, s| w(a::shl32(x, s)), (0, 31)),
        ("PSHR32", 8, |x, s| w(a::pshr32(x, s)), (0, 30)),
        ("VSHR32", 9, |x, s| w(a::vshr32(x, s)), (-31, 31)),
        ("ROUND16", 11, |x, s| w(a::round16(x, s)), (0, 30)),
        ("SROUND16", 12, |x, s| w(a::sround16(x, s)), (0, 30)),
        ("SHL32_ovflw", 50, |x, s| w(a::shl32_ovflw(x, s)), (0, 31)),
        ("PSHR32_ovflw", 51, |x, s| w(a::pshr32_ovflw(x, s)), (0, 30)),
        ("SHR", 54, |x, s| w(a::shr(x, s)), (0, 31)),
        ("PSHR", 55, |x, s| w(a::pshr(x, s)), (0, 30)),
    ]
}

/// Two-operand CELT macros.
fn celt_binary() -> Vec<(&'static str, i32, R2)> {
    vec![
        ("ADD16", 15, |x, y| w(a::add16(x, y))),
        ("SUB16", 16, |x, y| w(a::sub16(x, y))),
        ("ADD32", 17, |x, y| w(a::add32(x, y))),
        ("SUB32", 18, |x, y| w(a::sub32(x, y))),
        ("MULT32_32_32", 22, |x, y| w(a::mult32_32_32(x, y))),
        ("MULT32_32_Q16", 23, |x, y| w(a::mult32_32_q16(x, y))),
        ("MULT16_16", 24, |x, y| w(a::mult16_16(x, y))),
        ("MULT16_32_Q15", 26, |x, y| w(a::mult16_32_q15(x, y))),
        ("MULT16_32_P16", 27, |x, y| w(a::mult16_32_p16(x, y))),
        ("SATURATE", 30, |x, y| w(a::saturate(x, y))),
        ("MULT16_16_Q11_32", 32, |x, y| w(a::mult16_16_q11_32(x, y))),
        ("MULT16_16_Q13", 33, |x, y| w(a::mult16_16_q13(x, y))),
        ("MULT16_16_Q14", 34, |x, y| w(a::mult16_16_q14(x, y))),
        ("MULT16_16_Q15", 35, |x, y| w(a::mult16_16_q15(x, y))),
        ("MULT16_16_P13", 36, |x, y| w(a::mult16_16_p13(x, y))),
        ("MULT16_16_P14", 37, |x, y| w(a::mult16_16_p14(x, y))),
        ("MULT16_16_P15", 38, |x, y| w(a::mult16_16_p15(x, y))),
        ("DIV32_16", 39, |x, y| w(a::div32_16(x, y))),
        ("DIV32", 40, |x, y| w(a::div32(x, y))),
        ("MULT16_32_Q16", 42, |x, y| w(a::mult16_32_q16(x, y))),
        ("MULT32_32_Q31", 43, |x, y| w(a::mult32_32_q31(x, y))),
        ("MULT32_32_P31", 44, |x, y| w(a::mult32_32_p31(x, y))),
        ("MULT32_32_P31_ovflw", 45, |x, y| {
            w(a::mult32_32_p31_ovflw(x, y))
        }),
        ("MULT32_32_Q32", 46, |x, y| w(a::mult32_32_q32(x, y))),
        ("ADD32_ovflw", 47, |x, y| w(a::add32_ovflw(x, y))),
        ("SUB32_ovflw", 48, |x, y| w(a::sub32_ovflw(x, y))),
        ("MULT16_16SU", 52, |x, y| w(a::mult16_16su(x, y))),
        ("MULT_COEF_32", 70, |x, y| w(a::mult_coef_32(coef(x), y))),
        ("MULT_COEF", 72, |x, y| w(a::mult_coef(coef(x), coef(y)))),
        ("MULT_COEF_TAPS", 73, |x, y| {
            w(a::mult_coef_taps(x as i16, y as i16))
        }),
    ]
}

/// Three-operand CELT macros (`c` is the accumulator).
fn celt_ternary() -> Vec<(&'static str, i32, R3)> {
    vec![
        ("MAC16_16", 25, |x, y, z| w(a::mac16_16(z, x, y))),
        ("MAC16_32_Q15", 28, |x, y, z| w(a::mac16_32_q15(z, x, y))),
        ("MAC16_32_Q16", 29, |x, y, z| w(a::mac16_32_q16(z, x, y))),
        ("MAC_COEF_32_ARM", 71, |x, y, z| {
            w(a::mac_coef_32_arm(z, coef(x), y))
        }),
    ]
}

/// `x` as a `celt_coef` (C's conversion at the call: truncation to 16 bits without QEXT).
const fn coef(x: i32) -> CeltCoef {
    x as CeltCoef
}

/// `x` as an `opus_res` (C's conversion at the call).
const fn res(x: i32) -> OpusRes {
    x as OpusRes
}

#[test]
fn celt_macros_match_oracle() {
    let _g = lock();
    let mut rng = Rng::new(0xdeb6);
    let mut reported = 0usize;
    for (name, op, f, filt) in celt_unary() {
        for i in 0..N {
            let x = filt(operand(&mut rng, i as u32));
            let r = rust_run(|| f(x));
            let cc = c_run(|| c::celt(op, i64::from(x), 0, 0));
            reported += cc.1.len();
            check(&format!("{name}({x})"), r, cc, true);
        }
    }
    for (name, op, f, (lo, hi)) in celt_shifts() {
        for i in 0..N {
            let x = operand(&mut rng, i as u32);
            let sh = rng.range_i32(lo, hi);
            let r = rust_run(|| f(x, sh));
            let cc = c_run(|| c::celt(op, i64::from(x), i64::from(sh), 0));
            reported += cc.1.len();
            check(&format!("{name}({x}, {sh})"), r, cc, true);
        }
    }
    for (name, op, f) in celt_binary() {
        for i in 0..N {
            let x = operand(&mut rng, i as u32);
            let mut y = operand(&mut rng, (i / 6) as u32);
            match op {
                // C `int` products must not overflow (undefined behaviour).
                21 => y = y.clamp(-40000, 40000),
                // `-b` of INT_MIN is undefined.
                30 => y = no_min(y),
                // A zero divisor is reported (and 0 returned) by both.
                _ => {}
            }
            let r = rust_run(|| f(x, y));
            let cc = c_run(|| c::celt(op, i64::from(x), i64::from(y), 0));
            reported += cc.1.len();
            check(&format!("{name}({x}, {y})"), r, cc, true);
        }
    }
    // MULT16_16_16: an `int` product (undefined on overflow in C).
    for i in 0..N {
        let x = operand(&mut rng, i as u32).clamp(-46340, 46340);
        let y = operand(&mut rng, (i / 6) as u32).clamp(-46340, 46340);
        let r = rust_run(|| w(a::mult16_16_16(x, y)));
        let cc = c_run(|| c::celt(21, i64::from(x), i64::from(y), 0));
        reported += cc.1.len();
        check(&format!("MULT16_16_16({x}, {y})"), r, cc, true);
    }
    for (name, op, f) in celt_ternary() {
        for i in 0..N {
            let x = operand(&mut rng, i as u32);
            let y = operand(&mut rng, (i / 6) as u32);
            let z = operand(&mut rng, (i / 36) as u32);
            let r = rust_run(|| f(x, y, z));
            let cc = c_run(|| c::celt(op, i64::from(x), i64::from(y), i64::from(z)));
            reported += cc.1.len();
            check(&format!("{name}({x}, {y}, {z})"), r, cc, true);
        }
    }
    // Unsigned and 64-bit ones.
    for i in 0..N {
        let (x, y) = (rng.next_u32(), rng.next_u32());
        let (x, y) = if i % 4 == 0 { (x, u32::MAX) } else { (x, y) };
        let r = rust_run(|| w(a::uadd32(x, y)));
        let cc = c_run(|| c::celt(19, i64::from(x), i64::from(y), 0) as u32 as i64);
        check(&format!("UADD32({x}, {y})"), r, cc, true);
        let r = rust_run(|| w(a::usub32(x, y)));
        let cc = c_run(|| c::celt(20, i64::from(x), i64::from(y), 0) as u32 as i64);
        reported += cc.1.len();
        check(&format!("USUB32({x}, {y})"), r, cc, true);
        let r = rust_run(|| w(a::mult16_16u(x & 0xffff, y & 0xffff)));
        let cc =
            c_run(|| c::celt(53, i64::from(x & 0xffff), i64::from(y & 0xffff), 0) as u32 as i64);
        check("MULT16_16U", r, cc, true);
        let z = rng.next_u64() as i64;
        let sh = rng.range_i32(0, 63);
        let r = rust_run(|| a::shr64(z, sh));
        let cc = c_run(|| c::celt(10, z, i64::from(sh), 0));
        check(&format!("SHR64({z}, {sh})"), r, cc, true);
    }
    // arch.h macros on opus_res / int16.
    for i in 0..N {
        let x = operand(&mut rng, i as u32);
        let y = operand(&mut rng, (i / 6) as u32);
        let (xr, yr) = (res(x), res(y));
        let cases: [(&str, i32, i64, i64, fn(i32, i32) -> i64); 6] = [
            ("RES2INT16", 61, i64::from(xr), 0, |x, _| {
                w(a::res2int16(res(x)))
            }),
            ("RES2INT24", 62, i64::from(xr), 0, |x, _| {
                w(a::res2int24(res(x)))
            }),
            ("INT16TORES", 63, i64::from(x as i16), 0, |x, _| {
                w(a::int16tores(x as i16))
            }),
            ("ADD_RES", 65, i64::from(xr), i64::from(yr), |x, y| {
                w(a::add_res(res(x), res(y)))
            }),
            ("RES2SIG", 66, i64::from(xr), 0, |x, _| {
                w(a::res2sig(res(x)))
            }),
            ("INT16TOSIG", 68, i64::from(x as i16), 0, |x, _| {
                w(a::int16tosig(x as i16))
            }),
        ];
        for (name, op, ca, cb, f) in cases {
            let r = rust_run(|| f(x, y));
            let cc = c_run(|| c::celt(op, ca, cb, 0));
            reported += cc.1.len();
            check(&format!("{name}({x}, {y})"), r, cc, true);
        }
        let r = rust_run(|| w(a::mult16_res_q15(x as i16, yr)));
        let cc = c_run(|| c::celt(67, i64::from(x as i16), i64::from(yr), 0));
        check(&format!("MULT16_RES_Q15({x}, {y})"), r, cc, true);
        let r = rust_run(|| w(a::coef2val16(coef(x))));
        let cc = c_run(|| c::celt(74, i64::from(coef(x)), 0, 0));
        check(&format!("COEF2VAL16({x})"), r, cc, true);
    }
    // Operands outside the documented ranges are reported.
    assert!(reported > 10_000, "only {reported} CELT diagnostics");
}

/// SILK macros on 32-bit operands: `(name, C op id, Rust)`.
fn silk_binary() -> Vec<(&'static str, i32, R2)> {
    vec![
        ("silk_ADD32", 1, |x, y| w(s::silk_add32(x, y))),
        ("silk_SUB32", 4, |x, y| w(s::silk_sub32(x, y))),
        ("silk_ADD_SAT32", 7, |x, y| w(s::silk_add_sat32(x, y))),
        ("silk_SUB_SAT32", 10, |x, y| w(s::silk_sub_sat32(x, y))),
        ("silk_MUL", 12, |x, y| w(s::silk_mul(x, y))),
        ("silk_SMULWB", 16, |x, y| w(s::silk_smulwb(x, y))),
        ("silk_SMULWT", 18, |x, y| w(s::silk_smulwt(x, y))),
        ("silk_SMULL", 20, |x, y| s::silk_smull(x, y)),
        ("silk_SMULWW", 24, |x, y| w(s::silk_smulww(x, y))),
        ("silk_ADD_SAT16", 6, |x, y| {
            w(s::silk_add_sat16(x as i16, y))
        }),
        ("silk_SUB_SAT16", 9, |x, y| {
            w(s::silk_sub_sat16(x as i16, y))
        }),
        ("silk_ADD16", 0, |x, y| w(s::silk_add16(x as i16, y as i16))),
        ("silk_SUB16", 3, |x, y| w(s::silk_sub16(x as i16, y as i16))),
    ]
}

/// SILK macros with three 32-bit operands.
fn silk_ternary() -> Vec<(&'static str, i32, R3)> {
    vec![
        ("silk_MLA", 14, |x, y, z| w(s::silk_mla(x, y, z))),
        ("silk_SMLAWB", 17, |x, y, z| w(s::silk_smlawb(x, y, z))),
        ("silk_SMLAWT", 19, |x, y, z| w(s::silk_smlawt(x, y, z))),
        ("silk_SMLABB", 21, |x, y, z| w(s::silk_smlabb(x, y, z))),
        ("silk_SMLABT", 22, |x, y, z| w(s::silk_smlabt(x, y, z))),
        ("silk_SMLATT", 23, |x, y, z| w(s::silk_smlatt(x, y, z))),
        ("silk_SMLAWW", 25, |x, y, z| w(s::silk_smlaww(x, y, z))),
    ]
}

/// SILK shifts: `(name, C op id, Rust, shift range)`; the ranges include invalid shifts where
/// the C function checks them before shifting (and does not shift out of range).
fn silk_shifts() -> Vec<(&'static str, i32, R3, (i32, i32))> {
    vec![
        (
            "silk_LSHIFT32",
            30,
            |x, _, s| w(s::silk_lshift32(x, s)),
            (0, 31),
        ),
        (
            "silk_LSHIFT",
            30,
            |x, _, s| w(s::silk_lshift(x, s)),
            (0, 31),
        ),
        (
            "silk_LSHIFT_ovflw",
            32,
            |x, _, s| w(s::silk_lshift_ovflw(x, s)),
            (0, 31),
        ),
        (
            "silk_RSHIFT32",
            34,
            |x, _, s| w(s::silk_rshift32(x, s)),
            (0, 31),
        ),
        (
            "silk_RSHIFT",
            34,
            |x, _, s| w(s::silk_rshift(x, s)),
            (0, 31),
        ),
        (
            "silk_ADD_LSHIFT",
            37,
            |x, y, s| w(s::silk_add_lshift(x, y, s)),
            (0, 31),
        ),
        (
            "silk_ADD_LSHIFT32",
            38,
            |x, y, s| w(s::silk_add_lshift32(x, y, s)),
            (0, 31),
        ),
        (
            "silk_ADD_RSHIFT",
            40,
            |x, y, s| w(s::silk_add_rshift(x, y, s)),
            (0, 31),
        ),
        (
            "silk_ADD_RSHIFT32",
            41,
            |x, y, s| w(s::silk_add_rshift32(x, y, s)),
            (0, 31),
        ),
        (
            "silk_SUB_LSHIFT32",
            43,
            |x, y, s| w(s::silk_sub_lshift32(x, y, s)),
            (0, 31),
        ),
        (
            "silk_SUB_RSHIFT32",
            44,
            |x, y, s| w(s::silk_sub_rshift32(x, y, s)),
            (0, 31),
        ),
        (
            "silk_RSHIFT_ROUND",
            45,
            |x, _, s| w(s::silk_rshift_round(x, s)),
            (1, 31),
        ),
    ]
}

#[test]
fn silk_macros_match_oracle() {
    let _g = lock();
    let mut rng = Rng::new(0x5d8b);
    let mut reported = 0usize;
    for (name, op, f) in silk_binary() {
        for i in 0..N {
            let x = operand(&mut rng, i as u32);
            let y = operand(&mut rng, (i / 6) as u32);
            let r = rust_run(|| f(x, y));
            let cc = c_run(|| c::silk(op, i64::from(x), i64::from(y), 0));
            reported += cc.1.len();
            check(&format!("{name}({x}, {y})"), r, cc, false);
        }
    }
    for (name, op, f) in silk_ternary() {
        for i in 0..N {
            let x = operand(&mut rng, i as u32);
            let y = operand(&mut rng, (i / 6) as u32);
            let z = operand(&mut rng, (i / 36) as u32);
            let r = rust_run(|| f(x, y, z));
            let cc = c_run(|| c::silk(op, i64::from(x), i64::from(y), i64::from(z)));
            reported += cc.1.len();
            check(&format!("{name}({x}, {y}, {z})"), r, cc, false);
        }
    }
    for (name, op, f, (lo, hi)) in silk_shifts() {
        for i in 0..N {
            let x = operand(&mut rng, i as u32);
            let y = operand(&mut rng, (i / 6) as u32);
            let sh = rng.range_i32(lo, hi);
            // silk_ADD_LSHIFT / silk_ADD_RSHIFT compute in 16 bits (`(opus_uint16)b << shift`
            // is an `int` shift: keep it below 32).
            let r = rust_run(|| f(x, y, sh));
            // The two-operand shifts take the shift as their second C argument.
            let (cb, cs) = if matches!(op, 30 | 32 | 34 | 45) {
                (i64::from(sh), 0)
            } else {
                (i64::from(y), i64::from(sh))
            };
            let cc = c_run(|| c::silk(op, i64::from(x), cb, cs));
            reported += cc.1.len();
            check(&format!("{name}({x}, {y}, {sh})"), r, cc, false);
        }
    }
    for i in 0..N {
        let x = operand(&mut rng, i as u32);
        let y = operand(&mut rng, (i / 6) as u32);
        let (xl, yl) = (
            i64::from(x) << rng.range_i32(0, 31),
            i64::from(y) << rng.range_i32(0, 31),
        );
        let (xl, yl) = if i % 50 == 0 {
            (
                i64::MAX - i64::from(x.unsigned_abs()),
                i64::MAX / 2 + i64::from(y),
            )
        } else {
            (xl, yl)
        };
        type C64 = (&'static str, i32, i64, i64, fn(i64, i64) -> i64);
        let cases: [C64; 4] = [
            ("silk_ADD64", 2, xl, yl, s::silk_add64),
            ("silk_SUB64", 5, xl, yl, s::silk_sub64),
            ("silk_ADD_SAT64", 8, xl, yl, s::silk_add_sat64),
            ("silk_SUB_SAT64", 11, xl, yl, s::silk_sub_sat64),
        ];
        for (name, op, p, q, f) in cases {
            let r = rust_run(|| f(p, q));
            let cc = c_run(|| c::silk(op, p, q, 0));
            reported += cc.1.len();
            check(&format!("{name}({p}, {q})"), r, cc, false);
        }
        let sh = rng.range_i32(0, 63);
        let r = rust_run(|| s::silk_lshift64(xl >> 16, sh));
        let cc = c_run(|| c::silk(31, xl >> 16, i64::from(sh), 0));
        check(&format!("silk_LSHIFT64({}, {sh})", xl >> 16), r, cc, false);
        let r = rust_run(|| s::silk_rshift64(xl, sh));
        let cc = c_run(|| c::silk(35, xl, i64::from(sh), 0));
        check(&format!("silk_RSHIFT64({xl}, {sh})"), r, cc, false);
        let sh1 = sh.max(1);
        let r = rust_run(|| s::silk_rshift_round64(xl, sh1));
        let cc = c_run(|| c::silk(46, xl, i64::from(sh1), 0));
        check(&format!("silk_RSHIFT_ROUND64({xl}, {sh1})"), r, cc, false);
        let r = rust_run(|| s::silk_abs_int64(if i % 97 == 0 { i64::MIN } else { xl }));
        let cc = c_run(|| c::silk(47, if i % 97 == 0 { i64::MIN } else { xl }, 0, 0));
        reported += cc.1.len();
        check("silk_abs_int64", r, cc, false);
        let r = rust_run(|| w(s::silk_abs_int32(if i % 97 == 0 { i32::MIN } else { x })));
        let cc = c_run(|| c::silk(48, i64::from(if i % 97 == 0 { i32::MIN } else { x }), 0, 0));
        check("silk_abs_int32", r, cc, false);
        for (name, op, f) in [
            (
                "silk_CHECK_FIT8",
                49,
                (|v| w(s::silk_check_fit8(v))) as fn(i64) -> i64,
            ),
            ("silk_CHECK_FIT16", 50, |v| w(s::silk_check_fit16(v))),
            ("silk_CHECK_FIT32", 51, |v| w(s::silk_check_fit32(v))),
        ] {
            let v = if i % 3 == 0 {
                xl
            } else {
                i64::from(x) >> (i % 20)
            };
            let r = rust_run(|| f(v));
            let cc = c_run(|| c::silk(op, v, 0, 0));
            reported += cc.1.len();
            check(&format!("{name}({v})"), r, cc, false);
        }
        // Unsigned forms.
        let (u, v) = (x as u32, y as u32);
        let sh = rng.range_i32(0, 31);
        type CU = (&'static str, i32, fn(u32, u32, i32) -> u32);
        let unsigned: [CU; 6] = [
            ("silk_MUL_uint", 13, |u, v, _| s::silk_mul_uint(u, v)),
            ("silk_MLA_uint", 15, |u, v, _| s::silk_mla_uint(u, v, 3)),
            ("silk_LSHIFT_uint", 33, |u, _, sh| {
                s::silk_lshift_uint(u, sh)
            }),
            ("silk_RSHIFT_uint", 36, |u, _, sh| {
                s::silk_rshift_uint(u, sh)
            }),
            ("silk_ADD_LSHIFT_uint", 39, s::silk_add_lshift_uint),
            ("silk_ADD_RSHIFT_uint", 42, s::silk_add_rshift_uint),
        ];
        for (name, op, f) in unsigned {
            let r = rust_run(|| i64::from(f(u, v, sh)));
            let (cb, cc3) = match op {
                15 => (i64::from(v), 3),
                33 | 36 => (i64::from(sh), 0),
                _ => (i64::from(v), i64::from(sh)),
            };
            let cc = c_run(|| i64::from(c::silk(op, i64::from(u), cb, cc3) as u32));
            reported += cc.1.len();
            check(&format!("{name}({u}, {v}, {sh})"), r, cc, false);
        }
        // Divisions (non-zero divisors: C divides after reporting a zero one).
        let d = if y == 0 { 1 } else { y };
        let d = if x == i32::MIN && d == -1 { 1 } else { d };
        let r = rust_run(|| w(s::silk_div32(x, d)));
        let cc = c_run(|| c::silk(26, i64::from(x), i64::from(d), 0));
        check(&format!("silk_DIV32({x}, {d})"), r, cc, false);
        let r = rust_run(|| w(s::silk_div32_16(x, d)));
        let cc = c_run(|| c::silk(27, i64::from(x), i64::from(d), 0));
        reported += cc.1.len();
        check(&format!("silk_DIV32_16({x}, {d})"), r, cc, false);
        // 8/16-bit shifts.
        let sh = rng.range_i32(0, 15) as u32;
        let r = rust_run(|| w(s::silk_lshift16(x as i16, sh)));
        let cc = c_run(|| c::silk(29, i64::from(x as i16), i64::from(sh), 0));
        check(&format!("silk_LSHIFT16({x}, {sh})"), r, cc, false);
        let r = rust_run(|| w(s::silk_lshift8(x as i8, sh)));
        let cc = c_run(|| c::silk(28, i64::from(x as i8), i64::from(sh), 0));
        reported += cc.1.len();
        check(&format!("silk_LSHIFT8({x}, {sh})"), r, cc, false);
    }
    assert!(reported > 10_000, "only {reported} SILK diagnostics");
}

/// The diagnostic of each check, as C prints it, on a handful of known inputs (the location
/// is the Rust call site).
#[test]
fn diagnostics_text() {
    let _g = lock();
    let (v, msgs, mips) = rust_run(|| a::add32(i32::MAX, 1));
    assert_eq!(v, i32::MIN);
    assert_eq!(msgs.len(), 1);
    assert!(
        msgs[0].starts_with("ADD32: output is not int: -2147483648 in ")
            && msgs[0].contains("fixed_debug.rs: line ")
            && msgs[0].ends_with('\n'),
        "{msgs:?}"
    );
    assert_eq!(mips, 2);
    let (v, msgs, _) = rust_run(|| a::mult16_16(40000, 2));
    assert_eq!(v, 80000); // no truncation of the operands, unlike the release macro
    assert!(msgs[0].starts_with("MULT16_16: inputs are not short: 40000 2 in "));
    let (v, msgs, mips) = rust_run(|| a::div32_16(5, 0));
    assert_eq!(v, 0);
    assert!(msgs[0].starts_with("DIV32_16: divide by zero: 5/0 in "));
    assert_eq!(mips, 0);
    let (v, msgs, _) = rust_run(|| a::neg32(i32::MIN + 1));
    assert_eq!(v, i32::MAX);
    assert!(msgs.is_empty());
    let (_, msgs, _) = rust_run(|| a::shr32(-5, 1));
    assert!(msgs.is_empty());
    // An out-of-range shift is reported, then shifts (undefined behaviour in C: an
    // overflow-checked Rust build panics).
    let (_, msgs, _) = rust_run(|| std::panic::catch_unwind(|| s::silk_rshift32(-8, 32)));
    assert!(msgs[0].starts_with("silk_RSHITF32(-8, 32) in "), "{msgs:?}");
    let (v, msgs, _) = rust_run(|| s::silk_add_lshift(1, 0x7fff, 2));
    assert_eq!(v, i32::from(1 + (0x7fffu32 << 2) as u16 as i16));
    assert!(msgs[0].starts_with("silk_ADD_LSHIFT(1, 32767, 2) in "));
    // Counted per thread, reset on demand.
    fd::reset_diagnostics_count();
    let (_, msgs, _) = rust_run(|| s::silk_sub32(i32::MIN, 1));
    assert_eq!(msgs.len(), 1);
    assert_eq!(fd::diagnostics_count(), 1);
}

/// `celt_mips` and the diagnostics of whole streams: the Rust and the C encoder/decoder in
/// lockstep on normal and extreme signals (full-scale square waves and noise, DC), every
/// application, both channel counts, several rates and bitrates.
#[test]
fn streams_match_oracle() {
    let _g = lock();
    let mut rng = Rng::new(0x57e4);
    let mut total_c = 0usize;
    for case in 0..32 {
        let fs = [48000, 16000, 24000, 8000, 12000][case % 5];
        let ch = 1 + (case / 5) % 2;
        let app = [2048, 2049, 2051][case % 3];
        let bitrate = [12000, 32000, 64000, 128_000][case % 4];
        // The last cases: stereo CELT encoders with `OPUS_SET_LFE(1)` on full-scale noise,
        // where libopus overflows `opus_val32` (see docs/FIXED_POINT.md: the checking macros
        // report it), and decoders at the maximum gain.
        let extreme = case >= 24;
        let (fs, ch, app) = if extreme {
            (48000, 2, 2051)
        } else {
            (fs, ch, app)
        };
        let n = fs as usize / 50;
        let sig: Vec<f32> = match case % 4 {
            _ if extreme => signals::noise(n * 10, ch, 1.0, rng.next_u64()),
            0 => signals::speech_like(n * 10, ch, fs as u32, rng.next_u64()),
            1 => signals::music_like(n * 10, ch, fs as u32, rng.next_u64()),
            2 => (0..n * 10 * ch)
                .map(|i| if (i / (7 * ch)) % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
            _ => signals::noise(n * 10, ch, 1.0, rng.next_u64()),
        };
        let pcm = signals::to_i16(&sig);
        let mut re = opusorus::Encoder::new_raw(fs, ch as i32, app).unwrap();
        let mut ce = capi::Encoder::new(fs, ch as i32, app).unwrap();
        let mut rd = opusorus::Decoder::new(fs, ch as i32).unwrap();
        let mut cd = capi::Decoder::new(fs, ch as i32).unwrap();
        let mut ctls = vec![(4002, bitrate), (4010, (case % 11) as i32)];
        if extreme {
            ctls.push((10024, 1)); // OPUS_SET_LFE
        }
        for (req, v) in ctls {
            re.ctl_set(req, v).unwrap();
            ce.ctl_set(req, v).unwrap();
        }
        if extreme {
            let gain = [32767, -32768][case % 2];
            rd.set_gain(gain).unwrap();
            cd.ctl_set(4034, gain).unwrap(); // OPUS_SET_GAIN
        }
        for f in 0..10 {
            let x = &pcm[f * n * ch..(f + 1) * n * ch];
            let mut out_r = vec![0u8; 1500];
            let mut out_c = vec![0u8; 1500];
            let (lr, rmsgs, _) = rust_run(|| re.encode(x, n, &mut out_r).unwrap());
            let (lc, cmsgs, _) = c_run(|| ce.encode(x, n, &mut out_c).unwrap());
            let what = format!("case {case} fs {fs} ch {ch} app {app} frame {f}");
            assert_eq!(out_r[..lr], out_c[..lc], "{what}: packet");
            same_diagnostics(&format!("{what}: encoder"), &rmsgs, &cmsgs);
            total_c += cmsgs.len();
            let mut dr = vec![0i16; n * ch];
            let mut dc = vec![0i16; n * ch];
            let (_, rmsgs, _) = rust_run(|| rd.decode(Some(&out_r[..lr]), &mut dr, n, false));
            let (_, cmsgs, _) = c_run(|| cd.decode(Some(&out_c[..lc]), &mut dc, n, false));
            assert_eq!(dr, dc, "{what}: decoded");
            same_diagnostics(&format!("{what}: decoder"), &rmsgs, &cmsgs);
            total_c += cmsgs.len();
        }
    }
    // The extreme cases make libopus report.
    assert!(total_c > 0, "no diagnostics in the extreme streams");
    eprintln!("stream diagnostics (C and Rust): {total_c}");
}
