//! Oracle bindings for the foundation unit: range coder and float mathops.
//!
//! The range coder and `isqrt32` / `float2int*` bindings are available in every oracle build;
//! the float mathops ones only in the float build (fixed-point: [`crate::fixed_foundation`]).

use core::ffi::c_int;

unsafe extern "C" {
    fn oracle_ec_encode_ops(
        ops: *const u32,
        nops: c_int,
        buf: *mut u8,
        size: c_int,
        tells: *mut c_int,
        rng_out: *mut u32,
        shrink_to: c_int,
    ) -> c_int;
    fn oracle_ec_decode_ops(
        ops: *const u32,
        nops: c_int,
        buf: *mut u8,
        size: c_int,
        out: *mut u32,
        tells: *mut c_int,
        rng_out: *mut u32,
    ) -> c_int;
}

#[cfg(not(feature = "fixed-point"))]
unsafe extern "C" {
    fn oracle_celt_log2(x: f32) -> f32;
    fn oracle_celt_exp2(x: f32) -> f32;
    fn oracle_celt_cos_norm(x: f32) -> f32;
    fn oracle_celt_cos_norm2(x: f32) -> f32;
    fn oracle_celt_sin(x: f32) -> f32;
    fn oracle_fast_atan2f(y: f32, x: f32) -> f32;
    fn oracle_celt_atan2p_norm(y: f32, x: f32) -> f32;
    fn oracle_celt_sqrt(x: f32) -> f32;
    fn oracle_celt_rsqrt(x: f32) -> f32;
}

unsafe extern "C" {
    fn oracle_isqrt32(x: u32) -> u32;
    fn oracle_float2int(x: f32) -> c_int;
    fn oracle_float2int16(x: f32) -> i16;
}

/// Fixed ICDF table used by the `icdf` ops (must match `csrc/foundation.c`).
pub const ORACLE_ICDF: [u8; 8] = [250, 200, 150, 100, 60, 30, 10, 0];

/// Result of an encoder op sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncOut {
    /// Encoded buffer (full storage size).
    pub buf: Vec<u8>,
    /// `(ec_tell, ec_tell_frac)` after every op.
    pub tells: Vec<(i32, u32)>,
    /// Final `rng`.
    pub rng: u32,
    /// Error flag.
    pub error: i32,
}

/// Runs encoder ops `[kind, a, b, c]` in C (see `csrc/foundation.c` for kinds).
pub fn ec_encode_ops(ops: &[[u32; 4]], size: usize, shrink_to: usize) -> EncOut {
    let mut buf = vec![0u8; size];
    let mut tells = vec![0 as c_int; ops.len() * 2];
    let mut rng = 0u32;
    // SAFETY: all buffers are sized as the shim expects; ops is contiguous [u32;4] array.
    let error = unsafe {
        oracle_ec_encode_ops(
            ops.as_ptr().cast(),
            ops.len() as c_int,
            buf.as_mut_ptr(),
            size as c_int,
            tells.as_mut_ptr(),
            &mut rng,
            shrink_to as c_int,
        )
    };
    let tells = tells.chunks(2).map(|t| (t[0], t[1] as u32)).collect();
    EncOut {
        buf,
        tells,
        rng,
        error,
    }
}

/// Result of a decoder op sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecOut {
    /// Decoded values.
    pub values: Vec<u32>,
    /// `(ec_tell, ec_tell_frac)` after every op.
    pub tells: Vec<(i32, u32)>,
    /// Final `rng`.
    pub rng: u32,
    /// Error flag.
    pub error: i32,
}

/// Runs decoder ops `[kind, param]` in C.
pub fn ec_decode_ops(ops: &[[u32; 2]], data: &[u8]) -> DecOut {
    let mut buf = data.to_vec();
    let mut values = vec![0u32; ops.len()];
    let mut tells = vec![0 as c_int; ops.len() * 2];
    let mut rng = 0u32;
    // SAFETY: buffers sized as expected; the decoder only reads `buf`.
    let error = unsafe {
        oracle_ec_decode_ops(
            ops.as_ptr().cast(),
            ops.len() as c_int,
            buf.as_mut_ptr(),
            buf.len() as c_int,
            values.as_mut_ptr(),
            tells.as_mut_ptr(),
            &mut rng,
        )
    };
    let tells = tells.chunks(2).map(|t| (t[0], t[1] as u32)).collect();
    DecOut {
        values,
        tells,
        rng,
        error,
    }
}

#[cfg(not(feature = "fixed-point"))]
macro_rules! f1 {
    ($($name:ident => $c:ident),+ $(,)?) => {$(
        /// C oracle for the same-named libopus function/macro.
        #[must_use]
        pub fn $name(x: f32) -> f32 {
            // SAFETY: pure function of its argument.
            unsafe { $c(x) }
        }
    )+};
}
#[cfg(not(feature = "fixed-point"))]
f1!(
    celt_log2 => oracle_celt_log2,
    celt_exp2 => oracle_celt_exp2,
    celt_cos_norm => oracle_celt_cos_norm,
    celt_cos_norm2 => oracle_celt_cos_norm2,
    celt_sin => oracle_celt_sin,
    celt_sqrt => oracle_celt_sqrt,
    celt_rsqrt => oracle_celt_rsqrt,
);

/// C `fast_atan2f`.
#[cfg(not(feature = "fixed-point"))]
#[must_use]
pub fn fast_atan2f(y: f32, x: f32) -> f32 {
    // SAFETY: pure function.
    unsafe { oracle_fast_atan2f(y, x) }
}
/// C `celt_atan2p_norm`.
#[cfg(not(feature = "fixed-point"))]
#[must_use]
pub fn celt_atan2p_norm(y: f32, x: f32) -> f32 {
    // SAFETY: pure function.
    unsafe { oracle_celt_atan2p_norm(y, x) }
}
/// C `isqrt32`.
#[must_use]
pub fn isqrt32(x: u32) -> u32 {
    // SAFETY: pure function.
    unsafe { oracle_isqrt32(x) }
}
/// C `float2int`.
#[must_use]
pub fn float2int(x: f32) -> i32 {
    // SAFETY: pure function.
    unsafe { oracle_float2int(x) }
}
/// C `FLOAT2INT16`.
#[must_use]
pub fn float2int16(x: f32) -> i16 {
    // SAFETY: pure function.
    unsafe { oracle_float2int16(x) }
}
