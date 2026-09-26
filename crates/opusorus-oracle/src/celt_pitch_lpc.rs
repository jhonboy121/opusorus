//! Oracle bindings for unit `celt_pitch_lpc`: celt/pitch.c and celt/celt_lpc.c, in the float
//! and in the fixed-point oracle (`csrc/celt_pitch_lpc.c` is `// oracle-build: any`; the
//! fixed-point oracle adds `csrc/celt_pitch_lpc_int32.c`).
//!
//! The libopus types are mirrored by [`Val16`], [`Val32`], [`Sig`] and [`Coef`] (float in the
//! float build, integers in the fixed-point build). Every wrapper checks the slice lengths the C
//! function will touch before calling it, so the wrappers are safe for any input.

use core::ffi::c_int;

/// `opus_val16`.
#[cfg(not(feature = "fixed-point"))]
pub type Val16 = f32;
/// `opus_val32`.
#[cfg(not(feature = "fixed-point"))]
pub type Val32 = f32;
/// `celt_sig`.
#[cfg(not(feature = "fixed-point"))]
pub type Sig = f32;
/// `celt_coef`.
#[cfg(not(feature = "fixed-point"))]
pub type Coef = f32;
/// `opus_val16`.
#[cfg(feature = "fixed-point")]
pub type Val16 = i16;
/// `opus_val32`.
#[cfg(feature = "fixed-point")]
pub type Val32 = i32;
/// `celt_sig`.
#[cfg(feature = "fixed-point")]
pub type Sig = i32;
/// `celt_coef` (Q31 with QEXT).
#[cfg(all(feature = "fixed-point", feature = "qext"))]
pub type Coef = i32;
/// `celt_coef` (Q15 without QEXT).
#[cfg(all(feature = "fixed-point", not(feature = "qext")))]
pub type Coef = i16;

unsafe extern "C" {
    fn oracle_xcorr_kernel(x: *const Val16, y: *const Val16, sum: *mut Val32, len: c_int);
    fn oracle_dual_inner_prod(
        x: *const Val16,
        y01: *const Val16,
        y02: *const Val16,
        n: c_int,
        xy1: *mut Val32,
        xy2: *mut Val32,
    );
    fn oracle_celt_inner_prod(x: *const Val16, y: *const Val16, n: c_int) -> Val32;
    fn oracle_celt_pitch_xcorr(
        x: *const Val16,
        y: *const Val16,
        xcorr: *mut Val32,
        len: c_int,
        max_pitch: c_int,
    ) -> Val32;
    fn oracle_find_best_pitch(
        xcorr: *const Val32,
        y: *const Val16,
        len: c_int,
        max_pitch: c_int,
        best_pitch: *mut c_int,
        yshift: c_int,
        maxcorr: Val32,
    );
    fn oracle_celt_fir5(x: *mut Val16, num: *const Val16, n: c_int);
    fn oracle_pitch_downsample(
        x0: *const Sig,
        x1: *const Sig,
        x_lp: *mut Val16,
        len: c_int,
        c: c_int,
        factor: c_int,
    );
    fn oracle_pitch_search(
        x_lp: *const Val16,
        y: *const Val16,
        len: c_int,
        max_pitch: c_int,
    ) -> c_int;
    fn oracle_compute_pitch_gain(xy: Val32, xx: Val32, yy: Val32) -> Val16;
    fn oracle_remove_doubling(
        x: *const Val16,
        maxperiod: c_int,
        minperiod: c_int,
        n: c_int,
        t0: *mut c_int,
        prev_period: c_int,
        prev_gain: Val16,
    ) -> Val16;
    fn oracle_celt_lpc(lpc: *mut Val16, ac: *const Val32, p: c_int);
    #[cfg(feature = "fixed-point")]
    fn oracle_celt_lpc_int32(lpc: *mut Val16, ac: *const Val32, p: c_int);
    fn oracle_celt_fir(x: *const Val16, num: *const Val16, y: *mut Val16, n: c_int, ord: c_int);
    fn oracle_celt_iir(
        x: *const Val32,
        den: *const Val16,
        y: *mut Val32,
        n: c_int,
        ord: c_int,
        mem: *mut Val16,
        in_place: c_int,
    );
    fn oracle_celt_autocorr(
        x: *const Val16,
        ac: *mut Val32,
        window: *const Coef,
        overlap: c_int,
        lag: c_int,
        n: c_int,
    ) -> c_int;
}

/// C `xcorr_kernel_c`: accumulates into `sum`. `x` needs `len`, `y` needs `len + 3` elements.
pub fn xcorr_kernel(x: &[Val16], y: &[Val16], sum: &mut [Val32; 4], len: usize) {
    assert!(len >= 3 && x.len() >= len && y.len() >= len + 3);
    // SAFETY: lengths checked above; the kernel reads x[..len], y[..len+3], writes sum[..4].
    unsafe { oracle_xcorr_kernel(x.as_ptr(), y.as_ptr(), sum.as_mut_ptr(), len as c_int) }
}

/// C `dual_inner_prod_c`, returns `(xy1, xy2)`.
#[must_use]
pub fn dual_inner_prod(x: &[Val16], y01: &[Val16], y02: &[Val16], n: usize) -> (Val32, Val32) {
    assert!(x.len() >= n && y01.len() >= n && y02.len() >= n);
    let (mut a, mut b) = (Val32::default(), Val32::default());
    // SAFETY: lengths checked above; outputs are valid locals.
    unsafe {
        oracle_dual_inner_prod(
            x.as_ptr(),
            y01.as_ptr(),
            y02.as_ptr(),
            n as c_int,
            &mut a,
            &mut b,
        );
    }
    (a, b)
}

/// C `celt_inner_prod_c`.
#[must_use]
pub fn celt_inner_prod(x: &[Val16], y: &[Val16], n: usize) -> Val32 {
    assert!(x.len() >= n && y.len() >= n);
    // SAFETY: lengths checked above.
    unsafe { oracle_celt_inner_prod(x.as_ptr(), y.as_ptr(), n as c_int) }
}

/// Number of `y` samples `celt_pitch_xcorr_c` may read.
const fn xcorr_y_need(len: usize, max_pitch: usize) -> usize {
    // Unrolled groups read y[i..i+len+3] for i <= max_pitch-4; the tail reads y[i..i+len].
    let unrolled = if max_pitch >= 4 {
        (max_pitch & !3) - 4 + len + 3
    } else {
        0
    };
    let tail = len + max_pitch - 1;
    if unrolled > tail { unrolled } else { tail }
}

/// C `celt_pitch_xcorr_c`. Returns `maxcorr` in the fixed-point build (0 in the float build,
/// where C returns nothing).
pub fn celt_pitch_xcorr(
    x: &[Val16],
    y: &[Val16],
    xcorr: &mut [Val32],
    len: usize,
    max_pitch: usize,
) -> Val32 {
    assert!(max_pitch > 0 && (max_pitch < 4 || len >= 3));
    assert!(x.len() >= len && y.len() >= xcorr_y_need(len, max_pitch) && xcorr.len() >= max_pitch);
    // SAFETY: lengths checked above.
    unsafe {
        oracle_celt_pitch_xcorr(
            x.as_ptr(),
            y.as_ptr(),
            xcorr.as_mut_ptr(),
            len as c_int,
            max_pitch as c_int,
        )
    }
}

/// C static `find_best_pitch` (the fixed-point build takes `yshift` and `maxcorr`).
pub fn find_best_pitch(
    xcorr: &[Val32],
    y: &[Val16],
    len: usize,
    max_pitch: usize,
    best_pitch: &mut [i32; 2],
    #[cfg(feature = "fixed-point")] yshift: i32,
    #[cfg(feature = "fixed-point")] maxcorr: Val32,
) {
    #[cfg(not(feature = "fixed-point"))]
    let (yshift, maxcorr) = (0, 0.0);
    assert!(xcorr.len() >= max_pitch && y.len() >= len + max_pitch);
    // SAFETY: lengths checked above; C does not modify xcorr/y despite non-const pointers.
    unsafe {
        oracle_find_best_pitch(
            xcorr.as_ptr(),
            y.as_ptr(),
            len as c_int,
            max_pitch as c_int,
            best_pitch.as_mut_ptr(),
            yshift,
            maxcorr,
        );
    }
}

/// C static `celt_fir5` (in place on `x[..n]`).
pub fn celt_fir5(x: &mut [Val16], num: &[Val16; 5], n: usize) {
    assert!(x.len() >= n);
    // SAFETY: length checked above.
    unsafe { oracle_celt_fir5(x.as_mut_ptr(), num.as_ptr(), n as c_int) }
}

/// C `pitch_downsample`. `x[0]` (and `x[1]` when `c == 2`) must hold
/// `(len-1)*factor + factor/2 + 1` samples.
pub fn pitch_downsample(x: &[&[Sig]], x_lp: &mut [Val16], len: usize, c: usize, factor: usize) {
    assert!(len >= 1 && factor >= 1 && x_lp.len() >= len);
    let need = (len - 1) * factor + factor / 2 + 1;
    assert!(x[0].len() >= need);
    let x1 = if c == 2 {
        assert!(x[1].len() >= need);
        x[1].as_ptr()
    } else {
        core::ptr::null()
    };
    // SAFETY: lengths checked above; x1 is only read when c == 2.
    unsafe {
        oracle_pitch_downsample(
            x[0].as_ptr(),
            x1,
            x_lp.as_mut_ptr(),
            len as c_int,
            c as c_int,
            factor as c_int,
        );
    }
}

/// C `pitch_search`, returns `*pitch`.
#[must_use]
pub fn pitch_search(x_lp: &[Val16], y: &[Val16], len: usize, max_pitch: usize) -> i32 {
    assert!(len > 0 && max_pitch > 0);
    assert!(x_lp.len() >= len >> 1 && y.len() >= (len >> 1) + (max_pitch >> 1));
    assert!(y.len() >= 2 * ((len + max_pitch) >> 2));
    assert!(len >> 2 >= 3 || max_pitch >> 2 < 4);
    // SAFETY: lengths checked above (the coarse search reads the 4x-decimated copies).
    unsafe { oracle_pitch_search(x_lp.as_ptr(), y.as_ptr(), len as c_int, max_pitch as c_int) }
}

/// C static `compute_pitch_gain`.
#[must_use]
pub fn compute_pitch_gain(xy: Val32, xx: Val32, yy: Val32) -> Val16 {
    // SAFETY: pure function of its arguments.
    unsafe { oracle_compute_pitch_gain(xy, xx, yy) }
}

/// C `remove_doubling`; `t0` is `*T0_` (in/out). `x` must hold `maxperiod/2 + n/2` samples.
#[must_use]
pub fn remove_doubling(
    x: &[Val16],
    maxperiod: i32,
    minperiod: i32,
    n: i32,
    t0: &mut i32,
    prev_period: i32,
    prev_gain: Val16,
) -> Val16 {
    assert!(maxperiod >= 2 && n >= 2 && *t0 >= 2);
    assert!(x.len() >= (maxperiod / 2 + n / 2) as usize);
    // SAFETY: length checked above; remove_doubling only reads x within [0, maxperiod/2+n/2).
    unsafe {
        oracle_remove_doubling(
            x.as_ptr(),
            maxperiod,
            minperiod,
            n,
            t0,
            prev_period,
            prev_gain,
        )
    }
}

/// C `_celt_lpc`: writes `lpc[..p]` from `ac[..=p]`.
pub fn celt_lpc(lpc: &mut [Val16], ac: &[Val32], p: usize) {
    assert!(lpc.len() >= p && ac.len() > p);
    // The fixed-point build computes in a local `opus_val32 lpc[CELT_LPC_ORDER]`.
    #[cfg(feature = "fixed-point")]
    assert!(p <= 24);
    // SAFETY: lengths checked above.
    unsafe { oracle_celt_lpc(lpc.as_mut_ptr(), ac.as_ptr(), p as c_int) }
}

/// C `_celt_lpc` compiled with `OPUS_FAST_INT64 == 0` (the 32-bit multiply forms).
#[cfg(feature = "fixed-point")]
pub fn celt_lpc_int32(lpc: &mut [Val16], ac: &[Val32], p: usize) {
    assert!(p <= 24 && lpc.len() >= p && ac.len() > p);
    // SAFETY: lengths checked above (p <= CELT_LPC_ORDER, the size of C's local array).
    unsafe { oracle_celt_lpc_int32(lpc.as_mut_ptr(), ac.as_ptr(), p as c_int) }
}

/// C `celt_fir_c`; `x` holds `ord` history samples then `n` inputs (C's `x` = `&x[ord]`).
pub fn celt_fir(x: &[Val16], num: &[Val16], y: &mut [Val16], n: usize, ord: usize) {
    assert!(x.len() >= n + ord && num.len() >= ord && y.len() >= n);
    assert!(n < 4 || ord >= 3);
    // SAFETY: lengths checked above; y does not alias x.
    unsafe {
        oracle_celt_fir(
            x.as_ptr(),
            num.as_ptr(),
            y.as_mut_ptr(),
            n as c_int,
            ord as c_int,
        )
    }
}

/// C `celt_iir` (out of place).
pub fn celt_iir(
    x: &[Val32],
    den: &[Val16],
    y: &mut [Val32],
    n: usize,
    ord: usize,
    mem: &mut [Val16],
) {
    assert!(ord.is_multiple_of(4) && ord >= 4 && n >= ord);
    assert!(x.len() >= n && den.len() >= ord && y.len() >= n && mem.len() >= ord);
    // SAFETY: lengths checked above.
    unsafe {
        oracle_celt_iir(
            x.as_ptr(),
            den.as_ptr(),
            y.as_mut_ptr(),
            n as c_int,
            ord as c_int,
            mem.as_mut_ptr(),
            0,
        );
    }
}

/// C `celt_iir` called in place (`_x == _y == buf`).
pub fn celt_iir_inplace(buf: &mut [Val32], den: &[Val16], n: usize, ord: usize, mem: &mut [Val16]) {
    assert!(ord.is_multiple_of(4) && ord >= 4 && n >= ord);
    assert!(buf.len() >= n && den.len() >= ord && mem.len() >= ord);
    // SAFETY: lengths checked above; the C code supports x == y.
    unsafe {
        oracle_celt_iir(
            buf.as_ptr(),
            den.as_ptr(),
            buf.as_mut_ptr(),
            n as c_int,
            ord as c_int,
            mem.as_mut_ptr(),
            1,
        );
    }
}

/// C `_celt_autocorr`: writes `ac[..=lag]`, returns the shift. `window` may be empty when
/// `overlap == 0` (C `NULL`).
pub fn celt_autocorr(
    x: &[Val16],
    ac: &mut [Val32],
    window: &[Coef],
    overlap: usize,
    lag: usize,
    n: usize,
) -> i32 {
    assert!(n > lag && x.len() >= n && ac.len() > lag && window.len() >= overlap);
    assert!(overlap <= n);
    assert!(lag + 1 < 4 || n - lag >= 3);
    // SAFETY: lengths checked above.
    unsafe {
        oracle_celt_autocorr(
            x.as_ptr(),
            ac.as_mut_ptr(),
            window.as_ptr(),
            overlap as c_int,
            lag as c_int,
            n as c_int,
        )
    }
}
