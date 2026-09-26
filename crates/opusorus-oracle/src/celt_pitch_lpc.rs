//! Oracle bindings for unit `celt_pitch_lpc`: celt/pitch.c and celt/celt_lpc.c (float build).
//!
//! Every wrapper checks the slice lengths the C function will touch before calling it, so the
//! wrappers are safe for any input.

use core::ffi::c_int;

unsafe extern "C" {
    fn oracle_xcorr_kernel(x: *const f32, y: *const f32, sum: *mut f32, len: c_int);
    fn oracle_dual_inner_prod(
        x: *const f32,
        y01: *const f32,
        y02: *const f32,
        n: c_int,
        xy1: *mut f32,
        xy2: *mut f32,
    );
    fn oracle_celt_inner_prod(x: *const f32, y: *const f32, n: c_int) -> f32;
    fn oracle_celt_pitch_xcorr(
        x: *const f32,
        y: *const f32,
        xcorr: *mut f32,
        len: c_int,
        max_pitch: c_int,
    );
    fn oracle_find_best_pitch(
        xcorr: *const f32,
        y: *const f32,
        len: c_int,
        max_pitch: c_int,
        best_pitch: *mut c_int,
    );
    fn oracle_celt_fir5(x: *mut f32, num: *const f32, n: c_int);
    fn oracle_pitch_downsample(
        x0: *const f32,
        x1: *const f32,
        x_lp: *mut f32,
        len: c_int,
        c: c_int,
        factor: c_int,
    );
    fn oracle_pitch_search(x_lp: *const f32, y: *const f32, len: c_int, max_pitch: c_int) -> c_int;
    fn oracle_compute_pitch_gain(xy: f32, xx: f32, yy: f32) -> f32;
    fn oracle_remove_doubling(
        x: *const f32,
        maxperiod: c_int,
        minperiod: c_int,
        n: c_int,
        t0: *mut c_int,
        prev_period: c_int,
        prev_gain: f32,
    ) -> f32;
    fn oracle_celt_lpc(lpc: *mut f32, ac: *const f32, p: c_int);
    fn oracle_celt_fir(x: *const f32, num: *const f32, y: *mut f32, n: c_int, ord: c_int);
    fn oracle_celt_iir(
        x: *const f32,
        den: *const f32,
        y: *mut f32,
        n: c_int,
        ord: c_int,
        mem: *mut f32,
        in_place: c_int,
    );
    fn oracle_celt_autocorr(
        x: *const f32,
        ac: *mut f32,
        window: *const f32,
        overlap: c_int,
        lag: c_int,
        n: c_int,
    ) -> c_int;
}

/// C `xcorr_kernel_c`: accumulates into `sum`. `x` needs `len`, `y` needs `len + 3` elements.
pub fn xcorr_kernel(x: &[f32], y: &[f32], sum: &mut [f32; 4], len: usize) {
    assert!(len >= 3 && x.len() >= len && y.len() >= len + 3);
    // SAFETY: lengths checked above; the kernel reads x[..len], y[..len+3], writes sum[..4].
    unsafe { oracle_xcorr_kernel(x.as_ptr(), y.as_ptr(), sum.as_mut_ptr(), len as c_int) }
}

/// C `dual_inner_prod_c`, returns `(xy1, xy2)`.
#[must_use]
pub fn dual_inner_prod(x: &[f32], y01: &[f32], y02: &[f32], n: usize) -> (f32, f32) {
    assert!(x.len() >= n && y01.len() >= n && y02.len() >= n);
    let (mut a, mut b) = (0f32, 0f32);
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
pub fn celt_inner_prod(x: &[f32], y: &[f32], n: usize) -> f32 {
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

/// C `celt_pitch_xcorr_c`.
pub fn celt_pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, max_pitch: usize) {
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
        );
    }
}

/// C static `find_best_pitch` (float build).
pub fn find_best_pitch(
    xcorr: &[f32],
    y: &[f32],
    len: usize,
    max_pitch: usize,
    best_pitch: &mut [i32; 2],
) {
    assert!(xcorr.len() >= max_pitch && y.len() >= len + max_pitch);
    // SAFETY: lengths checked above; C does not modify xcorr/y despite non-const pointers.
    unsafe {
        oracle_find_best_pitch(
            xcorr.as_ptr(),
            y.as_ptr(),
            len as c_int,
            max_pitch as c_int,
            best_pitch.as_mut_ptr(),
        );
    }
}

/// C static `celt_fir5` (in place on `x[..n]`).
pub fn celt_fir5(x: &mut [f32], num: &[f32; 5], n: usize) {
    assert!(x.len() >= n);
    // SAFETY: length checked above.
    unsafe { oracle_celt_fir5(x.as_mut_ptr(), num.as_ptr(), n as c_int) }
}

/// C `pitch_downsample`. `x[0]` (and `x[1]` when `c == 2`) must hold
/// `(len-1)*factor + factor/2 + 1` samples.
pub fn pitch_downsample(x: &[&[f32]], x_lp: &mut [f32], len: usize, c: usize, factor: usize) {
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
pub fn pitch_search(x_lp: &[f32], y: &[f32], len: usize, max_pitch: usize) -> i32 {
    assert!(len > 0 && max_pitch > 0);
    assert!(x_lp.len() >= len >> 1 && y.len() >= (len >> 1) + (max_pitch >> 1));
    assert!(y.len() >= 2 * ((len + max_pitch) >> 2));
    assert!(len >> 2 >= 3 || max_pitch >> 2 < 4);
    // SAFETY: lengths checked above (the coarse search reads the 4x-decimated copies).
    unsafe { oracle_pitch_search(x_lp.as_ptr(), y.as_ptr(), len as c_int, max_pitch as c_int) }
}

/// C static `compute_pitch_gain` (float build).
#[must_use]
pub fn compute_pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    // SAFETY: pure function of its arguments.
    unsafe { oracle_compute_pitch_gain(xy, xx, yy) }
}

/// C `remove_doubling`; `t0` is `*T0_` (in/out). `x` must hold `maxperiod/2 + n/2` samples.
#[must_use]
pub fn remove_doubling(
    x: &[f32],
    maxperiod: i32,
    minperiod: i32,
    n: i32,
    t0: &mut i32,
    prev_period: i32,
    prev_gain: f32,
) -> f32 {
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
pub fn celt_lpc(lpc: &mut [f32], ac: &[f32], p: usize) {
    assert!(lpc.len() >= p && ac.len() > p);
    // SAFETY: lengths checked above.
    unsafe { oracle_celt_lpc(lpc.as_mut_ptr(), ac.as_ptr(), p as c_int) }
}

/// C `celt_fir_c`; `x` holds `ord` history samples then `n` inputs (C's `x` = `&x[ord]`).
pub fn celt_fir(x: &[f32], num: &[f32], y: &mut [f32], n: usize, ord: usize) {
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
pub fn celt_iir(x: &[f32], den: &[f32], y: &mut [f32], n: usize, ord: usize, mem: &mut [f32]) {
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
pub fn celt_iir_inplace(buf: &mut [f32], den: &[f32], n: usize, ord: usize, mem: &mut [f32]) {
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
    x: &[f32],
    ac: &mut [f32],
    window: &[f32],
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
