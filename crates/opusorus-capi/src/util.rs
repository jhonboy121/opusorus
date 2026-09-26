//! FFI plumbing shared by every export: error codes, panic guards, and the conversions from C
//! `(pointer, length)` pairs to Rust slices.

use core::ffi::c_int;
use core::panic::UnwindSafe;
use std::panic::{AssertUnwindSafe, catch_unwind};

use opus::Error;

/// `OPUS_OK`.
pub(crate) const OPUS_OK: c_int = 0;
/// `OPUS_BAD_ARG`.
pub(crate) const OPUS_BAD_ARG: c_int = -1;
/// `OPUS_INTERNAL_ERROR`.
pub(crate) const OPUS_INTERNAL_ERROR: c_int = -3;
/// `OPUS_UNIMPLEMENTED`.
pub(crate) const OPUS_UNIMPLEMENTED: c_int = -5;
/// `OPUS_INVALID_STATE`.
pub(crate) const OPUS_INVALID_STATE: c_int = -6;
/// `OPUS_ALLOC_FAIL`.
pub(crate) const OPUS_ALLOC_FAIL: c_int = -7;

/// Result of an FFI body: `Err` carries the negative `OPUS_*` code returned to C.
pub(crate) type CResult<T> = Result<T, c_int>;

/// Maps an opusorus error to its libopus code.
#[inline]
pub(crate) const fn code(e: Error) -> c_int {
    e.code()
}

/// Maps an opusorus result to a C return value (`value` on success, the error code otherwise).
#[inline]
pub(crate) fn ret_code<T>(r: opus::Result<T>, value: impl FnOnce(T) -> c_int) -> CResult<c_int> {
    r.map(value).map_err(code)
}

/// Runs an FFI body returning an `int`, converting `Err(code)` to `code` and a caught panic to
/// `OPUS_INTERNAL_ERROR`, so no panic ever unwinds into C.
///
/// Release builds use `panic = "abort"` (workspace profile), where a panic aborts the process
/// instead; development and test builds unwind and are caught here.
#[inline]
pub(crate) fn guard(f: impl FnOnce() -> CResult<c_int>) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v) | Err(v)) => v,
        Err(_) => OPUS_INTERNAL_ERROR,
    }
}

/// Like [`guard`] for bodies with another return type: a caught panic yields `on_panic`.
#[inline]
pub(crate) fn guard_or<T>(on_panic: T, f: impl FnOnce() -> T + UnwindSafe) -> T {
    match catch_unwind(f) {
        Ok(v) => v,
        Err(_) => on_panic,
    }
}

/// Like [`guard_or`] for bodies that are not `UnwindSafe` (they only touch state that a panic
/// leaves in a defined, if unspecified, condition: the C caller gets an error either way).
#[inline]
pub(crate) fn guard_any<T>(on_panic: T, f: impl FnOnce() -> T) -> T {
    guard_or(on_panic, AssertUnwindSafe(f))
}

/// Writes `value` through the optional C out-pointer `error` (C: `if (error) *error = value`).
///
/// # Safety
/// `error` is NULL or valid for a write of one `int`.
#[inline]
pub(crate) const unsafe fn set_error(error: *mut c_int, value: c_int) {
    if !error.is_null() {
        // SAFETY: non-NULL and valid for writes per the caller contract.
        unsafe { error.write(value) };
    }
}

/// Element count `a * b` of a C buffer, or `OPUS_BAD_ARG` if either is negative or the byte size
/// of `T` elements would not fit in `isize` (a slice of it could not exist).
#[inline]
pub(crate) fn buf_len<T>(a: c_int, b: c_int) -> CResult<usize> {
    let a = usize::try_from(a).map_err(|_| OPUS_BAD_ARG)?;
    let b = usize::try_from(b).map_err(|_| OPUS_BAD_ARG)?;
    let n = a.checked_mul(b).ok_or(OPUS_BAD_ARG)?;
    let bytes = n.checked_mul(size_of::<T>()).ok_or(OPUS_BAD_ARG)?;
    if bytes > isize::MAX as usize {
        return Err(OPUS_BAD_ARG);
    }
    Ok(n)
}

/// A read-only C buffer of `len` elements. NULL is `OPUS_BAD_ARG` unless `len == 0` (then an
/// empty slice).
///
/// # Safety
/// If non-NULL, `ptr` is valid for reads of `len` elements for `'a`, and not written during `'a`.
#[inline]
pub(crate) const unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> CResult<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(OPUS_BAD_ARG);
    }
    // SAFETY: non-NULL, valid for `len` reads per the caller contract; `len * size_of::<T>()`
    // fits in `isize` (checked by the callers through `buf_len` or bounded C lengths).
    Ok(unsafe { core::slice::from_raw_parts(ptr, len) })
}

/// A writable C buffer of `len` elements. NULL is `OPUS_BAD_ARG` unless `len == 0`.
///
/// # Safety
/// If non-NULL, `ptr` is valid for reads and writes of `len` elements for `'a`, and not
/// accessed through any other pointer during `'a`.
#[inline]
pub(crate) const unsafe fn slice_mut<'a, T>(ptr: *mut T, len: usize) -> CResult<&'a mut [T]> {
    if len == 0 {
        return Ok(&mut []);
    }
    if ptr.is_null() {
        return Err(OPUS_BAD_ARG);
    }
    // SAFETY: as `slice`, plus exclusive access per the caller contract.
    Ok(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
}

/// A C `opus_int32` byte length as a slice length (negative lengths are `OPUS_BAD_ARG`).
#[inline]
pub(crate) fn byte_len(len: i32) -> CResult<usize> {
    usize::try_from(len).map_err(|_| OPUS_BAD_ARG)
}

/// An input packet as the decoders see it (`opus_decode*`): `data == NULL` or `len == 0` is a
/// lost packet (`None`, packet-loss concealment); a negative `len` with a packet is
/// `OPUS_BAD_ARG` (as C returns for every such call).
///
/// # Safety
/// If non-NULL and `len > 0`, `data` is valid for reads of `len` bytes during `'a`.
#[inline]
pub(crate) unsafe fn packet<'a>(data: *const u8, len: i32) -> CResult<Option<&'a [u8]>> {
    if data.is_null() || len == 0 {
        return Ok(None);
    }
    let len = byte_len(len)?;
    // SAFETY: forwarded caller contract.
    unsafe { slice(data, len) }.map(Some)
}

/// Converts a Rust size to a C `int` / `opus_int32`, saturating (sizes beyond 2 GiB cannot be
/// represented by the libopus API).
#[inline]
pub(crate) fn size_to_int(size: usize) -> c_int {
    match c_int::try_from(size) {
        Ok(s) => s,
        // Saturate: the libopus API has no way to express larger sizes.
        Err(_) => c_int::MAX,
    }
}
