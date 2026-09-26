//! `opus.h` DRED (Deep REDundancy) decoder API.
//!
//! Semantics of a libopus build without `ENABLE_DRED`: an `OpusDREDDecoder` can be created
//! (it holds no model), `opus_dred_decoder_ctl` answers `OPUS_UNIMPLEMENTED`,
//! `opus_dred_get_size` is 0, `opus_dred_alloc` fails with `OPUS_UNIMPLEMENTED`, and parsing,
//! processing and DRED decoding return `OPUS_UNIMPLEMENTED`.
//!
//! TODO(dnn_integration): with the `dred` feature these should load the RDOVAE decoder
//! (`OPUS_SET_DNN_BLOB`), parse DRED extensions and feed the deep PLC. The Rust decoder does
//! not expose that API yet (`opus_decode_native` DRED hook, `OpusDRED`/`OpusDREDDecoder`
//! types), so the `dred` build keeps the no-DRED behaviour for now.

use core::ffi::c_int;
use core::ptr;

use crate::handle::{self, Kind};
use crate::types::{OpusDRED, OpusDREDDecoder, OpusDecoder};
use crate::util::{OPUS_ALLOC_FAIL, OPUS_OK, OPUS_UNIMPLEMENTED, guard, guard_any, set_error};

/// `opus_dred_decoder_get_size`.
#[unsafe(no_mangle)]
pub const extern "C" fn opus_dred_decoder_get_size() -> c_int {
    handle::HEADER_SIZE as c_int
}

/// `opus_dred_decoder_init`.
///
/// # Safety
/// `dec` is valid for writes of `opus_dred_decoder_get_size()` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_dred_decoder_init(dec: *mut OpusDREDDecoder) -> c_int {
    guard(|| {
        if dec.is_null() {
            return Err(crate::util::OPUS_BAD_ARG);
        }
        // SAFETY: caller contract.
        unsafe { handle::write_plain_header(dec.cast(), Kind::DredDecoder) };
        Ok(OPUS_OK)
    })
}

/// `opus_dred_decoder_create`.
///
/// # Safety
/// `error` is NULL or valid for one `int` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_dred_decoder_create(error: *mut c_int) -> *mut OpusDREDDecoder {
    guard_any(ptr::null_mut(), || {
        let block = handle::c_alloc(handle::HEADER_SIZE);
        if block.is_null() {
            // SAFETY: caller contract.
            unsafe { set_error(error, OPUS_ALLOC_FAIL) };
            return ptr::null_mut();
        }
        // SAFETY: fresh allocation of `HEADER_SIZE` bytes.
        unsafe { handle::write_plain_header(block, Kind::DredDecoder) };
        // SAFETY: caller contract.
        unsafe { set_error(error, OPUS_OK) };
        block.cast()
    })
}

/// `opus_dred_decoder_destroy`.
///
/// # Safety
/// `dec` is NULL or a DRED decoder on the C heap, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_dred_decoder_destroy(dec: *mut OpusDREDDecoder) {
    // SAFETY: caller contract (no registry object behind a DRED decoder).
    guard_any((), || unsafe { handle::c_free(dec.cast()) });
}

/// `opus_dred_get_size`: 0 without DRED support.
#[unsafe(no_mangle)]
pub const extern "C" fn opus_dred_get_size() -> c_int {
    0
}

/// `opus_dred_alloc`: fails with `OPUS_UNIMPLEMENTED` without DRED support.
///
/// # Safety
/// `error` is NULL or valid for one `int` write.
#[unsafe(no_mangle)]
pub const unsafe extern "C" fn opus_dred_alloc(error: *mut c_int) -> *mut OpusDRED {
    // SAFETY: caller contract.
    unsafe { set_error(error, OPUS_UNIMPLEMENTED) };
    ptr::null_mut()
}

/// `opus_dred_free`: nothing to free without DRED support.
#[unsafe(no_mangle)]
pub const extern "C" fn opus_dred_free(_dec: *mut OpusDRED) {}

/// `opus_dred_parse`: `OPUS_UNIMPLEMENTED` without DRED support.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub const extern "C" fn opus_dred_parse(
    _dred_dec: *mut OpusDREDDecoder,
    _dred: *mut OpusDRED,
    _data: *const u8,
    _len: i32,
    _max_dred_samples: i32,
    _sampling_rate: i32,
    _dred_end: *mut c_int,
    _defer_processing: c_int,
) -> c_int {
    OPUS_UNIMPLEMENTED
}

/// `opus_dred_process`: `OPUS_UNIMPLEMENTED` without DRED support.
#[unsafe(no_mangle)]
pub const extern "C" fn opus_dred_process(
    _dred_dec: *mut OpusDREDDecoder,
    _src: *const OpusDRED,
    _dst: *mut OpusDRED,
) -> c_int {
    OPUS_UNIMPLEMENTED
}

/// `opus_decoder_dred_decode`: `OPUS_UNIMPLEMENTED` without DRED support.
#[unsafe(no_mangle)]
pub const extern "C" fn opus_decoder_dred_decode(
    _st: *mut OpusDecoder,
    _dred: *const OpusDRED,
    _dred_offset: i32,
    _pcm: *mut i16,
    _frame_size: i32,
) -> c_int {
    OPUS_UNIMPLEMENTED
}

/// `opus_decoder_dred_decode24`: `OPUS_UNIMPLEMENTED` without DRED support.
#[unsafe(no_mangle)]
pub const extern "C" fn opus_decoder_dred_decode24(
    _st: *mut OpusDecoder,
    _dred: *const OpusDRED,
    _dred_offset: i32,
    _pcm: *mut i32,
    _frame_size: i32,
) -> c_int {
    OPUS_UNIMPLEMENTED
}

/// `opus_decoder_dred_decode_float`: `OPUS_UNIMPLEMENTED` without DRED support.
#[unsafe(no_mangle)]
pub const extern "C" fn opus_decoder_dred_decode_float(
    _st: *mut OpusDecoder,
    _dred: *const OpusDRED,
    _dred_offset: i32,
    _pcm: *mut f32,
    _frame_size: i32,
) -> c_int {
    OPUS_UNIMPLEMENTED
}
