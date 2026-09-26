//! Non-variadic halves of the `opus_*_ctl` functions.
//!
//! `csrc/ctl.c` implements the variadic C entry points: it reads the arguments according to the
//! request and calls one of the `opusorus_ctl_*` functions below with the state kind of the
//! API that was called. On the architectures listed in `build.rs`, the libopus symbol names
//! themselves are Rust naked functions that tail-jump to the C code (see [`trampoline!`]),
//! because a Rust `cdylib` only exports symbols defined in Rust.

use core::ffi::{c_int, c_void};

#[cfg(any(feature = "deep-plc", feature = "osce", feature = "dred"))]
use opus::encoder::request::OPUS_SET_DNN_BLOB_REQUEST;
use opus::encoder::request::{
    OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST, OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST,
    OPUS_SET_ENERGY_MASK_REQUEST,
};
use opus::ms_decoder::OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST;

use crate::decoder::decoder;
use crate::encoder::encoder;
use crate::handle::{self, Access, Kind, Object};
use crate::multistream::{ms_decoder, ms_encoder};
use crate::projection::{projection_decoder, projection_encoder};
use crate::util::{
    CResult, OPUS_BAD_ARG, OPUS_INVALID_STATE, OPUS_OK, OPUS_UNIMPLEMENTED, byte_len, code, guard,
    slice, slice_mut,
};

/// The API kind passed by `csrc/ctl.c`.
fn api_kind(kind: c_int) -> CResult<Kind> {
    Kind::from_ctl(kind).ok_or(OPUS_BAD_ARG)
}

/// `opus_*_ctl` for requests taking one `opus_int32` (and `OPUS_RESET_STATE`).
///
/// # Safety
/// `st` is a state of the given kind (or NULL).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opusorus_ctl_set(
    st: *mut c_void,
    kind: c_int,
    request: c_int,
    value: i32,
) -> c_int {
    guard(|| {
        let a = Access::Write;
        let r = match api_kind(kind)? {
            // SAFETY: caller contract; the reference is used for this call only.
            Kind::Encoder => unsafe { encoder(st.cast(), a) }?.ctl_set(request, value),
            // SAFETY: caller contract; the reference is used for this call only.
            Kind::Decoder => unsafe { decoder(st.cast(), a) }?.ctl_set(request, value),
            // SAFETY: caller contract; the reference is used for this call only.
            Kind::MsEncoder => unsafe { ms_encoder(st.cast(), a) }?.ctl_set(request, value),
            // SAFETY: caller contract; the reference is used for this call only.
            Kind::MsDecoder => unsafe { ms_decoder(st.cast(), a) }?.ctl_set(request, value),
            Kind::ProjectionEncoder => {
                // SAFETY: caller contract; the reference is used for this call only.
                unsafe { projection_encoder(st.cast(), a) }?.ctl_set(request, value)
            }
            Kind::ProjectionDecoder => {
                // SAFETY: caller contract; the reference is used for this call only.
                unsafe { projection_decoder(st.cast(), a) }?.ctl_set(request, value)
            }
            #[cfg(feature = "custom-modes")]
            Kind::CustomEncoder => {
                // SAFETY: caller contract; the reference is used for this call only.
                unsafe { crate::custom::custom_encoder(st.cast(), a) }?.ctl_set(request, value)
            }
            #[cfg(feature = "custom-modes")]
            Kind::CustomDecoder => {
                // SAFETY: caller contract; the reference is used for this call only.
                unsafe { crate::custom::custom_decoder(st.cast(), a) }?.ctl_set(request, value)
            }
            #[cfg(feature = "dred")]
            Kind::DredDecoder => {
                // SAFETY: caller contract; the reference is used for this call only.
                unsafe { crate::dred::dred_decoder(st.cast(), a) }?.ctl_set(request, value)
            }
            _ => return Err(OPUS_UNIMPLEMENTED),
        };
        r.map(|()| OPUS_OK).map_err(code)
    })
}

/// Runs a GET request and returns its value.
///
/// # Safety
/// `st` is a state of the given kind (or NULL).
unsafe fn get(st: *mut c_void, kind: c_int, request: c_int) -> CResult<i32> {
    let a = Access::Read;
    let r = match api_kind(kind)? {
        // SAFETY: caller contract; the reference is used for this call only.
        Kind::Encoder => unsafe { encoder(st.cast(), a) }?.ctl_get(request),
        // SAFETY: caller contract; the reference is used for this call only.
        Kind::Decoder => unsafe { decoder(st.cast(), a) }?.ctl_get(request),
        // SAFETY: caller contract; the reference is used for this call only.
        Kind::MsEncoder => unsafe { ms_encoder(st.cast(), a) }?.ctl_get(request),
        // SAFETY: caller contract; the reference is used for this call only.
        Kind::MsDecoder => unsafe { ms_decoder(st.cast(), a) }?.ctl_get(request),
        // SAFETY: caller contract; the reference is used for this call only.
        Kind::ProjectionEncoder => unsafe { projection_encoder(st.cast(), a) }?.ctl_get(request),
        // SAFETY: caller contract; the reference is used for this call only.
        Kind::ProjectionDecoder => unsafe { projection_decoder(st.cast(), a) }?.ctl_get(request),
        #[cfg(feature = "custom-modes")]
        Kind::CustomEncoder => {
            // CELT_GET_AND_CLEAR_ERROR clears state, so custom GETs resolve for writing.
            // SAFETY: caller contract; the reference is used for this call only.
            unsafe { crate::custom::custom_encoder(st.cast(), Access::Write) }?.ctl_get(request)
        }
        #[cfg(feature = "custom-modes")]
        Kind::CustomDecoder => {
            // SAFETY: caller contract; the reference is used for this call only.
            unsafe { crate::custom::custom_decoder(st.cast(), Access::Write) }?.ctl_get(request)
        }
        _ => return Err(OPUS_UNIMPLEMENTED),
    };
    r.map_err(code)
}

/// `opus_*_ctl` for requests writing one `opus_int32`.
///
/// # Safety
/// `st` is a state of the given kind (or NULL); `value` is NULL or valid for one write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opusorus_ctl_get(
    st: *mut c_void,
    kind: c_int,
    request: c_int,
    value: *mut i32,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let v = unsafe { get(st, kind, request) }?;
        if value.is_null() {
            return Err(OPUS_BAD_ARG);
        }
        // SAFETY: non-NULL, caller contract.
        unsafe { value.write(v) };
        Ok(OPUS_OK)
    })
}

/// `opus_*_ctl` for requests writing one `opus_uint32` (`OPUS_GET_FINAL_RANGE`).
///
/// # Safety
/// `st` is a state of the given kind (or NULL); `value` is NULL or valid for one write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opusorus_ctl_get_u32(
    st: *mut c_void,
    kind: c_int,
    request: c_int,
    value: *mut u32,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let v = unsafe { get(st, kind, request) }?;
        if value.is_null() {
            return Err(OPUS_BAD_ARG);
        }
        // SAFETY: non-NULL, caller contract. The getters return the u32 bits as i32.
        unsafe { value.write(v as u32) };
        Ok(OPUS_OK)
    })
}

/// `OPUS_SET_DNN_BLOB` argument check (C: `len < 0 || data == NULL` is `OPUS_BAD_ARG`).
///
/// # Safety
/// `data` is valid for reads of `len` bytes during `'a`.
#[cfg(any(feature = "deep-plc", feature = "osce", feature = "dred"))]
unsafe fn blob<'a>(data: *const c_void, len: i32) -> CResult<&'a [u8]> {
    if data.is_null() || len < 0 {
        return Err(OPUS_BAD_ARG);
    }
    // SAFETY: caller contract.
    unsafe { slice(data.cast::<u8>(), byte_len(len)?) }
}

/// `opus_*_ctl` for requests taking a buffer: `OPUS_SET_DNN_BLOB` (`ptr`, `len`),
/// `OPUS_PROJECTION_GET_DEMIXING_MATRIX` (`ptr`, `len`) and the private
/// `OPUS_SET_ENERGY_MASK` (`ptr` to `21 * channels` floats, `len` unused).
///
/// # Safety
/// `st` is a state of the given kind (or NULL); `ptr` is valid for the access the request
/// implies.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opusorus_ctl_ptr(
    st: *mut c_void,
    kind: c_int,
    request: c_int,
    ptr: *mut c_void,
    len: i32,
) -> c_int {
    guard(|| {
        let kind = api_kind(kind)?;
        match (kind, request) {
            (Kind::Encoder, OPUS_SET_ENERGY_MASK_REQUEST) => {
                // SAFETY: caller contract.
                let enc = unsafe { encoder(st.cast(), Access::Write) }?;
                if ptr.is_null() {
                    enc.set_energy_mask(None);
                } else {
                    let n = 21 * enc.channels() as usize;
                    // SAFETY: caller contract (C reads `21 * channels` values).
                    let mask = unsafe { slice(ptr.cast::<f32>(), n) }?;
                    enc.set_energy_mask(Some(mask));
                }
                Ok(OPUS_OK)
            }
            #[cfg(feature = "dred")]
            (Kind::Encoder, OPUS_SET_DNN_BLOB_REQUEST) => {
                // SAFETY: caller contract.
                let enc = unsafe { encoder(st.cast(), Access::Write) }?;
                // SAFETY: caller contract.
                let data = unsafe { blob(ptr, len) }?;
                enc.set_dnn_blob(data).map(|()| OPUS_OK).map_err(code)
            }
            #[cfg(any(feature = "deep-plc", feature = "osce"))]
            (Kind::Decoder, OPUS_SET_DNN_BLOB_REQUEST) => {
                // SAFETY: caller contract.
                let dec = unsafe { decoder(st.cast(), Access::Write) }?;
                // SAFETY: caller contract.
                let data = unsafe { blob(ptr, len) }?;
                dec.set_dnn_blob(data).map(|()| OPUS_OK).map_err(code)
            }
            (Kind::ProjectionEncoder, OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST) => {
                // SAFETY: caller contract.
                let enc = unsafe { projection_encoder(st.cast(), Access::Read) }?;
                if ptr.is_null() {
                    return Err(OPUS_BAD_ARG);
                }
                let n = byte_len(len)?;
                if n != enc.demixing_matrix_size() {
                    return Err(OPUS_BAD_ARG);
                }
                // SAFETY: caller contract (`len` bytes).
                let out = unsafe { slice_mut(ptr.cast::<u8>(), n) }?;
                enc.write_demixing_matrix(out)
                    .map(|()| OPUS_OK)
                    .map_err(code)
            }
            #[cfg(feature = "dred")]
            (Kind::DredDecoder, OPUS_SET_DNN_BLOB_REQUEST) => {
                // SAFETY: caller contract.
                let dec = unsafe { crate::dred::dred_decoder(st.cast(), Access::Write) }?;
                // SAFETY: caller contract.
                let data = unsafe { blob(ptr, len) }?;
                dec.set_dnn_blob(data).map(|()| OPUS_OK).map_err(code)
            }
            _ => Err(OPUS_UNIMPLEMENTED),
        }
    })
}

/// `OPUS_MULTISTREAM_GET_ENCODER_STATE` / `OPUS_MULTISTREAM_GET_DECODER_STATE`: hands out a
/// handle for one stream of a multistream or projection state, usable with the
/// `opus_encoder_*` / `opus_decoder_*` functions for as long as the parent lives.
///
/// # Safety
/// `st` is a state of the given kind (or NULL); `value` is NULL or valid for one pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opusorus_ctl_state(
    st: *mut c_void,
    kind: c_int,
    request: c_int,
    stream_id: i32,
    value: *mut *mut c_void,
) -> c_int {
    guard(|| {
        let kind = api_kind(kind)?;
        let stream_kind = match (kind, request) {
            (
                Kind::MsEncoder | Kind::ProjectionEncoder,
                OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST,
            ) => Kind::StreamEncoder,
            (
                Kind::MsDecoder | Kind::ProjectionDecoder,
                OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST,
            ) => Kind::StreamDecoder,
            _ => return Err(OPUS_UNIMPLEMENTED),
        };
        // Materialize byte copies so the stream handle refers to this state's own object.
        // SAFETY: caller contract.
        let r = unsafe { handle::resolve(st, &[kind], Access::Write) }?;
        // SAFETY: resolved just now; single-threaded use per the libopus contract.
        let entry = unsafe { r.entry() };
        let valid = match &mut entry.object {
            Object::MsEncoder(ms) => ms.encoder_state(stream_id).map(drop),
            Object::ProjectionEncoder(p) => p.ms_mut().encoder_state(stream_id).map(drop),
            Object::MsDecoder(ms) => ms.decoder_state(stream_id).map(drop),
            Object::ProjectionDecoder(p) => p.ms_decoder().decoder_state(stream_id).map(drop),
            _ => return Err(OPUS_INVALID_STATE),
        };
        valid.map_err(code)?;
        if value.is_null() {
            return Err(OPUS_BAD_ARG);
        }
        let block = entry.stream_block(stream_kind, stream_id)?;
        // SAFETY: non-NULL, caller contract.
        unsafe { value.write(block) };
        Ok(OPUS_OK)
    })
}

/// Defines `$name` as a naked function that tail-jumps to the C function `$target`, leaving
/// the argument registers and the stack (hence C variadic arguments) untouched.
macro_rules! trampoline {
    ($($name:ident => $target:ident;)*) => {$(
        unsafe extern "C" {
            fn $target();
        }

        #[doc = concat!("`", stringify!($name), "(st, request, ...)`: libopus CTL entry point.")]
        ///
        /// The variadic body is C (`csrc/ctl.c`); this symbol only jumps there.
        ///
        /// # Safety
        /// Called from C with the libopus variadic signature and arguments.
        #[unsafe(naked)]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name() {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            core::arch::naked_asm!("jmp {}", sym $target);
            #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
            core::arch::naked_asm!("b {}", sym $target);
            #[cfg(target_arch = "riscv64")]
            core::arch::naked_asm!("tail {}", sym $target);
        }
    )*};
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "x86",
    target_arch = "aarch64",
    target_arch = "arm",
    target_arch = "riscv64"
))]
mod trampolines {
    trampoline! {
        opus_encoder_ctl => opusorus_va_opus_encoder_ctl;
        opus_decoder_ctl => opusorus_va_opus_decoder_ctl;
        opus_multistream_encoder_ctl => opusorus_va_opus_multistream_encoder_ctl;
        opus_multistream_decoder_ctl => opusorus_va_opus_multistream_decoder_ctl;
        opus_projection_encoder_ctl => opusorus_va_opus_projection_encoder_ctl;
        opus_projection_decoder_ctl => opusorus_va_opus_projection_decoder_ctl;
        opus_dred_decoder_ctl => opusorus_va_opus_dred_decoder_ctl;
    }
    #[cfg(feature = "custom-modes")]
    trampoline! {
        opus_custom_encoder_ctl => opusorus_va_opus_custom_encoder_ctl;
        opus_custom_decoder_ctl => opusorus_va_opus_custom_decoder_ctl;
    }
}
