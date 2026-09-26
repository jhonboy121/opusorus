//! `opus.h` encoder API: `opus_encoder_*`, `opus_encode*`.

use core::ffi::c_int;

use opus::Encoder;
use opus::encoder::request::{
    OPUS_GET_APPLICATION_REQUEST, OPUS_GET_EXPERT_FRAME_DURATION_REQUEST,
};
use opus::encoder::{encoder_get_size, frame_size_select};

use crate::handle::{self, Access, Kind, Object};
use crate::types::OpusEncoder;
use crate::util::{
    CResult, OPUS_BAD_ARG, OPUS_INVALID_STATE, OPUS_OK, buf_len, code, guard, guard_any, ret_code,
    size_to_int, slice, slice_mut,
};

/// Resolves an `OpusEncoder*` (an encoder, or one stream of a multistream/projection encoder).
///
/// # Safety
/// `st` is NULL or a state block from this library (see [`handle::resolve`]); the reference is
/// only used during the current call. With `Access::Read` it must not be mutated.
pub(crate) unsafe fn encoder<'a>(
    st: *const OpusEncoder,
    access: Access,
) -> CResult<&'a mut Encoder> {
    // SAFETY: caller contract.
    let r = unsafe { handle::resolve(st.cast(), &[Kind::Encoder, Kind::StreamEncoder], access) }?;
    // SAFETY: resolved just now; single-threaded use per the libopus contract.
    let entry = unsafe { r.entry() };
    match (&mut entry.object, r.stream) {
        (Object::Encoder(e), None) => Ok(e),
        (Object::MsEncoder(ms), Some(i)) => ms.encoder_state(i).map_err(code),
        (Object::ProjectionEncoder(p), Some(i)) => p.ms_mut().encoder_state(i).map_err(code),
        _ => Err(OPUS_INVALID_STATE),
    }
}

/// C `frame_size_select` check done before touching the input: `OPUS_BAD_ARG` for a frame
/// size the encoder would reject (so no slice is formed over a buffer the caller never
/// promised), otherwise the number of input samples per channel (`frame_size`).
pub(crate) fn checked_frames(
    application: i32,
    frame_size: c_int,
    variable_duration: i32,
    fs: i32,
) -> CResult<c_int> {
    if frame_size_select(application, frame_size, variable_duration, fs) <= 0 {
        return Err(OPUS_BAD_ARG);
    }
    Ok(frame_size)
}

/// The C `(pcm, frame_size)` input of an encoder call as a slice, after [`checked_frames`].
///
/// # Safety
/// `pcm` is valid for reads of `frames * channels` samples during `'a`.
pub(crate) unsafe fn pcm_in<'a, T>(
    pcm: *const T,
    frames: c_int,
    channels: c_int,
) -> CResult<&'a [T]> {
    let n = buf_len::<T>(frames, channels)?;
    // SAFETY: caller contract.
    unsafe { slice(pcm, n) }
}

/// The C `(data, max_data_bytes)` output buffer of an encoder call. A non-positive size is an
/// empty buffer (the encoder then fails with `OPUS_BAD_ARG`, as in C).
///
/// # Safety
/// `data` is valid for writes of `max_data_bytes` bytes during `'a`.
pub(crate) unsafe fn packet_out<'a>(data: *mut u8, max_data_bytes: i32) -> CResult<&'a mut [u8]> {
    let n = usize::try_from(max_data_bytes.max(0)).map_err(|_| OPUS_BAD_ARG)?;
    // SAFETY: caller contract.
    unsafe { slice_mut(data, n) }
}

/// Validated input length of `opus_encode*` for this encoder: `frame_size`, or 0 for a frame
/// size the encoder rejects. The call then goes to the Rust encoder with no input, so it fails
/// with `OPUS_BAD_ARG` exactly where C does: the entry points that pass their input to
/// `opus_encode_native` unconverted (`opus_encode_float` in the float build, `opus_encode` /
/// `opus_encode24` in 16/24-bit fixed-point builds) reset the final range first, the
/// converting ones return before touching the state.
fn encoder_frames(enc: &Encoder, frame_size: c_int) -> CResult<c_int> {
    let app = enc.ctl_get(OPUS_GET_APPLICATION_REQUEST).map_err(code)?;
    let vd = enc
        .ctl_get(OPUS_GET_EXPERT_FRAME_DURATION_REQUEST)
        .map_err(code)?;
    if frame_size_select(app, frame_size, vd, enc.sample_rate()) <= 0 {
        return Ok(0);
    }
    Ok(frame_size)
}

/// Gets the size of an `OpusEncoder` structure (`opus_encoder_get_size`): 0 for an invalid
/// channel count.
#[unsafe(no_mangle)]
pub extern "C" fn opus_encoder_get_size(channels: c_int) -> c_int {
    guard(|| Ok(size_to_int(handle::block_size(encoder_get_size(channels)))))
}

/// Allocates and initializes an encoder state (`opus_encoder_create`).
///
/// # Safety
/// `error` is NULL or valid for writing one `int`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encoder_create(
    fs: i32,
    channels: c_int,
    application: c_int,
    error: *mut c_int,
) -> *mut OpusEncoder {
    // SAFETY: forwarded caller contract.
    unsafe {
        handle::create(error, || {
            let enc = Encoder::new_raw(fs, channels, application).map_err(code)?;
            Ok((encoder_get_size(channels), Object::Encoder(enc)))
        })
    }
}

/// Initializes a previously allocated encoder state (`opus_encoder_init`); the memory must be
/// at least `opus_encoder_get_size(channels)` bytes.
///
/// # Safety
/// `st` is NULL or valid for writes of `opus_encoder_get_size(channels)` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encoder_init(
    st: *mut OpusEncoder,
    fs: i32,
    channels: c_int,
    application: c_int,
) -> c_int {
    guard(|| {
        let enc = Encoder::new_raw(fs, channels, application).map_err(code)?;
        // SAFETY: caller contract.
        unsafe { handle::install(st.cast(), Object::Encoder(enc)) }?;
        Ok(OPUS_OK)
    })
}

/// Encodes a frame of 16-bit PCM (`opus_encode`).
///
/// # Safety
/// `st` is an encoder state; `pcm` holds `frame_size * channels` samples; `data` holds
/// `max_data_bytes` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encode(
    st: *mut OpusEncoder,
    pcm: *const i16,
    frame_size: c_int,
    data: *mut u8,
    max_data_bytes: i32,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let enc = unsafe { encoder(st, Access::Write) }?;
        let frames = encoder_frames(enc, frame_size)?;
        // SAFETY: caller contract.
        let pcm = unsafe { pcm_in(pcm, frames, enc.channels()) }?;
        // SAFETY: caller contract.
        let out = unsafe { packet_out(data, max_data_bytes) }?;
        ret_code(enc.encode(pcm, frames as usize, out), size_to_int)
    })
}

/// Encodes a frame of 24-bit PCM in `opus_int32` (`opus_encode24`).
///
/// # Safety
/// As [`opus_encode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encode24(
    st: *mut OpusEncoder,
    pcm: *const i32,
    frame_size: c_int,
    data: *mut u8,
    max_data_bytes: i32,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let enc = unsafe { encoder(st, Access::Write) }?;
        let frames = encoder_frames(enc, frame_size)?;
        // SAFETY: caller contract.
        let pcm = unsafe { pcm_in(pcm, frames, enc.channels()) }?;
        // SAFETY: caller contract.
        let out = unsafe { packet_out(data, max_data_bytes) }?;
        ret_code(enc.encode24(pcm, frames as usize, out), size_to_int)
    })
}

/// Encodes a frame of float PCM (`opus_encode_float`).
///
/// # Safety
/// As [`opus_encode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encode_float(
    st: *mut OpusEncoder,
    pcm: *const f32,
    frame_size: c_int,
    data: *mut u8,
    max_data_bytes: i32,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let enc = unsafe { encoder(st, Access::Write) }?;
        let frames = encoder_frames(enc, frame_size)?;
        // SAFETY: caller contract.
        let pcm = unsafe { pcm_in(pcm, frames, enc.channels()) }?;
        // SAFETY: caller contract.
        let out = unsafe { packet_out(data, max_data_bytes) }?;
        ret_code(enc.encode_float(pcm, frames as usize, out), size_to_int)
    })
}

/// Frees an encoder allocated by [`opus_encoder_create`] (`opus_encoder_destroy`).
///
/// # Safety
/// `st` is NULL or an encoder state on the C heap, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encoder_destroy(st: *mut OpusEncoder) {
    // SAFETY: caller contract.
    guard_any((), || unsafe { handle::destroy(st.cast()) });
}
