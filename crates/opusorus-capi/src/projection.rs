//! `opus_projection.h`: ambisonics projection encoder and decoder.

use core::ffi::c_int;

use opus::projection_encoder::projection_ambisonics_encoder_get_size;
use opus::{ProjectionDecoder, ProjectionEncoder};

use crate::encoder::{packet_out, pcm_in};
use crate::handle::{self, Access, Kind, Object};
use crate::multistream::{ms_frames, ms_packet};
use crate::types::{OpusProjectionDecoder, OpusProjectionEncoder};
use crate::util::{
    CResult, OPUS_ALLOC_FAIL, OPUS_BAD_ARG, OPUS_INVALID_STATE, OPUS_OK, buf_len, byte_len, code,
    guard, guard_any, ret_code, size_to_int, slice, slice_mut,
};

/// Resolves an `OpusProjectionEncoder*`.
///
/// # Safety
/// As [`handle::resolve`]; the reference is only used during the current call.
pub(crate) unsafe fn projection_encoder<'a>(
    st: *const OpusProjectionEncoder,
    access: Access,
) -> CResult<&'a mut ProjectionEncoder> {
    // SAFETY: caller contract.
    let r = unsafe { handle::resolve(st.cast(), &[Kind::ProjectionEncoder], access) }?;
    // SAFETY: resolved just now; single-threaded use per the libopus contract.
    match &mut unsafe { r.entry() }.object {
        Object::ProjectionEncoder(e) => Ok(e),
        _ => Err(OPUS_INVALID_STATE),
    }
}

/// Resolves an `OpusProjectionDecoder*`.
///
/// # Safety
/// As [`handle::resolve`]; the reference is only used during the current call.
pub(crate) unsafe fn projection_decoder<'a>(
    st: *const OpusProjectionDecoder,
    access: Access,
) -> CResult<&'a mut ProjectionDecoder> {
    // SAFETY: caller contract.
    let r = unsafe { handle::resolve(st.cast(), &[Kind::ProjectionDecoder], access) }?;
    // SAFETY: resolved just now; single-threaded use per the libopus contract.
    match &mut unsafe { r.entry() }.object {
        Object::ProjectionDecoder(d) => Ok(d),
        _ => Err(OPUS_INVALID_STATE),
    }
}

/// C `get_streams_from_channels` (family 3 only): the stream layout written to the caller's
/// `*streams` / `*coupled_streams` before the encoder is initialized.
const fn streams_from_channels(channels: c_int, mapping_family: c_int) -> Option<(c_int, c_int)> {
    if mapping_family != 3 || channels < 1 || channels > 227 {
        return None;
    }
    let order_plus_one = channels.isqrt();
    let nondiegetic = channels - order_plus_one * order_plus_one;
    if nondiegetic != 0 && nondiegetic != 2 {
        return None;
    }
    Some(((channels + 1) / 2, channels / 2))
}

/// Writes the stream layout outputs and builds the encoder (the body of
/// `opus_projection_ambisonics_encoder_init`).
///
/// # Safety
/// `streams` / `coupled_streams` are NULL or valid for one `int` write.
unsafe fn ambisonics_new(
    fs: i32,
    channels: c_int,
    mapping_family: c_int,
    streams: *mut c_int,
    coupled_streams: *mut c_int,
    application: c_int,
) -> CResult<ProjectionEncoder> {
    if streams.is_null() || coupled_streams.is_null() {
        return Err(OPUS_BAD_ARG);
    }
    let Some((s, c)) = streams_from_channels(channels, mapping_family) else {
        return Err(OPUS_BAD_ARG);
    };
    // SAFETY: non-NULL, caller contract.
    unsafe {
        streams.write(s);
        coupled_streams.write(c);
    }
    // Where C's size query fails, its init fails with OPUS_BAD_ARG.
    if projection_ambisonics_encoder_get_size(channels, mapping_family) == 0 {
        return Err(OPUS_BAD_ARG);
    }
    let enc = ProjectionEncoder::new_ambisonics_raw(fs, channels, mapping_family, application)
        .map_err(code)?;
    debug_assert!(enc.streams() == s && enc.coupled_streams() == c);
    Ok(enc)
}

/// The `(demixing_matrix, demixing_matrix_size)` argument.
///
/// # Safety
/// `matrix` is valid for reads of `size` bytes during `'a`.
unsafe fn matrix_in<'a>(matrix: *const u8, size: i32) -> CResult<&'a [u8]> {
    // SAFETY: caller contract.
    unsafe { slice(matrix, byte_len(size)?) }
}

// ---------------------------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------------------------

/// `opus_projection_ambisonics_encoder_get_size`: 0 for unsupported arguments.
#[unsafe(no_mangle)]
pub extern "C" fn opus_projection_ambisonics_encoder_get_size(
    channels: c_int,
    mapping_family: c_int,
) -> i32 {
    guard(|| {
        Ok(size_to_int(handle::block_size(
            projection_ambisonics_encoder_get_size(channels, mapping_family),
        )))
    })
}

/// `opus_projection_ambisonics_encoder_create`.
///
/// # Safety
/// `streams` / `coupled_streams` are valid for one `int` write; `error` is NULL or valid for
/// one `int` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_ambisonics_encoder_create(
    fs: i32,
    channels: c_int,
    mapping_family: c_int,
    streams: *mut c_int,
    coupled_streams: *mut c_int,
    application: c_int,
    error: *mut c_int,
) -> *mut OpusProjectionEncoder {
    // SAFETY: forwarded caller contract.
    unsafe {
        handle::create(error, || {
            let size = projection_ambisonics_encoder_get_size(channels, mapping_family);
            if size == 0 {
                return Err(OPUS_ALLOC_FAIL);
            }
            let enc = ambisonics_new(
                fs,
                channels,
                mapping_family,
                streams,
                coupled_streams,
                application,
            )?;
            Ok((size, Object::ProjectionEncoder(enc)))
        })
    }
}

/// `opus_projection_ambisonics_encoder_init`.
///
/// # Safety
/// `st` is valid for writes of `opus_projection_ambisonics_encoder_get_size(channels,
/// mapping_family)` bytes; `streams` / `coupled_streams` as for the create function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_ambisonics_encoder_init(
    st: *mut OpusProjectionEncoder,
    fs: i32,
    channels: c_int,
    mapping_family: c_int,
    streams: *mut c_int,
    coupled_streams: *mut c_int,
    application: c_int,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let enc = unsafe {
            ambisonics_new(
                fs,
                channels,
                mapping_family,
                streams,
                coupled_streams,
                application,
            )
        }?;
        // SAFETY: caller contract.
        unsafe { handle::install(st.cast(), Object::ProjectionEncoder(enc)) }?;
        Ok(OPUS_OK)
    })
}

/// Shared body of the `opus_projection_encode*` functions.
///
/// # Safety
/// `st` is a projection encoder; `pcm` holds `frame_size * channels` samples; `data` holds
/// `max_data_bytes` bytes.
unsafe fn projection_encode<T>(
    st: *mut OpusProjectionEncoder,
    pcm: *const T,
    frame_size: c_int,
    data: *mut u8,
    max_data_bytes: i32,
    encode: impl FnOnce(&mut ProjectionEncoder, &[T], usize, &mut [u8]) -> opus::Result<usize>,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let enc = unsafe { projection_encoder(st, Access::Write) }?;
        let frames = ms_frames(enc.ms(), frame_size)?;
        // SAFETY: caller contract.
        let pcm = unsafe { pcm_in(pcm, frames, enc.channels()) }?;
        // SAFETY: caller contract.
        let out = unsafe { packet_out(data, max_data_bytes) }?;
        ret_code(encode(enc, pcm, frames as usize, out), size_to_int)
    })
}

/// `opus_projection_encode`.
///
/// # Safety
/// See [`projection_encode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_encode(
    st: *mut OpusProjectionEncoder,
    pcm: *const i16,
    frame_size: c_int,
    data: *mut u8,
    max_data_bytes: i32,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        projection_encode(
            st,
            pcm,
            frame_size,
            data,
            max_data_bytes,
            ProjectionEncoder::encode,
        )
    }
}

/// `opus_projection_encode24`.
///
/// # Safety
/// See [`projection_encode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_encode24(
    st: *mut OpusProjectionEncoder,
    pcm: *const i32,
    frame_size: c_int,
    data: *mut u8,
    max_data_bytes: i32,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        projection_encode(
            st,
            pcm,
            frame_size,
            data,
            max_data_bytes,
            ProjectionEncoder::encode24,
        )
    }
}

/// `opus_projection_encode_float`.
///
/// # Safety
/// See [`projection_encode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_encode_float(
    st: *mut OpusProjectionEncoder,
    pcm: *const f32,
    frame_size: c_int,
    data: *mut u8,
    max_data_bytes: i32,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        projection_encode(
            st,
            pcm,
            frame_size,
            data,
            max_data_bytes,
            ProjectionEncoder::encode_float,
        )
    }
}

/// `opus_projection_encoder_destroy`.
///
/// # Safety
/// `st` is NULL or a projection encoder on the C heap, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_encoder_destroy(st: *mut OpusProjectionEncoder) {
    // SAFETY: caller contract.
    guard_any((), || unsafe { handle::destroy(st.cast()) });
}

// ---------------------------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------------------------

/// `opus_projection_decoder_get_size`: 0 for unsupported arguments.
#[unsafe(no_mangle)]
pub extern "C" fn opus_projection_decoder_get_size(
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
) -> i32 {
    guard(|| {
        Ok(size_to_int(handle::block_size(
            ProjectionDecoder::get_size(channels, streams, coupled_streams),
        )))
    })
}

/// `opus_projection_decoder_create`.
///
/// # Safety
/// `demixing_matrix` holds `demixing_matrix_size` bytes; `error` is NULL or valid for one `int`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decoder_create(
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    demixing_matrix: *mut u8,
    demixing_matrix_size: i32,
    error: *mut c_int,
) -> *mut OpusProjectionDecoder {
    // SAFETY: forwarded caller contract.
    unsafe {
        handle::create(error, || {
            let size = ProjectionDecoder::get_size(channels, streams, coupled_streams);
            if size == 0 {
                return Err(OPUS_ALLOC_FAIL);
            }
            let matrix = matrix_in(demixing_matrix, demixing_matrix_size)?;
            let dec = ProjectionDecoder::new(fs, channels, streams, coupled_streams, matrix)
                .map_err(code)?;
            Ok((size, Object::ProjectionDecoder(dec)))
        })
    }
}

/// `opus_projection_decoder_init`.
///
/// # Safety
/// `st` is valid for writes of `opus_projection_decoder_get_size(channels, streams,
/// coupled_streams)` bytes; `demixing_matrix` holds `demixing_matrix_size` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decoder_init(
    st: *mut OpusProjectionDecoder,
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    demixing_matrix: *mut u8,
    demixing_matrix_size: i32,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let matrix = unsafe { matrix_in(demixing_matrix, demixing_matrix_size) }?;
        // Where C's size query fails, its init fails with OPUS_BAD_ARG.
        if ProjectionDecoder::get_size(channels, streams, coupled_streams) == 0 {
            return Err(OPUS_BAD_ARG);
        }
        let dec =
            ProjectionDecoder::new(fs, channels, streams, coupled_streams, matrix).map_err(code)?;
        // SAFETY: caller contract.
        unsafe { handle::install(st.cast(), Object::ProjectionDecoder(dec)) }?;
        Ok(OPUS_OK)
    })
}

/// Shared body of the `opus_projection_decode*` functions.
///
/// # Safety
/// `st` is a projection decoder; `data` is NULL or holds `len` bytes; `pcm` holds
/// `frame_size * channels` samples.
unsafe fn projection_decode<T>(
    st: *mut OpusProjectionDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut T,
    frame_size: c_int,
    decode_fec: c_int,
    decode: impl FnOnce(&mut ProjectionDecoder, Option<&[u8]>, &mut [T], i32, i32) -> opus::Result<i32>,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let dec = unsafe { projection_decoder(st, Access::Write) }?;
        if frame_size <= 0 {
            return Err(OPUS_BAD_ARG);
        }
        // SAFETY: caller contract.
        let data = unsafe { ms_packet(data, len) }?;
        let n = buf_len::<T>(frame_size, dec.channels() as c_int)?;
        // SAFETY: caller contract.
        let pcm = unsafe { slice_mut(pcm, n) }?;
        ret_code(decode(dec, data, pcm, frame_size, decode_fec), |n| n)
    })
}

/// `opus_projection_decode`.
///
/// # Safety
/// See [`projection_decode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decode(
    st: *mut OpusProjectionDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut i16,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        projection_decode(
            st,
            data,
            len,
            pcm,
            frame_size,
            decode_fec,
            ProjectionDecoder::opus_projection_decode,
        )
    }
}

/// `opus_projection_decode24`.
///
/// # Safety
/// See [`projection_decode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decode24(
    st: *mut OpusProjectionDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut i32,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        projection_decode(
            st,
            data,
            len,
            pcm,
            frame_size,
            decode_fec,
            ProjectionDecoder::opus_projection_decode24,
        )
    }
}

/// `opus_projection_decode_float`.
///
/// # Safety
/// See [`projection_decode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decode_float(
    st: *mut OpusProjectionDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut f32,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        projection_decode(
            st,
            data,
            len,
            pcm,
            frame_size,
            decode_fec,
            ProjectionDecoder::opus_projection_decode_float,
        )
    }
}

/// `opus_projection_decoder_destroy`.
///
/// # Safety
/// `st` is NULL or a projection decoder on the C heap, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decoder_destroy(st: *mut OpusProjectionDecoder) {
    // SAFETY: caller contract.
    guard_any((), || unsafe { handle::destroy(st.cast()) });
}
