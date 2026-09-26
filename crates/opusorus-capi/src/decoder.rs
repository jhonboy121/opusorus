//! `opus.h` decoder API: `opus_decoder_*`, `opus_decode*`.

use core::ffi::c_int;

use opus::Decoder;

use crate::handle::{self, Access, Kind, Object};
use crate::types::OpusDecoder;
use crate::util::{
    CResult, OPUS_BAD_ARG, OPUS_INVALID_STATE, OPUS_OK, buf_len, byte_len, code, guard, guard_any,
    packet, ret_code, size_to_int, slice, slice_mut,
};

/// Resolves an `OpusDecoder*` (a decoder, or one stream of a multistream/projection decoder).
///
/// # Safety
/// `st` is NULL or a state block from this library (see [`handle::resolve`]); the reference is
/// only used during the current call. With `Access::Read` it must not be mutated.
pub(crate) unsafe fn decoder<'a>(
    st: *const OpusDecoder,
    access: Access,
) -> CResult<&'a mut Decoder> {
    // SAFETY: caller contract.
    let r = unsafe { handle::resolve(st.cast(), &[Kind::Decoder, Kind::StreamDecoder], access) }?;
    // SAFETY: resolved just now; single-threaded use per the libopus contract.
    let entry = unsafe { r.entry() };
    match (&mut entry.object, r.stream) {
        (Object::Decoder(d), None) => Ok(d),
        (Object::MsDecoder(ms), Some(i)) => ms.decoder_state(i).map_err(code),
        (Object::ProjectionDecoder(p), Some(i)) => p.ms_decoder().decoder_state(i).map_err(code),
        _ => Err(OPUS_INVALID_STATE),
    }
}

/// Gets the size of an `OpusDecoder` structure (`opus_decoder_get_size`): 0 for an invalid
/// channel count.
#[unsafe(no_mangle)]
pub extern "C" fn opus_decoder_get_size(channels: c_int) -> c_int {
    guard(|| Ok(size_to_int(handle::block_size(Decoder::get_size(channels)))))
}

/// Allocates and initializes a decoder state (`opus_decoder_create`).
///
/// # Safety
/// `error` is NULL or valid for writing one `int`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_create(
    fs: i32,
    channels: c_int,
    error: *mut c_int,
) -> *mut OpusDecoder {
    // SAFETY: forwarded caller contract.
    unsafe {
        handle::create(error, || {
            let dec = Decoder::new(fs, channels).map_err(code)?;
            Ok((Decoder::get_size(channels), Object::Decoder(dec)))
        })
    }
}

/// Initializes a previously allocated decoder state (`opus_decoder_init`); the memory must be
/// at least `opus_decoder_get_size(channels)` bytes.
///
/// # Safety
/// `st` is NULL or valid for writes of `opus_decoder_get_size(channels)` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_init(
    st: *mut OpusDecoder,
    fs: i32,
    channels: c_int,
) -> c_int {
    guard(|| {
        let dec = Decoder::new(fs, channels).map_err(code)?;
        // SAFETY: caller contract.
        unsafe { handle::install(st.cast(), Object::Decoder(dec)) }?;
        Ok(OPUS_OK)
    })
}

/// Decodes a packet to 16-bit PCM (`opus_decode`).
///
/// # Safety
/// `st` is a decoder state; `data` is NULL or holds `len` bytes; `pcm` holds
/// `frame_size * channels` samples.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decode(
    st: *mut OpusDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut i16,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let dec = unsafe { decoder(st, Access::Write) }?;
        if frame_size <= 0 {
            return Err(OPUS_BAD_ARG);
        }
        // SAFETY: caller contract.
        let data = unsafe { packet(data, len) }?;
        let n = buf_len::<i16>(frame_size, dec.channels() as c_int)?;
        // SAFETY: caller contract (`frame_size * channels` samples).
        let pcm = unsafe { slice_mut(pcm, n) }?;
        ret_code(dec.opus_decode(data, pcm, frame_size, decode_fec), |n| n)
    })
}

/// Decodes a packet to 24-bit PCM in `opus_int32` (`opus_decode24`).
///
/// # Safety
/// As [`opus_decode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decode24(
    st: *mut OpusDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut i32,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let dec = unsafe { decoder(st, Access::Write) }?;
        if frame_size <= 0 {
            return Err(OPUS_BAD_ARG);
        }
        // SAFETY: caller contract.
        let data = unsafe { packet(data, len) }?;
        let n = buf_len::<i32>(frame_size, dec.channels() as c_int)?;
        // SAFETY: caller contract.
        let pcm = unsafe { slice_mut(pcm, n) }?;
        ret_code(dec.opus_decode24(data, pcm, frame_size, decode_fec), |n| n)
    })
}

/// Decodes a packet to float PCM (`opus_decode_float`).
///
/// # Safety
/// As [`opus_decode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decode_float(
    st: *mut OpusDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut f32,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let dec = unsafe { decoder(st, Access::Write) }?;
        if frame_size <= 0 {
            return Err(OPUS_BAD_ARG);
        }
        // SAFETY: caller contract.
        let data = unsafe { packet(data, len) }?;
        let n = buf_len::<f32>(frame_size, dec.channels() as c_int)?;
        // SAFETY: caller contract.
        let pcm = unsafe { slice_mut(pcm, n) }?;
        ret_code(
            dec.opus_decode_float(data, pcm, frame_size, decode_fec),
            |n| n,
        )
    })
}

/// Frees a decoder allocated by [`opus_decoder_create`] (`opus_decoder_destroy`).
///
/// # Safety
/// `st` is NULL or a decoder state on the C heap, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_destroy(st: *mut OpusDecoder) {
    // SAFETY: caller contract.
    guard_any((), || unsafe { handle::destroy(st.cast()) });
}

/// Number of samples of a packet at the decoder's rate (`opus_decoder_get_nb_samples`).
///
/// # Safety
/// `dec` is a decoder state; `packet` holds `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_get_nb_samples(
    dec: *const OpusDecoder,
    packet: *const u8,
    len: i32,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract; only read.
        let dec = unsafe { decoder(dec, Access::Read) }?;
        if len < 1 {
            return Err(OPUS_BAD_ARG);
        }
        // SAFETY: caller contract.
        let data = unsafe { slice(packet, byte_len(len)?) }?;
        ret_code(dec.nb_samples(data), size_to_int)
    })
}
