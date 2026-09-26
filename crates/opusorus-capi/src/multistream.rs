//! `opus_multistream.h`: multistream encoder and decoder.

use core::ffi::c_int;

use opus::encoder::request::{
    OPUS_GET_APPLICATION_REQUEST, OPUS_GET_EXPERT_FRAME_DURATION_REQUEST,
};
use opus::ms_encoder::{ms_encoder_get_size, ms_surround_encoder_get_size};
use opus::{MsDecoder, MsEncoder};

use crate::encoder::{checked_frames, packet_out, pcm_in};
use crate::handle::{self, Access, Kind, Object};
use crate::types::{OpusMSDecoder, OpusMSEncoder};
use crate::util::{
    CResult, OPUS_BAD_ARG, OPUS_INVALID_STATE, OPUS_OK, OPUS_UNIMPLEMENTED, buf_len, byte_len,
    code, guard, guard_any, ret_code, size_to_int, slice, slice_mut,
};

/// Resolves an `OpusMSEncoder*`.
///
/// # Safety
/// As [`handle::resolve`]; the reference is only used during the current call.
pub(crate) unsafe fn ms_encoder<'a>(
    st: *const OpusMSEncoder,
    access: Access,
) -> CResult<&'a mut MsEncoder> {
    // SAFETY: caller contract.
    let r = unsafe { handle::resolve(st.cast(), &[Kind::MsEncoder], access) }?;
    // SAFETY: resolved just now; single-threaded use per the libopus contract.
    match &mut unsafe { r.entry() }.object {
        Object::MsEncoder(e) => Ok(e),
        _ => Err(OPUS_INVALID_STATE),
    }
}

/// Resolves an `OpusMSDecoder*`.
///
/// # Safety
/// As [`handle::resolve`]; the reference is only used during the current call.
pub(crate) unsafe fn ms_decoder<'a>(
    st: *const OpusMSDecoder,
    access: Access,
) -> CResult<&'a mut MsDecoder> {
    // SAFETY: caller contract.
    let r = unsafe { handle::resolve(st.cast(), &[Kind::MsDecoder], access) }?;
    // SAFETY: resolved just now; single-threaded use per the libopus contract.
    match &mut unsafe { r.entry() }.object {
        Object::MsDecoder(d) => Ok(d),
        _ => Err(OPUS_INVALID_STATE),
    }
}

/// The channel mapping argument (`channels` bytes). An invalid channel count is left to the
/// Rust constructor (C checks it before reading the mapping).
///
/// # Safety
/// For `1 <= channels <= 255`, `mapping` is valid for reads of `channels` bytes during `'a`.
pub(crate) unsafe fn mapping_in<'a>(mapping: *const u8, channels: c_int) -> CResult<&'a [u8]> {
    if !(1..=255).contains(&channels) {
        return Ok(&[]);
    }
    // SAFETY: caller contract.
    unsafe { slice(mapping, channels as usize) }
}

/// Validated input length of `opus_multistream_encode*` (C `frame_size_select` with the
/// multistream settings).
pub(crate) fn ms_frames(ms: &MsEncoder, frame_size: c_int) -> CResult<c_int> {
    let app = ms.ctl_get(OPUS_GET_APPLICATION_REQUEST).map_err(code)?;
    let vd = ms
        .ctl_get(OPUS_GET_EXPERT_FRAME_DURATION_REQUEST)
        .map_err(code)?;
    checked_frames(app, frame_size, vd, ms.sample_rate())
}

/// A multistream packet argument: `len < 0` is `OPUS_BAD_ARG`, `len == 0` (or no data) a lost
/// packet.
///
/// # Safety
/// If non-NULL and `len > 0`, `data` is valid for reads of `len` bytes during `'a`.
pub(crate) unsafe fn ms_packet<'a>(data: *const u8, len: i32) -> CResult<Option<&'a [u8]>> {
    let n = byte_len(len)?;
    if n == 0 || data.is_null() {
        return Ok(None);
    }
    // SAFETY: caller contract.
    unsafe { slice(data, n) }.map(Some)
}

/// Channel layout of `opus_multistream_surround_encoder_init` (C computes and returns it
/// before initializing the stream encoders, even if that later fails).
fn surround_layout(channels: c_int, family: c_int) -> CResult<(c_int, c_int, [u8; 255])> {
    /// `vorbis_mappings` of `src/opus_multistream_encoder.c` (index `channels - 1`).
    const VORBIS: [(c_int, c_int, [u8; 8]); 8] = [
        (1, 0, [0, 0, 0, 0, 0, 0, 0, 0]),
        (1, 1, [0, 1, 0, 0, 0, 0, 0, 0]),
        (2, 1, [0, 2, 1, 0, 0, 0, 0, 0]),
        (2, 2, [0, 1, 2, 3, 0, 0, 0, 0]),
        (3, 2, [0, 4, 1, 2, 3, 0, 0, 0]),
        (4, 2, [0, 4, 1, 2, 3, 5, 0, 0]),
        (4, 3, [0, 4, 1, 2, 3, 5, 6, 0]),
        (5, 3, [0, 6, 1, 2, 3, 4, 5, 7]),
    ];
    if !(1..=255).contains(&channels) {
        return Err(OPUS_BAD_ARG);
    }
    let ch = channels as usize;
    let mut mapping = [0u8; 255];
    let (streams, coupled) = match family {
        0 if channels == 1 => (1, 0),
        0 if channels == 2 => {
            mapping[1] = 1;
            (1, 1)
        }
        0 => return Err(OPUS_UNIMPLEMENTED),
        1 if channels <= 8 => {
            let (s, c, m) = VORBIS[ch - 1];
            mapping[..ch].copy_from_slice(&m[..ch]);
            (s, c)
        }
        255 => {
            for (i, m) in mapping[..ch].iter_mut().enumerate() {
                *m = i as u8;
            }
            (channels, 0)
        }
        2 => {
            // validate_ambisonics
            if channels > 227 {
                return Err(OPUS_BAD_ARG);
            }
            let order_plus_one = channels.isqrt();
            let acn = order_plus_one * order_plus_one;
            let nondiegetic = channels - acn;
            if nondiegetic != 0 && nondiegetic != 2 {
                return Err(OPUS_BAD_ARG);
            }
            let s = acn + c_int::from(nondiegetic != 0);
            let c = c_int::from(nondiegetic != 0);
            for i in 0..(s - c) {
                mapping[i as usize] = (i + c * 2) as u8;
            }
            for i in 0..c * 2 {
                mapping[(i + (s - c)) as usize] = i as u8;
            }
            (s, c)
        }
        _ => return Err(OPUS_UNIMPLEMENTED),
    };
    Ok((streams, coupled, mapping))
}

/// Writes the surround layout outputs (`*streams`, `*coupled_streams`, `mapping[..channels]`)
/// and builds the encoder.
///
/// # Safety
/// `streams`/`coupled_streams` are NULL or valid for one `int` write; `mapping` is NULL or valid
/// for writes of `channels` bytes.
unsafe fn surround_new(
    fs: i32,
    channels: c_int,
    family: c_int,
    streams: *mut c_int,
    coupled_streams: *mut c_int,
    mapping: *mut u8,
    application: c_int,
) -> CResult<MsEncoder> {
    let (s, c, m) = surround_layout(channels, family)?;
    if streams.is_null() || coupled_streams.is_null() {
        return Err(OPUS_BAD_ARG);
    }
    // SAFETY: caller contract (`channels` is 1..=255 after `surround_layout`).
    let out = unsafe { slice_mut(mapping, channels as usize) }?;
    // SAFETY: non-NULL, caller contract.
    unsafe {
        streams.write(s);
        coupled_streams.write(c);
    }
    out.copy_from_slice(&m[..channels as usize]);
    let enc = MsEncoder::new_surround_raw(fs, channels, family, application).map_err(code)?;
    // Rust recomputes the same layout; keep the outputs authoritative from the encoder.
    debug_assert!(enc.streams() == s && enc.coupled_streams() == c && enc.mapping() == &out[..]);
    Ok(enc)
}

// ---------------------------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------------------------

/// `opus_multistream_encoder_get_size`: 0 for invalid stream counts.
#[unsafe(no_mangle)]
pub extern "C" fn opus_multistream_encoder_get_size(streams: c_int, coupled_streams: c_int) -> i32 {
    guard(|| {
        Ok(size_to_int(handle::block_size(ms_encoder_get_size(
            streams,
            coupled_streams,
        ))))
    })
}

/// `opus_multistream_surround_encoder_get_size`: 0 for invalid arguments.
#[unsafe(no_mangle)]
pub extern "C" fn opus_multistream_surround_encoder_get_size(
    channels: c_int,
    mapping_family: c_int,
) -> i32 {
    guard(|| {
        Ok(size_to_int(handle::block_size(
            ms_surround_encoder_get_size(channels, mapping_family),
        )))
    })
}

/// `opus_multistream_encoder_create`.
///
/// # Safety
/// `mapping` holds `channels` bytes; `error` is NULL or valid for one `int` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encoder_create(
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    mapping: *const u8,
    application: c_int,
    error: *mut c_int,
) -> *mut OpusMSEncoder {
    // SAFETY: forwarded caller contract.
    unsafe {
        handle::create(error, || {
            let mapping = mapping_in(mapping, channels)?;
            let enc =
                MsEncoder::new_raw(fs, channels, streams, coupled_streams, mapping, application)
                    .map_err(code)?;
            Ok((
                ms_encoder_get_size(streams, coupled_streams),
                Object::MsEncoder(enc),
            ))
        })
    }
}

/// `opus_multistream_surround_encoder_create`.
///
/// # Safety
/// `streams`/`coupled_streams` are valid for one `int` write, `mapping` for `channels` bytes;
/// `error` is NULL or valid for one `int` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_surround_encoder_create(
    fs: i32,
    channels: c_int,
    mapping_family: c_int,
    streams: *mut c_int,
    coupled_streams: *mut c_int,
    mapping: *mut u8,
    application: c_int,
    error: *mut c_int,
) -> *mut OpusMSEncoder {
    // SAFETY: forwarded caller contract.
    unsafe {
        handle::create(error, || {
            let enc = surround_new(
                fs,
                channels,
                mapping_family,
                streams,
                coupled_streams,
                mapping,
                application,
            )?;
            Ok((
                ms_surround_encoder_get_size(channels, mapping_family),
                Object::MsEncoder(enc),
            ))
        })
    }
}

/// `opus_multistream_encoder_init`.
///
/// # Safety
/// `st` is valid for writes of `opus_multistream_encoder_get_size(streams, coupled_streams)`
/// bytes; `mapping` holds `channels` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encoder_init(
    st: *mut OpusMSEncoder,
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    mapping: *const u8,
    application: c_int,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let mapping = unsafe { mapping_in(mapping, channels) }?;
        let enc = MsEncoder::new_raw(fs, channels, streams, coupled_streams, mapping, application)
            .map_err(code)?;
        // SAFETY: caller contract.
        unsafe { handle::install(st.cast(), Object::MsEncoder(enc)) }?;
        Ok(OPUS_OK)
    })
}

/// `opus_multistream_surround_encoder_init`.
///
/// # Safety
/// As [`opus_multistream_surround_encoder_create`], plus `st` valid for writes of
/// `opus_multistream_surround_encoder_get_size(channels, mapping_family)` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_surround_encoder_init(
    st: *mut OpusMSEncoder,
    fs: i32,
    channels: c_int,
    mapping_family: c_int,
    streams: *mut c_int,
    coupled_streams: *mut c_int,
    mapping: *mut u8,
    application: c_int,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let enc = unsafe {
            surround_new(
                fs,
                channels,
                mapping_family,
                streams,
                coupled_streams,
                mapping,
                application,
            )
        }?;
        // SAFETY: caller contract.
        unsafe { handle::install(st.cast(), Object::MsEncoder(enc)) }?;
        Ok(OPUS_OK)
    })
}

/// Shared body of the `opus_multistream_encode*` functions.
///
/// # Safety
/// `st` is a multistream encoder; `pcm` holds `frame_size * channels` samples; `data` holds
/// `max_data_bytes` bytes.
unsafe fn ms_encode<T>(
    st: *mut OpusMSEncoder,
    pcm: *const T,
    frame_size: c_int,
    data: *mut u8,
    max_data_bytes: i32,
    encode: impl FnOnce(&mut MsEncoder, &[T], usize, &mut [u8]) -> opus::Result<usize>,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let ms = unsafe { ms_encoder(st, Access::Write) }?;
        let frames = ms_frames(ms, frame_size)?;
        // SAFETY: caller contract.
        let pcm = unsafe { pcm_in(pcm, frames, ms.channels()) }?;
        // SAFETY: caller contract.
        let out = unsafe { packet_out(data, max_data_bytes) }?;
        ret_code(encode(ms, pcm, frames as usize, out), size_to_int)
    })
}

/// `opus_multistream_encode`.
///
/// # Safety
/// See [`ms_encode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encode(
    st: *mut OpusMSEncoder,
    pcm: *const i16,
    frame_size: c_int,
    data: *mut u8,
    max_data_bytes: i32,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe { ms_encode(st, pcm, frame_size, data, max_data_bytes, MsEncoder::encode) }
}

/// `opus_multistream_encode24`.
///
/// # Safety
/// See [`ms_encode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encode24(
    st: *mut OpusMSEncoder,
    pcm: *const i32,
    frame_size: c_int,
    data: *mut u8,
    max_data_bytes: i32,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        ms_encode(
            st,
            pcm,
            frame_size,
            data,
            max_data_bytes,
            MsEncoder::encode24,
        )
    }
}

/// `opus_multistream_encode_float`.
///
/// # Safety
/// See [`ms_encode`].
#[cfg(not(feature = "disable-float-api"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encode_float(
    st: *mut OpusMSEncoder,
    pcm: *const f32,
    frame_size: c_int,
    data: *mut u8,
    max_data_bytes: i32,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        ms_encode(
            st,
            pcm,
            frame_size,
            data,
            max_data_bytes,
            MsEncoder::encode_float,
        )
    }
}

/// `opus_multistream_encoder_destroy`.
///
/// # Safety
/// `st` is NULL or a multistream encoder on the C heap, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encoder_destroy(st: *mut OpusMSEncoder) {
    // SAFETY: caller contract.
    guard_any((), || unsafe { handle::destroy(st.cast()) });
}

// ---------------------------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------------------------

/// `opus_multistream_decoder_get_size`: 0 for invalid stream counts.
#[unsafe(no_mangle)]
pub extern "C" fn opus_multistream_decoder_get_size(streams: c_int, coupled_streams: c_int) -> i32 {
    guard(|| {
        Ok(size_to_int(handle::block_size(MsDecoder::get_size(
            streams,
            coupled_streams,
        ))))
    })
}

/// `opus_multistream_decoder_create`.
///
/// # Safety
/// `mapping` holds `channels` bytes; `error` is NULL or valid for one `int` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decoder_create(
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    mapping: *const u8,
    error: *mut c_int,
) -> *mut OpusMSDecoder {
    // SAFETY: forwarded caller contract.
    unsafe {
        handle::create(error, || {
            let mapping = mapping_in(mapping, channels)?;
            let dec =
                MsDecoder::new(fs, channels, streams, coupled_streams, mapping).map_err(code)?;
            Ok((
                MsDecoder::get_size(streams, coupled_streams),
                Object::MsDecoder(dec),
            ))
        })
    }
}

/// `opus_multistream_decoder_init`.
///
/// # Safety
/// `st` is valid for writes of `opus_multistream_decoder_get_size(streams, coupled_streams)`
/// bytes; `mapping` holds `channels` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decoder_init(
    st: *mut OpusMSDecoder,
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    mapping: *const u8,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let mapping = unsafe { mapping_in(mapping, channels) }?;
        let dec = MsDecoder::new(fs, channels, streams, coupled_streams, mapping).map_err(code)?;
        // SAFETY: caller contract.
        unsafe { handle::install(st.cast(), Object::MsDecoder(dec)) }?;
        Ok(OPUS_OK)
    })
}

/// Shared body of the `opus_multistream_decode*` functions.
///
/// # Safety
/// `st` is a multistream decoder; `data` is NULL or holds `len` bytes; `pcm` holds
/// `frame_size * channels` samples.
unsafe fn ms_decode<T>(
    st: *mut OpusMSDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut T,
    frame_size: c_int,
    decode_fec: c_int,
    decode: impl FnOnce(&mut MsDecoder, Option<&[u8]>, &mut [T], i32, i32) -> opus::Result<i32>,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let ms = unsafe { ms_decoder(st, Access::Write) }?;
        if frame_size <= 0 {
            return Err(OPUS_BAD_ARG);
        }
        // SAFETY: caller contract.
        let data = unsafe { ms_packet(data, len) }?;
        let n = buf_len::<T>(frame_size, ms.channels() as c_int)?;
        // SAFETY: caller contract.
        let pcm = unsafe { slice_mut(pcm, n) }?;
        ret_code(decode(ms, data, pcm, frame_size, decode_fec), |n| n)
    })
}

/// `opus_multistream_decode`.
///
/// # Safety
/// See [`ms_decode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decode(
    st: *mut OpusMSDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut i16,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        ms_decode(
            st,
            data,
            len,
            pcm,
            frame_size,
            decode_fec,
            MsDecoder::opus_multistream_decode,
        )
    }
}

/// `opus_multistream_decode24`.
///
/// # Safety
/// See [`ms_decode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decode24(
    st: *mut OpusMSDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut i32,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        ms_decode(
            st,
            data,
            len,
            pcm,
            frame_size,
            decode_fec,
            MsDecoder::opus_multistream_decode24,
        )
    }
}

/// `opus_multistream_decode_float`.
///
/// # Safety
/// See [`ms_decode`].
#[cfg(not(feature = "disable-float-api"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decode_float(
    st: *mut OpusMSDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut f32,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        ms_decode(
            st,
            data,
            len,
            pcm,
            frame_size,
            decode_fec,
            MsDecoder::opus_multistream_decode_float,
        )
    }
}

/// `opus_multistream_decoder_destroy`.
///
/// # Safety
/// `st` is NULL or a multistream decoder on the C heap, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decoder_destroy(st: *mut OpusMSDecoder) {
    // SAFETY: caller contract.
    guard_any((), || unsafe { handle::destroy(st.cast()) });
}
