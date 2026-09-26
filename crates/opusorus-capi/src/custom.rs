//! `opus_custom.h`: the Opus Custom API (feature `custom-modes`).
//!
//! `OpusCustomMode*` points at an [`opus::celt::static_modes::CeltMode`]: one of the static
//! modes, or a mode created on the heap by [`opus_custom_mode_create`], which stays alive until
//! [`opus_custom_mode_destroy`] (as in C, states must not outlive their mode).

use core::ffi::c_int;
use core::ptr;
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::sync::Mutex;

use opus::celt::celt::{
    CELT_SET_CHANNELS_REQUEST, CELT_SET_END_BAND_REQUEST, CELT_SET_INPUT_CLIPPING_REQUEST,
    CELT_SET_PREDICTION_REQUEST, CELT_SET_SIGNALLING_REQUEST, CELT_SET_START_BAND_REQUEST,
    OPUS_SET_LFE_REQUEST,
};
use opus::celt::celt_decoder::{self, CustomDecoder as RsCustomDecoder};
use opus::celt::celt_encoder::{self, CeltEncoder};
use opus::celt::modes;
use opus::celt::static_modes::CeltMode;
use opus::encoder::request::{
    OPUS_GET_FINAL_RANGE_REQUEST, OPUS_GET_LSB_DEPTH_REQUEST,
    OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST, OPUS_RESET_STATE, OPUS_SET_BITRATE_REQUEST,
    OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_LSB_DEPTH_REQUEST, OPUS_SET_PACKET_LOSS_PERC_REQUEST,
    OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, OPUS_SET_VBR_CONSTRAINT_REQUEST,
    OPUS_SET_VBR_REQUEST,
};
#[cfg(feature = "qext")]
use opus::encoder::request::{OPUS_GET_QEXT_REQUEST, OPUS_SET_QEXT_REQUEST};
use opus::{Error, Result};

use crate::encoder::packet_out;
use crate::handle::{self, Access, Kind, Object};
use crate::types::{OpusCustomDecoder, OpusCustomEncoder, OpusCustomMode};
use crate::util::{
    CResult, OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_INVALID_STATE, OPUS_OK, buf_len, byte_len,
    code, guard, guard_any, ret_code, set_error, size_to_int, slice, slice_mut,
};

/// Heap modes created by [`opus_custom_mode_create`] and not yet destroyed (addresses).
static HEAP_MODES: Mutex<BTreeSet<usize>> = Mutex::new(BTreeSet::new());

/// Resolves an `OpusCustomMode*` (a static mode or a live heap mode).
///
/// # Safety
/// `mode` is NULL or a pointer returned by [`opus_custom_mode_create`] (heap modes must not be
/// destroyed while the returned reference, or a state created from it, is in use).
unsafe fn mode_ref(mode: *const OpusCustomMode) -> CResult<&'static CeltMode> {
    if mode.is_null() {
        return Err(OPUS_BAD_ARG);
    }
    let p = mode.cast::<CeltMode>();
    let is_static = static_modes().any(|m| ptr::eq(m, p));
    let is_heap = HEAP_MODES
        .lock()
        .map_err(|_| OPUS_INTERNAL_ERROR)?
        .contains(&(p as usize));
    if !is_static && !is_heap {
        return Err(OPUS_INVALID_STATE);
    }
    // SAFETY: a static mode, or a leaked heap mode that is still registered (caller contract
    // for its lifetime).
    Ok(unsafe { &*p })
}

/// The statically compiled modes.
fn static_modes() -> impl Iterator<Item = &'static CeltMode> {
    let list: [(i32, i32); 2] = [(48000, 960), (96000, 1920)];
    list.into_iter()
        .filter_map(|(fs, n)| modes::opus_custom_mode_create(fs, n).ok())
}

/// Frame sizes a mode accepts (`shortMdctSize << LM`, `LM <= maxLM`).
fn valid_frame_size(mode: &CeltMode, frame_size: c_int) -> CResult<c_int> {
    if (0..=mode.max_lm).any(|lm| mode.short_mdct_size << lm == frame_size) {
        Ok(frame_size)
    } else {
        Err(OPUS_BAD_ARG)
    }
}

/// An `OpusCustomEncoder`.
#[derive(Clone, Debug)]
pub(crate) struct CustomEncoder(CeltEncoder);

impl CustomEncoder {
    /// Numeric `opus_custom_encoder_ctl` for requests taking one `opus_int32`.
    pub(crate) fn ctl_set(&mut self, request: i32, value: i32) -> Result<()> {
        let e = &mut self.0;
        match request {
            OPUS_SET_COMPLEXITY_REQUEST => e.set_complexity(value),
            CELT_SET_START_BAND_REQUEST => e.set_start_band(value),
            CELT_SET_END_BAND_REQUEST => e.set_end_band(value),
            CELT_SET_PREDICTION_REQUEST => e.set_prediction(value),
            OPUS_SET_PACKET_LOSS_PERC_REQUEST => e.set_packet_loss_perc(value),
            OPUS_SET_VBR_CONSTRAINT_REQUEST => {
                e.set_vbr_constraint(value);
                Ok(())
            }
            OPUS_SET_VBR_REQUEST => {
                e.set_vbr(value);
                Ok(())
            }
            OPUS_SET_BITRATE_REQUEST => e.set_bitrate(value),
            CELT_SET_CHANNELS_REQUEST => e.set_channels(value),
            OPUS_SET_LSB_DEPTH_REQUEST => e.set_lsb_depth(value),
            OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST => e.set_phase_inversion_disabled(value),
            #[cfg(feature = "qext")]
            OPUS_SET_QEXT_REQUEST => e.set_qext(value),
            OPUS_RESET_STATE => {
                e.reset();
                Ok(())
            }
            CELT_SET_INPUT_CLIPPING_REQUEST => {
                e.set_input_clipping(value);
                Ok(())
            }
            CELT_SET_SIGNALLING_REQUEST => {
                e.set_signalling(value);
                Ok(())
            }
            OPUS_SET_LFE_REQUEST => {
                e.set_lfe(value);
                Ok(())
            }
            _ => Err(Error::Unimplemented),
        }
    }

    /// Numeric `opus_custom_encoder_ctl` for requests writing one `opus_int32` / `opus_uint32`.
    pub(crate) const fn ctl_get(&mut self, request: i32) -> Result<i32> {
        let e = &self.0;
        match request {
            OPUS_GET_LSB_DEPTH_REQUEST => Ok(e.lsb_depth()),
            OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST => Ok(e.phase_inversion_disabled()),
            #[cfg(feature = "qext")]
            OPUS_GET_QEXT_REQUEST => Ok(e.qext()),
            OPUS_GET_FINAL_RANGE_REQUEST => Ok(e.final_range() as i32),
            _ => Err(Error::Unimplemented),
        }
    }
}

/// An `OpusCustomDecoder`.
#[derive(Clone, Debug)]
pub(crate) struct CustomDecoder(RsCustomDecoder<'static>);

impl CustomDecoder {
    /// Numeric `opus_custom_decoder_ctl` for requests taking one `opus_int32`.
    pub(crate) fn ctl_set(&mut self, request: i32, value: i32) -> Result<()> {
        self.0.st.ctl_set(request, value)
    }

    /// Numeric `opus_custom_decoder_ctl` for requests writing one `opus_int32` / `opus_uint32`.
    pub(crate) const fn ctl_get(&mut self, request: i32) -> Result<i32> {
        self.0.st.ctl_get(request)
    }
}

/// Resolves an `OpusCustomEncoder*`.
///
/// # Safety
/// As [`handle::resolve`]; the reference is only used during the current call.
pub(crate) unsafe fn custom_encoder<'a>(
    st: *const OpusCustomEncoder,
    access: Access,
) -> CResult<&'a mut CustomEncoder> {
    // SAFETY: caller contract.
    let r = unsafe { handle::resolve(st.cast(), &[Kind::CustomEncoder], access) }?;
    // SAFETY: resolved just now; single-threaded use per the libopus contract.
    match &mut unsafe { r.entry() }.object {
        Object::CustomEncoder(e) => Ok(e),
        _ => Err(OPUS_INVALID_STATE),
    }
}

/// Resolves an `OpusCustomDecoder*`.
///
/// # Safety
/// As [`handle::resolve`]; the reference is only used during the current call.
pub(crate) unsafe fn custom_decoder<'a>(
    st: *const OpusCustomDecoder,
    access: Access,
) -> CResult<&'a mut CustomDecoder> {
    // SAFETY: caller contract.
    let r = unsafe { handle::resolve(st.cast(), &[Kind::CustomDecoder], access) }?;
    // SAFETY: resolved just now; single-threaded use per the libopus contract.
    match &mut unsafe { r.entry() }.object {
        Object::CustomDecoder(d) => Ok(d),
        _ => Err(OPUS_INVALID_STATE),
    }
}

// ---------------------------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------------------------

/// `opus_custom_mode_create`: a mode for sampling rate `fs` and `frame_size` samples.
///
/// # Safety
/// `error` is NULL or valid for one `int` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_mode_create(
    fs: i32,
    frame_size: c_int,
    error: *mut c_int,
) -> *mut OpusCustomMode {
    let r = guard_any(
        Err(OPUS_INTERNAL_ERROR),
        || -> CResult<*mut OpusCustomMode> {
            let mode = modes::opus_custom_mode_create_custom(fs, frame_size).map_err(code)?;
            Ok(match mode {
                Cow::Borrowed(m) => ptr::from_ref(m).cast_mut().cast(),
                Cow::Owned(m) => {
                    let mut heap = HEAP_MODES.lock().map_err(|_| OPUS_INTERNAL_ERROR)?;
                    let p = Box::into_raw(Box::new(m));
                    heap.insert(p as usize);
                    p.cast()
                }
            })
        },
    );
    match r {
        Ok(p) => {
            // SAFETY: caller contract.
            unsafe { set_error(error, OPUS_OK) };
            p
        }
        Err(e) => {
            // SAFETY: caller contract.
            unsafe { set_error(error, e) };
            ptr::null_mut()
        }
    }
}

/// `opus_custom_mode_destroy`: frees a mode created by [`opus_custom_mode_create`] (static
/// modes and unknown pointers are ignored).
///
/// # Safety
/// No state created from `mode` is used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_mode_destroy(mode: *mut OpusCustomMode) {
    guard_any((), || {
        let p = mode.cast::<CeltMode>();
        // Without the lock (poisoned) the mode leaks, which is safe.
        let removed = match HEAP_MODES.lock() {
            Ok(mut heap) => heap.remove(&(p as usize)),
            Err(_) => false,
        };
        if removed {
            // SAFETY: allocated with `Box::into_raw` by `opus_custom_mode_create` and removed
            // from the registry just now, so freed exactly once.
            drop(unsafe { Box::from_raw(p) });
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------------------------

/// `opus_custom_encoder_get_size`.
///
/// # Safety
/// `mode` is a mode from [`opus_custom_mode_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_encoder_get_size(
    mode: *const OpusCustomMode,
    channels: c_int,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let m = unsafe { mode_ref(mode) }?;
        let size = celt_encoder::opus_custom_encoder_get_size(m, channels);
        Ok(size_to_int(handle::block_size(size.max(0) as usize)))
    })
}

/// Builds a custom encoder.
fn new_encoder(mode: &'static CeltMode, channels: c_int) -> CResult<(usize, Object)> {
    let enc = CeltEncoder::opus_custom_encoder_init(Cow::Borrowed(mode), channels).map_err(code)?;
    let size = celt_encoder::opus_custom_encoder_get_size(mode, channels).max(0) as usize;
    Ok((size, Object::CustomEncoder(CustomEncoder(enc))))
}

/// `opus_custom_encoder_init`.
///
/// # Safety
/// `st` is valid for writes of `opus_custom_encoder_get_size(mode, channels)` bytes; `mode` is a
/// mode from [`opus_custom_mode_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_encoder_init(
    st: *mut OpusCustomEncoder,
    mode: *const OpusCustomMode,
    channels: c_int,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let (_, obj) = new_encoder(unsafe { mode_ref(mode) }?, channels)?;
        // SAFETY: caller contract.
        unsafe { handle::install(st.cast(), obj) }?;
        Ok(OPUS_OK)
    })
}

/// `opus_custom_encoder_create`.
///
/// # Safety
/// `mode` is a mode from [`opus_custom_mode_create`]; `error` is NULL or valid for one `int`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_encoder_create(
    mode: *const OpusCustomMode,
    channels: c_int,
    error: *mut c_int,
) -> *mut OpusCustomEncoder {
    // SAFETY: forwarded caller contract.
    unsafe { handle::create(error, || new_encoder(mode_ref(mode)?, channels)) }
}

/// `opus_custom_encoder_destroy`.
///
/// # Safety
/// `st` is NULL or a custom encoder on the C heap, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_encoder_destroy(st: *mut OpusCustomEncoder) {
    // SAFETY: caller contract.
    guard_any((), || unsafe { handle::destroy(st.cast()) });
}

/// Shared body of the `opus_custom_encode*` functions.
///
/// # Safety
/// `st` is a custom encoder; `pcm` holds `frame_size * channels` samples; `compressed` holds
/// `max_bytes` bytes.
unsafe fn custom_encode<T>(
    st: *mut OpusCustomEncoder,
    pcm: *const T,
    frame_size: c_int,
    compressed: *mut u8,
    max_bytes: c_int,
    encode: impl FnOnce(&mut CeltEncoder, &[T], i32, &mut [u8], i32) -> i32,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let enc = &mut unsafe { custom_encoder(st, Access::Write) }?.0;
        if max_bytes < 2 || pcm.is_null() {
            return Err(OPUS_BAD_ARG);
        }
        let frames = valid_frame_size(enc.mode(), frame_size)?;
        let n = buf_len::<T>(frames, enc.channels)?;
        // SAFETY: caller contract.
        let pcm = unsafe { slice(pcm, n) }?;
        // SAFETY: caller contract.
        let out = unsafe { packet_out(compressed, max_bytes) }?;
        let ret = encode(enc, pcm, frames, out, max_bytes);
        if ret < 0 { Err(ret) } else { Ok(ret) }
    })
}

/// `opus_custom_encode_float`.
///
/// # Safety
/// See [`custom_encode`].
#[cfg(not(feature = "disable-float-api"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_encode_float(
    st: *mut OpusCustomEncoder,
    pcm: *const f32,
    frame_size: c_int,
    compressed: *mut u8,
    max_compressed_bytes: c_int,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        custom_encode(
            st,
            pcm,
            frame_size,
            compressed,
            max_compressed_bytes,
            CeltEncoder::opus_custom_encode_float,
        )
    }
}

/// `opus_custom_encode`.
///
/// # Safety
/// See [`custom_encode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_encode(
    st: *mut OpusCustomEncoder,
    pcm: *const i16,
    frame_size: c_int,
    compressed: *mut u8,
    max_compressed_bytes: c_int,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        custom_encode(
            st,
            pcm,
            frame_size,
            compressed,
            max_compressed_bytes,
            CeltEncoder::opus_custom_encode,
        )
    }
}

/// `opus_custom_encode24`.
///
/// # Safety
/// See [`custom_encode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_encode24(
    st: *mut OpusCustomEncoder,
    pcm: *const i32,
    frame_size: c_int,
    compressed: *mut u8,
    max_compressed_bytes: c_int,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        custom_encode(
            st,
            pcm,
            frame_size,
            compressed,
            max_compressed_bytes,
            CeltEncoder::opus_custom_encode24,
        )
    }
}

// ---------------------------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------------------------

/// `opus_custom_decoder_get_size`.
///
/// # Safety
/// `mode` is a mode from [`opus_custom_mode_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_decoder_get_size(
    mode: *const OpusCustomMode,
    channels: c_int,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let m = unsafe { mode_ref(mode) }?;
        let size = celt_decoder::opus_custom_decoder_get_size(m, channels);
        Ok(size_to_int(handle::block_size(size.max(0) as usize)))
    })
}

/// Builds a custom decoder.
fn new_decoder(mode: &'static CeltMode, channels: c_int) -> CResult<(usize, Object)> {
    let dec = RsCustomDecoder::opus_custom_decoder_create(mode, channels).map_err(code)?;
    let size = celt_decoder::opus_custom_decoder_get_size(mode, channels).max(0) as usize;
    Ok((size, Object::CustomDecoder(CustomDecoder(dec))))
}

/// `opus_custom_decoder_init`.
///
/// # Safety
/// `st` is valid for writes of `opus_custom_decoder_get_size(mode, channels)` bytes; `mode` is a
/// mode from [`opus_custom_mode_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_decoder_init(
    st: *mut OpusCustomDecoder,
    mode: *const OpusCustomMode,
    channels: c_int,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let (_, obj) = new_decoder(unsafe { mode_ref(mode) }?, channels)?;
        // SAFETY: caller contract.
        unsafe { handle::install(st.cast(), obj) }?;
        Ok(OPUS_OK)
    })
}

/// `opus_custom_decoder_create`.
///
/// # Safety
/// `mode` is a mode from [`opus_custom_mode_create`]; `error` is NULL or valid for one `int`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_decoder_create(
    mode: *const OpusCustomMode,
    channels: c_int,
    error: *mut c_int,
) -> *mut OpusCustomDecoder {
    // SAFETY: forwarded caller contract.
    unsafe { handle::create(error, || new_decoder(mode_ref(mode)?, channels)) }
}

/// `opus_custom_decoder_destroy`.
///
/// # Safety
/// `st` is NULL or a custom decoder on the C heap, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_decoder_destroy(st: *mut OpusCustomDecoder) {
    // SAFETY: caller contract.
    guard_any((), || unsafe { handle::destroy(st.cast()) });
}

/// Shared body of the `opus_custom_decode*` functions.
///
/// # Safety
/// `st` is a custom decoder; `data` is NULL or holds `len` bytes; `pcm` holds
/// `frame_size * channels` samples.
unsafe fn custom_decode<T>(
    st: *mut OpusCustomDecoder,
    data: *const u8,
    len: c_int,
    pcm: *mut T,
    frame_size: c_int,
    decode: impl FnOnce(&mut RsCustomDecoder<'static>, Option<&[u8]>, i32, &mut [T], i32) -> Result<i32>,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let dec = &mut unsafe { custom_decoder(st, Access::Write) }?.0;
        if len < 0 || pcm.is_null() {
            return Err(OPUS_BAD_ARG);
        }
        let frames = valid_frame_size(dec.st.mode(), frame_size)?;
        let packet = if data.is_null() {
            None
        } else {
            // SAFETY: caller contract.
            Some(unsafe { slice(data, byte_len(len)?) }?)
        };
        let n = buf_len::<T>(frames, dec.st.channels)?;
        // SAFETY: caller contract.
        let pcm = unsafe { slice_mut(pcm, n) }?;
        ret_code(decode(dec, packet, len, pcm, frames), |n| n)
    })
}

/// `opus_custom_decode_float`.
///
/// # Safety
/// See [`custom_decode`].
#[cfg(not(feature = "disable-float-api"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_decode_float(
    st: *mut OpusCustomDecoder,
    data: *const u8,
    len: c_int,
    pcm: *mut f32,
    frame_size: c_int,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        custom_decode(
            st,
            data,
            len,
            pcm,
            frame_size,
            RsCustomDecoder::opus_custom_decode_float,
        )
    }
}

/// `opus_custom_decode`.
///
/// # Safety
/// See [`custom_decode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_decode(
    st: *mut OpusCustomDecoder,
    data: *const u8,
    len: c_int,
    pcm: *mut i16,
    frame_size: c_int,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        custom_decode(
            st,
            data,
            len,
            pcm,
            frame_size,
            RsCustomDecoder::opus_custom_decode,
        )
    }
}

/// `opus_custom_decode24`.
///
/// # Safety
/// See [`custom_decode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_custom_decode24(
    st: *mut OpusCustomDecoder,
    data: *const u8,
    len: c_int,
    pcm: *mut i32,
    frame_size: c_int,
) -> c_int {
    // SAFETY: forwarded caller contract.
    unsafe {
        custom_decode(
            st,
            data,
            len,
            pcm,
            frame_size,
            RsCustomDecoder::opus_custom_decode24,
        )
    }
}
