//! libopus-internal functions (`src/opus_private.h`) that upstream's `test_opus_extensions.c`
//! links against. Only built with the `internal-api` feature; not part of the libopus ABI.

use core::ffi::c_int;

use opus::extensions::{self, Extension, ExtensionIterator};

use crate::handle::Access;
use crate::packet::{out_range, parse_impl, repacketizer};
use crate::types::OpusRepacketizer;
use crate::util::{CResult, OPUS_BAD_ARG, byte_len, code, guard, size_to_int, slice, slice_mut};

/// `OPUS_BUFFER_TOO_SMALL`.
const OPUS_BUFFER_TOO_SMALL: c_int = -2;

/// C `opus_extension_data`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OpusExtensionData {
    /// Extension ID.
    pub id: c_int,
    /// Frame index.
    pub frame: c_int,
    /// Payload.
    pub data: *const u8,
    /// Payload length.
    pub len: i32,
}

/// Converts an extension of the padding `base` to its C form (`data` points into `base`).
fn to_c(ext: &Extension<'_>, base: &[u8], base_ptr: *const u8) -> OpusExtensionData {
    // Payloads are sub-slices of the padding, so the offset is in range.
    let off = ext.data.as_ptr() as usize - base.as_ptr() as usize;
    OpusExtensionData {
        id: ext.id,
        frame: ext.frame,
        data: base_ptr.wrapping_add(off),
        len: size_to_int(ext.data.len()),
    }
}

/// Converts C extensions to Rust ones.
///
/// # Safety
/// `exts` holds `n` entries whose `data` hold `len` bytes during `'a`.
unsafe fn from_c<'a>(exts: *const OpusExtensionData, n: i32) -> CResult<Vec<Extension<'a>>> {
    // SAFETY: caller contract.
    let exts = unsafe { slice(exts, byte_len(n)?) }?;
    exts.iter()
        .map(|e| {
            // Rust extensions cannot express a negative length (C: OPUS_BAD_ARG when written).
            let len = byte_len(e.len)?;
            Ok(Extension {
                id: e.id,
                frame: e.frame,
                // SAFETY: caller contract.
                data: unsafe { slice(e.data, len) }?,
            })
        })
        .collect()
}

/// The padding argument `(data, len)` of the extension functions.
///
/// # Safety
/// `data` holds `len` bytes during `'a`.
unsafe fn padding<'a>(data: *const u8, len: i32) -> CResult<&'a [u8]> {
    // SAFETY: caller contract.
    unsafe { slice(data, byte_len(len)?) }
}

/// `opus_packet_parse_impl`.
///
/// # Safety
/// See `opus_packet_parse`; `packet_offset` / `padding` / `padding_len` are NULL or valid for
/// one write.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub unsafe extern "C" fn opus_packet_parse_impl(
    data: *const u8,
    len: i32,
    self_delimited: c_int,
    out_toc: *mut u8,
    frames: *mut *const u8,
    size: *mut i16,
    payload_offset: *mut c_int,
    packet_offset: *mut i32,
    padding: *mut *const u8,
    padding_len: *mut i32,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded caller contract.
        unsafe {
            parse_impl(
                data,
                len,
                self_delimited != 0,
                out_toc,
                frames,
                size,
                payload_offset,
                packet_offset,
                padding,
                padding_len,
            )
        }
    })
}

/// `opus_repacketizer_out_range_impl`.
///
/// # Safety
/// `rp` is a repacketizer; `data` holds `maxlen` bytes; `extensions` holds `nb_extensions`
/// entries.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub unsafe extern "C" fn opus_repacketizer_out_range_impl(
    rp: *mut OpusRepacketizer,
    begin: c_int,
    end: c_int,
    data: *mut u8,
    maxlen: i32,
    self_delimited: c_int,
    pad: c_int,
    extensions: *const OpusExtensionData,
    nb_extensions: c_int,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let st = unsafe { repacketizer(rp, Access::Read) }?;
        // SAFETY: caller contract.
        let exts = unsafe { from_c(extensions, nb_extensions) }?;
        // SAFETY: caller contract.
        unsafe {
            out_range(
                st,
                begin,
                end,
                data,
                maxlen,
                self_delimited != 0,
                pad != 0,
                &exts,
            )
        }
    })
}

/// `opus_packet_extensions_count`.
///
/// # Safety
/// `data` holds `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_extensions_count(
    data: *const u8,
    len: i32,
    nb_frames: c_int,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let d = unsafe { padding(data, len) }?;
        Ok(extensions::count(d, nb_frames))
    })
}

/// `opus_packet_extensions_count_ext`.
///
/// # Safety
/// `data` holds `len` bytes; `nb_frame_exts` holds `nb_frames` entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_extensions_count_ext(
    data: *const u8,
    len: i32,
    nb_frame_exts: *mut i32,
    nb_frames: c_int,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let d = unsafe { padding(data, len) }?;
        // SAFETY: caller contract.
        let counts = unsafe { slice_mut(nb_frame_exts, byte_len(nb_frames)?) }?;
        Ok(extensions::count_ext(d, counts))
    })
}

/// `opus_packet_extensions_parse`: extensions in bitstream order. On a malformed extension the
/// count found so far is stored and the error returned, as in C.
///
/// # Safety
/// `data` holds `len` bytes; `nb_extensions` is valid; `extensions` holds `*nb_extensions`
/// entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_extensions_parse(
    data: *const u8,
    len: i32,
    extensions: *mut OpusExtensionData,
    nb_extensions: *mut i32,
    nb_frames: c_int,
) -> i32 {
    guard(|| {
        if nb_extensions.is_null() {
            return Err(OPUS_BAD_ARG);
        }
        // SAFETY: caller contract.
        let d = unsafe { padding(data, len) }?;
        // SAFETY: non-NULL, caller contract.
        let cap = unsafe { nb_extensions.read() };
        // SAFETY: caller contract.
        let out = unsafe { slice_mut(extensions, byte_len(cap)?) }?;
        let mut iter = ExtensionIterator::new(d, nb_frames);
        let mut count = 0usize;
        let ret = loop {
            match iter.next_extension() {
                Ok(Some(ext)) => {
                    let Some(slot) = out.get_mut(count) else {
                        return Err(OPUS_BUFFER_TOO_SMALL);
                    };
                    *slot = to_c(&ext, d, data);
                    count += 1;
                }
                Ok(None) => break 0,
                Err(e) => break code(e),
            }
        };
        // SAFETY: non-NULL, caller contract.
        unsafe { nb_extensions.write(size_to_int(count)) };
        Ok(ret)
    })
}

/// `opus_packet_extensions_parse_ext`: extensions in frame order (`nb_frame_exts` from
/// `opus_packet_extensions_count_ext`).
///
/// # Safety
/// As [`opus_packet_extensions_parse`]; `nb_frame_exts` holds `nb_frames` entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_extensions_parse_ext(
    data: *const u8,
    len: i32,
    extensions: *mut OpusExtensionData,
    nb_extensions: *mut i32,
    nb_frame_exts: *const i32,
    nb_frames: c_int,
) -> i32 {
    guard(|| {
        if nb_extensions.is_null() || !(0..=48).contains(&nb_frames) {
            return Err(OPUS_BAD_ARG);
        }
        // SAFETY: caller contract.
        let d = unsafe { padding(data, len) }?;
        // SAFETY: non-NULL, caller contract.
        let cap = unsafe { nb_extensions.read() };
        // SAFETY: caller contract.
        let out = unsafe { slice_mut(extensions, byte_len(cap)?) }?;
        // SAFETY: caller contract.
        let per_frame = unsafe { slice(nb_frame_exts, nb_frames as usize) }?;
        // Convert the frame extension count array to a cumulative sum.
        let mut cum = [0i32; 49];
        let mut prev_total = 0i32;
        for (c, &n) in cum.iter_mut().zip(per_frame) {
            *c = prev_total;
            prev_total += n;
        }
        cum[per_frame.len()] = prev_total;
        let mut iter = ExtensionIterator::new(d, nb_frames);
        let mut count = 0usize;
        let ret = loop {
            match iter.next_extension() {
                Ok(Some(ext)) => {
                    let f = usize::try_from(ext.frame).map_err(|_| OPUS_BAD_ARG)?;
                    let idx = cum[f];
                    cum[f] += 1;
                    if idx >= cap {
                        return Err(OPUS_BUFFER_TOO_SMALL);
                    }
                    let slot = out
                        .get_mut(usize::try_from(idx).map_err(|_| OPUS_BAD_ARG)?)
                        .ok_or(OPUS_BAD_ARG)?;
                    *slot = to_c(&ext, d, data);
                    count += 1;
                }
                Ok(None) => break 0,
                Err(e) => break code(e),
            }
        };
        // SAFETY: non-NULL, caller contract.
        unsafe { nb_extensions.write(size_to_int(count)) };
        Ok(ret)
    })
}

/// `opus_packet_extensions_generate`: writes the extensions (or, with `data == NULL`, only
/// computes the size).
///
/// # Safety
/// `data` is NULL or holds `len` bytes; `extensions` holds `nb_extensions` entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_extensions_generate(
    data: *mut u8,
    len: i32,
    extensions: *const OpusExtensionData,
    nb_extensions: i32,
    nb_frames: c_int,
    pad: c_int,
) -> i32 {
    guard(|| {
        if nb_frames > 48 {
            return Err(OPUS_BAD_ARG);
        }
        // SAFETY: caller contract.
        let exts = unsafe { from_c(extensions, nb_extensions) }?;
        let n = byte_len(len)?;
        let r = if data.is_null() {
            extensions::generated_size(n, &exts, nb_frames, pad != 0)
        } else {
            // SAFETY: caller contract.
            let out = unsafe { slice_mut(data, n) }?;
            extensions::generate(out, &exts, nb_frames, pad != 0)
        };
        r.map(size_to_int).map_err(code)
    })
}
