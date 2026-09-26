//! `opus.h` packet helpers, soft clipping, the repacketizer and packet padding.

use core::ffi::{c_int, c_void};
use core::ptr;

use opus::Repacketizer;
use opus::extensions::Extension;
use opus::packet::{self as pkt, encode_size};
use opus::repacketizer as rpk;

use crate::handle::{self, Access, Kind, Object};
use crate::types::OpusRepacketizer;
use crate::util::{
    CResult, OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_INVALID_STATE, OPUS_OK, buf_len, byte_len,
    code, guard, guard_any, ret_code, size_to_int, slice, slice_mut,
};

/// `OPUS_INVALID_PACKET`.
const OPUS_INVALID_PACKET: c_int = -4;

// ---------------------------------------------------------------------------------------------
// Packet inspection
// ---------------------------------------------------------------------------------------------

/// The TOC byte of a packet pointer.
///
/// # Safety
/// `data` is NULL or valid for reading one byte.
const unsafe fn toc(data: *const u8) -> CResult<u8> {
    if data.is_null() {
        return Err(OPUS_BAD_ARG);
    }
    // SAFETY: non-NULL; caller contract.
    Ok(unsafe { data.read() })
}

/// A C packet `(data, len)` for the helpers that reject `len < 1` with `OPUS_BAD_ARG`.
///
/// # Safety
/// `data` is valid for reads of `len` bytes during `'a`.
unsafe fn packet_arg<'a>(data: *const u8, len: i32) -> CResult<&'a [u8]> {
    if len < 1 {
        return Err(OPUS_BAD_ARG);
    }
    // SAFETY: caller contract.
    unsafe { slice(data, byte_len(len)?) }
}

/// `opus_packet_parse`: splits a packet into its frames.
///
/// # Safety
/// `data` holds `len` bytes; `out_toc` / `payload_offset` are NULL or valid for one write;
/// `frames` is NULL or holds 48 pointers; `size` holds 48 `opus_int16`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_parse(
    data: *const u8,
    len: i32,
    out_toc: *mut u8,
    frames: *mut *const u8,
    size: *mut i16,
    payload_offset: *mut c_int,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded caller contract (no self-delimiting, no packet offset/padding).
        unsafe {
            parse_impl(
                data,
                len,
                false,
                out_toc,
                frames,
                size,
                payload_offset,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        }
    })
}

/// Body of `opus_packet_parse` / `opus_packet_parse_impl` (the latter is exported with the
/// `internal-api` feature).
///
/// # Safety
/// As [`opus_packet_parse`]; `packet_offset` / `padding_len` are NULL or valid for one write,
/// `padding` is NULL or valid for one pointer write.
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub(crate) unsafe fn parse_impl(
    data: *const u8,
    len: i32,
    self_delimited: bool,
    out_toc: *mut u8,
    frames: *mut *const u8,
    size: *mut i16,
    payload_offset: *mut c_int,
    packet_offset: *mut i32,
    padding: *mut *const u8,
    padding_len: *mut i32,
) -> CResult<c_int> {
    // Make sure we return NULL/0 on error.
    if !padding.is_null() {
        // SAFETY: caller contract.
        unsafe { padding.write(ptr::null()) };
        if !padding_len.is_null() {
            // SAFETY: caller contract.
            unsafe { padding_len.write(0) };
        }
    }
    if size.is_null() || len < 0 {
        return Err(OPUS_BAD_ARG);
    }
    if len == 0 {
        return Err(OPUS_INVALID_PACKET);
    }
    // SAFETY: caller contract.
    let bytes = unsafe { slice(data, byte_len(len)?) }?;
    let parsed = pkt::parse_impl(bytes, self_delimited).map_err(code)?;
    let base = bytes.as_ptr() as usize;
    let n = parsed.nb_frames;
    // SAFETY: caller contract (48 entries).
    let size = unsafe { slice_mut(size, n) }?;
    for (i, f) in parsed.frames().iter().enumerate() {
        size[i] = f.len() as i16;
        if !frames.is_null() {
            // Frames are sub-slices of `bytes`, so the offset is in range.
            let off = f.as_ptr() as usize - base;
            // SAFETY: caller contract (48 entries); `off <= len`.
            unsafe { frames.add(i).write(data.add(off)) };
        }
    }
    if !out_toc.is_null() {
        // SAFETY: caller contract.
        unsafe { out_toc.write(parsed.toc) };
    }
    if !payload_offset.is_null() {
        // SAFETY: caller contract.
        unsafe { payload_offset.write(size_to_int(parsed.payload_offset)) };
    }
    if !packet_offset.is_null() {
        // SAFETY: caller contract.
        unsafe { packet_offset.write(size_to_int(parsed.packet_offset)) };
    }
    if !padding.is_null() {
        let off = parsed.padding.as_ptr() as usize - base;
        // SAFETY: caller contract; the padding is a sub-slice of `bytes`.
        unsafe { padding.write(data.add(off)) };
        if !padding_len.is_null() {
            // SAFETY: caller contract.
            unsafe { padding_len.write(size_to_int(parsed.padding.len())) };
        }
    }
    Ok(size_to_int(n))
}

/// `opus_packet_get_bandwidth`.
///
/// # Safety
/// `data` points to at least one byte.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_bandwidth(data: *const u8) -> c_int {
    // SAFETY: caller contract.
    guard(|| Ok(pkt::toc_bandwidth(unsafe { toc(data) }?).to_raw()))
}

/// `opus_packet_get_samples_per_frame`.
///
/// # Safety
/// `data` points to at least one byte.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_samples_per_frame(data: *const u8, fs: i32) -> c_int {
    // SAFETY: caller contract.
    guard(|| Ok(pkt::toc_samples_per_frame(unsafe { toc(data) }?, fs)))
}

/// `opus_packet_get_nb_channels`.
///
/// # Safety
/// `data` points to at least one byte.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_nb_channels(data: *const u8) -> c_int {
    // SAFETY: caller contract.
    guard(|| Ok(pkt::toc_nb_channels(unsafe { toc(data) }?)))
}

/// `opus_packet_get_nb_frames`.
///
/// # Safety
/// `packet` holds `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_nb_frames(packet: *const u8, len: i32) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let p = unsafe { packet_arg(packet, len) }?;
        ret_code(pkt::get_nb_frames(p), |n| n)
    })
}

/// `opus_packet_get_nb_samples`.
///
/// # Safety
/// `packet` holds `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_nb_samples(packet: *const u8, len: i32, fs: i32) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let p = unsafe { packet_arg(packet, len) }?;
        ret_code(pkt::get_nb_samples(p, fs), |n| n)
    })
}

/// `opus_packet_has_lbrr`.
///
/// # Safety
/// `packet` holds `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_has_lbrr(packet: *const u8, len: i32) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let p = unsafe { slice(packet, byte_len(len)?) }?;
        ret_code(pkt::has_lbrr(p), c_int::from)
    })
}

/// `opus_pcm_soft_clip`: soft-clips float PCM to `[-1, 1]`; does nothing for invalid
/// arguments.
///
/// # Safety
/// `pcm` holds `frame_size * channels` samples and `softclip_mem` `channels` values (or they are
/// NULL).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_pcm_soft_clip(
    pcm: *mut f32,
    frame_size: c_int,
    channels: c_int,
    softclip_mem: *mut f32,
) {
    guard_any((), || {
        if channels < 1 || frame_size < 1 || pcm.is_null() || softclip_mem.is_null() {
            return;
        }
        let Ok(n) = buf_len::<f32>(frame_size, channels) else {
            return;
        };
        // SAFETY: non-NULL, caller contract.
        let (Ok(x), Ok(mem)) = (unsafe { slice_mut(pcm, n) }, unsafe {
            slice_mut(softclip_mem, channels as usize)
        }) else {
            return;
        };
        pkt::opus_pcm_soft_clip_impl(x, frame_size, channels, mem);
    });
}

// ---------------------------------------------------------------------------------------------
// Padding
// ---------------------------------------------------------------------------------------------

/// C `opus_packet_pad` argument checks: `Ok(None)` for `len == new_len` (nothing to do).
fn pad_checks(len: i32, new_len: i32) -> CResult<Option<(usize, usize)>> {
    if len < 1 {
        return Err(OPUS_BAD_ARG);
    }
    if len == new_len {
        return Ok(None);
    }
    if len > new_len {
        return Err(OPUS_BAD_ARG);
    }
    Ok(Some((byte_len(len)?, byte_len(new_len)?)))
}

/// `opus_packet_pad`: pads a packet in place to `new_len` bytes.
///
/// # Safety
/// `data` holds `max(len, new_len)` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_pad(data: *mut u8, len: i32, new_len: i32) -> c_int {
    guard(|| {
        let Some((len, new_len)) = pad_checks(len, new_len)? else {
            return Ok(OPUS_OK);
        };
        // SAFETY: caller contract.
        let buf = unsafe { slice_mut(data, new_len) }?;
        ret_code(rpk::packet_pad(buf, len, new_len), |()| OPUS_OK)
    })
}

/// `opus_packet_unpad`: removes all padding in place; returns the new length.
///
/// # Safety
/// `data` holds `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_unpad(data: *mut u8, len: i32) -> i32 {
    guard(|| {
        if len < 1 {
            return Err(OPUS_BAD_ARG);
        }
        // SAFETY: caller contract.
        let buf = unsafe { slice_mut(data, byte_len(len)?) }?;
        ret_code(rpk::packet_unpad(buf), size_to_int)
    })
}

/// `opus_multistream_packet_pad`.
///
/// # Safety
/// `data` holds `max(len, new_len)` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_packet_pad(
    data: *mut u8,
    len: i32,
    new_len: i32,
    nb_streams: c_int,
) -> c_int {
    guard(|| {
        let Some((len, new_len)) = pad_checks(len, new_len)? else {
            return Ok(OPUS_OK);
        };
        // SAFETY: caller contract.
        let buf = unsafe { slice_mut(data, new_len) }?;
        ret_code(
            rpk::multistream_packet_pad(buf, len, new_len, nb_streams),
            |()| OPUS_OK,
        )
    })
}

/// `opus_multistream_packet_unpad`.
///
/// # Safety
/// `data` holds `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_packet_unpad(
    data: *mut u8,
    len: i32,
    nb_streams: c_int,
) -> i32 {
    guard(|| {
        if len < 1 {
            return Err(OPUS_BAD_ARG);
        }
        // SAFETY: caller contract.
        let buf = unsafe { slice_mut(data, byte_len(len)?) }?;
        ret_code(rpk::multistream_packet_unpad(buf, nb_streams), size_to_int)
    })
}

// ---------------------------------------------------------------------------------------------
// Repacketizer
// ---------------------------------------------------------------------------------------------

/// One packet added with `opus_repacketizer_cat`: the addresses and lengths of its frames and
/// padding in the caller's memory.
///
/// Like libopus, the repacketizer does not copy the packets (the caller keeps them valid until
/// the next `opus_repacketizer_init`/`destroy`) and reads the frame bytes when a packet is
/// output. Rust borrows cannot span C calls, so each call rebuilds an [`opus::Repacketizer`]
/// from these records (see [`RepacketizerState::packets`]).
#[derive(Clone, Debug)]
struct CatRecord {
    frames: Vec<(usize, usize)>,
    padding: (usize, usize),
}

/// State of an `OpusRepacketizer`.
#[derive(Clone, Debug, Default)]
pub(crate) struct RepacketizerState {
    /// First TOC byte (`rp->toc`).
    toc: u8,
    nb_frames: i32,
    cats: Vec<CatRecord>,
}

/// Copies `len` bytes at C address `addr` (recorded by `cat`, valid per the libopus contract).
///
/// # Safety
/// `addr` is valid for reads of `len` bytes (or `len == 0`).
unsafe fn read_bytes(out: &mut Vec<u8>, addr: usize, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: caller contract; the bytes are copied out immediately.
    let src = unsafe { core::slice::from_raw_parts(addr as *const u8, len) };
    out.extend_from_slice(src);
}

impl RepacketizerState {
    /// The packets added so far, each re-encoded from the current bytes of its frames and
    /// padding as a code-3 VBR packet, which the Rust repacketizer parses into exactly the C
    /// state (see [`rebuild`]).
    ///
    /// # Safety
    /// Every recorded packet is still valid (libopus contract).
    unsafe fn packets(&self) -> Vec<Vec<u8>> {
        let mut bufs: Vec<Vec<u8>> = Vec::with_capacity(self.cats.len());
        for c in &self.cats {
            let n = c.frames.len();
            let pad = c.padding.1;
            let mut b = Vec::new();
            b.push((self.toc & 0xFC) | 0x3);
            b.push(n as u8 | 0x80 | if pad > 0 { 0x40 } else { 0 });
            if pad > 0 {
                b.extend(core::iter::repeat_n(255u8, pad / 254));
                b.push((pad % 254) as u8);
            }
            for &(_, len) in &c.frames[..n.saturating_sub(1)] {
                let mut sz = [0u8; 2];
                let k = encode_size(len as i32, &mut sz);
                b.extend_from_slice(&sz[..k]);
            }
            for &(addr, len) in &c.frames {
                // SAFETY: caller contract.
                unsafe { read_bytes(&mut b, addr, len) };
            }
            // SAFETY: caller contract.
            unsafe { read_bytes(&mut b, c.padding.0, pad) };
            bufs.push(b);
        }
        bufs
    }

    /// `opus_repacketizer_cat`.
    ///
    /// # Safety
    /// Recorded packets are valid; `data` stays valid until the next init/destroy.
    unsafe fn cat(&mut self, data: &[u8]) -> CResult<()> {
        // SAFETY: caller contract.
        let bufs = unsafe { self.packets() };
        let mut rp = rebuild(&bufs)?;
        rp.cat(data).map_err(code)?;
        let parsed = pkt::parse(data).map_err(code)?;
        let frames = parsed
            .frames()
            .iter()
            .map(|f| (f.as_ptr() as usize, f.len()))
            .collect();
        if self.nb_frames == 0
            && let Some(&t) = data.first()
        {
            self.toc = t;
        }
        self.cats.push(CatRecord {
            frames,
            padding: (parsed.padding.as_ptr() as usize, parsed.padding.len()),
        });
        self.nb_frames = rp.nb_frames();
        Ok(())
    }

    /// Mirror of the C struct's leading `unsigned char toc; int nb_frames;` fields.
    fn prefix(&self) -> [u8; 8] {
        let mut p = [0u8; 8];
        p[0] = self.toc;
        p[4..].copy_from_slice(&self.nb_frames.to_ne_bytes());
        p
    }
}

/// An [`opus::Repacketizer`] holding the given packets (from [`RepacketizerState::packets`]).
fn rebuild(bufs: &[Vec<u8>]) -> CResult<Repacketizer<'_>> {
    let mut rp = Repacketizer::new();
    for b in bufs {
        // The records come from packets that were accepted, so this cannot fail.
        rp.cat(b).map_err(|_| OPUS_INTERNAL_ERROR)?;
    }
    Ok(rp)
}

/// Resolves an `OpusRepacketizer*`.
///
/// # Safety
/// As [`handle::resolve`]; the reference is only used during the current call.
pub(crate) unsafe fn repacketizer<'a>(
    rp: *const OpusRepacketizer,
    access: Access,
) -> CResult<&'a mut RepacketizerState> {
    // SAFETY: caller contract.
    let r = unsafe { handle::resolve(rp.cast(), &[Kind::Repacketizer], access) }?;
    // SAFETY: resolved just now; single-threaded use per the libopus contract.
    match &mut unsafe { r.entry() }.object {
        Object::Repacketizer(s) => Ok(s),
        _ => Err(OPUS_INVALID_STATE),
    }
}

/// `opus_repacketizer_out_range_impl` on a resolved state.
///
/// # Safety
/// `data` holds `maxlen` bytes (when positive); recorded packets are valid.
#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub(crate) unsafe fn out_range(
    st: &RepacketizerState,
    begin: c_int,
    end: c_int,
    data: *mut u8,
    maxlen: i32,
    self_delimited: bool,
    pad: bool,
    extensions: &[Extension<'_>],
) -> CResult<i32> {
    if begin < 0 || begin >= end || end > st.nb_frames {
        return Err(OPUS_BAD_ARG);
    }
    // The input frames are copied before the output buffer is formed, so an output buffer
    // overlapping the input packets behaves like the C `memmove`-based code.
    // SAFETY: caller contract.
    let bufs = unsafe { st.packets() };
    let rp = rebuild(&bufs)?;
    // SAFETY: caller contract.
    let out = unsafe {
        slice_mut(
            data,
            usize::try_from(maxlen.max(0)).map_err(|_| OPUS_BAD_ARG)?,
        )
    }?;
    rp.out_range_impl(begin, end, out, self_delimited, pad, extensions)
        .map_err(code)
}

/// `opus_repacketizer_get_size`.
#[unsafe(no_mangle)]
pub extern "C" fn opus_repacketizer_get_size() -> c_int {
    guard(|| {
        Ok(size_to_int(handle::block_size(
            size_of::<RepacketizerState>(),
        )))
    })
}

/// `opus_repacketizer_init`: (re)initializes a repacketizer in caller memory of
/// `opus_repacketizer_get_size()` bytes and returns it.
///
/// # Safety
/// `rp` is NULL or valid for writes of `opus_repacketizer_get_size()` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_init(
    rp: *mut OpusRepacketizer,
) -> *mut OpusRepacketizer {
    guard_any(ptr::null_mut(), || {
        let st = RepacketizerState::default();
        let prefix = st.prefix();
        // SAFETY: caller contract.
        match unsafe { handle::install(rp.cast(), Object::Repacketizer(st)) } {
            Ok(()) => {
                // SAFETY: caller contract (installed, so non-NULL).
                unsafe { handle::write_prefix(rp.cast(), prefix) };
                rp
            }
            Err(_) => ptr::null_mut(),
        }
    })
}

/// `opus_repacketizer_create`.
#[unsafe(no_mangle)]
pub extern "C" fn opus_repacketizer_create() -> *mut OpusRepacketizer {
    // SAFETY: NULL `error` pointer.
    unsafe {
        handle::create(ptr::null_mut(), || {
            Ok((
                size_of::<RepacketizerState>(),
                Object::Repacketizer(RepacketizerState::default()),
            ))
        })
    }
}

/// `opus_repacketizer_destroy`.
///
/// # Safety
/// `rp` is NULL or a repacketizer on the C heap, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_destroy(rp: *mut OpusRepacketizer) {
    // SAFETY: caller contract.
    guard_any((), || unsafe { handle::destroy(rp.cast::<c_void>()) });
}

/// `opus_repacketizer_cat`: adds a packet (which must stay valid until the next init/destroy).
///
/// # Safety
/// `rp` is a repacketizer; `data` holds `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_cat(
    rp: *mut OpusRepacketizer,
    data: *const u8,
    len: i32,
) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let st = unsafe { repacketizer(rp, Access::Write) }?;
        if len < 1 {
            return Err(OPUS_INVALID_PACKET);
        }
        // SAFETY: caller contract.
        let bytes = unsafe { slice(data, byte_len(len)?) }?;
        // SAFETY: caller contract.
        unsafe { st.cat(bytes) }?;
        // SAFETY: caller contract (resolved, so non-NULL and writable).
        unsafe { handle::write_prefix(rp.cast(), st.prefix()) };
        Ok(OPUS_OK)
    })
}

/// `opus_repacketizer_out_range`.
///
/// # Safety
/// `rp` is a repacketizer; `data` holds `maxlen` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_out_range(
    rp: *mut OpusRepacketizer,
    begin: c_int,
    end: c_int,
    data: *mut u8,
    maxlen: i32,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let st = unsafe { repacketizer(rp, Access::Read) }?;
        // SAFETY: caller contract.
        unsafe { out_range(st, begin, end, data, maxlen, false, false, &[]) }
    })
}

/// `opus_repacketizer_get_nb_frames`.
///
/// # Safety
/// `rp` is a repacketizer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_get_nb_frames(rp: *mut OpusRepacketizer) -> c_int {
    guard(|| {
        // SAFETY: caller contract.
        let st = unsafe { repacketizer(rp, Access::Read) }?;
        Ok(st.nb_frames)
    })
}

/// `opus_repacketizer_out`.
///
/// # Safety
/// `rp` is a repacketizer; `data` holds `maxlen` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_out(
    rp: *mut OpusRepacketizer,
    data: *mut u8,
    maxlen: i32,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let st = unsafe { repacketizer(rp, Access::Read) }?;
        // SAFETY: caller contract.
        unsafe { out_range(st, 0, st.nb_frames, data, maxlen, false, false, &[]) }
    })
}
