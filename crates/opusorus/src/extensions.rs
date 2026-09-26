//! Opus packet extensions (carried in the packet padding), including the "Repeat These
//! Extensions" mechanism of libopus 1.6.
//!
//! Port of `src/extensions.c` and the `OpusExtensionIterator` / `opus_extension_data` types of
//! `src/opus_private.h`.
//!
//! Extension payload pointers become borrowed slices ([`Extension::data`]); the C `len` field
//! is the slice length. Iterator pointers become byte offsets into the padding buffer.

use crate::packet::len_i32;
use crate::{Error, Result};

/// One packet extension (`opus_extension_data`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Extension<'a> {
    /// Extension ID (3..=127 for real extensions).
    pub id: i32,
    /// Index of the frame the extension belongs to.
    pub frame: i32,
    /// Extension payload. IDs below 32 carry at most one byte.
    pub data: &'a [u8],
}

impl Extension<'_> {
    /// Payload length as an `opus_int32` (C `len` field).
    #[inline]
    const fn len(&self) -> i32 {
        len_i32(self.data.len())
    }
}

/// Port of `src/extensions.c:skip_extension_payload` (static).
///
/// Given an extension payload (i.e., excluding the initial ID byte) at `*pdata` in `buf`,
/// advance `*pdata` to the next extension and return the length of the remaining extensions.
/// A "Repeat These Extensions" extension (ID==2) does not advance past the repeated extension
/// payloads; that requires higher-level logic.
const fn skip_extension_payload(
    buf: &[u8],
    pdata: &mut usize,
    mut len: i32,
    pheader_size: &mut i32,
    id_byte: i32,
    trailing_short_len: i32,
) -> i32 {
    let mut data = *pdata;
    let mut header_size = 0;
    let id = id_byte >> 1;
    let l = id_byte & 1;
    if (id == 0 && l == 1) || id == 2 {
        // Nothing to do.
    } else if id > 0 && id < 32 {
        if len < l {
            return -1;
        }
        data += l as usize;
        len -= l;
    } else if l == 0 {
        if len < trailing_short_len {
            return -1;
        }
        data += (len - trailing_short_len) as usize;
        len = trailing_short_len;
    } else {
        let mut bytes: i32 = 0;
        loop {
            if len < 1 {
                return -1;
            }
            let lacing = buf[data] as i32;
            data += 1;
            bytes += lacing;
            header_size += 1;
            len -= lacing + 1;
            if lacing != 255 {
                break;
            }
        }
        if len < 0 {
            return -1;
        }
        data += bytes as usize;
    }
    *pdata = data;
    *pheader_size = header_size;
    len
}

/// Port of `src/extensions.c:skip_extension` (static).
///
/// Given an extension at `*pdata`, advance to the next extension and return the length of the
/// remaining extensions. A "Repeat These Extensions" extension (ID==2) only advances past the
/// extension ID byte.
const fn skip_extension(
    buf: &[u8],
    pdata: &mut usize,
    mut len: i32,
    pheader_size: &mut i32,
) -> i32 {
    if len == 0 {
        *pheader_size = 0;
        return 0;
    }
    if len < 1 {
        return -1;
    }
    let mut data = *pdata;
    let id_byte = buf[data] as i32;
    data += 1;
    len -= 1;
    len = skip_extension_payload(buf, &mut data, len, pheader_size, id_byte, 0);
    if len >= 0 {
        *pdata = data;
        *pheader_size += 1;
    }
    len
}

/// Iterator over the extensions in a packet's padding (`OpusExtensionIterator`).
///
/// Yields extensions (excluding real padding, separators and repeat indicators, but including
/// repeated extensions) in bitstream order, which due to the repetition mechanism is not
/// necessarily frame order.
#[derive(Debug, Clone)]
pub struct ExtensionIterator<'a> {
    data: &'a [u8],
    curr_data: usize,
    repeat_data: usize,
    last_long: Option<usize>,
    src_data: usize,
    len: i32,
    curr_len: i32,
    repeat_len: i32,
    src_len: i32,
    trailing_short_len: i32,
    nb_frames: i32,
    frame_max: i32,
    curr_frame: i32,
    repeat_frame: i32,
    repeat_l: i32,
}

impl<'a> ExtensionIterator<'a> {
    /// Port of `src/extensions.c:opus_extension_iterator_init`: iterates over the extensions in
    /// `data` (a packet's padding) for a packet of `nb_frames` frames (0..=48).
    #[must_use]
    pub const fn new(data: &'a [u8], nb_frames: i32) -> Self {
        debug_assert!(nb_frames >= 0 && nb_frames <= 48);
        let len = len_i32(data.len());
        Self {
            data,
            curr_data: 0,
            repeat_data: 0,
            last_long: None,
            src_data: 0,
            len,
            curr_len: len,
            repeat_len: 0,
            src_len: 0,
            trailing_short_len: 0,
            nb_frames,
            frame_max: nb_frames,
            curr_frame: 0,
            repeat_frame: 0,
            repeat_l: 0,
        }
    }

    /// Port of `src/extensions.c:opus_extension_iterator_reset`: restarts the iteration from
    /// the first extension.
    pub const fn reset(&mut self) {
        self.repeat_data = 0;
        self.curr_data = 0;
        self.last_long = None;
        self.curr_len = self.len;
        self.repeat_frame = 0;
        self.curr_frame = 0;
        self.trailing_short_len = 0;
    }

    /// Port of `src/extensions.c:opus_extension_iterator_set_frame_max`: do not return any
    /// extensions for frames of index `frame_max` or larger (lets iteration stop early).
    pub const fn set_frame_max(&mut self, frame_max: i32) {
        self.frame_max = frame_max;
    }

    /// Port of `src/extensions.c:opus_extension_iterator_next_repeat` (static).
    ///
    /// Returns the next repeated extension, `Ok(None)` when repeating is finished.
    fn next_repeat(&mut self) -> Result<Option<Extension<'a>>> {
        let data: &'a [u8] = self.data;
        let mut header_size: i32 = 0;
        debug_assert!(self.repeat_frame > 0);
        while self.repeat_frame < self.nb_frames {
            while self.src_len > 0 {
                let mut repeat_id_byte = data[self.src_data] as i32;
                self.src_len =
                    skip_extension(data, &mut self.src_data, self.src_len, &mut header_size);
                // We skipped this extension earlier, so it should not fail now.
                debug_assert!(self.src_len >= 0);
                // Don't repeat padding or frame separators with a 0 increment.
                if repeat_id_byte <= 3 {
                    continue;
                }
                // If the "Repeat These Extensions" extension had L == 0 and this
                // is the last repeated long extension, then force decoding the
                // payload with L = 0.
                if self.repeat_l == 0
                    && self.repeat_frame + 1 >= self.nb_frames
                    && Some(self.src_data) == self.last_long
                {
                    repeat_id_byte &= !1;
                }
                let curr_data0 = self.curr_data;
                self.curr_len = skip_extension_payload(
                    data,
                    &mut self.curr_data,
                    self.curr_len,
                    &mut header_size,
                    repeat_id_byte,
                    self.trailing_short_len,
                );
                if self.curr_len < 0 {
                    return Err(Error::InvalidPacket);
                }
                debug_assert!(self.curr_data as i32 == self.len - self.curr_len);
                // If we were asked to stop at frame_max, skip extensions for later
                // frames.
                if self.repeat_frame >= self.frame_max {
                    continue;
                }
                return Ok(Some(Extension {
                    id: repeat_id_byte >> 1,
                    frame: self.repeat_frame,
                    data: &data[curr_data0 + header_size as usize..self.curr_data],
                }));
            }
            // We finished repeating the extensions for this frame.
            self.src_data = self.repeat_data;
            self.src_len = self.repeat_len;
            self.repeat_frame += 1;
        }
        // We finished repeating extensions.
        self.repeat_data = self.curr_data;
        self.last_long = None;
        // If L == 0, advance the frame number to handle the case where we did
        // not consume all of the data with an L == 0 long extension.
        if self.repeat_l == 0 {
            self.curr_frame += 1;
            // Ignore additional padding if this was already the last frame.
            if self.curr_frame >= self.nb_frames {
                self.curr_len = 0;
            }
        }
        self.repeat_frame = 0;
        Ok(None)
    }

    /// Port of `src/extensions.c:opus_extension_iterator_next`: returns the next extension
    /// (excluding real padding, separators, and repeat indicators, but including the repeated
    /// extensions) in bitstream order, or `Ok(None)` when there are no more.
    ///
    /// # Errors
    /// [`Error::InvalidPacket`] if the extension data is malformed (the iterator then keeps
    /// returning this error).
    pub fn next_extension(&mut self) -> Result<Option<Extension<'a>>> {
        let data: &'a [u8] = self.data;
        let mut header_size: i32 = 0;
        if self.curr_len < 0 {
            return Err(Error::InvalidPacket);
        }
        if self.repeat_frame > 0 {
            // We are in the process of repeating some extensions.
            if let Some(ext) = self.next_repeat()? {
                return Ok(Some(ext));
            }
        }
        // Checking this here allows set_frame_max() to be called at any point.
        if self.curr_frame >= self.frame_max {
            return Ok(None);
        }
        while self.curr_len > 0 {
            let curr_data0 = self.curr_data;
            let id = (data[curr_data0] >> 1) as i32;
            let l = (data[curr_data0] & 1) as i32;
            self.curr_len =
                skip_extension(data, &mut self.curr_data, self.curr_len, &mut header_size);
            if self.curr_len < 0 {
                return Err(Error::InvalidPacket);
            }
            debug_assert!(self.curr_data as i32 == self.len - self.curr_len);
            if id == 1 {
                if l == 0 {
                    self.curr_frame += 1;
                } else {
                    // A frame increment of 0 is a no-op.
                    let inc = data[curr_data0 + 1];
                    if inc == 0 {
                        continue;
                    }
                    self.curr_frame += inc as i32;
                }
                if self.curr_frame >= self.nb_frames {
                    self.curr_len = -1;
                    return Err(Error::InvalidPacket);
                }
                // If we were asked to stop at frame_max, skip extensions for later
                // frames.
                if self.curr_frame >= self.frame_max {
                    self.curr_len = 0;
                }
                self.repeat_data = self.curr_data;
                self.last_long = None;
                self.trailing_short_len = 0;
            } else if id == 2 {
                self.repeat_l = l;
                self.repeat_frame = self.curr_frame + 1;
                self.repeat_len = (curr_data0 - self.repeat_data) as i32;
                self.src_data = self.repeat_data;
                self.src_len = self.repeat_len;
                if let Some(ext) = self.next_repeat()? {
                    return Ok(Some(ext));
                }
            } else if id > 2 {
                // Update the location of the last long extension.
                // This lets us know when we need to modify the last L flag if we
                // repeat these extensions with L=0.
                if id >= 32 {
                    self.last_long = Some(self.curr_data);
                    self.trailing_short_len = 0;
                } else {
                    // Otherwise, keep track of how many payload bytes follow the last
                    // long extension.
                    self.trailing_short_len += l;
                }
                return Ok(Some(Extension {
                    id,
                    frame: self.curr_frame,
                    data: &data[curr_data0 + header_size as usize..self.curr_data],
                }));
            }
        }
        Ok(None)
    }

    /// Port of `src/extensions.c:opus_extension_iterator_find`: returns the next extension
    /// with the given `id`, or `Ok(None)` if there is none.
    ///
    /// # Errors
    /// [`Error::InvalidPacket`] if the extension data is malformed.
    pub fn find(&mut self, id: i32) -> Result<Option<Extension<'a>>> {
        loop {
            match self.next_extension()? {
                None => return Ok(None),
                Some(ext) if ext.id == id => return Ok(Some(ext)),
                Some(_) => {}
            }
        }
    }
}

impl<'a> Iterator for ExtensionIterator<'a> {
    type Item = Result<Extension<'a>>;

    /// Yields `Ok(extension)` items; a malformed extension yields one `Err` and then ends.
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr_len < 0 {
            // Every error path leaves `curr_len < 0`: the error was already reported.
            return None;
        }
        self.next_extension().transpose()
    }
}

/// Port of `src/extensions.c:opus_packet_extensions_count`: counts the extensions in `data`
/// (excluding real padding, separators, and repeat indicators, but including the repeated
/// extensions). Counting stops silently at the first malformed extension, as in libopus.
#[must_use]
pub fn count(data: &[u8], nb_frames: i32) -> i32 {
    let mut iter = ExtensionIterator::new(data, nb_frames);
    let mut count = 0;
    while let Ok(Some(_)) = iter.next_extension() {
        count += 1;
    }
    count
}

/// Port of `src/extensions.c:opus_packet_extensions_count_ext`: like [`count`], also counting
/// the extensions of each frame into `nb_frame_exts` (whose length is the packet's number of
/// frames).
pub fn count_ext(data: &[u8], nb_frame_exts: &mut [i32]) -> i32 {
    let nb_frames = len_i32(nb_frame_exts.len());
    let mut iter = ExtensionIterator::new(data, nb_frames);
    nb_frame_exts.fill(0);
    let mut count = 0;
    while let Ok(Some(ext)) = iter.next_extension() {
        nb_frame_exts[ext.frame as usize] += 1;
        count += 1;
    }
    count
}

/// Port of `src/extensions.c:opus_packet_extensions_parse`: extracts the extensions of `data`
/// in bitstream order into `extensions`, returning how many were found.
///
/// # Errors
/// [`Error::BufferTooSmall`] if `extensions` cannot hold them all, [`Error::InvalidPacket`] if
/// the extension data is malformed.
pub fn parse<'a>(
    data: &'a [u8],
    extensions: &mut [Extension<'a>],
    nb_frames: i32,
) -> Result<usize> {
    let mut iter = ExtensionIterator::new(data, nb_frames);
    let mut count = 0usize;
    while let Some(ext) = iter.next_extension()? {
        let Some(slot) = extensions.get_mut(count) else {
            return Err(Error::BufferTooSmall);
        };
        *slot = ext;
        count += 1;
    }
    Ok(count)
}

/// Port of `src/extensions.c:opus_packet_extensions_parse_ext`: extracts the extensions of
/// `data` in frame order into `extensions`. `nb_frame_exts` (one entry per frame of the packet)
/// must be filled by [`count_ext`].
///
/// # Errors
/// [`Error::BufferTooSmall`] if `extensions` cannot hold them all, [`Error::InvalidPacket`] if
/// the extension data is malformed, [`Error::BadArg`] for more than 48 frames or inconsistent
/// `nb_frame_exts` (C asserts / undefined behaviour).
pub fn parse_ext<'a>(
    data: &'a [u8],
    extensions: &mut [Extension<'a>],
    nb_frame_exts: &[i32],
) -> Result<usize> {
    let nb_frames = nb_frame_exts.len();
    if nb_frames > 48 {
        return Err(Error::BadArg);
    }
    let mut nb_frames_cum = [0i32; 49];
    // Convert the frame extension count array to a cumulative sum.
    let mut prev_total = 0i32;
    for (cum, &n) in nb_frames_cum.iter_mut().zip(nb_frame_exts) {
        let total = n + prev_total;
        *cum = prev_total;
        prev_total = total;
    }
    nb_frames_cum[nb_frames] = prev_total;
    let mut iter = ExtensionIterator::new(data, nb_frames as i32);
    let nb_extensions = len_i32(extensions.len());
    let mut count = 0usize;
    while let Some(ext) = iter.next_extension()? {
        let f = ext.frame as usize;
        let idx = nb_frames_cum[f];
        nb_frames_cum[f] += 1;
        if idx >= nb_extensions {
            return Err(Error::BufferTooSmall);
        }
        debug_assert!(idx < nb_frames_cum[f + 1]);
        let Ok(idx) = usize::try_from(idx) else {
            return Err(Error::BadArg);
        };
        extensions[idx] = ext;
        count += 1;
    }
    Ok(count)
}

/// Optional output buffer of the extension writer (C: `data` may be NULL to only measure).
struct Out<'b>(Option<&'b mut [u8]>);

impl Out<'_> {
    #[inline]
    fn put(&mut self, pos: i32, v: u8) {
        if let Some(d) = self.0.as_deref_mut() {
            d[pos as usize] = v;
        }
    }
}

/// Port of `src/extensions.c:write_extension_payload` (static).
fn write_extension_payload(
    data: &mut Out<'_>,
    len: i32,
    mut pos: i32,
    ext: &Extension<'_>,
    last: bool,
) -> Result<i32> {
    debug_assert!(ext.id >= 3 && ext.id <= 127);
    let ext_len = ext.len();
    if ext.id < 32 {
        if !(0..=1).contains(&ext_len) {
            return Err(Error::BadArg);
        }
        if ext_len > 0 {
            if len - pos < ext_len {
                return Err(Error::BufferTooSmall);
            }
            data.put(pos, ext.data[0]);
            pos += 1;
        }
    } else {
        // `ext->len < 0` (OPUS_BAD_ARG) cannot happen with slices.
        let mut length_bytes = 1 + ext_len / 255;
        if last {
            length_bytes = 0;
        }
        if len - pos < length_bytes + ext_len {
            return Err(Error::BufferTooSmall);
        }
        if !last {
            for _ in 0..ext_len / 255 {
                data.put(pos, 255);
                pos += 1;
            }
            data.put(pos, (ext_len % 255) as u8);
            pos += 1;
        }
        if let Some(d) = data.0.as_deref_mut() {
            d[pos as usize..(pos + ext_len) as usize].copy_from_slice(ext.data);
        }
        pos += ext_len;
    }
    Ok(pos)
}

/// Port of `src/extensions.c:write_extension` (static).
fn write_extension(
    data: &mut Out<'_>,
    len: i32,
    mut pos: i32,
    ext: &Extension<'_>,
    last: bool,
) -> Result<i32> {
    if len - pos < 1 {
        return Err(Error::BufferTooSmall);
    }
    debug_assert!(ext.id >= 3 && ext.id <= 127);
    let flag = if ext.id < 32 {
        ext.len()
    } else {
        i32::from(!last)
    };
    // C stores `(id<<1) + flag` into an unsigned char (truncating).
    data.put(pos, ((ext.id << 1) + flag) as u8);
    pos += 1;
    write_extension_payload(data, len, pos, ext, last)
}

/// Port of `src/extensions.c:opus_packet_extensions_generate` with the C calling convention:
/// `data == None` only computes the size. Writes at most `len` bytes and returns the number of
/// bytes produced (exactly `len` when `pad` is set and the extensions fit).
///
/// # Errors
/// [`Error::BadArg`] for invalid extensions (ID outside 3..=127, frame outside
/// `0..nb_frames`, short extension with more than one byte), more than 48 frames, or
/// `data` shorter than `len`; [`Error::BufferTooSmall`] if they do not fit in `len` bytes.
#[expect(
    clippy::too_many_lines,
    reason = "straight port of one C function; splitting it would obscure the correspondence"
)]
pub fn generate_impl(
    data: Option<&mut [u8]>,
    len: i32,
    extensions: &[Extension<'_>],
    nb_frames: i32,
    pad: bool,
) -> Result<i32> {
    let mut frame_min_idx = [0i32; 48];
    let mut frame_max_idx = [0i32; 48];
    let mut frame_repeat_idx = [0i32; 48];
    let mut curr_frame: i32 = 0;
    let mut pos: i32 = 0;
    let mut written: i32 = 0;
    let nb_extensions = len_i32(extensions.len());

    debug_assert!(len >= 0);
    if let Some(d) = &data
        && (d.len() as u64) < len.max(0) as u64
    {
        return Err(Error::BadArg);
    }
    let mut data = Out(data);
    if nb_frames > 48 {
        return Err(Error::BadArg);
    }
    let nbf = nb_frames.max(0) as usize;
    let ext = |i: i32| &extensions[i as usize];

    // Do a little work up-front to make this O(nb_extensions) instead of
    // O(nb_extensions*nb_frames) so long as the extensions are in frame
    // order (without requiring that they be in frame order).
    frame_min_idx[..nbf].fill(nb_extensions);
    frame_max_idx[..nbf].fill(0);
    for (i, e) in extensions.iter().enumerate() {
        let i = i as i32;
        let f = e.frame;
        if f < 0 || f >= nb_frames {
            return Err(Error::BadArg);
        }
        if e.id < 3 || e.id > 127 {
            return Err(Error::BadArg);
        }
        let f = f as usize;
        frame_min_idx[f] = frame_min_idx[f].min(i);
        frame_max_idx[f] = frame_max_idx[f].max(i + 1);
    }
    frame_repeat_idx[..nbf].copy_from_slice(&frame_min_idx[..nbf]);
    for f in 0..nb_frames {
        let fu = f as usize;
        let mut repeat_count = 0;
        let mut last_long_idx: i32 = -1;
        if f + 1 < nb_frames {
            let mut i = frame_min_idx[fu];
            while i < frame_max_idx[fu] {
                if ext(i).frame == f {
                    // Test if we can repeat this extension in future frames.
                    let mut g = f + 1;
                    while g < nb_frames {
                        let gu = g as usize;
                        if frame_repeat_idx[gu] >= frame_max_idx[gu] {
                            break;
                        }
                        let rep = ext(frame_repeat_idx[gu]);
                        debug_assert!(rep.frame == g);
                        if rep.id != ext(i).id {
                            break;
                        }
                        if rep.id < 32 && rep.len() != ext(i).len() {
                            break;
                        }
                        g += 1;
                    }
                    if g < nb_frames {
                        break;
                    }
                    // We can!
                    // If this is a long extension, save the index of the last
                    // instance, so we can modify its L flag.
                    if ext(i).id >= 32 {
                        last_long_idx = frame_repeat_idx[(nb_frames - 1) as usize];
                    }
                    // Using the repeat mechanism almost always makes the encoding smaller
                    // (or at least no larger); libopus always uses it when possible.
                    // Advance the repeat pointers.
                    for g in (f + 1)..nb_frames {
                        let gu = g as usize;
                        let mut j = frame_repeat_idx[gu] + 1;
                        while j < frame_max_idx[gu] && ext(j).frame != g {
                            j += 1;
                        }
                        frame_repeat_idx[gu] = j;
                    }
                    repeat_count += 1;
                    // Point the repeat pointer for this frame to the current
                    // extension, so we know when to trigger the repeats.
                    frame_repeat_idx[fu] = i;
                }
                i += 1;
            }
        }
        for i in frame_min_idx[fu]..frame_max_idx[fu] {
            if ext(i).frame != f {
                continue;
            }
            // Insert separator when needed.
            if f != curr_frame {
                let diff = f - curr_frame;
                if len - pos < 2 {
                    return Err(Error::BufferTooSmall);
                }
                if diff == 1 {
                    data.put(pos, 0x02);
                    pos += 1;
                } else {
                    data.put(pos, 0x03);
                    pos += 1;
                    data.put(pos, diff as u8);
                    pos += 1;
                }
                curr_frame = f;
            }

            pos = write_extension(&mut data, len, pos, ext(i), written == nb_extensions - 1)?;
            written += 1;

            if repeat_count > 0 && frame_repeat_idx[fu] == i {
                // Add the repeat indicator.
                let nb_repeated = repeat_count * (nb_frames - (f + 1));
                let last = written + nb_repeated == nb_extensions
                    || (last_long_idx < 0 && i + 1 >= frame_max_idx[fu]);
                if len - pos < 1 {
                    return Err(Error::BufferTooSmall);
                }
                data.put(pos, 0x04 + u8::from(!last));
                pos += 1;
                for g in (f + 1)..nb_frames {
                    let gu = g as usize;
                    let mut j = frame_min_idx[gu];
                    while j < frame_repeat_idx[gu] {
                        if ext(j).frame == g {
                            pos = write_extension_payload(
                                &mut data,
                                len,
                                pos,
                                ext(j),
                                last && j == last_long_idx,
                            )?;
                            written += 1;
                        }
                        j += 1;
                    }
                    frame_min_idx[gu] = j;
                }
                if last {
                    curr_frame += 1;
                }
            }
        }
    }
    debug_assert!(written == nb_extensions);
    // If we need to pad, just prepend 0x01 bytes. Even better would be to fill the
    // end with zeros, but that requires checking that turning the last extension into
    // an L=1 case still fits.
    if pad && pos < len {
        let padding = len - pos;
        if let Some(d) = data.0.as_deref_mut() {
            d.copy_within(0..pos as usize, padding as usize);
            d[..padding as usize].fill(0x01);
        }
        pos += padding;
    }
    Ok(pos)
}

/// Port of `src/extensions.c:opus_packet_extensions_generate`: writes `extensions` (for a
/// packet of `nb_frames` frames) into `out` and returns the number of bytes written. With
/// `pad`, the output is padded with leading `0x01` bytes to fill all of `out`.
///
/// # Errors
/// See [`generate_impl`].
pub fn generate(
    out: &mut [u8],
    extensions: &[Extension<'_>],
    nb_frames: i32,
    pad: bool,
) -> Result<usize> {
    let len = len_i32(out.len());
    generate_impl(Some(out), len, extensions, nb_frames, pad).map(|n| n as usize)
}

/// Port of `src/extensions.c:opus_packet_extensions_generate` with a NULL output buffer:
/// returns the number of bytes [`generate`] would write into a buffer of `max_len` bytes.
///
/// # Errors
/// See [`generate_impl`].
pub fn generated_size(
    max_len: usize,
    extensions: &[Extension<'_>],
    nb_frames: i32,
    pad: bool,
) -> Result<usize> {
    generate_impl(None, len_i32(max_len), extensions, nb_frames, pad).map(|n| n as usize)
}
