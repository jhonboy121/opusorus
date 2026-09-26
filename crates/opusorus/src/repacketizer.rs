//! Repacketizer: merges frames of several Opus packets into one packet or splits a packet,
//! plus packet padding / unpadding helpers.
//!
//! Port of `src/repacketizer.c` (`OpusRepacketizer`, `opus_repacketizer_*`,
//! `opus_packet_pad`/`opus_packet_unpad`, `opus_multistream_packet_pad`/`unpad`).
//!
//! Like libopus, a [`Repacketizer`] does not copy the packets it is given: it borrows them for
//! its lifetime `'a`. libopus pads/unpads in place, relying on `memmove` semantics while frames
//! point into the output buffer; the Rust port first copies the input packet (as
//! `opus_packet_pad` does in C), which produces identical output.

use alloc::vec;
use alloc::vec::Vec;

use crate::extensions::{self, Extension};
use crate::packet::{self, MAX_FRAMES, encode_size, len_i32, toc_samples_per_frame};
use crate::{Error, Result};

/// Repacketizer state (`OpusRepacketizer`).
///
/// Frames are borrowed from the packets passed to [`Repacketizer::cat`], so those packets must
/// outlive the repacketizer (as in libopus, where they must stay valid until the next
/// `opus_repacketizer_init`).
#[derive(Debug, Clone)]
pub struct Repacketizer<'a> {
    toc: u8,
    nb_frames: i32,
    /// Frame payloads; `len[i]` of the C struct is `frames[i].len()`.
    frames: [&'a [u8]; MAX_FRAMES],
    framesize: i32,
    /// Padding of the packet each frame came from (only for the first frame of each packet);
    /// `padding_len[i]` of the C struct is `paddings[i].len()`.
    paddings: [&'a [u8]; MAX_FRAMES],
    padding_nb_frames: [u8; MAX_FRAMES],
}

impl Default for Repacketizer<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Repacketizer<'a> {
    /// Port of `src/repacketizer.c:opus_repacketizer_create` / `opus_repacketizer_init`: a new
    /// empty repacketizer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            toc: 0,
            nb_frames: 0,
            frames: [&[]; MAX_FRAMES],
            framesize: 0,
            paddings: [&[]; MAX_FRAMES],
            padding_nb_frames: [0; MAX_FRAMES],
        }
    }

    /// Port of `src/repacketizer.c:opus_repacketizer_init`: resets the repacketizer so it can
    /// accept packets with a different TOC.
    pub const fn init(&mut self) {
        self.nb_frames = 0;
    }

    /// Port of `src/repacketizer.c:opus_repacketizer_cat_impl` (static): adds a packet,
    /// optionally self-delimited.
    ///
    /// # Errors
    /// [`Error::InvalidPacket`] if the packet is malformed, its TOC is incompatible with the
    /// frames already added, or the total would exceed 120 ms.
    pub fn cat_impl(&mut self, data: &'a [u8], self_delimited: bool) -> Result<()> {
        // Set of check ToC
        let Some(&toc) = data.first() else {
            return Err(Error::InvalidPacket);
        };
        if self.nb_frames == 0 {
            self.toc = toc;
            self.framesize = toc_samples_per_frame(toc, 8000);
        } else if (self.toc & 0xFC) != (toc & 0xFC) {
            return Err(Error::InvalidPacket);
        }
        let mut curr_nb_frames = match packet::get_nb_frames(data) {
            Ok(n) if n >= 1 => n,
            _ => return Err(Error::InvalidPacket),
        };

        // Check the 120 ms maximum packet size
        if (curr_nb_frames + self.nb_frames) * self.framesize > 960 {
            return Err(Error::InvalidPacket);
        }

        let parsed = packet::parse_impl(data, self_delimited)?;
        let base = self.nb_frames as usize;
        let n = parsed.nb_frames;
        self.frames[base..base + n].copy_from_slice(&parsed.frames[..n]);
        self.paddings[base] = parsed.padding;
        self.padding_nb_frames[base] = n as u8;

        // set padding length to zero for all but the first frame
        while curr_nb_frames > 1 {
            self.nb_frames += 1;
            let i = self.nb_frames as usize;
            self.paddings[i] = &[];
            self.padding_nb_frames[i] = 0;
            curr_nb_frames -= 1;
        }
        self.nb_frames += 1;
        Ok(())
    }

    /// Port of `src/repacketizer.c:opus_repacketizer_cat`: adds a packet to the repacketizer.
    /// All packets must have the same TOC configuration (mode, bandwidth, frame size, channel
    /// count), and the total duration may not exceed 120 ms.
    ///
    /// # Errors
    /// [`Error::InvalidPacket`] as for [`Repacketizer::cat_impl`].
    pub fn cat(&mut self, data: &'a [u8]) -> Result<()> {
        self.cat_impl(data, false)
    }

    /// Port of `src/repacketizer.c:opus_repacketizer_get_nb_frames`: total number of frames
    /// added since the last [`Repacketizer::init`].
    #[must_use]
    pub const fn nb_frames(&self) -> i32 {
        self.nb_frames
    }

    /// Discards the padding (and therefore extensions) of every frame, as
    /// `opus_packet_unpad` does before re-emitting the packet.
    fn discard_padding(&mut self) {
        for p in &mut self.paddings[..self.nb_frames as usize] {
            *p = &[];
        }
    }

    /// Port of `src/repacketizer.c:opus_repacketizer_out_range_impl`: writes frames
    /// `begin..end` into `data` (whose length is the C `maxlen`) as one packet, optionally
    /// self-delimited, optionally padded to fill `data`, adding `extensions` plus the
    /// extensions found in the source packets' padding. Returns the packet length.
    ///
    /// # Errors
    /// [`Error::BadArg`] for an invalid range, [`Error::BufferTooSmall`] if the packet does not
    /// fit, [`Error::InternalError`] if a source packet's extensions are malformed, and the
    /// errors of [`extensions::generate_impl`].
    #[expect(
        clippy::too_many_lines,
        reason = "straight port of one C function; splitting it would obscure the correspondence"
    )]
    pub fn out_range_impl(
        &self,
        begin: i32,
        end: i32,
        data: &mut [u8],
        self_delimited: bool,
        pad: bool,
        extensions: &[Extension<'_>],
    ) -> Result<i32> {
        let maxlen = len_i32(data.len());
        let mut ones_begin: i32 = 0;
        let mut ones_end: i32 = 0;
        let mut ext_begin: i32 = 0;
        let mut ext_len: i32 = 0;

        if begin < 0 || begin >= end || end > self.nb_frames {
            return Err(Error::BadArg);
        }
        let count = (end - begin) as usize;
        let (b, e) = (begin as usize, end as usize);

        let frames = &self.frames[b..e];
        let len = |i: usize| len_i32(frames[i].len());
        let mut tot_size: i32 = if self_delimited {
            1 + i32::from(len(count - 1) >= 252)
        } else {
            0
        };

        // figure out total number of extensions
        let mut total_ext_count = len_i32(extensions.len());
        for i in b..e {
            let n = extensions::count(self.paddings[i], i32::from(self.padding_nb_frames[i]));
            if n > 0 {
                total_ext_count += n;
            }
        }
        let mut all_extensions: Vec<Extension<'_>> = if total_ext_count > 0 {
            vec![Extension::default(); total_ext_count as usize]
        } else {
            Vec::new()
        };
        // copy over any extensions that were passed in
        all_extensions[..extensions.len()].copy_from_slice(extensions);
        let mut ext_count = extensions.len();

        // incorporate any extensions from the repacketizer padding
        for i in b..e {
            let Ok(frame_ext_count) = extensions::parse(
                self.paddings[i],
                &mut all_extensions[ext_count..],
                i32::from(self.padding_nb_frames[i]),
            ) else {
                return Err(Error::InternalError);
            };
            // renumber the extension frame numbers
            for x in &mut all_extensions[ext_count..ext_count + frame_ext_count] {
                x.frame += (i - b) as i32;
            }
            ext_count += frame_ext_count;
        }
        let all_extensions = &all_extensions[..ext_count];

        let mut ptr: usize = 0;
        if count == 1 {
            // Code 0
            tot_size += len(0) + 1;
            if tot_size > maxlen {
                return Err(Error::BufferTooSmall);
            }
            data[ptr] = self.toc & 0xFC;
            ptr += 1;
        } else if count == 2 {
            if len(1) == len(0) {
                // Code 1
                tot_size += 2 * len(0) + 1;
                if tot_size > maxlen {
                    return Err(Error::BufferTooSmall);
                }
                data[ptr] = (self.toc & 0xFC) | 0x1;
                ptr += 1;
            } else {
                // Code 2
                tot_size += len(0) + len(1) + 2 + i32::from(len(0) >= 252);
                if tot_size > maxlen {
                    return Err(Error::BufferTooSmall);
                }
                data[ptr] = (self.toc & 0xFC) | 0x2;
                ptr += 1;
                ptr += encode_size(len(0), &mut data[ptr..]);
            }
        }
        if count > 2 || (pad && tot_size < maxlen) || ext_count > 0 {
            // Code 3
            let mut pad_amount: i32;

            // Restart the process for the padding case
            ptr = 0;
            tot_size = if self_delimited {
                1 + i32::from(len(count - 1) >= 252)
            } else {
                0
            };
            let vbr = (1..count).any(|i| len(i) != len(0));
            if vbr {
                tot_size += 2;
                for i in 0..count - 1 {
                    tot_size += 1 + i32::from(len(i) >= 252) + len(i);
                }
                tot_size += len(count - 1);

                if tot_size > maxlen {
                    return Err(Error::BufferTooSmall);
                }
                data[ptr] = (self.toc & 0xFC) | 0x3;
                ptr += 1;
                data[ptr] = count as u8 | 0x80;
                ptr += 1;
            } else {
                tot_size += count as i32 * len(0) + 2;
                if tot_size > maxlen {
                    return Err(Error::BufferTooSmall);
                }
                data[ptr] = (self.toc & 0xFC) | 0x3;
                ptr += 1;
                data[ptr] = count as u8;
                ptr += 1;
            }
            pad_amount = if pad { maxlen - tot_size } else { 0 };
            if ext_count > 0 {
                // figure out how much space we need for the extensions
                ext_len = extensions::generate_impl(
                    None,
                    maxlen - tot_size,
                    all_extensions,
                    count as i32,
                    false,
                )?;
                if !pad {
                    pad_amount = ext_len
                        + if ext_len != 0 {
                            (ext_len + 253) / 254
                        } else {
                            1
                        };
                }
            }
            if pad_amount != 0 {
                data[1] |= 0x40;
                let nb_255s = (pad_amount - 1) / 255;
                if tot_size + ext_len + nb_255s + 1 > maxlen {
                    return Err(Error::BufferTooSmall);
                }
                ext_begin = tot_size + pad_amount - ext_len;
                // Prepend 0x01 padding
                ones_begin = tot_size + nb_255s + 1;
                ones_end = tot_size + pad_amount - ext_len;
                for _ in 0..nb_255s {
                    data[ptr] = 255;
                    ptr += 1;
                }
                data[ptr] = (pad_amount - 255 * nb_255s - 1) as u8;
                ptr += 1;
                tot_size += pad_amount;
            }
            if vbr {
                for i in 0..count - 1 {
                    ptr += encode_size(len(i), &mut data[ptr..]);
                }
            }
        }
        if self_delimited {
            let sdlen = encode_size(len(count - 1), &mut data[ptr..]);
            ptr += sdlen;
        }
        // Copy the actual data
        for f in frames {
            data[ptr..ptr + f.len()].copy_from_slice(f);
            ptr += f.len();
        }
        if ext_len > 0 {
            let (eb, el) = (ext_begin as usize, ext_len as usize);
            let ret = extensions::generate_impl(
                Some(&mut data[eb..eb + el]),
                ext_len,
                all_extensions,
                count as i32,
                false,
            );
            debug_assert!(ret == Ok(ext_len));
        }
        for i in ones_begin..ones_end {
            data[i as usize] = 0x01;
        }
        if pad && ext_count == 0 {
            // Fill padding with zeros.
            data[ptr..maxlen as usize].fill(0);
        }
        Ok(tot_size)
    }

    /// Port of `src/repacketizer.c:opus_repacketizer_out_range`: writes frames `begin..end`
    /// (in the order they were added) as one packet into `data` and returns its length.
    ///
    /// # Errors
    /// [`Error::BadArg`] for an invalid range, [`Error::BufferTooSmall`] if `data` is too
    /// small, [`Error::InternalError`] if a source packet's extensions are malformed.
    pub fn out_range(&self, begin: i32, end: i32, data: &mut [u8]) -> Result<usize> {
        self.out_range_impl(begin, end, data, false, false, &[])
            .map(|n| n as usize)
    }

    /// Port of `src/repacketizer.c:opus_repacketizer_out`: writes all frames added so far as
    /// one packet into `data` and returns its length.
    ///
    /// # Errors
    /// As for [`Repacketizer::out_range`].
    pub fn out(&self, data: &mut [u8]) -> Result<usize> {
        self.out_range(0, self.nb_frames, data)
    }
}

/// Port of `src/repacketizer.c:opus_packet_pad_impl`: pads the packet in `data[..len]` to
/// `new_len` bytes (`data` must hold at least `new_len` bytes), adding `extensions`. With
/// `pad == false` the packet only grows as much as the extensions need. Returns the new
/// length (`0`, i.e. `OPUS_OK`, when `len == new_len`).
///
/// # Errors
/// [`Error::BadArg`] if `len < 1`, `len > new_len` or `data` is shorter than `new_len`; the
/// errors of [`Repacketizer::cat`] and [`Repacketizer::out_range_impl`].
pub fn opus_packet_pad_impl(
    data: &mut [u8],
    len: i32,
    new_len: i32,
    pad: bool,
    extensions: &[Extension<'_>],
) -> Result<i32> {
    if len < 1 {
        return Err(Error::BadArg);
    }
    if len == new_len {
        return Ok(0);
    } else if len > new_len {
        return Err(Error::BadArg);
    }
    let Some(out) = data.get_mut(..new_len as usize) else {
        return Err(Error::BadArg);
    };
    // Moving payload to the end of the packet so we can do in-place padding
    let copy = out[..len as usize].to_vec();
    let mut rp = Repacketizer::new();
    rp.cat(&copy)?;
    rp.out_range_impl(0, rp.nb_frames, out, false, pad, extensions)
}

/// Port of `src/repacketizer.c:opus_packet_pad`: pads the packet in `data[..len]` to `new_len`
/// bytes in place (`data` must hold at least `new_len` bytes). The padded packet decodes
/// identically.
///
/// # Errors
/// [`Error::BadArg`] if `len == 0`, `len > new_len` or `data` is too short;
/// [`Error::InvalidPacket`] if the packet is malformed.
pub fn packet_pad(data: &mut [u8], len: usize, new_len: usize) -> Result<()> {
    opus_packet_pad_impl(data, len_i32(len), len_i32(new_len), true, &[]).map(drop)
}

/// Port of `src/repacketizer.c:opus_packet_unpad`: removes all padding (and extensions) from
/// the packet `data` in place and returns its new length.
///
/// # Errors
/// [`Error::BadArg`] if `data` is empty; [`Error::InvalidPacket`] if it is malformed.
pub fn packet_unpad(data: &mut [u8]) -> Result<usize> {
    let len = len_i32(data.len());
    if len < 1 {
        return Err(Error::BadArg);
    }
    let copy = data[..len as usize].to_vec();
    let mut rp = Repacketizer::new();
    rp.cat(&copy)?;
    // Discard all padding and extensions.
    rp.discard_padding();
    let ret = rp.out_range_impl(
        0,
        rp.nb_frames,
        &mut data[..len as usize],
        false,
        false,
        &[],
    )?;
    debug_assert!(ret > 0 && ret <= len);
    Ok(ret as usize)
}

/// Port of `src/repacketizer.c:opus_multistream_packet_pad`: pads the multistream packet in
/// `data[..len]` (with `nb_streams` streams) to `new_len` bytes by padding its last stream.
///
/// # Errors
/// [`Error::BadArg`] if `len == 0`, `len > new_len` or `data` is too short;
/// [`Error::InvalidPacket`] if the packet is malformed.
pub fn multistream_packet_pad(
    data: &mut [u8],
    len: usize,
    new_len: usize,
    nb_streams: i32,
) -> Result<()> {
    let mut len = len_i32(len);
    let new_len = len_i32(new_len);
    if len < 1 {
        return Err(Error::BadArg);
    }
    if len == new_len {
        return Ok(());
    } else if len > new_len {
        return Err(Error::BadArg);
    }
    if data.len() < new_len as usize {
        return Err(Error::BadArg);
    }
    let amount = new_len - len;
    let mut off: usize = 0;
    // Seek to last stream
    for _ in 0..nb_streams - 1 {
        if len <= 0 {
            return Err(Error::InvalidPacket);
        }
        let parsed = packet::parse_impl(&data[off..off + len as usize], true)?;
        let packet_offset = parsed.packet_offset as i32;
        off += packet_offset as usize;
        len -= packet_offset;
    }
    opus_packet_pad_impl(&mut data[off..], len, len + amount, true, &[]).map(drop)
}

/// Port of `src/repacketizer.c:opus_multistream_packet_unpad`: removes all padding from each
/// stream of the multistream packet `data` in place and returns its new length.
///
/// # Errors
/// [`Error::BadArg`] if `data` is empty; [`Error::InvalidPacket`] if it is malformed.
pub fn multistream_packet_unpad(data: &mut [u8], nb_streams: i32) -> Result<usize> {
    let mut len = len_i32(data.len());
    if len < 1 {
        return Err(Error::BadArg);
    }
    // The C code works in place; streams only ever shrink, so reading from a copy of the
    // input is equivalent.
    let copy = data[..len as usize].to_vec();
    let mut src: usize = 0;
    let mut dst: usize = 0;
    let mut dst_len: i32 = 0;
    // Unpad all frames
    for s in 0..nb_streams {
        let self_delimited = s != nb_streams - 1;
        if len <= 0 {
            return Err(Error::InvalidPacket);
        }
        let mut rp = Repacketizer::new();
        let input = &copy[src..src + len as usize];
        let packet_offset = packet::parse_impl(input, self_delimited)?.packet_offset;
        rp.cat_impl(&input[..packet_offset], self_delimited)?;
        // Discard all padding and extensions.
        rp.discard_padding();
        let ret = rp.out_range_impl(
            0,
            rp.nb_frames,
            &mut data[dst..dst + len as usize],
            self_delimited,
            false,
            &[],
        )?;
        dst_len += ret;
        dst += ret as usize;
        src += packet_offset;
        len -= packet_offset as i32;
    }
    Ok(dst_len as usize)
}
