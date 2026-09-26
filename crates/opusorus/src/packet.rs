//! Opus packet helpers: TOC inspection, packet parsing and PCM soft clipping.
//!
//! Port of `src/opus.c` (packet parsing, soft clip, `encode_size`), the TOC helpers of
//! `src/opus_decoder.c` (`opus_packet_get_bandwidth`, `opus_packet_get_nb_channels`,
//! `opus_packet_get_nb_frames`, `opus_packet_get_nb_samples`, `opus_packet_has_lbrr`,
//! `opus_packet_get_mode`) and the packet-related helpers of `src/opus_private.h` (`align`,
//! `MODE_*`).
//!
//! The public functions take the packet as a byte slice. Where libopus reads `data[0]`
//! unconditionally (undefined behaviour for an empty packet), the Rust functions return
//! [`Error::BadArg`] for an empty slice instead.

use crate::celt::arch::{abs16, max16, min16};
use crate::celt::mathops::opus_limit2_checkwithin1;
use crate::{Bandwidth, Error, Result};

/// SILK-only coding mode (`MODE_SILK_ONLY`, `src/opus_private.h`).
pub const MODE_SILK_ONLY: i32 = 1000;
/// Hybrid SILK + CELT coding mode (`MODE_HYBRID`, `src/opus_private.h`).
pub const MODE_HYBRID: i32 = 1001;
/// CELT-only coding mode (`MODE_CELT_ONLY`, `src/opus_private.h`).
pub const MODE_CELT_ONLY: i32 = 1002;

/// Maximum number of frames in one Opus packet (120 ms of 2.5 ms frames).
pub const MAX_FRAMES: usize = 48;

/// Port of `src/opus_private.h:align`: rounds `i` up to the alignment of a union of
/// `void*`, `opus_int32` and `opus_val32` (the platform pointer alignment, at least 4).
///
/// As in C, the arithmetic is done in `unsigned int` (so negative inputs wrap).
#[must_use]
pub const fn align(i: i32) -> i32 {
    let ptr_align = align_of::<*const u8>();
    let alignment = if ptr_align > 4 { ptr_align } else { 4 } as u32;
    ((i as u32).wrapping_add(alignment - 1) / alignment).wrapping_mul(alignment) as i32
}

/// Clamps a slice length to the `opus_int32` range used by libopus lengths.
#[inline]
pub(crate) const fn len_i32(len: usize) -> i32 {
    if len > i32::MAX as usize {
        i32::MAX
    } else {
        len as i32
    }
}

/// Returns the first byte of `data` (the TOC byte), or [`Error::BadArg`] for an empty slice.
#[inline]
const fn first_byte(data: &[u8]) -> Result<u8> {
    match data.first() {
        Some(&b) => Ok(b),
        None => Err(Error::BadArg),
    }
}

// ---------------------------------------------------------------------------------------------
// TOC helpers
// ---------------------------------------------------------------------------------------------

/// Port of `src/opus.c:opus_packet_get_samples_per_frame`, operating on the TOC byte.
///
/// Integer overflow for absurd `fs` values (undefined behaviour in C) wraps.
#[must_use]
pub const fn toc_samples_per_frame(toc: u8, fs: i32) -> i32 {
    let mut audiosize: i32;
    if toc & 0x80 != 0 {
        audiosize = ((toc >> 3) & 0x3) as i32;
        audiosize = fs.wrapping_shl(audiosize as u32) / 400;
    } else if (toc & 0x60) == 0x60 {
        audiosize = if toc & 0x08 != 0 { fs / 50 } else { fs / 100 };
    } else {
        audiosize = ((toc >> 3) & 0x3) as i32;
        if audiosize == 3 {
            audiosize = fs.wrapping_mul(60) / 1000;
        } else {
            audiosize = fs.wrapping_shl(audiosize as u32) / 100;
        }
    }
    audiosize
}

/// Port of `src/opus.c:opus_packet_get_samples_per_frame`: number of samples per frame at
/// sampling rate `fs` for the packet `data`.
///
/// # Errors
/// [`Error::BadArg`] if `data` is empty.
pub const fn get_samples_per_frame(data: &[u8], fs: i32) -> Result<i32> {
    match first_byte(data) {
        Ok(toc) => Ok(toc_samples_per_frame(toc, fs)),
        Err(e) => Err(e),
    }
}

/// Port of `src/opus_decoder.c:opus_packet_get_bandwidth`, operating on the TOC byte.
#[must_use]
pub const fn toc_bandwidth(toc: u8) -> Bandwidth {
    if toc & 0x80 != 0 {
        // OPUS_BANDWIDTH_MEDIUMBAND + ((toc>>5)&3), with MEDIUMBAND mapped to NARROWBAND.
        match (toc >> 5) & 0x3 {
            0 => Bandwidth::Narrowband,
            1 => Bandwidth::Wideband,
            2 => Bandwidth::Superwideband,
            _ => Bandwidth::Fullband,
        }
    } else if (toc & 0x60) == 0x60 {
        if toc & 0x10 != 0 {
            Bandwidth::Fullband
        } else {
            Bandwidth::Superwideband
        }
    } else {
        match (toc >> 5) & 0x3 {
            0 => Bandwidth::Narrowband,
            1 => Bandwidth::Mediumband,
            _ => Bandwidth::Wideband,
        }
    }
}

/// Port of `src/opus_decoder.c:opus_packet_get_bandwidth`: the bandwidth of an Opus packet.
///
/// # Errors
/// [`Error::BadArg`] if `data` is empty.
pub const fn get_bandwidth(data: &[u8]) -> Result<Bandwidth> {
    match first_byte(data) {
        Ok(toc) => Ok(toc_bandwidth(toc)),
        Err(e) => Err(e),
    }
}

/// Port of `src/opus_decoder.c:opus_packet_get_nb_channels`, operating on the TOC byte.
#[must_use]
pub const fn toc_nb_channels(toc: u8) -> i32 {
    if toc & 0x4 != 0 { 2 } else { 1 }
}

/// Port of `src/opus_decoder.c:opus_packet_get_nb_channels`: number of channels (1 or 2)
/// coded in the packet.
///
/// # Errors
/// [`Error::BadArg`] if `data` is empty.
pub const fn get_nb_channels(data: &[u8]) -> Result<i32> {
    match first_byte(data) {
        Ok(toc) => Ok(toc_nb_channels(toc)),
        Err(e) => Err(e),
    }
}

/// Port of `src/opus_decoder.c:opus_packet_get_mode` (static in C), operating on the TOC byte.
/// Returns one of [`MODE_SILK_ONLY`], [`MODE_HYBRID`], [`MODE_CELT_ONLY`].
#[must_use]
pub const fn toc_mode(toc: u8) -> i32 {
    if toc & 0x80 != 0 {
        MODE_CELT_ONLY
    } else if (toc & 0x60) == 0x60 {
        MODE_HYBRID
    } else {
        MODE_SILK_ONLY
    }
}

/// Port of `src/opus_decoder.c:opus_packet_get_nb_frames`: number of frames in a packet.
///
/// Like libopus, a code-3 packet may report `0` frames (it is only rejected by [`parse`]).
///
/// # Errors
/// [`Error::BadArg`] if `packet` is empty, [`Error::InvalidPacket`] if a code-3 packet has no
/// frame count byte.
pub const fn get_nb_frames(packet: &[u8]) -> Result<i32> {
    let len = packet.len();
    if len < 1 {
        return Err(Error::BadArg);
    }
    let count = packet[0] & 0x3;
    if count == 0 {
        Ok(1)
    } else if count != 3 {
        Ok(2)
    } else if len < 2 {
        Err(Error::InvalidPacket)
    } else {
        Ok((packet[1] & 0x3F) as i32)
    }
}

/// Port of `src/opus_decoder.c:opus_packet_get_nb_samples`: number of samples of a packet at
/// sampling rate `fs`.
///
/// # Errors
/// Errors of [`get_nb_frames`]; [`Error::InvalidPacket`] if the packet is longer than 120 ms.
pub const fn get_nb_samples(packet: &[u8], fs: i32) -> Result<i32> {
    let count = match get_nb_frames(packet) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    // Overflow here needs an absurd `fs` (undefined behaviour in C); wrap instead of panicking.
    let samples = count.wrapping_mul(toc_samples_per_frame(packet[0], fs));
    // Can't have more than 120 ms
    if samples.wrapping_mul(25) > fs.wrapping_mul(3) {
        Err(Error::InvalidPacket)
    } else {
        Ok(samples)
    }
}

/// Port of `src/opus_decoder.c:opus_packet_has_lbrr`: whether the (first frame of the) packet
/// carries SILK LBRR (in-band FEC) data.
///
/// # Errors
/// Errors of [`parse`] (an empty packet is [`Error::InvalidPacket`], as `parse` reports it).
pub fn has_lbrr(packet: &[u8]) -> Result<bool> {
    let Some(&toc) = packet.first() else {
        return Err(Error::InvalidPacket);
    };
    let packet_mode = toc_mode(toc);
    if packet_mode == MODE_CELT_ONLY {
        return Ok(false);
    }
    let packet_frame_size = toc_samples_per_frame(toc, 48000);
    let mut nb_frames = 1;
    if packet_frame_size > 960 {
        nb_frames = packet_frame_size / 960;
    }
    let packet_stream_channels = toc_nb_channels(toc);
    let parsed = parse(packet)?;
    let frame0 = parsed.frames[0];
    let Some(&b0) = frame0.first() else {
        return Ok(false);
    };
    let mut lbrr = (b0 >> (7 - nb_frames)) & 0x1 != 0;
    if packet_stream_channels == 2 {
        lbrr = lbrr || ((b0 >> (6 - 2 * nb_frames)) & 0x1) != 0;
    }
    Ok(lbrr)
}

// ---------------------------------------------------------------------------------------------
// Frame size coding
// ---------------------------------------------------------------------------------------------

/// Port of `src/opus.c:encode_size`: writes the 1- or 2-byte frame length code for `size` into
/// `data` and returns the number of bytes written.
///
/// # Panics
/// If `data` is shorter than the code (1 byte for `size < 252`, else 2).
pub const fn encode_size(size: i32, data: &mut [u8]) -> usize {
    if size < 252 {
        data[0] = size as u8;
        1
    } else {
        data[0] = (252 + (size & 0x3)) as u8;
        data[1] = ((size - data[0] as i32) >> 2) as u8;
        2
    }
}

/// Port of `src/opus.c:parse_size` (static). `data` starts at the size field; `len` is the
/// number of bytes libopus considers available (`<= data.len()`).
const fn parse_size(data: &[u8], len: i32, size: &mut i16) -> i32 {
    if len < 1 {
        *size = -1;
        -1
    } else if data[0] < 252 {
        *size = data[0] as i16;
        1
    } else if len < 2 {
        *size = -1;
        -1
    } else {
        *size = 4 * data[1] as i16 + data[0] as i16;
        2
    }
}

// ---------------------------------------------------------------------------------------------
// Packet parsing
// ---------------------------------------------------------------------------------------------

/// Result of parsing an Opus packet ([`parse`] / [`parse_impl`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedPacket<'a> {
    /// The TOC byte.
    pub toc: u8,
    /// Number of frames in the packet (1..=48).
    pub nb_frames: usize,
    /// The frames' payloads (only the first `nb_frames` entries are meaningful; the rest are
    /// empty).
    pub frames: [&'a [u8]; MAX_FRAMES],
    /// Offset of the first frame's payload from the start of the packet.
    pub payload_offset: usize,
    /// Total size of the packet including padding (for self-delimited packets, the offset of
    /// the next packet).
    pub packet_offset: usize,
    /// The packet padding (which may hold extensions, see [`crate::extensions`]).
    pub padding: &'a [u8],
}

impl<'a> ParsedPacket<'a> {
    /// The frame payloads.
    #[must_use]
    pub fn frames(&self) -> &[&'a [u8]] {
        &self.frames[..self.nb_frames]
    }
}

/// Port of `src/opus.c:opus_packet_parse`: parses an Opus packet into its frames.
///
/// # Errors
/// [`Error::InvalidPacket`] if the packet is malformed.
pub fn parse(data: &[u8]) -> Result<ParsedPacket<'_>> {
    parse_impl(data, false)
}

/// Port of `src/opus.c:opus_packet_parse_impl`: parses a packet, optionally with
/// self-delimited framing (as used by all but the last stream of a multistream packet).
///
/// The C out-parameters (`out_toc`, `frames`, `size`, `payload_offset`, `packet_offset`,
/// `padding`, `padding_len`) are fields of the returned [`ParsedPacket`]; the C return value
/// (frame count) is [`ParsedPacket::nb_frames`].
///
/// # Errors
/// [`Error::InvalidPacket`] if the packet is malformed.
pub fn parse_impl(data: &[u8], self_delimited: bool) -> Result<ParsedPacket<'_>> {
    let mut len = len_i32(data.len());
    let mut size = [0i16; MAX_FRAMES];
    let mut pad: i32 = 0;
    // `size==NULL || len<0` (OPUS_BAD_ARG) cannot happen with slices.
    if len == 0 {
        return Err(Error::InvalidPacket);
    }

    let framesize = toc_samples_per_frame(data[0], 48000);

    let mut cbr = false;
    let toc = data[0];
    let mut pos: usize = 1;
    len -= 1;
    let mut last_size = len;
    let count: usize;
    match toc & 0x3 {
        // One frame
        0 => count = 1,
        // Two CBR frames
        1 => {
            count = 2;
            cbr = true;
            if !self_delimited {
                if len & 0x1 != 0 {
                    return Err(Error::InvalidPacket);
                }
                last_size = len / 2;
                // If last_size doesn't fit in size[0], we'll catch it later
                size[0] = last_size as i16;
            }
        }
        // Two VBR frames
        2 => {
            count = 2;
            let bytes = parse_size(&data[pos..], len, &mut size[0]);
            len -= bytes;
            if size[0] < 0 || size[0] as i32 > len {
                return Err(Error::InvalidPacket);
            }
            pos += bytes as usize;
            last_size = len - size[0] as i32;
        }
        // Multiple CBR/VBR frames (from 0 to 120 ms)
        _ => {
            if len < 1 {
                return Err(Error::InvalidPacket);
            }
            // Number of frames encoded in bits 0 to 5
            let ch = data[pos];
            pos += 1;
            count = (ch & 0x3F) as usize;
            if count == 0 || framesize * count as i32 > 5760 {
                return Err(Error::InvalidPacket);
            }
            len -= 1;
            // Padding flag is bit 6
            if ch & 0x40 != 0 {
                loop {
                    if len <= 0 {
                        return Err(Error::InvalidPacket);
                    }
                    let p = data[pos] as i32;
                    pos += 1;
                    len -= 1;
                    let tmp = if p == 255 { 254 } else { p };
                    len -= tmp;
                    pad += tmp;
                    if p != 255 {
                        break;
                    }
                }
            }
            if len < 0 {
                return Err(Error::InvalidPacket);
            }
            // VBR flag is bit 7
            cbr = ch & 0x80 == 0;
            if !cbr {
                // VBR case
                last_size = len;
                for sz in &mut size[..count - 1] {
                    let bytes = parse_size(&data[pos..], len, sz);
                    len -= bytes;
                    if *sz < 0 || *sz as i32 > len {
                        return Err(Error::InvalidPacket);
                    }
                    pos += bytes as usize;
                    last_size -= bytes + *sz as i32;
                }
                if last_size < 0 {
                    return Err(Error::InvalidPacket);
                }
            } else if !self_delimited {
                // CBR case
                last_size = len / count as i32;
                if last_size * count as i32 != len {
                    return Err(Error::InvalidPacket);
                }
                for s in &mut size[..count - 1] {
                    *s = last_size as i16;
                }
            }
        }
    }
    // Self-delimited framing has an extra size for the last frame.
    if self_delimited {
        let bytes = parse_size(&data[pos..], len, &mut size[count - 1]);
        len -= bytes;
        if size[count - 1] < 0 || size[count - 1] as i32 > len {
            return Err(Error::InvalidPacket);
        }
        pos += bytes as usize;
        // For CBR packets, apply the size to all the frames.
        if cbr {
            if size[count - 1] as i32 * count as i32 > len {
                return Err(Error::InvalidPacket);
            }
            let last = size[count - 1];
            for s in &mut size[..count - 1] {
                *s = last;
            }
        } else if bytes + size[count - 1] as i32 > last_size {
            return Err(Error::InvalidPacket);
        }
    } else {
        // Because it's not encoded explicitly, it's possible the size of the
        // last packet (or all the packets, for the CBR case) is larger than
        // 1275. Reject them here.
        if last_size > 1275 {
            return Err(Error::InvalidPacket);
        }
        size[count - 1] = last_size as i16;
    }

    let payload_offset = pos;

    let mut frames: [&[u8]; MAX_FRAMES] = [&[]; MAX_FRAMES];
    for (frame, &sz) in frames.iter_mut().zip(&size).take(count) {
        let end = pos + sz as usize;
        // The checks above guarantee the frames lie within the packet; `get` only guards
        // against a porting error.
        *frame = data.get(pos..end).ok_or(Error::InternalError)?;
        pos = end;
    }

    let padding = data
        .get(pos..pos + pad as usize)
        .ok_or(Error::InternalError)?;
    let packet_offset = pad as usize + pos;

    Ok(ParsedPacket {
        toc,
        nb_frames: count,
        frames,
        payload_offset,
        packet_offset,
        padding,
    })
}

// ---------------------------------------------------------------------------------------------
// Soft clipping
// ---------------------------------------------------------------------------------------------

/// Port of `src/opus.c:opus_pcm_soft_clip`: applies soft clipping to bring interleaved float
/// PCM into the `[-1, 1]` range, carrying the non-linearity state across calls in
/// `declip_mem` (one value per channel, initialize to zero).
///
/// `pcm` holds `frame_size * channels` interleaved samples. Like libopus with NULL pointers,
/// the call does nothing if `frame_size` or `channels` is zero or a buffer is too short.
pub fn pcm_soft_clip(pcm: &mut [f32], frame_size: usize, channels: usize, declip_mem: &mut [f32]) {
    let n = frame_size.min(i32::MAX as usize) as i32;
    let c = channels.min(i32::MAX as usize) as i32;
    opus_pcm_soft_clip_impl(pcm, n, c, declip_mem);
}

/// Port of `src/opus.c:opus_pcm_soft_clip_impl` (the `arch` argument is dropped).
///
/// `x` holds `n * c` interleaved samples; `declip_mem` at least `c` values. Returns without
/// doing anything when `c < 1`, `n < 1` or a buffer is too short (C: NULL pointer).
#[expect(
    clippy::many_single_char_names,
    reason = "names mirror the C implementation"
)]
pub fn opus_pcm_soft_clip_impl(x: &mut [f32], n: i32, c: i32, declip_mem: &mut [f32]) {
    if c < 1 || n < 1 {
        return;
    }
    let (n, c) = (n as usize, c as usize);
    let Some(total) = n.checked_mul(c) else {
        return;
    };
    if x.len() < total || declip_mem.len() < c {
        return;
    }
    let x = &mut x[..total];

    // Clamp everything within the range [-2, +2] which is the domain of the soft
    // clipping non-linearity. Outside the defined range the derivative will be zero,
    // therefore there is no discontinuity introduced here. The C implementation never
    // provides the "all within [-1, 1]" hint.
    let all_within_neg1pos1 = opus_limit2_checkwithin1(x);

    for ch in 0..c {
        let xs = &mut x[ch..];
        let mut a = declip_mem[ch];
        // Continue applying the non-linearity from the previous frame to avoid
        // any discontinuity.
        for i in 0..n {
            let v = xs[i * c];
            if v * a >= 0.0 {
                break;
            }
            xs[i * c] = v + a * v * v;
        }

        let mut curr: usize = 0;
        let x0 = xs[0];
        loop {
            let mut i: usize;
            // Detection for early exit can be skipped if hinted by `all_within_neg1pos1`
            if all_within_neg1pos1 {
                i = n;
            } else {
                i = curr;
                while i < n {
                    if xs[i * c] > 1.0 || xs[i * c] < -1.0 {
                        break;
                    }
                    i += 1;
                }
            }
            if i == n {
                a = 0.0;
                break;
            }
            let mut peak_pos = i;
            let mut start = i;
            let mut end = i;
            let mut maxval = abs16(xs[i * c]);
            // Look for first zero crossing before clipping
            while start > 0 && xs[i * c] * xs[(start - 1) * c] >= 0.0 {
                start -= 1;
            }
            // Look for first zero crossing after clipping
            while end < n && xs[i * c] * xs[end * c] >= 0.0 {
                // Look for other peaks until the next zero-crossing.
                if abs16(xs[end * c]) > maxval {
                    maxval = abs16(xs[end * c]);
                    peak_pos = end;
                }
                end += 1;
            }
            // Detect the special case where we clip before the first zero crossing
            let special = start == 0 && xs[i * c] * xs[0] >= 0.0;

            // Compute a such that maxval + a*maxval^2 = 1
            a = (maxval - 1.0) / (maxval * maxval);
            // Slightly boost "a" by 2^-22. This is just enough to ensure -ffast-math
            // does not cause output values larger than +/-1, but small enough not
            // to matter even for 24-bit output.
            a += a * 2.4e-7f32;
            if xs[i * c] > 0.0 {
                a = -a;
            }
            // Apply soft clipping
            for j in start..end {
                let v = xs[j * c];
                xs[j * c] = v + a * v * v;
            }

            if special && peak_pos >= 2 {
                // Add a linear ramp from the first sample to the signal peak.
                // This avoids a discontinuity at the beginning of the frame.
                let mut offset = x0 - xs[0];
                let delta = offset / peak_pos as f32;
                for j in curr..peak_pos {
                    offset -= delta;
                    xs[j * c] += offset;
                    xs[j * c] = max16(-1.0, min16(1.0, xs[j * c]));
                }
            }
            curr = end;
            if curr == n {
                break;
            }
        }
        declip_mem[ch] = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toc_helpers() {
        assert_eq!(toc_samples_per_frame(0x00, 48000), 480);
        assert_eq!(toc_samples_per_frame(0x18, 48000), 2880);
        assert_eq!(toc_samples_per_frame(0x80, 48000), 120);
        assert_eq!(toc_samples_per_frame(0x78, 48000), 960);
        assert_eq!(toc_bandwidth(0x80), Bandwidth::Narrowband);
        assert_eq!(toc_bandwidth(0x70), Bandwidth::Fullband);
        assert_eq!(get_bandwidth(&[]), Err(Error::BadArg));
        assert_eq!(get_nb_frames(&[0x03]), Err(Error::InvalidPacket));
        assert_eq!(get_nb_frames(&[0x03, 0x05]), Ok(5));
    }

    #[test]
    fn parse_basic() {
        let pkt = [0x01u8, 1, 2, 3, 4];
        let p = parse(&pkt).expect("valid");
        assert_eq!(p.nb_frames, 2);
        assert_eq!(p.frames(), &[&[1u8, 2][..], &[3, 4][..]]);
        assert_eq!(parse(&[]), Err(Error::InvalidPacket));
        assert_eq!(parse(&[0x01, 1, 2, 3]), Err(Error::InvalidPacket));
    }

    #[test]
    fn size_roundtrip() {
        for s in 0..=1275 {
            let mut b = [0u8; 2];
            let n = encode_size(s, &mut b);
            let mut out = 0i16;
            assert_eq!(parse_size(&b, 2, &mut out) as usize, n);
            assert_eq!(out as i32, s);
        }
    }
}
