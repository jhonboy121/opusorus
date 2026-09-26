//! The multistream Opus decoder: port of `src/opus_multistream_decoder.c`.
//!
//! [`MsDecoder`] (C `OpusMSDecoder`) decodes a multistream packet — the concatenation of
//! `streams` Opus packets, all but the last self-delimited — into up to 255 output channels. The
//! first `coupled_streams` streams are stereo, the rest mono; the channel `mapping` routes each
//! coded channel to its output channel(s) (255 = silent channel). This is the decoder for Ogg
//! Opus channel mapping families 1 (Vorbis surround) and 255 (discrete), and the base of the
//! projection (ambisonics) decoder.
//!
//! # Example
//!
//! ```
//! use opusorus::ms_decoder::MsDecoder;
//!
//! # fn main() -> opusorus::Result<()> {
//! // 5.1 surround (Vorbis order): 4 streams, 2 of them coupled.
//! let mut dec = MsDecoder::new(48000, 6, 4, 2, &[0, 4, 1, 2, 3, 5])?;
//! let mut pcm = vec![0f32; 960 * 6];
//! // A lost packet runs the concealment of every stream.
//! assert_eq!(dec.decode_float(None, &mut pcm, 960, false)?, 960);
//! # Ok(())
//! # }
//! ```
//!
//! # Mapping to the C API
//!
//! | C | Rust |
//! |---|---|
//! | `opus_multistream_decoder_get_size` | [`MsDecoder::get_size`] (Rust footprint) |
//! | `opus_multistream_decoder_create` / `_init` | [`MsDecoder::new`] / [`MsDecoder::init`] |
//! | `opus_multistream_decode` / `_decode24` / `_decode_float` | [`MsDecoder::decode`] / [`MsDecoder::decode24`] / [`MsDecoder::decode_float`] |
//! | `opus_multistream_decoder_ctl` | typed methods, [`MsDecoder::decoder_state`] and [`MsDecoder::ctl_set`] / [`MsDecoder::ctl_get`] |
//! | `opus_multistream_decode_native` | [`MsDecoder::opus_multistream_decode_native`] (hidden) |
//!
//! Output buffers are slices: one shorter than `frame_size * channels` (with `frame_size`
//! limited to 120 ms) returns [`Error::BadArg`] where C would write out of bounds. A `mapping`
//! shorter than `channels` is [`Error::BadArg`].

use alloc::vec::Vec;

use crate::celt::arch::{OpusRes, imin, res2float, res2int16, res2int24};
use crate::decoder::{
    Decoder, OPTIONAL_CLIP, OPUS_GET_BANDWIDTH_REQUEST, OPUS_GET_COMPLEXITY_REQUEST,
    OPUS_GET_FINAL_RANGE_REQUEST, OPUS_GET_GAIN_REQUEST, OPUS_GET_LAST_PACKET_DURATION_REQUEST,
    OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST, OPUS_GET_SAMPLE_RATE_REQUEST, OPUS_RESET_STATE,
    OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_GAIN_REQUEST, OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
    max_over_rates, res_buf,
};
use crate::multistream::{
    ChannelLayout, get_left_channel, get_mono_channel, get_right_channel, validate_layout,
};
use crate::packet::{get_nb_samples, parse_impl};
use crate::{Bandwidth, Error, Result};

/// `OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST` (takes a stream id and a pointer; use
/// [`MsDecoder::decoder_state`]).
pub const OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST: i32 = 5122;

/// A `copy_channel_out` callback (C `opus_copy_channel_out_func`): writes one decoded channel
/// (`src` = `None` for a silent channel) of `frame_size` samples, read every `src_stride`
/// values, to channel `dst_channel` of the `dst_stride`-channel interleaved output `dst`.
#[doc(hidden)]
pub type CopyChannelOut<'a, T> =
    dyn FnMut(&mut [T], usize, usize, Option<&[OpusRes]>, usize, usize) + 'a;

/// Port of `src/opus_multistream_decoder.c:opus_copy_channel_out_float`.
fn opus_copy_channel_out_float(
    dst: &mut [f32],
    dst_stride: usize,
    dst_channel: usize,
    src: Option<&[OpusRes]>,
    src_stride: usize,
    frame_size: usize,
) {
    match src {
        Some(src) => {
            for i in 0..frame_size {
                dst[i * dst_stride + dst_channel] = res2float(src[i * src_stride]);
            }
        }
        None => {
            for i in 0..frame_size {
                dst[i * dst_stride + dst_channel] = 0.0;
            }
        }
    }
}

/// Port of `src/opus_multistream_decoder.c:opus_copy_channel_out_short`.
fn opus_copy_channel_out_short(
    dst: &mut [i16],
    dst_stride: usize,
    dst_channel: usize,
    src: Option<&[OpusRes]>,
    src_stride: usize,
    frame_size: usize,
) {
    match src {
        Some(src) => {
            for i in 0..frame_size {
                dst[i * dst_stride + dst_channel] = res2int16(src[i * src_stride]);
            }
        }
        None => {
            for i in 0..frame_size {
                dst[i * dst_stride + dst_channel] = 0;
            }
        }
    }
}

/// Port of `src/opus_multistream_decoder.c:opus_copy_channel_out_int24`.
fn opus_copy_channel_out_int24(
    dst: &mut [i32],
    dst_stride: usize,
    dst_channel: usize,
    src: Option<&[OpusRes]>,
    src_stride: usize,
    frame_size: usize,
) {
    match src {
        Some(src) => {
            for i in 0..frame_size {
                dst[i * dst_stride + dst_channel] = res2int24(src[i * src_stride]);
            }
        }
        None => {
            for i in 0..frame_size {
                dst[i * dst_stride + dst_channel] = 0;
            }
        }
    }
}

/// Port of `src/opus_multistream_decoder.c:opus_multistream_packet_validate`: checks that
/// `data` holds `nb_streams` valid packets with the same duration and returns that duration
/// (samples per channel at `fs`).
fn opus_multistream_packet_validate(data: &[u8], nb_streams: i32, fs: i32) -> Result<i32> {
    let mut data = data;
    let mut samples = 0;
    for s in 0..nb_streams {
        if data.is_empty() {
            return Err(Error::InvalidPacket);
        }
        let parsed = parse_impl(data, s != nb_streams - 1)?;
        let packet_offset = parsed.packet_offset;
        let tmp_samples = get_nb_samples(&data[..packet_offset], fs)?;
        if s != 0 && samples != tmp_samples {
            return Err(Error::InvalidPacket);
        }
        samples = tmp_samples;
        data = &data[packet_offset..];
    }
    Ok(samples)
}

/// A multistream Opus decoder (C `OpusMSDecoder`).
///
/// See the [module documentation](self) for the stream/channel layout.
#[derive(Debug, Clone)]
pub struct MsDecoder {
    layout: ChannelLayout,
    /// The per-stream decoders: coupled (stereo) streams first.
    decoders: Vec<Decoder>,
    fs: i32,
    /// `buf` of `opus_multistream_decode_native` (C: a stack VLA of 2 channels x
    /// `frame_size`; see [`res_buf`]).
    buf: Vec<OpusRes>,
}

impl MsDecoder {
    /// Port of `src/opus_multistream_decoder.c:opus_multistream_decoder_create` /
    /// `opus_multistream_decoder_init`.
    ///
    /// * `fs`: sampling rate (8000, 12000, 16000, 24000, 48000; 96000 with `qext`).
    /// * `channels`: output channels (1..=255).
    /// * `streams`: coded streams (1..=255 - `coupled_streams`), of which `coupled_streams`
    ///   are stereo.
    /// * `mapping`: for each output channel, the coded channel it takes (coupled stream `k`
    ///   is coded channels `2k` and `2k+1`, mono stream `m` is `coupled_streams + m`), or 255
    ///   for silence. At least `channels` entries.
    ///
    /// # Errors
    /// [`Error::BadArg`] for invalid dimensions, an invalid mapping or sampling rate.
    pub fn new(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
    ) -> Result<Self> {
        if !(1..=255).contains(&channels)
            || coupled_streams > streams
            || streams < 1
            || coupled_streams < 0
            || streams > 255 - coupled_streams
        {
            return Err(Error::BadArg);
        }
        if mapping.len() < channels as usize {
            return Err(Error::BadArg);
        }
        let mut layout = ChannelLayout {
            nb_channels: channels,
            nb_streams: streams,
            nb_coupled_streams: coupled_streams,
            ..ChannelLayout::default()
        };
        layout.mapping[..channels as usize].copy_from_slice(&mapping[..channels as usize]);
        if !validate_layout(&layout) {
            return Err(Error::BadArg);
        }
        let mut decoders = Vec::with_capacity(streams as usize);
        for i in 0..streams {
            decoders.push(Decoder::new(fs, if i < coupled_streams { 2 } else { 1 })?);
        }
        Ok(Self {
            layout,
            decoders,
            fs,
            buf: Vec::new(),
        })
    }

    /// Port of `opus_multistream_decoder_init`: re-initializes in place (unchanged on error).
    ///
    /// # Errors
    /// As [`MsDecoder::new`].
    pub fn init(
        &mut self,
        fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
    ) -> Result<()> {
        *self = Self::new(fs, channels, streams, coupled_streams, mapping)?;
        Ok(())
    }

    /// Port of `opus_multistream_decoder_get_size`: the memory footprint in bytes (the Rust
    /// layout; see [`Decoder::get_size`]), or 0 for invalid stream counts. It includes the
    /// `buf` of `opus_multistream_decode_native`, which grows to two channels of the longest
    /// frame decoded (20 ms here).
    #[must_use]
    pub fn get_size(streams: i32, coupled_streams: i32) -> usize {
        if streams < 1 || coupled_streams > streams || coupled_streams < 0 {
            return 0;
        }
        let coupled = coupled_streams as usize;
        let mono = (streams - coupled_streams) as usize;
        max_over_rates(|fs| {
            size_of::<Self>()
                + coupled * Decoder::footprint_at(2, fs)
                + mono * Decoder::footprint_at(1, fs)
                + 2 * (fs / 50) as usize * size_of::<OpusRes>()
        })
    }

    /// Number of output channels.
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.layout.nb_channels as usize
    }

    /// Number of coded streams.
    #[must_use]
    pub const fn streams(&self) -> usize {
        self.layout.nb_streams as usize
    }

    /// Number of coupled (stereo) streams.
    #[must_use]
    pub const fn coupled_streams(&self) -> usize {
        self.layout.nb_coupled_streams as usize
    }

    /// Port of `validate_ms_decoder` (assertions only).
    fn validate(&self) {
        debug_assert!(validate_layout(&self.layout));
    }

    /// Port of `src/opus_multistream_decoder.c:opus_multistream_decode_native`: decodes a
    /// multistream packet (`None`/empty = lost) through a `copy_channel_out` callback.
    ///
    /// # Errors
    /// The C error codes.
    #[doc(hidden)]
    pub fn opus_multistream_decode_native<T>(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [T],
        copy_channel_out: &mut CopyChannelOut<'_, T>,
        frame_size: i32,
        decode_fec: i32,
        soft_clip: bool,
    ) -> Result<i32> {
        self.validate();
        let mut frame_size = frame_size;
        if frame_size <= 0 {
            return Err(Error::BadArg);
        }
        // Limit frame_size to avoid excessive stack allocations.
        let fs = self.fs;
        frame_size = imin(frame_size, fs / 25 * 3);
        let nb_channels = self.layout.nb_channels as usize;
        let Self {
            layout,
            decoders,
            buf: buf_heap,
            ..
        } = self;

        let mut data: &[u8] = match data {
            Some(d) => d,
            None => &[],
        };
        let do_plc = data.is_empty();
        if !do_plc && (data.len() as i64) < 2 * layout.nb_streams as i64 - 1 {
            return Err(Error::InvalidPacket);
        }
        // Samples per channel C will write: the packet duration when decoding a packet normally,
        // otherwise (PLC / FEC) the full frame_size.
        let mut out_samples = frame_size;
        if !do_plc {
            let ret = opus_multistream_packet_validate(data, layout.nb_streams, fs)?;
            if ret > frame_size {
                return Err(Error::BufferTooSmall);
            }
            if decode_fec == 0 {
                out_samples = ret;
            }
        }
        // Rust-only guard (checked after validation, like the output size C actually writes):
        // C would write past the caller's buffer.
        if pcm.len() < out_samples as usize * nb_channels {
            return Err(Error::BadArg);
        }
        // C allocates `2*frame_size` samples; the streams write at most `out_samples` each.
        let buf = res_buf(buf_heap, 2 * out_samples as usize);
        {
            for (s, dec) in decoders.iter_mut().enumerate() {
                let s = s as i32;
                if !do_plc && data.is_empty() {
                    return Err(Error::InternalError);
                }
                let mut packet_offset: usize = 0;
                let ret = dec.opus_decode_native(
                    if do_plc { None } else { Some(data) },
                    buf,
                    frame_size,
                    decode_fec,
                    s != layout.nb_streams - 1,
                    Some(&mut packet_offset),
                    soft_clip,
                )?;
                if !do_plc {
                    data = &data[packet_offset..];
                }
                if ret <= 0 {
                    return Ok(ret);
                }
                frame_size = ret;
                let fsz = frame_size as usize;
                if s < layout.nb_coupled_streams {
                    // Copy "left" audio to the channel(s) where it belongs
                    let mut prev = -1;
                    loop {
                        let chan = get_left_channel(layout, s, prev);
                        if chan == -1 {
                            break;
                        }
                        copy_channel_out(pcm, nb_channels, chan as usize, Some(&buf[..]), 2, fsz);
                        prev = chan;
                    }
                    // Copy "right" audio to the channel(s) where it belongs
                    let mut prev = -1;
                    loop {
                        let chan = get_right_channel(layout, s, prev);
                        if chan == -1 {
                            break;
                        }
                        copy_channel_out(pcm, nb_channels, chan as usize, Some(&buf[1..]), 2, fsz);
                        prev = chan;
                    }
                } else {
                    // Copy audio to the channel(s) where it belongs
                    let mut prev = -1;
                    loop {
                        let chan = get_mono_channel(layout, s, prev);
                        if chan == -1 {
                            break;
                        }
                        copy_channel_out(pcm, nb_channels, chan as usize, Some(&buf[..]), 1, fsz);
                        prev = chan;
                    }
                }
            }
            // Handle muted channels
            for c in 0..nb_channels {
                if layout.mapping[c] == 255 {
                    copy_channel_out(pcm, nb_channels, c, None, 0, frame_size as usize);
                }
            }
            Ok(frame_size)
        }
    }

    /// Port of `opus_multistream_decode` with C argument types (for the C ABI).
    ///
    /// # Errors
    /// As [`MsDecoder::decode`].
    #[doc(hidden)]
    pub fn opus_multistream_decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: i32,
    ) -> Result<i32> {
        self.opus_multistream_decode_native(
            data,
            pcm,
            &mut opus_copy_channel_out_short,
            frame_size,
            decode_fec,
            OPTIONAL_CLIP,
        )
    }

    /// Port of `opus_multistream_decode24` with C argument types (for the C ABI).
    ///
    /// # Errors
    /// As [`MsDecoder::decode24`].
    #[doc(hidden)]
    pub fn opus_multistream_decode24(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i32],
        frame_size: i32,
        decode_fec: i32,
    ) -> Result<i32> {
        self.opus_multistream_decode_native(
            data,
            pcm,
            &mut opus_copy_channel_out_int24,
            frame_size,
            decode_fec,
            false,
        )
    }

    /// Port of `opus_multistream_decode_float` with C argument types (for the C ABI).
    ///
    /// # Errors
    /// As [`MsDecoder::decode_float`].
    #[doc(hidden)]
    pub fn opus_multistream_decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: i32,
        decode_fec: i32,
    ) -> Result<i32> {
        self.opus_multistream_decode_native(
            data,
            pcm,
            &mut opus_copy_channel_out_float,
            frame_size,
            decode_fec,
            false,
        )
    }

    /// Decodes a multistream packet to interleaved 16-bit PCM (soft-clipped in the float build,
    /// like `opus_multistream_decode`; fixed-point builds do not clip).
    ///
    /// * `data`: the packet, or `None` for a lost packet (PLC of every stream).
    /// * `pcm`: `frame_size * channels` interleaved samples (`frame_size` limited to 120 ms).
    /// * `frame_size`, `decode_fec`: as [`Decoder::decode`].
    ///
    /// Returns the number of decoded samples per channel.
    ///
    /// # Errors
    /// [`Error::BadArg`], [`Error::BufferTooSmall`], [`Error::InvalidPacket`],
    /// [`Error::InternalError`] as in C.
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: usize,
        decode_fec: bool,
    ) -> Result<usize> {
        let fs = i32::try_from(frame_size).map_err(|_| Error::BadArg)?;
        self.opus_multistream_decode(data, pcm, fs, i32::from(decode_fec))
            .map(|n| n as usize)
    }

    /// Decodes a multistream packet to interleaved 24-bit PCM in `i32`. See
    /// [`MsDecoder::decode`].
    ///
    /// # Errors
    /// As [`MsDecoder::decode`].
    pub fn decode24(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i32],
        frame_size: usize,
        decode_fec: bool,
    ) -> Result<usize> {
        let fs = i32::try_from(frame_size).map_err(|_| Error::BadArg)?;
        self.opus_multistream_decode24(data, pcm, fs, i32::from(decode_fec))
            .map(|n| n as usize)
    }

    /// Decodes a multistream packet to interleaved float PCM. See [`MsDecoder::decode`].
    ///
    /// # Errors
    /// As [`MsDecoder::decode`].
    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: usize,
        decode_fec: bool,
    ) -> Result<usize> {
        let fs = i32::try_from(frame_size).map_err(|_| Error::BadArg)?;
        self.opus_multistream_decode_float(data, pcm, fs, i32::from(decode_fec))
            .map(|n| n as usize)
    }

    // -----------------------------------------------------------------------------------------
    // CTLs (opus_multistream_decoder_ctl_va_list)
    // -----------------------------------------------------------------------------------------

    /// The first stream's decoder (C: GET requests query the first stream).
    fn first(&self) -> &Decoder {
        // `new` guarantees at least one stream.
        &self.decoders[0]
    }

    /// `OPUS_GET_BANDWIDTH` (of the first stream).
    #[must_use]
    pub fn bandwidth(&self) -> Option<Bandwidth> {
        self.first().bandwidth()
    }

    /// `OPUS_GET_SAMPLE_RATE`.
    #[must_use]
    pub fn sample_rate(&self) -> i32 {
        self.first().sample_rate()
    }

    /// `OPUS_GET_GAIN` (of the first stream).
    #[must_use]
    pub fn gain(&self) -> i32 {
        self.first().gain()
    }

    /// `OPUS_GET_LAST_PACKET_DURATION` (of the first stream).
    #[must_use]
    pub fn last_packet_duration(&self) -> usize {
        self.first().last_packet_duration()
    }

    /// `OPUS_GET_PHASE_INVERSION_DISABLED` (of the first stream).
    #[must_use]
    pub fn phase_inversion_disabled(&self) -> bool {
        self.first().phase_inversion_disabled()
    }

    /// `OPUS_GET_COMPLEXITY` (of the first stream).
    #[must_use]
    pub fn complexity(&self) -> i32 {
        self.first().complexity()
    }

    /// `OPUS_GET_FINAL_RANGE`: the XOR of the final range of every stream.
    #[must_use]
    pub fn final_range(&self) -> u32 {
        self.decoders.iter().fold(0, |acc, d| acc ^ d.final_range())
    }

    /// `OPUS_RESET_STATE` of every stream.
    pub fn reset(&mut self) {
        for d in &mut self.decoders {
            d.reset();
        }
    }

    /// `OPUS_MULTISTREAM_GET_DECODER_STATE`: the decoder of stream `stream_id` (coupled
    /// streams first).
    ///
    /// # Errors
    /// [`Error::BadArg`] if `stream_id` is out of range.
    pub fn decoder_state(&mut self, stream_id: i32) -> Result<&mut Decoder> {
        if stream_id < 0 || stream_id >= self.layout.nb_streams {
            return Err(Error::BadArg);
        }
        Ok(&mut self.decoders[stream_id as usize])
    }

    /// Loads DNN weights into every stream decoder ([`Decoder::set_dnn_blob`]).
    ///
    /// Rust extension: C `opus_multistream_decoder_ctl` does not forward `OPUS_SET_DNN_BLOB`
    /// (upstream relies on compiled-in weights there; a `USE_WEIGHTS_FILE` build would load each
    /// stream through `OPUS_MULTISTREAM_GET_DECODER_STATE`). Every stream is attempted.
    ///
    /// # Errors
    /// [`Error::BadArg`] if a model fails to load in any stream.
    #[cfg(any(feature = "deep-plc", feature = "osce"))]
    pub fn set_dnn_blob(&mut self, data: &[u8]) -> Result<()> {
        let mut ret = Ok(());
        for d in &mut self.decoders {
            if let Err(e) = d.set_dnn_blob(data) {
                ret = Err(e);
            }
        }
        ret
    }

    /// Applies a SET request to every stream, stopping at the first error (C behaviour).
    fn set_all(&mut self, f: impl Fn(&mut Decoder) -> Result<()>) -> Result<()> {
        for d in &mut self.decoders {
            f(d)?;
        }
        Ok(())
    }

    /// `OPUS_SET_GAIN` on every stream (Q8 dB, -32768..=32767).
    ///
    /// # Errors
    /// [`Error::BadArg`] outside the range.
    pub fn set_gain(&mut self, value: i32) -> Result<()> {
        self.set_all(|d| d.set_gain(value))
    }

    /// `OPUS_SET_COMPLEXITY` on every stream (0..=10).
    ///
    /// # Errors
    /// [`Error::BadArg`] outside the range.
    pub fn set_complexity(&mut self, value: i32) -> Result<()> {
        self.set_all(|d| d.set_complexity(value))
    }

    /// `OPUS_SET_PHASE_INVERSION_DISABLED` on every stream.
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        for d in &mut self.decoders {
            d.set_phase_inversion_disabled(disabled);
        }
    }

    /// Numeric `opus_multistream_decoder_ctl` for requests taking an `opus_int32` value
    /// (`OPUS_SET_GAIN`, `OPUS_SET_COMPLEXITY`, `OPUS_SET_PHASE_INVERSION_DISABLED` — forwarded
    /// to every stream — and `OPUS_RESET_STATE`). The supported GET requests and
    /// `OPUS_MULTISTREAM_GET_DECODER_STATE` return [`Error::BadArg`]; anything else
    /// [`Error::Unimplemented`], like C.
    ///
    /// # Errors
    /// See above.
    pub fn ctl_set(&mut self, request: i32, value: i32) -> Result<()> {
        match request {
            OPUS_SET_GAIN_REQUEST
            | OPUS_SET_COMPLEXITY_REQUEST
            | OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST => {
                self.set_all(|d| d.ctl_set(request, value))
            }
            OPUS_RESET_STATE => {
                self.reset();
                Ok(())
            }
            OPUS_GET_BANDWIDTH_REQUEST
            | OPUS_GET_SAMPLE_RATE_REQUEST
            | OPUS_GET_GAIN_REQUEST
            | OPUS_GET_LAST_PACKET_DURATION_REQUEST
            | OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST
            | OPUS_GET_COMPLEXITY_REQUEST
            | OPUS_GET_FINAL_RANGE_REQUEST
            | OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST => Err(Error::BadArg),
            _ => Err(Error::Unimplemented),
        }
    }

    /// Numeric `opus_multistream_decoder_ctl` for GET requests writing one `opus_int32`
    /// (first stream), or `OPUS_GET_FINAL_RANGE` (XOR of all streams, bit-cast). SET requests
    /// and `OPUS_MULTISTREAM_GET_DECODER_STATE` return [`Error::BadArg`]; anything else
    /// [`Error::Unimplemented`], like C.
    ///
    /// # Errors
    /// See above.
    pub fn ctl_get(&mut self, request: i32) -> Result<i32> {
        match request {
            OPUS_GET_BANDWIDTH_REQUEST
            | OPUS_GET_SAMPLE_RATE_REQUEST
            | OPUS_GET_GAIN_REQUEST
            | OPUS_GET_LAST_PACKET_DURATION_REQUEST
            | OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST
            | OPUS_GET_COMPLEXITY_REQUEST => self.decoders[0].ctl_get(request),
            OPUS_GET_FINAL_RANGE_REQUEST => Ok(self.final_range() as i32),
            OPUS_SET_GAIN_REQUEST
            | OPUS_SET_COMPLEXITY_REQUEST
            | OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST
            | OPUS_RESET_STATE
            | OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST => Err(Error::BadArg),
            _ => Err(Error::Unimplemented),
        }
    }
}
