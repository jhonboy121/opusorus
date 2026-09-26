//! The projection (ambisonics) Opus decoder: port of `src/opus_projection_decoder.c`.
//!
//! [`ProjectionDecoder`] (C `OpusProjectionDecoder`) decodes the multistream packets of the
//! projection encoder (Ogg Opus channel mapping families 2 and 3) and applies the demixing
//! matrix transmitted in the stream header to recover the ambisonic channels.
//!
//! # Example
//!
//! ```
//! use opusorus::projection_decoder::ProjectionDecoder;
//!
//! # fn main() -> opusorus::Result<()> {
//! // First-order ambisonics: 4 channels in 2 coupled streams, identity demixing (Q15).
//! let mut matrix = Vec::new();
//! for col in 0..4 {
//!     for row in 0..4 {
//!         let v: i16 = if row == col { 32767 } else { 0 };
//!         matrix.extend_from_slice(&v.to_le_bytes());
//!     }
//! }
//! let mut dec = ProjectionDecoder::new(48000, 4, 2, 2, &matrix)?;
//! let mut pcm = vec![0i16; 960 * 4];
//! assert_eq!(dec.decode(None, &mut pcm, 960, false)?, 960);
//! # Ok(())
//! # }
//! ```
//!
//! # Mapping to the C API
//!
//! | C | Rust |
//! |---|---|
//! | `opus_projection_decoder_get_size` | [`ProjectionDecoder::get_size`] (Rust footprint) |
//! | `opus_projection_decoder_create` / `_init` | [`ProjectionDecoder::new`] / [`ProjectionDecoder::init`] |
//! | `opus_projection_decode` / `_decode24` / `_decode_float` | [`ProjectionDecoder::decode`] / [`ProjectionDecoder::decode24`] / [`ProjectionDecoder::decode_float`] |
//! | `opus_projection_decoder_ctl` | [`ProjectionDecoder::ms_decoder`] (every multistream CTL) and [`ProjectionDecoder::ctl_set`] / [`ProjectionDecoder::ctl_get`] |

use alloc::vec::Vec;

use crate::celt::arch::OpusRes;
use crate::mapping_matrix::{
    MappingMatrix, mapping_matrix_get_size, mapping_matrix_multiply_channel_out_float,
    mapping_matrix_multiply_channel_out_int24, mapping_matrix_multiply_channel_out_short,
};
use crate::ms_decoder::MsDecoder;
use crate::{Error, Result};

/// Port of `src/opus_projection_decoder.c:opus_projection_copy_channel_out_float`.
fn opus_projection_copy_channel_out_float(
    matrix: &MappingMatrix,
    dst: &mut [f32],
    dst_stride: usize,
    dst_channel: usize,
    src: Option<&[OpusRes]>,
    src_stride: usize,
    frame_size: usize,
) {
    if dst_channel == 0 {
        dst[..frame_size * dst_stride].fill(0.0);
    }
    if let Some(src) = src {
        mapping_matrix_multiply_channel_out_float(
            matrix,
            src,
            dst_channel,
            src_stride,
            dst,
            dst_stride,
            frame_size,
        );
    }
}

/// Port of `src/opus_projection_decoder.c:opus_projection_copy_channel_out_short`.
fn opus_projection_copy_channel_out_short(
    matrix: &MappingMatrix,
    dst: &mut [i16],
    dst_stride: usize,
    dst_channel: usize,
    src: Option<&[OpusRes]>,
    src_stride: usize,
    frame_size: usize,
) {
    if dst_channel == 0 {
        dst[..frame_size * dst_stride].fill(0);
    }
    if let Some(src) = src {
        mapping_matrix_multiply_channel_out_short(
            matrix,
            src,
            dst_channel,
            src_stride,
            dst,
            dst_stride,
            frame_size,
        );
    }
}

/// Port of `src/opus_projection_decoder.c:opus_projection_copy_channel_out_int24`.
fn opus_projection_copy_channel_out_int24(
    matrix: &MappingMatrix,
    dst: &mut [i32],
    dst_stride: usize,
    dst_channel: usize,
    src: Option<&[OpusRes]>,
    src_stride: usize,
    frame_size: usize,
) {
    if dst_channel == 0 {
        dst[..frame_size * dst_stride].fill(0);
    }
    if let Some(src) = src {
        mapping_matrix_multiply_channel_out_int24(
            matrix,
            src,
            dst_channel,
            src_stride,
            dst,
            dst_stride,
            frame_size,
        );
    }
}

/// A projection (ambisonics) Opus decoder (C `OpusProjectionDecoder`): a [`MsDecoder`] with a
/// trivial channel mapping followed by a demixing matrix.
#[derive(Debug, Clone)]
pub struct ProjectionDecoder {
    /// The demixing matrix (`get_dec_demixing_matrix`): `channels` rows x
    /// `streams + coupled_streams` columns.
    demixing_matrix: MappingMatrix,
    /// The multistream decoder (`get_multistream_decoder`).
    ms: MsDecoder,
}

impl ProjectionDecoder {
    /// Port of `src/opus_projection_decoder.c:opus_projection_decoder_create` /
    /// `opus_projection_decoder_init`.
    ///
    /// * `fs`, `channels`, `streams`, `coupled_streams`: as [`MsDecoder::new`].
    /// * `demixing_matrix`: the serialized demixing matrix as stored in the Ogg Opus header (and
    ///   returned by the projection encoder's `OPUS_PROJECTION_GET_DEMIXING_MATRIX`): column-major
    ///   little-endian `i16` values, `(streams + coupled_streams) * channels` of them.
    ///
    /// # Errors
    /// [`Error::AllocFail`] where C's size computation fails (more than 255 coded channels or
    /// invalid stream counts), [`Error::BadArg`] for a matrix of the wrong size or invalid
    /// parameters.
    pub fn new(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        demixing_matrix: &[u8],
    ) -> Result<Self> {
        // Allocate space for the projection decoder (C fails with OPUS_ALLOC_FAIL when
        // opus_projection_decoder_get_size() is 0).
        if Self::get_size(channels, streams, coupled_streams) == 0 {
            return Err(Error::AllocFail);
        }
        Self::opus_projection_decoder_init(fs, channels, streams, coupled_streams, demixing_matrix)
    }

    /// Port of `src/opus_projection_decoder.c:opus_projection_decoder_init`.
    fn opus_projection_decoder_init(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        demixing_matrix: &[u8],
    ) -> Result<Self> {
        // Verify supplied matrix size.
        let nb_input_streams = streams + coupled_streams;
        // C computes this in size_t and stores it back into an opus_int32.
        let expected_matrix_size = (nb_input_streams as i64 * channels as i64 * 2) as i32;
        if expected_matrix_size < 0 || expected_matrix_size as usize != demixing_matrix.len() {
            return Err(Error::BadArg);
        }

        // Convert demixing matrix input into internal format.
        let n = (nb_input_streams * channels) as usize;
        let buf: Vec<i16> = demixing_matrix[..2 * n]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| {
                let s = (b[1] as i32) << 8 | b[0] as i32;
                (((s & 0xFFFF) ^ 0x8000) - 0x8000) as i16
            })
            .collect();

        // Assign demixing matrix.
        if mapping_matrix_get_size(channels, nb_input_streams) == 0 {
            return Err(Error::BadArg);
        }
        let demixing_matrix = MappingMatrix::new(channels, nb_input_streams, 0, &buf);

        // Set trivial mapping so each input channel pairs with a matrix column.
        let mut mapping = [0u8; 255];
        let nch = channels.clamp(0, 255) as usize;
        for (i, m) in mapping.iter_mut().enumerate().take(nch) {
            *m = i as u8;
        }

        let ms = MsDecoder::new(fs, channels, streams, coupled_streams, &mapping[..nch])?;
        Ok(Self {
            demixing_matrix,
            ms,
        })
    }

    /// Port of `opus_projection_decoder_init`: re-initializes in place (unchanged on error).
    ///
    /// # Errors
    /// As [`ProjectionDecoder::new`].
    pub fn init(
        &mut self,
        fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        demixing_matrix: &[u8],
    ) -> Result<()> {
        *self = Self::new(fs, channels, streams, coupled_streams, demixing_matrix)?;
        Ok(())
    }

    /// Port of `opus_projection_decoder_get_size`: approximate memory footprint in bytes (Rust
    /// layout), or 0 where C returns 0 (unsupported matrix dimensions or stream counts).
    #[must_use]
    pub fn get_size(channels: i32, streams: i32, coupled_streams: i32) -> usize {
        let matrix_size = mapping_matrix_get_size(streams + coupled_streams, channels);
        if matrix_size == 0 {
            return 0;
        }
        let decoder_size = MsDecoder::get_size(streams, coupled_streams);
        if decoder_size == 0 {
            return 0;
        }
        size_of::<Self>() + matrix_size as usize + decoder_size
    }

    /// The underlying multistream decoder, for the CTLs (`opus_projection_decoder_ctl` forwards
    /// every request to it).
    #[must_use]
    pub const fn ms_decoder(&mut self) -> &mut MsDecoder {
        &mut self.ms
    }

    /// Number of output (ambisonic) channels.
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.ms.channels()
    }

    /// Port of `opus_projection_decode` with C argument types (for the C ABI).
    ///
    /// # Errors
    /// As [`ProjectionDecoder::decode`].
    #[doc(hidden)]
    pub fn opus_projection_decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: i32,
    ) -> Result<i32> {
        let m = &self.demixing_matrix;
        // FIXED_POINT: not ported (float build) — OPTIONAL_CLIP is 1.
        self.ms.opus_multistream_decode_native(
            data,
            pcm,
            &mut |dst, ds, dc, src, ss, n| {
                opus_projection_copy_channel_out_short(m, dst, ds, dc, src, ss, n);
            },
            frame_size,
            decode_fec,
            true,
        )
    }

    /// Port of `opus_projection_decode24` with C argument types (for the C ABI).
    ///
    /// # Errors
    /// As [`ProjectionDecoder::decode24`].
    #[doc(hidden)]
    pub fn opus_projection_decode24(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i32],
        frame_size: i32,
        decode_fec: i32,
    ) -> Result<i32> {
        let m = &self.demixing_matrix;
        self.ms.opus_multistream_decode_native(
            data,
            pcm,
            &mut |dst, ds, dc, src, ss, n| {
                opus_projection_copy_channel_out_int24(m, dst, ds, dc, src, ss, n);
            },
            frame_size,
            decode_fec,
            false,
        )
    }

    /// Port of `opus_projection_decode_float` with C argument types (for the C ABI).
    ///
    /// # Errors
    /// As [`ProjectionDecoder::decode_float`].
    #[doc(hidden)]
    pub fn opus_projection_decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: i32,
        decode_fec: i32,
    ) -> Result<i32> {
        let m = &self.demixing_matrix;
        self.ms.opus_multistream_decode_native(
            data,
            pcm,
            &mut |dst, ds, dc, src, ss, n| {
                opus_projection_copy_channel_out_float(m, dst, ds, dc, src, ss, n);
            },
            frame_size,
            decode_fec,
            false,
        )
    }

    /// Decodes a projection packet to interleaved 16-bit PCM (soft-clipped). Arguments and
    /// errors as [`MsDecoder::decode`].
    ///
    /// # Errors
    /// As [`MsDecoder::decode`].
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: usize,
        decode_fec: bool,
    ) -> Result<usize> {
        let fs = i32::try_from(frame_size).map_err(|_| Error::BadArg)?;
        self.opus_projection_decode(data, pcm, fs, i32::from(decode_fec))
            .map(|n| n as usize)
    }

    /// Decodes a projection packet to interleaved 24-bit PCM in `i32`. See
    /// [`ProjectionDecoder::decode`].
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
        self.opus_projection_decode24(data, pcm, fs, i32::from(decode_fec))
            .map(|n| n as usize)
    }

    /// Decodes a projection packet to interleaved float PCM. See
    /// [`ProjectionDecoder::decode`].
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
        self.opus_projection_decode_float(data, pcm, fs, i32::from(decode_fec))
            .map(|n| n as usize)
    }

    /// `OPUS_RESET_STATE` (all streams).
    pub fn reset(&mut self) {
        self.ms.reset();
    }

    /// `OPUS_GET_FINAL_RANGE` (XOR of all streams).
    #[must_use]
    pub fn final_range(&self) -> u32 {
        self.ms.final_range()
    }

    /// Numeric `opus_projection_decoder_ctl` SET requests (forwarded to
    /// [`MsDecoder::ctl_set`]).
    ///
    /// # Errors
    /// As [`MsDecoder::ctl_set`].
    pub fn ctl_set(&mut self, request: i32, value: i32) -> Result<()> {
        self.ms.ctl_set(request, value)
    }

    /// Numeric `opus_projection_decoder_ctl` GET requests (forwarded to
    /// [`MsDecoder::ctl_get`]).
    ///
    /// # Errors
    /// As [`MsDecoder::ctl_get`].
    pub fn ctl_get(&mut self, request: i32) -> Result<i32> {
        self.ms.ctl_get(request)
    }
}
