//! The ambisonics projection encoder: port of `src/opus_projection_encoder.c`.
//!
//! [`ProjectionEncoder`] is the C `OpusProjectionEncoder`: it mixes an ambisonics input
//! (mapping family 3) with a fixed mixing matrix and encodes the result with a
//! [`MsEncoder`]. The decoder needs the matching demixing matrix
//! ([`ProjectionEncoder::demixing_matrix`]), e.g. from the Ogg Opus header.
//!
//! * `opus_projection_ambisonics_encoder_create` / `_init` →
//!   [`ProjectionEncoder::new_ambisonics`].
//! * `opus_projection_encode` / `_encode24` / `_encode_float` → [`ProjectionEncoder::encode`]
//!   / [`ProjectionEncoder::encode24`] / [`ProjectionEncoder::encode_float`].
//! * `opus_projection_encoder_ctl`: the demixing matrix requests are methods; everything else
//!   goes to the multistream encoder ([`ProjectionEncoder::ms`] /
//!   [`ProjectionEncoder::ms_mut`], or [`ProjectionEncoder::ctl_set`] /
//!   [`ProjectionEncoder::ctl_get`]).
//! * `opus_projection_ambisonics_encoder_get_size` →
//!   [`projection_ambisonics_encoder_get_size`] (Rust footprint).

use alloc::vec;
use alloc::vec::Vec;

use crate::analysis::{downmix_float, downmix_int};
use crate::celt::arch::{MAX_ENCODING_DEPTH, OpusRes, OpusVal32};
use crate::celt::mathops::isqrt32;
use crate::encoder::code_to_result;
use crate::encoder::request::{
    OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST,
    OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST,
};
use crate::mapping_matrix::{
    MAPPING_MATRIX_FIFTHOA_DEMIXING, MAPPING_MATRIX_FIFTHOA_DEMIXING_DATA,
    MAPPING_MATRIX_FIFTHOA_MIXING, MAPPING_MATRIX_FIFTHOA_MIXING_DATA, MAPPING_MATRIX_FOA_DEMIXING,
    MAPPING_MATRIX_FOA_DEMIXING_DATA, MAPPING_MATRIX_FOA_MIXING, MAPPING_MATRIX_FOA_MIXING_DATA,
    MAPPING_MATRIX_FOURTHOA_DEMIXING, MAPPING_MATRIX_FOURTHOA_DEMIXING_DATA,
    MAPPING_MATRIX_FOURTHOA_MIXING, MAPPING_MATRIX_FOURTHOA_MIXING_DATA,
    MAPPING_MATRIX_SOA_DEMIXING, MAPPING_MATRIX_SOA_DEMIXING_DATA, MAPPING_MATRIX_SOA_MIXING,
    MAPPING_MATRIX_SOA_MIXING_DATA, MAPPING_MATRIX_TOA_DEMIXING, MAPPING_MATRIX_TOA_DEMIXING_DATA,
    MAPPING_MATRIX_TOA_MIXING, MAPPING_MATRIX_TOA_MIXING_DATA, MappingMatrix, MappingMatrixHeader,
    mapping_matrix_get_size, mapping_matrix_multiply_channel_in_float,
    mapping_matrix_multiply_channel_in_int24, mapping_matrix_multiply_channel_in_short,
};
use crate::ms_encoder::{MsEncoder, ms_encoder_get_size};
use crate::{Application, Error, Result};

/// Port of src/opus_projection_encoder.c:opus_projection_copy_channel_in_float.
fn opus_projection_copy_channel_in_float(
    dst: &mut [OpusRes],
    dst_stride: usize,
    src: &[f32],
    src_stride: usize,
    src_channel: usize,
    frame_size: usize,
    user_data: Option<&MappingMatrix>,
) {
    let Some(matrix) = user_data else {
        debug_assert!(
            false,
            "the projection encoder always passes its mixing matrix"
        );
        return;
    };
    mapping_matrix_multiply_channel_in_float(
        matrix,
        src,
        src_stride,
        dst,
        src_channel,
        dst_stride,
        frame_size,
    );
}

/// Port of src/opus_projection_encoder.c:opus_projection_copy_channel_in_short.
fn opus_projection_copy_channel_in_short(
    dst: &mut [OpusRes],
    dst_stride: usize,
    src: &[i16],
    src_stride: usize,
    src_channel: usize,
    frame_size: usize,
    user_data: Option<&MappingMatrix>,
) {
    let Some(matrix) = user_data else {
        debug_assert!(
            false,
            "the projection encoder always passes its mixing matrix"
        );
        return;
    };
    mapping_matrix_multiply_channel_in_short(
        matrix,
        src,
        src_stride,
        dst,
        src_channel,
        dst_stride,
        frame_size,
    );
}

/// Port of src/opus_projection_encoder.c:opus_projection_copy_channel_in_int24.
fn opus_projection_copy_channel_in_int24(
    dst: &mut [OpusRes],
    dst_stride: usize,
    src: &[i32],
    src_stride: usize,
    src_channel: usize,
    frame_size: usize,
    user_data: Option<&MappingMatrix>,
) {
    let Some(matrix) = user_data else {
        debug_assert!(
            false,
            "the projection encoder always passes its mixing matrix"
        );
        return;
    };
    mapping_matrix_multiply_channel_in_int24(
        matrix,
        src,
        src_stride,
        dst,
        src_channel,
        dst_stride,
        frame_size,
    );
}

/// `downmix_int` applied to the `opus_int32` input of `opus_projection_encode24`.
///
/// C quirk kept for bit-exactness: `opus_projection_encode24` passes `downmix_int` (not
/// `downmix_int24`) to the tonality analysis, which then reads the 32-bit sample buffer as if
/// it held 16-bit samples. This reproduces that memory reinterpretation (native byte order):
/// 16-bit element `k` is half `k % 2` of 32-bit sample `k / 2`.
fn downmix_int_on_int24(
    x: &[i32],
    y: &mut [OpusVal32],
    subframe: i32,
    offset: i32,
    c1: i32,
    c2: i32,
    c: i32,
) {
    let at = |k: i32| -> OpusVal32 {
        let k = k as usize;
        let b = x[k / 2].to_ne_bytes();
        let h = k % 2 * 2;
        i16::from_ne_bytes([b[h], b[h + 1]]) as OpusVal32
    };
    let n = subframe as usize;
    let y = &mut y[..n];
    for (j, yj) in y.iter_mut().enumerate() {
        *yj = at((j as i32 + offset) * c + c1);
    }
    if c2 > -1 {
        for (j, yj) in y.iter_mut().enumerate() {
            *yj += at((j as i32 + offset) * c + c2);
        }
    } else if c2 == -2 {
        for ch in 1..c {
            for (j, yj) in y.iter_mut().enumerate() {
                *yj += at((j as i32 + offset) * c + ch);
            }
        }
    }
}

/// Port of src/opus_projection_encoder.c:get_order_plus_one_from_channels.
fn get_order_plus_one_from_channels(channels: i32) -> Result<i32> {
    // Allowed numbers of channels:
    // (1 + n)^2 + 2j, for n = 0...14 and j = 0 or 1.
    if !(1..=227).contains(&channels) {
        return Err(Error::BadArg);
    }

    let order_plus_one = isqrt32(channels as u32) as i32;
    let acn_channels = order_plus_one * order_plus_one;
    let nondiegetic_channels = channels - acn_channels;
    if nondiegetic_channels != 0 && nondiegetic_channels != 2 {
        return Err(Error::BadArg);
    }
    Ok(order_plus_one)
}

/// Port of src/opus_projection_encoder.c:get_streams_from_channels: `(streams,
/// coupled_streams, order_plus_one)`.
fn get_streams_from_channels(channels: i32, mapping_family: i32) -> Result<(i32, i32, i32)> {
    if mapping_family == 3 {
        let order_plus_one = get_order_plus_one_from_channels(channels)?;
        return Ok(((channels + 1) / 2, channels / 2, order_plus_one));
    }
    Err(Error::BadArg)
}

/// The mixing and demixing matrix headers and data for an ambisonics order.
type MatrixPair = (
    MappingMatrixHeader,
    &'static [i16],
    MappingMatrixHeader,
    &'static [i16],
);

/// The pre-computed mixing / demixing matrices for `order_plus_one` (2..=6).
fn matrices_for_order(order_plus_one: i32) -> Option<MatrixPair> {
    Some(match order_plus_one {
        2 => (
            MAPPING_MATRIX_FOA_MIXING,
            &MAPPING_MATRIX_FOA_MIXING_DATA[..],
            MAPPING_MATRIX_FOA_DEMIXING,
            &MAPPING_MATRIX_FOA_DEMIXING_DATA[..],
        ),
        3 => (
            MAPPING_MATRIX_SOA_MIXING,
            &MAPPING_MATRIX_SOA_MIXING_DATA[..],
            MAPPING_MATRIX_SOA_DEMIXING,
            &MAPPING_MATRIX_SOA_DEMIXING_DATA[..],
        ),
        4 => (
            MAPPING_MATRIX_TOA_MIXING,
            &MAPPING_MATRIX_TOA_MIXING_DATA[..],
            MAPPING_MATRIX_TOA_DEMIXING,
            &MAPPING_MATRIX_TOA_DEMIXING_DATA[..],
        ),
        5 => (
            MAPPING_MATRIX_FOURTHOA_MIXING,
            &MAPPING_MATRIX_FOURTHOA_MIXING_DATA[..],
            MAPPING_MATRIX_FOURTHOA_DEMIXING,
            &MAPPING_MATRIX_FOURTHOA_DEMIXING_DATA[..],
        ),
        6 => (
            MAPPING_MATRIX_FIFTHOA_MIXING,
            &MAPPING_MATRIX_FIFTHOA_MIXING_DATA[..],
            MAPPING_MATRIX_FIFTHOA_DEMIXING,
            &MAPPING_MATRIX_FIFTHOA_DEMIXING_DATA[..],
        ),
        _ => return None,
    })
}

/// Port of src/opus_projection_encoder.c:opus_projection_ambisonics_encoder_get_size: the
/// memory footprint of a projection encoder, or 0 for unsupported configurations. The value is
/// the Rust footprint, not the C `sizeof`.
#[must_use]
pub fn projection_ambisonics_encoder_get_size(channels: i32, mapping_family: i32) -> usize {
    let Ok((nb_streams, nb_coupled_streams, order_plus_one)) =
        get_streams_from_channels(channels, mapping_family)
    else {
        return 0;
    };
    let Some((mixing, _, demixing, _)) = matrices_for_order(order_plus_one) else {
        return 0;
    };
    let mixing_matrix_size = mapping_matrix_get_size(mixing.rows, mixing.cols);
    if mixing_matrix_size == 0 {
        return 0;
    }
    let demixing_matrix_size = mapping_matrix_get_size(demixing.rows, demixing.cols);
    if demixing_matrix_size == 0 {
        return 0;
    }
    let encoder_size = ms_encoder_get_size(nb_streams, nb_coupled_streams);
    if encoder_size == 0 {
        return 0;
    }
    size_of::<ProjectionEncoder>()
        + mixing_matrix_size as usize
        + demixing_matrix_size as usize
        + encoder_size
}

/// An ambisonics projection encoder (port of `struct OpusProjectionEncoder`,
/// `src/opus_projection_encoder.c`).
///
/// ```
/// use opusorus::Application;
/// use opusorus::projection_encoder::ProjectionEncoder;
///
/// // First-order ambisonics (4 channels), mapping family 3.
/// let mut enc = ProjectionEncoder::new_ambisonics(48000, 4, 3, Application::Audio)?;
/// assert_eq!((enc.streams(), enc.coupled_streams()), (2, 2));
/// let matrix = enc.demixing_matrix(); // for the decoder / Ogg header
/// assert_eq!(matrix.len(), enc.demixing_matrix_size());
/// let pcm = vec![0.0f32; 960 * 4];
/// let mut packet = vec![0u8; 4000];
/// let len = enc.encode_float(&pcm, 960, &mut packet)?;
/// assert!(len >= 3);
/// # Ok::<(), opusorus::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct ProjectionEncoder {
    mixing_matrix: MappingMatrix,
    demixing_matrix: MappingMatrix,
    ms: MsEncoder,
}

/// Converts a `usize` sample count to the C `int`.
#[inline]
fn usize_to_i32(v: usize) -> Result<i32> {
    i32::try_from(v).map_err(|_| Error::BadArg)
}

impl ProjectionEncoder {
    /// Port of src/opus_projection_encoder.c:opus_projection_ambisonics_encoder_create /
    /// opus_projection_ambisonics_encoder_init: creates an ambisonics encoder.
    ///
    /// * `channels`: `(1+n)^2 + 2j` for ambisonics order `n` in 1..=5 and `j` in 0..=1.
    /// * `mapping_family`: must be 3 (projection-based ambisonics).
    ///
    /// # Errors
    /// [`Error::AllocFail`] for an unsupported channel count or mapping family (as the C
    /// `create`, whose size query fails); [`Error::BadArg`] for an invalid rate or
    /// application.
    pub fn new_ambisonics(
        fs: i32,
        channels: i32,
        mapping_family: i32,
        application: Application,
    ) -> Result<Self> {
        Self::new_ambisonics_raw(fs, channels, mapping_family, application.to_raw())
    }

    /// [`ProjectionEncoder::new_ambisonics`] taking the raw `OPUS_APPLICATION_*` value (C ABI
    /// layer).
    ///
    /// # Errors
    /// As [`ProjectionEncoder::new_ambisonics`].
    #[doc(hidden)]
    pub fn new_ambisonics_raw(
        fs: i32,
        channels: i32,
        mapping_family: i32,
        application: i32,
    ) -> Result<Self> {
        // Allocate space for the projection encoder.
        if projection_ambisonics_encoder_get_size(channels, mapping_family) == 0 {
            return Err(Error::AllocFail);
        }

        // Initialize projection encoder with provided settings.
        let (streams, coupled_streams, order_plus_one) =
            get_streams_from_channels(channels, mapping_family)?;

        if mapping_family != 3 {
            return Err(Error::Unimplemented);
        }
        // Assign mixing and demixing matrices based on available pre-computed matrices.
        let Some((mix, mix_data, demix, demix_data)) = matrices_for_order(order_plus_one) else {
            return Err(Error::BadArg);
        };
        let mixing_matrix = MappingMatrix::new(mix.rows, mix.cols, mix.gain, mix_data);
        if mapping_matrix_get_size(mixing_matrix.rows, mixing_matrix.cols) == 0 {
            return Err(Error::BadArg);
        }
        let demixing_matrix = MappingMatrix::new(demix.rows, demix.cols, demix.gain, demix_data);
        if mapping_matrix_get_size(demixing_matrix.rows, demixing_matrix.cols) == 0 {
            return Err(Error::BadArg);
        }

        // Ensure matrices are large enough for desired coding scheme.
        if streams + coupled_streams > mixing_matrix.rows
            || channels > mixing_matrix.cols
            || channels > demixing_matrix.rows
            || streams + coupled_streams > demixing_matrix.cols
        {
            return Err(Error::BadArg);
        }

        // Set trivial mapping so each input channel pairs with a matrix column.
        let mapping: Vec<u8> = (0..channels).map(|i| i as u8).collect();

        // Initialize multistream encoder with provided settings.
        let ms = MsEncoder::new_raw(
            fs,
            channels,
            streams,
            coupled_streams,
            &mapping,
            application,
        )?;
        Ok(Self {
            mixing_matrix,
            demixing_matrix,
            ms,
        })
    }

    /// Number of streams.
    #[must_use]
    pub const fn streams(&self) -> i32 {
        self.ms.streams()
    }

    /// Number of coupled (stereo) streams.
    #[must_use]
    pub const fn coupled_streams(&self) -> i32 {
        self.ms.coupled_streams()
    }

    /// Number of input channels.
    #[must_use]
    pub const fn channels(&self) -> i32 {
        self.ms.channels()
    }

    /// The underlying multistream encoder (for the CTLs `opus_projection_encoder_ctl`
    /// forwards to it).
    #[must_use]
    pub const fn ms(&self) -> &MsEncoder {
        &self.ms
    }

    /// Mutable access to the underlying multistream encoder (for the CTLs
    /// `opus_projection_encoder_ctl` forwards to it).
    pub const fn ms_mut(&mut self) -> &mut MsEncoder {
        &mut self.ms
    }

    /// Port of src/opus_projection_encoder.c:opus_projection_encode: encodes a frame of
    /// 16-bit PCM (`frame_size * channels` interleaved samples). Returns the packet length.
    ///
    /// # Errors
    /// As [`MsEncoder::encode`].
    pub fn encode(&mut self, pcm: &[i16], frame_size: usize, out: &mut [u8]) -> Result<usize> {
        let frame_size = usize_to_i32(frame_size)?;
        code_to_result(self.ms.opus_multistream_encode_native(
            opus_projection_copy_channel_in_short,
            pcm,
            frame_size,
            out,
            16,
            downmix_int,
            0,
            Some(&self.mixing_matrix),
        ))
    }

    /// Port of src/opus_projection_encoder.c:opus_projection_encode24: encodes a frame of
    /// 24-bit PCM (in `i32`). Otherwise as [`ProjectionEncoder::encode`].
    ///
    /// Like libopus, the tonality analysis of this entry point reads the 32-bit input as if it
    /// held 16-bit samples (C passes `downmix_int`); the output is identical to libopus.
    ///
    /// # Errors
    /// As [`MsEncoder::encode`].
    pub fn encode24(&mut self, pcm: &[i32], frame_size: usize, out: &mut [u8]) -> Result<usize> {
        let frame_size = usize_to_i32(frame_size)?;
        code_to_result(self.ms.opus_multistream_encode_native(
            opus_projection_copy_channel_in_int24,
            pcm,
            frame_size,
            out,
            MAX_ENCODING_DEPTH,
            downmix_int_on_int24,
            0,
            Some(&self.mixing_matrix),
        ))
    }

    /// Port of src/opus_projection_encoder.c:opus_projection_encode_float: encodes a frame of
    /// float PCM. Otherwise as [`ProjectionEncoder::encode`].
    ///
    /// # Errors
    /// As [`MsEncoder::encode`].
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        out: &mut [u8],
    ) -> Result<usize> {
        let frame_size = usize_to_i32(frame_size)?;
        code_to_result(self.ms.opus_multistream_encode_native(
            opus_projection_copy_channel_in_float,
            pcm,
            frame_size,
            out,
            MAX_ENCODING_DEPTH,
            downmix_float,
            1,
            Some(&self.mixing_matrix),
        ))
    }

    /// `OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE`: size in bytes of the demixing matrix
    /// (`channels * (streams + coupled_streams) * 2`).
    #[must_use]
    pub const fn demixing_matrix_size(&self) -> usize {
        (self.ms.channels() * (self.ms.streams() + self.ms.coupled_streams())) as usize * 2
    }

    /// `OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN`: the demixing matrix gain in dB (S7.8).
    #[must_use]
    pub const fn demixing_matrix_gain(&self) -> i32 {
        self.demixing_matrix.gain
    }

    /// `OPUS_PROJECTION_GET_DEMIXING_MATRIX`: writes the demixing matrix (little-endian
    /// 16-bit values, column-major) into `out`, whose length must be exactly
    /// [`ProjectionEncoder::demixing_matrix_size`].
    ///
    /// # Errors
    /// [`Error::BadArg`] if `out` has the wrong length.
    pub fn write_demixing_matrix(&self, out: &mut [u8]) -> Result<()> {
        // (I/O is in relation to the decoder's perspective).
        let nb_input_streams = (self.ms.streams() + self.ms.coupled_streams()) as usize;
        let nb_output_streams = self.ms.channels() as usize;

        let internal_short = self.demixing_matrix.get_data();
        let internal_size = nb_input_streams * nb_output_streams * 2;
        if out.len() != internal_size {
            return Err(Error::BadArg);
        }

        // Copy demixing matrix subset to output destination.
        let rows = self.demixing_matrix.rows as usize;
        let mut l = 0;
        for i in 0..nb_input_streams {
            for j in 0..nb_output_streams {
                let k = rows * i + j;
                out[2 * l] = internal_short[k] as u8;
                out[2 * l + 1] = (internal_short[k] >> 8) as u8;
                l += 1;
            }
        }
        Ok(())
    }

    /// The demixing matrix as a new vector (see
    /// [`ProjectionEncoder::write_demixing_matrix`]).
    #[must_use]
    pub fn demixing_matrix(&self) -> Vec<u8> {
        let mut m = vec![0u8; self.demixing_matrix_size()];
        // The buffer has exactly the right size.
        if self.write_demixing_matrix(&mut m).is_err() {
            debug_assert!(false, "buffer sized by demixing_matrix_size");
        }
        m
    }

    /// Numeric `opus_projection_encoder_ctl` for requests taking one `opus_int32` argument:
    /// forwarded to the multistream encoder ([`MsEncoder::ctl_set`]).
    ///
    /// # Errors
    /// As [`MsEncoder::ctl_set`].
    pub fn ctl_set(&mut self, request: i32, value: i32) -> Result<()> {
        self.ms.ctl_set(request, value)
    }

    /// Numeric `opus_projection_encoder_ctl` for requests writing one `opus_int32`: the
    /// demixing matrix size and gain, or forwarded to the multistream encoder
    /// ([`MsEncoder::ctl_get`]).
    ///
    /// # Errors
    /// As [`MsEncoder::ctl_get`].
    pub fn ctl_get(&self, request: i32) -> Result<i32> {
        match request {
            OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST => {
                Ok(self.demixing_matrix_size() as i32)
            }
            OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST => Ok(self.demixing_matrix_gain()),
            _ => self.ms.ctl_get(request),
        }
    }
}
