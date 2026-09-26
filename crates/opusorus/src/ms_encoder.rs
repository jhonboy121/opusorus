//! The multistream Opus encoder: port of `src/opus_multistream_encoder.c`.
//!
//! [`MsEncoder`] is the C `OpusMSEncoder`: it splits an interleaved multichannel input into
//! several coupled (stereo) and uncoupled (mono) Opus streams according to a channel mapping,
//! encodes each with an [`Encoder`], allocates the bitrate between them and concatenates the
//! packets (all but the last self-delimited) into one multistream packet.
//!
//! * `opus_multistream_encoder_create` / `_init` → [`MsEncoder::new`];
//!   `opus_multistream_surround_encoder_create` / `_init` → [`MsEncoder::new_surround`]
//!   (Vorbis channel order, LFE handling, surround masking analysis and rate allocation).
//! * `opus_multistream_encode` / `_encode24` / `_encode_float` → [`MsEncoder::encode`] /
//!   [`MsEncoder::encode24`] / [`MsEncoder::encode_float`].
//! * `opus_multistream_encoder_ctl`: typed methods, [`MsEncoder::encoder_state`] for
//!   `OPUS_MULTISTREAM_GET_ENCODER_STATE`, and [`MsEncoder::ctl_set`] / [`MsEncoder::ctl_get`].
//! * `opus_multistream_encoder_get_size` / `opus_multistream_surround_encoder_get_size` →
//!   [`ms_encoder_get_size`] / [`ms_surround_encoder_get_size`] (Rust footprints).
//!
//! The per-stream encoders are owned `Encoder`s instead of being laid out after the struct;
//! the surround window / pre-emphasis memories are owned vectors.

#![allow(
    clippy::too_many_arguments,
    reason = "internal functions mirror the C signatures"
)]
#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "`!(x < y)` is the C idiom that also catches NaN"
)]

use alloc::vec;
use alloc::vec::Vec;

use crate::analysis::{DownmixFunc, downmix_float, downmix_int, downmix_int24};
use crate::celt::arch::{
    CeltGlog, MAX_ENCODING_DEPTH, OpusRes, OpusVal16, OpusVal32, celt_isnan, half16, imax, imin,
    int16tores, int24tores, max32, maxg, min32,
};
use crate::celt::bands::compute_band_energies;
use crate::celt::celt::{bitrate_to_bits, bits_to_bitrate, resampling_factor};
use crate::celt::celt_encoder::celt_preemphasis;
use crate::celt::mathops::{celt_log2, isqrt32};
use crate::celt::mdct::clt_mdct_forward;
#[cfg(feature = "qext")]
use crate::celt::modes::QEXT_PACKET_SIZE_CAP;
use crate::celt::pitch::celt_inner_prod;
use crate::celt::quant_bands::amp2_log2;
use crate::celt::static_modes::CeltMode;
use crate::constants::raw::{
    OPUS_APPLICATION_RESTRICTED_SILK, OPUS_AUTO, OPUS_BANDWIDTH_FULLBAND,
    OPUS_BANDWIDTH_NARROWBAND, OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_WIDEBAND,
    OPUS_BITRATE_MAX, OPUS_FRAMESIZE_ARG,
};
use crate::encoder::request::*;
use crate::encoder::{Encoder, c_ignored, check_init_args, code_to_result, frame_size_select};
use crate::mapping_matrix::MappingMatrix;
use crate::multistream::{
    ChannelLayout, MappingType, get_left_channel, get_mono_channel, get_right_channel,
    validate_layout,
};
use crate::packet::{MODE_CELT_ONLY, len_i32};
use crate::repacketizer::Repacketizer;
use crate::{Application, Bandwidth, Bitrate, Error, FrameSize, Result, Signal};

/// One Vorbis channel layout (`VorbisLayout`).
struct VorbisLayout {
    nb_streams: i32,
    nb_coupled_streams: i32,
    mapping: [u8; 8],
}

/// Index is nb_channel-1 (`vorbis_mappings`).
static VORBIS_MAPPINGS: [VorbisLayout; 8] = [
    // 1: mono
    VorbisLayout {
        nb_streams: 1,
        nb_coupled_streams: 0,
        mapping: [0, 0, 0, 0, 0, 0, 0, 0],
    },
    // 2: stereo
    VorbisLayout {
        nb_streams: 1,
        nb_coupled_streams: 1,
        mapping: [0, 1, 0, 0, 0, 0, 0, 0],
    },
    // 3: 1-d surround
    VorbisLayout {
        nb_streams: 2,
        nb_coupled_streams: 1,
        mapping: [0, 2, 1, 0, 0, 0, 0, 0],
    },
    // 4: quadraphonic surround
    VorbisLayout {
        nb_streams: 2,
        nb_coupled_streams: 2,
        mapping: [0, 1, 2, 3, 0, 0, 0, 0],
    },
    // 5: 5-channel surround
    VorbisLayout {
        nb_streams: 3,
        nb_coupled_streams: 2,
        mapping: [0, 4, 1, 2, 3, 0, 0, 0],
    },
    // 6: 5.1 surround
    VorbisLayout {
        nb_streams: 4,
        nb_coupled_streams: 2,
        mapping: [0, 4, 1, 2, 3, 5, 0, 0],
    },
    // 7: 6.1 surround
    VorbisLayout {
        nb_streams: 4,
        nb_coupled_streams: 3,
        mapping: [0, 4, 1, 2, 3, 5, 6, 0],
    },
    // 8: 7.1 surround
    VorbisLayout {
        nb_streams: 5,
        nb_coupled_streams: 3,
        mapping: [0, 6, 1, 2, 3, 4, 5, 7],
    },
];

/// `MAX_OVERLAP`.
#[cfg(feature = "qext")]
const MAX_OVERLAP: usize = 240;
/// `MAX_OVERLAP`.
#[cfg(not(feature = "qext"))]
const MAX_OVERLAP: usize = 120;

/// Max size in case the encoder decides to return six frames (6 x 20 ms = 120 ms)
/// (`MS_FRAME_TMP`).
#[cfg(feature = "qext")]
const MS_FRAME_TMP: usize = 6 * QEXT_PACKET_SIZE_CAP as usize + 12;
/// Max size in case the encoder decides to return six frames (6 x 20 ms = 120 ms)
/// (`MS_FRAME_TMP`).
#[cfg(not(feature = "qext"))]
const MS_FRAME_TMP: usize = 6 * 1275 + 12;

/// `opus_copy_channel_in_func`: copies `frame_size` samples of channel `src_channel` of the
/// interleaved `src_stride`-channel input `src` to `dst[i*dst_stride]`. The last argument is
/// the C `user_data` (the projection mixing matrix).
pub(crate) type CopyChannelIn<T> =
    fn(&mut [OpusRes], usize, &[T], usize, usize, usize, Option<&MappingMatrix>);

/// Port of src/opus_multistream_encoder.c:validate_ambisonics: whether `nb_channels` is a
/// valid ambisonics channel count, with the resulting stream counts.
fn validate_ambisonics(nb_channels: i32) -> Option<(i32, i32)> {
    if !(1..=227).contains(&nb_channels) {
        return None;
    }

    let order_plus_one = isqrt32(nb_channels as u32) as i32;
    let acn_channels = order_plus_one * order_plus_one;
    let nondiegetic_channels = nb_channels - acn_channels;

    if nondiegetic_channels != 0 && nondiegetic_channels != 2 {
        return None;
    }

    Some((
        acn_channels + (nondiegetic_channels != 0) as i32,
        (nondiegetic_channels != 0) as i32,
    ))
}

/// Port of src/opus_multistream_encoder.c:validate_encoder_layout: every stream must have its
/// channels present in the mapping.
fn validate_encoder_layout(layout: &ChannelLayout) -> bool {
    for s in 0..layout.nb_streams {
        if s < layout.nb_coupled_streams {
            if get_left_channel(layout, s, -1) == -1 {
                return false;
            }
            if get_right_channel(layout, s, -1) == -1 {
                return false;
            }
        } else if get_mono_channel(layout, s, -1) == -1 {
            return false;
        }
    }
    true
}

/// Port of src/opus_multistream_encoder.c:channel_pos: position of each channel in the
/// surround mix (0: don't mix, 1: left, 2: center, 3: right).
#[doc(hidden)]
pub const fn channel_pos(channels: i32, pos: &mut [i32; 8]) {
    // Position in the mix: 0 don't mix, 1: left, 2: center, 3:right
    if channels == 4 {
        pos[0] = 1;
        pos[1] = 3;
        pos[2] = 1;
        pos[3] = 3;
    } else if channels == 3 || channels == 5 || channels == 6 {
        pos[0] = 1;
        pos[1] = 2;
        pos[2] = 3;
        pos[3] = 1;
        pos[4] = 3;
        pos[5] = 0;
    } else if channels == 7 {
        pos[0] = 1;
        pos[1] = 2;
        pos[2] = 3;
        pos[3] = 1;
        pos[4] = 3;
        pos[5] = 2;
        pos[6] = 0;
    } else if channels == 8 {
        pos[0] = 1;
        pos[1] = 2;
        pos[2] = 3;
        pos[3] = 1;
        pos[4] = 3;
        pos[5] = 1;
        pos[6] = 3;
        pos[7] = 0;
    }
}

/// Port of src/opus_multistream_encoder.c:logSum (float build): a rough approximation of
/// `log2(2^a + 2^b)`.
#[must_use]
#[doc(hidden)]
pub fn log_sum(a: CeltGlog, b: CeltGlog) -> OpusVal16 {
    static DIFF_TABLE: [CeltGlog; 17] = [
        0.5000000, 0.2924813, 0.1609640, 0.0849625, 0.0437314, 0.0221971, 0.0111839, 0.0056136,
        0.0028123, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let (max, diff) = if a > b { (a, a - b) } else { (b, b - a) };
    if !(diff < 8.0f32) {
        // inverted to catch NaNs
        return max;
    }
    // FIXED_POINT: fixed-point low/frac split not ported (float build).
    let low = crate::math::floor((2.0f32 * diff) as f64) as i32;
    let frac: CeltGlog = 2.0f32 * diff - low as f32;
    let low = low as usize;
    max + DIFF_TABLE[low] + frac * (DIFF_TABLE[low + 1] - DIFF_TABLE[low])
}

/// Scratch buffers of [`surround_analysis`] (C VLAs).
#[derive(Debug, Clone, Default)]
struct SurroundScratch {
    input: Vec<OpusVal32>,
    x: Vec<OpusRes>,
    freq: Vec<OpusVal32>,
}

/// Port of src/opus_multistream_encoder.c:surround_analysis: computes the per-channel
/// surround masking curves (`bandLogE`, 21 bands per channel) of one frame of `len` samples.
fn surround_analysis<T>(
    celt_mode: &CeltMode,
    pcm: &[T],
    band_log_e: &mut [CeltGlog],
    mem: &mut [OpusVal32],
    preemph_mem: &mut [OpusVal32],
    len: i32,
    overlap: i32,
    channels: i32,
    rate: i32,
    copy_channel_in: CopyChannelIn<T>,
    scratch: &mut SurroundScratch,
) {
    let mut pos = [0i32; 8];
    let mut mask_log_e = [[0.0 as CeltGlog; 21]; 3];

    let upsample = resampling_factor(rate);
    let frame_size = len * upsample;

    // LM = log2(frame_size / 120)
    let mut lm = 0;
    while lm < celt_mode.max_lm {
        if celt_mode.short_mdct_size << lm == frame_size {
            break;
        }
        lm += 1;
    }

    let freq_size = celt_mode.short_mdct_size << lm;
    let fsz = frame_size as usize;
    let ov = overlap as usize;
    let fq = freq_size as usize;
    if scratch.input.len() < fsz + ov {
        scratch.input.resize(fsz + ov, 0.0);
    }
    if scratch.x.len() < len as usize {
        scratch.x.resize(len as usize, 0.0);
    }
    if scratch.freq.len() < fq {
        scratch.freq.resize(fq, 0.0);
    }
    let input = &mut scratch.input[..fsz + ov];
    let x = &mut scratch.x[..len as usize];
    let freq = &mut scratch.freq[..fq];

    channel_pos(channels, &mut pos);

    for m in &mut mask_log_e {
        m.fill(-28.0);
    }

    for c in 0..channels as usize {
        let nb_frames = frame_size / freq_size;
        debug_assert!(nb_frames * freq_size == frame_size);
        input[..ov].copy_from_slice(&mem[c * ov..(c + 1) * ov]);
        copy_channel_in(x, 1, pcm, channels as usize, c, len as usize, None);
        celt_preemphasis(
            x,
            &mut input[ov..],
            frame_size,
            1,
            upsample,
            &celt_mode.preemph,
            &mut preemph_mem[c],
            false,
        );
        let sum = celt_inner_prod(input, input, fsz + ov);
        // This should filter out both NaNs and ridiculous signals that could cause NaNs
        // further down.
        if !(sum < 1e18f32) || celt_isnan(sum) {
            input.fill(0.0);
            preemph_mem[c] = 0.0;
        }
        // FIXED_POINT: the fixed build skips this check (not ported).
        let mut band_e = [0.0 as OpusVal32; 21];
        for frame in 0..nb_frames as usize {
            let mut tmp_e = [0.0 as OpusVal32; 21];
            clt_mdct_forward(
                &celt_mode.mdct,
                &input[fq * frame..],
                freq,
                &celt_mode.window,
                ov,
                (celt_mode.max_lm - lm) as usize,
                1,
            );
            if upsample != 1 {
                let bound = (freq_size / upsample) as usize;
                for f in &mut freq[..bound] {
                    *f *= upsample as f32;
                }
                freq[bound..].fill(0.0);
            }

            compute_band_energies(celt_mode, freq, &mut tmp_e, 21, 1, lm);
            // If we have multiple frames, take the max energy.
            for i in 0..21 {
                band_e[i] = max32(band_e[i], tmp_e[i]);
            }
        }
        let ble = &mut band_log_e[21 * c..21 * c + 21];
        amp2_log2(celt_mode, 21, 21, &band_e, ble, 1);
        // Apply spreading function with -6 dB/band going up and -12 dB/band going down.
        for i in 1..21 {
            ble[i] = maxg(ble[i], ble[i - 1] - 1.0f32);
        }
        for i in (0..=19).rev() {
            ble[i] = maxg(ble[i], ble[i + 1] - 2.0f32);
        }
        if pos[c] == 1 {
            for i in 0..21 {
                mask_log_e[0][i] = log_sum(mask_log_e[0][i], ble[i]);
            }
        } else if pos[c] == 3 {
            for i in 0..21 {
                mask_log_e[2][i] = log_sum(mask_log_e[2][i], ble[i]);
            }
        } else if pos[c] == 2 {
            for i in 0..21 {
                mask_log_e[0][i] = log_sum(mask_log_e[0][i], ble[i] - 0.5f32);
                mask_log_e[2][i] = log_sum(mask_log_e[2][i], ble[i] - 0.5f32);
            }
        }
        mem[c * ov..(c + 1) * ov].copy_from_slice(&input[fsz..fsz + ov]);
    }
    for i in 0..21 {
        mask_log_e[1][i] = min32(mask_log_e[0][i], mask_log_e[2][i]);
    }
    let channel_offset: OpusVal16 = half16(celt_log2(2.0f32 / (channels - 1) as f32));
    for m in &mut mask_log_e {
        for v in m.iter_mut() {
            *v += channel_offset;
        }
    }
    for c in 0..channels as usize {
        let ble = &mut band_log_e[21 * c..21 * c + 21];
        if pos[c] != 0 {
            let mask = &mask_log_e[(pos[c] - 1) as usize];
            for i in 0..21 {
                ble[i] -= mask[i];
            }
        } else {
            ble.fill(0.0);
        }
    }
}

/// [`surround_analysis`] on interleaved float input with the default CELT mode for `rate`
/// (the 96 kHz mode at 96 kHz with QEXT, else the 48 kHz mode). `mem` holds
/// `channels*overlap` values, `preemph_mem` `channels`, `band_log_e` `21*channels`. For
/// differential tests.
///
/// # Errors
/// [`Error::BadArg`] if the mode cannot be created or a buffer is too short.
#[doc(hidden)]
pub fn surround_analysis_float(
    pcm: &[f32],
    band_log_e: &mut [CeltGlog],
    mem: &mut [OpusVal32],
    preemph_mem: &mut [OpusVal32],
    len: i32,
    channels: i32,
    rate: i32,
) -> Result<()> {
    let mode = if rate == 96000 {
        crate::celt::modes::opus_custom_mode_create(96000, 1920)?
    } else {
        crate::celt::modes::opus_custom_mode_create(48000, 960)?
    };
    let ch = channels as usize;
    if pcm.len() < len as usize * ch
        || band_log_e.len() < 21 * ch
        || mem.len() < mode.overlap as usize * ch
        || preemph_mem.len() < ch
    {
        return Err(Error::BadArg);
    }
    let mut scratch = SurroundScratch::default();
    surround_analysis(
        mode,
        pcm,
        band_log_e,
        mem,
        preemph_mem,
        len,
        mode.overlap,
        channels,
        rate,
        opus_copy_channel_in_float,
        &mut scratch,
    );
    Ok(())
}

/// Port of src/opus_multistream_encoder.c:opus_multistream_encoder_get_size: the memory
/// footprint of a multistream encoder, or 0 for invalid stream counts. The value is the Rust
/// footprint, not the C `sizeof`.
#[must_use]
pub fn ms_encoder_get_size(nb_streams: i32, nb_coupled_streams: i32) -> usize {
    if nb_streams < 1 || nb_coupled_streams > nb_streams || nb_coupled_streams < 0 {
        return 0;
    }
    let coupled_size = crate::encoder::encoder_get_size(2);
    let mono_size = crate::encoder::encoder_get_size(1);
    size_of::<MsEncoder>()
        + nb_coupled_streams as usize * coupled_size
        + (nb_streams - nb_coupled_streams) as usize * mono_size
}

/// Port of src/opus_multistream_encoder.c:opus_multistream_surround_encoder_get_size: the
/// memory footprint of a surround encoder, or 0 for unsupported configurations. The value is
/// the Rust footprint, not the C `sizeof`.
#[must_use]
pub fn ms_surround_encoder_get_size(channels: i32, mapping_family: i32) -> usize {
    let (nb_streams, nb_coupled_streams) = if mapping_family == 0 {
        if channels == 1 {
            (1, 0)
        } else if channels == 2 {
            (1, 1)
        } else {
            return 0;
        }
    } else if mapping_family == 1 && (1..=8).contains(&channels) {
        let l = &VORBIS_MAPPINGS[(channels - 1) as usize];
        (l.nb_streams, l.nb_coupled_streams)
    } else if mapping_family == 255 {
        (channels, 0)
    } else if mapping_family == 2 {
        match validate_ambisonics(channels) {
            Some(v) => v,
            None => return 0,
        }
    } else {
        return 0;
    };
    let mut size = ms_encoder_get_size(nb_streams, nb_coupled_streams);
    if channels > 2 {
        size += channels as usize * (MAX_OVERLAP * size_of::<OpusVal32>() + size_of::<OpusVal32>());
    }
    size
}

/// Scratch buffers of `opus_multistream_encode_native` (C VLAs / stack arrays).
#[derive(Debug, Clone, Default)]
struct MsScratch {
    /// `buf`: `2*frame_size`.
    buf: Vec<OpusRes>,
    /// `bandSMR`: `21*nb_channels`.
    band_smr: Vec<CeltGlog>,
    /// `tmp_data[MS_FRAME_TMP]`.
    tmp_data: Vec<u8>,
    surround: SurroundScratch,
}

/// A multistream Opus encoder (port of `struct OpusMSEncoder`,
/// `src/opus_multistream_encoder.c`).
///
/// ```
/// use opusorus::ms_encoder::MsEncoder;
/// use opusorus::{Application, Bitrate};
///
/// // 5.1 surround in Vorbis channel order (mapping family 1).
/// let mut enc = MsEncoder::new_surround(48000, 6, 1, Application::Audio)?;
/// assert_eq!((enc.streams(), enc.coupled_streams()), (4, 2));
/// enc.set_bitrate(Bitrate::Bits(256000))?;
/// let pcm = vec![0i16; 960 * 6];
/// let mut packet = vec![0u8; 4000];
/// let len = enc.encode(&pcm, 960, &mut packet)?;
/// assert!(len >= 7);
/// # Ok::<(), opusorus::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct MsEncoder {
    layout: ChannelLayout,
    lfe_stream: i32,
    application: i32,
    // C `Fs` is only used for the struct size computations; the rate is read back from the
    // first stream encoder, as in C.
    variable_duration: i32,
    mapping_type: MappingType,
    bitrate_bps: i32,
    /// The stream encoders (coupled streams first).
    encoders: Vec<Encoder>,
    /// `window_mem`: `channels*MAX_OVERLAP` (surround only).
    window_mem: Vec<OpusVal32>,
    /// `preemph_mem`: `channels` (surround only).
    preemph_mem: Vec<OpusVal32>,
    scratch: MsScratch,
}

/// Converts a `usize` sample count to the C `int`.
#[inline]
fn usize_to_i32(v: usize) -> Result<i32> {
    i32::try_from(v).map_err(|_| Error::BadArg)
}

/// Port of src/opus_multistream_encoder.c:opus_copy_channel_in_float.
fn opus_copy_channel_in_float(
    dst: &mut [OpusRes],
    dst_stride: usize,
    src: &[f32],
    src_stride: usize,
    src_channel: usize,
    frame_size: usize,
    _user_data: Option<&MappingMatrix>,
) {
    for i in 0..frame_size {
        // FLOAT2RES is the identity in the float build.
        dst[i * dst_stride] = src[i * src_stride + src_channel];
    }
}

/// Port of src/opus_multistream_encoder.c:opus_copy_channel_in_short.
fn opus_copy_channel_in_short(
    dst: &mut [OpusRes],
    dst_stride: usize,
    src: &[i16],
    src_stride: usize,
    src_channel: usize,
    frame_size: usize,
    _user_data: Option<&MappingMatrix>,
) {
    for i in 0..frame_size {
        dst[i * dst_stride] = int16tores(src[i * src_stride + src_channel]);
    }
}

/// Port of src/opus_multistream_encoder.c:opus_copy_channel_in_int24.
fn opus_copy_channel_in_int24(
    dst: &mut [OpusRes],
    dst_stride: usize,
    src: &[i32],
    src_stride: usize,
    src_channel: usize,
    frame_size: usize,
    _user_data: Option<&MappingMatrix>,
) {
    for i in 0..frame_size {
        dst[i * dst_stride] = int24tores(src[i * src_stride + src_channel]);
    }
}

impl MsEncoder {
    /// Port of src/opus_multistream_encoder.c:opus_multistream_encoder_create /
    /// opus_multistream_encoder_init: creates a multistream encoder.
    ///
    /// * `channels`: input channels (1..=255).
    /// * `streams` / `coupled_streams`: total and coupled (stereo) stream counts
    ///   (`streams + coupled_streams <= channels`).
    /// * `mapping`: for each input channel, the coded channel it goes to (values below
    ///   `2*coupled_streams` are the left/right channels of the coupled streams, then the mono
    ///   streams; 255 = unused). Must hold at least `channels` entries.
    ///
    /// # Errors
    /// [`Error::BadArg`] for invalid counts, rate, application or mapping (every stream must
    /// be fed by at least one input channel).
    pub fn new(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
        application: Application,
    ) -> Result<Self> {
        Self::new_raw(
            fs,
            channels,
            streams,
            coupled_streams,
            mapping,
            application.to_raw(),
        )
    }

    /// [`MsEncoder::new`] taking the raw `OPUS_APPLICATION_*` value (C ABI layer).
    ///
    /// # Errors
    /// As [`MsEncoder::new`].
    #[doc(hidden)]
    pub fn new_raw(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
        application: i32,
    ) -> Result<Self> {
        if !(1..=255).contains(&channels)
            || coupled_streams > streams
            || streams < 1
            || coupled_streams < 0
            || streams > 255 - coupled_streams
            || streams + coupled_streams > channels
        {
            return Err(Error::BadArg);
        }
        if mapping.len() < channels as usize {
            // Rust-only check (C reads out of bounds).
            return Err(Error::BadArg);
        }
        Self::init_impl(
            fs,
            channels,
            streams,
            coupled_streams,
            mapping,
            application,
            MappingType::None,
            -1,
        )
    }

    /// Port of src/opus_multistream_encoder.c:opus_multistream_encoder_init_impl.
    fn init_impl(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
        application: i32,
        mapping_type: MappingType,
        lfe_stream: i32,
    ) -> Result<Self> {
        if !(1..=255).contains(&channels)
            || coupled_streams > streams
            || streams < 1
            || coupled_streams < 0
            || streams > 255 - coupled_streams
            || streams + coupled_streams > channels
        {
            return Err(Error::BadArg);
        }
        // C: opus_encoder_init(NULL, Fs, 2/1, application) validates the rate and application.
        check_init_args(fs, 2, application)?;
        check_init_args(fs, 1, application)?;

        let mut layout = ChannelLayout {
            nb_channels: channels,
            nb_streams: streams,
            nb_coupled_streams: coupled_streams,
            ..ChannelLayout::default()
        };
        let lfe_stream = if mapping_type != MappingType::Surround {
            -1
        } else {
            lfe_stream
        };
        layout.mapping[..channels as usize].copy_from_slice(&mapping[..channels as usize]);
        if !validate_layout(&layout) {
            return Err(Error::BadArg);
        }
        if !validate_encoder_layout(&layout) {
            return Err(Error::BadArg);
        }
        if mapping_type == MappingType::Ambisonics
            && validate_ambisonics(layout.nb_channels).is_none()
        {
            return Err(Error::BadArg);
        }
        let mut encoders = Vec::with_capacity(streams as usize);
        for i in 0..streams {
            let mut enc =
                Encoder::new_raw(fs, if i < coupled_streams { 2 } else { 1 }, application)?;
            if i == lfe_stream {
                enc.set_lfe(1);
            }
            encoders.push(enc);
        }
        let surround = mapping_type == MappingType::Surround;
        let ch = channels as usize;
        let max_frame = (6 * fs / 50) as usize;
        let mut scratch = MsScratch {
            buf: vec![0.0; 2 * max_frame],
            band_smr: vec![0.0; 21 * ch],
            tmp_data: vec![0; MS_FRAME_TMP],
            surround: SurroundScratch::default(),
        };
        if surround
            && application != OPUS_APPLICATION_RESTRICTED_SILK
            && let Some(mode) = encoders[0].celt_mode()
        {
            let mode_frame = (6 * mode.fs / 50) as usize;
            scratch.surround = SurroundScratch {
                input: vec![0.0; mode_frame + mode.overlap as usize],
                x: vec![0.0; max_frame],
                freq: vec![0.0; (mode.short_mdct_size << mode.max_lm) as usize],
            };
        }
        Ok(Self {
            layout,
            lfe_stream,
            application,
            variable_duration: OPUS_FRAMESIZE_ARG,
            mapping_type,
            bitrate_bps: OPUS_AUTO,
            encoders,
            window_mem: if surround {
                vec![0.0; ch * MAX_OVERLAP]
            } else {
                Vec::new()
            },
            preemph_mem: if surround { vec![0.0; ch] } else { Vec::new() },
            scratch,
        })
    }

    /// Port of src/opus_multistream_encoder.c:opus_multistream_surround_encoder_create /
    /// opus_multistream_surround_encoder_init: creates an encoder for a standard channel
    /// mapping family and computes the stream layout. Read it back with
    /// [`MsEncoder::streams`], [`MsEncoder::coupled_streams`] and [`MsEncoder::mapping`].
    ///
    /// * Family 0: mono or stereo (1 or 2 channels).
    /// * Family 1: Vorbis channel order, 1 to 8 channels (surround masking and LFE handling
    ///   for 3 or more channels).
    /// * Family 2: ambisonics (`(1+n)^2 + 2j` channels), one stream per channel.
    /// * Family 255: `channels` independent mono streams.
    ///
    /// # Errors
    /// [`Error::BadArg`] for invalid channel counts, rate or application;
    /// [`Error::Unimplemented`] for unsupported families (or family 0 with more than 2
    /// channels).
    pub fn new_surround(
        fs: i32,
        channels: i32,
        mapping_family: i32,
        application: Application,
    ) -> Result<Self> {
        Self::new_surround_raw(fs, channels, mapping_family, application.to_raw())
    }

    /// [`MsEncoder::new_surround`] taking the raw `OPUS_APPLICATION_*` value (C ABI layer).
    ///
    /// # Errors
    /// As [`MsEncoder::new_surround`].
    #[doc(hidden)]
    pub fn new_surround_raw(
        fs: i32,
        channels: i32,
        mapping_family: i32,
        application: i32,
    ) -> Result<Self> {
        if !(1..=255).contains(&channels) {
            return Err(Error::BadArg);
        }
        let mut mapping = [0u8; 255];
        let mut lfe_stream = -1;
        let (streams, coupled_streams);
        if mapping_family == 0 {
            if channels == 1 {
                streams = 1;
                coupled_streams = 0;
                mapping[0] = 0;
            } else if channels == 2 {
                streams = 1;
                coupled_streams = 1;
                mapping[0] = 0;
                mapping[1] = 1;
            } else {
                return Err(Error::Unimplemented);
            }
        } else if mapping_family == 1 && (1..=8).contains(&channels) {
            let l = &VORBIS_MAPPINGS[(channels - 1) as usize];
            streams = l.nb_streams;
            coupled_streams = l.nb_coupled_streams;
            mapping[..channels as usize].copy_from_slice(&l.mapping[..channels as usize]);
            if channels >= 6 {
                lfe_stream = streams - 1;
            }
        } else if mapping_family == 255 {
            streams = channels;
            coupled_streams = 0;
            for (i, m) in mapping[..channels as usize].iter_mut().enumerate() {
                *m = i as u8;
            }
        } else if mapping_family == 2 {
            let Some((s, c)) = validate_ambisonics(channels) else {
                return Err(Error::BadArg);
            };
            streams = s;
            coupled_streams = c;
            for i in 0..(streams - coupled_streams) {
                mapping[i as usize] = (i + coupled_streams * 2) as u8;
            }
            for i in 0..coupled_streams * 2 {
                mapping[(i + (streams - coupled_streams)) as usize] = i as u8;
            }
        } else {
            return Err(Error::Unimplemented);
        }

        let mapping_type = if channels > 2 && mapping_family == 1 {
            MappingType::Surround
        } else if mapping_family == 2 {
            MappingType::Ambisonics
        } else {
            MappingType::None
        };
        Self::init_impl(
            fs,
            channels,
            streams,
            coupled_streams,
            &mapping,
            application,
            mapping_type,
            lfe_stream,
        )
    }

    /// Number of input channels.
    #[must_use]
    pub const fn channels(&self) -> i32 {
        self.layout.nb_channels
    }

    /// Number of streams.
    #[must_use]
    pub const fn streams(&self) -> i32 {
        self.layout.nb_streams
    }

    /// Number of coupled (stereo) streams.
    #[must_use]
    pub const fn coupled_streams(&self) -> i32 {
        self.layout.nb_coupled_streams
    }

    /// The channel mapping (one entry per input channel), as needed by the decoder.
    #[must_use]
    pub fn mapping(&self) -> &[u8] {
        &self.layout.mapping[..self.layout.nb_channels as usize]
    }

    // -----------------------------------------------------------------------------------------
    // Rate allocation
    // -----------------------------------------------------------------------------------------

    /// Port of src/opus_multistream_encoder.c:surround_rate_allocation.
    fn surround_rate_allocation(&self, rate: &mut [i32], frame_size: i32, fs: i32) {
        let nb_lfe = (self.lfe_stream != -1) as i32;
        let nb_coupled = self.layout.nb_coupled_streams;
        let nb_uncoupled = self.layout.nb_streams - nb_coupled - nb_lfe;
        let nb_normal = 2 * nb_coupled + nb_uncoupled;

        // Give each non-LFE channel enough bits per channel for coding band energy.
        let channel_offset = 40 * imax(50, fs / frame_size);

        let bitrate = if self.bitrate_bps == OPUS_AUTO {
            nb_normal * (channel_offset + fs + 10000) + 8000 * nb_lfe
        } else if self.bitrate_bps == OPUS_BITRATE_MAX {
            nb_normal * 750000 + nb_lfe * 128000
        } else {
            self.bitrate_bps
        };

        // Give LFE some basic stream_channel allocation but never exceed 1/20 of the total
        // rate for the non-energy part to avoid problems at really low rate.
        let lfe_offset = imin(bitrate / 20, 3000) + 15 * imax(50, fs / frame_size);

        // We give each stream (coupled or uncoupled) a starting bitrate. This models the main
        // saving of coupled channels over uncoupled.
        let mut stream_offset =
            (bitrate - channel_offset * nb_normal - lfe_offset * nb_lfe) / nb_normal / 2;
        stream_offset = imax(0, imin(20000, stream_offset));

        // Coupled streams get twice the mono rate after the offset is allocated.
        let coupled_ratio = 512;
        // Should depend on the bitrate, for now we assume LFE gets 1/8 the bits of mono
        let lfe_ratio = 32;

        let total = (nb_uncoupled << 8) // mono
            + coupled_ratio * nb_coupled // stereo
            + nb_lfe * lfe_ratio;
        let channel_rate = (256
            * (bitrate
                - lfe_offset * nb_lfe
                - stream_offset * (nb_coupled + nb_uncoupled)
                - channel_offset * nb_normal) as i64
            / total as i64) as i32;

        for i in 0..self.layout.nb_streams {
            rate[i as usize] = if i < self.layout.nb_coupled_streams {
                2 * channel_offset + imax(0, stream_offset + ((channel_rate * coupled_ratio) >> 8))
            } else if i != self.lfe_stream {
                channel_offset + imax(0, stream_offset + channel_rate)
            } else {
                imax(0, lfe_offset + ((channel_rate * lfe_ratio) >> 8))
            };
        }
    }

    /// Port of src/opus_multistream_encoder.c:ambisonics_rate_allocation.
    fn ambisonics_rate_allocation(&self, rate: &mut [i32], frame_size: i32, fs: i32) {
        let nb_channels = self.layout.nb_streams + self.layout.nb_coupled_streams;

        let total_rate = if self.bitrate_bps == OPUS_AUTO {
            (self.layout.nb_coupled_streams + self.layout.nb_streams) * (fs + 60 * fs / frame_size)
                + self.layout.nb_streams * 15000
        } else if self.bitrate_bps == OPUS_BITRATE_MAX {
            nb_channels * 750000
        } else {
            self.bitrate_bps
        };

        // Allocate equal number of bits to Ambisonic (uncoupled) and non-diegetic (coupled)
        // streams
        let per_stream_rate = total_rate / self.layout.nb_streams;
        for r in &mut rate[..self.layout.nb_streams as usize] {
            *r = per_stream_rate;
        }
    }

    /// Port of src/opus_multistream_encoder.c:rate_allocation: per-stream bitrates; returns
    /// their sum.
    fn rate_allocation(&self, rate: &mut [i32], frame_size: i32) -> i32 {
        let fs = self.encoders[0].sample_rate();

        if self.mapping_type == MappingType::Ambisonics {
            self.ambisonics_rate_allocation(rate, frame_size, fs);
        } else {
            self.surround_rate_allocation(rate, frame_size, fs);
        }

        let mut rate_sum = 0;
        for r in &mut rate[..self.layout.nb_streams as usize] {
            *r = imax(*r, 500);
            rate_sum += *r;
        }
        rate_sum
    }

    // -----------------------------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------------------------

    /// Port of src/opus_multistream_encoder.c:opus_multistream_encode_native, with the C
    /// semantics (returns the packet length or a negative `OPUS_*` error code). `data.len()`
    /// is `max_data_bytes`.
    #[expect(
        clippy::too_many_lines,
        reason = "one C function; splitting it would obscure the correspondence"
    )]
    pub(crate) fn opus_multistream_encode_native<T>(
        &mut self,
        copy_channel_in: CopyChannelIn<T>,
        pcm: &[T],
        analysis_frame_size: i32,
        data: &mut [u8],
        lsb_depth: i32,
        downmix: DownmixFunc<T>,
        float_api: i32,
        user_data: Option<&MappingMatrix>,
    ) -> i32 {
        let mut max_data_bytes = len_i32(data.len());
        let fs = self.encoders[0].sample_rate();
        let vbr = self.encoders[0].vbr() as i32;

        let frame_size = frame_size_select(
            self.application,
            analysis_frame_size,
            self.variable_duration,
            fs,
        );
        if frame_size <= 0 {
            return Error::BadArg.code();
        }

        // Smallest packet the encoder can produce.
        let mut smallest_packet = self.layout.nb_streams * 2 - 1;
        // 100 ms needs an extra byte per stream for the ToC.
        if fs / frame_size == 10 {
            smallest_packet += self.layout.nb_streams;
        }
        if max_data_bytes < smallest_packet {
            return Error::BufferTooSmall.code();
        }
        let nb_channels = self.layout.nb_channels;
        if pcm.len() < analysis_frame_size as usize * nb_channels as usize {
            // Rust-only check (C reads out of bounds).
            return Error::BadArg.code();
        }
        let fsu = frame_size as usize;

        let Self {
            encoders,
            scratch,
            window_mem,
            preemph_mem,
            ..
        } = self;
        if scratch.buf.len() < 2 * fsu {
            scratch.buf.resize(2 * fsu, 0.0);
        }
        let surround = self.mapping_type == MappingType::Surround
            && self.application != OPUS_APPLICATION_RESTRICTED_SILK;
        if surround {
            let Some(celt_mode) = encoders[0].celt_mode() else {
                return Error::InternalError.code();
            };
            surround_analysis(
                celt_mode,
                pcm,
                &mut scratch.band_smr,
                window_mem,
                preemph_mem,
                frame_size,
                celt_mode.overlap,
                nb_channels,
                fs,
                copy_channel_in,
                &mut scratch.surround,
            );
        }

        // Compute bitrate allocation between streams (this could be a lot better)
        let mut bitrates = [0i32; 256];
        let rate_sum = self.rate_allocation(&mut bitrates, frame_size);
        let Self {
            layout,
            encoders,
            scratch,
            ..
        } = self;

        if vbr == 0 {
            if self.bitrate_bps == OPUS_AUTO {
                max_data_bytes = imin(
                    max_data_bytes,
                    (bitrate_to_bits(rate_sum, fs, frame_size) + 4) / 8,
                );
            } else if self.bitrate_bps != OPUS_BITRATE_MAX {
                max_data_bytes = imin(
                    max_data_bytes,
                    imax(
                        smallest_packet,
                        (bitrate_to_bits(self.bitrate_bps, fs, frame_size) + 4) / 8,
                    ),
                );
            }
        }
        for (s, enc) in encoders.iter_mut().enumerate() {
            // C ignores the return values of these CTLs.
            if let Err(e) = c_ignored(enc.set_bitrate(Bitrate::from_raw(bitrates[s]))) {
                return e.code();
            }
            if self.mapping_type == MappingType::Surround {
                let mut equiv_rate = self.bitrate_bps;
                if frame_size * 50 < fs {
                    equiv_rate -= 60 * (fs / frame_size - 50) * layout.nb_channels;
                }
                let bw = if equiv_rate > 10000 * layout.nb_channels {
                    OPUS_BANDWIDTH_FULLBAND
                } else if equiv_rate > 7000 * layout.nb_channels {
                    OPUS_BANDWIDTH_SUPERWIDEBAND
                } else if equiv_rate > 5000 * layout.nb_channels {
                    OPUS_BANDWIDTH_WIDEBAND
                } else {
                    OPUS_BANDWIDTH_NARROWBAND
                };
                if let Err(e) = c_ignored(enc.ctl_set(OPUS_SET_BANDWIDTH_REQUEST, bw)) {
                    return e.code();
                }
                if (s as i32) < layout.nb_coupled_streams {
                    // To preserve the spatial image, force stereo CELT on coupled streams
                    if let Err(e) = c_ignored(enc.set_force_mode(MODE_CELT_ONLY)) {
                        return e.code();
                    }
                    if let Err(e) = c_ignored(enc.set_force_channels(Some(2))) {
                        return e.code();
                    }
                }
            } else if self.mapping_type == MappingType::Ambisonics
                && let Err(e) = c_ignored(enc.set_force_mode(MODE_CELT_ONLY))
            {
                return e.code();
            }
        }

        // Counting ToC
        let mut tot_size: i32 = 0;
        let mut band_log_e = [0.0 as CeltGlog; 42];
        let nb_streams = layout.nb_streams;
        for s in 0..nb_streams {
            let enc = &mut encoders[s as usize];
            let (c1, c2);
            let enc_channels;
            if s < layout.nb_coupled_streams {
                let left = get_left_channel(layout, s, -1);
                let right = get_right_channel(layout, s, -1);
                copy_channel_in(
                    &mut scratch.buf,
                    2,
                    pcm,
                    nb_channels as usize,
                    left as usize,
                    fsu,
                    user_data,
                );
                copy_channel_in(
                    &mut scratch.buf[1..],
                    2,
                    pcm,
                    nb_channels as usize,
                    right as usize,
                    fsu,
                    user_data,
                );
                if surround {
                    for i in 0..21 {
                        band_log_e[i] = scratch.band_smr[21 * left as usize + i];
                        band_log_e[21 + i] = scratch.band_smr[21 * right as usize + i];
                    }
                }
                c1 = left;
                c2 = right;
                enc_channels = 2;
            } else {
                let chan = get_mono_channel(layout, s, -1);
                copy_channel_in(
                    &mut scratch.buf,
                    1,
                    pcm,
                    nb_channels as usize,
                    chan as usize,
                    fsu,
                    user_data,
                );
                if surround {
                    for i in 0..21 {
                        band_log_e[i] = scratch.band_smr[21 * chan as usize + i];
                    }
                }
                c1 = chan;
                c2 = -1;
                enc_channels = 1;
            }
            if surround {
                enc.set_energy_mask(Some(&band_log_e));
            }
            // number of bytes left (+Toc)
            let mut curr_max = max_data_bytes - tot_size;
            // Reserve one byte for the last stream and two for the others
            curr_max -= imax(0, 2 * (nb_streams - s - 1) - 1);
            // For 100 ms, reserve an extra byte per stream for the ToC
            if fs / frame_size == 10 {
                curr_max -= nb_streams - s - 1;
            }
            curr_max = imin(curr_max, MS_FRAME_TMP as i32);
            // Repacketizer will add one or two bytes for self-delimited frames
            if s != nb_streams - 1 {
                curr_max -= if curr_max > 253 { 2 } else { 1 };
            }
            if vbr == 0 && s == nb_streams - 1 {
                // C ignores the BAD_ARG of a non-positive bitrate.
                let r = enc.set_bitrate(Bitrate::from_raw(bits_to_bitrate(
                    curr_max * 8,
                    fs,
                    frame_size,
                )));
                if let Err(e) = c_ignored(r) {
                    return e.code();
                }
            }
            let len = enc.opus_encode_native(
                &scratch.buf[..fsu * enc_channels],
                frame_size,
                &mut scratch.tmp_data,
                curr_max,
                lsb_depth,
                Some(pcm),
                analysis_frame_size,
                c1,
                c2,
                nb_channels,
                downmix,
                float_api,
            );
            if len < 0 {
                return len;
            }
            // We need to use the repacketizer to add the self-delimiting lengths while taking
            // into account the fact that the encoder can now return more than one frame at a
            // time (e.g. 60 ms CELT-only)
            let mut rp = Repacketizer::new();
            // If the opus_repacketizer_cat() fails, then something's seriously wrong with the
            // encoder.
            if rp.cat(&scratch.tmp_data[..len as usize]).is_err() {
                return Error::InternalError.code();
            }
            let off = tot_size as usize;
            let len = match rp.out_range_impl(
                0,
                rp.nb_frames(),
                &mut data[off..max_data_bytes as usize],
                s != nb_streams - 1,
                vbr == 0 && s == nb_streams - 1,
                &[],
            ) {
                Ok(n) => n,
                Err(e) => e.code(),
            };
            // C adds the (possibly negative) result without checking it.
            tot_size += len;
            if tot_size < 0 {
                // C would move its output pointer before the buffer (undefined behaviour).
                return Error::InternalError.code();
            }
        }
        tot_size
    }

    /// Port of src/opus_multistream_encoder.c:opus_multistream_encode: encodes a frame of
    /// 16-bit PCM (`frame_size * channels` interleaved samples). Returns the packet length.
    ///
    /// # Errors
    /// [`Error::BadArg`] for an invalid frame size or a too-short `pcm`;
    /// [`Error::BufferTooSmall`] if `out` is too small; other errors of the stream encoders.
    pub fn encode(&mut self, pcm: &[i16], frame_size: usize, out: &mut [u8]) -> Result<usize> {
        let frame_size = usize_to_i32(frame_size)?;
        code_to_result(self.opus_multistream_encode_native(
            opus_copy_channel_in_short,
            pcm,
            frame_size,
            out,
            16,
            downmix_int,
            0,
            None,
        ))
    }

    /// Port of src/opus_multistream_encoder.c:opus_multistream_encode24: encodes a frame of
    /// 24-bit PCM (in `i32`). Otherwise as [`MsEncoder::encode`].
    ///
    /// # Errors
    /// As [`MsEncoder::encode`].
    pub fn encode24(&mut self, pcm: &[i32], frame_size: usize, out: &mut [u8]) -> Result<usize> {
        let frame_size = usize_to_i32(frame_size)?;
        code_to_result(self.opus_multistream_encode_native(
            opus_copy_channel_in_int24,
            pcm,
            frame_size,
            out,
            MAX_ENCODING_DEPTH,
            downmix_int24,
            0,
            None,
        ))
    }

    /// Port of src/opus_multistream_encoder.c:opus_multistream_encode_float: encodes a frame
    /// of float PCM. Otherwise as [`MsEncoder::encode`].
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
        code_to_result(self.opus_multistream_encode_native(
            opus_copy_channel_in_float,
            pcm,
            frame_size,
            out,
            MAX_ENCODING_DEPTH,
            downmix_float,
            1,
            None,
        ))
    }

    // -----------------------------------------------------------------------------------------
    // CTLs (src/opus_multistream_encoder.c:opus_multistream_encoder_ctl_va_list)
    // -----------------------------------------------------------------------------------------

    /// Applies a SET request to every stream encoder, stopping at the first error.
    fn for_each_stream(&mut self, mut f: impl FnMut(&mut Encoder) -> Result<()>) -> Result<()> {
        for enc in &mut self.encoders {
            f(enc)?;
        }
        Ok(())
    }

    /// `OPUS_SET_BITRATE`: total bitrate over all streams (explicit values are clamped to
    /// 500..=750000 b/s per channel).
    ///
    /// # Errors
    /// [`Error::BadArg`] for explicit values <= 0.
    pub const fn set_bitrate(&mut self, bitrate: Bitrate) -> Result<()> {
        let mut value = bitrate.to_raw();
        if value != OPUS_AUTO && value != OPUS_BITRATE_MAX {
            if value <= 0 {
                return Err(Error::BadArg);
            }
            value = imin(
                750000 * self.layout.nb_channels,
                imax(500 * self.layout.nb_channels, value),
            );
        }
        self.bitrate_bps = value;
        Ok(())
    }

    /// `OPUS_GET_BITRATE`: the sum of the stream bitrates.
    #[must_use]
    pub fn bitrate(&self) -> i32 {
        self.encoders.iter().map(Encoder::bitrate).sum()
    }

    /// `OPUS_SET_LSB_DEPTH` on every stream.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `8..=24`.
    pub fn set_lsb_depth(&mut self, depth: i32) -> Result<()> {
        self.for_each_stream(|e| e.set_lsb_depth(depth))
    }

    /// `OPUS_GET_LSB_DEPTH` (first stream).
    #[must_use]
    pub fn lsb_depth(&self) -> i32 {
        self.encoders[0].lsb_depth()
    }

    /// `OPUS_SET_COMPLEXITY` on every stream.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `0..=10`.
    pub fn set_complexity(&mut self, complexity: i32) -> Result<()> {
        self.for_each_stream(|e| e.set_complexity(complexity))
    }

    /// `OPUS_GET_COMPLEXITY` (first stream).
    #[must_use]
    pub fn complexity(&self) -> i32 {
        self.encoders[0].complexity()
    }

    /// `OPUS_SET_VBR` on every stream.
    pub fn set_vbr(&mut self, enabled: bool) {
        for e in &mut self.encoders {
            e.set_vbr(enabled);
        }
    }

    /// `OPUS_GET_VBR` (first stream).
    #[must_use]
    pub fn vbr(&self) -> bool {
        self.encoders[0].vbr()
    }

    /// `OPUS_SET_VBR_CONSTRAINT` on every stream.
    pub fn set_vbr_constraint(&mut self, enabled: bool) {
        for e in &mut self.encoders {
            e.set_vbr_constraint(enabled);
        }
    }

    /// `OPUS_GET_VBR_CONSTRAINT` (first stream).
    #[must_use]
    pub fn vbr_constraint(&self) -> bool {
        self.encoders[0].vbr_constraint()
    }

    /// `OPUS_SET_MAX_BANDWIDTH` on every stream.
    pub fn set_max_bandwidth(&mut self, bandwidth: Bandwidth) {
        for e in &mut self.encoders {
            e.set_max_bandwidth(bandwidth);
        }
    }

    /// `OPUS_SET_BANDWIDTH` on every stream (`None` = automatic).
    pub fn set_bandwidth(&mut self, bandwidth: Option<Bandwidth>) {
        for e in &mut self.encoders {
            e.set_bandwidth(bandwidth);
        }
    }

    /// `OPUS_GET_BANDWIDTH` (first stream).
    #[must_use]
    pub fn bandwidth(&self) -> Bandwidth {
        self.encoders[0].bandwidth()
    }

    /// `OPUS_SET_SIGNAL` on every stream.
    pub fn set_signal(&mut self, signal: Signal) {
        for e in &mut self.encoders {
            e.set_signal(signal);
        }
    }

    /// `OPUS_GET_SIGNAL` (first stream).
    #[must_use]
    pub fn signal(&self) -> Signal {
        self.encoders[0].signal()
    }

    /// `OPUS_SET_APPLICATION` on every stream.
    ///
    /// # Errors
    /// As [`Encoder::set_application`].
    pub fn set_application(&mut self, application: Application) -> Result<()> {
        self.for_each_stream(|e| e.set_application(application))
    }

    /// `OPUS_GET_APPLICATION` (first stream).
    #[must_use]
    pub fn application(&self) -> Application {
        self.encoders[0].application()
    }

    /// `OPUS_SET_INBAND_FEC` on every stream.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `0..=2`.
    pub fn set_inband_fec(&mut self, value: i32) -> Result<()> {
        self.for_each_stream(|e| e.set_inband_fec(value))
    }

    /// `OPUS_GET_INBAND_FEC` (first stream).
    #[must_use]
    pub fn inband_fec(&self) -> i32 {
        self.encoders[0].inband_fec()
    }

    /// `OPUS_SET_PACKET_LOSS_PERC` on every stream.
    ///
    /// # Errors
    /// [`Error::BadArg`] outside `0..=100`.
    pub fn set_packet_loss_perc(&mut self, percentage: i32) -> Result<()> {
        self.for_each_stream(|e| e.set_packet_loss_perc(percentage))
    }

    /// `OPUS_GET_PACKET_LOSS_PERC` (first stream).
    #[must_use]
    pub fn packet_loss_perc(&self) -> i32 {
        self.encoders[0].packet_loss_perc()
    }

    /// `OPUS_SET_DTX` on every stream.
    pub fn set_dtx(&mut self, enabled: bool) {
        for e in &mut self.encoders {
            e.set_dtx(enabled);
        }
    }

    /// `OPUS_GET_DTX` (first stream).
    #[must_use]
    pub fn dtx(&self) -> bool {
        self.encoders[0].dtx()
    }

    /// `OPUS_SET_FORCE_MODE` (private) on every stream.
    ///
    /// # Errors
    /// As [`Encoder::set_force_mode`].
    #[doc(hidden)]
    pub fn set_force_mode(&mut self, mode: i32) -> Result<()> {
        self.for_each_stream(|e| e.set_force_mode(mode))
    }

    /// `OPUS_GET_VOICE_RATIO` (private, first stream).
    #[doc(hidden)]
    #[must_use]
    pub fn voice_ratio(&self) -> i32 {
        self.encoders[0].voice_ratio()
    }

    /// `OPUS_SET_FORCE_CHANNELS` on every stream (`None` = automatic).
    ///
    /// # Errors
    /// As [`Encoder::set_force_channels`] (e.g. forcing 2 channels on a mono stream).
    pub fn set_force_channels(&mut self, channels: Option<u8>) -> Result<()> {
        self.for_each_stream(|e| e.set_force_channels(channels))
    }

    /// `OPUS_GET_FORCE_CHANNELS` (first stream).
    #[must_use]
    pub fn force_channels(&self) -> Option<u8> {
        self.encoders[0].force_channels()
    }

    /// `OPUS_SET_PREDICTION_DISABLED` on every stream.
    pub fn set_prediction_disabled(&mut self, disabled: bool) {
        for e in &mut self.encoders {
            e.set_prediction_disabled(disabled);
        }
    }

    /// `OPUS_GET_PREDICTION_DISABLED` (first stream).
    #[must_use]
    pub fn prediction_disabled(&self) -> bool {
        self.encoders[0].prediction_disabled()
    }

    /// `OPUS_SET_PHASE_INVERSION_DISABLED` on every stream.
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        for e in &mut self.encoders {
            e.set_phase_inversion_disabled(disabled);
        }
    }

    /// `OPUS_GET_PHASE_INVERSION_DISABLED` (first stream).
    #[must_use]
    pub fn phase_inversion_disabled(&self) -> bool {
        self.encoders[0].phase_inversion_disabled()
    }

    /// `OPUS_SET_QEXT` on every stream.
    #[cfg(feature = "qext")]
    pub fn set_qext(&mut self, enabled: bool) {
        for e in &mut self.encoders {
            e.set_qext(enabled);
        }
    }

    /// `OPUS_GET_QEXT` (first stream).
    #[cfg(feature = "qext")]
    #[must_use]
    pub fn qext(&self) -> bool {
        self.encoders[0].qext()
    }

    /// `OPUS_GET_LOOKAHEAD` (first stream).
    #[must_use]
    pub fn lookahead(&self) -> usize {
        self.encoders[0].lookahead()
    }

    /// `OPUS_GET_SAMPLE_RATE` (first stream).
    #[must_use]
    pub fn sample_rate(&self) -> i32 {
        self.encoders[0].sample_rate()
    }

    /// `OPUS_GET_FINAL_RANGE`: the XOR of the final range of every stream.
    #[must_use]
    pub fn final_range(&self) -> u32 {
        self.encoders
            .iter()
            .fold(0u32, |acc, e| acc ^ e.final_range())
    }

    /// `OPUS_SET_EXPERT_FRAME_DURATION` (stored without validation, as in C; an invalid value
    /// makes encoding fail with [`Error::BadArg`]).
    pub const fn set_expert_frame_duration(&mut self, duration: FrameSize) {
        self.variable_duration = duration.to_raw();
    }

    /// `OPUS_GET_EXPERT_FRAME_DURATION`.
    ///
    /// # Errors
    /// [`Error::BadArg`] if an invalid raw value was set through [`MsEncoder::ctl_set`].
    pub const fn expert_frame_duration(&self) -> Result<FrameSize> {
        FrameSize::from_raw(self.variable_duration)
    }

    /// `OPUS_MULTISTREAM_GET_ENCODER_STATE`: the encoder of stream `stream_id`.
    ///
    /// # Errors
    /// [`Error::BadArg`] if `stream_id` is out of range.
    pub fn encoder_state(&mut self, stream_id: i32) -> Result<&mut Encoder> {
        if stream_id < 0 || stream_id >= self.layout.nb_streams {
            return Err(Error::BadArg);
        }
        Ok(&mut self.encoders[stream_id as usize])
    }

    /// `OPUS_RESET_STATE`: resets every stream encoder and the surround analysis memories.
    pub fn reset(&mut self) {
        if self.mapping_type == MappingType::Surround {
            self.preemph_mem.fill(0.0);
            self.window_mem.fill(0.0);
        }
        for e in &mut self.encoders {
            e.reset();
        }
    }

    /// Numeric `opus_multistream_encoder_ctl` for requests taking one `opus_int32` argument
    /// (and `OPUS_RESET_STATE`, whose `value` is ignored). Most requests are forwarded to every
    /// stream encoder, stopping at the first error.
    ///
    /// # Errors
    /// The request's error, or [`Error::Unimplemented`] for requests the multistream encoder
    /// does not handle.
    pub fn ctl_set(&mut self, request: i32, value: i32) -> Result<()> {
        match request {
            OPUS_SET_BITRATE_REQUEST => self.set_bitrate(Bitrate::from_raw(value)),
            OPUS_SET_LSB_DEPTH_REQUEST
            | OPUS_SET_COMPLEXITY_REQUEST
            | OPUS_SET_VBR_REQUEST
            | OPUS_SET_VBR_CONSTRAINT_REQUEST
            | OPUS_SET_MAX_BANDWIDTH_REQUEST
            | OPUS_SET_BANDWIDTH_REQUEST
            | OPUS_SET_SIGNAL_REQUEST
            | OPUS_SET_APPLICATION_REQUEST
            | OPUS_SET_INBAND_FEC_REQUEST
            | OPUS_SET_PACKET_LOSS_PERC_REQUEST
            | OPUS_SET_DTX_REQUEST
            | OPUS_SET_FORCE_MODE_REQUEST
            | OPUS_SET_FORCE_CHANNELS_REQUEST
            | OPUS_SET_PREDICTION_DISABLED_REQUEST
            | OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST
            | OPUS_SET_QEXT_REQUEST => self.for_each_stream(|e| e.ctl_set(request, value)),
            OPUS_SET_EXPERT_FRAME_DURATION_REQUEST => {
                self.variable_duration = value;
                Ok(())
            }
            OPUS_RESET_STATE => {
                self.reset();
                Ok(())
            }
            _ => Err(Error::Unimplemented),
        }
    }

    /// Numeric `opus_multistream_encoder_ctl` for requests writing one `opus_int32`
    /// (`OPUS_GET_FINAL_RANGE` returns the `u32` bits as `i32`). Most requests query the first
    /// stream encoder.
    ///
    /// # Errors
    /// The request's error, or [`Error::Unimplemented`] for requests the multistream encoder
    /// does not handle.
    pub fn ctl_get(&self, request: i32) -> Result<i32> {
        match request {
            OPUS_GET_BITRATE_REQUEST => Ok(self.bitrate()),
            OPUS_GET_LSB_DEPTH_REQUEST
            | OPUS_GET_VBR_REQUEST
            | OPUS_GET_APPLICATION_REQUEST
            | OPUS_GET_BANDWIDTH_REQUEST
            | OPUS_GET_COMPLEXITY_REQUEST
            | OPUS_GET_PACKET_LOSS_PERC_REQUEST
            | OPUS_GET_DTX_REQUEST
            | OPUS_GET_VOICE_RATIO_REQUEST
            | OPUS_GET_VBR_CONSTRAINT_REQUEST
            | OPUS_GET_SIGNAL_REQUEST
            | OPUS_GET_LOOKAHEAD_REQUEST
            | OPUS_GET_SAMPLE_RATE_REQUEST
            | OPUS_GET_INBAND_FEC_REQUEST
            | OPUS_GET_FORCE_CHANNELS_REQUEST
            | OPUS_GET_PREDICTION_DISABLED_REQUEST
            | OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST
            | OPUS_GET_QEXT_REQUEST => self.encoders[0].ctl_get(request),
            OPUS_GET_FINAL_RANGE_REQUEST => Ok(self.final_range() as i32),
            OPUS_GET_EXPERT_FRAME_DURATION_REQUEST => Ok(self.variable_duration),
            _ => Err(Error::Unimplemented),
        }
    }
}
