//! Raw FFI declarations for the public libopus v1.6.1 API (`opus.h`, `opus_multistream.h`,
//! `opus_projection.h`). Prefer the safe wrappers in [`crate::api`].

#![allow(
    missing_docs,
    non_camel_case_types,
    reason = "direct mirror of the C header names"
)]

use core::ffi::{c_char, c_int, c_uchar};

/// Opaque C encoder state.
#[repr(C)]
#[derive(Debug)]
pub struct OpusEncoder {
    _private: [u8; 0],
}
/// Opaque C decoder state.
#[repr(C)]
#[derive(Debug)]
pub struct OpusDecoder {
    _private: [u8; 0],
}
/// Opaque C multistream encoder state.
#[repr(C)]
#[derive(Debug)]
pub struct OpusMSEncoder {
    _private: [u8; 0],
}
/// Opaque C multistream decoder state.
#[repr(C)]
#[derive(Debug)]
pub struct OpusMSDecoder {
    _private: [u8; 0],
}
/// Opaque C projection encoder state.
#[repr(C)]
#[derive(Debug)]
pub struct OpusProjectionEncoder {
    _private: [u8; 0],
}
/// Opaque C projection decoder state.
#[repr(C)]
#[derive(Debug)]
pub struct OpusProjectionDecoder {
    _private: [u8; 0],
}
/// Opaque C repacketizer state.
#[repr(C)]
#[derive(Debug)]
pub struct OpusRepacketizer {
    _private: [u8; 0],
}

pub type opus_int32 = i32;
pub type opus_int16 = i16;

unsafe extern "C" {
    pub fn opus_get_version_string() -> *const c_char;
    pub fn opus_strerror(error: c_int) -> *const c_char;

    // --- Encoder ---
    pub fn opus_encoder_get_size(channels: c_int) -> c_int;
    pub fn opus_encoder_create(
        fs: opus_int32,
        channels: c_int,
        application: c_int,
        error: *mut c_int,
    ) -> *mut OpusEncoder;
    pub fn opus_encoder_init(
        st: *mut OpusEncoder,
        fs: opus_int32,
        channels: c_int,
        application: c_int,
    ) -> c_int;
    pub fn opus_encode(
        st: *mut OpusEncoder,
        pcm: *const opus_int16,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: opus_int32,
    ) -> opus_int32;
    pub fn opus_encode24(
        st: *mut OpusEncoder,
        pcm: *const opus_int32,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: opus_int32,
    ) -> opus_int32;
    pub fn opus_encode_float(
        st: *mut OpusEncoder,
        pcm: *const f32,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: opus_int32,
    ) -> opus_int32;
    pub fn opus_encoder_destroy(st: *mut OpusEncoder);
    pub fn opus_encoder_ctl(st: *mut OpusEncoder, request: c_int, ...) -> c_int;

    // --- Decoder ---
    pub fn opus_decoder_get_size(channels: c_int) -> c_int;
    pub fn opus_decoder_create(
        fs: opus_int32,
        channels: c_int,
        error: *mut c_int,
    ) -> *mut OpusDecoder;
    pub fn opus_decoder_init(st: *mut OpusDecoder, fs: opus_int32, channels: c_int) -> c_int;
    pub fn opus_decode(
        st: *mut OpusDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut opus_int16,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_decode24(
        st: *mut OpusDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut opus_int32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_decode_float(
        st: *mut OpusDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut f32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_decoder_ctl(st: *mut OpusDecoder, request: c_int, ...) -> c_int;
    pub fn opus_decoder_destroy(st: *mut OpusDecoder);
    pub fn opus_decoder_get_nb_samples(
        dec: *const OpusDecoder,
        packet: *const c_uchar,
        len: opus_int32,
    ) -> c_int;

    // --- Packet helpers ---
    pub fn opus_packet_parse(
        data: *const c_uchar,
        len: opus_int32,
        out_toc: *mut c_uchar,
        frames: *mut *const c_uchar,
        size: *mut opus_int16,
        payload_offset: *mut c_int,
    ) -> c_int;
    pub fn opus_packet_get_bandwidth(data: *const c_uchar) -> c_int;
    pub fn opus_packet_get_samples_per_frame(data: *const c_uchar, fs: opus_int32) -> c_int;
    pub fn opus_packet_get_nb_channels(data: *const c_uchar) -> c_int;
    pub fn opus_packet_get_nb_frames(packet: *const c_uchar, len: opus_int32) -> c_int;
    pub fn opus_packet_get_nb_samples(
        packet: *const c_uchar,
        len: opus_int32,
        fs: opus_int32,
    ) -> c_int;
    pub fn opus_packet_has_lbrr(packet: *const c_uchar, len: opus_int32) -> c_int;
    pub fn opus_pcm_soft_clip(
        pcm: *mut f32,
        frame_size: c_int,
        channels: c_int,
        softclip_mem: *mut f32,
    );
    pub fn opus_packet_pad(data: *mut c_uchar, len: opus_int32, new_len: opus_int32) -> c_int;
    pub fn opus_packet_unpad(data: *mut c_uchar, len: opus_int32) -> opus_int32;
    pub fn opus_multistream_packet_pad(
        data: *mut c_uchar,
        len: opus_int32,
        new_len: opus_int32,
        nb_streams: c_int,
    ) -> c_int;
    pub fn opus_multistream_packet_unpad(
        data: *mut c_uchar,
        len: opus_int32,
        nb_streams: c_int,
    ) -> opus_int32;

    // --- Repacketizer ---
    pub fn opus_repacketizer_get_size() -> c_int;
    pub fn opus_repacketizer_init(rp: *mut OpusRepacketizer) -> *mut OpusRepacketizer;
    pub fn opus_repacketizer_create() -> *mut OpusRepacketizer;
    pub fn opus_repacketizer_destroy(rp: *mut OpusRepacketizer);
    pub fn opus_repacketizer_cat(
        rp: *mut OpusRepacketizer,
        data: *const c_uchar,
        len: opus_int32,
    ) -> c_int;
    pub fn opus_repacketizer_out_range(
        rp: *mut OpusRepacketizer,
        begin: c_int,
        end: c_int,
        data: *mut c_uchar,
        maxlen: opus_int32,
    ) -> opus_int32;
    pub fn opus_repacketizer_get_nb_frames(rp: *mut OpusRepacketizer) -> c_int;
    pub fn opus_repacketizer_out(
        rp: *mut OpusRepacketizer,
        data: *mut c_uchar,
        maxlen: opus_int32,
    ) -> opus_int32;

    // --- Multistream ---
    pub fn opus_multistream_encoder_get_size(streams: c_int, coupled_streams: c_int) -> opus_int32;
    pub fn opus_multistream_surround_encoder_get_size(
        channels: c_int,
        mapping_family: c_int,
    ) -> opus_int32;
    pub fn opus_multistream_encoder_create(
        fs: opus_int32,
        channels: c_int,
        streams: c_int,
        coupled_streams: c_int,
        mapping: *const c_uchar,
        application: c_int,
        error: *mut c_int,
    ) -> *mut OpusMSEncoder;
    pub fn opus_multistream_surround_encoder_create(
        fs: opus_int32,
        channels: c_int,
        mapping_family: c_int,
        streams: *mut c_int,
        coupled_streams: *mut c_int,
        mapping: *mut c_uchar,
        application: c_int,
        error: *mut c_int,
    ) -> *mut OpusMSEncoder;
    pub fn opus_multistream_encode(
        st: *mut OpusMSEncoder,
        pcm: *const opus_int16,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: opus_int32,
    ) -> c_int;
    pub fn opus_multistream_encode24(
        st: *mut OpusMSEncoder,
        pcm: *const opus_int32,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: opus_int32,
    ) -> c_int;
    pub fn opus_multistream_encode_float(
        st: *mut OpusMSEncoder,
        pcm: *const f32,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: opus_int32,
    ) -> c_int;
    pub fn opus_multistream_encoder_destroy(st: *mut OpusMSEncoder);
    pub fn opus_multistream_encoder_ctl(st: *mut OpusMSEncoder, request: c_int, ...) -> c_int;
    pub fn opus_multistream_decoder_get_size(streams: c_int, coupled_streams: c_int) -> opus_int32;
    pub fn opus_multistream_decoder_create(
        fs: opus_int32,
        channels: c_int,
        streams: c_int,
        coupled_streams: c_int,
        mapping: *const c_uchar,
        error: *mut c_int,
    ) -> *mut OpusMSDecoder;
    pub fn opus_multistream_decode(
        st: *mut OpusMSDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut opus_int16,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_multistream_decode24(
        st: *mut OpusMSDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut opus_int32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_multistream_decode_float(
        st: *mut OpusMSDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut f32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_multistream_decoder_ctl(st: *mut OpusMSDecoder, request: c_int, ...) -> c_int;
    pub fn opus_multistream_decoder_destroy(st: *mut OpusMSDecoder);

    // --- Projection (ambisonics) ---
    pub fn opus_projection_ambisonics_encoder_get_size(
        channels: c_int,
        mapping_family: c_int,
    ) -> opus_int32;
    pub fn opus_projection_ambisonics_encoder_create(
        fs: opus_int32,
        channels: c_int,
        mapping_family: c_int,
        streams: *mut c_int,
        coupled_streams: *mut c_int,
        application: c_int,
        error: *mut c_int,
    ) -> *mut OpusProjectionEncoder;
    pub fn opus_projection_encode(
        st: *mut OpusProjectionEncoder,
        pcm: *const opus_int16,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: opus_int32,
    ) -> c_int;
    pub fn opus_projection_encode24(
        st: *mut OpusProjectionEncoder,
        pcm: *const opus_int32,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: opus_int32,
    ) -> c_int;
    pub fn opus_projection_encode_float(
        st: *mut OpusProjectionEncoder,
        pcm: *const f32,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: opus_int32,
    ) -> c_int;
    pub fn opus_projection_encoder_destroy(st: *mut OpusProjectionEncoder);
    pub fn opus_projection_encoder_ctl(
        st: *mut OpusProjectionEncoder,
        request: c_int,
        ...
    ) -> c_int;
    pub fn opus_projection_decoder_get_size(
        channels: c_int,
        streams: c_int,
        coupled_streams: c_int,
    ) -> opus_int32;
    pub fn opus_projection_decoder_create(
        fs: opus_int32,
        channels: c_int,
        streams: c_int,
        coupled_streams: c_int,
        demixing_matrix: *mut c_uchar,
        demixing_matrix_size: opus_int32,
        error: *mut c_int,
    ) -> *mut OpusProjectionDecoder;
    pub fn opus_projection_decode(
        st: *mut OpusProjectionDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut opus_int16,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_projection_decode24(
        st: *mut OpusProjectionDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut opus_int32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_projection_decode_float(
        st: *mut OpusProjectionDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut f32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_projection_decoder_ctl(
        st: *mut OpusProjectionDecoder,
        request: c_int,
        ...
    ) -> c_int;
    pub fn opus_projection_decoder_destroy(st: *mut OpusProjectionDecoder);
}

// Error codes.
pub const OPUS_OK: c_int = 0;
pub const OPUS_BAD_ARG: c_int = -1;
pub const OPUS_BUFFER_TOO_SMALL: c_int = -2;
pub const OPUS_INTERNAL_ERROR: c_int = -3;
pub const OPUS_INVALID_PACKET: c_int = -4;
pub const OPUS_UNIMPLEMENTED: c_int = -5;
pub const OPUS_INVALID_STATE: c_int = -6;
pub const OPUS_ALLOC_FAIL: c_int = -7;

// CTL requests.
pub const OPUS_SET_APPLICATION_REQUEST: c_int = 4000;
pub const OPUS_GET_APPLICATION_REQUEST: c_int = 4001;
pub const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
pub const OPUS_GET_BITRATE_REQUEST: c_int = 4003;
pub const OPUS_SET_MAX_BANDWIDTH_REQUEST: c_int = 4004;
pub const OPUS_GET_MAX_BANDWIDTH_REQUEST: c_int = 4005;
pub const OPUS_SET_VBR_REQUEST: c_int = 4006;
pub const OPUS_GET_VBR_REQUEST: c_int = 4007;
pub const OPUS_SET_BANDWIDTH_REQUEST: c_int = 4008;
pub const OPUS_GET_BANDWIDTH_REQUEST: c_int = 4009;
pub const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;
pub const OPUS_GET_COMPLEXITY_REQUEST: c_int = 4011;
pub const OPUS_SET_INBAND_FEC_REQUEST: c_int = 4012;
pub const OPUS_GET_INBAND_FEC_REQUEST: c_int = 4013;
pub const OPUS_SET_PACKET_LOSS_PERC_REQUEST: c_int = 4014;
pub const OPUS_GET_PACKET_LOSS_PERC_REQUEST: c_int = 4015;
pub const OPUS_SET_DTX_REQUEST: c_int = 4016;
pub const OPUS_GET_DTX_REQUEST: c_int = 4017;
pub const OPUS_SET_VBR_CONSTRAINT_REQUEST: c_int = 4020;
pub const OPUS_GET_VBR_CONSTRAINT_REQUEST: c_int = 4021;
pub const OPUS_SET_FORCE_CHANNELS_REQUEST: c_int = 4022;
pub const OPUS_GET_FORCE_CHANNELS_REQUEST: c_int = 4023;
pub const OPUS_SET_SIGNAL_REQUEST: c_int = 4024;
pub const OPUS_GET_SIGNAL_REQUEST: c_int = 4025;
pub const OPUS_GET_LOOKAHEAD_REQUEST: c_int = 4027;
pub const OPUS_RESET_STATE: c_int = 4028;
pub const OPUS_GET_SAMPLE_RATE_REQUEST: c_int = 4029;
pub const OPUS_GET_FINAL_RANGE_REQUEST: c_int = 4031;
pub const OPUS_GET_PITCH_REQUEST: c_int = 4033;
pub const OPUS_SET_GAIN_REQUEST: c_int = 4034;
pub const OPUS_GET_GAIN_REQUEST: c_int = 4045;
pub const OPUS_SET_LSB_DEPTH_REQUEST: c_int = 4036;
pub const OPUS_GET_LSB_DEPTH_REQUEST: c_int = 4037;
pub const OPUS_GET_LAST_PACKET_DURATION_REQUEST: c_int = 4039;
pub const OPUS_SET_EXPERT_FRAME_DURATION_REQUEST: c_int = 4040;
pub const OPUS_GET_EXPERT_FRAME_DURATION_REQUEST: c_int = 4041;
pub const OPUS_SET_PREDICTION_DISABLED_REQUEST: c_int = 4042;
pub const OPUS_GET_PREDICTION_DISABLED_REQUEST: c_int = 4043;
pub const OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST: c_int = 4046;
pub const OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST: c_int = 4047;
pub const OPUS_GET_IN_DTX_REQUEST: c_int = 4049;
pub const OPUS_SET_DRED_DURATION_REQUEST: c_int = 4050;
pub const OPUS_GET_DRED_DURATION_REQUEST: c_int = 4051;
pub const OPUS_SET_DNN_BLOB_REQUEST: c_int = 4052;
pub const OPUS_SET_OSCE_BWE_REQUEST: c_int = 4054;
pub const OPUS_GET_OSCE_BWE_REQUEST: c_int = 4055;
pub const OPUS_SET_QEXT_REQUEST: c_int = 4056;
pub const OPUS_GET_QEXT_REQUEST: c_int = 4057;
pub const OPUS_SET_IGNORE_EXTENSIONS_REQUEST: c_int = 4058;
pub const OPUS_GET_IGNORE_EXTENSIONS_REQUEST: c_int = 4059;
pub const OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST: c_int = 5120;
pub const OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST: c_int = 5122;
pub const OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST: c_int = 6001;
pub const OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST: c_int = 6003;
pub const OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST: c_int = 6005;

// Values.
pub const OPUS_AUTO: c_int = -1000;
pub const OPUS_BITRATE_MAX: c_int = -1;
pub const OPUS_APPLICATION_VOIP: c_int = 2048;
pub const OPUS_APPLICATION_AUDIO: c_int = 2049;
pub const OPUS_APPLICATION_RESTRICTED_LOWDELAY: c_int = 2051;
pub const OPUS_APPLICATION_RESTRICTED_SILK: c_int = 2052;
pub const OPUS_APPLICATION_RESTRICTED_CELT: c_int = 2053;
pub const OPUS_SIGNAL_VOICE: c_int = 3001;
pub const OPUS_SIGNAL_MUSIC: c_int = 3002;
pub const OPUS_BANDWIDTH_NARROWBAND: c_int = 1101;
pub const OPUS_BANDWIDTH_MEDIUMBAND: c_int = 1102;
pub const OPUS_BANDWIDTH_WIDEBAND: c_int = 1103;
pub const OPUS_BANDWIDTH_SUPERWIDEBAND: c_int = 1104;
pub const OPUS_BANDWIDTH_FULLBAND: c_int = 1105;
pub const OPUS_FRAMESIZE_ARG: c_int = 5000;
pub const OPUS_FRAMESIZE_2_5_MS: c_int = 5001;
pub const OPUS_FRAMESIZE_5_MS: c_int = 5002;
pub const OPUS_FRAMESIZE_10_MS: c_int = 5003;
pub const OPUS_FRAMESIZE_20_MS: c_int = 5004;
pub const OPUS_FRAMESIZE_40_MS: c_int = 5005;
pub const OPUS_FRAMESIZE_60_MS: c_int = 5006;
pub const OPUS_FRAMESIZE_80_MS: c_int = 5007;
pub const OPUS_FRAMESIZE_100_MS: c_int = 5008;
pub const OPUS_FRAMESIZE_120_MS: c_int = 5009;
