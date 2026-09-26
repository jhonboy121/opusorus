//! Port of libopus `tests/opus_encode_regressions.c` (`regression_test`, run by
//! `test_opus_encode`): encoder inputs and settings that triggered bugs fixed in libopus.
//!
//! The C functions ignore the return values of the CTL calls; the port checks that each one
//! succeeds, except where C relies on an ignored failure (commented at the call).

// Float-only: not compiled in fixed-point builds until this unit is converted
// (docs/FIXED_POINT.md).
#![cfg(not(feature = "fixed-point"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failed call is a test failure"
)]

mod regressions_data;

use opusorus::{
    Application, Bandwidth, Bitrate, Encoder, Error, MsEncoder, ProjectionEncoder, Signal,
};
use regressions_data::*;

/// Expands a `(declared length, explicit values)` C array (zero-filling the rest).
fn arr<T: Copy + Default>((len, vals): (usize, &[T])) -> Vec<T> {
    let mut v = vec![T::default(); len];
    v[..vals.len()].copy_from_slice(vals);
    v
}

/// The 13-CTL settings block of the multistream regressions.
#[derive(Debug, Clone, Copy)]
struct MsSettings {
    signal: Signal,
    vbr: bool,
    vbr_constraint: bool,
    prediction_disabled: bool,
    phase_inversion_disabled: bool,
    dtx: bool,
    complexity: i32,
    max_bandwidth: Bandwidth,
    bandwidth: Option<Bandwidth>,
    lsb_depth: i32,
    inband_fec: i32,
    packet_loss_perc: i32,
    bitrate: Bitrate,
}

fn apply(enc: &mut MsEncoder, s: MsSettings) {
    enc.set_signal(s.signal);
    enc.set_vbr(s.vbr);
    enc.set_vbr_constraint(s.vbr_constraint);
    enc.set_prediction_disabled(s.prediction_disabled);
    enc.set_phase_inversion_disabled(s.phase_inversion_disabled);
    enc.set_dtx(s.dtx);
    enc.set_complexity(s.complexity).unwrap();
    enc.set_max_bandwidth(s.max_bandwidth);
    enc.set_bandwidth(s.bandwidth);
    enc.set_lsb_depth(s.lsb_depth).unwrap();
    enc.set_inband_fec(s.inband_fec).unwrap();
    enc.set_packet_loss_perc(s.packet_loss_perc).unwrap();
    enc.set_bitrate(s.bitrate).unwrap();
}

/// Port of `celt_ec_internal_error`.
#[test]
fn celt_ec_internal_error() {
    let mut data = [0u8; 2460];
    let mut enc = MsEncoder::new_surround(16000, 1, 1, Application::Voip).unwrap();
    assert_eq!((enc.streams(), enc.coupled_streams()), (1, 0));

    let music_280k = MsSettings {
        signal: Signal::Music,
        vbr: false,
        vbr_constraint: true,
        prediction_disabled: true,
        phase_inversion_disabled: false,
        dtx: true,
        complexity: 10,
        max_bandwidth: Bandwidth::Fullband,
        bandwidth: Some(Bandwidth::Fullband),
        lsb_depth: 18,
        inband_fec: 1,
        packet_loss_perc: 90,
        bitrate: Bitrate::Bits(280_130),
    };

    apply(
        &mut enc,
        MsSettings {
            vbr_constraint: false,
            dtx: false,
            complexity: 0,
            max_bandwidth: Bandwidth::Narrowband,
            bandwidth: None,
            lsb_depth: 8,
            inband_fec: 0,
            packet_loss_perc: 0,
            bitrate: Bitrate::Auto,
            ..music_280k
        },
    );
    let pcm = arr(CELT_EC_INTERNAL_ERROR_PCM_1);
    assert!(enc.encode(&pcm, 320, &mut data).unwrap() > 0);

    for pcm in [
        CELT_EC_INTERNAL_ERROR_PCM_2,
        CELT_EC_INTERNAL_ERROR_PCM_3,
        CELT_EC_INTERNAL_ERROR_PCM_4,
        CELT_EC_INTERNAL_ERROR_PCM_5,
    ] {
        apply(&mut enc, music_280k);
        let pcm = arr(pcm);
        assert!(enc.encode(&pcm, 160, &mut data).unwrap() > 0);
    }

    apply(
        &mut enc,
        MsSettings {
            signal: Signal::Voice,
            phase_inversion_disabled: true,
            complexity: 0,
            max_bandwidth: Bandwidth::Narrowband,
            bandwidth: None,
            lsb_depth: 12,
            inband_fec: 0,
            packet_loss_perc: 41,
            bitrate: Bitrate::Bits(21425),
            ..music_280k
        },
    );
    let pcm = arr(CELT_EC_INTERNAL_ERROR_PCM_6);
    assert!(enc.encode(&pcm, 40, &mut data).unwrap() > 0); // returns -3
}

/// Port of `mscbr_encode_fail10`.
#[test]
fn mscbr_encode_fail10() {
    let mut data = vec![0u8; 627_300];
    let mapping = arr(MSCBR_ENCODE_FAIL10_MAPPING);
    let mut enc =
        MsEncoder::new(8000, 255, 254, 1, &mapping, Application::RestrictedLowDelay).unwrap();
    enc.set_signal(Signal::Voice);
    enc.set_vbr(false);
    enc.set_vbr_constraint(false);
    enc.set_prediction_disabled(false);
    // C ignores the failure: the coupled stream accepts 2 channels, the first mono stream
    // rejects it.
    assert_eq!(enc.set_force_channels(Some(2)), Err(Error::BadArg));
    enc.set_phase_inversion_disabled(true);
    enc.set_dtx(true);
    enc.set_complexity(2).unwrap();
    enc.set_max_bandwidth(Bandwidth::Narrowband);
    enc.set_bandwidth(None);
    enc.set_lsb_depth(14).unwrap();
    enc.set_inband_fec(0).unwrap();
    enc.set_packet_loss_perc(57).unwrap();
    enc.set_bitrate(Bitrate::Bits(3_642_675)).unwrap();
    let pcm = arr(MSCBR_ENCODE_FAIL10_PCM);
    let len = data.len();
    assert!(enc.encode(&pcm, 20, &mut data[..len]).unwrap() > 0); // returns -1
}

/// Port of `mscbr_encode_fail`.
#[test]
fn mscbr_encode_fail() {
    let mut data = vec![0u8; 472_320];
    let mapping = arr(MSCBR_ENCODE_FAIL_MAPPING);
    let mut enc =
        MsEncoder::new(8000, 192, 189, 3, &mapping, Application::RestrictedLowDelay).unwrap();
    enc.set_signal(Signal::Music);
    enc.set_vbr(false);
    enc.set_vbr_constraint(false);
    enc.set_prediction_disabled(false);
    enc.set_force_channels(None).unwrap();
    enc.set_phase_inversion_disabled(false);
    enc.set_dtx(false);
    enc.set_complexity(0).unwrap();
    enc.set_max_bandwidth(Bandwidth::Mediumband);
    enc.set_bandwidth(None);
    enc.set_lsb_depth(8).unwrap();
    enc.set_inband_fec(0).unwrap();
    enc.set_packet_loss_perc(0).unwrap();
    enc.set_bitrate(Bitrate::Bits(15360)).unwrap();
    let pcm = arr(MSCBR_ENCODE_FAIL_PCM);
    assert!(enc.encode(&pcm, 20, &mut data).unwrap() > 0); // returns -1
}

/// Port of `surround_analysis_uninit`.
#[test]
fn surround_analysis_uninit() {
    let mut data = [0u8; 7380];
    let mut enc = MsEncoder::new_surround(24000, 3, 1, Application::Audio).unwrap();
    enc.set_signal(Signal::Voice);
    enc.set_vbr(true);
    enc.set_vbr_constraint(true);
    enc.set_prediction_disabled(false);
    enc.set_force_channels(None).unwrap();
    enc.set_phase_inversion_disabled(false);
    enc.set_dtx(false);
    enc.set_complexity(0).unwrap();
    enc.set_max_bandwidth(Bandwidth::Narrowband);
    enc.set_bandwidth(Some(Bandwidth::Narrowband));
    enc.set_lsb_depth(8).unwrap();
    enc.set_inband_fec(1).unwrap();
    enc.set_bitrate(Bitrate::Bits(84315)).unwrap();
    let pcm = arr(SURROUND_ANALYSIS_UNINIT_PCM_1);
    assert!(enc.encode(&pcm, 960, &mut data).unwrap() > 0);

    enc.set_signal(Signal::Music);
    enc.set_vbr(true);
    enc.set_vbr_constraint(false);
    enc.set_prediction_disabled(true);
    enc.set_force_channels(None).unwrap();
    enc.set_phase_inversion_disabled(true);
    enc.set_dtx(true);
    enc.set_complexity(6).unwrap();
    enc.set_max_bandwidth(Bandwidth::Narrowband);
    enc.set_bandwidth(None);
    enc.set_lsb_depth(9).unwrap();
    enc.set_inband_fec(1).unwrap();
    enc.set_packet_loss_perc(5).unwrap();
    enc.set_bitrate(Bitrate::Bits(775_410)).unwrap();
    let pcm = arr(SURROUND_ANALYSIS_UNINIT_PCM_2);
    // reads uninitialized data at src/opus_multistream_encoder.c:293
    assert!(enc.encode(&pcm, 1440, &mut data).unwrap() > 0);
}

/// Port of `ec_enc_shrink_assert`.
#[test]
fn ec_enc_shrink_assert() {
    let mut data = [0u8; 2000];
    let pcm1 = arr(EC_ENC_SHRINK_ASSERT_PCM1);
    let pcm2 = arr(EC_ENC_SHRINK_ASSERT_PCM2);
    let pcm3 = arr(EC_ENC_SHRINK_ASSERT_PCM3);

    let mut enc = Encoder::new(48000, 1, Application::Audio).unwrap();
    enc.set_complexity(10).unwrap();
    enc.set_packet_loss_perc(6).unwrap();
    enc.set_bitrate(Bitrate::Bits(6000)).unwrap();
    assert!(enc.encode(&pcm1, 960, &mut data).unwrap() > 0);

    enc.set_signal(Signal::Voice);
    enc.set_prediction_disabled(true);
    enc.set_bandwidth(Some(Bandwidth::Superwideband));
    enc.set_inband_fec(1).unwrap();
    enc.set_bitrate(Bitrate::Bits(15600)).unwrap();
    assert!(enc.encode(&pcm2, 2880, &mut data[..122]).unwrap() > 0);

    enc.set_signal(Signal::Music);
    enc.set_bitrate(Bitrate::Bits(27000)).unwrap();
    // assertion failure
    assert!(enc.encode(&pcm3, 2880, &mut data[..122]).unwrap() > 0);
}

/// Port of `ec_enc_shrink_assert2`.
#[test]
fn ec_enc_shrink_assert2() {
    let mut data = [0u8; 2000];
    let mut enc = Encoder::new(48000, 1, Application::Audio).unwrap();
    enc.set_complexity(6).unwrap();
    enc.set_signal(Signal::Voice);
    enc.set_bandwidth(Some(Bandwidth::Fullband));
    enc.set_packet_loss_perc(26).unwrap();
    enc.set_bitrate(Bitrate::Bits(27000)).unwrap();
    let pcm = arr(EC_ENC_SHRINK_ASSERT2_PCM_1);
    assert!(enc.encode(&pcm, 960, &mut data).unwrap() > 0);
    enc.set_signal(Signal::Music);
    let pcm = arr(EC_ENC_SHRINK_ASSERT2_PCM_2);
    assert!(enc.encode(&pcm, 480, &mut data[..19]).unwrap() > 0);
}

/// Port of `silk_gain_assert`.
#[test]
fn silk_gain_assert() {
    let mut data = [0u8; 1000];
    let pcm1 = arr(SILK_GAIN_ASSERT_PCM1);
    let pcm2 = arr(SILK_GAIN_ASSERT_PCM2);

    let mut enc = Encoder::new(8000, 1, Application::Audio).unwrap();
    enc.set_complexity(3).unwrap();
    enc.set_max_bandwidth(Bandwidth::Narrowband);
    enc.set_bitrate(Bitrate::Bits(6000)).unwrap();
    assert!(enc.encode(&pcm1, 160, &mut data).unwrap() > 0);

    enc.set_vbr(false);
    enc.set_complexity(0).unwrap();
    enc.set_max_bandwidth(Bandwidth::Mediumband);
    enc.set_bitrate(Bitrate::Bits(2867)).unwrap();
    assert!(enc.encode(&pcm2, 960, &mut data).unwrap() > 0);
}

/// Port of `analysis_overflow`.
#[test]
fn analysis_overflow() {
    let mut data = [0u8; 200];
    let pcm = arr(ANALYSIS_OVERFLOW_PCM);
    let mut enc = Encoder::new(16000, 2, Application::Audio).unwrap();
    enc.set_complexity(10).unwrap();
    let len = enc.encode_float(&pcm, 320, &mut data).unwrap();
    assert!(len > 0 && len <= 200);
}

/// Port of `projection_overflow2`.
#[test]
fn projection_overflow2() {
    let mut data = [0u8; 480];
    let pcm = arr(PROJECTION_OVERFLOW2_PCM);
    let mut enc =
        ProjectionEncoder::new_ambisonics(12000, 9, 3, Application::RestrictedLowDelay).unwrap();
    let len = enc.encode_float(&pcm, 30, &mut data).unwrap();
    assert!(len > 0 && len <= 480);
}

/// Port of `projection_overflow3`.
#[test]
fn projection_overflow3() {
    let mut data = [0u8; 500];
    let mut enc = ProjectionEncoder::new_ambisonics(24000, 4, 3, Application::Audio).unwrap();

    let pcm = [-1e38f32; 60 * 4];
    let len = enc.encode_float(&pcm, 60, &mut data).unwrap();
    assert!((5..=500).contains(&len));

    let pcm = [1e38f32; 60 * 4];
    let len = enc.encode_float(&pcm, 60, &mut data).unwrap();
    assert!((5..=500).contains(&len));
}

/// Port of `projection_overflow`.
#[test]
fn projection_overflow() {
    let r = ProjectionEncoder::new_ambisonics(96000, 36, 3, Application::Audio);
    #[cfg(not(feature = "qext"))]
    assert_eq!(r.err(), Some(Error::BadArg));
    #[cfg(feature = "qext")]
    {
        let mut data = vec![0u8; 10000];
        let pcm = arr(PROJECTION_OVERFLOW_PCM);
        let mut enc = r.unwrap();
        enc.ms_mut().set_qext(true);
        let len = enc.encode(&pcm, 1920, &mut data).unwrap();
        assert!(len > 0 && len <= 10000);
    }
}

/// Port of `projection_overflow4`.
#[cfg(feature = "qext")]
#[test]
fn projection_overflow4() {
    let mut data = [0u8; 1000];
    let pcm = arr(PROJECTION_OVERFLOW4_PCM);
    let mut enc = ProjectionEncoder::new_ambisonics(96000, 36, 3, Application::Audio).unwrap();
    enc.ms_mut().set_qext(true);
    enc.ms_mut().set_bitrate(Bitrate::Bits(256_000)).unwrap();
    let len = enc.encode(&pcm, 480, &mut data).unwrap();
    assert!(len > 0 && len <= 1000);
}

/// Port of `qext_repacketize_fail`.
#[cfg(feature = "qext")]
#[test]
fn qext_repacketize_fail() {
    let mut data = vec![0u8; 9000];
    let pcm = arr(QEXT_REPACKETIZE_FAIL_PCM);
    let mut enc = Encoder::new(16000, 1, Application::Voip).unwrap();
    enc.set_vbr(false);
    enc.set_qext(true);
    enc.set_bitrate(Bitrate::Max).unwrap();
    // returns OPUS_INTERNAL_ERROR (-3)
    let len = enc.encode(&pcm, 960, &mut data).unwrap();
    assert!(len > 0 && len <= 9000);
}

/// Port of `qext_stereo_overflow`.
#[cfg(feature = "qext")]
#[test]
fn qext_stereo_overflow() {
    let mut data = [0u8; 2000];
    let pcm = vec![32767i16; 11520 * 2];
    let mut enc = Encoder::new(96000, 2, Application::RestrictedLowDelay).unwrap();
    let len = enc.encode(&pcm, 11520, &mut data).unwrap();
    assert!(len > 0 && len <= 2000);
}

/// Port of `qext_dred_combination`.
#[cfg(all(feature = "qext", feature = "dred"))]
#[test]
fn qext_dred_combination() {
    let mut data = [0u8; 2560];
    let pcm = arr(QEXT_DRED_COMBINATION_PCM);
    let mut enc = MsEncoder::new_surround(16000, 5, 1, Application::Voip).unwrap();
    enc.set_complexity(3).unwrap();
    enc.set_packet_loss_perc(12).unwrap();
    enc.set_qext(true);
    enc.encoder_state(0)
        .unwrap()
        .set_dred_duration(103)
        .unwrap();
    enc.set_bitrate(Bitrate::Bits(755_850)).unwrap();
    // returns OPUS_BAD_ARG (-1)
    let len = enc.encode(&pcm, 320, &mut data).unwrap();
    assert!(len > 0 && len <= 2560);
}
