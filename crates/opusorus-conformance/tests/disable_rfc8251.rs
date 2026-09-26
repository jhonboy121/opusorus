//! Feature `disable-rfc8251` (libopus `--disable-rfc8251`, `DISABLE_UPDATE_DRAFT`) vs the oracle
//! built with the same define.
//!
//! * The decoder state after init: mono CELT decoders keep stereo phase inversion enabled
//!   (RFC 6716; RFC 8251 disables it for mono), through the Opus, multistream and CELT APIs.
//! * The folding changes of `celt/bands.c` (no `special_hybrid_folding`, the RFC 6716
//!   `lowband_offset` and `fold_end` rules) are covered by running the band, CELT and Opus
//!   encoder/decoder differential suites with `--features disable-rfc8251`
//!   (`just test-options`); `tests/vectors.rs` and `tests/opus_decoder.rs` then decode the
//!   original RFC 6716 test vectors (`testdata/vectors/rfc6716`, fetched by
//!   `scripts/fetch_vectors.sh`) bit-exactly with C and through the `run_vectors.sh`
//!   procedure (the RFC 8251 decoder does not pass them).
#![allow(clippy::unwrap_used, reason = "test code")]

use opusorus::{Decoder, MsDecoder};
use opusorus_oracle::api;
use opusorus_oracle::build_options as c;
use opusorus_oracle::sys;

const OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST: i32 = 4047;

#[test]
fn oracle_has_the_same_option() {
    assert_eq!(
        c::options().disable_update_draft,
        cfg!(feature = "disable-rfc8251")
    );
}

#[test]
fn initial_phase_inversion_matches_c() {
    for fs in [8000, 12000, 16000, 24000, 48000] {
        for ch in 1..=2 {
            let r = Decoder::new(fs, ch).unwrap();
            let mut cd = api::Decoder::new(fs, ch).unwrap();
            let c_disabled = cd
                .ctl_get(OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST)
                .unwrap();
            assert_eq!(
                i32::from(r.phase_inversion_disabled()),
                c_disabled,
                "{fs} Hz {ch} ch"
            );
            let expect_disabled = ch == 1 && !cfg!(feature = "disable-rfc8251");
            assert_eq!(
                r.phase_inversion_disabled(),
                expect_disabled,
                "{fs} Hz {ch} ch"
            );
        }
    }
    // A multistream decoder with a mono and a coupled stream.
    let mapping = [0u8, 1, 2];
    let mut r = MsDecoder::new(48000, 3, 2, 1, &mapping).unwrap();
    let mut cd = api::MsDecoder::new(48000, 3, 2, 1, &mapping).unwrap();
    assert_eq!(
        r.ctl_get(OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST)
            .unwrap(),
        cd.ctl_get(OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST)
            .unwrap()
    );
}

/// Mono streams through both decoders: with phase inversion enabled in the mono decoder
/// (RFC 6716), a stereo-coded CELT stream downmixed by it matches C's too.
#[test]
fn mono_decoder_of_stereo_streams_matches_c() {
    let fs = 48000;
    let n = 960;
    let x = opusorus_conformance::signals::music_like(n * 50, 2, fs as u32, 11);
    let pcm = opusorus_conformance::signals::to_i16(&x);
    let mut enc = api::Encoder::new(fs, 2, sys::OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
    enc.ctl_set(sys::OPUS_SET_BITRATE_REQUEST, 64000).unwrap();
    let mut rd = Decoder::new(fs, 1).unwrap();
    let mut cd = api::Decoder::new(fs, 1).unwrap();
    let mut buf = vec![0u8; 1500];
    let mut ro = vec![0i16; n];
    let mut co = vec![0i16; n];
    for f in 0..50 {
        let l = enc
            .encode(&pcm[f * 2 * n..(f + 1) * 2 * n], n, &mut buf)
            .unwrap();
        let a = rd.decode(Some(&buf[..l]), &mut ro, n, false).unwrap();
        let b = cd.decode(Some(&buf[..l]), &mut co, n, false).unwrap();
        assert_eq!(a, b);
        assert_eq!(ro, co, "frame {f}");
        assert_eq!(rd.final_range(), cd.final_range().unwrap(), "frame {f}");
    }
}
