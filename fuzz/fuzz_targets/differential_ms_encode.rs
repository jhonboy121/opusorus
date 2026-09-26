//! Differential multistream / projection encoding (beyond the requested target list): the
//! surround encoder (mapping families 0, 1, 2 and 255, plus invalid families) and the
//! ambisonics projection encoder (family 3) against C libopus. Creation errors, layouts
//! (streams, coupled streams, mapping, demixing matrix), every return code, packet byte, final
//! range and GET CTL must be identical while CTLs change and arbitrary PCM is encoded.
//!
//! Input: `fs_sel:u8 channels:u8 family:u8 app_sel:u8` followed by the operation stream of
//! `opusorus_fuzz::enc` (without its header). `channels` is taken modulo 40.

#![no_main]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "fuzz harness: a failed check must panic so libFuzzer records the input"
)]

use libfuzzer_sys::fuzz_target;
use opusorus::encoder::request::{
    OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST,
    OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST,
};
use opusorus::{MsEncoder, ProjectionEncoder};
use opusorus_fuzz::enc::{self, APPS, ENC_GETS, EncHeader, EncOp, Pcm, Synth};
use opusorus_fuzz::{Reader, assert_slice_eq, code, pick_fs};
use opusorus_oracle::api as capi;

/// Mapping families tried (254 is invalid).
const FAMILIES: [i32; 6] = [0, 1, 2, 3, 255, 254];

enum Pair {
    Ms(MsEncoder, capi::MsEncoder),
    Proj(ProjectionEncoder, capi::ProjectionEncoder),
}

impl Pair {
    fn encode(
        &mut self,
        pcm: &Pcm,
        n: usize,
        a: &mut [u8],
        b: &mut [u8],
    ) -> [Result<usize, i32>; 2] {
        match self {
            Self::Ms(r, c) => match pcm {
                Pcm::I16(x) => [code(r.encode(x, n, a)), c.encode(x, n, b)],
                Pcm::I24(x) => [code(r.encode24(x, n, a)), c.encode24(x, n, b)],
                Pcm::F32(x) => [code(r.encode_float(x, n, a)), c.encode_float(x, n, b)],
            },
            Self::Proj(r, c) => match pcm {
                Pcm::I16(x) => [code(r.encode(x, n, a)), c.encode(x, n, b)],
                // The C wrapper has no encode24: feed the same samples as float to both.
                Pcm::I24(x) => {
                    let f: Vec<f32> = x.iter().map(|&v| v as f32 / 8_388_608.0).collect();
                    [code(r.encode_float(&f, n, a)), c.encode_float(&f, n, b)]
                }
                Pcm::F32(x) => [code(r.encode_float(x, n, a)), c.encode_float(x, n, b)],
            },
        }
    }
    fn ctl_set(&mut self, req: i32, value: i32) -> [Result<(), i32>; 2] {
        match self {
            Self::Ms(r, c) => [code(r.ctl_set(req, value)), c.ctl_set(req, value)],
            Self::Proj(r, c) => [code(r.ctl_set(req, value)), c.ctl_set(req, value)],
        }
    }
    fn ctl_get(&mut self, req: i32) -> [Result<i32, i32>; 2] {
        match self {
            Self::Ms(r, c) => [code(r.ctl_get(req)), c.ctl_get(req)],
            Self::Proj(r, c) => [code(r.ctl_get(req)), c.ctl_get(req)],
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut r = Reader::new(data);
    let fs = pick_fs(r.u8());
    let channels = i32::from(r.u8() % 40);
    let family = FAMILIES[usize::from(r.u8()) % FAMILIES.len()];
    let app = APPS[usize::from(r.u8()) % APPS.len()];
    let what = format!("({fs}, {channels}, family {family}, app {app})");
    let mut pair = if family == 3 {
        let rr = code(ProjectionEncoder::new_ambisonics_raw(
            fs, channels, family, app,
        ));
        let cr = capi::ProjectionEncoder::new(fs, channels, family, app);
        match (rr, cr) {
            (Ok(a), Ok(mut b)) => {
                assert_eq!(
                    (a.streams(), a.coupled_streams()),
                    (b.streams, b.coupled_streams),
                    "{what}: layout"
                );
                assert_eq!(
                    a.demixing_matrix(),
                    b.demixing_matrix().expect("C matrix"),
                    "{what}"
                );
                Pair::Proj(a, b)
            }
            (Err(a), Err(b)) => {
                assert_eq!(a, b, "{what}: create error");
                return;
            }
            (a, b) => panic!("{what}: create mismatch {:?} vs {:?}", a.err(), b.err()),
        }
    } else {
        let rr = code(MsEncoder::new_surround_raw(fs, channels, family, app));
        let cr = capi::MsEncoder::new_surround(fs, channels, family, app);
        match (rr, cr) {
            (Ok(a), Ok(b)) => {
                assert_eq!(
                    (a.streams(), a.coupled_streams(), a.mapping()),
                    (b.streams, b.coupled_streams, b.mapping.as_slice()),
                    "{what}: layout"
                );
                Pair::Ms(a, b)
            }
            (Err(a), Err(b)) => {
                assert_eq!(a, b, "{what}: create error");
                return;
            }
            (a, b) => panic!("{what}: create mismatch {:?} vs {:?}", a.err(), b.err()),
        }
    };
    let h = EncHeader { fs, channels, app };
    let mut synth = Synth::new();
    let mut i = 0;
    while let Some(op) = enc::read_op(&mut r, &h, &mut synth) {
        let what = format!("{what} op {i}");
        match op {
            EncOp::Frame {
                pcm,
                frame_size,
                max_bytes,
            } => {
                // Multistream packets need room for every stream.
                let max_bytes = max_bytes * (1 + channels as usize / 4);
                let mut a = vec![0xA5u8; max_bytes];
                let mut b = a.clone();
                let [rr, cr] = pair.encode(&pcm, frame_size, &mut a, &mut b);
                assert_eq!(rr, cr, "{what}: encode(n {frame_size}, max {max_bytes})");
                if let Ok(n) = rr {
                    assert_slice_eq(&format!("{what}: packet"), &a[..n], &b[..n]);
                }
            }
            EncOp::Ctl { req, value } => {
                let [rr, cr] = pair.ctl_set(req, value);
                assert_eq!(rr, cr, "{what}: ctl_set({req}, {value})");
            }
        }
        let extra = [
            OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST,
            OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST,
        ];
        for req in ENC_GETS.into_iter().chain(extra) {
            let [rr, cr] = pair.ctl_get(req);
            assert_eq!(rr, cr, "{what}: ctl_get({req})");
        }
        i += 1;
    }
});
