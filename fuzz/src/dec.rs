//! Decoder operation stream and decoder adapters.
//!
//! Input format (after a target-specific header), repeated until the input ends:
//!
//! ```text
//! tag:u8   tag % 16:
//!   0..=8, 15  decode    flags:u8 fsel:u8 len:u16 packet[len]
//!   9, 10      lost      flags:u8 fsel:u8                 (data = NULL: PLC)
//!   11         ctl       sel:u8 vsel:u8 [value:i32 if vsel >= 128]
//!   12         nb_samples len:u16 packet[len]
//!   13         reset
//!   14         empty     flags:u8 fsel:u8                 (data = non-NULL, len 0)
//! flags: bits 0-1 output format (0 i16, 1 i24, 2/3 float), bits 2-3 FEC (0/2 off, 1 on,
//!        3 the invalid value 2), bits 4-5 output-buffer mode (non-differential targets only).
//! fsel:  see [`frame_size`].
//! ```

use crate::{Reader, Writer, assert_f32_bits_eq, assert_slice_eq, code};
use opusorus::decoder::{
    OPUS_GET_BANDWIDTH_REQUEST, OPUS_GET_COMPLEXITY_REQUEST, OPUS_GET_FINAL_RANGE_REQUEST,
    OPUS_GET_GAIN_REQUEST, OPUS_GET_IGNORE_EXTENSIONS_REQUEST,
    OPUS_GET_LAST_PACKET_DURATION_REQUEST, OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
    OPUS_GET_PITCH_REQUEST, OPUS_GET_SAMPLE_RATE_REQUEST, OPUS_RESET_STATE,
    OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_GAIN_REQUEST, OPUS_SET_IGNORE_EXTENSIONS_REQUEST,
    OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
};
use opusorus::{Decoder, MsDecoder, ProjectionDecoder, packet};
use opusorus_oracle::opus_decoder as od;

/// Largest frame size (per channel) the harness ever requests (64 x 2.5 ms at 96 kHz); bounds
/// allocations.
pub const MAX_FRAME: i32 = 15360;

/// Every GET request `opus_decoder_ctl` implements (default build).
pub const DEC_GETS: [i32; 9] = [
    OPUS_GET_BANDWIDTH_REQUEST,
    OPUS_GET_COMPLEXITY_REQUEST,
    OPUS_GET_FINAL_RANGE_REQUEST,
    OPUS_GET_SAMPLE_RATE_REQUEST,
    OPUS_GET_PITCH_REQUEST,
    OPUS_GET_GAIN_REQUEST,
    OPUS_GET_LAST_PACKET_DURATION_REQUEST,
    OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
    OPUS_GET_IGNORE_EXTENSIONS_REQUEST,
];

/// SET requests the harness issues, with the values seeds pick from (`vsel < 128`).
pub const DEC_SETS: [(i32, &[i32]); 6] = [
    (
        OPUS_SET_COMPLEXITY_REQUEST,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, 11],
    ),
    (
        OPUS_SET_GAIN_REQUEST,
        &[0, 256, -256, 1536, -6000, 32767, -32768, 32768, -32769],
    ),
    (OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, &[0, 1, 2]),
    (OPUS_SET_IGNORE_EXTENSIONS_REQUEST, &[0, 1, -1]),
    (OPUS_RESET_STATE, &[0]),
    // Not a decoder request: must be rejected identically (OPUS_UNIMPLEMENTED).
    (4002, &[64000]),
];

/// Tag values for the [`Writer`] helpers.
pub mod tag {
    /// Decode a packet.
    pub const DECODE: u8 = 0;
    /// Lost packet (PLC).
    pub const LOST: u8 = 9;
    /// A SET CTL.
    pub const CTL: u8 = 11;
    /// `get_nb_samples`.
    pub const NB_SAMPLES: u8 = 12;
    /// `OPUS_RESET_STATE`.
    pub const RESET: u8 = 13;
    /// Decode with an empty, non-NULL packet.
    pub const EMPTY: u8 = 14;
}

/// Output sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fmt {
    /// `opus_decode` (soft-clipped 16-bit).
    I16,
    /// `opus_decode24`.
    I24,
    /// `opus_decode_float`.
    F32,
}

/// One decoder operation.
#[derive(Debug, Clone, Copy)]
pub enum DecOp<'a> {
    /// A decode call (`data = None` is a lost packet).
    Decode {
        /// Packet (`None` = NULL pointer).
        data: Option<&'a [u8]>,
        /// Output format.
        fmt: Fmt,
        /// `decode_fec` argument (0, 1, or the invalid 2).
        fec: i32,
        /// Frame-size selector, see [`frame_size`].
        fsel: u8,
        /// Output-buffer mode (0..=3), used by the non-differential target.
        buf: u8,
    },
    /// A SET CTL (or `OPUS_RESET_STATE`).
    Ctl {
        /// Request.
        req: i32,
        /// Value.
        value: i32,
    },
    /// `opus_decoder_get_nb_samples` / `opus_packet_get_nb_samples`.
    NbSamples(&'a [u8]),
}

const fn flags(b: u8) -> (Fmt, i32, u8) {
    let fmt = match b & 3 {
        0 => Fmt::I16,
        1 => Fmt::I24,
        _ => Fmt::F32,
    };
    let fec = match (b >> 2) & 3 {
        1 => 1,
        3 => 2,
        _ => 0,
    };
    (fmt, fec, (b >> 4) & 3)
}

/// Encodes the flags byte.
#[must_use]
pub const fn make_flags(fmt: Fmt, fec: bool, buf: u8) -> u8 {
    let f = match fmt {
        Fmt::I16 => 0,
        Fmt::I24 => 1,
        Fmt::F32 => 2,
    };
    f | if fec { 4 } else { 0 } | ((buf & 3) << 4)
}

/// Reads a CTL selector + value (shared with the encoder stream).
pub fn read_ctl(r: &mut Reader<'_>, table: &[(i32, &[i32])]) -> (i32, i32) {
    let (req, vals) = table[r.u8() as usize % table.len()];
    let vsel = r.u8();
    let value = if vsel < 128 {
        vals[vsel as usize % vals.len()]
    } else {
        r.i32()
    };
    (req, value)
}

/// Writes a CTL selector picking `vals[vidx]` of `table[sel]`.
pub fn write_ctl(w: &mut Writer, sel: usize, vidx: usize) {
    w.u8(sel as u8).u8(vidx as u8);
}

/// Reads the next operation, or `None` at the end of the input.
pub fn read_op<'a>(r: &mut Reader<'a>) -> Option<DecOp<'a>> {
    if r.is_empty() {
        return None;
    }
    let t = r.u8() % 16;
    Some(match t {
        9 | 10 | 14 => {
            let (fmt, fec, buf) = flags(r.u8());
            let fsel = r.u8();
            DecOp::Decode {
                data: if t == 14 { Some(&[]) } else { None },
                fmt,
                fec,
                fsel,
                buf,
            }
        }
        11 => {
            let (req, value) = read_ctl(r, &DEC_SETS);
            DecOp::Ctl { req, value }
        }
        12 => DecOp::NbSamples(r.blob()),
        13 => DecOp::Ctl {
            req: OPUS_RESET_STATE,
            value: 0,
        },
        _ => {
            let (fmt, fec, buf) = flags(r.u8());
            let fsel = r.u8();
            DecOp::Decode {
                data: Some(r.blob()),
                fmt,
                fec,
                fsel,
                buf,
            }
        }
    })
}

/// Writes a decode op.
pub fn write_decode(w: &mut Writer, data: &[u8], fmt: Fmt, fec: bool, fsel: u8) {
    w.u8(tag::DECODE)
        .u8(make_flags(fmt, fec, 0))
        .u8(fsel)
        .blob(data);
}

/// Writes a lost-packet op.
pub fn write_lost(w: &mut Writer, fmt: Fmt, fsel: u8) {
    w.u8(tag::LOST).u8(make_flags(fmt, false, 0)).u8(fsel);
}

/// Frame-size selector: packet duration (the common case for real streams).
pub const FSEL_PACKET: u8 = 0x40;
/// Frame-size selector: 120 ms (the maximum).
pub const FSEL_MAX: u8 = 0x00;
/// Frame-size selector: `k` x 2.5 ms (`k` in 1..=64).
#[must_use]
pub const fn fsel_2_5ms(k: u8) -> u8 {
    0x80 | ((k - 1) & 63)
}

/// Odd frame sizes (selector `0xC0 | i`).
const RAW_FRAME: [i32; 24] = [
    0,
    1,
    -1,
    2,
    7,
    119,
    120,
    121,
    479,
    480,
    481,
    959,
    960,
    961,
    1919,
    1920,
    2880,
    5759,
    5760,
    5761,
    11520,
    12000,
    i32::MIN,
    -5760,
];

/// The `frame_size` argument for selector `fsel`:
/// * `0x00..=0x3F`: 120 ms (the maximum),
/// * `0x40..=0x7F`: the packet's duration (20 ms if unknown),
/// * `0x80..=0xBF`: `(fsel & 63) + 1` times 2.5 ms,
/// * `0xC0..=0xFF`: an odd raw value (including 0 and negatives).
#[must_use]
pub fn frame_size(fsel: u8, fs: i32, data: Option<&[u8]>) -> i32 {
    match fsel >> 6 {
        0 => fs / 25 * 3,
        1 => match data.map(|d| packet::get_nb_samples(d, fs)) {
            Some(Ok(n)) if n > 0 => n.min(MAX_FRAME),
            _ => fs / 50,
        },
        2 => (i32::from(fsel & 63) + 1) * (fs / 400),
        _ => RAW_FRAME[usize::from(fsel & 63) % RAW_FRAME.len()],
    }
}

/// A decoder state dump: the Opus-level ints and the soft-clip memory bits.
pub type StateDump = ([i32; 21], [u32; 2]);

fn rust_state(d: &Decoder) -> StateDump {
    let s = d.snapshot();
    (
        [
            s.channels,
            s.fs,
            s.dec_control.n_channels_api,
            s.dec_control.n_channels_internal,
            s.dec_control.api_sample_rate,
            s.dec_control.internal_sample_rate,
            s.dec_control.payload_size_ms,
            s.dec_control.prev_pitch_lag,
            s.dec_control.enable_deep_plc,
            s.decode_gain,
            s.complexity,
            s.ignore_extensions,
            s.stream_channels,
            s.bandwidth,
            s.mode,
            s.prev_mode,
            s.frame_size,
            s.prev_redundancy,
            s.last_packet_duration,
            s.range_final as i32,
            0,
        ],
        s.softclip_mem.map(f32::to_bits),
    )
}

fn c_state(s: &od::DecState) -> StateDump {
    (s.ints, s.softclip_mem.map(f32::to_bits))
}

/// The decode API shared by the single-stream, multistream and projection decoders, in both
/// implementations. Results use the C convention.
pub trait DecApi {
    /// Output channels.
    fn channels(&self) -> usize;
    /// `*_decode` (16-bit).
    fn dec_i16(
        &mut self,
        d: Option<&[u8]>,
        pcm: &mut [i16],
        n: i32,
        fec: i32,
    ) -> Result<usize, i32>;
    /// `*_decode24`.
    fn dec_i24(
        &mut self,
        d: Option<&[u8]>,
        pcm: &mut [i32],
        n: i32,
        fec: i32,
    ) -> Result<usize, i32>;
    /// `*_decode_float`.
    fn dec_f32(
        &mut self,
        d: Option<&[u8]>,
        pcm: &mut [f32],
        n: i32,
        fec: i32,
    ) -> Result<usize, i32>;
    /// A SET CTL.
    fn ctl_set(&mut self, req: i32, value: i32) -> Result<(), i32>;
    /// A GET CTL.
    fn ctl_get(&mut self, req: i32) -> Result<i32, i32>;
    /// `opus_decoder_get_nb_samples` (single-stream only).
    fn nb_samples(&mut self, _pkt: &[u8]) -> Option<Result<usize, i32>> {
        None
    }
    /// State dumps of every stream decoder.
    fn states(&mut self) -> Vec<StateDump>;
}

fn cnt(r: opusorus::Result<i32>) -> Result<usize, i32> {
    code(r).map(|n| usize::try_from(n).expect("decoders return non-negative counts"))
}

impl DecApi for Decoder {
    fn channels(&self) -> usize {
        Self::channels(self)
    }
    fn dec_i16(&mut self, d: Option<&[u8]>, p: &mut [i16], n: i32, f: i32) -> Result<usize, i32> {
        cnt(self.opus_decode(d, p, n, f))
    }
    fn dec_i24(&mut self, d: Option<&[u8]>, p: &mut [i32], n: i32, f: i32) -> Result<usize, i32> {
        cnt(self.opus_decode24(d, p, n, f))
    }
    fn dec_f32(&mut self, d: Option<&[u8]>, p: &mut [f32], n: i32, f: i32) -> Result<usize, i32> {
        cnt(self.opus_decode_float(d, p, n, f))
    }
    fn ctl_set(&mut self, req: i32, value: i32) -> Result<(), i32> {
        code(Self::ctl_set(self, req, value))
    }
    fn ctl_get(&mut self, req: i32) -> Result<i32, i32> {
        code(Self::ctl_get(self, req))
    }
    fn nb_samples(&mut self, pkt: &[u8]) -> Option<Result<usize, i32>> {
        Some(code(Self::nb_samples(self, pkt)))
    }
    fn states(&mut self) -> Vec<StateDump> {
        vec![rust_state(self)]
    }
}

impl DecApi for od::Dec {
    fn channels(&self) -> usize {
        unreachable!("the channel count is taken from the Rust side")
    }
    fn dec_i16(&mut self, d: Option<&[u8]>, p: &mut [i16], n: i32, f: i32) -> Result<usize, i32> {
        self.decode(d, p, n, f)
    }
    fn dec_i24(&mut self, d: Option<&[u8]>, p: &mut [i32], n: i32, f: i32) -> Result<usize, i32> {
        self.decode24(d, p, n, f)
    }
    fn dec_f32(&mut self, d: Option<&[u8]>, p: &mut [f32], n: i32, f: i32) -> Result<usize, i32> {
        self.decode_float(d, p, n, f)
    }
    fn ctl_set(&mut self, req: i32, value: i32) -> Result<(), i32> {
        Self::ctl_set(self, req, value)
    }
    fn ctl_get(&mut self, req: i32) -> Result<i32, i32> {
        Self::ctl_get(self, req)
    }
    fn nb_samples(&mut self, pkt: &[u8]) -> Option<Result<usize, i32>> {
        // The C wrapper passes the slice pointer; an empty slice is rejected before any read,
        // but keep the dangling pointer away from C altogether.
        Some(if pkt.is_empty() {
            Err(opusorus_oracle::sys::OPUS_BAD_ARG)
        } else {
            Self::nb_samples(self, pkt)
        })
    }
    fn states(&mut self) -> Vec<StateDump> {
        vec![c_state(&self.dump())]
    }
}

impl DecApi for MsDecoder {
    fn channels(&self) -> usize {
        Self::channels(self)
    }
    fn dec_i16(&mut self, d: Option<&[u8]>, p: &mut [i16], n: i32, f: i32) -> Result<usize, i32> {
        cnt(self.opus_multistream_decode(d, p, n, f))
    }
    fn dec_i24(&mut self, d: Option<&[u8]>, p: &mut [i32], n: i32, f: i32) -> Result<usize, i32> {
        cnt(self.opus_multistream_decode24(d, p, n, f))
    }
    fn dec_f32(&mut self, d: Option<&[u8]>, p: &mut [f32], n: i32, f: i32) -> Result<usize, i32> {
        cnt(self.opus_multistream_decode_float(d, p, n, f))
    }
    fn ctl_set(&mut self, req: i32, value: i32) -> Result<(), i32> {
        code(Self::ctl_set(self, req, value))
    }
    fn ctl_get(&mut self, req: i32) -> Result<i32, i32> {
        code(Self::ctl_get(self, req))
    }
    fn states(&mut self) -> Vec<StateDump> {
        (0..self.streams() as i32)
            .map(|s| rust_state(self.decoder_state(s).expect("stream index in range")))
            .collect()
    }
}

/// A C multistream decoder with its stream count.
#[derive(Debug)]
pub struct CMs {
    /// The decoder.
    pub d: od::MsDec,
    /// Number of streams.
    pub streams: i32,
}

impl DecApi for CMs {
    fn channels(&self) -> usize {
        unreachable!("the channel count is taken from the Rust side")
    }
    fn dec_i16(&mut self, d: Option<&[u8]>, p: &mut [i16], n: i32, f: i32) -> Result<usize, i32> {
        self.d.decode(d, p, n, f)
    }
    fn dec_i24(&mut self, d: Option<&[u8]>, p: &mut [i32], n: i32, f: i32) -> Result<usize, i32> {
        self.d.decode24(d, p, n, f)
    }
    fn dec_f32(&mut self, d: Option<&[u8]>, p: &mut [f32], n: i32, f: i32) -> Result<usize, i32> {
        self.d.decode_float(d, p, n, f)
    }
    fn ctl_set(&mut self, req: i32, value: i32) -> Result<(), i32> {
        self.d.ctl_set(req, value)
    }
    fn ctl_get(&mut self, req: i32) -> Result<i32, i32> {
        self.d.ctl_get(req)
    }
    fn states(&mut self) -> Vec<StateDump> {
        (0..self.streams)
            .map(|s| c_state(&self.d.stream_dump(s)))
            .collect()
    }
}

impl DecApi for ProjectionDecoder {
    fn channels(&self) -> usize {
        Self::channels(self)
    }
    fn dec_i16(&mut self, d: Option<&[u8]>, p: &mut [i16], n: i32, f: i32) -> Result<usize, i32> {
        cnt(self.opus_projection_decode(d, p, n, f))
    }
    fn dec_i24(&mut self, d: Option<&[u8]>, p: &mut [i32], n: i32, f: i32) -> Result<usize, i32> {
        cnt(self.opus_projection_decode24(d, p, n, f))
    }
    fn dec_f32(&mut self, d: Option<&[u8]>, p: &mut [f32], n: i32, f: i32) -> Result<usize, i32> {
        cnt(self.opus_projection_decode_float(d, p, n, f))
    }
    fn ctl_set(&mut self, req: i32, value: i32) -> Result<(), i32> {
        code(Self::ctl_set(self, req, value))
    }
    fn ctl_get(&mut self, req: i32) -> Result<i32, i32> {
        code(Self::ctl_get(self, req))
    }
    fn states(&mut self) -> Vec<StateDump> {
        let ms = self.ms_decoder();
        (0..ms.streams() as i32)
            .map(|s| rust_state(ms.decoder_state(s).expect("stream index in range")))
            .collect()
    }
}

/// A C projection decoder with its stream count.
#[derive(Debug)]
pub struct CProj {
    /// The decoder.
    pub d: od::ProjDec,
    /// Number of streams.
    pub streams: i32,
}

impl DecApi for CProj {
    fn channels(&self) -> usize {
        unreachable!("the channel count is taken from the Rust side")
    }
    fn dec_i16(&mut self, d: Option<&[u8]>, p: &mut [i16], n: i32, f: i32) -> Result<usize, i32> {
        self.d.decode(d, p, n, f)
    }
    fn dec_i24(&mut self, d: Option<&[u8]>, p: &mut [i32], n: i32, f: i32) -> Result<usize, i32> {
        self.d.decode24(d, p, n, f)
    }
    fn dec_f32(&mut self, d: Option<&[u8]>, p: &mut [f32], n: i32, f: i32) -> Result<usize, i32> {
        self.d.decode_float(d, p, n, f)
    }
    fn ctl_set(&mut self, req: i32, value: i32) -> Result<(), i32> {
        self.d.ctl_set(req, value)
    }
    fn ctl_get(&mut self, req: i32) -> Result<i32, i32> {
        self.d.ctl_get(req)
    }
    fn states(&mut self) -> Vec<StateDump> {
        (0..self.streams)
            .map(|s| c_state(&self.d.stream_dump(s)))
            .collect()
    }
}

/// Compares every GET CTL and the full per-stream Opus-level state.
#[track_caller]
pub fn check_state<R: DecApi, C: DecApi>(r: &mut R, c: &mut C, what: &str) {
    for req in DEC_GETS {
        assert_eq!(r.ctl_get(req), c.ctl_get(req), "{what}: ctl_get({req})");
    }
    assert_eq!(r.states(), c.states(), "{what}: decoder state");
}

/// Runs one operation on a Rust and a C decoder and asserts identical behaviour: return codes,
/// every output sample (bit-exact, including untouched sentinel samples), GET CTLs and state.
#[track_caller]
pub fn diff_step<R: DecApi, C: DecApi>(r: &mut R, c: &mut C, op: &DecOp<'_>, fs: i32, i: usize) {
    let ch = r.channels();
    match *op {
        DecOp::Decode {
            data,
            fmt,
            fec,
            fsel,
            ..
        } => {
            let n = frame_size(fsel, fs, data);
            let len = n.clamp(1, MAX_FRAME) as usize * ch + 3;
            let what = format!(
                "op {i}: {fmt:?} decode(len {:?}, n {n}, fec {fec})",
                data.map(<[u8]>::len)
            );
            match fmt {
                Fmt::I16 => {
                    let mut a = vec![0x5A5A_i16; len];
                    let mut b = a.clone();
                    let rr = r.dec_i16(data, &mut a, n, fec);
                    let cr = c.dec_i16(data, &mut b, n, fec);
                    assert_eq!(rr, cr, "{what}: return");
                    assert_slice_eq(&what, &a, &b);
                }
                Fmt::I24 => {
                    let mut a = vec![0x5A5A_5A5A_i32; len];
                    let mut b = a.clone();
                    let rr = r.dec_i24(data, &mut a, n, fec);
                    let cr = c.dec_i24(data, &mut b, n, fec);
                    assert_eq!(rr, cr, "{what}: return");
                    assert_slice_eq(&what, &a, &b);
                }
                Fmt::F32 => {
                    let mut a = vec![1234.5_f32; len];
                    let mut b = a.clone();
                    let rr = r.dec_f32(data, &mut a, n, fec);
                    let cr = c.dec_f32(data, &mut b, n, fec);
                    assert_eq!(rr, cr, "{what}: return");
                    assert_f32_bits_eq(&what, &a, &b);
                }
            }
            check_state(r, c, &what);
        }
        DecOp::Ctl { req, value } => {
            let what = format!("op {i}: ctl_set({req}, {value})");
            assert_eq!(r.ctl_set(req, value), c.ctl_set(req, value), "{what}");
            check_state(r, c, &what);
        }
        DecOp::NbSamples(p) => {
            assert_eq!(r.nb_samples(p), c.nb_samples(p), "op {i}: nb_samples");
        }
    }
}
