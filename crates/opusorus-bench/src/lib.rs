//! # opusorus-bench
//!
//! Shared workloads for the Criterion benchmarks (`benches/codec.rs`, `benches/micro.rs`), the
//! parity tests (`tests/parity.rs`), the markdown report (`bench_report`) and the profiling
//! driver (`bench_profile`).
//!
//! Three implementations are compared ([`Impl`]):
//! * `rust`: opusorus;
//! * `c_scalar`: the `opusorus-oracle` build of libopus (no intrinsics, `-ffp-contract=off`),
//!   which the Rust port is bit-exact with;
//! * `c_opt`: libopus built with its default CMake `Release` configuration (`-O3`, intrinsics,
//!   RTCD), from `opus-sys-optimized`.
//!
//! Every codec workload works on 20 ms frames of deterministic synthetic audio
//! (`opusorus_conformance::signals`) so one benchmark iteration is one 20 ms frame.
//!
//! With the `fixed-point` features (the in-progress fixed-point build of `opusorus`,
//! docs/FIXED_POINT.md) this crate is empty and the benchmarks are no-ops.

#![cfg(not(feature = "fixed-point"))]

use core::fmt;

use opus_sys_optimized as copt;
use opusorus_conformance::signals;
use opusorus_oracle::{api as cscalar, sys};

pub mod micro;

/// Frame duration of every codec workload, in milliseconds.
pub const FRAME_MS: usize = 20;
/// Number of distinct 20 ms frames each workload cycles through (2 s of audio).
pub const FRAMES: usize = 100;
/// Maximum packet size used for encoding.
pub const MAX_PACKET: usize = 4000;

/// Private CTL `OPUS_SET_FORCE_MODE_REQUEST` (`src/opus_private.h`), honoured by all three
/// implementations.
pub const OPUS_SET_FORCE_MODE_REQUEST: i32 = 11002;
/// `MODE_HYBRID` (`src/opus_private.h`).
pub const MODE_HYBRID: i32 = 1001;

/// Error from a workload setup or run: which implementation failed, and how.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchError(pub String);

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BenchError {}

/// Result alias for this crate.
pub type Result<T> = core::result::Result<T, BenchError>;

fn rust_err(what: &str, e: opusorus::Error) -> BenchError {
    BenchError(format!("rust {what}: {e:?}"))
}

fn c_err(which: &str, what: &str, code: i32) -> BenchError {
    BenchError(format!("{which} {what}: C error {code}"))
}

/// Implementation under test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Impl {
    /// opusorus.
    Rust,
    /// The scalar C oracle.
    CScalar,
    /// Upstream optimized C build.
    COpt,
}

impl Impl {
    /// All implementations, in report column order.
    pub const ALL: [Self; 3] = [Self::Rust, Self::CScalar, Self::COpt];

    /// Benchmark id (`rust`, `c_scalar`, `c_opt`).
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::CScalar => "c_scalar",
            Self::COpt => "c_opt",
        }
    }

    /// Parses a benchmark id.
    #[must_use]
    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|i| i.id() == id)
    }
}

/// Test signal family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// `signals::music_like`.
    Music,
    /// `signals::speech_like`.
    Speech,
}

/// A single-stream codec configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodecConfig {
    /// Benchmark id suffix, e.g. `celt_48k_stereo_128k`.
    pub id: &'static str,
    /// Sampling rate (Hz).
    pub fs: i32,
    /// Channel count.
    pub channels: i32,
    /// `OPUS_APPLICATION_*`.
    pub application: i32,
    /// Target bitrate (bit/s).
    pub bitrate: i32,
    /// `OPUS_SIGNAL_*` hint.
    pub signal_hint: i32,
    /// Input signal.
    pub signal: Signal,
    /// `OPUS_SET_FORCE_MODE` value, if the mode is pinned.
    pub force_mode: Option<i32>,
    /// Enables Opus HD (QEXT) on the encoder.
    pub qext: bool,
    /// Expected coding mode of every packet (checked by the parity tests).
    pub expect_mode: Mode,
}

/// Opus coding mode as signalled by the TOC byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// SILK only (configs 0..=11).
    Silk,
    /// Hybrid (configs 12..=15).
    Hybrid,
    /// CELT only (configs 16..=31).
    Celt,
}

impl Mode {
    /// Mode of a packet from its TOC byte.
    #[must_use]
    pub const fn of_toc(toc: u8) -> Self {
        match toc >> 3 {
            0..=11 => Self::Silk,
            12..=15 => Self::Hybrid,
            _ => Self::Celt,
        }
    }
}

impl CodecConfig {
    /// Samples per channel in one 20 ms frame.
    #[must_use]
    pub const fn frame_size(&self) -> usize {
        self.fs as usize * FRAME_MS / 1000
    }

    /// [`FRAMES`] frames of interleaved input audio.
    #[must_use]
    pub fn input(&self) -> Vec<f32> {
        let n = self.frame_size() * FRAMES;
        let ch = self.channels as usize;
        let fs = self.fs as u32;
        match self.signal {
            Signal::Music => signals::music_like(n, ch, fs, 0x0B5E_55ED),
            Signal::Speech => signals::speech_like(n, ch, fs, 0x5EEC_4000),
        }
    }
}

/// The single-stream configurations benchmarked (the QEXT one only with feature `qext`).
#[must_use]
pub fn codec_configs() -> Vec<CodecConfig> {
    let mut v = vec![
        CodecConfig {
            id: "celt_48k_stereo_128k",
            fs: 48000,
            channels: 2,
            application: sys::OPUS_APPLICATION_AUDIO,
            bitrate: 128_000,
            signal_hint: sys::OPUS_SIGNAL_MUSIC,
            signal: Signal::Music,
            force_mode: None,
            qext: false,
            expect_mode: Mode::Celt,
        },
        CodecConfig {
            id: "hybrid_48k_mono_32k",
            fs: 48000,
            channels: 1,
            application: sys::OPUS_APPLICATION_AUDIO,
            bitrate: 32_000,
            signal_hint: sys::OPUS_SIGNAL_VOICE,
            signal: Signal::Speech,
            force_mode: Some(MODE_HYBRID),
            qext: false,
            expect_mode: Mode::Hybrid,
        },
        CodecConfig {
            id: "silk_16k_mono_16k_voip",
            fs: 16000,
            channels: 1,
            application: sys::OPUS_APPLICATION_VOIP,
            bitrate: 16_000,
            signal_hint: sys::OPUS_SIGNAL_VOICE,
            signal: Signal::Speech,
            force_mode: None,
            qext: false,
            expect_mode: Mode::Silk,
        },
    ];
    if cfg!(feature = "qext") {
        v.push(CodecConfig {
            id: "qext_96k_stereo_256k",
            fs: 96000,
            channels: 2,
            application: sys::OPUS_APPLICATION_AUDIO,
            bitrate: 256_000,
            signal_hint: sys::OPUS_SIGNAL_MUSIC,
            signal: Signal::Music,
            force_mode: None,
            qext: true,
            expect_mode: Mode::Celt,
        });
    }
    v
}

/// Complexities of the encoder benchmarks.
pub const ENCODE_COMPLEXITIES: [i32; 2] = [10, 5];

/// An encoder of any implementation.
#[derive(Debug)]
pub enum AnyEncoder {
    /// opusorus.
    Rust(Box<opusorus::Encoder>),
    /// Scalar C oracle.
    CScalar(cscalar::Encoder),
    /// Optimized C.
    COpt(copt::Encoder),
}

impl AnyEncoder {
    /// Creates and configures an encoder for `cfg` at `complexity`.
    ///
    /// # Errors
    /// If creation or a CTL fails.
    pub fn new(imp: Impl, cfg: &CodecConfig, complexity: i32) -> Result<Self> {
        let mut ctls = vec![
            (sys::OPUS_SET_BITRATE_REQUEST, cfg.bitrate),
            (sys::OPUS_SET_COMPLEXITY_REQUEST, complexity),
            (sys::OPUS_SET_SIGNAL_REQUEST, cfg.signal_hint),
        ];
        if let Some(m) = cfg.force_mode {
            ctls.push((OPUS_SET_FORCE_MODE_REQUEST, m));
        }
        if cfg.qext {
            ctls.push((sys::OPUS_SET_QEXT_REQUEST, 1));
        }
        let (fs, ch, app) = (cfg.fs, cfg.channels, cfg.application);
        let mut enc = match imp {
            Impl::Rust => Self::Rust(Box::new(
                opusorus::Encoder::new_raw(fs, ch, app).map_err(|e| rust_err("encoder", e))?,
            )),
            Impl::CScalar => Self::CScalar(
                cscalar::Encoder::new(fs, ch, app).map_err(|e| c_err("c_scalar", "encoder", e))?,
            ),
            Impl::COpt => Self::COpt(
                copt::Encoder::new(fs, ch, app).map_err(|e| c_err("c_opt", "encoder", e))?,
            ),
        };
        for (req, val) in ctls {
            enc.ctl_set(req, val)?;
        }
        Ok(enc)
    }

    /// Setter CTL.
    ///
    /// # Errors
    /// If the implementation rejects the request.
    pub fn ctl_set(&mut self, req: i32, val: i32) -> Result<()> {
        match self {
            Self::Rust(e) => e
                .ctl_set(req, val)
                .map_err(|x| rust_err(&format!("ctl {req}"), x)),
            Self::CScalar(e) => e
                .ctl_set(req, val)
                .map_err(|x| c_err("c_scalar", &format!("ctl {req}"), x)),
            Self::COpt(e) => e
                .ctl_set(req, val)
                .map_err(|x| c_err("c_opt", &format!("ctl {req}"), x)),
        }
    }

    /// Encodes one frame of float PCM; returns the packet length.
    ///
    /// # Errors
    /// If encoding fails.
    pub fn encode(&mut self, pcm: &[f32], frame_size: usize, out: &mut [u8]) -> Result<usize> {
        match self {
            Self::Rust(e) => e
                .encode_float(pcm, frame_size, out)
                .map_err(|x| rust_err("encode", x)),
            Self::CScalar(e) => e
                .encode_float(pcm, frame_size, out)
                .map_err(|x| c_err("c_scalar", "encode", x)),
            Self::COpt(e) => e
                .encode_float(pcm, frame_size, out)
                .map_err(|x| c_err("c_opt", "encode", x)),
        }
    }
}

/// A decoder of any implementation.
#[derive(Debug)]
pub enum AnyDecoder {
    /// opusorus.
    Rust(Box<opusorus::Decoder>),
    /// Scalar C oracle.
    CScalar(cscalar::Decoder),
    /// Optimized C.
    COpt(copt::Decoder),
}

impl AnyDecoder {
    /// Creates a decoder.
    ///
    /// # Errors
    /// If creation fails.
    pub fn new(imp: Impl, fs: i32, channels: i32) -> Result<Self> {
        Ok(match imp {
            Impl::Rust => Self::Rust(Box::new(
                opusorus::Decoder::new(fs, channels).map_err(|e| rust_err("decoder", e))?,
            )),
            Impl::CScalar => Self::CScalar(
                cscalar::Decoder::new(fs, channels).map_err(|e| c_err("c_scalar", "decoder", e))?,
            ),
            Impl::COpt => Self::COpt(
                copt::Decoder::new(fs, channels).map_err(|e| c_err("c_opt", "decoder", e))?,
            ),
        })
    }

    /// Decodes one packet to float PCM; returns samples per channel.
    ///
    /// # Errors
    /// If decoding fails.
    pub fn decode(&mut self, packet: &[u8], pcm: &mut [f32], frame_size: usize) -> Result<usize> {
        match self {
            Self::Rust(d) => d
                .decode_float(Some(packet), pcm, frame_size, false)
                .map_err(|x| rust_err("decode", x)),
            Self::CScalar(d) => d
                .decode_float(Some(packet), pcm, frame_size, false)
                .map_err(|x| c_err("c_scalar", "decode", x)),
            Self::COpt(d) => d
                .decode_float(Some(packet), pcm, frame_size, false)
                .map_err(|x| c_err("c_opt", "decode", x)),
        }
    }
}

/// Encodes all [`FRAMES`] frames of `cfg`'s input with `imp` at `complexity`.
///
/// # Errors
/// If encoding fails.
pub fn encode_all(imp: Impl, cfg: &CodecConfig, complexity: i32) -> Result<Vec<Vec<u8>>> {
    let mut enc = AnyEncoder::new(imp, cfg, complexity)?;
    let input = cfg.input();
    let n = cfg.frame_size() * cfg.channels as usize;
    let mut out = vec![0u8; MAX_PACKET];
    input
        .chunks_exact(n)
        .map(|f| {
            let len = enc.encode(f, cfg.frame_size(), &mut out)?;
            Ok(out[..len].to_vec())
        })
        .collect()
}

/// The packets every decode benchmark decodes: the Rust encoder's output at complexity 10
/// (bit-identical to the scalar C encoder's).
///
/// # Errors
/// If encoding fails.
pub fn decode_packets(cfg: &CodecConfig) -> Result<Vec<Vec<u8>>> {
    encode_all(Impl::Rust, cfg, 10)
}

/// 5.1 surround multistream configuration (mapping family 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurroundConfig {
    /// Benchmark id suffix.
    pub id: &'static str,
    /// Sampling rate.
    pub fs: i32,
    /// Channels (6).
    pub channels: i32,
    /// Total bitrate.
    pub bitrate: i32,
}

/// The benchmarked 5.1 configuration: 48 kHz, 256 kbit/s total, music.
pub const SURROUND: SurroundConfig = SurroundConfig {
    id: "surround51_48k_256k",
    fs: 48000,
    channels: 6,
    bitrate: 256_000,
};

impl SurroundConfig {
    /// Samples per channel in one 20 ms frame.
    #[must_use]
    pub const fn frame_size(&self) -> usize {
        self.fs as usize * FRAME_MS / 1000
    }

    /// [`FRAMES`] frames of interleaved 6-channel music.
    #[must_use]
    pub fn input(&self) -> Vec<f32> {
        signals::music_like(
            self.frame_size() * FRAMES,
            self.channels as usize,
            self.fs as u32,
            0x51,
        )
    }
}

/// Multistream surround encoder of any implementation.
#[derive(Debug)]
pub enum AnyMsEncoder {
    /// opusorus.
    Rust(Box<opusorus::MsEncoder>),
    /// Scalar C oracle.
    CScalar(cscalar::MsEncoder),
    /// Optimized C.
    COpt(copt::MsEncoder),
}

/// Stream layout of a surround encoder: `(streams, coupled_streams, mapping)`.
pub type Layout = (i32, i32, Vec<u8>);

impl AnyMsEncoder {
    /// Creates a family-1 surround encoder for `cfg` at `complexity`.
    ///
    /// # Errors
    /// If creation or a CTL fails.
    pub fn new(imp: Impl, cfg: &SurroundConfig, complexity: i32) -> Result<Self> {
        let app = sys::OPUS_APPLICATION_AUDIO;
        let (fs, ch) = (cfg.fs, cfg.channels);
        let ctls = [
            (sys::OPUS_SET_BITRATE_REQUEST, cfg.bitrate),
            (sys::OPUS_SET_COMPLEXITY_REQUEST, complexity),
        ];
        Ok(match imp {
            Impl::Rust => {
                let mut e = opusorus::MsEncoder::new_surround_raw(fs, ch, 1, app)
                    .map_err(|e| rust_err("ms encoder", e))?;
                for (r, v) in ctls {
                    e.ctl_set(r, v).map_err(|e| rust_err("ms ctl", e))?;
                }
                Self::Rust(Box::new(e))
            }
            Impl::CScalar => {
                let mut e = cscalar::MsEncoder::new_surround(fs, ch, 1, app)
                    .map_err(|e| c_err("c_scalar", "ms encoder", e))?;
                for (r, v) in ctls {
                    e.ctl_set(r, v)
                        .map_err(|e| c_err("c_scalar", "ms ctl", e))?;
                }
                Self::CScalar(e)
            }
            Impl::COpt => {
                let mut e = copt::MsEncoder::new_surround(fs, ch, 1, app)
                    .map_err(|e| c_err("c_opt", "ms encoder", e))?;
                for (r, v) in ctls {
                    e.ctl_set(r, v).map_err(|e| c_err("c_opt", "ms ctl", e))?;
                }
                Self::COpt(e)
            }
        })
    }

    /// `(streams, coupled_streams, mapping)` chosen by the encoder.
    #[must_use]
    pub fn layout(&self) -> Layout {
        match self {
            Self::Rust(e) => (e.streams(), e.coupled_streams(), e.mapping().to_vec()),
            Self::CScalar(e) => (e.streams, e.coupled_streams, e.mapping.clone()),
            Self::COpt(e) => (e.streams, e.coupled_streams, e.mapping.clone()),
        }
    }

    /// Encodes one frame; returns the packet length.
    ///
    /// # Errors
    /// If encoding fails.
    pub fn encode(&mut self, pcm: &[f32], frame_size: usize, out: &mut [u8]) -> Result<usize> {
        match self {
            Self::Rust(e) => e
                .encode_float(pcm, frame_size, out)
                .map_err(|x| rust_err("ms encode", x)),
            Self::CScalar(e) => e
                .encode_float(pcm, frame_size, out)
                .map_err(|x| c_err("c_scalar", "ms encode", x)),
            Self::COpt(e) => e
                .encode_float(pcm, frame_size, out)
                .map_err(|x| c_err("c_opt", "ms encode", x)),
        }
    }
}

/// Multistream decoder of any implementation.
#[derive(Debug)]
pub enum AnyMsDecoder {
    /// opusorus.
    Rust(Box<opusorus::MsDecoder>),
    /// Scalar C oracle.
    CScalar(cscalar::MsDecoder),
    /// Optimized C.
    COpt(copt::MsDecoder),
}

impl AnyMsDecoder {
    /// Creates a multistream decoder for `layout`.
    ///
    /// # Errors
    /// If creation fails.
    pub fn new(imp: Impl, cfg: &SurroundConfig, layout: &Layout) -> Result<Self> {
        let (s, c, m) = (layout.0, layout.1, layout.2.as_slice());
        let (fs, ch) = (cfg.fs, cfg.channels);
        Ok(match imp {
            Impl::Rust => Self::Rust(Box::new(
                opusorus::MsDecoder::new(fs, ch, s, c, m).map_err(|e| rust_err("ms decoder", e))?,
            )),
            Impl::CScalar => Self::CScalar(
                cscalar::MsDecoder::new(fs, ch, s, c, m)
                    .map_err(|e| c_err("c_scalar", "ms decoder", e))?,
            ),
            Impl::COpt => Self::COpt(
                copt::MsDecoder::new(fs, ch, s, c, m)
                    .map_err(|e| c_err("c_opt", "ms decoder", e))?,
            ),
        })
    }

    /// Decodes one packet; returns samples per channel.
    ///
    /// # Errors
    /// If decoding fails.
    pub fn decode(&mut self, packet: &[u8], pcm: &mut [f32], frame_size: usize) -> Result<usize> {
        match self {
            Self::Rust(d) => d
                .decode_float(Some(packet), pcm, frame_size, false)
                .map_err(|x| rust_err("ms decode", x)),
            Self::CScalar(d) => d
                .decode_float(Some(packet), pcm, frame_size, false)
                .map_err(|x| c_err("c_scalar", "ms decode", x)),
            Self::COpt(d) => d
                .decode_float(Some(packet), pcm, frame_size, false)
                .map_err(|x| c_err("c_opt", "ms decode", x)),
        }
    }
}

/// Encodes all [`FRAMES`] frames of the surround input with `imp` at `complexity`; returns the
/// stream layout and the packets.
///
/// # Errors
/// If encoding fails.
pub fn surround_encode_all(
    imp: Impl,
    cfg: &SurroundConfig,
    complexity: i32,
) -> Result<(Layout, Vec<Vec<u8>>)> {
    let mut enc = AnyMsEncoder::new(imp, cfg, complexity)?;
    let input = cfg.input();
    let n = cfg.frame_size() * cfg.channels as usize;
    let mut out = vec![0u8; MAX_PACKET * 4];
    let packets = input
        .chunks_exact(n)
        .map(|f| {
            let len = enc.encode(f, cfg.frame_size(), &mut out)?;
            Ok(out[..len].to_vec())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((enc.layout(), packets))
}

/// One row of the markdown report: a Criterion group with one benchmark per [`Impl`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportRow {
    /// Criterion group name.
    pub group: String,
    /// Human-readable label.
    pub label: String,
    /// Whether one iteration is one 20 ms frame (so a realtime factor applies).
    pub per_frame: bool,
}

/// All report rows, codec benchmarks first, then the micro kernels.
#[must_use]
pub fn report_rows() -> Vec<ReportRow> {
    let mut rows: Vec<ReportRow> = CodecBench::all()
        .iter()
        .map(|b| ReportRow {
            group: b.group(),
            label: b.label(),
            per_frame: true,
        })
        .collect();
    rows.extend(micro::MicroBench::ALL.iter().map(|m| ReportRow {
        group: m.group().to_owned(),
        label: m.label().to_owned(),
        per_frame: m.per_frame(),
    }));
    rows
}

/// Looks up a benchmark (codec or micro) by its Criterion group name and builds its runner.
///
/// # Errors
/// Unknown group, or setup failure.
pub fn runner_for_group(group: &str, imp: Impl) -> Result<Runner> {
    if let Some(b) = CodecBench::all().into_iter().find(|b| b.group() == group) {
        return b.runner(imp);
    }
    if let Some(m) = micro::MicroBench::ALL
        .into_iter()
        .find(|m| m.group() == group)
    {
        return m.runner(imp);
    }
    Err(BenchError(format!("unknown benchmark group {group:?}")))
}

/// A ready-to-run benchmark body: each call processes one 20 ms frame (or one micro-kernel
/// call) and cycles through its prepared inputs. Used by `bench_profile` and the report's
/// description of each row.
pub type Runner = Box<dyn FnMut() -> Result<()>>;

/// Codec benchmark kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecBench {
    /// Single-stream decode of `codec_configs()[i]`.
    Decode(CodecConfig),
    /// Single-stream encode at the given complexity.
    Encode(CodecConfig, i32),
    /// 5.1 surround decode.
    SurroundDecode,
    /// 5.1 surround encode at the given complexity.
    SurroundEncode(i32),
}

impl CodecBench {
    /// All codec benchmarks.
    #[must_use]
    pub fn all() -> Vec<Self> {
        let cfgs = codec_configs();
        let mut v: Vec<Self> = cfgs.iter().map(|c| Self::Decode(*c)).collect();
        for c in &cfgs {
            for cx in ENCODE_COMPLEXITIES {
                v.push(Self::Encode(*c, cx));
            }
        }
        v.push(Self::SurroundDecode);
        for cx in ENCODE_COMPLEXITIES {
            v.push(Self::SurroundEncode(cx));
        }
        v
    }

    /// Criterion group name, e.g. `decode_celt_48k_stereo_128k`, `encode_c10_...`.
    #[must_use]
    pub fn group(&self) -> String {
        match self {
            Self::Decode(c) => format!("decode_{}", c.id),
            Self::Encode(c, cx) => format!("encode_c{cx}_{}", c.id),
            Self::SurroundDecode => format!("decode_{}", SURROUND.id),
            Self::SurroundEncode(cx) => format!("encode_c{cx}_{}", SURROUND.id),
        }
    }

    /// Report label, e.g. `decode celt_48k_stereo_128k`.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Decode(c) => format!("decode {}", c.id),
            Self::Encode(c, cx) => format!("encode {} (cx{cx})", c.id),
            Self::SurroundDecode => format!("decode {}", SURROUND.id),
            Self::SurroundEncode(cx) => format!("encode {} (cx{cx})", SURROUND.id),
        }
    }

    /// Builds the per-frame runner for `imp`.
    ///
    /// # Errors
    /// If setup (encoder/decoder creation, packet preparation) fails.
    pub fn runner(&self, imp: Impl) -> Result<Runner> {
        match *self {
            Self::Decode(cfg) => {
                let packets = decode_packets(&cfg)?;
                let mut dec = AnyDecoder::new(imp, cfg.fs, cfg.channels)?;
                let fsz = cfg.frame_size();
                let mut pcm = vec![0f32; fsz * cfg.channels as usize];
                let mut i = 0;
                Ok(Box::new(move || {
                    let p = &packets[i];
                    i = (i + 1) % packets.len();
                    dec.decode(core::hint::black_box(p), &mut pcm, fsz)?;
                    core::hint::black_box(&pcm);
                    Ok(())
                }))
            }
            Self::Encode(cfg, cx) => {
                let input = cfg.input();
                let mut enc = AnyEncoder::new(imp, &cfg, cx)?;
                let fsz = cfg.frame_size();
                let n = fsz * cfg.channels as usize;
                let mut out = vec![0u8; MAX_PACKET];
                let mut i = 0;
                Ok(Box::new(move || {
                    let f = &input[i * n..(i + 1) * n];
                    i = (i + 1) % FRAMES;
                    let len = enc.encode(core::hint::black_box(f), fsz, &mut out)?;
                    core::hint::black_box(len);
                    Ok(())
                }))
            }
            Self::SurroundDecode => {
                let (layout, packets) = surround_encode_all(Impl::Rust, &SURROUND, 10)?;
                let mut dec = AnyMsDecoder::new(imp, &SURROUND, &layout)?;
                let fsz = SURROUND.frame_size();
                let mut pcm = vec![0f32; fsz * SURROUND.channels as usize];
                let mut i = 0;
                Ok(Box::new(move || {
                    let p = &packets[i];
                    i = (i + 1) % packets.len();
                    dec.decode(core::hint::black_box(p), &mut pcm, fsz)?;
                    core::hint::black_box(&pcm);
                    Ok(())
                }))
            }
            Self::SurroundEncode(cx) => {
                let input = SURROUND.input();
                let mut enc = AnyMsEncoder::new(imp, &SURROUND, cx)?;
                let fsz = SURROUND.frame_size();
                let n = fsz * SURROUND.channels as usize;
                let mut out = vec![0u8; MAX_PACKET * 4];
                let mut i = 0;
                Ok(Box::new(move || {
                    let f = &input[i * n..(i + 1) * n];
                    i = (i + 1) % FRAMES;
                    let len = enc.encode(core::hint::black_box(f), fsz, &mut out)?;
                    core::hint::black_box(len);
                    Ok(())
                }))
            }
        }
    }
}
