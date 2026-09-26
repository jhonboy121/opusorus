//! `opus_demo`: port of libopus `src/opus_demo.c`, the reference encoder/decoder command-line
//! tool (also the decoder the RFC 6716/8251 conformance procedure runs).
//!
//! ```text
//! opus_demo [-e] <application> <sampling rate (Hz)> <channels (1/2)> <bits per second> [options] <input> <output>
//! opus_demo -d <sampling rate (Hz)> <channels (1/2)> [options] <input> <output>
//! ```
//!
//! [`opus_demo_main`] reproduces the C command line, the files it writes (raw little-endian PCM
//! in 16-bit, 24-bit or float format, and the bitstream format below) and the exact text it
//! prints, so the `opus_demo` binary is a drop-in replacement for the C one.
//!
//! Bitstream file format (`-e` output, `-d` input): for every packet, its length as a 4-byte
//! big-endian integer, the encoder's final range coder state as a 4-byte big-endian integer, then
//! the packet bytes. A lost packet is stored with length 0 and final range 0.
//!
//! Faithfulness notes:
//! * Packet loss simulation (`-loss`), `-random_framesize` and `-random_fec` draw from C `rand()`
//!   without seeding it; [`GlibcRand`] reproduces the glibc generator, so on glibc hosts the
//!   losses (and therefore the output files) are identical to the C tool's.
//! * The C quirks are kept: the input sample buffer is refilled from its start even when encoded
//!   samples are carried over (`-delayed-decision`), stale bytes of the shared file buffer are
//!   converted after a short read, errors from `opus_*_ctl` calls are ignored, the process exit
//!   status after a DRED parse attempt is that call's return value, and an error from the
//!   decoder is added to the sample total used by the statistics.
//! * DRED: `-dred <n>` sets `OPUS_SET_DRED_DURATION` on the encoder (a no-op returning
//!   `OPUS_UNIMPLEMENTED` without the `dred` feature, as in C). With this crate's `dred` feature
//!   the decoder side is upstream's: after a burst of losses the next packet's redundancy is
//!   parsed (`DredDecoder::parse`, `opus_dred_parse`) and the lost frames are concealed from it
//!   (`Decoder::opus_decoder_dred_decode24`); without it, it behaves like a C build without
//!   `ENABLE_DRED` (`opus_dred_parse` returns `OPUS_UNIMPLEMENTED`, recovery uses regular
//!   PLC/FEC).
//! * DNN weights (features `deep-plc`, `dred`, `osce`): with `dnn-weights-embedded` the models
//!   are compiled in, as in a default libopus DNN build. Otherwise [`opus_demo_main`] reads
//!   `weights_blob.bin` from the working directory and sets it on the encoder, decoder and DRED
//!   decoder like a libopus `USE_WEIGHTS_FILE` build (silently running without models if the
//!   file does not exist); [`opus_demo_main_with_weights`] takes the blob as an argument.
//! * `-sim_loss <perc>` (feature `lossgen`, libopus `--enable-lossgen`): losses drawn from the
//!   generative loss model (`opusorus::lossgen`), which takes its randomness from the same
//!   [`GlibcRand`] as the other options, as C's `sample_loss` calls `rand()`.
//! * `ENABLE_OSCE_TRAINING_DATA` (feature `osce-training-data`, libopus
//!   `--enable-osce-training-data`): the encoder is forced to SILK-only mode and `rand()` is
//!   reseeded with `srand(0)`, `-silk_random_switching <n>` picks a random bitrate, complexity,
//!   loss percentage and VBR setting every `n`-th frame, and the library writes the training
//!   files into the working directory (`opusorus::osce_training_data`; [`opus_demo_main`]
//!   closes them before returning, as the C process exit does).
//! * Where C has undefined behaviour (a negative sample count to `fread` after a frame-size
//!   change, out-of-range float input), the Rust port reads nothing / saturates instead.
//! * I/O errors while reading an opened input file are returned as `io::Error` (C treats them as
//!   end of file); write errors print C's `Error writing.` message.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};

#[cfg(not(feature = "dred"))]
use opusorus::Error;
use opusorus::celt::celt::{opus_get_version_string, opus_strerror};
use opusorus::constants::raw::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_CELT, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    OPUS_APPLICATION_RESTRICTED_SILK, OPUS_APPLICATION_VOIP, OPUS_AUTO, OPUS_BANDWIDTH_FULLBAND,
    OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND, OPUS_BANDWIDTH_SUPERWIDEBAND,
    OPUS_BANDWIDTH_WIDEBAND, OPUS_FRAMESIZE_2_5_MS, OPUS_FRAMESIZE_5_MS, OPUS_FRAMESIZE_10_MS,
    OPUS_FRAMESIZE_20_MS, OPUS_FRAMESIZE_40_MS, OPUS_FRAMESIZE_60_MS, OPUS_FRAMESIZE_80_MS,
    OPUS_FRAMESIZE_100_MS, OPUS_FRAMESIZE_120_MS, OPUS_FRAMESIZE_ARG,
};
#[cfg(feature = "dred")]
use opusorus::dred::{Dred, DredDecoder};
#[cfg(feature = "qext")]
use opusorus::encoder::request::OPUS_SET_QEXT_REQUEST;
use opusorus::encoder::request::{
    OPUS_SET_BANDWIDTH_REQUEST, OPUS_SET_BITRATE_REQUEST, OPUS_SET_COMPLEXITY_REQUEST,
    OPUS_SET_DRED_DURATION_REQUEST, OPUS_SET_DTX_REQUEST, OPUS_SET_EXPERT_FRAME_DURATION_REQUEST,
    OPUS_SET_FORCE_CHANNELS_REQUEST, OPUS_SET_FORCE_MODE_REQUEST, OPUS_SET_INBAND_FEC_REQUEST,
    OPUS_SET_LSB_DEPTH_REQUEST, OPUS_SET_PACKET_LOSS_PERC_REQUEST, OPUS_SET_VBR_CONSTRAINT_REQUEST,
    OPUS_SET_VBR_REQUEST,
};
#[cfg(feature = "lossgen")]
use opusorus::lossgen::{LossGenState, sample_loss};
use opusorus::packet::{
    MODE_CELT_ONLY, MODE_SILK_ONLY, get_nb_frames, get_samples_per_frame, has_lbrr,
};
use opusorus::{Decoder, Encoder};

#[cfg(feature = "lossgen")]
use crate::compare::c_atof;
use crate::compare::{EXIT_FAILURE, EXIT_SUCCESS, argv0, c_atoi, c_fmt_f, c_isspace};

/// `MAX_PACKET`: largest payload accepted by `-max_payload` (and the default).
pub const MAX_PACKET: i32 = 15000;

/// Whether this build has deep PLC (feature `deep-plc`, implied by `dred` and `osce`).
pub const DEEP_PLC: bool = cfg!(feature = "deep-plc");
/// Whether this build has DRED decoding (feature `dred`).
pub const DRED: bool = cfg!(feature = "dred");
/// Whether this build has OSCE (feature `osce`).
pub const OSCE: bool = cfg!(feature = "osce");
/// Whether this build has `-sim_loss` (feature `lossgen`, libopus `ENABLE_LOSSGEN`).
pub const LOSSGEN: bool = cfg!(feature = "lossgen");
/// Whether this build writes OSCE training data (feature `osce-training-data`, libopus
/// `ENABLE_OSCE_TRAINING_DATA`).
pub const OSCE_TRAINING_DATA: bool = cfg!(feature = "osce-training-data");
/// Whether the DNN weights are compiled in (feature `dnn-weights-embedded`).
pub const WEIGHTS_EMBEDDED: bool = cfg!(feature = "dnn-weights-embedded");

/// The weight file of a libopus `USE_WEIGHTS_FILE` `opus_demo`.
pub const WEIGHTS_FILE: &str = "weights_blob.bin";

/// `MAX_SAMPLING_RATE`.
#[cfg(feature = "qext")]
pub const MAX_SAMPLING_RATE: i32 = 96000;
/// `MAX_SAMPLING_RATE`.
#[cfg(not(feature = "qext"))]
pub const MAX_SAMPLING_RATE: i32 = 48000;

/// PCM file formats (`FORMAT_S16_LE`, `FORMAT_S24_LE`, `FORMAT_F32_LE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    S16,
    S24,
    F32,
}

impl Format {
    /// `format_size[format]`.
    const fn size(self) -> usize {
        match self {
            Self::S16 => 2,
            Self::S24 => 3,
            Self::F32 => 4,
        }
    }
}

/// One row of the `-*_test` mode lists: `{mode, bandwidth, frame size at 48 kHz, channels}`.
type ModeEntry = [i32; 4];

const NB: i32 = OPUS_BANDWIDTH_NARROWBAND;
const MB: i32 = OPUS_BANDWIDTH_MEDIUMBAND;
const WB: i32 = OPUS_BANDWIDTH_WIDEBAND;
const SWB: i32 = OPUS_BANDWIDTH_SUPERWIDEBAND;
const FB: i32 = OPUS_BANDWIDTH_FULLBAND;
const SILK: i32 = MODE_SILK_ONLY;
const CELT: i32 = MODE_CELT_ONLY;

/// `silk8_test`.
static SILK8_TEST: [ModeEntry; 8] = [
    [SILK, NB, 960 * 3, 1],
    [SILK, NB, 960 * 2, 1],
    [SILK, NB, 960, 1],
    [SILK, NB, 480, 1],
    [SILK, NB, 960 * 3, 2],
    [SILK, NB, 960 * 2, 2],
    [SILK, NB, 960, 2],
    [SILK, NB, 480, 2],
];

/// `silk12_test`.
static SILK12_TEST: [ModeEntry; 8] = [
    [SILK, MB, 960 * 3, 1],
    [SILK, MB, 960 * 2, 1],
    [SILK, MB, 960, 1],
    [SILK, MB, 480, 1],
    [SILK, MB, 960 * 3, 2],
    [SILK, MB, 960 * 2, 2],
    [SILK, MB, 960, 2],
    [SILK, MB, 480, 2],
];

/// `silk16_test`.
static SILK16_TEST: [ModeEntry; 8] = [
    [SILK, WB, 960 * 3, 1],
    [SILK, WB, 960 * 2, 1],
    [SILK, WB, 960, 1],
    [SILK, WB, 480, 1],
    [SILK, WB, 960 * 3, 2],
    [SILK, WB, 960 * 2, 2],
    [SILK, WB, 960, 2],
    [SILK, WB, 480, 2],
];

/// `silk_bw_switch_test`.
static SILK_BW_SWITCH_TEST: [ModeEntry; 20] = [
    [SILK, WB, 960, 1],
    [SILK, NB, 960, 1],
    [SILK, MB, 960, 1],
    [SILK, SWB, 960, 1],
    [SILK, FB, 960, 1],
    [SILK, WB, 960, 2],
    [SILK, NB, 960, 2],
    [SILK, MB, 960, 2],
    [SILK, SWB, 960, 2],
    [SILK, FB, 960, 2],
    [SILK, WB, 480, 1],
    [SILK, NB, 480, 1],
    [SILK, MB, 480, 1],
    [SILK, SWB, 480, 1],
    [SILK, FB, 480, 1],
    [SILK, WB, 480, 2],
    [SILK, NB, 480, 2],
    [SILK, MB, 480, 2],
    [SILK, SWB, 480, 2],
    [SILK, FB, 480, 2],
];

/// `hybrid24_test` (SILK-only mode forced, as in C).
static HYBRID24_TEST: [ModeEntry; 4] = [
    [SILK, SWB, 960, 1],
    [SILK, SWB, 480, 1],
    [SILK, SWB, 960, 2],
    [SILK, SWB, 480, 2],
];

/// `hybrid48_test` (SILK-only mode forced, as in C).
static HYBRID48_TEST: [ModeEntry; 4] = [
    [SILK, FB, 960, 1],
    [SILK, FB, 480, 1],
    [SILK, FB, 960, 2],
    [SILK, FB, 480, 2],
];

/// `celt_test`.
static CELT_TEST: [ModeEntry; 32] = [
    [CELT, FB, 960, 1],
    [CELT, SWB, 960, 1],
    [CELT, WB, 960, 1],
    [CELT, NB, 960, 1],
    [CELT, FB, 480, 1],
    [CELT, SWB, 480, 1],
    [CELT, WB, 480, 1],
    [CELT, NB, 480, 1],
    [CELT, FB, 240, 1],
    [CELT, SWB, 240, 1],
    [CELT, WB, 240, 1],
    [CELT, NB, 240, 1],
    [CELT, FB, 120, 1],
    [CELT, SWB, 120, 1],
    [CELT, WB, 120, 1],
    [CELT, NB, 120, 1],
    [CELT, FB, 960, 2],
    [CELT, SWB, 960, 2],
    [CELT, WB, 960, 2],
    [CELT, NB, 960, 2],
    [CELT, FB, 480, 2],
    [CELT, SWB, 480, 2],
    [CELT, WB, 480, 2],
    [CELT, NB, 480, 2],
    [CELT, FB, 240, 2],
    [CELT, SWB, 240, 2],
    [CELT, WB, 240, 2],
    [CELT, NB, 240, 2],
    [CELT, FB, 120, 2],
    [CELT, SWB, 120, 2],
    [CELT, WB, 120, 2],
    [CELT, NB, 120, 2],
];

/// `celt_hq_test`.
static CELT_HQ_TEST: [ModeEntry; 4] = [
    [CELT, FB, 120, 2],
    [CELT, FB, 240, 2],
    [CELT, FB, 480, 2],
    [CELT, FB, 960, 2],
];

/// The glibc `rand()` generator (`random_r` with the default `TYPE_3` state: an additive lagged
/// Fibonacci generator `r[i] = r[i-3] + r[i-31]`, outputs `r >> 1`), in the state an unseeded
/// program starts with (`srand(1)`).
///
/// `opus_demo` never seeds `rand()`, so this reproduces the C tool's packet loss pattern and
/// random frame size / FEC switching on glibc hosts.
#[derive(Debug, Clone)]
pub struct GlibcRand {
    r: [u32; 31],
    f: usize,
    b: usize,
}

impl GlibcRand {
    /// `srand(seed)` (glibc maps seed 0 to 1).
    #[must_use]
    pub fn new(seed: u32) -> Self {
        let mut r = [0u32; 31];
        let seed = if seed == 0 { 1 } else { seed };
        // glibc keeps the state as int32_t and the seed as a (possibly negative) int32_t.
        let mut word = i64::from(seed as i32);
        r[0] = seed;
        for v in r.iter_mut().skip(1) {
            let hi = word / 127_773;
            let lo = word % 127_773;
            word = 16807 * lo - 2836 * hi;
            if word < 0 {
                word += 2_147_483_647;
            }
            *v = word as u32;
        }
        let mut s = Self { r, f: 3, b: 0 };
        for _ in 0..310 {
            s.next_value();
        }
        s
    }

    /// `rand()`: the next value in `0..=RAND_MAX` (`RAND_MAX` = 2^31 - 1).
    pub const fn next_value(&mut self) -> i32 {
        let val = self.r[self.f].wrapping_add(self.r[self.b]);
        self.r[self.f] = val;
        self.f = if self.f == 30 { 0 } else { self.f + 1 };
        self.b = if self.b == 30 { 0 } else { self.b + 1 };
        (val >> 1) as i32
    }
}

impl Default for GlibcRand {
    fn default() -> Self {
        Self::new(1)
    }
}

/// `-sim_loss` state: `(lossgen_perc, lossgen)` once the option was given.
#[cfg(feature = "lossgen")]
type SimLoss = Option<(f32, LossGenState)>;
/// Without `lossgen` there is no `-sim_loss` (always `None`).
#[cfg(not(feature = "lossgen"))]
type SimLoss = Option<core::convert::Infallible>;

/// `else if (lossgen_perc >= 0) lost = sample_loss(&lossgen, lossgen_perc*.01f);`: `None` when
/// the branch is not taken.
#[cfg(feature = "lossgen")]
fn sim_loss_draw(sim: &mut SimLoss, rng: &mut GlibcRand) -> Option<i32> {
    match sim {
        Some((perc, st)) if *perc >= 0.0 => {
            Some(sample_loss(st, *perc * 0.01f32, &mut || rng.next_value()))
        }
        _ => None,
    }
}

/// Without `lossgen` the branch does not exist.
#[cfg(not(feature = "lossgen"))]
const fn sim_loss_draw(_sim: &mut SimLoss, _rng: &mut GlibcRand) -> Option<i32> {
    None
}

/// `ENABLE_OSCE_TRAINING_DATA` settings of `new_random_setting`.
#[cfg(feature = "osce-training-data")]
mod training {
    pub const COMPLEXITY_MIN: i32 = 0;
    pub const COMPLEXITY_MAX: i32 = 10;
    pub const PACKET_LOSS_PERC_MIN: i32 = 0;
    pub const PACKET_LOSS_PERC_MAX: i32 = 50;
    pub const PACKET_LOSS_PERC_STEP: i32 = 5;
    pub const CBR_BITRATE_LIMIT: i32 = 80000;
    pub const NUM_BITRATES: i32 = 102;
    #[rustfmt::skip]
    pub static BITRATES: [i32; NUM_BITRATES as usize] = [
         6000,  6060,  6120,  6180,  6240,  6300,  6360,  6420,  6480,
         6525,  6561,  6598,  6634,  6670,  6707,  6743,  6780,  6816,
         6853,  6889,  6926,  6962,  6999,  7042,  7085,  7128,  7171,
         7215,  7258,  7301,  7344,  7388,  7431,  7474,  7512,  7541,
         7570,  7599,  7628,  7657,  7686,  7715,  7744,  7773,  7802,
         7831,  7860,  7889,  7918,  7947,  7976,  8013,  8096,  8179,
         8262,  8344,  8427,  8511,  8605,  8699,  8792,  8886,  8980,
         9100,  9227,  9354,  9480,  9561,  9634,  9706,  9779,  9851,
         9924,  9996, 10161, 10330, 10499, 10698, 10898, 11124, 11378,
        11575, 11719, 11862, 12014, 12345, 12751, 13195, 13561, 13795,
        14069, 14671, 15403, 15790, 16371, 17399, 17968, 19382, 20468,
        22000, 32000, 64000,
    ];
}

/// `randint(min, max, step)` (`ENABLE_OSCE_TRAINING_DATA`): `rand()` scaled to `[min, max]` in
/// steps of `step`, with C's double arithmetic.
#[cfg(feature = "osce-training-data")]
fn randint(rng: &mut GlibcRand, min: i32, max: i32, step: i32) -> i32 {
    // RAND_MAX + 1. (glibc RAND_MAX = 2^31 - 1)
    let r = f64::from(rng.next_value()) / (2_147_483_647.0 + 1.0);
    // (int) ((max + 1 - min) * r / step) * step + min
    ((f64::from(max + 1 - min) * r / f64::from(step)) as i32) * step + min
}

/// `new_random_setting` (`ENABLE_OSCE_TRAINING_DATA`): random bitrate, complexity, loss
/// percentage and VBR, announced on stdout.
#[cfg(feature = "osce-training-data")]
fn new_random_setting(
    enc: &mut Encoder,
    rng: &mut GlibcRand,
    stdout: &mut dyn Write,
) -> io::Result<()> {
    use training::*;
    let bitrate_bps = BITRATES[randint(rng, 0, NUM_BITRATES - 1, 1) as usize];
    let complexity = randint(rng, COMPLEXITY_MIN, COMPLEXITY_MAX, 1);
    let packet_loss_perc = randint(
        rng,
        PACKET_LOSS_PERC_MIN,
        PACKET_LOSS_PERC_MAX,
        PACKET_LOSS_PERC_STEP,
    );
    let use_vbr = if bitrate_bps < CBR_BITRATE_LIMIT {
        1
    } else {
        randint(rng, 0, 1, 1)
    };
    writeln!(
        stdout,
        "changing settings to {bitrate_bps}\t{complexity}\t{packet_loss_perc}\t{use_vbr}"
    )?;
    ctl_ignored(enc.ctl_set(OPUS_SET_BITRATE_REQUEST, bitrate_bps));
    ctl_ignored(enc.ctl_set(OPUS_SET_COMPLEXITY_REQUEST, complexity));
    ctl_ignored(enc.ctl_set(OPUS_SET_PACKET_LOSS_PERC_REQUEST, packet_loss_perc));
    ctl_ignored(enc.ctl_set(OPUS_SET_VBR_REQUEST, use_vbr));
    Ok(())
}

/// The `-lossfile` stream, read with C `fscanf(f, "%d", &lost)` semantics.
#[derive(Debug)]
struct LossFile {
    data: Vec<u8>,
    pos: usize,
}

impl LossFile {
    /// `fscanf(f, "%d", &v) == 1 ? Some(v) : None`: skips white space, then converts an optional
    /// sign and decimal digits (like `strtol`, saturating to `long`, then truncated to `int`).
    /// On a matching failure the offending character stays unread, so every later call fails
    /// too, as in C.
    fn scan_int(&mut self) -> Option<i32> {
        let b = &self.data;
        let mut i = self.pos;
        while i < b.len() && c_isspace(b[i]) {
            i += 1;
        }
        let start = i;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let digits = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        self.pos = i;
        if i == digits {
            return None;
        }
        // The token is ASCII (sign + digits), so it is valid UTF-8.
        let token: String = b[start..i].iter().map(|&c| char::from(c)).collect();
        Some(c_atoi(&token))
    }
}

/// `print_usage`.
fn print_usage(stderr: &mut dyn Write, args: &[String]) -> io::Result<()> {
    let a0 = argv0(args);
    writeln!(
        stderr,
        "Usage: {a0} [-e] <application> <sampling rate (Hz)> <channels (1/2)> <bits per second>  [options] <input> <output>"
    )?;
    writeln!(
        stderr,
        "       {a0} -d <sampling rate (Hz)> <channels (1/2)> [options] <input> <output>\n"
    )?;
    writeln!(
        stderr,
        "application: voip | audio | restricted-lowdelay | restricted-silk | restricted-celt"
    )?;
    writeln!(stderr, "options:")?;
    writeln!(
        stderr,
        "-e                   : only runs the encoder (output the bit-stream)"
    )?;
    writeln!(
        stderr,
        "-d                   : only runs the decoder (reads the bit-stream as input)"
    )?;
    writeln!(
        stderr,
        "-cbr                 : enable constant bitrate; default: variable bitrate"
    )?;
    writeln!(
        stderr,
        "-cvbr                : enable constrained variable bitrate; default: unconstrained"
    )?;
    writeln!(
        stderr,
        "-delayed-decision    : use look-ahead for speech/music detection (experts only); default: disabled"
    )?;
    writeln!(
        stderr,
        "-bandwidth <NB|MB|WB|SWB|FB> : audio bandwidth (from narrowband to fullband); default: sampling rate"
    )?;
    writeln!(
        stderr,
        "-framesize <2.5|5|10|20|40|60|80|100|120> : frame size in ms; default: 20 "
    )?;
    writeln!(
        stderr,
        "-max_payload <bytes> : maximum payload size in bytes, default: 1024"
    )?;
    writeln!(
        stderr,
        "-complexity <comp>   : encoder complexity, 0 (lowest) ... 10 (highest); default: 10"
    )?;
    writeln!(
        stderr,
        "-dec_complexity <comp> : decoder complexity, 0 (lowest) ... 10 (highest); default: 0"
    )?;
    writeln!(stderr, "-inbandfec           : enable SILK inband FEC")?;
    writeln!(
        stderr,
        "-forcemono           : force mono encoding, even for stereo input"
    )?;
    writeln!(stderr, "-dtx                 : enable SILK DTX")?;
    writeln!(
        stderr,
        "-loss <perc>         : optimize for loss percentage and simulate packet loss, in percent (0-100); default: 0"
    )?;
    #[cfg(feature = "lossgen")]
    writeln!(
        stderr,
        "-sim_loss <perc>     : simulate realistic (bursty) packet loss from percentage, using generative model"
    )?;
    writeln!(
        stderr,
        "-lossfile <file>     : simulate packet loss, reading loss from file"
    )?;
    writeln!(
        stderr,
        "-dred <frames>       : add Deep REDundancy (in units of 10-ms frames)"
    )?;
    writeln!(
        stderr,
        "-enc_loss            : Apply loss on the encoder side (store empty packets)"
    )?;
    #[cfg(feature = "osce")]
    writeln!(
        stderr,
        "-enable_osce_bwe     : enable OSCE bandwidth extension for wideband signals (48 kHz sampling rate only), raises dec_complexity to 4"
    )?;
    #[cfg(feature = "qext")]
    writeln!(stderr, "-qext                : enable QEXT")?;
    Ok(())
}

/// `int_to_char`: big-endian bytes of a 32-bit value.
const fn int_to_char(i: u32) -> [u8; 4] {
    i.to_be_bytes()
}

/// `char_to_int`: big-endian 32-bit value.
const fn char_to_int(ch: [u8; 4]) -> u32 {
    u32::from_be_bytes(ch)
}

/// `opus_demo` ignores the return value of every `opus_encoder_ctl`/`opus_decoder_ctl` call: a
/// rejected setting (e.g. `-complexity 11`) leaves the previous value in place and the run
/// continues. This makes that C behaviour explicit at each call site.
const fn ctl_ignored(result: opusorus::Result<()>) {
    match result {
        Ok(()) | Err(_) => {}
    }
}

/// A libopus function result in C's `int` convention (value, or negative error code).
const fn c_ret(result: opusorus::Result<i32>) -> i32 {
    match result {
        Ok(v) => v,
        Err(e) => e.code(),
    }
}

/// `check_encoder_option` / `check_decoder_option`: prints the message and reports whether the
/// option is misplaced (`goto failure` in C).
fn misplaced_option(
    stderr: &mut dyn Write,
    wrong_mode: bool,
    opt: &str,
    what: &str,
) -> io::Result<bool> {
    if wrong_mode {
        writeln!(stderr, "option {opt} is only for {what}")?;
    }
    Ok(wrong_mode)
}

/// `fread(buf, 1, buf.len(), f)`: reads until `buf` is full or end of file; returns the number
/// of bytes read.
fn fread(r: &mut dyn Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(n)
}

/// Writes to the output file; `Ok(false)` after printing C's `Error writing.` message.
fn fwrite(stderr: &mut dyn Write, fout: &mut dyn Write, bytes: &[u8]) -> io::Result<bool> {
    if fout.write_all(bytes).is_err() {
        // C only checks the item count fwrite returns and reports the failure this way.
        writeln!(stderr, "Error writing.")?;
        return Ok(false);
    }
    Ok(true)
}

/// Port of `src/opus_demo.c:main`: command line `args` (including `argv[0]`), `printf` output
/// to `stdout`, `fprintf(stderr, ...)` output to `stderr`; returns the process exit status
/// (`EXIT_SUCCESS`, `EXIT_FAILURE`, or, as in C after a DRED parse attempt, that call's return
/// value; the caller truncates it to 8 bits like the OS does).
///
/// # Errors
/// I/O errors writing to `stdout`/`stderr`, reading an opened input file or flushing the
/// output file.
pub fn opus_demo_main(
    args: &[String],
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> io::Result<i32> {
    let weights = if DEEP_PLC && !WEIGHTS_EMBEDDED {
        // USE_WEIGHTS_FILE: `load_blob("weights_blob.bin")`.
        match std::fs::read(WEIGHTS_FILE) {
            Ok(b) => Some(b),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        }
    } else {
        None
    };
    opus_demo_main_with_weights(args, stdout, stderr, weights.as_deref())
}

/// [`opus_demo_main`] with the DNN weight blob given by the caller instead of read from
/// [`WEIGHTS_FILE`]: `Some(blob)` is set with `OPUS_SET_DNN_BLOB` on the encoder, decoder and
/// DRED decoder (errors ignored, as in C); `None` leaves them with the compiled-in models
/// (feature `dnn-weights-embedded`) or without models. Ignored without DNN features.
///
/// # Errors
/// As [`opus_demo_main`].
pub fn opus_demo_main_with_weights(
    args: &[String],
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
    weights: Option<&[u8]>,
) -> io::Result<i32> {
    let mut fout = None;
    let ret = run(args, stdout, stderr, &mut fout, weights);
    // The OSCE training files are closed at process exit in C: close them (and report their
    // first write error) on every path, so that the next run starts new files.
    #[cfg(feature = "osce-training-data")]
    let training = opusorus::osce_training_data::close_all();
    let ret = ret?;
    // fclose(fout)
    if let Some(mut f) = fout {
        f.flush()?;
    }
    #[cfg(feature = "osce-training-data")]
    training?;
    Ok(ret)
}

/// The body of `main`; `fout` receives the output file once opened so that the caller closes
/// (flushes) it on every exit path, like C's `failure:` label.
fn run(
    args: &[String],
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
    fout_slot: &mut Option<BufWriter<File>>,
    weights: Option<&[u8]>,
) -> io::Result<i32> {
    let argc = args.len();
    let mut ret = EXIT_FAILURE;

    if argc < 5 {
        print_usage(stderr, args)?;
        return Ok(ret);
    }

    let mut tot_in: u64 = 0;
    let mut tot_out: u64 = 0;
    writeln!(stderr, "{}", opus_get_version_string())?;

    let mut a = 1usize;
    let mut encode_only = false;
    let mut decode_only = false;
    if args[a] == "-e" {
        encode_only = true;
        a += 1;
    } else if args[a] == "-d" {
        decode_only = true;
        a += 1;
    }
    if !decode_only && argc < 7 {
        print_usage(stderr, args)?;
        return Ok(ret);
    }

    let mut application = OPUS_APPLICATION_AUDIO;
    if !decode_only {
        match args[a].as_str() {
            "voip" => application = OPUS_APPLICATION_VOIP,
            "restricted-lowdelay" => application = OPUS_APPLICATION_RESTRICTED_LOWDELAY,
            "restricted-silk" => application = OPUS_APPLICATION_RESTRICTED_SILK,
            "restricted-celt" => application = OPUS_APPLICATION_RESTRICTED_CELT,
            "audio" => {}
            other => {
                writeln!(stderr, "unknown application: {other}")?;
                print_usage(stderr, args)?;
                return Ok(ret);
            }
        }
        a += 1;
    }
    // (opus_int32)atol(argv[args]): on LP64 `atol` and `atoi` both go through strtol.
    let sampling_rate = c_atoi(&args[a]);
    a += 1;

    let rate_ok = matches!(sampling_rate, 8000 | 12000 | 16000 | 24000 | 48000)
        || (cfg!(feature = "qext") && sampling_rate == 96000);
    if !rate_ok {
        if cfg!(feature = "qext") {
            writeln!(
                stderr,
                "Supported sampling rates are 8000, 12000, 16000, 24000, 48000 and 96000."
            )?;
        } else {
            writeln!(
                stderr,
                "Supported sampling rates are 8000, 12000, 16000, 24000 and 48000."
            )?;
        }
        return Ok(ret);
    }
    let mut frame_size = sampling_rate / 50;

    let channels = c_atoi(&args[a]);
    a += 1;
    if !(1..=2).contains(&channels) {
        writeln!(stderr, "Opus_demo supports only 1 or 2 channels.")?;
        return Ok(ret);
    }

    let mut bitrate_bps = 0i32;
    if !decode_only {
        bitrate_bps = c_atoi(&args[a]);
        a += 1;
    }

    // defaults:
    let mut use_vbr = 1;
    let mut max_payload_bytes = MAX_PACKET;
    let mut complexity = 10;
    let mut dec_complexity = 0;
    let mut use_inbandfec = 0i32;
    let mut forcechannels = OPUS_AUTO;
    let mut use_dtx = 0;
    let mut packet_loss_perc = 0;
    let mut cvbr = 0;
    let mut format = Format::S16;
    let mut bandwidth = OPUS_AUTO;
    let mut delayed_decision = false;
    let mut packet_loss_file: Option<LossFile> = None;
    let mut dred_duration = 0;
    let mut encoder_loss = false;
    let mut sweep_bps = 0i32;
    let mut random_framesize = false;
    let mut sweep_max = 0i32;
    let mut random_fec = false;
    let mut mode_list: Option<&'static [ModeEntry]> = None;
    let mut ignore_extensions = false;
    #[cfg(feature = "qext")]
    let mut enable_qext = 0;
    // `lossgen_perc` and `lossgen` (-sim_loss).
    let mut sim_loss = SimLoss::default();
    #[cfg(feature = "osce-training-data")]
    let mut silk_random_switching = 0i32;
    #[cfg(feature = "osce-training-data")]
    let mut silk_frame_counter = 0i32;
    #[cfg(feature = "osce")]
    let mut enable_osce_bwe = false;

    while a < argc - 2 {
        // process command line options
        let opt = args[a].as_str();
        // `argv[args + 1]` exists: the loop condition leaves the two file names after it.
        let next = || args[a + 1].as_str();
        match opt {
            "-cbr" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                use_vbr = 0;
                a += 1;
            }
            "-bandwidth" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                bandwidth = match next() {
                    "NB" => OPUS_BANDWIDTH_NARROWBAND,
                    "MB" => OPUS_BANDWIDTH_MEDIUMBAND,
                    "WB" => OPUS_BANDWIDTH_WIDEBAND,
                    "SWB" => OPUS_BANDWIDTH_SUPERWIDEBAND,
                    "FB" => OPUS_BANDWIDTH_FULLBAND,
                    other => {
                        writeln!(
                            stderr,
                            "Unknown bandwidth {other}. Supported are NB, MB, WB, SWB, FB."
                        )?;
                        return Ok(ret);
                    }
                };
                a += 2;
            }
            "-framesize" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                frame_size = match next() {
                    "2.5" => sampling_rate / 400,
                    "5" => sampling_rate / 200,
                    "10" => sampling_rate / 100,
                    "20" => sampling_rate / 50,
                    "40" => sampling_rate / 25,
                    "60" => 3 * sampling_rate / 50,
                    "80" => 4 * sampling_rate / 50,
                    "100" => 5 * sampling_rate / 50,
                    "120" => 6 * sampling_rate / 50,
                    other => {
                        writeln!(
                            stderr,
                            "Unsupported frame size: {other} ms. Supported are 2.5, 5, 10, 20, 40, 60, 80, 100, 120."
                        )?;
                        return Ok(ret);
                    }
                };
                a += 2;
            }
            "-max_payload" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                max_payload_bytes = c_atoi(next());
                a += 2;
            }
            "-complexity" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                complexity = c_atoi(next());
                a += 2;
            }
            "-dec_complexity" => {
                if misplaced_option(stderr, encode_only, opt, "decoding")? {
                    return Ok(ret);
                }
                dec_complexity = c_atoi(next());
                a += 2;
            }
            "-16" => {
                format = Format::S16;
                a += 1;
            }
            "-24" => {
                format = Format::S24;
                a += 1;
            }
            "-f32" => {
                format = Format::F32;
                a += 1;
            }
            "-inbandfec" => {
                use_inbandfec = 1;
                a += 1;
            }
            "-forcemono" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                forcechannels = 1;
                a += 1;
            }
            "-cvbr" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                cvbr = 1;
                a += 1;
            }
            "-delayed-decision" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                delayed_decision = true;
                a += 1;
            }
            "-dtx" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                use_dtx = 1;
                a += 1;
            }
            "-loss" => {
                packet_loss_perc = c_atoi(next());
                a += 2;
            }
            #[cfg(feature = "lossgen")]
            "-sim_loss" => {
                // `lossgen_perc = atof(...)` (double to float) and `lossgen_init(&lossgen)`.
                sim_loss = Some((c_atof(next()) as f32, LossGenState::lossgen_init()));
                a += 2;
            }
            "-lossfile" => {
                let name = next();
                let Ok(mut f) = File::open(name) else {
                    // C reports the failed fopen without the reason, then exit(1).
                    writeln!(stderr, "failed to open loss file {name}")?;
                    return Ok(1);
                };
                let mut data = Vec::new();
                f.read_to_end(&mut data)?;
                packet_loss_file = Some(LossFile { data, pos: 0 });
                a += 2;
            }
            "-dred" => {
                dred_duration = c_atoi(next());
                a += 2;
            }
            "-enc_loss" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                encoder_loss = true;
                a += 1;
            }
            "-sweep" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                sweep_bps = c_atoi(next());
                a += 2;
            }
            "-random_framesize" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                random_framesize = true;
                a += 1;
            }
            "-sweep_max" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                sweep_max = c_atoi(next());
                a += 2;
            }
            "-random_fec" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                random_fec = true;
                a += 1;
            }
            "-silk8k_test"
            | "-silk12k_test"
            | "-silk16k_test"
            | "-silk_bw_switch_test"
            | "-hybrid24k_test"
            | "-hybrid48k_test"
            | "-celt_test"
            | "-celt_hq_test" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                mode_list = Some(match opt {
                    "-silk8k_test" => &SILK8_TEST,
                    "-silk12k_test" => &SILK12_TEST,
                    "-silk16k_test" => &SILK16_TEST,
                    "-silk_bw_switch_test" => &SILK_BW_SWITCH_TEST,
                    "-hybrid24k_test" => &HYBRID24_TEST,
                    "-hybrid48k_test" => &HYBRID48_TEST,
                    "-celt_test" => &CELT_TEST,
                    _ => &CELT_HQ_TEST,
                });
                a += 1;
            }
            "-ignore_extensions" => {
                if misplaced_option(stderr, encode_only, opt, "decoding")? {
                    return Ok(ret);
                }
                ignore_extensions = true;
                a += 1;
            }
            #[cfg(feature = "qext")]
            "-qext" => {
                if misplaced_option(stderr, decode_only, opt, "encoding")? {
                    return Ok(ret);
                }
                enable_qext = 1;
                a += 1;
            }
            #[cfg(feature = "osce-training-data")]
            "-silk_random_switching" => {
                silk_random_switching = c_atoi(next());
                writeln!(
                    stdout,
                    "switching encoding parameters every {silk_random_switching}th frame"
                )?;
                a += 2;
            }
            #[cfg(feature = "osce")]
            "-enable_osce_bwe" => {
                enable_osce_bwe = true;
                a += 1;
            }
            other => {
                // C prints this line with printf (stdout), the usage on stderr.
                write!(stdout, "Error: unrecognized setting: {other}\n\n")?;
                print_usage(stderr, args)?;
                return Ok(ret);
            }
        }
    }

    let mut sweep_min = 0i32;
    if sweep_max != 0 {
        sweep_min = bitrate_bps;
    }

    if !(0..=MAX_PACKET).contains(&max_payload_bytes) {
        writeln!(
            stderr,
            "max_payload_bytes must be between 0 and {MAX_PACKET}"
        )?;
        return Ok(ret);
    }

    let in_file = &args[argc - 2];
    let Ok(fin_file) = File::open(in_file) else {
        // C reports a failed fopen with the path only.
        writeln!(stderr, "Could not open input file {in_file}")?;
        return Ok(ret);
    };
    let mut mode_switch_time = 48000i32;
    let mut nb_modes_in_list = 0i32;
    if let Some(list) = mode_list {
        nb_modes_in_list = list.len() as i32;
        let sample_size = format.size() as i32;
        // `int size = ftell(fin)` after seeking to the end: the file length truncated to int.
        let size = fin_file.metadata()?.len() as i32;
        writeln!(stderr, "File size is {size} bytes")?;
        mode_switch_time = size / sample_size / channels / nb_modes_in_list;
        writeln!(stderr, "Switching mode every {mode_switch_time} samples")?;
    }
    let mut fin = BufReader::new(fin_file);

    let out_file = &args[argc - 1];
    let Ok(f) = File::create(out_file) else {
        writeln!(stderr, "Could not open output file {out_file}")?;
        return Ok(ret);
    };
    let fout = fout_slot.insert(BufWriter::new(f));

    let mut skip = 0i32;
    let mut variable_duration = OPUS_FRAMESIZE_ARG;
    #[cfg(feature = "osce-training-data")]
    let mut reseed_rand = false;
    let mut enc: Option<Box<Encoder>> = None;
    if !decode_only {
        let mut e = match Encoder::new_raw(sampling_rate, channels, application) {
            Ok(e) => Box::new(e),
            Err(err) => {
                writeln!(
                    stderr,
                    "Cannot create encoder: {}",
                    opus_strerror(err.code())
                )?;
                return Ok(ret);
            }
        };
        ctl_ignored(e.ctl_set(OPUS_SET_BITRATE_REQUEST, bitrate_bps));
        ctl_ignored(e.ctl_set(OPUS_SET_BANDWIDTH_REQUEST, bandwidth));
        ctl_ignored(e.ctl_set(OPUS_SET_VBR_REQUEST, use_vbr));
        ctl_ignored(e.ctl_set(OPUS_SET_VBR_CONSTRAINT_REQUEST, cvbr));
        ctl_ignored(e.ctl_set(OPUS_SET_COMPLEXITY_REQUEST, complexity));
        ctl_ignored(e.ctl_set(OPUS_SET_INBAND_FEC_REQUEST, use_inbandfec));
        ctl_ignored(e.ctl_set(OPUS_SET_FORCE_CHANNELS_REQUEST, forcechannels));
        ctl_ignored(e.ctl_set(OPUS_SET_DTX_REQUEST, use_dtx));
        ctl_ignored(e.ctl_set(OPUS_SET_PACKET_LOSS_PERC_REQUEST, packet_loss_perc));

        skip = e.lookahead() as i32;
        let lsb_depth = if format == Format::S16 { 16 } else { 24 };
        ctl_ignored(e.ctl_set(OPUS_SET_LSB_DEPTH_REQUEST, lsb_depth));
        ctl_ignored(e.ctl_set(OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, variable_duration));
        if dred_duration > 0 {
            ctl_ignored(e.ctl_set(OPUS_SET_DRED_DURATION_REQUEST, dred_duration));
        }
        #[cfg(feature = "osce-training-data")]
        {
            ctl_ignored(e.ctl_set(OPUS_SET_FORCE_MODE_REQUEST, MODE_SILK_ONLY));
            // srand(0), applied where `rng` is created below (nothing draws before it).
            reseed_rand = true;
        }
        #[cfg(feature = "qext")]
        ctl_ignored(e.ctl_set(OPUS_SET_QEXT_REQUEST, enable_qext));
        enc = Some(e);
    }
    let mut dec: Option<Box<Decoder>> = None;
    if !encode_only {
        let mut d = match Decoder::new(sampling_rate, channels) {
            Ok(d) => Box::new(d),
            Err(err) => {
                writeln!(
                    stderr,
                    "Cannot create decoder: {}",
                    opus_strerror(err.code())
                )?;
                return Ok(ret);
            }
        };
        #[cfg(feature = "osce")]
        if enable_osce_bwe {
            ctl_ignored(d.set_osce_bwe(1));
            if dec_complexity < 4 {
                dec_complexity = 4;
            }
        }
        ctl_ignored(d.set_complexity(dec_complexity));
        d.set_ignore_extensions(ignore_extensions);
        dec = Some(d);
    }
    let bandwidth_string = match bandwidth {
        OPUS_BANDWIDTH_NARROWBAND => "narrowband",
        OPUS_BANDWIDTH_MEDIUMBAND => "mediumband",
        OPUS_BANDWIDTH_WIDEBAND => "wideband",
        OPUS_BANDWIDTH_SUPERWIDEBAND => "superwideband",
        OPUS_BANDWIDTH_FULLBAND => "fullband",
        OPUS_AUTO => "auto bandwidth",
        _ => "unknown",
    };

    if decode_only {
        writeln!(
            stderr,
            "Decoding with {sampling_rate} Hz output ({channels} channels)"
        )?;
    } else {
        writeln!(
            stderr,
            "Encoding {sampling_rate} Hz input at {} kb/s in {bandwidth_string} with {frame_size}-sample frames.",
            c_fmt_f(f64::from(bitrate_bps) * 0.001, 3)
        )?;
    }

    let ch = channels as usize;
    let max_frame_size = MAX_SAMPLING_RATE * 2;
    let mut in_buf = vec![0i32; max_frame_size as usize * ch];
    let mut out_buf = vec![0i32; max_frame_size as usize * ch];
    // We need to allocate for 16-bit PCM data, but we store it as unsigned char.
    let mut fbytes = vec![0u8; max_frame_size as usize * ch * 4];
    let mut data = vec![0u8; max_payload_bytes as usize];
    if delayed_decision {
        variable_duration = if frame_size == sampling_rate / 400 {
            OPUS_FRAMESIZE_2_5_MS
        } else if frame_size == sampling_rate / 200 {
            OPUS_FRAMESIZE_5_MS
        } else if frame_size == sampling_rate / 100 {
            OPUS_FRAMESIZE_10_MS
        } else if frame_size == sampling_rate / 50 {
            OPUS_FRAMESIZE_20_MS
        } else if frame_size == sampling_rate / 25 {
            OPUS_FRAMESIZE_40_MS
        } else if frame_size == 3 * sampling_rate / 50 {
            OPUS_FRAMESIZE_60_MS
        } else if frame_size == 4 * sampling_rate / 50 {
            OPUS_FRAMESIZE_80_MS
        } else if frame_size == 5 * sampling_rate / 50 {
            OPUS_FRAMESIZE_100_MS
        } else {
            OPUS_FRAMESIZE_120_MS
        };
        if let Some(e) = enc.as_mut() {
            ctl_ignored(e.ctl_set(OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, variable_duration));
        }
        frame_size = 2 * sampling_rate;
    }
    // opus_dred_decoder_create() / opus_dred_alloc() (compiled-in weights: bound here).
    #[cfg(feature = "dred")]
    let mut dred_dec = DredDecoder::new();
    #[cfg(feature = "dred")]
    let mut dred = Dred::new();
    // USE_WEIGHTS_FILE: OPUS_SET_DNN_BLOB on the encoder, decoder and DRED decoder.
    #[cfg(feature = "deep-plc")]
    if let Some(blob) = weights {
        #[cfg(feature = "dred")]
        if let Some(e) = enc.as_mut() {
            ctl_ignored(e.set_dnn_blob(blob));
        }
        if let Some(d) = dec.as_mut() {
            ctl_ignored(d.set_dnn_blob(blob));
        }
        #[cfg(feature = "dred")]
        ctl_ignored(dred_dec.set_dnn_blob(blob));
    }
    #[cfg(not(feature = "deep-plc"))]
    let _ = weights;

    let mut rng = GlibcRand::default();
    #[cfg(feature = "osce-training-data")]
    if reseed_rand {
        rng = GlibcRand::new(0);
    }
    let mut stop = false;
    let mut count = 0i32;
    let mut count_act = 0i32;
    let mut bits = 0.0f64;
    let mut bits_max = 0.0f64;
    let mut bits_act = 0.0f64;
    let mut bits2 = 0.0f64;
    let mut tot_samples = 0.0f64;
    let mut lost_prev = 1i32;
    let mut dec_final_range = 0u32;
    let mut delayed_celt = false;
    let mut newsize = 0i32;
    let mut curr_mode = 0usize;
    let mut curr_mode_count = 0i32;
    let mut nb_encoded = 0i32;
    let mut remaining = 0i32;
    let mut lost_count = 0i32;

    while !stop {
        // Set on every iteration before use (function-scope variables in C).
        let mut len: i32;
        let mut enc_final_range: u32;
        let mut lost: i32;
        if delayed_celt {
            frame_size = newsize;
            delayed_celt = false;
        } else if random_framesize && rng.next_value() % 20 == 0 {
            newsize = match rng.next_value() % 6 {
                0 => sampling_rate / 400,
                1 => sampling_rate / 200,
                2 => sampling_rate / 100,
                3 => sampling_rate / 50,
                4 => sampling_rate / 25,
                _ => 3 * sampling_rate / 50,
            };
            while newsize < sampling_rate / 25
                && bitrate_bps - sweep_bps.abs() <= 3 * 12 * sampling_rate / newsize
            {
                newsize *= 2;
            }
            if newsize < sampling_rate / 100 && frame_size >= sampling_rate / 100 {
                if let Some(e) = enc.as_mut() {
                    ctl_ignored(e.ctl_set(OPUS_SET_FORCE_MODE_REQUEST, MODE_CELT_ONLY));
                }
                delayed_celt = true;
            } else {
                frame_size = newsize;
            }
        }
        if random_fec && rng.next_value() % 30 == 0 {
            // OPUS_SET_INBAND_FEC(x) expands to `__opus_check_int(x)`, which evaluates its
            // argument twice: C draws `rand()%4==0` twice and passes the second value.
            let _type_check_draw = rng.next_value();
            let fec = i32::from(rng.next_value() % 4 == 0);
            if let Some(e) = enc.as_mut() {
                ctl_ignored(e.ctl_set(OPUS_SET_INBAND_FEC_REQUEST, fec));
            }
        }
        if let Some(e) = enc.as_mut() {
            // Encoder path (the C `else` of `if (decode_only)`).
            if let Some(list) = mode_list {
                let m = list[curr_mode];
                ctl_ignored(e.ctl_set(OPUS_SET_BANDWIDTH_REQUEST, m[1]));
                ctl_ignored(e.ctl_set(OPUS_SET_FORCE_MODE_REQUEST, m[0]));
                ctl_ignored(e.ctl_set(OPUS_SET_FORCE_CHANNELS_REQUEST, m[3]));
                frame_size = m[2] * sampling_rate / 48000;
            }
            #[cfg(feature = "osce-training-data")]
            if silk_random_switching != 0 {
                silk_frame_counter += 1;
                if silk_frame_counter % silk_random_switching == 0 {
                    new_random_setting(e, &mut rng, stdout)?;
                }
            }
            let fsz = format.size();
            // A negative count (frame size shrunk below the carried-over samples) is undefined
            // behaviour in C; nothing is read here.
            let want = (frame_size - remaining).max(0) as usize;
            let nbytes = fread(&mut fin, &mut fbytes[..want * fsz * ch])?;
            let curr_read = (nbytes / (fsz * ch)) as i32;
            tot_in += curr_read as u64;
            let n = frame_size as usize * ch;
            match format {
                Format::S16 => {
                    for (d, b) in in_buf[..n].iter_mut().zip(fbytes.as_chunks::<2>().0) {
                        let mut s = i32::from(b[1]) << 8 | i32::from(b[0]);
                        s = ((s & 0xFFFF) ^ 0x8000) - 0x8000;
                        *d = s * 256;
                    }
                }
                Format::S24 => {
                    for (d, b) in in_buf[..n].iter_mut().zip(fbytes.as_chunks::<3>().0) {
                        let mut s = i32::from(b[2]) << 16 | i32::from(b[1]) << 8 | i32::from(b[0]);
                        s = ((s & 0xFFFFFF) ^ 0x800000) - 0x800000;
                        *d = s;
                    }
                }
                Format::F32 => {
                    for (d, b) in in_buf[..n].iter_mut().zip(fbytes.as_chunks::<4>().0) {
                        let f = f32::from_le_bytes(*b);
                        // (int)floor(.5 + s.f*8388608): float multiply, double add.
                        *d = (0.5 + f64::from(f * 8388608.0)).floor() as i32;
                    }
                }
            }
            if curr_read + remaining < frame_size {
                let from = ((curr_read + remaining) * channels) as usize;
                in_buf[from..n].fill(0);
                if encode_only || decode_only {
                    stop = true;
                }
            }
            len = match e.encode24(
                &in_buf,
                frame_size as usize,
                &mut data[..max_payload_bytes as usize],
            ) {
                Ok(l) => l as i32,
                Err(err) => {
                    writeln!(stderr, "opus_encode() returned {}", err.code())?;
                    return Ok(ret);
                }
            };
            nb_encoded = c_ret(get_samples_per_frame(&data, sampling_rate))
                * c_ret(get_nb_frames(&data[..len as usize]));
            remaining = frame_size - nb_encoded;
            // (A negative nb_encoded cannot come from a successful encode; C would read out of
            // bounds.)
            if remaining > 0 && nb_encoded >= 0 {
                let r = remaining as usize * ch;
                let src = nb_encoded as usize * ch;
                in_buf.copy_within(src..src + r, 0);
            }
            if sweep_bps != 0 {
                bitrate_bps += sweep_bps;
                if sweep_max != 0 && (bitrate_bps > sweep_max || bitrate_bps < sweep_min) {
                    sweep_bps = -sweep_bps;
                }
                // safety
                if bitrate_bps < 1000 {
                    bitrate_bps = 1000;
                }
                ctl_ignored(e.ctl_set(OPUS_SET_BITRATE_REQUEST, bitrate_bps));
            }
            enc_final_range = e.final_range();
            curr_mode_count += frame_size;
            if curr_mode_count > mode_switch_time && (curr_mode as i32) < nb_modes_in_list - 1 {
                curr_mode += 1;
                curr_mode_count = 0;
            }
        } else {
            // decode_only: read one packet record.
            let mut ch4 = [0u8; 4];
            if fread(&mut fin, &mut ch4)? != 4 {
                break;
            }
            len = char_to_int(ch4) as i32;
            if len > max_payload_bytes || len < 0 {
                writeln!(stderr, "Invalid payload length: {len}")?;
                break;
            }
            if fread(&mut fin, &mut ch4)? != 4 {
                break;
            }
            enc_final_range = char_to_int(ch4);
            let num_read = fread(&mut fin, &mut data[..len as usize])?;
            if num_read != len as usize {
                writeln!(
                    stderr,
                    "Ran out of input, expecting {len} bytes got {num_read}"
                )?;
                break;
            }
        }

        if encode_only && !encoder_loss {
            lost = 0;
        } else if let Some(lf) = packet_loss_file.as_mut() {
            // `if (fscanf(packet_loss_file, "%d", &lost) != 1) lost = 0;`
            match lf.scan_int() {
                Some(v) => lost = v,
                None => lost = 0,
            }
        } else if let Some(l) = sim_loss_draw(&mut sim_loss, &mut rng) {
            lost = l;
        } else {
            lost = i32::from(packet_loss_perc > 0 && rng.next_value() % 100 < packet_loss_perc);
        }
        if let Some(d) = dec.as_mut() {
            if len == 0 {
                lost = 1;
            }
            let mut run_decoder = if lost != 0 {
                lost_count += 1;
                0
            } else {
                1
            };
            if run_decoder != 0 {
                run_decoder += lost_count;
            }
            let packet = &data[..len as usize];
            #[cfg(feature = "dred")]
            let mut dred_input = 0i32;
            if lost == 0 && lost_count > 0 {
                #[cfg(feature = "dred")]
                {
                    let output_samples = d.last_packet_duration() as i32;
                    dred_input = lost_count * output_samples;
                    // Only decode the amount we need to fill in the gap.
                    ret = match dred_dec.parse(
                        &mut dred,
                        packet,
                        sampling_rate.min(dred_input.max(0)),
                        sampling_rate,
                        false,
                    ) {
                        Ok((offset, _dred_end)) => offset,
                        Err(e) => e.code(),
                    };
                    dred_input = if ret > 0 { ret } else { 0 };
                }
                // Without DRED support opus_dred_parse returns OPUS_UNIMPLEMENTED, so dred_input
                // is 0 and lost frames are concealed with regular PLC below.
                #[cfg(not(feature = "dred"))]
                {
                    ret = Error::Unimplemented.code();
                }
            }
            // FIXME (C): figure out how to trigger the decoder when the last packet of the file
            // is lost.
            for fr in 0..run_decoder {
                let mut output_samples;
                // opus_packet_has_lbrr returns a negative error code (true in C) for a packet
                // it cannot parse; the FEC decode then reports the error.
                if fr == lost_count - 1 && !matches!(has_lbrr(packet), Ok(false)) {
                    output_samples = d.last_packet_duration() as i32;
                    output_samples =
                        c_ret(d.opus_decode24(Some(packet), &mut out_buf, output_samples, 1));
                } else if fr < lost_count {
                    output_samples = d.last_packet_duration() as i32;
                    #[cfg(feature = "dred")]
                    if dred_input > 0 {
                        output_samples = c_ret(d.opus_decoder_dred_decode24(
                            &dred,
                            (lost_count - fr) * output_samples,
                            &mut out_buf,
                            output_samples,
                        ));
                    } else {
                        output_samples =
                            c_ret(d.opus_decode24(None, &mut out_buf, output_samples, 0));
                    }
                    #[cfg(not(feature = "dred"))]
                    {
                        output_samples =
                            c_ret(d.opus_decode24(None, &mut out_buf, output_samples, 0));
                    }
                } else {
                    output_samples = max_frame_size;
                    output_samples =
                        c_ret(d.opus_decode24(Some(packet), &mut out_buf, output_samples, 0));
                }
                if output_samples > 0 {
                    if !decode_only && tot_out + output_samples as u64 > tot_in {
                        stop = true;
                        // opus_uint64 arithmetic, as in C.
                        output_samples = tot_in.wrapping_sub(tot_out) as i32;
                    }
                    if output_samples > skip {
                        let n = ((output_samples - skip) * channels) as usize;
                        let src = &out_buf[(skip * channels) as usize..][..n];
                        match format {
                            Format::S16 => {
                                for (o, &v) in fbytes.as_chunks_mut::<2>().0.iter_mut().zip(src) {
                                    let s = v.clamp(-0x007fff00, 0x007fff00);
                                    let s = (s + 128) >> 8;
                                    *o = [(s & 0xFF) as u8, ((s >> 8) & 0xFF) as u8];
                                }
                            }
                            Format::S24 => {
                                for (o, &v) in fbytes.as_chunks_mut::<3>().0.iter_mut().zip(src) {
                                    let s = v.clamp(-0x007fffff, 0x007fffff);
                                    *o = [
                                        (s & 0xFF) as u8,
                                        ((s >> 8) & 0xFF) as u8,
                                        ((s >> 16) & 0xFF) as u8,
                                    ];
                                }
                            }
                            Format::F32 => {
                                for (o, &v) in fbytes.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                                    *o = (v as f32 * (1.0f32 / 8388608.0f32)).to_le_bytes();
                                }
                            }
                        }
                        if !fwrite(stderr, fout, &fbytes[..n * format.size()])? {
                            return Ok(ret);
                        }
                        tot_out += (output_samples - skip) as u64;
                    }
                    if output_samples < skip {
                        skip -= output_samples;
                    } else {
                        skip = 0;
                    }
                } else {
                    writeln!(
                        stderr,
                        "error decoding frame: {}",
                        opus_strerror(output_samples)
                    )?;
                }
                tot_samples += f64::from(output_samples);
            }
            dec_final_range = d.final_range();
        } else {
            // encode_only: write the packet record.
            if lost != 0 {
                enc_final_range = 0;
                len = 0;
            }
            if !fwrite(stderr, fout, &int_to_char(len as u32))?
                || !fwrite(stderr, fout, &int_to_char(enc_final_range))?
                || !fwrite(stderr, fout, &data[..len as usize])?
            {
                return Ok(ret);
            }
            tot_samples += f64::from(nb_encoded);
        }

        // compare final range encoder rng values of encoder and decoder
        if enc_final_range != 0
            && !encode_only
            && lost == 0
            && lost_prev == 0
            && dec_final_range != enc_final_range
        {
            writeln!(
                stderr,
                "Error: Range coder state mismatch between encoder and decoder in frame {count}: 0x{enc_final_range:8x} vs 0x{dec_final_range:8x}"
            )?;
            return Ok(ret);
        }

        lost_prev = lost;
        if lost == 0 {
            lost_count = 0;
        }
        if count >= use_inbandfec {
            // count bits
            bits += f64::from(len * 8);
            bits_max = if f64::from(len * 8) > bits_max {
                f64::from(len * 8)
            } else {
                bits_max
            };
            bits2 += f64::from(len) * f64::from(len) * 64.0;
            if !decode_only {
                let n = (frame_size * channels) as usize;
                let mut nrg = 0.0f64;
                for &v in &in_buf[..n] {
                    nrg += f64::from(v) * f64::from(v);
                }
                nrg /= f64::from(frame_size * channels);
                if nrg > 1e5 {
                    bits_act += f64::from(len * 8);
                    count_act += 1;
                }
            }
        }
        count += 1;
    }

    if decode_only && count > 0 {
        frame_size = (tot_samples / f64::from(count)) as i32;
    }
    count -= use_inbandfec;
    if tot_samples >= 1.0 && count > 0 && frame_size != 0 {
        // Print out bitrate statistics
        let fs = f64::from(sampling_rate);
        let fsz = f64::from(frame_size);
        writeln!(
            stderr,
            "average bitrate:             {:>7} kb/s",
            c_fmt_f(1e-3 * bits * fs / tot_samples, 3)
        )?;
        writeln!(
            stderr,
            "maximum bitrate:             {:>7} kb/s",
            c_fmt_f(1e-3 * bits_max * fs / fsz, 3)
        )?;
        if !decode_only {
            writeln!(
                stderr,
                "active bitrate:              {:>7} kb/s",
                c_fmt_f(
                    1e-3 * bits_act * fs / (1e-15 + fsz * f64::from(count_act)),
                    3
                )
            )?;
        }
        let c = f64::from(count);
        let mut var = bits2 / c - bits * bits / (c * c);
        if var < 0.0 {
            var = 0.0;
        }
        writeln!(
            stderr,
            "bitrate standard deviation:  {:>7} kb/s",
            c_fmt_f(1e-3 * var.sqrt() * fs / fsz, 3)
        )?;
    } else {
        writeln!(stderr, "bitrate statistics are undefined")?;
    }
    ret = EXIT_SUCCESS;
    Ok(ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glibc_rand_sequence() {
        // First outputs of glibc rand() in an unseeded program.
        let mut r = GlibcRand::default();
        let v: Vec<i32> = (0..5).map(|_| r.next_value()).collect();
        assert_eq!(
            v,
            [1804289383, 846930886, 1681692777, 1714636915, 1957747793]
        );
        // srand(0) is srand(1).
        let mut z = GlibcRand::new(0);
        assert_eq!(z.next_value(), 1804289383);
        let mut s = GlibcRand::new(42);
        assert_eq!(s.next_value(), 71876166);
    }

    #[test]
    fn loss_file_scanf() {
        let mut f = LossFile {
            data: b" 0 1\n-2 +3 x 4".to_vec(),
            pos: 0,
        };
        assert_eq!(f.scan_int(), Some(0));
        assert_eq!(f.scan_int(), Some(1));
        assert_eq!(f.scan_int(), Some(-2));
        assert_eq!(f.scan_int(), Some(3));
        assert_eq!(f.scan_int(), None);
        assert_eq!(f.scan_int(), None);
        let mut e = LossFile {
            data: b"7".to_vec(),
            pos: 0,
        };
        assert_eq!(e.scan_int(), Some(7));
        assert_eq!(e.scan_int(), None);
    }

    #[test]
    fn usage_and_bad_args() {
        let run = |args: &[&str]| {
            let args: Vec<String> = args.iter().map(|s| (*s).to_owned()).collect();
            let mut out = Vec::new();
            let mut err = Vec::new();
            let code = opus_demo_main(&args, &mut out, &mut err).unwrap();
            (
                code,
                String::from_utf8(out).unwrap(),
                String::from_utf8(err).unwrap(),
            )
        };
        let (code, out, err) = run(&["opus_demo"]);
        assert_eq!(code, EXIT_FAILURE);
        assert!(out.is_empty());
        assert!(err.starts_with("Usage: opus_demo [-e] <application>"));
        let (code, _, err) = run(&["opus_demo", "-d", "44100", "1", "a", "b"]);
        assert_eq!(code, EXIT_FAILURE);
        assert!(err.contains("Supported sampling rates are 8000"));
        let (code, out, _) = run(&["opus_demo", "-d", "48000", "1", "-bogus", "a", "b"]);
        assert_eq!(code, EXIT_FAILURE);
        assert_eq!(out, "Error: unrecognized setting: -bogus\n\n");
        let (code, _, err) = run(&["opus_demo", "-d", "48000", "1", "-cbr", "a", "b"]);
        assert_eq!(code, EXIT_FAILURE);
        assert!(err.ends_with("option -cbr is only for encoding\n"));
    }
}
