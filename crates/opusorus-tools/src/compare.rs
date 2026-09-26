//! Audio quality comparison tools: ports of `src/opus_compare.c` and (feature `qext`)
//! `src/qext_compare.c`.
//!
//! The analysis functions work on in-memory interleaved sample slices (as produced by
//! [`read_pcm`]) and return a result struct; the `*_main` functions are the command-line front
//! ends (argument parsing, file reading, stderr text and exit codes identical to the C tools).
//!
//! Differences from C, all confined to cases where the C tools have undefined behaviour or crash
//! (NULL `argv` entries, out-of-bounds reads, division by zero): the Rust port reports an error
//! instead. Read errors on an opened file are propagated as `io::Error` (C treats them as EOF).

use core::fmt;
use std::fs::File;
use std::io::{self, Read, Write};

mod opus;
pub use opus::{
    BANDS as OPUS_COMPARE_BANDS, NBANDS as OPUS_COMPARE_NBANDS, OpusCompareOptions,
    OpusCompareResult, opus_compare, opus_compare_main,
};

#[cfg(feature = "qext")]
mod qext;
#[cfg(feature = "qext")]
pub use qext::{
    BANDS as QEXT_COMPARE_BANDS, NBANDS as QEXT_COMPARE_NBANDS, QextCompareOptions,
    QextCompareResult, qext_compare, qext_compare_main,
};

/// C `EXIT_SUCCESS`.
pub const EXIT_SUCCESS: i32 = 0;
/// C `EXIT_FAILURE`.
pub const EXIT_FAILURE: i32 = 1;

/// `OPUS_PI` as defined by both compare tools (`3.14159265F`).
const OPUS_PI: f32 = 3.14159265;

/// PCM sample format of the input files (`FORMAT_S16_LE`, `FORMAT_S24_LE`, `FORMAT_F32_LE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SampleFormat {
    /// Signed 16-bit little endian (the only format `opus_compare` reads).
    #[default]
    S16Le,
    /// Signed 24-bit little endian, scaled by `1/256` to the 16-bit range.
    S24Le,
    /// 32-bit IEEE float little endian, scaled by `32768` to the 16-bit range.
    F32Le,
}

impl SampleFormat {
    /// Bytes per sample (`format_size[]`).
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Self::S16Le => 2,
            Self::S24Le => 3,
            Self::F32Le => 4,
        }
    }
}

/// Port of `src/qext_compare.c:read_pcm` (and, for [`SampleFormat::S16Le`],
/// `src/opus_compare.c:read_pcm16`) on an in-memory file image.
///
/// Returns interleaved samples (`nchannels` per frame) in the 16-bit range. Like the C `fread`
/// loop, a trailing partial frame is dropped.
///
/// # Panics
/// If `nchannels` is 0.
#[must_use]
pub fn read_pcm(bytes: &[u8], nchannels: usize, format: SampleFormat) -> Vec<f32> {
    assert!(nchannels > 0, "nchannels must be positive");
    let size = format.size();
    let frame = size * nchannels;
    let nsamples = bytes.len() / frame;
    let bytes = &bytes[..nsamples * frame];
    match format {
        SampleFormat::S16Le => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| {
                let mut s = i32::from(b[1]) << 8 | i32::from(b[0]);
                s = ((s & 0xFFFF) ^ 0x8000) - 0x8000;
                s as f32
            })
            .collect(),
        SampleFormat::S24Le => bytes
            .as_chunks::<3>()
            .0
            .iter()
            .map(|b| {
                let mut s = i32::from(b[2]) << 16 | i32::from(b[1]) << 8 | i32::from(b[0]);
                s = ((s & 0xFFFFFF) ^ 0x800000) - 0x800000;
                (1.0f32 / 256.0f32) * s as f32
            })
            .collect(),
        SampleFormat::F32Le => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from_le_bytes([b[0], b[1], b[2], b[3]])) * 32768.0)
            .collect(),
    }
}

/// Port of `src/opus_compare.c:read_pcm16` on an in-memory file image.
#[must_use]
pub fn read_pcm16(bytes: &[u8], nchannels: usize) -> Vec<f32> {
    read_pcm(bytes, nchannels, SampleFormat::S16Le)
}

/// Reads a whole PCM file with [`read_pcm`].
///
/// # Errors
/// Any I/O error opening or reading `path`.
pub fn read_pcm_file(
    path: impl AsRef<std::path::Path>,
    nchannels: usize,
    format: SampleFormat,
) -> io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    Ok(read_pcm(&bytes, nchannels, format))
}

/// Errors reported by the comparison functions. `Display` produces the exact C message (without
/// the trailing newline) where the C tools have one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareError {
    /// `Sample counts do not match (%lu!=%lu).`
    SampleCountMismatch {
        /// Frames in the reference signal.
        xlength: usize,
        /// Frames in the test signal times the downsampling factor.
        ylength_scaled: usize,
    },
    /// `Insufficient sample data (%lu<%lu).`
    InsufficientData {
        /// Frames in the reference signal.
        xlength: usize,
        /// Analysis window size.
        window: usize,
    },
    /// Unsupported sampling rate (`opus_compare` message).
    OpusRate(u32),
    /// Unsupported sampling rate (`qext_compare` message).
    QextRate(u32),
    /// Unsupported base rate (C only ever sets 48000 or 96000).
    BaseRate(u32),
    /// The test rate is above the base rate: C divides by zero.
    RateAboveBase {
        /// Test signal rate.
        rate: u32,
        /// Base (reference) rate.
        base_rate: u32,
    },
    /// `-skip` would move outside the test signal: C reads out of bounds.
    SkipOutOfRange(i32),
    /// Invalid channel count (must be 1 or 2).
    Channels(usize),
    /// FFT setup failed (cannot happen for the window sizes the tools use).
    Fft(opusorus::Error),
}

impl fmt::Display for CompareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::SampleCountMismatch {
                xlength,
                ylength_scaled,
            } => write!(
                f,
                "Sample counts do not match ({xlength}!={ylength_scaled})."
            ),
            Self::InsufficientData { xlength, window } => {
                write!(f, "Insufficient sample data ({xlength}<{window}).")
            }
            Self::OpusRate(_) => {
                f.write_str("Sampling rate must be 8000, 12000, 16000, 24000, or 48000")
            }
            Self::QextRate(_) => {
                f.write_str("Sampling rate must be 8000, 12000, 16000, 24000, 48000, or 96000")
            }
            Self::BaseRate(r) => write!(f, "Base rate must be 48000 or 96000 (got {r})"),
            Self::RateAboveBase { rate, base_rate } => {
                write!(f, "Sampling rate {rate} exceeds the base rate {base_rate}")
            }
            Self::SkipOutOfRange(s) => write!(f, "Skip {s} is outside the test signal"),
            Self::Channels(n) => write!(f, "Channel count must be 1 or 2 (got {n})"),
            Self::Fft(e) => write!(f, "FFT setup failed: {e}"),
        }
    }
}

impl std::error::Error for CompareError {}

/// `OPUS_COSF(_x)`: `(float)cos(_x)` on a float argument.
fn opus_cosf(x: f32) -> f32 {
    opusorus::math::cos(f64::from(x)) as f32
}

/// `OPUS_SINF(_x)`: `(float)sin(_x)` on a float argument.
fn opus_sinf(x: f32) -> f32 {
    opusorus::math::sin(f64::from(x)) as f32
}

/// glibc `printf("%.*f", prec, v)`: like Rust's `{:.prec$}` (both exact, ties to even) except for
/// the spelling of non-finite values.
#[must_use]
pub fn c_fmt_f(v: f64, prec: usize) -> String {
    if v.is_nan() {
        if v.is_sign_negative() {
            "-nan".to_owned()
        } else {
            "nan".to_owned()
        }
    } else if v.is_infinite() {
        if v.is_sign_negative() {
            "-inf".to_owned()
        } else {
            "inf".to_owned()
        }
    } else {
        format!("{v:.prec$}")
    }
}

/// C `isspace` in the "C" locale.
pub(crate) const fn c_isspace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// glibc `atoi`: `(int)strtol(s, NULL, 10)` (leading whitespace, optional sign, decimal digits,
/// saturating to the `long` range, then truncated to `int`).
#[must_use]
pub fn c_atoi(s: &str) -> i32 {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() && c_isspace(b[i]) {
        i += 1;
    }
    let mut neg = false;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        neg = b[i] == b'-';
        i += 1;
    }
    // Accumulate as a negative number so i64::MIN is representable (like strtol).
    let mut acc: i64 = 0;
    let mut overflow = false;
    while i < b.len() && b[i].is_ascii_digit() {
        let d = i64::from(b[i] - b'0');
        match acc.checked_mul(10).and_then(|v| v.checked_sub(d)) {
            Some(v) => acc = v,
            None => overflow = true,
        }
        i += 1;
    }
    let v = if overflow {
        if neg { i64::MIN } else { i64::MAX }
    } else if neg {
        acc
    } else {
        // `acc == i64::MIN` means the positive value overflows `long`: saturate like strtol.
        acc.saturating_neg()
    };
    v as i32
}

/// glibc `atof` (`strtod`) for the decimal and `inf`/`nan` forms: parses the longest valid
/// prefix after leading whitespace, `0.0` if there is none. Hexadecimal floats are not supported
/// (they parse as their leading `0`).
#[must_use]
pub fn c_atof(s: &str) -> f64 {
    let b = s.as_bytes();
    let mut start = 0;
    while start < b.len() && c_isspace(b[start]) {
        start += 1;
    }
    let b = &b[start..];
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let rest = &b[i..];
    let lower: Vec<u8> = rest.iter().take(8).map(u8::to_ascii_lowercase).collect();
    let end = if lower.starts_with(b"infinity") {
        i + 8
    } else if lower.starts_with(b"inf") || lower.starts_with(b"nan") {
        i + 3
    } else {
        let mut j = i;
        let mut digits = 0;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
            digits += 1;
        }
        if j < b.len() && b[j] == b'.' {
            j += 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
                digits += 1;
            }
        }
        if digits == 0 {
            return 0.0;
        }
        if j < b.len() && (b[j] == b'e' || b[j] == b'E') {
            let mut k = j + 1;
            if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
                k += 1;
            }
            if k < b.len() && b[k].is_ascii_digit() {
                while k < b.len() && b[k].is_ascii_digit() {
                    k += 1;
                }
                j = k;
            }
        }
        j
    };
    parse_scanned_float(&b[..end])
}

/// Parses a prefix already validated by [`c_atof`]'s scanner.
#[expect(
    clippy::expect_used,
    reason = "the scanner only accepts ASCII strings of Rust's float grammar \
              ([sign] digits[.digits][exp] with at least one digit, inf, infinity, nan)"
)]
fn parse_scanned_float(b: &[u8]) -> f64 {
    let s = core::str::from_utf8(b).expect("scanned prefix is ASCII");
    s.parse::<f64>().expect("scanned prefix is a valid float")
}

/// C `argv[i]` for the tools' argument walking; `None` where C would read a NULL pointer or past
/// the end of `argv` (the C tools crash there; the Rust front ends print the usage instead).
fn arg(args: &[String], base: usize, i: usize) -> Option<&str> {
    args.get(base + i).map(String::as_str)
}

/// `argv[0]` as printed in the usage text (glibc prints `(null)` for a NULL `%s`).
pub(crate) fn argv0(args: &[String]) -> &str {
    match args.first() {
        Some(a) => a,
        None => "(null)",
    }
}

/// Outcome of opening both input files (`fopen` order and messages of the C tools).
fn open_inputs(
    stderr: &mut dyn Write,
    path1: &str,
    path2: &str,
) -> io::Result<Option<(Vec<u8>, Vec<u8>)>> {
    // The C tools report a failed `fopen` with the path only (no reason).
    let Ok(mut fin1) = File::open(path1) else {
        writeln!(stderr, "Error opening '{path1}'.")?;
        return Ok(None);
    };
    let Ok(mut fin2) = File::open(path2) else {
        writeln!(stderr, "Error opening '{path2}'.")?;
        return Ok(None);
    };
    let mut b1 = Vec::new();
    fin1.read_to_end(&mut b1)?;
    let mut b2 = Vec::new();
    fin2.read_to_end(&mut b2)?;
    Ok(Some((b1, b2)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atoi_matches_glibc() {
        assert_eq!(c_atoi("48000"), 48000);
        assert_eq!(c_atoi("  \t-12abc"), -12);
        assert_eq!(c_atoi("+7"), 7);
        assert_eq!(c_atoi(""), 0);
        assert_eq!(c_atoi("x1"), 0);
        assert_eq!(c_atoi("- 1"), 0);
        // strtol saturates to LONG_MAX/LONG_MIN, then the int conversion truncates.
        assert_eq!(c_atoi("99999999999"), 99_999_999_999i64 as i32);
        assert_eq!(c_atoi("99999999999999999999"), i64::MAX as i32);
        assert_eq!(c_atoi("-99999999999999999999"), i64::MIN as i32);
        assert_eq!(c_atoi("9223372036854775808"), i64::MAX as i32);
        assert_eq!(c_atoi("-9223372036854775808"), i64::MIN as i32);
    }

    #[test]
    fn atof_matches_glibc() {
        assert_eq!(c_atof("0.05"), 0.05);
        assert_eq!(c_atof(".1"), 0.1);
        assert_eq!(c_atof("1."), 1.0);
        assert_eq!(c_atof(" -2.5e3xyz"), -2500.0);
        assert_eq!(c_atof("7e"), 7.0);
        assert_eq!(c_atof("7e+"), 7.0);
        assert_eq!(c_atof("abc"), 0.0);
        assert_eq!(c_atof("."), 0.0);
        assert_eq!(c_atof("-INFINITY"), f64::NEG_INFINITY);
        assert_eq!(c_atof("infx"), f64::INFINITY);
        assert!(c_atof("nan").is_nan());
    }

    #[test]
    fn fmt_f_matches_glibc() {
        assert_eq!(c_fmt_f(0.25, 1), "0.2");
        assert_eq!(c_fmt_f(0.35, 1), "0.3");
        assert_eq!(c_fmt_f(-0.0, 1), "-0.0");
        assert_eq!(c_fmt_f(-1.0, 6), "-1.000000");
        assert_eq!(c_fmt_f(f64::NAN, 6), "nan");
        assert_eq!(c_fmt_f(-f64::NAN, 6), "-nan");
        assert_eq!(c_fmt_f(f64::NEG_INFINITY, 6), "-inf");
    }

    #[test]
    fn read_pcm_formats() {
        let b = [0x01, 0x80, 0xff, 0x7f, 0x00];
        assert_eq!(read_pcm(&b, 1, SampleFormat::S16Le), [-32767.0, 32767.0]);
        assert_eq!(read_pcm(&b, 2, SampleFormat::S16Le), [-32767.0, 32767.0]);
        assert_eq!(read_pcm(&b, 1, SampleFormat::S24Le), [-32767.0 / 256.0]);
        let f = 0.5f32.to_le_bytes();
        assert_eq!(read_pcm(&f, 1, SampleFormat::F32Le), [16384.0]);
        assert!(read_pcm(&f, 2, SampleFormat::F32Le).is_empty());
    }
}
