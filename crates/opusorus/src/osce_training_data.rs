//! OSCE training data output (feature `osce-training-data`, libopus
//! `--enable-osce-training-data` / `ENABLE_OSCE_TRAINING_DATA`).
//!
//! Like upstream, the library then appends raw training data to files in the **current working
//! directory**, opened (truncated) on first use:
//!
//! | File | Written by | Per call |
//! |---|---|---|
//! | [`CLEAN_HP`] | encoder, `voip` application: the high-passed input (`src/opus_encoder.c`) | `frame_size` `i16` |
//! | [`FEATURES_NUM_BITS`] | OSCE enhancer (20 ms SILK frame at 16 kHz, `dnn/osce.c`) | 1 `i32`: the frame's bits |
//! | [`FEATURES_NUM_BITS_SMOOTH`] | OSCE enhancer | 1 `f32`: smoothed bits |
//! | [`FEATURES_GAIN`] | OSCE enhancer | 4 `f32`: subframe gains |
//! | [`FEATURES_LPC`] | OSCE enhancer | 4 × 16 `f32`: LPC coefficients |
//! | [`FEATURES_LTP`] | OSCE enhancer | 4 × 5 `f32`: LTP coefficients |
//! | [`FEATURES_PERIOD`] | OSCE enhancer | 4 `i16`: pitch lags (0 if unvoiced) |
//! | [`NOISY_16K`] | OSCE enhancer | 320 `i16`: the decoded frame before enhancement |
//!
//! All values are native-endian, as C's `fwrite` writes them. The files are process-wide (one
//! set shared by every encoder and decoder, as C's `static FILE *`). C keeps them open until
//! the process exits; [`close_all`] closes them early (the next write then starts new files),
//! which `opus_demo` does before returning. Writes are unbuffered, so the files are complete
//! after every encoder/decoder call even if [`close_all`] is never called.
//!
//! I/O errors cannot be returned from the codec calls that write (C would crash on a failed
//! `fopen`): the first one is kept, further training output is dropped, and [`close_all`]
//! returns it.

use std::fs::File;
use std::io::{self, Write};
use std::sync::{Mutex, MutexGuard};
use std::vec::Vec;

/// High-pass filtered clean signal (encoder, `voip`).
pub const CLEAN_HP: &str = "clean_hp.s16";
/// LPC coefficients (OSCE).
pub const FEATURES_LPC: &str = "features_lpc.f32";
/// Subframe gains (OSCE).
pub const FEATURES_GAIN: &str = "features_gain.f32";
/// LTP coefficients (OSCE).
pub const FEATURES_LTP: &str = "features_ltp.f32";
/// Pitch lags (OSCE).
pub const FEATURES_PERIOD: &str = "features_period.s16";
/// Decoded 16 kHz signal before enhancement (OSCE).
pub const NOISY_16K: &str = "noisy_16k.s16";
/// SILK payload bits per frame (OSCE).
pub const FEATURES_NUM_BITS: &str = "features_num_bits.s32";
/// Smoothed payload bits (OSCE).
pub const FEATURES_NUM_BITS_SMOOTH: &str = "features_num_bits_smooth.f32";

/// The open files and the first I/O error.
struct Files {
    open: Vec<(&'static str, File)>,
    error: Option<io::Error>,
}

static FILES: Mutex<Files> = Mutex::new(Files {
    open: Vec::new(),
    error: None,
});

fn lock() -> MutexGuard<'static, Files> {
    match FILES.lock() {
        Ok(g) => g,
        // A panic while writing leaves the registry consistent (entries are pushed complete).
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// `fopen(name, "wb")` on first use, then `fwrite(bytes)`. After an I/O error nothing more is
/// written (the error is kept for [`close_all`]).
pub(crate) fn write(name: &'static str, bytes: &[u8]) {
    let mut files = lock();
    if files.error.is_some() {
        return;
    }
    let pos = match files.open.iter().position(|(n, _)| *n == name) {
        Some(pos) => pos,
        None => match File::create(name) {
            Ok(f) => {
                files.open.push((name, f));
                files.open.len() - 1
            }
            Err(e) => {
                files.error = Some(e);
                return;
            }
        },
    };
    if let Err(e) = files.open[pos].1.write_all(bytes) {
        files.error = Some(e);
    }
}

/// Opens every file of `names` that is not open yet, in order (C opens the whole set on the
/// first call, before writing any of them).
pub(crate) fn open_all(names: &[&'static str]) {
    for &name in names {
        write(name, &[]);
    }
}

/// Closes every training file (the next write truncates and starts it again, as a new C
/// process would).
///
/// # Errors
/// The first I/O error since the previous call (opening, writing or syncing a file); the files
/// are closed and the error cleared either way.
pub fn close_all() -> io::Result<()> {
    let mut files = lock();
    let mut result = match files.error.take() {
        Some(e) => Err(e),
        None => Ok(()),
    };
    for (_, mut f) in files.open.drain(..) {
        // `File` is unbuffered; flush is a no-op kept for the contract.
        if let Err(e) = f.flush()
            && result.is_ok()
        {
            result = Err(e);
        }
    }
    result
}
