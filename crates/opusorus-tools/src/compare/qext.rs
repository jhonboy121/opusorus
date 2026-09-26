//! Port of `src/qext_compare.c`: the Opus HD (QEXT) quality metric (up to 96 kHz, 16/24-bit and
//! float input), using the `celt/mini_kfft.c` port for the spectra.

use super::{
    CompareError, EXIT_FAILURE, EXIT_SUCCESS, OPUS_PI, SampleFormat, arg, argv0, c_atof, c_atoi,
    c_fmt_f, open_inputs, read_pcm,
};
use opusorus::celt::mini_kfft::{MiniKissFftCpx, mini_kiss_fftr, mini_kiss_fftr_alloc};
use opusorus::math;
use std::io::{self, Write};

/// `NBANDS`.
pub const NBANDS: usize = 28;
/// `NFREQS`.
const NFREQS: usize = 240 * 2;
/// `TEST_WIN_SIZE`.
const TEST_WIN_SIZE: usize = 480 * 2;
/// `TEST_WIN_STEP`.
const TEST_WIN_STEP: usize = 120 * 2;

/// `BANDS`: bands on which we compute the pseudo-NMR (Bark-derived CELT bands).
pub const BANDS: [i32; NBANDS + 1] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 68, 80, 96, 120, 156, 200, 240, 280,
    320, 360, 400, 440, 480,
];

/// `MAX(a,b)`: `((a)>(b) ? (a) : (b))`.
const fn max_f32(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// `MAX(a,b)` on doubles.
const fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

/// Port of `src/qext_compare.c:band_energy`: Blackman-Harris windowed real-FFT power spectrum
/// (`ps`) and, if `out` is given, per-band average power of `input` for `nframes` frames.
#[expect(clippy::too_many_arguments, reason = "mirrors the C signature")]
#[expect(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]
fn band_energy(
    mut out: Option<&mut [f32]>,
    ps: &mut [f32],
    bands: &[i32],
    nbands: usize,
    input: &[f32],
    nchannels: usize,
    nframes: usize,
    window_sz: usize,
    step: usize,
    downsample: i32,
) -> Result<(), CompareError> {
    let mut window = [0f32; TEST_WIN_SIZE];
    let mut x = [0f32; TEST_WIN_SIZE];
    let mut xf = [[MiniKissFftCpx::default(); NFREQS + 1]; 2];
    let ps_sz = window_sz / 2;
    // Blackman-Harris window.
    for xj in 0..window_sz {
        let n: f64 = (xj as f64 + 0.5) / window_sz as f64;
        window[xj] = (0.35875 - 0.48829 * math::cos(f64::from(2.0 * OPUS_PI) * n)
            + 0.14128 * math::cos(f64::from(4.0 * OPUS_PI) * n)
            - 0.01168 * math::cos(f64::from(6.0 * OPUS_PI) * n)) as f32;
    }
    let mut kfft = mini_kiss_fftr_alloc(window_sz as i32, false).map_err(CompareError::Fft)?;
    for xi in 0..nframes {
        for ci in 0..nchannels {
            for xk in 0..window_sz {
                x[xk] = window[xk] * input[(xi * step + xk) * nchannels + ci];
            }
            mini_kiss_fftr(&mut kfft, &x[..window_sz], &mut xf[ci]);
        }
        let mut xj = 0usize;
        for bi in 0..nbands {
            let mut p = [0f32; 2];
            while (xj as i32) < bands[bi + 1] {
                for ci in 0..nchannels {
                    let re: f32 = xf[ci][xj].r * downsample as f32;
                    let im: f32 = xf[ci][xj].i * downsample as f32;
                    ps[(xi * ps_sz + xj) * nchannels + ci] =
                        (f64::from(re * re + im * im) + 0.1) as f32;
                    p[ci] += ps[(xi * ps_sz + xj) * nchannels + ci];
                }
                xj += 1;
            }
            if let Some(out) = out.as_deref_mut() {
                out[(xi * nbands + bi) * nchannels] = p[0] / (bands[bi + 1] - bands[bi]) as f32;
                if nchannels == 2 {
                    out[(xi * nbands + bi) * nchannels + 1] =
                        p[1] / (bands[bi + 1] - bands[bi]) as f32;
                }
            }
        }
    }
    Ok(())
}

/// Options of [`qext_compare`] (the command-line switches except the file format).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QextCompareOptions {
    /// Channels of the comparison (`-s` selects 2; with 1 the reference is downmixed).
    pub nchannels: usize,
    /// Rate of the reference signal: 96000 (default) or 48000 (`-48k`).
    pub base_rate: u32,
    /// Rate of the test signal (`-r`); 0 means `base_rate`.
    pub rate: u32,
    /// Frames to skip at the start of the test signal (`-skip`, applied with the C quirks).
    pub skip: i32,
}

impl Default for QextCompareOptions {
    fn default() -> Self {
        Self {
            nchannels: 1,
            base_rate: 96000,
            rate: 0,
            skip: 0,
        }
    }
}

/// Result of [`qext_compare`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QextCompareResult {
    /// `err4`.
    pub err4: f64,
    /// `err16`.
    pub err16: f64,
    /// RMS sample error (`-1` unless stereo at the base rate).
    pub rms: f64,
}

impl QextCompareResult {
    /// The C `-thresholds` check: all three errors at or below their threshold.
    #[must_use]
    pub fn passes(&self, err4_threshold: f64, err16_threshold: f64, rms_threshold: f64) -> bool {
        self.err4 <= err4_threshold && self.err16 <= err16_threshold && self.rms <= rms_threshold
    }
}

/// Port of the analysis part of `src/qext_compare.c:main`.
///
/// `x` is the reference file decoded as **stereo** interleaved samples (the C tool always reads
/// file 1 as stereo and downmixes it for a mono comparison); `y` is the test signal with
/// `opts.nchannels` interleaved channels. Trailing partial frames are ignored.
///
/// C quirk kept: `skip` is scaled by the channel count and then used both as a sample offset
/// into `y` and as a frame count subtracted from its length.
///
/// # Errors
/// [`CompareError::SampleCountMismatch`] and [`CompareError::InsufficientData`] like C; invalid
/// options, and cases where C divides by zero or reads out of bounds
/// ([`CompareError::RateAboveBase`], [`CompareError::SkipOutOfRange`]).
pub fn qext_compare(
    x: &[f32],
    y: &[f32],
    opts: &QextCompareOptions,
) -> Result<QextCompareResult, CompareError> {
    let nchannels = opts.nchannels;
    if nchannels != 1 && nchannels != 2 {
        return Err(CompareError::Channels(nchannels));
    }
    let base_rate = opts.base_rate;
    if base_rate != 48000 && base_rate != 96000 {
        return Err(CompareError::BaseRate(base_rate));
    }
    let mut rate = opts.rate;
    let mut nbands = NBANDS;
    let mut nfreqs = NFREQS;
    let mut test_win_size = TEST_WIN_SIZE;
    let mut test_win_step = TEST_WIN_STEP;
    if rate == 0 {
        rate = base_rate;
    }
    if base_rate == 48000 {
        test_win_size /= 2;
        test_win_step /= 2;
        nfreqs /= 2;
        nbands = 22;
    }
    let ybands = match rate {
        8000 => 13,
        12000 => 15,
        16000 => 17,
        24000 => 19,
        48000 => 22,
        96000 => NBANDS,
        _ => return Err(CompareError::QextRate(rate)),
    };
    let downsample = (base_rate / rate) as usize;
    if downsample == 0 {
        return Err(CompareError::RateAboveBase { rate, base_rate });
    }
    let yfreqs = nfreqs / downsample;

    // Read in the data and allocate scratch space.
    let xlength = x.len() / 2;
    let mut x = x[..xlength * 2].to_vec();
    if nchannels == 1 {
        for xi in 0..xlength {
            x[xi] = (0.5 * f64::from(x[2 * xi] + x[2 * xi + 1])) as f32;
        }
    }
    let mut ylength = y.len() / nchannels;
    let skip = opts.skip * nchannels as i32;
    let yoff = skip / downsample as i32;
    let Ok(yoff) = usize::try_from(yoff) else {
        // C: negative pointer offset (undefined behaviour).
        return Err(CompareError::SkipOutOfRange(opts.skip));
    };
    // C size_t arithmetic (wraps).
    ylength = ylength.wrapping_sub(yoff);
    if skip != 0 && ylength.wrapping_mul(downsample) > xlength {
        ylength = xlength / downsample;
    }
    if xlength != ylength.wrapping_mul(downsample) {
        return Err(CompareError::SampleCountMismatch {
            xlength,
            ylength_scaled: ylength.wrapping_mul(downsample),
        });
    }
    if xlength < test_win_size {
        return Err(CompareError::InsufficientData {
            xlength,
            window: test_win_size,
        });
    }
    // Everything the C code reads from `y` lies in `[yoff, yoff + ylength*nchannels)`.
    let Some(y) = y.get(yoff..yoff + ylength * nchannels) else {
        return Err(CompareError::SkipOutOfRange(opts.skip));
    };
    let mut rms = -1f64;
    if nchannels == 2 && downsample == 1 {
        rms = 0.0;
        for i in 0..xlength * nchannels {
            let e = f64::from(x[i] - y[i]);
            rms += e * e;
        }
        rms = math::sqrt(rms / (xlength * nchannels) as f64);
    }
    let nframes = (xlength - test_win_size + test_win_step) / test_win_step;
    let mut xb = vec![0f32; nframes * nbands * nchannels];
    let mut xs = vec![0f32; nframes * nfreqs * nchannels];
    let mut ys = vec![0f32; nframes * yfreqs * nchannels];
    // Compute the per-band spectral energy of the original signal and the error.
    band_energy(
        Some(&mut xb),
        &mut xs,
        &BANDS,
        nbands,
        &x,
        nchannels,
        nframes,
        test_win_size,
        test_win_step,
        1,
    )?;
    drop(x);
    band_energy(
        None,
        &mut ys,
        &BANDS,
        ybands,
        y,
        nchannels,
        nframes,
        test_win_size / downsample,
        test_win_step / downsample,
        downsample as i32,
    )?;
    for xi in 0..nframes {
        let mut max_e = [0f32; 2];
        for bi in 0..nbands {
            for ci in 0..nchannels {
                max_e[ci] = max_f32(max_e[ci], xb[(xi * nbands + bi) * nchannels + ci]);
            }
        }
        // Allow for up to 105 dB instantaneous dynamic range (95dB here + 10dB in mask
        // application).
        for bi in 0..nbands {
            for ci in 0..nchannels {
                let v = &mut xb[(xi * nbands + bi) * nchannels + ci];
                *v = max_f64(3.16e-10 * f64::from(max_e[ci]), f64::from(*v)) as f32;
            }
        }

        // Frequency masking (low to high): 10 dB/Bark slope.
        for bi in 1..nbands {
            for ci in 0..nchannels {
                xb[(xi * nbands + bi) * nchannels + ci] +=
                    0.1f32 * xb[(xi * nbands + bi - 1) * nchannels + ci];
            }
        }
        // Frequency masking (high to low): 15 dB/Bark slope.
        for bi in (0..nbands - 2).rev() {
            for ci in 0..nchannels {
                xb[(xi * nbands + bi) * nchannels + ci] +=
                    0.03f32 * xb[(xi * nbands + bi + 1) * nchannels + ci];
            }
        }
        if xi > 0 {
            // Forward temporal masking: -3 dB/2.5ms slope.
            for bi in 0..nbands {
                for ci in 0..nchannels {
                    xb[(xi * nbands + bi) * nchannels + ci] +=
                        0.5f32 * xb[((xi - 1) * nbands + bi) * nchannels + ci];
                }
            }
        }
    }
    // Backward temporal masking: -10 dB/2.5ms slope.
    // C: `for(xi=nframes-2;xi-->0;)`; with nframes == 1 the size_t start wraps and C reads out of
    // bounds, here the loop is simply empty.
    for xi in (0..nframes.saturating_sub(2)).rev() {
        for bi in 0..nbands {
            for ci in 0..nchannels {
                xb[(xi * nbands + bi) * nchannels + ci] +=
                    0.1f32 * xb[((xi + 1) * nbands + bi) * nchannels + ci];
            }
        }
    }
    for xi in 0..nframes {
        // Allowing some cross-talk
        if nchannels == 2 {
            for bi in 0..nbands {
                let l = xb[(xi * nbands + bi) * nchannels];
                let r = xb[(xi * nbands + bi) * nchannels + 1];
                xb[(xi * nbands + bi) * nchannels] += 0.000001f32 * r;
                xb[(xi * nbands + bi) * nchannels + 1] += 0.000001f32 * l;
            }
        }

        // Apply masking
        for bi in 0..ybands {
            for xj in BANDS[bi] as usize..BANDS[bi + 1] as usize {
                for ci in 0..nchannels {
                    xs[(xi * nfreqs + xj) * nchannels + ci] +=
                        0.1f32 * xb[(xi * nbands + bi) * nchannels + ci];
                    ys[(xi * yfreqs + xj) * nchannels + ci] +=
                        0.1f32 * xb[(xi * nbands + bi) * nchannels + ci];
                }
            }
        }
    }

    // Average of consecutive frames to make comparison slightly less sensitive
    for bi in 0..ybands {
        for xj in BANDS[bi] as usize..BANDS[bi + 1] as usize {
            for ci in 0..nchannels {
                let mut xtmp = xs[xj * nchannels + ci];
                let mut ytmp = ys[xj * nchannels + ci];
                for xi in 1..nframes {
                    let xtmp2 = xs[(xi * nfreqs + xj) * nchannels + ci];
                    let ytmp2 = ys[(xi * yfreqs + xj) * nchannels + ci];
                    xs[(xi * nfreqs + xj) * nchannels + ci] += xtmp;
                    ys[(xi * yfreqs + xj) * nchannels + ci] += ytmp;
                    xtmp = xtmp2;
                    ytmp = ytmp2;
                }
            }
        }
    }

    // If working at a lower sampling rate, don't take into account the last 300 Hz to allow for
    // different transition bands. For 12 kHz, we don't skip anything, because the last band
    // already skips 400 Hz.
    let max_compare = if rate == base_rate {
        BANDS[nbands]
    } else if rate == 12000 {
        BANDS[ybands]
    } else {
        BANDS[ybands] - 3
    };
    let mut err4 = 0f64;
    let mut err16 = 0f64;
    for xi in 0..nframes {
        let mut ef2 = 0f64;
        let mut ef4 = 0f64;
        for bi in 0..ybands {
            let mut eb2 = 0f64;
            let mut eb4 = 0f64;
            let w: f64 = 0.5 + 0.5 * math::tanh(0.5 * f64::from(22 - bi as i32));
            let mut xj = BANDS[bi];
            while xj < BANDS[bi + 1] && xj < max_compare {
                let j = xj as usize;
                let f: f32 = xj as f32 * OPUS_PI / 240.0;
                // Shape the lower threshold similar to 1/(1 - 0.85*z^-1) deemphasis filter at
                // 48 kHz.
                let thresh: f32 = (0.1 / (0.15 * 0.15 + f64::from(f * f))) as f32;
                for ci in 0..nchannels {
                    let yv = ys[(xi * yfreqs + j) * nchannels + ci];
                    let xv = xs[(xi * nfreqs + j) * nchannels + ci];
                    let mut re: f32 = (yv + thresh) / (xv + thresh);
                    let mut im = (f64::from(re) - math::log(f64::from(re)) - 1.0) as f32;
                    // Per-band error weighting.
                    im = (f64::from(im) * w) as f32;
                    eb2 += f64::from(im);
                    // Same for 4th power, but make it less sensitive to very low energies.
                    re = (yv + 10.0 * thresh) / (xv + 10.0 * thresh);
                    im = (f64::from(re) - math::log(f64::from(re)) - 1.0) as f32;
                    // Per-band error weighting.
                    im = (f64::from(im) * w) as f32;
                    eb4 += f64::from(im);
                }
                xj += 1;
            }
            let norm = f64::from((BANDS[bi + 1] - BANDS[bi]) * nchannels as i32);
            eb2 /= norm;
            eb4 /= norm;
            ef2 += eb2;
            ef4 += eb4 * eb4;
        }
        // Using a fixed normalization value means we're willing to accept slightly lower quality
        // for lower sampling rates.
        ef2 /= nbands as f64;
        ef4 /= nbands as f64;
        ef4 *= ef4;
        err4 += ef2 * ef2;
        err16 += ef4 * ef4;
    }
    err4 = math::pow(err4 / nframes as f64, 1.0 / 4.0);
    err16 = math::pow(err16 / nframes as f64, 1.0 / 16.0);
    Ok(QextCompareResult { err4, err16, rms })
}

/// Port of `src/qext_compare.c:usage`.
fn usage(stderr: &mut dyn Write, argv0: &str) -> io::Result<i32> {
    writeln!(
        stderr,
        "Usage: {argv0} [-s] [-48k] [-s16|-s24|-f32] [-r rate2] [-thresholds err4 err16 rms] <file1.sw> <file2.sw>"
    )?;
    Ok(EXIT_FAILURE)
}

/// Port of `src/qext_compare.c:main`: command line `args` (including `argv[0]`), messages
/// written to `stderr`, returns the process exit code.
///
/// # Errors
/// I/O errors writing to `stderr` or reading an opened input file.
pub fn qext_compare_main(args: &[String], stderr: &mut dyn Write) -> io::Result<i32> {
    let argv0 = argv0(args);
    let mut argc = args.len();
    if argc < 3 {
        return usage(stderr, argv0);
    }
    let mut base = 0usize;
    let mut opts = QextCompareOptions::default();
    let mut format = SampleFormat::S16Le;
    let mut thresholds: Option<(f64, f64, f64)> = None;
    while argc > 3 {
        // argc > 3 guarantees argv[1..=3] exist.
        let (Some(a1), Some(a2)) = (arg(args, base, 1), arg(args, base, 2)) else {
            return usage(stderr, argv0);
        };
        match a1 {
            "-s" => {
                opts.nchannels = 2;
                base += 1;
                argc -= 1;
            }
            "-48k" => {
                opts.base_rate = 48000;
                base += 1;
                argc -= 1;
            }
            "-s16" => {
                format = SampleFormat::S16Le;
                base += 1;
                argc -= 1;
            }
            "-s24" => {
                format = SampleFormat::S24Le;
                base += 1;
                argc -= 1;
            }
            "-f32" => {
                format = SampleFormat::F32Le;
                base += 1;
                argc -= 1;
            }
            "-skip" => {
                opts.skip = c_atoi(a2);
                base += 2;
                argc -= 2;
            }
            "-thresholds" => {
                if argc < 7 {
                    return usage(stderr, argv0);
                }
                let (Some(a3), Some(a4)) = (arg(args, base, 3), arg(args, base, 4)) else {
                    return usage(stderr, argv0);
                };
                thresholds = Some((c_atof(a2), c_atof(a3), c_atof(a4)));
                base += 4;
                argc -= 4;
            }
            "-r" => {
                let rate = c_atoi(a2) as u32;
                if rate != 8000
                    && rate != 12000
                    && rate != 16000
                    && rate != 24000
                    && rate != 48000
                    && rate != 96000
                {
                    writeln!(stderr, "{}", CompareError::QextRate(rate))?;
                    return Ok(EXIT_FAILURE);
                }
                opts.rate = rate;
                base += 2;
                argc -= 2;
            }
            _ => return usage(stderr, argv0),
        }
    }
    if argc != 3 {
        return usage(stderr, argv0);
    }
    let rate = if opts.rate == 0 {
        opts.base_rate
    } else {
        opts.rate
    };
    if base_rate_exceeded(rate, opts.base_rate) {
        // C divides by zero here (`nfreqs/downsample`).
        writeln!(
            stderr,
            "{}",
            CompareError::RateAboveBase {
                rate,
                base_rate: opts.base_rate
            }
        )?;
        return Ok(EXIT_FAILURE);
    }
    let (Some(path1), Some(path2)) = (arg(args, base, 1), arg(args, base, 2)) else {
        return usage(stderr, argv0);
    };
    let Some((b1, b2)) = open_inputs(stderr, path1, path2)? else {
        return Ok(EXIT_FAILURE);
    };
    let x = read_pcm(&b1, 2, format);
    drop(b1);
    let y = read_pcm(&b2, opts.nchannels, format);
    drop(b2);
    let res = match qext_compare(&x, &y, &opts) {
        Ok(r) => r,
        Err(e) => {
            writeln!(stderr, "{e}")?;
            return Ok(EXIT_FAILURE);
        }
    };
    writeln!(
        stderr,
        "err4 = {}, err16 = {}, rms = {}",
        c_fmt_f(res.err4, 6),
        c_fmt_f(res.err16, 6),
        c_fmt_f(res.rms, 6)
    )?;
    if let Some((t4, t16, trms)) = thresholds {
        if res.passes(t4, t16, trms) {
            writeln!(stderr, "Comparison PASSED")?;
        } else {
            writeln!(
                stderr,
                "*** Comparison FAILED *** (thresholds were {} {} {})",
                c_fmt_f(t4, 6),
                c_fmt_f(t16, 6),
                c_fmt_f(trms, 6)
            )?;
            return Ok(EXIT_FAILURE);
        }
    }
    Ok(EXIT_SUCCESS)
}

/// `base_rate/rate == 0` (the C tool divides by zero afterwards).
const fn base_rate_exceeded(rate: u32, base_rate: u32) -> bool {
    base_rate / rate == 0
}
