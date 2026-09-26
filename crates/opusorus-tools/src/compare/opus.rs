//! Port of `src/opus_compare.c`: the RFC 6716 conformance quality metric.

use super::{
    CompareError, EXIT_FAILURE, EXIT_SUCCESS, OPUS_PI, SampleFormat, arg, argv0, c_atoi, c_fmt_f,
    open_inputs, opus_cosf, opus_sinf, read_pcm,
};
use opusorus::math;
use std::io::{self, Write};

/// `NBANDS`.
pub const NBANDS: usize = 21;
/// `NFREQS`.
const NFREQS: usize = 240;

/// `BANDS`: bands on which we compute the pseudo-NMR (Bark-derived CELT bands).
pub const BANDS: [i32; NBANDS + 1] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 68, 80, 96, 120, 156, 200,
];

/// `TEST_WIN_SIZE`.
const TEST_WIN_SIZE: usize = 480;
/// `TEST_WIN_STEP`.
const TEST_WIN_STEP: usize = 120;

/// Port of `src/opus_compare.c:band_energy`: windowed DFT power spectrum (`_ps`) and, if `out`
/// is given, per-band average power of `input` for `nframes` frames.
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
) {
    // C allocates window, c, s and x as one block.
    let mut window = vec![0f32; window_sz];
    let mut c = vec![0f32; window_sz];
    let mut s = vec![0f32; window_sz];
    let mut x = vec![0f32; nchannels * window_sz];
    let ps_sz = window_sz / 2;
    for xj in 0..window_sz {
        window[xj] = 0.5f32
            - 0.5f32 * opus_cosf((2.0 * OPUS_PI / (window_sz as i32 - 1) as f32) * xj as f32);
    }
    for xj in 0..window_sz {
        c[xj] = opus_cosf((2.0 * OPUS_PI / window_sz as f32) * xj as f32);
    }
    for xj in 0..window_sz {
        s[xj] = opus_sinf((2.0 * OPUS_PI / window_sz as f32) * xj as f32);
    }
    let window = &window[..window_sz];
    // The C inner loop computes, for bin `xj`, `re += c[ti]*x[xk]; im -= s[ti]*x[xk]` over
    // `xk = 0..window_sz` with `ti = xj*xk mod window_sz`. The same products and sums are
    // evaluated here in the same per-bin order, but for all bins at once from twiddle tables
    // transposed to `[xk][xj]`, so the bin loop is contiguous (vectorizable) — identical
    // results, several times faster.
    let nbins = bands[nbands] as usize;
    let mut tc = vec![0f32; window_sz * nbins];
    let mut ts = vec![0f32; window_sz * nbins];
    for xk in 0..window_sz {
        let mut ti = 0usize;
        for xj in 0..nbins {
            tc[xk * nbins + xj] = c[ti];
            ts[xk * nbins + xj] = s[ti];
            // ti = xj*xk mod window_sz, stepped by xk.
            ti += xk;
            if ti >= window_sz {
                ti -= window_sz;
            }
        }
    }
    let mut re_all = vec![0f32; nchannels * nbins];
    let mut im_all = vec![0f32; nchannels * nbins];
    for xi in 0..nframes {
        for ci in 0..nchannels {
            for xk in 0..window_sz {
                x[ci * window_sz + xk] = window[xk] * input[(xi * step + xk) * nchannels + ci];
            }
        }
        re_all.fill(0.0);
        im_all.fill(0.0);
        for xk in 0..window_sz {
            let rc = &tc[xk * nbins..(xk + 1) * nbins];
            let rs = &ts[xk * nbins..(xk + 1) * nbins];
            for ci in 0..nchannels {
                let xv = x[ci * window_sz + xk];
                let re = &mut re_all[ci * nbins..(ci + 1) * nbins];
                for (r, &cv) in re.iter_mut().zip(rc) {
                    *r += cv * xv;
                }
                let im = &mut im_all[ci * nbins..(ci + 1) * nbins];
                for (i, &sv) in im.iter_mut().zip(rs) {
                    *i -= sv * xv;
                }
            }
        }
        let mut xj = 0usize;
        for bi in 0..nbands {
            let mut p = [0f32; 2];
            while (xj as i32) < bands[bi + 1] {
                for ci in 0..nchannels {
                    let mut re = re_all[ci * nbins + xj];
                    let mut im = im_all[ci * nbins + xj];
                    re *= downsample as f32;
                    im *= downsample as f32;
                    ps[(xi * ps_sz + xj) * nchannels + ci] = re * re + im * im + 100000.0;
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
}

/// Options of [`opus_compare`] (the `-s` and `-r` switches).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpusCompareOptions {
    /// Channels of the comparison (`-s` selects 2; with 1 the reference is downmixed).
    pub nchannels: usize,
    /// Sampling rate of the test signal (`-r`): 8000, 12000, 16000, 24000 or 48000.
    pub rate: u32,
}

impl Default for OpusCompareOptions {
    fn default() -> Self {
        Self {
            nchannels: 1,
            rate: 48000,
        }
    }
}

/// Result of [`opus_compare`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpusCompareResult {
    /// Internal weighted error (`err`).
    pub err: f64,
    /// Opus quality metric in percent (`Q`).
    pub q: f32,
    /// Whether the test vector passes (C: `!(Q < 0)`).
    pub passes: bool,
}

/// Port of the analysis part of `src/opus_compare.c:main`.
///
/// `x` is the reference file decoded as **stereo** interleaved samples (the C tool always reads
/// file 1 as stereo and downmixes it for a mono comparison); `y` is the test signal with
/// `opts.nchannels` interleaved channels at `opts.rate`. Trailing partial frames are ignored.
///
/// # Errors
/// [`CompareError::Channels`], [`CompareError::OpusRate`], and the C tool's
/// [`CompareError::SampleCountMismatch`] and [`CompareError::InsufficientData`].
pub fn opus_compare(
    x: &[f32],
    y: &[f32],
    opts: &OpusCompareOptions,
) -> Result<OpusCompareResult, CompareError> {
    let nchannels = opts.nchannels;
    if nchannels != 1 && nchannels != 2 {
        return Err(CompareError::Channels(nchannels));
    }
    let rate = opts.rate;
    let mut ybands = NBANDS;
    let mut yfreqs = NFREQS;
    let mut downsample = 1usize;
    if rate != 48000 {
        if rate != 8000 && rate != 12000 && rate != 16000 && rate != 24000 {
            return Err(CompareError::OpusRate(rate));
        }
        downsample = (48000 / rate) as usize;
        ybands = match rate {
            8000 => 13,
            12000 => 15,
            16000 => 17,
            _ => 19,
        };
        yfreqs = NFREQS / downsample;
    }

    // Read in the data and allocate scratch space.
    let xlength = x.len() / 2;
    let mut x = x[..xlength * 2].to_vec();
    if nchannels == 1 {
        for xi in 0..xlength {
            x[xi] = (0.5 * f64::from(x[2 * xi] + x[2 * xi + 1])) as f32;
        }
    }
    let ylength = y.len() / nchannels;
    let y = &y[..ylength * nchannels];
    if xlength != ylength * downsample {
        return Err(CompareError::SampleCountMismatch {
            xlength,
            ylength_scaled: ylength * downsample,
        });
    }
    if xlength < TEST_WIN_SIZE {
        return Err(CompareError::InsufficientData {
            xlength,
            window: TEST_WIN_SIZE,
        });
    }
    let nframes = (xlength - TEST_WIN_SIZE + TEST_WIN_STEP) / TEST_WIN_STEP;
    let mut xb = vec![0f32; nframes * NBANDS * nchannels];
    let mut xs = vec![0f32; nframes * NFREQS * nchannels];
    let mut ys = vec![0f32; nframes * yfreqs * nchannels];
    // Compute the per-band spectral energy of the original signal and the error.
    band_energy(
        Some(&mut xb),
        &mut xs,
        &BANDS,
        NBANDS,
        &x,
        nchannels,
        nframes,
        TEST_WIN_SIZE,
        TEST_WIN_STEP,
        1,
    );
    drop(x);
    band_energy(
        None,
        &mut ys,
        &BANDS,
        ybands,
        y,
        nchannels,
        nframes,
        TEST_WIN_SIZE / downsample,
        TEST_WIN_STEP / downsample,
        downsample as i32,
    );
    for xi in 0..nframes {
        // Frequency masking (low to high): 10 dB/Bark slope.
        for bi in 1..NBANDS {
            for ci in 0..nchannels {
                xb[(xi * NBANDS + bi) * nchannels + ci] +=
                    0.1f32 * xb[(xi * NBANDS + bi - 1) * nchannels + ci];
            }
        }
        // Frequency masking (high to low): 15 dB/Bark slope.
        for bi in (0..NBANDS - 1).rev() {
            for ci in 0..nchannels {
                xb[(xi * NBANDS + bi) * nchannels + ci] +=
                    0.03f32 * xb[(xi * NBANDS + bi + 1) * nchannels + ci];
            }
        }
        if xi > 0 {
            // Temporal masking: -3 dB/2.5ms slope.
            for bi in 0..NBANDS {
                for ci in 0..nchannels {
                    xb[(xi * NBANDS + bi) * nchannels + ci] +=
                        0.5f32 * xb[((xi - 1) * NBANDS + bi) * nchannels + ci];
                }
            }
        }
        // Allowing some cross-talk
        if nchannels == 2 {
            for bi in 0..NBANDS {
                let l = xb[(xi * NBANDS + bi) * nchannels];
                let r = xb[(xi * NBANDS + bi) * nchannels + 1];
                xb[(xi * NBANDS + bi) * nchannels] += 0.01f32 * r;
                xb[(xi * NBANDS + bi) * nchannels + 1] += 0.01f32 * l;
            }
        }

        // Apply masking
        for bi in 0..ybands {
            for xj in BANDS[bi] as usize..BANDS[bi + 1] as usize {
                for ci in 0..nchannels {
                    xs[(xi * NFREQS + xj) * nchannels + ci] +=
                        0.1f32 * xb[(xi * NBANDS + bi) * nchannels + ci];
                    ys[(xi * yfreqs + xj) * nchannels + ci] +=
                        0.1f32 * xb[(xi * NBANDS + bi) * nchannels + ci];
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
                    let xtmp2 = xs[(xi * NFREQS + xj) * nchannels + ci];
                    let ytmp2 = ys[(xi * yfreqs + xj) * nchannels + ci];
                    xs[(xi * NFREQS + xj) * nchannels + ci] += xtmp;
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
    let max_compare = if rate == 48000 {
        BANDS[NBANDS]
    } else if rate == 12000 {
        BANDS[ybands]
    } else {
        BANDS[ybands] - 3
    };
    let mut err = 0f64;
    for xi in 0..nframes {
        let mut ef = 0f64;
        for bi in 0..ybands {
            let mut eb = 0f64;
            let mut xj = BANDS[bi];
            while xj < BANDS[bi + 1] && xj < max_compare {
                let j = xj as usize;
                for ci in 0..nchannels {
                    let re: f32 = ys[(xi * yfreqs + j) * nchannels + ci]
                        / xs[(xi * NFREQS + j) * nchannels + ci];
                    let mut im = (f64::from(re) - math::log(f64::from(re)) - 1.0) as f32;
                    // Make comparison less sensitive around the SILK/CELT cross-over to allow for
                    // mode freedom in the filters.
                    if (79..=81).contains(&xj) {
                        im *= 0.1f32;
                    }
                    if xj == 80 {
                        im *= 0.1f32;
                    }
                    eb += f64::from(im);
                }
                xj += 1;
            }
            eb /= f64::from((BANDS[bi + 1] - BANDS[bi]) * nchannels as i32);
            ef += eb * eb;
        }
        // Using a fixed normalization value means we're willing to accept slightly lower quality
        // for lower sampling rates.
        ef /= NBANDS as f64;
        ef *= ef;
        err += ef * ef;
    }
    err = math::pow(err / nframes as f64, 1.0 / 16.0);
    let q = (100.0 * (1.0 - 0.5 * math::log(1.0 + err) / math::log(1.13))) as f32;
    Ok(OpusCompareResult {
        err,
        q,
        passes: !q.lt(&0.0),
    })
}

/// Prints the `opus_compare` usage line.
fn usage(stderr: &mut dyn Write, args: &[String]) -> io::Result<i32> {
    writeln!(
        stderr,
        "Usage: {} [-s] [-r rate2] <file1.sw> <file2.sw>",
        argv0(args)
    )?;
    Ok(EXIT_FAILURE)
}

/// Port of `src/opus_compare.c:main`: command line `args` (including `argv[0]`), messages
/// written to `stderr`, returns the process exit code.
///
/// # Errors
/// I/O errors writing to `stderr` or reading an opened input file.
pub fn opus_compare_main(args: &[String], stderr: &mut dyn Write) -> io::Result<i32> {
    let argc = args.len();
    if !(3..=6).contains(&argc) {
        return usage(stderr, args);
    }
    let mut base = 0usize;
    let mut opts = OpusCompareOptions::default();
    if arg(args, base, 1) == Some("-s") {
        opts.nchannels = 2;
        base += 1;
    }
    if arg(args, base, 1) == Some("-r") {
        let Some(r) = arg(args, base, 2) else {
            return usage(stderr, args);
        };
        let rate = c_atoi(r) as u32;
        if rate != 8000 && rate != 12000 && rate != 16000 && rate != 24000 && rate != 48000 {
            writeln!(stderr, "{}", CompareError::OpusRate(rate))?;
            return Ok(EXIT_FAILURE);
        }
        opts.rate = rate;
        base += 2;
    }
    let (Some(path1), Some(path2)) = (arg(args, base, 1), arg(args, base, 2)) else {
        return usage(stderr, args);
    };
    let Some((b1, b2)) = open_inputs(stderr, path1, path2)? else {
        return Ok(EXIT_FAILURE);
    };
    let x = read_pcm(&b1, 2, SampleFormat::S16Le);
    drop(b1);
    let y = read_pcm(&b2, opts.nchannels, SampleFormat::S16Le);
    drop(b2);
    let res = match opus_compare(&x, &y, &opts) {
        Ok(r) => r,
        Err(e) => {
            writeln!(stderr, "{e}")?;
            return Ok(EXIT_FAILURE);
        }
    };
    if res.passes {
        writeln!(stderr, "Test vector PASSES")?;
        writeln!(
            stderr,
            "Opus quality metric: {} % (internal weighted error is {})",
            c_fmt_f(f64::from(res.q), 1),
            c_fmt_f(res.err, 6)
        )?;
        Ok(EXIT_SUCCESS)
    } else {
        writeln!(stderr, "Test vector FAILS")?;
        writeln!(stderr, "Internal weighted error is {}", c_fmt_f(res.err, 6))?;
        Ok(EXIT_FAILURE)
    }
}
