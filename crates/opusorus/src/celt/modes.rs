//! Port of celt/modes.c, celt/modes.h: CELT mode lookup/creation.
//!
//! The mode/pulse-cache types and all static mode data live in
//! [`static_modes`](crate::celt::static_modes) (port of `static_modes_float.h`, or
//! `static_modes_fixed.h` in fixed-point builds). This module holds the mode constructor
//! (`opus_custom_mode_create`), the QEXT helper mode (`compute_qext_mode`) and, with
//! `custom-modes`, the dynamic mode creation helpers (fixed-point builds compute the
//! pre-emphasis and the `celt_coef` window in Q format, like `modes.c`).
//!
//! `opus_custom_mode_destroy` has no port: modes are either `'static` or owned values dropped by
//! Rust.

#[cfg(feature = "custom-modes")]
use alloc::{borrow::Cow, vec, vec::Vec};

#[cfg(feature = "custom-modes")]
use crate::celt::arch::{CeltCoef, OpusVal16};
#[cfg(all(feature = "custom-modes", feature = "fixed-point"))]
use crate::celt::arch::{SIG_SHIFT, min32, qconst16};
#[cfg(feature = "custom-modes")]
use crate::celt::static_modes::{BAND_ALLOCATION, EBAND5MS, MdctLookup, PulseCache};
use crate::celt::static_modes::{CeltMode, STATIC_MODE_LIST};
use crate::{Error, Result};

/// `MAX_PERIOD`.
pub const MAX_PERIOD: i32 = 1024;
/// `DEC_PITCH_BUF_SIZE`.
pub const DEC_PITCH_BUF_SIZE: i32 = 2048;

/// `QEXT_PACKET_SIZE_CAP`.
#[cfg(feature = "qext")]
pub const QEXT_PACKET_SIZE_CAP: i32 = 3825;
/// `NB_QEXT_BANDS`.
#[cfg(feature = "qext")]
pub const NB_QEXT_BANDS: i32 = 14;

/// `SIG_SHIFT` (`arch.h`): only a `QCONST16` argument in the float build, where it is ignored.
#[cfg(all(feature = "custom-modes", not(feature = "fixed-point")))]
const SIG_SHIFT: i32 = 12;

/// `QCONST16(x, bits)` of an `f`-suffixed pre-emphasis literal: `x` itself in the float build.
#[cfg(all(feature = "custom-modes", not(feature = "fixed-point")))]
const fn preemph_const(x: f32, _bits: i32) -> OpusVal16 {
    x
}

/// `QCONST16(x, bits)` of an `f`-suffixed pre-emphasis literal (fixed-point: Q`bits`).
#[cfg(all(feature = "custom-modes", feature = "fixed-point"))]
const fn preemph_const(x: f32, bits: i32) -> OpusVal16 {
    qconst16(x as f64, bits)
}

/// `BITALLOC_SIZE`: number of rows of `band_allocation`.
#[cfg(feature = "custom-modes")]
const BITALLOC_SIZE: i32 = 11;

/// Port of celt/modes.c:opus_custom_mode_create — the static-mode lookup path.
///
/// Returns the statically defined mode whose sampling rate is `fs` and whose frame size is
/// `frame_size` times a power of two (`<< 0..=3`). Anything else is `Err(Error::BadArg)`, exactly
/// as the C function without `CUSTOM_MODES`. With `custom-modes`, see
/// [`opus_custom_mode_create_custom`] for the full (dynamic) behaviour.
pub fn opus_custom_mode_create(fs: i32, frame_size: i32) -> Result<&'static CeltMode> {
    for m in STATIC_MODE_LIST.iter() {
        for j in 0..4 {
            if fs == m.fs && (frame_size << j) == m.short_mdct_size * m.nb_short_mdcts {
                return Ok(m);
            }
        }
    }
    Err(Error::BadArg)
}

// ---------------------------------------------------------------------------------------------
// QEXT
// ---------------------------------------------------------------------------------------------

/// `qext_eBands_180`.
#[cfg(feature = "qext")]
static QEXT_EBANDS_180: [i16; 15] = [
    74, 82, 90, 98, 106, 114, 122, 130, 138, 146, 154, 162, 168, 174, 180,
];
/// `qext_logN_180`.
#[cfg(feature = "qext")]
static QEXT_LOGN_180: [i16; 15] = [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 21, 21, 21];
/// `qext_eBands_240`: extra bands.
#[cfg(feature = "qext")]
static QEXT_EBANDS_240: [i16; 15] = [
    100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
];
/// `qext_logN_240`.
#[cfg(feature = "qext")]
static QEXT_LOGN_240: [i16; 14] = [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27];

/// Whether [`compute_qext_mode`] supports `m` (C `celt_assert(0)`s otherwise): a short MDCT of
/// 2.5 ms (`qext_eBands_240`) or 1.875 ms (`qext_eBands_180`).
#[cfg(feature = "qext")]
#[must_use]
pub const fn qext_mode_supported(m: &CeltMode) -> bool {
    m.short_mdct_size * 48000 == 120 * m.fs || m.short_mdct_size * 48000 == 90 * m.fs
}

/// Port of celt/modes.c:compute_qext_mode.
///
/// Returns a copy of `m` describing the QEXT extra bands (above 20 kHz), with `cache` set to
/// `m.qext_cache`. Cheap for static modes (all tables are borrowed). Like C (whose
/// `celt_assert(0)` is compiled out), an unsupported mode (see [`qext_mode_supported`]) keeps
/// the bands of `m`; debug builds assert here.
#[cfg(feature = "qext")]
#[must_use]
pub fn compute_qext_mode(m: &CeltMode) -> CeltMode {
    debug_assert!(
        qext_mode_supported(m),
        "compute_qext_mode: unsupported mode"
    );
    compute_qext_mode_unchecked(m)
}

/// [`compute_qext_mode`] without its debug assertion, for callers that compute the QEXT mode
/// ahead of time (the CELT encoder, at init) and assert where C calls `compute_qext_mode`.
#[cfg(feature = "qext")]
#[must_use]
pub fn compute_qext_mode_unchecked(m: &CeltMode) -> CeltMode {
    use alloc::borrow::Cow;

    let mut qext = m.clone();
    if m.short_mdct_size * 48000 == 120 * m.fs {
        qext.e_bands = Cow::Borrowed(&QEXT_EBANDS_240);
        qext.log_n = Cow::Borrowed(&QEXT_LOGN_240);
    } else if m.short_mdct_size * 48000 == 90 * m.fs {
        qext.e_bands = Cow::Borrowed(&QEXT_EBANDS_180);
        qext.log_n = Cow::Borrowed(&QEXT_LOGN_180);
    }
    qext.nb_ebands = NB_QEXT_BANDS;
    qext.eff_ebands = NB_QEXT_BANDS;
    while i32::from(qext.e_bands[qext.eff_ebands as usize]) > qext.short_mdct_size {
        qext.eff_ebands -= 1;
    }
    qext.nb_alloc_vectors = 0;
    qext.alloc_vectors = Cow::Borrowed(&[]);
    qext.cache = m.qext_cache.clone();
    qext
}

// ---------------------------------------------------------------------------------------------
// CUSTOM_MODES
// ---------------------------------------------------------------------------------------------

/// `BARK_BANDS`: 25 critical bands for the full 0-20 kHz audio bandwidth.
#[cfg(feature = "custom-modes")]
const BARK_BANDS: usize = 25;

/// `bark_freq`: critical band edges, taken from
/// <http://ccrma.stanford.edu/~jos/bbt/Bark_Frequency_Scale.html>.
#[cfg(feature = "custom-modes")]
pub static BARK_FREQ: [i16; BARK_BANDS + 1] = [
    0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150,
    3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20000,
];

/// Port of celt/modes.c:compute_ebands.
///
/// Returns the band edges (`nbEBands + 1` meaningful entries; the vector may hold one more,
/// unspecified, entry exactly like the C allocation) and `nbEBands`.
#[cfg(feature = "custom-modes")]
#[must_use]
pub fn compute_ebands(fs: i32, frame_size: i32, res: i32) -> (Vec<i16>, i32) {
    let bark = |i: i32| i32::from(BARK_FREQ[i as usize]);

    // All modes that have 2.5 ms short blocks use the same definition
    if fs == 400 * frame_size {
        let nb = EBAND5MS.len() as i32 - 1;
        return (EBAND5MS.to_vec(), nb);
    }
    // Find the number of critical bands supported by our sampling rate
    let mut n_bark: i32 = 1;
    while n_bark < BARK_BANDS as i32 {
        if bark(n_bark + 1) * 2 >= fs {
            break;
        }
        n_bark += 1;
    }

    // Find where the linear part ends (i.e. where the spacing is more than min_width
    let mut lin: i32 = 0;
    while lin < n_bark {
        if bark(lin + 1) - bark(lin) >= res {
            break;
        }
        lin += 1;
    }

    let low = (bark(lin) + res / 2) / res;
    let high = n_bark - lin;
    let mut nb = low + high;
    let mut e: Vec<i16> = vec![0; (nb + 2) as usize];
    let mut offset: i32 = 0;

    // Linear spacing (min_width)
    for i in 0..low {
        e[i as usize] = i as i16;
    }
    if low > 0 {
        offset = i32::from(e[(low - 1) as usize]) * res - bark(lin - 1);
    }
    // Spacing follows critical bands
    for i in 0..high {
        let target = bark(lin + i);
        // Round to an even value
        e[(i + low) as usize] = ((target + offset / 2 + res) / (2 * res) * 2) as i16;
        offset = i32::from(e[(i + low) as usize]) * res - target;
    }
    // Enforce the minimum spacing at the boundary
    for i in 0..nb {
        if i32::from(e[i as usize]) < i {
            e[i as usize] = i as i16;
        }
    }
    // Round to an even value
    e[nb as usize] = ((bark(n_bark) + res) / (2 * res) * 2) as i16;
    if i32::from(e[nb as usize]) > frame_size {
        e[nb as usize] = frame_size as i16;
    }
    for i in 1..nb - 1 {
        let (a, b, c) = (
            i32::from(e[(i - 1) as usize]),
            i32::from(e[i as usize]),
            i32::from(e[(i + 1) as usize]),
        );
        if c - b < b - a {
            e[i as usize] = (b - (2 * b - a - c) / 2) as i16;
        }
    }
    // Remove any empty bands.
    let mut j: i32 = 0;
    for i in 0..nb {
        if e[(i + 1) as usize] > e[j as usize] {
            j += 1;
            e[j as usize] = e[(i + 1) as usize];
        }
    }
    nb = j;

    // C then checks, with `celt_assert` (only in ENABLE_ASSERTIONS builds), that every band is
    // no wider than the last one and at most twice as wide as the previous one. Those checks
    // are not ported: they fail for some modes that default libopus builds accept, e.g.
    // `opus_custom_mode_create(12000, 208)` (bands `..., 16, 20, 22`), where a debug assertion
    // would turn a valid call into a panic.

    (e, nb)
}

/// Port of celt/modes.c:compute_allocation_table.
///
/// Returns `allocVectors` (`BITALLOC_SIZE * nbEBands` entries) for a mode with the given band
/// layout; `nbAllocVectors` is always `BITALLOC_SIZE`.
#[cfg(feature = "custom-modes")]
#[must_use]
pub fn compute_allocation_table(
    fs: i32,
    short_mdct_size: i32,
    e_bands: &[i16],
    nb_ebands: i32,
) -> Vec<u8> {
    let max_bands = EBAND5MS.len() as i32 - 1;
    let mut alloc_vectors: Vec<u8> = vec![0; (BITALLOC_SIZE * nb_ebands) as usize];

    // Check for standard mode
    if fs == 400 * short_mdct_size {
        let n = (BITALLOC_SIZE * nb_ebands) as usize;
        alloc_vectors.copy_from_slice(&BAND_ALLOCATION[..n]);
        return alloc_vectors;
    }
    // If not the standard mode, interpolate
    // Compute per-codec-band allocation from per-critical-band matrix
    for i in 0..BITALLOC_SIZE {
        for j in 0..nb_ebands {
            let ebj = i32::from(e_bands[j as usize]) * fs / short_mdct_size;
            let mut k = 0;
            while k < max_bands {
                if 400 * i32::from(EBAND5MS[k as usize]) > ebj {
                    break;
                }
                k += 1;
            }
            let ba = |idx: i32| i32::from(BAND_ALLOCATION[idx as usize]);
            let idx = (i * nb_ebands + j) as usize;
            if k > max_bands - 1 {
                alloc_vectors[idx] = ba(i * max_bands + max_bands - 1) as u8;
            } else {
                let a1 = ebj - 400 * i32::from(EBAND5MS[(k - 1) as usize]);
                let a0 = 400 * i32::from(EBAND5MS[k as usize]) - ebj;
                alloc_vectors[idx] = ((a0 * ba(i * max_bands + k - 1) + a1 * ba(i * max_bands + k))
                    / (a0 + a1)) as u8;
            }
        }
    }
    alloc_vectors
}

/// Port of celt/modes.c:opus_custom_mode_create with `CUSTOM_MODES`.
///
/// Returns a static mode when one matches (see [`opus_custom_mode_create`]), otherwise creates
/// a custom mode. Errors are `BadArg` for unsupported parameters and `AllocFail` where C would
/// jump to `failure`.
#[cfg(feature = "custom-modes")]
pub fn opus_custom_mode_create_custom(fs: i32, frame_size: i32) -> Result<Cow<'static, CeltMode>> {
    opus_custom_mode_create_with(fs, frame_size, crate::celt::mdct::clt_mdct_init)
}

/// Port of celt/modes.c:opus_custom_mode_create with `CUSTOM_MODES`, with the MDCT set-up
/// (`clt_mdct_init(&mode->mdct, N, maxLM)`, returning an error on failure) supplied by the caller.
#[cfg(feature = "custom-modes")]
pub fn opus_custom_mode_create_with(
    fs: i32,
    frame_size: i32,
    mdct_init: impl FnOnce(i32, i32) -> Result<MdctLookup>,
) -> Result<Cow<'static, CeltMode>> {
    use crate::celt::cwrs::log2_frac;
    use crate::celt::entcode::BITRES;
    use crate::celt::rate::compute_pulse_cache;
    use crate::math;

    // CUSTOM_MODES_ONLY: not supported (static modes are always compiled in).
    if let Ok(m) = opus_custom_mode_create(fs, frame_size) {
        return Ok(Cow::Borrowed(m));
    }

    // The good thing here is that permutation of the arguments will automatically be invalid

    if !(8000..=96000).contains(&fs) {
        return Err(Error::BadArg);
    }
    #[cfg(feature = "qext")]
    let max_frame = 2048;
    #[cfg(not(feature = "qext"))]
    let max_frame = 1024;
    if !(40..=max_frame).contains(&frame_size) || frame_size % 2 != 0 {
        return Err(Error::BadArg);
    }
    // Frames of less than 1ms are not supported.
    if frame_size * 1000 < fs {
        return Err(Error::BadArg);
    }

    let lm = if frame_size * 75 >= fs && (frame_size % 16) == 0 {
        3
    } else if frame_size * 150 >= fs && (frame_size % 8) == 0 {
        2
    } else if frame_size * 300 >= fs && (frame_size % 4) == 0 {
        1
    } else {
        0
    };

    // Shorts longer than 3.3ms are not supported.
    if (frame_size >> lm) * 300 > fs {
        return Err(Error::BadArg);
    }

    // Pre/de-emphasis depends on sampling rate. The "standard" pre-emphasis is defined as
    // A(z) = 1 - 0.85*z^-1 at 48 kHz. Other rates should approximate that.
    let preemph: [OpusVal16; 4] = if cfg!(feature = "qext") && fs == 96000 {
        // 96 kHz
        [
            preemph_const(0.9230041504, 15),
            preemph_const(0.2200012207, 15),
            preemph_const(1.5128347184, SIG_SHIFT), // exact 1/preemph[3]
            preemph_const(0.6610107422, 13),
        ]
    } else if fs < 12000 {
        // 8 kHz
        [
            preemph_const(0.3500061035, 15),
            -preemph_const(0.1799926758, 15),
            preemph_const(0.2719968125, SIG_SHIFT), // exact 1/preemph[3]
            preemph_const(3.6765136719, 13),
        ]
    } else if fs < 24000 {
        // 16 kHz
        [
            preemph_const(0.6000061035, 15),
            -preemph_const(0.1799926758, 15),
            preemph_const(0.4424998650, SIG_SHIFT), // exact 1/preemph[3]
            preemph_const(2.2598876953, 13),
        ]
    } else if fs < 40000 {
        // 32 kHz
        [
            preemph_const(0.7799987793, 15),
            -preemph_const(0.1000061035, 15),
            preemph_const(0.7499771125, SIG_SHIFT), // exact 1/preemph[3]
            preemph_const(1.3333740234, 13),
        ]
    } else {
        // 48 kHz
        [
            preemph_const(0.8500061035, 15),
            preemph_const(0.0, 15),
            preemph_const(1.0, SIG_SHIFT),
            preemph_const(1.0, 13),
        ]
    };

    let max_lm = lm;
    let nb_short_mdcts = 1 << lm;
    let short_mdct_size = frame_size / nb_short_mdcts;
    let res = (fs + short_mdct_size) / (2 * short_mdct_size);

    let (e_bands, nb_ebands) = compute_ebands(fs, short_mdct_size, res);
    // Not SMALL_FOOTPRINT: make sure we don't allocate a band larger than our PVQ table.
    // 208 should be enough, but let's be paranoid.
    if (i32::from(e_bands[nb_ebands as usize]) - i32::from(e_bands[(nb_ebands - 1) as usize])) << lm
        > 208
    {
        return Err(Error::AllocFail);
    }

    let mut eff_ebands = nb_ebands;
    while i32::from(e_bands[eff_ebands as usize]) > short_mdct_size {
        eff_ebands -= 1;
    }

    // Overlap must be divisible by 4
    let overlap = (short_mdct_size >> 2) << 2;

    let alloc_vectors = compute_allocation_table(fs, short_mdct_size, &e_bands, nb_ebands);

    let mut window: Vec<CeltCoef> = Vec::with_capacity(overlap as usize);
    for i in 0..overlap {
        let pi = core::f64::consts::PI; // M_PI
        let a = math::sin(0.5 * pi * (f64::from(i) + 0.5) / f64::from(overlap));
        let w = math::sin(0.5 * pi * a * a);
        // Float: `Q15ONE*sin(...)`.
        #[cfg(not(feature = "fixed-point"))]
        window.push((1.0f64 * w) as f32);
        // Fixed QEXT (Q31 `celt_coef`): `MIN32(2147483647, 2147483648*sin(...))`, a double
        // truncated by the conversion.
        #[cfg(all(feature = "fixed-point", feature = "qext"))]
        window.push(min32(2147483647.0f64, 2147483648.0 * w) as i32);
        // Fixed (Q15 `celt_coef`): `MIN32(32767, floor(.5+32768.*sin(...)))`.
        #[cfg(all(feature = "fixed-point", not(feature = "qext")))]
        window.push(min32(32767.0f64, math::floor(0.5 + 32768.0 * w)) as i16);
    }

    let mut log_n: Vec<i16> = Vec::with_capacity(nb_ebands as usize);
    for i in 0..nb_ebands as usize {
        log_n.push(log2_frac(
            (i32::from(e_bands[i + 1]) - i32::from(e_bands[i])) as u32,
            BITRES,
        ) as i16);
    }

    // C jumps to `failure` (OPUS_ALLOC_FAIL) whenever clt_mdct_init fails, whatever the cause.
    let mdct =
        mdct_init(2 * short_mdct_size * nb_short_mdcts, max_lm).map_err(|_| Error::AllocFail)?;

    let empty_cache = PulseCache {
        size: 0,
        index: Cow::Borrowed(&[]),
        bits: Cow::Borrowed(&[]),
        caps: Cow::Borrowed(&[]),
    };
    let mut mode = CeltMode {
        fs,
        overlap,
        nb_ebands,
        eff_ebands,
        preemph,
        e_bands: Cow::Owned(e_bands),
        max_lm,
        nb_short_mdcts,
        short_mdct_size,
        nb_alloc_vectors: BITALLOC_SIZE,
        alloc_vectors: Cow::Owned(alloc_vectors),
        log_n: Cow::Owned(log_n),
        window: Cow::Owned(window),
        mdct,
        cache: empty_cache.clone(),
        #[cfg(feature = "qext")]
        qext_cache: empty_cache,
    };

    compute_pulse_cache(&mut mode, max_lm);
    #[cfg(feature = "qext")]
    {
        if (mode.fs == 48000 && (mode.short_mdct_size == 120 || mode.short_mdct_size == 90))
            || (mode.fs == 96000 && (mode.short_mdct_size == 240 || mode.short_mdct_size == 180))
        {
            let mut dummy = compute_qext_mode(&mode);
            let dummy_lm = dummy.max_lm;
            compute_pulse_cache(&mut dummy, dummy_lm);
            mode.qext_cache = dummy.cache;
        }
    }

    Ok(Cow::Owned(mode))
}
