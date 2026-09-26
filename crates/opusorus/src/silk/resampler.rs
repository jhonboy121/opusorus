//! Port of the SILK resampler: `silk/resampler.c`, `silk/resampler_down2.c`,
//! `silk/resampler_down2_3.c`, `silk/resampler_private_AR2.c`,
//! `silk/resampler_private_IIR_FIR.c`, `silk/resampler_private_down_FIR.c`,
//! `silk/resampler_private_up2_HQ.c`, `silk/resampler_rom.c` and the headers
//! `silk/resampler_rom.h`, `silk/resampler_private.h`, `silk/resampler_structs.h`.
//!
//! Matrix of resampling methods used (from `resampler.c`):
//!
//! ```text
//!                                 Fs_out (kHz)
//!                        8      12     16     24     48
//!
//!               8        C      UF     U      UF     UF
//!              12        AF     C      UF     U      UF
//! Fs_in (kHz)  16        D      AF     C      UF     UF
//!              24        AF     D      AF     C      U
//!              48        AF     AF     AF     D      C
//!
//! C   -> Copy (no resampling)
//! D   -> Allpass-based 2x downsampling
//! U   -> Allpass-based 2x upsampling
//! UF  -> Allpass-based 2x upsampling followed by FIR interpolation
//! AF  -> AR2 filter followed by FIR interpolation
//! ```
//!
//! With `qext`, the encoder additionally accepts 96 kHz input and the decoder 96 kHz output.

use crate::Error;
use crate::celt::arch::imin;
use crate::silk::macros::{
    silk_add_lshift32, silk_add32, silk_div32, silk_div32_16, silk_lshift, silk_lshift32, silk_min,
    silk_mul, silk_rshift, silk_rshift_round, silk_rshift32, silk_sat16, silk_smlabb, silk_smlawb,
    silk_smulbb, silk_smulwb, silk_smulww, silk_sub32,
};

// ---------------------------------------------------------------------------------------------
// resampler_structs.h
// ---------------------------------------------------------------------------------------------

/// `SILK_RESAMPLER_MAX_FIR_ORDER`.
pub const SILK_RESAMPLER_MAX_FIR_ORDER: usize = 36;
/// `SILK_RESAMPLER_MAX_IIR_ORDER`.
pub const SILK_RESAMPLER_MAX_IIR_ORDER: usize = 6;

/// Length of `delayBuf` in `silk_resampler_state_struct`.
pub const SILK_RESAMPLER_DELAY_BUF_LEN: usize = 96;

/// `resampler_function` selector (`USE_silk_resampler_*` in `resampler.c`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum ResamplerFunction {
    /// `USE_silk_resampler_copy` (0): no resampling. This is also the value of a cleared state.
    #[default]
    Copy = 0,
    /// `USE_silk_resampler_private_up2_HQ_wrapper` (1).
    Up2Hq = 1,
    /// `USE_silk_resampler_private_IIR_FIR` (2).
    IirFir = 2,
    /// `USE_silk_resampler_private_down_FIR` (3).
    DownFir = 3,
}

impl ResamplerFunction {
    /// The C integer value (`USE_silk_resampler_*`).
    #[must_use]
    pub const fn to_c(self) -> i32 {
        self as i32
    }
}

/// `silk_resampler_state_struct`.
///
/// The C `sFIR` union (`opus_int32 i32[36]` / `opus_int16 i16[36]`) is modelled as two separate
/// arrays: `s_fir_i32` is the view used by `silk_resampler_private_down_FIR`, `s_fir_i16` the view
/// used by `silk_resampler_private_IIR_FIR`. A given state only ever uses one view (the state is
/// cleared by [`silk_resampler_init`] and the function selector never changes afterwards), so
/// the aliasing of the C union is never observable.
#[derive(Debug, Clone)]
pub struct SilkResamplerState {
    /// `sIIR`: IIR (allpass / AR2) filter state.
    pub s_iir: [i32; SILK_RESAMPLER_MAX_IIR_ORDER],
    /// `sFIR.i32`: FIR history of the down-FIR resampler.
    pub s_fir_i32: [i32; SILK_RESAMPLER_MAX_FIR_ORDER],
    /// `sFIR.i16`: FIR history of the IIR/FIR upsampler.
    pub s_fir_i16: [i16; SILK_RESAMPLER_MAX_FIR_ORDER],
    /// `delayBuf`: input delay compensation buffer.
    pub delay_buf: [i16; SILK_RESAMPLER_DELAY_BUF_LEN],
    /// `resampler_function`.
    pub resampler_function: ResamplerFunction,
    /// `batchSize`: number of input samples processed per inner batch.
    pub batch_size: i32,
    /// `invRatio_Q16`.
    pub inv_ratio_q16: i32,
    /// `FIR_Order`.
    pub fir_order: i32,
    /// `FIR_Fracs`.
    pub fir_fracs: i32,
    /// `Fs_in_kHz`.
    pub fs_in_khz: i32,
    /// `Fs_out_kHz`.
    pub fs_out_khz: i32,
    /// `inputDelay`.
    pub input_delay: i32,
    /// `Coefs`: IIR + FIR coefficient table (a `resampler_rom` table; empty when unused, which
    /// corresponds to the C `NULL`).
    pub coefs: &'static [i16],
}

impl Default for SilkResamplerState {
    /// The all-zero state (C `silk_memset( S, 0, sizeof( silk_resampler_state_struct ) )`).
    fn default() -> Self {
        Self {
            s_iir: [0; SILK_RESAMPLER_MAX_IIR_ORDER],
            s_fir_i32: [0; SILK_RESAMPLER_MAX_FIR_ORDER],
            s_fir_i16: [0; SILK_RESAMPLER_MAX_FIR_ORDER],
            delay_buf: [0; SILK_RESAMPLER_DELAY_BUF_LEN],
            resampler_function: ResamplerFunction::Copy,
            batch_size: 0,
            inv_ratio_q16: 0,
            fir_order: 0,
            fir_fracs: 0,
            fs_in_khz: 0,
            fs_out_khz: 0,
            input_delay: 0,
            coefs: &[],
        }
    }
}

impl SilkResamplerState {
    /// Creates a resampler for the given rates (see [`silk_resampler_init`]).
    ///
    /// # Errors
    /// [`Error::BadArg`] if the rate pair is not supported (C returns `-1`).
    pub fn new(fs_hz_in: i32, fs_hz_out: i32, for_enc: bool) -> crate::Result<Self> {
        let mut s = Self::default();
        if silk_resampler_init(&mut s, fs_hz_in, fs_hz_out, i32::from(for_enc)) != 0 {
            return Err(Error::BadArg);
        }
        Ok(s)
    }
}

// ---------------------------------------------------------------------------------------------
// resampler_private.h
// ---------------------------------------------------------------------------------------------

/// `RESAMPLER_MAX_BATCH_SIZE_MS`: number of milliseconds of input processed in the inner loop.
pub const RESAMPLER_MAX_BATCH_SIZE_MS: i32 = 10;
/// `RESAMPLER_MAX_FS_KHZ`.
pub const RESAMPLER_MAX_FS_KHZ: i32 = 48;
/// `RESAMPLER_MAX_BATCH_SIZE_IN`.
pub const RESAMPLER_MAX_BATCH_SIZE_IN: usize =
    (RESAMPLER_MAX_BATCH_SIZE_MS * RESAMPLER_MAX_FS_KHZ) as usize;

/// Largest `batchSize` any valid state can have: `Fs_in_kHz * RESAMPLER_MAX_BATCH_SIZE_MS` with
/// the largest supported input rate (48 kHz, or 96 kHz with `qext`). The C code sizes its
/// scratch buffers from `S->batchSize` with `ALLOC`; this bounds the equivalent stack arrays.
#[cfg(not(feature = "qext"))]
const MAX_BATCH_SIZE: usize = (RESAMPLER_MAX_BATCH_SIZE_MS * 48) as usize;
#[cfg(feature = "qext")]
const MAX_BATCH_SIZE: usize = (RESAMPLER_MAX_BATCH_SIZE_MS * 96) as usize;

// ---------------------------------------------------------------------------------------------
// resampler_rom.h / resampler_rom.c
// ---------------------------------------------------------------------------------------------

/// `RESAMPLER_DOWN_ORDER_FIR0`.
pub const RESAMPLER_DOWN_ORDER_FIR0: usize = 18;
/// `RESAMPLER_DOWN_ORDER_FIR1`.
pub const RESAMPLER_DOWN_ORDER_FIR1: usize = 24;
/// `RESAMPLER_DOWN_ORDER_FIR2`.
pub const RESAMPLER_DOWN_ORDER_FIR2: usize = 36;
/// `RESAMPLER_ORDER_FIR_12`.
pub const RESAMPLER_ORDER_FIR_12: usize = 8;

/// `silk_resampler_down2_0` (tables for the 2x downsampler).
pub const SILK_RESAMPLER_DOWN2_0: i16 = 9872;
/// `silk_resampler_down2_1`.
pub const SILK_RESAMPLER_DOWN2_1: i16 = (39809 - 65536) as i16;

/// `silk_resampler_up2_hq_0` (tables for the high-quality 2x upsampler).
pub const SILK_RESAMPLER_UP2_HQ_0: [i16; 3] = [1746, 14986, (39083 - 65536) as i16];
/// `silk_resampler_up2_hq_1`.
pub const SILK_RESAMPLER_UP2_HQ_1: [i16; 3] = [6854, 25769, (55542 - 65536) as i16];

/// `silk_Resampler_3_4_COEFS`: IIR + FIR coefficients for 3:4 downsampling.
#[rustfmt::skip]
pub static SILK_RESAMPLER_3_4_COEFS: [i16; 2 + 3 * RESAMPLER_DOWN_ORDER_FIR0 / 2] = [
    -20694, -13867,
       -49,     64,     17,   -157,    353,   -496,    163,  11047,  22205,
       -39,      6,     91,   -170,    186,     23,   -896,   6336,  19928,
       -19,    -36,    102,    -89,    -24,    328,   -951,   2568,  15909,
];

/// `silk_Resampler_2_3_COEFS`: IIR + FIR coefficients for 2:3 downsampling.
#[rustfmt::skip]
pub static SILK_RESAMPLER_2_3_COEFS: [i16; 2 + 2 * RESAMPLER_DOWN_ORDER_FIR0 / 2] = [
    -14457, -14019,
        64,    128,   -122,     36,    310,   -768,    584,   9267,  17733,
        12,    128,     18,   -142,    288,   -117,   -865,   4123,  14459,
];

/// `silk_Resampler_1_2_COEFS`: IIR + FIR coefficients for 1:2 downsampling.
#[rustfmt::skip]
pub static SILK_RESAMPLER_1_2_COEFS: [i16; 2 + RESAMPLER_DOWN_ORDER_FIR1 / 2] = [
       616, -14323,
       -10,     39,     58,    -46,    -84,    120,    184,   -315,   -541,   1284,   5380,   9024,
];

/// `silk_Resampler_1_3_COEFS`: IIR + FIR coefficients for 1:3 downsampling.
#[rustfmt::skip]
pub static SILK_RESAMPLER_1_3_COEFS: [i16; 2 + RESAMPLER_DOWN_ORDER_FIR2 / 2] = [
     16102, -15162,
       -13,      0,     20,     26,      5,    -31,    -43,     -4,     65,     90,      7,   -157,   -248,    -44,    593,   1583,   2612,   3271,
];

/// `silk_Resampler_1_4_COEFS`: IIR + FIR coefficients for 1:4 downsampling.
#[rustfmt::skip]
pub static SILK_RESAMPLER_1_4_COEFS: [i16; 2 + RESAMPLER_DOWN_ORDER_FIR2 / 2] = [
     22500, -15099,
         3,    -14,    -20,    -15,      2,     25,     37,     25,    -16,    -71,   -107,    -79,     50,    292,    623,    982,   1288,   1464,
];

/// `silk_Resampler_1_6_COEFS`: IIR + FIR coefficients for 1:6 downsampling.
#[rustfmt::skip]
pub static SILK_RESAMPLER_1_6_COEFS: [i16; 2 + RESAMPLER_DOWN_ORDER_FIR2 / 2] = [
     27540, -15257,
        17,     12,      8,      1,    -10,    -22,    -30,    -32,    -22,      3,     44,    100,    168,    243,    317,    381,    429,    455,
];

/// `silk_Resampler_2_3_COEFS_LQ`: low-quality 2:3 downsampler coefficients.
#[rustfmt::skip]
pub static SILK_RESAMPLER_2_3_COEFS_LQ: [i16; 2 + 2 * 2] = [
     -2797,  -6507,
      4697,  10739,
      1567,   8276,
];

/// `silk_resampler_frac_FIR_12`: interpolation fractions of 1/24, 3/24, 5/24, ... , 23/24.
#[rustfmt::skip]
pub static SILK_RESAMPLER_FRAC_FIR_12: [[i16; RESAMPLER_ORDER_FIR_12 / 2]; 12] = [
    [  189,  -600,   617, 30567 ],
    [  117,  -159, -1070, 29704 ],
    [   52,   221, -2392, 28276 ],
    [   -4,   529, -3350, 26341 ],
    [  -48,   758, -3956, 23973 ],
    [  -80,   905, -4235, 21254 ],
    [  -99,   972, -4222, 18278 ],
    [ -107,   967, -3957, 15143 ],
    [ -103,   896, -3487, 11950 ],
    [  -91,   773, -2865,  8798 ],
    [  -71,   611, -2143,  5784 ],
    [  -46,   425, -1375,  2996 ],
];

// ---------------------------------------------------------------------------------------------
// resampler.c
// ---------------------------------------------------------------------------------------------

/// `delay_matrix_enc`: delay compensation values to equalize total delay for different modes
/// (rows: input 8/12/16/24/48/96 kHz; columns: output 8/12/16 kHz).
#[rustfmt::skip]
static DELAY_MATRIX_ENC: [[i8; 3]; 6] = [
/* in  \ out  8  12  16 */
/*  8 */   [  6,  0,  3 ],
/* 12 */   [  0,  7,  3 ],
/* 16 */   [  0,  1, 10 ],
/* 24 */   [  0,  2,  6 ],
/* 48 */   [ 18, 10, 12 ],
/* 96 */   [  0,  0, 44 ],
];

/// `delay_matrix_dec` (rows: input 8/12/16 kHz; columns: output 8/12/16/24/48/96 kHz).
#[rustfmt::skip]
static DELAY_MATRIX_DEC: [[i8; 6]; 3] = [
/* in  \ out  8  12  16  24  48  96 */
/*  8 */   [  4,  0,  2,  0,  0,  0 ],
/* 12 */   [  0,  9,  4,  7,  4,  4 ],
/* 16 */   [  0,  3, 12,  7,  7,  7 ],
];

/// `rateID(R)`: maps `[8000, 12000, 16000, 24000, 48000, 96000]` to `[0, 1, 2, 3, 4, 5]`.
#[must_use]
pub const fn rate_id(r: i32) -> usize {
    imin(
        5,
        (((r >> 12) - (r > 16000) as i32) >> (r > 24000) as i32) - 1,
    ) as usize
}

/// Returns `true` if `fs` is an input rate accepted by the encoder-side resampler.
const fn enc_in_rate_ok(fs: i32) -> bool {
    #[cfg(feature = "qext")]
    if fs == 96000 {
        return true;
    }
    matches!(fs, 8000 | 12000 | 16000 | 24000 | 48000)
}

/// Returns `true` if `fs` is an output rate accepted by the decoder-side resampler.
const fn dec_out_rate_ok(fs: i32) -> bool {
    #[cfg(feature = "qext")]
    if fs == 96000 {
        return true;
    }
    matches!(fs, 8000 | 12000 | 16000 | 24000 | 48000)
}

/// Port of `silk/resampler.c:silk_resampler_init`.
///
/// Initializes/resets the resampler state for a given pair of input/output sampling rates.
/// `for_enc`: if nonzero, encoder rate table (API rate → 8/12/16 kHz); otherwise decoder table
/// (8/12/16 kHz → API rate). Returns `0` on success and `-1` on an unsupported rate pair
/// (as C; the state may be partially written in that case, exactly as in C).
///
/// The C code executes `celt_assert( 0 )` before returning `-1`; that assertion is only a
/// hard check with the feature `assertions`, so the `-1` path (what a C build without
/// assertions does) stays reachable otherwise.
pub fn silk_resampler_init(
    s: &mut SilkResamplerState,
    fs_hz_in: i32,
    fs_hz_out: i32,
    for_enc: i32,
) -> i32 {
    // Clear state
    *s = SilkResamplerState::default();

    // Input checking
    if for_enc != 0 {
        if !enc_in_rate_ok(fs_hz_in) || !matches!(fs_hz_out, 8000 | 12000 | 16000) {
            assertion_failure!("0");
            return -1;
        }
        s.input_delay = DELAY_MATRIX_ENC[rate_id(fs_hz_in)][rate_id(fs_hz_out)] as i32;
    } else {
        if !matches!(fs_hz_in, 8000 | 12000 | 16000) || !dec_out_rate_ok(fs_hz_out) {
            assertion_failure!("0");
            return -1;
        }
        s.input_delay = DELAY_MATRIX_DEC[rate_id(fs_hz_in)][rate_id(fs_hz_out)] as i32;
    }

    s.fs_in_khz = silk_div32_16(fs_hz_in, 1000);
    s.fs_out_khz = silk_div32_16(fs_hz_out, 1000);

    // Number of samples processed per batch
    s.batch_size = s.fs_in_khz * RESAMPLER_MAX_BATCH_SIZE_MS;

    // Find resampler with the right sampling ratio
    let mut up2x = 0;
    if fs_hz_out > fs_hz_in {
        // Upsample
        if fs_hz_out == silk_mul(fs_hz_in, 2) {
            // Fs_out : Fs_in = 2 : 1. Special case: directly use 2x upsampler
            s.resampler_function = ResamplerFunction::Up2Hq;
        } else {
            // Default resampler
            s.resampler_function = ResamplerFunction::IirFir;
            up2x = 1;
        }
    } else if fs_hz_out < fs_hz_in {
        // Downsample
        s.resampler_function = ResamplerFunction::DownFir;
        if silk_mul(fs_hz_out, 4) == silk_mul(fs_hz_in, 3) {
            // Fs_out : Fs_in = 3 : 4
            s.fir_fracs = 3;
            s.fir_order = RESAMPLER_DOWN_ORDER_FIR0 as i32;
            s.coefs = &SILK_RESAMPLER_3_4_COEFS;
        } else if silk_mul(fs_hz_out, 3) == silk_mul(fs_hz_in, 2) {
            // Fs_out : Fs_in = 2 : 3
            s.fir_fracs = 2;
            s.fir_order = RESAMPLER_DOWN_ORDER_FIR0 as i32;
            s.coefs = &SILK_RESAMPLER_2_3_COEFS;
        } else if silk_mul(fs_hz_out, 2) == fs_hz_in {
            // Fs_out : Fs_in = 1 : 2
            s.fir_fracs = 1;
            s.fir_order = RESAMPLER_DOWN_ORDER_FIR1 as i32;
            s.coefs = &SILK_RESAMPLER_1_2_COEFS;
        } else if silk_mul(fs_hz_out, 3) == fs_hz_in {
            // Fs_out : Fs_in = 1 : 3
            s.fir_fracs = 1;
            s.fir_order = RESAMPLER_DOWN_ORDER_FIR2 as i32;
            s.coefs = &SILK_RESAMPLER_1_3_COEFS;
        } else if silk_mul(fs_hz_out, 4) == fs_hz_in {
            // Fs_out : Fs_in = 1 : 4
            s.fir_fracs = 1;
            s.fir_order = RESAMPLER_DOWN_ORDER_FIR2 as i32;
            s.coefs = &SILK_RESAMPLER_1_4_COEFS;
        } else if silk_mul(fs_hz_out, 6) == fs_hz_in {
            // Fs_out : Fs_in = 1 : 6
            s.fir_fracs = 1;
            s.fir_order = RESAMPLER_DOWN_ORDER_FIR2 as i32;
            s.coefs = &SILK_RESAMPLER_1_6_COEFS;
        } else {
            // None available (C: celt_assert( 0 ), see the doc comment)
            assertion_failure!("0");
            return -1;
        }
    } else {
        // Input and output sampling rates are equal: copy
        s.resampler_function = ResamplerFunction::Copy;
    }

    // Ratio of input/output samples
    s.inv_ratio_q16 = silk_lshift32(silk_div32(silk_lshift32(fs_hz_in, 14 + up2x), fs_hz_out), 2);
    // Make sure the ratio is rounded up
    while silk_smulww(s.inv_ratio_q16, fs_hz_out) < silk_lshift32(fs_hz_in, up2x) {
        s.inv_ratio_q16 += 1;
    }

    0
}

/// Port of `silk/resampler.c:silk_resampler`.
///
/// Resamples `in_len` samples of `input` into `out` (which receives
/// `in_len * Fs_out_kHz / Fs_in_kHz` samples). `in_len` must be at least 1 ms of input
/// (`Fs_in_kHz` samples). Always returns `0` (as C).
pub fn silk_resampler(
    s: &mut SilkResamplerState,
    out: &mut [i16],
    input: &[i16],
    in_len: i32,
) -> i32 {
    // Need at least 1 ms of input data
    celt_assert!(in_len >= s.fs_in_khz);
    // Delay can't exceed the 1 ms of buffering
    celt_assert!(s.input_delay <= s.fs_in_khz);

    let fs_in = s.fs_in_khz as usize;
    let fs_out = s.fs_out_khz as usize;
    let input_delay = s.input_delay as usize;
    let n_samples = fs_in - input_delay;
    let in_len_u = in_len as usize;
    let fs_in_khz = s.fs_in_khz;

    // Copy to delay buffer
    s.delay_buf[input_delay..input_delay + n_samples].copy_from_slice(&input[..n_samples]);

    // The C code passes `S->delayBuf` as input while also passing `S`; the private resamplers
    // never touch `delayBuf`, so a copy of it is equivalent (and satisfies the borrow checker).
    let delay_buf = s.delay_buf;

    match s.resampler_function {
        ResamplerFunction::Up2Hq => {
            silk_resampler_private_up2_hq_wrapper(s, out, &delay_buf, fs_in_khz);
            silk_resampler_private_up2_hq_wrapper(
                s,
                &mut out[fs_out..],
                &input[n_samples..],
                in_len - fs_in_khz,
            );
        }
        ResamplerFunction::IirFir => {
            silk_resampler_private_iir_fir(s, out, &delay_buf, fs_in_khz);
            silk_resampler_private_iir_fir(
                s,
                &mut out[fs_out..],
                &input[n_samples..],
                in_len - fs_in_khz,
            );
        }
        ResamplerFunction::DownFir => {
            silk_resampler_private_down_fir(s, out, &delay_buf, fs_in_khz);
            silk_resampler_private_down_fir(
                s,
                &mut out[fs_out..],
                &input[n_samples..],
                in_len - fs_in_khz,
            );
        }
        ResamplerFunction::Copy => {
            out[..fs_in].copy_from_slice(&delay_buf[..fs_in]);
            let rest = in_len_u - fs_in;
            out[fs_out..fs_out + rest].copy_from_slice(&input[n_samples..n_samples + rest]);
        }
    }

    // Copy to delay buffer
    s.delay_buf[..input_delay].copy_from_slice(&input[in_len_u - input_delay..in_len_u]);

    0
}

// ---------------------------------------------------------------------------------------------
// resampler_down2.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/resampler_down2.c:silk_resampler_down2`.
///
/// Downsamples by a factor 2. `s`: state `[2]`; `out`: `floor(in_len / 2)` samples.
pub fn silk_resampler_down2(s: &mut [i32], out: &mut [i16], input: &[i16], in_len: i32) {
    let len2 = silk_rshift32(in_len, 1) as usize;
    let s = &mut s[..2];
    let out = &mut out[..len2];
    let input = &input[..2 * len2];

    const _: () = assert!(SILK_RESAMPLER_DOWN2_0 > 0);
    const _: () = assert!(SILK_RESAMPLER_DOWN2_1 < 0);

    // Internal variables and state are in Q10 format
    for (o, pair) in out.iter_mut().zip(input.as_chunks::<2>().0) {
        // Convert to Q10
        let mut in32 = silk_lshift(i32::from(pair[0]), 10);

        // All-pass section for even input sample
        let mut y = silk_sub32(in32, s[0]);
        let mut x = silk_smlawb(y, y, i32::from(SILK_RESAMPLER_DOWN2_1));
        let mut out32 = silk_add32(s[0], x);
        s[0] = silk_add32(in32, x);

        // Convert to Q10
        in32 = silk_lshift(i32::from(pair[1]), 10);

        // All-pass section for odd input sample, and add to output of previous section
        y = silk_sub32(in32, s[1]);
        x = silk_smulwb(y, i32::from(SILK_RESAMPLER_DOWN2_0));
        out32 = silk_add32(out32, s[1]);
        out32 = silk_add32(out32, x);
        s[1] = silk_add32(in32, x);

        // Add, convert back to int16 and store to output
        *o = silk_sat16(silk_rshift_round(out32, 11)) as i16;
    }
}

// ---------------------------------------------------------------------------------------------
// resampler_down2_3.c
// ---------------------------------------------------------------------------------------------

/// `ORDER_FIR` in `resampler_down2_3.c`.
const DOWN2_3_ORDER_FIR: usize = 4;

/// Port of `silk/resampler_down2_3.c:silk_resampler_down2_3`.
///
/// Downsamples by a factor 2/3, low quality. `s`: state `[6]`; `out`:
/// `floor(2 * in_len / 3)` samples.
pub fn silk_resampler_down2_3(s: &mut [i32], out: &mut [i16], input: &[i16], mut in_len: i32) {
    let mut buf = [0i32; RESAMPLER_MAX_BATCH_SIZE_IN + DOWN2_3_ORDER_FIR];
    let (s_fir, s_ar2) = s[..6].split_at_mut(DOWN2_3_ORDER_FIR);
    let coefs = &SILK_RESAMPLER_2_3_COEFS_LQ;

    // Copy buffered samples to start of buffer
    buf[..DOWN2_3_ORDER_FIR].copy_from_slice(s_fir);

    let mut in_pos = 0usize;
    let mut out_pos = 0usize;
    let mut n_samples_in;
    // Iterate over blocks of frameSizeIn input samples
    loop {
        n_samples_in = silk_min(in_len, RESAMPLER_MAX_BATCH_SIZE_IN as i32);
        let n = n_samples_in as usize;

        // Second-order AR filter (output in Q8)
        silk_resampler_private_ar2(
            s_ar2,
            &mut buf[DOWN2_3_ORDER_FIR..],
            &input[in_pos..],
            coefs,
            n_samples_in,
        );

        // Interpolate filtered signal
        let mut p = 0usize;
        let mut counter = n_samples_in;
        while counter > 2 {
            let b = &buf[p..p + 5];
            // Inner product
            let mut res_q6 = silk_smulwb(b[0], i32::from(coefs[2]));
            res_q6 = silk_smlawb(res_q6, b[1], i32::from(coefs[3]));
            res_q6 = silk_smlawb(res_q6, b[2], i32::from(coefs[5]));
            res_q6 = silk_smlawb(res_q6, b[3], i32::from(coefs[4]));

            // Scale down, saturate and store in output array
            out[out_pos] = silk_sat16(silk_rshift_round(res_q6, 6)) as i16;
            out_pos += 1;

            res_q6 = silk_smulwb(b[1], i32::from(coefs[4]));
            res_q6 = silk_smlawb(res_q6, b[2], i32::from(coefs[5]));
            res_q6 = silk_smlawb(res_q6, b[3], i32::from(coefs[3]));
            res_q6 = silk_smlawb(res_q6, b[4], i32::from(coefs[2]));

            // Scale down, saturate and store in output array
            out[out_pos] = silk_sat16(silk_rshift_round(res_q6, 6)) as i16;
            out_pos += 1;

            p += 3;
            counter -= 3;
        }

        in_pos += n;
        in_len -= n_samples_in;

        if in_len > 0 {
            // More iterations to do; copy last part of filtered signal to beginning of buffer
            buf.copy_within(n..n + DOWN2_3_ORDER_FIR, 0);
        } else {
            break;
        }
    }

    // Copy last part of filtered signal to the state for the next call
    let n = n_samples_in as usize;
    s_fir.copy_from_slice(&buf[n..n + DOWN2_3_ORDER_FIR]);
}

// ---------------------------------------------------------------------------------------------
// resampler_private_AR2.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/resampler_private_AR2.c:silk_resampler_private_AR2`.
///
/// Second order AR filter with single delay elements. `s`: state `[2]`; `a_q14`: AR
/// coefficients (first two entries used); `len` samples of output in Q8.
pub fn silk_resampler_private_ar2(
    s: &mut [i32],
    out_q8: &mut [i32],
    input: &[i16],
    a_q14: &[i16],
    len: i32,
) {
    let len = len.max(0) as usize;
    let s = &mut s[..2];
    let (a0, a1) = (i32::from(a_q14[0]), i32::from(a_q14[1]));
    for (o, &x) in out_q8[..len].iter_mut().zip(&input[..len]) {
        let mut out32 = silk_add_lshift32(s[0], i32::from(x), 8);
        *o = out32;
        out32 = silk_lshift(out32, 2);
        s[0] = silk_smlawb(s[1], out32, a0);
        s[1] = silk_smulwb(out32, a1);
    }
}

// ---------------------------------------------------------------------------------------------
// resampler_private_up2_HQ.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/resampler_private_up2_HQ.c:silk_resampler_private_up2_HQ`.
///
/// Upsamples by a factor 2, high quality. Uses 2nd order allpass filters for the 2x upsampling,
/// followed by a notch filter just above Nyquist. `s`: state `[6]`; `out`: `2 * len` samples.
pub fn silk_resampler_private_up2_hq(s: &mut [i32], out: &mut [i16], input: &[i16], len: i32) {
    let len = len.max(0) as usize;
    let s = &mut s[..6];
    let out = &mut out[..2 * len];
    let input = &input[..len];

    const _: () = assert!(SILK_RESAMPLER_UP2_HQ_0[0] > 0);
    const _: () = assert!(SILK_RESAMPLER_UP2_HQ_0[1] > 0);
    const _: () = assert!(SILK_RESAMPLER_UP2_HQ_0[2] < 0);
    const _: () = assert!(SILK_RESAMPLER_UP2_HQ_1[0] > 0);
    const _: () = assert!(SILK_RESAMPLER_UP2_HQ_1[1] > 0);
    const _: () = assert!(SILK_RESAMPLER_UP2_HQ_1[2] < 0);

    let c0 = SILK_RESAMPLER_UP2_HQ_0.map(i32::from);
    let c1 = SILK_RESAMPLER_UP2_HQ_1.map(i32::from);

    // Internal variables and state are in Q10 format
    for (o, &x) in out.as_chunks_mut::<2>().0.iter_mut().zip(input) {
        // Convert to Q10
        let in32 = silk_lshift(i32::from(x), 10);

        // First all-pass section for even output sample
        let mut y = silk_sub32(in32, s[0]);
        let mut xx = silk_smulwb(y, c0[0]);
        let mut out32_1 = silk_add32(s[0], xx);
        s[0] = silk_add32(in32, xx);

        // Second all-pass section for even output sample
        y = silk_sub32(out32_1, s[1]);
        xx = silk_smulwb(y, c0[1]);
        let mut out32_2 = silk_add32(s[1], xx);
        s[1] = silk_add32(out32_1, xx);

        // Third all-pass section for even output sample
        y = silk_sub32(out32_2, s[2]);
        xx = silk_smlawb(y, y, c0[2]);
        out32_1 = silk_add32(s[2], xx);
        s[2] = silk_add32(out32_2, xx);

        // Apply gain in Q15, convert back to int16 and store to output
        o[0] = silk_sat16(silk_rshift_round(out32_1, 10)) as i16;

        // First all-pass section for odd output sample
        y = silk_sub32(in32, s[3]);
        xx = silk_smulwb(y, c1[0]);
        out32_1 = silk_add32(s[3], xx);
        s[3] = silk_add32(in32, xx);

        // Second all-pass section for odd output sample
        y = silk_sub32(out32_1, s[4]);
        xx = silk_smulwb(y, c1[1]);
        out32_2 = silk_add32(s[4], xx);
        s[4] = silk_add32(out32_1, xx);

        // Third all-pass section for odd output sample
        y = silk_sub32(out32_2, s[5]);
        xx = silk_smlawb(y, y, c1[2]);
        out32_1 = silk_add32(s[5], xx);
        s[5] = silk_add32(out32_2, xx);

        // Apply gain in Q15, convert back to int16 and store to output
        o[1] = silk_sat16(silk_rshift_round(out32_1, 10)) as i16;
    }
}

/// Port of `silk/resampler_private_up2_HQ.c:silk_resampler_private_up2_HQ_wrapper`.
pub fn silk_resampler_private_up2_hq_wrapper(
    s: &mut SilkResamplerState,
    out: &mut [i16],
    input: &[i16],
    len: i32,
) {
    silk_resampler_private_up2_hq(&mut s.s_iir, out, input, len);
}

// ---------------------------------------------------------------------------------------------
// resampler_private_IIR_FIR.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/resampler_private_IIR_FIR.c:silk_resampler_private_IIR_FIR_INTERPOL`.
///
/// Interpolates the 2x-upsampled signal in `buf` and stores it in `out`. Returns the number of
/// output samples written (C returns the advanced `out` pointer).
#[must_use]
pub fn silk_resampler_private_iir_fir_interpol(
    out: &mut [i16],
    buf: &[i16],
    max_index_q16: i32,
    index_increment_q16: i32,
) -> usize {
    let mut n_out = 0usize;
    // Interpolate upsampled signal and store in output array
    let mut index_q16 = 0i32;
    while index_q16 < max_index_q16 {
        let table_index = silk_smulwb(index_q16 & 0xFFFF, 12) as usize;
        let b = &buf[(index_q16 >> 16) as usize..][..RESAMPLER_ORDER_FIR_12];
        let f = &SILK_RESAMPLER_FRAC_FIR_12[table_index];
        let g = &SILK_RESAMPLER_FRAC_FIR_12[11 - table_index];

        let mut res_q15 = silk_smulbb(i32::from(b[0]), i32::from(f[0]));
        res_q15 = silk_smlabb(res_q15, i32::from(b[1]), i32::from(f[1]));
        res_q15 = silk_smlabb(res_q15, i32::from(b[2]), i32::from(f[2]));
        res_q15 = silk_smlabb(res_q15, i32::from(b[3]), i32::from(f[3]));
        res_q15 = silk_smlabb(res_q15, i32::from(b[4]), i32::from(g[3]));
        res_q15 = silk_smlabb(res_q15, i32::from(b[5]), i32::from(g[2]));
        res_q15 = silk_smlabb(res_q15, i32::from(b[6]), i32::from(g[1]));
        res_q15 = silk_smlabb(res_q15, i32::from(b[7]), i32::from(g[0]));
        out[n_out] = silk_sat16(silk_rshift_round(res_q15, 15)) as i16;
        n_out += 1;
        index_q16 += index_increment_q16;
    }
    n_out
}

/// Port of `silk/resampler_private_IIR_FIR.c:silk_resampler_private_IIR_FIR`.
///
/// Upsamples using a combination of allpass-based 2x upsampling and FIR interpolation.
pub fn silk_resampler_private_iir_fir(
    s: &mut SilkResamplerState,
    out: &mut [i16],
    input: &[i16],
    mut in_len: i32,
) {
    let mut buf_storage = [0i16; 2 * MAX_BATCH_SIZE + RESAMPLER_ORDER_FIR_12];
    let buf = &mut buf_storage[..2 * s.batch_size as usize + RESAMPLER_ORDER_FIR_12];

    // Copy buffered samples to start of buffer
    buf[..RESAMPLER_ORDER_FIR_12].copy_from_slice(&s.s_fir_i16[..RESAMPLER_ORDER_FIR_12]);

    // Iterate over blocks of frameSizeIn input samples
    let index_increment_q16 = s.inv_ratio_q16;
    let mut in_pos = 0usize;
    let mut out_pos = 0usize;
    let mut n_samples_in;
    loop {
        n_samples_in = silk_min(in_len, s.batch_size);

        // Upsample 2x
        silk_resampler_private_up2_hq(
            &mut s.s_iir,
            &mut buf[RESAMPLER_ORDER_FIR_12..],
            &input[in_pos..],
            n_samples_in,
        );

        let max_index_q16 = silk_lshift32(n_samples_in, 16 + 1); // + 1 because 2x upsampling
        out_pos += silk_resampler_private_iir_fir_interpol(
            &mut out[out_pos..],
            buf,
            max_index_q16,
            index_increment_q16,
        );
        in_pos += n_samples_in as usize;
        in_len -= n_samples_in;

        if in_len > 0 {
            // More iterations to do; copy last part of filtered signal to beginning of buffer
            let n2 = (n_samples_in << 1) as usize;
            buf.copy_within(n2..n2 + RESAMPLER_ORDER_FIR_12, 0);
        } else {
            break;
        }
    }

    // Copy last part of filtered signal to the state for the next call
    let n2 = (n_samples_in << 1) as usize;
    s.s_fir_i16[..RESAMPLER_ORDER_FIR_12].copy_from_slice(&buf[n2..n2 + RESAMPLER_ORDER_FIR_12]);
}

// ---------------------------------------------------------------------------------------------
// resampler_private_down_FIR.c
// ---------------------------------------------------------------------------------------------

/// Port of `silk/resampler_private_down_FIR.c:silk_resampler_private_down_FIR_INTERPOL`.
///
/// `fir_coefs` is the FIR part of a `resampler_rom` table (C `&S->Coefs[ 2 ]`). Returns the
/// number of output samples written (C returns the advanced `out` pointer).
#[must_use]
pub fn silk_resampler_private_down_fir_interpol(
    out: &mut [i16],
    buf: &[i32],
    fir_coefs: &[i16],
    fir_order: i32,
    fir_fracs: i32,
    max_index_q16: i32,
    index_increment_q16: i32,
) -> usize {
    let mut n_out = 0usize;
    let c = |i: usize| i32::from(fir_coefs[i]);
    match fir_order as usize {
        RESAMPLER_DOWN_ORDER_FIR0 => {
            const HALF: usize = RESAMPLER_DOWN_ORDER_FIR0 / 2;
            let mut index_q16 = 0i32;
            while index_q16 < max_index_q16 {
                // Integer part gives pointer to buffered input
                let b = &buf[silk_rshift(index_q16, 16) as usize..][..RESAMPLER_DOWN_ORDER_FIR0];

                // Fractional part gives interpolation coefficients
                let interpol_ind = silk_smulwb(index_q16 & 0xFFFF, fir_fracs);

                // Inner product
                let ip = &fir_coefs[HALF * interpol_ind as usize..][..HALF];
                let mut res_q6 = silk_smulwb(b[0], i32::from(ip[0]));
                res_q6 = silk_smlawb(res_q6, b[1], i32::from(ip[1]));
                res_q6 = silk_smlawb(res_q6, b[2], i32::from(ip[2]));
                res_q6 = silk_smlawb(res_q6, b[3], i32::from(ip[3]));
                res_q6 = silk_smlawb(res_q6, b[4], i32::from(ip[4]));
                res_q6 = silk_smlawb(res_q6, b[5], i32::from(ip[5]));
                res_q6 = silk_smlawb(res_q6, b[6], i32::from(ip[6]));
                res_q6 = silk_smlawb(res_q6, b[7], i32::from(ip[7]));
                res_q6 = silk_smlawb(res_q6, b[8], i32::from(ip[8]));
                let ip = &fir_coefs[HALF * (fir_fracs - 1 - interpol_ind) as usize..][..HALF];
                res_q6 = silk_smlawb(res_q6, b[17], i32::from(ip[0]));
                res_q6 = silk_smlawb(res_q6, b[16], i32::from(ip[1]));
                res_q6 = silk_smlawb(res_q6, b[15], i32::from(ip[2]));
                res_q6 = silk_smlawb(res_q6, b[14], i32::from(ip[3]));
                res_q6 = silk_smlawb(res_q6, b[13], i32::from(ip[4]));
                res_q6 = silk_smlawb(res_q6, b[12], i32::from(ip[5]));
                res_q6 = silk_smlawb(res_q6, b[11], i32::from(ip[6]));
                res_q6 = silk_smlawb(res_q6, b[10], i32::from(ip[7]));
                res_q6 = silk_smlawb(res_q6, b[9], i32::from(ip[8]));

                // Scale down, saturate and store in output array
                out[n_out] = silk_sat16(silk_rshift_round(res_q6, 6)) as i16;
                n_out += 1;
                index_q16 += index_increment_q16;
            }
        }
        RESAMPLER_DOWN_ORDER_FIR1 => {
            let mut index_q16 = 0i32;
            while index_q16 < max_index_q16 {
                // Integer part gives pointer to buffered input
                let b = &buf[silk_rshift(index_q16, 16) as usize..][..RESAMPLER_DOWN_ORDER_FIR1];

                // Inner product
                let mut res_q6 = silk_smulwb(silk_add32(b[0], b[23]), c(0));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[1], b[22]), c(1));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[2], b[21]), c(2));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[3], b[20]), c(3));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[4], b[19]), c(4));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[5], b[18]), c(5));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[6], b[17]), c(6));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[7], b[16]), c(7));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[8], b[15]), c(8));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[9], b[14]), c(9));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[10], b[13]), c(10));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[11], b[12]), c(11));

                // Scale down, saturate and store in output array
                out[n_out] = silk_sat16(silk_rshift_round(res_q6, 6)) as i16;
                n_out += 1;
                index_q16 += index_increment_q16;
            }
        }
        RESAMPLER_DOWN_ORDER_FIR2 => {
            let mut index_q16 = 0i32;
            while index_q16 < max_index_q16 {
                // Integer part gives pointer to buffered input
                let b = &buf[silk_rshift(index_q16, 16) as usize..][..RESAMPLER_DOWN_ORDER_FIR2];

                // Inner product
                let mut res_q6 = silk_smulwb(silk_add32(b[0], b[35]), c(0));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[1], b[34]), c(1));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[2], b[33]), c(2));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[3], b[32]), c(3));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[4], b[31]), c(4));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[5], b[30]), c(5));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[6], b[29]), c(6));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[7], b[28]), c(7));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[8], b[27]), c(8));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[9], b[26]), c(9));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[10], b[25]), c(10));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[11], b[24]), c(11));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[12], b[23]), c(12));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[13], b[22]), c(13));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[14], b[21]), c(14));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[15], b[20]), c(15));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[16], b[19]), c(16));
                res_q6 = silk_smlawb(res_q6, silk_add32(b[17], b[18]), c(17));

                // Scale down, saturate and store in output array
                out[n_out] = silk_sat16(silk_rshift_round(res_q6, 6)) as i16;
                n_out += 1;
                index_q16 += index_increment_q16;
            }
        }
        _ => celt_assert!(false, "invalid FIR_Order {fir_order}"),
    }
    n_out
}

/// Port of `silk/resampler_private_down_FIR.c:silk_resampler_private_down_FIR`.
///
/// Resamples with a 2nd order AR filter followed by FIR interpolation.
///
/// Quirk (ported faithfully): the batch loop continues only while more than one input sample
/// remains (`inLen > 1`), so a trailing single sample after a full batch is not processed.
pub fn silk_resampler_private_down_fir(
    s: &mut SilkResamplerState,
    out: &mut [i16],
    input: &[i16],
    mut in_len: i32,
) {
    let fir_order = s.fir_order as usize;
    let mut buf_storage = [0i32; MAX_BATCH_SIZE + SILK_RESAMPLER_MAX_FIR_ORDER];
    let buf = &mut buf_storage[..s.batch_size as usize + fir_order];

    // Copy buffered samples to start of buffer
    buf[..fir_order].copy_from_slice(&s.s_fir_i32[..fir_order]);

    let coefs = s.coefs;
    let fir_coefs = &coefs[2..];

    // Iterate over blocks of frameSizeIn input samples
    let index_increment_q16 = s.inv_ratio_q16;
    let mut in_pos = 0usize;
    let mut out_pos = 0usize;
    let mut n_samples_in;
    loop {
        n_samples_in = silk_min(in_len, s.batch_size);

        // Second-order AR filter (output in Q8)
        silk_resampler_private_ar2(
            &mut s.s_iir,
            &mut buf[fir_order..],
            &input[in_pos..],
            coefs,
            n_samples_in,
        );

        let max_index_q16 = silk_lshift32(n_samples_in, 16);

        // Interpolate filtered signal
        out_pos += silk_resampler_private_down_fir_interpol(
            &mut out[out_pos..],
            buf,
            fir_coefs,
            s.fir_order,
            s.fir_fracs,
            max_index_q16,
            index_increment_q16,
        );

        in_pos += n_samples_in as usize;
        in_len -= n_samples_in;

        if in_len > 1 {
            // More iterations to do; copy last part of filtered signal to beginning of buffer
            let n = n_samples_in as usize;
            buf.copy_within(n..n + fir_order, 0);
        } else {
            break;
        }
    }

    // Copy last part of filtered signal to the state for the next call
    let n = n_samples_in as usize;
    s.s_fir_i32[..fir_order].copy_from_slice(&buf[n..n + fir_order]);
}
