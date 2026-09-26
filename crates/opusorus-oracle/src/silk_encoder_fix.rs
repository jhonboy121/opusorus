//! Oracle bindings for unit `silk_encoder_fix`: the fixed-point SILK encoder
//! (`silk/fixed/*.c`, `silk/enc_API.c`, `silk/init_encoder.c`, `silk/control_codec.c` in a
//! `FIXED_POINT` build), see `csrc/silk_encoder_fix.c`. Fixed-point oracle only.
//!
//! * [`SilkEnc`] owns a C `silk_encoder` plus a range encoder over a private buffer, so tests
//!   can drive `silk_Encode` call by call exactly like `src/opus_encoder.c`, and dump the
//!   complete encoder state.
//! * Free functions wrap the exported `silk/fixed` leaf functions (declared directly) and the
//!   shims for static / state-based helpers. Every wrapper checks buffer sizes with `assert!`
//!   before calling into C.

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "wrappers mirror the C signatures one-to-one"
)]

use core::ffi::{c_int, c_uint, c_void};
use core::ptr::NonNull;

pub use crate::silk_decoder::OpusRes;

unsafe extern "C" {
    fn oracle_sefx_ctl_size() -> c_int;
    fn oracle_sefx_res_size() -> c_int;
    fn oracle_sefx_sizes(out: *mut c_int);
    fn oracle_sefx_new() -> *mut c_void;
    fn oracle_sefx_free(p: *mut c_void);
    fn oracle_sefx_init(p: *mut c_void, channels: c_int, status: *mut c_int) -> c_int;
    fn oracle_sefx_ec_init(p: *mut c_void, size: c_int);
    fn oracle_sefx_ec_state(p: *mut c_void, out: *mut c_uint);
    fn oracle_sefx_ec_done(p: *mut c_void);
    fn oracle_sefx_ec_buf(p: *mut c_void, out: *mut u8, n: c_int);
    fn oracle_sefx_encode(
        p: *mut c_void,
        ctl: *mut c_int,
        input: *const OpusRes,
        n: c_int,
        nbytes: *mut c_int,
        prefill: c_int,
        activity: c_int,
        null_ec: c_int,
    ) -> c_int;
    fn oracle_sefx_dump(p: *mut c_void, out: *mut c_int, cap: c_int) -> c_int;

    fn oracle_sefx_warped_gain(coefs: *const i32, lambda_q16: c_int, order: c_int) -> c_int;
    fn oracle_sefx_limit_warped_coefs(
        coefs: *mut i32,
        lambda_q16: c_int,
        limit_q24: c_int,
        order: c_int,
    );
    fn oracle_sefx_calc_corr_st3(
        out: *mut i32,
        frame: *const i16,
        start_lag: c_int,
        sf_length: c_int,
        nb_subfr: c_int,
        complexity: c_int,
    );
    fn oracle_sefx_calc_energy_st3(
        out: *mut i32,
        frame: *const i16,
        start_lag: c_int,
        sf_length: c_int,
        nb_subfr: c_int,
        complexity: c_int,
    );
    fn oracle_sefx_pitch_analysis_core(
        frame: *const i16,
        pitch_out: *mut c_int,
        lag_index: *mut i16,
        contour_index: *mut i8,
        ltp_corr_q15: *mut c_int,
        prev_lag: c_int,
        thres1_q16: c_int,
        thres2_q13: c_int,
        fs_khz: c_int,
        complexity: c_int,
        nb_subfr: c_int,
    ) -> c_int;
    fn oracle_sefx_find_lpc(
        order: c_int,
        subfr_length: c_int,
        nb_subfr: c_int,
        use_interp: c_int,
        first_frame: c_int,
        prev_nlsf: *const i16,
        nlsf_out: *mut i16,
        x: *const i16,
        min_inv_gain_q30: c_int,
    ) -> c_int;
    fn oracle_sefx_residual_energy(
        nrgs: *mut c_int,
        nrgs_q: *mut c_int,
        x: *const i16,
        a_q12: *const i16,
        gains: *const c_int,
        subfr_length: c_int,
        nb_subfr: c_int,
        lpc_order: c_int,
    );

    // Exported silk/fixed functions (fixed-point build, `arch` = 0).
    fn silk_apply_sine_window(px_win: *mut i16, px: *const i16, win_type: c_int, length: c_int);
    fn silk_autocorr(
        results: *mut i32,
        scale: *mut c_int,
        input: *const i16,
        input_size: c_int,
        correlation_count: c_int,
        arch: c_int,
    );
    fn silk_burg_modified_c(
        res_nrg: *mut i32,
        res_nrg_q: *mut c_int,
        a_q16: *mut i32,
        x: *const i16,
        min_inv_gain_q30: i32,
        subfr_length: c_int,
        nb_subfr: c_int,
        d: c_int,
        arch: c_int,
    );
    fn silk_k2a(a_q24: *mut i32, rc_q15: *const i16, order: i32);
    fn silk_k2a_Q16(a_q24: *mut i32, rc_q16: *const i32, order: i32);
    fn silk_schur(rc_q15: *mut i16, c: *const i32, order: i32) -> i32;
    fn silk_schur64(rc_q16: *mut i32, c: *const i32, order: i32) -> i32;
    fn silk_scale_copy_vector16(out: *mut i16, input: *const i16, gain_q16: i32, n: c_int);
    fn silk_scale_vector32_Q26_lshift_18(data: *mut i32, gain_q26: i32, n: c_int);
    fn silk_inner_prod_aligned(a: *const i16, b: *const i16, len: c_int, arch: c_int) -> i32;
    fn silk_corrVector_FIX(
        x: *const i16,
        t: *const i16,
        l: c_int,
        order: c_int,
        xt: *mut i32,
        rshifts: c_int,
        arch: c_int,
    );
    fn silk_corrMatrix_FIX(
        x: *const i16,
        l: c_int,
        order: c_int,
        xx: *mut i32,
        nrg: *mut i32,
        rshifts: *mut c_int,
        arch: c_int,
    );
    fn silk_residual_energy16_covar_FIX(
        c: *const i16,
        w_xx_mat: *const i32,
        w_xx_vec: *const i32,
        wxx: i32,
        d: c_int,
        c_q: c_int,
    ) -> i32;
    fn silk_regularize_correlations_FIX(xx_mat: *mut i32, xx: *mut i32, noise: i32, d: c_int);
    fn silk_warped_autocorrelation_FIX_c(
        corr: *mut i32,
        scale: *mut c_int,
        input: *const i16,
        warping_q16: c_int,
        length: c_int,
        order: c_int,
    );
    fn silk_LTP_analysis_filter_FIX(
        ltp_res: *mut i16,
        x: *const i16,
        ltp_coef_q14: *const i16,
        pitch_l: *const c_int,
        inv_gains_q16: *const i32,
        subfr_length: c_int,
        nb_subfr: c_int,
        pre_length: c_int,
    );
    fn silk_find_LTP_FIX(
        xx_ltp_q17: *mut i32,
        x_x_ltp_q17: *mut i32,
        r_ptr: *const i16,
        lag: *const c_int,
        subfr_length: c_int,
        nb_subfr: c_int,
        arch: c_int,
    );
}

/// `silk_EncControlStruct` fields in C declaration order (26 ints), see
/// `silk_encoder_flp::EncCtrl`.
pub type EncCtrl = [i32; 26];

/// Range encoder state: `storage`, `end_offs`, `end_window`, `nend_bits`, `nbits_total`,
/// `offs`, `rng`, `val`, `ext`, `rem`, `error`, `ec_tell_frac`.
pub type EcStateDump = [u32; 12];

/// Size of the range encoder buffer owned by [`SilkEnc`].
pub const EC_BUF: usize = 4096;

/// `sizeof(silk_EncControlStruct)` in C (checks the [`EncCtrl`] layout assumption).
pub fn ctl_size() -> i32 {
    // SAFETY: no arguments.
    unsafe { oracle_sefx_ctl_size() }
}

/// `sizeof(opus_res)` in C (checks the [`OpusRes`] assumption).
pub fn res_size() -> i32 {
    // SAFETY: no arguments.
    unsafe { oracle_sefx_res_size() }
}

/// `silk_Get_Encoder_Size` for 1 and 2 channels.
pub fn encoder_sizes() -> [i32; 2] {
    let mut out = [0; 2];
    // SAFETY: writes 2 ints.
    unsafe { oracle_sefx_sizes(out.as_mut_ptr()) };
    out
}

/// C `silk_encoder` (allocated for two channels, zeroed) + a range encoder.
#[derive(Debug)]
pub struct SilkEnc {
    ptr: NonNull<c_void>,
}

impl SilkEnc {
    /// Allocates a zeroed encoder (call [`Self::init`] next).
    pub fn new() -> Self {
        // SAFETY: plain allocation; returns NULL only on allocation failure.
        let p = unsafe { oracle_sefx_new() };
        Self {
            ptr: NonNull::new(p).expect("oracle_sefx_new: allocation failed"),
        }
    }
    /// `silk_InitEncoder`; returns (ret, status control struct).
    pub fn init(&mut self, channels: i32) -> (i32, EncCtrl) {
        let mut st = [0; 26];
        // SAFETY: valid handle; writes 26 ints.
        let r = unsafe { oracle_sefx_init(self.ptr.as_ptr(), channels, st.as_mut_ptr()) };
        (r, st)
    }
    /// Restarts the range encoder over a zeroed buffer with `size` bytes of storage.
    pub fn ec_init(&mut self, size: usize) {
        assert!(size <= EC_BUF);
        // SAFETY: valid handle; size checked.
        unsafe { oracle_sefx_ec_init(self.ptr.as_ptr(), size as c_int) };
    }
    /// Range encoder state.
    pub fn ec_state(&mut self) -> EcStateDump {
        let mut out = [0; 12];
        // SAFETY: valid handle; writes 12 values.
        unsafe { oracle_sefx_ec_state(self.ptr.as_ptr(), out.as_mut_ptr()) };
        out
    }
    /// `ec_enc_done`.
    pub fn ec_done(&mut self) {
        // SAFETY: valid handle.
        unsafe { oracle_sefx_ec_done(self.ptr.as_ptr()) };
    }
    /// First `n` bytes of the range encoder buffer.
    pub fn ec_buf(&mut self, n: usize) -> Vec<u8> {
        assert!(n <= EC_BUF);
        let mut out = vec![0u8; n];
        // SAFETY: valid handle; out holds n bytes.
        unsafe { oracle_sefx_ec_buf(self.ptr.as_ptr(), out.as_mut_ptr(), n as c_int) };
        out
    }
    /// `silk_Encode` with `n` samples per channel from `input` (interleaved for 2 API
    /// channels). `null_ec` passes `psRangeEnc = NULL` like the Opus encoder's prefill call.
    /// Returns (ret, nBytesOut).
    pub fn encode(
        &mut self,
        ctl: &mut EncCtrl,
        input: &[OpusRes],
        n: i32,
        nbytes_in: i32,
        prefill: i32,
        activity: i32,
        null_ec: bool,
    ) -> (i32, i32) {
        assert!(input.len() >= (n.max(0) * ctl[0]) as usize);
        let mut nb = nbytes_in;
        // SAFETY: valid handle; ctl holds 26 ints; input holds n * nChannelsAPI samples.
        let r = unsafe {
            oracle_sefx_encode(
                self.ptr.as_ptr(),
                ctl.as_mut_ptr(),
                input.as_ptr(),
                n,
                &mut nb,
                prefill,
                activity,
                c_int::from(null_ec),
            )
        };
        (r, nb)
    }
    /// Complete encoder state as ints (see `oracle_sefx_dump`).
    pub fn dump(&mut self) -> Vec<i32> {
        // SAFETY: cap 0 only counts.
        let n = unsafe { oracle_sefx_dump(self.ptr.as_ptr(), core::ptr::null_mut(), 0) };
        let mut v = vec![0i32; n as usize];
        // SAFETY: v holds n ints.
        unsafe { oracle_sefx_dump(self.ptr.as_ptr(), v.as_mut_ptr(), n) };
        v
    }
}

impl Default for SilkEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SilkEnc {
    fn drop(&mut self) {
        // SAFETY: handle from oracle_sefx_new, freed once.
        unsafe { oracle_sefx_free(self.ptr.as_ptr()) };
    }
}

// ---------------------------------------------------------------------------------------------
// Static and state-based helpers
// ---------------------------------------------------------------------------------------------

/// Static `warped_gain` (noise_shape_analysis_FIX.c).
pub fn warped_gain(coefs_q24: &[i32], lambda_q16: i32, order: usize) -> i32 {
    assert!(coefs_q24.len() >= order && order > 0);
    // SAFETY: size checked.
    unsafe { oracle_sefx_warped_gain(coefs_q24.as_ptr(), lambda_q16, order as c_int) }
}
/// Static `limit_warped_coefs` (noise_shape_analysis_FIX.c).
pub fn limit_warped_coefs(coefs_q24: &mut [i32], lambda_q16: i32, limit_q24: i32, order: usize) {
    assert!(coefs_q24.len() >= order && order > 0);
    // SAFETY: size checked.
    unsafe {
        oracle_sefx_limit_warped_coefs(
            coefs_q24.as_mut_ptr(),
            lambda_q16,
            limit_q24,
            order as c_int,
        )
    };
}

/// Stage-3 array `[4 * 34][5]` flattened (entry `k * nb_cbk_search + j`).
pub type Stage3 = [i32; 4 * 34 * 5];

/// Static `silk_P_Ana_calc_corr_st3`. `frame` must hold `(4 + nb_subfr) * sf_length` samples.
pub fn calc_corr_st3(
    frame: &[i16],
    start_lag: i32,
    sf_length: usize,
    nb_subfr: i32,
    complexity: i32,
) -> Stage3 {
    assert!(frame.len() >= (4 + nb_subfr as usize) * sf_length);
    let mut out = [0i32; 4 * 34 * 5];
    // SAFETY: sizes checked; out holds 680 ints.
    unsafe {
        oracle_sefx_calc_corr_st3(
            out.as_mut_ptr(),
            frame.as_ptr(),
            start_lag,
            sf_length as c_int,
            nb_subfr,
            complexity,
        );
    }
    out
}
/// Static `silk_P_Ana_calc_energy_st3`.
pub fn calc_energy_st3(
    frame: &[i16],
    start_lag: i32,
    sf_length: usize,
    nb_subfr: i32,
    complexity: i32,
) -> Stage3 {
    assert!(frame.len() >= (4 + nb_subfr as usize) * sf_length);
    let mut out = [0i32; 4 * 34 * 5];
    // SAFETY: sizes checked; out holds 680 ints.
    unsafe {
        oracle_sefx_calc_energy_st3(
            out.as_mut_ptr(),
            frame.as_ptr(),
            start_lag,
            sf_length as c_int,
            nb_subfr,
            complexity,
        );
    }
    out
}

/// Output of [`pitch_analysis_core`]: (return value, pitch_out, lagIndex, contourIndex,
/// LTPCorr_Q15).
pub type PitchOut = (i32, [i32; 4], i16, i8, i32);

/// `silk_pitch_analysis_core`. `frame` holds `(20 + nb_subfr * 5) * fs_khz` samples.
pub fn pitch_analysis_core(
    frame: &[i16],
    ltp_corr_q15: i32,
    prev_lag: i32,
    thres1_q16: i32,
    thres2_q13: i32,
    fs_khz: i32,
    complexity: i32,
    nb_subfr: i32,
) -> PitchOut {
    assert!(frame.len() >= ((20 + nb_subfr * 5) * fs_khz) as usize);
    assert!(matches!(fs_khz, 8 | 12 | 16) && matches!(nb_subfr, 2 | 4));
    let mut pitch = [0i32; 4];
    let mut li = 0i16;
    let mut ci = 0i8;
    let mut corr = ltp_corr_q15;
    // SAFETY: sizes checked; pitch holds 4 values.
    let r = unsafe {
        oracle_sefx_pitch_analysis_core(
            frame.as_ptr(),
            pitch.as_mut_ptr(),
            &mut li,
            &mut ci,
            &mut corr,
            prev_lag,
            thres1_q16,
            thres2_q13,
            fs_khz,
            complexity,
            nb_subfr,
        )
    };
    (r, pitch, li, ci, corr)
}

/// `silk_find_LPC_FIX` on a zeroed state with the given fields; returns
/// (NLSF_Q15, NLSFInterpCoef_Q2).
pub fn find_lpc(
    order: usize,
    subfr_length: usize,
    nb_subfr: usize,
    use_interp: bool,
    first_frame: bool,
    prev_nlsf: &[i16; 16],
    x: &[i16],
    min_inv_gain_q30: i32,
) -> ([i16; 16], i32) {
    assert!(x.len() >= nb_subfr * (subfr_length + order));
    let mut nlsf = [0i16; 16];
    // SAFETY: sizes checked; nlsf holds 16 values.
    let r = unsafe {
        oracle_sefx_find_lpc(
            order as c_int,
            subfr_length as c_int,
            nb_subfr as c_int,
            c_int::from(use_interp),
            c_int::from(first_frame),
            prev_nlsf.as_ptr(),
            nlsf.as_mut_ptr(),
            x.as_ptr(),
            min_inv_gain_q30,
        )
    };
    (nlsf, r)
}

/// `silk_residual_energy_FIX`; returns (nrgs, nrgsQ).
pub fn residual_energy(
    x: &[i16],
    a_q12: &[[i16; 16]; 2],
    gains: &[i32; 4],
    subfr_length: usize,
    nb_subfr: usize,
    lpc_order: usize,
) -> ([i32; 4], [i32; 4]) {
    assert!(x.len() >= nb_subfr * (subfr_length + lpc_order));
    assert!(nb_subfr == 2 || nb_subfr == 4);
    let mut n = [0i32; 4];
    let mut q = [0i32; 4];
    // SAFETY: sizes checked.
    unsafe {
        oracle_sefx_residual_energy(
            n.as_mut_ptr(),
            q.as_mut_ptr(),
            x.as_ptr(),
            a_q12.as_ptr().cast(),
            gains.as_ptr(),
            subfr_length as c_int,
            nb_subfr as c_int,
            lpc_order as c_int,
        );
    }
    (n, q)
}

// ---------------------------------------------------------------------------------------------
// Exported silk/fixed leaf functions
// ---------------------------------------------------------------------------------------------

/// `silk_apply_sine_window`.
pub fn apply_sine_window(px_win: &mut [i16], px: &[i16], win_type: i32, length: usize) {
    assert!(px_win.len() >= length && px.len() >= length);
    assert!(win_type == 1 || win_type == 2);
    assert!(length.is_multiple_of(4) && (16..=120).contains(&length));
    // SAFETY: sizes checked.
    unsafe { silk_apply_sine_window(px_win.as_mut_ptr(), px.as_ptr(), win_type, length as c_int) };
}
/// `silk_autocorr`; returns the scale.
pub fn autocorr(results: &mut [i32], input: &[i16], size: usize, count: usize) -> i32 {
    assert!(input.len() >= size && results.len() >= count.min(size) && size > 0);
    let mut scale = 0;
    // SAFETY: sizes checked.
    unsafe {
        silk_autocorr(
            results.as_mut_ptr(),
            &mut scale,
            input.as_ptr(),
            size as c_int,
            count as c_int,
            0,
        )
    };
    scale
}
/// `silk_burg_modified_c`; returns (res_nrg, res_nrg_Q).
pub fn burg_modified(
    a_q16: &mut [i32],
    x: &[i16],
    min_inv_gain_q30: i32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) -> (i32, i32) {
    assert!(x.len() >= subfr_length * nb_subfr && subfr_length * nb_subfr <= 384);
    assert!(a_q16.len() >= d && d <= 24 && subfr_length > d);
    let mut nrg = 0;
    let mut q = 0;
    // SAFETY: sizes checked.
    unsafe {
        silk_burg_modified_c(
            &mut nrg,
            &mut q,
            a_q16.as_mut_ptr(),
            x.as_ptr(),
            min_inv_gain_q30,
            subfr_length as c_int,
            nb_subfr as c_int,
            d as c_int,
            0,
        )
    };
    (nrg, q)
}
/// `silk_k2a`.
pub fn k2a(a_q24: &mut [i32], rc_q15: &[i16], order: usize) {
    assert!(a_q24.len() >= order && rc_q15.len() >= order);
    // SAFETY: sizes checked.
    unsafe { silk_k2a(a_q24.as_mut_ptr(), rc_q15.as_ptr(), order as i32) };
}
/// `silk_k2a_Q16`.
pub fn k2a_q16(a_q24: &mut [i32], rc_q16: &[i32], order: usize) {
    assert!(a_q24.len() >= order && rc_q16.len() >= order);
    // SAFETY: sizes checked.
    unsafe { silk_k2a_Q16(a_q24.as_mut_ptr(), rc_q16.as_ptr(), order as i32) };
}
/// `silk_schur`.
pub fn schur(rc_q15: &mut [i16], c: &[i32], order: usize) -> i32 {
    assert!(rc_q15.len() >= order && c.len() > order && order <= 24);
    // SAFETY: sizes checked.
    unsafe { silk_schur(rc_q15.as_mut_ptr(), c.as_ptr(), order as i32) }
}
/// `silk_schur64`.
pub fn schur64(rc_q16: &mut [i32], c: &[i32], order: usize) -> i32 {
    assert!(rc_q16.len() >= order && c.len() > order && order <= 24);
    // SAFETY: sizes checked.
    unsafe { silk_schur64(rc_q16.as_mut_ptr(), c.as_ptr(), order as i32) }
}
/// `silk_scale_copy_vector16`.
pub fn scale_copy_vector16(out: &mut [i16], input: &[i16], gain_q16: i32, n: usize) {
    assert!(out.len() >= n && input.len() >= n);
    // SAFETY: sizes checked.
    unsafe { silk_scale_copy_vector16(out.as_mut_ptr(), input.as_ptr(), gain_q16, n as c_int) };
}
/// `silk_scale_vector32_Q26_lshift_18`.
pub fn scale_vector32_q26_lshift_18(data: &mut [i32], gain_q26: i32, n: usize) {
    assert!(data.len() >= n);
    // SAFETY: size checked.
    unsafe { silk_scale_vector32_Q26_lshift_18(data.as_mut_ptr(), gain_q26, n as c_int) };
}
/// `silk_inner_prod_aligned`.
pub fn inner_prod_aligned(a: &[i16], b: &[i16], len: usize) -> i32 {
    assert!(a.len() >= len && b.len() >= len);
    // SAFETY: sizes checked.
    unsafe { silk_inner_prod_aligned(a.as_ptr(), b.as_ptr(), len as c_int, 0) }
}
/// `silk_corrVector_FIX`.
pub fn corr_vector(x: &[i16], t: &[i16], l: usize, order: usize, xt: &mut [i32], rshifts: i32) {
    assert!(x.len() >= l + order - 1 && t.len() >= l && xt.len() >= order && order > 0);
    // SAFETY: sizes checked.
    unsafe {
        silk_corrVector_FIX(
            x.as_ptr(),
            t.as_ptr(),
            l as c_int,
            order as c_int,
            xt.as_mut_ptr(),
            rshifts,
            0,
        )
    };
}
/// `silk_corrMatrix_FIX`; returns (nrg, rshifts).
pub fn corr_matrix(x: &[i16], l: usize, order: usize, xx: &mut [i32]) -> (i32, i32) {
    assert!(x.len() >= l + order - 1 && xx.len() >= order * order && order > 1);
    let mut nrg = 0;
    let mut rshifts = 0;
    // SAFETY: sizes checked.
    unsafe {
        silk_corrMatrix_FIX(
            x.as_ptr(),
            l as c_int,
            order as c_int,
            xx.as_mut_ptr(),
            &mut nrg,
            &mut rshifts,
            0,
        )
    };
    (nrg, rshifts)
}
/// `silk_residual_energy16_covar_FIX`.
pub fn residual_energy16_covar(
    c: &[i16],
    w_xx_mat: &[i32],
    w_xx_vec: &[i32],
    wxx: i32,
    d: usize,
    c_q: i32,
) -> i32 {
    assert!(c.len() >= d && w_xx_mat.len() >= d * d && w_xx_vec.len() >= d && d <= 16);
    assert!(d > 0 && c_q > 0 && c_q < 16);
    // SAFETY: sizes checked.
    unsafe {
        silk_residual_energy16_covar_FIX(
            c.as_ptr(),
            w_xx_mat.as_ptr(),
            w_xx_vec.as_ptr(),
            wxx,
            d as c_int,
            c_q,
        )
    }
}
/// `silk_regularize_correlations_FIX`.
pub fn regularize_correlations(xx_mat: &mut [i32], xx: &mut [i32], noise: i32, d: usize) {
    assert!(xx_mat.len() >= d * d && !xx.is_empty());
    // SAFETY: sizes checked.
    unsafe {
        silk_regularize_correlations_FIX(xx_mat.as_mut_ptr(), xx.as_mut_ptr(), noise, d as c_int)
    };
}
/// `silk_warped_autocorrelation_FIX_c`; returns the scale.
pub fn warped_autocorrelation(
    corr: &mut [i32],
    input: &[i16],
    warping_q16: i32,
    length: usize,
    order: usize,
) -> i32 {
    assert!(corr.len() > order && input.len() >= length && order <= 24 && order.is_multiple_of(2));
    let mut scale = 0;
    // SAFETY: sizes checked.
    unsafe {
        silk_warped_autocorrelation_FIX_c(
            corr.as_mut_ptr(),
            &mut scale,
            input.as_ptr(),
            warping_q16,
            length as c_int,
            order as c_int,
        )
    };
    scale
}
/// `silk_LTP_analysis_filter_FIX`. The C `x` pointer is `&x_buf[x_off]`.
pub fn ltp_analysis_filter(
    ltp_res: &mut [i16],
    x_buf: &[i16],
    x_off: usize,
    ltp_coef_q14: &[i16; 20],
    pitch_l: &[i32; 4],
    inv_gains_q16: &[i32; 4],
    subfr_length: usize,
    nb_subfr: usize,
    pre_length: usize,
) {
    assert!(ltp_res.len() >= nb_subfr * (subfr_length + pre_length) && nb_subfr <= 4);
    let max_lag = pitch_l[..nb_subfr].iter().copied().fold(0, i32::max) as usize;
    assert!(x_off >= max_lag + 2);
    assert!(x_buf.len() >= x_off + nb_subfr * subfr_length + pre_length);
    // SAFETY: sizes checked (the filter reads x[-pitchL-2 ..] up to the end of the last
    // subframe plus pre_length samples).
    unsafe {
        silk_LTP_analysis_filter_FIX(
            ltp_res.as_mut_ptr(),
            x_buf.as_ptr().add(x_off),
            ltp_coef_q14.as_ptr(),
            pitch_l.as_ptr(),
            inv_gains_q16.as_ptr(),
            subfr_length as c_int,
            nb_subfr as c_int,
            pre_length as c_int,
        )
    };
}
/// `silk_find_LTP_FIX`. The C `r_ptr` is `&r_buf[r_off]`; returns (XX [4*25], xX [4*5]).
pub fn find_ltp(
    r_buf: &[i16],
    r_off: usize,
    lag: &[i32; 4],
    subfr_length: usize,
    nb_subfr: usize,
) -> ([i32; 100], [i32; 20]) {
    assert!(nb_subfr <= 4);
    let max_lag = lag[..nb_subfr].iter().copied().fold(0, i32::max) as usize;
    assert!(r_off >= max_lag + 2);
    assert!(r_buf.len() >= r_off + nb_subfr * subfr_length + 5);
    let mut xx = [0i32; 100];
    let mut xv = [0i32; 20];
    // SAFETY: sizes checked.
    unsafe {
        silk_find_LTP_FIX(
            xx.as_mut_ptr(),
            xv.as_mut_ptr(),
            r_buf.as_ptr().add(r_off),
            lag.as_ptr(),
            subfr_length as c_int,
            nb_subfr as c_int,
            0,
        )
    };
    (xx, xv)
}
