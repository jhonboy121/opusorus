//! Oracle bindings for unit `silk_encoder_flp`: the floating-point SILK encoder
//! (`silk/float/*.c`, `silk/enc_API.c`, `silk/init_encoder.c`, `silk/control_codec.c`), see
//! `csrc/silk_encoder_flp.c`.
//!
//! * [`SilkEnc`] owns a C `silk_encoder` plus a range encoder over a private buffer, so tests
//!   can drive `silk_Encode` call by call exactly like `src/opus_encoder.c`, and dump the
//!   complete encoder state.
//! * Free functions wrap the exported `silk/float` leaf functions (declared directly) and the
//!   shims for static / inline helpers. Every wrapper checks buffer sizes with `assert!` before
//!   calling into C.

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "wrappers mirror the C signatures one-to-one"
)]

use core::ffi::{c_int, c_uint, c_void};
use core::ptr::NonNull;

unsafe extern "C" {
    fn oracle_sef_ctl_size() -> c_int;
    fn oracle_sef_sizes(out: *mut c_int);
    fn oracle_sef_new() -> *mut c_void;
    fn oracle_sef_free(p: *mut c_void);
    fn oracle_sef_init(p: *mut c_void, channels: c_int, status: *mut c_int) -> c_int;
    fn oracle_sef_ec_init(p: *mut c_void, size: c_int);
    fn oracle_sef_ec_state(p: *mut c_void, out: *mut c_uint);
    fn oracle_sef_ec_done(p: *mut c_void);
    fn oracle_sef_ec_buf(p: *mut c_void, out: *mut u8, n: c_int);
    fn oracle_sef_encode(
        p: *mut c_void,
        ctl: *mut c_int,
        input: *const f32,
        n: c_int,
        nbytes: *mut c_int,
        prefill: c_int,
        activity: c_int,
        null_ec: c_int,
    ) -> c_int;
    fn oracle_sef_dump(p: *mut c_void, out: *mut c_int, cap: c_int) -> c_int;

    fn oracle_sef_sigmoid(x: f32) -> f32;
    fn oracle_sef_log2(x: f64) -> f32;
    fn oracle_sef_float2int(x: f32) -> c_int;
    fn oracle_sef_float2short_array(out: *mut i16, input: *const f32, n: c_int);
    fn oracle_sef_short2float_array(out: *mut f32, input: *const i16, n: c_int);
    fn oracle_sef_warped_gain(coefs: *const f32, lambda: f32, order: c_int) -> f32;
    fn oracle_sef_warped_true2monic_coefs(coefs: *mut f32, lambda: f32, limit: f32, order: c_int);
    fn oracle_sef_limit_coefs(coefs: *mut f32, limit: f32, order: c_int);
    fn oracle_sef_calc_corr_st3(
        out: *mut f32,
        frame: *const f32,
        start_lag: c_int,
        sf_length: c_int,
        nb_subfr: c_int,
        complexity: c_int,
    );
    fn oracle_sef_calc_energy_st3(
        out: *mut f32,
        frame: *const f32,
        start_lag: c_int,
        sf_length: c_int,
        nb_subfr: c_int,
        complexity: c_int,
    );
    fn oracle_sef_find_lpc(
        order: c_int,
        subfr_length: c_int,
        nb_subfr: c_int,
        use_interp: c_int,
        first_frame: c_int,
        prev_nlsf: *const i16,
        nlsf_out: *mut i16,
        x: *const f32,
        min_inv_gain: f32,
    ) -> c_int;
    fn oracle_sef_quant_ltp_gains(
        b: *mut f32,
        cbk_index: *mut i8,
        per_index: *mut i8,
        sum_log_gain_q7: *mut c_int,
        pred_gain_db: *mut f32,
        xx: *const f32,
        x_x: *const f32,
        subfr_len: c_int,
        nb_subfr: c_int,
    );

    // Exported silk/float functions (float build, `arch` = 0).
    fn silk_apply_sine_window_FLP(px_win: *mut f32, px: *const f32, win_type: c_int, length: c_int);
    fn silk_autocorrelation_FLP(
        results: *mut f32,
        input: *const f32,
        input_size: c_int,
        correlation_count: c_int,
        arch: c_int,
    );
    fn silk_burg_modified_FLP(
        a: *mut f32,
        x: *const f32,
        min_inv_gain: f32,
        subfr_length: c_int,
        nb_subfr: c_int,
        d: c_int,
        arch: c_int,
    ) -> f32;
    fn silk_bwexpander_FLP(ar: *mut f32, d: c_int, chirp: f32);
    fn silk_corrVector_FLP(
        x: *const f32,
        t: *const f32,
        l: c_int,
        order: c_int,
        xt: *mut f32,
        arch: c_int,
    );
    fn silk_corrMatrix_FLP(x: *const f32, l: c_int, order: c_int, xx: *mut f32, arch: c_int);
    fn silk_energy_FLP(data: *const f32, size: c_int) -> f64;
    fn silk_inner_product_FLP_c(data1: *const f32, data2: *const f32, size: c_int) -> f64;
    fn silk_k2a_FLP(a: *mut f32, rc: *const f32, order: c_int);
    fn silk_LPC_analysis_filter_FLP(
        r_lpc: *mut f32,
        pred_coef: *const f32,
        s: *const f32,
        length: c_int,
        order: c_int,
    );
    fn silk_LPC_inverse_pred_gain_FLP(a: *const f32, order: c_int) -> f32;
    fn silk_LTP_analysis_filter_FLP(
        ltp_res: *mut f32,
        x: *const f32,
        b: *const f32,
        pitch_l: *const c_int,
        inv_gains: *const f32,
        subfr_length: c_int,
        nb_subfr: c_int,
        pre_length: c_int,
    );
    fn silk_regularize_correlations_FLP(xx_mat: *mut f32, xx: *mut f32, noise: f32, d: c_int);
    fn silk_residual_energy_covar_FLP(
        c: *const f32,
        w_xx_mat: *mut f32,
        w_xx_vec: *const f32,
        wxx: f32,
        d: c_int,
    ) -> f32;
    fn silk_residual_energy_FLP(
        nrgs: *mut f32,
        x: *const f32,
        a: *const f32,
        gains: *const f32,
        subfr_length: c_int,
        nb_subfr: c_int,
        lpc_order: c_int,
    );
    fn silk_scale_copy_vector_FLP(out: *mut f32, input: *const f32, gain: f32, size: c_int);
    fn silk_scale_vector_FLP(data: *mut f32, gain: f32, size: c_int);
    fn silk_schur_FLP(refl_coef: *mut f32, auto_corr: *const f32, order: c_int) -> f32;
    fn silk_insertion_sort_decreasing_FLP(a: *mut f32, idx: *mut c_int, l: c_int, k: c_int);
    fn silk_warped_autocorrelation_FLP(
        corr: *mut f32,
        input: *const f32,
        warping: f32,
        length: c_int,
        order: c_int,
    );
    fn silk_A2NLSF_FLP(nlsf_q15: *mut i16, p_ar: *const f32, lpc_order: c_int);
    fn silk_NLSF2A_FLP(p_ar: *mut f32, nlsf_q15: *const i16, lpc_order: c_int, arch: c_int);
    fn silk_find_LTP_FLP(
        xx_mat: *mut f32,
        xx_vec: *mut f32,
        r_ptr: *const f32,
        lag: *const c_int,
        subfr_length: c_int,
        nb_subfr: c_int,
        arch: c_int,
    );
    fn silk_pitch_analysis_core_FLP(
        frame: *const f32,
        pitch_out: *mut c_int,
        lag_index: *mut i16,
        contour_index: *mut i8,
        ltp_corr: *mut f32,
        prev_lag: c_int,
        search_thres1: f32,
        search_thres2: f32,
        fs_khz: c_int,
        complexity: c_int,
        nb_subfr: c_int,
        arch: c_int,
    ) -> c_int;
}

/// `silk_EncControlStruct` fields in C declaration order (26 ints): `nChannelsAPI`,
/// `nChannelsInternal`, `API_sampleRate`, `maxInternalSampleRate`, `minInternalSampleRate`,
/// `desiredInternalSampleRate`, `payloadSize_ms`, `bitRate`, `packetLossPercentage`,
/// `complexity`, `useInBandFEC`, `useDRED`, `LBRR_coded`, `useDTX`, `useCBR`, `maxBits`,
/// `toMono`, `opusCanSwitch`, `reducedDependency`, `internalSampleRate`,
/// `allowBandwidthSwitch`, `inWBmodeWithoutVariableLP`, `stereoWidth_Q14`, `switchReady`,
/// `signalType`, `offset`.
pub type EncCtrl = [i32; 26];

/// Range encoder state: `storage`, `end_offs`, `end_window`, `nend_bits`, `nbits_total`,
/// `offs`, `rng`, `val`, `ext`, `rem`, `error`, `ec_tell_frac`.
pub type EcStateDump = [u32; 12];

/// Size of the range encoder buffer owned by [`SilkEnc`].
pub const EC_BUF: usize = 4096;

/// `sizeof(silk_EncControlStruct)` in C (checks the [`EncCtrl`] layout assumption).
pub fn ctl_size() -> i32 {
    // SAFETY: no arguments.
    unsafe { oracle_sef_ctl_size() }
}

/// `silk_Get_Encoder_Size` for 1 and 2 channels.
pub fn encoder_sizes() -> [i32; 2] {
    let mut out = [0; 2];
    // SAFETY: writes 2 ints.
    unsafe { oracle_sef_sizes(out.as_mut_ptr()) };
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
        let p = unsafe { oracle_sef_new() };
        Self {
            ptr: NonNull::new(p).expect("oracle_sef_new: allocation failed"),
        }
    }
    /// `silk_InitEncoder`; returns (ret, status control struct).
    pub fn init(&mut self, channels: i32) -> (i32, EncCtrl) {
        let mut st = [0; 26];
        // SAFETY: valid handle; writes 26 ints.
        let r = unsafe { oracle_sef_init(self.ptr.as_ptr(), channels, st.as_mut_ptr()) };
        (r, st)
    }
    /// Restarts the range encoder over a zeroed buffer with `size` bytes of storage.
    pub fn ec_init(&mut self, size: usize) {
        assert!(size <= EC_BUF);
        // SAFETY: valid handle; size checked.
        unsafe { oracle_sef_ec_init(self.ptr.as_ptr(), size as c_int) };
    }
    /// Range encoder state.
    pub fn ec_state(&mut self) -> EcStateDump {
        let mut out = [0; 12];
        // SAFETY: valid handle; writes 12 values.
        unsafe { oracle_sef_ec_state(self.ptr.as_ptr(), out.as_mut_ptr()) };
        out
    }
    /// `ec_enc_done`.
    pub fn ec_done(&mut self) {
        // SAFETY: valid handle.
        unsafe { oracle_sef_ec_done(self.ptr.as_ptr()) };
    }
    /// First `n` bytes of the range encoder buffer.
    pub fn ec_buf(&mut self, n: usize) -> Vec<u8> {
        assert!(n <= EC_BUF);
        let mut out = vec![0u8; n];
        // SAFETY: valid handle; out holds n bytes.
        unsafe { oracle_sef_ec_buf(self.ptr.as_ptr(), out.as_mut_ptr(), n as c_int) };
        out
    }
    /// `silk_Encode` with `n` samples per channel from `input` (interleaved for 2 API
    /// channels). `null_ec` passes `psRangeEnc = NULL` like the Opus encoder's prefill call.
    /// Returns (ret, nBytesOut).
    pub fn encode(
        &mut self,
        ctl: &mut EncCtrl,
        input: &[f32],
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
            oracle_sef_encode(
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
    /// Complete encoder state as ints (see `oracle_sef_dump`).
    pub fn dump(&mut self) -> Vec<i32> {
        // SAFETY: cap 0 only counts.
        let n = unsafe { oracle_sef_dump(self.ptr.as_ptr(), core::ptr::null_mut(), 0) };
        let mut v = vec![0i32; n as usize];
        // SAFETY: v holds n ints.
        unsafe { oracle_sef_dump(self.ptr.as_ptr(), v.as_mut_ptr(), n) };
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
        // SAFETY: handle from oracle_sef_new, freed once.
        unsafe { oracle_sef_free(self.ptr.as_ptr()) };
    }
}

// ---------------------------------------------------------------------------------------------
// SigProc_FLP.h helpers and static helpers
// ---------------------------------------------------------------------------------------------

/// `silk_sigmoid`.
pub fn sigmoid(x: f32) -> f32 {
    // SAFETY: pure function.
    unsafe { oracle_sef_sigmoid(x) }
}
/// `silk_log2` (SigProc_FLP.h).
pub fn log2(x: f64) -> f32 {
    // SAFETY: pure function.
    unsafe { oracle_sef_log2(x) }
}
/// `silk_float2int`.
pub fn float2int(x: f32) -> i32 {
    // SAFETY: pure function.
    unsafe { oracle_sef_float2int(x) }
}
/// `silk_float2short_array`.
pub fn float2short_array(out: &mut [i16], input: &[f32], n: usize) {
    assert!(out.len() >= n && input.len() >= n);
    // SAFETY: sizes checked.
    unsafe { oracle_sef_float2short_array(out.as_mut_ptr(), input.as_ptr(), n as c_int) };
}
/// `silk_short2float_array`.
pub fn short2float_array(out: &mut [f32], input: &[i16], n: usize) {
    assert!(out.len() >= n && input.len() >= n);
    // SAFETY: sizes checked.
    unsafe { oracle_sef_short2float_array(out.as_mut_ptr(), input.as_ptr(), n as c_int) };
}
/// Static `warped_gain` (noise_shape_analysis_FLP.c).
pub fn warped_gain(coefs: &[f32], lambda: f32, order: usize) -> f32 {
    assert!(coefs.len() >= order && order > 0);
    // SAFETY: size checked.
    unsafe { oracle_sef_warped_gain(coefs.as_ptr(), lambda, order as c_int) }
}
/// Static `warped_true2monic_coefs` (noise_shape_analysis_FLP.c).
pub fn warped_true2monic_coefs(coefs: &mut [f32], lambda: f32, limit: f32, order: usize) {
    assert!(coefs.len() >= order && order > 0);
    // SAFETY: size checked.
    unsafe {
        oracle_sef_warped_true2monic_coefs(coefs.as_mut_ptr(), lambda, limit, order as c_int)
    };
}
/// Static `limit_coefs` (noise_shape_analysis_FLP.c).
pub fn limit_coefs(coefs: &mut [f32], limit: f32, order: usize) {
    assert!(coefs.len() >= order && order > 0);
    // SAFETY: size checked.
    unsafe { oracle_sef_limit_coefs(coefs.as_mut_ptr(), limit, order as c_int) };
}

/// Stage-3 array `[4][34][5]` flattened.
pub type Stage3 = [f32; 4 * 34 * 5];

/// Static `silk_P_Ana_calc_corr_st3`. `frame` must hold `(4 + nb_subfr) * sf_length` samples.
pub fn calc_corr_st3(
    frame: &[f32],
    start_lag: i32,
    sf_length: usize,
    nb_subfr: i32,
    complexity: i32,
) -> Stage3 {
    assert!(frame.len() >= (4 + nb_subfr as usize) * sf_length);
    let mut out = [0f32; 4 * 34 * 5];
    // SAFETY: sizes checked; out holds 680 floats.
    unsafe {
        oracle_sef_calc_corr_st3(
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
    frame: &[f32],
    start_lag: i32,
    sf_length: usize,
    nb_subfr: i32,
    complexity: i32,
) -> Stage3 {
    assert!(frame.len() >= (4 + nb_subfr as usize) * sf_length);
    let mut out = [0f32; 4 * 34 * 5];
    // SAFETY: sizes checked; out holds 680 floats.
    unsafe {
        oracle_sef_calc_energy_st3(
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

/// `silk_find_LPC_FLP` on a zeroed state with the given fields; returns
/// (NLSF_Q15, NLSFInterpCoef_Q2).
pub fn find_lpc(
    order: usize,
    subfr_length: usize,
    nb_subfr: usize,
    use_interp: bool,
    first_frame: bool,
    prev_nlsf: &[i16; 16],
    x: &[f32],
    min_inv_gain: f32,
) -> ([i16; 16], i32) {
    assert!(x.len() >= nb_subfr * (subfr_length + order));
    let mut nlsf = [0i16; 16];
    // SAFETY: sizes checked; nlsf holds 16 values.
    let r = unsafe {
        oracle_sef_find_lpc(
            order as c_int,
            subfr_length as c_int,
            nb_subfr as c_int,
            c_int::from(use_interp),
            c_int::from(first_frame),
            prev_nlsf.as_ptr(),
            nlsf.as_mut_ptr(),
            x.as_ptr(),
            min_inv_gain,
        )
    };
    (nlsf, r)
}

/// Output of [`quant_ltp_gains`]: B (20), cbk_index (4), periodicity index, sum_log_gain_Q7,
/// pred_gain_dB.
pub type QuantLtpOut = ([f32; 20], [i8; 4], i8, i32, f32);

/// `silk_quant_LTP_gains_FLP`.
pub fn quant_ltp_gains(
    xx: &[f32],
    x_x: &[f32],
    subfr_len: i32,
    nb_subfr: usize,
    sum_log_gain_q7: i32,
) -> QuantLtpOut {
    assert!(xx.len() >= nb_subfr * 25 && x_x.len() >= nb_subfr * 5 && nb_subfr <= 4);
    let mut b = [0f32; 20];
    let mut cbk = [0i8; 4];
    let mut per = 0i8;
    let mut slg = sum_log_gain_q7;
    let mut pg = 0f32;
    // SAFETY: sizes checked.
    unsafe {
        oracle_sef_quant_ltp_gains(
            b.as_mut_ptr(),
            cbk.as_mut_ptr(),
            &mut per,
            &mut slg,
            &mut pg,
            xx.as_ptr(),
            x_x.as_ptr(),
            subfr_len,
            nb_subfr as c_int,
        );
    }
    (b, cbk, per, slg, pg)
}

// ---------------------------------------------------------------------------------------------
// Exported silk/float leaf functions
// ---------------------------------------------------------------------------------------------

/// `silk_apply_sine_window_FLP`.
pub fn apply_sine_window(px_win: &mut [f32], px: &[f32], win_type: i32, length: usize) {
    assert!(px_win.len() >= length && px.len() >= length);
    assert!(win_type == 1 || win_type == 2);
    assert!(length.is_multiple_of(4));
    // SAFETY: sizes checked.
    unsafe {
        silk_apply_sine_window_FLP(px_win.as_mut_ptr(), px.as_ptr(), win_type, length as c_int)
    };
}
/// `silk_autocorrelation_FLP`.
pub fn autocorrelation(results: &mut [f32], input: &[f32], size: usize, count: usize) {
    assert!(input.len() >= size && results.len() >= count.min(size));
    // SAFETY: sizes checked.
    unsafe {
        silk_autocorrelation_FLP(
            results.as_mut_ptr(),
            input.as_ptr(),
            size as c_int,
            count as c_int,
            0,
        )
    };
}
/// `silk_burg_modified_FLP`.
pub fn burg_modified(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) -> f32 {
    assert!(a.len() >= d && x.len() >= subfr_length * nb_subfr && d <= 24);
    assert!(subfr_length * nb_subfr <= 384);
    // SAFETY: sizes checked.
    unsafe {
        silk_burg_modified_FLP(
            a.as_mut_ptr(),
            x.as_ptr(),
            min_inv_gain,
            subfr_length as c_int,
            nb_subfr as c_int,
            d as c_int,
            0,
        )
    }
}
/// `silk_bwexpander_FLP`.
pub fn bwexpander(ar: &mut [f32], d: usize, chirp: f32) {
    assert!(ar.len() >= d && d > 0);
    // SAFETY: size checked.
    unsafe { silk_bwexpander_FLP(ar.as_mut_ptr(), d as c_int, chirp) };
}
/// `silk_corrVector_FLP`.
pub fn corr_vector(x: &[f32], t: &[f32], l: usize, order: usize, xt: &mut [f32]) {
    assert!(x.len() >= l + order - 1 && t.len() >= l && xt.len() >= order);
    // SAFETY: sizes checked.
    unsafe {
        silk_corrVector_FLP(
            x.as_ptr(),
            t.as_ptr(),
            l as c_int,
            order as c_int,
            xt.as_mut_ptr(),
            0,
        )
    };
}
/// `silk_corrMatrix_FLP`.
pub fn corr_matrix(x: &[f32], l: usize, order: usize, xx: &mut [f32]) {
    assert!(x.len() >= l + order - 1 && xx.len() >= order * order && order >= 2);
    // SAFETY: sizes checked.
    unsafe { silk_corrMatrix_FLP(x.as_ptr(), l as c_int, order as c_int, xx.as_mut_ptr(), 0) };
}
/// `silk_energy_FLP`.
pub fn energy(data: &[f32], n: usize) -> f64 {
    assert!(data.len() >= n);
    // SAFETY: size checked.
    unsafe { silk_energy_FLP(data.as_ptr(), n as c_int) }
}
/// `silk_inner_product_FLP_c`.
pub fn inner_product(a: &[f32], b: &[f32], n: usize) -> f64 {
    assert!(a.len() >= n && b.len() >= n);
    // SAFETY: sizes checked.
    unsafe { silk_inner_product_FLP_c(a.as_ptr(), b.as_ptr(), n as c_int) }
}
/// `silk_k2a_FLP`.
pub fn k2a(a: &mut [f32], rc: &[f32], order: usize) {
    assert!(a.len() >= order && rc.len() >= order);
    // SAFETY: sizes checked.
    unsafe { silk_k2a_FLP(a.as_mut_ptr(), rc.as_ptr(), order as c_int) };
}
/// `silk_LPC_analysis_filter_FLP` (orders 6, 8, 10, 12, 16).
pub fn lpc_analysis_filter(r: &mut [f32], coef: &[f32], s: &[f32], length: usize, order: usize) {
    assert!(matches!(order, 6 | 8 | 10 | 12 | 16) && order <= length);
    assert!(r.len() >= length && s.len() >= length && coef.len() >= order);
    // SAFETY: sizes checked.
    unsafe {
        silk_LPC_analysis_filter_FLP(
            r.as_mut_ptr(),
            coef.as_ptr(),
            s.as_ptr(),
            length as c_int,
            order as c_int,
        )
    };
}
/// `silk_LPC_inverse_pred_gain_FLP`.
pub fn lpc_inverse_pred_gain(a: &[f32], order: usize) -> f32 {
    assert!(a.len() >= order && (1..=24).contains(&order));
    // SAFETY: size checked.
    unsafe { silk_LPC_inverse_pred_gain_FLP(a.as_ptr(), order as c_int) }
}
/// `silk_LTP_analysis_filter_FLP` with `x = &x_buf[x_off]`.
pub fn ltp_analysis_filter(
    ltp_res: &mut [f32],
    x_buf: &[f32],
    x_off: usize,
    b: &[f32],
    pitch_l: &[i32],
    inv_gains: &[f32],
    subfr_length: usize,
    nb_subfr: usize,
    pre_length: usize,
) {
    assert!(ltp_res.len() >= nb_subfr * (subfr_length + pre_length));
    assert!(b.len() >= 5 * nb_subfr && pitch_l.len() >= nb_subfr && inv_gains.len() >= nb_subfr);
    for k in 0..nb_subfr {
        assert!(x_off + k * subfr_length >= pitch_l[k] as usize + 2);
    }
    assert!(x_off + nb_subfr * subfr_length + pre_length <= x_buf.len());
    // SAFETY: bounds checked above (reads stay within x_buf).
    unsafe {
        silk_LTP_analysis_filter_FLP(
            ltp_res.as_mut_ptr(),
            x_buf.as_ptr().add(x_off),
            b.as_ptr(),
            pitch_l.as_ptr(),
            inv_gains.as_ptr(),
            subfr_length as c_int,
            nb_subfr as c_int,
            pre_length as c_int,
        )
    };
}
/// `silk_regularize_correlations_FLP`.
pub fn regularize_correlations(xx_mat: &mut [f32], xx: &mut [f32], noise: f32, d: usize) {
    assert!(xx_mat.len() >= d * d && !xx.is_empty());
    // SAFETY: sizes checked.
    unsafe {
        silk_regularize_correlations_FLP(xx_mat.as_mut_ptr(), xx.as_mut_ptr(), noise, d as c_int)
    };
}
/// `silk_residual_energy_covar_FLP`.
pub fn residual_energy_covar(
    c: &[f32],
    w_xx_mat: &mut [f32],
    w_xx_vec: &[f32],
    wxx: f32,
    d: usize,
) -> f32 {
    assert!(d > 0 && c.len() >= d && w_xx_mat.len() >= d * d && w_xx_vec.len() >= d);
    // SAFETY: sizes checked.
    unsafe {
        silk_residual_energy_covar_FLP(
            c.as_ptr(),
            w_xx_mat.as_mut_ptr(),
            w_xx_vec.as_ptr(),
            wxx,
            d as c_int,
        )
    }
}
/// `silk_residual_energy_FLP`.
pub fn residual_energy(
    x: &[f32],
    a: &[[f32; 16]; 2],
    gains: &[f32],
    subfr_length: usize,
    nb_subfr: usize,
    lpc_order: usize,
) -> [f32; 4] {
    assert!(x.len() >= nb_subfr * (subfr_length + lpc_order) && gains.len() >= nb_subfr);
    let mut nrgs = [0f32; 4];
    // SAFETY: sizes checked; `a` is the C `silk_float a[2][MAX_LPC_ORDER]`.
    unsafe {
        silk_residual_energy_FLP(
            nrgs.as_mut_ptr(),
            x.as_ptr(),
            a.as_ptr().cast(),
            gains.as_ptr(),
            subfr_length as c_int,
            nb_subfr as c_int,
            lpc_order as c_int,
        )
    };
    nrgs
}
/// `silk_scale_copy_vector_FLP`.
pub fn scale_copy_vector(out: &mut [f32], input: &[f32], gain: f32, n: usize) {
    assert!(out.len() >= n && input.len() >= n);
    // SAFETY: sizes checked.
    unsafe { silk_scale_copy_vector_FLP(out.as_mut_ptr(), input.as_ptr(), gain, n as c_int) };
}
/// `silk_scale_vector_FLP`.
pub fn scale_vector(data: &mut [f32], gain: f32, n: usize) {
    assert!(data.len() >= n);
    // SAFETY: size checked.
    unsafe { silk_scale_vector_FLP(data.as_mut_ptr(), gain, n as c_int) };
}
/// `silk_schur_FLP`.
pub fn schur(refl: &mut [f32], auto_corr: &[f32], order: usize) -> f32 {
    assert!(refl.len() >= order && auto_corr.len() > order && order <= 24);
    // SAFETY: sizes checked.
    unsafe { silk_schur_FLP(refl.as_mut_ptr(), auto_corr.as_ptr(), order as c_int) }
}
/// `silk_insertion_sort_decreasing_FLP`.
pub fn insertion_sort_decreasing(a: &mut [f32], idx: &mut [i32], l: usize, k: usize) {
    assert!(a.len() >= l && idx.len() >= k && k > 0 && l >= k);
    // SAFETY: sizes checked.
    unsafe {
        silk_insertion_sort_decreasing_FLP(a.as_mut_ptr(), idx.as_mut_ptr(), l as c_int, k as c_int)
    };
}
/// `silk_warped_autocorrelation_FLP`.
pub fn warped_autocorrelation(
    corr: &mut [f32],
    input: &[f32],
    warping: f32,
    length: usize,
    order: usize,
) {
    assert!(corr.len() > order && input.len() >= length && order.is_multiple_of(2) && order <= 24);
    // SAFETY: sizes checked.
    unsafe {
        silk_warped_autocorrelation_FLP(
            corr.as_mut_ptr(),
            input.as_ptr(),
            warping,
            length as c_int,
            order as c_int,
        )
    };
}
/// `silk_A2NLSF_FLP`.
pub fn a2nlsf(nlsf: &mut [i16], ar: &[f32], order: usize) {
    assert!(nlsf.len() >= order && ar.len() >= order && order <= 16 && order.is_multiple_of(2));
    // SAFETY: sizes checked.
    unsafe { silk_A2NLSF_FLP(nlsf.as_mut_ptr(), ar.as_ptr(), order as c_int) };
}
/// `silk_NLSF2A_FLP`.
pub fn nlsf2a(ar: &mut [f32], nlsf: &[i16], order: usize) {
    assert!(nlsf.len() >= order && ar.len() >= order && (order == 10 || order == 16));
    // SAFETY: sizes checked.
    unsafe { silk_NLSF2A_FLP(ar.as_mut_ptr(), nlsf.as_ptr(), order as c_int, 0) };
}
/// `silk_find_LTP_FLP` with `r_ptr = &r_buf[r_off]`; returns (XX [4*25], xX [4*5]).
pub fn find_ltp(
    r_buf: &[f32],
    r_off: usize,
    lag: &[i32; 4],
    subfr_length: usize,
    nb_subfr: usize,
) -> ([f32; 100], [f32; 20]) {
    for k in 0..nb_subfr {
        assert!(r_off + k * subfr_length >= lag[k] as usize + 2);
    }
    assert!(r_off + nb_subfr * subfr_length + 5 <= r_buf.len());
    let mut xx = [0f32; 100];
    let mut x_x = [0f32; 20];
    // SAFETY: bounds checked above (reads stay within r_buf).
    unsafe {
        silk_find_LTP_FLP(
            xx.as_mut_ptr(),
            x_x.as_mut_ptr(),
            r_buf.as_ptr().add(r_off),
            lag.as_ptr(),
            subfr_length as c_int,
            nb_subfr as c_int,
            0,
        )
    };
    (xx, x_x)
}

/// Output of [`pitch_analysis_core`]: (return value, pitch_out[4], lagIndex, contourIndex,
/// LTPCorr).
pub type PitchOut = (i32, [i32; 4], i16, i8, f32);

/// `silk_pitch_analysis_core_FLP`. `frame` holds `(20 + 5 * nb_subfr) * fs_khz` samples.
pub fn pitch_analysis_core(
    frame: &[f32],
    ltp_corr_in: f32,
    prev_lag: i32,
    search_thres1: f32,
    search_thres2: f32,
    fs_khz: i32,
    complexity: i32,
    nb_subfr: i32,
    pitch_init: [i32; 4],
) -> PitchOut {
    assert!(matches!(fs_khz, 8 | 12 | 16) && (0..=2).contains(&complexity));
    assert!(nb_subfr == 2 || nb_subfr == 4);
    assert!(frame.len() >= ((20 + 5 * nb_subfr) * fs_khz) as usize);
    let mut pitch = pitch_init;
    let mut lag_index = 0i16;
    let mut contour = 0i8;
    let mut ltp_corr = ltp_corr_in;
    // SAFETY: sizes checked; pitch holds 4 ints.
    let r = unsafe {
        silk_pitch_analysis_core_FLP(
            frame.as_ptr(),
            pitch.as_mut_ptr(),
            &mut lag_index,
            &mut contour,
            &mut ltp_corr,
            prev_lag,
            search_thres1,
            search_thres2,
            fs_khz,
            complexity,
            nb_subfr,
            0,
        )
    };
    (r, pitch, lag_index, contour, ltp_corr)
}
