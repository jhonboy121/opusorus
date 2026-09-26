//! Oracle bindings for unit `silk_common`: SILK tables, struct layouts, signal-processing
//! helpers, NLSF code and entropy-coding helpers (see `csrc/silk_common.c`).
//!
//! Wrappers check buffer sizes with `assert!` before calling into C so that a wrong test can
//! never make the C code write out of bounds.

use core::ffi::{c_int, c_longlong, c_uint};

unsafe extern "C" {
    fn oracle_silk_table_u8(id: c_int, out: *mut u8, cap: c_int) -> c_int;
    fn oracle_silk_table_i8(id: c_int, out: *mut i8, cap: c_int) -> c_int;
    fn oracle_silk_table_i16(id: c_int, out: *mut i16, cap: c_int) -> c_int;
    fn oracle_silk_table_i32(id: c_int, out: *mut i32, cap: c_int) -> c_int;
    fn oracle_silk_struct_dims(out: *mut c_int) -> c_int;

    fn oracle_silk_lin2log(x: i32) -> i32;
    fn oracle_silk_log2lin(x: i32) -> i32;
    fn oracle_silk_sigm_Q15(x: c_int) -> c_int;
    fn oracle_silk_insertion_sort_increasing(a: *mut i32, idx: *mut c_int, l: c_int, k: c_int);
    fn oracle_silk_insertion_sort_increasing_all_values_int16(a: *mut i16, l: c_int);
    #[cfg(feature = "fixed-point")]
    fn oracle_silk_insertion_sort_decreasing_int16(
        a: *mut i16,
        idx: *mut c_int,
        l: c_int,
        k: c_int,
    );
    fn oracle_silk_bwexpander(ar: *mut i16, d: c_int, chirp: i32);
    fn oracle_silk_bwexpander_32(ar: *mut i32, d: c_int, chirp: i32);
    fn oracle_silk_inner_prod_aligned_scale(
        a: *const i16,
        b: *const i16,
        scale: c_int,
        len: c_int,
    ) -> i32;
    fn oracle_silk_inner_prod16(a: *const i16, b: *const i16, len: c_int) -> c_longlong;
    fn oracle_silk_sum_sqr_shift(energy: *mut i32, shift: *mut c_int, x: *const i16, len: c_int);
    fn oracle_silk_interpolate(
        xi: *mut i16,
        x0: *const i16,
        x1: *const i16,
        ifact: c_int,
        d: c_int,
    );
    fn oracle_silk_biquad_alt_stride1(
        input: *const i16,
        b: *const i32,
        a: *const i32,
        s: *mut i32,
        out: *mut i16,
        len: c_int,
    );
    fn oracle_silk_biquad_alt_stride2(
        input: *const i16,
        b: *const i32,
        a: *const i32,
        s: *mut i32,
        out: *mut i16,
        len: c_int,
    );
    fn oracle_silk_LP_variable_cutoff(st: *mut i32, frame: *mut i16, len: c_int);
    fn oracle_silk_ana_filt_bank_1(
        input: *const i16,
        s: *mut i32,
        out_l: *mut i16,
        out_h: *mut i16,
        n: c_int,
    );
    fn oracle_silk_LPC_inverse_pred_gain(a: *const i16, order: c_int) -> i32;
    fn oracle_silk_LPC_fit(a_qout: *mut i16, a_qin: *mut i32, qout: c_int, qin: c_int, d: c_int);
    fn oracle_silk_LPC_analysis_filter(
        out: *mut i16,
        input: *const i16,
        b: *const i16,
        len: c_int,
        d: c_int,
    );

    fn oracle_silk_NLSF2A(a_q12: *mut i16, nlsf: *const i16, d: c_int);
    fn oracle_silk_A2NLSF(nlsf: *mut i16, a_q16: *mut i32, d: c_int);
    fn oracle_silk_NLSF_decode(nlsf: *mut i16, indices: *mut i8, wb: c_int);
    fn oracle_silk_NLSF_unpack(ec_ix: *mut i16, pred_q8: *mut u8, wb: c_int, cb1_index: c_int);
    fn oracle_silk_NLSF_stabilize(nlsf: *mut i16, ndelta: *const i16, l: c_int);
    fn oracle_silk_NLSF_VQ_weights_laroia(out: *mut i16, nlsf: *const i16, d: c_int);
    fn oracle_silk_NLSF_VQ(
        err: *mut i32,
        input: *const i16,
        cb: *const u8,
        w: *const i16,
        k: c_int,
        order: c_int,
    );
    fn oracle_silk_NLSF_encode(
        indices: *mut i8,
        nlsf: *mut i16,
        wb: c_int,
        w_q2: *const i16,
        mu_q20: c_int,
        n_survivors: c_int,
        signal_type: c_int,
    ) -> i32;
    fn oracle_silk_NLSF_del_dec_quant(
        indices: *mut i8,
        x_q10: *const i16,
        w_q5: *const i16,
        pred_coef_q8: *const u8,
        ec_ix: *const i16,
        ec_rates_q5: *const u8,
        quant_step_size_q16: c_int,
        inv_quant_step_size_q6: c_int,
        mu_q20: c_int,
        order: c_int,
    ) -> i32;

    fn oracle_silk_encode_signs(
        buf: *mut u8,
        size: c_int,
        pre: c_uint,
        pre_n: c_int,
        pulses: *const i8,
        length: c_int,
        signal_type: c_int,
        quant_offset_type: c_int,
        sum_pulses: *const c_int,
        res: *mut u32,
    );
    fn oracle_silk_decode_signs(
        buf: *mut u8,
        size: c_int,
        pre_n: c_int,
        pulses: *mut i16,
        length: c_int,
        signal_type: c_int,
        quant_offset_type: c_int,
        sum_pulses: *const c_int,
        res: *mut u32,
    );
    fn oracle_silk_shell_encoder(
        buf: *mut u8,
        size: c_int,
        pre: c_uint,
        pre_n: c_int,
        pulses0: *const c_int,
        res: *mut u32,
    );
    fn oracle_silk_shell_decoder(
        buf: *mut u8,
        size: c_int,
        pre_n: c_int,
        pulses0: *mut i16,
        pulses4: c_int,
        res: *mut u32,
    );
    fn oracle_silk_encode_pulses(
        buf: *mut u8,
        size: c_int,
        pre: c_uint,
        pre_n: c_int,
        signal_type: c_int,
        quant_offset_type: c_int,
        pulses: *mut i8,
        frame_length: c_int,
        res: *mut u32,
    );
    fn oracle_silk_decode_pulses(
        buf: *mut u8,
        size: c_int,
        pre_n: c_int,
        pulses: *mut i16,
        signal_type: c_int,
        quant_offset_type: c_int,
        frame_length: c_int,
        res: *mut u32,
    );
    fn oracle_silk_stereo_encode(
        buf: *mut u8,
        size: c_int,
        pre: c_uint,
        pre_n: c_int,
        ix: *const i8,
        mid_only: c_int,
        res: *mut u32,
    );
    fn oracle_silk_stereo_decode(
        buf: *mut u8,
        size: c_int,
        pre_n: c_int,
        pred_q13: *mut i32,
        want_mid: c_int,
        mid_only: *mut c_int,
        res: *mut u32,
    );
    fn oracle_silk_gains_quant(
        ind: *mut i8,
        gain_q16: *mut i32,
        prev_ind: *mut i8,
        conditional: c_int,
        nb_subfr: c_int,
    );
    fn oracle_silk_gains_dequant(
        gain_q16: *mut i32,
        ind: *const i8,
        prev_ind: *mut i8,
        conditional: c_int,
        nb_subfr: c_int,
    );
    fn oracle_silk_gains_ID(ind: *const i8, nb_subfr: c_int) -> i32;
    fn oracle_silk_decode_pitch(
        lag_index: c_int,
        contour_index: c_int,
        pitch_lags: *mut c_int,
        fs_khz: c_int,
        nb_subfr: c_int,
    );
}

// ---------------------------------------------------------------------------------------------
// Tables / structs
// ---------------------------------------------------------------------------------------------

macro_rules! table_fn {
    ($name:ident, $c:ident, $t:ty) => {
        /// Copy of the C table with the given shim id (see `csrc/silk_common.c`), or `None` if
        /// the id is unknown.
        #[must_use]
        pub fn $name(id: i32) -> Option<Vec<$t>> {
            let mut buf = vec![0 as $t; 1024];
            // SAFETY: `buf` has room for `cap` elements; the shim writes at most `cap`.
            let n = unsafe { $c(id, buf.as_mut_ptr(), buf.len() as c_int) };
            if n < 0 {
                return None;
            }
            buf.truncate(n as usize);
            Some(buf)
        }
    };
}
table_fn!(table_u8, oracle_silk_table_u8, u8);
table_fn!(table_i8, oracle_silk_table_i8, i8);
table_fn!(table_i16, oracle_silk_table_i16, i16);
table_fn!(table_i32, oracle_silk_table_i32, i32);

/// Element counts of the array fields of the C SILK structs (order: see the shim).
#[must_use]
pub fn struct_dims() -> Vec<i32> {
    let mut out = vec![0 as c_int; 256];
    // SAFETY: the shim writes fewer than 256 ints.
    let n = unsafe { oracle_silk_struct_dims(out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

// ---------------------------------------------------------------------------------------------
// Signal processing
// ---------------------------------------------------------------------------------------------

/// C `silk_lin2log`.
#[must_use]
pub fn lin2log(x: i32) -> i32 {
    // SAFETY: pure function.
    unsafe { oracle_silk_lin2log(x) }
}
/// C `silk_log2lin`.
#[must_use]
pub fn log2lin(x: i32) -> i32 {
    // SAFETY: pure function.
    unsafe { oracle_silk_log2lin(x) }
}
/// C `silk_sigm_Q15`.
#[must_use]
pub fn sigm_q15(x: i32) -> i32 {
    // SAFETY: pure function.
    unsafe { oracle_silk_sigm_Q15(x) }
}

/// C `silk_insertion_sort_increasing` on `a[..l]`; returns `idx[..k]`.
pub fn insertion_sort_increasing(a: &mut [i32], l: usize, k: usize) -> Vec<i32> {
    assert!(a.len() >= l && l >= k && k > 0);
    let mut idx = vec![0 as c_int; k];
    // SAFETY: `a` has `l` elements and `idx` has `k`.
    unsafe {
        oracle_silk_insertion_sort_increasing(
            a.as_mut_ptr(),
            idx.as_mut_ptr(),
            l as c_int,
            k as c_int,
        )
    };
    idx
}
/// C `silk_insertion_sort_increasing_all_values_int16`.
pub fn insertion_sort_increasing_all_values_int16(a: &mut [i16]) {
    assert!(!a.is_empty());
    // SAFETY: `a` has `len` elements.
    unsafe {
        oracle_silk_insertion_sort_increasing_all_values_int16(a.as_mut_ptr(), a.len() as c_int)
    };
}
/// C `silk_insertion_sort_decreasing_int16` on `a[..l]` (fixed-point build only); returns
/// `idx[..k]`.
#[cfg(feature = "fixed-point")]
pub fn insertion_sort_decreasing_int16(a: &mut [i16], l: usize, k: usize) -> Vec<i32> {
    assert!(a.len() >= l && l >= k && k > 0);
    let mut idx = vec![0 as c_int; k];
    // SAFETY: `a` has `l` elements and `idx` has `k`.
    unsafe {
        oracle_silk_insertion_sort_decreasing_int16(
            a.as_mut_ptr(),
            idx.as_mut_ptr(),
            l as c_int,
            k as c_int,
        )
    };
    idx
}
/// C `silk_bwexpander` on the whole slice.
pub fn bwexpander(ar: &mut [i16], chirp_q16: i32) {
    assert!(!ar.is_empty());
    // SAFETY: `ar` has `len` elements.
    unsafe { oracle_silk_bwexpander(ar.as_mut_ptr(), ar.len() as c_int, chirp_q16) };
}
/// C `silk_bwexpander_32` on the whole slice.
pub fn bwexpander_32(ar: &mut [i32], chirp_q16: i32) {
    assert!(!ar.is_empty());
    // SAFETY: `ar` has `len` elements.
    unsafe { oracle_silk_bwexpander_32(ar.as_mut_ptr(), ar.len() as c_int, chirp_q16) };
}
/// C `silk_inner_prod_aligned_scale`.
#[must_use]
pub fn inner_prod_aligned_scale(a: &[i16], b: &[i16], scale: i32) -> i32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: both inputs have `len` elements.
    unsafe { oracle_silk_inner_prod_aligned_scale(a.as_ptr(), b.as_ptr(), scale, a.len() as c_int) }
}
/// C `silk_inner_prod16_c`.
#[must_use]
pub fn inner_prod16(a: &[i16], b: &[i16]) -> i64 {
    assert_eq!(a.len(), b.len());
    // SAFETY: both inputs have `len` elements.
    unsafe { oracle_silk_inner_prod16(a.as_ptr(), b.as_ptr(), a.len() as c_int) }
}
/// C `silk_sum_sqr_shift`; returns `(energy, shift)`.
#[must_use]
pub fn sum_sqr_shift(x: &[i16]) -> (i32, i32) {
    let (mut e, mut s) = (0i32, 0 as c_int);
    // SAFETY: `x` has `len` elements; outputs are valid pointers.
    unsafe { oracle_silk_sum_sqr_shift(&mut e, &mut s, x.as_ptr(), x.len() as c_int) };
    (e, s)
}
/// C `silk_interpolate`.
#[must_use]
pub fn interpolate(x0: &[i16], x1: &[i16], ifact_q2: i32) -> Vec<i16> {
    assert_eq!(x0.len(), x1.len());
    let mut xi = vec![0i16; x0.len()];
    // SAFETY: all buffers have `d` elements.
    unsafe {
        oracle_silk_interpolate(
            xi.as_mut_ptr(),
            x0.as_ptr(),
            x1.as_ptr(),
            ifact_q2,
            x0.len() as c_int,
        )
    };
    xi
}
/// C `silk_biquad_alt_stride1` (`s`: state `[2]`); returns the output.
pub fn biquad_alt_stride1(input: &[i16], b: &[i32; 3], a: &[i32; 2], s: &mut [i32; 2]) -> Vec<i16> {
    let mut out = vec![0i16; input.len()];
    // SAFETY: input/out have `len` elements, coefficient/state arrays have fixed sizes.
    unsafe {
        oracle_silk_biquad_alt_stride1(
            input.as_ptr(),
            b.as_ptr(),
            a.as_ptr(),
            s.as_mut_ptr(),
            out.as_mut_ptr(),
            input.len() as c_int,
        );
    };
    out
}
/// C `silk_biquad_alt_stride2_c` (`s`: state `[4]`, `input`: interleaved pairs).
pub fn biquad_alt_stride2(input: &[i16], b: &[i32; 3], a: &[i32; 2], s: &mut [i32; 4]) -> Vec<i16> {
    assert!(input.len().is_multiple_of(2));
    let mut out = vec![0i16; input.len()];
    // SAFETY: input/out have `2 * len` elements, coefficient/state arrays have fixed sizes.
    unsafe {
        oracle_silk_biquad_alt_stride2(
            input.as_ptr(),
            b.as_ptr(),
            a.as_ptr(),
            s.as_mut_ptr(),
            out.as_mut_ptr(),
            (input.len() / 2) as c_int,
        );
    };
    out
}
/// C `silk_LP_variable_cutoff`; `st` = `{In_LP_State[0..2], transition_frame_no, mode,
/// saved_fs_kHz}`.
pub fn lp_variable_cutoff(st: &mut [i32; 5], frame: &mut [i16]) {
    // SAFETY: `st` has 5 elements, `frame` has `len`.
    unsafe {
        oracle_silk_LP_variable_cutoff(st.as_mut_ptr(), frame.as_mut_ptr(), frame.len() as c_int)
    };
}
/// C `silk_ana_filt_bank_1`; returns `(low, high)`.
pub fn ana_filt_bank_1(input: &[i16], s: &mut [i32; 2]) -> (Vec<i16>, Vec<i16>) {
    let n2 = input.len() / 2;
    let mut l = vec![0i16; n2];
    let mut h = vec![0i16; n2];
    // SAFETY: outputs have `N/2` elements.
    unsafe {
        oracle_silk_ana_filt_bank_1(
            input.as_ptr(),
            s.as_mut_ptr(),
            l.as_mut_ptr(),
            h.as_mut_ptr(),
            input.len() as c_int,
        );
    };
    (l, h)
}
/// C `silk_LPC_inverse_pred_gain_c`.
#[must_use]
pub fn lpc_inverse_pred_gain(a_q12: &[i16]) -> i32 {
    assert!(!a_q12.is_empty() && a_q12.len() <= 24);
    // SAFETY: `a_q12` has `order` elements.
    unsafe { oracle_silk_LPC_inverse_pred_gain(a_q12.as_ptr(), a_q12.len() as c_int) }
}
/// C `silk_LPC_fit`; returns `a_QOUT` (and updates `a_qin`).
pub fn lpc_fit(a_qin: &mut [i32], qout: i32, qin: i32) -> Vec<i16> {
    let mut out = vec![0i16; a_qin.len()];
    // SAFETY: both buffers have `d` elements.
    unsafe {
        oracle_silk_LPC_fit(
            out.as_mut_ptr(),
            a_qin.as_mut_ptr(),
            qout,
            qin,
            a_qin.len() as c_int,
        )
    };
    out
}
/// C `silk_LPC_analysis_filter`.
#[must_use]
pub fn lpc_analysis_filter(input: &[i16], b: &[i16]) -> Vec<i16> {
    assert!(b.len() <= input.len());
    let mut out = vec![0i16; input.len()];
    // SAFETY: in/out have `len` elements, `b` has `d`.
    unsafe {
        oracle_silk_LPC_analysis_filter(
            out.as_mut_ptr(),
            input.as_ptr(),
            b.as_ptr(),
            input.len() as c_int,
            b.len() as c_int,
        );
    };
    out
}

// ---------------------------------------------------------------------------------------------
// NLSF
// ---------------------------------------------------------------------------------------------

/// C `silk_NLSF2A`.
#[must_use]
pub fn nlsf2a(nlsf: &[i16]) -> Vec<i16> {
    assert!(nlsf.len() == 10 || nlsf.len() == 16);
    let mut a = vec![0i16; nlsf.len()];
    // SAFETY: both buffers have `d` elements.
    unsafe { oracle_silk_NLSF2A(a.as_mut_ptr(), nlsf.as_ptr(), nlsf.len() as c_int) };
    a
}
/// C `silk_A2NLSF`; returns the NLSFs (and updates `a_q16`).
pub fn a2nlsf(a_q16: &mut [i32]) -> Vec<i16> {
    assert!(a_q16.len().is_multiple_of(2) && a_q16.len() <= 24);
    let mut nlsf = vec![0i16; a_q16.len()];
    // SAFETY: both buffers have `d` elements.
    unsafe { oracle_silk_A2NLSF(nlsf.as_mut_ptr(), a_q16.as_mut_ptr(), a_q16.len() as c_int) };
    nlsf
}
/// C `silk_NLSF_decode` with the NB/MB (`wb == false`) or WB codebook.
#[must_use]
pub fn nlsf_decode(indices: &[i8; 17], wb: bool) -> Vec<i16> {
    let mut idx = *indices;
    let mut nlsf = vec![0i16; 16];
    // SAFETY: 17 indices and 16 outputs cover both codebook orders.
    unsafe { oracle_silk_NLSF_decode(nlsf.as_mut_ptr(), idx.as_mut_ptr(), c_int::from(wb)) };
    nlsf.truncate(if wb { 16 } else { 10 });
    nlsf
}
/// C `silk_NLSF_unpack`; returns `(ec_ix, pred_q8)`.
#[must_use]
pub fn nlsf_unpack(wb: bool, cb1_index: i32) -> (Vec<i16>, Vec<u8>) {
    let mut ec_ix = vec![0i16; 16];
    let mut pred = vec![0u8; 16];
    // SAFETY: outputs have room for 16 entries (max order).
    unsafe {
        oracle_silk_NLSF_unpack(
            ec_ix.as_mut_ptr(),
            pred.as_mut_ptr(),
            c_int::from(wb),
            cb1_index,
        )
    };
    let order = if wb { 16 } else { 10 };
    ec_ix.truncate(order);
    pred.truncate(order);
    (ec_ix, pred)
}
/// C `silk_NLSF_stabilize` (`ndelta` has `nlsf.len() + 1` entries).
pub fn nlsf_stabilize(nlsf: &mut [i16], ndelta: &[i16]) {
    assert_eq!(ndelta.len(), nlsf.len() + 1);
    // SAFETY: buffer sizes checked above.
    unsafe { oracle_silk_NLSF_stabilize(nlsf.as_mut_ptr(), ndelta.as_ptr(), nlsf.len() as c_int) };
}
/// C `silk_NLSF_VQ_weights_laroia`.
#[must_use]
pub fn nlsf_vq_weights_laroia(nlsf: &[i16]) -> Vec<i16> {
    assert!(nlsf.len() >= 2 && nlsf.len().is_multiple_of(2));
    let mut out = vec![0i16; nlsf.len()];
    // SAFETY: both buffers have `D` elements.
    unsafe {
        oracle_silk_NLSF_VQ_weights_laroia(out.as_mut_ptr(), nlsf.as_ptr(), nlsf.len() as c_int)
    };
    out
}
/// C `silk_NLSF_VQ` over `k` codebook vectors of `input.len()` entries.
#[must_use]
pub fn nlsf_vq(input: &[i16], cb: &[u8], w: &[i16], k: usize) -> Vec<i32> {
    let order = input.len();
    assert!(cb.len() >= k * order && w.len() >= k * order);
    let mut err = vec![0i32; k];
    // SAFETY: sizes checked above.
    unsafe {
        oracle_silk_NLSF_VQ(
            err.as_mut_ptr(),
            input.as_ptr(),
            cb.as_ptr(),
            w.as_ptr(),
            k as c_int,
            order as c_int,
        );
    };
    err
}
/// C `silk_NLSF_encode`; returns `(indices, RD_Q25)` and updates `nlsf`.
pub fn nlsf_encode(
    nlsf: &mut [i16],
    wb: bool,
    w_q2: &[i16],
    mu_q20: i32,
    n_survivors: i32,
    signal_type: i32,
) -> ([i8; 17], i32) {
    let order = if wb { 16 } else { 10 };
    assert!(nlsf.len() == order && w_q2.len() == order);
    assert!((1..=32).contains(&n_survivors));
    let mut idx = [0i8; 17];
    // SAFETY: sizes checked above.
    let rd = unsafe {
        oracle_silk_NLSF_encode(
            idx.as_mut_ptr(),
            nlsf.as_mut_ptr(),
            c_int::from(wb),
            w_q2.as_ptr(),
            mu_q20,
            n_survivors,
            signal_type,
        )
    };
    (idx, rd)
}
/// C `silk_NLSF_del_dec_quant`; returns `(indices, RD_Q25)`.
#[expect(clippy::too_many_arguments, reason = "mirrors C signature")]
#[must_use]
pub fn nlsf_del_dec_quant(
    x_q10: &[i16],
    w_q5: &[i16],
    pred_coef_q8: &[u8],
    ec_ix: &[i16],
    ec_rates_q5: &[u8],
    quant_step_size_q16: i32,
    inv_quant_step_size_q6: i16,
    mu_q20: i32,
) -> (Vec<i8>, i32) {
    let order = x_q10.len();
    assert!(w_q5.len() == order && pred_coef_q8.len() == order && ec_ix.len() == order);
    assert!(order <= 16);
    let max_ix = ec_ix.iter().fold(0usize, |m, &v| m.max(v as usize));
    assert!(ec_rates_q5.len() >= max_ix + 9);
    let mut idx = vec![0i8; order];
    // SAFETY: sizes checked above.
    let rd = unsafe {
        oracle_silk_NLSF_del_dec_quant(
            idx.as_mut_ptr(),
            x_q10.as_ptr(),
            w_q5.as_ptr(),
            pred_coef_q8.as_ptr(),
            ec_ix.as_ptr(),
            ec_rates_q5.as_ptr(),
            quant_step_size_q16,
            inv_quant_step_size_q6 as c_int,
            mu_q20,
            order as c_int,
        )
    };
    (idx, rd)
}

// ---------------------------------------------------------------------------------------------
// Entropy coding helpers
// ---------------------------------------------------------------------------------------------

/// Result of an encoder-side shim: the full buffer and `[tell_frac before done, rng, error]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncRes {
    /// Output buffer (full size).
    pub buf: Vec<u8>,
    /// `[ec_tell_frac before ec_enc_done, rng after done, error]`.
    pub res: [u32; 3],
}

/// Result of a decoder-side shim: `[tell_frac, rng, error]` after the operation.
pub type DecRes = [u32; 3];

/// `silk_encode_signs` after `pre_n` raw bits `pre`.
#[expect(clippy::too_many_arguments, reason = "mirrors C signature")]
#[must_use]
pub fn encode_signs(
    size: usize,
    pre: u32,
    pre_n: i32,
    pulses: &[i8],
    length: i32,
    signal_type: i32,
    quant_offset_type: i32,
    sum_pulses: &[i32],
) -> EncRes {
    let blocks = ((length + 8) >> 4) as usize;
    assert!(pulses.len() >= blocks * 16 && sum_pulses.len() >= blocks);
    let mut buf = vec![0u8; size];
    let mut res = [0u32; 3];
    // SAFETY: sizes checked above.
    unsafe {
        oracle_silk_encode_signs(
            buf.as_mut_ptr(),
            size as c_int,
            pre,
            pre_n,
            pulses.as_ptr(),
            length,
            signal_type,
            quant_offset_type,
            sum_pulses.as_ptr(),
            res.as_mut_ptr(),
        );
    };
    EncRes { buf, res }
}
/// `silk_decode_signs` after `pre_n` raw bits.
pub fn decode_signs(
    data: &[u8],
    pre_n: i32,
    pulses: &mut [i16],
    length: i32,
    signal_type: i32,
    quant_offset_type: i32,
    sum_pulses: &[i32],
) -> DecRes {
    let blocks = ((length + 8) >> 4) as usize;
    assert!(pulses.len() >= blocks * 16 && sum_pulses.len() >= blocks);
    let mut buf = data.to_vec();
    let mut res = [0u32; 3];
    // SAFETY: sizes checked above; the decoder only reads `buf`.
    unsafe {
        oracle_silk_decode_signs(
            buf.as_mut_ptr(),
            buf.len() as c_int,
            pre_n,
            pulses.as_mut_ptr(),
            length,
            signal_type,
            quant_offset_type,
            sum_pulses.as_ptr(),
            res.as_mut_ptr(),
        );
    };
    res
}
/// `silk_shell_encoder` on 16 pulses.
#[must_use]
pub fn shell_encoder(size: usize, pre: u32, pre_n: i32, pulses0: &[i32; 16]) -> EncRes {
    let mut buf = vec![0u8; size];
    let mut res = [0u32; 3];
    // SAFETY: 16 pulses as the shim expects.
    unsafe {
        oracle_silk_shell_encoder(
            buf.as_mut_ptr(),
            size as c_int,
            pre,
            pre_n,
            pulses0.as_ptr(),
            res.as_mut_ptr(),
        );
    };
    EncRes { buf, res }
}
/// `silk_shell_decoder`; returns the 16 pulses.
#[must_use]
pub fn shell_decoder(data: &[u8], pre_n: i32, pulses4: i32) -> ([i16; 16], DecRes) {
    let mut buf = data.to_vec();
    let mut out = [0i16; 16];
    let mut res = [0u32; 3];
    // SAFETY: 16 outputs as the shim expects.
    unsafe {
        oracle_silk_shell_decoder(
            buf.as_mut_ptr(),
            buf.len() as c_int,
            pre_n,
            out.as_mut_ptr(),
            pulses4,
            res.as_mut_ptr(),
        );
    };
    (out, res)
}
/// `silk_encode_pulses` (`pulses` must be padded to a multiple of 16; it may be modified).
#[must_use]
pub fn encode_pulses(
    size: usize,
    pre: u32,
    pre_n: i32,
    signal_type: i32,
    quant_offset_type: i32,
    pulses: &mut [i8],
    frame_length: i32,
) -> EncRes {
    // C zeroes `pulses[frame_length..frame_length + 16]` when frame_length % 16 != 0.
    let need = if frame_length % 16 != 0 {
        frame_length as usize + 16
    } else {
        frame_length as usize
    };
    assert!(pulses.len() >= need);
    let mut buf = vec![0u8; size];
    let mut res = [0u32; 3];
    // SAFETY: sizes checked above.
    unsafe {
        oracle_silk_encode_pulses(
            buf.as_mut_ptr(),
            size as c_int,
            pre,
            pre_n,
            signal_type,
            quant_offset_type,
            pulses.as_mut_ptr(),
            frame_length,
            res.as_mut_ptr(),
        );
    };
    EncRes { buf, res }
}
/// `silk_decode_pulses`; returns the (padded) pulses.
#[must_use]
pub fn decode_pulses(
    data: &[u8],
    pre_n: i32,
    signal_type: i32,
    quant_offset_type: i32,
    frame_length: i32,
) -> (Vec<i16>, DecRes) {
    let mut buf = data.to_vec();
    let mut out = vec![0i16; (frame_length as usize + 15) & !15];
    let mut res = [0u32; 3];
    // SAFETY: output padded to a multiple of 16 as the C decoder requires.
    unsafe {
        oracle_silk_decode_pulses(
            buf.as_mut_ptr(),
            buf.len() as c_int,
            pre_n,
            out.as_mut_ptr(),
            signal_type,
            quant_offset_type,
            frame_length,
            res.as_mut_ptr(),
        );
    };
    (out, res)
}
/// `silk_stereo_encode_pred` (+ `silk_stereo_encode_mid_only` if `mid_only >= 0`).
#[must_use]
pub fn stereo_encode(
    size: usize,
    pre: u32,
    pre_n: i32,
    ix: &[[i8; 3]; 2],
    mid_only: i32,
) -> EncRes {
    let mut buf = vec![0u8; size];
    let mut res = [0u32; 3];
    let flat = [ix[0][0], ix[0][1], ix[0][2], ix[1][0], ix[1][1], ix[1][2]];
    // SAFETY: 6 indices as the shim expects.
    unsafe {
        oracle_silk_stereo_encode(
            buf.as_mut_ptr(),
            size as c_int,
            pre,
            pre_n,
            flat.as_ptr(),
            mid_only,
            res.as_mut_ptr(),
        );
    };
    EncRes { buf, res }
}
/// `silk_stereo_decode_pred` (+ `silk_stereo_decode_mid_only` if `want_mid`).
#[must_use]
pub fn stereo_decode(data: &[u8], pre_n: i32, want_mid: bool) -> ([i32; 2], i32, DecRes) {
    let mut buf = data.to_vec();
    let mut pred = [0i32; 2];
    let mut mid: c_int = 0;
    let mut res = [0u32; 3];
    // SAFETY: two predictor outputs, one flag output.
    unsafe {
        oracle_silk_stereo_decode(
            buf.as_mut_ptr(),
            buf.len() as c_int,
            pre_n,
            pred.as_mut_ptr(),
            c_int::from(want_mid),
            &mut mid,
            res.as_mut_ptr(),
        );
    };
    (pred, mid, res)
}
/// `silk_gains_quant`.
pub fn gains_quant(
    ind: &mut [i8; 4],
    gain_q16: &mut [i32; 4],
    prev_ind: &mut i8,
    conditional: i32,
    nb_subfr: i32,
) {
    assert!((1..=4).contains(&nb_subfr));
    // SAFETY: arrays hold MAX_NB_SUBFR entries.
    unsafe {
        oracle_silk_gains_quant(
            ind.as_mut_ptr(),
            gain_q16.as_mut_ptr(),
            prev_ind,
            conditional,
            nb_subfr,
        )
    };
}
/// `silk_gains_dequant`.
pub fn gains_dequant(
    gain_q16: &mut [i32; 4],
    ind: &[i8; 4],
    prev_ind: &mut i8,
    conditional: i32,
    nb_subfr: i32,
) {
    assert!((1..=4).contains(&nb_subfr));
    // SAFETY: arrays hold MAX_NB_SUBFR entries.
    unsafe {
        oracle_silk_gains_dequant(
            gain_q16.as_mut_ptr(),
            ind.as_ptr(),
            prev_ind,
            conditional,
            nb_subfr,
        )
    };
}
/// `silk_gains_ID`.
#[must_use]
pub fn gains_id(ind: &[i8; 4], nb_subfr: i32) -> i32 {
    assert!((1..=4).contains(&nb_subfr));
    // SAFETY: array holds MAX_NB_SUBFR entries.
    unsafe { oracle_silk_gains_ID(ind.as_ptr(), nb_subfr) }
}
/// `silk_decode_pitch`; returns `nb_subfr` lags.
#[must_use]
pub fn decode_pitch(lag_index: i16, contour_index: i8, fs_khz: i32, nb_subfr: i32) -> Vec<i32> {
    assert!(nb_subfr == 2 || nb_subfr == 4);
    let mut lags = vec![0 as c_int; 4];
    // SAFETY: room for 4 lags.
    unsafe {
        oracle_silk_decode_pitch(
            lag_index as c_int,
            contour_index as c_int,
            lags.as_mut_ptr(),
            fs_khz,
            nb_subfr,
        );
    };
    lags.truncate(nb_subfr as usize);
    lags
}
