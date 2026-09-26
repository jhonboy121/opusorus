//! Oracle bindings for unit `silk_resampler` (`silk/resampler*.c`), via `csrc/silk_resampler.c`.

use core::ffi::{c_int, c_void};

unsafe extern "C" {
    fn oracle_silk_resampler_new() -> *mut c_void;
    fn oracle_silk_resampler_free(s: *mut c_void);
    fn oracle_silk_resampler_init(
        s: *mut c_void,
        fs_in: c_int,
        fs_out: c_int,
        for_enc: c_int,
    ) -> c_int;
    fn oracle_silk_resampler(
        s: *mut c_void,
        out: *mut i16,
        input: *const i16,
        in_len: c_int,
    ) -> c_int;
    fn oracle_silk_resampler_private_IIR_FIR(
        s: *mut c_void,
        out: *mut i16,
        input: *const i16,
        in_len: c_int,
    );
    fn oracle_silk_resampler_private_down_FIR(
        s: *mut c_void,
        out: *mut i16,
        input: *const i16,
        in_len: c_int,
    );
    fn oracle_silk_resampler_private_up2_HQ_wrapper(
        s: *mut c_void,
        out: *mut i16,
        input: *const i16,
        len: c_int,
    );
    fn oracle_silk_resampler_get(
        s: *const c_void,
        s_iir: *mut i32,
        s_fir_i32: *mut i32,
        s_fir_i16: *mut i16,
        delay_buf: *mut i16,
        ints: *mut c_int,
    );
    fn oracle_silk_resampler_set_mem(
        s: *mut c_void,
        s_iir: *const i32,
        s_fir_i32: *const i32,
        s_fir_i16: *const i16,
        use_i16: c_int,
        delay_buf: *const i16,
    );
    fn oracle_silk_resampler_down2(s: *mut i32, out: *mut i16, input: *const i16, len: c_int);
    fn oracle_silk_resampler_down2_3(s: *mut i32, out: *mut i16, input: *const i16, len: c_int);
    fn oracle_silk_resampler_private_AR2(
        s: *mut i32,
        out_q8: *mut i32,
        input: *const i16,
        a_q14: *const i16,
        len: c_int,
    );
    fn oracle_silk_resampler_private_up2_HQ(
        s: *mut i32,
        out: *mut i16,
        input: *const i16,
        len: c_int,
    );
    fn oracle_silk_resampler_private_IIR_FIR_INTERPOL(
        out: *mut i16,
        buf: *mut i16,
        max_index_q16: c_int,
        index_increment_q16: c_int,
    ) -> c_int;
    fn oracle_silk_resampler_private_down_FIR_INTERPOL(
        out: *mut i16,
        buf: *mut i32,
        coefs_id: c_int,
        fir_order: c_int,
        fir_fracs: c_int,
        max_index_q16: c_int,
        index_increment_q16: c_int,
    ) -> c_int;
    fn oracle_silk_resampler_rom(out: *mut i16) -> c_int;
}

/// Full snapshot of a C `silk_resampler_state_struct`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateDump {
    /// `sIIR`.
    pub s_iir: [i32; 6],
    /// `sFIR.i32` view.
    pub s_fir_i32: [i32; 36],
    /// `sFIR.i16` view (aliases the first 18 entries of `s_fir_i32`).
    pub s_fir_i16: [i16; 36],
    /// `delayBuf`.
    pub delay_buf: [i16; 96],
    /// `resampler_function`.
    pub resampler_function: i32,
    /// `batchSize`.
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
    /// Which ROM table `Coefs` points to: 0 = NULL, 1 = 3_4, 2 = 2_3, 3 = 1_2, 4 = 1_3,
    /// 5 = 1_4, 6 = 1_6, -1 = unknown.
    pub coefs_id: i32,
}

/// Owned C resampler state (heap allocated in C).
#[derive(Debug)]
pub struct Resampler(*mut c_void);

impl Resampler {
    /// Allocates a zeroed state.
    ///
    /// # Panics
    /// If the C allocation fails.
    #[must_use]
    pub fn new() -> Self {
        // SAFETY: plain allocation, no arguments.
        let p = unsafe { oracle_silk_resampler_new() };
        assert!(!p.is_null(), "oracle_silk_resampler_new: allocation failed");
        Self(p)
    }

    /// `silk_resampler_init`.
    pub fn init(&mut self, fs_in: i32, fs_out: i32, for_enc: i32) -> i32 {
        // SAFETY: self.0 is a valid, exclusively owned state.
        unsafe { oracle_silk_resampler_init(self.0, fs_in, fs_out, for_enc) }
    }

    /// `silk_resampler`; `out` must be large enough for the output of `input.len()` samples.
    pub fn resample(&mut self, out: &mut [i16], input: &[i16]) -> i32 {
        // SAFETY: valid state; `input` holds `input.len()` samples; the caller sizes `out` for
        // the resampled length (the C code writes exactly that many samples).
        unsafe {
            oracle_silk_resampler(
                self.0,
                out.as_mut_ptr(),
                input.as_ptr(),
                input.len() as c_int,
            )
        }
    }

    /// `silk_resampler_private_IIR_FIR` on this state.
    pub fn private_iir_fir(&mut self, out: &mut [i16], input: &[i16]) {
        // SAFETY: as `resample`; the caller sizes `out`.
        unsafe {
            oracle_silk_resampler_private_IIR_FIR(
                self.0,
                out.as_mut_ptr(),
                input.as_ptr(),
                input.len() as c_int,
            );
        }
    }

    /// `silk_resampler_private_down_FIR` on this state.
    pub fn private_down_fir(&mut self, out: &mut [i16], input: &[i16]) {
        // SAFETY: as `resample`; the caller sizes `out`.
        unsafe {
            oracle_silk_resampler_private_down_FIR(
                self.0,
                out.as_mut_ptr(),
                input.as_ptr(),
                input.len() as c_int,
            );
        }
    }

    /// `silk_resampler_private_up2_HQ_wrapper` on this state.
    pub fn private_up2_hq_wrapper(&mut self, out: &mut [i16], input: &[i16]) {
        assert!(out.len() >= 2 * input.len());
        // SAFETY: valid state; `out` holds 2 * len samples as checked.
        unsafe {
            oracle_silk_resampler_private_up2_HQ_wrapper(
                self.0,
                out.as_mut_ptr(),
                input.as_ptr(),
                input.len() as c_int,
            );
        }
    }

    /// Snapshot of the whole state.
    #[must_use]
    pub fn dump(&self) -> StateDump {
        let mut d = StateDump {
            s_iir: [0; 6],
            s_fir_i32: [0; 36],
            s_fir_i16: [0; 36],
            delay_buf: [0; 96],
            resampler_function: 0,
            batch_size: 0,
            inv_ratio_q16: 0,
            fir_order: 0,
            fir_fracs: 0,
            fs_in_khz: 0,
            fs_out_khz: 0,
            input_delay: 0,
            coefs_id: 0,
        };
        let mut ints = [0 as c_int; 9];
        // SAFETY: all output arrays have the sizes the shim writes (6, 36, 36, 96, 9).
        unsafe {
            oracle_silk_resampler_get(
                self.0,
                d.s_iir.as_mut_ptr(),
                d.s_fir_i32.as_mut_ptr(),
                d.s_fir_i16.as_mut_ptr(),
                d.delay_buf.as_mut_ptr(),
                ints.as_mut_ptr(),
            );
        }
        [
            d.resampler_function,
            d.batch_size,
            d.inv_ratio_q16,
            d.fir_order,
            d.fir_fracs,
            d.fs_in_khz,
            d.fs_out_khz,
            d.input_delay,
            d.coefs_id,
        ] = ints;
        d
    }

    /// Overwrites the filter memories; `use_i16` selects the `sFIR` union view written.
    pub fn set_mem(
        &mut self,
        s_iir: &[i32; 6],
        s_fir_i32: &[i32; 36],
        s_fir_i16: &[i16; 36],
        use_i16: bool,
        delay_buf: &[i16; 96],
    ) {
        // SAFETY: arrays have the exact sizes the shim reads.
        unsafe {
            oracle_silk_resampler_set_mem(
                self.0,
                s_iir.as_ptr(),
                s_fir_i32.as_ptr(),
                s_fir_i16.as_ptr(),
                c_int::from(use_i16),
                delay_buf.as_ptr(),
            );
        }
    }
}

impl Default for Resampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Resampler {
    fn drop(&mut self) {
        // SAFETY: self.0 came from oracle_silk_resampler_new and is freed exactly once.
        unsafe { oracle_silk_resampler_free(self.0) }
    }
}

/// `silk_resampler_down2`: `s` state `[2]`, returns `floor(len/2)` samples.
pub fn down2(s: &mut [i32; 2], input: &[i16]) -> Vec<i16> {
    let mut out = vec![0i16; input.len() / 2];
    // SAFETY: state has 2 entries; out holds floor(len/2) samples.
    unsafe {
        oracle_silk_resampler_down2(
            s.as_mut_ptr(),
            out.as_mut_ptr(),
            input.as_ptr(),
            input.len() as c_int,
        );
    }
    out
}

/// `silk_resampler_down2_3`: `s` state `[6]`, writes into `out` (sized by the caller, at least
/// `2 * len / 3 + 2`).
pub fn down2_3(s: &mut [i32; 6], out: &mut [i16], input: &[i16]) {
    assert!(out.len() >= 2 * input.len() / 3 + 2);
    // SAFETY: state has 6 entries; out is large enough for the samples written.
    unsafe {
        oracle_silk_resampler_down2_3(
            s.as_mut_ptr(),
            out.as_mut_ptr(),
            input.as_ptr(),
            input.len() as c_int,
        );
    }
}

/// `silk_resampler_private_AR2`: `s` state `[2]`, returns `len` Q8 samples.
pub fn private_ar2(s: &mut [i32; 2], input: &[i16], a_q14: &[i16; 2]) -> Vec<i32> {
    let mut out = vec![0i32; input.len()];
    // SAFETY: state has 2 entries; out holds len samples; a_q14 has 2 entries.
    unsafe {
        oracle_silk_resampler_private_AR2(
            s.as_mut_ptr(),
            out.as_mut_ptr(),
            input.as_ptr(),
            a_q14.as_ptr(),
            input.len() as c_int,
        );
    }
    out
}

/// `silk_resampler_private_up2_HQ`: `s` state `[6]`, returns `2 * len` samples.
pub fn private_up2_hq(s: &mut [i32; 6], input: &[i16]) -> Vec<i16> {
    let mut out = vec![0i16; 2 * input.len()];
    // SAFETY: state has 6 entries; out holds 2 * len samples.
    unsafe {
        oracle_silk_resampler_private_up2_HQ(
            s.as_mut_ptr(),
            out.as_mut_ptr(),
            input.as_ptr(),
            input.len() as c_int,
        );
    }
    out
}

/// `silk_resampler_private_IIR_FIR_INTERPOL`. The caller guarantees `buf` covers
/// `(max_index_q16 - 1) >> 16` + 8 samples and `out` the number of outputs; returns the count.
pub fn private_iir_fir_interpol(
    out: &mut [i16],
    buf: &[i16],
    max_index_q16: i32,
    index_increment_q16: i32,
) -> usize {
    assert!(index_increment_q16 > 0);
    let n_out = if max_index_q16 > 0 {
        ((max_index_q16 as i64 + index_increment_q16 as i64 - 1) / index_increment_q16 as i64)
            as usize
    } else {
        0
    };
    assert!(out.len() >= n_out);
    if n_out > 0 {
        assert!(buf.len() >= ((max_index_q16 - 1) >> 16) as usize + 8);
    }
    let mut b = buf.to_vec();
    // SAFETY: sizes checked above; the C function only reads `buf` (non-const in its signature).
    let n = unsafe {
        oracle_silk_resampler_private_IIR_FIR_INTERPOL(
            out.as_mut_ptr(),
            b.as_mut_ptr(),
            max_index_q16,
            index_increment_q16,
        )
    };
    n as usize
}

/// `silk_resampler_private_down_FIR_INTERPOL` with the FIR part of ROM table `coefs_id`
/// (1..=6 as in [`StateDump::coefs_id`]). Returns the output count.
pub fn private_down_fir_interpol(
    out: &mut [i16],
    buf: &[i32],
    coefs_id: i32,
    fir_order: i32,
    fir_fracs: i32,
    max_index_q16: i32,
    index_increment_q16: i32,
) -> usize {
    assert!((1..=6).contains(&coefs_id));
    assert!(index_increment_q16 > 0);
    let n_out = if max_index_q16 > 0 {
        ((max_index_q16 as i64 + index_increment_q16 as i64 - 1) / index_increment_q16 as i64)
            as usize
    } else {
        0
    };
    assert!(out.len() >= n_out);
    if n_out > 0 {
        assert!(buf.len() >= ((max_index_q16 - 1) >> 16) as usize + fir_order as usize);
    }
    let mut b = buf.to_vec();
    // SAFETY: sizes checked above; the C function only reads `buf`.
    let n = unsafe {
        oracle_silk_resampler_private_down_FIR_INTERPOL(
            out.as_mut_ptr(),
            b.as_mut_ptr(),
            coefs_id,
            fir_order,
            fir_fracs,
            max_index_q16,
            index_increment_q16,
        )
    };
    n as usize
}

/// All resampler ROM tables concatenated (see `oracle_silk_resampler_rom` for the order).
#[must_use]
pub fn rom() -> Vec<i16> {
    let mut out = vec![0i16; 512];
    // SAFETY: the shim writes fewer than 256 values.
    let n = unsafe { oracle_silk_resampler_rom(out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}
