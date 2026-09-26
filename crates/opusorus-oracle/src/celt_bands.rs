//! Oracle bindings for unit `celt_bands`: celt/vq.c, celt/quant_bands.c, celt/bands.c and
//! celt/celt.c (float build).
//!
//! Modes are selected by sampling rate (`48000`, or `96000` with QEXT) plus a `qext` flag that
//! substitutes `compute_qext_mode(mode)`. Wrappers assert the slice lengths the C code touches.

#![allow(
    clippy::too_many_arguments,
    reason = "wrappers mirror the flat C shim signatures"
)]

use core::ffi::c_int;

unsafe extern "C" {
    fn oracle_cb_has_qext() -> c_int;
    fn oracle_resampling_factor(rate: c_int) -> c_int;
    fn oracle_init_caps(fs: c_int, qext: c_int, cap: *mut c_int, lm: c_int, c: c_int);
    fn oracle_celt_tables(tf: *mut i8, trim: *mut u8, spread: *mut u8, tapset: *mut u8);
    fn oracle_comb_filter(
        y: *mut f32,
        x_base: *mut f32,
        x_off: c_int,
        inplace: c_int,
        t0: c_int,
        t1: c_int,
        n: c_int,
        g0: f32,
        g1: f32,
        tapset0: c_int,
        tapset1: c_int,
        fs: c_int,
        use_window: c_int,
        overlap: c_int,
    );
    fn oracle_exp_rotation(
        x: *mut f32,
        len: c_int,
        dir: c_int,
        stride: c_int,
        k: c_int,
        spread: c_int,
    );
    fn oracle_op_pvq_search(x: *mut f32, iy: *mut c_int, k: c_int, n: c_int) -> f32;
    fn oracle_alg_quant(
        x: *mut f32,
        n: c_int,
        k: c_int,
        spread: c_int,
        b: c_int,
        buf: *mut u8,
        size: c_int,
        gain: f32,
        resynth: c_int,
        ext_buf: *mut u8,
        ext_size: c_int,
        extra_bits: c_int,
        out: *mut u32,
    );
    fn oracle_alg_unquant(
        x: *mut f32,
        n: c_int,
        k: c_int,
        spread: c_int,
        b: c_int,
        buf: *mut u8,
        size: c_int,
        gain: f32,
        ext_buf: *mut u8,
        ext_size: c_int,
        extra_bits: c_int,
        out: *mut u32,
    );
    fn oracle_renormalise_vector(x: *mut f32, n: c_int, gain: f32);
    fn oracle_stereo_itheta(x: *const f32, y: *const f32, stereo: c_int, n: c_int) -> c_int;
    fn oracle_cubic_quant(
        x: *mut f32,
        n: c_int,
        res: c_int,
        b: c_int,
        buf: *mut u8,
        size: c_int,
        gain: f32,
        resynth: c_int,
        out: *mut u32,
    );
    fn oracle_cubic_unquant(
        x: *mut f32,
        n: c_int,
        res: c_int,
        b: c_int,
        buf: *mut u8,
        size: c_int,
        gain: f32,
        out: *mut u32,
    );
    fn oracle_hysteresis_decision(
        val: f32,
        thresholds: *const f32,
        hysteresis: *const f32,
        n: c_int,
        prev: c_int,
    ) -> c_int;
    fn oracle_celt_lcg_rand(seed: u32) -> u32;
    fn oracle_bitexact_cos(x: c_int) -> c_int;
    fn oracle_bitexact_log2tan(isin: c_int, icos: c_int) -> c_int;
    fn oracle_compute_band_energies(
        fs: c_int,
        qext: c_int,
        x: *const f32,
        band_e: *mut f32,
        end: c_int,
        c: c_int,
        lm: c_int,
    );
    fn oracle_normalise_bands(
        fs: c_int,
        qext: c_int,
        freq: *const f32,
        x: *mut f32,
        band_e: *const f32,
        end: c_int,
        c: c_int,
        m: c_int,
    );
    fn oracle_denormalise_bands(
        fs: c_int,
        qext: c_int,
        x: *const f32,
        freq: *mut f32,
        band_log_e: *const f32,
        start: c_int,
        end: c_int,
        m: c_int,
        downsample: c_int,
        silence: c_int,
    );
    fn oracle_anti_collapse(
        fs: c_int,
        qext: c_int,
        x: *mut f32,
        cm: *mut u8,
        lm: c_int,
        c: c_int,
        size: c_int,
        start: c_int,
        end: c_int,
        log_e: *const f32,
        prev1: *const f32,
        prev2: *const f32,
        pulses: *const c_int,
        seed: u32,
        encode: c_int,
    );
    fn oracle_spreading_decision(
        fs: c_int,
        x: *const f32,
        io: *mut c_int,
        last_decision: c_int,
        update_hf: c_int,
        end: c_int,
        c: c_int,
        m: c_int,
        spread_weight: *const c_int,
    ) -> c_int;
    fn oracle_haar1(x: *mut f32, n0: c_int, stride: c_int);
    fn oracle_quant_all_bands(
        ip: *const c_int,
        x: *mut f32,
        collapse_masks: *mut u8,
        band_e: *const f32,
        pulses: *mut c_int,
        tf_res: *const c_int,
        offsets: *const c_int,
        cap: *const c_int,
        extra_pulses: *const c_int,
        buf: *mut u8,
        ext_buf: *mut u8,
        seed: *mut u32,
        out: *mut c_int,
    );
    fn oracle_emeans(out: *mut f32);
    fn oracle_amp2log2(
        fs: c_int,
        qext: c_int,
        eff_end: c_int,
        end: c_int,
        band_e: *mut f32,
        band_log_e: *mut f32,
        c: c_int,
    );
    fn oracle_quant_energy(
        ip: *const c_int,
        e_bands: *const f32,
        old_e_bands: *mut f32,
        error: *mut f32,
        buf: *mut u8,
        delayed_intra: *mut f32,
        fine_quant: *mut c_int,
        prev_quant: *mut c_int,
        fine_priority: *mut c_int,
        out: *mut c_int,
    );
    fn oracle_unquant_energy(
        ip: *const c_int,
        old_e_bands: *mut f32,
        buf: *mut u8,
        fine_quant: *mut c_int,
        prev_quant: *mut c_int,
        fine_priority: *mut c_int,
        out: *mut c_int,
    );
    fn oracle_exp_rotation1(x: *mut f32, len: c_int, stride: c_int, c: f32, s: f32);
    fn oracle_normalise_residual(iy: *mut c_int, x: *mut f32, n: c_int, ryy: f32, gain: f32);
    fn oracle_extract_collapse_mask(iy: *mut c_int, n: c_int, b: c_int) -> u32;
    fn oracle_op_pvq_search_n2(
        x: *const f32,
        iy: *mut c_int,
        up_iy: *mut c_int,
        k: c_int,
        up: c_int,
        refine: *mut c_int,
    ) -> f32;
    fn oracle_op_pvq_search_extra(
        x: *const f32,
        iy: *mut c_int,
        up_iy: *mut c_int,
        k: c_int,
        up: c_int,
        refine: *mut c_int,
        n: c_int,
    ) -> f32;
    fn oracle_compute_qn(
        n: c_int,
        b: c_int,
        offset: c_int,
        pulse_cap: c_int,
        stereo: c_int,
    ) -> c_int;
    fn oracle_compute_channel_weights(ex: f32, ey: f32, w: *mut f32);
    fn oracle_intensity_stereo(
        fs: c_int,
        x: *mut f32,
        y: *const f32,
        band_e: *const f32,
        band_id: c_int,
        n: c_int,
    );
    fn oracle_stereo_split(x: *mut f32, y: *mut f32, n: c_int);
    fn oracle_stereo_merge(x: *mut f32, y: *mut f32, mid: f32, n: c_int);
    fn oracle_deinterleave_hadamard(x: *mut f32, n0: c_int, stride: c_int, hadamard: c_int);
    fn oracle_interleave_hadamard(x: *mut f32, n0: c_int, stride: c_int, hadamard: c_int);
    fn oracle_loss_distortion(
        e_bands: *const f32,
        old_e_bands: *mut f32,
        start: c_int,
        end: c_int,
        len: c_int,
        c: c_int,
    ) -> f32;
}

/// Whether the oracle was built with `ENABLE_QEXT`.
pub fn has_qext() -> bool {
    // SAFETY: plain call.
    unsafe { oracle_cb_has_qext() != 0 }
}

// ---------------------------------------------------------------------------------------------
// celt.c
// ---------------------------------------------------------------------------------------------

/// `opus_strerror` (defined in celt.c).
pub fn strerror(error: i32) -> String {
    // SAFETY: opus_strerror returns a pointer to a static NUL-terminated string.
    unsafe { core::ffi::CStr::from_ptr(crate::sys::opus_strerror(error)) }
        .to_string_lossy()
        .into_owned()
}

/// `resampling_factor`.
pub fn resampling_factor(rate: i32) -> i32 {
    // SAFETY: plain call.
    unsafe { oracle_resampling_factor(rate) }
}

/// `init_caps` into `cap` (`nbEBands` entries).
pub fn init_caps(fs: i32, qext: bool, cap: &mut [i32], lm: i32, c: i32) {
    assert!(cap.len() >= 21);
    // SAFETY: cap holds nbEBands (<= 21) entries.
    unsafe { oracle_init_caps(fs, i32::from(qext), cap.as_mut_ptr(), lm, c) }
}

/// `(tf_select_table, trim_icdf, spread_icdf, tapset_icdf)`.
pub fn celt_tables() -> ([[i8; 8]; 4], [u8; 11], [u8; 4], [u8; 3]) {
    let mut tf = [[0i8; 8]; 4];
    let mut trim = [0u8; 11];
    let mut spread = [0u8; 4];
    let mut tapset = [0u8; 3];
    // SAFETY: buffers have the sizes of the C tables.
    unsafe {
        oracle_celt_tables(
            tf.as_mut_ptr().cast(),
            trim.as_mut_ptr(),
            spread.as_mut_ptr(),
            tapset.as_mut_ptr(),
        );
    }
    (tf, trim, spread, tapset)
}

/// Comb filter parameters.
#[derive(Debug, Clone, Copy)]
pub struct CombParams {
    pub t0: i32,
    pub t1: i32,
    pub n: i32,
    pub g0: f32,
    pub g1: f32,
    pub tapset0: i32,
    pub tapset1: i32,
    /// Mode whose window is used (`use_window`), or no window.
    pub fs: i32,
    pub use_window: bool,
    pub overlap: i32,
}

/// `comb_filter(y, x_base+x_off, ...)` (separate buffers).
pub fn comb_filter(y: &mut [f32], x_base: &mut [f32], x_off: usize, p: CombParams) {
    assert!(y.len() >= p.n as usize && x_base.len() >= x_off + p.n as usize);
    assert!(x_off >= 2 * 1024 + 2);
    // SAFETY: lengths checked, including enough history before x_off for any period.
    unsafe {
        oracle_comb_filter(
            y.as_mut_ptr(),
            x_base.as_mut_ptr(),
            x_off as c_int,
            0,
            p.t0,
            p.t1,
            p.n,
            p.g0,
            p.g1,
            p.tapset0,
            p.tapset1,
            p.fs,
            i32::from(p.use_window),
            p.overlap,
        );
    }
}

/// `comb_filter(x, x, ...)` in place on `buf[off..off+n]`.
pub fn comb_filter_inplace(buf: &mut [f32], off: usize, p: CombParams) {
    assert!(buf.len() >= off + p.n as usize && off >= 2 * 1024 + 2);
    let mut dummy = [0.0f32; 1];
    // SAFETY: lengths checked (see comb_filter).
    unsafe {
        oracle_comb_filter(
            dummy.as_mut_ptr(),
            buf.as_mut_ptr(),
            off as c_int,
            1,
            p.t0,
            p.t1,
            p.n,
            p.g0,
            p.g1,
            p.tapset0,
            p.tapset1,
            p.fs,
            i32::from(p.use_window),
            p.overlap,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// vq.c
// ---------------------------------------------------------------------------------------------

/// `exp_rotation`.
pub fn exp_rotation(x: &mut [f32], len: i32, dir: i32, stride: i32, k: i32, spread: i32) {
    assert!(x.len() >= len as usize);
    // SAFETY: x has len elements.
    unsafe { oracle_exp_rotation(x.as_mut_ptr(), len, dir, stride, k, spread) }
}

/// `op_pvq_search_c`.
pub fn op_pvq_search(x: &mut [f32], iy: &mut [i32], k: i32, n: i32) -> f32 {
    assert!(x.len() >= n as usize && iy.len() >= n as usize);
    // SAFETY: lengths checked.
    unsafe { oracle_op_pvq_search(x.as_mut_ptr(), iy.as_mut_ptr(), k, n) }
}

/// Result of an `alg_quant`/`alg_unquant`/cubic call: `[cm, tell_frac, rng, error, ext
/// tell_frac, ext rng, ext error]`.
pub type VqOut = [u32; 7];

/// `alg_quant` on fresh encoders over `buf`/`ext_buf` (finished with `ec_enc_done`).
pub fn alg_quant(
    x: &mut [f32],
    n: i32,
    k: i32,
    spread: i32,
    b: i32,
    buf: &mut [u8],
    gain: f32,
    resynth: bool,
    ext_buf: &mut [u8],
    extra_bits: i32,
) -> VqOut {
    assert!(x.len() >= n as usize);
    let mut out = [0u32; 7];
    // SAFETY: lengths checked; buffers are passed with their sizes.
    unsafe {
        oracle_alg_quant(
            x.as_mut_ptr(),
            n,
            k,
            spread,
            b,
            buf.as_mut_ptr(),
            buf.len() as c_int,
            gain,
            i32::from(resynth),
            ext_buf.as_mut_ptr(),
            ext_buf.len() as c_int,
            extra_bits,
            out.as_mut_ptr(),
        );
    }
    out
}

/// `alg_unquant` on fresh decoders.
pub fn alg_unquant(
    x: &mut [f32],
    n: i32,
    k: i32,
    spread: i32,
    b: i32,
    buf: &mut [u8],
    gain: f32,
    ext_buf: &mut [u8],
    extra_bits: i32,
) -> VqOut {
    assert!(x.len() >= n as usize);
    let mut out = [0u32; 7];
    // SAFETY: lengths checked; buffers are passed with their sizes.
    unsafe {
        oracle_alg_unquant(
            x.as_mut_ptr(),
            n,
            k,
            spread,
            b,
            buf.as_mut_ptr(),
            buf.len() as c_int,
            gain,
            ext_buf.as_mut_ptr(),
            ext_buf.len() as c_int,
            extra_bits,
            out.as_mut_ptr(),
        );
    }
    out
}

/// `renormalise_vector`.
pub fn renormalise_vector(x: &mut [f32], n: i32, gain: f32) {
    assert!(x.len() >= n as usize);
    // SAFETY: length checked.
    unsafe { oracle_renormalise_vector(x.as_mut_ptr(), n, gain) }
}

/// `stereo_itheta`.
pub fn stereo_itheta(x: &[f32], y: &[f32], stereo: bool, n: i32) -> i32 {
    assert!(x.len() >= n as usize && y.len() >= n as usize);
    // SAFETY: lengths checked.
    unsafe { oracle_stereo_itheta(x.as_ptr(), y.as_ptr(), i32::from(stereo), n) }
}

/// `cubic_quant` (QEXT) on a fresh encoder: `[cm, tell_frac, rng, error]`.
pub fn cubic_quant(
    x: &mut [f32],
    n: i32,
    res: i32,
    b: i32,
    buf: &mut [u8],
    gain: f32,
    resynth: bool,
) -> [u32; 4] {
    assert!(x.len() >= n as usize);
    let mut out = [0u32; 4];
    // SAFETY: lengths checked.
    unsafe {
        oracle_cubic_quant(
            x.as_mut_ptr(),
            n,
            res,
            b,
            buf.as_mut_ptr(),
            buf.len() as c_int,
            gain,
            i32::from(resynth),
            out.as_mut_ptr(),
        );
    }
    out
}

/// `cubic_unquant` (QEXT) on a fresh decoder: `[cm, tell_frac, rng, error]`.
pub fn cubic_unquant(
    x: &mut [f32],
    n: i32,
    res: i32,
    b: i32,
    buf: &mut [u8],
    gain: f32,
) -> [u32; 4] {
    assert!(x.len() >= n as usize);
    let mut out = [0u32; 4];
    // SAFETY: lengths checked.
    unsafe {
        oracle_cubic_unquant(
            x.as_mut_ptr(),
            n,
            res,
            b,
            buf.as_mut_ptr(),
            buf.len() as c_int,
            gain,
            out.as_mut_ptr(),
        );
    }
    out
}

/// `exp_rotation1` (static).
pub fn exp_rotation1(x: &mut [f32], len: i32, stride: i32, c: f32, s: f32) {
    assert!(x.len() >= len as usize);
    // SAFETY: length checked.
    unsafe { oracle_exp_rotation1(x.as_mut_ptr(), len, stride, c, s) }
}

/// `normalise_residual` (static).
pub fn normalise_residual(iy: &mut [i32], x: &mut [f32], n: i32, ryy: f32, gain: f32) {
    assert!(x.len() >= n as usize && iy.len() >= n as usize);
    // SAFETY: lengths checked.
    unsafe { oracle_normalise_residual(iy.as_mut_ptr(), x.as_mut_ptr(), n, ryy, gain) }
}

/// `extract_collapse_mask` (static).
pub fn extract_collapse_mask(iy: &mut [i32], n: i32, b: i32) -> u32 {
    assert!(iy.len() >= n as usize);
    // SAFETY: length checked.
    unsafe { oracle_extract_collapse_mask(iy.as_mut_ptr(), n, b) }
}

/// `op_pvq_search_N2` (static, QEXT): `(yy, refine)`.
pub fn op_pvq_search_n2(
    x: &[f32],
    iy: &mut [i32],
    up_iy: &mut [i32],
    k: i32,
    up: i32,
) -> (f32, i32) {
    assert!(x.len() >= 2 && iy.len() >= 2 && up_iy.len() >= 2);
    let mut refine = 0;
    // SAFETY: lengths checked.
    let yy = unsafe {
        oracle_op_pvq_search_n2(
            x.as_ptr(),
            iy.as_mut_ptr(),
            up_iy.as_mut_ptr(),
            k,
            up,
            &mut refine,
        )
    };
    (yy, refine)
}

/// `op_pvq_search_extra` (static, QEXT).
pub fn op_pvq_search_extra(
    x: &[f32],
    iy: &mut [i32],
    up_iy: &mut [i32],
    k: i32,
    up: i32,
    refine: &mut [i32],
    n: i32,
) -> f32 {
    let nu = n as usize;
    assert!(x.len() >= nu && iy.len() >= nu && up_iy.len() >= nu && refine.len() >= nu);
    // SAFETY: lengths checked.
    unsafe {
        oracle_op_pvq_search_extra(
            x.as_ptr(),
            iy.as_mut_ptr(),
            up_iy.as_mut_ptr(),
            k,
            up,
            refine.as_mut_ptr(),
            n,
        )
    }
}

// ---------------------------------------------------------------------------------------------
// bands.c
// ---------------------------------------------------------------------------------------------

/// `hysteresis_decision`.
pub fn hysteresis_decision(
    val: f32,
    thresholds: &[f32],
    hysteresis: &[f32],
    n: i32,
    prev: i32,
) -> i32 {
    assert!(thresholds.len() >= n as usize && hysteresis.len() >= n as usize);
    // SAFETY: lengths checked.
    unsafe { oracle_hysteresis_decision(val, thresholds.as_ptr(), hysteresis.as_ptr(), n, prev) }
}

/// `celt_lcg_rand`.
pub fn celt_lcg_rand(seed: u32) -> u32 {
    // SAFETY: plain call.
    unsafe { oracle_celt_lcg_rand(seed) }
}

/// `bitexact_cos`.
pub fn bitexact_cos(x: i16) -> i16 {
    // SAFETY: plain call.
    unsafe { oracle_bitexact_cos(i32::from(x)) as i16 }
}

/// `bitexact_log2tan`.
pub fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    // SAFETY: plain call.
    unsafe { oracle_bitexact_log2tan(isin, icos) }
}

/// `compute_band_energies`.
pub fn compute_band_energies(
    fs: i32,
    qext: bool,
    x: &[f32],
    band_e: &mut [f32],
    end: i32,
    c: i32,
    lm: i32,
) {
    // SAFETY: the test passes full-size buffers (C*N samples, C*nbEBands energies).
    unsafe {
        oracle_compute_band_energies(
            fs,
            i32::from(qext),
            x.as_ptr(),
            band_e.as_mut_ptr(),
            end,
            c,
            lm,
        );
    }
}

/// `normalise_bands`.
pub fn normalise_bands(
    fs: i32,
    qext: bool,
    freq: &[f32],
    x: &mut [f32],
    band_e: &[f32],
    end: i32,
    c: i32,
    m: i32,
) {
    assert_eq!(freq.len(), x.len());
    // SAFETY: full-size buffers.
    unsafe {
        oracle_normalise_bands(
            fs,
            i32::from(qext),
            freq.as_ptr(),
            x.as_mut_ptr(),
            band_e.as_ptr(),
            end,
            c,
            m,
        );
    }
}

/// `denormalise_bands`.
pub fn denormalise_bands(
    fs: i32,
    qext: bool,
    x: &[f32],
    freq: &mut [f32],
    band_log_e: &[f32],
    start: i32,
    end: i32,
    m: i32,
    downsample: i32,
    silence: bool,
) {
    // SAFETY: full-size buffers.
    unsafe {
        oracle_denormalise_bands(
            fs,
            i32::from(qext),
            x.as_ptr(),
            freq.as_mut_ptr(),
            band_log_e.as_ptr(),
            start,
            end,
            m,
            downsample,
            i32::from(silence),
        );
    }
}

/// `anti_collapse`.
pub fn anti_collapse(
    fs: i32,
    qext: bool,
    x: &mut [f32],
    cm: &mut [u8],
    lm: i32,
    c: i32,
    size: i32,
    start: i32,
    end: i32,
    log_e: &[f32],
    prev1: &[f32],
    prev2: &[f32],
    pulses: &[i32],
    seed: u32,
    encode: bool,
) {
    assert!(x.len() >= (c * size) as usize);
    // SAFETY: full-size buffers.
    unsafe {
        oracle_anti_collapse(
            fs,
            i32::from(qext),
            x.as_mut_ptr(),
            cm.as_mut_ptr(),
            lm,
            c,
            size,
            start,
            end,
            log_e.as_ptr(),
            prev1.as_ptr(),
            prev2.as_ptr(),
            pulses.as_ptr(),
            seed,
            i32::from(encode),
        );
    }
}

/// `spreading_decision`; `io = [average, hf_average, tapset_decision]` (in/out).
pub fn spreading_decision(
    fs: i32,
    x: &[f32],
    io: &mut [i32; 3],
    last_decision: i32,
    update_hf: bool,
    end: i32,
    c: i32,
    m: i32,
    spread_weight: &[i32],
) -> i32 {
    // SAFETY: full-size buffers.
    unsafe {
        oracle_spreading_decision(
            fs,
            x.as_ptr(),
            io.as_mut_ptr(),
            last_decision,
            i32::from(update_hf),
            end,
            c,
            m,
            spread_weight.as_ptr(),
        )
    }
}

/// `haar1`.
pub fn haar1(x: &mut [f32], n0: i32, stride: i32) {
    assert!(x.len() >= (n0 * stride) as usize);
    // SAFETY: length checked.
    unsafe { oracle_haar1(x.as_mut_ptr(), n0, stride) }
}

/// `compute_qn` (static).
pub fn compute_qn(n: i32, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    // SAFETY: plain call.
    unsafe { oracle_compute_qn(n, b, offset, pulse_cap, i32::from(stereo)) }
}

/// `compute_channel_weights` (static).
pub fn compute_channel_weights(ex: f32, ey: f32) -> [f32; 2] {
    let mut w = [0.0f32; 2];
    // SAFETY: w has 2 elements.
    unsafe { oracle_compute_channel_weights(ex, ey, w.as_mut_ptr()) };
    w
}

/// `intensity_stereo` (static).
pub fn intensity_stereo(fs: i32, x: &mut [f32], y: &[f32], band_e: &[f32], band_id: i32, n: i32) {
    assert!(x.len() >= n as usize && y.len() >= n as usize && band_e.len() >= 42);
    // SAFETY: lengths checked.
    unsafe { oracle_intensity_stereo(fs, x.as_mut_ptr(), y.as_ptr(), band_e.as_ptr(), band_id, n) }
}

/// `stereo_split` (static).
pub fn stereo_split(x: &mut [f32], y: &mut [f32], n: i32) {
    assert!(x.len() >= n as usize && y.len() >= n as usize);
    // SAFETY: lengths checked.
    unsafe { oracle_stereo_split(x.as_mut_ptr(), y.as_mut_ptr(), n) }
}

/// `stereo_merge` (static).
pub fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32, n: i32) {
    assert!(x.len() >= n as usize && y.len() >= n as usize);
    // SAFETY: lengths checked.
    unsafe { oracle_stereo_merge(x.as_mut_ptr(), y.as_mut_ptr(), mid, n) }
}

/// `deinterleave_hadamard` (static).
pub fn deinterleave_hadamard(x: &mut [f32], n0: i32, stride: i32, hadamard: bool) {
    assert!(x.len() >= (n0 * stride) as usize);
    // SAFETY: length checked.
    unsafe { oracle_deinterleave_hadamard(x.as_mut_ptr(), n0, stride, i32::from(hadamard)) }
}

/// `interleave_hadamard` (static).
pub fn interleave_hadamard(x: &mut [f32], n0: i32, stride: i32, hadamard: bool) {
    assert!(x.len() >= (n0 * stride) as usize);
    // SAFETY: length checked.
    unsafe { oracle_interleave_hadamard(x.as_mut_ptr(), n0, stride, i32::from(hadamard)) }
}

/// Parameters of [`quant_all_bands`] (field names follow the C arguments).
#[derive(Debug, Clone, Default)]
pub struct QabParams {
    pub encode: bool,
    pub fs: i32,
    pub qext_mode: bool,
    pub start: i32,
    pub end: i32,
    pub c: i32,
    pub lm: i32,
    pub short_blocks: bool,
    pub spread: i32,
    pub dual_stereo: bool,
    pub intensity: i32,
    pub total_bits: i32,
    pub balance: i32,
    pub coded_bands: i32,
    pub complexity: i32,
    pub disable_inv: bool,
    /// Run `clt_compute_allocation` first (like the codec), replacing pulses, balance,
    /// codedBands, intensity and dual_stereo.
    pub do_alloc: bool,
    pub alloc_trim: i32,
    pub prev: i32,
    pub signal_bandwidth: i32,
    pub ext_total_bits: i32,
    /// Pass `cap` to `quant_all_bands` (QEXT).
    pub has_cap: bool,
    /// When > 1, `ec_enc_uint(prefix_val, prefix_ft)` is coded first.
    pub prefix_ft: i32,
    pub prefix_val: i32,
}

impl QabParams {
    /// The C `ip` array.
    pub fn to_ip(&self) -> [c_int; 26] {
        [
            i32::from(self.encode),
            self.fs,
            i32::from(self.qext_mode),
            self.start,
            self.end,
            self.c,
            self.lm,
            i32::from(self.short_blocks),
            self.spread,
            i32::from(self.dual_stereo),
            self.intensity,
            self.total_bits,
            self.balance,
            self.coded_bands,
            self.complexity,
            i32::from(self.disable_inv),
            0,
            i32::from(self.do_alloc),
            self.alloc_trim,
            self.prev,
            self.signal_bandwidth,
            0,
            self.ext_total_bits,
            i32::from(self.has_cap),
            self.prefix_ft,
            self.prefix_val,
        ]
    }
}

/// `quant_all_bands` (optionally after `clt_compute_allocation`). Returns
/// `[codedBands, balance, intensity, dual_stereo, tell_frac, rng, error, ext tell_frac, ext
/// rng, ext error, tell_frac before quant_all_bands]`.
pub fn quant_all_bands(
    p: &QabParams,
    x: &mut [f32],
    collapse_masks: &mut [u8],
    band_e: &[f32],
    pulses: &mut [i32],
    tf_res: &[i32],
    offsets: &[i32],
    cap: &[i32],
    extra_pulses: &[i32],
    buf: &mut [u8],
    ext_buf: &mut [u8],
    seed: &mut u32,
) -> [i32; 11] {
    let mut ip = p.to_ip();
    ip[16] = buf.len() as c_int;
    ip[21] = ext_buf.len() as c_int;
    let nb = 21;
    assert!(pulses.len() >= nb && tf_res.len() >= nb && offsets.len() >= nb && cap.len() >= nb);
    assert!(extra_pulses.len() >= nb && collapse_masks.len() >= 2 * nb && band_e.len() >= 2 * nb);
    let mut out = [0 as c_int; 11];
    // SAFETY: tables have nbEBands entries, x holds C*N samples (test responsibility, checked by
    // the caller's layout), buffers are passed with their sizes.
    unsafe {
        oracle_quant_all_bands(
            ip.as_ptr(),
            x.as_mut_ptr(),
            collapse_masks.as_mut_ptr(),
            band_e.as_ptr(),
            pulses.as_mut_ptr(),
            tf_res.as_ptr(),
            offsets.as_ptr(),
            cap.as_ptr(),
            extra_pulses.as_ptr(),
            buf.as_mut_ptr(),
            ext_buf.as_mut_ptr(),
            seed,
            out.as_mut_ptr(),
        );
    }
    out
}

// ---------------------------------------------------------------------------------------------
// quant_bands.c
// ---------------------------------------------------------------------------------------------

/// `eMeans`.
pub fn e_means() -> [f32; 25] {
    let mut out = [0.0f32; 25];
    // SAFETY: out has 25 elements.
    unsafe { oracle_emeans(out.as_mut_ptr()) };
    out
}

/// `amp2Log2`.
pub fn amp2log2(
    fs: i32,
    qext: bool,
    eff_end: i32,
    end: i32,
    band_e: &mut [f32],
    band_log_e: &mut [f32],
    c: i32,
) {
    assert!(band_e.len() >= 21 * c as usize && band_log_e.len() >= 21 * c as usize);
    // SAFETY: lengths checked.
    unsafe {
        oracle_amp2log2(
            fs,
            i32::from(qext),
            eff_end,
            end,
            band_e.as_mut_ptr(),
            band_log_e.as_mut_ptr(),
            c,
        );
    }
}

/// Parameters of [`quant_energy`].
#[derive(Debug, Clone, Default)]
pub struct EnergyParams {
    pub fs: i32,
    pub qext: bool,
    pub start: i32,
    pub end: i32,
    pub eff_end: i32,
    pub c: i32,
    pub lm: i32,
    pub budget: u32,
    pub nb_available_bytes: i32,
    pub force_intra: bool,
    pub two_pass: bool,
    pub loss_rate: i32,
    pub lfe: bool,
    pub prefix_ft: i32,
    pub prefix_val: i32,
    /// Pass `NULL` as `oldEBands` to the finalise stage.
    pub null_old: bool,
}

/// `quant_coarse_energy` + `quant_fine_energy` + `quant_energy_finalise` on a fresh encoder
/// (finished with `ec_enc_done`). Returns `[tell after coarse, after fine, after finalise,
/// rng, error]`.
pub fn quant_energy(
    p: &EnergyParams,
    e_bands: &[f32],
    old_e_bands: &mut [f32],
    error: &mut [f32],
    buf: &mut [u8],
    delayed_intra: &mut f32,
    fine_quant: &mut [i32],
    prev_quant: Option<&mut [i32]>,
    fine_priority: &mut [i32],
) -> [i32; 5] {
    assert!(e_bands.len() >= 42 && old_e_bands.len() >= 42 && error.len() >= 42);
    assert!(fine_quant.len() >= 21 && fine_priority.len() >= 21);
    let ip: [c_int; 17] = [
        p.fs,
        i32::from(p.qext),
        p.start,
        p.end,
        p.eff_end,
        p.c,
        p.lm,
        buf.len() as c_int,
        p.budget as c_int,
        p.nb_available_bytes,
        i32::from(p.force_intra),
        i32::from(p.two_pass),
        p.loss_rate,
        i32::from(p.lfe),
        p.prefix_ft,
        p.prefix_val,
        i32::from(p.null_old),
    ];
    let prev_ptr = match prev_quant {
        Some(p) => {
            assert!(p.len() >= 21);
            p.as_mut_ptr()
        }
        None => core::ptr::null_mut(),
    };
    let mut out = [0 as c_int; 5];
    // SAFETY: lengths checked; prev_ptr is NULL or 21 entries.
    unsafe {
        oracle_quant_energy(
            ip.as_ptr(),
            e_bands.as_ptr(),
            old_e_bands.as_mut_ptr(),
            error.as_mut_ptr(),
            buf.as_mut_ptr(),
            delayed_intra,
            fine_quant.as_mut_ptr(),
            prev_ptr,
            fine_priority.as_mut_ptr(),
            out.as_mut_ptr(),
        );
    }
    out
}

/// `unquant_coarse_energy` + `unquant_fine_energy` + `unquant_energy_finalise`. `intra < 0`
/// reads the intra flag like the decoder. Returns `[intra, tell after coarse, after fine,
/// after finalise, rng, error]`.
pub fn unquant_energy(
    fs: i32,
    qext: bool,
    start: i32,
    end: i32,
    c: i32,
    lm: i32,
    intra: i32,
    prefix_ft: i32,
    null_old: bool,
    old_e_bands: &mut [f32],
    buf: &mut [u8],
    fine_quant: &mut [i32],
    prev_quant: Option<&mut [i32]>,
    fine_priority: &mut [i32],
) -> [i32; 6] {
    assert!(old_e_bands.len() >= 42 && fine_quant.len() >= 21 && fine_priority.len() >= 21);
    let ip: [c_int; 10] = [
        fs,
        i32::from(qext),
        start,
        end,
        c,
        lm,
        buf.len() as c_int,
        intra,
        prefix_ft,
        i32::from(null_old),
    ];
    let prev_ptr = match prev_quant {
        Some(p) => {
            assert!(p.len() >= 21);
            p.as_mut_ptr()
        }
        None => core::ptr::null_mut(),
    };
    let mut out = [0 as c_int; 6];
    // SAFETY: lengths checked; prev_ptr is NULL or 21 entries.
    unsafe {
        oracle_unquant_energy(
            ip.as_ptr(),
            old_e_bands.as_mut_ptr(),
            buf.as_mut_ptr(),
            fine_quant.as_mut_ptr(),
            prev_ptr,
            fine_priority.as_mut_ptr(),
            out.as_mut_ptr(),
        );
    }
    out
}

/// `loss_distortion` (static).
pub fn loss_distortion(
    e_bands: &[f32],
    old_e_bands: &mut [f32],
    start: i32,
    end: i32,
    len: i32,
    c: i32,
) -> f32 {
    assert!(e_bands.len() >= (len * c) as usize && old_e_bands.len() >= (len * c) as usize);
    // SAFETY: lengths checked.
    unsafe {
        oracle_loss_distortion(
            e_bands.as_ptr(),
            old_e_bands.as_mut_ptr(),
            start,
            end,
            len,
            c,
        )
    }
}
