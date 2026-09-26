//! Oracle bindings for unit `celt_fft`: kiss FFT, MDCT and (QEXT) mini kiss FFT.
//!
//! Complex buffers are passed as interleaved `f32` (`re, im, re, im, ...`), matching the C
//! `kiss_fft_cpx` layout.

use core::ffi::c_int;
use core::ptr;

unsafe extern "C" {
    fn oracle_celt_fft_static_info(
        fs: c_int,
        idx: c_int,
        nfft: *mut c_int,
        scale: *mut f32,
        shift: *mut c_int,
        factors: *mut i16,
        bitrev: *mut i16,
    );
    fn oracle_celt_fft_static(fs: c_int, idx: c_int, kind: c_int, fin: *const f32, fout: *mut f32);
    fn oracle_celt_mdct_static(
        fs: c_int,
        dir: c_int,
        input: *const f32,
        out: *mut f32,
        window: *const f32,
        overlap: c_int,
        shift: c_int,
        stride: c_int,
    );
    #[cfg(feature = "custom-modes")]
    fn oracle_celt_fft_alloc(
        nfft: c_int,
        base_nfft: c_int,
        scale: *mut f32,
        shift: *mut c_int,
        factors: *mut i16,
        bitrev: *mut i16,
        twiddles: *mut f32,
        ntw: *mut c_int,
        kind: c_int,
        fin: *const f32,
        fout: *mut f32,
    ) -> c_int;
    #[cfg(feature = "custom-modes")]
    fn oracle_celt_mdct_init(
        n: c_int,
        maxshift: c_int,
        trig: *mut f32,
        nffts: *mut c_int,
        shifts: *mut c_int,
        dir: c_int,
        input: *const f32,
        out: *mut f32,
        window: *const f32,
        overlap: c_int,
        shift: c_int,
        stride: c_int,
    ) -> c_int;
    #[cfg(feature = "qext")]
    fn oracle_mini_kfft(
        nfft: c_int,
        inverse: c_int,
        fin: *const f32,
        fout: *mut f32,
        in_stride: c_int,
        factors: *mut c_int,
        twiddles: *mut f32,
    );
    #[cfg(feature = "qext")]
    fn oracle_mini_kfftr(
        nfft: c_int,
        timedata: *const f32,
        freqdata: *mut f32,
        super_twiddles: *mut f32,
    );
}

/// Which transform to run on an FFT state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftKind {
    /// `opus_fft_c`.
    Fft = 0,
    /// `opus_ifft_c`.
    Ifft = 1,
    /// `opus_fft_impl` in place on a copy of the input.
    Impl = 2,
}

/// Fields of a C `kiss_fft_state`.
#[derive(Debug, Clone, PartialEq)]
pub struct FftStateInfo {
    /// `nfft`.
    pub nfft: i32,
    /// `scale`.
    pub scale: f32,
    /// `shift`.
    pub shift: i32,
    /// `factors`.
    pub factors: [i16; 16],
    /// `bitrev` (`nfft` entries).
    pub bitrev: Vec<i16>,
    /// Twiddle table the state points to (interleaved; only filled by [`fft_alloc`]).
    pub twiddles: Vec<f32>,
}

/// Maximum static FFT size (`nfft` of `kfft[0]` of the 96 kHz QEXT mode).
const MAX_STATIC_NFFT: usize = 960;

/// Reads the static FFT state `idx` (0..4) of the static mode at `fs` (48000 or, with QEXT,
/// 96000).
#[must_use]
pub fn fft_static_info(fs: i32, idx: usize) -> FftStateInfo {
    assert!(idx < 4);
    let mut nfft = 0;
    let mut scale = 0.0f32;
    let mut shift = 0;
    let mut factors = [0i16; 16];
    let mut bitrev = vec![0i16; MAX_STATIC_NFFT];
    // SAFETY: all out-pointers are valid; bitrev holds the largest static nfft.
    unsafe {
        oracle_celt_fft_static_info(
            fs,
            idx as c_int,
            &mut nfft,
            &mut scale,
            &mut shift,
            factors.as_mut_ptr(),
            bitrev.as_mut_ptr(),
        );
    }
    bitrev.truncate(nfft as usize);
    FftStateInfo {
        nfft,
        scale,
        shift,
        factors,
        bitrev,
        twiddles: Vec::new(),
    }
}

/// Runs `kind` on static FFT state `idx` of the static mode at `fs`. `fin` is interleaved
/// complex with `2 * nfft` floats; returns the interleaved output.
#[must_use]
pub fn fft_static(fs: i32, idx: usize, kind: FftKind, fin: &[f32]) -> Vec<f32> {
    let info = fft_static_info(fs, idx);
    assert_eq!(fin.len(), 2 * info.nfft as usize);
    let mut fout = vec![0.0f32; fin.len()];
    // SAFETY: fin/fout hold 2*nfft floats (nfft complex values) as the C code expects.
    unsafe {
        oracle_celt_fft_static(
            fs,
            idx as c_int,
            kind as c_int,
            fin.as_ptr(),
            fout.as_mut_ptr(),
        );
    }
    fout
}

/// Checks MDCT buffer sizes for lookup size `n` (the C code does no bounds checking).
#[expect(clippy::too_many_arguments, reason = "flat MDCT parameter list")]
fn check_mdct_sizes(
    n: usize,
    dir: i32,
    input: &[f32],
    out: &[f32],
    window: Option<&[f32]>,
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let n = n >> shift;
    let n2 = n / 2;
    assert!(stride >= 1 && overlap.is_multiple_of(2) && overlap <= n2);
    if let Some(w) = window {
        assert!(w.len() >= overlap);
    }
    if dir == 0 {
        assert!(input.len() >= n2 + overlap);
        assert!(out.len() > stride * (n2 - 1));
    } else {
        assert!(input.len() > stride * (n2 - 1));
        assert!(out.len() >= n2 + overlap / 2 && out.len() >= overlap);
    }
}

/// Runs `clt_mdct_forward_c` (`dir == 0`) or `clt_mdct_backward_c` (`dir == 1`) with the static
/// mode at `fs`. `window == None` uses the mode window (whose length is the mode overlap).
/// `out` is updated in place (the backward transform reads its first `overlap/2` values).
#[expect(clippy::too_many_arguments, reason = "mirrors C signature")]
pub fn mdct_static(
    fs: i32,
    dir: i32,
    input: &[f32],
    out: &mut [f32],
    window: Option<&[f32]>,
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let n = if fs == 96000 { 3840 } else { 1920 };
    assert!(shift <= 3);
    if window.is_none() {
        assert_eq!(overlap, n / 16);
    }
    check_mdct_sizes(n, dir, input, out, window, overlap, shift, stride);
    // SAFETY: buffer sizes checked above against the C access pattern.
    unsafe {
        oracle_celt_mdct_static(
            fs,
            dir,
            input.as_ptr(),
            out.as_mut_ptr(),
            window.map_or(ptr::null(), <[f32]>::as_ptr),
            overlap as c_int,
            shift as c_int,
            stride as c_int,
        );
    }
}

/// `opus_fft_alloc_twiddles(nfft, base)` where `base = opus_fft_alloc(base_nfft)` if
/// `base_nfft > 0`. Returns `None` where C returns `NULL`; otherwise the state fields plus, if
/// `run` is given, the output of that transform on the given interleaved input.
#[cfg(feature = "custom-modes")]
#[must_use]
pub fn fft_alloc(
    nfft: i32,
    base_nfft: i32,
    run: Option<(FftKind, &[f32])>,
) -> Option<(FftStateInfo, Vec<f32>)> {
    assert!(nfft > 0);
    let cap = nfft.max(base_nfft) as usize;
    let mut scale = 0.0f32;
    let mut shift = 0;
    let mut factors = [0i16; 16];
    let mut bitrev = vec![0i16; nfft as usize];
    let mut twiddles = vec![0.0f32; 2 * cap];
    let mut ntw = 0;
    let (kind, fin) = match run {
        Some((k, fin)) => {
            assert_eq!(fin.len(), 2 * nfft as usize);
            (k as c_int, fin)
        }
        None => (-1, &[][..]),
    };
    let mut fout = vec![0.0f32; fin.len()];
    // SAFETY: bitrev holds nfft, twiddles 2*max(nfft, base_nfft); fin/fout hold 2*nfft floats
    // when kind >= 0 (otherwise unused).
    let ok = unsafe {
        oracle_celt_fft_alloc(
            nfft,
            base_nfft,
            &mut scale,
            &mut shift,
            factors.as_mut_ptr(),
            bitrev.as_mut_ptr(),
            twiddles.as_mut_ptr(),
            &mut ntw,
            kind,
            fin.as_ptr(),
            fout.as_mut_ptr(),
        )
    };
    if ok == 0 {
        return None;
    }
    twiddles.truncate(2 * ntw as usize);
    Some((
        FftStateInfo {
            nfft,
            scale,
            shift,
            factors,
            bitrev,
            twiddles,
        },
        fout,
    ))
}

/// Result of [`mdct_init`].
#[cfg(feature = "custom-modes")]
#[derive(Debug, Clone, PartialEq)]
pub struct MdctInitInfo {
    /// `trig` table.
    pub trig: Vec<f32>,
    /// `kfft[i]->nfft` for `i <= maxshift`.
    pub nffts: Vec<i32>,
    /// `kfft[i]->shift` for `i <= maxshift`.
    pub shifts: Vec<i32>,
}

/// An MDCT to run on a lookup built by [`mdct_init`].
#[cfg(feature = "custom-modes")]
#[derive(Debug)]
pub struct MdctRun<'a> {
    /// 0 forward, 1 backward.
    pub dir: i32,
    /// Input.
    pub input: &'a [f32],
    /// Output (updated in place).
    pub out: &'a mut [f32],
    /// Window (`overlap` values).
    pub window: &'a [f32],
    /// Overlap.
    pub overlap: usize,
    /// Shift.
    pub shift: usize,
    /// Stride.
    pub stride: usize,
}

/// `clt_mdct_init(n, maxshift)`, optionally running an MDCT on the resulting lookup. Returns
/// `None` where C returns 0.
#[cfg(feature = "custom-modes")]
#[must_use]
pub fn mdct_init(n: i32, maxshift: i32, run: Option<MdctRun<'_>>) -> Option<MdctInitInfo> {
    assert!(n > 0 && (0..=3).contains(&maxshift));
    let mut trig = vec![0.0f32; (n - ((n >> 1) >> maxshift)) as usize];
    let mut nffts = vec![0; maxshift as usize + 1];
    let mut shifts = vec![0; maxshift as usize + 1];
    // SAFETY: trig/nffts/shifts sized as the shim writes them; the optional run buffers are
    // checked against the C access pattern.
    let ok = unsafe {
        match run {
            Some(r) => {
                assert!(r.shift as i32 <= maxshift);
                check_mdct_sizes(
                    n as usize,
                    r.dir,
                    r.input,
                    r.out,
                    Some(r.window),
                    r.overlap,
                    r.shift,
                    r.stride,
                );
                oracle_celt_mdct_init(
                    n,
                    maxshift,
                    trig.as_mut_ptr(),
                    nffts.as_mut_ptr(),
                    shifts.as_mut_ptr(),
                    r.dir,
                    r.input.as_ptr(),
                    r.out.as_mut_ptr(),
                    r.window.as_ptr(),
                    r.overlap as c_int,
                    r.shift as c_int,
                    r.stride as c_int,
                )
            }
            None => oracle_celt_mdct_init(
                n,
                maxshift,
                trig.as_mut_ptr(),
                nffts.as_mut_ptr(),
                shifts.as_mut_ptr(),
                -1,
                ptr::null(),
                ptr::null_mut(),
                ptr::null(),
                0,
                0,
                1,
            ),
        }
    };
    (ok != 0).then_some(MdctInitInfo {
        trig,
        nffts,
        shifts,
    })
}

/// Result of [`mini_kfft`].
#[cfg(feature = "qext")]
#[derive(Debug, Clone, PartialEq)]
pub struct MiniKfftOut {
    /// Interleaved output (`nfft` complex values).
    pub fout: Vec<f32>,
    /// `factors` (64 ints).
    pub factors: Vec<i32>,
    /// Interleaved twiddles (`nfft` complex values).
    pub twiddles: Vec<f32>,
}

/// `mini_kiss_fft_alloc(nfft, inverse)` + `mini_kiss_fft_stride(fin, in_stride)`. `nfft` must
/// only have factors 2..=5 (C asserts otherwise).
#[cfg(feature = "qext")]
#[must_use]
pub fn mini_kfft(nfft: i32, inverse: bool, fin: &[f32], in_stride: usize) -> MiniKfftOut {
    assert!(nfft >= 2 && in_stride >= 1);
    assert!(fin.len() >= 2 * ((nfft as usize - 1) * in_stride + 1));
    let mut fout = vec![0.0f32; 2 * nfft as usize];
    let mut factors = vec![0; 64];
    let mut twiddles = vec![0.0f32; 2 * nfft as usize];
    // SAFETY: buffer sizes checked/allocated as the shim expects.
    unsafe {
        oracle_mini_kfft(
            nfft,
            c_int::from(inverse),
            fin.as_ptr(),
            fout.as_mut_ptr(),
            in_stride as c_int,
            factors.as_mut_ptr(),
            twiddles.as_mut_ptr(),
        );
    }
    MiniKfftOut {
        fout,
        factors,
        twiddles,
    }
}

/// `mini_kiss_fftr_alloc(nfft, 0)` + `mini_kiss_fftr(timedata)`. Returns
/// `(freqdata, super_twiddles)`, both interleaved.
#[cfg(feature = "qext")]
#[must_use]
pub fn mini_kfftr(nfft: i32, timedata: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert!(nfft >= 4 && (nfft as u32).is_multiple_of(2));
    assert_eq!(timedata.len(), nfft as usize);
    let mut freq = vec![0.0f32; 2 * (nfft as usize / 2 + 1)];
    let mut stw = vec![0.0f32; 2 * (nfft as usize / 4)];
    // SAFETY: buffer sizes allocated as the shim expects.
    unsafe {
        oracle_mini_kfftr(nfft, timedata.as_ptr(), freq.as_mut_ptr(), stw.as_mut_ptr());
    }
    (freq, stw)
}
