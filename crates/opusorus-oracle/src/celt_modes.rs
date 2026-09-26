//! Oracle bindings for unit `celt_modes`: CELT modes (`modes.c`), Laplace coding
//! (`laplace.c`), PVQ codeword coding (`cwrs.c`) and bit allocation (`rate.c`).

use core::ffi::{c_int, c_void};

unsafe extern "C" {
    fn oracle_mode_create(fs: c_int, frame_size: c_int, err: *mut c_int) -> *const c_void;
    fn oracle_mode_destroy(m: *const c_void);
    fn oracle_mode_ints(m: *const c_void, out: *mut c_int);
    fn oracle_mode_preemph(m: *const c_void, out: *mut f32);
    fn oracle_mode_i16(m: *const c_void, which: c_int, out: *mut i16, n: c_int);
    fn oracle_mode_u8(m: *const c_void, which: c_int, out: *mut u8, n: c_int);
    fn oracle_mode_f32(m: *const c_void, which: c_int, out: *mut f32, n: c_int);
    fn oracle_mode_fft(
        m: *const c_void,
        idx: c_int,
        ints: *mut c_int,
        scale: *mut f32,
        factors: *mut i16,
        bitrev: *mut i16,
        twiddles: *mut f32,
        tw_n: c_int,
    );
    fn oracle_laplace_encode(
        ops: *const c_int,
        nops: c_int,
        buf: *mut u8,
        size: c_int,
        vals_out: *mut c_int,
        tells: *mut u32,
        rng_out: *mut u32,
    ) -> c_int;
    fn oracle_laplace_decode(
        ops: *const c_int,
        nops: c_int,
        buf: *const u8,
        size: c_int,
        vals_out: *mut c_int,
        tells: *mut u32,
        rng_out: *mut u32,
    ) -> c_int;
    fn oracle_get_pulses(i: c_int) -> c_int;
    fn oracle_bits2pulses(
        m: *const c_void,
        use_qext: c_int,
        band: c_int,
        lm: c_int,
        bits: c_int,
    ) -> c_int;
    fn oracle_pulses2bits(
        m: *const c_void,
        use_qext: c_int,
        band: c_int,
        lm: c_int,
        pulses: c_int,
    ) -> c_int;
    fn oracle_compute_qext_mode(
        m: *const c_void,
        ints: *mut c_int,
        ebands: *mut i16,
        logn: *mut i16,
    );
    fn oracle_clt_compute_allocation(
        m: *const c_void,
        use_qext: c_int,
        start: c_int,
        end: c_int,
        offsets: *const c_int,
        cap: *const c_int,
        alloc_trim: c_int,
        total: c_int,
        pulses: *mut c_int,
        ebits: *mut c_int,
        fine_priority: *mut c_int,
        c: c_int,
        lm: c_int,
        buf: *mut u8,
        size: c_int,
        encode: c_int,
        prev: c_int,
        signal_bandwidth: c_int,
        io: *mut c_int,
    );
    fn oracle_clt_compute_extra_allocation(
        m: *const c_void,
        with_qext: c_int,
        start: c_int,
        end: c_int,
        qext_end: c_int,
        band_log_e: *const f32,
        qext_band_log_e: *const f32,
        total: c_int,
        extra_pulses: *mut c_int,
        extra_equant: *mut c_int,
        c: c_int,
        lm: c_int,
        buf: *mut u8,
        size: c_int,
        encode: c_int,
        tone_freq: f32,
        toneishness: f32,
        io: *mut c_int,
    );
    fn oracle_log2_frac(val: u32, frac: c_int) -> c_int;
    fn oracle_get_required_bits(bits: *mut i16, n: c_int, maxk: c_int, frac: c_int);
    fn oracle_cwrs_encode(
        ys: *const c_int,
        ns: *const c_int,
        ks: *const c_int,
        count: c_int,
        buf: *mut u8,
        size: c_int,
        rng_out: *mut u32,
    ) -> c_int;
    fn oracle_cwrs_decode(
        ns: *const c_int,
        ks: *const c_int,
        count: c_int,
        buf: *const u8,
        size: c_int,
        ys_out: *mut c_int,
        yy_out: *mut f32,
        rng_out: *mut u32,
    ) -> c_int;
    fn oracle_pvq_u(n: c_int, k: c_int) -> u32;
    fn oracle_pvq_v(n: c_int, k: c_int) -> u32;
}

// ---------------------------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------------------------

/// A `CELTMode` returned by the oracle's `opus_custom_mode_create` (destroyed on drop; static
/// modes are left alone by `opus_custom_mode_destroy`).
#[derive(Debug)]
pub struct Mode(*const c_void);

impl Drop for Mode {
    fn drop(&mut self) {
        // SAFETY: the pointer came from opus_custom_mode_create and is destroyed once.
        unsafe { oracle_mode_destroy(self.0) }
    }
}

/// `kiss_fft_state` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct FftState {
    pub nfft: i32,
    pub scale: f32,
    pub shift: i32,
    pub factors: Vec<i16>,
    pub bitrev: Vec<i16>,
    /// `(r, i)` pairs; `kfft[0].nfft` entries (sub-states share the base twiddles).
    pub twiddles: Vec<(f32, f32)>,
}

/// `PulseCache` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct PulseCache {
    pub size: i32,
    pub index: Vec<i16>,
    pub bits: Vec<u8>,
    pub caps: Vec<u8>,
}

/// All fields of a `CELTMode`, copied out of the oracle.
#[derive(Debug, Clone, PartialEq)]
pub struct ModeData {
    pub fs: i32,
    pub overlap: i32,
    pub nb_ebands: i32,
    pub eff_ebands: i32,
    pub preemph: [f32; 4],
    /// `nbEBands + 1` entries.
    pub e_bands: Vec<i16>,
    pub max_lm: i32,
    pub nb_short_mdcts: i32,
    pub short_mdct_size: i32,
    pub nb_alloc_vectors: i32,
    pub alloc_vectors: Vec<u8>,
    pub log_n: Vec<i16>,
    pub window: Vec<f32>,
    pub mdct_n: i32,
    pub mdct_maxshift: i32,
    pub kfft: Vec<FftState>,
    pub trig: Vec<f32>,
    pub cache: PulseCache,
    /// `None` without QEXT or when the mode has no QEXT cache.
    pub qext_cache: Option<PulseCache>,
}

impl Mode {
    /// `opus_custom_mode_create(fs, frame_size, &err)`; `Err(err)` on failure.
    pub fn create(fs: i32, frame_size: i32) -> Result<Self, i32> {
        let mut err: c_int = 0;
        // SAFETY: plain call; err is a valid out pointer.
        let p = unsafe { oracle_mode_create(fs, frame_size, &mut err) };
        if p.is_null() { Err(err) } else { Ok(Self(p)) }
    }

    fn i16s(&self, which: c_int, n: usize) -> Vec<i16> {
        let mut v = vec![0i16; n];
        // SAFETY: v has n entries; the shim copies n entries from the mode table.
        unsafe { oracle_mode_i16(self.0, which, v.as_mut_ptr(), n as c_int) };
        v
    }

    fn u8s(&self, which: c_int, n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n];
        // SAFETY: as above.
        unsafe { oracle_mode_u8(self.0, which, v.as_mut_ptr(), n as c_int) };
        v
    }

    fn f32s(&self, which: c_int, n: usize) -> Vec<f32> {
        let mut v = vec![0f32; n];
        // SAFETY: as above.
        unsafe { oracle_mode_f32(self.0, which, v.as_mut_ptr(), n as c_int) };
        v
    }

    /// Copies every field of the mode. `with_mdct` = false skips the FFT/MDCT data.
    #[must_use]
    pub fn data(&self, with_mdct: bool) -> ModeData {
        let mut ints = [0 as c_int; 12];
        // SAFETY: ints has the 12 entries the shim writes.
        unsafe { oracle_mode_ints(self.0, ints.as_mut_ptr()) };
        let mut preemph = [0f32; 4];
        // SAFETY: 4 entries.
        unsafe { oracle_mode_preemph(self.0, preemph.as_mut_ptr()) };
        let nb = ints[2] as usize;
        let max_lm = ints[4] as usize;
        let mdct_n = ints[8];
        let maxshift = ints[9];
        let mut kfft = Vec::new();
        let mut trig = Vec::new();
        if with_mdct {
            let mut tw_n = 0usize;
            for idx in 0..=maxshift {
                let mut fi = [0 as c_int; 2];
                let mut scale = 0f32;
                let mut factors = vec![0i16; 16];
                // SAFETY: first call only reads the header (null bitrev/twiddles).
                unsafe {
                    oracle_mode_fft(
                        self.0,
                        idx,
                        fi.as_mut_ptr(),
                        &mut scale,
                        factors.as_mut_ptr(),
                        core::ptr::null_mut(),
                        core::ptr::null_mut(),
                        0,
                    );
                }
                let nfft = fi[0] as usize;
                if idx == 0 {
                    tw_n = nfft;
                }
                let mut bitrev = vec![0i16; nfft];
                let mut tw = vec![0f32; 2 * tw_n];
                // SAFETY: bitrev has nfft entries, tw has 2*tw_n floats.
                unsafe {
                    oracle_mode_fft(
                        self.0,
                        idx,
                        fi.as_mut_ptr(),
                        &mut scale,
                        factors.as_mut_ptr(),
                        bitrev.as_mut_ptr(),
                        tw.as_mut_ptr(),
                        tw_n as c_int,
                    );
                }
                kfft.push(FftState {
                    nfft: fi[0],
                    scale,
                    shift: fi[1],
                    factors: factors.clone(),
                    bitrev,
                    twiddles: tw.chunks(2).map(|c| (c[0], c[1])).collect(),
                });
            }
            let n = mdct_n as usize;
            let trig_len = n - ((n >> 1) >> maxshift);
            trig = self.f32s(1, trig_len);
        }
        let cache = PulseCache {
            size: ints[10],
            index: self.i16s(2, nb * (max_lm + 2)),
            bits: self.u8s(1, ints[10] as usize),
            caps: self.u8s(2, (max_lm + 1) * 2 * nb),
        };
        let qext_cache = (ints[11] >= 0).then(|| {
            let qnb = 14usize;
            PulseCache {
                size: ints[11],
                index: self.i16s(3, qnb * (max_lm + 2)),
                bits: self.u8s(3, ints[11] as usize),
                caps: self.u8s(4, (max_lm + 1) * 2 * qnb),
            }
        });
        ModeData {
            fs: ints[0],
            overlap: ints[1],
            nb_ebands: ints[2],
            eff_ebands: ints[3],
            preemph,
            e_bands: self.i16s(0, nb + 1),
            max_lm: ints[4],
            nb_short_mdcts: ints[5],
            short_mdct_size: ints[6],
            nb_alloc_vectors: ints[7],
            alloc_vectors: self.u8s(0, ints[7] as usize * nb),
            log_n: self.i16s(1, nb),
            window: self.f32s(0, ints[1] as usize),
            mdct_n,
            mdct_maxshift: maxshift,
            kfft,
            trig,
            cache,
            qext_cache,
        }
    }

    /// `bits2pulses(m or qext mode, band, LM, bits)`.
    #[must_use]
    pub fn bits2pulses(&self, use_qext: bool, band: i32, lm: i32, bits: i32) -> i32 {
        // SAFETY: plain call on a valid mode.
        unsafe { oracle_bits2pulses(self.0, c_int::from(use_qext), band, lm, bits) }
    }

    /// `pulses2bits(m or qext mode, band, LM, pulses)`.
    #[must_use]
    pub fn pulses2bits(&self, use_qext: bool, band: i32, lm: i32, pulses: i32) -> i32 {
        // SAFETY: plain call on a valid mode.
        unsafe { oracle_pulses2bits(self.0, c_int::from(use_qext), band, lm, pulses) }
    }

    /// `compute_qext_mode` (QEXT builds): `([nbEBands, effEBands, nbAllocVectors, cache.size],
    /// eBands, logN)`.
    #[must_use]
    pub fn compute_qext_mode(&self) -> ([i32; 4], Vec<i16>, Vec<i16>) {
        let mut ints = [0 as c_int; 4];
        let mut eb = vec![0i16; 15];
        let mut ln = vec![0i16; 14];
        // SAFETY: buffers sized for NB_QEXT_BANDS=14 bands.
        unsafe {
            oracle_compute_qext_mode(self.0, ints.as_mut_ptr(), eb.as_mut_ptr(), ln.as_mut_ptr());
        }
        (ints, eb, ln)
    }

    /// Runs `clt_compute_allocation` (see [`AllocIn`]); `buf` is the encoder output buffer
    /// (encode) or the input (decode).
    #[must_use]
    pub fn clt_compute_allocation(&self, a: &AllocIn, buf: &mut [u8], encode: bool) -> AllocOut {
        let nb = a.offsets.len();
        let mut pulses = vec![0 as c_int; nb];
        let mut ebits = vec![0 as c_int; nb];
        let mut fine_priority = vec![0 as c_int; nb];
        let mut io = [a.intensity, a.dual_stereo, 0, 0, 0, 0, 0];
        assert_eq!(a.cap.len(), nb);
        // SAFETY: arrays hold nbEBands entries of the mode used; buf is `size` bytes.
        unsafe {
            oracle_clt_compute_allocation(
                self.0,
                c_int::from(a.use_qext),
                a.start,
                a.end,
                a.offsets.as_ptr(),
                a.cap.as_ptr(),
                a.alloc_trim,
                a.total,
                pulses.as_mut_ptr(),
                ebits.as_mut_ptr(),
                fine_priority.as_mut_ptr(),
                a.c,
                a.lm,
                buf.as_mut_ptr(),
                buf.len() as c_int,
                c_int::from(encode),
                a.prev,
                a.signal_bandwidth,
                io.as_mut_ptr(),
            );
        }
        AllocOut {
            intensity: io[0],
            dual_stereo: io[1],
            balance: io[2],
            coded_bands: io[3],
            tell_frac: io[4] as u32,
            rng: io[5] as u32,
            error: io[6],
            pulses,
            ebits,
            fine_priority,
        }
    }

    /// Runs `clt_compute_extra_allocation` (QEXT builds).
    #[must_use]
    pub fn clt_compute_extra_allocation(
        &self,
        a: &ExtraAllocIn,
        buf: &mut [u8],
        encode: bool,
    ) -> ExtraAllocOut {
        let n = a.n_out;
        let mut extra_pulses = vec![0 as c_int; n];
        let mut extra_equant = vec![0 as c_int; n];
        let mut io = [0 as c_int; 3];
        // SAFETY: arrays sized by the caller for nbEBands + NB_QEXT_BANDS entries.
        unsafe {
            oracle_clt_compute_extra_allocation(
                self.0,
                c_int::from(a.with_qext),
                a.start,
                a.end,
                a.qext_end,
                a.band_log_e.as_ptr(),
                a.qext_band_log_e.as_ptr(),
                a.total,
                extra_pulses.as_mut_ptr(),
                extra_equant.as_mut_ptr(),
                a.c,
                a.lm,
                buf.as_mut_ptr(),
                buf.len() as c_int,
                c_int::from(encode),
                a.tone_freq,
                a.toneishness,
                io.as_mut_ptr(),
            );
        }
        ExtraAllocOut {
            extra_pulses,
            extra_equant,
            tell_frac: io[0] as u32,
            rng: io[1] as u32,
            error: io[2],
        }
    }
}

/// Inputs of `clt_compute_allocation`.
#[derive(Debug, Clone)]
pub struct AllocIn {
    /// Use `compute_qext_mode(mode)` instead of the mode itself.
    pub use_qext: bool,
    pub start: i32,
    pub end: i32,
    /// `nbEBands` entries.
    pub offsets: Vec<i32>,
    /// `nbEBands` entries.
    pub cap: Vec<i32>,
    pub alloc_trim: i32,
    pub intensity: i32,
    pub dual_stereo: i32,
    pub total: i32,
    pub c: i32,
    pub lm: i32,
    pub prev: i32,
    pub signal_bandwidth: i32,
}

/// Outputs of `clt_compute_allocation`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocOut {
    pub intensity: i32,
    pub dual_stereo: i32,
    pub balance: i32,
    pub coded_bands: i32,
    pub tell_frac: u32,
    pub rng: u32,
    pub error: i32,
    pub pulses: Vec<i32>,
    pub ebits: Vec<i32>,
    pub fine_priority: Vec<i32>,
}

/// Inputs of `clt_compute_extra_allocation`.
#[derive(Debug, Clone)]
pub struct ExtraAllocIn {
    pub with_qext: bool,
    pub start: i32,
    pub end: i32,
    pub qext_end: i32,
    /// `C * nbEBands` entries.
    pub band_log_e: Vec<f32>,
    /// `C * NB_QEXT_BANDS` entries.
    pub qext_band_log_e: Vec<f32>,
    pub total: i32,
    pub c: i32,
    pub lm: i32,
    pub tone_freq: f32,
    pub toneishness: f32,
    /// Length of the output arrays (`nbEBands + NB_QEXT_BANDS`).
    pub n_out: usize,
}

/// Outputs of `clt_compute_extra_allocation`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtraAllocOut {
    pub extra_pulses: Vec<i32>,
    pub extra_equant: Vec<i32>,
    pub tell_frac: u32,
    pub rng: u32,
    pub error: i32,
}

// ---------------------------------------------------------------------------------------------
// Laplace
// ---------------------------------------------------------------------------------------------

/// Result of a Laplace encode sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaplaceEnc {
    pub buf: Vec<u8>,
    /// Values actually coded (`ec_laplace_encode` may clamp).
    pub values: Vec<i32>,
    /// `ec_tell_frac` after each op.
    pub tells: Vec<u32>,
    pub rng: u32,
    pub error: i32,
}

/// Encodes ops `[kind (0 laplace, 1 p0), value, fs|p0, decay]`.
#[must_use]
pub fn laplace_encode(ops: &[[i32; 4]], size: usize) -> LaplaceEnc {
    let mut buf = vec![0u8; size];
    let mut values = vec![0 as c_int; ops.len()];
    let mut tells = vec![0u32; ops.len()];
    let mut rng = 0u32;
    // SAFETY: all arrays are sized as the shim expects.
    let error = unsafe {
        oracle_laplace_encode(
            ops.as_ptr().cast(),
            ops.len() as c_int,
            buf.as_mut_ptr(),
            size as c_int,
            values.as_mut_ptr(),
            tells.as_mut_ptr(),
            &mut rng,
        )
    };
    LaplaceEnc {
        buf,
        values,
        tells,
        rng,
        error,
    }
}

/// Result of a Laplace decode sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaplaceDec {
    pub values: Vec<i32>,
    pub tells: Vec<u32>,
    pub rng: u32,
    pub error: i32,
}

/// Decodes ops `[kind, fs|p0, decay]` from `buf`.
#[must_use]
pub fn laplace_decode(ops: &[[i32; 3]], buf: &[u8]) -> LaplaceDec {
    let mut values = vec![0 as c_int; ops.len()];
    let mut tells = vec![0u32; ops.len()];
    let mut rng = 0u32;
    // SAFETY: all arrays are sized as the shim expects; the decoder only reads buf.
    let error = unsafe {
        oracle_laplace_decode(
            ops.as_ptr().cast(),
            ops.len() as c_int,
            buf.as_ptr(),
            buf.len() as c_int,
            values.as_mut_ptr(),
            tells.as_mut_ptr(),
            &mut rng,
        )
    };
    LaplaceDec {
        values,
        tells,
        rng,
        error,
    }
}

// ---------------------------------------------------------------------------------------------
// cwrs / rate helpers
// ---------------------------------------------------------------------------------------------

/// `get_pulses(i)`.
#[must_use]
pub fn get_pulses(i: i32) -> i32 {
    // SAFETY: pure function.
    unsafe { oracle_get_pulses(i) }
}

/// `log2_frac(val, frac)` (custom-modes builds only; 0 otherwise).
#[must_use]
pub fn log2_frac(val: u32, frac: i32) -> i32 {
    // SAFETY: pure function.
    unsafe { oracle_log2_frac(val, frac) }
}

/// `get_required_bits(bits, n, maxk, frac)` (custom-modes builds only).
#[must_use]
pub fn get_required_bits(n: i32, maxk: i32, frac: i32) -> Vec<i16> {
    let mut bits = vec![0i16; maxk as usize + 1];
    // SAFETY: bits has maxk+1 entries.
    unsafe { oracle_get_required_bits(bits.as_mut_ptr(), n, maxk, frac) };
    bits
}

/// `CELT_PVQ_U(n, k)`.
#[must_use]
pub fn pvq_u(n: i32, k: i32) -> u32 {
    // SAFETY: pure table lookup (caller keeps (n,k) within the table).
    unsafe { oracle_pvq_u(n, k) }
}

/// `CELT_PVQ_V(n, k)`.
#[must_use]
pub fn pvq_v(n: i32, k: i32) -> u32 {
    // SAFETY: pure table lookup (caller keeps (n,k) within the table).
    unsafe { oracle_pvq_v(n, k) }
}

/// Encodes consecutive pulse vectors (`ys` holds `ns[v]` entries for each vector `v`).
/// Returns `(buffer, rng, error)`.
#[must_use]
pub fn cwrs_encode(ys: &[i32], ns: &[i32], ks: &[i32], size: usize) -> (Vec<u8>, u32, i32) {
    assert_eq!(ns.len(), ks.len());
    assert_eq!(ys.len(), ns.iter().map(|&n| n as usize).sum::<usize>());
    let mut buf = vec![0u8; size];
    let mut rng = 0u32;
    // SAFETY: arrays sized as checked above.
    let err = unsafe {
        oracle_cwrs_encode(
            ys.as_ptr(),
            ns.as_ptr(),
            ks.as_ptr(),
            ns.len() as c_int,
            buf.as_mut_ptr(),
            size as c_int,
            &mut rng,
        )
    };
    (buf, rng, err)
}

/// Decodes consecutive pulse vectors. Returns `(ys, yy per vector, rng, error)`.
#[must_use]
pub fn cwrs_decode(ns: &[i32], ks: &[i32], buf: &[u8]) -> (Vec<i32>, Vec<f32>, u32, i32) {
    assert_eq!(ns.len(), ks.len());
    let total: usize = ns.iter().map(|&n| n as usize).sum();
    let mut ys = vec![0 as c_int; total];
    let mut yy = vec![0f32; ns.len()];
    let mut rng = 0u32;
    // SAFETY: output arrays sized for all vectors; decoder only reads buf.
    let err = unsafe {
        oracle_cwrs_decode(
            ns.as_ptr(),
            ks.as_ptr(),
            ns.len() as c_int,
            buf.as_ptr(),
            buf.len() as c_int,
            ys.as_mut_ptr(),
            yy.as_mut_ptr(),
            &mut rng,
        )
    };
    (ys, yy, rng, err)
}
