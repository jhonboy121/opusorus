//! Oracle bindings for unit `celt_encoder` (celt/celt_encoder.c, float build).
//!
//! * [`CeltEnc`]: a persistent library CELT encoder (CTLs, encode with/without an existing
//!   range coder, full state dumps as [`CeState`]).
//! * [`OpusRec`]: a private copy of the Opus encoder whose calls into the CELT encoder are
//!   recorded ([`CeltCall`]: complete CELT state, input, buffers and range coder before/after),
//!   so every CELT call made by the real Opus encoder can be replayed.
//! * Free functions: the static helpers of celt_encoder.c (from a renamed private copy).
//!
//! Wrappers assert the slice lengths the C code touches.

#![allow(
    clippy::too_many_arguments,
    reason = "wrappers mirror the flat C shim signatures"
)]

use core::ffi::{c_int, c_void};

/// `CE_MAX_OVERLAP` etc. of the shim (dump array capacities).
pub const CE_MAX_OVERLAP: usize = 1024;
pub const CE_MAX_PERIOD: usize = 2048;
pub const CE_MAX_BANDS: usize = 64;
pub const CE_QEXT_BANDS: usize = 14;

/// Flat dump of a C `CELTEncoder` (mirrors `OracleCeState` in the shim).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CeState {
    pub channels: c_int,
    pub stream_channels: c_int,
    pub force_intra: c_int,
    pub clip: c_int,
    pub disable_pf: c_int,
    pub complexity: c_int,
    pub upsample: c_int,
    pub start: c_int,
    pub end: c_int,
    pub bitrate: c_int,
    pub vbr: c_int,
    pub signalling: c_int,
    pub constrained_vbr: c_int,
    pub loss_rate: c_int,
    pub lsb_depth: c_int,
    pub lfe: c_int,
    pub disable_inv: c_int,
    pub enable_qext: c_int,
    pub qext_scale: c_int,
    pub mode_fs: c_int,
    pub mode_short_mdct_size: c_int,
    pub mode_nb_short_mdcts: c_int,
    pub mode_nb_ebands: c_int,
    pub mode_overlap: c_int,
    pub rng: u32,
    pub spread_decision: c_int,
    pub delayed_intra: f32,
    pub tonal_average: c_int,
    pub last_coded_bands: c_int,
    pub hf_average: c_int,
    pub tapset_decision: c_int,
    pub prefilter_period: c_int,
    pub prefilter_gain: f32,
    pub prefilter_tapset: c_int,
    pub consec_transient: c_int,
    pub an_valid: c_int,
    pub an_tonality: f32,
    pub an_tonality_slope: f32,
    pub an_noisiness: f32,
    pub an_activity: f32,
    pub an_music_prob: f32,
    pub an_music_prob_min: f32,
    pub an_music_prob_max: f32,
    pub an_bandwidth: c_int,
    pub an_activity_probability: f32,
    pub an_max_pitch_ratio: f32,
    pub an_leak_boost: [u8; 20],
    pub silk_signal_type: c_int,
    pub silk_offset: c_int,
    pub preemph_mem_e: [f32; 2],
    pub preemph_mem_d: [f32; 2],
    pub vbr_reservoir: c_int,
    pub vbr_drift: c_int,
    pub vbr_offset: c_int,
    pub vbr_count: c_int,
    pub overlap_max: f32,
    pub stereo_saving: f32,
    pub intensity: c_int,
    pub has_energy_mask: c_int,
    pub energy_mask: [f32; 2 * CE_MAX_BANDS],
    pub spec_avg: f32,
    pub in_mem: [f32; 2 * CE_MAX_OVERLAP],
    pub prefilter_mem: [f32; 2 * CE_MAX_PERIOD],
    pub old_band_e: [f32; 2 * CE_MAX_BANDS],
    pub old_log_e: [f32; 2 * CE_MAX_BANDS],
    pub old_log_e2: [f32; 2 * CE_MAX_BANDS],
    pub energy_error: [f32; 2 * CE_MAX_BANDS],
    pub qext_old_band_e: [f32; 2 * CE_QEXT_BANDS],
}

impl CeState {
    /// An all-zero state (filled by the shim).
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            channels: 0,
            stream_channels: 0,
            force_intra: 0,
            clip: 0,
            disable_pf: 0,
            complexity: 0,
            upsample: 0,
            start: 0,
            end: 0,
            bitrate: 0,
            vbr: 0,
            signalling: 0,
            constrained_vbr: 0,
            loss_rate: 0,
            lsb_depth: 0,
            lfe: 0,
            disable_inv: 0,
            enable_qext: 0,
            qext_scale: 0,
            mode_fs: 0,
            mode_short_mdct_size: 0,
            mode_nb_short_mdcts: 0,
            mode_nb_ebands: 0,
            mode_overlap: 0,
            rng: 0,
            spread_decision: 0,
            delayed_intra: 0.0,
            tonal_average: 0,
            last_coded_bands: 0,
            hf_average: 0,
            tapset_decision: 0,
            prefilter_period: 0,
            prefilter_gain: 0.0,
            prefilter_tapset: 0,
            consec_transient: 0,
            an_valid: 0,
            an_tonality: 0.0,
            an_tonality_slope: 0.0,
            an_noisiness: 0.0,
            an_activity: 0.0,
            an_music_prob: 0.0,
            an_music_prob_min: 0.0,
            an_music_prob_max: 0.0,
            an_bandwidth: 0,
            an_activity_probability: 0.0,
            an_max_pitch_ratio: 0.0,
            an_leak_boost: [0; 20],
            silk_signal_type: 0,
            silk_offset: 0,
            preemph_mem_e: [0.0; 2],
            preemph_mem_d: [0.0; 2],
            vbr_reservoir: 0,
            vbr_drift: 0,
            vbr_offset: 0,
            vbr_count: 0,
            overlap_max: 0.0,
            stereo_saving: 0.0,
            intensity: 0,
            has_energy_mask: 0,
            energy_mask: [0.0; 2 * CE_MAX_BANDS],
            spec_avg: 0.0,
            in_mem: [0.0; 2 * CE_MAX_OVERLAP],
            prefilter_mem: [0.0; 2 * CE_MAX_PERIOD],
            old_band_e: [0.0; 2 * CE_MAX_BANDS],
            old_log_e: [0.0; 2 * CE_MAX_BANDS],
            old_log_e2: [0.0; 2 * CE_MAX_BANDS],
            energy_error: [0.0; 2 * CE_MAX_BANDS],
            qext_old_band_e: [0.0; 2 * CE_QEXT_BANDS],
        }
    }
}

/// Range coder fields (mirrors `OracleEcState`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EcSnap {
    pub storage: u32,
    pub end_offs: u32,
    pub end_window: u32,
    pub nend_bits: c_int,
    pub nbits_total: c_int,
    pub offs: u32,
    pub rng: u32,
    pub val: u32,
    pub ext: u32,
    pub rem: c_int,
    pub error: c_int,
}

/// Raw mirror of `OracleCeCall`.
#[repr(C)]
struct RawCall {
    pre: CeState,
    post: CeState,
    frame_size: c_int,
    nb_compressed_bytes: c_int,
    ret: c_int,
    has_compressed: c_int,
    has_enc: c_int,
    pcm: *const f32,
    pcm_len: c_int,
    comp_before: *const u8,
    comp_after: *const u8,
    comp_len: c_int,
    enc_before: EcSnap,
    enc_after: EcSnap,
    buf_before: *const u8,
    buf_after: *const u8,
    buf_len: c_int,
    buf_shift: c_int,
}

/// One recorded CELT call made by the Opus encoder.
#[derive(Debug, Clone)]
pub struct CeltCall {
    pub pre: Box<CeState>,
    pub post: Box<CeState>,
    pub frame_size: i32,
    pub nb_compressed_bytes: i32,
    pub ret: i32,
    pub pcm: Vec<f32>,
    /// `compressed[0..nb]` before/after when the call had a `compressed` buffer.
    pub compressed: Option<(Vec<u8>, Vec<u8>)>,
    /// Range coder state before/after and `[toc, buf[0..storage)]` before/after when the call
    /// had an existing range coder, plus how far `enc->buf` moved (QEXT).
    pub enc: Option<EncRecord>,
}

/// Range-coder part of a [`CeltCall`].
#[derive(Debug, Clone)]
pub struct EncRecord {
    pub before: EcSnap,
    pub after: EcSnap,
    pub buf_before: Vec<u8>,
    pub buf_after: Vec<u8>,
    pub buf_shift: i32,
}

/// Opaque `CeHandle`.
#[repr(C)]
struct CeHandle {
    _p: [u8; 0],
}

unsafe extern "C" {
    fn oracle_ce_has_qext() -> c_int;
    fn oracle_ce_has_custom() -> c_int;
    fn oracle_ce_state_size() -> usize;
    fn oracle_ce_state_offset(which: c_int) -> usize;
    fn oracle_ce_call_size() -> usize;
    fn oracle_ce_rec_clear();
    fn oracle_ce_rec_count() -> c_int;
    fn oracle_ce_rec_get(i: c_int) -> *const RawCall;
    fn oracle_ce_opus_create(
        fs: c_int,
        channels: c_int,
        app: c_int,
        err: *mut c_int,
    ) -> *mut c_void;
    fn oracle_ce_opus_destroy(st: *mut c_void);
    fn oracle_ce_opus_ctl(st: *mut c_void, request: c_int, value: c_int) -> c_int;
    fn oracle_ce_opus_encode_float(
        st: *mut c_void,
        pcm: *const f32,
        frame_size: c_int,
        out: *mut u8,
        max_bytes: c_int,
    ) -> c_int;
    fn oracle_ce_new(fs: c_int, channels: c_int, err: *mut c_int) -> *mut CeHandle;
    fn oracle_ce_custom_new(
        fs: c_int,
        frame_size: c_int,
        channels: c_int,
        err: *mut c_int,
    ) -> *mut CeHandle;
    fn oracle_ce_free(h: *mut CeHandle);
    fn oracle_ce_ctl(h: *mut CeHandle, request: c_int, value: c_int) -> c_int;
    fn oracle_ce_ctl_get(h: *mut CeHandle, request: c_int, value: *mut c_int) -> c_int;
    fn oracle_ce_set_analysis(
        h: *mut CeHandle,
        valid: c_int,
        f: *const f32,
        bandwidth: c_int,
        leak_boost: *const u8,
    );
    fn oracle_ce_set_silk_info(h: *mut CeHandle, signal_type: c_int, offset: c_int);
    fn oracle_ce_set_energy_mask(h: *mut CeHandle, mask: *const f32, n: c_int);
    fn oracle_ce_get_state(h: *const CeHandle, out: *mut CeState);
    fn oracle_ce_encode(
        h: *mut CeHandle,
        pcm: *const f32,
        frame_size: c_int,
        out: *mut u8,
        nb: c_int,
    ) -> c_int;
    fn oracle_ce_encode_with_ec(
        h: *mut CeHandle,
        pcm: *const f32,
        frame_size: c_int,
        buf: *mut u8,
        buf_size: c_int,
        nb: c_int,
        ops: *const c_int,
        n_ops: c_int,
        out_enc: *mut EcSnap,
        buf_shift: *mut c_int,
    ) -> c_int;
    fn oracle_ce_custom_encode(
        h: *mut CeHandle,
        pcm: *const i16,
        frame_size: c_int,
        out: *mut u8,
        nb: c_int,
    ) -> c_int;
    fn oracle_ce_custom_encode24(
        h: *mut CeHandle,
        pcm: *const i32,
        frame_size: c_int,
        out: *mut u8,
        nb: c_int,
    ) -> c_int;
    fn oracle_ce_custom_encode_float(
        h: *mut CeHandle,
        pcm: *const f32,
        frame_size: c_int,
        out: *mut u8,
        nb: c_int,
    ) -> c_int;
    fn oracle_ce_transient_analysis(
        input: *const f32,
        len: c_int,
        c: c_int,
        tf_estimate: *mut f32,
        tf_chan: *mut c_int,
        allow_weak: c_int,
        weak: *mut c_int,
        tone_freq: f32,
        toneishness: f32,
    ) -> c_int;
    fn oracle_ce_patch_transient_decision(
        new_e: *mut f32,
        old_e: *mut f32,
        nb_ebands: c_int,
        start: c_int,
        end: c_int,
        c: c_int,
    ) -> c_int;
    fn oracle_ce_compute_mdcts(
        fs: c_int,
        short_blocks: c_int,
        input: *mut f32,
        out: *mut f32,
        c: c_int,
        cc: c_int,
        lm: c_int,
        upsample: c_int,
    );
    fn oracle_ce_preemphasis(
        pcm: *const f32,
        inp: *mut f32,
        n: c_int,
        cc: c_int,
        upsample: c_int,
        coef: *const f32,
        mem: *mut f32,
        clip: c_int,
    );
    fn oracle_ce_tf_analysis(
        fs: c_int,
        len: c_int,
        is_transient: c_int,
        tf_res: *mut c_int,
        lambda: c_int,
        x: *mut f32,
        n0: c_int,
        lm: c_int,
        tf_estimate: f32,
        tf_chan: c_int,
        importance: *mut c_int,
    ) -> c_int;
    fn oracle_ce_tf_encode(
        start: c_int,
        end: c_int,
        is_transient: c_int,
        tf_res: *mut c_int,
        lm: c_int,
        tf_select: c_int,
        buf: *mut u8,
        size: c_int,
        ops: *const c_int,
        n_ops: c_int,
        out: *mut EcSnap,
    );
    fn oracle_ce_alloc_trim_analysis(
        fs: c_int,
        x: *const f32,
        band_log_e: *const f32,
        end: c_int,
        lm: c_int,
        c: c_int,
        n0: c_int,
        an_valid: c_int,
        tonality_slope: f32,
        stereo_saving: *mut f32,
        tf_estimate: f32,
        intensity: c_int,
        surround_trim: f32,
        equiv_rate: c_int,
    ) -> c_int;
    fn oracle_ce_stereo_analysis(fs: c_int, x: *const f32, lm: c_int, n0: c_int) -> c_int;
    fn oracle_ce_median_of_5(x: *const f32) -> f32;
    fn oracle_ce_median_of_3(x: *const f32) -> f32;
    fn oracle_ce_dynalloc_analysis(
        fs: c_int,
        band_log_e: *const f32,
        band_log_e2: *const f32,
        old_band_e: *const f32,
        start: c_int,
        end: c_int,
        c: c_int,
        offsets: *mut c_int,
        lsb_depth: c_int,
        is_transient: c_int,
        vbr: c_int,
        constrained_vbr: c_int,
        lm: c_int,
        effective_bytes: c_int,
        tot_boost: *mut c_int,
        lfe: c_int,
        surround_dynalloc: *mut f32,
        an_valid: c_int,
        leak_boost: *const u8,
        importance: *mut c_int,
        spread_weight: *mut c_int,
        tone_freq: f32,
        toneishness: f32,
    ) -> f32;
    fn oracle_ce_tone_lpc(x: *const f32, len: c_int, delay: c_int, lpc: *mut f32) -> c_int;
    fn oracle_ce_tone_detect(
        input: *const f32,
        cc: c_int,
        n: c_int,
        toneishness: *mut f32,
        fs: c_int,
    ) -> f32;
    fn oracle_ce_compute_vbr(
        fs: c_int,
        an_valid: c_int,
        activity: f32,
        tonality: f32,
        base_target: c_int,
        lm: c_int,
        bitrate: c_int,
        last_coded_bands: c_int,
        c: c_int,
        intensity: c_int,
        constrained_vbr: c_int,
        stereo_saving: f32,
        tot_boost: c_int,
        tf_estimate: f32,
        pitch_change: c_int,
        max_depth: f32,
        lfe: c_int,
        has_surround_mask: c_int,
        surround_masking: f32,
        temporal_vbr: f32,
        enable_qext: c_int,
    ) -> c_int;
}

/// True when the oracle was built with `ENABLE_QEXT`.
#[must_use]
pub fn has_qext() -> bool {
    // SAFETY: pure function without arguments.
    unsafe { oracle_ce_has_qext() != 0 }
}

/// True when the oracle was built with `CUSTOM_MODES`.
#[must_use]
pub fn has_custom() -> bool {
    // SAFETY: pure function without arguments.
    unsafe { oracle_ce_has_custom() != 0 }
}

/// Checks the `#[repr(C)]` mirrors against the C layout.
#[must_use]
pub fn layout_ok() -> bool {
    use core::mem::{offset_of, size_of};
    // SAFETY: pure functions returning sizes/offsets.
    unsafe {
        oracle_ce_state_size() == size_of::<CeState>()
            && oracle_ce_call_size() == size_of::<RawCall>()
            && oracle_ce_state_offset(0) == offset_of!(CeState, rng)
            && oracle_ce_state_offset(1) == offset_of!(CeState, an_leak_boost)
            && oracle_ce_state_offset(2) == offset_of!(CeState, silk_signal_type)
            && oracle_ce_state_offset(3) == offset_of!(CeState, energy_mask)
            && oracle_ce_state_offset(4) == offset_of!(CeState, in_mem)
            && oracle_ce_state_offset(5) == offset_of!(CeState, qext_old_band_e)
    }
}

/// A persistent library CELT encoder.
#[derive(Debug)]
pub struct CeltEnc {
    h: *mut CeHandle,
}

impl CeltEnc {
    /// `celt_encoder_init(fs, channels)` + `CELT_SET_SIGNALLING(0)`.
    ///
    /// # Errors
    /// The C error code.
    pub fn new(fs: i32, channels: i32) -> Result<Self, i32> {
        let mut err: c_int = 0;
        // SAFETY: err is a valid out pointer.
        let h = unsafe { oracle_ce_new(fs, channels, &mut err) };
        if h.is_null() {
            Err(err)
        } else {
            Ok(Self { h })
        }
    }

    /// `opus_custom_encoder_create(opus_custom_mode_create(fs, frame_size), channels)`
    /// (custom-modes oracle only).
    ///
    /// # Errors
    /// The C error code.
    pub fn new_custom(fs: i32, frame_size: i32, channels: i32) -> Result<Self, i32> {
        let mut err: c_int = 0;
        // SAFETY: err is a valid out pointer.
        let h = unsafe { oracle_ce_custom_new(fs, frame_size, channels, &mut err) };
        if h.is_null() {
            Err(err)
        } else {
            Ok(Self { h })
        }
    }

    /// `opus_custom_encoder_ctl(st, request, (opus_int32)value)` for int-valued requests.
    pub fn ctl(&mut self, request: i32, value: i32) -> i32 {
        // SAFETY: h is a live handle; the request takes an opus_int32 argument.
        unsafe { oracle_ce_ctl(self.h, request, value) }
    }

    /// GET request taking an `opus_int32*` (or `opus_uint32*` for the final range).
    pub fn ctl_get(&mut self, request: i32) -> (i32, i32) {
        let mut v: c_int = 0;
        // SAFETY: h is a live handle; v is a valid out pointer.
        let r = unsafe { oracle_ce_ctl_get(self.h, request, &mut v) };
        (r, v)
    }

    /// `CELT_SET_ANALYSIS`. `f` = tonality, tonality_slope, noisiness, activity, music_prob,
    /// music_prob_min, music_prob_max, activity_probability, max_pitch_ratio.
    pub fn set_analysis(&mut self, valid: i32, f: &[f32; 9], bandwidth: i32, leak: &[u8; 19]) {
        // SAFETY: h is live; f has 9 floats and leak 19 bytes as the shim reads.
        unsafe { oracle_ce_set_analysis(self.h, valid, f.as_ptr(), bandwidth, leak.as_ptr()) }
    }

    /// `CELT_SET_SILK_INFO`.
    pub fn set_silk_info(&mut self, signal_type: i32, offset: i32) {
        // SAFETY: h is live.
        unsafe { oracle_ce_set_silk_info(self.h, signal_type, offset) }
    }

    /// `OPUS_SET_ENERGY_MASK` (the shim keeps a copy alive; `None` clears it).
    pub fn set_energy_mask(&mut self, mask: Option<&[f32]>) {
        match mask {
            // SAFETY: h is live; NULL clears the mask.
            None => unsafe { oracle_ce_set_energy_mask(self.h, core::ptr::null(), 0) },
            Some(m) => {
                assert!(m.len() <= 2 * CE_MAX_BANDS);
                // SAFETY: h is live; m has m.len() floats (<= the shim's buffer).
                unsafe { oracle_ce_set_energy_mask(self.h, m.as_ptr(), m.len() as c_int) }
            }
        }
    }

    /// Full state dump.
    #[must_use]
    pub fn state(&self) -> Box<CeState> {
        let mut s = Box::new(CeState::zeroed());
        // SAFETY: h is live; s is a valid CeState (layout checked by `layout_ok`).
        unsafe { oracle_ce_get_state(self.h, &mut *s) };
        s
    }

    /// `celt_encode_with_ec(st, pcm, frame_size, out, nb, NULL)`.
    pub fn encode(
        &mut self,
        pcm: &[f32],
        channels: usize,
        frame_size: i32,
        out: &mut [u8],
        nb: i32,
    ) -> i32 {
        assert!(pcm.len() >= channels * frame_size.max(0) as usize);
        assert!(out.len() >= nb.max(0) as usize);
        // SAFETY: lengths checked above; h is live.
        unsafe { oracle_ce_encode(self.h, pcm.as_ptr(), frame_size, out.as_mut_ptr(), nb) }
    }

    /// Hybrid-style call with a shared range coder over `buf[1..]` (`buf[0]` = TOC); `ops` are
    /// `(kind, value, param)` writes (0: bit_logp, 1: uint, 2: bits) done first; the coder is
    /// shrunk to `nb` when smaller. Returns `(ret, coder state, enc->buf shift)`.
    pub fn encode_with_ec(
        &mut self,
        pcm: &[f32],
        channels: usize,
        frame_size: i32,
        buf: &mut [u8],
        nb: i32,
        ops: &[[i32; 3]],
    ) -> (i32, EcSnap, i32) {
        assert!(pcm.len() >= channels * frame_size.max(0) as usize);
        assert!(buf.len() >= 2 && (nb as usize) < buf.len());
        let mut snap = EcSnap::default();
        let mut shift: c_int = 0;
        // SAFETY: lengths checked above; ops is a flat array of 3*len ints; h is live.
        let r = unsafe {
            oracle_ce_encode_with_ec(
                self.h,
                pcm.as_ptr(),
                frame_size,
                buf.as_mut_ptr(),
                buf.len() as c_int,
                nb,
                ops.as_ptr().cast(),
                ops.len() as c_int,
                &mut snap,
                &mut shift,
            )
        };
        (r, snap, shift)
    }

    /// `opus_custom_encode` (custom-modes oracle).
    pub fn custom_encode(
        &mut self,
        pcm: &[i16],
        channels: usize,
        frame_size: i32,
        out: &mut [u8],
        nb: i32,
    ) -> i32 {
        assert!(pcm.len() >= channels * frame_size.max(0) as usize);
        assert!(out.len() >= nb.max(0) as usize);
        // SAFETY: lengths checked above; h is live.
        unsafe { oracle_ce_custom_encode(self.h, pcm.as_ptr(), frame_size, out.as_mut_ptr(), nb) }
    }

    /// `opus_custom_encode24` (custom-modes oracle).
    pub fn custom_encode24(
        &mut self,
        pcm: &[i32],
        channels: usize,
        frame_size: i32,
        out: &mut [u8],
        nb: i32,
    ) -> i32 {
        assert!(pcm.len() >= channels * frame_size.max(0) as usize);
        assert!(out.len() >= nb.max(0) as usize);
        // SAFETY: lengths checked above; h is live.
        unsafe { oracle_ce_custom_encode24(self.h, pcm.as_ptr(), frame_size, out.as_mut_ptr(), nb) }
    }

    /// `opus_custom_encode_float` (custom-modes oracle).
    pub fn custom_encode_float(
        &mut self,
        pcm: &[f32],
        channels: usize,
        frame_size: i32,
        out: &mut [u8],
        nb: i32,
    ) -> i32 {
        assert!(pcm.len() >= channels * frame_size.max(0) as usize);
        assert!(out.len() >= nb.max(0) as usize);
        // SAFETY: lengths checked above; h is live.
        unsafe {
            oracle_ce_custom_encode_float(self.h, pcm.as_ptr(), frame_size, out.as_mut_ptr(), nb)
        }
    }
}

impl Drop for CeltEnc {
    fn drop(&mut self) {
        // SAFETY: h was created by oracle_ce_new/custom_new and is freed once.
        unsafe { oracle_ce_free(self.h) }
    }
}

/// A private copy of the Opus encoder whose CELT calls are recorded.
#[derive(Debug)]
pub struct OpusRec {
    st: *mut c_void,
    channels: usize,
}

impl OpusRec {
    /// `opus_encoder_create(fs, channels, application)`.
    ///
    /// # Errors
    /// The C error code.
    pub fn new(fs: i32, channels: i32, application: i32) -> Result<Self, i32> {
        let mut err: c_int = 0;
        // SAFETY: err is a valid out pointer.
        let st = unsafe { oracle_ce_opus_create(fs, channels, application, &mut err) };
        if st.is_null() {
            Err(err)
        } else {
            Ok(Self {
                st,
                channels: channels as usize,
            })
        }
    }

    /// `opus_encoder_ctl(st, request, (opus_int32)value)` for int-valued SET requests.
    pub fn ctl(&mut self, request: i32, value: i32) -> i32 {
        // SAFETY: st is live; the request takes an opus_int32 argument.
        unsafe { oracle_ce_opus_ctl(self.st, request, value) }
    }

    /// `opus_encode_float`, returning the packet length (or error) and the recorded CELT calls.
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: i32,
        out: &mut [u8],
    ) -> (i32, Vec<CeltCall>) {
        assert!(pcm.len() >= self.channels * frame_size.max(0) as usize);
        // SAFETY: lengths checked above; st is live.
        let ret = unsafe {
            oracle_ce_opus_encode_float(
                self.st,
                pcm.as_ptr(),
                frame_size,
                out.as_mut_ptr(),
                out.len() as c_int,
            )
        };
        // SAFETY: reads the thread-local record list filled by the call above.
        let n = unsafe { oracle_ce_rec_count() };
        let mut calls = Vec::with_capacity(n as usize);
        for i in 0..n {
            // SAFETY: i < count, so the pointer is valid until the next clear.
            let r = unsafe { &*oracle_ce_rec_get(i) };
            // SAFETY: the shim allocated pcm_len floats / comp_len / buf_len bytes.
            let pcm = unsafe { core::slice::from_raw_parts(r.pcm, r.pcm_len as usize) }.to_vec();
            let compressed = (r.has_compressed != 0).then(|| {
                let n = r.comp_len as usize;
                // SAFETY: see above.
                unsafe {
                    (
                        core::slice::from_raw_parts(r.comp_before, n).to_vec(),
                        core::slice::from_raw_parts(r.comp_after, n).to_vec(),
                    )
                }
            });
            let enc = (r.has_enc != 0).then(|| {
                let n = r.buf_len as usize;
                EncRecord {
                    before: r.enc_before,
                    after: r.enc_after,
                    // SAFETY: see above.
                    buf_before: unsafe { core::slice::from_raw_parts(r.buf_before, n) }.to_vec(),
                    // SAFETY: see above.
                    buf_after: unsafe { core::slice::from_raw_parts(r.buf_after, n) }.to_vec(),
                    buf_shift: r.buf_shift,
                }
            });
            calls.push(CeltCall {
                pre: Box::new(r.pre),
                post: Box::new(r.post),
                frame_size: r.frame_size,
                nb_compressed_bytes: r.nb_compressed_bytes,
                ret: r.ret,
                pcm,
                compressed,
                enc,
            });
        }
        // SAFETY: frees the records copied above.
        unsafe { oracle_ce_rec_clear() };
        (ret, calls)
    }
}

impl Drop for OpusRec {
    fn drop(&mut self) {
        // SAFETY: st was created by oracle_ce_opus_create and is destroyed once.
        unsafe { oracle_ce_opus_destroy(self.st) }
    }
}

/// `transient_analysis` → `(is_transient, tf_estimate, tf_chan, weak_transient)`.
#[must_use]
pub fn transient_analysis(
    input: &[f32],
    len: i32,
    c: i32,
    allow_weak: bool,
    tone_freq: f32,
    toneishness: f32,
) -> (bool, f32, i32, bool) {
    assert!(input.len() >= (len * c) as usize);
    let (mut tf, mut chan, mut weak) = (0.0f32, 0 as c_int, 0 as c_int);
    // SAFETY: input has len*c floats; out pointers are valid.
    let r = unsafe {
        oracle_ce_transient_analysis(
            input.as_ptr(),
            len,
            c,
            &mut tf,
            &mut chan,
            c_int::from(allow_weak),
            &mut weak,
            tone_freq,
            toneishness,
        )
    };
    (r != 0, tf, chan, weak != 0)
}

/// `patch_transient_decision`.
#[must_use]
pub fn patch_transient_decision(
    new_e: &[f32],
    old_e: &[f32],
    nb_ebands: i32,
    start: i32,
    end: i32,
    c: i32,
) -> bool {
    assert!(new_e.len() >= (c * nb_ebands) as usize && old_e.len() >= (c * nb_ebands) as usize);
    let mut n = new_e.to_vec();
    let mut o = old_e.to_vec();
    // SAFETY: both arrays hold c*nb_ebands floats (the function only reads them).
    unsafe {
        oracle_ce_patch_transient_decision(n.as_mut_ptr(), o.as_mut_ptr(), nb_ebands, start, end, c)
            != 0
    }
}

/// `compute_mdcts` on the mode for `fs` (48000, or 96000 with QEXT). `input` needs
/// `cc*(N+overlap)` samples and `out` `cc*N`.
pub fn compute_mdcts(
    fs: i32,
    short_blocks: i32,
    input: &[f32],
    out: &mut [f32],
    c: i32,
    cc: i32,
    lm: i32,
    upsample: i32,
) {
    let mut inp = input.to_vec();
    // SAFETY: the caller sizes input/out for the mode (asserted by the Rust twin in tests).
    unsafe {
        oracle_ce_compute_mdcts(
            fs,
            short_blocks,
            inp.as_mut_ptr(),
            out.as_mut_ptr(),
            c,
            cc,
            lm,
            upsample,
        )
    }
}

/// `celt_preemphasis` (library). `pcm` starts at the channel's first sample.
pub fn preemphasis(
    pcm: &[f32],
    inp: &mut [f32],
    n: i32,
    cc: i32,
    upsample: i32,
    coef: &[f32; 4],
    mem: &mut f32,
    clip: bool,
) {
    assert!(inp.len() >= n as usize);
    assert!(pcm.len() > (cc as usize) * ((n / upsample) as usize - 1));
    // SAFETY: lengths checked above.
    unsafe {
        oracle_ce_preemphasis(
            pcm.as_ptr(),
            inp.as_mut_ptr(),
            n,
            cc,
            upsample,
            coef.as_ptr(),
            mem,
            c_int::from(clip),
        )
    }
}

/// `tf_analysis` → `tf_select`; writes `tf_res[..len]`.
pub fn tf_analysis(
    fs: i32,
    len: i32,
    is_transient: bool,
    tf_res: &mut [i32],
    lambda: i32,
    x: &[f32],
    n0: i32,
    lm: i32,
    tf_estimate: f32,
    tf_chan: i32,
    importance: &[i32],
) -> i32 {
    assert!(tf_res.len() >= len as usize && importance.len() >= len as usize);
    let mut xx = x.to_vec();
    let mut imp = importance.to_vec();
    // SAFETY: arrays sized for len bands; x covers the channel data read.
    unsafe {
        oracle_ce_tf_analysis(
            fs,
            len,
            c_int::from(is_transient),
            tf_res.as_mut_ptr(),
            lambda,
            xx.as_mut_ptr(),
            n0,
            lm,
            tf_estimate,
            tf_chan,
            imp.as_mut_ptr(),
        )
    }
}

/// `tf_encode` on a fresh coder over `buf` after the `ops` pre-writes, then `ec_enc_done`.
pub fn tf_encode(
    start: i32,
    end: i32,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: i32,
    tf_select: i32,
    buf: &mut [u8],
    ops: &[[i32; 3]],
) -> EcSnap {
    assert!(tf_res.len() >= end as usize);
    let mut snap = EcSnap::default();
    // SAFETY: tf_res has end entries; buf.len() bytes are the coder storage.
    unsafe {
        oracle_ce_tf_encode(
            start,
            end,
            c_int::from(is_transient),
            tf_res.as_mut_ptr(),
            lm,
            tf_select,
            buf.as_mut_ptr(),
            buf.len() as c_int,
            ops.as_ptr().cast(),
            ops.len() as c_int,
            &mut snap,
        )
    };
    snap
}

/// `alloc_trim_analysis`.
pub fn alloc_trim_analysis(
    fs: i32,
    x: &[f32],
    band_log_e: &[f32],
    end: i32,
    lm: i32,
    c: i32,
    n0: i32,
    an_valid: bool,
    tonality_slope: f32,
    stereo_saving: &mut f32,
    tf_estimate: f32,
    intensity: i32,
    surround_trim: f32,
    equiv_rate: i32,
) -> i32 {
    // SAFETY: the caller sizes x (c*n0) and band_log_e (c*nbEBands).
    unsafe {
        oracle_ce_alloc_trim_analysis(
            fs,
            x.as_ptr(),
            band_log_e.as_ptr(),
            end,
            lm,
            c,
            n0,
            c_int::from(an_valid),
            tonality_slope,
            stereo_saving,
            tf_estimate,
            intensity,
            surround_trim,
            equiv_rate,
        )
    }
}

/// `stereo_analysis`.
#[must_use]
pub fn stereo_analysis(fs: i32, x: &[f32], lm: i32, n0: i32) -> bool {
    assert!(x.len() >= 2 * n0 as usize);
    // SAFETY: x holds two channels of n0 samples.
    unsafe { oracle_ce_stereo_analysis(fs, x.as_ptr(), lm, n0) != 0 }
}

/// `median_of_5`.
#[must_use]
pub fn median_of_5(x: &[f32; 5]) -> f32 {
    // SAFETY: 5 floats.
    unsafe { oracle_ce_median_of_5(x.as_ptr()) }
}

/// `median_of_3`.
#[must_use]
pub fn median_of_3(x: &[f32; 3]) -> f32 {
    // SAFETY: 3 floats.
    unsafe { oracle_ce_median_of_3(x.as_ptr()) }
}

/// `dynalloc_analysis` on the mode for `fs` → `(maxDepth, tot_boost)`; writes offsets,
/// importance and spread_weight (`nbEBands` each).
pub fn dynalloc_analysis(
    fs: i32,
    band_log_e: &[f32],
    band_log_e2: &[f32],
    old_band_e: &[f32],
    start: i32,
    end: i32,
    c: i32,
    offsets: &mut [i32],
    lsb_depth: i32,
    is_transient: bool,
    vbr: bool,
    constrained_vbr: bool,
    lm: i32,
    effective_bytes: i32,
    lfe: bool,
    surround_dynalloc: &[f32],
    an_valid: bool,
    leak_boost: &[u8; 19],
    importance: &mut [i32],
    spread_weight: &mut [i32],
    tone_freq: f32,
    toneishness: f32,
) -> (f32, i32) {
    let mut tot = 0 as c_int;
    let mut sd = surround_dynalloc.to_vec();
    // SAFETY: the caller sizes the per-band arrays for the mode (c*nbEBands / nbEBands).
    let md = unsafe {
        oracle_ce_dynalloc_analysis(
            fs,
            band_log_e.as_ptr(),
            band_log_e2.as_ptr(),
            old_band_e.as_ptr(),
            start,
            end,
            c,
            offsets.as_mut_ptr(),
            lsb_depth,
            c_int::from(is_transient),
            c_int::from(vbr),
            c_int::from(constrained_vbr),
            lm,
            effective_bytes,
            &mut tot,
            c_int::from(lfe),
            sd.as_mut_ptr(),
            c_int::from(an_valid),
            leak_boost.as_ptr(),
            importance.as_mut_ptr(),
            spread_weight.as_mut_ptr(),
            tone_freq,
            toneishness,
        )
    };
    (md, tot)
}

/// `tone_lpc` → `(fail, lpc)`.
#[must_use]
pub fn tone_lpc(x: &[f32], len: i32, delay: i32, lpc_in: [f32; 2]) -> (bool, [f32; 2]) {
    assert!(x.len() >= len as usize);
    let mut lpc = lpc_in;
    // SAFETY: x has len floats; lpc has 2.
    let r = unsafe { oracle_ce_tone_lpc(x.as_ptr(), len, delay, lpc.as_mut_ptr()) };
    (r != 0, lpc)
}

/// `tone_detect` → `(freq, toneishness)`.
#[must_use]
pub fn tone_detect(input: &[f32], cc: i32, n: i32, fs: i32) -> (f32, f32) {
    assert!(input.len() >= (cc * n) as usize);
    let mut t = 0.0f32;
    // SAFETY: input has cc*n floats.
    let f = unsafe { oracle_ce_tone_detect(input.as_ptr(), cc, n, &mut t, fs) };
    (f, t)
}

/// `compute_vbr` on the mode for `fs`.
#[must_use]
pub fn compute_vbr(
    fs: i32,
    an_valid: bool,
    activity: f32,
    tonality: f32,
    base_target: i32,
    lm: i32,
    bitrate: i32,
    last_coded_bands: i32,
    c: i32,
    intensity: i32,
    constrained_vbr: bool,
    stereo_saving: f32,
    tot_boost: i32,
    tf_estimate: f32,
    pitch_change: bool,
    max_depth: f32,
    lfe: bool,
    has_surround_mask: bool,
    surround_masking: f32,
    temporal_vbr: f32,
    enable_qext: bool,
) -> i32 {
    // SAFETY: scalar arguments only.
    unsafe {
        oracle_ce_compute_vbr(
            fs,
            c_int::from(an_valid),
            activity,
            tonality,
            base_target,
            lm,
            bitrate,
            last_coded_bands,
            c,
            intensity,
            c_int::from(constrained_vbr),
            stereo_saving,
            tot_boost,
            tf_estimate,
            c_int::from(pitch_change),
            max_depth,
            c_int::from(lfe),
            c_int::from(has_surround_mask),
            surround_masking,
            temporal_vbr,
            c_int::from(enable_qext),
        )
    }
}
