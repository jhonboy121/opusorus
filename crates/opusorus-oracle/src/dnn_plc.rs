//! Oracle bindings for unit `dnn_plc`: dnn/fargan.c (FARGAN vocoder) and dnn/lpcnet_plc.c
//! (neural PLC), generic C path.
//!
//! Only available when the oracle is built with a DNN feature (`deep-plc`, `dred` or `osce`
//! all define `ENABLE_DEEP_PLC`, see `build.rs`). States live on the C heap behind RAII
//! handles, initialised with the compiled-in model tables; every wrapper checks the sizes the
//! C code reads or writes.
#![cfg(any(feature = "deep-plc", feature = "dred", feature = "osce"))]

use core::ffi::{c_int, c_void};

/// Samples per FARGAN / PLC frame (`FARGAN_FRAME_SIZE` = `FRAME_SIZE` = 160).
pub const FRAME: usize = 160;
/// `FARGAN_SUBFRAME_SIZE`.
pub const SUBFRAME: usize = 40;
/// `NB_FEATURES`.
pub const NB_FEATURES: usize = 20;
/// `FARGAN_CONT_SAMPLES`.
pub const CONT_SAMPLES: usize = 320;
/// `CONT_VECTORS`.
pub const CONT_VECTORS: usize = 5;
/// `COND_NET_FDENSE2_OUT_SIZE`.
pub const COND_SIZE: usize = 320;
/// `FARGAN_COND_SIZE`.
pub const SUB_COND_SIZE: usize = 80;
/// PLC network input size (`2*NB_BANDS+NB_FEATURES+1`).
pub const PLC_INPUT_SIZE: usize = 57;
/// `PLC_BUF_SIZE`.
pub const PLC_BUF_SIZE: usize = 2400;
/// `PLC_MAX_FEC`.
pub const PLC_MAX_FEC: usize = 104;

/// Floats in [`Fargan::state`]: deemph_mem, pitch_buf, cond_conv1_state, fwc0_mem,
/// gru1/2/3_state.
pub const FARGAN_STATE_LEN: usize = 1 + 256 + 128 + 328 + 160 + 128 + 128;
/// Floats of the `LPCNetEncState` part of [`Plc::state`] (dnn_core order).
pub const ENC_STATE_LEN: usize = 160
    + 1
    + 60
    + 88
    + 224
    + 1
    + 16
    + 1
    + 576
    + 576
    + 4
    + 16
    + 36
    + 16
    + 36
    + 64
    + 2 * 226
    + 16 * 226;
/// Floats in [`Plc::state`]: fec, pcm, features, cont_features, plc_net, plc_bak, FARGAN,
/// encoder.
pub const PLC_STATE_LEN: usize = PLC_MAX_FEC * NB_FEATURES
    + PLC_BUF_SIZE
    + 36
    + CONT_VECTORS * NB_FEATURES
    + 3 * 384
    + FARGAN_STATE_LEN
    + ENC_STATE_LEN;
/// Ints in [`Plc::state`]: loaded, analysis_gap, fec_read_pos, fec_fill_pos, fec_skip,
/// analysis_pos, predict_pos, blend, loss_count, fargan.cont_initialized, fargan.last_period.
pub const PLC_STATE_INTS: usize = 11;

unsafe extern "C" {
    fn oracle_pc_const(which: c_int) -> c_int;

    fn oracle_pc_fargan_new() -> *mut c_void;
    fn oracle_pc_fargan_free(p: *mut c_void);
    fn oracle_pc_fargan_init(p: *mut c_void);
    fn oracle_pc_fargan_load_model(p: *mut c_void, data: *const u8, len: c_int) -> c_int;
    fn oracle_pc_fargan_cont(p: *mut c_void, pcm0: *const f32, features0: *const f32);
    fn oracle_pc_fargan_synthesize(p: *mut c_void, pcm: *mut f32, features: *const f32);
    fn oracle_pc_fargan_synthesize_int(p: *mut c_void, pcm: *mut i16, features: *const f32);
    fn oracle_pc_fargan_state(p: *mut c_void, out: *mut f32, ints: *mut c_int) -> c_int;
    fn oracle_pc_fargan_set_state(p: *mut c_void, input: *const f32, ints: *const c_int) -> c_int;
    fn oracle_pc_compute_fargan_cond(
        p: *mut c_void,
        cond: *mut f32,
        features: *const f32,
        period: c_int,
    );
    fn oracle_pc_fargan_deemphasis(pcm: *mut f32, mem: *mut f32);
    fn oracle_pc_run_fargan_subframe(
        p: *mut c_void,
        pcm: *mut f32,
        cond: *const f32,
        period: c_int,
    );

    fn oracle_pc_plc_new() -> *mut c_void;
    fn oracle_pc_plc_free(p: *mut c_void);
    fn oracle_pc_plc_reset(p: *mut c_void);
    fn oracle_pc_plc_load_model(p: *mut c_void, data: *const u8, len: c_int) -> c_int;
    fn oracle_pc_plc_update(p: *mut c_void, pcm: *mut i16) -> c_int;
    fn oracle_pc_plc_conceal(p: *mut c_void, pcm: *mut i16) -> c_int;
    fn oracle_pc_plc_fec_add(p: *mut c_void, features: *const f32);
    fn oracle_pc_plc_fec_clear(p: *mut c_void);
    fn oracle_pc_compute_plc_pred(p: *mut c_void, out: *mut f32, input: *const f32);
    fn oracle_pc_get_fec_or_pred(p: *mut c_void, out: *mut f32) -> c_int;
    fn oracle_pc_queue_features(p: *mut c_void, features: *const f32);
    fn oracle_pc_plc_state(p: *mut c_void, out: *mut f32, ints: *mut c_int) -> c_int;
}

fn ci(v: usize) -> c_int {
    c_int::try_from(v).expect("size fits in a C int")
}

/// Compile-time sizes seen by the C side: (FARGAN_COND_SIZE, COND_NET_FDENSE2_OUT_SIZE,
/// SIG_NET_INPUT_SIZE, PLC_BUF_SIZE, PLC_MAX_FEC).
#[must_use]
pub fn c_sizes() -> [usize; 5] {
    // SAFETY: pure function returning constants.
    let get = |w| usize::try_from(unsafe { oracle_pc_const(w) }).expect("valid constant");
    [get(0), get(1), get(2), get(3), get(4)]
}

/// C `FARGANState` (heap, `fargan_init` with the compiled-in model).
#[derive(Debug)]
pub struct Fargan(*mut c_void);

impl Fargan {
    #[must_use]
    #[expect(clippy::new_without_default, reason = "allocates C state")]
    pub fn new() -> Self {
        // SAFETY: allocates and initialises a C state.
        Self(unsafe { oracle_pc_fargan_new() })
    }

    /// C `fargan_init` (clears the state, re-binds the compiled-in model).
    pub fn init(&mut self) {
        // SAFETY: valid handle.
        unsafe { oracle_pc_fargan_init(self.0) }
    }

    /// C `fargan_load_model` (0 on success). The blob must be well formed (C crashes on an
    /// unparsable one).
    pub fn load_model(&mut self, blob: &[u8]) -> i32 {
        // SAFETY: valid handle; C reads `len` bytes.
        unsafe { oracle_pc_fargan_load_model(self.0, blob.as_ptr(), ci(blob.len())) }
    }

    /// C `fargan_cont` (320 samples, 5*20 features).
    pub fn cont(&mut self, pcm0: &[f32], features0: &[f32]) {
        assert!(pcm0.len() >= CONT_SAMPLES && features0.len() >= CONT_VECTORS * NB_FEATURES);
        // SAFETY: sizes checked.
        unsafe { oracle_pc_fargan_cont(self.0, pcm0.as_ptr(), features0.as_ptr()) }
    }

    /// C `fargan_synthesize` (C reads `features[0..NB_FEATURES]`).
    pub fn synthesize(&mut self, features: &[f32]) -> [f32; FRAME] {
        assert!(features.len() >= NB_FEATURES);
        let mut pcm = [0f32; FRAME];
        // SAFETY: sizes checked.
        unsafe { oracle_pc_fargan_synthesize(self.0, pcm.as_mut_ptr(), features.as_ptr()) };
        pcm
    }

    /// C `fargan_synthesize_int`.
    pub fn synthesize_int(&mut self, features: &[f32]) -> [i16; FRAME] {
        assert!(features.len() >= NB_FEATURES);
        let mut pcm = [0i16; FRAME];
        // SAFETY: sizes checked.
        unsafe { oracle_pc_fargan_synthesize_int(self.0, pcm.as_mut_ptr(), features.as_ptr()) };
        pcm
    }

    /// State dump: ([`FARGAN_STATE_LEN`] floats, [cont_initialized, last_period]).
    #[must_use]
    pub fn state(&self) -> (Vec<f32>, [i32; 2]) {
        let mut f = vec![0f32; FARGAN_STATE_LEN];
        let mut i = [0 as c_int; 2];
        // SAFETY: buffers sized to the C dump.
        let n = unsafe { oracle_pc_fargan_state(self.0, f.as_mut_ptr(), i.as_mut_ptr()) };
        assert_eq!(usize::try_from(n).expect("count"), FARGAN_STATE_LEN);
        (f, i)
    }

    /// Overwrites the state (same layout as [`Self::state`]).
    pub fn set_state(&mut self, floats: &[f32], ints: [i32; 2]) {
        assert_eq!(floats.len(), FARGAN_STATE_LEN);
        // SAFETY: sizes checked.
        let n = unsafe { oracle_pc_fargan_set_state(self.0, floats.as_ptr(), ints.as_ptr()) };
        assert_eq!(usize::try_from(n).expect("count"), FARGAN_STATE_LEN);
    }

    /// C `compute_fargan_cond` (static; 20 features → 320 cond values).
    pub fn compute_cond(&mut self, features: &[f32], period: i32) -> [f32; COND_SIZE] {
        assert!(features.len() >= NB_FEATURES);
        let mut cond = [0f32; COND_SIZE];
        // SAFETY: sizes checked; the pembed row index is clamped by C.
        unsafe {
            oracle_pc_compute_fargan_cond(self.0, cond.as_mut_ptr(), features.as_ptr(), period);
        }
        cond
    }

    /// C `run_fargan_subframe` (static; 80 cond values → 40 samples). `period` must be ≥ 1
    /// or give a negative start position (C reads past `pitch_buf` otherwise).
    pub fn run_subframe(&mut self, cond: &[f32], period: i32) -> [f32; SUBFRAME] {
        assert!(cond.len() >= SUB_COND_SIZE);
        let mut pcm = [0f32; SUBFRAME];
        // SAFETY: sizes checked; C stays within the state struct for any period.
        unsafe { oracle_pc_run_fargan_subframe(self.0, pcm.as_mut_ptr(), cond.as_ptr(), period) };
        pcm
    }
}

impl Drop for Fargan {
    fn drop(&mut self) {
        // SAFETY: allocated by oracle_pc_fargan_new.
        unsafe { oracle_pc_fargan_free(self.0) }
    }
}

/// C `fargan_deemphasis` (static; 40 samples in place).
pub fn fargan_deemphasis(pcm: &mut [f32], mem: &mut f32) {
    assert!(pcm.len() >= SUBFRAME);
    // SAFETY: sizes checked.
    unsafe { oracle_pc_fargan_deemphasis(pcm.as_mut_ptr(), mem) }
}

/// C `LPCNetPLCState` (heap, `lpcnet_plc_init` with the compiled-in models: `loaded` = 1).
#[derive(Debug)]
pub struct Plc(*mut c_void);

impl Plc {
    #[must_use]
    #[expect(clippy::new_without_default, reason = "allocates C state")]
    pub fn new() -> Self {
        // SAFETY: allocates and initialises a C state.
        Self(unsafe { oracle_pc_plc_new() })
    }

    /// C `lpcnet_plc_reset`.
    pub fn reset(&mut self) {
        // SAFETY: valid handle.
        unsafe { oracle_pc_plc_reset(self.0) }
    }

    /// C `lpcnet_plc_load_model` (0 on success). The blob must be well formed.
    pub fn load_model(&mut self, blob: &[u8]) -> i32 {
        // SAFETY: valid handle; C reads `len` bytes.
        unsafe { oracle_pc_plc_load_model(self.0, blob.as_ptr(), ci(blob.len())) }
    }

    /// C `lpcnet_plc_update` (160 samples); returns the C return value.
    pub fn update(&mut self, pcm: &[i16]) -> i32 {
        assert!(pcm.len() >= FRAME);
        let mut buf = [0i16; FRAME];
        buf.copy_from_slice(&pcm[..FRAME]);
        // SAFETY: 160 samples; C only reads them.
        unsafe { oracle_pc_plc_update(self.0, buf.as_mut_ptr()) }
    }

    /// C `lpcnet_plc_conceal`: (return value, 160 samples).
    pub fn conceal(&mut self) -> (i32, [i16; FRAME]) {
        let mut pcm = [0i16; FRAME];
        // SAFETY: 160-sample output buffer.
        let ret = unsafe { oracle_pc_plc_conceal(self.0, pcm.as_mut_ptr()) };
        (ret, pcm)
    }

    /// C `lpcnet_plc_fec_add` (`None` = `NULL`). Panics if the queue is full (C would write
    /// out of bounds).
    pub fn fec_add(&mut self, features: Option<&[f32]>) {
        if let Some(f) = features {
            assert!(f.len() >= NB_FEATURES);
            let (_, ints) = self.state();
            assert!(usize::try_from(ints[3]).expect("fill pos") < PLC_MAX_FEC);
        }
        let p = features.map_or(core::ptr::null(), <[f32]>::as_ptr);
        // SAFETY: NULL or 20 readable floats; queue not full (checked).
        unsafe { oracle_pc_plc_fec_add(self.0, p) }
    }

    /// C `lpcnet_plc_fec_clear`.
    pub fn fec_clear(&mut self) {
        // SAFETY: valid handle.
        unsafe { oracle_pc_plc_fec_clear(self.0) }
    }

    /// C `compute_plc_pred` (static; 57 inputs → 20 outputs).
    pub fn compute_plc_pred(&mut self, input: &[f32]) -> [f32; NB_FEATURES] {
        assert!(input.len() >= PLC_INPUT_SIZE);
        let mut out = [0f32; NB_FEATURES];
        // SAFETY: sizes checked.
        unsafe { oracle_pc_compute_plc_pred(self.0, out.as_mut_ptr(), input.as_ptr()) };
        out
    }

    /// C `get_fec_or_pred` (static): (return value, the 20 values written to `out`, which
    /// starts as `init`).
    pub fn get_fec_or_pred(&mut self, init: &[f32; NB_FEATURES]) -> (i32, [f32; NB_FEATURES]) {
        let mut out = *init;
        // SAFETY: 20-float output (C writes NB_FEATURES).
        let ret = unsafe { oracle_pc_get_fec_or_pred(self.0, out.as_mut_ptr()) };
        (ret, out)
    }

    /// C `queue_features` (static).
    pub fn queue_features(&mut self, features: &[f32]) {
        assert!(features.len() >= NB_FEATURES);
        // SAFETY: sizes checked.
        unsafe { oracle_pc_queue_features(self.0, features.as_ptr()) }
    }

    /// State dump: ([`PLC_STATE_LEN`] floats, [`PLC_STATE_INTS`] ints), see the constants.
    #[must_use]
    pub fn state(&self) -> (Vec<f32>, Vec<i32>) {
        let mut f = vec![0f32; PLC_STATE_LEN];
        let mut i = vec![0 as c_int; PLC_STATE_INTS];
        // SAFETY: buffers sized to the C dump.
        let n = unsafe { oracle_pc_plc_state(self.0, f.as_mut_ptr(), i.as_mut_ptr()) };
        assert_eq!(usize::try_from(n).expect("count"), PLC_STATE_LEN);
        (f, i)
    }
}

impl Drop for Plc {
    fn drop(&mut self) {
        // SAFETY: allocated by oracle_pc_plc_new.
        unsafe { oracle_pc_plc_free(self.0) }
    }
}
