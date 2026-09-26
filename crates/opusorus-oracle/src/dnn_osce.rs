//! Oracle bindings for unit `dnn_osce`: dnn/osce.c and osce_features.c (LACE, NoLACE and the
//! BBWENet bandwidth extension), via `csrc/dnn_osce.c`.
//!
//! Only available with the `osce` feature. Three kinds of wrappers:
//! * [`CapDecoder`]: a C SILK decoder (renamed copies of silk/dec_API.c + decode_frame.c) whose
//!   OSCE calls are logged with their inputs and outputs ([`Event`]), so tests can replay the
//!   exact OSCE call sequence of real SILK decoding.
//! * [`Model`], [`DecState`], [`BweState`]: C `OSCEModel`, `silk_decoder_state` (OSCE part) and
//!   `silk_OSCE_BWE_struct` handles for direct calls.
//! * Free functions for the static helpers of osce.c / osce_features.c.
//!
//! Every wrapper checks the sizes the C code will read or write before calling it.
#![cfg(feature = "osce")]

use core::ffi::{c_int, c_uint, c_void};
use core::ptr::NonNull;

/// Event kinds of [`Event::kind`].
pub const EV_INIT: i32 = 1;
/// `silk_reset_decoder(channel)`.
pub const EV_RESETDEC: i32 = 2;
/// `osce_reset(channel, method)`.
pub const EV_RESET: i32 = 3;
/// `osce_enhance_frame`.
pub const EV_ENHANCE: i32 = 4;
/// `osce_bwe_reset(channel)`.
pub const EV_BWE_RESET: i32 = 5;
/// `osce_bwe`.
pub const EV_BWE: i32 = 6;
/// `osce_bwe_cross_fade_10ms`.
pub const EV_BWE_XFADE: i32 = 7;

/// One logged OSCE call (C `oracle_osce_event`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub kind: c_int,
    /// Channel (0/1) or -1 (cross-fade).
    pub ch: c_int,
    /// `osce_reset` method, or the channel's `osce.method` before `osce_enhance_frame`.
    pub method: c_int,
    pub fs_khz: c_int,
    pub nb_subfr: c_int,
    pub lpc_order: c_int,
    pub signal_type: c_int,
    pub num_bits: c_int,
    /// Samples of `input`: `frame_length` (enhance), `xq16_len` (bwe), `length` (cross-fade).
    pub len: c_int,
    pub model_loaded: c_int,
    pub pitch_l: [c_int; 4],
    pub gains_q16: [c_int; 4],
    pub pred_coef_q12: [i16; 32],
    pub ltp_coef_q14: [i16; 20],
    /// Enhance: `xq` before; bwe: `xq16`; cross-fade: `x_fadein` before (480).
    pub input: [i16; 960],
    /// Cross-fade: `x_fadeout` (480).
    pub input2: [i16; 960],
    /// Enhance: `xq` after; bwe: `xq48` (`3*len`); cross-fade: `x_fadein` after (480).
    pub output: [i16; 960],
}

impl Default for Event {
    fn default() -> Self {
        Self {
            kind: 0,
            ch: 0,
            method: 0,
            fs_khz: 0,
            nb_subfr: 0,
            lpc_order: 0,
            signal_type: 0,
            num_bits: 0,
            len: 0,
            model_loaded: 0,
            pitch_l: [0; 4],
            gains_q16: [0; 4],
            pred_coef_q12: [0; 32],
            ltp_coef_q14: [0; 20],
            input: [0; 960],
            input2: [0; 960],
            output: [0; 960],
        }
    }
}

/// Maximum payload accepted by [`CapDecoder::ec_init`].
pub const MAX_PAYLOAD: usize = 4096;

unsafe extern "C" {
    fn oracle_osce_event_size() -> c_int;
    fn oracle_osce_cap_new() -> *mut c_void;
    fn oracle_osce_cap_free(p: *mut c_void);
    fn oracle_osce_cap_load(p: *mut c_void, data: *const u8, len: c_int) -> c_int;
    fn oracle_osce_cap_loaded(p: *mut c_void) -> c_int;
    fn oracle_osce_cap_set_loaded(p: *mut c_void, v: c_int);
    fn oracle_osce_cap_reset(p: *mut c_void) -> c_int;
    fn oracle_osce_cap_init(p: *mut c_void) -> c_int;
    fn oracle_osce_cap_ec_init(p: *mut c_void, data: *const u8, len: c_int);
    fn oracle_osce_cap_decode(
        p: *mut c_void,
        cfg: *const c_int,
        lost: c_int,
        new_packet: c_int,
        n_out: *mut c_int,
    ) -> c_int;
    fn oracle_osce_cap_prev_ext_mode(p: *mut c_void) -> c_int;
    fn oracle_osce_cap_set_prev_ext_mode(p: *mut c_void, v: c_int);
    fn oracle_osce_cap_nevents(p: *mut c_void) -> c_int;
    fn oracle_osce_cap_event(p: *mut c_void, i: c_int, out: *mut Event);
    fn oracle_osce_cap_dump_osce(p: *mut c_void, ch: c_int, out: *mut c_uint) -> c_int;
    fn oracle_osce_cap_dump_bwe(p: *mut c_void, ch: c_int, out: *mut c_uint) -> c_int;

    fn oracle_osce_model_new(data: *const u8, len: c_int, ret: *mut c_int) -> *mut c_void;
    fn oracle_osce_model_free(p: *mut c_void);
    fn oracle_osce_model_set_loaded(p: *mut c_void, v: c_int);
    fn oracle_osce_model_window(p: *mut c_void, which: c_int, out: *mut f32) -> c_int;

    fn oracle_osce_dec_new() -> *mut c_void;
    fn oracle_osce_dec_free(p: *mut c_void);
    fn oracle_osce_dec_reset(p: *mut c_void, method: c_int);
    fn oracle_osce_dec_features(
        p: *mut c_void,
        info: *const c_int,
        pitch_l: *const c_int,
        gains: *const c_int,
        pred: *const i16,
        ltp: *const i16,
        xq: *const i16,
        num_bits: c_int,
        features: *mut f32,
        numbits: *mut f32,
        periods: *mut c_int,
    );
    fn oracle_osce_dec_enhance(
        p: *mut c_void,
        model: *mut c_void,
        info: *const c_int,
        pitch_l: *const c_int,
        gains: *const c_int,
        pred: *const i16,
        ltp: *const i16,
        xq: *mut i16,
        num_bits: c_int,
    );
    fn oracle_osce_dec_dump(p: *mut c_void, out: *mut c_uint) -> c_int;
    fn oracle_osce_dec_pitch_postprocessing(p: *mut c_void, lag: c_int, type_: c_int) -> c_int;
    fn oracle_osce_lace_feature_net(
        model: *mut c_void,
        p: *mut c_void,
        out: *mut f32,
        features: *const f32,
        numbits: *const f32,
        periods: *const c_int,
    );
    fn oracle_osce_lace_frame(
        model: *mut c_void,
        p: *mut c_void,
        x_out: *mut f32,
        x_in: *const f32,
        features: *const f32,
        numbits: *const f32,
        periods: *const c_int,
    );
    fn oracle_osce_nolace_feature_net(
        model: *mut c_void,
        p: *mut c_void,
        out: *mut f32,
        features: *const f32,
        numbits: *const f32,
        periods: *const c_int,
    );
    fn oracle_osce_nolace_frame(
        model: *mut c_void,
        p: *mut c_void,
        x_out: *mut f32,
        x_in: *const f32,
        features: *const f32,
        numbits: *const f32,
        periods: *const c_int,
    );

    fn oracle_osce_bwe_new() -> *mut c_void;
    fn oracle_osce_bwe_free(p: *mut c_void);
    fn oracle_osce_bwe_reset(p: *mut c_void);
    fn oracle_osce_bwe_features(p: *mut c_void, features: *mut f32, xq: *const i16, n: c_int);
    fn oracle_osce_bwe_run(
        model: *mut c_void,
        p: *mut c_void,
        xq48: *mut i16,
        xq16: *mut i16,
        len: c_int,
    );
    fn oracle_osce_bwe_process_frames(
        model: *mut c_void,
        p: *mut c_void,
        x_out: *mut f32,
        x_in: *const f32,
        features: *const f32,
        num_frames: c_int,
    );
    fn oracle_osce_bwe_feature_net(
        model: *mut c_void,
        p: *mut c_void,
        out: *mut f32,
        features: *const f32,
        num_frames: c_int,
    );
    fn oracle_osce_bwe_dump(p: *mut c_void, out: *mut c_uint) -> c_int;

    fn oracle_osce_numbits_embedding(
        nolace: c_int,
        emb: *mut f32,
        numbits: f32,
        min_val: f32,
        max_val: f32,
        logscale: c_int,
    );
    fn oracle_osce_apply_filterbank(bank: c_int, x_out: *mut f32, x_in: *const f32);
    fn oracle_osce_mag_spec(out: *mut f32, input: *const f32);
    fn oracle_osce_log_spectrum_from_lpc(spec: *mut f32, a_q12: *const i16, lpc_order: c_int);
    fn oracle_osce_cepstrum(cepstrum: *mut f32, signal: *const f32);
    fn oracle_osce_acorr(acorr: *mut f32, buf: *const f32, pos: c_int, lag: c_int);
    fn oracle_osce_upsamp_2x(st: *mut f32, x_out: *mut f32, x_in: *const f32, n: c_int);
    fn oracle_osce_interpol_3_2(st: *mut f32, x_out: *mut f32, x_in: *const f32, n: c_int);
    fn oracle_osce_resamp_state_size() -> c_int;
    fn oracle_osce_valin_activation(x: *mut f32, len: c_int);
    fn oracle_osce_cross_fade_10ms(x_enhanced: *mut f32, x_in: *const f32, length: c_int);
    fn oracle_osce_bwe_cross_fade_10ms(x_fadein: *mut i16, x_fadeout: *const i16, length: c_int);
}

/// Checks that the Rust [`Event`] mirror matches the C layout size.
#[must_use]
pub fn event_layout_ok() -> bool {
    // SAFETY: pure function.
    let c = unsafe { oracle_osce_event_size() };
    c as usize == size_of::<Event>()
}

fn dump_with(f: impl Fn(*mut c_uint) -> c_int) -> Vec<u32> {
    let n = f(core::ptr::null_mut());
    let mut v = vec![0u32; usize::try_from(n).expect("non-negative count")];
    let m = f(v.as_mut_ptr());
    assert_eq!(n, m);
    v
}

/// SILK decoder control of [`CapDecoder::decode`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CapCtrl {
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub api_sample_rate: i32,
    pub internal_sample_rate: i32,
    pub payload_size_ms: i32,
    pub enable_deep_plc: i32,
    pub osce_method: i32,
    pub enable_osce_bwe: i32,
    pub osce_extended_mode: i32,
}

/// C SILK decoder (`silk_decoder`, created with `silk_InitDecoder`, which loads the compiled-in
/// OSCE models) whose OSCE calls are logged.
#[derive(Debug)]
pub struct CapDecoder {
    ptr: NonNull<c_void>,
    init_events: Vec<Event>,
}

impl CapDecoder {
    /// Allocates and initializes a decoder; the `silk_InitDecoder` events are kept in
    /// [`Self::take_init_events`].
    #[must_use]
    pub fn new() -> Self {
        assert!(event_layout_ok(), "oracle_osce_event layout mismatch");
        // SAFETY: plain allocation + init; NULL only on allocation failure.
        let p = unsafe { oracle_osce_cap_new() };
        let mut s = Self {
            ptr: NonNull::new(p).expect("oracle_osce_cap_new: allocation failed"),
            init_events: Vec::new(),
        };
        s.init_events = s.events();
        s
    }
    /// Events logged by `silk_InitDecoder` in [`Self::new`].
    pub fn take_init_events(&mut self) -> Vec<Event> {
        core::mem::take(&mut self.init_events)
    }
    /// `silk_LoadOSCEModels(data)` (`None`: the compiled-in tables).
    pub fn load(&mut self, data: Option<&[u8]>) -> i32 {
        let (p, n) = match data {
            Some(d) => (d.as_ptr(), c_int::try_from(d.len()).expect("blob size")),
            None => (core::ptr::null(), 0),
        };
        // SAFETY: valid handle; data valid for n bytes (or NULL/0).
        unsafe { oracle_osce_cap_load(self.ptr.as_ptr(), p, n) }
    }
    /// `osce_model.loaded`.
    pub fn loaded(&mut self) -> bool {
        // SAFETY: valid handle.
        unsafe { oracle_osce_cap_loaded(self.ptr.as_ptr()) != 0 }
    }
    /// Sets `osce_model.loaded`.
    pub fn set_loaded(&mut self, v: bool) {
        // SAFETY: valid handle.
        unsafe { oracle_osce_cap_set_loaded(self.ptr.as_ptr(), c_int::from(v)) }
    }
    /// `silk_ResetDecoder` (events via [`Self::events`]).
    pub fn reset(&mut self) -> i32 {
        // SAFETY: valid handle.
        unsafe { oracle_osce_cap_reset(self.ptr.as_ptr()) }
    }
    /// `silk_InitDecoder` (events via [`Self::events`]).
    pub fn init(&mut self) -> i32 {
        // SAFETY: valid handle.
        unsafe { oracle_osce_cap_init(self.ptr.as_ptr()) }
    }
    /// Starts decoding a new payload (`ec_dec_init` over a copy of `data`).
    pub fn ec_init(&mut self, data: &[u8]) {
        assert!(data.len() <= MAX_PAYLOAD);
        // SAFETY: valid handle; data valid for len bytes (copied by the shim).
        unsafe { oracle_osce_cap_ec_init(self.ptr.as_ptr(), data.as_ptr(), data.len() as c_int) }
    }
    /// `silk_Decode` (output samples are discarded). Returns `(ret, nSamplesOut)`; the OSCE
    /// calls it made are in [`Self::events`].
    pub fn decode(&mut self, ctrl: &CapCtrl, lost: i32, new_packet: i32) -> (i32, i32) {
        assert!((1..=2).contains(&ctrl.n_channels_api));
        assert!((1..=2).contains(&ctrl.n_channels_internal));
        assert!(ctrl.api_sample_rate <= 48000);
        let cfg = [
            ctrl.n_channels_api,
            ctrl.n_channels_internal,
            ctrl.api_sample_rate,
            ctrl.internal_sample_rate,
            ctrl.payload_size_ms,
            ctrl.enable_deep_plc,
            ctrl.osce_method,
            ctrl.enable_osce_bwe,
            ctrl.osce_extended_mode,
        ];
        let mut n = 0;
        // SAFETY: valid handle; cfg has 9 entries; the handle owns the output buffer.
        let r = unsafe {
            oracle_osce_cap_decode(self.ptr.as_ptr(), cfg.as_ptr(), lost, new_packet, &mut n)
        };
        (r, n)
    }
    /// `DecControl.prev_osce_extended_mode`.
    pub fn prev_ext_mode(&mut self) -> i32 {
        // SAFETY: valid handle.
        unsafe { oracle_osce_cap_prev_ext_mode(self.ptr.as_ptr()) }
    }
    /// Sets `DecControl.prev_osce_extended_mode` (the Opus decoder does so on a CELT->SILK
    /// transition).
    pub fn set_prev_ext_mode(&mut self, v: i32) {
        // SAFETY: valid handle.
        unsafe { oracle_osce_cap_set_prev_ext_mode(self.ptr.as_ptr(), v) }
    }
    /// The events logged by the last call.
    pub fn events(&mut self) -> Vec<Event> {
        // SAFETY: valid handle.
        let n = unsafe { oracle_osce_cap_nevents(self.ptr.as_ptr()) };
        assert!(n >= 0, "OSCE event log overflow");
        (0..n)
            .map(|i| {
                let mut e = Event::default();
                // SAFETY: valid handle; i < n; e is a matching repr(C) struct.
                unsafe { oracle_osce_cap_event(self.ptr.as_ptr(), i, &mut e) };
                e
            })
            .collect()
    }
    /// Dump of `channel_state[ch].osce` (see `oc_dump_osce`).
    pub fn dump_osce(&mut self, ch: usize) -> Vec<u32> {
        assert!(ch < 2);
        let p = self.ptr.as_ptr();
        // SAFETY: valid handle; ch < 2; out NULL or sized by the counting call.
        dump_with(|o| unsafe { oracle_osce_cap_dump_osce(p, ch as c_int, o) })
    }
    /// Dump of `channel_state[ch].osce_bwe` (see `oc_dump_bwe`).
    pub fn dump_bwe(&mut self, ch: usize) -> Vec<u32> {
        assert!(ch < 2);
        let p = self.ptr.as_ptr();
        // SAFETY: valid handle; ch < 2; out NULL or sized by the counting call.
        dump_with(|o| unsafe { oracle_osce_cap_dump_bwe(p, ch as c_int, o) })
    }
}

impl Default for CapDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CapDecoder {
    fn drop(&mut self) {
        // SAFETY: handle from oracle_osce_cap_new, freed once.
        unsafe { oracle_osce_cap_free(self.ptr.as_ptr()) }
    }
}

/// C `OSCEModel` loaded with `osce_load_models`.
#[derive(Debug)]
pub struct Model {
    ptr: NonNull<c_void>,
}

impl Model {
    /// `osce_load_models(data)` (`None`: compiled-in tables); returns the model and the C
    /// result; `loaded` is set to `result == 0`.
    #[must_use]
    pub fn new(data: Option<&[u8]>) -> (Self, i32) {
        let (p, n) = match data {
            Some(d) => (d.as_ptr(), c_int::try_from(d.len()).expect("blob size")),
            None => (core::ptr::null(), 0),
        };
        let mut ret = 0;
        // SAFETY: data valid for n bytes (or NULL/0); ret is a valid out pointer.
        let m = unsafe { oracle_osce_model_new(p, n, &mut ret) };
        (
            Self {
                ptr: NonNull::new(m).expect("oracle_osce_model_new: allocation failed"),
            },
            ret,
        )
    }
    /// Sets `loaded`.
    pub fn set_loaded(&mut self, v: bool) {
        // SAFETY: valid handle.
        unsafe { oracle_osce_model_set_loaded(self.ptr.as_ptr(), c_int::from(v)) }
    }
    /// Overlap window: 0 LACE, 1 NoLACE, 2/3/4 BBWENet 16/32/48 kHz.
    pub fn window(&mut self, which: i32) -> Vec<f32> {
        let mut out = [0f32; 120];
        // SAFETY: valid handle; out holds the largest window (120).
        let n = unsafe { oracle_osce_model_window(self.ptr.as_ptr(), which, out.as_mut_ptr()) };
        out[..n as usize].to_vec()
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: handle from oracle_osce_model_new, freed once.
        unsafe { oracle_osce_model_free(self.ptr.as_ptr()) }
    }
}

/// The decoder-state fields OSCE reads: `[fs_kHz, nb_subfr, LPC_order, signalType]`.
pub type DecInfo = [i32; 4];

/// The decoder-control fields OSCE reads.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ctrl {
    pub pitch_l: [i32; 4],
    pub gains_q16: [i32; 4],
    /// `PredCoef_Q12[0]` then `PredCoef_Q12[1]` (16 each).
    pub pred_coef_q12: [i16; 32],
    pub ltp_coef_q14: [i16; 20],
}

/// Frame length implied by a [`DecInfo`].
fn frame_len(info: &DecInfo) -> usize {
    assert!(info[0] > 0 && info[1] > 0 && info[1] <= 4 && info[2] <= 16);
    (info[1] * 5 * info[0]) as usize
}

/// C `silk_decoder_state` used for its `osce` member.
#[derive(Debug)]
pub struct DecState {
    ptr: NonNull<c_void>,
}

impl DecState {
    /// Zeroed state (as after `silk_init_decoder`'s memset, before `osce_reset`).
    #[must_use]
    pub fn new() -> Self {
        // SAFETY: plain allocation.
        let p = unsafe { oracle_osce_dec_new() };
        Self {
            ptr: NonNull::new(p).expect("oracle_osce_dec_new: allocation failed"),
        }
    }
    /// `osce_reset(&psDec->osce, method)`.
    pub fn reset(&mut self, method: i32) {
        assert!((0..=2).contains(&method));
        // SAFETY: valid handle.
        unsafe { oracle_osce_dec_reset(self.ptr.as_ptr(), method) }
    }
    /// `osce_calculate_features`: returns `(features[nb_subfr*93], numbits, periods)`.
    pub fn features(
        &mut self,
        info: &DecInfo,
        ctrl: &Ctrl,
        xq: &[i16],
        num_bits: i32,
    ) -> (Vec<f32>, [f32; 2], [i32; 4]) {
        let len = frame_len(info);
        assert!(info[0] == 16 && xq.len() >= len);
        let mut features = vec![0f32; 4 * 93];
        let mut numbits = [0f32; 2];
        let mut periods = [0i32; 4];
        // SAFETY: valid handle; arrays sized as C reads/writes them (xq: nb_subfr*80 samples).
        unsafe {
            oracle_osce_dec_features(
                self.ptr.as_ptr(),
                info.as_ptr(),
                ctrl.pitch_l.as_ptr(),
                ctrl.gains_q16.as_ptr(),
                ctrl.pred_coef_q12.as_ptr(),
                ctrl.ltp_coef_q14.as_ptr(),
                xq.as_ptr(),
                num_bits,
                features.as_mut_ptr(),
                numbits.as_mut_ptr(),
                periods.as_mut_ptr(),
            );
        }
        features.truncate(info[1] as usize * 93);
        (features, numbits, periods)
    }
    /// `osce_enhance_frame` on `xq` (frame_length samples) in place.
    pub fn enhance(
        &mut self,
        model: &mut Model,
        info: &DecInfo,
        ctrl: &Ctrl,
        xq: &mut [i16],
        num_bits: i32,
    ) {
        let len = frame_len(info);
        assert!(xq.len() >= len.max(320));
        // SAFETY: valid handles; xq holds at least 320 samples (the most C touches).
        unsafe {
            oracle_osce_dec_enhance(
                self.ptr.as_ptr(),
                model.ptr.as_ptr(),
                info.as_ptr(),
                ctrl.pitch_l.as_ptr(),
                ctrl.gains_q16.as_ptr(),
                ctrl.pred_coef_q12.as_ptr(),
                ctrl.ltp_coef_q14.as_ptr(),
                xq.as_mut_ptr(),
                num_bits,
            );
        }
    }
    /// Dump of `osce` (see `oc_dump_osce`).
    pub fn dump(&mut self) -> Vec<u32> {
        let p = self.ptr.as_ptr();
        // SAFETY: valid handle; out NULL or sized by the counting call.
        dump_with(|o| unsafe { oracle_osce_dec_dump(p, o) })
    }
    /// `pitch_postprocessing(&osce.features, lag, type)`.
    pub fn pitch_postprocessing(&mut self, lag: i32, type_: i32) -> i32 {
        // SAFETY: valid handle.
        unsafe { oracle_osce_dec_pitch_postprocessing(self.ptr.as_ptr(), lag, type_) }
    }
    fn check_net_args(features: &[f32], periods: &[i32; 4]) {
        assert!(features.len() >= 4 * 93);
        assert!(periods.iter().all(|&p| (0..=300).contains(&p)));
    }
    /// `lace_feature_net` on `osce.state.lace`: returns `4*128` floats.
    pub fn lace_feature_net(
        &mut self,
        model: &mut Model,
        features: &[f32],
        numbits: &[f32; 2],
        periods: &[i32; 4],
    ) -> Vec<f32> {
        Self::check_net_args(features, periods);
        let mut out = vec![0f32; 4 * 128];
        // SAFETY: valid handles; sizes checked above.
        unsafe {
            oracle_osce_lace_feature_net(
                model.ptr.as_ptr(),
                self.ptr.as_ptr(),
                out.as_mut_ptr(),
                features.as_ptr(),
                numbits.as_ptr(),
                periods.as_ptr(),
            );
        }
        out
    }
    /// `nolace_feature_net` on `osce.state.nolace`: returns `4*160` floats.
    pub fn nolace_feature_net(
        &mut self,
        model: &mut Model,
        features: &[f32],
        numbits: &[f32; 2],
        periods: &[i32; 4],
    ) -> Vec<f32> {
        Self::check_net_args(features, periods);
        let mut out = vec![0f32; 4 * 160];
        // SAFETY: valid handles; sizes checked above.
        unsafe {
            oracle_osce_nolace_feature_net(
                model.ptr.as_ptr(),
                self.ptr.as_ptr(),
                out.as_mut_ptr(),
                features.as_ptr(),
                numbits.as_ptr(),
                periods.as_ptr(),
            );
        }
        out
    }
    /// `lace_process_20ms_frame` (`nolace == false`) or `nolace_process_20ms_frame` on 320
    /// samples; periods must be >= 32 (adacomb lag bound, see the dnn_core notes).
    pub fn process_frame(
        &mut self,
        nolace: bool,
        model: &mut Model,
        x_in: &[f32],
        features: &[f32],
        numbits: &[f32; 2],
        periods: &[i32; 4],
    ) -> Vec<f32> {
        Self::check_net_args(features, periods);
        assert!(x_in.len() >= 320);
        let mut out = vec![0f32; 320];
        let f = if nolace {
            oracle_osce_nolace_frame
        } else {
            oracle_osce_lace_frame
        };
        // SAFETY: valid handles; sizes checked above.
        unsafe {
            f(
                model.ptr.as_ptr(),
                self.ptr.as_ptr(),
                out.as_mut_ptr(),
                x_in.as_ptr(),
                features.as_ptr(),
                numbits.as_ptr(),
                periods.as_ptr(),
            );
        }
        out
    }
}

impl Default for DecState {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DecState {
    fn drop(&mut self) {
        // SAFETY: handle from oracle_osce_dec_new, freed once.
        unsafe { oracle_osce_dec_free(self.ptr.as_ptr()) }
    }
}

/// C `silk_OSCE_BWE_struct` (zeroed on creation).
#[derive(Debug)]
pub struct BweState {
    ptr: NonNull<c_void>,
}

impl BweState {
    /// Zeroed state.
    #[must_use]
    pub fn new() -> Self {
        // SAFETY: plain allocation.
        let p = unsafe { oracle_osce_bwe_new() };
        Self {
            ptr: NonNull::new(p).expect("oracle_osce_bwe_new: allocation failed"),
        }
    }
    /// `osce_bwe_reset`.
    pub fn reset(&mut self) {
        // SAFETY: valid handle.
        unsafe { oracle_osce_bwe_reset(self.ptr.as_ptr()) }
    }
    /// `osce_bwe_calculate_features` on `xq` (160 or 320 samples).
    pub fn features(&mut self, xq: &[i16]) -> Vec<f32> {
        assert!(xq.len() == 160 || xq.len() == 320);
        let mut f = vec![0f32; 2 * 114];
        // SAFETY: valid handle; features holds (len/160)*114 floats.
        unsafe {
            oracle_osce_bwe_features(
                self.ptr.as_ptr(),
                f.as_mut_ptr(),
                xq.as_ptr(),
                xq.len() as c_int,
            );
        }
        f.truncate(xq.len() / 160 * 114);
        f
    }
    /// `osce_bwe`: returns `3*len` samples.
    pub fn run(&mut self, model: &mut Model, xq16: &[i16]) -> Vec<i16> {
        assert!(xq16.len() == 160 || xq16.len() == 320);
        let mut input = xq16.to_vec();
        let mut out = vec![0i16; 3 * xq16.len()];
        // SAFETY: valid handles; out holds 3*len samples.
        unsafe {
            oracle_osce_bwe_run(
                model.ptr.as_ptr(),
                self.ptr.as_ptr(),
                out.as_mut_ptr(),
                input.as_mut_ptr(),
                input.len() as c_int,
            );
        }
        out
    }
    /// `bbwenet_process_frames`: returns `num_frames*480` samples.
    pub fn process_frames(
        &mut self,
        model: &mut Model,
        x_in: &[f32],
        features: &[f32],
        num_frames: usize,
    ) -> Vec<f32> {
        assert!((1..=2).contains(&num_frames));
        assert!(x_in.len() >= 160 * num_frames && features.len() >= 114 * num_frames);
        let mut out = vec![0f32; 480 * num_frames];
        // SAFETY: valid handles; sizes checked above.
        unsafe {
            oracle_osce_bwe_process_frames(
                model.ptr.as_ptr(),
                self.ptr.as_ptr(),
                out.as_mut_ptr(),
                x_in.as_ptr(),
                features.as_ptr(),
                num_frames as c_int,
            );
        }
        out
    }
    /// `bbwe_feature_net`: returns `2*num_frames*128` floats.
    pub fn feature_net(
        &mut self,
        model: &mut Model,
        features: &[f32],
        num_frames: usize,
    ) -> Vec<f32> {
        assert!((1..=2).contains(&num_frames));
        assert!(features.len() >= 114 * num_frames);
        let mut out = vec![0f32; 2 * 128 * num_frames];
        // SAFETY: valid handles; sizes checked above.
        unsafe {
            oracle_osce_bwe_feature_net(
                model.ptr.as_ptr(),
                self.ptr.as_ptr(),
                out.as_mut_ptr(),
                features.as_ptr(),
                num_frames as c_int,
            );
        }
        out
    }
    /// Dump (see `oc_dump_bwe`).
    pub fn dump(&mut self) -> Vec<u32> {
        let p = self.ptr.as_ptr();
        // SAFETY: valid handle; out NULL or sized by the counting call.
        dump_with(|o| unsafe { oracle_osce_bwe_dump(p, o) })
    }
}

impl Default for BweState {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BweState {
    fn drop(&mut self) {
        // SAFETY: handle from oracle_osce_bwe_new, freed once.
        unsafe { oracle_osce_bwe_free(self.ptr.as_ptr()) }
    }
}

/// `compute_lace_numbits_embedding` / `compute_nolace_numbits_embedding`: 8 values.
#[must_use]
pub fn numbits_embedding(
    nolace: bool,
    numbits: f32,
    min_val: f32,
    max_val: f32,
    logscale: bool,
) -> [f32; 8] {
    let mut emb = [0f32; 8];
    // SAFETY: emb has 8 entries.
    unsafe {
        oracle_osce_numbits_embedding(
            c_int::from(nolace),
            emb.as_mut_ptr(),
            numbits,
            min_val,
            max_val,
            c_int::from(logscale),
        );
    }
    emb
}

/// `apply_filterbank` with bank 0 clean (64), 1 noisy (18), 2 bwe (32) on 161 bins.
#[must_use]
pub fn apply_filterbank(bank: i32, x_in: &[f32]) -> Vec<f32> {
    assert!((0..=2).contains(&bank) && x_in.len() >= 161);
    let n = [64, 18, 32][bank as usize];
    let mut out = vec![0f32; 64];
    // SAFETY: x_in holds 161 floats (copied), out 64.
    unsafe { oracle_osce_apply_filterbank(bank, out.as_mut_ptr(), x_in.as_ptr()) };
    out.truncate(n);
    out
}

/// `mag_spec_320_onesided`: 161 magnitudes of 320 samples.
#[must_use]
pub fn mag_spec(input: &[f32]) -> Vec<f32> {
    assert!(input.len() >= 320);
    let mut out = vec![0f32; 161];
    // SAFETY: sizes as above (input copied).
    unsafe { oracle_osce_mag_spec(out.as_mut_ptr(), input.as_ptr()) };
    out
}

/// `calculate_log_spectrum_from_lpc`: 64 values.
#[must_use]
pub fn log_spectrum_from_lpc(a_q12: &[i16], lpc_order: usize) -> Vec<f32> {
    assert!(lpc_order <= 16 && a_q12.len() >= lpc_order);
    let mut out = vec![0f32; 64];
    // SAFETY: sizes as above.
    unsafe {
        oracle_osce_log_spectrum_from_lpc(out.as_mut_ptr(), a_q12.as_ptr(), lpc_order as c_int)
    };
    out
}

/// `calculate_cepstrum`: 18 values of 320 samples.
#[must_use]
pub fn cepstrum(signal: &[f32]) -> Vec<f32> {
    assert!(signal.len() >= 320);
    let mut out = vec![0f32; 18];
    // SAFETY: sizes as above (signal copied).
    unsafe { oracle_osce_cepstrum(out.as_mut_ptr(), signal.as_ptr()) };
    out
}

/// `calculate_acorr(acorr, buf + pos, lag)`: 5 values.
#[must_use]
pub fn acorr(buf: &[f32], pos: usize, lag: i32) -> [f32; 5] {
    assert!(pos + 80 <= buf.len() && lag >= 2 && pos as i64 - lag as i64 - 2 >= 0);
    let mut out = [0f32; 5];
    // SAFETY: C reads buf[pos-lag-2 .. pos+80+...] within bounds (checked above).
    unsafe { oracle_osce_acorr(out.as_mut_ptr(), buf.as_ptr(), pos as c_int, lag) };
    out
}

/// `resamp_state` as 14 floats (`upsamp_buffer[2][3]`, `interpol_buffer[8]`).
pub type ResampState = [f32; 14];

/// `upsamp_2x`: `2*len` samples.
pub fn upsamp_2x(st: &mut ResampState, x_in: &[f32]) -> Vec<f32> {
    // SAFETY: pure function.
    assert_eq!(unsafe { oracle_osce_resamp_state_size() }, 56);
    assert!(x_in.len() > 1 && x_in.len() < 320);
    let mut out = vec![0f32; 2 * x_in.len()];
    // SAFETY: st has 14 floats; sizes as above.
    unsafe {
        oracle_osce_upsamp_2x(
            st.as_mut_ptr(),
            out.as_mut_ptr(),
            x_in.as_ptr(),
            x_in.len() as c_int,
        )
    };
    out
}

/// `interpol_3_2`: `3*len/2` samples (`len` even).
pub fn interpol_3_2(st: &mut ResampState, x_in: &[f32]) -> Vec<f32> {
    assert!(x_in.len() > 1 && x_in.len() < 640 && x_in.len().is_multiple_of(2));
    let mut out = vec![0f32; 3 * x_in.len() / 2];
    // SAFETY: st has 14 floats; sizes as above.
    unsafe {
        oracle_osce_interpol_3_2(
            st.as_mut_ptr(),
            out.as_mut_ptr(),
            x_in.as_ptr(),
            x_in.len() as c_int,
        );
    }
    out
}

/// `apply_valin_activation` in place (`len <= 480`).
pub fn valin_activation(x: &mut [f32]) {
    assert!(x.len() <= 480);
    // SAFETY: x valid for len floats.
    unsafe { oracle_osce_valin_activation(x.as_mut_ptr(), x.len() as c_int) };
}

/// `osce_cross_fade_10ms` (first 160 samples).
pub fn cross_fade_10ms(x_enhanced: &mut [f32], x_in: &[f32]) {
    assert!(x_enhanced.len() >= 160 && x_in.len() >= 160);
    // SAFETY: sizes as above.
    unsafe {
        oracle_osce_cross_fade_10ms(
            x_enhanced.as_mut_ptr(),
            x_in.as_ptr(),
            x_enhanced.len() as c_int,
        )
    };
}

/// `osce_bwe_cross_fade_10ms` (first 480 samples).
pub fn bwe_cross_fade_10ms(x_fadein: &mut [i16], x_fadeout: &[i16]) {
    assert!(x_fadein.len() >= 480 && x_fadeout.len() >= 480);
    // SAFETY: sizes as above.
    unsafe {
        oracle_osce_bwe_cross_fade_10ms(
            x_fadein.as_mut_ptr(),
            x_fadeout.as_ptr(),
            x_fadein.len() as c_int,
        );
    }
}
