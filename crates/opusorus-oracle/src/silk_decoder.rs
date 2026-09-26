//! Oracle bindings for unit `silk_decoder`: the SILK decoder (`silk/dec_API.c` and everything
//! it calls), see `csrc/silk_decoder.c`.
//!
//! [`SilkDec`] owns a C `silk_decoder` plus a range decoder over a private copy of the current
//! payload, so tests can drive `silk_Decode` call-by-call exactly like `src/opus_decoder.c`.

use core::ffi::{c_int, c_short, c_uint, c_void};
use core::ptr::NonNull;

unsafe extern "C" {
    fn oracle_sd_new() -> *mut c_void;
    fn oracle_sd_free(p: *mut c_void);
    fn oracle_sd_size() -> c_int;
    fn oracle_sd_init(p: *mut c_void) -> c_int;
    fn oracle_sd_reset(p: *mut c_void) -> c_int;
    fn oracle_sd_ec_init(p: *mut c_void, data: *const u8, len: c_int);
    fn oracle_sd_ec_state(p: *mut c_void, out: *mut c_uint);
    fn oracle_sd_decode(
        p: *mut c_void,
        ctrl: *mut c_int,
        lost: c_int,
        new_packet: c_int,
        out: *mut f32,
        n_out: *mut c_int,
    ) -> c_int;
    fn oracle_sd_dump(p: *mut c_void, out: *mut c_int, cap: c_int) -> c_int;
    fn oracle_sd_stereo_MS_to_LR(
        state: *mut c_short,
        x1: *mut c_short,
        x2: *mut c_short,
        pred: *const c_int,
        fs_khz: c_int,
        frame_length: c_int,
    );
    fn oracle_sd_decode_indices(
        data: *const u8,
        len: c_int,
        fs_khz: c_int,
        nb_subfr: c_int,
        vad: c_int,
        decode_lbrr: c_int,
        cond: c_int,
        prev_type: c_int,
        prev_lag: c_int,
        out: *mut c_int,
        pulses: *mut c_short,
    );
}

/// Maximum payload size accepted by [`SilkDec::ec_init`].
pub const MAX_PAYLOAD: usize = 4096;

/// `silk_DecControlStruct` fields in C order: `nChannelsAPI`, `nChannelsInternal`,
/// `API_sampleRate`, `internalSampleRate`, `payloadSize_ms`, `prevPitchLag`,
/// `enable_deep_plc`.
pub type DecCtrl = [i32; 7];

/// Range decoder state: `storage`, `end_offs`, `end_window`, `nend_bits`, `nbits_total`,
/// `offs`, `rng`, `val`, `ext`, `rem`, `error`, `ec_tell_frac`.
pub type EcStateDump = [u32; 12];

/// C `silk_decoder` (initialized with `silk_InitDecoder` on zeroed memory) + a range decoder.
#[derive(Debug)]
pub struct SilkDec {
    ptr: NonNull<c_void>,
}

impl SilkDec {
    /// Allocates and initializes a decoder.
    pub fn new() -> Self {
        // SAFETY: plain allocation + init; returns NULL only on allocation failure.
        let p = unsafe { oracle_sd_new() };
        Self {
            ptr: NonNull::new(p).expect("oracle_sd_new: allocation failed"),
        }
    }
    /// `silk_InitDecoder`.
    pub fn init(&mut self) -> i32 {
        // SAFETY: valid handle.
        unsafe { oracle_sd_init(self.ptr.as_ptr()) }
    }
    /// `silk_ResetDecoder`.
    pub fn reset(&mut self) -> i32 {
        // SAFETY: valid handle.
        unsafe { oracle_sd_reset(self.ptr.as_ptr()) }
    }
    /// Starts decoding a new payload (`ec_dec_init` over a copy of `data`).
    pub fn ec_init(&mut self, data: &[u8]) {
        assert!(data.len() <= MAX_PAYLOAD);
        // SAFETY: valid handle; data valid for len bytes (copied by the shim).
        unsafe { oracle_sd_ec_init(self.ptr.as_ptr(), data.as_ptr(), data.len() as c_int) }
    }
    /// Current range decoder state.
    pub fn ec_state(&mut self) -> EcStateDump {
        let mut out = [0u32; 12];
        // SAFETY: valid handle; out has 12 entries.
        unsafe { oracle_sd_ec_state(self.ptr.as_ptr(), out.as_mut_ptr()) };
        out
    }
    /// `silk_Decode`. `out` must hold at least `20 ms * API rate * nChannelsAPI` samples.
    /// Returns `(ret, nSamplesOut)`.
    pub fn decode(
        &mut self,
        ctrl: &mut DecCtrl,
        lost: i32,
        new_packet: i32,
        out: &mut [f32],
    ) -> (i32, i32) {
        assert!(out.len() >= (ctrl[2].max(0) as usize / 50) * ctrl[0].clamp(1, 2) as usize);
        let mut n = 0;
        // SAFETY: valid handle; ctrl has 7 entries; out is large enough for one 20 ms frame
        // at the API rate (the most silk_Decode writes).
        let r = unsafe {
            oracle_sd_decode(
                self.ptr.as_ptr(),
                ctrl.as_mut_ptr(),
                lost,
                new_packet,
                out.as_mut_ptr(),
                &mut n,
            )
        };
        (r, n)
    }
    /// Full state dump (see `oracle_sd_dump`).
    pub fn dump(&mut self) -> Vec<i32> {
        // SAFETY: valid handle; cap 0 only counts.
        let n = unsafe { oracle_sd_dump(self.ptr.as_ptr(), core::ptr::null_mut(), 0) };
        let mut v = vec![0i32; n as usize];
        // SAFETY: v has n entries.
        unsafe { oracle_sd_dump(self.ptr.as_ptr(), v.as_mut_ptr(), n) };
        v
    }
}

impl Default for SilkDec {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SilkDec {
    fn drop(&mut self) {
        // SAFETY: handle came from oracle_sd_new and is freed once.
        unsafe { oracle_sd_free(self.ptr.as_ptr()) }
    }
}

/// `silk_Get_Decoder_Size` of the C build.
pub fn decoder_size() -> i32 {
    // SAFETY: no arguments.
    unsafe { oracle_sd_size() }
}

/// `silk_stereo_MS_to_LR`. `state`: `pred_prev_Q13[2]`, `sMid[2]`, `sSide[2]`.
pub fn stereo_ms_to_lr(
    state: &mut [i16; 6],
    x1: &mut [i16],
    x2: &mut [i16],
    pred: &[i32; 2],
    fs_khz: i32,
    frame_length: usize,
) {
    assert!(x1.len() >= frame_length + 2 && x2.len() >= frame_length + 2);
    assert!(frame_length >= 8 * fs_khz as usize);
    // SAFETY: buffers valid for frame_length + 2 samples; state has 6 entries.
    unsafe {
        oracle_sd_stereo_MS_to_LR(
            state.as_mut_ptr(),
            x1.as_mut_ptr(),
            x2.as_mut_ptr(),
            pred.as_ptr(),
            fs_khz,
            frame_length as c_int,
        )
    }
}

/// `silk_decode_indices` + `silk_decode_pulses` on a fresh channel state (see the shim).
/// Returns the 37 output ints and the pulses (`MAX_FRAME_LENGTH` entries).
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the flat C shim signature"
)]
pub fn decode_indices(
    data: &[u8],
    fs_khz: i32,
    nb_subfr: i32,
    vad: i32,
    decode_lbrr: i32,
    cond: i32,
    prev_type: i32,
    prev_lag: i32,
) -> ([i32; 37], Vec<i16>) {
    assert!(matches!(fs_khz, 8 | 12 | 16) && matches!(nb_subfr, 2 | 4));
    let mut out = [0i32; 37];
    let mut pulses = vec![0i16; 320];
    // SAFETY: data valid for len; out has 37 entries; pulses has MAX_FRAME_LENGTH entries
    // (>= frame_length rounded up to 16).
    unsafe {
        oracle_sd_decode_indices(
            data.as_ptr(),
            data.len() as c_int,
            fs_khz,
            nb_subfr,
            vad,
            decode_lbrr,
            cond,
            prev_type,
            prev_lag,
            out.as_mut_ptr(),
            pulses.as_mut_ptr(),
        )
    };
    (out, pulses)
}
