//! Oracle bindings for unit `celt_decoder` (celt/celt_decoder.c), in the float and in the
//! fixed-point oracle (`csrc/celt_decoder.c` is `// oracle-build: any`).
//!
//! The libopus types are mirrored by [`Res`], [`Sig`], [`Norm`], [`Glog`], [`Val16`] and
//! [`StateVal`] (float in the float build, integers in the fixed-point build).
//!
//! * [`CeltDec`]: a C `CELTDecoder` (library `celt_decoder_init` / `celt_decode_with_ec[_dred]`
//!   / `opus_custom_decoder_ctl`, or with `custom-modes` an `opus_custom_decoder_create`d one)
//!   with a full state dump.
//! * [`CeltEnc`]: a C `CELTEncoder` used to generate realistic packets.
//! * Direct shims for the static helpers `tf_decode`, `deemphasis`, `celt_synthesis` and
//!   `celt_plc_pitch_search` (reached through a renamed copy of celt_decoder.c).
//!
//! Errors are raw negative libopus codes.

#![allow(
    clippy::too_many_arguments,
    reason = "wrappers mirror the flat C shim signatures"
)]

use core::ffi::{c_int, c_void};
use core::ptr;

/// `opus_res`.
#[cfg(not(feature = "fixed-point"))]
pub type Res = f32;
/// `opus_res` (24-bit resolution).
#[cfg(feature = "fixed-res24")]
pub type Res = i32;
/// `opus_res` (16-bit resolution).
#[cfg(all(feature = "fixed-point", not(feature = "fixed-res24")))]
pub type Res = i16;
/// `celt_sig`.
#[cfg(not(feature = "fixed-point"))]
pub type Sig = f32;
/// `celt_sig`.
#[cfg(feature = "fixed-point")]
pub type Sig = i32;
/// `celt_norm`.
#[cfg(not(feature = "fixed-point"))]
pub type Norm = f32;
/// `celt_norm` (Q24).
#[cfg(feature = "fixed-point")]
pub type Norm = i32;
/// `celt_glog`.
#[cfg(not(feature = "fixed-point"))]
pub type Glog = f32;
/// `celt_glog` (Q24).
#[cfg(feature = "fixed-point")]
pub type Glog = i32;
/// `opus_val16`.
#[cfg(not(feature = "fixed-point"))]
pub type Val16 = f32;
/// `opus_val16`.
#[cfg(feature = "fixed-point")]
pub type Val16 = i16;
/// Element type of the value array of a [`CeltDecState`] dump (`cd_val` in the shim): every
/// field widened to `opus_int32` in the fixed-point build.
#[cfg(not(feature = "fixed-point"))]
pub type StateVal = f32;
/// Element type of the value array of a [`CeltDecState`] dump (`cd_val` in the shim): every
/// field widened to `opus_int32` in the fixed-point build.
#[cfg(feature = "fixed-point")]
pub type StateVal = i32;

unsafe extern "C" {
    fn oracle_cd_has_qext() -> c_int;
    fn oracle_cd_has_custom() -> c_int;
    fn oracle_cd_new(fs: c_int, channels: c_int, err: *mut c_int) -> *mut c_void;
    #[cfg(feature = "custom-modes")]
    fn oracle_cd_custom_new(
        fs: c_int,
        frame_size: c_int,
        channels: c_int,
        err: *mut c_int,
    ) -> *mut c_void;
    fn oracle_cd_free(p: *mut c_void);
    fn oracle_cd_ctl_set(p: *mut c_void, request: c_int, value: c_int) -> c_int;
    fn oracle_cd_ctl_get(p: *mut c_void, request: c_int, value: *mut c_int) -> c_int;
    fn oracle_cd_get_mode_ok(p: *mut c_void) -> c_int;
    fn oracle_cd_decode(
        p: *mut c_void,
        data: *const u8,
        len: c_int,
        pcm: *mut Res,
        frame_size: c_int,
        accum: c_int,
        qext: *const u8,
        qext_len: c_int,
    ) -> c_int;
    fn oracle_cd_decode_shared(
        p: *mut c_void,
        data: *const u8,
        len: c_int,
        pcm: *mut Res,
        frame_size: c_int,
        accum: c_int,
        fts: *const u32,
        vals: *mut u32,
        n: c_int,
        ec_out: *mut u32,
    ) -> c_int;
    #[cfg(feature = "custom-modes")]
    fn oracle_cd_custom_decode(
        p: *mut c_void,
        data: *const u8,
        len: c_int,
        pcm: *mut i16,
        frame_size: c_int,
    ) -> c_int;
    #[cfg(feature = "custom-modes")]
    fn oracle_cd_custom_decode24(
        p: *mut c_void,
        data: *const u8,
        len: c_int,
        pcm: *mut i32,
        frame_size: c_int,
    ) -> c_int;
    #[cfg(not(feature = "disable-float-api"))]
    #[cfg(feature = "custom-modes")]
    fn oracle_cd_custom_decode_float(
        p: *mut c_void,
        data: *const u8,
        len: c_int,
        pcm: *mut f32,
        frame_size: c_int,
    ) -> c_int;
    fn oracle_cd_state(p: *mut c_void, ints: *mut c_int, vals: *mut StateVal) -> c_int;
    fn oracle_cd_tf_decode(
        start: c_int,
        end: c_int,
        is_transient: c_int,
        tf_res: *mut c_int,
        lm: c_int,
        buf: *const u8,
        len: c_int,
        pre: c_int,
        ec_out: *mut u32,
    );
    fn oracle_cd_deemphasis(
        in0: *const Sig,
        in1: *const Sig,
        pcm: *mut Res,
        n: c_int,
        c: c_int,
        downsample: c_int,
        coef: *const Val16,
        mem: *mut Sig,
        accum: c_int,
    );
    fn oracle_cd_celt_synthesis(
        fs: c_int,
        x: *mut Norm,
        out0: *mut Sig,
        out1: *mut Sig,
        off: c_int,
        old_band_e: *mut Glog,
        start: c_int,
        eff_end: c_int,
        c: c_int,
        cc: c_int,
        is_transient: c_int,
        lm: c_int,
        downsample: c_int,
        silence: c_int,
        use_qext: c_int,
        qext_band_log_e: *mut Glog,
        qext_end: c_int,
    );
    fn oracle_cd_plc_pitch_search(fs: c_int, mem0: *mut Sig, mem1: *mut Sig, c: c_int) -> c_int;
    fn oracle_cd_enc_new(fs: c_int, channels: c_int, err: *mut c_int) -> *mut c_void;
    #[cfg(feature = "custom-modes")]
    fn oracle_cd_enc_custom_new(
        fs: c_int,
        frame_size: c_int,
        channels: c_int,
        err: *mut c_int,
    ) -> *mut c_void;
    fn oracle_cd_enc_free(p: *mut c_void);
    fn oracle_cd_enc_ctl(p: *mut c_void, request: c_int, value: c_int) -> c_int;
    fn oracle_cd_enc_encode(
        p: *mut c_void,
        pcm: *const f32,
        frame_size: c_int,
        out: *mut u8,
        nbytes: c_int,
    ) -> c_int;
    fn oracle_cd_enc_encode_shared(
        p: *mut c_void,
        pcm: *const f32,
        frame_size: c_int,
        out: *mut u8,
        nbytes: c_int,
        fts: *const u32,
        vals: *const u32,
        n: c_int,
    ) -> c_int;
}

/// Number of integers in a [`CeltDec::state`] dump.
pub const STATE_NINTS: usize = 22;

/// True when the oracle was built with `ENABLE_QEXT`.
pub fn has_qext() -> bool {
    // SAFETY: no arguments.
    unsafe { oracle_cd_has_qext() != 0 }
}

/// True when the oracle was built with `CUSTOM_MODES`.
pub fn has_custom() -> bool {
    // SAFETY: no arguments.
    unsafe { oracle_cd_has_custom() != 0 }
}

const fn opt_ptr(d: Option<&[u8]>) -> *const u8 {
    match d {
        Some(s) => s.as_ptr(),
        None => ptr::null(),
    }
}

/// Full decoder state dump (see `oracle_cd_state` in csrc/celt_decoder.c for the layout).
#[derive(Debug, Clone, PartialEq)]
pub struct CeltDecState {
    /// Integer fields.
    pub ints: [i32; STATE_NINTS],
    /// The other fields and arrays, in order (see [`StateVal`]).
    pub vals: Vec<StateVal>,
}

/// A C CELT decoder.
#[derive(Debug)]
pub struct CeltDec {
    ptr: *mut c_void,
    channels: usize,
}

impl CeltDec {
    /// `celt_decoder_init(st, fs, channels)`; `Err(code)` if it fails.
    pub fn new(fs: i32, channels: i32) -> Result<Self, i32> {
        let mut err = 0;
        // SAFETY: err is a valid out pointer.
        let ptr = unsafe { oracle_cd_new(fs, channels, &mut err) };
        assert!(!ptr.is_null(), "allocation failed");
        let d = Self {
            ptr,
            channels: channels as usize,
        };
        if err != 0 { Err(err) } else { Ok(d) }
    }

    /// `opus_custom_decoder_create(opus_custom_mode_create(fs, frame_size), channels)`.
    #[cfg(feature = "custom-modes")]
    pub fn new_custom(fs: i32, frame_size: i32, channels: i32) -> Result<Self, i32> {
        let mut err = 0;
        // SAFETY: err is a valid out pointer.
        let ptr = unsafe { oracle_cd_custom_new(fs, frame_size, channels, &mut err) };
        if ptr.is_null() {
            return Err(err);
        }
        Ok(Self {
            ptr,
            channels: channels as usize,
        })
    }

    /// `opus_custom_decoder_ctl(st, request, (opus_int32)value)` (value ignored for
    /// `OPUS_RESET_STATE`).
    pub fn ctl_set(&mut self, request: i32, value: i32) -> i32 {
        // SAFETY: handle valid; SET requests take an opus_int32 by value.
        unsafe { oracle_cd_ctl_set(self.ptr, request, value) }
    }

    /// `opus_custom_decoder_ctl(st, request, &value)` → `(ret, value)`.
    pub fn ctl_get(&mut self, request: i32) -> (i32, i32) {
        let mut v = 0;
        // SAFETY: handle valid; GET requests write one opus_int32/opus_uint32.
        let r = unsafe { oracle_cd_ctl_get(self.ptr, request, &mut v) };
        (r, v)
    }

    /// `CELT_GET_MODE` returns the decoder's mode.
    pub fn get_mode_ok(&mut self) -> bool {
        // SAFETY: handle valid.
        unsafe { oracle_cd_get_mode_ok(self.ptr) != 0 }
    }

    /// `celt_decode_with_ec_dred(st, data, len, pcm, frame_size, NULL, accum, qext...)`
    /// (`celt_decode_with_ec` without QEXT). `data` must hold `len` bytes when `Some`; `pcm`
    /// must hold `frame_size*channels` samples.
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        len: i32,
        pcm: &mut [Res],
        frame_size: i32,
        accum: bool,
        qext: Option<&[u8]>,
    ) -> i32 {
        if let Some(d) = data {
            assert!(len <= 0 || d.len() >= len as usize);
        }
        assert!(pcm.len() >= frame_size.max(0) as usize * self.channels);
        let ql = qext.map_or(0, |q| q.len() as c_int);
        // SAFETY: data/qext valid for the lengths passed (or NULL), pcm sized for the output.
        unsafe {
            oracle_cd_decode(
                self.ptr,
                opt_ptr(data),
                len,
                pcm.as_mut_ptr(),
                frame_size,
                c_int::from(accum),
                opt_ptr(qext),
                ql,
            )
        }
    }

    /// Hybrid-style decode (see `oracle_cd_decode_shared`): decodes `fts.len()` uniform symbols
    /// from a range decoder over `data`, then CELT from the same decoder. Returns
    /// `(ret, symbols, [rng, tell, tell_frac, error])`.
    pub fn decode_shared(
        &mut self,
        data: &[u8],
        pcm: &mut [Res],
        frame_size: i32,
        accum: bool,
        fts: &[u32],
    ) -> (i32, Vec<u32>, [u32; 4]) {
        assert!(pcm.len() >= frame_size.max(0) as usize * self.channels);
        let mut vals = vec![0u32; fts.len()];
        let mut ec = [0u32; 4];
        // SAFETY: all buffers valid for the lengths passed.
        let r = unsafe {
            oracle_cd_decode_shared(
                self.ptr,
                data.as_ptr(),
                data.len() as c_int,
                pcm.as_mut_ptr(),
                frame_size,
                c_int::from(accum),
                fts.as_ptr(),
                vals.as_mut_ptr(),
                fts.len() as c_int,
                ec.as_mut_ptr(),
            )
        };
        (r, vals, ec)
    }

    /// `opus_custom_decode` (`pcm`: `frame_size*channels`).
    #[cfg(feature = "custom-modes")]
    pub fn custom_decode(
        &mut self,
        data: Option<&[u8]>,
        len: i32,
        pcm: &mut [i16],
        frame_size: i32,
    ) -> i32 {
        assert!(pcm.len() >= frame_size.max(0) as usize * self.channels);
        // SAFETY: buffers valid for the lengths passed.
        unsafe {
            oracle_cd_custom_decode(self.ptr, opt_ptr(data), len, pcm.as_mut_ptr(), frame_size)
        }
    }

    /// `opus_custom_decode24`.
    #[cfg(feature = "custom-modes")]
    pub fn custom_decode24(
        &mut self,
        data: Option<&[u8]>,
        len: i32,
        pcm: &mut [i32],
        frame_size: i32,
    ) -> i32 {
        assert!(pcm.len() >= frame_size.max(0) as usize * self.channels);
        // SAFETY: buffers valid for the lengths passed.
        unsafe {
            oracle_cd_custom_decode24(self.ptr, opt_ptr(data), len, pcm.as_mut_ptr(), frame_size)
        }
    }

    #[cfg(not(feature = "disable-float-api"))]
    /// `opus_custom_decode_float`.
    #[cfg(feature = "custom-modes")]
    pub fn custom_decode_float(
        &mut self,
        data: Option<&[u8]>,
        len: i32,
        pcm: &mut [f32],
        frame_size: i32,
    ) -> i32 {
        assert!(pcm.len() >= frame_size.max(0) as usize * self.channels);
        // SAFETY: buffers valid for the lengths passed.
        unsafe {
            oracle_cd_custom_decode_float(
                self.ptr,
                opt_ptr(data),
                len,
                pcm.as_mut_ptr(),
                frame_size,
            )
        }
    }

    /// Full state dump.
    pub fn state(&mut self) -> CeltDecState {
        // SAFETY: querying the value count with NULL buffers.
        let n = unsafe { oracle_cd_state(self.ptr, ptr::null_mut(), ptr::null_mut()) };
        let mut ints = [0i32; STATE_NINTS];
        let mut vals = vec![StateVal::default(); n as usize];
        // SAFETY: ints has STATE_NINTS entries and vals the queried count.
        unsafe { oracle_cd_state(self.ptr, ints.as_mut_ptr(), vals.as_mut_ptr()) };
        CeltDecState { ints, vals }
    }
}

impl Drop for CeltDec {
    fn drop(&mut self) {
        // SAFETY: handle from oracle_cd_new/oracle_cd_custom_new, freed once.
        unsafe { oracle_cd_free(self.ptr) }
    }
}

/// A C CELT encoder (`celt_encoder_init`, signalling off), for packet generation.
#[derive(Debug)]
pub struct CeltEnc {
    ptr: *mut c_void,
    channels: usize,
}

impl CeltEnc {
    /// `celt_encoder_init(st, fs, channels)` + `CELT_SET_SIGNALLING(0)`.
    pub fn new(fs: i32, channels: i32) -> Self {
        let mut err = 0;
        // SAFETY: err is a valid out pointer.
        let ptr = unsafe { oracle_cd_enc_new(fs, channels, &mut err) };
        assert!(
            !ptr.is_null() && err == 0,
            "celt_encoder_init failed: {err}"
        );
        Self {
            ptr,
            channels: channels as usize,
        }
    }

    /// `opus_custom_encoder_create` for a custom mode (signalling on).
    #[cfg(feature = "custom-modes")]
    pub fn new_custom(fs: i32, frame_size: i32, channels: i32) -> Result<Self, i32> {
        let mut err = 0;
        // SAFETY: err is a valid out pointer.
        let ptr = unsafe { oracle_cd_enc_custom_new(fs, frame_size, channels, &mut err) };
        if ptr.is_null() {
            return Err(err);
        }
        Ok(Self {
            ptr,
            channels: channels as usize,
        })
    }

    /// `opus_custom_encoder_ctl(st, request, value)`.
    pub fn ctl(&mut self, request: i32, value: i32) -> i32 {
        // SAFETY: handle valid; request takes an opus_int32 by value.
        unsafe { oracle_cd_enc_ctl(self.ptr, request, value) }
    }

    /// `celt_encode_with_ec(st, pcm, frame_size, out+1, nbytes, NULL)` with the float input
    /// converted with `FLOAT2RES` (`FLOAT2INT16`/`FLOAT2INT24` in the fixed-point build). Returns
    /// the C return value and the buffer `out` (`out[0]` is a TOC placeholder the QEXT path may modify;
    /// the packet is `out[1..1+ret]`).
    pub fn encode(&mut self, pcm: &[f32], frame_size: i32, nbytes: i32) -> (i32, Vec<u8>) {
        assert!(pcm.len() >= frame_size.max(0) as usize * self.channels);
        let mut out = vec![0u8; nbytes as usize + 1];
        // SAFETY: pcm holds frame_size*channels samples and out nbytes+1 bytes.
        let r = unsafe {
            oracle_cd_enc_encode(self.ptr, pcm.as_ptr(), frame_size, out.as_mut_ptr(), nbytes)
        };
        (r, out)
    }

    /// Hybrid-style encode: `vals[i] < fts[i]` range coded first, then CELT into the same
    /// coder. Returns `(ret, out)`.
    pub fn encode_shared(
        &mut self,
        pcm: &[f32],
        frame_size: i32,
        nbytes: i32,
        fts: &[u32],
        vals: &[u32],
    ) -> (i32, Vec<u8>) {
        assert!(pcm.len() >= frame_size.max(0) as usize * self.channels);
        assert_eq!(fts.len(), vals.len());
        let mut out = vec![0u8; nbytes as usize];
        // SAFETY: buffers valid for the lengths passed.
        let r = unsafe {
            oracle_cd_enc_encode_shared(
                self.ptr,
                pcm.as_ptr(),
                frame_size,
                out.as_mut_ptr(),
                nbytes,
                fts.as_ptr(),
                vals.as_ptr(),
                fts.len() as c_int,
            )
        };
        (r, out)
    }
}

impl Drop for CeltEnc {
    fn drop(&mut self) {
        // SAFETY: handle from oracle_cd_enc_new/oracle_cd_enc_custom_new, freed once.
        unsafe { oracle_cd_enc_free(self.ptr) }
    }
}

/// `tf_decode` after `pre` `ec_dec_bit_logp(1)` reads from a decoder over `buf`. Writes
/// `tf_res[start..end]`; returns `[rng, tell]`.
pub fn tf_decode(
    start: i32,
    end: i32,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: i32,
    buf: &[u8],
    pre: i32,
) -> [u32; 2] {
    assert!(tf_res.len() >= end as usize);
    let mut ec = [0u32; 2];
    // SAFETY: tf_res holds `end` entries, buf its length, ec 2 entries.
    unsafe {
        oracle_cd_tf_decode(
            start,
            end,
            c_int::from(is_transient),
            tf_res.as_mut_ptr(),
            lm,
            buf.as_ptr(),
            buf.len() as c_int,
            pre,
            ec.as_mut_ptr(),
        );
    }
    ec
}

/// `deemphasis(in, pcm, N, C, downsample, coef, mem, accum)`.
pub fn deemphasis(
    in0: &[Sig],
    in1: &[Sig],
    pcm: &mut [Res],
    n: usize,
    c: usize,
    downsample: i32,
    coef: &[Val16; 4],
    mem: &mut [Sig; 2],
    accum: bool,
) {
    assert!(in0.len() >= n && (c < 2 || in1.len() >= n));
    assert!(pcm.len() >= n * c);
    // SAFETY: inputs hold n samples, pcm n*c, coef 4 and mem 2.
    unsafe {
        oracle_cd_deemphasis(
            in0.as_ptr(),
            if c == 2 { in1.as_ptr() } else { ptr::null() },
            pcm.as_mut_ptr(),
            n as c_int,
            c as c_int,
            downsample,
            coef.as_ptr(),
            mem.as_mut_ptr(),
            c_int::from(accum),
        );
    }
}

/// `celt_synthesis` for the standard mode at `fs` (48000, or 96000 with QEXT); `out_syn[c]` is
/// `outs[c][off..]`. With `use_qext`, `qext_mode = compute_qext_mode(mode)`.
pub fn celt_synthesis(
    fs: i32,
    x: &mut [Norm],
    out0: &mut [Sig],
    out1: Option<&mut [Sig]>,
    off: usize,
    old_band_e: &mut [Glog],
    start: i32,
    eff_end: i32,
    c: i32,
    cc: i32,
    is_transient: bool,
    lm: i32,
    downsample: i32,
    silence: bool,
    use_qext: bool,
    qext_band_log_e: &mut [Glog],
    qext_end: i32,
) {
    let n = (fs / 400) << lm;
    assert!(x.len() >= (c * n) as usize);
    assert!(out0.len() >= off + n as usize);
    let o1 = match out1 {
        Some(o) => {
            assert!(o.len() >= off + n as usize);
            o.as_mut_ptr()
        }
        None => ptr::null_mut(),
    };
    // SAFETY: buffers sized for the C accesses (checked above; the band-energy slices hold
    // C*nbEBands entries by construction in the tests).
    unsafe {
        oracle_cd_celt_synthesis(
            fs,
            x.as_mut_ptr(),
            out0.as_mut_ptr(),
            o1,
            off as c_int,
            old_band_e.as_mut_ptr(),
            start,
            eff_end,
            c,
            cc,
            c_int::from(is_transient),
            lm,
            downsample,
            c_int::from(silence),
            c_int::from(use_qext),
            qext_band_log_e.as_mut_ptr(),
            qext_end,
        );
    }
}

/// `celt_plc_pitch_search` on `mem0`/`mem1` (each `DECODE_BUFFER_SIZE*qext_scale` samples) for a
/// decoder at `fs` (48000, or 96000 with QEXT).
pub fn plc_pitch_search(fs: i32, mem0: &mut [Sig], mem1: &mut [Sig], c: i32) -> i32 {
    let dbs = if fs == 96000 { 4096 } else { 2048 };
    assert!(mem0.len() >= dbs && (c < 2 || mem1.len() >= dbs));
    // SAFETY: buffers hold the decode buffer size read by the C function.
    unsafe { oracle_cd_plc_pitch_search(fs, mem0.as_mut_ptr(), mem1.as_mut_ptr(), c) }
}
