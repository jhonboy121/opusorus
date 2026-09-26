//! Oracle bindings for unit `opus_encoder` (src/opus_encoder.c, src/opus_multistream_encoder.c,
//! src/opus_projection_encoder.c).
//!
//! * [`Enc`], [`MsEnc`], [`ProjEnc`]: RAII wrappers of the library encoders with every CTL the
//!   tests need (energy masks, per-stream access through
//!   `OPUS_MULTISTREAM_GET_ENCODER_STATE`, `opus_projection_encode24`, demixing matrices).
//! * [`DupEnc`]: a private copy of the Opus encoder (same source, renamed symbols) whose struct
//!   can be dumped ([`DupEnc::dump`]).
//! * Free functions: the static helpers of opus_encoder.c / opus_multistream_encoder.c.
//!
//! Errors are the raw negative libopus codes, as in [`crate::api`].
//!
//! Built in the float and in the fixed-point oracle: signal / state values use the build's
//! types ([`Res`], [`Val16`], [`Val32`], [`Glog`]).

#![allow(
    clippy::too_many_arguments,
    reason = "wrappers mirror the flat C shim signatures"
)]

use core::ffi::{c_int, c_uchar};
use core::ptr::NonNull;

use crate::api::OResult;
use crate::sys;

/// `opus_res` of the oracle build.
#[cfg(not(feature = "fixed-point"))]
pub type Res = f32;
/// `opus_res` of the oracle build.
#[cfg(all(feature = "fixed-point", not(feature = "fixed-res24")))]
pub type Res = i16;
/// `opus_res` of the oracle build.
#[cfg(feature = "fixed-res24")]
pub type Res = i32;
/// `opus_val16` of the oracle build.
#[cfg(not(feature = "fixed-point"))]
pub type Val16 = f32;
/// `opus_val16` of the oracle build.
#[cfg(feature = "fixed-point")]
pub type Val16 = i16;
/// `opus_val32` of the oracle build.
#[cfg(not(feature = "fixed-point"))]
pub type Val32 = f32;
/// `opus_val32` of the oracle build.
#[cfg(feature = "fixed-point")]
pub type Val32 = i32;
/// `celt_glog` of the oracle build (Q24 in the fixed-point build).
#[cfg(not(feature = "fixed-point"))]
pub type Glog = f32;
/// `celt_glog` of the oracle build (Q24 in the fixed-point build).
#[cfg(feature = "fixed-point")]
pub type Glog = i32;

const fn check(ret: c_int) -> OResult<usize> {
    if ret < 0 { Err(ret) } else { Ok(ret as usize) }
}

/// `OPUS_SET_ENERGY_MASK_REQUEST` (celt.h).
pub const OPUS_SET_ENERGY_MASK_REQUEST: c_int = 10026;

unsafe extern "C" {
    fn oracle_oe_gen_toc(mode: c_int, framerate: c_int, bandwidth: c_int, channels: c_int)
    -> c_int;
    fn oracle_oe_hp_cutoff(
        input: *const Res,
        cutoff_hz: c_int,
        out: *mut Res,
        hp_mem: *mut Val32,
        len: c_int,
        channels: c_int,
        fs: c_int,
    );
    fn oracle_oe_dc_reject(
        input: *const Res,
        cutoff_hz: c_int,
        out: *mut Res,
        hp_mem: *mut Val32,
        len: c_int,
        channels: c_int,
        fs: c_int,
    );
    fn oracle_oe_stereo_fade(
        buf: *mut Res,
        g1: Val16,
        g2: Val16,
        fs96: c_int,
        frame_size: c_int,
        channels: c_int,
        fs: c_int,
    );
    fn oracle_oe_gain_fade(
        buf: *mut Res,
        g1: Val16,
        g2: Val16,
        fs96: c_int,
        frame_size: c_int,
        channels: c_int,
        fs: c_int,
    );
    fn oracle_oe_frame_size_select(
        application: c_int,
        frame_size: c_int,
        variable_duration: c_int,
        fs: c_int,
    ) -> c_int;
    fn oracle_oe_compute_stereo_width(
        pcm: *const Res,
        frame_size: c_int,
        fs: c_int,
        mem: *mut Val32,
    ) -> Val16;
    fn oracle_oe_decide_fec(
        use_fec: c_int,
        loss: c_int,
        last_fec: c_int,
        mode: c_int,
        bandwidth: *mut c_int,
        rate: c_int,
    ) -> c_int;
    fn oracle_oe_compute_silk_rate_for_hybrid(
        rate: c_int,
        bandwidth: c_int,
        frame20ms: c_int,
        vbr: c_int,
        fec: c_int,
        channels: c_int,
    ) -> c_int;
    fn oracle_oe_compute_equiv_rate(
        bitrate: c_int,
        channels: c_int,
        frame_rate: c_int,
        vbr: c_int,
        mode: c_int,
        complexity: c_int,
        loss: c_int,
    ) -> c_int;
    fn oracle_oe_compute_frame_energy(pcm: *const Res, frame_size: c_int, channels: c_int)
    -> Val32;
    fn oracle_oe_decide_dtx_mode(activity: c_int, nb: *mut c_int, frame_size_ms_q1: c_int)
    -> c_int;
    fn oracle_oe_compute_redundancy_bytes(
        max_data_bytes: c_int,
        bitrate_bps: c_int,
        frame_rate: c_int,
        channels: c_int,
    ) -> c_int;
    fn oracle_oe_log_sum(a: Glog, b: Glog) -> Val16;
    fn oracle_oe_channel_pos(channels: c_int, pos: *mut c_int);
    #[cfg(not(feature = "disable-float-api"))]
    fn oracle_oe_surround_analysis(
        pcm: *const f32,
        band_log_e: *mut Glog,
        mem: *mut Val32,
        preemph_mem: *mut Val32,
        len: c_int,
        channels: c_int,
        rate: c_int,
    );

    fn oracle_oe_create(
        fs: c_int,
        channels: c_int,
        application: c_int,
        err: *mut c_int,
    ) -> *mut sys::OpusEncoder;
    fn oracle_oe_destroy(st: *mut sys::OpusEncoder);
    fn oracle_oe_ctl_set(st: *mut sys::OpusEncoder, request: c_int, value: c_int) -> c_int;
    #[cfg(not(feature = "disable-float-api"))]
    fn oracle_oe_encode_float(
        st: *mut sys::OpusEncoder,
        pcm: *const f32,
        frame_size: c_int,
        out: *mut c_uchar,
        max_bytes: c_int,
    ) -> c_int;
    fn oracle_oe_encode(
        st: *mut sys::OpusEncoder,
        pcm: *const i16,
        frame_size: c_int,
        out: *mut c_uchar,
        max_bytes: c_int,
    ) -> c_int;
    fn oracle_oe_encode24(
        st: *mut sys::OpusEncoder,
        pcm: *const i32,
        frame_size: c_int,
        out: *mut c_uchar,
        max_bytes: c_int,
    ) -> c_int;
    fn oracle_oe_ctl_get(st: *mut sys::OpusEncoder, request: c_int, value: *mut i32) -> c_int;
    fn oracle_oe_dump(st: *const sys::OpusEncoder, v: *mut u32, delay: *mut Res) -> c_int;
}

// ---------------------------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------------------------

/// `gen_toc`.
pub fn gen_toc(mode: i32, framerate: i32, bandwidth: i32, channels: i32) -> u8 {
    // SAFETY: pure function of its arguments.
    unsafe { oracle_oe_gen_toc(mode, framerate, bandwidth, channels) as u8 }
}

/// `hp_cutoff`: `input`/`out` hold `len*channels` samples.
pub fn hp_cutoff(
    input: &[Res],
    cutoff_hz: i32,
    out: &mut [Res],
    hp_mem: &mut [Val32; 4],
    len: i32,
    channels: i32,
    fs: i32,
) {
    let n = (len * channels) as usize;
    assert!(input.len() >= n && out.len() >= n);
    // SAFETY: buffers hold len*channels samples; hp_mem has 4 entries.
    unsafe {
        oracle_oe_hp_cutoff(
            input.as_ptr(),
            cutoff_hz,
            out.as_mut_ptr(),
            hp_mem.as_mut_ptr(),
            len,
            channels,
            fs,
        );
    }
}

/// `dc_reject`: `input`/`out` hold `len*channels` samples.
pub fn dc_reject(
    input: &[Res],
    cutoff_hz: i32,
    out: &mut [Res],
    hp_mem: &mut [Val32; 4],
    len: i32,
    channels: i32,
    fs: i32,
) {
    let n = (len * channels) as usize;
    assert!(input.len() >= n && out.len() >= n);
    // SAFETY: buffers hold len*channels samples; hp_mem has 4 entries.
    unsafe {
        oracle_oe_dc_reject(
            input.as_ptr(),
            cutoff_hz,
            out.as_mut_ptr(),
            hp_mem.as_mut_ptr(),
            len,
            channels,
            fs,
        );
    }
}

/// `stereo_fade` in place with the 48 kHz (or, `fs96`, 96 kHz) mode window.
pub fn stereo_fade(
    buf: &mut [Res],
    g1: Val16,
    g2: Val16,
    fs96: bool,
    frame_size: i32,
    channels: i32,
    fs: i32,
) {
    assert!(buf.len() >= (frame_size * channels) as usize && channels == 2);
    // SAFETY: buf holds frame_size*channels samples.
    unsafe {
        oracle_oe_stereo_fade(
            buf.as_mut_ptr(),
            g1,
            g2,
            fs96 as c_int,
            frame_size,
            channels,
            fs,
        );
    }
}

/// `gain_fade` in place with the 48 kHz (or, `fs96`, 96 kHz) mode window.
pub fn gain_fade(
    buf: &mut [Res],
    g1: Val16,
    g2: Val16,
    fs96: bool,
    frame_size: i32,
    channels: i32,
    fs: i32,
) {
    assert!(buf.len() >= (frame_size * channels) as usize);
    // SAFETY: buf holds frame_size*channels samples.
    unsafe {
        oracle_oe_gain_fade(
            buf.as_mut_ptr(),
            g1,
            g2,
            fs96 as c_int,
            frame_size,
            channels,
            fs,
        );
    }
}

/// `frame_size_select`.
pub fn frame_size_select(
    application: i32,
    frame_size: i32,
    variable_duration: i32,
    fs: i32,
) -> i32 {
    // SAFETY: pure function of its arguments.
    unsafe { oracle_oe_frame_size_select(application, frame_size, variable_duration, fs) }
}

/// `compute_stereo_width`; `mem` = XX, XY, YY, smoothed_width, max_follower.
pub fn compute_stereo_width(pcm: &[Res], frame_size: i32, fs: i32, mem: &mut [Val32; 5]) -> Val16 {
    assert!(pcm.len() >= 2 * frame_size as usize);
    // SAFETY: pcm holds 2*frame_size samples; mem has 5 entries.
    unsafe { oracle_oe_compute_stereo_width(pcm.as_ptr(), frame_size, fs, mem.as_mut_ptr()) }
}

/// `decide_fec`.
pub fn decide_fec(
    use_fec: i32,
    loss: i32,
    last_fec: i32,
    mode: i32,
    bandwidth: &mut i32,
    rate: i32,
) -> i32 {
    // SAFETY: bandwidth is a valid in/out pointer.
    unsafe { oracle_oe_decide_fec(use_fec, loss, last_fec, mode, bandwidth, rate) }
}

/// `compute_silk_rate_for_hybrid`.
pub fn compute_silk_rate_for_hybrid(
    rate: i32,
    bandwidth: i32,
    frame20ms: i32,
    vbr: i32,
    fec: i32,
    channels: i32,
) -> i32 {
    // SAFETY: pure function of its arguments.
    unsafe {
        oracle_oe_compute_silk_rate_for_hybrid(rate, bandwidth, frame20ms, vbr, fec, channels)
    }
}

/// `compute_equiv_rate`.
pub fn compute_equiv_rate(
    bitrate: i32,
    channels: i32,
    frame_rate: i32,
    vbr: i32,
    mode: i32,
    complexity: i32,
    loss: i32,
) -> i32 {
    // SAFETY: pure function of its arguments.
    unsafe {
        oracle_oe_compute_equiv_rate(bitrate, channels, frame_rate, vbr, mode, complexity, loss)
    }
}

/// `compute_frame_energy`.
pub fn compute_frame_energy(pcm: &[Res], frame_size: i32, channels: i32) -> Val32 {
    assert!(pcm.len() >= (frame_size * channels) as usize);
    // SAFETY: pcm holds frame_size*channels samples.
    unsafe { oracle_oe_compute_frame_energy(pcm.as_ptr(), frame_size, channels) }
}

/// `decide_dtx_mode`.
pub fn decide_dtx_mode(
    activity: i32,
    nb_no_activity_ms_q1: &mut i32,
    frame_size_ms_q1: i32,
) -> i32 {
    // SAFETY: valid in/out pointer.
    unsafe { oracle_oe_decide_dtx_mode(activity, nb_no_activity_ms_q1, frame_size_ms_q1) }
}

/// `compute_redundancy_bytes`.
pub fn compute_redundancy_bytes(
    max_data_bytes: i32,
    bitrate_bps: i32,
    frame_rate: i32,
    channels: i32,
) -> i32 {
    // SAFETY: pure function of its arguments.
    unsafe { oracle_oe_compute_redundancy_bytes(max_data_bytes, bitrate_bps, frame_rate, channels) }
}

/// `logSum`.
pub fn log_sum(a: Glog, b: Glog) -> Val16 {
    // SAFETY: pure function of its arguments.
    unsafe { oracle_oe_log_sum(a, b) }
}

/// `channel_pos`.
pub fn channel_pos(channels: i32) -> [i32; 8] {
    let mut pos = [0i32; 8];
    // SAFETY: pos has 8 entries.
    unsafe { oracle_oe_channel_pos(channels, pos.as_mut_ptr()) };
    pos
}

#[cfg(not(feature = "disable-float-api"))]
/// `surround_analysis` on float input (`len*channels` samples) with the mode for `rate`.
/// `mem`: `channels*overlap`, `preemph_mem`: `channels`, `band_log_e`: `21*channels`.
pub fn surround_analysis(
    pcm: &[f32],
    band_log_e: &mut [Glog],
    mem: &mut [Val32],
    preemph_mem: &mut [Val32],
    len: i32,
    channels: i32,
    rate: i32,
    overlap: usize,
) {
    let ch = channels as usize;
    assert!(pcm.len() >= len as usize * ch);
    assert!(band_log_e.len() >= 21 * ch && mem.len() >= overlap * ch && preemph_mem.len() >= ch);
    // SAFETY: buffer sizes checked above.
    unsafe {
        oracle_oe_surround_analysis(
            pcm.as_ptr(),
            band_log_e.as_mut_ptr(),
            mem.as_mut_ptr(),
            preemph_mem.as_mut_ptr(),
            len,
            channels,
            rate,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Private encoder copy with state dumps
// ---------------------------------------------------------------------------------------------

/// A private copy of the C Opus encoder whose struct can be dumped.
#[derive(Debug)]
pub struct DupEnc {
    ptr: NonNull<sys::OpusEncoder>,
    channels: usize,
}

impl DupEnc {
    /// `opus_encoder_create`.
    pub fn new(fs: i32, channels: i32, application: i32) -> OResult<Self> {
        let mut err = 0;
        // SAFETY: err is a valid out pointer.
        let p = unsafe { oracle_oe_create(fs, channels, application, &mut err) };
        check(err)?;
        let ptr = NonNull::new(p).ok_or(sys::OPUS_ALLOC_FAIL)?;
        Ok(Self {
            ptr,
            channels: channels as usize,
        })
    }
    /// `opus_encoder_ctl` with one int argument.
    pub fn ctl_set(&mut self, request: i32, value: i32) -> OResult<()> {
        // SAFETY: valid state; request takes one opus_int32.
        check(unsafe { oracle_oe_ctl_set(self.ptr.as_ptr(), request, value) }).map(drop)
    }
    #[cfg(not(feature = "disable-float-api"))]
    /// `opus_encode_float`.
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        out: &mut [u8],
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        // SAFETY: buffers valid for the sizes passed.
        check(unsafe {
            oracle_oe_encode_float(
                self.ptr.as_ptr(),
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                out.len() as c_int,
            )
        })
    }
    /// `opus_encode`.
    pub fn encode(&mut self, pcm: &[i16], frame_size: usize, out: &mut [u8]) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        // SAFETY: buffers valid for the sizes passed.
        check(unsafe {
            oracle_oe_encode(
                self.ptr.as_ptr(),
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                out.len() as c_int,
            )
        })
    }
    /// `opus_encode24`.
    pub fn encode24(&mut self, pcm: &[i32], frame_size: usize, out: &mut [u8]) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        // SAFETY: buffers valid for the sizes passed.
        check(unsafe {
            oracle_oe_encode24(
                self.ptr.as_ptr(),
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                out.len() as c_int,
            )
        })
    }
    /// `opus_encoder_ctl` writing one `opus_int32` (`OPUS_GET_FINAL_RANGE` writes the u32 bits).
    pub fn ctl_get(&mut self, request: i32) -> OResult<i32> {
        let mut v: i32 = 0;
        // SAFETY: valid state; the request writes one 32-bit value.
        check(unsafe { oracle_oe_ctl_get(self.ptr.as_ptr(), request, &mut v) })?;
        Ok(v)
    }
    /// Flat state dump (see `oracle_oe_dump` in the shim for the field order) and the delay
    /// buffer (`encoder_buffer*channels` samples).
    pub fn dump(&self) -> (Vec<u32>, Vec<Res>) {
        let mut v = vec![0u32; 64];
        let mut d: Vec<Res> = vec![Res::default(); 2 * 960];
        // SAFETY: v has 64 entries (the shim writes 63), d holds the largest delay buffer.
        let n = unsafe { oracle_oe_dump(self.ptr.as_ptr(), v.as_mut_ptr(), d.as_mut_ptr()) };
        v.truncate(n as usize);
        // encoder_buffer * channels
        let len = v[16] as usize * v[1] as usize;
        d.truncate(len);
        (v, d)
    }
}

impl Drop for DupEnc {
    fn drop(&mut self) {
        // SAFETY: created by oracle_oe_create, destroyed once.
        unsafe { oracle_oe_destroy(self.ptr.as_ptr()) }
    }
}

// ---------------------------------------------------------------------------------------------
// Library encoders
// ---------------------------------------------------------------------------------------------

/// Library `OpusEncoder` with energy-mask support.
#[derive(Debug)]
pub struct Enc {
    ptr: NonNull<sys::OpusEncoder>,
    channels: usize,
    /// Storage for `OPUS_SET_ENERGY_MASK` (C keeps the pointer).
    mask: Box<[Glog; 42]>,
}

macro_rules! encode_fns {
    ($enc:path, $enc24:path, $encf:path) => {
        /// 16-bit encode.
        pub fn encode(&mut self, pcm: &[i16], frame_size: usize, out: &mut [u8]) -> OResult<usize> {
            assert!(pcm.len() >= frame_size * self.channels);
            // SAFETY: buffers valid for the sizes passed.
            check(unsafe {
                $enc(
                    self.ptr.as_ptr(),
                    pcm.as_ptr(),
                    frame_size as c_int,
                    out.as_mut_ptr(),
                    out.len() as i32,
                )
            })
        }
        /// 24-bit encode.
        pub fn encode24(
            &mut self,
            pcm: &[i32],
            frame_size: usize,
            out: &mut [u8],
        ) -> OResult<usize> {
            assert!(pcm.len() >= frame_size * self.channels);
            // SAFETY: buffers valid for the sizes passed.
            check(unsafe {
                $enc24(
                    self.ptr.as_ptr(),
                    pcm.as_ptr(),
                    frame_size as c_int,
                    out.as_mut_ptr(),
                    out.len() as i32,
                )
            })
        }
        #[cfg(not(feature = "disable-float-api"))]
        /// Float encode.
        pub fn encode_float(
            &mut self,
            pcm: &[f32],
            frame_size: usize,
            out: &mut [u8],
        ) -> OResult<usize> {
            assert!(pcm.len() >= frame_size * self.channels);
            // SAFETY: buffers valid for the sizes passed.
            check(unsafe {
                $encf(
                    self.ptr.as_ptr(),
                    pcm.as_ptr(),
                    frame_size as c_int,
                    out.as_mut_ptr(),
                    out.len() as i32,
                )
            })
        }
    };
}

macro_rules! ctl_fns {
    ($ctl:path) => {
        /// CTL with one `opus_int32` argument.
        pub fn ctl_set(&mut self, request: i32, value: i32) -> OResult<()> {
            // SAFETY: valid state; request takes one opus_int32 by value.
            check(unsafe { $ctl(self.ptr.as_ptr(), request, value) }).map(drop)
        }
        /// CTL writing one `opus_int32`.
        pub fn ctl_get(&mut self, request: i32) -> OResult<i32> {
            let mut v: i32 = 0;
            // SAFETY: valid state; request writes one opus_int32.
            check(unsafe { $ctl(self.ptr.as_ptr(), request, &mut v as *mut i32) })?;
            Ok(v)
        }
        /// `OPUS_GET_FINAL_RANGE`.
        pub fn final_range(&mut self) -> OResult<u32> {
            let mut v: u32 = 0;
            // SAFETY: valid state; request writes one opus_uint32.
            check(unsafe {
                $ctl(
                    self.ptr.as_ptr(),
                    sys::OPUS_GET_FINAL_RANGE_REQUEST,
                    &mut v as *mut u32,
                )
            })?;
            Ok(v)
        }
        /// `OPUS_RESET_STATE`.
        pub fn reset(&mut self) -> OResult<()> {
            // SAFETY: valid state; no argument.
            check(unsafe { $ctl(self.ptr.as_ptr(), sys::OPUS_RESET_STATE) }).map(drop)
        }
    };
}

impl Enc {
    /// `opus_encoder_create`.
    pub fn new(fs: i32, channels: i32, application: i32) -> OResult<Self> {
        let mut err = 0;
        // SAFETY: err is a valid out pointer.
        let p = unsafe { sys::opus_encoder_create(fs, channels, application, &mut err) };
        check(err)?;
        let ptr = NonNull::new(p).ok_or(sys::OPUS_ALLOC_FAIL)?;
        Ok(Self {
            ptr,
            channels: channels as usize,
            mask: Box::new([Glog::default(); 42]),
        })
    }
    encode_fns!(sys::opus_encode, sys::opus_encode24, sys::opus_encode_float);
    ctl_fns!(sys::opus_encoder_ctl);

    /// `OPUS_SET_ENERGY_MASK`: copies `mask` into storage owned by this wrapper (C keeps the
    /// pointer) or passes NULL.
    pub fn set_energy_mask(&mut self, mask: Option<&[Glog]>) -> OResult<()> {
        let p: *const Glog = match mask {
            Some(m) => {
                self.mask.fill(Glog::default());
                self.mask[..m.len()].copy_from_slice(m);
                self.mask.as_ptr()
            }
            None => core::ptr::null(),
        };
        // SAFETY: the mask storage lives as long as the encoder (boxed, never moved).
        check(unsafe { sys::opus_encoder_ctl(self.ptr.as_ptr(), OPUS_SET_ENERGY_MASK_REQUEST, p) })
            .map(drop)
    }
}

impl Drop for Enc {
    fn drop(&mut self) {
        // SAFETY: created by opus_encoder_create, destroyed once.
        unsafe { sys::opus_encoder_destroy(self.ptr.as_ptr()) }
    }
}

/// Library `OpusMSEncoder` with per-stream CTL access.
#[derive(Debug)]
pub struct MsEnc {
    ptr: NonNull<sys::OpusMSEncoder>,
    channels: usize,
    /// Number of streams.
    pub streams: i32,
    /// Number of coupled streams.
    pub coupled_streams: i32,
    /// Channel mapping.
    pub mapping: Vec<u8>,
}

impl MsEnc {
    /// `opus_multistream_encoder_create`.
    pub fn new(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled: i32,
        mapping: &[u8],
        app: i32,
    ) -> OResult<Self> {
        assert!(mapping.len() >= channels.max(0) as usize);
        let mut err = 0;
        // SAFETY: mapping has >= channels entries; err is valid.
        let p = unsafe {
            sys::opus_multistream_encoder_create(
                fs,
                channels,
                streams,
                coupled,
                mapping.as_ptr(),
                app,
                &mut err,
            )
        };
        check(err)?;
        let ptr = NonNull::new(p).ok_or(sys::OPUS_ALLOC_FAIL)?;
        Ok(Self {
            ptr,
            channels: channels as usize,
            streams,
            coupled_streams: coupled,
            mapping: mapping[..channels as usize].to_vec(),
        })
    }
    /// `opus_multistream_surround_encoder_create`.
    pub fn new_surround(fs: i32, channels: i32, family: i32, app: i32) -> OResult<Self> {
        let mut err = 0;
        let (mut s, mut c) = (0, 0);
        let mut mapping = vec![0u8; 256];
        // SAFETY: mapping has 256 entries (>= channels); outs valid.
        let p = unsafe {
            sys::opus_multistream_surround_encoder_create(
                fs,
                channels,
                family,
                &mut s,
                &mut c,
                mapping.as_mut_ptr(),
                app,
                &mut err,
            )
        };
        check(err)?;
        let ptr = NonNull::new(p).ok_or(sys::OPUS_ALLOC_FAIL)?;
        mapping.truncate(channels as usize);
        Ok(Self {
            ptr,
            channels: channels as usize,
            streams: s,
            coupled_streams: c,
            mapping,
        })
    }
    encode_fns!(
        sys::opus_multistream_encode,
        sys::opus_multistream_encode24,
        sys::opus_multistream_encode_float
    );
    ctl_fns!(sys::opus_multistream_encoder_ctl);

    fn stream_ptr(&mut self, stream: i32) -> OResult<*mut sys::OpusEncoder> {
        let mut enc: *mut sys::OpusEncoder = core::ptr::null_mut();
        // SAFETY: valid state; the request takes the stream id and an OpusEncoder** out pointer.
        check(unsafe {
            sys::opus_multistream_encoder_ctl(
                self.ptr.as_ptr(),
                sys::OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST,
                stream,
                &mut enc as *mut *mut sys::OpusEncoder,
            )
        })?;
        Ok(enc)
    }
    /// `opus_encoder_ctl` GET on stream `stream`.
    pub fn stream_ctl_get(&mut self, stream: i32, request: i32) -> OResult<i32> {
        let enc = self.stream_ptr(stream)?;
        let mut v: i32 = 0;
        // SAFETY: enc points into the multistream state; request writes one opus_int32.
        check(unsafe { sys::opus_encoder_ctl(enc, request, &mut v as *mut i32) })?;
        Ok(v)
    }
    /// `opus_encoder_ctl` SET on stream `stream`.
    pub fn stream_ctl_set(&mut self, stream: i32, request: i32, value: i32) -> OResult<()> {
        let enc = self.stream_ptr(stream)?;
        // SAFETY: enc points into the multistream state; request takes one opus_int32.
        check(unsafe { sys::opus_encoder_ctl(enc, request, value) }).map(drop)
    }
    /// `OPUS_GET_FINAL_RANGE` of stream `stream`.
    pub fn stream_final_range(&mut self, stream: i32) -> OResult<u32> {
        let enc = self.stream_ptr(stream)?;
        let mut v: u32 = 0;
        // SAFETY: enc points into the multistream state; request writes one opus_uint32.
        check(unsafe {
            sys::opus_encoder_ctl(enc, sys::OPUS_GET_FINAL_RANGE_REQUEST, &mut v as *mut u32)
        })?;
        Ok(v)
    }
    /// Raw `OPUS_MULTISTREAM_GET_ENCODER_STATE` result code (for bad stream ids).
    pub fn encoder_state_check(&mut self, stream: i32) -> OResult<()> {
        self.stream_ptr(stream).map(drop)
    }
}

impl Drop for MsEnc {
    fn drop(&mut self) {
        // SAFETY: created by create(), destroyed once.
        unsafe { sys::opus_multistream_encoder_destroy(self.ptr.as_ptr()) }
    }
}

/// Library `OpusProjectionEncoder` (with `encode24`).
#[derive(Debug)]
pub struct ProjEnc {
    ptr: NonNull<sys::OpusProjectionEncoder>,
    channels: usize,
    /// Number of streams.
    pub streams: i32,
    /// Number of coupled streams.
    pub coupled_streams: i32,
}

impl ProjEnc {
    /// `opus_projection_ambisonics_encoder_create`.
    pub fn new(fs: i32, channels: i32, family: i32, app: i32) -> OResult<Self> {
        let (mut s, mut c, mut err) = (0, 0, 0);
        // SAFETY: out pointers valid.
        let p = unsafe {
            sys::opus_projection_ambisonics_encoder_create(
                fs, channels, family, &mut s, &mut c, app, &mut err,
            )
        };
        check(err)?;
        let ptr = NonNull::new(p).ok_or(sys::OPUS_ALLOC_FAIL)?;
        Ok(Self {
            ptr,
            channels: channels as usize,
            streams: s,
            coupled_streams: c,
        })
    }
    encode_fns!(
        sys::opus_projection_encode,
        sys::opus_projection_encode24,
        sys::opus_projection_encode_float
    );
    ctl_fns!(sys::opus_projection_encoder_ctl);

    /// `OPUS_PROJECTION_GET_DEMIXING_MATRIX` into a buffer of `size` bytes (raw result code
    /// for wrong sizes).
    pub fn demixing_matrix_sized(&mut self, size: usize) -> OResult<Vec<u8>> {
        let mut m = vec![0u8; size.max(1)];
        // SAFETY: m has max(size,1) bytes; the request writes `size` bytes when it succeeds.
        check(unsafe {
            sys::opus_projection_encoder_ctl(
                self.ptr.as_ptr(),
                sys::OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST,
                m.as_mut_ptr(),
                size as i32,
            )
        })?;
        m.truncate(size);
        Ok(m)
    }
}

impl Drop for ProjEnc {
    fn drop(&mut self) {
        // SAFETY: created by create(), destroyed once.
        unsafe { sys::opus_projection_encoder_destroy(self.ptr.as_ptr()) }
    }
}
