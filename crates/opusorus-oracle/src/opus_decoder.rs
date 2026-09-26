//! Oracle bindings for unit `opus_decoder` (src/opus_decoder.c, src/opus_multistream_decoder.c,
//! src/opus_projection_decoder.c).
//!
//! * [`Dec`]: a C `OpusDecoder` with every public call plus a dump of its private Opus-level
//!   state ([`DecState`]).
//! * [`MsDec`]: a C `OpusMSDecoder` with per-stream state access.
//! * [`ProjDec`]: a C `OpusProjectionDecoder` (incl. `opus_projection_decode24`, missing from
//!   [`crate::api`]) with per-stream state access.
//!
//! Errors are raw negative libopus codes (`Err(code)`). The shim compiles in both oracles
//! (`// oracle-build: any`); the public API has the same C types in every build.

use crate::sys;
use core::ffi::c_int;
use core::ptr::NonNull;

unsafe extern "C" {
    fn oracle_odec_has_qext() -> c_int;
    fn oracle_odec_dump(dec: *const sys::OpusDecoder, iout: *mut i32, fout: *mut f32);
    fn oracle_odec_ms_stream(
        st: *mut sys::OpusMSDecoder,
        stream_id: c_int,
    ) -> *mut sys::OpusDecoder;
    fn oracle_odec_proj_stream(
        st: *mut sys::OpusProjectionDecoder,
        stream_id: c_int,
    ) -> *mut sys::OpusDecoder;
    fn oracle_odec_ms_stream_ret(st: *mut sys::OpusMSDecoder, stream_id: c_int) -> c_int;
}

/// Result type carrying a raw negative libopus error code on failure.
pub type OResult<T> = Result<T, i32>;

const fn check(ret: c_int) -> OResult<usize> {
    if ret < 0 { Err(ret) } else { Ok(ret as usize) }
}

/// Whether the oracle was built with `ENABLE_QEXT`.
pub fn has_qext() -> bool {
    // SAFETY: pure function.
    unsafe { oracle_odec_has_qext() != 0 }
}

/// The Opus-level fields of a C `OpusDecoder` (see `oracle_odec_dump`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecState {
    /// `channels, Fs, DecControl.{nChannelsAPI, nChannelsInternal, API_sampleRate,
    /// internalSampleRate, payloadSize_ms, prevPitchLag, enable_deep_plc}, decode_gain,
    /// complexity, ignore_extensions, stream_channels, bandwidth, mode, prev_mode, frame_size,
    /// prev_redundancy, last_packet_duration, rangeFinal, 0`.
    pub ints: [i32; 21],
    /// `softclip_mem` (zeros in a fixed-point oracle, whose `OpusDecoder` has none).
    pub softclip_mem: [f32; 2],
}

/// Dumps a decoder reached through a raw pointer.
///
/// # Safety
/// `p` must point to a live, initialized C `OpusDecoder`.
unsafe fn dump_ptr(p: *const sys::OpusDecoder) -> DecState {
    let mut ints = [0i32; 21];
    let mut softclip_mem = [0f32; 2];
    // SAFETY: caller guarantees p is valid; the output arrays have the sizes the shim writes.
    unsafe { oracle_odec_dump(p, ints.as_mut_ptr(), softclip_mem.as_mut_ptr()) };
    DecState { ints, softclip_mem }
}

fn data_ptr(data: Option<&[u8]>) -> (*const u8, i32) {
    match data {
        Some(d) => (d.as_ptr(), d.len() as i32),
        None => (core::ptr::null(), 0),
    }
}

macro_rules! ctl_methods {
    ($ctl:path) => {
        /// CTL taking one `opus_int32` argument (SET-style requests).
        pub fn ctl_set(&mut self, request: i32, value: i32) -> OResult<()> {
            // SAFETY: state pointer valid; the request takes an opus_int32 by value.
            check(unsafe { $ctl(self.ptr.as_ptr(), request, value) }).map(drop)
        }
        /// CTL taking one `opus_int32*` out-argument (GET-style requests).
        pub fn ctl_get(&mut self, request: i32) -> OResult<i32> {
            let mut v: i32 = 0;
            // SAFETY: state pointer valid; the request writes one opus_int32.
            check(unsafe { $ctl(self.ptr.as_ptr(), request, &mut v as *mut i32) })?;
            Ok(v)
        }
        /// `OPUS_GET_FINAL_RANGE`.
        pub fn final_range(&mut self) -> OResult<u32> {
            let mut v: u32 = 0;
            // SAFETY: state pointer valid; the request writes one opus_uint32.
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
            // SAFETY: state pointer valid; reset takes no argument.
            check(unsafe { $ctl(self.ptr.as_ptr(), sys::OPUS_RESET_STATE) }).map(drop)
        }
    };
}

/// A C `OpusDecoder`.
#[derive(Debug)]
pub struct Dec {
    ptr: NonNull<sys::OpusDecoder>,
    channels: usize,
}

// SAFETY: the C decoder has no thread affinity; the handle is uniquely owned.
unsafe impl Send for Dec {}

impl Dec {
    /// `opus_decoder_create`.
    pub fn new(fs: i32, channels: i32) -> OResult<Self> {
        let mut err = 0;
        // SAFETY: err is a valid out pointer.
        let p = unsafe { sys::opus_decoder_create(fs, channels, &mut err) };
        check(err)?;
        let ptr = NonNull::new(p).ok_or(sys::OPUS_ALLOC_FAIL)?;
        Ok(Self {
            ptr,
            channels: channels as usize,
        })
    }
    /// `opus_decode` (`fec` passed through as an int, so invalid values can be tested).
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        fec: i32,
    ) -> OResult<usize> {
        assert!(frame_size <= 0 || pcm.len() >= frame_size as usize * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid for the lengths passed; null data means PLC.
        check(unsafe {
            sys::opus_decode(self.ptr.as_ptr(), d, l, pcm.as_mut_ptr(), frame_size, fec)
        })
    }
    /// `opus_decode24`.
    pub fn decode24(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i32],
        frame_size: i32,
        fec: i32,
    ) -> OResult<usize> {
        assert!(frame_size <= 0 || pcm.len() >= frame_size as usize * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid for the lengths passed.
        check(unsafe {
            sys::opus_decode24(self.ptr.as_ptr(), d, l, pcm.as_mut_ptr(), frame_size, fec)
        })
    }
    #[cfg(not(feature = "disable-float-api"))]
    /// `opus_decode_float`.
    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: i32,
        fec: i32,
    ) -> OResult<usize> {
        assert!(frame_size <= 0 || pcm.len() >= frame_size as usize * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid for the lengths passed.
        check(unsafe {
            sys::opus_decode_float(self.ptr.as_ptr(), d, l, pcm.as_mut_ptr(), frame_size, fec)
        })
    }
    /// `opus_decoder_get_nb_samples`.
    pub fn nb_samples(&self, packet: &[u8]) -> OResult<usize> {
        // SAFETY: valid state and packet buffer.
        check(unsafe {
            sys::opus_decoder_get_nb_samples(
                self.ptr.as_ptr(),
                packet.as_ptr(),
                packet.len() as i32,
            )
        })
    }
    /// Dump of the private Opus-level state.
    pub fn dump(&self) -> DecState {
        // SAFETY: the pointer is a live decoder owned by self.
        unsafe { dump_ptr(self.ptr.as_ptr()) }
    }
    ctl_methods!(sys::opus_decoder_ctl);
}

impl Drop for Dec {
    fn drop(&mut self) {
        // SAFETY: created by opus_decoder_create, destroyed once.
        unsafe { sys::opus_decoder_destroy(self.ptr.as_ptr()) }
    }
}

/// A C `OpusMSDecoder`.
#[derive(Debug)]
pub struct MsDec {
    ptr: NonNull<sys::OpusMSDecoder>,
    channels: usize,
    streams: i32,
}

// SAFETY: uniquely owned C handle without thread affinity.
unsafe impl Send for MsDec {}

impl MsDec {
    /// `opus_multistream_decoder_create`.
    pub fn new(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled: i32,
        mapping: &[u8],
    ) -> OResult<Self> {
        assert!(mapping.len() >= channels.max(0) as usize);
        let mut err = 0;
        // SAFETY: mapping has >= channels entries; err is valid.
        let p = unsafe {
            sys::opus_multistream_decoder_create(
                fs,
                channels,
                streams,
                coupled,
                mapping.as_ptr(),
                &mut err,
            )
        };
        check(err)?;
        let ptr = NonNull::new(p).ok_or(sys::OPUS_ALLOC_FAIL)?;
        Ok(Self {
            ptr,
            channels: channels as usize,
            streams,
        })
    }
    /// `opus_multistream_decode`.
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        fec: i32,
    ) -> OResult<usize> {
        assert!(frame_size <= 0 || pcm.len() >= frame_size.min(11520) as usize * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_multistream_decode(self.ptr.as_ptr(), d, l, pcm.as_mut_ptr(), frame_size, fec)
        })
    }
    /// `opus_multistream_decode24`.
    pub fn decode24(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i32],
        frame_size: i32,
        fec: i32,
    ) -> OResult<usize> {
        assert!(frame_size <= 0 || pcm.len() >= frame_size.min(11520) as usize * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_multistream_decode24(
                self.ptr.as_ptr(),
                d,
                l,
                pcm.as_mut_ptr(),
                frame_size,
                fec,
            )
        })
    }
    #[cfg(not(feature = "disable-float-api"))]
    /// `opus_multistream_decode_float`.
    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: i32,
        fec: i32,
    ) -> OResult<usize> {
        assert!(frame_size <= 0 || pcm.len() >= frame_size.min(11520) as usize * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_multistream_decode_float(
                self.ptr.as_ptr(),
                d,
                l,
                pcm.as_mut_ptr(),
                frame_size,
                fec,
            )
        })
    }
    /// Dump of stream `s`'s decoder state (via `OPUS_MULTISTREAM_GET_DECODER_STATE`).
    pub fn stream_dump(&mut self, s: i32) -> DecState {
        assert!(s >= 0 && s < self.streams);
        // SAFETY: valid state; the returned pointer lives inside it.
        let p = unsafe { oracle_odec_ms_stream(self.ptr.as_ptr(), s) };
        assert!(!p.is_null());
        // SAFETY: p points into the live multistream state.
        unsafe { dump_ptr(p) }
    }
    /// A GET CTL on stream `s`'s decoder.
    pub fn stream_ctl_get(&mut self, s: i32, request: i32) -> OResult<i32> {
        assert!(s >= 0 && s < self.streams);
        // SAFETY: valid state; the returned pointer lives inside it.
        let p = unsafe { oracle_odec_ms_stream(self.ptr.as_ptr(), s) };
        assert!(!p.is_null());
        let mut v = 0i32;
        // SAFETY: p is a live decoder; the request writes one opus_int32.
        check(unsafe { sys::opus_decoder_ctl(p, request, &mut v as *mut i32) })?;
        Ok(v)
    }
    /// Return code of `OPUS_MULTISTREAM_GET_DECODER_STATE(stream_id, &dec)`.
    pub fn stream_state_ret(&mut self, stream_id: i32) -> i32 {
        // SAFETY: valid state; out pointer handled by the shim.
        unsafe { oracle_odec_ms_stream_ret(self.ptr.as_ptr(), stream_id) }
    }
    ctl_methods!(sys::opus_multistream_decoder_ctl);
}

impl Drop for MsDec {
    fn drop(&mut self) {
        // SAFETY: created by create(), destroyed once.
        unsafe { sys::opus_multistream_decoder_destroy(self.ptr.as_ptr()) }
    }
}

/// A C `OpusProjectionDecoder`.
#[derive(Debug)]
pub struct ProjDec {
    ptr: NonNull<sys::OpusProjectionDecoder>,
    channels: usize,
    streams: i32,
}

// SAFETY: uniquely owned C handle without thread affinity.
unsafe impl Send for ProjDec {}

impl ProjDec {
    /// `opus_projection_decoder_create`.
    pub fn new(fs: i32, channels: i32, streams: i32, coupled: i32, matrix: &[u8]) -> OResult<Self> {
        let mut m = matrix.to_vec();
        let mut err = 0;
        // SAFETY: matrix buffer valid for its length (the C API only reads it).
        let p = unsafe {
            sys::opus_projection_decoder_create(
                fs,
                channels,
                streams,
                coupled,
                m.as_mut_ptr(),
                m.len() as i32,
                &mut err,
            )
        };
        check(err)?;
        let ptr = NonNull::new(p).ok_or(sys::OPUS_ALLOC_FAIL)?;
        Ok(Self {
            ptr,
            channels: channels as usize,
            streams,
        })
    }
    /// `opus_projection_decode`.
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        fec: i32,
    ) -> OResult<usize> {
        assert!(frame_size <= 0 || pcm.len() >= frame_size.min(11520) as usize * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_projection_decode(self.ptr.as_ptr(), d, l, pcm.as_mut_ptr(), frame_size, fec)
        })
    }
    /// `opus_projection_decode24`.
    pub fn decode24(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i32],
        frame_size: i32,
        fec: i32,
    ) -> OResult<usize> {
        assert!(frame_size <= 0 || pcm.len() >= frame_size.min(11520) as usize * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_projection_decode24(
                self.ptr.as_ptr(),
                d,
                l,
                pcm.as_mut_ptr(),
                frame_size,
                fec,
            )
        })
    }
    #[cfg(not(feature = "disable-float-api"))]
    /// `opus_projection_decode_float`.
    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: i32,
        fec: i32,
    ) -> OResult<usize> {
        assert!(frame_size <= 0 || pcm.len() >= frame_size.min(11520) as usize * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_projection_decode_float(
                self.ptr.as_ptr(),
                d,
                l,
                pcm.as_mut_ptr(),
                frame_size,
                fec,
            )
        })
    }
    /// Dump of stream `s`'s decoder state.
    pub fn stream_dump(&mut self, s: i32) -> DecState {
        assert!(s >= 0 && s < self.streams);
        // SAFETY: valid state; the returned pointer lives inside it.
        let p = unsafe { oracle_odec_proj_stream(self.ptr.as_ptr(), s) };
        assert!(!p.is_null());
        // SAFETY: p points into the live projection state.
        unsafe { dump_ptr(p) }
    }
    ctl_methods!(sys::opus_projection_decoder_ctl);
}

impl Drop for ProjDec {
    fn drop(&mut self) {
        // SAFETY: created by create(), destroyed once.
        unsafe { sys::opus_projection_decoder_destroy(self.ptr.as_ptr()) }
    }
}
