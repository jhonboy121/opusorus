//! Safe RAII wrappers around the C libopus public API, for differential testing.
//!
//! Errors are returned as the raw libopus error code (`Err(code)`, always negative) so tests can
//! compare them directly with the Rust port's error mapping.

use crate::sys;
use core::ffi::c_int;
use core::ptr::NonNull;

/// Result type carrying a raw negative libopus error code on failure.
pub type OResult<T> = Result<T, i32>;

const fn check(ret: c_int) -> OResult<usize> {
    if ret < 0 { Err(ret) } else { Ok(ret as usize) }
}

/// Returns the libopus version string (e.g. `"libopus 1.6.1"`).
pub fn version_string() -> String {
    // SAFETY: returns a pointer to a static NUL-terminated string.
    let p = unsafe { sys::opus_get_version_string() };
    // SAFETY: p is non-null and NUL-terminated for the life of the program.
    unsafe { core::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned()
}

macro_rules! ctl_methods {
    ($ctl:path) => {
        /// Issues a CTL taking one `i32` argument (SET-style requests).
        pub fn ctl_set(&mut self, request: i32, value: i32) -> OResult<()> {
            // SAFETY: state pointer valid for self's lifetime; request takes an opus_int32 by value.
            check(unsafe { $ctl(self.ptr.as_ptr(), request, value) }).map(drop)
        }
        /// Issues a CTL taking one `*mut i32` out-argument (GET-style requests).
        pub fn ctl_get(&mut self, request: i32) -> OResult<i32> {
            let mut v: i32 = 0;
            // SAFETY: state pointer valid; request writes one opus_int32 through the pointer.
            check(unsafe { $ctl(self.ptr.as_ptr(), request, &mut v as *mut i32) })?;
            Ok(v)
        }
        /// Issues `OPUS_GET_FINAL_RANGE` (which writes an `opus_uint32`).
        pub fn final_range(&mut self) -> OResult<u32> {
            let mut v: u32 = 0;
            // SAFETY: state pointer valid; request writes one opus_uint32.
            check(unsafe {
                $ctl(
                    self.ptr.as_ptr(),
                    sys::OPUS_GET_FINAL_RANGE_REQUEST,
                    &mut v as *mut u32,
                )
            })?;
            Ok(v)
        }
        /// Issues `OPUS_RESET_STATE`.
        pub fn reset(&mut self) -> OResult<()> {
            // SAFETY: state pointer valid; reset takes no argument.
            check(unsafe { $ctl(self.ptr.as_ptr(), sys::OPUS_RESET_STATE) }).map(drop)
        }
    };
}

/// C `OpusEncoder`.
#[derive(Debug)]
pub struct Encoder {
    ptr: NonNull<sys::OpusEncoder>,
    channels: usize,
}

impl Encoder {
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
        })
    }
    /// `opus_encode`.
    pub fn encode(&mut self, pcm: &[i16], frame_size: usize, out: &mut [u8]) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        // SAFETY: buffers are valid for the lengths passed.
        check(unsafe {
            sys::opus_encode(
                self.ptr.as_ptr(),
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        })
    }
    /// `opus_encode24`.
    pub fn encode24(&mut self, pcm: &[i32], frame_size: usize, out: &mut [u8]) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        // SAFETY: buffers are valid for the lengths passed.
        check(unsafe {
            sys::opus_encode24(
                self.ptr.as_ptr(),
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        })
    }
    /// `opus_encode_float`.
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        out: &mut [u8],
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        // SAFETY: buffers are valid for the lengths passed.
        check(unsafe {
            sys::opus_encode_float(
                self.ptr.as_ptr(),
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        })
    }
    ctl_methods!(sys::opus_encoder_ctl);
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY: pointer came from opus_encoder_create and is destroyed exactly once.
        unsafe { sys::opus_encoder_destroy(self.ptr.as_ptr()) }
    }
}

/// C `OpusDecoder`.
#[derive(Debug)]
pub struct Decoder {
    ptr: NonNull<sys::OpusDecoder>,
    channels: usize,
}

fn data_ptr(data: Option<&[u8]>) -> (*const u8, i32) {
    match data {
        Some(d) => (d.as_ptr(), d.len() as i32),
        None => (core::ptr::null(), 0),
    }
}

impl Decoder {
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
    /// `opus_decode`; `data = None` requests PLC. Returns samples per channel.
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: usize,
        fec: bool,
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid for the lengths passed; null data means PLC.
        check(unsafe {
            sys::opus_decode(
                self.ptr.as_ptr(),
                d,
                l,
                pcm.as_mut_ptr(),
                frame_size as c_int,
                fec as c_int,
            )
        })
    }
    /// `opus_decode24`.
    pub fn decode24(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i32],
        frame_size: usize,
        fec: bool,
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid for the lengths passed.
        check(unsafe {
            sys::opus_decode24(
                self.ptr.as_ptr(),
                d,
                l,
                pcm.as_mut_ptr(),
                frame_size as c_int,
                fec as c_int,
            )
        })
    }
    /// `opus_decode_float`.
    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: usize,
        fec: bool,
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid for the lengths passed.
        check(unsafe {
            sys::opus_decode_float(
                self.ptr.as_ptr(),
                d,
                l,
                pcm.as_mut_ptr(),
                frame_size as c_int,
                fec as c_int,
            )
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
    ctl_methods!(sys::opus_decoder_ctl);
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: pointer came from opus_decoder_create and is destroyed exactly once.
        unsafe { sys::opus_decoder_destroy(self.ptr.as_ptr()) }
    }
}

/// Parsed packet as reported by `opus_packet_parse`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedPacket {
    /// TOC byte.
    pub toc: u8,
    /// (offset, length) of each frame within the input packet.
    pub frames: Vec<(usize, usize)>,
    /// Payload offset.
    pub payload_offset: usize,
}

/// `opus_packet_parse`.
pub fn packet_parse(data: &[u8]) -> OResult<ParsedPacket> {
    let mut toc = 0u8;
    let mut frames = [core::ptr::null::<u8>(); 48];
    let mut sizes = [0i16; 48];
    let mut off: c_int = 0;
    // SAFETY: all out pointers valid, arrays have 48 entries as required by the API.
    let n = check(unsafe {
        sys::opus_packet_parse(
            data.as_ptr(),
            data.len() as i32,
            &mut toc,
            frames.as_mut_ptr(),
            sizes.as_mut_ptr(),
            &mut off,
        )
    })?;
    let base = data.as_ptr() as usize;
    let frames = (0..n)
        .map(|i| (frames[i] as usize - base, sizes[i] as usize))
        .collect();
    Ok(ParsedPacket {
        toc,
        frames,
        payload_offset: off as usize,
    })
}

/// `opus_packet_get_bandwidth`.
pub fn packet_get_bandwidth(data: &[u8]) -> i32 {
    assert!(!data.is_empty());
    // SAFETY: reads only the first byte.
    unsafe { sys::opus_packet_get_bandwidth(data.as_ptr()) }
}
/// `opus_packet_get_samples_per_frame`.
pub fn packet_get_samples_per_frame(data: &[u8], fs: i32) -> i32 {
    assert!(!data.is_empty());
    // SAFETY: reads only the first byte.
    unsafe { sys::opus_packet_get_samples_per_frame(data.as_ptr(), fs) }
}
/// `opus_packet_get_nb_channels`.
pub fn packet_get_nb_channels(data: &[u8]) -> i32 {
    assert!(!data.is_empty());
    // SAFETY: reads only the first byte.
    unsafe { sys::opus_packet_get_nb_channels(data.as_ptr()) }
}
/// `opus_packet_get_nb_frames`.
pub fn packet_get_nb_frames(data: &[u8]) -> OResult<usize> {
    // SAFETY: buffer valid for len.
    check(unsafe { sys::opus_packet_get_nb_frames(data.as_ptr(), data.len() as i32) })
}
/// `opus_packet_get_nb_samples`.
pub fn packet_get_nb_samples(data: &[u8], fs: i32) -> OResult<usize> {
    // SAFETY: buffer valid for len.
    check(unsafe { sys::opus_packet_get_nb_samples(data.as_ptr(), data.len() as i32, fs) })
}
/// `opus_packet_has_lbrr`.
pub fn packet_has_lbrr(data: &[u8]) -> OResult<usize> {
    // SAFETY: buffer valid for len.
    check(unsafe { sys::opus_packet_has_lbrr(data.as_ptr(), data.len() as i32) })
}
/// `opus_pcm_soft_clip`.
pub fn pcm_soft_clip(pcm: &mut [f32], frame_size: usize, channels: usize, mem: &mut [f32]) {
    assert!(pcm.len() >= frame_size * channels && mem.len() >= channels);
    // SAFETY: buffers valid for the sizes passed.
    unsafe {
        sys::opus_pcm_soft_clip(
            pcm.as_mut_ptr(),
            frame_size as c_int,
            channels as c_int,
            mem.as_mut_ptr(),
        )
    }
}
/// `opus_packet_pad`: `buf[..len]` holds the packet, `buf.len()` must be >= `new_len`.
pub fn packet_pad(buf: &mut [u8], len: usize, new_len: usize) -> OResult<()> {
    assert!(buf.len() >= new_len.max(len));
    // SAFETY: buffer valid for max(len,new_len) bytes.
    check(unsafe { sys::opus_packet_pad(buf.as_mut_ptr(), len as i32, new_len as i32) }).map(drop)
}
/// `opus_packet_unpad`.
pub fn packet_unpad(buf: &mut [u8]) -> OResult<usize> {
    // SAFETY: buffer valid for len.
    check(unsafe { sys::opus_packet_unpad(buf.as_mut_ptr(), buf.len() as i32) })
}
/// `opus_multistream_packet_pad`.
pub fn multistream_packet_pad(
    buf: &mut [u8],
    len: usize,
    new_len: usize,
    nb_streams: i32,
) -> OResult<()> {
    assert!(buf.len() >= new_len.max(len));
    // SAFETY: buffer valid for max(len,new_len) bytes.
    check(unsafe {
        sys::opus_multistream_packet_pad(buf.as_mut_ptr(), len as i32, new_len as i32, nb_streams)
    })
    .map(drop)
}
/// `opus_multistream_packet_unpad`.
pub fn multistream_packet_unpad(buf: &mut [u8], nb_streams: i32) -> OResult<usize> {
    // SAFETY: buffer valid for len.
    check(unsafe {
        sys::opus_multistream_packet_unpad(buf.as_mut_ptr(), buf.len() as i32, nb_streams)
    })
}

/// C `OpusRepacketizer`. Holds copies of the packets it was fed so pointers stay valid.
#[derive(Debug)]
pub struct Repacketizer {
    ptr: NonNull<sys::OpusRepacketizer>,
    held: Vec<Box<[u8]>>,
}

impl Repacketizer {
    /// `opus_repacketizer_create`.
    pub fn new() -> OResult<Self> {
        // SAFETY: plain allocation.
        let p = unsafe { sys::opus_repacketizer_create() };
        let ptr = NonNull::new(p).ok_or(sys::OPUS_ALLOC_FAIL)?;
        Ok(Self {
            ptr,
            held: Vec::new(),
        })
    }
    /// `opus_repacketizer_init`.
    pub fn init(&mut self) {
        // SAFETY: valid state pointer.
        unsafe { sys::opus_repacketizer_init(self.ptr.as_ptr()) };
        self.held.clear();
    }
    /// `opus_repacketizer_cat`.
    pub fn cat(&mut self, data: &[u8]) -> OResult<()> {
        let copy: Box<[u8]> = data.into();
        // SAFETY: copy is kept alive in `held` until the next init/drop.
        let r = check(unsafe {
            sys::opus_repacketizer_cat(self.ptr.as_ptr(), copy.as_ptr(), copy.len() as i32)
        });
        self.held.push(copy);
        r.map(drop)
    }
    /// `opus_repacketizer_get_nb_frames`.
    pub fn nb_frames(&mut self) -> usize {
        // SAFETY: valid state pointer.
        unsafe { sys::opus_repacketizer_get_nb_frames(self.ptr.as_ptr()) as usize }
    }
    /// `opus_repacketizer_out_range`.
    pub fn out_range(&mut self, begin: i32, end: i32, out: &mut [u8]) -> OResult<usize> {
        // SAFETY: valid state and output buffer.
        check(unsafe {
            sys::opus_repacketizer_out_range(
                self.ptr.as_ptr(),
                begin,
                end,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        })
    }
    /// `opus_repacketizer_out`.
    pub fn out(&mut self, out: &mut [u8]) -> OResult<usize> {
        // SAFETY: valid state and output buffer.
        check(unsafe {
            sys::opus_repacketizer_out(self.ptr.as_ptr(), out.as_mut_ptr(), out.len() as i32)
        })
    }
}

impl Drop for Repacketizer {
    fn drop(&mut self) {
        // SAFETY: created by opus_repacketizer_create, destroyed once.
        unsafe { sys::opus_repacketizer_destroy(self.ptr.as_ptr()) }
    }
}

/// C `OpusMSEncoder`.
#[derive(Debug)]
pub struct MsEncoder {
    ptr: NonNull<sys::OpusMSEncoder>,
    channels: usize,
    /// Number of streams.
    pub streams: i32,
    /// Number of coupled streams.
    pub coupled_streams: i32,
    /// Channel mapping.
    pub mapping: Vec<u8>,
}

impl MsEncoder {
    /// `opus_multistream_encoder_create`.
    pub fn new(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled: i32,
        mapping: &[u8],
        app: i32,
    ) -> OResult<Self> {
        assert!(mapping.len() >= channels as usize);
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
            mapping: mapping.to_vec(),
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
    /// `opus_multistream_encode`.
    pub fn encode(&mut self, pcm: &[i16], frame_size: usize, out: &mut [u8]) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_multistream_encode(
                self.ptr.as_ptr(),
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        })
    }
    /// `opus_multistream_encode_float`.
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        out: &mut [u8],
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_multistream_encode_float(
                self.ptr.as_ptr(),
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        })
    }
    /// `opus_multistream_encode24`.
    pub fn encode24(&mut self, pcm: &[i32], frame_size: usize, out: &mut [u8]) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_multistream_encode24(
                self.ptr.as_ptr(),
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        })
    }
    ctl_methods!(sys::opus_multistream_encoder_ctl);
}

impl Drop for MsEncoder {
    fn drop(&mut self) {
        // SAFETY: created by create(), destroyed once.
        unsafe { sys::opus_multistream_encoder_destroy(self.ptr.as_ptr()) }
    }
}

/// C `OpusMSDecoder`.
#[derive(Debug)]
pub struct MsDecoder {
    ptr: NonNull<sys::OpusMSDecoder>,
    channels: usize,
}

impl MsDecoder {
    /// `opus_multistream_decoder_create`.
    pub fn new(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled: i32,
        mapping: &[u8],
    ) -> OResult<Self> {
        assert!(mapping.len() >= channels as usize);
        let mut err = 0;
        // SAFETY: mapping has >= channels entries.
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
        })
    }
    /// `opus_multistream_decode`.
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: usize,
        fec: bool,
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_multistream_decode(
                self.ptr.as_ptr(),
                d,
                l,
                pcm.as_mut_ptr(),
                frame_size as c_int,
                fec as c_int,
            )
        })
    }
    /// `opus_multistream_decode24`.
    pub fn decode24(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i32],
        frame_size: usize,
        fec: bool,
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_multistream_decode24(
                self.ptr.as_ptr(),
                d,
                l,
                pcm.as_mut_ptr(),
                frame_size as c_int,
                fec as c_int,
            )
        })
    }
    /// `opus_multistream_decode_float`.
    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: usize,
        fec: bool,
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_multistream_decode_float(
                self.ptr.as_ptr(),
                d,
                l,
                pcm.as_mut_ptr(),
                frame_size as c_int,
                fec as c_int,
            )
        })
    }
    ctl_methods!(sys::opus_multistream_decoder_ctl);
}

impl Drop for MsDecoder {
    fn drop(&mut self) {
        // SAFETY: created by create(), destroyed once.
        unsafe { sys::opus_multistream_decoder_destroy(self.ptr.as_ptr()) }
    }
}

/// C `OpusProjectionEncoder`.
#[derive(Debug)]
pub struct ProjectionEncoder {
    ptr: NonNull<sys::OpusProjectionEncoder>,
    channels: usize,
    /// Number of streams.
    pub streams: i32,
    /// Number of coupled streams.
    pub coupled_streams: i32,
}

impl ProjectionEncoder {
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
    /// `opus_projection_encode`.
    pub fn encode(&mut self, pcm: &[i16], frame_size: usize, out: &mut [u8]) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_projection_encode(
                self.ptr.as_ptr(),
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        })
    }
    /// `opus_projection_encode_float`.
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        out: &mut [u8],
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_projection_encode_float(
                self.ptr.as_ptr(),
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        })
    }
    /// Fetches the demixing matrix (as serialized for the decoder / Ogg header).
    pub fn demixing_matrix(&mut self) -> OResult<Vec<u8>> {
        let size = self.ctl_get(sys::OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST)?;
        let mut m = vec![0u8; size as usize];
        // SAFETY: m has `size` bytes, the request writes exactly that many.
        check(unsafe {
            sys::opus_projection_encoder_ctl(
                self.ptr.as_ptr(),
                sys::OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST,
                m.as_mut_ptr(),
                size,
            )
        })?;
        Ok(m)
    }
    ctl_methods!(sys::opus_projection_encoder_ctl);
}

impl Drop for ProjectionEncoder {
    fn drop(&mut self) {
        // SAFETY: created by create(), destroyed once.
        unsafe { sys::opus_projection_encoder_destroy(self.ptr.as_ptr()) }
    }
}

/// C `OpusProjectionDecoder`.
#[derive(Debug)]
pub struct ProjectionDecoder {
    ptr: NonNull<sys::OpusProjectionDecoder>,
    channels: usize,
}

impl ProjectionDecoder {
    /// `opus_projection_decoder_create`.
    pub fn new(fs: i32, channels: i32, streams: i32, coupled: i32, matrix: &[u8]) -> OResult<Self> {
        let mut m = matrix.to_vec();
        let mut err = 0;
        // SAFETY: matrix buffer valid for its length (C API takes non-const but only reads).
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
        })
    }
    /// `opus_projection_decode`.
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: usize,
        fec: bool,
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_projection_decode(
                self.ptr.as_ptr(),
                d,
                l,
                pcm.as_mut_ptr(),
                frame_size as c_int,
                fec as c_int,
            )
        })
    }
    /// `opus_projection_decode_float`.
    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: usize,
        fec: bool,
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid.
        check(unsafe {
            sys::opus_projection_decode_float(
                self.ptr.as_ptr(),
                d,
                l,
                pcm.as_mut_ptr(),
                frame_size as c_int,
                fec as c_int,
            )
        })
    }
    ctl_methods!(sys::opus_projection_decoder_ctl);
}

impl Drop for ProjectionDecoder {
    fn drop(&mut self) {
        // SAFETY: created by create(), destroyed once.
        unsafe { sys::opus_projection_decoder_destroy(self.ptr.as_ptr()) }
    }
}
