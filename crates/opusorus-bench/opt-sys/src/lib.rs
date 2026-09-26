//! # opus-sys-optimized
//!
//! Benchmark-only bindings to two C builds of the vendored libopus v1.6.1:
//!
//! * **Optimized** ([`Variant::Optimized`], [`Encoder`], [`Decoder`], [`MsEncoder`],
//!   [`MsDecoder`]): the default upstream CMake `Release` configuration (`-O3`, intrinsics and
//!   RTCD on, hardening on). Every global symbol of that static library carries the `optopus_`
//!   prefix (see `build.rs`), so it links next to the unprefixed oracle.
//! * **Scalar** ([`Variant::Scalar`]): a copy of libopus compiled with exactly the
//!   `opusorus-oracle` configuration (same sources, defines and flags: no intrinsics,
//!   `-O2 -ffp-contract=off`), prefixed `scalopus_`, with the same micro shims
//!   (`csrc/bench_shim.c`). It is self-contained so the shims never depend on the order in
//!   which the linker sees the oracle's archive. The scalar *codec* benchmarks use the oracle
//!   crate itself (`opusorus_oracle::api`).
//!
//! The micro entry points ([`Micro`], [`Resampler`]) expose the MDCT, FFT, range coder and
//! SILK resampler of either build with allocation-free calls.

use core::ffi::{c_int, c_uint, c_void};
use core::ptr::NonNull;

/// C error code (`OPUS_BAD_ARG`, ...) as returned by libopus.
pub type CResult<T> = Result<T, i32>;

/// Opaque C `OpusEncoder`.
#[repr(C)]
#[derive(Debug)]
pub struct OpusEncoder {
    _p: [u8; 0],
}
/// Opaque C `OpusDecoder`.
#[repr(C)]
#[derive(Debug)]
pub struct OpusDecoder {
    _p: [u8; 0],
}
/// Opaque C `OpusMSEncoder`.
#[repr(C)]
#[derive(Debug)]
pub struct OpusMsEncoder {
    _p: [u8; 0],
}
/// Opaque C `OpusMSDecoder`.
#[repr(C)]
#[derive(Debug)]
pub struct OpusMsDecoder {
    _p: [u8; 0],
}

unsafe extern "C" {
    fn optopus_opus_encoder_create(
        fs: i32,
        channels: c_int,
        application: c_int,
        error: *mut c_int,
    ) -> *mut OpusEncoder;
    fn optopus_opus_encoder_destroy(st: *mut OpusEncoder);
    fn optopus_opus_encoder_ctl(st: *mut OpusEncoder, request: c_int, ...) -> c_int;
    fn optopus_opus_encode_float(
        st: *mut OpusEncoder,
        pcm: *const f32,
        frame_size: c_int,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;

    fn optopus_opus_decoder_create(fs: i32, channels: c_int, error: *mut c_int)
    -> *mut OpusDecoder;
    fn optopus_opus_decoder_destroy(st: *mut OpusDecoder);
    fn optopus_opus_decoder_ctl(st: *mut OpusDecoder, request: c_int, ...) -> c_int;
    fn optopus_opus_decode_float(
        st: *mut OpusDecoder,
        data: *const u8,
        len: i32,
        pcm: *mut f32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;

    fn optopus_opus_multistream_surround_encoder_create(
        fs: i32,
        channels: c_int,
        mapping_family: c_int,
        streams: *mut c_int,
        coupled_streams: *mut c_int,
        mapping: *mut u8,
        application: c_int,
        error: *mut c_int,
    ) -> *mut OpusMsEncoder;
    fn optopus_opus_multistream_encoder_destroy(st: *mut OpusMsEncoder);
    fn optopus_opus_multistream_encoder_ctl(st: *mut OpusMsEncoder, request: c_int, ...) -> c_int;
    fn optopus_opus_multistream_encode_float(
        st: *mut OpusMsEncoder,
        pcm: *const f32,
        frame_size: c_int,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> c_int;

    fn optopus_opus_multistream_decoder_create(
        fs: i32,
        channels: c_int,
        streams: c_int,
        coupled_streams: c_int,
        mapping: *const u8,
        error: *mut c_int,
    ) -> *mut OpusMsDecoder;
    fn optopus_opus_multistream_decoder_destroy(st: *mut OpusMsDecoder);
    fn optopus_opus_multistream_decode_float(
        st: *mut OpusMsDecoder,
        data: *const u8,
        len: i32,
        pcm: *mut f32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
}

/// Declares the micro shim entry points once per variant (`scalopus_bench_*` /
/// `optopus_bench_*`).
macro_rules! shim_decls {
    ($m:ident, $arch:ident, $mdct_n:ident, $mdct_overlap:ident, $mdct_fwd:ident,
     $mdct_bwd:ident, $fft_nfft:ident, $fft:ident, $ec:ident, $rs_new:ident, $rs_run:ident,
     $rs_free:ident) => {
        mod $m {
            use super::{c_int, c_uint, c_void};
            unsafe extern "C" {
                pub(crate) fn $arch() -> c_int;
                pub(crate) fn $mdct_n() -> c_int;
                pub(crate) fn $mdct_overlap() -> c_int;
                pub(crate) fn $mdct_fwd(
                    input: *const f32,
                    out: *mut f32,
                    shift: c_int,
                    arch: c_int,
                );
                pub(crate) fn $mdct_bwd(
                    input: *const f32,
                    out: *mut f32,
                    shift: c_int,
                    arch: c_int,
                );
                pub(crate) fn $fft_nfft(idx: c_int) -> c_int;
                pub(crate) fn $fft(idx: c_int, fin: *const f32, fout: *mut f32, arch: c_int);
                pub(crate) fn $ec(
                    ops: *const c_uint,
                    nops: c_int,
                    buf: *mut u8,
                    size: c_int,
                ) -> c_uint;
                pub(crate) fn $rs_new(fs_in: c_int, fs_out: c_int, for_enc: c_int) -> *mut c_void;
                pub(crate) fn $rs_run(
                    st: *mut c_void,
                    out: *mut i16,
                    input: *const i16,
                    len: c_int,
                );
                pub(crate) fn $rs_free(st: *mut c_void);
            }
            /// Function table of this variant.
            pub(crate) const TABLE: super::Shim = super::Shim {
                arch: $arch,
                mdct_n: $mdct_n,
                mdct_overlap: $mdct_overlap,
                mdct_forward: $mdct_fwd,
                mdct_backward: $mdct_bwd,
                fft_nfft: $fft_nfft,
                fft: $fft,
                ec_roundtrip: $ec,
                resampler_new: $rs_new,
                resampler_run: $rs_run,
                resampler_free: $rs_free,
            };
        }
    };
}

shim_decls!(
    scalar,
    scalopus_bench_arch,
    scalopus_bench_mdct_n,
    scalopus_bench_mdct_overlap,
    scalopus_bench_mdct_forward,
    scalopus_bench_mdct_backward,
    scalopus_bench_fft_nfft,
    scalopus_bench_fft,
    scalopus_bench_ec_roundtrip,
    scalopus_bench_resampler_new,
    scalopus_bench_resampler_run,
    scalopus_bench_resampler_free
);
shim_decls!(
    optimized,
    optopus_bench_arch,
    optopus_bench_mdct_n,
    optopus_bench_mdct_overlap,
    optopus_bench_mdct_forward,
    optopus_bench_mdct_backward,
    optopus_bench_fft_nfft,
    optopus_bench_fft,
    optopus_bench_ec_roundtrip,
    optopus_bench_resampler_new,
    optopus_bench_resampler_run,
    optopus_bench_resampler_free
);

/// Function pointers of one variant's shims.
#[derive(Debug, Clone, Copy)]
struct Shim {
    arch: unsafe extern "C" fn() -> c_int,
    mdct_n: unsafe extern "C" fn() -> c_int,
    mdct_overlap: unsafe extern "C" fn() -> c_int,
    mdct_forward: unsafe extern "C" fn(*const f32, *mut f32, c_int, c_int),
    mdct_backward: unsafe extern "C" fn(*const f32, *mut f32, c_int, c_int),
    fft_nfft: unsafe extern "C" fn(c_int) -> c_int,
    fft: unsafe extern "C" fn(c_int, *const f32, *mut f32, c_int),
    ec_roundtrip: unsafe extern "C" fn(*const c_uint, c_int, *mut u8, c_int) -> c_uint,
    resampler_new: unsafe extern "C" fn(c_int, c_int, c_int) -> *mut c_void,
    resampler_run: unsafe extern "C" fn(*mut c_void, *mut i16, *const i16, c_int),
    resampler_free: unsafe extern "C" fn(*mut c_void),
}

/// Which C build a micro benchmark runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// The `opusorus-oracle` configuration: scalar C, `-O2 -ffp-contract=off`, no intrinsics.
    Scalar,
    /// Upstream CMake `Release`: `-O3`, intrinsics and RTCD enabled.
    Optimized,
}

impl Variant {
    const fn shim(self) -> Shim {
        match self {
            Self::Scalar => scalar::TABLE,
            Self::Optimized => optimized::TABLE,
        }
    }
}

/// Micro-benchmark entry points of one C build (MDCT, FFT, range coder) with the RTCD
/// architecture index resolved once.
#[derive(Debug, Clone, Copy)]
pub struct Micro {
    shim: Shim,
    arch: c_int,
    mdct_n: usize,
    overlap: usize,
}

impl Micro {
    /// Resolves the variant's architecture (`opus_select_arch`) and mode sizes.
    #[must_use]
    pub fn new(variant: Variant) -> Self {
        let shim = variant.shim();
        // SAFETY: the shims take no arguments and only read static data / CPU features.
        let (arch, n, ov) = unsafe { ((shim.arch)(), (shim.mdct_n)(), (shim.mdct_overlap)()) };
        Self {
            shim,
            arch,
            mdct_n: n as usize,
            overlap: ov as usize,
        }
    }

    /// The RTCD architecture index this build dispatches on (0 = plain C).
    #[must_use]
    pub const fn arch(&self) -> i32 {
        self.arch
    }

    /// MDCT size `N` of the 48 kHz static mode (1920).
    #[must_use]
    pub const fn mdct_n(&self) -> usize {
        self.mdct_n
    }

    /// Overlap of the 48 kHz static mode (120).
    #[must_use]
    pub const fn overlap(&self) -> usize {
        self.overlap
    }

    /// `clt_mdct_forward` of the 48 kHz static mode with its window, stride 1.
    ///
    /// # Panics
    /// If `input` is shorter than `N/2 + overlap` or `out` shorter than `N/2`
    /// (`N = 1920 >> shift`), or `shift > 3`.
    pub fn mdct_forward(&self, input: &[f32], out: &mut [f32], shift: usize) {
        assert!(shift <= 3);
        let n2 = (self.mdct_n >> shift) / 2;
        assert!(input.len() >= n2 + self.overlap && out.len() >= n2);
        // SAFETY: buffer sizes checked above against the C access pattern.
        unsafe {
            (self.shim.mdct_forward)(input.as_ptr(), out.as_mut_ptr(), shift as c_int, self.arch);
        }
    }

    /// `clt_mdct_backward` of the 48 kHz static mode with its window, stride 1.
    ///
    /// # Panics
    /// If `input` is shorter than `N/2` or `out` shorter than `N/2 + overlap`, or `shift > 3`.
    pub fn mdct_backward(&self, input: &[f32], out: &mut [f32], shift: usize) {
        assert!(shift <= 3);
        let n2 = (self.mdct_n >> shift) / 2;
        assert!(input.len() >= n2 && out.len() >= n2 + self.overlap);
        // SAFETY: buffer sizes checked above against the C access pattern.
        unsafe {
            (self.shim.mdct_backward)(input.as_ptr(), out.as_mut_ptr(), shift as c_int, self.arch);
        }
    }

    /// Size of FFT state `idx` (0..=3) of the 48 kHz mode (480, 240, 120, 60).
    ///
    /// # Panics
    /// If `idx > 3`.
    #[must_use]
    pub fn fft_nfft(&self, idx: usize) -> usize {
        assert!(idx <= 3);
        // SAFETY: idx is a valid kfft index of the static mode.
        unsafe { (self.shim.fft_nfft)(idx as c_int) as usize }
    }

    /// Scaled forward FFT (`opus_fft`) with FFT state `idx`; interleaved complex buffers.
    ///
    /// # Panics
    /// If a buffer holds fewer than `2 * nfft` floats or `idx > 3`.
    pub fn fft(&self, idx: usize, fin: &[f32], fout: &mut [f32]) {
        let n = self.fft_nfft(idx);
        assert!(fin.len() >= 2 * n && fout.len() >= 2 * n);
        // SAFETY: both buffers hold nfft complex values as checked.
        unsafe { (self.shim.fft)(idx as c_int, fin.as_ptr(), fout.as_mut_ptr(), self.arch) }
    }

    /// Encodes `ops` (see `csrc/bench_shim.c`) into `buf`, decodes them back and returns the
    /// checksum over final ranges and decoded values.
    ///
    /// # Panics
    /// If `ops` or `buf` do not fit in a C `int`.
    pub fn ec_roundtrip(&self, ops: &[[u32; 4]], buf: &mut [u8]) -> u32 {
        let nops = c_int::try_from(ops.len()).expect("op count fits in a C int");
        let size = c_int::try_from(buf.len()).expect("buffer size fits in a C int");
        // SAFETY: `ops` is a contiguous array of 4-word ops, `buf` holds `size` bytes; the
        // coder never writes past `size` (it sets its error flag instead).
        unsafe { (self.shim.ec_roundtrip)(ops.as_ptr().cast(), nops, buf.as_mut_ptr(), size) }
    }
}

/// A C `silk_resampler_state_struct` of one build.
#[derive(Debug)]
pub struct Resampler {
    shim: Shim,
    ptr: NonNull<c_void>,
    ratio: (usize, usize),
}

impl Resampler {
    /// `silk_resampler_init(fs_in, fs_out, for_enc)`; `None` for unsupported rate pairs.
    #[must_use]
    pub fn new(variant: Variant, fs_in: i32, fs_out: i32, for_enc: bool) -> Option<Self> {
        let shim = variant.shim();
        // SAFETY: plain allocation + init; returns NULL on failure.
        let p = unsafe { (shim.resampler_new)(fs_in, fs_out, c_int::from(for_enc)) };
        NonNull::new(p).map(|ptr| Self {
            shim,
            ptr,
            ratio: (fs_in as usize / 1000, fs_out as usize / 1000),
        })
    }

    /// `silk_resampler`: resamples all of `input` (at least 1 ms) into `out`.
    ///
    /// # Panics
    /// If `input` is shorter than 1 ms or `out` cannot hold the resampled length.
    pub fn run(&mut self, out: &mut [i16], input: &[i16]) {
        let (fin, fout) = self.ratio;
        assert!(input.len() >= fin && out.len() >= input.len() * fout / fin);
        let len = c_int::try_from(input.len()).expect("input length fits in a C int");
        // SAFETY: valid state; `out` holds the resampled length as checked above.
        unsafe {
            (self.shim.resampler_run)(self.ptr.as_ptr(), out.as_mut_ptr(), input.as_ptr(), len)
        }
    }
}

impl Drop for Resampler {
    fn drop(&mut self) {
        // SAFETY: allocated by resampler_new of the same build, freed exactly once.
        unsafe { (self.shim.resampler_free)(self.ptr.as_ptr()) }
    }
}

// SAFETY: the state is exclusively owned heap memory with no thread affinity.
unsafe impl Send for Resampler {}

const fn check(ret: c_int) -> CResult<usize> {
    if ret < 0 { Err(ret) } else { Ok(ret as usize) }
}

fn data_ptr(data: Option<&[u8]>) -> CResult<(*const u8, i32)> {
    match data {
        Some(d) => Ok((d.as_ptr(), i32::try_from(d.len()).map_err(|_| -1)?)),
        None => Ok((core::ptr::null(), 0)),
    }
}

fn len_i32(n: usize) -> CResult<i32> {
    // OPUS_BAD_ARG for lengths C cannot represent.
    i32::try_from(n).map_err(|_| -1)
}

/// Optimized-build `OpusEncoder`.
#[derive(Debug)]
pub struct Encoder {
    ptr: NonNull<OpusEncoder>,
    channels: usize,
}

impl Encoder {
    /// `opus_encoder_create`.
    ///
    /// # Errors
    /// The C error code.
    pub fn new(fs: i32, channels: i32, application: i32) -> CResult<Self> {
        let mut err = 0;
        // SAFETY: err is a valid out pointer.
        let p = unsafe { optopus_opus_encoder_create(fs, channels, application, &mut err) };
        check(err)?;
        let ptr = NonNull::new(p).ok_or(-7)?;
        Ok(Self {
            ptr,
            channels: channels as usize,
        })
    }

    /// `opus_encoder_ctl(request, (opus_int32)value)` for a setter request.
    ///
    /// # Errors
    /// The C error code.
    pub fn ctl_set(&mut self, request: i32, value: i32) -> CResult<()> {
        // SAFETY: setter requests take one opus_int32 vararg.
        check(unsafe { optopus_opus_encoder_ctl(self.ptr.as_ptr(), request, value) }).map(|_| ())
    }

    /// `opus_encoder_ctl(request, &value)` for a getter request.
    ///
    /// # Errors
    /// The C error code.
    pub fn ctl_get(&mut self, request: i32) -> CResult<i32> {
        let mut v: i32 = 0;
        // SAFETY: getter requests take one opus_int32* vararg.
        check(unsafe { optopus_opus_encoder_ctl(self.ptr.as_ptr(), request, &mut v as *mut i32) })?;
        Ok(v)
    }

    /// `opus_encode_float`.
    ///
    /// # Errors
    /// The C error code.
    ///
    /// # Panics
    /// If `pcm` holds fewer than `frame_size * channels` samples.
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        out: &mut [u8],
    ) -> CResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        let (fs, max) = (len_i32(frame_size)?, len_i32(out.len())?);
        // SAFETY: buffers are valid for the lengths passed.
        check(unsafe {
            optopus_opus_encode_float(self.ptr.as_ptr(), pcm.as_ptr(), fs, out.as_mut_ptr(), max)
        })
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY: created by opus_encoder_create, destroyed once.
        unsafe { optopus_opus_encoder_destroy(self.ptr.as_ptr()) }
    }
}

// SAFETY: exclusively owned C state with no thread affinity.
unsafe impl Send for Encoder {}

/// Optimized-build `OpusDecoder`.
#[derive(Debug)]
pub struct Decoder {
    ptr: NonNull<OpusDecoder>,
    channels: usize,
}

impl Decoder {
    /// `opus_decoder_create`.
    ///
    /// # Errors
    /// The C error code.
    pub fn new(fs: i32, channels: i32) -> CResult<Self> {
        let mut err = 0;
        // SAFETY: err is a valid out pointer.
        let p = unsafe { optopus_opus_decoder_create(fs, channels, &mut err) };
        check(err)?;
        let ptr = NonNull::new(p).ok_or(-7)?;
        Ok(Self {
            ptr,
            channels: channels as usize,
        })
    }

    /// `opus_decoder_ctl(request, &value)` for a getter request.
    ///
    /// # Errors
    /// The C error code.
    pub fn ctl_get(&mut self, request: i32) -> CResult<i32> {
        let mut v: i32 = 0;
        // SAFETY: getter requests take one opus_int32* vararg.
        check(unsafe { optopus_opus_decoder_ctl(self.ptr.as_ptr(), request, &mut v as *mut i32) })?;
        Ok(v)
    }

    /// `opus_decode_float`; `data = None` requests PLC. Returns samples per channel.
    ///
    /// # Errors
    /// The C error code.
    ///
    /// # Panics
    /// If `pcm` holds fewer than `frame_size * channels` samples.
    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: usize,
        fec: bool,
    ) -> CResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        let (d, l) = data_ptr(data)?;
        let fs = len_i32(frame_size)?;
        // SAFETY: buffers valid for the lengths passed; null data means PLC.
        check(unsafe {
            optopus_opus_decode_float(
                self.ptr.as_ptr(),
                d,
                l,
                pcm.as_mut_ptr(),
                fs,
                c_int::from(fec),
            )
        })
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: created by opus_decoder_create, destroyed once.
        unsafe { optopus_opus_decoder_destroy(self.ptr.as_ptr()) }
    }
}

// SAFETY: exclusively owned C state with no thread affinity.
unsafe impl Send for Decoder {}

/// Optimized-build `OpusMSEncoder` (surround API).
#[derive(Debug)]
pub struct MsEncoder {
    ptr: NonNull<OpusMsEncoder>,
    channels: usize,
    /// Number of streams.
    pub streams: i32,
    /// Number of coupled streams.
    pub coupled_streams: i32,
    /// Channel mapping.
    pub mapping: Vec<u8>,
}

impl MsEncoder {
    /// `opus_multistream_surround_encoder_create`.
    ///
    /// # Errors
    /// The C error code.
    pub fn new_surround(fs: i32, channels: i32, family: i32, application: i32) -> CResult<Self> {
        let (mut err, mut s, mut c) = (0, 0, 0);
        let mut mapping = vec![0u8; 256];
        // SAFETY: mapping has 256 entries (>= channels); out pointers valid.
        let p = unsafe {
            optopus_opus_multistream_surround_encoder_create(
                fs,
                channels,
                family,
                &mut s,
                &mut c,
                mapping.as_mut_ptr(),
                application,
                &mut err,
            )
        };
        check(err)?;
        let ptr = NonNull::new(p).ok_or(-7)?;
        mapping.truncate(channels as usize);
        Ok(Self {
            ptr,
            channels: channels as usize,
            streams: s,
            coupled_streams: c,
            mapping,
        })
    }

    /// `opus_multistream_encoder_ctl(request, (opus_int32)value)` for a setter request.
    ///
    /// # Errors
    /// The C error code.
    pub fn ctl_set(&mut self, request: i32, value: i32) -> CResult<()> {
        // SAFETY: setter requests take one opus_int32 vararg.
        check(unsafe { optopus_opus_multistream_encoder_ctl(self.ptr.as_ptr(), request, value) })
            .map(|_| ())
    }

    /// `opus_multistream_encode_float`.
    ///
    /// # Errors
    /// The C error code.
    ///
    /// # Panics
    /// If `pcm` holds fewer than `frame_size * channels` samples.
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        out: &mut [u8],
    ) -> CResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        let (fs, max) = (len_i32(frame_size)?, len_i32(out.len())?);
        // SAFETY: buffers are valid for the lengths passed.
        check(unsafe {
            optopus_opus_multistream_encode_float(
                self.ptr.as_ptr(),
                pcm.as_ptr(),
                fs,
                out.as_mut_ptr(),
                max,
            )
        })
    }
}

impl Drop for MsEncoder {
    fn drop(&mut self) {
        // SAFETY: created by the surround create call, destroyed once.
        unsafe { optopus_opus_multistream_encoder_destroy(self.ptr.as_ptr()) }
    }
}

// SAFETY: exclusively owned C state with no thread affinity.
unsafe impl Send for MsEncoder {}

/// Optimized-build `OpusMSDecoder`.
#[derive(Debug)]
pub struct MsDecoder {
    ptr: NonNull<OpusMsDecoder>,
    channels: usize,
}

impl MsDecoder {
    /// `opus_multistream_decoder_create`.
    ///
    /// # Errors
    /// The C error code.
    ///
    /// # Panics
    /// If `mapping` has fewer than `channels` entries.
    pub fn new(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled: i32,
        mapping: &[u8],
    ) -> CResult<Self> {
        assert!(mapping.len() >= channels as usize);
        let mut err = 0;
        // SAFETY: mapping has >= channels entries; err valid.
        let p = unsafe {
            optopus_opus_multistream_decoder_create(
                fs,
                channels,
                streams,
                coupled,
                mapping.as_ptr(),
                &mut err,
            )
        };
        check(err)?;
        let ptr = NonNull::new(p).ok_or(-7)?;
        Ok(Self {
            ptr,
            channels: channels as usize,
        })
    }

    /// `opus_multistream_decode_float`.
    ///
    /// # Errors
    /// The C error code.
    ///
    /// # Panics
    /// If `pcm` holds fewer than `frame_size * channels` samples.
    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: usize,
        fec: bool,
    ) -> CResult<usize> {
        assert!(pcm.len() >= frame_size * self.channels);
        let (d, l) = data_ptr(data)?;
        let fs = len_i32(frame_size)?;
        // SAFETY: buffers valid for the lengths passed.
        check(unsafe {
            optopus_opus_multistream_decode_float(
                self.ptr.as_ptr(),
                d,
                l,
                pcm.as_mut_ptr(),
                fs,
                c_int::from(fec),
            )
        })
    }
}

impl Drop for MsDecoder {
    fn drop(&mut self) {
        // SAFETY: created by opus_multistream_decoder_create, destroyed once.
        unsafe { optopus_opus_multistream_decoder_destroy(self.ptr.as_ptr()) }
    }
}

// SAFETY: exclusively owned C state with no thread affinity.
unsafe impl Send for MsDecoder {}
