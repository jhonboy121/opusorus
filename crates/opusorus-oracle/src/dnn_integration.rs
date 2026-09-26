//! Oracle bindings for unit `dnn_integration`: the DNN features (deep PLC, DRED, OSCE) wired into
//! the C Opus decoder/encoder, and the public `opus_dred_*` API.
//!
//! * Weight blobs serialized from the oracle's compiled-in model tables
//!   ([`decoder_blob`], [`encoder_blob`], [`dred_decoder_blob`], [`all_blob`]): the Rust port has
//!   no compiled-in weights, so tests load these into it to run the same models as the C oracle.
//! * [`Dec`]: a C `OpusDecoder` with the DNN state dump ([`DnnDecState`]) and
//!   `opus_decoder_dred_decode*`.
//! * [`DredDec`] / [`Dred`]: C `OpusDREDDecoder` / `OpusDRED` (`opus_dred_parse`,
//!   `opus_dred_process`, and a dump of the parsed data).
//!
//! Only compiled with a DNN feature (`deep-plc`, `dred` or `osce`).
#![cfg(any(feature = "deep-plc", feature = "dred", feature = "osce"))]

use crate::api::OResult;
use crate::dnn_core;
use crate::sys;
use core::ffi::c_int;
use core::ptr::NonNull;
use std::sync::OnceLock;

/// Opaque C `OpusDREDDecoder`.
#[repr(C)]
#[derive(Debug)]
pub struct OpusDREDDecoder {
    _p: [u8; 0],
}

/// Opaque C `OpusDRED`.
#[repr(C)]
#[derive(Debug)]
pub struct OpusDRED {
    _p: [u8; 0],
}

unsafe extern "C" {
    fn opus_dred_decoder_create(error: *mut c_int) -> *mut OpusDREDDecoder;
    fn opus_dred_decoder_destroy(dec: *mut OpusDREDDecoder);
    fn opus_dred_decoder_get_size() -> c_int;
    fn opus_dred_decoder_ctl(dec: *mut OpusDREDDecoder, request: c_int, ...) -> c_int;
    fn opus_dred_get_size() -> c_int;
    fn opus_dred_alloc(error: *mut c_int) -> *mut OpusDRED;
    fn opus_dred_free(dec: *mut OpusDRED);
    fn opus_dred_parse(
        dred_dec: *mut OpusDREDDecoder,
        dred: *mut OpusDRED,
        data: *const u8,
        len: i32,
        max_dred_samples: i32,
        sampling_rate: i32,
        dred_end: *mut c_int,
        defer_processing: c_int,
    ) -> c_int;
    fn opus_dred_process(
        dred_dec: *mut OpusDREDDecoder,
        src: *const OpusDRED,
        dst: *mut OpusDRED,
    ) -> c_int;
    fn opus_decoder_dred_decode(
        st: *mut sys::OpusDecoder,
        dred: *const OpusDRED,
        dred_offset: i32,
        pcm: *mut i16,
        frame_size: i32,
    ) -> c_int;
    fn opus_decoder_dred_decode24(
        st: *mut sys::OpusDecoder,
        dred: *const OpusDRED,
        dred_offset: i32,
        pcm: *mut i32,
        frame_size: i32,
    ) -> c_int;
    fn opus_decoder_dred_decode_float(
        st: *mut sys::OpusDecoder,
        dred: *const OpusDRED,
        dred_offset: i32,
        pcm: *mut f32,
        frame_size: i32,
    ) -> c_int;

    fn oracle_di_dec_dump(dec: *const sys::OpusDecoder, iout: *mut c_int, fout: *mut f32) -> c_int;
    fn oracle_di_has_dred() -> c_int;
    fn oracle_di_has_osce() -> c_int;
    fn oracle_di_dred_sizes(out: *mut c_int);
    fn oracle_di_dred_dump(
        d: *const OpusDRED,
        fec: *mut f32,
        state: *mut f32,
        latents: *mut f32,
        ints: *mut c_int,
    );
    fn oracle_di_dred_clear(d: *mut OpusDRED);
}

const fn check(ret: c_int) -> OResult<usize> {
    if ret < 0 { Err(ret) } else { Ok(ret as usize) }
}

/// Whether the oracle was built with `ENABLE_DRED`.
#[must_use]
pub fn has_dred() -> bool {
    // SAFETY: pure function.
    unsafe { oracle_di_has_dred() != 0 }
}

/// Whether the oracle was built with `ENABLE_OSCE` (+ `ENABLE_OSCE_BWE`).
#[must_use]
pub fn has_osce() -> bool {
    // SAFETY: pure function.
    unsafe { oracle_di_has_osce() != 0 }
}

/// Concatenation of the blobs of every `model` compiled into the oracle.
fn blob_of(models: &[i32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &m in models {
        if dnn_core::model_count(m).is_some() {
            out.extend_from_slice(&dnn_core::write_blob(m));
        }
    }
    assert!(!out.is_empty(), "no DNN model compiled into the oracle");
    out
}

/// Weight blob with every model the Opus decoder uses (pitch DNN, PLC, FARGAN; LACE, NoLACE,
/// BBWENet with `osce`), serialized from the oracle's compiled-in tables.
pub fn decoder_blob() -> &'static [u8] {
    static B: OnceLock<Vec<u8>> = OnceLock::new();
    B.get_or_init(|| {
        blob_of(&[
            dnn_core::MODEL_PITCHDNN,
            dnn_core::MODEL_PLC,
            dnn_core::MODEL_FARGAN,
            dnn_core::MODEL_LACE,
            dnn_core::MODEL_NOLACE,
            dnn_core::MODEL_BBWENET,
        ])
    })
}

/// Weight blob with the models of the Opus encoder's DRED encoder (pitch DNN, RDOVAE encoder).
pub fn encoder_blob() -> &'static [u8] {
    static B: OnceLock<Vec<u8>> = OnceLock::new();
    B.get_or_init(|| blob_of(&[dnn_core::MODEL_PITCHDNN, dnn_core::MODEL_RDOVAE_ENC]))
}

/// Weight blob with the RDOVAE decoder (`OpusDREDDecoder`).
pub fn dred_decoder_blob() -> &'static [u8] {
    static B: OnceLock<Vec<u8>> = OnceLock::new();
    B.get_or_init(|| blob_of(&[dnn_core::MODEL_RDOVAE_DEC]))
}

/// Weight blob with every model compiled into the oracle.
pub fn all_blob() -> &'static [u8] {
    static B: OnceLock<Vec<u8>> = OnceLock::new();
    B.get_or_init(|| {
        blob_of(&[
            dnn_core::MODEL_PITCHDNN,
            dnn_core::MODEL_PLC,
            dnn_core::MODEL_FARGAN,
            dnn_core::MODEL_RDOVAE_ENC,
            dnn_core::MODEL_RDOVAE_DEC,
            dnn_core::MODEL_LACE,
            dnn_core::MODEL_NOLACE,
            dnn_core::MODEL_BBWENET,
        ])
    })
}

/// Number of ints of [`DnnDecState::ints`].
pub const DEC_INTS: usize = 13;

/// DNN state of a C `OpusDecoder` (see `oracle_di_dec_dump`): `ints` = lpcnet `{loaded,
/// analysis_gap, fec_read_pos, fec_fill_pos, fec_skip, analysis_pos, predict_pos, blend,
/// loss_count}` + DecControl `{osce_method, enable_osce_bwe, osce_extended_mode,
/// prev_osce_extended_mode}`; `floats` = lpcnet `pcm` then `features`.
#[derive(Debug, Clone, PartialEq)]
pub struct DnnDecState {
    pub ints: [i32; DEC_INTS],
    pub floats: Vec<f32>,
}

/// C `OpusDecoder` (library build) with the DNN state dump and `opus_decoder_dred_decode*`.
#[derive(Debug)]
pub struct Dec {
    ptr: NonNull<sys::OpusDecoder>,
    channels: usize,
}

// SAFETY: the C state is owned exclusively by this handle.
unsafe impl Send for Dec {}

fn data_ptr(data: Option<&[u8]>) -> (*const u8, i32) {
    match data {
        Some(d) => (d.as_ptr(), d.len() as i32),
        None => (core::ptr::null(), 0),
    }
}

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

    /// `opus_decode`.
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        fec: i32,
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size.max(0) as usize * self.channels);
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
        assert!(pcm.len() >= frame_size.max(0) as usize * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid for the lengths passed.
        check(unsafe {
            sys::opus_decode24(self.ptr.as_ptr(), d, l, pcm.as_mut_ptr(), frame_size, fec)
        })
    }

    /// `opus_decode_float`.
    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: i32,
        fec: i32,
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size.max(0) as usize * self.channels);
        let (d, l) = data_ptr(data);
        // SAFETY: buffers valid for the lengths passed.
        check(unsafe {
            sys::opus_decode_float(self.ptr.as_ptr(), d, l, pcm.as_mut_ptr(), frame_size, fec)
        })
    }

    /// `opus_decoder_dred_decode`.
    pub fn dred_decode(
        &mut self,
        dred: &Dred,
        dred_offset: i32,
        pcm: &mut [i16],
        frame_size: i32,
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size.max(0) as usize * self.channels);
        // SAFETY: valid states; pcm holds frame_size*channels samples.
        check(unsafe {
            opus_decoder_dred_decode(
                self.ptr.as_ptr(),
                dred.ptr.as_ptr(),
                dred_offset,
                pcm.as_mut_ptr(),
                frame_size,
            )
        })
    }

    /// `opus_decoder_dred_decode24`.
    pub fn dred_decode24(
        &mut self,
        dred: &Dred,
        dred_offset: i32,
        pcm: &mut [i32],
        frame_size: i32,
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size.max(0) as usize * self.channels);
        // SAFETY: valid states; pcm holds frame_size*channels samples.
        check(unsafe {
            opus_decoder_dred_decode24(
                self.ptr.as_ptr(),
                dred.ptr.as_ptr(),
                dred_offset,
                pcm.as_mut_ptr(),
                frame_size,
            )
        })
    }

    /// `opus_decoder_dred_decode_float`.
    pub fn dred_decode_float(
        &mut self,
        dred: &Dred,
        dred_offset: i32,
        pcm: &mut [f32],
        frame_size: i32,
    ) -> OResult<usize> {
        assert!(pcm.len() >= frame_size.max(0) as usize * self.channels);
        // SAFETY: valid states; pcm holds frame_size*channels samples.
        check(unsafe {
            opus_decoder_dred_decode_float(
                self.ptr.as_ptr(),
                dred.ptr.as_ptr(),
                dred_offset,
                pcm.as_mut_ptr(),
                frame_size,
            )
        })
    }

    /// Issues a CTL taking one `i32` argument.
    pub fn ctl_set(&mut self, request: i32, value: i32) -> OResult<()> {
        // SAFETY: valid state; the request takes an opus_int32 by value.
        check(unsafe { sys::opus_decoder_ctl(self.ptr.as_ptr(), request, value) }).map(drop)
    }

    /// Issues a CTL writing one `i32`.
    pub fn ctl_get(&mut self, request: i32) -> OResult<i32> {
        let mut v: i32 = 0;
        // SAFETY: valid state; the request writes one opus_int32.
        check(unsafe { sys::opus_decoder_ctl(self.ptr.as_ptr(), request, &mut v as *mut i32) })?;
        Ok(v)
    }

    /// `OPUS_GET_FINAL_RANGE`.
    pub fn final_range(&mut self) -> u32 {
        let mut v: u32 = 0;
        // SAFETY: valid state; the request writes one opus_uint32.
        let r = unsafe {
            sys::opus_decoder_ctl(
                self.ptr.as_ptr(),
                sys::OPUS_GET_FINAL_RANGE_REQUEST,
                &mut v as *mut u32,
            )
        };
        assert_eq!(r, 0);
        v
    }

    /// DNN state dump.
    #[must_use]
    pub fn dump(&self) -> DnnDecState {
        let mut ints = [0 as c_int; DEC_INTS];
        // SAFETY: NULL fout only returns the float count.
        let nf = unsafe {
            oracle_di_dec_dump(self.ptr.as_ptr(), ints.as_mut_ptr(), core::ptr::null_mut())
        };
        let mut floats = vec![0f32; nf as usize];
        // SAFETY: ints has DEC_INTS entries, floats the returned count.
        unsafe { oracle_di_dec_dump(self.ptr.as_ptr(), ints.as_mut_ptr(), floats.as_mut_ptr()) };
        DnnDecState { ints, floats }
    }
}

impl Drop for Dec {
    fn drop(&mut self) {
        // SAFETY: pointer from opus_decoder_create, destroyed once.
        unsafe { sys::opus_decoder_destroy(self.ptr.as_ptr()) }
    }
}

/// C `OpusDREDDecoder` (compiled-in RDOVAE decoder).
#[derive(Debug)]
pub struct DredDec {
    ptr: NonNull<OpusDREDDecoder>,
}

// SAFETY: the C state is owned exclusively by this handle.
unsafe impl Send for DredDec {}

impl DredDec {
    /// `opus_dred_decoder_create`.
    pub fn new() -> OResult<Self> {
        let mut err = 0;
        // SAFETY: err is a valid out pointer.
        let p = unsafe { opus_dred_decoder_create(&mut err) };
        check(err)?;
        let ptr = NonNull::new(p).ok_or(sys::OPUS_ALLOC_FAIL)?;
        Ok(Self { ptr })
    }

    /// `opus_dred_decoder_get_size`.
    #[must_use]
    pub fn get_size() -> i32 {
        // SAFETY: pure function.
        unsafe { opus_dred_decoder_get_size() }
    }

    /// `opus_dred_decoder_ctl` with one `i32` argument.
    pub fn ctl_set(&mut self, request: i32, value: i32) -> OResult<()> {
        // SAFETY: valid state; the request takes an opus_int32 by value.
        check(unsafe { opus_dred_decoder_ctl(self.ptr.as_ptr(), request, value) }).map(drop)
    }

    /// `opus_dred_parse`: returns `(return value, dred_end)`.
    pub fn parse(
        &mut self,
        dred: &mut Dred,
        data: &[u8],
        max_dred_samples: i32,
        sampling_rate: i32,
        defer_processing: bool,
    ) -> OResult<(i32, i32)> {
        let mut end: c_int = -12345;
        // SAFETY: valid states and buffer.
        let r = unsafe {
            opus_dred_parse(
                self.ptr.as_ptr(),
                dred.ptr.as_ptr(),
                data.as_ptr(),
                data.len() as i32,
                max_dred_samples,
                sampling_rate,
                &mut end,
                c_int::from(defer_processing),
            )
        };
        check(r)?;
        Ok((r, end))
    }

    /// `opus_dred_process(src, dst)`.
    pub fn process(&mut self, src: &Dred, dst: &mut Dred) -> OResult<()> {
        // SAFETY: valid, distinct states.
        check(unsafe { opus_dred_process(self.ptr.as_ptr(), src.ptr.as_ptr(), dst.ptr.as_ptr()) })
            .map(drop)
    }

    /// `opus_dred_process(d, d)` (in place).
    pub fn process_in_place(&mut self, d: &mut Dred) -> OResult<()> {
        // SAFETY: valid states; C handles src == dst.
        check(unsafe { opus_dred_process(self.ptr.as_ptr(), d.ptr.as_ptr(), d.ptr.as_ptr()) })
            .map(drop)
    }
}

impl Drop for DredDec {
    fn drop(&mut self) {
        // SAFETY: pointer from opus_dred_decoder_create, destroyed once.
        unsafe { opus_dred_decoder_destroy(self.ptr.as_ptr()) }
    }
}

/// Dump of a C `OpusDRED`.
#[derive(Debug, Clone, PartialEq)]
pub struct DredState {
    pub fec_features: Vec<f32>,
    pub state: Vec<f32>,
    pub latents: Vec<f32>,
    /// `{nb_latents, process_stage, dred_offset}`.
    pub ints: [i32; 3],
}

/// C `OpusDRED` (zeroed at allocation for reproducible comparisons).
#[derive(Debug)]
pub struct Dred {
    ptr: NonNull<OpusDRED>,
}

// SAFETY: the C state is owned exclusively by this handle.
unsafe impl Send for Dred {}

impl Dred {
    /// `opus_dred_alloc` + zeroing.
    pub fn new() -> OResult<Self> {
        let mut err = 0;
        // SAFETY: err is a valid out pointer.
        let p = unsafe { opus_dred_alloc(&mut err) };
        let ptr = NonNull::new(p).ok_or(err)?;
        // SAFETY: freshly allocated OpusDRED.
        unsafe { oracle_di_dred_clear(ptr.as_ptr()) };
        Ok(Self { ptr })
    }

    /// `opus_dred_get_size`.
    #[must_use]
    pub fn get_size() -> i32 {
        // SAFETY: pure function.
        unsafe { opus_dred_get_size() }
    }

    /// Dumps the struct.
    #[must_use]
    pub fn dump(&self) -> DredState {
        let mut sizes = [0 as c_int; 3];
        // SAFETY: writes 3 ints.
        unsafe { oracle_di_dred_sizes(sizes.as_mut_ptr()) };
        let mut fec = vec![0f32; sizes[0] as usize];
        let mut state = vec![0f32; sizes[1] as usize];
        let mut latents = vec![0f32; sizes[2] as usize];
        let mut ints = [0 as c_int; 3];
        // SAFETY: buffers sized from oracle_di_dred_sizes.
        unsafe {
            oracle_di_dred_dump(
                self.ptr.as_ptr(),
                fec.as_mut_ptr(),
                state.as_mut_ptr(),
                latents.as_mut_ptr(),
                ints.as_mut_ptr(),
            );
        }
        DredState {
            fec_features: fec,
            state,
            latents,
            ints,
        }
    }
}

impl Drop for Dred {
    fn drop(&mut self) {
        // SAFETY: pointer from opus_dred_alloc, freed once.
        unsafe { opus_dred_free(self.ptr.as_ptr()) }
    }
}
