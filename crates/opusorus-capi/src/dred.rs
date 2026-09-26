//! `opus.h` DRED (Deep REDundancy) decoder API: `opus_dred_decoder_*`, `opus_dred_*`,
//! `opus_decoder_dred_decode*`.
//!
//! # With the `dred` feature (libopus `ENABLE_DRED`)
//!
//! * `OpusDREDDecoder` is a registry-backed state like the other decoders (it owns the RDOVAE
//!   decoder model, [`opus::dred::DredDecoder`]). Without the `dnn-weights-embedded` feature it
//!   behaves like a libopus `USE_WEIGHTS_FILE` build: the model is loaded with
//!   `opus_dred_decoder_ctl(dec, OPUS_SET_DNN_BLOB(data, len))`, and until then
//!   `opus_dred_parse` / `opus_dred_process` return `OPUS_UNIMPLEMENTED`. With
//!   `dnn-weights-embedded` the model is bound at `opus_dred_decoder_init` / `_create`, like
//!   libopus' compiled-in weights (`OPUS_SET_DNN_BLOB` still works and replaces it).
//! * `OpusDRED` is a plain C struct with exactly libopus' layout ([`OpusDredData`]): it holds
//!   only numbers, so callers may `malloc(opus_dred_get_size())` it or copy it with `memcpy`
//!   as with libopus. The calls copy it to and from a per-thread [`opus::dred::Dred`] (the
//!   Rust API boxes its data). `opus_dred_parse` only copies back what parsing wrote, so
//!   probing packets without DRED costs no copy.
//!
//! Rust-only argument checks (C would crash): NULL states or `OpusDRED` pointers are
//! `OPUS_BAD_ARG`, and a non-positive `sampling_rate` passed to `opus_dred_parse` with a DRED
//! packet is `OPUS_BAD_ARG`. `opus_decoder_dred_decode*` with a NULL `OpusDRED` conceals like
//! a lost packet, as C does.
//!
//! # Without the `dred` feature
//!
//! Semantics of a libopus build without `ENABLE_DRED`: an `OpusDREDDecoder` can be created
//! (it holds no model), `opus_dred_decoder_ctl` answers `OPUS_UNIMPLEMENTED`,
//! `opus_dred_get_size` is 0, `opus_dred_alloc` fails with `OPUS_UNIMPLEMENTED`, and parsing,
//! processing and DRED decoding return `OPUS_UNIMPLEMENTED`.

#[cfg(feature = "dred")]
pub(crate) use with_dred::dred_decoder;
#[cfg(feature = "dred")]
pub use with_dred::{
    OpusDredData, opus_decoder_dred_decode, opus_decoder_dred_decode_float,
    opus_decoder_dred_decode24, opus_dred_alloc, opus_dred_decoder_create,
    opus_dred_decoder_destroy, opus_dred_decoder_get_size, opus_dred_decoder_init, opus_dred_free,
    opus_dred_get_size, opus_dred_parse, opus_dred_process,
};
#[cfg(not(feature = "dred"))]
pub use without_dred::{
    opus_decoder_dred_decode, opus_decoder_dred_decode_float, opus_decoder_dred_decode24,
    opus_dred_alloc, opus_dred_decoder_create, opus_dred_decoder_destroy,
    opus_dred_decoder_get_size, opus_dred_decoder_init, opus_dred_free, opus_dred_get_size,
    opus_dred_parse, opus_dred_process,
};

/// The API with DRED support (feature `dred`).
#[cfg(feature = "dred")]
mod with_dred {
    use core::cell::RefCell;
    use core::ffi::c_int;
    use core::ptr;

    use opus::Decoder;
    use opus::dred::{Dred, DredDecoder};

    use crate::decoder::decoder;
    use crate::handle::{self, Access, Kind, Object};
    use crate::types::{OpusDRED, OpusDREDDecoder, OpusDecoder};
    use crate::util::{
        CResult, OPUS_ALLOC_FAIL, OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_INVALID_STATE, OPUS_OK,
        OPUS_UNIMPLEMENTED, buf_len, byte_len, code, guard, guard_any, ret_code, set_error,
        size_to_int, slice, slice_mut,
    };

    /// `2*DRED_NUM_REDUNDANCY_FRAMES*DRED_NUM_FEATURES`.
    const FEC_FEATURES: usize = 2 * 52 * 20;
    /// `DRED_STATE_DIM`.
    const STATE_DIM: usize = 50;
    /// `(DRED_NUM_REDUNDANCY_FRAMES/2)*(DRED_LATENT_DIM+1)`.
    const LATENTS: usize = 26 * (25 + 1);

    /// The memory layout of libopus' `struct OpusDRED` (`dnn/dred_decoder.h`), which is what an
    /// `OpusDRED*` points to.
    #[repr(C)]
    #[derive(Debug)]
    pub struct OpusDredData {
        pub fec_features: [f32; FEC_FEATURES],
        pub state: [f32; STATE_DIM],
        pub latents: [f32; LATENTS],
        pub nb_latents: c_int,
        pub process_stage: c_int,
        pub dred_offset: c_int,
    }

    // The Rust `OpusDred` has the same fields, hence the same size as the C struct.
    const _: () = assert!(size_of::<OpusDredData>() == Dred::get_size());

    thread_local! {
        /// The Rust-side copy of the `OpusDRED` being worked on (see the module docs).
        static SCRATCH: RefCell<Dred> = RefCell::new(Dred::new());
    }

    /// Runs `f` on this thread's scratch [`Dred`].
    fn with_scratch<R>(f: impl FnOnce(&mut Dred) -> CResult<R>) -> CResult<R> {
        SCRATCH.with(|s| {
            let mut s = s.try_borrow_mut().map_err(|_| OPUS_INTERNAL_ERROR)?;
            f(&mut s)
        })
    }

    /// Copies a C `OpusDRED` into a [`Dred`].
    fn load(dst: &mut Dred, src: &OpusDredData) {
        let d = dst.inner_mut();
        d.fec_features.copy_from_slice(&src.fec_features);
        d.state.copy_from_slice(&src.state);
        d.latents.copy_from_slice(&src.latents);
        d.nb_latents = src.nb_latents;
        d.process_stage = src.process_stage;
        d.dred_offset = src.dred_offset;
    }

    /// Copies a [`Dred`] into a C `OpusDRED`.
    fn store(dst: &mut OpusDredData, src: &Dred) {
        let s = src.inner();
        dst.fec_features.copy_from_slice(&s.fec_features);
        dst.state.copy_from_slice(&s.state);
        dst.latents.copy_from_slice(&s.latents);
        dst.nb_latents = s.nb_latents;
        dst.process_stage = s.process_stage;
        dst.dred_offset = s.dred_offset;
    }

    /// Resolves an `OpusDREDDecoder*`.
    ///
    /// # Safety
    /// `st` is NULL or a state block from this library (see [`handle::resolve`]); the reference
    /// is only used during the current call. With `Access::Read` it must not be mutated.
    pub(crate) unsafe fn dred_decoder<'a>(
        st: *const OpusDREDDecoder,
        access: Access,
    ) -> CResult<&'a mut DredDecoder> {
        // SAFETY: caller contract.
        let r = unsafe { handle::resolve(st.cast(), &[Kind::DredDecoder], access) }?;
        // SAFETY: resolved just now; single-threaded use per the libopus contract.
        match &mut unsafe { r.entry() }.object {
            Object::DredDecoder(d) => Ok(d),
            _ => Err(OPUS_INVALID_STATE),
        }
    }

    /// A new DRED decoder and the `opus_dred_decoder_init` status: `OPUS_UNIMPLEMENTED` when
    /// compiled-in (embedded) weights lack the RDOVAE decoder, as C returns when
    /// `init_rdovaedec` fails.
    fn new_dred_decoder() -> (DredDecoder, c_int) {
        let d = DredDecoder::new();
        let status = if cfg!(feature = "dnn-weights-embedded") && !d.loaded() {
            OPUS_UNIMPLEMENTED
        } else {
            OPUS_OK
        };
        (d, status)
    }

    /// `opus_dred_decoder_get_size`.
    #[unsafe(no_mangle)]
    pub extern "C" fn opus_dred_decoder_get_size() -> c_int {
        guard(|| Ok(size_to_int(handle::block_size(DredDecoder::get_size()))))
    }

    /// `opus_dred_decoder_init`.
    ///
    /// # Safety
    /// `dec` is NULL or valid for writes of `opus_dred_decoder_get_size()` bytes.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn opus_dred_decoder_init(dec: *mut OpusDREDDecoder) -> c_int {
        guard(|| {
            let (d, status) = new_dred_decoder();
            // SAFETY: caller contract.
            unsafe { handle::install(dec.cast(), Object::DredDecoder(d)) }?;
            Ok(status)
        })
    }

    /// `opus_dred_decoder_create`.
    ///
    /// # Safety
    /// `error` is NULL or valid for one `int` write.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn opus_dred_decoder_create(error: *mut c_int) -> *mut OpusDREDDecoder {
        // SAFETY: forwarded caller contract.
        unsafe {
            handle::create(error, || {
                let (d, status) = new_dred_decoder();
                if status != OPUS_OK {
                    return Err(status);
                }
                Ok((DredDecoder::get_size(), Object::DredDecoder(d)))
            })
        }
    }

    /// `opus_dred_decoder_destroy`.
    ///
    /// # Safety
    /// `dec` is NULL or a DRED decoder on the C heap, not used afterwards.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn opus_dred_decoder_destroy(dec: *mut OpusDREDDecoder) {
        // SAFETY: caller contract.
        guard_any((), || unsafe { handle::destroy(dec.cast()) });
    }

    /// `opus_dred_get_size`: `sizeof(OpusDRED)`.
    #[unsafe(no_mangle)]
    pub const extern "C" fn opus_dred_get_size() -> c_int {
        size_of::<OpusDredData>() as c_int
    }

    /// `opus_dred_alloc`: a zeroed `OpusDRED` on the C heap (C leaves it uninitialized and, like
    /// C, `*error` is only written on failure).
    ///
    /// # Safety
    /// `error` is NULL or valid for one `int` write.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn opus_dred_alloc(error: *mut c_int) -> *mut OpusDRED {
        let block = handle::c_alloc(size_of::<OpusDredData>());
        if block.is_null() {
            // SAFETY: caller contract.
            unsafe { set_error(error, OPUS_ALLOC_FAIL) };
        }
        block.cast()
    }

    /// `opus_dred_free`.
    ///
    /// # Safety
    /// `dec` is NULL or an `OpusDRED` on the C heap, not used afterwards.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn opus_dred_free(dec: *mut OpusDRED) {
        // SAFETY: caller contract.
        unsafe { handle::c_free(dec.cast()) };
    }

    /// `opus_dred_parse`: extracts the DRED redundancy of a packet into `dred`; returns the
    /// offset (in samples at `sampling_rate`) of the oldest sample it can reconstruct, 0
    /// without DRED, or an error code.
    ///
    /// # Safety
    /// `dred_dec` is a DRED decoder; `dred` points to an `OpusDRED`; `data` holds `len` bytes;
    /// `dred_end` is NULL or valid for one `int` write.
    #[unsafe(no_mangle)]
    #[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
    pub unsafe extern "C" fn opus_dred_parse(
        dred_dec: *mut OpusDREDDecoder,
        dred: *mut OpusDRED,
        data: *const u8,
        len: i32,
        max_dred_samples: i32,
        sampling_rate: i32,
        dred_end: *mut c_int,
        defer_processing: c_int,
    ) -> c_int {
        guard(|| {
            // SAFETY: caller contract; only read.
            let dd = unsafe { dred_decoder(dred_dec, Access::Read) }?;
            if dred.is_null() {
                return Err(OPUS_BAD_ARG);
            }
            // SAFETY: caller contract (an `OpusDRED`, which has this layout).
            let cd = unsafe { &mut *dred.cast::<OpusDredData>() };
            if !dd.loaded() {
                return Err(OPUS_UNIMPLEMENTED);
            }
            // C sets the stage before parsing the packet (which rejects a negative length).
            cd.process_stage = -1;
            // SAFETY: caller contract (`len` bytes).
            let data = unsafe { slice(data, byte_len(len)?) }?;
            with_scratch(|s| {
                let r = dd.parse(
                    s,
                    data,
                    max_dred_samples,
                    sampling_rate,
                    defer_processing != 0,
                );
                // Parsing sets the stage to -1 first; the other fields are only written when
                // DRED data was decoded.
                if s.process_stage() != -1 {
                    store(cd, s);
                }
                let (offset, end) = r.map_err(code)?;
                // SAFETY: caller contract.
                unsafe { set_error(dred_end, end) };
                Ok(offset)
            })
        })
    }

    /// `opus_dred_process`: decodes the parsed latents of `src` into features, in `dst` (which
    /// may be `src`).
    ///
    /// # Safety
    /// `dred_dec` is a DRED decoder; `src` and `dst` point to `OpusDRED`s.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn opus_dred_process(
        dred_dec: *mut OpusDREDDecoder,
        src: *const OpusDRED,
        dst: *mut OpusDRED,
    ) -> c_int {
        guard(|| {
            if dred_dec.is_null() || src.is_null() || dst.is_null() {
                return Err(OPUS_BAD_ARG);
            }
            let src = src.cast::<OpusDredData>();
            let dst = dst.cast::<OpusDredData>();
            // SAFETY: caller contract (an `OpusDRED`).
            let stage = unsafe { (*src).process_stage };
            if stage != 1 && stage != 2 {
                return Err(OPUS_BAD_ARG);
            }
            // SAFETY: caller contract; only read.
            let dd = unsafe { dred_decoder(dred_dec, Access::Read) }?;
            if !dd.loaded() {
                return Err(OPUS_UNIMPLEMENTED);
            }
            if stage == 2 {
                // Already processed: C copies and returns.
                if !ptr::eq(src, dst) {
                    // SAFETY: caller contract; two `OpusDRED`s (distinct objects never
                    // overlap).
                    unsafe { ptr::copy(src, dst, 1) };
                }
                return Ok(OPUS_OK);
            }
            with_scratch(|s| {
                // SAFETY: caller contract; the shared borrow ends before `dst` is written.
                load(s, unsafe { &*src });
                dd.process_in_place(s).map_err(code)?;
                // SAFETY: caller contract.
                store(unsafe { &mut *dst }, s);
                Ok(OPUS_OK)
            })
        })
    }

    /// Shared body of `opus_decoder_dred_decode*`: `decode(decoder, dred, dred_offset, pcm,
    /// frame_size)` with `pcm` of `frame_size * channels` samples.
    ///
    /// # Safety
    /// `st` is a decoder; `dred` is NULL or points to an `OpusDRED`; `pcm` holds
    /// `frame_size * channels` samples.
    unsafe fn dred_decode_with<T>(
        st: *mut OpusDecoder,
        dred: *const OpusDRED,
        dred_offset: i32,
        pcm: *mut T,
        frame_size: i32,
        decode: impl FnOnce(&mut Decoder, &Dred, i32, &mut [T], i32) -> opus::Result<i32>,
    ) -> c_int {
        guard(|| {
            // SAFETY: caller contract.
            let dec = unsafe { decoder(st, Access::Write) }?;
            if frame_size <= 0 {
                return Err(OPUS_BAD_ARG);
            }
            let n = buf_len::<T>(frame_size, dec.channels() as c_int)?;
            // SAFETY: caller contract (`frame_size * channels` samples).
            let pcm = unsafe { slice_mut(pcm, n) }?;
            with_scratch(|s| {
                let dred = dred.cast::<OpusDredData>();
                // C only uses processed data (`dred != NULL && dred->process_stage == 2`);
                // anything else is plain packet-loss concealment.
                // SAFETY: caller contract (NULL or an `OpusDRED`).
                if !dred.is_null() && unsafe { (*dred).process_stage } == 2 {
                    // SAFETY: as above.
                    load(s, unsafe { &*dred });
                } else {
                    s.inner_mut().process_stage = -1;
                }
                ret_code(decode(dec, s, dred_offset, pcm, frame_size), |n| n)
            })
        })
    }

    /// `opus_decoder_dred_decode`: conceals `frame_size` samples from DRED data (16-bit).
    ///
    /// # Safety
    /// `st` is a decoder; `dred` is NULL or points to an `OpusDRED`; `pcm` holds
    /// `frame_size * channels` samples.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn opus_decoder_dred_decode(
        st: *mut OpusDecoder,
        dred: *const OpusDRED,
        dred_offset: i32,
        pcm: *mut i16,
        frame_size: i32,
    ) -> c_int {
        // SAFETY: forwarded caller contract.
        unsafe {
            dred_decode_with(st, dred, dred_offset, pcm, frame_size, |d, r, o, p, f| {
                d.opus_decoder_dred_decode(r, o, p, f)
            })
        }
    }

    /// `opus_decoder_dred_decode24`: as [`opus_decoder_dred_decode`], 24-bit in `opus_int32`.
    ///
    /// # Safety
    /// As [`opus_decoder_dred_decode`].
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn opus_decoder_dred_decode24(
        st: *mut OpusDecoder,
        dred: *const OpusDRED,
        dred_offset: i32,
        pcm: *mut i32,
        frame_size: i32,
    ) -> c_int {
        // SAFETY: forwarded caller contract.
        unsafe {
            dred_decode_with(st, dred, dred_offset, pcm, frame_size, |d, r, o, p, f| {
                d.opus_decoder_dred_decode24(r, o, p, f)
            })
        }
    }

    /// `opus_decoder_dred_decode_float`: as [`opus_decoder_dred_decode`], float output.
    ///
    /// # Safety
    /// As [`opus_decoder_dred_decode`].
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn opus_decoder_dred_decode_float(
        st: *mut OpusDecoder,
        dred: *const OpusDRED,
        dred_offset: i32,
        pcm: *mut f32,
        frame_size: i32,
    ) -> c_int {
        // SAFETY: forwarded caller contract.
        unsafe {
            dred_decode_with(st, dred, dred_offset, pcm, frame_size, |d, r, o, p, f| {
                d.opus_decoder_dred_decode_float(r, o, p, f)
            })
        }
    }
}

/// The API of a libopus build without `ENABLE_DRED`.
#[cfg(not(feature = "dred"))]
mod without_dred {
    use core::ffi::c_int;
    use core::ptr;

    use crate::handle::{self, Kind};
    use crate::types::{OpusDRED, OpusDREDDecoder, OpusDecoder};
    use crate::util::{OPUS_ALLOC_FAIL, OPUS_OK, OPUS_UNIMPLEMENTED, guard, guard_any, set_error};

    /// `opus_dred_decoder_get_size`.
    #[unsafe(no_mangle)]
    pub const extern "C" fn opus_dred_decoder_get_size() -> c_int {
        handle::HEADER_SIZE as c_int
    }

    /// `opus_dred_decoder_init`.
    ///
    /// # Safety
    /// `dec` is valid for writes of `opus_dred_decoder_get_size()` bytes.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn opus_dred_decoder_init(dec: *mut OpusDREDDecoder) -> c_int {
        guard(|| {
            if dec.is_null() {
                return Err(crate::util::OPUS_BAD_ARG);
            }
            // SAFETY: caller contract.
            unsafe { handle::write_plain_header(dec.cast(), Kind::DredDecoder) };
            Ok(OPUS_OK)
        })
    }

    /// `opus_dred_decoder_create`.
    ///
    /// # Safety
    /// `error` is NULL or valid for one `int` write.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn opus_dred_decoder_create(error: *mut c_int) -> *mut OpusDREDDecoder {
        guard_any(ptr::null_mut(), || {
            let block = handle::c_alloc(handle::HEADER_SIZE);
            if block.is_null() {
                // SAFETY: caller contract.
                unsafe { set_error(error, OPUS_ALLOC_FAIL) };
                return ptr::null_mut();
            }
            // SAFETY: fresh allocation of `HEADER_SIZE` bytes.
            unsafe { handle::write_plain_header(block, Kind::DredDecoder) };
            // SAFETY: caller contract.
            unsafe { set_error(error, OPUS_OK) };
            block.cast()
        })
    }

    /// `opus_dred_decoder_destroy`.
    ///
    /// # Safety
    /// `dec` is NULL or a DRED decoder on the C heap, not used afterwards.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn opus_dred_decoder_destroy(dec: *mut OpusDREDDecoder) {
        // SAFETY: caller contract (no registry object behind a DRED decoder).
        guard_any((), || unsafe { handle::c_free(dec.cast()) });
    }

    /// `opus_dred_get_size`: 0 without DRED support.
    #[unsafe(no_mangle)]
    pub const extern "C" fn opus_dred_get_size() -> c_int {
        0
    }

    /// `opus_dred_alloc`: fails with `OPUS_UNIMPLEMENTED` without DRED support.
    ///
    /// # Safety
    /// `error` is NULL or valid for one `int` write.
    #[unsafe(no_mangle)]
    pub const unsafe extern "C" fn opus_dred_alloc(error: *mut c_int) -> *mut OpusDRED {
        // SAFETY: caller contract.
        unsafe { set_error(error, OPUS_UNIMPLEMENTED) };
        ptr::null_mut()
    }

    /// `opus_dred_free`: nothing to free without DRED support.
    #[unsafe(no_mangle)]
    pub const extern "C" fn opus_dred_free(_dec: *mut OpusDRED) {}

    /// `opus_dred_parse`: `OPUS_UNIMPLEMENTED` without DRED support.
    #[unsafe(no_mangle)]
    #[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
    pub const extern "C" fn opus_dred_parse(
        _dred_dec: *mut OpusDREDDecoder,
        _dred: *mut OpusDRED,
        _data: *const u8,
        _len: i32,
        _max_dred_samples: i32,
        _sampling_rate: i32,
        _dred_end: *mut c_int,
        _defer_processing: c_int,
    ) -> c_int {
        OPUS_UNIMPLEMENTED
    }

    /// `opus_dred_process`: `OPUS_UNIMPLEMENTED` without DRED support.
    #[unsafe(no_mangle)]
    pub const extern "C" fn opus_dred_process(
        _dred_dec: *mut OpusDREDDecoder,
        _src: *const OpusDRED,
        _dst: *mut OpusDRED,
    ) -> c_int {
        OPUS_UNIMPLEMENTED
    }

    /// `opus_decoder_dred_decode`: `OPUS_UNIMPLEMENTED` without DRED support.
    #[unsafe(no_mangle)]
    pub const extern "C" fn opus_decoder_dred_decode(
        _st: *mut OpusDecoder,
        _dred: *const OpusDRED,
        _dred_offset: i32,
        _pcm: *mut i16,
        _frame_size: i32,
    ) -> c_int {
        OPUS_UNIMPLEMENTED
    }

    /// `opus_decoder_dred_decode24`: `OPUS_UNIMPLEMENTED` without DRED support.
    #[unsafe(no_mangle)]
    pub const extern "C" fn opus_decoder_dred_decode24(
        _st: *mut OpusDecoder,
        _dred: *const OpusDRED,
        _dred_offset: i32,
        _pcm: *mut i32,
        _frame_size: i32,
    ) -> c_int {
        OPUS_UNIMPLEMENTED
    }

    /// `opus_decoder_dred_decode_float`: `OPUS_UNIMPLEMENTED` without DRED support.
    #[unsafe(no_mangle)]
    pub const extern "C" fn opus_decoder_dred_decode_float(
        _st: *mut OpusDecoder,
        _dred: *const OpusDRED,
        _dred_offset: i32,
        _pcm: *mut f32,
        _frame_size: i32,
    ) -> c_int {
        OPUS_UNIMPLEMENTED
    }
}
