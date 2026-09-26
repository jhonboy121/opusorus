//! Oracle bindings for unit `dnn_core`: dnn/nnet.c, nnet_arch.h (generic C path), vec.h,
//! common.h, kiss99.c, parse_lpcnet_weights.c, write_lpcnet_weights.c, burg.c, freq.c,
//! lpcnet_tables.c, pitchdnn.c, lpcnet_enc.c and (osce) nndsp.c.
//!
//! Only available when the oracle is built with a DNN feature (`deep-plc`, `dred` or `osce`,
//! see `build.rs`). Layers and states live on the C heap behind RAII handles; every wrapper
//! checks the sizes the C code will read or write before calling it.
#![cfg(any(feature = "deep-plc", feature = "dred", feature = "osce"))]

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::ffi::CString;

/// Model ids for the compiled-in weight tables.
pub const MODEL_PITCHDNN: i32 = 0;
/// `plcmodel_arrays`.
pub const MODEL_PLC: i32 = 1;
/// `fargan_arrays`.
pub const MODEL_FARGAN: i32 = 2;
/// `rdovaeenc_arrays` (dred).
pub const MODEL_RDOVAE_ENC: i32 = 3;
/// `rdovaedec_arrays` (dred).
pub const MODEL_RDOVAE_DEC: i32 = 4;
/// `lacelayers_arrays` (osce).
pub const MODEL_LACE: i32 = 5;
/// `nolacelayers_arrays` (osce).
pub const MODEL_NOLACE: i32 = 6;
/// `bbwenetlayers_arrays` (osce).
pub const MODEL_BBWENET: i32 = 7;

unsafe extern "C" {
    fn oracle_dc_model_count(model: c_int) -> c_int;
    fn oracle_dc_model_array_info(
        model: c_int,
        i: c_int,
        name: *mut u8,
        ty: *mut c_int,
        size: *mut c_int,
    );
    fn oracle_dc_model_array_data(model: c_int, i: c_int, out: *mut c_void);
    fn oracle_dc_write_blob(model: c_int, out: *mut u8) -> c_int;
    fn oracle_dc_parse_weights(data: *const u8, len: c_int, info: *mut c_int, max: c_int) -> c_int;

    fn oracle_dc_linear_new(
        bias: *const f32,
        subias: *const f32,
        weights: *const i8,
        nweights: c_int,
        float_weights: *const f32,
        nfloat: c_int,
        idx: *const c_int,
        nidx: c_int,
        diag: *const f32,
        scale: *const f32,
        nb_in: c_int,
        nb_out: c_int,
    ) -> *mut c_void;
    fn oracle_dc_linear_init(
        model: c_int,
        bias: *const c_char,
        subias: *const c_char,
        weights: *const c_char,
        float_weights: *const c_char,
        weights_idx: *const c_char,
        diag: *const c_char,
        scale: *const c_char,
        nb_in: c_int,
        nb_out: c_int,
        ret: *mut c_int,
    ) -> *mut c_void;
    fn oracle_dc_linear_free(p: *mut c_void);
    fn oracle_dc_linear_field(p: *mut c_void, f: c_int, out: *mut c_void, nbytes: c_int) -> c_int;
    fn oracle_dc_linear_dims(p: *mut c_void, nb_in: *mut c_int, nb_out: *mut c_int);
    fn oracle_dc_compute_linear(p: *mut c_void, out: *mut f32, input: *const f32);
    fn oracle_dc_compute_generic_dense(
        p: *mut c_void,
        out: *mut f32,
        input: *const f32,
        act: c_int,
    );
    fn oracle_dc_compute_generic_gru(
        pin: *mut c_void,
        prec: *mut c_void,
        state: *mut f32,
        input: *const f32,
    );
    fn oracle_dc_compute_glu(p: *mut c_void, out: *mut f32, input: *const f32, inplace: c_int);
    fn oracle_dc_compute_generic_conv1d(
        p: *mut c_void,
        out: *mut f32,
        mem: *mut f32,
        input: *const f32,
        input_size: c_int,
        act: c_int,
    );
    fn oracle_dc_compute_generic_conv1d_dilation(
        p: *mut c_void,
        out: *mut f32,
        mem: *mut f32,
        input: *const f32,
        input_size: c_int,
        dilation: c_int,
        act: c_int,
    );
    fn oracle_dc_compute_activation(
        out: *mut f32,
        input: *const f32,
        n: c_int,
        act: c_int,
        inplace: c_int,
    );
    fn oracle_dc_vec_swish(y: *mut f32, x: *const f32, n: c_int);

    fn oracle_dc_conv2d_new(
        bias: *const f32,
        w: *const f32,
        in_ch: c_int,
        out_ch: c_int,
        ktime: c_int,
        kheight: c_int,
    ) -> *mut c_void;
    fn oracle_dc_conv2d_init(
        model: c_int,
        bias: *const c_char,
        float_weights: *const c_char,
        in_ch: c_int,
        out_ch: c_int,
        ktime: c_int,
        kheight: c_int,
        ret: *mut c_int,
    ) -> *mut c_void;
    fn oracle_dc_conv2d_free(p: *mut c_void);
    fn oracle_dc_conv2d_field(p: *mut c_void, f: c_int, out: *mut c_void, nbytes: c_int) -> c_int;
    fn oracle_dc_compute_conv2d(
        p: *mut c_void,
        out: *mut f32,
        mem: *mut f32,
        input: *const f32,
        height: c_int,
        hstride: c_int,
        act: c_int,
    );
    fn oracle_dc_conv2d_float(
        out: *mut f32,
        w: *const f32,
        in_ch: c_int,
        out_ch: c_int,
        ktime: c_int,
        kheight: c_int,
        input: *const f32,
        height: c_int,
        hstride: c_int,
    );
    fn oracle_dc_conv2d_3x3_float(
        out: *mut f32,
        w: *const f32,
        in_ch: c_int,
        out_ch: c_int,
        input: *const f32,
        height: c_int,
        hstride: c_int,
    );

    fn oracle_dc_sgemv(
        out: *mut f32,
        w: *const f32,
        rows: c_int,
        cols: c_int,
        stride: c_int,
        x: *const f32,
    );
    fn oracle_dc_sparse_sgemv8x4(
        out: *mut f32,
        w: *const f32,
        idx: *const c_int,
        rows: c_int,
        x: *const f32,
    );
    fn oracle_dc_cgemv8x4(
        out: *mut f32,
        w: *const i8,
        scale: *const f32,
        rows: c_int,
        cols: c_int,
        x: *const f32,
    );
    fn oracle_dc_sparse_cgemv8x4(
        out: *mut f32,
        w: *const i8,
        idx: *const c_int,
        scale: *const f32,
        rows: c_int,
        cols: c_int,
        x: *const f32,
    );
    fn oracle_dc_scalar(which: c_int, x: *const f32, y: *mut f32, n: c_int);
    fn oracle_dc_lin2ulaw(x: *const f32, y: *mut c_int, n: c_int);
    fn oracle_dc_kiss99(
        seed: *const u8,
        nseed: c_int,
        state4: *mut c_uint,
        out: *mut c_uint,
        n: c_int,
    );

    fn oracle_dc_silk_burg_analysis(
        a: *mut f32,
        x: *const f32,
        min_inv_gain: f32,
        subfr_length: c_int,
        nb_subfr: c_int,
        d: c_int,
    ) -> f32;
    fn oracle_dc_silk_energy_flp(x: *const f32, n: c_int) -> f64;
    fn oracle_dc_silk_inner_product_flp(x: *const f32, y: *const f32, n: c_int) -> f64;

    fn oracle_dc_lpcn_compute_band_energy(band_e: *mut f32, x: *const f32);
    fn oracle_dc_compute_band_energy_inverse(band_e: *mut f32, x: *const f32);
    fn oracle_dc_lpcn_lpc(lpc: *mut f32, rc: *mut f32, ac: *const f32, p: c_int) -> f32;
    fn oracle_dc_compute_burg_cepstrum(pcm: *const f32, ceps: *mut f32, len: c_int, order: c_int);
    fn oracle_dc_burg_cepstral_analysis(ceps: *mut f32, x: *const f32);
    fn oracle_dc_interp_band_gain(g: *mut f32, band_e: *const f32);
    fn oracle_dc_dct(out: *mut f32, input: *const f32);
    fn oracle_dc_idct(out: *mut f32, input: *const f32);
    fn oracle_dc_forward_transform(out: *mut f32, input: *const f32);
    fn oracle_dc_inverse_transform(out: *mut f32, input: *const f32);
    fn oracle_dc_lpc_from_bands(lpc: *mut f32, ex: *const f32) -> f32;
    fn oracle_dc_lpc_weighting(lpc: *mut f32, gamma: f32);
    fn oracle_dc_lpc_from_cepstrum(lpc: *mut f32, ceps: *const f32) -> f32;
    fn oracle_dc_apply_window(x: *mut f32);
    fn oracle_dc_tables(
        nfft: *mut c_int,
        shift: *mut c_int,
        scale: *mut f32,
        factors: *mut i16,
        bitrev: *mut i16,
        twiddles: *mut f32,
        half_win: *mut f32,
        dct_tab: *mut f32,
    );

    fn oracle_dc_pitchdnn_new() -> *mut c_void;
    fn oracle_dc_pitchdnn_free(p: *mut c_void);
    fn oracle_dc_compute_pitchdnn(
        p: *mut c_void,
        if_features: *const f32,
        xcorr: *const f32,
    ) -> f32;
    fn oracle_dc_pitchdnn_state(p: *mut c_void, out: *mut f32);

    fn oracle_dc_lpcnet_enc_new() -> *mut c_void;
    fn oracle_dc_lpcnet_enc_free(p: *mut c_void);
    fn oracle_dc_lpcnet_features(p: *mut c_void, pcm: *const i16, features: *mut f32);
    fn oracle_dc_lpcnet_features_float(p: *mut c_void, pcm: *const f32, features: *mut f32);
    fn oracle_dc_compute_frame_features(p: *mut c_void, input: *const f32);
    fn oracle_dc_frame_analysis(p: *mut c_void, x: *mut f32, ex: *mut f32, input: *const f32);
    fn oracle_dc_biquad(
        y: *mut f32,
        mem: *mut f32,
        x: *const f32,
        b: *const f32,
        a: *const f32,
        n: c_int,
        inplace: c_int,
    );
    fn oracle_dc_preemphasis(
        y: *mut f32,
        mem: *mut f32,
        x: *const f32,
        coef: f32,
        n: c_int,
        inplace: c_int,
    );
    fn oracle_dc_lpcnet_enc_state(p: *mut c_void, out: *mut f32) -> c_int;
}

#[cfg(feature = "osce")]
unsafe extern "C" {
    fn oracle_dc_adaconv_new() -> *mut c_void;
    fn oracle_dc_adacomb_new() -> *mut c_void;
    fn oracle_dc_adashape_new() -> *mut c_void;
    fn oracle_dc_nndsp_free(p: *mut c_void);
    fn oracle_dc_adaconv_state(p: *mut c_void, out: *mut f32);
    fn oracle_dc_adacomb_state(p: *mut c_void, out: *mut f32, lag: *mut c_int);
    fn oracle_dc_adashape_state(p: *mut c_void, out: *mut f32);
    fn oracle_dc_compute_overlap_window(window: *mut f32, n: c_int);
    fn oracle_dc_scale_kernel(
        kernel: *mut f32,
        in_ch: c_int,
        out_ch: c_int,
        ksize: c_int,
        gain: *mut f32,
    );
    fn oracle_dc_transform_gains(gains: *mut f32, n: c_int, a: f32, b: f32);
    fn oracle_dc_adaconv(
        st: *mut c_void,
        x_out: *mut f32,
        x_in: *const f32,
        features: *const f32,
        kernel: *mut c_void,
        gain: *mut c_void,
        feature_dim: c_int,
        frame_size: c_int,
        overlap_size: c_int,
        in_ch: c_int,
        out_ch: c_int,
        ksize: c_int,
        left_padding: c_int,
        ga: f32,
        gb: f32,
        shape_gain: f32,
        window: *mut f32,
        inplace: c_int,
    );
    fn oracle_dc_adacomb(
        st: *mut c_void,
        x_out: *mut f32,
        x_in: *const f32,
        features: *const f32,
        kernel: *mut c_void,
        gain: *mut c_void,
        global_gain: *mut c_void,
        pitch_lag: c_int,
        feature_dim: c_int,
        frame_size: c_int,
        overlap_size: c_int,
        ksize: c_int,
        left_padding: c_int,
        ga: f32,
        gb: f32,
        log_gain_limit: f32,
        window: *mut f32,
        inplace: c_int,
    );
    fn oracle_dc_adashape(
        st: *mut c_void,
        x_out: *mut f32,
        x_in: *const f32,
        features: *const f32,
        a1f: *mut c_void,
        a1t: *mut c_void,
        a2: *mut c_void,
        feature_dim: c_int,
        frame_size: c_int,
        avg_pool_k: c_int,
        interpolate_k: c_int,
        inplace: c_int,
    );
}

fn ci(v: usize) -> c_int {
    c_int::try_from(v).expect("size fits in a C int")
}

fn opt_ptr<T>(v: Option<&[T]>) -> *const T {
    v.map_or(core::ptr::null(), <[T]>::as_ptr)
}

// ---------------------------------------------------------------------------------------------
// Model tables and blobs

/// One compiled-in array of a model table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArray {
    pub name: String,
    pub type_: i32,
    pub size: i32,
    pub data: Vec<u8>,
}

/// Number of arrays of `model`, or `None` if the model is not compiled into this oracle build.
#[must_use]
pub fn model_count(model: i32) -> Option<usize> {
    // SAFETY: only reads the static table selected by `model` (NULL-checked in C).
    let n = unsafe { oracle_dc_model_count(model) };
    usize::try_from(n).ok()
}

/// All arrays of `model` (compiled-in `*_arrays` table), in table order.
#[must_use]
pub fn model_arrays(model: i32) -> Vec<ModelArray> {
    let n = model_count(model).expect("model compiled into the oracle");
    (0..n)
        .map(|i| {
            let mut name = [0u8; 256];
            let (mut ty, mut size) = (0, 0);
            // SAFETY: i < count; name has the 256 bytes C writes at most.
            unsafe {
                oracle_dc_model_array_info(model, ci(i), name.as_mut_ptr(), &mut ty, &mut size);
            }
            let len = name.iter().position(|&c| c == 0).expect("NUL-terminated");
            let name: Vec<u8> = name[..len].to_vec();
            let mut data = vec![0u8; usize::try_from(size).expect("non-negative size")];
            // SAFETY: data holds exactly `size` bytes, which is what C copies.
            unsafe { oracle_dc_model_array_data(model, ci(i), data.as_mut_ptr().cast()) };
            ModelArray {
                name: String::from_utf8(name).expect("ASCII names"),
                type_: ty,
                size,
                data,
            }
        })
        .collect()
}

/// Serializes the compiled-in `model` table into a weight blob (write_lpcnet_weights.c format).
#[must_use]
pub fn write_blob(model: i32) -> Vec<u8> {
    assert!(
        model_count(model).is_some(),
        "model compiled into the oracle"
    );
    // SAFETY: NULL output only computes the size.
    let len = unsafe { oracle_dc_write_blob(model, core::ptr::null_mut()) };
    let mut out = vec![0u8; usize::try_from(len).expect("non-negative")];
    // SAFETY: `out` has the size computed above.
    unsafe { oracle_dc_write_blob(model, out.as_mut_ptr()) };
    out
}

/// Result of C `parse_weights` on a blob: per array `[type, size, data offset, name offset]`
/// (offsets from the blob start); `None` when C returns -1.
#[must_use]
pub fn parse_weights(blob: &[u8]) -> Option<Vec<[i32; 4]>> {
    // C reads the headers as ints: give it an 8-byte aligned copy.
    let mut aligned = vec![0u64; blob.len().div_ceil(8) + 1];
    let bytes: &mut [u8] = bytemuck_u8(&mut aligned);
    bytes[..blob.len()].copy_from_slice(blob);
    let mut info = vec![0 as c_int; 4 * (blob.len() / 64 + 1)];
    let max = ci(info.len() / 4);
    // SAFETY: C reads at most `len` bytes of the (larger) buffer and writes at most `max`
    // info entries.
    let n =
        unsafe { oracle_dc_parse_weights(bytes.as_ptr(), ci(blob.len()), info.as_mut_ptr(), max) };
    let n = usize::try_from(n).ok()?;
    Some(
        (0..n)
            .map(|i| {
                [
                    info[4 * i],
                    info[4 * i + 1],
                    info[4 * i + 2],
                    info[4 * i + 3],
                ]
            })
            .collect(),
    )
}

fn bytemuck_u8(v: &mut [u64]) -> &mut [u8] {
    let len = v.len() * 8;
    // SAFETY: u64 has no padding and any byte pattern is valid for u8; the length matches.
    unsafe { core::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), len) }
}

// ---------------------------------------------------------------------------------------------
// LinearLayer / Conv2dLayer handles

/// Validates a sparse index array (as C `find_idx_check`) and returns its total block count.
fn idx_blocks(idx: &[i32], nb_in: usize, nb_out: usize) -> usize {
    let mut p = 0;
    let mut total = 0usize;
    let mut rows = 0usize;
    while p < idx.len() {
        let n = usize::try_from(idx[p]).expect("non-negative block count");
        p += 1;
        for _ in 0..n {
            let pos = usize::try_from(idx[p]).expect("non-negative position");
            assert!(
                pos.is_multiple_of(4) && pos + 3 < nb_in,
                "bad sparse position"
            );
            p += 1;
        }
        total += n;
        rows += 8;
    }
    assert_eq!(rows, nb_out, "sparse index rows");
    total
}

/// A C `LinearLayer` on the heap.
#[derive(Debug)]
pub struct Linear {
    p: *mut c_void,
    nb_in: usize,
    nb_out: usize,
    sparse_blocks: Option<usize>,
    has_weights: bool,
    has_float: bool,
    /// False for a failed `linear_init` (the C layer may be half bound: never compute on it).
    usable: bool,
}

/// Arrays for [`Linear::new`] (`None` is C `NULL`).
#[derive(Debug, Default, Clone, Copy)]
pub struct LinearArrays<'a> {
    pub bias: Option<&'a [f32]>,
    pub subias: Option<&'a [f32]>,
    pub weights: Option<&'a [i8]>,
    pub float_weights: Option<&'a [f32]>,
    pub weights_idx: Option<&'a [i32]>,
    pub diag: Option<&'a [f32]>,
    pub scale: Option<&'a [f32]>,
}

impl Linear {
    /// Builds a layer from explicit arrays (copied into C memory).
    #[must_use]
    pub fn new(a: LinearArrays<'_>, nb_in: usize, nb_out: usize) -> Self {
        for v in [a.bias, a.subias, a.diag, a.scale].into_iter().flatten() {
            assert!(v.len() >= nb_out);
        }
        let sparse_blocks = a.weights_idx.map(|idx| idx_blocks(idx, nb_in, nb_out));
        let nw = sparse_blocks.map_or(nb_in * nb_out, |b| 32 * b);
        if let Some(w) = a.weights {
            assert_eq!(w.len(), nw);
            assert!(a.scale.is_some(), "int8 weights need a scale");
            assert!(nb_in <= 2048);
            if sparse_blocks.is_none() {
                assert!(nb_out.is_multiple_of(8) && nb_in.is_multiple_of(4));
            }
        }
        if let Some(w) = a.float_weights {
            assert_eq!(w.len(), nw);
        }
        if sparse_blocks.is_some() {
            assert!(nb_out.is_multiple_of(8));
        }
        if a.diag.is_some() {
            assert_eq!(3 * nb_in, nb_out);
        }
        // SAFETY: every array is at least as long as C copies (checked above).
        let p = unsafe {
            oracle_dc_linear_new(
                opt_ptr(a.bias),
                opt_ptr(a.subias),
                opt_ptr(a.weights),
                ci(a.weights.map_or(0, <[i8]>::len)),
                opt_ptr(a.float_weights),
                ci(a.float_weights.map_or(0, <[f32]>::len)),
                opt_ptr(a.weights_idx),
                ci(a.weights_idx.map_or(0, <[i32]>::len)),
                opt_ptr(a.diag),
                opt_ptr(a.scale),
                ci(nb_in),
                ci(nb_out),
            )
        };
        Self {
            p,
            nb_in,
            nb_out,
            sparse_blocks,
            has_weights: a.weights.is_some(),
            has_float: a.float_weights.is_some(),
            usable: true,
        }
    }

    /// C `linear_init` on the compiled-in table of `model`. Returns the handle and the C return
    /// value; the handle is only usable for computation when the return value is 0.
    #[must_use]
    #[expect(clippy::too_many_arguments, reason = "mirrors linear_init")]
    pub fn init(
        model: i32,
        bias: Option<&str>,
        subias: Option<&str>,
        weights: Option<&str>,
        float_weights: Option<&str>,
        weights_idx: Option<&str>,
        diag: Option<&str>,
        scale: Option<&str>,
        nb_in: usize,
        nb_out: usize,
    ) -> (Self, i32) {
        assert!(
            model_count(model).is_some(),
            "model compiled into the oracle"
        );
        let names: Vec<Option<CString>> = [
            bias,
            subias,
            weights,
            float_weights,
            weights_idx,
            diag,
            scale,
        ]
        .iter()
        .map(|n| n.map(|s| CString::new(s).expect("no NUL")))
        .collect();
        let np = |i: usize| names[i].as_ref().map_or(core::ptr::null(), |c| c.as_ptr());
        let mut ret = 0;
        // With weights but no scale name C would strcmp(NULL): forbid it.
        assert!(weights.is_none() || scale.is_some());
        // SAFETY: NUL-terminated names or NULL; C's linear_init only reads the static table.
        let p = unsafe {
            oracle_dc_linear_init(
                model,
                np(0),
                np(1),
                np(2),
                np(3),
                np(4),
                np(5),
                np(6),
                ci(nb_in),
                ci(nb_out),
                &mut ret,
            )
        };
        let mut h = Self {
            p,
            nb_in,
            nb_out,
            sparse_blocks: None,
            has_weights: false,
            has_float: false,
            usable: ret == 0,
        };
        if ret == 0 {
            h.has_weights = h.field_raw(2, &mut []);
            h.has_float = h.field_raw(3, &mut []);
            if weights_idx.is_some() {
                // Recover the block count from the idx array actually bound.
                let n = h.idx_len();
                let idx = h.field_i32(4, n).expect("idx bound");
                h.sparse_blocks = Some(idx_blocks(&idx, nb_in, nb_out));
            }
        }
        (h, ret)
    }

    fn field_raw(&self, f: i32, out: &mut [u8]) -> bool {
        // SAFETY: C copies exactly `out.len()` bytes from the bound array (callers pass lengths
        // derived from the layer dimensions).
        unsafe { oracle_dc_linear_field(self.p, f, out.as_mut_ptr().cast(), ci(out.len())) != 0 }
    }

    /// Length (in ints) of the bound sparse index array, walking it as C does.
    fn idx_len(&self) -> usize {
        let mut len = 0usize;
        let mut rows = 0;
        while rows < self.nb_out {
            let head = self.field_i32(4, len + 1).expect("idx bound");
            let n = usize::try_from(head[len]).expect("non-negative");
            len += 1 + n;
            rows += 8;
        }
        len
    }

    /// `n` floats of field `f` (0 bias, 1 subias, 3 float_weights, 5 diag, 6 scale), `None`
    /// if NULL.
    #[must_use]
    pub fn field_f32(&self, f: i32, n: usize) -> Option<Vec<f32>> {
        assert!(self.field_len_ok(f, n));
        let mut v = vec![0f32; n];
        // SAFETY: see field_raw.
        let some = unsafe { oracle_dc_linear_field(self.p, f, v.as_mut_ptr().cast(), ci(4 * n)) };
        (some != 0).then_some(v)
    }

    /// `n` bytes of the int8 weights, `None` if NULL.
    #[must_use]
    pub fn field_i8(&self, n: usize) -> Option<Vec<i8>> {
        assert!(self.field_len_ok(2, n));
        let mut v = vec![0i8; n];
        // SAFETY: see field_raw.
        let some = unsafe { oracle_dc_linear_field(self.p, 2, v.as_mut_ptr().cast(), ci(n)) };
        (some != 0).then_some(v)
    }

    /// `n` ints of field `f` (4: weights_idx), `None` if NULL.
    #[must_use]
    pub fn field_i32(&self, f: i32, n: usize) -> Option<Vec<i32>> {
        let mut v = vec![0i32; n];
        // SAFETY: see field_raw; idx lengths come from walking the array (idx_len).
        let some = unsafe { oracle_dc_linear_field(self.p, f, v.as_mut_ptr().cast(), ci(4 * n)) };
        (some != 0).then_some(v)
    }

    fn field_len_ok(&self, f: i32, n: usize) -> bool {
        match f {
            0 | 1 | 5 | 6 => n <= self.nb_out,
            2 | 3 => {
                n <= self
                    .sparse_blocks
                    .map_or(self.nb_in * self.nb_out, |b| 32 * b)
            }
            _ => false,
        }
    }

    /// `(nb_inputs, nb_outputs)` as stored in C.
    #[must_use]
    pub fn dims(&self) -> (usize, usize) {
        let (mut i, mut o) = (0, 0);
        // SAFETY: plain reads of the handle.
        unsafe { oracle_dc_linear_dims(self.p, &mut i, &mut o) };
        (i as usize, o as usize)
    }

    fn check_io(&self, out: usize, input: usize) {
        assert!(self.usable, "layer from a failed linear_init");
        assert!(out >= self.nb_out && input >= self.nb_in);
        if self.has_weights && !self.has_float {
            assert!(self.nb_in <= 2048);
        }
    }

    /// C `compute_linear_c`.
    pub fn compute_linear(&self, out: &mut [f32], input: &[f32]) {
        self.check_io(out.len(), input.len());
        // SAFETY: sizes checked (out >= nb_out, input >= nb_in; the layer arrays were
        // validated at construction).
        unsafe { oracle_dc_compute_linear(self.p, out.as_mut_ptr(), input.as_ptr()) }
    }

    /// C `compute_generic_dense`.
    pub fn compute_generic_dense(&self, out: &mut [f32], input: &[f32], act: i32) {
        self.check_io(out.len(), input.len());
        // SAFETY: as compute_linear.
        unsafe { oracle_dc_compute_generic_dense(self.p, out.as_mut_ptr(), input.as_ptr(), act) }
    }

    /// C `compute_glu` (`inplace`: `out` is also the input).
    pub fn compute_glu(&self, out: &mut [f32], input: Option<&[f32]>) {
        assert_eq!(self.nb_in, self.nb_out);
        assert!(self.nb_out <= 2048);
        self.check_io(out.len(), input.map_or(out.len(), <[f32]>::len));
        // SAFETY: sizes checked; the C scratch holds MAX_INPUTS = 2048 floats.
        unsafe {
            oracle_dc_compute_glu(
                self.p,
                out.as_mut_ptr(),
                opt_ptr(input),
                c_int::from(input.is_none()),
            );
        }
    }

    /// C `compute_generic_conv1d`.
    pub fn compute_generic_conv1d(
        &self,
        out: &mut [f32],
        mem: &mut [f32],
        input: &[f32],
        input_size: usize,
        act: i32,
    ) {
        self.check_io(out.len(), self.nb_in);
        assert!(self.nb_in <= 1024 && input_size <= self.nb_in);
        assert!(out.len() >= self.nb_out && input.len() >= input_size);
        assert!(mem.len() >= self.nb_in - input_size);
        // SAFETY: sizes checked; the C scratch holds 1024 floats.
        unsafe {
            oracle_dc_compute_generic_conv1d(
                self.p,
                out.as_mut_ptr(),
                mem.as_mut_ptr(),
                input.as_ptr(),
                ci(input_size),
                act,
            );
        }
    }

    /// C `compute_generic_conv1d_dilation`.
    pub fn compute_generic_conv1d_dilation(
        &self,
        out: &mut [f32],
        mem: &mut [f32],
        input: &[f32],
        input_size: usize,
        dilation: usize,
        act: i32,
    ) {
        self.check_io(out.len(), self.nb_in);
        assert!(self.nb_in <= 1024 && input_size > 0 && self.nb_in.is_multiple_of(input_size));
        let ksize = self.nb_in / input_size;
        assert!(ksize >= 2 && dilation >= 1);
        assert!(out.len() >= self.nb_out && input.len() >= input_size);
        let need = if dilation == 1 {
            self.nb_in - input_size
        } else {
            input_size * dilation * (ksize - 1)
        };
        assert!(mem.len() >= need);
        // SAFETY: sizes checked.
        unsafe {
            oracle_dc_compute_generic_conv1d_dilation(
                self.p,
                out.as_mut_ptr(),
                mem.as_mut_ptr(),
                input.as_ptr(),
                ci(input_size),
                ci(dilation),
                act,
            );
        }
    }

    fn ptr(&self) -> *mut c_void {
        self.p
    }
}

impl Drop for Linear {
    fn drop(&mut self) {
        // SAFETY: handle created by oracle_dc_linear_new/init, freed once.
        unsafe { oracle_dc_linear_free(self.p) }
    }
}

/// C `compute_generic_gru`.
pub fn compute_generic_gru(
    input_weights: &Linear,
    recurrent: &Linear,
    state: &mut [f32],
    input: &[f32],
) {
    assert_eq!(3 * recurrent.nb_in, recurrent.nb_out);
    assert_eq!(input_weights.nb_out, recurrent.nb_out);
    assert!(recurrent.nb_out <= 3 * 192);
    input_weights.check_io(usize::MAX, input.len());
    recurrent.check_io(usize::MAX, state.len());
    // SAFETY: sizes checked; C scratch holds 3*MAX_RNN_NEURONS_ALL = 576 floats.
    unsafe {
        oracle_dc_compute_generic_gru(
            input_weights.ptr(),
            recurrent.ptr(),
            state.as_mut_ptr(),
            input.as_ptr(),
        );
    }
}

/// C `compute_activation_c` (`input: None` = in place on `out`).
pub fn compute_activation(out: &mut [f32], input: Option<&[f32]>, n: usize, act: i32) {
    assert!(out.len() >= n && input.is_none_or(|i| i.len() >= n) && n <= 4096);
    // SAFETY: sizes checked (swish scratch is MAX_ACTIVATIONS = 4096).
    unsafe {
        oracle_dc_compute_activation(
            out.as_mut_ptr(),
            opt_ptr(input),
            ci(n),
            act,
            c_int::from(input.is_none()),
        );
    }
}

/// C `vec_swish` (nnet_arch.h copy).
pub fn vec_swish(y: &mut [f32], x: &[f32], n: usize) {
    assert!(y.len() >= n && x.len() >= n && n <= 4096);
    // SAFETY: sizes checked.
    unsafe { oracle_dc_vec_swish(y.as_mut_ptr(), x.as_ptr(), ci(n)) }
}

/// A C `Conv2dLayer` on the heap.
#[derive(Debug)]
pub struct Conv2d {
    p: *mut c_void,
    in_ch: usize,
    out_ch: usize,
    ktime: usize,
    kheight: usize,
}

impl Conv2d {
    /// Builds a layer from explicit arrays.
    #[must_use]
    pub fn new(
        bias: Option<&[f32]>,
        w: &[f32],
        in_ch: usize,
        out_ch: usize,
        ktime: usize,
        kheight: usize,
    ) -> Self {
        assert!(bias.is_none_or(|b| b.len() >= out_ch));
        assert_eq!(w.len(), in_ch * out_ch * ktime * kheight);
        assert!(ktime >= 1 && kheight >= 1);
        // SAFETY: sizes checked.
        let p = unsafe {
            oracle_dc_conv2d_new(
                opt_ptr(bias),
                w.as_ptr(),
                ci(in_ch),
                ci(out_ch),
                ci(ktime),
                ci(kheight),
            )
        };
        Self {
            p,
            in_ch,
            out_ch,
            ktime,
            kheight,
        }
    }

    /// C `conv2d_init` on the compiled-in table of `model` (handle + C return value).
    #[must_use]
    pub fn init(
        model: i32,
        bias: Option<&str>,
        float_weights: Option<&str>,
        in_ch: usize,
        out_ch: usize,
        ktime: usize,
        kheight: usize,
    ) -> (Self, i32) {
        assert!(
            model_count(model).is_some(),
            "model compiled into the oracle"
        );
        let b = bias.map(|s| CString::new(s).expect("no NUL"));
        let w = float_weights.map(|s| CString::new(s).expect("no NUL"));
        let mut ret = 0;
        // SAFETY: NUL-terminated names or NULL.
        let p = unsafe {
            oracle_dc_conv2d_init(
                model,
                b.as_ref().map_or(core::ptr::null(), |c| c.as_ptr()),
                w.as_ref().map_or(core::ptr::null(), |c| c.as_ptr()),
                ci(in_ch),
                ci(out_ch),
                ci(ktime),
                ci(kheight),
                &mut ret,
            )
        };
        (
            Self {
                p,
                in_ch,
                out_ch,
                ktime,
                kheight,
            },
            ret,
        )
    }

    /// Field `f` (0 bias: `out_ch` floats, 1 weights: all), `None` if NULL.
    #[must_use]
    pub fn field(&self, f: i32) -> Option<Vec<f32>> {
        let n = if f == 0 {
            self.out_ch
        } else {
            self.in_ch * self.out_ch * self.ktime * self.kheight
        };
        let mut v = vec![0f32; n];
        // SAFETY: the bound arrays have exactly these sizes (conv2d_init / new check them).
        let some = unsafe { oracle_dc_conv2d_field(self.p, f, v.as_mut_ptr().cast(), ci(4 * n)) };
        (some != 0).then_some(v)
    }

    /// C `compute_conv2d_c`.
    pub fn compute_conv2d(
        &self,
        out: &mut [f32],
        mem: &mut [f32],
        input: &[f32],
        height: usize,
        hstride: usize,
        act: i32,
    ) {
        let ts = self.in_ch * (height + self.kheight - 1);
        assert!(self.ktime * ts <= 8192);
        assert!(mem.len() >= (self.ktime - 1) * ts && input.len() >= ts);
        assert!(hstride >= height && out.len() >= (self.out_ch - 1) * hstride + height);
        assert!(self.field(1).is_some());
        // SAFETY: sizes checked.
        unsafe {
            oracle_dc_compute_conv2d(
                self.p,
                out.as_mut_ptr(),
                mem.as_mut_ptr(),
                input.as_ptr(),
                ci(height),
                ci(hstride),
                act,
            );
        }
    }
}

impl Drop for Conv2d {
    fn drop(&mut self) {
        // SAFETY: handle created by oracle_dc_conv2d_new/init, freed once.
        unsafe { oracle_dc_conv2d_free(self.p) }
    }
}

/// C `conv2d_float` (nnet_arch.h copy).
#[expect(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub fn conv2d_float(
    out: &mut [f32],
    w: &[f32],
    in_ch: usize,
    out_ch: usize,
    ktime: usize,
    kheight: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    assert!(w.len() >= in_ch * out_ch * ktime * kheight);
    assert!(input.len() >= ktime * in_ch * (height + kheight - 1));
    assert!(out_ch >= 1 && out.len() >= (out_ch - 1) * hstride + height);
    // SAFETY: sizes checked.
    unsafe {
        oracle_dc_conv2d_float(
            out.as_mut_ptr(),
            w.as_ptr(),
            ci(in_ch),
            ci(out_ch),
            ci(ktime),
            ci(kheight),
            input.as_ptr(),
            ci(height),
            ci(hstride),
        );
    }
}

/// C `conv2d_3x3_float` (nnet_arch.h copy).
pub fn conv2d_3x3_float(
    out: &mut [f32],
    w: &[f32],
    in_ch: usize,
    out_ch: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    assert!(w.len() >= in_ch * out_ch * 9);
    assert!(input.len() >= 3 * in_ch * (height + 2));
    assert!(out_ch >= 1 && out.len() >= (out_ch - 1) * hstride + height);
    // SAFETY: sizes checked.
    unsafe {
        oracle_dc_conv2d_3x3_float(
            out.as_mut_ptr(),
            w.as_ptr(),
            ci(in_ch),
            ci(out_ch),
            input.as_ptr(),
            ci(height),
            ci(hstride),
        );
    }
}

// ---------------------------------------------------------------------------------------------
// vec.h / common.h / kiss99

/// C `sgemv`.
pub fn sgemv(out: &mut [f32], w: &[f32], rows: usize, cols: usize, stride: usize, x: &[f32]) {
    assert!(out.len() >= rows && x.len() >= cols && stride >= rows);
    assert!(cols == 0 || w.len() >= (cols - 1) * stride + rows);
    // SAFETY: sizes checked.
    unsafe {
        oracle_dc_sgemv(
            out.as_mut_ptr(),
            w.as_ptr(),
            ci(rows),
            ci(cols),
            ci(stride),
            x.as_ptr(),
        )
    }
}

/// C `sparse_sgemv8x4`.
pub fn sparse_sgemv8x4(out: &mut [f32], w: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    assert!(rows.is_multiple_of(8) && out.len() >= rows);
    let blocks = idx_blocks(idx, x.len(), rows);
    assert!(w.len() >= 32 * blocks);
    // SAFETY: sizes checked (idx validated against x.len()).
    unsafe {
        oracle_dc_sparse_sgemv8x4(
            out.as_mut_ptr(),
            w.as_ptr(),
            idx.as_ptr(),
            ci(rows),
            x.as_ptr(),
        )
    }
}

/// C `cgemv8x4`.
pub fn cgemv8x4(out: &mut [f32], w: &[i8], scale: &[f32], rows: usize, cols: usize, x: &[f32]) {
    assert!(rows.is_multiple_of(8) && cols.is_multiple_of(4) && cols <= 2048);
    assert!(out.len() >= rows && scale.len() >= rows && x.len() >= cols && w.len() >= rows * cols);
    // SAFETY: sizes checked.
    unsafe {
        oracle_dc_cgemv8x4(
            out.as_mut_ptr(),
            w.as_ptr(),
            scale.as_ptr(),
            ci(rows),
            ci(cols),
            x.as_ptr(),
        )
    }
}

/// C `sparse_cgemv8x4`.
pub fn sparse_cgemv8x4(
    out: &mut [f32],
    w: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
) {
    assert!(rows.is_multiple_of(8) && cols <= 2048 && x.len() >= cols);
    assert!(out.len() >= rows && scale.len() >= rows);
    let blocks = idx_blocks(idx, cols, rows);
    assert!(w.len() >= 32 * blocks);
    // SAFETY: sizes checked.
    unsafe {
        oracle_dc_sparse_cgemv8x4(
            out.as_mut_ptr(),
            w.as_ptr(),
            idx.as_ptr(),
            scale.as_ptr(),
            ci(rows),
            ci(cols),
            x.as_ptr(),
        )
    }
}

/// Scalar helpers, applied element-wise: 0 `tanh_approx`, 1 `sigmoid_approx`, 2 `lpcnet_exp`,
/// 3 `lpcnet_exp2`, 4 `log2_approx`, 5 `log_approx`, 6 `ulaw2lin`, 7 `relu`.
#[must_use]
pub fn scalar(which: i32, x: &[f32]) -> Vec<f32> {
    assert!((0..8).contains(&which));
    let mut y = vec![0f32; x.len()];
    // SAFETY: y has x.len() elements.
    unsafe { oracle_dc_scalar(which, x.as_ptr(), y.as_mut_ptr(), ci(x.len())) };
    y
}

/// C `lin2ulaw`, element-wise.
#[must_use]
pub fn lin2ulaw(x: &[f32]) -> Vec<i32> {
    let mut y = vec![0; x.len()];
    // SAFETY: y has x.len() elements.
    unsafe { oracle_dc_lin2ulaw(x.as_ptr(), y.as_mut_ptr(), ci(x.len())) };
    y
}

/// C `kiss99_srand(seed)` then `n` `kiss99_rand`: returns the state after seeding and the
/// outputs.
#[must_use]
pub fn kiss99(seed: &[u8], n: usize) -> ([u32; 4], Vec<u32>) {
    let mut st = [0u32; 4];
    let mut out = vec![0u32; n];
    // SAFETY: sizes as C expects.
    unsafe {
        oracle_dc_kiss99(
            seed.as_ptr(),
            ci(seed.len()),
            st.as_mut_ptr(),
            out.as_mut_ptr(),
            ci(n),
        )
    };
    (st, out)
}

// ---------------------------------------------------------------------------------------------
// burg / freq / tables

/// C `silk_burg_analysis` (returns the residual energy).
pub fn silk_burg_analysis(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) -> f32 {
    assert!(d <= 16 && a.len() >= d && subfr_length > d && x.len() >= subfr_length * nb_subfr);
    assert!(subfr_length * nb_subfr <= 384);
    // SAFETY: sizes checked.
    unsafe {
        oracle_dc_silk_burg_analysis(
            a.as_mut_ptr(),
            x.as_ptr(),
            min_inv_gain,
            ci(subfr_length),
            ci(nb_subfr),
            ci(d),
        )
    }
}

/// C `silk_energy_FLP` (burg.c static copy).
#[must_use]
pub fn silk_energy_flp(x: &[f32]) -> f64 {
    // SAFETY: reads x.len() floats.
    unsafe { oracle_dc_silk_energy_flp(x.as_ptr(), ci(x.len())) }
}

/// C `silk_inner_product_FLP` (burg.c static copy).
#[must_use]
pub fn silk_inner_product_flp(x: &[f32], y: &[f32]) -> f64 {
    assert_eq!(x.len(), y.len());
    // SAFETY: reads x.len() floats of each.
    unsafe { oracle_dc_silk_inner_product_flp(x.as_ptr(), y.as_ptr(), ci(x.len())) }
}

/// `NB_BANDS`.
const NB: usize = 18;
/// `FREQ_SIZE`.
const FREQ: usize = 161;
/// `WINDOW_SIZE`.
const WIN: usize = 320;

/// C `lpcn_compute_band_energy`; `x` is `FREQ_SIZE` interleaved complex values.
#[must_use]
pub fn lpcn_compute_band_energy(x: &[f32]) -> [f32; NB] {
    assert!(x.len() >= 2 * FREQ);
    let mut e = [0f32; NB];
    // SAFETY: sizes checked.
    unsafe { oracle_dc_lpcn_compute_band_energy(e.as_mut_ptr(), x.as_ptr()) };
    e
}

/// C `compute_band_energy_inverse` (freq.c static copy).
#[must_use]
pub fn compute_band_energy_inverse(x: &[f32]) -> [f32; NB] {
    assert!(x.len() >= 2 * FREQ);
    let mut e = [0f32; NB];
    // SAFETY: sizes checked.
    unsafe { oracle_dc_compute_band_energy_inverse(e.as_mut_ptr(), x.as_ptr()) };
    e
}

/// C `lpcn_lpc` (freq.c static copy): returns the error.
pub fn lpcn_lpc(lpc: &mut [f32], rc: &mut [f32], ac: &[f32], p: usize) -> f32 {
    assert!(lpc.len() >= p && rc.len() >= p && ac.len() > p);
    // SAFETY: sizes checked.
    unsafe { oracle_dc_lpcn_lpc(lpc.as_mut_ptr(), rc.as_mut_ptr(), ac.as_ptr(), ci(p)) }
}

/// C `compute_burg_cepstrum` (freq.c static copy).
pub fn compute_burg_cepstrum(pcm: &[f32], ceps: &mut [f32], len: usize, order: usize) {
    assert!(
        (2..=160).contains(&len)
            && order <= 16
            && len - 1 > order
            && pcm.len() >= len
            && ceps.len() >= NB
    );
    // SAFETY: sizes checked.
    unsafe { oracle_dc_compute_burg_cepstrum(pcm.as_ptr(), ceps.as_mut_ptr(), ci(len), ci(order)) }
}

/// C `burg_cepstral_analysis`.
pub fn burg_cepstral_analysis(ceps: &mut [f32], x: &[f32]) {
    assert!(ceps.len() >= 2 * NB && x.len() >= 160);
    // SAFETY: sizes checked.
    unsafe { oracle_dc_burg_cepstral_analysis(ceps.as_mut_ptr(), x.as_ptr()) }
}

/// C `interp_band_gain` (freq.c static copy); `g` gets `FREQ_SIZE` floats (the last one is
/// only partially cleared by C and is overwritten with 0 here before the call).
pub fn interp_band_gain(g: &mut [f32], band_e: &[f32]) {
    assert!(g.len() >= FREQ && band_e.len() >= NB);
    // SAFETY: sizes checked.
    unsafe { oracle_dc_interp_band_gain(g.as_mut_ptr(), band_e.as_ptr()) }
}

/// C `dct`.
pub fn dct(out: &mut [f32], input: &[f32]) {
    assert!(out.len() >= NB && input.len() >= NB);
    // SAFETY: sizes checked.
    unsafe { oracle_dc_dct(out.as_mut_ptr(), input.as_ptr()) }
}

/// C `idct` (freq.c static copy).
pub fn idct(out: &mut [f32], input: &[f32]) {
    assert!(out.len() >= NB && input.len() >= NB);
    // SAFETY: sizes checked.
    unsafe { oracle_dc_idct(out.as_mut_ptr(), input.as_ptr()) }
}

/// C `forward_transform`: `out` gets `FREQ_SIZE` interleaved complex values.
pub fn forward_transform(out: &mut [f32], input: &[f32]) {
    assert!(out.len() >= 2 * FREQ && input.len() >= WIN);
    // SAFETY: sizes checked.
    unsafe { oracle_dc_forward_transform(out.as_mut_ptr(), input.as_ptr()) }
}

/// C `inverse_transform` (freq.c static copy).
pub fn inverse_transform(out: &mut [f32], input: &[f32]) {
    assert!(out.len() >= WIN && input.len() >= 2 * FREQ);
    // SAFETY: sizes checked.
    unsafe { oracle_dc_inverse_transform(out.as_mut_ptr(), input.as_ptr()) }
}

/// C `lpc_from_bands` (freq.c static copy).
pub fn lpc_from_bands(lpc: &mut [f32], ex: &[f32]) -> f32 {
    assert!(lpc.len() >= 16 && ex.len() >= NB);
    // SAFETY: sizes checked.
    unsafe { oracle_dc_lpc_from_bands(lpc.as_mut_ptr(), ex.as_ptr()) }
}

/// C `lpc_weighting`.
pub fn lpc_weighting(lpc: &mut [f32], gamma: f32) {
    assert!(lpc.len() >= 16);
    // SAFETY: sizes checked.
    unsafe { oracle_dc_lpc_weighting(lpc.as_mut_ptr(), gamma) }
}

/// C `lpc_from_cepstrum`.
pub fn lpc_from_cepstrum(lpc: &mut [f32], ceps: &[f32]) -> f32 {
    assert!(lpc.len() >= 16 && ceps.len() >= NB);
    // SAFETY: sizes checked.
    unsafe { oracle_dc_lpc_from_cepstrum(lpc.as_mut_ptr(), ceps.as_ptr()) }
}

/// C `apply_window`.
pub fn apply_window(x: &mut [f32]) {
    assert!(x.len() >= WIN);
    // SAFETY: sizes checked.
    unsafe { oracle_dc_apply_window(x.as_mut_ptr()) }
}

/// The lpcnet_tables.c tables.
#[derive(Debug, Clone)]
pub struct Tables {
    pub nfft: i32,
    pub shift: i32,
    pub scale: f32,
    pub factors: [i16; 16],
    pub bitrev: Vec<i16>,
    /// Interleaved `(r, i)`.
    pub twiddles: Vec<f32>,
    pub half_window: Vec<f32>,
    pub dct_table: Vec<f32>,
}

/// Reads `kfft`, `half_window` and `dct_table`.
#[must_use]
pub fn tables() -> Tables {
    let mut t = Tables {
        nfft: 0,
        shift: 0,
        scale: 0.0,
        factors: [0; 16],
        bitrev: vec![0; WIN],
        twiddles: vec![0.0; 2 * WIN],
        half_window: vec![0.0; 160],
        dct_table: vec![0.0; NB * NB],
    };
    // SAFETY: buffers sized for kfft.nfft = 320, OVERLAP_SIZE = 160, NB_BANDS^2.
    unsafe {
        oracle_dc_tables(
            &mut t.nfft,
            &mut t.shift,
            &mut t.scale,
            t.factors.as_mut_ptr(),
            t.bitrev.as_mut_ptr(),
            t.twiddles.as_mut_ptr(),
            t.half_window.as_mut_ptr(),
            t.dct_table.as_mut_ptr(),
        );
    }
    assert_eq!(t.nfft, 320);
    t
}

// ---------------------------------------------------------------------------------------------
// pitchdnn / lpcnet_enc

/// Floats in [`PitchDnn::state`]: gru_state[64], xcorr_mem1[452], xcorr_mem2[3616].
pub const PITCHDNN_STATE_LEN: usize = 64 + 452 + 3616;

/// C `PitchDNNState` initialised by `pitchdnn_init` (compiled-in model).
#[derive(Debug)]
pub struct PitchDnn(*mut c_void);

impl PitchDnn {
    #[must_use]
    #[expect(clippy::new_without_default, reason = "allocates C state")]
    pub fn new() -> Self {
        // SAFETY: allocates and initialises a C state.
        Self(unsafe { oracle_dc_pitchdnn_new() })
    }

    /// C `compute_pitchdnn`.
    pub fn compute(&mut self, if_features: &[f32], xcorr: &[f32]) -> f32 {
        assert!(if_features.len() >= 88 && xcorr.len() >= 224);
        // SAFETY: sizes checked.
        unsafe { oracle_dc_compute_pitchdnn(self.0, if_features.as_ptr(), xcorr.as_ptr()) }
    }

    /// Flattened state (see [`PITCHDNN_STATE_LEN`]).
    #[must_use]
    pub fn state(&self) -> Vec<f32> {
        let mut v = vec![0f32; PITCHDNN_STATE_LEN];
        // SAFETY: v has the size C writes.
        unsafe { oracle_dc_pitchdnn_state(self.0, v.as_mut_ptr()) };
        v
    }
}

impl Drop for PitchDnn {
    fn drop(&mut self) {
        // SAFETY: allocated by oracle_dc_pitchdnn_new.
        unsafe { oracle_dc_pitchdnn_free(self.0) }
    }
}

/// Floats in [`LpcnetEnc::state`].
pub const LPCNET_ENC_STATE_LEN: usize =
    160 + 1 + 60 + 88 + 224 + 1 + 16 + 1 + 576 + 576 + 4 + 16 + 36 + 16 + 36 + PITCHDNN_STATE_LEN;

/// C `LPCNetEncState` from `lpcnet_encoder_create` (compiled-in pitch model).
#[derive(Debug)]
pub struct LpcnetEnc(*mut c_void);

impl LpcnetEnc {
    #[must_use]
    #[expect(clippy::new_without_default, reason = "allocates C state")]
    pub fn new() -> Self {
        // SAFETY: allocates and initialises a C state.
        Self(unsafe { oracle_dc_lpcnet_enc_new() })
    }

    /// C `lpcnet_compute_single_frame_features` (160 samples → 36 features).
    pub fn features(&mut self, pcm: &[i16]) -> [f32; 36] {
        assert!(pcm.len() >= 160);
        let mut f = [0f32; 36];
        // SAFETY: sizes checked.
        unsafe { oracle_dc_lpcnet_features(self.0, pcm.as_ptr(), f.as_mut_ptr()) };
        f
    }

    /// C `lpcnet_compute_single_frame_features_float`.
    pub fn features_float(&mut self, pcm: &[f32]) -> [f32; 36] {
        assert!(pcm.len() >= 160);
        let mut f = [0f32; 36];
        // SAFETY: sizes checked.
        unsafe { oracle_dc_lpcnet_features_float(self.0, pcm.as_ptr(), f.as_mut_ptr()) };
        f
    }

    /// C `compute_frame_features` on 160 samples.
    pub fn compute_frame_features(&mut self, input: &[f32]) {
        assert!(input.len() >= 160);
        // SAFETY: sizes checked.
        unsafe { oracle_dc_compute_frame_features(self.0, input.as_ptr()) }
    }

    /// C `frame_analysis` (lpcnet_enc.c static copy): returns (X interleaved, Ex).
    pub fn frame_analysis(&mut self, input: &[f32]) -> (Vec<f32>, [f32; NB]) {
        assert!(input.len() >= 160);
        let mut x = vec![0f32; 2 * FREQ];
        let mut ex = [0f32; NB];
        // SAFETY: sizes checked.
        unsafe {
            oracle_dc_frame_analysis(self.0, x.as_mut_ptr(), ex.as_mut_ptr(), input.as_ptr())
        };
        (x, ex)
    }

    /// Flattened float fields (see the C `oracle_dc_lpcnet_enc_state` order).
    #[must_use]
    pub fn state(&self) -> Vec<f32> {
        let mut v = vec![0f32; LPCNET_ENC_STATE_LEN];
        // SAFETY: v has the size C writes (asserted below).
        let n = unsafe { oracle_dc_lpcnet_enc_state(self.0, v.as_mut_ptr()) };
        assert_eq!(n as usize, LPCNET_ENC_STATE_LEN);
        v
    }
}

impl Drop for LpcnetEnc {
    fn drop(&mut self) {
        // SAFETY: allocated by lpcnet_encoder_create.
        unsafe { oracle_dc_lpcnet_enc_free(self.0) }
    }
}

/// C `biquad` (lpcnet_enc.c static copy); `x: None` = in place on `y`.
pub fn biquad(
    y: &mut [f32],
    mem: &mut [f32; 2],
    x: Option<&[f32]>,
    b: &[f32; 2],
    a: &[f32; 2],
    n: usize,
) {
    assert!(y.len() >= n && x.is_none_or(|x| x.len() >= n));
    // SAFETY: sizes checked.
    unsafe {
        oracle_dc_biquad(
            y.as_mut_ptr(),
            mem.as_mut_ptr(),
            opt_ptr(x),
            b.as_ptr(),
            a.as_ptr(),
            ci(n),
            c_int::from(x.is_none()),
        );
    }
}

/// C `preemphasis`; `x: None` = in place on `y`.
pub fn preemphasis(y: &mut [f32], mem: &mut f32, x: Option<&[f32]>, coef: f32, n: usize) {
    assert!(y.len() >= n && x.is_none_or(|x| x.len() >= n));
    // SAFETY: sizes checked.
    unsafe {
        oracle_dc_preemphasis(
            y.as_mut_ptr(),
            mem,
            opt_ptr(x),
            coef,
            ci(n),
            c_int::from(x.is_none()),
        )
    }
}

// ---------------------------------------------------------------------------------------------
// nndsp (osce)

/// Floats in [`AdaConv::state`]: history[96], last_kernel[288], last_gain.
#[cfg(feature = "osce")]
pub const ADACONV_STATE_LEN: usize = 96 + 288 + 1;
/// Floats in [`AdaComb::state`]: history[316], last_kernel[16], last_global_gain.
#[cfg(feature = "osce")]
pub const ADACOMB_STATE_LEN: usize = 316 + 16 + 1;
/// Floats in [`AdaShape::state`]: alpha1f[512], alpha1t[512], alpha2[240], interpolate[1].
#[cfg(feature = "osce")]
pub const ADASHAPE_STATE_LEN: usize = 512 + 512 + 240 + 1;

/// C `compute_overlap_window`.
#[cfg(feature = "osce")]
#[must_use]
pub fn compute_overlap_window(n: usize) -> Vec<f32> {
    let mut w = vec![0f32; n];
    // SAFETY: w has n floats.
    unsafe { oracle_dc_compute_overlap_window(w.as_mut_ptr(), ci(n)) };
    w
}

/// C `scale_kernel` (nndsp.c static copy).
#[cfg(feature = "osce")]
pub fn scale_kernel(kernel: &mut [f32], in_ch: usize, out_ch: usize, ksize: usize, gain: &[f32]) {
    assert!(kernel.len() >= in_ch * out_ch * ksize && gain.len() >= out_ch);
    let mut g = gain.to_vec();
    // SAFETY: sizes checked (C only reads gain).
    unsafe {
        oracle_dc_scale_kernel(
            kernel.as_mut_ptr(),
            ci(in_ch),
            ci(out_ch),
            ci(ksize),
            g.as_mut_ptr(),
        )
    }
}

/// C `transform_gains` (nndsp.c static copy).
#[cfg(feature = "osce")]
pub fn transform_gains(gains: &mut [f32], a: f32, b: f32) {
    // SAFETY: n = gains.len().
    unsafe { oracle_dc_transform_gains(gains.as_mut_ptr(), ci(gains.len()), a, b) }
}

/// C `AdaConvState`.
#[cfg(feature = "osce")]
#[derive(Debug)]
pub struct AdaConv(*mut c_void);

/// Parameters of an `adaconv_process_frame` call.
#[cfg(feature = "osce")]
#[derive(Debug, Clone, Copy)]
pub struct AdaConvParams {
    pub feature_dim: usize,
    pub frame_size: usize,
    pub overlap_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub left_padding: usize,
    pub filter_gain_a: f32,
    pub filter_gain_b: f32,
    pub shape_gain: f32,
}

#[cfg(feature = "osce")]
impl AdaConv {
    #[must_use]
    #[expect(clippy::new_without_default, reason = "allocates C state")]
    pub fn new() -> Self {
        // SAFETY: allocates and initialises a C state.
        Self(unsafe { oracle_dc_adaconv_new() })
    }

    /// C `adaconv_process_frame` (`x_in: None` = in place).
    #[expect(clippy::too_many_arguments, reason = "mirrors the C call")]
    pub fn process(
        &mut self,
        x_out: &mut [f32],
        x_in: Option<&[f32]>,
        features: &[f32],
        kernel: &Linear,
        gain: &Linear,
        p: &AdaConvParams,
        window: &[f32],
    ) {
        assert!(p.in_channels <= 3 && p.out_channels <= 3 && p.kernel_size <= 32);
        assert!(p.frame_size <= 240 && p.overlap_size <= 120 && p.overlap_size <= p.frame_size);
        assert!(p.kernel_size < p.frame_size && p.left_padding + 1 == p.kernel_size);
        // Largest input read: last channel start + frame_size + 32 + 3 within 3*(240+32).
        assert!(
            1 + (p.in_channels - 1) * (p.frame_size + p.kernel_size) + p.frame_size + 35 <= 3 * 272
        );
        assert!(
            kernel.nb_out == p.in_channels * p.out_channels * p.kernel_size && kernel.nb_out <= 288
        );
        assert!(gain.nb_out == p.out_channels);
        assert!(kernel.usable && gain.usable);
        assert!(features.len() >= kernel.nb_in && features.len() >= gain.nb_in);
        assert!(x_out.len() >= p.out_channels * p.frame_size);
        assert!(x_in.map_or(x_out.len(), <[f32]>::len) >= p.in_channels * p.frame_size);
        assert!(window.len() >= p.overlap_size);
        let mut w = window.to_vec();
        // SAFETY: sizes checked (C only reads `window`).
        unsafe {
            oracle_dc_adaconv(
                self.0,
                x_out.as_mut_ptr(),
                opt_ptr(x_in),
                features.as_ptr(),
                kernel.ptr(),
                gain.ptr(),
                ci(p.feature_dim),
                ci(p.frame_size),
                ci(p.overlap_size),
                ci(p.in_channels),
                ci(p.out_channels),
                ci(p.kernel_size),
                ci(p.left_padding),
                p.filter_gain_a,
                p.filter_gain_b,
                p.shape_gain,
                w.as_mut_ptr(),
                c_int::from(x_in.is_none()),
            );
        }
    }

    /// Flattened state (see [`ADACONV_STATE_LEN`]).
    #[must_use]
    pub fn state(&self) -> Vec<f32> {
        let mut v = vec![0f32; ADACONV_STATE_LEN];
        // SAFETY: v has the size C writes.
        unsafe { oracle_dc_adaconv_state(self.0, v.as_mut_ptr()) };
        v
    }
}

#[cfg(feature = "osce")]
impl Drop for AdaConv {
    fn drop(&mut self) {
        // SAFETY: allocated by oracle_dc_adaconv_new.
        unsafe { oracle_dc_nndsp_free(self.0) }
    }
}

/// C `AdaCombState`.
#[cfg(feature = "osce")]
#[derive(Debug)]
pub struct AdaComb(*mut c_void);

/// Parameters of an `adacomb_process_frame` call.
#[cfg(feature = "osce")]
#[derive(Debug, Clone, Copy)]
pub struct AdaCombParams {
    pub feature_dim: usize,
    pub frame_size: usize,
    pub overlap_size: usize,
    pub kernel_size: usize,
    pub left_padding: usize,
    pub filter_gain_a: f32,
    pub filter_gain_b: f32,
    pub log_gain_limit: f32,
}

#[cfg(feature = "osce")]
impl AdaComb {
    #[must_use]
    #[expect(clippy::new_without_default, reason = "allocates C state")]
    pub fn new() -> Self {
        // SAFETY: allocates and initialises a C state.
        Self(unsafe { oracle_dc_adacomb_new() })
    }

    /// C `adacomb_process_frame` (`x_in: None` = in place).
    #[expect(clippy::too_many_arguments, reason = "mirrors the C call")]
    pub fn process(
        &mut self,
        x_out: &mut [f32],
        x_in: Option<&[f32]>,
        features: &[f32],
        kernel: &Linear,
        gain: &Linear,
        global_gain: &Linear,
        pitch_lag: usize,
        p: &AdaCombParams,
        window: &[f32],
    ) {
        assert!(p.kernel_size <= 16 && p.frame_size <= 80 && p.overlap_size <= 40);
        assert!(p.overlap_size <= p.frame_size && p.left_padding < p.kernel_size.max(1));
        assert!(p.frame_size >= p.kernel_size);
        let last = self.last_lag();
        for (lag, max_pitch) in [(pitch_lag, p.frame_size), (last, p.overlap_size)] {
            // Reads input_buffer[start .. start + max_pitch + 16 - 2] (celt_pitch_xcorr with
            // len 16), start = kernel_size + 300 - left_padding - lag, within 396 floats.
            assert!(lag + p.left_padding <= p.kernel_size + 300);
            assert!(p.kernel_size + 300 - p.left_padding - lag + max_pitch + 14 < 396);
        }
        assert!(kernel.nb_out == p.kernel_size && gain.nb_out == 1 && global_gain.nb_out == 1);
        assert!(kernel.usable && gain.usable && global_gain.usable);
        assert!(features.len() >= kernel.nb_in.max(gain.nb_in).max(global_gain.nb_in));
        assert!(
            x_out.len() >= p.frame_size && x_in.map_or(x_out.len(), <[f32]>::len) >= p.frame_size
        );
        assert!(window.len() >= p.overlap_size);
        let mut w = window.to_vec();
        // SAFETY: sizes checked (C only reads `window`).
        unsafe {
            oracle_dc_adacomb(
                self.0,
                x_out.as_mut_ptr(),
                opt_ptr(x_in),
                features.as_ptr(),
                kernel.ptr(),
                gain.ptr(),
                global_gain.ptr(),
                ci(pitch_lag),
                ci(p.feature_dim),
                ci(p.frame_size),
                ci(p.overlap_size),
                ci(p.kernel_size),
                ci(p.left_padding),
                p.filter_gain_a,
                p.filter_gain_b,
                p.log_gain_limit,
                w.as_mut_ptr(),
                c_int::from(x_in.is_none()),
            );
        }
    }

    fn last_lag(&self) -> usize {
        let mut v = vec![0f32; ADACOMB_STATE_LEN];
        let mut lag = 0;
        // SAFETY: v has the size C writes.
        unsafe { oracle_dc_adacomb_state(self.0, v.as_mut_ptr(), &mut lag) };
        lag as usize
    }

    /// Flattened state (see [`ADACOMB_STATE_LEN`]) and `last_pitch_lag`.
    #[must_use]
    pub fn state(&self) -> (Vec<f32>, i32) {
        let mut v = vec![0f32; ADACOMB_STATE_LEN];
        let mut lag = 0;
        // SAFETY: v has the size C writes.
        unsafe { oracle_dc_adacomb_state(self.0, v.as_mut_ptr(), &mut lag) };
        (v, lag)
    }
}

#[cfg(feature = "osce")]
impl Drop for AdaComb {
    fn drop(&mut self) {
        // SAFETY: allocated by oracle_dc_adacomb_new.
        unsafe { oracle_dc_nndsp_free(self.0) }
    }
}

/// C `AdaShapeState`.
#[cfg(feature = "osce")]
#[derive(Debug)]
pub struct AdaShape(*mut c_void);

#[cfg(feature = "osce")]
impl AdaShape {
    #[must_use]
    #[expect(clippy::new_without_default, reason = "allocates C state")]
    pub fn new() -> Self {
        // SAFETY: allocates and initialises a C state.
        Self(unsafe { oracle_dc_adashape_new() })
    }

    /// C `adashape_process_frame` (`x_in: None` = in place).
    #[expect(clippy::too_many_arguments, reason = "mirrors the C call")]
    pub fn process(
        &mut self,
        x_out: &mut [f32],
        x_in: Option<&[f32]>,
        features: &[f32],
        a1f: &Linear,
        a1t: &Linear,
        a2: &Linear,
        feature_dim: usize,
        frame_size: usize,
        avg_pool_k: usize,
        interpolate_k: usize,
    ) {
        assert!(
            frame_size <= 240
                && frame_size.is_multiple_of(avg_pool_k)
                && frame_size.is_multiple_of(interpolate_k)
        );
        let tenv = frame_size / avg_pool_k;
        let hidden = frame_size / interpolate_k;
        assert!(feature_dim + tenv + 1 < 512);
        assert!(a1f.nb_in <= 512 && a1t.nb_in <= 512 && a2.nb_in <= 240);
        assert!(a1f.nb_in >= feature_dim && a1t.nb_in > tenv && a2.nb_in >= hidden);
        assert!(
            a1f.nb_out >= hidden && a1t.nb_out >= hidden && a1f.nb_out <= 240 && a1t.nb_out <= 240
        );
        assert!(a2.nb_out >= hidden && a2.nb_out <= 240);
        assert!(features.len() >= feature_dim);
        assert!(a1f.usable && a1t.usable && a2.usable);
        assert!(x_out.len() >= frame_size && x_in.map_or(x_out.len(), <[f32]>::len) >= frame_size);
        // SAFETY: sizes checked.
        unsafe {
            oracle_dc_adashape(
                self.0,
                x_out.as_mut_ptr(),
                opt_ptr(x_in),
                features.as_ptr(),
                a1f.ptr(),
                a1t.ptr(),
                a2.ptr(),
                ci(feature_dim),
                ci(frame_size),
                ci(avg_pool_k),
                ci(interpolate_k),
                c_int::from(x_in.is_none()),
            );
        }
    }

    /// Flattened state (see [`ADASHAPE_STATE_LEN`]).
    #[must_use]
    pub fn state(&self) -> Vec<f32> {
        let mut v = vec![0f32; ADASHAPE_STATE_LEN];
        // SAFETY: v has the size C writes.
        unsafe { oracle_dc_adashape_state(self.0, v.as_mut_ptr()) };
        v
    }
}

#[cfg(feature = "osce")]
impl Drop for AdaShape {
    fn drop(&mut self) {
        // SAFETY: allocated by oracle_dc_adashape_new.
        unsafe { oracle_dc_nndsp_free(self.0) }
    }
}
