//! Port of dnn/dred_rdovae_enc.c, dred_rdovae_dec.c, dred_rdovae.h, dred_rdovae_enc.h,
//! dred_rdovae_dec.h, the generated dred_rdovae_constants.h, the model bindings
//! (`init_rdovaeenc` / `init_rdovaedec`) and layer sizes of the generated
//! dred_rdovae_enc_data.c/h and dred_rdovae_dec_data.c/h (libopus 1.6.1 model, checkpoint
//! `checkpoint_epoch_1.pth`), and the quantization statistics of dred_rdovae_stats_data.c/h
//! (submodule `stats_data`, re-exported here).
//!
//! Upstream compiles the model arrays in; the port always binds them from a weight blob
//! ([`dred_rdovae_enc_load_model`] / [`dred_rdovae_dec_load_model`], or [`init_rdovaeenc`] /
//! [`init_rdovaedec`] on already parsed arrays), per PLAN D-015. The statistics tables are not
//! part of the blob upstream (always compiled in), so they are Rust constants.

mod stats_data;

use alloc::format;
use alloc::sync::Arc;

pub use stats_data::*;

use crate::Result;

use super::nnet::{
    ACTIVATION_LINEAR, ACTIVATION_TANH, LinearLayer, compute_generic_conv1d,
    compute_generic_conv1d_dilation, compute_generic_dense, compute_generic_gru, compute_glu,
};
use super::parse_lpcnet_weights::{
    ModelCache, WeightArray, linear_init, load_shared, parse_weights,
};

// ---- dred_rdovae_constants.h ----

/// `DRED_NUM_FEATURES`.
pub const DRED_NUM_FEATURES: usize = 20;
/// `DRED_LATENT_DIM`.
pub const DRED_LATENT_DIM: usize = 25;
/// `DRED_STATE_DIM`.
pub const DRED_STATE_DIM: usize = 50;
/// `DRED_PADDED_LATENT_DIM`.
pub const DRED_PADDED_LATENT_DIM: usize = 32;
/// `DRED_PADDED_STATE_DIM`.
pub const DRED_PADDED_STATE_DIM: usize = 56;
/// `DRED_NUM_QUANTIZATION_LEVELS`.
pub const DRED_NUM_QUANTIZATION_LEVELS: usize = 16;
/// `DRED_MAX_RNN_NEURONS`.
pub const DRED_MAX_RNN_NEURONS: usize = 64;
/// `DRED_MAX_CONV_INPUTS`.
pub const DRED_MAX_CONV_INPUTS: usize = 128;
/// `DRED_ENC_MAX_RNN_NEURONS`.
pub const DRED_ENC_MAX_RNN_NEURONS: usize = 128;
/// `DRED_ENC_MAX_CONV_INPUTS`.
pub const DRED_ENC_MAX_CONV_INPUTS: usize = 128;
/// `DRED_DEC_MAX_RNN_NEURONS`.
pub const DRED_DEC_MAX_RNN_NEURONS: usize = 64;

// ---- dred_rdovae_enc_data.h ----

/// `ENC_DENSE1_OUT_SIZE`.
pub const ENC_DENSE1_OUT_SIZE: usize = 64;
/// `ENC_ZDENSE_OUT_SIZE`.
pub const ENC_ZDENSE_OUT_SIZE: usize = 32;
/// `GDENSE2_OUT_SIZE`.
pub const GDENSE2_OUT_SIZE: usize = 56;
/// `GDENSE1_OUT_SIZE`.
pub const GDENSE1_OUT_SIZE: usize = 128;
/// `ENC_CONV_DENSE1_OUT_SIZE` (all five `ENC_CONV_DENSE*_OUT_SIZE` are equal).
pub const ENC_CONV_DENSE_OUT_SIZE: usize = 64;
/// `ENC_GRU1_OUT_SIZE` .. `ENC_GRU5_OUT_SIZE`.
pub const ENC_GRU_OUT_SIZE: usize = 32;
/// `ENC_GRU1_STATE_SIZE` .. `ENC_GRU5_STATE_SIZE`.
pub const ENC_GRU_STATE_SIZE: usize = 32;
/// `ENC_CONV1_OUT_SIZE` .. `ENC_CONV5_OUT_SIZE`.
pub const ENC_CONV_OUT_SIZE: usize = 64;
/// `ENC_CONV1_IN_SIZE` .. `ENC_CONV5_IN_SIZE`.
pub const ENC_CONV_IN_SIZE: usize = 64;
/// `ENC_CONV1_STATE_SIZE` .. `ENC_CONV5_STATE_SIZE` (`64 * (1)`).
pub const ENC_CONV_STATE_SIZE: usize = 64;
/// Size of the concatenation buffer of `dred_rdovae_encode_dframe`.
pub const ENC_BUFFER_SIZE: usize =
    ENC_DENSE1_OUT_SIZE + 5 * ENC_GRU_OUT_SIZE + 5 * ENC_CONV_OUT_SIZE;

// ---- dred_rdovae_dec_data.h ----

/// `DEC_DENSE1_OUT_SIZE`.
pub const DEC_DENSE1_OUT_SIZE: usize = 96;
/// `DEC_GLU1_OUT_SIZE` .. `DEC_GLU5_OUT_SIZE`.
pub const DEC_GLU_OUT_SIZE: usize = 64;
/// `DEC_HIDDEN_INIT_OUT_SIZE`.
pub const DEC_HIDDEN_INIT_OUT_SIZE: usize = 128;
/// `DEC_OUTPUT_OUT_SIZE`.
pub const DEC_OUTPUT_OUT_SIZE: usize = 80;
/// `DEC_CONV_DENSE1_OUT_SIZE` .. `DEC_CONV_DENSE5_OUT_SIZE`.
pub const DEC_CONV_DENSE_OUT_SIZE: usize = 32;
/// `DEC_GRU_INIT_OUT_SIZE`.
pub const DEC_GRU_INIT_OUT_SIZE: usize = 320;
/// `DEC_GRU1_OUT_SIZE` .. `DEC_GRU5_OUT_SIZE`.
pub const DEC_GRU_OUT_SIZE: usize = 64;
/// `DEC_GRU1_STATE_SIZE` .. `DEC_GRU5_STATE_SIZE`.
pub const DEC_GRU_STATE_SIZE: usize = 64;
/// `DEC_CONV1_OUT_SIZE` .. `DEC_CONV5_OUT_SIZE`.
pub const DEC_CONV_OUT_SIZE: usize = 32;
/// `DEC_CONV1_IN_SIZE` .. `DEC_CONV5_IN_SIZE`.
pub const DEC_CONV_IN_SIZE: usize = 32;
/// `DEC_CONV1_STATE_SIZE` .. `DEC_CONV5_STATE_SIZE` (`32 * (1)`).
pub const DEC_CONV_STATE_SIZE: usize = 32;
/// Size of the concatenation buffer of `dred_rdovae_decode_qframe`.
pub const DEC_BUFFER_SIZE: usize =
    DEC_DENSE1_OUT_SIZE + 5 * DEC_GRU_OUT_SIZE + 5 * DEC_CONV_OUT_SIZE;

// ---- model structs and bindings ----

/// `struct RDOVAEEnc` (dred_rdovae_enc_data.h): the encoder model layers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RdovaeEnc {
    pub enc_dense1: LinearLayer,
    pub enc_zdense: LinearLayer,
    pub gdense2: LinearLayer,
    pub gdense1: LinearLayer,
    pub enc_conv_dense1: LinearLayer,
    pub enc_conv_dense2: LinearLayer,
    pub enc_conv_dense3: LinearLayer,
    pub enc_conv_dense4: LinearLayer,
    pub enc_conv_dense5: LinearLayer,
    pub enc_gru1_input: LinearLayer,
    pub enc_gru1_recurrent: LinearLayer,
    pub enc_gru2_input: LinearLayer,
    pub enc_gru2_recurrent: LinearLayer,
    pub enc_gru3_input: LinearLayer,
    pub enc_gru3_recurrent: LinearLayer,
    pub enc_gru4_input: LinearLayer,
    pub enc_gru4_recurrent: LinearLayer,
    pub enc_gru5_input: LinearLayer,
    pub enc_gru5_recurrent: LinearLayer,
    pub enc_conv1: LinearLayer,
    pub enc_conv2: LinearLayer,
    pub enc_conv3: LinearLayer,
    pub enc_conv4: LinearLayer,
    pub enc_conv5: LinearLayer,
}

/// `struct RDOVAEDec` (dred_rdovae_dec_data.h): the decoder model layers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RdovaeDec {
    pub dec_dense1: LinearLayer,
    pub dec_glu1: LinearLayer,
    pub dec_glu2: LinearLayer,
    pub dec_glu3: LinearLayer,
    pub dec_glu4: LinearLayer,
    pub dec_glu5: LinearLayer,
    pub dec_hidden_init: LinearLayer,
    pub dec_output: LinearLayer,
    pub dec_conv_dense1: LinearLayer,
    pub dec_conv_dense2: LinearLayer,
    pub dec_conv_dense3: LinearLayer,
    pub dec_conv_dense4: LinearLayer,
    pub dec_conv_dense5: LinearLayer,
    pub dec_gru_init: LinearLayer,
    pub dec_gru1_input: LinearLayer,
    pub dec_gru1_recurrent: LinearLayer,
    pub dec_gru2_input: LinearLayer,
    pub dec_gru2_recurrent: LinearLayer,
    pub dec_gru3_input: LinearLayer,
    pub dec_gru3_recurrent: LinearLayer,
    pub dec_gru4_input: LinearLayer,
    pub dec_gru4_recurrent: LinearLayer,
    pub dec_gru5_input: LinearLayer,
    pub dec_gru5_recurrent: LinearLayer,
    pub dec_conv1: LinearLayer,
    pub dec_conv2: LinearLayer,
    pub dec_conv3: LinearLayer,
    pub dec_conv4: LinearLayer,
    pub dec_conv5: LinearLayer,
}

/// Array-name pattern of one generated `linear_init` call.
#[derive(Clone, Copy)]
enum Kind {
    /// `"<n>_bias", NULL, NULL, "<n>_weights_float", NULL, NULL, NULL`.
    Float,
    /// `"<n>_bias", "<n>_subias", "<n>_weights_int8", "<n>_weights_float", NULL, NULL,
    /// "<n>_scale"`.
    Int8,
    /// As [`Kind::Int8`] plus `"<n>_weights_idx"` (sparse).
    Sparse,
}

/// One generated `linear_init(&model-><name>, arrays, ...)` line.
fn layer(
    arrays: &[WeightArray<'_>],
    name: &str,
    kind: Kind,
    nb_inputs: usize,
    nb_outputs: usize,
) -> Result<LinearLayer> {
    let bias = format!("{name}_bias");
    let subias = format!("{name}_subias");
    let weights = format!("{name}_weights_int8");
    let float_weights = format!("{name}_weights_float");
    let idx = format!("{name}_weights_idx");
    let scale = format!("{name}_scale");
    let (subias, weights, idx, scale) = match kind {
        Kind::Float => (None, None, None, None),
        Kind::Int8 => (Some(&*subias), Some(&*weights), None, Some(&*scale)),
        Kind::Sparse => (Some(&*subias), Some(&*weights), Some(&*idx), Some(&*scale)),
    };
    linear_init(
        arrays,
        Some(&bias),
        subias,
        weights,
        Some(&float_weights),
        idx,
        None,
        scale,
        nb_inputs,
        nb_outputs,
    )
}

/// Port of the generated dnn/dred_rdovae_enc_data.c:init_rdovaeenc (fails with `BadArg` where
/// C returns 1).
pub fn init_rdovaeenc(arrays: &[WeightArray<'_>]) -> Result<RdovaeEnc> {
    use Kind::{Float, Int8, Sparse};
    let a = arrays;
    Ok(RdovaeEnc {
        enc_dense1: layer(a, "enc_dense1", Float, 40, 64)?,
        enc_zdense: layer(a, "enc_zdense", Int8, 544, 32)?,
        gdense2: layer(a, "gdense2", Int8, 128, 56)?,
        gdense1: layer(a, "gdense1", Sparse, 544, 128)?,
        enc_conv_dense1: layer(a, "enc_conv_dense1", Sparse, 96, 64)?,
        enc_conv_dense2: layer(a, "enc_conv_dense2", Sparse, 192, 64)?,
        enc_conv_dense3: layer(a, "enc_conv_dense3", Sparse, 288, 64)?,
        enc_conv_dense4: layer(a, "enc_conv_dense4", Sparse, 384, 64)?,
        enc_conv_dense5: layer(a, "enc_conv_dense5", Sparse, 480, 64)?,
        enc_gru1_input: layer(a, "enc_gru1_input", Sparse, 64, 96)?,
        enc_gru1_recurrent: layer(a, "enc_gru1_recurrent", Int8, 32, 96)?,
        enc_gru2_input: layer(a, "enc_gru2_input", Sparse, 160, 96)?,
        enc_gru2_recurrent: layer(a, "enc_gru2_recurrent", Int8, 32, 96)?,
        enc_gru3_input: layer(a, "enc_gru3_input", Sparse, 256, 96)?,
        enc_gru3_recurrent: layer(a, "enc_gru3_recurrent", Int8, 32, 96)?,
        enc_gru4_input: layer(a, "enc_gru4_input", Sparse, 352, 96)?,
        enc_gru4_recurrent: layer(a, "enc_gru4_recurrent", Int8, 32, 96)?,
        enc_gru5_input: layer(a, "enc_gru5_input", Sparse, 448, 96)?,
        enc_gru5_recurrent: layer(a, "enc_gru5_recurrent", Int8, 32, 96)?,
        enc_conv1: layer(a, "enc_conv1", Int8, 128, 64)?,
        enc_conv2: layer(a, "enc_conv2", Int8, 128, 64)?,
        enc_conv3: layer(a, "enc_conv3", Int8, 128, 64)?,
        enc_conv4: layer(a, "enc_conv4", Int8, 128, 64)?,
        enc_conv5: layer(a, "enc_conv5", Int8, 128, 64)?,
    })
}

/// Port of the generated dnn/dred_rdovae_dec_data.c:init_rdovaedec (fails with `BadArg` where
/// C returns 1).
pub fn init_rdovaedec(arrays: &[WeightArray<'_>]) -> Result<RdovaeDec> {
    use Kind::{Float, Int8, Sparse};
    let a = arrays;
    Ok(RdovaeDec {
        dec_dense1: layer(a, "dec_dense1", Float, 26, 96)?,
        dec_glu1: layer(a, "dec_glu1", Int8, 64, 64)?,
        dec_glu2: layer(a, "dec_glu2", Int8, 64, 64)?,
        dec_glu3: layer(a, "dec_glu3", Int8, 64, 64)?,
        dec_glu4: layer(a, "dec_glu4", Int8, 64, 64)?,
        dec_glu5: layer(a, "dec_glu5", Int8, 64, 64)?,
        dec_hidden_init: layer(a, "dec_hidden_init", Float, 50, 128)?,
        dec_output: layer(a, "dec_output", Sparse, 576, 80)?,
        dec_conv_dense1: layer(a, "dec_conv_dense1", Sparse, 160, 32)?,
        dec_conv_dense2: layer(a, "dec_conv_dense2", Sparse, 256, 32)?,
        dec_conv_dense3: layer(a, "dec_conv_dense3", Sparse, 352, 32)?,
        dec_conv_dense4: layer(a, "dec_conv_dense4", Sparse, 448, 32)?,
        dec_conv_dense5: layer(a, "dec_conv_dense5", Sparse, 544, 32)?,
        dec_gru_init: layer(a, "dec_gru_init", Sparse, 128, 320)?,
        dec_gru1_input: layer(a, "dec_gru1_input", Sparse, 96, 192)?,
        dec_gru1_recurrent: layer(a, "dec_gru1_recurrent", Int8, 64, 192)?,
        dec_gru2_input: layer(a, "dec_gru2_input", Sparse, 192, 192)?,
        dec_gru2_recurrent: layer(a, "dec_gru2_recurrent", Int8, 64, 192)?,
        dec_gru3_input: layer(a, "dec_gru3_input", Sparse, 288, 192)?,
        dec_gru3_recurrent: layer(a, "dec_gru3_recurrent", Int8, 64, 192)?,
        dec_gru4_input: layer(a, "dec_gru4_input", Sparse, 384, 192)?,
        dec_gru4_recurrent: layer(a, "dec_gru4_recurrent", Int8, 64, 192)?,
        dec_gru5_input: layer(a, "dec_gru5_input", Sparse, 480, 192)?,
        dec_gru5_recurrent: layer(a, "dec_gru5_recurrent", Int8, 64, 192)?,
        dec_conv1: layer(a, "dec_conv1", Int8, 64, 32)?,
        dec_conv2: layer(a, "dec_conv2", Int8, 64, 32)?,
        dec_conv3: layer(a, "dec_conv3", Int8, 64, 32)?,
        dec_conv4: layer(a, "dec_conv4", Int8, 64, 32)?,
        dec_conv5: layer(a, "dec_conv5", Int8, 64, 32)?,
    })
}

/// Parses a weight blob and binds the RDOVAE encoder model (the `parse_weights` +
/// `init_rdovaeenc` part of dnn/dred_encoder.c:dred_encoder_load_model). An unparsable blob is
/// an error (C would dereference a NULL list).
pub fn dred_rdovae_enc_load_model(data: &[u8]) -> Result<RdovaeEnc> {
    let list = parse_weights(data)?;
    init_rdovaeenc(&list)
}

/// Parses a weight blob and binds the RDOVAE decoder model (the body of
/// src/opus_decoder.c:dred_decoder_load_model, which the Opus-decoder unit wraps into
/// `OpusDREDDecoder`). An unparsable blob is an error (C would dereference a NULL list).
pub fn dred_rdovae_dec_load_model(data: &[u8]) -> Result<RdovaeDec> {
    let list = parse_weights(data)?;
    init_rdovaedec(&list)
}

/// The RDOVAE models of the embedded weight blob (see [`load_shared`]).
static ENC_CACHE: ModelCache<RdovaeEnc> = ModelCache::new();
static DEC_CACHE: ModelCache<RdovaeDec> = ModelCache::new();

/// [`dred_rdovae_enc_load_model`] into a shared [`Arc`] (the model of the embedded blob is
/// bound once and shared by every encoder, see [`load_shared`]).
///
/// # Errors
/// As [`dred_rdovae_enc_load_model`].
pub fn dred_rdovae_enc_load_shared(data: &[u8]) -> Result<Arc<RdovaeEnc>> {
    load_shared(data, init_rdovaeenc, &ENC_CACHE)
}

/// [`dred_rdovae_dec_load_model`] into a shared [`Arc`] (the model of the embedded blob is
/// bound once and shared by every DRED decoder, see [`load_shared`]).
///
/// # Errors
/// As [`dred_rdovae_dec_load_model`].
pub fn dred_rdovae_dec_load_shared(data: &[u8]) -> Result<Arc<RdovaeDec>> {
    load_shared(data, init_rdovaedec, &DEC_CACHE)
}

// ---- states ----

/// `struct RDOVAEEncStruct` / `RDOVAEEncState` (dred_rdovae_enc.h). `Default` is the C
/// `memset(0)` (dred_encoder.c:DRED_rdovae_init_encoder).
#[derive(Debug, Clone, PartialEq)]
pub struct RdovaeEncState {
    pub initialized: i32,
    pub gru1_state: [f32; ENC_GRU_STATE_SIZE],
    pub gru2_state: [f32; ENC_GRU_STATE_SIZE],
    pub gru3_state: [f32; ENC_GRU_STATE_SIZE],
    pub gru4_state: [f32; ENC_GRU_STATE_SIZE],
    pub gru5_state: [f32; ENC_GRU_STATE_SIZE],
    pub conv1_state: [f32; ENC_CONV_STATE_SIZE],
    pub conv2_state: [f32; 2 * ENC_CONV_STATE_SIZE],
    pub conv3_state: [f32; 2 * ENC_CONV_STATE_SIZE],
    pub conv4_state: [f32; 2 * ENC_CONV_STATE_SIZE],
    pub conv5_state: [f32; 2 * ENC_CONV_STATE_SIZE],
}

impl Default for RdovaeEncState {
    fn default() -> Self {
        Self::new()
    }
}

impl RdovaeEncState {
    /// A zeroed state (C `memset(enc_state, 0, sizeof(*enc_state))`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            initialized: 0,
            gru1_state: [0.0; ENC_GRU_STATE_SIZE],
            gru2_state: [0.0; ENC_GRU_STATE_SIZE],
            gru3_state: [0.0; ENC_GRU_STATE_SIZE],
            gru4_state: [0.0; ENC_GRU_STATE_SIZE],
            gru5_state: [0.0; ENC_GRU_STATE_SIZE],
            conv1_state: [0.0; ENC_CONV_STATE_SIZE],
            conv2_state: [0.0; 2 * ENC_CONV_STATE_SIZE],
            conv3_state: [0.0; 2 * ENC_CONV_STATE_SIZE],
            conv4_state: [0.0; 2 * ENC_CONV_STATE_SIZE],
            conv5_state: [0.0; 2 * ENC_CONV_STATE_SIZE],
        }
    }
}

/// `struct RDOVAEDecStruct` / `RDOVAEDecState` (dred_rdovae_dec.h). `Default` is the C
/// `memset(0)`.
#[derive(Debug, Clone, PartialEq)]
pub struct RdovaeDecState {
    pub initialized: i32,
    pub gru1_state: [f32; DEC_GRU_STATE_SIZE],
    pub gru2_state: [f32; DEC_GRU_STATE_SIZE],
    pub gru3_state: [f32; DEC_GRU_STATE_SIZE],
    pub gru4_state: [f32; DEC_GRU_STATE_SIZE],
    pub gru5_state: [f32; DEC_GRU_STATE_SIZE],
    pub conv1_state: [f32; DEC_CONV_STATE_SIZE],
    pub conv2_state: [f32; DEC_CONV_STATE_SIZE],
    pub conv3_state: [f32; DEC_CONV_STATE_SIZE],
    pub conv4_state: [f32; DEC_CONV_STATE_SIZE],
    pub conv5_state: [f32; DEC_CONV_STATE_SIZE],
}

impl Default for RdovaeDecState {
    fn default() -> Self {
        Self::new()
    }
}

impl RdovaeDecState {
    /// A zeroed state (C `memset(&dec, 0, sizeof(dec))`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            initialized: 0,
            gru1_state: [0.0; DEC_GRU_STATE_SIZE],
            gru2_state: [0.0; DEC_GRU_STATE_SIZE],
            gru3_state: [0.0; DEC_GRU_STATE_SIZE],
            gru4_state: [0.0; DEC_GRU_STATE_SIZE],
            gru5_state: [0.0; DEC_GRU_STATE_SIZE],
            conv1_state: [0.0; DEC_CONV_STATE_SIZE],
            conv2_state: [0.0; DEC_CONV_STATE_SIZE],
            conv3_state: [0.0; DEC_CONV_STATE_SIZE],
            conv4_state: [0.0; DEC_CONV_STATE_SIZE],
            conv5_state: [0.0; DEC_CONV_STATE_SIZE],
        }
    }
}

/// Port of dnn/dred_rdovae_enc.c:conv1_cond_init / dred_rdovae_dec.c:conv1_cond_init (static,
/// identical in both files).
///
/// Quirk (ported as-is): the RDOVAE states have a single `initialized` flag shared by all five
/// convolutions, so only the first call after a reset clears its memory.
fn conv1_cond_init(mem: &mut [f32], len: usize, dilation: usize, init: &mut i32) {
    if *init == 0 {
        for i in 0..dilation {
            mem[i * len..(i + 1) * len].fill(0.0);
        }
    }
    *init = 1;
}

/// Port of dnn/dred_rdovae_enc.c:dred_rdovae_encode_dframe.
///
/// `input` is a double feature frame (`2*DRED_NUM_FEATURES`); writes `DRED_LATENT_DIM` floats
/// to `latents` and `DRED_STATE_DIM` floats to `initial_state`.
pub fn dred_rdovae_encode_dframe(
    enc_state: &mut RdovaeEncState,
    model: &RdovaeEnc,
    latents: &mut [f32],
    initial_state: &mut [f32],
    input: &[f32],
) {
    let mut padded_latents = [0f32; DRED_PADDED_LATENT_DIM];
    let mut padded_state = [0f32; DRED_PADDED_STATE_DIM];
    let mut buffer = [0f32; ENC_BUFFER_SIZE];
    let mut state_hidden = [0f32; GDENSE1_OUT_SIZE];
    let mut conv_tmp = [0f32; DRED_MAX_CONV_INPUTS];
    let mut output_index = 0;

    // run encoder stack and concatenate output in buffer
    compute_generic_dense(
        &model.enc_dense1,
        &mut buffer[output_index..],
        input,
        ACTIVATION_TANH,
    );
    output_index += ENC_DENSE1_OUT_SIZE;

    let st = enc_state;
    let grus: [(&LinearLayer, &LinearLayer, &LinearLayer, &LinearLayer); 5] = [
        (
            &model.enc_gru1_input,
            &model.enc_gru1_recurrent,
            &model.enc_conv_dense1,
            &model.enc_conv1,
        ),
        (
            &model.enc_gru2_input,
            &model.enc_gru2_recurrent,
            &model.enc_conv_dense2,
            &model.enc_conv2,
        ),
        (
            &model.enc_gru3_input,
            &model.enc_gru3_recurrent,
            &model.enc_conv_dense3,
            &model.enc_conv3,
        ),
        (
            &model.enc_gru4_input,
            &model.enc_gru4_recurrent,
            &model.enc_conv_dense4,
            &model.enc_conv4,
        ),
        (
            &model.enc_gru5_input,
            &model.enc_gru5_recurrent,
            &model.enc_conv_dense5,
            &model.enc_conv5,
        ),
    ];
    for (k, (gru_in, gru_rec, conv_dense, conv)) in grus.into_iter().enumerate() {
        let (gru_state, conv_state): (&mut [f32], &mut [f32]) = match k {
            0 => (&mut st.gru1_state, &mut st.conv1_state),
            1 => (&mut st.gru2_state, &mut st.conv2_state),
            2 => (&mut st.gru3_state, &mut st.conv3_state),
            3 => (&mut st.gru4_state, &mut st.conv4_state),
            _ => (&mut st.gru5_state, &mut st.conv5_state),
        };
        compute_generic_gru(gru_in, gru_rec, gru_state, &buffer);
        buffer[output_index..output_index + ENC_GRU_OUT_SIZE]
            .copy_from_slice(&gru_state[..ENC_GRU_OUT_SIZE]);
        output_index += ENC_GRU_OUT_SIZE;
        // conv1 has no dilation; conv2..conv5 use dilation 2.
        let dilation = if k == 0 { 1 } else { 2 };
        conv1_cond_init(conv_state, ENC_CONV_IN_SIZE, dilation, &mut st.initialized);
        compute_generic_dense(conv_dense, &mut conv_tmp, &buffer, ACTIVATION_TANH);
        let out = &mut buffer[output_index..];
        if k == 0 {
            compute_generic_conv1d(
                conv,
                out,
                conv_state,
                &conv_tmp,
                ENC_CONV_OUT_SIZE,
                ACTIVATION_TANH,
            );
        } else {
            compute_generic_conv1d_dilation(
                conv,
                out,
                conv_state,
                &conv_tmp,
                ENC_CONV_OUT_SIZE,
                2,
                ACTIVATION_TANH,
            );
        }
        output_index += ENC_CONV_OUT_SIZE;
    }

    compute_generic_dense(
        &model.enc_zdense,
        &mut padded_latents,
        &buffer,
        ACTIVATION_LINEAR,
    );
    latents[..DRED_LATENT_DIM].copy_from_slice(&padded_latents[..DRED_LATENT_DIM]);

    // next, calculate initial state
    compute_generic_dense(&model.gdense1, &mut state_hidden, &buffer, ACTIVATION_TANH);
    compute_generic_dense(
        &model.gdense2,
        &mut padded_state,
        &state_hidden,
        ACTIVATION_LINEAR,
    );
    initial_state[..DRED_STATE_DIM].copy_from_slice(&padded_state[..DRED_STATE_DIM]);
}

/// Port of dnn/dred_rdovae_dec.c:DRED_rdovae_decode_all.
///
/// Decodes `nb_latents` latent vectors (each `DRED_LATENT_DIM+1` floats in `latents`) into
/// `4*nb_latents` feature frames (`features`, `DRED_NUM_FEATURES` each) starting from the
/// `DRED_STATE_DIM` initial `state`. Uses a fresh (zeroed) decoder state on the stack.
pub fn dred_rdovae_decode_all(
    model: &RdovaeDec,
    features: &mut [f32],
    state: &[f32],
    latents: &[f32],
    nb_latents: usize,
) {
    let mut dec = RdovaeDecState::new();
    dred_rdovae_dec_init_states(&mut dec, model, state);
    let mut i = 0;
    while i < 2 * nb_latents {
        dred_rdovae_decode_qframe(
            &mut dec,
            model,
            &mut features[2 * i * DRED_NUM_FEATURES..],
            &latents[(i / 2) * (DRED_LATENT_DIM + 1)..],
        );
        i += 2;
    }
}

/// Port of dnn/dred_rdovae_dec.c:dred_rdovae_dec_init_states.
///
/// Quirk (ported as-is): only the GRU states and the `initialized` flag are reset; conv2..conv5
/// memories are never cleared (see [`conv1_cond_init`]).
pub fn dred_rdovae_dec_init_states(
    h: &mut RdovaeDecState,
    model: &RdovaeDec,
    initial_state: &[f32],
) {
    let mut hidden = [0f32; DEC_HIDDEN_INIT_OUT_SIZE];
    let mut state_init = [0f32; 5 * DEC_GRU_STATE_SIZE];
    compute_generic_dense(
        &model.dec_hidden_init,
        &mut hidden,
        initial_state,
        ACTIVATION_TANH,
    );
    compute_generic_dense(
        &model.dec_gru_init,
        &mut state_init,
        &hidden,
        ACTIVATION_TANH,
    );
    let n = DEC_GRU_STATE_SIZE;
    h.gru1_state.copy_from_slice(&state_init[..n]);
    h.gru2_state.copy_from_slice(&state_init[n..2 * n]);
    h.gru3_state.copy_from_slice(&state_init[2 * n..3 * n]);
    h.gru4_state.copy_from_slice(&state_init[3 * n..4 * n]);
    h.gru5_state.copy_from_slice(&state_init[4 * n..5 * n]);
    h.initialized = 0;
}

/// Port of dnn/dred_rdovae_dec.c:dred_rdovae_decode_qframe.
///
/// `input` is one latent vector (`DRED_LATENT_DIM+1` floats, the last being the quantizer
/// level); writes `DEC_OUTPUT_OUT_SIZE` (four concatenated feature frames, newest first) to
/// `qframe`.
pub fn dred_rdovae_decode_qframe(
    dec_state: &mut RdovaeDecState,
    model: &RdovaeDec,
    qframe: &mut [f32],
    input: &[f32],
) {
    let mut buffer = [0f32; DEC_BUFFER_SIZE];
    let mut conv_tmp = [0f32; DRED_MAX_CONV_INPUTS];
    let mut output_index = 0;

    // run encoder stack and concatenate output in buffer
    compute_generic_dense(
        &model.dec_dense1,
        &mut buffer[output_index..],
        input,
        ACTIVATION_TANH,
    );
    output_index += DEC_DENSE1_OUT_SIZE;

    let st = dec_state;
    let stages: [(
        &LinearLayer,
        &LinearLayer,
        &LinearLayer,
        &LinearLayer,
        &LinearLayer,
    ); 5] = [
        (
            &model.dec_gru1_input,
            &model.dec_gru1_recurrent,
            &model.dec_glu1,
            &model.dec_conv_dense1,
            &model.dec_conv1,
        ),
        (
            &model.dec_gru2_input,
            &model.dec_gru2_recurrent,
            &model.dec_glu2,
            &model.dec_conv_dense2,
            &model.dec_conv2,
        ),
        (
            &model.dec_gru3_input,
            &model.dec_gru3_recurrent,
            &model.dec_glu3,
            &model.dec_conv_dense3,
            &model.dec_conv3,
        ),
        (
            &model.dec_gru4_input,
            &model.dec_gru4_recurrent,
            &model.dec_glu4,
            &model.dec_conv_dense4,
            &model.dec_conv4,
        ),
        (
            &model.dec_gru5_input,
            &model.dec_gru5_recurrent,
            &model.dec_glu5,
            &model.dec_conv_dense5,
            &model.dec_conv5,
        ),
    ];
    for (k, (gru_in, gru_rec, glu, conv_dense, conv)) in stages.into_iter().enumerate() {
        let (gru_state, conv_state): (&mut [f32], &mut [f32]) = match k {
            0 => (&mut st.gru1_state, &mut st.conv1_state),
            1 => (&mut st.gru2_state, &mut st.conv2_state),
            2 => (&mut st.gru3_state, &mut st.conv3_state),
            3 => (&mut st.gru4_state, &mut st.conv4_state),
            _ => (&mut st.gru5_state, &mut st.conv5_state),
        };
        compute_generic_gru(gru_in, gru_rec, gru_state, &buffer);
        compute_glu(glu, &mut buffer[output_index..], gru_state);
        output_index += DEC_GRU_OUT_SIZE;
        conv1_cond_init(conv_state, DEC_CONV_IN_SIZE, 1, &mut st.initialized);
        compute_generic_dense(conv_dense, &mut conv_tmp, &buffer, ACTIVATION_TANH);
        compute_generic_conv1d(
            conv,
            &mut buffer[output_index..],
            conv_state,
            &conv_tmp,
            DEC_CONV_OUT_SIZE,
            ACTIVATION_TANH,
        );
        output_index += DEC_CONV_OUT_SIZE;
    }

    compute_generic_dense(&model.dec_output, qframe, &buffer, ACTIVATION_LINEAR);
}
