//! Port of `dnn/lossgen.c` / `dnn/lossgen.h`: the generative packet loss model of libopus
//! `--enable-lossgen` (opus_demo `-sim_loss`, `lossgen_demo`), which produces realistic
//! (bursty) loss patterns for a given average loss rate.
//!
//! Upstream builds it into `opus_demo` and `lossgen_demo` only, never into the library; here it
//! is available with the `lossgen` feature (public as [`crate::lossgen`]). The model is
//! compiled in (the generated `lossgen_data` tables, PLAN D-027);
//! [`LossGenState::lossgen_load_model`] replaces it from a weight blob like C.
//!
//! The model draws its randomness from C `rand()`: [`sample_loss`] takes the `rand()` source
//! as a closure, so a caller that shares one generator with other draws (as `opus_demo` does)
//! reproduces the C sequence. `(float)rand()/(float)RAND_MAX` uses [`RAND_MAX`], glibc's value.
//!
//! The unused `PITCH_MIN_PERIOD` / `PITCH_MAX_PERIOD` / `NB_XCORR_FEATURES` macros of
//! `lossgen.h` are not ported.

pub use super::lossgen_data::LossGen;
use super::lossgen_data::{
    LOSSGEN_DENSE_IN_OUT_SIZE, LOSSGEN_GRU1_STATE_SIZE, LOSSGEN_GRU2_STATE_SIZE, as_weight_arrays,
    init_lossgen, lossgen_arrays,
};
use super::nnet::{
    ACTIVATION_SIGMOID, ACTIVATION_TANH, LinearLayer, compute_generic_dense, compute_generic_gru,
};
use super::parse_lpcnet_weights::parse_weights;
use crate::Result;

/// `RAND_MAX` of the C library whose `rand()` feeds [`sample_loss`] (glibc: `2^31 - 1`).
pub const RAND_MAX: i32 = 2_147_483_647;

/// `LossGenState`.
#[derive(Debug, Clone, PartialEq)]
pub struct LossGenState {
    pub model: LossGen,
    pub gru1_state: [f32; LOSSGEN_GRU1_STATE_SIZE],
    pub gru2_state: [f32; LOSSGEN_GRU2_STATE_SIZE],
    pub last_loss: i32,
    pub used: i32,
}

/// Port of dnn/lossgen.c:compute_generic_gru_lossgen, upstream's verbatim copy of
/// `nnet.c:compute_generic_gru` (with `MAX_RNN_NEURONS_ALL` = 32): the shared port.
pub fn compute_generic_gru_lossgen(
    input_weights: &LinearLayer,
    recurrent_weights: &LinearLayer,
    state: &mut [f32],
    input: &[f32],
) {
    compute_generic_gru(input_weights, recurrent_weights, state, input);
}

/// Port of dnn/lossgen.c:compute_generic_dense_lossgen, upstream's verbatim copy of
/// `nnet.c:compute_generic_dense`: the shared port.
pub fn compute_generic_dense_lossgen(
    layer: &LinearLayer,
    output: &mut [f32],
    input: &[f32],
    activation: i32,
) {
    compute_generic_dense(layer, output, input, activation);
}

/// Port of dnn/lossgen.c:sample_loss_impl.
fn sample_loss_impl(
    st: &mut LossGenState,
    percent_loss: f32,
    rand: &mut dyn FnMut() -> i32,
) -> i32 {
    let mut tmp = [0f32; LOSSGEN_DENSE_IN_OUT_SIZE];
    let mut out = [0f32; 1];
    let input = [st.last_loss as f32, percent_loss];
    let model = &st.model;
    compute_generic_dense_lossgen(&model.lossgen_dense_in, &mut tmp, &input, ACTIVATION_TANH);
    compute_generic_gru_lossgen(
        &model.lossgen_gru1_input,
        &model.lossgen_gru1_recurrent,
        &mut st.gru1_state,
        &tmp,
    );
    compute_generic_gru_lossgen(
        &model.lossgen_gru2_input,
        &model.lossgen_gru2_recurrent,
        &mut st.gru2_state,
        &st.gru1_state,
    );
    compute_generic_dense_lossgen(
        &model.lossgen_dense_out,
        &mut out,
        &st.gru2_state,
        ACTIVATION_SIGMOID,
    );
    let loss = i32::from((rand() as f32) / (RAND_MAX as f32) < out[0]);
    st.last_loss = loss;
    loss
}

/// Port of dnn/lossgen.c:sample_loss: whether the next packet is lost (1) or not (0), for an
/// average loss rate `percent_loss` in `0.0..=1.0` (`opus_demo -sim_loss <perc>` passes
/// `perc*.01f`). `rand` is C `rand()` (values in `0..=`[`RAND_MAX`]).
///
/// The first call runs the model 1000 times first ("due to GRU being initialized with zeros,
/// the first packets aren't quite random, so we skip them").
pub fn sample_loss(st: &mut LossGenState, percent_loss: f32, rand: &mut dyn FnMut() -> i32) -> i32 {
    if st.used == 0 {
        for _ in 0..1000 {
            sample_loss_impl(st, percent_loss, rand);
        }
        st.used = 1;
    }
    sample_loss_impl(st, percent_loss, rand)
}

impl LossGenState {
    /// Port of dnn/lossgen.c:lossgen_init: a cleared state with the compiled-in model.
    #[must_use]
    pub fn lossgen_init() -> Self {
        let owned = lossgen_arrays();
        let arrays = as_weight_arrays(&owned);
        // C: celt_assert(ret == 0). The compiled-in tables bind (checked by the unit test
        // below and the differential tests).
        #[expect(
            clippy::expect_used,
            reason = "the compiled-in lossgen tables always bind (C asserts it too)"
        )]
        let model = init_lossgen(&arrays).expect("compiled-in lossgen model binds");
        Self {
            model,
            gru1_state: [0.0; LOSSGEN_GRU1_STATE_SIZE],
            gru2_state: [0.0; LOSSGEN_GRU2_STATE_SIZE],
            last_loss: 0,
            used: 0,
        }
    }

    /// Port of dnn/lossgen.c:lossgen_load_model: rebinds the model from a weight blob (the
    /// recurrent state is kept, as in C).
    ///
    /// # Errors
    /// [`crate::Error::BadArg`] if the blob is malformed (C: `parse_weights` fails and
    /// `init_lossgen` dereferences the `NULL` list) or lacks a model array (C: -1). The model is
    /// then left unchanged (C leaves it partially rebound).
    pub fn lossgen_load_model(&mut self, data: &[u8]) -> Result<()> {
        let list = parse_weights(data)?;
        self.model = init_lossgen(&list)?;
        Ok(())
    }
}

impl Default for LossGenState {
    fn default() -> Self {
        Self::lossgen_init()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dnn::parse_lpcnet_weights::write_weights;
    use alloc::vec::Vec;

    /// Park-Miller, only to drive the model in these self-contained tests.
    fn lcg(seed: u32) -> impl FnMut() -> i32 {
        let mut s = u64::from(seed.max(1));
        move || {
            s = s * 48271 % 2_147_483_647;
            s as i32
        }
    }

    #[test]
    fn compiled_in_model_binds_and_round_trips_through_a_blob() {
        let st = LossGenState::lossgen_init();
        assert!(st.model.lossgen_gru2_recurrent.weights.is_some());
        let owned = lossgen_arrays();
        let mut blob = Vec::new();
        write_weights(&as_weight_arrays(&owned), &mut blob);
        let mut loaded = LossGenState::lossgen_init();
        loaded.model = LossGen::default();
        loaded.lossgen_load_model(&blob).unwrap();
        assert_eq!(loaded, st);
        assert!(loaded.lossgen_load_model(&blob[..blob.len() - 1]).is_err());
        assert!(loaded.lossgen_load_model(&blob[..64 + 64]).is_err());
    }

    #[test]
    fn loss_rate_follows_the_target() {
        for (target, lo, hi) in [(0.0f32, 0.0, 0.02), (0.1, 0.05, 0.15), (0.3, 0.22, 0.38)] {
            let mut st = LossGenState::lossgen_init();
            let mut rand = lcg(7);
            let n = 4000;
            let lost: i32 = (0..n)
                .map(|_| sample_loss(&mut st, target, &mut rand))
                .sum();
            let rate = lost as f32 / n as f32;
            assert!((lo..=hi).contains(&rate), "target {target}: rate {rate}");
            assert_eq!(st.used, 1);
        }
    }
}
