//! Port of `silk/float/*.c` and `silk/float/*.h`: the floating-point SILK encoder analysis
//! (`structs_FLP.h`, `main_FLP.h`, `SigProc_FLP.h` and every `*_FLP.c` file).
//!
//! The code is split into private submodules (one per group of C files) and re-exported here:
//! * `structs`: `structs_FLP.h` (encoder state / control structs),
//! * `sigproc`: `SigProc_FLP.h` helpers and the leaf DSP files (energy, inner product,
//!   autocorrelation, Burg, Schur, k2a, windows, LPC/LTP analysis filters, sorting, ...),
//! * `pitch_analysis_core`: `pitch_analysis_core_FLP.c`,
//! * `find`: `find_LPC_FLP.c`, `find_LTP_FLP.c`, `find_pitch_lags_FLP.c`,
//!   `find_pred_coefs_FLP.c`, `LTP_scale_ctrl_FLP.c`,
//! * `noise_shape`: `noise_shape_analysis_FLP.c`, `process_gains_FLP.c`,
//! * `wrappers`: `wrappers_FLP.c`,
//! * `encode_frame`: `encode_frame_FLP.c`.
//!
//! FIXED_POINT: not ported (float build) — `silk/fixed/*` is the fixed-point counterpart of
//! this module. The x86 AVX2 `inner_product_FLP` override is dropped (no intrinsics). These
//! files contain no QEXT, CUSTOM_MODES or DNN code.

#![allow(
    clippy::too_many_arguments,
    reason = "function signatures mirror the C sources one-to-one"
)]
#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]
#![cfg_attr(
    not(feature = "internals"),
    allow(
        unused_imports,
        reason = "re-exports for tests and later units; unused until the Opus encoder lands"
    )
)]

mod encode_frame;
mod find;
mod noise_shape;
mod pitch_analysis_core;
mod sigproc;
mod structs;
mod wrappers;

pub use encode_frame::{silk_encode_do_vad_flp, silk_encode_frame_flp};
pub use find::{
    silk_find_lpc_flp, silk_find_ltp_flp, silk_find_pitch_lags_flp, silk_find_pred_coefs_flp,
    silk_ltp_scale_ctrl_flp,
};
pub use noise_shape::{
    limit_coefs, silk_noise_shape_analysis_flp, silk_process_gains_flp, warped_gain,
    warped_true2monic_coefs,
};
pub use pitch_analysis_core::{
    Stage3Array, silk_p_ana_calc_corr_st3, silk_p_ana_calc_energy_st3, silk_pitch_analysis_core_flp,
};
pub use sigproc::{
    PI_FLP, silk_abs_float, silk_apply_sine_window_flp, silk_autocorrelation_flp,
    silk_burg_modified_flp, silk_bwexpander_flp, silk_corr_matrix_flp, silk_corr_vector_flp,
    silk_energy_flp, silk_float2int, silk_float2short_array, silk_inner_product_flp,
    silk_insertion_sort_decreasing_flp, silk_k2a_flp, silk_log2, silk_lpc_analysis_filter_flp,
    silk_lpc_inverse_pred_gain_flp, silk_ltp_analysis_filter_flp, silk_max_float, silk_min_float,
    silk_regularize_correlations_flp, silk_residual_energy_covar_flp, silk_residual_energy_flp,
    silk_scale_copy_vector_flp, silk_scale_vector_flp, silk_schur_flp, silk_short2float_array,
    silk_sigmoid, silk_warped_autocorrelation_flp,
};
pub use structs::{SilkEncoderControlFlp, SilkEncoderStateFlp, SilkShapeStateFlp, X_BUF_LEN};
pub use wrappers::{
    silk_a2nlsf_flp, silk_nlsf2a_flp, silk_nsq_wrapper_flp, silk_process_nlsfs_flp,
    silk_quant_ltp_gains_flp,
};
