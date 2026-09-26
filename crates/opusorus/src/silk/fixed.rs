//! Port of `silk/fixed/*.c` and `silk/fixed/*.h`: the fixed-point SILK encoder analysis
//! (`structs_FIX.h`, `main_FIX.h` and every `*_FIX.c` file of `SILK_SOURCES_FIXED`), used
//! instead of [`crate::silk::float`] by the fixed-point build (feature `fixed-point`).
//!
//! The code is split into private submodules (one per group of C files) and re-exported here:
//! * `structs`: `structs_FIX.h` (encoder state / control structs),
//! * `sigproc`: the leaf DSP files (sine window, autocorrelation, Burg, Schur, k2a,
//!   correlation matrices, residual energies, warped autocorrelation, LTP analysis filter,
//!   vector operations),
//! * `pitch_analysis_core`: `pitch_analysis_core_FIX.c`,
//! * `find`: `find_LPC_FIX.c`, `find_LTP_FIX.c`, `find_pitch_lags_FIX.c`,
//!   `find_pred_coefs_FIX.c`, `LTP_scale_ctrl_FIX.c`,
//! * `noise_shape`: `noise_shape_analysis_FIX.c`, `process_gains_FIX.c`,
//! * `encode_frame`: `encode_frame_FIX.c`.
//!
//! The ARM/MIPS/x86 overrides (`silk/fixed/{arm,mips,x86}`) are dropped (no intrinsics). These
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

pub use encode_frame::{silk_encode_do_vad_fix, silk_encode_frame_fix};
pub use find::{
    silk_find_lpc_fix, silk_find_ltp_fix, silk_find_pitch_lags_fix, silk_find_pred_coefs_fix,
    silk_ltp_scale_ctrl_fix,
};
pub use noise_shape::{
    limit_warped_coefs, silk_noise_shape_analysis_fix, silk_process_gains_fix, warped_gain,
};
pub use pitch_analysis_core::{
    PeStage3Vals, Stage3Array, silk_p_ana_calc_corr_st3, silk_p_ana_calc_energy_st3,
    silk_pitch_analysis_core,
};
pub use sigproc::{
    silk_apply_sine_window, silk_autocorr, silk_burg_modified, silk_corr_matrix_fix,
    silk_corr_vector_fix, silk_inner_prod_aligned, silk_k2a, silk_k2a_q16,
    silk_ltp_analysis_filter_fix, silk_regularize_correlations_fix, silk_residual_energy_fix,
    silk_residual_energy16_covar_fix, silk_scale_copy_vector16, silk_scale_vector32_q26_lshift_18,
    silk_schur, silk_schur64, silk_warped_autocorrelation_fix,
};
pub use structs::{SilkEncoderControlFix, SilkEncoderStateFix, SilkShapeStateFix, X_BUF_LEN};
