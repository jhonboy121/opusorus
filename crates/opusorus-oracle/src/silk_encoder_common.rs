//! Oracle bindings for unit `silk_encoder_common`: NSQ / NSQ_del_dec, VAD, encode_indices,
//! process_NLSFs, quant_LTP_gains, VQ_WMat_EC, stereo LR->MS, HP_variable_cutoff,
//! control_SNR, control_audio_bandwidth, check_control_input (see
//! `csrc/silk_encoder_common.c`).
//!
//! The NSQ, VAD, stereo and side-info states are `#[repr(C)]` mirrors of the C structs (layout
//! checked by [`c_sizes`]); encoder-state based shims take the flat [`EncParams`].
//! Wrappers check buffer sizes with `assert!` before calling into C.

use core::ffi::c_int;

const MAX_FRAME_LENGTH: usize = 320;
const MAX_SUB_FRAME_LENGTH: usize = 80;
const NSQ_LPC_BUF_LENGTH: usize = 16;
const MAX_SHAPE_LPC_ORDER: usize = 24;
const MAX_LPC_ORDER: usize = 16;
const MAX_NB_SUBFR: usize = 4;
const LTP_ORDER: usize = 5;
const MAX_FRAMES_PER_PACKET: usize = 3;

/// `silk_nsq_state` (C layout).
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NsqState {
    pub xq: [i16; 2 * MAX_FRAME_LENGTH],
    pub s_ltp_shp_q14: [i32; 2 * MAX_FRAME_LENGTH],
    pub s_lpc_q14: [i32; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
    pub s_ar2_q14: [i32; MAX_SHAPE_LPC_ORDER],
    pub s_lf_ar_shp_q14: i32,
    pub s_diff_shp_q14: i32,
    pub lag_prev: c_int,
    pub s_ltp_buf_idx: c_int,
    pub s_ltp_shp_buf_idx: c_int,
    pub rand_seed: i32,
    pub prev_gain_q16: i32,
    pub rewhite_flag: c_int,
}

impl Default for NsqState {
    fn default() -> Self {
        Self {
            xq: [0; 2 * MAX_FRAME_LENGTH],
            s_ltp_shp_q14: [0; 2 * MAX_FRAME_LENGTH],
            s_lpc_q14: [0; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
            s_ar2_q14: [0; MAX_SHAPE_LPC_ORDER],
            s_lf_ar_shp_q14: 0,
            s_diff_shp_q14: 0,
            lag_prev: 0,
            s_ltp_buf_idx: 0,
            s_ltp_shp_buf_idx: 0,
            rand_seed: 0,
            prev_gain_q16: 0,
            rewhite_flag: 0,
        }
    }
}

/// `silk_VAD_state` (C layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VadState {
    pub ana_state: [i32; 2],
    pub ana_state1: [i32; 2],
    pub ana_state2: [i32; 2],
    pub xnrg_subfr: [i32; 4],
    pub nrg_ratio_smth_q8: [i32; 4],
    pub hp_state: i16,
    pub nl: [i32; 4],
    pub inv_nl: [i32; 4],
    pub noise_level_bias: [i32; 4],
    pub counter: i32,
}

/// `stereo_enc_state` (C layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StereoEncState {
    pub pred_prev_q13: [i16; 2],
    pub s_mid: [i16; 2],
    pub s_side: [i16; 2],
    pub mid_side_amp_q0: [i32; 4],
    pub smth_width_q14: i16,
    pub width_prev_q14: i16,
    pub silent_side_len: i16,
    pub pred_ix: [[[i8; 3]; 2]; MAX_FRAMES_PER_PACKET],
    pub mid_only_flags: [i8; MAX_FRAMES_PER_PACKET],
}

/// `SideInfoIndices` (C layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SideInfoIndices {
    pub gains_indices: [i8; MAX_NB_SUBFR],
    pub ltp_index: [i8; MAX_NB_SUBFR],
    pub nlsf_indices: [i8; MAX_LPC_ORDER + 1],
    pub lag_index: i16,
    pub contour_index: i8,
    pub signal_type: i8,
    pub quant_offset_type: i8,
    pub nlsf_interp_coef_q2: i8,
    pub per_index: i8,
    pub ltp_scale_index: i8,
    pub seed: i8,
}

/// `silk_EncControlStruct` (C layout: 26 `int`s in declaration order).
pub type EncControl = [i32; 26];

/// Flat subset of `silk_encoder_state` (`oracle_sec_enc_params` in the shim).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EncParams {
    pub fs_khz: c_int,
    pub nb_subfr: c_int,
    pub frame_length: c_int,
    pub subfr_length: c_int,
    pub ltp_mem_length: c_int,
    pub predict_lpc_order: c_int,
    pub shaping_lpc_order: c_int,
    pub n_states_delayed_decision: c_int,
    pub warping_q16: c_int,
    pub speech_activity_q8: c_int,
    pub use_interpolated_nlsfs: c_int,
    pub nlsf_msvq_survivors: c_int,
    pub ec_prev_signal_type: c_int,
    pub ec_prev_lag_index: c_int,
    pub prev_signal_type: c_int,
    pub prev_lag: c_int,
    pub variable_hp_smth1_q15: c_int,
    pub input_quality_bands_q15: [c_int; 4],
    pub input_tilt_q15: c_int,
    pub target_rate_bps: c_int,
    pub snr_db_q7: c_int,
    pub api_fs_hz: c_int,
    pub max_internal_fs_hz: c_int,
    pub min_internal_fs_hz: c_int,
    pub desired_internal_fs_hz: c_int,
    pub allow_bandwidth_switch: c_int,
    pub s_lp_in_lp_state: [c_int; 2],
    pub s_lp_transition_frame_no: c_int,
    pub s_lp_mode: c_int,
    pub s_lp_saved_fs_khz: c_int,
}

/// Range-coder result: `{tell_frac before done, rng after done, error}`.
pub type EncRes = [u32; 3];

/// Per-frame NSQ inputs (same meaning as the `silk_NSQ_c` arguments).
#[derive(Debug, Clone)]
pub struct NsqFrame<'a> {
    pub x16: &'a [i16],
    pub pred_coef_q12: &'a [i16; 2 * MAX_LPC_ORDER],
    pub ltp_coef_q14: &'a [i16; LTP_ORDER * MAX_NB_SUBFR],
    pub ar_q13: &'a [i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
    pub harm_shape_gain_q14: &'a [i32; MAX_NB_SUBFR],
    pub tilt_q14: &'a [i32; MAX_NB_SUBFR],
    pub lf_shp_q14: &'a [i32; MAX_NB_SUBFR],
    pub gains_q16: &'a [i32; MAX_NB_SUBFR],
    pub pitch_l: &'a [i32; MAX_NB_SUBFR],
    pub lambda_q10: i32,
    pub ltp_scale_q14: i32,
}

unsafe extern "C" {
    fn oracle_sec_sizes(out: *mut c_int);
    fn oracle_silk_nsq_short_prediction(
        buf: *const i32,
        len: c_int,
        coef: *const i16,
        order: c_int,
    ) -> i32;
    fn oracle_silk_nsq_feedback_loop(
        data0: i32,
        data1: *mut i32,
        coef: *const i16,
        order: c_int,
    ) -> i32;
    fn oracle_silk_NSQ(
        del_dec: c_int,
        p: *const EncParams,
        nsq: *mut NsqState,
        idx: *mut c_int,
        x16: *const i16,
        pulses: *mut i8,
        pred_coef_q12: *const i16,
        ltp_coef_q14: *const i16,
        ar_q13: *const i16,
        harm_shape_gain_q14: *const c_int,
        tilt_q14: *const c_int,
        lf_shp_q14: *const i32,
        gains_q16: *const i32,
        pitch_l: *const c_int,
        lambda_q10: c_int,
        ltp_scale_q14: c_int,
    );
    fn oracle_silk_VAD_Init(vad: *mut VadState) -> c_int;
    fn oracle_silk_VAD_GetSA_Q8(
        vad: *mut VadState,
        frame_length: c_int,
        fs_khz: c_int,
        p_in: *const i16,
        out: *mut c_int,
    ) -> c_int;
    fn oracle_silk_VAD_GetNoiseLevels(p_x: *const i32, vad: *mut VadState);
    fn oracle_silk_encode_indices(
        p: *mut EncParams,
        sets: *const SideInfoIndices,
        ops: *const c_int,
        n: c_int,
        buf: *mut u8,
        size: c_int,
        res: *mut u32,
    );
    fn oracle_silk_process_NLSFs(
        p: *const EncParams,
        idx: *const c_int,
        nlsf_indices: *mut i8,
        pred_coef_q12: *mut i16,
        p_nlsf_q15: *mut i16,
        prev_nlsfq_q15: *const i16,
    );
    fn oracle_silk_quant_LTP_gains(
        b_q14: *mut i16,
        cbk_index: *mut i8,
        out: *mut c_int,
        xx_q17: *const i32,
        x_x_q17: *const i32,
        subfr_len: c_int,
        nb_subfr: c_int,
    );
    fn oracle_silk_VQ_WMat_EC(
        out: *mut c_int,
        xx_q17: *const i32,
        x_x_q17: *const i32,
        cbk: c_int,
        subfr_len: c_int,
        max_gain_q7: i32,
    );
    fn oracle_silk_stereo_LR_to_MS(
        state: *mut StereoEncState,
        x1: *mut i16,
        x2: *mut i16,
        ix: *mut i8,
        mid_only_flag: *mut i8,
        mid_side_rates_bps: *mut i32,
        total_rate_bps: i32,
        prev_speech_act_q8: c_int,
        to_mono: c_int,
        fs_khz: c_int,
        frame_length: c_int,
    );
    fn oracle_silk_stereo_find_predictor(
        ratio_q14: *mut i32,
        x: *const i16,
        y: *const i16,
        mid_res_amp_q0: *mut i32,
        length: c_int,
        smooth_coef_q16: c_int,
    ) -> i32;
    fn oracle_silk_stereo_quant_pred(pred_q13: *mut i32, ix: *mut i8);
    fn oracle_silk_HP_variable_cutoff(p: *mut EncParams);
    fn oracle_silk_control_SNR(p: *mut EncParams, target_rate_bps: i32) -> c_int;
    fn oracle_silk_control_audio_bandwidth(p: *mut EncParams, ctl: *mut i32) -> c_int;
    fn oracle_check_control_input(ctl: *const i32) -> c_int;
}

/// C `sizeof` of `{silk_nsq_state, silk_VAD_state, stereo_enc_state, SideInfoIndices,
/// silk_EncControlStruct, oracle_sec_enc_params}`.
#[must_use]
pub fn c_sizes() -> [i32; 6] {
    let mut out = [0; 6];
    // SAFETY: the shim writes exactly 6 ints.
    unsafe { oracle_sec_sizes(out.as_mut_ptr()) };
    out
}

/// Rust `size_of` of the mirrors, in the order of [`c_sizes`].
#[must_use]
pub const fn rust_sizes() -> [i32; 6] {
    [
        size_of::<NsqState>() as i32,
        size_of::<VadState>() as i32,
        size_of::<StereoEncState>() as i32,
        size_of::<SideInfoIndices>() as i32,
        size_of::<EncControl>() as i32,
        size_of::<EncParams>() as i32,
    ]
}

/// `silk_noise_shape_quantizer_short_prediction_c` with the C pointer at the last element of
/// `buf`.
#[must_use]
pub fn nsq_short_prediction(buf: &[i32], coef: &[i16], order: i32) -> i32 {
    assert!(order == 10 || order == 16);
    assert!(buf.len() >= order as usize && coef.len() >= order as usize);
    // SAFETY: reads buf[len - order .. len] and coef[..order], both in bounds.
    unsafe {
        oracle_silk_nsq_short_prediction(buf.as_ptr(), buf.len() as c_int, coef.as_ptr(), order)
    }
}

/// `silk_NSQ_noise_shape_feedback_loop_c`.
#[must_use]
pub fn nsq_feedback_loop(data0: i32, data1: &mut [i32], coef: &[i16], order: i32) -> i32 {
    assert!(order >= 2 && order % 2 == 0);
    assert!(data1.len() >= order as usize && coef.len() >= order as usize);
    // SAFETY: accesses data1[..order] and coef[..order], both in bounds.
    unsafe { oracle_silk_nsq_feedback_loop(data0, data1.as_mut_ptr(), coef.as_ptr(), order) }
}

/// `silk_NSQ_c` (`del_dec == false`) or `silk_NSQ_del_dec_c`. `idx = {signalType,
/// quantOffsetType, NLSFInterpCoef_Q2, Seed}`; the seed is updated. Returns the pulses.
pub fn nsq(
    del_dec: bool,
    p: &EncParams,
    nsq: &mut NsqState,
    idx: &mut [i32; 4],
    f: &NsqFrame<'_>,
) -> Vec<i8> {
    let fl = p.frame_length as usize;
    assert!(fl <= MAX_FRAME_LENGTH && f.x16.len() >= fl);
    assert!(p.nb_subfr as usize <= MAX_NB_SUBFR);
    assert!(p.ltp_mem_length as usize + fl <= 2 * MAX_FRAME_LENGTH);
    assert!(p.subfr_length as usize * p.nb_subfr as usize == fl);
    assert!(p.shaping_lpc_order as usize <= MAX_SHAPE_LPC_ORDER);
    assert!((1..=4).contains(&p.n_states_delayed_decision) || !del_dec);
    let mut pulses = vec![0i8; fl];
    // SAFETY: all arrays have the C sizes (fixed-size references) or were checked above; the
    // shim copies the params into a freshly allocated encoder state.
    unsafe {
        oracle_silk_NSQ(
            c_int::from(del_dec),
            p,
            nsq,
            idx.as_mut_ptr(),
            f.x16.as_ptr(),
            pulses.as_mut_ptr(),
            f.pred_coef_q12.as_ptr(),
            f.ltp_coef_q14.as_ptr(),
            f.ar_q13.as_ptr(),
            f.harm_shape_gain_q14.as_ptr(),
            f.tilt_q14.as_ptr(),
            f.lf_shp_q14.as_ptr(),
            f.gains_q16.as_ptr(),
            f.pitch_l.as_ptr(),
            f.lambda_q10,
            f.ltp_scale_q14,
        );
    }
    pulses
}

/// `silk_VAD_Init`.
pub fn vad_init(vad: &mut VadState) -> i32 {
    // SAFETY: `vad` is a valid, exclusively borrowed C-layout struct.
    unsafe { oracle_silk_VAD_Init(vad) }
}

/// `silk_VAD_GetSA_Q8_c`: returns `(ret, [speech_activity_Q8, input_tilt_Q15,
/// input_quality_bands_Q15[4]])`.
pub fn vad_get_sa_q8(
    vad: &mut VadState,
    frame_length: i32,
    fs_khz: i32,
    p_in: &[i16],
) -> (i32, [i32; 6]) {
    assert!(frame_length > 0 && frame_length as usize <= MAX_FRAME_LENGTH);
    assert!(frame_length % 8 == 0 && p_in.len() >= frame_length as usize);
    let mut out = [0; 6];
    // SAFETY: p_in holds frame_length samples, out has 6 ints.
    let ret = unsafe {
        oracle_silk_VAD_GetSA_Q8(vad, frame_length, fs_khz, p_in.as_ptr(), out.as_mut_ptr())
    };
    (ret, out)
}

/// `silk_VAD_GetNoiseLevels` (static in VAD.c).
pub fn vad_get_noise_levels(p_x: &[i32; 4], vad: &mut VadState) {
    // SAFETY: p_x has VAD_N_BANDS entries.
    unsafe { oracle_silk_VAD_GetNoiseLevels(p_x.as_ptr(), vad) }
}

/// Runs `silk_encode_indices` for each `ops[f] = (FrameIndex, encode_LBRR, condCoding)` on one
/// range coder of `size` bytes; before frame `f`, `psEncC->indices = sets[f][0]` and
/// `indices_LBRR = sets[f][1..]`. Returns `(buffer, result)`; `p` is updated.
pub fn encode_indices(
    p: &mut EncParams,
    sets: &[[SideInfoIndices; 1 + MAX_FRAMES_PER_PACKET]],
    ops: &[[i32; 3]],
    size: usize,
) -> (Vec<u8>, EncRes) {
    assert_eq!(sets.len(), ops.len());
    for op in ops {
        assert!((0..MAX_FRAMES_PER_PACKET as i32).contains(&op[0]));
    }
    let flat: Vec<c_int> = ops.iter().flatten().copied().collect();
    let mut buf = vec![0u8; size];
    let mut res = [0u32; 3];
    // SAFETY: sets has 4 entries per op, flat has 3 * ops.len() ints, the buffer has `size`
    // bytes and res 3 entries.
    unsafe {
        oracle_silk_encode_indices(
            p,
            sets.as_ptr().cast(),
            flat.as_ptr(),
            ops.len() as c_int,
            buf.as_mut_ptr(),
            size as c_int,
            res.as_mut_ptr(),
        );
    }
    (buf, res)
}

/// `silk_process_NLSFs` with `idx = {signalType, NLSFInterpCoef_Q2}`. Returns the NLSF
/// indices; `pred_coef_q12` and `p_nlsf_q15` are updated.
pub fn process_nlsfs(
    p: &EncParams,
    idx: [i32; 2],
    pred_coef_q12: &mut [[i16; MAX_LPC_ORDER]; 2],
    p_nlsf_q15: &mut [i16; MAX_LPC_ORDER],
    prev_nlsfq_q15: &[i16; MAX_LPC_ORDER],
) -> [i8; MAX_LPC_ORDER + 1] {
    assert!(p.predict_lpc_order == 10 || p.predict_lpc_order == 16);
    let mut ind = [0i8; MAX_LPC_ORDER + 1];
    // SAFETY: all arrays have their C sizes.
    unsafe {
        oracle_silk_process_NLSFs(
            p,
            idx.as_ptr(),
            ind.as_mut_ptr(),
            pred_coef_q12.as_mut_ptr().cast(),
            p_nlsf_q15.as_mut_ptr(),
            prev_nlsfq_q15.as_ptr(),
        );
    }
    ind
}

/// `silk_quant_LTP_gains`: returns `(B_Q14, cbk_index, periodicity_index, pred_gain_dB_Q7)`;
/// `sum_log_gain_q7` is updated.
pub fn quant_ltp_gains(
    sum_log_gain_q7: &mut i32,
    xx_q17: &[i32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER],
    x_x_q17: &[i32; MAX_NB_SUBFR * LTP_ORDER],
    subfr_len: i32,
    nb_subfr: i32,
) -> (
    [i16; MAX_NB_SUBFR * LTP_ORDER],
    [i8; MAX_NB_SUBFR],
    i32,
    i32,
) {
    assert!(nb_subfr == 2 || nb_subfr == 4);
    let mut b = [0i16; MAX_NB_SUBFR * LTP_ORDER];
    let mut cbk = [0i8; MAX_NB_SUBFR];
    let mut out = [0, *sum_log_gain_q7, 0];
    // SAFETY: all arrays have their C sizes.
    unsafe {
        oracle_silk_quant_LTP_gains(
            b.as_mut_ptr(),
            cbk.as_mut_ptr(),
            out.as_mut_ptr(),
            xx_q17.as_ptr(),
            x_x_q17.as_ptr(),
            subfr_len,
            nb_subfr,
        );
    }
    *sum_log_gain_q7 = out[1];
    (b, cbk, out[0], out[2])
}

/// `silk_VQ_WMat_EC_c` on codebook `cbk` (0..3). `gain_q7` is in/out. Returns
/// `(ind, res_nrg_Q15, rate_dist_Q8)`.
pub fn vq_wmat_ec(
    gain_q7: &mut i32,
    xx_q17: &[i32; LTP_ORDER * LTP_ORDER],
    x_x_q17: &[i32; LTP_ORDER],
    cbk: i32,
    subfr_len: i32,
    max_gain_q7: i32,
) -> (i32, i32, i32) {
    assert!((0..3).contains(&cbk));
    let mut out = [0, 0, 0, *gain_q7];
    // SAFETY: out has 4 ints, the matrices have their C sizes and cbk is a valid codebook.
    unsafe {
        oracle_silk_VQ_WMat_EC(
            out.as_mut_ptr(),
            xx_q17.as_ptr(),
            x_x_q17.as_ptr(),
            cbk,
            subfr_len,
            max_gain_q7,
        );
    }
    *gain_q7 = out[3];
    (out[0], out[1], out[2])
}

#[allow(clippy::too_many_arguments, reason = "mirrors the C signature")]
/// `silk_stereo_LR_to_MS`; `x1`/`x2` start two samples before the C pointers and hold
/// `frame_length + 2` samples. Returns `(ix, mid_only_flag, mid_side_rates_bps)`.
pub fn stereo_lr_to_ms(
    state: &mut StereoEncState,
    x1: &mut [i16],
    x2: &mut [i16],
    total_rate_bps: i32,
    prev_speech_act_q8: i32,
    to_mono: i32,
    fs_khz: i32,
    frame_length: i32,
) -> ([[i8; 3]; 2], i8, [i32; 2]) {
    assert!(frame_length > 0 && frame_length as usize <= MAX_FRAME_LENGTH);
    assert!(x1.len() >= frame_length as usize + 2 && x2.len() >= frame_length as usize + 2);
    assert!(frame_length >= 8 * fs_khz);
    let mut ix = [[0i8; 3]; 2];
    let mut mid_only = 0i8;
    let mut rates = [0i32; 2];
    // SAFETY: buffers checked above; ix has 6 bytes, rates 2 ints.
    unsafe {
        oracle_silk_stereo_LR_to_MS(
            state,
            x1.as_mut_ptr(),
            x2.as_mut_ptr(),
            ix.as_mut_ptr().cast(),
            &mut mid_only,
            rates.as_mut_ptr(),
            total_rate_bps,
            prev_speech_act_q8,
            to_mono,
            fs_khz,
            frame_length,
        );
    }
    (ix, mid_only, rates)
}

/// `silk_stereo_find_predictor`: returns `(pred_Q13, ratio_Q14)`.
pub fn stereo_find_predictor(
    x: &[i16],
    y: &[i16],
    mid_res_amp_q0: &mut [i32; 2],
    smooth_coef_q16: i32,
) -> (i32, i32) {
    assert_eq!(x.len(), y.len());
    let mut ratio = 0;
    // SAFETY: x and y hold `length` samples, mid_res_amp_q0 two ints.
    let pred = unsafe {
        oracle_silk_stereo_find_predictor(
            &mut ratio,
            x.as_ptr(),
            y.as_ptr(),
            mid_res_amp_q0.as_mut_ptr(),
            x.len() as c_int,
            smooth_coef_q16,
        )
    };
    (pred, ratio)
}

/// `silk_stereo_quant_pred`.
pub fn stereo_quant_pred(pred_q13: &mut [i32; 2], ix: &mut [[i8; 3]; 2]) {
    // SAFETY: pred has 2 ints, ix 6 bytes.
    unsafe { oracle_silk_stereo_quant_pred(pred_q13.as_mut_ptr(), ix.as_mut_ptr().cast()) }
}

/// `silk_HP_variable_cutoff` on `state_Fxx[ 0 ].sCmn` built from `p` (updated).
pub fn hp_variable_cutoff(p: &mut EncParams) {
    // SAFETY: p is a valid C-layout struct.
    unsafe { oracle_silk_HP_variable_cutoff(p) }
}

/// `silk_control_SNR` (`p` updated).
pub fn control_snr(p: &mut EncParams, target_rate_bps: i32) -> i32 {
    // SAFETY: p is a valid C-layout struct.
    unsafe { oracle_silk_control_SNR(p, target_rate_bps) }
}

/// `silk_control_audio_bandwidth` (`p` and `ctl` updated).
pub fn control_audio_bandwidth(p: &mut EncParams, ctl: &mut EncControl) -> i32 {
    // SAFETY: p and ctl are valid C-layout structs.
    unsafe { oracle_silk_control_audio_bandwidth(p, ctl.as_mut_ptr()) }
}

/// `check_control_input` (compiled without its `celt_assert( 0 )` so error paths return).
#[must_use]
pub fn check_control_input(ctl: &EncControl) -> i32 {
    // SAFETY: ctl is a valid C-layout struct.
    unsafe { oracle_check_control_input(ctl.as_ptr()) }
}
