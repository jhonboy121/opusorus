//! Differential tests for unit `silk_encoder_common`: NSQ / NSQ_del_dec, VAD, encode_indices,
//! process_NLSFs, quant_LTP_gains, VQ_WMat_EC, stereo LR->MS (+ find_predictor, quant_pred),
//! HP_variable_cutoff, control_SNR, control_audio_bandwidth and check_control_input vs the C
//! oracle (bit-exact, including the evolution of every state over many consecutive frames).

use opusorus::celt::entenc::EcEnc;
use opusorus::silk::coding::silk_decode_pitch;
use opusorus::silk::define::*;
use opusorus::silk::encoder_common::*;
use opusorus::silk::nlsf::{silk_a2nlsf, silk_nlsf2a};
use opusorus::silk::nsq::*;
use opusorus::silk::nsq_del_dec::silk_nsq_del_dec_c;
use opusorus::silk::sigproc::{silk_lin2log, silk_log2lin};
use opusorus::silk::structs::*;
use opusorus::silk::tables::*;
use opusorus::silk::vad::*;
use opusorus_conformance::{Rng, assert_slice_eq, signals};
use opusorus_oracle::silk_encoder_common as c;

// ---------------------------------------------------------------------------------------------
// Helpers: conversions between the port's structs and the oracle's C-layout mirrors
// ---------------------------------------------------------------------------------------------

const fn nsq_to_c(s: &SilkNsqState) -> c::NsqState {
    c::NsqState {
        xq: s.xq,
        s_ltp_shp_q14: s.s_ltp_shp_q14,
        s_lpc_q14: s.s_lpc_q14,
        s_ar2_q14: s.s_ar2_q14,
        s_lf_ar_shp_q14: s.s_lf_ar_shp_q14,
        s_diff_shp_q14: s.s_diff_shp_q14,
        lag_prev: s.lag_prev,
        s_ltp_buf_idx: s.s_ltp_buf_idx,
        s_ltp_shp_buf_idx: s.s_ltp_shp_buf_idx,
        rand_seed: s.rand_seed,
        prev_gain_q16: s.prev_gain_q16,
        rewhite_flag: s.rewhite_flag,
    }
}

#[track_caller]
fn assert_nsq_eq(what: &str, r: &SilkNsqState, c: &c::NsqState) {
    assert_slice_eq(&format!("{what}: xq"), &r.xq, &c.xq);
    assert_slice_eq(
        &format!("{what}: sLTP_shp_Q14"),
        &r.s_ltp_shp_q14,
        &c.s_ltp_shp_q14,
    );
    assert_slice_eq(&format!("{what}: sLPC_Q14"), &r.s_lpc_q14, &c.s_lpc_q14);
    assert_slice_eq(&format!("{what}: sAR2_Q14"), &r.s_ar2_q14, &c.s_ar2_q14);
    assert_eq!(
        [
            r.s_lf_ar_shp_q14,
            r.s_diff_shp_q14,
            r.lag_prev,
            r.s_ltp_buf_idx,
            r.s_ltp_shp_buf_idx,
            r.rand_seed,
            r.prev_gain_q16,
            r.rewhite_flag
        ],
        [
            c.s_lf_ar_shp_q14,
            c.s_diff_shp_q14,
            c.lag_prev,
            c.s_ltp_buf_idx,
            c.s_ltp_shp_buf_idx,
            c.rand_seed,
            c.prev_gain_q16,
            c.rewhite_flag
        ],
        "{what}: NSQ scalars"
    );
}

const fn vad_to_c(s: &SilkVadState) -> c::VadState {
    c::VadState {
        ana_state: s.ana_state,
        ana_state1: s.ana_state1,
        ana_state2: s.ana_state2,
        xnrg_subfr: s.xnrg_subfr,
        nrg_ratio_smth_q8: s.nrg_ratio_smth_q8,
        hp_state: s.hp_state,
        nl: s.nl,
        inv_nl: s.inv_nl,
        noise_level_bias: s.noise_level_bias,
        counter: s.counter,
    }
}

const fn stereo_to_c(s: &StereoEncState) -> c::StereoEncState {
    c::StereoEncState {
        pred_prev_q13: s.pred_prev_q13,
        s_mid: s.s_mid,
        s_side: s.s_side,
        mid_side_amp_q0: s.mid_side_amp_q0,
        smth_width_q14: s.smth_width_q14,
        width_prev_q14: s.width_prev_q14,
        silent_side_len: s.silent_side_len,
        pred_ix: s.pred_ix,
        mid_only_flags: s.mid_only_flags,
    }
}

const fn indices_to_c(s: &SideInfoIndices) -> c::SideInfoIndices {
    c::SideInfoIndices {
        gains_indices: s.gains_indices,
        ltp_index: s.ltp_index,
        nlsf_indices: s.nlsf_indices,
        lag_index: s.lag_index,
        contour_index: s.contour_index,
        signal_type: s.signal_type,
        quant_offset_type: s.quant_offset_type,
        nlsf_interp_coef_q2: s.nlsf_interp_coef_q2,
        per_index: s.per_index,
        ltp_scale_index: s.ltp_scale_index,
        seed: s.seed,
    }
}

const fn ctl_to_c(s: &SilkEncControlStruct) -> c::EncControl {
    [
        s.n_channels_api,
        s.n_channels_internal,
        s.api_sample_rate,
        s.max_internal_sample_rate,
        s.min_internal_sample_rate,
        s.desired_internal_sample_rate,
        s.payload_size_ms,
        s.bit_rate,
        s.packet_loss_percentage,
        s.complexity,
        s.use_in_band_fec,
        s.use_dred,
        s.lbrr_coded,
        s.use_dtx,
        s.use_cbr,
        s.max_bits,
        s.to_mono,
        s.opus_can_switch,
        s.reduced_dependency,
        s.internal_sample_rate,
        s.allow_bandwidth_switch,
        s.in_wb_mode_without_variable_lp,
        s.stereo_width_q14,
        s.switch_ready,
        s.signal_type,
        s.offset,
    ]
}

/// Encoder state with the table pointers `silk_setup_fs` would set for `fs_khz` / `nb_subfr`.
fn enc_state(fs_khz: i32, nb_subfr: i32) -> SilkEncoderState {
    let mut e = SilkEncoderState::new();
    e.fs_khz = fs_khz;
    e.nb_subfr = nb_subfr;
    e.subfr_length = SUB_FRAME_LENGTH_MS * fs_khz;
    e.frame_length = e.subfr_length * nb_subfr;
    e.ltp_mem_length = LTP_MEM_LENGTH_MS * fs_khz;
    e.predict_lpc_order = if fs_khz == 16 { 16 } else { 10 };
    e.ps_nlsf_cb = if fs_khz == 16 {
        &SILK_NLSF_CB_WB
    } else {
        &SILK_NLSF_CB_NB_MB
    };
    e.pitch_lag_low_bits_icdf = match fs_khz {
        16 => &SILK_UNIFORM8_ICDF,
        12 => &SILK_UNIFORM6_ICDF,
        _ => &SILK_UNIFORM4_ICDF,
    };
    e.pitch_contour_icdf = match (fs_khz == 8, nb_subfr == 4) {
        (true, true) => &SILK_PITCH_CONTOUR_NB_ICDF,
        (true, false) => &SILK_PITCH_CONTOUR_10_MS_NB_ICDF,
        (false, true) => &SILK_PITCH_CONTOUR_ICDF,
        (false, false) => &SILK_PITCH_CONTOUR_10_MS_ICDF,
    };
    e
}

/// The oracle's flat view of the encoder-state fields used by this unit.
const fn enc_params(e: &SilkEncoderState) -> c::EncParams {
    c::EncParams {
        fs_khz: e.fs_khz,
        nb_subfr: e.nb_subfr,
        frame_length: e.frame_length,
        subfr_length: e.subfr_length,
        ltp_mem_length: e.ltp_mem_length,
        predict_lpc_order: e.predict_lpc_order,
        shaping_lpc_order: e.shaping_lpc_order,
        n_states_delayed_decision: e.n_states_delayed_decision,
        warping_q16: e.warping_q16,
        speech_activity_q8: e.speech_activity_q8,
        use_interpolated_nlsfs: e.use_interpolated_nlsfs,
        nlsf_msvq_survivors: e.nlsf_msvq_survivors,
        ec_prev_signal_type: e.ec_prev_signal_type,
        ec_prev_lag_index: e.ec_prev_lag_index as i32,
        prev_signal_type: e.prev_signal_type as i32,
        prev_lag: e.prev_lag,
        variable_hp_smth1_q15: e.variable_hp_smth1_q15,
        input_quality_bands_q15: e.input_quality_bands_q15,
        input_tilt_q15: e.input_tilt_q15,
        target_rate_bps: e.target_rate_bps,
        snr_db_q7: e.snr_db_q7,
        api_fs_hz: e.api_fs_hz,
        max_internal_fs_hz: e.max_internal_fs_hz,
        min_internal_fs_hz: e.min_internal_fs_hz,
        desired_internal_fs_hz: e.desired_internal_fs_hz,
        allow_bandwidth_switch: e.allow_bandwidth_switch,
        s_lp_in_lp_state: e.s_lp.in_lp_state,
        s_lp_transition_frame_no: e.s_lp.transition_frame_no,
        s_lp_mode: e.s_lp.mode,
        s_lp_saved_fs_khz: e.s_lp.saved_fs_khz,
    }
}

/// Random sorted, well-separated NLSF vector in Q15 (a stable LPC filter).
fn rand_nlsf(rng: &mut Rng, d: usize) -> [i16; 16] {
    let mut out = [0i16; 16];
    let step = 32768 / (d as i32 + 1);
    let jitter = step / 3;
    for (i, v) in out.iter_mut().take(d).enumerate() {
        *v = (step * (i as i32 + 1) + rng.range_i32(-jitter, jitter)) as i16;
    }
    out
}

/// Stable prediction coefficients in Q12 from a random NLSF vector.
fn rand_a_q12(rng: &mut Rng, d: usize) -> [i16; 16] {
    let nlsf = rand_nlsf(rng, d);
    let mut a = [0i16; 16];
    silk_nlsf2a(&mut a, &nlsf, d);
    a
}

/// Noise-shaping AR coefficients in Q13 from random reflection coefficients (step-up
/// recursion in f64), so the shaping filter is stable. The prediction power gain
/// `1 / prod( 1 - k^2 )` is limited to `MAX_PREDICTION_POWER_GAIN` (1e4), like the encoder's
/// Burg analysis does.
fn rand_ar_q13(rng: &mut Rng, order: usize, kmax: f64) -> [i16; 24] {
    let mut ks: Vec<f64> = (0..order).map(|_| kmax * rng.f32_sym() as f64).collect();
    while ks.iter().map(|k| 1.0 - k * k).product::<f64>() < 1e-4 {
        ks.iter_mut().for_each(|k| *k *= 0.95);
    }
    let mut a = [0f64; 24];
    for m in 0..order {
        let k = ks[m];
        let prev = a;
        for i in 0..m {
            a[i] = prev[i] - k * prev[m - 1 - i];
        }
        a[m] = k;
    }
    let mut out = [0i16; 24];
    for i in 0..order {
        out[i] = (a[i] * 8192.0).round().clamp(-32768.0, 32767.0) as i16;
    }
    out
}

/// A random but realistic signal chunk in Q0.
fn rand_signal(rng: &mut Rng, n: usize, fs_khz: i32) -> Vec<i16> {
    let amp = [0.0005f32, 0.01, 0.1, 0.4, 0.9][rng.range_i32(0, 4) as usize];
    let seed = rng.next_u64();
    let x = match rng.range_i32(0, 3) {
        0 => signals::speech_like(n, 1, fs_khz as u32 * 1000, seed),
        1 => signals::music_like(n, 1, fs_khz as u32 * 1000, seed),
        2 => signals::noise(n, 1, amp, seed),
        _ => vec![0.0; n],
    };
    let gain = if rng.range_i32(0, 1) == 0 {
        1.0
    } else {
        amp * 4.0
    };
    signals::to_i16(&x.iter().map(|v| v * gain).collect::<Vec<_>>())
}

// ---------------------------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------------------------

#[test]
fn struct_layouts_match_c() {
    assert_eq!(c::c_sizes(), c::rust_sizes());
}

// ---------------------------------------------------------------------------------------------
// NSQ.h inlines
// ---------------------------------------------------------------------------------------------

#[test]
fn nsq_short_prediction_and_feedback_loop() {
    let mut rng = Rng::new(0x5EC0_0001);
    for it in 0..20000 {
        let order = if it % 2 == 0 { 10 } else { 16 };
        let len = order as usize + rng.range_i32(0, 20) as usize;
        let big = it % 3 == 0;
        let buf: Vec<i32> = (0..len)
            .map(|_| {
                if big {
                    rng.next_u32() as i32
                } else {
                    rng.range_i32(-(1 << 22), 1 << 22)
                }
            })
            .collect();
        let coef: Vec<i16> = (0..16).map(|_| rng.i16()).collect();
        assert_eq!(
            silk_noise_shape_quantizer_short_prediction_c(&buf, &coef, order),
            c::nsq_short_prediction(&buf, &coef, order),
            "short prediction it {it}"
        );

        let order = 2 * rng.range_i32(1, 12);
        let data0 = if big {
            rng.next_u32() as i32
        } else {
            rng.range_i32(-(1 << 22), 1 << 22)
        };
        let mut d_r: Vec<i32> = (0..24)
            .map(|_| {
                if big {
                    rng.next_u32() as i32
                } else {
                    rng.range_i32(-(1 << 22), 1 << 22)
                }
            })
            .collect();
        let mut d_c = d_r.clone();
        let coef: Vec<i16> = (0..24).map(|_| rng.i16()).collect();
        assert_eq!(
            silk_nsq_noise_shape_feedback_loop_c(data0, &mut d_r, &coef, order),
            c::nsq_feedback_loop(data0, &mut d_c, &coef, order),
            "feedback loop it {it}"
        );
        assert_slice_eq("feedback loop state", &d_r, &d_c);
    }
}

// ---------------------------------------------------------------------------------------------
// NSQ / NSQ_del_dec over many consecutive frames
// ---------------------------------------------------------------------------------------------

/// Per-frame NSQ inputs.
struct NsqInputs {
    x16: Vec<i16>,
    pred_coef_q12: [i16; 32],
    ltp_coef_q14: [i16; 20],
    ar_q13: [i16; 96],
    harm_shape_gain_q14: [i32; 4],
    tilt_q14: [i32; 4],
    lf_shp_q14: [i32; 4],
    gains_q16: [i32; 4],
    pitch_l: [i32; 4],
    lambda_q10: i32,
    ltp_scale_q14: i32,
    indices: SideInfoIndices,
}

fn rand_nsq_inputs(rng: &mut Rng, p: &NsqEncParams, fs_khz: i32, x16: Vec<i16>) -> NsqInputs {
    let nb = p.nb_subfr as usize;
    let d = p.predict_lpc_order as usize;
    let mut indices = SideInfoIndices::new();
    indices.signal_type = rng.range_i32(0, 2) as i8;
    indices.quant_offset_type = rng.range_i32(0, 1) as i8;
    indices.nlsf_interp_coef_q2 = if nb == 2 || rng.range_i32(0, 2) == 0 {
        4
    } else {
        rng.range_i32(0, 3) as i8
    };
    indices.seed = rng.range_i32(0, 3) as i8;
    let voiced = indices.signal_type as i32 == TYPE_VOICED;

    let mut pred_coef_q12 = [0i16; 32];
    let a0 = rand_a_q12(rng, d);
    let a1 = rand_a_q12(rng, d);
    pred_coef_q12[..16].copy_from_slice(&a0);
    pred_coef_q12[16..].copy_from_slice(&a1);

    let per = rng.range_i32(0, 2) as usize;
    let mut ltp_coef_q14 = [0i16; 20];
    for k in 0..nb {
        let row = SILK_LTP_VQ_PTRS_Q7[per][rng.range_i32(0, (8 << per) - 1) as usize];
        for j in 0..5 {
            ltp_coef_q14[k * 5 + j] = if voiced { (row[j] as i16) << 7 } else { 0 };
        }
    }

    let mut ar_q13 = [0i16; 96];
    let kmax = [0.3, 0.6, 0.85][rng.range_i32(0, 2) as usize];
    for k in 0..nb {
        let ar = rand_ar_q13(rng, p.shaping_lpc_order as usize, kmax);
        ar_q13[k * 24..(k + 1) * 24].copy_from_slice(&ar);
    }

    let mut harm_shape_gain_q14 = [0i32; 4];
    let mut tilt_q14 = [0i32; 4];
    let mut lf_shp_q14 = [0i32; 4];
    let mut gains_q16 = [0i32; 4];
    // Quantization step sizes follow the subframe level (as the encoder's gains do), with a
    // random SNR offset; clamped to the dequantizer's range.
    let snr = [0.1f64, 0.4, 1.0, 3.0][rng.range_i32(0, 3) as usize];
    let subfr = p.subfr_length as usize;
    for k in 0..nb {
        harm_shape_gain_q14[k] = if voiced || rng.range_i32(0, 3) == 0 {
            rng.range_i32(0, 8192)
        } else {
            0
        };
        tilt_q14[k] = rng.range_i32(-8000, 0);
        let lf_ar = rng.range_i32(0, 13000);
        let lf_ma = rng.range_i32(-6500, 0);
        lf_shp_q14[k] = (lf_ar << 16) | (lf_ma as u16 as i32);
        let x = &x16[k * subfr..(k + 1) * subfr];
        let rms = (x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / subfr as f64).sqrt();
        let g = rms * snr * (1.0 + 0.2 * rng.f32_sym() as f64) * 65536.0;
        gains_q16[k] = g.clamp(silk_log2lin(2090) as f64, silk_log2lin(3967) as f64 / 2.0) as i32;
    }

    let mut pitch_l = [0i32; 4];
    if voiced || rng.range_i32(0, 4) == 0 {
        let lag_index = rng.range_i32(0, 16 * fs_khz - 1) as i16;
        let n_contours = match (fs_khz == 8, nb == 4) {
            (true, true) => 11,
            (false, true) => 34,
            (true, false) => 3,
            (false, false) => 12,
        };
        let contour = rng.range_i32(0, n_contours - 1) as i8;
        silk_decode_pitch(lag_index, contour, &mut pitch_l, fs_khz, nb as i32);
    }

    let lambda_q10 = if rng.range_i32(0, 3) == 0 {
        rng.range_i32(2049, 4000)
    } else {
        rng.range_i32(200, 2048)
    };
    let ltp_scale_q14 = SILK_LTPSCALES_TABLE_Q14[rng.range_i32(0, 2) as usize] as i32;

    NsqInputs {
        x16,
        pred_coef_q12,
        ltp_coef_q14,
        ar_q13,
        harm_shape_gain_q14,
        tilt_q14,
        lf_shp_q14,
        gains_q16,
        pitch_l,
        lambda_q10,
        ltp_scale_q14,
        indices,
    }
}

/// Runs `frames` consecutive frames of NSQ (`n_states == 0`: `silk_NSQ_c`) or NSQ_del_dec on
/// both sides and compares pulses, seed and the whole NSQ state after every frame.
/// `n_states < 0` picks the quantizer, the number of states and warping at random per frame
/// (a complexity change mid-stream).
fn run_nsq_config(
    seed: u64,
    fs_khz: i32,
    nb_subfr: i32,
    n_states: i32,
    warping: bool,
    shaping_order: i32,
    frames: usize,
) {
    let mut rng = Rng::new(seed);
    let enc = enc_state(fs_khz, nb_subfr);
    let mut p = NsqEncParams {
        ltp_mem_length: enc.ltp_mem_length,
        frame_length: enc.frame_length,
        subfr_length: enc.subfr_length,
        nb_subfr,
        predict_lpc_order: enc.predict_lpc_order,
        shaping_lpc_order: shaping_order,
        n_states_delayed_decision: n_states.max(1),
        warping_q16: if warping { fs_khz * 983 } else { 0 },
    };
    let mut cp = enc_params(&enc);
    cp.shaping_lpc_order = p.shaping_lpc_order;
    cp.n_states_delayed_decision = p.n_states_delayed_decision;
    cp.warping_q16 = p.warping_q16;

    // As set by silk_setup_fs() on a sampling rate change.
    let mut nsq_r = SilkNsqState::new();
    nsq_r.lag_prev = 100;
    nsq_r.prev_gain_q16 = 65536;
    let mut nsq_c = nsq_to_c(&nsq_r);

    let fl = p.frame_length as usize;
    let signal = rand_signal(&mut rng, fl * frames, fs_khz);
    let what =
        format!("fs {fs_khz} nb {nb_subfr} states {n_states} warp {warping} shp {shaping_order}");
    let mut n_states = n_states;
    let mixed = n_states < 0;
    for f in 0..frames {
        if mixed {
            n_states = rng.range_i32(0, 4);
            p.n_states_delayed_decision = n_states.max(1);
            p.warping_q16 = if rng.range_i32(0, 1) == 0 {
                0
            } else {
                fs_khz * 983
            };
            cp.n_states_delayed_decision = p.n_states_delayed_decision;
            cp.warping_q16 = p.warping_q16;
        }
        let x16 = signal[f * fl..(f + 1) * fl].to_vec();
        let inp = rand_nsq_inputs(&mut rng, &p, fs_khz, x16);

        let mut ind_r = inp.indices;
        let mut pulses_r = vec![0i8; fl];
        if n_states == 0 {
            silk_nsq_c(
                &p,
                &mut nsq_r,
                &ind_r,
                &inp.x16,
                &mut pulses_r,
                &inp.pred_coef_q12,
                &inp.ltp_coef_q14,
                &inp.ar_q13,
                &inp.harm_shape_gain_q14,
                &inp.tilt_q14,
                &inp.lf_shp_q14,
                &inp.gains_q16,
                &inp.pitch_l,
                inp.lambda_q10,
                inp.ltp_scale_q14,
            );
        } else {
            silk_nsq_del_dec_c(
                &p,
                &mut nsq_r,
                &mut ind_r,
                &inp.x16,
                &mut pulses_r,
                &inp.pred_coef_q12,
                &inp.ltp_coef_q14,
                &inp.ar_q13,
                &inp.harm_shape_gain_q14,
                &inp.tilt_q14,
                &inp.lf_shp_q14,
                &inp.gains_q16,
                &inp.pitch_l,
                inp.lambda_q10,
                inp.ltp_scale_q14,
            );
        }

        let mut idx_c = [
            inp.indices.signal_type as i32,
            inp.indices.quant_offset_type as i32,
            inp.indices.nlsf_interp_coef_q2 as i32,
            inp.indices.seed as i32,
        ];
        let frame = c::NsqFrame {
            x16: &inp.x16,
            pred_coef_q12: &inp.pred_coef_q12,
            ltp_coef_q14: &inp.ltp_coef_q14,
            ar_q13: &inp.ar_q13,
            harm_shape_gain_q14: &inp.harm_shape_gain_q14,
            tilt_q14: &inp.tilt_q14,
            lf_shp_q14: &inp.lf_shp_q14,
            gains_q16: &inp.gains_q16,
            pitch_l: &inp.pitch_l,
            lambda_q10: inp.lambda_q10,
            ltp_scale_q14: inp.ltp_scale_q14,
        };
        let pulses_c = c::nsq(n_states > 0, &cp, &mut nsq_c, &mut idx_c, &frame);

        let what = format!("{what} frame {f}");
        assert_slice_eq(&format!("{what}: pulses"), &pulses_r, &pulses_c);
        assert_eq!(ind_r.seed as i32, idx_c[3], "{what}: seed");
        assert_nsq_eq(&what, &nsq_r, &nsq_c);
    }
}

#[test]
fn nsq_frames() {
    let mut seed = 0x5EC0_1000u64;
    for fs_khz in [8, 12, 16] {
        for nb_subfr in [2, 4] {
            for shaping_order in [8, 12, 16, 20, 24] {
                seed += 1;
                run_nsq_config(seed, fs_khz, nb_subfr, 0, false, shaping_order, 120);
            }
        }
    }
}

#[test]
fn nsq_del_dec_frames() {
    let mut seed = 0x5EC0_2000u64;
    for fs_khz in [8, 12, 16] {
        for nb_subfr in [2, 4] {
            for n_states in 1..=4 {
                for warping in [false, true] {
                    seed += 1;
                    let shaping_order = [12, 16, 20, 24][(seed % 4) as usize];
                    run_nsq_config(seed, fs_khz, nb_subfr, n_states, warping, shaping_order, 80);
                }
            }
        }
    }
}

#[test]
fn nsq_mixed_quantizers_frames() {
    let mut seed = 0x5EC0_2800u64;
    for fs_khz in [8, 12, 16] {
        for nb_subfr in [2, 4] {
            for shaping_order in [10, 16, 24] {
                seed += 1;
                run_nsq_config(seed, fs_khz, nb_subfr, -1, false, shaping_order, 150);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// VAD
// ---------------------------------------------------------------------------------------------

#[test]
fn vad_init_matches() {
    let mut r = SilkVadState {
        counter: 77,
        nl: [5; 4],
        ..SilkVadState::default()
    };
    let mut cs = vad_to_c(&r);
    assert_eq!(silk_vad_init(&mut r), c::vad_init(&mut cs));
    assert_eq!(vad_to_c(&r), cs);
}

#[test]
fn vad_get_sa_q8_frames() {
    let mut rng = Rng::new(0x5EC0_3000);
    for (cfg, (fs_khz, ms)) in [(8, 10), (8, 20), (12, 10), (12, 20), (16, 10), (16, 20)]
        .into_iter()
        .enumerate()
    {
        for run in 0..12 {
            let mut enc = enc_state(fs_khz, if ms == 10 { 2 } else { 4 });
            let fl = enc.frame_length as usize;
            silk_vad_init(&mut enc.s_vad);
            let mut vad_c = vad_to_c(&enc.s_vad);
            let frames = 300;
            // Alternate signal kinds so the noise estimator sees onsets and silence.
            let mut sig = Vec::with_capacity(fl * frames);
            while sig.len() < fl * frames {
                let chunk = fl * rng.range_i32(3, 20) as usize;
                sig.extend(rand_signal(&mut rng, chunk, fs_khz));
            }
            for f in 0..frames {
                let x = &sig[f * fl..(f + 1) * fl];
                let ret_r = if f % 2 == 0 {
                    silk_vad_get_sa_q8_c(&mut enc, x)
                } else {
                    enc.input_buf[1..=fl].copy_from_slice(x);
                    silk_vad_get_sa_q8_input_buf(&mut enc)
                };
                let (ret_c, out_c) = c::vad_get_sa_q8(&mut vad_c, fl as i32, fs_khz, x);
                let what = format!("cfg {cfg} run {run} frame {f}");
                assert_eq!(ret_r, ret_c, "{what}: ret");
                let q = enc.input_quality_bands_q15;
                assert_eq!(
                    [
                        enc.speech_activity_q8,
                        enc.input_tilt_q15,
                        q[0],
                        q[1],
                        q[2],
                        q[3]
                    ],
                    out_c,
                    "{what}: outputs"
                );
                assert_eq!(vad_to_c(&enc.s_vad), vad_c, "{what}: VAD state");
            }
        }
    }
}

#[test]
fn vad_get_noise_levels_random() {
    let mut rng = Rng::new(0x5EC0_3100);
    for it in 0..3000 {
        let mut r = SilkVadState::default();
        silk_vad_init(&mut r);
        r.counter = [0, 15, 500, 999, 1000, 5000][rng.range_i32(0, 5) as usize];
        let mut cs = vad_to_c(&r);
        for step in 0..20 {
            let px: [i32; 4] = core::array::from_fn(|_| match rng.range_i32(0, 3) {
                0 => rng.range_i32(0, 1000),
                1 => rng.range_i32(0, 1 << 20),
                2 => rng.range_i32(0, i32::MAX),
                _ => i32::MAX,
            });
            silk_vad_get_noise_levels(&px, &mut r);
            c::vad_get_noise_levels(&px, &mut cs);
            assert_eq!(vad_to_c(&r), cs, "it {it} step {step}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// encode_indices
// ---------------------------------------------------------------------------------------------

fn rand_indices(rng: &mut Rng, fs_khz: i32, nb_subfr: i32, lbrr: bool) -> SideInfoIndices {
    let mut ind = SideInfoIndices::new();
    ind.signal_type = if lbrr {
        rng.range_i32(1, 2)
    } else {
        rng.range_i32(0, 2)
    } as i8;
    ind.quant_offset_type = rng.range_i32(0, 1) as i8;
    for k in 0..4 {
        ind.gains_indices[k] = rng.range_i32(0, 40) as i8;
        ind.ltp_index[k] = 0;
    }
    let cb = if fs_khz == 16 {
        &SILK_NLSF_CB_WB
    } else {
        &SILK_NLSF_CB_NB_MB
    };
    ind.nlsf_indices[0] = rng.range_i32(0, cb.n_vectors as i32 - 1) as i8;
    for i in 1..=cb.order as usize {
        ind.nlsf_indices[i] = match rng.range_i32(0, 5) {
            0 => rng.range_i32(-10, 10),
            _ => rng.range_i32(-3, 3),
        } as i8;
    }
    ind.nlsf_interp_coef_q2 = rng.range_i32(0, 4) as i8;
    ind.lag_index = rng.range_i32(0, 16 * fs_khz - 1) as i16;
    let n_contours = match (fs_khz == 8, nb_subfr == 4) {
        (true, true) => 11,
        (false, true) => 34,
        (true, false) => 3,
        (false, false) => 12,
    };
    ind.contour_index = rng.range_i32(0, n_contours - 1) as i8;
    ind.per_index = rng.range_i32(0, 2) as i8;
    for k in 0..4 {
        ind.ltp_index[k] = rng.range_i32(0, (8 << ind.per_index) - 1) as i8;
    }
    ind.ltp_scale_index = rng.range_i32(0, 2) as i8;
    ind.seed = rng.range_i32(0, 3) as i8;
    ind
}

#[test]
fn encode_indices_sequences() {
    let mut rng = Rng::new(0x5EC0_4000);
    for it in 0..4000 {
        let fs_khz = [8, 12, 16][rng.range_i32(0, 2) as usize];
        let nb_subfr = if rng.range_i32(0, 1) == 0 { 2 } else { 4 };
        let mut enc = enc_state(fs_khz, nb_subfr);
        enc.ec_prev_signal_type = rng.range_i32(0, 2);
        enc.ec_prev_lag_index = rng.range_i32(0, 16 * fs_khz - 1) as i16;
        let mut p = enc_params(&enc);

        let n = rng.range_i32(1, 8) as usize;
        let mut ops = Vec::with_capacity(n);
        let mut sets_c = Vec::with_capacity(n);
        let mut buf = vec![0u8; 1275];
        let res_r = {
            let mut e = EcEnc::new(&mut buf);
            let mut prev_lag = enc.ec_prev_lag_index as i32;
            for _ in 0..n {
                let frame_index = rng.range_i32(0, 2);
                let lbrr = rng.range_i32(0, 2) == 0;
                let cond = rng.range_i32(0, 2);
                let mut set = [SideInfoIndices::new(); 4];
                for (k, ind) in set.iter_mut().enumerate() {
                    *ind = rand_indices(&mut rng, fs_khz, nb_subfr, k > 0);
                    if cond != CODE_INDEPENDENTLY {
                        ind.ltp_scale_index = 0;
                    }
                    if cond != CODE_CONDITIONALLY {
                        // independent gain coding: 6-bit absolute first index
                        ind.gains_indices[0] = rng.range_i32(0, N_LEVELS_QGAIN - 1) as i8;
                    }
                    // Make delta lag coding likely (and exercise both limits).
                    if rng.range_i32(0, 1) == 0 {
                        ind.lag_index =
                            (prev_lag + rng.range_i32(-10, 13)).clamp(0, 16 * fs_khz - 1) as i16;
                    }
                }
                enc.indices = set[0];
                enc.indices_lbrr.copy_from_slice(&set[1..]);
                silk_encode_indices(
                    &mut enc,
                    &mut e,
                    frame_index as usize,
                    i32::from(lbrr),
                    cond,
                );
                prev_lag = enc.ec_prev_lag_index as i32;
                ops.push([frame_index, i32::from(lbrr), cond]);
                sets_c.push(set.map(|s| indices_to_c(&s)));
            }
            let t = e.tell_frac();
            e.done();
            [t, e.rng, e.error as u32]
        };
        let (buf_c, res_c) = c::encode_indices(&mut p, &sets_c, &ops, 1275);
        let what = format!("it {it}");
        assert_eq!(res_r, res_c, "{what}: coder result");
        assert_slice_eq(&format!("{what}: bytes"), &buf, &buf_c);
        assert_eq!(enc_params(&enc), p, "{what}: state");
    }
}

// ---------------------------------------------------------------------------------------------
// process_NLSFs
// ---------------------------------------------------------------------------------------------

#[test]
fn process_nlsfs_random() {
    let mut rng = Rng::new(0x5EC0_5000);
    for it in 0..10000 {
        let fs_khz = [8, 12, 16][rng.range_i32(0, 2) as usize];
        let nb_subfr = if rng.range_i32(0, 1) == 0 { 2 } else { 4 };
        let mut enc = enc_state(fs_khz, nb_subfr);
        let d = enc.predict_lpc_order as usize;
        enc.speech_activity_q8 = rng.range_i32(0, 256);
        enc.use_interpolated_nlsfs = rng.range_i32(0, 1);
        enc.nlsf_msvq_survivors = [2, 3, 4, 6, 8, 16][rng.range_i32(0, 5) as usize];
        enc.indices.signal_type = rng.range_i32(0, 2) as i8;
        enc.indices.nlsf_interp_coef_q2 = if enc.use_interpolated_nlsfs == 1 {
            rng.range_i32(0, 4) as i8
        } else {
            4
        };
        let p = enc_params(&enc);

        // NLSFs as the encoder produces them: silk_A2NLSF of a stable LPC filter (including
        // sharp resonances), or evenly spread ones.
        let mut nlsf_r = if rng.range_i32(0, 3) == 0 {
            rand_nlsf(&mut rng, d)
        } else {
            // (kmax 0.99 can produce NLSF gaps of a few Q15 units, for which silk_MLA in
            // silk_NLSF_del_dec_quant overflows: C wraps, the Rust port panics in debug builds.)
            let kmax = [0.5, 0.8, 0.9][rng.range_i32(0, 2) as usize];
            let a = rand_ar_q13(&mut rng, d, kmax);
            let mut a_q16: [i32; 16] = core::array::from_fn(|i| (a[i] as i32) << 3);
            let mut v = [0i16; 16];
            silk_a2nlsf(&mut v, &mut a_q16, d);
            v
        };
        let prev = rand_nlsf(&mut rng, d);
        let init: [i16; 16] = core::array::from_fn(|_| rng.i16());
        let mut pc_r = [init, init];
        let mut nlsf_c = nlsf_r;
        let mut pc_c = pc_r;

        silk_process_nlsfs(&mut enc, &mut pc_r, &mut nlsf_r[..d], &prev[..d]);
        let ind_c = c::process_nlsfs(
            &p,
            [
                enc.indices.signal_type as i32,
                enc.indices.nlsf_interp_coef_q2 as i32,
            ],
            &mut pc_c,
            &mut nlsf_c,
            &prev,
        );
        let what = format!("it {it}");
        assert_slice_eq(&format!("{what}: NLSF"), &nlsf_r, &nlsf_c);
        assert_slice_eq(&format!("{what}: PredCoef[0]"), &pc_r[0], &pc_c[0]);
        assert_slice_eq(&format!("{what}: PredCoef[1]"), &pc_r[1], &pc_c[1]);
        assert_slice_eq(
            &format!("{what}: indices"),
            &enc.indices.nlsf_indices,
            &ind_c,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// quant_LTP_gains / VQ_WMat_EC
// ---------------------------------------------------------------------------------------------

/// LTP correlation matrix / vector in Q17 for one subframe, built like `silk_find_LTP_FLP`
/// (correlations of a lagged residual, normalized by
/// `max( xx, LTP_CORR_INV_MAX * 0.5 * ( XX[ 0 ] + XX[ 24 ] ) + 1 )`) and quantized like
/// `silk_quant_LTP_gains_FLP`. `quiet` scales the target down relative to its past, which
/// drives the normalization to its bound.
fn rand_ltp_corr(rng: &mut Rng, subfr_len: usize, quiet: f32) -> ([i32; 25], [i32; 5]) {
    let n = subfr_len + 4;
    let raw: Vec<f32> = (0..n).map(|_| 3000.0 * rng.f32_sym()).collect();
    // Smooth the lagged signal a bit to create correlations between taps.
    let x: Vec<f32> = (0..n)
        .map(|i| raw[i] + 0.7 * if i > 0 { raw[i - 1] } else { 0.0 })
        .collect();
    let b: [f32; 5] = core::array::from_fn(|_| 0.4 * rng.f32_sym());
    // Target residual r: prediction from the lagged signal plus noise (+ LTP_ORDER look-ahead
    // for the energy, as in C).
    let r: Vec<f32> = (0..subfr_len + LTP_ORDER as usize)
        .map(|t| {
            let mut v = 500.0 * rng.f32_sym();
            if t < subfr_len {
                for (j, bj) in b.iter().enumerate() {
                    v += bj * x[t + 4 - j];
                }
            }
            v * quiet
        })
        .collect();
    let mut xx = [0f32; 25];
    let mut x_x = [0f32; 5];
    for i in 0..5 {
        for j in 0..5 {
            xx[i * 5 + j] = (0..subfr_len).map(|t| x[t + 4 - i] * x[t + 4 - j]).sum();
        }
        x_x[i] = (0..subfr_len).map(|t| r[t] * x[t + 4 - i]).sum();
    }
    let energy: f32 = r.iter().map(|v| v * v).sum();
    let temp = 1.0 / energy.max(0.03 * 0.5 * (xx[0] + xx[24]) + 1.0);
    let q = |v: f32| (v * temp * 131072.0).round() as i32;
    (xx.map(q), x_x.map(q))
}

const fn rand_quiet(rng: &mut Rng) -> f32 {
    [1.0, 0.3, 0.05, 0.001, 0.0][rng.range_i32(0, 4) as usize]
}

#[test]
fn quant_ltp_gains_random() {
    let mut rng = Rng::new(0x5EC0_6000);
    for it in 0..10000 {
        let fs_khz = [8, 12, 16][rng.range_i32(0, 2) as usize];
        let nb_subfr = if rng.range_i32(0, 1) == 0 { 2 } else { 4 };
        let subfr_len = 5 * fs_khz;
        let mut xx = [0i32; 100];
        let mut x_x = [0i32; 20];
        for k in 0..nb_subfr as usize {
            let quiet = rand_quiet(&mut rng);
            let (m, v) = rand_ltp_corr(&mut rng, subfr_len as usize, quiet);
            xx[k * 25..(k + 1) * 25].copy_from_slice(&m);
            x_x[k * 5..(k + 1) * 5].copy_from_slice(&v);
        }
        let sum_log0 = rng.range_i32(0, 6000);

        let mut b_r = [0i16; 20];
        let mut cbk_r = [0i8; 4];
        let mut per_r = 0i8;
        let mut sum_log_r = sum_log0;
        let mut pred_gain_r = 0;
        silk_quant_ltp_gains(
            &mut b_r,
            &mut cbk_r,
            &mut per_r,
            &mut sum_log_r,
            &mut pred_gain_r,
            &xx,
            &x_x,
            subfr_len,
            nb_subfr as usize,
        );
        let mut sum_log_c = sum_log0;
        let (b_c, cbk_c, per_c, pred_gain_c) =
            c::quant_ltp_gains(&mut sum_log_c, &xx, &x_x, subfr_len, nb_subfr);
        let what = format!("it {it}");
        assert_slice_eq(&format!("{what}: B_Q14"), &b_r, &b_c);
        assert_slice_eq(&format!("{what}: cbk"), &cbk_r, &cbk_c);
        assert_eq!(
            [per_r as i32, sum_log_r, pred_gain_r],
            [per_c, sum_log_c, pred_gain_c],
            "{what}: scalars"
        );
    }
}

#[test]
fn vq_wmat_ec_random() {
    let mut rng = Rng::new(0x5EC0_6100);
    for it in 0..20000 {
        let subfr_len = [40, 60, 80][rng.range_i32(0, 2) as usize];
        let quiet = rand_quiet(&mut rng);
        let (mut xx, mut x_x) = rand_ltp_corr(&mut rng, subfr_len as usize, quiet);
        if it % 7 == 0 {
            // Unstructured (possibly indefinite) matrices, including the "no candidate" case.
            xx = core::array::from_fn(|_| rng.range_i32(-300_000, 300_000));
            x_x = core::array::from_fn(|_| rng.range_i32(-300_000, 300_000));
        }
        let cbk = rng.range_i32(0, 2);
        let max_gain_q7 = match rng.range_i32(0, 2) {
            0 => rng.range_i32(-100, 300),
            1 => rng.range_i32(0, 20000),
            _ => i32::MAX - 51,
        };
        let g0 = rng.range_i32(-5, 200);
        let (mut ind_r, mut res_r, mut rate_r, mut gain_r) = (0i8, 0, 0, g0);
        silk_vq_wmat_ec_c(
            &mut ind_r,
            &mut res_r,
            &mut rate_r,
            &mut gain_r,
            &xx,
            &x_x,
            SILK_LTP_VQ_PTRS_Q7[cbk as usize],
            SILK_LTP_VQ_GAIN_PTRS_Q7[cbk as usize],
            SILK_LTP_GAIN_BITS_Q5_PTRS[cbk as usize],
            subfr_len,
            max_gain_q7,
            SILK_LTP_VQ_SIZES[cbk as usize] as i32,
        );
        let mut gain_c = g0;
        let (ind_c, res_c, rate_c) =
            c::vq_wmat_ec(&mut gain_c, &xx, &x_x, cbk, subfr_len, max_gain_q7);
        assert_eq!(
            [ind_r as i32, res_r, rate_r, gain_r],
            [ind_c, res_c, rate_c, gain_c],
            "it {it}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Stereo
// ---------------------------------------------------------------------------------------------

#[test]
fn stereo_quant_pred_random() {
    let mut rng = Rng::new(0x5EC0_7000);
    for it in 0..20000 {
        let mut pred_r = [
            rng.range_i32(-(1 << 14), 1 << 14),
            rng.range_i32(-(1 << 14), 1 << 14),
        ];
        if it % 5 == 0 {
            pred_r[0] = SILK_STEREO_PRED_QUANT_Q13[rng.range_i32(0, 15) as usize] as i32;
        }
        let mut pred_c = pred_r;
        let mut ix_r = [[0i8; 3]; 2];
        let mut ix_c = ix_r;
        silk_stereo_quant_pred(&mut pred_r, &mut ix_r);
        c::stereo_quant_pred(&mut pred_c, &mut ix_c);
        assert_eq!((pred_r, ix_r), (pred_c, ix_c), "it {it}");
    }
}

#[test]
fn stereo_find_predictor_random() {
    let mut rng = Rng::new(0x5EC0_7100);
    for it in 0..5000 {
        let len = rng.range_i32(1, 320) as usize;
        let amp = [10, 300, 5000, 32767][rng.range_i32(0, 3) as usize];
        let x: Vec<i16> = (0..len).map(|_| rng.range_i32(-amp, amp) as i16).collect();
        let k = rng.range_i32(-20000, 20000);
        let y: Vec<i16> = x
            .iter()
            .map(|&v| {
                (((v as i32 * k) >> 14) + rng.range_i32(-amp / 4, amp / 4)).clamp(-32768, 32767)
                    as i16
            })
            .collect();
        let amp0 = [rng.range_i32(0, 40000), rng.range_i32(0, 40000)];
        let smooth = rng.range_i32(0, 32767 - 4097);
        let mut amp_r = amp0;
        let mut ratio_r = 0;
        let pred_r = silk_stereo_find_predictor(&mut ratio_r, &x, &y, &mut amp_r, len, smooth);
        let mut amp_c = amp0;
        let (pred_c, ratio_c) = c::stereo_find_predictor(&x, &y, &mut amp_c, smooth);
        assert_eq!(
            (pred_r, ratio_r, amp_r),
            (pred_c, ratio_c, amp_c),
            "it {it}"
        );
    }
}

#[test]
fn stereo_lr_to_ms_frames() {
    let mut rng = Rng::new(0x5EC0_7200);
    for run in 0..60 {
        let fs_khz = [8, 12, 16][run % 3];
        let fl = (if run % 2 == 0 { 20 } else { 10 } * fs_khz) as usize;
        let mut st_r = StereoEncState::default();
        if run % 7 != 0 {
            // Mono -> stereo initialization from enc_API.c.
            st_r.mid_side_amp_q0 = [0, 1, 0, 1];
            st_r.smth_width_q14 = 1 << 14;
        }
        let mut st_c = stereo_to_c(&st_r);
        let frames = 120;
        let run_rate = [12000, 24000, 40000, 64000, 100_000][run % 5];
        let n = fl * frames + 2;
        let seed = rng.next_u64();
        let mode = (run / 3) % 5;
        let base = if mode == 4 {
            signals::to_i16(&signals::music_like(n, 2, fs_khz as u32 * 1000, seed))
        } else {
            signals::to_i16(&signals::speech_like(n, 2, fs_khz as u32 * 1000, seed))
        };
        let mut left: Vec<i16> = base.iter().step_by(2).copied().collect();
        let mut right: Vec<i16> = base.iter().skip(1).step_by(2).copied().collect();
        match mode {
            // identical channels: panned mono
            0 => right.clone_from(&left),
            // amplitude panned
            1 => right.iter_mut().for_each(|v| *v = (*v as i32 / 3) as i16),
            // independent
            2 => right = signals::to_i16(&signals::noise(n, 1, 0.3, seed ^ 1)),
            // loud, near-clipping, partly anti-phase content
            3 => {
                let noise = signals::to_i16(&signals::noise(n, 1, 0.05, seed ^ 2));
                for ((l, r), e) in left.iter_mut().zip(right.iter_mut()).zip(noise) {
                    *l = (*l as i32 * 8).clamp(-32768, 32767) as i16;
                    *r = (-(*r as i32) * 6 + e as i32).clamp(-32768, 32767) as i16;
                }
            }
            // music-like stereo as generated
            _ => {}
        }
        for f in 0..frames {
            let mut x1_r = left[f * fl..f * fl + fl + 2].to_vec();
            let mut x2_r = right[f * fl..f * fl + fl + 2].to_vec();
            let mut x1_c = x1_r.clone();
            let mut x2_c = x2_r.clone();
            // Mostly a stable rate per run (so the width smoother converges), sometimes random.
            let total = match rng.range_i32(0, 9) {
                0 => rng.range_i32(-1000, 8000),
                1 => rng.range_i32(8000, 20000),
                _ => run_rate,
            };
            let prev_sa = if rng.range_i32(0, 3) == 0 {
                rng.range_i32(0, 255)
            } else {
                255
            };
            let to_mono = i32::from(rng.range_i32(0, 15) == 0);
            let mut ix_r = [[0i8; 3]; 2];
            let mut mid_only_r = 0i8;
            let mut rates_r = [0i32; 2];
            silk_stereo_lr_to_ms(
                &mut st_r,
                &mut x1_r,
                &mut x2_r,
                &mut ix_r,
                &mut mid_only_r,
                &mut rates_r,
                total,
                prev_sa,
                to_mono,
                fs_khz,
                fl as i32,
            );
            let (ix_c, mid_only_c, rates_c) = c::stereo_lr_to_ms(
                &mut st_c, &mut x1_c, &mut x2_c, total, prev_sa, to_mono, fs_khz, fl as i32,
            );
            let what = format!("run {run} frame {f}");
            assert_slice_eq(&format!("{what}: x1"), &x1_r, &x1_c);
            assert_slice_eq(&format!("{what}: x2"), &x2_r, &x2_c);
            assert_eq!(
                (ix_r, mid_only_r, rates_r),
                (ix_c, mid_only_c, rates_c),
                "{what}: outputs"
            );
            assert_eq!(stereo_to_c(&st_r), st_c, "{what}: state");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// HP_variable_cutoff / control_SNR / control_audio_bandwidth / check_control_input
// ---------------------------------------------------------------------------------------------

#[test]
fn hp_variable_cutoff_sequences() {
    let mut rng = Rng::new(0x5EC0_8000);
    let lo = silk_lin2log(60) << 8;
    let hi = silk_lin2log(100) << 8;
    for run in 0..500 {
        let fs_khz = [8, 12, 16][rng.range_i32(0, 2) as usize];
        let mut enc = enc_state(fs_khz, 4);
        enc.variable_hp_smth1_q15 = match rng.range_i32(0, 2) {
            0 => rng.range_i32(lo, hi),
            1 => rng.range_i32(0, 1 << 20),
            _ => lo,
        };
        for step in 0..40 {
            enc.prev_signal_type = rng.range_i32(0, 2) as i8;
            enc.prev_lag = rng.range_i32(2 * fs_khz, 18 * fs_khz);
            enc.speech_activity_q8 = rng.range_i32(0, 255);
            enc.input_quality_bands_q15[0] = rng.range_i32(0, 32767);
            let mut p = enc_params(&enc);
            silk_hp_variable_cutoff(&mut enc);
            c::hp_variable_cutoff(&mut p);
            assert_eq!(enc_params(&enc), p, "run {run} step {step}");
        }
    }
}

#[test]
fn control_snr_all_rates() {
    for fs_khz in [8, 12, 16] {
        for nb_subfr in [2, 4] {
            for rate in (-2000..110_000).step_by(37) {
                let mut enc = enc_state(fs_khz, nb_subfr);
                let mut p = enc_params(&enc);
                let r = silk_control_snr(&mut enc, rate);
                let rc = c::control_snr(&mut p, rate);
                assert_eq!(r, rc);
                assert_eq!(enc_params(&enc), p, "fs {fs_khz} nb {nb_subfr} rate {rate}");
            }
        }
    }
}

#[test]
fn control_audio_bandwidth_random() {
    let mut rng = Rng::new(0x5EC0_8100);
    let rates = [8000, 12000, 16000];
    let api_rates = [8000, 12000, 16000, 24000, 32000, 44100, 48000];
    for it in 0..50000 {
        let mut enc = enc_state(8, 4);
        enc.fs_khz = [0, 8, 12, 16][rng.range_i32(0, 3) as usize];
        enc.s_lp.saved_fs_khz = [0, 8, 12, 16][rng.range_i32(0, 3) as usize];
        enc.api_fs_hz = api_rates[rng.range_i32(0, 6) as usize];
        enc.max_internal_fs_hz = rates[rng.range_i32(0, 2) as usize];
        enc.min_internal_fs_hz = rates[rng.range_i32(0, 2) as usize];
        enc.desired_internal_fs_hz = rates[rng.range_i32(0, 2) as usize];
        enc.allow_bandwidth_switch = rng.range_i32(0, 1);
        enc.s_lp.transition_frame_no = rng.range_i32(0, TRANSITION_FRAMES + 1);
        enc.s_lp.mode = rng.range_i32(-2, 1);
        enc.s_lp.in_lp_state = [rng.next_u32() as i32, rng.next_u32() as i32];
        let mut ctl = SilkEncControlStruct {
            opus_can_switch: rng.range_i32(0, 1),
            max_bits: rng.range_i32(0, 10000),
            payload_size_ms: [10, 20, 40, 60][rng.range_i32(0, 3) as usize],
            switch_ready: rng.range_i32(0, 1),
            ..SilkEncControlStruct::default()
        };
        let mut p = enc_params(&enc);
        let mut ctl_c = ctl_to_c(&ctl);
        let r = silk_control_audio_bandwidth(&mut enc, &mut ctl);
        let rc = c::control_audio_bandwidth(&mut p, &mut ctl_c);
        assert_eq!(r, rc, "it {it}: fs_kHz");
        assert_eq!(enc_params(&enc), p, "it {it}: state");
        assert_eq!(ctl_to_c(&ctl), ctl_c, "it {it}: control");
    }
}

const fn pick(valid: &[i32], rng: &mut Rng) -> i32 {
    if rng.range_i32(0, 9) == 0 {
        rng.range_i32(-5, 100_000)
    } else {
        valid[rng.range_i32(0, valid.len() as i32 - 1) as usize]
    }
}

#[test]
fn check_control_input_random() {
    let mut rng = Rng::new(0x5EC0_8200);
    let mut seen = std::collections::BTreeSet::new();
    for it in 0..100_000 {
        let ctl = SilkEncControlStruct {
            api_sample_rate: pick(
                &[8000, 12000, 16000, 24000, 32000, 44100, 48000, 96000],
                &mut rng,
            ),
            desired_internal_sample_rate: pick(&[8000, 12000, 16000], &mut rng),
            max_internal_sample_rate: pick(&[8000, 12000, 16000], &mut rng),
            min_internal_sample_rate: pick(&[8000, 12000, 16000], &mut rng),
            payload_size_ms: pick(&[10, 20, 40, 60], &mut rng),
            packet_loss_percentage: pick(&[0, 1, 50, 100], &mut rng),
            use_dtx: pick(&[0, 1], &mut rng),
            use_cbr: pick(&[0, 1], &mut rng),
            use_in_band_fec: pick(&[0, 1], &mut rng),
            n_channels_api: pick(&[1, 2], &mut rng),
            n_channels_internal: pick(&[1, 2], &mut rng),
            complexity: pick(&[0, 3, 5, 10], &mut rng),
            ..SilkEncControlStruct::default()
        };
        let r = check_control_input(&ctl);
        assert_eq!(
            r,
            c::check_control_input(&ctl_to_c(&ctl)),
            "it {it}: {ctl:?}"
        );
        seen.insert(r);
    }
    // Every return code must have been exercised.
    assert_eq!(seen.len(), 9, "codes seen: {seen:?}");
}
