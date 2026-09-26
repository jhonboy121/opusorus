//! Differential tests for unit `silk_common`: SILK tables, struct layouts, signal-processing
//! helpers, NLSF code and entropy-coding helpers vs the C oracle (bit-exact).

use opusorus::celt::entdec::EcDec;
use opusorus::celt::entenc::EcEnc;
use opusorus::silk::coding::*;
use opusorus::silk::nlsf::*;
use opusorus::silk::sigproc::*;
use opusorus::silk::structs::*;
use opusorus::silk::tables::*;
use opusorus_conformance::{Rng, assert_slice_eq};
use opusorus_oracle::silk_common as c;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

/// Runs `f` on a fresh encoder after `pre_n` raw bits `pre`, mirroring the C harness.
fn rust_enc(size: usize, pre: u32, pre_n: i32, f: impl FnOnce(&mut EcEnc<'_>)) -> c::EncRes {
    let mut buf = vec![0u8; size];
    let res = {
        let mut e = EcEnc::new(&mut buf);
        if pre_n > 0 {
            e.enc_bits(pre, pre_n as u32);
        }
        f(&mut e);
        let t = e.tell_frac();
        e.done();
        [t, e.rng, e.error as u32]
    };
    c::EncRes { buf, res }
}

/// Runs `f` on a decoder over `data` after skipping `pre_n` raw bits.
fn rust_dec<T>(data: &[u8], pre_n: i32, f: impl FnOnce(&mut EcDec<'_>) -> T) -> (T, c::DecRes) {
    let mut d = EcDec::new(data);
    if pre_n > 0 {
        d.dec_bits(pre_n as u32);
    }
    let v = f(&mut d);
    (v, [d.tell_frac(), d.rng, d.error as u32])
}

const fn rand_pre(rng: &mut Rng) -> (u32, i32) {
    let n = rng.range_i32(0, 16);
    let v = if n == 0 {
        0
    } else {
        rng.next_u32() & ((1u32 << n) - 1)
    };
    (v, n)
}

/// Random sorted NLSF vector in Q15 with a minimum spacing.
fn rand_nlsf(rng: &mut Rng, d: usize, min_gap: i32) -> Vec<i16> {
    loop {
        let mut v: Vec<i32> = (0..d).map(|_| rng.range_i32(1, 32766)).collect();
        v.sort_unstable();
        let mut ok = true;
        for i in 1..d {
            if v[i] - v[i - 1] < min_gap {
                ok = false;
            }
        }
        if ok || min_gap == 0 {
            return v.into_iter().map(|x| x as i16).collect();
        }
        // Spread uniformly instead of rejecting forever.
        if rng.range_i32(0, 3) == 0 {
            let step = 32768 / (d as i32 + 1);
            return (0..d)
                .map(|i| (step * (i as i32 + 1) + rng.range_i32(-step / 3, step / 3)) as i16)
                .collect();
        }
    }
}

const fn frame_lengths() -> [i32; 6] {
    [80, 120, 160, 240, 320, 160]
}

// ---------------------------------------------------------------------------------------------
// Tables and structs
// ---------------------------------------------------------------------------------------------

#[test]
fn tables_match_oracle() {
    let nb = &SILK_NLSF_CB_NB_MB;
    let wb = &SILK_NLSF_CB_WB;
    let u8_tables: Vec<(&str, &[u8])> = vec![
        ("gain_iCDF", SILK_GAIN_ICDF.as_flattened()),
        ("delta_gain_iCDF", &SILK_DELTA_GAIN_ICDF),
        ("pitch_lag_iCDF", &SILK_PITCH_LAG_ICDF),
        ("pitch_delta_iCDF", &SILK_PITCH_DELTA_ICDF),
        ("pitch_contour_iCDF", &SILK_PITCH_CONTOUR_ICDF),
        ("pitch_contour_NB_iCDF", &SILK_PITCH_CONTOUR_NB_ICDF),
        ("pitch_contour_10_ms_iCDF", &SILK_PITCH_CONTOUR_10_MS_ICDF),
        (
            "pitch_contour_10_ms_NB_iCDF",
            &SILK_PITCH_CONTOUR_10_MS_NB_ICDF,
        ),
        (
            "pulses_per_block_iCDF",
            SILK_PULSES_PER_BLOCK_ICDF.as_flattened(),
        ),
        (
            "pulses_per_block_BITS_Q5",
            SILK_PULSES_PER_BLOCK_BITS_Q5.as_flattened(),
        ),
        ("rate_levels_iCDF", SILK_RATE_LEVELS_ICDF.as_flattened()),
        (
            "rate_levels_BITS_Q5",
            SILK_RATE_LEVELS_BITS_Q5.as_flattened(),
        ),
        ("max_pulses_table", &SILK_MAX_PULSES_TABLE),
        ("shell_code_table0", &SILK_SHELL_CODE_TABLE0),
        ("shell_code_table1", &SILK_SHELL_CODE_TABLE1),
        ("shell_code_table2", &SILK_SHELL_CODE_TABLE2),
        ("shell_code_table3", &SILK_SHELL_CODE_TABLE3),
        ("shell_code_table_offsets", &SILK_SHELL_CODE_TABLE_OFFSETS),
        ("lsb_iCDF", &SILK_LSB_ICDF),
        ("sign_iCDF", &SILK_SIGN_ICDF),
        ("uniform3_iCDF", &SILK_UNIFORM3_ICDF),
        ("uniform4_iCDF", &SILK_UNIFORM4_ICDF),
        ("uniform5_iCDF", &SILK_UNIFORM5_ICDF),
        ("uniform6_iCDF", &SILK_UNIFORM6_ICDF),
        ("uniform8_iCDF", &SILK_UNIFORM8_ICDF),
        ("NLSF_EXT_iCDF", &SILK_NLSF_EXT_ICDF),
        ("LTP_per_index_iCDF", &SILK_LTP_PER_INDEX_ICDF),
        ("LTP_gain_iCDF_ptrs[0]", SILK_LTP_GAIN_ICDF_PTRS[0]),
        ("LTP_gain_iCDF_ptrs[1]", SILK_LTP_GAIN_ICDF_PTRS[1]),
        ("LTP_gain_iCDF_ptrs[2]", SILK_LTP_GAIN_ICDF_PTRS[2]),
        ("LTP_gain_BITS_Q5_ptrs[0]", SILK_LTP_GAIN_BITS_Q5_PTRS[0]),
        ("LTP_gain_BITS_Q5_ptrs[1]", SILK_LTP_GAIN_BITS_Q5_PTRS[1]),
        ("LTP_gain_BITS_Q5_ptrs[2]", SILK_LTP_GAIN_BITS_Q5_PTRS[2]),
        ("LTP_vq_gain_ptrs_Q7[0]", SILK_LTP_VQ_GAIN_PTRS_Q7[0]),
        ("LTP_vq_gain_ptrs_Q7[1]", SILK_LTP_VQ_GAIN_PTRS_Q7[1]),
        ("LTP_vq_gain_ptrs_Q7[2]", SILK_LTP_VQ_GAIN_PTRS_Q7[2]),
        ("LTPscale_iCDF", &SILK_LTPSCALE_ICDF),
        ("type_offset_VAD_iCDF", &SILK_TYPE_OFFSET_VAD_ICDF),
        ("type_offset_no_VAD_iCDF", &SILK_TYPE_OFFSET_NO_VAD_ICDF),
        ("stereo_pred_joint_iCDF", &SILK_STEREO_PRED_JOINT_ICDF),
        ("stereo_only_code_mid_iCDF", &SILK_STEREO_ONLY_CODE_MID_ICDF),
        ("LBRR_flags_iCDF_ptr[0]", SILK_LBRR_FLAGS_ICDF_PTR[0]),
        ("LBRR_flags_iCDF_ptr[1]", SILK_LBRR_FLAGS_ICDF_PTR[1]),
        (
            "NLSF_interpolation_factor_iCDF",
            &SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
        ),
        ("NB_MB.CB1_NLSF_Q8", nb.cb1_nlsf_q8),
        ("NB_MB.CB1_iCDF", nb.cb1_icdf),
        ("NB_MB.pred_Q8", nb.pred_q8),
        ("NB_MB.ec_sel", nb.ec_sel),
        ("NB_MB.ec_iCDF", nb.ec_icdf),
        ("NB_MB.ec_Rates_Q5", nb.ec_rates_q5),
        ("WB.CB1_NLSF_Q8", wb.cb1_nlsf_q8),
        ("WB.CB1_iCDF", wb.cb1_icdf),
        ("WB.pred_Q8", wb.pred_q8),
        ("WB.ec_sel", wb.ec_sel),
        ("WB.ec_iCDF", wb.ec_icdf),
        ("WB.ec_Rates_Q5", wb.ec_rates_q5),
    ];
    for (id, (name, t)) in u8_tables.iter().enumerate() {
        let ct = c::table_u8(id as i32).unwrap_or_else(|| panic!("no C table {name}"));
        assert_slice_eq(name, t, &ct);
    }
    assert!(c::table_u8(u8_tables.len() as i32).is_none());

    let i8_tables: Vec<(&str, &[i8])> = vec![
        ("LTP_vq_ptrs_Q7[0]", SILK_LTP_VQ_PTRS_Q7[0].as_flattened()),
        ("LTP_vq_ptrs_Q7[1]", SILK_LTP_VQ_PTRS_Q7[1].as_flattened()),
        ("LTP_vq_ptrs_Q7[2]", SILK_LTP_VQ_PTRS_Q7[2].as_flattened()),
        ("LTP_vq_sizes", &SILK_LTP_VQ_SIZES),
        ("CB_lags_stage2", SILK_CB_LAGS_STAGE2.as_flattened()),
        ("CB_lags_stage3", SILK_CB_LAGS_STAGE3.as_flattened()),
        (
            "Lag_range_stage3",
            SILK_LAG_RANGE_STAGE3.as_flattened().as_flattened(),
        ),
        ("nb_cbk_searchs_stage3", &SILK_NB_CBK_SEARCHS_STAGE3),
        (
            "CB_lags_stage2_10_ms",
            SILK_CB_LAGS_STAGE2_10_MS.as_flattened(),
        ),
        (
            "CB_lags_stage3_10_ms",
            SILK_CB_LAGS_STAGE3_10_MS.as_flattened(),
        ),
        (
            "Lag_range_stage3_10_ms",
            SILK_LAG_RANGE_STAGE3_10_MS.as_flattened(),
        ),
    ];
    for (id, (name, t)) in i8_tables.iter().enumerate() {
        let ct = c::table_i8(id as i32).unwrap_or_else(|| panic!("no C table {name}"));
        assert_slice_eq(name, t, &ct);
    }
    assert!(c::table_i8(i8_tables.len() as i32).is_none());

    let nb_scal = [
        nb.n_vectors,
        nb.order,
        nb.quant_step_size_q16,
        nb.inv_quant_step_size_q6,
    ];
    let wb_scal = [
        wb.n_vectors,
        wb.order,
        wb.quant_step_size_q16,
        wb.inv_quant_step_size_q6,
    ];
    let i16_tables: Vec<(&str, &[i16])> = vec![
        ("LTPScales_table_Q14", &SILK_LTPSCALES_TABLE_Q14),
        ("stereo_pred_quant_Q13", &SILK_STEREO_PRED_QUANT_Q13),
        (
            "Quantization_Offsets_Q10",
            SILK_QUANTIZATION_OFFSETS_Q10.as_flattened(),
        ),
        ("LSFCosTab_FIX_Q12", &SILK_LSFCOSTAB_FIX_Q12),
        ("NB_MB.CB1_Wght_Q9", nb.cb1_wght_q9),
        ("NB_MB.deltaMin_Q15", nb.delta_min_q15),
        ("WB.CB1_Wght_Q9", wb.cb1_wght_q9),
        ("WB.deltaMin_Q15", wb.delta_min_q15),
        ("NB_MB scalars", &nb_scal),
        ("WB scalars", &wb_scal),
    ];
    for (id, (name, t)) in i16_tables.iter().enumerate() {
        let ct = c::table_i16(id as i32).unwrap_or_else(|| panic!("no C table {name}"));
        assert_slice_eq(name, t, &ct);
    }
    assert!(c::table_i16(i16_tables.len() as i32).is_none());

    let i32_tables: Vec<(&str, &[i32])> = vec![
        (
            "Transition_LP_B_Q28",
            SILK_TRANSITION_LP_B_Q28.as_flattened(),
        ),
        (
            "Transition_LP_A_Q28",
            SILK_TRANSITION_LP_A_Q28.as_flattened(),
        ),
    ];
    for (id, (name, t)) in i32_tables.iter().enumerate() {
        let ct = c::table_i32(id as i32).unwrap_or_else(|| panic!("no C table {name}"));
        assert_slice_eq(name, t, &ct);
    }
    assert!(c::table_i32(i32_tables.len() as i32).is_none());

    // pitch_est_defines.h constants that size the tables above.
    assert_eq!(SILK_NB_CBK_SEARCHS_STAGE3[2] as i32, PE_NB_CBKS_STAGE3_MAX);
    assert_eq!(PE_MAX_FRAME_LENGTH, 640);
    assert_eq!(PE_MAX_LAG, 288);
}

#[test]
fn struct_array_sizes_match_oracle() {
    let nsq = SilkNsqState::new();
    let vad = SilkVadState::default();
    let lp = SilkLpState::default();
    let se = StereoEncState::default();
    let sd = StereoDecState::default();
    let ind = SideInfoIndices::default();
    let enc = SilkEncoderState::new();
    let plc = SilkPlcStruct::default();
    let cng = SilkCngStruct::new();
    let dec = SilkDecoderState::new();
    let ctl = SilkDecoderControl::default();
    let rust = vec![
        nsq.xq.len(),
        nsq.s_ltp_shp_q14.len(),
        nsq.s_lpc_q14.len(),
        nsq.s_ar2_q14.len(),
        vad.ana_state.len(),
        vad.ana_state1.len(),
        vad.ana_state2.len(),
        vad.xnrg_subfr.len(),
        vad.nrg_ratio_smth_q8.len(),
        vad.nl.len(),
        vad.inv_nl.len(),
        vad.noise_level_bias.len(),
        lp.in_lp_state.len(),
        se.pred_prev_q13.len(),
        se.s_mid.len(),
        se.s_side.len(),
        se.mid_side_amp_q0.len(),
        se.pred_ix.len(),
        se.mid_only_flags.len(),
        sd.pred_prev_q13.len(),
        sd.s_mid.len(),
        sd.s_side.len(),
        ind.gains_indices.len(),
        ind.ltp_index.len(),
        ind.nlsf_indices.len(),
        enc.in_hp_state.len(),
        enc.prev_nlsfq_q15.len(),
        enc.input_quality_bands_q15.len(),
        enc.vad_flags.len(),
        enc.lbrr_flags.len(),
        enc.pulses.len(),
        enc.input_buf.len(),
        enc.indices_lbrr.len(),
        enc.pulses_lbrr.len(),
        enc.pulses_lbrr[0].len(),
        plc.ltp_coef_q14.len(),
        plc.prev_lpc_q12.len(),
        plc.prev_gain_q16.len(),
        cng.cng_exc_buf_q14.len(),
        cng.cng_smth_nlsf_q15.len(),
        cng.cng_synth_state.len(),
        dec.exc_q14.len(),
        dec.s_lpc_q14_buf.len(),
        dec.out_buf.len(),
        dec.prev_nlsf_q15.len(),
        dec.vad_flags.len(),
        dec.lbrr_flags.len(),
        ctl.pitch_l.len(),
        ctl.gains_q16.len(),
        ctl.pred_coef_q12.len(),
        ctl.pred_coef_q12[0].len(),
        ctl.ltp_coef_q14.len(),
    ];
    let rust: Vec<i32> = rust.into_iter().map(|v| v as i32).collect();
    assert_slice_eq("struct array sizes", &rust, &c::struct_dims());

    // new()/reset() give the all-zero (memset) state.
    let mut d = SilkDecoderState::new();
    d.prev_gain_q16 = 5;
    d.out_buf[3] = 7;
    d.s_cng.cng_exc_buf_q14[10] = 1;
    d.reset();
    assert_eq!(d.prev_gain_q16, 0);
    assert!(d.out_buf.iter().all(|&v| v == 0));
    assert!(d.s_cng.cng_exc_buf_q14.iter().all(|&v| v == 0));
    let mut e = SilkEncoderState::default();
    e.pulses_lbrr[2][5] = 3;
    e.reset();
    assert!(e.pulses_lbrr.iter().all(|r| r.iter().all(|&v| v == 0)));
}

// ---------------------------------------------------------------------------------------------
// Scalar functions
// ---------------------------------------------------------------------------------------------

#[test]
fn lin2log_log2lin_sigm_match_oracle() {
    let mut rng = Rng::new(21);
    let edges = [
        0,
        1,
        2,
        3,
        127,
        128,
        129,
        32767,
        65536,
        i32::MAX,
        -1,
        i32::MIN,
        -65536,
    ];
    for &x in &edges {
        assert_eq!(silk_lin2log(x), c::lin2log(x), "lin2log({x})");
    }
    for _ in 0..200_000 {
        let x = rng.next_u32() as i32 >> rng.range_i32(0, 31);
        assert_eq!(silk_lin2log(x), c::lin2log(x), "lin2log({x})");
    }
    for x in -200..5000 {
        assert_eq!(silk_log2lin(x), c::log2lin(x), "log2lin({x})");
    }
    for _ in 0..100_000 {
        let x = rng.next_u32() as i32 >> rng.range_i32(0, 31);
        assert_eq!(silk_log2lin(x), c::log2lin(x), "log2lin({x})");
    }
    for x in -500..500 {
        assert_eq!(silk_sigm_q15(x), c::sigm_q15(x), "sigm({x})");
    }
    for _ in 0..100_000 {
        let x = rng.range_i32(-1_000_000, 1_000_000);
        assert_eq!(silk_sigm_q15(x), c::sigm_q15(x), "sigm({x})");
    }
}

#[test]
fn insertion_sorts_match_oracle() {
    let mut rng = Rng::new(22);
    for iter in 0..20_000 {
        let l = rng.range_i32(1, 48) as usize;
        let k = rng.range_i32(1, l as i32) as usize;
        let spread = [3, 100, i32::MAX][iter % 3];
        let a0: Vec<i32> = (0..l).map(|_| rng.range_i32(-spread, spread)).collect();
        let mut ar = a0.clone();
        let mut ac = a0.clone();
        let mut idx = vec![0i32; k];
        silk_insertion_sort_increasing(&mut ar, &mut idx, l, k);
        let idc = c::insertion_sort_increasing(&mut ac, l, k);
        assert_slice_eq("sort a", &ar, &ac);
        assert_slice_eq("sort idx", &idx, &idc);

        let b0: Vec<i16> = (0..l)
            .map(|_| rng.range_i32(-spread.min(32768), spread.min(32767)) as i16)
            .collect();
        let mut br = b0.clone();
        let mut bc = b0;
        silk_insertion_sort_increasing_all_values_int16(&mut br, l);
        c::insertion_sort_increasing_all_values_int16(&mut bc);
        assert_slice_eq("sort16", &br, &bc);
    }
}

#[test]
fn bwexpander_match_oracle() {
    let mut rng = Rng::new(23);
    for _ in 0..30_000 {
        let d = rng.range_i32(1, 24) as usize;
        let chirp = if rng.range_i32(0, 4) == 0 {
            [0, 1, 65535, 65536, 32768][rng.range_i32(0, 4) as usize]
        } else {
            rng.range_i32(0, 65536)
        };
        let a16: Vec<i16> = (0..d).map(|_| rng.i16()).collect();
        let mut r16 = a16.clone();
        let mut c16 = a16;
        silk_bwexpander(&mut r16, d, chirp);
        c::bwexpander(&mut c16, chirp);
        assert_slice_eq("bwexpander", &r16, &c16);

        let a32: Vec<i32> = (0..d)
            .map(|_| rng.next_u32() as i32 >> rng.range_i32(0, 20))
            .collect();
        let mut r32 = a32.clone();
        let mut c32 = a32;
        silk_bwexpander_32(&mut r32, d, chirp);
        c::bwexpander_32(&mut c32, chirp);
        assert_slice_eq("bwexpander_32", &r32, &c32);
    }
}

#[test]
fn inner_products_and_sum_sqr_match_oracle() {
    let mut rng = Rng::new(24);
    for iter in 0..20_000 {
        let len = rng.range_i32(0, 480) as usize;
        let scale = rng.range_i32(0, 15);
        // Keep len * amp^2 >> scale inside int32 (C would overflow otherwise).
        let max_bits = ((31 + scale - 10) / 2).min(15);
        let amp = 1i32 << rng.range_i32(0, max_bits);
        let a: Vec<i16> = (0..len)
            .map(|_| rng.range_i32(-amp, amp - 1) as i16)
            .collect();
        let b: Vec<i16> = (0..len)
            .map(|_| rng.range_i32(-amp, amp - 1) as i16)
            .collect();
        assert_eq!(
            silk_inner_prod_aligned_scale(&a, &b, scale, len),
            c::inner_prod_aligned_scale(&a, &b, scale),
            "inner_prod_aligned_scale iter {iter}"
        );
        let fa: Vec<i16> = (0..len).map(|_| rng.i16()).collect();
        let fb: Vec<i16> = (0..len).map(|_| rng.i16()).collect();
        assert_eq!(
            silk_inner_prod16(&fa, &fb, len),
            c::inner_prod16(&fa, &fb),
            "inner_prod16"
        );

        let x: Vec<i16> = match iter % 4 {
            0 => fa.clone(),
            1 => vec![-32768; len],
            2 => (0..len).map(|_| rng.range_i32(-3, 3) as i16).collect(),
            _ => a.clone(),
        };
        assert_eq!(
            silk_sum_sqr_shift(&x, len),
            c::sum_sqr_shift(&x),
            "sum_sqr_shift len {len}"
        );
    }
}

#[test]
fn interpolate_match_oracle() {
    let mut rng = Rng::new(25);
    for _ in 0..20_000 {
        let d = rng.range_i32(1, 16) as usize;
        let x0: Vec<i16> = (0..d).map(|_| rng.i16()).collect();
        let x1: Vec<i16> = (0..d).map(|_| rng.i16()).collect();
        let f = rng.range_i32(0, 4);
        let mut xr = vec![0i16; d];
        silk_interpolate(&mut xr, &x0, &x1, f, d);
        assert_slice_eq("interpolate", &xr, &c::interpolate(&x0, &x1, f));
    }
}

// ---------------------------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------------------------

/// High-pass biquad coefficients as computed by the SILK encoder (`silk_HP_variable_cutoff`
/// feeding `opus_encoder.c:hp_cutoff`), for realistic filter tests.
#[expect(
    clippy::approx_constant,
    reason = "3.14159 is the literal the C encoder uses"
)]
fn hp_coefs(cutoff_hz: i32, fs: i32) -> ([i32; 3], [i32; 2]) {
    use opusorus::silk::macros::*;
    let fc_q19 = silk_div32_16(
        silk_smulbb(silk_fix_const(1.5 * 3.14159 / 1000.0, 19), cutoff_hz),
        fs / 1000,
    );
    let r_q28 = silk_fix_const(1.0, 28) - silk_mul(silk_fix_const(0.92, 9), fc_q19);
    let b = [r_q28, silk_lshift(-r_q28, 1), r_q28];
    let r_q22 = silk_rshift(r_q28, 6);
    let a = [
        silk_smulww(r_q22, silk_smulww(fc_q19, fc_q19) - silk_fix_const(2.0, 22)),
        silk_smulww(r_q22, r_q22),
    ];
    (b, a)
}

fn rand_signal(rng: &mut Rng, n: usize) -> Vec<i16> {
    match rng.range_i32(0, 3) {
        0 => (0..n).map(|_| rng.i16()).collect(),
        1 => {
            let f = 0.01 + 0.3 * (rng.next_u32() as f64 / u32::MAX as f64);
            let a = 32767.0 * (rng.next_u32() as f64 / u32::MAX as f64);
            (0..n).map(|i| (a * (f * i as f64).sin()) as i16).collect()
        }
        2 => (0..n)
            .map(|i| if (i / 7) % 2 == 0 { 32767 } else { -32768 })
            .collect(),
        _ => (0..n).map(|_| rng.range_i32(-100, 100) as i16).collect(),
    }
}

#[test]
fn biquads_match_oracle() {
    let mut rng = Rng::new(26);
    for iter in 0..6000 {
        let (b, a) = if iter % 2 == 0 {
            let fs = [8000, 12000, 16000, 24000, 48000][rng.range_i32(0, 4) as usize];
            hp_coefs(rng.range_i32(20, 400), fs)
        } else {
            let k = rng.range_i32(0, 4) as usize;
            (SILK_TRANSITION_LP_B_Q28[k], SILK_TRANSITION_LP_A_Q28[k])
        };
        let len = rng.range_i32(0, 400) as usize;
        let x = rand_signal(&mut rng, 2 * len);

        let s0 = [rng.range_i32(-1000, 1000), rng.range_i32(-1000, 1000)];
        let mut sr = s0;
        let mut sc = s0;
        let mut yr = vec![0i16; len];
        silk_biquad_alt_stride1(&x[..len], &b, &a, &mut sr, &mut yr, len);
        let yc = c::biquad_alt_stride1(&x[..len], &b, &a, &mut sc);
        assert_slice_eq("biquad stride1", &yr, &yc);
        assert_slice_eq("biquad stride1 state", &sr, &sc);

        // In-place variant equals the out-of-place one.
        let mut buf = x[..len].to_vec();
        let mut si = s0;
        silk_biquad_alt_stride1_inplace(&mut buf, &b, &a, &mut si, len);
        assert_slice_eq("biquad inplace", &buf, &yc);
        assert_slice_eq("biquad inplace state", &si, &sc);

        let s4 = [
            rng.range_i32(-1000, 1000),
            rng.range_i32(-1000, 1000),
            rng.range_i32(-1000, 1000),
            rng.range_i32(-1000, 1000),
        ];
        let mut sr = s4;
        let mut sc = s4;
        let mut yr = vec![0i16; 2 * len];
        silk_biquad_alt_stride2(&x, &b, &a, &mut sr, &mut yr, len);
        let yc = c::biquad_alt_stride2(&x, &b, &a, &mut sc);
        assert_slice_eq("biquad stride2", &yr, &yc);
        assert_slice_eq("biquad stride2 state", &sr, &sc);
    }
}

#[test]
fn lp_variable_cutoff_matches_oracle() {
    let mut rng = Rng::new(27);
    for _ in 0..1500 {
        let mut lp = SilkLpState {
            in_lp_state: [0, 0],
            transition_frame_no: rng.range_i32(0, 256),
            mode: rng.range_i32(-1, 1),
            saved_fs_khz: rng.range_i32(0, 16),
        };
        let mut st = [
            lp.in_lp_state[0],
            lp.in_lp_state[1],
            lp.transition_frame_no,
            lp.mode,
            lp.saved_fs_khz,
        ];
        let frame_len = [80usize, 120, 160, 240, 320][rng.range_i32(0, 4) as usize];
        for _ in 0..rng.range_i32(1, 12) {
            let x = rand_signal(&mut rng, frame_len);
            let mut fr = x.clone();
            let mut fc = x;
            silk_lp_variable_cutoff(&mut lp, &mut fr, frame_len);
            c::lp_variable_cutoff(&mut st, &mut fc);
            assert_slice_eq("LP_variable_cutoff", &fr, &fc);
            assert_eq!(
                [
                    lp.in_lp_state[0],
                    lp.in_lp_state[1],
                    lp.transition_frame_no,
                    lp.mode,
                    lp.saved_fs_khz
                ],
                st
            );
        }
    }
    // Every interpolation point.
    for t in 0..=256 {
        for mode in [-1, 1] {
            let mut lp = SilkLpState {
                in_lp_state: [0, 0],
                transition_frame_no: t,
                mode,
                saved_fs_khz: 0,
            };
            let mut st = [0, 0, t, mode, 0];
            let x = rand_signal(&mut rng, 160);
            let mut fr = x.clone();
            let mut fc = x;
            silk_lp_variable_cutoff(&mut lp, &mut fr, 160);
            c::lp_variable_cutoff(&mut st, &mut fc);
            assert_slice_eq("LP_variable_cutoff sweep", &fr, &fc);
            assert_eq!(lp.transition_frame_no, st[2]);
        }
    }
}

#[test]
fn ana_filt_bank_1_matches_oracle() {
    let mut rng = Rng::new(28);
    for _ in 0..3000 {
        let mut sr = [
            rng.range_i32(-100_000, 100_000),
            rng.range_i32(-100_000, 100_000),
        ];
        let mut sc = sr;
        for _ in 0..rng.range_i32(1, 4) {
            let n = 2 * rng.range_i32(0, 320) as usize + (rng.range_i32(0, 1) as usize);
            let x = rand_signal(&mut rng, n);
            let mut lr = vec![0i16; n / 2];
            let mut hr = vec![0i16; n / 2];
            silk_ana_filt_bank_1(&x, &mut sr, &mut lr, &mut hr, n);
            let (lc, hc) = c::ana_filt_bank_1(&x, &mut sc);
            assert_slice_eq("ana low", &lr, &lc);
            assert_slice_eq("ana high", &hr, &hc);
            assert_eq!(sr, sc);
        }
    }
}

/// Random stable Q12 LPC filter via NLSF2A of a random NLSF vector.
fn rand_stable_a_q12(rng: &mut Rng, d: usize) -> Vec<i16> {
    let nlsf = rand_nlsf(rng, d, 1);
    let mut a = vec![0i16; d];
    silk_nlsf2a(&mut a, &nlsf, d);
    a
}

#[test]
fn lpc_inverse_pred_gain_matches_oracle() {
    let mut rng = Rng::new(29);
    for iter in 0..40_000 {
        let a: Vec<i16> = match iter % 4 {
            0 => {
                let d = if rng.range_i32(0, 1) == 0 { 10 } else { 16 };
                rand_stable_a_q12(&mut rng, d)
            }
            1 => {
                // Stable filter with bandwidth expansion undone a bit (near instability).
                let d = if rng.range_i32(0, 1) == 0 { 10 } else { 16 };
                let mut a = rand_stable_a_q12(&mut rng, d);
                for v in &mut a {
                    *v = (*v as i32 * rng.range_i32(90, 130) / 100).clamp(-32768, 32767) as i16;
                }
                a
            }
            2 => {
                let d = rng.range_i32(1, 24) as usize;
                (0..d).map(|_| rng.range_i32(-2000, 2000) as i16).collect()
            }
            _ => {
                let d = rng.range_i32(1, 24) as usize;
                (0..d).map(|_| rng.i16()).collect()
            }
        };
        assert_eq!(
            silk_lpc_inverse_pred_gain(&a, a.len()),
            c::lpc_inverse_pred_gain(&a),
            "inv pred gain {a:?}"
        );
    }
}

#[test]
fn lpc_fit_matches_oracle() {
    let mut rng = Rng::new(30);
    for iter in 0..30_000 {
        let d = rng.range_i32(1, 24) as usize;
        let (qout, qin) = [(12, 17), (12, 24), (12, 16), (14, 17), (12, 13)][iter % 5];
        let bits = rng.range_i32(8, 31);
        let a0: Vec<i32> = (0..d)
            .map(|_| {
                if bits == 31 {
                    rng.next_u32() as i32
                } else {
                    rng.range_i32(-(1 << bits), 1 << bits)
                }
            })
            .collect();
        let mut ar = a0.clone();
        let mut ac = a0;
        let mut outr = vec![0i16; d];
        silk_lpc_fit(&mut outr, &mut ar, qout, qin, d);
        let outc = c::lpc_fit(&mut ac, qout, qin);
        assert_slice_eq("LPC_fit out", &outr, &outc);
        assert_slice_eq("LPC_fit in", &ar, &ac);
    }
}

#[test]
fn lpc_analysis_filter_matches_oracle() {
    let mut rng = Rng::new(31);
    for iter in 0..8000 {
        let d = 2 * rng.range_i32(3, 12) as usize;
        let len = rng.range_i32(d as i32, 400) as usize;
        let b: Vec<i16> = if iter % 2 == 0 && (d == 10 || d == 16) {
            rand_stable_a_q12(&mut rng, d)
        } else {
            (0..d).map(|_| rng.i16()).collect()
        };
        let x = rand_signal(&mut rng, len);
        let mut yr = vec![0i16; len];
        silk_lpc_analysis_filter(&mut yr, &x, &b, len, d);
        assert_slice_eq("LPC_analysis_filter", &yr, &c::lpc_analysis_filter(&x, &b));
    }
}

// ---------------------------------------------------------------------------------------------
// NLSF
// ---------------------------------------------------------------------------------------------

#[test]
fn nlsf2a_matches_oracle() {
    let mut rng = Rng::new(32);
    for iter in 0..30_000 {
        let d = if iter % 2 == 0 { 10 } else { 16 };
        let nlsf = match iter % 5 {
            0 => rand_nlsf(&mut rng, d, 0),
            1 => {
                // Clustered / degenerate sorted vectors.
                let base = rng.range_i32(0, 32767);
                let mut v: Vec<i16> = (0..d)
                    .map(|_| (base + rng.range_i32(-300, 300)).clamp(0, 32767) as i16)
                    .collect();
                v.sort_unstable();
                v
            }
            2 => vec![rng.range_i32(0, 32767) as i16; d],
            _ => rand_nlsf(&mut rng, d, 30),
        };
        let mut ar = vec![0i16; d];
        silk_nlsf2a(&mut ar, &nlsf, d);
        assert_slice_eq("NLSF2A", &ar, &c::nlsf2a(&nlsf));
    }
}

#[test]
fn a2nlsf_matches_oracle() {
    let mut rng = Rng::new(33);
    for iter in 0..20_000 {
        let d = if iter % 2 == 0 { 10 } else { 16 };
        let a0: Vec<i32> = match iter % 6 {
            0..=2 => {
                // Round trip of a stable filter (as the encoder produces).
                let a = rand_stable_a_q12(&mut rng, d);
                a.iter().map(|&v| (v as i32) << 4).collect()
            }
            3 => {
                let a = rand_stable_a_q12(&mut rng, d);
                a.iter()
                    .map(|&v| ((v as i32) << 4) + rng.range_i32(-8, 8))
                    .collect()
            }
            4 => (0..d).map(|_| rng.range_i32(-(1 << 15), 1 << 15)).collect(),
            _ => (0..d).map(|_| rng.range_i32(-(1 << 17), 1 << 17)).collect(),
        };
        let mut ar = a0.clone();
        let mut ac = a0;
        let mut nr = vec![0i16; d];
        silk_a2nlsf(&mut nr, &mut ar, d);
        let nc = c::a2nlsf(&mut ac);
        assert_slice_eq("A2NLSF nlsf", &nr, &nc);
        assert_slice_eq("A2NLSF a", &ar, &ac);
    }
}

fn cb(wb: bool) -> &'static SilkNlsfCbStruct {
    if wb {
        &SILK_NLSF_CB_WB
    } else {
        &SILK_NLSF_CB_NB_MB
    }
}

#[test]
fn nlsf_unpack_and_decode_match_oracle() {
    for wb in [false, true] {
        for i in 0..32 {
            let (ecc, pc) = c::nlsf_unpack(wb, i);
            let order = cb(wb).order as usize;
            let mut ecr = vec![0i16; order];
            let mut pr = vec![0u8; order];
            silk_nlsf_unpack(&mut ecr, &mut pr, cb(wb), i);
            assert_slice_eq("unpack ec_ix", &ecr, &ecc);
            assert_slice_eq("unpack pred", &pr, &pc);
        }
    }
    let mut rng = Rng::new(34);
    for iter in 0..40_000 {
        let wb = iter % 2 == 0;
        let order = cb(wb).order as usize;
        let mut idx = [0i8; 17];
        idx[0] = rng.range_i32(0, 31) as i8;
        let amp = if iter % 3 == 0 { 10 } else { 3 };
        for v in idx.iter_mut().skip(1).take(order) {
            *v = rng.range_i32(-amp, amp) as i8;
        }
        let mut nr = vec![0i16; order];
        silk_nlsf_decode(&mut nr, &idx, cb(wb));
        assert_slice_eq("NLSF_decode", &nr, &c::nlsf_decode(&idx, wb));
    }
}

#[test]
fn nlsf_stabilize_matches_oracle() {
    let mut rng = Rng::new(35);
    for iter in 0..40_000 {
        let (l, delta): (usize, Vec<i16>) = match iter % 3 {
            0 => (10, SILK_NLSF_CB_NB_MB.delta_min_q15.to_vec()),
            1 => (16, SILK_NLSF_CB_WB.delta_min_q15.to_vec()),
            _ => {
                let l = rng.range_i32(2, 20) as usize;
                let maxd = 32768 / (l as i32 + 1);
                (
                    l,
                    (0..=l)
                        .map(|_| rng.range_i32(1, maxd.min(700)) as i16)
                        .collect(),
                )
            }
        };
        let nlsf: Vec<i16> = match rng.range_i32(0, 4) {
            0 => (0..l).map(|_| rng.range_i32(0, 32767) as i16).collect(),
            1 => {
                let mut v = rand_nlsf(&mut rng, l, 0);
                for x in &mut v {
                    *x = (*x as i32 + rng.range_i32(-500, 500)).clamp(0, 32767) as i16;
                }
                v
            }
            2 => {
                // Heavy clustering (forces the fall-back path).
                let c0 = rng.range_i32(0, 32767);
                (0..l)
                    .map(|_| (c0 + rng.range_i32(-20, 20)).clamp(0, 32767) as i16)
                    .collect()
            }
            _ => rand_nlsf(&mut rng, l, 0),
        };
        let mut nr = nlsf.clone();
        let mut nc = nlsf;
        silk_nlsf_stabilize(&mut nr, &delta, l);
        c::nlsf_stabilize(&mut nc, &delta);
        assert_slice_eq("NLSF_stabilize", &nr, &nc);
    }
}

#[test]
fn nlsf_vq_weights_laroia_matches_oracle() {
    let mut rng = Rng::new(36);
    for iter in 0..40_000 {
        let d = [10, 16, 2, 4, 24][iter % 5];
        let nlsf: Vec<i16> = if iter % 7 == 0 {
            (0..d).map(|_| rng.range_i32(0, 32767) as i16).collect()
        } else {
            rand_nlsf(&mut rng, d, 0)
        };
        let mut wr = vec![0i16; d];
        silk_nlsf_vq_weights_laroia(&mut wr, &nlsf, d);
        assert_slice_eq("laroia", &wr, &c::nlsf_vq_weights_laroia(&nlsf));
    }
}

#[test]
fn nlsf_vq_matches_oracle() {
    let mut rng = Rng::new(37);
    for iter in 0..10_000 {
        let wb = iter % 2 == 0;
        let cbk = cb(wb);
        let order = cbk.order as usize;
        // Sorted NLSFs (as the encoder passes): keeps the int32 error sums of C in range.
        let input: Vec<i16> = rand_nlsf(&mut rng, order, 0);
        let k = cbk.n_vectors as usize;
        let mut er = vec![0i32; k];
        silk_nlsf_vq(&mut er, &input, cbk.cb1_nlsf_q8, cbk.cb1_wght_q9, k, order);
        let ec = c::nlsf_vq(&input, cbk.cb1_nlsf_q8, cbk.cb1_wght_q9, k);
        assert_slice_eq("NLSF_VQ", &er, &ec);

        // Random codebooks of other (even) orders.
        let order = 2 * rng.range_i32(1, 12) as usize;
        let k = rng.range_i32(1, 40) as usize;
        let rcb: Vec<u8> = (0..k * order).map(|_| rng.next_u32() as u8).collect();
        // Bound so that `order * 1.5 * 32767 * w` stays below 2^31 (C overflows otherwise).
        let wmax = (1_400_000_000 / (order as i64 * 49_151)) as i32;
        let rw: Vec<i16> = (0..k * order)
            .map(|_| rng.range_i32(0, wmax) as i16)
            .collect();
        let input: Vec<i16> = (0..order).map(|_| rng.range_i32(0, 32767) as i16).collect();
        let mut er = vec![0i32; k];
        silk_nlsf_vq(&mut er, &input, &rcb, &rw, k, order);
        assert_slice_eq("NLSF_VQ random cb", &er, &c::nlsf_vq(&input, &rcb, &rw, k));
    }
}

#[test]
fn nlsf_del_dec_quant_matches_oracle() {
    let mut rng = Rng::new(38);
    for iter in 0..20_000 {
        let wb = iter % 2 == 0;
        let cbk = cb(wb);
        let order = cbk.order as usize;
        let cb1 = rng.range_i32(0, 31);
        let mut ec_ix = vec![0i16; order];
        let mut pred = vec![0u8; order];
        silk_nlsf_unpack(&mut ec_ix, &mut pred, cbk, cb1);
        // (input amplitude, max weight) tiers that keep C's int32 RD sums from overflowing;
        // the last tier drives the indices into the clamped +-10 range.
        let (xamp, wmax) = [(300, 2048), (800, 512), (2500, 4)][iter % 3];
        let x: Vec<i16> = (0..order)
            .map(|_| rng.range_i32(-xamp, xamp) as i16)
            .collect();
        let w: Vec<i16> = (0..order).map(|_| rng.range_i32(1, wmax) as i16).collect();
        let mu = rng.range_i32(0, 32767);
        let mut ir = vec![0i8; order];
        let rdr = silk_nlsf_del_dec_quant(
            &mut ir,
            &x,
            &w,
            &pred,
            &ec_ix,
            cbk.ec_rates_q5,
            cbk.quant_step_size_q16 as i32,
            cbk.inv_quant_step_size_q6,
            mu,
            cbk.order,
        );
        let (ic, rdc) = c::nlsf_del_dec_quant(
            &x,
            &w,
            &pred,
            &ec_ix,
            cbk.ec_rates_q5,
            cbk.quant_step_size_q16 as i32,
            cbk.inv_quant_step_size_q6,
            mu,
        );
        assert_slice_eq("del_dec_quant indices", &ir, &ic);
        assert_eq!(rdr, rdc, "del_dec_quant RD");
    }
}

#[test]
fn nlsf_encode_matches_oracle() {
    let mut rng = Rng::new(39);
    for iter in 0..6000 {
        let wb = iter % 2 == 0;
        let cbk = cb(wb);
        let order = cbk.order as usize;
        // Realistic input: a (possibly perturbed) decoded codebook vector or a spread vector.
        // Realistic inputs (near a codebook vector, or a flat spectrum), as produced by the
        // encoder; arbitrary vectors far from every codebook entry overflow C's int32 RD sums.
        let mut nlsf = if iter % 4 != 0 {
            let mut idx = [0i8; 17];
            idx[0] = rng.range_i32(0, 31) as i8;
            for v in idx.iter_mut().skip(1).take(order) {
                *v = rng.range_i32(-4, 4) as i8;
            }
            let mut n = vec![0i16; order];
            silk_nlsf_decode(&mut n, &idx, cbk);
            let jit = [50, 200, 600][iter % 3];
            for v in &mut n {
                *v = (*v as i32 + rng.range_i32(-jit, jit)).clamp(0, 32767) as i16;
            }
            n
        } else {
            let step = 32768 / (order as i32 + 1);
            (0..order)
                .map(|i| (step * (i as i32 + 1) + rng.range_i32(-step / 4, step / 4)) as i16)
                .collect()
        };
        // Keep NLSFs >= 100 apart: tightly clustered NLSFs give Laroia weights large enough to
        // overflow C's int32 RD accumulation (undefined behaviour in C).
        let min_gap = vec![100i16; order + 1];
        silk_nlsf_stabilize(&mut nlsf, &min_gap, order);
        let mut w = vec![0i16; order];
        silk_nlsf_vq_weights_laroia(&mut w, &nlsf, order);
        let mu = rng.range_i32(1000, 6000);
        let n_surv = [2, 3, 4, 6, 8, 16, 32, 1][iter % 8];
        let signal_type = rng.range_i32(0, 2);

        let mut nr = nlsf.clone();
        let mut nc = nlsf;
        let mut ir = [0i8; 17];
        let rdr = silk_nlsf_encode(&mut ir, &mut nr, cbk, &w, mu, n_surv as usize, signal_type);
        let (ic, rdc) = c::nlsf_encode(&mut nc, wb, &w, mu, n_surv, signal_type);
        assert_slice_eq("NLSF_encode indices", &ir[..=order], &ic[..=order]);
        assert_slice_eq("NLSF_encode nlsf", &nr, &nc);
        assert_eq!(rdr, rdc, "NLSF_encode RD");
    }
}

// ---------------------------------------------------------------------------------------------
// Entropy coding helpers
// ---------------------------------------------------------------------------------------------

/// Random excitation with a mix of small and large magnitudes.
fn rand_pulses(rng: &mut Rng, n: usize) -> Vec<i8> {
    let kind = rng.range_i32(0, 5);
    (0..n)
        .map(|_| match kind {
            0 => 0,
            1 => rng.range_i32(-1, 1) as i8,
            2 => {
                if rng.range_i32(0, 4) == 0 {
                    rng.range_i32(-5, 5) as i8
                } else {
                    0
                }
            }
            3 => rng.range_i32(-127, 127) as i8,
            4 => rng.range_i32(-30, 30) as i8,
            _ => rng.range_i32(-128, 127) as i8,
        })
        .collect()
}

#[test]
fn signs_match_oracle() {
    let mut rng = Rng::new(40);
    for iter in 0..15_000 {
        let length = frame_lengths()[iter % 6];
        let blocks = ((length + 8) >> 4) as usize;
        let pulses = rand_pulses(&mut rng, blocks * 16);
        let sum_pulses: Vec<i32> = (0..blocks)
            .map(|b| {
                let s: i32 = pulses[b * 16..(b + 1) * 16]
                    .iter()
                    .map(|&p| (p as i32).abs())
                    .sum();
                if rng.range_i32(0, 5) == 0 {
                    s | (rng.range_i32(1, 10) << 5)
                } else {
                    s.min(16)
                }
            })
            .collect();
        let st = rng.range_i32(0, 2);
        let qo = rng.range_i32(0, 1);
        let (pre, pre_n) = rand_pre(&mut rng);
        let size = if iter % 17 == 0 { 8 } else { 1275 };
        let ce = c::encode_signs(size, pre, pre_n, &pulses, length, st, qo, &sum_pulses);
        let re = rust_enc(size, pre, pre_n, |e| {
            silk_encode_signs(e, &pulses, length, st, qo, &sum_pulses)
        });
        assert_eq!(re, ce, "encode_signs iter {iter}");

        // Decode on the C stream: magnitudes in, signed pulses out.
        let mags: Vec<i16> = pulses.iter().map(|&p| (p as i32).abs() as i16).collect();
        let mut dc = mags.clone();
        let rc = c::decode_signs(&ce.buf, pre_n, &mut dc, length, st, qo, &sum_pulses);
        let mut dr = mags;
        let (_, rr) = rust_dec(&ce.buf, pre_n, |d| {
            silk_decode_signs(d, &mut dr, length, st, qo, &sum_pulses)
        });
        assert_slice_eq("decode_signs", &dr, &dc);
        assert_eq!(rr, rc, "decode_signs state iter {iter}");
        if ce.res[2] == 0 {
            let want: Vec<i16> = pulses.iter().map(|&p| p as i16).collect();
            // Only blocks with a positive sum carry signs.
            for b in 0..blocks {
                if sum_pulses[b] > 0 {
                    assert_slice_eq(
                        "signs roundtrip",
                        &dr[b * 16..(b + 1) * 16],
                        &want[b * 16..(b + 1) * 16],
                    );
                }
            }
        }

        // Garbage input.
        let mut data = vec![0u8; rng.range_i32(0, 40) as usize];
        rng.fill_bytes(&mut data);
        let mags: Vec<i16> = (0..blocks * 16)
            .map(|_| rng.range_i32(0, 3) as i16)
            .collect();
        let mut dc = mags.clone();
        let rc = c::decode_signs(&data, pre_n, &mut dc, length, st, qo, &sum_pulses);
        let mut dr = mags;
        let (_, rr) = rust_dec(&data, pre_n, |d| {
            silk_decode_signs(d, &mut dr, length, st, qo, &sum_pulses)
        });
        assert_slice_eq("decode_signs garbage", &dr, &dc);
        assert_eq!(rr, rc);
    }
}

/// 16 non-negative pulses satisfying the shell coder limits (8/10/12/16 per level).
fn rand_shell_block(rng: &mut Rng) -> [i32; 16] {
    loop {
        let total = rng.range_i32(0, 16);
        let mut p = [0i32; 16];
        for _ in 0..total {
            p[rng.range_i32(0, 15) as usize] += 1;
        }
        let ok = (0..8).all(|k| p[2 * k] + p[2 * k + 1] <= 8)
            && (0..4).all(|k| p[4 * k..4 * k + 4].iter().sum::<i32>() <= 10)
            && (0..2).all(|k| p[8 * k..8 * k + 8].iter().sum::<i32>() <= 12);
        if ok {
            return p;
        }
    }
}

#[test]
fn shell_coder_matches_oracle() {
    let mut rng = Rng::new(41);
    for iter in 0..30_000 {
        let p = rand_shell_block(&mut rng);
        let (pre, pre_n) = rand_pre(&mut rng);
        let size = if iter % 13 == 0 { 3 } else { 64 };
        let ce = c::shell_encoder(size, pre, pre_n, &p);
        let re = rust_enc(size, pre, pre_n, |e| silk_shell_encoder(e, &p));
        assert_eq!(re, ce, "shell_encoder iter {iter}");

        let sum: i32 = p.iter().sum();
        let (oc, rc) = c::shell_decoder(&ce.buf, pre_n, sum);
        let mut or = [0i16; 16];
        let (_, rr) = rust_dec(&ce.buf, pre_n, |d| silk_shell_decoder(&mut or, d, sum));
        assert_slice_eq("shell_decoder", &or, &oc);
        assert_eq!(rr, rc);
        if ce.res[2] == 0 {
            let want: Vec<i16> = p.iter().map(|&v| v as i16).collect();
            assert_slice_eq("shell roundtrip", &or, &want);
        }

        // Garbage.
        let mut data = vec![0u8; rng.range_i32(0, 20) as usize];
        rng.fill_bytes(&mut data);
        let s = rng.range_i32(0, 16);
        let (oc, rc) = c::shell_decoder(&data, pre_n, s);
        let (_, rr) = rust_dec(&data, pre_n, |d| silk_shell_decoder(&mut or, d, s));
        assert_slice_eq("shell_decoder garbage", &or, &oc);
        assert_eq!(rr, rc);
    }
}

#[test]
fn pulses_match_oracle() {
    let mut rng = Rng::new(42);
    for iter in 0..8000 {
        let fl = frame_lengths()[iter % 6];
        let padded = ((fl + 15) & !15) as usize;
        // Encoder-side buffer is `pulses[MAX_FRAME_LENGTH]` in C; for 10 ms @ 12 kHz C zeroes
        // `pulses[frame_length..frame_length + 16]`, i.e. past the padded length.
        let mut pulses = rand_pulses(&mut rng, 320);
        if iter % 11 == 0 {
            pulses[rng.range_i32(0, fl - 1) as usize] = -128;
        }
        let st = rng.range_i32(0, 2);
        let qo = rng.range_i32(0, 1);
        let (pre, pre_n) = rand_pre(&mut rng);
        let size = if iter % 19 == 0 {
            rng.range_i32(1, 60) as usize
        } else {
            1500
        };

        let mut pc = pulses.clone();
        let ce = c::encode_pulses(size, pre, pre_n, st, qo, &mut pc, fl);
        let mut pr = pulses.clone();
        let re = rust_enc(size, pre, pre_n, |e| {
            silk_encode_pulses(e, st, qo, &mut pr, fl)
        });
        assert_eq!(re, ce, "encode_pulses iter {iter} fl {fl}");
        assert_slice_eq("encode_pulses pulses", &pr, &pc);

        let (dc, rc) = c::decode_pulses(&ce.buf, pre_n, st, qo, fl);
        let mut dr = vec![0i16; padded];
        let (_, rr) = rust_dec(&ce.buf, pre_n, |d| {
            silk_decode_pulses(d, &mut dr, st, qo, fl)
        });
        assert_slice_eq("decode_pulses", &dr, &dc);
        assert_eq!(rr, rc, "decode_pulses state iter {iter}");
        if ce.res[2] == 0 && !pulses[..fl as usize].contains(&-128) {
            let want: Vec<i16> = pc[..fl as usize].iter().map(|&v| v as i16).collect();
            assert_slice_eq("pulses roundtrip", &dr[..fl as usize], &want);
        }

        // Garbage.
        let mut data = vec![0u8; rng.range_i32(0, 200) as usize];
        rng.fill_bytes(&mut data);
        let (dc, rc) = c::decode_pulses(&data, pre_n, st, qo, fl);
        let (_, rr) = rust_dec(&data, pre_n, |d| silk_decode_pulses(d, &mut dr, st, qo, fl));
        assert_slice_eq("decode_pulses garbage", &dr, &dc);
        assert_eq!(rr, rc);
    }
}

#[test]
fn stereo_pred_matches_oracle() {
    let mut rng = Rng::new(43);
    for iter in 0..30_000 {
        let mut ix = [[0i8; 3]; 2];
        for row in &mut ix {
            row[0] = rng.range_i32(0, 2) as i8;
            row[1] = rng.range_i32(0, 4) as i8;
            row[2] = rng.range_i32(0, 4) as i8;
        }
        let mid = rng.range_i32(-1, 1);
        let (pre, pre_n) = rand_pre(&mut rng);
        let size = if iter % 23 == 0 { 2 } else { 32 };
        let ce = c::stereo_encode(size, pre, pre_n, &ix, mid);
        let re = rust_enc(size, pre, pre_n, |e| {
            silk_stereo_encode_pred(e, &ix);
            if mid >= 0 {
                silk_stereo_encode_mid_only(e, mid as i8);
            }
        });
        assert_eq!(re, ce, "stereo_encode iter {iter}");

        let (pc, mc, rc) = c::stereo_decode(&ce.buf, pre_n, mid >= 0);
        let ((pr, mr), rr) = rust_dec(&ce.buf, pre_n, |d| {
            let mut p = [0i32; 2];
            silk_stereo_decode_pred(d, &mut p);
            let m = if mid >= 0 {
                silk_stereo_decode_mid_only(d)
            } else {
                0
            };
            (p, m)
        });
        assert_eq!((pr, mr, rr), (pc, mc, rc), "stereo_decode iter {iter}");
        if ce.res[2] == 0 && mid >= 0 {
            assert_eq!(mr, mid);
        }

        let mut data = vec![0u8; rng.range_i32(0, 8) as usize];
        rng.fill_bytes(&mut data);
        let (pc, mc, rc) = c::stereo_decode(&data, pre_n, true);
        let ((pr, mr), rr) = rust_dec(&data, pre_n, |d| {
            let mut p = [0i32; 2];
            silk_stereo_decode_pred(d, &mut p);
            (p, silk_stereo_decode_mid_only(d))
        });
        assert_eq!((pr, mr, rr), (pc, mc, rc), "stereo_decode garbage");
    }
}

#[test]
fn gains_match_oracle() {
    let mut rng = Rng::new(44);
    for iter in 0..60_000 {
        let nb_subfr = if rng.range_i32(0, 1) == 0 { 2 } else { 4 };
        let conditional = rng.range_i32(0, 1);
        let mut gains = [0i32; 4];
        for g in &mut gains {
            *g = match rng.range_i32(0, 5) {
                0 => rng.range_i32(0, 1 << 16),
                1 => rng.range_i32(0, i32::MAX),
                2 => rng.next_u32() as i32 >> rng.range_i32(0, 31),
                3 => [0, 1, i32::MAX, 65536, -1, i32::MIN][rng.range_i32(0, 5) as usize],
                _ => rng.range_i32(1 << 10, 1 << 24),
            };
        }
        let prev0 = if iter % 10 == 0 {
            rng.range_i32(-128, 127)
        } else {
            rng.range_i32(0, 63)
        } as i8;

        let mut ind_r = [0i8; 4];
        let mut g_r = gains;
        let mut p_r = prev0;
        silk_gains_quant(&mut ind_r, &mut g_r, &mut p_r, conditional, nb_subfr);
        let mut ind_c = [0i8; 4];
        let mut g_c = gains;
        let mut p_c = prev0;
        c::gains_quant(&mut ind_c, &mut g_c, &mut p_c, conditional, nb_subfr as i32);
        assert_eq!(
            (ind_r, g_r, p_r),
            (ind_c, g_c, p_c),
            "gains_quant {gains:?} prev {prev0}"
        );

        // Dequantize what was quantized, and random indices.
        let ind = if iter % 2 == 0 {
            ind_c
        } else {
            let mut v = [0i8; 4];
            for (k, x) in v.iter_mut().enumerate() {
                *x = if k == 0 && conditional == 0 {
                    rng.range_i32(0, 63)
                } else {
                    rng.range_i32(0, 40)
                } as i8;
            }
            v
        };
        let mut g_r = [0i32; 4];
        let mut p_r = prev0;
        silk_gains_dequant(&mut g_r, &ind, &mut p_r, conditional, nb_subfr);
        let mut g_c = [0i32; 4];
        let mut p_c = prev0;
        c::gains_dequant(&mut g_c, &ind, &mut p_c, conditional, nb_subfr as i32);
        assert_eq!((g_r, p_r), (g_c, p_c), "gains_dequant {ind:?} prev {prev0}");

        assert_eq!(
            silk_gains_id(&ind, nb_subfr),
            c::gains_id(&ind, nb_subfr as i32)
        );
    }
}

#[test]
fn decode_pitch_matches_oracle() {
    for fs in [8, 12, 16] {
        for nb_subfr in [2, 4] {
            let cbk = match (fs, nb_subfr) {
                (8, 4) => PE_NB_CBKS_STAGE2_EXT,
                (8, _) => PE_NB_CBKS_STAGE2_10MS,
                (_, 4) => PE_NB_CBKS_STAGE3_MAX,
                _ => PE_NB_CBKS_STAGE3_10MS,
            };
            for lag in 0..(PE_MAX_LAG_MS - PE_MIN_LAG_MS) * fs + 8 {
                for contour in 0..cbk {
                    let mut lr = [0i32; 4];
                    silk_decode_pitch(lag as i16, contour as i8, &mut lr, fs, nb_subfr);
                    let lc = c::decode_pitch(lag as i16, contour as i8, fs, nb_subfr);
                    assert_slice_eq("decode_pitch", &lr[..nb_subfr as usize], &lc);
                }
            }
        }
    }
}
