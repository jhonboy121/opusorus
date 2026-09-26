//! Port of silk/VAD.c: voice activity detection.

#![allow(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]

use crate::silk::define::{
    MAX_FRAME_LENGTH, VAD_INTERNAL_SUBFRAMES, VAD_INTERNAL_SUBFRAMES_LOG2, VAD_N_BANDS,
    VAD_NEGATIVE_OFFSET_Q5, VAD_NOISE_LEVEL_SMOOTH_COEF_Q16, VAD_NOISE_LEVELS_BIAS,
    VAD_SNR_FACTOR_Q16, VAD_SNR_SMOOTH_COEF_Q18,
};
use crate::silk::macros::{
    SILK_INT16_MAX, SILK_INT32_MAX, SILK_UINT8_MAX, silk_add_pos_sat32, silk_div32, silk_div32_16,
    silk_lshift, silk_lshift32, silk_max_32, silk_max_int, silk_min, silk_min_int, silk_mul,
    silk_rshift, silk_rshift32, silk_smlabb, silk_smlawb, silk_smulwb, silk_smulww,
    silk_sqrt_approx,
};
use crate::silk::sigproc::{silk_ana_filt_bank_1, silk_lin2log, silk_sigm_q15};
use crate::silk::structs::{SilkEncoderState, SilkVadState};

const NB: usize = VAD_N_BANDS as usize;
const MFL: usize = MAX_FRAME_LENGTH as usize;

/// Port of silk/VAD.c:silk_VAD_Init — initialization of the Silk VAD. Returns 0 (success).
pub fn silk_vad_init(ps_silk_vad: &mut SilkVadState) -> i32 {
    let ret = 0;

    // reset state memory
    *ps_silk_vad = SilkVadState::default();

    // init noise levels
    // Initialize array with approx pink noise levels (psd proportional to inverse of frequency)
    for b in 0..NB {
        ps_silk_vad.noise_level_bias[b] =
            silk_max_32(silk_div32_16(VAD_NOISE_LEVELS_BIAS, b as i32 + 1), 1);
    }

    // Initialize state
    for b in 0..NB {
        ps_silk_vad.nl[b] = silk_mul(100, ps_silk_vad.noise_level_bias[b]);
        ps_silk_vad.inv_nl[b] = silk_div32(SILK_INT32_MAX, ps_silk_vad.nl[b]);
    }
    ps_silk_vad.counter = 15;

    // init smoothed energy-to-noise ratio
    for b in 0..NB {
        ps_silk_vad.nrg_ratio_smth_q8[b] = 100 * 256; // 100 * 256 --> 20 dB SNR
    }

    ret
}

/// Weighting factors for tilt measure (`tiltWeights`).
static TILT_WEIGHTS: [i32; NB] = [30000, 6000, -12000, -12000];

/// Port of silk/VAD.c:silk_VAD_GetSA_Q8_c — get the speech activity level in Q8.
///
/// Updates `ps_enc_c.s_vad`, `speech_activity_q8`, `input_tilt_q15` and
/// `input_quality_bands_q15`. `p_in` holds `frame_length` samples. Returns 0 (success).
/// For the C call with `pIn = psEncC->inputBuf + 1` use [`silk_vad_get_sa_q8_input_buf`].
pub fn silk_vad_get_sa_q8_c(ps_enc_c: &mut SilkEncoderState, p_in: &[i16]) -> i32 {
    let SilkEncoderState {
        s_vad,
        frame_length,
        fs_khz,
        input_tilt_q15,
        speech_activity_q8,
        input_quality_bands_q15,
        ..
    } = ps_enc_c;
    vad_get_sa_q8(
        s_vad,
        *frame_length,
        *fs_khz,
        p_in,
        input_tilt_q15,
        speech_activity_q8,
        input_quality_bands_q15,
    )
}

/// [`silk_vad_get_sa_q8_c`] on `&ps_enc_c.input_buf[ 1.. ]` (the C call site
/// `silk_VAD_GetSA_Q8( &psEnc->sCmn, psEnc->sCmn.inputBuf + 1 )`).
pub fn silk_vad_get_sa_q8_input_buf(ps_enc_c: &mut SilkEncoderState) -> i32 {
    let SilkEncoderState {
        s_vad,
        frame_length,
        fs_khz,
        input_tilt_q15,
        speech_activity_q8,
        input_quality_bands_q15,
        input_buf,
        ..
    } = ps_enc_c;
    vad_get_sa_q8(
        s_vad,
        *frame_length,
        *fs_khz,
        &input_buf[1..],
        input_tilt_q15,
        speech_activity_q8,
        input_quality_bands_q15,
    )
}

/// Body of `silk_VAD_GetSA_Q8_c` on split borrows of the encoder state.
fn vad_get_sa_q8(
    ps_silk_vad: &mut SilkVadState,
    frame_length: i32,
    fs_khz: i32,
    p_in: &[i16],
    input_tilt_q15: &mut i32,
    speech_activity_q8: &mut i32,
    input_quality_bands_q15: &mut [i32; NB],
) -> i32 {
    let ret = 0;
    let mut xnrg = [0i32; NB];
    let mut nrg_to_noise_ratio_q8 = [0i32; NB];
    let mut x_offset = [0usize; NB];

    // Safety checks
    const { assert!(VAD_N_BANDS == 4) };
    celt_assert!(MAX_FRAME_LENGTH >= frame_length);
    celt_assert!(frame_length <= 512);
    celt_assert!(frame_length == 8 * silk_rshift(frame_length, 3));

    // Filter and Decimate
    let decimated_framelength1 = silk_rshift(frame_length, 1) as usize;
    let decimated_framelength2 = silk_rshift(frame_length, 2) as usize;
    let mut decimated_framelength = silk_rshift(frame_length, 3) as usize;
    // Decimate into 4 bands:
    //  0       L      3L       L              3L                             5L
    //          -      --       -              --                             --
    //          8       8       2               4                              4
    //
    //  [0-1 kHz| temp. |1-2 kHz|    2-4 kHz    |            4-8 kHz           |
    //
    // They're arranged to allow the minimal ( frame_length / 4 ) extra
    // scratch space during the downsampling process
    x_offset[0] = 0;
    x_offset[1] = decimated_framelength + decimated_framelength2;
    x_offset[2] = x_offset[1] + decimated_framelength;
    x_offset[3] = x_offset[2] + decimated_framelength2;
    // C: ALLOC( X, X_offset[ 3 ] + decimated_framelength1 )
    let mut x_buf = [0i16; MFL + MFL / 4];
    let x = &mut x_buf[..x_offset[3] + decimated_framelength1];

    // 0-8 kHz to 0-4 kHz and 4-8 kHz
    {
        let (lo, hi) = x.split_at_mut(x_offset[3]);
        silk_ana_filt_bank_1(
            p_in,
            &mut ps_silk_vad.ana_state,
            lo,
            hi,
            frame_length as usize,
        );
    }

    // 0-4 kHz to 0-2 kHz and 2-4 kHz
    // (C filters X in place: outL = X overlaps the input; every input sample is read before
    // its slot is overwritten, so filtering a copy of the input is equivalent.)
    {
        let mut tmp = [0i16; MFL / 2];
        tmp[..decimated_framelength1].copy_from_slice(&x[..decimated_framelength1]);
        let (lo, hi) = x.split_at_mut(x_offset[2]);
        silk_ana_filt_bank_1(
            &tmp[..decimated_framelength1],
            &mut ps_silk_vad.ana_state1,
            lo,
            hi,
            decimated_framelength1,
        );
    }

    // 0-2 kHz to 0-1 kHz and 1-2 kHz
    {
        let mut tmp = [0i16; MFL / 4];
        tmp[..decimated_framelength2].copy_from_slice(&x[..decimated_framelength2]);
        let (lo, hi) = x.split_at_mut(x_offset[1]);
        silk_ana_filt_bank_1(
            &tmp[..decimated_framelength2],
            &mut ps_silk_vad.ana_state2,
            lo,
            hi,
            decimated_framelength2,
        );
    }

    // HP filter on lowest band (differentiator)
    x[decimated_framelength - 1] >>= 1;
    let hp_state_tmp = x[decimated_framelength - 1];
    for i in (1..decimated_framelength).rev() {
        x[i - 1] >>= 1;
        // C: X[ i ] -= X[ i - 1 ] on opus_int16 (int arithmetic, truncated on store)
        x[i] = (x[i] as i32 - x[i - 1] as i32) as i16;
    }
    x[0] = (x[0] as i32 - ps_silk_vad.hp_state as i32) as i16;
    ps_silk_vad.hp_state = hp_state_tmp;

    // Calculate the energy in each band
    for b in 0..NB {
        // Find the decimated framelength in the non-uniformly divided bands
        decimated_framelength = silk_rshift(
            frame_length,
            silk_min_int(VAD_N_BANDS - b as i32, VAD_N_BANDS - 1),
        ) as usize;

        // Split length into subframe lengths
        let dec_subframe_length =
            silk_rshift(decimated_framelength as i32, VAD_INTERNAL_SUBFRAMES_LOG2) as usize;
        let mut dec_subframe_offset = 0;

        // Compute energy per sub-frame
        // initialize with summed energy of last subframe
        xnrg[b] = ps_silk_vad.xnrg_subfr[b];
        let mut sum_squared = 0;
        for s in 0..VAD_INTERNAL_SUBFRAMES {
            sum_squared = 0;
            for i in 0..dec_subframe_length {
                // The energy will be less than dec_subframe_length * ( silk_int16_MIN / 8 ) ^ 2.
                // Therefore we can accumulate with no risk of overflow (unless
                // dec_subframe_length > 128)
                let x_tmp = silk_rshift(x[x_offset[b] + i + dec_subframe_offset] as i32, 3);
                sum_squared = silk_smlabb(sum_squared, x_tmp, x_tmp);

                // Safety check
                silk_assert!(sum_squared >= 0);
            }

            // Add/saturate summed energy of current subframe
            if s < VAD_INTERNAL_SUBFRAMES - 1 {
                xnrg[b] = silk_add_pos_sat32(xnrg[b], sum_squared);
            } else {
                // Look-ahead subframe
                xnrg[b] = silk_add_pos_sat32(xnrg[b], silk_rshift(sum_squared, 1));
            }

            dec_subframe_offset += dec_subframe_length;
        }
        ps_silk_vad.xnrg_subfr[b] = sum_squared;
    }

    // Noise estimation
    silk_vad_get_noise_levels(&xnrg, ps_silk_vad);

    // Signal-plus-noise to noise ratio estimation
    let mut sum_squared: i32 = 0;
    let mut input_tilt: i32 = 0;
    for b in 0..NB {
        let speech_nrg = xnrg[b] - ps_silk_vad.nl[b];
        if speech_nrg > 0 {
            // Divide, with sufficient resolution
            if (xnrg[b] as u32 & 0xFF80_0000) == 0 {
                nrg_to_noise_ratio_q8[b] =
                    silk_div32(silk_lshift(xnrg[b], 8), ps_silk_vad.nl[b] + 1);
            } else {
                nrg_to_noise_ratio_q8[b] =
                    silk_div32(xnrg[b], silk_rshift(ps_silk_vad.nl[b], 8) + 1);
            }

            // Convert to log domain
            let mut snr_q7 = silk_lin2log(nrg_to_noise_ratio_q8[b]) - 8 * 128;

            // Sum-of-squares
            sum_squared = silk_smlabb(sum_squared, snr_q7, snr_q7); // Q14

            // Tilt measure
            if speech_nrg < (1i32 << 20) {
                // Scale down SNR value for small subband speech energies
                snr_q7 = silk_smulwb(silk_lshift(silk_sqrt_approx(speech_nrg), 6), snr_q7);
            }
            input_tilt = silk_smlawb(input_tilt, TILT_WEIGHTS[b], snr_q7);
        } else {
            nrg_to_noise_ratio_q8[b] = 256;
        }
    }

    // Mean-of-squares
    sum_squared = silk_div32_16(sum_squared, VAD_N_BANDS); // Q14

    // Root-mean-square approximation, scale to dBs, and write to output pointer
    let p_snr_db_q7 = (3 * silk_sqrt_approx(sum_squared)) as i16 as i32; // Q7

    // Speech Probability Estimation
    let mut sa_q15 =
        silk_sigm_q15(silk_smulwb(VAD_SNR_FACTOR_Q16, p_snr_db_q7) - VAD_NEGATIVE_OFFSET_Q5);

    // Frequency Tilt Measure
    *input_tilt_q15 = silk_lshift(silk_sigm_q15(input_tilt) - 16384, 1);

    // Scale the sigmoid output based on power levels
    let mut speech_nrg: i32 = 0;
    for b in 0..NB {
        // Accumulate signal-without-noise energies, higher frequency bands have more weight
        speech_nrg += (b as i32 + 1) * silk_rshift(xnrg[b] - ps_silk_vad.nl[b], 4);
    }

    if frame_length == 20 * fs_khz {
        speech_nrg = silk_rshift32(speech_nrg, 1);
    }
    // Power scaling
    if speech_nrg <= 0 {
        sa_q15 = silk_rshift(sa_q15, 1);
    } else if speech_nrg < 16384 {
        speech_nrg = silk_lshift32(speech_nrg, 16);

        // square-root
        speech_nrg = silk_sqrt_approx(speech_nrg);
        sa_q15 = silk_smulwb(32768 + speech_nrg, sa_q15);
    }

    // Copy the resulting speech activity in Q8
    *speech_activity_q8 = silk_min_int(silk_rshift(sa_q15, 7), SILK_UINT8_MAX);

    // Energy Level and SNR estimation
    // Smoothing coefficient
    let mut smooth_coef_q16 = silk_smulwb(VAD_SNR_SMOOTH_COEF_Q18, silk_smulwb(sa_q15, sa_q15));

    if frame_length == 10 * fs_khz {
        smooth_coef_q16 >>= 1;
    }

    for b in 0..NB {
        // compute smoothed energy-to-noise ratio per band
        ps_silk_vad.nrg_ratio_smth_q8[b] = silk_smlawb(
            ps_silk_vad.nrg_ratio_smth_q8[b],
            nrg_to_noise_ratio_q8[b] - ps_silk_vad.nrg_ratio_smth_q8[b],
            smooth_coef_q16,
        );

        // signal to noise ratio in dB per band
        let snr_q7 = 3 * (silk_lin2log(ps_silk_vad.nrg_ratio_smth_q8[b]) - 8 * 128);
        // quality = sigmoid( 0.25 * ( SNR_dB - 16 ) );
        input_quality_bands_q15[b] = silk_sigm_q15(silk_rshift(snr_q7 - 16 * 128, 4));
    }

    ret
}

/// Port of silk/VAD.c:silk_VAD_GetNoiseLevels (static) — noise level estimation.
pub fn silk_vad_get_noise_levels(p_x: &[i32; NB], ps_silk_vad: &mut SilkVadState) {
    // Initially faster smoothing
    let min_coef = if ps_silk_vad.counter < 1000 {
        // 1000 = 20 sec
        let c = silk_div32_16(SILK_INT16_MAX, silk_rshift(ps_silk_vad.counter, 4) + 1);
        // Increment frame counter
        ps_silk_vad.counter += 1;
        c
    } else {
        0
    };

    for k in 0..NB {
        // Get old noise level estimate for current band
        let mut nl = ps_silk_vad.nl[k];
        silk_assert!(nl >= 0);

        // Add bias
        let nrg = silk_add_pos_sat32(p_x[k], ps_silk_vad.noise_level_bias[k]);
        silk_assert!(nrg > 0);

        // Invert energies
        let inv_nrg = silk_div32(SILK_INT32_MAX, nrg);
        silk_assert!(inv_nrg >= 0);

        // Less update when subband energy is high
        let mut coef = if nrg > silk_lshift(nl, 3) {
            VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 >> 3
        } else if nrg < nl {
            VAD_NOISE_LEVEL_SMOOTH_COEF_Q16
        } else {
            silk_smulwb(
                silk_smulww(inv_nrg, nl),
                VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 << 1,
            )
        };

        // Initially faster smoothing
        coef = silk_max_int(coef, min_coef);

        // Smooth inverse energies
        ps_silk_vad.inv_nl[k] =
            silk_smlawb(ps_silk_vad.inv_nl[k], inv_nrg - ps_silk_vad.inv_nl[k], coef);
        silk_assert!(ps_silk_vad.inv_nl[k] >= 0);

        // Compute noise level by inverting again
        nl = silk_div32(SILK_INT32_MAX, ps_silk_vad.inv_nl[k]);
        silk_assert!(nl >= 0);

        // Limit noise levels (guarantee 7 bits of head room)
        nl = silk_min(nl, 0x00FF_FFFF);

        // Store as part of state
        ps_silk_vad.nl[k] = nl;
    }
}
