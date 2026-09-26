//! Port of dnn/osce_features.c and osce_features.h: the LACE/NoLACE input features computed
//! from the SILK decoder parameters and output (`osce_calculate_features`), the BBWENet
//! features (`osce_bwe_calculate_features`, `ENABLE_OSCE_BWE`) and the cross-fades.
//!
//! `osce_calculate_features` reads a few `silk_decoder_state` fields; they are passed as an
//! [`OsceDecInfo`] (built with [`OsceDecInfo::from_decoder`]) next to the OSCE feature state, so
//! the OSCE state can live inside the decoder state without borrow conflicts. The debug dumps
//! (`WRITE_FEATURES`, `DEBUG_PRINT`) are not ported.

use crate::celt::static_modes::KissFftCpx;
use crate::math;
use crate::silk::define::TYPE_VOICED;
use crate::silk::structs::{SilkDecoderControl, SilkDecoderState};

use super::freq::{dct, forward_transform};
use super::osce::{
    OSCE_ACORR_START, OSCE_BWE_FEATURE_DIM, OSCE_BWE_HALF_WINDOW_SIZE, OSCE_BWE_MAX_INSTAFREQ_BIN,
    OSCE_BWE_NUM_BANDS, OSCE_BWE_WINDOW_SIZE, OSCE_CLEAN_SPEC_LENGTH, OSCE_CLEAN_SPEC_NUM_BANDS,
    OSCE_CLEAN_SPEC_START, OSCE_FEATURE_DIM, OSCE_FEATURES_MAX_HISTORY, OSCE_LOG_GAIN_START,
    OSCE_LTP_LENGTH, OSCE_LTP_START, OSCE_MAX_FEATURE_FRAMES, OSCE_NO_PITCH_VALUE,
    OSCE_NOISY_CEPSTRUM_LENGTH, OSCE_NOISY_CEPSTRUM_START, OSCE_NOISY_SPEC_NUM_BANDS,
    OSCE_PITCH_HANGOVER, OsceBweFeatureState, OsceFeatureState,
};

/// `OSCE_SPEC_WINDOW_SIZE`.
pub const OSCE_SPEC_WINDOW_SIZE: usize = 320;
/// `OSCE_SPEC_NUM_FREQS`.
pub const OSCE_SPEC_NUM_FREQS: usize = 161;

/// `LTP_ORDER` (silk/define.h) as an index.
const LTP_ORDER: usize = crate::silk::define::LTP_ORDER as usize;

/// `OSCE_HANGOVER_BUGFIX` is not defined upstream (`TESTBIT` 0): the pitch hangover is off.
const OSCE_HANGOVER_BUGFIX: bool = false;

/// The `silk_decoder_state` fields read by [`osce_calculate_features`] and
/// [`osce_enhance_frame`](super::osce::osce_enhance_frame).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OsceDecInfo {
    /// `fs_kHz`.
    pub fs_khz: i32,
    /// `nb_subfr`.
    pub nb_subfr: i32,
    /// `LPC_order`.
    pub lpc_order: i32,
    /// `indices.signalType`.
    pub signal_type: i32,
}

impl OsceDecInfo {
    /// Reads the fields from a SILK decoder channel state.
    #[must_use]
    pub const fn from_decoder(ps_dec: &SilkDecoderState) -> Self {
        Self {
            fs_khz: ps_dec.fs_khz,
            nb_subfr: ps_dec.nb_subfr,
            lpc_order: ps_dec.lpc_order,
            signal_type: ps_dec.indices.signal_type as i32,
        }
    }
}

/// `center_bins_clean`.
const CENTER_BINS_CLEAN: [usize; 64] = [
    0, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 33, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60,
    62, 65, 68, 70, 73, 75, 78, 80, 82, 85, 88, 90, 92, 95, 98, 100, 102, 105, 108, 110, 112, 115,
    118, 120, 122, 125, 128, 130, 132, 135, 138, 140, 142, 145, 148, 150, 152, 155, 160,
];

/// `center_bins_noisy`.
const CENTER_BINS_NOISY: [usize; 18] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160,
];

/// `center_bins_bwe`.
const CENTER_BINS_BWE: [usize; 32] = [
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110,
    115, 120, 125, 130, 135, 140, 145, 150, 160,
];

/// `band_weights_clean`.
#[rustfmt::skip]
const BAND_WEIGHTS_CLEAN: [f32; 64] = [
    0.666666666667, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.400000000000, 0.400000000000, 0.400000000000, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.400000000000, 0.400000000000, 0.400000000000, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.250000000000, 0.333333333333,
];

/// `band_weights_noisy`.
#[rustfmt::skip]
const BAND_WEIGHTS_NOISY: [f32; 18] = [
    0.400000000000, 0.250000000000, 0.250000000000, 0.250000000000,
    0.250000000000, 0.250000000000, 0.250000000000, 0.250000000000,
    0.166666666667, 0.125000000000, 0.125000000000, 0.125000000000,
    0.083333333333, 0.062500000000, 0.062500000000, 0.050000000000,
    0.041666666667, 0.080000000000,
];

/// `band_weights_bwe` (double literals in a float array: converted through f64 like C).
#[rustfmt::skip]
const BAND_WEIGHTS_BWE: [f32; 32] = [
    0.333333333_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32,
    0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32,
    0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32,
    0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32,
    0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32,
    0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32,
    0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32,
    0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.133333333_f64 as f32, 0.181818182_f64 as f32,
];

/// `osce_window` (sine window, 320 taps).
#[rustfmt::skip]
pub const OSCE_WINDOW: [f32; OSCE_SPEC_WINDOW_SIZE] = [
    0.004908718808, 0.014725683311, 0.024541228523, 0.034354408400, 0.044164277127,
    0.053969889210, 0.063770299562, 0.073564563600, 0.083351737332, 0.093130877450,
    0.102901041421, 0.112661287575, 0.122410675199, 0.132148264628, 0.141873117332,
    0.151584296010, 0.161280864678, 0.170961888760, 0.180626435180, 0.190273572448,
    0.199902370753, 0.209511902052, 0.219101240157, 0.228669460829, 0.238215641862,
    0.247738863176, 0.257238206902, 0.266712757475, 0.276161601717, 0.285583828929,
    0.294978530977, 0.304344802381, 0.313681740399, 0.322988445118, 0.332264019538,
    0.341507569661, 0.350718204573, 0.359895036535, 0.369037181064, 0.378143757022,
    0.387213886697, 0.396246695891, 0.405241314005, 0.414196874117, 0.423112513073,
    0.431987371563, 0.440820594212, 0.449611329655, 0.458358730621, 0.467061954019,
    0.475720161014, 0.484332517110, 0.492898192230, 0.501416360796, 0.509886201809,
    0.518306898929, 0.526677640552, 0.534997619887, 0.543266035038, 0.551482089078,
    0.559644990127, 0.567753951426, 0.575808191418, 0.583806933818, 0.591749407690,
    0.599634847523, 0.607462493302, 0.615231590581, 0.622941390558, 0.630591150148,
    0.638180132051, 0.645707604824, 0.653172842954, 0.660575126926, 0.667913743292,
    0.675187984742, 0.682397150168, 0.689540544737, 0.696617479953, 0.703627273726,
    0.710569250438, 0.717442741007, 0.724247082951, 0.730981620454, 0.737645704427,
    0.744238692572, 0.750759949443, 0.757208846506, 0.763584762206, 0.769887082016,
    0.776115198508, 0.782268511401, 0.788346427627, 0.794348361383, 0.800273734191,
    0.806121974951, 0.811892519997, 0.817584813152, 0.823198305781, 0.828732456844,
    0.834186732948, 0.839560608398, 0.844853565250, 0.850065093356, 0.855194690420,
    0.860241862039, 0.865206121757, 0.870086991109, 0.874883999665, 0.879596685080,
    0.884224593137, 0.888767277786, 0.893224301196, 0.897595233788, 0.901879654283,
    0.906077149740, 0.910187315596, 0.914209755704, 0.918144082372, 0.921989916403,
    0.925746887127, 0.929414632439, 0.932992798835, 0.936481041442, 0.939879024058,
    0.943186419177, 0.946402908026, 0.949528180593, 0.952561935658, 0.955503880820,
    0.958353732530, 0.961111216112, 0.963776065795, 0.966348024735, 0.968826845041,
    0.971212287799, 0.973504123096, 0.975702130039, 0.977806096779, 0.979815820533,
    0.981731107599, 0.983551773378, 0.985277642389, 0.986908548290, 0.988444333892,
    0.989884851171, 0.991229961288, 0.992479534599, 0.993633450666, 0.994691598273,
    0.995653875433, 0.996520189401, 0.997290456679, 0.997964603026, 0.998542563469,
    0.999024282300, 0.999409713092, 0.999698818696, 0.999891571247, 0.999987952167,
    0.999987952167, 0.999891571247, 0.999698818696, 0.999409713092, 0.999024282300,
    0.998542563469, 0.997964603026, 0.997290456679, 0.996520189401, 0.995653875433,
    0.994691598273, 0.993633450666, 0.992479534599, 0.991229961288, 0.989884851171,
    0.988444333892, 0.986908548290, 0.985277642389, 0.983551773378, 0.981731107599,
    0.979815820533, 0.977806096779, 0.975702130039, 0.973504123096, 0.971212287799,
    0.968826845041, 0.966348024735, 0.963776065795, 0.961111216112, 0.958353732530,
    0.955503880820, 0.952561935658, 0.949528180593, 0.946402908026, 0.943186419177,
    0.939879024058, 0.936481041442, 0.932992798835, 0.929414632439, 0.925746887127,
    0.921989916403, 0.918144082372, 0.914209755704, 0.910187315596, 0.906077149740,
    0.901879654283, 0.897595233788, 0.893224301196, 0.888767277786, 0.884224593137,
    0.879596685080, 0.874883999665, 0.870086991109, 0.865206121757, 0.860241862039,
    0.855194690420, 0.850065093356, 0.844853565250, 0.839560608398, 0.834186732948,
    0.828732456844, 0.823198305781, 0.817584813152, 0.811892519997, 0.806121974951,
    0.800273734191, 0.794348361383, 0.788346427627, 0.782268511401, 0.776115198508,
    0.769887082016, 0.763584762206, 0.757208846506, 0.750759949443, 0.744238692572,
    0.737645704427, 0.730981620454, 0.724247082951, 0.717442741007, 0.710569250438,
    0.703627273726, 0.696617479953, 0.689540544737, 0.682397150168, 0.675187984742,
    0.667913743292, 0.660575126926, 0.653172842954, 0.645707604824, 0.638180132051,
    0.630591150148, 0.622941390558, 0.615231590581, 0.607462493302, 0.599634847523,
    0.591749407690, 0.583806933818, 0.575808191418, 0.567753951426, 0.559644990127,
    0.551482089078, 0.543266035038, 0.534997619887, 0.526677640552, 0.518306898929,
    0.509886201809, 0.501416360796, 0.492898192230, 0.484332517110, 0.475720161014,
    0.467061954019, 0.458358730621, 0.449611329655, 0.440820594212, 0.431987371563,
    0.423112513073, 0.414196874117, 0.405241314005, 0.396246695891, 0.387213886697,
    0.378143757022, 0.369037181064, 0.359895036535, 0.350718204573, 0.341507569661,
    0.332264019538, 0.322988445118, 0.313681740399, 0.304344802381, 0.294978530977,
    0.285583828929, 0.276161601717, 0.266712757475, 0.257238206902, 0.247738863176,
    0.238215641862, 0.228669460829, 0.219101240157, 0.209511902052, 0.199902370753,
    0.190273572448, 0.180626435180, 0.170961888760, 0.161280864678, 0.151584296010,
    0.141873117332, 0.132148264628, 0.122410675199, 0.112661287575, 0.102901041421,
    0.093130877450, 0.083351737332, 0.073564563600, 0.063770299562, 0.053969889210,
    0.044164277127, 0.034354408400, 0.024541228523, 0.014725683311, 0.004908718808,
];

/// Selects one of the three filterbanks of osce_features.c (`center_bins_*` /
/// `band_weights_*` pairs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filterbank {
    /// `center_bins_clean` / `band_weights_clean` (64 bands).
    Clean,
    /// `center_bins_noisy` / `band_weights_noisy` (18 bands).
    Noisy,
    /// `center_bins_bwe` / `band_weights_bwe` (32 bands).
    Bwe,
}

impl Filterbank {
    const fn tables(self) -> (&'static [usize], &'static [f32]) {
        match self {
            Self::Clean => (&CENTER_BINS_CLEAN, &BAND_WEIGHTS_CLEAN),
            Self::Noisy => (&CENTER_BINS_NOISY, &BAND_WEIGHTS_NOISY),
            Self::Bwe => (&CENTER_BINS_BWE, &BAND_WEIGHTS_BWE),
        }
    }
}

/// Port of dnn/osce_features.c:apply_filterbank (static): triangular filterbank over
/// `x_in[..=center_bins[num_bands-1]]` into `x_out[..num_bands]` (`x_out != x_in`).
#[expect(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]
pub fn apply_filterbank(x_out: &mut [f32], x_in: &[f32], bank: Filterbank) {
    let (center_bins, band_weights) = bank.tables();
    let num_bands = center_bins.len();
    let x_out = &mut x_out[..num_bands];

    x_out[0] = 0.0;
    for b in 0..num_bands - 1 {
        x_out[b + 1] = 0.0;
        for i in center_bins[b]..center_bins[b + 1] {
            let frac =
                (center_bins[b + 1] - i) as f32 / (center_bins[b + 1] - center_bins[b]) as f32;
            x_out[b] += band_weights[b] * frac * x_in[i];
            x_out[b + 1] += band_weights[b + 1] * (1.0 - frac) * x_in[i];
        }
    }
    x_out[num_bands - 1] += band_weights[num_bands - 1] * x_in[center_bins[num_bands - 1]];
}

/// Port of dnn/osce_features.c:mag_spec_320_onesided (static): `out[..161]` = magnitude
/// spectrum of `input[..320]`. C calls it in place; the input is copied into the FFT buffer
/// before any output is written, so `out` may be the same buffer as `input` in the callers.
fn mag_spec_320_onesided_inplace(buf: &mut [f32; OSCE_SPEC_WINDOW_SIZE]) {
    let mut buffer = [KissFftCpx::default(); OSCE_SPEC_WINDOW_SIZE];
    forward_transform(&mut buffer, buf);
    for (o, c) in buf[..OSCE_SPEC_NUM_FREQS].iter_mut().zip(&buffer) {
        *o = (OSCE_SPEC_WINDOW_SIZE as f64 * math::sqrt((c.r * c.r + c.i * c.i) as f64)) as f32;
    }
}

/// Port of dnn/osce_features.c:mag_spec_320_onesided (static), out of place.
pub fn mag_spec_320_onesided(out: &mut [f32], input: &[f32]) {
    let mut buf = [0f32; OSCE_SPEC_WINDOW_SIZE];
    buf.copy_from_slice(&input[..OSCE_SPEC_WINDOW_SIZE]);
    mag_spec_320_onesided_inplace(&mut buf);
    out[..OSCE_SPEC_NUM_FREQS].copy_from_slice(&buf[..OSCE_SPEC_NUM_FREQS]);
}

/// Port of dnn/osce_features.c:calculate_log_spectrum_from_lpc (static): 64 log band energies
/// of the LPC synthesis filter `a_q12[..lpc_order]`.
pub fn calculate_log_spectrum_from_lpc(spec: &mut [f32], a_q12: &[i16], lpc_order: usize) {
    let mut buffer = [0f32; OSCE_SPEC_WINDOW_SIZE];

    // Zero expansion.
    buffer[0] = 1.0;
    for (b, &a) in buffer[1..=lpc_order].iter_mut().zip(&a_q12[..lpc_order]) {
        *b = -(a as f32) / (1u32 << 12) as f32;
    }

    // Calculate and invert magnitude spectrum.
    mag_spec_320_onesided_inplace(&mut buffer);

    for b in &mut buffer[..OSCE_SPEC_NUM_FREQS] {
        *b = 1.0f32 / (*b + 1e-9f32);
    }

    // Apply filterbank.
    apply_filterbank(spec, &buffer, Filterbank::Clean);

    // Log and scaling.
    for s in &mut spec[..OSCE_CLEAN_SPEC_NUM_BANDS] {
        *s = (0.3f32 as f64 * math::log((*s + 1e-9f32) as f64)) as f32;
    }
}

/// Port of dnn/osce_features.c:calculate_cepstrum (static): 18 cepstral coefficients of the
/// windowed `signal[..320]`.
pub fn calculate_cepstrum(cepstrum: &mut [f32], signal: &[f32]) {
    let mut buffer = [0f32; OSCE_SPEC_WINDOW_SIZE];

    for ((b, &w), &s) in buffer
        .iter_mut()
        .zip(&OSCE_WINDOW)
        .zip(&signal[..OSCE_SPEC_WINDOW_SIZE])
    {
        *b = w * s;
    }

    // Calculate magnitude spectrum.
    mag_spec_320_onesided_inplace(&mut buffer);

    // Accumulate bands: C writes them to `spec = &buffer[OSCE_SPEC_NUM_FREQS + 3]`.
    let (mag, spec) = buffer.split_at_mut(OSCE_SPEC_NUM_FREQS + 3);
    apply_filterbank(spec, mag, Filterbank::Noisy);

    // Log domain conversion.
    for s in &mut spec[..OSCE_NOISY_SPEC_NUM_BANDS] {
        *s = math::log((*s + 1e-9f32) as f64) as f32;
    }

    // DCT-II (orthonormal).
    dct(cepstrum, spec);
}

/// Port of dnn/osce_features.c:calculate_acorr (static): normalized correlation of
/// `buf[pos..pos+80]` with the signal `lag-2..=lag+2` samples earlier. C takes the pointer
/// `signal = buf + pos` and indexes it negatively.
pub fn calculate_acorr(acorr: &mut [f32], buf: &[f32], pos: usize, lag: i32) {
    let signal = &buf[pos..pos + 80];
    for k in -2i32..=2 {
        let start = pos as isize - lag as isize + k as isize;
        debug_assert!(start >= 0);
        let delayed = &buf[start as usize..start as usize + 80];
        let mut xx = 0f32;
        let mut xy = 0f32;
        let mut yy = 0f32;
        for (&s, &d) in signal.iter().zip(delayed) {
            // "obviously wasteful -> fix later"
            xx += s * s;
            yy += d * d;
            xy += s * d;
        }
        acorr[(k + 2) as usize] = (xy as f64 / math::sqrt((xx * yy + 1e-9f32) as f64)) as f32;
    }
}

/// Port of dnn/osce_features.c:pitch_postprocessing (static): replaces the lag of unvoiced
/// frames by `OSCE_NO_PITCH_VALUE` (the hangover is disabled upstream).
pub fn pitch_postprocessing(ps_features: &mut OsceFeatureState, lag: i32, type_: i32) -> i32 {
    let new_lag;
    let mut modulus = OSCE_PITCH_HANGOVER;
    if modulus == 0 {
        modulus += 1;
    }

    // "hangover is currently disabled to reflect a bug in the python code. ToDo: re-evaluate
    // hangover"
    if type_ != TYPE_VOICED && ps_features.last_type == TYPE_VOICED && OSCE_HANGOVER_BUGFIX {
        // Enter hangover.
        let mut l = OSCE_NO_PITCH_VALUE;
        if ps_features.pitch_hangover_count < OSCE_PITCH_HANGOVER {
            l = ps_features.last_lag;
            ps_features.pitch_hangover_count = (ps_features.pitch_hangover_count + 1) % modulus;
        }
        new_lag = l;
    } else if type_ != TYPE_VOICED && ps_features.pitch_hangover_count != 0 && OSCE_HANGOVER_BUGFIX
    {
        // Continue hangover.
        new_lag = ps_features.last_lag;
        ps_features.pitch_hangover_count = (ps_features.pitch_hangover_count + 1) % modulus;
    } else if type_ != TYPE_VOICED {
        // Unvoiced frame after hangover.
        new_lag = OSCE_NO_PITCH_VALUE;
        ps_features.pitch_hangover_count = 0;
    } else {
        // Voiced frame: update last_lag.
        new_lag = lag;
        ps_features.last_lag = lag;
        ps_features.pitch_hangover_count = 0;
    }

    // Buffer update.
    ps_features.last_type = type_;

    // "with the current setup this should never happen (but who knows...)"
    debug_assert!(new_lag != 0);

    new_lag
}

/// Port of dnn/osce_features.c:osce_calculate_features: `nb_subfr` frames of
/// `OSCE_FEATURE_DIM` features, the bit count and its smoothed value, and the per-subframe
/// pitch lags for the SILK frame `xq[..nb_subfr*80]` (16 kHz).
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the C signature of osce_calculate_features"
)]
#[expect(
    clippy::needless_range_loop,
    reason = "index arithmetic mirrors C across several arrays"
)]
pub fn osce_calculate_features(
    ps_features: &mut OsceFeatureState,
    dec: OsceDecInfo,
    ps_dec_ctrl: &SilkDecoderControl,
    features: &mut [f32],
    numbits: &mut [f32; 2],
    periods: &mut [i32; 4],
    xq: &[i16],
    num_bits: i32,
) {
    let mut buffer = [0f32; OSCE_FEATURES_MAX_HISTORY + OSCE_MAX_FEATURE_FRAMES * 80];

    let num_subframes = dec.nb_subfr as usize;
    let num_samples = num_subframes * 80;
    debug_assert!(num_subframes <= OSCE_MAX_FEATURE_FRAMES);

    // Smooth bit count.
    ps_features.numbits_smooth = 0.9f32 * ps_features.numbits_smooth + 0.1f32 * num_bits as f32;
    numbits[0] = num_bits as f32;
    numbits[1] = ps_features.numbits_smooth;

    for (b, &x) in buffer[OSCE_FEATURES_MAX_HISTORY..OSCE_FEATURES_MAX_HISTORY + num_samples]
        .iter_mut()
        .zip(&xq[..num_samples])
    {
        *b = x as f32 / (1u32 << 15) as f32;
    }
    buffer[..OSCE_FEATURES_MAX_HISTORY].copy_from_slice(&ps_features.signal_history);

    for k in 0..num_subframes {
        let pf = k * OSCE_FEATURE_DIM;
        let frame = OSCE_FEATURES_MAX_HISTORY + k * 80;
        // C: memset(pfeatures, 0, OSCE_FEATURE_DIM) "precaution" (clears 93 bytes only); every
        // feature is overwritten below, so it has no effect.

        // Clean spectrum from lpcs (update every other frame).
        if k % 2 == 0 {
            calculate_log_spectrum_from_lpc(
                &mut features[pf + OSCE_CLEAN_SPEC_START..],
                &ps_dec_ctrl.pred_coef_q12[k >> 1],
                dec.lpc_order as usize,
            );
        } else {
            features.copy_within(
                pf + OSCE_CLEAN_SPEC_START - OSCE_FEATURE_DIM
                    ..pf + OSCE_CLEAN_SPEC_START - OSCE_FEATURE_DIM + OSCE_CLEAN_SPEC_LENGTH,
                pf + OSCE_CLEAN_SPEC_START,
            );
        }

        // Noisy cepstrum from signal (update every other frame).
        if k % 2 == 0 {
            calculate_cepstrum(
                &mut features[pf + OSCE_NOISY_CEPSTRUM_START..],
                &buffer[frame - 160..],
            );
        } else {
            features.copy_within(
                pf + OSCE_NOISY_CEPSTRUM_START - OSCE_FEATURE_DIM
                    ..pf + OSCE_NOISY_CEPSTRUM_START - OSCE_FEATURE_DIM
                        + OSCE_NOISY_CEPSTRUM_LENGTH,
                pf + OSCE_NOISY_CEPSTRUM_START,
            );
        }

        // Pitch hangover and zero value replacement.
        periods[k] = pitch_postprocessing(ps_features, ps_dec_ctrl.pitch_l[k], dec.signal_type);

        // Auto-correlation around pitch lag.
        calculate_acorr(
            &mut features[pf + OSCE_ACORR_START..],
            &buffer,
            frame,
            periods[k],
        );

        // LTP.
        const { assert!(OSCE_LTP_LENGTH == LTP_ORDER) };
        for i in 0..OSCE_LTP_LENGTH {
            features[pf + OSCE_LTP_START + i] =
                ps_dec_ctrl.ltp_coef_q14[k * LTP_ORDER + i] as f32 / (1u32 << 14) as f32;
        }

        // Frame gain.
        features[pf + OSCE_LOG_GAIN_START] =
            math::log((ps_dec_ctrl.gains_q16[k] as f32 / (1u64 << 16) as f32 + 1e-9f32) as f64)
                as f32;
    }

    // Buffer update.
    ps_features
        .signal_history
        .copy_from_slice(&buffer[num_samples..num_samples + OSCE_FEATURES_MAX_HISTORY]);
}

/// Port of dnn/osce_features.c:osce_bwe_calculate_features: `num_samples/160` frames of
/// `OSCE_BWE_FEATURE_DIM` features (32 log band magnitudes + 2x41 instantaneous-frequency
/// values) of the 16 kHz signal `xq[..num_samples]`.
pub fn osce_bwe_calculate_features(
    ps_features: &mut OsceBweFeatureState,
    features: &mut [f32],
    xq: &[i16],
    num_samples: usize,
) {
    let mut fft_buffer = [KissFftCpx::default(); OSCE_BWE_WINDOW_SIZE];
    let mut spec = [0f32; 2 * OSCE_BWE_MAX_INSTAFREQ_BIN + 2];
    let mut buffer = [0f32; OSCE_BWE_WINDOW_SIZE];
    let mut mag_spec = [0f32; OSCE_SPEC_NUM_FREQS];

    // OSCE_BWE_WINDOW_SIZE == 320 is a hard requirement.
    debug_assert!(
        num_samples.is_multiple_of(OSCE_BWE_HALF_WINDOW_SIZE) && OSCE_BWE_WINDOW_SIZE == 320
    );

    let num_frames = num_samples / OSCE_BWE_HALF_WINDOW_SIZE;

    for frame in 0..num_frames {
        // Clear features.
        let feat = &mut features[frame * OSCE_BWE_FEATURE_DIM..(frame + 1) * OSCE_BWE_FEATURE_DIM];
        feat.fill(0.0);
        let (lmspec, instafreq) = feat.split_at_mut(OSCE_BWE_NUM_BANDS);
        let x = &xq[frame * OSCE_BWE_HALF_WINDOW_SIZE..(frame + 1) * OSCE_BWE_HALF_WINDOW_SIZE];

        buffer[..OSCE_BWE_HALF_WINDOW_SIZE].copy_from_slice(&ps_features.signal_history);
        for (b, &v) in buffer[OSCE_BWE_HALF_WINDOW_SIZE..].iter_mut().zip(x) {
            *b = v as f32 / (1u32 << 15) as f32;
        }

        // Update signal history buffer.
        ps_features
            .signal_history
            .copy_from_slice(&buffer[OSCE_BWE_HALF_WINDOW_SIZE..]);

        // Apply window.
        for (b, &w) in buffer.iter_mut().zip(&OSCE_WINDOW) {
            *b *= w;
        }

        // DFT.
        forward_transform(&mut fft_buffer, &buffer);

        // Instafreq.
        for k in 0..=OSCE_BWE_MAX_INSTAFREQ_BIN {
            // "ToDo: remove 1e-9 from python code" (a double addition).
            spec[2 * k] = ((OSCE_BWE_WINDOW_SIZE as f32 * fft_buffer[k].r) as f64 + 1e-9) as f32;
            spec[2 * k + 1] = OSCE_BWE_WINDOW_SIZE as f32 * fft_buffer[k].i;
            let re1 = spec[2 * k];
            let im1 = spec[2 * k + 1];
            let re2 = ps_features.last_spec[2 * k];
            let im2 = ps_features.last_spec[2 * k + 1];
            let aux_r = re1 * re2 + im1 * im2;
            let aux_i = im1 * re2 - re1 * im2;
            let aux_abs = math::sqrt((aux_r * aux_r + aux_i * aux_i) as f64) as f32;
            instafreq[k] = (aux_r as f64 / (aux_abs as f64 + 1e-9)) as f32;
            instafreq[k + OSCE_BWE_MAX_INSTAFREQ_BIN + 1] =
                (aux_i as f64 / (aux_abs as f64 + 1e-9)) as f32;
        }

        // ERB-scale magnitude spectrogram.
        for (m, c) in mag_spec.iter_mut().zip(&fft_buffer) {
            *m = (OSCE_BWE_WINDOW_SIZE as f64 * math::sqrt((c.r * c.r + c.i * c.i) as f64)) as f32;
        }

        apply_filterbank(lmspec, &mag_spec, Filterbank::Bwe);

        for l in lmspec.iter_mut() {
            *l = math::log(*l as f64 + 1e-9) as f32;
        }

        // Update instafreq buffer.
        ps_features.last_spec.copy_from_slice(&spec);
    }
}

/// Port of dnn/osce_features.c:osce_cross_fade_10ms: fades from `x_in` into `x_enhanced` over
/// the first 160 samples (`length >= 160`).
pub fn osce_cross_fade_10ms(x_enhanced: &mut [f32], x_in: &[f32], length: usize) {
    debug_assert!(length >= 160);
    for ((e, &x), &w) in x_enhanced[..160]
        .iter_mut()
        .zip(&x_in[..160])
        .zip(&OSCE_WINDOW)
    {
        *e = w * *e + (1.0f32 - w) * x;
    }
}

/// Port of dnn/osce_features.c:osce_bwe_cross_fade_10ms: fades from `x_fadeout` into
/// `x_fadein` over the first 480 (48 kHz) samples (`length >= 480`).
pub fn osce_bwe_cross_fade_10ms(x_fadein: &mut [i16], x_fadeout: &[i16], length: usize) {
    debug_assert!(length >= 480);
    let f = 1.0f32 / 3.0;
    let x_fadein = &mut x_fadein[..480];
    let x_fadeout = &x_fadeout[..480];
    for i in 0..160 {
        let diff = if i == 159 {
            0.0f32
        } else {
            OSCE_WINDOW[i + 1] - OSCE_WINDOW[i]
        };
        let mut w_curr = OSCE_WINDOW[i];
        for j in 0..3 {
            let n = 3 * i + j;
            // C: (int)(w*in + (1.f-w)*out + 0.5), the + 0.5 in double, truncated.
            x_fadein[n] = ((w_curr * x_fadein[n] as f32 + (1.0f32 - w_curr) * x_fadeout[n] as f32)
                as f64
                + 0.5) as i32 as i16;
            if j < 2 {
                w_curr += diff * f;
            }
        }
    }
}
