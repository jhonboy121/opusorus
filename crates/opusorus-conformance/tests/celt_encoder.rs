//! Differential tests for unit `celt_encoder` (celt/celt_encoder.c) vs the C oracle.
//!
//! * Static helpers (transient/tone analysis, MDCTs, pre-emphasis, TF analysis/encoding, trim,
//!   stereo analysis, dynalloc, VBR target) on randomized realistic inputs — bit-exact.
//! * Persistent encoder streams: the library CELT encoder and the Rust one driven with
//!   identical CTLs and input across many frames and configurations (all LM, mono/stereo,
//!   downmixed stereo, 8-48 kHz input, 6-510 kb/s, CBR/VBR/CVBR, complexity 0-10, prediction,
//!   loss, LSB depth, LFE, energy masks, analysis/SILK info, band limits, resets, silence,
//!   impulses, tones, clipping...). Packet bytes, length, final range and the full encoder state
//!   are compared bit-exactly.
//! * Hybrid calls with a shared range coder (start band 17, simulated SILK payload).
//! * Opus replay: a private copy of the C Opus encoder records every call it makes into the
//!   CELT encoder (CELT-only, hybrid, redundancy, prefill, QEXT). Each call is replayed in Rust
//!   from the recorded C state and every output is compared.

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "test drivers mirror the C shim signatures and index several arrays"
)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_const_for_fn,
    reason = "test code: failures should panic"
)]

use std::borrow::Cow;

use opusorus::analysis::AnalysisInfo;
use opusorus::celt::celt::SilkInfo;
use opusorus::celt::celt_encoder::{self as ce, CeltEncoder, DynallocScratch};
use opusorus::celt::entenc::EcEnc;
use opusorus::celt::modes::opus_custom_mode_create;
use opusorus::celt::static_modes::CeltMode;
use opusorus_conformance::{Rng, assert_bits_eq_f32, assert_slice_eq, signals};
use opusorus_oracle::celt_encoder as oc;

// CTL request numbers (opus_defines.h / celt.h).
const OPUS_SET_BITRATE: i32 = 4002;
const OPUS_SET_MAX_BANDWIDTH: i32 = 4004;
const OPUS_SET_VBR: i32 = 4006;
const OPUS_SET_BANDWIDTH: i32 = 4008;
const OPUS_SET_COMPLEXITY: i32 = 4010;
const OPUS_SET_INBAND_FEC: i32 = 4012;
const OPUS_SET_PACKET_LOSS_PERC: i32 = 4014;
const OPUS_SET_DTX: i32 = 4016;
const OPUS_SET_VBR_CONSTRAINT: i32 = 4020;
const OPUS_SET_FORCE_CHANNELS: i32 = 4022;
const OPUS_SET_SIGNAL: i32 = 4024;
const OPUS_RESET_STATE: i32 = 4028;
const OPUS_GET_FINAL_RANGE: i32 = 4031;
const OPUS_SET_LSB_DEPTH: i32 = 4036;
const OPUS_GET_LSB_DEPTH: i32 = 4037;
const OPUS_SET_EXPERT_FRAME_DURATION: i32 = 4040;
const OPUS_SET_PREDICTION_DISABLED: i32 = 4042;
const OPUS_SET_PHASE_INVERSION_DISABLED: i32 = 4046;
const OPUS_GET_PHASE_INVERSION_DISABLED: i32 = 4047;
#[cfg(feature = "qext")]
const OPUS_SET_QEXT: i32 = 4056;
#[cfg(feature = "qext")]
const OPUS_GET_QEXT: i32 = 4057;
const CELT_SET_PREDICTION: i32 = 10002;
#[cfg(feature = "custom-modes")]
const CELT_SET_INPUT_CLIPPING: i32 = 10004;
const CELT_SET_CHANNELS: i32 = 10008;
const CELT_SET_START_BAND: i32 = 10010;
const CELT_SET_END_BAND: i32 = 10012;
const CELT_SET_SIGNALLING: i32 = 10016;
const OPUS_SET_LFE: i32 = 10024;

const APP_VOIP: i32 = 2048;
const APP_AUDIO: i32 = 2049;
const APP_RESTRICTED_LOWDELAY: i32 = 2051;
const APP_RESTRICTED_CELT: i32 = 2053;

// ---------------------------------------------------------------------------------------------
// State conversion / comparison
// ---------------------------------------------------------------------------------------------

/// The Rust mode for a dumped C state.
fn mode_for(d: &oc::CeState) -> Cow<'static, CeltMode> {
    let frame = d.mode_short_mdct_size * d.mode_nb_short_mdcts;
    if let Ok(m) = opus_custom_mode_create(d.mode_fs, frame)
        && m.short_mdct_size == d.mode_short_mdct_size
        && m.nb_short_mdcts == d.mode_nb_short_mdcts
    {
        return Cow::Borrowed(m);
    }
    #[cfg(feature = "custom-modes")]
    {
        opusorus::celt::modes::opus_custom_mode_create_custom(d.mode_fs, frame).unwrap()
    }
    #[cfg(not(feature = "custom-modes"))]
    panic!("no static mode for fs={} frame={frame}", d.mode_fs)
}

/// Dumps a Rust encoder in the oracle's flat layout.
fn dump(r: &CeltEncoder) -> Box<oc::CeState> {
    let mut d = Box::new(oc::CeState::zeroed());
    let m = r.mode();
    let (cc, nb, ov) = (
        r.channels as usize,
        m.nb_ebands as usize,
        m.overlap as usize,
    );
    d.channels = r.channels;
    d.stream_channels = r.stream_channels;
    d.force_intra = r.force_intra;
    d.clip = r.clip;
    d.disable_pf = r.disable_pf;
    d.complexity = r.complexity;
    d.upsample = r.upsample;
    d.start = r.start;
    d.end = r.end;
    d.bitrate = r.bitrate;
    d.vbr = r.vbr;
    d.signalling = r.signalling;
    d.constrained_vbr = r.constrained_vbr;
    d.loss_rate = r.loss_rate;
    d.lsb_depth = r.lsb_depth;
    d.lfe = r.lfe;
    d.disable_inv = r.disable_inv;
    #[cfg(feature = "qext")]
    {
        d.enable_qext = r.enable_qext;
        d.qext_scale = r.qext_scale;
    }
    #[cfg(not(feature = "qext"))]
    {
        d.qext_scale = 1;
    }
    d.mode_fs = m.fs;
    d.mode_short_mdct_size = m.short_mdct_size;
    d.mode_nb_short_mdcts = m.nb_short_mdcts;
    d.mode_nb_ebands = m.nb_ebands;
    d.mode_overlap = m.overlap;
    d.rng = r.rng;
    d.spread_decision = r.spread_decision;
    d.delayed_intra = r.delayed_intra;
    d.tonal_average = r.tonal_average;
    d.last_coded_bands = r.last_coded_bands;
    d.hf_average = r.hf_average;
    d.tapset_decision = r.tapset_decision;
    d.prefilter_period = r.prefilter_period;
    d.prefilter_gain = r.prefilter_gain;
    d.prefilter_tapset = r.prefilter_tapset;
    d.consec_transient = r.consec_transient;
    let a = &r.analysis;
    d.an_valid = a.valid;
    d.an_tonality = a.tonality;
    d.an_tonality_slope = a.tonality_slope;
    d.an_noisiness = a.noisiness;
    d.an_activity = a.activity;
    d.an_music_prob = a.music_prob;
    d.an_music_prob_min = a.music_prob_min;
    d.an_music_prob_max = a.music_prob_max;
    d.an_bandwidth = a.bandwidth;
    d.an_activity_probability = a.activity_probability;
    d.an_max_pitch_ratio = a.max_pitch_ratio;
    d.an_leak_boost[..19].copy_from_slice(&a.leak_boost);
    d.silk_signal_type = r.silk_info.signal_type;
    d.silk_offset = r.silk_info.offset;
    d.preemph_mem_e = r.preemph_mem_e;
    d.preemph_mem_d = r.preemph_mem_d;
    d.vbr_reservoir = r.vbr_reservoir;
    d.vbr_drift = r.vbr_drift;
    d.vbr_offset = r.vbr_offset;
    d.vbr_count = r.vbr_count;
    d.overlap_max = r.overlap_max;
    d.stereo_saving = r.stereo_saving;
    d.intensity = r.intensity;
    d.has_energy_mask = i32::from(r.has_energy_mask);
    if r.has_energy_mask {
        d.energy_mask[..cc * nb].copy_from_slice(&r.energy_mask[..cc * nb]);
    }
    d.spec_avg = r.spec_avg;
    d.in_mem[..cc * ov].copy_from_slice(&r.in_mem);
    d.prefilter_mem[..r.prefilter_mem.len()].copy_from_slice(&r.prefilter_mem);
    d.old_band_e[..cc * nb].copy_from_slice(&r.old_band_e);
    d.old_log_e[..cc * nb].copy_from_slice(&r.old_log_e);
    d.old_log_e2[..cc * nb].copy_from_slice(&r.old_log_e2);
    d.energy_error[..cc * nb].copy_from_slice(&r.energy_error);
    #[cfg(feature = "qext")]
    d.qext_old_band_e[..r.qext_old_band_e.len()].copy_from_slice(&r.qext_old_band_e);
    d
}

/// Builds a Rust encoder holding exactly the dumped C state.
fn rust_from_state(d: &oc::CeState) -> CeltEncoder {
    let mut r = CeltEncoder::opus_custom_encoder_init_arch(mode_for(d), d.channels).unwrap();
    let nb = d.mode_nb_ebands as usize;
    let (cc, ov) = (d.channels as usize, d.mode_overlap as usize);
    r.stream_channels = d.stream_channels;
    r.force_intra = d.force_intra;
    r.clip = d.clip;
    r.disable_pf = d.disable_pf;
    r.complexity = d.complexity;
    r.upsample = d.upsample;
    r.start = d.start;
    r.end = d.end;
    r.bitrate = d.bitrate;
    r.vbr = d.vbr;
    r.signalling = d.signalling;
    r.constrained_vbr = d.constrained_vbr;
    r.loss_rate = d.loss_rate;
    r.lsb_depth = d.lsb_depth;
    r.lfe = d.lfe;
    r.disable_inv = d.disable_inv;
    #[cfg(feature = "qext")]
    {
        r.enable_qext = d.enable_qext;
        assert_eq!(r.qext_scale, d.qext_scale);
    }
    r.rng = d.rng;
    r.spread_decision = d.spread_decision;
    r.delayed_intra = d.delayed_intra;
    r.tonal_average = d.tonal_average;
    r.last_coded_bands = d.last_coded_bands;
    r.hf_average = d.hf_average;
    r.tapset_decision = d.tapset_decision;
    r.prefilter_period = d.prefilter_period;
    r.prefilter_gain = d.prefilter_gain;
    r.prefilter_tapset = d.prefilter_tapset;
    r.consec_transient = d.consec_transient;
    let mut leak = [0u8; 19];
    leak.copy_from_slice(&d.an_leak_boost[..19]);
    r.analysis = AnalysisInfo {
        valid: d.an_valid,
        tonality: d.an_tonality,
        tonality_slope: d.an_tonality_slope,
        noisiness: d.an_noisiness,
        activity: d.an_activity,
        music_prob: d.an_music_prob,
        music_prob_min: d.an_music_prob_min,
        music_prob_max: d.an_music_prob_max,
        bandwidth: d.an_bandwidth,
        activity_probability: d.an_activity_probability,
        max_pitch_ratio: d.an_max_pitch_ratio,
        leak_boost: leak,
    };
    r.silk_info = SilkInfo {
        signal_type: d.silk_signal_type,
        offset: d.silk_offset,
    };
    r.preemph_mem_e = d.preemph_mem_e;
    r.preemph_mem_d = d.preemph_mem_d;
    r.vbr_reservoir = d.vbr_reservoir;
    r.vbr_drift = d.vbr_drift;
    r.vbr_offset = d.vbr_offset;
    r.vbr_count = d.vbr_count;
    r.overlap_max = d.overlap_max;
    r.stereo_saving = d.stereo_saving;
    r.intensity = d.intensity;
    if d.has_energy_mask != 0 {
        r.set_energy_mask(Some(&d.energy_mask[..cc * nb]));
    } else {
        r.set_energy_mask(None);
    }
    r.spec_avg = d.spec_avg;
    r.in_mem.copy_from_slice(&d.in_mem[..cc * ov]);
    let pm = r.prefilter_mem.len();
    r.prefilter_mem.copy_from_slice(&d.prefilter_mem[..pm]);
    r.old_band_e.copy_from_slice(&d.old_band_e[..cc * nb]);
    r.old_log_e.copy_from_slice(&d.old_log_e[..cc * nb]);
    r.old_log_e2.copy_from_slice(&d.old_log_e2[..cc * nb]);
    r.energy_error.copy_from_slice(&d.energy_error[..cc * nb]);
    #[cfg(feature = "qext")]
    {
        let q = r.qext_old_band_e.len();
        r.qext_old_band_e.copy_from_slice(&d.qext_old_band_e[..q]);
    }
    r
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Compares two flat states field by field (floats bit-exactly).
#[track_caller]
fn cmp_state(what: &str, r: &oc::CeState, c: &oc::CeState) {
    macro_rules! int {
        ($($f:ident),*) => {$(
            assert_eq!(r.$f, c.$f, "{what}: field {}", stringify!($f));
        )*};
    }
    macro_rules! flt {
        ($($f:ident),*) => {$(
            assert_eq!(r.$f.to_bits(), c.$f.to_bits(), "{what}: field {} rust={} c={}",
                stringify!($f), r.$f, c.$f);
        )*};
    }
    macro_rules! arr {
        ($($f:ident),*) => {$(
            assert_slice_eq(&format!("{what}: {}", stringify!($f)), &bits(&r.$f), &bits(&c.$f));
        )*};
    }
    int!(
        channels,
        stream_channels,
        force_intra,
        clip,
        disable_pf,
        complexity,
        upsample,
        start,
        end,
        bitrate,
        vbr,
        signalling,
        constrained_vbr,
        loss_rate,
        lsb_depth,
        lfe,
        disable_inv,
        qext_scale,
        mode_fs,
        mode_short_mdct_size,
        mode_nb_short_mdcts,
        mode_nb_ebands,
        mode_overlap,
        rng,
        spread_decision,
        tonal_average,
        last_coded_bands,
        hf_average,
        tapset_decision,
        prefilter_period,
        prefilter_tapset,
        consec_transient,
        an_valid,
        an_bandwidth,
        an_leak_boost,
        silk_signal_type,
        silk_offset,
        vbr_reservoir,
        vbr_drift,
        vbr_offset,
        vbr_count,
        intensity,
        has_energy_mask
    );
    if cfg!(feature = "qext") {
        int!(enable_qext);
    }
    flt!(
        delayed_intra,
        prefilter_gain,
        an_tonality,
        an_tonality_slope,
        an_noisiness,
        an_activity,
        an_music_prob,
        an_music_prob_min,
        an_music_prob_max,
        an_activity_probability,
        an_max_pitch_ratio,
        overlap_max,
        stereo_saving,
        spec_avg
    );
    arr!(
        preemph_mem_e,
        preemph_mem_d,
        energy_mask,
        in_mem,
        prefilter_mem,
        old_band_e,
        old_log_e,
        old_log_e2,
        energy_error,
        qext_old_band_e
    );
}

/// A Rust range encoder with exactly the snapshot's state over `buf` (`storage` bytes used).
fn make_enc<'a>(buf: &'a mut [u8], s: &oc::EcSnap) -> EcEnc<'a> {
    let mut e = EcEnc::with_size(buf, s.storage);
    e.end_offs = s.end_offs;
    e.end_window = s.end_window;
    e.nend_bits = s.nend_bits;
    e.nbits_total = s.nbits_total;
    e.offs = s.offs;
    e.rng = s.rng;
    e.val = s.val;
    e.ext = s.ext;
    e.rem = s.rem;
    e.error = s.error;
    e
}

fn snap(e: &EcEnc<'_>) -> oc::EcSnap {
    oc::EcSnap {
        storage: e.storage,
        end_offs: e.end_offs,
        end_window: e.end_window,
        nend_bits: e.nend_bits,
        nbits_total: e.nbits_total,
        offs: e.offs,
        rng: e.rng,
        val: e.val,
        ext: e.ext,
        rem: e.rem,
        error: e.error,
    }
}

/// Applies the shim's `(kind, value, param)` writes to a Rust range coder.
fn apply_ops(e: &mut EcEnc<'_>, ops: &[[i32; 3]]) {
    for &[k, v, p] in ops {
        match k {
            0 => e.enc_bit_logp(v != 0, p as u32),
            1 => e.enc_uint(v as u32, p as u32),
            2 => e.enc_bits(v as u32, p as u32),
            _ => {}
        }
    }
}

/// Calls the Rust encoder (no QEXT TOC byte).
fn r_encode(
    r: &mut CeltEncoder,
    pcm: &[f32],
    frame_size: i32,
    out: Option<&mut [u8]>,
    nb: i32,
    enc: Option<&mut EcEnc<'_>>,
) -> i32 {
    r.celt_encode_with_ec(
        pcm,
        frame_size,
        out,
        nb,
        enc,
        #[cfg(feature = "qext")]
        None,
    )
}

/// Maps an int CTL request onto the typed Rust methods; returns the C return code.
fn rust_ctl(r: &mut CeltEncoder, request: i32, value: i32) -> i32 {
    fn code(res: opusorus::Result<()>) -> i32 {
        match res {
            Ok(()) => 0,
            Err(e) => e.code(),
        }
    }
    match request {
        OPUS_SET_COMPLEXITY => code(r.set_complexity(value)),
        CELT_SET_START_BAND => code(r.set_start_band(value)),
        CELT_SET_END_BAND => code(r.set_end_band(value)),
        CELT_SET_PREDICTION => code(r.set_prediction(value)),
        OPUS_SET_PACKET_LOSS_PERC => code(r.set_packet_loss_perc(value)),
        OPUS_SET_VBR_CONSTRAINT => {
            r.set_vbr_constraint(value);
            0
        }
        OPUS_SET_VBR => {
            r.set_vbr(value);
            0
        }
        OPUS_SET_BITRATE => code(r.set_bitrate(value)),
        CELT_SET_CHANNELS => code(r.set_channels(value)),
        OPUS_SET_LSB_DEPTH => code(r.set_lsb_depth(value)),
        OPUS_SET_PHASE_INVERSION_DISABLED => code(r.set_phase_inversion_disabled(value)),
        OPUS_RESET_STATE => {
            r.reset();
            0
        }
        CELT_SET_SIGNALLING => {
            r.set_signalling(value);
            0
        }
        OPUS_SET_LFE => {
            r.set_lfe(value);
            0
        }
        #[cfg(feature = "qext")]
        OPUS_SET_QEXT => code(r.set_qext(value)),
        #[cfg(feature = "custom-modes")]
        CELT_SET_INPUT_CLIPPING => {
            r.set_input_clipping(value);
            0
        }
        _ => panic!("unmapped request {request}"),
    }
}

/// A C and a Rust encoder driven in lockstep.
struct Pair {
    c: oc::CeltEnc,
    r: CeltEncoder,
    channels: usize,
    what: String,
}

impl Pair {
    fn new(fs: i32, channels: i32, what: &str) -> Self {
        let c = oc::CeltEnc::new(fs, channels).unwrap();
        let mut r = CeltEncoder::celt_encoder_init(fs, channels).unwrap();
        r.set_signalling(0);
        let p = Self {
            c,
            r,
            channels: channels as usize,
            what: what.to_string(),
        };
        p.check_state("init");
        p
    }

    fn ctl(&mut self, request: i32, value: i32) {
        let rc = self.c.ctl(request, value);
        let rr = rust_ctl(&mut self.r, request, value);
        assert_eq!(rr, rc, "{}: ctl {request}({value})", self.what);
    }

    fn check_state(&self, ctx: &str) {
        cmp_state(
            &format!("{} [{ctx}]", self.what),
            &dump(&self.r),
            &self.c.state(),
        );
    }

    fn set_analysis(&mut self, rng: &mut Rng) {
        let valid = i32::from(rng.range_i32(0, 4) != 0);
        let f: [f32; 9] = [
            0.5 + 0.5 * rng.f32_sym(),
            0.3 * rng.f32_sym(),
            0.5 + 0.5 * rng.f32_sym(),
            0.5 + 0.5 * rng.f32_sym(),
            0.5 + 0.5 * rng.f32_sym(),
            0.5 + 0.5 * rng.f32_sym(),
            0.5 + 0.5 * rng.f32_sym(),
            0.5 + 0.5 * rng.f32_sym(),
            0.5 + 0.5 * rng.f32_sym(),
        ];
        let bw = rng.range_i32(0, 20);
        let mut leak = [0u8; 19];
        for l in &mut leak {
            *l = if rng.range_i32(0, 2) == 0 {
                0
            } else {
                rng.range_i32(0, 255) as u8
            };
        }
        self.c.set_analysis(valid, &f, bw, &leak);
        self.r.set_analysis(Some(&AnalysisInfo {
            valid,
            tonality: f[0],
            tonality_slope: f[1],
            noisiness: f[2],
            activity: f[3],
            music_prob: f[4],
            music_prob_min: f[5],
            music_prob_max: f[6],
            bandwidth: bw,
            activity_probability: f[7],
            max_pitch_ratio: f[8],
            leak_boost: leak,
        }));
    }

    fn set_silk_info(&mut self, signal_type: i32, offset: i32) {
        self.c.set_silk_info(signal_type, offset);
        self.r.set_silk_info(Some(&SilkInfo {
            signal_type,
            offset,
        }));
    }

    fn set_energy_mask(&mut self, mask: Option<&[f32]>) {
        self.c.set_energy_mask(mask);
        self.r.set_energy_mask(mask);
    }

    /// Encodes one frame into a fresh buffer on both sides; compares everything.
    fn encode(&mut self, pcm: &[f32], frame_size: i32, nb: i32, ctx: &str) -> Vec<u8> {
        let size = nb.max(0) as usize;
        let mut oc_ = vec![0xA5u8; size];
        let mut or = vec![0xA5u8; size];
        let rc = self.c.encode(pcm, self.channels, frame_size, &mut oc_, nb);
        let rr = r_encode(&mut self.r, pcm, frame_size, Some(&mut or), nb, None);
        assert_eq!(rr, rc, "{} [{ctx}]: return value", self.what);
        assert_slice_eq(&format!("{} [{ctx}]: bytes", self.what), &or, &oc_);
        let (_, rng_c) = self.c.ctl_get(OPUS_GET_FINAL_RANGE);
        assert_eq!(
            self.r.final_range(),
            rng_c as u32,
            "{} [{ctx}]: final range",
            self.what
        );
        if rc > 0 {
            or.truncate(rc as usize);
        }
        or
    }

    /// Hybrid-style encode with a shared range coder (simulated SILK `ops` first).
    fn encode_with_ec(
        &mut self,
        pcm: &[f32],
        frame_size: i32,
        buf_size: usize,
        nb: i32,
        ops: &[[i32; 3]],
        ctx: &str,
    ) {
        let mut bc = vec![0u8; buf_size];
        let (rc, sc, shift_c) =
            self.c
                .encode_with_ec(pcm, self.channels, frame_size, &mut bc, nb, ops);
        let mut br = vec![0u8; buf_size];
        let (toc, rest) = br.split_at_mut(1);
        let base = rest.as_ptr() as usize;
        let mut e = EcEnc::new(rest);
        apply_ops(&mut e, ops);
        if (nb as usize) < buf_size - 1 {
            e.shrink(nb as u32);
        }
        #[cfg(feature = "qext")]
        let rr =
            self.r
                .celt_encode_with_ec(pcm, frame_size, None, nb, Some(&mut e), Some(&mut toc[0]));
        #[cfg(not(feature = "qext"))]
        let rr = {
            let _ = &toc;
            self.r
                .celt_encode_with_ec(pcm, frame_size, None, nb, Some(&mut e))
        };
        let shift_r = (e.buf.as_ptr() as usize - base) as i32;
        let sr = snap(&e);
        let _ = e; // end the coder's borrow of the buffer
        assert_eq!(rr, rc, "{} [{ctx}]: return value", self.what);
        assert_eq!(sr, sc, "{} [{ctx}]: range coder state", self.what);
        assert_eq!(shift_r, shift_c, "{} [{ctx}]: buffer shift", self.what);
        assert_slice_eq(&format!("{} [{ctx}]: buffer", self.what), &br, &bc);
        let (_, rng_c) = self.c.ctl_get(OPUS_GET_FINAL_RANGE);
        assert_eq!(
            self.r.final_range(),
            rng_c as u32,
            "{} [{ctx}]: final range",
            self.what
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------------------------

/// Kinds of test input.
const SIGNAL_KINDS: usize = 14;

/// `n` samples per channel of signal `kind` (interleaved).
fn gen_signal(kind: usize, n: usize, ch: usize, fs: u32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed ^ 0x5eed);
    let sine = |f: f64, amp: f64| -> Vec<f32> {
        let mut v = Vec::with_capacity(n * ch);
        for i in 0..n {
            for c in 0..ch {
                let t = i as f64 / fs as f64;
                v.push(
                    (amp * (2.0 * std::f64::consts::PI * f * (1.0 + 0.003 * c as f64) * t).sin())
                        as f32,
                );
            }
        }
        v
    };
    match kind {
        0 => signals::music_like(n, ch, fs, seed),
        1 => signals::speech_like(n, ch, fs, seed),
        2 => signals::noise(n, ch, 0.3, seed),
        3 => vec![0.0; n * ch],
        4 => {
            // Impulses / clicks on a quiet noise floor (transients).
            let mut v = signals::noise(n, ch, 0.001, seed);
            let period = 200 + rng.range_i32(0, 3000) as usize;
            for i in (rng.range_i32(0, 500) as usize..n).step_by(period) {
                for c in 0..ch {
                    v[i * ch + c] = if c == 0 { 0.9 } else { -0.7 };
                }
            }
            v
        }
        5 => {
            // Full-scale square wave.
            let p = 20 + rng.range_i32(0, 400) as usize;
            (0..n * ch)
                .map(|i| {
                    if (i / ch / p).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect()
        }
        6 => {
            // Clipping input (beyond full scale, triggers the 65536 clip in CELT).
            let mut v = signals::music_like(n, ch, fs, seed);
            let g = 2.0 + 3.0 * rng.f32_sym().abs();
            for x in &mut v {
                *x *= g;
            }
            v
        }
        7 => sine(50.0 + 5000.0 * f64::from(rng.f32_sym().abs()), 0.5), // pure tone
        8 => sine(10.0 + 30.0 * f64::from(rng.f32_sym().abs()), 0.8),   // very low tone
        9 => sine(
            0.45 * fs as f64 * f64::from(0.8 + 0.2 * rng.f32_sym().abs()),
            0.3,
        ), // near Nyquist
        10 => {
            // Bursts: switches between silence and loud noise (strong transients).
            let mut v = signals::noise(n, ch, 0.5, seed);
            let seg = 300 + rng.range_i32(0, 2000) as usize;
            for i in 0..n {
                if (i / seg).is_multiple_of(3) {
                    for c in 0..ch {
                        v[i * ch + c] = 0.0;
                    }
                }
            }
            v
        }
        11 => {
            // Very quiet signal near the LSB (digital silence detection with lsb_depth).
            signals::noise(n, ch, 2.0f32.powi(-(8 + rng.range_i32(0, 14))), seed)
        }
        12 => {
            // Chirp + DC offset, with anti-correlated stereo.
            let mut v = Vec::with_capacity(n * ch);
            for i in 0..n {
                let t = i as f64 / fs as f64;
                let s = 0.4 * (2.0 * std::f64::consts::PI * (100.0 + 4000.0 * t) * t).sin() + 0.1;
                for c in 0..ch {
                    v.push(if c == 0 { s } else { -s * 0.9 } as f32);
                }
            }
            v
        }
        _ => {
            // Speech/music mixture with a hard switch.
            let a = signals::speech_like(n, ch, fs, seed);
            let b = signals::music_like(n, ch, fs, seed + 1);
            let sw = n / 2;
            (0..n * ch)
                .map(|i| if i / ch < sw { a[i] } else { b[i] })
                .collect()
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------------------------

#[test]
fn layout() {
    assert!(oc::layout_ok());
    assert_eq!(oc::has_qext(), cfg!(feature = "qext"));
    assert_eq!(oc::has_custom(), cfg!(feature = "custom-modes"));
}

/// Rust mode for a standard sampling rate.
fn std_mode(fs: i32) -> &'static CeltMode {
    opus_custom_mode_create(fs, fs / 50).unwrap()
}

fn mode_rates() -> Vec<i32> {
    if cfg!(feature = "qext") {
        vec![48000, 96000]
    } else {
        vec![48000]
    }
}

/// Pre-emphasised-looking time signal for the analysis helpers (C channels of `len`).
fn analysis_input(rng: &mut Rng, len: usize, c: usize) -> Vec<f32> {
    let kind = rng.range_i32(0, SIGNAL_KINDS as i32 - 1) as usize;
    let s = gen_signal(kind, len, c, 48000, rng.next_u64());
    let scale = 32768.0 * if rng.range_i32(0, 3) == 0 { 0.01 } else { 1.0 };
    let mut out = vec![0.0f32; len * c];
    for ch in 0..c {
        for i in 0..len {
            out[ch * len + i] = scale * s[i * c + ch];
        }
    }
    out
}

#[test]
fn transient_and_tone_analysis() {
    let mut rng = Rng::new(0x7ea1);
    let mut tmp = vec![0.0f32; 4096];
    for it in 0..3000 {
        let c = rng.range_i32(1, 2);
        let fs = mode_rates()[rng.range_i32(0, mode_rates().len() as i32 - 1) as usize];
        let short = if fs == 96000 { 240 } else { 120 };
        let lm = rng.range_i32(0, 3);
        let len = (short << lm) + short;
        let input = analysis_input(&mut rng, len as usize, c as usize);
        // tone_detect first (its output feeds transient_analysis in the encoder)
        let (fc, tc) = oc::tone_detect(&input, c, len, fs);
        let mut tr = 0.0f32;
        let fr = ce::tone_detect(&input, c, len, &mut tr, fs, &mut tmp);
        assert_eq!(
            (fr.to_bits(), tr.to_bits()),
            (fc.to_bits(), tc.to_bits()),
            "tone_detect {it}"
        );
        let allow_weak = rng.range_i32(0, 1) == 1;
        let (tone_freq, toneishness) = if rng.range_i32(0, 3) == 0 {
            (
                0.02 * rng.f32_sym().abs(),
                0.97 + 0.03 * rng.f32_sym().abs(),
            )
        } else {
            (fc, tc)
        };
        let (ic, tfc, chc, wc) =
            oc::transient_analysis(&input, len, c, allow_weak, tone_freq, toneishness);
        let (mut tfr, mut chr, mut wr) = (0.0f32, 0i32, false);
        let ir = ce::transient_analysis(
            &input,
            len,
            c,
            &mut tfr,
            &mut chr,
            allow_weak,
            &mut wr,
            tone_freq,
            toneishness,
            &mut tmp,
        );
        assert_eq!(
            (ir, tfr.to_bits(), chr, wr),
            (ic, tfc.to_bits(), chc, wc),
            "transient_analysis {it}"
        );
    }
    // tone_lpc directly (including failures and every delay)
    for it in 0..2000 {
        let len = rng.range_i32(40, 600);
        let delay = 1 << rng.range_i32(0, 4);
        if len <= 2 * delay {
            continue;
        }
        let x = analysis_input(&mut rng, len as usize, 1);
        let init = [rng.f32_sym(), rng.f32_sym()];
        let (fc, lc) = oc::tone_lpc(&x, len, delay, init);
        let mut lr = init;
        let fr = ce::tone_lpc(&x, len, delay, &mut lr);
        assert_eq!(fr, fc, "tone_lpc fail {it}");
        assert_bits_eq_f32(&format!("tone_lpc {it}"), &lr, &lc);
    }
}

#[test]
fn preemphasis_and_mdcts() {
    let mut rng = Rng::new(0x3dc7);
    for it in 0..1500 {
        let fs = mode_rates()[rng.range_i32(0, mode_rates().len() as i32 - 1) as usize];
        let m = std_mode(fs);
        let cc = rng.range_i32(1, 2);
        let c = if cc == 2 && rng.range_i32(0, 2) == 0 {
            1
        } else {
            cc
        };
        let lm = rng.range_i32(0, m.max_lm);
        let n = m.short_mdct_size << lm;
        let ups = [1, 1, 2, 3, 4, 6][rng.range_i32(0, 5) as usize];
        let ups = if n % ups == 0 { ups } else { 1 };
        let clip = rng.range_i32(0, 1) == 1;
        // pre-emphasis
        let kind = rng.range_i32(0, SIGNAL_KINDS as i32 - 1) as usize;
        let pcm = gen_signal(kind, (n / ups) as usize, cc as usize, 48000, rng.next_u64());
        let ov = m.overlap as usize;
        let mut in_c = vec![0.0f32; cc as usize * (n as usize + ov)];
        let mut in_r = in_c.clone();
        for ch in 0..cc as usize {
            let mem0 = 1000.0 * rng.f32_sym();
            let (mut mc, mut mr) = (mem0, mem0);
            let off = ch * (n as usize + ov) + ov;
            oc::preemphasis(
                &pcm[ch..],
                &mut in_c[off..],
                n,
                cc,
                ups,
                &m.preemph,
                &mut mc,
                clip,
            );
            ce::celt_preemphasis(
                &pcm[ch..],
                &mut in_r[off..],
                n,
                cc,
                ups,
                &m.preemph,
                &mut mr,
                clip,
            );
            assert_eq!(mr.to_bits(), mc.to_bits(), "preemph mem {it}");
            for i in 0..ov {
                let v = 3000.0 * rng.f32_sym();
                in_c[ch * (n as usize + ov) + i] = v;
                in_r[ch * (n as usize + ov) + i] = v;
            }
        }
        assert_bits_eq_f32(&format!("preemphasis {it}"), &in_r, &in_c);
        // MDCTs
        let short_blocks = if lm > 0 && rng.range_i32(0, 1) == 1 {
            1 << lm
        } else {
            0
        };
        let mut fc = vec![0.0f32; cc as usize * n as usize];
        let mut fr = fc.clone();
        oc::compute_mdcts(fs, short_blocks, &in_c, &mut fc, c, cc, lm, ups);
        ce::compute_mdcts(m, short_blocks, &in_r, &mut fr, c, cc, lm, ups);
        assert_bits_eq_f32(&format!("compute_mdcts {it}"), &fr, &fc);
    }
}

/// Random normalised-band-like spectrum (`c` channels of `n`).
fn rand_spectrum(rng: &mut Rng, m: &CeltMode, lm: i32, c: usize) -> Vec<f32> {
    let n = (m.short_mdct_size << lm) as usize;
    let mut x = vec![0.0f32; c * n];
    for ch in 0..c {
        for b in 0..m.nb_ebands as usize {
            let lo = (m.e_bands[b] as usize) << lm;
            let hi = (m.e_bands[b + 1] as usize) << lm;
            let peaky = rng.range_i32(0, 2) == 0;
            let mut e = 0.0f32;
            for j in lo..hi {
                let v = if peaky && rng.range_i32(0, 3) != 0 {
                    0.0
                } else {
                    rng.f32_sym()
                };
                x[ch * n + j] = v;
                e += v * v;
            }
            let g = if e > 0.0 { 1.0 / e.sqrt() } else { 0.0 };
            for j in lo..hi {
                x[ch * n + j] *= g;
            }
        }
    }
    if c == 2 && rng.range_i32(0, 2) == 0 {
        // Correlated stereo.
        for j in 0..n {
            x[n + j] = 0.9 * x[j] + 0.1 * x[n + j];
        }
    }
    x
}

#[test]
fn tf_trim_stereo() {
    let mut rng = Rng::new(0x7f7f);
    let mut metric = vec![0i32; 64];
    let mut p0 = vec![0i32; 64];
    let mut p1 = vec![0i32; 64];
    let mut t0 = vec![0.0f32; 512];
    let mut t1 = vec![0.0f32; 512];
    for it in 0..3000 {
        let fs = mode_rates()[rng.range_i32(0, mode_rates().len() as i32 - 1) as usize];
        let m = std_mode(fs);
        let lm = rng.range_i32(0, m.max_lm);
        let c = rng.range_i32(1, 2);
        let n0 = m.short_mdct_size << lm;
        let x = rand_spectrum(&mut rng, m, lm, c as usize);
        let nb = m.nb_ebands as usize;
        // tf_analysis
        let len = rng.range_i32(13, m.eff_ebands);
        let is_t = rng.range_i32(0, 1) == 1;
        let lambda = rng.range_i32(80, 1500);
        let tf_est = rng.f32_sym().abs();
        let tf_chan = rng.range_i32(0, c - 1);
        let importance: Vec<i32> = (0..nb).map(|_| rng.range_i32(0, 250)).collect();
        let mut rc = vec![0i32; nb];
        let mut rr = vec![0i32; nb];
        let sc = oc::tf_analysis(
            fs,
            len,
            is_t,
            &mut rc,
            lambda,
            &x,
            n0,
            lm,
            tf_est,
            tf_chan,
            &importance,
        );
        let sr = ce::tf_analysis(
            m,
            len,
            is_t,
            &mut rr,
            lambda,
            &x,
            n0,
            lm,
            tf_est,
            tf_chan,
            &importance,
            &mut metric,
            &mut p0,
            &mut p1,
            &mut t0,
            &mut t1,
        );
        assert_eq!(sr, sc, "tf_select {it}");
        assert_slice_eq(
            &format!("tf_res {it}"),
            &rr[..len as usize],
            &rc[..len as usize],
        );
        // tf_encode on a coder with random prior content and small budgets
        let end = len;
        let start = if rng.range_i32(0, 3) == 0 {
            17.min(end - 1)
        } else {
            0
        };
        let size = rng.range_i32(2, 40) as usize;
        let ops: Vec<[i32; 3]> = (0..rng.range_i32(0, (size as i32) * 6))
            .map(|_| [0, rng.range_i32(0, 1), rng.range_i32(1, 4)])
            .collect();
        let mut tfr_c = rc.clone();
        let mut tfr_r = rc.clone();
        let tf_select = rng.range_i32(0, 1);
        let mut bc = vec![0u8; size];
        let sc = oc::tf_encode(start, end, is_t, &mut tfr_c, lm, tf_select, &mut bc, &ops);
        let mut br = vec![0u8; size];
        let sr = {
            let mut e = EcEnc::new(&mut br);
            apply_ops(&mut e, &ops);
            ce::tf_encode(start, end, is_t, &mut tfr_r, lm, tf_select, &mut e);
            e.done();
            snap(&e)
        };
        assert_eq!(sr, sc, "tf_encode coder {it}");
        assert_slice_eq(&format!("tf_encode bytes {it}"), &br, &bc);
        assert_slice_eq(&format!("tf_encode tf_res {it}"), &tfr_r, &tfr_c);
        // alloc_trim_analysis
        let band_log_e: Vec<f32> = (0..c as usize * nb)
            .map(|_| 8.0 * rng.f32_sym() + 4.0)
            .collect();
        let end = rng.range_i32(13, m.eff_ebands);
        let an_valid = rng.range_i32(0, 1) == 1;
        let slope = 0.3 * rng.f32_sym();
        let ss0 = rng.f32_sym();
        let intensity = rng.range_i32(0, end);
        let surround_trim = if rng.range_i32(0, 2) == 0 {
            rng.f32_sym()
        } else {
            0.0
        };
        let equiv = rng.range_i32(5000, 600_000);
        let (mut ssc, mut ssr) = (ss0, ss0);
        let ac = oc::alloc_trim_analysis(
            fs,
            &x,
            &band_log_e,
            end,
            lm,
            c,
            n0,
            an_valid,
            slope,
            &mut ssc,
            tf_est,
            intensity,
            surround_trim,
            equiv,
        );
        let an = AnalysisInfo {
            valid: i32::from(an_valid),
            tonality_slope: slope,
            ..Default::default()
        };
        let ar = ce::alloc_trim_analysis(
            m,
            &x,
            &band_log_e,
            end,
            lm,
            c,
            n0,
            &an,
            &mut ssr,
            tf_est,
            intensity,
            surround_trim,
            equiv,
        );
        assert_eq!((ar, ssr.to_bits()), (ac, ssc.to_bits()), "alloc_trim {it}");
        // stereo_analysis
        if c == 2 {
            assert_eq!(
                ce::stereo_analysis(m, &x, lm, n0),
                oc::stereo_analysis(fs, &x, lm, n0),
                "stereo_analysis {it}"
            );
        }
    }
}

#[test]
fn medians_patch_dynalloc_vbr() {
    let mut rng = Rng::new(0xd1a1);
    for it in 0..20000 {
        let q = |rng: &mut Rng| (rng.range_i32(-3, 3) as f32) * 0.5;
        let x5: [f32; 5] = std::array::from_fn(|_| {
            if it % 2 == 0 {
                q(&mut rng)
            } else {
                rng.f32_sym()
            }
        });
        assert_eq!(
            ce::median_of_5(&x5).to_bits(),
            oc::median_of_5(&x5).to_bits(),
            "median5 {x5:?}"
        );
        let x3: [f32; 3] = [x5[0], x5[1], x5[2]];
        assert_eq!(
            ce::median_of_3(&x3).to_bits(),
            oc::median_of_3(&x3).to_bits(),
            "median3 {x3:?}"
        );
    }
    let mut scratch = DynallocScratch::new(64, 2);
    for it in 0..4000 {
        let fs = mode_rates()[rng.range_i32(0, mode_rates().len() as i32 - 1) as usize];
        let m = std_mode(fs);
        let nb = m.nb_ebands;
        let nbu = nb as usize;
        let c = rng.range_i32(1, 2);
        let lm = rng.range_i32(0, m.max_lm);
        let hybrid = rng.range_i32(0, 4) == 0;
        let start = if hybrid { 17 } else { 0 };
        let end = rng.range_i32(if hybrid { 19 } else { 13 }, m.eff_ebands);
        let mk = |rng: &mut Rng, base: f32, amp: f32| -> Vec<f32> {
            (0..2 * nbu).map(|_| base + amp * rng.f32_sym()).collect()
        };
        let ble = mk(&mut rng, 3.0, 10.0);
        let ble2: Vec<f32> = ble.iter().map(|&v| v + 0.5 * rng.f32_sym()).collect();
        let old = mk(&mut rng, 2.0, 12.0);
        // patch_transient_decision
        if !hybrid && end > 3 {
            assert_eq!(
                ce::patch_transient_decision(&ble, &old, nb, start, end, c),
                oc::patch_transient_decision(&ble, &old, nb, start, end, c),
                "patch_transient {it}"
            );
        }
        let lsb = rng.range_i32(8, 24);
        let is_t = rng.range_i32(0, 1) == 1;
        let vbr = rng.range_i32(0, 1) == 1;
        let cvbr = rng.range_i32(0, 1) == 1;
        let eff = rng.range_i32(2, 1275);
        let lfe = rng.range_i32(0, 8) == 0;
        let sd: Vec<f32> = (0..2 * nbu)
            .map(|_| {
                if rng.range_i32(0, 3) == 0 {
                    2.0 * rng.f32_sym().abs()
                } else {
                    0.0
                }
            })
            .collect();
        let an_valid = rng.range_i32(0, 1) == 1;
        let leak: [u8; 19] = std::array::from_fn(|_| rng.range_i32(0, 255) as u8);
        let (tone_freq, toneish) = if rng.range_i32(0, 2) == 0 {
            (0.6 * rng.f32_sym().abs(), 0.97 + 0.03 * rng.f32_sym().abs())
        } else {
            (-1.0, 0.0)
        };
        let mut oc_ = vec![-7i32; nbu];
        let mut or = vec![-7i32; nbu];
        let mut ic = vec![-7i32; nbu];
        let mut ir = vec![-7i32; nbu];
        let mut swc = vec![-7i32; nbu];
        let mut swr = vec![-7i32; nbu];
        let (mdc, tbc) = oc::dynalloc_analysis(
            fs, &ble, &ble2, &old, start, end, c, &mut oc_, lsb, is_t, vbr, cvbr, lm, eff, lfe,
            &sd, an_valid, &leak, &mut ic, &mut swc, tone_freq, toneish,
        );
        let an = AnalysisInfo {
            valid: i32::from(an_valid),
            leak_boost: leak,
            ..Default::default()
        };
        let mut tbr = 0;
        let qs = if fs == 96000 { 2 } else { 1 };
        let mdr = ce::dynalloc_analysis(
            &ble,
            &ble2,
            &old,
            nb,
            start,
            end,
            c,
            &mut or,
            lsb,
            &m.log_n,
            is_t,
            vbr,
            cvbr,
            &m.e_bands,
            lm,
            eff,
            &mut tbr,
            lfe,
            &sd,
            &an,
            &mut ir,
            &mut swr,
            tone_freq,
            toneish,
            qs,
            &mut scratch,
        );
        assert_eq!(
            (mdr.to_bits(), tbr),
            (mdc.to_bits(), tbc),
            "dynalloc maxDepth/tot_boost {it}"
        );
        assert_slice_eq(&format!("dynalloc offsets {it}"), &or, &oc_);
        let e = end as usize;
        let s = start as usize;
        assert_slice_eq(&format!("dynalloc importance {it}"), &ir[s..e], &ic[s..e]);
        assert_slice_eq(
            &format!("dynalloc spread_weight {it}"),
            &swr[..e],
            &swc[..e],
        );
        // compute_vbr
        let base_target = rng.range_i32(0, 10000);
        let bitrate = rng.range_i32(6000, 510_000);
        let lcb = rng.range_i32(0, nb);
        let intensity = rng.range_i32(0, nb);
        let act = rng.f32_sym().abs();
        let ton = rng.f32_sym().abs();
        let ss = rng.f32_sym();
        let tb = rng.range_i32(0, 3000);
        let tfe = rng.f32_sym().abs();
        let pc = rng.range_i32(0, 1) == 1;
        let hsm = rng.range_i32(0, 3) == 0;
        let smask = -rng.f32_sym().abs();
        let tvbr = 2.0 * rng.f32_sym();
        let eq = cfg!(feature = "qext") && rng.range_i32(0, 1) == 1;
        let vc = oc::compute_vbr(
            fs,
            an_valid,
            act,
            ton,
            base_target,
            lm,
            bitrate,
            lcb,
            c,
            intensity,
            cvbr,
            ss,
            tb,
            tfe,
            pc,
            mdc,
            lfe,
            hsm,
            smask,
            tvbr,
            eq,
        );
        let an = AnalysisInfo {
            valid: i32::from(an_valid),
            activity: act,
            tonality: ton,
            ..Default::default()
        };
        let vr = ce::compute_vbr(
            m,
            &an,
            base_target,
            lm,
            bitrate,
            lcb,
            c,
            intensity,
            cvbr,
            ss,
            tb,
            tfe,
            pc,
            mdc,
            lfe,
            hsm,
            smask,
            tvbr,
            #[cfg(feature = "qext")]
            eq,
        );
        let _ = eq;
        assert_eq!(vr, vc, "compute_vbr {it}");
    }
}

// ---------------------------------------------------------------------------------------------
// Persistent encoder streams
// ---------------------------------------------------------------------------------------------

fn random_stream(p: &mut Pair, rng: &mut Rng, fs: i32, frame_size: i32) -> i32 {
    // Bitrate mode: 0 = CBR by bytes, 1 = CBR by bitrate, 2 = VBR, 3 = CVBR.
    let rate_mode = rng.range_i32(0, 3);
    let bitrate = if rng.range_i32(0, 3) == 0 {
        rng.range_i32(6000, 40000)
    } else {
        rng.range_i32(6000, 510_000)
    } * if p.channels == 2 && rng.range_i32(0, 1) == 1 {
        2
    } else {
        1
    };
    let max_bytes = (bitrate as i64 * frame_size as i64 / (8 * fs as i64)).clamp(2, 1275) as i32;
    let nb = match rate_mode {
        0 => {
            p.ctl(OPUS_SET_VBR, 0);
            p.ctl(OPUS_SET_BITRATE, -1);
            max_bytes
        }
        1 => {
            p.ctl(OPUS_SET_VBR, 0);
            p.ctl(OPUS_SET_BITRATE, bitrate);
            rng.range_i32(max_bytes, 1275)
        }
        _ => {
            p.ctl(OPUS_SET_VBR, 1);
            p.ctl(OPUS_SET_VBR_CONSTRAINT, i32::from(rate_mode == 3));
            p.ctl(OPUS_SET_BITRATE, bitrate);
            if rng.range_i32(0, 3) == 0 {
                rng.range_i32(2, 1275)
            } else {
                1275
            }
        }
    };
    p.ctl(OPUS_SET_COMPLEXITY, rng.range_i32(0, 10));
    p.ctl(
        CELT_SET_PREDICTION,
        [2, 2, 2, 1, 0][rng.range_i32(0, 4) as usize],
    );
    p.ctl(
        OPUS_SET_PACKET_LOSS_PERC,
        [0, 0, 1, 3, 5, 9, 25, 100][rng.range_i32(0, 7) as usize],
    );
    p.ctl(
        OPUS_SET_LSB_DEPTH,
        if rng.range_i32(0, 2) == 0 {
            rng.range_i32(8, 24)
        } else {
            24
        },
    );
    p.ctl(OPUS_SET_PHASE_INVERSION_DISABLED, rng.range_i32(0, 1));
    if rng.range_i32(0, 3) == 0 {
        p.ctl(
            CELT_SET_END_BAND,
            [13, 17, 19, 21][rng.range_i32(0, 3) as usize],
        );
    }
    if p.channels == 2 && rng.range_i32(0, 4) == 0 {
        p.ctl(CELT_SET_CHANNELS, 1);
    }
    if rng.range_i32(0, 12) == 0 {
        p.ctl(OPUS_SET_LFE, 1);
    }
    if rng.range_i32(0, 2) == 0 {
        p.set_analysis(rng);
    }
    nb
}

fn random_mask(rng: &mut Rng) -> Vec<f32> {
    (0..42)
        .map(|_| match rng.range_i32(0, 3) {
            0 => -3.0 * rng.f32_sym().abs(),
            1 => 0.5 * rng.f32_sym(),
            _ => -0.5 - rng.f32_sym().abs(),
        })
        .collect()
}

/// Runs `nstreams` random streams through both encoders.
fn run_streams(seed: u64, nstreams: usize, rates: &[i32], frames: usize) {
    let mut rng = Rng::new(seed);
    for s in 0..nstreams {
        let fs = rates[rng.range_i32(0, rates.len() as i32 - 1) as usize];
        let channels = rng.range_i32(1, 2);
        let lm = rng.range_i32(0, 3);
        let ups = if fs == 96000 { 1 } else { 48000 / fs };
        let base = if fs == 96000 { 240 } else { 120 };
        let frame_size = (base << lm) / ups;
        let kind = rng.range_i32(0, SIGNAL_KINDS as i32 - 1) as usize;
        let what = format!(
            "stream {s} (seed {seed:#x}) fs={fs} ch={channels} fsz={frame_size} kind={kind}"
        );
        let mut p = Pair::new(fs, channels, &what);
        #[allow(unused_mut, reason = "only changed with qext")]
        let mut nb = random_stream(&mut p, &mut rng, fs, frame_size);
        // With QEXT the encoder writes the TOC byte before the payload (C `compressed[-1]`),
        // so such streams go through a range coder with a TOC slot, as in the Opus encoder.
        #[allow(unused_mut, reason = "only changed with qext")]
        let mut use_ec = rng.range_i32(0, 5) == 0;
        #[cfg(feature = "qext")]
        if rng.range_i32(0, 2) == 0 {
            use_ec = true;
            p.ctl(OPUS_SET_QEXT, 1);
            if rng.range_i32(0, 1) == 1 {
                nb = rng.range_i32(nb, 3825);
            }
        }
        let use_mask = rng.range_i32(0, 5) == 0;
        let total = frames * frame_size as usize;
        let pcm = gen_signal(kind, total, channels as usize, fs as u32, rng.next_u64());
        for f in 0..frames {
            let ctx = format!("frame {f}");
            if use_mask {
                let m = random_mask(&mut rng);
                p.set_energy_mask(if rng.range_i32(0, 6) == 0 {
                    None
                } else {
                    Some(&m)
                });
            }
            // Occasional mid-stream CTL changes and resets.
            match rng.range_i32(0, 40) {
                0 => p.ctl(OPUS_RESET_STATE, 0),
                1 => p.ctl(OPUS_SET_COMPLEXITY, rng.range_i32(0, 10)),
                2 => p.ctl(OPUS_SET_BITRATE, rng.range_i32(6000, 300_000)),
                3 => p.set_analysis(&mut rng),
                4 => p.ctl(CELT_SET_PREDICTION, rng.range_i32(0, 2)),
                5 => p.ctl(OPUS_SET_PACKET_LOSS_PERC, rng.range_i32(0, 20)),
                _ => {}
            }
            let off = f * frame_size as usize * channels as usize;
            let frame = &pcm[off..off + frame_size as usize * channels as usize];
            if use_ec {
                p.encode_with_ec(frame, frame_size, nb as usize + 1, nb, &[], &ctx);
            } else {
                p.encode(frame, frame_size, nb, &ctx);
            }
            p.check_state(&ctx);
        }
    }
}

#[test]
fn streams_48k() {
    run_streams(0xce11_0001, 1000, &[48000], 40);
}

#[test]
fn streams_upsampled_rates() {
    run_streams(0xce11_0002, 500, &[24000, 16000, 12000, 8000], 30);
}

#[test]
fn streams_long() {
    // Fewer, longer streams for VBR reservoir / drift evolution.
    run_streams(0xce11_0003, 12, &[48000], 400);
    // VBR past the 970-frame averaging window (vbr_count saturation).
    for (s, cvbr) in [(0u64, 1), (1, 0)] {
        let mut p = Pair::new(48000, 2, &format!("vbr saturation {s}"));
        p.ctl(OPUS_SET_VBR, 1);
        p.ctl(OPUS_SET_VBR_CONSTRAINT, cvbr);
        p.ctl(OPUS_SET_BITRATE, 48000);
        let pcm = gen_signal(13, 1100 * 240, 2, 48000, 99 + s);
        for f in 0..1100 {
            let fr = &pcm[f * 480..(f + 1) * 480];
            p.encode(fr, 240, 1275, &format!("frame {f}"));
            if f % 50 == 0 || f > 960 {
                p.check_state(&format!("frame {f}"));
            }
        }
    }
}

/// Exhaustive random sweep (about 10x the default streams); run with `--ignored`.
#[test]
#[ignore = "exhaustive sweep (minutes); run explicitly with --ignored"]
fn streams_exhaustive() {
    let mut rates = vec![48000, 48000, 24000, 16000, 12000, 8000];
    if cfg!(feature = "qext") {
        rates.push(96000);
    }
    run_streams(0xce11_e000, 15000, &rates, 40);
}

#[cfg(feature = "qext")]
#[test]
fn streams_96k_qext() {
    run_streams(0xce11_0096, 450, &[96000], 30);
}

/// Stereo custom modes with `end > effEBands`: C reads past its `X` buffer (undefined
/// behaviour), so this is Rust-only: the port must stay in bounds and produce packets.
#[cfg(feature = "custom-modes")]
#[test]
fn custom_end_beyond_eff_bands_rust_only() {
    for &(fs, fsz) in &[(48000, 256), (16000, 320), (44100, 1024)] {
        let Ok(mode) = opusorus::celt::modes::opus_custom_mode_create_custom(fs, fsz) else {
            continue;
        };
        let nb = mode.nb_ebands;
        let mut r = CeltEncoder::opus_custom_encoder_create(mode, 2).unwrap();
        r.set_signalling(0);
        r.set_end_band(nb).unwrap();
        r.set_complexity(10).unwrap();
        let pcm = gen_signal(0, 20 * fsz as usize, 2, fs as u32, 5);
        for f in 0..20 {
            let mut out = vec![0u8; 400];
            let fr = &pcm[f * 2 * fsz as usize..(f + 1) * 2 * fsz as usize];
            assert!(r.opus_custom_encode_float(fr, fsz, &mut out, 400) > 0);
        }
    }
}

#[test]
fn init_and_sizes() {
    assert!(ce::celt_encoder_get_size(1) > 0);
    assert!(ce::celt_encoder_get_size(2) > ce::celt_encoder_get_size(1));
    for fs in [44100, 0, -1, 96001] {
        assert!(CeltEncoder::celt_encoder_init(fs, 1).is_err(), "fs={fs}");
    }
    for ch in [0, 3, -1] {
        assert!(
            CeltEncoder::celt_encoder_init(48000, ch).is_err(),
            "ch={ch}"
        );
    }
    let e = CeltEncoder::celt_encoder_init(16000, 2).unwrap();
    assert_eq!(e.upsample, 3);
    assert_eq!(e.mode().fs, 48000);
}

#[test]
fn hybrid_shared_coder() {
    let mut rng = Rng::new(0x4b1d);
    for s in 0..600 {
        let channels = rng.range_i32(1, 2);
        let lm = rng.range_i32(2, 3);
        let frame_size = 120 << lm;
        let kind = [1, 1, 0, 10, 13, 4][rng.range_i32(0, 5) as usize];
        let what = format!("hybrid {s} ch={channels} fsz={frame_size} kind={kind}");
        let mut p = Pair::new(48000, channels, &what);
        p.ctl(CELT_SET_START_BAND, 17);
        p.ctl(CELT_SET_END_BAND, [19, 21][rng.range_i32(0, 1) as usize]);
        p.ctl(OPUS_SET_COMPLEXITY, rng.range_i32(0, 10));
        let vbr = rng.range_i32(0, 1);
        p.ctl(OPUS_SET_VBR, vbr);
        if vbr != 0 {
            p.ctl(OPUS_SET_VBR_CONSTRAINT, 0);
            p.ctl(OPUS_SET_BITRATE, rng.range_i32(2000, 60000));
        } else {
            p.ctl(OPUS_SET_BITRATE, -1);
        }
        if rng.range_i32(0, 1) == 1 {
            p.set_analysis(&mut rng);
        }
        let frames = 40;
        let total_rate = rng.range_i32(12000, 64000);
        let pcm = gen_signal(
            kind,
            frames * frame_size as usize,
            channels as usize,
            48000,
            rng.next_u64(),
        );
        for f in 0..frames {
            let ctx = format!("frame {f}");
            p.set_silk_info(
                rng.range_i32(0, 2),
                [50, 99, 100, 101, 150, 200][rng.range_i32(0, 5) as usize],
            );
            let nb = (total_rate * frame_size / (8 * 48000)).clamp(8, 1275);
            let buf_size = nb as usize + 1 + rng.range_i32(0, 20) as usize;
            // Simulated SILK payload: ~40% of the bytes.
            let silk_bits = rng.range_i32(8, (nb * 8 * 2 / 5).max(9));
            let mut ops = Vec::new();
            let mut used = 0;
            while used < silk_bits {
                match rng.range_i32(0, 2) {
                    0 => {
                        let lp = rng.range_i32(1, 6);
                        ops.push([0, i32::from(rng.range_i32(0, (1 << lp) - 1) == 0), lp]);
                        used += 1;
                    }
                    1 => {
                        let ft = rng.range_i32(2, 300);
                        ops.push([1, rng.range_i32(0, ft - 1), ft]);
                        used += 9;
                    }
                    _ => {
                        let nbits = rng.range_i32(1, 8);
                        ops.push([2, rng.range_i32(0, (1 << nbits) - 1), nbits]);
                        used += nbits;
                    }
                }
            }
            let off = f * frame_size as usize * channels as usize;
            let frame = &pcm[off..off + frame_size as usize * channels as usize];
            p.encode_with_ec(frame, frame_size, buf_size, nb, &ops, &ctx);
            p.check_state(&ctx);
        }
    }
}

#[test]
fn edge_cases_and_ctls() {
    // CTL argument validation (return codes compared with C).
    let mut p = Pair::new(48000, 2, "ctls");
    for &(req, vals) in &[
        (OPUS_SET_COMPLEXITY, &[-1, 0, 10, 11][..]),
        (CELT_SET_START_BAND, &[-1, 0, 20, 21][..]),
        (CELT_SET_END_BAND, &[0, 1, 21, 22][..]),
        (CELT_SET_PREDICTION, &[-1, 0, 1, 2, 3][..]),
        (OPUS_SET_PACKET_LOSS_PERC, &[-1, 0, 100, 101][..]),
        (OPUS_SET_BITRATE, &[-1, 0, 500, 501, 2_000_000][..]),
        (CELT_SET_CHANNELS, &[0, 1, 2, 3][..]),
        (OPUS_SET_LSB_DEPTH, &[7, 8, 24, 25][..]),
        (OPUS_SET_PHASE_INVERSION_DISABLED, &[-1, 0, 1, 2][..]),
    ] {
        for &v in vals {
            p.ctl(req, v);
        }
    }
    p.ctl(CELT_SET_START_BAND, 0);
    p.ctl(CELT_SET_END_BAND, 21);
    p.ctl(CELT_SET_CHANNELS, 2);
    p.check_state("after ctls");
    assert_eq!(p.c.ctl_get(OPUS_GET_LSB_DEPTH).1, p.r.lsb_depth());
    assert_eq!(
        p.c.ctl_get(OPUS_GET_PHASE_INVERSION_DISABLED).1,
        p.r.phase_inversion_disabled()
    );
    #[cfg(feature = "qext")]
    {
        p.ctl(OPUS_SET_QEXT, 2);
        p.ctl(OPUS_SET_QEXT, 1);
        assert_eq!(p.c.ctl_get(OPUS_GET_QEXT).1, p.r.qext());
        p.ctl(OPUS_SET_QEXT, 0);
    }

    // Bad arguments.
    let pcm = gen_signal(0, 960, 2, 48000, 1);
    let mut out = [0u8; 100];
    for &(fsz, nb) in &[(960, 1), (960, 0), (961, 100), (100, 100), (1920, 100)] {
        let rc = p.c.encode(&pcm, 2, fsz.min(960), &mut out, nb);
        let rr = r_encode(&mut p.r, &pcm, fsz.min(960), Some(&mut out), nb, None);
        assert_eq!(rr, rc, "bad args fsz={fsz} nb={nb}");
    }
    assert_eq!(
        r_encode(&mut p.r, &pcm[..10], 960, Some(&mut out), 100, None),
        -1
    );
    assert_eq!(r_encode(&mut p.r, &pcm, 960, None, 100, None), -1);
    p.check_state("after bad args");

    // Tiny budgets, every LM, all byte counts up to 12, both VBR modes.
    let mut rng = Rng::new(0xed9e);
    for ch in 1..=2 {
        for lm in 0..4 {
            let fsz = 120 << lm;
            for vbr in 0..3 {
                let mut p = Pair::new(48000, ch, &format!("tiny ch={ch} lm={lm} vbr={vbr}"));
                p.ctl(OPUS_SET_VBR, i32::from(vbr > 0));
                p.ctl(OPUS_SET_VBR_CONSTRAINT, i32::from(vbr == 2));
                if vbr > 0 {
                    p.ctl(OPUS_SET_BITRATE, 6000);
                }
                let pcm = gen_signal(
                    rng.range_i32(0, SIGNAL_KINDS as i32 - 1) as usize,
                    fsz * 13,
                    ch as usize,
                    48000,
                    rng.next_u64(),
                );
                for nb in 2..=14 {
                    let off = (nb - 2) as usize * fsz * ch as usize;
                    p.encode(
                        &pcm[off..off + fsz * ch as usize],
                        fsz as i32,
                        nb,
                        &format!("nb={nb}"),
                    );
                    p.check_state(&format!("nb={nb}"));
                }
            }
        }
    }

    // Silence then loud (VBR silence handling, drift not updated), lsb-depth silence.
    for &lsb in &[8, 16, 24] {
        let mut p = Pair::new(48000, 1, &format!("silence lsb={lsb}"));
        p.ctl(OPUS_SET_VBR, 1);
        p.ctl(OPUS_SET_BITRATE, 64000);
        p.ctl(OPUS_SET_LSB_DEPTH, lsb);
        let quiet = signals::noise(960 * 10, 1, 2.0f32.powi(-lsb - 1), 7);
        let loud = gen_signal(0, 960 * 10, 1, 48000, 8);
        for f in 0..20 {
            let src = if f % 10 < 5 { &quiet } else { &loud };
            let off = (f % 10) * 960;
            p.encode(&src[off..off + 960], 960, 1275, &format!("f={f}"));
            p.check_state(&format!("f={f}"));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Opus replay
// ---------------------------------------------------------------------------------------------

/// Replays one recorded C call in Rust from the recorded pre-state; compares everything.
fn replay(call: &oc::CeltCall, what: &str) {
    let mut r = rust_from_state(&call.pre);
    cmp_state(&format!("{what}: loaded pre-state"), &dump(&r), &call.pre);
    let ret = match (&call.compressed, &call.enc) {
        (Some((before, after)), None) => {
            let mut out = before.clone();
            let ret = r_encode(
                &mut r,
                &call.pcm,
                call.frame_size,
                Some(&mut out),
                call.nb_compressed_bytes,
                None,
            );
            assert_slice_eq(&format!("{what}: compressed"), &out, after);
            ret
        }
        (None, Some(er)) => {
            let mut buf = er.buf_before.clone();
            let (toc, rest) = buf.split_at_mut(1);
            let base = rest.as_ptr() as usize;
            let mut e = make_enc(rest, &er.before);
            #[cfg(feature = "qext")]
            let ret = r.celt_encode_with_ec(
                &call.pcm,
                call.frame_size,
                None,
                call.nb_compressed_bytes,
                Some(&mut e),
                Some(&mut toc[0]),
            );
            #[cfg(not(feature = "qext"))]
            let ret = {
                let _ = &toc;
                r.celt_encode_with_ec(
                    &call.pcm,
                    call.frame_size,
                    None,
                    call.nb_compressed_bytes,
                    Some(&mut e),
                )
            };
            let shift = (e.buf.as_ptr() as usize - base) as i32;
            let s = snap(&e);
            let _ = e; // end the coder's borrow of the buffer
            assert_eq!(s, er.after, "{what}: range coder state");
            assert_eq!(shift, er.buf_shift, "{what}: buffer shift");
            assert_slice_eq(&format!("{what}: range coder buffer"), &buf, &er.buf_after);
            ret
        }
        _ => panic!("{what}: unexpected call shape"),
    };
    assert_eq!(ret, call.ret, "{what}: return value");
    cmp_state(&format!("{what}: post-state"), &dump(&r), &call.post);
}

struct OpusCfg {
    fs: i32,
    channels: i32,
    app: i32,
    frame_ms_x2: i32, // frame duration in units of 1.25 ms... (2.5 ms = 2)
}

/// Runs an Opus stream through the recording C encoder and replays all CELT calls.
fn run_opus(
    seed: u64,
    cfg: &OpusCfg,
    frames: usize,
    ctls: &[(i32, i32)],
    bitrate_walk: bool,
    qext: bool,
) -> usize {
    let mut rng = Rng::new(seed);
    let mut enc = oc::OpusRec::new(cfg.fs, cfg.channels, cfg.app).unwrap();
    for &(req, v) in ctls {
        enc.ctl(req, v);
    }
    #[cfg(feature = "qext")]
    if qext {
        enc.ctl(OPUS_SET_QEXT, 1);
    }
    let _ = qext;
    let frame_size = cfg.fs / 800 * cfg.frame_ms_x2;
    let kind = rng.range_i32(0, SIGNAL_KINDS as i32 - 1) as usize;
    let pcm = gen_signal(
        kind,
        frames * frame_size as usize,
        cfg.channels as usize,
        cfg.fs as u32,
        rng.next_u64(),
    );
    let mut out = vec![0u8; 4000];
    let mut ncalls = 0;
    for f in 0..frames {
        if bitrate_walk && rng.range_i32(0, 5) == 0 {
            enc.ctl(OPUS_SET_BITRATE, rng.range_i32(6000, 160_000));
        }
        let off = f * frame_size as usize * cfg.channels as usize;
        let frame = &pcm[off..off + frame_size as usize * cfg.channels as usize];
        let max = if qext { 4000 } else { 1276 };
        let (ret, calls) = enc.encode_float(frame, frame_size, &mut out[..max]);
        assert!(ret > 0, "opus encode failed: {ret}");
        for (i, call) in calls.iter().enumerate() {
            replay(
                call,
                &format!(
                    "opus fs={} ch={} app={} fsz={frame_size} kind={kind} seed={seed:#x} frame {f} call {i}",
                    cfg.fs, cfg.channels, cfg.app
                ),
            );
        }
        ncalls += calls.len();
    }
    ncalls
}

#[test]
fn opus_replay_restricted_celt() {
    let mut rng = Rng::new(0x0905);
    let mut n = 0;
    for s in 0..500 {
        let fs = [48000, 48000, 24000, 16000, 12000, 8000][rng.range_i32(0, 5) as usize];
        let channels = rng.range_i32(1, 2);
        let frame_ms_x2 = [2, 4, 8, 16, 32, 48][rng.range_i32(0, 5) as usize];
        let cfg = OpusCfg {
            fs,
            channels,
            app: [APP_RESTRICTED_CELT, APP_RESTRICTED_LOWDELAY][rng.range_i32(0, 1) as usize],
            frame_ms_x2,
        };
        let vbr = rng.range_i32(0, 2);
        let ctls = vec![
            (OPUS_SET_BITRATE, rng.range_i32(6000, 510_000)),
            (OPUS_SET_VBR, i32::from(vbr > 0)),
            (OPUS_SET_VBR_CONSTRAINT, i32::from(vbr == 1)),
            (OPUS_SET_COMPLEXITY, rng.range_i32(0, 10)),
            (
                OPUS_SET_PACKET_LOSS_PERC,
                [0, 0, 5, 20][rng.range_i32(0, 3) as usize],
            ),
            (OPUS_SET_LSB_DEPTH, rng.range_i32(8, 24)),
            (
                OPUS_SET_PREDICTION_DISABLED,
                i32::from(rng.range_i32(0, 4) == 0),
            ),
            (OPUS_SET_LFE, i32::from(rng.range_i32(0, 10) == 0)),
            (OPUS_SET_PHASE_INVERSION_DISABLED, rng.range_i32(0, 1)),
            (
                OPUS_SET_MAX_BANDWIDTH,
                [1101, 1103, 1104, 1105, 1105][rng.range_i32(0, 4) as usize],
            ),
        ];
        n += run_opus(
            0x0905_0000 + s,
            &cfg,
            12,
            &ctls,
            rng.range_i32(0, 2) == 0,
            false,
        );
    }
    assert!(n > 300);
}

#[test]
fn opus_replay_hybrid_and_transitions() {
    let mut rng = Rng::new(0x4719);
    let mut n = 0;
    for s in 0..240 {
        let channels = rng.range_i32(1, 2);
        let cfg = OpusCfg {
            fs: [48000, 48000, 24000][rng.range_i32(0, 2) as usize],
            channels,
            app: [APP_AUDIO, APP_VOIP][rng.range_i32(0, 1) as usize],
            frame_ms_x2: [8, 16, 16, 32, 48][rng.range_i32(0, 4) as usize],
        };
        let vbr = rng.range_i32(0, 2);
        let ctls = vec![
            (OPUS_SET_BITRATE, rng.range_i32(12000, 48000) * channels),
            (OPUS_SET_VBR, i32::from(vbr > 0)),
            (OPUS_SET_VBR_CONSTRAINT, i32::from(vbr == 1)),
            (OPUS_SET_COMPLEXITY, rng.range_i32(0, 10)),
            (OPUS_SET_INBAND_FEC, rng.range_i32(0, 1)),
            (
                OPUS_SET_PACKET_LOSS_PERC,
                [0, 5, 20][rng.range_i32(0, 2) as usize],
            ),
            (OPUS_SET_DTX, i32::from(rng.range_i32(0, 5) == 0)),
            (
                OPUS_SET_SIGNAL,
                [-1000, 3001, 3002][rng.range_i32(0, 2) as usize],
            ),
            (
                OPUS_SET_FORCE_CHANNELS,
                [-1000, 1][rng.range_i32(0, 1) as usize],
            ),
        ];
        // Bitrate walks force SILK <-> hybrid <-> CELT transitions (redundancy + prefill).
        n += run_opus(0x4719_0000 + s, &cfg, 40, &ctls, true, false);
    }
    // Explicit bandwidth/mode switching.
    for s in 0..6 {
        let cfg = OpusCfg {
            fs: 48000,
            channels: 2,
            app: APP_AUDIO,
            frame_ms_x2: 16,
        };
        let ctls = vec![
            (OPUS_SET_BITRATE, 32000),
            (OPUS_SET_BANDWIDTH, [1104, 1105][s % 2]),
            (OPUS_SET_EXPERT_FRAME_DURATION, 5004),
            (OPUS_SET_COMPLEXITY, 10),
        ];
        n += run_opus(0x4719_1000 + s as u64, &cfg, 50, &ctls, true, false);
    }
    assert!(n > 500);
}

#[cfg(feature = "qext")]
#[test]
fn opus_replay_qext() {
    let mut rng = Rng::new(0x9e47);
    for s in 0..240 {
        let fs = [96000, 48000][s % 2];
        let channels = rng.range_i32(1, 2);
        let cfg = OpusCfg {
            fs,
            channels,
            app: [APP_RESTRICTED_CELT, APP_AUDIO][rng.range_i32(0, 1) as usize],
            frame_ms_x2: [4, 8, 16][rng.range_i32(0, 2) as usize],
        };
        let vbr = rng.range_i32(0, 2);
        let ctls = vec![
            (OPUS_SET_BITRATE, rng.range_i32(96_000, 510_000) * channels),
            (OPUS_SET_VBR, i32::from(vbr > 0)),
            (OPUS_SET_VBR_CONSTRAINT, i32::from(vbr == 1)),
            (OPUS_SET_COMPLEXITY, rng.range_i32(0, 10)),
        ];
        run_opus(0x9e47_0000 + s as u64, &cfg, 12, &ctls, false, true);
    }
}

// ---------------------------------------------------------------------------------------------
// Custom modes (Opus custom API)
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "custom-modes")]
#[test]
fn custom_api() {
    let mut rng = Rng::new(0xc057);
    // (fs, frame_size): the standard mode (TOC conversion) and non-standard modes.
    let mut configs = vec![
        (48000, 960),
        (48000, 480),
        (44100, 882),
        (44100, 1024),
        (32000, 640),
        (48000, 256),
        (24000, 400),
        (16000, 320),
        (48000, 720),
        (48000, 1024),
    ];
    if cfg!(feature = "qext") {
        configs.extend([(48000, 2048), (96000, 1920), (96000, 1440), (48000, 1440)]);
    }
    for (ci, &(fs, fsz)) in configs.iter().enumerate() {
        for ch in 1..=2 {
            let what = format!("custom fs={fs} fsz={fsz} ch={ch}");
            let rmode = opusorus::celt::modes::opus_custom_mode_create_custom(fs, fsz);
            let Ok(mut c) = oc::CeltEnc::new_custom(fs, fsz, ch) else {
                assert!(
                    rmode.is_err(),
                    "{what}: C rejects the mode, Rust accepts it"
                );
                continue;
            };
            let mut r = CeltEncoder::opus_custom_encoder_create(rmode.unwrap(), ch).unwrap();
            cmp_state(&what, &dump(&r), &c.state());
            let vbr = rng.range_i32(0, 1);
            let signalling = i32::from(!(ci + ch as usize).is_multiple_of(3));
            let nb_bands = r.mode().nb_ebands;
            for (req, v) in [
                (OPUS_SET_VBR, vbr),
                (
                    OPUS_SET_BITRATE,
                    if vbr != 0 {
                        rng.range_i32(16000, 256_000)
                    } else {
                        -1
                    },
                ),
                (OPUS_SET_COMPLEXITY, rng.range_i32(0, 10)),
                (CELT_SET_INPUT_CLIPPING, rng.range_i32(0, 1)),
                (CELT_SET_SIGNALLING, signalling),
                // All bands (beyond effEBands for some modes; rejected with signalling). Only
                // mono: for stereo, C then reads past its `X` buffer (undefined behaviour).
                (
                    CELT_SET_END_BAND,
                    if ci % 2 == 0 && ch == 1 {
                        nb_bands
                    } else {
                        r.mode().eff_ebands
                    },
                ),
            ] {
                assert_eq!(rust_ctl(&mut r, req, v), c.ctl(req, v), "{what}: ctl {req}");
            }
            // QEXT through the custom API writes the code-3 bits into the signalling header
            // (C `compressed[-1]`), so it is only valid with signalling.
            #[cfg(feature = "qext")]
            if signalling != 0 && ci % 2 == 1 {
                assert_eq!(rust_ctl(&mut r, OPUS_SET_QEXT, 1), c.ctl(OPUS_SET_QEXT, 1));
            }
            let frames = 20;
            let pcm = gen_signal(
                rng.range_i32(0, SIGNAL_KINDS as i32 - 1) as usize,
                frames * fsz as usize,
                ch as usize,
                fs as u32,
                rng.next_u64(),
            );
            for f in 0..frames {
                let off = f * fsz as usize * ch as usize;
                let frame = &pcm[off..off + fsz as usize * ch as usize];
                let nb = if rng.range_i32(0, 2) == 0 {
                    rng.range_i32(2, 1275)
                } else {
                    rng.range_i32(2, 3825)
                };
                let mut oc_ = vec![0u8; nb as usize];
                let mut or = vec![0u8; nb as usize];
                let (rc, rr) = match f % 3 {
                    0 => (
                        c.custom_encode_float(frame, ch as usize, fsz, &mut oc_, nb),
                        r.opus_custom_encode_float(frame, fsz, &mut or, nb),
                    ),
                    1 => {
                        let p16 = signals::to_i16(frame);
                        (
                            c.custom_encode(&p16, ch as usize, fsz, &mut oc_, nb),
                            r.opus_custom_encode(&p16, fsz, &mut or, nb),
                        )
                    }
                    _ => {
                        let p24: Vec<i32> = frame
                            .iter()
                            .map(|&v| (v.clamp(-1.0, 1.0) * 8_388_607.0) as i32)
                            .collect();
                        (
                            c.custom_encode24(&p24, ch as usize, fsz, &mut oc_, nb),
                            r.opus_custom_encode24(&p24, fsz, &mut or, nb),
                        )
                    }
                };
                assert_eq!(rr, rc, "{what} frame {f}: ret");
                assert_slice_eq(&format!("{what} frame {f}: bytes"), &or, &oc_);
                cmp_state(&format!("{what} frame {f}"), &dump(&r), &c.state());
            }
        }
    }
}
