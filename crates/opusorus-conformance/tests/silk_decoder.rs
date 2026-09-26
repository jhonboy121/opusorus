//! Differential tests for unit `silk_decoder`: the SILK decoder (`silk_Decode` and everything it
//! calls: frame decoding, indices/parameters, core, PLC, CNG, stereo MS->LR, resampling) vs the
//! C oracle, bit-exact on PCM output, return codes, control struct, range decoder state and the
//! complete decoder state after every call.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: failures should panic"
)]

use opusorus::celt::entdec::EcDec;
use opusorus::silk::coding::silk_decode_pulses;
use opusorus::silk::decoder::*;
use opusorus::silk::resampler::{ResamplerFunction, SilkResamplerState};
use opusorus::silk::structs::*;
use opusorus_conformance::{Rng, assert_slice_eq, signals};
use opusorus_oracle::api::Encoder;
use opusorus_oracle::silk_decoder as c;
use opusorus_oracle::sys;

// ---------------------------------------------------------------------------------------------
// State dump (same order as oracle_sd_dump in csrc/silk_decoder.c)
// ---------------------------------------------------------------------------------------------

fn dump_resampler(r: &SilkResamplerState, v: &mut Vec<i32>) {
    v.extend(r.s_iir.iter().copied());
    match r.resampler_function {
        ResamplerFunction::DownFir => v.extend(r.s_fir_i32.iter().copied()),
        ResamplerFunction::IirFir => v.extend(r.s_fir_i16.iter().map(|&x| x as i32)),
        _ => v.extend([0; 36]),
    }
    v.extend(r.delay_buf.iter().map(|&x| x as i32));
    v.push(r.resampler_function.to_c());
    v.extend([
        r.batch_size,
        r.inv_ratio_q16,
        r.fir_order,
        r.fir_fracs,
        r.fs_in_khz,
        r.fs_out_khz,
        r.input_delay,
    ]);
}

fn dump_channel(s: &SilkDecoderState, v: &mut Vec<i32>) {
    v.push(s.prev_gain_q16);
    v.extend(s.exc_q14.iter().copied());
    v.extend(s.s_lpc_q14_buf.iter().copied());
    v.extend(s.out_buf.iter().map(|&x| x as i32));
    v.extend([
        s.lag_prev,
        s.last_gain_index as i32,
        s.fs_khz,
        s.fs_api_hz,
        s.nb_subfr,
        s.frame_length,
        s.subfr_length,
        s.ltp_mem_length,
        s.lpc_order,
    ]);
    v.extend(s.prev_nlsf_q15.iter().map(|&x| x as i32));
    v.push(s.first_frame_after_reset);
    v.push(s.pitch_lag_low_bits_icdf.len() as i32);
    v.push(s.pitch_contour_icdf.len() as i32);
    v.extend([
        s.n_frames_decoded,
        s.n_frames_per_packet,
        s.ec_prev_signal_type,
        s.ec_prev_lag_index as i32,
    ]);
    v.extend(s.vad_flags.iter().copied());
    v.push(s.lbrr_flag);
    v.extend(s.lbrr_flags.iter().copied());
    dump_resampler(&s.resampler_state, v);
    v.push(if s.lpc_order == 0 {
        0
    } else {
        s.ps_nlsf_cb.order as i32
    });
    let ix = &s.indices;
    v.extend(ix.gains_indices.iter().map(|&x| x as i32));
    v.extend(ix.ltp_index.iter().map(|&x| x as i32));
    v.extend(ix.nlsf_indices.iter().map(|&x| x as i32));
    v.extend([
        ix.lag_index as i32,
        ix.contour_index as i32,
        ix.signal_type as i32,
        ix.quant_offset_type as i32,
        ix.nlsf_interp_coef_q2 as i32,
        ix.per_index as i32,
        ix.ltp_scale_index as i32,
        ix.seed as i32,
    ]);
    let cng = &s.s_cng;
    v.extend(cng.cng_exc_buf_q14.iter().copied());
    v.extend(cng.cng_smth_nlsf_q15.iter().map(|&x| x as i32));
    v.extend(cng.cng_synth_state.iter().copied());
    v.extend([cng.cng_smth_gain_q16, cng.rand_seed, cng.fs_khz]);
    v.extend([s.loss_cnt, s.prev_signal_type]);
    let p = &s.s_plc;
    v.push(p.pitch_l_q8);
    v.extend(p.ltp_coef_q14.iter().map(|&x| x as i32));
    v.extend(p.prev_lpc_q12.iter().map(|&x| x as i32));
    v.extend([
        p.last_frame_lost,
        p.rand_seed,
        p.rand_scale_q14 as i32,
        p.conc_energy,
        p.conc_energy_shift,
        p.prev_ltp_scale_q14 as i32,
    ]);
    v.extend(p.prev_gain_q16.iter().copied());
    v.extend([p.fs_khz, p.nb_subfr, p.subfr_length, p.enable_deep_plc]);
}

fn dump_state(d: &SilkDecoder) -> Vec<i32> {
    let mut v = Vec::with_capacity(4096);
    for s in &d.channel_state {
        dump_channel(s, &mut v);
    }
    v.extend(d.s_stereo.pred_prev_q13.iter().map(|&x| x as i32));
    v.extend(d.s_stereo.s_mid.iter().map(|&x| x as i32));
    v.extend(d.s_stereo.s_side.iter().map(|&x| x as i32));
    v.extend([
        d.n_channels_api,
        d.n_channels_internal,
        d.prev_decode_only_middle,
    ]);
    v
}

fn ec_dump(d: &EcDec<'_>) -> c::EcStateDump {
    [
        d.storage,
        d.end_offs,
        d.end_window,
        d.nend_bits as u32,
        d.nbits_total as u32,
        d.offs,
        d.rng,
        d.val,
        d.ext,
        d.rem as u32,
        d.error as u32,
        d.tell_frac(),
    ]
}

const fn ctrl_to_c(r: &SilkDecControlStruct) -> c::DecCtrl {
    [
        r.n_channels_api,
        r.n_channels_internal,
        r.api_sample_rate,
        r.internal_sample_rate,
        r.payload_size_ms,
        r.prev_pitch_lag,
        r.enable_deep_plc,
    ]
}

// ---------------------------------------------------------------------------------------------
// Harness: drives both decoders exactly like src/opus_decoder.c does for SILK-only frames
// ---------------------------------------------------------------------------------------------

/// Output buffer size: 20 ms at the highest API rate, stereo.
const OUT_LEN: usize = 96 * 20 * 2;

struct Harness {
    r: Box<SilkDecoder>,
    c: c::SilkDec,
    ctrl: SilkDecControlStruct,
    label: String,
    calls: usize,
    /// Compare the full state dump every `dump_every` calls (1 = every call).
    dump_every: usize,
    cov: Cov,
}

/// Coverage counters (checked by the tests so the interesting paths are really exercised).
#[derive(Debug, Default, Clone, Copy)]
struct Cov {
    /// Calls that decoded a frame while the previous frame was mid-only.
    mid_only: usize,
    /// Side channel restarts after mid-only frames.
    side_restart: usize,
    /// FEC calls that decoded real LBRR data.
    lbrr_decoded: usize,
    /// Lost-frame calls with a non-zero CNG level.
    cng: usize,
    /// Lost-frame calls concealing a voiced frame.
    voiced_plc: usize,
    /// Stereo stream -> mono stream transitions with stereo output.
    stereo_to_mono: usize,
    /// Internal sampling rate changes.
    fs_change: usize,
}

impl Cov {
    const fn add(&mut self, o: &Self) {
        self.mid_only += o.mid_only;
        self.side_restart += o.side_restart;
        self.lbrr_decoded += o.lbrr_decoded;
        self.cng += o.cng;
        self.voiced_plc += o.voiced_plc;
        self.stereo_to_mono += o.stereo_to_mono;
        self.fs_change += o.fs_change;
    }
}

impl Harness {
    fn new(api_fs: i32, api_ch: i32, label: &str) -> Self {
        let r = Box::new(SilkDecoder::new());
        let mut h = Self {
            r,
            c: c::SilkDec::new(),
            ctrl: SilkDecControlStruct {
                n_channels_api: api_ch,
                n_channels_internal: 1,
                api_sample_rate: api_fs,
                internal_sample_rate: 16000,
                payload_size_ms: 20,
                prev_pitch_lag: 0,
                enable_deep_plc: 0,
            },
            label: label.to_string(),
            calls: 0,
            dump_every: 1,
            cov: Cov::default(),
        };
        h.check_state("init");
        h
    }

    fn check_state(&mut self, what: &str) {
        let rs = dump_state(&self.r);
        let cs = self.c.dump();
        assert_slice_eq(&format!("{} {what}: state dump", self.label), &rs, &cs);
    }

    /// `silk_ResetDecoder` on both (the Opus decoder does this after CELT-only frames).
    fn reset(&mut self) {
        let a = self.r.reset();
        let b = self.c.reset();
        assert_eq!(a, b, "{}: reset ret", self.label);
        self.check_state("reset");
    }

    /// One silk_Decode call on both; returns (ret, nSamplesOut).
    fn call(&mut self, rdec: &mut EcDec<'_>, lost: i32, new_packet: i32, what: &str) -> (i32, i32) {
        let prev_dom = self.r.prev_decode_only_middle;
        let prev_nci = self.r.n_channels_internal;
        let prev_fs = self.r.channel_state[0].fs_khz;
        if lost == FLAG_PACKET_LOST {
            let s0 = &self.r.channel_state[0];
            if s0.s_cng.cng_smth_gain_q16 > 0 {
                self.cov.cng += 1;
            }
            if s0.prev_signal_type == 2 {
                self.cov.voiced_plc += 1;
            }
        }
        let mut cctrl = ctrl_to_c(&self.ctrl);
        let mut cout = vec![0f32; OUT_LEN];
        let (cret, cn) = self.c.decode(&mut cctrl, lost, new_packet, &mut cout);
        let mut rout = vec![0f32; OUT_LEN];
        let mut rn = 0;
        let rret = self
            .r
            .silk_decode(&mut self.ctrl, lost, new_packet, rdec, &mut rout, &mut rn);
        let tag = format!("{} call {} ({what}, lost={lost})", self.label, self.calls);
        assert_eq!(rret, cret, "{tag}: return code");
        assert_eq!(rn, cn, "{tag}: nSamplesOut");
        assert_eq!(ctrl_to_c(&self.ctrl), cctrl, "{tag}: control struct");
        assert_eq!(ec_dump(rdec), self.c.ec_state(), "{tag}: range decoder");
        if cret == 0 {
            let n = cn as usize * self.ctrl.n_channels_api as usize;
            let rb: Vec<u32> = rout[..n].iter().map(|x| x.to_bits()).collect();
            let cb: Vec<u32> = cout[..n].iter().map(|x| x.to_bits()).collect();
            assert_slice_eq(&format!("{tag}: pcm"), &rb, &cb);
        }
        if cret == 0 {
            let s0 = &self.r.channel_state[0];
            if lost == FLAG_DECODE_LBRR && s0.lbrr_flags[(s0.n_frames_decoded - 1) as usize] == 1 {
                self.cov.lbrr_decoded += 1;
            }
            if prev_dom == 1 && self.ctrl.n_channels_internal == 2 {
                self.cov.mid_only += 1;
                if self.r.prev_decode_only_middle == 0 && lost == 0 {
                    self.cov.side_restart += 1;
                }
            }
            if prev_nci == 2 && self.ctrl.n_channels_internal == 1 && self.ctrl.n_channels_api == 2
            {
                self.cov.stereo_to_mono += 1;
            }
            if prev_fs != 0 && prev_fs != s0.fs_khz {
                self.cov.fs_change += 1;
            }
        }
        if self.calls.is_multiple_of(self.dump_every) || cret != 0 {
            self.check_state(&tag);
        }
        self.calls += 1;
        (cret, cn)
    }

    /// Decodes one SILK frame (10/20/40/60 ms) like `opus_decode_frame`: `payload` is `None`
    /// for PLC (lost_flag 1), otherwise `fec` selects lost_flag 2.
    fn frame(&mut self, payload: Option<&[u8]>, frame_ms: i32, fec: bool) {
        let empty: [u8; 0] = [];
        let data = payload.unwrap_or(&empty);
        let lost = if payload.is_none() {
            FLAG_PACKET_LOST
        } else if fec {
            FLAG_DECODE_LBRR
        } else {
            FLAG_DECODE_NORMAL
        };
        self.ctrl.payload_size_ms = frame_ms.max(10);
        let frame_size = frame_ms * self.ctrl.api_sample_rate / 1000;
        let mut rdec = EcDec::new(data);
        self.c.ec_init(data);
        assert_eq!(ec_dump(&rdec), self.c.ec_state(), "{}: ec init", self.label);
        let mut decoded = 0;
        while decoded < frame_size {
            let (ret, n) = self.call(&mut rdec, lost, i32::from(decoded == 0), "frame");
            let n = if ret != 0 {
                if lost == 0 {
                    return;
                }
                frame_size
            } else {
                n
            };
            if n <= 0 {
                return;
            }
            decoded += n;
        }
    }

    /// Decodes a full Opus packet (SILK-only, code 0) or runs PLC for a DTX / lost packet.
    fn packet(&mut self, pkt: Option<&[u8]>, last_ms: &mut i32, fec: bool) {
        match pkt {
            Some(p) if p.len() > 1 => {
                let toc = p[0];
                let config = (toc >> 3) as i32;
                assert!(config < 12, "not a SILK-only packet: config {config}");
                assert_eq!(toc & 3, 0, "expected a code-0 packet");
                let ms = [10, 20, 40, 60][(config & 3) as usize];
                self.ctrl.n_channels_internal = if toc & 4 != 0 { 2 } else { 1 };
                self.ctrl.internal_sample_rate = [8000, 12000, 16000][(config >> 2) as usize];
                *last_ms = ms;
                self.frame(Some(&p[1..]), ms, fec);
            }
            _ => {
                let ms = *last_ms;
                self.frame(None, ms, false);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Packet generation with the oracle Opus encoder (RESTRICTED_SILK)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Sig {
    Speech,
    Music,
    Noise,
    /// Speech with silent stretches (DTX / CNG).
    Bursty,
    /// Stereo alternating between identical channels (side = 0, mid-only) and wide stereo.
    StereoMidOnly,
}

#[derive(Clone, Copy, Debug)]
struct EncCfg {
    channels: i32,
    bitrate: i32,
    max_bw: i32,
    frame_ms: i32,
    fec: bool,
    loss_perc: i32,
    dtx: bool,
    complexity: i32,
    sig: Sig,
    /// Randomly vary bitrate / bandwidth / forced channels per packet.
    vary: bool,
    n_packets: usize,
    seed: u64,
}

fn gen_signal(cfg: &EncCfg, n: usize) -> Vec<i16> {
    let ch = cfg.channels as usize;
    let fs = 48000;
    let mut x = match cfg.sig {
        Sig::Speech | Sig::Bursty | Sig::StereoMidOnly => signals::speech_like(n, ch, fs, cfg.seed),
        Sig::Music => signals::music_like(n, ch, fs, cfg.seed),
        Sig::Noise => signals::noise(n, ch, 0.3, cfg.seed),
    };
    match cfg.sig {
        Sig::Bursty => {
            // 0.6 s on, 0.6 s off (with a tiny noise floor in some off stretches)
            let mut rng = Rng::new(cfg.seed ^ 0x55);
            for i in 0..n {
                let seg = i / (fs as usize * 6 / 10);
                if seg % 2 == 1 {
                    for c in 0..ch {
                        x[i * ch + c] = if seg % 4 == 3 {
                            0.0003 * rng.f32_sym()
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
        Sig::StereoMidOnly if ch == 2 => {
            let music = signals::music_like(n, 1, fs, cfg.seed ^ 7);
            for i in 0..n {
                let seg = i / (fs as usize / 4);
                match seg % 3 {
                    0 => x[i * 2 + 1] = x[i * 2],
                    1 => x[i * 2 + 1] = 0.5 * x[i * 2] + 0.5 * music[i],
                    _ => x[i * 2 + 1] = x[i * 2] + 0.00002 * music[i],
                }
            }
        }
        _ => {}
    }
    signals::to_i16(&x)
}

fn encode_stream(cfg: &EncCfg) -> Vec<Vec<u8>> {
    let mut enc =
        Encoder::new(48000, cfg.channels, sys::OPUS_APPLICATION_RESTRICTED_SILK).expect("encoder");
    enc.ctl_set(sys::OPUS_SET_BITRATE_REQUEST, cfg.bitrate)
        .unwrap();
    enc.ctl_set(sys::OPUS_SET_MAX_BANDWIDTH_REQUEST, cfg.max_bw)
        .unwrap();
    enc.ctl_set(sys::OPUS_SET_INBAND_FEC_REQUEST, i32::from(cfg.fec))
        .unwrap();
    enc.ctl_set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, cfg.loss_perc)
        .unwrap();
    enc.ctl_set(sys::OPUS_SET_DTX_REQUEST, i32::from(cfg.dtx))
        .unwrap();
    enc.ctl_set(sys::OPUS_SET_COMPLEXITY_REQUEST, cfg.complexity)
        .unwrap();
    let frame = (48 * cfg.frame_ms) as usize;
    let pcm = gen_signal(cfg, frame * cfg.n_packets);
    let ch = cfg.channels as usize;
    let mut rng = Rng::new(cfg.seed ^ 0xABCD);
    let mut out = Vec::with_capacity(cfg.n_packets);
    let mut buf = vec![0u8; 1500];
    for k in 0..cfg.n_packets {
        if cfg.vary && rng.range_i32(0, 7) == 0 {
            let br = [6000, 9000, 12000, 16000, 24000, 32000, 40000][rng.range_i32(0, 6) as usize];
            enc.ctl_set(sys::OPUS_SET_BITRATE_REQUEST, br * cfg.channels)
                .unwrap();
            let bw = [
                sys::OPUS_BANDWIDTH_NARROWBAND,
                sys::OPUS_BANDWIDTH_MEDIUMBAND,
                sys::OPUS_BANDWIDTH_WIDEBAND,
            ][rng.range_i32(0, 2) as usize];
            enc.ctl_set(sys::OPUS_SET_MAX_BANDWIDTH_REQUEST, bw)
                .unwrap();
            if cfg.channels == 2 {
                let fc = [sys::OPUS_AUTO, 1, 2][rng.range_i32(0, 2) as usize];
                enc.ctl_set(sys::OPUS_SET_FORCE_CHANNELS_REQUEST, fc)
                    .unwrap();
            }
        }
        let n = enc
            .encode(&pcm[k * frame * ch..(k + 1) * frame * ch], frame, &mut buf)
            .expect("encode");
        out.push(buf[..n].to_vec());
    }
    out
}

/// Loss pattern for a packet stream.
#[derive(Clone, Copy, Debug)]
enum Loss {
    None,
    /// Random losses concealed by PLC.
    Plc(i32),
    /// Random losses recovered with FEC from the next packet (then PLC if that is lost too).
    Fec(i32),
}

fn run_stream(cfg: &EncCfg, api_fs: i32, api_ch: i32, loss: Loss, dump_every: usize) -> Cov {
    let pkts = encode_stream(cfg);
    let label = format!("{cfg:?} api {api_fs}x{api_ch} {loss:?}");
    let mut h = Harness::new(api_fs, api_ch, &label);
    h.dump_every = dump_every;
    let mut rng = Rng::new(cfg.seed ^ 0x1234);
    let mut last_ms = cfg.frame_ms;
    let mut i = 0;
    while i < pkts.len() {
        let (pct, fec) = match loss {
            Loss::None => (0, false),
            Loss::Plc(p) => (p, false),
            Loss::Fec(p) => (p, true),
        };
        let lost = rng.range_i32(0, 99) < pct;
        if !lost {
            h.packet(Some(&pkts[i]), &mut last_ms, false);
        } else if fec && i + 1 < pkts.len() && pkts[i + 1].len() > 1 {
            // Decode the lost packet from the LBRR data of the next one (opus_decode with
            // decode_fec=1), using the duration of the next packet.
            h.packet(Some(&pkts[i + 1]), &mut last_ms, true);
        } else {
            h.packet(None, &mut last_ms, false);
        }
        i += 1;
    }
    // Trailing PLC run
    for _ in 0..3 {
        h.packet(None, &mut last_ms, false);
    }
    h.check_state("end");
    h.cov
}

const fn cfg(
    channels: i32,
    bitrate: i32,
    max_bw: i32,
    frame_ms: i32,
    sig: Sig,
    seed: u64,
) -> EncCfg {
    EncCfg {
        channels,
        bitrate,
        max_bw,
        frame_ms,
        fec: false,
        loss_perc: 0,
        dtx: false,
        complexity: 10,
        sig,
        vary: false,
        n_packets: 1000 / frame_ms as usize * 2,
        seed,
    }
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

const API_RATES: &[i32] = &[
    8000,
    12000,
    16000,
    24000,
    48000,
    #[cfg(feature = "qext")]
    96000,
];

const BWS: [i32; 3] = [
    sys::OPUS_BANDWIDTH_NARROWBAND,
    sys::OPUS_BANDWIDTH_MEDIUMBAND,
    sys::OPUS_BANDWIDTH_WIDEBAND,
];

#[test]
fn decoder_size_and_init() {
    let mut n = 0;
    assert_eq!(silk_get_decoder_size(&mut n), 0);
    assert!(n > 0);
    assert!(c::decoder_size() > 0);
    let mut h = Harness::new(48000, 2, "init");
    assert_eq!(h.r.init(), h.c.init());
    h.check_state("re-init");
    h.reset();
    let mut d = SilkDecoder::new();
    // Without OSCE there is nothing to load. With OSCE, `None` means "compiled-in weights" in C;
    // the Rust port has none (weights come from a runtime blob, PLAN D-015), so it reports -1.
    #[cfg(not(feature = "osce"))]
    assert_eq!(d.load_osce_models(None), 0);
    #[cfg(feature = "osce")]
    assert_eq!(d.load_osce_models(None), -1);
}

#[test]
fn mono_all_bandwidths_frame_sizes_api_rates() {
    let mut cov = Cov::default();
    let mut seed = 1;
    for &bw in &BWS {
        for &ms in &[10, 20, 40, 60] {
            for &api in API_RATES {
                seed += 1;
                let sig = [Sig::Speech, Sig::Music, Sig::Noise][(seed % 3) as usize];
                let br = [8000, 12000, 20000, 32000][(seed % 4) as usize];
                let mut c = cfg(1, br, bw, ms, sig, seed);
                c.n_packets = (600 / ms) as usize;
                c.complexity = (seed % 11) as i32;
                cov.add(&run_stream(&c, api, 1, Loss::None, 1));
            }
        }
    }
    assert!(cov.voiced_plc > 0, "{cov:?}");
}

#[test]
fn mono_stream_stereo_output() {
    let mut cov = Cov::default();
    for (k, &api) in API_RATES.iter().enumerate() {
        let mut c = cfg(
            1,
            16000,
            BWS[k % 3],
            [20, 60, 10, 40][k % 4],
            Sig::Speech,
            50 + k as u64,
        );
        c.n_packets = (800 / c.frame_ms) as usize;
        cov.add(&run_stream(&c, api, 2, Loss::Plc(10), 1));
    }
    assert!(cov.voiced_plc > 0, "{cov:?}");
}

#[test]
fn stereo_streams() {
    let mut cov = Cov::default();
    let mut seed = 100;
    for &ms in &[10, 20, 40, 60] {
        for &api_ch in &[1, 2] {
            for &api in &[8000, 16000, 48000] {
                seed += 1;
                let bw = BWS[(seed % 3) as usize];
                let mut c = cfg(
                    2,
                    24000 + 8000 * (seed % 3) as i32,
                    bw,
                    ms,
                    Sig::StereoMidOnly,
                    seed,
                );
                c.n_packets = (1500 / ms) as usize;
                cov.add(&run_stream(&c, api, api_ch, Loss::None, 2));
            }
        }
    }
    assert!(cov.mid_only > 0 && cov.side_restart > 0, "{cov:?}");
}

#[test]
fn stereo_low_bitrate_mid_only_and_switching() {
    let mut cov = Cov::default();
    // Low bitrates make the encoder collapse the side channel (mid-only frames); `vary` toggles
    // forced channels / bandwidth / bitrate, giving mono<->stereo and internal-rate transitions.
    let mut seed = 300;
    for &ms in &[20, 40, 60, 10] {
        for &api in &[16000, 48000, 24000] {
            seed += 1;
            let mut c = cfg(
                2,
                8000,
                sys::OPUS_BANDWIDTH_WIDEBAND,
                ms,
                Sig::StereoMidOnly,
                seed,
            );
            c.vary = true;
            c.n_packets = (4000 / ms) as usize;
            cov.add(&run_stream(&c, api, 2, Loss::Plc(5), 3));
            let mut c1 = c;
            c1.seed += 1000;
            cov.add(&run_stream(&c1, api, 1, Loss::None, 3));
        }
    }
    assert!(
        cov.mid_only > 0 && cov.side_restart > 0 && cov.stereo_to_mono > 0 && cov.fs_change > 0,
        "{cov:?}"
    );
}

#[test]
fn plc_sequences() {
    let mut cov = Cov::default();
    let mut seed = 500;
    for &ch in &[1, 2] {
        for &ms in &[10, 20, 40, 60] {
            for &pct in &[10, 30, 60] {
                seed += 1;
                let sig = [Sig::Speech, Sig::Music, Sig::Noise][(seed % 3) as usize];
                let mut c = cfg(ch, 12000 * ch, BWS[(seed % 3) as usize], ms, sig, seed);
                c.n_packets = (2000 / ms) as usize;
                let api = [8000, 12000, 16000, 24000, 48000][(seed % 5) as usize];
                cov.add(&run_stream(&c, api, ch, Loss::Plc(pct), 2));
            }
        }
    }
    assert!(cov.voiced_plc > 0 && cov.cng > 0, "{cov:?}");
}

#[test]
fn fec_lbrr() {
    let mut cov = Cov::default();
    let mut seed = 700;
    for &ch in &[1, 2] {
        for &ms in &[10, 20, 40, 60] {
            for &pct in &[15, 40] {
                seed += 1;
                let mut c = cfg(
                    ch,
                    20000 * ch,
                    BWS[(seed % 3) as usize],
                    ms,
                    Sig::Speech,
                    seed,
                );
                c.fec = true;
                c.loss_perc = 25;
                c.vary = seed % 2 == 0;
                c.n_packets = (3000 / ms) as usize;
                let api = [48000, 16000, 8000, 12000, 24000][(seed % 5) as usize];
                cov.add(&run_stream(
                    &c,
                    api,
                    [1, 2][(seed % 2) as usize],
                    Loss::Fec(pct),
                    2,
                ));
            }
        }
    }
    assert!(cov.lbrr_decoded > 20, "{cov:?}");
}

#[test]
fn dtx_cng() {
    let mut cov = Cov::default();
    let mut seed = 900;
    for &ch in &[1, 2] {
        for &ms in &[10, 20, 40, 60] {
            seed += 1;
            let mut c = cfg(
                ch,
                10000 * ch,
                BWS[(seed % 3) as usize],
                ms,
                Sig::Bursty,
                seed,
            );
            c.dtx = true;
            c.n_packets = (6000 / ms) as usize;
            let api = [48000, 16000, 8000, 24000][(seed % 4) as usize];
            let pkts = encode_stream(&c);
            assert!(
                pkts.iter().any(|p| p.len() <= 1),
                "{c:?}: expected DTX packets"
            );
            cov.add(&run_stream(&c, api, ch, Loss::Plc(3), 2));
        }
    }
    assert!(cov.cng > 20, "{cov:?}");
}

#[test]
fn reset_mid_stream_and_rate_changes() {
    // Resets (as after CELT-only frames) and API rate changes between packets.
    let c1 = {
        let mut c = cfg(
            2,
            20000,
            sys::OPUS_BANDWIDTH_WIDEBAND,
            20,
            Sig::Speech,
            1100,
        );
        c.vary = true;
        c.n_packets = 150;
        c
    };
    let pkts = encode_stream(&c1);
    let mut h = Harness::new(48000, 2, "reset/rates");
    let mut rng = Rng::new(77);
    let mut last_ms = 20;
    for p in &pkts {
        match rng.range_i32(0, 19) {
            0 => h.reset(),
            1 => {
                h.ctrl.api_sample_rate =
                    API_RATES[rng.range_i32(0, API_RATES.len() as i32 - 1) as usize]
            }
            2 => h.ctrl.n_channels_api = rng.range_i32(1, 2),
            3 => h.packet(None, &mut last_ms, false),
            _ => {}
        }
        h.packet(Some(p), &mut last_ms, false);
    }
}

/// Invalid control values. The hardened C oracle aborts on these (`celt_assert( 0 )` before
/// the error return), so only the Rust error codes are checked here; the C build without
/// assertions returns the same codes.
#[test]
fn invalid_control() {
    use opusorus::silk::errors::*;
    let c1 = cfg(
        1,
        16000,
        sys::OPUS_BANDWIDTH_WIDEBAND,
        20,
        Sig::Speech,
        1200,
    );
    let pkts = encode_stream(&EncCfg { n_packets: 4, ..c1 });
    let payload = &pkts[1][1..];
    let mut out = vec![0f32; OUT_LEN];
    let mut n = 0;
    let base = SilkDecControlStruct {
        n_channels_api: 1,
        n_channels_internal: 1,
        api_sample_rate: 16000,
        internal_sample_rate: 16000,
        payload_size_ms: 20,
        prev_pitch_lag: 0,
        enable_deep_plc: 0,
    };
    for (ctrl, want) in [
        (
            SilkDecControlStruct {
                payload_size_ms: 30,
                ..base
            },
            SILK_DEC_INVALID_FRAME_SIZE,
        ),
        (
            SilkDecControlStruct {
                internal_sample_rate: 44100,
                ..base
            },
            SILK_DEC_INVALID_SAMPLING_FREQUENCY,
        ),
        (
            SilkDecControlStruct {
                api_sample_rate: 7999,
                ..base
            },
            SILK_DEC_INVALID_SAMPLING_FREQUENCY,
        ),
        (
            SilkDecControlStruct {
                api_sample_rate: 192000,
                ..base
            },
            SILK_DEC_INVALID_SAMPLING_FREQUENCY,
        ),
    ] {
        let mut d = SilkDecoder::new();
        let mut ctrl = ctrl;
        let mut ec = EcDec::new(payload);
        assert_eq!(
            d.silk_decode(&mut ctrl, 0, 1, &mut ec, &mut out, &mut n),
            want,
            "{ctrl:?}"
        );
        // A valid call afterwards decodes normally.
        let mut ctrl = base;
        let mut ec = EcDec::new(payload);
        assert_eq!(d.silk_decode(&mut ctrl, 0, 1, &mut ec, &mut out, &mut n), 0);
        assert_eq!(n, 320);
    }
}

#[test]
fn garbage_payloads() {
    let mut rng = Rng::new(0xDEAD_BEEF);
    for run in 0..60 {
        let api_ch = rng.range_i32(1, 2);
        let api = API_RATES[rng.range_i32(0, API_RATES.len() as i32 - 1) as usize];
        let mut h = Harness::new(api, api_ch, &format!("garbage run {run}"));
        h.dump_every = 2;
        for _ in 0..60 {
            let ms = [10, 20, 40, 60][rng.range_i32(0, 3) as usize];
            if rng.range_i32(0, 3) != 0 {
                h.ctrl.n_channels_internal = rng.range_i32(1, 2);
                h.ctrl.internal_sample_rate = [8000, 12000, 16000][rng.range_i32(0, 2) as usize];
            }
            let len = match rng.range_i32(0, 4) {
                0 => rng.range_i32(0, 4) as usize,
                1 => rng.range_i32(0, 40) as usize,
                _ => rng.range_i32(0, 400) as usize,
            };
            let mut data = vec![0u8; len];
            rng.fill_bytes(&mut data);
            if rng.range_i32(0, 5) == 0 {
                // Low-entropy payloads (long runs of equal bytes)
                let b = rng.next_u32() as u8;
                data.iter_mut().for_each(|x| *x = b);
            }
            match rng.range_i32(0, 9) {
                0 => h.frame(None, ms, false),
                1 => h.frame(Some(&data), ms, true),
                _ => h.frame(Some(&data), ms, false),
            }
            h.ctrl.enable_deep_plc = rng.range_i32(0, 1);
        }
    }
}

#[test]
fn stereo_ms_to_lr() {
    let mut rng = Rng::new(42);
    for it in 0..20000 {
        let fs_khz = [8, 12, 16][rng.range_i32(0, 2) as usize];
        let fl = (fs_khz * [10, 20][rng.range_i32(0, 1) as usize]) as usize;
        let extreme = it % 4 == 0;
        let sample = |rng: &mut Rng| -> i16 {
            if extreme {
                [i16::MIN, i16::MAX, 0, -1, 1, rng.i16()][rng.range_i32(0, 5) as usize]
            } else {
                (rng.i16() as i32 >> rng.range_i32(0, 12)) as i16
            }
        };
        let mut x1: Vec<i16> = (0..fl + 2).map(|_| sample(&mut rng)).collect();
        let mut x2: Vec<i16> = (0..fl + 2).map(|_| sample(&mut rng)).collect();
        let mut st6 = [0i16; 6];
        for v in &mut st6 {
            *v = sample(&mut rng);
        }
        // Predictors are in [-2^14, 2^14] (Q13, |pred| <= ~2) in valid streams.
        let pred = [rng.range_i32(-16384, 16384), rng.range_i32(-16384, 16384)];
        st6[0] = rng.range_i32(-16384, 16384) as i16;
        st6[1] = rng.range_i32(-16384, 16384) as i16;
        let mut state = StereoDecState {
            pred_prev_q13: [st6[0], st6[1]],
            s_mid: [st6[2], st6[3]],
            s_side: [st6[4], st6[5]],
        };
        let (mut cx1, mut cx2) = (x1.clone(), x2.clone());
        c::stereo_ms_to_lr(&mut st6, &mut cx1, &mut cx2, &pred, fs_khz, fl);
        silk_stereo_ms_to_lr(&mut state, &mut x1, &mut x2, &pred, fs_khz, fl as i32);
        assert_eq!(x1, cx1, "it {it}: x1");
        assert_eq!(x2, cx2, "it {it}: x2");
        let rs = [
            state.pred_prev_q13[0],
            state.pred_prev_q13[1],
            state.s_mid[0],
            state.s_mid[1],
            state.s_side[0],
            state.s_side[1],
        ];
        assert_eq!(rs, st6, "it {it}: state");
    }
}

#[test]
fn decode_indices_random() {
    let mut rng = Rng::new(7);
    for it in 0..20000 {
        let fs_khz = [8, 12, 16][rng.range_i32(0, 2) as usize];
        let nb_subfr = [2, 4][rng.range_i32(0, 1) as usize];
        let vad = rng.range_i32(0, 1);
        let lbrr = [0, 2][rng.range_i32(0, 1) as usize];
        let cond = rng.range_i32(0, 2);
        let prev_type = rng.range_i32(0, 2);
        let prev_lag = if it % 3 == 0 {
            rng.i16() as i32
        } else {
            rng.range_i32(0, 300)
        };
        let len = rng.range_i32(0, 200) as usize;
        let mut data = vec![0u8; len];
        rng.fill_bytes(&mut data);

        let (cout, cpulses) = c::decode_indices(
            &data, fs_khz, nb_subfr, vad, lbrr, cond, prev_type, prev_lag,
        );

        let mut s = Box::new(SilkDecoderState::new());
        silk_init_decoder(&mut s);
        s.nb_subfr = nb_subfr;
        silk_decoder_set_fs(&mut s, fs_khz, 48000);
        s.vad_flags[0] = vad;
        s.ec_prev_signal_type = prev_type;
        s.ec_prev_lag_index = prev_lag as i16;
        let mut d = EcDec::new(&data);
        silk_decode_indices(&mut s, &mut d, 0, lbrr, cond);
        let mut pulses = vec![0i16; 320];
        silk_decode_pulses(
            &mut d,
            &mut pulses,
            s.indices.signal_type as i32,
            s.indices.quant_offset_type as i32,
            s.frame_length,
        );
        let ix = &s.indices;
        let mut rout = vec![ix.signal_type as i32, ix.quant_offset_type as i32];
        rout.extend(ix.gains_indices.iter().map(|&x| x as i32));
        rout.extend(ix.ltp_index.iter().map(|&x| x as i32));
        rout.extend(ix.nlsf_indices.iter().map(|&x| x as i32));
        rout.extend([
            ix.lag_index as i32,
            ix.contour_index as i32,
            ix.nlsf_interp_coef_q2 as i32,
            ix.per_index as i32,
            ix.ltp_scale_index as i32,
            ix.seed as i32,
            s.ec_prev_signal_type,
            s.ec_prev_lag_index as i32,
            d.tell_frac() as i32,
            d.rng as i32,
        ]);
        assert_slice_eq(&format!("it {it}: indices"), &rout, &cout);
        assert_slice_eq(&format!("it {it}: pulses"), &pulses, &cpulses);
    }
}
