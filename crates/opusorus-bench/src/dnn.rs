//! DNN workloads (feature `dnn`, float build): the deep PLC under 20 % packet loss, the OSCE
//! enhancers (LACE, NoLACE) and DRED encoding. One iteration is one 20 ms frame, as for the
//! other codec benchmarks.
//!
//! The same models run everywhere: opusorus loads the oracle's serialized weight blob, the
//! oracle and the optimized CMake build use their compiled-in weights (the same model data).

use opusorus_oracle::dnn_integration as di;
use opusorus_oracle::sys;

use crate::{
    AnyDecoder, AnyEncoder, CodecConfig, FRAMES, Impl, MAX_PACKET, Result, Runner, Sample,
    codec_configs, rust_err,
};

/// Every 5th packet is lost (20 % loss) in the PLC workloads.
const LOSS_PERIOD: usize = 5;

/// A DNN benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DnnBench {
    /// Decode SILK 16 kHz with 20 % loss at decoder complexity 5 (deep PLC, FARGAN).
    PlcSilk,
    /// Decode hybrid 48 kHz with 20 % loss at decoder complexity 5 (deep PLC, FARGAN).
    PlcHybrid,
    /// Decode without loss at decoder complexity 6 (LACE) or 7 (NoLACE) on SILK 16 kHz.
    Osce(i32),
    /// Encode SILK 16 kHz VoIP with 1 s of DRED at 20 % expected loss (RDOVAE encoder).
    DredEncode,
}

impl DnnBench {
    /// All DNN benchmarks.
    pub const ALL: [Self; 5] = [
        Self::PlcSilk,
        Self::PlcHybrid,
        Self::Osce(6),
        Self::Osce(7),
        Self::DredEncode,
    ];

    /// Group id suffix.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::PlcSilk => "plc_loss20_silk_16k_mono",
            Self::PlcHybrid => "plc_loss20_hybrid_48k_mono",
            Self::Osce(6) => "lace_silk_16k_mono",
            Self::Osce(_) => "nolace_silk_16k_mono",
            Self::DredEncode => "dred_encode_silk_16k_mono",
        }
    }

    /// Report label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::PlcSilk => "DNN deep PLC, 20 % loss, silk_16k_mono",
            Self::PlcHybrid => "DNN deep PLC, 20 % loss, hybrid_48k_mono",
            Self::Osce(6) => "DNN LACE decode silk_16k_mono",
            Self::Osce(_) => "DNN NoLACE decode silk_16k_mono",
            Self::DredEncode => "DNN DRED encode silk_16k_mono (1 s)",
        }
    }

    /// Per-frame closure for `imp`.
    ///
    /// # Errors
    /// If the setup fails.
    pub fn runner(self, imp: Impl) -> Result<Runner> {
        match self {
            Self::PlcSilk => decode_runner(imp, &config("silk_16k_mono_16k_voip")?, 5, true),
            Self::PlcHybrid => decode_runner(imp, &config("hybrid_48k_mono_32k")?, 5, true),
            Self::Osce(cx) => decode_runner(imp, &config("silk_16k_mono_16k_voip")?, cx, false),
            Self::DredEncode => dred_encode_runner(imp),
        }
    }
}

fn config(id: &str) -> Result<CodecConfig> {
    codec_configs()
        .into_iter()
        .find(|c| c.id == id)
        .ok_or_else(|| crate::BenchError(format!("no codec config {id}")))
}

fn decode_runner(imp: Impl, cfg: &CodecConfig, complexity: i32, lossy: bool) -> Result<Runner> {
    let packets = crate::decode_packets(cfg)?;
    let mut dec = AnyDecoder::new(imp, cfg.fs, cfg.channels)?;
    if let AnyDecoder::Rust(d) = &mut dec {
        d.set_dnn_blob(di::decoder_blob())
            .map_err(|e| rust_err("dnn blob", e))?;
    }
    dec.ctl_set(sys::OPUS_SET_COMPLEXITY_REQUEST, complexity)?;
    let fsz = cfg.frame_size();
    let mut pcm: Vec<Sample> = vec![Sample::default(); fsz * cfg.channels as usize];
    let mut i = 0;
    Ok(Box::new(move || {
        let lost = lossy && i % LOSS_PERIOD == LOSS_PERIOD - 1;
        let p = (!lost).then(|| packets[i].as_slice());
        i = (i + 1) % packets.len();
        dec.decode_opt(core::hint::black_box(p), &mut pcm, fsz)?;
        core::hint::black_box(&pcm);
        Ok(())
    }))
}

fn dred_encode_runner(imp: Impl) -> Result<Runner> {
    let cfg = config("silk_16k_mono_16k_voip")?;
    let mut enc = AnyEncoder::new(imp, &cfg, 10)?;
    if let AnyEncoder::Rust(e) = &mut enc {
        e.set_dnn_blob(di::encoder_blob())
            .map_err(|e| rust_err("dnn blob", e))?;
    }
    enc.ctl_set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, 20)?;
    // In 10 ms units: 1 s of redundancy.
    enc.ctl_set(sys::OPUS_SET_DRED_DURATION_REQUEST, 100)?;
    let input = cfg.samples();
    let fsz = cfg.frame_size();
    let n = fsz * cfg.channels as usize;
    let mut out = vec![0u8; MAX_PACKET];
    let mut i = 0;
    Ok(Box::new(move || {
        let f = &input[i * n..(i + 1) * n];
        i = (i + 1) % FRAMES;
        let len = enc.encode(core::hint::black_box(f), fsz, &mut out)?;
        core::hint::black_box(len);
        Ok(())
    }))
}
