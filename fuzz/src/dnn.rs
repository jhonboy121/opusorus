//! Decoder adapters for the DNN decoder features (`deep-plc`, `osce`): an opusorus `Decoder`
//! with the oracle's weights loaded (`set_dnn_blob` of the blob the oracle serializes from its
//! compiled-in model tables) and a C `OpusDecoder` of the DNN oracle, both comparing the DNN
//! state (LPCNet PLC counters, FEC queue, `pcm` / `features` history, OSCE method and BWE mode)
//! through [`DecApi::extra_state`].
//!
//! The Opus-level state dump of [`crate::dec::DecApi::states`] is not available for the DNN
//! oracle decoder (`dnn_integration::Dec`), so both sides report none; outputs, return codes,
//! every GET CTL and the DNN state are compared after every operation.

use crate::code;
use crate::dec::DecApi;
use opusorus::Decoder;
use opusorus_oracle::dnn_integration as di;

/// An opusorus decoder running the oracle's DNN models.
#[derive(Debug)]
pub struct RustDnn(pub Decoder);

impl RustDnn {
    /// `Decoder::new` + `set_dnn_blob(decoder_blob())`.
    ///
    /// # Errors
    /// The creation error code.
    pub fn new(fs: i32, channels: i32) -> Result<Self, i32> {
        let mut d = code(Decoder::new(fs, channels))?;
        d.set_dnn_blob(di::decoder_blob())
            .expect("the oracle's decoder blob loads");
        Ok(Self(d))
    }
}

fn state_bits(ints: &[i32], floats: &[f32]) -> Vec<u32> {
    ints.iter()
        .map(|&v| v as u32)
        .chain(floats.iter().map(|v| v.to_bits()))
        .collect()
}

impl DecApi for RustDnn {
    fn channels(&self) -> usize {
        self.0.channels()
    }
    fn dec_i16(&mut self, d: Option<&[u8]>, p: &mut [i16], n: i32, f: i32) -> Result<usize, i32> {
        self.0.dec_i16(d, p, n, f)
    }
    fn dec_i24(&mut self, d: Option<&[u8]>, p: &mut [i32], n: i32, f: i32) -> Result<usize, i32> {
        self.0.dec_i24(d, p, n, f)
    }
    fn dec_f32(&mut self, d: Option<&[u8]>, p: &mut [f32], n: i32, f: i32) -> Result<usize, i32> {
        self.0.dec_f32(d, p, n, f)
    }
    fn ctl_set(&mut self, req: i32, value: i32) -> Result<(), i32> {
        DecApi::ctl_set(&mut self.0, req, value)
    }
    fn ctl_get(&mut self, req: i32) -> Result<i32, i32> {
        DecApi::ctl_get(&mut self.0, req)
    }
    fn states(&mut self) -> Vec<crate::dec::StateDump> {
        Vec::new()
    }
    fn extra_state(&mut self) -> Vec<u32> {
        let (ints, floats) = self.0.dnn_debug_state();
        state_bits(&ints, &floats)
    }
}

/// A C decoder of the DNN oracle (compiled-in weights).
#[derive(Debug)]
pub struct CDnn(pub di::Dec);

impl DecApi for CDnn {
    fn channels(&self) -> usize {
        unreachable!("the channel count is taken from the Rust side")
    }
    fn dec_i16(&mut self, d: Option<&[u8]>, p: &mut [i16], n: i32, f: i32) -> Result<usize, i32> {
        self.0.decode(d, p, n, f)
    }
    fn dec_i24(&mut self, d: Option<&[u8]>, p: &mut [i32], n: i32, f: i32) -> Result<usize, i32> {
        self.0.decode24(d, p, n, f)
    }
    fn dec_f32(&mut self, d: Option<&[u8]>, p: &mut [f32], n: i32, f: i32) -> Result<usize, i32> {
        self.0.decode_float(d, p, n, f)
    }
    fn ctl_set(&mut self, req: i32, value: i32) -> Result<(), i32> {
        self.0.ctl_set(req, value)
    }
    fn ctl_get(&mut self, req: i32) -> Result<i32, i32> {
        self.0.ctl_get(req)
    }
    fn states(&mut self) -> Vec<crate::dec::StateDump> {
        Vec::new()
    }
    fn extra_state(&mut self) -> Vec<u32> {
        let s = self.0.dump();
        state_bits(&s.ints, &s.floats)
    }
}
