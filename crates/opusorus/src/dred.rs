//! Deep REDundancy (DRED) decoding API (feature `dred`): port of the `OpusDREDDecoder` /
//! `OpusDRED` functions of `src/opus_decoder.c` (libopus `ENABLE_DRED`).
//!
//! An encoder with a DRED duration ([`crate::Encoder::set_dred_duration`]) attaches to its
//! packets an extension carrying a heavily compressed description (RDOVAE latents) of up to
//! 1.04 s of past audio. After a burst of losses, the receiver parses the redundancy of the next
//! packet that arrives and conceals the lost audio from it:
//!
//! 1. [`DredDecoder::parse`] (`opus_dred_parse`) extracts the redundancy of a packet into a
//!    [`Dred`] and returns how many samples before the packet it covers;
//! 2. [`DredDecoder::process`] (`opus_dred_process`, done by `parse` unless deferred) runs the
//!    RDOVAE decoder to get the acoustic features;
//! 3. [`crate::Decoder::dred_decode`] (`opus_decoder_dred_decode`) synthesizes the missing audio
//!    with the deep PLC (FARGAN), which needs its model loaded with
//!    [`crate::Decoder::set_dnn_blob`].
//!
//! | C | Rust |
//! |---|---|
//! | `opus_dred_decoder_get_size` / `_create` / `_init` / `_destroy` | [`DredDecoder::get_size`] / [`DredDecoder::new`] / [`DredDecoder::init`] / `Drop` |
//! | `opus_dred_decoder_ctl(OPUS_SET_DNN_BLOB)` | [`DredDecoder::set_dnn_blob`] (numeric: [`DredDecoder::ctl_set`]) |
//! | `opus_dred_get_size` / `opus_dred_alloc` / `opus_dred_free` | [`Dred::get_size`] / [`Dred::new`] / `Drop` |
//! | `opus_dred_parse` | [`DredDecoder::parse`] |
//! | `opus_dred_process` | [`DredDecoder::process`] / [`DredDecoder::process_in_place`] |
//! | `opus_decoder_dred_decode{,24,_float}` | [`crate::Decoder::dred_decode`] / [`crate::Decoder::dred_decode24`] / [`crate::Decoder::dred_decode_float`] |
//!
//! Weights: the RDOVAE decoder model is not compiled in (upstream `USE_WEIGHTS_FILE`, PLAN
//! D-015). A new [`DredDecoder`] is therefore not loaded (`parse` / `process` return
//! [`Error::Unimplemented`], like upstream) until [`DredDecoder::set_dnn_blob`] binds a libopus
//! weight blob holding the `rdovaedec` arrays.
//!
//! Deviations from C (Rust-only guards where C has undefined behaviour): a non-positive
//! `sampling_rate` in [`DredDecoder::parse`] is [`Error::BadArg`]; a [`Dred`] is zeroed at
//! allocation (C `opus_dred_alloc` does not initialize it).
//!
//! ```
//! use opusorus::dred::{Dred, DredDecoder};
//!
//! let dred_dec = DredDecoder::new();
//! let mut dred = Dred::new();
//! // Without a loaded RDOVAE model (see `DredDecoder::set_dnn_blob`) parsing is unavailable.
//! assert_eq!(
//!     dred_dec.parse(&mut dred, &[0xF8, 0xFF, 0xFE], 48000, 48000, false),
//!     Err(opusorus::Error::Unimplemented)
//! );
//! ```

use alloc::boxed::Box;

use crate::dnn::dred_coding::{
    DRED_EXPERIMENTAL_BYTES, DRED_EXPERIMENTAL_VERSION, DRED_EXTENSION_ID,
    DRED_NUM_REDUNDANCY_FRAMES,
};
use crate::dnn::dred_decoder::{OpusDred, dred_ec_decode};
use crate::dnn::dred_rdovae::{RdovaeDec, dred_rdovae_dec_load_model, dred_rdovae_decode_all};
use crate::extensions::ExtensionIterator;
use crate::packet::{parse_impl, toc_samples_per_frame};
use crate::{Error, Result};

/// `OPUS_SET_DNN_BLOB_REQUEST`.
pub const OPUS_SET_DNN_BLOB_REQUEST: i32 = 4052;

/// The DRED decoder (C `OpusDREDDecoder`): the RDOVAE decoder model used to turn parsed DRED
/// latents into acoustic features. It is stateless between packets and can be shared by several
/// streams.
#[derive(Debug, Clone)]
pub struct DredDecoder {
    /// `model` (boxed: the RDOVAE decoder layers).
    model: Box<RdovaeDec>,
    /// `loaded`.
    loaded: bool,
}

impl Default for DredDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl DredDecoder {
    /// Port of `opus_dred_decoder_create` / `opus_dred_decoder_init`: a DRED decoder without a
    /// model (upstream `USE_WEIGHTS_FILE`: `init` succeeds and leaves `loaded` at 0). Load one
    /// with [`DredDecoder::set_dnn_blob`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            model: Box::default(),
            loaded: false,
        }
    }

    /// Port of `opus_dred_decoder_init`: re-initializes the decoder. With upstream
    /// `USE_WEIGHTS_FILE` semantics this clears `loaded` (the model must be loaded again).
    pub const fn init(&mut self) {
        self.loaded = false;
    }

    /// Port of `opus_dred_decoder_get_size` (the Rust footprint of the struct and its model
    /// box, not the C `sizeof`).
    #[must_use]
    pub const fn get_size() -> usize {
        size_of::<Self>() + size_of::<RdovaeDec>()
    }

    /// Whether an RDOVAE decoder model is loaded (C `loaded`).
    #[must_use]
    pub const fn loaded(&self) -> bool {
        self.loaded
    }

    /// `opus_dred_decoder_ctl(OPUS_SET_DNN_BLOB)` (port of `dred_decoder_load_model`): binds
    /// the RDOVAE decoder from a libopus weight blob.
    ///
    /// # Errors
    /// [`Error::BadArg`] if the blob cannot be parsed or lacks an `rdovaedec` layer (the
    /// previous model, if any, is kept).
    pub fn set_dnn_blob(&mut self, data: &[u8]) -> Result<()> {
        match dred_rdovae_dec_load_model(data) {
            Ok(m) => {
                *self.model = m;
                self.loaded = true;
                Ok(())
            }
            Err(_) => Err(Error::BadArg),
        }
    }

    /// Numeric `opus_dred_decoder_ctl`: every request is [`Error::Unimplemented`], including
    /// the pointer-taking `OPUS_SET_DNN_BLOB` (use [`DredDecoder::set_dnn_blob`]), as in the C
    /// library built with compiled-in weights.
    ///
    /// # Errors
    /// Always [`Error::Unimplemented`].
    pub const fn ctl_set(&mut self, _request: i32, _value: i32) -> Result<()> {
        Err(Error::Unimplemented)
    }

    /// Port of `opus_dred_parse`: extracts the DRED redundancy of the Opus packet `data` into
    /// `dred`.
    ///
    /// * `max_dred_samples`: the maximum amount of redundancy to decode (at `sampling_rate`);
    /// * `sampling_rate`: the rate the sample counts are expressed in (normally the decoder's);
    /// * `defer_processing`: skip [`DredDecoder::process`] (to run it later, e.g. on another
    ///   thread).
    ///
    /// Returns `(offset, dred_end)`: `offset` (the C return value) is the positive offset, in
    /// samples before the packet's timestamp, of the first (oldest) sample the redundancy can
    /// reconstruct; `dred_end` is the number of non-encoded (silence) samples between the
    /// packet's timestamp and the newest DRED sample. A packet without DRED returns `(0, 0)`.
    ///
    /// # Errors
    /// [`Error::Unimplemented`] without a loaded model, [`Error::InvalidPacket`] for a malformed
    /// packet, [`Error::BadArg`] for a non-positive `sampling_rate` (Rust-only).
    pub fn parse(
        &self,
        dred: &mut Dred,
        data: &[u8],
        max_dred_samples: i32,
        sampling_rate: i32,
        defer_processing: bool,
    ) -> Result<(i32, i32)> {
        if !self.loaded {
            return Err(Error::Unimplemented);
        }
        let dred = &mut *dred.inner;
        dred.process_stage = -1;
        let (payload, dred_frame_offset) = dred_find_payload(data)?;
        if let Some(payload) = payload {
            if sampling_rate <= 0 {
                return Err(Error::BadArg);
            }
            // C: `100*max_dred_samples/sampling_rate` in int (overflow is UB; i64 here).
            let offset = (100 * i64::from(max_dred_samples) / i64::from(sampling_rate))
                .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
            let min_feature_frames =
                (2i32.saturating_add(offset)).min(2 * DRED_NUM_REDUNDANCY_FRAMES as i32);
            dred_ec_decode(dred, payload, min_feature_frames, dred_frame_offset);
            if !defer_processing {
                // Cannot fail: the stage is 1 and the model is loaded.
                self.process_stage(dred);
            }
            let sr = i64::from(sampling_rate);
            let dred_offset = i64::from(dred.dred_offset);
            let dred_end = (-dred_offset * sr / 400).max(0);
            let available = (i64::from(dred.nb_latents) * sr / 25 - dred_offset * sr / 400).max(0);
            return Ok((sat_i32(available), sat_i32(dred_end)));
        }
        Ok((0, 0))
    }

    /// Port of `opus_dred_process` with `src != dst`: copies `src` to `dst` and decodes the
    /// latents into features (a no-op copy when `src` is already processed).
    ///
    /// # Errors
    /// [`Error::BadArg`] if `src` has not been parsed; [`Error::Unimplemented`] without a
    /// loaded model.
    pub fn process(&self, src: &Dred, dst: &mut Dred) -> Result<()> {
        if src.inner.process_stage != 1 && src.inner.process_stage != 2 {
            return Err(Error::BadArg);
        }
        if !self.loaded {
            return Err(Error::Unimplemented);
        }
        dst.inner.clone_from(&src.inner);
        self.process_stage(&mut dst.inner);
        Ok(())
    }

    /// Port of `opus_dred_process` with `src == dst` (in place).
    ///
    /// # Errors
    /// As [`DredDecoder::process`].
    pub fn process_in_place(&self, dred: &mut Dred) -> Result<()> {
        if dred.inner.process_stage != 1 && dred.inner.process_stage != 2 {
            return Err(Error::BadArg);
        }
        if !self.loaded {
            return Err(Error::Unimplemented);
        }
        self.process_stage(&mut dred.inner);
        Ok(())
    }

    /// The body of `opus_dred_process` after the checks and the copy.
    fn process_stage(&self, dst: &mut OpusDred) {
        if dst.process_stage == 2 {
            return;
        }
        dred_rdovae_decode_all(
            &self.model,
            &mut dst.fec_features,
            &dst.state,
            &dst.latents,
            dst.nb_latents as usize,
        );
        dst.process_stage = 2;
    }
}

/// Saturating `i64` → `i32` (the C values are ints; they only differ on C overflow).
fn sat_i32(v: i64) -> i32 {
    v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// Port of `src/opus_decoder.c:dred_find_payload`: the DRED extension payload of `data` (after
/// the experimental header) and its position in the packet (C `dred_frame_offset`, units of
/// 2.5 ms), or `None` when the packet carries no DRED.
fn dred_find_payload(data: &[u8]) -> Result<(Option<&[u8]>, i32)> {
    // Get the padding section of the packet.
    let parsed = parse_impl(data, false)?;
    let nb_frames = parsed.nb_frames as i32;
    let frame_size = toc_samples_per_frame(data[0], 48000);
    let mut iter = ExtensionIterator::new(parsed.padding, nb_frames);
    let mut dred_frame_offset = 0;
    loop {
        let Some(ext) = iter.find(DRED_EXTENSION_ID)? else {
            return Ok((None, dred_frame_offset));
        };
        // DRED position in the packet, in units of 2.5 ms like for the signaled DRED offset.
        dred_frame_offset = ext.frame * frame_size / 120;
        // Check that temporary extension type and version match. This check will be removed
        // once extension is finalized.
        if ext.data.len() as i32 > DRED_EXPERIMENTAL_BYTES
            && ext.data[0] == b'D'
            && i32::from(ext.data[1]) == DRED_EXPERIMENTAL_VERSION
        {
            return Ok((
                Some(&ext.data[DRED_EXPERIMENTAL_BYTES as usize..]),
                dred_frame_offset,
            ));
        }
    }
}

/// Parsed DRED redundancy (C `OpusDRED`, about 11 KB, boxed): the output of
/// [`DredDecoder::parse`] / [`DredDecoder::process`], the input of
/// [`crate::Decoder::dred_decode`].
#[derive(Debug, Clone, PartialEq)]
pub struct Dred {
    inner: Box<OpusDred>,
}

impl Default for Dred {
    fn default() -> Self {
        Self::new()
    }
}

impl Dred {
    /// Port of `opus_dred_alloc` (zero-initialized, unlike C).
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Box::new(OpusDred::new()),
        }
    }

    /// Port of `opus_dred_get_size` (`sizeof(OpusDRED)`).
    #[must_use]
    pub const fn get_size() -> usize {
        size_of::<OpusDred>()
    }

    /// The C `OpusDRED` fields.
    #[doc(hidden)]
    #[must_use]
    pub fn inner(&self) -> &OpusDred {
        &self.inner
    }

    /// The C `OpusDRED` fields (mutable, for tests and the C ABI).
    #[doc(hidden)]
    pub fn inner_mut(&mut self) -> &mut OpusDred {
        &mut self.inner
    }

    /// `process_stage`: -1 after a failed/empty parse, 1 when parsed, 2 when processed.
    #[must_use]
    pub fn process_stage(&self) -> i32 {
        self.inner.process_stage
    }

    /// `nb_latents`: the number of 40 ms chunks of redundancy.
    #[must_use]
    pub fn nb_latents(&self) -> i32 {
        self.inner.nb_latents
    }

    /// `dred_offset`: the position of the redundancy, in 2.5 ms units.
    #[must_use]
    pub fn dred_offset(&self) -> i32 {
        self.inner.dred_offset
    }
}
