//! Compiled-in DNN weights (feature `dnn-weights-embedded`): the equivalent of a libopus build
//! without `USE_WEIGHTS_FILE`, whose model tables are compiled into the library.
//!
//! The weight blob is embedded with `include_bytes!` from the path in the `OPUSORUS_DNN_BLOB`
//! environment variable at build time (an absolute path; `scripts/gen_dnn_blob.sh` writes one
//! holding every model of a full DNN build). The decoders and encoders then bind their models
//! at initialization exactly as upstream binds its compiled-in tables:
//!
//! | C (compiled-in weights) | Rust |
//! |---|---|
//! | `lpcnet_plc_init` / `silk_InitDecoder` (`silk_LoadOSCEModels(NULL)`) | [`crate::Decoder::new`] |
//! | `dred_encoder_init` (`init_rdovaeenc`) | [`crate::Encoder::new`] |
//! | `opus_dred_decoder_init` (`init_rdovaedec`) | `DredDecoder::new` / `DredDecoder::init` |
//!
//! A blob that cannot provide a model an enabled feature needs makes `Decoder::new` /
//! `Encoder::new` fail with [`crate::Error::InternalError`] (C: `celt_assert` on the
//! `init_*` result) and leaves a `DredDecoder` unloaded (C: `opus_dred_decoder_init` returns
//! `OPUS_UNIMPLEMENTED`). The structure of the blob is validated at compile time.
//!
//! `set_dnn_blob` still works and replaces the embedded models.

/// The embedded weight blob (the file named by `OPUSORUS_DNN_BLOB` at build time).
pub const DNN_BLOB: &[u8] = include_bytes!(env!(
    "OPUSORUS_DNN_BLOB",
    "the `dnn-weights-embedded` feature needs OPUSORUS_DNN_BLOB set to the absolute path of a \
     libopus DNN weight blob; generate one with scripts/gen_dnn_blob.sh (after \
     scripts/fetch_dnn_models.sh) and build with OPUSORUS_DNN_BLOB=$PWD/target/dnn/weights_blob.bin"
));

/// Whether `b` is a well-formed weight blob: a non-empty sequence of records with the `DNNw`
/// magic, the current `WEIGHT_BLOB_VERSION`, a NUL-terminated name and a payload that fits
/// (the checks of `parse_weights`, plus magic and version to catch a wrong file early).
const fn is_weight_blob(b: &[u8]) -> bool {
    use super::nnet::{WEIGHT_BLOB_VERSION, WEIGHT_BLOCK_SIZE};
    const fn read_i32(b: &[u8], off: usize) -> i32 {
        i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    }
    if b.is_empty() {
        return false;
    }
    let mut pos = 0;
    while pos < b.len() {
        let rem = b.len() - pos;
        if rem < WEIGHT_BLOCK_SIZE {
            return false;
        }
        if b[pos] != b'D' || b[pos + 1] != b'N' || b[pos + 2] != b'N' || b[pos + 3] != b'w' {
            return false;
        }
        if read_i32(b, pos + 4) != WEIGHT_BLOB_VERSION {
            return false;
        }
        let size = read_i32(b, pos + 12);
        let block_size = read_i32(b, pos + 16);
        if size <= 0 || block_size < size || block_size as usize > rem - WEIGHT_BLOCK_SIZE {
            return false;
        }
        if b[pos + WEIGHT_BLOCK_SIZE - 1] != 0 {
            return false;
        }
        pos += WEIGHT_BLOCK_SIZE + block_size as usize;
    }
    true
}

const _: () = assert!(
    is_weight_blob(DNN_BLOB),
    "OPUSORUS_DNN_BLOB is not a libopus DNN weight blob (generate one with scripts/gen_dnn_blob.sh)"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_validation() {
        assert!(is_weight_blob(DNN_BLOB));
        assert!(!is_weight_blob(&[]));
        assert!(!is_weight_blob(&DNN_BLOB[..DNN_BLOB.len() - 1]));
        let mut bad = DNN_BLOB[..128].to_vec();
        bad[0] = b'X';
        assert!(!is_weight_blob(&bad));
    }

    /// Every model of the enabled features is bound at initialization, as with upstream's
    /// compiled-in weights.
    #[test]
    fn models_loaded_at_init() {
        #[cfg(any(feature = "deep-plc", feature = "osce"))]
        {
            let mut d = crate::Decoder::new(48000, 2).unwrap();
            assert_eq!(
                d.dnn_loaded(),
                (cfg!(feature = "deep-plc"), cfg!(feature = "osce"))
            );
            d.init(16000, 1).unwrap();
            d.reset();
            assert_eq!(
                d.dnn_loaded(),
                (cfg!(feature = "deep-plc"), cfg!(feature = "osce"))
            );
        }
        #[cfg(feature = "dred")]
        {
            let mut e = crate::Encoder::new(16000, 1, crate::Application::Voip).unwrap();
            assert!(e.dred_loaded());
            e.init(48000, 2, crate::Application::Audio).unwrap();
            assert!(e.dred_loaded());
            let mut dd = crate::dred::DredDecoder::new();
            assert!(dd.loaded());
            dd.init();
            assert!(dd.loaded());
        }
    }
}
