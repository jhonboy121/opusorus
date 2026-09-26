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
//! The compiled-in tables of libopus depend on `DISABLE_DEBUG_FLOAT`: by default the
//! int8-quantized layers have no float copy; with `--enable-dnn-debug-float` (feature
//! `dnn-debug-float`) they have one and the DNN computes with it. The embedded blob must match
//! the feature (checked at compile time): generate it with `scripts/gen_dnn_blob.sh`, or
//! `scripts/gen_dnn_blob.sh --debug-float` for `dnn-debug-float`.
//!
//! `set_dnn_blob` still works and replaces the embedded models.
//!
//! With `std`, each model of the embedded blob is bound once per process and shared (`Arc`)
//! by every decoder and encoder, as C shares its compiled-in tables; without `std` (no
//! `OnceLock`) each creation binds its own copy.

/// The embedded weight blob (the file named by `OPUSORUS_DNN_BLOB` at build time).
///
/// A `static`, so that every use has the same address: the models bound from it are cached
/// and shared by address (`parse_lpcnet_weights::load_shared`).
pub static DNN_BLOB: &[u8] = BLOB;

/// The embedded bytes (a `const` for the compile-time validation below).
const BLOB: &[u8] = include_bytes!(env!(
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
    is_weight_blob(BLOB),
    "OPUSORUS_DNN_BLOB is not a libopus DNN weight blob (generate one with scripts/gen_dnn_blob.sh)"
);

/// The NUL-terminated name of the record at `pos` (a well-formed blob, see [`is_weight_blob`]).
const fn record_name(b: &[u8], pos: usize) -> &[u8] {
    let name = b
        .split_at(pos + 20)
        .1
        .split_at(super::nnet::WEIGHT_BLOCK_SIZE - 20)
        .0;
    let mut n = 0;
    while name[n] != 0 {
        n += 1;
    }
    name.split_at(n).0
}

/// Offset of the record after the one at `pos` (a well-formed blob).
const fn next_record(b: &[u8], pos: usize) -> usize {
    let block_size = i32::from_le_bytes([b[pos + 16], b[pos + 17], b[pos + 18], b[pos + 19]]);
    pos + super::nnet::WEIGHT_BLOCK_SIZE + block_size as usize
}

/// Counts the int8-quantized weight arrays (`<layer>_weights_int8`) of a well-formed blob that
/// are / are not followed by their float copy `<layer>_weights_float`, as the generated tables
/// order them when `DISABLE_DEBUG_FLOAT` is undefined (`--enable-dnn-debug-float`).
const fn int8_float_copies(b: &[u8]) -> (usize, usize) {
    const INT8: &[u8] = b"_int8";
    const FLOAT: &[u8] = b"_float";
    let (mut with, mut without) = (0, 0);
    let mut pos = 0;
    while pos < b.len() {
        let name = record_name(b, pos);
        let next = next_record(b, pos);
        if name.len() > INT8.len() && ends_with(name, INT8) {
            let prefix = name.split_at(name.len() - INT8.len()).0;
            let mut copy = false;
            if next < b.len() {
                let n2 = record_name(b, next);
                copy = n2.len() == prefix.len() + FLOAT.len()
                    && starts_with(n2, prefix)
                    && ends_with(n2, FLOAT);
            }
            if copy {
                with += 1;
            } else {
                without += 1;
            }
        }
        pos = next;
    }
    (with, without)
}

const fn starts_with(s: &[u8], p: &[u8]) -> bool {
    if s.len() < p.len() {
        return false;
    }
    let mut i = 0;
    while i < p.len() {
        if s[i] != p[i] {
            return false;
        }
        i += 1;
    }
    true
}

const fn ends_with(s: &[u8], p: &[u8]) -> bool {
    if s.len() < p.len() {
        return false;
    }
    starts_with(s.split_at(s.len() - p.len()).1, p)
}

#[cfg(feature = "dnn-debug-float")]
const _: () = assert!(
    int8_float_copies(BLOB).1 == 0,
    "feature dnn-debug-float: OPUSORUS_DNN_BLOB lacks the float copies of the int8 layers \
     (generate it with scripts/gen_dnn_blob.sh --debug-float)"
);
#[cfg(not(feature = "dnn-debug-float"))]
const _: () = assert!(
    int8_float_copies(BLOB).0 == 0,
    "OPUSORUS_DNN_BLOB has float copies of the int8 layers (a --debug-float blob): enable \
     the dnn-debug-float feature or generate the blob with scripts/gen_dnn_blob.sh"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_float_copies_match_the_feature() {
        let (with, without) = int8_float_copies(DNN_BLOB);
        assert!(with + without > 0, "the models have int8 layers");
        if cfg!(feature = "dnn-debug-float") {
            assert_eq!(without, 0);
        } else {
            assert_eq!(with, 0);
        }
    }

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
