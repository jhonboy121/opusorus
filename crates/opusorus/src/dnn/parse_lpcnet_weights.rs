//! Port of `dnn/parse_lpcnet_weights.c` (weight blob parsing and the `linear_init` /
//! `conv2d_init` layer binding helpers) and of `write_weights` from
//! `dnn/write_lpcnet_weights.c` (the blob writer, used as the format reference).
//!
//! ## Blob format
//! A blob is a sequence of records. Each record is a 64-byte `WeightHead`
//! (`char head[4]="DNNw"; int version; int type; int size; int block_size; char name[44]`,
//! little-endian as written by `fwrite` on the supported targets) followed by `block_size`
//! bytes whose first `size` bytes are the array payload (`block_size` is `size` rounded up to a
//! multiple of 64). `head` and `version` are not checked by the parser, exactly as in C.
//!
//! The parsed [`WeightArray`]s borrow the blob; [`linear_init`] / [`conv2d_init`] copy the
//! bound arrays into owned [`LinearLayer`] / [`Conv2dLayer`] storage.

use alloc::sync::Arc;
use alloc::vec::Vec;

use super::nnet::{Conv2dLayer, LinearLayer, WEIGHT_BLOB_VERSION, WEIGHT_BLOCK_SIZE};
use crate::{Error, Result};

/// `SPARSE_BLOCK_SIZE`: weights per 8x4 sparse block.
pub const SPARSE_BLOCK_SIZE: usize = 32;

/// Size of `WeightHead::name` (including the terminating NUL).
pub const WEIGHT_NAME_SIZE: usize = 44;

/// `WeightArray`: one named array of a weight list. `name` excludes the NUL terminator;
/// `data` is the `size`-byte payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightArray<'a> {
    pub name: &'a [u8],
    pub type_: i32,
    pub size: i32,
    pub data: &'a [u8],
}

#[inline]
fn read_i32(b: &[u8], off: usize) -> i32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(&b[off..off + 4]);
    i32::from_le_bytes(a)
}

/// Port of dnn/parse_lpcnet_weights.c:parse_record.
///
/// Parses the record at the start of `*data` and advances `*data` past it. Returns `None` where
/// C returns -1 (`*data` is then left unchanged).
#[must_use]
pub fn parse_record<'a>(data: &mut &'a [u8]) -> Option<WeightArray<'a>> {
    let d: &'a [u8] = data;
    let len = d.len();
    if len < WEIGHT_BLOCK_SIZE {
        return None;
    }
    let type_ = read_i32(d, 8);
    let size = read_i32(d, 12);
    let block_size = read_i32(d, 16);
    let name_field = &d[20..20 + WEIGHT_NAME_SIZE];
    if block_size < size {
        return None;
    }
    if i64::from(block_size) > len as i64 - WEIGHT_BLOCK_SIZE as i64 {
        return None;
    }
    if name_field[WEIGHT_NAME_SIZE - 1] != 0 {
        return None;
    }
    if size < 0 {
        return None;
    }
    // Here 0 <= size <= block_size <= len - WEIGHT_BLOCK_SIZE.
    // Terminates: name_field[WEIGHT_NAME_SIZE-1] == 0.
    let mut name_len = 0;
    while name_field[name_len] != 0 {
        name_len += 1;
    }
    let payload = &d[WEIGHT_BLOCK_SIZE..];
    *data = &payload[block_size as usize..];
    Some(WeightArray {
        name: &name_field[..name_len],
        type_,
        size,
        data: &payload[..size as usize],
    })
}

/// Port of dnn/parse_lpcnet_weights.c:parse_weights.
///
/// Returns the arrays of the blob in order (C also appends a `NULL`-named terminator, which a
/// slice does not need). Fails (`BadArg`, C: -1) on a malformed record or a zero-sized array.
pub fn parse_weights(data: &[u8]) -> Result<Vec<WeightArray<'_>>> {
    let mut list = Vec::with_capacity(20);
    let mut data = data;
    while !data.is_empty() {
        match parse_record(&mut data) {
            Some(array) if array.size > 0 => list.push(array),
            _ => return Err(Error::BadArg),
        }
    }
    Ok(list)
}

/// Port of dnn/parse_lpcnet_weights.c:find_array_entry (`None` is the C `NULL`-named end
/// entry).
#[must_use]
pub fn find_array_entry<'b, 'a>(
    arrays: &'b [WeightArray<'a>],
    name: &str,
) -> Option<&'b WeightArray<'a>> {
    arrays.iter().find(|a| a.name == name.as_bytes())
}

/// Port of dnn/parse_lpcnet_weights.c:find_array_check: the payload of the first array named
/// `name` if its size is `size` bytes.
#[must_use]
pub fn find_array_check<'a>(
    arrays: &[WeightArray<'a>],
    name: &str,
    size: Option<usize>,
) -> Option<&'a [u8]> {
    let a = find_array_entry(arrays, name)?;
    if size.is_some_and(|s| a.size as usize == s) {
        Some(a.data)
    } else {
        None
    }
}

/// Port of dnn/parse_lpcnet_weights.c:opt_array_check: `Ok(None)` if absent, an error (C:
/// `*error = 1`) if present with the wrong size.
pub fn opt_array_check<'a>(
    arrays: &[WeightArray<'a>],
    name: &str,
    size: Option<usize>,
) -> Result<Option<&'a [u8]>> {
    match find_array_entry(arrays, name) {
        None => Ok(None),
        Some(a) if size.is_some_and(|s| a.size as usize == s) => Ok(Some(a.data)),
        Some(_) => Err(Error::BadArg),
    }
}

/// Port of dnn/parse_lpcnet_weights.c:find_idx_check: validates a sparse index array for an
/// `nb_in x nb_out` layer and returns it with its total number of 8x4 blocks.
///
/// Deviations (C undefined behaviour): a negative block count (C loops forever) or a negative
/// block position (C accepts it and later reads before the input) is rejected.
#[must_use]
pub fn find_idx_check<'a>(
    arrays: &[WeightArray<'a>],
    name: &str,
    nb_in: usize,
    nb_out: usize,
) -> Option<(&'a [u8], usize)> {
    let a = find_array_entry(arrays, name)?;
    let idx = a.data;
    let mut total_blocks: i64 = 0;
    let mut remain = i64::from(a.size) / 4;
    let mut nb_out = nb_out as i64;
    let nb_in = nb_in as i64;
    let mut p = 0usize;
    while remain > 0 {
        let nb_blocks = i64::from(read_i32(idx, 4 * p));
        p += 1;
        if nb_blocks < 0 || remain < nb_blocks + 1 {
            return None;
        }
        for _ in 0..nb_blocks {
            let pos = i64::from(read_i32(idx, 4 * p));
            p += 1;
            if pos < 0 || pos + 3 >= nb_in || (pos & 0x3) != 0 {
                return None;
            }
        }
        nb_out -= 8;
        remain -= nb_blocks + 1;
        total_blocks += nb_blocks;
    }
    if nb_out != 0 {
        return None;
    }
    Some((idx, total_blocks as usize))
}

fn to_f32_vec(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

fn to_i32_vec(b: &[u8]) -> Vec<i32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|&c| i32::from_le_bytes(c))
        .collect()
}

fn to_i8_vec(b: &[u8]) -> Vec<i8> {
    b.iter().map(|&v| v as i8).collect()
}

/// `n * elem` bytes, `None` on overflow or beyond the C `int` range (never matches a record).
fn bytes(n: usize, elem: usize) -> Option<usize> {
    n.checked_mul(elem).filter(|&v| v <= i32::MAX as usize)
}

/// Port of dnn/parse_lpcnet_weights.c:linear_init.
///
/// Binds the named arrays (C `NULL` names are `None`) into a [`LinearLayer`] with
/// `nb_inputs` inputs and `nb_outputs` outputs. Fails with `BadArg` where C returns 1.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the C signature of linear_init"
)]
pub fn linear_init(
    arrays: &[WeightArray<'_>],
    bias: Option<&str>,
    subias: Option<&str>,
    weights: Option<&str>,
    float_weights: Option<&str>,
    weights_idx: Option<&str>,
    diag: Option<&str>,
    scale: Option<&str>,
    nb_inputs: usize,
    nb_outputs: usize,
) -> Result<LinearLayer> {
    let mut layer = LinearLayer::default();
    if let Some(name) = bias {
        let d = find_array_check(arrays, name, bytes(nb_outputs, 4)).ok_or(Error::BadArg)?;
        layer.bias = Some(to_f32_vec(d));
    }
    if let Some(name) = subias {
        let d = find_array_check(arrays, name, bytes(nb_outputs, 4)).ok_or(Error::BadArg)?;
        layer.subias = Some(to_f32_vec(d));
    }
    if let Some(name) = weights_idx {
        let (idx, total_blocks) =
            find_idx_check(arrays, name, nb_inputs, nb_outputs).ok_or(Error::BadArg)?;
        layer.weights_idx = Some(to_i32_vec(&idx[..idx.len() / 4 * 4]));
        let nw = SPARSE_BLOCK_SIZE.checked_mul(total_blocks);
        if let Some(name) = weights {
            let d = find_array_check(arrays, name, nw.and_then(|n| bytes(n, 1)))
                .ok_or(Error::BadArg)?;
            layer.weights = Some(to_i8_vec(d));
        }
        if let Some(name) = float_weights {
            layer.float_weights =
                opt_array_check(arrays, name, nw.and_then(|n| bytes(n, 4)))?.map(to_f32_vec);
        }
    } else {
        let nw = nb_inputs.checked_mul(nb_outputs);
        if let Some(name) = weights {
            let d = find_array_check(arrays, name, nw.and_then(|n| bytes(n, 1)))
                .ok_or(Error::BadArg)?;
            layer.weights = Some(to_i8_vec(d));
        }
        if let Some(name) = float_weights {
            layer.float_weights =
                opt_array_check(arrays, name, nw.and_then(|n| bytes(n, 4)))?.map(to_f32_vec);
        }
    }
    if let Some(name) = diag {
        let d = find_array_check(arrays, name, bytes(nb_outputs, 4)).ok_or(Error::BadArg)?;
        layer.diag = Some(to_f32_vec(d));
    }
    if weights.is_some() {
        // C calls find_array_check(arrays, scale, ...) here; a NULL scale name would crash.
        let name = scale.ok_or(Error::BadArg)?;
        let d = find_array_check(arrays, name, bytes(nb_outputs, 4)).ok_or(Error::BadArg)?;
        layer.scale = Some(to_f32_vec(d));
    }
    layer.nb_inputs = nb_inputs;
    layer.nb_outputs = nb_outputs;
    Ok(layer)
}

/// Port of dnn/parse_lpcnet_weights.c:conv2d_init. Fails with `BadArg` where C returns 1.
pub fn conv2d_init(
    arrays: &[WeightArray<'_>],
    bias: Option<&str>,
    float_weights: Option<&str>,
    in_channels: usize,
    out_channels: usize,
    ktime: usize,
    kheight: usize,
) -> Result<Conv2dLayer> {
    let mut layer = Conv2dLayer::default();
    if let Some(name) = bias {
        let d = find_array_check(arrays, name, bytes(out_channels, 4)).ok_or(Error::BadArg)?;
        layer.bias = Some(to_f32_vec(d));
    }
    if let Some(name) = float_weights {
        let n = in_channels
            .checked_mul(out_channels)
            .and_then(|v| v.checked_mul(ktime))
            .and_then(|v| v.checked_mul(kheight));
        layer.float_weights =
            opt_array_check(arrays, name, n.and_then(|n| bytes(n, 4)))?.map(to_f32_vec);
    }
    layer.in_channels = in_channels;
    layer.out_channels = out_channels;
    layer.ktime = ktime;
    layer.kheight = kheight;
    Ok(layer)
}

/// Port of dnn/write_lpcnet_weights.c:write_weights: appends the records of `list` to `out`.
///
/// Names are truncated to 43 bytes like the C `strncpy` + terminator (C also prints a warning);
/// `data` must hold at least `size` bytes.
pub fn write_weights(list: &[WeightArray<'_>], out: &mut Vec<u8>) {
    for a in list {
        let size = a.size.max(0) as usize;
        let block_size = size.div_ceil(WEIGHT_BLOCK_SIZE) * WEIGHT_BLOCK_SIZE;
        out.extend_from_slice(b"DNNw");
        out.extend_from_slice(&WEIGHT_BLOB_VERSION.to_le_bytes());
        out.extend_from_slice(&a.type_.to_le_bytes());
        out.extend_from_slice(&a.size.to_le_bytes());
        out.extend_from_slice(&(block_size as i32).to_le_bytes());
        let mut name = [0u8; WEIGHT_NAME_SIZE];
        let n = a.name.len().min(WEIGHT_NAME_SIZE - 1);
        name[..n].copy_from_slice(&a.name[..n]);
        out.extend_from_slice(&name);
        out.extend_from_slice(&a.data[..size]);
        out.resize(out.len() + (block_size - size), 0);
    }
}

/// Cache of one model bound from the embedded weight blob (`dnn-weights-embedded` with `std`):
/// the model is bound once per process and shared by every state that loads it.
#[cfg(all(
    feature = "std",
    feature = "deep-plc",
    feature = "dnn-weights-embedded"
))]
pub type ModelCache<T> = std::sync::OnceLock<Result<Arc<T>>>;

/// Cache of one model bound from the embedded weight blob: without `std` (no `OnceLock`) or
/// without an embedded blob there is nothing to cache.
#[cfg(not(all(
    feature = "std",
    feature = "deep-plc",
    feature = "dnn-weights-embedded"
)))]
#[derive(Debug)]
pub struct ModelCache<T>(core::marker::PhantomData<fn() -> T>);

#[cfg(not(all(
    feature = "std",
    feature = "deep-plc",
    feature = "dnn-weights-embedded"
)))]
impl<T> ModelCache<T> {
    /// An empty cache.
    #[must_use]
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

#[cfg(not(all(
    feature = "std",
    feature = "deep-plc",
    feature = "dnn-weights-embedded"
)))]
impl<T> Default for ModelCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Binds a model from the weight blob `data` with `init` (`parse_weights` + `init_*`) into a
/// shared [`Arc`]. The embedded blob (`dnn-weights-embedded`, with `std`) is bound only once:
/// every later load returns the model kept in `cache`, so the decoders and encoders created
/// with compiled-in weights share one copy of the weights instead of each parsing and copying
/// the blob (C points to its compiled-in tables).
///
/// # Errors
/// As `parse_weights` / `init`.
#[cfg_attr(
    not(all(
        feature = "std",
        feature = "deep-plc",
        feature = "dnn-weights-embedded"
    )),
    expect(
        unused_variables,
        reason = "the cache only exists for the embedded blob with std"
    )
)]
pub fn load_shared<T>(
    data: &[u8],
    init: fn(&[WeightArray<'_>]) -> Result<T>,
    cache: &'static ModelCache<T>,
) -> Result<Arc<T>> {
    #[cfg(all(
        feature = "std",
        feature = "deep-plc",
        feature = "dnn-weights-embedded"
    ))]
    if core::ptr::eq(data, super::embedded::DNN_BLOB) {
        return cache.get_or_init(|| bind(data, init)).clone();
    }
    bind(data, init)
}

/// `parse_weights` then `init`, into a new [`Arc`].
fn bind<T>(data: &[u8], init: fn(&[WeightArray<'_>]) -> Result<T>) -> Result<Arc<T>> {
    let list = parse_weights(data)?;
    Ok(Arc::new(init(&list)?))
}
