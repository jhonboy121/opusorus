//! Oracle bindings for unit `opus_packet`: packet parsing/soft clip (`src/opus.c`), extensions
//! (`src/extensions.c`), repacketizer (`src/repacketizer.c`), mapping matrix
//! (`src/mapping_matrix.c`), analysis MLP (`src/mlp.c`) and multistream layout helpers
//! (`src/opus_multistream.c`). Shims live in `csrc/opus_packet.c`.
//!
//! Public-API functions (`opus_packet_pad`, `opus_repacketizer_*`, ...) are in [`crate::api`].

use core::ffi::{c_int, c_void};

unsafe extern "C" {
    fn oracle_pk_align(i: c_int) -> c_int;
    fn oracle_pk_encode_size(size: c_int, data: *mut u8) -> c_int;
    fn oracle_pk_parse_impl(
        data: *const u8,
        len: c_int,
        self_delimited: c_int,
        toc: *mut u8,
        frame_off: *mut c_int,
        frame_len: *mut c_int,
        payload_offset: *mut c_int,
        packet_offset: *mut c_int,
        padding_off: *mut c_int,
        padding_len: *mut c_int,
    ) -> c_int;
    fn oracle_pk_soft_clip_impl(x: *mut f32, n: c_int, c: c_int, mem: *mut f32);

    fn oracle_ext_generate(
        out: *mut u8,
        use_null: c_int,
        len: c_int,
        ids: *const c_int,
        frames: *const c_int,
        offs: *const c_int,
        lens: *const c_int,
        n: c_int,
        buf: *const u8,
        nb_frames: c_int,
        pad: c_int,
    ) -> c_int;
    fn oracle_ext_count(data: *const u8, len: c_int, nb_frames: c_int) -> c_int;
    fn oracle_ext_count_ext(
        data: *const u8,
        len: c_int,
        nb_frame_exts: *mut c_int,
        nb_frames: c_int,
    ) -> c_int;
    fn oracle_ext_parse(
        data: *const u8,
        len: c_int,
        nb_frames: c_int,
        nb_ext: *mut c_int,
        ids: *mut c_int,
        frames: *mut c_int,
        offs: *mut c_int,
        lens: *mut c_int,
    ) -> c_int;
    fn oracle_ext_parse_ext(
        data: *const u8,
        len: c_int,
        nb_frame_exts: *const c_int,
        nb_frames: c_int,
        nb_ext: *mut c_int,
        ids: *mut c_int,
        frames: *mut c_int,
        offs: *mut c_int,
        lens: *mut c_int,
    ) -> c_int;
    fn oracle_ext_iter_ops(
        data: *const u8,
        len: c_int,
        nb_frames: c_int,
        ops: *const c_int,
        nops: c_int,
        res: *mut c_int,
    );

    fn oracle_rp_run(
        buf: *const u8,
        lens: *const c_int,
        npk: c_int,
        cat_sd: c_int,
        cat_rets: *mut c_int,
        nb_frames: *mut c_int,
        begin: c_int,
        end: c_int,
        out: *mut u8,
        maxlen: c_int,
        sd: c_int,
        pad: c_int,
        ids: *const c_int,
        frames: *const c_int,
        offs: *const c_int,
        elens: *const c_int,
        next: c_int,
        ext_buf: *const u8,
    ) -> c_int;
    fn oracle_pad_impl(
        data: *mut u8,
        len: c_int,
        new_len: c_int,
        pad: c_int,
        ids: *const c_int,
        frames: *const c_int,
        offs: *const c_int,
        elens: *const c_int,
        next: c_int,
        ext_buf: *const u8,
    ) -> c_int;

    fn oracle_mm_get_size(rows: c_int, cols: c_int) -> c_int;
    fn oracle_mm_multiply(
        kind: c_int,
        rows: c_int,
        cols: c_int,
        mdata: *const i16,
        input: *const c_void,
        in_rows: c_int,
        out: *mut c_void,
        row: c_int,
        out_rows: c_int,
        frame_size: c_int,
    );
    fn oracle_mm_static(
        idx: c_int,
        rows: *mut c_int,
        cols: *mut c_int,
        gain: *mut c_int,
        data: *mut i16,
        cap: c_int,
    ) -> c_int;

    fn oracle_mlp_dense(
        bias: *const i8,
        weights: *const i8,
        nb_inputs: c_int,
        nb_neurons: c_int,
        sigmoid: c_int,
        output: *mut f32,
        input: *const f32,
    );
    fn oracle_mlp_gru(
        bias: *const i8,
        weights: *const i8,
        recur: *const i8,
        nb_inputs: c_int,
        nb_neurons: c_int,
        state: *mut f32,
        input: *const f32,
    );
    fn oracle_mlp_builtin(which: c_int, out_or_state: *mut f32, input: *const f32);
    fn oracle_mlp_tansig(x: f32) -> f32;
    fn oracle_mlp_sigmoid(x: f32) -> f32;

    fn oracle_ms_validate_layout(
        nb_channels: c_int,
        nb_streams: c_int,
        nb_coupled: c_int,
        mapping: *const u8,
    ) -> c_int;
    fn oracle_ms_get_channel(
        which: c_int,
        nb_channels: c_int,
        nb_streams: c_int,
        nb_coupled: c_int,
        mapping: *const u8,
        stream_id: c_int,
        prev: c_int,
    ) -> c_int;
}

// ------------------------------------------------------------------------------- opus.c

/// `align` (`src/opus_private.h`).
pub fn align(i: i32) -> i32 {
    // SAFETY: pure function.
    unsafe { oracle_pk_align(i) }
}

/// `encode_size`: returns the bytes written (at most 2).
pub fn encode_size(size: i32) -> Vec<u8> {
    let mut buf = [0u8; 2];
    // SAFETY: buf has room for the 2-byte maximum.
    let n = unsafe { oracle_pk_encode_size(size, buf.as_mut_ptr()) };
    buf[..n as usize].to_vec()
}

/// Output of `opus_packet_parse_impl` as offsets into the input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parsed {
    /// TOC byte.
    pub toc: u8,
    /// (offset, length) of each frame.
    pub frames: Vec<(usize, usize)>,
    /// Payload offset.
    pub payload_offset: usize,
    /// Packet offset (size including padding).
    pub packet_offset: usize,
    /// (offset, length) of the padding.
    pub padding: (usize, usize),
}

/// `opus_packet_parse_impl`: `Err(code)` on failure.
pub fn parse_impl(data: &[u8], self_delimited: bool) -> Result<Parsed, i32> {
    let mut toc = 0u8;
    let mut off = [0 as c_int; 48];
    let mut len = [0 as c_int; 48];
    let (mut po, mut pko, mut pad_off, mut pad_len) = (0, 0, 0, 0);
    // SAFETY: data valid for its length (an empty slice is never dereferenced by C since
    // len == 0 is rejected first); out arrays hold 48 entries.
    let ret = unsafe {
        oracle_pk_parse_impl(
            data.as_ptr(),
            data.len() as c_int,
            c_int::from(self_delimited),
            &mut toc,
            off.as_mut_ptr(),
            len.as_mut_ptr(),
            &mut po,
            &mut pko,
            &mut pad_off,
            &mut pad_len,
        )
    };
    if ret < 0 {
        return Err(ret);
    }
    let n = ret as usize;
    Ok(Parsed {
        toc,
        frames: (0..n).map(|i| (off[i] as usize, len[i] as usize)).collect(),
        payload_offset: po as usize,
        packet_offset: pko as usize,
        padding: (pad_off as usize, pad_len as usize),
    })
}

/// `opus_pcm_soft_clip_impl` (arch 0).
pub fn soft_clip_impl(x: &mut [f32], n: i32, c: i32, mem: &mut [f32]) {
    assert!(n < 1 || c < 1 || (x.len() >= (n * c) as usize && mem.len() >= c as usize));
    // SAFETY: buffers sized as asserted (C returns early for n<1 / c<1).
    unsafe { oracle_pk_soft_clip_impl(x.as_mut_ptr(), n, c, mem.as_mut_ptr()) }
}

// ------------------------------------------------------------------------------- extensions.c

/// An extension description with its payload as (offset, len) into some buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ext {
    /// Extension ID.
    pub id: i32,
    /// Frame index.
    pub frame: i32,
    /// Payload offset into the buffer.
    pub off: i32,
    /// Payload length.
    pub len: i32,
}

fn split(exts: &[Ext]) -> [Vec<c_int>; 4] {
    [
        exts.iter().map(|e| e.id).collect(),
        exts.iter().map(|e| e.frame).collect(),
        exts.iter().map(|e| e.off).collect(),
        exts.iter().map(|e| e.len).collect(),
    ]
}

/// `opus_packet_extensions_generate`. With `out == None` measures only (`len` is the limit).
/// Payloads are taken from `buf`.
pub fn ext_generate(
    out: Option<&mut [u8]>,
    len: i32,
    exts: &[Ext],
    buf: &[u8],
    nb_frames: i32,
    pad: bool,
) -> i32 {
    assert!(exts.len() <= 512);
    for e in exts {
        assert!(e.off >= 0 && e.len >= 0 && (e.off + e.len) as usize <= buf.len());
    }
    let [ids, frames, offs, lens] = split(exts);
    let use_null = out.is_none();
    let mut dummy = [0u8; 1];
    let out = match out {
        Some(o) => {
            assert!(o.len() >= len as usize);
            o
        }
        None => &mut dummy[..],
    };
    // SAFETY: arrays hold exts.len() entries, payloads lie within buf, out holds len bytes
    // (or is unused when use_null).
    unsafe {
        oracle_ext_generate(
            out.as_mut_ptr(),
            c_int::from(use_null),
            len,
            ids.as_ptr(),
            frames.as_ptr(),
            offs.as_ptr(),
            lens.as_ptr(),
            exts.len() as c_int,
            buf.as_ptr(),
            nb_frames,
            c_int::from(pad),
        )
    }
}

/// `opus_packet_extensions_count`.
pub fn ext_count(data: &[u8], nb_frames: i32) -> i32 {
    // SAFETY: data valid for its length.
    unsafe { oracle_ext_count(data.as_ptr(), data.len() as c_int, nb_frames) }
}

/// `opus_packet_extensions_count_ext`: returns (count, per-frame counts).
pub fn ext_count_ext(data: &[u8], nb_frames: i32) -> (i32, Vec<i32>) {
    let mut per = vec![0 as c_int; nb_frames.max(1) as usize];
    // SAFETY: per has nb_frames entries.
    let n = unsafe {
        oracle_ext_count_ext(
            data.as_ptr(),
            data.len() as c_int,
            per.as_mut_ptr(),
            nb_frames,
        )
    };
    per.truncate(nb_frames.max(0) as usize);
    (n, per)
}

fn collect_ext(
    n: i32,
    ids: &[c_int],
    frames: &[c_int],
    offs: &[c_int],
    lens: &[c_int],
) -> Vec<Ext> {
    (0..n.max(0) as usize)
        .map(|i| Ext {
            id: ids[i],
            frame: frames[i],
            off: offs[i],
            len: lens[i],
        })
        .collect()
}

/// `opus_packet_extensions_parse` with capacity `cap`: (return value, count, extensions).
pub fn ext_parse(data: &[u8], nb_frames: i32, cap: usize) -> (i32, i32, Vec<Ext>) {
    assert!(cap <= 4096);
    let mut n = cap as c_int;
    let (mut ids, mut frames, mut offs, mut lens) =
        (vec![0; cap], vec![0; cap], vec![0; cap], vec![0; cap]);
    // SAFETY: output arrays hold cap entries.
    let ret = unsafe {
        oracle_ext_parse(
            data.as_ptr(),
            data.len() as c_int,
            nb_frames,
            &mut n,
            ids.as_mut_ptr(),
            frames.as_mut_ptr(),
            offs.as_mut_ptr(),
            lens.as_mut_ptr(),
        )
    };
    let exts = if ret >= 0 {
        collect_ext(n, &ids, &frames, &offs, &lens)
    } else {
        Vec::new()
    };
    (ret, n, exts)
}

/// `opus_packet_extensions_parse_ext` with capacity `cap`: (return value, count, extensions).
pub fn ext_parse_ext(data: &[u8], nb_frame_exts: &[i32], cap: usize) -> (i32, i32, Vec<Ext>) {
    assert!(cap <= 4096 && nb_frame_exts.len() <= 48);
    let mut n = cap as c_int;
    let (mut ids, mut frames, mut offs, mut lens) =
        (vec![0; cap], vec![0; cap], vec![0; cap], vec![0; cap]);
    // SAFETY: output arrays hold cap entries; nb_frame_exts has nb_frames entries.
    let ret = unsafe {
        oracle_ext_parse_ext(
            data.as_ptr(),
            data.len() as c_int,
            nb_frame_exts.as_ptr(),
            nb_frame_exts.len() as c_int,
            &mut n,
            ids.as_mut_ptr(),
            frames.as_mut_ptr(),
            offs.as_mut_ptr(),
            lens.as_mut_ptr(),
        )
    };
    let exts = if ret >= 0 {
        collect_ext(n, &ids, &frames, &offs, &lens)
    } else {
        Vec::new()
    };
    (ret, n, exts)
}

/// Iterator operation for [`ext_iter_ops`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterOp {
    /// `opus_extension_iterator_next`.
    Next,
    /// `opus_extension_iterator_find(id)`.
    Find(i32),
    /// `opus_extension_iterator_reset`.
    Reset,
    /// `opus_extension_iterator_set_frame_max(v)`.
    SetFrameMax(i32),
}

/// Runs iterator ops; returns `(ret, ext)` per op (`ext` meaningful when `ret > 0`).
pub fn ext_iter_ops(data: &[u8], nb_frames: i32, ops: &[IterOp]) -> Vec<(i32, Ext)> {
    let flat: Vec<c_int> = ops
        .iter()
        .flat_map(|o| match *o {
            IterOp::Next => [0, 0],
            IterOp::Find(id) => [1, id],
            IterOp::Reset => [2, 0],
            IterOp::SetFrameMax(v) => [3, v],
        })
        .collect();
    let mut res = vec![0 as c_int; 5 * ops.len()];
    // SAFETY: ops has 2*nops entries, res 5*nops.
    unsafe {
        oracle_ext_iter_ops(
            data.as_ptr(),
            data.len() as c_int,
            nb_frames,
            flat.as_ptr(),
            ops.len() as c_int,
            res.as_mut_ptr(),
        )
    };
    res.chunks(5)
        .map(|r| {
            (
                r[0],
                Ext {
                    id: r[1],
                    frame: r[2],
                    off: r[3],
                    len: r[4],
                },
            )
        })
        .collect()
}

// ------------------------------------------------------------------------------- repacketizer.c

/// Result of [`rp_run`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpRun {
    /// Return value of each cat.
    pub cat_rets: Vec<i32>,
    /// Frames in the repacketizer after the cats.
    pub nb_frames: i32,
    /// Return value of `opus_repacketizer_out_range_impl`.
    pub ret: i32,
}

/// Cats `packets` (self-delimited if `cat_sd`) into a fresh repacketizer, then calls
/// `opus_repacketizer_out_range_impl(begin, end, out, out.len(), sd, pad, exts)` with
/// extension payloads from `ext_buf`.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the C out_range_impl signature"
)]
pub fn rp_run(
    packets: &[&[u8]],
    cat_sd: bool,
    begin: i32,
    end: i32,
    out: &mut [u8],
    sd: bool,
    pad: bool,
    exts: &[Ext],
    ext_buf: &[u8],
) -> RpRun {
    assert!(exts.len() <= 512);
    let buf: Vec<u8> = packets.concat();
    let lens: Vec<c_int> = packets.iter().map(|p| p.len() as c_int).collect();
    let mut cat_rets = vec![0 as c_int; packets.len()];
    let mut nb_frames = 0;
    let [ids, frames, offs, elens] = split(exts);
    // SAFETY: packets concatenated in buf with lengths in lens (kept alive for the call);
    // extension payloads lie within ext_buf; out valid for out.len() bytes.
    let ret = unsafe {
        oracle_rp_run(
            buf.as_ptr(),
            lens.as_ptr(),
            packets.len() as c_int,
            c_int::from(cat_sd),
            cat_rets.as_mut_ptr(),
            &mut nb_frames,
            begin,
            end,
            out.as_mut_ptr(),
            out.len() as c_int,
            c_int::from(sd),
            c_int::from(pad),
            ids.as_ptr(),
            frames.as_ptr(),
            offs.as_ptr(),
            elens.as_ptr(),
            exts.len() as c_int,
            ext_buf.as_ptr(),
        )
    };
    RpRun {
        cat_rets,
        nb_frames,
        ret,
    }
}

/// `opus_packet_pad_impl(data, len, data.len(), pad, exts)`.
pub fn pad_impl(data: &mut [u8], len: i32, pad: bool, exts: &[Ext], ext_buf: &[u8]) -> i32 {
    assert!(exts.len() <= 512 && len as usize <= data.len());
    let [ids, frames, offs, elens] = split(exts);
    // SAFETY: data valid for data.len() >= len bytes; payloads lie within ext_buf.
    unsafe {
        oracle_pad_impl(
            data.as_mut_ptr(),
            len,
            data.len() as c_int,
            c_int::from(pad),
            ids.as_ptr(),
            frames.as_ptr(),
            offs.as_ptr(),
            elens.as_ptr(),
            exts.len() as c_int,
            ext_buf.as_ptr(),
        )
    }
}

// ------------------------------------------------------------------------------- mapping_matrix.c

/// `mapping_matrix_get_size`.
pub fn mm_get_size(rows: i32, cols: i32) -> i32 {
    // SAFETY: pure function.
    unsafe { oracle_mm_get_size(rows, cols) }
}

/// A static projection matrix: (rows, cols, gain, data).
pub fn mm_static(idx: i32) -> (i32, i32, i32, Vec<i16>) {
    let (mut r, mut c, mut g) = (0, 0, 0);
    let mut data = vec![0i16; 38 * 38];
    // SAFETY: data holds the largest matrix (38x38).
    let n = unsafe { oracle_mm_static(idx, &mut r, &mut c, &mut g, data.as_mut_ptr(), 38 * 38) };
    assert!(n >= 0);
    data.truncate(n as usize);
    (r, c, g, data)
}

/// Matrix description for the multiply shims.
#[derive(Debug, Clone, Copy)]
pub struct Mm<'a> {
    /// Rows.
    pub rows: i32,
    /// Columns.
    pub cols: i32,
    /// Column-major cells (`rows*cols`).
    pub data: &'a [i16],
}

#[allow(clippy::too_many_arguments, reason = "flat FFI shim arguments")]
fn mm_call<I, O>(
    kind: i32,
    m: Mm<'_>,
    input: &[I],
    in_rows: usize,
    out: &mut [O],
    row: usize,
    out_rows: usize,
    frame_size: usize,
) {
    assert!(m.data.len() >= (m.rows * m.cols) as usize);
    // SAFETY: the caller-facing wrappers size input/output for the index ranges the C code
    // accesses (checked by the asserts in each wrapper).
    unsafe {
        oracle_mm_multiply(
            kind,
            m.rows,
            m.cols,
            m.data.as_ptr(),
            input.as_ptr().cast(),
            in_rows as c_int,
            out.as_mut_ptr().cast(),
            row as c_int,
            out_rows as c_int,
            frame_size as c_int,
        )
    }
}

/// `mapping_matrix_multiply_channel_in_float`.
pub fn mm_in_float(
    m: Mm<'_>,
    input: &[f32],
    input_rows: usize,
    output: &mut [f32],
    output_row: usize,
    output_rows: usize,
    frame_size: usize,
) {
    assert!(frame_size == 0 || input.len() >= input_rows * frame_size);
    assert!(frame_size == 0 || output.len() > output_rows * (frame_size - 1));
    mm_call(
        0,
        m,
        input,
        input_rows,
        output,
        output_row,
        output_rows,
        frame_size,
    );
}

/// `mapping_matrix_multiply_channel_out_float`.
pub fn mm_out_float(
    m: Mm<'_>,
    input: &[f32],
    input_row: usize,
    input_rows: usize,
    output: &mut [f32],
    output_rows: usize,
    frame_size: usize,
) {
    assert!(frame_size == 0 || input.len() > input_rows * (frame_size - 1));
    assert!(output.len() >= output_rows * frame_size);
    mm_call(
        1,
        m,
        input,
        input_rows,
        output,
        input_row,
        output_rows,
        frame_size,
    );
}

/// `mapping_matrix_multiply_channel_in_short`.
pub fn mm_in_short(
    m: Mm<'_>,
    input: &[i16],
    input_rows: usize,
    output: &mut [f32],
    output_row: usize,
    output_rows: usize,
    frame_size: usize,
) {
    assert!(frame_size == 0 || input.len() >= input_rows * frame_size);
    assert!(frame_size == 0 || output.len() > output_rows * (frame_size - 1));
    mm_call(
        2,
        m,
        input,
        input_rows,
        output,
        output_row,
        output_rows,
        frame_size,
    );
}

/// `mapping_matrix_multiply_channel_out_short`.
pub fn mm_out_short(
    m: Mm<'_>,
    input: &[f32],
    input_row: usize,
    input_rows: usize,
    output: &mut [i16],
    output_rows: usize,
    frame_size: usize,
) {
    assert!(frame_size == 0 || input.len() > input_rows * (frame_size - 1));
    assert!(output.len() >= output_rows * frame_size);
    mm_call(
        3,
        m,
        input,
        input_rows,
        output,
        input_row,
        output_rows,
        frame_size,
    );
}

/// `mapping_matrix_multiply_channel_in_int24`.
pub fn mm_in_int24(
    m: Mm<'_>,
    input: &[i32],
    input_rows: usize,
    output: &mut [f32],
    output_row: usize,
    output_rows: usize,
    frame_size: usize,
) {
    assert!(frame_size == 0 || input.len() >= input_rows * frame_size);
    assert!(frame_size == 0 || output.len() > output_rows * (frame_size - 1));
    mm_call(
        4,
        m,
        input,
        input_rows,
        output,
        output_row,
        output_rows,
        frame_size,
    );
}

/// `mapping_matrix_multiply_channel_out_int24`.
pub fn mm_out_int24(
    m: Mm<'_>,
    input: &[f32],
    input_row: usize,
    input_rows: usize,
    output: &mut [i32],
    output_rows: usize,
    frame_size: usize,
) {
    assert!(frame_size == 0 || input.len() > input_rows * (frame_size - 1));
    assert!(output.len() >= output_rows * frame_size);
    mm_call(
        5,
        m,
        input,
        input_rows,
        output,
        input_row,
        output_rows,
        frame_size,
    );
}

// ------------------------------------------------------------------------------- mlp.c

/// `analysis_compute_dense` on a custom layer.
pub fn mlp_dense(
    bias: &[i8],
    weights: &[i8],
    nb_inputs: usize,
    nb_neurons: usize,
    sigmoid: bool,
    output: &mut [f32],
    input: &[f32],
) {
    assert!(bias.len() >= nb_neurons && weights.len() >= nb_inputs * nb_neurons);
    assert!(output.len() >= nb_neurons && input.len() >= nb_inputs);
    // SAFETY: sizes asserted above.
    unsafe {
        oracle_mlp_dense(
            bias.as_ptr(),
            weights.as_ptr(),
            nb_inputs as c_int,
            nb_neurons as c_int,
            c_int::from(sigmoid),
            output.as_mut_ptr(),
            input.as_ptr(),
        )
    }
}

/// `analysis_compute_gru` on a custom layer (`nb_neurons <= 32`).
pub fn mlp_gru(
    bias: &[i8],
    weights: &[i8],
    recur: &[i8],
    nb_inputs: usize,
    nb_neurons: usize,
    state: &mut [f32],
    input: &[f32],
) {
    assert!(nb_neurons <= 32);
    assert!(bias.len() >= 3 * nb_neurons && weights.len() >= 3 * nb_inputs * nb_neurons);
    assert!(recur.len() >= 3 * nb_neurons * nb_neurons);
    assert!(state.len() >= nb_neurons && input.len() >= nb_inputs);
    // SAFETY: sizes asserted above.
    unsafe {
        oracle_mlp_gru(
            bias.as_ptr(),
            weights.as_ptr(),
            recur.as_ptr(),
            nb_inputs as c_int,
            nb_neurons as c_int,
            state.as_mut_ptr(),
            input.as_ptr(),
        )
    }
}

/// Built-in analysis layers: 0 = `layer0` (25 -> 32), 1 = `layer1` GRU (32 in, 24 state),
/// 2 = `layer2` (24 -> 2).
pub fn mlp_builtin(which: i32, out_or_state: &mut [f32], input: &[f32]) {
    let (nin, nout) = match which {
        0 => (25, 32),
        1 => (32, 24),
        _ => (24, 2),
    };
    assert!(out_or_state.len() >= nout && input.len() >= nin);
    // SAFETY: sizes asserted above.
    unsafe { oracle_mlp_builtin(which, out_or_state.as_mut_ptr(), input.as_ptr()) }
}

/// `tansig_approx` (static in mlp.c).
pub fn mlp_tansig(x: f32) -> f32 {
    // SAFETY: pure function.
    unsafe { oracle_mlp_tansig(x) }
}

/// `sigmoid_approx` (static in mlp.c).
pub fn mlp_sigmoid(x: f32) -> f32 {
    // SAFETY: pure function.
    unsafe { oracle_mlp_sigmoid(x) }
}

// ------------------------------------------------------------------------------- opus_multistream.c

/// `validate_layout`.
pub fn ms_validate_layout(
    nb_channels: i32,
    nb_streams: i32,
    nb_coupled: i32,
    mapping: &[u8; 256],
) -> bool {
    assert!(nb_channels <= 256);
    // SAFETY: mapping has 256 entries.
    unsafe { oracle_ms_validate_layout(nb_channels, nb_streams, nb_coupled, mapping.as_ptr()) != 0 }
}

/// `get_left_channel` (which 0), `get_right_channel` (1), `get_mono_channel` (2).
pub fn ms_get_channel(
    which: i32,
    nb_channels: i32,
    nb_streams: i32,
    nb_coupled: i32,
    mapping: &[u8; 256],
    stream_id: i32,
    prev: i32,
) -> i32 {
    assert!(nb_channels <= 256);
    // SAFETY: mapping has 256 entries.
    unsafe {
        oracle_ms_get_channel(
            which,
            nb_channels,
            nb_streams,
            nb_coupled,
            mapping.as_ptr(),
            stream_id,
            prev,
        )
    }
}
