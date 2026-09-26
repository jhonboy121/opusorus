//! Opaque C state blocks and the registry that maps them to Rust objects.
//!
//! # Design
//!
//! libopus states (`OpusEncoder`, `OpusDecoder`, ...) are flat C structs without pointers: C
//! callers allocate `*_get_size()` bytes and `*_init` them in place, and may copy a state with
//! `memcpy` (upstream's own tests do: they move states to new memory, back decoders up before a
//! trial decode, and clone a decoder to compare FEC decoding). The Rust objects own heap memory,
//! so they cannot live *inside* the C block: a byte copy would alias their buffers and a double
//! free would follow.
//!
//! Instead every C block holds a small [`Header`] (32 bytes at offset 8: magic, kind, object
//! id, the block's own address, and a generation counter), and the objects live in a process-wide
//! registry of boxed entries keyed by id, with a second map recording which block address owns
//! which object. Resolving a block checks that
//!
//! * the header is valid for the expected kind (else `OPUS_INVALID_STATE`), and
//! * the block is the registered owner of the id in its header.
//!
//! If it is not (the bytes were copied from another block), the copy is *materialized* on its
//! first mutable use: when the source object is still owned by another block it is cloned
//! (C copy semantics: two independent states), and when its owner block was overwritten or
//! freed without `*_destroy` (a C "move": `memcpy` to new memory, then `memset` + destroy of the
//! old block) the orphaned object is adopted. Read-only calls on an unmaterialized copy read the
//! source object directly. Whatever object a block owned before being overwritten is dropped.
//!
//! The generation counter is bumped on every mutating call so the block bytes change whenever
//! the state does (upstream `test_opus_api` checks `OPUS_RESET_STATE` via `memcmp`).
//!
//! Limitations (documented in the crate docs):
//! * a copy is materialized when first used, so it sees the source state as of that moment,
//!   not as of the `memcpy` (identical in the usual copy-then-use pattern);
//! * a copy whose source was destroyed with `*_destroy` before the copy was first used, or a
//!   block restored from a byte backup older than its current object, cannot be recovered
//!   (`OPUS_INVALID_STATE`, respectively the current object state is used).
//!
//! Per-stream encoders/decoders of multistream and projection states
//! (`OPUS_MULTISTREAM_GET_ENCODER_STATE`) are handed out as small header-only blocks owned by
//! the parent entry, with the parent id and stream index in the header.

use core::ffi::c_void;
use core::ptr::{self, NonNull};
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use opus::{Decoder, Encoder, MsDecoder, MsEncoder, ProjectionDecoder, ProjectionEncoder};

use crate::packet::RepacketizerState;
use crate::util::{CResult, OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_INVALID_STATE};

unsafe extern "C" {
    fn calloc(nmemb: usize, size: usize) -> *mut c_void;
    fn free(ptr: *mut c_void);
}

/// Allocates `size` zeroed bytes with the C allocator (so `*_destroy` can `free` blocks the
/// caller allocated with `malloc`, as libopus does). Returns NULL on failure.
pub(crate) fn c_alloc(size: usize) -> *mut c_void {
    // SAFETY: `calloc` has no preconditions; a NULL result is handled by the callers.
    unsafe { calloc(1, size.max(HEADER_SIZE)) }
}

/// Frees memory from [`c_alloc`] or the caller's `malloc` (C `opus_free`).
///
/// # Safety
/// `ptr` is NULL or a live C-heap allocation that is not used afterwards.
pub(crate) unsafe fn c_free(ptr: *mut c_void) {
    // SAFETY: forwarded caller contract; `free(NULL)` is a no-op.
    unsafe { free(ptr) }
}

/// Header magic ("OPUS" in ASCII, big-endian reading).
const MAGIC: u32 = 0x4F50_5553;

/// Kind of state a block holds (C type of the handle).
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    Encoder = 1,
    Decoder = 2,
    MsEncoder = 3,
    MsDecoder = 4,
    ProjectionEncoder = 5,
    ProjectionDecoder = 6,
    Repacketizer = 7,
    CustomEncoder = 8,
    CustomDecoder = 9,
    DredDecoder = 10,
    /// A stream encoder inside a multistream/projection encoder (`aux` = stream index).
    StreamEncoder = 11,
    /// A stream decoder inside a multistream/projection decoder (`aux` = stream index).
    StreamDecoder = 12,
}

impl Kind {
    const fn from_raw(v: u32) -> Option<Self> {
        Some(match v {
            1 => Self::Encoder,
            2 => Self::Decoder,
            3 => Self::MsEncoder,
            4 => Self::MsDecoder,
            5 => Self::ProjectionEncoder,
            6 => Self::ProjectionDecoder,
            7 => Self::Repacketizer,
            8 => Self::CustomEncoder,
            9 => Self::CustomDecoder,
            10 => Self::DredDecoder,
            11 => Self::StreamEncoder,
            12 => Self::StreamDecoder,
            _ => return None,
        })
    }

    /// The `kind` argument of the non-variadic ctl entry points (see `csrc/ctl.c`).
    pub(crate) const fn from_ctl(v: core::ffi::c_int) -> Option<Self> {
        if v < 0 {
            return None;
        }
        Self::from_raw(v as u32)
    }
}

/// The bytes at the start of every C state block. Accessed unaligned: C only guarantees the
/// alignment of its own struct, and stack-allocated `OpusRepacketizer`s exist.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Header {
    magic: u32,
    kind: u32,
    /// Registry id of the object (for stream blocks: of the parent).
    id: u64,
    /// Address of the owning block (stream blocks: the stream index).
    aux: u64,
    /// Bumped on every mutating call.
    generation: u64,
}

/// Offset of the [`Header`] in a block. The first bytes are left to mirror a few C struct
/// fields that code outside the public API reads directly (`OpusRepacketizer.toc` and
/// `.nb_frames`, read by upstream's internal `test_opus_extensions`); see [`write_prefix`].
const HEADER_OFFSET: usize = 8;

/// Minimum size of a C state block.
pub(crate) const HEADER_SIZE: usize = HEADER_OFFSET + size_of::<Header>();

/// Size reported by the `*_get_size` functions for an object with the given Rust footprint:
/// the header plus the footprint (the block itself only needs the header, but callers use the
/// size as a memory estimate and upstream tests check its range).
pub(crate) const fn block_size(footprint: usize) -> usize {
    if footprint == 0 {
        0
    } else {
        HEADER_SIZE + footprint
    }
}

/// A Rust state owned by the registry.
#[derive(Clone, Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "always stored boxed in an `Entry`; boxing each variant again buys nothing"
)]
pub(crate) enum Object {
    Encoder(Encoder),
    Decoder(Decoder),
    MsEncoder(MsEncoder),
    MsDecoder(MsDecoder),
    ProjectionEncoder(ProjectionEncoder),
    ProjectionDecoder(ProjectionDecoder),
    Repacketizer(RepacketizerState),
    #[cfg(feature = "custom-modes")]
    CustomEncoder(crate::custom::CustomEncoder),
    #[cfg(feature = "custom-modes")]
    CustomDecoder(crate::custom::CustomDecoder),
    #[cfg(feature = "dred")]
    DredDecoder(opus::dred::DredDecoder),
}

impl Object {
    const fn kind(&self) -> Kind {
        match self {
            Self::Encoder(_) => Kind::Encoder,
            Self::Decoder(_) => Kind::Decoder,
            Self::MsEncoder(_) => Kind::MsEncoder,
            Self::MsDecoder(_) => Kind::MsDecoder,
            Self::ProjectionEncoder(_) => Kind::ProjectionEncoder,
            Self::ProjectionDecoder(_) => Kind::ProjectionDecoder,
            Self::Repacketizer(_) => Kind::Repacketizer,
            #[cfg(feature = "custom-modes")]
            Self::CustomEncoder(_) => Kind::CustomEncoder,
            #[cfg(feature = "custom-modes")]
            Self::CustomDecoder(_) => Kind::CustomDecoder,
            #[cfg(feature = "dred")]
            Self::DredDecoder(_) => Kind::DredDecoder,
        }
    }
}

/// A header-only block handed out for one stream of a multistream/projection state.
#[derive(Debug)]
struct StreamBlock {
    stream_id: i32,
    kind: Kind,
    /// C-heap address of the block (kept as an integer so `Entry` stays `Send`).
    addr: usize,
}

/// A registry entry.
#[derive(Debug)]
pub(crate) struct Entry {
    id: u64,
    pub(crate) object: Object,
    /// Address of the block owning this object; `None` for an orphan (its block was
    /// overwritten or freed without destroy) that a byte copy may adopt.
    owner: Option<usize>,
    streams: Vec<StreamBlock>,
}

impl Drop for Entry {
    fn drop(&mut self) {
        for s in self.streams.drain(..) {
            // SAFETY: stream blocks are allocated with `c_alloc` by `stream_block` and only
            // freed here, once, when their parent entry goes away.
            unsafe { c_free(s.addr as *mut c_void) };
        }
    }
}

impl Entry {
    /// Returns (allocating on first use) the C handle of stream `stream_id` of this entry.
    pub(crate) fn stream_block(&mut self, kind: Kind, stream_id: i32) -> CResult<*mut c_void> {
        if let Some(s) = self
            .streams
            .iter()
            .find(|s| s.stream_id == stream_id && s.kind == kind)
        {
            return Ok(s.addr as *mut c_void);
        }
        let block = c_alloc(HEADER_SIZE);
        if block.is_null() {
            return Err(crate::util::OPUS_ALLOC_FAIL);
        }
        let header = Header {
            magic: MAGIC,
            kind: kind as u32,
            id: self.id,
            aux: stream_id as u64,
            generation: 0,
        };
        // SAFETY: `block` is a fresh allocation of at least `HEADER_SIZE` bytes.
        unsafe { write_header(block, header) };
        self.streams.push(StreamBlock {
            stream_id,
            kind,
            addr: block as usize,
        });
        Ok(block)
    }
}

/// Owning pointer to a registry entry (a leaked `Box<Entry>`, freed on drop).
///
/// Raw rather than `Box`: [`resolve`] hands out pointers to entries that are used after the
/// registry lock is released, and those must stay valid while the map moves its values around
/// (moving a `Box` asserts uniqueness, a raw pointer does not).
#[derive(Debug)]
struct EntryPtr(NonNull<Entry>);

// SAFETY: `EntryPtr` owns its `Entry` exactly like a `Box<Entry>` would, and `Entry` is `Send`
// (checked by `entry_is_send`). Access from several threads is serialized by the registry lock,
// or by the libopus contract that one state is used by one thread at a time.
unsafe impl Send for EntryPtr {}

/// Compile-time check backing the `Send` impl above.
const fn entry_is_send() {
    const fn check<T: Send>() {}
    check::<Entry>();
}
const _: () = entry_is_send();

impl EntryPtr {
    fn new(entry: Entry) -> Self {
        Self(NonNull::from(Box::leak(Box::new(entry))))
    }

    const fn as_ptr(&self) -> *mut Entry {
        self.0.as_ptr()
    }

    /// The entry, for use while the registry lock is held.
    ///
    /// # Safety
    /// No reference obtained through [`Resolved::entry`] is in use for this entry on another
    /// thread (libopus contract: a state, and the state it was copied from, are not used
    /// concurrently with the copy's first use).
    const unsafe fn get_mut(&mut self) -> &mut Entry {
        // SAFETY: allocated by `new`, freed only by `drop`; caller contract for exclusivity.
        unsafe { self.0.as_mut() }
    }
}

impl Drop for EntryPtr {
    fn drop(&mut self) {
        // SAFETY: allocated with `Box::leak` in `new`; each `EntryPtr` is dropped once.
        drop(unsafe { Box::from_raw(self.0.as_ptr()) });
    }
}

/// Registry of all live objects.
struct Registry {
    next_id: u64,
    objects: BTreeMap<u64, EntryPtr>,
    /// Block address -> id of the object that block owns.
    blocks: BTreeMap<usize, u64>,
}

impl Registry {
    const fn new() -> Self {
        Self {
            next_id: 1,
            objects: BTreeMap::new(),
            blocks: BTreeMap::new(),
        }
    }

    const fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry::new());

/// Locks the registry. A poisoned lock (a panic while it was held, which only an allocation
/// failure can cause) is reported as `OPUS_INTERNAL_ERROR`.
fn lock() -> CResult<MutexGuard<'static, Registry>> {
    REGISTRY.lock().map_err(|_| OPUS_INTERNAL_ERROR)
}

/// Reads the header of a block.
///
/// # Safety
/// `block` is valid for reads of `HEADER_SIZE` bytes.
const unsafe fn read_header(block: *const c_void) -> Header {
    // SAFETY: forwarded caller contract; unaligned read inside the block.
    unsafe { ptr::read_unaligned(block.cast::<u8>().add(HEADER_OFFSET).cast::<Header>()) }
}

/// Writes the header of a block.
///
/// # Safety
/// `block` is valid for writes of `HEADER_SIZE` bytes.
const unsafe fn write_header(block: *mut c_void, h: Header) {
    // SAFETY: forwarded caller contract; unaligned write inside the block.
    unsafe { ptr::write_unaligned(block.cast::<u8>().add(HEADER_OFFSET).cast::<Header>(), h) }
}

/// Writes the C-visible prefix of a block (the bytes before the header), e.g. the mirrored
/// `toc` / `nb_frames` fields of an `OpusRepacketizer`.
///
/// # Safety
/// `block` is valid for writes of `HEADER_SIZE` bytes.
pub(crate) const unsafe fn write_prefix(block: *mut c_void, prefix: [u8; HEADER_OFFSET]) {
    // SAFETY: forwarded caller contract.
    unsafe { ptr::write_unaligned(block.cast::<[u8; HEADER_OFFSET]>(), prefix) }
}

/// Makes `object` the state of the C block at `block` (`*_init`, and `*_create` after
/// allocating). Whatever object the block owned before is dropped.
///
/// # Safety
/// `block` is valid for writes of `HEADER_SIZE` bytes.
pub(crate) unsafe fn install(block: *mut c_void, object: Object) -> CResult<()> {
    if block.is_null() {
        return Err(OPUS_BAD_ARG);
    }
    let addr = block as usize;
    let kind = object.kind();
    let mut reg = lock()?;
    let id = reg.alloc_id();
    let stale = match reg.blocks.insert(addr, id) {
        Some(old) => reg.objects.remove(&old),
        None => None,
    };
    reg.objects.insert(
        id,
        EntryPtr::new(Entry {
            id,
            object,
            owner: Some(addr),
            streams: Vec::new(),
        }),
    );
    drop(reg);
    drop(stale);
    // SAFETY: caller contract.
    unsafe {
        write_header(
            block,
            Header {
                magic: MAGIC,
                kind: kind as u32,
                id,
                aux: addr as u64,
                generation: 0,
            },
        );
    }
    Ok(())
}

/// Writes a registry-less header (states without Rust data: `OpusDREDDecoder` without the
/// `dred` feature).
///
/// # Safety
/// `block` is valid for writes of `HEADER_SIZE` bytes.
#[cfg(not(feature = "dred"))]
pub(crate) unsafe fn write_plain_header(block: *mut c_void, kind: Kind) {
    // SAFETY: caller contract.
    unsafe {
        write_header(
            block,
            Header {
                magic: MAGIC,
                kind: kind as u32,
                id: 0,
                aux: block as u64,
                generation: 0,
            },
        );
    }
}

/// How a resolved state is going to be used.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Access {
    /// Read-only (`const` C API, GET requests): byte copies are not materialized.
    Read,
    /// Mutating: byte copies are materialized and the generation is bumped.
    Write,
}

/// A resolved block: the entry and, for stream blocks, the stream index.
#[derive(Debug)]
pub(crate) struct Resolved {
    entry: *mut Entry,
    pub(crate) stream: Option<i32>,
}

impl Resolved {
    /// The entry.
    ///
    /// # Safety
    /// The entry must still be alive (no `*_destroy` / `*_init` of its owner block since it was
    /// resolved) and not accessed by another thread: the libopus contract that a state is used
    /// by one thread at a time, and not destroyed while in use.
    pub(crate) const unsafe fn entry<'a>(&self) -> &'a mut Entry {
        // SAFETY: entries have a stable heap address (`EntryPtr`) and are only freed by
        // destroy/init/overwrite of their owner block; see the caller contract.
        unsafe { &mut *self.entry }
    }
}

/// Resolves a C state pointer of one of the `kinds`.
///
/// # Safety
/// `block` is NULL or valid for reads (and, with `Access::Write`, writes) of `HEADER_SIZE`
/// bytes.
pub(crate) unsafe fn resolve(
    block: *const c_void,
    kinds: &[Kind],
    access: Access,
) -> CResult<Resolved> {
    if block.is_null() {
        return Err(OPUS_BAD_ARG);
    }
    // SAFETY: caller contract.
    let mut h = unsafe { read_header(block) };
    let kind = match Kind::from_raw(h.kind) {
        Some(k) if h.magic == MAGIC && kinds.contains(&k) => k,
        _ => return Err(OPUS_INVALID_STATE),
    };
    let mut reg = lock()?;
    if matches!(kind, Kind::StreamEncoder | Kind::StreamDecoder) {
        let Some(parent) = reg.objects.get(&h.id) else {
            return Err(OPUS_INVALID_STATE);
        };
        let stream = i32::try_from(h.aux).map_err(|_| OPUS_INVALID_STATE)?;
        return Ok(Resolved {
            entry: parent.as_ptr(),
            stream: Some(stream),
        });
    }
    let addr = block as usize;
    if reg.blocks.get(&addr) == Some(&h.id) {
        let Some(entry) = reg.objects.get(&h.id) else {
            return Err(OPUS_INTERNAL_ERROR);
        };
        let entry = entry.as_ptr();
        drop(reg);
        if access == Access::Write {
            h.generation = h.generation.wrapping_add(1);
            // SAFETY: caller contract (writable for `Access::Write`).
            unsafe { write_header(block.cast_mut(), h) };
        }
        return Ok(Resolved {
            entry,
            stream: None,
        });
    }
    // The block holds a byte copy of another block's state.
    if access == Access::Read {
        let Some(src) = reg.objects.get(&h.id) else {
            return Err(OPUS_INVALID_STATE);
        };
        return Ok(Resolved {
            entry: src.as_ptr(),
            stream: None,
        });
    }
    let Some(src) = reg.objects.get_mut(&h.id) else {
        return Err(OPUS_INVALID_STATE);
    };
    // SAFETY: under the registry lock; caller contract (the source state is not in use).
    let src = unsafe { src.get_mut() };
    let id = if src.owner.is_none() {
        // Orphan (its block was overwritten or freed without destroy): this block adopts it.
        src.owner = Some(addr);
        h.id
    } else {
        let object = src.object.clone();
        let id = reg.alloc_id();
        reg.objects.insert(
            id,
            EntryPtr::new(Entry {
                id,
                object,
                owner: Some(addr),
                streams: Vec::new(),
            }),
        );
        id
    };
    // Whatever this block owned before was overwritten by the copy: drop it.
    let stale = match reg.blocks.insert(addr, id) {
        Some(old) if old != id => reg.objects.remove(&old),
        _ => None,
    };
    let Some(entry) = reg.objects.get(&id) else {
        return Err(OPUS_INTERNAL_ERROR);
    };
    let entry = entry.as_ptr();
    drop(reg);
    drop(stale);
    h.id = id;
    h.aux = addr as u64;
    h.generation = h.generation.wrapping_add(1);
    // SAFETY: caller contract (writable for `Access::Write`).
    unsafe { write_header(block.cast_mut(), h) };
    Ok(Resolved {
        entry,
        stream: None,
    })
}

/// `*_create`: builds the object with `make` (which also returns the Rust footprint for the
/// block size), allocates a zeroed C block of [`block_size`] bytes, installs the object and
/// reports the result through the optional `error` out-pointer. Returns NULL on failure.
///
/// # Safety
/// `error` is NULL or valid for writing one `int`.
pub(crate) unsafe fn create<T>(
    error: *mut core::ffi::c_int,
    make: impl FnOnce() -> CResult<(usize, Object)>,
) -> *mut T {
    let r = crate::util::guard_any(Err(OPUS_INTERNAL_ERROR), || -> CResult<*mut T> {
        let (footprint, object) = make()?;
        let block = c_alloc(block_size(footprint));
        if block.is_null() {
            return Err(crate::util::OPUS_ALLOC_FAIL);
        }
        // SAFETY: `block` is a fresh allocation of at least `HEADER_SIZE` bytes.
        if let Err(e) = unsafe { install(block, object) } {
            // SAFETY: allocated above, not registered.
            unsafe { c_free(block) };
            return Err(e);
        }
        Ok(block.cast())
    });
    match r {
        Ok(p) => {
            // SAFETY: caller contract.
            unsafe { crate::util::set_error(error, crate::util::OPUS_OK) };
            p
        }
        Err(e) => {
            // SAFETY: caller contract.
            unsafe { crate::util::set_error(error, e) };
            ptr::null_mut()
        }
    }
}

/// `*_destroy`: drops the object owned by the block and frees the block with C `free`.
///
/// A block whose header was scrambled (e.g. `memset` after moving the state elsewhere with
/// `memcpy`) leaves its object as an orphan for the copy to adopt. Stream blocks (interior
/// pointers of a multistream state, which C must never free) are ignored.
///
/// # Safety
/// `block` is NULL or a C-heap block of at least `HEADER_SIZE` bytes that is not used
/// afterwards.
pub(crate) unsafe fn destroy(block: *mut c_void) {
    if block.is_null() {
        return;
    }
    // SAFETY: caller contract.
    let h = unsafe { read_header(block) };
    let kind = if h.magic == MAGIC {
        Kind::from_raw(h.kind)
    } else {
        None
    };
    if matches!(kind, Some(Kind::StreamEncoder | Kind::StreamDecoder)) {
        return;
    }
    let addr = block as usize;
    // Without the registry lock (poisoned) the memory is still freed; the object leaks.
    if let Ok(mut reg) = lock() {
        let dropped = match reg.blocks.remove(&addr) {
            Some(id) if kind.is_some() => {
                // Valid header: either this block's own object, or it was overwritten by a copy
                // of another state (then its former object is unreachable). Drop it either way.
                reg.objects.remove(&id)
            }
            Some(id) => {
                // Scrambled header: the state may have been moved to another block.
                if let Some(e) = reg.objects.get_mut(&id) {
                    // SAFETY: under the registry lock; the block is being destroyed, so its
                    // state is not in use.
                    unsafe { e.get_mut() }.owner = None;
                }
                None
            }
            None => None,
        };
        drop(reg);
        drop(dropped);
    }
    // SAFETY: caller contract.
    unsafe { c_free(block) };
}
