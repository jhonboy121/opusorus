//! Oracle bindings for the packet loss generator of libopus `--enable-lossgen`
//! (dnn/lossgen.c + dnn/lossgen_data.c, `csrc/dnn_lossgen.c`).
//!
//! Only available with the `lossgen` feature. The generator draws from the C library's global
//! `rand()`; [`with_rand`] serializes the tests that use it.
#![cfg(feature = "lossgen")]

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::{Mutex, MutexGuard};

unsafe extern "C" {
    fn oracle_lg_array_count() -> c_int;
    fn oracle_lg_array_info(i: c_int, name: *mut c_char, ty: *mut c_int, size: *mut c_int);
    fn oracle_lg_array_data(i: c_int, out: *mut c_void);
    fn oracle_lg_new() -> *mut c_void;
    fn oracle_lg_new_from_blob(data: *const c_void, len: c_int, ret: *mut c_int) -> *mut c_void;
    fn oracle_lg_free(st: *mut c_void);
    fn oracle_lg_sample(st: *mut c_void, percent_loss: f32) -> c_int;
    fn oracle_lg_state(
        st: *const c_void,
        gru1: *mut f32,
        gru2: *mut f32,
        last_loss: *mut c_int,
        used: *mut c_int,
    );
    fn oracle_lg_gru1_size() -> c_int;
    fn oracle_lg_gru2_size() -> c_int;
    fn oracle_lg_srand(seed: c_uint);
    fn oracle_lg_rand() -> c_int;
}

/// One entry of the compiled-in `lossgen_arrays` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LossgenArray {
    pub name: String,
    pub type_: i32,
    pub size: i32,
    pub data: Vec<u8>,
}

/// The oracle's compiled-in `lossgen_arrays` (without the `NULL` terminator).
#[must_use]
pub fn arrays() -> Vec<LossgenArray> {
    // SAFETY: no arguments.
    let n = unsafe { oracle_lg_array_count() };
    (0..n)
        .map(|i| {
            let mut name = [0u8; 64];
            let mut ty = 0;
            let mut size = 0;
            // SAFETY: `i` < count; `name` has the 64 bytes the shim writes at most.
            unsafe { oracle_lg_array_info(i, name.as_mut_ptr().cast(), &mut ty, &mut size) };
            let mut data = vec![0u8; usize::try_from(size).expect("non-negative size")];
            // SAFETY: `data` has the array's `size` bytes.
            unsafe { oracle_lg_array_data(i, data.as_mut_ptr().cast()) };
            let len = name.iter().position(|&b| b == 0).expect("NUL-terminated");
            LossgenArray {
                name: String::from_utf8(name[..len].to_vec()).expect("ASCII name"),
                type_: ty,
                size,
                data,
            }
        })
        .collect()
}

/// Snapshot of a `LossGenState`'s recurrent state.
#[derive(Debug, Clone, PartialEq)]
pub struct LossGenSnapshot {
    pub gru1: Vec<f32>,
    pub gru2: Vec<f32>,
    pub last_loss: i32,
    pub used: i32,
}

/// A C `LossGenState` on the C heap (with the blob its layers point into, if loaded from one).
#[derive(Debug)]
pub struct LossGen(
    *mut c_void,
    #[expect(dead_code, reason = "keeps the blob the C layers point into alive")] Option<Box<[u8]>>,
);

impl LossGen {
    /// `lossgen_init` (the compiled-in model).
    #[must_use]
    pub fn new() -> Self {
        // SAFETY: no arguments; the result is checked.
        let p = unsafe { oracle_lg_new() };
        assert!(!p.is_null(), "out of memory");
        Self(p, None)
    }

    /// `lossgen_load_model(st, blob)` on a zeroed state: the state and the C return value.
    #[must_use]
    pub fn from_blob(blob: &[u8]) -> (Self, i32) {
        let mut ret = 0;
        // The bound layers point into the blob: keep a copy alive as long as the state.
        let blob: Box<[u8]> = blob.into();
        // SAFETY: `blob` is valid for `blob.len()` bytes and outlives the state (owned by the
        // returned value); C only reads it.
        let p = unsafe {
            oracle_lg_new_from_blob(
                blob.as_ptr().cast(),
                c_int::try_from(blob.len()).expect("blob size fits in int"),
                &mut ret,
            )
        };
        assert!(!p.is_null(), "out of memory");
        (Self(p, Some(blob)), ret)
    }

    /// `sample_loss(st, percent_loss)` (draws from C `rand()`).
    pub fn sample(&mut self, percent_loss: f32) -> i32 {
        // SAFETY: `self.0` is a live state.
        unsafe { oracle_lg_sample(self.0, percent_loss) }
    }

    /// The GRU states, `last_loss` and `used`.
    #[must_use]
    pub fn snapshot(&self) -> LossGenSnapshot {
        // SAFETY: no arguments.
        let (n1, n2) = unsafe { (oracle_lg_gru1_size(), oracle_lg_gru2_size()) };
        let mut gru1 = vec![0f32; usize::try_from(n1).expect("size")];
        let mut gru2 = vec![0f32; usize::try_from(n2).expect("size")];
        let mut last_loss = 0;
        let mut used = 0;
        // SAFETY: the buffers have the state sizes reported by the shim.
        unsafe {
            oracle_lg_state(
                self.0,
                gru1.as_mut_ptr(),
                gru2.as_mut_ptr(),
                &mut last_loss,
                &mut used,
            );
        }
        LossGenSnapshot {
            gru1,
            gru2,
            last_loss,
            used,
        }
    }
}

impl Default for LossGen {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for LossGen {
    fn drop(&mut self) {
        // SAFETY: allocated by the shim, freed once.
        unsafe { oracle_lg_free(self.0) };
    }
}

static RAND_LOCK: Mutex<()> = Mutex::new(());

/// Exclusive use of the C library's global `rand()` state for the duration of the guard.
pub fn with_rand() -> MutexGuard<'static, ()> {
    match RAND_LOCK.lock() {
        Ok(g) => g,
        // A panicking test only poisons the lock; the rand state is reseeded by every user.
        Err(p) => p.into_inner(),
    }
}

/// C `srand(seed)`.
pub fn srand(seed: u32) {
    // SAFETY: plain libc call.
    unsafe { oracle_lg_srand(seed) }
}

/// C `rand()`.
#[must_use]
pub fn rand() -> i32 {
    // SAFETY: plain libc call.
    unsafe { oracle_lg_rand() }
}
