# Porting guide (read before porting any unit)

opusorus is a **faithful, function-by-function** port of libopus v1.6.1 (`vendor/libopus`). The
goal is output that is **bit-exact** with the C library compiled by `crates/opusorus-oracle`
(float build, `-ffp-contract=off`, no SIMD intrinsics, `ENABLE_HARDENING`).
SIMD in the port (`fearless_simd`, `celt/simd.rs`) is vertical only: every lane repeats the scalar
operation sequence, so it is bit-identical to the scalar code (PLAN D-031).

## 1. Layout and ownership

| What | Where |
|---|---|
| Port code | `crates/opusorus/src/**` — one Rust file per C file (or per group, see `docs/TRACKER.md`) |
| Oracle FFI (safe wrappers) | `crates/opusorus-oracle/src/<unit>.rs` |
| Oracle C shims | `crates/opusorus-oracle/csrc/<unit>.c` (auto-compiled; include any internal libopus header) |
| Differential tests | `crates/opusorus-conformance/tests/<unit>.rs` |

A **unit** owns exactly the files listed for it in `docs/TRACKER.md`. Never edit files owned by
another unit or shared files (`lib.rs`, `mod.rs`, `Cargo.toml`, foundation modules). If you need a
helper that belongs elsewhere and does not exist yet, write a private copy in your own file and
note it in your report so it can be deduplicated later. A Rust file may declare private
submodules in a same-named directory (e.g. `silk/float.rs` → `silk/float/*.rs`).

The foundation (already done, reuse it):
* `crate::math` — **all** libm calls (`math::cos(f64)`, `math::sqrtf(f32)`, `math::lrintf` ...).
* `crate::celt::arch` — type aliases (`OpusVal16`, `CeltSig`, ...) and C arithmetic macros as
  `const fn` (`mult16_32_q15`, `shr32`, `qconst16`, `max16`, `imin`, ...).
* `crate::celt::mathops` — `celt_sqrt`, `celt_log2`, `celt_exp2`, `celt_cos_norm`, `fast_atan2f`,
  `float2int`, `float2int16`, `isqrt32`, `frac_mul16`, `celt_maxabs16`, `PI` (f64) ...
* `crate::celt::entcode|entenc|entdec` — `EcEnc`, `EcDec`, `EcCoder` (for `ec_ctx*` + `encode`
  flag code), `ec_ilog`, `BITRES`, `celt_udiv`. `ec_tell(ec)` → `ec.tell()`.
* `crate::silk::macros` — every SILK fixed-point macro (`silk_smulwb`, `silk_rshift_round`,
  `silk_div32_varq`, `silk_sat16`, `silk_limit`, ...), `crate::silk::define`,
  `crate::silk::tuning_parameters`, `crate::silk::errors`.
* `crate::Error` / `crate::Result`, `crate::constants` (`Application`, `Bandwidth`, ...).

## 2. Translation rules

**Numerics (the #1 source of mismatches):**
* `float` → `f32`, `double` → `f64`. Reproduce C promotions exactly: in C, `x*0.5` with
  `float x` is a *double* multiply (`(x as f64 * 0.5) as f32` at the assignment), while
  `x*.5f` is float. Integer literals mixed with floats convert the int to float. Calls to
  `cos`, `exp`, `log`, `sqrt`, `pow`, `floor` take/return double.
* Keep operation order and association exactly (`a+b+c` is `(a+b)+c`). Never use `mul_add`,
  never reorder sums, never "simplify" `a*(1/b)` to `a/b`.
* Float literals: copy the C literal text verbatim (the crate allows `excessive_precision`).
* `(int)x` casts truncate toward zero: use `x as i32` (saturating in Rust, identical for in-range
  values). `float2int`/`lrintf` round half to even (`mathops::float2int`).
* `MIN`/`MAX`/`FMIN`/`FMAX`/`MIN16`/... have C ternary semantics — use the `arch` helpers, not
  `f32::min/max` (NaN handling differs).
* Integers: `opus_int32` → `i32`, `opus_int16` → `i16`, `opus_uint32` → `u32`, `int` → `i32`.
  Reproduce C integer promotion (i16*i16 is computed in i32). Use `wrapping_*` **only** where C
  relies on wraparound (unsigned arithmetic, `_ovflw` macros); elsewhere use plain ops so
  overflow panics in debug/test builds and exposes bugs.
* Do not "fix" anything that looks odd in C: bit-exactness wins. Port quirks faithfully and add
  a comment if it's surprising.

**Structure:**
* Keep C function names (snake_case already) with a doc comment `/// Port of celt/bands.c:name`.
* Pointers + lengths → slices. Pointer arithmetic → index offsets (`&x[off..]`). Aliasing
  in-place C calls (`f(x, x)`) → split borrows or a temporary copy (document it).
* Structs → Rust structs with the same field names. `OPUS_CLEAR`/`memset` resets → a
  `reset()`/`Default`. C's "copy struct to save state" → `#[derive(Clone)]`.
* VLAs/`ALLOC(...)` → fixed-size stack arrays sized by the C maximum when small (< ~8 KB),
  otherwise a scratch `Vec` owned by the state (allocate at init, not per call).
* `celt_assert` / `celt_assert2` → `celt_assert!`, `celt_sig_assert` → `celt_sig_assert!`,
  `silk_assert` → `silk_assert!` (crate macros in `src/assertions.rs`: `debug_assert!`s, hard
  `assert!`s with the feature `assertions` = `ENABLE_ASSERTIONS`); a `celt_assert(0)` that the
  port keeps reachable (C returns an error after it) → `assertion_failure!`. Rust-side
  invariants that C does not assert stay `debug_assert!`. Hardening checks that return errors in
  C must return errors in Rust.
* `#ifdef ENABLE_QEXT` → `#[cfg(feature = "qext")]` (port these blocks in the same pass).
  `#ifdef CUSTOM_MODES` → `#[cfg(feature = "custom-modes")]`. `#ifdef FIXED_POINT` →
  `#[cfg(feature = "fixed-point")]` when a fixed-point unit converts the file (see
  `docs/FIXED_POINT.md` for the typing rules and gating); until then the branch keeps its
  `// FIXED_POINT: not ported (float build)` marker. `FLOAT_APPROX` → `float-approx`,
  `FUZZING` → `fuzzing` (random draws from `crate::glibc_rand`), `DISABLE_UPDATE_DRAFT` →
  `disable-rfc8251`. `ENABLE_DEEP_PLC`,
  `ENABLE_DRED`, `ENABLE_OSCE`, `ENABLE_OSCE_BWE` → skip with `// DNN: <feature> not ported yet`
  marker (they are a later phase). `OPUS_ARM_*`, `OPUS_X86_*`, `arch` params → drop.
* `arch` arguments disappear. `RESTORE_STACK`/`SAVE_STACK` disappear.

**Rust quality (enforced by clippy, see `CLAUDE.md`):**
* `#![forbid(unsafe_code)]` everywhere. No `unwrap_or*`/silent error discard. `unwrap/expect` only
  with a reason. `const fn` wherever possible. `#[must_use]` on pure functions.
* Lint suppressions need `reason = "..."`; `clippy::too_many_arguments` and
  `clippy::needless_range_loop` may be allowed per-function/module with a reason like
  "mirrors C signature" / "index arithmetic mirrors C across several arrays". Prefer iterators
  when they are genuinely clearer and don't change evaluation order.
* Hot loops: prefer slicing to exact lengths up front (`let x = &x[..n];`) so bounds checks are
  hoisted.

## 3. Verification (required for every function)

1. **Unit-level differential tests** vs the oracle: write C shims in `csrc/<unit>.c` that call
   the internal libopus function with flat arguments (arrays + ints) and expose a safe Rust
   wrapper in `opusorus-oracle/src/<unit>.rs` (every `unsafe` block needs `// SAFETY:`). Then
   compare outputs bit-for-bit (`to_bits()` for floats) on thousands of randomized and edge
   inputs using `opusorus_conformance::Rng` (deterministic). Internal libopus symbols are
   linkable (static lib); static/inline functions can be reached by `#include`-ing the `.c`/`.h`
   into the shim or by re-exposing via the shim. Static tables in the oracle
   (e.g. `mode48000_960_120` for CELT) can be accessed through `opus_custom_mode_create(48000,
   960, NULL)` in a shim.
2. Tests must run in < ~30 s in the `test` profile (opt-level 2).
3. `cargo test -p opusorus-conformance --test <unit>` and the same with `--features qext`
   (if your unit has QEXT code) must pass (fixed-point units: with `--features fixed-point`,
   `fixed-res24` and their `qext` combinations). `just clippy` (float and fixed-point feature
   sets; `--all-features` is not a valid configuration since `fixed-point` is not additive, see
   `docs/FIXED_POINT.md`) and `cargo fmt --all -- --check` must be clean for your files.
4. Build must still pass for `cargo build -p opusorus --no-default-features` (no_std).

See `crates/opusorus-conformance/tests/foundation.rs` + `crates/opusorus-oracle/csrc/foundation.c`
for the reference pattern.
