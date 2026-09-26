# opusorus — project rules

Pure safe-Rust port of **libopus v1.6.1** (vendored at `vendor/libopus`, never modify it).
C libopus is the **oracle**: every ported component must be verified against it.

## Hard rules
- **Git commits: plain messages. NO `Co-Authored-By`, no Claude/session trailers.**
- Edition 2024. `unsafe_code = "forbid"` in every crate except `opusorus-oracle` (test-only FFI) and
  `opusorus-capi` (C ABI shim); unsafe there must be minimal and each block commented `// SAFETY:`.
- No error swallowing: never `unwrap_or(..)`, `unwrap_or_default()`, `unwrap_or_else(..)`, `.ok()` to
  discard errors, or `let _ =` on a `Result`. Propagate with `?` / typed `Error`.
- `unwrap()`/`expect()` only with `#[expect(clippy::unwrap_used/expect_used, reason = "...")]` or an
  adjacent comment stating why it cannot fail. Tests may unwrap freely.
- Mark functions `const fn` wherever possible (clippy `missing_const_for_fn` is on).
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` must pass. Lint suppressions
  need `reason = "..."` (use `#[expect]` over `#[allow]` where the lint always fires).
- `cargo fmt --all` clean.
- Must build for host, wasm32-unknown-unknown, wasm32-wasip1, aarch64/armv7/x86_64 Android,
  aarch64/x86_64 iOS (+ sim). Run `just cross` to verify.
- `opusorus` crate is `no_std` + `alloc`; feature `std` (default) uses std float math, otherwise `libm`.
  All transcendental math goes through `crate::math` — never call `f32::cos` etc. directly.

## Porting conventions (see docs/PORTING.md for full detail)
- Faithful function-by-function translation; keep C names (snake_case) and cite the C source in doc
  comments: `/// Port of celt/bands.c:quant_all_bands`.
- Float build semantics: bit-exact to the oracle (built with `-ffp-contract=off`, no intrinsics).
  Preserve operation order; replicate C float→double promotions (`x*0.5` with `x: float` is a double
  multiply!); never use `mul_add`; `float2int`/`lrintf` = round-half-even.
- Integer code (SILK, fixed-point) must be bit-exact. Overflow-wrapping macros (`*_ovflw`, unsigned
  math) use `wrapping_*`; everything else uses plain ops so debug builds catch bugs.
- Differential tests live in `crates/opusorus-conformance/tests/<module>.rs`; FFI declarations for
  them in `crates/opusorus-oracle/src/<module>.rs`.

## Docs to keep current
`docs/FEATURES.md` (feature list), `docs/TRACKER.md` (per-file port status, tests, perf),
`docs/STATUS.md` (overall status, benchmarks, size comparisons), `docs/PLAN.md` (plan + decision log).

## Commands
`just` lists recipes: `just check`, `just test`, `just clippy`, `just cross`, `just bench`, `just size`,
`just fuzz <target>`, `just vectors`.
