//! Oracle bindings for the tools_compare unit: the unmodified C `opus_compare` and
//! `qext_compare` tools run in-process (see `csrc/tools_compare.c`).

use core::ffi::{c_char, c_int};
use std::ffi::{CStr, CString};

unsafe extern "C" {
    fn oracle_tools_opus_compare(
        argc: c_int,
        argv: *const *const c_char,
        text: *mut c_char,
        text_cap: usize,
        vals: *mut f64,
        max_vals: c_int,
        nvals: *mut c_int,
    ) -> c_int;
    fn oracle_tools_qext_compare(
        argc: c_int,
        argv: *const *const c_char,
        text: *mut c_char,
        text_cap: usize,
        vals: *mut f64,
        max_vals: c_int,
        nvals: *mut c_int,
    ) -> c_int;
}

/// What a C tool run produced.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolRun {
    /// `main`'s return value.
    pub exit_code: i32,
    /// Everything printed on stderr.
    pub stderr: String,
    /// Every `double` passed to a `%f` conversion, in print order (full precision).
    pub values: Vec<f64>,
}

type ToolFn = unsafe extern "C" fn(
    c_int,
    *const *const c_char,
    *mut c_char,
    usize,
    *mut f64,
    c_int,
    *mut c_int,
) -> c_int;

fn run(f: ToolFn, args: &[&str]) -> ToolRun {
    let owned: Vec<CString> = args
        .iter()
        .map(|a| CString::new(*a).expect("argument without NUL"))
        .collect();
    let mut argv: Vec<*const c_char> = owned.iter().map(|a| a.as_ptr()).collect();
    // C guarantees argv[argc] == NULL.
    argv.push(core::ptr::null());
    let mut text = vec![0 as c_char; 8192];
    let mut vals = [0f64; 16];
    let mut nvals: c_int = 0;
    let argc = c_int::try_from(args.len()).expect("argc fits in int");
    // SAFETY: argv holds `argc` valid NUL-terminated strings (kept alive by `owned`) followed by
    // NULL; `text`/`vals` are writable with the capacities passed; `nvals` is a valid out pointer.
    let exit_code = unsafe {
        f(
            argc,
            argv.as_ptr(),
            text.as_mut_ptr(),
            text.len(),
            vals.as_mut_ptr(),
            vals.len() as c_int,
            &mut nvals,
        )
    };
    // SAFETY: the shim always NUL-terminates `text` within its capacity.
    let stderr = unsafe { CStr::from_ptr(text.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    ToolRun {
        exit_code,
        stderr,
        values: vals[..nvals as usize].to_vec(),
    }
}

/// Runs C `opus_compare` (`src/opus_compare.c:main`) with `args` (including `argv[0]`).
///
/// # Panics
/// If an argument contains a NUL byte.
#[must_use]
pub fn opus_compare_main(args: &[&str]) -> ToolRun {
    run(oracle_tools_opus_compare, args)
}

/// Runs C `qext_compare` (`src/qext_compare.c:main`) with `args` (including `argv[0]`).
///
/// # Panics
/// If an argument contains a NUL byte.
#[must_use]
pub fn qext_compare_main(args: &[&str]) -> ToolRun {
    run(oracle_tools_qext_compare, args)
}
