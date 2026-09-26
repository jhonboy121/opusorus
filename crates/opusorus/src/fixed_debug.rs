//! Diagnostics of the checking fixed-point arithmetic (feature `fixed-point-debug`, libopus
//! `--enable-fixed-point-debug` / `FIXED_DEBUG`).
//!
//! With this feature the fixed-point macros are the ports of `celt/fixed_debug.h` and
//! `silk/MacroDebug.h` instead of `celt/fixed_generic.h` / the release SILK macros: every macro
//! checks the range of its operands and of its result, and on a violation reports the same
//! message libopus prints (`fprintf(stderr, ...)`), then **continues** with the value the C
//! debug macro computes (so the output stays bit-exact with a libopus `FIXED_DEBUG` build, which
//! differs from a release build where operands are out of range, and in a few macro forms, e.g.
//! `MULT32_32_Q31` is always the 32-bit partial-product form). The CELT macros also count
//! "operations" in libopus' `celt_mips` counter.
//!
//! Where C passes `__FILE__` / `__LINE__`, the message names the Rust call site of the macro
//! (`#[track_caller]`) instead of the C one.
//!
//! # Delivery of the messages
//!
//! * With `std` (default), each message is written to standard error, exactly like libopus,
//!   unless a handler is installed on the current thread with [`set_handler`] (for example to
//!   collect the messages, log them, or panic, the Rust counterpart of libopus'
//!   `FIXED_DEBUG_ASSERT`).
//! * Without `std` there is no standard error: messages are only counted
//!   ([`diagnostics_count`]).
//!
//! The counters ([`celt_mips`], [`diagnostics_count`]) are per thread with `std` (the codec
//! runs on the caller's thread, so a test or a tool sees its own counts) and global without it
//! (a wrapping 32-bit counter on targets without 64-bit atomics).

#[cfg(feature = "std")]
mod imp {
    use alloc::boxed::Box;
    use core::cell::{Cell, RefCell};

    /// A diagnostic handler: receives each message (with its trailing `\n`).
    pub type Handler = Box<dyn FnMut(&str)>;

    std::thread_local! {
        static MIPS: Cell<i64> = const { Cell::new(0) };
        static COUNT: Cell<u64> = const { Cell::new(0) };
        static HANDLER: RefCell<Option<Handler>> = const { RefCell::new(None) };
    }

    #[inline(always)]
    pub fn mips_add(n: i64) {
        MIPS.with(|m| m.set(m.get().wrapping_add(n)));
    }
    pub fn celt_mips() -> i64 {
        MIPS.with(Cell::get)
    }
    pub fn set_celt_mips(v: i64) {
        MIPS.with(|m| m.set(v));
    }
    pub fn diagnostics_count() -> u64 {
        COUNT.with(Cell::get)
    }
    pub fn reset_diagnostics_count() {
        COUNT.with(|c| c.set(0));
    }
    pub fn set_handler(handler: Option<Handler>) -> Option<Handler> {
        HANDLER.with(|h| core::mem::replace(&mut *h.borrow_mut(), handler))
    }
    #[cold]
    pub fn report(args: core::fmt::Arguments<'_>) {
        COUNT.with(|c| c.set(c.get().wrapping_add(1)));
        // Take the handler out while it runs, so a handler that itself uses the codec (and
        // reports) cannot deadlock on the RefCell: nested reports go to stderr.
        let handler = HANDLER.with(|h| h.borrow_mut().take());
        match handler {
            Some(mut h) => {
                let msg = alloc::fmt::format(args);
                h(&msg);
                HANDLER.with(|slot| {
                    let mut slot = slot.borrow_mut();
                    if slot.is_none() {
                        *slot = Some(h);
                    }
                });
            }
            None => {
                #[expect(
                    clippy::print_stderr,
                    reason = "FIXED_DEBUG reports to stderr, like libopus' fprintf(stderr, ...)"
                )]
                {
                    std::eprint!("{args}");
                }
            }
        }
    }
}

#[cfg(not(feature = "std"))]
#[allow(
    clippy::useless_conversion,
    reason = "the counters are 32-bit on targets without 64-bit atomics"
)]
mod imp {
    use core::sync::atomic::Ordering::Relaxed;
    #[cfg(not(target_has_atomic = "64"))]
    use core::sync::atomic::{AtomicI32 as AtomicMips, AtomicU32 as AtomicCount};
    #[cfg(target_has_atomic = "64")]
    use core::sync::atomic::{AtomicI64 as AtomicMips, AtomicU64 as AtomicCount};

    static MIPS: AtomicMips = AtomicMips::new(0);
    static COUNT: AtomicCount = AtomicCount::new(0);

    #[inline(always)]
    pub fn mips_add(n: i64) {
        // Truncation on targets without 64-bit atomics is the documented wrapping.
        MIPS.fetch_add(n as _, Relaxed);
    }
    pub fn celt_mips() -> i64 {
        i64::from(MIPS.load(Relaxed))
    }
    pub fn set_celt_mips(v: i64) {
        MIPS.store(v as _, Relaxed);
    }
    pub fn diagnostics_count() -> u64 {
        u64::from(COUNT.load(Relaxed))
    }
    pub fn reset_diagnostics_count() {
        COUNT.store(0, Relaxed);
    }
    #[cold]
    pub fn report(_args: core::fmt::Arguments<'_>) {
        COUNT.fetch_add(1, Relaxed);
    }
}

#[cfg(feature = "std")]
pub use imp::Handler;

/// libopus' `celt_mips`: the number of fixed-point "operations" the CELT macros counted (each
/// macro adds its C weight, e.g. 1 for `MULT16_16`, 35 for `DIV32_16`).
#[must_use]
pub fn celt_mips() -> i64 {
    imp::celt_mips()
}

/// Sets the `celt_mips` counter (e.g. to 0 before measuring).
pub fn set_celt_mips(value: i64) {
    imp::set_celt_mips(value);
}

/// Number of diagnostics reported since the last [`reset_diagnostics_count`].
#[must_use]
pub fn diagnostics_count() -> u64 {
    imp::diagnostics_count()
}

/// Resets [`diagnostics_count`].
pub fn reset_diagnostics_count() {
    imp::reset_diagnostics_count();
}

/// Installs `handler` for the diagnostics reported on the current thread (`None`: print to
/// standard error, the default) and returns the previous one.
///
/// ```
/// use std::sync::{Arc, Mutex};
///
/// let seen = Arc::new(Mutex::new(Vec::new()));
/// let sink = Arc::clone(&seen);
/// let previous = opusorus::fixed_debug::set_handler(Some(Box::new(move |msg: &str| {
///     sink.lock().unwrap().push(msg.to_owned());
/// })));
/// // ... run the codec ...
/// opusorus::fixed_debug::set_handler(previous);
/// ```
#[cfg(feature = "std")]
pub fn set_handler(handler: Option<Handler>) -> Option<Handler> {
    imp::set_handler(handler)
}

/// Adds `n` to `celt_mips` (`celt_mips+=n` in `celt/fixed_debug.h`).
#[inline(always)]
pub(crate) fn mips(n: i64) {
    imp::mips_add(n);
}

/// Reports a diagnostic (`fprintf(stderr, ...)` in `celt/fixed_debug.h` / `silk/MacroDebug.h`).
#[cold]
#[inline(never)]
pub(crate) fn report(args: core::fmt::Arguments<'_>) {
    imp::report(args);
}

/// `fprintf(stderr, ...)` of the checking macros: formats and reports a diagnostic.
macro_rules! fdbg {
    ($($arg:tt)*) => {
        $crate::fixed_debug::report(format_args!($($arg)*))
    };
}
pub(crate) use fdbg;
