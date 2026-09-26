//! Error type mirroring libopus error codes (`opus_defines.h`).

use core::fmt;

/// Errors returned by opusorus. Each variant maps 1:1 to a libopus `OPUS_*` error code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    /// One or more invalid/out of range arguments (`OPUS_BAD_ARG`).
    BadArg,
    /// Not enough bytes allocated in the buffer (`OPUS_BUFFER_TOO_SMALL`).
    BufferTooSmall,
    /// An internal error was detected (`OPUS_INTERNAL_ERROR`).
    InternalError,
    /// The compressed data passed is corrupted (`OPUS_INVALID_PACKET`).
    InvalidPacket,
    /// Invalid/unsupported request number (`OPUS_UNIMPLEMENTED`).
    Unimplemented,
    /// An encoder or decoder structure is invalid or already freed (`OPUS_INVALID_STATE`).
    InvalidState,
    /// Memory allocation has failed (`OPUS_ALLOC_FAIL`).
    AllocFail,
}

/// Convenience alias.
pub type Result<T> = core::result::Result<T, Error>;

impl Error {
    /// The libopus integer error code (always negative).
    #[must_use]
    pub const fn code(self) -> i32 {
        match self {
            Self::BadArg => -1,
            Self::BufferTooSmall => -2,
            Self::InternalError => -3,
            Self::InvalidPacket => -4,
            Self::Unimplemented => -5,
            Self::InvalidState => -6,
            Self::AllocFail => -7,
        }
    }

    /// Maps a negative libopus error code back to an [`Error`]. Returns `None` for `0`/positive
    /// values; unknown negative codes map to [`Error::InternalError`] (as `opus_strerror` does).
    #[must_use]
    pub const fn from_code(code: i32) -> Option<Self> {
        Some(match code {
            c if c >= 0 => return None,
            -1 => Self::BadArg,
            -2 => Self::BufferTooSmall,
            -4 => Self::InvalidPacket,
            -5 => Self::Unimplemented,
            -6 => Self::InvalidState,
            -7 => Self::AllocFail,
            _ => Self::InternalError,
        })
    }

    /// Same strings as libopus `opus_strerror`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BadArg => "invalid argument",
            Self::BufferTooSmall => "buffer too small",
            Self::InternalError => "internal error",
            Self::InvalidPacket => "corrupted stream",
            Self::Unimplemented => "request not implemented",
            Self::InvalidState => "invalid state",
            Self::AllocFail => "memory allocation failed",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl core::error::Error for Error {}
