//! Public constants and typed enums mirroring `opus_defines.h`.

use crate::Error;

/// Raw libopus constant values (for FFI mapping and interop).
pub mod raw {
    #![allow(missing_docs, reason = "values mirror opus_defines.h one-to-one")]
    pub const OPUS_AUTO: i32 = -1000;
    pub const OPUS_BITRATE_MAX: i32 = -1;
    pub const OPUS_APPLICATION_VOIP: i32 = 2048;
    pub const OPUS_APPLICATION_AUDIO: i32 = 2049;
    pub const OPUS_APPLICATION_RESTRICTED_LOWDELAY: i32 = 2051;
    pub const OPUS_APPLICATION_RESTRICTED_SILK: i32 = 2052;
    pub const OPUS_APPLICATION_RESTRICTED_CELT: i32 = 2053;
    pub const OPUS_SIGNAL_VOICE: i32 = 3001;
    pub const OPUS_SIGNAL_MUSIC: i32 = 3002;
    pub const OPUS_BANDWIDTH_NARROWBAND: i32 = 1101;
    pub const OPUS_BANDWIDTH_MEDIUMBAND: i32 = 1102;
    pub const OPUS_BANDWIDTH_WIDEBAND: i32 = 1103;
    pub const OPUS_BANDWIDTH_SUPERWIDEBAND: i32 = 1104;
    pub const OPUS_BANDWIDTH_FULLBAND: i32 = 1105;
    pub const OPUS_FRAMESIZE_ARG: i32 = 5000;
    pub const OPUS_FRAMESIZE_2_5_MS: i32 = 5001;
    pub const OPUS_FRAMESIZE_5_MS: i32 = 5002;
    pub const OPUS_FRAMESIZE_10_MS: i32 = 5003;
    pub const OPUS_FRAMESIZE_20_MS: i32 = 5004;
    pub const OPUS_FRAMESIZE_40_MS: i32 = 5005;
    pub const OPUS_FRAMESIZE_60_MS: i32 = 5006;
    pub const OPUS_FRAMESIZE_80_MS: i32 = 5007;
    pub const OPUS_FRAMESIZE_100_MS: i32 = 5008;
    pub const OPUS_FRAMESIZE_120_MS: i32 = 5009;
}

use raw::*;

macro_rules! raw_enum {
    ($(#[$m:meta])* $name:ident { $($(#[$vm:meta])* $var:ident = $val:expr),+ $(,)? }) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name {
            $($(#[$vm])* $var),+
        }
        impl $name {
            /// The libopus integer value.
            #[must_use]
            pub const fn to_raw(self) -> i32 {
                match self { $(Self::$var => $val),+ }
            }
            /// Parses a libopus integer value.
            ///
            /// # Errors
            /// [`Error::BadArg`] for unknown values.
            pub const fn from_raw(v: i32) -> Result<Self, Error> {
                $(if v == $val { return Ok(Self::$var); })+
                Err(Error::BadArg)
            }
        }
    };
}

raw_enum! {
    /// Coding mode / intended application (`OPUS_APPLICATION_*`).
    Application {
        /// Best for most VoIP/videoconference applications where listening quality and
        /// intelligibility matter most.
        Voip = OPUS_APPLICATION_VOIP,
        /// Best for broadcast/high-fidelity application where the decoded audio should be as
        /// close as possible to the input.
        Audio = OPUS_APPLICATION_AUDIO,
        /// Only use when lowest-achievable latency is what matters most.
        RestrictedLowDelay = OPUS_APPLICATION_RESTRICTED_LOWDELAY,
        /// SILK-only mode (no CELT support compiled in the encoder path).
        RestrictedSilk = OPUS_APPLICATION_RESTRICTED_SILK,
        /// CELT-only mode.
        RestrictedCelt = OPUS_APPLICATION_RESTRICTED_CELT,
    }
}

raw_enum! {
    /// Signal type hint (`OPUS_SIGNAL_*` / `OPUS_AUTO`).
    Signal {
        /// Let the encoder decide.
        Auto = OPUS_AUTO,
        /// Bias thresholds towards choosing LPC or Hybrid modes.
        Voice = OPUS_SIGNAL_VOICE,
        /// Bias thresholds towards choosing MDCT modes.
        Music = OPUS_SIGNAL_MUSIC,
    }
}

raw_enum! {
    /// Audio bandwidth (`OPUS_BANDWIDTH_*`).
    Bandwidth {
        /// 4 kHz passband.
        Narrowband = OPUS_BANDWIDTH_NARROWBAND,
        /// 6 kHz passband.
        Mediumband = OPUS_BANDWIDTH_MEDIUMBAND,
        /// 8 kHz passband.
        Wideband = OPUS_BANDWIDTH_WIDEBAND,
        /// 12 kHz passband.
        Superwideband = OPUS_BANDWIDTH_SUPERWIDEBAND,
        /// 20 kHz passband.
        Fullband = OPUS_BANDWIDTH_FULLBAND,
    }
}

raw_enum! {
    /// Expert frame duration control (`OPUS_FRAMESIZE_*`).
    FrameSize {
        /// Select frame size from the argument (default).
        Arg = OPUS_FRAMESIZE_ARG,
        /// 2.5 ms.
        Ms2_5 = OPUS_FRAMESIZE_2_5_MS,
        /// 5 ms.
        Ms5 = OPUS_FRAMESIZE_5_MS,
        /// 10 ms.
        Ms10 = OPUS_FRAMESIZE_10_MS,
        /// 20 ms.
        Ms20 = OPUS_FRAMESIZE_20_MS,
        /// 40 ms.
        Ms40 = OPUS_FRAMESIZE_40_MS,
        /// 60 ms.
        Ms60 = OPUS_FRAMESIZE_60_MS,
        /// 80 ms.
        Ms80 = OPUS_FRAMESIZE_80_MS,
        /// 100 ms.
        Ms100 = OPUS_FRAMESIZE_100_MS,
        /// 120 ms.
        Ms120 = OPUS_FRAMESIZE_120_MS,
    }
}

/// Encoder bitrate setting (`OPUS_SET_BITRATE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bitrate {
    /// `OPUS_AUTO`.
    Auto,
    /// `OPUS_BITRATE_MAX`.
    Max,
    /// Explicit bits per second.
    Bits(i32),
}

impl Bitrate {
    /// The libopus integer value.
    #[must_use]
    pub const fn to_raw(self) -> i32 {
        match self {
            Self::Auto => OPUS_AUTO,
            Self::Max => OPUS_BITRATE_MAX,
            Self::Bits(b) => b,
        }
    }
    /// Parses a libopus integer value (any value is representable; validation happens in the
    /// encoder, as in libopus).
    #[must_use]
    pub const fn from_raw(v: i32) -> Self {
        match v {
            OPUS_AUTO => Self::Auto,
            OPUS_BITRATE_MAX => Self::Max,
            b => Self::Bits(b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for v in [2048, 2049, 2051, 2052, 2053] {
            assert_eq!(Application::from_raw(v).map(Application::to_raw), Ok(v));
        }
        assert_eq!(Application::from_raw(0), Err(Error::BadArg));
        for v in 1101..=1105 {
            assert_eq!(Bandwidth::from_raw(v).map(Bandwidth::to_raw), Ok(v));
        }
        for v in 5000..=5009 {
            assert_eq!(FrameSize::from_raw(v).map(FrameSize::to_raw), Ok(v));
        }
        assert_eq!(Bitrate::from_raw(-1000), Bitrate::Auto);
        assert_eq!(Bitrate::from_raw(64000).to_raw(), 64000);
    }
}
