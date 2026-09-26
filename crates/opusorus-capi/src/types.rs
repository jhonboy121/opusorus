//! Opaque C handle types (`typedef struct OpusEncoder OpusEncoder;` etc.).
//!
//! C code only ever sees pointers to these; the memory behind them starts with a
//! [`crate::handle`] header.

macro_rules! opaque {
    ($($(#[$m:meta])* $name:ident;)*) => {$(
        $(#[$m])*
        #[repr(C)]
        #[derive(Debug)]
        pub struct $name {
            _opaque: [u8; 0],
        }
    )*};
}

opaque! {
    /// `OpusEncoder`.
    OpusEncoder;
    /// `OpusDecoder`.
    OpusDecoder;
    /// `OpusMSEncoder`.
    OpusMSEncoder;
    /// `OpusMSDecoder`.
    OpusMSDecoder;
    /// `OpusProjectionEncoder`.
    OpusProjectionEncoder;
    /// `OpusProjectionDecoder`.
    OpusProjectionDecoder;
    /// `OpusRepacketizer`.
    OpusRepacketizer;
    /// `OpusDREDDecoder`.
    OpusDREDDecoder;
    /// `OpusDRED`.
    OpusDRED;
    /// `OpusCustomEncoder`.
    OpusCustomEncoder;
    /// `OpusCustomDecoder`.
    OpusCustomDecoder;
    /// `OpusCustomMode`.
    OpusCustomMode;
}
