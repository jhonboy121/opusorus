//! Encodes a generated tone with opusorus, decodes it again and prints the signal-to-noise ratio
//! of the round trip.
//!
//! ```text
//! cargo run --release -p opusorus --example roundtrip [bitrate_bps] [frequency_hz]
//! ```

#![cfg_attr(
    not(feature = "fixed-point"),
    expect(clippy::print_stdout, reason = "command-line example")
)]

#[cfg(not(feature = "fixed-point"))]
use std::error::Error;

#[cfg(not(feature = "fixed-point"))]
use opusorus::{Application, Bitrate, Decoder, Encoder};

#[cfg(not(feature = "fixed-point"))]
const FS: i32 = 48000;
#[cfg(not(feature = "fixed-point"))]
const CHANNELS: usize = 2;
/// 20 ms frames.
#[cfg(not(feature = "fixed-point"))]
const FRAME_SIZE: usize = 960;
#[cfg(not(feature = "fixed-point"))]
const SECONDS: usize = 2;

#[cfg(not(feature = "fixed-point"))]
fn arg<T: std::str::FromStr>(n: usize, default: T) -> Result<T, Box<dyn Error>>
where
    T::Err: Error + 'static,
{
    match std::env::args().nth(n) {
        Some(s) => Ok(s.parse()?),
        None => Ok(default),
    }
}

#[cfg(not(feature = "fixed-point"))]
fn main() -> Result<(), Box<dyn Error>> {
    let bitrate: i32 = arg(1, 64000)?;
    let freq: f64 = arg(2, 440.0)?;

    // A stereo tone (slightly different frequency on the right channel) at -6 dBFS.
    let total = FS as usize * SECONDS;
    let mut input = vec![0i16; total * CHANNELS];
    for (i, frame) in input.as_chunks_mut::<CHANNELS>().0.iter_mut().enumerate() {
        let t = i as f64 / f64::from(FS);
        frame[0] = (16384.0 * (2.0 * std::f64::consts::PI * freq * t).sin()) as i16;
        frame[1] = (16384.0 * (2.0 * std::f64::consts::PI * freq * 1.5 * t).sin()) as i16;
    }

    let mut enc = Encoder::new(FS, CHANNELS as i32, Application::Audio)?;
    enc.set_bitrate(Bitrate::Bits(bitrate))?;
    let mut dec = Decoder::new(FS, CHANNELS as i32)?;

    let mut packet = [0u8; 1500];
    let mut output = vec![0i16; total * CHANNELS];
    let mut bytes = 0usize;
    let mut packets = 0usize;
    for (pcm, out) in input
        .as_chunks::<{ FRAME_SIZE * CHANNELS }>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<{ FRAME_SIZE * CHANNELS }>().0)
    {
        let len = enc.encode(pcm, FRAME_SIZE, &mut packet)?;
        let n = dec.decode(Some(&packet[..len]), out, FRAME_SIZE, false)?;
        assert_eq!(n, FRAME_SIZE);
        // The decoder ends in the same range coder state as the encoder.
        assert_eq!(enc.final_range(), dec.final_range());
        bytes += len;
        packets += 1;
    }

    // The decoded signal lags the input by the encoder look-ahead.
    let delay = enc.lookahead();
    let mut signal = 0f64;
    let mut noise = 0f64;
    for i in 0..(total - delay) * CHANNELS {
        let x = f64::from(input[i]);
        let y = f64::from(output[i + delay * CHANNELS]);
        signal += x * x;
        noise += (x - y) * (x - y);
    }
    let snr = 10.0 * (signal / noise).log10();
    let kbps = bytes as f64 * 8.0 / SECONDS as f64 / 1000.0;
    println!(
        "{packets} packets, {bytes} bytes ({kbps:.1} kb/s, target {} kb/s), look-ahead {delay} \
         samples",
        bitrate / 1000
    );
    println!("round-trip SNR: {snr:.2} dB");
    Ok(())
}

/// The in-progress fixed-point build (feature `fixed-point`) has no codec API yet.
#[cfg(feature = "fixed-point")]
fn main() {}
