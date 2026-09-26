//! Encodes raw interleaved 16-bit little-endian PCM to an `opus_demo` bitstream file (for every
//! packet a 32-bit big-endian length, the 32-bit big-endian encoder final range, then the
//! packet), like `opus_demo -e`. The output decodes with `decode_file` or `opus_demo -d`.
//!
//! ```text
//! cargo run --release -p opusorus --example encode_file -- in.pcm out.bit \
//!     [rate=48000] [channels=2] [bitrate=64000] [voip|audio|lowdelay] [frame_ms=20]
//! ```
//!
//! The last partial frame is padded with silence.

#![expect(clippy::print_stdout, reason = "command-line example")]

use std::error::Error;
use std::fs;

use opusorus::{Application, Bitrate, Encoder};

fn parse_arg<T: std::str::FromStr>(
    args: &[String],
    n: usize,
    default: T,
) -> Result<T, Box<dyn Error>>
where
    T::Err: Error + 'static,
{
    match args.get(n) {
        Some(s) => Ok(s.parse()?),
        None => Ok(default),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err(format!(
            "usage: {} <input.pcm> <output.bit> [rate] [channels] [bitrate] \
             [voip|audio|lowdelay] [frame_ms]",
            args[0]
        )
        .into());
    }
    let rate: i32 = parse_arg(&args, 3, 48000)?;
    let channels: i32 = parse_arg(&args, 4, 2)?;
    let bitrate: i32 = parse_arg(&args, 5, 64000)?;
    let application = match args.get(6).map(String::as_str) {
        None | Some("audio") => Application::Audio,
        Some("voip") => Application::Voip,
        Some("lowdelay") => Application::RestrictedLowDelay,
        Some(other) => return Err(format!("unknown application {other:?}").into()),
    };
    let frame_ms: f64 = parse_arg(&args, 7, 20.0)?;
    let frame_size = (f64::from(rate) * frame_ms / 1000.0) as usize;

    let raw = fs::read(&args[1])?;
    let samples: Vec<i16> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| i16::from_le_bytes(b))
        .collect();

    let mut enc = Encoder::new(rate, channels, application)?;
    enc.set_bitrate(Bitrate::Bits(bitrate))?;

    let ch = channels as usize;
    let mut frame = vec![0i16; frame_size * ch];
    let mut packet = vec![0u8; 1500 * 6];
    let mut out = Vec::new();
    let mut packets = 0usize;
    for chunk in samples.chunks(frame_size * ch) {
        frame[..chunk.len()].copy_from_slice(chunk);
        frame[chunk.len()..].fill(0);
        let len = enc.encode(&frame, frame_size, &mut packet)?;
        out.extend_from_slice(&(len as u32).to_be_bytes());
        out.extend_from_slice(&enc.final_range().to_be_bytes());
        out.extend_from_slice(&packet[..len]);
        packets += 1;
    }
    fs::write(&args[2], &out)?;
    let seconds = samples.len() as f64 / ch as f64 / f64::from(rate);
    println!(
        "{packets} packets, {} bytes of payload for {seconds:.2} s ({:.1} kb/s)",
        out.len() - 8 * packets,
        (out.len() - 8 * packets) as f64 * 8.0 / seconds.max(1e-9) / 1000.0
    );
    Ok(())
}
