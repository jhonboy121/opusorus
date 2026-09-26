//! Decodes an `opus_demo` bitstream file (the framed `.bit` format: for every packet a 32-bit
//! big-endian length, the 32-bit big-endian encoder final range, then the packet) to raw
//! interleaved 16-bit little-endian PCM, like `opus_demo -d`.
//!
//! ```text
//! cargo run --release -p opusorus --example decode_file -- in.bit out.pcm [rate] [channels]
//! ```
//!
//! A zero-length packet is a lost packet (concealed with the last packet duration). The final
//! range of every packet is compared with the decoder's, as `opus_demo` does.

#![expect(clippy::print_stdout, reason = "command-line example")]

use std::error::Error;
use std::fs;

use opusorus::Decoder;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err(format!(
            "usage: {} <input.bit> <output.pcm> [rate] [channels]",
            args[0]
        )
        .into());
    }
    let rate: i32 = match args.get(3) {
        Some(s) => s.parse()?,
        None => 48000,
    };
    let channels: i32 = match args.get(4) {
        Some(s) => s.parse()?,
        None => 2,
    };

    let input = fs::read(&args[1])?;
    let mut dec = Decoder::new(rate, channels)?;
    let max_frame = rate as usize * 120 / 1000;
    let ch = channels as usize;
    let mut pcm = vec![0i16; max_frame * ch];
    let mut out = Vec::new();
    let (mut packets, mut lost, mut mismatches, mut samples) = (0usize, 0usize, 0usize, 0usize);

    let mut pos = 0;
    while pos < input.len() {
        let header = input.get(pos..pos + 8).ok_or("truncated packet header")?;
        let len = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let enc_range = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);
        pos += 8;
        let data = input.get(pos..pos + len).ok_or("truncated packet")?;
        pos += len;

        let n = if len == 0 {
            // Lost packet: conceal as much audio as the last packet held.
            lost += 1;
            let dur = match dec.last_packet_duration() {
                0 => rate as usize / 50,
                d => d,
            };
            dec.decode(None, &mut pcm, dur, false)?
        } else {
            let n = dec.decode(Some(data), &mut pcm, max_frame, false)?;
            if dec.final_range() != enc_range {
                mismatches += 1;
            }
            n
        };
        packets += 1;
        samples += n;
        for s in &pcm[..n * ch] {
            out.extend_from_slice(&s.to_le_bytes());
        }
    }
    fs::write(&args[2], &out)?;
    println!(
        "{packets} packets ({lost} lost), {samples} samples per channel at {rate} Hz, {} final \
         range mismatches",
        mismatches
    );
    if mismatches > 0 {
        return Err("final range mismatch".into());
    }
    Ok(())
}
