//! Multistream channel layout helpers shared by the multistream encoder and decoder.
//!
//! Port of `src/opus_multistream.c` and the `ChannelLayout` / `MappingType` types of
//! `src/opus_private.h`.

/// Maps the output channels of a multistream stream to/from the coded streams
/// (`ChannelLayout`).
///
/// Channel `i` is carried by coded channel `mapping[i]`: values below
/// `2 * nb_coupled_streams` are the left/right channels of the coupled streams, the following
/// values the mono streams, and `255` a silent channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelLayout {
    /// Number of output channels.
    pub nb_channels: i32,
    /// Total number of streams.
    pub nb_streams: i32,
    /// Number of coupled (stereo) streams.
    pub nb_coupled_streams: i32,
    /// Coded channel for each output channel.
    pub mapping: [u8; 256],
}

impl Default for ChannelLayout {
    fn default() -> Self {
        Self {
            nb_channels: 0,
            nb_streams: 0,
            nb_coupled_streams: 0,
            mapping: [0; 256],
        }
    }
}

/// Kind of channel mapping used by the multistream encoder (`MappingType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MappingType {
    /// No special mapping (`MAPPING_TYPE_NONE`).
    #[default]
    None,
    /// Vorbis-order surround (`MAPPING_TYPE_SURROUND`).
    Surround,
    /// Ambisonics (`MAPPING_TYPE_AMBISONICS`).
    Ambisonics,
}

/// Number of mapping entries to inspect: `nb_channels`, clamped to the table size (C reads
/// out of bounds for more than 256 channels).
#[inline]
const fn channels(layout: &ChannelLayout) -> usize {
    if layout.nb_channels <= 0 {
        0
    } else if layout.nb_channels > 256 {
        256
    } else {
        layout.nb_channels as usize
    }
}

/// Port of `src/opus_multistream.c:validate_layout`: whether every channel maps to an existing
/// coded channel (or is silent) and there are at most 255 coded channels.
#[must_use]
pub const fn validate_layout(layout: &ChannelLayout) -> bool {
    let max_channel = layout.nb_streams + layout.nb_coupled_streams;
    if max_channel > 255 {
        return false;
    }
    let n = channels(layout);
    let mut i = 0;
    while i < n {
        let m = layout.mapping[i] as i32;
        if m >= max_channel && m != 255 {
            return false;
        }
        i += 1;
    }
    true
}

/// Shared search of `get_left_channel` & co: first channel after `prev` (or from the start if
/// `prev < 0`) whose mapping is `target`, or -1.
const fn find_channel(layout: &ChannelLayout, target: i32, prev: i32) -> i32 {
    let mut i: i32 = if prev < 0 { 0 } else { prev + 1 };
    let n = channels(layout) as i32;
    while i < n {
        if layout.mapping[i as usize] as i32 == target {
            return i;
        }
        i += 1;
    }
    -1
}

/// Port of `src/opus_multistream.c:get_left_channel`: the next output channel after `prev`
/// (pass -1 to start) carrying the left channel of coupled stream `stream_id`, or -1.
#[must_use]
pub const fn get_left_channel(layout: &ChannelLayout, stream_id: i32, prev: i32) -> i32 {
    find_channel(layout, stream_id * 2, prev)
}

/// Port of `src/opus_multistream.c:get_right_channel`: the next output channel after `prev`
/// (pass -1 to start) carrying the right channel of coupled stream `stream_id`, or -1.
#[must_use]
pub const fn get_right_channel(layout: &ChannelLayout, stream_id: i32, prev: i32) -> i32 {
    find_channel(layout, stream_id * 2 + 1, prev)
}

/// Port of `src/opus_multistream.c:get_mono_channel`: the next output channel after `prev`
/// (pass -1 to start) carrying mono stream `stream_id` (counted from 0 over all streams, so
/// mono streams start at `nb_coupled_streams`), or -1.
#[must_use]
pub const fn get_mono_channel(layout: &ChannelLayout, stream_id: i32, prev: i32) -> i32 {
    find_channel(layout, stream_id + layout.nb_coupled_streams, prev)
}
