//! Oracle bindings for unit `analysis` (`src/analysis.c` + the downmix helpers of
//! `src/opus_encoder.c`), via `csrc/analysis.c`.
//!
//! The C state `TonalityAnalysisState` and `AnalysisInfo` are mirrored as `#[repr(C)]` structs
//! ([`CTonalityAnalysisState`], [`CAnalysisInfo`]) so tests can read and write the complete C
//! state; [`layout_matches`] checks the mirror against the C compiler's layout.
//!
//! Built in both oracles: [`Val32`] (`opus_val32`: signal buffers, `inmem`, the resampler
//! state) and [`Res`] (`opus_res`) follow the build.

use core::ffi::{c_int, c_void};

/// `opus_val32` (float build).
#[cfg(not(feature = "fixed-point"))]
pub type Val32 = f32;
/// `opus_val32` (fixed-point build).
#[cfg(feature = "fixed-point")]
pub type Val32 = i32;
/// `opus_res` (float build).
#[cfg(not(feature = "fixed-point"))]
pub type Res = f32;
/// `opus_res` (fixed-point build, 16-bit resolution).
#[cfg(all(feature = "fixed-point", not(feature = "fixed-res24")))]
pub type Res = i16;
/// `opus_res` (fixed-point build, `ENABLE_RES24`).
#[cfg(feature = "fixed-res24")]
pub type Res = i32;

/// `NB_FRAMES`.
pub const NB_FRAMES: usize = 8;
/// `NB_TBANDS`.
pub const NB_TBANDS: usize = 18;
/// `ANALYSIS_BUF_SIZE`.
pub const ANALYSIS_BUF_SIZE: usize = 720;
/// `DETECT_SIZE`.
pub const DETECT_SIZE: usize = 100;
/// `LEAK_BANDS`.
pub const LEAK_BANDS: usize = 19;
/// `MAX_NEURONS`.
pub const MAX_NEURONS: usize = 32;

/// `#[repr(C)]` mirror of `AnalysisInfo` (celt/celt.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[allow(missing_docs, reason = "fields mirror the C struct")]
pub struct CAnalysisInfo {
    pub valid: c_int,
    pub tonality: f32,
    pub tonality_slope: f32,
    pub noisiness: f32,
    pub activity: f32,
    pub music_prob: f32,
    pub music_prob_min: f32,
    pub music_prob_max: f32,
    pub bandwidth: c_int,
    pub activity_probability: f32,
    pub max_pitch_ratio: f32,
    pub leak_boost: [u8; LEAK_BANDS],
}

/// `#[repr(C)]` mirror of `TonalityAnalysisState` (src/analysis.h).
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs, reason = "fields mirror the C struct")]
pub struct CTonalityAnalysisState {
    pub arch: c_int,
    pub application: c_int,
    pub fs: i32,
    pub angle: [f32; 240],
    pub d_angle: [f32; 240],
    pub d2_angle: [f32; 240],
    pub inmem: [Val32; ANALYSIS_BUF_SIZE],
    pub mem_fill: c_int,
    pub prev_band_tonality: [f32; NB_TBANDS],
    pub prev_tonality: f32,
    pub prev_bandwidth: c_int,
    pub e: [[f32; NB_TBANDS]; NB_FRAMES],
    pub log_e: [[f32; NB_TBANDS]; NB_FRAMES],
    pub low_e: [f32; NB_TBANDS],
    pub high_e: [f32; NB_TBANDS],
    pub mean_e: [f32; NB_TBANDS + 1],
    pub mem: [f32; 32],
    pub cmean: [f32; 8],
    pub std: [f32; 9],
    pub e_tracker: f32,
    pub low_e_count: f32,
    pub e_count: c_int,
    pub count: c_int,
    pub analysis_offset: c_int,
    pub write_pos: c_int,
    pub read_pos: c_int,
    pub read_subframe: c_int,
    pub hp_ener_accum: f32,
    pub initialized: c_int,
    pub rnn_state: [f32; MAX_NEURONS],
    pub downmix_state: [Val32; 3],
    pub info: [CAnalysisInfo; DETECT_SIZE],
}

impl CTonalityAnalysisState {
    /// An all-zero state (as after `memset`); call [`tonality_analysis_init`] before use.
    #[must_use]
    pub fn zeroed() -> Box<Self> {
        Box::new(Self {
            arch: 0,
            application: 0,
            fs: 0,
            angle: [0.0; 240],
            d_angle: [0.0; 240],
            d2_angle: [0.0; 240],
            inmem: [Val32::default(); ANALYSIS_BUF_SIZE],
            mem_fill: 0,
            prev_band_tonality: [0.0; NB_TBANDS],
            prev_tonality: 0.0,
            prev_bandwidth: 0,
            e: [[0.0; NB_TBANDS]; NB_FRAMES],
            log_e: [[0.0; NB_TBANDS]; NB_FRAMES],
            low_e: [0.0; NB_TBANDS],
            high_e: [0.0; NB_TBANDS],
            mean_e: [0.0; NB_TBANDS + 1],
            mem: [0.0; 32],
            cmean: [0.0; 8],
            std: [0.0; 9],
            e_tracker: 0.0,
            low_e_count: 0.0,
            e_count: 0,
            count: 0,
            analysis_offset: 0,
            write_pos: 0,
            read_pos: 0,
            read_subframe: 0,
            hp_ener_accum: 0.0,
            initialized: 0,
            rnn_state: [0.0; MAX_NEURONS],
            downmix_state: [Val32::default(); 3],
            info: [CAnalysisInfo::default(); DETECT_SIZE],
        })
    }

    /// A state initialized with C `tonality_analysis_init`.
    #[must_use]
    pub fn new(fs: i32) -> Box<Self> {
        let mut s = Self::zeroed();
        tonality_analysis_init(&mut s, fs);
        s
    }

    fn as_mut_ptr(&mut self) -> *mut c_void {
        core::ptr::from_mut(self).cast()
    }
}

unsafe extern "C" {
    fn oracle_analysis_state_size() -> usize;
    fn oracle_analysis_info_size() -> usize;
    fn oracle_analysis_state_offset(which: c_int) -> usize;
    fn oracle_analysis_m_pi() -> f64;
    fn oracle_tonality_analysis_init(st: *mut c_void, fs: c_int);
    fn oracle_tonality_analysis_reset(st: *mut c_void);
    fn oracle_tonality_get_info(st: *mut c_void, info_out: *mut c_void, len: c_int);
    fn oracle_run_analysis(
        st: *mut c_void,
        pcm_type: c_int,
        pcm: *const c_void,
        analysis_frame_size: c_int,
        frame_size: c_int,
        c1: c_int,
        c2: c_int,
        c: c_int,
        fs: c_int,
        lsb_depth: c_int,
        info_out: *mut c_void,
    );
    fn oracle_downmix(
        pcm_type: c_int,
        x: *const c_void,
        y: *mut Val32,
        subframe: c_int,
        offset: c_int,
        c1: c_int,
        c2: c_int,
        c: c_int,
    );
    fn oracle_is_digital_silence(
        pcm: *const Res,
        frame_size: c_int,
        channels: c_int,
        lsb_depth: c_int,
    ) -> c_int;
    fn oracle_silk_resampler_down2_hp(
        s: *mut Val32,
        out: *mut Val32,
        input: *const Val32,
        in_len: c_int,
    ) -> Val32;
    fn oracle_downmix_and_resample(
        pcm_type: c_int,
        x: *const c_void,
        y: *mut Val32,
        s: *mut Val32,
        subframe: c_int,
        offset: c_int,
        c1: c_int,
        c2: c_int,
        c: c_int,
        fs: c_int,
    ) -> Val32;
    fn oracle_tonality_analysis(
        st: *mut c_void,
        pcm_type: c_int,
        x: *const c_void,
        len: c_int,
        offset: c_int,
        c1: c_int,
        c2: c_int,
        c: c_int,
        lsb_depth: c_int,
    );
}

/// Interleaved PCM input for the analysis, selecting the C downmix callback
/// (`downmix_float`, `downmix_int`, `downmix_int24`).
#[derive(Debug, Clone, Copy)]
pub enum Pcm<'a> {
    /// Float PCM (`downmix_float`).
    Float(&'a [f32]),
    /// 16-bit PCM (`downmix_int`).
    Int16(&'a [i16]),
    /// 24-bit PCM in `i32` (`downmix_int24`).
    Int24(&'a [i32]),
}

impl Pcm<'_> {
    fn raw(self) -> (c_int, *const c_void, usize) {
        match self {
            Self::Float(x) => (0, x.as_ptr().cast(), x.len()),
            Self::Int16(x) => (1, x.as_ptr().cast(), x.len()),
            Self::Int24(x) => (2, x.as_ptr().cast(), x.len()),
        }
    }
}

/// Highest sample index (exclusive, in interleaved samples) a downmix of `subframe` samples at
/// `offset` touches.
const fn downmix_extent(subframe: i32, offset: i32, c: i32) -> usize {
    ((subframe + offset) * c) as usize
}

/// Checks the `#[repr(C)]` mirrors against the C layout.
#[must_use]
pub fn layout_matches() -> bool {
    use core::mem::{offset_of, size_of};
    let expected = [
        offset_of!(CTonalityAnalysisState, angle),
        offset_of!(CTonalityAnalysisState, inmem),
        offset_of!(CTonalityAnalysisState, e),
        offset_of!(CTonalityAnalysisState, e_tracker),
        offset_of!(CTonalityAnalysisState, hp_ener_accum),
        offset_of!(CTonalityAnalysisState, rnn_state),
        offset_of!(CTonalityAnalysisState, downmix_state),
        offset_of!(CTonalityAnalysisState, info),
        offset_of!(CAnalysisInfo, leak_boost),
        offset_of!(CAnalysisInfo, max_pitch_ratio),
    ];
    // SAFETY: pure functions returning sizes/offsets.
    unsafe {
        oracle_analysis_state_size() == size_of::<CTonalityAnalysisState>()
            && oracle_analysis_info_size() == size_of::<CAnalysisInfo>()
            && expected
                .iter()
                .enumerate()
                .all(|(i, &off)| oracle_analysis_state_offset(i as c_int) == off)
    }
}

/// The value of `M_PI` seen by `analysis.c`.
#[must_use]
pub fn m_pi() -> f64 {
    // SAFETY: pure function.
    unsafe { oracle_analysis_m_pi() }
}

/// C `tonality_analysis_init`.
pub fn tonality_analysis_init(st: &mut CTonalityAnalysisState, fs: i32) {
    // SAFETY: `st` is a valid, exclusively borrowed state with the C layout.
    unsafe { oracle_tonality_analysis_init(st.as_mut_ptr(), fs) }
}

/// C `tonality_analysis_reset`.
pub fn tonality_analysis_reset(st: &mut CTonalityAnalysisState) {
    // SAFETY: `st` is a valid, exclusively borrowed state with the C layout.
    unsafe { oracle_tonality_analysis_reset(st.as_mut_ptr()) }
}

/// C `tonality_get_info`.
pub fn tonality_get_info(st: &mut CTonalityAnalysisState, len: i32) -> CAnalysisInfo {
    let mut info = CAnalysisInfo::default();
    // SAFETY: valid state and output struct with the C layouts.
    unsafe {
        oracle_tonality_get_info(st.as_mut_ptr(), core::ptr::from_mut(&mut info).cast(), len);
    }
    info
}

/// C `run_analysis` with the 48 kHz static CELT mode. `pcm` of `None` passes `NULL`.
#[allow(clippy::too_many_arguments, reason = "mirrors C signature")]
pub fn run_analysis(
    st: &mut CTonalityAnalysisState,
    pcm: Option<Pcm<'_>>,
    analysis_frame_size: i32,
    frame_size: i32,
    c1: i32,
    c2: i32,
    c: i32,
    fs: i32,
    lsb_depth: i32,
) -> CAnalysisInfo {
    let mut info = CAnalysisInfo::default();
    let (ty, ptr) = match pcm {
        Some(p) => {
            let (ty, ptr, len) = p.raw();
            assert!(
                len >= downmix_extent(analysis_frame_size, 0, c),
                "pcm too short for analysis_frame_size"
            );
            (ty, ptr)
        }
        None => (0, core::ptr::null()),
    };
    // SAFETY: valid state/output; `pcm` holds at least `analysis_frame_size * c` samples
    // (checked above), which bounds every read of `run_analysis`.
    unsafe {
        oracle_run_analysis(
            st.as_mut_ptr(),
            ty,
            ptr,
            analysis_frame_size,
            frame_size,
            c1,
            c2,
            c,
            fs,
            lsb_depth,
            core::ptr::from_mut(&mut info).cast(),
        );
    }
    info
}

/// C `downmix_float` / `downmix_int` / `downmix_int24` (by `pcm` type) into `y[..subframe]`.
#[allow(clippy::too_many_arguments, reason = "mirrors C signature")]
pub fn downmix(
    pcm: Pcm<'_>,
    y: &mut [Val32],
    subframe: i32,
    offset: i32,
    c1: i32,
    c2: i32,
    c: i32,
) {
    let (ty, ptr, len) = pcm.raw();
    assert!(len >= downmix_extent(subframe, offset, c));
    assert!(y.len() >= subframe as usize);
    // SAFETY: bounds checked above.
    unsafe { oracle_downmix(ty, ptr, y.as_mut_ptr(), subframe, offset, c1, c2, c) }
}

/// C `is_digital_silence` (on `opus_res` samples).
#[must_use]
pub fn is_digital_silence(pcm: &[Res], frame_size: i32, channels: i32, lsb_depth: i32) -> bool {
    assert!(pcm.len() >= (frame_size * channels) as usize);
    // SAFETY: bounds checked above.
    unsafe { oracle_is_digital_silence(pcm.as_ptr(), frame_size, channels, lsb_depth) != 0 }
}

/// C `silk_resampler_down2_hp` (static in analysis.c).
pub fn silk_resampler_down2_hp(s: &mut [Val32; 3], out: &mut [Val32], input: &[Val32]) -> Val32 {
    let in_len = input.len() as c_int;
    assert!(out.len() >= input.len() / 2);
    // SAFETY: bounds checked above; `s` holds the 3 state values.
    unsafe {
        oracle_silk_resampler_down2_hp(s.as_mut_ptr(), out.as_mut_ptr(), input.as_ptr(), in_len)
    }
}

/// C `downmix_and_resample` (static in analysis.c). `y` must hold the resampled output.
#[allow(clippy::too_many_arguments, reason = "mirrors C signature")]
pub fn downmix_and_resample(
    pcm: Pcm<'_>,
    y: &mut [Val32],
    s: &mut [Val32; 3],
    subframe: i32,
    offset: i32,
    c1: i32,
    c2: i32,
    c: i32,
    fs: i32,
) -> Val32 {
    let (ty, ptr, len) = pcm.raw();
    let (sub_in, off_in) = match fs {
        48000 => (subframe * 2, offset * 2),
        16000 => (subframe * 2 / 3, offset * 2 / 3),
        _ => (subframe, offset),
    };
    assert!(subframe == 0 || len >= downmix_extent(sub_in, off_in, c));
    assert!(y.len() >= subframe as usize);
    // SAFETY: bounds checked above.
    unsafe {
        oracle_downmix_and_resample(
            ty,
            ptr,
            y.as_mut_ptr(),
            s.as_mut_ptr(),
            subframe,
            offset,
            c1,
            c2,
            c,
            fs,
        )
    }
}

/// C `tonality_analysis` (static in analysis.c) with the 48 kHz static CELT mode.
#[allow(clippy::too_many_arguments, reason = "mirrors C signature")]
pub fn tonality_analysis(
    st: &mut CTonalityAnalysisState,
    pcm: Pcm<'_>,
    len: i32,
    offset: i32,
    c1: i32,
    c2: i32,
    c: i32,
    lsb_depth: i32,
) {
    let (ty, ptr, n) = pcm.raw();
    assert!(n >= downmix_extent(len, offset, c));
    // SAFETY: valid state; the pcm holds `len` samples past `offset` (checked above).
    unsafe { oracle_tonality_analysis(st.as_mut_ptr(), ty, ptr, len, offset, c1, c2, c, lsb_depth) }
}
