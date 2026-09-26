//! Port of silk/resampler*.c, silk/resampler_rom.c, silk/resampler_*.h.
//!
//! Status: pending (unit `silk_resampler`).

/// `silk_resampler_state_struct`.
///
/// Placeholder so dependent units compile; the `silk_resampler` unit replaces the body with the
/// real fields. The type name and the `Debug + Clone + Default` derives are part of the contract.
#[derive(Debug, Clone, Default)]
pub struct SilkResamplerState {}
