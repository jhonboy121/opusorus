//! Port of `celt/entdec.c`: the range decoder.

use crate::celt::entcode::{
    EC_CODE_BITS, EC_CODE_BOT, EC_CODE_EXTRA, EC_CODE_TOP, EC_SYM_BITS, EC_SYM_MAX, EC_UINT_BITS,
    EC_WINDOW_SIZE, EcState, EcWindow, celt_udiv, ec_ilog,
};

/// Range decoder (`ec_dec`). Reads range-coded data from the front of the buffer and raw bits
/// from the back.
#[derive(Debug, Clone)]
pub struct EcDec<'a> {
    /// Input buffer (`buf`).
    pub buf: &'a [u8],
    /// The size of the buffer in use.
    pub storage: u32,
    /// The offset at which the last byte containing raw bits was read.
    pub end_offs: u32,
    /// Bits that will be read from the end.
    pub end_window: EcWindow,
    /// Number of valid bits in `end_window`.
    pub nend_bits: i32,
    /// The total number of whole bits read.
    pub nbits_total: i32,
    /// The offset at which the next range coder byte will be read.
    pub offs: u32,
    /// The number of values in the current range.
    pub rng: u32,
    /// The difference between the top of the current range and the input value, minus one.
    pub val: u32,
    /// The saved normalization factor from `ec_decode()`.
    pub ext: u32,
    /// A buffered input symbol.
    pub rem: i32,
    /// Nonzero if an error occurred.
    pub error: i32,
}

impl EcState for EcDec<'_> {
    #[inline(always)]
    fn nbits_total(&self) -> i32 {
        self.nbits_total
    }
    #[inline(always)]
    fn rng(&self) -> u32 {
        self.rng
    }
}

impl<'a> EcDec<'a> {
    /// Port of `ec_dec_init`, using the whole buffer as storage.
    #[must_use]
    pub fn new(buf: &'a [u8]) -> Self {
        let size = buf.len().min(u32::MAX as usize) as u32;
        Self::with_size(buf, size)
    }

    /// Port of `ec_dec_init` with explicit storage (`storage <= buf.len()`).
    #[must_use]
    pub fn with_size(buf: &'a [u8], storage: u32) -> Self {
        debug_assert!(storage as usize <= buf.len());
        let mut d = Self {
            buf,
            storage,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            // This is the offset from which ec_tell() will subtract partial bits. The final value
            // after the normalize() call will be the same as in the encoder, but we have to
            // compensate for the bits that are added there.
            nbits_total: EC_CODE_BITS + 1
                - ((EC_CODE_BITS - EC_CODE_EXTRA) / EC_SYM_BITS) * EC_SYM_BITS,
            offs: 0,
            rng: 1u32 << EC_CODE_EXTRA,
            val: 0,
            ext: 0,
            rem: 0,
            error: 0,
        };
        d.rem = d.read_byte();
        d.val = d.rng - 1 - ((d.rem as u32) >> (EC_SYM_BITS - EC_CODE_EXTRA));
        // Normalize the interval.
        d.normalize();
        d
    }

    /// `ec_range_bytes`.
    #[inline(always)]
    #[must_use]
    pub const fn range_bytes(&self) -> u32 {
        self.offs
    }

    /// `ec_get_error`.
    #[inline(always)]
    #[must_use]
    pub const fn get_error(&self) -> i32 {
        self.error
    }

    /// `ec_tell`.
    #[inline(always)]
    #[must_use]
    pub fn tell(&self) -> i32 {
        EcState::tell(self)
    }

    /// `ec_tell_frac`.
    #[inline(always)]
    #[must_use]
    pub fn tell_frac(&self) -> u32 {
        EcState::tell_frac(self)
    }

    #[inline(always)]
    const fn read_byte(&mut self) -> i32 {
        if self.offs < self.storage {
            let b = self.buf[self.offs as usize];
            self.offs += 1;
            b as i32
        } else {
            0
        }
    }

    #[inline(always)]
    const fn read_byte_from_end(&mut self) -> i32 {
        if self.end_offs < self.storage {
            self.end_offs += 1;
            self.buf[(self.storage - self.end_offs) as usize] as i32
        } else {
            0
        }
    }

    /// Normalizes the contents of `val` and `rng` so that `rng` lies entirely in the high-order
    /// symbol.
    #[inline(always)]
    const fn normalize(&mut self) {
        // If the range is too small, rescale it and input some bits.
        while self.rng <= EC_CODE_BOT {
            self.nbits_total += EC_SYM_BITS;
            self.rng <<= EC_SYM_BITS;
            // Use up the remaining bits from our last symbol.
            let mut sym = self.rem;
            // Read the next value from the input.
            self.rem = self.read_byte();
            // Take the rest of the bits we need from this new symbol.
            sym = ((sym << EC_SYM_BITS) | self.rem) >> (EC_SYM_BITS - EC_CODE_EXTRA);
            // And subtract them from val, capped to be less than EC_CODE_TOP.
            self.val = ((self.val << EC_SYM_BITS).wrapping_add(EC_SYM_MAX & !(sym as u32)))
                & (EC_CODE_TOP - 1);
        }
    }

    /// Port of `ec_decode`: calculates the cumulative frequency for the next symbol. Must be
    /// followed by [`Self::update`].
    #[inline]
    pub fn decode(&mut self, ft: u32) -> u32 {
        self.ext = celt_udiv(self.rng, ft);
        let s = self.val / self.ext;
        ft - (s + 1).min(ft)
    }

    /// Port of `ec_decode_bin`: equivalent to `decode` with `ft == 1 << bits`.
    #[inline]
    pub fn decode_bin(&mut self, bits: u32) -> u32 {
        self.ext = self.rng >> bits;
        let s = self.val / self.ext;
        (1u32 << bits) - (s + 1).min(1u32 << bits)
    }

    /// Port of `ec_dec_update`: advances the decoder past the symbol `[fl, fh)` of total `ft`.
    #[inline]
    pub const fn update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.ext.wrapping_mul(ft - fh);
        self.val = self.val.wrapping_sub(s);
        self.rng = if fl > 0 {
            self.ext.wrapping_mul(fh - fl)
        } else {
            self.rng.wrapping_sub(s)
        };
        self.normalize();
    }

    /// Port of `ec_dec_bit_logp`: decodes a bit that has a `1/(1<<logp)` probability of being one.
    #[inline]
    pub const fn dec_bit_logp(&mut self, logp: u32) -> bool {
        let r = self.rng;
        let d = self.val;
        let s = r >> logp;
        let ret = d < s;
        if !ret {
            self.val = d - s;
        }
        self.rng = if ret { s } else { r - s };
        self.normalize();
        ret
    }

    /// Port of `ec_dec_icdf`: decodes a symbol given an inverse CDF table (8-bit entries).
    #[inline]
    pub const fn dec_icdf(&mut self, icdf: &[u8], ftb: u32) -> usize {
        let mut s = self.rng;
        let d = self.val;
        let r = s >> ftb;
        let mut ret: usize = 0;
        let mut t;
        loop {
            t = s;
            s = r.wrapping_mul(icdf[ret] as u32);
            if d >= s {
                break;
            }
            ret += 1;
        }
        self.val = d - s;
        self.rng = t - s;
        self.normalize();
        ret
    }

    /// Port of `ec_dec_icdf16`: like [`Self::dec_icdf`] with 16-bit entries.
    // Size: not inlined (tens of call sites; the call is cheap next to the symbol search).
    #[inline(never)]
    pub const fn dec_icdf16(&mut self, icdf: &[u16], ftb: u32) -> usize {
        let mut s = self.rng;
        let d = self.val;
        let r = s >> ftb;
        let mut ret: usize = 0;
        let mut t;
        loop {
            t = s;
            s = r.wrapping_mul(icdf[ret] as u32);
            if d >= s {
                break;
            }
            ret += 1;
        }
        self.val = d - s;
        self.rng = t - s;
        self.normalize();
        ret
    }

    /// Port of `ec_dec_uint`: extracts a raw unsigned integer in `[0, ft)`; `ft > 1`.
    /// Sets the error flag if the decoded value is out of range.
    pub fn dec_uint(&mut self, ft: u32) -> u32 {
        // In order to optimize EC_ILOG(), it is undefined for the value 0.
        celt_assert!(ft > 1);
        let ft = ft - 1;
        let mut ftb = ec_ilog(ft);
        if ftb > EC_UINT_BITS {
            ftb -= EC_UINT_BITS;
            let ft1 = (ft >> ftb) + 1;
            let s = self.decode(ft1);
            self.update(s, s + 1, ft1);
            let t = (s << ftb) | self.dec_bits(ftb as u32);
            if t <= ft {
                return t;
            }
            self.error = 1;
            ft
        } else {
            let ft = ft + 1;
            let s = self.decode(ft);
            self.update(s, s + 1, ft);
            s
        }
    }

    /// Port of `ec_dec_bits`: extracts `bits` raw bits (`0 <= bits <= 25`) from the end.
    pub const fn dec_bits(&mut self, bits: u32) -> u32 {
        let mut window = self.end_window;
        let mut available = self.nend_bits;
        if (available as u32) < bits {
            loop {
                window |= (self.read_byte_from_end() as EcWindow) << available;
                available += EC_SYM_BITS;
                if available > EC_WINDOW_SIZE - EC_SYM_BITS {
                    break;
                }
            }
        }
        let ret = window & ((1u32 << bits).wrapping_sub(1));
        window = if bits >= 32 { 0 } else { window >> bits };
        available -= bits as i32;
        self.end_window = window;
        self.nend_bits = available;
        self.nbits_total += bits as i32;
        ret
    }
}
