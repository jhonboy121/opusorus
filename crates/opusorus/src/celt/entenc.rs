//! Port of `celt/entenc.c`: the range encoder.

use crate::celt::entcode::{
    EC_CODE_BITS, EC_CODE_BOT, EC_CODE_SHIFT, EC_CODE_TOP, EC_SYM_BITS, EC_SYM_MAX, EC_UINT_BITS,
    EC_WINDOW_SIZE, EcState, EcWindow, celt_udiv, ec_ilog,
};

/// Range encoder (`ec_enc`). Writes range-coded data from the front of `buf` and raw bits from
/// the back.
#[derive(Debug)]
pub struct EcEnc<'a> {
    /// Output buffer (`buf`); `storage` bytes of it are used.
    pub buf: &'a mut [u8],
    /// The size of the buffer in use.
    pub storage: u32,
    /// The offset at which the last byte containing raw bits was written.
    pub end_offs: u32,
    /// Bits that will be written at the end.
    pub end_window: EcWindow,
    /// Number of valid bits in `end_window`.
    pub nend_bits: i32,
    /// The total number of whole bits written.
    pub nbits_total: i32,
    /// The offset at which the next range coder byte will be written.
    pub offs: u32,
    /// The number of values in the current range.
    pub rng: u32,
    /// The low end of the current range.
    pub val: u32,
    /// The number of outstanding carry propagating symbols.
    pub ext: u32,
    /// A buffered output symbol, awaiting carry propagation.
    pub rem: i32,
    /// Nonzero if an error occurred.
    pub error: i32,
}

/// Snapshot of the numeric encoder state (C code copies `ec_enc` by value to roll back).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EcEncSnapshot {
    storage: u32,
    end_offs: u32,
    end_window: EcWindow,
    nend_bits: i32,
    nbits_total: i32,
    offs: u32,
    rng: u32,
    val: u32,
    ext: u32,
    rem: i32,
    error: i32,
}

impl EcState for EcEnc<'_> {
    #[inline(always)]
    fn nbits_total(&self) -> i32 {
        self.nbits_total
    }
    #[inline(always)]
    fn rng(&self) -> u32 {
        self.rng
    }
}

impl<'a> EcEnc<'a> {
    /// Port of `ec_enc_init`. The whole of `buf` is used as storage.
    #[must_use]
    pub fn new(buf: &'a mut [u8]) -> Self {
        let size = buf.len().min(u32::MAX as usize) as u32;
        Self::with_size(buf, size)
    }

    /// Port of `ec_enc_init` with an explicit storage size (`size <= buf.len()`).
    #[must_use]
    pub const fn with_size(buf: &'a mut [u8], size: u32) -> Self {
        debug_assert!(size as usize <= buf.len());
        Self {
            buf,
            storage: size,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            // This is the offset from which ec_tell() will subtract partial bits.
            nbits_total: EC_CODE_BITS + 1,
            offs: 0,
            rng: EC_CODE_TOP,
            rem: -1,
            val: 0,
            ext: 0,
            error: 0,
        }
    }

    /// Saves the numeric state (the C code does `ec_save = *enc`).
    #[must_use]
    pub const fn snapshot(&self) -> EcEncSnapshot {
        EcEncSnapshot {
            storage: self.storage,
            end_offs: self.end_offs,
            end_window: self.end_window,
            nend_bits: self.nend_bits,
            nbits_total: self.nbits_total,
            offs: self.offs,
            rng: self.rng,
            val: self.val,
            ext: self.ext,
            rem: self.rem,
            error: self.error,
        }
    }

    /// Restores a snapshot taken with [`Self::snapshot`] (the C code does `*enc = ec_save`).
    pub const fn restore(&mut self, s: &EcEncSnapshot) {
        self.storage = s.storage;
        self.end_offs = s.end_offs;
        self.end_window = s.end_window;
        self.nend_bits = s.nend_bits;
        self.nbits_total = s.nbits_total;
        self.offs = s.offs;
        self.rng = s.rng;
        self.val = s.val;
        self.ext = s.ext;
        self.rem = s.rem;
        self.error = s.error;
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

    /// `ec_get_buffer`.
    #[inline(always)]
    #[must_use]
    pub const fn buffer(&self) -> &[u8] {
        self.buf
    }

    /// `ec_get_buffer` (mutable).
    #[inline(always)]
    pub const fn buffer_mut(&mut self) -> &mut [u8] {
        self.buf
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
    const fn write_byte(&mut self, value: u32) -> i32 {
        if self.offs + self.end_offs >= self.storage {
            return -1;
        }
        self.buf[self.offs as usize] = value as u8;
        self.offs += 1;
        0
    }

    #[inline(always)]
    const fn write_byte_at_end(&mut self, value: u32) -> i32 {
        if self.offs + self.end_offs >= self.storage {
            return -1;
        }
        self.end_offs += 1;
        self.buf[(self.storage - self.end_offs) as usize] = value as u8;
        0
    }

    /// Outputs a symbol, with a carry bit. If there is a potential to propagate a carry over
    /// several symbols, they are buffered until it can be determined whether or not an actual
    /// carry will occur.
    const fn carry_out(&mut self, c: i32) {
        if c as u32 != EC_SYM_MAX {
            // No further carry propagation possible, flush buffer.
            let carry = c >> EC_SYM_BITS;
            // Don't output a byte on the first write.
            if self.rem >= 0 {
                self.error |= self.write_byte((self.rem + carry) as u32);
            }
            if self.ext > 0 {
                let sym = (EC_SYM_MAX.wrapping_add(carry as u32)) & EC_SYM_MAX;
                loop {
                    self.error |= self.write_byte(sym);
                    self.ext -= 1;
                    if self.ext == 0 {
                        break;
                    }
                }
            }
            self.rem = (c as u32 & EC_SYM_MAX) as i32;
        } else {
            self.ext += 1;
        }
    }

    #[inline(always)]
    const fn normalize(&mut self) {
        // If the range is too small, output some bits and rescale it.
        while self.rng <= EC_CODE_BOT {
            self.carry_out((self.val >> EC_CODE_SHIFT) as i32);
            // Move the next-to-high-order symbol into the high-order position.
            self.val = (self.val << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            self.rng <<= EC_SYM_BITS;
            self.nbits_total += EC_SYM_BITS;
        }
    }

    /// Port of `ec_encode`: encodes a symbol given its frequency information `[fl, fh)` out of
    /// total `ft`.
    #[inline]
    pub const fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        let r = celt_udiv(self.rng, ft);
        if fl > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(ft - fl)));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(ft - fh));
        }
        self.normalize();
    }

    /// Port of `ec_encode_bin`: equivalent to `encode` with `ft == 1 << bits`.
    #[inline]
    pub const fn encode_bin(&mut self, fl: u32, fh: u32, bits: u32) {
        let r = self.rng >> bits;
        if fl > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul((1u32 << bits) - fl)));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul((1u32 << bits) - fh));
        }
        self.normalize();
    }

    /// Port of `ec_enc_bit_logp`: the probability of `val` being true is `1/(1<<logp)`.
    #[inline]
    pub const fn enc_bit_logp(&mut self, val: bool, logp: u32) {
        let mut r = self.rng;
        let l = self.val;
        let s = r >> logp;
        r -= s;
        if val {
            self.val = l.wrapping_add(r);
        }
        self.rng = if val { s } else { r };
        self.normalize();
    }

    /// Port of `ec_enc_icdf`: encodes symbol `s` with an inverse CDF table (8-bit entries).
    // Size: not inlined (tens of call sites; the call is cheap next to the symbol search).
    #[inline(never)]
    pub const fn enc_icdf(&mut self, s: usize, icdf: &[u8], ftb: u32) {
        let r = self.rng >> ftb;
        if s > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(icdf[s - 1] as u32)));
            self.rng = r.wrapping_mul(icdf[s - 1] as u32 - icdf[s] as u32);
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(icdf[s] as u32));
        }
        self.normalize();
    }

    /// Port of `ec_enc_icdf16`: like [`Self::enc_icdf`] with 16-bit entries.
    // Size: not inlined (tens of call sites; the call is cheap next to the symbol search).
    #[inline(never)]
    pub const fn enc_icdf16(&mut self, s: usize, icdf: &[u16], ftb: u32) {
        let r = self.rng >> ftb;
        if s > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(icdf[s - 1] as u32)));
            self.rng = r.wrapping_mul(icdf[s - 1] as u32 - icdf[s] as u32);
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(icdf[s] as u32));
        }
        self.normalize();
    }

    /// Port of `ec_enc_uint`: encodes a raw unsigned integer `fl` in `[0, ft)`; `ft > 1`.
    pub fn enc_uint(&mut self, fl: u32, ft: u32) {
        // In order to optimize EC_ILOG(), it is undefined for the value 0.
        debug_assert!(ft > 1);
        let ft = ft - 1;
        let mut ftb = ec_ilog(ft);
        if ftb > EC_UINT_BITS {
            ftb -= EC_UINT_BITS;
            let ft1 = (ft >> ftb) + 1;
            let fl1 = fl >> ftb;
            self.encode(fl1, fl1 + 1, ft1);
            self.enc_bits(fl & ((1u32 << ftb) - 1), ftb as u32);
        } else {
            self.encode(fl, fl + 1, ft + 1);
        }
    }

    /// Port of `ec_enc_bits`: writes `bits` raw bits (`0 < bits <= 25`) to the end of the stream.
    pub fn enc_bits(&mut self, fl: u32, bits: u32) {
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        debug_assert!(bits > 0);
        if used + bits as i32 > EC_WINDOW_SIZE {
            loop {
                self.error |= self.write_byte_at_end(window & EC_SYM_MAX);
                window >>= EC_SYM_BITS;
                used -= EC_SYM_BITS;
                if used < EC_SYM_BITS {
                    break;
                }
            }
        }
        window |= fl << used;
        used += bits as i32;
        self.end_window = window;
        self.nend_bits = used;
        self.nbits_total += bits as i32;
    }

    /// Port of `ec_enc_patch_initial_bits`: overwrites a few bits at the very start of an
    /// existing stream.
    pub fn patch_initial_bits(&mut self, val: u32, nbits: u32) {
        debug_assert!(nbits <= EC_SYM_BITS as u32);
        let shift = EC_SYM_BITS as u32 - nbits;
        let mask: u32 = ((1u32 << nbits) - 1) << shift;
        if self.offs > 0 {
            // The first byte has been finalized.
            self.buf[0] = ((self.buf[0] as u32 & !mask) | (val << shift)) as u8;
        } else if self.rem >= 0 {
            // The first byte is still awaiting carry propagation.
            self.rem = ((self.rem as u32 & !mask) | (val << shift)) as i32;
        } else if self.rng <= (EC_CODE_TOP >> nbits) {
            // The renormalization loop has never been run.
            self.val =
                (self.val & !(mask << EC_CODE_SHIFT)) | (val << (EC_CODE_SHIFT as u32 + shift));
        } else {
            // The encoder hasn't even encoded nbits of data yet.
            self.error = -1;
        }
    }

    /// Port of `ec_enc_shrink`: compacts the data to fit in the target size.
    pub fn shrink(&mut self, size: u32) {
        debug_assert!(self.offs + self.end_offs <= size);
        let src = (self.storage - self.end_offs) as usize;
        let dst = (size - self.end_offs) as usize;
        self.buf.copy_within(src..src + self.end_offs as usize, dst);
        self.storage = size;
    }

    /// Port of `ec_enc_done`: flushes remaining bits. After this call the buffer contains the
    /// complete stream; check [`Self::get_error`].
    pub fn done(&mut self) {
        // We output the minimum number of bits that ensures that the symbols encoded thus far
        // will be decoded correctly regardless of the bits that follow.
        let mut l = EC_CODE_BITS - ec_ilog(self.rng);
        let mut msk: u32 = (EC_CODE_TOP - 1) >> l;
        let mut end: u32 = self.val.wrapping_add(msk) & !msk;
        if (end | msk) >= self.val.wrapping_add(self.rng) {
            l += 1;
            msk >>= 1;
            end = self.val.wrapping_add(msk) & !msk;
        }
        while l > 0 {
            self.carry_out((end >> EC_CODE_SHIFT) as i32);
            end = (end << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            l -= EC_SYM_BITS;
        }
        // If we have a buffered byte flush it into the output buffer.
        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }
        // If we have buffered extra bits, flush them as well.
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        while used >= EC_SYM_BITS {
            self.error |= self.write_byte_at_end(window & EC_SYM_MAX);
            window >>= EC_SYM_BITS;
            used -= EC_SYM_BITS;
        }
        // Clear any excess space and add any remaining extra bits to the last byte.
        if self.error == 0 {
            let start = self.offs as usize;
            let stop = (self.storage - self.end_offs) as usize;
            if start < stop {
                self.buf[start..stop].fill(0);
            }
            if used > 0 {
                // If there's no range coder data at all, give up.
                if self.end_offs >= self.storage {
                    self.error = -1;
                } else {
                    l = -l;
                    // If we've busted, don't add too many extra bits to the last byte; it would
                    // corrupt the range coder data, and that's more important.
                    if self.offs + self.end_offs >= self.storage && l < used {
                        window &= (1u32 << l) - 1;
                        self.error = -1;
                    }
                    let idx = (self.storage - self.end_offs - 1) as usize;
                    self.buf[idx] |= window as u8;
                }
            }
        }
    }
}
