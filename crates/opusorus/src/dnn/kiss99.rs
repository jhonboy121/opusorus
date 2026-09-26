//! Port of `dnn/kiss99.c` / `dnn/kiss99.h`: the KISS99 PRNG (George Marsaglia, 1999 version with
//! the 13/17 shifts swapped as in KISS11).

/// `kiss99_ctx`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Kiss99Ctx {
    pub z: u32,
    pub w: u32,
    pub jsr: u32,
    pub jcong: u32,
}

impl Kiss99Ctx {
    /// Port of dnn/kiss99.c:kiss99_srand.
    pub fn kiss99_srand(&mut self, data: &[u8]) {
        let ndata = data.len();
        self.z = 362436069;
        self.w = 521288629;
        self.jsr = 123456789;
        self.jcong = 380116160;
        let mut i = 3;
        while i < ndata {
            self.z ^= u32::from(data[i - 3]);
            self.w ^= u32::from(data[i - 2]);
            self.jsr ^= u32::from(data[i - 1]);
            self.jcong ^= u32::from(data[i]);
            self.kiss99_rand();
            i += 4;
        }
        if i - 3 < ndata {
            self.z ^= u32::from(data[i - 3]);
        }
        if i - 2 < ndata {
            self.w ^= u32::from(data[i - 2]);
        }
        if i - 1 < ndata {
            self.jsr ^= u32::from(data[i - 1]);
        }
        // Fix any potential short cycles that show up (see Gregory G. Rose: "KISS: A Bit Too
        // Simple", Cryptographic Communications No. 10, pp. 123---137, 2018).
        if self.z == 0 || self.z == 0x9068_FFFF {
            self.z += 1;
        }
        if self.w == 0 || self.w == 0x464F_FFFF {
            self.w += 1;
        }
        if self.jsr == 0 {
            self.jsr += 1;
        }
    }

    /// Port of dnn/kiss99.c:kiss99_rand.
    pub const fn kiss99_rand(&mut self) -> u32 {
        let znew = 36969u32
            .wrapping_mul(self.z & 0xFFFF)
            .wrapping_add(self.z >> 16);
        let wnew = 18000u32
            .wrapping_mul(self.w & 0xFFFF)
            .wrapping_add(self.w >> 16);
        let mwc = (znew << 16).wrapping_add(wnew);
        let mut shr3 = self.jsr ^ (self.jsr << 13);
        shr3 ^= shr3 >> 17;
        shr3 ^= shr3 << 5;
        let cong = 69069u32.wrapping_mul(self.jcong).wrapping_add(1234567);
        self.z = znew;
        self.w = wnew;
        self.jsr = shr3;
        self.jcong = cong;
        (mwc ^ cong).wrapping_add(shr3)
    }
}

/// Port of dnn/kiss99.c:kiss99_srand (free-function form).
pub fn kiss99_srand(ctx: &mut Kiss99Ctx, data: &[u8]) {
    ctx.kiss99_srand(data);
}

/// Port of dnn/kiss99.c:kiss99_rand (free-function form).
pub const fn kiss99_rand(ctx: &mut Kiss99Ctx) -> u32 {
    ctx.kiss99_rand()
}
