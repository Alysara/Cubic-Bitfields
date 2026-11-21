use core::arch::x86_64::*;
use std::arch::x86_64::_pdep_u64;
use std::iter::zip;
use std::marker::PhantomData;
use std::mem::transmute;
use std::ops::*;
use std::simd::prelude::*;

use simply_simd::{Arch, Mask, Simd, StaticArch};

use crate::loader_new::*;
use crate::logger::*;
use crate::transposes::*;

const EVEN_U8_SWIZ: [usize; 32] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48,
    50, 52, 54, 56, 58, 60, 62,
];

const ODD_U8_SWIZ: [usize; 32] = [
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49,
    51, 53, 55, 57, 59, 61, 63,
];

const EVEN_U16_SWIZ: [usize; 32] = [
    0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48,
    49, 52, 53, 56, 57, 60, 61,
];

const ODD_U16_SWIZ: [usize; 32] = [
    2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50,
    51, 54, 55, 58, 59, 62, 63,
];

#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Bitfield32<A: Arch = StaticArch> {
    data: [u32; 1024],
    _marker: PhantomData<A>,
}

impl<A: Arch> PartialEq for Bitfield32<A> {
    fn eq(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).all(|(x, y)| x == y)
    }
}

// Bit operations.

impl<A: Arch> BitOr for Bitfield32<A> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        let mut result = Self::new(0);
        result |= rhs;
        result
    }
}

impl<A: Arch> BitAnd for Bitfield32<A> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        let mut result = Self::new(0);
        result &= rhs;
        result
    }
}

impl<A: Arch> BitXor for Bitfield32<A> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut result = Self::new(0);
        result ^= rhs;
        result
    }
}

impl<A: Arch> BitOrAssign for Bitfield32<A> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        for i in 0..1024 {
            self.data[i] |= rhs.data[i];
        }
    }
}

impl<A: Arch> BitAndAssign for Bitfield32<A> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        for i in 0..1024 {
            self.data[i] &= rhs.data[i];
        }
    }
}

impl<A: Arch> BitXorAssign for Bitfield32<A> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        for i in 0..1024 {
            self.data[i] ^= rhs.data[i];
        }
    }
}

impl<A: Arch> Shl<usize> for Bitfield32<A> {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: usize) -> Self::Output {
        let mut new_bitfield = self;
        for i in 0..1024 {
            new_bitfield.data[i] <<= rhs;
        }
        new_bitfield
    }
}

impl<A: Arch> Shr<usize> for Bitfield32<A> {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: usize) -> Self::Output {
        let mut new_bitfield = self;
        for i in 0..1024 {
            new_bitfield.data[i] >>= rhs;
        }
        new_bitfield
    }
}

impl<A: Arch> ShlAssign<usize> for Bitfield32<A> {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: usize) {
        for i in 0..1024 {
            self.data[i] <<= rhs;
        }
    }
}

impl<A: Arch> ShrAssign<usize> for Bitfield32<A> {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: usize) {
        for i in 0..1024 {
            self.data[i] >>= rhs;
        }
    }
}

impl<A: Arch> Bitfield32<A> {
    #[inline(always)]
    pub fn new(val: u32) -> Self {
        Self {
            data: std::array::from_fn(|_| val),
            _marker: PhantomData::<A>,
        }
    }

    #[inline(always)]
    pub fn fill(&mut self, val: u32) {
        for i in 0..1024 {
            self.data[i] = val;
        }
    }

    #[inline(always)]
    fn set_u32<const SET: u8>(&mut self, index: usize, value: u32) {
        apply_set::<SET, u32>(&mut self.data[index], value);
    }

    #[inline(always)]
    fn set_u64<const SET: u8>(&mut self, index: usize, value: u64) {
        let bitfield_64: &mut [u64; 1024] = unsafe { transmute(&mut self.data) };
        apply_set::<SET, u64>(&mut bitfield_64[index], value);
    }

    #[inline(always)]
    pub fn count_ones(&self) -> u32 {
        self.data.iter().map(|x| x.count_ones()).sum()
    }

    #[inline(always)]
    pub fn andnot(&self, rhs: &Self) -> Self {
        let mut new_bitfield = *self;
        for i in 0..1024 {
            new_bitfield.data[i] &= !rhs.data[i];
        }
        new_bitfield
    }

    #[inline(always)]
    pub fn andnot_assign(&mut self, rhs: &Self) -> &mut Self {
        for i in 0..1024 {
            self.data[i] &= !rhs.data[i];
        }
        self
    }

    // Core load-into implementations.

    // u1 loaders.
    #[inline(always)]
    fn load_packed_u1_inner<'a, const SET: u8, const CMP: u8, const MATCH: bool>(
        &mut self,
        src: impl Iterator<Item = (usize, &'a u32)>,
    ) {
        for (i, val) in src {
            self.set_u32::<SET>(i, process_packed_u1::<CMP, MATCH>(val));
        }
    }

    #[inline(always)]
    pub fn load_packed_u1_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 512],
        match_val: bool,
    ) {
        let data_32: &[u32; 1024] = unsafe { transmute(data) };
        let iter = data_32.iter().enumerate();
        match match_val {
            true => self.load_packed_u1_inner::<SET, CMP, true>(iter),
            false => self.load_packed_u1_inner::<SET, CMP, false>(iter),
        }
    }

    #[inline(always)]
    pub fn load_yz_packed_u1_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 512],
        x_slice: usize,
        match_val: bool,
    ) {
        let data_32: &[u32; 1024] = unsafe { transmute(data) };
        let iter = data_32.iter().enumerate().skip(x_slice * 32).take(32);
        match match_val {
            true => self.load_packed_u1_inner::<SET, CMP, true>(iter),
            false => self.load_packed_u1_inner::<SET, CMP, false>(iter),
        }
    }

    #[inline(always)]
    pub fn load_xz_packed_u1_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 512],
        y_slice: usize,
        match_val: bool,
    ) {
        let data_32: &[u32; 1024] = unsafe { transmute(data) };
        let iter = data_32.iter().enumerate().skip(y_slice).step_by(32);
        match match_val {
            true => self.load_packed_u1_inner::<SET, CMP, true>(iter),
            false => self.load_packed_u1_inner::<SET, CMP, false>(iter),
        }
    }

    // u2 loaders.
    #[inline(always)]
    fn load_packed_u2_inner<const SET: u8, const CMP: u8, const MATCH: u8>(
        &mut self,
        data: &[u64; 1024],
        src: impl Iterator<Item = usize>,
    ) {
        for i in src {
            let (a, b, c, d) = process_packed_u2::<CMP, MATCH, A>(&data[i..]);

            let b_idx = i >> 1;
            self.set_u64::<SET>(b_idx, a);
            self.set_u64::<SET>(b_idx + 1, b);
            self.set_u64::<SET>(b_idx + 2, c);
            self.set_u64::<SET>(b_idx + 3, d);
        }
    }

    #[inline(always)]
    fn load_xz_packed_u2_inner<const SET: u8, const CMP: u8, const MATCH: u8>(
        &mut self,
        data: &[u64; 1024],
        src: impl Iterator<Item = usize>,
    ) {
        for i in src {
            #[rustfmt::skip]
            let slice = [
                data[i],       data[i + 32],  data[i + 64],  data[i + 96],
                data[i + 128], data[i + 160], data[i + 192], data[i + 224],
            ];
            let (a, b, c, d) = process_packed_u2::<CMP, MATCH, A>(&slice);

            self.set_u32::<SET>(i, a as u32);
            self.set_u32::<SET>(i + 32, (a >> 32) as u32);
            self.set_u32::<SET>(i + 64, b as u32);
            self.set_u32::<SET>(i + 96, (b >> 32) as u32);
            self.set_u32::<SET>(i + 128, c as u32);
            self.set_u32::<SET>(i + 160, (c >> 32) as u32);
            self.set_u32::<SET>(i + 192, d as u32);
            self.set_u32::<SET>(i + 224, (d >> 32) as u32);
        }
    }

    #[inline(always)]
    pub fn load_packed_u2_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 1024],
        match_val: u8,
    ) {
        let iter = (0..1024).step_by(8);
        match match_val {
            0 => self.load_packed_u2_inner::<SET, CMP, 0>(data, iter),
            1 => self.load_packed_u2_inner::<SET, CMP, 1>(data, iter),
            2 => self.load_packed_u2_inner::<SET, CMP, 2>(data, iter),
            3 => self.load_packed_u2_inner::<SET, CMP, 3>(data, iter),
            _ => unreachable!(),
        };
    }

    #[inline(always)]
    pub fn load_yz_packed_u2_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 1024],
        x_slice: usize,
        match_val: u8,
    ) {
        let iter = (0..1024).skip(x_slice * 32).step_by(8).take(4);
        match match_val {
            0 => self.load_packed_u2_inner::<SET, CMP, 0>(data, iter),
            1 => self.load_packed_u2_inner::<SET, CMP, 1>(data, iter),
            2 => self.load_packed_u2_inner::<SET, CMP, 2>(data, iter),
            3 => self.load_packed_u2_inner::<SET, CMP, 3>(data, iter),
            _ => unreachable!(),
        };
    }

    #[inline(always)]
    pub fn load_xz_packed_u2_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 1024],
        y_slice: usize,
        match_val: u8,
    ) {
        let iter = (0..1024).skip(y_slice).step_by(256);
        match match_val {
            0 => self.load_xz_packed_u2_inner::<SET, CMP, 0>(data, iter),
            1 => self.load_xz_packed_u2_inner::<SET, CMP, 1>(data, iter),
            2 => self.load_xz_packed_u2_inner::<SET, CMP, 2>(data, iter),
            3 => self.load_xz_packed_u2_inner::<SET, CMP, 3>(data, iter),
            _ => unreachable!(),
        };
    }

    // u4 loaders.
    #[inline(always)]
    fn load_packed_u4_inner<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 2048],
        src: impl Iterator<Item = usize>,
        cmp_val: u8,
    ) {
        for i in src {
            let (a, b) = process_packed_u4::<CMP, A>(&data[i..], cmp_val);

            let b_idx = i >> 2;
            self.set_u64::<SET>(b_idx, a);
            self.set_u64::<SET>(b_idx + 1, b);
        }
    }

    #[inline(always)]
    fn load_xz_packed_u4_inner<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 2048],
        src: impl Iterator<Item = usize>,
        cmp_val: u8,
    ) {
        for i in src {
            #[rustfmt::skip]
            let slice = [
                data[i], data[i + 1], data[i + 64], data[i + 65],
                data[i + 128], data[i + 129], data[i + 192], data[i + 193]
            ];
            let (a, b) = process_packed_u4::<CMP, A>(&slice, cmp_val);

            let bitfield_idx = i >> 1;
            self.set_u32::<SET>(bitfield_idx, a as u32);
            self.set_u32::<SET>(bitfield_idx + 32, (a >> 32) as u32);
            self.set_u32::<SET>(bitfield_idx + 64, b as u32);
            self.set_u32::<SET>(bitfield_idx + 96, (b >> 32) as u32);
        }
    }

    #[inline(always)]
    pub fn load_packed_u4_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 2048],
        match_val: u8,
    ) {
        let iter = (0..2048).step_by(8);
        self.load_packed_u4_inner::<SET, CMP>(data, iter, match_val);
    }

    #[inline(always)]
    pub fn load_yz_packed_u4_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 2048],
        x_slice: usize,
        match_val: u8,
    ) {
        let iter = (0..2048).skip(x_slice * 64).step_by(8).take(8);
        self.load_packed_u4_inner::<SET, CMP>(data, iter, match_val);
    }

    #[inline(always)]
    pub fn load_xz_packed_u4_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 2048],
        y_slice: usize,
        match_val: u8,
    ) {
        let iter = (0..2048).skip(y_slice * 2).step_by(256);
        self.load_xz_packed_u4_inner::<SET, CMP>(data, iter, match_val);
    }

    // u8 loaders.
    #[inline(always)]
    fn load_packed_u8_inner<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 4096],
        src: impl Iterator<Item = usize>,
        cmp_val: u8,
    ) {
        for i in src {
            let bits = process_packed_u8::<CMP, A>(&data[i..], cmp_val);
            self.set_u64::<SET>(i >> 3, bits);
        }
    }

    #[inline(always)]
    fn load_xz_packed_u8_inner<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 4096],
        src: impl Iterator<Item = usize>,
        cmp_val: u8,
    ) {
        for i in src {
            #[rustfmt::skip]
            let slice = [
                data[i], data[i + 1], data[i + 2], data[i + 3],
                data[i + 128], data[i + 129], data[i + 130], data[i + 131],
            ];
            let bits = process_packed_u8::<CMP, A>(&slice, cmp_val);

            let b_idx = i >> 2;
            self.set_u32::<SET>(b_idx, bits as u32);
            self.set_u32::<SET>(b_idx + 32, (bits >> 32) as u32);
        }
    }

    #[inline(always)]
    pub fn load_packed_u8_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 4096],
        match_val: u8,
    ) {
        let iter = (0..4096).step_by(8);
        self.load_packed_u8_inner::<SET, CMP>(data, iter, match_val);
    }

    #[inline(always)]
    pub fn load_yz_packed_u8_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 4096],
        x_slice: usize,
        match_val: u8,
    ) {
        let iter = (0..4096).skip(x_slice * 128).step_by(8).take(16);
        self.load_packed_u8_inner::<SET, CMP>(data, iter, match_val);
    }

    #[inline(always)]
    pub fn load_xz_packed_u8_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 4096],
        y_slice: usize,
        match_val: u8,
    ) {
        let iter = (0..4096).skip(y_slice * 4).step_by(256);
        self.load_xz_packed_u8_inner::<SET, CMP>(data, iter, match_val);
    }

    // u16 loaders.
    #[inline(always)]
    fn load_packed_u16_inner<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 8192],
        src: impl Iterator<Item = usize>,
        cmp_val: u16,
    ) {
        for i in src {
            let bits = process_packed_u16::<CMP, A>(&data[i..], cmp_val);
            self.set_u32::<SET>(i >> 3, bits);
        }
    }

    #[inline(always)]
    pub fn load_packed_u16_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 8192],
        match_val: u16,
    ) {
        let iter = (0..8192).step_by(8);
        self.load_packed_u16_inner::<SET, CMP>(data, iter, match_val);
    }

    #[inline(always)]
    pub fn load_yz_packed_u16_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 8192],
        x_slice: usize,
        match_val: u16,
    ) {
        let iter = (0..8192).skip(x_slice * 256).step_by(8).take(32);
        self.load_packed_u16_inner::<SET, CMP>(data, iter, match_val);
    }

    #[inline(always)]
    pub fn load_xz_packed_u16_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 8192],
        y_slice: usize,
        match_val: u16,
    ) {
        let iter = (0..8192).skip(y_slice * 8).step_by(256);
        self.load_packed_u16_inner::<SET, CMP>(data, iter, match_val);
    }

    #[inline(always)]
    pub fn store_into_packed_u1(&self, array: &mut [u64; 512], index: bool) {
        let data_u64: &[u64; 512] = unsafe { transmute(&self.data) };
        if index {
            zip(array.iter_mut(), data_u64.iter()).for_each(|(array, bits)| *array |= *bits);
        } else {
            zip(array.iter_mut(), data_u64.iter()).for_each(|(array, bits)| *array &= !*bits);
        }
    }

    #[inline(always)]
    pub fn store_into_packed_u2(&self, array: &mut [u64; 1024], index: u8) {
        assert!(index < 4, "Index is invalid!");

        let index_casted = index as u64 * 0x5555_5555_5555_5555;
        #[rustfmt::skip]
        let lut: [u8; 64] = [
            0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F,
            0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3, 0xFC, 0xFF,
            0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F,
            0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3, 0xFC, 0xFF,
            0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F,
            0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3, 0xFC, 0xFF,
            0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F,
            0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3, 0xFC, 0xFF,
        ];
        let pre_permute: [u64; 8] = match A::SIMD_WIDTH {
            16 => [0, 1, 0, 0, 0, 0, 0, 0],
            32 => [0, 2, 1, 3, 0, 0, 0, 0],
            64 => [0, 4, 1, 5, 2, 6, 3, 7],
            _ => unreachable!(),
        };

        let lut = Simd::from_slice(&lut);
        let pre_permute = Simd::from_slice(&pre_permute);
        let simd_index = Simd::splat(index_casted);
        let lo_bit_mask = Simd::splat(0xFu8);
        for i in (0..1024).step_by(Simd::<u32, A>::LANES) {
            let bits = Simd::<u32, A>::from_slice(&self.data[i..]);
            let bits = bits.permute_64(pre_permute);
            let shifted = bits.shift_rl_16::<4>();

            let lo = lut.shuffle_8(bits.raw_cast() & lo_bit_mask);
            let hi = lut.shuffle_8(shifted.raw_cast() & lo_bit_mask);

            let lo_spread = lo.interleave_lower_block_8(hi);
            let hi_spread = lo.interleave_upper_block_8(hi);

            let lo_cur = Simd::from_slice(&array[i..]);
            let hi_cur = Simd::from_slice(&array[i + Simd::<u64, A>::LANES..]);

            let lo_swap = (lo_cur ^ simd_index) & lo_spread.raw_cast();
            let hi_swap = (hi_cur ^ simd_index) & hi_spread.raw_cast();

            let lo_new = lo_cur ^ lo_swap;
            let hi_new = hi_cur ^ hi_swap;

            lo_new.copy_to_slice(&mut array[i..]);
            hi_new.copy_to_slice(&mut array[i + Simd::<u64, A>::LANES..]);
        }
    }

    #[inline(always)]
    pub fn store_into_packed_u4(&self, array: &mut [u64; 2048], index: u8) {
        assert!(index < 16, "Index is invalid!");

        #[rustfmt::skip]
        let lut: [u8; 64] = [
            0x0, 0x0F, 0xF0, 0xFF, 0x0, 0x0F, 0xF0, 0xFF,
            0x0, 0x0F, 0xF0, 0xFF, 0x0, 0x0F, 0xF0, 0xFF,
            0x0, 0x0F, 0xF0, 0xFF, 0x0, 0x0F, 0xF0, 0xFF,
            0x0, 0x0F, 0xF0, 0xFF, 0x0, 0x0F, 0xF0, 0xFF,
            0x0, 0x0F, 0xF0, 0xFF, 0x0, 0x0F, 0xF0, 0xFF,
            0x0, 0x0F, 0xF0, 0xFF, 0x0, 0x0F, 0xF0, 0xFF,
            0x0, 0x0F, 0xF0, 0xFF, 0x0, 0x0F, 0xF0, 0xFF,
            0x0, 0x0F, 0xF0, 0xFF, 0x0, 0x0F, 0xF0, 0xFF,
        ];

        let lut = Simd::from_slice(&lut);

        let index_casted = index as u64 * 0x1111_1111_1111_1111;
        let simd_index = Simd::splat(index_casted);
        let lo_bit_mask = Simd::splat(0x3u8);

        for i in (0..1024).step_by(Simd::<u32, A>::LANES) {
            let b1 = Simd::<u32, A>::from_slice(&self.data[i..]);
            let b2 = b1.shift_rl_16::<2>();
            let b3 = b1.shift_rl_16::<4>();
            let b4 = b1.shift_rl_16::<6>();

            let c1 = lut.shuffle_8(b1.raw_cast() & lo_bit_mask);
            let c2 = lut.shuffle_8(b2.raw_cast() & lo_bit_mask);
            let c3 = lut.shuffle_8(b3.raw_cast() & lo_bit_mask);
            let c4 = lut.shuffle_8(b4.raw_cast() & lo_bit_mask);

            let d1 = c1.interleave_lower_8(c2);
            let d2 = c1.interleave_upper_8(c2);
            let d3 = c3.interleave_lower_8(c4);
            let d4 = c3.interleave_upper_8(c4);

            let e1 = d1.interleave_lower_16(d3);
            let e2 = d1.interleave_upper_16(d3);
            let e3 = d2.interleave_lower_16(d4);
            let e4 = d2.interleave_upper_16(d4);

            let lanes = Simd::<u64, A>::LANES;
            let cur1 = Simd::from_slice(&array[i * 2..]);
            let cur2 = Simd::from_slice(&array[i * 2 + lanes..]);
            let cur3 = Simd::from_slice(&array[i * 2 + 2 * lanes..]);
            let cur4 = Simd::from_slice(&array[i * 2 + 3 * lanes..]);

            let swap1 = (cur1 ^ simd_index) & e1.raw_cast();
            let swap2 = (cur2 ^ simd_index) & e2.raw_cast();
            let swap3 = (cur3 ^ simd_index) & e3.raw_cast();
            let swap4 = (cur4 ^ simd_index) & e4.raw_cast();

            let new1 = cur1 ^ swap1;
            let new2 = cur2 ^ swap2;
            let new3 = cur3 ^ swap3;
            let new4 = cur4 ^ swap4;

            new1.copy_to_slice(&mut array[i * 2..]);
            new2.copy_to_slice(&mut array[i * 2 + lanes..]);
            new3.copy_to_slice(&mut array[i * 2 + 2 * lanes..]);
            new4.copy_to_slice(&mut array[i * 2 + 3 * lanes..]);
        }
    }

    #[inline(always)]
    pub fn store_into_packed_u8(&self, array: &mut [u64; 4096], index: u8) {
        let index = Simd::<u8, A>::splat(index);
        let array_u8: &mut [u8; 32768] = unsafe { transmute(array) };
        let data_16: &[u16; 2048] = unsafe { transmute(&self.data) };
        let data_64: &[u64; 512] = unsafe { transmute(&self.data) };
        for i in (0..32768).step_by(Simd::<u8, A>::LANES) {
            let bits = match A::SIMD_WIDTH {
                16 => data_16[i / 16] as u64,
                32 => self.data[i / 32] as u64,
                64 => data_64[i / 64] as u64,
                _ => unreachable!(),
            };

            let mask = Mask::from_bits(bits);
            let cur = Simd::from_slice(&array_u8[i..]);
            let new_bits = mask.select(index, cur);

            new_bits.copy_to_slice(&mut array_u8[i..]);
        }
    }

    #[inline(always)]
    pub fn store_into_packed_u16(&self, array: &mut [u64; 8192], index: u16) {
        let index = Simd::<u16, A>::splat(index);
        let array_u16: &mut [u16; 32768] = unsafe { transmute(array) };
        let data_8: &[u8; 4096] = unsafe { transmute(&self.data) };
        let data_16: &[u16; 2048] = unsafe { transmute(&self.data) };
        for i in (0..32768).step_by(Simd::<u16, A>::LANES) {
            let bits = match A::SIMD_WIDTH {
                16 => data_8[i / 8] as u64,
                32 => data_16[i / 16] as u64,
                64 => self.data[i / 32] as u64,
                _ => unreachable!(),
            };

            let mask = Mask::from_bits(bits);
            let cur = Simd::from_slice(&array_u16[i..]);
            let new_bits = mask.select(index, cur);

            new_bits.copy_to_slice(&mut array_u16[i..]);
        }
    }

    pub fn to_array(self) -> [u32; 1024] {
        self.data
    }

    pub fn to_array_64(self) -> [u64; 512] {
        unsafe { transmute(self.data) }
    }

    pub fn as_array(&self) -> &[u32; 1024] {
        &self.data
    }

    pub fn as_slice(&self) -> &[u32] {
        self.data.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [u32] {
        self.data.as_mut_slice()
    }

    pub fn cull_most_sig_bits(&mut self) -> &mut Self {
        for i in (0..1024).step_by(16) {
            let block = u32x16::from_slice(&self.data[i..i + 16]);
            let culled = block & !(block << u32x16::splat(1));
            culled.copy_to_slice(&mut self.data[i..i + 16]);
        }
        self
    }

    pub fn cull_least_sig_bits(&mut self) -> &mut Self {
        for i in (0..1024).step_by(16) {
            let block = u32x16::from_slice(&self.data[i..i + 16]);
            let culled = block & !(block >> u32x16::splat(1));
            culled.copy_to_slice(&mut self.data[i..i + 16]);
        }
        self
    }

    /// Transposes all 1024 elements as a 32x32 matrix.
    pub fn outer_transpose(&mut self) -> &mut Self {
        outer_transpose_simply_simd::<A>(&mut self.data);
        self
    }

    pub fn outer_transpose_scalar(&mut self) -> &mut Self {
        outer_transpose_scalar(&mut self.data);
        self
    }

    /// Transposes each chunk of 32 elements as a 32x32 bit matrix.
    pub fn inner_transpose(&mut self) -> &mut Self {
        for i in (0..1024).step_by(32) {
            inner_transpose_slice_simply_simd::<StaticArch>(&mut self.data[i..]);
        }
        self
    }

    pub fn inner_transpose_scalar(&mut self) -> &mut Self {
        for i in (0..1024).step_by(32) {
            inner_transpose_slice_scalar(&mut self.data[i..]);
        }
        self
    }

    pub fn print_inner_slices(&self, range: Range<usize>) {
        print_matrix_inner_slices(&self.data, range);
    }

    pub fn print_outer_slices(&self, range: Range<usize>) {
        print_matrix_outer_slices(&self.data, range);
    }
}
