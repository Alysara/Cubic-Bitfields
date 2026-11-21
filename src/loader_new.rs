use std::mem::transmute;
use std::ops::{BitAndAssign, BitOrAssign, BitXorAssign};

use simply_simd::{Arch, Mask, Simd, SimdElement};

pub const CMP_EQ: u8 = CmpFlag::Eq as u8;
pub const CMP_NE: u8 = CmpFlag::Ne as u8;
pub const CMP_GT: u8 = CmpFlag::Gt as u8;
pub const CMP_LT: u8 = CmpFlag::Lt as u8;

pub const SET_ASSIGN: u8 = SetFlag::Assign as u8;
pub const SET_OR: u8 = SetFlag::Or as u8;
pub const SET_XOR: u8 = SetFlag::Xor as u8;
pub const SET_AND: u8 = SetFlag::And as u8;

#[repr(u8)]
pub enum CmpFlag {
    Eq,
    Ne,
    Gt,
    Lt,
}

#[repr(u8)]
pub enum SetFlag {
    Assign,
    Or,
    Xor,
    And,
}

#[inline(always)]
pub(crate) fn apply_set<const SET: u8, T>(target: &mut T, data: T)
where
    T: BitOrAssign + BitXorAssign + BitAndAssign,
{
    let flag: SetFlag = unsafe { transmute(SET) };
    match flag {
        SetFlag::Assign => *target = data,
        SetFlag::Or => *target |= data,
        SetFlag::Xor => *target ^= data,
        SetFlag::And => *target &= data,
    };
}

#[inline(always)]
fn apply_cmp<const CMP: u8, T: SimdElement, A: Arch>(
    lhs: Simd<T, A>,
    rhs: Simd<T, A>,
) -> Mask<T, A> {
    let flag: CmpFlag = unsafe { transmute(CMP) };
    match flag {
        CmpFlag::Eq => lhs.simd_eq(rhs),
        CmpFlag::Ne => lhs.simd_ne(rhs),
        CmpFlag::Gt => lhs.simd_gt(rhs),
        CmpFlag::Lt => lhs.simd_lt(rhs),
    }
}

#[inline(always)]
pub(crate) fn process_packed_u1<const CMP: u8, const MATCH: bool>(data: &u32) -> u32 {
    if CMP >= 4 {
        panic!("Invalid CMP flag!");
    }

    let cmp_type: CmpFlag = unsafe { transmute(CMP) };

    if MATCH {
        match cmp_type {
            CmpFlag::Eq => *data,
            CmpFlag::Ne => !*data,
            CmpFlag::Gt => 0,
            CmpFlag::Lt => !*data,
        }
    } else {
        match cmp_type {
            CmpFlag::Eq => !*data,
            CmpFlag::Ne => *data,
            CmpFlag::Gt => *data,
            CmpFlag::Lt => 0,
        }
    }
}

#[cfg(target_feature = "bmi2")]
#[inline(always)]
pub(crate) fn process_packed_u2<const CMP: u8, const MATCH: u8, A: Arch>(
    data: &[u64],
) -> (u64, u64, u64, u64) {
    if MATCH >= 4 {
        panic!("Invalid match value!");
    }

    if CMP >= 4 {
        panic!("Invalid CMP flag!");
    }

    let target1 = Simd::splat(MATCH);
    let target2 = Simd::splat(MATCH << 2);
    let target3 = Simd::splat(MATCH << 4);
    let target4 = Simd::splat(MATCH << 6);

    let mask1 = Simd::splat(0x03);
    let mask2 = Simd::splat(0x0C);
    let mask3 = Simd::splat(0x30);
    let mask4 = Simd::splat(0xC0);

    let mut c1 = 0u64;
    let mut c2 = 0u64;
    let mut c3 = 0u64;
    let mut c4 = 0u64;

    for i in (0..64).step_by(Simd::<u8, A>::LANES) {
        let block: Simd<u8, A> = Simd::from_slice(&data[(i / 8)..]).raw_cast();

        let b1 = block & mask1;
        let b2 = block & mask2;
        let b3 = block & mask3;
        let b4 = block & mask4;

        c1 |= apply_cmp::<CMP, _, _>(b1, target1).to_bits() << i;
        c2 |= apply_cmp::<CMP, _, _>(b2, target2).to_bits() << i;
        c3 |= apply_cmp::<CMP, _, _>(b3, target3).to_bits() << i;
        c4 |= apply_cmp::<CMP, _, _>(b4, target4).to_bits() << i;
    }

    const M1: u64 = 0x1111111111111111;
    const M2: u64 = 0x2222222222222222;
    const M3: u64 = 0x4444444444444444;
    const M4: u64 = 0x8888888888888888;

    unsafe {
        use std::arch::x86_64::_pdep_u64;

        let bits1 = _pdep_u64(c1, M1) | _pdep_u64(c2, M2) | _pdep_u64(c3, M3) | _pdep_u64(c4, M4);
        let bits2 = _pdep_u64(c1 >> 16, M1)
            | _pdep_u64(c2 >> 16, M2)
            | _pdep_u64(c3 >> 16, M3)
            | _pdep_u64(c4 >> 16, M4);
        let bits3 = _pdep_u64(c1 >> 32, M1)
            | _pdep_u64(c2 >> 32, M2)
            | _pdep_u64(c3 >> 32, M3)
            | _pdep_u64(c4 >> 32, M4);
        let bits4 = _pdep_u64(c1 >> 48, M1)
            | _pdep_u64(c2 >> 48, M2)
            | _pdep_u64(c3 >> 48, M3)
            | _pdep_u64(c4 >> 48, M4);

        (bits1, bits2, bits3, bits4)
    }
}

#[cfg(not(target_feature = "bmi2"))]
#[inline(always)]
pub(crate) fn process_packed_u2<const CMP: u8, const MATCH: u8>(
    data: &[u64],
) -> (u64, u64, u64, u64) {
    if MATCH >= 4 {
        panic!("Invalid match value!");
    }

    if CMP >= 4 {
        panic!("Invalid CMP flag!");
    }

    let target1 = u8x64::splat(MATCH);
    let target2 = u8x64::splat(MATCH << 2);
    let target3 = u8x64::splat(MATCH << 4);
    let target4 = u8x64::splat(MATCH << 6);

    let mask1 = u8x64::splat(0x03);
    let mask2 = u8x64::splat(0x0C);
    let mask3 = u8x64::splat(0x30);
    let mask4 = u8x64::splat(0xC0);

    let raw_block = u64x8::from_slice(data);
    let block: u8x64 = unsafe { transmute(raw_block) };

    let b1 = block & mask1;
    let b2 = block & mask2;
    let b3 = block & mask3;
    let b4 = block & mask4;

    let s1_res1 = apply_cmp::<CMP, _, _>(b1, target1).to_simd();
    let s1_res2 = apply_cmp::<CMP, _, _>(b2, target2).to_simd();
    let s1_res3 = apply_cmp::<CMP, _, _>(b3, target3).to_simd();
    let s1_res4 = apply_cmp::<CMP, _, _>(b4, target4).to_simd();

    let (s2_res1, s2_res2): (u16x32, u16x32) = unsafe { transmute(s1_res1.interleave(s1_res2)) };
    let (s2_res3, s2_res4): (u16x32, u16x32) = unsafe { transmute(s1_res3.interleave(s1_res4)) };

    let (s3_res1, s3_res2): (u8x64, u8x64) = unsafe { transmute(s2_res1.interleave(s2_res3)) };
    let (s3_res3, s3_res4): (u8x64, u8x64) = unsafe { transmute(s2_res2.interleave(s2_res4)) };

    let zero = u8x64::splat(0);
    let bits1 = s3_res1.simd_ne(zero).to_bitmask();
    let bits2 = s3_res2.simd_ne(zero).to_bitmask();
    let bits3 = s3_res3.simd_ne(zero).to_bitmask();
    let bits4 = s3_res4.simd_ne(zero).to_bitmask();

    (bits1, bits2, bits3, bits4)
}

#[cfg(target_feature = "bmi2")]
#[inline(always)]
pub(crate) fn process_packed_u4<const CMP: u8, A: Arch>(data: &[u64], cmp_val: u8) -> (u64, u64) {
    if CMP >= 4 {
        panic!("Invalid CMP flag!");
    }

    let mut b1 = 0u64;
    let mut b2 = 0u64;

    let target1 = Simd::splat(cmp_val);
    let target2 = Simd::splat(cmp_val << 4);

    let mask1 = Simd::splat(0x0F);
    let mask2 = Simd::splat(0xF0);

    for i in (0..64).step_by(Simd::<u8, A>::LANES) {
        let block: Simd<u8, A> = Simd::from_slice(&data[(i / 8)..]).raw_cast();

        let a1 = block & mask1;
        let a2 = block & mask2;

        b1 |= apply_cmp::<CMP, _, _>(a1, target1).to_bits() << i;
        b2 |= apply_cmp::<CMP, _, _>(a2, target2).to_bits() << i;
    }

    const M1: u64 = 0x5555555555555555;
    const M2: u64 = 0xAAAAAAAAAAAAAAAA;

    unsafe {
        use std::arch::x86_64::_pdep_u64;

        let bits1 = _pdep_u64(b1, M1) | _pdep_u64(b2, M2);
        let bits2 = _pdep_u64(b1 >> 32, M1) | _pdep_u64(b2 >> 32, M2);

        (bits1, bits2)
    }
}

#[cfg(not(target_feature = "bmi2"))]
#[inline(always)]
pub(crate) fn process_packed_u4<const CMP: u8>(data: &[u64], cmp_val: u8) -> (u64, u64) {
    if CMP >= 4 {
        panic!("Invalid CMP flag!");
    }

    let target1 = u8x64::splat(cmp_val);
    let target2 = u8x64::splat(cmp_val << 4);

    let mask1 = u8x64::splat(0x0F);
    let mask2 = u8x64::splat(0xF0);

    let raw_block = u64x8::from_slice(data);
    let block: u8x64 = unsafe { transmute(raw_block) };

    let a1 = block & mask1;
    let a2 = block & mask2;

    let b1 = apply_cmp::<CMP, _, _>(a1, target1).to_simd();
    let b2 = apply_cmp::<CMP, _, _>(a2, target2).to_simd();

    let (c1, c2) = b1.interleave(b2);

    let zero = i8x64::splat(0);
    let bits1 = c1.simd_ne(zero).to_bitmask();
    let bits2 = c2.simd_ne(zero).to_bitmask();

    (bits1, bits2)
}

#[inline(always)]
pub(crate) fn process_packed_u8<const CMP: u8, A: Arch>(data: &[u64], cmp_val: u8) -> u64 {
    if CMP >= 4 {
        panic!("Invalid CMP flag!");
    }

    let mut bitmask = 0u64;
    let target = Simd::splat(cmp_val);
    for i in (0..64).step_by(Simd::<u8, A>::LANES) {
        let block: Simd<u8, A> = Simd::from_slice(&data[(i / 8)..]).raw_cast();
        bitmask |= apply_cmp::<CMP, _, _>(block, target).to_bits() << i;
    }

    bitmask
}

#[inline(always)]
pub(crate) fn process_packed_u16<const CMP: u8, A: Arch>(data: &[u64], cmp_val: u16) -> u32 {
    if CMP >= 4 {
        panic!("Invalid CMP flag!");
    }

    let mut bitmask = 0u32;
    let target = Simd::splat(cmp_val);

    for i in (0..32).step_by(Simd::<u16, A>::LANES) {
        let block: Simd<u16, A> = Simd::from_slice(&data[(i / 4)..]).raw_cast();
        bitmask |= (apply_cmp::<CMP, _, _>(block, target).to_bits() as u32) << i;
    }

    bitmask
}
