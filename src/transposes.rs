use std::arch::is_aarch64_feature_detected;
use std::simd::{simd_swizzle, u32x4, u32x16};

use simply_simd::{Arch, Simd};

/// Transposes all 1024 elements as a 32x32 matrix.
#[inline(always)]
pub fn outer_transpose(matrix: &mut [u32; 1024]) {
    for y in 0..8 {
        // Handle middle block case.
        let i = y * 128 + y * 4;
        let block = load_4x4_block(matrix, i);
        let tblock = transpose_4x4_block(block);
        store_4x4_block(matrix, i, tblock);

        for z in (y + 1)..8 {
            // Perform 4x4 block swaps.
            let i = y * 128 + z * 4;
            let j = y * 4 + z * 128;

            let i_block = load_4x4_block(matrix, i);
            let j_block = load_4x4_block(matrix, j);

            let i_tblock = transpose_4x4_block(i_block);
            let j_tblock = transpose_4x4_block(j_block);

            store_4x4_block(matrix, i, j_tblock);
            store_4x4_block(matrix, j, i_tblock);
        }
    }
}

#[inline(always)]
fn load_4x4_block(matrix: &[u32; 1024], index: usize) -> (u32x4, u32x4, u32x4, u32x4) {
    (
        u32x4::from_slice(&matrix[index..]),
        u32x4::from_slice(&matrix[index + 32..]),
        u32x4::from_slice(&matrix[index + 64..]),
        u32x4::from_slice(&matrix[index + 96..]),
    )
}

#[inline(always)]
fn store_4x4_block(matrix: &mut [u32; 1024], index: usize, block: (u32x4, u32x4, u32x4, u32x4)) {
    block.0.copy_to_slice(&mut matrix[index..]);
    block.1.copy_to_slice(&mut matrix[index + 32..]);
    block.2.copy_to_slice(&mut matrix[index + 64..]);
    block.3.copy_to_slice(&mut matrix[index + 96..]);
}

fn transpose_4x4_block(block: (u32x4, u32x4, u32x4, u32x4)) -> (u32x4, u32x4, u32x4, u32x4) {
    const SWIZZLE_64_LO: [usize; 4] = [0, 1, 4, 5];
    const SWIZZLE_64_HI: [usize; 4] = [2, 3, 6, 7];
    const SWIZZLE_32_LO: [usize; 4] = [0, 4, 2, 6];
    const SWIZZLE_32_HI: [usize; 4] = [1, 5, 3, 7];

    let t1 = simd_swizzle!(block.0, block.2, SWIZZLE_64_LO);
    let t2 = simd_swizzle!(block.1, block.3, SWIZZLE_64_LO);
    let t3 = simd_swizzle!(block.0, block.2, SWIZZLE_64_HI);
    let t4 = simd_swizzle!(block.1, block.3, SWIZZLE_64_HI);

    (
        simd_swizzle!(t1, t2, SWIZZLE_32_LO),
        simd_swizzle!(t1, t2, SWIZZLE_32_HI),
        simd_swizzle!(t3, t4, SWIZZLE_32_LO),
        simd_swizzle!(t3, t4, SWIZZLE_32_HI),
    )
}

#[inline(always)]
pub fn outer_transpose_scalar(matrix: &mut [u32; 1024]) {
    for z in 0..32 {
        for y in (z + 1)..32 {
            matrix.swap(z * 32 + y, y * 32 + z);
        }
    }
}

// --- Inner transpose algorithm obtained from:
// --- https://github.com/Pnoenix/fast-bit-matrix-transpose

/// Transposes each chunk of 32 elements as a 32x32 bit matrix.
#[inline(always)]
pub fn inner_transpose_slice(slice: &mut [u32]) {
    // 16 bits
    let hi = u32x16::from_slice(&slice[..]);
    let lo = u32x16::from_slice(&slice[16..]);
    const MASK_16: u32x16 = u32x16::splat(0x0000FFFF);
    const SHIFT_16: u32x16 = u32x16::splat(16);
    let hi16 = (hi & MASK_16) | lo << SHIFT_16;
    let lo16 = (lo & !MASK_16) | hi >> SHIFT_16;

    // 8 bits
    const S8H: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23];
    const S8L: [usize; 16] = [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31];
    let hi8_prep = simd_swizzle!(hi16, lo16, S8H);
    let lo8_prep = simd_swizzle!(hi16, lo16, S8L);
    const MASK_8: u32x16 = u32x16::splat(0x00FF00FF);
    const SHIFT_8: u32x16 = u32x16::splat(8);
    let hi8 = (hi8_prep & MASK_8) | ((lo8_prep & MASK_8) << SHIFT_8);
    let lo8 = (lo8_prep & !MASK_8) | ((hi8_prep & !MASK_8) >> SHIFT_8);

    // 4 bits
    const S4H: [usize; 16] = [0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27];
    const S4L: [usize; 16] = [4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31];
    let hi4_prep = simd_swizzle!(hi8, lo8, S4H);
    let lo4_prep = simd_swizzle!(hi8, lo8, S4L);
    const MASK_4: u32x16 = u32x16::splat(0x0F0F0F0F);
    const SHIFT_4: u32x16 = u32x16::splat(4);
    let hi4 = (hi4_prep & MASK_4) | ((lo4_prep & MASK_4) << SHIFT_4);
    let lo4 = (lo4_prep & !MASK_4) | ((hi4_prep & !MASK_4) >> SHIFT_4);

    // 2 bits
    const S2H: [usize; 16] = [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29];
    const S2L: [usize; 16] = [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31];
    let hi2_prep = simd_swizzle!(hi4, lo4, S2H);
    let lo2_prep = simd_swizzle!(hi4, lo4, S2L);
    const MASK_2: u32x16 = u32x16::splat(0x33333333);
    const SHIFT_2: u32x16 = u32x16::splat(2);
    let hi2 = (hi2_prep & MASK_2) | ((lo2_prep & MASK_2) << SHIFT_2);
    let lo2 = (lo2_prep & !MASK_2) | ((hi2_prep & !MASK_2) >> SHIFT_2);

    // 1 bit
    const S1H: [usize; 16] = [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30];
    const S1L: [usize; 16] = [1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31];
    let hi1_prep = simd_swizzle!(hi2, lo2, S1H);
    let lo1_prep = simd_swizzle!(hi2, lo2, S1L);
    const MASK_1: u32x16 = u32x16::splat(0x55555555);
    const SHIFT_1: u32x16 = u32x16::splat(1);
    let hi1 = (hi1_prep & MASK_1) | ((lo1_prep & MASK_1) << SHIFT_1);
    let lo1 = (lo1_prep & !MASK_1) | ((hi1_prep & !MASK_1) >> SHIFT_1);

    // Final swizzle
    const SFH: [usize; 16] = [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23];
    const SFL: [usize; 16] = [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31];
    let final_hi = simd_swizzle!(hi1, lo1, SFH);
    let final_lo = simd_swizzle!(hi1, lo1, SFL);
    final_hi.copy_to_slice(&mut slice[..]);
    final_lo.copy_to_slice(&mut slice[16..]);
}

#[inline(always)]
pub fn inner_transpose_slice_simply_simd<A: Arch>(slice: &mut [u32]) {
    match A::SIMD_WIDTH {
        // sse4.2 + Neon
        16 => {
            let a1 = Simd::<u32, A>::from_slice(&slice[0..4]);
            let a2 = Simd::<u32, A>::from_slice(&slice[4..8]);
            let a3 = Simd::<u32, A>::from_slice(&slice[8..12]);
            let a4 = Simd::<u32, A>::from_slice(&slice[12..16]);
            let a5 = Simd::<u32, A>::from_slice(&slice[16..20]);
            let a6 = Simd::<u32, A>::from_slice(&slice[20..24]);
            let a7 = Simd::<u32, A>::from_slice(&slice[24..28]);
            let a8 = Simd::<u32, A>::from_slice(&slice[28..32]);

            let b1 = a1.interleave_lower_8(a3);
            let b2 = a2.interleave_lower_8(a4);
            let b3 = a1.interleave_upper_8(a3);
            let b4 = a2.interleave_upper_8(a4);
            let b5 = a5.interleave_lower_8(a7);
            let b6 = a6.interleave_lower_8(a8);
            let b7 = a5.interleave_upper_8(a7);
            let b8 = a6.interleave_upper_8(a8);

            let c1 = b1.interleave_lower_16(b5);
            let c2 = b2.interleave_lower_16(b6);
            let c3 = b1.interleave_upper_16(b5);
            let c4 = b2.interleave_upper_16(b6);
            let c5 = b3.interleave_lower_16(b7);
            let c6 = b4.interleave_lower_16(b8);
            let c7 = b3.interleave_upper_16(b7);
            let c8 = b4.interleave_upper_16(b8);

            // Perform swaps
            let mask4x4 = Simd::splat(0x0F0F_0F0F);
            let mask2x2 = Simd::splat(0x3333_3333);
            let mask1x1 = Simd::splat(0x5555_5555);

            #[cfg(target_arch = "aarch64")]
            {
                // Use mask approach for Neon to use bit instructions.
                // 4x4 Swap
                let not_mask4x4 = !mask4x4;
                let d1 = (c1 & mask4x4) | (c2.shift_left::<4>() & not_mask4x4);
                let d2 = (c2 & not_mask4x4) | (c1.shift_right::<4>() & mask4x4);
                let d3 = (c3 & mask4x4) | (c4.shift_left::<4>() & not_mask4x4);
                let d4 = (c4 & not_mask4x4) | (c3.shift_right::<4>() & mask4x4);
                let d5 = (c5 & mask4x4) | (c6.shift_left::<4>() & not_mask4x4);
                let d6 = (c6 & not_mask4x4) | (c5.shift_right::<4>() & mask4x4);
                let d7 = (c7 & mask4x4) | (c8.shift_left::<4>() & not_mask4x4);
                let d8 = (c8 & not_mask4x4) | (c7.shift_right::<4>() & mask4x4);

                // 2x2 Swap
                let not_mask2x2 = !mask2x2;
                let e1 = (d1 & mask2x2) | (d5.shift_left::<2>() & not_mask2x2);
                let e5 = (d5 & not_mask2x2) | (d1.shift_right::<2>() & mask2x2);
                let e2 = (d2 & mask2x2) | (d6.shift_left::<2>() & not_mask2x2);
                let e6 = (d6 & not_mask2x2) | (d2.shift_right::<2>() & mask2x2);
                let e3 = (d3 & mask2x2) | (d7.shift_left::<2>() & not_mask2x2);
                let e7 = (d7 & not_mask2x2) | (d3.shift_right::<2>() & mask2x2);
                let e4 = (d4 & mask2x2) | (d8.shift_left::<2>() & not_mask2x2);
                let e8 = (d8 & not_mask2x2) | (d4.shift_right::<2>() & mask2x2);

                // 1x1 Swap
                let not_mask1x1 = !mask1x1;
                let f1 = (e1 & mask1x1) | (e3.shift_left::<1>() & not_mask1x1);
                let f3 = (e3 & not_mask1x1) | (e1.shift_right::<1>() & mask1x1);
                let f2 = (e2 & mask1x1) | (e4.shift_left::<1>() & not_mask1x1);
                let f4 = (e4 & not_mask1x1) | (e2.shift_right::<1>() & mask1x1);
                let f5 = (e5 & mask1x1) | (e7.shift_left::<1>() & not_mask1x1);
                let f7 = (e7 & not_mask1x1) | (e5.shift_right::<1>() & mask1x1);
                let f6 = (e6 & mask1x1) | (e8.shift_left::<1>() & not_mask1x1);
                let f8 = (e8 & not_mask1x1) | (e6.shift_right::<1>() & mask1x1);

                // Use specialized st4 instruction for interleaved store.

                use core::arch::aarch64::{uint32x4_t, uint32x4x4_t, vst4q_u32};
                use core::mem::transmute_copy;

                let p0 = f1.interleave_lower_32(f2);
                let p1 = f3.interleave_lower_32(f4);
                let p2 = f5.interleave_lower_32(f6);
                let p3 = f7.interleave_lower_32(f8);
                let q0 = f1.interleave_upper_32(f2);
                let q1 = f3.interleave_upper_32(f4);
                let q2 = f5.interleave_upper_32(f6);
                let q3 = f7.interleave_upper_32(f8);

                unsafe {
                    let r = |v: Simd<u32, A>| -> uint32x4_t { transmute_copy(&v) };

                    vst4q_u32(
                        slice[0..16].as_mut_ptr(),
                        uint32x4x4_t(r(p0), r(p1), r(p2), r(p3)),
                    );
                    vst4q_u32(
                        slice[16..32].as_mut_ptr(),
                        uint32x4x4_t(r(q0), r(q1), r(q2), r(q3)),
                    );
                }
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                // Otherwise use cheaper xor swap for sse4.2
                let swap1 = (c1.shift_right::<4>() ^ c2) & mask4x4;
                let swap2 = (c3.shift_right::<4>() ^ c4) & mask4x4;
                let swap3 = (c5.shift_right::<4>() ^ c6) & mask4x4;
                let swap4 = (c7.shift_right::<4>() ^ c8) & mask4x4;
                let d1 = c1 ^ swap1.shift_left::<4>();
                let d2 = c2 ^ swap1;
                let d3 = c3 ^ swap2.shift_left::<4>();
                let d4 = c4 ^ swap2;
                let d5 = c5 ^ swap3.shift_left::<4>();
                let d6 = c6 ^ swap3;
                let d7 = c7 ^ swap4.shift_left::<4>();
                let d8 = c8 ^ swap4;

                // 2x2 Swap
                let swap1 = (d1.shift_right::<2>() ^ d5) & mask2x2;
                let swap2 = (d2.shift_right::<2>() ^ d6) & mask2x2;
                let swap3 = (d3.shift_right::<2>() ^ d7) & mask2x2;
                let swap4 = (d4.shift_right::<2>() ^ d8) & mask2x2;
                let e1 = d1 ^ swap1.shift_left::<2>();
                let e2 = d2 ^ swap2.shift_left::<2>();
                let e3 = d3 ^ swap3.shift_left::<2>();
                let e4 = d4 ^ swap4.shift_left::<2>();
                let e5 = d5 ^ swap1;
                let e6 = d6 ^ swap2;
                let e7 = d7 ^ swap3;
                let e8 = d8 ^ swap4;

                // 1x1 Swap
                let swap1 = (e1.shift_right::<1>() ^ e3) & mask1x1;
                let swap2 = (e2.shift_right::<1>() ^ e4) & mask1x1;
                let swap3 = (e5.shift_right::<1>() ^ e7) & mask1x1;
                let swap4 = (e6.shift_right::<1>() ^ e8) & mask1x1;
                let f1 = e1 ^ swap1.shift_left::<1>();
                let f2 = e2 ^ swap2.shift_left::<1>();
                let f3 = e3 ^ swap1;
                let f4 = e4 ^ swap2;
                let f5 = e5 ^ swap3.shift_left::<1>();
                let f6 = e6 ^ swap4.shift_left::<1>();
                let f7 = e7 ^ swap3;
                let f8 = e8 ^ swap4;

                let g1 = f1.interleave_lower_32(f3);
                let g2 = f2.interleave_lower_32(f4);
                let g3 = f1.interleave_upper_32(f3);
                let g4 = f2.interleave_upper_32(f4);
                let g5 = f5.interleave_lower_32(f7);
                let g6 = f6.interleave_lower_32(f8);
                let g7 = f5.interleave_upper_32(f7);
                let g8 = f6.interleave_upper_32(f8);

                let h1 = g1.interleave_lower_64(g5);
                let h2 = g2.interleave_lower_64(g6);
                let h3 = g1.interleave_upper_64(g5);
                let h4 = g2.interleave_upper_64(g6);
                let h5 = g3.interleave_lower_64(g7);
                let h6 = g4.interleave_lower_64(g8);
                let h7 = g3.interleave_upper_64(g7);
                let h8 = g4.interleave_upper_64(g8);

                h1.copy_to_slice(&mut slice[0..4]);
                h2.copy_to_slice(&mut slice[4..8]);
                h3.copy_to_slice(&mut slice[8..12]);
                h4.copy_to_slice(&mut slice[12..16]);
                h5.copy_to_slice(&mut slice[16..20]);
                h6.copy_to_slice(&mut slice[20..24]);
                h7.copy_to_slice(&mut slice[24..28]);
                h8.copy_to_slice(&mut slice[28..32]);
            }
        }
        // avx2
        32 => {
            let a1 = Simd::<u32, A>::from_slice(&slice[..8]);
            let a2 = Simd::<u32, A>::from_slice(&slice[8..16]);
            let a3 = Simd::<u32, A>::from_slice(&slice[16..24]);
            let a4 = Simd::<u32, A>::from_slice(&slice[24..32]);

            let b1 = a1.interleave_lower_block_8(a2);
            let b2 = a1.interleave_upper_block_8(a2);
            let b3 = a3.interleave_lower_block_8(a4);
            let b4 = a3.interleave_upper_block_8(a4);
            let c1 = b1.interleave_lower_block_16(b3);
            let c2 = b1.interleave_upper_block_16(b3);
            let c3 = b2.interleave_lower_block_16(b4);
            let c4 = b2.interleave_upper_block_16(b4);

            // 2x2 Swap
            let mask2x2 = Simd::splat(0x3333_3333);
            let swap1 = (c1.shift_right::<2>() ^ c3) & mask2x2;
            let swap2 = (c2.shift_right::<2>() ^ c4) & mask2x2;
            let d1 = c1 ^ swap1.shift_left::<2>();
            let d2 = c2 ^ swap2.shift_left::<2>();
            let d3 = c3 ^ swap1;
            let d4 = c4 ^ swap2;

            // 1x1 Swap
            let mask1x1 = Simd::splat(0x5555_5555);
            let swap1 = (d1.shift_right::<1>() ^ d2) & mask1x1;
            let swap2 = (d3.shift_right::<1>() ^ d4) & mask1x1;
            let e1 = d1 ^ swap1.shift_left::<1>();
            let e2 = d2 ^ swap1;
            let e3 = d3 ^ swap2.shift_left::<1>();
            let e4 = d4 ^ swap2;

            // 4x4 Swap
            let f1 = e1.interleave_even_128(e2);
            let f2 = e3.interleave_even_128(e4);
            let f3 = e1.interleave_odd_128(e2);
            let f4 = e3.interleave_odd_128(e4);
            let mask4x4 = Simd::splat(0x0F0F_0F0F);
            let swap1 = (f1.shift_right::<4>() ^ f3) & mask4x4;
            let swap2 = (f2.shift_right::<4>() ^ f4) & mask4x4;
            let g1 = f1 ^ swap1.shift_left::<4>();
            let g2 = f2 ^ swap2.shift_left::<4>();
            let g3 = f3 ^ swap1;
            let g4 = f4 ^ swap2;
            let h1 = g1.interleave_even_128(g3);
            let h2 = g1.interleave_odd_128(g3);
            let h3 = g2.interleave_even_128(g4);
            let h4 = g2.interleave_odd_128(g4);

            // Unpack
            let i1 = h1.interleave_lower_block_32(h2);
            let i2 = h3.interleave_lower_block_32(h4);
            let i3 = h1.interleave_upper_block_32(h2);
            let i4 = h3.interleave_upper_block_32(h4);
            let j1 = i1.interleave_lower_block_64(i2);
            let j2 = i1.interleave_upper_block_64(i2);
            let j3 = i3.interleave_lower_block_64(i4);
            let j4 = i3.interleave_upper_block_64(i4);

            j1.copy_to_slice(&mut slice[..8]);
            j2.copy_to_slice(&mut slice[8..16]);
            j3.copy_to_slice(&mut slice[16..24]);
            j4.copy_to_slice(&mut slice[24..32]);
        }
        // avx512f, avx512bw, avx512vl
        64 => {
            let a1 = Simd::<u32, A>::from_slice(&slice[..16]);
            let a2 = Simd::<u32, A>::from_slice(&slice[16..32]);

            let b1 = a1.interleave_lower_block_8(a2);
            let b2 = a1.interleave_upper_block_8(a2);

            // 2x2 Swap
            let mask2x2 = Simd::splat(0x3333_3333);
            let c1 = (b1 & mask2x2) | b2.shift_left::<2>().andnot(mask2x2);
            let c2 = (b2.andnot(mask2x2)) | b1.shift_right::<2>() & mask2x2;

            let d1 = c1.interleave_lower_block_16(c2);
            let d2 = c1.interleave_upper_block_16(c2);

            // 1x1 Swap
            let mask1x1 = Simd::splat(0x5555_5555);
            let e1 = (d1 & mask1x1) | d2.shift_left::<1>().andnot(mask1x1);
            let e2 = (d2.andnot(mask1x1)) | d1.shift_right::<1>() & mask1x1;

            let f1 = e1.interleave_lower_block_16(e2);
            let f2 = e1.interleave_upper_block_16(e2);
            let g1 = f1.concat_lo_lo(f2);
            let g2 = f1.concat_hi_hi(f2);
            let h1 = g1.interleave_lower_block_8(g2);
            let h2 = g1.interleave_upper_block_8(g2);
            let i1 = h1.deinterleave_even_128(h2);
            let i2 = h1.deinterleave_odd_128(h2);

            // 4x4 Swap
            let mask4x4 = Simd::splat(0x0F0F_0F0F);
            let j1 = (i1 & mask4x4) | i2.shift_left::<4>().andnot(mask4x4);
            let j2 = (i2.andnot(mask4x4)) | i1.shift_right::<4>() & mask4x4;

            let k1 = j1.interleave_even_128(j2);
            let k2 = j1.interleave_odd_128(j2);

            k1.copy_to_slice(&mut slice[0..16]);
            k2.copy_to_slice(&mut slice[16..32]);
        }
        _ => unreachable!(),
    }
}

pub fn inner_transpose_slice_scalar(slice: &mut [u32]) {
    for j in 0..32 {
        for k in (j + 1)..32 {
            let swap = ((slice[j] >> k) ^ (slice[k] >> j)) & 1;
            slice[j] ^= swap << k;
            slice[k] ^= swap << j;
        }
    }
}
