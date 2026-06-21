use std::mem::transmute;

use cubic_bitfields::util::{gen_alternating_value, gen_rand_packed};
use cubic_bitfields::*;

#[test]
fn from_packed_u1_test() {
    let array = gen_rand_packed::<512>(0);
    let mut bitfield = Bitfield::new(0);
    bitfield.load_packed_u1_into::<SET_ASSIGN, CMP_EQ>(&array, true);
    let array_32: [u32; 1024] = unsafe { transmute(array) };
    assert_eq!(array_32, *bitfield.as_array());
}

#[test]
fn from_packed_u2_test() {
    let array = gen_rand_packed::<1024>(0);
    let mut bitfield = Bitfield::new(0);
    bitfield.load_packed_u2_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    let b_array = bitfield.to_array_64();

    for i in 0..512 {
        for bit in 0..64 {
            let bitfield_truth = ((b_array[i] >> bit) & 1) == 1;

            let src_idx = i * 2 + bit / 32;
            let src_bit = (bit % 32) * 2;
            let src_truth = ((array[src_idx] >> src_bit) & 0b11) == 0;
            assert_eq!(bitfield_truth, src_truth);
        }
    }
}

#[test]
fn from_packed_u4_test() {
    let array = gen_rand_packed::<2048>(0);
    let mut bitfield = Bitfield::new(0);
    bitfield.load_packed_u4_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    let b_array = bitfield.to_array_64();

    for i in 0..512 {
        for bit in 0..64 {
            let bitfield_truth = ((b_array[i] >> bit) & 1) == 1;

            let src_idx = i * 4 + bit / 16;
            let src_bit = (bit % 16) * 4;
            let src_truth = ((array[src_idx] >> src_bit) & 0xF) == 0;
            assert_eq!(bitfield_truth, src_truth);
        }
    }
}

#[test]
fn from_packed_u8_test() {
    let array: [u64; 4096] = std::array::from_fn(|_| 0xFFFF0000FFFF0000);
    let mut bitfield = Bitfield::new(0);
    bitfield.load_packed_u8_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    let b_array = bitfield.to_array_64();

    for i in 0..512 {
        for bit in 0..64 {
            let bitfield_truth = ((b_array[i] >> bit) & 1) == 1;

            let src_idx = i * 8 + bit / 8;
            let src_bit = (bit % 8) * 8;
            let src_truth = ((array[src_idx] >> src_bit) & 0xFF) == 0;
            assert_eq!(bitfield_truth, src_truth);
        }
    }
}

#[test]
fn from_packed_u16_test() {
    let array: [u64; 8192] = std::array::from_fn(|_| 0xFFFF0000FFFF0000);
    let mut bitfield = Bitfield::new(0);
    bitfield.load_packed_u16_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    let b_array = bitfield.to_array_64();

    for i in 0..512 {
        for bit in 0..64 {
            let bitfield_truth = ((b_array[i] >> bit) & 1) == 1;

            let src_idx = i * 16 + bit / 4;
            let src_bit = (bit % 4) * 16;
            let src_truth = ((array[src_idx] >> src_bit) & 0xFFFF) == 0;
            assert_eq!(bitfield_truth, src_truth);
        }
    }
}

#[test]
fn from_yz_packed_u1_test() {
    let array = gen_rand_packed::<512>(0);

    let mut bitfield1 = Bitfield::new(0);
    let mut bitfield2 = Bitfield::new(0);

    bitfield1.load_packed_u1_into::<SET_ASSIGN, CMP_EQ>(&array, false);
    bitfield2.load_yz_packed_u1_into::<SET_ASSIGN, CMP_EQ>(&array, 31, false);
    *bitfield1.outer_transpose().inner_transpose() >>= 31;
    *bitfield2.outer_transpose().inner_transpose() >>= 31;

    bitfield1.load_packed_u1_into::<SET_ASSIGN, CMP_EQ>(&array, false);
    bitfield2.load_yz_packed_u1_into::<SET_ASSIGN, CMP_EQ>(&array, 0, false);
    *bitfield1.outer_transpose().inner_transpose() <<= 31;
    *bitfield2.outer_transpose().inner_transpose() <<= 31;

    assert_eq!(bitfield1, bitfield2);
}

#[test]
fn from_yz_packed_u2_test() {
    let array = gen_rand_packed::<1024>(0);

    let mut bitfield1 = Bitfield::new(0);
    let mut bitfield2 = Bitfield::new(0);

    bitfield1.load_packed_u2_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    bitfield2.load_yz_packed_u2_into::<SET_ASSIGN, CMP_EQ>(&array, 31, 0);
    *bitfield1.outer_transpose().inner_transpose() >>= 31;
    *bitfield2.outer_transpose().inner_transpose() >>= 31;

    bitfield1.load_packed_u2_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    bitfield2.load_yz_packed_u2_into::<SET_ASSIGN, CMP_EQ>(&array, 0, 0);
    *bitfield1.outer_transpose().inner_transpose() <<= 31;
    *bitfield2.outer_transpose().inner_transpose() <<= 31;

    assert_eq!(bitfield1, bitfield2);
}

#[test]
fn from_yz_packed_u4_test() {
    let array = gen_rand_packed::<2048>(0);

    let mut bitfield1 = Bitfield::new(0);
    let mut bitfield2 = Bitfield::new(0);

    bitfield1.load_packed_u4_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    bitfield2.load_yz_packed_u4_into::<SET_ASSIGN, CMP_EQ>(&array, 31, 0);
    *bitfield1.outer_transpose().inner_transpose() >>= 31;
    *bitfield2.outer_transpose().inner_transpose() >>= 31;

    bitfield1.load_packed_u4_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    bitfield2.load_yz_packed_u4_into::<SET_ASSIGN, CMP_EQ>(&array, 0, 0);
    *bitfield1.outer_transpose().inner_transpose() <<= 31;
    *bitfield2.outer_transpose().inner_transpose() <<= 31;

    assert_eq!(bitfield1, bitfield2);
}

#[test]
fn from_yz_packed_u8_test() {
    let array: [u64; 4096] = std::array::from_fn(|_| 0xFFFF0000FFFF0000);

    let mut bitfield1 = Bitfield::new(0);
    let mut bitfield2 = Bitfield::new(0);

    bitfield1.load_packed_u8_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    bitfield2.load_yz_packed_u8_into::<SET_ASSIGN, CMP_EQ>(&array, 31, 0);
    *bitfield1.outer_transpose().inner_transpose() >>= 31;
    *bitfield2.outer_transpose().inner_transpose() >>= 31;

    bitfield1.load_packed_u8_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    bitfield2.load_yz_packed_u8_into::<SET_ASSIGN, CMP_EQ>(&array, 0, 0);
    *bitfield1.outer_transpose().inner_transpose() <<= 31;
    *bitfield2.outer_transpose().inner_transpose() <<= 31;

    assert_eq!(bitfield1, bitfield2);
}

#[test]
fn from_yz_packed_u16_test() {
    let array: [u64; 8192] = std::array::from_fn(|_| 0xFFFF0000FFFF0000);

    let mut bitfield1 = Bitfield::new(0);
    let mut bitfield2 = Bitfield::new(0);

    bitfield1.load_packed_u16_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    bitfield2.load_yz_packed_u16_into::<SET_ASSIGN, CMP_EQ>(&array, 31, 0);
    *bitfield1.outer_transpose().inner_transpose() >>= 31;
    *bitfield2.outer_transpose().inner_transpose() >>= 31;

    bitfield1.load_packed_u16_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    bitfield2.load_yz_packed_u16_into::<SET_ASSIGN, CMP_EQ>(&array, 0, 0);
    *bitfield1.outer_transpose().inner_transpose() <<= 31;
    *bitfield2.outer_transpose().inner_transpose() <<= 31;

    assert_eq!(bitfield1, bitfield2);
}

#[test]
fn from_xz_packed_u1_test() {
    let array = gen_rand_packed::<512>(0);

    let mut bitfield1 = Bitfield::new(0);
    let mut bitfield2 = Bitfield::new(0);

    bitfield1.load_packed_u1_into::<SET_ASSIGN, CMP_EQ>(&array, false);
    bitfield2.load_xz_packed_u1_into::<SET_ASSIGN, CMP_EQ>(&array, 31, false);
    *bitfield1.inner_transpose() >>= 31;
    *bitfield2.inner_transpose() >>= 31;

    bitfield1.load_packed_u1_into::<SET_OR, CMP_EQ>(&array, false);
    bitfield2.load_xz_packed_u1_into::<SET_OR, CMP_EQ>(&array, 0, false);
    *bitfield1.inner_transpose() <<= 31;
    *bitfield2.inner_transpose() <<= 31;

    assert_eq!(bitfield1, bitfield2);
}

#[test]
fn from_xz_packed_u2_test() {
    let array = gen_rand_packed::<1024>(0);

    let mut bitfield1 = Bitfield::new(0);
    let mut bitfield2 = Bitfield::new(0);

    bitfield1.load_packed_u2_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    bitfield2.load_xz_packed_u2_into::<SET_ASSIGN, CMP_EQ>(&array, 31, 0);
    *bitfield1.inner_transpose() >>= 31;
    *bitfield2.inner_transpose() >>= 31;

    bitfield1.load_packed_u2_into::<SET_OR, CMP_EQ>(&array, 0);
    bitfield2.load_xz_packed_u2_into::<SET_OR, CMP_EQ>(&array, 0, 0);
    *bitfield1.inner_transpose() <<= 31;
    *bitfield2.inner_transpose() <<= 31;

    assert_eq!(bitfield1, bitfield2);
}

#[test]
fn from_xz_packed_u4_test() {
    let array = gen_rand_packed::<2048>(0);

    let mut bitfield1 = Bitfield::new(0);
    let mut bitfield2 = Bitfield::new(0);

    bitfield1.load_packed_u4_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    bitfield2.load_xz_packed_u4_into::<SET_ASSIGN, CMP_EQ>(&array, 31, 0);
    *bitfield1.inner_transpose() >>= 31;
    *bitfield2.inner_transpose() >>= 31;

    bitfield1.load_packed_u4_into::<SET_OR, CMP_EQ>(&array, 0);
    bitfield2.load_xz_packed_u4_into::<SET_OR, CMP_EQ>(&array, 0, 0);
    *bitfield1.inner_transpose() <<= 31;
    *bitfield2.inner_transpose() <<= 31;

    assert_eq!(bitfield1, bitfield2);
}

#[test]
fn from_xz_packed_u8_test() {
    let array = gen_alternating_value(0xFFFF0000FFFF0000, 0x0000FFFF0000FFFF);

    let mut bitfield1 = Bitfield::new(0);
    let mut bitfield2 = Bitfield::new(0);

    bitfield1.load_packed_u8_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    bitfield2.load_xz_packed_u8_into::<SET_ASSIGN, CMP_EQ>(&array, 31, 0);
    *bitfield1.inner_transpose() >>= 31;
    *bitfield2.inner_transpose() >>= 31;

    bitfield1.load_packed_u8_into::<SET_OR, CMP_EQ>(&array, 0);
    bitfield2.load_xz_packed_u8_into::<SET_OR, CMP_EQ>(&array, 0, 0);
    *bitfield1.inner_transpose() <<= 31;
    *bitfield2.inner_transpose() <<= 31;

    assert_eq!(bitfield1, bitfield2);
}

#[test]
fn from_xz_packed_u16_test() {
    let array = gen_alternating_value(0xFFFF0000FFFF0000, 0x0000FFFF0000FFFF);

    let mut bitfield1 = Bitfield::new(0);
    let mut bitfield2 = Bitfield::new(0);

    bitfield1.load_packed_u16_into::<SET_ASSIGN, CMP_EQ>(&array, 0);
    bitfield2.load_xz_packed_u16_into::<SET_ASSIGN, CMP_EQ>(&array, 31, 0);
    *bitfield1.inner_transpose() >>= 31;
    *bitfield2.inner_transpose() >>= 31;

    bitfield1.load_packed_u16_into::<SET_OR, CMP_EQ>(&array, 0);
    bitfield2.load_xz_packed_u16_into::<SET_OR, CMP_EQ>(&array, 0, 0);
    *bitfield1.inner_transpose() <<= 31;
    *bitfield2.inner_transpose() <<= 31;

    assert_eq!(bitfield1, bitfield2);
}

#[test]
fn store_packed_u1_test() {
    let cur = gen_rand_packed::<512>(0);
    let pre_array = gen_rand_packed::<512>(1);
    let mut post_array = pre_array;
    let mut bitfield = Bitfield::new(0);
    bitfield.load_packed_u1_into::<SET_ASSIGN, CMP_EQ>(&cur, true);
    bitfield.store_into_packed_u1(&mut post_array, true);

    let b_array = bitfield.to_array_64();
    for i in 0..512 {
        for bit in 0..64 {
            let cur_bit = ((b_array[i] >> bit) & 1) == 1;

            let post_val = (post_array[i] >> bit) & 1;
            let pre_val = (post_array[i] >> bit) & 1;

            if cur_bit {
                assert_eq!(post_val, 1, "Does not match index after store!");
            } else {
                assert_eq!(post_val, pre_val, "Does not match index before store!");
            }
        }
    }
}

#[test]
fn store_packed_u2_test() {
    let cur = gen_rand_packed::<1024>(0);
    let pre_array = gen_rand_packed::<1024>(1);
    let mut post_array = pre_array;
    let mut bitfield = Bitfield::new(0);

    const INDEX: u8 = 3;
    bitfield.load_packed_u2_into::<SET_ASSIGN, CMP_EQ>(&cur, INDEX);
    bitfield.store_into_packed_u2(&mut post_array, INDEX);

    let b_array = bitfield.to_array_64();
    for i in 0..512 {
        for bit in 0..64 {
            let cur_bit = ((b_array[i] >> bit) & 1) == 1;

            let src_idx = i * 2 + bit / 32;
            let src_bit = (bit % 32) * 2;
            let post_val = (post_array[src_idx] >> src_bit) & 0b11;
            let pre_val = (pre_array[src_idx] >> src_bit) & 0b11;

            if cur_bit {
                assert_eq!(post_val, INDEX as u64, "Does not match index after store!");
            } else {
                assert_eq!(post_val, pre_val, "Does not match index before store!");
            }
        }
    }
}

#[test]
fn store_packed_u4_test() {
    let cur = gen_rand_packed::<2048>(0);
    let pre_array = gen_rand_packed::<2048>(1);
    let mut post_array = pre_array;
    let mut bitfield = Bitfield::new(0);

    const INDEX: u8 = 15;
    bitfield.load_packed_u4_into::<SET_ASSIGN, CMP_EQ>(&cur, INDEX);
    bitfield.store_into_packed_u4(&mut post_array, INDEX);

    let b_array = bitfield.to_array_64();
    for i in 0..512 {
        for bit in 0..64 {
            let cur_bit = ((b_array[i] >> bit) & 1) == 1;

            let src_idx = i * 4 + bit / 16;
            let src_bit = (bit % 16) * 4;
            let post_val = (post_array[src_idx] >> src_bit) & 0xF;
            let pre_val = (pre_array[src_idx] >> src_bit) & 0xF;

            if cur_bit {
                assert_eq!(post_val, INDEX as u64, "Does not match index after store!");
            } else {
                assert_eq!(post_val, pre_val, "Does not match index before store!");
            }
        }
    }
}

#[test]
fn store_packed_u8_test() {
    let cur = gen_rand_packed::<512>(0);
    let pre_array = gen_alternating_value(0xFFFF0000FFFF0000, 0x0000FFFF0000FFFF);
    let mut post_array = pre_array;
    let mut bitfield = Bitfield::new(0);

    const INDEX: u8 = 100;
    bitfield.load_packed_u1_into::<SET_ASSIGN, CMP_EQ>(&cur, true);
    bitfield.store_into_packed_u8(&mut post_array, INDEX);

    let b_array = bitfield.to_array_64();
    for i in 0..512 {
        for bit in 0..64 {
            let cur_bit = ((b_array[i] >> bit) & 1) == 1;

            let src_idx = i * 8 + bit / 8;
            let src_bit = (bit % 8) * 8;
            let post_val = (post_array[src_idx] >> src_bit) & 0xFF;
            let pre_val = (pre_array[src_idx] >> src_bit) & 0xFF;

            if cur_bit {
                assert_eq!(post_val, INDEX as u64, "Does not match index after store!");
            } else {
                assert_eq!(post_val, pre_val, "Does not match index before store!");
            }
        }
    }
}

#[test]
fn store_packed_u16_test() {
    let cur = gen_rand_packed::<512>(0);
    let pre_array = gen_alternating_value(0xFFFF0000FFFF0000, 0x0000FFFF0000FFFF);
    let mut post_array = pre_array;
    let mut bitfield = Bitfield::new(0);

    const INDEX: u16 = 100;
    bitfield.load_packed_u1_into::<SET_ASSIGN, CMP_EQ>(&cur, true);
    bitfield.store_into_packed_u16(&mut post_array, INDEX);

    let b_array = bitfield.to_array_64();
    for i in 0..512 {
        for bit in 0..64 {
            let cur_bit = ((b_array[i] >> bit) & 1) == 1;

            let src_idx = i * 16 + bit / 4;
            let src_bit = (bit % 4) * 16;
            let post_val = (post_array[src_idx] >> src_bit) & 0xFFFF;
            let pre_val = (pre_array[src_idx] >> src_bit) & 0xFFFF;

            if cur_bit {
                assert_eq!(post_val, INDEX as u64, "Does not match index after store!");
            } else {
                assert_eq!(post_val, pre_val, "Does not match index before store!");
            }
        }
    }
}
