use std::hint::black_box;

use cubic_bitfields::util::gen_sparse;
use cubic_bitfields::*;

fn main() {
    let bitfield = Bitfield::new(1);

    // black_box(&bitfield);
    //
    // for z in 0..32 {
    //     for y in 0..32 {
    //         bitfield.as_mut_slice()[z * 32 + y] = u32::MAX >> (31 - y);
    //     }
    // }
    //
    // for i in 0..1024 {
    //     bitfield.as_mut_slice()[i] = (i * 12341) as u32;
    // }

    // for z in 0..32 {
    //     for y in (z + 1)..32 {
    //         bitfield.as_mut_slice()[y * 32 + z] = u32::MAX;
    //     }
    // }
    // let mut bitfield = Bitfield::new(0);
    //
    // let array_u1: [u64; 512] = std::array::from_fn(|_| 0xF0F0F0F0F0F0F0F0);
    // let array_u2: [u64; 1024] = std::array::from_fn(|_| 0xF0F0F0F0F0F0F0F0);
    // let array_u4: [u64; 2048] = std::array::from_fn(|_| 0x00FF00FF00FF00FF);
    // let array_u8: [u64; 4096] = std::array::from_fn(|_| 0x00FF00FF00FF00FF);
    // let array_u16: [u64; 8192] = std::array::from_fn(|_| 0x0000FFFF0000FFFF);
    // // let bitfield = black_box(Bitfield::from_packed_u16::<true>(&array_u16, black_box(0)));
    // // let bitfield = black_box(Bitfield::from_packed_u8::<true>(&array_u8, black_box(0)));
    // let bitfield1 = black_box(Bitfield::from_packed_u1::<true>(&array_u1, black_box(0)));
    // bitfield.load_packed_u2_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u2, 0);
    // let bitfield3 = black_box(Bitfield::from_packed_u4::<true>(&array_u4, black_box(0)));
    // let bitfield4 = black_box(Bitfield::from_packed_u8::<true>(&array_u8, black_box(0)));
    // let bitfield5 = black_box(Bitfield::from_packed_u16::<true>(&array_u16, black_box(0)));
    //
    // let bitfield1 = Bitfield::new(2);

    // black_box(black_box(bitfield1) ^ black_box(bitfield));

    // black_box(bitfield);
    // bitfield.print_inner_slices(0..32);
    // let array1 = gen_sparse::<2048>(1, 6, 6, 60);
    // println!("array: {:?}", array1);

    const NUM_BATCHES: usize = 1;

    let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);
    let array2 = gen_sparse::<2048>(1, NUM_BATCHES, 6, 60);

    let mut untracked1 = Bitfield::new(0);
    let mut untracked2 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);
    let mut tracked2 = TrackedBitfield::new(0);

    untracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    untracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);
    tracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    tracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);

    // untracked1 |= untracked2;

    println!("array1: {:?}", array1);

    black_box(&tracked1);
    black_box(&tracked2);

    tracked1 ^= tracked2;

    black_box(&tracked1);

    // assert_eq!(untracked1, tracked1.to_bitfield());

    let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);
    let mut untracked1 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);
    untracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    tracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    assert_eq!(
        untracked1,
        *tracked1.as_bitfield(),
        "Bitfield as not equal."
    );

    let mut iter = tracked1.active_bit_iter();

    let true_array = untracked1.as_array();

    for (i, cur_entry) in true_array.iter().enumerate() {
        let mut entry = *cur_entry;
        while entry != 0 {
            let bit = entry.trailing_zeros();
            let val = i * 32 + bit as usize;

            let iter_val = iter.next().expect("No next iter value??");

            println!(
                "{:<20} {}",
                format!("true: {val}"),
                format!("iter: {iter_val}")
            );
            assert_eq!(val, iter_val);

            entry &= entry - 1;
        }
    }
}
