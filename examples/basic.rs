use std::hint::black_box;

use cubic_bitfields::util::gen_sparse;
use cubic_bitfields::*;

fn main() {
    const NUM_BATCHES: usize = 1;

    let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);
    let array2 = gen_sparse::<2048>(1, NUM_BATCHES, 6, 60);

    let mut untracked1 = Bitfield::new(0);
    let mut untracked2 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);
    let mut tracked2 = TrackedBitfield::new(0);

    // untracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    // untracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    // tracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    // tracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);

    for x in 0..32 {
        for y in 0..32 {
            untracked1.as_mut_slice()[x * 32 + y] = (1 << (y as u32)) - 1;
            untracked2.as_mut_slice()[x * 32 + y] = (1 << (y as u32)) - 1;
        }
    }

    println!(" -=-=- Start -=-=- ");
    untracked1.print_inner_slices(0..1);

    untracked1.inner_transpose();
    untracked2.inner_transpose_simply_simd();

    println!(" -=-=- Solution -=-=- ");
    untracked1.print_inner_slices(0..1);

    println!(" -=-=- Simply Simd -=-=- ");
    untracked2.print_inner_slices(0..1);

    // // untracked1 |= untracked2;
    //
    // println!("array1: {:?}", array1);
    //
    // black_box(&tracked1);
    // black_box(&tracked2);
    //
    // tracked1 ^= tracked2;
    //
    // black_box(&tracked1);
    //
    // // assert_eq!(untracked1, tracked1.to_bitfield());
    //
    // let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);
    // let mut untracked1 = Bitfield::new(0);
    // let mut tracked1 = TrackedBitfield::new(0);
    // untracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    // tracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    // assert_eq!(
    //     untracked1,
    //     *tracked1.as_bitfield(),
    //     "Bitfield as not equal."
    // );
    //
    // let mut iter = tracked1.active_bit_iter();
    //
    // let true_array = untracked1.as_array();
    //
    // for (i, cur_entry) in true_array.iter().enumerate() {
    //     let mut entry = *cur_entry;
    //     while entry != 0 {
    //         let bit = entry.trailing_zeros();
    //         let val = i * 32 + bit as usize;
    //
    //         let iter_val = iter.next().expect("No next iter value??");
    //
    //         println!(
    //             "{:<20} {}",
    //             format!("true: {val}"),
    //             format!("iter: {iter_val}")
    //         );
    //         assert_eq!(val, iter_val);
    //
    //         entry &= entry - 1;
    //     }
    // }
    //
    // tracked1.inner_transpose();

    // black_box(&tracked1);
}
