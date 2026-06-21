use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use cubic_bitfields::util::gen_rand_packed;
use cubic_bitfields::*;

fn store_data_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_bitfield");
    let mut array_u1: [u64; 512] = std::array::from_fn(|_| 0x0F0F0F0F0F0F0F0F);
    let mut array_u2: [u64; 1024] = std::array::from_fn(|_| 0x0F0F0F0F0F0F0F0F);
    let mut array_u4: [u64; 2048] = std::array::from_fn(|_| 0x0F0F0F0F0F0F0F0F);
    let mut array_u8: [u64; 4096] = std::array::from_fn(|_| 0x00FF00FF00FF00FF);
    let mut array_u16: [u64; 8192] = std::array::from_fn(|_| 0x0000FFFF0000FFFF);

    black_box(array_u1);
    black_box(array_u2);
    black_box(array_u4);
    black_box(array_u8);
    black_box(array_u16);

    let rand = gen_rand_packed::<512>(0);
    let mut bitfield = Bitfield::new(0);
    bitfield.load_packed_u1_into::<SET_ASSIGN, CMP_EQ>(&rand, true);

    group.throughput(Throughput::Elements(32768));
    group.bench_function("u1_store", |b| {
        b.iter(|| {
            bitfield.store_into_packed_u1(&mut array_u1, false);
            black_box(&bitfield);
        });
    });
    group.bench_function("u2_store", |b| {
        b.iter(|| {
            bitfield.store_into_packed_u2(&mut array_u2, 3);
            black_box(&bitfield);
        });
    });
    group.bench_function("u4_store", |b| {
        b.iter(|| {
            bitfield.store_into_packed_u4(&mut array_u4, 15);
            black_box(&bitfield);
        });
    });
    group.bench_function("u8_store", |b| {
        b.iter(|| {
            bitfield.store_into_packed_u8(&mut array_u8, 100);
            black_box(&bitfield);
        });
    });
    group.bench_function("u16_store", |b| {
        b.iter(|| {
            bitfield.store_into_packed_u16(&mut array_u16, 100);
            black_box(&bitfield);
        });
    });
}

criterion_group!(benches, store_data_bench);
criterion_main!(benches);
