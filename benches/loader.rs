use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use cubic_bitfields::util::gen_sparse;
use cubic_bitfields::*;

fn load_data_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_bitfield");
    let array_u1: [u64; 512] = std::array::from_fn(|_| 0x0F0F0F0F0F0F0F0F);
    let array_u2: [u64; 1024] = std::array::from_fn(|_| 0x0F0F0F0F0F0F0F0F);
    let array_u4: [u64; 2048] = std::array::from_fn(|_| 0x0F0F0F0F0F0F0F0F);
    let array_u8: [u64; 4096] = std::array::from_fn(|_| 0x00FF00FF00FF00FF);
    let array_u16: [u64; 8192] = std::array::from_fn(|_| 0x0000FFFF0000FFFF);

    black_box(array_u1);
    black_box(array_u2);
    black_box(array_u4);
    black_box(array_u8);
    black_box(array_u16);

    let mut bitfield = Bitfield::new(0);
    let mut bitfield_new: Bitfield32 = Bitfield32::new(0);
    group.throughput(Throughput::Elements(32768));
    group.bench_function("u1_load", |b| {
        b.iter(|| {
            bitfield.load_packed_u1_into::<SET_ASSIGN, CMP_EQ>(&array_u1, false);
            black_box(&bitfield);
        });
    });
    group.bench_function("u1_load_new", |b| {
        b.iter(|| {
            bitfield_new.load_packed_u1_into::<SET_ASSIGN, CMP_EQ>(&array_u1, false);
            black_box(&bitfield);
        });
    });
    group.bench_function("u2_load", |b| {
        b.iter(|| {
            bitfield.load_packed_u2_into::<SET_ASSIGN, CMP_EQ>(&array_u2, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u2_load_new", |b| {
        b.iter(|| {
            bitfield_new.load_packed_u2_into::<SET_ASSIGN, CMP_EQ>(&array_u2, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u4_load", |b| {
        b.iter(|| {
            bitfield.load_packed_u4_into::<SET_ASSIGN, CMP_EQ>(&array_u4, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u4_load_new", |b| {
        b.iter(|| {
            bitfield_new.load_packed_u4_into::<SET_ASSIGN, CMP_EQ>(&array_u4, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u8_load", |b| {
        b.iter(|| {
            bitfield.load_packed_u8_into::<SET_ASSIGN, CMP_EQ>(&array_u8, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u8_load_new", |b| {
        b.iter(|| {
            bitfield_new.load_packed_u8_into::<SET_ASSIGN, CMP_EQ>(&array_u8, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u16_load", |b| {
        b.iter(|| {
            bitfield.load_packed_u16_into::<SET_ASSIGN, CMP_EQ>(&array_u16, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u16_load_new", |b| {
        b.iter(|| {
            bitfield_new.load_packed_u16_into::<SET_ASSIGN, CMP_EQ>(&array_u16, 0);
            black_box(&bitfield);
        });
    });

    group.bench_function("u1_yz_load", |b| {
        b.iter(|| {
            bitfield.load_yz_packed_u1_into::<SET_ASSIGN, CMP_EQ>(&array_u1, 31, false);
            black_box(&bitfield);
        });
    });
    group.bench_function("u2_yz_load", |b| {
        b.iter(|| {
            bitfield.load_yz_packed_u2_into::<SET_ASSIGN, CMP_EQ>(&array_u2, 31, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u4_yz_load", |b| {
        b.iter(|| {
            bitfield.load_yz_packed_u4_into::<SET_ASSIGN, CMP_EQ>(&array_u4, 31, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u8_yz_load", |b| {
        b.iter(|| {
            bitfield.load_yz_packed_u8_into::<SET_ASSIGN, CMP_EQ>(&array_u8, 31, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u16_yz_load", |b| {
        b.iter(|| {
            bitfield.load_yz_packed_u16_into::<SET_ASSIGN, CMP_EQ>(&array_u16, 31, 0);
            black_box(&bitfield);
        });
    });

    group.bench_function("u1_xz_load", |b| {
        b.iter(|| {
            bitfield.load_xz_packed_u1_into::<SET_ASSIGN, CMP_EQ>(&array_u1, 31, false);
            black_box(&bitfield);
        });
    });
    group.bench_function("u2_xz_load", |b| {
        b.iter(|| {
            bitfield.load_xz_packed_u2_into::<SET_ASSIGN, CMP_EQ>(&array_u2, 31, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u4_xz_load", |b| {
        b.iter(|| {
            bitfield.load_xz_packed_u4_into::<SET_ASSIGN, CMP_EQ>(&array_u4, 31, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u8_xz_load", |b| {
        b.iter(|| {
            bitfield.load_xz_packed_u8_into::<SET_ASSIGN, CMP_EQ>(&array_u8, 31, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u16_xz_load", |b| {
        b.iter(|| {
            bitfield.load_xz_packed_u16_into::<SET_ASSIGN, CMP_EQ>(&array_u16, 31, 0);
            black_box(&bitfield);
        });
    });
}

criterion_group!(benches, load_data_bench);
criterion_main!(benches);
