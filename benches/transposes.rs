use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use cubic_bitfields::util::gen_sparse;
use cubic_bitfields::*;

fn transpose_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitfield");
    let mut bitfield = Bitfield::new(12);
    black_box(bitfield);

    group.throughput(Throughput::Elements(1024));
    group.bench_function("inner_transpose", |b| {
        b.iter(|| {
            bitfield.inner_transpose();
            black_box(bitfield);
        });
    });
    group.bench_function("inner_transpose_simply_simd", |b| {
        b.iter(|| {
            bitfield.inner_transpose_simply_simd();
            black_box(bitfield);
        });
    });
    group.bench_function("outer_transpose", |b| {
        b.iter(|| {
            bitfield.outer_transpose();
            black_box(bitfield);
        });
    });
    group.bench_function("outer_transpose_scalar", |b| {
        b.iter(|| {
            bitfield.outer_transpose_scalar();
            black_box(bitfield);
        });
    });
    group.finish();
}

criterion_group!(benches, transpose_bench);
criterion_main!(benches);
