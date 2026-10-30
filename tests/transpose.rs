use cubic_bitfields::Bitfield;
#[test]
fn outer_transpose_test() {
    let mut field1 = Bitfield::new(0);
    let mut field2 = Bitfield::new(0);
    for i in 0..1024 {
        field1.as_mut_slice()[i] = i as u32 * 12341;
        field2.as_mut_slice()[i] = i as u32 * 12341;
    }

    field1.outer_transpose();
    field2.outer_transpose_scalar();

    assert!(field1 == field2);
}

#[test]
fn inner_transpose_test() {
    let mut field1 = Bitfield::new(0);
    let mut field2 = Bitfield::new(0);
    for i in 0..1024 {
        field1.as_mut_slice()[i] = i as u32 * 12341;
        field2.as_mut_slice()[i] = i as u32 * 12341;
    }

    field1.inner_transpose();
    field2.inner_transpose_scalar();

    assert!(field1 == field2);
}

#[test]
fn inner_transpose_test_simply_simd() {
    let mut field1 = Bitfield::new(0);
    let mut field2 = Bitfield::new(0);
    for i in 0..1024 {
        field1.as_mut_slice()[i] = i as u32 * 12341;
        field2.as_mut_slice()[i] = i as u32 * 12341;
    }

    field1.inner_transpose_simply_simd();
    field2.inner_transpose_scalar();

    assert!(field1 == field2);
}
