pub fn gen_rand_packed<const N: usize>(seed: u64) -> [u64; N] {
    let mut rand: u64 = 0x9E3779B97F4A7C15 ^ seed;
    std::array::from_fn(|_| {
        mix_bits(&mut rand);
        rand
    })
}

pub fn gen_alternating_value<const N: usize>(val1: u64, val2: u64) -> [u64; N] {
    let mut counter = 0;
    std::array::from_fn(|_| {
        counter += 1;
        if (counter % 2) == 0 { val1 } else { val2 }
    })
}

pub fn gen_sparse<const N: usize>(
    seed: u64,
    num_chunks: usize,
    min_size: usize,
    max_size: usize,
) -> [u64; N] {
    let mut array = [0; N];
    let mut rand: u64 = seed ^ 0x9E3779B97F4A7C15;

    for _ in 0..num_chunks {
        mix_bits(&mut rand);
        let start = rand as usize % N;

        mix_bits(&mut rand);
        let pre_size = min_size + rand as usize % (max_size - min_size);

        let size = if (pre_size + start) >= N {
            N.saturating_sub(start)
        } else {
            pre_size
        };

        for i in 0..size {
            mix_bits(&mut rand);
            array[start + i] ^= rand;
        }
    }

    array
}

fn mix_bits(data: &mut u64) {
    *data ^= *data >> 33;
    *data = data.wrapping_mul(0xff5afd7ed558ccd);
    *data ^= *data >> 33;
    *data = data.wrapping_mul(0xc4ceb9fea85ec53);
    *data ^= *data >> 33;
    *data = data.wrapping_mul(0x9e3779b97f4a7c15);
    *data ^= *data >> 33;
}
