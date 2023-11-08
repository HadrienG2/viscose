use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use sched_local::{pool::FlatPool, LocalFloats, LocalFloatsSlice};

fn criterion_benchmark(c: &mut Criterion) {
    fn bench_backend<const BLOCK_SIZE: usize>(
        c: &mut Criterion,
        group_name: &str,
        mut bench_impl: impl FnMut(&mut Bencher, &mut LocalFloatsSlice<'_, BLOCK_SIZE>),
    ) {
        // 2^22 floats is 2^24 bytes = 16 MiB, which is enough to saturate the
        // L3 of the largest computer I'm currently testing on. Tune this up for
        // testing on larger compute nodes with huge L3 caches.
        // FIXME: ...or better, autotune this using hwlocality cache stats
        const MAX_DATASET_SIZE_POW2: u32 = 22;
        assert!(BLOCK_SIZE.is_power_of_two());
        let block_size_pow2 = BLOCK_SIZE.trailing_zeros();
        let mut group = c.benchmark_group(group_name);
        for num_blocks_pow2 in 0..=MAX_DATASET_SIZE_POW2 - block_size_pow2 {
            let num_blocks = 1usize << num_blocks_pow2;
            let mut data = LocalFloats::<BLOCK_SIZE>::new(num_blocks);
            group.throughput(criterion::Throughput::Elements(
                (num_blocks * BLOCK_SIZE) as _,
            ));
            group.bench_function(&format!("{num_blocks}x{BLOCK_SIZE}"), |b| {
                bench_impl(b, &mut data.as_slice())
            });
        }
    }

    macro_rules! bench_square {
        ($($block_size_pow2:expr),*) => {$(
            {
                const BLOCK_SIZE: usize = 1usize << $block_size_pow2;

                bench_backend::<BLOCK_SIZE>(c, "square/rayon", |b: &mut Bencher, slice| {
                    b.iter(|| square_rayon(pessimize::hide(slice)))
                });

                let pool = FlatPool::new();
                bench_backend::<BLOCK_SIZE>(c, "square/flat", |b: &mut Bencher, slice| {
                    pool.run(|scope| b.iter(|| sched_local::square_flat(scope, pessimize::hide(slice))))
                });
            }
        )*};
    }
    bench_square!(13, 14, 15, 16);
}

fn square_rayon<const BLOCK_SIZE: usize>(slice: &mut LocalFloatsSlice<'_, BLOCK_SIZE>) {
    slice.process(
        |[mut left, mut right]| {
            rayon::join(|| square_rayon(&mut left), || square_rayon(&mut right));
        },
        |block, _locality| {
            block.iter_mut().for_each(|elem| *elem = elem.powi(2));
        },
        (),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
