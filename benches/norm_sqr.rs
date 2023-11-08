use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use iterator_ilp::IteratorILP;
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

    macro_rules! bench_norm_sqr {
        () => {
            // I picked these values because...
            // - 4 is the sweet spot for hyperthreaded SSE
            // - 8 is the sweet spot for non-HT SSE and HT AVX
            // - 16 is the sweet spot for non-HT AVX
            bench_norm_sqr!(ilp[4, 8, 16]);
        };
        (
            ilp[$($ilp_streams:literal),*]
        ) => {$(
            // I picked these values because...
            // - Each float is 4 bytes
            // - The hyperthreaded sweet spot is <=16 KiB/thread -> 4096 floats
            // - The non-HT sweet spot is <=32 KiB/thread -> 8192 floats
            bench_norm_sqr!(ilp$ilp_streams/block_pow2[11, 12, 13]);
        )*};
        (
            ilp$ilp_streams:literal/block_pow2[$($block_size_pow2:literal),*]
        ) => {$({
            const BLOCK_SIZE: usize = 1usize << $block_size_pow2;
            const ILP_STREAMS: usize = $ilp_streams;

            bench_backend::<BLOCK_SIZE>(
                c,
                &format!("norm_sqr/rayon/ilp{ILP_STREAMS}"),
                |b: &mut Bencher, slice| {
                    b.iter(|| norm_sqr_rayon::<BLOCK_SIZE, ILP_STREAMS>(pessimize::hide(slice)))
                },
            );

            let pool = FlatPool::new();
            bench_backend::<BLOCK_SIZE>(
                c,
                &format!("norm_sqr/flat/ilp{ILP_STREAMS}"),
                |b: &mut Bencher, slice| {
                    pool.run(|scope| {
                        b.iter(|| {
                            sched_local::norm_sqr_flat::<BLOCK_SIZE, ILP_STREAMS>(
                                scope,
                                pessimize::hide(slice),
                            )
                        })
                    })
                },
            );
        })*};
    }
    bench_norm_sqr!();
}

fn norm_sqr_rayon<const BLOCK_SIZE: usize, const REDUCE_ILP_STREAMS: usize>(
    slice: &mut LocalFloatsSlice<'_, BLOCK_SIZE>,
) -> f32 {
    slice.process(
        |[mut left, mut right]| {
            rayon::join(
                || square_rayon::<BLOCK_SIZE>(&mut left),
                || square_rayon::<BLOCK_SIZE>(&mut right),
            );
            let (left, right) = rayon::join(
                || sum_rayon::<BLOCK_SIZE, REDUCE_ILP_STREAMS>(&mut left),
                || sum_rayon::<BLOCK_SIZE, REDUCE_ILP_STREAMS>(&mut right),
            );
            left + right
        },
        |block, _locality| {
            block.iter().copied().fold_ilp::<REDUCE_ILP_STREAMS, f32>(
                || 0.0,
                |acc, elem| {
                    if cfg!(target_feature = "fma") {
                        elem.mul_add(elem, acc)
                    } else {
                        acc + elem * elem
                    }
                },
                |acc1, acc2| acc1 + acc2,
            )
        },
        0.0,
    )
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

fn sum_rayon<const BLOCK_SIZE: usize, const ILP_STREAMS: usize>(
    slice: &mut LocalFloatsSlice<'_, BLOCK_SIZE>,
) -> f32 {
    slice.process(
        |[mut left, mut right]| {
            let (left, right) = rayon::join(
                || sum_rayon::<BLOCK_SIZE, ILP_STREAMS>(&mut left),
                || sum_rayon::<BLOCK_SIZE, ILP_STREAMS>(&mut right),
            );
            left + right
        },
        |block, _locality| block.iter().copied().sum_ilp::<ILP_STREAMS, f32>(),
        0.0,
    )
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
