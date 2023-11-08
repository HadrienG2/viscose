use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use iterator_ilp::IteratorILP;
use sched_local::{pool::FlatPool, LocalFloats, LocalFloatsSlice};

fn criterion_benchmark(c: &mut Criterion) {
    fn bench_backend<const BLOCK_SIZE: usize>(
        c: &mut Criterion,
        group_name: &str,
        mut bench_impl: impl FnMut(&mut Bencher, &mut LocalFloatsSlice<'_, BLOCK_SIZE>),
    ) {
        // 2^21 floats is 2^23 bytes = 8 MiB, which is enough to saturate the
        // laptop on which I'm currently testing. Tune this up for testing on
        // larger compute nodes with huge L3 caches.
        const MAX_DATASET_SIZE_POW2: u32 = 21;
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

    macro_rules! bench_sum {
        () => {
            // I picked these values because...
            // - 4 is the sweet spot for hyperthreaded SSE
            // - 8 is the sweet spot for non-HT SSE and HT AVX
            // - 16 is the sweet spot for non-HT AVX
            bench_sum!(ilp[4, 8, 16]);
        };
        (
            ilp[$($ilp_streams:literal),*]
        ) => {$(
            // I picked these values because...
            // - Each float is 4 bytes
            // - The hyperthreaded sweet spot is <=16 KiB/thread -> 4096 floats
            // - The non-HT sweet spot is <=32 KiB/thread -> 8192 floats
            bench_sum!(ilp$ilp_streams/block_pow2[11, 12, 13]);
        )*};
        (
            ilp$ilp_streams:literal/block_pow2[$($block_size_pow2:literal),*]
        ) => {$({
            const BLOCK_SIZE: usize = 1usize << $block_size_pow2;
            const ILP_STREAMS: usize = $ilp_streams;

            bench_backend::<BLOCK_SIZE>(
                c,
                &format!("sum/rayon/ilp{ILP_STREAMS}"),
                |b: &mut Bencher, slice| {
                    b.iter(|| sum_rayon::<BLOCK_SIZE, ILP_STREAMS>(pessimize::hide(slice)))
                },
            );

            let pool = FlatPool::new();
            bench_backend::<BLOCK_SIZE>(
                c,
                &format!("sum/flat/ilp{ILP_STREAMS}"),
                |b: &mut Bencher, slice| {
                    pool.run(|scope| {
                        b.iter(|| {
                            sched_local::sum_flat::<BLOCK_SIZE, ILP_STREAMS>(
                                scope,
                                pessimize::hide(slice),
                            )
                        })
                    })
                },
            );
        })*};
    }
    bench_sum!();

    // TODO: Also bench norm_sqr
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
