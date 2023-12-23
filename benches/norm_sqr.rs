use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use viscose::bench::{self, bench_local_floats};

fn criterion_benchmark(c: &mut Criterion) {
    bench::for_each_locality(
        |rayon_name, mut make_rayon_pool, our_name, mut make_our_pool| {
            macro_rules! bench_norm_sqr {
            () => {
                // I picked these values because...
                // - 8 is the sweet spot for two SSE (4xf32) ALUs
                // - 16 is the sweet spot for two AVX (8xf32) ALUs
                //
                // Since I'm not currently testing in SSE mode, the ilp8 mode is
                // currently disabled for faster builds and runs.
                bench_norm_sqr!(ilp[/*8, */16]);
            };
            (
                ilp[$($ilp_streams:literal),*]
            ) => {$(
                // I picked these values because...
                // - Each f32 is 4 bytes
                // - 2^13 = 8192 f32 = 32 KiB = capacity of L1 cache, which
                //   would be the expected optimal sequential working set if
                //   work distribution overheads were not an issue.
                // - Typical L2 capacity is between 256 KiB and 1024 KiB, with
                //   lower being most common
                bench_norm_sqr!(ilp$ilp_streams/block_pow2[13, 16/*, 17, 18*/]);
            )*};
            (
                ilp$ilp_streams:literal/block_pow2[$($block_size_pow2:literal),*]
            ) => {$({
                const BLOCK_SIZE: usize = 1usize << $block_size_pow2;
                const ILP_STREAMS: usize = $ilp_streams;
                let bench_name = &format!("norm_sqr/ilp{ILP_STREAMS}");
                {
                    let rayon_pool = make_rayon_pool();
                    bench_local_floats::<BLOCK_SIZE>(
                        c,
                        bench_name,
                        rayon_name,
                        |b: &mut Bencher, slice| {
                            rayon_pool.install(|| {
                                b.iter(|| {
                                    bench::norm_sqr_rayon::<BLOCK_SIZE, ILP_STREAMS>(
                                        pessimize::hide(slice)
                                    )
                                })
                            })
                        },
                    );
                }
                {
                    let our_pool = make_our_pool();
                    bench_local_floats::<BLOCK_SIZE>(
                        c,
                        bench_name,
                        our_name,
                        |b: &mut Bencher, slice| {
                            our_pool.run(|scope| {
                                b.iter(|| {
                                    bench::norm_sqr_ours::<BLOCK_SIZE, ILP_STREAMS>(
                                        scope,
                                        pessimize::hide(slice),
                                    )
                                })
                            })
                        },
                    );
                }
            })*};
        }
            bench_norm_sqr!();
        },
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
