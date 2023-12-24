use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use viscose::bench::{self, bench_local_floats};

fn criterion_benchmark(c: &mut Criterion) {
    bench::for_each_locality(
        |rayon_name, mut make_rayon_pool, our_name, mut make_our_pool| {
            macro_rules! bench_sum {
            () => {
                // I picked these values because...
                // - 8 is the sweet spot for two SSE (4xf32) ALUs
                // - 16 is the sweet spot for two AVX (8xf32) ALUs
                //
                // Since I'm not currently testing in SSE mode, the ilp8 mode is
                // currently disabled for faster builds and runs.
                bench_sum!(ilp[/*8, */16]);
            };
            (
                ilp[$($ilp_streams:literal),*]
            ) => {$(
                // I picked these values because...
                // - Each f32 is 4 bytes
                // - 2^12 = 8192 f32 = 16 KiB = 1/2 capacity of L1 cache, which
                //   would be the expected optimal sequential working set if
                //   work distribution overheads were not an issue.
                // - Typical L2 capacity is between 256 KiB and 1024 KiB, with
                //   lower being most common, and again using half capacity is
                //   often best.
                bench_sum!(ilp$ilp_streams/block_pow2[12, 15/*, 16, 17, 18*/]);
            )*};
            (
                ilp$ilp_streams:literal/block_pow2[$($block_size_pow2:literal),*]
            ) => {$({
                const BLOCK_SIZE: usize = 1usize << $block_size_pow2;
                const ILP_STREAMS: usize = $ilp_streams;
                let bench_name = &format!("sum/ilp{ILP_STREAMS}");
                {
                    let rayon_pool = make_rayon_pool();
                    bench_local_floats::<BLOCK_SIZE>(
                        c,
                        bench_name,
                        rayon_name,
                        |b: &mut Bencher, slice| {
                            rayon_pool.install(|| {
                                b.iter(|| {
                                    bench::sum_rayon::<BLOCK_SIZE, ILP_STREAMS>(
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
                                    bench::sum_ours::<BLOCK_SIZE, ILP_STREAMS>(
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
            bench_sum!();
        },
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
