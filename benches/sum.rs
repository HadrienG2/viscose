use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use viscose::bench::{self, bench_local_floats};

fn criterion_benchmark(c: &mut Criterion) {
    bench::for_each_locality(
        |rayon_name, mut make_rayon_pool, our_name, mut make_our_pool| {
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
