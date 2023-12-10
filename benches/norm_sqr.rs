use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use viscose::bench::{self, bench_local_floats};

fn criterion_benchmark(c: &mut Criterion) {
    bench::for_each_locality(
        |rayon_name,
         mut make_rayon_pool,
         flat_name,
         mut make_flat_pool,
         hierarchical_name,
         mut make_hierarchical_pool| {
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
                    let flat_pool = make_flat_pool();
                    bench_local_floats::<BLOCK_SIZE>(
                        c,
                        bench_name,
                        flat_name,
                        |b: &mut Bencher, slice| {
                            flat_pool.run(|scope| {
                                b.iter(|| {
                                    bench::norm_sqr_pool::<BLOCK_SIZE, ILP_STREAMS, _>(
                                        scope,
                                        pessimize::hide(slice),
                                    )
                                })
                            })
                        },
                    );
                }
                {
                    let hierarchical_pool = make_hierarchical_pool();
                    bench_local_floats::<BLOCK_SIZE>(
                        c,
                        bench_name,
                        hierarchical_name,
                        |b: &mut Bencher, slice| {
                            hierarchical_pool.run(|scope| {
                                b.iter(|| {
                                    bench::norm_sqr_pool::<BLOCK_SIZE, ILP_STREAMS, _>(
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
