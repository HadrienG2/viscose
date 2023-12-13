use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use viscose::bench::{self, bench_local_floats};

fn criterion_benchmark(c: &mut Criterion) {
    bench::for_each_locality(
        |rayon_name, mut make_rayon_pool, our_name, mut make_our_pool| {
            macro_rules! bench_square {
            () => {
                // I picked these values because...
                // - Each f32 is 4 bytes
                // - Hyperthreads share the L1 cache, so the theoretical HT
                //   sweet spot is to use half of it: 16 KiB/thread -> 4096 f32s
                // - The non-HT sweet spot is 32 KiB/thread -> 8192 f32s
                bench_square!(12, 13);
            };
            ($($block_size_pow2:expr),*) => {$(
                {
                    const BLOCK_SIZE: usize = 1usize << $block_size_pow2;
                    {
                        let rayon_pool = make_rayon_pool();
                        bench_local_floats::<BLOCK_SIZE>(c, "square", rayon_name, |b: &mut Bencher, slice| {
                            rayon_pool.install(|| b.iter(|| bench::square_rayon(pessimize::hide(slice))))
                        });
                    }
                    {
                        let our_pool = make_our_pool();
                        bench_local_floats::<BLOCK_SIZE>(c, "square", our_name, |b: &mut Bencher, slice| {
                            our_pool.run(|scope| b.iter(|| bench::square_ours(scope, pessimize::hide(slice))))
                        });
                    }
                }
            )*};
        }
            bench_square!();
        },
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
