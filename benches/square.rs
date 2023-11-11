use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use viscose::bench::{self, bench_local_floats};

fn criterion_benchmark(c: &mut Criterion) {
    bench::for_each_locality(
        |rayon_name, mut make_rayon_pool, flat_name, mut make_flat_pool| {
            macro_rules! bench_square {
            () => {
                // I picked these values because...
                // - Each float is 4 bytes
                // - The hyperthreaded sweet spot is <=16 KiB/thread -> 4096 floats
                // - The non-HT sweet spot is <=32 KiB/thread -> 8192 floats
                bench_square!(11, 12, 13);
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
                        let flat_pool = make_flat_pool();
                        bench_local_floats::<BLOCK_SIZE>(c, "square", flat_name, |b: &mut Bencher, slice| {
                            flat_pool.run(|scope| b.iter(|| bench::square_flat(scope, pessimize::hide(slice))))
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
