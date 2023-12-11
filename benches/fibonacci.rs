use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use viscose::bench;

fn criterion_benchmark(c: &mut Criterion) {
    bench::for_each_locality(
        |rayon_name, mut make_rayon_pool, our_name, mut make_our_pool| {
            fn bench_backend(
                c: &mut Criterion,
                backend_name: &str,
                mut bench_impl: impl FnMut(&mut Bencher, u64),
            ) {
                let mut group = c.benchmark_group(&format!("{backend_name}/fibonacci"));
                for size in [1, 2, 4, 8, 12, 16, 20, 24, 28, 30, 32] {
                    let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
                    group.throughput(criterion::Throughput::Elements(
                        phi.powi(i32::try_from(size).unwrap()) as u64,
                    ));
                    group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, size| {
                        bench_impl(b, *size)
                    });
                }
            }
            {
                let rayon_pool = make_rayon_pool();
                bench_backend(c, rayon_name, |b: &mut Bencher, size| {
                    rayon_pool.install(|| b.iter(|| bench::fibonacci_rayon(pessimize::hide(size))))
                });
            }
            {
                let our_pool = make_our_pool();
                bench_backend(c, our_name, |b: &mut Bencher, size| {
                    our_pool
                        .run(|scope| b.iter(|| bench::fibonacci_ours(scope, pessimize::hide(size))))
                })
            }
        },
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
