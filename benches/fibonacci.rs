use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use sched_local::pool::FlatPool;

fn criterion_benchmark(c: &mut Criterion) {
    fn bench_backend(
        c: &mut Criterion,
        backend_name: &str,
        mut bench_impl: impl FnMut(&mut Bencher, u64),
    ) {
        let group_name = format!("fibonacci/{backend_name}");
        let mut group = c.benchmark_group(group_name);
        for size in [1, 2, 4, 8, 12, 16, 20, 24, 28, 30, 32, 34] {
            let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
            group.throughput(criterion::Throughput::Elements(
                phi.powi(i32::try_from(size).unwrap()) as u64,
            ));
            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, size| {
                bench_impl(b, *size)
            });
        }
    }

    bench_backend(c, "rayon", |b: &mut Bencher, size| {
        b.iter(|| fibonacci_rayon(pessimize::hide(size)))
    });

    let pool = FlatPool::new();
    bench_backend(c, "flat", |b: &mut Bencher, size| {
        pool.run(|scope| b.iter(|| sched_local::fibonacci_flat(scope, pessimize::hide(size))))
    })
}

fn fibonacci_rayon(n: u64) -> u64 {
    if n > 1 {
        let (x, y) = rayon::join(|| fibonacci_rayon(n - 1), || fibonacci_rayon(n - 2));
        x + y
    } else {
        n
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
