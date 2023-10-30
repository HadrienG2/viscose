use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn fibonacci_rayon(n: u64) -> u64 {
    if n > 1 {
        let (x, y) = rayon::join(|| fibonacci_rayon(n - 1), || fibonacci_rayon(n - 2));
        x + y
    } else {
        n
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    fn bench_backend(c: &mut Criterion, backend_name: &str, mut fibonacci: impl FnMut(u64) -> u64) {
        let group_name = format!("{backend_name}/fibonacci");
        let mut group = c.benchmark_group(group_name);
        for size_pow2 in 0..=5 {
            let size = 2u64.pow(size_pow2);
            group.throughput(criterion::Throughput::Elements(size));
            group.bench_function(BenchmarkId::from_parameter(size), |b| {
                b.iter(|| fibonacci(black_box(size)))
            });
        }
    }
    bench_backend(c, "rayon", fibonacci_rayon);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
