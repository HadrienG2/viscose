use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use sched_local::{flags::AtomicFlags, pool::FlatPool};
use std::sync::atomic::Ordering;

fn criterion_benchmark(c: &mut Criterion) {
    bench_flags(c);
    bench_fibonacci(c);
}

fn bench_flags(c: &mut Criterion) {
    for len_pow2 in 0..=8 {
        let len = 2usize.pow(len_pow2);
        let flags = AtomicFlags::new(len);
        let header = format!("flags/{len}");

        // General logic for benchmarks that target a specific index
        fn bench_indexed_op<R>(
            c: &mut Criterion,
            flags: &AtomicFlags,
            group_name: &str,
            mut op: impl FnMut(&AtomicFlags, usize) -> R,
        ) {
            use pessimize::hide;
            let mut group = c.benchmark_group(group_name);
            group.bench_function("first", |b| b.iter(|| op(hide(flags), hide(0))));
            group.bench_function("center", |b| {
                let center = flags.len() / 2;
                b.iter(|| op(hide(flags), hide(center)))
            });
            group.bench_function("last", |b| {
                let last = flags.len() - 1;
                b.iter(|| op(hide(flags), hide(last)))
            });
        }

        // Operations that test a single bit
        bench_indexed_op(c, &flags, &format!("{header}/is_set"), |flags, pos| {
            flags.is_set(pos, Ordering::Relaxed)
        });
        bench_indexed_op(c, &flags, &format!("{header}/fetch_set"), |flags, pos| {
            flags.fetch_set(pos, Ordering::Relaxed)
        });
        bench_indexed_op(c, &flags, &format!("{header}/fetch_clear"), |flags, pos| {
            flags.fetch_clear(pos, Ordering::Relaxed)
        });

        // Operations that set all bits to the same value
        c.bench_function(&format!("{header}/set_all"), |b| {
            b.iter(|| pessimize::hide(&flags).set_all(Ordering::Relaxed))
        });
        c.bench_function(&format!("{header}/clear_all"), |b| {
            b.iter(|| pessimize::hide(&flags).clear_all(Ordering::Relaxed))
        });

        // Benchark the iterators over set and unset indices
        fn bench_iterator<Item>(
            c: &mut Criterion,
            flags: &AtomicFlags,
            name_header: &str,
            next: impl Fn(&AtomicFlags, usize) -> Option<Item>,
            count: impl Fn(&AtomicFlags, usize) -> usize,
        ) {
            flags.set_all(Ordering::Relaxed);
            bench_indexed_op(c, flags, &format!("{name_header}/ones/once"), &next);
            bench_indexed_op(c, flags, &format!("{name_header}/ones/all"), &count);
            flags.clear_all(Ordering::Relaxed);
            bench_indexed_op(c, flags, &format!("{name_header}/zeroes/once"), &next);
            bench_indexed_op(c, flags, &format!("{name_header}/zeroes/all"), &count);
        }
        bench_iterator(
            c,
            &flags,
            &format!("{header}/iter_set_around/inclusive"),
            |flags, pos| flags.iter_set_around::<true>(pos, Ordering::Relaxed).next(),
            |flags, pos| {
                flags
                    .iter_set_around::<true>(pos, Ordering::Relaxed)
                    .count()
            },
        );
        bench_iterator(
            c,
            &flags,
            &format!("{header}/iter_set_around/exclusive"),
            |flags, pos| {
                flags
                    .iter_set_around::<false>(pos, Ordering::Relaxed)
                    .next()
            },
            |flags, pos| {
                flags
                    .iter_set_around::<false>(pos, Ordering::Relaxed)
                    .count()
            },
        );
        bench_iterator(
            c,
            &flags,
            &format!("{header}/iter_unset_around/inclusive"),
            |flags, pos| {
                flags
                    .iter_unset_around::<true>(pos, Ordering::Relaxed)
                    .next()
            },
            |flags, pos| {
                flags
                    .iter_unset_around::<true>(pos, Ordering::Relaxed)
                    .count()
            },
        );
        bench_iterator(
            c,
            &flags,
            &format!("{header}/iter_unset_around/exclusive"),
            |flags, pos| {
                flags
                    .iter_unset_around::<false>(pos, Ordering::Relaxed)
                    .next()
            },
            |flags, pos| {
                flags
                    .iter_unset_around::<false>(pos, Ordering::Relaxed)
                    .count()
            },
        );
    }
}

fn bench_fibonacci(c: &mut Criterion) {
    fn bench_backend(
        c: &mut Criterion,
        backend_name: &str,
        mut bench_impl: impl FnMut(&mut Bencher, u64),
    ) {
        let group_name = format!("fibonacci/{backend_name}");
        let mut group = c.benchmark_group(group_name);
        for size in [1, 2, 4, 8, 16, 20, 24, 28, 30, 32, 34] {
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
