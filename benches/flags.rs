use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use sched_local::flags::AtomicFlags;
use std::sync::atomic::Ordering;

fn criterion_benchmark(c: &mut Criterion) {
    for len_pow2 in 0..=8 {
        let len = 2usize.pow(len_pow2);
        let flags = AtomicFlags::new(len);
        let header = format!("flags/{len}");

        // General logic for benchmarks that target a specific index
        fn bench_indexed_op<R>(
            c: &mut Criterion,
            flags: &AtomicFlags,
            group_name: &str,
            num_inner_ops: usize,
            mut op: impl FnMut(&AtomicFlags, usize) -> R,
        ) {
            use pessimize::hide;
            let mut group = c.benchmark_group(group_name);
            group.throughput(Throughput::Elements(num_inner_ops as _));
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
        bench_indexed_op(c, &flags, &format!("{header}/is_set"), 1, |flags, pos| {
            flags.is_set(pos, Ordering::Relaxed)
        });
        bench_indexed_op(
            c,
            &flags,
            &format!("{header}/fetch_set"),
            1,
            |flags, pos| flags.fetch_set(pos, Ordering::Relaxed),
        );
        bench_indexed_op(
            c,
            &flags,
            &format!("{header}/fetch_clear"),
            1,
            |flags, pos| flags.fetch_clear(pos, Ordering::Relaxed),
        );

        // Operations that set all bits to the same value
        {
            let mut group = c.benchmark_group(&format!("{header}/all"));
            group.throughput(Throughput::Elements(flags.len() as _));
            group.bench_function("set", |b| {
                b.iter(|| pessimize::hide(&flags).set_all(Ordering::Relaxed))
            });
            group.bench_function("clear", |b| {
                b.iter(|| pessimize::hide(&flags).clear_all(Ordering::Relaxed))
            });
        }

        // Benchark the iterators over set and unset indices
        fn bench_iterator<Item>(
            c: &mut Criterion,
            flags: &AtomicFlags,
            name_header: &str,
            next: impl Fn(&AtomicFlags, usize) -> Option<Item>,
            count: impl Fn(&AtomicFlags, usize) -> usize,
        ) {
            flags.set_all(Ordering::Relaxed);
            bench_indexed_op(c, flags, &format!("{name_header}/ones/once"), 1, &next);
            bench_indexed_op(
                c,
                flags,
                &format!("{name_header}/ones/all"),
                flags.len(),
                &count,
            );
            flags.clear_all(Ordering::Relaxed);
            bench_indexed_op(c, flags, &format!("{name_header}/zeroes/once"), 1, &next);
            bench_indexed_op(
                c,
                flags,
                &format!("{name_header}/zeroes/all"),
                flags.len(),
                &count,
            );
        }
        bench_iterator(
            c,
            &flags,
            &format!("{header}/iter_set_around/inclusive"),
            |flags, pos| {
                flags
                    .iter_set_around::<true>(pos, Ordering::Relaxed)
                    .map(|mut it| it.next())
                    .unwrap_or(None)
            },
            |flags, pos| {
                flags
                    .iter_set_around::<true>(pos, Ordering::Relaxed)
                    .map(Iterator::count)
                    .unwrap_or(0)
            },
        );
        bench_iterator(
            c,
            &flags,
            &format!("{header}/iter_set_around/exclusive"),
            |flags, pos| {
                flags
                    .iter_set_around::<false>(pos, Ordering::Relaxed)
                    .map(|mut it| it.next())
                    .unwrap_or(None)
            },
            |flags, pos| {
                flags
                    .iter_set_around::<false>(pos, Ordering::Relaxed)
                    .map(Iterator::count)
                    .unwrap_or(0)
            },
        );
        bench_iterator(
            c,
            &flags,
            &format!("{header}/iter_unset_around/inclusive"),
            |flags, pos| {
                flags
                    .iter_unset_around::<true>(pos, Ordering::Relaxed)
                    .map(|mut it| it.next())
                    .unwrap_or(None)
            },
            |flags, pos| {
                flags
                    .iter_unset_around::<true>(pos, Ordering::Relaxed)
                    .map(Iterator::count)
                    .unwrap_or(0)
            },
        );
        bench_iterator(
            c,
            &flags,
            &format!("{header}/iter_unset_around/exclusive"),
            |flags, pos| {
                flags
                    .iter_unset_around::<false>(pos, Ordering::Relaxed)
                    .map(|mut it| it.next())
                    .unwrap_or(None)
            },
            |flags, pos| {
                flags
                    .iter_unset_around::<false>(pos, Ordering::Relaxed)
                    .map(Iterator::count)
                    .unwrap_or(0)
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
