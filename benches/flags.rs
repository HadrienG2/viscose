use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use sched_local::flags::{bitref::BitRef, AtomicFlags};
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

        // Querying a flag
        bench_indexed_op(c, &flags, &format!("{header}/bit"), 1, |flags, pos| {
            pessimize::consume(&flags.bit(pos))
        });
        bench_indexed_op(
            c,
            &flags,
            &format!("{header}/bit_with_cache"),
            1,
            |flags, pos| pessimize::consume(&flags.bit_with_cache(pos)),
        );

        // General logic for benchmarks that target a specific BitRef
        fn bench_ref_op_uncached<R>(
            c: &mut Criterion,
            flags: &AtomicFlags,
            group_name: &str,
            num_inner_ops: usize,
            mut op: impl FnMut(&AtomicFlags, &BitRef<'_, false>) -> R,
        ) {
            use pessimize::hide;
            let mut group = c.benchmark_group(group_name);
            group.throughput(Throughput::Elements(num_inner_ops as _));
            let first = flags.bit(0);
            group.bench_function("first/uncached", |b| {
                b.iter(|| op(hide(flags), hide(&first)))
            });
            let center = flags.bit(flags.len() / 2);
            group.bench_function("center/uncached", |b| {
                b.iter(|| op(hide(flags), hide(&center)))
            });
            let last = flags.bit(flags.len() - 1);
            group.bench_function("last/uncached", |b| b.iter(|| op(hide(flags), hide(&last))));
        }
        fn bench_ref_op_cached<R>(
            c: &mut Criterion,
            flags: &AtomicFlags,
            group_name: &str,
            num_inner_ops: usize,
            mut op: impl FnMut(&AtomicFlags, &BitRef<'_, true>) -> R,
        ) {
            use pessimize::hide;
            let mut group = c.benchmark_group(group_name);
            group.throughput(Throughput::Elements(num_inner_ops as _));
            let first = flags.bit_with_cache(0);
            group.bench_function("first/cached", |b| b.iter(|| op(hide(flags), hide(&first))));
            let center = flags.bit_with_cache(flags.len() / 2);
            group.bench_function("center/cached", |b| {
                b.iter(|| op(hide(flags), hide(&center)))
            });
            let last = flags.bit_with_cache(flags.len() - 1);
            group.bench_function("last/cached", |b| b.iter(|| op(hide(flags), hide(&last))));
        }

        // Operations that test a single bit
        bench_ref_op_uncached(c, &flags, &format!("{header}/is_set"), 1, |_flags, bit| {
            bit.is_set(Ordering::Relaxed)
        });
        bench_ref_op_uncached(
            c,
            &flags,
            &format!("{header}/fetch_set"),
            1,
            |_flags, bit| bit.fetch_set(Ordering::Relaxed),
        );
        bench_ref_op_uncached(
            c,
            &flags,
            &format!("{header}/fetch_clear"),
            1,
            |_flags, bit| bit.fetch_clear(Ordering::Relaxed),
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
        fn bench_iterator(
            c: &mut Criterion,
            flags: &AtomicFlags,
            name_header: &str,
            has_next_uncached: impl Fn(&AtomicFlags, &BitRef<'_, false>) -> bool,
            has_next_cached: impl Fn(&AtomicFlags, &BitRef<'_, true>) -> bool,
            count_uncached: impl Fn(&AtomicFlags, &BitRef<'_, false>) -> usize,
            count_cached: impl Fn(&AtomicFlags, &BitRef<'_, true>) -> usize,
        ) {
            flags.set_all(Ordering::Relaxed);
            bench_ref_op_uncached(
                c,
                flags,
                &format!("{name_header}/ones/once"),
                1,
                &has_next_uncached,
            );
            bench_ref_op_cached(
                c,
                flags,
                &format!("{name_header}/ones/once"),
                1,
                &has_next_cached,
            );
            bench_ref_op_uncached(
                c,
                flags,
                &format!("{name_header}/ones/all"),
                flags.len(),
                &count_uncached,
            );
            bench_ref_op_cached(
                c,
                flags,
                &format!("{name_header}/ones/all"),
                flags.len(),
                &count_cached,
            );
            flags.clear_all(Ordering::Relaxed);
            bench_ref_op_uncached(
                c,
                flags,
                &format!("{name_header}/zeroes/once"),
                1,
                &has_next_uncached,
            );
            bench_ref_op_cached(
                c,
                flags,
                &format!("{name_header}/zeroes/once"),
                1,
                &has_next_cached,
            );
            bench_ref_op_uncached(
                c,
                flags,
                &format!("{name_header}/zeroes/all"),
                flags.len(),
                &count_uncached,
            );
            bench_ref_op_cached(
                c,
                flags,
                &format!("{name_header}/zeroes/all"),
                flags.len(),
                &count_cached,
            );
        }
        bench_iterator(
            c,
            &flags,
            &format!("{header}/iter_set_around/inclusive"),
            |flags, pos| {
                flags
                    .iter_set_around::<true, false>(pos, Ordering::Relaxed)
                    .map(|mut it| it.next())
                    .unwrap_or(None)
                    .is_some()
            },
            |flags, pos| {
                flags
                    .iter_set_around::<true, true>(pos, Ordering::Relaxed)
                    .map(|mut it| it.next())
                    .unwrap_or(None)
                    .is_some()
            },
            |flags, pos| {
                flags
                    .iter_set_around::<true, false>(pos, Ordering::Relaxed)
                    .map(Iterator::count)
                    .unwrap_or(0)
            },
            |flags, pos| {
                flags
                    .iter_set_around::<true, true>(pos, Ordering::Relaxed)
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
                    .iter_set_around::<false, false>(pos, Ordering::Relaxed)
                    .map(|mut it| it.next())
                    .unwrap_or(None)
                    .is_some()
            },
            |flags, pos| {
                flags
                    .iter_set_around::<false, true>(pos, Ordering::Relaxed)
                    .map(|mut it| it.next())
                    .unwrap_or(None)
                    .is_some()
            },
            |flags, pos| {
                flags
                    .iter_set_around::<false, false>(pos, Ordering::Relaxed)
                    .map(Iterator::count)
                    .unwrap_or(0)
            },
            |flags, pos| {
                flags
                    .iter_set_around::<false, true>(pos, Ordering::Relaxed)
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
                    .iter_unset_around::<true, false>(pos, Ordering::Relaxed)
                    .map(|mut it| it.next())
                    .unwrap_or(None)
                    .is_some()
            },
            |flags, pos| {
                flags
                    .iter_unset_around::<true, true>(pos, Ordering::Relaxed)
                    .map(|mut it| it.next())
                    .unwrap_or(None)
                    .is_some()
            },
            |flags, pos| {
                flags
                    .iter_unset_around::<true, false>(pos, Ordering::Relaxed)
                    .map(Iterator::count)
                    .unwrap_or(0)
            },
            |flags, pos| {
                flags
                    .iter_unset_around::<true, true>(pos, Ordering::Relaxed)
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
                    .iter_unset_around::<false, false>(pos, Ordering::Relaxed)
                    .map(|mut it| it.next())
                    .unwrap_or(None)
                    .is_some()
            },
            |flags, pos| {
                flags
                    .iter_unset_around::<false, true>(pos, Ordering::Relaxed)
                    .map(|mut it| it.next())
                    .unwrap_or(None)
                    .is_some()
            },
            |flags, pos| {
                flags
                    .iter_unset_around::<false, false>(pos, Ordering::Relaxed)
                    .map(Iterator::count)
                    .unwrap_or(0)
            },
            |flags, pos| {
                flags
                    .iter_unset_around::<false, true>(pos, Ordering::Relaxed)
                    .map(Iterator::count)
                    .unwrap_or(0)
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
