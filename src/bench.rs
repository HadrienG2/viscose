//! Benchmarking utilities

use crate::{pool::ThreadPool, worker::scope::Scope};
use criterion::{Bencher, Criterion};
use crossbeam::utils::CachePadded;
use hwlocality::{
    cpu::binding::CpuBindingFlags,
    object::{depth::NormalDepth, types::ObjectType},
};
use iterator_ilp::IteratorILP;
use std::{collections::BTreeSet, fmt::Write, sync::OnceLock};

/// Re-export atomic flags for benchmarking
pub use crate::shared::flags::{bitref::BitRef, AtomicFlags};

/// Run a benchmark for all interesting  localities
pub fn for_each_locality(
    mut bench: impl FnMut(
        &str,
        Box<dyn FnMut() -> rayon::ThreadPool>,
        &str,
        Box<dyn FnMut() -> ThreadPool>,
    ),
) {
    crate::setup_logger_once();
    let topology = crate::topology();
    let mut seen_affinities = BTreeSet::new();
    for depth in NormalDepth::iter_range(NormalDepth::MIN, topology.depth()).rev() {
        for hyperthreading in [false, true] {
            // Pick the first object at this depth
            let first = topology.objects_at_depth(depth).next().unwrap();

            // Compute cpuset and name for the current locality
            let ty = first.object_type();
            let mut locality_name = ty.to_string();
            if ty == ObjectType::Group {
                write!(locality_name, "@depth={depth}").unwrap();
            }
            let mut affinity = first.cpuset().unwrap().clone_target();
            if !hyperthreading {
                affinity.singlify_per_core(topology, 0);
                locality_name.push_str("/noSMT");
            } else {
                locality_name.push_str("/fullSMT");
            }

            // Check if we have already processed an equivalent locality
            if !seen_affinities.insert(affinity.clone()) {
                continue;
            }

            // Enforce CPU affinity constraint for every thread including
            // rayon-spawned threads and the benchmark's main thread
            topology
                .bind_cpu(
                    &affinity,
                    CpuBindingFlags::PROCESS | CpuBindingFlags::STRICT,
                )
                .unwrap();

            // Prepare to build thread pools
            let affinity2 = affinity.clone();
            let make_our_pool = move || ThreadPool::with_affinity(topology.clone(), &affinity2);
            let make_rayon_pool = move || {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(affinity.weight().unwrap())
                    .thread_name(|idx| format!("Rayon worker #{idx}"))
                    .build()
                    .unwrap()
            };

            // Run the benchmark
            bench(
                &format!("{locality_name}/rayon"),
                Box::new(make_rayon_pool),
                &format!("{locality_name}/flat"),
                Box::new(make_our_pool),
            )
        }
    }
}

/// Recursive parallel fibonacci based on rayon
///
/// This is obviously not how you would efficiently compute the nth term of the
/// Fibonacci sequence (see `tests::fibonacci_ref()` for that), but it's an
/// excellent microbenchmark for the overhead of `join()`.
#[inline]
pub fn fibonacci_rayon(n: u64) -> u64 {
    if n > 1 {
        let (x, y) = rayon::join(|| fibonacci_rayon(n - 1), || fibonacci_rayon(n - 2));
        x + y
    } else {
        n
    }
}

/// Like `fibonacci_rayon()`, but uses a `ThreadPool`
#[inline]
pub fn fibonacci_ours(scope: &Scope<'_>, n: u64) -> u64 {
    if n > 1 {
        let (x, y) = scope.join(
            || fibonacci_ours(scope, n - 1),
            move |scope| fibonacci_ours(scope, n - 2),
        );
        x + y
    } else {
        n
    }
}

/// Array of floats that can be split into blocks, where each block tracks which
/// thread pool worker it is local to
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LocalFloats<const BLOCK_LEN: usize> {
    /// Inner floating-point data (size must be a multiple of BLOCK_LEN)
    data: Box<[f32]>,

    /// Per-block tracking of which worker processes data is local to
    locality: Box<[CachePadded<Option<usize>>]>,
}
//
impl<const BLOCK_LEN: usize> LocalFloats<BLOCK_LEN> {
    /// Set up storage for N data blocks
    pub fn new(num_blocks: usize) -> Self {
        Self {
            data: std::iter::repeat(0.0)
                .take(num_blocks * BLOCK_LEN)
                .collect(),
            locality: std::iter::repeat(CachePadded::new(None))
                .take(num_blocks)
                .collect(),
        }
    }

    /// Acquire a slice covering the whole dataset
    pub fn as_slice(&mut self) -> LocalFloatsSlice<'_, BLOCK_LEN> {
        LocalFloatsSlice {
            data: &mut self.data[..],
            locality: &mut self.locality[..],
        }
    }
}
//
/// Slice to some LocalFloats
#[derive(Debug, PartialEq)]
pub struct LocalFloatsSlice<'target, const BLOCK_LEN: usize> {
    /// Slice of the underlying LocalFloats::data
    data: &'target mut [f32],

    /// Slice of the underlying LocalFloats::locality
    locality: &'target mut [CachePadded<Option<usize>>],
}
//
impl<'target, const BLOCK_LEN: usize> LocalFloatsSlice<'target, BLOCK_LEN> {
    /// Split into two halves and process the halves if this slice more than one
    /// block long, otherwise process that single block sequentially
    pub fn process<R: Send>(
        &mut self,
        parallel: impl FnOnce([LocalFloatsSlice<'_, BLOCK_LEN>; 2]) -> R,
        sequential: impl FnOnce(&mut [f32; BLOCK_LEN], &mut Option<usize>) -> R,
        default: R,
    ) -> R {
        match self.locality.len() {
            0 => default,
            1 => {
                debug_assert_eq!(self.data.len(), BLOCK_LEN);
                let data: *mut [f32; BLOCK_LEN] = self.data.as_mut_ptr().cast();
                sequential(unsafe { &mut *data }, &mut self.locality[0])
            }
            _ => parallel(self.split()),
        }
    }

    /// Split this slice into two halves (second half may be empty)
    fn split<'self_, 'borrow>(&'self_ mut self) -> [LocalFloatsSlice<'borrow, BLOCK_LEN>; 2]
    where
        'self_: 'borrow,
        'target: 'borrow, // Probably redundant, but clarifies the safety proof
    {
        let num_blocks = self.locality.len();
        let left_blocks = num_blocks / 2;
        let right_blocks = num_blocks - left_blocks;
        // SAFETY: This is basically a double split_at_mut(), but for some
        //         reason the borrow checker does not accept the safe code that
        //         calls split_at_mut() twice and shoves the outputs into Self.
        //
        //         I believe this to be a false positive because the lifetime
        //         bounds ensure that...
        //
        //         - Since 'self_: 'borrow, self cannot be used as long as the
        //           output LocalFloatsSlice object is live, Therefore, it is
        //           impossible to use the original slice as long as the
        //           reborrowed slice is live, so although there is some &mut
        //           aliasing, it is harmless just as in split_at_mut()..
        //         - Since 'target: 'borrow, the output slice cannot outlive the
        //           source data, so no use-after-free/move is possible.
        unsafe {
            [
                Self {
                    data: std::slice::from_raw_parts_mut(
                        self.data.as_mut_ptr(),
                        left_blocks * BLOCK_LEN,
                    ),
                    locality: std::slice::from_raw_parts_mut(
                        self.locality.as_mut_ptr(),
                        left_blocks,
                    ),
                },
                Self {
                    data: std::slice::from_raw_parts_mut(
                        self.data.as_mut_ptr().add(left_blocks * BLOCK_LEN),
                        right_blocks * BLOCK_LEN,
                    ),
                    locality: std::slice::from_raw_parts_mut(
                        self.locality.as_mut_ptr().add(left_blocks),
                        right_blocks,
                    ),
                },
            ]
        }
    }
}

/// Run a LocalFloats-based benchmark at a given block size
pub fn bench_local_floats<const BLOCK_LEN: usize>(
    c: &mut Criterion,
    benchmark_name: &str,
    backend_name: &str,
    mut bench_impl: impl FnMut(&mut Bencher, &mut LocalFloatsSlice<'_, BLOCK_LEN>),
) {
    assert!(BLOCK_LEN.is_power_of_two());
    let elem_size = std::mem::size_of::<f32>();
    let block_size_pow2 = (BLOCK_LEN * elem_size).trailing_zeros();
    let mut group = c.benchmark_group(&format!("{backend_name}/{benchmark_name}"));
    for num_blocks_pow2 in 0..=max_data_size_pow2() - block_size_pow2 {
        let num_blocks = 1usize << num_blocks_pow2;
        let mut data = LocalFloats::<BLOCK_LEN>::new(num_blocks);
        group.throughput(criterion::Throughput::Elements(
            (num_blocks * BLOCK_LEN) as _,
        ));
        let block_size_kib = BLOCK_LEN * elem_size / 1024;
        group.bench_function(&format!("{num_blocks}x{block_size_kib}KiB"), |b| {
            bench_impl(b, &mut data.as_slice())
        });
    }
}

/// Determine the maximal interesting benchmark dataset size on this machine, as
/// a power of two
///
/// We cut off at the point where we have overflowed all reachable L3 caches and
/// start hitting the DRAM bandwidth limit.
fn max_data_size_pow2() -> u32 {
    static RESULT: OnceLock<u32> = OnceLock::new();
    *RESULT.get_or_init(|| {
        let cache_stats = crate::topology().cpu_cache_stats().unwrap();
        let total_l3_capacity = cache_stats.total_data_cache_sizes().last().unwrap();
        let mut max_size = 8 * total_l3_capacity;
        if !max_size.is_power_of_two() {
            max_size = max_size.next_power_of_two();
        }
        max_size.trailing_zeros()
    })
}

/// Square each number inside of a LocalFloatsSlice
///
/// This is our simplest memory-bound microbenchmark with a load-to-store memory
/// access pattern and zero dependency chains between loop iterations.
#[inline]
pub fn square_rayon<const BLOCK_LEN: usize>(slice: &mut LocalFloatsSlice<'_, BLOCK_LEN>) {
    slice.process(
        |[mut left, mut right]| {
            rayon::join(|| square_rayon(&mut left), || square_rayon(&mut right));
        },
        |block, _locality| {
            block.iter_mut().for_each(|elem| *elem = elem.powi(2));
        },
        (),
    );
}

/// Like `square_rayon()`, but using a `ThreadPool`
#[inline]
pub fn square_ours<const BLOCK_LEN: usize>(
    scope: &Scope<'_>,
    slice: &mut LocalFloatsSlice<'_, BLOCK_LEN>,
) {
    slice.process(
        |[mut left, mut right]| {
            scope.join(
                || square_ours(scope, &mut left),
                move |scope| square_ours(scope, &mut right),
            );
        },
        |block, locality| {
            *locality = Some(scope.worker_id());
            // TODO: Experiment with pinning allocations, etc
            block.iter_mut().for_each(|elem| *elem = elem.powi(2));
        },
        (),
    )
}

/// Sum the numbers inside of a LocalFloatSlice
///
/// This memory-bound microbenchmark does not perform stores, but it features
/// dependencies between loop iterations and is thus more strongly affected by
/// increasing memory access latencies.
#[inline]
pub fn sum_rayon<const BLOCK_LEN: usize, const ILP_STREAMS: usize>(
    slice: &mut LocalFloatsSlice<'_, BLOCK_LEN>,
) -> f32 {
    slice.process(
        |[mut left, mut right]| {
            let (left, right) = rayon::join(
                || sum_rayon::<BLOCK_LEN, ILP_STREAMS>(&mut left),
                || sum_rayon::<BLOCK_LEN, ILP_STREAMS>(&mut right),
            );
            left + right
        },
        |block, _locality| block.iter().copied().sum_ilp::<ILP_STREAMS, f32>(),
        0.0,
    )
}

/// Like `sum_rayon()`, but uses a `ThreadPool`
#[inline]
pub fn sum_ours<const BLOCK_LEN: usize, const ILP_STREAMS: usize>(
    scope: &Scope<'_>,
    slice: &mut LocalFloatsSlice<'_, BLOCK_LEN>,
) -> f32 {
    slice.process(
        |[mut left, mut right]| {
            let (left, right) = scope.join(
                || sum_ours::<BLOCK_LEN, ILP_STREAMS>(scope, &mut left),
                move |scope| sum_ours::<BLOCK_LEN, ILP_STREAMS>(scope, &mut right),
            );
            left + right
        },
        |block, _locality| block.iter().copied().sum_ilp::<ILP_STREAMS, f32>(),
        0.0,
    )
}

/// Memory-bound recursive squared vector norm computation based on ThreadPool
///
/// This computation is not written for optimal efficiency (a single-pass
/// algorithm would be more efficient), but to highlight the importance of NUMA
/// and cache locality in multi-threaded work. It purposely writes down the
/// squares of vector elements, then reads them back, to expose how the
/// performance of such store-to-load memory access patterns is affected by task
/// CPU migrations or lack thereof.
///
/// While artificial in this particular case, this store-to-load memory access
/// pattern is commonly seen in real-world numpy-style unoptimized array
/// computations, where the result of each computation step is written down to a
/// temporary array and later re-read by a later computation step.
#[inline]
pub fn norm_sqr_rayon<const BLOCK_LEN: usize, const REDUCE_ILP_STREAMS: usize>(
    slice: &mut LocalFloatsSlice<'_, BLOCK_LEN>,
) -> f32 {
    slice.process(
        |[mut left, mut right]| {
            rayon::join(|| square_rayon(&mut left), || square_rayon(&mut right));
            let (left, right) = rayon::join(
                || sum_rayon::<BLOCK_LEN, REDUCE_ILP_STREAMS>(&mut left),
                || sum_rayon::<BLOCK_LEN, REDUCE_ILP_STREAMS>(&mut right),
            );
            left + right
        },
        |mut block, _locality| {
            block.iter_mut().for_each(|elem| *elem = elem.powi(2));
            block = pessimize::hide(block);
            block.iter().copied().sum_ilp::<REDUCE_ILP_STREAMS, f32>()
        },
        0.0,
    )
}

/// Like `norm_sqr_rayon()`, but uses a `ThreadPool`
#[inline]
pub fn norm_sqr_ours<const BLOCK_LEN: usize, const REDUCE_ILP_STREAMS: usize>(
    scope: &Scope<'_>,
    slice: &mut LocalFloatsSlice<'_, BLOCK_LEN>,
) -> f32 {
    slice.process(
        |[mut left, mut right]| {
            scope.join(
                || square_ours(scope, &mut left),
                |scope| square_ours(scope, &mut right),
            );
            let (left, right) = scope.join(
                || sum_ours::<BLOCK_LEN, REDUCE_ILP_STREAMS>(scope, &mut left),
                move |scope| sum_ours::<BLOCK_LEN, REDUCE_ILP_STREAMS>(scope, &mut right),
            );
            left + right
        },
        |mut block, locality| {
            *locality = Some(scope.worker_id());
            // TODO: Experiment with pinning allocations, etc
            block.iter_mut().for_each(|elem| *elem = elem.powi(2));
            block = pessimize::hide(block);
            block.iter().copied().sum_ilp::<REDUCE_ILP_STREAMS, f32>()
        },
        0.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::ThreadPool;

    /// Reference computation of the N-th fibonacci sequence term
    fn fibonacci_ref(n: u64) -> u64 {
        if n > 0 {
            let sqrt_5 = 5.0f64.sqrt();
            let phi = (1.0 + sqrt_5) / 2.0;
            let f_n = phi.powi(i32::try_from(n).unwrap()) / sqrt_5;
            f_n.round() as u64
        } else {
            0
        }
    }

    #[test]
    fn fibonacci() {
        crate::setup_logger_once();
        let pool = ThreadPool::new();
        pool.run(|scope| {
            for i in 0..=34 {
                assert_eq!(fibonacci_ours(scope, i), fibonacci_ref(i));
            }
        });
    }
}
