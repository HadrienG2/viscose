//! Benchmarking utilities

use crate::{pool::FlatPool, worker::Scope};
use criterion::{Bencher, Criterion};
use crossbeam::utils::CachePadded;
use hwlocality::{cpu::binding::CpuBindingFlags, object::types::ObjectType, Topology};
use iterator_ilp::IteratorILP;
use std::sync::{Arc, OnceLock};

/// Run a benchmark for all interesting named CPU localities
pub fn for_each_locality(mut bench: impl FnMut(&str, rayon::ThreadPool, &str, FlatPool)) {
    let topology = topology();
    let mut last_affinity = None;
    for ty in [
        ObjectType::L1Cache,
        ObjectType::L3Cache,
        ObjectType::NUMANode,
        ObjectType::Machine,
    ] {
        for hyperthreading in [false, true] {
            // Pick the last object of this type
            let Some(last) = topology.objects_with_type(ty).last() else {
                continue;
            };

            // Compute cpuset and name for the current locality
            let mut locality_name = ty.to_string();
            let mut affinity = last.cpuset().unwrap().clone_target();
            if !hyperthreading {
                affinity.singlify_per_core(topology, 0);
                locality_name.push_str("/noSMT");
            } else {
                locality_name.push_str("/fullSMT");
            }

            // Check if this is the same as the last affinity we processed
            if let Some(last_affinity) = &last_affinity {
                if affinity == last_affinity {
                    continue;
                }
            }
            last_affinity = Some(affinity.clone());

            // Help rayon at CPU affinity ;)
            topology
                .bind_cpu(
                    &affinity,
                    CpuBindingFlags::PROCESS | CpuBindingFlags::STRICT,
                )
                .unwrap();

            // Build thread pools
            let flat_pool = FlatPool::with_affinity(topology.clone(), &affinity);
            let rayon_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(affinity.weight().unwrap())
                .thread_name(|idx| format!("Rayon worker #{idx}"))
                .build()
                .unwrap();

            // Run the benchmark
            bench(
                &format!("{locality_name}/rayon"),
                rayon_pool,
                &format!("{locality_name}/flat"),
                flat_pool,
            )
        }
    }
}

/// Recursive parallel fibonacci based on rayon
///
/// This is obviously not how you would efficiently compute the nth term of the
/// Fibonacci sequence (see tests::fibonacci_ref), but it's an excellent
/// microbenchmark for the overhead of join().
#[inline]
pub fn fibonacci_rayon(n: u64) -> u64 {
    if n > 1 {
        let (x, y) = rayon::join(|| fibonacci_rayon(n - 1), || fibonacci_rayon(n - 2));
        x + y
    } else {
        n
    }
}

/// Like `fibonacci_rayon()`, but uses a `FlatPool`
#[inline]
pub fn fibonacci_flat(scope: &Scope<'_>, n: u64) -> u64 {
    if n > 1 {
        let (x, y) = scope.join(
            || fibonacci_flat(scope, n - 1),
            move |scope| fibonacci_flat(scope, n - 2),
        );
        x + y
    } else {
        n
    }
}

/// Array of floats that can be split into blocks, where each block tracks which
/// thread pool worker it is local to
pub struct LocalFloats<const BLOCK_SIZE: usize> {
    /// Inner floating-point data (size must be a multiple of BLOCK_SIZE)
    data: Box<[f32]>,

    /// Per-block tracking of which worker processes data is local to
    locality: Box<[CachePadded<Option<usize>>]>,
}
//
impl<const BLOCK_SIZE: usize> LocalFloats<BLOCK_SIZE> {
    /// Set up storage for N data blocks
    pub fn new(num_blocks: usize) -> Self {
        Self {
            data: std::iter::repeat(0.0)
                .take(num_blocks * BLOCK_SIZE)
                .collect(),
            locality: std::iter::repeat(CachePadded::new(None))
                .take(num_blocks)
                .collect(),
        }
    }

    /// Acquire a slice covering the whole dataset
    pub fn as_slice(&mut self) -> LocalFloatsSlice<'_, BLOCK_SIZE> {
        LocalFloatsSlice {
            data: &mut self.data[..],
            locality: &mut self.locality[..],
        }
    }
}
//
/// Slice to some LocalFloats
pub struct LocalFloatsSlice<'target, const BLOCK_SIZE: usize> {
    /// Slice of the underlying LocalFloats::data
    data: &'target mut [f32],

    /// Slice of the underlying LocalFloats::locality
    locality: &'target mut [CachePadded<Option<usize>>],
}
//
impl<'target, const BLOCK_SIZE: usize> LocalFloatsSlice<'target, BLOCK_SIZE> {
    /// Split into two halves and process the halves if this slice more than one
    /// block long, otherwise process that single block sequentially
    pub fn process<R: Send>(
        &mut self,
        parallel: impl FnOnce([LocalFloatsSlice<'_, BLOCK_SIZE>; 2]) -> R,
        sequential: impl FnOnce(&mut [f32; BLOCK_SIZE], &mut Option<usize>) -> R,
        default: R,
    ) -> R {
        match self.locality.len() {
            0 => default,
            1 => {
                debug_assert_eq!(self.data.len(), BLOCK_SIZE);
                let data: *mut [f32; BLOCK_SIZE] = self.data.as_mut_ptr().cast();
                sequential(unsafe { &mut *data }, &mut self.locality[0])
            }
            _ => parallel(self.split()),
        }
    }

    /// Split this slice into two halves (second half may be empty)
    fn split<'self_, 'borrow>(&'self_ mut self) -> [LocalFloatsSlice<'borrow, BLOCK_SIZE>; 2]
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
                        left_blocks * BLOCK_SIZE,
                    ),
                    locality: std::slice::from_raw_parts_mut(
                        self.locality.as_mut_ptr(),
                        left_blocks,
                    ),
                },
                Self {
                    data: std::slice::from_raw_parts_mut(
                        self.data.as_mut_ptr().add(left_blocks * BLOCK_SIZE),
                        right_blocks * BLOCK_SIZE,
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
pub fn bench_local_floats<const BLOCK_SIZE: usize>(
    c: &mut Criterion,
    benchmark_name: &str,
    backend_name: &str,
    mut bench_impl: impl FnMut(&mut Bencher, &mut LocalFloatsSlice<'_, BLOCK_SIZE>),
) {
    assert!(BLOCK_SIZE.is_power_of_two());
    let block_size_pow2 = BLOCK_SIZE.trailing_zeros();
    let mut group = c.benchmark_group(&format!("{backend_name}/{benchmark_name}"));
    for num_blocks_pow2 in 0..=max_data_size_pow2() - block_size_pow2 {
        let num_blocks = 1usize << num_blocks_pow2;
        let mut data = LocalFloats::<BLOCK_SIZE>::new(num_blocks);
        group.throughput(criterion::Throughput::Elements(
            (num_blocks * BLOCK_SIZE) as _,
        ));
        group.bench_function(&format!("{num_blocks}x{BLOCK_SIZE}"), |b| {
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
        let cache_stats = topology().cpu_cache_stats().unwrap();
        let total_l3_capacity = cache_stats.total_data_cache_sizes().last().unwrap();
        let mut max_size = 2 * total_l3_capacity;
        if !max_size.is_power_of_two() {
            max_size = max_size.next_power_of_two();
        }
        max_size.trailing_zeros()
    })
}

/// Square each number inside of a LocalFloatsSlice
#[inline]
pub fn square_rayon<const BLOCK_SIZE: usize>(slice: &mut LocalFloatsSlice<'_, BLOCK_SIZE>) {
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

/// Like `square_rayon()`, but using a `FlatPool`
#[inline]
pub fn square_flat<const BLOCK_SIZE: usize>(
    scope: &Scope<'_>,
    slice: &mut LocalFloatsSlice<'_, BLOCK_SIZE>,
) {
    slice.process(
        |[mut left, mut right]| {
            scope.join(
                || square_flat(scope, &mut left),
                move |scope| square_flat(scope, &mut right),
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
#[inline]
pub fn sum_rayon<const BLOCK_SIZE: usize, const ILP_STREAMS: usize>(
    slice: &mut LocalFloatsSlice<'_, BLOCK_SIZE>,
) -> f32 {
    slice.process(
        |[mut left, mut right]| {
            let (left, right) = rayon::join(
                || sum_rayon::<BLOCK_SIZE, ILP_STREAMS>(&mut left),
                || sum_rayon::<BLOCK_SIZE, ILP_STREAMS>(&mut right),
            );
            left + right
        },
        |block, _locality| block.iter().copied().sum_ilp::<ILP_STREAMS, f32>(),
        0.0,
    )
}

/// Like `sum_rayon()`, but uses a `FlatPool`
#[inline]
pub fn sum_flat<const BLOCK_SIZE: usize, const ILP_STREAMS: usize>(
    scope: &Scope<'_>,
    slice: &mut LocalFloatsSlice<'_, BLOCK_SIZE>,
) -> f32 {
    slice.process(
        |[mut left, mut right]| {
            let (left, right) = scope.join(
                || sum_flat::<BLOCK_SIZE, ILP_STREAMS>(scope, &mut left),
                move |scope| sum_flat::<BLOCK_SIZE, ILP_STREAMS>(scope, &mut right),
            );
            left + right
        },
        |block, _locality| block.iter().copied().sum_ilp::<ILP_STREAMS, f32>(),
        0.0,
    )
}

/// Memory-bound recursive squared vector norm computation based on FlatPool
///
/// This computation is not written for optimal efficiency (a single-pass
/// algorithm would be more efficient), but to highlight the importance of NUMA
/// and cache locality in multi-threaded work. It purposely writes down the
/// squares of vector elements and then reads them back to assess how the
/// performance of such memory-bound code is affected by allocation locality.
#[inline]
pub fn norm_sqr_rayon<const BLOCK_SIZE: usize, const REDUCE_ILP_STREAMS: usize>(
    slice: &mut LocalFloatsSlice<'_, BLOCK_SIZE>,
) -> f32 {
    slice.process(
        |[mut left, mut right]| {
            rayon::join(
                || square_rayon::<BLOCK_SIZE>(&mut left),
                || square_rayon::<BLOCK_SIZE>(&mut right),
            );
            let (left, right) = rayon::join(
                || sum_rayon::<BLOCK_SIZE, REDUCE_ILP_STREAMS>(&mut left),
                || sum_rayon::<BLOCK_SIZE, REDUCE_ILP_STREAMS>(&mut right),
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

/// Like `norm_sqr_rayon()`, but uses a `FlatPool`
#[inline]
pub fn norm_sqr_flat<const BLOCK_SIZE: usize, const REDUCE_ILP_STREAMS: usize>(
    scope: &Scope<'_>,
    slice: &mut LocalFloatsSlice<'_, BLOCK_SIZE>,
) -> f32 {
    slice.process(
        |[mut left, mut right]| {
            scope.join(
                || square_flat(scope, &mut left),
                |scope| square_flat(scope, &mut right),
            );
            let (left, right) = scope.join(
                || sum_flat::<BLOCK_SIZE, REDUCE_ILP_STREAMS>(scope, &mut left),
                move |scope| sum_flat::<BLOCK_SIZE, REDUCE_ILP_STREAMS>(scope, &mut right),
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

/// Topology instance shared by all above functions
fn topology() -> &'static Arc<Topology> {
    static INSTANCE: OnceLock<Arc<Topology>> = OnceLock::new();
    INSTANCE.get_or_init(|| Arc::new(Topology::new().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::FlatPool;

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
        let flat = FlatPool::new();
        flat.run(|scope| {
            for i in 0..=34 {
                assert_eq!(fibonacci_flat(scope, i), fibonacci_ref(i));
            }
        });
    }
}