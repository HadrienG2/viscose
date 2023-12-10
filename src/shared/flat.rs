//! Flat thread pool state
//!
//! This version of the thread pool state assumes that workers with neighboring
//! OS CPU number are topologically close in hardware. This is a relatively fair
//! approximation on AMD CPUs, where hyperthreads have consecutive CPU numbers,
//! but quite bad on Intel where it's cores that have consecutive CPU numbers.

use super::{
    flags::{bitref::BitRef, AtomicFlags},
    futex::{StealLocation, WorkerFutex},
    hierarchical::path::WorkAvailabilityPath,
    job::DynJob,
    SharedState, WorkerAvailability, WorkerConfig, WorkerInterface,
};
use crossbeam::{
    deque::{Injector, Steal},
    utils::CachePadded,
};
use hwlocality::{cpu::cpuset::CpuSet, Topology};
use std::{
    borrow::Borrow,
    ops::Deref,
    sync::{atomic::Ordering, Arc},
};

/// State shared between all thread pool users and workers, with
/// non-hierarchical work availability tracking.
#[derive(Debug)]
pub struct FlatState {
    /// Global work injector
    injector: Injector<DynJob<Self>>,

    /// Worker interfaces
    workers: Box<[CachePadded<WorkerInterface<Self>>]>,

    /// Per-worker truth that each worker _might_ have work ready to be stolen
    /// inside of its work queue
    ///
    /// Set by the associated worker upon pushing work, cleared by the worker
    /// when it tries to pop work and the queue is empty. See note in the
    /// work-stealing code to understand why it is preferrable that stealers do
    /// not clear this flag upon observing an empty work queue.
    ///
    /// The flag is set with Release ordering, so if the readout of a set flag
    /// is performed with Acquire ordering or followed by an Acquire barrier,
    /// the pushed work should be observable.
    work_availability: AtomicFlags,
}
//
impl SharedState for FlatState {
    fn with_worker_config(
        topology: &Topology,
        affinity: impl Borrow<CpuSet>,
    ) -> (Arc<Self>, Box<[WorkerConfig<Self>]>) {
        // Determine which CPUs we can actually use, and cross-check that the
        // implementation supports that
        let cpuset = topology.cpuset() & affinity;
        let num_workers = cpuset.weight().unwrap();
        assert_ne!(
            num_workers, 0,
            "a thread pool without threads can't make progress and will deadlock on first request"
        );
        assert!(
            num_workers < WorkerFutex::MAX_WORKERS,
            "unsupported number of worker threads"
        );

        // Set up worker-local state
        let mut worker_configs = Vec::with_capacity(num_workers);
        let mut worker_interfaces = Vec::with_capacity(num_workers);
        for cpu in cpuset {
            let (config, interface) = super::new_worker(cpu);
            worker_configs.push(config);
            worker_interfaces.push(CachePadded::new(interface));
        }

        // Set up global shared state
        let result = Arc::new(Self {
            injector: Injector::new(),
            workers: worker_interfaces.into(),
            work_availability: AtomicFlags::new(num_workers),
        });
        (result, worker_configs.into())
    }

    fn worker_interfaces(&self) -> impl Iterator<Item = &'_ WorkerInterface<Self>> {
        self.workers.iter().map(Deref::deref)
    }

    fn worker_availability(&self, worker_idx: usize) -> WorkAvailabilityPath<'_> {
        WorkAvailabilityPath::new_flat_with_cache(&self.work_availability, worker_idx)
    }

    fn find_work_to_steal<'result>(
        &'result self,
        _thief_worker_idx: usize,
        thief_availability: &'result WorkAvailabilityPath<'result>,
    ) -> Option<impl Iterator<Item = usize> + 'result> {
        let work_availability = &self.work_availability;
        work_availability
            .iter_set_around::<false, true>(
                thief_availability.flat_bit(),
                // Need at least Acquire ordering to ensure that the work we
                // observed to be available is actually visible in the worker's
                // queue, and don't need anything stronger:
                //
                // - Don't need AcqRel (which would require replacing the load
                //   with a RMW) since we're not trying to get any other thread
                //   in sync with our current state.
                // - Don't need SeqCst since there is no need for everyone to
                //   agree on the global order in which workers look for work.
                Ordering::Acquire,
            )
            .map(|iter| iter.map(|bit| bit.linear_idx(work_availability)))
    }

    #[inline]
    fn suggest_stealing_from_worker<'self_>(
        &'self_ self,
        target_worker_idx: usize,
        target_availability: &WorkAvailabilityPath<'self_>,
        update: Ordering,
    ) {
        self.suggest_stealing::<false, true>(
            target_availability.flat_bit(),
            StealLocation::Worker(target_worker_idx),
            update,
        )
    }

    fn inject_job(&self, job: DynJob<Self>, local_worker_idx: usize) {
        self.injector.push(job);
        self.suggest_stealing::<true, false>(
            &self.work_availability.bit(local_worker_idx),
            StealLocation::Injector,
            // Need at least Release ordering to ensure injected job is visible
            // to the target and don't need anything stronger:
            //
            // - Don't need AcqRel ordering since the thread that's pushing work
            //   does not want to get in sync with the target worker's state.
            // - Don't need SeqCst since there is no need for everyone to agree
            //   on the global order of job injection events.
            Ordering::Release,
        )
    }

    fn steal_from_injector(&self) -> Steal<DynJob<Self>> {
        self.injector.steal()
    }
}
//
impl FlatState {
    /// Recommend that the worker closest to a certain originating locality
    /// steal a task from the specified location
    fn suggest_stealing<'self_, const INCLUDE_CENTER: bool, const CACHE_SEARCH_MASKS: bool>(
        &'self_ self,
        local_worker: &BitRef<'self_, CACHE_SEARCH_MASKS>,
        task_location: StealLocation,
        update: Ordering,
    ) {
        // Check if there are job-less neighbors to submit work to...
        //
        // Need Acquire ordering so the futex is only read/modified after a work
        // unavailability signal is observed: compilers and CPUs should not
        // cache the work availability bit value or speculate on it here.
        let Some(mut asleep_neighbors) = self
            .work_availability
            .iter_unset_around::<INCLUDE_CENTER, CACHE_SEARCH_MASKS>(
                local_worker,
                Ordering::Acquire,
            )
        else {
            return;
        };

        // ...and if so, tell the closest one about our newly submitted job
        #[cold]
        fn unlikely<'self_, const CACHE_SEARCH_MASKS: bool>(
            self_: &'self_ FlatState,
            asleep_neighbors: impl Iterator<Item = BitRef<'self_, false>>,
            local_worker: &BitRef<'self_, CACHE_SEARCH_MASKS>,
            task_location: StealLocation,
            update: Ordering,
        ) {
            // Iterate over increasingly remote job-less neighbors
            let local_worker = local_worker.linear_idx(&self_.work_availability);
            for closest_asleep in asleep_neighbors {
                // Update their futex recommendation as appropriate
                //
                // Can use Relaxed ordering on failure because failing to
                // suggest work to a worker has no observable consequences and
                // isn't used to inform any decision other than looking up the
                // state of the next worker, which is independent from this one.
                let closest_asleep = closest_asleep.linear_idx(&self_.work_availability);
                let accepted = self_.workers[closest_asleep].futex.suggest_steal(
                    task_location,
                    local_worker,
                    update,
                    Ordering::Relaxed,
                );
                if accepted.is_ok() {
                    return;
                }
            }
        }
        unlikely(
            self,
            &mut asleep_neighbors,
            local_worker,
            task_location,
            update,
        )
    }
}
//
impl WorkerAvailability for BitRef<'_, true> {
    fn is_set(&self, order: Ordering) -> Option<bool> {
        Some(self.is_set(order))
    }

    fn fetch_set(&self, order: Ordering) -> Option<bool> {
        Some(self.fetch_set(order))
    }

    fn fetch_clear(&self, order: Ordering) -> Option<bool> {
        Some(self.fetch_clear(order))
    }
}
