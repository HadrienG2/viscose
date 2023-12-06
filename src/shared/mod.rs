//! State shared between the thread pool and all of its workers

pub(crate) mod flags;
pub(crate) mod futex;
pub(crate) mod hierarchical;
pub(crate) mod job;

use self::{
    flags::{bitref::BitRef, AtomicFlags},
    futex::{StealLocation, WorkerFutex},
    job::DynJob,
};
use crossbeam::{
    deque::{self, Injector, Steal, Stealer},
    utils::CachePadded,
};
use hwlocality::{bitmap::BitmapIndex, cpu::cpuset::CpuSet, Topology};
use std::{
    borrow::Borrow,
    ops::Deref,
    sync::{atomic::Ordering, Arc},
};

/// State shared between all thread pool users and workers, with
/// non-hierarchical work availability tracking.
#[derive(Debug, Default)]
pub(crate) struct SharedState {
    /// Global work injector
    injector: Injector<DynJob>,

    /// Worker interfaces
    workers: Box<[CachePadded<WorkerInterface>]>,

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
impl SharedState {
    /// Set up the shared and worker-local state
    pub fn with_worker_config(
        topology: &Topology,
        affinity: impl Borrow<CpuSet>,
    ) -> (Arc<Self>, Box<[WorkerConfig]>) {
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
            let (config, interface) = new_worker(cpu);
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

    /// Access the worker interfaces
    pub fn worker_interfaces(&self) -> impl Iterator<Item = &'_ WorkerInterface> {
        self.workers.iter().map(Deref::deref)
    }

    /// Generate a worker-private work availability bit accessor
    ///
    /// Workers can use this to signal when they have work available to steal
    /// and when they stop having work available to steal.
    ///
    /// This accessor is meant to constructed by workers at thread pool
    /// initialization time and then retained for the entire lifetime of the
    /// thread pool. As a result, it is optimized for efficiency of repeated
    /// usage, but initial construction may be expensive.
    pub fn worker_availability(&self, worker_idx: usize) -> BitRef<'_, true> {
        self.work_availability.bit_with_cache(worker_idx)
    }

    /// Enumerate workers with work available to steal at increasing distances
    /// from a certain "thief" worker
    pub fn find_work_to_steal<'result>(
        &'result self,
        worker_availability: &BitRef<'result, true>,
    ) -> Option<impl Iterator<Item = usize> + 'result> {
        let work_availability = &self.work_availability;
        work_availability
            .iter_set_around::<false, true>(
                worker_availability,
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

    /// Given a worker with work available for stealing, find the closest cousin
    /// that doesn't have work, if any, and suggest that it steal from there
    ///
    /// There should be a `Release` barrier between the moment where work is
    /// pushed in the worker's work queue and the moment where work is signaled
    /// to be available like this. You can either bundle the `Release` barrier
    /// into this transaction, or put a separate `atomic::fence(Release)` before
    /// this transaction and make it `Relaxed` if you have multiple work
    /// availability signaling transactions to do.
    pub fn suggest_stealing_from_worker<'self_>(
        &'self_ self,
        target_idx: usize,
        target_availability: &BitRef<'self_, true>,
        update: Ordering,
    ) {
        self.suggest_stealing::<false, true>(
            target_availability,
            StealLocation::Worker(target_idx),
            update,
        )
    }

    /// Inject work from outside the thread pool, and tell the worker closest to
    /// the originating locality that doesn't have work about it
    pub fn inject_job(&self, job: DynJob, local_worker_idx: usize) {
        self.injector.push(job);
        self.suggest_stealing::<true, false>(
            &self.work_availability.bit(local_worker_idx),
            StealLocation::Injector,
            // Need at least Release ordering to ensure injected job is visible
            // to the target and don't need anything stronger:
            //
            // - Don't need AcqRel ordering since the thread that's pushing work
            //   does not want to get in sync with the current worker state.
            // - Don't need SeqCst since there is no need for everyone to agree
            //   on the global order of job injection events.
            Ordering::Release,
        )
    }

    /// Try to steal a job from the global work injector
    pub fn steal_from_injector(&self) -> Steal<DynJob> {
        self.injector.steal()
    }

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
            self_: &'self_ SharedState,
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

/// Internal state needed to configure a new worker
#[derive(Debug)]
pub(crate) struct WorkerConfig {
    /// Work queue
    pub work_queue: deque::Worker<DynJob>,

    /// CPU which this worker should be pinned to
    pub cpu: BitmapIndex,
}

/// External interface to a single worker in a thread pool
#[derive(Debug)]
pub(crate) struct WorkerInterface {
    /// A way to steal from the worker
    pub stealer: Stealer<DynJob>,

    /// Futex that the worker sleeps on when it has nothing to do, used to
    /// instruct it what to do when it is awakened.
    pub futex: WorkerFutex,
}

/// Prepare to add a new worker to the thread pool
///
/// This builds both the internal state that will be used to configure the
/// worker on startup and the external interface that will be used by the rest
/// of the world to communicate with the worker.
pub(crate) fn new_worker(cpu: BitmapIndex) -> (WorkerConfig, WorkerInterface) {
    let config = WorkerConfig {
        work_queue: deque::Worker::new_lifo(),
        cpu,
    };
    let interface = WorkerInterface {
        stealer: config.work_queue.stealer(),
        futex: WorkerFutex::new(),
    };
    (config, interface)
}
