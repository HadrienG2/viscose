pub mod flags;
/// Thread pool shared state
pub(crate) mod futex;

use self::{
    flags::{bitref::BitRef, AtomicFlags},
    futex::{StealLocation, WorkerFutex},
};
use crate::job::DynJob;
use crossbeam::{
    deque::{self, Injector, Stealer},
    utils::CachePadded,
};
use std::sync::{atomic::Ordering, Arc};

/// State that is shared between users of the thread pool and all thread pool
/// workers
#[derive(Debug, Default)]
pub(crate) struct SharedState {
    /// Global work injector
    pub injector: Injector<DynJob>,

    /// Worker interfaces
    pub workers: Box<[CachePadded<WorkerInterface>]>,

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
    pub work_availability: AtomicFlags,
}
//
impl SharedState {
    /// Set up the shared state
    pub fn with_work_queues(num_workers: usize) -> (Arc<Self>, Box<[deque::Worker<DynJob>]>) {
        assert!(
            num_workers < WorkerFutex::MAX_WORKERS,
            "unsupported number of worker threads"
        );
        let injector = Injector::new();
        let mut work_queues = Vec::with_capacity(num_workers);
        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let (worker, work_queue) = WorkerInterface::with_work_queue();
            workers.push(CachePadded::new(worker));
            work_queues.push(work_queue);
        }
        let result = Arc::new(Self {
            injector,
            workers: workers.into(),
            work_availability: AtomicFlags::new(num_workers),
        });
        (result, work_queues.into())
    }

    /// Recommend that the work-less thread closest to a certain originating
    /// locality steal a task from the specified location
    pub fn recommend_steal<'self_, const INCLUDE_CENTER: bool, const CACHE_ITER_MASKS: bool>(
        &'self_ self,
        local_worker: &BitRef<'self_, CACHE_ITER_MASKS>,
        task_location: StealLocation,
        update: Ordering,
    ) {
        // Check if there are job-less neighbors to submit work to...
        let Some(mut asleep_neighbors) = self
            .work_availability
            .iter_unset_around::<INCLUDE_CENTER, CACHE_ITER_MASKS>(local_worker, Ordering::Acquire)
        else {
            return;
        };

        // ...and if so, tell the closest one about our newly submitted job
        #[cold]
        fn unlikely<'self_, const CACHE_ITER_MASKS: bool>(
            self_: &'self_ SharedState,
            asleep_neighbors: impl Iterator<Item = BitRef<'self_, false>>,
            local_worker: &BitRef<'self_, CACHE_ITER_MASKS>,
            task_location: StealLocation,
            update: Ordering,
        ) {
            // Iterate over increasingly remote job-less neighbors
            //
            // Need Acquire ordering so the futex is read after the status flag
            let local_worker = local_worker.linear_idx(&self_.work_availability);
            for closest_asleep in asleep_neighbors {
                // Update their futex recommendation as appropriate
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

/// External interface to a single worker in a thread pool
#[derive(Debug)]
pub(crate) struct WorkerInterface {
    /// A way to steal from the worker
    pub stealer: Stealer<DynJob>,

    /// Futex that the worker sleeps on when it has nothing to do, used to
    /// instruct it what to do when it is awakened.
    pub futex: WorkerFutex,
}
//
impl WorkerInterface {
    /// Set up a worker's work queue and external interface
    pub fn with_work_queue() -> (Self, deque::Worker<DynJob>) {
        let worker = deque::Worker::new_lifo();
        let interface = Self {
            stealer: worker.stealer(),
            futex: WorkerFutex::new(),
        };
        (interface, worker)
    }
}
