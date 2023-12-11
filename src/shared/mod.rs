//! State shared between the thread pool and all of its workers

pub(crate) mod flags;
pub(crate) mod futex;
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
use std::sync::{atomic::Ordering, Arc};

/// State that is shared between users of the thread pool and all thread pool
/// workers
#[derive(Debug, Default)]
pub(crate) struct SharedState {
    /// Global work injector
    injector: Injector<DynJob>,

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
    pub fn recommend_stealing<'self_, const INCLUDE_CENTER: bool, const CACHE_ITER_MASKS: bool>(
        &'self_ self,
        local_worker: &BitRef<'self_, CACHE_ITER_MASKS>,
        task_location: StealLocation,
        update: Ordering,
    ) {
        // Check if there are job-less neighbors to submit work to...
        //
        // Need Acquire ordering so the futex is read after the work
        // availability flag, no work availability caching/speculation allowed.
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
            let local_worker = local_worker.linear_idx(&self_.work_availability);
            for closest_asleep in asleep_neighbors {
                // Update their futex recommendation as appropriate
                //
                // Can use Relaxed ordering on failure because failing to
                // suggest work to a worker has no observable consequences and
                // isn't used to inform any decision other than looking up the
                // state of the next worker. Worker states are independent.
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

    /// Enumerate workers that a thief could steal work from, at increasing
    /// distance from said thief
    pub fn find_workers_to_rob<'self_, const CACHE_ITER_MASKS: bool>(
        &'self_ self,
        thief: &BitRef<'self_, CACHE_ITER_MASKS>,
    ) -> Option<impl Iterator<Item = usize> + '_> {
        let work_availability = &self.work_availability;
        work_availability
            // We need Acquire ordering so that when we later attempt to steal
            // from the worker, we are guaranteed to see the work in its work
            // queue, if no one else stole it since.
            //
            // We don't need stronger-than-Acquire ordering because we're not
            // trying to get anyone else in sync with us (which is what Release
            // is for) and we're not trying to get all thread to agree on a
            // global order of searches for work (which is what SeqCst is for).
            .iter_set_around::<false, CACHE_ITER_MASKS>(thief, Ordering::Acquire)
            .map(|iter| iter.map(|bit| bit.linear_idx(work_availability)))
    }

    /// Inject work into the thread pool
    ///
    /// # Safety
    ///
    /// The [`Job`] API contract must be honored as long as the completion
    /// notification has not been received. This entails in particular that all
    /// code including spawn_unchecked until the point where the remote task has
    /// signaled completion should translate unwinding panics to aborts.
    pub unsafe fn inject_job(&self, job: DynJob, local_worker_idx: usize) {
        // Schedule the work to be executed
        self.injector.push(job);

        // Find the nearest available thread and recommend it to process this
        //
        // Need Release ordering to make sure they see the pushed work
        self.recommend_stealing::<true, false>(
            &self.work_availability.bit(local_worker_idx),
            StealLocation::Injector,
            Ordering::Release,
        );
    }

    /// Steal a job from the global work injector
    pub fn steal_from_injector(&self) -> Steal<DynJob> {
        self.injector.steal()
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
