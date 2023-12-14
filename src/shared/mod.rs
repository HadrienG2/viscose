//! State shared between the thread pool and all of its workers

pub(crate) mod distances;
pub(crate) mod flags;
pub(crate) mod futex;
pub(crate) mod job;

use self::{
    distances::{Distance, Distances},
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
    sync::{atomic::Ordering, Arc},
};

/// State that is shared between users of the thread pool and all thread pool
/// workers
#[derive(Debug)]
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

    /// Distances between workers
    pub distances: Distances,
}
//
impl SharedState {
    /// Set up the shared and worker-local state
    pub fn with_worker_config(
        topology: &Topology,
        affinity: impl Borrow<CpuSet>,
    ) -> (Arc<Self>, Box<[WorkerConfig]>) {
        // Collect the PU objects that fit in our affinity mask, and cross-check
        // that the count is valid
        let mut pus = topology
            .pus_from_cpuset(affinity.borrow())
            .collect::<Vec<_>>();
        let num_workers = pus.len();
        assert_ne!(
            num_workers, 0,
            "a thread pool without threads can't make progress and will deadlock on first request"
        );
        assert!(
            num_workers < WorkerFutex::MAX_WORKERS,
            "unsupported number of worker threads"
        );

        // Compute and display distances between workers
        let distances = Distances::measure_and_sort(&mut pus[..]);

        // Set up worker-local state
        let mut worker_configs = Vec::with_capacity(num_workers);
        let mut worker_interfaces = Vec::with_capacity(num_workers);
        for pu in pus {
            let cpu = BitmapIndex::try_from(pu.os_index().expect("PUs should have an OS index"))
                .expect("PU OS indices should fit in a BitmapIndex (since they fit in a CpuSet)");
            let (config, interface) = new_worker(cpu);
            worker_configs.push(config);
            worker_interfaces.push(CachePadded::new(interface));
        }

        // Set up global shared state
        let result = Arc::new(Self {
            injector: Injector::new(),
            workers: worker_interfaces.into(),
            work_availability: AtomicFlags::new(num_workers),
            distances,
        });
        (result, worker_configs.into())
    }

    /// Enumerate workers that a thief could steal work from, at increasing
    /// distance from said thief
    pub fn find_workers_to_rob<'self_, const CACHE_SEARCH_MASKS: bool>(
        &'self_ self,
        thief: &BitRef<'self_, CACHE_SEARCH_MASKS>,
        distances_from_worker: &'self_ [Distance],
    ) -> Option<impl Iterator<Item = usize> + 'self_> {
        let work_availability = &self.work_availability;
        work_availability
            // Need at least Acquire ordering to ensure that the work we
            // observed to be available is actually visible in the worker's
            // queue, and don't need anything stronger:
            //
            // - Don't need AcqRel (which would require replacing the load
            //   with a RMW) since we're not trying to get any other thread
            //   in sync with our current state.
            // - Don't need SeqCst since there is no need for everyone to
            //   agree on the global order in which workers look for work.
            .iter_set_around::<false, CACHE_SEARCH_MASKS>(
                thief,
                distances_from_worker,
                Ordering::Acquire,
            )
            .map(|iter| iter.map(|bit| bit.linear_idx(work_availability)))
    }

    /// Given a worker that has just pushed new work in its work queue, find the
    /// closest worker that is looking for work (if any) and suggest that it
    /// steal from the pushing worker.
    ///
    /// There should be a `Release` barrier between the moment where the task is
    /// pushed and the moment where this notification is sent, so that the
    /// recipient of the notification is guaranteed to observe the freshly
    /// pushed work. You can either bundle the `Release` barrier into this
    /// transaction or put a `Release` fence before this transaction.
    #[inline]
    pub fn suggest_stealing_from_worker<'self_>(
        &'self_ self,
        local_worker: &BitRef<'self_, true>,
        distances_from_worker: &'self_ [Distance],
        update: Ordering,
    ) {
        self.suggest_stealing::<false, true>(
            local_worker,
            distances_from_worker,
            StealLocation::Worker(local_worker.linear_idx(&self.work_availability)),
            update,
        );
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
        self.injector.push(job);
        self.suggest_stealing::<true, false>(
            &self.work_availability.bit(local_worker_idx),
            self.distances.from(local_worker_idx),
            StealLocation::Injector,
            // Need at least Release ordering to ensure injected job is visible
            // to the target and don't need anything stronger:
            //
            // - Don't need AcqRel ordering since the thread that's pushing work
            //   does not want to get in sync with the target worker's state.
            // - Don't need SeqCst since there is no need for everyone to agree
            //   on the global order of job injection events.
            Ordering::Release,
        );
    }

    /// Steal a job from the global work injector
    pub fn steal_from_injector(&self) -> Steal<DynJob> {
        self.injector.steal()
    }

    /// Given a location where a new task has just been pushed and an assessment
    /// of which worker would be best placed to process this task, find the
    /// nearest worker that's looking for work (if any) and suggest that it
    /// steal from the recommended location.
    ///
    /// `INCLUDE_CENTER` should be `false` when the task has been pushed into
    /// the local worker's work queue (in this case the worker obviously has
    /// work, so we should not include it in the search space for workers
    /// looking for work). It should be `true` when the task is pushed into the
    /// global injector, since any worker is potentially interested then.
    ///
    /// There should be a `Release` barrier between the moment where the task is
    /// pushed and the moment where this notification is sent, so that the
    /// recipient of the notification is guaranteed to observe the freshly
    /// pushed work. You can either bundle the `Release` barrier into this
    /// transaction or put a `Release` fence before this transaction.
    #[inline]
    fn suggest_stealing<'self_, const INCLUDE_CENTER: bool, const CACHE_SEARCH_MASKS: bool>(
        &'self_ self,
        local_worker: &BitRef<'self_, CACHE_SEARCH_MASKS>,
        distances_from_worker: &'self_ [Distance],
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
                distances_from_worker,
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
#[doc(hidden)]
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
