//! State shared between the thread pool and all of its workers

pub(crate) mod flags;
pub(crate) mod flat;
pub(crate) mod futex;
pub(crate) mod hierarchical;
pub(crate) mod job;

use self::{futex::WorkerFutex, job::DynJob};
use crossbeam::deque::{self, Steal, Stealer};
use hwlocality::{bitmap::BitmapIndex, cpu::cpuset::CpuSet, Topology};
use std::{
    borrow::Borrow,
    fmt::Debug,
    sync::{atomic::Ordering, Arc},
};

/// State shared between all thread pool users and workers
pub trait SharedState: Send + Sized + Sync + 'static {
    /// Set up the shared and worker-local state
    #[doc(hidden)]
    fn with_worker_config(
        topology: &Topology,
        affinity: impl Borrow<CpuSet>,
    ) -> (Arc<Self>, Box<[WorkerConfig<Self>]>);

    /// Access the worker interfaces
    #[doc(hidden)]
    fn worker_interfaces(&self) -> impl Iterator<Item = &'_ WorkerInterface<Self>>;

    /// Worker-private work availability bit
    ///
    /// Workers can use this to signal when they have extra work available to
    /// steal and when they stop having work available (and thus are looking for
    /// more work).
    #[doc(hidden)]
    type WorkerAvailability<'a>: Clone + Debug + Eq + PartialEq + WorkerAvailability
    where
        Self: 'a;

    /// Make a worker availability bit
    ///
    /// This accessor is meant to constructed by workers at thread pool
    /// initialization time and then retained for the entire lifetime of the
    /// thread pool. As a result, it is optimized for efficiency of repeated
    /// usage, but initial construction may be expensive.
    #[doc(hidden)]
    fn worker_availability(&self, worker_idx: usize) -> Self::WorkerAvailability<'_>;

    /// Enumerate workers with work available to steal at increasing distances
    /// from a certain "thief" worker
    #[doc(hidden)]
    fn find_work_to_steal<'result>(
        &'result self,
        thief_worker_idx: usize,
        thief_availability: &'result Self::WorkerAvailability<'result>,
    ) -> Option<impl Iterator<Item = usize> + 'result>;

    /// Given a worker with work available for stealing, find the closest cousin
    /// that doesn't have work, if any, and suggest that it steal from there
    ///
    /// There should be a `Release` barrier between the moment where work is
    /// pushed in the worker's work queue and the moment where work is signaled
    /// to be available like this. You can either bundle the `Release` barrier
    /// into this transaction, or put a separate `atomic::fence(Release)` before
    /// this transaction and make it `Relaxed` if you have multiple work
    /// availability signaling transactions to do.
    #[doc(hidden)]
    fn suggest_stealing_from_worker<'self_>(
        &'self_ self,
        target_worker_idx: usize,
        target_availability: &Self::WorkerAvailability<'self_>,
        update: Ordering,
    );

    /// Inject work from outside the thread pool, and tell the worker closest to
    /// the originating locality that doesn't have work about it
    #[doc(hidden)]
    fn inject_job(&self, job: DynJob<Self>, local_worker_idx: usize);

    /// Try to steal a job from the global work injector
    #[doc(hidden)]
    fn steal_from_injector(&self) -> Steal<DynJob<Self>>;
}

/// Minimal work availability bit interface
///
/// All methods return the former work availability state, and may be
/// implemented as a no-op that returns `None` if there is no other worker to
/// advertise work to.
#[doc(hidden)]
pub trait WorkerAvailability {
    /// Truth that this worker currently advertises having work available
    ///
    /// May return `None` if there is no other worker to advertise work to.
    fn is_set(&self, order: Ordering) -> Option<bool>;

    /// Notify the world that this worker has work available for stealing
    fn fetch_set(&self, order: Ordering) -> Option<bool>;

    /// Notify the world that this worker is looking for work
    fn fetch_clear(&self, order: Ordering) -> Option<bool>;
}

/// Internal state needed to configure a new worker
#[derive(Debug)]
#[doc(hidden)]
pub struct WorkerConfig<Shared: SharedState> {
    /// Work queue
    pub work_queue: deque::Worker<DynJob<Shared>>,

    /// CPU which this worker should be pinned to
    pub cpu: BitmapIndex,
}

/// External interface to a single worker in a thread pool
#[derive(Debug)]
#[doc(hidden)]
pub struct WorkerInterface<Shared: SharedState> {
    /// A way to steal from the worker
    pub stealer: Stealer<DynJob<Shared>>,

    /// Futex that the worker sleeps on when it has nothing to do, used to
    /// instruct it what to do when it is awakened.
    pub futex: WorkerFutex,
}

/// Prepare to add a new worker to the thread pool
///
/// This builds both the internal state that will be used to configure the
/// worker on startup and the external interface that will be used by the rest
/// of the world to communicate with the worker.
pub(crate) fn new_worker<Shared: SharedState>(
    cpu: BitmapIndex,
) -> (WorkerConfig<Shared>, WorkerInterface<Shared>) {
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
