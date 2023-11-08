//! Thread pool-wide interfaces

use crate::{
    flags::AtomicFlags,
    futex::{StealLocation, WorkerFutex},
    job::{DynJob, Job, Notify},
    worker::{Worker, WorkerInterface},
    AbortGuard, Work,
};
use crossbeam::{
    deque::{self, Injector},
    utils::CachePadded,
};
use hwlocality::{
    bitmap::BitmapIndex,
    cpu::{binding::CpuBindingFlags, cpuset::CpuSet},
    topology::Topology,
};
use std::{
    collections::HashMap,
    sync::{
        atomic::{self, AtomicBool, Ordering},
        Arc,
    },
    thread::{JoinHandle, Thread},
};

/// Simple flat pinned thread pool, used to check hypothesis that pinning and
/// avoidance of TLS alone won't let us significantly outperform rayon
pub struct FlatPool {
    /// Shared state
    shared: Arc<SharedState>,

    /// Hwloc topology
    topology: Arc<Topology>,

    /// Mapping from OS CPU to worker thread index
    cpu_to_worker: HashMap<BitmapIndex, usize>,

    /// Worker threads
    workers: Vec<JoinHandle<()>>,
}
//
impl FlatPool {
    /// Create a thread pool
    pub fn new() -> Self {
        // Probe the hwloc topology
        let topology = Arc::new(Topology::new().unwrap());
        let cpuset = topology.cpuset();

        // Set up the shared state and work queues
        let (shared, work_queues) = SharedState::with_work_queues(cpuset.weight().unwrap());

        // Start worker threads
        let workers = cpuset
            .iter_set()
            .zip(work_queues.into_vec())
            .enumerate()
            .map(|(worker_idx, (cpu, work_queue))| {
                let topology = topology.clone();
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("FlatPool worker #{worker_idx} (CPU {cpu})"))
                    .spawn(move || {
                        // Pin the worker thread to its assigned CPU
                        topology
                            .bind_cpu(
                                &CpuSet::from(cpu),
                                CpuBindingFlags::THREAD | CpuBindingFlags::STRICT,
                            )
                            .unwrap();

                        // We won't need the hwloc topology again after this
                        std::mem::drop(topology);

                        // Start processing work
                        Worker::run(&shared, worker_idx, work_queue);
                    })
                    .unwrap()
            })
            .collect();

        // Record the mapping from OS CPU to worker thread index
        let cpu_to_worker = cpuset
            .iter_set()
            .enumerate()
            .map(|(idx, cpu)| (cpu, idx))
            .collect();

        // Put it all together
        Self {
            shared,
            topology,
            cpu_to_worker,
            workers,
        }
    }

    /// Synchronously execute work inside of the thread pool
    pub fn run<W, Res>(&self, work: W) -> Res
    where
        W: Work<Res>,
        Res: Send,
    {
        // Propagate worker thread panics
        for worker in &self.workers {
            if worker.is_finished() {
                panic!("a worker thread has panicked");
            }
        }

        // Prepare job for execution, notify completion via thread unparking
        let finished = AtomicBool::new(false);
        let notify = NotifyParked {
            finished: &finished,
            thread: std::thread::current(),
        };
        let mut job = Job::new(notify, work);

        // From the point where the task is scheduled, until the point where it
        // has signaled end of execution, panics should translate into aborts
        {
            // Schedule work execution
            // SAFETY: We wait for the job to complete before letting it go out
            //         of scope or otherwise touching it in any way, and panics
            //         are translated to aborts until it's done executing.
            let abort = AbortGuard;
            unsafe { self.spawn_unchecked(job.as_dyn()) };

            // Wait for the end of job execution then synchronize
            while !finished.load(Ordering::Relaxed) {
                std::thread::park();
            }
            atomic::fence(Ordering::Acquire);
            std::mem::forget(abort);
        }
        // SAFETY: We waited for the job to finish before collecting the result
        //         and used an Acquire barrier to synchronize
        unsafe { job.result_or_panic() }
    }

    /// Schedule work for execution on the thread pool, without lifetime checks
    ///
    /// The closest worker thread is hinted to pick this work up, but it may be
    /// picked up by any other worker thread in the pool.
    ///
    /// # Safety
    ///
    /// The [`Job`] API contract must be honored as long as the completion
    /// notification has not been received. This entails in particular that all
    /// code including spawn_unchecked until the point where the remote task has
    /// signaled completion should translate unwinding panics to aborts.
    unsafe fn spawn_unchecked(&self, job: DynJob) {
        // Determine the caller's current CPU location to decide which worker
        // thread would be best placed for processing this task
        let caller_cpu = self
            .topology
            .last_cpu_location(CpuBindingFlags::THREAD)
            .unwrap();
        let best_worker_idx = self
            .cpu_to_worker
            .get(&caller_cpu.first_set().unwrap())
            .copied()
            // FIXME: Pick true closest CPU
            .unwrap_or(self.workers.len() / 2);

        // Schedule the work to be executed
        self.shared.injector.push(job);

        // Find the nearest available thread and recommend it to process this
        //
        // Need Release ordering to make sure they see the pushed work
        self.shared.recommend_steal::<true>(
            best_worker_idx,
            StealLocation::Injector,
            Ordering::Release,
        );
    }
}
//
impl Default for FlatPool {
    fn default() -> Self {
        Self::new()
    }
}
//
impl Drop for FlatPool {
    fn drop(&mut self) {
        // Tell workers that no further work will be coming and wake them all up
        //
        // Need Release ordering to make sure they see all previous pushed work
        for worker in self.shared.workers.iter() {
            worker.futex.notify_shutdown(Ordering::Release);
        }

        // Join all the worker thread
        for worker in self.workers.drain(..) {
            worker.join().unwrap();
        }
    }
}

/// Job completion notification that unparks a thread
struct NotifyParked<'flag> {
    /// Flag to be set to notify completion
    finished: &'flag AtomicBool,

    /// Thread to be unparked
    thread: Thread,
}
//
// SAFETY: finished is set with Release ordering and is the signal that the main
//         thread uses to synchronize.
unsafe impl Notify for NotifyParked<'_> {
    fn notify(self) {
        self.finished.store(true, Ordering::Release);
        self.thread.unpark()
    }
}

/// State that is shared between users of the thread pool and all thread pool
/// workers
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
    fn with_work_queues(num_workers: usize) -> (Arc<Self>, Box<[deque::Worker<DynJob>]>) {
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
    ///
    /// # Panics
    ///
    /// `task_location` must not be `WORK_OVER` nor `WORK_NOWHERE`, since these
    /// are not actual task locations. The code will panic upon encountering.
    /// these location values.
    pub(crate) fn recommend_steal<const INCLUDE_CENTER: bool>(
        &self,
        local_worker: usize,
        task_location: StealLocation,
        update: Ordering,
    ) {
        // Check if there are job-less neighbors to submit work to
        let Some(asleep_neighbors) = self
            .work_availability
            .iter_unset_around::<INCLUDE_CENTER>(local_worker, Ordering::Acquire)
        else {
            return;
        };

        // Iterate over increasingly remote job-less neighbors
        //
        // Need Acquire ordering so the futex is read after the status flag
        for closest_asleep in asleep_neighbors {
            // Update their futex recommendation as appropriate
            let accepted = self.workers[closest_asleep].futex.suggest_steal(
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle() {
        // Check that thread pool initializes and shuts down correctly
        FlatPool::new();
    }

    #[test]
    fn fibonacci() {
        let flat = FlatPool::new();
        flat.run(|scope| {
            for i in 0..=34 {
                assert_eq!(crate::fibonacci_flat(scope, i), crate::fibonacci_ref(i));
            }
        });
    }
}
