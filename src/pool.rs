//! Thread pool-wide interfaces

use crate::{
    shared::{
        futex::StealLocation,
        job::{AbortOnUnwind, DynJob, Job, Notify},
        SharedState,
    },
    worker::Worker,
    Work,
};
use hwlocality::{
    bitmap::BitmapIndex,
    cpu::{binding::CpuBindingFlags, cpuset::CpuSet},
    topology::Topology,
};
use std::{
    borrow::Borrow,
    collections::HashMap,
    sync::{
        atomic::{self, AtomicBool, Ordering},
        Arc,
    },
    thread::{JoinHandle, Thread},
};

/// Simple flat pinned thread pool, used to check hypothesis that pinning and
/// avoidance of TLS alone won't let us significantly outperform rayon
#[derive(Debug)]
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
    /// Create a thread pool that uses all system CPU cores
    pub fn new() -> Self {
        let topology = Arc::new(Topology::new().unwrap());
        Self::with_affinity(topology, CpuSet::full())
    }

    /// Create a thread pool with certain CPU affinity
    ///
    /// Only CPU cores that belong to this CpuSet will be used.
    pub fn with_affinity(topology: Arc<Topology>, affinity: impl Borrow<CpuSet>) -> Self {
        // Set up the shared state and work queues
        let (shared, worker_configs) = SharedState::with_worker_config(&topology, affinity);

        // Start worker threads
        let mut cpu_to_worker = HashMap::with_capacity(worker_configs.len());
        let mut workers = Vec::with_capacity(worker_configs.len());
        for (worker_idx, worker_config) in worker_configs.into_vec().into_iter().enumerate() {
            let topology = topology.clone();
            let shared = shared.clone();
            workers.push(
                std::thread::Builder::new()
                    .name(format!(
                        "FlatPool worker #{worker_idx} (CPU {})",
                        worker_config.cpu
                    ))
                    .spawn(move || {
                        // Pin the worker thread to its assigned CPU
                        topology
                            .bind_cpu(
                                &CpuSet::from(worker_config.cpu),
                                CpuBindingFlags::THREAD | CpuBindingFlags::STRICT,
                            )
                            .unwrap();

                        // We won't need the hwloc topology again after this
                        std::mem::drop(topology);

                        // Start processing work
                        Worker::run(&shared, worker_idx, worker_config.work_queue);
                    })
                    .expect("failed to spawn worker thread"),
            );
            cpu_to_worker.insert(worker_config.cpu, worker_idx);
        }

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
            let abort_on_unwind = AbortOnUnwind;
            unsafe { self.spawn_unchecked(job.as_dyn()) };

            // Wait for the end of job execution then synchronize
            while !finished.load(Ordering::Relaxed) {
                std::thread::park();
            }
            atomic::fence(Ordering::Acquire);
            std::mem::forget(abort_on_unwind);
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
        self.shared.injector().push(job);

        // Find the nearest available thread and recommend it to process this
        //
        // Need Release ordering to make sure they see the pushed work
        self.shared.recommend_steal::<true, false>(
            &self.shared.work_availability.bit(best_worker_idx),
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
        for worker in self.shared.workers().iter() {
            worker.futex.notify_shutdown(Ordering::Release);
        }

        // Join all the worker thread
        for worker in self.workers.drain(..) {
            worker.join().unwrap();
        }
    }
}

/// Job completion notification that unparks a thread
#[derive(Clone, Debug)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle() {
        // Check that thread pool initializes and shuts down correctly
        FlatPool::new();
    }
}
