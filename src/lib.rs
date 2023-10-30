mod flags;

use crossbeam::deque::{Injector, Steal, Stealer, Worker};
use flags::AtomicFlags;
use hwlocality::{
    cpu::{binding::CpuBindingFlags, cpuset::CpuSet},
    topology::Topology,
};
use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    thread::JoinHandle,
};

/// Simple flat pinned thread pool, used to check hypothesis that pinning and
/// avoidance of TLS alone won't let us significantly outperform rayon
struct FlatPool {
    /// Shared state
    shared: Arc<SharedState>,

    /// Worker threads
    workers: Box<[JoinHandle<()>]>,
}
//
impl FlatPool {
    /// Create a thread pool
    pub fn new() -> Self {
        // Probe the hwloc topology
        let topology = Arc::new(Topology::new().unwrap());

        // Set up the shared state and work queues
        let (shared, work_queues) =
            SharedState::with_work_queues(topology.cpuset().weight().unwrap());

        // Start worker threads
        let workers = topology
            .cpuset()
            .iter_set()
            .zip(work_queues.into_iter())
            .enumerate()
            .map(|(my_idx, (cpu, work_queue))| {
                let topology = topology.clone();
                let shared = shared.clone();
                std::thread::spawn(move || {
                    // TODO: Modularize this big function into an OO struct

                    // Bind the worker to its assigned CPU
                    topology
                        .bind_cpu(
                            &CpuSet::from(cpu),
                            CpuBindingFlags::ASSUME_SINGLE_THREAD | CpuBindingFlags::STRICT,
                        )
                        .unwrap();

                    // We won't need the hwloc topology again after this
                    std::mem::drop(topology);

                    // Process a task, irrespective of where we got it from
                    let process_task = |task: Task| {
                        // TODO: Do the work
                        // TODO: Handle panics within the task + completion
                        //       notifications
                        // TODO: If we spawn tasks here, do a Release fence, set
                        //       flag to notify others that we have work to
                        //       steal, then look for the closest neighboring
                        //       sleeping thread, make ourselves its
                        //       recommendation if we're closer than the
                        //       previous recommendation using a futex CAS, and
                        //       wake it up.
                        unimplemented!()
                    };

                    // Try to steal and execute a task from another worker,
                    // return truth that this operation was successful
                    let try_steal_worker = |idx: usize| loop {
                        match shared.workers[idx].stealer.steal() {
                            Steal::Success(task) => {
                                process_task(task);
                                return true;
                            }
                            Steal::Empty => return false,
                            Steal::Retry => continue,
                        }
                    };

                    // Try to steal and execute a task from the global injector
                    let try_steal_injector = || loop {
                        match shared.injector.steal() {
                            Steal::Success(task) => {
                                process_task(task);
                                return true;
                            }
                            Steal::Empty => return false,
                            Steal::Retry => continue,
                        }
                    };

                    // Access our personal shard of the shared state
                    let my = &shared.workers[my_idx];

                    // Last-resort work stealing strategy: try to steal
                    // everywhere at increasing distances from this worker,
                    // using CPU ID distance as a rough distance metric
                    let num_workers = shared.workers.len();
                    let try_steal_anywhere = |update_futex_from_nowhere: bool| {
                        // If we find a place to steal from, update our futex so
                        // we check there immediately next time.
                        let update_futex = |new_value: u32| {
                            if update_futex_from_nowhere {
                                my.futex.compare_exchange(
                                    WORK_NOWHERE,
                                    new_value,
                                    Ordering::Relaxed,
                                    Ordering::Relaxed,
                                )
                            }
                        };

                        // Check other workers at increasing distances
                        for idx in shared
                            .steal_flags
                            .iter_set_around(my_idx, Ordering::Acquire)
                        {
                            if try_steal_worker(idx) {
                                update_futex(idx);
                                return true;
                            }
                        }

                        // Check global injector
                        let robbed_injector = try_steal_injector();
                        if robbed_injector {
                            update_futex(WORK_FROM_INJECTOR);
                        }
                        robbed_injector
                    };

                    // Main worker loop
                    'main: loop {
                        // Process all work from our private work queue
                        while let Some(task) = work_queue.pop() {
                            process_task(task);
                        }
                        shared.steal_flags.fetch_clear(my_idx, Ordering::Relaxed);

                        // We ran out of private work, check our futex to know
                        // where we should try to steal from next
                        match my.futex.load(Ordering::Acquire) {
                            // Thread pool is shutting down, so if we can't find
                            // a task to steal now, we won't ever find one
                            WORK_OVER => {
                                if !try_steal_anywhere(false) {
                                    break 'main;
                                }
                            }

                            // No particular recommandation, look around in
                            // order of decreasing locality and go to sleep if
                            // we don't find any work anywhere
                            WORK_NOWHERE => {
                                if !try_steal_anywhere(true) {
                                    // TODO: Add a grace period of spinning if
                                    //       it can be proven beneficial
                                    atomic_wait::wait(&my.futex, WORK_NOWHERE);
                                }
                            }

                            // There is a recommandation to steal somewhere
                            work_available => {
                                // Try to steal from the recommended location
                                let succeeded = match work_available {
                                    WORK_FROM_INJECTOR => try_steal_injector(),
                                    idx => try_steal_worker(idx),
                                };

                                // Once stealing starts to fail, the
                                // recommandation has become outdated, so it's
                                // time to remove it
                                if !succeeded {
                                    my.futex.compare_exchange(
                                        work_available,
                                        WORK_NOWHERE,
                                        Ordering::Relaxed,
                                        Ordering::Relaxed,
                                    )
                                }
                            }
                        }
                    }
                })
            })
            .collect();
        Self { shared, workers }
    }

    // TODO: Add scope() method and associated struct, steal from std::thread
    // TODO: Make scope.join() wake up the sleeping thread closest to the
    //       current thread, if any, with an appropriate futex recommendation.
    //       Worker threads can remember which CPU they're pinned to and tell
    //       the scope constructor, but to also handle threads external to the
    //       thread pool we leave an Option at None and we query the current CPU
    //       location with hwloc right before taking this decision.
    // TODO: Catch worker thread panics and propagate them to the main thread
    //       upon attempts to create tasks.
}
//
impl Drop for FlatPool {
    fn drop(&mut self) {
        // Tell workers that no further work will be coming and wake them all up
        for worker in self.shared.workers.iter() {
            worker.futex.store(WORK_OVER, Ordering::Release);
            atomic_wait::wake_all(&worker.futex);
        }
    }
}

/// State that is shared between users of the thread pool and all thread pool
/// workers
struct SharedState {
    /// Global work injector
    injector: Injector<Task>,

    /// Worker interfaces
    workers: Box<[WorkerInterface]>,

    /// Per-worker truth that each worker _might_ have work ready to be stolen
    /// inside of its work queue
    ///
    /// Set by the associated worker when pushing work, cleared by the worker
    /// when it tries to pop work and the queue is empty. In an ideal world,
    /// other workers could update it quicker by clearing it when they fail to
    /// steal work, but this would race with the setting of the flag, and it's
    /// better to have the flag set when it shouldn't be than to have it cleared
    /// when it should be set.
    ///
    /// The flag is set with Release ordering, so if the readout of a set flag
    /// is performed with Acquire ordering, the pushed work is observable.
    steal_flags: AtomicFlags,
}
//
impl SharedState {
    /// Set up the shared state
    fn with_work_queues(num_workers: usize) -> (Arc<Self>, Box<[Worker<Task>]>) {
        assert!(
            num_workers < usize::try_from(WORK_SPECIAL_START).unwrap(),
            "unsupported number of worker threads"
        );
        let injector = Injector::new();
        let mut work_queues = Vec::with_capacity(num_workers);
        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let (worker, work_queue) = WorkerInterface::with_work_queue();
            workers.push(worker);
            work_queues.push(work_queue);
        }
        let result = Arc::new(Self {
            injector,
            workers: workers.into(),
            steal_flags: AtomicFlags::new(num_workers),
        });
        (result, work_queues.into())
    }
}

/// External interface to a single worker in a thread pool
struct WorkerInterface {
    /// A way to steal from the worker
    stealer: Stealer<Task>,

    /// Futex that the worker sleeps on when it has nothing to do, used to
    /// instruct it what to do when it is awakened.
    ///
    /// - Most values designate the index of a thread that the freshly awakened
    ///   worker should steal from
    /// - WORK_OVER means that the flat pool has been dropped, so no more work
    ///   will be coming up and the worker should exit as soon as it has
    ///   exhausted its work pile.
    /// - WORK_NOWHERE means that there is no work left to steal but more work
    ///   might still be coming up, and the worker should fall to sleep when it
    ///   has no work to do in its work queue  (this is used to handle spurious
    ///   wakeup).
    /// - WORK_FROM_INJECTOR means that there is work available in the global
    ///   work injector.
    /// - WORK_SPECIAL_START is set to the lowest special value of the futex and
    ///   used to detect when the number of workers exceeds the implementation's
    ///   capabilities.
    ///
    futex: AtomicU32,
}
//
/// `Worker::work_futex` value used to signal that the thread pool was dropped
const WORK_OVER: u32 = u32::MAX;
//
/// `Worker::work_futex` value used to signal that there is no work available at
/// the moment and the thread should fall back asleep.
const WORK_NOWHERE: u32 = u32::MAX - 1;
//
/// `Worker::work_futex` value used to signal that there is work available from
/// the thread pool's top level injector.
const WORK_FROM_INJECTOR: u32 = u32::MAX - 2;
//
/// Lowest special value of `Worker::work_futex` aka maximal number of worker
/// threads that this thread pool implementation supports.
const WORK_SPECIAL_START: u32 = u32::MAX - 2;
//
impl WorkerInterface {
    /// Set up a worker's work queue and external interface
    fn with_work_queue() -> (Self, Worker<Task>) {
        let worker = Worker::new_lifo();
        let interface = Self {
            stealer: worker.stealer(),
            futex: AtomicU32::new(WORK_NOWHERE),
        };
        (interface, worker)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference computation of the N-th fibonacci term
    fn fibonacci_ref(n: u64) -> u64 {
        if n > 0 {
            let sqrt_5 = 5.0f64.sqrt();
            let phi = (1.0 + sqrt_5) / 2.0;
            let F_n = phi.powi(i32::try_from(n).unwrap()) / sqrt_5;
            F_n.round() as u64
        } else {
            0
        }
    }

    /// Recursive parallel fibonacci based on FlatPool
    fn fibonacci_flat(scope: &Scope<'_, '_>, n: u64) -> u64 {
        if n > 1 {
            scope.join(
                |scope| fibonacci_flat(scope, n - 1),
                move |scope| fibonacci_flat(scope, n - 2),
            )
        } else {
            n
        }
    }

    #[test]
    fn lifecycle() {
        // Check that thread pool initializes and shuts down correctly
        FlatPool::new();
    }

    #[test]
    fn fibonacci() {
        let flat = FlatPool::new();
        flat.scope(|scope| {
            assert_eq!(fibonacci_flat(scope, 32), fibonacci_ref(32));
        });
    }
}
