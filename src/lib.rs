#![warn(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)]

pub mod flags;

use crossbeam::deque::{self, Injector, Steal, Stealer};
use flags::AtomicFlags;
use hwlocality::{
    bitmap::BitmapIndex,
    cpu::{binding::CpuBindingFlags, cpuset::CpuSet},
    topology::Topology,
};
use std::{
    cell::Cell,
    collections::HashMap,
    panic::AssertUnwindSafe,
    sync::{
        atomic::{self, AtomicBool, AtomicU32, Ordering},
        Arc, Mutex,
    },
    thread::JoinHandle,
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
                    .name(format!("FlatPool thread #{worker_idx} on CPU #{cpu}"))
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
    pub fn scope<Body, Res>(&self, body: Body) -> Res
    where
        Body: AsyncCallback<Res>,
        Res: Send,
    {
        // Propagate worker thread panics
        for worker in &self.workers {
            if worker.is_finished() {
                panic!("a worker thread has panicked");
            }
        }

        // From the point where the task is scheduled, until the point where it
        // has signaled end of execution, panics should translate into aborts
        let result = abort_on_panic(|| {
            // Schedule work execution
            let me = std::thread::current();
            let result = Mutex::new(None);
            // SAFETY: This is fine because we wait for work completion below
            unsafe {
                let result = &result;
                self.spawn_unchecked(move |scope| {
                    // Run the callable, use mutex poisoning to notify the caller
                    // about any panic during task execution
                    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        let mut lock = result.lock().unwrap();
                        *lock = Some(body(scope));
                    }));

                    // Notify the caller that the task has been processed
                    me.unpark();
                });
            }

            // Wait for work completion and collect result
            std::thread::park();
            result
        });
        let result = result.lock().unwrap().take().unwrap();
        result
    }

    /// Schedule work for execution on the thread pool, without lifetime checks
    ///
    /// The closest worker thread is hinted to pick this work up, but it may be
    /// picked up by any other worker thread in the pool.
    ///
    /// Although an UnwindSafe bound on f cannot be enforced as a result of
    /// AssertUnwindSafe being limited to FnOnce() only, closures passed to this
    /// function should propagate panics to their client. Failure to do so will
    /// break the thread pool and likely result in a program abort.
    ///
    /// # Safety
    ///
    /// The input callable is allowed to borrow variables from the surrounding
    /// scope, and it is the caller's responsibility to ensure that said scope
    /// is not exited before the caller is done executing. This notably entails
    /// that the caller is not allowed do panic, but should abort on panics.
    ///
    /// This function is guaranteed not to panic after scheduling the callable.
    /// More precisely, panics will be transleted to aborts.
    unsafe fn spawn_unchecked(&self, f: impl AsyncCallback<()>) {
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
            .unwrap_or(self.workers.len() / 2);

        // Schedule the work to be executed
        let f = Box::new(f);
        // SAFETY: Per safety precondition, the caller promises that it will
        //         keep any state borrowed by f live until f is done executing,
        //         and therefore casting f to 'static is unobservable.
        let f = unsafe {
            std::mem::transmute::<Box<dyn AsyncCallback<()>>, Box<dyn AsyncCallback<()> + 'static>>(
                Box::new(f),
            )
        };
        self.shared.injector.push(f);

        // Find the nearest available thread and recommend it to process this
        self.shared
            .recommend_steal(best_worker_idx, WORK_FROM_INJECTOR);
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
        for worker in self.shared.workers.iter() {
            worker.futex.store(WORK_OVER, Ordering::Release);
            atomic_wait::wake_all(&worker.futex);
        }

        // Join all the worker thread
        for worker in self.workers.drain(..) {
            worker.join().unwrap();
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
    fn with_work_queues(num_workers: usize) -> (Arc<Self>, Box<[deque::Worker<Task>]>) {
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

    /// Recommend that the work-less thread closest to a certain originating
    /// locality steal a task from the specified location
    ///
    /// # Panics
    ///
    /// `task_location` must not be `WORK_OVER` nor `WORK_NOWHERE`, since these
    /// are not actual task locations. The code will panic upon encountering.
    /// these location values.
    fn recommend_steal(&self, local_worker: usize, task_location: u32) {
        // Iterate over increasingly remote job-less neighbors
        'search: for closest_asleep in self
            .steal_flags
            .iter_unset_around(local_worker, Ordering::Relaxed)
        {
            // Update their futex recommendation as appropriate
            let closest_futex = &self.workers[closest_asleep].futex;
            let closest_asleep_u32 = closest_asleep as u32;
            let mut futex_value = closest_futex.load(Ordering::Relaxed);
            'compare_exchange: loop {
                // Once it is determined that the current stealing
                // recommendation is worse than stealing from us, call this
                // macro to update the recommendation.
                macro_rules! try_recommend {
                    () => {
                        match closest_futex.compare_exchange_weak(
                            futex_value,
                            task_location,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => {
                                // Recommendation updated, wake up the thread if
                                // needed and go away
                                if futex_value == WORK_NOWHERE {
                                    atomic_wait::wake_all(closest_futex);
                                }
                                break 'search;
                            }
                            Err(new_futex_value) => {
                                futex_value = new_futex_value;
                                continue 'compare_exchange;
                            }
                        }
                    };
                }

                // Analyze current value of the neighbor's futex
                match (futex_value, task_location) {
                    // Never override the thread pool's stop signal
                    (WORK_OVER, _) => {
                        break 'search;
                    }

                    // Only actual task locations are accepted
                    (_, WORK_OVER | WORK_NOWHERE) => unreachable!(),

                    // If the neighbor currently has nothing to do, any location
                    // is accepted. But any other worker is considered more
                    // local than the injector.
                    (WORK_NOWHERE, _) => {
                        try_recommend!();
                    }
                    (_, WORK_FROM_INJECTOR) => {
                        continue 'search;
                    }
                    (WORK_FROM_INJECTOR, _) => {
                        try_recommend!();
                    }

                    // At this point, it is clear that both recommendation point
                    // to actual worker indices. Accept new recommendation if it
                    // is closer than the previous recommendation...
                    (old_idx, new_idx)
                        if (closest_asleep_u32.abs_diff(old_idx)
                            > closest_asleep_u32.abs_diff(new_idx)) =>
                    {
                        try_recommend!()
                    }

                    // ...otherwise, leave the existing recommendation alone
                    _ => {
                        continue 'search;
                    }
                }
            }
        }
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
    fn with_work_queue() -> (Self, deque::Worker<Task>) {
        let worker = deque::Worker::new_lifo();
        let interface = Self {
            stealer: worker.stealer(),
            futex: AtomicU32::new(WORK_NOWHERE),
        };
        (interface, worker)
    }
}

/// Worker thread
struct Worker<'pool> {
    /// Access to the shared state
    shared: &'pool SharedState,

    /// Index of this thread in the shared state tables
    idx: usize,

    /// Quick access to this thread's futex
    futex: &'pool AtomicU32,

    /// Work queue
    work_queue: deque::Worker<Task>,

    /// Truth that there might still be work coming from other worker threads or
    /// from thread pool clients.
    work_incoming: Cell<bool>,
}
//
impl<'pool> Worker<'pool> {
    /// Set up and run the worker
    pub fn run(shared: &'pool SharedState, idx: usize, work_queue: deque::Worker<Task>) {
        let worker = Self {
            shared,
            idx,
            futex: &shared.workers[idx].futex,
            work_queue,
            work_incoming: Cell::new(true),
        };
        worker.main();
    }

    /// Main worker loop
    fn main(&self) {
        while self.work_incoming.get() {
            self.step(true);
        }
    }

    /// Single step of the main worker loop
    #[inline]
    fn step(&self, can_sleep: bool) {
        // Process work from our private work queue
        if let Some(task) = self.work_queue.pop() {
            self.process_task(task);
        } else {
            // Signal that there's no work to steal from us for now
            self.shared
                .steal_flags
                .fetch_clear(self.idx, Ordering::Relaxed);

            // Look for work elsewhere using our trusty futex as a guide
            self.look_for_work(can_sleep);
        }
    }

    /// Process one incoming task
    #[inline]
    fn process_task(&self, task: Task) {
        let scope = Scope::new(self);
        task(&scope);
    }

    /// Look for work using our futex as a guide
    fn look_for_work(&self, can_sleep: bool) {
        match self.futex.load(Ordering::Acquire) {
            // Thread pool is shutting down, so if we can't find
            // a task to steal now, we won't ever find one
            WORK_OVER => {
                if self.steal_from_anyone().is_none() {
                    self.work_incoming.set(false);
                }
            }

            // No particular work stealing recommandation, check all
            // possibilities and go to sleep if we don't find any work anywhere
            WORK_NOWHERE => {
                if let Some(location) = self.steal_from_anyone() {
                    // Record any place where we found work so we can try to
                    // steal from there right away next time.
                    let _ = self.futex.compare_exchange(
                        WORK_NOWHERE,
                        location,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    );
                } else {
                    // No work found, go to sleep if allowed to do so
                    if can_sleep {
                        atomic_wait::wait(self.futex, WORK_NOWHERE);
                    }
                }
            }

            // There is a recommandation to steal work from somewhere
            work_location => {
                // Try to steal from the current recommended location
                let succeeded = match work_location {
                    WORK_FROM_INJECTOR => self.steal_from_injector(),
                    idx => self.steal_from_worker(idx as usize),
                };

                // Once stealing starts to fail, the recommandation becomes
                // outdated, so discard it: we'll try exhaustive search next.
                if !succeeded {
                    let _ = self.futex.compare_exchange(
                        work_location,
                        WORK_NOWHERE,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    );
                }
            }
        }
    }

    /// Try to steal work from one worker, identified by index in shared tables
    ///
    /// Return truth that a task was successfully stolen and run.
    fn steal_from_worker(&self, idx: usize) -> bool {
        self.steal(|| self.shared.workers[idx].stealer.steal())
    }

    /// Try to steal work from the global task injector
    ///
    /// Return truth that a task was successfully stolen and run.
    fn steal_from_injector(&self) -> bool {
        self.steal(|| self.shared.injector.steal())
    }

    /// Try to steal work from all possible sources
    ///
    /// Return from which source work was stolen (if any), using the conventions
    /// of `self.futex`, so that `self.futex` can be updated if appropriate.
    fn steal_from_anyone(&self) -> Option<u32> {
        // Try to steal from other workers at increasing distances
        for idx in self
            .shared
            .steal_flags
            .iter_set_around(self.idx, Ordering::Acquire)
        {
            if self.steal_from_worker(idx) {
                return Some(idx as u32);
            }
        }

        // Try to steal from the global injector
        self.steal_from_injector().then_some(WORK_FROM_INJECTOR)
    }

    /// Try to steal work using the specified procedure
    ///
    /// Return truth that a task was successfully stolen and run.
    fn steal(&self, mut attempt: impl FnMut() -> Steal<Task>) -> bool {
        loop {
            match attempt() {
                Steal::Success(task) => {
                    self.process_task(task);
                    return true;
                }
                Steal::Empty => return false,
                Steal::Retry => continue,
            }
        }
    }
}

/// Scope for executing parallel work
///
/// This is a token which attests that work is executing within the context of a
/// worker thread in the thread pool, and can be used to schedule work on said
/// thread pool.
pub struct Scope<'scope>(AssertUnwindSafe<&'scope Worker<'scope>>);
//
impl<'scope> Scope<'scope> {
    /// Provide an opportunity for fork-join parallelism
    ///
    /// Run the `local` task, while leaving the `remote` task available for
    /// other worker threads to steal. If no other thread takes over the job, do
    /// it ourselves. Wait for both tasks to be complete before moving on, while
    /// participating to thread pool execution in meantime.
    pub fn join<LocalFn, LocalRes, RemoteFn, RemoteRes>(
        &self,
        local: LocalFn,
        remote: RemoteFn,
    ) -> (LocalRes, RemoteRes)
    where
        LocalFn: FnOnce() -> LocalRes,
        RemoteFn: AsyncCallback<RemoteRes>,
        RemoteRes: Send,
    {
        // No unwinding panics allowed until the remote task has completed
        let (local_result, remote_result) = abort_on_panic(|| {
            // Spawn remote task
            // SAFETY: This is safe because we will wait for remote work to
            //         complete before exiting the scope of borrowed variables,
            //         and panics are translated to aborts until the remote task
            //         is done executing.
            let remote_finished = AtomicBool::new(false);
            let remote_result = Mutex::new(None);
            unsafe {
                let futex = self.0.futex;
                let remote_finished = &remote_finished;
                let remote_result = &remote_result;
                self.spawn_unchecked(move |scope| {
                    // Run the callable, use mutex poisoning to notify the
                    // caller about any panic during task execution
                    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        let mut result_lock = remote_result.lock().unwrap();
                        *result_lock = Some(remote(scope));
                    }));

                    // Notify the worker thread that the task has been processed
                    remote_finished.store(true, Ordering::Release);
                    // TODO: See if I can find a way to avoid calling wake when
                    //       the thread is awake without causing a race
                    //       condition when it's concurrently falling asleep.
                    atomic_wait::wake_all(futex);
                })
            };

            // Run local task
            let local_result = std::panic::catch_unwind(AssertUnwindSafe(local));

            // Execute thread pool work while waiting for remote task
            while !remote_finished.load(Ordering::Relaxed) {
                // FIXME: Figure out if I can allow sleep here with a more
                //        clever futex protocol.
                self.0.step(false);
            }
            atomic::fence(Ordering::Acquire);
            (local_result, remote_result)
        });

        // Collect and return results
        let remote_result = remote_result.lock().unwrap().take().unwrap();
        (local_result.unwrap(), remote_result)
    }

    /// Set up a scope associated with a particular worker thread
    fn new(worker: &'scope Worker<'scope>) -> Self {
        Self(AssertUnwindSafe(worker))
    }

    /// Schedule work for execution on the thread pool, without lifetime checks
    ///
    /// The work is scheduled on the active worker's thread work queue, but it
    /// may be picked up by any other worker thread in the thread pool.
    ///
    /// Although an UnwindSafe bound on f cannot be enforced as a result of
    /// AssertUnwindSafe being limited to FnOnce() only, closures passed to this
    /// function should propagate panics to their client. Failure to do so will
    /// break the thread pool and likely result in a program abort.
    ///
    /// # Safety
    ///
    /// The input callable is allowed to borrow variables from the surrounding
    /// scope, and it is the caller's responsibility to ensure that said scope
    /// is not exited before the caller is done executing.
    ///
    /// This entails in particular that all code including spawn_unchecked until
    /// the point where the remote task has signaled completion should translate
    /// unwinding panics to aborts.
    unsafe fn spawn_unchecked(&self, f: impl AsyncCallback<()>) {
        // Schedule the work to be executed
        let f = Box::new(f);
        // SAFETY: Per safety precondition, the caller promises that it will
        //         keep any state borrowed by f live until f is done executing,
        //         and therefore casting f to 'static is unobservable.
        let f = unsafe {
            std::mem::transmute::<Box<dyn AsyncCallback<()>>, Box<dyn AsyncCallback<()> + 'static>>(
                Box::new(f),
            )
        };
        self.0.work_queue.push(f);

        // Notify others that we have work available for stealing.
        atomic::fence(Ordering::Release);
        let shared = &self.0.shared;
        let me = self.0.idx;
        if !shared.steal_flags.is_set(me, Ordering::Relaxed) {
            shared.steal_flags.fetch_set(me, Ordering::Relaxed);
        }
        self.0.shared.recommend_steal(me, me as u32);
    }
}
//
// TODO: If I add a safe spawn(), bind its callable on F: 'scope, add tracking
//       of spawned tasks and make the function that created the scope ensure
//       that they are all finished before returning.

/// A task is basically an FnOnce trait object
///
/// For now, we always implement it as Box<dyn FnOnce>, but we may try
/// allocation elision optimizations later on if memory allocator overhead
/// becomes a problem.
///
/// The callable is run by a particular worker thread and receives a Scope that
/// allows it to interact with said thread.
type Task = Box<dyn AsyncCallback<()>>;

/// Asynchronous callback, to be scheduled on the thread pool
pub trait AsyncCallback<Res: Send>: for<'scope> FnOnce(&Scope<'scope>) -> Res + Send {}
//
impl<Res, Body> AsyncCallback<Res> for Body
where
    Res: Send,
    Body: for<'scope> FnOnce(&Scope<'scope>) -> Res + Send,
{
}

/// Translate unwinding panics to aborts
pub fn abort_on_panic<R>(f: impl FnOnce() -> R) -> R {
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(_) => std::process::abort(),
    }
}

/// Reference computation of the N-th fibonacci term
pub fn fibonacci_ref(n: u64) -> u64 {
    if n > 0 {
        let sqrt_5 = 5.0f64.sqrt();
        let phi = (1.0 + sqrt_5) / 2.0;
        let f_n = phi.powi(i32::try_from(n).unwrap()) / sqrt_5;
        f_n.round() as u64
    } else {
        0
    }
}

/// Recursive parallel fibonacci based on FlatPool
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
        flat.scope(|scope| {
            for i in 0..=34 {
                assert_eq!(fibonacci_flat(scope, i), fibonacci_ref(i));
            }
        });
    }
}
