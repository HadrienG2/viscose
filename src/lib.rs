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
    cell::{Cell, UnsafeCell},
    collections::HashMap,
    panic::AssertUnwindSafe,
    sync::{
        atomic::{self, AtomicBool, AtomicU32, Ordering},
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

        // Prepare job for execution, notify completion via thread park token
        let mut job = Job::new(std::thread::current(), work);

        // From the point where the task is scheduled, until the point where it
        // has signaled end of execution, panics should translate into aborts
        abort_on_panic(|| {
            // Schedule work execution
            // SAFETY: We wait for the job to complete before letting it go out
            //         of scope or otherwise touching it in any way, and panics
            //         are translated to aborts until it's done executing.
            unsafe { self.spawn_unchecked(job.as_dyn()) };

            // Wait for the end of job execution
            std::thread::park();
        });
        // SAFETY: We waited for the job to finish before collecting the result,
        //         and thread::park() wakeup has >= Acquire memory ordering.
        unsafe { job.result() }
    }

    /// Schedule work for execution on the thread pool, without lifetime checks
    ///
    /// The closest worker thread is hinted to pick this work up, but it may be
    /// picked up by any other worker thread in the pool.
    ///
    /// # Safety
    ///
    /// The [`Job`] API contract must be honored as long as the completion
    /// notification has not been received. This enteils in particular that all
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

// SAFETY: Thread::unpark() calls into OS primitives with >=Release ordering
unsafe impl Notify for Thread {
    fn notify(self) {
        self.unpark()
    }
}

/// State that is shared between users of the thread pool and all thread pool
/// workers
struct SharedState {
    /// Global work injector
    injector: Injector<DynJob>,

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
    fn with_work_queues(num_workers: usize) -> (Arc<Self>, Box<[deque::Worker<DynJob>]>) {
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
    stealer: Stealer<DynJob>,

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
    fn with_work_queue() -> (Self, deque::Worker<DynJob>) {
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
    work_queue: deque::Worker<DynJob>,

    /// Truth that there might still be work coming from other worker threads or
    /// from thread pool clients.
    work_incoming: Cell<bool>,
}
//
impl<'pool> Worker<'pool> {
    /// Set up and run the worker
    pub fn run(shared: &'pool SharedState, idx: usize, work_queue: deque::Worker<DynJob>) {
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
    fn process_task(&self, job: DynJob) {
        let scope = Scope::new(self);
        // SAFETY: All methods that push [`DynJob`]s into the thread pool ensure
        //         that the associated [`Job`] cannot go out of scope until it
        //         is done executing.
        unsafe { job.run(&scope) };
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
    fn steal(&self, mut attempt: impl FnMut() -> Steal<DynJob>) -> bool {
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
        RemoteFn: Work<RemoteRes>,
        RemoteRes: Send,
    {
        // Set up remote job and its completion notification mechanism
        let remote_finished = AtomicBool::new(false);
        let notify = NotifyJoin {
            remote_finished: &remote_finished,
            futex: self.0.futex,
        };
        let mut remote_job = Job::new(notify, remote);

        // No unwinding panics allowed until the remote task has completed
        let local_result = abort_on_panic(|| {
            // Spawn remote task
            // SAFETY: We wait for the job to complete before letting it go out
            //         of scope or otherwise touching it in any way, and panics
            //         are translated to aborts until it's done executing.
            unsafe { self.spawn_unchecked(remote_job.as_dyn()) };

            // Run local task
            let local_result = std::panic::catch_unwind(AssertUnwindSafe(local));

            // Execute thread pool work while waiting for remote task
            while !remote_finished.load(Ordering::Relaxed) {
                // FIXME: Figure out if I can allow sleep here with a more
                //        clever futex protocol.
                self.0.step(false);
            }
            atomic::fence(Ordering::Acquire);
            local_result
        });

        // Collect and return results
        // SAFETY: Collecting the remote result is safe because we have waited
        //         for the end of the job and the completion signal has been
        //         acknowledged with an Acquire memory barrier.
        (local_result.unwrap(), unsafe { remote_job.result() })
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
    /// # Safety
    ///
    /// The [`Job`] API contract must be honored as long as the completion
    /// notification has not been received. This enteils in particular that all
    /// code including spawn_unchecked until the point where the remote task has
    /// signaled completion should translate unwinding panics to aborts.
    unsafe fn spawn_unchecked(&self, job: DynJob) {
        // Schedule the work to be executed
        self.0.work_queue.push(job);

        // Notify other workers that we have work available for stealing.
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

/// Mechanism to notify worker threads of join() completion
struct NotifyJoin<'stack> {
    /// Flag to be set once the remote job of this join() is finished
    remote_finished: &'stack AtomicBool,

    /// Futex of the worker thread to be awakened, if sleeping
    futex: &'stack AtomicU32,
}
//
// SAFETY: remote_finished is set with Release ordering
unsafe impl Notify for NotifyJoin<'_> {
    fn notify(self) {
        self.remote_finished.store(true, Ordering::Release);
        // TODO: See if I can find a way to avoid calling wake when
        //       the thread is awake without causing a race
        //       condition when it's concurrently falling asleep.
        atomic_wait::wake_all(self.futex);
    }
}

/// [`Work`] that has been prepared for execution by the thread pool
///
/// # Safety
///
/// Safe use of [`Job`] requires carefully following the following procedure:
///
/// - Create a [`Job`] on the stack frame where it will be executed.
/// - Create a type-erased job handle with `as_dyn()` and submit it for
///   execution on the thread pool.
/// - Until the job completion signal is received, do not exit the current stack
///   frame or interact with the Job in any way, including but not limited to...
///     - Moving or dropping the job
///     - Calling any Job method
/// - Once a job completion signal has been received with Acquire memory
///   ordering, you may extract the result and propagate panics with `result()`.
struct Job<Res: Send, ImplWork: Work<Res>, ImplNotify: Notify>(
    UnsafeCell<JobState<Res, ImplWork, ImplNotify>>,
);
//
impl<Res: Send, ImplWork: Work<Res>, ImplNotify: Notify> Job<Res, ImplWork, ImplNotify> {
    /// Prepare [`Work`] for execution by the thread pool
    pub fn new(notify: ImplNotify, work: ImplWork) -> Self {
        Self(UnsafeCell::new(JobState::Scheduled(notify, work)))
    }

    /// Create a type-erased handle that can be pushed on a work queue
    ///
    /// # Safety
    ///
    /// Should only be called once, as preparation for submitting the job to the
    /// thread pool.
    pub unsafe fn as_dyn(&mut self) -> DynJob {
        let state = self.0.get();
        let state = state.cast::<()>();
        let run = |state: *mut (), scope: &Scope<'_>| {
            let state = state.cast::<JobState<Res, ImplWork, ImplNotify>>();
            // SAFETY: Per `Job` API contract
            unsafe { (*state).run(scope) };
        };
        DynJob { state, run }
    }

    /// Extract the job result or propagate job panic
    ///
    /// # Safety
    ///
    /// Should only be called after the job completion notification has been
    /// received.
    pub unsafe fn result(mut self) -> Res {
        match std::mem::replace(self.0.get_mut(), JobState::Collected) {
            JobState::Scheduled(_, _) | JobState::Running => {
                panic!("Job result shouldn't be collected before completion notification")
            }
            JobState::Finished(result) => match result {
                Ok(result) => result,
                Err(payload) => std::panic::resume_unwind(payload),
            },
            JobState::Collected => unreachable!(),
        }
    }
}

/// [`Job`] state machine
enum JobState<Res: Send, ImplWork: Work<Res>, ImplNotify: Notify> {
    /// Job has not started executing yet
    Scheduled(ImplNotify, ImplWork),

    /// Job is executing on some worker thread
    Running,

    /// Job execution is complete and result can be collected
    Finished(std::thread::Result<Res>),

    /// Job result has been collected
    Collected,
}
//
impl<Res: Send, ImplWork: Work<Res>, ImplNotify: Notify> JobState<Res, ImplWork, ImplNotify> {
    /// Run the job
    ///
    /// This should only be called once, failure to do so will result in at
    /// least a panic and likely a full program abort.
    fn run(&mut self, scope: &Scope<'_>) {
        let Self::Scheduled(notify, work) = std::mem::replace(self, Self::Running) else {
            panic!("attempted to execute a Job in an invalid state");
        };
        *self = Self::Finished(std::panic::catch_unwind(AssertUnwindSafe(|| work(scope))));
        notify.notify()
    }
}

/// Type-erased handle to a [`Job`]
struct DynJob {
    /// Type-erased `&mut JobState<...>` pointer
    state: *mut (),

    /// Type-erased `JobState<...>::run()` method
    ///
    /// The first parameter must be `self.job`.
    run: fn(*mut (), &Scope<'_>),
}
//
impl DynJob {
    /// Execute the job
    ///
    /// # Safety
    ///
    /// See top-level [`Job`] documentation.
    pub unsafe fn run(self, scope: &Scope<'_>) {
        (self.run)(self.state, scope)
    }
}
//
// SAFETY: It is safe to send a DynJob to another thread because inner Work is
//         safe to send across thread and the Job API contract ensures that the
//         current thread will not touch the Job in any way until the other
//         thread is done with the DynJob, which means that for all intents and
//         purposes we effectively own the inner Work.
unsafe impl Send for DynJob {}

/// Mechanism to notify the program that a job is done executing
///
/// # Safety
///
/// No notification should be received until `self.notify()` is called, and the
/// notification process must feature an `Ordering::Release` memory barrier.
unsafe trait Notify {
    /// Send in the notification
    fn notify(self);
}

/// Function that can be scheduled for execution by the thread pool
///
/// The input [`Scope`] allows the scheduled work to interact with the thread
/// pool by e.g. spawning new tasks.
pub trait Work<Res: Send>: for<'scope> FnOnce(&Scope<'scope>) -> Res + Send {}
//
impl<Res, Body> Work<Res> for Body
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
#[inline]
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
        flat.run(|scope| {
            for i in 0..=34 {
                assert_eq!(fibonacci_flat(scope, i), fibonacci_ref(i));
            }
        });
    }
}
