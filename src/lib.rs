#![warn(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)]

pub mod flags;
mod futex;
mod job;

use crate::job::{DynJob, Job, Notify};
use crossbeam::deque::{self, Injector, Steal, Stealer};
use flags::AtomicFlags;
use futex::{StealLocation, WorkerFutex};
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
        atomic::{self, AtomicBool, Ordering},
        Arc,
    },
    thread::{JoinHandle, Thread},
    time::{Duration, Instant},
};

/// Busy-waiting time between declaring sleepiness and falling asleep
///
/// This is a compromise between avoiding sleep and wakeup latency on one side,
/// and keeping idle threads asleep on the other side.
const SLEEPY_DURATION: Duration = Duration::from_micros(100);

/// Sleep duration used for busy waiting
///
/// Need a non-zero sleep duration to trigger a scheduler yield on Linux.
const YIELD_DURATION: Duration = Duration::from_nanos(1);

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
        abort_on_panic(|| {
            // Schedule work execution
            // SAFETY: We wait for the job to complete before letting it go out
            //         of scope or otherwise touching it in any way, and panics
            //         are translated to aborts until it's done executing.
            unsafe { self.spawn_unchecked(job.as_dyn()) };

            // Wait for the end of job execution then synchronize
            while !finished.load(Ordering::Relaxed) {
                std::thread::park();
            }
            atomic::fence(Ordering::Acquire);
        });
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
        self.shared
            .recommend_steal(best_worker_idx, StealLocation::Injector, Ordering::Release);
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
struct SharedState {
    /// Global work injector
    injector: Injector<DynJob>,

    /// Worker interfaces
    workers: Box<[WorkerInterface]>,

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
            workers.push(worker);
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
    fn recommend_steal(&self, local_worker: usize, task_location: StealLocation, update: Ordering) {
        // Iterate over increasingly remote job-less neighbors
        //
        // Need Acquire ordering so the futex is read after the status flag
        for closest_asleep in self
            .work_availability
            .iter_unset_around(local_worker, Ordering::Acquire)
        {
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

/// External interface to a single worker in a thread pool
struct WorkerInterface {
    /// A way to steal from the worker
    stealer: Stealer<DynJob>,

    /// Futex that the worker sleeps on when it has nothing to do, used to
    /// instruct it what to do when it is awakened.
    futex: WorkerFutex,
}
//
impl WorkerInterface {
    /// Set up a worker's work queue and external interface
    fn with_work_queue() -> (Self, deque::Worker<DynJob>) {
        let worker = deque::Worker::new_lifo();
        let interface = Self {
            stealer: worker.stealer(),
            futex: WorkerFutex::new(),
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
    futex: &'pool WorkerFutex,

    /// Work queue
    work_queue: deque::Worker<DynJob>,

    /// Private copy of our bit in the shared work_availability flags
    ///
    /// This ensures that in the happy path, we don't need to access any shared
    /// variable subjected to concurrent access cache contention, besides the
    /// state of our work queue.
    work_available: Cell<bool>,

    /// Timestamp where we entered the sleepy state
    sleepy_timestamp: Cell<Option<Instant>>,

    /// Truth that we reached the end of observable work and expect no more work
    work_over: Cell<bool>,
}
//
impl<'pool> Worker<'pool> {
    /// Set up and run the worker
    pub fn run(shared: &'pool SharedState, idx: usize, work_queue: deque::Worker<DynJob>) {
        let futex = &shared.workers[idx].futex;
        let worker = Self {
            shared,
            idx,
            futex,
            work_queue,
            work_available: Cell::new(false),
            sleepy_timestamp: Cell::new(None),
            work_over: Cell::new(false),
        };
        worker.main();
    }

    /// Main worker loop
    fn main(&self) {
        while !self.work_over.get() {
            self.step();
        }
    }

    /// Single step of the main worker loop
    #[inline]
    fn step(&self) {
        // If we have recorded work in our private work queue...
        if self.work_available.get() {
            // Process work from our private work queue, if still there
            if let Some(task) = self.work_queue.pop() {
                self.process_task(task);
                return;
            } else {
                // Otherwise notify others that our work queue is now empty
                //
                // Use Release ordering to make sure this happens after emptying
                // the work queue.
                self.shared
                    .work_availability
                    .fetch_clear(self.idx, Ordering::Release);
                self.work_available.set(false);
            }
        }

        // Look for work elsewhere using our trusty futex as a guide
        self.look_for_work();
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
    fn look_for_work(&self) {
        // Need an up-to-date futex readout as steal location can evolve
        //
        // Acquire ordering ensures we see the world like the previous threads
        // that updated the futex.
        let futex_state = self.futex.load_from_worker(Ordering::Acquire);

        // Any futex event clears the futex sleepy flag, which we should
        // acknowledge by resetting our internal sleepy timer.
        if self.sleepy_timestamp.get().is_some() && !futex_state.is_sleepy() {
            self.sleepy_timestamp.set(None);
        }

        // Check if we've been recommended to steal from one specific location
        if let Some(location) = futex_state.steal_location() {
            // Try to steal from that recommended location
            let successful = match location {
                StealLocation::Worker(idx) => self.steal_from_worker(idx),
                StealLocation::Injector => self.steal_from_injector(),
            };

            // Once stealing starts to fail, the recommandation has become
            // outdated, so discard it: we'll try exhaustive search next unless
            // another recommendation has come up in meantime.
            //
            // Here we can use Relaxed because no one else should read from our
            // futex location (we're just talking to ourself) and we aren't
            // using the updated futex state at the moment.
            if !successful {
                let _new_futex_state = self.futex.clear_outdated_location(
                    futex_state,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
            }
        } else {
            // No particular recommended stealing location, try to steal a task
            // from all possible locations
            if let Some(location) = self.steal_from_anyone() {
                // Record any place where we found work so we can try to steal
                // from there right away next time, unless a better
                // recommendation has concurrently come up.
                //
                // Can use Relaxed because we are talking to ourselves and we
                // aren't using the updated futex state at the moment.
                let _new_futex_state = self.futex.suggest_steal(
                    location,
                    self.idx,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
            } else if futex_state.work_incoming() {
                // No work available for now, but more work might still be
                // coming, just wait for it after warning others
                if futex_state.is_sleepy() {
                    // Start with some busy waiting...
                    if self.sleepy_timestamp.get().unwrap().elapsed() < SLEEPY_DURATION {
                        std::thread::sleep(YIELD_DURATION)
                    } else {
                        // ...then truly fall asleep after a while
                        //
                        // Synchronize with other threads manipulating the futex
                        // during out sleep.
                        let _new_futex_state = self.futex.wait_for_change(
                            futex_state,
                            Ordering::Release,
                            Ordering::Relaxed,
                        );
                    }
                } else {
                    // Tell others we're going to busy-wait and then sleep if
                    // nothing happens soon + start associated timer.
                    //
                    // Synchronize with other threads manipulating the futex.
                    self.sleepy_timestamp.set(Some(Instant::now()));
                    let _new_futex_state =
                        self.futex
                            .notify_sleepy(futex_state, Ordering::Release, Ordering::Acquire);
                }
            } else {
                // No work available for now and no more work expected, nothing
                // to do (this should only be a transient state when we are
                // waiting for a signal that will be sent by other workers
                // processing their own work)
                self.work_over.set(true);
                std::thread::sleep(YIELD_DURATION)
            }
        }
    }

    /// Try to steal work from one worker, identified by index in shared tables
    ///
    /// Return truth that a task was successfully stolen and run.
    fn steal_from_worker(&self, idx: usize) -> bool {
        // NOTE: It may seem that upon failure to steal, we should clear the
        // corresponding work_availability flag to tell other workers that there
        // is no work left to steal here. However, this would create an ABA
        // problem : the worker from which we're trying to steal might have
        // concurrently pushed new work, and then our assessment that there's no
        // work to steal is outdated and should not be published to the world.
        //
        // We could try to resolve this ABA problem by modifying the
        // work_available flags so that they contain not one bit per worker in
        // an AtomicUsize word, but two : truth that there's work available to
        // steal (work_available), and truth that new work has been pushed since
        // our last failed work-stealing attempt (new_work). With this extra
        // state, we could imagine the following synchronization protocol...
        //
        // Thief side steal algorithm:
        //
        // - Attempt to steal work, exit on success
        // - On failure, clear new_work with fetch_and(Acquire)
        // - Attempt to steal work again, exit on success
        // - On failure, use compare_exchange_weak to tell if new_work is still
        //   unset, and if so clear work_available too, with a complicated CAS
        //   recovery procedure on failure.
        //
        // Worker side push algorithm:
        //
        // - Push work
        // - Set new_work and work_available with fetch_or(Release).
        //
        // ...but going in this direction would have many serious drawbacks:
        //
        // - It would add the overhead of a RMW operation every time a worker
        //   pushes work, even on the happy path where nobody is attempting to
        //   steal any work from this worker.
        //     * The worker can't avoid this overhead because no matter when it
        //       last checked work_available, another thread might have cleared
        //       it since, and only the worker can set it back.
        // - I'm not convinced that this algorithm is robust in the presence of
        //   multiple concurrent thieves and multiple worker pushes. Ensuring
        //   correctness in this situation might require even more per-worker
        //   state and an even more complex/costly synchronization protocol.
        // - In any case, this would already massively complicate the
        //   synchronization protocol, which increases the risk of subtle
        //   concurrency bugs that can only be reproduced on weird hardware and
        //   decreases future maintainablity of the codebase.
        //
        // The current approach where the worker is the only one that can clear
        // the work_available flag instead ensures, with a relatively simple
        // synchronization protocol that does not entail any shared memory
        // operation on the worker happy path, that the work_available flag will
        // _eventually_ be cleared next time the worker checks its work queue
        // and figures out it's empty. There might be some useless work stealing
        // while the worker finishes processing its current task, but it seems
        // like a fair price to pay in exchange for a clean synchronization
        // protocol and a cheap happy path when every worker stays fed.
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
    fn steal_from_anyone(&self) -> Option<StealLocation> {
        // Try to steal from other workers at increasing distances
        //
        // Need Acquire so stealing happens after checking work availability.
        for idx in self
            .shared
            .work_availability
            .iter_set_around(self.idx, Ordering::Acquire)
        {
            if self.steal_from_worker(idx) {
                return Some(StealLocation::Worker(idx));
            }
        }

        // Try to steal from the global injector
        self.steal_from_injector()
            .then_some(StealLocation::Injector)
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
        let notify = NotifyFutex {
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

            // Execute thread pool work while waiting for remote task,
            // synchronize with the remote task once it completes
            while !remote_finished.load(Ordering::Relaxed) {
                self.0.step();
            }
            atomic::fence(Ordering::Acquire);
            local_result
        });

        // Return local and remote results, propagating panics
        // SAFETY: Collecting the remote result is safe because we have waited
        //         for the end of the job and the completion signal has been
        //         acknowledged with an Acquire memory barrier.
        (result_or_panic(local_result), unsafe {
            remote_job.result_or_panic()
        })
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
    /// notification has not been received. This entails in particular that all
    /// code including spawn_unchecked until the point where the remote task has
    /// signaled completion should translate unwinding panics to aborts.
    #[inline]
    unsafe fn spawn_unchecked(&self, job: DynJob) {
        // Schedule the work to be executed
        self.0.work_queue.push(job);

        // ...and once this push is visible...
        atomic::fence(Ordering::Release);

        // ...tell the world that we now have work available to steal if they
        // didn't know about it before...
        let shared = &self.0.shared;
        let me = self.0.idx;
        if !self.0.work_available.get() {
            self.0.work_available.set(true);
            shared.work_availability.fetch_set(me, Ordering::Relaxed);
        }

        // ...and personally notify the closest starving thread about it
        self.0
            .shared
            .recommend_steal(me, StealLocation::Worker(me), Ordering::Relaxed);
    }
}
//
// TODO: If I add a safe spawn(), bind its callable on F: 'scope, add tracking
//       of spawned tasks and make the function that created the scope ensure
//       that they are all finished before returning.

/// Mechanism to notify worker threads of join() completion
struct NotifyFutex<'stack> {
    /// Flag to be set once the remote job of this join() is finished
    remote_finished: &'stack AtomicBool,

    /// Futex of the worker thread to be awakened, if sleeping
    futex: &'stack WorkerFutex,
}
//
// SAFETY: remote_finished is set with Release ordering and is the signal that
//         the worker uses to synchronize.
unsafe impl Notify for NotifyFutex<'_> {
    fn notify(self) {
        self.remote_finished.store(true, Ordering::Release);
        self.futex.wake(Ordering::Release);
    }
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

/// Translate unwinding panics into aborts
fn abort_on_panic<R>(f: impl FnOnce() -> R) -> R {
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(_) => std::process::abort(),
    }
}

/// Extract the result or propagate the panic from a `thread::Result`
fn result_or_panic<R>(result: std::thread::Result<R>) -> R {
    match result {
        Ok(result) => result,
        Err(payload) => std::panic::resume_unwind(payload),
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
