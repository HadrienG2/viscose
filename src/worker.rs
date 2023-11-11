//! Single thread pool worker

use crate::{
    flags::bitref::BitRef,
    job::{DynJob, Job, Notify},
    shared::{
        futex::{StealLocation, WorkerFutex},
        SharedState,
    },
    AbortGuard, Work, MAX_SPIN_ITERS_PER_CHECK, OS_WAIT_DELAY, SPIN_ITERS_BEFORE_YIELD,
    YIELD_DURATION,
};
use crossbeam::deque::{self, Steal};
use std::{
    cell::Cell,
    panic::AssertUnwindSafe,
    sync::atomic::{self, AtomicBool, Ordering},
    time::Instant,
};

/// Worker thread
pub(crate) struct Worker<'pool> {
    /// Access to the shared state
    shared: &'pool SharedState,

    /// Index of this thread in the shared state tables
    idx: usize,

    /// Bit of this thread in `SharedState::work_availability`
    work_available_bit: BitRef<'pool, true>,

    /// Truth that `work_available_bit` is currently set
    ///
    /// This ensures that in the happy path, we don't need to access any shared
    /// variable subjected to concurrent access cache contention, besides the
    /// state of our work queue.
    work_available_set: Cell<bool>,

    /// Quick access to this thread's futex
    futex: &'pool WorkerFutex,

    /// Work queue
    work_queue: deque::Worker<DynJob>,

    /// Timestamp at which we entered the sleepy state, assuming we're sleepy
    sleepy_start: Cell<Option<Instant>>,

    /// Number of spin-loop iterations since we last yielded to the OS
    spin_iters_since_yield: Cell<usize>,

    /// Number of spin-loop iterations between attempts to look for work
    spin_iters_per_check: Cell<u8>,

    /// Truth that we reached the end of observable work and expect no more work
    ///
    /// This will lead the thread to shut down once all pending join() have been
    /// fully processed.
    work_over: Cell<bool>,
}
//
impl<'pool> Worker<'pool> {
    /// Set up and run the worker
    pub fn run(shared: &'pool SharedState, idx: usize, work_queue: deque::Worker<DynJob>) {
        let futex = &shared.workers[idx].futex;
        let bit = shared.work_availability.bit_with_cache(idx);
        let worker = Self {
            shared,
            idx,
            work_available_bit: bit,
            futex,
            work_queue,
            work_available_set: Cell::new(false),
            sleepy_start: Cell::new(None),
            spin_iters_since_yield: Cell::new(0),
            spin_iters_per_check: Cell::new(1),
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
    fn step(&self) {
        // If we have recorded work in our private work queue...
        if self.work_available_set.get() {
            // Process work from our private work queue, if still there
            if let Some(task) = self.work_queue.pop() {
                self.process_task(task);
                return;
            } else {
                // Otherwise notify others that our work queue is now empty
                //
                // Use Release ordering to make sure this happens after emptying
                // the work queue.
                self.work_available_bit.fetch_clear(Ordering::Release);
                self.work_available_set.set(false);
            }
        }

        // Look for work elsewhere using our trusty futex as a guide
        self.look_for_work();
    }

    /// Process one incoming task
    fn process_task(&self, job: DynJob) {
        let scope = Scope::new(self);
        // SAFETY: All methods that push [`DynJob`]s into the thread pool ensure
        //         that the associated [`Job`] cannot go out of scope until it
        //         is done executing.
        unsafe { job.run(&scope) };
    }

    /// Look for work using our futex as a guide
    #[cold]
    fn look_for_work(&self) {
        // Need an up-to-date futex readout as steal location can evolve
        //
        // Acquire ordering ensures we see the world like the previous threads
        // that updated the futex.
        let futex_state = self.futex.load_from_worker(Ordering::Acquire);

        // Any futex event clears the futex sleepy flag, which we should
        // acknowledge by resetting our internal sleepy state.
        if self.sleepy_start.get().is_some() && !futex_state.is_sleepy() {
            self.sleepy_start.set(None);
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
                    let mut spin_iters_per_check = self.spin_iters_per_check.get();
                    let spin_iters_since_yield = self.spin_iters_since_yield.get();
                    if spin_iters_since_yield <= SPIN_ITERS_BEFORE_YIELD {
                        if spin_iters_per_check < MAX_SPIN_ITERS_PER_CHECK {
                            spin_iters_per_check =
                                (2 * spin_iters_per_check).min(MAX_SPIN_ITERS_PER_CHECK);
                            self.spin_iters_per_check.set(spin_iters_per_check);
                        }
                        for _ in 0..spin_iters_per_check {
                            std::hint::spin_loop()
                        }
                    } else if self.sleepy_start.get().unwrap().elapsed() < OS_WAIT_DELAY {
                        std::thread::sleep(YIELD_DURATION);
                        self.spin_iters_since_yield.set(0);
                    } else {
                        // ...then truly fall asleep after a while
                        //
                        // Synchronize with other threads manipulating the futex
                        // during our sleep.
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
                    self.sleepy_start.set(Some(Instant::now()));
                    self.spin_iters_since_yield.set(0);
                    self.spin_iters_per_check.set(1);
                    let _new_futex_state =
                        self.futex
                            .notify_sleepy(futex_state, Ordering::Release, Ordering::Relaxed);
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
    #[inline(always)]
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
        // Are there other workers we could steal work from?
        if let Some(neighbors_with_work) = self
            .shared
            .work_availability
            .iter_set_around::<false, true>(&self.work_available_bit, Ordering::Acquire)
        {
            // Try to steal from other workers at increasing distances
            //
            // Need Acquire so stealing happens after checking work availability.
            for bit in neighbors_with_work {
                let idx = bit.linear_idx(&self.shared.work_availability);
                if self.steal_from_worker(idx) {
                    return Some(StealLocation::Worker(idx));
                }
            }
        }

        // Try to steal from the global injector
        self.steal_from_injector()
            .then_some(StealLocation::Injector)
    }

    /// Try to steal work using the specified procedure
    ///
    /// Return truth that a task was successfully stolen and run.
    #[inline(always)]
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
        let local_result = {
            // Spawn remote task
            // SAFETY: We wait for the job to complete before letting it go out
            //         of scope or otherwise touching it in any way, and panics
            //         are translated to aborts until it's done executing.
            let abort = AbortGuard;
            unsafe { self.spawn_unchecked(remote_job.as_dyn()) };

            // Run local task
            let local_result = std::panic::catch_unwind(AssertUnwindSafe(local));

            // Execute thread pool work while waiting for remote task,
            // synchronize with the remote task once it completes
            while !remote_finished.load(Ordering::Relaxed) {
                self.0.step();
            }
            atomic::fence(Ordering::Acquire);
            std::mem::forget(abort);
            local_result
        };

        // Return local and remote results, propagating panics
        // SAFETY: Collecting the remote result is safe because we have waited
        //         for the end of the job and the completion signal has been
        //         acknowledged with an Acquire memory barrier.
        (crate::result_or_panic(local_result), unsafe {
            remote_job.result_or_panic()
        })
    }

    /// Numerical identifier of the worker thread this job runs on
    pub fn worker_id(&self) -> usize {
        self.0.idx
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
    #[inline(always)]
    unsafe fn spawn_unchecked(&self, job: DynJob) {
        // Schedule the work to be executed
        self.0.work_queue.push(job);

        // ...and once this push is visible...
        atomic::fence(Ordering::Release);

        // ...tell the world that we now have work available to steal if they
        // didn't know about it before...
        if !self.0.work_available_set.get() {
            self.0.work_available_set.set(true);
            self.0.work_available_bit.fetch_set(Ordering::Relaxed);
        }

        // ...and personally notify the closest starving thread about it
        self.0.shared.recommend_steal::<false, true>(
            &self.0.work_available_bit,
            StealLocation::Worker(self.0.idx),
            Ordering::Relaxed,
        );
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
