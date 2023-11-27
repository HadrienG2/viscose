//! Single thread pool worker

pub mod scope;

use self::scope::Scope;
use crate::{
    shared::{
        flags::bitref::BitRef,
        futex::{StealLocation, WorkerFutex, WorkerFutexState},
        job::DynJob,
        SharedState,
    },
    MAX_SPIN_ITERS_PER_CHECK, OS_WAIT_DELAY, SPIN_ITERS_BEFORE_YIELD, YIELD_DURATION,
};
use crossbeam::deque::{self, Steal};
use std::{cell::Cell, debug_assert, sync::atomic::Ordering, time::Instant};

/// Worker thread state
#[derive(Debug)]
pub(crate) struct Worker<'pool> {
    /// Access to the shared state
    shared: &'pool SharedState,

    /// Index of this thread in the shared state tables
    idx: usize,

    /// Work queue tracking work privately spawned by this worker
    work_queue: deque::Worker<DynJob>,

    /// Bit of this thread in `SharedState::work_availability`
    ///
    /// Tracks whether we think there is work available to steal in our work
    /// queue (this information may become obsolete if the work gets stolen)
    work_available: WorkAvailabilityBit<'pool>,

    /// Quick access to this thread's futex
    ///
    /// The futex directly or indirectly tracks all the information about the
    /// outside world that the worker cares about, so whenever the world
    /// changes, the futex value will change. This lets the worker fall asleep
    /// while waiting for the world to change (new work available to steal,
    /// `join()` finished...).
    futex: &'pool WorkerFutex,

    /// State used when the worker thread is waiting for work
    waiting_state: Cell<Option<WaitingState>>,

    /// Truth that we reached the end of observable work and expect no more work
    ///
    /// This will lead the thread to shut down once all pending `join()`s have
    /// been fully processed.
    work_over: Cell<bool>,
}
//
impl<'pool> Worker<'pool> {
    /// Set up and run the worker
    pub fn run(shared: &'pool SharedState, idx: usize, work_queue: deque::Worker<DynJob>) {
        let worker = Self {
            shared,
            idx,
            work_queue,
            work_available: WorkAvailabilityBit::new(shared, idx),
            futex: &shared
                .worker_interfaces()
                .nth(idx)
                .expect("worker not found in shared state")
                .futex,
            waiting_state: Cell::new(None),
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
    #[inline(always)]
    fn step(&self) {
        // If we have recorded work in our private work queue...
        if self.work_available.is_set() {
            // Process work from our private work queue, if still there
            if let Some(task) = self.work_queue.pop() {
                self.process_task(task);
                return;
            } else {
                // Otherwise notify others that our work queue is now empty
                //
                // Use Release ordering to make sure this is perceived to happen
                // after we actually empty the work queue.
                self.work_available.clear(Ordering::Release);
            }
        }

        // No work in our private work queue, figure out what to do next
        self.handle_starvation();
    }

    /// Process one incoming task
    fn process_task(&self, job: DynJob) {
        let scope = Scope::new(self);
        // SAFETY: All methods that push [`DynJob`]s into the thread pool ensure
        //         that the associated [`Job`] cannot go out of scope until it
        //         is done executing.
        unsafe { job.run(&scope) };
    }

    /// Handle absence of work in our private work queue
    #[cold]
    fn handle_starvation(&self) {
        // Learn more about the state of the world using our futex
        //
        // Acquire ordering ensures we see the world like the previous thread
        // that updated the futex did.
        let futex_state = self.futex.load_from_worker(Ordering::Acquire);

        // Try to steal work from other workers and the global injector, and
        // stop waiting on success.
        if self.steal_work(&futex_state) {
            self.waiting_state.set(None);
            return;
        }

        // At this point, we've failed to find more work. Figure out if more
        // work could still be coming up.
        if futex_state.work_incoming() {
            // If there isn't any work to steal anywhere, but more work might
            // still be coming up, wait for more work to come up...
            let mut waiting_state = self.waiting_state.get().unwrap_or_else(WaitingState::new);
            waiting_state.wait(|| {
                // ...and completely fall asleep after some grace period
                //
                // Make sure that other threads manipulating the futex can
                // faithfully observe our pre-sleep state, and that we end up
                // seeing the world like they did at the end.
                self.futex
                    .wait_for_change(futex_state, Ordering::Release, Ordering::Acquire);
            });
        } else {
            // No work available for now and no more work expected, nothing
            // to do (this should only be a transient state when we are
            // waiting for a signal that will be sent by other workers
            // processing their own work)
            if !self.work_over.get() {
                self.work_over.set(true);
            } else {
                std::thread::sleep(YIELD_DURATION)
            }
        }
    }

    /// Try to steal work, return truth that we managed to steal and run a task
    fn steal_work(&self, futex_state: &WorkerFutexState) -> bool {
        // Try to steal from the recommended location, if any
        let recommended_location = futex_state.steal_location();
        if let Some(location) = recommended_location {
            let successful = match location {
                StealLocation::Worker(idx) => self.steal_from_worker(idx),
                StealLocation::Injector => crate::unlikely(|| self.steal_from_injector()),
            };
            if successful {
                return true;
            }
        }

        // If there is no recommendation, or if the recommendation is outdated,
        // try to steal from all the possible sources of work
        if let Some(location) = self.steal_from_anyone() {
            // Record any place where we found work so we can try to steal from
            // there again next time, unless a better recommendation has
            // concurrently come up from another thread.
            //
            // Can use Relaxed because we are talking to ourselves and not using
            // the updated futex state, if we observe it again it will be with
            // proper Acquire ordering on the next load().
            let _new_futex_state =
                self.futex
                    .suggest_steal(location, self.idx, Ordering::Relaxed, Ordering::Relaxed);
            return true;
        }

        // No work available anywhere at the moment, just clear our
        // recommended stealing location and enter the waiting state.
        //
        // Can use Relaxed because we are talking to ourselves and not using
        // the updated futex state, if we observe it again it will be with
        // proper Acquire ordering on the next load().
        if recommended_location.is_some() {
            let _new_futex_state = self.futex.clear_outdated_location(
                *futex_state,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
        false
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
        self.steal_with(|| {
            self.shared
                .worker_interfaces()
                .nth(idx)
                .expect("worker not found in shared state")
                .stealer
                .steal()
        })
    }

    /// Try to steal work from the global task injector
    ///
    /// Return truth that a task was successfully stolen and run.
    #[inline(always)]
    fn steal_from_injector(&self) -> bool {
        self.steal_with(|| self.shared.injector().steal())
    }

    /// Try to steal work from all possible sources
    ///
    /// Return from which source work was stolen (if any), using the conventions
    /// of `self.futex`, so that `self.futex` can be updated if appropriate.
    fn steal_from_anyone(&self) -> Option<StealLocation> {
        // Are there other workers we could steal work from?
        //
        // Need Acquire so stealing happens after observing available work.
        if let Some(neighbors_with_work) = self
            .shared
            .find_work_to_steal(&self.work_available.bit, Ordering::Acquire)
        {
            // Try to steal from other workers at increasing distances
            for idx in neighbors_with_work {
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
    fn steal_with(&self, mut attempt: impl FnMut() -> Steal<DynJob>) -> bool {
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

/// Flag telling the world that a worker might have work available to steal
///
/// Set when we the worker pushes new work into its work queue, cleared when the
/// worker fails to pop work from its work queue. May be incorrectly set if
/// other workers have stolen work, but will never be incorrectly unset.
#[derive(Clone, Debug)]
struct WorkAvailabilityBit<'pool> {
    /// Bit of this worker in `SharedState::work_availability`
    bit: BitRef<'pool, true>,

    /// Truth that `bit` is currently set
    ///
    /// Caching this in a private variable ensures that in the happy path, we
    /// don't need to access any shared variable subjected to concurrent access
    /// cache contention, besides the state of our work queue.
    bit_is_set: Cell<bool>,
}
//
impl<'pool> WorkAvailabilityBit<'pool> {
    /// Set up work availability notifications
    fn new(shared: &'pool SharedState, worker_idx: usize) -> Self {
        let bit = shared.worker_availability(worker_idx);
        // Can be a relaxed load since no one else should be modifying this bit
        let bit_is_set = Cell::new(bit.is_set(Ordering::Relaxed));
        Self { bit, bit_is_set }
    }

    /// Truth that our work availability bit is currently set
    fn is_set(&self) -> bool {
        self.bit_is_set.get()
    }

    /// Set our work availability bit with the specific atomic ordering
    fn set(&self, order: Ordering) {
        if !self.bit_is_set.replace(true) {
            let old = self.bit.fetch_set(order);
            debug_assert!(!old);
        }
    }

    /// Clear our work availability bit with the specific atomic ordering
    fn clear(&self, order: Ordering) {
        if self.bit_is_set.replace(false) {
            let old = self.bit.fetch_clear(order);
            debug_assert!(old);
        }
    }
}

/// State used when the worker is waiting for work
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
struct WaitingState {
    /// Timestamp at which we entered the waiting state, assuming we're in it
    waiting_start: Instant,

    /// Number of spin-loop iterations since we last yielded to the OS
    spin_iters_since_yield: usize,

    /// Number of spin-loop iterations between attempts to look for work
    spin_iters_per_check: u8,
}
//
impl WaitingState {
    /// Start waiting
    fn new() -> Self {
        Self {
            waiting_start: Instant::now(),
            spin_iters_since_yield: 0,
            spin_iters_per_check: 1,
        }
    }

    /// Do a spin loop iteration, periodically yielding to the kernel scheduler,
    /// until the time comes to block the thread
    fn wait(&mut self, blocking_wait: impl FnOnce()) {
        if self.spin_iters_since_yield <= SPIN_ITERS_BEFORE_YIELD {
            if self.spin_iters_per_check < MAX_SPIN_ITERS_PER_CHECK {
                self.spin_iters_per_check =
                    (2 * self.spin_iters_per_check).min(MAX_SPIN_ITERS_PER_CHECK);
            }
            for _ in 0..self.spin_iters_per_check {
                std::hint::spin_loop()
            }
        } else if self.waiting_start.elapsed() < OS_WAIT_DELAY {
            // ...periodically yielding to the kernel scheduler...
            std::thread::sleep(YIELD_DURATION);
            self.spin_iters_since_yield = 0;
        } else {
            blocking_wait()
        }
    }
}
