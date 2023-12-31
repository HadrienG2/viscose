//! Worker scope
//!
//! Used by tasks running on the thread pool to interact with the worker thread
//! that's running them, and by extension with the thread pool at large.

use super::Worker;
use crate::{
    shared::{
        futex::WorkerFutex,
        job::{AbortOnUnwind, Job, Notify, Schedule, Task},
    },
    Work,
};
use std::{
    fmt::Debug,
    panic::AssertUnwindSafe,
    sync::atomic::{self, AtomicBool, Ordering},
};

/// Scope for executing parallel work
///
/// This is a token which attests that work is executing within the context of a
/// worker thread inside of the thread pool, and can be used to schedule work on
/// said thread pool.
#[derive(Debug)]
pub struct Scope<'scope> {
    /// Worker that is executing this work
    worker: AssertUnwindSafe<&'scope Worker<'scope>>,

    /// Scheduling constraints of this job
    schedule: Schedule,
}
//
impl<'scope> Scope<'scope> {
    /// Provide an opportunity for fork-join parallelism
    ///
    /// Run the `local` work on this thread, possibly making the `remote` work
    /// available for other threads to steal and execute. If no other thread
    /// takes over the `remote` work, do it ourselves. Wait for both tasks to be
    /// complete before moving on.
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
        // Handle sequential join fast path
        if !self.schedule.parallelize_join() {
            return (
                local(),
                remote(&Self {
                    worker: AssertUnwindSafe(self.worker.0),
                    schedule: self.schedule.child_schedule(),
                }),
            );
        }

        // Set up remote job and its completion notification mechanism
        //
        // The join start notification, if any, must be Acquire so it's not
        // reordered after the beginning (or the end!) of the join. Although it
        // should not be reordered before the completion of the previous join to
        // avoid spurious join counter overflow, there is no need to add a
        // Release barrier for this because the Acquire barrier at the end of
        // the previous join will already enforce the desired ordering.
        let futex = &self.worker.futex;
        #[cfg(feature = "detect-excessive-joins")]
        self.worker.futex.start_join(Ordering::Acquire);
        let remote_finished = AtomicBool::new(false);
        let notify = NotifyJoin {
            futex,
            remote_finished: &remote_finished,
        };
        let mut remote_job = Job::new(notify, remote);

        // No unwinding panics allowed until the remote task has completed
        let local_result = {
            // Spawn remote task
            // SAFETY: We'll wait for the job to complete before letting it go
            //         out of scope or otherwise touching it, and panics are
            //         translated to aborts until it's done executing.
            let abort_on_unwind = AbortOnUnwind;
            unsafe { self.spawn_unchecked(remote_job.make_task(self.schedule.child_schedule())) };

            // Run local task
            let local_result = std::panic::catch_unwind(AssertUnwindSafe(local));

            // Execute thread pool work while waiting for remote task,
            // synchronize with the remote task once it completes
            while !remote_finished.load(Ordering::Relaxed) {
                self.worker.step();
            }
            atomic::fence(Ordering::Acquire);
            std::mem::forget(abort_on_unwind);
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
        self.worker.idx
    }

    /// Set up a scope, associated with a particular worker thread and with
    /// certain scheduling properties
    pub(super) fn new(worker: &'scope Worker<'scope>, schedule: Schedule) -> Self {
        Self {
            worker: AssertUnwindSafe(worker),
            schedule,
        }
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
    unsafe fn spawn_unchecked(&self, task: Task) {
        // Schedule the work to be executed
        self.worker.work_queue.push(task);

        // ...and once this push is visible...
        //
        // It doesn't matter which of the stores below completes first, as long
        // as they execute after the work becomes visibly available, so they can
        // be Relaxed stores after a Release barrier.
        atomic::fence(Ordering::Release);

        // ...tell the entire world that we now have work available to steal if
        // they didn't know about it before...
        self.worker.work_available.set(Ordering::Relaxed);

        // ...and personally notify the closest starving thread about it
        // This doesn't need to be ordered after the setting of work_available
        // because workers following a direct work-stealing recommendation do
        // not check the work_availability bits.
        self.worker.shared.suggest_stealing_from_worker(
            &self.worker.work_available.bit,
            self.worker.distances,
            Ordering::Relaxed,
        );
    }
}
//
// TODO: If I add a safe spawn(), bind its callable on F: 'scope, add tracking
//       of spawned tasks and make the function that created the scope ensure
//       that they are all finished before returning.

/// Notification mechanism for `join()`
///
/// Can be used to notify the worker that the remote task that was previously
/// spawned by a `join()` is complete.
struct NotifyJoin<'worker> {
    /// Atomic variable signaling join completion
    remote_finished: &'worker AtomicBool,

    /// Futex of the worker that started this join
    futex: &'worker WorkerFutex,
}
//
// SAFETY: The join is made observable with Release ordering
unsafe impl Notify for NotifyJoin<'_> {
    fn notify(self) {
        // Mark the remote task as finished
        //
        // This makes the join observable, so it must be Release
        self.remote_finished.store(true, Ordering::Release);

        // Ping the worker about the finished join, waking it up if necessary
        //
        // This may replace a previous join notifications that wasn't observed
        // by the worker, but that's okay: due to atomic variable coherence, a
        // worker that sees our futex update with Acquire ordering still
        // transitively sees the machine effects associated with all futex
        // updates performed by previous joins.
        //
        // This must be Release because otherwise a worker could be awakened
        // without observing remote_finished to be true, which would lead to an
        // incorrectly failed join() completion check and thus a lost wakeup.
        self.futex.notify_join(Ordering::Release);
    }
}
