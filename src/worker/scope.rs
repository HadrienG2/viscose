//! Worker scope
//!
//! Used by tasks running on the thread pool to interact with the worker thread
//! that's running them, and by extension with the thread pool at large.

use super::Worker;
use crate::{
    shared::{
        futex::{StealLocation, WorkerFutex},
        job::{AbortOnUnwind, DynJob, Job, Notify},
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
/// worker thread insid of the thread pool, and can be used to schedule work on
/// said thread pool.
#[derive(Debug)]
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
        //
        // The join start notification, if any, must be Acquire so it's not
        // reordered after the beginning (or the end!) of the join. Although it
        // should not be reordered before the completion of the previous join to
        // avoid spurious join counter overflow, there is no need to add a
        // Release barrier for this because the Acquire barrier at the end of
        // the previous join will already enforce the desired ordering.
        let futex = &self.0.futex;
        #[cfg(feature = "detect-excessive-joins")]
        self.0.futex.start_join(Ordering::Acquire);
        let remote_finished = AtomicBool::new(false);
        let notify = NotifyJoin {
            futex,
            remote_finished: &remote_finished,
        };
        let mut remote_job = Job::new(notify, remote);

        // No unwinding panics allowed until the remote task has completed
        let local_result = {
            // Spawn remote task
            // SAFETY: We wait for the job to complete before letting it go out
            //         of scope or otherwise touching it in any way, and panics
            //         are translated to aborts until it's done executing.
            let abort_on_unwind = AbortOnUnwind;
            unsafe { self.spawn_unchecked(remote_job.as_dyn()) };

            // Run local task
            let local_result = std::panic::catch_unwind(AssertUnwindSafe(local));

            // Execute thread pool work while waiting for remote task,
            // synchronize with the remote task once it completes
            while !remote_finished.load(Ordering::Relaxed) {
                self.0.step();
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
        self.0.idx
    }

    /// Set up a scope associated with a particular worker thread
    pub(super) fn new(worker: &'scope Worker<'scope>) -> Self {
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
        //
        // It doesn't matter which of the stores below completes first, as long
        // as they execute after the work becomes visibly available, so they can
        // be Relaxed stores after a Release barrier.
        atomic::fence(Ordering::Release);

        // ...tell the entire world that we now have work available to steal if
        // they didn't know about it before...
        self.0.work_available.set(Ordering::Relaxed);

        // ...and personally notify the closest starving thread about it
        // This doesn't need to be ordered after the setting of work_available
        // because workers following a recommendation don't read work_available.
        self.0.shared.recommend_steal::<false, true>(
            &self.0.work_available.bit,
            StealLocation::Worker(self.0.idx),
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
        // Announce job completion
        //
        // This makes the join observable, so it must be Release
        self.remote_finished.store(true, Ordering::Release);

        // Announce join completion
        //
        // This may replace a previous join notifications that wasn't observed
        // by the worker, but that's okay: due to atomic variable coherence, a
        // worker that sees our futex update with Acquire ordering still
        // transitively sees the machine effects associated with all futex
        // updates performed by previous joins.
        //
        // This must be Release because otherwise one could get the notification
        // before remote_finished is true, which would lead to an incorrectly
        // failed join() check and thus lost wakeup.
        self.futex.notify_join(Ordering::Release);
    }
}
