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
#[derive(Copy, Clone, Debug)]
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
