//! Worker scope
//!
//! Used by tasks running on the thread pool to interact with the worker thread
//! that's running them, and by extension with the thread pool at large.

use crossbeam::utils::CachePadded;

use super::Worker;
use crate::{
    shared::{
        futex::{StealLocation, WorkerFutex},
        job::{AbortOnUnwind, DynJob, Job, Notify},
    },
    Work,
};
use std::{
    debug_assert_eq,
    panic::AssertUnwindSafe,
    sync::atomic::{self, AtomicBool, AtomicU8, Ordering},
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
        let notify = NotifyJoin::new(&self.0);
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

    /// Allocate a new join ID
    fn new_join_id(&self) -> JoinID {
        // Query the ID used for the last join
        let last_join_id = self.0.last_join_id.get();

        // Join IDs are basically the indices of bytes in the
        // `Worker::join_statuses` table
        let join_statuses = &self.0.join_statuses;
        let mut next_join_idx = usize::from(last_join_id);
        let next_join_idx = || {
            // Pick a status byte on the same byte of the next cache line to
            // avoid false sharing between join()s started in short succession
            const BYTES_PER_CACHE_LINE: usize = std::mem::align_of::<CachePadded<u8>>();
            next_join_idx += BYTES_PER_CACHE_LINE;
            if next_join_idx < join_statuses.len() {
                return next_join_idx;
            }

            // Once we're done iterating over the same byte of all cache lines,
            // move to the next byte of all cache lines and iterate again.
            next_join_idx -= join_statuses.len() - 1;
            if next_join_idx < BYTES_PER_CACHE_LINE.min(join_statuses.len()) {
                return next_join_idx;
            }

            // Once we've probed all bytes of all cache lines, go back to the
            // first status byte of `join_statuses`.
            next_join_idx = 0;
            next_join_idx
        };

        // Most of the time, the first join ID we end up using should be unused.
        // Try to allocate it for this join with a single eager RMW.
        let first_join_idx = next_join_idx();
        let first_status =
            join_statuses[first_join_idx].swap(JOIN_STATUS_RUNNING, Ordering::Acquire);
        let join_idx = if first_status = JOIN_STATUS_FINISHED {
            first_join_idx
        } else {
            // If next join ID is busy, try to all other IDs more cautiously
            #[cold]
            fn probe_all_join_statuses(
                first_join_idx: usize,
                join_statuses: &[AtomicJoinStatus],
                mut next_join_idx: impl FnMut() -> usize,
            ) -> usize {
                loop {
                    // Iterate until we've went through all join indices
                    let curr_join_idx = next_join_idx();
                    if curr_join_idx == first_join_idx {
                        panic!("exceeded implementation join() concurrency limit");
                    }

                    // Try to use the current join status byte
                    let join_status = &join_statuses[curr_join_idx];
                    if join_status.load(Ordering::Relaxed) == JOIN_STATUS_FINISHED {
                        let old_status = join_status.swap(JOIN_STATUS_RUNNING, Ordering::Acquire);
                        debug_assert_eq!(old_status, JOIN_STATUS_FINISHED);
                        return curr_join_idx;
                    }
                }
            }
            probe_all_join_statuses(first_join_idx, join_statuses, &mut next_join_idx)
        };

        // Report freshly allocated join ID
        let first_join_id = u16::try_from(first_join_idx).unwrap();
        self.0.last_join_id.set(first_join_id);
        first_join_id
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

/// Unique identifier of an ongoing [`Scope::join()`] operation
pub(crate) type JoinID = u16;

/// Join status byte
pub(super) type JoinStatus = u8;

/// Atomic version of [`JoinStatus`]
pub(super) type AtomicJoinStatus = AtomicU8;

/// [`JoinStatus`] value used for joins where the remote task is running
pub(super) const JOIN_STATUS_RUNNING: JoinStatus = 0;

/// [`JoinStatus`] value used for joins where the remote task has completed but
/// the join completion notification hasn't been sent to the worker futex and
/// thus the associated JoinID should not be reused yet.
pub(super) const JOIN_STATUS_NOTIFYING: JoinStatus = 1;

/// [`JoinStatus`] value used for joins where the remote task is done using the
/// JoinID and it can be reallocated to another join.
pub(super) const JOIN_STATUS_FINISHED: JoinStatus = 2;

/// Mechanism to notify worker threads of join() completion
#[derive(Copy, Clone, Debug)]
struct NotifyJoin<'worker> {
    /// Access to the worker state
    worker: &'worker Worker<'worker>,

    /// Unique identifier of this join
    join_id: JoinID,
}
//
// SAFETY: join_status is set with Release ordering and is the main signal that
//         the worker uses to detect job completion.
unsafe impl<'worker> Notify for NotifyJoin<'worker> {
    /// Set up a join completion notification channel
    fn new(worker: &'worker Worker<'worker>) {
        let join_id = Self::allocate_join_id(worker);
        Self { worker, join_id }
    }

    /// Allocate a JoinID for this new join
    fn allocate_join_id(worker: &Worker<'_>) -> JoinID {
        // Query the ID used for the last join
        let last_join_id = worker.last_join_id.get();

        // Join IDs are basically the indices of bytes in the
        // `Worker::join_statuses` table
        let join_statuses = &worker.join_statuses;
        let mut next_join_idx = usize::from(last_join_id);
        let next_join_idx = || {
            // Pick a status byte on the same byte of the next cache line to
            // avoid false sharing between join()s started in short succession
            const BYTES_PER_CACHE_LINE: usize = std::mem::align_of::<CachePadded<u8>>();
            next_join_idx += BYTES_PER_CACHE_LINE;
            if next_join_idx < join_statuses.len() {
                return next_join_idx;
            }

            // Once we're done iterating over the same byte of all cache lines,
            // move to the next byte of all cache lines and iterate again.
            next_join_idx -= join_statuses.len() - 1;
            if next_join_idx < BYTES_PER_CACHE_LINE.min(join_statuses.len()) {
                return next_join_idx;
            }

            // Once we've probed all bytes of all cache lines, go back to the
            // first status byte of `join_statuses`.
            next_join_idx = 0;
            next_join_idx
        };

        // Most of the time, the first join ID we end up using should be unused.
        // Try to allocate it for this join with a single eager RMW.
        let first_join_idx = next_join_idx();
        let first_status =
            join_statuses[first_join_idx].swap(JOIN_STATUS_RUNNING, Ordering::Acquire);
        let join_idx = if first_status = JOIN_STATUS_FINISHED {
            first_join_idx
        } else {
            // If next join ID is busy, try to all other IDs more cautiously
            #[cold]
            fn probe_all_join_statuses(
                first_join_idx: usize,
                join_statuses: &[AtomicJoinStatus],
                mut next_join_idx: impl FnMut() -> usize,
            ) -> usize {
                loop {
                    // Iterate until we've went through all join indices
                    let curr_join_idx = next_join_idx();
                    if curr_join_idx == first_join_idx {
                        panic!("exceeded implementation join() concurrency limit");
                    }

                    // Try to use the current join status byte
                    let join_status = &join_statuses[curr_join_idx];
                    if join_status.load(Ordering::Relaxed) == JOIN_STATUS_FINISHED {
                        let old_status = join_status.swap(JOIN_STATUS_RUNNING, Ordering::Acquire);
                        debug_assert_eq!(old_status, JOIN_STATUS_FINISHED);
                        return curr_join_idx;
                    }
                }
            }
            probe_all_join_statuses(first_join_idx, join_statuses, &mut next_join_idx)
        };

        // Report freshly allocated join ID
        let first_join_id = u16::try_from(first_join_idx).unwrap();
        worker.last_join_id.set(first_join_id);
        first_join_id
    }

    fn notify(self) {
        // Access the status byte for this join
        let join_status = &self.worker.join_statuses[usize::from(self.join_id)];

        // First, announce job completion. This must be Release, as per the
        // NotifyFutex contract, so it's not reordered before job completion.
        if cfg!(debug_assertions) {
            assert_eq!(
                join_status.swap(JOIN_STATUS_NOTIFYING, Ordering::Release),
                JOIN_STATUS_RUNNING,
            );
        } else {
            join_status.store(JOIN_STATUS_NOTIFYING, Ordering::Release);
        }

        // Wake the worker if it was asleep or going ot sleep by updating its
        // "latest join ID" to our join ID. This may cause the worker to miss
        // previous join notifications, but that's okay: since we're using
        // AcqRel ordering, a worker that sees our join_id with Acquire ordering
        // transitively sees all memory updates performed by previous joins.
        self.worker
            .futex
            .notify_join(self.join_id, Ordering::AcqRel);

        // Tell the worker that we're done using this join_id and it can be
        // reclaimed for use by a future join().
        if cfg!(debug_assertions) {
            assert_eq!(
                join_status.swap(JOIN_STATUS_FINISHED, Ordering::Release),
                JOIN_STATUS_NOTIFYING,
            );
        } else {
            join_status.store(JOIN_STATUS_FINISHED, Ordering::Release);
        }
    }
}
