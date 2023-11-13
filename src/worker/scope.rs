//! Worker scope
//!
//! Used by tasks running on the thread pool to interact with the worker thread
//! that's running them, and by extension with the thread pool at large.

use crossbeam::utils::CachePadded;

use super::Worker;
use crate::{
    shared::{
        futex::StealLocation,
        job::{AbortOnUnwind, DynJob, Job, Notify},
    },
    Work,
};
use std::{
    cell::Cell,
    collections::VecDeque,
    debug_assert, debug_assert_eq,
    fmt::{self, Debug},
    ops::Deref,
    panic::AssertUnwindSafe,
    sync::atomic::{self, AtomicU8, Ordering},
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
        let (join, notify) = self
            .0
            .join_tracker
            .allocate_join(&self.0, Ordering::Relaxed, Ordering::Acquire)
            .expect("reached internal join() concurrency limit");
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
            while !join.remote_status(Ordering::Relaxed).task_processed() {
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
        self.0.work_available.set(Ordering::Relaxed);

        // ...and personally notify the closest starving thread about it
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

/// Unique identifier of an ongoing [`Scope::join()`] operation
pub(crate) type JoinID = u16;

/// Local end of a `join()`
///
/// Will automatically deallocate the `join()`'s state once its scope is exited
struct Join<'worker>(JoinState<'worker>);
//
impl<'worker> Deref for Join<'worker> {
    type Target = JoinState<'worker>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
//
impl Drop for Join<'_> {
    fn drop(&mut self) {
        self.0.deallocate_local();
    }
}

/// Remote end of a `join()`
///
/// Can be used to notify the worker that the remote task that was previously
/// spawned by a `join()` is complete.
struct NotifyJoin<'worker> {
    /// Worker that ran this `join()`
    worker: &'worker Worker<'worker>,

    /// Identifier of the `join()`
    id: JoinID,
}
//
impl NotifyJoin<'_> {
    /// Access the join state
    fn join(&self) -> JoinState<'_> {
        self.worker.join_tracker.get(self.id)
    }
}
//
// SAFETY: Every operation which can make the join observable is `Release`
unsafe impl Notify for NotifyJoin<'_> {
    #[inline]
    fn notify(self) {
        // Access the join state
        let join = self.join();

        // First, announce job completion. This must be Release per the
        // NotifyFutex contract, as it makes job completion observable
        join.notify_complete(Ordering::Release);

        // Wake the worker if it was asleep or going ot sleep by updating its
        // "latest join ID" to our join ID. This may cause the worker to miss
        // previous join notifications, but that's okay: since we're using
        // AcqRel ordering, a worker that sees our join_id with Acquire ordering
        // transitively sees all memory updates performed by previous joins.
        self.worker.futex.notify_join(self.id, Ordering::AcqRel);
    }
}
//
impl Drop for NotifyJoin<'_> {
    fn drop(&mut self) {
        // Tell the worker that we're done using this join_id and it can be
        // reclaimed for use by a future join().
        let join = self.join();
        debug_assert_eq!(
            join.remote_status(Ordering::Relaxed),
            RemoteStatus::Notifying,
            "remote task exited without sending a notification, deadlock will ensue"
        );
        join.deallocate_remote(Ordering::Release);
    }
}

/// State of a `join()` statement, as tracked by [`JoinTracker`]
struct JoinState<'worker> {
    /// Identifier of this `join()`
    id: JoinID,

    /// `worker_usage` word where the worker usage flag is located
    worker_usage_word: &'worker Cell<usize>,

    /// Bit of `worker_usage_word` where the worker usage bit is located
    worker_usage_bit_shift: u32,

    /// Remote task status byte
    remote_status: &'worker AtomicU8,
}
//
impl<'worker> JoinState<'worker> {
    /// Unique identifier of this `join()`
    fn id(&self) -> JoinID {
        self.id
    }

    /// Truth that this `join()` state is in use
    fn used_by_worker(&self) -> bool {
        self.worker_usage_word.get() & self.worker_usage_bit() != 0
    }

    /// Current value of the remote status word
    fn remote_status(&self, order: Ordering) -> RemoteStatus {
        RemoteStatus::from_raw(self.remote_status.load(order))
    }

    /// Mask selecting the worker usage bit
    fn worker_usage_bit(&self) -> usize {
        1 << self.worker_usage_bit_shift
    }

    /// Attempt to allocate this `JoinState` for a new `join()`
    #[inline]
    fn try_allocate(
        self,
        worker: &'worker Worker,
        allocate: Ordering,
        find: Ordering,
        check: Ordering,
    ) -> Option<(Join<'worker>, NotifyJoin<'worker>)> {
        // Make sure the join isn't used by the worker or a remote task
        if self.used_by_worker() {
            return None;
        }
        if self.remote_status(check) != RemoteStatus::Finished {
            return None;
        }
        if find != check {
            atomic::fence(find);
        }

        // Allocate the worker side of the join state
        self.worker_usage_word
            .set(self.worker_usage_word.get() | self.worker_usage_bit());
        let local = Join(self);

        // Allocate the remote side of the join state
        let pending = RemoteStatus::Pending.to_raw();
        if cfg!(debug_assertions) {
            assert_eq!(
                local.remote_status.swap(pending, allocate),
                RemoteStatus::Finished.to_raw()
            );
        } else {
            local.remote_status.store(pending, allocate);
        }
        let remote = NotifyJoin {
            worker,
            id: local.id(),
        };
        Some((local, remote))
    }

    /// Announce that the remote end of a join is complete and the futex
    /// notification is in the process of being sent
    fn notify_complete(&self, order: Ordering) {
        let notifying = RemoteStatus::Notifying.to_raw();
        if cfg!(debug_assertions) {
            assert_eq!(
                self.remote_status.swap(notifying, order),
                RemoteStatus::Pending.to_raw()
            );
        } else {
            self.remote_status.store(notifying, order);
        }
    }

    /// Deallocate a previously allocated `JoinState` from the worker side
    ///
    /// Importantly, the `JoinState` is _not_ deallocated from the remote task's
    /// perspective. It is the remote task that will eventually perform the
    /// deallocation by setting the remote status to `RemoteStatus::Finished`.
    fn deallocate_local(&self) {
        debug_assert!(self.used_by_worker());
        self.worker_usage_word
            .set(self.worker_usage_word.get() & !self.worker_usage_bit());
    }

    /// Announce that the remote end of a join has been dropped and the join can
    /// be recycled as long as the worker is also done with it
    fn deallocate_remote(&self, order: Ordering) {
        let finished = RemoteStatus::Finished.to_raw();
        if cfg!(debug_assertions) {
            assert_eq!(
                self.remote_status.swap(finished, order),
                RemoteStatus::Pending.to_raw()
            );
        } else {
            self.remote_status.store(finished, order);
        }
    }
}

/// Centralized state of all ongoing `join()` statements in a worker thread
///
/// This is needed because to work around futex API limitations, we want to give
/// each ongoing join a small identifier that fits in 16 bits, and the simplest
/// way to do that in a thread-safe manner is to preallocate an array of 65k
/// join states somewhere with Join IDs being indexes inside this array. We can
/// afford to use this preallocation strategy because the state that must be
/// preallocated is tiny (1 bytes and 1 bit per join).
pub(super) struct JoinTracker {
    /// Join state which we should try to allocate next
    next_join_id: Cell<JoinID>,

    /// Flags tracking which join states are currently used by this worker
    worker_usage: Box<[Cell<usize>]>,

    /// Status bytes tracking the progress of a join's remote task
    ///
    /// To avoid false sharing, this is not stored in the intuitive order where
    /// join #N maps into the N-th status byte. Instead, the first join maps
    /// into the first byte of the first cache line, the second join maps into
    /// the first byte of the second cache line, and so on until we've covered
    /// all cache lines. Only then do we start allocating the second byte of the
    /// first cache line, the second byte of the second cache line, and so on.
    remote_status: Box<[AtomicU8]>,
}
//
impl JoinTracker {
    /// Set up join statement tracking
    pub fn new() -> Self {
        let num_joins = usize::from(u16::MAX) + 1;
        let num_usage_words = num_joins.div_ceil(usize::BITS as usize);

        let next_join_id = Cell::new(0);
        let worker_usage = std::iter::repeat(Cell::new(0))
            .take(num_usage_words)
            .collect();
        let remote_status =
            std::iter::repeat_with(|| AtomicU8::new(RemoteStatus::Finished.to_raw()))
                .take(num_joins)
                .collect();

        Self {
            next_join_id,
            worker_usage,
            remote_status,
        }
    }

    /// Allocate state for a new `join()` statement
    #[inline]
    fn allocate_join<'worker>(
        &'worker self,
        worker: &'worker Worker,
        update: Ordering,
        find: Ordering,
    ) -> Option<(Join<'_>, NotifyJoin<'_>)> {
        let (local, remote) = self
            .iter_from(self.next_join_id.get())
            .find_map(|state| state.try_allocate(worker, update, find, Ordering::Relaxed))?;
        let next_id = JoinID::try_from((local.id as usize + 1) % self.len()).unwrap();
        self.next_join_id.set(next_id);
        Some((local, remote))
    }

    /// Access a previously allocated join
    fn get(&self, id: JoinID) -> JoinState<'_> {
        let (word_idx, bit_shift, remote_idx) = self.decode_join_id(id);
        JoinState {
            id,
            worker_usage_word: &self.worker_usage[word_idx],
            worker_usage_bit_shift: bit_shift,
            remote_status: &self.remote_status[remote_idx],
        }
    }

    /// Iterate over all join statement states
    fn iter(&self) -> impl Iterator<Item = JoinState<'_>> + '_ {
        self.iter_from(0)
    }

    /// Iterate over all join statement states, starting at a given index
    fn iter_from(&self, start: JoinID) -> impl Iterator<Item = JoinState<'_>> + '_ {
        let (start_word_idx, mut curr_bit_shift, mut curr_remote_idx) = self.decode_join_id(start);
        let enumerated_words = self.worker_usage.iter().enumerate();
        let mut enumerated_words = enumerated_words
            .clone()
            .skip(start_word_idx)
            .chain(enumerated_words.take(start_word_idx));
        let mut curr_idx_word = enumerated_words.next();
        std::iter::from_fn(move || {
            // Emit a result associated with the current state
            let (curr_word_idx, worker_usage_word) = curr_idx_word?;
            let id =
                JoinID::try_from(curr_word_idx * (usize::BITS as usize) + curr_bit_shift as usize)
                    .unwrap();
            let result = JoinState {
                id,
                worker_usage_word,
                worker_usage_bit_shift: curr_bit_shift,
                remote_status: &self.remote_status[curr_remote_idx],
            };

            // Advance to the next worker usage flag
            curr_bit_shift += 1;
            if curr_bit_shift == usize::BITS {
                curr_bit_shift = 0;
                curr_idx_word = enumerated_words.next();
            }

            // Advance to the next remote status byte in a manner that avoids
            // false sharing between consecutive JoinStates
            curr_remote_idx += BYTES_PER_CACHE_LINE;
            if curr_remote_idx >= self.remote_status.len() {
                curr_remote_idx -= self.remote_status.len() - 1;
            }
            Some(result)
        })
    }

    /// Decode a JoinID into a (worker_usage word idx, worker_usage bit idx,
    /// remote_status byte idx) tuple
    fn decode_join_id(&self, id: JoinID) -> (usize, u32, usize) {
        let linear_idx = usize::from(id);

        let worker_word_idx = linear_idx / (usize::BITS as usize);
        let worker_bit_shift = (linear_idx % (usize::BITS as usize)) as u32;

        let num_cache_lines = self.len().div_ceil(BYTES_PER_CACHE_LINE);
        let byte_idx = linear_idx / num_cache_lines;
        let cache_line_idx = linear_idx % num_cache_lines;
        let remote_idx = byte_idx + cache_line_idx * BYTES_PER_CACHE_LINE;

        (worker_word_idx, worker_bit_shift, remote_idx)
    }

    /// Number of JoinStates in this collection
    fn len(&self) -> usize {
        let expected_len = usize::from(JoinID::MAX);
        debug_assert_eq!(self.remote_status.len(), expected_len);
        expected_len
    }
}
//
impl Debug for JoinTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display worker_usage and remote_statuses in a readable fashion
        let mut worker_usage = VecDeque::new();
        let mut remote_status = VecDeque::new();
        for (idx, join_state) in self.iter().enumerate() {
            // Add separators every 8 chars for clarity
            if idx % 8 == 0 && idx != 0 {
                worker_usage.push_front(b'_');
                remote_status.push_front(b'_');
            }

            // Display worker usage
            worker_usage.push_front(if join_state.used_by_worker() {
                b'1'
            } else {
                b'0'
            });

            // Display remote status
            remote_status.push_front(match join_state.remote_status(Ordering::Relaxed) {
                RemoteStatus::Pending => b'P',
                RemoteStatus::Notifying => b'N',
                RemoteStatus::Finished => b'F',
            });
        }

        // Visually align worker_usage display with remote_status display
        remote_status.push_front(b' ');

        // Convert display buffers to strings
        let to_string = |buf: VecDeque<u8>| String::from_utf8(Vec::from(buf)).unwrap();
        f.debug_struct("JoinTracker")
            .field("next_join_id", &self.next_join_id)
            .field("worker_usage", &to_string(worker_usage))
            .field("remote_status", &to_string(remote_status))
            .finish()
    }
}

/// Remote task status
#[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
enum RemoteStatus {
    /// Remote task is queued or being processed
    Pending = 0,

    /// Remote task has been processed, worker futex is being notified of it
    Notifying,

    /// Remote task has been processed and notifications have been set, this
    /// status byte can be reused once the worker exits the associated `join()`
    #[default]
    Finished,
}
//
impl RemoteStatus {
    /// Truth that this status indicates the remote task is completed
    fn task_processed(self) -> bool {
        match self {
            Self::Pending => false,
            Self::Notifying | Self::Finished => true,
        }
    }

    /// Convert to integer encoding for atomic operations
    fn to_raw(self) -> u8 {
        match self {
            Self::Pending => 0,
            Self::Notifying => 1,
            Self::Finished => 2,
        }
    }

    /// Convert back from integer encoding, panics on unknown representation
    fn from_raw(x: u8) -> Self {
        match x {
            0 => Self::Pending,
            1 => Self::Notifying,
            2 => Self::Finished,
            _ => unreachable!(),
        }
    }
}

/// Number of bytes in a cache line, according to crossbeam
const BYTES_PER_CACHE_LINE: usize = std::mem::align_of::<CachePadded<u8>>();
