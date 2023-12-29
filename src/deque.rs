//! Per-worker work queue
//!
//! While it has served this project well for a long time, `crossbeam_deque` has
//! two important limitations that become increasingly problematic over time:
//!
//! - Its design and implementation does not seem to be optimized for the right
//!   workloads. For example, it is not very important to have unbounded work
//!   queues in a binary fork-join world where you can compress 2^64 work items
//!   into 64 splittable tasks, yet the performance price we pay for this
//!   growability (an `MFENCE` on the hot code path just in case reallocations
//!   might occur someday) can be quite expensive on join-heavy workloads.
//! - Work can never be given to a worker, only stolen from it. This makes
//!   perfect sense if you live in the convenient world of symmetric
//!   multiprocessing, but as soon as hardware locality concerns come into play,
//!   it is good to have a way to explicitly spread the workload of newly
//!   spawned jobs to precise locations of the hardware topology, and this in
//!   turn is best done by having full doubled-ended work queues that allow for
//!   remote pushes.
//!
//! Of course, nothing is free in the world of concurrent programming, and
//! solving these two problems will have unavoidable consequences elsewhere...
//!
//! - By doing away with work queue growability and the `MFENCE` that comes with
//!   it, we now need to decide what happens when a work queue fills up. Since
//!   this can only happen with pathological workloads that could use a little
//!   performance backpressure on task spawning to avoid running out of RAM,
//!   spilling to the slow global injector seems like a fine response here.
//! - A deque where N threads can push and pop work on one side cannot have both
//!   a simple and efficient ring buffer data layout and lock-freedom for all
//!   transactions. That's because if multiple threads can be concurrently
//!   pushing work on one side of the ring, and another thread can concurrently
//!   come around and pop a task from that same side of the ring, we will end up
//!   with "holes" in the ring buffer... We choose to handle this by sacrificing
//!   lock-freedom on the "outside world" side of the queue, which should not be
//!   under high pressure when executing non-pathological workloads.

use crossbeam::utils::CachePadded;
#[cfg(test)]
use proptest::prelude::*;
use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    num::NonZeroU32,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
};

// === High-level deque interface ===

/// Maximal number of elements a work queue can hold
pub const MAX_CAPACITY: usize = 2usize.pow(Range::INDEX_BITS) - 2;

/// Worker-side interface to the work queue
#[derive(Debug)]
pub struct Worker<T>(Arc<SharedDeque<T>>);
//
impl<T> Worker<T> {
    /// Set up the work queue
    ///
    /// The capacity will be rounded up a little below a power of two in order
    /// to allow for more efficient transactions later on. It cannot be larger
    /// than [`MAX_CAPACITY`].
    pub fn new(capacity: usize) -> Self {
        Self(SharedDeque::new(capacity))
    }

    /// Set up a remote interface to the work queue
    ///
    /// This interface enables other threads to interact with the work queue, in
    /// order to give work to the worker or steal work from it.
    pub fn remote(&self) -> Remote<T> {
        Remote(self.0.clone())
    }

    /// Push work on the worker side of the deque
    ///
    /// If it is not stolen by other threads, this work will be executed by the
    /// worker in a LIFO fashion, i.e. the most recently pushed task is executed
    /// first. This optimizes cache locality and load balancing granularity at
    /// the expense of fairness.
    pub fn push(&mut self, work: T) -> Result<(), Full<T>> {
        // SAFETY: Unique access to the non-clonable Worker provides momentary
        //         unique access to the local side of the ring buffer.
        unsafe { self.0.deque.push(work) }
    }

    /// Get the next task from the worker side of the deque, if any
    ///
    /// This will execute tasks from the worker in a LIFO fashion and tasks from
    /// the outside world in a FIFO fashion. When both kinds of tasks are
    /// present, their interleaving is unspecified and will depend on the order
    /// in which tasks were submitted on both ends.
    pub fn pop(&mut self) -> Option<T> {
        // SAFETY: Unique access to the non-clonable Worker provides momentary
        //         unique access to the local side of the ring buffer.
        unsafe { self.0.deque.pop() }
    }
}
//
#[cfg(test)]
impl<T: Arbitrary> Arbitrary for Worker<T>
where
    SharedDeque<T>: Arbitrary,
{
    type Parameters = <SharedDeque<T> as Arbitrary>::Parameters;
    type Strategy =
        prop::strategy::Map<<SharedDeque<T> as Arbitrary>::Strategy, fn(SharedDeque<T>) -> Self>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        <SharedDeque<T> as Arbitrary>::arbitrary_with(args)
            .prop_map(|shared| Self(Arc::new(shared)))
    }
}
//
// SAFETY: There is no risk associated with sending a Worker to another thread
unsafe impl<T> Send for Worker<T> {}

/// Error returned when attempting to push work to a full work queue
///
/// The submitted element that could not be pushed is provided back for reuse.
#[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct Full<T>(pub T);

/// External thread interface to the work queue
#[derive(Clone, Debug)]
pub struct Remote<T>(Arc<SharedDeque<T>>);
//
impl<T> Remote<T> {
    /// Set up an external thread interface
    pub fn new(worker: &Worker<T>) -> Self {
        Self(worker.0.clone())
    }

    /// Add work on the remote side of the deque
    ///
    /// If it is not stolen by other threads, this work will be executed by the
    /// worker in a FIFO fashion, i.e. the most recently pushed task is executed
    /// last. This effectively priorizes work spawned by the worker against work
    /// spawned by other threads, which is good for cache locality.
    pub fn give(&self, work: T) -> Result<(), GiveError<T>> {
        let mut lock = match self.0.try_lock_remote() {
            Ok(lock) => lock,
            Err(Locked) => {
                return Err(GiveError {
                    data: work,
                    cause: GiveErrorCause::Locked,
                })
            }
        };
        Ok(lock.give(work)?)
    }

    /// Take work from the remote side of the deque
    ///
    /// This steals the work that the worker is least in a hurry to process at
    /// this point in time, i.e. that should be coldest in worker thread caches.
    pub fn steal(&self) -> Result<T, StealError> {
        let mut lock = self.0.try_lock_remote()?;
        lock.steal().ok_or(StealError::Empty)
    }
}
//
// SAFETY: There is no risk associated with sending a Remote to another thread
unsafe impl<T> Send for Remote<T> {}
//
// SAFETY: There is no risk associated with sharing a Remote between threads
//         because all shared state accesses are appropriately synchronized
unsafe impl<T> Sync for Remote<T> {}

/// Error returned when attempting to give work to a work queue
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct GiveError<T> {
    /// Data that one was attempting to give away
    data: T,

    /// Reason why the giveaway failed
    cause: GiveErrorCause,
}
//
impl<T> From<Full<T>> for GiveError<T> {
    fn from(Full(data): Full<T>) -> Self {
        Self {
            data,
            cause: GiveErrorCause::Full,
        }
    }
}

/// Errors that can occur when giving work to a worker
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum GiveErrorCause {
    /// The queue was full and could not accept any other element
    Full,

    /// The remote end of the queue was locked
    ///
    /// Locks are only held for the duration of an element memcpy, so if the
    /// queue element type is small, spinning on the lock might be fine.
    Locked,
}
//
impl From<Locked> for GiveErrorCause {
    fn from(Locked: Locked) -> Self {
        Self::Locked
    }
}

/// Errors that can occur when stealing work from a worker
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum StealError {
    /// There was no element to steal in the work queue
    Empty,

    /// The remote end of the queue was locked
    ///
    /// Locks are only held for the duration of an element memcpy, so if the
    /// queue element type is small, spinning on the lock might be fine.
    Locked,
}
//
impl From<Locked> for StealError {
    fn from(Locked: Locked) -> Self {
        Self::Locked
    }
}

// === Shared state backing the deque ===

/// [`WorkDeque`] that is shared between a worker and other "remote" threads
#[derive(Debug)]
#[doc(hidden)]
pub struct SharedDeque<T> {
    /// Double-ended work queue
    deque: CachePadded<WorkDeque<T>>,

    /// Lock on the remote end of the work queue
    ///
    /// Set to false when unlocked, true when locked.
    remote_lock: CachePadded<AtomicBool>,
}
//
impl<T> SharedDeque<T> {
    /// Set up the shared state
    ///
    /// See [`Worker::new()`] docs to know more about acceptable capacities.
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            deque: CachePadded::new(WorkDeque::new(capacity)),
            remote_lock: CachePadded::new(AtomicBool::new(false)),
        })
    }

    /// Attempt to acquire the lock on the remote end of the deque
    pub fn try_lock_remote(&self) -> Result<RemoteLock<T>, Locked> {
        RemoteLock::new(self)
    }
}

/// Lock on the remote end of a [`SharedDeque`]
///
/// Must be acquired using [`SharedDeque::try_lock_remote()`] before doing
/// anything on the remote side of the ring buffer (modify remote_idx or use the
/// private storage slot on that side), unless it can be guaranteed that no
/// other thread has access to the ring buffer.
///
/// Will automatically release the lock on `Drop`.
#[derive(Debug)]
struct RemoteLock<'shared, T>(&'shared SharedDeque<T>);
//
impl<'shared, T> RemoteLock<'shared, T> {
    /// Acquire unique access to the remote side of the deque
    pub fn new(shared: &'shared SharedDeque<T>) -> Result<Self, Locked> {
        // We assume that there is insignificant pressure on the remote side of
        // the deque, and thus testing the lock before attempting to acquire it
        // is not worthwhile.
        if shared.remote_lock.swap(true, Ordering::Acquire) {
            return Err(Locked);
        }
        Ok(Self(shared))
    }

    /// Add work on the remote side of the deque
    ///
    /// See [`Remote::give()`] for more information.
    pub fn give(&mut self, work: T) -> Result<(), Full<T>> {
        // SAFETY: Unique access to the non-clonable RemoteLock, which can only
        //         be acquired by one thread at a time, provides momentary
        //         unique access to the remote side of the ring buffer.
        unsafe { self.0.deque.give(work) }
    }

    /// Take work from the remote side of the deque
    ///
    /// See [`Remote::steal()`] for more information.
    pub fn steal(&mut self) -> Option<T> {
        // SAFETY: Unique access to the non-clonable RemoteLock, which can only
        //         be acquired by one thread at a time, provides momentary
        //         unique access to the remote side of the ring buffer.
        unsafe { self.0.deque.steal() }
    }
}
//
#[cfg(test)]
impl<T: Arbitrary> Arbitrary for SharedDeque<T>
where
    WorkDeque<T>: Arbitrary,
{
    type Parameters = <WorkDeque<T> as Arbitrary>::Parameters;
    type Strategy = prop::strategy::Map<
        (
            <WorkDeque<T> as Arbitrary>::Strategy,
            <bool as Arbitrary>::Strategy,
        ),
        fn((WorkDeque<T>, bool)) -> Self,
    >;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        (
            <WorkDeque<T> as Arbitrary>::arbitrary_with(args),
            any::<bool>(),
        )
            .prop_map(|(deque, locked)| Self {
                deque: CachePadded::new(deque),
                remote_lock: CachePadded::new(AtomicBool::new(locked)),
            })
    }
}
//
impl<T> Drop for RemoteLock<'_, T> {
    fn drop(&mut self) {
        if cfg!(debug_assertions) {
            assert!(
                self.0.remote_lock.swap(false, Ordering::Release),
                "lock was unlocked incorrectly or not forgotten when it should have been"
            );
        } else {
            self.0.remote_lock.store(false, Ordering::Release);
        }
    }
}

/// Error returned when attempting to acquire a busy work queue lock
#[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
struct Locked;

// === Lock-free ring buffer ===

/// Lock-free ring buffer specialized for mostly-local work scheduling
///
/// Like all ring buffers, this implements a bounded `VecDeque`-like container.
/// The right-hand side or "front" of this ring buffer is meant to be owned by a
/// worker thread and will also be referred to as the local side, while the
/// left-hand side or "front" of this ring buffer is shared with all other OS
/// threads and will be referred to as the remote side.
///
/// The performance of this ring buffer is optimized for scenarios where pushing
/// and popping work on the local side are the most common transactions that
/// must have the best performance, and trading work with other threads on the
/// remote side is a less common transaction that can be less efficient.
///
/// Concurrent access to the remote side of the ring buffer must be synchronized
/// through a lock, which is not part of this data structure because it should
/// be placed on a separate cache line in order to minimize interference between
/// remote threads and the local worker.
#[derive(Debug)]
#[doc(hidden)]
pub struct WorkDeque<T> {
    /// Queue elements in a ring buffer layout
    ///
    /// Guaranteed to contain initialized elements from the remote index
    /// **exclusive** to the local index **inclusive**. Iterating from the
    /// remote index to the local index may or may not require circular
    /// wraparound at the end of the buffer. At any point in time, the current
    /// queue length is `local_idx.wrapping_sub(remote_idx) % elements.len()`
    /// and the queue is considered empty when `local_idx == remote_idx`.
    ///
    /// The off-by-one queue elements at the remote index and at the index after
    /// the local index are reserved for use when pushing data to the remote and
    /// local ends of the queue. To avoid all possibilities of data race UB,
    /// these two indices must not be allowed to overlap with other queue
    /// elements or with each other, which means that the queue must be
    /// considered full when `remote_idx == (local_idx + 2) % elements.len()`.
    ///
    /// Because there must be 2 unused elements at all times for the above
    /// purpose, the useful queue capacity is `elements.len() - 2`.
    elements: Box<[UnsafeCell<MaybeUninit<T>>]>,

    /// The length of `elements` is guaranteed to `2.pow(elements_len_pow2)`
    ///
    /// The compiler cannot figure out that `elements.len()` is a power of two
    /// without this help, and letting it know makes the modular computations
    /// much more efficient (can use bit masking instead of true remainder).
    elements_len_pow2: NonZeroU32,

    /// Bit field with the following layout:
    ///
    /// ```text
    /// +----------------+----------------+
    /// |RRRRRRRRRRRRRRRR|LLLLLLLLLLLLLLLL|
    /// +----------------+----------------+
    /// ```
    ///
    /// - Word R is the index before the first element, aka remote index
    /// - Word L is the index of the last element, aka local index
    range: AtomicUsize,
}
//
impl<T> WorkDeque<T> {
    // === High-level interface ===

    /// Set up the ring buffer
    ///
    /// See [`Worker::new()`] docs to know more about acceptable capacities.
    pub fn new(capacity: usize) -> Self {
        let elements_len = Self::num_elements_from_capacity(capacity);
        let elements = std::iter::repeat_with(|| UnsafeCell::new(MaybeUninit::uninit()))
            .take(elements_len)
            .collect();
        let elements_len_pow2 = NonZeroU32::new(elements_len.trailing_zeros())
            .expect("will have at least two elements due to push buffers on both ends");
        Self {
            elements,
            elements_len_pow2,
            range: AtomicUsize::new(0),
        }
    }

    /// Determine internal buffer size for a certain user-desired capacity
    fn num_elements_from_capacity(capacity: usize) -> usize {
        assert!(
            capacity < MAX_CAPACITY,
            "requested capacity is above implementation limit"
        );
        let mut elements_len = capacity + 2;
        if !elements_len.is_power_of_two() {
            elements_len = elements_len.next_power_of_two();
        }
        elements_len
    }

    /// Push work on the worker side of the deque
    ///
    /// See [`Worker::push()`] for more information.
    ///
    /// # Safety
    ///
    /// This function may only be called by the thread that owns the work queue.
    pub unsafe fn push(&self, work: T) -> Result<(), Full<T>> {
        // Make sure the queue isn't full
        //
        // Can use Relaxed ordering because at this stage, we're only using the
        // range word to know the position of our private storage slot, which is
        // a value that only us, the active worker thread, may have previously
        // set. Therefore, this is the worker thread communicating with itself,
        // which is one of the textbook use cases for Relaxed memory ordering.
        let mut range = self.load_range(Ordering::Relaxed);
        if self.is_full(range) {
            return Err(Full(work));
        }

        // Write data in our private storage slot
        let push_idx = self.push_idx(range);
        // SAFETY: This is safe because per the method's precondition, only the
        //         worker thread can execute this code, and the queue's
        //         synchronization protocol ensures that this storage slot is
        //         only accessible by the worker thread.
        let push_slot = unsafe { &mut *self.elements[push_idx].get() };
        push_slot.write(work);

        // Repeatedly try to shift the local end of the queue forward to make
        // our write visible to other threads, until we succeed or another
        // thread fills up the queue, making our write unsafe to commit.
        while let Err(new_range) = self.try_update_range(
            range,
            Range {
                local_idx: push_idx,
                ..range
            },
            // Need Release ordering on success to ensure our push_slot writes
            // are visible to a thread which observes the range word update.
            // Need Acquire ordering as well to ensure that our future writes to
            // the newly acquired storage slot will be ordered after we have
            // acquired access to it.
            Ordering::AcqRel,
            // Can use Relaxed ordering on failure because we're not reading any
            // state set by another thread except for the range word.
            Ordering::Relaxed,
        ) {
            debug_assert_eq!(
                new_range.local_idx, range.local_idx,
                "only the worker thread is allowed to change local_idx"
            );
            if self.is_full(new_range) {
                return Err(Full(
                    // SAFETY: We can only get here by initializing push_slot,
                    //         and double drop cannot happen because we did not
                    //         manage to move local_idx, and thus this slot will
                    //         be considered uninitialized by any future
                    //         interaction with the queue, which will inhibit
                    //         readout and read-dependent operations like Drop.
                    unsafe { push_slot.assume_init_read() },
                ));
            }
            range = new_range;
        }
        Ok(())
    }

    /// Get the next task from the worker side of the deque, if any
    ///
    /// See [`Worker::pop()`] for more information.
    ///
    /// # Safety
    ///
    /// This function may only be called by the thread that owns the work queue.
    pub unsafe fn pop(&self) -> Option<T> {
        // Make sure the queue isn't empty
        //
        // Relaxed ordering is fine at this stage because we're going to re-read
        // the range word with Acquire ordering when we commit the pop later.
        let mut range = self.load_range(Ordering::Relaxed);
        if self.is_empty(range) {
            return None;
        }

        // Repeatedly try to shift the local end of the queue backwards, which
        // moves the next local element to our private storage slot and thus
        // protects it from concurrent pushing and popping by other threads.
        // Stop once we succeed or other threads steal all remaining work.
        let pop_idx = self.pop_idx(range);
        let next_idx = self.index_sub(pop_idx, 1);
        while let Err(new_range) = self.try_update_range(
            range,
            Range {
                local_idx: next_idx,
                ..range
            },
            // Need Acquire ordering on success to ensure we observe every write
            // from the thread that added this element to the queue.
            // Need Release ordering as well to ensure that all writes to our
            // former private storage slot have completed before we free up
            // said slot for reuse by other threads.
            Ordering::AcqRel,
            // Can use Relaxed ordering on failure because we're not reading any
            // state set by another thread except for the range word
            Ordering::Relaxed,
        ) {
            debug_assert_eq!(
                new_range.local_idx, range.local_idx,
                "only the worker thread is allowed to change local_idx"
            );
            if self.is_empty(range) {
                return None;
            }
            range = new_range;
        }

        // Now, take the element from our private storage slot
        // SAFETY: This is safe because we made the slot at pop_idx the worker
        //         thread's private storage slot, and per the method's
        //         precondition, we assume that only the worker thread may
        //         execute this method.
        let pop_slot = unsafe { &mut *self.elements[pop_idx].get() };
        // SAFETY: This storage slot was previously advertised as being part of
        //         the initialized range of the queue, so it should contain an
        //         initialized value. Double drop cannot happen because we
        //         shifted local_idx backward, so all future queue interactions
        //         will consider this element to be uninitialized and won't do
        //         any read or read-dependent operation like Drop there.
        Some(unsafe { pop_slot.assume_init_read() })
    }

    /// Add work on the remote side of the deque
    ///
    /// See [`Remote::give()`] for more information.
    ///
    /// # Safety
    ///
    /// Access to the remote end of the queue should be synchronized via an
    /// external lock, which controls access to `remote_idx` and the associated
    /// private `elements[remote_idx]` storage cell.
    pub unsafe fn give(&self, work: T) -> Result<(), Full<T>> {
        // Make sure the queue isn't full
        //
        // Can use Relaxed ordering because acquisition of the remote end's lock
        // has already provided the necessary `Acquire` memory barrier for us to
        // get a consistent view of remote_idx and the associated storage slot.
        let mut range = self.load_range(Ordering::Relaxed);
        if self.is_full(range) {
            return Err(Full(work));
        }

        // Write data in our private storage slot
        let give_idx = self.give_idx(range);
        // SAFETY: This is safe because we have acquired the remote lock and per
        //         the synchronization protocol, this storage slot is reserved
        //         for use by the thread which has acquired that lock.
        let give_slot = unsafe { &mut *self.elements[give_idx].get() };
        give_slot.write(work);

        // Repeatedly try to shift the remote end of the queue backward to make
        // our write visible to other threads, until either we succeed or the
        // worker thread fills up the queue, making our write unsafe to commit.
        let next_idx = self.index_sub(give_idx, 1);
        while let Err(new_range) = self.try_update_range(
            range,
            Range {
                remote_idx: next_idx,
                ..range
            },
            // Need Release ordering on success to ensure our give_slot writes
            // are visible to a thread which observes the range word update.
            // Need Acquire ordering as well to ensure that our future writes to
            // the newly acquired storage slot cannot be reordered before we
            // have acquired access to said storage slot.
            Ordering::AcqRel,
            // Can use Relaxed ordering on failure because we're not reading any
            // state set by another thread except for the range word
            Ordering::Relaxed,
        ) {
            debug_assert_eq!(
                new_range.remote_idx, range.remote_idx,
                "only the remote lock holder is allowed to change remote_idx"
            );
            if self.is_full(new_range) {
                return Err(Full(
                    // SAFETY: We can only get here by initializing give_slot,
                    //         and double drop cannot happen because we did not
                    //         manage to shift remote_idx, and thus this slot
                    //         will be considered uninitialized by any future
                    //         interaction with the queue and not be read.
                    unsafe { give_slot.assume_init_read() },
                ));
            }
            range = new_range;
        }
        Ok(())
    }

    /// Take work from the remote side of the deque
    ///
    /// See [`Remote::steal()`] for more information.
    ///
    /// # Safety
    ///
    /// Access to the remote end of the queue should be synchronized via an
    /// external lock, which controls access to `remote_idx` and the associated
    /// private `elements[remote_idx]` storage cell.
    pub unsafe fn steal(&self) -> Option<T> {
        // Make sure the queue isn't empty
        //
        // Relaxed ordering is fine at this stage because we're going to re-read
        // the range word with Acquire ordering when we commit the pop later.
        let mut range = self.load_range(Ordering::Relaxed);
        if self.is_empty(range) {
            return None;
        }

        // Repeatedly try to shift the remote end of the queue forward, which
        // moves the next remote element to our private storage slot and thus
        // protects it from concurrent pushing and popping by the worker thread.
        // Stop once we succeed or the worker thread pops all remaining work.
        let steal_idx = self.steal_idx(range);
        while let Err(new_range) = self.try_update_range(
            range,
            Range {
                remote_idx: steal_idx,
                ..range
            },
            // Need Acquire ordering on success to ensure we observe every write
            // from the thread that added this element to the queue.
            // Need Release ordering as well to ensure that any previous writes
            // to our former storage slot have completed before we release it
            // for re-use by the outside world.
            Ordering::AcqRel,
            // Can use Relaxed ordering on failure because we're not reading any
            // state set by another thread except for the range word
            Ordering::Relaxed,
        ) {
            debug_assert_eq!(
                new_range.remote_idx, range.remote_idx,
                "only the remote lock holder is allowed to change remote_idx"
            );
            if self.is_empty(range) {
                return None;
            }
            range = new_range;
        }

        // Now, take the element from our private storage slot
        // SAFETY: This is safe because...
        //         - steal_idx is now our private remote storage slot
        //         - The remote lock is assumed to be held as a precondition, so
        //           remote_idx cannot change and we are protected from access
        //           to the associated elements[remote_idx] private storage
        //           block by other remote threads.
        //         - The queue's synchronization protocol ensures that the
        //           worker thread won't access our private storage slot either.
        let steal_slot = unsafe { &mut *self.elements[steal_idx].get() };
        // SAFETY: This storage slot was previously advertised as being part of
        //         the initialized range of the queue, so it should contain an
        //         initialized value. Double drop cannot happen because we
        //         shifted remote_idx forward, so future queue interactions will
        //         consider this element to be uninitialized and won't read it.
        Some(unsafe { steal_slot.assume_init_read() })
    }

    // === Range word manipulation ===

    /// Read and decode the current ring buffer range
    fn load_range(&self, order: Ordering) -> Range {
        Range::new(self.range.load(order), self.index_mask())
    }

    /// Attempt to update the range word, assuming a certain initial state
    ///
    /// This function is just a high-level wrapper over
    /// [`AtomicUsize::compare_exchange_weak()`] and has nearly identical
    /// semantics: it is guaranteed to return Ok() if replacement succeeds, it
    /// has `success` memory ordering on success and `failure` memory ordering
    /// on failure, and if it fails it gives back the new range value.
    fn try_update_range(
        &self,
        current: Range,
        new: Range,
        success: Ordering,
        failure: Ordering,
    ) -> Result<(), Range> {
        let index_mask = self.index_mask();
        self.range
            .compare_exchange_weak(
                current.into_raw(index_mask),
                new.into_raw(index_mask),
                success,
                failure,
            )
            .map(std::mem::drop)
            .map_err(|changed| Range::new(changed, index_mask))
    }

    // === Lengths and indices ===

    /// Like `self.elements.len()`, but optimized for use as a modulo argument
    #[inline(always)]
    fn elements_len(&self) -> usize {
        let elements_len = 2usize.pow(self.elements_len_pow2.get());
        debug_assert_eq!(elements_len, self.elements.len());
        elements_len
    }

    /// Mask used to decode either the bit-shifted local or remote index
    fn index_mask(&self) -> RawRange {
        let result = self.elements_len() - 1;
        debug_assert_eq!(
            result & ((1 << Range::INDEX_BITS) - 1),
            result,
            "ring buffer index mask cannot go above the range word's index bit budget"
        );
        result
    }

    /// Go N places after a ring buffer index
    fn index_add(&self, idx: usize, offset: usize) -> usize {
        idx.wrapping_add(offset) % self.elements_len()
    }

    /// Go N places before a ring buffer index
    fn index_sub(&self, idx: usize, offset: usize) -> usize {
        idx.wrapping_sub(offset) % self.elements_len()
    }

    // === Interpreting the ring buffer range ===

    /// Truth that the ring buffer is empty
    fn is_empty(&self, range: Range) -> bool {
        range.local_idx == range.remote_idx
    }

    /// Truth that the ring buffer is full
    fn is_full(&self, range: Range) -> bool {
        self.index_add(range.local_idx, 2) == range.remote_idx
    }

    /// Index where `push()` would insert an element
    fn push_idx(&self, range: Range) -> usize {
        self.index_add(range.local_idx, 1)
    }

    /// Index where `pop()` would take an element
    fn pop_idx(&self, range: Range) -> usize {
        range.local_idx
    }

    /// Index where `give()` would insert an element
    ///
    /// If the [`WorkDeque`]'s remote end is accessible to multiple threads
    /// (i.e. at any time except right after construction and right before the
    /// final `Drop`), you must acquire a lock before doing anything on this
    /// side of the ring buffer.
    fn give_idx(&self, range: Range) -> usize {
        range.remote_idx
    }

    /// Index where `steal()` would take an element
    ///
    /// If the [`WorkDeque`]'s remote end is accessible to multiple threads
    /// (i.e. at any time except right after construction and right before the
    /// final `Drop`), you must acquire a lock before doing anything on this
    /// side of the ring buffer.
    fn steal_idx(&self, range: Range) -> usize {
        self.index_add(range.remote_idx, 1)
    }
}
//
#[cfg(test)]
impl<T: Arbitrary> Arbitrary for WorkDeque<T>
where
    <T as Arbitrary>::Parameters: Clone + 'static,
{
    type Parameters = (prop::collection::SizeRange, <T as Arbitrary>::Parameters);
    type Strategy = prop::strategy::BoxedStrategy<Self>;

    fn arbitrary_with((capacity_range, value_params): Self::Parameters) -> Self::Strategy {
        (capacity_range.start()..=capacity_range.end_incl())
            .prop_flat_map(move |capacity| {
                let num_elements = Self::num_elements_from_capacity(capacity);
                let elements = prop::collection::vec(
                    <T as Arbitrary>::arbitrary_with(value_params.clone()),
                    num_elements,
                );
                let range = (0..num_elements).prop_flat_map(move |remote_idx| {
                    (remote_idx..=(remote_idx + num_elements - 2)).prop_map(
                        move |unwrapped_local_idx| Range {
                            local_idx: unwrapped_local_idx % num_elements,
                            remote_idx,
                        },
                    )
                });
                (elements, range).prop_map(move |(elements, range)| {
                    let mut deque = Self::new(capacity);
                    assert_eq!(deque.elements.len(), num_elements);
                    for (element, value) in deque.elements.iter_mut().zip(elements) {
                        element.get_mut().write(value);
                    }
                    deque
                        .range
                        .store(range.into_raw(deque.index_mask()), Ordering::Relaxed);
                    deque
                })
            })
            .boxed()
    }
}
//
impl<T> Drop for WorkDeque<T> {
    fn drop(&mut self) {
        let range = self.load_range(Ordering::Acquire);
        if self.is_empty(range) {
            return;
        }
        let first_elem_idx = self.steal_idx(range);
        let last_elem_idx = self.pop_idx(range);
        if first_elem_idx <= last_elem_idx {
            for element in &mut self.elements[first_elem_idx..=last_elem_idx] {
                // SAFETY: Queue elements from `first_elem_idx` to
                //         `last_elem_idx` should be initialized and ready to be
                //         popped from both ends, therefore dropping them should
                //         be valid. No data race should be possible in the
                //         presence of `&mut self` exclusive access.
                unsafe { element.get_mut().assume_init_drop() }
            }
        } else {
            let (head_elems, first_elems) = self.elements.split_at_mut(first_elem_idx);
            let last_elems = &mut head_elems[..=last_elem_idx];
            for element in first_elems.iter_mut().chain(last_elems) {
                // SAFETY: Queue elements from `first_elem_idx` to
                //         `last_elem_idx`, with wraparound in the middle,
                //         should be initialized and ready to be popped from
                //         both ends, therefore dropping them should be valid.
                //         No data race should be possible in the presence of
                //         `&mut self` exclusive access.
                unsafe { element.get_mut().assume_init_drop() }
            }
        }
    }
}

/// Packed [`WorkDeque`] range
///
/// See [`WorkDeque::range`] documentation for more information
type RawRange = usize;

/// Decoded [`WorkDeque`] range
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
struct Range {
    /// Local end of the queue
    ///
    /// This is the index of [`WorkDeque::elements`] from which
    /// [`WorkDeque::pop()`] would pop the next work-item.
    local_idx: usize,

    /// Remote end of the queue
    ///
    /// This is the index of [`WorkDeque::elements`] at which
    /// [`WorkDeque::give()`] would add a new work-item.
    ///
    /// If the remote end of the [`WorkDeque`] is accessible to multiple threads
    /// (i.e. at any time except right after construction and right before the
    /// final `Drop`), you must acquire a lock before modifying `remote_idx`
    /// and/or accessing the private element at this location.
    remote_idx: usize,
}
//
impl Range {
    /// Decode the [`WorkDeque`]'s packed range
    pub fn new(raw: RawRange, index_mask: RawRange) -> Self {
        let local_idx = raw & index_mask;
        let remote_idx = (raw >> Self::REMOTE_SHIFT) & index_mask;
        Self {
            local_idx,
            remote_idx,
        }
    }

    /// Convert back to a packed range
    pub fn into_raw(self, index_mask: RawRange) -> RawRange {
        let mut result = 0;
        debug_assert_eq!(self.local_idx & index_mask, self.local_idx);
        result |= self.local_idx;
        debug_assert_eq!(self.remote_idx & index_mask, self.remote_idx);
        result |= self.remote_idx << Self::REMOTE_SHIFT;
        result
    }

    /// Available bits for storing the local and remote indices
    pub const INDEX_BITS: u32 = Self::REMOTE_SHIFT;

    /// Bit shift from the start of the word to the start of the end index
    const REMOTE_SHIFT: u32 = RawRange::BITS / 2;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_panics;
    use proptest::sample::SizeRange;
    use std::ptr;

    /// Buffer capacity that is usually valid, but may not be
    fn buffer_capacity() -> impl Strategy<Value = usize> {
        let max_size = SizeRange::default().end_incl();
        if max_size > MAX_CAPACITY {
            prop_oneof![
                1 => prop_oneof![Just(0), Just(MAX_CAPACITY)],
                2 => 1..MAX_CAPACITY,
                2 => (MAX_CAPACITY+1)..=max_size,
            ]
            .boxed()
        } else {
            prop_oneof![
                1 => Just(0),
                4 => 1..=max_size,
            ]
            .boxed()
        }
    }

    proptest! {
        #[test]
        fn new(capacity in buffer_capacity()) {
            if capacity > MAX_CAPACITY {
                assert_panics(|| WorkDeque::<bool>::new(capacity))?;
                assert_panics(|| SharedDeque::<u8>::new(capacity))?;
                assert_panics(|| Worker::<f32>::new(capacity))?;
                return Ok(());
            }

            fn check_initial_deque<T>(deque: &WorkDeque<T>, capacity: usize) -> Result<(), TestCaseError> {
                let expected_elements = WorkDeque::<T>::num_elements_from_capacity(capacity);
                prop_assert_eq!(deque.elements.len(), expected_elements);
                prop_assert_eq!(2usize.pow(deque.elements_len_pow2.get()), expected_elements);
                prop_assert_eq!(deque.range.load(Ordering::Relaxed), 0);
                Ok(())
            }
            check_initial_deque(&WorkDeque::<i16>::new(capacity), capacity)?;

            fn check_initial_shared<T>(shared: &SharedDeque<T>, capacity: usize) -> Result<(), TestCaseError> {
                check_initial_deque(&shared.deque, capacity)?;
                prop_assert!(!shared.remote_lock.load(Ordering::Relaxed));
                Ok(())
            }
            let shared = SharedDeque::<isize>::new(capacity);
            check_initial_shared(&shared, capacity)?;

            let worker = Worker::<u128>::new(capacity);
            check_initial_shared(&worker.0, capacity)?;

            let remote = worker.remote();
            prop_assert!(ptr::eq(&*worker.0, &*remote.0));
            check_initial_shared(&remote.0, capacity)?;
        }
    }

    /// Test read-only properties of WorkDeque
    fn check_deque_state<T>(deque: &WorkDeque<T>) -> Result<(), TestCaseError> {
        // Fast path to power-of-2 elements len and associated bitmask
        prop_assert_eq!(deque.elements_len(), deque.elements.len());
        prop_assert!(deque.elements_len().is_power_of_two());
        prop_assert_eq!(deque.index_mask(), deque.elements_len() - 1);

        // Range upholds invariants
        let range = deque.load_range(Ordering::Relaxed);
        prop_assert!(range.local_idx < deque.elements_len());
        prop_assert!(range.remote_idx < deque.elements_len());
        prop_assert_eq!(
            range.into_raw(deque.index_mask()),
            deque.range.load(Ordering::Relaxed)
        );
        prop_assert_eq!(deque.is_empty(range), range.local_idx == range.remote_idx);
        prop_assert_eq!(
            deque.is_full(range),
            range.remote_idx == (range.local_idx + 2) % deque.elements_len()
        );
        prop_assert_eq!(deque.pop_idx(range), range.local_idx);
        prop_assert_eq!(
            deque.push_idx(range),
            (range.local_idx + 1) % deque.elements_len()
        );
        prop_assert_eq!(deque.give_idx(range), range.remote_idx);
        prop_assert_eq!(
            deque.steal_idx(range),
            (range.remote_idx + 1) % deque.elements_len()
        );
        prop_assert_ne!(
            deque.push_idx(range),
            deque.give_idx(range),
            "private push should nver overlap"
        );
        Ok(())
    }
    //
    proptest! {
        #[test]
        fn work_state(deque: WorkDeque<isize>) {
            check_deque_state(&deque)?;
        }
        //
        #[test]
        fn read_only_shared(shared: SharedDeque<u32>) {
            check_deque_state(&shared.deque)?;
        }
        //
        #[test]
        fn read_only_worker(worker: Worker<f64>) {
            check_deque_state(&worker.0.deque)?;
        }
    }

    /// Test inputs for ring buffer indexing
    fn indexing_test_params() -> impl Strategy<Value = (WorkDeque<isize>, usize, usize)> {
        buffer_capacity().prop_flat_map(|capacity| {
            let num_elems = WorkDeque::<isize>::num_elements_from_capacity(capacity);
            (0..num_elems, 0..=num_elems)
                .prop_map(move |(idx, offset)| (WorkDeque::new(capacity), idx, offset))
        })
    }
    //
    proptest! {
        #[test]
        fn index_ops((deque, idx, offset) in indexing_test_params()) {
            prop_assert_eq!(
                deque.index_add(idx, offset),
                (idx + offset) % deque.elements.len()
            );
            let sub = deque.index_sub(idx, offset);
            let offset = offset % deque.elements.len();
            if idx >= offset {
                prop_assert_eq!(sub, idx - offset);
            } else {
                let expected = deque.elements.len() - offset + idx;
                prop_assert_eq!(sub, expected);
            }
        }
    }

    // TODO: Test basic transactions of RingBuffer from arbitrary state...
    //       - From a single thread
    //       - From two concurrent threads, one taking the local side and one
    //         taking the remote side, reverting transactions after running them
    // TODO: Test transactions of one Worker + 2 Remote, each in own thread
    // TODO: Also add benchmarks once tests are ready, and benchmark
    //       crossbeam_deque on the same task when feasible (+ add
    //       crossbeam/crossbeam_deque to benchmark feature)
}
