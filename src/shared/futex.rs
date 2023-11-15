//! Futex used for blocking worker synchronization

#[cfg(test)]
use proptest::{
    prelude::*,
    strategy::{Map, TupleUnion},
};
use std::{
    fmt::{self, Debug},
    sync::atomic::{self, AtomicU32, Ordering},
};
#[cfg(test)]
use std::{ops::Range, sync::Arc};

/// Futex used for blocking worker synchronization
///
/// This tracks the following information:
///
/// - A preferred steal location (index of another worker, RAW_LOCATION_INJECTOR
///   to steal from the global task injector, or RAW_LOCATION_NONE if there is
///   no preferred steal location at the moment)
/// - Truth that the thread pool is shutting down (this tells the worker that it
///   should finish draining remaining work from all sources and then exit)
/// - Truth that the worker fell asleep (in which case threads which submit new
///   info to the worker using this futex should wake it up with
///   atomic_wait::wake_all)
pub(crate) struct WorkerFutex(AtomicU32);
//
impl WorkerFutex {
    // --- Shared state initialization interface ---

    /// Maximal supported number of workers
    pub const MAX_WORKERS: usize = RAW_LOCATION_NUM_NORMAL as usize;

    /// Set up a worker futex
    pub const fn new() -> Self {
        Self::with_state(WorkerFutexState::INITIAL)
    }

    /// Set up a worker futex with a custom initial state
    const fn with_state(state: WorkerFutexState) -> Self {
        Self(AtomicU32::new(state.to_raw()))
    }

    // --- Interface for the worker that the futex belongs to ---

    /// Read out the current futex state
    ///
    /// This operation accepts all valid atomic orderings, not just load
    /// operation orderings, but store orderings will be less efficient.
    pub fn load_from_worker(&self, order: Ordering) -> WorkerFutexState {
        let result = self.load(order);
        debug_assert!(!result.sleeping);
        result
    }

    /// Notify futex that a `join()` operation has started
    ///
    /// This is only needed when attempting to detect situations where there are
    /// too many `join()`s in flight to guarantee ABA-safety.
    #[cfg(feature = "detect-excessive-joins")]
    #[inline]
    pub fn start_join(&self, order: Ordering) {
        // Failure ordering can be `Relaxed` because the operation does not read
        // any other application state and will eventually succeed.
        let _ = self.0.fetch_update(order, Ordering::Relaxed, |old_raw| {
            let old = WorkerFutexState::from_raw(old_raw);
            debug_assert!(!old.sleeping);
            if old.last_join_id == JoinID::MAX {
                // To clarify, the problem here is that once we let JoinID wrap
                // above its maximum value, the same futex value can appear
                // every JoinID::MAX + 1 joins. This means the worker thread
                // cannot differentiate between "no join happened since we last
                // looked up the futex" and "JoinID::MAX + 1 joins happened
                // since we last looked up the futex". This can lead the worker
                // thread to wrongly fall asleep while waiting for joins that
                // actually happened, and if there are no more pending remote
                // joins to wake it up at this point then you have a worker
                // thread deadlock on your hands...
                crate::unlikely(|| {
                    log::error!("\
                        A worker thread has more join()s in flight than the ABA-safe limit of {} concurrent joins. \
                        This worker thread has a small chance of deadlocking under bad scheduling circumstances. \
                        Consider tightening your program's sequential processing threshold, as it is highly \
                        unlikely that having this many concurrent tasks benefits your performance...\
                    ", JoinID::MAX);
                });
            }
            Some(WorkerFutexState {
                last_join_id: old.last_join_id.wrapping_add(1),
                ..old
            }.to_raw())
        });
    }

    /// Clear steal location after it becomes outdated, fail if the steal
    /// location has changed concurrently, get the new futex state in any case.
    ///
    /// Note that unlike `compare_exchange`, this returns the _updated_ futex
    /// state when it's been successfully updated.
    pub fn clear_outdated_location(
        &self,
        initial: WorkerFutexState,
        update: Ordering,
        load: Ordering,
    ) -> Result<WorkerFutexState, WorkerFutexState> {
        debug_assert!(initial.steal_location.is_some());
        let mut current = initial;
        'try_clear_location: loop {
            debug_assert!(!current.sleeping);

            // Try to clear the steal location that became outdated
            let cleared = WorkerFutexState {
                steal_location: None,
                ..current
            };
            match self
                .0
                .compare_exchange_weak(current.to_raw(), cleared.to_raw(), update, load)
            {
                // Return updated state on success
                Ok(_current_raw) => return Ok(cleared),
                Err(updated_raw) => {
                    current = WorkerFutexState::from_raw(updated_raw);
                    if current.steal_location == initial.steal_location {
                        // Keep trying as long as the recommended steal location
                        // remains the one we've been tasked to remove
                        continue 'try_clear_location;
                    } else {
                        // Abort if the recommended steal location changes
                        return Err(current);
                    }
                }
            }
        }
    }

    /// Wait for a futex state change, return new futex state
    ///
    /// OS wait is costly, so a thread should busy-wait for a reasonable amount
    /// of time before calling this method.
    #[cold]
    pub fn wait_for_change(
        &self,
        initial: WorkerFutexState,
        sleep: Ordering,
        wake: Ordering,
    ) -> WorkerFutexState {
        // We should only go to sleep in very specific circumstances
        debug_assert!(initial.steal_location.is_none() & initial.work_incoming & !initial.sleeping);

        // First notify that we're going to sleep
        //
        // Need Acquire ordering on success so that the action of setting the
        // sleeping flag cannot be reordered after that of falling asleep
        let sleeping = WorkerFutexState {
            sleeping: true,
            ..initial
        };
        let sleeping_raw = sleeping.to_raw();
        let result = self.0.compare_exchange(
            initial.to_raw(),
            sleeping_raw,
            Self::at_least_acquire(sleep),
            wake,
        );

        // If the state has changed, there's no need to sleep
        if let Err(updated_raw) = result {
            let updated = WorkerFutexState::from_raw(updated_raw);
            debug_assert!(!updated.sleeping);
            return updated;
        }

        // Otherwise, go to sleep until the state changes
        //
        // Need AcqRel ordering so each readout is not reordered before the
        // previous wait or after the next wait.
        let mut current = sleeping;
        while current == sleeping {
            atomic_wait::wait(&self.0, current.to_raw());
            current = self.load(Ordering::AcqRel);
        }

        // Apply user-requested wakeup ordering if stronger than Acquire
        if ![Ordering::Relaxed, Ordering::Acquire].contains(&wake) {
            atomic::fence(wake);
        }

        // By the time we wake up, the thread that awakened us will have cleared
        // the sleeping flag, so we can just return the new state
        debug_assert!(!current.sleeping);
        current
    }

    // --- Interface usable by other entities interacting with the futex ---

    /// Notify the worker of a new recommended stealing location, returns truth
    /// that the recommendation was accepted (it's better than the previous one)
    ///
    /// Note that unlike `compare_exchange`, this returns the _updated_ futex
    /// state when the state has been successfully updated.
    #[inline]
    pub fn suggest_steal(
        &self,
        proposed_location: StealLocation,
        worker_idx: usize,
        update: Ordering,
        load: Ordering,
    ) -> Result<WorkerFutexState, WorkerFutexState> {
        // Need Acquire ordering on success so that the action of updating the
        // location cannot be reordered after that of waking up the worker
        debug_assert!(worker_idx < Self::MAX_WORKERS);
        let update = Self::at_least_acquire(update);

        // Check out the current futex state
        let mut current_raw = self.0.load(load);
        'try_update_location: loop {
            let current = WorkerFutexState::from_raw(current_raw);

            // Abort if the proposed location isn't better than current one
            let should_update = match current.steal_location {
                Some(current_location) => proposed_location.is_closer(current_location, worker_idx),
                None => true,
            };
            if !should_update {
                debug_assert!(current.steal_location.is_some());
                return Err(current);
            }

            // Try to update the steal location with our proposal
            let new = WorkerFutexState {
                steal_location: Some(proposed_location),
                sleeping: false,
                ..current
            };
            match self
                .0
                .compare_exchange_weak(current_raw, new.to_raw(), update, load)
            {
                // Successfully updated the location, wake up the worker if it
                // was sleeping so it can acknowledge it
                Ok(old_raw) => {
                    self.wake_if_asleep(old_raw);
                    return Ok(new);
                }

                // Someone else updated the stealing location, try again
                Err(updated_raw) => {
                    current_raw = updated_raw;
                    continue 'try_update_location;
                }
            }
        }
    }

    /// Tell this worker thread that the remote end of a `join()` has completed
    #[inline]
    pub fn notify_join(&self, order: Ordering) {
        // Switch to a new join ID and cancel impeding attempts to sleep
        //
        // Need Acquire ordering so this is not reordered after wake_if_asleep.
        // Failed load ordering can be Relaxed as we're not reading any other
        // state and the operation will eventually succeed.
        let order = Self::at_least_acquire(order);
        let old_raw = self
            .0
            .fetch_update(order, Ordering::Relaxed, |old_raw| {
                let old = WorkerFutexState::from_raw(old_raw);
                Some(
                    WorkerFutexState {
                        last_join_id: old.last_join_id.wrapping_sub(1),
                        sleeping: false,
                        ..old
                    }
                    .to_raw(),
                )
            })
            .expect("not allowed to fail");

        // If we updated the futex of a sleeping thread, wake it up
        self.wake_if_asleep(old_raw);
    }

    // --- Thread pool interface ---

    /// Notify the worker that the thread pool is shutting down and won't be
    /// accepting any more work
    #[cold]
    pub fn notify_shutdown(&self, order: Ordering) {
        // Record pool shutdown and cancel any impeding attempt to sleep
        //
        // Need Acquire ordering so this is not reordered after wake_if_asleep
        let old_raw = self.0.fetch_and(
            !(FUTEX_BIT_WORK_INCOMING | FUTEX_BIT_SLEEPING),
            Self::at_least_acquire(order),
        );

        // Thread pool shutdown should only happen once in the futex's lifetime
        debug_assert_ne!(old_raw & FUTEX_BIT_WORK_INCOMING, 0);

        // If we updated the futex of a sleeping thread, wake it up
        self.wake_if_asleep(old_raw);
    }

    // --- Internal utilities ---

    /// Load without state validity checks
    ///
    /// This operation accepts all valid atomic orderings, not just load
    /// operation orderings, but store orderings will be less efficient.
    #[inline]
    fn load(&self, order: Ordering) -> WorkerFutexState {
        if [Ordering::Relaxed, Ordering::Acquire, Ordering::SeqCst].contains(&order) {
            WorkerFutexState::from_raw(self.0.load(order))
        } else {
            WorkerFutexState::from_raw(self.0.fetch_add(0, order))
        }
    }

    /// If we updated the futex of a sleeping thread, wake it up
    ///
    /// This should be done after performing a RMW operation with Acquire or
    /// stronger ordering that clears the SLEEPING bit and returns the previous
    /// futex state.
    fn wake_if_asleep(&self, old_raw: RawWorkerFutexState) {
        if old_raw & FUTEX_BIT_SLEEPING != 0 {
            if cfg!(debug_assertions) {
                let old = WorkerFutexState::from_raw(old_raw);
                assert!(old.steal_location().is_none());
                assert!(old.work_incoming());
            }
            atomic_wait::wake_all(&self.0)
        }
    }

    /// Add an Acquire barrier to a user-specified ordering
    #[inline]
    fn at_least_acquire(order: Ordering) -> Ordering {
        match order {
            Ordering::Relaxed | Ordering::Acquire => Ordering::Acquire,
            Ordering::Release | Ordering::AcqRel => Ordering::AcqRel,
            Ordering::SeqCst => Ordering::SeqCst,
            _ => unimplemented!(),
        }
    }
}
//
#[cfg(test)]
impl Arbitrary for WorkerFutex {
    type Parameters = <WorkerFutexState as Arbitrary>::Parameters;
    type Strategy = Map<<WorkerFutexState as Arbitrary>::Strategy, fn(WorkerFutexState) -> Self>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        <WorkerFutexState as Arbitrary>::arbitrary_with(args).prop_map(Self::with_state)
    }
}
//
impl Debug for WorkerFutex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.load(Ordering::Relaxed);
        let s = if f.alternate() {
            format!("WorkerFutex({state:#?})")
        } else {
            format!("WorkerFutex({state:?})")
        };
        f.pad(&s)
    }
}

/// Current worker futex state
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct WorkerFutexState {
    /// Truth that the thread pool is reachable and may receive more work
    work_incoming: bool,

    /// Truth that the worker is sleeping
    sleeping: bool,

    /// Location from which this worker is recommended to steal
    steal_location: Option<StealLocation>,

    /// Identifier of the last completed `join()` on this futex
    ///
    /// When the `detect-excessive-joins` feature is enabled, this tracks the
    /// number of ongoing joins whose remote task has not completed yet: it's
    /// incremented when a join() begins, and it's decremented when the remote
    /// task of a join() ends.
    ///
    /// When that feature is not enabled, this is just a free-running counter
    /// that changes every time the remote task of a join completes.
    ///
    /// In both cases, the counter can wrap around, but in the former case it's
    /// considered to be an error whereas in the latter case it's a normal fact
    /// of life that does not warrant any particular notification.
    last_join_id: JoinID,
}
//
impl WorkerFutexState {
    /// Initial worker futex state
    const INITIAL: WorkerFutexState = WorkerFutexState {
        steal_location: None,
        work_incoming: true,
        sleeping: false,
        last_join_id: 0,
    };

    /// Truth that the thread pool is reachable and may receive more work
    ///
    /// Once this has become `false`, it can never become `true` again.
    pub(crate) const fn work_incoming(&self) -> bool {
        self.work_incoming
    }

    /// Location that this thread is recommended stealing from
    pub(crate) const fn steal_location(&self) -> Option<StealLocation> {
        self.steal_location
    }

    /// Decode the raw state from the futex data
    const fn from_raw(raw: RawWorkerFutexState) -> Self {
        let work_incoming = raw & FUTEX_BIT_WORK_INCOMING != 0;
        let sleeping = raw & FUTEX_BIT_SLEEPING != 0;
        let raw_location = (raw & FUTEX_LOCATION_MASK) >> FUTEX_LOCATION_SHIFT;
        let steal_location = StealLocation::from_raw(raw_location);
        let raw_last_join_id = (raw & FUTEX_JOIN_ID_MASK) >> FUTEX_JOIN_ID_SHIFT;
        let last_join_id = raw_last_join_id as JoinID;
        let result = Self {
            work_incoming,
            sleeping,
            steal_location,
            last_join_id,
        };
        result.debug_check_state();
        result
    }

    /// Convert back to raw futex data
    const fn to_raw(self) -> RawWorkerFutexState {
        self.debug_check_state();
        let mut raw = StealLocation::to_raw(self.steal_location) << FUTEX_LOCATION_SHIFT;
        raw |= (self.last_join_id as RawWorkerFutexState) << FUTEX_JOIN_ID_SHIFT;
        if self.work_incoming {
            raw |= FUTEX_BIT_WORK_INCOMING;
        }
        if self.sleeping {
            raw |= FUTEX_BIT_SLEEPING;
        }
        raw
    }

    /// Check that current futex state makes sense in debug builds
    const fn debug_check_state(&self) {
        debug_assert!(self.last_join_id == 0 || self.last_join_id.ilog2() < FUTEX_JOIN_ID_BITS);
        let raw_steal_location = StealLocation::to_raw(self.steal_location);
        debug_assert!(raw_steal_location == 0 || raw_steal_location.ilog2() < FUTEX_LOCATION_BITS);
        if self.sleeping {
            debug_assert!(self.steal_location.is_none());
            debug_assert!(self.work_incoming);
        }
    }
}
//
#[cfg(test)]
impl Arbitrary for WorkerFutexState {
    type Parameters = <Option<StealLocation> as Arbitrary>::Parameters;
    type Strategy = TupleUnion<(
        (
            u32,
            Arc<
                Map<
                    (
                        <Option<StealLocation> as Arbitrary>::Strategy,
                        <(bool, JoinID) as Arbitrary>::Strategy,
                    ),
                    fn((Option<StealLocation>, (bool, JoinID))) -> Self,
                >,
            >,
        ),
        (
            u32,
            Arc<Map<<(bool, JoinID) as Arbitrary>::Strategy, fn((bool, JoinID)) -> Self>>,
        ),
    )>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        prop_oneof![
            // Non-sleeping state may have arbitrary recommended stealing
            // location and no more work incoming
            4 => (<Option<StealLocation> as Arbitrary>::arbitrary_with(args), any::<(bool, JoinID)>())
                .prop_map(|(steal_location, (work_incoming, last_join_id))| Self {

                    work_incoming,
                    sleeping: false,
                    steal_location,
                    last_join_id,
                }),
            // Worker may only sleep when there is no recommended stealing
            // location and work might still be incoming
            1 => any::<(bool, JoinID)>().prop_map(|(sleeping, last_join_id)| Self {
                work_incoming: true,
                sleeping,
                steal_location: None,
                last_join_id,
            })
        ]
    }
}
//
impl Default for WorkerFutexState {
    fn default() -> Self {
        Self::INITIAL
    }
}

/// Recommended work-stealing location
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum StealLocation {
    /// Steal from this worker thread
    Worker(usize),

    /// Steal from the global injector
    Injector,
}
//
impl StealLocation {
    /// Translate a raw location from the futex to a high-level location
    const fn from_raw(raw: RawStealLocation) -> Option<Self> {
        match raw {
            RAW_LOCATION_NONE => None,
            RAW_LOCATION_INJECTOR => Some(StealLocation::Injector),
            worker_idx => {
                assert!(worker_idx < RAW_LOCATION_NUM_NORMAL);
                Some(StealLocation::Worker(worker_idx as usize))
            }
        }
    }

    /// Translate a high-level location back into a raw location for the futex
    const fn to_raw(opt: Option<Self>) -> RawStealLocation {
        match opt {
            Some(StealLocation::Worker(worker_idx)) => {
                assert!(worker_idx < WorkerFutex::MAX_WORKERS);
                worker_idx as RawStealLocation
            }
            Some(StealLocation::Injector) => RAW_LOCATION_INJECTOR,
            None => RAW_LOCATION_NONE,
        }
    }

    /// Truth that this location is closer to a specific worker thread than
    /// another location
    fn is_closer(self, other: StealLocation, worker_idx: usize) -> bool {
        debug_assert!(worker_idx < WorkerFutex::MAX_WORKERS);
        match (self, other) {
            (Self::Worker(self_idx), Self::Worker(other_idx)) => {
                self_idx.abs_diff(worker_idx) < other_idx.abs_diff(worker_idx)
            }
            (Self::Worker(_), Self::Injector) => true,
            (Self::Injector, Self::Worker(_) | Self::Injector) => false,
        }
    }
}
//
#[cfg(test)]
impl Arbitrary for StealLocation {
    type Parameters = ();
    type Strategy = TupleUnion<(
        (
            u32,
            Arc<prop::strategy::Map<Range<usize>, fn(usize) -> Self>>,
        ),
        (u32, Arc<Just<Self>>),
    )>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        prop_oneof![
            4 => (0..WorkerFutex::MAX_WORKERS).prop_map(Self::Worker),
            1 => Just(Self::Injector)
        ]
    }
}

/// Inner futex data
///
/// This is actually a bitfield that is organized as follows:
///
/// ```text
/// WSLLLLLLLLLLLLLLJJJJJJJJJJJJJJJJ
/// |||             |
/// |||             Join identifier (if detect-excessive-joins feature is
/// |||             enabled, tracks how many joins have a remote task that's
/// |||             still running, otherwise simple free-running counter that's
/// |||             changed every time the remote task of a join completes).
/// |||
/// ||Recommended steal location (MAX = No recommended location,
/// ||                            MAX-1 = Global injector,
/// ||                            other = Index of worker thread to steal from)
/// ||
/// |"Sleeping" bit (set by the worker when it falls asleep waiting for work. A
/// |                non-worker thread that modifies the futex to submit work to
/// |                the worker should clear this bit and wake up the worker if
/// |                the bit was formerly set)
/// |
/// "Work incoming" bit (thread pool is live, more work might still come)
/// ```
type RawWorkerFutexState = u32;

/// Futex status bit signaling that the thread pool is still reachable from
/// outside and thus more work might still be coming up
const FUTEX_BIT_WORK_INCOMING: RawWorkerFutexState = 1 << (RawWorkerFutexState::BITS - 1);

/// Futex status bit signaling that a thread is sleeping (or going to sleep)
const FUTEX_BIT_SLEEPING: RawWorkerFutexState = 1 << (RawWorkerFutexState::BITS - 2);

/// Last futex status bit before start of steal location data
const FUTEX_BIT_LAST: RawWorkerFutexState = FUTEX_BIT_SLEEPING;

/// Number of steal location bits
const FUTEX_LOCATION_BITS: u32 = FUTEX_BIT_LAST.trailing_zeros() - FUTEX_JOIN_ID_BITS;

/// Start of the location word
const FUTEX_LOCATION_SHIFT: u32 = FUTEX_JOIN_ID_BITS;

/// Number of low-order join identifier bits
const FUTEX_JOIN_ID_BITS: u32 = JoinID::BITS;

/// Start of the join identifier word
const FUTEX_JOIN_ID_SHIFT: u32 = 0;

/// Mask of the location word
const FUTEX_LOCATION_MASK: RawWorkerFutexState =
    ((1 << FUTEX_LOCATION_BITS) - 1) << FUTEX_LOCATION_SHIFT;

/// Mask of the join identifier
const FUTEX_JOIN_ID_MASK: RawWorkerFutexState =
    ((1 << FUTEX_JOIN_ID_BITS) - 1) << FUTEX_JOIN_ID_SHIFT;

/// Futex-internal encoding of [`StealLocation`]
type RawStealLocation = RawWorkerFutexState;
//
/// No recommended location at the moment
const RAW_LOCATION_NONE: RawStealLocation = (1 << FUTEX_LOCATION_BITS) - 1;
//
/// Steal from the global injector
const RAW_LOCATION_INJECTOR: RawStealLocation = RAW_LOCATION_NONE - 1;
//
/// Number of normal locations
const RAW_LOCATION_NUM_NORMAL: RawStealLocation = RAW_LOCATION_INJECTOR;

/// Join identifier
type JoinID = u16;

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fmt::Debug,
        sync::{atomic::AtomicBool, Barrier},
        time::Duration,
    };

    /// Atomic ordering suitable for load operations
    fn load_order() -> impl Strategy<Value = Ordering> {
        prop::sample::select(&[Ordering::Relaxed, Ordering::Acquire, Ordering::SeqCst][..])
    }

    /// Atomic ordering suitable for read-modify-write operations
    fn rmw_order() -> impl Strategy<Value = Ordering> {
        any::<Ordering>()
    }

    proptest! {
        #[test]
        fn new(load_order in rmw_order()) {
            let futex = WorkerFutex::new();
            prop_assert_eq!(futex.load(load_order), WorkerFutexState::default());
            prop_assert_eq!(futex.load_from_worker(load_order), WorkerFutexState::default());
        }

        #[test]
        fn with_state(state: WorkerFutexState, load_order in rmw_order()) {
            let futex = WorkerFutex::with_state(state);
            prop_assert_eq!(futex.load(load_order), state);
            if !state.sleeping {
                prop_assert_eq!(futex.load_from_worker(load_order), state);
            }
        }
    }

    /// Transform a Strategy for WorkerFutexState into one for WorkerFutex
    fn futex_with_state(
        state_strategy: impl Strategy<Value = WorkerFutexState>,
    ) -> impl Strategy<Value = WorkerFutex> {
        state_strategy.prop_map(WorkerFutex::with_state)
    }

    /// Futex state from a worker that's not sleeping
    fn awake_state() -> impl Strategy<Value = WorkerFutexState> {
        any::<(Option<StealLocation>, bool, JoinID)>().prop_map(
            |(steal_location, work_incoming, last_join_id)| WorkerFutexState {
                work_incoming,
                sleeping: false,
                steal_location,
                last_join_id,
            },
        )
    }

    proptest! {
        #[cfg(feature = "detecte-excessive-joins")]
        #[test]
        fn start_join(state in awake_state(), order in rmw_order()) {
            let futex = WorkerFutex::with_state(state);
            futex.start_join(order);
            assert_eq!(futex.load(Ordering::Relaxed), WorkerFutexState {
                last_join_id: state.last_join_id.wrapping_add(1),
                ..state
            });
        }
    }

    /// Futex state where steal_location is always set
    fn state_with_location() -> impl Strategy<Value = WorkerFutexState> {
        any::<(StealLocation, bool, JoinID)>().prop_map(
            |(steal_location, work_incoming, last_join_id)| WorkerFutexState {
                work_incoming,
                sleeping: false,
                steal_location: Some(steal_location),
                last_join_id,
            },
        )
    }

    /// Check that clear_outdated_location does clear the steal_location,
    /// assuming there is one steal_location set initially.
    fn check_clear_location_success(
        futex: WorkerFutex,
        update: Ordering,
        load: Ordering,
    ) -> Result<(), TestCaseError> {
        let current = futex.load_from_worker(Ordering::Relaxed);
        prop_assert!(current.steal_location.is_some() && !current.sleeping);
        let expected_out = WorkerFutexState {
            steal_location: None,
            ..current
        };
        prop_assert_eq!(
            futex.clear_outdated_location(current, update, load),
            Ok(expected_out)
        );
        prop_assert_eq!(futex.load_from_worker(Ordering::Relaxed), expected_out);
        Ok(())
    }

    proptest! {
        #[test]
        fn clear_location_success(
            futex in futex_with_state(state_with_location()),
            update in rmw_order(),
            load in load_order()
        ) {
            check_clear_location_success(futex, update, load)?;
        }

        #[test]
        fn clear_location(
            futex in futex_with_state(state_with_location()),
            expected in state_with_location(),
            update in rmw_order(),
            load in load_order()
        ) {
            let current = futex.load_from_worker(Ordering::Relaxed);
            if expected.steal_location == current.steal_location {
                check_clear_location_success(futex, update, load)?;
            } else {
                prop_assert_eq!(
                    futex.clear_outdated_location(expected, update, load),
                    Err(current)
                );
                prop_assert_eq!(futex.load_from_worker(Ordering::Relaxed), current);
            }
        }
    }

    /// State from which a thread can go to sleep
    fn waiting_state() -> impl Strategy<Value = WorkerFutexState> {
        any::<JoinID>().prop_map(|last_join_id| WorkerFutexState {
            work_incoming: true,
            sleeping: false,
            steal_location: None,
            last_join_id,
        })
    }

    /// Reasonable delay to wait for threads to go to sleep
    const WAIT_FOR_SLEEP: Duration = Duration::from_millis(100);

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 5, .. ProptestConfig::default()
        })]
        #[test]
        fn wait_for_change(
            waiting_state in waiting_state(),
            final_state in awake_state(),
            sleep in rmw_order(),
            wake in load_order(),
        ) {
            let futex = WorkerFutex::with_state(waiting_state);
            let done_waiting = AtomicBool::new(false);
            let barrier = Barrier::new(2);
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    let new_state = futex.wait_for_change(waiting_state, sleep, wake);
                    done_waiting.store(true, Ordering::Release);
                    barrier.wait();
                    assert_eq!(new_state, final_state);
                    assert_eq!(futex.load_from_worker(Ordering::Relaxed), final_state);
                });

                std::thread::sleep(WAIT_FOR_SLEEP);
                assert!(!done_waiting.load(Ordering::Relaxed));
                assert_eq!(futex.load(Ordering::Relaxed), WorkerFutexState {
                    sleeping: true,
                    ..waiting_state
                });

                futex.0.store(final_state.to_raw(), Ordering::Release);
                atomic_wait::wake_all(&futex.0);
                barrier.wait();
            });
        }

        #[test]
        fn location_is_closer(
            location1: StealLocation,
            location2: StealLocation,
            worker_idx in  0..WorkerFutex::MAX_WORKERS,
        ) {
            let assert_location1_closer = || {
                prop_assert!(location1.is_closer(location2, worker_idx));
                prop_assert!(!location2.is_closer(location1, worker_idx));
                Ok(())
            };
            let assert_location2_closer = || {
                prop_assert!(!location1.is_closer(location2, worker_idx));
                prop_assert!(location2.is_closer(location1, worker_idx));
                Ok(())
            };
            let assert_distance_equal = || {
                prop_assert!(!location1.is_closer(location2, worker_idx));
                prop_assert!(!location2.is_closer(location1, worker_idx));
                Ok(())
            };
            match (location1, location2) {
                (StealLocation::Worker(_), StealLocation::Injector) => {
                    assert_location1_closer()?;
                }
                (StealLocation::Injector, StealLocation::Injector) => {
                    assert_distance_equal()?;
                }
                (StealLocation::Injector, StealLocation::Worker(_)) => {
                    assert_location2_closer()?;
                }
                (StealLocation::Worker(idx1), StealLocation::Worker(idx2)) => {
                    let distance1 = worker_idx.abs_diff(idx1);
                    let distance2 = worker_idx.abs_diff(idx2);
                    match distance1.cmp(&distance2) {
                        std::cmp::Ordering::Less => assert_location1_closer()?,
                        std::cmp::Ordering::Equal => assert_distance_equal()?,
                        std::cmp::Ordering::Greater => assert_location2_closer()?,
                    }
                }
            }
        }
    }

    /// Test that an operation that should wake up a thread waiting on the futex
    /// does do so, and otherwise meets expectations
    fn test_waking_op<R: Debug + Eq>(
        futex: WorkerFutex,
        operation: impl FnOnce(&WorkerFutex) -> R,
        expected_result: R,
        expected_state_change: impl FnOnce(WorkerFutexState) -> WorkerFutexState + Send,
    ) -> Result<(), TestCaseError> {
        let initial = futex.load(Ordering::Relaxed);
        let start = Barrier::new(2);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                start.wait();
                let mut current = futex.load(Ordering::Relaxed);
                if initial.sleeping {
                    let initial_raw = initial.to_raw();
                    while current == initial {
                        atomic_wait::wait(&futex.0, initial_raw);
                        current = futex.load(Ordering::AcqRel);
                    }
                }

                let expected = expected_state_change(initial);
                assert!(!expected.sleeping);
                assert_eq!(current, expected);
                assert_eq!(futex.load_from_worker(Ordering::Relaxed), expected);
            });

            if initial.sleeping {
                start.wait();
                std::thread::sleep(WAIT_FOR_SLEEP);
                assert_eq!(operation(&futex), expected_result);
            } else {
                prop_assert_eq!(operation(&futex), expected_result);
                start.wait();
            }
            Ok(())
        })
    }

    proptest! {
        #[test]
        fn suggest_steal(
            futex: WorkerFutex,
            proposed_location: StealLocation,
            worker_idx in 0..WorkerFutex::MAX_WORKERS,
            update in rmw_order(),
            load in load_order(),
        ) {
            let initial = futex.load(Ordering::Relaxed);

            let closer = match initial.steal_location {
                Some(initial_location) => proposed_location.is_closer(initial_location, worker_idx),
                None => true,
            };
            let suggest = |futex: &WorkerFutex| futex.suggest_steal(proposed_location, worker_idx, update, load);
            if !closer {
                prop_assert_eq!(suggest(&futex), Err(initial));
                prop_assert_eq!(futex.load(Ordering::Relaxed), initial);
                return Ok(());
            }

            let updated = WorkerFutexState {
                steal_location: Some(proposed_location),
                sleeping: false,
                ..initial
            };
            test_waking_op(
                futex,
                suggest,
                Ok(updated),
                |_initial| updated
            )?;
        }

        #[test]
        fn notify_join(
            futex: WorkerFutex,
            order in rmw_order(),
        ) {
            test_waking_op(
                futex,
                |futex| futex.notify_join(order),
                (),
                |initial| WorkerFutexState {
                    sleeping: false,
                    last_join_id: initial.last_join_id.wrapping_sub(1),
                    ..initial
                }
            )?;
        }
    }

    /// Futex state where work_incoming is always true
    fn live_pool_state() -> impl Strategy<Value = WorkerFutexState> {
        prop_oneof![
            4 => any::<(Option<StealLocation>, JoinID)>()
                .prop_map(|(steal_location, last_join_id)| WorkerFutexState {
                    work_incoming: true,
                    sleeping: false,
                    steal_location,
                    last_join_id
                }),
            1 => any::<(bool, JoinID)>().prop_map(|(sleeping, last_join_id)| WorkerFutexState {
                work_incoming: true,
                sleeping,
                steal_location: None,
                last_join_id,
            })
        ]
    }

    proptest! {
        #[test]
        fn notify_shutdown(
            futex in futex_with_state(live_pool_state()),
            order in rmw_order(),
        ) {
            test_waking_op(
                futex,
                |futex| futex.notify_shutdown(order),
                (),
                |initial| WorkerFutexState {
                    work_incoming: false,
                    sleeping: false,
                    ..initial
                }
            )?;
        }
    }
}
