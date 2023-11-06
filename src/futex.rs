//! Futex used for blocking worker synchronization

#[cfg(test)]
use proptest::{
    prelude::*,
    strategy::{Flatten, Map, TupleUnion},
};
use std::{
    debug_assert, debug_assert_eq, debug_assert_ne,
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
/// - Truth that the worker is asleep due to lack of work (in which case threads
///   which submit new info to the worker using this futex should wake it up
///   with atomic_wait::wake_all), or that it is going to fall asleep ("sleepy")
#[derive(Debug)]
pub(crate) struct WorkerFutex(AtomicU32);
//
impl WorkerFutex {
    // --- Shared state initialization interface ---

    /// Maximal supported number of workers
    pub const MAX_WORKERS: usize = RAW_LOCATION_NUM_NORMAL as usize;

    /// Set up a worker futex
    pub fn new() -> Self {
        Self::with_state(WorkerFutexState::default())
    }

    /// Set up a worker futex with a custom initial state
    fn with_state(state: WorkerFutexState) -> Self {
        Self(AtomicU32::new(state.to_raw()))
    }

    // --- Futex owner interface ---

    /// Read out the current futex state
    pub fn load(&self, order: Ordering) -> WorkerFutexState {
        let result = WorkerFutexState::from_raw(self.0.load(order));
        debug_assert!(!result.sleeping);
        result
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
            debug_assert!(!(current.sleepy | current.sleeping));

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

    /// Notify other threads that we are going to sleep soon if nothing happens
    /// (with monitored activity being changes in futex state). Fail if
    /// something already happened to the futex since we last checked it.
    ///
    /// Other threads can prevent the worker from sleeping by clearing the
    /// SLEEPY flag or altering the futex state in any other fashion.
    ///
    /// Note that unlike `compare_exchange`, this returns the _updated_ futex
    /// state when the state has been successfully updated.
    pub fn notify_sleepy(
        &self,
        initial: WorkerFutexState,
        update: Ordering,
        load: Ordering,
    ) -> Result<WorkerFutexState, WorkerFutexState> {
        debug_assert!(
            initial.steal_location.is_none()
                & initial.work_incoming
                & !initial.sleepy
                & !initial.sleeping
        );
        let sleepy = WorkerFutexState {
            sleepy: true,
            ..initial
        };
        self.0
            .compare_exchange(initial.to_raw(), sleepy.to_raw(), update, load)
            .map(|_initial_raw| sleepy)
            .map_err(WorkerFutexState::from_raw)
    }

    /// Wait for a futex state change, return new futex state
    ///
    /// Should have announced intent to sleep with prepare_wait() and
    /// busy-waited for a reasonable amount of time before calling this method.
    #[inline]
    pub fn wait_for_change(
        &self,
        initial: WorkerFutexState,
        sleep: Ordering,
        wake: Ordering,
    ) -> WorkerFutexState {
        // We should only go to sleep in very specific circumstances
        debug_assert!(
            initial.steal_location.is_none()
                & initial.work_incoming
                & initial.sleepy
                & !initial.sleeping
        );

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

        // If the state has changed, there's no need to sleep, and the thread
        // that has updated the state will have cleared the sleepy flag for us
        if let Err(updated_raw) = result {
            let updated = WorkerFutexState::from_raw(updated_raw);
            debug_assert!(!(updated.sleepy | updated.sleeping));
            return updated;
        }

        // Otherwise, go to sleep until the state changes
        //
        // Need AcqRel ordering so each readout is not reordered before the
        // previous wait or after the next wait.
        let mut current_raw = sleeping_raw;
        while current_raw == sleeping_raw {
            atomic_wait::wait(&self.0, current_raw);
            current_raw = self.0.fetch_add(0, Ordering::AcqRel);
        }

        // Apply user-requested readout ordering if stronger than Acquire
        if ![Ordering::Relaxed, Ordering::Acquire].contains(&wake) {
            atomic::fence(wake);
        }

        // By the time we wake up, the thread that awakened us will have cleared
        // the sleepy and sleeping flags, so we can just return the new state
        let updated = WorkerFutexState::from_raw(current_raw);
        assert!(!(updated.sleepy | updated.sleeping));
        updated
    }

    // --- Other worker interface ---

    /// Notify the worker of a new recommended stealing location, returns truth
    /// that the recommendation was accepted (it's better than the previous one)
    pub fn suggest_steal(
        &self,
        proposed_location: StealLocation,
        worker_idx: usize,
        update: Ordering,
        load: Ordering,
    ) -> bool {
        // Need Acquire ordering on success so that the action of updating the
        // location cannot be reordered after that of waking up the worker
        let update = Self::at_least_acquire(update);

        // Update the stealing location if our proposal is better
        let result = self.0.fetch_update(update, load, |current_raw| {
            let current = WorkerFutexState::from_raw(current_raw);
            debug_assert!(current.sleepy | !current.sleeping);

            // Only update if the proposed location is better than current one
            let should_update = match current.steal_location {
                Some(current_location) => proposed_location.is_closer(current_location, worker_idx),
                None => true,
            };
            if !should_update {
                debug_assert!(current.steal_location.is_some() & !current.sleepy);
                return None;
            }

            // Update location and cancel any impeding attempt to sleep
            let new = WorkerFutexState {
                steal_location: Some(proposed_location),
                sleepy: false,
                sleeping: false,
                ..current
            };
            Some(new.to_raw())
        });

        // If we updated the futex of a sleeping thread, wake it up
        if let Ok(old_raw) = result {
            self.wake_if_asleep(old_raw);
        }

        // Truth that the location recommendation was accepted
        result.is_ok()
    }

    /// Wake up this worker thread if it's asleep
    ///
    /// Use this in situations where no other futex state change fits, such as
    /// when the remote end of a join() has been processed (we can't fit that in
    /// the futex because we would need an unbounded number of futex bits to
    /// encode the unbounded number of nested join() completion flags).
    pub fn wake(&self, order: Ordering) {
        // Cancel any impeding attempt to sleep
        //
        // Need Acquire ordering so this is not reordered after wake_if_asleep
        let old_raw = self.0.fetch_and(
            !(FUTEX_BIT_SLEEPY | FUTEX_BIT_SLEEPING),
            Self::at_least_acquire(order),
        );

        // If we updated the futex of a sleeping thread, wake it up
        self.wake_if_asleep(old_raw);
    }

    // --- Thread pool interface ---

    /// Notify the worker that the thread pool is shutting down and won't be
    /// accepting any more work
    pub fn notify_shutdown(&self, order: Ordering) {
        // Record pool shutdown and cancel any impeding attempt to sleep
        //
        // Need Acquire ordering so this is not reordered after wake_if_asleep
        let old_raw = self.0.fetch_and(
            !(FUTEX_BIT_WORK_INCOMING | FUTEX_BIT_SLEEPING | FUTEX_BIT_SLEEPY),
            Self::at_least_acquire(order),
        );

        // Thread pool shutdown should only happen once in the futex's lifetime
        debug_assert_ne!(old_raw & FUTEX_BIT_WORK_INCOMING, 0);

        // If we updated the futex of a sleeping thread, wake it up
        self.wake_if_asleep(old_raw);
    }

    // --- Internal utilities ---

    /// If we updated the futex of a sleeping thread, wake it up
    ///
    /// This should be done after performing a RMW operation with Acquire or
    /// stronger ordering that clears the SLEEPY and SLEEPING bits, and returns
    /// the previous futex state.
    fn wake_if_asleep(&self, old_raw: RawWorkerFutexState) {
        if old_raw & FUTEX_BIT_SLEEPING != 0 {
            debug_assert_eq!(old_raw & RAW_LOCATION_MASK, RAW_LOCATION_NONE);
            debug_assert_ne!(old_raw & (FUTEX_BIT_SLEEPY | FUTEX_BIT_WORK_INCOMING), 0);
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

/// Current worker futex state
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WorkerFutexState {
    /// Location from which this worker is recommended to steal
    steal_location: Option<StealLocation>,

    /// Truth that the worker announced intent to go to sleep
    sleepy: bool,

    /// Truth that the worker is sleeping
    sleeping: bool,

    /// Truth that the thread pool is reachable and may receive more work
    work_incoming: bool,
}
//
impl WorkerFutexState {
    /// Location that this thread is recommended stealing from
    pub(crate) fn location(&self) -> Option<StealLocation> {
        self.steal_location
    }

    /// Truth that the thread pool is reachable and may receive more work
    ///
    /// Once this has become `false`, it can never become `true` again.
    pub(crate) fn work_incoming(&self) -> bool {
        self.work_incoming
    }

    /// Decode the raw state from the futex data
    fn from_raw(raw: RawWorkerFutexState) -> Self {
        let work_incoming = raw & FUTEX_BIT_WORK_INCOMING != 0;
        let sleepy = raw & FUTEX_BIT_SLEEPY != 0;
        let sleeping = raw & FUTEX_BIT_SLEEPING != 0;
        let raw_location = raw & RAW_LOCATION_MASK;
        let steal_location = StealLocation::from_raw(raw_location);
        let result = Self {
            steal_location,
            sleeping,
            sleepy,
            work_incoming,
        };
        result.debug_check_state();
        result
    }

    /// Convert back to raw futex data
    fn to_raw(self) -> RawWorkerFutexState {
        self.debug_check_state();
        let mut raw = StealLocation::to_raw(self.steal_location);
        if self.sleepy {
            raw |= FUTEX_BIT_SLEEPY
        }
        if self.sleeping {
            raw |= FUTEX_BIT_SLEEPING;
        }
        if self.work_incoming {
            raw |= FUTEX_BIT_WORK_INCOMING;
        }
        raw
    }

    /// Check that current futex state makes sense in debug builds
    fn debug_check_state(&self) {
        if self.sleepy {
            debug_assert_eq!(self.steal_location, None);
            debug_assert!(self.work_incoming);
        }
        if self.sleeping {
            debug_assert!(self.sleepy);
        }
    }
}
//
#[cfg(test)]
impl Arbitrary for WorkerFutexState {
    type Parameters = <Option<StealLocation> as Arbitrary>::Parameters;
    type Strategy = Map<
        (
            <Option<StealLocation> as Arbitrary>::Strategy,
            <[bool; 3] as Arbitrary>::Strategy,
        ),
        fn((Option<StealLocation>, [bool; 3])) -> Self,
    >;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        (
            <Option<StealLocation> as Arbitrary>::arbitrary_with(args),
            any::<[bool; 3]>(),
        )
            .prop_map(|(steal_location, [sleepy, sleeping, work_incoming])| Self {
                steal_location,
                sleepy,
                sleeping,
                work_incoming,
            })
    }
}
//
impl Default for WorkerFutexState {
    fn default() -> Self {
        Self {
            steal_location: None,
            sleepy: false,
            sleeping: false,
            work_incoming: true,
        }
    }
}

/// Recommended work-stealing location
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StealLocation {
    /// Steal from this worker thread
    Worker(usize),

    /// Steal from the global injector
    Injector,
}
//
impl StealLocation {
    /// Translate a raw location from the futex to a high-level location
    fn from_raw(raw: RawStealLocation) -> Option<Self> {
        match raw {
            RAW_LOCATION_NONE => None,
            RAW_LOCATION_INJECTOR => Some(StealLocation::Injector),
            worker_idx => Some(StealLocation::Worker(usize::try_from(worker_idx).unwrap())),
        }
    }

    /// Translate a high-level location back into a raw location for the futex
    fn to_raw(opt: Option<Self>) -> RawStealLocation {
        match opt {
            Some(StealLocation::Worker(worker_idx)) => {
                assert!(worker_idx < WorkerFutex::MAX_WORKERS);
                u32::try_from(worker_idx).unwrap()
            }
            Some(StealLocation::Injector) => RAW_LOCATION_INJECTOR,
            None => RAW_LOCATION_NONE,
        }
    }

    /// Truth that this location is closer to a specific worker thread than
    /// another location
    fn is_closer(self, other: StealLocation, worker_idx: usize) -> bool {
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
type RawWorkerFutexState = u32;

/// Futex status bit signaling that the thread pool is still reachable from
/// outside and thus more work might still be coming up
const FUTEX_BIT_WORK_INCOMING: RawWorkerFutexState = 1 << (RawWorkerFutexState::BITS - 1);

/// Futex status bit signaling that a thread is sleeping or going to sleep
const FUTEX_BIT_SLEEPING: RawWorkerFutexState = 1 << (RawWorkerFutexState::BITS - 2);

/// Futex status bit signaling thread intent to go to sleep
const FUTEX_BIT_SLEEPY: RawWorkerFutexState = 1 << (RawWorkerFutexState::BITS - 3);

/// Last futex status bit before start of location word
const FUTEX_BIT_LAST: RawWorkerFutexState = FUTEX_BIT_SLEEPY;

/// Futex-internal encoding of [`StealLocation`]
type RawStealLocation = RawWorkerFutexState;
//
/// Mask that extracts the location word from the futex state
const RAW_LOCATION_MASK: RawWorkerFutexState = FUTEX_BIT_LAST - 1;
//
/// No recommended location at the moment
const RAW_LOCATION_NONE: RawStealLocation = FUTEX_BIT_LAST - 1;
//
/// Steal from the global injector
const RAW_LOCATION_INJECTOR: RawStealLocation = FUTEX_BIT_LAST - 2;
//
/// Number of normal locations
const RAW_LOCATION_NUM_NORMAL: RawStealLocation = RAW_LOCATION_INJECTOR;

#[cfg(test)]
mod tests {
    use super::*;

    // Atomic ordering suitable for load operations
    fn load_order() -> impl Strategy<Value = Ordering> {
        prop::sample::select(&[Ordering::Relaxed, Ordering::Acquire, Ordering::SeqCst][..])
    }

    // Atomic ordering suitable for read-modify-write operations
    fn rmw_order() -> impl Strategy<Value = Ordering> {
        any::<Ordering>()
    }

    proptest! {
        #[test]
        fn new(load_order in load_order()) {
            let futex = WorkerFutex::new();
            assert_eq!(futex.load(load_order), WorkerFutexState::default());
        }

        #[test]
        fn with_state(state: WorkerFutexState, load_order in load_order()) {
            let futex = WorkerFutex::with_state(state);
            assert_eq!(futex.load(load_order), state);
        }

        #[test]
        fn clear_outdated_location_success(
            futex: WorkerFutex,
            load_order in load_order(),
            success_order in rmw_order(),
            failure_order in load_order()
        ) {
            let current = futex.load(load_order);
            if can_clear_outdated_location(current) {
                check_clear_outdated_location_success(futex, load_order, success_order, failure_order);
            }
        }

        #[test]
        fn clear_outdated_location(
            futex: WorkerFutex,
            expected: WorkerFutexState,
            load_order in load_order(),
            success_order in rmw_order(),
            failure_order in load_order()
        ) {
            let current = futex.load(load_order);
            if can_clear_outdated_location(current) {
                if expected == current {
                    check_clear_outdated_location_success(futex, load_order, success_order, failure_order);
                } else {
                    assert_eq!(
                        futex.clear_outdated_location(expected, success_order, failure_order),
                        Err(current)
                    );
                    assert_eq!(futex.load(load_order), current);
                }
            }
        }

        // TODO: Test other methods
    }

    fn can_clear_outdated_location(state: WorkerFutexState) -> bool {
        state.location().is_some() & !(state.sleepy & state.sleeping)
    }

    fn check_clear_outdated_location_success(
        futex: WorkerFutex,
        load_order: Ordering,
        success_order: Ordering,
        failure_order: Ordering,
    ) {
        let current = futex.load(load_order);
        assert!(can_clear_outdated_location(current));
        let expected_out = WorkerFutexState {
            steal_location: None,
            ..current
        };
        assert_eq!(
            futex.clear_outdated_location(current, success_order, failure_order),
            Ok(expected_out)
        );
        assert_eq!(futex.load(load_order), expected_out);
    }
}
