//! Futex used for blocking worker synchronization

#[cfg(test)]
use proptest::{
    prelude::*,
    strategy::{Flatten, Map, TupleUnion},
};
use std::{
    debug_assert,
    sync::atomic::{AtomicU32, Ordering},
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
        Self(AtomicU32::new(RAW_LOCATION_NONE))
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

    /// Clear steal location after it becomes outdated, detect races with
    /// concurrent futex updates and return updated state in any case
    pub fn clear_outdated_location(
        &self,
        initial: WorkerFutexState,
        update: Ordering,
        load: Ordering,
    ) -> Result<WorkerFutexState, WorkerFutexState> {
        // Try replacing the futex value until we either succeed or observe a
        // different steal location, which makes this operation obsolete
        let mut current = initial;
        'compare_exchange: loop {
            debug_assert!(current.steal_location.is_some() && !(current.sleepy | current.sleeping));
            let new = WorkerFutexState {
                steal_location: None,
                ..current
            };
            match self
                .0
                .compare_exchange_weak(current.to_raw(), new.to_raw(), update, load)
            {
                Ok(_current_raw) => return Ok(new),
                Err(current_raw) => {
                    current = WorkerFutexState::from_raw(current_raw);
                    if current.steal_location == initial.steal_location {
                        continue 'compare_exchange;
                    } else {
                        return Err(current);
                    }
                }
            }
        }
    }

    /// Notify other threads that we are going to sleep soon if nothing happens
    /// (activity being detected by change in futex state)
    ///
    /// Other threads can cancel this process by clearing the SLEEPY flag or
    /// altering the futex state in any other fashion.
    pub fn notify_sleepy(
        &self,
        current: WorkerFutexState,
        update: Ordering,
        load: Ordering,
    ) -> Result<WorkerFutexState, WorkerFutexState> {
        debug_assert!(
            current.steal_location.is_none()
                && !(current.sleepy || current.sleeping || current.shutting_down)
        );
        let new = WorkerFutexState {
            sleepy: true,
            ..current
        };
        self.0
            .compare_exchange(current.to_raw(), new.to_raw(), update, load)
            .map(|_current_raw| new)
            .map_err(WorkerFutexState::from_raw)
    }

    /// Wait for a futex notification or a futex state change, return new state
    ///
    /// Should have announced intent to sleep with prepare_wait() beforehand,
    /// and ideally waited a bit before calling this method.
    #[inline]
    pub fn wait(
        &self,
        mut current: WorkerFutexState,
        sleep: Ordering,
        wake: Ordering,
    ) -> WorkerFutexState {
        debug_assert!(
            current.steal_location.is_none()
                && current.sleepy
                && !(current.sleeping || current.shutting_down)
        );
        // Need Acquire ordering so this is not reordered after falling asleep
        // FIXME: Don't unconditionally fetch_or, use compare_exchange here
        let actual_raw = self
            .0
            .fetch_or(FUTEX_BIT_SLEEPING, Self::at_least_acquire(sleep));
        if actual_raw == current.to_raw() {
            current.sleeping = true;
            atomic_wait::wait(&self.0, current.to_raw());
        }
        // Need Release ordering so this is not reordered before falling asleep
        WorkerFutexState::from_raw(self.0.fetch_and(
            !(FUTEX_BIT_SLEEPY | FUTEX_BIT_SLEEPING),
            Self::at_least_release(wake),
        ))
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
        // Need Acquire ordering on success so that the updating CAS cannot be
        // reordered after the futex wakeup operation.
        let update = Self::at_least_acquire(update);

        // Update stealing location if our proposal is better
        let result = self.0.fetch_update(update, load, |current_raw| {
            let current = WorkerFutexState::from_raw(current_raw);

            // Only update if proposed location is better than current one
            let should_update = match current.steal_location {
                Some(current_location) => proposed_location.is_closer(current_location, worker_idx),
                None => true,
            };
            if !should_update {
                return None;
            }

            // Compute updated futex state
            let new = WorkerFutexState {
                steal_location: Some(proposed_location),
                sleepy: false,
                sleeping: false,
                ..current
            };
            Some(new.to_raw())
        });

        // If we updated the futex of a sleepy thread, wake it up
        if let Ok(previous_raw) = result {
            if previous_raw & FUTEX_BIT_SLEEPING != 0 {
                atomic_wait::wake_all(&self.0)
            }
        }

        // Tell whether the recommendation was accepted
        result.is_ok()
    }

    /// Wake up this worker thread if it's asleep, without a particular motive
    ///
    /// Used to signal that the remote end of a join() has been processed.
    pub fn wake(&self, order: Ordering) {
        // Cancel any impeding attempt to sleep
        //
        // Need Acquire ordering so this is not reordered after waking the futex
        let old_raw = self.0.fetch_and(
            !(FUTEX_BIT_SLEEPY | FUTEX_BIT_SLEEPING),
            Self::at_least_acquire(order),
        );
        if old_raw & FUTEX_BIT_SLEEPING != 0 {
            // If the thread might be sleeping, wake it up
            atomic_wait::wake_all(&self.0)
        }
    }

    // --- Thread pool interface ---

    /// Notify the worker that the thread pool is shutting down
    pub fn notify_shutdown(&self, order: Ordering) {
        // Cancel any impeding attempt to sleep
        //
        // Need Acquire ordering so this is not reordered after waking the futex
        // FIXME: Use compare_exchange loop to also clear sleepy and sleeping
        //        bits
        let old_raw = self
            .0
            .fetch_or(FUTEX_BIT_SHUTTING_DOWN, Self::at_least_acquire(order));
        debug_assert_eq!(old_raw & FUTEX_BIT_SHUTTING_DOWN, 0);
        if old_raw & FUTEX_BIT_SLEEPING != 0 {
            // If the thread was sleeping, wake it up
            atomic_wait::wake_all(&self.0)
        }
    }

    // --- Internal utilities ---

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

    /// Add a Release barrier to a user-specified ordering
    #[inline]
    fn at_least_release(order: Ordering) -> Ordering {
        match order {
            Ordering::Relaxed | Ordering::Release => Ordering::Release,
            Ordering::Acquire | Ordering::AcqRel => Ordering::AcqRel,
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
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct WorkerFutexState {
    /// Location from which this worker is recommended to steal
    steal_location: Option<StealLocation>,

    /// Truth that the worker announced intent to go to sleep
    sleepy: bool,

    /// Truth that the worker is sleeping
    sleeping: bool,

    /// Truth that the thread pool is shutting down
    shutting_down: bool,
}
//
impl WorkerFutexState {
    /// Location that this thread is recommended stealing from
    pub(crate) fn location(&self) -> Option<StealLocation> {
        self.steal_location
    }

    /// Truth that the thread pool is shutting down
    ///
    /// Once this has become `true`, it can never become `false` again.
    pub(crate) fn shutting_down(&self) -> bool {
        self.shutting_down
    }

    /// Decode the raw state from the futex data
    fn from_raw(raw: FutexData) -> Self {
        let shutting_down = raw & FUTEX_BIT_SHUTTING_DOWN != 0;
        let sleepy = raw & FUTEX_BIT_SLEEPY != 0;
        let sleeping = raw & FUTEX_BIT_SLEEPING != 0;
        let raw_location = raw & (FUTEX_BIT_LAST - 1);
        let steal_location = StealLocation::from_raw(raw_location);
        let result = Self {
            steal_location,
            sleeping,
            sleepy,
            shutting_down,
        };
        result
    }

    /// Convert back to raw futex data
    fn to_raw(self) -> FutexData {
        let mut raw = StealLocation::to_raw(self.steal_location);
        if self.sleepy {
            raw |= FUTEX_BIT_SLEEPY
        }
        if self.sleeping {
            raw |= FUTEX_BIT_SLEEPING;
        }
        if self.shutting_down {
            raw |= FUTEX_BIT_SHUTTING_DOWN;
        }
        raw
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
            .prop_map(|(steal_location, [sleepy, sleeping, shutting_down])| Self {
                steal_location,
                sleepy,
                sleeping,
                shutting_down,
            })
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
type FutexData = u32;

/// Futex status bit signaling thread pool shutdown
const FUTEX_BIT_SHUTTING_DOWN: FutexData = 1 << (FutexData::BITS - 1);

/// Futex status bit signaling thread sleep
const FUTEX_BIT_SLEEPING: FutexData = 1 << (FutexData::BITS - 2);

/// Futex status bit signaling thread intent to go to sleep
const FUTEX_BIT_SLEEPY: FutexData = 1 << (FutexData::BITS - 3);

/// Last futex status bit before start of location word
const FUTEX_BIT_LAST: FutexData = FUTEX_BIT_SLEEPY;

/// Futex-internal encoding of [`StealLocation`]
type RawStealLocation = FutexData;
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
        state.location().is_some() && !(state.sleepy && state.sleeping)
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
