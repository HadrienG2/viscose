//! Futex used for blocking worker synchronization

#[cfg(test)]
use proptest::{
    prelude::*,
    strategy::{Map, TupleUnion},
};
use std::{
    debug_assert, debug_assert_eq, debug_assert_ne,
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
/// - Truth that the worker is asleep due to lack of work (in which case threads
///   which submit new info to the worker using this futex should wake it up
///   with atomic_wait::wake_all), or that it is going to fall asleep ("sleepy")
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

    // --- Futex owner interface ---

    /// Read out the current futex state
    ///
    /// This operation accepts all valid atomic orderings, not just load
    /// operation orderings, but store orderings will be less efficient.
    pub fn load_from_worker(&self, order: Ordering) -> WorkerFutexState {
        let result = self.load(order);
        debug_assert!(!result.sleeping);
        result
    }

    /// Load without state validity checks
    ///
    /// This operation accepts all valid atomic orderings, not just load
    /// operation orderings, but store orderings will be less efficient.
    fn load(&self, order: Ordering) -> WorkerFutexState {
        if [Ordering::Relaxed, Ordering::Acquire, Ordering::SeqCst].contains(&order) {
            WorkerFutexState::from_raw(self.0.load(order))
        } else {
            WorkerFutexState::from_raw(self.0.fetch_add(0, order))
        }
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
        let mut current = sleeping;
        while current == sleeping {
            atomic_wait::wait(&self.0, current.to_raw());
            current = self.load(Ordering::AcqRel);
        }

        // Apply user-requested readout ordering if stronger than Acquire
        if ![Ordering::Relaxed, Ordering::Acquire].contains(&wake) {
            atomic::fence(wake);
        }

        // By the time we wake up, the thread that awakened us will have cleared
        // the sleepy and sleeping flags, so we can just return the new state
        debug_assert!(!(current.sleepy | current.sleeping));
        current
    }

    // --- Other worker interface ---

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
            debug_assert!(current.sleepy | !current.sleeping);

            // Abort if the proposed location isn't better than current one
            let should_update = match current.steal_location {
                Some(current_location) => proposed_location.is_closer(current_location, worker_idx),
                None => true,
            };
            if !should_update {
                debug_assert!(current.steal_location.is_some() & !current.sleepy);
                return Err(current);
            }

            // Try to update the steal location with our proposal
            let new = WorkerFutexState {
                steal_location: Some(proposed_location),
                sleepy: false,
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
            debug_assert_ne!(old_raw & (FUTEX_BIT_WORK_INCOMING | FUTEX_BIT_SLEEPY), 0);
            atomic_wait::wake_all(&self.0)
        }
    }

    /// Add an Acquire barrier to a user-specified ordering
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
    /// Location from which this worker is recommended to steal
    steal_location: Option<StealLocation>,

    /// Truth that the thread pool is reachable and may receive more work
    work_incoming: bool,

    /// Truth that the worker announced intent to go to sleep
    sleepy: bool,

    /// Truth that the worker is sleeping
    sleeping: bool,
}
//
impl WorkerFutexState {
    /// Initial worker futex state
    const INITIAL: WorkerFutexState = WorkerFutexState {
        steal_location: None,
        work_incoming: true,
        sleepy: false,
        sleeping: false,
    };

    /// Location that this thread is recommended stealing from
    pub(crate) const fn steal_location(&self) -> Option<StealLocation> {
        self.steal_location
    }

    /// Truth that the thread pool is reachable and may receive more work
    ///
    /// Once this has become `false`, it can never become `true` again.
    pub(crate) const fn work_incoming(&self) -> bool {
        self.work_incoming
    }

    /// Truth that the worker has declared itself sleepy with
    /// [`WorkerFutex::notify_sleepy()`] and no one has taken it out of the
    /// sleepy state by updating the futex since.
    pub(crate) const fn is_sleepy(&self) -> bool {
        self.sleepy
    }

    /// Decode the raw state from the futex data
    const fn from_raw(raw: RawWorkerFutexState) -> Self {
        let raw_location = raw & RAW_LOCATION_MASK;
        let steal_location = StealLocation::from_raw(raw_location);
        let work_incoming = raw & FUTEX_BIT_WORK_INCOMING != 0;
        let sleepy = raw & FUTEX_BIT_SLEEPY != 0;
        let sleeping = raw & FUTEX_BIT_SLEEPING != 0;
        let result = Self {
            steal_location,
            work_incoming,
            sleeping,
            sleepy,
        };
        result.debug_check_state();
        result
    }

    /// Convert back to raw futex data
    const fn to_raw(self) -> RawWorkerFutexState {
        self.debug_check_state();
        let mut raw = StealLocation::to_raw(self.steal_location);
        if self.work_incoming {
            raw |= FUTEX_BIT_WORK_INCOMING;
        }
        if self.sleepy {
            raw |= FUTEX_BIT_SLEEPY
        }
        if self.sleeping {
            raw |= FUTEX_BIT_SLEEPING;
        }
        raw
    }

    /// Check that current futex state makes sense in debug builds
    const fn debug_check_state(&self) {
        if self.sleepy {
            debug_assert!(self.steal_location.is_none());
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
    type Strategy = TupleUnion<(
        (
            u32,
            Arc<
                Map<
                    (
                        <Option<StealLocation> as Arbitrary>::Strategy,
                        <bool as Arbitrary>::Strategy,
                    ),
                    fn((Option<StealLocation>, bool)) -> Self,
                >,
            >,
        ),
        (
            u32,
            Arc<Map<<bool as Arbitrary>::Strategy, fn(bool) -> Self>>,
        ),
    )>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        prop_oneof![
            // Non-sleepy state may have arbitrary recommended stealing
            // location, no more work incoming
            4 => (<Option<StealLocation> as Arbitrary>::arbitrary_with(args), any::<bool>())
                .prop_map(|(steal_location, work_incoming)| Self {
                    steal_location,
                    work_incoming,
                    sleepy: false,
                    sleeping: false,
                }),
            // Worker may only become sleepy when there is no recommended
            // stealing location and work might still be incoming, and worker
            // may only start to sleep after becoming sleepy
            1 => any::<bool>().prop_map(|sleeping| Self {
                steal_location: None,
                work_incoming: true,
                sleepy: true,
                sleeping,
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
    use std::{
        fmt::Debug,
        sync::{atomic::AtomicBool, Barrier},
        time::Duration,
    };

    /// Atomic ordering suitable for load operations
    fn load_order() -> impl Strategy<Value = Ordering> {
        prop::sample::select(&[Ordering::Relaxed, Ordering::Acquire, Ordering::SeqCst][..])
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

    /// Atomic ordering suitable for read-modify-write operations
    fn rmw_order() -> impl Strategy<Value = Ordering> {
        any::<Ordering>()
    }

    /// Futex state where steal_location is always set
    fn state_with_location() -> impl Strategy<Value = WorkerFutexState> {
        any::<(StealLocation, bool)>().prop_map(|(steal_location, work_incoming)| {
            WorkerFutexState {
                steal_location: Some(steal_location),
                work_incoming,
                sleepy: false,
                sleeping: false,
            }
        })
    }

    /// Check that clear_outdated_location does clear the steal_location,
    /// assuming there is one steal_location set initially.
    fn check_clear_location_success(
        futex: WorkerFutex,
        update: Ordering,
        load: Ordering,
    ) -> Result<(), TestCaseError> {
        let current = futex.load_from_worker(Ordering::Relaxed);
        prop_assert!(current.steal_location.is_some() && !(current.sleepy | current.sleeping));
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

    /// Only valid futex state from which sleepy flag can be set
    const FUTURE_SLEEPY_STATE: WorkerFutexState = WorkerFutexState {
        steal_location: None,
        work_incoming: true,
        sleepy: false,
        sleeping: false,
    };

    /// Only valid futex state with sleepy flag set
    const SLEEPY_STATE: WorkerFutexState = WorkerFutexState {
        steal_location: None,
        work_incoming: true,
        sleepy: true,
        sleeping: false,
    };

    /// Futex state where work_incoming is false
    fn done_state() -> impl Strategy<Value = WorkerFutexState> {
        any::<Option<StealLocation>>().prop_map(|steal_location| WorkerFutexState {
            steal_location,
            work_incoming: false,
            sleepy: false,
            sleeping: false,
        })
    }

    /// Futex state that can't transition to the sleepy state
    fn sleepless_state() -> impl Strategy<Value = WorkerFutexState> {
        prop_oneof![
            4 => state_with_location(),
            1 => done_state(),
        ]
    }

    proptest! {
        #[test]
        fn sleepy_success(
            update in rmw_order(),
            load in load_order(),
        ) {
            let futex = WorkerFutex::with_state(FUTURE_SLEEPY_STATE);
            prop_assert_eq!(futex.notify_sleepy(FUTURE_SLEEPY_STATE, update, load), Ok(SLEEPY_STATE));
            prop_assert_eq!(futex.load_from_worker(Ordering::Relaxed), SLEEPY_STATE);
        }

        #[test]
        fn sleepy_failure(
            futex in futex_with_state(sleepless_state()),
            update in rmw_order(),
            load in load_order(),
        ) {
            let current = futex.load_from_worker(Ordering::Relaxed);
            prop_assert_eq!(
                futex.notify_sleepy(FUTURE_SLEEPY_STATE, update, load),
                Err(current)
            );
            prop_assert_eq!(futex.load_from_worker(Ordering::Relaxed), current);
        }
    }

    /// Arbitrary non-sleeping and non-sleepy state
    fn wakeup_state() -> impl Strategy<Value = WorkerFutexState> {
        prop_oneof![
            2 => state_with_location(),
            1 => done_state(),
            1 => Just(FUTURE_SLEEPY_STATE),
        ]
    }

    /// Only currently allowed sleeping futex state
    const SLEEPING_STATE: WorkerFutexState = WorkerFutexState {
        steal_location: None,
        work_incoming: true,
        sleepy: true,
        sleeping: true,
    };

    /// Reasonable delay to wait for threads to go to sleep
    const WAIT_FOR_SLEEP: Duration = Duration::from_millis(100);

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 5, .. ProptestConfig::default()
        })]
        #[test]
        fn wait_for_change(
            final_state in wakeup_state(),
            sleep in rmw_order(),
            wake in load_order(),
        ) {
            let futex = WorkerFutex::with_state(SLEEPY_STATE);
            let done_waiting = AtomicBool::new(false);
            let barrier = Barrier::new(2);
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    let new_state = futex.wait_for_change(SLEEPY_STATE, sleep, wake);
                    done_waiting.store(true, Ordering::Release);
                    barrier.wait();
                    assert_eq!(new_state, final_state);
                    assert_eq!(futex.load_from_worker(Ordering::Relaxed), final_state);
                });

                std::thread::sleep(WAIT_FOR_SLEEP);
                assert!(!done_waiting.load(Ordering::Relaxed));
                assert_eq!(futex.load(Ordering::Relaxed), SLEEPING_STATE);

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
                assert!(!(expected.sleepy | expected.sleeping));
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
                sleepy: false,
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
        fn wake(
            futex: WorkerFutex,
            order in rmw_order(),
        ) {
            test_waking_op(
                futex,
                |futex| futex.wake(order),
                (),
                |initial| WorkerFutexState {
                    sleepy: false,
                    sleeping: false,
                    ..initial
                }
            )?;
        }
    }

    /// Futex state where work_incoming is always true
    fn live_pool_state() -> impl Strategy<Value = WorkerFutexState> {
        prop_oneof![
            4 => any::<Option<StealLocation>>()
                .prop_map(|steal_location| WorkerFutexState {
                    steal_location,
                    work_incoming: true,
                    sleepy: false,
                    sleeping: false,
                }),
            1 => any::<bool>().prop_map(|sleeping| WorkerFutexState {
                steal_location: None,
                work_incoming: true,
                sleepy: true,
                sleeping,
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
                    sleepy: false,
                    sleeping: false,
                    ..initial
                }
            )?;
        }
    }
}
