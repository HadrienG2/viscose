//! Futex used for blocking worker synchronization

use std::{
    debug_assert,
    sync::atomic::{AtomicU32, Ordering},
};

/// Futex used for blocking worker synchronization
///
/// This tracks the following information:
///
/// - A preferred steal location (index of another worker, RAW_LOCATION_INJECTOR
///   to steal from the global task injector, or RAW_LOCATION_NONE if there is
///   no preferred steal location at the moment).
/// - Truth that the thread pool is shutting down (this tells the worker that it
///   should finish draining remaining work from all sources and then exit).
/// - Truth that the worker is asleep due to lack of work (in which case threads
///   which submit new info to the worker using this futex should wake it up
///   with atomic_wait::wake_all).
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

    // --- Futex owner interface ---

    /// Read out the current futex state
    pub fn load(&self, order: Ordering) -> WorkerFutexState {
        let result = WorkerFutexState::from_raw(self.0.load(order));
        debug_assert!(!result.sleeping);
        result
    }

    /// Clear steal location after it becomes outdated, detect races with
    /// concurrent futex updates and return new state in all cases
    pub fn clear_outdated_location(
        &self,
        current: WorkerFutexState,
        success: Ordering,
        failure: Ordering,
    ) -> Result<WorkerFutexState, WorkerFutexState> {
        debug_assert!(current.steal_location.is_some() && !(current.sleepy | current.sleeping));
        let new = WorkerFutexState {
            steal_location: None,
            ..current
        };
        self.0
            .compare_exchange(current.to_raw(), new.to_raw(), success, failure)
            .map(|_| new)
            .map_err(WorkerFutexState::from_raw)
    }

    /// Notify other threads that we are going to sleep soon if nothing happens
    ///
    /// Other threads can cancel this by clearing the SLEEPY flag or altering
    /// the futex state in any other fashion.
    pub fn notify_wait(
        &self,
        current: WorkerFutexState,
        success: Ordering,
        failure: Ordering,
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
            .compare_exchange(current.to_raw(), new.to_raw(), success, failure)
            .map(|_| new)
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
        location: StealLocation,
        worker_idx: usize,
        load: Ordering,
        update: Ordering,
    ) -> bool {
        // Need Acquire ordering on success so that the updating CAS cannot be
        // reordered after the futex wakeup operation.
        let update = Self::at_least_acquire(update);

        // Check ald futex state
        let mut old_raw = self.0.load(load);
        'compare_exchange: loop {
            let old = WorkerFutexState::from_raw(old_raw);

            // Only update if new location is better than old one
            let should_update = match old.steal_location {
                Some(old_location) => location.is_closer(old_location, worker_idx),
                None => true,
            };
            if !should_update {
                return false;
            }

            // Compute updated futex state
            let new = WorkerFutexState {
                steal_location: Some(location),
                ..old
            };
            let new_raw = new.to_raw();

            // Commit futex change via CAS loop
            match self.0.compare_exchange_weak(old_raw, new_raw, load, update) {
                Ok(_) => {
                    if old.sleeping {
                        atomic_wait::wake_all(&self.0);
                    }
                    return true;
                }
                Err(updated_raw) => {
                    old_raw = updated_raw;
                    continue 'compare_exchange;
                }
            }
        }
    }

    /// Wake up this worker thread if it's asleep
    pub fn wake(&self, order: Ordering) {
        // Cancel any impeding attempt to sleep
        //
        // Need Acquire ordering so this is not reordered after waking the futex
        let old_raw = self
            .0
            .fetch_and(!FUTEX_BIT_SLEEPY, Self::at_least_acquire(order));
        if old_raw & FUTEX_BIT_SLEEPING != 0 {
            // If the thread was already sleeping, wake it up
            debug_assert_ne!(old_raw & FUTEX_BIT_SLEEPY, 0);
            atomic_wait::wake_all(&self.0)
        }
    }

    // --- Thread pool interface ---

    /// Notify the worker that the thread pool is shutting down
    pub fn notify_shutdown(&self, order: Ordering) {
        // Cancel any impeding attempt to sleep
        //
        // Need Acquire ordering so this is not reordered after waking the futex
        let old_raw = self
            .0
            .fetch_or(FUTEX_BIT_SHUTTING_DOWN, Self::at_least_acquire(order));
        debug_assert_eq!(old_raw & FUTEX_BIT_SHUTTING_DOWN, 0);
        if old_raw & FUTEX_BIT_SLEEPING != 0 {
            // If the thread was sleeping, wake it up
            debug_assert_ne!(old_raw & FUTEX_BIT_SLEEPY, 0);
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

/// Current worker futex state
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
        debug_assert!(sleepy || !sleeping);
        let raw_location = raw & (FUTEX_BIT_LAST - 1);
        let steal_location = StealLocation::from_raw(raw_location);
        Self {
            steal_location,
            sleeping,
            sleepy,
            shutting_down,
        }
    }

    /// Convert back to raw futex data
    fn to_raw(self) -> FutexData {
        let mut raw = StealLocation::to_raw(self.steal_location);
        if self.sleepy {
            raw |= FUTEX_BIT_SLEEPY
        }
        if self.sleeping {
            debug_assert!(self.sleepy);
            raw |= FUTEX_BIT_SLEEPING;
        }
        if self.shutting_down {
            raw |= FUTEX_BIT_SHUTTING_DOWN;
        }
        raw
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

// TODO: Add tests
