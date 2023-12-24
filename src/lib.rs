#![warn(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::dbg_macro,
    clippy::unimplemented
)]

#[cfg(feature = "bench")]
pub mod bench;
mod pool;
mod shared;
mod worker;

use std::{sync::atomic::Ordering, time::Duration};

// Re-export components that are part of the public interface
pub use crate::{pool::ThreadPool, worker::scope::Scope};

/// Minimal busy-waiting time between declaring sleepiness and falling asleep
///
/// This is a compromise between avoiding sleep and wakeup latency on one side,
/// and keeping idle threads asleep on the other side.
const OS_WAIT_DELAY: Duration = Duration::from_nanos(0);

/// Sleep duration used to yield the CPU to the OS
///
/// Need a non-zero sleep duration to trigger a true scheduler yield on Linux.
const YIELD_DURATION: Duration = Duration::from_nanos(1);

/// Maximal number of spinning iterations between starting to yield to the os
const SPIN_ITERS_BEFORE_YIELD: usize = 1 << 7;

/// Maximal number of spinning iterations between attempts to look for work
const MAX_SPIN_ITERS_PER_CHECK: u8 = 4;

/// Desired parallel execution efficiency on odd-core-count CPUs
///
/// On CPUs whose number of cores is not a power of two, the convenience and
/// scalability of binary fork-join APIs comes at the cost of some structural
/// load imbalance. For example, assuming you have 12 CPU cores, splitting a
/// task 3 times creates 2^3 = 8 subtasks, which is not enough to feed all 12
/// CPU cores. But splitting it 4 times creates 2^4 = 16 subtasks, which after
/// handing each core a subtask leaves 4 remaining subtasks, which again does
/// not distribute evenly across 12 CPU cores.
///
/// Because powers of two are not divisble by any number other than smaller
/// powers of two, and because the only scalable approach (not requiring
/// all-to-all worker communication) to creating more subtasks in binary fork
/// join is to split each previously spawned subtask in two, we cannot never
/// fully eliminate this load imbalance. However, we can make the imbalance
/// arbitrarily low by splitting tasks more and more, at the expense of spending
/// more CPU time scheduling and awaiting subtasks, and reducing the sequential
/// execution efficiency of individual subtasks.
///
/// This tuning parameter should be set between 0.5 and 1.0, exclusive on both
/// bounds. It lets you tune the compromise between the two aforementioned
/// concerns. Tuning it higher encourages the runtime to split tasks more in
/// order to achieve a more balanced workload.
const DESIRED_PARALLEL_EFFICIENCY: f32 = 0.85;

/// Default oversubscription factor
///
/// In an ideal world, running tasks efficiently over 16 CPU cores would only
/// require creating 16 subtasks through 4 recursive task-splitting passes.
/// However, the real world tends to be messier than this: the two arms of a
/// `join()` statement will rarely have a perfectly equal workload, the system's
/// CPU cores may not have equal performance (think big.LITTLE ARM chips like
/// Apple Mx), run-time issues like intermittent background tasks may slow down
/// some workers relative to others, etc.
///
/// To account for this, you may want to split tasks a bit more than you need,
/// so that CPUs which processed their tasks quickly can help other CPUs which
/// are still struggling with their own workload, a form of load balancing.
/// However, splitting tasks is not free, so you should not overdo it more than
/// necessary either. Indeed, it is possible to find pathological workloads for
/// which such oversubscription is never worthwhile.
///
/// The right compromise here is both system- and workload-dependent, so it
/// should ideally be tuned on a per-`join()` and per-target-system basis. But
/// of course this would have terrible ergonomics in typical use cases where
/// trading a little inefficiency for convenience is perfectly fine. Hence we go
/// for a tiered approach:
///
/// - By default, we apply a certain safety margin which is empirically tuned to
///   work well for typical hardware and workloads.
/// - When creating a thread pool, if you know that all your tasks are very
///   (irr-)regular or that your hardware is heterogeneous, you can override the
///   default margin for all tasks spawned in that thread pool.
/// - For ultimate performance, you can fine tune individual `run()` and
///   `join()` statements by applying a multiplicative factor to the pool's
///   default safety margin.
///
/// This is a multiplicative factor applied on top of the system's CPU count to
/// determine the optimal number of tasks, which should be higher than or equal
/// to 1.
const DEFAULT_OVERSUBSCRIPTION: f32 = 1.0;

/// Function that can be scheduled for execution by the thread pool
///
/// The input [`Scope`] allows the scheduled work to interact with the thread
/// pool by e.g. spawning new tasks.
pub trait Work<Res: Send>: for<'scope> FnOnce(&Scope<'scope>) -> Res + Send {}
//
impl<Res, Body> Work<Res> for Body
where
    Res: Send,
    Body: for<'scope> FnOnce(&Scope<'scope>) -> Res + Send,
{
}

/// Extract the result or propagate the panic from a `thread::Result`
#[track_caller]
fn result_or_panic<R>(result: std::thread::Result<R>) -> R {
    match result {
        Ok(result) => result,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// Tell the optimizer that a code path is unlikely to be taken and should be
/// out-lined into a separate function call.
#[cold]
#[inline(never)]
fn unlikely<T>(f: impl FnOnce() -> T) -> T {
    f()
}

/// Add an Acquire barrier to a user-specified atomic operation ordering
#[inline]
fn at_least_acquire(order: Ordering) -> Ordering {
    match order {
        Ordering::Relaxed | Ordering::Acquire => Ordering::Acquire,
        Ordering::Release | Ordering::AcqRel => Ordering::AcqRel,
        Ordering::SeqCst => Ordering::SeqCst,
        _ => unreachable!(),
    }
}

// Set up optional logging
#[doc(hidden)]
#[macro_export]
macro_rules! log {
    ($level:expr, $($args:expr),*) => {
        #[cfg(feature = "log")]
        log::log!($level, $($args),*);
    };
}
#[doc(hidden)]
#[macro_export]
macro_rules! error {
    ($($args:expr),*) => {
        $crate::log!(log::Level::Error, $($args),*);
    };
}
#[doc(hidden)]
#[macro_export]
macro_rules! warn {
    ($($args:expr),*) => {
        $crate::log!(log::Level::Warn, $($args),*);
    };
}
#[doc(hidden)]
#[macro_export]
macro_rules! info {
    ($($args:expr),*) => {
        $crate::log!(log::Level::Info, $($args),*);
    };
}
#[doc(hidden)]
#[macro_export]
macro_rules! debug {
    ($($args:expr),*) => {
        $crate::log!(log::Level::Debug, $($args),*);
    };
}
#[doc(hidden)]
#[macro_export]
macro_rules! trace {
    ($($args:expr),*) => {
        $crate::log!(log::Level::Trace, $($args),*);
    };
}
//
/// Ensure logging to stderr is set up during benchmarking
#[cfg(any(test, feature = "bench"))]
pub(crate) fn setup_logger_once() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        env_logger::init();
    })
}

/// Topology instance shared by all tests and benchmarks
#[cfg(any(test, feature = "bench"))]
pub(crate) fn topology() -> &'static std::sync::Arc<hwlocality::Topology> {
    use hwlocality::Topology;
    use std::sync::{Arc, OnceLock};
    static INSTANCE: OnceLock<Arc<Topology>> = OnceLock::new();
    INSTANCE.get_or_init(|| Arc::new(Topology::new().unwrap()))
}

#[allow(unused_imports)]
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::panic::UnwindSafe;

    pub(crate) fn assert_panics<R>(op: impl FnOnce() -> R + UnwindSafe) {
        assert!(std::panic::catch_unwind(op).is_err());
    }
}
