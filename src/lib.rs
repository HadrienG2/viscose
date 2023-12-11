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
const OS_WAIT_DELAY: Duration = Duration::from_nanos(1000);

/// Sleep duration used to yield the CPU to the OS
///
/// Need a non-zero sleep duration to trigger a true scheduler yield on Linux.
const YIELD_DURATION: Duration = Duration::from_nanos(1);

/// Maximal number of spinning iterations between starting to yield to the os
const SPIN_ITERS_BEFORE_YIELD: usize = 1 << 7;

/// Maximal number of spinning iterations between attempts to look for work
const MAX_SPIN_ITERS_PER_CHECK: u8 = 8;

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
