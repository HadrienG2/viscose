#![warn(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)]

pub mod bench;
pub mod pool;
mod shared;
pub mod worker;

use std::time::Duration;

/// Token used by tasks executing on the thread pool to interact with it
pub use crate::worker::scope::Scope;

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
