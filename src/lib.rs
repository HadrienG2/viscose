#![warn(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)]

pub mod bench;
pub mod flags;
mod job;
pub mod pool;
mod state;
mod worker;

use crate::worker::Scope;
use std::time::Duration;

/// Busy-waiting time between declaring sleepiness and falling asleep
///
/// This is a compromise between avoiding sleep and wakeup latency on one side,
/// and keeping idle threads asleep on the other side.
const SLEEPY_DURATION: Duration = Duration::from_nanos(1);

/// Maximal number of spinning iterations before starting to sleep
const MAX_SPIN_ITERS: usize = 1 << 7;

/// Sleep duration used for busy waiting
///
/// Need a non-zero sleep duration to trigger a scheduler yield on Linux.
const YIELD_DURATION: Duration = Duration::from_nanos(1);

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

/// Aborts if dropped, used to translate panics to aborts
struct AbortGuard;
//
impl Drop for AbortGuard {
    fn drop(&mut self) {
        std::process::abort()
    }
}

/// Extract the result or propagate the panic from a `thread::Result`
fn result_or_panic<R>(result: std::thread::Result<R>) -> R {
    match result {
        Ok(result) => result,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}
