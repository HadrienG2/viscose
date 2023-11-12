//! Thread pool job

use crate::{worker::scope::Scope, Work};
use std::{cell::UnsafeCell, panic::AssertUnwindSafe};

/// Aborts the process if dropped
///
/// Create one of these at the start of code sections where unwinding panics
/// must not be allowed to escape the current stack frame, and `mem::forget()`
/// it at the end of the danger zone.
pub(crate) struct AbortOnUnwind;
//
impl Drop for AbortOnUnwind {
    fn drop(&mut self) {
        std::process::abort()
    }
}

/// [`Work`] that has been prepared for execution by the thread pool
///
/// # Safety
///
/// Safe use of [`Job`] requires carefully following the following procedure:
///
/// - Create a [`Job`] on the stack frame where it will be executed.
/// - Create a type-erased job handle with `as_dyn()` and submit it for
///   execution on the thread pool.
/// - Until the job completion signal is received, do not exit the current stack
///   frame or interact with the Job in any way, including but not limited to...
///     - Moving or dropping the job
///     - Calling any Job or DynJob method
///     - Letting a panic unwind the stack (use [`AbortOnUnwind`] for this)
/// - Once a job completion signal has been received with Acquire memory
///   ordering, you may extract the result and propagate panics with
///   `result_or_panic()`.
pub(crate) struct Job<Res: Send, ImplWork: Work<Res>, ImplNotify: Notify>(
    UnsafeCell<JobState<Res, ImplWork, ImplNotify>>,
);
//
impl<Res: Send, ImplWork: Work<Res>, ImplNotify: Notify> Job<Res, ImplWork, ImplNotify> {
    /// Prepare [`Work`] for execution by the thread pool
    pub fn new(notify: ImplNotify, work: ImplWork) -> Self {
        Self(UnsafeCell::new(JobState::Scheduled(notify, work)))
    }

    /// Create a type-erased handle that can be pushed on a work queue
    ///
    /// # Safety
    ///
    /// Should only be called once, as preparation for submitting the job to the
    /// thread pool.
    pub unsafe fn as_dyn(&mut self) -> DynJob {
        let state = self.0.get();
        let state = state.cast::<()>();
        let run = |state: *mut (), scope: &Scope<'_>| {
            let state = state.cast::<JobState<Res, ImplWork, ImplNotify>>();
            // SAFETY: Per `Job` API contract
            unsafe { (*state).run(scope) };
        };
        DynJob { state, run }
    }

    /// Extract the job result or propagate job panic
    ///
    /// # Safety
    ///
    /// Should only be called after the job completion notification has been
    /// received.
    #[track_caller]
    pub unsafe fn result_or_panic(mut self) -> Res {
        match std::mem::replace(self.0.get_mut(), JobState::Collected) {
            JobState::Scheduled(_, _) | JobState::Running => {
                panic!("Job result shouldn't be collected before completion notification")
            }
            JobState::Finished(result) => crate::result_or_panic(result),
            JobState::Collected => unreachable!("prevented by consuming self"),
        }
    }
}

/// [`Job`] state machine
#[derive(Debug)]
enum JobState<Res: Send, ImplWork: Work<Res>, ImplNotify: Notify> {
    /// Job has not started executing yet
    Scheduled(ImplNotify, ImplWork),

    /// Job is executing on some worker thread
    Running,

    /// Job execution is complete and result can be collected
    Finished(std::thread::Result<Res>),

    /// Job result has been collected
    Collected,
}
//
impl<Res: Send, ImplWork: Work<Res>, ImplNotify: Notify> JobState<Res, ImplWork, ImplNotify> {
    /// Run the job
    ///
    /// # Safety
    ///
    /// This should only be called once, on a job fresh from the work queue.
    unsafe fn run(&mut self, scope: &Scope<'_>) {
        let (notify, work) = match std::mem::replace(self, Self::Running) {
            Self::Scheduled(notify, work) => (notify, work),
            other => {
                if cfg!(debug_assertions) {
                    panic!("attempted to execute a Job in an invalid state");
                } else {
                    std::mem::forget(other);
                    // SAFETY: Initial job state is Scheduled and Job contract
                    //         ensures we'll only see it in that state.
                    unsafe { std::hint::unreachable_unchecked() }
                }
            }
        };
        *self = Self::Finished(std::panic::catch_unwind(AssertUnwindSafe(|| work(scope))));
        notify.notify()
    }
}

/// Type-erased handle to a [`Job`]
#[derive(Debug, Eq, Hash, PartialEq)]
pub(crate) struct DynJob {
    /// Type-erased `&mut JobState<...>` pointer
    state: *mut (),

    /// Type-erased `JobState<...>::run()` method
    ///
    /// The first parameter must be `self.job`.
    run: fn(*mut (), &Scope<'_>),
}
//
impl DynJob {
    /// Execute the job
    ///
    /// # Safety
    ///
    /// See top-level [`Job`] documentation.
    pub unsafe fn run(self, scope: &Scope<'_>) {
        (self.run)(self.state, scope)
    }
}
//
// SAFETY: It is safe to send a DynJob to another thread because inner Work is
//         safe to send across thread and the Job API contract ensures that the
//         current thread will not touch the Job in any way until the other
//         thread is done with the DynJob, which means that for all intents and
//         purposes we effectively own the inner Work.
unsafe impl Send for DynJob {}

/// Mechanism to notify the program that a job is done executing
///
/// # Safety
///
/// No notification should be received until `self.notify()` is called, and the
/// notification process must feature an `Ordering::Release` memory barrier at
/// the point where work completion becomes observable.
pub(crate) unsafe trait Notify {
    /// Send in the notification
    fn notify(self);
}
