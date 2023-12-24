//! Thread pool job

use crate::{
    worker::scope::Scope, Work, DEFAULT_LOAD_BALANCING_MARGIN, DESIRED_PARALLEL_EFFICIENCY,
};
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
/// - Create a type-erased task with `make_task()` and submit it for execution
///   on the thread pool.
/// - Until the job completion signal is received, do not exit the current stack
///   frame or interact with the Job in any way, including but not limited to...
///     - Moving or dropping the job
///     - Calling any Job method
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
    pub unsafe fn make_task(&mut self, schedule: Schedule) -> Task {
        let state = self.0.get();
        let state = state.cast::<()>();
        let run = |state: *mut (), scope: &Scope<'_>| {
            let state = state.cast::<JobState<Res, ImplWork, ImplNotify>>();
            // SAFETY: Per `Job` API contract
            unsafe { (*state).run(scope) };
        };
        Task {
            state,
            run,
            schedule,
        }
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

/// Type-erased handle to a [`Job`] with scheduling properties
#[derive(Debug, Eq, Hash, PartialEq)]
pub(crate) struct Task {
    /// Type-erased `&mut JobState<...>` pointer
    state: *mut (),

    /// Type-erased `JobState<...>::run()` method
    ///
    /// The first parameter must be `self.job`.
    run: fn(*mut (), &Scope<'_>),

    /// Scheduling properties of this task
    pub schedule: Schedule,
}
//
impl Task {
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
// SAFETY: It is safe to send a Task to another thread because inner Work is
//         safe to send across thread and the Job API contract ensures that the
//         current thread will not touch the Job in any way until the other
//         thread is done with the Task, which means that for all intents and
//         purposes we effectively own the inner Work.
unsafe impl Send for Task {}

/// Scheduling properties of a [`Task`]
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct Schedule {
    /// Remaining number of nested parallel `join()`s before we switch to the
    /// sequential implementation of `join()`
    remaining_par_joins: u8,
}
//
impl Schedule {
    /// Compute the scheduling properties of a task that is new to the thread
    /// pool, as opposed to being the child of an existing thread pool task
    #[allow(clippy::assertions_on_constants)]
    pub fn new(num_workers: usize) -> Self {
        // Here our goal is to compute the optimial join() nesting depth,
        // assuming join()s are balanced i.e. each branch processes a similar
        // workload by running recursive join()s an equal number of time
        let mut desired_join_nesting = 0;
        let max_tasks = |join_nesting| 2usize.pow(join_nesting);

        // First we need enough tasks to provide work to all workers, with a
        // safety margin to account for non-balanced joins, heterogeneous
        // hardware and non-ideal execution conditions.
        assert!(
            DEFAULT_LOAD_BALANCING_MARGIN >= 1.0,
            "load balancing margins are about exposing more concurrency, not less"
        );
        let desired_num_tasks = ((num_workers as f32) * DEFAULT_LOAD_BALANCING_MARGIN).ceil();
        if desired_num_tasks > usize::MAX as f32 {
            return Self {
                remaining_par_joins: u8::MAX,
            };
        }
        while max_tasks(desired_join_nesting) < desired_num_tasks as usize {
            desired_join_nesting += 1;
        }

        // On odd CPU core counts, we additionally need to ensure that the load
        // imbalance created by using a binary tree to distribute work over a
        // non-power-of-two core count is acceptable
        if !num_workers.is_power_of_two() {
            assert!(
                DESIRED_PARALLEL_EFFICIENCY > 0.5,
                "a parallel efficiency smaller than or requal to 0.5 means \
                that join() would stop before using all of your thread pool"
            );
            assert!(
                DESIRED_PARALLEL_EFFICIENCY < 1.0,
                "parallel efficiency is an actual/ideal ratio which can \
                neither reach nor go above 1.0"
            );
            loop {
                let max_tasks = max_tasks(desired_join_nesting);
                let num_balanced_chunks = (max_tasks / num_workers) as f32;
                let num_trailing_tasks = (max_tasks % num_workers) as f32;
                let parallel_efficiency = (num_balanced_chunks
                    + num_trailing_tasks / num_workers as f32)
                    / (num_balanced_chunks + 1.0);
                if parallel_efficiency >= DESIRED_PARALLEL_EFFICIENCY {
                    break;
                } else {
                    desired_join_nesting += 1;
                    continue;
                }
            }
        }
        Self {
            remaining_par_joins: u8::try_from(desired_join_nesting)
                .expect("2^256 tasks ought to be enough for anybody"),
        }
    }

    /// Compute the scheduling properties of a child of the active task
    pub fn child_schedule(self) -> Self {
        Self {
            remaining_par_joins: self.remaining_par_joins.saturating_sub(1),
        }
    }

    /// Truth that this task's inner `join()` statements should be parallelized
    ///
    /// If this is true, the remote end of this task's `join()` statements will
    /// be injected into the thread pool, creating an opportunity for
    /// paralellism and load balancing at the expense of scheduling overhead.
    /// Otherwise, both ends of this task's `join()` statements will be executed
    /// sequentially by the active worker thread.
    pub fn parallelize_join(self) -> bool {
        self.remaining_par_joins != 0
    }
}

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
