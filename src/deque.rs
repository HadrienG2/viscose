//! Per-worker work queue
//!
//! While it has served this project well for a long time, `crossbeam_deque` has
//! two important limitations that become increasingly problematic over time:
//!
//! - Its design and implementation does not seem to be optimized for the right
//!   workloads. For example, it is not very important to have unbounded work
//!   queues in a binary fork-join world where you can compress 2^64 work items
//!   into 64 splittable tasks, yet the performance price we pay for this
//!   growability (an `MFENCE` on the hot code path just in case reallocations
//!   might occur someday) can be quite expensive on join-heavy workloads.
//! - Work can never be given to a worker, only stolen from it. This makes
//!   perfect sense if you live in the convenient world of symmetric
//!   multiprocessing, but as soon as hardware locality concerns come into play,
//!   it is good to have a way to explicitly spread the workload of newly
//!   spawned jobs to precise locations of the hardware topology, and this in
//!   turn is best done by having full doubled-ended work queues that allow for
//!   remote pushes.
//!
//! Of course, nothing is free in the world of concurrent programming, and
//! solving these two problems will have unavoidable consequences elsewhere...
//!
//! - By doing away with work queue growability and the `MFENCE` that comes with
//!   it, we now need to decide what happens when a work queue fills up. Since
//!   this can only happen with pathological workloads that could use a little
//!   performance backpressure on task spawning to avoid running out of RAM,
//!   spilling to the slow global injector seems like a fine response here.
//! - A deque where N threads can push and pop work on one side cannot have both
//!   a simple and efficient ring buffer data layout and lock-freedom for all
//!   transactions. That's because if multiple threads can be concurrently
//!   pushing work on one side of the ring, and another thread can concurrently
//!   come around and pop a task from that same side of the ring, we will end up
//!   with "holes" in the ring buffer... We choose to handle this by sacrificing
//!   lock-freedom on the "outside world" side of the queue, which should not be
//!   under high pressure when executing non-pathological workloads.

// TODO: Implement
