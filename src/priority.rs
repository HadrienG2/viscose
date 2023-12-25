//! Priorization of work distribution and load balancing
//!
//! When new work is spawned by one worker thread and at least two other worker
//! threads are ready to take it, the task scheduling logic must decide which
//! other thread should preferably have it.
//!
//! This choice is a compromise that should strike the right,
//! application-dependent balance between several hardware concerns:
//!
//! 1. Communication is most efficient when threads share as many CPU resources
//!    as possible, e.g. on x86 hyperthreads within the same core communicate as
//!    efficiently as possible, then cores sharing an L3 cache shard communicate
//!    less efficiently, then cores in different L3 shards but the same package
//!    communicate even less efficiently... So all other things being equal, the
//!    interconnect latency and bandwidth cost of starting, executing and
//!    synchronizing with a subtask will be smallest when the target thread is
//!    as close a neighbor in the hardware topology tree as possible.
//! 2. Hyperthreads share a lot of core execution resources, therefore executing
//!    two subtasks of the same job on two hyperthreads of the same CPU core
//!    will only be effective if the tasks' performanec critical inner loops
//!    makes very inefficient or complex use of the core's execution resources.
//!    Otherwise, the two hyperthreads will end up competing for the same core
//!    execution resources, and hyperthreading will at best provide little
//!    performance improvements and at worst degrade performance. Therefore, if
//!    tasks have simple and well-tuned inner loops, which is the optimal
//!    performance scenario, hyperthreads should ideally execute unrelated jobs,
//!    and thus spreading the workload of a single job to multiple hyperthreads
//!    of the same CPU core should be bottom priority for the scheduler.
//! 3. On systems with multiple NUMA nodes, the full system DRAM bandwidth may
//!    only be leveraged by making use of CPU cores from all NUMA nodes. So if
//!    tasks are bound by DRAM bandwidth, which is common when the code has not
//!    been tuned for performance or performs computationally simple work with a
//!    large working set, low-concurrency jobs should violate rule 1 by
//!    priorizing the spreading of work across NUMA nodes, not CPU cores in the
//!    same L3 cache shards.
//!
//! For this thread pool with a work-stealing fork-join design, correctly
//! applying rule 1 requires a surprisingly subtle strategy :
//!
//! - When the workload is being initially recursively split in halves, we
//!   should priorize the spreading of halves to CPUs that start out as remote
//!   as we want to go in the hardware topology, then get increasingly closer,
//!   for the following reasons:
//!     * Communication with remote CPU cores takes the longest time, so in the
//!       interest of using all CPU cores for the longest possible share of the
//!       total job execution time, it makes sense to initiate the remote
//!       execution process as early as possible.
//!     * Communication with remote CPU cores is expensive, so we want to do it
//!       as few times as possible throughout the job execution process. By
//!       sending the biggest possible chunks of work to remote workers, we
//!       maximally amortize the long-distance communication overhead and ensure
//!       that they won't need to get back to us looking for more work during
//!       the longest possible amount of time.
//!     * Greedy scheduling that starts by communicating with the closest
//!       worker, then moves on to communicate with increasingly remote workers,
//!       is not a good idea because while it initially works very well as tasks
//!       spread with optimal efficiency, it means that by the time we will have
//!       filled, say, the current CPU package, all CPUs in that package will
//!       then simultaneously try to go and send work to the other package,
//!       resulting in a "thundering" herd scenario of many CPU cores all trying
//!       to send many relatively small tasks across the relatively
//!       low-performance interconnect between CPU packages.
//! - One important issue with the above work distribution strategy is that in a
//!   pure binary fork join API, we don't know how much a task can be split,
//!   which gets in the way of taking optimal scheduling decisions.
//!     * If we optimize for optimally large, highly splittable tasks by
//!       spreading to the most remote CPUs on the system first, then we will
//!       pessimize smaller tasks that cannot leverage the full system and would
//!       be happier executing on a tightly packed set of CPU cores sharing more
//!       execution resources.
//!     * The solution here is to have a scheduling hint that tells us how big
//!       we should expect tasks to be, either as a thread pool wide scheduling
//!       hint or as a per-task scheduling hint.
//!     * A per-task hint is more flexible as it allows the thread pool to
//!       perform optimally when processing tasks with different concurrency
//!       characteristics, but it is also obviously a lot less user-friendly.
//! - If load balancing through work stealing is necessary after the initial
//!   work distribution cycle, then it should follow the pattern of priorizing
//!   communication with the closest CPU cores by the metric of interest (which
//!   is typically "as close as possible in the hardware topology, except for
//!   hyperthreads which are treated as maximally remote to optimize the odds
//!   that they will eventually end up running unrelated thread pool jobs".
//!
//! To summarize, we should have optional metadata that is job specific, but may
//! take a thread pool wide default as a usability/performance/scheduling speed
//! compromise, which gives us the scheduling priority information handled by
//! this module.

use hwlocality::{
    cpu::cpuset::CpuSet,
    object::{depth::NormalDepth, types::ObjectType, TopologyObjectID},
    Topology,
};
use std::collections::{HashMap, VecDeque};

/// Workload properties that influence scheduling
///
/// Specifying any of these properties is optional. The defaults are intended to
/// work well for well-tuned jobs (enough concurrency to use all CPU cores,
/// simple and efficient inner loops, good CPU cache locality), but in less
/// ideal usage scenarios you may benefit from deviating from them.
///
/// Because job scheduling has to be very fast (we're talking microseconds
/// here), per-job scheduling decisions have to be less clever than global
/// thread-pool-wide decisions, and may sometimes need to ignore some of this
/// metadata for scheduler performance reasons. As a result, if you know that
/// you will always be dealing with a certain kind of job, you should favor
/// setting these parameters at the thread pool defaults level, rather than at
/// the per-job level.
//
// --- Implementation nodes ---
//
// TODO: Start by making this a thread pool configuration option, then if needed
//       add a subset of this to ThreadPool::run()
#[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct JobProperties {
    /// Maximal job concurrency
    ///
    /// This is an estimate of the maximal number of subtasks that a job can be
    /// split into, i.e. the maximal parallelism that a job can achieve for some
    /// subset of its execution time, assuming perfect scheduling.
    ///
    /// By default, we optimize scheduling for jobs that can be split enough to
    /// use all available CPU cores in the thread pool.
    maximal_concurrency: Option<usize>,

    /// Job benefits more from extra hyperthreads than from extra CPU cores
    ///
    /// This asserts that subtasks of a job can efficiently execute on multiple
    /// threads of the same CPU core, to the point where when there isn't enough
    /// concurrency to cover all CPU cores, spreading work across packed threads
    /// should be considered more beneficial than spreading it across CPU cores.
    ///
    /// Hyperthreading normally works best when it is used to execute unrelated
    /// jobs, as multiple subtasks of a single job would contend for the same
    /// shared CPU core resources. This means that distributing work from a
    /// single job across hyperthreads should be at the bottom of a CPU
    /// scheduler's work distribution priority list, which is what leaving this
    /// setting at its default `false` value achieves.
    ///
    /// However, there are a few edge cases where distributing across
    /// hyperthreads is more beneficial, typically involving very short-lived
    /// subtasks bound by scheduling overhead or some other kind of inter-thread
    /// communication bottleneck. When you encounter one of these edge cases,
    /// you will want to flip this setting to `true`.
    prefer_threads_over_cores: bool,

    /// Job benefits more from extra NUMA nodes than from extra CPU cores
    ///
    /// This asserts that a job's execution speed is bound by DRAM bandwidth
    /// more than any CPU resource. As a result, the top priority when
    /// scheduling this job should be to spread its subtasks across NUMA nodes,
    /// not to optimize the use of intra-node ressources like CPU caches.
    ///
    /// While most real-world programs are memory bound, not all of them can
    /// efficiently leverage multiple NUMA nodes. Automatic OS NUMA management
    /// tends to be unsatisfactory, requiring some degree of manual memory
    /// locality tuning that typical programs do not engage in. As a result, the
    /// best default is to spread unrelated tasks across NUMA nodes, and this is
    /// what leaving this setting at its default `false` value achieves.
    ///
    /// If your job does leverage NUMA correctly, however, you may priorize work
    /// distribution across NUMA nodes by flipping this setting to `true`.
    prefer_numa_over_cores: bool,
}

/// Priority of a topology node
///
/// When distributing work across CPU cores, the task scheduler will focus first
/// on having the workload cover all children of the highest-priority nodes,
/// then all children of the next highest-priority nodes, and so on.
pub type Priority = usize;

/// Priorize multi-children hardware topology nodes for load balancing, assuming
/// a certain typical thread pool workload
///
/// This function gives topology nodes a priority for the purpose of load
/// balancing, i.e. exchanging work between worker threads after the initial job
/// startup to minimize the number of idle CPU threads.
///
/// For each PU in use, each ancestor that has other children with PUs is given
/// a priority that differs from that of the other ancestors of that PU. At
/// runtime, the worker associated with the PU will first try to exchange work
/// with other children of the highest-priority ancestor node, then with other
/// children of the next highest-priority ancestor node, and so on.
pub fn priorize_load_balancing(
    topology: &Topology,
    affinity: &CpuSet,
    default_job_properties: &JobProperties,
) -> HashMap<TopologyObjectID, Priority> {
    // Group multi-children nodes by increasing depth / locality
    let mut parents_by_typed_depth = NormalDepth::iter_range(NormalDepth::MIN, topology.depth())
        .filter_map(|depth| {
            // Pick nodes with multiple children
            let parents = topology
                .objects_at_depth(depth)
                .filter(|obj| crate::children_in_cpuset(obj, affinity).count() > 1)
                .collect::<Vec<_>>();

            // To each depth, associate a (type, nodes) tuple
            (!parents.is_empty()).then(|| {
                (
                    topology
                        .type_at_depth(depth)
                        .expect("depth is valid by construction"),
                    parents,
                )
            })
        })
        .collect::<Vec<_>>();
    crate::debug!(
        "Priorizing distribution of work across multi-children parents \
        grouped by typed depth {parents_by_typed_depth:#?}"
    );

    // Increasing depth provides a natural load balancing priority order, as
    // higher-depth nodes have lower inter-child communication latencies and are
    // thus a best first candidate for load balancing. But we do need to make
    // some adjustments depending on the kind of job we're targeting.
    let mut parent_priority_classes = VecDeque::with_capacity(parents_by_typed_depth.len());

    // Except in edge cases where hyperthreading of related tasks is known to be
    // worthwhile, hyperthreads should be lowest priority, i.e. go first
    if !default_job_properties.prefer_threads_over_cores {
        if let Some((ObjectType::Core, cores)) = parents_by_typed_depth.pop() {
            parent_priority_classes.push_front(cores);
        }
    }

    // That's all we need for local load balancing (as opposed to initial task
    // distribution at the start of a job)
    parent_priority_classes.extend(
        parents_by_typed_depth
            .drain(..)
            .map(|(_ty, parents)| parents),
    );

    // Give each multi-children node a priority according to our conclusions
    parent_priority_classes
        .into_iter()
        .enumerate()
        .flat_map(|(priority, parents)| {
            parents
                .into_iter()
                .map(move |parent| (parent.global_persistent_index(), priority))
        })
        .collect()
}

// TODO: Now we need a different priorization mechanism for the job startup
//       process. As explained above, this process comes with different
//       priorities than load balancing:
//
//       - For a maximally scalable task that can use all of the thread pool's
//         worker threads, we want to distribute the first (biggest) shards of
//         the work across the most remote hardware locations, then distribute
//         subsequent (smaller) shards of work to increasingly closer hardware
//         PUs, until we get to the point where we distribute work across CPU
//         cores within the same locality, or hyperthreads if the task benefits
//         from internal hyperthreading.
//       - For a task that cannot use all of the thread pool's worker threads,
//         what happens depends on whether the task is expected to be compute or
//         memory-bound
//          * If the task is expected to be compute-bound, then we ignore the
//            most remote hardware locations at the beginning of the
//            aforementioned priority list, and otherwise follow the same
//            process. For example if we are working on a system with two CPU
//            sockets that each have 32 CPU cores, but our job only exposes
//            16-ways concurrency, we do not attempt to distribute across CPU
//            sockets and instead start with the most remote _relevant_ resource
//            within a CPU socket for the purpose of distributing work to 16 CPU
//            cores: L3 shard, CPU core within an L3 shard...
//          * Note that this may require tracking how many CPU cores we fail to
//            distribute work to when we skip over a topology depth. And the
//            answer may be worker dependent on asymmetric topologies.
//          * If the task is expected to be memory-bound, then we use a variant
//            of the above process where we do not drop _all_ entries at the
//            beginning of the priority list, but instead keep topology nodes
//            that represent distributing work across NUMA nodes (as evidenced
//            by a change of nodeset from parent to children) at the top of our
//            priority list, ignoring remote intra-NUMA-node localities like L3
//            cache shards instead.
//       - This still needs to be rigorously checked, but I believe it's
//         possible to address all of the above with a single priority list,
//         with little overhead for compute-bound tasks but at the expense of
//         some overhead for memory-bound tasks (which are slower, and thus less
//         sensitive to scheduling overhead, so this is probably okay).
//          * Start by baking, within each worker, an array of indices of other
//            workers to distribute work to, from decreasingly remote locality.
//            For example, on a 4-socket system, the first worker to which we
//            distribute work would be another worker from socket N+2, the next
//            worker would be from socket N+1 (while meanwhile the worker from
//            socket N+2 distributes its own work to socket (N+2)+1 = N+3), the
//            next worker would be within another L3 cache shard of the current
//            socket, etc. Which index we should process next is tracked by some
//            kind of "where are we" generational counter on the task's Schedule
//            that's incremented when creating a child task's Schedule.
//          * Next to the above list of indices, track which range of indices
//            are about distributing work across NUMA nodes, and how many NUMA
//            nodes are being distributed over: sockets, sub-NUMA clustering...
//          * For compute-bound tasks, map the maximal task concurrency to an
//            index within the list of remote worker indices, which we will be
//            starting from. This is just a matter of initializing the task's
//            Schedule with the right "where are we" generational counter value.
//          * For memory-bound task, add a flag within the task's Schedule which
//            turns on memory-bound mode. When this mode is on, we start by
//            attempting the above "just skip the first few indices" strategy,
//            but if we detect that this leads us to skip distributing over NUMA
//            nodes, and we end up covering less NUMA nodes than our task
//            concurrency allows that way, then we end up keeping the NUMA
//            topology layers instead and skipping other things afterwards in
//            exchange.
//          * Now, the "interesting" part is figuring our how to do this without
//            memory allocation. This will probably require a faire bit of work,
//            so I should start with getting the compute-bound case right, and
//            only later add support for memory-bound tasks on NUMA systems.
