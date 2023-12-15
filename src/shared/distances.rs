//! Matrix of distances between ThreadPool workers

use hwlocality::{
    bitmap::BitmapIndex,
    cpu::cpuset::CpuSet,
    object::{depth::NormalDepth, types::ObjectType, TopologyObject, TopologyObjectID},
    Topology,
};
use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Debug},
    ptr,
};

/// Distances between ThreadPool workers
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct Distances {
    /// Distance data from all workers
    ///
    /// Currently uses a simple unpadded 2D matrix layout such that
    /// distance(i, j) = data[i * Nworkers + j]. Could try alternate layouts
    /// optimized for extra cache locality later, if the need arises.
    data: Box<[Distance]>,

    /// Number of workers
    num_workers: usize,
}
//
impl Distances {
    /// Given an hwloc topology and an affinity mask, order selected CPUs in a
    /// manner that minimizes the distance between nearest neighbors and compute
    /// the associated matrix of inter-CPU distances
    pub fn with_sorted_cpus(topology: &Topology, affinity: &CpuSet) -> (Self, Vec<BitmapIndex>) {
        // Check preconditions
        let cpuset = topology.cpuset() & affinity;
        let num_workers = cpuset.weight().unwrap();
        assert!(
            num_workers < usize::from(Distance::MAX),
            "CPUs with >{} cores aren't supported yet, time to switch to the next integer width?",
            Distance::MAX
        );

        // Look up in which place of the topology work distribution decisions
        // must be made, and priorize those decisions
        let parent_priorities =
            Self::priorize_parents(topology, &cpuset, Self::optimize_for_compute);
        let sorted_pus =
            PUIterator::new(topology, affinity, &parent_priorities).collect::<Vec<_>>();
        assert_eq!(sorted_pus.len(), num_workers);

        unimplemented!(
            "compute distance matrix using a variant of the old strategy below, \
            the main new thing being that the ancestor chain must be re-sorted \
            by parent priority for topological distance evaluation"
        );

        // Order workers to put nearest neighbors close to each other
        worker_pus.sort_unstable_by_key(|pu| pu.logical_index());

        // Compute distance matrix
        let mut data = vec![Distance::MAX; num_workers * num_workers].into_boxed_slice();
        for worker_idx in 0..num_workers {
            // Access distances from current worker and define distance metric
            let distances = &mut data[worker_idx * num_workers..(worker_idx + 1) * num_workers];
            let topological_distance = |neighbor_idx: usize| {
                let worker = &worker_pus[worker_idx];
                let common = worker.common_ancestor(worker_pus[neighbor_idx]).unwrap();
                worker
                    .ancestors()
                    .take_while(|ancestor| !ptr::eq(common, *ancestor))
                    .count()
            };

            // Initialize distance computation
            let mut curr_distance = 0;
            let mut left_idx = worker_idx;
            let mut right_idx = worker_idx;
            let last_right_idx = num_workers - 1;
            distances[worker_idx] = 0;

            // Do bidirectional iteration as long as relevant
            while left_idx > 0 && right_idx < last_right_idx {
                curr_distance += 1;
                let topological_distance_left = topological_distance(left_idx - 1);
                let topological_distance_right = topological_distance(right_idx + 1);
                if topological_distance_left <= topological_distance_right {
                    left_idx -= 1;
                    distances[left_idx] = curr_distance;
                }
                if topological_distance_left >= topological_distance_right {
                    right_idx += 1;
                    distances[right_idx] = curr_distance;
                }
            }

            // Finish with unidirectional iteration
            loop {
                curr_distance += 1;
                if left_idx > 0 {
                    left_idx -= 1;
                    distances[left_idx] = curr_distance;
                } else if right_idx < last_right_idx {
                    right_idx += 1;
                    distances[right_idx] = curr_distance;
                } else {
                    break;
                }
            }
        }
        Self { data, num_workers }
    }

    /// Access the distances from a certain worker to others
    pub fn from(&self, worker_idx: usize) -> &[Distance] {
        assert!(worker_idx < self.num_workers, "invalid worker index");
        &self.data[worker_idx * self.num_workers..(worker_idx + 1) * self.num_workers]
    }

    /// Priorize work distribution according to a certain policy
    ///
    /// A typical hwloc topology tree contains multiple branching points (NUMA
    /// nodes, L3 cache shards, multicore, hyperthreading...) and when dealing
    /// with smaller tasks that cannot cover the full CPU, we must decide how
    /// important it is to spread tasks over these various branching points.
    ///
    /// The simplest policy to implement is to follow the hwloc topology tree :
    /// spread work over hyperthreads first, then cores, then L3 shards, then
    /// NUMA nodes). This policy has optimal cache locality and inter-task
    /// communication latencies, however it is usually suboptimal in real-world
    /// use cases because hyperthreads contend over shared core ressources.
    ///
    /// The way you actually want to do it depends on the kind of work you're
    /// submitting to the thread pool. Therefore, the work distribution policy
    /// is configurable through the `policy` callback. This callback receives
    /// parent nodes grouped by depth as a parameter, and is in charge of
    /// producing a list of priority classes in increasing priority order. Newly
    /// spawned tasks will then be distributed over parents in the highest
    /// priority class that is not full yet.
    fn priorize_parents<'topology>(
        topology: &'topology Topology,
        affinity: &CpuSet,
        policy: impl for<'parents> FnOnce(
            &CpuSet,
            Vec<(ObjectType, NormalDepth, Vec<&'parents TopologyObject>)>,
        ) -> Vec<Vec<&'parents TopologyObject>>,
    ) -> HashMap<TopologyObjectID, Priority> {
        // Group multi-children nodes by increasing depth / locality
        let mut initial_parents = HashSet::new();
        let type_depth_parents = NormalDepth::iter_range(NormalDepth::MIN, topology.depth())
            .filter_map(|depth| {
                // Pick nodes with multiple children that are covered by our
                // current affinity mask
                let parents = topology
                    .objects_at_depth(depth)
                    .filter(|obj| {
                        obj.cpuset().unwrap().intersects(affinity)
                            && children_in_cpuset(obj, affinity).count() > 1
                    })
                    .inspect(|parent| {
                        initial_parents.insert(parent.global_persistent_index());
                    })
                    .collect::<Vec<_>>();

                // For each depth, provide a (type, depth, nodes) tuple
                (!parents.is_empty()).then(|| {
                    (
                        topology
                            .type_at_depth(depth)
                            .expect("this is a valid depth"),
                        depth,
                        parents,
                    )
                })
            })
            .collect::<Vec<_>>();

        // Let policy callback compute priority classes accordingly
        let priority_classes = Self::optimize_for_compute(affinity, type_depth_parents);
        let final_parents = priority_classes
            .iter()
            .flatten()
            .map(|parent| parent.global_persistent_index())
            .collect::<HashSet<_>>();
        assert_eq!(
            initial_parents, final_parents,
            "priorization policies should not add or remove parents"
        );

        // Give each parent node a numerical priority accordingly
        priority_classes
            .into_iter()
            .enumerate()
            .flat_map(|(priority, parents)| {
                parents
                    .into_iter()
                    .map(move |parent| (parent.global_persistent_index(), priority))
            })
            .collect()
    }

    /// Work distribution policy optimized for latency-bound tasks
    ///
    /// Rigorously follows the hwloc hierarchy under the premise that what's
    /// local in that hierarchy is also likely to exhibit the highest cache
    /// locality and the lowest inter-task communication latency.
    #[allow(unused)]
    fn optimize_for_latency<'parents>(
        _affinity: &CpuSet,
        type_depth_parents: Vec<(ObjectType, NormalDepth, Vec<&'parents TopologyObject>)>,
    ) -> Vec<Vec<&'parents TopologyObject>> {
        Self::depths_to_priorities(type_depth_parents).collect()
    }

    /// Turn `type_depth_parents` into priority classes
    fn depths_to_priorities(
        type_depth_parents: Vec<(ObjectType, NormalDepth, Vec<&TopologyObject>)>,
    ) -> impl Iterator<Item = Vec<&TopologyObject>> {
        type_depth_parents
            .into_iter()
            .map(|(_, _, parents)| parents)
    }

    /// Work distribution policy optimized for compute-bound tasks
    ///
    /// Deviates from the standard hwloc hierarchy by filling up hyperthreaded
    /// cores last: since each hyperthread shares compute resources with its
    /// sibling, hyperthreading is usally very ineffective on well-optimized
    /// compute-bound tasks, and thus the use of independent cores should
    /// normally be priorized over that of hyperthreads.
    #[allow(unused)]
    fn optimize_for_compute<'parents>(
        _affinity: &CpuSet,
        mut type_depth_parents: Vec<(ObjectType, NormalDepth, Vec<&'parents TopologyObject>)>,
    ) -> Vec<Vec<&'parents TopologyObject>> {
        let mut result = Vec::with_capacity(type_depth_parents.len());
        Self::depriorize_hyperthreads(&mut type_depth_parents, &mut result);
        result.extend(Self::depths_to_priorities(type_depth_parents));
        result
    }

    /// Hyperthread-depriorization part of `optimize_for_compute`
    fn depriorize_hyperthreads(
        type_depth_parents: &mut Vec<(ObjectType, NormalDepth, Vec<&TopologyObject>)>,
        result: &mut Vec<Vec<&TopologyObject>>,
    ) {
        match type_depth_parents.pop() {
            Some((ObjectType::Core, _, cores)) => result.push(cores),
            Some(other) => type_depth_parents.push(other),
            None => {}
        }
    }

    /// Work distribution policy optimized for DRAM-bound tasks
    ///
    /// Picks up where `optimize_for_compute` left off, and additionally
    /// priorizes spreading work across NUMA nodes. This optimizes DRAM
    /// bandwidth at the cost of reducing cache locality, which should be the
    /// right tradeoff for memory-bound tasks that cannot fit in caches.
    #[allow(unused)]
    fn optimize_for_bandwidth<'parents>(
        affinity: &CpuSet,
        mut type_depth_parents: Vec<(ObjectType, NormalDepth, Vec<&'parents TopologyObject>)>,
    ) -> Vec<Vec<&'parents TopologyObject>> {
        // Start like optimize_for_compute
        let mut result = Vec::with_capacity(type_depth_parents.len());
        Self::depriorize_hyperthreads(&mut type_depth_parents, &mut result);

        // Extract the NUMA-relevant subset of the hwloc hierarchy in LIFO order
        let mut numa_depths_rev = Vec::new();
        'depths: for (depth_idx, (_, _, parents)) in type_depth_parents.iter().enumerate() {
            if parents.iter().any(|parent| {
                let parent_nodeset = parent.nodeset().unwrap();
                children_in_cpuset(parent, affinity)
                    .any(|child| child.nodeset().unwrap() != parent_nodeset)
            }) {
                numa_depths_rev.push(depth_idx);
            }
        }

        // Put these nodes back in FIFO order and priorize filling them first
        for numa_idx in numa_depths_rev.into_iter().rev() {
            let numa = type_depth_parents.remove(numa_idx);
            type_depth_parents.push(numa);
        }
        result.extend(Self::depths_to_priorities(type_depth_parents));
        result
    }
}
//
impl Debug for Distances {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Distances ")?;
        let mut worker_to_distances = f.debug_map();
        for worker_idx in 0..self.num_workers {
            worker_to_distances.entry(&worker_idx, &self.from(worker_idx));
        }
        worker_to_distances.finish()
    }
}

/// Distance between two workers
///
/// Should be wide enough to hold the maximum envisioned number of workers.
pub type Distance = u16;

/// Priority of a node for children enumeration (higher is higher priority)
pub type Priority = usize;

/// Topologically meaningful children of an hwloc "parent" object
///
/// Once there are no more children, a node should be deleted, and if it has a
/// parent the matching child should be set to None.
enum Children<'pu> {
    /// Other nodes
    Nodes {
        /// Globally persistent indices of nodes that still have PUs
        ///
        /// Once a child no longer has PUs, it is tombstoned away as a None.
        /// Deleting children is not a good idea because it would require
        /// rewriting the parent links of all children that come after.
        ///
        /// Remember to detect once all indices are None too.
        children_ids: Vec<Option<TopologyObjectID>>,

        /// Next child to yield
        next_child_idx: usize,

        /// For each child, we track the highest-priority task switch that ever
        /// occured below this child. This is pre-initialized to maximal
        /// priority, and used when iteration over a children list wraps around
        /// (since this is not switching to a new child, but rather to a new
        /// descendant with a certain priority below a child we've seen before).
        children_priorities: Vec<Priority>,
    },

    /// Processing units, ordered by decreasing logical index
    ///
    /// Pop vec to take next processing unit, once there are no more
    /// delete the corresponding child in the parent
    PUs(Vec<&'pu TopologyObject>),
}

/// Tree used to iterate over PUs in an order that matches work distribution
/// priorities (i.e. PUs that should share work first are close)
struct PUIterator<'caller> {
    /// Multi-child node work distribution priorities
    parent_priorities: &'caller HashMap<TopologyObjectID, Priority>,

    /// Summarized hwloc tree that only includes multi-children nodes
    parents: HashMap<TopologyObjectID, Children<'caller>>,

    /// Root of the parents tree
    root: Option<TopologyObjectID>,
}
//
impl<'caller> PUIterator<'caller> {
    /// Set up iteration over PUs
    fn new(
        topology: &'caller Topology,
        affinity: &CpuSet,
        parent_priorities: &'caller HashMap<TopologyObjectID, Priority>,
    ) -> Self {
        let mut root = None;
        let parents = topology
            .objects()
            .filter(|obj| {
                parent_priorities.contains_key(&obj.global_persistent_index())
                    || children_in_cpuset(obj, affinity)
                        .next()
                        .map(|obj| obj.object_type() == ObjectType::PU)
                        .unwrap_or(false)
            })
            .map(|mut parent| {
                // Find the simplified topology's root
                if !parent.ancestors().any(|ancestor| {
                    parent_priorities.contains_key(&ancestor.global_persistent_index())
                }) {
                    assert!(root.is_none(), "there should only be one root");
                    root = Some(parent.global_persistent_index());
                }

                // Enumerate children
                let children_type = parent.normal_children().next().unwrap().object_type();
                let children = children_in_cpuset(parent, affinity);
                let children = if children_type == ObjectType::PU {
                    let pus = children.rev().collect::<Vec<_>>();
                    debug_assert!(pus.len() > 1);
                    Children::PUs(pus)
                } else {
                    let children_ids = children
                        .map(|child| Some(child.global_persistent_index()))
                        .collect::<Vec<_>>();
                    let num_children = children_ids.len();
                    assert!(num_children >= 1);
                    Children::Nodes {
                        children_ids,
                        next_child_idx: 0,
                        children_priorities: vec![Priority::MAX],
                    }
                };

                // Collect parents in a hash map to easily find them
                (parent.global_persistent_index(), children)
            })
            .collect::<HashMap<_, _>>();
        let current_pu = topology
            .pus_from_cpuset(affinity)
            .min_by_key(|pu| pu.logical_index());
        Self {
            parent_priorities,
            parents,
            root,
        }
    }
}
//
impl<'caller> Iterator for PUIterator<'caller> {
    type Item = &'caller TopologyObject;

    fn next(&mut self) -> Option<Self::Item> {
        let root = self.root?;
        unimplemented!(
            "- Go to the root node of the tree
            - Trickle down the current child chain to find the PU-bearing leaf
                * Keep track of the (parent node, child idx) path we used to get
                  there, along with the ID of the leaf, we'll need these later.
            - Pop the next PU from the leaf node, this will be our result
            - If after this the leaf node doesn't have PUs anymore...
                * Delete the leaf node's entry in the tree
                * Propagate this child deletion to the parent:
                    - Look for the next child with PUs, allowing wraparound
                    - If there is one, mark the child we just deleted as None
                      and move to that next child
                    - Otherwise delete parent node and recurse to ancestor
                * For each object we delete, pop an entry in the (parent node,
                  child idx) path we previously tracked and update the leaf ID
                  tracker to the ID of the popped-entry, so the path keeps
                  matching the object we're currently looking at
                * If we end up deleting the root node, set self.root to None and yield result
            - Complement current (parent node, child idx) path with node
              priority information, check current object priority too.
            - Switch each object whose priority is higher than that of the
              current object to its next child, wrapping around as necessary.

              FIXME: I think the logic is a lot more subtle than this. For one
                     thing, the highest priority node should move on every cycle.
                     For another, there seems to be something about avoiding
                     wraparound at each parent until opportunities in higher
                     priority parents have been exhausted"
        )
    }
}

/// Select normal children of a node that match the affinity mask
fn children_in_cpuset<'out>(
    parent: &'out TopologyObject,
    affinity: &'out CpuSet,
) -> impl DoubleEndedIterator<Item = &'out TopologyObject> + 'out {
    parent
        .normal_children()
        .filter(move |child| child.cpuset().unwrap().intersects(affinity))
}
