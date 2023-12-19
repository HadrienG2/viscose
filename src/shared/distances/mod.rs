//! Matrix of distances between ThreadPool workers

mod pu_iterator;

use self::pu_iterator::PUIterator;
use hwlocality::{
    bitmap::BitmapIndex,
    cpu::cpuset::CpuSet,
    object::{depth::NormalDepth, types::ObjectType, TopologyObject, TopologyObjectID},
    Topology,
};
use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Debug},
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
        log::info!(
            "Building a thread pool with affinity {affinity} corresponding to cpuset {cpuset}"
        );
        let num_workers = cpuset.weight().unwrap();
        assert!(
            num_workers < usize::from(Distance::MAX),
            "CPUs with >{} cores aren't supported yet, time to switch to the next integer width?",
            Distance::MAX
        );

        // Look up in which place of the topology work distribution decisions
        // must be made, and priorize those decisions
        let parent_priorities =
            Self::priorize_parents(topology, affinity, Self::optimize_for_compute);

        // Order PUs in such a way that neighbor PUs have high odds of being the
        // best targets for load balancing transactions (and indeed always are
        // when the hardware topology is symmetric)
        let worker_pus =
            PUIterator::new(topology, affinity, &parent_priorities).collect::<Vec<_>>();
        assert_eq!(worker_pus.len(), num_workers);
        let sorted_cpus = worker_pus
            .iter()
            .map(|pu| {
                BitmapIndex::try_from(pu.os_index().expect("PUs should have an OS index"))
                    .expect("PU logical index should fit in a cpuset")
            })
            .collect::<Vec<_>>();
        crate::info!("Workers were assigned CPUs {sorted_cpus:?}");
        assert_eq!(
            sorted_cpus.iter().copied().collect::<CpuSet>(),
            cpuset,
            "workers should cover all requested CPUs"
        );

        // Compute distance matrix
        crate::info!("Computing inter-worker distances...");
        let mut data = vec![Distance::MAX; num_workers * num_workers].into_boxed_slice();
        let mut neighborhood = Vec::with_capacity(num_workers);
        for worker_idx in 0..num_workers {
            // FIXME: The following is NOT a correct way to compute inter-PU
            //        distances. It would be better to...
            //
            //        - Start with a list composed of the worker + its ancestor
            //          list
            //        - For the worker + each ancestor except the root, iterate
            //          over (left, right) sibling pairs, then linearly over
            //          remainder
            //        - Priorize iterators using a BinaryHeap keyed by parent
            //          node priority
            //        - For each node yielded by above iteration, yield the
            //          children from closest to worker to furthest (right to
            //          left for left sibling, left to right for right sibling).
            //        - If these children are not PUs, recurse over them.
            //        - All this recursive iteration probably means that I'll
            //          not just need to key iterators by parent node priority,
            //          but by some kind of hierarchical priority
            //          (Vec<Priority>?).

            // Access distances from current worker and define distance metric
            crate::debug!("Computing distances from worker {worker_idx}...");
            let distances = &mut data[worker_idx * num_workers..(worker_idx + 1) * num_workers];
            let neighbor_priority = |neighborhood: &[&TopologyObject], new_neighbor_idx: usize| {
                crate::trace!(
                    "  * Computing priority of neighbor {new_neighbor_idx} (CPU {:?})",
                    sorted_cpus[new_neighbor_idx]
                );
                let new_neighbor = worker_pus[new_neighbor_idx];
                let priority = neighborhood.iter().map(|pu| {
                    let common_ancestor = pu.first_common_ancestor(new_neighbor).unwrap();
                    let common_priority = parent_priorities[&common_ancestor.global_persistent_index()];
                    crate::trace!("  * Neighborhood ancestor {common_ancestor} has priority {common_priority}");
                    common_priority
                })
                .min().unwrap();
                crate::trace!("  * Neighbor priority is {}", priority);
                priority
            };

            // Initialize distance computation
            let mut curr_distance = 0;
            let mut left_idx = worker_idx;
            let mut right_idx = worker_idx;
            let last_right_idx = num_workers - 1;
            neighborhood.clear();
            neighborhood.push(worker_pus[worker_idx]);
            distances[worker_idx] = 0;

            // Do bidirectional iteration as long as relevant
            crate::debug!("Resolving left/right neighbor priorities...");
            while left_idx > 0 && right_idx < last_right_idx {
                curr_distance += 1;
                crate::trace!("- At distance {curr_distance}");
                let left_priority = neighbor_priority(&neighborhood, left_idx - 1);
                let right_priority = neighbor_priority(&neighborhood, right_idx + 1);
                if left_priority >= right_priority {
                    left_idx -= 1;
                    crate::trace!("  * Left neighbor {left_idx} is at top priority, register it at current distance {curr_distance}");
                    distances[left_idx] = curr_distance;
                    neighborhood.push(worker_pus[left_idx]);
                }
                if left_priority <= right_priority {
                    right_idx += 1;
                    crate::trace!("  * Right neighbor {right_idx} is at top priority, register it at current distance {curr_distance}");
                    distances[right_idx] = curr_distance;
                    neighborhood.push(worker_pus[right_idx]);
                }
            }

            // Finish with unidirectional iteration
            crate::debug!("Adding remaining neighbors...");
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
            crate::debug!("Distances are ready!");
            crate::info!("{worker_idx:>3} -> {distances:>3?}");
        }
        (Self { data, num_workers }, sorted_cpus)
    }

    /// Access the distances from a certain worker to others
    pub fn from(&self, worker_idx: usize) -> &[Distance] {
        assert!(worker_idx < self.num_workers, "invalid worker index");
        &self.data[worker_idx * self.num_workers..(worker_idx + 1) * self.num_workers]
    }

    /// Priorize work distribution according to a certain policy
    ///
    /// A typical hwloc topology tree contains multiple branching points (NUMA
    /// nodes, L3 cache shards, multicore, hyperthreading...). To handle smaller
    /// tasks that cannot cover the full CPU, we must decide how important it is
    /// to spread tasks over these various branching points.
    ///
    /// The simplest policy to implement is to follow the hwloc topology tree :
    /// spread work over hyperthreads first, then cores, then L3 shards, then
    /// NUMA nodes). This policy has optimal cache locality and inter-task
    /// communication latency, however it is usually suboptimal in real-world
    /// use cases because hyperthreads contend over shared core ressources.
    ///
    /// The way you actually want to do it depends on the kind of work you're
    /// submitting to the thread pool. Therefore, the work distribution policy
    /// is configurable through the `make_priority_classes` callback. This
    /// callback receives parent nodes grouped by depth as a parameter, and is
    /// in charge of producing a list of priority classes in increasing priority
    /// order. Newly spawned tasks will then be distributed over parents in the
    /// highest priority class that is not fully covered yet.
    fn priorize_parents(
        topology: &Topology,
        affinity: &CpuSet,
        make_priority_classes: impl FnOnce(
            Vec<(ObjectType, NormalDepth, Vec<&TopologyObject>)>,
        ) -> Vec<Vec<&TopologyObject>>,
    ) -> HashMap<TopologyObjectID, Priority> {
        // Group multi-children nodes by increasing depth / locality
        let mut initial_parents = HashSet::new();
        let type_depth_parents = NormalDepth::iter_range(NormalDepth::MIN, topology.depth())
            .filter_map(|depth| {
                // Pick nodes with multiple children
                let parents = topology
                    .objects_at_depth(depth)
                    .filter(|obj| children_in_cpuset(obj, affinity).count() > 1)
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
        crate::debug!("Priorizing distribution of work across multi-children parents grouped by typed depth {type_depth_parents:#?}");

        // Let policy callback compute priority classes
        let priority_classes = make_priority_classes(type_depth_parents);
        crate::info!(
            "Work distribution priority classes in increasing priority order: {priority_classes:#?}"
        );
        let final_parents = priority_classes
            .iter()
            .flatten()
            .map(|parent| parent.global_persistent_index())
            .collect::<HashSet<_>>();
        assert_eq!(
            initial_parents, final_parents,
            "priorization policies should not add or remove parents"
        );

        // Give each multi node a numerical priority accordingly
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
    fn optimize_for_latency(
        type_depth_parents: Vec<(ObjectType, NormalDepth, Vec<&TopologyObject>)>,
    ) -> Vec<Vec<&TopologyObject>> {
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
    /// sibling, hyperthreading is usally quite ineffective on well-optimized
    /// compute-bound tasks, and the use of independent cores should normally be
    /// priorized over that of hyperthreads.
    #[allow(unused)]
    fn optimize_for_compute(
        mut type_depth_parents: Vec<(ObjectType, NormalDepth, Vec<&TopologyObject>)>,
    ) -> Vec<Vec<&TopologyObject>> {
        let mut result = Vec::with_capacity(type_depth_parents.len());
        Self::depriorize_hyperthreads(&mut type_depth_parents, &mut result);
        result.extend(Self::depths_to_priorities(type_depth_parents));
        result
    }

    /// Hyperthread-depriorization part of `optimize_for_compute`
    fn depriorize_hyperthreads<'topology>(
        type_depth_parents: &mut Vec<(ObjectType, NormalDepth, Vec<&'topology TopologyObject>)>,
        result: &mut Vec<Vec<&'topology TopologyObject>>,
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
    fn optimize_for_bandwidth(
        mut type_depth_parents: Vec<(ObjectType, NormalDepth, Vec<&TopologyObject>)>,
    ) -> Vec<Vec<&TopologyObject>> {
        // Start like optimize_for_compute
        let mut result = Vec::with_capacity(type_depth_parents.len());
        Self::depriorize_hyperthreads(&mut type_depth_parents, &mut result);

        // Extract the NUMA-relevant subset of the hwloc hierarchy in LIFO order
        let mut numa_depths_rev = Vec::new();
        'depths: for (depth_idx, (_, _, parents)) in type_depth_parents.iter().enumerate() {
            if parents.iter().any(|parent| {
                const NORMAL_NODESET_ERROR: &str = "normal objects should have a nodeset";
                let parent_nodeset = parent.nodeset().expect(NORMAL_NODESET_ERROR);
                parent
                    .normal_children()
                    .any(|child| child.nodeset().expect(NORMAL_NODESET_ERROR) != parent_nodeset)
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

/// Select normal children of a node that match the affinity mask
fn children_in_cpuset<'iterator, 'parent: 'iterator>(
    parent: &'parent TopologyObject,
    affinity: &'iterator CpuSet,
) -> impl DoubleEndedIterator<Item = &'parent TopologyObject> + Clone + 'iterator {
    parent.normal_children().filter(move |child| {
        child
            .cpuset()
            .expect("normal objects should have cpuset")
            .intersects(affinity)
    })
}
