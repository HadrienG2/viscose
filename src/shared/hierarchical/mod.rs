//! Hierarchical thread pool state
//!
//! This version of the thread pool state translates the hwloc topology into a
//! tree of work availability flags, which can in turn be used by worker threads
//! to priorize giving or stealing work from the threads which are closest in
//! the topology and share the most resources with them.

mod builder;
pub(crate) mod path;

use self::{
    builder::{ChildObjects, HierarchicalStateBuilder},
    path::WorkAvailabilityPath,
};
use super::{flags::AtomicFlags, futex::WorkerFutex, job::DynJob, WorkerConfig, WorkerInterface};
use crate::bench::BitRef;
use crossbeam::{deque::Injector, utils::CachePadded};
use hwlocality::{cpu::cpuset::CpuSet, Topology};
use rand::Rng;
use std::{
    borrow::Borrow,
    sync::{atomic::Ordering, Arc},
};

/// State shared between all thread pool users and workers, with hierarchical
/// work availability tracking.
#[derive(Debug)]
pub(crate) struct HierarchicalState {
    /// Global injector
    injector: Injector<DynJob>,

    /// One worker interface per worker thread
    ///
    /// All workers associated with a given tree node reside at consecutive
    /// indices, but the ordering of workers is otherwise unspecified.
    workers: Box<[CachePadded<Child<WorkerInterface>>]>,

    /// One node per hwloc TopologyObject with multiple children.
    ///
    /// Sorted in breadth-first order, first by source hwloc object depth and
    /// then within each depth by hwloc object cousin rank. This ensures that
    /// all the node children of a given parent reside at consecutive indices.
    work_availability_tree: Box<[Child<Node>]>,
}
//
impl HierarchicalState {
    /// Set up the shared and worker-local state
    pub fn with_worker_config(
        topology: &Topology,
        affinity: impl Borrow<CpuSet>,
    ) -> (Arc<Self>, Box<[WorkerConfig]>) {
        // Start building the tree
        let mut builder = HierarchicalStateBuilder::new(topology, affinity.borrow());

        // Double buffer of node with associated parent pointer in
        // work_availability_tree, for current tree depth + next tree depth
        let mut curr_child_objects = vec![ChildObjects {
            parent_idx: None,
            objects: vec![topology.root_object()],
        }];
        let mut next_child_objects = Vec::new();

        // Start of tree nodes for parent depth
        let mut first_node_at_prev_depth = 0;
        'depths: while !curr_child_objects.is_empty() {
            // Log current tree-building status, if enabled
            crate::debug!("Starting a new tree layer...");
            builder.trace_state();

            // End of tree nodes for parent depth
            let first_node_at_this_depth = builder.next_node_idx();

            // At each depth, topology objects are grouped into child sets that
            // represent group of children associated with the same parent node.
            let mut saw_root_child_set = false;
            'child_sets: for child_object_set in curr_child_objects.drain(..) {
                // Check that what looks like a lone root node is truly alone
                assert!(!saw_root_child_set, "root child set should be alone");

                // Log current tree-building status, if enabled
                crate::debug!(
                    "Will now process children of tree node {:?}",
                    child_object_set.parent_idx
                );

                // Track worker and node children
                let first_child_node_idx = builder.next_node_idx();
                let first_child_worker_idx = builder.next_worker_idx();

                // Iterate over topology objects from the current child set
                for object in child_object_set.objects {
                    if let Some(child_objects) =
                        builder.add_object(child_object_set.parent_idx, object)
                    {
                        next_child_objects.push(child_objects);
                    }
                }

                // At this point, we have collected all children of the active
                // work availability tree node
                let num_worker_children = builder
                    .num_workers()
                    .checked_sub(first_child_worker_idx)
                    .expect("number of workers can only increase");
                let num_node_children = builder
                    .num_nodes()
                    .checked_sub(first_child_node_idx)
                    .expect("number of tree nodes can only increase");
                crate::debug!(
                    "Done processing children of node {:?}, with a total of {num_worker_children} worker(s) and {num_node_children} node(s)",
                    child_object_set.parent_idx
                );

                // Handle edge cases without a parent node
                let Some(parent_idx) = child_object_set.parent_idx else {
                    match builder.num_nodes() {
                        0 => {
                            assert!(
                                builder.expected_workers() == 1
                                    && num_worker_children == 1
                                    && num_node_children == 0,
                                "parent-less node in a worker-only tree is \
                                only expected in a uniprocessor environment"
                            );
                            crate::debug!("No parent node due to uniprocessor edge case");
                            break 'depths;
                        }
                        1 => {
                            saw_root_child_set = true;
                            crate::debug!("This is the root node, there is no parent to set up");
                            continue 'child_sets;
                        }
                        _more => unreachable!("unexpected parent-less node"),
                    }
                };

                // Attach children to parent node
                crate::debug!("Will now attach children to parent node");
                assert!(
                    parent_idx >= first_node_at_prev_depth && parent_idx < first_node_at_this_depth,
                    "parent should be within last tree layer"
                );
                builder.setup_children(
                    parent_idx,
                    ChildrenLink::new(first_child_worker_idx, num_worker_children),
                    ChildrenLink::new(first_child_node_idx, num_node_children),
                );
            }

            // Done with this tree layer, go to next depth
            std::mem::swap(&mut curr_child_objects, &mut next_child_objects);
            first_node_at_prev_depth = first_node_at_this_depth;
        }
        builder.build()
    }

    /// Access the global work injector
    pub fn injector(&self) -> &Injector<DynJob> {
        &self.injector
    }

    /// Access the worker futexes
    pub fn worker_futexes(&self) -> impl Iterator<Item = &'_ WorkerFutex> {
        self.workers.iter().map(|child| &child.object.futex)
    }

    /// Generate a worker-private work availability path
    ///
    /// Workers can use this to signal when they have work available to steal
    /// and when they stop having work available to steal. It is also used as a
    /// worker identifier in other methods of this class.
    ///
    /// This accessor is meant to constructed by workers at thread pool
    /// initialization time and then retained for the entire lifetime of the
    /// thread pool. As a result, it is optimized for efficiency of repeated
    /// usage, but initial construction may be expensive.
    pub fn worker_availability(&self, worker_idx: usize) -> WorkAvailabilityPath<'_> {
        WorkAvailabilityPath::new(self, worker_idx)
    }

    /* /// Enumerate workers with work available to steal at increasing distances
    /// from a certain "thief" worker
    pub fn find_work_to_steal<'result>(
        &'result self,
        worker_availability: &'result WorkAvailabilityPath<'result>,
        load: Ordering,
    ) -> Option<impl Iterator<Item = usize> + 'result> {
        let mut ancestors = worker_availability.ancestors();
        let parent = ancestors.next()?;
        let mut searchers = Vec::with_capacity(worker_availability.num_ancestors());
        // FIXME: Start iterating for work to steal in parent, then grandparent,
        //        until we reach the point where either we have covered all
        //        ancestors without finding work (in which case we return None),
        //        or we have found a specific ancestor with work. Then dive down
        //        that ancestor's children until we successfully build a worker
        //        iterator. Repeat as many times as necessary.
        Some(std::iter::from_fn(move || {}))
    } */

    // TODO: Add more methods, reach feature parity with current SharedState
}

/// Object tagged with a [`ParentLink`]
///
/// In the [`HierarchicalState`], it is common to look up a worker or node's
/// parent node in order to perform work stealing or advertise work
/// availability. Therefore, these objects are tagged with a pointer to their
/// parent node, if any.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct Child<T> {
    /// Index of the parent node
    pub parent: ParentLink,

    /// Worker or node
    pub object: T,
}

/// Index of a worker or node's parent in
/// [`HierarchicalState::work_availability_tree`]
///
/// There are only two cases in which a worker or node may not have a parent:
///
/// 1. In uniprocessor systems, there is no need for work availability tracking
///    (the only purpose of this tracking is for workers to communicate with
///    other workers for work stealing coordination purposes). Therefore, there
///    is only a single worker with no parent.
/// 2. In multiprocessor systems, there is at least one node in the work
///    availability tree, but by definition the root node of the tree does not
///    have parents.
type ParentLink = Option<usize>;

/// Node of `HierarchicalState::work_availability_tree`
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Node {
    /// Location and availability of node children
    node_children: ChildrenLink,

    /// Location and availability of worker children
    worker_children: ChildrenLink,
}
//
impl Node {
    /// Clear all work availability flags
    ///
    /// This method is solely intended for use by unit tests as a quick "reset
    /// to initial state" knob.
    #[cfg(test)]
    pub(crate) fn clear_work_availability(&self) {
        self.node_children
            .work_availability
            .clear_all(Ordering::Relaxed);
        self.worker_children
            .work_availability
            .clear_all(Ordering::Relaxed);
    }
}

/// Number and availability of [`Node`] children of a certain kind
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ChildrenLink {
    /// Availability flags
    work_availability: AtomicFlags,

    /// First child index in the relevant global table
    first_child_idx: usize,
}
//
impl ChildrenLink {
    /// Set up a children link for a set of N children, contiguously stored in
    /// the relevant global object list starting at a certain index
    pub fn new(first_child_idx: usize, num_children: usize) -> Self {
        Self {
            work_availability: AtomicFlags::new(num_children),
            first_child_idx,
        }
    }

    /// Number of children of this kind
    pub fn num_children(&self) -> usize {
        self.work_availability.len()
    }

    /// Build a fast accessor to a child's work availability bit
    ///
    /// Children are identified by their index in the relevant global object
    /// list, i.e. `HierarchicalState::workers` for workers and
    /// `HierarchicalState::work_availability_tree` for nodes.
    ///
    /// Workers are encouraged to cache the output of this function for all of
    /// their ancestor nodes in the work availability tree. It is a bit
    /// expensive to compute initially, but ensures faster operations on work
    /// availability bits in the long run.
    pub fn child_availability(&self, global_child_idx: usize) -> BitRef<'_, true> {
        let bit_idx = self.child_bit_idx(global_child_idx);
        self.work_availability.bit_with_cache(bit_idx)
    }

    /// Find children that might have work to steal
    ///
    /// For worker children, "might have work to steal" means the worker has
    /// pushed work to its work queue and not seen it empty since (though it
    /// might well become empty as a result of other workers stealing work from
    /// it ). Result indices refer to [`WorkerInterface`] indices in the global
    /// `HierarchicalState::workers` table.
    ///
    /// For node children, "might have work to steal" means that the associated
    /// node has some worker(s) beneath it which announced having work available
    /// for stealing as described above. Result indices refer to [`Node`]
    /// indices in the global `HierarchicalState::work_availability_tree` table.
    ///
    /// If the thief is another child from the same children list, then it
    /// should set `thief_bit` to its own bit in `work_availability`. This will
    /// bias the work-stealing algorithm towards stealing from other children
    /// closest in the hwloc-provided child list, which may enjoy slightly
    /// faster communication depending on CPU microarchitecture.
    ///
    /// Otherwise, `thief_bit` should be set to `None`, which will lead to a
    /// fair load balancing of the work-stealing workload through victim search
    /// order randomization.
    pub fn find_children_to_rob<'self_>(
        &'self_ self,
        thief_bit: Option<&BitRef<'self_, true>>,
        load: Ordering,
    ) -> Option<impl Iterator<Item = usize> + 'self_> {
        let work_availability = &self.work_availability;
        // Need at least Acquire ordering to ensure work is visible
        let load = crate::at_least_acquire(load);
        let bit_iter = if let Some(thief_bit) = thief_bit {
            // If the thief is a child of ours, look for work at increasing
            // distances from this child
            work_availability.iter_set_around::<false, true>(thief_bit, load)
        } else {
            // Otherwise, start search from a random point of the child list to
            // balance the work-stealing workload across children
            let mut rng = rand::thread_rng();
            let start_bit_idx = rng.gen_range(0..work_availability.len());
            let start_bit = work_availability.bit(start_bit_idx);
            work_availability.iter_set_around::<false, false>(&start_bit, load)
        }?;
        // Translate local bits into global child indices
        Some(bit_iter.map(|bit| self.first_child_idx + bit.linear_idx(work_availability)))
    }

    /// Translate a global child object index into a local work availability bit
    /// index if this object is truly our child
    fn child_bit_idx(&self, global_child_idx: usize) -> usize {
        let bit_idx = global_child_idx
            .checked_sub(self.first_child_idx)
            .expect("global index too low to be a child");
        assert!(
            bit_idx < self.num_children(),
            "global index too high to be a child"
        );
        bit_idx
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::{collections::VecDeque, sync::atomic::Ordering};

    /// Arbitrary HierarchicalState
    ///
    /// Unlike the HierarchicalState constructor, this uses an affinity mask
    /// which only contains CPUs from the topology, resulting in superior state
    /// shrinking since the state search space is smaller.
    pub(crate) fn hierarchical_state() -> impl Strategy<Value = Arc<HierarchicalState>> {
        let topology = crate::topology();
        let cpus = topology.cpuset().iter_set().collect::<Vec<_>>();
        let num_cpus = cpus.len();
        prop::sample::subsequence(cpus, 1..=num_cpus).prop_map(move |cpus| {
            HierarchicalState::with_worker_config(topology, cpus.into_iter().collect::<CpuSet>()).0
        })
    }

    proptest! {
        /// Test HierarchicalState construction
        #[test]
        fn new_hierarchical_state(affinity: CpuSet) {
            crate::setup_logger_once();

            let topology = crate::topology();
            let make_state = || HierarchicalState::with_worker_config(topology, &affinity);
            let (state, worker_configs) = if topology.cpuset().intersects(&affinity) {
                make_state()
            } else {
                crate::tests::assert_panics(make_state);
                return Ok(());
            };

            let expected_cpuset = topology.cpuset() & affinity;
            let expected_num_workers = expected_cpuset.weight().unwrap();

            assert_eq!(worker_configs.len(), expected_num_workers);
            assert_eq!(
                worker_configs
                    .iter()
                    .map(|config| config.cpu)
                    .collect::<CpuSet>(),
                expected_cpuset
            );

            assert_eq!(state.workers.len(), expected_num_workers);

            'check_tree: {
                // The tree will be empty if and only if there is only a single
                // worker (uniprocessor system), in which case that worker doesn't
                // need a tree to synchronize with other workers.
                if expected_num_workers == 1 {
                    assert_eq!(state.work_availability_tree.len(), 0);
                    assert_eq!(state.workers[0].parent, None);
                    break 'check_tree;
                }
                assert!(state.work_availability_tree.len() >= 1);

                // Otherwise, explore tree node in (breadth-first) order
                let mut expected_parents = VecDeque::new();
                let mut expected_next_worker_idx = 0;
                let mut expected_next_node_idx = 1;
                for (idx, node) in state.work_availability_tree.iter().enumerate() {
                    // All work availability bits should initially be empty
                    let node_children = &node.object.node_children;
                    let worker_children = &node.object.worker_children;
                    let check_work_unavailable = |children: &ChildrenLink| {
                        assert!(children
                            .work_availability
                            .iter()
                            .all(|bit| !bit.is_set(Ordering::Relaxed)));
                    };
                    check_work_unavailable(node_children);
                    check_work_unavailable(worker_children);

                    // Check node parent index coherence
                    if idx == 0 {
                        assert_eq!(node.parent, None);
                    } else {
                        assert_eq!(node.parent, Some(expected_parents.pop_front().unwrap()));
                    }
                    for _ in 0..node_children.num_children() {
                        expected_parents.push_back(idx);
                    }

                    // Check worker parent index coherence
                    for worker_child_idx in 0..worker_children.num_children() {
                        let worker_idx = worker_children.first_child_idx + worker_child_idx;
                        assert_eq!(state.workers[worker_idx].parent, Some(idx));
                    }

                    // Check child index coherence
                    assert_eq!(node_children.first_child_idx, expected_next_node_idx);
                    expected_next_node_idx += node_children.num_children();
                    assert_eq!(worker_children.first_child_idx, expected_next_worker_idx);
                    expected_next_worker_idx += worker_children.num_children();
                }

                // The root node should indirectly point to all other trees or node
                assert_eq!(expected_next_worker_idx, state.workers.len());
                assert_eq!(expected_next_node_idx, state.work_availability_tree.len());
            }
        }
    }
}
