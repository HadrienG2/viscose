//! Hierarchical thread pool state
//!
//! This version of the thread pool state translates the hwloc topology into a
//! tree of work availability flags, which can in turn be used by worker threads
//! to priorize giving or stealing work from the threads which are closest in
//! the topology and share the most resources with them.

mod builder;
pub(crate) mod path;

use self::{
    builder::HierarchicalStateBuilder,
    path::WorkAvailabilityPath,
};
use super::{flags::AtomicFlags, futex::WorkerFutex, job::DynJob, WorkerConfig, WorkerInterface};
use crate::bench::BitRef;
use crossbeam::{deque::Injector, utils::CachePadded};
use hwlocality::{cpu::cpuset::CpuSet, Topology};
use rand::Rng;
use std::{
    borrow::Borrow,
    sync::{atomic::Ordering, Arc}, collections::VecDeque,
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
        HierarchicalStateBuilder::from_topology_affinity(topology, affinity.borrow()).build()
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

    /// Enumerate workers with work available to steal at increasing distances
    /// from a certain "thief" worker
    pub fn find_work_to_steal<'result>(
        &'result self,
        worker_idx: usize,
        worker_availability: &'result WorkAvailabilityPath<'result>,
        load: Ordering,
    ) -> Option<impl Iterator<Item = usize> + 'result> {
        // We'll explore workers from increasingly remote parents
        let mut ancestor_bits = worker_availability.ancestors();

        // Locate our direct parent or return None in case we have no parent
        // (this means there are no other workers to steal work from)
        let worker_bit = ancestor_bits.next()?;
        let parent_idx = self.workers[worker_idx].parent.expect(
            "if a worker has a parent in its WorkAvailabilityPath, \
            then it should have a parent node index",
        );

        // Need at least Acquire load ordering to ensure that after reading out
        // a worker's set work availability flag, the worker's work is visible,
        // and after reading out a node's set work availability flag, the node
        // childrens' set work availability flags are also visible.
        let load = crate::at_least_acquire(load);

        // At each point in time, we are processing the subtree rooted at a
        // certain ancestor of the thief, and within that subtree we are
        // processing a certain Node. The thief worker may be a direct child of
        // that node, an indirect descendant, or neither.
        let mut ancestor_idx = parent_idx;
        let mut node_idx = parent_idx;
        let mut thief_bit = ThiefBit::Worker(worker_bit);

        // Nodes that we are going to process next in the subtree of the current
        // ancestor. The thief is not a descendent of any of these nodes.
        let mut next_foreign_nodes = VecDeque::new();

        // Yield iterator over increasingly remote workers
        // NOTE: genawaiter is just used as a convenience here as the iterator
        //       state machine used here starts to get quite complex to code by
        //       hand. But if genawaiter shows up too much in perf profiles,
        //       just bite the bullet and implement the iterator manually...
        Some(genawaiter::rc::Gen::new(|co| async move {
            'process_node: loop {
                // Access the current node
                let node = &self.work_availability_tree[node_idx].object;

                // Enumerate direct worker children with work to steal
                if let Some(worker_bit) = thief_bit.take_worker() {
                    if let Some(workers) = node.worker_children.find_relatives_to_rob(worker_bit, load) {
                        for victim_idx in workers {
                            co.yield_(victim_idx).await;
                        }
                    }
                } else if let Some(workers) = node.worker_children.find_strangers_to_rob(load) {
                    for victim_idx in workers {
                        co.yield_(victim_idx).await;
                    }
                }

                // Schedule looking at the node's children with work later
                if let Some(node_bit) = thief_bit.take_node() {
                    if let Some(nodes) = node.node_children.find_relatives_to_rob(node_bit, load) {
                        for node_idx in nodes {
                            next_foreign_nodes.push_back(node_idx);
                        }
                    }
                } else if let Some(nodes) = node.node_children.find_strangers_to_rob(load) {
                    for node_idx in nodes {
                        next_foreign_nodes.push_back(node_idx);
                    }
                }

                // Now we need to determine which node we're going to look at
                // next. First check if there are more nodes to process in the
                // subtree of the current ancestor. Process subtree nodes
                // breadth-first for an optimal memory access pattern.
                if let Some(next_node_idx) = next_foreign_nodes.pop_front() {
                    node_idx = next_node_idx;
                    continue 'process_node;
                }

                // If we ran out of nodes, then it means we're done exploring
                // the subtree associated with our current ancestor, so it's
                // time to go to the next ancestor.
                let Some(node_bit) = ancestor_bits.next() else {
                    // ...or end the search if we've run out of ancestors.
                    break 'process_node;
                };
                let ancestor = &self.work_availability_tree[ancestor_idx];
                ancestor_idx = ancestor.parent.expect(
                    "if an ancestor node has a parent in the WorkAvailabilityPath, \
                    then it should have a parent node index"
                );
                node_idx = ancestor_idx;
                thief_bit = ThiefBit::Node(node_bit);
            }
            })
            .into_iter()
        )
    }

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

    /// Find children that might have work to steal around a certain child
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
    /// The provided `thief_bit` indicates which child is looking for work.
    /// Potential targets will be enumerated at increasing distance from it.
    pub fn find_relatives_to_rob<'self_>(
        &'self_ self,
        thief_bit: &BitRef<'self_, true>,
        load: Ordering,
    ) -> Option<impl Iterator<Item = usize> + 'self_> {
        // Need at least Acquire ordering to ensure work is visible
        let load = crate::at_least_acquire(load);
        let bit_iter = self.work_availability.iter_set_around::<false, true>(thief_bit, load)?;
        Some(bit_iter.map(|bit| self.first_child_idx + bit.linear_idx(&self.work_availability)))
    }

    /// Find children that might have work to steal
    ///
    /// Like `find_relatives_to_rob`, but used when the thief is not part of
    /// this list. Potential targets will be enumerated in an unspecified order.
    pub fn find_strangers_to_rob(
        &self,
        load: Ordering,
    ) -> Option<impl Iterator<Item = usize> + '_> {
        // Need at least Acquire ordering to ensure work is visible
        let load = crate::at_least_acquire(load);

        // Pick a random search starting point so workers looking for work do
        // not end up always hammering the same targets.
        let mut rng = rand::thread_rng();
        let work_availability = &self.work_availability;
        let start_bit_idx = rng.gen_range(0..work_availability.len());
        let start_bit = work_availability.bit(start_bit_idx);

        // FIXME: Don't use NearestBitIterator here, just iterate linearly
        //        over set flags starting from a random starting point, i.e.
        //        consider flags x..N then flags 0..x. Add a new AtomicFlags
        //        iterator type for this, call it maybe LinearBitIterator.
        let bit_iter = work_availability.iter_set_around::<true, false>(&start_bit, load)?;
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

/// Relationship of a thief worker to a node that it's trying to steal from
#[derive(Debug, Eq, PartialEq)]
enum ThiefBit<'a> {
    /// Direct worker child
    Worker(&'a BitRef<'a, true>),

    /// Indirect descendant via this direct node child
    Node(&'a BitRef<'a, true>),

    /// Not a descendant
    Neither,
}
//
impl<'a> ThiefBit<'a> {
    /// If thief a direct worker child, take its bit reference
    fn take_worker(&mut self) -> Option<&'a BitRef<'a, true>> {
        if let Self::Worker(bit) = self {
            let result = Some(*bit);
            *self = Self::Neither;
            result
        } else {
            None
        }
    }

    /// If thief descends from a node child, take that child's bit reference
    fn take_node(&mut self) -> Option<&'a BitRef<'a, true>> {
        if let Self::Node(bit) = self {
            let result = Some(*bit);
            *self = Self::Neither;
            result
        } else {
            None
        }
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
