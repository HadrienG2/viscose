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
use super::{flags::AtomicFlags, job::DynJob, WorkerConfig, WorkerInterface, futex::StealLocation};
use crate::bench::BitRef;
use crossbeam::{deque::{Injector, Steal}, utils::CachePadded};
use hwlocality::{cpu::cpuset::CpuSet, Topology};
use rand::Rng;
use std::{
    borrow::{Borrow, Cow},
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

    /// Access the worker interfaces
    pub fn worker_interfaces(&self) -> impl Iterator<Item = &'_ WorkerInterface> {
        self.workers.iter().map(|child| &child.object)
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
        thief_worker_idx: usize,
        thief_availability: &'result WorkAvailabilityPath<'result>,
    ) -> Option<impl Iterator<Item = usize> + 'result> {
        self.find_workers(
            thief_worker_idx,
            thief_availability.ancestors().map(Cow::Borrowed),
            |children, thief_bit, load| children.find_relatives_to_rob(thief_bit, load),
            |children, load| children.find_strangers_to_rob(load)
        )
    }

    /// Given a worker with work available for stealing, find the closest cousin
    /// that doesn't have work, if any, and suggest that it steal from there
    ///
    /// There should be a `Release` barrier between the moment where work is
    /// pushed in the worker's work queue and the moment where work is signaled
    /// to be available like this. You can either bundle the `Release` barrier
    /// into this transaction, or put a separate `atomic::fence(Release)` before
    /// this transaction and make it `Relaxed` if you have multiple work
    /// availability signaling transactions to do.
    pub fn suggest_stealing_from_worker<'self_>(
        &'self_ self,
        target_worker_idx: usize,
        target_availability: &WorkAvailabilityPath<'self_>,
        update: Ordering,
    ) {
        self.suggest_stealing::<false, true>(
            target_worker_idx,
            target_availability.ancestors().map(Cow::Borrowed),
            StealLocation::Worker(target_worker_idx),
            update,
        )
    }

    /// Inject work from outside the thread pool, and tell the worker closest to
    /// the originating locality that doesn't have work about it
    pub fn inject_job(&self, job: DynJob, local_worker_idx: usize) {
        self.injector.push(job);
        self.suggest_stealing::<true, false>(
            local_worker_idx,
            WorkAvailabilityPath::lazy_ancestors(self, local_worker_idx).map(Cow::Owned),
            StealLocation::Injector,
            // Need at least Release ordering to ensure injected job is visible
            // to the target and don't need anything stronger:
            //
            // - Don't need AcqRel ordering since the thread that's pushing work
            //   does not want to get in sync with the target worker's state.
            // - Don't need SeqCst since there is no need for everyone to agree
            //   on the global order of job injection events.
            Ordering::Release,
        );
    }

    /// Try to steal a job from the global work injector
    pub fn steal_from_injector(&self) -> Steal<DynJob> {
        self.injector.steal()
    }

    /// Enumerate workers that match a certain work availability condition
    /// (either they have work available for stealing, or they are looking for
    /// work) at increasing distances from a certain "local" worker
    ///
    /// - `local_worker_idx` should identify the local worker around which we're
    ///   searching for opportunities to steal or give work
    /// - `ancestor_bits` should yield owned or borrowed [`BitRef`]s to each
    ///   node ancestor of the local worker, in order, as if you were iterating
    ///   over that worker's `WorkAvailabilityPath`.
    /// - `find_relatives` should be `find_relatives_to_rob` when stealing work,
    ///   `find_jobless_relatives` when giving work.
    /// - `find_strangers` should be `find_strangers_to_rob` when stealing work,
    ///   `find_jobless_strangers` when giving work.
    fn find_workers<'result, const CACHE_SEARCH_MASKS: bool, Relatives, Strangers>(
        &'result self,
        local_worker_idx: usize,
        mut ancestor_bits: impl Iterator<Item = Cow<'result, BitRef<'result, CACHE_SEARCH_MASKS>>> + 'result,
        mut find_relatives: impl FnMut(&'result ChildrenLink, &BitRef<'result, CACHE_SEARCH_MASKS>, Ordering) -> Option<Relatives> + 'result,
        mut find_strangers: impl FnMut(&'result ChildrenLink, Ordering) -> Option<Strangers> + 'result,
    ) -> Option<impl Iterator<Item = usize> + 'result>
        where
            Relatives: Iterator<Item = usize> + 'result,
            Strangers: Iterator<Item = usize> + 'result,
    {
        // Locate our direct parent or return None in case we have no parent
        // (this means there are no other workers to exchange work with)
        let worker_bit = ancestor_bits.next()?;
        let parent_idx = self.workers[local_worker_idx].parent.expect(
            "if a worker has a parent in its WorkAvailabilityPath, \
            then it should have a parent node index",
        );

        // All loads need to have at least Acquire ordering because...
        //
        // - Between observing the value of a worker's work availability bit and
        //   trying to act on the rest of that worker's state accordingly, an
        //   Acquire barrier is needed to make sure that we have an up-to-date
        //   view of the worker's state in question (as of the last work
        //   availability bit update at least).
        // - Between observing a node's work availability bit and querying the
        //   work availability bits of its children, an Acquire barrier is
        //   needed to make sure that the updates to the work availability bits
        //   of child nodes and workers that eventually led the node's work
        //   availability bit to be updated are visible.
        //
        // Further, load ordering doesn't need to be stronger than Acquire:
        //
        // - Don't need AcqRel (which would require replacing the load with a
        //   RMW) since we're not trying to get any other thread in sync with
        //   our current state during this specific lock-free transaction.
        // - Don't need SeqCst since there is no need for everyone to agree on
        //   the global order in which work is looked for and exchanged.
        let load = Ordering::Acquire;

        // At each point in time, we are processing the subtree rooted at a
        // certain ancestor of the thief, and within that subtree we are
        // processing a certain Node. The local worker may be a direct child of
        // that node, an indirect descendant, or neither.
        let mut ancestor_idx = parent_idx;
        let mut node_idx = parent_idx;
        let mut local_bit = LocalBit::Worker(worker_bit);

        // Nodes that we are going to process next in the subtree of the current
        // ancestor. The local worker is not a descendent of any of these nodes.
        let mut next_subtree_nodes = VecDeque::new();

        // Yield iterator over increasingly remote workers
        // NOTE: genawaiter is just used as a convenience here as the iterator
        //       state machine used here starts to get quite complex to code by
        //       hand. But if genawaiter shows up too much in perf profiles,
        //       just bite the bullet and implement the iterator manually...
        Some(
            genawaiter::rc::Gen::new(|co| async move {
                'process_nodes: loop {
                    // Access the current node
                    let node = &self.work_availability_tree[node_idx].object;

                    // Enumerate direct matching worker children
                    if let Some(worker_bit) = local_bit.take_worker() {
                        if let Some(workers) = find_relatives(&node.worker_children, worker_bit.borrow(), load) {
                            for victim_idx in workers {
                                co.yield_(victim_idx).await;
                            }
                        }
                    } else if let Some(workers) = find_strangers(&node.worker_children, load) {
                        for victim_idx in workers {
                            co.yield_(victim_idx).await;
                        }
                    }

                    // Schedule looking at the node's matching node children later
                    if let Some(node_bit) = local_bit.take_node() {
                        if let Some(nodes) = find_relatives(&node.node_children, node_bit.borrow(), load) {
                            for node_idx in nodes {
                                next_subtree_nodes.push_back(node_idx);
                            }
                        }
                    } else if let Some(nodes) = find_strangers(&node.node_children, load) {
                        for node_idx in nodes {
                            next_subtree_nodes.push_back(node_idx);
                        }
                    }

                    // Now we need to determine which node we're going to look at
                    // next. First check if there are more nodes to process in the
                    // subtree of the current ancestor. Process subtree nodes
                    // breadth-first for an optimal memory access pattern.
                    if let Some(next_node_idx) = next_subtree_nodes.pop_front() {
                        node_idx = next_node_idx;
                        continue 'process_nodes;
                    }

                    // If we ran out of nodes, then it means we're done exploring
                    // the subtree associated with our current ancestor, so it's
                    // time to go to the next ancestor.
                    let Some(node_bit) = ancestor_bits.next() else {
                        // ...or end the search if we've run out of ancestors.
                        break 'process_nodes;
                    };
                    let ancestor = &self.work_availability_tree[ancestor_idx];
                    ancestor_idx = ancestor.parent.expect(
                        "if an ancestor node has a parent in the WorkAvailabilityPath, \
                        then it should have a parent node index"
                    );
                    node_idx = ancestor_idx;
                    local_bit = LocalBit::Node(node_bit);
                }
            })
            .into_iter()
        )
    }

    /// Suggest stealing work from a particular worker
    fn suggest_stealing<'self_, const INCLUDE_CENTER: bool, const CACHE_SEARCH_MASKS: bool>(
        &'self_ self,
        local_worker_idx: usize,
        ancestor_bits: impl Iterator<Item = Cow<'self_, BitRef<'self_, CACHE_SEARCH_MASKS>>> + 'self_,
        task_location: StealLocation,
        update: Ordering,
    ) {
        // Check if there are job-less neighbors to submit work to...
        //
        // Need Acquire ordering so the futex is only read/modified after a work
        // unavailability signal is observed: compilers and CPUs should not
        // cache the work availability bit value or speculate on it here.
        let Some(mut asleep_neighbors) = self.find_workers(
            local_worker_idx,
            ancestor_bits,
            |children, target_bit, load| children.find_jobless_relatives::<INCLUDE_CENTER, CACHE_SEARCH_MASKS>(target_bit, load),
            |children, load| children.find_jobless_strangers(load)
        )
        else {
            return;
        };

        // ...and if so, tell the closest one about our newly submitted job
        #[cold]
        fn unlikely(
            self_: &HierarchicalState,
            jobless_neighbors: impl Iterator<Item = usize>,
            local_worker_idx: usize,
            task_location: StealLocation,
            update: Ordering,
        ) {
            // Iterate over increasingly remote job-less neighbors
            for closest_asleep in jobless_neighbors {
                // Update their futex recommendation as appropriate
                //
                // Can use Relaxed ordering on failure because failing to
                // suggest work to a worker has no observable consequences and
                // isn't used to inform any decision other than looking up the
                // state of the next worker, which is independent from this one.
                let accepted = self_.workers[closest_asleep].object.futex.suggest_steal(
                    task_location,
                    local_worker_idx,
                    update,
                    Ordering::Relaxed,
                );
                if accepted.is_ok() {
                    return;
                }
            }
        }
        unlikely(
            self,
            &mut asleep_neighbors,
            local_worker_idx,
            task_location,
            update,
        )
    }
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
    pub fn child_availability_with_cache(&self, global_child_idx: usize) -> BitRef<'_, true> {
        let bit_idx = self.child_bit_idx(global_child_idx);
        self.work_availability.bit_with_cache(bit_idx)
    }

    /// Like `child_availability_with_cache()`, but optimizes for accessor
    /// construction speed rather than accessor usage speed
    ///
    /// This is what you want if for whatever reason, you are forced to build a
    /// child availability accessor, use it once, and then throw it away.
    pub fn child_availability(&self, global_child_idx: usize) -> BitRef<'_, false> {
        let bit_idx = self.child_bit_idx(global_child_idx);
        self.work_availability.bit(bit_idx)
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
        self.find_relatives_around(thief_bit, load, AtomicFlags::iter_set_around::<false, true>)
    }

    /// Like `find_relatives_to_rob`, but finds relatives that are looking for
    /// work instead of relatives that might have extra work to give
    ///
    /// This is more generic than `find_relatives_to_rob` because we need to
    /// find jobless workers in two different circumstances: when injecting work
    /// from outside the thread pool and when spawning work inside of it.
    pub fn find_jobless_relatives<'self_, const INCLUDE_CENTER: bool, const CACHE_SEARCH_MASKS: bool>(
        &'self_ self,
        giver_bit: &BitRef<'self_, CACHE_SEARCH_MASKS>,
        load: Ordering,
    ) -> Option<impl Iterator<Item = usize> + 'self_> {
        self.find_relatives_around(giver_bit, load, AtomicFlags::iter_unset_around::<INCLUDE_CENTER, CACHE_SEARCH_MASKS>)
    }

    /// Find children that might have work to steal
    ///
    /// Like `find_relatives_to_rob`, but used when the thief is not part of
    /// this list. Potential targets will be enumerated in an unspecified order.
    pub fn find_strangers_to_rob(
        &self,
        load: Ordering,
    ) -> Option<impl Iterator<Item = usize> + '_> {
        self.find_strangers(load, AtomicFlags::iter_set_from)
    }

    /// Find children that are looking for work
    ///
    /// Like `find_stranger_to_rob`, but finds children that are looking ofr
    /// work instead of children that might have extra work to give
    pub fn find_jobless_strangers(
        &self,
        load: Ordering
    ) -> Option<impl Iterator<Item = usize> + '_> {
        self.find_strangers(load, AtomicFlags::iter_unset_from)
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

    /// Find children that might have/want work to steal around a certain child
    ///
    /// This is the common subset of `find_relatives_to_rob` and
    /// `find_jobless_relatives`. `iter_around` should be set to
    /// `AtomicFlags::iter_set_around` if you want to steal work and
    /// `AtomicFlags::iter_unset_around` if you want to give work, in both case
    /// with `INCLUDE_CENTER` set to false and `CACHE_SEARCH_MASKS` set to true.
    pub fn find_relatives_around<'self_, const CACHE_SEARCH_MASKS: bool, IterAround>(
        &'self_ self,
        center_bit: &BitRef<'self_, CACHE_SEARCH_MASKS>,
        load: Ordering,
        iter_around: impl FnOnce(&'self_ AtomicFlags, &BitRef<'self_, CACHE_SEARCH_MASKS>, Ordering) -> Option<IterAround>
    ) -> Option<impl Iterator<Item = usize> + 'self_>
        where IterAround: Iterator<Item = BitRef<'self_, false>> + 'self_
    {
        // Need at least Acquire ordering to ensure work is visible
        let load = crate::at_least_acquire(load);
        let bit_iter = iter_around(&self.work_availability, center_bit, load)?;
        Some(bit_iter.map(|bit| self.first_child_idx + bit.linear_idx(&self.work_availability)))
    }

    /// Find children that might have/want work to steal
    ///
    /// This is the common subset of `find_strangers_to_rob` and
    /// `find_jobless_stranger`. `iter_from` should be set to
    /// `AtomicFlags::iter_set_from` if you want to steal work and
    /// `AtomicFlags::iter_unset_from` if you want to give work.
    pub fn find_strangers<'self_, IterFrom>(
        &'self_ self,
        load: Ordering,
        iter_from: impl FnOnce(&'self_ AtomicFlags, &BitRef<'self_, false>, Ordering) -> Option<IterFrom>,
    ) -> Option<impl Iterator<Item = usize> + 'self_>
        where IterFrom: Iterator<Item = BitRef<'self_, false>> + 'self_
    {
        // Need at least Acquire ordering to ensure work is visible
        let load = crate::at_least_acquire(load);

        // Pick a random search starting point so workers looking for work do
        // not end up always hammering the same targets.
        let mut rng = rand::thread_rng();
        let work_availability = &self.work_availability;
        if work_availability.is_empty() {
            return None;
        }
        let start_bit_idx = rng.gen_range(0..work_availability.len());
        let start_bit = work_availability.bit(start_bit_idx);

        // Iterate over all set work availability bits, starting from that point
        // of the work availability flags and wrapping around
        let bit_iter = iter_from(work_availability, &start_bit, load)?;
        Some(bit_iter.map(|bit| self.first_child_idx + bit.linear_idx(work_availability)))
    }
}

/// Relationship of a worker to a node of the [`HierarchicalState`]
#[derive(Debug, Eq, PartialEq)]
enum LocalBit<'bit_ref, 'flags: 'bit_ref, const CACHE_SEARCH_MASKS: bool> {
    /// Direct worker child
    Worker(Cow<'bit_ref, BitRef<'flags, CACHE_SEARCH_MASKS>>),

    /// Indirect descendant via this direct node child
    Node(Cow<'bit_ref, BitRef<'flags, CACHE_SEARCH_MASKS>>),

    /// Not a descendant
    Neither,
}
//
impl<'bit_ref, 'flags: 'bit_ref, const CACHE_SEARCH_MASKS: bool> LocalBit<'bit_ref, 'flags, CACHE_SEARCH_MASKS> {
    /// If the worker is a direct worker child, take its [`BitRef`]
    fn take_worker(&mut self) -> Option<Cow<'bit_ref, BitRef<'flags, CACHE_SEARCH_MASKS>>> {
        if let Self::Worker(_) = self {
            let Self::Worker(bit) = std::mem::replace(self, Self::Neither) else {
                unreachable!()
            };
            Some(bit)
        } else {
            None
        }
    }

    /// If the worker descends from a node child, take that child's [`BitRef`]
    fn take_node(&mut self) -> Option<Cow<'bit_ref, BitRef<'flags, CACHE_SEARCH_MASKS>>> {
        if let Self::Node(_) = self {
            let Self::Node(bit) = std::mem::replace(self, Self::Neither) else {
                unreachable!()
            };
            Some(bit)
        } else {
            None
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::{collections::{VecDeque, HashSet}, sync::atomic::Ordering};

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

    /// Generate a HierarchicalState and a valid worker index in that state
    fn state_and_worker_idx() -> impl Strategy<Value = (Arc<HierarchicalState>, usize)> {
        hierarchical_state().prop_flat_map(|state| {
            let num_workers = state.workers.len();
            (Just(state), 0..num_workers)
        })
    }

    proptest! {
        /// Test search for work to steal
        #[test]
        fn find_work_to_steal((state, thief_idx) in state_and_worker_idx()) {
            // Determine set of possible work-stealing victims
            let mut remaining_victims = state.work_availability_tree.iter().flat_map(|node| {
                let workers = &node.object.worker_children;
                workers.find_strangers_to_rob(Ordering::Relaxed).into_iter().flatten()
            }).collect::<HashSet<_>>();

            // Compute all useful flavors of worker ancestor path
            let ancestor_nodes = |worker_idx: usize| std::iter::successors(
                state.workers[worker_idx].parent,
                |&parent_idx| state.work_availability_tree[parent_idx].parent
            );
            let thief_ancestor_nodes = ancestor_nodes(thief_idx).collect::<Vec<_>>();
            let thief_availability = state.worker_availability(thief_idx);

            // Track how far away the last observed victim was
            let mut last_common_ancestor_height = 0;
            let mut last_common_ancestor_child_distance = 0;
            let mut last_victim_depth = 0;

            // Check find_work_to_steal enumerates victims in the expected order
            let victims =
                state
                    .find_work_to_steal(thief_idx, &thief_availability)
                    .into_iter().flatten();
            for victim_idx in victims {
                // Make sure this was an expected work-stealing victim and that
                // this victim has only been yielded once
                assert!(remaining_victims.remove(&victim_idx));

                // Find common ancestor of thief and victim, and its direct node 
                // child pointing towards the victim
                let mut common_ancestor_node_child = None;
                let (height_above_victim, height_above_thief) =
                    // First enumerate ancestors of the victim + their height
                    ancestor_nodes(victim_idx).enumerate()
                        .find_map(|(height_above_victim, victim_ancestor)| {
                            // Then find height of matching thief ancestor
                            let result = thief_ancestor_nodes.iter()
                                .position(|&thief_ancestor| thief_ancestor == victim_ancestor)
                                // If search succeeds, we only care about the
                                // height of the common ancestor above the
                                // victim and the thief.
                                .map(|height_above_thief| (height_above_victim, height_above_thief));
                            // Track the last victim ancestor for which search
                            // fails, this will be the direct common ancestor
                            // node child pointing towards the victim.
                            if result.is_none() {
                                common_ancestor_node_child = Some(victim_ancestor);
                            }
                            result
                        })
                        .expect("thief and victim should share a common ancestor");

                    // Check that victims are ordered by increasing common
                    // ancestor height: the highest the common ancestor is, the
                    // more expensive it will be to synchronize with the victim.
                    use std::cmp::Ordering as CmpOrdering;
                    match height_above_thief.cmp(&last_common_ancestor_height) {
                        CmpOrdering::Less => panic!("victims are not ordered by ancestor height"),
                        CmpOrdering::Equal => {}
                        CmpOrdering::Greater => {
                            // Going to a higher common ancestor resets all
                            // other victim ordering tracking variables
                            last_common_ancestor_height = height_above_thief;
                            last_common_ancestor_child_distance = 0;
                            last_victim_depth = height_above_victim;
                        }
                    }

                    // Check that victims below a given common ancestor are
                    // ordered by increasing victim depth. Deeper victims are
                    // more expensive to enumerate as more of the tree must be
                    // traversed to query their state, so they should come last.
                    match height_above_victim.cmp(&last_victim_depth) {
                        CmpOrdering::Less => panic!("victims are not ordered by depth below common ancestor"),
                        CmpOrdering::Equal => {}
                        CmpOrdering::Greater => {
                            last_victim_depth = height_above_victim;
                        }
                    }

                    // Check that victims are ordered by increasing common
                    // ancestor child index distance. There is no strong
                    // rationale for this, but it balances work stealing without
                    // randomization and could ensure better performance if
                    // hwloc/OS enumeration is done in such a way that hardware
                    // with close indices is located closer to each other.
                    let child_distance = match (height_above_thief, (height_above_victim, common_ancestor_node_child)) {
                        // If the victim is a worker child of the common
                        // ancestor, it should not have node ancestors
                        (_, (0, Some(_))) => unreachable!(),

                        // Thief and victim are both direct worker children of
                        // the common ancestor: use worker child distance
                        (0, (0, None)) => thief_idx.abs_diff(victim_idx),

                        // Thief is a worker child of the common ancestor but
                        // victim is below a node child of the common ancestor:
                        // use an artificial distance to order this after worker
                        // victims, since node children of the common ancestor
                        // should be enumerated after worker children
                        (0, (_, Some(_))) => usize::MAX,

                        // If the victim is below a node child of the common
                        // ancestor, it should have node ancestors
                        (0, (_, None)) => unreachable!(),

                        // At this point, we've covered all cases where
                        // height_above_thief == 0, so we've established the
                        // thief is below a node child of the common ancestor.
                        //
                        // Order victims that are direct worker children of the
                        // common ancestor first, as the zero distance...
                        (_, (0, None)) => 0,

                        // ...then victims which are descendants of a node child
                        // of the common ancestor, shifting distances above the
                        // artificial zero distance that was already used.
                        (_, (_, Some(victim_child_idx))) => 1 + {
                            let thief_child_idx = thief_ancestor_nodes[height_above_thief - 1];
                            thief_child_idx.abs_diff(victim_child_idx)
                        },

                        // Once again, if the victim is below a node child of
                        // the common ancestor, it should have node ancestors
                        (_, (_, None)) => unreachable!(),
                    };
                    match child_distance.cmp(&last_common_ancestor_child_distance) {
                        CmpOrdering::Less => panic!("victims are not ordered by common ancestor child distance"),
                        CmpOrdering::Equal => {}
                        CmpOrdering::Greater => {
                            last_common_ancestor_child_distance = child_distance;
                        }
                    }
            }

            // Check find_work_to_steal enumerates all victims
            assert!(remaining_victims.is_empty());
        }
    }
}
