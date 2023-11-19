//! Hierarchical thread pool state
//!
//! This version of the thread pool state translates the hwloc topology into a
//! tree of work availability flags, which can in turn be used by worker threads
//! to priorize giving or stealing work from the threads which are closest in
//! the topology and share the most resources with them.

use super::{
    flags::{bitref::BitRef, AtomicFlags},
    job::DynJob,
    WorkerConfig, WorkerInterface,
};
use crate::shared::{flags::bitref::FormerWordState, futex::WorkerFutex};
use crossbeam::{deque::Injector, utils::CachePadded};
use hwlocality::{cpu::cpuset::CpuSet, object::TopologyObject, Topology};
use std::{
    assert_ne,
    borrow::Borrow,
    num::NonZeroUsize,
    sync::{
        atomic::{self, Ordering},
        Arc,
    },
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
    workers: Box<[CachePadded<WorkerInterface>]>,

    /// One node per hwloc TopologyObject with multiple children.
    ///
    /// Sorted in breadth-first order, first by source hwloc object depth and
    /// then within each depth by hwloc object cousin rank. This ensures that
    /// all the node children of a given parent reside at consecutive indices.
    work_availability_tree: Box<[Node]>,
}
//
impl HierarchicalState {
    /// Set up the shared and worker-local state
    // FIXME: Figure out a way to modularize this
    pub fn with_worker_config(
        topology: &Topology,
        affinity: impl Borrow<CpuSet>,
    ) -> (Arc<Self>, Box<[WorkerConfig]>) {
        // Check if the affinity-restricted topology cpuset fits within
        // implementation limits and permits forward progress
        let affinity = affinity.borrow();
        crate::debug!("Setting up a thread pool with affinity {affinity}");
        let cpuset = topology.cpuset() & affinity;
        let num_workers = cpuset.weight().expect("topology cpuset should be finite");
        crate::debug!(
            "Affinity-constrainted topology has cpuset {cpuset} containing {num_workers} CPU(s)"
        );
        assert_ne!(
            num_workers, 0,
            "a thread pool without threads can't make progress and will deadlock on first request"
        );
        assert!(
            num_workers < WorkerFutex::MAX_WORKERS,
            "number of worker threads is above implementation limits"
        );

        // Traverse the topology tree to construct the state iteratively
        //
        // This will eventually become the tree state
        let mut worker_configs = Vec::with_capacity(num_workers);
        let mut worker_interfaces = Vec::with_capacity(num_workers);
        let mut work_availability_tree = Vec::new();
        //
        /// Topology objects that we need to attach to the tree
        #[derive(Debug)]
        struct ChildObjects<'topology> {
            /// Parent node, if any
            parent_idx: Option<usize>,

            /// Hwloc objects attached to this parent node
            objects: Vec<&'topology TopologyObject>,
        }
        //
        // Double buffer of node with associated parent pointer in
        // work_availability_tree, for current depth + next depth
        let mut curr_child_objects = vec![ChildObjects {
            parent_idx: None,
            objects: vec![topology.root_object()],
        }];
        let mut next_child_objects = Vec::new();
        //
        // Start of tree nodes for parent depth
        let mut last_depth_start = 0;
        'depths: while !curr_child_objects.is_empty() {
            // Log current tree-building status, if enabled
            crate::debug!("Starting a new tree layer...");
            crate::trace!(
                "Tree at current depth:\n  \
                worker_configs: {worker_configs:?},\n  \
                work_availability_tree: {work_availability_tree:#?}"
            );

            // End of tree nodes for parent depth
            let curr_depth_start = work_availability_tree.len();

            // At each depth, topology objects are grouped into child sets that
            // represent group of children associated with the same parent
            // work_availability_tree node.
            let mut saw_root_child_set = false;
            'child_sets: for mut child_object_set in curr_child_objects.drain(..) {
                // Check that lone root note is truly alone
                assert!(!saw_root_child_set, "root child set should be alone");

                // Log current tree-building status, if enabled
                crate::debug!(
                    "Will now process children of tree node {:?}",
                    child_object_set.parent_idx
                );

                // Track worker and node children
                let first_child_node_idx = work_availability_tree.len();
                assert_eq!(worker_interfaces.len(), worker_configs.len());
                let first_child_worker_idx = worker_configs.len();

                // Iterate over topology objects from the current child set
                'objects: for mut object in child_object_set.objects.drain(..) {
                    // Discriminate between different kinds of children
                    let restricted_children = 'single_child: loop {
                        // Log current tree-building status, if enabled
                        crate::trace!("Evaluating child {object} for insertion in the tree");

                        // Count object CPUs
                        let cpuset = object
                            .cpuset()
                            .expect("root object and its normal children should have cpusets");
                        let cpuset = cpuset & affinity;
                        let num_cpus = cpuset
                            .weight()
                            .expect("topology objects should have finite cpusets");
                        crate::trace!(
                            "Child has affinity-constrained cpuset {cpuset} containing {num_cpus} CPU(s)"
                        );

                        // Classify object accordingly
                        match num_cpus {
                            // Children without CPUs should be ignored earlier
                            0 => unreachable!("children without CPUs should have been weeded out in previous steps"),

                            // Single-CPU children are attached as workers
                            1 => {
                                let cpu = cpuset
                                    .first_set()
                                    .expect("cpusets with weight == 1 should have one entry");
                                crate::debug!("Adding child or grandchild worker for CPU {cpu}");
                                let (interface, work_queue) = WorkerInterface::with_work_queue();
                                worker_interfaces.push(CachePadded::new(interface));
                                worker_configs.push(WorkerConfig { work_queue, cpu });
                                continue 'objects;
                            }

                            // Multi-CPU children contain multiple branches down
                            // their children subtree. We'll recurse down to
                            // that branch and add it as a node child.
                            _multiple => {
                                // Filter out child objects by affinity
                                let mut restricted_children =
                                    object.normal_children().filter(|child| {
                                        let child_cpuset = child
                                            .cpuset()
                                            .expect("normal children should have cpusets");
                                        child_cpuset.intersects(affinity)
                                    });

                                // Exit child recursion loop once a child with
                                // multiple significant children is detected
                                if restricted_children.clone().count() > 1 {
                                    break 'single_child restricted_children;
                                }

                                // Single-child nodes are not significant for
                                // tree building, ignore them and recurse into
                                // their single child until a multi-child object
                                // comes out.
                                crate::trace!("Should not become a tree node as it only has one child, recursing to grandchild...");
                                object = restricted_children
                                    .next()
                                    .expect("cannot happen if count is 1");
                                let child_cpuset = object
                                    .cpuset()
                                    .expect("normal children should have cpusets");
                                let child_cpuset = child_cpuset & affinity;
                                assert_eq!(
                                    child_cpuset,
                                    cpuset,
                                    "if an object has a single child, the child should have the same cpuset as the parent"
                                );
                                continue 'single_child;
                            }
                        }
                    };

                    // If control reached this point, then we have a hwloc
                    // topology tree node with multiple normal children on our
                    // hands. Translate this node into a tree node.
                    let num_children = restricted_children.clone().count();
                    crate::debug!(
                        "Will turn child {object} into a new tree node with {num_children} children \
                        + schedule granchildren for processing on next 'depth iteration"
                    );
                    assert!(
                        num_children > 1,
                        "this point should only be reached for multi-children nodes"
                    );
                    let new_node_idx = work_availability_tree.len();
                    work_availability_tree.push(Node {
                        parent_idx: child_object_set.parent_idx,
                        // Cannot fill in the details of the node until we've
                        // processed the children, but thankfully we have a "no
                        // children yet" placeholder state.
                        work_availability: AtomicFlags::new(0),
                        node_children: None,
                        worker_children: None,
                    });

                    // Collect children to be processed once we move to the next
                    // tree depth (next iteration of the 'depth loop).
                    next_child_objects.push(ChildObjects {
                        parent_idx: Some(new_node_idx),
                        objects: restricted_children.collect(),
                    });
                }

                // At this point, we have collected all children of the active
                // work availability tree node
                let num_worker_children = worker_configs
                    .len()
                    .checked_sub(first_child_worker_idx)
                    .expect("number of workers can only increase");
                let num_node_children = work_availability_tree
                    .len()
                    .checked_sub(first_child_node_idx)
                    .expect("number of tree nodes can only increase");
                crate::debug!(
                    "Done processing children of node {:?}, with a total of {num_worker_children} worker(s) and {num_node_children} node(s)",
                    child_object_set.parent_idx
                );

                // Try to access the parent node, handle root node (= first
                // node) & uniprocessor tree (= no node, only one worker) cases
                let Some(parent_idx) = child_object_set.parent_idx else {
                    match work_availability_tree.len() {
                        0 => {
                            assert!(
                                num_workers == 1
                                    && num_worker_children == 1
                                    && num_node_children == 0
                            );
                            crate::debug!("No parent node due to uniprocessor edge case");
                            break 'depths;
                        }
                        1 => {
                            saw_root_child_set = true;
                            crate::debug!("No parent node due to root node edge case");
                            continue 'child_sets;
                        }
                        _more => unreachable!("unexpected parent-less node"),
                    }
                };

                // Access existing parent node
                crate::debug!("Will now attach children to parent node");
                assert!(
                    parent_idx >= last_depth_start && parent_idx < curr_depth_start,
                    "parent should be within last tree layer"
                );
                let parent = &mut work_availability_tree[parent_idx];

                // Allocate parent work availability flags
                let num_children = num_worker_children + num_node_children;
                assert_ne!(
                    num_children, 0,
                    "nodes without children should have been filtered out"
                );
                parent.work_availability = AtomicFlags::new(num_children);

                // Attach children to the parent node
                parent.node_children =
                    NonZeroUsize::new(num_node_children).map(|num_children| ChildrenLink {
                        first_child_idx: first_child_node_idx,
                        num_children,
                    });
                parent.worker_children =
                    NonZeroUsize::new(num_worker_children).map(|num_children| ChildrenLink {
                        first_child_idx: first_child_worker_idx,
                        num_children,
                    });
            }

            // Done with this tree layer, go to next depth
            std::mem::swap(&mut curr_child_objects, &mut next_child_objects);
            last_depth_start = curr_depth_start;
        }
        debug_assert_eq!(worker_configs.len(), num_workers);
        debug_assert_eq!(worker_interfaces.len(), num_workers);

        // Set up the global shared state
        let result = Arc::new(Self {
            injector: Injector::new(),
            workers: worker_interfaces.into(),
            work_availability_tree: work_availability_tree.into(),
        });
        crate::debug!("Final worker configs: {worker_configs:?}");
        crate::debug!("Final tree: {result:#?}");
        (result, worker_configs.into())
    }

    // TODO: Finish constructor, then rest
}

/// Node of `HierarchicalState::work_availability_tree`
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Node {
    /// Index of parent tree node, if any
    ///
    /// Child index within the parent can be deduced from the child index in the
    /// relevant global table by subtracting this global child index from the
    /// parent's first child index
    parent_idx: Option<usize>,

    /// Work availability flags for this depth level
    ///
    /// Node children come first, then worker children.
    ///
    /// Provide flag set methods that automatically propagate the setting of the
    /// first flag and the unsetting of the last flag to the parent node,
    /// recursively all the way up to the root. The BitRef cached by worker
    /// threads must honor this logic.
    work_availability: AtomicFlags,

    /// Index and number of node children
    node_children: Option<ChildrenLink>,

    /// Index and number of worker children
    worker_children: Option<ChildrenLink>,
}
//
impl Node {
    /// Index of the bit associated with a certain node child within
    /// self.work_availability, or None if this is not a child of this node
    fn node_bit_idx(&self, global_node_idx: usize) -> Option<usize> {
        let node_children = self.node_children?;
        node_children.child_idx(global_node_idx)
    }

    /// Index of the bit associated with a certain worker child within
    /// self.work_availability, or None if this is not a child of this node
    fn worker_bit_idx(&self, global_node_idx: usize) -> Option<usize> {
        let worker_children = self.worker_children?;
        let worker_idx = worker_children.child_idx(global_node_idx)?;
        Some(worker_idx + self.num_node_children())
    }

    /// Number of node children
    fn num_node_children(&self) -> usize {
        self.node_children
            .map_or(0, |node_children| usize::from(node_children.num_children))
    }
}

/// Number of children and index of the first child within the relevant list
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct ChildrenLink {
    /// First child index
    first_child_idx: usize,

    /// Number of children of this type
    num_children: NonZeroUsize,
}
//
impl ChildrenLink {
    /// Index of a certain globally numbered entity within this local child list
    fn child_idx(&self, global_idx: usize) -> Option<usize> {
        let child_idx = global_idx.checked_sub(self.first_child_idx)?;
        (child_idx < usize::from(self.num_children)).then_some(child_idx)
    }
}

/// Trail of `work_availability` bits from a worker to the root node
#[derive(Debug)]
pub(crate) struct WorkAvailabilityPath<'shared>(Box<[BitRef<'shared, true>]>);
//
impl<'shared> WorkAvailabilityPath<'shared> {
    /// Compute the work availability path for a given worker
    ///
    /// This is a fairly expensive computation, and workers are very strongly
    /// advised to cache the result instead or repeating the query.
    fn new(shared: &'shared HierarchicalState, worker_idx: usize) -> Self {
        // Handle uniprocessor system special case (single worker w/o a parent)
        if shared.work_availability_tree.is_empty() {
            assert_eq!(worker_idx, 0);
            return Self(Vec::new().into_boxed_slice());
        }

        // Find the direct parent of the worker and the relative index of the
        // worker within this parent's child list.
        let ((mut parent_idx, mut parent), worker_bit) = shared
            .work_availability_tree
            .iter()
            .enumerate()
            .rev()
            .find_map(|(node_idx, node)| {
                node.worker_bit_idx(worker_idx).map(|worker_bit_idx| {
                    (
                        (node_idx, node),
                        node.work_availability.bit_with_cache(worker_bit_idx),
                    )
                })
            })
            .expect("worker index is out of bounds");

        // From the first parent, we can deduce the full work availability path
        let mut path = vec![worker_bit];
        loop {
            // Find parent node, if any
            let Some(grandparent_idx) = parent.parent_idx else {
                break;
            };
            let grandparent = &shared.work_availability_tree[grandparent_idx];

            // Push work availability bit for this node with parent node
            let parent_bit_idx = grandparent
                .node_bit_idx(parent_idx)
                .expect("tree parent <-> grandparent links are inconsistent");
            path.push(grandparent.work_availability.bit_with_cache(parent_bit_idx));

            // Adjust iteration state to use grandparent as new parent
            parent_idx = grandparent_idx;
            parent = grandparent;
        }
        Self(path.into())
    }

    /// Set this worker's work availability bit, propagating information that
    /// work is available throughout the hierarchy
    ///
    /// Return the former worker-private work availability bit value, if any
    pub fn fetch_set(&self, order: Ordering) -> Option<bool> {
        self.fetch_op(BitRef::check_empty_and_set, order, true)
    }

    /// Clear this worker's work availability bit, propagating information that
    /// work isn't available anymore throughout the hierarchy
    ///
    /// Return the former worker-private work availability bit value, if any
    pub fn fetch_clear(&self, order: Ordering) -> Option<bool> {
        self.fetch_op(BitRef::check_full_and_clear, order, false)
    }

    /// Shared commonalities between `fetch_set` and `fetch_clear`
    ///
    /// `final_bit` is the bit value that the worker's work availability bit is
    /// expected to have after work completes. Since workers cache their work
    /// availability bit value and only update the public version when
    /// necessary, the initial value of the worker's work availability bit
    /// should always be `!final_bit`.
    fn fetch_op(
        &self,
        mut op: impl FnMut(&BitRef<'shared, true>, Ordering) -> FormerWordState,
        order: Ordering,
        final_bit: bool,
    ) -> Option<bool> {
        // Enforce stronger-than-Release store ordering if requested
        match order {
            Ordering::Relaxed | Ordering::Acquire | Ordering::Release | Ordering::AcqRel => {}
            Ordering::SeqCst => atomic::fence(order),
            _ => unimplemented!(),
        }

        // Propagate the info that this worker started or stopped having work
        // available throughout the hierarachical work availability state
        let mut old_worker_bit = None;
        for (idx, bit) in self.0.iter().enumerate() {
            // Adjust the work availability bit at this layer of the
            // hierarchical state, and check former word state
            //
            // This must be Release so that someone observing the work
            // availability bit at depth N with an Acquire load gets a
            // consistent view of the work availability bit at lower depths.
            //
            // An Acquire barrier is not necessary for us since we do not probe
            // any other state that's dependent on the former value of the work
            // availability bit. If the user requests an Acquire barrier for
            // their own purposes, it will be enforced by the fence below.
            let old_word = op(bit, Ordering::Release);

            // Collect the former worker-private work availability bit
            if idx == 0 {
                old_worker_bit = Some(match old_word {
                    FormerWordState::EmptyOrFull => !final_bit,
                    FormerWordState::OtherWithBit(bit) => bit,
                });
            }

            // If the word was previously all-cleared when setting the work
            // availability bit, or all-set when clearing it, propagate the work
            // availability information up the hierarchy.
            //
            // Otherwise, another worker has already done it for us.
            match old_word {
                FormerWordState::EmptyOrFull => continue,
                FormerWordState::OtherWithBit(_) => break,
            }
        }

        // Enforce stronger-than-relaxed load ordering if requested
        match order {
            Ordering::Relaxed | Ordering::Release => {}
            Ordering::Acquire | Ordering::AcqRel | Ordering::SeqCst => atomic::fence(order),
            _ => unimplemented!(),
        }
        old_worker_bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::VecDeque;

    proptest! {
        #[test]
        fn with_worker_config(affinity: CpuSet) {
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
                    break 'check_tree;
                }
                assert!(state.work_availability_tree.len() >= 1);

                // Otherwise, explore tree node in (breadth-first) order
                let mut expected_parents = VecDeque::new();
                let mut expected_next_worker_idx = 0;
                let mut expected_next_node_idx = 1;
                for (idx, node) in state.work_availability_tree.iter().enumerate() {
                    // All work availability bits should initially be empty
                    assert!(node
                        .work_availability
                        .iter()
                        .all(|bit| !bit.is_set(Ordering::Relaxed)));

                    // Check parent index coherence
                    if idx == 0 {
                        assert_eq!(node.parent_idx, None);
                    } else {
                        assert_eq!(node.parent_idx, Some(expected_parents.pop_front().unwrap()));
                    }
                    if let Some(ChildrenLink { num_children, .. }) = node.node_children {
                        for _ in 0..num_children.into() {
                            expected_parents.push_back(idx);
                        }
                    }

                    // Check child index coherence
                    if let Some(ChildrenLink {
                        first_child_idx,
                        num_children,
                    }) = node.node_children
                    {
                        assert_eq!(first_child_idx, expected_next_node_idx);
                        expected_next_node_idx += usize::from(num_children);
                    }
                    if let Some(ChildrenLink {
                        first_child_idx,
                        num_children,
                    }) = node.worker_children
                    {
                        assert_eq!(first_child_idx, expected_next_worker_idx);
                        expected_next_worker_idx += usize::from(num_children);
                    }
                }

                // The root node should indirectly point to all other trees or node
                assert_eq!(expected_next_worker_idx, state.workers.len());
                assert_eq!(expected_next_node_idx, state.work_availability_tree.len());
            }
        }
    }

    /// Arbitrary HierarchicalState
    ///
    /// Unlike the HierarchicalState constructor, this uses an affinity mask
    /// which only contains CPUs from the topology, resulting in superior state
    /// shrinking since the state search space is smaller.
    fn hierarchical_state() -> impl Strategy<Value = Arc<HierarchicalState>> {
        let topology = crate::topology();
        let cpus = topology.cpuset().iter_set().collect::<Vec<_>>();
        let num_cpus = cpus.len();
        prop::sample::subsequence(cpus, 1..=num_cpus).prop_map(move |cpus| {
            HierarchicalState::with_worker_config(topology, cpus.into_iter().collect::<CpuSet>()).0
        })
    }

    proptest! {
        #[test]
        fn new_work_availability_path(state in hierarchical_state()) {
            // Test harness setup
            crate::setup_logger_once();
            crate::info!("Testing WorkAvailabilityPath construction over {state:#?}");
            let initial_tree = state.work_availability_tree.clone();

            // Uniprocessor edge case: no need for a synchronization tree
            if state.work_availability_tree.len() == 0 {
                assert_eq!(state.workers.len(), 1);
                return Ok(());
            }

            // Otherwise, iterate over node-attached workers and their parents
            let worker_parents = state
                .work_availability_tree
                .iter()
                .enumerate()
                .filter(|(_idx, node)| node.worker_children.is_some());
            for (parent_idx, parent) in worker_parents {
                crate::debug!("Checking WorkAvailabilityPath construction for workers below node #{parent_idx} ({parent:#?})");
                let workers = parent.worker_children.unwrap();
                for rel_worker_idx in 0..usize::from(workers.num_children) {
                    let global_worker_idx = rel_worker_idx + workers.first_child_idx;
                    let worker_bit_idx = parent.num_node_children() + rel_worker_idx;
                    crate::debug!("Checking child worker #{rel_worker_idx} (global #{global_worker_idx}, bit #{worker_bit_idx})");

                    // Check that Node::worker_bit_idx works
                    assert_eq!(
                        parent.worker_bit_idx(global_worker_idx),
                        Some(worker_bit_idx)
                    );

                    // Build a path and check that tree is unaffected
                    let path = WorkAvailabilityPath::new(&state, global_worker_idx);
                    crate::debug!("Got work availability path {path:#?}");
                    assert_eq!(state.work_availability_tree, initial_tree);

                    // Check that first path element points to the worker's parent
                    crate::debug!("Checking parent node #{parent_idx}...");
                    let mut path_elems = path.0.iter();
                    let worker_elem = path_elems.next().unwrap();
                    assert_eq!(worker_elem, &parent.work_availability.bit(worker_bit_idx));

                    // Check that num_node_children works for worker-only parents
                    if parent.node_children.is_none() {
                        assert_eq!(parent.num_node_children(), 0);
                    }

                    // Regursively check parent nodes
                    let mut curr_node_idx = parent_idx;
                    let mut curr_node = parent;
                    for curr_parent_elem in path_elems {
                        // If path says there's a parent, there should be one...
                        let curr_parent_idx = curr_node.parent_idx.unwrap();
                        let curr_parent = &state.work_availability_tree[curr_parent_idx];
                        crate::debug!("Checking ancestor node #{curr_parent_idx} ({curr_parent:#?})");

                        // ...and it should know about us
                        let our_child_list = curr_parent.node_children.unwrap();
                        assert!(our_child_list.first_child_idx <= curr_node_idx);
                        let rel_node_idx = curr_node_idx - our_child_list.first_child_idx;
                        assert!(rel_node_idx < usize::from(our_child_list.num_children));
                        assert_eq!(
                            curr_parent.node_bit_idx(curr_node_idx).unwrap(),
                            rel_node_idx
                        );
                        let node_bit_idx = rel_node_idx;

                        // Check that path element is located correctly
                        assert_eq!(
                            curr_parent_elem,
                            &curr_parent.work_availability.bit(node_bit_idx)
                        );

                        // Check that num_node_children works for node parents
                        assert_eq!(
                            curr_parent.num_node_children(),
                            usize::from(our_child_list.num_children)
                        );

                        // Update state for next recursion step
                        curr_node_idx = curr_parent_idx;
                        curr_node = curr_parent;
                    }
                    assert_eq!(curr_node.parent_idx, None);
                }
            }
        }
    }

    // TODO: Test that takes a HierarchicalState and a list of worker indices as
    //       input, and for each worker index try fetch_noop and fetch_flip +
    //       validate effect on HierarchicalState.
}
