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
use hwlocality::{
    cpu::cpuset::CpuSet, object::TopologyObject, topology::editor::RestrictFlags, Topology,
};
use std::{
    borrow::Borrow,
    debug_assert, debug_assert_eq,
    sync::{
        atomic::{self, Ordering},
        Arc,
    },
};

/// State shared between all thread pool users and workers, with hierarchical
/// work availability tracking.
//
// --- Implementation notes ---
//
// All inner arrays are sorted in breadth-first order, first by source hwloc
// object depth and then within each depth by hwloc object cousin rank.
pub(crate) struct HierarchicalState {
    /// Global injector
    injector: Injector<DynJob>,

    /// One worker interface per worker thread
    workers: Box<[CachePadded<WorkerInterface>]>,

    /// One node per hwloc TopologyObject with multiple children.
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
        // Restrict topology to desired affinity mask, and check if the residual
        // cpuset fits within implementation limits.
        let affinity = affinity.borrow();
        let topology = {
            let mut restricted = topology.clone();
            restricted.edit(|editor| {
                editor
                    .restrict(affinity, RestrictFlags::REMOVE_EMPTIED)
                    .expect("failed to restrict topology to affinity mask")
            });
            restricted
        };
        let num_workers = topology.cpuset().weight().unwrap();
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
        // Used to track worker children of the active parent
        let mut worker_children = CpuSet::new();
        // Used for recycling used Vecs instead of reallocating
        let mut objects_vec_morgue = Vec::<Vec<&TopologyObject>>::new();
        //
        /// Children of a certain parent node that we need to process
        struct Children<'topology> {
            /// Parent node, if any
            parent_idx: Option<usize>,

            /// Hwloc objects below this parent
            ///
            /// May be workers (cpuset == 1), locality-significant nodes
            /// (normal_children.count() > 1), or uninteresting nodes to be
            /// traversed (cpuset != 1 && normal_children.count() == 1).
            objects: Vec<&'topology TopologyObject>,
        }
        //
        // Double buffer of node with associated parent pointer in
        // work_availability_tree, for current depth + next depth
        let mut curr_children = vec![Children {
            parent_idx: None,
            objects: vec![topology.root_object()],
        }];
        let mut next_children = Vec::new();
        while !curr_children.is_empty() {
            for mut children_set in curr_children.drain(..) {
                // Prepare to track worker children
                //
                // Do not add worker children to the parent right away, buffer
                // worker children privately until we know if this node has both
                // node and worker children, or only worker children. OTOH we
                // can add node children to the tree right away.
                worker_children.clear();
                for object in children_set.objects.drain(..) {
                    // TODO: Assert cpuset.weight != 0
                    // Discriminate worker child (cpuset.weight == 1), node
                    // child (normal_children().count() > 1) and uninteresting
                    // topology detail to be skipped until we find a node
                    // (normal_children.count() == 1). For the latter, replace
                    // object with its only child until we find a node child.
                    todo!();
                }
                objects_vec_morgue.push(children_set.objects);

                // TODO: If there is >=1 node child, update parent to point to
                //       first node child and create tree nodes for all node
                //       children.
                // TODO: If node has both worker and node children, create an
                //       artificial node child to group all the worker children,
                //       then push all the worker children.
                // TODO: If node has only worker children, directly attach
                //       workers to the parent. Create workers as in SharedState
                //       constructor.
                // TODO: Recycle resources for reuse on next loop iteration
                // TODO: Put debug_assert!()s everywhere to check for correct tree
                //       building as rigorously as I can
            }
            std::mem::swap(&mut curr_children, &mut next_children);
        }
        debug_assert_eq!(worker_configs.len(), num_workers);
        debug_assert_eq!(worker_interfaces.len(), num_workers);

        // Set up the global shared state
        let result = Arc::new(Self {
            injector: Injector::new(),
            workers: worker_interfaces.into(),
            work_availability_tree: work_availability_tree.into(),
        });
        (result, worker_configs.into())
    }

    // TODO: Finish constructor, then rest
}

/// Node of `HierarchicalState::work_availability_tree`
struct Node {
    /// Index of parent tree node, if any
    ///
    /// Child index within the parent can be deduced from the child index in the
    /// relevant global table by subtracting this global child index from the
    /// parent's first child index
    parent_idx: Option<usize>,

    /// Work availability flags for this depth level
    ///
    /// Provide flag set methods that automatically propagate the setting of the
    /// first flag and the unsetting of the last flag to the parent node,
    /// recursively all the way up to the root. The BitRef cached by worker
    /// threads must honor this logic.
    work_availability: AtomicFlags,

    /// Link to the first child node or worker
    ///
    /// The number of children is tracked via `work_availability.len()`.
    children: ChildrenLink,
}

/// Node children
///
/// If a node has both nodes and workers as children (as happens on e.g. Intel
/// Adler Lake), create an artificial node child with all the worker children.
enum ChildrenLink {
    /// Children are normal tree nodes, starting at this index in
    /// `HierarchicalState::tree`
    Nodes { first_node_idx: usize },

    /// Children are workers, with public interfaces starting at this index in
    /// `HierarchicalState::workers`
    Workers { first_worker_idx: usize },
}

/// Trail of `work_availability` bits from a worker to the root node
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
        let ((mut parent_idx, mut parent), mut child_idx) = shared
            .work_availability_tree
            .iter()
            .enumerate()
            .rev()
            .find_map(|(node_idx, node)| {
                let ChildrenLink::Workers { first_worker_idx } = node.children else {
                    return None;
                };
                if worker_idx < first_worker_idx {
                    return None;
                }
                let rel_idx = worker_idx - first_worker_idx;
                (rel_idx < node.work_availability.len()).then_some(((node_idx, node), rel_idx))
            })
            .expect("invalid worker_idx or tree was incorrectly built");

        // From the first parent, we can deduce the full work availability path
        let mut path = Vec::new();
        loop {
            // Push current path node
            path.push(parent.work_availability.bit_with_cache(child_idx));

            // Find parent node, if any
            let Some(grandparent_idx) = parent.parent_idx else {
                break;
            };
            let grandparent = &shared.work_availability_tree[grandparent_idx];

            // Adjust iteration state to use grandparent as new parent
            let ChildrenLink::Nodes { first_node_idx } = grandparent.children else {
                panic!("tree was incorrectly built, parent <-> child link isn't consistent");
            };
            child_idx = parent_idx - first_node_idx;
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
