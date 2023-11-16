//! Hierarchical thread pool state
//!
//! This version of the thread pool state translates the hwloc topology into a
//! tree of work availability flags, which can in turn be used by worker threads
//! to priorize giving or stealing work from the threads which are closest in
//! the topology and share the most resources with them.

use std::sync::atomic::{self, Ordering};

use crate::shared::flags::bitref::FormerWordState;

use super::{
    flags::{bitref::BitRef, AtomicFlags},
    job::DynJob,
    WorkerInterface,
};
use crossbeam::{deque::Injector, utils::CachePadded};

/// State shared between all thread pool users and workers, with hierarchical
/// work availability tracking
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
    // TODO: Make SharedState::with_work_queues receive an affinity cpuset and
    //       topology as input and provide the worker index to CPU mapping as
    //       output (in the form of an array of logical CPU indices), because
    //       with HierarchicalState this mapping will become much less obvious
    //       and shouldn't be managed by the thread pool. Could merge that with
    //       the existing mechanism for returning work queues and call it
    //       WorkerConfig or something.
    // TODO: with_work_queues constructor that kinda does what SharedState
    //       version did but builds the work_availability_tree by probing hwloc,
    //       with the rules outlined above. Make SharedState::with_work_queues
    //       take an hwloc &Topology as a parameter for consistency.
    // TODO: injector() accessor + add one to SharedState too, then use it
    //       everywhere and hide SharedState::injector member.
    // TODO: worker(worker_idx) accessor + add one to SharedState too, then use
    //       it everywhere and hide SharedState::workers member.
    // TODO: work_availability(worker_idx) accessor that offloads to
    //       WorkAvailabilityPath::new() + add one to SharedState that offloads
    //       to AtomicFlags::bit_with_cache.
    // TODO: Provide replacement for usage shared.work_availability.bit() in
    //       ThreadPool, idea would be to lazily construct uncached BitRefs as
    //       needed as we move up the hierarchy.
    // TODO: recommend_steal and find_steal with an API that looks just like the
    //       SharedState version except it takes a WorkAvailabilityPath instead
    //       of a BitRef.
    // TODO: If I didn't miss anything, at this point I should be able to...
    //       - Hide SharedState::work_availability
    //       - Turn the current SharedState into a flat::FlatState
    //       - Create a SharedState trait that abstracts commonalities between
    //         FlatState and HierarchicalState
    //       - Rename FlatPool to ThreadPool, make that + Worker + anything else
    //         that needs it generic over the implementation of SharedState
    //       - Add a ThreadPoolBuilder that makes it easy to build either a flat
    //         or hierarchical thread pool.
    //       - Modify benchmark infrastructure to test over both flat and
    //         hierarchical thread pools
    //       - Run the updated benchmarks, profile them to understand results,
    //         fine-tune inlining and other affected performance stuff.
    //       - Report back findings to rayon issue
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
