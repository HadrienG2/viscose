//! Node path from a worker to the top of the `work_availability_tree`

use super::HierarchicalState;
use crate::shared::flags::bitref::BitRef;
use std::{
    iter::FusedIterator,
    sync::atomic::{self, Ordering},
};

/// Trail of `work_availability` bits from a worker to the root node
///
/// Note that the first `BitRef`, if any, targets the `work_availability` of the
/// parent node's `worker_children`. Subsequent `BitRef`s target the
/// `work_availability` of ancestor nodes' `node_children`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WorkAvailabilityPath<'shared>(Box<[BitRef<'shared, true>]>);
//
impl<'shared> WorkAvailabilityPath<'shared> {
    /// Compute the work availability path for a given worker
    ///
    /// This is a fairly expensive computation, and workers are very strongly
    /// advised to cache the result instead or repeating the query.
    pub fn new(shared: &'shared HierarchicalState, worker_idx: usize) -> Self {
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
                node.worker_children
                    .child_availability(worker_idx)
                    .map(|worker_bit| ((node_idx, node), worker_bit))
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
            let parent_bit = grandparent
                .node_children
                .child_availability(parent_idx)
                .expect("tree parent <-> grandparent links are inconsistent");
            path.push(parent_bit);

            // Adjust iteration state to use grandparent as new parent
            parent_idx = grandparent_idx;
            parent = grandparent;
        }
        Self(path.into())
    }

    /// Number of ancestor nodes above this worker in the tree
    pub(super) fn num_ancestors(&self) -> usize {
        self.0.len()
    }

    /// Ancestor nodes above this worker
    pub(super) fn ancestors<'self_>(
        &'self_ self,
    ) -> impl ExactSizeIterator<Item = &'self_ BitRef<'shared, true>> + FusedIterator {
        self.0.iter()
    }

    /// Set this worker's work availability bit, propagating information that
    /// work is available throughout the hierarchy
    ///
    /// Return the former worker-private work availability bit value, if any
    pub fn fetch_set(&self, order: Ordering) -> Option<bool> {
        self.fetch_op(BitRef::check_empty_and_set, order)
    }

    /// Clear this worker's work availability bit, propagating information that
    /// work isn't available anymore throughout the hierarchy
    ///
    /// Return the former worker-private work availability bit value, if any
    pub fn fetch_clear(&self, order: Ordering) -> Option<bool> {
        self.fetch_op(BitRef::clear_and_check_emptied, order)
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
        mut op: impl FnMut(&BitRef<'shared, true>, Ordering) -> (bool, bool),
        order: Ordering,
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
            // hierarchical state, and check if the next word in path must be
            // updated as well
            //
            // This must be Release so that someone observing the work
            // availability bit at depth N with an Acquire load gets a
            // consistent view of the work availability bit at lower depths.
            //
            // An Acquire barrier is not necessary for us since we do not probe
            // any other state that's dependent on the former value of the work
            // availability bit. If the user requests an Acquire barrier for
            // their own purposes, it will be enforced by the fence below.
            let (old_bit, must_update_parent) = op(bit, Ordering::Release);

            // Collect the former worker-private work availability bit
            if idx == 0 {
                old_worker_bit = Some(old_bit);
            }

            // If the word was previously all-cleared when setting the work
            // availability bit, or if we cleared tbe last bit, propagate the
            // work availability information up the hierarchy.
            //
            // Otherwise, another worker has already done it for us.
            if must_update_parent {
                continue;
            } else {
                break;
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
    use crate::shared::hierarchical::Node;
    use hwlocality::cpu::cpuset::CpuSet;
    use proptest::{collection::SizeRange, prelude::*};
    use std::sync::Arc;

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
        /// Test WorkAvailiabilityPath construction
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
                .filter(|(_idx, node)| node.worker_children.num_children() != 0);
            for (parent_idx, parent) in worker_parents {
                crate::debug!("Checking WorkAvailabilityPath construction for workers below node #{parent_idx} ({parent:#?})");
                let workers = &parent.worker_children;
                for worker_bit_idx in 0..workers.num_children() {
                    let global_worker_idx = worker_bit_idx + workers.first_child_idx;
                    crate::debug!("Checking child worker #{worker_bit_idx} (global worker #{global_worker_idx})");

                    // Build a path and check that tree is unaffected
                    let path = WorkAvailabilityPath::new(&state, global_worker_idx);
                    crate::debug!("Got work availability path {path:#?}");
                    assert_eq!(state.work_availability_tree, initial_tree);

                    // Check that first path element points to the worker's parent
                    crate::debug!("Checking parent node #{parent_idx}...");
                    let mut path_elems = path.0.iter();
                    let worker_elem = path_elems.next().unwrap();
                    assert_eq!(worker_elem, &workers.work_availability.bit(worker_bit_idx));

                    // Regursively check parent nodes
                    let mut curr_node_idx = parent_idx;
                    let mut curr_node = parent;
                    for curr_parent_elem in path_elems {
                        // If path says there's a parent, there should be one...
                        let curr_parent_idx = curr_node.parent_idx.unwrap();
                        let curr_parent = &state.work_availability_tree[curr_parent_idx];
                        crate::debug!("Checking ancestor node #{curr_parent_idx} ({curr_parent:#?})");

                        // ...and it should know about us
                        let our_child_list = &curr_parent.node_children;
                        assert!(our_child_list.first_child_idx <= curr_node_idx);
                        let rel_node_idx = curr_node_idx - our_child_list.first_child_idx;
                        assert!(rel_node_idx < our_child_list.num_children());
                        let node_bit_idx = rel_node_idx;

                        // Check that path element is located correctly
                        assert_eq!(
                            curr_parent_elem,
                            &our_child_list.work_availability.bit(node_bit_idx)
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

    /// Arbitrary HierarchicalState + ordered list of worker indices
    ///
    /// This is used to test WorkAvailliabilityPath operation correctness
    fn state_and_worker_indices() -> impl Strategy<Value = (Arc<HierarchicalState>, Vec<usize>)> {
        hierarchical_state().prop_flat_map(|state| {
            let worker_indices =
                prop::collection::vec(0..state.workers.len(), SizeRange::default());
            worker_indices.prop_map(move |worker_indices| {
                for node in state.work_availability_tree.iter() {
                    node.clear_work_availability();
                }
                (state.clone(), worker_indices)
            })
        })
    }

    proptest! {
        /// Test WorkAvailabilityPath usage
        #[test]
        fn use_work_availability_path(
            (state, worker_indices) in state_and_worker_indices()
        ) {
            // Test harness setup
            crate::setup_logger_once();
            crate::info!("Testing WorkAvailabilityPath usage pattern {worker_indices:?} over {state:#?}");

            // Handle uniprocessor edge case
            if state.workers.len() == 1 {
                crate::debug!("Handling uniprocessor edge case");
                assert_eq!(state.work_availability_tree.len(), 0);
                let path = WorkAvailabilityPath::new(&state, 0);
                assert_eq!(path.num_ancestors(), 0);
                for worker_idx in worker_indices {
                    assert_eq!(worker_idx, 0);
                    assert_eq!(path.fetch_set(Ordering::Relaxed), None);
                    assert_eq!(path.fetch_clear(Ordering::Relaxed), None);
                }
                return Ok(());
            }

            // Precompute the work availability path of every worker, both in
            // its optimized WorkAvailabilityPath form and in its unoptimized
            // (node idx, child idx) form.
            let worker_paths = (0..state.workers.len()).map(|worker_idx| {
                crate::debug!("Building work availability path for worker {worker_idx}");

                // Compute optimized WorkAvailabilityPath
                let old_tree = state.work_availability_tree.clone();
                let path = WorkAvailabilityPath::new(&state, worker_idx);
                assert_eq!(state.work_availability_tree, old_tree);

                // Reference work availability path construction method
                let mut ancestor_nodes = Vec::with_capacity(path.num_ancestors() - 1);
                let worker_parent = state.work_availability_tree.iter().enumerate().rev().find_map(|(node_idx, node)| {
                    node.worker_children.child_bit_idx(worker_idx).map(|_| {
                        ((node_idx, node), worker_idx)
                    })
                }).unwrap();
                let ((worker_parent_idx, mut node), worker_idx) = worker_parent;
                let mut node_idx = worker_parent_idx;
                //
                while let Some(parent_idx) = node.parent_idx {
                    node = &state.work_availability_tree[parent_idx];
                    node.node_children.child_bit_idx(node_idx).unwrap();
                    ancestor_nodes.push((parent_idx, node_idx));
                    node_idx = parent_idx;
                };
                crate::debug!("Worker {worker_idx} has a raw (node, child) path \
                    composed of worker child ({worker_parent_idx}, {worker_idx}) \
                    followed by node children {ancestor_nodes:?}");

                (path, ((worker_parent_idx, worker_idx), ancestor_nodes))
            }).collect::<Vec<_>>();

            fn checked_path_op(
                inout_tree: &[Node],
                raw_path: &((usize, usize), Vec<(usize, usize)>),
                node_op: fn(&BitRef<'_, true>, Ordering) -> (bool, bool),
                expected_set: bool
            ) {
                let ((worker_parent_idx, worker_idx), ancestor_nodes) = raw_path;

                let worker_bit = inout_tree[*worker_parent_idx].worker_children.child_availability(*worker_idx).unwrap();
                let (was_set, empty_changed) = node_op(&worker_bit, Ordering::Relaxed);
                assert_eq!(was_set, expected_set);
                if !empty_changed {
                    return;
                }

                for (node_parent_idx, node_idx) in ancestor_nodes {
                    let node_bit = inout_tree[*node_parent_idx].node_children.child_availability(*node_idx).unwrap();
                    let (was_set, empty_changed) = node_op(&node_bit, Ordering::Relaxed);
                    assert_eq!(was_set, expected_set);
                    if !empty_changed {
                        return;
                    }
                }
            }


            // Toggle worker availability and check effect on state
            for worker_idx in worker_indices {
                crate::debug!("Toggling work availability of worker {worker_idx}");
                let (path, raw_path) = &worker_paths[worker_idx];
                let was_set = path.0[0].is_set(Ordering::Relaxed);
                let expected_tree = state.work_availability_tree.clone();
                if was_set {
                    assert_eq!(path.fetch_set(Ordering::Relaxed), Some(true));
                    assert_eq!(state.work_availability_tree, expected_tree);
                    assert_eq!(path.fetch_clear(Ordering::Relaxed), Some(true));
                    checked_path_op(&expected_tree, raw_path, |bit, ordering| bit.clear_and_check_emptied(ordering), true);
                } else {
                    assert_eq!(path.fetch_clear(Ordering::Relaxed), Some(false));
                    assert_eq!(state.work_availability_tree, expected_tree);
                    assert_eq!(path.fetch_set(Ordering::Relaxed), Some(false));
                    checked_path_op(&expected_tree, raw_path, |bit, ordering| bit.check_empty_and_set(ordering), false);
                }
                crate::trace!("Tree is now in state {:#?}", state.work_availability_tree);
                assert_eq!(state.work_availability_tree, expected_tree);
            }
        }
    }
}
