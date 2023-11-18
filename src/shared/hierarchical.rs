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
        // Restrict topology to desired affinity mask, and check if the residual
        // cpuset fits within implementation limits.
        let affinity = affinity.borrow();
        crate::debug!("Setting up a thread pool with affinity {affinity}");
        let topology = {
            let mut restricted = topology.clone();
            restricted.edit(|editor| {
                editor
                    .restrict(affinity, RestrictFlags::REMOVE_EMPTIED)
                    .expect("failed to restrict topology to affinity mask")
            });
            restricted
        };
        let cpuset = topology.cpuset();
        let num_workers = cpuset.weight().unwrap();
        crate::debug!("Restricted topology has cpuset {cpuset} containing {num_workers} CPUs");
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
        let mut curr_child_objects = vec![ChildObjects {
            parent_idx: None,
            objects: vec![topology.root_object()],
        }];
        let mut next_child_objects = Vec::new();
        //
        // Pool of old allocations to be reused
        let mut objects_vec_morgue = Vec::<Vec<&TopologyObject>>::new();
        //
        // Length of the tree array at previous depth
        let mut last_depth_start = 0;
        'depth: while !curr_child_objects.is_empty() {
            // Log current tree-building status, if enabled
            crate::debug!("Constructing a new tree depth");
            crate::debug!(
                "Tree at current depth:\n    \
                worker_configs: {worker_configs:?},\n    \
                worker_interfaces: {worker_interfaces:?},\n    \
                work_availability_tree: {work_availability_tree:?}"
            );
            let curr_depth_start = work_availability_tree.len();

            // At each depth, topology objects are grouped into child sets that
            // represent group of children associated with the same parent
            // work_availability_tree node.
            for mut child_object_set in curr_child_objects.drain(..) {
                // Log current tree-building status, if enabled
                crate::debug!(
                    "Will now insert children of node {:?}",
                    child_object_set.parent_idx
                );
                if child_object_set.parent_idx.is_none() {
                    assert_eq!(
                        work_availability_tree.len(),
                        0,
                        "Only the very first node of the tree may not have parents"
                    );
                }

                // Track worker and node children
                //
                // Children that manage a single CPU are not translated into
                // nodes of `work_availability_tree`, instead they are
                // translated into workers that are attached to leaf nodes of
                // `work_availability_tree` that manage multiple CPUs.
                //
                // We do not add worker children to their parent node eagerly
                // because if a parent node has both worker and node children
                // (as happens on e.g. Intel Adler Lake), we need to take the
                // special action of creating an extra work availability node to
                // which all workers will be attached. And we may not know if a
                // parent node with worker children also has node children until
                // the last child of this parent is observed, thus we need to
                // buffer worker children until the last node child is observed.
                //
                // This tree building contortion of creating artificial node
                // children does add a little headache at this tree building
                // time, but in exchange it ensures that the final tree will
                // have a more regular structure and thus will be more
                // easy/efficient to process (e.g. no need to track a node's
                // children individually).
                let first_child_node_idx = work_availability_tree.len();
                assert_eq!(worker_interfaces.len(), worker_configs.len());
                let first_child_worker_idx = worker_configs.len();

                // Iterate over topology objects from the current child set
                'objects: for mut object in child_object_set.objects.drain(..) {
                    // Log current tree-building status, if enabled
                    crate::debug!("Attempting to insert child {object:?}");

                    // Discriminate between different kinds of children
                    'single_child: loop {
                        let cpuset = object
                            .cpuset()
                            .expect("root object and its normal children should have cpusets");
                        let num_cpus = cpuset
                            .weight()
                            .expect("topology objects should have finite cpusets");
                        match num_cpus {
                            // Objects that manage no CPU should have been filtered
                            // out by the restricted topology preparation and the
                            // exclusion of empty affinity masks above.
                            0 => unreachable!("objects without CPUs should have been filtered out by TopologyEditor::restrict()"),

                            // Objects that manage a single CPU will be directly
                            // attached to their parent as workers. We do not do it
                            // eagerly because if a parent has both direct worker
                            // children and other children with an internal
                            // structure (as happens on e.g. Intel Adler Lake), we
                            // may need to create an artificial child node so that
                            // the the parent only has node children.
                            1 => {
                                let cpu = cpuset.first_set().expect("cpusets with weight == 1 should have one entry");
                                crate::debug!("Child will be the worker child or grandchild for CPU {cpu}");
                                let (interface, work_queue) = WorkerInterface::with_work_queue();
                                worker_interfaces.push(CachePadded::new(interface));
                                worker_configs.push(WorkerConfig { work_queue, cpu });
                                continue 'objects;
                            }

                            // Since the hwloc topology tree has a single root
                            // and one leaf per logical CPU, objects that manage
                            // multiple CPUs must have an object with multiple
                            // normal children somewhere below them. However,
                            // this may not be their direct child, but some
                            // grand-child, grand-grand-child... we handle this
                            // by iterating over single children as necessary.
                            _multiple => if object.normal_children().count() == 1 {
                                crate::debug!("Not a meaningful tree node as it only has one child, recursing to grandchild...");
                                object = object.normal_children().next().expect("cannot happen if count is 1");
                                assert_eq!(
                                    object.cpuset().expect("normal children should have cpusets"),
                                    cpuset,
                                    "if an object has a single child, the child should have the same cpuset as the parent"
                                );
                                continue 'single_child;
                            } else {
                                break 'single_child;
                            }
                        }
                    }

                    // If control reached this point, then we have a hwloc
                    // topology tree node with multiple normal children on our
                    // hands. Translate this node into a
                    // `work_availability_tree` node.
                    let num_children = object.normal_children().count();
                    crate::debug!("Will now turn child object with {num_children} children into a new tree node");
                    assert!(
                        num_children > 1,
                        "this point should only be reached for multiple-children nodes"
                    );
                    let new_node_idx = work_availability_tree.len();
                    work_availability_tree.push(Node {
                        parent_idx: child_object_set.parent_idx,
                        // Cannot fill in the details of the node until we've
                        // processed the children, but placeholder them in such
                        // a way that users of the tree will bomb if we somehow
                        // fail to fill in the details later.
                        work_availability: AtomicFlags::new(0),
                        children: ChildrenLink::Nodes {
                            first_node_idx: usize::MAX,
                        },
                    });

                    // Collect children to be processed once we move to the next
                    // tree depth (next iteration of the 'depth loop).
                    let mut children = ChildObjects {
                        parent_idx: Some(new_node_idx),
                        objects: objects_vec_morgue.pop().unwrap_or_default(),
                    };
                    children.objects.extend(object.normal_children());
                    next_child_objects.push(children);
                }

                // Done processing this child list, recycle its inner allocation
                objects_vec_morgue.push(std::mem::take(&mut child_object_set.objects));

                // At this point, we have processed all children of the active
                // work availability tree node, and it's time for the moment of
                // truth: does this node only have node children, only worker
                // children, or both?
                let num_worker_children = worker_configs
                    .len()
                    .checked_sub(first_child_worker_idx)
                    .expect("numboer of workers can only increase");
                let num_node_children = work_availability_tree
                    .len()
                    .checked_sub(first_child_node_idx)
                    .expect("number of tree nodes can only increase");
                crate::debug!(
                    "Done processing children of node {:?}, with a total of {num_worker_children} workers and {num_node_children} nodes",
                    child_object_set.parent_idx
                );

                // In any case

                match (num_node_children, num_worker_children) {
                    // We shouldn't create useless tree nodes without children
                    (0, 0) => {
                        unreachable!("nodes without children shouldn't have made it this far")
                    }

                    // If we only have worker children, we can add them as
                    // direct children to this node
                    (0, num_workers) => {
                        crate::debug!("Parent node only has worker children: attach them directly");

                        // Handle uniprocessor edge case of a single worker
                        // without parents
                        let Some(parent_idx) = child_object_set.parent_idx else {
                            assert_eq!(
                                num_workers, 1,
                                "this branch can only be reached on uniprocessor systems"
                            );
                            assert_eq!(
                                num_worker_children, 1,
                                "should cover the system's only CPU/worker and no more"
                            );
                            crate::debug!(
                                "Reached uniprocessor edge case of a single worker without parents"
                            );
                            break 'depth;
                        };

                        // In the normal case, workers have a parent, configure
                        // it now that its full child list is known.
                        assert!(
                            parent_idx >= last_depth_start && parent_idx < curr_depth_start,
                            "parent node should belong to last tree layer"
                        );
                        let parent = &mut work_availability_tree[parent_idx];
                        parent.work_availability = AtomicFlags::new(num_workers);
                        parent.children = ChildrenLink::Workers {
                            first_worker_idx: first_child_worker_idx,
                        };
                    }

                    // If this parent has at least one node child, then it will
                    // only have node children: create a new node to contain
                    // worker children if needed, then configure parent to point
                    // at all of these nodes
                    (mut num_nodes, num_workers) => {
                        // Create a new node child to hold workers if need be
                        if num_workers > 0 {
                            crate::debug!("Parent node has both node and worker children, create an artificial extra node child to hold workers");
                            work_availability_tree.push(Node {
                                parent_idx: child_object_set.parent_idx,
                                work_availability: AtomicFlags::new(num_workers),
                                children: ChildrenLink::Nodes {
                                    first_node_idx: first_child_worker_idx,
                                },
                            });
                            num_nodes += 1;
                        }

                        crate::debug!(
                            "Now parent only has node children, update it to point to them"
                        );
                        if let Some(parent_idx) = child_object_set.parent_idx {
                            // FIXME: Deduplicate wrt above
                            assert!(
                                parent_idx >= last_depth_start && parent_idx < curr_depth_start,
                                "parent node should belong to last tree layer"
                            );
                            let parent = &mut work_availability_tree[parent_idx];
                            parent.work_availability = AtomicFlags::new(num_nodes);
                            parent.children = ChildrenLink::Nodes {
                                first_node_idx: first_child_node_idx,
                            };
                        }
                    }
                }
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
        (result, worker_configs.into())
    }

    // TODO: Finish constructor, then rest
}

/// Node of `HierarchicalState::work_availability_tree`
#[derive(Debug)]
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
#[derive(Debug)]
enum ChildrenLink {
    /// Children are normal tree nodes, starting at this index in
    /// `HierarchicalState::tree`
    Nodes { first_node_idx: usize },

    /// Children are workers, with public interfaces starting at this index in
    /// `HierarchicalState::workers`
    Workers { first_worker_idx: usize },
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
