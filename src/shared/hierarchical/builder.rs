//! Hierarchical state building

use super::{Child, ChildrenLink, HierarchicalState, Node, ParentLink};
use crate::shared::{futex::WorkerFutex, WorkerConfig, WorkerInterface};
use crossbeam::{deque::Injector, utils::CachePadded};
use hwlocality::{bitmap::BitmapIndex, cpu::cpuset::CpuSet, object::TopologyObject, Topology};
use std::{assert_ne, sync::Arc};

/// [`HierarchicalState`] that's in the process of being built
pub(crate) struct HierarchicalStateBuilder {
    /// Affinity of the thread pool being built
    cpuset: CpuSet,

    /// Configurations for future workers attached to the tree
    worker_configs: Vec<WorkerConfig<HierarchicalState>>,

    /// Public interfaces of workers attached to the tree
    worker_interfaces: Vec<CachePadded<Child<WorkerInterface<HierarchicalState>>>>,

    /// Tree for local work availability signaling
    work_availability_tree: Vec<Child<Node>>,
}
//
impl HierarchicalStateBuilder {
    /// Start building a [`HierarchicalState`] for a certain number of workers
    pub fn new(topology: &Topology, affinity: &CpuSet) -> Self {
        // Check if the affinity-restricted topology cpuset fits within
        // implementation limits and results in a valid pool
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

        // Initialize inner data stores
        Self {
            cpuset,
            worker_configs: Vec::with_capacity(num_workers),
            worker_interfaces: Vec::with_capacity(num_workers),
            work_availability_tree: Vec::new(),
        }
    }

    /// Create a builder and fill its inner vecs following a certain topology
    /// woth an associated affinity mask
    pub fn from_topology_affinity(topology: &Topology, affinity: &CpuSet) -> Self {
        // Start building the tree
        let mut result = Self::new(topology, affinity);

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
            result.trace_state();

            // End of tree nodes for parent depth
            let first_node_at_this_depth = result.next_node_idx();

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
                let first_child_node_idx = result.next_node_idx();
                let first_child_worker_idx = result.next_worker_idx();

                // Iterate over topology objects from the current child set
                for object in child_object_set.objects {
                    if let Some(child_objects) =
                        result.add_object(child_object_set.parent_idx, object)
                    {
                        next_child_objects.push(child_objects);
                    }
                }

                // At this point, we have collected all children of the active
                // work availability tree node
                let num_worker_children = result
                    .num_workers()
                    .checked_sub(first_child_worker_idx)
                    .expect("number of workers can only increase");
                let num_node_children = result
                    .num_nodes()
                    .checked_sub(first_child_node_idx)
                    .expect("number of tree nodes can only increase");
                crate::debug!(
                    "Done processing children of node {:?}, with a total of {num_worker_children} worker(s) and {num_node_children} node(s)",
                    child_object_set.parent_idx
                );

                // Handle edge cases without a parent node
                let Some(parent_idx) = child_object_set.parent_idx else {
                    match result.num_nodes() {
                        0 => {
                            assert!(
                                result.expected_workers() == 1
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
                result.setup_children(
                    parent_idx,
                    ChildrenLink::new(first_child_worker_idx, num_worker_children),
                    ChildrenLink::new(first_child_node_idx, num_node_children),
                );
            }

            // Done with this tree layer, go to next depth
            std::mem::swap(&mut curr_child_objects, &mut next_child_objects);
            first_node_at_prev_depth = first_node_at_this_depth;
        }
        result
    }

    /// Number of workers expected at the end of tree building
    pub fn expected_workers(&self) -> usize {
        assert_eq!(
            self.worker_configs.capacity(),
            self.worker_interfaces.capacity()
        );
        self.worker_configs.capacity()
    }

    /// Number of workers currently present in the tree
    pub fn num_workers(&self) -> usize {
        assert_eq!(self.worker_configs.len(), self.worker_interfaces.len());
        self.worker_configs.len()
    }

    /// Index that the next inserted worker will get
    pub fn next_worker_idx(&self) -> usize {
        self.num_workers()
    }

    /// Number of nodes currently present in the tree
    pub fn num_nodes(&self) -> usize {
        self.work_availability_tree.len()
    }

    /// Index that the next inserted tree node will get
    pub fn next_node_idx(&self) -> usize {
        self.num_nodes()
    }

    /// Dump the current builder state as a trace log
    pub fn trace_state(&self) {
        crate::trace!(
            "Tree at current depth:\n  \
            worker_configs: {:?},\n  \
            work_availability_tree: {:#?}",
            self.worker_configs,
            self.work_availability_tree
        );
    }

    /// Add a [`TopologyObject`] to the tree
    ///
    /// Only normal objects (i.e. the root object + all of its transitive normal
    /// children) should be passed into this method.
    ///
    /// If the [`TopologyObject`] is translated into a tree node, then provide a
    /// list of children to be subsequently attached to that tree node, once we
    /// start processing the next tree depth. Once all children are attached,
    /// call `setup_children()` to finalize the tree node.
    pub fn add_object<'object>(
        &mut self,
        parent: ParentLink,
        mut object: &'object TopologyObject,
    ) -> Option<ChildObjects<'object>> {
        // Parameter validation
        assert!(object.object_type().is_normal());

        // Only the very first object of the tree will not have a parent, all
        // subsequent objects should have a parent.
        assert!(parent.is_some() || (self.num_workers() == 0 && self.num_nodes() == 0));

        // Discriminate between different kinds of children
        let builder_cpuset = self.cpuset.clone();
        let restricted_children = 'single_child: loop {
            // Log current tree-building status, if enabled
            crate::trace!("Evaluating child {object} for insertion in the tree");

            // Count object CPUs
            let object_cpuset = self.restrict_cpuset(object);
            let num_cpus = object_cpuset
                .weight()
                .expect("topology objects should have finite cpusets");
            crate::trace!(
                "Child has affinity-constrained cpuset {object_cpuset} containing {num_cpus} CPU(s)"
            );

            // Classify object accordingly
            match num_cpus {
                // Children without CPUs should be ignored earlier
                0 => unreachable!(
                    "children without CPUs should have been weeded out in previous steps"
                ),

                // Single-CPU children are attached as workers
                1 => {
                    let cpu = object_cpuset
                        .first_set()
                        .expect("cpusets with weight == 1 should have one entry");
                    self.add_worker(parent, cpu);
                    return None;
                }

                // Multi-CPU children contain multiple branches down
                // their children subtree. We'll recurse down to
                // that branch and add it as a node child.
                _multiple => {
                    // Filter out child objects by affinity
                    let mut restricted_children = object.normal_children().filter(|child| {
                        let child_cpuset =
                            child.cpuset().expect("normal children should have cpusets");
                        child_cpuset.intersects(&builder_cpuset)
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
                        .expect("child count can't be zero if object has non-empty cpuset");
                    let child_cpuset = self.restrict_cpuset(object);
                    assert_eq!(
                        child_cpuset,
                        object_cpuset,
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
            "only objects with multiple children should become nodes"
        );
        let new_node_idx = self.add_node(parent);

        // Collect children to be processed once we move to the next
        // tree depth (next iteration of the 'depth loop).
        Some(ChildObjects {
            parent_idx: Some(new_node_idx),
            objects: restricted_children.collect(),
        })
    }

    /// Set up a node's children
    ///
    /// This should only be done once per node, at the point where all the
    /// children of the node have been inserted.
    pub fn setup_children(
        &mut self,
        node_idx: usize,
        worker_children: ChildrenLink,
        node_children: ChildrenLink,
    ) {
        // Basic input validation
        assert!(node_idx < self.num_nodes());
        let check_children = |link: &ChildrenLink, current_len: usize| {
            assert!(link.first_child_idx + link.num_children() <= current_len);
        };
        check_children(&worker_children, self.num_workers());
        check_children(&node_children, self.num_nodes());

        // Nodes should have at least two children, because nodes without
        // children can be removed entirely and nodes with only one child can be
        // simplified by replacing them with their single child.
        let num_children = node_children.num_children() + worker_children.num_children();
        assert!(
            num_children >= 2,
            "nodes with < 2 children should be filtered out"
        );

        // Set the node's children descriptors
        let node = &mut self.work_availability_tree[node_idx];
        node.object.worker_children = worker_children;
        node.object.node_children = node_children;
    }

    /// Finish building the tree
    pub fn build(
        self,
    ) -> (
        Arc<HierarchicalState>,
        Box<[WorkerConfig<HierarchicalState>]>,
    ) {
        // Make sure the final state meets expectations
        assert_eq!(self.worker_configs.len(), self.expected_workers());
        assert_eq!(self.worker_interfaces.len(), self.expected_workers());

        // Set up the global shared state
        let result = Arc::new(HierarchicalState {
            injector: Injector::new(),
            workers: self.worker_interfaces.into(),
            work_availability_tree: self.work_availability_tree.into(),
        });
        crate::debug!("Final worker configs: {:?}", self.worker_configs);
        crate::debug!("Final tree: {result:#?}");
        (result, self.worker_configs.into())
    }

    /// Affinity mask
    fn cpuset(&self) -> &CpuSet {
        &self.cpuset
    }

    /// Apply our affinity mask to the cpuset of a [`TopologyObject`] proposed
    /// for insertion into the tree
    fn restrict_cpuset(&self, object: &TopologyObject) -> CpuSet {
        assert!(object.object_type().is_normal());
        let cpuset = object.cpuset().expect("normal objects should have cpusets");
        cpuset & (&self.cpuset)
    }

    /// Add a worker to the tree
    ///
    /// The worker knows about its parent from the start, but the parent node
    /// will only know the list of its worker children once all of that node's
    /// children have been processed.
    ///
    /// Returns index of the freshly inserted worker.
    fn add_worker(&mut self, parent: ParentLink, cpu: BitmapIndex) -> usize {
        assert!(self.num_workers() < self.expected_workers());
        assert!(self.cpuset.is_set(cpu));
        let (config, interface) = crate::shared::new_worker(cpu);
        let result = self.next_worker_idx();
        self.worker_configs.push(config);
        self.worker_interfaces.push(CachePadded::new(Child {
            parent,
            object: interface,
        }));
        result
    }

    /// Add a node to the tree
    ///
    /// Children are not attached yet, they will be attached once we start
    /// processing the next tree depth.
    ///
    /// Returns index of the freshly inserted node.
    fn add_node(&mut self, parent: ParentLink) -> usize {
        let result = self.next_node_idx();
        self.work_availability_tree.push(Child {
            parent,
            object: Node {
                // Cannot fill in the details of the node until we've processed the
                // children, but we have a "no children yet" placeholder state.
                node_children: ChildrenLink::default(),
                worker_children: ChildrenLink::default(),
            },
        });
        result
    }
}

/// Topology objects that we need to eventually add to the tree
///
/// Because tree nodes are inserted in breadth-first order, at each step of tree
/// building we will be processing a list of child [`TopologyObject`]s at the
/// current depth and building a list of [`TopologyObject`]s at the next depth.
#[derive(Debug)]
pub(crate) struct ChildObjects<'topology> {
    /// Parent node, if any
    pub parent_idx: ParentLink,

    /// Hwloc objects attached to this parent node
    pub objects: Vec<&'topology TopologyObject>,
}
