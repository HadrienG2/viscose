//! Iteration over PUs in a manner that respects work distribution priorities

use self::node::{Child, NodeIterator, RemoveOutcome};
use super::Priority;
use hwlocality::{
    cpu::cpuset::CpuSet,
    object::{types::ObjectType, TopologyObject, TopologyObjectID},
    Topology,
};
use std::{
    collections::{HashMap, VecDeque},
    fmt::Debug,
    ptr,
};

/// Tree used to iterate over PUs in an order that matches work distribution
/// priorities (i.e. PUs that should share work first are close)
#[derive(Default)]
pub struct PUIterator<'topology> {
    /// Summarized hwloc tree that only includes multi-children nodes, and
    /// priorizes the traversal of each child from each node
    nodes: HashMap<TopologyObjectID, NodeIterator<'topology>>,

    /// Path to the next PU to be yielded, if any
    next_pu_path: Vec<TopologyObjectID>,
}
//
impl<'topology> PUIterator<'topology> {
    /// Set up iteration over PUs
    pub fn new(
        topology: &'topology Topology,
        affinity: &CpuSet,
        parent_priorities: &HashMap<TopologyObjectID, Priority>,
    ) -> Self {
        // This iterator will operate in a simplified hwloc topology that only
        // includes nodes from the affinity set that have either at least one PU
        // children or multiple non-PU children
        let id = |obj: &TopologyObject| obj.global_persistent_index();
        let children = |obj| Self::children_in_cpuset(obj, affinity);
        let has_multiple_children = |obj| children(obj).count() >= 2;
        let has_pu_children =
            |obj| children(obj).any(|child| child.object_type() == ObjectType::PU);
        let keep_node = |obj| !(has_multiple_children(obj) || has_pu_children(obj));

        // Find the root of the simplified topology tree, if any
        let mut root = topology.root_object();
        while !keep_node(root) {
            debug_assert!(
                children(root).count() <= 1,
                "rejected root candidate should have 0 or 1 children"
            );
            if let Some(only_child) = children(root).next() {
                root = only_child;
                continue;
            } else {
                // No accepted root means no PU, i.e. an empty iterator
                return Self::default();
            }
        }

        // Start building the topology tree by inserting the root
        let mut nodes = HashMap::new();
        let priority = |parent| parent_priorities.get(&id(parent)).copied();
        nodes.insert(id(root), NodeIterator::new(priority(root)));

        // Build the tree from top to bottom
        //
        // At each point of the tree building process, we have sets of nodes
        // considered for inclusion in the tree, grouped by parent node.
        let mut curr_node_candidates = vec![(id(root), children(root).collect::<Vec<_>>())];
        let mut next_node_candidates = Vec::new();
        let mut children_morgue = Vec::new();
        while !curr_node_candidates.is_empty() {
            // We grab the associated parent object of each group...
            for (parent_id, mut curr_children) in curr_node_candidates.drain(..) {
                const PARENT_SHOULD_BE_THERE: &str =
                    "parent should already be present because traversal is top-down";
                // ...then process this parent object's children...
                'children: for child in curr_children.drain(..) {
                    // PUs always get added to parents
                    if child.object_type() == ObjectType::PU {
                        nodes
                            .get_mut(&parent_id)
                            .expect(PARENT_SHOULD_BE_THERE)
                            .add_child(child);
                        assert_eq!(child.normal_arity(), 0, "PUs shouldn't have children");
                        continue 'children;
                    }

                    // Nodes only get added if they pass the node filter
                    let parent_id = if keep_node(child) {
                        nodes.insert(id(child), NodeIterator::new(priority(child)));
                        nodes
                            .get_mut(&parent_id)
                            .expect(PARENT_SHOULD_BE_THERE)
                            .add_child(child);
                        id(child)
                    } else {
                        parent_id
                    };

                    // Grandchildren will be attached to the newly created node
                    // if it was kept, otherwise to its last kept ancestor
                    let mut next_children: Vec<_> = children_morgue.pop().unwrap_or_default();
                    next_children.extend(children(child));
                    next_node_candidates.push((parent_id, next_children));
                }
                children_morgue.push(curr_children);
            }
            std::mem::swap(&mut curr_node_candidates, &mut next_node_candidates);
        }
        Self {
            nodes,
            next_pu_path: vec![id(root)],
        }
    }

    /// Select normal children of a node that match the affinity mask
    fn children_in_cpuset<'iterator, 'parent: 'iterator>(
        parent: &'parent TopologyObject,
        affinity: &'iterator CpuSet,
    ) -> impl DoubleEndedIterator<Item = &'parent TopologyObject> + Clone + 'iterator {
        parent
            .normal_children()
            .filter(move |child| child.cpuset().unwrap().intersects(affinity))
    }
}
//
impl<'topology> Iterator for PUIterator<'topology> {
    type Item = &'topology TopologyObject;

    fn next(&mut self) -> Option<Self::Item> {
        // Dive down from the active node to find the next PU that we'll
        // eventually yield
        let mut current_node_id = *self.next_pu_path.last()?;
        const PATH_ERROR: &str = "node in next_pu_path should exist";
        let mut current_node = self.nodes.get_mut(&current_node_id).expect(PATH_ERROR);
        let yielded_pu = loop {
            match current_node.current_child() {
                Child::Node(next_node_id) => {
                    current_node_id = *next_node_id;
                    self.next_pu_path.push(current_node_id);
                    current_node = self
                        .nodes
                        .get_mut(&current_node_id)
                        .expect("child node should exist");
                    continue;
                }
                Child::PU(pu) => {
                    break *pu;
                }
            }
        };

        // Remove the PU and every ancestor node that it transitively emptied by
        // this operation, until we get to an ancestor node that still has more
        // children to yield and merely switches to the next of its children.
        let mut downstream_priority = loop {
            match current_node.remove_child() {
                RemoveOutcome::NewChild(priority) => break priority,
                RemoveOutcome::Empty => {
                    self.next_pu_path.pop();
                    self.nodes.remove(&current_node_id);
                    let Some(next_node_id) = self.next_pu_path.last() else {
                        // We deleted all nodes in the path, so there is no
                        // iteration state left to be advanced...
                        assert!(
                            self.nodes.is_empty(),
                            "no path left should mean all nodes have been fully processed"
                        );
                        return Some(yielded_pu);
                    };
                    current_node_id = *next_node_id;
                    current_node = self.nodes.get_mut(&current_node_id).expect(PATH_ERROR);
                    continue;
                }
            }
        };

        // Next, walk up the remaining ancestors and figure out if any other
        // ancestor should switch to its next child.
        //
        // This is needed to correctly handle situations where objects down the
        // hierarchy have lower filling priority than the objects above them.
        //
        // For example, if you (wisely) think that filling up Cores below L3
        // caches is more important than filling up hyperthreaded PUs below
        // Cores, then you don't want to only switch to the next PU of the
        // current core, you want to switch to the next core as well.
        //
        // Similarly, if you are memory-bound and think that covering NUMA nodes
        // is more important than covering cores below them, you will not just
        // want to switch to the next PU/Core below the active NUMA node, but
        // also to switch to the next NUMA node on every iteration.
        let mut valid_path_len = self.next_pu_path.len();
        for (num_ancestors, &ancestor_id) in self.next_pu_path.iter().enumerate().rev().skip(1) {
            current_node_id = ancestor_id;
            current_node = self.nodes.get_mut(&current_node_id).expect(PATH_ERROR);
            if let Some(switch_priority) =
                current_node.switch_child_if_priorized(downstream_priority)
            {
                downstream_priority = switch_priority;
                // Switching children here invalidates the rest of the PU path
                valid_path_len = num_ancestors + 1;
            }
        }
        self.next_pu_path.truncate(valid_path_len);
        Some(yielded_pu)
    }
}

/// Iteration over nodes of the [`PUIterator`] tree
mod node {
    use super::*;

    /// Node of the [`PUIterator`] tree
    pub struct NodeIterator<'topology> {
        /// Remaining children in the current run over the children list
        ///
        /// This deque should almost always contain children. The only circumstances
        /// where it's okay for it to be empty is when a node has just been created
        /// and children have not been added yet, or when the last child of the node
        /// has just been deleted and the node is to be deleted right afterwards.
        current_children: VecDeque<Child<'topology>>,

        /// Children that have already been yielded in this run of iteration
        ///
        /// Will be fed back into the `current_children` deque once we wrap around
        /// the children list. This dual-queue system lets us detect when we reach
        /// the end of the child list and wrap around to the beginning.
        next_children: Vec<Child<'topology>>,

        /// How important it is to distribute work across children of this node
        ///
        /// Only nodes with a single child can leave this at `None`.
        normal_child_priority: Option<Priority>,

        /// What switching back to the first child of this Node amounts to in terms
        /// of work distribution priority
        ///
        /// For example, on x86, after iterating over all the Cores of an L3 cache,
        /// switching back to the first core amounts to switching to a new PU within
        /// a core we've already visited.
        ///
        /// This is set when switching away from the first child, read when
        /// contemplating switching back to the first child, and cleared when
        /// actually switching back to the first child.
        first_child_priority: Option<Priority>,
    }
    //
    impl<'topology> NodeIterator<'topology> {
        /// Create an empty node with an optional child-switching priority
        ///
        /// You should then add at least one child using `add_child()`. After that
        /// you can use the other methods of this type.
        ///
        /// Nodes that have at least two children should additionally get a
        /// non-`None` child-switching priority.
        pub fn new(child_switch_priority: Option<Priority>) -> Self {
            Self {
                current_children: VecDeque::new(),
                next_children: Vec::new(),
                normal_child_priority: child_switch_priority,
                first_child_priority: None,
            }
        }

        /// Add a child at the end of this node's child list
        pub fn add_child(&mut self, child: &'topology TopologyObject) {
            let child = Child::from(child);
            assert!(
                self.current_children
                    .iter()
                    .chain(&self.next_children)
                    .all(|&obj| obj != child),
                "attempted to add the same child twice"
            );
            self.current_children.push_back(child);
            if self.current_children.len() > 1 {
                assert!(
                    self.normal_child_priority.is_some(),
                    "multi-children nodes should have a child-switching priority"
                );
            }
        }

        /// Check out the active child of this node
        pub fn current_child(&self) -> &Child<'topology> {
            self.check_child_access();
            self.current_children
                .front()
                .expect("enforced by check_child_access")
        }

        /// Remove the active child from this node
        ///
        /// If this returns [`RemoveOutcome::Empty`], you should stop using this
        /// node, discard it, and remove it from the child list of ancestor nodes,
        /// recursing over ancestors until reaching a node where other children
        /// remain after the deletion operation.
        ///
        /// Otherwise, this simply amounts to switching to the next child, and you
        /// get the priority of the associated child-switching event.
        #[must_use]
        pub fn remove_child(&mut self) -> RemoveOutcome {
            // Switch to the next child, discarding the active child in the process
            self.check_child_access();
            let switch_priority = self.child_switch_priority();
            self.pop_child();

            // Check if we removed the last child of this node
            if self.current_children.is_empty() {
                assert!(
                    self.next_children.is_empty(),
                    "pop_child should spill next_children to current_children when necessary"
                );
                RemoveOutcome::Empty
            } else {
                RemoveOutcome::NewChild(
                    switch_priority
                        .expect("multi-children nodes should have a child-switching priority"),
                )
            }
        }

        /// Switch to the next child of this node
        ///
        /// Called on multi-children ancestors after one node downstream of their
        /// active child has switched to its next child.
        ///
        /// Receives the priority of the highest-priority child switch that has
        /// occured downstream. If this node does switch to its next child, an
        /// updated child switch priority is returned, to be used as the new
        /// `downstream_priotity` when iterating over higher ancestors.
        #[must_use]
        pub fn switch_child_if_priorized(
            &mut self,
            downstream_priority: Priority,
        ) -> Option<Priority> {
            // Check if it is time for us to switch to our next child
            self.check_child_access();
            let switch_priority = self
                .child_switch_priority()
                .expect("multi-children nodes should have a child-switching priority");
            if switch_priority <= downstream_priority {
                return None;
            }

            // If we're on the first child, remember the priority of the downstream
            // child switch in order to use it on the next child list wraparound
            if self.on_first_child() {
                self.first_child_priority = Some(downstream_priority);
            }

            // Switch to the next child, schedule looking at this child later
            let active_child = self.pop_child();
            self.current_children.push_back(active_child);
            Some(switch_priority)
        }

        /// Extract the current child of the node
        fn pop_child(&mut self) -> Child<'topology> {
            // Get the current child
            self.check_child_access();
            let child = self
                .current_children
                .pop_front()
                .expect("enforced by check_child_access");

            // Handle wraparound from last child to first child
            if self.current_children.is_empty() {
                self.current_children.extend(self.next_children.drain(..));
                self.first_child_priority = None;
            }
            child
        }

        /// Assess the priority of switching to the next child of this node
        ///
        /// The output may only be `None` when the node only has a single child.
        fn child_switch_priority(&self) -> Option<Priority> {
            self.check_child_access();
            if self.on_last_child() {
                self.first_child_priority
            } else {
                self.normal_child_priority
            }
        }

        /// Truth that the active child is our first child
        fn on_first_child(&self) -> bool {
            self.check_child_access();
            self.next_children.is_empty()
        }

        /// Truth that the active child is our last child
        fn on_last_child(&self) -> bool {
            self.check_child_access();
            self.current_children.len() == 1
        }

        /// Check for methods that should only be called when a node has children
        fn check_child_access(&self) {
            assert!(
                !self.current_children.is_empty(),
                "this method should not be called on a node without children"
            );
        }
    }

    /// Child of a [`NodeIterator`]
    //
    // --- Implementation notes ---
    //
    // This type doesn't work in the presence of objects from multiple topology, but
    // we're not exposing this type to the outside world so we don't care.
    #[derive(Copy, Clone, Debug)]
    pub enum Child<'topology> {
        /// Processing unit leaf
        PU(&'topology TopologyObject),

        /// Lower-level [`Node`]
        Node(TopologyObjectID),
    }
    //
    impl Eq for Child<'_> {}
    //
    impl<'topology> From<&'topology TopologyObject> for Child<'topology> {
        fn from(obj: &'topology TopologyObject) -> Self {
            if obj.object_type() == ObjectType::PU {
                Self::PU(obj)
            } else {
                debug_assert!(obj.normal_arity() >= 1);
                Self::Node(obj.global_persistent_index())
            }
        }
    }
    //
    impl PartialEq for Child<'_> {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::PU(pu1), Self::PU(pu2)) => ptr::eq(pu1, pu2),
                (Self::Node(id1), Self::Node(id2)) => id1 == id2,
                (Self::PU(_), Self::Node(_)) | (Self::Node(_), Self::PU(_)) => false,
            }
        }
    }

    /// Outcome of removing a [`Child`] from a [`NodeIterator`]
    pub enum RemoveOutcome {
        /// Switched to the next child, which has the specified priority
        NewChild(Priority),

        /// Removed the last child, this [`Node`] should be removed from the tree
        Empty,
    }
}
