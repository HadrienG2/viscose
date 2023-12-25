//! Iteration over PUs in a manner that respects work distribution priorities

use self::node::{Child, NodeIterator, RemoveOutcome};
use super::Priority;
use hwlocality::{
    cpu::cpuset::CpuSet,
    object::{types::ObjectType, TopologyObject, TopologyObjectID},
    Topology,
};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fmt::Debug,
    ptr,
};

/// Tree used to iterate over PUs in an order that matches work distribution
/// priorities (i.e. PUs that should share work first are close)
#[derive(Clone, Debug, Default)]
pub struct PUIterator<'topology> {
    /// Summarized hwloc tree that only includes multi-children nodes, and
    /// priorizes the traversal of each child from each node
    nodes: HashMap<TopologyObjectID, NodeIterator<'topology>>,

    /// Path to the next PU to be yielded, if any
    ///
    /// The path may not always be complete, but as long as there are PUs to be
    /// yielded, it should always feature at least one node
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
        // Handle empty cpuset case
        crate::debug!(
            "Will now set up iterator over topology PUs with affinity mask {affinity} \
            and work distribution priorities {:#?}",
            parent_priorities
                .iter()
                .map(|(id, priority)| (*id, *priority))
                .collect::<BTreeMap<_, _>>()
        );
        if !topology.cpuset().intersects(affinity) {
            crate::info!("No PUs to iterate over, will just yield empty iterator");
            return Self::default();
        }

        // We'll simplify the topology by collapsing chains of nodes with a
        // single child into a single node
        let children_in_cpuset = |obj| crate::children_in_cpuset(obj, affinity);
        let id = |obj: &TopologyObject| obj.global_persistent_index();
        let simplify_node = |mut obj| {
            crate::debug!("Simplifying topology object {obj}...");
            while children_in_cpuset(obj).count() == 1 {
                let Some(only_child) = children_in_cpuset(obj).next() else {
                    unreachable!("checked above that there is one child")
                };
                crate::trace!("- Trying {obj} -> {only_child}");
                if only_child.object_type() == ObjectType::PU {
                    crate::trace!("- Can't simplify non-PU into PU, revert to {obj}");
                    break;
                }
                obj = only_child;
            }
            crate::debug!("...into object #{}: {obj}", id(obj));
            obj
        };

        // Start building the topology tree by inserting the (simplified) root
        crate::debug!("Creating root node");
        let mut nodes = HashMap::new();
        let add_node = |nodes: &mut HashMap<_, _>, obj: &TopologyObject| {
            assert!(
                obj.cpuset()
                    .expect("normal objects should have cpusets")
                    .intersects(affinity),
                "iteration tree nodes should have children"
            );
            let id = id(obj);
            let priority = parent_priorities.get(&id).copied();
            crate::debug!("Adding node {id} with priority {priority:?}");
            nodes.insert(id, NodeIterator::new(priority));
        };
        let root = simplify_node(topology.root_object());
        add_node(&mut nodes, root);

        // Finish building the tree through top-down breadth-first traversal
        crate::debug!("Recursing over root descendants");
        let mut curr_node_children = vec![(id(root), children_in_cpuset(root).collect::<Vec<_>>())];
        let mut next_node_children = Vec::new();
        let mut child_list_morgue = Vec::new();
        while !curr_node_children.is_empty() {
            // Iterate over parent nodes and associated lists of children...
            crate::debug!("Parents and children at current depth: {curr_node_children:#?}");
            for (parent_id, mut child_list) in curr_node_children.drain(..) {
                // ...then process this parent object's children...
                for mut child in child_list.drain(..) {
                    // Simplify the child object, hopefully down to a PU object
                    child = simplify_node(child);

                    // Register child into parent object
                    crate::debug!("Adding child {} to parent {parent_id}", id(child));
                    nodes
                        .get_mut(&parent_id)
                        .expect("parent should be present since traversal is top-down")
                        .add_child(child);

                    // If the child has grandchildren, make it a node and
                    // schedule processing the grandchildren
                    if children_in_cpuset(child).count() > 0 {
                        crate::debug!("Child has grandchildren, make it a node...");
                        add_node(&mut nodes, child);
                        let mut grandchild_list: Vec<_> =
                            child_list_morgue.pop().unwrap_or_default();
                        grandchild_list.extend(children_in_cpuset(child));
                        crate::debug!(
                            "...and schedule processing grandchildren {grandchild_list:#?}"
                        );
                        next_node_children.push((id(child), grandchild_list));
                    } else {
                        crate::debug!("Child is a PU, we're done with this branch");
                        assert_eq!(child.object_type(), ObjectType::PU);
                    }
                }
                child_list_morgue.push(child_list);
            }
            std::mem::swap(&mut curr_node_children, &mut next_node_children);
        }
        let result = Self {
            nodes,
            next_pu_path: vec![id(root)],
        };
        crate::debug!("Initial work distribution tree state: {result:#?}");
        result
    }
}
//
impl<'topology> Iterator for PUIterator<'topology> {
    type Item = &'topology TopologyObject;

    fn next(&mut self) -> Option<Self::Item> {
        // Dive down next_pu_path to find the next PU we'll yield
        crate::debug!("Looking for a PU to yield...");
        let mut current_node_id = *self.next_pu_path.last()?;
        const PATH_ERROR: &str = "node in next_pu_path should exist";
        let mut current_node = self.nodes.get_mut(&current_node_id).expect(PATH_ERROR);
        crate::trace!("- Start from path end node {current_node_id}");
        let yielded_pu = loop {
            match current_node.current_child() {
                Child::Node(next_node_id) => {
                    crate::trace!("- Recurse to child node {next_node_id}");
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
        crate::debug!("Found {yielded_pu} through path {:?}", self.next_pu_path);

        // Remove the PU and every ancestor node that is transitively emptied by
        // this operation, until we reach to an ancestor that still has more
        // children to yield and merely switches to the next of these children.
        crate::debug!("Deleting PU and its emptied ancestors...");
        let mut downstream_priority = loop {
            match current_node.remove_child() {
                RemoveOutcome::NewChild(priority) => {
                    crate::trace!("- Ancestor {current_node_id} switched to next child with priority {priority:?}");
                    break priority;
                }
                RemoveOutcome::Empty => {
                    crate::debug!("- Ancestor node {current_node_id} is now empty, delete it");
                    self.next_pu_path.pop();
                    self.nodes.remove(&current_node_id);
                    let Some(next_node_id) = self.next_pu_path.last() else {
                        // We deleted all nodes in the path, so there is no
                        // iteration state left to be advanced...
                        assert!(
                            self.nodes.is_empty(),
                            "no path left should mean all nodes have been fully processed"
                        );
                        crate::trace!(
                            "- Done iterating over topology PUs, next run will yield None"
                        );
                        return Some(yielded_pu);
                    };
                    crate::trace!("- Will switch to parent {next_node_id}");
                    current_node_id = *next_node_id;
                    current_node = self.nodes.get_mut(&current_node_id).expect(PATH_ERROR);
                    continue;
                }
            }
        };
        crate::debug!(
            "Updated next PU path is {:?} with updated leaf node {:#?} and child switch priority {}",
            self.next_pu_path,
            current_node,
            downstream_priority
        );

        // Next, walk up the remaining ancestors and figure out if any other
        // ancestor should switch to its next child.
        //
        // This is needed to handle situations where objects down the hierarchy
        // have a lower work distribution priority than those above them.
        //
        // For example, if you think that filling up Cores below L3 caches is
        // more important than filling up hyperthreaded PUs below Cores, then
        // you don't want to only switch to the next PU of the current core, you
        // want to switch to the next core of the current L3 cache as well.
        //
        // Similarly, if you are memory-bound and think that covering NUMA nodes
        // is more important than covering cores below them, you will not just
        // want to switch to the next PU/Core below the active NUMA node, but
        // also to switch to the next NUMA node on every iteration.
        crate::debug!("Switching more ancestors to their next child if priorized...");
        let mut valid_path_len = self.next_pu_path.len();
        for (remaining_ancestors, &ancestor_id) in
            self.next_pu_path.iter().enumerate().rev().skip(1)
        {
            crate::trace!("- Considering ancestor {ancestor_id}...");
            current_node_id = ancestor_id;
            current_node = self.nodes.get_mut(&current_node_id).expect(PATH_ERROR);
            if let Some(switch_priority) =
                current_node.switch_child_if_priorized(downstream_priority)
            {
                crate::debug!(
                    "- Ancestor {ancestor_id} at path index {remaining_ancestors} \
                    switched to new child with priority {switch_priority}, \
                    updated ancestor state is {current_node:#?}"
                );
                downstream_priority = switch_priority;
                // Switching children here invalidates the rest of the PU path
                valid_path_len = remaining_ancestors + 1;
            }
        }
        self.next_pu_path.truncate(valid_path_len);
        crate::debug!("Final next PU path is {:?}", self.next_pu_path);
        Some(yielded_pu)
    }
}

/// Iteration over nodes of the [`PUIterator`] tree
mod node {
    use super::*;

    /// Node of the [`PUIterator`] tree
    #[derive(Clone, Debug)]
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
        /// Only nodes with a single child are allowed to leave this at `None`.
        child_switch_priority: Option<Priority>,

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
        /// You should then add at least one child using `add_child()`. After
        /// that you can use the other methods of this type.
        ///
        /// Only nodes that have a single child can leave
        /// `child_switch_priority` set to `None`.
        pub fn new(child_switch_priority: Option<Priority>) -> Self {
            Self {
                current_children: VecDeque::new(),
                next_children: Vec::new(),
                child_switch_priority,
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
            assert!(
                self.child_switch_priority.is_some() || self.num_children() == 1,
                "{}",
                Self::NEED_SWITCH_PRIORITY
            );
        }

        /// Check out the active child of this node
        pub fn current_child(&self) -> &Child<'topology> {
            self.current_children
                .front()
                .expect("this method should not be called on a node without children")
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
            // Switch to the next child, discarding the active child
            self.check_child_access();
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
                    self.child_switch_priority
                        .expect(Self::NEED_SWITCH_PRIORITY),
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
            let Some(switch_priority) = self.child_switch_priority else {
                assert_eq!(self.num_children(), 1, "{}", Self::NEED_SWITCH_PRIORITY);
                return None;
            };
            if switch_priority <= downstream_priority {
                return None;
            }

            // Remember the priority of the first child when switching away from
            // it in order to emit this priority when switching back to it
            if self.on_first_child() {
                self.first_child_priority = Some(downstream_priority);
            }
            let upstream_priority = if self.on_last_child() {
                self.first_child_priority.expect("should have set first_child_priority back when we switched away from the first child")
            } else {
                switch_priority
            };

            // Switch to the next child, schedule looking at this child later
            let current_child = self.pop_child();
            self.next_children.push(current_child);
            Some(upstream_priority)
        }

        /// Current number of children below this node
        fn num_children(&self) -> usize {
            self.current_children.len() + self.next_children.len()
        }

        /// Extract the current child of the node
        fn pop_child(&mut self) -> Child<'topology> {
            // Get the current child
            let child = self
                .current_children
                .pop_front()
                .expect("this method should not be called on a node without children");

            // Handle wraparound from last child to first child
            if self.current_children.is_empty() {
                self.current_children.extend(self.next_children.drain(..));
                self.first_child_priority = None;
            }
            child
        }

        /// Error message when `normal_child_priority` should be set but isn't
        const NEED_SWITCH_PRIORITY: &'static str =
            "multi-children nodes should have a child-switching priority";

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
        #[track_caller]
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
