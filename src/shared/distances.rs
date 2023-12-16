//! Matrix of distances between ThreadPool workers

use hwlocality::{
    bitmap::BitmapIndex,
    cpu::cpuset::CpuSet,
    object::{depth::NormalDepth, types::ObjectType, TopologyObject, TopologyObjectID},
    Topology,
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::{self, Debug},
    ptr,
};

/// Distances between ThreadPool workers
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct Distances {
    /// Distance data from all workers
    ///
    /// Currently uses a simple unpadded 2D matrix layout such that
    /// distance(i, j) = data[i * Nworkers + j]. Could try alternate layouts
    /// optimized for extra cache locality later, if the need arises.
    data: Box<[Distance]>,

    /// Number of workers
    num_workers: usize,
}
//
impl Distances {
    /// Given an hwloc topology and an affinity mask, order selected CPUs in a
    /// manner that minimizes the distance between nearest neighbors and compute
    /// the associated matrix of inter-CPU distances
    pub fn with_sorted_cpus(topology: &Topology, affinity: &CpuSet) -> (Self, Vec<BitmapIndex>) {
        // Check preconditions
        let cpuset = topology.cpuset() & affinity;
        let num_workers = cpuset.weight().unwrap();
        assert!(
            num_workers < usize::from(Distance::MAX),
            "CPUs with >{} cores aren't supported yet, time to switch to the next integer width?",
            Distance::MAX
        );

        // Look up in which place of the topology work distribution decisions
        // must be made, and priorize those decisions
        let parent_priorities =
            Self::priorize_parents(topology, &cpuset, Self::optimize_for_compute);

        // Sort PUs in such a way that neighbor PUs have high odds of being the
        // best targets for load balancing transactions (and indeed always are
        // if the hardware topology is sufficiently symmetric)
        let sorted_pus =
            PUIterator::new(topology, affinity, &parent_priorities).collect::<Vec<_>>();
        assert_eq!(sorted_pus.len(), num_workers);

        unimplemented!(
            "compute distance matrix using a variant of the old strategy below, \
            the main new thing being that the ancestor chain must be re-sorted \
            by parent priority for topological distance evaluation"
        );

        // Order workers to put nearest neighbors close to each other
        worker_pus.sort_unstable_by_key(|pu| pu.logical_index());

        // Compute distance matrix
        let mut data = vec![Distance::MAX; num_workers * num_workers].into_boxed_slice();
        for worker_idx in 0..num_workers {
            // Access distances from current worker and define distance metric
            let distances = &mut data[worker_idx * num_workers..(worker_idx + 1) * num_workers];
            let topological_distance = |neighbor_idx: usize| {
                let worker = &worker_pus[worker_idx];
                let common = worker.common_ancestor(worker_pus[neighbor_idx]).unwrap();
                worker
                    .ancestors()
                    .take_while(|ancestor| !ptr::eq(common, *ancestor))
                    .count()
            };

            // Initialize distance computation
            let mut curr_distance = 0;
            let mut left_idx = worker_idx;
            let mut right_idx = worker_idx;
            let last_right_idx = num_workers - 1;
            distances[worker_idx] = 0;

            // Do bidirectional iteration as long as relevant
            while left_idx > 0 && right_idx < last_right_idx {
                curr_distance += 1;
                let topological_distance_left = topological_distance(left_idx - 1);
                let topological_distance_right = topological_distance(right_idx + 1);
                if topological_distance_left <= topological_distance_right {
                    left_idx -= 1;
                    distances[left_idx] = curr_distance;
                }
                if topological_distance_left >= topological_distance_right {
                    right_idx += 1;
                    distances[right_idx] = curr_distance;
                }
            }

            // Finish with unidirectional iteration
            loop {
                curr_distance += 1;
                if left_idx > 0 {
                    left_idx -= 1;
                    distances[left_idx] = curr_distance;
                } else if right_idx < last_right_idx {
                    right_idx += 1;
                    distances[right_idx] = curr_distance;
                } else {
                    break;
                }
            }
        }
        Self { data, num_workers }
    }

    /// Access the distances from a certain worker to others
    pub fn from(&self, worker_idx: usize) -> &[Distance] {
        assert!(worker_idx < self.num_workers, "invalid worker index");
        &self.data[worker_idx * self.num_workers..(worker_idx + 1) * self.num_workers]
    }

    /// Priorize work distribution according to a certain policy
    ///
    /// A typical hwloc topology tree contains multiple branching points (NUMA
    /// nodes, L3 cache shards, multicore, hyperthreading...) and when dealing
    /// with smaller tasks that cannot cover the full CPU, we must decide how
    /// important it is to spread tasks over these various branching points.
    ///
    /// The simplest policy to implement is to follow the hwloc topology tree :
    /// spread work over hyperthreads first, then cores, then L3 shards, then
    /// NUMA nodes). This policy has optimal cache locality and inter-task
    /// communication latencies, however it is usually suboptimal in real-world
    /// use cases because hyperthreads contend over shared core ressources.
    ///
    /// The way you actually want to do it depends on the kind of work you're
    /// submitting to the thread pool. Therefore, the work distribution policy
    /// is configurable through the `policy` callback. This callback receives
    /// parent nodes grouped by depth as a parameter, and is in charge of
    /// producing a list of priority classes in increasing priority order. Newly
    /// spawned tasks will then be distributed over parents in the highest
    /// priority class that is not full yet.
    fn priorize_parents<'topology>(
        topology: &'topology Topology,
        affinity: &CpuSet,
        policy: impl for<'parents> FnOnce(
            &CpuSet,
            Vec<(ObjectType, NormalDepth, Vec<&'parents TopologyObject>)>,
        ) -> Vec<Vec<&'parents TopologyObject>>,
    ) -> HashMap<TopologyObjectID, Priority> {
        // Group multi-children nodes by increasing depth / locality
        let mut initial_parents = HashSet::new();
        let type_depth_parents = NormalDepth::iter_range(NormalDepth::MIN, topology.depth())
            .filter_map(|depth| {
                // Pick nodes with multiple children that are covered by our
                // current affinity mask
                let parents = topology
                    .objects_at_depth(depth)
                    .filter(|obj| {
                        obj.cpuset().unwrap().intersects(affinity)
                            && children_in_cpuset(obj, affinity).count() > 1
                    })
                    .inspect(|parent| {
                        initial_parents.insert(parent.global_persistent_index());
                    })
                    .collect::<Vec<_>>();

                // For each depth, provide a (type, depth, nodes) tuple
                (!parents.is_empty()).then(|| {
                    (
                        topology
                            .type_at_depth(depth)
                            .expect("this is a valid depth"),
                        depth,
                        parents,
                    )
                })
            })
            .collect::<Vec<_>>();

        // Let policy callback compute priority classes accordingly
        let priority_classes = Self::optimize_for_compute(affinity, type_depth_parents);
        let final_parents = priority_classes
            .iter()
            .flatten()
            .map(|parent| parent.global_persistent_index())
            .collect::<HashSet<_>>();
        assert_eq!(
            initial_parents, final_parents,
            "priorization policies should not add or remove parents"
        );

        // Give each parent node a numerical priority accordingly
        priority_classes
            .into_iter()
            .enumerate()
            .flat_map(|(priority, parents)| {
                parents
                    .into_iter()
                    .map(move |parent| (parent.global_persistent_index(), priority))
            })
            .collect()
    }

    /// Work distribution policy optimized for latency-bound tasks
    ///
    /// Rigorously follows the hwloc hierarchy under the premise that what's
    /// local in that hierarchy is also likely to exhibit the highest cache
    /// locality and the lowest inter-task communication latency.
    #[allow(unused)]
    fn optimize_for_latency<'parents>(
        _affinity: &CpuSet,
        type_depth_parents: Vec<(ObjectType, NormalDepth, Vec<&'parents TopologyObject>)>,
    ) -> Vec<Vec<&'parents TopologyObject>> {
        Self::depths_to_priorities(type_depth_parents).collect()
    }

    /// Turn `type_depth_parents` into priority classes
    fn depths_to_priorities(
        type_depth_parents: Vec<(ObjectType, NormalDepth, Vec<&TopologyObject>)>,
    ) -> impl Iterator<Item = Vec<&TopologyObject>> {
        type_depth_parents
            .into_iter()
            .map(|(_, _, parents)| parents)
    }

    /// Work distribution policy optimized for compute-bound tasks
    ///
    /// Deviates from the standard hwloc hierarchy by filling up hyperthreaded
    /// cores last: since each hyperthread shares compute resources with its
    /// sibling, hyperthreading is usally very ineffective on well-optimized
    /// compute-bound tasks, and thus the use of independent cores should
    /// normally be priorized over that of hyperthreads.
    #[allow(unused)]
    fn optimize_for_compute<'parents>(
        _affinity: &CpuSet,
        mut type_depth_parents: Vec<(ObjectType, NormalDepth, Vec<&'parents TopologyObject>)>,
    ) -> Vec<Vec<&'parents TopologyObject>> {
        let mut result = Vec::with_capacity(type_depth_parents.len());
        Self::depriorize_hyperthreads(&mut type_depth_parents, &mut result);
        result.extend(Self::depths_to_priorities(type_depth_parents));
        result
    }

    /// Hyperthread-depriorization part of `optimize_for_compute`
    fn depriorize_hyperthreads(
        type_depth_parents: &mut Vec<(ObjectType, NormalDepth, Vec<&TopologyObject>)>,
        result: &mut Vec<Vec<&TopologyObject>>,
    ) {
        match type_depth_parents.pop() {
            Some((ObjectType::Core, _, cores)) => result.push(cores),
            Some(other) => type_depth_parents.push(other),
            None => {}
        }
    }

    /// Work distribution policy optimized for DRAM-bound tasks
    ///
    /// Picks up where `optimize_for_compute` left off, and additionally
    /// priorizes spreading work across NUMA nodes. This optimizes DRAM
    /// bandwidth at the cost of reducing cache locality, which should be the
    /// right tradeoff for memory-bound tasks that cannot fit in caches.
    #[allow(unused)]
    fn optimize_for_bandwidth<'parents>(
        affinity: &CpuSet,
        mut type_depth_parents: Vec<(ObjectType, NormalDepth, Vec<&'parents TopologyObject>)>,
    ) -> Vec<Vec<&'parents TopologyObject>> {
        // Start like optimize_for_compute
        let mut result = Vec::with_capacity(type_depth_parents.len());
        Self::depriorize_hyperthreads(&mut type_depth_parents, &mut result);

        // Extract the NUMA-relevant subset of the hwloc hierarchy in LIFO order
        let mut numa_depths_rev = Vec::new();
        'depths: for (depth_idx, (_, _, parents)) in type_depth_parents.iter().enumerate() {
            if parents.iter().any(|parent| {
                let parent_nodeset = parent.nodeset().unwrap();
                children_in_cpuset(parent, affinity)
                    .any(|child| child.nodeset().unwrap() != parent_nodeset)
            }) {
                numa_depths_rev.push(depth_idx);
            }
        }

        // Put these nodes back in FIFO order and priorize filling them first
        for numa_idx in numa_depths_rev.into_iter().rev() {
            let numa = type_depth_parents.remove(numa_idx);
            type_depth_parents.push(numa);
        }
        result.extend(Self::depths_to_priorities(type_depth_parents));
        result
    }
}
//
impl Debug for Distances {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Distances ")?;
        let mut worker_to_distances = f.debug_map();
        for worker_idx in 0..self.num_workers {
            worker_to_distances.entry(&worker_idx, &self.from(worker_idx));
        }
        worker_to_distances.finish()
    }
}

/// Distance between two workers
///
/// Should be wide enough to hold the maximum envisioned number of workers.
pub type Distance = u16;

/// Priority of a node for children enumeration (higher is higher priority)
pub type Priority = usize;

/// Tree used to iterate over PUs in an order that matches work distribution
/// priorities (i.e. PUs that should share work first are close)
#[derive(Default)]
struct PUIterator<'caller> {
    /// Summarized hwloc tree that only includes multi-children nodes, and
    /// priorizes the traversal of each child from each node
    nodes: HashMap<TopologyObjectID, Node<'caller>>,

    /// Path to the next PU to be yielded, if any
    next_pu_path: Vec<TopologyObjectID>,
}
//
impl<'caller> PUIterator<'caller> {
    /// Set up iteration over PUs
    fn new(
        topology: &'caller Topology,
        affinity: &CpuSet,
        parent_priorities: &'caller HashMap<TopologyObjectID, Priority>,
    ) -> Self {
        // This iterator will operate in a simplified hwloc topology that only
        // includes nodes from the affinity set that have either at least one PU
        // children or multiple non-PU children
        let id = |obj: &TopologyObject| obj.global_persistent_index();
        let children = |obj| children_in_cpuset(obj, affinity);
        let has_multiple_children = |obj| {
            let result = parent_priorities.contains_key(&id(obj));
            if !result {
                debug_assert!(
                    children(obj).count() < 2,
                    "multi-children nodes should have a priority"
                );
            }
            result
        };
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
        nodes.insert(id(root), Node::new(priority(root)));

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
                let parent = &mut nodes[&parent_id];
                // ...then process this parent object's children...
                'children: for child in curr_children.drain(..) {
                    // PUs always get added to parents
                    if child.object_type() == ObjectType::PU {
                        parent.add_child(child);
                        assert_eq!(child.normal_arity(), 0, "PUs shouldn't have children");
                        continue 'children;
                    }

                    // Nodes only get added if they pass the node filter
                    let parent_id = if keep_node(child) {
                        nodes.insert(id(child), Node::new(priority(child)));
                        parent.add_child(child);
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

        // The tree is now completely built, finalize it
        for node in nodes.values_mut() {
            node.finalize_children();
        }
        Self {
            nodes,
            next_pu_path: vec![id(root)],
        }
    }
}
//
impl<'caller> Iterator for PUIterator<'caller> {
    type Item = &'caller TopologyObject;

    fn next(&mut self) -> Option<Self::Item> {
        // Dive down from the active node to find the next PU that we'll
        // eventually yield
        let mut current_node_id = self.next_pu_path.last()?;
        let mut current_node = &mut self.nodes[current_node_id];
        let yielded_pu;
        loop {
            match current_node.current_child() {
                Child::Node(next_node_id) => {
                    current_node_id = next_node_id;
                    self.next_pu_path.push(*current_node_id);
                    current_node = &mut self.nodes[next_node_id];
                    continue;
                }
                Child::PU(pu) => {
                    yielded_pu = *pu;
                    break;
                }
                Child::Terminator => unreachable!(
                    "current Node child should always be on to the path to the next PU"
                ),
            }
        }

        // Remove the PU and every ancestor node that it transitively emptied by
        // this operation, until we get to an ancestor node that still has more
        // children to yield and merely switches to the next of its children.
        let mut downstream_priority = loop {
            match current_node.remove_child() {
                RemoveOutcome::NewChild(priority) => break priority,
                RemoveOutcome::Empty => {
                    self.nodes.remove(current_node_id);
                    let Some(next_node_id) = self.next_pu_path.last() else {
                        // We deleted all nodes in the path, so there is no
                        // iteration state left to be advanced...
                        assert!(
                            self.nodes.is_empty(),
                            "no path left should mean all nodes have been fully processed"
                        );
                        return Some(yielded_pu);
                    };
                    current_node_id = next_node_id;
                    current_node = &mut self.nodes[next_node_id];
                    continue;
                }
            }
        };

        // Next, walk up the remaining ancestors and figure out if any other
        // ancestor should switch to its next child.
        //
        // This is needed to correctly handle situations where objects down the
        // hierarchy have lower filling priority than the objects above them.
        // For example, if you (wisely) think that filling up Cores below L3
        // caches is more important than filling up PUs below Cores, then you
        // don't want to only switch to the next PU of the current core, you
        // want to switch to the next core as well. Similarly, if you are
        // memory-bound and think that covering NUMA nodes is more important
        // than covering cores below these NUMA nodes, you will not just want to
        // switch to the next PU/Core below the active NUMA node, but also to
        // switch to the other NUMA node on every iteration.
        let mut final_path_len = self.next_pu_path.len();
        for (remaining_len, ancestor_id) in self.next_pu_path.iter().enumerate().rev().skip(1) {
            current_node_id = ancestor_id;
            current_node = &mut self.nodes[current_node_id];
            let current_switch_priority = current_node.child_switch_priority();
            if current_switch_priority > downstream_priority {
                current_node.switch_child(downstream_priority);
                downstream_priority = current_switch_priority;
                // Switching children here invalidates the rest of the path
                final_path_len = remaining_len + 1;
            }
        }
        self.next_pu_path.truncate(final_path_len);
        Some(yielded_pu)
    }
}

/// Node of the [`PUIterator`] tree
struct Node<'topology> {
    /// Active child, or [`Terminator`] if not initialized yet
    active_child: Child<'topology>,

    /// Next children to be yieled, in logical index order
    ///
    /// Contains a [`Child::Terminator`], initially at the end of the list,
    /// which is used to detect when iteration over children wraps around. When
    /// that happens, switching back to the first child is not considered to be
    /// a work distribution event of priority `normal_child_priority`, but a
    /// work distribution event of priority `first_child_priority`.
    next_children: VecDeque<Child<'topology>>,

    /// How important it is to distribute work across children of this node
    ///
    /// Only nodes with a single PU child can and should have no priority.
    normal_child_priority: Option<Priority>,

    /// What switching back to the first child of this Node amounts to
    ///
    /// For example, on x86, after iterating over all the Cores of an L3 cache,
    /// switching back to the first core amounts to switching to a new PU.
    first_child_priority: Option<Priority>,
}
//
impl<'topology> Node<'topology> {
    /// Create an empty node with a certain priority
    ///
    /// You should then add a set of children with `add_children()`, then call
    /// `finalize_children()`. Then you can use the other methods of this type
    /// (and should not use the former methods anymore).
    ///
    /// Nodes should have one or more children. If they have one single child,
    /// they don't need a child-switching priority, and should not have one.
    /// Otherwise, they should have a child-switching priority.
    pub fn new(priority: Option<Priority>) -> Self {
        Self {
            active_child: Child::Terminator,
            next_children: VecDeque::new(),
            normal_child_priority: priority,
            first_child_priority: None,
        }
    }

    /// Add a child to a node
    pub fn add_child(&mut self, child: &'topology TopologyObject) {
        self.check_adding_children();
        let child = Child::from(child);
        if self.active_child == Child::Terminator {
            assert!(
                self.next_children.is_empty(),
                "should fill active_child before next_children"
            );
            self.active_child = child;
        } else {
            assert!(
                std::iter::once(&self.active_child)
                    .chain(&self.next_children)
                    .all(|&obj| obj != child),
                "attempted to add the same child twice"
            );
            self.next_children.push_back(child);
        }
    }

    /// Declare that we're done adding children to a node
    pub fn finalize_children(&mut self) {
        self.check_adding_children();
        assert_ne!(
            self.active_child,
            Child::Terminator,
            "should have added at least one child before finalizing"
        );
        if self.normal_child_priority.is_some() {
            assert!(
                !self.next_children.is_empty(),
                "single-child nodes don't need children priorities"
            );
        } else {
            assert!(
                self.next_children.is_empty(),
                "multi-children nodes need children priorities"
            );
        }
        self.next_children.push_back(Child::Terminator);
    }

    /// Check out the active child of this node
    pub fn current_child(&self) -> &Child {
        self.check_children_finalized();
        assert_ne!(
            self.active_child,
            Child::Terminator,
            "should never happen in correct usage"
        );
        &self.active_child
    }

    /// Remove the active child from this node
    ///
    /// If this returns [`RemoveOutcome::Empty`], you should stop using this
    /// node, discard it, and remove it from the child list of ancestor nodes,
    /// recursing if necessary.
    ///
    /// Otherwise, this simply amounts to switching to the next child, and you
    /// get the priority of the associated child-switching event.
    #[must_use]
    pub fn remove_child(&mut self) -> RemoveOutcome {
        self.check_children_finalized();
        self.active_child = Child::Terminator;
        if self.next_children.len() == 1 {
            assert_eq!(self.next_children.front(), Some(&Child::Terminator));
            RemoveOutcome::Empty
        } else {
            let priority = self.child_switch_priority();
            self.next_child();
            RemoveOutcome::NewChild(priority)
        }
    }

    /// Assess the priority of switching to the next child of this node
    pub fn child_switch_priority(&self) -> Priority {
        self.check_children_finalized();
        if self.on_last_child() {
            self.first_child_priority.expect(
                "first child priority should be set by the time we go back to the first child",
            )
        } else {
            self.normal_child_priority
                .expect("nodes with multiple children should have a child priority")
        }
    }

    /// Switch to the next child of this node
    ///
    /// Called on ancestors after lower-level nodes have had children removed,
    /// if it's determined that switching children at this layer of the tree is
    /// higher-priority than switching children in lower layers of the tree.
    pub fn switch_child(&mut self, downstream_priority: Priority) {
        // Check preconditions
        self.check_children_finalized();
        assert!(
            self.child_switch_priority() > downstream_priority,
            "should not switch children here when downstream switches are more important"
        );

        // If we're on the first child, record the priority of the child switch
        // that occured underneath so we can report it on the next wraparound
        if self.on_first_child() {
            self.first_child_priority = Some(downstream_priority);
        }

        // Put this child back in the queue, switch to the next one
        self.next_children.push_back(self.active_child);
        self.next_child();
    }

    /// Switch to the next child
    fn next_child(&mut self) {
        // Handle end of list: terminator handling, first child priority reset
        if self.on_last_child() {
            let terminator = self.next_children.pop_front();
            assert_eq!(
                terminator,
                Some(Child::Terminator),
                "true by definition of on_last_child"
            );
            self.next_children.push_back(Child::Terminator);
            self.first_child_priority = None;
        }

        // Switch to the next child
        let next_child = self
            .next_children
            .pop_front()
            .expect("should always have children");
        assert_ne!(
            next_child,
            Child::Terminator,
            "there should only be one terminator + other children"
        );
        self.active_child = next_child;
    }

    /// Truth that the active child is our first child
    fn on_first_child(&self) -> bool {
        self.next_children.back() == Some(&Child::Terminator)
    }

    /// Truth that the active child is our last child
    fn on_last_child(&self) -> bool {
        self.next_children.front() == Some(&Child::Terminator)
    }

    /// Check that we are in the children-building stage
    #[track_caller]
    fn check_adding_children(&self) {
        assert!(
            self.adding_children(),
            "this method should only be called before the child set is finalized"
        );
    }

    /// Check that our child set is finalized
    #[track_caller]
    fn check_children_finalized(&self) {
        assert!(
            !self.adding_children(),
            "this method should only be called after finalizing the child set"
        );
    }

    /// Truth that we are in the process of adding children
    fn adding_children(&self) -> bool {
        self.next_children
            .iter()
            .all(|&existing_child| existing_child != Child::Terminator)
    }
}

/// Child of a [`PUIterator`] tree [`Node`]
//
// --- Implementation notes ---
//
// This type doesn't work in the presence of objects from multiple topology, but
// we're not exposing this type to the outside world and won't it ourselves.
#[derive(Copy, Clone, Debug)]
enum Child<'topology> {
    /// Processing unit leaf
    PU(&'topology TopologyObject),

    /// Lower-level [`Node`]
    Node(TopologyObjectID),

    /// Child list terminator, used to detect when child iteration wraps around
    Terminator,
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
            (Self::Terminator, Self::Terminator) => true,
            _ => false,
        }
    }
}

/// Outcome of removing a [`Child`] from a [`Node`]
enum RemoveOutcome {
    /// Switched to the next child, which has the specified priority
    NewChild(Priority),

    /// Removed the last child, this [`Node`] should be removed from the tree
    Empty,
}

/// Select normal children of a node that match the affinity mask
fn children_in_cpuset<'out>(
    parent: &'out TopologyObject,
    affinity: &'out CpuSet,
) -> impl DoubleEndedIterator<Item = &'out TopologyObject> + Clone + 'out {
    parent
        .normal_children()
        .filter(move |child| child.cpuset().unwrap().intersects(affinity))
}
