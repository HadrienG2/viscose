//! Search for workers matching a predicate in the [`HierarchicalState`]

use super::{Child, ChildrenLink, HierarchicalState, Node};
use crate::shared::flags::bitref::BitRef;
use std::{
    borrow::Cow, cell::RefCell, collections::VecDeque, sync::atomic::Ordering, thread_local,
};

/// Enumerate workers that match a certain work availability condition
/// (either they have work available for stealing, or they are looking for
/// work) at increasing distances from a certain "local" worker
///
/// - `local_worker_idx` should identify the local worker around which we're
///   searching for opportunities to steal or give work
/// - `ancestor_bits` should yield owned or borrowed [`BitRef`]s to each
///   node ancestor of the local worker, in order, as if you were iterating
///   over that worker's `WorkAvailabilityPath`.
/// - `find_siblings` should be `find_siblings_to_rob` when stealing work,
///   `find_jobless_siblings` when giving work.
/// - `find_strangers` should be `find_strangers_to_rob` when stealing work,
///   `find_jobless_strangers` when giving work.
pub(super) fn find_workers<'result, const CACHE_SEARCH_MASKS: bool, Siblings, Strangers>(
    state: &'result HierarchicalState,
    local_worker_idx: usize,
    mut ancestor_bits: impl Iterator<Item = Cow<'result, BitRef<'result, CACHE_SEARCH_MASKS>>> + 'result,
    mut find_siblings: impl FnMut(
            &'result ChildrenLink,
            &BitRef<'result, CACHE_SEARCH_MASKS>,
            Ordering,
        ) -> Option<Siblings>
        + 'result,
    find_strangers: impl FnMut(&'result ChildrenLink, Ordering) -> Option<Strangers> + 'result,
) -> Option<impl Iterator<Item = usize> + 'result>
where
    Siblings: Iterator<Item = usize> + 'result,
    Strangers: Iterator<Item = usize> + 'result,
{
    // Locate our direct parent or return None in case we have no parent
    // (it means there are no other workers to exchange work with)
    let worker_bit = ancestor_bits.next()?;
    let parent_idx = state.workers[local_worker_idx].parent.expect(
        "if a worker has a parent in its WorkAvailabilityPath, \
        then it should have a parent node index",
    );
    let parent = &state.work_availability_tree[parent_idx];

    // Start iterating over sibling workers, if any
    let curr_node = &parent.object;
    let worker_iter = find_siblings(&curr_node.worker_children, &worker_bit, LOAD_ORDER)
        .map(NodeChildren::Siblings);

    // Grab a deque of next subtree nodes from the allocation cache
    let mut next_subtree_nodes =
        DEQUES.with(|deques| deques.borrow_mut().pop().unwrap_or_default());
    next_subtree_nodes.clear();

    // Yield iterator over increasingly remote workers
    Some(WorkerSearch {
        state,
        worker_iter,
        curr_node,
        next_subtree_nodes,
        ancestor_bits,
        curr_ancestor: parent,
        node_bit: None,
        find_siblings,
        find_strangers,
    })
}

/// Atomic ordering to be used for work availability flags readout
///
/// All loads need to have at least Acquire ordering because...
///
/// - Between observing the value of a worker's work availability bit and
///   trying to act on the rest of that worker's state accordingly, an
///   Acquire barrier is needed to make sure that we have an up-to-date
///   view of the worker's state in question (as of the last work
///   availability bit update at least).
/// - Between observing a node's work availability bit and querying the
///   work availability bits of its children, an Acquire barrier is
///   needed to make sure that the updates to the work availability bits
///   of child nodes and workers that eventually led the node's work
///   availability bit to be updated are visible.
///
/// Further, load ordering doesn't need to be stronger than Acquire:
///
/// - Don't need AcqRel (which would require replacing the load with a
///   RMW) since we're not trying to get any other thread in sync with
///   our current state during this specific lock-free transaction.
/// - Don't need SeqCst since there is no need for everyone to agree on
///   the global order in which work is looked for and exchanged.
const LOAD_ORDER: Ordering = Ordering::Acquire;

/// Iterator which enumerates workers that match a certain work availability
/// condition (either they have work available for stealing, or they are looking
/// for work) at increasing distances from a certain "local" worker
pub(super) struct WorkerSearch<
    'state,
    AncestorBits,
    FindSiblings,
    FindStrangers,
    Siblings,
    Strangers,
    const CACHE_SEARCH_MASKS: bool,
> {
    /// Underlying hierarchical state
    state: &'state HierarchicalState,

    /// [`HierarchicalState`] node whose children we are currently processing
    curr_node: &'state Node,

    /// Iterator over worker children of `curr_node`
    worker_iter: Option<NodeChildren<Siblings, Strangers>>,

    /// This is set when `curr_node` is an higher-order ancestor of the worker
    /// (not its direct parent, but its grandparent or higher)
    ///
    /// It indicates the position of the work availability bit of the
    /// lower-level ancestor in the `node_children` list of `curr_node`.
    node_bit: Option<Cow<'state, BitRef<'state, CACHE_SEARCH_MASKS>>>,

    /// Other [`HierarchicalState`] nodes in the current ancestor subtree
    ///
    /// Nodes in a subtree are processed in breadth-first order. The local
    /// worker is not a descendant of any of these nodes.
    next_subtree_nodes: VecDeque<usize>,

    /// Ancestor of the local worker whose subtree we're currently exploring
    curr_ancestor: &'state Child<Node>,

    /// Higher-order local worker ancestor nodes
    ancestor_bits: AncestorBits,

    /// Callback used to iterate over siblings of the active ancestor node
    find_siblings: FindSiblings,

    /// Callback used to iterate over unrelated children of the active node
    find_strangers: FindStrangers,
}
//
impl<
        'state,
        AncestorBits,
        FindSiblings,
        FindStrangers,
        Siblings,
        Strangers,
        const CACHE_SEARCH_MASKS: bool,
    > Iterator
    for WorkerSearch<
        'state,
        AncestorBits,
        FindSiblings,
        FindStrangers,
        Siblings,
        Strangers,
        CACHE_SEARCH_MASKS,
    >
where
    AncestorBits: Iterator<Item = Cow<'state, BitRef<'state, CACHE_SEARCH_MASKS>>>,
    FindSiblings: FnMut(
        &'state ChildrenLink,
        &BitRef<'state, CACHE_SEARCH_MASKS>,
        Ordering,
    ) -> Option<Siblings>,
    FindStrangers: FnMut(&'state ChildrenLink, Ordering) -> Option<Strangers>,
    Siblings: Iterator<Item = usize>,
    Strangers: Iterator<Item = usize>,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        'process_nodes: loop {
            // Try directly yielding from the active worker iterator
            if let Some(worker_iter) = self.worker_iter.as_mut() {
                if let Some(worker) = worker_iter.next() {
                    return Some(worker);
                } else {
                    self.worker_iter = None;
                }
            }

            // Once we're done looking at the worker children of the active
            // node, schedule looking at its node children later on
            if let Some(node_bit) = self.node_bit.take() {
                if let Some(nodes) =
                    (self.find_siblings)(&self.curr_node.node_children, &node_bit, LOAD_ORDER)
                {
                    for node_idx in nodes {
                        self.next_subtree_nodes.push_back(node_idx);
                    }
                }
            } else if let Some(nodes) =
                (self.find_strangers)(&self.curr_node.node_children, LOAD_ORDER)
            {
                for node_idx in nodes {
                    self.next_subtree_nodes.push_back(node_idx);
                }
            }

            // Switch to the next node in the current subtree, if any
            if let Some(next_node_idx) = self.next_subtree_nodes.pop_front() {
                self.curr_node = &self.state.work_availability_tree[next_node_idx].object;
                continue 'process_nodes;
            }

            // If we ran out of nodes, then it means we're done exploring the
            // subtree associated with our current ancestor, so it's time to go
            // to the next ancestor
            //
            // If there is no next ancestor, we end the search by yielding None.
            self.node_bit = Some(self.ancestor_bits.next()?);
            let ancestor_idx = self.curr_ancestor.parent.expect(
                "if an ancestor node has a parent in the WorkAvailabilityPath, \
                then it should have a parent node index too",
            );
            self.curr_ancestor = &self.state.work_availability_tree[ancestor_idx];
            self.curr_node = &self.curr_ancestor.object;
            self.worker_iter = (self.find_strangers)(&self.curr_node.worker_children, LOAD_ORDER)
                .map(NodeChildren::Strangers);
        }
    }
}
//
impl<
        AncestorBits,
        FindSiblings,
        FindStrangers,
        Siblings,
        Strangers,
        const CACHE_SEARCH_MASKS: bool,
    > Drop
    for WorkerSearch<
        '_,
        AncestorBits,
        FindSiblings,
        FindStrangers,
        Siblings,
        Strangers,
        CACHE_SEARCH_MASKS,
    >
{
    #[inline(always)]
    fn drop(&mut self) {
        DEQUES.with(|deques| {
            deques
                .borrow_mut()
                .push(std::mem::take(&mut self.next_subtree_nodes))
        })
    }
}

thread_local! {
    /// VecDeque allocation cache
    static DEQUES: RefCell<Vec<VecDeque<usize>>> = RefCell::new(Vec::new());
}

/// Iterator over either neighboring or "foreign" children in a node child list
enum NodeChildren<Siblings, Strangers> {
    /// Iterating over siblings of the thief worker
    Siblings(Siblings),

    /// Iterating over workers unrelated to the thief
    Strangers(Strangers),
}
//
impl<Siblings, Strangers> Iterator for NodeChildren<Siblings, Strangers>
where
    Siblings: Iterator<Item = usize>,
    Strangers: Iterator<Item = usize>,
{
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Siblings(siblings) => siblings.next(),
            Self::Strangers(strangers) => strangers.next(),
        }
    }
}
