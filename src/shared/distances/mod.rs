//! Matrix of distances between ThreadPool workers

mod pu_iterator;

use self::pu_iterator::PUIterator;
use crate::priority::{self, JobProperties, Priority};
use hwlocality::{bitmap::BitmapIndex, cpu::cpuset::CpuSet, object::TopologyObject, Topology};
use std::fmt::{self, Debug};

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
        log::info!(
            "Building a thread pool with affinity {affinity} corresponding to cpuset {cpuset}"
        );
        let num_workers = cpuset.weight().unwrap();
        assert!(
            num_workers < usize::from(Distance::MAX),
            "CPUs with >{} cores aren't supported yet, time to switch to the next integer width?",
            Distance::MAX
        );

        // Handle empty and singleton cpuset edge case
        if num_workers == 0 {
            return (
                Self {
                    data: Vec::new().into_boxed_slice(),
                    num_workers: 0,
                },
                Vec::new(),
            );
        } else if num_workers == 1 {
            return (
                Self {
                    data: vec![0].into_boxed_slice(),
                    num_workers: 1,
                },
                vec![cpuset.first_set().expect("should be there if weight > 0")],
            );
        }

        // Look up in which place of the topology work distribution decisions
        // must be made, and priorize those decisions
        let parent_priorities = priority::load_balancing_priorities(
            &priority::parents_by_typed_depth(topology, affinity),
            // FIXME: Propagate JobProperties from the thread pool
            &JobProperties::default(),
        );

        // Order PUs in such a way that neighbor PUs have high odds of being the
        // best targets for load balancing transactions (and indeed always are
        // when the hardware topology is symmetric)
        let worker_pus =
            PUIterator::new(topology, affinity, &parent_priorities).collect::<Vec<_>>();
        assert_eq!(worker_pus.len(), num_workers);
        let sorted_cpus = worker_pus
            .iter()
            .map(|pu| {
                BitmapIndex::try_from(pu.os_index().expect("PUs should have an OS index"))
                    .expect("PU logical index should fit in a cpuset")
            })
            .collect::<Vec<_>>();
        crate::info!("Workers were assigned CPUs {sorted_cpus:?}");
        assert_eq!(
            sorted_cpus.iter().copied().collect::<CpuSet>(),
            cpuset,
            "workers should cover all requested CPUs"
        );

        // Group workers by nearest neighbor
        //
        // More precisely, we collect a list of boundaries between
        // nearest-neighbor groups, with the associated common ancestor priority
        crate::debug!("Grouping workers into nearest neighbor sets...");
        let mut indexed_pus = worker_pus.iter().enumerate();
        let mut group_boundaries = Vec::new();
        let mut leader_pu = indexed_pus
            .next()
            .expect("asserted above there are >= 2 workers")
            .1;
        let nearest_neighbor_priority = |pu: &TopologyObject| {
            pu.ancestors()
                .filter_map(|ancestor| {
                    parent_priorities
                        .get(&ancestor.global_persistent_index())
                        .copied()
                })
                .max()
                .expect("asserted above there are >= 2 workers, hence multi-children ancestors")
        };
        let mut leader_neighbor_priority = nearest_neighbor_priority(leader_pu);
        crate::trace!(
            "- Initial leader is worker #0 ({leader_pu}) with nearest neighbor priority {leader_neighbor_priority}");
        for (idx, pu) in indexed_pus {
            let common_ancestor = pu
                .first_common_ancestor(leader_pu)
                .expect("if there are two PUs, at least Machine can be a common ancestor");
            let common_ancestor_priority =
                parent_priorities[&common_ancestor.global_persistent_index()];
            crate::trace!(
                "- Common ancestor of worker #{idx} ({pu}) with leader \
                is {common_ancestor} with priority {common_ancestor_priority}"
            );
            if common_ancestor_priority < leader_neighbor_priority {
                group_boundaries.push((idx, common_ancestor_priority));
                leader_pu = pu;
                leader_neighbor_priority = nearest_neighbor_priority(pu);
                crate::trace!(
                    "* Worker #{idx} isn't part of leader's nearest neigbors: \
                    record a boundary with priority {common_ancestor_priority} \
                    and make it the new leader with nearest neighbor priority \
                    {leader_neighbor_priority}"
                );
            }
        }
        crate::info!("Nearest neighbor set (boundaries, priorities) are {group_boundaries:?}");

        // Compute distance matrix
        crate::info!("Computing inter-worker distances...");
        let mut data = vec![Distance::MAX; num_workers * num_workers].into_boxed_slice();
        for worker_idx in 0..num_workers {
            // Access distances from current worker and define distance metric
            crate::debug!("Computing distances from worker {worker_idx}...");
            let distances = &mut data[worker_idx * num_workers..(worker_idx + 1) * num_workers];

            // Locate the closest left and right nearest neighbor group
            // boundaries in group_boundaries, if any, then start iterating over
            // nearest neighbor group boundaries in both directions
            let next_right_boundary_idx = group_boundaries
                .binary_search_by_key(&worker_idx, |(leader_idx, _priority)| *leader_idx)
                // - If binary_search fails, it returns the location where a new
                //   boundary could be inserted, which is to the right of any
                //   existing left group boundary and at the position of any
                //   existing right group boundary
                // - If binary_search succeeds, it returns the location of an
                //   existing boundary positioned where the worker is. Since
                //   boundaries mark the position of the leftmost element in a
                //   group, this is the left boundary of the worker's group. And
                //   the right boundary, if any, is located after it.
                .map_or_else(|insert_idx| insert_idx, |existing_idx| existing_idx + 1);
            let mut left_boundaries = group_boundaries
                .iter()
                .take(next_right_boundary_idx)
                .rev()
                .copied();
            let mut right_boundaries = group_boundaries
                .iter()
                .skip(next_right_boundary_idx)
                .copied();
            let mut curr_left_bound = left_boundaries.next();
            let mut curr_right_bound = right_boundaries.next();
            crate::debug!(
                "The boundaries of the worker's nearest neighbor group are \
                {curr_left_bound:?} and {curr_right_bound:?}"
            );

            // Fill distances through nearest neighbor set iteration
            distances[worker_idx] = 0;
            let mut left_end = worker_idx;
            let mut left_distance_offset = 1;
            let mut right_start = worker_idx + 1;
            let mut right_distance_offset = 1;
            while left_end > 0 || right_start < num_workers {
                // Find boundaries of the next nearest neighbor sets
                let (left_bound_idx, left_bound_priority) = curr_left_bound.unwrap_or((0, 0));
                let (right_bound_idx, right_bound_priority) =
                    curr_right_bound.unwrap_or((num_workers, 0));

                // Fill linear distances until the next left and right bounds
                crate::debug!(
                    "Filling left neighbor distances \
                    {left_bound_idx}..{left_end} \
                    with offset {left_distance_offset}..."
                );
                for (local_distance, left_idx) in (left_bound_idx..left_end).rev().enumerate() {
                    distances[left_idx] =
                        Distance::try_from(left_distance_offset + local_distance).unwrap();
                }
                crate::debug!(
                    "Filling right neighbor distances \
                    {right_start}..{right_bound_idx} \
                    with offset {right_distance_offset}..."
                );
                for (local_distance, right_idx) in (right_start..right_bound_idx).enumerate() {
                    distances[right_idx] =
                        Distance::try_from(right_distance_offset + local_distance).unwrap();
                }

                // Account for the linear distances we just filled
                let num_left_elems = left_end - left_bound_idx;
                let num_right_elems = right_bound_idx - right_start;
                left_end = left_bound_idx;
                right_start = right_bound_idx;

                // Adjust subsequent left and right distances every worker
                // outside of the local nearest neighbour group is at a higher
                // distance than workers inside of the nearest neighbor group
                let max_elems = num_left_elems.max(num_right_elems);
                left_distance_offset += max_elems;
                right_distance_offset += max_elems;

                // Schedule exploring workers further away on the left and/or
                // the right hand side, depending on relative priorities
                if curr_left_bound.is_some() && left_bound_priority >= right_bound_priority {
                    curr_left_bound = left_boundaries.next();
                    crate::debug!("Will now explore left neigbors until {curr_left_bound:?}");
                }
                if curr_right_bound.is_some() && left_bound_priority <= right_bound_priority {
                    curr_right_bound = right_boundaries.next();
                    crate::debug!("Will now explore right neigbors until {curr_right_bound:?}");
                }
            }
            crate::debug!("Distances are ready!");
            crate::info!("{worker_idx:>3} -> {distances:>3?}");
        }
        (Self { data, num_workers }, sorted_cpus)
    }

    /// Access the distances from a certain worker to others
    pub fn from(&self, worker_idx: usize) -> &[Distance] {
        assert!(worker_idx < self.num_workers, "invalid worker index");
        &self.data[worker_idx * self.num_workers..(worker_idx + 1) * self.num_workers]
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
