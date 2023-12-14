//! Matrix of distances between ThreadPool workers

use hwlocality::object::{types::ObjectType, TopologyObject};
use std::{
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
    /// Compute the matrix of distances between a set of workers, represented by
    /// the matching processing unit objects in the hwloc topology.
    ///
    /// Workers will be sorted in a way that minimizes the distance between
    /// nearest neighbors.
    pub fn measure_and_sort(workers: &mut [&TopologyObject]) -> Self {
        // Check preconditions on worker objects
        assert!(
            workers
                .iter()
                .all(|worker| worker.object_type() == ObjectType::PU),
            "workers should be identified via PU objects"
        );
        let num_workers = workers.len();
        assert!(
            num_workers < usize::from(Distance::MAX),
            "CPUs with >{} cores aren't supported yet, time to switch to the next integer width?",
            Distance::MAX
        );

        // Order workers to put nearest neighbors close to each other
        workers.sort_unstable_by_key(|pu| pu.logical_index());

        // Compute distance matrix
        let mut data = vec![Distance::MAX; num_workers * num_workers].into_boxed_slice();
        for worker_idx in 0..num_workers {
            let distances = &mut data[worker_idx * num_workers..(worker_idx + 1) * num_workers];
            let mut curr_distance = 0;
            let mut left_idx = worker_idx;
            let mut right_idx = worker_idx;
            let last_right_idx = num_workers - 1;
            let topological_distance = |neighbor_idx: usize| {
                let worker = &workers[worker_idx];
                let common = worker.common_ancestor(workers[neighbor_idx]).unwrap();
                worker
                    .ancestors()
                    .take_while(|ancestor| !ptr::eq(common, *ancestor))
                    .count()
            };
            distances[worker_idx] = 0;
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
