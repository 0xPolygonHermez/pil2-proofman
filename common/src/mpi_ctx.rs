#[cfg(distributed)]
use mpi::traits::*;
#[cfg(distributed)]
use mpi::collective::CommunicatorCollectives;
#[cfg(distributed)]
use mpi::datatype::PartitionMut;
#[cfg(distributed)]
use mpi::environment::Universe;
#[cfg(distributed)]
use mpi::topology::Communicator;
use std::sync::atomic::AtomicU64;
#[cfg(distributed)]
use std::sync::atomic::Ordering;

use fields::PrimeField64;
#[cfg(distributed)]
use crate::ExtensionField;
use crate::GlobalInfo;

pub struct MpiCtx {
    #[cfg(distributed)]
    pub universe: Universe,
    #[cfg(distributed)]
    pub world: mpi::topology::SimpleCommunicator,
    pub rank: i32,
    pub n_processes: i32,
    pub node_rank: i32,
    pub node_n_processes: i32,
}

impl Default for MpiCtx {
    fn default() -> Self {
        MpiCtx::new()
    }
}

impl MpiCtx {
    pub fn new() -> Self {
        #[cfg(distributed)]
        {
            let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Multiple)
                .expect("Failed to initialize MPI with threading");
            let world = universe.world();
            let rank = world.rank();
            let n_processes = world.size();
            let local_comm = world.split_shared(rank);
            let node_rank = local_comm.rank();
            let node_n_processes = local_comm.size();

            MpiCtx { rank, n_processes, universe, world, node_rank, node_n_processes }
        }
        #[cfg(not(distributed))]
        {
            MpiCtx { rank: 0, n_processes: 1, node_rank: 0, node_n_processes: 1 }
        }
    }

    #[cfg(distributed)]
    pub fn new_with_universe(universe: Universe) -> Self {
        let world = universe.world();
        let rank = world.rank();
        let n_processes = world.size();
        let local_comm = world.split_shared(rank);
        let node_rank = local_comm.rank();
        let node_n_processes = local_comm.size();

        MpiCtx { rank, n_processes, universe, world, node_rank, node_n_processes }
    }

    #[inline]
    pub fn barrier(&self) {
        #[cfg(distributed)]
        {
            self.world.barrier();
        }
    }

    pub fn distribute_roots(&self, values: [u64; 10]) -> Vec<u64> {
        #[cfg(distributed)]
        {
            let mut all_values: Vec<u64> = vec![0u64; 10 * self.n_processes as usize];
            self.world.all_gather_into(&values, &mut all_values);
            all_values
        }
        #[cfg(not(distributed))]
        {
            roots
        }
    }

    pub fn distribute_airgroupvalues<F: PrimeField64>(
        &self,
        airgroupvalues: Vec<Vec<u64>>,
        _global_info: &GlobalInfo,
    ) -> Vec<Vec<F>> {
        #[cfg(distributed)]
        {
            let airgroupvalues_flatten: Vec<u64> = airgroupvalues.into_iter().flatten().collect();
            let mut gathered_data: Vec<u64> = vec![0; airgroupvalues_flatten.len() * self.n_processes as usize];

            const FIELD_EXTENSION: usize = 3;

            self.world.all_gather_into(&airgroupvalues_flatten, &mut gathered_data);

            let mut airgroupvalues_full: Vec<Vec<F>> = Vec::new();
            for agg_types in _global_info.agg_types.iter() {
                let mut values = vec![F::ZERO; agg_types.len() * FIELD_EXTENSION];
                for (idx, agg_type) in agg_types.iter().enumerate() {
                    if agg_type.agg_type == 1 {
                        values[idx * FIELD_EXTENSION] = F::ONE;
                    }
                }
                airgroupvalues_full.push(values);
            }

            for p in 0..self.n_processes as usize {
                let mut pos = 0;
                for (airgroup_id, agg_types) in _global_info.agg_types.iter().enumerate() {
                    for (idx, agg_type) in agg_types.iter().enumerate() {
                        if agg_type.agg_type == 0 {
                            airgroupvalues_full[airgroup_id][idx * FIELD_EXTENSION] +=
                                F::from_u64(gathered_data[airgroupvalues_flatten.len() * p + pos]);
                            airgroupvalues_full[airgroup_id][idx * FIELD_EXTENSION + 1] +=
                                F::from_u64(gathered_data[airgroupvalues_flatten.len() * p + pos + 1]);
                            airgroupvalues_full[airgroup_id][idx * FIELD_EXTENSION + 2] +=
                                F::from_u64(gathered_data[airgroupvalues_flatten.len() * p + pos + 2]);
                        } else {
                            let mut acc = ExtensionField {
                                value: [
                                    airgroupvalues_full[airgroup_id][idx * FIELD_EXTENSION],
                                    airgroupvalues_full[airgroup_id][idx * FIELD_EXTENSION + 1],
                                    airgroupvalues_full[airgroup_id][idx * FIELD_EXTENSION + 2],
                                ],
                            };
                            let val = ExtensionField {
                                value: [
                                    F::from_u64(gathered_data[airgroupvalues_flatten.len() * p + pos]),
                                    F::from_u64(gathered_data[airgroupvalues_flatten.len() * p + pos + 1]),
                                    F::from_u64(gathered_data[airgroupvalues_flatten.len() * p + pos + 2]),
                                ],
                            };
                            acc *= val;
                            airgroupvalues_full[airgroup_id][idx * FIELD_EXTENSION] = acc.value[0];
                            airgroupvalues_full[airgroup_id][idx * FIELD_EXTENSION + 1] = acc.value[1];
                            airgroupvalues_full[airgroup_id][idx * FIELD_EXTENSION + 2] = acc.value[2];
                        }
                        pos += FIELD_EXTENSION;
                    }
                }
            }
            airgroupvalues_full
        }
        #[cfg(not(distributed))]
        {
            airgroupvalues
                .into_iter()
                .map(|inner_vec| inner_vec.into_iter().map(|x| F::from_u64(x)).collect::<Vec<F>>())
                .collect()
        }
    }

    pub fn distribute_publics(&self, publics: Vec<u64>) -> Vec<u64> {
        #[cfg(distributed)]
        {
            let local_size = publics.len() as i32;
            let mut sizes: Vec<i32> = vec![0; self.n_processes as usize];
            self.world.all_gather_into(&local_size, &mut sizes);

            // Compute displacements and total size
            let mut displacements: Vec<i32> = vec![0; self.n_processes as usize];
            for i in 1..self.n_processes as usize {
                displacements[i] = displacements[i - 1] + sizes[i - 1];
            }

            let total_size: i32 = sizes.iter().sum();

            // Flattened buffer to receive all the data
            let mut all_publics: Vec<u64> = vec![0; total_size as usize];

            let publics_sizes = &sizes;
            let publics_displacements = &displacements;

            let mut partitioned_all_publics =
                PartitionMut::new(&mut all_publics, publics_sizes.as_slice(), publics_displacements.as_slice());

            // Use all_gather_varcount_into to gather all data from all processes
            self.world.all_gather_varcount_into(&publics, &mut partitioned_all_publics);

            // Each process will now have the same complete dataset
            all_publics
        }
        #[cfg(not(distributed))]
        {
            publics
        }
    }

    pub fn distribute_multiplicity(&self, _multiplicity: &[AtomicU64], _owner: i32) {
        #[cfg(distributed)]
        {
            //assert that I can operate with u32
            assert!(_multiplicity.len() < u32::MAX as usize);

            if _owner != self.rank {
                //pack multiplicities in a sparce vector
                let mut packed_multiplicity = Vec::new();
                packed_multiplicity.push(0u32); //this will be the counter
                for (idx, mul) in _multiplicity.iter().enumerate() {
                    let m = mul.load(Ordering::Relaxed);
                    if m != 0 {
                        assert!(m < u32::MAX as u64);
                        packed_multiplicity.push(idx as u32);
                        packed_multiplicity.push(m as u32);
                        packed_multiplicity[0] += 2;
                    }
                }
                self.world.process_at_rank(_owner).send(&packed_multiplicity[..]);
            } else {
                let mut packed_multiplicity: Vec<u32> = vec![0; _multiplicity.len() * 2 + 1];
                for i in 0..self.n_processes {
                    if i != _owner {
                        self.world.process_at_rank(i).receive_into(&mut packed_multiplicity);
                        for j in (1..packed_multiplicity[0]).step_by(2) {
                            let idx = packed_multiplicity[j as usize] as usize;
                            let m = packed_multiplicity[j as usize + 1] as u64;
                            _multiplicity[idx].fetch_add(m, Ordering::Relaxed);
                        }
                    }
                }
            }
        }
    }

    pub fn distribute_multiplicities(&self, _multiplicities: &[Vec<AtomicU64>], _owner: i32) {
        #[cfg(distributed)]
        {
            // Ensure that each multiplicity vector can be operated with u32
            let mut buff_size = 0;
            for multiplicity in _multiplicities.iter() {
                assert!(multiplicity.len() < u32::MAX as usize);
                buff_size += multiplicity.len() + 1;
            }

            let n_columns = _multiplicities.len();
            if _owner != self.rank {
                // Pack multiplicities in a sparse vector
                let mut packed_multiplicities = vec![0u32; n_columns];
                for (col_idx, multiplicity) in _multiplicities.iter().enumerate() {
                    for (idx, mul) in multiplicity.iter().enumerate() {
                        let m = mul.load(Ordering::Relaxed);
                        if m != 0 {
                            assert!(m < u32::MAX as u64);
                            packed_multiplicities[col_idx] += 1;
                            packed_multiplicities.push(idx as u32);
                            packed_multiplicities.push(m as u32);
                        }
                    }
                }
                self.world.process_at_rank(_owner).send(&packed_multiplicities[..]);
            } else {
                let mut packed_multiplicities: Vec<u32> = vec![0; buff_size * 2];
                for i in 0..self.n_processes {
                    if i != _owner {
                        self.world.process_at_rank(i).receive_into(&mut packed_multiplicities);

                        // Read counters
                        let mut counters = vec![0usize; n_columns];
                        for col_idx in 0..n_columns {
                            counters[col_idx] = packed_multiplicities[col_idx] as usize;
                        }

                        // Unpack multiplicities
                        let mut idx = n_columns;
                        for col_idx in 0..n_columns {
                            for _ in 0..counters[col_idx] {
                                let row_idx = packed_multiplicities[idx] as usize;
                                let m = packed_multiplicities[idx + 1] as u64;
                                _multiplicities[col_idx][row_idx].fetch_add(m, Ordering::Relaxed);
                                idx += 2;
                            }
                        }
                    }
                }
            }
        }
    }

    #[allow(unused_variables)]
    pub fn distribute_recursive2_proofs(&self, alives: &[usize], proofs: &mut [Vec<Option<Vec<u64>>>]) {
        #[cfg(distributed)]
        {
            // Count number of aggregations that will be done
            let n_groups = alives.len();
            let n_agregations: usize = alives.iter().map(|&alive| alive.div_ceil(3)).sum();
            let aggs_per_process = (n_agregations / self.n_processes as usize).max(1);

            let mut i_proof = 0;
            // tags codes:
            // 0,...,ngroups-1: proofs that need to be sent to rank0 from another rank for a group with alive == 1
            // ngroups, ..., ngroups + 2*n_aggregations - 1: proofs that need to be sent to the owner of the aggregation task

            for (group_idx, &alive) in alives.iter().enumerate() {
                let group_proofs: &mut Vec<Option<Vec<u64>>> = &mut proofs[group_idx];
                let n_aggs_group = alive.div_ceil(3);

                if n_aggs_group == 0 {
                    assert!(alive == 1);
                    if self.rank == 0 {
                        if group_proofs[0].is_none() {
                            // Receive proof from the owner process
                            let tag = group_idx as i32;
                            let (msg, _status) = self.world.any_process().receive_vec_with_tag::<u64>(tag);
                            group_proofs[0] = Some(msg);
                        }
                    } else if let Some(proof) = group_proofs[0].take() {
                        let tag = group_idx as i32;
                        self.world.process_at_rank(0).send_with_tag(&proof[..], tag);
                    }
                }

                for i in 0..n_aggs_group {
                    let chunk = i_proof / aggs_per_process;
                    let owner_rank =
                        if chunk < self.n_processes as usize { chunk } else { i_proof % self.n_processes as usize };
                    let left_idx = i * 3;
                    let mid_idx = i * 3 + 1;
                    let right_idx = i * 3 + 2;

                    if owner_rank == self.rank as usize {
                        for &idx in &[left_idx, mid_idx, right_idx] {
                            if idx < alive && group_proofs[idx].is_none() {
                                let tag = if idx == left_idx {
                                    i_proof * 3 + n_groups
                                } else if idx == mid_idx {
                                    i_proof * 3 + n_groups + 1
                                } else {
                                    i_proof * 3 + n_groups + 2
                                };
                                let (msg, _status) = self.world.any_process().receive_vec_with_tag::<u64>(tag as i32);
                                group_proofs[idx] = Some(msg);
                            }
                        }
                    } else if self.n_processes > 1 {
                        for &idx in &[left_idx, mid_idx, right_idx] {
                            if idx < alive {
                                if let Some(proof) = group_proofs[idx].take() {
                                    let tag = if idx == left_idx {
                                        i_proof * 3 + n_groups
                                    } else if idx == mid_idx {
                                        i_proof * 3 + n_groups + 1
                                    } else {
                                        i_proof * 3 + n_groups + 2
                                    };
                                    self.world.process_at_rank(owner_rank as i32).send_with_tag(&proof[..], tag as i32);
                                }
                            }
                        }
                    }
                    i_proof += 1;
                }
            }
        }
    }
}

unsafe impl Send for MpiCtx {}
unsafe impl Sync for MpiCtx {}
