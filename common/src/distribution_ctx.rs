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
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
#[cfg(distributed)]
use std::sync::atomic::Ordering;

use fields::PrimeField64;
#[cfg(distributed)]
use crate::ExtensionField;
use crate::GlobalInfo;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InstanceInfo {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub table: bool,
    pub min_threads_witness: usize,
    pub n_chunks: usize,
    pub range: (usize, usize),
}

impl InstanceInfo {
    pub fn new(airgroup_id: usize, air_id: usize, table: bool, min_threads_witness: usize) -> Self {
        Self { airgroup_id, air_id, table, min_threads_witness, n_chunks: 1, range: (0, 0) }
    }
}

/// Represents the context of distributed computing
pub struct DistributionCtx {
    pub rank: i32,
    pub n_processes: i32,
    #[cfg(distributed)]
    pub universe: Universe,
    #[cfg(distributed)]
    pub world: mpi::topology::SimpleCommunicator,
    pub n_instances: usize,
    pub my_instances: Vec<usize>,
    pub instances: Vec<InstanceInfo>,
    pub instances_owner: Vec<(i32, usize, u64)>, //owner_rank, owner_instance_idx, weight
    pub owners_count: Vec<i32>,
    pub owners_weight: Vec<u64>,
    pub airgroup_instances_alives: Vec<Vec<usize>>,
    pub balance_distribution: bool,
    pub node_rank: i32,
    pub node_n_processes: i32,
}

impl std::fmt::Debug for DistributionCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(distributed)]
        {
            f.debug_struct("DistributionCtx")
                .field("rank", &self.rank)
                .field("n_processes", &self.n_processes)
                .field("n_instances", &self.n_instances)
                .field("my_instances", &self.my_instances)
                .field("instances", &self.instances)
                .field("instances_owner", &self.instances_owner)
                .field("owners_count", &self.owners_count)
                .field("owners_weight", &self.owners_weight)
                .field("airgroup_instances_alives", &self.airgroup_instances_alives)
                .field("balance_distribution", &self.balance_distribution)
                .field("node_rank", &self.node_rank)
                .field("node_n_processes", &self.node_n_processes)
                .finish()
        }
        #[cfg(not(distributed))]
        {
            f.debug_struct("DistributionCtx")
                .field("rank", &self.rank)
                .field("n_processes", &self.n_processes)
                .field("n_instances", &self.n_instances)
                .field("my_instances", &self.my_instances)
                .field("instances", &self.instances)
                .field("instances_owner", &self.instances_owner)
                .field("owners_count", &self.owners_count)
                .field("owners_weight", &self.owners_weight)
                .field("airgroup_instances_alives", &self.airgroup_instances_alives)
                .field("balance_distribution", &self.balance_distribution)
                .field("node_rank", &self.node_rank)
                .field("node_n_processes", &self.node_n_processes)
                .finish()
        }
    }
}

impl DistributionCtx {
    pub fn new() -> Self {
        #[cfg(distributed)]
        {
            Self::with_universe(None)
        }
        #[cfg(not(distributed))]
        {
            DistributionCtx {
                rank: 0,
                n_processes: 1,
                n_instances: 0,
                my_instances: Vec::new(),
                instances: Vec::new(),
                instances_owner: Vec::new(),
                owners_count: vec![0; 1],
                owners_weight: vec![0; 1],
                airgroup_instances_alives: Vec::new(),
                balance_distribution: false,
                node_rank: 0,
                node_n_processes: 1,
            }
        }
    }

    #[cfg(distributed)]
    pub fn with_universe(mpi_universe: Option<Universe>) -> Self {
        let universe = mpi_universe.unwrap_or_else(|| {
            let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Multiple)
                .expect("Failed to initialize MPI with threading");
            universe
        });

        let world = universe.world();
        let rank = world.rank();
        let n_processes = world.size();

        let local_comm = world.split_shared(rank);
        let node_rank = local_comm.rank();
        let node_n_processes = local_comm.size();

        DistributionCtx {
            rank,
            n_processes,
            universe,
            world,
            n_instances: 0,
            my_instances: Vec::new(),
            instances: Vec::new(),
            instances_owner: Vec::new(),
            owners_count: vec![0; n_processes as usize],
            owners_weight: vec![0; n_processes as usize],
            airgroup_instances_alives: Vec::new(),
            balance_distribution: true,
            node_rank,
            node_n_processes,
        }
    }

    pub fn reset(&mut self) {
        self.n_instances = 0;
        self.my_instances.clear();
        self.instances.clear();
        self.instances_owner.clear();

        self.owners_count = vec![0; self.n_processes as usize];
        self.owners_weight = vec![0; self.n_processes as usize];

        self.airgroup_instances_alives.clear();
        self.balance_distribution = true;
    }

    #[inline]
    pub fn barrier(&self) {
        #[cfg(distributed)]
        {
            self.world.barrier();
        }
    }

    #[inline]
    pub fn is_distributed(&self) -> bool {
        self.n_processes > 1
    }

    #[inline]
    pub fn is_my_instance(&self, global_idx: usize) -> bool {
        self.owner(global_idx) == self.rank
    }

    #[inline]
    pub fn owner(&self, global_idx: usize) -> i32 {
        self.instances_owner[global_idx].0
    }

    #[inline]
    pub fn get_instance_info(&self, global_idx: usize) -> (usize, usize) {
        (self.instances[global_idx].airgroup_id, self.instances[global_idx].air_id)
    }

    #[inline]
    pub fn get_instance_idx(&self, global_idx: usize) -> usize {
        self.my_instances.iter().position(|&x| x == global_idx).unwrap()
    }

    #[inline]
    pub fn get_instance_chunks(&self, global_idx: usize) -> usize {
        self.instances[global_idx].n_chunks
    }

    #[inline]
    pub fn set_balance_distribution(&mut self, balance: bool) {
        self.balance_distribution = balance;
    }

    #[inline]
    pub fn find_air_instance_id(&self, global_idx: usize) -> usize {
        let mut air_instance_id = 0;
        let (airgroup_id, air_id) = self.get_instance_info(global_idx);
        for idx in 0..global_idx {
            let (instance_airgroup_id, instance_air_id) = self.get_instance_info(idx);
            if (instance_airgroup_id, instance_air_id) == (airgroup_id, air_id) {
                air_instance_id += 1;
            }
        }
        air_instance_id
    }

    #[inline]
    pub fn find_instance_mine(&self, airgroup_id: usize, air_id: usize) -> (bool, usize) {
        let mut matches = self
            .my_instances
            .iter()
            .enumerate()
            .filter(|&(_pos, &id)| {
                let inst = &self.instances[id];
                inst.airgroup_id == airgroup_id && inst.air_id == air_id
            })
            .map(|(pos, _)| pos);

        match (matches.next(), matches.next()) {
            (None, _) => (false, 0),
            (Some(pos), None) => (true, pos),
            (Some(_), Some(_)) => {
                panic!("Multiple instances found for airgroup_id: {airgroup_id}, air_id: {air_id}");
            }
        }
    }

    #[inline]
    pub fn find_instance_id(&self, airgroup_id: usize, air_id: usize, air_instance_id: usize) -> Option<usize> {
        let mut count = 0;
        for (global_idx, instance) in self.instances.iter().enumerate() {
            let (inst_airgroup_id, inst_air_id) = (instance.airgroup_id, instance.air_id);
            if airgroup_id == inst_airgroup_id && air_id == inst_air_id {
                if count == air_instance_id {
                    return Some(global_idx);
                }
                count += 1;
            }
        }
        None
    }

    #[inline]
    pub fn is_min_rank_owner(&self, airgroup_id: usize, air_id: usize) -> bool {
        let mut min_owner = self.n_processes + 1;
        for (idx, instance) in self.instances.iter().enumerate() {
            let (inst_airgroup_id, inst_air_id) = (instance.airgroup_id, instance.air_id);
            if airgroup_id == inst_airgroup_id && air_id == inst_air_id && self.instances_owner[idx].0 < min_owner {
                min_owner = self.instances_owner[idx].0;
            }
        }

        if min_owner == self.n_processes + 1 {
            panic!("No instance found for airgroup_id: {airgroup_id}, air_id: {air_id}");
        }

        min_owner == self.rank
    }

    #[inline]
    pub fn add_instance(
        &mut self,
        airgroup_id: usize,
        air_id: usize,
        min_threads_witness: usize,
        weight: u64,
    ) -> usize {
        let idx = self.instances.len();
        self.instances.push(InstanceInfo::new(airgroup_id, air_id, false, min_threads_witness));
        self.n_instances += 1;
        let new_owner = (idx % self.n_processes as usize) as i32;
        let count = self.owners_count[new_owner as usize] as usize;
        self.instances_owner.push((new_owner, count, weight));
        self.owners_count[new_owner as usize] += 1;
        self.owners_weight[new_owner as usize] += weight;
        if new_owner == self.rank {
            self.my_instances.push(idx);
        }
        idx
    }

    #[inline]
    pub fn add_instance_assign_rank(
        &mut self,
        airgroup_id: usize,
        air_id: usize,
        owner_idx: usize,
        min_threads_witness: usize,
        weight: u64,
    ) -> usize {
        let idx = self.instances.len();
        self.instances.push(InstanceInfo::new(airgroup_id, air_id, false, min_threads_witness));
        self.n_instances += 1;
        let count = self.owners_count[owner_idx] as usize;
        self.instances_owner.push((owner_idx as i32, count, weight));
        self.owners_count[owner_idx] += 1;
        self.owners_weight[owner_idx] += weight;
        if owner_idx as i32 == self.rank {
            self.my_instances.push(idx);
        }
        idx
    }

    #[inline]
    pub fn add_instance_no_assign(
        &mut self,
        airgroup_id: usize,
        air_id: usize,
        min_threads_witness: usize,
        weight: u64,
    ) -> usize {
        self.instances.push(InstanceInfo::new(airgroup_id, air_id, false, min_threads_witness));
        self.instances_owner.push((-1, 0, weight));
        self.n_instances += 1;
        self.n_instances - 1
    }

    pub fn add_instance_no_assign_table(&mut self, airgroup_id: usize, air_id: usize, weight: u64) -> usize {
        let mut idx = 0;
        for rank in 0..self.n_processes {
            self.n_instances += 1;
            self.instances.push(InstanceInfo::new(airgroup_id, air_id, true, PreCalculate::None, 1));
            let new_owner = rank;
            let count = self.owners_count[new_owner as usize] as usize;
            self.instances_owner.push((new_owner, count, weight));
            self.owners_count[new_owner as usize] += 1;
            self.owners_weight[new_owner as usize] += weight;
            if new_owner == self.rank {
                self.my_instances.push(self.instances.len() - 1);
                idx = self.instances.len() - 1;
            }
        }
        idx
    }

    pub fn set_chunks(&mut self, global_idx: usize, chunks: Vec<usize>) {
        let instance_info = &mut self.instances[global_idx];
        instance_info.n_chunks = chunks.len();
        if let (Some(&first), Some(&last)) = (chunks.first(), chunks.last()) {
            instance_info.range = (first, last);
        }
    }

    pub fn assign_instances(&mut self, minimal_memory: bool) {
        if self.balance_distribution {
            if minimal_memory {
                // Sort unassigned instances according to wc_weights
                let mut unassigned_instances = Vec::new();
                for (idx, &(owner, _, _)) in self.instances_owner.iter().enumerate() {
                    if owner == -1 {
                        unassigned_instances.push((idx, self.instances[idx].n_chunks));
                    }
                }

                // Sort the unassigned instances by weight
                unassigned_instances.sort_by(|a, b| b.1.cmp(&a.1));

                // Assign half of the unassigned instances in round-robin fashion
                let mut owner_idx = 0;
                for (idx, _) in unassigned_instances.iter().take(unassigned_instances.len() / 2) {
                    self.instances_owner[*idx].0 = owner_idx as i32;
                    self.instances_owner[*idx].1 = self.owners_count[owner_idx] as usize;
                    self.owners_count[owner_idx] += 1;
                    self.owners_weight[owner_idx] += self.instances_owner[*idx].2;
                    if owner_idx == self.rank as usize {
                        self.my_instances.push(*idx);
                    }
                    owner_idx = (owner_idx + 1) % self.n_processes as usize;
                }
            }

            // Sort the unassigned instances by proof weight
            let mut unassigned_instances = Vec::new();
            for (idx, &(owner, _, weight)) in self.instances_owner.iter().enumerate() {
                if owner == -1 {
                    unassigned_instances.push((idx, weight));
                }
            }

            // Sort the unassigned instances by proof weight
            unassigned_instances.sort_by(|a, b| b.1.cmp(&a.1));

            // Distribute the unassigned instances to the process with minimum weight each time
            // cost: O(n^2) may be optimized if needed
            for (idx, _) in unassigned_instances {
                let mut min_weight = u64::MAX;
                let mut min_weight_idx = 0;
                for (i, &weight) in self.owners_weight.iter().enumerate() {
                    if weight < min_weight {
                        min_weight = weight;
                        min_weight_idx = i;
                    } else if (min_weight == weight) && (self.owners_count[i] < self.owners_count[min_weight_idx]) {
                        min_weight_idx = i;
                    }
                }
                self.instances_owner[idx].0 = min_weight_idx as i32;
                self.instances_owner[idx].1 = self.owners_count[min_weight_idx] as usize;
                self.owners_count[min_weight_idx] += 1;
                self.owners_weight[min_weight_idx] += self.instances_owner[idx].2;
                if min_weight_idx == self.rank as usize {
                    self.my_instances.push(idx);
                }
            }
        } else {
            let mut air_info = HashMap::new();
            for (idx, &(owner, _, _)) in self.instances_owner.iter().enumerate() {
                if owner == -1 {
                    let (airgroup_id, air_id) = self.get_instance_info(idx);
                    air_info.entry((airgroup_id, air_id)).or_insert_with(Vec::new).push(idx);
                }
            }

            // Sort groups descending by size
            let mut grouped_instances: Vec<_> = air_info.into_iter().collect();
            grouped_instances.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

            // Step 1: Find the best starting point
            let (start_idx, _) = self.owners_count.iter().enumerate().min_by_key(|&(_, count)| count).unwrap();

            // Step 2: Round-robin from there
            let mut owner_idx = start_idx;

            for (_, instance_indices) in grouped_instances {
                for &idx in &instance_indices {
                    self.instances_owner[idx].0 = owner_idx as i32;
                    self.instances_owner[idx].1 = self.owners_count[owner_idx] as usize;
                    self.owners_count[owner_idx] += 1;
                    self.owners_weight[owner_idx] += self.instances_owner[idx].2;
                    if owner_idx == self.rank as usize {
                        self.my_instances.push(idx);
                    }
                    owner_idx = (owner_idx + 1) % self.n_processes as usize;
                }
            }
        }
    }

    // Returns the maximum weight deviation from the average weight
    // This is calculated as the maximum weight divided by the average weight
    pub fn load_balance_info(&self) -> (f64, u64, u64, f64) {
        let mut average_weight = 0.0;
        let mut max_weight = 0;
        let mut min_weight = u64::MAX;
        for i in 0..self.n_processes as usize {
            average_weight += self.owners_weight[i] as f64;
            if self.owners_weight[i] > max_weight {
                max_weight = self.owners_weight[i];
            }
            if self.owners_weight[i] < min_weight {
                min_weight = self.owners_weight[i];
            }
        }
        average_weight /= self.n_processes as f64;
        let max_deviation = max_weight as f64 / average_weight;
        (average_weight, max_weight, min_weight, max_deviation)
    }

    pub fn close(&mut self, n_airgroups: usize) {
        //Calculate for each airgroup how many processes have instances of that airgroup alive
        self.airgroup_instances_alives = vec![vec![0; self.n_processes as usize]; n_airgroups];
        for (idx, &instance_info) in self.instances.iter().enumerate() {
            let owner = self.instances_owner[idx].0;
            self.airgroup_instances_alives[instance_info.airgroup_id][owner as usize] = 1;
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
            values.to_vec()
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
            let size = self.n_processes;

            let local_size = publics.len() as i32;
            let mut sizes: Vec<i32> = vec![0; self.n_processes as usize];
            self.world.all_gather_into(&local_size, &mut sizes);

            // Compute displacements and total size
            let mut displacements: Vec<i32> = vec![0; size as usize];
            for i in 1..size as usize {
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
    pub fn distribute_recursive2_proofs(&mut self, alives: &[usize], proofs: &mut [Vec<Option<Vec<u64>>>]) {
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

impl Default for DistributionCtx {
    fn default() -> Self {
        DistributionCtx::new()
    }
}
unsafe impl Send for DistributionCtx {}
unsafe impl Sync for DistributionCtx {}
