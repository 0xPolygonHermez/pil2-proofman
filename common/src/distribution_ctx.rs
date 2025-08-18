use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InstanceInfo {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub table: bool,
    pub threads_witness: usize,
    pub n_chunks: usize,
    pub range: (usize, usize),
}

impl InstanceInfo {
    pub fn new(airgroup_id: usize, air_id: usize, table: bool, threads_witness: usize) -> Self {
        Self { airgroup_id, air_id, table, threads_witness, n_chunks: 1, range: (0, 0) }
    }
}

/// Represents the context of distributed computing
#[derive(Default, Clone)]
pub struct DistributionCtx {
    pub n_instances: usize,
    pub my_instances: Vec<usize>,
    pub instances: Vec<InstanceInfo>,
    pub instances_owner: Vec<(i32, usize, u64)>, //owner_rank, owner_instance_idx, weight
    pub owners_count: Vec<i32>,
    pub owners_weight: Vec<u64>,
    pub airgroup_instances_alives: Vec<Vec<usize>>,
    pub balance_distribution: bool,
    pub n_processes: Option<usize>,
    pub rank: Option<i32>,
}

impl std::fmt::Debug for DistributionCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("DistributionCtx");
        dbg.field("n_instances", &self.n_instances)
            .field("my_instances", &self.my_instances)
            .field("instances", &self.instances)
            .field("instances_owner", &self.instances_owner)
            .field("owners_count", &self.owners_count)
            .field("owners_weight", &self.owners_weight)
            .field("airgroup_instances_alives", &self.airgroup_instances_alives)
            .field("balance_distribution", &self.balance_distribution)
            .field("n_processes", &self.n_processes)
            .field("rank", &self.rank);

        dbg.finish()
    }
}

impl DistributionCtx {
    pub fn new() -> Self {
        DistributionCtx {
            n_instances: 0,
            my_instances: Vec::new(),
            instances: Vec::new(),
            instances_owner: Vec::new(),
            owners_count: Vec::new(),
            owners_weight: Vec::new(),
            airgroup_instances_alives: Vec::new(),
            balance_distribution: true,
            n_processes: None,
            rank: None,
        }
    }

    pub fn add_rank(&mut self, n_processes: usize, rank: i32) {
        self.n_processes = Some(n_processes);
        self.rank = Some(rank);
        self.owners_count = vec![0; n_processes];
        self.owners_weight = vec![0; n_processes];
    }

    pub fn reset(&mut self) {
        self.n_instances = 0;
        self.my_instances.clear();
        self.instances.clear();
        self.instances_owner.clear();

        self.airgroup_instances_alives.clear();
        self.balance_distribution = true;
    }

    #[inline]
    pub fn is_my_instance(&self, global_idx: usize) -> bool {
        self.owner(global_idx) == self.rank.unwrap()
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
    pub fn add_instance(&mut self, airgroup_id: usize, air_id: usize, threads_witness: usize, weight: u64) -> usize {
        let idx = self.instances.len();
        self.instances.push(InstanceInfo::new(airgroup_id, air_id, false, threads_witness));
        self.n_instances += 1;
        let new_owner = (idx % self.n_processes.unwrap()) as i32;
        let count = self.owners_count[new_owner as usize] as usize;
        self.instances_owner.push((new_owner, count, weight));
        self.owners_count[new_owner as usize] += 1;
        self.owners_weight[new_owner as usize] += weight;
        if new_owner == self.rank.unwrap() {
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
        threads_witness: usize,
        weight: u64,
    ) -> usize {
        let idx = self.instances.len();
        self.instances.push(InstanceInfo::new(airgroup_id, air_id, false, threads_witness));
        self.n_instances += 1;
        let count = self.owners_count[owner_idx] as usize;
        self.instances_owner.push((owner_idx as i32, count, weight));
        self.owners_count[owner_idx] += 1;
        self.owners_weight[owner_idx] += weight;
        if owner_idx as i32 == self.rank.unwrap() {
            self.my_instances.push(idx);
        }
        idx
    }

    #[inline]
    pub fn add_instance_no_assign(
        &mut self,
        airgroup_id: usize,
        air_id: usize,
        threads_witness: usize,
        weight: u64,
    ) -> usize {
        self.instances.push(InstanceInfo::new(airgroup_id, air_id, false, threads_witness));
        self.instances_owner.push((-1, 0, weight));
        self.n_instances += 1;
        self.n_instances - 1
    }

    pub fn add_instance_no_assign_table(&mut self, airgroup_id: usize, air_id: usize, weight: u64) -> usize {
        let mut idx = 0;
        for rank in 0..self.n_processes.unwrap() {
            self.n_instances += 1;
            self.instances.push(InstanceInfo::new(airgroup_id, air_id, true, 1));
            let new_owner = rank;
            let count = self.owners_count[new_owner] as usize;
            self.instances_owner.push((new_owner as i32, count, weight));
            self.owners_count[new_owner] += 1;
            self.owners_weight[new_owner] += weight;
            if new_owner == self.rank.unwrap() as usize {
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
                    if owner_idx == self.rank.unwrap() as usize {
                        self.my_instances.push(*idx);
                    }
                    owner_idx = (owner_idx + 1) % self.n_processes.unwrap();
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
                if min_weight_idx == self.rank.unwrap() as usize {
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
                    if owner_idx == self.rank.unwrap() as usize {
                        self.my_instances.push(idx);
                    }
                    owner_idx = (owner_idx + 1) % self.n_processes.unwrap();
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
        for i in 0..self.n_processes.unwrap() {
            average_weight += self.owners_weight[i] as f64;
            if self.owners_weight[i] > max_weight {
                max_weight = self.owners_weight[i];
            }
            if self.owners_weight[i] < min_weight {
                min_weight = self.owners_weight[i];
            }
        }
        average_weight /= self.n_processes.unwrap() as f64;
        let max_deviation = max_weight as f64 / average_weight;
        (average_weight, max_weight, min_weight, max_deviation)
    }

    pub fn close(&mut self, n_airgroups: usize) {
        //Calculate for each airgroup how many processes have instances of that airgroup alive
        self.airgroup_instances_alives = vec![vec![0; self.n_processes.unwrap()]; n_airgroups];
        for (idx, &instance_info) in self.instances.iter().enumerate() {
            let owner = self.instances_owner[idx].0;
            self.airgroup_instances_alives[instance_info.airgroup_id][owner as usize] = 1;
        }
    }
}
