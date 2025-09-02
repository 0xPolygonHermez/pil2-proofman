use core::panic;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InstanceInfo {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub table: bool,
    pub shared: bool,
    pub threads_witness: usize,
    pub n_chunks: usize,
    pub weight: u64,
}

impl InstanceInfo {
    pub fn new(
        airgroup_id: usize,
        air_id: usize,
        table: bool,
        shared: bool,
        threads_witness: usize,
        weight: u64,
    ) -> Self {
        Self { airgroup_id, air_id, table, threads_witness, shared, n_chunks: 0, weight }
    }
}

/// Context for distributed computing with two-level hierarchy:
/// 1. Distributed Prover MANAGER distributes PARTITIONS across WORKERS
/// 2. Instances are distributed across PARTITIONS as they are created
/// 3. Instances are distributed across local PROCESSES within each WORKER
#[derive(Default, Clone)]
pub struct DistributionCtx {
    // STATIC PARAMETERS:
    pub load_balancing: bool, // Whether to balance distribution (true) or use round-robin (false)

    // Worker-level
    pub n_partitions: usize,       // Total number of partitions in the system
    pub partition_mask: Vec<bool>, // Which partitions are assigned to this worker

    // Process-level
    pub n_processes: usize, // Number of processes used by this worker
    pub process_id: usize,  // ID of current process within this worker

    // DYNAMIC PARAMETERS

    // Instances
    pub n_instances: usize,            // Total number of instances
    pub instances: Vec<InstanceInfo>,  // Instances info
    pub n_non_tables: usize,           // Number of non-table instances
    pub n_tables: usize,               // Number of table instances
    pub aux_tables: Vec<InstanceInfo>, // Table instances info (lately appended to instances)
    pub aux_table_map: Vec<i32>,       // Map from aux tables to original instances

    // Worker-level distribution
    pub instance_partition: Vec<i32>, // Which partition each instance belongs to (>=0 assigned, -1 unassigned, -2 appended table)
    pub worker_instances: Vec<usize>, // Indexes of instances assigned to this worker
    pub partition_count: Vec<u32>,    // #instances in each partition (does not include tables)
    pub partition_weight: Vec<u64>,   // Total computational weight per partition (does not include tables)

    // Process-level distribution
    pub instance_process: Vec<(i32, usize)>, // For each instance: (process_id or -1 if other worker, local_idx)
    pub process_instances: Vec<usize>,       // Indexes of instances assigned to current process
    pub process_count: Vec<usize>,           // #instances assigned to each process
    pub process_weight: Vec<u64>,            // Total computational weight per process

    // Control
    pub assignation_done: bool, // Whether the instance assignation is done
}

impl std::fmt::Debug for DistributionCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("DistributionCtx");

        // STATIC PARAMETERS
        dbg.field("=== STATIC PARAMS ===", &"");
        dbg.field("load_balancing", &self.load_balancing)
            .field("n_partitions", &self.n_partitions)
            .field("partition_mask", &self.partition_mask)
            .field("n_processes", &self.n_processes)
            .field("process_id", &self.process_id);

        // DYNAMIC PARAMETERS
        dbg.field("=== DYNAMIC PARAMS ===", &"");
        dbg.field("n_instances", &self.n_instances)
            .field("instances", &self.instances)
            .field("n_non_tables", &self.n_non_tables)
            .field("n_tables", &self.n_tables)
            .field("tables", &self.aux_tables)
            .field("instance_partition", &self.instance_partition)
            .field("worker_instances", &self.worker_instances)
            .field("partition_count", &self.partition_count)
            .field("partition_weight", &self.partition_weight)
            .field("instance_process", &self.instance_process)
            .field("process_instances", &self.process_instances)
            .field("process_count", &self.process_count)
            .field("process_weight", &self.process_weight)
            .field("assignation_done", &self.assignation_done);
        dbg.finish()
    }
}

impl DistributionCtx {
    pub fn new() -> Self {
        DistributionCtx {
            load_balancing: true,
            n_partitions: 0,
            partition_mask: Vec::new(),
            n_processes: 0,
            process_id: 0,
            n_instances: 0,
            instances: Vec::new(),
            n_non_tables: 0,
            n_tables: 0,
            aux_tables: Vec::new(),
            aux_table_map: Vec::new(),
            instance_partition: Vec::new(),
            worker_instances: Vec::new(),
            partition_count: Vec::new(),
            partition_weight: Vec::new(),
            instance_process: Vec::new(),
            process_instances: Vec::new(),
            process_count: Vec::new(),
            process_weight: Vec::new(),
            assignation_done: false,
        }
    }

    /// Configure the static partitioning parameters
    /// - `n_partitions`: Total number of partitions in the distributed system
    /// - `partition_ids`: Which partition IDs are assigned to this worker
    /// - `balance`: Whether to balance load across partitions or use round-robin
    pub fn setup_partitions(&mut self, n_partitions: usize, partition_ids: Vec<u32>, balance: bool) {
        self.load_balancing = balance;
        self.n_partitions = n_partitions;
        self.partition_mask = vec![false; n_partitions];

        for id in partition_ids {
            if id < n_partitions as u32 {
                self.partition_mask[id as usize] = true;
            } else {
                panic!("Partition ID {} exceeds total partitions {}", id, n_partitions);
            }
        }

        self.partition_count = vec![0; n_partitions];
        self.partition_weight = vec![0; n_partitions];
    }

    /// Configure the static processes parameters
    /// - `n_processes`: Number of processes available to this worker
    /// - `process_id`: The rank/ID of the current process (must be < n_processes)
    pub fn setup_processes(&mut self, n_processes: usize, process_id: usize) {
        if process_id >= n_processes {
            panic!("Process rank {} exceeds total processes {}", process_id, n_processes);
        }
        self.n_processes = n_processes;
        self.process_id = process_id;
        self.process_count = vec![0; n_processes];
        self.process_weight = vec![0; n_processes];
    }

    /// Reset all DYNAMIC parameters for a new proof
    /// This clears all instance-specific data while preserving the STATIC configuration
    pub fn reset_instances(&mut self) {
        // Global instance management
        self.n_instances = 0;
        self.instances.clear();
        self.n_non_tables = 0;
        self.n_tables = 0;
        self.aux_tables.clear();
        self.aux_table_map.clear();

        // Worker-level
        self.instance_partition.clear();
        self.worker_instances.clear();
        self.partition_count.fill(0);
        self.partition_weight.fill(0);

        // Process-level
        self.instance_process.clear();
        self.process_instances.clear();
        self.process_count.fill(0);
        self.process_weight.fill(0);

        //control
        self.assignation_done = false
    }

    /// Verify that the static configuration has been properly set up
    #[inline]
    pub fn validate_static_config(&self) -> Result<(), String> {
        // Check partition configuration
        if self.n_partitions == 0 {
            return Err("Partition configuration not set. Call setup_partitions() first.".to_string());
        }
        if self.partition_mask.len() != self.n_partitions {
            return Err("Partition mask size mismatch with n_partitions".to_string());
        }

        // Check process configuration
        if self.n_processes == 0 {
            return Err("Process configuration not set. Call setup_processes() first.".to_string());
        }
        if self.process_id >= self.n_processes {
            return Err(format!("Invalid process rank {} >= {}", self.process_id, self.n_processes));
        }

        Ok(())
    }

    /// Check if the current process is the owner of a given instance
    #[inline]
    pub fn is_my_process_instance(&self, instance_id: usize) -> bool {
        if instance_id >= self.instance_process.len() {
            panic!("Instance index {} out of bounds (max: {})", instance_id, self.instance_process.len());
        }
        self.instance_process[instance_id].0 == self.process_id as i32
    }

    #[inline]
    pub fn get_process_owner_instance(&self, instance_id: usize) -> i32 {
        if instance_id >= self.instance_process.len() {
            panic!("Instance index {} out of bounds (max: {})", instance_id, self.instance_process.len());
        }
        let owner = self.instance_process[instance_id].0;
        assert!(owner != -1, "Instance {} is not owned by any process", instance_id);
        owner
    }

    /// Get the airgroup and air ID for a given instance
    /// Returns (airgroup_id, air_id)
    #[inline]
    pub fn get_instance_info(&self, instance_id: usize) -> (usize, usize) {
        if instance_id >= self.instances.len() {
            panic!("Instance index {} out of bounds (max: {})", instance_id, self.instances.len());
        }
        (self.instances[instance_id].airgroup_id, self.instances[instance_id].air_id)
    }

    /// Get the airgroup and air ID of a given table
    /// Returns (airgroup_id, air_id)
    #[inline]
    pub fn get_table_info(&self, table_idx: usize) -> (usize, usize) {
        if self.assignation_done {
            let instance_id = self.aux_table_map[table_idx] as usize;
            self.get_instance_info(instance_id)
        } else {
            if table_idx >= self.aux_tables.len() {
                panic!("Table index {} out of bounds (max: {})", table_idx, self.aux_tables.len());
            }
            (self.aux_tables[table_idx].airgroup_id, self.aux_tables[table_idx].air_id)
        }
    }

    pub fn get_table_instance_idx(&self, table_idx: usize) -> usize {
        if self.assignation_done {
            self.aux_table_map[table_idx] as usize
        } else {
            panic!("Table instances not yet assigned");
        }
    }

    /// Get the local index of the instance within its owner process
    #[inline]
    pub fn get_instance_local_idx(&self, instance_id: usize) -> usize {
        if instance_id >= self.instance_process.len() {
            panic!("Instance index {} out of bounds (max: {})", instance_id, self.instance_process.len());
        }
        self.instance_process[instance_id].1
    }

    /// Get the number of Minimum Trace chunks to be processes for a given global instance
    #[inline]
    pub fn get_instance_chunks(&self, instance_id: usize) -> usize {
        if instance_id >= self.instances.len() {
            panic!("Instance index {} out of bounds (max: {})", instance_id, self.instances.len());
        }
        self.instances[instance_id].n_chunks
    }

    /// Set the number of chunks for a given instance (these may be used for balancing purposes)
    pub fn set_n_chunks(&mut self, instance_id: usize, n_chunks: usize) {
        if instance_id >= self.instances.len() {
            panic!("Instance index {} out of bounds (max: {})", instance_id, self.instances.len());
        }
        let instance_info = &mut self.instances[instance_id];
        instance_info.n_chunks = n_chunks;
    }

    /// Check if the current worker is the owner of a given instance
    #[inline]
    pub fn is_my_worker_instance(&self, instance_id: usize) -> bool {
        if instance_id >= self.instance_process.len() {
            panic!("Instance index {} out of bounds (max: {})", instance_id, self.instance_process.len());
        }
        self.instance_process[instance_id].0 >= 0
    }

    /// Among the air instances with the same airgroup and air_id find the relative position of the current one
    #[inline]
    pub fn find_air_instance_id(&self, instance_id: usize) -> usize {
        if instance_id >= self.instances.len() {
            panic!("Instance index {} out of bounds (max: {})", instance_id, self.instances.len());
        }
        let mut air_instance_id = 0;
        let (airgroup_id, air_id) = self.get_instance_info(instance_id);
        for idx in 0..instance_id {
            let (instance_airgroup_id, instance_air_id) = self.get_instance_info(idx);
            if (instance_airgroup_id, instance_air_id) == (airgroup_id, air_id) {
                air_instance_id += 1;
            }
        }
        air_instance_id
    }

    /// Find an instance with the given airgroup_id and air_id among the current process's instances
    /// Returns (found, local_index)
    #[inline]
    pub fn find_process_instance(&self, airgroup_id: usize, air_id: usize) -> (bool, usize) {
        let mut matches = self
            .process_instances
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

    /// Find a table with the given airgroup_id and air_id among the current process's tables
    /// Returns (found, local_index)
    #[inline]
    pub fn find_process_table(&self, airgroup_id: usize, air_id: usize) -> (bool, usize) {
        if self.assignation_done {
            self.find_process_instance(airgroup_id, air_id)
        } else {
            let mut matches = self
                .aux_tables
                .iter()
                .enumerate()
                .filter(|&(_pos, &info)| info.airgroup_id == airgroup_id && info.air_id == air_id)
                .map(|(pos, _)| pos);

            match (matches.next(), matches.next()) {
                (None, _) => (false, 0),
                (Some(pos), None) => (true, pos),
                (Some(_), Some(_)) => {
                    panic!("Multiple tables found for airgroup_id: {airgroup_id}, air_id: {air_id}");
                }
            }
        }
    }

    /// Of all the instances with the same airgroup and air_id find the one with the given air_instance_id
    #[inline]
    pub fn find_instance_id(&self, airgroup_id: usize, air_id: usize, air_instance_id: usize) -> Option<usize> {
        let mut count = 0;
        for (instance_idx, instance) in self.instances.iter().enumerate() {
            let (inst_airgroup_id, inst_air_id) = (instance.airgroup_id, instance.air_id);
            if airgroup_id == inst_airgroup_id && air_id == inst_air_id {
                if count == air_instance_id {
                    return Some(instance_idx);
                }
                count += 1;
            }
        }
        None
    }

    /// add an instance and assign it to a partition/process based only in the gid
    /// the instance added is not a table
    #[inline]
    pub fn add_instance(&mut self, airgroup_id: usize, air_id: usize, threads_witness: usize, weight: u64) -> usize {
        if self.assignation_done {
            panic!("Instances already assigned");
        }
        self.validate_static_config().expect("Static configuration invalid or incomplete");
        let gid: usize = self.instances.len();
        self.instances.push(InstanceInfo::new(airgroup_id, air_id, false, false, threads_witness, weight));
        self.n_instances += 1;
        self.n_non_tables += 1;
        let partition_id = (gid % self.n_partitions) as u32;
        self.instance_partition.push(partition_id as i32);
        self.partition_count[partition_id as usize] += 1;
        self.partition_weight[partition_id as usize] += weight;
        let mut local_idx = 0;
        let mut owner = -1;
        if self.partition_mask[partition_id as usize] {
            let worker_instance_id = self.worker_instances.len();
            self.worker_instances.push(gid);
            let process_id = worker_instance_id % self.n_processes;
            owner = process_id as i32;
            local_idx = self.process_count[process_id];
            self.process_count[process_id] += 1;
            self.process_weight[process_id] += weight;
            if process_id == self.process_id {
                self.process_instances.push(gid);
            }
        }
        self.instance_process.push((owner, local_idx));
        gid
    }

    /// add an instance and assign it to a partition/process based only in the gid
    /// the instance added is not a table
    #[inline]
    pub fn add_instance_partition(
        &mut self,
        airgroup_id: usize,
        air_id: usize,
        partition_id: usize,
        threads_witness: usize,
        weight: u64,
    ) -> usize {
        if self.assignation_done {
            panic!("Instances already assigned");
        }
        self.validate_static_config().expect("Static configuration invalid or incomplete");
        let gid: usize = self.instances.len();
        self.instances.push(InstanceInfo::new(airgroup_id, air_id, false, false, threads_witness, weight));
        self.n_instances += 1;
        self.n_non_tables += 1;
        self.instance_partition.push(partition_id as i32);
        self.partition_count[partition_id] += 1;
        self.partition_weight[partition_id] += weight;
        let mut local_idx = 0;
        let mut owner = -1;
        if self.partition_mask[partition_id] {
            let worker_instance_id = self.worker_instances.len();
            self.worker_instances.push(gid);
            let process_id = worker_instance_id % self.n_processes;
            owner = process_id as i32;
            local_idx = self.process_count[process_id];
            self.process_count[process_id] += 1;
            self.process_weight[process_id] += weight;
            if process_id == self.process_id {
                self.process_instances.push(gid);
            }
        }
        self.instance_process.push((owner, local_idx));
        gid
    }

    /// add an instance without assigning it to any partition/process
    /// It will be assigned later by assign_instances()
    /// the instance added is not a table
    #[inline]
    pub fn add_instance_no_assign(
        &mut self,
        airgroup_id: usize,
        air_id: usize,
        threads_witness: usize,
        weight: u64,
    ) -> usize {
        if self.assignation_done {
            panic!("Instances already assigned");
        }
        self.validate_static_config().expect("Static configuration invalid or incomplete");
        self.instances.push(InstanceInfo::new(airgroup_id, air_id, false, false, threads_witness, weight));
        self.instance_partition.push(-1);
        self.instance_process.push((-1, 0_usize));
        self.n_instances += 1;
        self.n_non_tables += 1;
        self.n_instances - 1
    }

    /// Add local table instances
    pub fn add_table(&mut self, airgroup_id: usize, air_id: usize, weight: u64) -> usize {
        if self.assignation_done {
            panic!("Instances already assigned");
        }
        self.validate_static_config().expect("Static configuration invalid or incomplete");
        let lid = self.aux_tables.len();
        self.aux_tables.push(InstanceInfo::new(airgroup_id, air_id, true, true, 1, weight));
        self.aux_table_map.push(-1);
        self.n_tables += 1;
        lid
    }

    pub fn add_table_all(&mut self, airgroup_id: usize, air_id: usize, weight: u64) -> usize {
        if self.assignation_done {
            panic!("Instances already assigned");
        }
        self.validate_static_config().expect("Static configuration invalid or incomplete");
        let lid = self.aux_tables.len();
        self.aux_tables.push(InstanceInfo::new(airgroup_id, air_id, true, false, 1, weight));
        self.aux_table_map.push(-1);
        self.n_tables += 1;
        lid
    }

    /// Assign instances to partitions and processes
    pub fn assign_instances(&mut self, minimal_memory: bool) {
        if self.assignation_done {
            panic!("Instances already assigned");
        }
        //assign instances
        self.validate_static_config().expect("Static configuration invalid or incomplete");
        if self.load_balancing {
            if minimal_memory {
                // Sort unassigned instances according to n_chunks
                let mut unassigned_instances = Vec::new();
                for (gid, &partition_id) in self.instance_partition.iter().enumerate() {
                    if partition_id == -1 {
                        unassigned_instances.push((gid, self.instances[gid].n_chunks));
                    }
                }

                // Sort the unassigned instances by weight
                unassigned_instances.sort_by(|a, b| b.1.cmp(&a.1));

                // Assign half of the unassigned instances in round-robin fashion
                let mut partition_id: usize = 0;
                let mut process_id: usize = 0;
                for (gid, _) in unassigned_instances.iter().take(unassigned_instances.len() / 2) {
                    self.instance_partition[*gid] = partition_id as i32;
                    self.partition_count[partition_id] += 1;
                    self.partition_weight[partition_id] += self.instances[*gid].weight;
                    if self.partition_mask[partition_id] {
                        self.worker_instances.push(*gid);
                        let lid = self.process_count[process_id];
                        self.process_count[process_id] += 1;
                        self.process_weight[process_id] += self.instances[*gid].weight;
                        if process_id == self.process_id {
                            self.process_instances.push(*gid);
                        }
                        self.instance_process[*gid].0 = process_id as i32;
                        self.instance_process[*gid].1 = lid;
                        process_id = (process_id + 1) % self.n_processes;
                    }
                    partition_id = (partition_id + 1) % self.n_partitions;
                }
            }

            // Sort the unassigned instances by proof weight
            let mut unassigned_instances = Vec::new();
            for (gid, &partition_id) in self.instance_partition.iter().enumerate() {
                if partition_id == -1 {
                    unassigned_instances.push((gid, self.instances[gid].weight));
                }
            }

            // Sort the unassigned instances by proof weight
            unassigned_instances.sort_by(|a, b| b.1.cmp(&a.1));

            // Distribute the unassigned instances to the process with minimum weight each time
            // cost: O(n^2) may be optimized if needed
            for (gid, _) in unassigned_instances {
                let mut min_weight = u64::MAX;
                let mut partition_id = 0;
                for (i, &weight) in self.partition_weight.iter().enumerate() {
                    if weight < min_weight {
                        min_weight = weight;
                        partition_id = i;
                    } else if (min_weight == weight) && (self.partition_count[i] < self.partition_count[partition_id]) {
                        partition_id = i;
                    }
                }
                self.instance_partition[gid] = partition_id as i32;
                self.partition_count[partition_id] += 1;
                self.partition_weight[partition_id] += self.instances[gid].weight;
                if self.partition_mask[partition_id] {
                    self.worker_instances.push(gid);
                    let mut min_weight = u64::MAX;
                    let mut process_id = 0;
                    for (i, &weight) in self.process_weight.iter().enumerate() {
                        if weight < min_weight {
                            min_weight = weight;
                            process_id = i;
                        } else if (min_weight == weight) && (self.process_count[i] < self.process_count[process_id]) {
                            process_id = i;
                        }
                    }
                    let lid = self.process_count[process_id];
                    self.process_count[process_id] += 1;
                    self.process_weight[process_id] += self.instances[gid].weight;
                    if process_id == self.process_id {
                        self.process_instances.push(gid);
                    }
                    self.instance_process[gid].0 = process_id as i32;
                    self.instance_process[gid].1 = lid;
                }
            }
        } else {
            // Round-robin assignment for unassigned instances
            let mut partition_id_iter: usize = 0;
            let mut process_id_iter: usize = 0;
            for (gid, partition_id) in self.instance_partition.iter_mut().enumerate() {
                if *partition_id == -1 {
                    *partition_id = partition_id_iter as i32;
                    self.partition_count[*partition_id as usize] += 1;
                    self.partition_weight[*partition_id as usize] += self.instances[gid].weight;
                    if self.partition_mask[*partition_id as usize] {
                        self.worker_instances.push(gid);
                        let lid = self.process_count[process_id_iter];
                        self.process_count[process_id_iter] += 1;
                        self.process_weight[process_id_iter] += self.instances[gid].weight;
                        if process_id_iter == self.process_id {
                            self.process_instances.push(gid);
                        }
                        self.instance_process[gid].0 = process_id_iter as i32;
                        self.instance_process[gid].1 = lid;
                        process_id_iter = (process_id_iter + 1) % self.n_processes;
                    }
                    partition_id_iter = (partition_id_iter + 1) % self.n_partitions;
                }
            }
        }

        // Add tables that
        self.n_tables = 0;
        for (table_idx, table) in self.aux_tables.iter().enumerate() {
            if table.shared {
                let mut min_weight = u64::MAX;
                let mut process_id = 0;
                for (i, &weight) in self.process_weight.iter().enumerate() {
                    if weight < min_weight {
                        min_weight = weight;
                        process_id = i;
                    } else if (min_weight == weight) && (self.process_count[i] < self.process_count[process_id]) {
                        process_id = i;
                    }
                }
                let gid = self.instances.len();
                self.instances.push(*table);
                self.n_instances += 1;
                self.n_tables += 1;
                self.instance_partition.push(-2); // Mark as table
                self.worker_instances.push(gid);
                let lid = self.process_count[process_id];
                self.process_count[process_id] += 1;
                self.process_weight[process_id] += table.weight;
                if process_id == self.process_id {
                    self.process_instances.push(gid);
                }
                self.aux_table_map[table_idx] = gid as i32;
                self.instance_process.push((process_id as i32, lid));
            } else {
                for rank in 0..self.n_processes {
                    let gid = self.instances.len();
                    self.instances.push(InstanceInfo::new(
                        table.airgroup_id,
                        table.air_id,
                        true,
                        false,
                        table.threads_witness,
                        table.weight,
                    ));
                    self.n_instances += 1;
                    self.n_tables += 1;
                    self.instance_partition.push(-2); // Mark as table
                    self.worker_instances.push(gid);
                    let lid = self.process_count[rank];
                    self.process_count[rank] += 1;
                    self.process_weight[rank] += table.weight;
                    if rank == self.process_id {
                        self.process_instances.push(gid);
                        self.aux_table_map[table_idx] = gid as i32;
                    }
                    self.instance_process.push((rank as i32, lid));
                }
            }
        }
        self.aux_tables.clear();
        self.assignation_done = true;
    }

    ///  Load balance info for partitions
    ///  Does not include tables
    pub fn load_balance_info_partition(&self) -> (f64, u64, u64, f64) {
        let mut average_partition_weight = 0.0;
        let mut max_partition_weight = 0;
        let mut min_partition_weight = u64::MAX;
        for i in 0..self.n_partitions {
            average_partition_weight += self.partition_weight[i] as f64;
            if self.partition_weight[i] > max_partition_weight {
                max_partition_weight = self.partition_weight[i];
            }
            if self.partition_weight[i] < min_partition_weight {
                min_partition_weight = self.partition_weight[i];
            }
        }
        average_partition_weight /= self.n_partitions as f64;
        let max_deviation = max_partition_weight as f64 / average_partition_weight;
        (average_partition_weight, max_partition_weight, min_partition_weight, max_deviation)
    }

    /// Load balance info for processes
    pub fn load_balance_info_process(&self) -> (f64, u64, u64, f64) {
        let mut average_process_weight = 0.0;
        let mut max_process_weight = 0;
        let mut min_process_weight = u64::MAX;

        for i in 0..self.n_processes {
            average_process_weight += self.process_weight[i] as f64;
            if self.process_weight[i] > max_process_weight {
                max_process_weight = self.process_weight[i];
            }
            if self.process_weight[i] < min_process_weight {
                min_process_weight = self.process_weight[i];
            }
        }
        average_process_weight /= self.n_processes as f64;
        let max_deviation = max_process_weight as f64 / average_process_weight;
        (average_process_weight, max_process_weight, min_process_weight, max_deviation)
    }
}
