#[cfg(feature = "mpi")]
use mpi::traits::*;
#[cfg(feature = "mpi")]
use mpi::collective::CommunicatorCollectives;
#[cfg(feature = "mpi")]
use mpi::datatype::PartitionMut;
#[cfg(feature = "mpi")]
use mpi::environment::Universe;
#[cfg(feature = "mpi")]
use mpi::topology::Communicator;

use std::sync::atomic::{Ordering, AtomicU64, AtomicI32, AtomicU32};
use proofman_fields::PrimeField64;
#[cfg(feature = "mpi")]
use proofman_fields::CubicExtensionField;
use crate::{GlobalInfo, ProofmanError};
use crate::Proof;

use crate::ProofmanResult;

#[cfg(feature = "mpi")]
use proofman_starks_lib_c::{
    initialize_agg_readiness_tracker_c, free_agg_readiness_tracker_c, agg_is_ready_c, reset_agg_readiness_tracker_c,
};

#[derive(Clone, Debug)]
pub struct RankInfo {
    pub world_rank: i32,
    pub local_rank: i32,
    pub n_processes: i32,
}

/// Detect the NUMA node of the current process based on CPU affinity
/// Assumes processes already pinned to specific CPUs
#[cfg(target_os = "linux")]
fn get_process_numa_node() -> i32 {
    let mut cpu: libc::c_uint = 0;
    let mut node: libc::c_uint = 0;

    // getcpu syscall returns both CPU and NUMA node directly
    let ret = unsafe {
        libc::syscall(
            libc::SYS_getcpu,
            &mut cpu as *mut libc::c_uint,
            &mut node as *mut libc::c_uint,
            std::ptr::null_mut::<libc::c_void>(),
        )
    };

    if ret == 0 {
        node as i32
    } else {
        -1
    }
}

#[cfg(not(target_os = "linux"))]
fn get_process_numa_node() -> i32 {
    -1 // Non-Linux: unknown NUMA node
}

pub struct MpiCtx {
    #[cfg(feature = "mpi")]
    pub universe: Universe,
    #[cfg(feature = "mpi")]
    pub world: mpi::topology::SimpleCommunicator,
    pub rank: i32,
    pub n_processes: i32,
    pub node_rank: i32,
    pub node_n_processes: i32,
    pub numa_nodes: Vec<i32>, // NUMA node for each process (indexed by node_rank)
    pub outer_agg_rank: AtomicI32,
    pub cancelled: AtomicU32,
}

const _MPI_TAG_CANCEL_JOB: i32 = 999999;
const _MPI_TAG_BROADCAST: i32 = 999998;
const _MPI_TAG_DISTRIBUTE_MULTIPLICITY: i32 = 999997;
const _MPI_TAG_DISTRIBUTE_MULTIPLICITIES: i32 = 999996;

impl Default for MpiCtx {
    fn default() -> Self {
        MpiCtx::new()
    }
}

impl MpiCtx {
    pub fn new() -> Self {
        #[cfg(feature = "mpi")]
        {
            let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Multiple)
                .expect("Failed to initialize MPI with threading");
            let world = universe.world();
            let rank = world.rank();
            let n_processes = world.size();
            let local_comm = world.split_shared(rank);
            let node_rank = local_comm.rank();
            let node_n_processes = local_comm.size();

            // Detect NUMA node for this process
            let numa_node = get_process_numa_node();

            // Gather NUMA nodes from all processes on the same physical node
            let mut numa_nodes = vec![0i32; node_n_processes as usize];
            local_comm.all_gather_into(&numa_node, &mut numa_nodes[..]);

            // Initialize the agg readiness tracker in the C library
            initialize_agg_readiness_tracker_c();

            MpiCtx {
                rank,
                n_processes,
                universe,
                world,
                node_rank,
                node_n_processes,
                numa_nodes,
                outer_agg_rank: AtomicI32::new(-1),
                cancelled: AtomicU32::new(0),
            }
        }
        #[cfg(not(feature = "mpi"))]
        {
            let numa_node = get_process_numa_node();
            MpiCtx {
                rank: 0,
                n_processes: 1,
                node_rank: 0,
                node_n_processes: 1,
                numa_nodes: vec![numa_node],
                outer_agg_rank: AtomicI32::new(0),
                cancelled: AtomicU32::new(0),
            }
        }
    }

    /// add an instance and assign it to a partition/process based only in the gid
    /// the instance added is not a table
    #[inline]
    pub fn process_ready_for_outer_agg(&self) {
        #[cfg(feature = "mpi")]
        {
            self.outer_agg_rank.store(agg_is_ready_c(), Ordering::SeqCst);
        }
    }

    pub fn get_outer_agg_rank(&self) -> ProofmanResult<i32> {
        if self.outer_agg_rank.load(Ordering::SeqCst) == -1 {
            return Err(ProofmanError::InvalidAssignation(
                "Aggregation rank not yet determined. Call process_ready_for_aggregation() first.".into(),
            ));
        }
        Ok(self.outer_agg_rank.load(Ordering::SeqCst))
    }

    pub fn reset_outer_agg_tracker(&self) {
        #[cfg(feature = "mpi")]
        {
            self.outer_agg_rank.store(-1, Ordering::SeqCst);
            reset_agg_readiness_tracker_c();
        }
    }

    pub fn reset(&self) {
        self.reset_outer_agg_tracker();
        self.cancelled.store(0, Ordering::SeqCst);
        #[cfg(feature = "mpi")]
        {
            while self.world.any_process().immediate_probe_with_tag(_MPI_TAG_CANCEL_JOB).is_some() {
                let (_msg, _status) = self.world.any_process().receive_vec_with_tag::<i32>(_MPI_TAG_CANCEL_JOB);
            }
        }
    }

    #[cfg(feature = "mpi")]
    pub fn new_with_universe(universe: Universe) -> Self {
        let world = universe.world();
        let rank = world.rank();
        let n_processes = world.size();
        let local_comm = world.split_shared(rank);
        let node_rank = local_comm.rank();
        let node_n_processes = local_comm.size();

        // Detect NUMA node for this process
        let numa_node = get_process_numa_node();

        // Gather NUMA nodes from all processes on the same physical node
        let mut numa_nodes = vec![0i32; node_n_processes as usize];
        local_comm.all_gather_into(&numa_node, &mut numa_nodes[..]);

        MpiCtx {
            rank,
            n_processes,
            universe,
            world,
            node_rank,
            node_n_processes,
            numa_nodes,
            outer_agg_rank: AtomicI32::new(-1),
            cancelled: AtomicU32::new(0),
        }
    }

    /// Get the NUMA node for this process
    #[inline]
    pub fn numa_node(&self) -> i32 {
        self.numa_nodes[self.node_rank as usize]
    }

    pub fn split_active_processes(&self, _is_active: bool) {
        #[cfg(feature = "mpi")]
        {
            let color =
                if _is_active { mpi::topology::Color::with_value(1) } else { mpi::topology::Color::undefined() };
            let _sub_comm = self.world.split_by_color(color);
            self.world.split_shared(self.rank);
        }
    }

    #[inline]
    pub fn barrier(&self) {
        #[cfg(feature = "mpi")]
        {
            self.world.barrier();
        }
    }

    /// Barrier that also checks all ranks succeeded.
    /// Every rank must call this — pass `false` on failure so others detect it.
    /// Returns `true` only if all ranks passed `true`.
    pub fn all_finished_ok(&self, success: bool) -> bool {
        #[cfg(feature = "mpi")]
        {
            if self.n_processes <= 1 {
                return success;
            }
            let send: [u8; 1] = [if success { 1 } else { 0 }];
            let mut all: Vec<u8> = vec![0u8; self.n_processes as usize];
            self.world.all_gather_into(&send[..], &mut all[..]);
            all.iter().all(|&v| v != 0)
        }
        #[cfg(not(feature = "mpi"))]
        {
            success
        }
    }

    pub fn distribute_roots(&self, values: Vec<u64>) -> Vec<u64> {
        #[cfg(feature = "mpi")]
        {
            let mut all_values: Vec<u64> = vec![0u64; values.len() * self.n_processes as usize];
            self.world.all_gather_into(&values, &mut all_values);
            all_values
        }
        #[cfg(not(feature = "mpi"))]
        {
            values.to_vec()
        }
    }

    pub fn distribute_airgroupvalues<F: PrimeField64>(
        &self,
        airgroupvalues: Vec<Vec<u64>>,
        _global_info: &GlobalInfo,
    ) -> Vec<Vec<F>> {
        #[cfg(feature = "mpi")]
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
                            let mut acc = CubicExtensionField {
                                value: [
                                    airgroupvalues_full[airgroup_id][idx * FIELD_EXTENSION],
                                    airgroupvalues_full[airgroup_id][idx * FIELD_EXTENSION + 1],
                                    airgroupvalues_full[airgroup_id][idx * FIELD_EXTENSION + 2],
                                ],
                            };
                            let val = CubicExtensionField {
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
        #[cfg(not(feature = "mpi"))]
        {
            airgroupvalues
                .into_iter()
                .map(|inner_vec| inner_vec.into_iter().map(|x| F::from_u64(x)).collect::<Vec<F>>())
                .collect()
        }
    }

    pub fn distribute_publics(&self, publics: Vec<u64>) -> Vec<u64> {
        #[cfg(feature = "mpi")]
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
        #[cfg(not(feature = "mpi"))]
        {
            publics
        }
    }

    //rank 0 broadcasts to the rest of processes a msg of unknown size
    //Root provides data in buf; others can pass an empty Vec that is filled with the message
    //Uses point-to-point with matched_probe for thread safety (no message interleaving)
    pub fn broadcast(&self, _buf: &mut Vec<u8>) {
        #[cfg(feature = "mpi")]
        {
            if self.n_processes > 1 {
                if self.rank == 0 {
                    // Root sends to all other processes
                    for dest in 1..self.n_processes {
                        self.world.process_at_rank(dest).send_with_tag(&_buf[..], _MPI_TAG_BROADCAST);
                    }
                } else {
                    // Non-root: matched_probe_with_tag + matched_receive_into for thread-safe receive
                    // This atomically binds to a specific message, preventing interleaving
                    // Using a dedicated tag prevents capturing unrelated internal messages
                    let (msg, status) = self.world.process_at_rank(0).matched_probe_with_tag(_MPI_TAG_BROADCAST);
                    let count = status.count(u8::equivalent_datatype()) as usize;
                    _buf.resize(count, 0u8);
                    msg.matched_receive_into(&mut _buf[..]);
                }
            }
        }
    }

    pub fn send_proof_to_rank(&self, _proof: &Vec<u64>, _airgroup_id: usize, _rank: i32) {
        #[cfg(feature = "mpi")]
        // Send the proof directly - the vector already contains its length information
        self.world.process_at_rank(_rank).send_with_tag(_proof, _airgroup_id as i32);
    }

    pub fn recv_proof_from_rank(&self, _airgroup_id: usize, _rank: i32) -> Vec<u64> {
        #[cfg(feature = "mpi")]
        {
            // Receive the proof directly as a vector
            let (proof_buffer, _) = self.world.process_at_rank(_rank).receive_vec_with_tag::<u64>(_airgroup_id as i32);
            proof_buffer
        }
        #[cfg(not(feature = "mpi"))]
        {
            Vec::new()
        }
    }

    pub fn send_proof_agg_rank<F: PrimeField64>(&self, _proof: &Proof<F>) {
        #[cfg(feature = "mpi")]
        self.world
            .process_at_rank(self.outer_agg_rank.load(Ordering::SeqCst))
            .send_with_tag(&_proof.proof[..], _proof.airgroup_id as i32);
    }

    pub fn check_incoming_proofs(&self, airgroup_id: usize) -> Option<Vec<u64>> {
        #[cfg(feature = "mpi")]
        {
            if let Some(_status) = self.world.any_process().immediate_probe_with_tag(airgroup_id as i32) {
                let (proof_data, _status) = self.world.any_process().receive_vec_with_tag::<u64>(airgroup_id as i32);
                Some(proof_data)
            } else {
                None
            }
        }
        #[cfg(not(feature = "mpi"))]
        {
            _ = airgroup_id;
            None
        }
    }

    /// Fail loudly if the ranks disagree on the aggregation arity.
    ///
    /// Tags are arity-derived, so disagreeing ranks would send to tags nobody receives on and
    /// hang silently. The all-gather is symmetric: every rank sees the mismatch and panics.
    #[cfg(feature = "mpi")]
    fn check_aggregation_arity_agreement(&self, arity: usize) {
        if self.n_processes <= 1 {
            return;
        }
        let send: [u64; 1] = [arity as u64];
        let mut all: Vec<u64> = vec![0u64; self.n_processes as usize];
        self.world.all_gather_into(&send[..], &mut all[..]);
        if let Some((other_rank, &other)) = all.iter().enumerate().find(|&(_, &v)| v != arity as u64) {
            panic!(
                "aggregation arity disagreement across MPI ranks: rank {} has {}, rank {} has {}. \
                 Every rank must load a proving key built with the same aggregationArity.",
                self.rank, arity, other_rank, other
            );
        }
    }

    #[allow(unused_variables)]
    pub fn distribute_recursive2_proofs(&self, alives: &[usize], proofs: &mut [Vec<Option<Vec<u64>>>], arity: usize) {
        #[cfg(feature = "mpi")]
        {
            // Must run before any tagged send/receive below: the tags are arity-derived.
            self.check_aggregation_arity_agreement(arity);

            // Count number of aggregations that will be done
            let n_groups = alives.len();
            let n_agregations: usize = alives.iter().map(|&alive| alive.div_ceil(arity)).sum();
            let aggs_per_process = (n_agregations / self.n_processes as usize).max(1);

            let mut i_proof = 0;
            // tags codes:
            // 0,...,ngroups-1: proofs sent to rank0 from another rank for a group with alive == 1
            // ngroups, ..., ngroups + arity*n_aggregations - 1: proofs sent to the owner of the
            // aggregation task. Every rank must derive `arity` from the same proving key, or
            // senders and receivers use different tags and the exchange deadlocks.

            for (group_idx, &alive) in alives.iter().enumerate() {
                let group_proofs: &mut Vec<Option<Vec<u64>>> = &mut proofs[group_idx];
                let n_aggs_group = alive.div_ceil(arity);

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

                    if owner_rank == self.rank as usize {
                        for k in 0..arity {
                            let idx = i * arity + k;
                            if idx < alive && group_proofs[idx].is_none() {
                                let tag = (i_proof * arity + n_groups + k) as i32;
                                let (msg, _status) = self.world.any_process().receive_vec_with_tag::<u64>(tag);
                                group_proofs[idx] = Some(msg);
                            }
                        }
                    } else if self.n_processes > 1 {
                        for k in 0..arity {
                            let idx = i * arity + k;
                            if idx < alive {
                                if let Some(proof) = group_proofs[idx].take() {
                                    let tag = (i_proof * arity + n_groups + k) as i32;
                                    self.world.process_at_rank(owner_rank as i32).send_with_tag(&proof[..], tag);
                                }
                            }
                        }
                    }
                    i_proof += 1;
                }
            }
        }
    }

    pub fn distribute_multiplicity(&self, _multiplicity: &[AtomicU64], _owner: i32) {
        #[cfg(feature = "mpi")]
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
                self.world
                    .process_at_rank(_owner)
                    .send_with_tag(&packed_multiplicity[..], _MPI_TAG_DISTRIBUTE_MULTIPLICITY);
            } else {
                let mut packed_multiplicity: Vec<u32> = vec![0; _multiplicity.len() * 2 + 1];
                for i in 0..self.n_processes {
                    if i != _owner {
                        let (msg, _) =
                            self.world.process_at_rank(i).matched_probe_with_tag(_MPI_TAG_DISTRIBUTE_MULTIPLICITY);
                        msg.matched_receive_into(&mut packed_multiplicity);
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

    pub fn distribute_multiplicities(
        &self,
        _multiplicities: &[AtomicU64],
        _n_cols: usize,
        _col_len: usize,
        _owner: i32,
    ) {
        #[cfg(feature = "mpi")]
        {
            if self.n_processes <= 1 {
                return;
            }

            assert!(_col_len < u32::MAX as usize);
            assert_eq!(_multiplicities.len(), _n_cols * _col_len);
            let buff_size = _n_cols * (_col_len + 1);

            if _owner != self.rank {
                // Pack multiplicities in a sparse vector
                let mut packed_multiplicities = vec![0u32; _n_cols];
                for (col_idx, column) in _multiplicities.chunks(_col_len).enumerate() {
                    for (idx, mul) in column.iter().enumerate() {
                        let m = mul.load(Ordering::Relaxed);
                        if m != 0 {
                            assert!(m < u32::MAX as u64);
                            packed_multiplicities[col_idx] += 1;
                            packed_multiplicities.push(idx as u32);
                            packed_multiplicities.push(m as u32);
                        }
                    }
                }

                self.world
                    .process_at_rank(_owner)
                    .send_with_tag(&packed_multiplicities[..], _MPI_TAG_DISTRIBUTE_MULTIPLICITIES);
            } else {
                let mut packed_multiplicities: Vec<u32> = vec![0; buff_size * 2];
                for i in 0..self.n_processes {
                    if i != _owner {
                        let (msg, _) =
                            self.world.process_at_rank(i).matched_probe_with_tag(_MPI_TAG_DISTRIBUTE_MULTIPLICITIES);
                        msg.matched_receive_into(&mut packed_multiplicities);

                        // Read counters
                        let mut counters = vec![0usize; _n_cols];
                        for col_idx in 0.._n_cols {
                            counters[col_idx] = packed_multiplicities[col_idx] as usize;
                        }

                        // Unpack multiplicities
                        let mut idx = _n_cols;
                        for (col_idx, &count) in counters.iter().enumerate() {
                            let col_base = col_idx * _col_len;
                            for _ in 0..count {
                                let row_idx = packed_multiplicities[idx] as usize;
                                let m = packed_multiplicities[idx + 1] as u64;
                                _multiplicities[col_base + row_idx].fetch_add(m, Ordering::Relaxed);
                                idx += 2;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Notify all other MPI processes to cancel their current job
    /// This sends a cancellation message to all ranks except the current one
    pub fn notify_cancellation(&self) {
        #[cfg(feature = "mpi")]
        {
            if self.cancelled.load(Ordering::SeqCst) == 1 {
                // Already cancelled, no need to send again
                return;
            }
            self.cancelled.store(1, Ordering::SeqCst);
            if self.n_processes > 1 {
                // Include the sender’s rank in the cancel message
                let cancel_msg: [i32; 1] = [self.rank];
                for rank in 0..self.n_processes {
                    if rank != self.rank {
                        self.world.process_at_rank(rank).send_with_tag(&cancel_msg, _MPI_TAG_CANCEL_JOB);
                    }
                }
            }
        }
    }

    /// Check for incoming cancellation messages
    pub fn check_cancellation(&self) -> Option<ProofmanError> {
        #[cfg(feature = "mpi")]
        {
            if self.cancelled.load(Ordering::SeqCst) == 0 {
                if let Some(_status) = self.world.any_process().immediate_probe_with_tag(_MPI_TAG_CANCEL_JOB) {
                    let (msg, _) = self.world.any_process().receive_vec_with_tag::<i32>(_MPI_TAG_CANCEL_JOB);

                    if let Some(&failed_rank) = msg.first() {
                        self.cancelled.store(1, Ordering::SeqCst);
                        return Some(ProofmanError::MpiCancellation(format!(
                            "Process {} received cancellation message from failed rank {}.",
                            self.rank, failed_rank
                        )));
                    }
                }
            }
        }
        None
    }
}

impl Drop for MpiCtx {
    fn drop(&mut self) {
        #[cfg(feature = "mpi")]
        {
            free_agg_readiness_tracker_c();
        }
    }
}

unsafe impl Send for MpiCtx {}
unsafe impl Sync for MpiCtx {}
