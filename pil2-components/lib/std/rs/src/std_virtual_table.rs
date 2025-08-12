use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    }
};
use proofman_util::create_buffer_fast;
use rayon::prelude::*;

use fields::PrimeField64;
use std::path::PathBuf;

use witness::WitnessComponent;
use proofman_common::{AirInstance, BufferPool, ProofCtx, SetupCtx, TraceInfo};
use proofman_hints::{get_hint_ids_by_name, HintFieldOptions};

use crate::{get_global_hint_field_constant_a_as, get_hint_field_constant_a_as, get_hint_field_constant_as};

pub struct StdVirtualTable<F: PrimeField64> {
    _phantom: std::marker::PhantomData<F>,
    pub global_id_map: Vec<(usize, usize, usize)>, // global_idx -> (air_idx, uid, uid_idx)
    pub virtual_table_airs: Option<Vec<Arc<VirtualTableAir>>>,
}
pub struct VirtualTableAir {
    airgroup_id: usize,
    air_id: usize,
    shift: u64,
    mask: u64,
    num_rows: usize,
    num_cols: usize,
    acc_heights: Vec<u64>,
    multiplicities: Vec<Vec<AtomicU64>>,
    instance_id: AtomicU64,
    calculated: AtomicBool,
}

impl<F: PrimeField64> StdVirtualTable<F> {
    pub fn new(pctx: Arc<ProofCtx<F>>, sctx: &SetupCtx<F>) -> Arc<Self> {
        // Get relevant data from the global hint
        let virtual_table_global_hint = get_hint_ids_by_name(sctx.get_global_bin(), "virtual_table_data_global");
        if virtual_table_global_hint.is_empty() {
            return Arc::new(Self { 
                _phantom: std::marker::PhantomData, 
                global_id_map: Vec::new(),
                virtual_table_airs: None,
            });
        }

        // Get the airgroup ids and air ids of the virtual tables
        let airgroup_ids = get_global_hint_field_constant_a_as::<usize, F>(sctx, virtual_table_global_hint[0], "airgroup_ids");
        let air_ids = get_global_hint_field_constant_a_as::<usize, F>(sctx, virtual_table_global_hint[0], "air_ids");

        let num_virtual_tables = airgroup_ids.len();
        let mut virtual_tables = Vec::with_capacity(num_virtual_tables);
        let mut global_id_map = Vec::new();
        let mut current_global_id = 0;
        for i in 0..num_virtual_tables {
            let airgroup_id = airgroup_ids[i];
            let air_id = air_ids[i];

            // Get the Virtual Table structure
            let setup = sctx.get_setup(airgroup_id, air_id);
            let hint_id = get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "virtual_table_data")[0] as usize;

            let hint_opt = HintFieldOptions::default();
            let uids =
                get_hint_field_constant_a_as::<usize, F>(sctx, airgroup_id, air_id, hint_id, "uids", hint_opt.clone());
            let acc_heights =
                get_hint_field_constant_a_as::<u64, F>(sctx, airgroup_id, air_id, hint_id, "acc_heights", hint_opt.clone());
            let num_groups =
                get_hint_field_constant_as::<usize, F>(sctx, airgroup_id, air_id, hint_id, "num_groups", hint_opt.clone());

            // Map each uid to an ordered set of indexes
            let num_uids = uids.len();
            for j in 0..num_uids {
                // Update global ID mapping: global_idx -> (air_idx, uid, uid_idx)
                global_id_map.insert(current_global_id, (i, uids[j], j));
                current_global_id += 1;
            }

            let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;
            let multiplicities = (0..num_groups as usize)
                .into_par_iter()
                .map(|_| (0..num_rows).into_par_iter().map(|_| AtomicU64::new(0)).collect())
                .collect();

            let virtual_table_air = VirtualTableAir {
                airgroup_id,
                air_id,
                shift: num_rows.trailing_zeros() as u64,
                mask: (num_rows - 1) as u64,
                num_rows,
                num_cols: num_groups as usize,
                acc_heights,
                multiplicities,
                instance_id: AtomicU64::new(0),
                calculated: AtomicBool::new(false),
            };
            virtual_tables.push(Arc::new(virtual_table_air));
        }

        Arc::new(Self { 
            _phantom: std::marker::PhantomData, 
            global_id_map, 
            virtual_table_airs: Some(virtual_tables),
        })
    }

    /// Returns the global ID for a given UID.
    pub fn get_global_id_from_uid(&self, uid: usize) -> usize {
        self.global_id_map.iter()
            .position(|&(_, u, _)| u == uid)
            .unwrap_or_else(|| panic!("UID {uid} not found in the global ID map"))
    }

    pub fn inc_virtual_row(&self, global_id: usize, row: u64, multiplicity: u64) {
        let (air_idx, _, uid_idx) = self.global_id_map[global_id];

        self.virtual_table_airs.as_ref().unwrap()[air_idx].inc_virtual_row(uid_idx, row, multiplicity);
    }

    pub fn inc_virtual_rows(&self, global_id: usize, rows: &[u64], multiplicities: &[u32]) {
        let (air_idx, _, uid_idx) = self.global_id_map[global_id];

        self.virtual_table_airs.as_ref().unwrap()[air_idx].inc_virtual_rows(uid_idx, rows, multiplicities);
    }

    pub fn inc_virtual_rows_same_mul(&self, global_id: usize, rows: &[u64], multiplicity: u64) {
        let (air_idx, _, uid_idx) = self.global_id_map[global_id];

        self.virtual_table_airs.as_ref().unwrap()[air_idx].inc_virtual_rows_same_mul(uid_idx, rows, multiplicity);
    }

    pub fn inc_virtual_rows_ranged(&self, global_id: usize, ranged_values: &[u64]) {
        let (air_idx, _, uid_idx) = self.global_id_map[global_id];

        self.virtual_table_airs.as_ref().unwrap()[air_idx].inc_virtual_rows_ranged(uid_idx, ranged_values);
    }
}

impl<F: PrimeField64> WitnessComponent<F> for StdVirtualTable<F> {}

impl VirtualTableAir {
    /// Processes a slice of input data and updates the multiplicity table.
    pub fn inc_virtual_row(&self, idx: usize, row: u64, multiplicity: u64) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        // Get the table offset
        let table_offset = self.acc_heights[idx];

        // Get the offset
        let offset = table_offset + row;

        // Map it to the appropriate multiplicity
        let sub_table_idx = offset >> self.shift;

        // Get the row index
        let row_idx = offset & self.mask;

        // Update the multiplicity
        self.multiplicities[sub_table_idx as usize][row_idx as usize].fetch_add(multiplicity, Ordering::Relaxed);
    }

    pub fn inc_virtual_rows(&self, idx: usize, rows: &[u64], multiplicities: &[u32]) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        // Get the table offset
        let table_offset = self.acc_heights[idx];

        for (&row, &multiplicity) in rows.iter().zip(multiplicities.iter()) {
            if multiplicity == 0 {
                continue;
            }

            // Get the offset
            let offset = table_offset + row;

            // Map it to the appropriate multiplicity
            let sub_table_idx = offset >> self.shift;

            // Get the row index
            let row_idx = offset & self.mask;

            // Update the multiplicity
            self.multiplicities[sub_table_idx as usize][row_idx as usize]
                .fetch_add(multiplicity as u64, Ordering::Relaxed);
        }
    }

    /// Processes a slice of input data and updates the multiplicity table.
    pub fn inc_virtual_rows_same_mul(&self, idx: usize, rows: &[u64], multiplicity: u64) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        // Get the table offset
        let table_offset = self.acc_heights[idx];

        for row in rows.iter() {
            // Get the offset
            let offset = table_offset + row;

            // Map it to the appropriate multiplicity
            let sub_table_idx = offset >> self.shift;

            // Get the row index
            let row_idx = offset & self.mask;

            // Update the multiplicity
            self.multiplicities[sub_table_idx as usize][row_idx as usize].fetch_add(multiplicity, Ordering::Relaxed);
        }
    }

    pub fn inc_virtual_rows_ranged(&self, idx: usize, ranged_values: &[u64]) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        // Get the table offset
        let table_offset = self.acc_heights[idx];

        for (row, &multiplicity) in ranged_values.iter().enumerate() {
            if multiplicity == 0 {
                continue;
            }

            // Get the offset
            let offset = table_offset + row as u64;

            // Map it to the appropriate multiplicity
            let sub_table_idx = offset >> self.shift;

            // Get the row index
            let row_idx = offset & self.mask;

            // Update the multiplicity
            self.multiplicities[sub_table_idx as usize][row_idx as usize]
                .fetch_add(multiplicity as u64, Ordering::Relaxed);
        }
    }
}

impl<F: PrimeField64> WitnessComponent<F> for VirtualTableAir {
    fn execute(&self, pctx: Arc<ProofCtx<F>>, _input_data_path: Option<PathBuf>) -> Vec<usize> {
        let (instance_found, mut instance_id) = pctx.dctx_find_instance(self.airgroup_id, self.air_id);

        if !instance_found {
            instance_id = pctx.add_instance_all(self.airgroup_id, self.air_id);
        }

        self.calculated.store(false, Ordering::Relaxed);
        self.instance_id.store(instance_id as u64, Ordering::SeqCst);

        Vec::new()
    }

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) {
        if stage == 1 {
            let instance_id = self.instance_id.load(Ordering::Relaxed) as usize;

            if !_instance_ids.contains(&instance_id) {
                return;
            }

            self.calculated.store(true, Ordering::Relaxed);

            pctx.dctx_distribute_multiplicities(&self.multiplicities, instance_id);

            if pctx.dctx_is_my_instance(instance_id) {
                let buffer_size = self.num_cols * self.num_rows;
                let mut buffer = create_buffer_fast(buffer_size);
                buffer.par_chunks_mut(self.num_cols).enumerate().for_each(|(row, chunk)| {
                    for (col, vec) in self.multiplicities.iter().enumerate() {
                        chunk[col] = F::from_u64(vec[row].swap(0, Ordering::Relaxed));
                    }
                });

                let air_instance = AirInstance::new(TraceInfo::new(self.airgroup_id, self.air_id, buffer, false));
                pctx.add_air_instance(air_instance, instance_id);
            } else {
                self.multiplicities.par_iter().for_each(|vec| {
                    for vec_row in vec.iter() {
                        vec_row.swap(0, Ordering::Relaxed);
                    }
                });
            }
        }
    }
}
