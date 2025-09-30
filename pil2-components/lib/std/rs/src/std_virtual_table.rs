use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, RwLock,
    },
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
    pub global_id_by_uid: HashMap<usize, usize>,   // uid -> global_id
    pub indices_by_global_id: Vec<(usize, usize)>, // global_id -> (air_idx, uid_idx)
    pub virtual_table_airs: Option<Vec<Arc<VirtualTableAir>>>,
}
pub struct VirtualTableAir {
    airgroup_id: usize,
    air_id: usize,
    shift: u64,
    mask: u64,
    num_rows: usize,
    num_cols: usize,
    pub table_ids: Vec<(usize, u64)>, // (table_id, acc_height)
    multiplicities: Vec<Vec<AtomicU64>>,
    pub table_instance_id: AtomicU64,
    calculated: AtomicBool,
    shared_tables: bool,
    is_unique: bool,
}

impl<F: PrimeField64> StdVirtualTable<F> {
    pub fn new(
        pctx: Arc<ProofCtx<F>>,
        sctx: &SetupCtx<F>,
        shared_tables: bool,
        unique_worker_table_ids: Vec<usize>,
    ) -> Arc<Self> {
        // Get relevant data from the global hint
        let virtual_table_global_hint = get_hint_ids_by_name(sctx.get_global_bin(), "virtual_table_data_global");
        if virtual_table_global_hint.is_empty() {
            return Arc::new(Self {
                _phantom: std::marker::PhantomData,
                global_id_by_uid: HashMap::new(),
                indices_by_global_id: Vec::new(),
                virtual_table_airs: None,
            });
        }

        let airgroup_ids =
            get_global_hint_field_constant_a_as::<usize, F>(sctx, virtual_table_global_hint[0], "airgroup_ids");
        let air_ids = get_global_hint_field_constant_a_as::<usize, F>(sctx, virtual_table_global_hint[0], "air_ids");

        let num_virtual_tables = airgroup_ids.len();
        let mut virtual_tables = Vec::with_capacity(num_virtual_tables);
        let mut global_id_by_uid = HashMap::new();
        let mut indices_by_global_id = Vec::new();
        let mut current_global_id = 0;
        for i in 0..num_virtual_tables {
            let airgroup_id = airgroup_ids[i];
            let air_id = air_ids[i];

            // Get the Virtual Table structure
            let setup = sctx.get_setup(airgroup_id, air_id);
            let hint_id = get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "virtual_table_data")[0] as usize;

            let hint_opt = HintFieldOptions::default();
            let table_ids = get_hint_field_constant_a_as::<usize, F>(
                sctx,
                airgroup_id,
                air_id,
                hint_id,
                "table_ids",
                hint_opt.clone(),
            );
            let acc_heights = get_hint_field_constant_a_as::<u64, F>(
                sctx,
                airgroup_id,
                air_id,
                hint_id,
                "acc_heights",
                hint_opt.clone(),
            );
            let num_muls = get_hint_field_constant_as::<usize, F>(
                sctx,
                airgroup_id,
                air_id,
                hint_id,
                "num_muls",
                hint_opt.clone(),
            );

            // Map each table_id to an ordered set of indexes
            let num_table_ids = table_ids.len();
            let mut idxs = vec![(0, 0); num_table_ids];
            for j in 0..num_table_ids {
                idxs[j] = (table_ids[j], acc_heights[j]);

                // Update global ID mapping: global_idx -> (air_idx, uid, uid_idx)
                // global_id_map.insert(current_global_id, (i, table_ids[j], j));
                global_id_by_uid.insert(table_ids[j], current_global_id);
                indices_by_global_id.push((i, j));
                current_global_id += 1;
            }

            let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;
            let multiplicities = (0..num_muls as usize)
                .into_par_iter()
                .map(|_| (0..num_rows).into_par_iter().map(|_| AtomicU64::new(0)).collect())
                .collect();

            let is_unique = unique_worker_table_ids.iter().any(|id| table_ids.contains(id));

            let virtual_table_air = VirtualTableAir {
                airgroup_id,
                air_id,
                shift: num_rows.trailing_zeros() as u64,
                mask: (num_rows - 1) as u64,
                num_rows,
                num_cols: num_muls as usize,
                table_ids: idxs,
                multiplicities,
                table_instance_id: AtomicU64::new(0),
                calculated: AtomicBool::new(false),
                shared_tables,
                is_unique,
            };
            virtual_tables.push(Arc::new(virtual_table_air));
        }

        Arc::new(Self {
            _phantom: std::marker::PhantomData,
            global_id_by_uid,
            indices_by_global_id,
            virtual_table_airs: Some(virtual_tables),
        })
    }

    pub fn get_global_id(&self, id: usize) -> usize {
        self.global_id_by_uid.get(&id).copied().unwrap_or_else(|| panic!("ID {id} not found in the global ID map"))
    }

    pub fn inc_virtual_row(&self, global_id: usize, row: u64, multiplicity: u64) {
        let (air_idx, uid_idx) = self.indices_by_global_id[global_id];
        self.virtual_table_airs.as_ref().unwrap()[air_idx].inc_virtual_row(uid_idx, row, multiplicity);
    }

    pub fn inc_virtual_rows(&self, global_id: usize, rows: &[u64], multiplicities: &[u32]) {
        let (air_idx, uid_idx) = self.indices_by_global_id[global_id];
        self.virtual_table_airs.as_ref().unwrap()[air_idx].inc_virtual_rows(uid_idx, rows, multiplicities);
    }

    pub fn inc_virtual_rows_same_mul(&self, global_id: usize, rows: &[u64], multiplicity: u64) {
        let (air_idx, uid_idx) = self.indices_by_global_id[global_id];
        self.virtual_table_airs.as_ref().unwrap()[air_idx].inc_virtual_rows_same_mul(uid_idx, rows, multiplicity);
    }

    pub fn inc_virtual_rows_ranged(&self, global_id: usize, ranged_values: &[u64]) {
        let (air_idx, uid_idx) = self.indices_by_global_id[global_id];
        self.virtual_table_airs.as_ref().unwrap()[air_idx].inc_virtual_rows_ranged(uid_idx, ranged_values);
    }

    pub fn get_virtual_table_id_unique(&self, ids: &[usize]) -> Option<usize> {
        let mut instance_id = None;
        for vta in self.virtual_table_airs.as_ref().unwrap().iter() {
            if vta.is_unique == false {
                continue;
            }
            for id in ids.iter() {
                if vta.table_ids.iter().any(|&(table_id, _)| table_id == *id) {
                    if instance_id.is_none() {
                        instance_id = Some(*id);
                    } else {
                        // Check it is the same
                        if instance_id.unwrap() != *id {
                            panic!("Multiple unique table IDs found in the same Virtual Table Air");
                        }
                    }
                }
            }
        }
        instance_id
    }
}

impl<F: PrimeField64> WitnessComponent<F> for StdVirtualTable<F> {
    fn pre_calculate_witness(
        &self,
        _stage: u32,
        _pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) {
    }
}

impl VirtualTableAir {
    pub fn get_id(&self, id: usize) -> usize {
        if let Some(pos) = self.table_ids.iter().position(|&(table_id, _)| table_id == id) {
            pos
        } else {
            tracing::error!("ID {} not found in the virtual table", id);
            panic!();
        }
    }

    /// Processes a slice of input data and updates the multiplicity table.
    pub fn inc_virtual_row(&self, id: usize, row: u64, multiplicity: u64) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        // Get the table offset
        let table_offset = self.table_ids[id].1; // Acc height of the table

        // Get the offset
        let offset = table_offset + row;

        // Map it to the appropriate multiplicity
        let sub_table_idx = offset >> self.shift;

        // Get the row index
        let row_idx = offset & self.mask;

        // Update the multiplicity
        self.multiplicities[sub_table_idx as usize][row_idx as usize].fetch_add(multiplicity, Ordering::Relaxed);
    }

    pub fn inc_virtual_rows(&self, id: usize, rows: &[u64], multiplicities: &[u32]) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        // Get the table offset
        let table_offset = self.table_ids[id].1; // Acc height of the table

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
    pub fn inc_virtual_rows_same_mul(&self, id: usize, rows: &[u64], multiplicity: u64) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        // Get the table offset
        let table_offset = self.table_ids[id].1; // Acc height of the table

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

    pub fn inc_virtual_rows_ranged(&self, id: usize, ranged_values: &[u64]) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        // Get the table offset
        let table_offset = self.table_ids[id].1; // Acc height of the table

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
            self.multiplicities[sub_table_idx as usize][row_idx as usize].fetch_add(multiplicity, Ordering::Relaxed);
        }
    }
}

impl<F: PrimeField64> WitnessComponent<F> for VirtualTableAir {
    fn execute(&self, pctx: Arc<ProofCtx<F>>, _global_ids: &RwLock<Vec<usize>>, _input_data_path: Option<PathBuf>) {
        let (instance_found, mut table_instance_id) = pctx.dctx_find_process_table(self.airgroup_id, self.air_id);

        if !instance_found {
            if !self.shared_tables {
                table_instance_id = pctx.add_table_all(self.airgroup_id, self.air_id);
            } else if self.is_unique {
                table_instance_id = pctx.add_table_unique(self.airgroup_id, self.air_id);
            } else {
                table_instance_id = pctx.add_table(self.airgroup_id, self.air_id);
            }
        }

        self.calculated.store(false, Ordering::Relaxed);
        self.multiplicities.par_iter().for_each(|vec| {
            for v in vec.iter() {
                v.store(0, Ordering::Relaxed);
            }
        });

        self.table_instance_id.store(table_instance_id as u64, Ordering::SeqCst);
    }

    fn pre_calculate_witness(
        &self,
        _stage: u32,
        _pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) {
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
            let table_instance_id = self.table_instance_id.load(Ordering::Relaxed) as usize;

            let instance_id = match self.is_unique {
                true => table_instance_id,
                false => pctx.dctx_get_table_instance_idx(table_instance_id),
            };

            if !_instance_ids.contains(&instance_id) {
                return;
            }

            self.calculated.store(true, Ordering::Relaxed);

            if self.shared_tables {
                let owner_idx = pctx.dctx_get_process_owner_instance(instance_id);
                pctx.mpi_ctx.distribute_multiplicities(&self.multiplicities, owner_idx);
            }

            if !self.shared_tables || pctx.dctx_is_my_process_instance(instance_id) {
                let buffer_size = self.num_cols * self.num_rows;
                let mut buffer = create_buffer_fast(buffer_size);
                buffer.par_chunks_mut(self.num_cols).enumerate().for_each(|(row, chunk)| {
                    for (col, vec) in self.multiplicities.iter().enumerate() {
                        chunk[col] = F::from_u64(vec[row].load(Ordering::Relaxed));
                    }
                });
                let air_instance =
                    AirInstance::new(TraceInfo::new(self.airgroup_id, self.air_id, self.num_rows, buffer, false));
                pctx.add_air_instance(air_instance, instance_id);
            }
        }
    }
}
