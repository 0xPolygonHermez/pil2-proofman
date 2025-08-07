use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
};
use proofman_util::create_buffer_fast;
use rayon::prelude::*;

use fields::PrimeField64;
use std::path::PathBuf;

use witness::WitnessComponent;
use proofman_common::{AirInstance, BufferPool, ProofCtx, SetupCtx, TraceInfo};
use proofman_hints::{get_hint_ids_by_name, HintFieldOptions};

use crate::{get_global_hint_field_constant_as, get_hint_field_constant_a_as, get_hint_field_constant_as};

pub struct StdVirtualTable<F: PrimeField64> {
    _phantom: std::marker::PhantomData<F>,
    pub virtual_table_air: Option<Arc<VirtualTableAir>>,
}
pub struct VirtualTableAir {
    airgroup_id: usize,
    air_id: usize,
    shift: u64,
    mask: u64,
    num_rows: usize,
    num_cols: usize,
    uids: Vec<(usize, u64)>, // (uid, acc_height)
    multiplicities: Vec<Vec<AtomicU64>>,
    instance_id: AtomicU64,
    calculated: AtomicBool,
}

impl<F: PrimeField64> StdVirtualTable<F> {
    pub fn new(pctx: Arc<ProofCtx<F>>, sctx: &SetupCtx<F>) -> Arc<Self> {
        // Get relevant data from the global hint
        let virtual_table_global_hint = get_hint_ids_by_name(sctx.get_global_bin(), "virtual_table_data_global");
        if virtual_table_global_hint.is_empty() {
            return Arc::new(Self { _phantom: std::marker::PhantomData, virtual_table_air: None });
        }

        let airgroup_id = get_global_hint_field_constant_as(sctx, virtual_table_global_hint[0], "airgroup_id");
        let air_id = get_global_hint_field_constant_as(sctx, virtual_table_global_hint[0], "air_id");

        // Get the Virtual Table structure
        let setup = sctx.get_setup(airgroup_id, air_id);
        let hint_id = get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "virtual_table_data")[0] as usize;

        let hint_opt = HintFieldOptions::default();
        let uids =
            get_hint_field_constant_a_as::<usize, F>(sctx, airgroup_id, air_id, hint_id, "uids", hint_opt.clone());
        let acc_heights =
            get_hint_field_constant_a_as::<u64, F>(sctx, airgroup_id, air_id, hint_id, "acc_heights", hint_opt.clone());

        // Map each uid to an ordered set of indexes
        let num_uids = uids.len();
        let mut idxs = vec![(0, 0); num_uids];
        for i in 0..num_uids {
            idxs[i] = (uids[i], acc_heights[i]);
        }

        let hint_id = get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "virtual_table_data")[1] as usize;
        let num_groups =
            get_hint_field_constant_as::<usize, F>(sctx, airgroup_id, air_id, hint_id, "num_groups", hint_opt.clone());
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
            uids: idxs,
            multiplicities,
            instance_id: AtomicU64::new(0),
            calculated: AtomicBool::new(false),
        };

        Arc::new(Self { _phantom: std::marker::PhantomData, virtual_table_air: Some(Arc::new(virtual_table_air)) })
    }

    pub fn get_id(&self, id: usize) -> usize {
        self.virtual_table_air.as_ref().unwrap().get_id(id)
    }

    pub fn inc_virtual_row(&self, id: usize, row: u64, multiplicity: u64) {
        self.virtual_table_air.as_ref().unwrap().inc_virtual_row(id, row, multiplicity);
    }

    pub fn inc_virtual_rows(&self, id: usize, rows: Vec<u64>, multiplicities: Vec<u64>) {
        self.virtual_table_air.as_ref().unwrap().inc_virtual_rows(id, rows, multiplicities);
    }

    pub fn inc_virtual_rows_same_mul(&self, id: usize, rows: Vec<u64>, multiplicity: u64) {
        self.virtual_table_air.as_ref().unwrap().inc_virtual_rows_same_mul(id, rows, multiplicity);
    }
}

impl<F: PrimeField64> WitnessComponent<F> for StdVirtualTable<F> {}

impl VirtualTableAir {
    pub fn get_id(&self, id: usize) -> usize {
        if let Some(pos) = self.uids.iter().position(|&(uid, _)| uid == id) {
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
        let table_offset = self.uids[id].1; // Acc height of the table

        // Get the offset
        let offset = table_offset + row;

        // Map it to the appropriate multiplicity
        let sub_table_idx = offset >> self.shift;

        // Get the row index
        let row_idx = offset & self.mask;

        // Update the multiplicity
        self.multiplicities[sub_table_idx as usize][row_idx as usize].fetch_add(multiplicity, Ordering::Relaxed);
    }

    pub fn inc_virtual_rows(&self, id: usize, rows: Vec<u64>, multiplicities: Vec<u64>) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        // Get the table offset
        let table_offset = self.uids[id].1; // Acc height of the table

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
            self.multiplicities[sub_table_idx as usize][row_idx as usize].fetch_add(multiplicity, Ordering::Relaxed);
        }
    }

    /// Processes a slice of input data and updates the multiplicity table.
    pub fn inc_virtual_rows_same_mul(&self, id: usize, rows: Vec<u64>, multiplicity: u64) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        // Get the table offset
        let table_offset = self.uids[id].1; // Acc height of the table

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
