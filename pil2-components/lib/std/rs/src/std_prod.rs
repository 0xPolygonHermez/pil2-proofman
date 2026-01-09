use std::sync::{Arc, RwLock};

use fields::PrimeField64;

use rayon::prelude::*;

use rustc_hash::FxHashMap;

use proofman_util::{timer_start_info, timer_stop_and_log_info};
use witness::WitnessComponent;
use proofman_common::{
    skip_prover_instance, ProofmanError, BufferPool, DebugInfo, ModeName, ProofCtx, ProofmanResult, SetupCtx,
};
use proofman_hints::{
    acc_mul_hint_fields, get_hint_field, get_hint_field_a, get_hint_field_gc, get_hint_field_gc_a,
    get_hint_ids_by_name, mul_hint_fields, update_airgroupvalue, HintFieldOptions, HintFieldValue, HintFieldValuesVec,
    HintFieldOutput,
};

use crate::{
    check_invalid_opids, get_global_hint_field, get_global_hint_field_constant_a_as, get_global_hint_field_constant_as,
    get_hint_field_constant_as, get_hint_field_constant_as_field, get_row_field_value, print_debug_info,
    update_debug_data, update_debug_data_fast, DebugData, DebugDataFast, STD_MODE_DEFAULT, STD_MODE_ONE_INSTANCE,
    HintMetadata,
};

pub struct StdProd<F: PrimeField64> {
    num_users: usize,
    std_mode: Vec<usize>,
    airgroup_ids: Vec<usize>,
    air_ids: Vec<usize>,
    debug_data: RwLock<DebugData<F>>,
    debug_data_fast: Vec<RwLock<DebugDataFast<F>>>,
}

impl<F: PrimeField64> StdProd<F> {
    pub fn new(sctx: &Arc<SetupCtx<F>>) -> ProofmanResult<Arc<Self>> {
        // Get the product check global data related to its users
        let std_prod_users = get_hint_ids_by_name(sctx.get_global_bin(), "std_prod_users");

        let Some(&std_prod_users) = std_prod_users.first() else {
            return Ok(Arc::new(Self {
                num_users: 0,
                std_mode: Vec::new(),
                airgroup_ids: Vec::new(),
                air_ids: Vec::new(),
                debug_data: RwLock::new(FxHashMap::default()),
                debug_data_fast: (0..1000).map(|_| RwLock::new(FxHashMap::default())).collect(),
            }));
        };

        let num_users = get_global_hint_field_constant_as::<usize, F>(sctx, std_prod_users, "num_users")?;
        let std_mode = get_global_hint_field_constant_a_as::<usize, F>(sctx, std_prod_users, "std_mode")?;
        let airgroup_ids = get_global_hint_field_constant_a_as::<usize, F>(sctx, std_prod_users, "airgroup_ids")?;
        let air_ids = get_global_hint_field_constant_a_as::<usize, F>(sctx, std_prod_users, "air_ids")?;

        Ok(Arc::new(Self {
            num_users,
            std_mode,
            airgroup_ids,
            air_ids,
            debug_data: RwLock::new(FxHashMap::default()),
            debug_data_fast: (0..1000).map(|_| RwLock::new(FxHashMap::default())).collect(),
        }))
    }
}

impl<F: PrimeField64> WitnessComponent<F> for StdProd<F> {
    fn pre_calculate_witness(
        &self,
        _stage: u32,
        _pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        Ok(())
    }

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        if stage == 2 {
            let instances = pctx.dctx_get_instances();
            // Process each product check user
            for i in 0..self.num_users {
                let airgroup_id = self.airgroup_ids[i];
                let air_id = self.air_ids[i];

                for instance_id in instance_ids.iter() {
                    if instances[*instance_id].airgroup_id != airgroup_id
                        || instances[*instance_id].air_id != air_id
                        || skip_prover_instance(&pctx, *instance_id)?.0
                    {
                        continue;
                    }

                    // Get the air associated with the air_instance
                    let air_name = &pctx.global_info.airs[airgroup_id][air_id].name;

                    let setup = sctx.get_setup(airgroup_id, air_id)?;
                    let p_expressions_bin = setup.p_setup.p_expressions_bin;

                    let im_hints = get_hint_ids_by_name(p_expressions_bin, "im_col");
                    let gprod_hints = get_hint_ids_by_name(p_expressions_bin, "gprod_col");

                    let n_im_hints = im_hints.len();

                    if !im_hints.is_empty() {
                        mul_hint_fields(
                            &sctx,
                            &pctx,
                            *instance_id,
                            n_im_hints as u64,
                            im_hints,
                            vec!["reference"; n_im_hints],
                            vec!["numerator"; n_im_hints],
                            vec![HintFieldOptions::default(); n_im_hints],
                            vec!["denominator"; n_im_hints],
                            vec![HintFieldOptions::inverse(); n_im_hints],
                        )?;
                    }

                    // We know that at most one product hint exists
                    let gprod_hint = if gprod_hints.len() > 1 {
                        return Err(ProofmanError::StdError(format!(
                            "Multiple gprod hints found for AIR '{air_name}'"
                        )));
                    } else {
                        gprod_hints[0] as usize
                    };

                    let std_mode = self.std_mode[i];
                    let result = match std_mode {
                        STD_MODE_DEFAULT => Some("result"),
                        STD_MODE_ONE_INSTANCE => None,
                        _ => {
                            return Err(ProofmanError::StdError(format!(
                                "Unknown std_mode {std_mode} for AIR '{air_name}'"
                            )));
                        }
                    };
                    // This call calculates "numerator" / "denominator" and accumulates it into "reference". Its last value is stored into "result"
                    // Alternatively, this could be done using get_hint_field and set_hint_field methods and calculating the operations in Rust
                    acc_mul_hint_fields(
                        &sctx,
                        &pctx,
                        *instance_id,
                        gprod_hint,
                        "reference",
                        result,
                        "numerator_air",
                        "denominator_air",
                        HintFieldOptions::default(),
                        HintFieldOptions::inverse(),
                        false,
                    )?;

                    update_airgroupvalue(
                        &sctx,
                        &pctx,
                        *instance_id,
                        gprod_hint,
                        result,
                        "numerator_direct",
                        "denominator_direct",
                        HintFieldOptions::default(),
                        HintFieldOptions::inverse(),
                        false,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn debug(&self, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) -> ProofmanResult<()> {
        timer_start_info!(DEBUG_MODE_PROD);
        if self.num_users > 0 {
            // Find which instances is using the std_prod
            let instances = pctx.dctx_get_instances();
            let my_instances = pctx.dctx_get_process_instances();
            let mut global_instance_ids = Vec::new();
            for i in 0..self.num_users {
                let airgroup_id = self.airgroup_ids[i];
                let air_id = self.air_ids[i];

                // Get all air instances ids for this airgroup and air_id
                for instance_id in my_instances.iter() {
                    if instances[*instance_id].airgroup_id == airgroup_id
                        && instances[*instance_id].air_id == air_id
                        && instance_ids.contains(instance_id)
                        && !skip_prover_instance(&pctx, *instance_id)?.0
                    {
                        global_instance_ids.push(instance_id);
                    }
                }
            }

            let fast_mode = pctx.debug_info.read().unwrap().std_mode.fast_mode;
            if fast_mode {
                for &global_instance_id in &global_instance_ids {
                    if !instance_ids.contains(global_instance_id) {
                        continue;
                    }

                    self.extract_hint_fields(&pctx, &sctx, *global_instance_id, true)?;
                }
            } else {
                for global_instance_id in global_instance_ids {
                    self.extract_hint_fields(&pctx, &sctx, *global_instance_id, false)?;
                }
            }
        }
        timer_stop_and_log_info!(DEBUG_MODE_PROD);
        Ok(())
    }

    fn end(&self, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>, debug_info: &DebugInfo) -> ProofmanResult<()> {
        if debug_info.std_mode.name == ModeName::Debug || !debug_info.debug_instances.is_empty() {
            let fast_mode = debug_info.std_mode.fast_mode;

            // Perform the global hint update
            if fast_mode {
                let local_debug_data_fast = &mut self.debug_data_fast[0].write().unwrap();
                Self::extract_global_hint_fields(&pctx, &sctx, &mut FxHashMap::default(), local_debug_data_fast, true)?;
            } else {
                let mut debug_data = self.debug_data.write().unwrap();
                Self::extract_global_hint_fields(&pctx, &sctx, &mut debug_data, &mut FxHashMap::default(), false)?;
            }

            // At the end, check all the debug data
            if fast_mode {
                check_invalid_opids(&pctx, &self.debug_data_fast);
            } else {
                let mut debug_data = self.debug_data.write().unwrap();
                let max_values_to_print = debug_info.std_mode.n_vals;
                let print_to_file = debug_info.std_mode.print_to_file;
                print_debug_info(&pctx, &sctx, max_values_to_print, print_to_file, &mut debug_data)?;
            }
        }
        Ok(())
    }
}

impl<F: PrimeField64> StdProd<F> {
    const PROD_TYPE_ASSUMES: u64 = 0;
    const PROD_TYPE_PROVES: u64 = 1;
    const PROD_TYPE_FREE: u64 = 2;

    fn extract_global_hint_fields(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        debug_data: &mut DebugData<F>,
        debug_data_fast: &mut DebugDataFast<F>,
        fast_mode: bool,
    ) -> ProofmanResult<()> {
        let gprod_debug_data = get_hint_ids_by_name(sctx.get_global_bin(), "gprod_debug_data_global");
        if !gprod_debug_data.is_empty() {
            let num_global_hints =
                get_global_hint_field_constant_as::<usize, F>(sctx, gprod_debug_data[0], "num_global_hints")?;
            for i in 0..num_global_hints {
                let airgroup_id =
                    get_global_hint_field_constant_as::<usize, F>(sctx, gprod_debug_data[1 + i], "airgroup_id")?;
                let type_piop =
                    get_global_hint_field_constant_as::<u64, F>(sctx, gprod_debug_data[1 + i], "type_piop")?;
                if ![Self::PROD_TYPE_ASSUMES, Self::PROD_TYPE_PROVES, Self::PROD_TYPE_FREE].contains(&type_piop) {
                    return Err(ProofmanError::StdError(format!("Invalid type_piop: {type_piop}")));
                }

                let opid = get_global_hint_field(sctx, gprod_debug_data[1 + i], "busid")?;

                // If opids are specified, then only update the bus if the opid is in the list
                if !pctx.debug_info.read().unwrap().std_mode.opids.is_empty()
                    && !pctx.debug_info.read().unwrap().std_mode.opids.contains(&opid.as_canonical_u64())
                {
                    continue;
                }

                let num_reps = get_hint_field_gc(pctx, sctx, gprod_debug_data[1 + i], "num_reps", false)?;

                // If the number of repetitions is zero, continue
                let mut num_reps = get_row_field_value(&num_reps, 0, "num_reps")?;
                if num_reps.is_zero() {
                    continue;
                }

                // If the type is free and the num_reps is minus_one, simply flip the num_reps
                if type_piop == Self::PROD_TYPE_FREE {
                    if num_reps == F::NEG_ONE {
                        num_reps = -num_reps;
                    } else if num_reps != F::ONE {
                        return Err(ProofmanError::StdError(format!(
                            "The number of repetitions in a free piop can only be {{-1, 0, 1}}, received: {num_reps}"
                        )));
                    }
                }

                let expressions = get_hint_field_gc_a(pctx, sctx, gprod_debug_data[1 + i], "expressions", false)?;
                let is_proves = type_piop == Self::PROD_TYPE_PROVES;
                if fast_mode {
                    update_debug_data_fast(debug_data_fast, opid, expressions.get(0), is_proves, num_reps, true)?;
                } else {
                    update_debug_data(
                        debug_data,
                        i,
                        opid,
                        expressions.get(0),
                        airgroup_id,
                        None,
                        None,
                        0,
                        is_proves,
                        num_reps,
                        true,
                        true,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn extract_hint_fields(
        &self,
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        instance_id: usize,
        fast_mode: bool,
    ) -> ProofmanResult<()> {
        // Process the AIR debug hints
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;
        let air_instance_id = pctx.dctx_find_air_instance_id(instance_id)?;

        let setup = sctx.get_setup(airgroup_id, air_id)?;
        let p_expressions_bin = setup.p_setup.p_expressions_bin;

        let debug_data_hints = get_hint_ids_by_name(p_expressions_bin, "gprod_debug_data");

        let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;

        let hint_metadatas: Result<Vec<_>, ProofmanError> = debug_data_hints
            .iter()
            .enumerate()
            .map(|(i, &hint)| {
                let busid =
                    get_hint_field(sctx, pctx, instance_id, hint as usize, "busid", HintFieldOptions::default())?;

                let type_piop = get_hint_field_constant_as::<u64, F>(
                    sctx,
                    airgroup_id,
                    air_id,
                    hint as usize,
                    "type_piop",
                    HintFieldOptions::default(),
                )?;
                if ![Self::PROD_TYPE_ASSUMES, Self::PROD_TYPE_PROVES, Self::PROD_TYPE_FREE].contains(&type_piop) {
                    return Err(ProofmanError::StdError(format!("Invalid type_piop: {type_piop}")));
                }

                let num_reps =
                    get_hint_field(sctx, pctx, instance_id, hint as usize, "num_reps", HintFieldOptions::default())?;

                let deg_expr = get_hint_field_constant_as_field(
                    sctx,
                    airgroup_id,
                    air_id,
                    hint as usize,
                    "deg_expr",
                    HintFieldOptions::default(),
                )?;

                let deg_mul = get_hint_field_constant_as_field(
                    sctx,
                    airgroup_id,
                    air_id,
                    hint as usize,
                    "deg_sel",
                    HintFieldOptions::default(),
                )?;

                let expressions = get_hint_field_a(
                    sctx,
                    pctx,
                    instance_id,
                    hint as usize,
                    "expressions",
                    HintFieldOptions::default(),
                )?;

                Ok(HintMetadata { hint, hint_id: i, busid, type_piop, num_reps, expressions, deg_expr, deg_mul })
            })
            .collect();

        let hint_metadatas = hint_metadatas?;
        if fast_mode {
            let opids = pctx.debug_info.read().unwrap().std_mode.opids.clone();

            // Process hints in chunks of 1000 to reuse pre-allocated HashMaps
            for chunk in hint_metadatas.chunks(1000) {
                chunk.par_iter().enumerate().try_for_each(|(idx, hint_metadata)| -> ProofmanResult<()> {
                    // Directly acquire write lock and work with it
                    let mut debug_data_fast = self.debug_data_fast[idx].write().unwrap();

                    // If both the expression and the mul are of degree zero, then simply update the bus once
                    if hint_metadata.deg_expr.is_zero() && hint_metadata.deg_mul.is_zero() {
                        let opid = match hint_metadata.busid {
                            HintFieldValue::Field(opid) => {
                                if !opids.is_empty() && !opids.contains(&opid.as_canonical_u64()) {
                                    return Ok(());
                                }
                                opid
                            }
                            _ => return Err(ProofmanError::StdError("busid must be a field element".to_string())),
                        };

                        Self::update_bus_fast(
                            opid,
                            hint_metadata.type_piop,
                            &hint_metadata.num_reps,
                            &hint_metadata.expressions,
                            0,
                            &mut debug_data_fast,
                            false,
                        )?;
                    }
                    // Otherwise, update the bus for each row
                    else {
                        for j in 0..num_rows {
                            let opid = match hint_metadata.busid.get(j) {
                                HintFieldOutput::Field(opid) => {
                                    if !opids.is_empty() && !opids.contains(&opid.as_canonical_u64()) {
                                        continue;
                                    }
                                    opid
                                }
                                _ => return Err(ProofmanError::StdError("busid must be a field element".to_string())),
                            };

                            Self::update_bus_fast(
                                opid,
                                hint_metadata.type_piop,
                                &hint_metadata.num_reps,
                                &hint_metadata.expressions,
                                j,
                                &mut debug_data_fast,
                                false,
                            )?;
                        }
                    }

                    Ok(())
                })?;
            }
        } else {
            for hint_metadata in hint_metadatas.iter() {
                // If both the expresion and the mul are of degree zero, then simply update the bus once
                if hint_metadata.deg_expr.is_zero() && hint_metadata.deg_mul.is_zero() {
                    // In this case, the busid must be a field element
                    let opid = match hint_metadata.busid {
                        HintFieldValue::Field(opid) => {
                            // If opids are specified, then only update the bus if the opid is in the list
                            let opids = &pctx.debug_info.read().unwrap().std_mode.opids;
                            if !opids.is_empty() && !opids.contains(&opid.as_canonical_u64()) {
                                continue;
                            }
                            opid
                        }
                        _ => return Err(ProofmanError::StdError("busid must be a field element".to_string())),
                    };

                    Self::update_bus(
                        hint_metadata.hint_id,
                        airgroup_id,
                        air_id,
                        air_instance_id,
                        opid,
                        hint_metadata.type_piop,
                        &hint_metadata.num_reps,
                        &hint_metadata.expressions,
                        0,
                        &mut self.debug_data.write().unwrap(),
                        false,
                    )?;
                }
                // Otherwise, update the bus for each row
                else {
                    for j in 0..num_rows {
                        // Get the opid for this row
                        let opid = match hint_metadata.busid.get(j) {
                            HintFieldOutput::Field(opid) => {
                                // If opids are specified, then only update the bus if the opid is in the list
                                let opids = &pctx.debug_info.read().unwrap().std_mode.opids;
                                if !opids.is_empty() && !opids.contains(&opid.as_canonical_u64()) {
                                    continue;
                                }

                                opid
                            }
                            _ => return Err(ProofmanError::StdError("busid must be a field element".to_string())),
                        };

                        Self::update_bus(
                            hint_metadata.hint_id,
                            airgroup_id,
                            air_id,
                            air_instance_id,
                            opid,
                            hint_metadata.type_piop,
                            &hint_metadata.num_reps,
                            &hint_metadata.expressions,
                            j,
                            &mut self.debug_data.write().unwrap(),
                            false,
                        )?;
                    }
                }
            }
        }

        std::thread::spawn(move || {
            drop(hint_metadatas);
        });

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn update_bus(
        hint_id: usize,
        airgroup_id: usize,
        air_id: usize,
        air_instance_id: usize,
        opid: F,
        type_piop: u64,
        num_reps: &HintFieldValue<F>,
        expressions: &HintFieldValuesVec<F>,
        row: usize,
        debug_data: &mut DebugData<F>,
        is_global: bool,
    ) -> ProofmanResult<()> {
        let mut num_reps = get_row_field_value(num_reps, row, "num_reps")?;
        if num_reps.is_zero() {
            return Ok(());
        }

        let is_proves = match type_piop {
            Self::PROD_TYPE_ASSUMES => false,
            Self::PROD_TYPE_PROVES => true,
            Self::PROD_TYPE_FREE => {
                if num_reps == F::NEG_ONE {
                    // If the type is free and the num_reps is minus_one, simply flip the num_reps
                    num_reps = -num_reps;
                    false
                } else if num_reps == F::ONE {
                    true
                } else {
                    return Err(ProofmanError::StdError(format!(
                        "The number of repetitions in a free piop can only be {{-1, 0, 1}}, received: {num_reps}"
                    )));
                }
            }
            _ => unreachable!(),
        };

        update_debug_data(
            debug_data,
            hint_id,
            opid,
            expressions.get(row),
            airgroup_id,
            Some(air_id),
            Some(air_instance_id),
            row,
            is_proves,
            num_reps,
            is_global,
            true,
        )
    }

    fn update_bus_fast(
        opid: F,
        type_piop: u64,
        num_reps: &HintFieldValue<F>,
        expressions: &HintFieldValuesVec<F>,
        row: usize,
        debug_data_fast: &mut DebugDataFast<F>,
        is_global: bool,
    ) -> ProofmanResult<()> {
        let mut num_reps = get_row_field_value(num_reps, row, "num_reps")?;
        if num_reps.is_zero() {
            return Ok(());
        }

        let is_proves = match type_piop {
            Self::PROD_TYPE_ASSUMES => false,
            Self::PROD_TYPE_PROVES => true,
            Self::PROD_TYPE_FREE => {
                if num_reps == F::NEG_ONE {
                    // If the type is free and the num_reps is minus_one, simply flip the num_reps
                    num_reps = -num_reps;
                    false
                } else if num_reps == F::ONE {
                    true
                } else {
                    return Err(ProofmanError::StdError(format!(
                        "The number of repetitions in a free piop can only be {{-1, 0, 1}}, received: {num_reps}"
                    )));
                }
            }
            _ => unreachable!(),
        };

        update_debug_data_fast(debug_data_fast, opid, expressions.get(row), is_proves, num_reps, is_global)
    }
}
