use std::sync::{Arc, RwLock};

use rustc_hash::FxHashMap;

use fields::PrimeField64;

use proofman_util::{timer_start_info, timer_stop_and_log_info};
use witness::WitnessComponent;
use proofman_common::{
    skip_prover_instance, BufferPool, DebugInfo, ModeName, ProofCtx, ProofmanError, ProofmanResult, SetupCtx,
};
use proofman_hints::{acc_mul_hint_fields, get_hint_ids_by_name, mul_hint_fields, update_airgroupvalue, HintFieldOptions};

use crate::{
    get_global_hint_field_constant_a_as, get_global_hint_field_constant_as, DebugData, DebugDataInfo, DebugDataFast,
    DebugDataFastGlobal, STD_MODE_DEFAULT, STD_MODE_ONE_INSTANCE, extract_hint_fields, print_std_debug_info,
    parse_debug_values_to_hashes,
};

pub struct StdSum<F: PrimeField64> {
    num_users: usize,
    std_mode: Vec<usize>,
    airgroup_ids: Vec<usize>,
    air_ids: Vec<usize>,
    debug_data: RwLock<DebugData>,
    debug_data_info: RwLock<DebugDataInfo>,
    debug_data_fast: RwLock<DebugDataFast>,
    debug_data_fast_global: RwLock<DebugDataFastGlobal>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField64> StdSum<F> {
    pub fn new(sctx: &Arc<SetupCtx<F>>) -> ProofmanResult<Arc<Self>> {
        // Get the sum check global data related to its users
        let std_sum_users = get_hint_ids_by_name(sctx.get_global_bin(), "std_sum_users");

        let Some(&std_sum_users) = std_sum_users.first() else {
            return Ok(Arc::new(Self {
                num_users: 0,
                std_mode: Vec::new(),
                airgroup_ids: Vec::new(),
                air_ids: Vec::new(),
                debug_data: RwLock::new(FxHashMap::default()),
                debug_data_info: RwLock::new(FxHashMap::default()),
                debug_data_fast: RwLock::new(DebugDataFast::new()),
                debug_data_fast_global: RwLock::new(DebugDataFastGlobal::new()),
                _phantom: std::marker::PhantomData,
            }));
        };

        let num_users = get_global_hint_field_constant_as::<usize, F>(sctx, std_sum_users, "num_users")?;
        let std_mode = get_global_hint_field_constant_a_as::<usize, F>(sctx, std_sum_users, "std_mode")?;
        let airgroup_ids = get_global_hint_field_constant_a_as::<usize, F>(sctx, std_sum_users, "airgroup_ids")?;
        let air_ids = get_global_hint_field_constant_a_as::<usize, F>(sctx, std_sum_users, "air_ids")?;

        Ok(Arc::new(Self {
            num_users,
            std_mode,
            airgroup_ids,
            air_ids,
            debug_data: RwLock::new(FxHashMap::default()),
            debug_data_info: RwLock::new(FxHashMap::default()),
            debug_data_fast: RwLock::new(DebugDataFast::new()),
            debug_data_fast_global: RwLock::new(DebugDataFastGlobal::new()),
            _phantom: std::marker::PhantomData,
        }))
    }
}

impl<F: PrimeField64> WitnessComponent<F> for StdSum<F> {
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
            // Process each sum check user
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

                    let setup = sctx.get_setup(airgroup_id, air_id)?;
                    let p_expressions_bin = setup.p_setup.p_expressions_bin;

                    let im_hints = get_hint_ids_by_name(p_expressions_bin, "im_col");
                    let im_airval_hints = get_hint_ids_by_name(p_expressions_bin, "im_airval");
                    let im_total_hints: Vec<u64> = im_hints.iter().chain(im_airval_hints.iter()).cloned().collect();

                    let n_im_total_hints = im_total_hints.len();
                    if !im_total_hints.is_empty() {
                        mul_hint_fields(
                            &sctx,
                            &pctx,
                            *instance_id,
                            n_im_total_hints as u64,
                            im_total_hints,
                            vec!["reference"; n_im_total_hints],
                            vec!["numerator"; n_im_total_hints],
                            vec![HintFieldOptions::default(); n_im_total_hints],
                            vec!["denominator"; n_im_total_hints],
                            vec![HintFieldOptions::inverse(); n_im_total_hints],
                        )?;
                    }

                    // We know that exactly one gsum hint must exist
                    let air_name = &pctx.global_info.airs[airgroup_id][air_id].name;
                    let gsum_hints = get_hint_ids_by_name(p_expressions_bin, "gsum_col");

                    let gsum_hint = match gsum_hints.as_slice() {
                        [] => {
                            return Err(ProofmanError::StdError(format!(
                                "No 'gsum_col' hint found for air: {}",
                                air_name
                            )))
                        }
                        [single] => *single as usize,
                        _ => {
                            return Err(ProofmanError::StdError(format!(
                                "Multiple gsum hints found for AIR '{air_name}'"
                            )))
                        }
                    };

                    let std_mode = self.std_mode[i];
                    let result = match std_mode {
                        STD_MODE_DEFAULT => Some("result"),
                        STD_MODE_ONE_INSTANCE => None,
                        _ => {
                            return Err(ProofmanError::StdError(format!(
                                "Unknown std_mode {std_mode} for AIR '{air_name}'"
                            )))
                        }
                    };
                    // This call accumulates "expression" into "reference" expression and stores its last value to "result"
                    // Alternatively, this could be done using get_hint_field and set_hint_field methods and doing the accumulation in Rust
                    acc_mul_hint_fields(
                        &sctx,
                        &pctx,
                        *instance_id,
                        gsum_hint,
                        "reference",
                        result,
                        "numerator_air",
                        "denominator_air",
                        HintFieldOptions::default(),
                        HintFieldOptions::inverse(),
                        true,
                    )?;

                    update_airgroupvalue(
                        &sctx,
                        &pctx,
                        *instance_id,
                        gsum_hint,
                        result,
                        "numerator_direct",
                        "denominator_direct",
                        HintFieldOptions::default(),
                        HintFieldOptions::inverse(),
                        true,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn debug(&self, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) -> ProofmanResult<()> {
        timer_start_info!(DEBUG_MODE_SUM);
        if self.num_users > 0 {
            let debug_hashes = parse_debug_values_to_hashes::<F>(&pctx)?;
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

                    extract_hint_fields(
                        &pctx,
                        &sctx,
                        *global_instance_id,
                        &mut self.debug_data.write().unwrap(),
                        &mut self.debug_data_info.write().unwrap(),
                        &self.debug_data_fast,
                        &self.debug_data_fast_global,
                        true,
                        false,
                        &debug_hashes,
                    )?;
                }
            } else {
                for global_instance_id in global_instance_ids {
                    extract_hint_fields(
                        &pctx,
                        &sctx,
                        *global_instance_id,
                        &mut self.debug_data.write().unwrap(),
                        &mut self.debug_data_info.write().unwrap(),
                        &self.debug_data_fast,
                        &self.debug_data_fast_global,
                        false,
                        false,
                        &debug_hashes,
                    )?;
                }
            }
        }
        timer_stop_and_log_info!(DEBUG_MODE_SUM);

        Ok(())
    }

    fn end(&self, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>, debug_info: &DebugInfo) -> ProofmanResult<()> {
        if debug_info.std_mode.name == ModeName::Debug {
            let debug_hashes = parse_debug_values_to_hashes::<F>(&pctx)?;
            print_std_debug_info(
                &pctx,
                &sctx,
                &self.debug_data,
                &self.debug_data_info,
                &self.debug_data_fast,
                &self.debug_data_fast_global,
                debug_info,
                false,
                &debug_hashes,
            )?;
        }
        Ok(())
    }
}
