use std::collections::{HashMap, HashSet};
use std::ffi::c_void;

use proofman_fields::PrimeField64;
use proofman_starks_lib_c::{expressions_bin_new_c, expressions_bin_free_c};

use crate::format_bytes;
use crate::load_const_pols;
use crate::{GlobalInfo, GlobalInfoAir};
use crate::ProofmanError;
use crate::ProofmanResult;
use crate::Setup;
use crate::ProofType;

pub struct PreLoadedConstTree {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub proof_type: ProofType,
}

impl PreLoadedConstTree {
    pub fn new(airgroup_id: usize, air_id: usize, proof_type: ProofType) -> Self {
        PreLoadedConstTree { airgroup_id, air_id, proof_type }
    }
}

/// The slot an air's packed const pols occupy in the GPU const buffer. Airs with identical
/// fixed columns -- same verkey, i.e. same const-tree root -- share one slot; `owner` is the
/// first of the group and the only one that uploads. `load_tree` is a group property, so the
/// slot layout does not depend on which member is asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedGroup {
    pub owner: (usize, usize),
    pub load_tree: bool,
}

pub fn is_preload_fixed(
    airgroup_id: usize,
    air_id: usize,
    proof_type: &ProofType,
    preloaded_const: &[PreLoadedConstTree],
) -> bool {
    preloaded_const
        .iter()
        .any(|pc| pc.airgroup_id == airgroup_id && pc.air_id == air_id && pc.proof_type == *proof_type)
}

pub struct SetupsVadcop<F: PrimeField64> {
    pub sctx_compressor: Option<SetupCtx<F>>,
    pub sctx_recursive1: Option<SetupCtx<F>>,
    pub sctx_recursive2: Option<SetupCtx<F>>,
    pub setup_vadcop_final: Option<Setup<F>>,
    pub setup_vadcop_final_compressed: Option<Setup<F>>,
    pub max_witness_size: usize,
    pub max_witness_size_compressor: usize,
    pub max_trace_size: usize,
    pub max_trace_size_compressor: usize,
    pub max_const_size: usize,
    pub max_const_tree_size: usize,
    pub max_prover_trace_size: usize,
    pub max_prover_buffer_size: usize,
    pub max_prover_recursive_buffer_size: usize,
    pub max_prover_recursive2_buffer_size: usize,
    pub max_pinned_proof_size: usize,
    pub max_n_bits_ext: usize,
    pub total_const_pols_size: usize,
    pub total_const_tree_size: usize,
    pub recurser_const_slot_size: usize,
}

unsafe impl<F: PrimeField64> Send for SetupsVadcop<F> {}
unsafe impl<F: PrimeField64> Sync for SetupsVadcop<F> {}

impl<F: PrimeField64> SetupsVadcop<F> {
    pub fn new(
        global_info: &GlobalInfo,
        verify_constraints: bool,
        aggregation: bool,
        preloaded_const: &[PreLoadedConstTree],
        gpu: bool,
    ) -> ProofmanResult<Self> {
        if aggregation {
            // No `table_airs` here: the const-pols alias needs calculateFixedExtended, which
            // stark_info.cpp only ever sets for non-recursive setups.
            let sctx_compressor =
                SetupCtx::new(global_info, &ProofType::Compressor, verify_constraints, preloaded_const, &[], gpu)?;
            let sctx_recursive1 =
                SetupCtx::new(global_info, &ProofType::Recursive1, verify_constraints, preloaded_const, &[], gpu)?;
            let sctx_recursive2 =
                SetupCtx::new(global_info, &ProofType::Recursive2, verify_constraints, preloaded_const, &[], gpu)?;
            let preallocate_final = is_preload_fixed(0, 0, &ProofType::VadcopFinal, preloaded_const);
            let setup_vadcop_final = Setup::new(
                &global_info.get_setup_path("vadcop_final"),
                0,
                0,
                &GlobalInfoAir::new("VadcopFinal".to_string()),
                &ProofType::VadcopFinal,
                verify_constraints,
                preallocate_final,
                false,
                gpu,
                None,
            )?;

            // Only if the key says it carries the stage. `Setup::new` reads the starkinfo from disk
            // and panics if it is absent, so a key built without the compressed final -- which is
            // the default for blake3, where it measured a 2% smaller proof for a whole extra
            // recursion layer -- would fail here, before proving even starts.
            let setup_vadcop_final_compressed = if global_info.has_compressed_final {
                Some(Setup::new(
                    &global_info.get_setup_path("vadcop_final_compressed"),
                    0,
                    0,
                    &GlobalInfoAir::new("VadcopFinalCompressed".to_string()),
                    &ProofType::VadcopFinalCompressed,
                    verify_constraints,
                    false,
                    false,
                    gpu,
                    None,
                )?)
            } else {
                None
            };

            let recurser_const_slot_size = if gpu {
                let n_constants = setup_vadcop_final.stark_info.n_constants as usize;
                let n_rows = 1usize << setup_vadcop_final.stark_info.stark_struct.n_bits;
                1 + n_constants + n_rows * n_constants
            } else {
                0
            };

            let total_const_pols_size = sctx_compressor.total_const_pols_size
                + sctx_recursive1.total_const_pols_size
                + sctx_recursive2.total_const_pols_size
                + setup_vadcop_final.const_pols_size_packed
                + setup_vadcop_final_compressed.as_ref().map_or(0, |s| s.const_pols_size_packed)
                + recurser_const_slot_size;

            let mut total_const_tree_size = sctx_compressor.total_const_tree_size
                + sctx_recursive1.total_const_tree_size
                + sctx_recursive2.total_const_tree_size;
            if preallocate_final {
                total_const_tree_size += setup_vadcop_final.const_tree_size;
            }

            let vadcop_final_trace_size = setup_vadcop_final.stark_info.map_sections_n["cm1"]
                * (1 << setup_vadcop_final.stark_info.stark_struct.n_bits)
                + setup_vadcop_final.stark_info.n_publics;

            // Zero when the stage is absent: nothing will ever be written into a buffer for it.
            let vadcop_final_compressed_trace_size = setup_vadcop_final_compressed.as_ref().map_or(0, |s| {
                s.stark_info.map_sections_n["cm1"] * (1 << s.stark_info.stark_struct.n_bits)
                    + s.stark_info.n_publics
            });

            let max_const_size = sctx_compressor
                .max_const_size
                .max(sctx_recursive1.max_const_size)
                .max(sctx_recursive2.max_const_size)
                .max(setup_vadcop_final.const_pols_size);
            let max_const_tree_size = sctx_compressor
                .max_const_tree_size
                .max(sctx_recursive1.max_const_tree_size)
                .max(sctx_recursive2.max_const_tree_size)
                .max(setup_vadcop_final.const_tree_size)
                .max(setup_vadcop_final_compressed.as_ref().map_or(0, |s| s.const_tree_size));
            let max_prover_trace_size = sctx_compressor
                .max_prover_trace_size
                .max(sctx_recursive1.max_prover_trace_size)
                .max(sctx_recursive2.max_prover_trace_size)
                .max(vadcop_final_trace_size as usize);
            let max_prover_buffer_size = sctx_compressor
                .max_prover_buffer_size
                .max(sctx_recursive1.max_prover_buffer_size)
                .max(sctx_recursive2.max_prover_buffer_size)
                .max(setup_vadcop_final.prover_buffer_size as usize)
                .max(setup_vadcop_final_compressed.as_ref().map_or(0, |s| s.prover_buffer_size as usize));

            let max_prover_recursive2_buffer_size = (sctx_recursive2.max_prover_buffer_size
                + sctx_recursive2.max_prover_trace_size)
                .max(sctx_recursive1.max_prover_buffer_size + sctx_recursive1.max_prover_trace_size);

            let max_prover_recursive_buffer_size = (sctx_recursive2.max_prover_buffer_size
                + sctx_recursive2.max_prover_trace_size)
                .max(sctx_recursive1.max_prover_buffer_size + sctx_recursive1.max_prover_trace_size)
                .max(sctx_compressor.max_prover_buffer_size + sctx_compressor.max_prover_trace_size)
                .max(setup_vadcop_final.prover_buffer_size as usize + vadcop_final_trace_size as usize)
                // Its OWN trace size, not vadcop_final's. The two airs have had identical geometry
                // so far, which is why pairing the compressed buffer with the uncompressed trace
                // never showed; it would undersize the buffer the moment they diverged.
                .max(setup_vadcop_final_compressed.as_ref().map_or(0, |c| {
                    c.prover_buffer_size as usize + vadcop_final_compressed_trace_size as usize
                }));

            // This floors every non-recursive GPU stream class, so which term dominates decides
            // whether a regular class fits at all.
            tracing::debug!(
                "Recursive buffer requirement: compressor {}, recursive1 {}, recursive2 {}, vadcop_final {}, vadcop_final_compressed {}",
                format_bytes(
                    (sctx_compressor.max_prover_buffer_size + sctx_compressor.max_prover_trace_size) as f64 * 8.0
                ),
                format_bytes(
                    (sctx_recursive1.max_prover_buffer_size + sctx_recursive1.max_prover_trace_size) as f64 * 8.0
                ),
                format_bytes(
                    (sctx_recursive2.max_prover_buffer_size + sctx_recursive2.max_prover_trace_size) as f64 * 8.0
                ),
                format_bytes(
                    (setup_vadcop_final.prover_buffer_size as usize + vadcop_final_trace_size as usize) as f64 * 8.0
                ),
                format_bytes(
                    setup_vadcop_final_compressed.as_ref().map_or(0, |c| {
                        c.prover_buffer_size as usize + vadcop_final_compressed_trace_size as usize
                    })
                        as f64
                        * 8.0
                ),
            );

            let max_pinned_proof_size = sctx_compressor
                .max_pinned_proof_size
                .max(sctx_recursive1.max_pinned_proof_size)
                .max(sctx_recursive2.max_pinned_proof_size)
                .max(setup_vadcop_final.proof_size as usize)
                .max(setup_vadcop_final_compressed.as_ref().map_or(0, |c| c.proof_size as usize));

            let max_n_bits_ext = sctx_compressor
                .max_n_bits_ext
                .max(sctx_recursive1.max_n_bits_ext)
                .max(sctx_recursive2.max_n_bits_ext)
                .max(setup_vadcop_final.stark_info.stark_struct.n_bits_ext as usize);

            let max_witness_size = sctx_recursive1
                .max_witness_size
                .max(sctx_recursive2.max_witness_size)
                .max(setup_vadcop_final.get_circom_witness_size())
                .max(setup_vadcop_final_compressed.as_ref().map_or(0, |c| c.get_circom_witness_size()));

            let max_trace_size = sctx_recursive1
                .max_trace_size
                .max(sctx_recursive2.max_trace_size)
                .max(vadcop_final_trace_size as usize)
                .max(vadcop_final_compressed_trace_size as usize);

            let max_witness_size_compressor = sctx_compressor.max_witness_size;
            let max_trace_size_compressor = sctx_compressor.max_trace_size;

            Ok(SetupsVadcop {
                sctx_compressor: Some(sctx_compressor),
                sctx_recursive1: Some(sctx_recursive1),
                sctx_recursive2: Some(sctx_recursive2),
                setup_vadcop_final: Some(setup_vadcop_final),
                setup_vadcop_final_compressed,
                max_const_tree_size,
                max_const_size,
                max_prover_trace_size,
                max_prover_buffer_size,
                max_prover_recursive_buffer_size,
                max_prover_recursive2_buffer_size,
                max_pinned_proof_size,
                max_n_bits_ext,
                max_witness_size,
                max_trace_size,
                max_witness_size_compressor,
                max_trace_size_compressor,
                total_const_pols_size,
                total_const_tree_size,
                recurser_const_slot_size,
            })
        } else {
            Ok(SetupsVadcop {
                sctx_compressor: None,
                sctx_recursive1: None,
                sctx_recursive2: None,
                setup_vadcop_final: None,
                setup_vadcop_final_compressed: None,
                total_const_pols_size: 0,
                total_const_tree_size: 0,
                recurser_const_slot_size: 0,
                max_const_tree_size: 0,
                max_const_size: 0,
                max_prover_trace_size: 0,
                max_prover_buffer_size: 0,
                max_prover_recursive_buffer_size: 0,
                max_prover_recursive2_buffer_size: 0,
                max_pinned_proof_size: 0,
                max_n_bits_ext: 0,
                max_witness_size: 0,
                max_trace_size: 0,
                max_witness_size_compressor: 0,
                max_trace_size_compressor: 0,
            })
        }
    }

    pub fn get_setup(&self, airgroup_id: usize, air_id: usize, setup_type: &ProofType) -> ProofmanResult<&Setup<F>> {
        match setup_type {
            ProofType::Compressor => self.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id),
            ProofType::Recursive1 => self.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id),
            ProofType::Recursive2 => self.sctx_recursive2.as_ref().unwrap().get_setup(airgroup_id, air_id),
            ProofType::VadcopFinal => Ok(self.setup_vadcop_final.as_ref().unwrap()),
            ProofType::VadcopFinalCompressed => Ok(self.setup_vadcop_final_compressed.as_ref().unwrap()),
            _ => Err(ProofmanError::InvalidSetup("Invalid setup type".into())),
        }
    }
}

#[derive(Debug)]
pub struct SetupRepository<F: PrimeField64> {
    setups: HashMap<(usize, usize), Setup<F>>,
    max_const_tree_size: usize,
    max_const_size: usize,
    max_prover_buffer_size: usize,
    prover_buffer_sizes: Vec<((usize, usize), usize)>,
    max_prover_contributions_size: usize,
    max_prover_trace_size: usize,
    max_pinned_proof_size: usize,
    max_witness_size: usize,
    max_trace_size: usize,
    total_const_pols_size: usize,
    total_const_tree_size: usize,
    fixed_groups: HashMap<(usize, usize), FixedGroup>,
    global_bin: Option<*mut c_void>,
    global_info_file: String,
    max_n_bits_ext: usize,
}

unsafe impl<F: PrimeField64> Send for SetupRepository<F> {}
unsafe impl<F: PrimeField64> Sync for SetupRepository<F> {}

impl<F: PrimeField64> Drop for SetupRepository<F> {
    fn drop(&mut self) {
        if let Some(global_bin_ptr) = self.global_bin {
            expressions_bin_free_c(global_bin_ptr);
        }
    }
}

impl<F: PrimeField64> SetupRepository<F> {
    pub fn new(
        global_info: &GlobalInfo,
        setup_type: &ProofType,
        verify_constraints: bool,
        preloaded_const: &[PreLoadedConstTree],
        table_airs: &[(usize, usize)],
        gpu: bool,
    ) -> ProofmanResult<Self> {
        let mut setups = HashMap::new();

        let global_bin = match setup_type == &ProofType::Basic {
            true => {
                let global_bin_path =
                    &global_info.get_proving_key_path().join("pilout.globalConstraints.bin").display().to_string();
                Some(expressions_bin_new_c(global_bin_path.as_str(), true, false))
            }
            false => None,
        };

        let global_info_path = &global_info.get_proving_key_path().join("pilout.globalInfo.json");
        let global_info_file = global_info_path.to_str().unwrap().to_string();

        let mut max_const_tree_size = 0;
        let mut max_const_size = 0;
        let mut max_n_bits_ext = 0;
        let mut max_prover_contributions_size = 0;
        let mut max_prover_buffer_size = 0;
        let mut prover_buffer_sizes: Vec<((usize, usize), usize)> = Vec::new();
        let mut max_prover_trace_size = 0;
        let mut max_pinned_proof_size = 0;
        let mut total_const_pols_size = 0;
        let mut total_const_tree_size = 0;
        let mut max_witness_size = 0;
        let mut max_trace_size = 0;

        // Airs in the order load_device_const_pols walks them, and the slot each verkey maps
        // to. `preallocate` is OR-ed over a group: members share one tree, so preloading it
        // for one preloads it for all.
        let mut sized_airs: Vec<(usize, usize)> = Vec::new();
        let mut groups: HashMap<Vec<u64>, ((usize, usize), bool)> = HashMap::new();

        // Initialize Hashmap for each airgroup_id, air_id

        for (airgroup_id, air_group) in global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let preallocate = is_preload_fixed(airgroup_id, air_id, setup_type, preloaded_const);
                let single_use = table_airs.contains(&(airgroup_id, air_id));
                let setup_path = global_info.get_air_setup_path(airgroup_id, air_id, setup_type);
                let setup = Setup::new(
                    &setup_path,
                    airgroup_id,
                    air_id,
                    &global_info.airs[airgroup_id][air_id],
                    setup_type,
                    verify_constraints,
                    preallocate,
                    single_use,
                    gpu,
                    Some(&global_info.get_air_setup_path(airgroup_id, 0, &ProofType::Recursive2)),
                )?;
                if setup_type != &ProofType::Compressor || global_info.get_air_has_compressor(airgroup_id, air_id) {
                    let n = 1 << setup.stark_info.stark_struct.n_bits;
                    let n_bits_ext = setup.stark_info.stark_struct.n_bits_ext;
                    let trace_size = setup.stark_info.map_sections_n["cm1"] * n;
                    let mut total_prover_trace_size = trace_size as usize;
                    total_prover_trace_size += setup.stark_info.n_publics as usize;
                    total_prover_trace_size += setup.stark_info.airvalues_map.as_ref().map_or(0, |v| 3 * v.len());
                    total_prover_trace_size += setup.stark_info.airgroupvalues_map.as_ref().map_or(0, |v| 3 * v.len());
                    total_prover_trace_size += global_info.proof_values_map.as_ref().map_or(0, |v| 3 * v.len());
                    total_prover_trace_size += 3;
                    if max_const_tree_size < setup.const_tree_size {
                        max_const_tree_size = setup.const_tree_size;
                    }
                    if max_const_size < setup.const_pols_size {
                        max_const_size = setup.const_pols_size;
                    }
                    if max_prover_buffer_size < setup.prover_buffer_size {
                        max_prover_buffer_size = setup.prover_buffer_size;
                    }
                    prover_buffer_sizes.push(((airgroup_id, air_id), setup.prover_buffer_size as usize));
                    if max_prover_contributions_size < setup.contributions_size {
                        max_prover_contributions_size = setup.contributions_size;
                    }
                    max_prover_trace_size = max_prover_trace_size.max(total_prover_trace_size);

                    if setup.gpu {
                        sized_airs.push((airgroup_id, air_id));
                        if !setup.verkey.is_empty() {
                            let group = groups.entry(setup.get_vk()).or_insert(((airgroup_id, air_id), false));
                            group.1 |= preallocate;
                        }
                    }
                    max_pinned_proof_size = max_pinned_proof_size.max(setup.pinned_proof_size);
                    max_n_bits_ext = max_n_bits_ext.max(n_bits_ext);
                    max_witness_size = max_witness_size.max(setup.get_circom_witness_size());
                    max_trace_size = max_trace_size.max(trace_size as usize);
                }
                setups.insert((airgroup_id, air_id), setup);
                if setup_type == &ProofType::Recursive2 {
                    break;
                }
            }
        }

        // Second pass: `load_tree` is only settled once every member has been seen. Sizes one
        // slot per group, in the same order the loader assigns offsets -- must not drift.
        let mut fixed_groups: HashMap<(usize, usize), FixedGroup> = HashMap::new();
        let mut sized_slots: HashSet<(usize, usize)> = HashSet::new();
        let mut shared_airs = 0;
        let mut saved = 0;
        for air in sized_airs {
            let setup = &setups[&air];
            let group = match groups.get(&setup.get_vk()) {
                Some(&(owner, load_tree)) => FixedGroup { owner, load_tree },
                // Nothing to fingerprint with (no verkey): the air is its own group.
                None => FixedGroup { owner: air, load_tree: setup.preallocate },
            };
            fixed_groups.insert(air, group);
            if sized_slots.insert(group.owner) {
                total_const_pols_size += setup.const_pols_size_packed;
                if group.load_tree {
                    total_const_tree_size += setup.const_tree_size;
                }
            } else {
                shared_airs += 1;
                saved += setup.const_pols_size_packed;
                if group.load_tree {
                    saved += setup.const_tree_size;
                }
            }
        }
        if shared_airs > 0 {
            let proof_type: &str = (*setup_type).into();
            tracing::info!(
                "Sharing GPU const pols: {shared_airs} {proof_type} airs reuse another air's fixed columns, \
                 saving {} MB",
                saved * std::mem::size_of::<u64>() / (1024 * 1024)
            );
        }

        prover_buffer_sizes.sort_by(|(ka, sa), (kb, sb)| sb.cmp(sa).then(ka.cmp(kb)));

        Ok(Self {
            setups,
            fixed_groups,
            global_bin,
            global_info_file,
            max_const_tree_size,
            max_const_size,
            max_prover_contributions_size: max_prover_contributions_size as usize,
            max_prover_buffer_size: max_prover_buffer_size as usize,
            prover_buffer_sizes,
            max_prover_trace_size,
            max_pinned_proof_size: max_pinned_proof_size as usize,
            total_const_pols_size,
            total_const_tree_size,
            max_witness_size,
            max_trace_size,
            max_n_bits_ext: max_n_bits_ext as usize,
        })
    }
}

/// Air instance context for managing air instances (traces)
#[allow(dead_code)]
pub struct SetupCtx<F: PrimeField64> {
    setup_repository: SetupRepository<F>,
    pub max_const_tree_size: usize,
    pub max_const_size: usize,
    pub max_prover_contributions_size: usize,
    pub max_prover_buffer_size: usize,
    /// Per-air prover buffer size, largest first. See `SetupRepository::prover_buffer_sizes`.
    pub prover_buffer_sizes: Vec<((usize, usize), usize)>,
    pub max_prover_trace_size: usize,
    pub max_pinned_proof_size: usize,
    pub max_witness_size: usize,
    pub max_trace_size: usize,
    pub max_n_bits_ext: usize,
    pub total_const_pols_size: usize,
    pub total_const_tree_size: usize,
    setup_type: ProofType,
}

impl<F: PrimeField64> SetupCtx<F> {
    pub fn new(
        global_info: &GlobalInfo,
        setup_type: &ProofType,
        verify_constraints: bool,
        preloaded_const: &[PreLoadedConstTree],
        table_airs: &[(usize, usize)],
        gpu: bool,
    ) -> ProofmanResult<Self> {
        let setup_repository =
            SetupRepository::new(global_info, setup_type, verify_constraints, preloaded_const, table_airs, gpu)?;
        let max_const_tree_size = setup_repository.max_const_tree_size;
        let max_const_size = setup_repository.max_const_size;
        let max_prover_contributions_size = setup_repository.max_prover_contributions_size;
        let max_prover_buffer_size = setup_repository.max_prover_buffer_size;
        let prover_buffer_sizes = setup_repository.prover_buffer_sizes.clone();
        let max_prover_trace_size = setup_repository.max_prover_trace_size;
        let max_pinned_proof_size = setup_repository.max_pinned_proof_size;
        let total_const_pols_size = setup_repository.total_const_pols_size;
        let total_const_tree_size = setup_repository.total_const_tree_size;
        let max_witness_size = setup_repository.max_witness_size;
        let max_trace_size = setup_repository.max_trace_size;
        let max_n_bits_ext = setup_repository.max_n_bits_ext;
        Ok(SetupCtx {
            setup_repository,
            max_const_tree_size,
            max_const_size,
            max_prover_contributions_size,
            max_prover_buffer_size,
            prover_buffer_sizes,
            max_witness_size,
            max_trace_size,
            max_prover_trace_size,
            max_pinned_proof_size,
            max_n_bits_ext,
            total_const_pols_size,
            total_const_tree_size,
            setup_type: *setup_type,
        })
    }

    pub fn get_setup(&self, airgroup_id: usize, air_id: usize) -> ProofmanResult<&Setup<F>> {
        match self.setup_repository.setups.get(&(airgroup_id, air_id)) {
            Some(setup) => Ok(setup),
            None => Err(ProofmanError::InvalidSetup(format!(
                "Setup not found for airgroup_id: {}, air_id: {}",
                airgroup_id, air_id
            ))),
        }
    }

    /// `None` for airs never uploaded to the GPU (CPU setups, compressor-less airs).
    pub fn get_fixed_group(&self, airgroup_id: usize, air_id: usize) -> Option<FixedGroup> {
        self.setup_repository.fixed_groups.get(&(airgroup_id, air_id)).copied()
    }

    pub fn get_fixed(&self, airgroup_id: usize, air_id: usize) -> ProofmanResult<Vec<F>> {
        match self.setup_repository.setups.get(&(airgroup_id, air_id)) {
            Some(setup) => {
                let mut const_pols: Vec<F> = vec![F::ZERO; setup.const_pols_size];
                load_const_pols(setup, &mut const_pols);
                Ok(const_pols)
            }
            None => Err(ProofmanError::InvalidSetup(format!(
                "Setup not found for airgroup_id: {}, air_id: {}",
                airgroup_id, air_id
            ))),
        }
    }

    pub fn get_setups_list(&self) -> Vec<(usize, usize)> {
        self.setup_repository.setups.keys().cloned().collect()
    }

    pub fn get_global_bin(&self) -> *mut c_void {
        self.setup_repository.global_bin.unwrap()
    }

    pub fn get_global_info_file(&self) -> String {
        self.setup_repository.global_info_file.clone()
    }
}
