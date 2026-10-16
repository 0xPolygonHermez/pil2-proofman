use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::path::PathBuf;

use proofman_fields::PrimeField64;
use proofman_starks_lib_c::{expressions_bin_new_c, expressions_bin_free_c};

use crate::format_bytes;
use crate::load_const_pols;
use crate::{GlobalInfo, GlobalInfoAir};
use crate::useful_lru_slots;
use crate::ProofmanError;
use crate::ProofmanResult;
use crate::Setup;
use crate::exec_header;
use crate::ProofType;

/// The slot an air's packed const pols occupy in the GPU const buffer. Airs with identical
/// fixed columns -- same verkey, i.e. same const-tree root -- share one slot; `owner` is the
/// first of the group and the only one that uploads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedGroup {
    pub owner: (usize, usize),
    /// Words the slot reserves for custom commits: the largest member's, since members register
    /// their own files into it. `None` = the air's own reservation (a group of one).
    pub custom_words: Option<usize>,
}

/// Columns the recursion's HOST trace buffer needs for one air.
///
/// On GPU it only stages the exec map's width for `widenCompactWitnessGPU` to widen device-side. On
/// CPU it is the air's full width: since the solve and the scatter are fused, `generate_witness`
/// takes a pooled buffer before the prover buffer is in scope, so the fill can no longer go straight
/// into the prover's own cm1 slot. Must agree with `recursion_trace_stride`, which decides the fill
/// from the same exec header -- sizing narrow while the fill goes wide overruns the buffer.
///
/// One function because three call sites decide this, and the one that did it inline sized the whole
/// pool at the air's full width while the other two were compact.
pub fn recursion_staging_cols<F: PrimeField64>(setup: &Setup<F>, gpu: bool) -> u64 {
    let cm1 = setup.stark_info.map_sections_n["cm1"];
    if !gpu {
        return cm1;
    }
    setup.exec_data.as_deref().map(|e| exec_header(e).map_cols).filter(|&m| m > 0 && m < cm1).unwrap_or(cm1)
}

pub struct SetupsVadcop<F: PrimeField64> {
    pub sctx_compressor: Option<SetupCtx<F>>,
    pub sctx_recursive1: Option<SetupCtx<F>>,
    pub sctx_recursive2: Option<SetupCtx<F>>,
    pub setup_vadcop_final: Option<Setup<F>>,
    pub setup_vadcop_final_compressed: Option<Setup<F>>,
    /// The key's recursivef (setup-recursivef, blake3): loaded when the key's hash is blake3 and it
    /// has one, as the key's other setups are; the final SNARK wraps it
    /// (`ProofMan::generate_final_snark_proof`).
    pub setup_recursivef: Option<Setup<F>>,
    pub max_compact_trace_size: usize,
    pub max_const_size: usize,
    pub max_const_tree_size: usize,
    pub max_prover_buffer_size: usize,
    pub max_prover_recursive_buffer_size: usize,
    pub max_prover_recursive2_buffer_size: usize,
    pub max_pinned_proof_size: usize,
    pub max_n_bits_ext: usize,
    /// Resident aggregation const area, for callers without the const cache (prove-air sets it); 0
    /// otherwise: the planner sizes the cache.
    pub total_const_pols_size: usize,
    /// Const cache slot 0 (largest set it can take) and normal (recursive1-sized) slots, in words.
    pub cache_big_words: u64,
    pub cache_normal_words: u64,
    /// LRU slots the sets that always use the LRU can keep busy (see useful_lru_slots).
    pub cache_n_lru_keys: usize,
    /// recursive2 pins, all airgroups; each set fits slot 0, so they can go through the LRU instead.
    pub rec2_words: u64,
    pub rec2_fits_lru: bool,
    /// Final pins the planner takes when it has room.
    pub finals_pin_words: u64,
    /// Pinned slots each pin kind takes.
    pub rec2_sets: usize,
    pub finals_sets: usize,
}

unsafe impl<F: PrimeField64> Send for SetupsVadcop<F> {}
unsafe impl<F: PrimeField64> Sync for SetupsVadcop<F> {}

/// How an aggregation const set uses the GPU const cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheSetKind {
    /// Recursive1 and compressor sets: always through the LRU slots.
    Lru,
    /// Pinned, one slot per airgroup.
    Recursive2,
    /// vadcop_final, its compressed variant and a blake3 key's recursivef: pinned when the planner
    /// has room, else LRU.
    Final,
}

/// One aggregation const set: the cache sizing and the loader walk the same list.
pub struct CacheSet<'a, F: PrimeField64> {
    pub setup: &'a Setup<F>,
    pub airgroup_id: usize,
    pub air_id: usize,
    pub kind: CacheSetKind,
}

fn cache_sets<'a, F: PrimeField64>(
    global_info: &GlobalInfo,
    recursive1: &'a SetupCtx<F>,
    compressor: &'a SetupCtx<F>,
    recursive2: &'a SetupCtx<F>,
    vadcop_final: &'a Setup<F>,
    vadcop_final_compressed: Option<&'a Setup<F>>,
    recursivef: Option<&'a Setup<F>>,
) -> ProofmanResult<Vec<CacheSet<'a, F>>> {
    let mut sets = Vec::new();
    for (ag, group) in global_info.airs.iter().enumerate() {
        for air in 0..group.len() {
            sets.push(CacheSet {
                setup: recursive1.get_setup(ag, air)?,
                airgroup_id: ag,
                air_id: air,
                kind: CacheSetKind::Lru,
            });
            if global_info.get_air_has_compressor(ag, air) {
                sets.push(CacheSet {
                    setup: compressor.get_setup(ag, air)?,
                    airgroup_id: ag,
                    air_id: air,
                    kind: CacheSetKind::Lru,
                });
            }
        }
        sets.push(CacheSet {
            setup: recursive2.get_setup(ag, 0)?,
            airgroup_id: ag,
            air_id: 0,
            kind: CacheSetKind::Recursive2,
        });
    }
    for setup in std::iter::once(vadcop_final).chain(vadcop_final_compressed).chain(recursivef) {
        sets.push(CacheSet { setup, airgroup_id: 0, air_id: 0, kind: CacheSetKind::Final });
    }
    Ok(sets)
}

impl<F: PrimeField64> SetupsVadcop<F> {
    /// The aggregation const sets in cache order (aggregation runs only).
    pub fn cache_sets(&self, global_info: &GlobalInfo) -> ProofmanResult<Vec<CacheSet<'_, F>>> {
        cache_sets(
            global_info,
            self.sctx_recursive1.as_ref().unwrap(),
            self.sctx_compressor.as_ref().unwrap(),
            self.sctx_recursive2.as_ref().unwrap(),
            self.setup_vadcop_final.as_ref().unwrap(),
            self.setup_vadcop_final_compressed.as_ref(),
            self.setup_recursivef.as_ref(),
        )
    }

    /// `final_snark`: the run makes final SNARKs, so a blake3 key's recursivef is loaded.
    pub fn new(
        global_info: &GlobalInfo,
        verify_constraints: bool,
        aggregation: bool,
        gpu: bool,
        final_snark: bool,
    ) -> ProofmanResult<Self> {
        let custom_commits_fixed = &HashMap::new();
        if aggregation {
            let sctx_compressor = SetupCtx::new(global_info, &ProofType::Compressor, verify_constraints, gpu)?;
            let sctx_recursive1 = SetupCtx::new(global_info, &ProofType::Recursive1, verify_constraints, gpu)?;
            let sctx_recursive2 = SetupCtx::new(global_info, &ProofType::Recursive2, verify_constraints, gpu)?;
            let setup_vadcop_final = Setup::new(
                &global_info.get_setup_path("vadcop_final"),
                0,
                0,
                &GlobalInfoAir::new("VadcopFinal".to_string()),
                &ProofType::VadcopFinal,
                verify_constraints,
                gpu,
                None,
                custom_commits_fixed,
                false,
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
                    gpu,
                    None,
                    custom_commits_fixed,
                    false,
                )?)
            } else {
                None
            };

            // A Goldilocks STARK of the key's own family, the layer after the vadcop_final; the
            // poseidon families' recursivef is setup-snark's (BN128) and the SnarkWrapper's.
            let path = global_info.get_setup_path("recursivef");
            let has_recursivef = PathBuf::from(format!("{}.starkinfo.json", path.display())).exists();
            let setup_recursivef = if final_snark && global_info.hash == "blake3" && has_recursivef {
                Some(Setup::new(
                    &path,
                    0,
                    0,
                    &GlobalInfoAir::new("RecursiveF".to_string()),
                    &ProofType::RecursiveF,
                    verify_constraints,
                    gpu,
                    None,
                    custom_commits_fixed,
                    false,
                )?)
            } else {
                None
            };
            let finals = || [setup_vadcop_final_compressed.as_ref(), setup_recursivef.as_ref()].into_iter().flatten();

            let sets = cache_sets(
                global_info,
                &sctx_recursive1,
                &sctx_compressor,
                &sctx_recursive2,
                &setup_vadcop_final,
                setup_vadcop_final_compressed.as_ref(),
                setup_recursivef.as_ref(),
            )?;
            let words_of = |kind: CacheSetKind| {
                sets.iter().filter(move |s| s.kind == kind).map(|s| s.setup.const_pols_size_packed as u64)
            };
            let lru_sizes: Vec<u64> = words_of(CacheSetKind::Lru).collect();
            let rec2_words: u64 = words_of(CacheSetKind::Recursive2).sum();
            let finals_pin_words: u64 = words_of(CacheSetKind::Final).sum();
            let rec2_sets = words_of(CacheSetKind::Recursive2).count();
            let finals_sets = words_of(CacheSetKind::Final).count();
            let cache_normal_words = sctx_recursive1.max_const_pols_size_packed as u64;
            // A recurser has vadcop_final's shape but its own constants, so its packed size is only
            // bounded by the unpacked worst case; it always uses the LRU, so slot 0 keeps room for it.
            let recurser_words = if gpu {
                let n_constants = setup_vadcop_final.stark_info.n_constants;
                let n_rows = 1u64 << setup_vadcop_final.stark_info.stark_struct.n_bits;
                1 + n_constants + n_rows * n_constants
            } else {
                0
            };
            // Unpinned finals land in the LRU, so slot 0 holds them too.
            let cache_big_words = lru_sizes
                .iter()
                .copied()
                .chain(words_of(CacheSetKind::Final))
                .fold(cache_normal_words.max(recurser_words), u64::max);
            let cache_n_lru_keys = useful_lru_slots(&lru_sizes, cache_normal_words);
            let rec2_fits_lru = words_of(CacheSetKind::Recursive2).all(|w| w <= cache_big_words);
            drop(sets);
            // The planner sizes the const cache; this resident area is for callers without one (prove-air).
            let total_const_pols_size = 0;

            let max_const_size = sctx_compressor
                .max_const_size
                .max(sctx_recursive1.max_const_size)
                .max(sctx_recursive2.max_const_size)
                .max(setup_vadcop_final.const_pols_size)
                .max(setup_recursivef.as_ref().map_or(0, |s| s.const_pols_size));
            let max_const_tree_size = sctx_compressor
                .max_const_tree_size
                .max(sctx_recursive1.max_const_tree_size)
                .max(sctx_recursive2.max_const_tree_size)
                .max(setup_vadcop_final.const_tree_size)
                .max(finals().map(|s| s.const_tree_size).max().unwrap_or(0));
            let max_prover_buffer_size = sctx_compressor
                .max_prover_buffer_size
                .max(sctx_recursive1.max_prover_buffer_size)
                .max(sctx_recursive2.max_prover_buffer_size)
                .max(setup_vadcop_final.prover_buffer_size as usize)
                .max(finals().map(|s| s.prover_buffer_size as usize).max().unwrap_or(0));

            // Recursive-capable buffers = prover buffer (mapTotalN) + witness tail: the room past the
            // proof layout where the recursive proof's COMPACT host trace lands (staging cols x N:
            // each repository's `max_compact_trace_size`, the finals' own staging width) before the device
            // widens it into cm1; gen_recursive_proof_gpu checks the room at launch. Sized exactly
            // to that trace -- 4e44d5549 sized it at the full trace, ~1 GB more than ever lands.
            let vadcop_final_tail = (recursion_staging_cols(&setup_vadcop_final, gpu)
                * (1 << setup_vadcop_final.stark_info.stark_struct.n_bits))
                as usize;
            // Its OWN staging width, not vadcop_final's: the two airs have had identical geometry so
            // far, which is why pairing them never showed; it would undersize the buffer the moment
            // they diverged.
            let vadcop_final_compressed_tail = setup_vadcop_final_compressed
                .as_ref()
                .map_or(0, |s| (recursion_staging_cols(s, gpu) * (1 << s.stark_info.stark_struct.n_bits)) as usize);
            let recursivef_tail = setup_recursivef
                .as_ref()
                .map_or(0, |s| (recursion_staging_cols(s, gpu) * (1 << s.stark_info.stark_struct.n_bits)) as usize);
            let max_prover_recursive2_buffer_size = (sctx_recursive2.max_prover_buffer_size
                + sctx_recursive2.max_compact_trace_size)
                .max(sctx_recursive1.max_prover_buffer_size + sctx_recursive1.max_compact_trace_size);

            let max_prover_recursive_buffer_size = (sctx_recursive2.max_prover_buffer_size
                + sctx_recursive2.max_compact_trace_size)
                .max(sctx_recursive1.max_prover_buffer_size + sctx_recursive1.max_compact_trace_size)
                .max(sctx_compressor.max_prover_buffer_size + sctx_compressor.max_compact_trace_size)
                .max(setup_vadcop_final.prover_buffer_size as usize + vadcop_final_tail)
                .max(
                    setup_vadcop_final_compressed
                        .as_ref()
                        .map_or(0, |c| c.prover_buffer_size as usize + vadcop_final_compressed_tail),
                )
                .max(setup_recursivef.as_ref().map_or(0, |c| c.prover_buffer_size as usize + recursivef_tail));

            // This floors every non-recursive GPU stream class, so which term dominates decides
            // whether a regular class fits at all.
            tracing::debug!(
                "Recursive buffer requirement: compressor {}, recursive1 {}, recursive2 {}, vadcop_final {}, vadcop_final_compressed {}",
                format_bytes((sctx_compressor.max_prover_buffer_size + sctx_compressor.max_compact_trace_size) as f64 * 8.0),
                format_bytes((sctx_recursive1.max_prover_buffer_size + sctx_recursive1.max_compact_trace_size) as f64 * 8.0),
                format_bytes((sctx_recursive2.max_prover_buffer_size + sctx_recursive2.max_compact_trace_size) as f64 * 8.0),
                format_bytes((setup_vadcop_final.prover_buffer_size as usize + vadcop_final_tail) as f64 * 8.0),
                format_bytes(
                    setup_vadcop_final_compressed
                        .as_ref()
                        .map_or(0, |c| c.prover_buffer_size as usize + vadcop_final_compressed_tail) as f64
                        * 8.0
                ),
            );

            let max_pinned_proof_size = sctx_compressor
                .max_pinned_proof_size
                .max(sctx_recursive1.max_pinned_proof_size)
                .max(sctx_recursive2.max_pinned_proof_size)
                .max(setup_vadcop_final.proof_size as usize)
                .max(finals().map(|c| c.proof_size as usize).max().unwrap_or(0));

            let max_n_bits_ext = sctx_compressor
                .max_n_bits_ext
                .max(sctx_recursive1.max_n_bits_ext)
                .max(sctx_recursive2.max_n_bits_ext)
                .max(setup_vadcop_final.stark_info.stark_struct.n_bits_ext as usize)
                .max(setup_recursivef.as_ref().map_or(0, |s| s.stark_info.stark_struct.n_bits_ext as usize));

            // Largest compact (staging-width) trace over every recursive proof kind, compressor
            // included: they share one host pool (see MemoryHandlerRecursive::trace), and it is also
            // the witness tail each recursive-capable stream class reserves past mapTotalN.
            let max_compact_trace_size = sctx_recursive1
                .max_compact_trace_size
                .max(sctx_recursive2.max_compact_trace_size)
                .max(sctx_compressor.max_compact_trace_size)
                // Their STAGING width, not `vadcop_final_trace_size` -- that one is the prover
                // buffer's and stays full. Sizing the shared pool from it put every buffer at the
                // air's full width, which is what made the pool 7x what it holds.
                .max(vadcop_final_tail)
                .max(vadcop_final_compressed_tail)
                .max(recursivef_tail);

            Ok(SetupsVadcop {
                sctx_compressor: Some(sctx_compressor),
                sctx_recursive1: Some(sctx_recursive1),
                sctx_recursive2: Some(sctx_recursive2),
                setup_vadcop_final: Some(setup_vadcop_final),
                setup_vadcop_final_compressed,
                setup_recursivef,
                max_const_tree_size,
                max_const_size,
                max_prover_buffer_size,
                max_prover_recursive_buffer_size,
                max_prover_recursive2_buffer_size,
                max_pinned_proof_size,
                max_n_bits_ext,
                max_compact_trace_size,
                total_const_pols_size,
                cache_big_words,
                cache_normal_words,
                cache_n_lru_keys,
                rec2_words,
                rec2_fits_lru,
                finals_pin_words,
                rec2_sets,
                finals_sets,
            })
        } else {
            Ok(SetupsVadcop {
                sctx_compressor: None,
                sctx_recursive1: None,
                sctx_recursive2: None,
                setup_vadcop_final: None,
                setup_vadcop_final_compressed: None,
                setup_recursivef: None,
                total_const_pols_size: 0,
                cache_big_words: 0,
                cache_normal_words: 0,
                cache_n_lru_keys: 0,
                rec2_words: 0,
                rec2_fits_lru: false,
                finals_pin_words: 0,
                rec2_sets: 0,
                finals_sets: 0,
                max_const_tree_size: 0,
                max_const_size: 0,
                max_prover_buffer_size: 0,
                max_prover_recursive_buffer_size: 0,
                max_prover_recursive2_buffer_size: 0,
                max_pinned_proof_size: 0,
                max_n_bits_ext: 0,
                max_compact_trace_size: 0,
            })
        }
    }

    /// Size of every pooled signalValues buffer (in u64 elements): the largest `total_signal_no`
    /// across the recursive/compressor/final setups, so every circuit takes a pooled buffer and
    /// none allocates its own.
    ///
    /// Sizing from the second largest would spare `n_streams x (largest - second)` of host memory
    /// -- 328 MB per stream on the blake3 key, where the sole compressor (Keccakf) is 3.1x the next
    /// -- and it is not worth it: that made the compressor allocate and first-touch 485 MB per
    /// proof, measured at +37.6 ms on its witness (204.1 vs 166.5 ms, n=36 each). 0 if no setup
    /// exposes a `total_signal_no`, and the pool then stays off.
    pub fn signal_pool_cap(&self) -> usize {
        let mut sizes: Vec<usize> = Vec::new();
        for sctx in [self.sctx_compressor.as_ref(), self.sctx_recursive1.as_ref(), self.sctx_recursive2.as_ref()]
            .into_iter()
            .flatten()
        {
            sizes.extend(sctx.total_signal_nos());
        }
        for setup in [
            self.setup_vadcop_final.as_ref(),
            self.setup_vadcop_final_compressed.as_ref(),
            self.setup_recursivef.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if let Some(n) = setup.total_signal_no {
                sizes.push(n as usize);
            }
        }
        sizes.into_iter().max().unwrap_or(0)
    }

    pub fn get_setup(&self, airgroup_id: usize, air_id: usize, setup_type: &ProofType) -> ProofmanResult<&Setup<F>> {
        match setup_type {
            ProofType::Compressor => self.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id),
            ProofType::Recursive1 => self.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id),
            ProofType::Recursive2 => self.sctx_recursive2.as_ref().unwrap().get_setup(airgroup_id, air_id),
            ProofType::VadcopFinal => Ok(self.setup_vadcop_final.as_ref().unwrap()),
            // The only optional setup of the family: keys built without the compressed-final
            // stage (blake3's default) legitimately have none.
            ProofType::VadcopFinalCompressed => self.setup_vadcop_final_compressed.as_ref().ok_or_else(|| {
                ProofmanError::InvalidSetup("Proving key was built without the vadcop_final_compressed stage".into())
            }),
            ProofType::RecursiveF => self.setup_recursivef.as_ref().ok_or_else(|| {
                ProofmanError::InvalidSetup(
                    "the proving key has no recursivef: a blake3 key's comes from setup-recursivef".into(),
                )
            }),
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
    max_pinned_proof_size: usize,
    max_compact_trace_size: usize,
    total_const_pols_size: usize,
    total_custom_commits_reserved_words: usize,
    fixed_groups: HashMap<(usize, usize), FixedGroup>,
    global_bin: Option<*mut c_void>,
    global_info_file: String,
    max_n_bits_ext: usize,
    max_const_pols_size_packed: usize,
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
        gpu: bool,
        custom_commits_fixed: &HashMap<String, PathBuf>,
        self_contained: bool,
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
        let mut max_prover_buffer_size = 0;
        let mut prover_buffer_sizes: Vec<((usize, usize), usize)> = Vec::new();
        let mut max_pinned_proof_size = 0;
        let mut total_const_pols_size = 0;
        let mut total_custom_commits_reserved_words = 0;
        let mut max_compact_trace_size = 0;
        let mut max_const_pols_size_packed = 0;

        // Airs in the order load_device_const_pols walks them, and the slot each verkey maps to.
        let mut sized_airs: Vec<(usize, usize)> = Vec::new();
        let mut groups: HashMap<Vec<u64>, (usize, usize)> = HashMap::new();

        // Initialize Hashmap for each airgroup_id, air_id

        for (airgroup_id, air_group) in global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let setup_path = global_info.get_air_setup_path(airgroup_id, air_id, setup_type);
                let setup = Setup::new(
                    &setup_path,
                    airgroup_id,
                    air_id,
                    &global_info.airs[airgroup_id][air_id],
                    setup_type,
                    verify_constraints,
                    gpu,
                    Some(&global_info.get_air_setup_path(airgroup_id, 0, &ProofType::Recursive2)),
                    custom_commits_fixed,
                    self_contained,
                )?;
                if setup_type != &ProofType::Compressor || global_info.get_air_has_compressor(airgroup_id, air_id) {
                    let n = 1 << setup.stark_info.stark_struct.n_bits;
                    let n_bits_ext = setup.stark_info.stark_struct.n_bits_ext;
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
                    if setup.gpu {
                        sized_airs.push((airgroup_id, air_id));
                        if !setup.verkey.is_empty() {
                            groups.entry(setup.get_vk()).or_insert((airgroup_id, air_id));
                        }
                    }
                    max_pinned_proof_size = max_pinned_proof_size.max(setup.pinned_proof_size);
                    max_n_bits_ext = max_n_bits_ext.max(n_bits_ext);
                    // Basic airs never land a compact witness (no exec map; their trace goes straight
                    // into cm1), so their repository reports 0 rather than a meaningless full width.
                    if setup_type != &ProofType::Basic {
                        max_compact_trace_size =
                            max_compact_trace_size.max((recursion_staging_cols(&setup, gpu) * n) as usize);
                    }
                }
                setups.insert((airgroup_id, air_id), setup);
                if setup_type == &ProofType::Recursive2 {
                    break;
                }
            }
        }

        // Second pass: the group owner is only settled once every member has been seen. Sizes
        // one slot per group, in the same order the loader assigns offsets -- must not drift.
        let mut fixed_groups: HashMap<(usize, usize), FixedGroup> = HashMap::new();
        let mut sized_slots: HashSet<(usize, usize)> = HashSet::new();
        let owner_of = |air: &(usize, usize)| groups.get(&setups[air].get_vk()).copied().unwrap_or(*air);
        let mut group_custom_words: HashMap<(usize, usize), usize> = HashMap::new();
        for air in &sized_airs {
            let words = group_custom_words.entry(owner_of(air)).or_insert(0);
            *words = (*words).max(setups[air].custom_commits_reserved_words);
        }
        let mut shared_airs = 0;
        let mut saved = 0;
        for air in sized_airs {
            let setup = &setups[&air];
            // No verkey to fingerprint with: the air is its own group.
            let owner = owner_of(&air);
            let group = FixedGroup { owner, custom_words: Some(group_custom_words[&owner]) };
            fixed_groups.insert(air, group);
            max_const_pols_size_packed = max_const_pols_size_packed.max(setup.const_pols_size_packed);
            if sized_slots.insert(group.owner) {
                total_const_pols_size += setup.const_pols_size_packed;
                // Custom commits ride the same buffer; the slot is filled later, when
                // register_custom_commits supplies the file path.
                total_const_pols_size += group_custom_words[&owner];
                total_custom_commits_reserved_words += group_custom_words[&owner];
            } else {
                shared_airs += 1;
                saved += setup.const_pols_size_packed;
                saved += setup.custom_commits_reserved_words;
            }
        }
        // Only basic airs share a const slot; aggregation sets go through the const cache per air.
        if shared_airs > 0 && *setup_type == ProofType::Basic {
            let proof_type: &str = (*setup_type).into();
            tracing::info!(
                "Sharing GPU const pols: {shared_airs} {proof_type} airs reuse another air's fixed columns, \
                 saving {} MB",
                saved * std::mem::size_of::<u64>() / (1024 * 1024)
            );
        }

        prover_buffer_sizes.sort_by(|(ka, sa), (kb, sb)| sb.cmp(sa).then(ka.cmp(kb)));

        // Aggregation sets live in the const slot cache (host pinned copies, see
        // DeviceCommitBuffers::constCache), which has no room for custom commits.
        if gpu && *setup_type != ProofType::Basic {
            assert!(
                total_custom_commits_reserved_words == 0,
                "aggregation setups with custom commits cannot use the const slot cache"
            );
            // Nor do they take basic-area words (prove-air passes this repository as its basic one).
            total_const_pols_size = 0;
        }

        Ok(Self {
            setups,
            fixed_groups,
            global_bin,
            global_info_file,
            max_const_tree_size,
            max_const_size,
            max_prover_buffer_size: max_prover_buffer_size as usize,
            prover_buffer_sizes,
            max_pinned_proof_size: max_pinned_proof_size as usize,
            total_const_pols_size,
            total_custom_commits_reserved_words,
            max_compact_trace_size,
            max_n_bits_ext: max_n_bits_ext as usize,
            max_const_pols_size_packed,
        })
    }
}

/// Air instance context for managing air instances (traces)
#[allow(dead_code)]
pub struct SetupCtx<F: PrimeField64> {
    setup_repository: SetupRepository<F>,
    pub max_const_tree_size: usize,
    pub max_const_size: usize,
    pub max_prover_buffer_size: usize,
    /// Per-air prover buffer size, largest first. See `SetupRepository::prover_buffer_sizes`.
    pub prover_buffer_sizes: Vec<((usize, usize), usize)>,
    pub max_pinned_proof_size: usize,
    pub max_compact_trace_size: usize,
    pub max_n_bits_ext: usize,
    pub total_const_pols_size: usize,
    /// Included in `total_const_pols_size`, and tracked separately because it stays resident even
    /// in no-const-buf mode: nothing stages custom commits per switch.
    pub total_custom_commits_reserved_words: usize,
    /// Largest packed const-pols set of the repository (elements): the slot size of the const cache.
    pub max_const_pols_size_packed: usize,
    setup_type: ProofType,
}

impl<F: PrimeField64> SetupCtx<F> {
    /// Every custom commit reserves its worst case. Use `new_with_commit_files` when the caller
    /// already knows the packed files: only then can the const buffer reserve their real width.
    pub fn new(
        global_info: &GlobalInfo,
        setup_type: &ProofType,
        verify_constraints: bool,
        gpu: bool,
    ) -> ProofmanResult<Self> {
        Self::new_with_commit_files(global_info, setup_type, verify_constraints, gpu, &HashMap::new(), false)
    }

    pub fn new_with_commit_files(
        global_info: &GlobalInfo,
        setup_type: &ProofType,
        verify_constraints: bool,
        gpu: bool,
        custom_commits_fixed: &HashMap<String, PathBuf>,
        self_contained: bool,
    ) -> ProofmanResult<Self> {
        let setup_repository = SetupRepository::new(
            global_info,
            setup_type,
            verify_constraints,
            gpu,
            custom_commits_fixed,
            self_contained,
        )?;
        let max_const_tree_size = setup_repository.max_const_tree_size;
        let max_const_size = setup_repository.max_const_size;
        let max_prover_buffer_size = setup_repository.max_prover_buffer_size;
        let prover_buffer_sizes = setup_repository.prover_buffer_sizes.clone();
        let max_pinned_proof_size = setup_repository.max_pinned_proof_size;
        let total_const_pols_size = setup_repository.total_const_pols_size;
        let total_custom_commits_reserved_words = setup_repository.total_custom_commits_reserved_words;
        let max_compact_trace_size = setup_repository.max_compact_trace_size;
        let max_n_bits_ext = setup_repository.max_n_bits_ext;
        let max_const_pols_size_packed = setup_repository.max_const_pols_size_packed;
        Ok(SetupCtx {
            setup_repository,
            max_const_tree_size,
            max_const_size,
            max_prover_buffer_size,
            prover_buffer_sizes,
            max_compact_trace_size,
            max_pinned_proof_size,
            max_n_bits_ext,
            total_const_pols_size,
            total_custom_commits_reserved_words,
            max_const_pols_size_packed,
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

    /// `total_signal_no` (circom signalValues length, u64 elements) for every setup
    /// in this repository. Used to size the signalValues buffer pool.
    pub fn total_signal_nos(&self) -> Vec<usize> {
        self.setup_repository.setups.values().filter_map(|s| s.total_signal_no.map(|n| n as usize)).collect()
    }

    pub fn get_global_bin(&self) -> *mut c_void {
        self.setup_repository.global_bin.unwrap()
    }

    pub fn get_global_info_file(&self) -> String {
        self.setup_repository.global_info_file.clone()
    }
}

#[cfg(test)]
mod staging_tests {
    /// Every recursive air's staging width has to come from `recursion_staging_cols`, or one of them
    /// sizes the shared pool at its full width and the others' compaction buys nothing. The final
    /// airs were computed inline and did exactly that: 6 buffers at 1.07 GB instead of 0.075 GB.
    ///
    /// Grep rather than a call, because the defect is a call site that does the arithmetic ITSELF --
    /// it cannot be caught by testing the function everyone else already uses.
    #[test]
    fn no_call_site_sizes_the_recursive_pool_by_hand() {
        let src = include_str!("setup_ctx.rs");
        // Every `let` that sizes a compact trace must go through recursion_staging_cols and never
        // read cm1 directly: the SetupsVadcop max, the per-repository max, and the two finals'
        // tails the SetupsVadcop max is assembled from.
        for decl in ["let max_compact_trace_size", "let vadcop_final_tail", "let vadcop_final_compressed_tail"] {
            for body in src.split(decl).skip(1) {
                // Only real declarations (`let x = ...`), not this test's own string literals.
                if !body.trim_start().starts_with('=') {
                    continue;
                }
                let body = &body[..body.find(';').unwrap_or(body.len())];
                assert!(
                    !body.contains("map_sections_n"),
                    "{decl} reads cm1 directly; it must go through recursion_staging_cols:\n{body}"
                );
                let via_rule = body.contains("recursion_staging_cols");
                let via_tails = body.contains("vadcop_final_tail") || body.contains("max_compact_trace_size");
                assert!(via_rule || via_tails, "{decl} must use the shared rule:\n{body}");
            }
        }
    }
}
