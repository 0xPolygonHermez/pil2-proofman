//! One pure builder per circuit kind: parameters in, `Vec<Step>` out (no I/O).

use pil2_stark_recurser::plonk2pil::r1cs_types::PlonkOptions;
use proofman_common::hash_family::HashFamily;

use anyhow::{bail, Result};
use pil2_stark_recurser::stark2circom::{CircuitTemplate, StarkVerifierOptions};
use proofman_common::hash_family as hf;

use super::steps::{CircomOpt, ConstSource, Prime, Step, VerifierGen};
use super::PipelineCtx;
use crate::proving_key::recursive::{self, RecursiveTemplate};
use crate::types::stark_struct::{generate_stark_struct, LowDegreeTestKind, StarkSettings, StarkStruct};

/// Basic AIR: `.const` comes from the user's fixed-column supply, no circom.
pub struct BasicCfg {
    pub user_fixed: bool,
}

/// `setup-recursive-test`: a raw user circom through the shared tail.
pub struct TestCfg {
    pub family: HashFamily,
    pub template: RecursiveTemplate,
    pub blowup: usize,
    pub blake3_lanes: Option<usize>,
    pub min_n_bits: Option<usize>,
}

pub struct CompressedFinalCfg {
    pub family: HashFamily,
    /// `None` is the family's (`compressed_final_stir`).
    pub low_degree_test: Option<LowDegreeTestKind>,
}

pub struct VadcopFinalCfg {
    pub family: HashFamily,
}

pub struct RecursivefCfg {
    pub family: HashFamily,
    /// The stark struct's blowup; blake3's constraint degree follows it.
    pub blowup: Option<usize>,
    pub blake3_lanes: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct RecursiveCfg {
    pub template: RecursiveTemplate,
    pub family: HashFamily,
    /// Reuse another air's pil_info (2nd+ airs of an airgroup, recursive1).
    pub reuse: bool,
    pub has_compressor: bool,
    /// More than one airgroup or air: recursive2's verifier takes an `enable` input.
    pub multi_air: bool,
    /// log2 of the rows recursive1/recursive2 run at.
    pub domain_bits: usize,
    pub agg_arity: usize,
    pub airgroup_name: String,
    pub air_name: String,
}

/// The plonk2pil circuit type of a recursion template.
fn plonk_kind(template: RecursiveTemplate) -> &'static str {
    match template {
        RecursiveTemplate::Compressor => "compressor",
        _ => "aggregation",
    }
}

/// plonk2pil's options for an air named `name` (copies always merged).
fn plonk_opts(
    name: String,
    family: HashFamily,
    max_constraint_degree: Option<usize>,
    blake3_lanes: Option<usize>,
    min_n_bits: Option<usize>,
) -> PlonkOptions {
    PlonkOptions {
        airgroup_name: Some(name),
        max_constraint_degree,
        hash_id: family.as_str().to_string(),
        merge_copies: true,
        blake3_lanes,
        min_n_bits,
    }
}

/// plonk2pil's fixed columns and PIL, then (after `settle`) a fresh STARK tail and its const tree.
fn plonk_tail(settle: Option<Step>) -> Vec<Step> {
    let mut s = vec![Step::WriteFixed { source: ConstSource::Plonk }, Step::CompilePil];
    s.extend(settle);
    s.extend([Step::StarkTail { reuse: false, starkinfo_only: false }, Step::ConstTree { required: true }]);
    s
}

/// A size floor pads up, never down: natural rows past it are fatal, with `why`.
pub(crate) fn ensure_within_floor(circuit: &str, natural: usize, floor: usize, why: &str) -> Result<()> {
    if natural > floor {
        bail!("{circuit} compiles to 2^{natural} rows but must be 2^{floor}: {why}");
    }
    Ok(())
}

/// A `CircomPrologue` over Goldilocks.
fn prologue(verifier: Option<VerifierGen>, template: Option<CircuitTemplate>, opt: CircomOpt) -> Step {
    Step::CircomPrologue { verifier, circom_template: template, opt, prime: Prime::Goldilocks }
}

pub fn steps_for_basic(cfg: &BasicCfg) -> Vec<Step> {
    let source = if cfg.user_fixed { ConstSource::UserFixed } else { ConstSource::Pilout };
    vec![
        Step::WriteFixed { source },
        Step::StarkTail { reuse: false, starkinfo_only: false },
        // An air without fixed columns has no `.const` and no verkey.
        Step::ConstTree { required: false },
    ]
}

pub fn steps_for_test(cfg: &TestCfg) -> Vec<Step> {
    let max_deg = Some(hf::max_constraint_degree_for_blowup(cfg.blowup));
    let opts = plonk_opts("Compressor".into(), cfg.family, max_deg, cfg.blake3_lanes, cfg.min_n_bits);
    let mut s = vec![
        prologue(None, None, CircomOpt::O2),
        Step::WitnessLib,
        Step::Plonk2Pil { kind: plonk_kind(cfg.template), opts },
    ];
    s.extend(plonk_tail(None));
    s.extend([
        Step::WitnessWait,
        Step::Custom {
            name: "global_info",
            run: Box::new(|c| {
                let hash = c.hash();
                // No recursive2 stage and no --agg-arity here: the family's documented defaults.
                crate::output::global_info::write_global_info(
                    &*c.pilout()?,
                    "build",
                    super::steps::path_str(&c.build_dir)?,
                    &Default::default(),
                    hash,
                    hf::default_aggregation_arity(hash),
                    hf::compressed_final_by_default(hash),
                )
            }),
        },
    ]);
    s
}

/// The verifier circom of a proof that skips `main`, with these inputs.
fn gl_verifier(family: HashFamily, verkey_input: bool, enable_input: bool, input_challenges: bool) -> VerifierGen {
    VerifierGen::Gl(StarkVerifierOptions {
        verkey_input,
        enable_input,
        input_challenges,
        fri_queries_batch_size: None,
        hash: family.as_str().to_string(),
    })
}

/// The compressed final's stark struct at 2^`n_bits` and whether its proof is a multiproof: no wrap
/// reads this proof, so its geometry is a proof-size knob.
pub fn compressed_final_settings(
    family: HashFamily,
    low_degree_test: Option<LowDegreeTestKind>,
    n_bits: usize,
) -> Result<(StarkStruct, bool)> {
    let hash = family.as_str();
    let ldt = low_degree_test.unwrap_or(if hf::compressed_final_stir(hash) {
        LowDegreeTestKind::Stir
    } else {
        LowDegreeTestKind::Fri
    });
    let multiproof = hf::compressed_final_multiproof(hash);
    let blowup = hf::compressed_final_blowup_factor(hash);
    // A pinned schedule applies only at the air size and blowup it was measured at.
    let here = (n_bits, blowup);
    let pinned = |measured: (usize, usize)| {
        if measured != here {
            tracing::warn!(
                "compressed final at (2^n, blowup) {here:?}: schedule pinned for {measured:?}, deriving one"
            );
        }
        measured == here
    };
    let stir_schedule = (ldt == LowDegreeTestKind::Stir)
        .then(|| hf::compressed_final_stir_schedule(hash))
        .flatten()
        .filter(|(measured, ..)| pinned(*measured));
    let fri_folds = (ldt == LowDegreeTestKind::Fri)
        .then(|| hf::compressed_final_fri_folds(hash))
        .flatten()
        .filter(|(measured, _)| pinned(*measured));
    let settings = StarkSettings {
        initial_blowup_factor: Some(blowup),
        initial_folding_factor: Some(if family.is_blake3() { 4 } else { 3 }),
        grinding_bits: Some(hf::compressed_final_grinding_bits(hash)),
        merkle_tree_arity: (!family.is_blake3()).then_some(2),
        last_level_verification: if multiproof { Some(0) } else { hf::compressed_final_last_level_verification(hash) },
        // final_degree is a log degree here; the helper gives the terminal FRI domain.
        final_degree: Some(hf::compressed_final_fri_terminal_degree(hash) - blowup),
        low_degree_test: Some(ldt),
        folding_factors: stir_schedule.as_ref().map(|(_, folds, _)| folds.clone()).or(fri_folds.map(|(_, f)| f)),
        grinding_bits_queries: stir_schedule.as_ref().map(|(_, _, grinding)| grinding.clone()),
        ..Default::default()
    };
    let ss = generate_stark_struct(&settings, n_bits, hash);
    let (mta, ta) = (ss.merkle_tree_arity, ss.transcript_arity);
    if multiproof && !hf::has_multiproof_codec(hash, mta as u64, ta as u64) {
        bail!("no {hash} multiproof codec at tree arity {mta}, transcript arity {ta}");
    }
    Ok((ss, multiproof))
}

pub fn steps_for_compressed_final(cfg: &CompressedFinalCfg) -> Vec<Step> {
    let (family, ldt) = (cfg.family, cfg.low_degree_test);
    let hash = family.as_str();
    // blake3 follows the blowup, as vadcop_final does; poseidon keeps the packer default.
    let max_deg =
        family.is_blake3().then(|| hf::max_constraint_degree_for_blowup(hf::compressed_final_blowup_factor(hash)));
    let opts =
        plonk_opts("VadcopFinalCompressed".into(), family, max_deg, hf::compressed_final_blake3_lanes(hash), None);
    let settings = Step::Custom {
        name: "compressed_settings",
        run: Box::new(move |c| {
            let (ss, multiproof) = compressed_final_settings(family, ldt, c.plonk()?.n_bits)?;
            c.stark_struct = Some(ss);
            c.key.multiproof = multiproof;
            Ok(())
        }),
    };
    let verifier = gl_verifier(family, false, false, false);
    let mut s = vec![
        prologue(Some(verifier), Some(CircuitTemplate::FinalCompressed), CircomOpt::O2),
        Step::WitnessLib,
        Step::Plonk2Pil { kind: "aggregation", opts },
    ];
    s.extend(plonk_tail(Some(settings)));
    s.extend([Step::WriteVerifierRs, Step::WitnessWait]);
    s
}

pub fn steps_for_vadcop_final(cfg: &VadcopFinalCfg) -> Vec<Step> {
    let hash = cfg.family.as_str();
    // Follows the blowup, not the packer's 5: stage2 packs maxDeg-1 bus terms per im pol.
    let max_deg = Some(hf::max_constraint_degree_for_blowup(hf::final_blowup_factor(hash)));
    // A floor: a smaller pilout pads up to the pinned size the committed verifier encodes.
    let pinned = hf::final_n_bits(hash);
    let opts = plonk_opts("FinalVadcop".into(), cfg.family, max_deg, None, pinned);
    let pin = Step::Custom {
        name: "final_pin",
        run: Box::new(move |c| match pinned {
            Some(p) => ensure_within_floor(
                "vadcop_final",
                c.plonk()?.n_bits_natural,
                p,
                "this pilout aggregates more airgroups than the pinned size holds. Raise \
                 `hash_family::final_n_bits` and regenerate the committed verifier, which encodes the size.",
            ),
            None => Ok(()),
        }),
    };
    // plonk2pil merges the linear constraints itself: --O2 yields the same air, slower.
    let verifier = gl_verifier(cfg.family, true, true, false);
    let mut s = vec![
        prologue(Some(verifier), Some(CircuitTemplate::VadcopFinal), CircomOpt::O1),
        Step::WitnessLib,
        Step::Plonk2Pil { kind: "aggregation", opts },
    ];
    s.extend(plonk_tail(Some(pin)));
    s.extend([Step::WriteVerifierRs, Step::WitnessWait]);
    s
}

/// The STARK verifying one vadcop_final proof. The final SNARK back-end runs after it, outside the list.
pub fn steps_for_recursivef(cfg: &RecursivefCfg) -> Vec<Step> {
    // blake3's follows its blowup, as the compressed final's does.
    let max_deg = cfg.blowup.filter(|_| cfg.family.is_blake3()).map(hf::max_constraint_degree_for_blowup);
    let opts = plonk_opts("Recursivef".into(), cfg.family, max_deg, cfg.blake3_lanes, None);
    let verifier = gl_verifier(cfg.family, true, false, false);
    let mut s = vec![
        prologue(Some(verifier), Some(CircuitTemplate::Recursivef), CircomOpt::O2),
        Step::WitnessLib,
        Step::Plonk2Pil { kind: "aggregation", opts },
    ];
    s.extend(plonk_tail(None));
    s
}

/// Compressor, recursive1 and recursive2 (`cfg.template`).
pub fn steps_for_recursive(cfg: &RecursiveCfg) -> Vec<Step> {
    use RecursiveTemplate as T;
    let (t, family) = (cfg.template, cfg.family);
    let verifier = match t {
        T::Compressor => gl_verifier(family, false, false, true),
        T::Recursive1 => gl_verifier(family, false, false, !cfg.has_compressor),
        T::Recursive2 => gl_verifier(family, true, cfg.multi_air, false),
    };
    // recursive2's air is "Recursive2" (as the JS); its key keeps the recursive1 name.
    let name = match t {
        T::Recursive2 => "Recursive2".to_string(),
        _ => format!("{}_{}_{}", cfg.airgroup_name, cfg.air_name, t.as_str()),
    };
    let max_deg =
        (t == T::Compressor).then(|| hf::max_constraint_degree_for_blowup(recursive::recursive_blowup(t, family)));
    // recursive1 and recursive2 share one starkSetup, so both pad to the domain; the compressor sizes itself.
    // blake3 chooses LANES in its own setup; None takes the air's default of 4.
    let opts = plonk_opts(name, family, max_deg, None, (t != T::Compressor).then_some(cfg.domain_bits));
    // plonk2pil merges the linear constraints itself: --O2 yields the same air, slower.
    let template = match t {
        T::Compressor => CircuitTemplate::Compressor,
        T::Recursive1 => CircuitTemplate::Recursive1,
        T::Recursive2 => CircuitTemplate::Recursive2,
    };
    let prologue = || prologue(Some(verifier.clone()), Some(template), CircomOpt::O1);
    let plonk2pil = || Step::Plonk2Pil { kind: plonk_kind(t), opts: opts.clone() };
    let custom = |name, f: fn(&RecursiveCfg, &mut PipelineCtx) -> Result<()>| {
        let cfg = cfg.clone();
        Step::Custom { name, run: Box::new(move |c| f(&cfg, c)) }
    };

    let mut s = vec![];
    match t {
        T::Recursive1 => s.push(custom("batch_fit", recursive::fit_batch)),
        T::Recursive2 => s.push(custom("arity_fit", recursive::fit_batch)),
        T::Compressor => {}
    }
    s.extend([prologue(), plonk2pil()]);
    if t == T::Recursive1 {
        let (cfg, rerun) = (cfg.clone(), [prologue(), plonk2pil()]);
        s.push(Step::Custom { name: "batch_retry", run: Box::new(move |c| recursive::retry_batch(&cfg, c, &rerun)) });
    }
    s.push(custom("compressor_check", recursive::check_compressor));
    s.extend([
        // After the size checks, so from the winning circom.
        Step::WitnessLib,
        Step::WriteFixed { source: ConstSource::Plonk },
        Step::CompilePil,
        // recursive1 keeps only the const tree's temp starkinfo: its airs share the first one's.
        Step::StarkTail { reuse: cfg.reuse, starkinfo_only: t == T::Recursive1 },
        Step::ConstTree { required: true },
    ]);
    if t == T::Recursive2 {
        s.extend([custom("vks_json", recursive::write_vks_json), Step::WriteVerifierRs]);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proving_key::pipeline::steps::{CircomOpt, ConstSource, VerifierGen};

    fn names(steps: &[Step]) -> Vec<&'static str> {
        steps
            .iter()
            .map(|s| match s {
                Step::CircomPrologue { .. } => "circom_prologue",
                Step::WitnessLib => "witness_lib",
                Step::Plonk2Pil { .. } => "plonk2pil",
                Step::WriteFixed { .. } => "write_fixed",
                Step::CompilePil => "compile_pil",
                Step::StarkTail { .. } => "stark_tail",
                Step::ConstTree { .. } => "const_tree",
                Step::WriteVerifierRs => "write_verifier_rs",
                Step::WitnessWait => "witness_wait",
                Step::Custom { name, .. } => name,
            })
            .collect()
    }

    fn test_cfg(family: HashFamily) -> TestCfg {
        TestCfg { family, template: RecursiveTemplate::Compressor, blowup: 2, blake3_lanes: None, min_n_bits: None }
    }

    fn rf_cfg(family: HashFamily) -> RecursivefCfg {
        RecursivefCfg { family, blowup: Some(2), blake3_lanes: None }
    }

    fn pos(steps: &[Step], n: &str) -> Option<usize> {
        names(steps).iter().position(|x| *x == n)
    }

    fn opt_of(steps: &[Step]) -> CircomOpt {
        steps
            .iter()
            .find_map(|s| if let Step::CircomPrologue { opt, .. } = s { Some(*opt) } else { None })
            .expect("no circom prologue")
    }

    fn verifier_of(steps: &[Step]) -> Option<VerifierGen> {
        steps.iter().find_map(|s| if let Step::CircomPrologue { verifier, .. } = s { verifier.clone() } else { None })
    }

    #[test]
    fn steps_for_basic_has_no_circom_prologue() {
        let s = steps_for_basic(&BasicCfg { user_fixed: true });
        let n = names(&s);
        assert!(!n.contains(&"circom_prologue") && !n.contains(&"plonk2pil"), "{n:?}");
        assert!(!n.contains(&"write_verifier_rs"), "basic omits WriteVerifierRs: {n:?}");
        assert!(pos(&s, "stark_tail").unwrap() < pos(&s, "const_tree").unwrap(), "{n:?}");
    }

    #[test]
    fn steps_for_basic_takes_its_const_from_the_user_fixed_columns() {
        let s = steps_for_basic(&BasicCfg { user_fixed: true });
        assert!(s.iter().any(|s| matches!(s, Step::WriteFixed { source: ConstSource::UserFixed })));
        let s = steps_for_basic(&BasicCfg { user_fixed: false });
        assert!(s.iter().any(|s| matches!(s, Step::WriteFixed { source: ConstSource::Pilout })));
    }

    #[test]
    fn steps_for_test_runs_the_prologue_before_the_shared_tail() {
        let s = steps_for_test(&test_cfg(HashFamily::Blake3));
        let n = names(&s);
        let order = ["circom_prologue", "plonk2pil", "write_fixed", "compile_pil", "stark_tail", "const_tree"];
        let idx: Vec<_> = order.iter().map(|o| pos(&s, o).unwrap_or_else(|| panic!("missing {o}: {n:?}"))).collect();
        assert!(idx.windows(2).all(|w| w[0] < w[1]), "{n:?}");
    }

    #[test]
    fn steps_for_test_compiles_a_raw_circom_at_o2_without_a_generator() {
        let s = steps_for_test(&test_cfg(HashFamily::Poseidon2));
        assert_eq!(opt_of(&s), CircomOpt::O2);
        assert!(verifier_of(&s).is_none(), "the test kind compiles the user's circom as is");
        assert!(s.iter().any(|s| matches!(s, Step::WriteFixed { source: ConstSource::Plonk })));
    }

    #[test]
    fn every_kind_with_a_stark_tail_ends_in_a_const_tree() {
        for (k, s) in [
            ("test", steps_for_test(&test_cfg(HashFamily::Blake3))),
            (
                "compressed",
                steps_for_compressed_final(&CompressedFinalCfg { family: HashFamily::Blake3, low_degree_test: None }),
            ),
            ("vadcop_final", steps_for_vadcop_final(&VadcopFinalCfg { family: HashFamily::Blake3 })),
        ] {
            assert!(pos(&s, "const_tree").unwrap() > pos(&s, "stark_tail").unwrap(), "{k}: {:?}", names(&s));
        }
    }

    #[test]
    fn compressed_final_uses_the_goldilocks_verifier_at_o2() {
        let s =
            steps_for_compressed_final(&CompressedFinalCfg { family: HashFamily::Poseidon2, low_degree_test: None });
        assert!(matches!(verifier_of(&s), Some(VerifierGen::Gl(_))));
        assert_eq!(opt_of(&s), CircomOpt::O2);
        assert!(pos(&s, "circom_prologue").unwrap() < pos(&s, "plonk2pil").unwrap());
    }

    #[test]
    fn steps_for_vadcop_final_ends_with_witness_wait() {
        let s = steps_for_vadcop_final(&VadcopFinalCfg { family: HashFamily::Blake3 });
        assert_eq!(names(&s).last(), Some(&"witness_wait"), "{:?}", names(&s));
        assert_eq!(opt_of(&s), CircomOpt::O1);
    }

    /// bn128_blake3 verifies the blake3 recursivef inside the final SNARK, not here.
    #[test]
    fn steps_for_recursivef_use_the_goldilocks_verifier_at_o2() {
        for f in HashFamily::ALL {
            let s = steps_for_recursivef(&rf_cfg(f));
            assert!(matches!(verifier_of(&s), Some(VerifierGen::Gl(_))), "{f:?}");
            assert_eq!(opt_of(&s), CircomOpt::O2, "{f:?}");
        }
    }

    #[test]
    fn recursivef_ends_at_its_const_tree_the_snark_backend_runs_after() {
        let s = steps_for_recursivef(&rf_cfg(HashFamily::Blake3));
        assert_eq!(names(&s).last(), Some(&"const_tree"), "{:?}", names(&s));
    }

    fn rec(template: RecursiveTemplate, reuse: bool) -> Vec<Step> {
        steps_for_recursive(&RecursiveCfg {
            template,
            family: HashFamily::Blake3,
            reuse,
            has_compressor: false,
            multi_air: true,
            domain_bits: 19,
            agg_arity: 2,
            airgroup_name: "G".into(),
            air_name: "A".into(),
        })
    }

    /// Only recursive1 batches, and its fit is logged before the circom compile.
    #[test]
    fn only_recursive1_fits_a_batch_before_compiling() {
        let s = rec(RecursiveTemplate::Recursive1, false);
        assert!(pos(&s, "batch_fit").unwrap() < pos(&s, "circom_prologue").unwrap(), "{:?}", names(&s));
        assert!(pos(&rec(RecursiveTemplate::Recursive2, false), "batch_fit").is_none());
        assert!(pos(&rec(RecursiveTemplate::Compressor, false), "batch_fit").is_none());
    }

    #[test]
    fn compressor_check_follows_plonk2pil_and_precedes_the_tail() {
        for t in [RecursiveTemplate::Compressor, RecursiveTemplate::Recursive1, RecursiveTemplate::Recursive2] {
            let s = rec(t, false);
            let n = names(&s);
            let p = pos(&s, "plonk2pil").unwrap();
            let c = pos(&s, "compressor_check").unwrap_or_else(|| panic!("{t:?}: {n:?}"));
            assert!(p < c && c < pos(&s, "stark_tail").unwrap(), "{t:?}: {n:?}");
        }
    }

    #[test]
    fn reuse_and_fresh_lists_differ_only_in_the_stark_tail_flag() {
        let fresh = rec(RecursiveTemplate::Recursive2, false);
        let reused = rec(RecursiveTemplate::Recursive2, true);
        assert_eq!(names(&fresh), names(&reused));
        let flag =
            |s: &[Step]| s.iter().find_map(|s| if let Step::StarkTail { reuse, .. } = s { Some(*reuse) } else { None });
        assert_eq!(flag(&fresh), Some(false));
        assert_eq!(flag(&reused), Some(true));
    }

    #[test]
    fn recursive_kinds_compile_at_o1_with_the_goldilocks_verifier() {
        for t in [RecursiveTemplate::Compressor, RecursiveTemplate::Recursive1, RecursiveTemplate::Recursive2] {
            let s = rec(t, false);
            assert_eq!(opt_of(&s), CircomOpt::O1, "{t:?}");
        }
        assert!(matches!(verifier_of(&rec(RecursiveTemplate::Recursive1, false)), Some(VerifierGen::Gl(_))));
    }
}
