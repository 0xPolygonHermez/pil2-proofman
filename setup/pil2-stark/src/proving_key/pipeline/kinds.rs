//! One pure builder per circuit kind: parameters in, `Vec<Step>` out (no I/O).

use pil2_stark_recurser::plonk2pil::{r1cs::types::PlonkOptions, SetupKind};
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
    /// `None` is STIR.
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
fn plonk_kind(template: RecursiveTemplate) -> SetupKind {
    match template {
        RecursiveTemplate::Compressor => SetupKind::Compressor,
        _ => SetupKind::Aggregation,
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
        hash: family,
        merge_copies: true,
        blake3_lanes,
        min_n_bits,
    }
}

/// plonk2pil's fixed columns, PIL and their check, then (after `settle`) the STARK tail and const tree.
fn plonk_tail(settle: Option<Step>, reuse: bool, starkinfo_only: bool) -> Vec<Step> {
    let mut s = vec![Step::WriteFixed { source: ConstSource::Plonk }, Step::CompilePil, Step::CheckPlonk];
    s.extend(settle);
    s.extend([Step::StarkTail { reuse, starkinfo_only }, Step::ConstTree { required: true }]);
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
    s.extend(plonk_tail(None, false, false));
    s.extend([
        Step::WitnessWait,
        Step::Custom {
            name: "global_info",
            run: Box::new(|c| {
                // No recursive2 stage and no --agg-arity here: the family's documented defaults.
                crate::output::global_info::write_global_info(
                    &*c.pilout()?,
                    "build",
                    super::steps::path_str(&c.build_dir)?,
                    &Default::default(),
                    c.family,
                    hf::default_aggregation_arity(c.family),
                    true,
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
        hash: family,
    })
}

/// The compressed final's stark struct at 2^`n_bits`; its proof is a multiproof (Fibonacci STIR: blake3 249,359 B
/// against 264,935 flat). No wrap reads this proof, so its geometry is a proof-size knob.
pub fn compressed_final_settings(
    family: HashFamily,
    low_degree_test: Option<LowDegreeTestKind>,
    n_bits: usize,
) -> Result<StarkStruct> {
    // STIR: Fibonacci blake3 249,359 B against 274,719 FRI, Poseidon2 201,826 against 254,826.
    let ldt = low_degree_test.unwrap_or(LowDegreeTestKind::Stir);
    let blowup = hf::compressed_final_blowup_factor(family);
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
        .then(|| hf::compressed_final_stir_schedule(family))
        .flatten()
        .filter(|(measured, ..)| pinned(*measured));
    let fri_folds = (ldt == LowDegreeTestKind::Fri)
        .then(|| hf::compressed_final_fri_folds(family))
        .flatten()
        .filter(|(measured, _)| pinned(*measured));
    let settings = StarkSettings {
        initial_blowup_factor: Some(blowup),
        initial_folding_factor: Some(if family.is_blake3() { 4 } else { 3 }),
        grinding_bits: Some(hf::compressed_final_grinding_bits(family)),
        merkle_tree_arity: (!family.is_blake3()).then_some(2),
        // a multiproof shares the tree tops already
        last_level_verification: Some(0),
        // final_degree is a log degree here; the helper gives the terminal FRI domain.
        final_degree: Some(hf::compressed_final_fri_terminal_degree(family) - blowup),
        low_degree_test: Some(ldt),
        folding_factors: stir_schedule.as_ref().map(|(_, folds, _)| folds.clone()).or(fri_folds.map(|(_, f)| f)),
        grinding_bits_queries: stir_schedule.as_ref().map(|(_, _, grinding)| grinding.clone()),
        ..Default::default()
    };
    let ss = generate_stark_struct(&settings, n_bits, family);
    let (mta, ta) = (ss.merkle_tree_arity, ss.transcript_arity);
    if !hf::has_multiproof_codec(family, mta as u64, ta as u64) {
        bail!("no {family} multiproof codec at tree arity {mta}, transcript arity {ta}");
    }
    Ok(ss)
}

pub fn steps_for_compressed_final(cfg: &CompressedFinalCfg) -> Vec<Step> {
    let (family, ldt) = (cfg.family, cfg.low_degree_test);
    // blake3 follows the blowup, as vadcop_final does; poseidon keeps the packer default.
    let max_deg =
        family.is_blake3().then(|| hf::max_constraint_degree_for_blowup(hf::compressed_final_blowup_factor(family)));
    let opts =
        plonk_opts("VadcopFinalCompressed".into(), family, max_deg, hf::compressed_final_blake3_lanes(family), None);
    let settings = Step::Custom {
        name: "compressed_settings",
        run: Box::new(move |c| {
            c.stark_struct = Some(compressed_final_settings(family, ldt, c.plonk()?.n_bits)?);
            c.key.multiproof = true;
            Ok(())
        }),
    };
    let verifier = gl_verifier(family, false, false, false);
    let mut s = vec![
        prologue(Some(verifier), Some(CircuitTemplate::FinalCompressed), CircomOpt::O2),
        Step::WitnessLib,
        Step::Plonk2Pil { kind: SetupKind::Aggregation, opts },
    ];
    s.extend(plonk_tail(Some(settings), false, false));
    s.extend([Step::WriteVerifierRs, Step::WitnessWait]);
    s
}

pub fn steps_for_vadcop_final(cfg: &VadcopFinalCfg) -> Vec<Step> {
    let family = cfg.family;
    // Follows the blowup, not the packer's 5: stage2 packs maxDeg-1 bus terms per im pol.
    let max_deg = Some(hf::max_constraint_degree_for_blowup(hf::final_blowup_factor(family)));
    // A floor: a smaller pilout pads up to the pinned size the committed verifier encodes.
    let pinned = hf::final_n_bits(family);
    let opts = plonk_opts("FinalVadcop".into(), family, max_deg, None, pinned);
    let pin = Step::Custom {
        name: "final_pin",
        run: Box::new(move |c| match pinned {
            Some(p) => ensure_within_floor(
                "vadcop_final",
                c.plonk()?.n_bits_natural,
                p,
                "this pilout aggregates more airgroups than the pinned size holds: raise `hash_family::final_n_bits`.",
            ),
            None => Ok(()),
        }),
    };
    // plonk2pil merges the linear constraints itself: --O2 yields the same air, slower.
    let verifier = gl_verifier(family, true, true, false);
    let mut s = vec![
        prologue(Some(verifier), Some(CircuitTemplate::VadcopFinal), CircomOpt::O1),
        Step::WitnessLib,
        Step::Plonk2Pil { kind: SetupKind::Aggregation, opts },
    ];
    s.extend(plonk_tail(Some(pin), false, false));
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
        Step::Plonk2Pil { kind: SetupKind::Aggregation, opts },
    ];
    s.extend(plonk_tail(None, false, false));
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
    // After the size checks, so from the winning circom.
    s.push(Step::WitnessLib);
    // recursive1 keeps only the const tree's temp starkinfo: its airs share the first one's.
    s.extend(plonk_tail(None, cfg.reuse, t == T::Recursive1));
    if t == T::Recursive2 {
        s.extend([custom("vks_json", recursive::write_vks_json), Step::WriteVerifierRs]);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proving_key::pipeline::steps::{CircomOpt, ConstSource, VerifierGen};

    /// A step's name and the flags it carries (opt, verifier inputs `verkey/enable/challenges`).
    fn describe(s: &Step) -> String {
        let bit = |b: bool| if b { '1' } else { '0' };
        match s {
            Step::CircomPrologue { verifier, opt, .. } => {
                let v = match verifier {
                    Some(VerifierGen::Gl(o)) => {
                        format!("gl{}{}{}", bit(o.verkey_input), bit(o.enable_input), bit(o.input_challenges))
                    }
                    Some(VerifierGen::Bn128Blake3(_)) => "bn128".into(),
                    None => "raw".into(),
                };
                format!("circom:{}:{v}", if *opt == CircomOpt::O1 { "O1" } else { "O2" })
            }
            Step::WitnessLib => "witness_lib".into(),
            Step::Plonk2Pil { .. } => "plonk2pil".into(),
            Step::WriteFixed { source } => match source {
                ConstSource::Plonk => "fixed:plonk",
                ConstSource::Pilout => "fixed:pilout",
                ConstSource::UserFixed => "fixed:user",
            }
            .into(),
            Step::CompilePil => "compile_pil".into(),
            Step::CheckPlonk => "check_plonk".into(),
            Step::StarkTail { reuse, starkinfo_only } => format!("tail:{}{}", bit(*reuse), bit(*starkinfo_only)),
            Step::ConstTree { required } => format!("const_tree:{}", bit(*required)),
            Step::WriteVerifierRs => "verifier_rs".into(),
            Step::WitnessWait => "witness_wait".into(),
            Step::Custom { name, .. } => name.to_string(),
        }
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

    /// Every kind's full step list: order, flags, and CheckPlonk on every plonk path.
    #[test]
    fn each_kind_runs_exactly_its_steps() {
        const PLONK: &str = "witness_lib plonk2pil fixed:plonk compile_pil check_plonk";
        let test = TestCfg {
            family: HashFamily::Blake3,
            template: RecursiveTemplate::Compressor,
            blowup: 2,
            blake3_lanes: None,
            min_n_bits: None,
        };
        let compressed = CompressedFinalCfg { family: HashFamily::Poseidon2, low_degree_test: None };
        let rec_tail = "witness_lib fixed:plonk compile_pil check_plonk";
        let mut cases = vec![
            ("basic user", steps_for_basic(&BasicCfg { user_fixed: true }), "fixed:user tail:00 const_tree:0".into()),
            ("basic", steps_for_basic(&BasicCfg { user_fixed: false }), "fixed:pilout tail:00 const_tree:0".into()),
            (
                "test",
                steps_for_test(&test),
                format!("circom:O2:raw {PLONK} tail:00 const_tree:1 witness_wait global_info"),
            ),
            (
                "compressed",
                steps_for_compressed_final(&compressed),
                format!("circom:O2:gl000 {PLONK} compressed_settings tail:00 const_tree:1 verifier_rs witness_wait"),
            ),
            (
                "vadcop_final",
                steps_for_vadcop_final(&VadcopFinalCfg { family: HashFamily::Blake3 }),
                format!("circom:O1:gl110 {PLONK} final_pin tail:00 const_tree:1 verifier_rs witness_wait"),
            ),
            (
                "compressor",
                rec(RecursiveTemplate::Compressor, false),
                format!("circom:O1:gl001 plonk2pil compressor_check {rec_tail} tail:00 const_tree:1"),
            ),
            (
                "recursive1",
                rec(RecursiveTemplate::Recursive1, false),
                format!(
                    "batch_fit circom:O1:gl001 plonk2pil batch_retry compressor_check {rec_tail} tail:01 const_tree:1"
                ),
            ),
            (
                "recursive2",
                rec(RecursiveTemplate::Recursive2, false),
                format!(
                    "arity_fit circom:O1:gl110 plonk2pil compressor_check {rec_tail} tail:00 const_tree:1 \
                     vks_json verifier_rs"
                ),
            ),
            (
                "recursive2 reused",
                rec(RecursiveTemplate::Recursive2, true),
                format!(
                    "arity_fit circom:O1:gl110 plonk2pil compressor_check {rec_tail} tail:10 const_tree:1 \
                     vks_json verifier_rs"
                ),
            ),
        ];
        // recursivef stops at its const tree: the SNARK back-end runs after the list.
        for f in HashFamily::ALL {
            let s = steps_for_recursivef(&RecursivefCfg { family: f, blowup: Some(2), blake3_lanes: None });
            cases.push(("recursivef", s, format!("circom:O2:gl100 {PLONK} tail:00 const_tree:1")));
        }
        for (kind, steps, want) in cases {
            let got: Vec<String> = steps.iter().map(describe).collect();
            assert_eq!(got.join(" "), want.split_whitespace().collect::<Vec<_>>().join(" "), "{kind}");
        }
    }
}
