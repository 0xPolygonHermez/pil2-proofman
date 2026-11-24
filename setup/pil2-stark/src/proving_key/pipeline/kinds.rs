//! One pure builder per circuit kind: parameters in, `Vec<Step>` out (no I/O).

use pil2_stark_recurser::plonk2pil::r1cs_types::PlonkOptions;
use proofman_common::hash_family::HashFamily;

use super::steps::{CircomOpt, ConstSource, Step};
use crate::proving_key::recursive::RecursiveTemplate;

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
}

pub struct VadcopFinalCfg {
    pub family: HashFamily,
}

pub struct RecursivefCfg {
    pub family: HashFamily,
}

pub struct RecursiveCfg {
    pub template: RecursiveTemplate,
    pub family: HashFamily,
    /// Reuse another air's pil_info (2nd+ air, recursive2).
    pub reuse: bool,
}

/// Waits for the background witness library builds.
fn witness_wait() -> Step {
    Step::Custom { name: "witness_wait", run: Box::new(|c| c.witness.as_ref().map_or(Ok(()), |w| w.await_all())) }
}

/// The plonk2pil circuit type of a recursion template.
fn plonk_kind(template: RecursiveTemplate) -> &'static str {
    match template {
        RecursiveTemplate::Compressor => "compressor",
        _ => "aggregation",
    }
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
    let opts = PlonkOptions {
        airgroup_name: Some("Compressor".to_string()),
        max_constraint_degree: Some(proofman_common::hash_family::max_constraint_degree_for_blowup(cfg.blowup)),
        hash_id: cfg.family.as_str().to_string(),
        merge_copies: true,
        blake3_lanes: cfg.blake3_lanes,
        min_n_bits: cfg.min_n_bits,
    };
    vec![
        Step::CircomPrologue { verifier: None, circom_template: None, opt: CircomOpt::O2 },
        Step::WitnessLib,
        Step::Plonk2Pil { kind: plonk_kind(cfg.template), opts },
        Step::WriteFixed { source: ConstSource::Plonk },
        Step::CompilePil,
        Step::StarkTail { reuse: false, starkinfo_only: false },
        Step::ConstTree { required: true },
        witness_wait(),
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
                    proofman_common::hash_family::default_aggregation_arity(hash),
                    proofman_common::hash_family::compressed_final_by_default(hash),
                )
            }),
        },
    ]
}

pub fn steps_for_compressed_final(_cfg: &CompressedFinalCfg) -> Vec<Step> {
    todo!()
}
pub fn steps_for_vadcop_final(_cfg: &VadcopFinalCfg) -> Vec<Step> {
    todo!()
}
pub fn steps_for_recursivef(_cfg: &RecursivefCfg) -> Vec<Step> {
    todo!()
}
/// Compressor, recursive1 and recursive2 (`cfg.template`).
pub fn steps_for_recursive(_cfg: &RecursiveCfg) -> Vec<Step> {
    todo!()
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
                Step::Custom { name, .. } => name,
            })
            .collect()
    }

    fn test_cfg(family: HashFamily) -> TestCfg {
        TestCfg { family, template: RecursiveTemplate::Compressor, blowup: 2, blake3_lanes: None, min_n_bits: None }
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

    // ---- Task 2 ----

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
        assert_eq!(opt_of(&s), CircomOpt::O2, "recursive_test.rs:88 uses --O2");
        assert!(verifier_of(&s).is_none(), "the test kind compiles the user's circom as is");
        assert!(s.iter().any(|s| matches!(s, Step::WriteFixed { source: ConstSource::Plonk })));
    }

    #[test]
    #[ignore = "Task 4"]
    fn every_kind_with_a_stark_tail_ends_in_a_const_tree() {
        for (k, s) in [
            ("test", steps_for_test(&test_cfg(HashFamily::Blake3))),
            ("compressed", steps_for_compressed_final(&CompressedFinalCfg { family: HashFamily::Blake3 })),
            ("vadcop_final", steps_for_vadcop_final(&VadcopFinalCfg { family: HashFamily::Blake3 })),
        ] {
            assert!(pos(&s, "const_tree").unwrap() > pos(&s, "stark_tail").unwrap(), "{k}: {:?}", names(&s));
        }
    }

    // ---- Task 3 ----

    #[test]
    #[ignore = "Task 5"]
    fn circom_prologue_args_are_o1_for_recursive_o2_for_compressed() {
        use crate::proving_key::pipeline::steps::circom_args;
        use std::path::Path;
        let rec = steps_for_recursive(&RecursiveCfg {
            template: RecursiveTemplate::Recursive2,
            family: HashFamily::Blake3,
            reuse: false,
        });
        let cmp = steps_for_compressed_final(&CompressedFinalCfg { family: HashFamily::Blake3 });
        let a = circom_args(opt_of(&rec), "goldilocks", &[], Path::new("x.circom"), Path::new("/o"));
        let b = circom_args(opt_of(&cmp), "goldilocks", &[], Path::new("x.circom"), Path::new("/o"));
        assert!(a.contains(&"--O1".to_string()), "recursive.rs:424 is --O1: {a:?}");
        assert!(b.contains(&"--O2".to_string()), "compressed_final.rs:124 is --O2: {b:?}");
    }

    #[test]
    #[ignore = "Task 3"]
    fn compressed_final_uses_the_goldilocks_verifier_generator() {
        let s = steps_for_compressed_final(&CompressedFinalCfg { family: HashFamily::Poseidon2 });
        assert!(matches!(verifier_of(&s), Some(VerifierGen::Gl(_))));
        assert!(pos(&s, "circom_prologue").unwrap() < pos(&s, "plonk2pil").unwrap());
    }

    // ---- Task 4 ----

    #[test]
    #[ignore = "Task 4"]
    fn steps_for_vadcop_final_ends_with_witness_wait() {
        let s = steps_for_vadcop_final(&VadcopFinalCfg { family: HashFamily::Blake3 });
        assert_eq!(names(&s).last(), Some(&"witness_wait"), "{:?}", names(&s));
        assert_eq!(opt_of(&s), CircomOpt::O1, "final_setup.rs:212 is --O1");
    }

    #[test]
    #[ignore = "Task 4"]
    fn steps_for_recursivef_blake3_uses_bn128_verifier() {
        let s = steps_for_recursivef(&RecursivefCfg { family: HashFamily::Blake3 });
        assert!(matches!(verifier_of(&s), Some(VerifierGen::Bn128Blake3(_))), "snark_setup.rs:317");
    }

    #[test]
    #[ignore = "Task 4"]
    fn steps_for_recursivef_poseidon_uses_the_goldilocks_verifier_at_o2() {
        for f in [HashFamily::Poseidon1, HashFamily::Poseidon2] {
            let s = steps_for_recursivef(&RecursivefCfg { family: f });
            assert!(matches!(verifier_of(&s), Some(VerifierGen::Gl(_))), "{f:?}");
            assert_eq!(opt_of(&s), CircomOpt::O2, "snark_setup.rs:500 is --O2");
        }
    }

    #[test]
    #[ignore = "Task 4"]
    fn recursivef_invokes_the_bespoke_snark_backend_as_one_custom_step() {
        let s = steps_for_recursivef(&RecursivefCfg { family: HashFamily::Blake3 });
        let n = names(&s);
        assert_eq!(n.iter().filter(|x| **x == "snark_backend").count(), 1, "{n:?}");
        assert!(pos(&s, "snark_backend").unwrap() > pos(&s, "const_tree").unwrap(), "{n:?}");
    }

    // ---- Task 5 ----

    fn rec(template: RecursiveTemplate, reuse: bool) -> Vec<Step> {
        steps_for_recursive(&RecursiveCfg { template, family: HashFamily::Blake3, reuse })
    }

    #[test]
    #[ignore = "Task 5"]
    fn batch_fit_log_precedes_circom_compile() {
        let s = rec(RecursiveTemplate::Recursive1, false);
        assert!(pos(&s, "batch_fit").unwrap() < pos(&s, "circom_prologue").unwrap(), "{:?}", names(&s));
    }

    #[test]
    #[ignore = "Task 5"]
    fn only_recursive1_has_a_batch_fit_step() {
        assert!(pos(&rec(RecursiveTemplate::Recursive2, false), "batch_fit").is_none());
        assert!(pos(&rec(RecursiveTemplate::Compressor, false), "batch_fit").is_none());
    }

    #[test]
    #[ignore = "Task 5"]
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
    #[ignore = "Task 5"]
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
    #[ignore = "Task 5"]
    fn recursive_kinds_compile_at_o1_with_the_goldilocks_verifier() {
        for t in [RecursiveTemplate::Compressor, RecursiveTemplate::Recursive1, RecursiveTemplate::Recursive2] {
            let s = rec(t, false);
            assert_eq!(opt_of(&s), CircomOpt::O1, "{t:?}");
        }
        assert!(matches!(verifier_of(&rec(RecursiveTemplate::Recursive1, false)), Some(VerifierGen::Gl(_))));
    }
}
