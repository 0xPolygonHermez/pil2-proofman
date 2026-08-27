//! Plonk-to-PIL setup generators, split by hash family.
//!
//! Each family lives in its own folder ([`poseidon1`], [`poseidon2`], [`blake3`]) and
//! exposes the same surface (`aggregation`, `compressor`, a `PilTemplateParams`
//! struct and a `gen_pil_str` helper). Dispatch happens in [`super::packers`].

pub mod blake3;
pub mod poseidon1;
pub mod poseidon2;

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::fs;
    use std::path::PathBuf;

    fn pil_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("plonk2pil/pil")
    }

    /// Names the `airtemplate <name> (...)` header of `file` declares, in order.
    fn airtemplate_params(file: &str, template: &str) -> Vec<String> {
        let src = fs::read_to_string(pil_dir().join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
        let head = src
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("airtemplate {template} ")))
            .unwrap_or_else(|| panic!("{file} has no `airtemplate {template}`"));
        let args = &head[head.find('(').unwrap() + 1..head.rfind(')').unwrap()];
        args.split(',').map(|a| a.trim().rsplit(' ').next().unwrap().to_string()).filter(|a| !a.is_empty()).collect()
    }

    /// Named arguments of the `Template (a: 1, b: 2, ...)` call inside a generated PIL string.
    fn generated_call_args(pil: &str) -> Vec<String> {
        let call = pil.rsplit_once('(').expect("generated PIL has no call").1;
        let args = call.split_once(')').expect("unterminated call").0;
        args.split(',').filter_map(|a| a.split_once(':').map(|(name, _)| name.trim().to_string())).collect()
    }

    /// Every named argument the generator emits must be a parameter the airtemplate declares.
    ///
    /// This is not hypothetical: the blake3 aggregator draft referenced an `nPoseidonRows` that no
    /// signature declared, so it could never have compiled, and renaming `nSelectVal1` to
    /// `nSelectValArity4` had to move the .pil parameter and the generated text in lockstep. Both
    /// classes of drift are invisible until `compile-pil` runs on a real r1cs.
    fn assert_no_drift(file: &str, template: &str, pil: &str) {
        let declared: HashSet<String> = airtemplate_params(file, template).into_iter().collect();
        let emitted = generated_call_args(pil);
        assert!(!emitted.is_empty(), "{file}: parsed no named arguments out of the generated call");
        for arg in &emitted {
            assert!(
                declared.contains(arg),
                "{file}: generator emits `{arg}:` but `airtemplate {template}` declares no such \
                 parameter (declared: {declared:?})"
            );
        }
    }

    #[test]
    fn poseidon1_generated_call_matches_its_airtemplates() {
        for (file, template) in [("poseidon1/aggregator.pil", "Aggregator"), ("poseidon1/compressor.pil", "Compressor")]
        {
            let p = super::poseidon1::PilTemplateParams {
                template_file: file.trim_end_matches(".pil"),
                template_name: template,
                namespace_name: "Recursion",
                n_bits: 19,
                n_publics: 0,
                max_constraint_degree: 5,
                n_plonk_rows: 0,
                n_poseidon1_compression: 0,
                n_poseidon1_sponge: 0,
                n_cmul_rows: 0,
                n_ev_pol4: 0,
                n_fft4: 0,
                n_tree_selector8: 0,
                n_select_val_arity4: 0,
            };
            assert_no_drift(file, template, &super::poseidon1::gen_pil_str(&p));
        }
    }

    #[test]
    fn poseidon2_generated_call_matches_its_airtemplates() {
        for (file, template) in [("poseidon2/aggregator.pil", "Aggregator"), ("poseidon2/compressor.pil", "Compressor")]
        {
            let p = super::poseidon2::PilTemplateParams {
                template_file: file.trim_end_matches(".pil"),
                template_name: template,
                namespace_name: "Recursion",
                n_bits: 19,
                n_publics: 0,
                max_constraint_degree: 5,
                n_plonk_rows: 0,
                n_poseidon_compressor: 0,
                n_poseidon_sponge: 0,
                n_cmul_rows: 0,
                n_ev_pol4: 0,
                n_fft4: 0,
                n_tree_selector4: 0,
                n_select_val_arity4: 0,
            };
            assert_no_drift(file, template, &super::poseidon2::gen_pil_str(&p));
        }
    }

    #[test]
    fn blake3_generated_call_matches_its_airtemplate() {
        let file = "blake3/aggregator.pil";
        let p = super::blake3::PilTemplateParams {
            template_file: file.trim_end_matches(".pil"),
            template_name: "Aggregator",
            namespace_name: "Recursion",
            n_bits: 19,
            n_publics: 0,
            max_constraint_degree: 5,
            n_plonk_rows: 0,
            n_cmul_rows: 0,
            n_ev_pol4: 0,
            n_fft4: 0,
            n_tree_selector4: 0,
            n_select_val_arity2: 0,
            n_node_blocks: 0,
            n_chunk_blocks: 0,
            n_parent_blocks: 0,
            lanes: 1,
        };
        assert_no_drift(file, "Aggregator", &super::blake3::gen_pil_str(&p));
    }

    /// The gate a family places follows its Merkle arity, and blake3's is forced to 2. If this ever
    /// flips, the blake3 aggregator's one-row SelectValueArity2 band becomes wrong.
    #[test]
    fn blake3_is_an_arity_2_family_so_its_selectval_is_arity_2() {
        use proofman_common::hash_family::{has_forced_tree_geometry, merkle_tree_arity};
        assert_eq!(merkle_tree_arity("blake3"), 2);
        assert!(has_forced_tree_geometry("blake3"));

        let src = fs::read_to_string(pil_dir().join("blake3/aggregator.pil")).unwrap();
        assert!(src.contains("selectValueArity2("), "blake3 aggregator must place the arity-2 gate");
        assert!(!src.contains("selectValueArity4("), "blake3 is arity 2, not 4");

        // TreeSelector is a different axis: its number is the gate's radix over a binary selection
        // tree, not a hash arity. Radix 4 is 17 signals and fits one row of the 18-wide band; radix
        // 8 is 30 and needs two.
        assert!(src.contains("treeselector4("), "blake3 aggregator should use the one-row radix");
        assert!(!src.contains("treeselector8("), "radix 8 needs two rows of an 18-wide band");
    }
}
