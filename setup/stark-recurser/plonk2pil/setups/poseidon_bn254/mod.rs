//! PoseidonBN254-family Plonk-to-PIL setup: the final SNARK wrap's AIR, over BN254, for pilfflonk
//! ([`proofman_common::hash_family::BN254_WRAP_FAMILY`]). One setup, [`wrap`], in layout L1 with
//! range checks: `pil/poseidon_bn254/wrap.pil`. Its pilfflonk setup takes the knobs
//! [`wrap::MAX_CONSTRAINT_DEGREE`] (the PIL's, by default) and [`wrap::EXTRA_MULS`].

pub mod constants;
pub mod wrap;

pub struct PilTemplateParams<'a> {
    pub template_file: &'a str,
    pub template_name: &'a str,
    pub namespace_name: &'a str,
    pub n_bits: usize,
    pub n_publics: u32,
    /// The range-check rows, one per `Num2Bytes` use: with none, the AIR has no range check.
    pub n_range_checks: usize,
    pub max_constraint_degree: usize,
}

pub fn gen_pil_str(p: &PilTemplateParams<'_>) -> String {
    format!(
        "require \"{tf}.pil\";\n\n\
         set_std_mode(STD_MODE_ONE_INSTANCE);\n\n\
         set_max_constraint_degree({md});\n\n\
         public publics[{np}];\n\n\
         airgroup {ns}  {{\n    \
         {tn} (N: 2**{nb}, nPublics: {np}, nRangeChecks: {nrc}) alias {ns};\n\
         }}",
        tf = p.template_file,
        tn = p.template_name,
        ns = p.namespace_name,
        nb = p.n_bits,
        np = p.n_publics,
        nrc = p.n_range_checks,
        md = p.max_constraint_degree,
    )
}
