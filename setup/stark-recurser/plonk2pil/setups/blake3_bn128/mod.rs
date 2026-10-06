//! blake3BN128-family Plonk-to-PIL setup: the final SNARK wrap's AIR of a blake3 proving key, over
//! BN128, for pilfflonk ([`proofman_common::hash_family::BLAKE3_BN128_WRAP_FAMILY`]). One setup,
//! [`wrap`], `pil/blake3_bn128/wrap.pil`: 64-row blocks whose columns are opened at their row (the
//! state at the next too), PLONK gates and range checks on the rows the blocks leave free. Its
//! pilfflonk setup takes the knobs [`wrap::MAX_CONSTRAINT_DEGREE`] and [`wrap::EXTRA_MULS`].

pub mod wrap;

pub struct PilTemplateParams<'a> {
    pub template_file: &'a str,
    pub template_name: &'a str,
    pub namespace_name: &'a str,
    pub n_bits: usize,
    pub n_publics: u32,
    pub max_constraint_degree: usize,
}

pub fn gen_pil_str(p: &PilTemplateParams<'_>) -> String {
    format!(
        "require \"{tf}.pil\";\n\n\
         set_std_mode(STD_MODE_ONE_INSTANCE);\n\n\
         set_max_constraint_degree({md});\n\n\
         public publics[{np}];\n\n\
         airgroup {ns}  {{\n    \
         {tn} (N: 2**{nb}, nPublics: {np}) alias {ns};\n\
         }}",
        tf = p.template_file,
        tn = p.template_name,
        ns = p.namespace_name,
        nb = p.n_bits,
        np = p.n_publics,
        md = p.max_constraint_degree,
    )
}
