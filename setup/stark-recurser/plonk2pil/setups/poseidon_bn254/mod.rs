//! PoseidonBN254-family Plonk-to-PIL setup: the final SNARK wrap's AIR, over BN254, for pilfflonk
//! ([`proofman_common::hash_family::BN254_WRAP_FAMILY`]). One setup, [`wrap`], in layout L1 (M45):
//! `pil/poseidon_bn254/wrap.pil`.

pub mod constants;
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
