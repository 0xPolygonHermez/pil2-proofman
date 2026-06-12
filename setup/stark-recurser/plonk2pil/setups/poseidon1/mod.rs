//! Poseidon1-family Plonk-to-PIL setups.

pub mod aggregation;
pub mod compressor;

pub struct PilTemplateParams<'a> {
    pub template_file: &'a str,
    pub template_name: &'a str,
    pub namespace_name: &'a str,
    pub n_bits: usize,
    pub n_publics: u32,
    pub max_constraint_degree: usize,
    pub n_plonk_rows: usize,
    pub n_poseidon1_compression: usize,
    pub n_poseidon1_sponge: usize,
    pub n_cmul_rows: usize,
    pub n_ev_pol4: usize,
    pub n_fft4: usize,
    pub n_tree_selector8: usize,
    pub n_select_val1: usize,
}

pub fn gen_pil_str(p: &PilTemplateParams<'_>) -> String {
    format!(
        "require \"{tf}.pil\";\n\n\
         set_std_mode(STD_MODE_ONE_INSTANCE);\n\n\
         set_max_constraint_degree({md});\n\n\
         public publics[{np}];\n\n\
         airgroup {ns}  {{\n    \
         {tn} (N: 2**{nb}, nPlonkRows: {npl}, nPoseidon1Compression: {np1c}, nPoseidon1Sponge: {np1s}, \
         nCMulRows: {ncm}, nEvPol4: {nev}, nFFT4: {nf4}, nTreeSelector8: {nts}, \
         nSelectVal1: {nsv}) alias {ns};\n\
         }}",
        tf = p.template_file,
        tn = p.template_name,
        ns = p.namespace_name,
        nb = p.n_bits,
        np = p.n_publics,
        md = p.max_constraint_degree,
        npl = p.n_plonk_rows,
        np1c = p.n_poseidon1_compression,
        np1s = p.n_poseidon1_sponge,
        ncm = p.n_cmul_rows,
        nev = p.n_ev_pol4,
        nf4 = p.n_fft4,
        nts = p.n_tree_selector8,
        nsv = p.n_select_val1,
    )
}
