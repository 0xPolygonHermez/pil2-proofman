pub mod aggregation;
pub mod compressor;
pub mod compressor_light;
pub mod final_vadcop;

/// Parameters for [`gen_pil_str`].
///
/// Mirrors the variables used by the shared `compressor.pil2.ejs` template in
/// `stark-recurser/src/circom2pil/pil/compressor.pil2.ejs`.
pub struct PilTemplateParams<'a> {
    /// Base name of the `.pil` file to `require` (e.g. `"compressor"`).
    pub template_file: &'a str,
    /// PIL airtemplate name (e.g. `"Compressor"`, `"Aggregator"`, `"FinalVadcop"`).
    pub template_name: &'a str,
    /// Airgroup / namespace name (typically the circuit name).
    pub namespace_name: &'a str,
    /// log₂ of number of rows.
    pub n_bits: usize,
    /// Number of public inputs.
    pub n_publics: u32,
    /// `set_max_constraint_degree` value (5 for compressor, 8 for others).
    pub max_constraint_degree: usize,
    pub n_plonk_rows: usize,
    pub n_poseidon_compressor: usize,
    pub n_poseidon_sponge: usize,
    pub n_cmul_rows: usize,
    pub n_ev_pol4: usize,
    pub n_fft4: usize,
    pub n_tree_selector4: usize,
    pub n_select_val1: usize,
}

/// Generate the PIL source string for a plonk2pil circuit.
///
/// This is the Rust equivalent of rendering `compressor.pil2.ejs` — a single
/// shared template used by all four circuit types (compressor, aggregation /
/// recursive1, final_vadcop, compressor_light).
pub fn gen_pil_str(p: &PilTemplateParams<'_>) -> String {
    format!(
        "require \"{tf}.pil\";\n\n\
         set_std_mode(STD_MODE_ONE_INSTANCE);\n\n\
         set_max_constraint_degree({md});\n\n\
         public publics[{np}];\n\n\
         airgroup {ns}  {{\n    \
         {tn} (N: 2**{nb}, nPlonkRows: {npl}, nPoseidonCompressor: {npc}, \
         nPoseidonSponge: {nps}, nCMulRows: {ncm}, nEvPol4: {nev}, \
         nFFT4: {nf4}, nTreeSelector4: {nts}, nSelectVal1: {nsv}) alias {ns};\n\
         }}",
        tf = p.template_file,
        tn = p.template_name,
        ns = p.namespace_name,
        nb = p.n_bits,
        np = p.n_publics,
        md = p.max_constraint_degree,
        npl = p.n_plonk_rows,
        npc = p.n_poseidon_compressor,
        nps = p.n_poseidon_sponge,
        ncm = p.n_cmul_rows,
        nev = p.n_ev_pol4,
        nf4 = p.n_fft4,
        nts = p.n_tree_selector4,
        nsv = p.n_select_val1,
    )
}
