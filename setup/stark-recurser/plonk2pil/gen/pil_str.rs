//! The PIL source plonk2pil writes: the layout's airtemplate, called with its sizes.

use super::spec::AirLayout;

/// The PIL instantiating `layout`'s airtemplate as airgroup `name` with `params` after `N` and
/// `nPublics`.
pub fn gen_pil_str(
    layout: &AirLayout,
    name: &str,
    n_bits: usize,
    n_publics: u32,
    max_degree: usize,
    params: &[(&str, usize)],
) -> String {
    let args: String = params.iter().map(|(k, v)| format!(", {k}: {v}")).collect();
    format!(
        "require \"{file}.pil\";\n\nset_std_mode(STD_MODE_ONE_INSTANCE);\n\nset_max_constraint_degree({max_degree});\n\n\
         public publics[{n_publics}];\n\nairgroup {name}  {{\n    {template} (N: 2**{n_bits}, nPublics: {n_publics}{args}) \
         alias {name};\n}}",
        file = layout.pil_file,
        template = layout.pil_template,
    )
}
