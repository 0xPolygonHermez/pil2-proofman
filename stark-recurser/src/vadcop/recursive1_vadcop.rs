use sailfish::TemplateSimple;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use proofman_common::{StarkInfo, GlobalInfo, ProofType};

use crate::Transcript;

#[derive(TemplateSimple)]
#[template(path = "../templates/vadcop/templates/recursive1.circom.stpl")]
pub struct Recursive1Vadcop {
    pub verifier_filename: String,
    pub stark_info: StarkInfo,
    pub global_info: GlobalInfo,
    pub has_compressor: bool,
    pub prefix: String,
    pub prefix_vadcop: String,
    pub add_publics: bool,
    pub set_enable_input: bool,
    pub is_signal_input: bool,
    pub is_aggregation: bool,
    pub is_final: bool,
    pub component_name: String,
}

pub fn main_gen_recursive1_vadcop(
    proving_key_path: PathBuf,
    airgroup_id: usize,
    air_id: usize,
    has_compressor: bool,
) -> Result<(), io::Error> {
    let global_info = GlobalInfo::new(&proving_key_path);

    let air_name: String = global_info.airs[airgroup_id][air_id].name.clone();
    let name_filename = format!("{}_recursive1.circom", air_name);
    let verifier_filename = format!("{}.verifier.circom", air_name);
    let output_file = Path::new(&global_info.folder_path).parent().unwrap().join("circom").join(name_filename);

    let setup_path = global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Basic);
    let stark_info_path = setup_path.display().to_string() + ".starkinfo.json";
    let stark_info_json = std::fs::read_to_string(&stark_info_path)
        .unwrap_or_else(|_| panic!("Failed to read file {}", &stark_info_path));

    let stark_info = StarkInfo::from_json(&stark_info_json);

    match gen_recursive1_vadcop(verifier_filename, stark_info, global_info, has_compressor) {
        Ok(rendered_output) => {
            let mut output = fs::File::create(output_file)?;
            output.write_all(rendered_output.as_bytes())?;
            Ok(())
        }
        Err(error) => {
            eprintln!("Error rendering template: {}", error);
            Err(io::Error::new(io::ErrorKind::Other, "Template rendering error"))
        }
    }
}

pub fn gen_recursive1_vadcop(
    verifier_filename: String,
    stark_info: StarkInfo,
    global_info: GlobalInfo,
    has_compressor: bool,
) -> Result<String, sailfish::RenderError> {
    let recursive1 = Recursive1Vadcop {
        verifier_filename,
        stark_info,
        global_info,
        has_compressor,
        prefix_vadcop: "sv_".to_string(),
        prefix: String::new(),
        add_publics: !has_compressor,
        set_enable_input: false,
        is_final: false,
        is_signal_input: has_compressor,
        is_aggregation: false,
        component_name: "sV".to_string(),
    };

    recursive1.render_once().map(|output| output.replace("&lt;", "<"))
}
