use sailfish::TemplateSimple;
use proofman_common::{StarkInfo, GlobalInfo};

#[derive(TemplateSimple)]
#[template(path = "../templates/basic/vadcop/init_vadcop_inputs.circom.stpl")]
pub struct InitVadcopInputs {
    pub stark_info: StarkInfo,
    pub global_info: GlobalInfo,
    pub prefix: String,
    pub prefix_vadcop: String,
    pub component_name: String,
}

pub fn gen_init_vadcop_inputs(
    stark_info: StarkInfo,
    global_info: GlobalInfo,
    prefix: String,
    prefix_vadcop: String,
) -> Result<String, sailfish::RenderError> {
    let init_vadcop_inputs: InitVadcopInputs = InitVadcopInputs {
        stark_info,
        global_info,
        prefix,
        prefix_vadcop,
        component_name: "sV".to_string(),
    };

    init_vadcop_inputs.render_once().map(|output| output.replace("&lt;", "<"))
}
