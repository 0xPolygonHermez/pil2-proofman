use sailfish::TemplateSimple;
use proofman_common::{StarkInfo, GlobalInfo};

#[derive(TemplateSimple)]
#[template(path = "../templates/basic/vadcop/assign_vadcop_inputs.circom.stpl")]
pub struct AssignVadcopInputs {
    pub stark_info: StarkInfo,
    pub global_info: GlobalInfo,
    pub prefix_vadcop: String,
    pub set_enable_input: bool,
    pub is_aggregation: bool,
    pub component_name: String,
}

pub fn gen_assign_vadcop_inputs(
    stark_info: StarkInfo,
    global_info: GlobalInfo,
    prefix_vadcop: String,
    set_enable_input: bool,
    is_aggregation: bool,
    component_name: String,
) -> Result<String, sailfish::RenderError> {
    let assign_vadcop_inputs = AssignVadcopInputs {
        stark_info,
        global_info,
        prefix_vadcop,
        set_enable_input,
        is_aggregation,
        component_name,
    };

    assign_vadcop_inputs.render_once().map(|output| output.replace("&lt;", "<"))
}