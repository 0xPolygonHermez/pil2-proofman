use sailfish::TemplateSimple;
use proofman_common::{StarkInfo, GlobalInfo};

#[derive(TemplateSimple)]
#[template(path = "../templates/basic/assign_stark_inputs.circom.stpl")]
pub struct AssignStarkInputs {
    pub stark_info: StarkInfo,
    pub global_info: GlobalInfo,
    pub component_name: String,
    pub prefix: String,
    pub set_enable_input: bool,
    pub add_publics: bool,
    pub is_final: bool,
}

pub fn gen_assign_stark_inputs(
    stark_info: StarkInfo,
    global_info: GlobalInfo,
    component_name: String,
    prefix: String,
    set_enable_input: bool,
    add_publics: bool,
    is_final: bool,
) -> Result<String, sailfish::RenderError> {
    let assign_stark_inputs = AssignStarkInputs {
        stark_info,
        global_info,
        component_name,
        prefix,
        set_enable_input,
        add_publics,
        is_final,
    };

    assign_stark_inputs.render_once().map(|output| output.replace("&lt;", "<"))
}