use sailfish::TemplateSimple;
use proofman_common::{StarkInfo, GlobalInfo};

#[derive(TemplateSimple)]
#[template(path = "../templates/basic/vadcop/define_vadcop_inputs.circom.stpl")]
pub struct DefineVadcopInputs {
    pub stark_info: StarkInfo,
    pub global_info: GlobalInfo,
    pub prefix_vadcop: String,
    pub is_signal_input: bool,
    pub is_aggregation: bool,
}

pub fn gen_define_vadcop_inputs(
    stark_info: StarkInfo,
    global_info: GlobalInfo,
    prefix_vadcop: String,
    is_signal_input: bool,
    is_aggregation: bool,
) -> Result<String, sailfish::RenderError> {
    let define_vadcop_inputs = DefineVadcopInputs { stark_info, global_info, prefix_vadcop, is_signal_input, is_aggregation };

    define_vadcop_inputs.render_once().map(|output| output.replace("&lt;", "<"))
}
