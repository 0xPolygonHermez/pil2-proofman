use sailfish::TemplateSimple;
use proofman_common::{StarkInfo, GlobalInfo};

#[derive(TemplateSimple)]
#[template(path = "../templates/basic/vadcop/agg_vadcop_inputs.circom.stpl")]
pub struct AggVadcopInputs {
    pub stark_info: StarkInfo,
    pub global_info: GlobalInfo,
    pub prefix_a: String,
    pub prefix_b: String,
    pub prefix_vadcop: String,
}

pub fn agg_vadcop_inputs(
    stark_info: StarkInfo,
    global_info: GlobalInfo,
    prefix_a: String,
    prefix_b: String,
    prefix_vadcop: String,
) -> Result<String, sailfish::RenderError> {
    let agg_vadcop_inputs = AggVadcopInputs {
        stark_info,
        global_info,
        prefix_vadcop,
        prefix_a,
        prefix_b,
    };

    agg_vadcop_inputs.render_once().map(|output| output.replace("&lt;", "<"))
}