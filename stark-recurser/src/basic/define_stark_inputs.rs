use sailfish::TemplateSimple;
use proofman_common::{StarkInfo, GlobalInfo};

#[derive(TemplateSimple)]
#[template(path = "../templates/basic/define_stark_inputs.circom.stpl")]
pub struct DefineStarkInputs {
    pub stark_info: StarkInfo,
    pub global_info: GlobalInfo,
    pub prefix: String,
    pub add_publics: bool,
}

pub fn gen_define_stark_inputs(
    stark_info: StarkInfo,
    global_info: GlobalInfo,
    prefix: String,
    add_publics: bool,
) -> Result<String, sailfish::RenderError> {
    let define_stark_inputs = DefineStarkInputs {
        stark_info,
        global_info,
        prefix,
        add_publics,
    };

    define_stark_inputs.render_once().map(|output| output.replace("&lt;", "<"))
}