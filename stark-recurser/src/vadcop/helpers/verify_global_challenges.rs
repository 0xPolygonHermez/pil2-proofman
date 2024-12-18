use sailfish::TemplateSimple;
use proofman_common::GlobalInfo;

use crate::Transcript;

#[derive(TemplateSimple)]
#[template(path = "../templates/vadcop/helpers/templates/verify_global_challenge.circom.stpl")]
pub struct VerifyGlobalChallenges {
    pub global_info: GlobalInfo,
}

pub fn gen_verify_global_challenges(
    global_info: GlobalInfo,
) -> Result<String, sailfish::RenderError> {
    let verify_global_challenges = VerifyGlobalChallenges {
        global_info,
    };

    verify_global_challenges.render_once().map(|output| output.replace("&lt;", "<"))
}