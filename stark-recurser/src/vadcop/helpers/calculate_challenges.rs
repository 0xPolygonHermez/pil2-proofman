use sailfish::TemplateSimple;
use proofman_common::StarkInfo;

use crate::Transcript;

#[derive(TemplateSimple)]
#[template(path = "../templates/vadcop/helpers/templates/calculate_hashes.circom.stpl")]
pub struct CalculateChallenges {
    pub stark_info: StarkInfo,
}

pub fn gen_calculate_challenges(
    stark_info: StarkInfo,
) -> Result<String, sailfish::RenderError> {
    let calculate_challenges = CalculateChallenges {
        stark_info,
    };

    calculate_challenges.render_once().map(|output| output.replace("&lt;", "<"))
}