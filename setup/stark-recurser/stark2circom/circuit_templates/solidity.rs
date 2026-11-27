//! The Solidity contracts. They are static too (`sol/`): [`gen_solidity`] and [`gen_iverifier`] fill
//! in their `__TOKEN__`s.

use anyhow::Result;
use serde_json::Value;

use super::get_sha256_inputs::{sha256_defs, ProgramVk};

// ── The static Solidity contracts ────────────────────────────────────────────

const VERIFIER_SOL: &str = include_str!("sol/verifier.sol");
const IVERIFIER_SOL: &str = include_str!("sol/iverifier.sol");

/// `src` with each `token` replaced by its value; every token must be in `src`.
fn fill(src: &str, tokens: &[(&str, &str)]) -> String {
    tokens.iter().fold(src.to_string(), |out, (token, value)| {
        assert!(out.contains(token), "the static contract has no {token}");
        out.replace(token, value)
    })
}

/// The verifier of the final SNARK that the project's Solidity verifier ([`gen_solidity`]) extends:
/// the contract that checks the final circuit's proof, where it is, and the proof its `verifyProof`
/// takes, with the publics hash as its one public.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SnarkVerifier {
    /// snarkjs's `FflonkVerifier`, in `FflonkVerifier.sol` beside the project's: a proof of 24
    /// `bytes32`.
    Fflonk,
    /// snarkjs's `PlonkVerifier`, in `PlonkVerifier.sol` beside the project's: a proof of 24
    /// `uint256`.
    Plonk,
    /// pilfflonk's `PilfflonkVerifier`, the verifier of the final circuit's vkey
    /// (`pilfflonk.verifier.sol`): a proof of `words` `bytes32`, the vkey's calldata
    /// (pilfflonk/docs/formats.md#calldata), whose number depends on the vkey.
    Pilfflonk {
        /// The path the project's verifier imports `pilfflonk.verifier.sol` from.
        source: String,
        /// The words of the `proof` argument of its `verifyProof`.
        words: u64,
    },
}

impl SnarkVerifier {
    /// The contract's name, without `Verifier`.
    fn name(&self) -> &'static str {
        match self {
            SnarkVerifier::Fflonk => "Fflonk",
            SnarkVerifier::Plonk => "Plonk",
            SnarkVerifier::Pilfflonk { .. } => "Pilfflonk",
        }
    }

    /// How the interface names the inherited verifier.
    fn label(&self) -> &'static str {
        match self {
            SnarkVerifier::Fflonk => "FFLONK",
            SnarkVerifier::Plonk => "PLONK",
            SnarkVerifier::Pilfflonk { .. } => "pilfflonk",
        }
    }

    /// The path the project's verifier imports the contract from.
    fn source(&self) -> String {
        match self {
            SnarkVerifier::Fflonk | SnarkVerifier::Plonk => format!("./{}Verifier.sol", self.name()),
            SnarkVerifier::Pilfflonk { source, .. } => source.clone(),
        }
    }

    /// The statement that decodes `proofBytes` into `proofDecoded`, the `proof` of `verifyProof`.
    fn proof_decode(&self) -> String {
        let (word, words) = match self {
            SnarkVerifier::Fflonk => ("bytes32", 24),
            SnarkVerifier::Plonk => ("uint256", 24),
            SnarkVerifier::Pilfflonk { words, .. } => ("bytes32", *words),
        };
        format!("{word}[{words}] memory proofDecoded = abi.decode(proofBytes, ({word}[{words}]));")
    }
}

/// Port of `src/recursion/contracts/verifier.sol.ejs`: the project's verifier, which extends
/// `verifier`. `version` is what the contract's `VERSION()` returns.
pub fn gen_solidity(
    name: &str,
    version: &str,
    root_c: &[u64; 4],
    publics: Option<&Value>,
    verifier: &SnarkVerifier,
) -> Result<String> {
    let camel = capitalise(name);

    let vk = sha256_defs(publics)?.vk;
    let packed = match vk {
        Some(ProgramVk { first: true, .. }) => "programVK, publicValues, rootCVadcopFinal",
        Some(ProgramVk { first: false, .. }) => "publicValues, programVK, rootCVadcopFinal",
        None => "publicValues, rootCVadcopFinal",
    };
    let has_program_vk = vk.is_some();
    let root_c = root_c.map(|r| r.to_string());
    Ok(fill(
        VERIFIER_SOL,
        &[
            ("__SNARK_SOURCE__", &verifier.source()),
            ("__SNARK__", verifier.name()),
            ("__NAME__", &camel),
            ("__VERSION__", version),
            ("__ROOT_C_0__", &root_c[0]),
            ("__ROOT_C_1__", &root_c[1]),
            ("__ROOT_C_2__", &root_c[2]),
            ("__ROOT_C_3__", &root_c[3]),
            ("__PROGRAM_VK_PARAM__", if has_program_vk { "bytes32 programVK,\n        " } else { "" }),
            ("__PROGRAM_VK_ARG__", if has_program_vk { "programVK, " } else { "" }),
            ("__PACKED__", packed),
            ("__PROOF_DECODE__", &verifier.proof_decode()),
        ],
    ))
}

/// Port of `src/recursion/contracts/iverifier.sol.ejs`.
pub fn gen_iverifier(name: &str, publics: Option<&Value>, verifier: &SnarkVerifier) -> Result<String> {
    let camel = capitalise(name);

    let (pin, pin_value, fold_tree, vk_doc, vk_param) = if sha256_defs(publics)?.vk.is_some() {
        (
            "Both keys MUST be constants",
            "values",
            " For an aggregated proof both are the recurser's verkey, and pinning\n    /// them together is what \
             extends that recurser's leaf allow-list over the whole\n    /// fold tree.",
            "/// @param programVK The identity committed by the proof: the ROM root of the\n    /// RISC-V program for \
             a plain proof, the recursion domain for an aggregated one.\n    ",
            "bytes32 programVK,\n        ",
        )
    } else {
        ("The root MUST be a constant", "a value", "", "", "")
    };
    Ok(fill(
        IVERIFIER_SOL,
        &[
            ("__SNARK_LABEL__", verifier.label()),
            ("__NAME__", &camel),
            ("__PIN_VALUE__", pin_value),
            ("__PIN__", pin),
            ("__FOLD_TREE__", fold_tree),
            ("__PROGRAM_VK_DOC__", vk_doc),
            ("__PROGRAM_VK_PARAM__", vk_param),
        ],
    ))
}

fn capitalise(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stark2circom::circuit_templates::mains::tests::out_and_vk;

    /// The project's verifier of a program whose publics end with its verification key, extending
    /// `verifier`.
    fn project_verifier(verifier: &SnarkVerifier) -> String {
        gen_solidity("build", "v0", &[1, 2, 3, u64::MAX], Some(&out_and_vk(false)), verifier).unwrap()
    }

    /// The lines of `verifier`'s project verifier that are not FFLONK's.
    fn lines_unlike_fflonk(verifier: &SnarkVerifier) -> Vec<String> {
        let fflonk = project_verifier(&SnarkVerifier::Fflonk);
        let out = project_verifier(verifier);
        assert_eq!(out.lines().count(), fflonk.lines().count(), "out:\n{out}");
        out.lines()
            .zip(fflonk.lines())
            .filter(|(line, other)| line != other)
            .map(|(line, _)| line.trim().to_string())
            .collect()
    }

    /// snarkjs's verifiers are beside the project's, and take a proof of 24 words.
    #[test]
    fn the_snarkjs_verifiers_are_imported_from_beside_the_contract() {
        let fflonk = project_verifier(&SnarkVerifier::Fflonk);
        for line in [
            "import {FflonkVerifier} from \"./FflonkVerifier.sol\";",
            "contract BuildVerifier is FflonkVerifier, IBuildVerifier {",
            "bytes32[24] memory proofDecoded = abi.decode(proofBytes, (bytes32[24]));",
            "bool success = this.verifyProof(proofDecoded, [publicValuesDigest]);",
        ] {
            assert!(fflonk.contains(line), "{line} missing from:\n{fflonk}");
        }
        assert_eq!(
            lines_unlike_fflonk(&SnarkVerifier::Plonk),
            [
                "import {PlonkVerifier} from \"./PlonkVerifier.sol\";",
                "contract BuildVerifier is PlonkVerifier, IBuildVerifier {",
                "uint256[24] memory proofDecoded = abi.decode(proofBytes, (uint256[24]));",
            ]
        );
    }

    /// pilfflonk's verifier is imported from where the key has it, and takes a proof of the words of
    /// the vkey's calldata; the rest of the contract is FFLONK's.
    #[test]
    fn the_pilfflonk_verifier_is_imported_from_its_source_with_a_proof_of_its_words() {
        let source = "./provingKey/final/pilfflonk/pilfflonk.verifier.sol".to_string();
        assert_eq!(
            lines_unlike_fflonk(&SnarkVerifier::Pilfflonk { source, words: 47 }),
            [
                "import {PilfflonkVerifier} from \"./provingKey/final/pilfflonk/pilfflonk.verifier.sol\";",
                "contract BuildVerifier is PilfflonkVerifier, IBuildVerifier {",
                "bytes32[47] memory proofDecoded = abi.decode(proofBytes, (bytes32[47]));",
            ]
        );
    }
}

#[cfg(test)]
mod solidity_version_tests {
    use super::*;

    fn zisk_publics() -> Value {
        serde_json::json!({
            "nPublics": 68,
            "definitions": [
                {"name": "rom_root", "initialPos": 0, "nValues": 4, "chunks": [1, 64], "verificationKey": true},
                {"name": "inputs", "initialPos": 4, "nValues": 64, "chunks": [1, 64]}
            ],
            "hasProgramVK": true
        })
    }

    #[test]
    fn the_verifier_returns_the_given_version() {
        let sol =
            gen_solidity("zisk", "v9.9.9-test", &[1, 2, 3, 4], Some(&zisk_publics()), &SnarkVerifier::Plonk).unwrap();
        assert!(sol.contains(r#"return "v9.9.9-test";"#), "{sol}");
        assert!(sol.contains("contract ZiskVerifier is PlonkVerifier, IZiskVerifier"));
        assert!(sol.contains("must be a constant of the calling contract"));
    }

    /// The interface carries the key-binding rules a caller needs to verify safely.
    #[test]
    fn the_interface_documents_how_to_pin_the_keys() {
        let isol = gen_iverifier("zisk", Some(&zisk_publics()), &SnarkVerifier::Plonk).unwrap();
        assert!(isol.contains("Both keys MUST be constants of the calling contract"), "{isol}");
        assert!(isol.contains("@param programVK"));
        assert!(isol.contains("@param rootCVadcopFinal"));
        assert!(isol.contains("the inherited raw PLONK verifier"));

        let fflonk = gen_iverifier("zisk", Some(&zisk_publics()), &SnarkVerifier::Fflonk).unwrap();
        assert!(fflonk.contains("the inherited raw FFLONK verifier"));

        // Without a program key there is one key to pin, and no programVK parameter.
        let no_vk = gen_iverifier("zisk", None, &SnarkVerifier::Plonk).unwrap();
        assert!(no_vk.contains("The root MUST be a constant of the calling contract"), "{no_vk}");
        assert!(!no_vk.contains("programVK"));
    }
}
