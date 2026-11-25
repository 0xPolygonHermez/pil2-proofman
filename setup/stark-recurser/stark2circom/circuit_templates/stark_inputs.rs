//! The proof inputs a circuit declares (`define_stark_inputs`) and wires into its `StarkVerifier`
//! (`assign_stark_inputs`), from the verifier's own list (`lib_params::proof_inputs`), whose order
//! is the flat proof's: the witness fill copies the proof into the inputs in declaration order.

use anyhow::Result;
use serde_json::Value;

use crate::stark2circom::circom_verifier::lib_params::{proof_inputs, Origin, ProofInput};

/// Options controlling signal generation.
#[derive(Debug, Default, Clone)]
pub struct StarkInputOptions {
    /// Emit `signal input {prefix_}publics[nPublics]` when true.
    pub add_publics: bool,
    /// The final circuit's verifier: `StarkVerifier()`, never suffixed with the airgroup id.
    pub is_final: bool,
    /// Emit `parallel` keyword on the `StarkVerifier` component (for recursive2).
    pub parallel: bool,
    /// With `add_publics`, wire the verifier to the template's own `publics` rather than to
    /// `{prefix_}publics`: every proof a recursive1 folds is of the same air and run, while a
    /// template verifying unrelated proofs (the ZisK recurser) gives each one its own.
    pub shared_publics: bool,
}

fn inputs(stark_info: &Value) -> Result<Vec<ProofInput>> {
    let info = serde_json::from_value(stark_info.clone())?;
    proof_inputs(&info)
}

fn prefix_(prefix: &str) -> String {
    if prefix.is_empty() {
        String::new()
    } else {
        format!("{prefix}_")
    }
}

/// The signal declarations of a StarkVerifier's proof inputs, named `{prefix}_<input>`.
pub fn define_stark_inputs(stark_info: &Value, prefix: &str, opts: &StarkInputOptions) -> Result<String> {
    let p = prefix_(prefix);
    let mut out = String::new();
    for i in inputs(stark_info)? {
        let declared = match i.origin {
            Origin::Proof => true,
            Origin::Publics => opts.add_publics,
            Origin::ProofValues => false,
        };
        if declared {
            out += &format!("    signal input {p}{}{};\n", i.name, i.shape());
        }
        if i.name == "evals" {
            out += "\n";
        }
    }
    Ok(out)
}

/// Options for the `enable` signal injection into the StarkVerifier component.
#[derive(Debug, Default, Clone)]
pub enum EnableInput {
    /// Do not emit an `enable` signal.
    #[default]
    None,
    /// Emit `{component_name}.enable <== 1;`
    One,
    /// Emit `{component_name}.enable <== {n};`
    Value(u64),
    /// Emit `{component_name}.enable <== {expr};`, for an enable driven by a signal.
    Expr(String),
}

/// The StarkVerifier component `component_name` and its inputs wired from `{prefix}_<input>`.
pub fn assign_stark_inputs(
    component_name: &str,
    prefix: &str,
    stark_info: &Value,
    opts: &StarkInputOptions,
    enable: &EnableInput,
) -> Result<String> {
    let p = prefix_(prefix);
    let c = component_name;
    let parallel = if opts.parallel { "parallel " } else { "" };
    let id = stark_info["airgroupId"].as_u64().filter(|_| !opts.is_final).map(|id| id.to_string()).unwrap_or_default();
    let mut out = format!("    component {c} = {parallel}StarkVerifier{id}();\n");
    for i in inputs(stark_info)? {
        let n = &i.name;
        match i.origin {
            Origin::Proof => out += &format!("    {c}.{n} <== {p}{n};\n"),
            Origin::ProofValues => out += &format!("    {c}.{n} <== proofValues;\n"),
            Origin::Publics if opts.add_publics => {
                let source = if opts.shared_publics { "publics".to_string() } else { format!("{p}publics") };
                out += &format!(
                    "    for (var i=0; i< {}; i++) {{\n        {c}.publics[i] <== {source}[i];\n    }}\n",
                    i.dims[0]
                );
            }
            Origin::Publics => {}
        }
    }
    match enable {
        EnableInput::None => {}
        EnableInput::One => out += &format!("    {c}.enable <== 1;\n"),
        EnableInput::Value(n) => out += &format!("    {c}.enable <== {n};\n"),
        EnableInput::Expr(e) => out += &format!("    {c}.enable <== {e};\n"),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stark2circom::circom_verifier::lib_bn128::tests as bn;
    use crate::stark2circom::circom_verifier::lib_gl::tests as gl;
    use crate::stark2circom::circom_verifier::{gen_stark_verifier, gen_stark_verifier_bn128_blake3, Pil2CircomOptions};
    use serde_json::json;

    /// `signal input` name -> shape of a render.
    fn declared(src: &str) -> std::collections::HashMap<String, String> {
        src.lines()
            .filter_map(|l| l.trim().strip_prefix("signal input ")?.strip_suffix(';'))
            .map(|d| {
                let at = d.find('[').unwrap_or(d.len());
                (d[..at].to_string(), d[at..].to_string())
            })
            .collect()
    }

    /// Every input the circuit wires into `sV` is one the verifier declares, of the shape the
    /// circuit declares it with; the publics too.
    fn cross_check(si: &serde_json::Value, vi: &serde_json::Value, blake3: bool) {
        let opts = Pil2CircomOptions {
            skip_main: true,
            hash: if blake3 { "blake3" } else { "Poseidon2" }.into(),
            ..Default::default()
        };
        let root = ["1".to_string(), "0".into(), "0".into(), "0".into()];
        let verifier = if blake3 {
            gen_stark_verifier_bn128_blake3(Some(&root), si, vi, &opts)
        } else {
            gen_stark_verifier(Some(&root), si, vi, &opts)
        }
        .unwrap();
        let body = &verifier[verifier.find("template StarkVerifier").unwrap()..];
        let vdecl = declared(&body[..body[1..].find("\ntemplate ").map_or(body.len(), |e| e + 1)]);
        let o = StarkInputOptions { add_publics: true, is_final: true, ..Default::default() };
        let cdecl = declared(&define_stark_inputs(si, "", &o).unwrap());
        let assign = assign_stark_inputs("sV", "", si, &o, &EnableInput::None).unwrap();
        let mut wired = 0;
        for l in assign.lines() {
            let Some(x) = l.trim().strip_prefix("sV.").and_then(|r| r.split([' ', '[']).next()) else { continue };
            let shape = vdecl.get(x).unwrap_or_else(|| panic!("sV.{x} is no input of the verifier:\n{verifier}"));
            if let Some(c) = cdecl.get(x) {
                assert_eq!(c, shape, "{x}");
            }
            wired += 1;
        }
        assert_eq!(wired, vdecl.len(), "the verifier declares inputs no circuit wires:\n{assign}");
    }

    #[test]
    fn circuits_wire_the_verifiers_own_inputs() {
        let (vi, bvi) = (gl::verifier_info(), bn::verifier_info());
        cross_check(&gl::stark_info(), &vi, false); // FRI, custom commit, last levels
        cross_check(&gl::stir_info(), &vi, false);
        for ll in [0, 2] {
            cross_check(&bn::stark_info(false, ll), &bvi, false);
        }
        let mut b3 = bn::stark_info(true, 2);
        b3["starkStruct"]["verificationHashType"] = json!("GL");
        b3["starkStruct"]["merkleTreeArity"] = json!(2);
        cross_check(&b3, &bvi, true);
    }

    /// A recursive1 folding k proofs of one air shares the template's publics; the ZisK recurser
    /// verifies two unrelated proofs and must read each one's own.
    #[test]
    fn publics_are_shared_only_when_asked() {
        let si = bn::stark_info(false, 0);
        let assign = |shared| {
            let opts = StarkInputOptions { add_publics: true, shared_publics: shared, ..Default::default() };
            assign_stark_inputs("vA", "a_sv", &si, &opts, &EnableInput::None).unwrap()
        };
        assert!(assign(false).contains("vA.publics[i] <== a_sv_publics[i];"), "{}", assign(false));
        assert!(assign(true).contains("vA.publics[i] <== publics[i];"), "{}", assign(true));
    }
}
