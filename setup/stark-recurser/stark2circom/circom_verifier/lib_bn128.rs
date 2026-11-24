//! The BN128 verifier (a recursivef's, in the final SNARK's circuit) on the static library
//! `helper_circuits/verifier/stark_verifier_bn128.circom`: per key only the data (`lib_params`, as the GL
//! verifier's), the constraints (`ConstraintEval`, from the constraint DAG) and the wiring of
//! `StarkVerifier`, whose proof inputs keep the names the final circuit assigns.

use anyhow::{bail, Context, Result};
use proofman_common::StarkInfo;
use serde_json::Value;

use pil_info::core::dag::format::ConstraintDag;

use super::constraint_roots::constraint_eval_bn128;
use super::lib_params::KeyData;
use super::Pil2CircomOptions;

/// The library templates `StarkVerifier` instantiates by name, with the key's vars as parameters.
const LIB: [&str; 2] = [
    include_str!("helper_circuits/verifier/transcript.circom"),
    include_str!("helper_circuits/verifier/stark_verifier_bn128.circom"),
];

fn call(name: &str) -> String {
    super::lib_gl::call_in(&LIB, name)
}

/// Whether the BN128 verifier of `stark_info` is written with circom custom templates, the ones of
/// `circuits.bn128/custom/`: exactly when its Merkle trees are custom. circom then wants
/// `pragma custom_templates;` in every file on the include path to it, the circuit that includes
/// the verifier as well.
pub(crate) fn uses_custom_templates(stark_info: &Value) -> bool {
    stark_info["starkStruct"]["merkleTreeCustom"].as_bool().unwrap_or(false)
}

/// The BN128 verifier of a BN128 STARK (Poseidon trees, `verificationHashType` BN128).
pub fn gen_stark_verifier_bn128(
    const_root: Option<&[String; 4]>,
    stark_info: &Value,
    verifier_info: &Value,
    opts: &Pil2CircomOptions,
) -> Result<String> {
    let hash_type = stark_info["starkStruct"]["verificationHashType"].as_str().unwrap_or("GL");
    if hash_type != "BN128" {
        bail!("gen_stark_verifier_bn128: expected BN128 hash type, got '{hash_type}'");
    }
    gen(const_root, stark_info, verifier_info, opts, false)
}

/// The BN128 verifier of a Goldilocks STARK hashed with blake3: the recursivef of a blake3 key,
/// which the final SNARK's circuit verifies. Its arithmetic is the BN128 verifier's, its hashing the
/// Goldilocks verifier's: roots of four words, the GL transcript over circuits.gl's blake3 circuits
/// (field parts in `circuits.bn128/b3_field.circom`), binary Merkle paths with their last levels.
pub fn gen_stark_verifier_bn128_blake3(
    const_root: Option<&[String; 4]>,
    stark_info: &Value,
    verifier_info: &Value,
    opts: &Pil2CircomOptions,
) -> Result<String> {
    let ss = &stark_info["starkStruct"];
    if ss["verificationHashType"].as_str().unwrap_or("GL") != "GL" || opts.hash != "blake3" {
        bail!("gen_stark_verifier_bn128_blake3: expected a Goldilocks STARK hashed with blake3");
    }
    if ss["merkleTreeArity"].as_u64() != Some(2) {
        bail!("gen_stark_verifier_bn128_blake3: blake3 trees are binary");
    }
    // Its transcript absorbs none of them, so they would be free after the challenges.
    for map in ["airValuesMap", "proofValuesMap"] {
        if stark_info[map].as_array().is_some_and(|a| !a.is_empty()) {
            bail!("gen_stark_verifier_bn128_blake3: {map} is not supported: the transcript does not absorb it");
        }
    }
    if !opts.skip_main {
        bail!("gen_stark_verifier_bn128_blake3: the verifier has no Main of its own (skip_main)");
    }
    if !uses_custom_templates(stark_info) {
        bail!("gen_stark_verifier_bn128_blake3: the blake3 gates are custom templates (merkleTreeCustom)");
    }
    gen(const_root, stark_info, verifier_info, opts, true)
}

/// The key's Merkle levels below the ones sent in clear, of a tree over 2^bits leaves (geometry.circom's
/// geo_pathLevels): 0 is an Empty fold tree, whose path the Poseidon proof does not carry.
fn path_levels(arity: u64, last_levels: u64, bits: u64) -> u64 {
    let nba = u64::from(arity.trailing_zeros());
    bits.div_ceil(nba).saturating_sub(last_levels)
}

fn gen(
    const_root: Option<&[String; 4]>,
    si: &Value,
    vi: &Value,
    opts: &Pil2CircomOptions,
    blake3: bool,
) -> Result<String> {
    let info: StarkInfo = serde_json::from_value(si.clone()).context("starkinfo")?;
    let ss = &info.stark_struct;
    let custom = ss.merkle_tree_custom;
    let fri = ss.low_degree_test.fri().context("the BN128 verifier checks FRI, and this STARK opens with STIR")?;
    if info.ev_map.is_empty() {
        bail!("evMap has no evaluations, so the DEEP polynomial is undefined");
    }
    // The Tera verifier did not support them either (its BN128 render did not compile).
    if !info.custom_commits.is_empty() {
        bail!("the BN128 verifier does not support custom commits");
    }
    // Only the GL transcript binds airgroupvalues.
    if info.airgroupvalues_map.as_ref().is_some_and(|m| !m.is_empty()) {
        bail!("BN128 verifier does not support airgroupvalues (they are not bound to the transcript)");
    }
    let ll = ss.last_level_verification;
    if !blake3 && custom && ll > 0 {
        bail!("lastLevelVerification > 0 with merkleTreeCustom: circuits.bn128/custom/merklehash.circom has no last-level templates");
    }
    let key = KeyData::new(&info, false, None, None)?;
    let steps = &fri.log_domain_sizes;
    let (q_stage, nf) = (info.n_stages + 1, steps.len() - 1);
    let pow = fri.grinding_bits_queries > 0;
    // Root, Merkle node and sibling shapes: blake3 words of four, Poseidon single elements.
    let (rd, sw, ll_name, lld) =
        if blake3 { ("[4]", "sibW", "last_mt_levels", "[4]") } else { ("", "arity", "last_levels", "") };
    let empty = |f: usize| !blake3 && ll > 0 && path_levels(ss.merkle_tree_arity, ll, steps[f]) == 0;

    let mut out = String::from("pragma circom 2.1.0;\n");
    if custom {
        out += "pragma custom_templates;\n";
    }
    let c = if custom { "custom/" } else { "" };
    out += &format!(
        "\ninclude \"{c}evalpol.circom\";\ninclude \"{c}fft.circom\";\ninclude \"{c}gl.circom\";\ninclude \"{c}lessthangl.circom\";\n"
    );
    out += "include \"bitifyT.circom\";\ninclude \"treeselector.circom\";\n";
    out += &if blake3 {
        "include \"../circuits.gl/hash/blake3/merklehash.circom\";\ninclude \"../circuits.gl/hash/blake3/pow.circom\";\ninclude \"verifier/merkle.circom\";\ninclude \"verifier/transcript_blake3.circom\";\n".to_string()
    } else {
        format!("include \"{c}merklehash.circom\";\ninclude \"pow.circom\";\ninclude \"{c}sponge.circom\";\ninclude \"verifier/transcript_sponge.circom\";\n")
    };
    out += "include \"verifier/stark_verifier_bn128.circom\";\n\n";
    if custom && !blake3 {
        // The custom sponge is the tree's CustomPoseidon(arity).
        out += &format!(
            "function sponge_rate() {{ return {}; }}\nfunction sponge_capacity() {{ return 1; }}\n\n",
            ss.merkle_tree_arity
        );
    }
    let dag = vi.get("constraintDag").context("verifierinfo has no constraintDag: regenerate the setup")?;
    let dag: ConstraintDag = serde_json::from_value(dag.clone()).context("constraintDag")?;
    out += &constraint_eval_bn128(&dag, &super::lower_ctx(&info)?)?;

    // ---- StarkVerifier: the key's vars, the proof inputs, the library ----
    let mut d = key.vars.clone();
    d += "    var L0 = geo_levels0(arity, lastLevels, STEPS);\n";
    d += "    var sibW = geo_sibW(arity);\n    var lastSize = geo_lastSize(arity, lastLevels);\n";
    d += "    var q = 0xFFFFFFFFFFFFFFFF;\n\n";
    if blake3 {
        d += "    // circuits.bn128's b3_field and selectval, not circuits.gl's: their pick is the include path's.\n    assert(b3_field_is_gl() == 0);\n";
    }
    if info.n_publics > 0 {
        d += "    signal input publics[nPublics];\n";
    }
    for st in 1..=q_stage {
        d += &format!("    signal input root{st}{rd};\n");
    }
    d += &if opts.verkey_input {
        format!("    signal input rootC{rd};\n")
    } else if blake3 {
        format!("    signal rootC[4] <== [{}];\n", const_root.map(|r| r.join(",")).unwrap_or_else(|| "0,0,0,0".into()))
    } else {
        // A BN128 verkey is one element, in slot 0.
        format!("    signal rootC <== {};\n", const_root.map(|r| r[0].clone()).unwrap_or_else(|| "0".into()))
    };
    d += "    signal input evals[nEvals][3];\n";
    for t in &key.trees {
        d += &format!("    signal input s0_vals{}[nQueries][{}];\n", t.suffix, t.width);
    }
    for t in &key.trees {
        d += &format!("    signal input s0_siblings{}[nQueries][L0][{sw}];\n", t.suffix);
    }
    if ll > 0 {
        for t in &key.trees {
            d += &format!("    signal input s0_{ll_name}{}[lastSize]{lld};\n", t.suffix);
        }
    }
    for f in 1..=nf {
        d += &format!("    signal input s{f}_root{rd};\n");
    }
    for f in 1..=nf {
        d += &format!("    signal input s{f}_vals[nQueries][3 * geo_foldExp(STEPS, {f})];\n");
        if !empty(f) {
            d += &format!(
                "    signal input s{f}_siblings[nQueries][geo_foldLevels(arity, lastLevels, STEPS, {f})][{sw}];\n"
            );
        }
        if ll > 0 {
            d += &format!("    signal input s{f}_{ll_name}[lastSize]{lld};\n");
        }
    }
    d += "    signal input finalPol[geo_finalSize(nSteps, STEPS)][3];\n";
    if pow {
        d += "    signal input nonce;\n";
    }
    d += "\n    signal {binary} enabled;\n";
    d += if opts.enable_input {
        "    signal input enable;\n    enable * (enable - 1) === 0;\n    enabled <== enable;\n"
    } else {
        "    enabled <== 1;\n"
    };

    let mut w = String::new();
    if info.n_publics > 0 {
        w += "\n    signal {maxNum} publicsGL[nPublics] <== GLVals(nPublics)(publics);\n";
    }
    w += "    signal {maxNum} evalsGL[nEvals][3] <== GLVals3(nEvals)(evals);\n";
    w += "    signal {maxNum} finalPolGL[geo_finalSize(nSteps, STEPS)][3] <== GLVals3(geo_finalSize(nSteps, STEPS))(finalPol);\n";

    // The GL verifier's transcript over this key's engine; its challenges reduced Goldilocks values.
    let root = |r: String| if blake3 { r } else { format!("[{r}]") };
    let pad3 = "[[0, 0, 0]]";
    w += &format!("\n    component transcript = {};\n", call("StarkVerifierTranscript"));
    w += "    transcript.globalChallenge <== [0, 0, 0];\n";
    w += &format!("    transcript.publics <== {};\n", if info.n_publics > 0 { "publicsGL" } else { "[0]" });
    w += &format!("    transcript.rootC <== {};\n", root("rootC".into()));
    for st in 1..=q_stage {
        w += &format!("    transcript.stageRoots[{}] <== {};\n", st - 1, root(format!("root{st}")));
    }
    w += &format!("    transcript.airvalues <== {pad3};\n    transcript.airgroupvalues <== {pad3};\n");
    w += "    transcript.evals <== evalsGL;\n";
    for f in 1..=nf {
        w += &format!("    transcript.foldRoots[{}] <== {};\n", f - 1, root(format!("s{f}_root")));
    }
    if nf == 0 {
        w += &format!("    transcript.foldRoots <== [{}];\n", if blake3 { "[0, 0, 0, 0]" } else { "[0]" });
    }
    w += &format!("    transcript.betas <== {pad3};\n    transcript.hints <== {pad3};\n");
    w += &format!(
        "    transcript.finalPol <== finalPolGL;\n    transcript.nonce <== {};\n    transcript.enable <== enabled;\n",
        if pow { "nonce" } else { "0" }
    );
    w += "    signal {maxNum} challenges[nChallenges][3];\n    challenges.maxNum = q;\n    challenges <== transcript.challenges;\n";
    w += "    signal {maxNum} rFold[geo_len1(nSteps - 1)][3];\n    rFold.maxNum = q;\n";
    w += if nf > 0 {
        "    for (var f = 0; f < nSteps - 1; f++) { rFold[f] <== transcript.ldt[f]; }\n"
    } else {
        "    rFold[0] <== [0, 0, 0];\n"
    };

    // The constraints at xi against the quotient; challenges end with vc, xi, vf1, vf2.
    w += "\n    component zh = ZhBN128(nBits);\n    zh.challengeOut <== challenges[nChallenges - 3];\n";
    w += "    component ce = ConstraintEval();\n    ce.challenges <== challenges;\n    ce.evals <== evalsGL;\n    ce.Zh <== zh.Zh;\n";
    if info.n_publics > 0 {
        w += "    ce.publics <== publicsGL;\n";
    }
    w += &format!("    {}(zh.xN, ce.evalP, evalsGL, enabled);\n", call("QuotientCheckBN128"));

    w += &format!("\n    component deepConsts = {};\n", call("DeepConstantsBN128"));
    w += "    deepConsts.vf[0] <== challenges[nChallenges - 2];\n    deepConsts.vf[1] <== challenges[nChallenges - 1];\n";
    w += "    deepConsts.evals <== evalsGL;\n";

    // Per query; the levels sent in clear once.
    w += "\n    component query[nQueries];\n    for (var i = 0; i < nQueries; i++) {\n        query[i] = VerifySingleQuery();\n";
    w += "        query[i].queriesL0 <== transcript.queriesL0[i];\n        query[i].challengeOut <== challenges[nChallenges - 3];\n";
    w += "        query[i].friConsts <== deepConsts.friConsts;\n        query[i].rFold <== rFold;\n        query[i].finalPol <== finalPolGL;\n";
    w += "        query[i].enable <== enabled;\n";
    for t in &key.trees {
        let s = &t.suffix;
        w += &format!(
            "        query[i].s0_vals{s} <== s0_vals{s}[i];\n        query[i].s0_siblings{s} <== s0_siblings{s}[i];\n"
        );
        w += &if ll > 0 {
            format!("        query[i].s0_{ll_name}{s} <== s0_{ll_name}{s};\n")
        } else {
            format!("        query[i].root{s} <== {};\n", t.root)
        };
    }
    for f in 1..=nf {
        w += &format!("        query[i].s{f}_vals <== s{f}_vals[i];\n");
        if !empty(f) {
            w += &format!("        query[i].s{f}_siblings <== s{f}_siblings[i];\n");
        }
        w += &if ll > 0 {
            format!("        query[i].s{f}_{ll_name} <== s{f}_{ll_name};\n")
        } else {
            format!("        query[i].s{f}_root <== s{f}_root;\n")
        };
    }
    w += "    }\n";
    if ll > 0 {
        for t in &key.trees {
            w += &format!(
                "    VerifyMerkleRoot(lastLevels, arity, 1 << STEPS[0])(s0_{ll_name}{}, {}, enabled);\n",
                t.suffix, t.root
            );
        }
        for f in 1..=nf {
            w += &format!(
                "    VerifyMerkleRoot(lastLevels, arity, 1 << STEPS[{f}])(s{f}_{ll_name}, s{f}_root, enabled);\n"
            );
        }
    }
    w += "    VerifyFinalPolBN128(STEPS[nSteps - 1], geo_maxDegSize(nBits, nBitsExt, nSteps, STEPS))(finalPolGL, enabled);\n";

    out += &format!("\n// Verifies one STARK proof on the BN128 library; the inputs keep the final circuit's names.\ntemplate StarkVerifier() {{\n{d}{w}}}\n");
    out += &single_query(&info, &key, blake3, &empty);
    Ok(out)
}

/// `VerifySingleQuery`: one query's L_0 rows and fold-tree cosets, reduced and opened against their
/// trees, then DEEP and every folding step (library).
fn single_query(info: &StarkInfo, key: &KeyData, blake3: bool, empty: &dyn Fn(usize) -> bool) -> String {
    let ll = info.stark_struct.last_level_verification;
    let nf = info.stark_struct.low_degree_test.fri().map_or(0, |f| f.log_domain_sizes.len() - 1);
    let (sw, ll_name, lld) = if blake3 { ("sibW", "last_mt_levels", "[4]") } else { ("arity", "last_levels", "") };
    let mut d = key.vars.clone();
    d += "    var L0 = geo_levels0(arity, lastLevels, STEPS);\n";
    d += "    var sibW = geo_sibW(arity);\n    var lastSize = geo_lastSize(arity, lastLevels);\n";
    d += "    var q = 0xFFFFFFFFFFFFFFFF;\n\n";
    d += "    signal input {binary} queriesL0[STEPS[0]];\n    signal input {maxNum} challengeOut[3];\n";
    d += "    signal input {maxNum} friConsts[2 + nPols + 2 * nGroups][3];\n    signal input {maxNum} rFold[geo_len1(nSteps - 1)][3];\n";
    d += "    signal input {maxNum} finalPol[geo_finalSize(nSteps, STEPS)][3];\n    signal input {binary} enable;\n";
    let mut w = String::from("\n    // The L_0 rows, reduced, flat in tree order (S0).\n    signal {maxNum} leaves[S0[nTrees]];\n    leaves.maxNum = q;\n");
    // Every row reduced before any tree is opened, as the Tera verifier ordered them.
    let mut merkle = String::new();
    if blake3 {
        merkle += "    component keys0 = QueryKeys(arity, L0 + lastLevels, STEPS[0], STEPS[0]);\n    keys0.queries <== queriesL0;\n";
    }
    for (k, t) in key.trees.iter().enumerate() {
        let (s, width) = (&t.suffix, t.width);
        d += &format!("    signal input s0_vals{s}[{width}];\n    signal input s0_siblings{s}[L0][{sw}];\n");
        let last = if ll > 0 {
            d += &format!("    signal input s0_{ll_name}{s}[lastSize]{lld};\n");
            format!("s0_{ll_name}{s}")
        } else {
            d += &format!("    signal input root{s}{};\n", if blake3 { "[4]" } else { "" });
            format!("root{s}")
        };
        w += &format!(
            "    for (var i = 0; i < {width}; i++) {{ leaves[S0[{k}] + i] <== LessThan64Bits()(s0_vals{s}[i]); }}\n"
        );
        if blake3 {
            let (last, root) =
                if ll > 0 { (last, "[0, 0, 0, 0]".to_string()) } else { ("[[0, 0, 0, 0]]".into(), last) };
            merkle +=
                &format!("    component tree{k} = VerifyTreeOpening(1, {width}, arity, lastLevels, L0, nBitsExt);\n");
            merkle += &format!("    tree{k}.keys <== keys0.keys;\n    for (var i = 0; i < {width}; i++) {{ tree{k}.values[i][0] <== leaves[S0[{k}] + i]; }}\n");
            merkle += &format!("    tree{k}.siblings <== s0_siblings{s};\n    tree{k}.last <== {last};\n    tree{k}.root <== {root};\n    tree{k}.enable <== enable;\n");
        } else {
            merkle += &format!("    signal v{k}[{width}][1];\n    for (var i = 0; i < {width}; i++) {{ v{k}[i][0] <== leaves[S0[{k}] + i]; }}\n");
            merkle += &if ll > 0 {
                format!("    VerifyMerkleHashUntilLevel(1, {width}, 1 << STEPS[0], arity, lastLevels)(v{k}, s0_siblings{s}, queriesL0, {last}, enable);\n")
            } else {
                format!("    VerifyMerkleHash(1, {width}, 1 << STEPS[0], arity)(v{k}, s0_siblings{s}, queriesL0, {last}, enable);\n")
            };
        }
    }
    w += &merkle;

    // Fold tree f's opened coset, reduced; its Merkle path; f_{f-1} folded at the query against it.
    for f in 1..=nf {
        let ml = format!("geo_foldLevels(arity, lastLevels, STEPS, {f})");
        d += &format!("    signal input s{f}_vals[3 * geo_foldExp(STEPS, {f})];\n");
        if !empty(f) {
            d += &format!("    signal input s{f}_siblings[{ml}][{sw}];\n");
        }
        let last = if ll > 0 {
            d += &format!("    signal input s{f}_{ll_name}[lastSize]{lld};\n");
            format!("s{f}_{ll_name}")
        } else {
            d += &format!("    signal input s{f}_root{};\n", if blake3 { "[4]" } else { "" });
            format!("s{f}_root")
        };
        w +=
            &format!("\n    signal {{maxNum}} s{f}_vals_p[geo_foldExp(STEPS, {f})][3];\n    s{f}_vals_p.maxNum = q;\n");
        w += &format!("    for (var c = 0; c < geo_foldExp(STEPS, {f}); c++) {{ for (var e = 0; e < 3; e++) {{ s{f}_vals_p[c][e] <== LessThan64Bits()(s{f}_vals[c * 3 + e]); }} }}\n");
        if blake3 {
            let (last, root) =
                if ll > 0 { (last, "[0, 0, 0, 0]".to_string()) } else { ("[[0, 0, 0, 0]]".into(), last) };
            w += &format!("    component open{f} = VerifyTreeOpening(3, geo_foldExp(STEPS, {f}), arity, lastLevels, {ml}, STEPS[{f}]);\n");
            w += &format!(
                "    open{f}.keys <== QueryKeys(arity, {ml} + lastLevels, STEPS[{f}], STEPS[0])(queriesL0);\n"
            );
            w += &format!("    open{f}.values <== s{f}_vals_p;\n    open{f}.siblings <== s{f}_siblings;\n");
            w += &format!(
                "    open{f}.last <== {last};\n    open{f}.root <== {root};\n    open{f}.enable <== enable;\n"
            );
        } else {
            w += &format!("    signal {{binary}} s{f}_keys[STEPS[{f}]];\n    for (var i = 0; i < STEPS[{f}]; i++) {{ s{f}_keys[i] <== queriesL0[i]; }}\n");
            let args = format!("3, geo_foldExp(STEPS, {f}), 1 << STEPS[{f}], arity");
            w += &if empty(f) {
                format!("    VerifyMerkleHashUntilLevelEmpty({args}, lastLevels)(s{f}_vals_p, s{f}_keys, {last}, enable);\n")
            } else if ll > 0 {
                format!("    VerifyMerkleHashUntilLevel({args}, lastLevels)(s{f}_vals_p, s{f}_siblings, s{f}_keys, {last}, enable);\n")
            } else {
                format!("    VerifyMerkleHash({args})(s{f}_vals_p, s{f}_siblings, s{f}_keys, {last}, enable);\n")
            };
        }
    }

    w += &format!(
        "\n    DeepQueryBN128(nGroups, nPols, nColGroups, nSums, STEPS[0], {}, S0[nTrees], nBits, OPEN, SUMS, COLS)(queriesL0, challengeOut, friConsts, leaves, {}, enable);\n",
        if nf > 0 { "STEPS[1]" } else { "0" },
        if nf > 0 { "s1_vals_p" } else { "finalPol" }
    );
    for f in 1..=nf {
        let next = if f < nf { format!("s{}_vals_p", f + 1) } else { "finalPol".into() };
        let next_bits = if f < nf { format!("STEPS[{}]", f + 1) } else { "0".into() };
        w += &format!(
            "    VerifyFRIBN128(STEPS[{}], STEPS[{f}], {next_bits}, nBitsExt, STEPS[0])(queriesL0, rFold[{}], s{f}_vals_p, {next}, enable);\n",
            f - 1,
            f - 1
        );
    }
    format!("\n// One query: its L_0 rows and fold-tree cosets against their trees, DEEP, every folding step.\ntemplate parallel VerifySingleQuery() {{\n{d}{w}}}\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn stark_info(custom: bool, ll: u64) -> Value {
        json!({
            "starkStruct": {
                "verificationHashType": "BN128", "numQueries": 4, "nBits": 6, "nBitsExt": 10, "grindingBitsQueries": 0,
                "merkleTreeArity": 4, "merkleTreeCustom": custom, "lastLevelVerification": ll, "hashCommits": false,
                "transcriptArity": 4, "foldingFactors": [4], "logDegrees": [6, 2], "logDomainSizes": [10, 6]
            },
            "nStages": 2, "qDim": 3, "cExpId": 0, "security": {"regime": "test"}, "nPublics": 2, "nConstants": 2,
            "evMap": [{"type": "const", "id": 0, "prime": 0, "openingPos": 0}, {"type": "cm", "id": 0, "prime": 0, "openingPos": 0}],
            "cmPolsMap": [{"name": "q", "stage": 3, "stagePos": 0, "stageId": 0, "dim": 3}],
            "customCommits": [], "customCommitsMap": [],
            "challengesMap": [
                {"name": "std_vc", "stage": 3, "stageId": 0, "dim": 3}, {"name": "xi", "stage": 4, "stageId": 0, "dim": 3},
                {"name": "vf1", "stage": 5, "stageId": 0, "dim": 3}, {"name": "vf2", "stage": 5, "stageId": 1, "dim": 3}
            ],
            "boundaries": [{"name": "everyRow"}], "airgroupValuesMap": [], "airValuesMap": [], "proofValuesMap": [],
            "mapSectionsN": {"cm1": 0, "cm2": 0, "cm3": 3}, "openingPoints": [0], "qDeg": 1
        })
    }

    fn verifier_info() -> Value {
        json!({ "constraintDag": { "version": 1, "field": "gl", "extDim": 3, "nodes": [[5, 1, 0, 0]], "roots": [[0, 0, "c"]] } })
    }

    fn render(si: &Value) -> Result<String> {
        let root = ["7".to_string(), "0".into(), "0".into(), "0".into()];
        gen_stark_verifier_bn128(
            Some(&root),
            si,
            &verifier_info(),
            &Pil2CircomOptions { skip_main: true, ..Default::default() },
        )
    }

    #[test]
    fn per_key_output_is_data_constraints_and_wiring() {
        let out = render(&stark_info(false, 0)).unwrap();
        assert!(out.starts_with("pragma circom 2.1.0;\n\ninclude \"evalpol.circom\";"), "out:\n{out}");
        assert!(
            out.contains("include \"sponge.circom\";\ninclude \"verifier/transcript_sponge.circom\";"),
            "out:\n{out}"
        );
        assert!(out.contains("    tmp_0 <== evals[0];\n    signal {maxNum} tmp_1[3];\n    tmp_1 <== GLCMul()(tmp_0, Zh);\n\n    evalP <== tmp_1;\n}"), "out:\n{out}");
        assert!(out.contains("    signal rootC <== 7;\n"), "out:\n{out}");
        assert!(out.contains("VerifyFRIBN128(STEPS[0], STEPS[1], 0, nBitsExt, STEPS[0])(queriesL0, rFold[0], s1_vals_p, finalPol, enable);"), "out:\n{out}");
        let custom = render(&stark_info(true, 0)).unwrap();
        assert!(
            custom.starts_with("pragma circom 2.1.0;\npragma custom_templates;\n\ninclude \"custom/evalpol.circom\";"),
            "out:\n{custom}"
        );
        assert!(custom.contains("include \"custom/sponge.circom\";"), "out:\n{custom}");
    }

    #[test]
    fn last_levels_keep_the_final_circuits_inputs() {
        let out = render(&stark_info(false, 2)).unwrap();
        assert!(out.contains("    signal input s0_siblingsC[nQueries][L0][arity];\n"), "out:\n{out}");
        assert!(out.contains("    signal input s0_last_levelsC[lastSize];\n"), "out:\n{out}");
        assert!(out.contains("    signal input s1_last_levels[lastSize];\n"), "out:\n{out}");
        assert!(
            out.contains("VerifyMerkleRoot(lastLevels, arity, 1 << STEPS[1])(s1_last_levels, s1_root, enabled);"),
            "out:\n{out}"
        );
        let none = render(&stark_info(false, 0)).unwrap();
        assert!(!none.contains("last_levels") && !none.contains("VerifyMerkleRoot"), "out:\n{none}");
    }

    /// Custom trees of any arity (the sponge rate follows it), and a key with no fold step.
    #[test]
    fn custom_arity_and_no_fold_step() {
        let mut si = stark_info(true, 0);
        si["starkStruct"]["merkleTreeArity"] = json!(2);
        let out = render(&si).unwrap();
        assert!(out.contains("function sponge_rate() { return 2; }"), "out:\n{out}");
        let mut si = stark_info(false, 0);
        si["starkStruct"]["logDomainSizes"] = json!([10]);
        si["starkStruct"]["logDegrees"] = json!([6]);
        si["starkStruct"]["foldingFactors"] = json!([]);
        let out = render(&si).unwrap();
        assert!(out.contains("STEPS[0], 0, S0[nTrees]") && out.contains("rFold[0] <== [0, 0, 0];"), "out:\n{out}");
        assert!(!out.contains("VerifyFRIBN128("), "out:\n{out}");
    }

    #[test]
    fn refuses_what_the_wrap_cannot_take() {
        let err = render(&stark_info(true, 2)).unwrap_err().to_string();
        assert!(err.contains("lastLevelVerification"), "{err}");
        let mut si = stark_info(false, 0);
        si["customCommits"] = json!([{"name": "rom", "publicValues": [{"idx": 0}], "stageWidths": [1]}]);
        assert!(render(&si).unwrap_err().to_string().contains("custom commits"));
        let mut si = stark_info(false, 0);
        si["starkStruct"]["verificationHashType"] = json!("GL");
        assert!(render(&si).unwrap_err().to_string().contains("GL"));
    }
}
