//! The GL verifier (FRI or STIR) on the static library `helper_circuits/verifier/`: per key only the data
//! (`lib_params`), the constraints (`ConstraintRoots<suffix>`) and `StarkVerifier<suffix>`, which wires
//! transcript -> roots -> checks and, until stage 4, keeps today's named proof inputs.

use anyhow::{bail, Context, Result};
use proofman_common::StarkInfo;
use serde_json::Value;

use super::constraint_roots::constraint_roots;
use super::Pil2CircomOptions;
use super::lib_params::KeyData;

/// The library templates `StarkVerifier<suffix>` instantiates.
const LIB: [&str; 5] = [
    include_str!("helper_circuits/verifier/transcript.circom"),
    include_str!("helper_circuits/verifier/stark_verifier.circom"),
    include_str!("helper_circuits/verifier/deep.circom"),
    include_str!("helper_circuits/verifier/fri.circom"),
    include_str!("helper_circuits/verifier/stir.circom"),
];

/// `Name(a, b, ...)` with the template's own parameter names: the key's vars carry those names.
fn call(name: &str) -> String {
    call_in(&LIB, name)
}

/// [`call`] over the library files `lib`.
pub(crate) fn call_in(lib: &[&str], name: &str) -> String {
    let head = format!("template {name}(");
    let params = lib
        .iter()
        .flat_map(|src| src.lines())
        .find_map(|l| l.strip_prefix(&head).or_else(|| l.strip_prefix(&format!("template parallel {name}("))))
        .and_then(|l| l.split_once(')'))
        .unwrap_or_else(|| panic!("the verifier library has no template {name}"))
        .0;
    format!("{name}({params})")
}

/// The family's hash, selector and transcript files; the library calls them by family-neutral names.
fn includes(hash: &str, split_linear_hash: bool) -> Result<String> {
    let (dir, selector, transcript) = match hash {
        "Poseidon1" => ("poseidon1", "treeselector8", "transcript_sponge"),
        "Poseidon2" => ("poseidon2", "treeselector4", "transcript_sponge"),
        // TreeSelector selects by FRI fold bits, not Merkle arity, so blake3 reuses the arity-4 one.
        "blake3" => ("blake3", "treeselector4", "transcript_blake3"),
        other => bail!("gen_stark_verifier_gl: unknown hash family '{other}'"),
    };
    if split_linear_hash && hash != "Poseidon2" {
        bail!(
            "gen_stark_verifier_gl: {hash} + splitLinearHash is not supported \
             (no hash/{dir}/merklehash_gpu.circom / linearhash_gpu.circom exist)"
        );
    }
    let merkle = if split_linear_hash { "merklehash_gpu" } else { "merklehash" };
    let mut out = format!(
        "include \"hash/{dir}/pow.circom\";\ninclude \"tree/{selector}.circom\";\ninclude \"hash/{dir}/{merkle}.circom\";\n"
    );
    if transcript == "transcript_sponge" {
        out += &format!("include \"hash/{dir}/sponge.circom\";\n");
    }
    out += &format!("include \"verifier/{transcript}.circom\";\ninclude \"verifier/stark_verifier.circom\";\n");
    Ok(out)
}

/// Generate the GL verifier of a starkinfo from its `constraintDag`.
pub(crate) fn gen_stark_verifier_gl(
    const_root: Option<&[String; 4]>,
    si: &Value,
    vi: &Value,
    opts: &Pil2CircomOptions,
) -> Result<String> {
    // Presence, not the typed default 0: a starkinfo without airgroupId names its templates unsuffixed.
    let sfx = si.get("airgroupId").and_then(Value::as_u64).map(|id| id.to_string()).unwrap_or_default();
    let info: StarkInfo = serde_json::from_value(si.clone()).context("starkinfo")?;
    // An empty evMap would leave f_0 undefined (the circuit must never check against 0).
    if info.ev_map.is_empty() {
        bail!("evMap has no evaluations, so the DEEP polynomial is undefined");
    }
    let (dag, ctx) = super::dag_and_ctx(si, vi)?;
    let roots = constraint_roots(&sfx, &dag, &ctx)?;
    let key = KeyData::new(
        &info,
        opts.input_challenges,
        opts.fri_queries_batch_size,
        Some((roots.n_roots, roots.first_root_dim, ctx.tables().vc)),
    )?;

    let mut out = String::from("pragma circom 2.1.0;\npragma custom_templates;\n\n");
    out += &includes(&opts.hash, info.stark_struct.split_linear_hash)?;
    out += &format!("\n{}\n{}", roots.template, stark_verifier(&sfx, &info, &key, const_root, opts));
    if !opts.skip_main {
        let public = if info.n_publics > 0 { "{public [publics]}" } else { "" };
        out += &format!("\ncomponent main {public}= StarkVerifier{sfx}();\n");
    }
    Ok(out)
}

/// `name` when the input exists, else the zero padding the library's one-entry input takes.
fn or_pad(n: u64, name: &str, pad: &str) -> String {
    if n > 0 {
        name.to_string()
    } else {
        pad.to_string()
    }
}

/// `StarkVerifier{sfx}`: the key's vars, the library wired around `ConstraintRoots{sfx}`, and (temporary,
/// until stage 4) today's named proof inputs copied into the library's flat ones.
fn stark_verifier(
    sfx: &str,
    si: &StarkInfo,
    key: &KeyData,
    const_root: Option<&[String; 4]>,
    opts: &Pil2CircomOptions,
) -> String {
    let ldt = &si.stark_struct.low_degree_test;
    let (stir_m, pow) = match ldt.stir() {
        Some(stir) => (stir.num_iterations(), false),
        None => (0, ldt.fri().is_some_and(|f| f.grinding_bits_queries > 0)),
    };
    let stir = stir_m > 0;
    // Committed oracles after f_0's: FRI fold trees s1..s{nf}, STIR trees s1..s{M}.
    let nf = if stir { stir_m } else { ldt.fri().map_or(0, |f| f.log_domain_sizes.len() - 1) };
    let (q_stage, ll) = (si.n_stages + 1, si.stark_struct.last_level_verification);
    let len = |m: &Option<Vec<proofman_common::PolMap>>| m.as_ref().map_or(0, Vec::len) as u64;
    let (n_av, n_agv, n_pv) = (len(&si.airvalues_map), len(&si.airgroupvalues_map), len(&si.proofvalues_map));
    let pad3 = "[[0, 0, 0]]";
    let mut d = key.vars.clone(); // declarations
    let mut w = String::new(); // wiring
    d += "    var L0 = geo_levels0(arity, lastLevels, STEPS);\n";
    d += "    var sibW = geo_sibW(arity);\n    var lastSize = geo_lastSize(arity, lastLevels);\n\n";

    // ---- Named proof inputs (temporary) ----
    if opts.input_challenges {
        d += "    signal input globalChallenge[3];\n";
    }
    if si.n_publics > 0 {
        d += "    signal input publics[nPublics];\n";
    }
    if n_av > 0 {
        d += "    signal input airvalues[nAirValues][3];\n";
    }
    if n_agv > 0 {
        d += "    signal input airgroupvalues[nAirgroupValues][3];\n";
    }
    if n_pv > 0 {
        d += &format!("    signal input proofvalues[{n_pv}][3];\n");
    }
    for st in 1..=q_stage {
        d += &format!("    signal input root{st}[4];\n");
    }
    let const_root = const_root.map(|r| r.join(",")).unwrap_or_else(|| "0,0,0,0".into());
    d += &if opts.verkey_input {
        "    signal input rootC[4];\n".to_string()
    } else if opts.input_challenges {
        format!("    signal output rootC[4] <== [{const_root}];\n")
    } else {
        format!("    signal rootC[4] <== [{const_root}];\n")
    };
    d += "    signal input evals[nEvals][3];\n";
    for t in &key.trees {
        d += &format!("    signal input s0_vals{}[nQueries][{}];\n", t.suffix, t.width);
    }
    for t in &key.trees {
        d += &format!("    signal input s0_siblings{}[nQueries][L0][sibW];\n", t.suffix);
        if ll > 0 {
            d += &format!("    signal input s0_last_mt_levels{}[lastSize][4];\n", t.suffix);
        }
    }
    for f in 1..=nf {
        d += &format!("    signal input s{f}_root[4];\n");
    }
    // Oracle f's opened cosets, Merkle paths and (lastLevels > 0) levels sent in clear.
    let last = |f: usize| fold_last(stir, f);
    for f in 1..=nf {
        let (t, ex, ml) = if stir {
            let r = f - 1;
            let ml = format!("geo_pathLevels(arity, lastLevels, STIR_L[{r}] - STIR_K[{r}])");
            (format!("STIR_T[{r}]"), format!("(1 << STIR_K[{r}])"), ml)
        } else {
            (
                "nQueries".to_string(),
                format!("geo_foldExp(STEPS, {f})"),
                format!("geo_foldLevels(arity, lastLevels, STEPS, {f})"),
            )
        };
        d += &format!("    signal input s{f}_vals[{t}][3 * {ex}];\n");
        d += &format!("    signal input s{f}_siblings[{t}][{ml}][sibW];\n");
        if ll > 0 {
            d += &format!("    signal input {}[lastSize][4];\n", last(f));
        }
    }
    d += "    signal input finalPol[geo_finalPolSize(stir, nSteps, STEPS, stirFinalBits)][3];\n";
    if stir {
        if stir_m > 1 {
            d += "    signal input betas[stirM - 1][3];\n";
        }
        d += "    signal input nonces[stirM];\n";
        for i in 0..stir_m - 1 {
            d += &format!("    signal input ansCoeffs{i}[STIR_T[{i}] + 1][3];\n    signal input shakeCoeffs{i}[STIR_T[{i}] + 1][3];\n");
        }
    } else if pow {
        d += "    signal input nonce;\n";
    }
    d += "\n    signal {binary} enabled;\n";
    d += if opts.enable_input {
        "    signal input enable;\n    enable * (enable - 1) === 0;\n    enabled <== enable;\n"
    } else {
        "    enabled <== 1;\n"
    };

    // VERIFY-CONSTRAINTS: constant padding of absent inputs (and of the clear levels when there are none) must fold away.
    let publics = or_pad(si.n_publics, "publics", "[0]");
    let airvalues = or_pad(n_av, "airvalues", pad3);
    let airgroupvalues = or_pad(n_agv, "airgroupvalues", pad3);
    let proofvalues = or_pad(n_pv, "proofvalues", pad3);
    let betas = or_pad(stir_m.saturating_sub(1) as u64, "betas", pad3);

    // STIR's Ans/shake hints, flat, once for the transcript and the rounds.
    if stir {
        w += "\n    signal stirHints[geo_len1(stir_nHints(1, stirM, STIR_T))][3];\n";
        for i in 0..stir_m - 1 {
            w += &format!(
                "    for (var m = 0; m <= STIR_T[{i}]; m++) {{ stirHints[stir_hintOff(STIR_T, {i}) + m] <== ansCoeffs{i}[m]; stirHints[stir_hintOff(STIR_T, {i}) + STIR_T[{i}] + 1 + m] <== shakeCoeffs{i}[m]; }}\n"
            );
        }
        if stir_m == 1 {
            w += &format!("    stirHints <== {pad3};\n");
        }
    }

    // ---- The verifier ----
    w += &format!("\n    component transcript = {};\n", call("StarkVerifierTranscript"));
    let global = if opts.input_challenges { "globalChallenge" } else { "[0, 0, 0]" };
    w += &format!("    transcript.globalChallenge <== {global};\n");
    w += &format!("    transcript.publics <== {publics};\n    transcript.rootC <== rootC;\n");
    for st in 1..=q_stage {
        w += &format!("    transcript.stageRoots[{}] <== root{st};\n", st - 1);
    }
    w += &format!("    transcript.airvalues <== {airvalues};\n    transcript.airgroupvalues <== {airgroupvalues};\n");
    w += "    transcript.evals <== evals;\n";
    for f in 1..=nf {
        w += &format!("    transcript.foldRoots[{}] <== s{f}_root;\n", f - 1);
    }
    if nf == 0 {
        w += "    transcript.foldRoots <== [[0, 0, 0, 0]];\n";
    }
    let hints = if stir { "stirHints" } else { pad3 };
    let nonce = if stir {
        "nonces[0]"
    } else if pow {
        "nonce"
    } else {
        "0"
    };
    w += &format!("    transcript.betas <== {betas};\n    transcript.hints <== {hints};\n");
    w += &format!("    transcript.finalPol <== finalPol;\n    transcript.nonce <== {nonce};\n    transcript.enable <== enabled;\n");

    w += &format!(
        "\n    signal roots[nRoots][3] <== ConstraintRoots{sfx}()(evals, transcript.challenges, {publics}, {proofvalues}, {airvalues}, {airgroupvalues});\n"
    );

    // challenges end with vc, xi, vf1, vf2.
    w += &format!(
        "\n    {}(transcript.challenges[nChallenges - 3], transcript.challenges[nChallenges - 4], roots, evals, enabled);\n",
        call("VerifyEvaluations")
    );
    w += &format!("\n    component deepConsts = {};\n", call("DeepConstants"));
    w += "    deepConsts.vf[0] <== transcript.challenges[nChallenges - 2];\n";
    w += "    deepConsts.vf[1] <== transcript.challenges[nChallenges - 1];\n    deepConsts.evals <== evals;\n";

    // The L_0 queries in batches; STIR's only fold tree there is T_0 (s1).
    let nq = if stir { 1 } else { nf };
    let ql = QueryLayout { key, ll, nq, fri: !stir };
    w += "\n    var nFull = nQueries \\ batchSize;\n    component batch[geo_len1(nFull)];\n";
    w += &format!("    for (var b = 0; b < nFull; b++) {{\n        batch[b] = QueryBatch{sfx}(batchSize);\n");
    w += &ql.wire_batch("batch[b]", "b * batchSize", "batchSize");
    w += "    }\n    component tail[1];\n    if (nQueries > nFull * batchSize) {\n";
    w += &format!("        tail[0] = QueryBatch{sfx}(nQueries - nFull * batchSize);\n");
    w += &ql.wire_batch("tail[0]", "nFull * batchSize", "nQueries - nFull * batchSize");
    w += "    }\n";

    // Once per proof: the levels sent in clear hash to the committed roots, finalPol's degree.
    if ll > 0 {
        for t in &key.trees {
            w += &format!(
                "    VerifyMerkleRoot(lastLevels, arity, 1 << nBitsExt)(s0_last_mt_levels{}, {}, enabled);\n",
                t.suffix, t.root
            );
        }
        for f in 1..=nq {
            w += &format!(
                "    VerifyMerkleRoot(lastLevels, arity, 1 << STEPS[{f}])({}, s{f}_root, enabled);\n",
                last(f)
            );
        }
    }
    if stir {
        w += &stir_rounds(stir_m, ll, &betas);
    } else {
        w += "    VerifyFinalPol(STEPS[nSteps - 1], geo_maxDegSize(nBits, nBitsExt, nSteps, STEPS))(finalPol, enabled);\n";
    }

    format!(
        "// Verifies one STARK proof on the library; the inputs keep today's names.\ntemplate StarkVerifier{sfx}() {{\n{d}{w}}}\n\n{}",
        ql.batch_template(sfx)
    )
}

/// Oracle f's levels sent in clear (the proof input).
fn fold_last(stir: bool, f: usize) -> String {
    if stir {
        format!("s{f}_last_levels")
    } else {
        format!("s{f}_last_mt_levels")
    }
}

/// The per-query checks: per L_0 tree and per fold tree (STIR: T_0) one component array of a single
/// template instance, a FRI fold array per step, DEEP.
struct QueryLayout<'a> {
    key: &'a KeyData,
    ll: u64,
    /// Fold trees opened at the L_0 queries.
    nq: usize,
    fri: bool,
}

impl QueryLayout<'_> {
    /// `{comp}` takes queries q0..q0+n of the proof inputs, and the per-proof data.
    fn wire_batch(&self, comp: &str, q0: &str, n: &str) -> String {
        let mut w = format!("        for (var q = 0; q < {n}; q++) {{\n");
        w += &format!("            {comp}.queriesL0[q] <== transcript.queriesL0[{q0} + q];\n");
        for t in &self.key.trees {
            let s = &t.suffix;
            w += &format!("            {comp}.s0_vals{s}[q] <== s0_vals{s}[{q0} + q];\n");
            w += &format!("            {comp}.s0_siblings{s}[q] <== s0_siblings{s}[{q0} + q];\n");
        }
        for f in 1..=self.nq {
            w += &format!("            {comp}.s{f}_vals[q] <== s{f}_vals[{q0} + q];\n");
            w += &format!("            {comp}.s{f}_siblings[q] <== s{f}_siblings[{q0} + q];\n");
        }
        w += "        }\n";
        w += &format!("        {comp}.challengeOut <== transcript.challenges[nChallenges - 3];\n");
        w += &format!("        {comp}.friConsts <== deepConsts.friConsts;\n");
        for t in &self.key.trees {
            w += &format!("        {comp}.s0_root{} <== {};\n", t.suffix, t.root);
            if self.ll > 0 {
                w += &format!("        {comp}.s0_last{} <== s0_last_mt_levels{};\n", t.suffix, t.suffix);
            }
        }
        for f in 1..=self.nq {
            w += &format!("        {comp}.s{f}_root <== s{f}_root;\n");
            if self.ll > 0 {
                w += &format!("        {comp}.s{f}_last <== {};\n", fold_last(!self.fri, f));
            }
            if self.fri {
                w += &format!("        {comp}.rFold{f} <== transcript.ldt[{}];\n", f - 1);
            }
        }
        if self.fri {
            w += &format!("        {comp}.finalPol <== finalPol;\n");
        }
        w += &format!("        {comp}.enable <== enabled;\n");
        w
    }

    /// `QueryBatch{sfx}(n)`: n queries, the witness-parallel unit.
    fn batch_template(&self, sfx: &str) -> String {
        let pad4 = "[[0, 0, 0, 0]]";
        let mut d = self.key.vars.clone();
        d += "    var L0 = geo_levels0(arity, lastLevels, STEPS);\n";
        d += "    var sibW = geo_sibW(arity);\n    var lastSize = geo_lastSize(arity, lastLevels);\n\n";
        d += "    signal input {binary} queriesL0[n][STEPS[0]];\n    signal input challengeOut[3];\n";
        d += "    signal input friConsts[2 + nPols + 2 * nGroups][3];\n";
        // One key split per query, shared by the L_0 trees.
        let mut w = String::from("    component keys0[n];\n");
        let mut body = String::from(
            "        keys0[q] = QueryKeys(arity, L0 + lastLevels, STEPS[0], STEPS[0]);\n        keys0[q].queries <== queriesL0[q];\n",
        );
        for (k, t) in self.key.trees.iter().enumerate() {
            let (s, width) = (&t.suffix, t.width);
            d += &format!("    signal input s0_vals{s}[n][{width}];\n    signal input s0_siblings{s}[n][L0][sibW];\n");
            d += &format!("    signal input s0_root{s}[4];\n");
            let last = if self.ll > 0 {
                d += &format!("    signal input s0_last{s}[lastSize][4];\n");
                format!("s0_last{s}")
            } else {
                pad4.to_string()
            };
            w += &format!("    component tree{k}[n];\n");
            body += &format!("        tree{k}[q] = VerifyTreeOpening(1, {width}, arity, lastLevels, L0, nBitsExt);\n");
            body += &format!("        tree{k}[q].keys <== keys0[q].keys;\n");
            body += &format!(
                "        for (var i = 0; i < {width}; i++) {{ tree{k}[q].values[i][0] <== s0_vals{s}[q][i]; }}\n"
            );
            body += &format!("        tree{k}[q].siblings <== s0_siblings{s}[q];\n");
            body += &format!("        tree{k}[q].last <== {last};\n        tree{k}[q].root <== s0_root{s};\n");
            body += &format!("        tree{k}[q].enable <== enable;\n");
        }
        // A coset input s{f}_vals[q] is flat, 3 limbs per member.
        let coset = |dst: &str, f: usize| {
            format!(
                "        for (var c = 0; c < geo_foldExp(STEPS, {f}); c++) {{ for (var e = 0; e < 3; e++) {{ {dst}[c][e] <== s{f}_vals[q][3 * c + e]; }} }}\n"
            )
        };
        for f in 1..=self.nq {
            let ml = format!("geo_foldLevels(arity, lastLevels, STEPS, {f})");
            d += &format!("    signal input s{f}_vals[n][3 * geo_foldExp(STEPS, {f})];\n");
            d += &format!("    signal input s{f}_siblings[n][{ml}][sibW];\n    signal input s{f}_root[4];\n");
            let last = if self.ll > 0 {
                d += &format!("    signal input s{f}_last[lastSize][4];\n");
                format!("s{f}_last")
            } else {
                pad4.to_string()
            };
            w += &format!("    component open{f}[n];\n");
            body += &format!(
                "        open{f}[q] = VerifyTreeOpening(3, geo_foldExp(STEPS, {f}), arity, lastLevels, {ml}, STEPS[{f}]);\n"
            );
            body += &format!(
                "        open{f}[q].keys <== QueryKeys(arity, {ml} + lastLevels, STEPS[{f}], STEPS[0])(queriesL0[q]);\n"
            );
            body += &coset(&format!("open{f}[q].values"), f);
            body += &format!("        open{f}[q].siblings <== s{f}_siblings[q];\n");
            body += &format!("        open{f}[q].last <== {last};\n        open{f}[q].root <== s{f}_root;\n");
            body += &format!("        open{f}[q].enable <== enable;\n");
            if self.fri {
                d += &format!("    signal input rFold{f}[3];\n");
                let next = if f < self.nq { format!("STEPS[{}]", f + 1) } else { "0".into() };
                w += &format!("    component fold{f}[n];\n");
                body += &format!(
                    "        fold{f}[q] = VerifyFRIFold(STEPS[{}], STEPS[{f}], {next}, geo_e0(nBitsExt, STEPS, {f}), STEPS[0]);\n",
                    f - 1
                );
                body += &format!(
                    "        fold{f}[q].queriesL0 <== queriesL0[q];\n        fold{f}[q].friChallenge <== rFold{f};\n"
                );
                body += &coset(&format!("fold{f}[q].s_vals_curr"), f);
                if f < self.nq {
                    body += &coset(&format!("fold{f}[q].s_vals_next"), f + 1);
                } else {
                    body += &format!("        fold{f}[q].s_vals_next <== finalPol;\n");
                }
                body += &format!("        fold{f}[q].enable <== enable;\n");
            }
        }
        if self.fri {
            d += "    signal input finalPol[geo_finalSize(nSteps, STEPS)][3];\n";
        }
        d += "    signal input {binary} enable;\n\n";

        // DEEP: f_0 from every L_0 row, against fold tree 1's coset.
        w += "    component deep[n];\n";
        // With no fold step f_0 is checked against finalPol itself.
        let next = if self.nq > 0 { "STEPS[1]" } else { "0" };
        body += &format!("        deep[q] = DeepQuery(nGroups, nPols, nColGroups, nSums, STEPS[0], {next}, S0[nTrees], OPEN_ROOT, SUMS, COLS);\n");
        body += "        deep[q].queriesL0 <== queriesL0[q];\n        deep[q].challengeOut <== challengeOut;\n";
        body += "        deep[q].friConsts <== friConsts;\n";
        for (k, t) in self.key.trees.iter().enumerate() {
            body += &format!(
                "        for (var i = 0; i < {}; i++) {{ deep[q].s0_vals[S0[{k}] + i] <== s0_vals{}[q][i]; }}\n",
                t.width, t.suffix
            );
        }
        body +=
            &if self.nq > 0 { coset("deep[q].s1_vals", 1) } else { "        deep[q].s1_vals <== finalPol;\n".into() };
        body += "        deep[q].enable <== enable;\n";

        format!(
            "// One batch of queries: the L_0 rows, the fold-tree cosets, DEEP{}.\ntemplate parallel QueryBatch{sfx}(n) {{\n{d}{w}\n    for (var q = 0; q < n; q++) {{\n{body}    }}\n}}\n",
            if self.fri { " and every folding step" } else { "" }
        )
    }
}

/// STIR's rounds after the L_0 queries: today's per-tree inputs flattened into `StirVerifyRounds`.
fn stir_rounds(m: usize, ll: u64, betas: &str) -> String {
    let mut w = String::from("\n    signal stirVals[stir_valsOff(STIR_T, STIR_K, stirM)][3];\n");
    w += "    signal stirSibs[geo_len1(stir_sibOff(arity, lastLevels, STIR_L, STIR_K, STIR_T, stirM))][sibW];\n";
    for r in 0..m {
        let f = r + 1;
        w += &format!(
            "    for (var q = 0; q < STIR_T[{r}]; q++) {{ for (var c = 0; c < (1 << STIR_K[{r}]); c++) {{ for (var e = 0; e < 3; e++) {{ stirVals[stir_valsOff(STIR_T, STIR_K, {r}) + (q << STIR_K[{r}]) + c][e] <== s{f}_vals[q][c * 3 + e]; }} }} }}\n"
        );
        if r > 0 {
            w += &format!(
                "    for (var q = 0; q < STIR_T[{r}]; q++) {{ var ml = geo_pathLevels(arity, lastLevels, STIR_L[{r}] - STIR_K[{r}]); for (var j = 0; j < ml; j++) {{ stirSibs[stir_sibOff(arity, lastLevels, STIR_L, STIR_K, STIR_T, {r}) + q * ml + j] <== s{f}_siblings[q][j]; }} }}\n"
            );
        }
    }
    w += "    if (stir_sibOff(arity, lastLevels, STIR_L, STIR_K, STIR_T, stirM) == 0) { for (var k = 0; k < sibW; k++) { stirSibs[0][k] <== 0; } }\n";

    w += &format!("\n    component rounds = {};\n", call("StirVerifyRounds"));
    w += "    rounds.queriesL0 <== transcript.queriesL0;\n    rounds.ldt <== transcript.ldt;\n    rounds.nonces <== nonces;\n";
    w += "    rounds.vals <== stirVals;\n    rounds.siblings <== stirSibs;\n";
    for r in 0..m {
        let f = r + 1;
        let lv = if ll > 0 { format!("s{f}_last_levels") } else { "[[0, 0, 0, 0]]".into() };
        w += &format!("    rounds.last[{r}] <== {lv};\n    rounds.roots[{r}] <== s{f}_root;\n");
    }
    w += &format!("    rounds.hints <== stirHints;\n    rounds.betas <== {betas};\n");
    w += "    rounds.finalPol <== finalPol;\n    rounds.enable <== enabled;\n";
    w
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    /// A FRI starkinfo with every input kind: 2 stages, a custom commit, last levels in clear.
    pub(crate) fn stark_info() -> Value {
        // Opening 1 is empty; columns: cm0 (base) at {0,2}, cm1 (ext) at {0}, const0 at {2}, custom at {0,2}.
        json!({
            "name": "Sample",
            "starkStruct": {
                "verificationHashType": "GL", "numQueries": 10, "nBits": 16, "nBitsExt": 17,
                "grindingBitsQueries": 8, "merkleTreeArity": 4, "transcriptArity": 4, "merkleTreeCustom": false,
                "lastLevelVerification": 1, "splitLinearHash": false, "hashCommits": false,
                "foldingFactors": [], "logDegrees": [], "logDomainSizes": [17, 14, 11]
            },
            "security": {"regime": "test"},
            "nStages": 2,
            "nPublics": 4,
            "nConstants": 1,
            "qDeg": 1,
            "qDim": 3,
            "cExpId": 0,
            "boundaries": [{"name": "everyRow"}],
            "challengesMap": [
                {"name": "std_alpha", "stage": 2, "stageId": 0},
                {"name": "std_vc", "stage": 3, "stageId": 0},
                {"name": "std_xi", "stage": 4, "stageId": 0},
                {"name": "std_vf1", "stage": 5, "stageId": 0},
                {"name": "std_vf2", "stage": 5, "stageId": 1}
            ],
            "airgroupValuesMap": [{"name": "g", "stage": 1}],
            "airValuesMap": [],
            "proofValuesMap": [],
            "mapSectionsN": {"cm1": 1, "cm2": 3, "cm3": 3, "rom0": 1},
            "openingPoints": [-1, 0, 1],
            "cmPolsMap": [
                {"name": "a", "stage": 1, "stagePos": 0, "stageId": 0, "dim": 1},
                {"name": "b", "stage": 2, "stagePos": 0, "stageId": 0, "dim": 3},
                {"name": "q", "stage": 3, "stagePos": 0, "stageId": 0, "dim": 3}
            ],
            "customCommits": [{"name": "rom", "publicValues": [{"idx": 0}, {"idx": 1}, {"idx": 2}, {"idx": 3}], "stageWidths": [1]}],
            "customCommitsMap": [[{"name": "r", "stage": 0, "stagePos": 0, "stageId": 0, "dim": 1}]],
            "evMap": [
                {"type": "cm", "id": 0, "prime": -1, "openingPos": 0},
                {"type": "cm", "id": 1, "prime": -1, "openingPos": 0},
                {"type": "custom", "id": 0, "prime": -1, "openingPos": 0, "commitId": 0},
                {"type": "cm", "id": 0, "prime": 1, "openingPos": 2},
                {"type": "const", "id": 0, "prime": 1, "openingPos": 2},
                {"type": "custom", "id": 0, "prime": 1, "openingPos": 2, "commitId": 0},
                {"type": "cm", "id": 2, "prime": 1, "openingPos": 2}
            ]
        })
    }

    /// Roots: cm0 * cm2' - alpha, and the quotient column read as is.
    pub(crate) fn verifier_info() -> Value {
        json!({ "constraintDag": {
            "version": 1, "field": "gl", "extDim": 3,
            "nodes": [[5, 0, 0, -1], [5, 0, 2, 1], [2, 0, 1], [6, 2, 0], [1, 2, 3]],
            "roots": [[4, 0, "c0"], [1, 0, "q"]]
        }})
    }

    fn render(si: &Value, opts: &Pil2CircomOptions) -> String {
        gen_stark_verifier_gl(None, si, &verifier_info(), opts).unwrap()
    }

    fn opts(hash: &str) -> Pil2CircomOptions {
        Pil2CircomOptions { hash: hash.to_string(), skip_main: true, ..Default::default() }
    }

    /// Per key only data, constraints and the wiring: no generated FRI, transcript or DEEP code.
    #[test]
    fn per_key_output_is_data_constraints_and_wiring() {
        let out = render(&stark_info(), &opts("Poseidon2"));
        let templates: Vec<&str> = out.lines().filter(|l| l.starts_with("template ")).collect();
        assert_eq!(
            templates,
            [
                "template parallel ConstraintRoots() {",
                "template StarkVerifier() {",
                "template parallel QueryBatch(n) {"
            ],
            "{out}"
        );
        assert!(out.contains("    var STEPS[3] = [17, 14, 11];"), "{out}");
        // Every library parameter is one of the key's vars.
        for t in ["StarkVerifierTranscript", "VerifyEvaluations", "DeepConstants"] {
            let c = call(t);
            assert!(out.contains(&c), "{t} is not instantiated:\n{out}");
            for p in c[t.len() + 1..c.len() - 1].split(", ") {
                let declared = out.contains(&format!("    var {p} =")) || out.contains(&format!("    var {p}["));
                assert!(declared, "{t}'s parameter {p} is no var:\n{out}");
            }
        }
        for gone in ["Transcript", "VerifyFRI", "DeepConstants", "VerifyQuery", "Zfirst", "Zlast", "Zframe"] {
            assert!(!out.contains(&format!("template {gone}")), "{gone} is generated:\n{out}");
        }
        assert!(out.contains("include \"verifier/stark_verifier.circom\";"), "{out}");
        assert!(out.contains("include \"verifier/transcript_sponge.circom\";"), "{out}");
        assert!(out.contains("include \"hash/poseidon2/sponge.circom\";"), "{out}");
    }

    /// The proof inputs keep the names the recursive templates wire.
    #[test]
    fn glue_keeps_the_proof_input_names() {
        let out = render(&stark_info(), &opts("Poseidon2"));
        for decl in [
            "signal input publics[nPublics];",
            "signal input root1[4];",
            "signal input root3[4];",
            "signal rootC[4] <== [0,0,0,0];",
            "signal input s0_vals1[nQueries][1];",
            "signal input s0_valsC[nQueries][1];",
            "signal input s0_vals_rom_0[nQueries][1];",
            "signal input s0_last_mt_levels_rom_0[lastSize][4];",
            "signal input s1_root[4];",
            "signal input s2_vals[nQueries][3 * geo_foldExp(STEPS, 2)];",
            "signal input finalPol[geo_finalPolSize(stir, nSteps, STEPS, stirFinalBits)][3];",
            "signal input nonce;",
            "batch[b].s0_root_rom_0 <== [publics[0], publics[1], publics[2], publics[3]];",
        ] {
            assert!(out.contains(decl), "missing `{decl}`:\n{out}");
        }
        assert!(!out.contains("s3_root"), "{out}");
    }

    #[test]
    fn data_carries_the_protocol_flags() {
        let mut si = stark_info();
        si["circuitType"] = json!(37);
        si["starkStruct"]["hashCommits"] = json!(true);
        let o = Pil2CircomOptions { input_challenges: true, ..opts("blake3") };
        let out = render(&si, &o);
        assert!(out.contains("    var hashCommits = 1;"), "{out}");
        assert!(out.contains("    var circuitType = 37;"), "{out}");
        assert!(out.contains("signal input globalChallenge[3];"), "{out}");
        assert!(out.contains("signal output rootC[4]"), "{out}");
        assert!(out.contains("include \"verifier/transcript_blake3.circom\";"), "{out}");
        assert!(!out.contains("sponge.circom"), "{out}");
    }

    #[test]
    #[should_panic(expected = "nStages")]
    fn basic_transcript_needs_stage2() {
        let mut si = stark_info();
        si["circuitType"] = json!(37);
        si["nStages"] = json!(1);
        render(&si, &Pil2CircomOptions { input_challenges: true, ..opts("Poseidon2") });
    }

    #[test]
    #[should_panic(expected = "circuitType")]
    fn basic_transcript_needs_circuit_type() {
        render(&stark_info(), &Pil2CircomOptions { input_challenges: true, ..opts("Poseidon2") });
    }

    #[test]
    fn deep_groups_reach_the_data() {
        let out = render(&stark_info(), &opts("Poseidon2"));
        // Openings -1 and 1 have evaluations; columns by first appearance: cm1_0, cm2_0, rom, const, cm3_0,
        // whose leaves sit at S0 offsets 0, 1, 8, 7, 4. Column groups {cm1_0, rom} at both, {cm2_0}, {const, cm3_0}.
        for var in [
            "var nGroups = 2;",
            "var nPols = 5;",
            "var nSums = 1;",
            "var S0[6] = [0, 1, 4, 7, 8, 9];",
            "var GROUP_LEN[2] = [3, 4];",
            "var REF_COL[7] = [0, 1, 2, 0, 3, 2, 4];",
            "var SUMS[2] = [0, 1];",
            "var COLS[21] = [2, 2, 0, 0, 1, 2, 8, 1, 0, 1, 1, 1, 3, 1, 2, 3, 7, 1, 4, 4, 3];",
        ] {
            assert!(out.contains(var), "missing `{var}`:\n{out}");
        }
    }

    #[test]
    fn no_evaluations_is_an_error() {
        let mut si = stark_info();
        si["evMap"] = json!([]);
        let err = gen_stark_verifier_gl(None, &si, &verifier_info(), &opts("Poseidon2")).unwrap_err();
        assert!(err.to_string().contains("no evaluations"), "{err}");
    }

    #[test]
    fn needs_a_constraint_dag() {
        let err = gen_stark_verifier_gl(None, &stark_info(), &json!({}), &opts("Poseidon2")).unwrap_err();
        assert!(err.to_string().contains("constraintDag"), "{err}");
    }

    #[test]
    fn refuses_challenges_the_transcript_does_not_draw() {
        let mut si = stark_info();
        si["challengesMap"].as_array_mut().unwrap().push(json!({"name": "x", "stage": 1, "stageId": 0}));
        let err = gen_stark_verifier_gl(None, &si, &verifier_info(), &opts("Poseidon2")).unwrap_err();
        assert!(err.to_string().contains("not the order the verifier transcript draws"), "{err}");
    }

    #[test]
    fn names_with_airgroup_id_and_main() {
        let mut si = stark_info();
        si["airgroupId"] = json!(42u64);
        let out = render(&si, &Pil2CircomOptions { skip_main: false, ..opts("Poseidon1") });
        assert!(out.contains("template StarkVerifier42() {"), "{out}");
        assert!(out.contains("template parallel ConstraintRoots42() {"), "{out}");
        assert!(out.contains("component main {public [publics]}= StarkVerifier42();"), "{out}");
        assert!(out.contains("include \"tree/treeselector8.circom\";"), "{out}");
    }

    /// Render a key's verifier with the options and fixed root another render of it used (Task 0 measurement):
    ///   LIB_SI=<starkinfo> LIB_VI=<verifierinfo> LIB_TODAY=<a .verifier.circom of the key> LIB_OUT=<file>
    ///   [LIB_AIRGROUP=<id> renames the per-key templates] [LIB_MAIN=1 adds `component main`]
    ///   cargo test --release -p pil2-stark-recurser render_lib_from_files -- --ignored
    #[test]
    #[ignore]
    fn render_lib_from_files() {
        let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"));
        let read = |k: &str| std::fs::read_to_string(env(k)).unwrap_or_else(|e| panic!("{k}: {e}"));
        let mut si: Value = serde_json::from_str(&read("LIB_SI")).unwrap();
        let vi: Value = serde_json::from_str(&read("LIB_VI")).unwrap();
        let today = read("LIB_TODAY");
        if let Ok(id) = std::env::var("LIB_AIRGROUP") {
            si["airgroupId"] = json!(id.parse::<u64>().unwrap());
        }
        let hash = ["Poseidon1", "Poseidon2", "blake3"]
            .into_iter()
            .find(|h| today.contains(&format!("include \"hash/{}/", h.to_lowercase())))
            .expect("today's render names no hash family");
        let const_root: Option<[String; 4]> = today.lines().find(|l| l.contains("rootC[4] <== [")).map(|l| {
            let inner = &l[l.find("<== [").unwrap() + 5..];
            let parts: Vec<String> =
                inner[..inner.find(']').unwrap()].split(',').map(|x| x.trim().to_string()).collect();
            parts.try_into().expect("rootC has 4 words")
        });
        let opts = Pil2CircomOptions {
            hash: hash.to_string(),
            skip_main: std::env::var("LIB_MAIN").is_err(),
            verkey_input: !today.contains("rootC[4] <== ["),
            enable_input: today.lines().any(|l| l.trim() == "signal input enable;"),
            input_challenges: today.contains("signal input globalChallenge[3];"),
            fri_queries_batch_size: None,
        };
        let out = gen_stark_verifier_gl(const_root.as_ref(), &si, &vi, &opts).unwrap();
        std::fs::write(env("LIB_OUT"), out).unwrap();
    }

    /// `stark_info()` with a 2-iteration STIR schedule.
    fn stir_info() -> Value {
        let mut si = stark_info();
        let ss = &mut si["starkStruct"];
        ss["lowDegreeTest"] = json!("STIR");
        ss["foldingFactors"] = json!([3, 3]);
        ss["logDegrees"] = json!([16, 13, 10]);
        ss["logDomainSizes"] = json!([17, 16, 15]);
        ss["numQueries"] = json!([10, 6]);
        ss["grindingBitsQueries"] = json!([4, 0]);
        si
    }

    /// STIR is the same library: the L_0 queries with T_0 as the one fold tree, then `StirVerifyRounds`.
    #[test]
    fn stir_key_is_data_and_wiring() {
        let out = render(&stir_info(), &opts("Poseidon2"));
        let templates: Vec<&str> = out.lines().filter(|l| l.starts_with("template ")).collect();
        assert_eq!(
            templates,
            [
                "template parallel ConstraintRoots() {",
                "template StarkVerifier() {",
                "template parallel QueryBatch(n) {"
            ],
            "{out}"
        );
        for t in ["StarkVerifierTranscript", "StirVerifyRounds"] {
            let c = call(t);
            assert!(out.contains(&c), "{t} is not instantiated:\n{out}");
            for p in c[t.len() + 1..c.len() - 1].split(", ") {
                let declared = out.contains(&format!("    var {p} =")) || out.contains(&format!("    var {p}["));
                assert!(declared, "{t}'s parameter {p} is no var:\n{out}");
            }
        }
        for var in [
            "var stir = 1;",
            "var friFold = 0;",
            "var nQueries = 10;",
            "var powBits = 4;",
            "var STEPS[2] = [17, 14];",
            "var STIR_T[2] = [10, 6];",
            "var stirFinalBits = 10;",
        ] {
            assert!(out.contains(var), "missing `{var}`:\n{out}");
        }
        for decl in [
            "signal input s2_root[4];",
            "signal input s2_vals[STIR_T[1]][3 * (1 << STIR_K[1])];",
            "signal input s1_last_levels[lastSize][4];",
            "signal input betas[stirM - 1][3];",
            "signal input nonces[stirM];",
            "signal input ansCoeffs0[STIR_T[0] + 1][3];",
            "signal input shakeCoeffs0[STIR_T[0] + 1][3];",
            "batch[b].s1_last <== s1_last_levels;",
            "transcript.nonce <== nonces[0];",
            "rounds.last[1] <== s2_last_levels;",
        ] {
            assert!(out.contains(decl), "missing `{decl}`:\n{out}");
        }
        assert!(!out.contains("signal input nonce;"), "{out}");
        assert!(!out.contains("shakeCoeffs1"), "{out}");
        assert!(!out.contains("VerifyFRIFold"), "STIR folds T_0 in the rounds:\n{out}");
    }

    #[test]
    fn fri_key_pads_the_stir_inputs() {
        let out = render(&stark_info(), &opts("Poseidon2"));
        for line in [
            "var stir = 0;",
            "var friFold = 1;",
            "transcript.betas <== [[0, 0, 0]];",
            "transcript.hints <== [[0, 0, 0]];",
        ] {
            assert!(out.contains(line), "missing `{line}`:\n{out}");
        }
        assert!(!out.contains("StirVerifyRounds"), "{out}");
    }

    #[test]
    fn malformed_stir_schedule_is_an_error() {
        let mut si = stir_info();
        si["starkStruct"]["numQueries"] = json!([10]);
        let err = gen_stark_verifier_gl(None, &si, &verifier_info(), &opts("Poseidon2")).unwrap_err();
        assert!(err.to_string().contains("malformed STIR schedule"), "{err}");
    }

    #[test]
    fn split_linear_hash_only_with_poseidon2() {
        let mut si = stark_info();
        si["starkStruct"]["splitLinearHash"] = json!(true);
        let out = render(&si, &opts("Poseidon2"));
        assert!(out.contains("include \"hash/poseidon2/merklehash_gpu.circom\";"), "{out}");
        let err = gen_stark_verifier_gl(None, &si, &verifier_info(), &opts("blake3")).unwrap_err();
        assert!(err.to_string().contains("blake3 + splitLinearHash is not supported"), "{err}");
    }
}
