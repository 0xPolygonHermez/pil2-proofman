//! Reading the generated circom: its template blocks, that they only constrain (`<==`, `===`), the main's key
//! dispatch.

use anyhow::{bail, Context, Result};

/// A `template` block: its name and its text, from the `template` line to the first line that is `}`.
pub struct Block<'a> {
    pub name: &'a str,
    pub text: &'a str,
}

pub fn template_blocks(src: &str) -> Result<Vec<Block<'_>>> {
    let mut blocks = Vec::new();
    let mut start: Option<(usize, &str)> = None;
    let mut at = 0;
    for line in src.split_inclusive('\n') {
        let l = line.trim_end();
        if let Some(head) = l.strip_prefix("template ") {
            if start.is_some() {
                bail!("a template starts inside another: {l}");
            }
            let head = head.strip_prefix("parallel ").unwrap_or(head);
            let name = head.split('(').next().unwrap_or("").trim();
            start = Some((at, name));
        } else if l == "}" {
            if let Some((s, name)) = start.take() {
                blocks.push(Block { name, text: &src[s..at + line.len()] });
            }
        }
        at += line.len();
    }
    if let Some((_, name)) = start {
        bail!("template {name} has no closing line");
    }
    Ok(blocks)
}

/// Words that would let a template do more than declare signals and constrain them.
const NOT_IN_A_CONSTRAINT: &[&str] = &[
    "var",
    "if",
    "else",
    "for",
    "while",
    "component",
    "function",
    "return",
    "assert",
    "log",
    "include",
    "pragma",
    "main",
    "custom",
    "bus",
];

/// Every line of `block` declares a signal or constrains one: `<==` or `===`, never `<--`, `-->` or a variable.
pub fn constraint_only(block: &Block) -> Result<()> {
    for (n, line) in block.text.lines().enumerate() {
        let l = line.trim();
        let bad = |why: &str| -> Result<()> { bail!("template {}, line {}: {why}: `{l}`", block.name, n + 1) };
        if n == 0 {
            if !(l.starts_with("template ") && l.ends_with('{')) {
                bad("not a template head")?;
            }
            continue;
        }
        if l.is_empty() || l.starts_with("//") || l == "}" {
            continue;
        }
        if l.contains("<--") || l.contains("-->") || l.contains("==>") {
            bad("an unconstrained assignment")?;
        }
        if l.contains("/*") || l.contains("//") {
            bad("a comment inside a statement")?;
        }
        let words = l.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'));
        if let Some(w) = words.into_iter().find(|w| NOT_IN_A_CONSTRAINT.contains(w)) {
            bad(&format!("`{w}`"))?;
        }
        let Some(stmt) = l.strip_suffix(';') else { return bad("not one statement") };
        let stmt = &without_tag(stmt);
        if stmt.contains(';') || stmt.contains('{') || stmt.contains('}') {
            bad("not one statement")?;
        }
        let (lhs, rhs) = match (stmt.split_once("<=="), stmt.split_once("===")) {
            (Some(s), None) | (None, Some(s)) => s,
            (None, None) => (stmt.as_str(), ""),
            (Some(_), Some(_)) => return bad("two constraints"),
        };
        // `=` only inside the operator: no assignment, no comparison
        if lhs.contains('=') || rhs.contains('=') {
            bad("an assignment")?;
        }
        if rhs.is_empty() && !stmt.contains("===") && !lhs.starts_with("signal ") {
            bad("neither a declaration nor a constraint")?;
        }
    }
    Ok(())
}

/// A declaration's tag dropped: `signal [input |output ]{tag} x` as `signal [input |output ]x`.
fn without_tag(stmt: &str) -> String {
    for head in ["signal input ", "signal output ", "signal "] {
        if let Some(rest) = stmt.strip_prefix(head).and_then(|r| r.strip_prefix('{')) {
            if let Some((tag, rest)) = rest.split_once("} ") {
                if tag.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    return format!("{head}{rest}");
                }
            }
        }
    }
    stmt.to_string()
}

/// The template `ConstraintRootsAt(k)` of a GL main must call for key `k`: `ConstraintRoots<suffix>`, with the
/// static library's inputs. Returns each key's suffix, the block checked line by line.
pub fn roots_dispatch(main: &str, n_keys: usize) -> Result<Vec<String>> {
    let blocks = template_blocks(main)?;
    let at: Vec<&Block> = blocks.iter().filter(|b| b.name == "ConstraintRootsAt").collect();
    let [block] = at[..] else { bail!("the main has {} ConstraintRootsAt templates, not one", at.len()) };
    let calls: Vec<&str> = block.text.lines().filter(|l| l.contains("roots <== ")).collect();
    if calls.len() != n_keys {
        bail!("ConstraintRootsAt calls {} templates for {n_keys} keys", calls.len());
    }
    let mut suffixes = Vec::new();
    for c in &calls {
        let s = c.split_once("roots <== ConstraintRoots").and_then(|(_, r)| r.split_once("()(")).map(|(s, _)| s);
        let s = s.with_context(|| format!("ConstraintRootsAt: `{c}`"))?;
        if !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            bail!("ConstraintRootsAt: suffix `{s}`");
        }
        suffixes.push(s.to_string());
    }
    let call = |s: &str| {
        format!("roots <== ConstraintRoots{s}()(evals, challenges, publics, proofvalues, airvalues, airgroupvalues);")
    };
    let body: String = match &suffixes[..] {
        [s] => format!("    {}\n", call(s)),
        _ => suffixes.iter().enumerate().map(|(k, s)| format!("    if (k == {k}) {{ {} }}\n", call(s))).collect(),
    };
    let want = format!(
        "template ConstraintRootsAt(k) {{
    signal input evals[key_nEvals(k)][3];
    signal input challenges[key_nChallenges(k)][3];
    signal input publics[nonEmpty(key_nPublics(k))];
    signal input proofvalues[nonEmpty(key_nProofValues(k))][3];
    signal input airvalues[nonEmpty(key_nAirValues(k))][3];
    signal input airgroupvalues[nonEmpty(key_nAirgroupValues(k))][3];
    signal output roots[key_nRoots(k)][3];

{body}}}
"
    );
    if block.text != want {
        bail!("ConstraintRootsAt is not the dispatch of its keys to their roots:\n{}", block.text);
    }
    Ok(suffixes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(text: &str) -> Block<'_> {
        Block { name: "T", text }
    }

    #[test]
    fn constraints_and_declarations_only() {
        let ok =
            "template parallel T() {\n    signal input a[2][3];\n    signal {maxNum} x <== GLMul()(a[0], a[1]);\n    \
                  signal output o;\n    (o) <== P0()(a);\n    o === 0;\n\n}\n";
        constraint_only(&block(ok)).unwrap();
        for bad in [
            "    o <-- a[0];\n",
            "    a[0] --> o;\n",
            "    var x = 1;\n",
            "    o.maxNum = 1;\n",
            "    if (1 == 1) { o <== 1; }\n",
            "    component c = P();\n",
            "    o <== a; o <-- a;\n",
            "    o <== a[0] == a[1];\n",
            "    o\n",
            "    log(o);\n",
        ] {
            let text = format!("template T() {{\n{bad}}}\n");
            assert!(constraint_only(&block(&text)).is_err(), "{bad}");
        }
    }

    #[test]
    fn blocks() {
        let src = "function key_nBits(k) { return 5; }\ntemplate A(n) {\n    signal input x;\n}\ntemplate parallel B() {\n}\n";
        let b = template_blocks(src).unwrap();
        assert_eq!(b.iter().map(|b| b.name).collect::<Vec<_>>(), ["A", "B"]);
        assert_eq!(b[1].text, "template parallel B() {\n}\n");
    }
}
