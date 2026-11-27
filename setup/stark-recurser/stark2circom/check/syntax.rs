//! Reading the generated circom: its template blocks, that they only constrain (`<==`, `===`), the main's key
//! dispatch and its data functions.

use std::collections::BTreeMap;

use anyhow::{bail, ensure, Context, Result};

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

/// A data function's value: an integer or a list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Val {
    Int(i128),
    /// A non-negative integer past i128, as canonical decimal digits: a BN128 key's root.
    Big(String),
    List(Vec<Val>),
}

impl Val {
    /// A decimal literal: `Int` when it fits, else `Big`.
    pub fn decimal(s: &str) -> Result<Val> {
        if let Ok(n) = s.parse::<i128>() {
            return Ok(Val::Int(n));
        }
        ensure!(!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()), "not an integer: `{s}`");
        Ok(Val::Big(s.trim_start_matches('0').to_string()))
    }

    pub fn ints(v: &[i128]) -> Val {
        if v.is_empty() {
            // circom has no empty arrays
            return Val::List(vec![Val::Int(0)]);
        }
        Val::List(v.iter().map(|&x| Val::Int(x)).collect())
    }

    fn parse(s: &str) -> Result<(Val, &str)> {
        let s = s.trim_start();
        if let Some(mut rest) = s.strip_prefix('[') {
            let mut items = Vec::new();
            loop {
                let (v, r) = Val::parse(rest)?;
                items.push(v);
                let r = r.trim_start();
                if let Some(r) = r.strip_prefix(',') {
                    rest = r;
                } else if let Some(r) = r.strip_prefix(']') {
                    return Ok((Val::List(items), r));
                } else {
                    bail!("a list item followed by `{}`", r.chars().take(20).collect::<String>());
                }
            }
        }
        let end = s.find(|c: char| !(c.is_ascii_digit() || c == '-')).unwrap_or(s.len());
        Ok((Val::decimal(&s[..end])?, &s[end..]))
    }
}

/// `function <prefix><name>(<param>) { if (<param> == i) { return V; } ... return V; }`: the value per index and
/// the last, unconditional one.
#[derive(Debug)]
pub struct DataFn {
    pub at: BTreeMap<u64, Val>,
    pub otherwise: Val,
}

impl DataFn {
    pub fn value(&self, i: u64) -> &Val {
        self.at.get(&i).unwrap_or(&self.otherwise)
    }
}

/// The main's data functions whose name starts with `prefix`, by the rest of their name.
pub fn data_functions(main: &str, prefix: &str) -> Result<BTreeMap<String, DataFn>> {
    let mut out = BTreeMap::new();
    for line in main.lines().filter(|l| l.starts_with(&format!("function {prefix}"))) {
        let bad = || format!("data function `{}`", line.chars().take(80).collect::<String>());
        let rest = &line["function ".len() + prefix.len()..];
        let (name, rest) = rest.split_once('(').with_context(bad)?;
        let (param, mut body) = rest.split_once(") { ").with_context(bad)?;
        let mut at = BTreeMap::new();
        while let Some(r) = body.strip_prefix(&format!("if ({param} == ")) {
            let (i, r) = r.split_once(") { return ").with_context(bad)?;
            let (v, r) = Val::parse(r).with_context(bad)?;
            body = r.strip_prefix("; } ").with_context(bad)?;
            at.insert(i.parse::<u64>().with_context(bad)?, v);
        }
        let r = body.strip_prefix("return ").with_context(bad)?;
        let (otherwise, r) = Val::parse(r).with_context(bad)?;
        if r != "; }" {
            bail!("{}: trailing `{r}`", bad());
        }
        if out.insert(name.to_string(), DataFn { at, otherwise }).is_some() {
            bail!("{} is defined twice", bad());
        }
    }
    Ok(out)
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
    fn blocks_and_data() {
        let src = "function key_nBits(k) { if (k == 0) { return 4; } return 5; }\n\
                   function key_L(k) { return [1, -2, [3]]; }\n\
                   template A(n) {\n    signal input x;\n}\ntemplate parallel B() {\n}\n";
        let b = template_blocks(src).unwrap();
        assert_eq!(b.iter().map(|b| b.name).collect::<Vec<_>>(), ["A", "B"]);
        assert_eq!(b[1].text, "template parallel B() {\n}\n");
        let f = data_functions(src, "key_").unwrap();
        assert_eq!(f["nBits"].value(0), &Val::Int(4));
        assert_eq!(f["nBits"].value(1), &Val::Int(5));
        let l = Val::List(vec![Val::Int(1), Val::Int(-2), Val::List(vec![Val::Int(3)])]);
        assert_eq!(f["L"].value(7), &l);
    }

    #[test]
    fn a_bn128_root_is_a_big_value() {
        let r = "20608390218202144980062952588509845451522457362545788034308647633103217152625";
        assert_eq!(Val::decimal(r).unwrap(), Val::Big(r.to_string()));
        assert_eq!(Val::decimal(&format!("00{r}")).unwrap(), Val::Big(r.to_string()));
        assert_eq!(Val::decimal("7").unwrap(), Val::Int(7));
        assert!(Val::decimal("1x").is_err() && Val::decimal("").is_err());
    }
}
