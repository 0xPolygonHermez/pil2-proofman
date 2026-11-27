//! Helpers shared by the setup commands.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use proofman_common::hash_family::HashFamily;
use serde_json::Value;

use crate::output::witness_gen::WitnessTracker;

/// A root of 4 limbs, each a u64 or a decimal string, as decimal strings.
pub(crate) fn parse_root_strings(json: &Value) -> Result<[String; 4]> {
    let arr = json.as_array().with_context(|| format!("a root is an array of 4 limbs, not {json}"))?;
    let limbs: Vec<String> = arr
        .iter()
        .map(|v| match v {
            Value::Number(n) if n.is_u64() => Ok(n.to_string()),
            Value::String(s) if s.parse::<u64>().is_ok() => Ok(s.clone()),
            _ => bail!("root limb {v} is not a u64"),
        })
        .collect::<Result<_>>()?;
    limbs.try_into().map_err(|l: Vec<String>| anyhow::anyhow!("a root has 4 limbs, not {}", l.len()))
}

/// Parse a verkey.json value (exactly 4 u64 limbs) into [u64;4].
pub(crate) fn parse_const_root(json: &Value) -> Result<[u64; 4]> {
    Ok(parse_root_strings(json)?.map(|s| s.parse().expect("checked u64")))
}

/// Resolve the circom executable path from env, well-known locations, or PATH.
pub fn resolve_circom_exec(circom_helpers_dir: &str) -> String {
    // Preferred binary name depends on the OS (mirrors JS logic)
    let bin_name = if cfg!(target_os = "macos") { "circom_mac" } else { "circom" };

    // Look inside circom_helpers_dir first
    let in_helpers = Path::new(circom_helpers_dir).join(bin_name);
    if in_helpers.is_file() {
        if let Ok(abs) = in_helpers.canonicalize() {
            return abs.to_string_lossy().to_string();
        }
        return in_helpers.to_string_lossy().to_string();
    }

    // Fall back to PATH
    bin_name.to_string()
}

/// Resolve a path from an environment variable; if not set, search for `fallback` by
/// checking (in order):
///   1. `<proofman-repo-root>/fallback`, where the root is captured at compile time
///      from `CARGO_MANIFEST_DIR` (this crate sits at `<root>/setup/pil2-stark`).
///      Works regardless of where the binary is invoked from, including when
///      proofman is consumed as a git dep from another workspace.
///   2. `fallback` relative to the current working directory
///   3. `fallback` relative to each ancestor directory of the running executable
///   4. `fallback` as a literal string (last resort / relative-path pass-through)
pub fn resolve_path_env(env_var: &str, fallback: &str) -> String {
    if let Ok(v) = std::env::var(env_var) {
        if !v.is_empty() {
            return v;
        }
    }
    // Compile-time path to the proofman repo root.
    const PROOFMAN_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let baked = std::path::Path::new(PROOFMAN_ROOT).join(fallback);
    if baked.exists() {
        if let Ok(abs) = baked.canonicalize() {
            return abs.to_string_lossy().into_owned();
        }
    }
    // CWD-relative
    let cwd_rel = std::path::Path::new(fallback);
    if cwd_rel.exists() {
        if let Ok(abs) = cwd_rel.canonicalize() {
            return abs.to_string_lossy().into_owned();
        }
    }
    // Walk up from the executable
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent();
        while let Some(d) = dir {
            let candidate = d.join(fallback);
            if candidate.exists() {
                if let Ok(abs) = candidate.canonicalize() {
                    return abs.to_string_lossy().into_owned();
                }
            }
            dir = d.parent();
        }
    }
    fallback.to_string()
}

/// The witness-library builder, with the Goldilocks sources `GOLDILOCKS_SRC_DIR` names.
pub(crate) fn witness_tracker_from_env() -> WitnessTracker {
    WitnessTracker::with_goldilocks_src(resolve_path_env("GOLDILOCKS_SRC_DIR", "pil2-stark/src/goldilocks/src"))
}

/// A recursive key's name and family (its globalInfo), which the post-setup commands build on.
pub(crate) struct RecursiveKey {
    pub build_dir: PathBuf,
    pub name: String,
    pub hash: HashFamily,
    pub global_info_path: PathBuf,
}

/// The vadcop_final artifacts of a [`RecursiveKey`].
pub(crate) struct VadcopFinal {
    pub const_root: [u64; 4],
    pub stark_info: Value,
    pub verifier_info: Value,
}

fn read_json(p: &Path) -> Result<Value> {
    let s =
        fs::read_to_string(p).with_context(|| format!("{} not found: run `setup --recursive` first", p.display()))?;
    serde_json::from_str(&s).with_context(|| format!("parsing {}", p.display()))
}

impl RecursiveKey {
    /// Loads its globalInfo from `build_dir` and registers its hash with the starks library (the const trees use it).
    pub fn load(build_dir: &str) -> Result<Self> {
        let global_info_path = Path::new(build_dir).join("provingKey").join("pilout.globalInfo.json");
        let global_info = read_json(&global_info_path)?;
        // From the setup that made the key: a default would let this stage disagree with the chain.
        let hash: HashFamily = global_info["hash"]
            .as_str()
            .with_context(|| format!("'hash' missing from {}; re-run `setup --recursive`", global_info_path.display()))?
            .parse()
            .map_err(anyhow::Error::msg)?;
        proofman_starks_lib_c::set_hash_family_c(hash.ffi_id());
        Ok(Self {
            build_dir: build_dir.into(),
            name: global_info["name"].as_str().unwrap_or("pilout").to_string(),
            hash,
            global_info_path,
        })
    }

    pub fn vadcop_final(&self) -> Result<VadcopFinal> {
        let dir = self.build_dir.join("provingKey").join(&self.name).join("vadcop_final");
        let root = read_json(&dir.join("vadcop_final.verkey.json"))?;
        Ok(VadcopFinal {
            const_root: parse_const_root(&root).context("Failed to parse vadcop_final.verkey.json")?,
            stark_info: read_json(&dir.join("vadcop_final.starkinfo.json"))?,
            verifier_info: read_json(&dir.join("vadcop_final.verifierinfo.json"))?,
        })
    }
}

/// Append the intermediate-polynomial section of a per-AIR stats entry, then the blank separator line.
pub(crate) fn push_im_pols_stats(lines: &mut Vec<String>, base_field: &[String], extended_field: &[String]) {
    if !base_field.is_empty() {
        lines.push("Intermediate polynomials baseField:".to_string());
        lines.extend(base_field.iter().map(|pol| format!("    {pol}")));
    }
    if !extended_field.is_empty() {
        lines.push("Intermediate polynomials extendedField:".to_string());
        lines.extend(extended_field.iter().map(|pol| format!("    {pol}")));
    }
    lines.push(String::new());
}

#[cfg(test)]
mod tests {
    use super::parse_root_strings;

    #[test]
    fn a_root_is_exactly_four_u64_limbs() {
        let ok = |s: &str| parse_root_strings(&serde_json::from_str(s).unwrap()).ok();
        assert_eq!(ok("[1,2,3,4]"), Some(["1", "2", "3", "4"].map(String::from)));
        assert_eq!(ok(r#"["1","2","3","18446744073709551615"]"#).unwrap()[3], "18446744073709551615");
        for bad in ["[1,2,3,4,5]", "[1,2]", r#"[1,"bad",3,4]"#, "[1,-2,3,4]", "[1,2.5,3,4]", r#"{"a":1}"#] {
            assert_eq!(ok(bad), None, "{bad}");
        }
    }
}
