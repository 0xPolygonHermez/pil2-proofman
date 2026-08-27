use anyhow::{bail, Context, Result};
use proofman_starks_lib_c::compute_const_tree_c;

/// What the C++ side will require the const file to be, from the starkinfo it reads:
/// one Goldilocks element per constant per row.
fn expected_const_bytes(starkinfo: &serde_json::Value) -> Result<(u64, usize, u64)> {
    let n_bits = starkinfo["starkStruct"]["nBits"]
        .as_u64()
        .context("starkinfo has no starkStruct.nBits")?;
    let n_constants = starkinfo["nConstants"].as_u64().context("starkinfo has no nConstants")? as usize;
    Ok((n_bits, n_constants, n_constants as u64 * 8 * (1u64 << n_bits)))
}

/// Build the constant polynomial Merkle tree and return the 4-element root.
///
/// Delegates to the C++ `build_const_tree_c` in libstarks.a, which reads `const_path`, parses the
/// tree parameters from `starkinfo_path`, extends the polynomials, builds the Merkle tree for the
/// air's hash family, and writes the root as a JSON array to `verkey_path`.
///
/// The size agreement between the two files is checked HERE rather than left to the C++, because the
/// C++ mismatch path calls `exitProcess()`, which kills the whole process with status 255 -- and the
/// `zklog.error` it emits first goes nowhere in this build. Left to it, a recursive air whose const
/// file disagrees with its starkinfo terminates the setup with no diagnostic at all, from inside a
/// worker thread. It is a real condition, not a theoretical one: reusing one air's starkSetup for
/// another air is only sound while the two airs are the same size.
pub fn compute_const_tree(const_path: &str, starkinfo_path: &str, verkey_path: &str) -> Result<[u64; 4]> {
    let starkinfo: serde_json::Value = serde_json::from_slice(
        &std::fs::read(starkinfo_path).with_context(|| format!("Failed to read starkinfo {starkinfo_path}"))?,
    )
    .with_context(|| format!("Failed to parse starkinfo {starkinfo_path}"))?;
    let (n_bits, n_constants, expected) = expected_const_bytes(&starkinfo)?;

    let actual = std::fs::metadata(const_path)
        .with_context(|| format!("Failed to stat const file {const_path}"))?
        .len();
    if actual != expected {
        let rows = if n_constants > 0 { actual / (n_constants as u64 * 8) } else { 0 };
        bail!(
            "const file {const_path} is {actual} B but its starkinfo {starkinfo_path} describes \
             {n_constants} constants over 2^{n_bits} rows, which needs {expected} B. The file holds \
             {rows} rows. The const file and the starkinfo were produced for different air sizes -- \
             typically an air that reused another air's starkSetup while its own PIL compiled to a \
             different number of rows."
        );
    }

    Ok(compute_const_tree_c(const_path, starkinfo_path, "", verkey_path))
}
