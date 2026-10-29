//! Zi at ξ per boundary: `Zh` (everyRow) and `Zd{k}`, k = boundaryId−1; circom in `tera/domain_zi_*`.

use anyhow::{bail, Result};
use serde_json::{json, Value};
use tera::Tera;

use super::gl_field::{gl_exp, GL_W};

pub(crate) const FRAGMENT_GL: &str = include_str!("tera/domain_zi_gl.circom.tera");
pub(crate) const FRAGMENT_BN128: &str = include_str!("tera/domain_zi_bn128.circom.tera");

/// The `domains` Tera context: one entry per non-everyRow boundary.
pub(crate) fn domains_ctx(boundaries: &[Value], n_bits: usize) -> Result<Vec<Value>> {
    if let Some(b0) = boundaries.first() {
        if b0["name"].as_str() != Some("everyRow") {
            bail!("boundary 0 must be everyRow, got {b0}");
        }
    }
    boundaries
        .iter()
        .enumerate()
        .skip(1)
        .map(|(b, d)| {
            if d["name"].as_str() != Some("domain") {
                bail!("boundary {b}: expected a domain, got {d}");
            }
            let c = d["cycleBits"].as_u64().ok_or_else(|| anyhow::anyhow!("boundary {b}: no cycleBits"))? as usize;
            if c > n_bits {
                bail!("boundary {b}: cycleBits {c} > nBits {n_bits}");
            }
            let offsets: Vec<u64> = d["offsets"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("boundary {b}: no offsets"))?
                .iter()
                .map(|o| o.as_u64().ok_or_else(|| anyhow::anyhow!("boundary {b}: bad offset {o}")))
                .collect::<Result<_>>()?;
            if offsets.is_empty() || offsets.iter().any(|&o| o >> c != 0) {
                bail!("boundary {b}: offsets {offsets:?} empty or outside the 2^{c} cycle");
            }
            // zMul[i] = z^(2^(i+1)), so z^(2^(n_bits-c)) is zMul[n_bits-c-1].
            let zm = if c == n_bits { "challengeOut".to_string() } else { format!("zMul[{}]", n_bits - c - 1) };
            let factors: Vec<Value> = offsets
                .iter()
                .enumerate()
                .map(|(i, &o)| json!({ "i": i, "root": gl_exp(GL_W[c], o).to_string() }))
                .collect();
            Ok(json!({
                "k": b - 1,
                "zm": zm,
                "complement": d["complement"].as_bool().unwrap_or(false),
                "factors": factors,
                "last": offsets.len() - 1,
            }))
        })
        .collect()
}

/// Signal holding Zi for `boundary_id`.
pub(crate) fn zi_signal(boundaries: &[Value], boundary_id: usize) -> Result<String> {
    let b = boundaries.get(boundary_id).ok_or_else(|| anyhow::anyhow!("Zi: invalid boundaryId {boundary_id}"))?;
    match b["name"].as_str().unwrap_or("") {
        "everyRow" if boundary_id == 0 => Ok("Zh".into()),
        "domain" if boundary_id > 0 => Ok(format!("Zd{}", boundary_id - 1)),
        other => bail!("Zi: unexpected boundary '{other}' at index {boundary_id}"),
    }
}

/// Render a verifier template that does `{% include "domain_zi" %}`.
pub(crate) fn render_with_fragment(main: &str, fragment: &str, ctx: &tera::Context) -> Result<String> {
    let mut t = Tera::default();
    t.add_raw_templates(vec![("domain_zi", fragment), ("main", main)])
        .map_err(|e| anyhow::anyhow!("Tera parse error: {e:?}"))?;
    t.render("main", ctx).map_err(|e| anyhow::anyhow!("Tera render error: {e:?}"))
}

/// The fragment alone, for tests.
#[cfg(test)]
pub(crate) fn render_fragment(fragment: &str, boundaries: &[Value], n_bits: usize) -> Result<String> {
    let mut ctx = tera::Context::new();
    ctx.insert("n_bits", &n_bits);
    ctx.insert("domains", &domains_ctx(boundaries, n_bits)?);
    Tera::one_off(fragment, &ctx, false).map_err(|e| anyhow::anyhow!("Tera render error: {e:?}"))
}
