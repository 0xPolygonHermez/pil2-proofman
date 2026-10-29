//! Zi at ξ per boundary: `Zh` (everyRow) and `Zd{k}`, k = boundaryId−1; circom in `tera/domain_zi_*`.

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use tera::Tera;

use super::gl_field::{gl_exp, GL_W};

pub(crate) const FRAGMENT_GL: &str = include_str!("tera/domain_zi_gl.circom.tera");
pub(crate) const FRAGMENT_BN128: &str = include_str!("tera/domain_zi_bn128.circom.tera");

/// The `domains` Tera context, one entry per domain, after the native verifier's validation.
pub(crate) fn domains_ctx(boundaries: &[Value], n_bits: usize) -> Result<Vec<Value>> {
    let bs: Vec<proofman_common::Boundary> = serde_json::from_value(Value::Array(boundaries.to_vec()))?;
    proofman_common::validate_boundaries(&bs, n_bits as u32).map_err(|e| anyhow!("{e}"))?;
    Ok(bs[1..]
        .iter()
        .enumerate()
        .map(|(k, d)| {
            let c = d.cycle_bits as usize;
            // zMul[i] = z^(2^(i+1)), so z^(2^(n_bits-c)) is zMul[n_bits-c-1].
            let zm = if c == n_bits { "challengeOut".to_string() } else { format!("zMul[{}]", n_bits - c - 1) };
            let factors: Vec<Value> = d
                .offsets
                .iter()
                .enumerate()
                .map(|(i, &o)| json!({ "i": i, "root": gl_exp(GL_W[c], o).to_string() }))
                .collect();
            json!({ "k": k, "zm": zm, "complement": d.complement, "factors": factors, "last": d.offsets.len() - 1 })
        })
        .collect())
}

/// Signal holding Zi for `boundary_id`.
pub(crate) fn zi_signal(boundaries: &[Value], boundary_id: usize) -> Result<String> {
    if boundary_id >= boundaries.len() {
        bail!("Zi: invalid boundaryId {boundary_id}");
    }
    Ok(if boundary_id == 0 { "Zh".into() } else { format!("Zd{}", boundary_id - 1) })
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
