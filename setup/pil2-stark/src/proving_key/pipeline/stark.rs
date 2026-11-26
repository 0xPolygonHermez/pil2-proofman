//! The STARK steps: pil_info and the starkinfo/verifierinfo/expressionsinfo/bin files, the const tree,
//! the Rust verifier.

use std::fs;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use super::steps::{path_str, write_exec};
use super::PipelineCtx;
use crate::io::fixed_cols;
use crate::output::json::to_json_string;
use crate::proving_key::bctree;
use crate::proving_key::util::{exact_log2, require_const_file};
use crate::types::stark_info::{ExpressionsInfo, StarkInfo, VerifierInfo};

fn write_bin(ctx: &PipelineCtx, si: &StarkInfo, ei: &ExpressionsInfo) -> Result<()> {
    crate::io::bin_file::write_expressions_bin_file(path_str(&ctx.file("bin"))?, si, ei)
}

pub(super) fn stark_tail(ctx: &mut PipelineCtx, reuse: bool, starkinfo_only: bool) -> Result<()> {
    fs::create_dir_all(&ctx.files_dir)?;
    let pilout = ctx.pilout()?;
    let (ag, ai) = ctx.air;
    let air = &pilout.air_groups[ag].airs[ai];
    let starkinfo_path = ctx.file("starkinfo.json");

    let constraints = constraint_digest(air);
    if reuse {
        let r = ctx.reuse.clone().context("StarkTail { reuse } without a setup to reuse")?;
        // The reused starkinfo encodes another air's constraints: they must be this one's.
        if r.constraints != constraints {
            bail!("{}: its constraints differ from those of the setup it reuses", ctx.label);
        }
        let (si, vi, ei) = r.pil_info;
        let si_loaded = StarkInfo::from_json(&si)?;
        tracing::info!("{} low-degree test (reused): {}", ctx.label, si_loaded.stark_struct.low_degree_test.describe());
        let plonk_n_bits = ctx.plonk()?.n_bits;
        if si_loaded.stark_struct.n_bits != plonk_n_bits {
            bail!(
                "{} has 2^{} rows but the reused setup expects 2^{} — the recursive circuits are not \
                 uniform (query-count equalization failed?)",
                ctx.label,
                plonk_n_bits,
                si_loaded.stark_struct.n_bits,
            );
        }
        fs::write(&starkinfo_path, to_json_string(&si)?)?;
        if !starkinfo_only {
            fs::write(ctx.file("verifierinfo.json"), to_json_string(&vi)?)?;
            fs::write(ctx.file("expressionsinfo.json"), to_json_string(&ei)?)?;
            write_bin(ctx, &si_loaded, &ExpressionsInfo::from_json(&ei)?)?;
        }
        ctx.pil_info = Some((si, vi, ei));
    } else {
        let stark_struct = match ctx.stark_struct.clone() {
            Some(s) => s,
            None => {
                let rows = u64::from(air.num_rows.unwrap_or(0));
                let n_bits = if rows > 0 {
                    exact_log2(rows).with_context(|| format!("{} row count", ctx.circuit))?
                } else {
                    ctx.plonk()?.n_bits
                };
                crate::types::stark_struct::generate_stark_struct(&ctx.settings, n_bits, ctx.family)
            }
        };
        let r = crate::pil::stark_setup::stark_setup(&pilout, ag, ai, &stark_struct, &ctx.prepare, &ctx.key)?;
        tracing::info!("{} low-degree test: {}", ctx.label, r.starkinfo.stark_struct.low_degree_test.describe());
        let si_json = to_json_string(&r.starkinfo)?;
        fs::write(&starkinfo_path, &si_json)?;
        let si: Value = serde_json::from_str(&si_json)?;
        let (vi_code, ei_code) = (&r.pil_code.verifier_info, &r.pil_code.expressions_info);
        if !starkinfo_only {
            fs::write(ctx.file("verifierinfo.json"), to_json_string(vi_code)?)?;
            fs::write(ctx.file("expressionsinfo.json"), to_json_string(ei_code)?)?;
            write_bin(ctx, &StarkInfo::from_json(&si)?, &ExpressionsInfo::try_from(ei_code)?)?;
        }
        ctx.pil_info = Some((si, serde_json::to_value(vi_code)?, serde_json::to_value(ei_code)?));
        ctx.stats = Some((r.summary, r.im_pols_info));
    }

    ctx.constraints = Some(constraints);
    ctx.temp_starkinfo = starkinfo_only;

    // pil_info fixes nConstants, which may exceed plonk2pil's columns: the pilout fills the rest.
    if let Some(p) = &ctx.plonk {
        let values = fixed_cols::reorder_plonk_pols_for_pilout(&p.fixed_pols, &pilout.symbols, 0, 0);
        fixed_cols::write_const_file(path_str(&ctx.file("const"))?, air, &values)?;
        tracing::info!(
            "Wrote {} const file: {} cols ({} from plonk + {} from pilout), {} rows",
            ctx.name,
            air.fixed_cols.len(),
            p.fixed_pols.len(),
            air.fixed_cols.len().saturating_sub(p.fixed_pols.len()),
            1usize << p.n_bits
        );
    }
    Ok(())
}

pub(super) fn const_tree(ctx: &mut PipelineCtx, required: bool) -> Result<()> {
    let const_path = ctx.file("const");
    if !const_path.exists() && !required {
        return Ok(());
    }
    require_const_file(&ctx.name, &const_path)?;
    tracing::info!("Computing constant tree for {}...", ctx.name);
    let root = bctree::compute_const_tree(
        path_str(&const_path)?,
        path_str(&ctx.file("starkinfo.json"))?,
        path_str(&ctx.file("verkey.json"))?,
    )?;
    write_exec(&ctx.file("verkey.bin"), &root)?;
    if ctx.temp_starkinfo {
        // recursive1 keeps its starkinfo only for the const tree (it reuses the first air's).
        fs::remove_file(ctx.file("starkinfo.json"))?;
    }
    ctx.const_root = Some(root);
    Ok(())
}

pub(super) fn write_verifier_rs(ctx: &mut PipelineCtx) -> Result<()> {
    let (si, vi, _) = ctx.pil_info()?;
    crate::output::verifier::write_verifier_rust_file(
        path_str(&ctx.file("verifier.rs"))?,
        &StarkInfo::from_json(si)?,
        &VerifierInfo::from_json(vi)?,
        ctx.family,
    )
}

/// A digest of the air's constraint system: what a reused starkinfo encodes (fixed values excluded).
fn constraint_digest(air: &pil2_pilout::pilout::Air) -> u64 {
    use prost::Message;
    use std::hash::{Hash, Hasher};
    let shape = pil2_pilout::pilout::Air {
        num_rows: air.num_rows,
        stage_widths: air.stage_widths.clone(),
        expressions: air.expressions.clone(),
        constraints: air.constraints.iter().map(without_debug_line).collect(),
        air_values: air.air_values.clone(),
        custom_commits: air.custom_commits.clone(),
        ..Default::default()
    };
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (air.fixed_cols.len(), shape.encode_to_vec()).hash(&mut h);
    h.finish()
}

// debug_line names the source air, so equal constraint systems would otherwise hash apart
fn without_debug_line(c: &pil2_pilout::pilout::Constraint) -> pil2_pilout::pilout::Constraint {
    use pil2_pilout::pilout::constraint::Constraint as C;
    let mut c = c.clone();
    match c.constraint.as_mut() {
        Some(C::EveryRow(x)) => x.debug_line = None,
        Some(C::FirstRow(x)) => x.debug_line = None,
        Some(C::LastRow(x)) => x.debug_line = None,
        Some(C::EveryFrame(x)) => x.debug_line = None,
        None => {}
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constraint_digest_ignores_debug_lines_but_not_constraints() {
        use pil2_pilout::pilout::{constraint, Air, Constraint};
        let row = |idx: u32, line: &str| Constraint {
            constraint: Some(constraint::Constraint::EveryRow(constraint::EveryRow {
                expression_idx: Some(pil2_pilout::pilout::operand::Expression { idx }),
                debug_line: Some(line.to_string()),
            })),
        };
        let a = Air { constraints: vec![row(0, "a.pil:1")], ..Default::default() };
        let b = Air { constraints: vec![row(0, "b.pil:9")], ..Default::default() };
        let c = Air { constraints: vec![row(1, "a.pil:1")], ..Default::default() };
        assert_eq!(constraint_digest(&a), constraint_digest(&b));
        assert_ne!(constraint_digest(&a), constraint_digest(&c));
    }
}
