//! The STARK steps: pil_info and the starkinfo/verifierinfo/expressionsinfo/bin files, the const tree,
//! the Rust verifier.

use std::fs;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use pil_info::core::dag::build::build_dag;
use pil_info::core::dag::format::ConstraintDag;
use pil_info::core::from_pilout::from_pilout;
use pil_info::core::im::ImChoice;
use pil_info::core::ir::ConstraintSystem;
use pil_info::{FieldCfg, Opening};

use super::steps::{path_str, write_exec};
use super::{AirConstraints, AirShape, PipelineCtx};
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

    // The system as the kernel reads it (the cheap part of pil_info: no im search).
    let cs = from_pilout(&pilout, ag, ai, &FieldCfg::goldilocks(), Opening::Deep)
        .map_err(|e| anyhow::anyhow!("{}: {e}", ctx.label))?;
    let constraints;
    if reuse {
        let r = ctx.reuse.clone().context("StarkTail { reuse } without a setup to reuse")?;
        // The reused starkinfo encodes another air's constraints: this air's DAG, built by the kernel with the
        // reused im choice, and its shape must be the same.
        constraints = air_constraints(&cs, &r.constraints.im).with_context(|| ctx.label.clone())?;
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
        constraints = air_constraints(&cs, &r.im).with_context(|| ctx.label.clone())?;
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

/// The kernel's DAG of `cs` with `im` (root labels, the source air's debug lines, cleared) and its shape.
fn air_constraints(cs: &ConstraintSystem, im: &ImChoice) -> Result<AirConstraints> {
    let mut dag: ConstraintDag = build_dag(cs, im).map_err(|e| anyhow::anyhow!("constraint DAG: {e:?}"))?;
    dag.roots.iter_mut().for_each(|r| r.label.clear());
    Ok(AirConstraints { dag, im: im.clone(), shape: air_shape(cs) })
}

fn air_shape(cs: &ConstraintSystem) -> AirShape {
    let stages = |v: &[pil_info::core::ir::NamedValue]| v.iter().map(|x| x.stage).collect();
    AirShape {
        n_rows_bits: cs.n_rows_bits,
        n_stages: cs.n_stages,
        witness: cs.witness.iter().map(|c| (c.stage, c.dim)).collect(),
        n_fixed: cs.fixed.len(),
        custom: (cs.custom_commits.iter())
            .map(|c| (c.name.clone(), c.public_values.clone(), c.stage_widths.clone()))
            .collect(),
        challenges: cs.challenges.clone(),
        n_publics: cs.publics.len(),
        proof_values: stages(&cs.proof_values),
        air_values: stages(&cs.air_values),
        airgroup_values: cs.airgroup_values.iter().map(|v| (v.stage, v.agg_type)).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::test_support::common::random_pilout;
    use pil2_pilout::pilout::{constraint, PilOut};

    fn constraints_of(p: &PilOut, im: &ImChoice) -> AirConstraints {
        air_constraints(&from_pilout(p, 0, 0, &FieldCfg::goldilocks(), Opening::Deep).unwrap(), im).unwrap()
    }

    /// Debug lines aside, a reusing air must have the reused air's DAG (the same im choice) and shape.
    #[test]
    fn reuse_needs_the_same_dag_and_shape() {
        let mut checked = 0;
        for seed in 0..40 {
            let p = random_pilout(seed).0;
            let Ok((_, im, _)) = pil_info::pipeline::search(&p, 0, 0, &pil_info::PilInfoCfg::goldilocks(2)) else {
                continue;
            };
            let reused = constraints_of(&p, &im);

            let mut relabelled = p.clone();
            for c in &mut relabelled.air_groups[0].airs[0].constraints {
                match c.constraint.as_mut() {
                    Some(constraint::Constraint::EveryRow(x)) => x.debug_line = Some("other.pil:1".into()),
                    Some(constraint::Constraint::FirstRow(x)) => x.debug_line = Some("other.pil:2".into()),
                    Some(constraint::Constraint::LastRow(x)) => x.debug_line = Some("other.pil:3".into()),
                    Some(constraint::Constraint::EveryFrame(x)) => x.debug_line = Some("other.pil:4".into()),
                    None => {}
                }
            }
            assert_eq!(constraints_of(&relabelled, &im), reused, "seed {seed}: debug lines do not count");

            let mut more = p.clone();
            let first = more.air_groups[0].airs[0].constraints[0].clone();
            more.air_groups[0].airs[0].constraints.push(first);
            assert_ne!(constraints_of(&more, &im), reused, "seed {seed}: one more constraint");

            // the shape: one more public, one more challenge
            let cs = from_pilout(&p, 0, 0, &FieldCfg::goldilocks(), Opening::Deep).unwrap();
            let mut publics = cs.clone();
            publics.publics.push(Default::default());
            assert_ne!(air_constraints(&publics, &im).unwrap(), reused, "seed {seed}: one more public");
            let mut stages = cs;
            stages.challenges.push((9, 0, "late".into()));
            assert_ne!(air_constraints(&stages, &im).unwrap(), reused, "seed {seed}: one more challenge");
            checked += 1;
        }
        assert!(checked >= 10, "only {checked} random pilouts searched");
    }
}
