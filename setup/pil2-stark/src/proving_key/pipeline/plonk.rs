//! The plonk steps: r1cs to plonk rows, the fixed columns and PIL, the pil2com compile, the trusted checker.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use pil2_pilout::pilout_proxy::PilOutProxy;
use pil2_stark_recurser::plonk2pil::{self, r1cs::types::PlonkOptions, SetupKind};

use super::steps::{path_str, write_exec};
use super::PipelineCtx;
use crate::io::fixed_cols;

pub(super) fn plonk2pil_step(ctx: &mut PipelineCtx, kind: SetupKind, opts: &PlonkOptions) -> Result<()> {
    let stem = ctx.compiled.as_deref().context("plonk2pil before the circom compile")?;
    let r1cs = ctx.build_dir.join("build").join(format!("{stem}.r1cs"));
    let data = fs::read(&r1cs).with_context(|| format!("Failed to read R1CS: {}", r1cs.display()))?;
    let r = plonk2pil::plonk2pil(&data, kind, opts).with_context(|| format!("plonk2pil failed for {}", ctx.circuit))?;
    ctx.plonk = Some(r);
    Ok(())
}

pub(super) fn compile_pil_step(ctx: &mut PipelineCtx) -> Result<()> {
    let pil = ctx.build_dir.join("pil").join(format!("{}.pil", ctx.circuit));
    let pilout = ctx.build_dir.join("build").join(format!("{}.pilout", ctx.circuit));
    let includes = [ctx.tools.std_pil_path.clone(), ctx.tools.recurser_pil_path.clone()];
    ctx.pilout = Some(Arc::new(compile_plonk_pil(&ctx.circuit, &pil, &pilout, &includes, None)?));
    ctx.air = (0, 0);
    Ok(())
}

pub(super) fn check_plonk(ctx: &mut PipelineCtx) -> Result<()> {
    let stem = ctx.compiled.as_deref().context("check_plonk before the circom compile")?;
    let r1cs = fs::read(ctx.build_dir.join("build").join(format!("{stem}.r1cs")))?;
    check_plonk_result(&ctx.circuit, &r1cs, ctx.plonk()?, ctx.pilout()?.as_ref())
}

/// pil2com over `field` (Goldilocks if `None`) of plonk2pil's PIL `pil` into `pilout`, loaded.
pub(crate) fn compile_plonk_pil(
    circuit: &str,
    pil: &Path,
    pilout: &Path,
    includes: &[String],
    field: Option<&str>,
) -> Result<pil2_pilout::pilout::PilOut> {
    crate::commands::compile_pil::run_compile_pil(&crate::commands::compile_pil::CompilePilOptions {
        pil_path: path_str(pil)?.to_string(),
        output_path: path_str(pilout)?.to_string(),
        include_paths: includes.to_vec(),
        field: field.map(str::to_string),
        ..Default::default()
    })?;
    let proxy = PilOutProxy::new(path_str(pilout)?)
        .map_err(|e| anyhow::anyhow!("Failed to load pilout {}: {e}", pilout.display()))?;
    if proxy.pilout.air_groups.first().is_none_or(|g| g.airs.is_empty()) {
        bail!("{circuit} pilout has no AIR groups: {}", pilout.display());
    }
    Ok(proxy.pilout)
}

/// The trusted checker over plonk2pil's in-memory result (`WriteFixed` and the `.const` writer are trusted).
pub(crate) fn check_plonk_result<V: plonk2pil::FixedValue>(
    circuit: &str,
    r1cs: &[u8],
    res: &plonk2pil::PlonkResult<V>,
    pilout: &pil2_pilout::pilout::PilOut,
) -> Result<()> {
    let report = plonk2pil::check::load::check_result(r1cs, res, pilout)
        .with_context(|| format!("{circuit}: the plonk layout does not implement its r1cs"))?;
    tracing::info!("{circuit} plonk check OK: {report}");
    Ok(())
}

/// fixed.bin and `.pil` in `build/` and `pil/`, the exec buffer in the files dir.
pub(super) fn write_plonk_fixed(ctx: &mut PipelineCtx) -> Result<()> {
    let p = ctx.plonk()?;
    let fixed_info: Vec<(String, Vec<u32>, Vec<u64>)> =
        p.fixed_pols.iter().map(|fp| (fp.name.clone(), vec![fp.index as u32], fp.values.clone())).collect();
    let fixed_bin = ctx.build_dir.join("build").join(format!("{}.fixed.bin", ctx.circuit));
    fixed_cols::write_fixed_pols_bin(
        path_str(&fixed_bin)?,
        &p.airgroup_name,
        &p.air_name,
        1u64 << p.n_bits,
        &fixed_info,
    )?;
    fs::write(ctx.build_dir.join("pil").join(format!("{}.pil", ctx.circuit)), &p.pil_str)?;
    write_exec(&ctx.file("exec"), &p.exec)
}

pub(super) fn write_user_fixed_const(ctx: &mut PipelineCtx) -> Result<()> {
    let fixed = ctx.fixed_dir.as_ref().map(|d| d.join(format!("{}.fixed", ctx.name)));
    write_basic_const(ctx, fixed)
}

/// A basic air's `.const`: `fixed_file` when it exists, else the pilout's inline columns. An air
/// with no fixed columns has none; one whose columns neither source supplies is an error.
pub(super) fn write_basic_const(ctx: &mut PipelineCtx, fixed_file: Option<std::path::PathBuf>) -> Result<()> {
    fs::create_dir_all(&ctx.files_dir)?;
    let pilout = ctx.pilout()?;
    let air = &pilout.air_groups[ctx.air.0].airs[ctx.air.1];
    let const_path = ctx.file("const");
    // A run that cannot produce `.const` drops an earlier run's, or the verkey comes from stale columns.
    let drop_stale_const = |why: &str| -> Result<()> {
        let mut dropped = false;
        for stale in [const_path.clone(), ctx.file("verkey.json"), ctx.file("verkey.bin")] {
            if stale.exists() {
                fs::remove_file(&stale).with_context(|| format!("removing stale {} ({why})", stale.display()))?;
                dropped = true;
            }
        }
        if dropped {
            tracing::warn!("Air '{}': dropped stale .const/.verkey in {} — {why}", ctx.name, ctx.files_dir.display());
        }
        Ok(())
    };
    if let Some(src) = fixed_file.as_ref().filter(|src| src.exists()) {
        fs::copy(src, &const_path)?;
    } else if air.fixed_cols.is_empty() {
        tracing::debug!("Air '{}': no fixed columns — skipping .const generation", ctx.name);
        drop_stale_const("the air has no fixed columns")?;
    } else if air.fixed_cols.iter().all(|fc| !fc.values.is_empty()) {
        fixed_cols::write_const_file(path_str(&const_path)?, air, &[])?;
        tracing::info!(
            "Generated .const from pilout inline fixed_cols for air '{}' ({} columns, {} rows)",
            ctx.name,
            air.fixed_cols.len(),
            air.num_rows.unwrap_or(0)
        );
    } else {
        drop_stale_const("its fixed columns have no source")?;
        match fixed_file {
            Some(src) => bail!(
                "Air '{}': fixed file not found: {} (its fixed columns are not inline in the pilout)",
                ctx.name,
                src.display()
            ),
            None => bail!(
                "Air '{}': some fixed columns have no inline values in the pilout; \
                 provide --fixed-dir (-u) with its .fixed file",
                ctx.name
            ),
        }
    }
    Ok(())
}
