//! One variant per shared step; parameters live in the variant, behaviour in `run`.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use pil2_pilout::pilout_proxy::PilOutProxy;
use pil2_stark_recurser::plonk2pil::{self, r1cs_types::PlonkOptions};
use pil2_stark_recurser::stark2circom::{
    gen_circom_circuit, gen_stark_verifier, gen_stark_verifier_bn128_blake3, GenCircomCircuitInput,
    StarkVerifierOptions,
};
use serde_json::Value;

use super::PipelineCtx;
use crate::io::fixed_cols;
use crate::output::json::to_json_string;
use crate::proving_key::util::{exact_log2, require_const_file};
use crate::proving_key::{bctree, recursive::compile_pil};
use crate::types::stark_info::{ExpressionsInfo, StarkInfo, VerifierInfo};

/// circom optimisation level, a per-kind builder parameter (owner question 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircomOpt {
    O1,
    O2,
}

impl CircomOpt {
    fn flag(self) -> &'static str {
        match self {
            CircomOpt::O1 => "--O1",
            CircomOpt::O2 => "--O2",
        }
    }
}

/// Which verifier circom generator feeds the prologue.
#[derive(Debug, Clone)]
pub enum VerifierGen {
    /// `gen_stark_verifier`: Goldilocks verifier.
    Gl(StarkVerifierOptions),
    /// `gen_stark_verifier_bn128_blake3`: the final SNARK's verifier of a blake3 recursivef.
    Bn128Blake3(StarkVerifierOptions),
}

impl VerifierGen {
    pub fn opts_mut(&mut self) -> &mut StarkVerifierOptions {
        match self {
            VerifierGen::Gl(o) | VerifierGen::Bn128Blake3(o) => o,
        }
    }

    /// The verifier circom of a proof of (`stark_info`, `verifier_info`) under `const_root`.
    pub fn generate(&self, const_root: &[String; 4], stark_info: &Value, verifier_info: &Value) -> Result<String> {
        match self {
            VerifierGen::Gl(o) => {
                let root = (!o.verkey_input).then_some(const_root);
                gen_stark_verifier(root, stark_info, verifier_info, o).context("gen_stark_verifier failed")
            }
            VerifierGen::Bn128Blake3(o) => {
                let root = (!o.verkey_input).then_some(const_root);
                gen_stark_verifier_bn128_blake3(root, stark_info, verifier_info, o)
                    .context("gen_stark_verifier_bn128_blake3 failed")
            }
        }
    }
}

/// Where the `.const` file of the STARK tail comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstSource {
    /// plonk2pil's columns; `WriteFixed` writes fixed.bin/.pil/.exec, the tail the `.const`.
    Plonk,
    /// The pilout's inline fixed columns.
    Pilout,
    /// `<fixed_dir>/<name>.fixed` when present, else as `Pilout`.
    UserFixed,
}

pub type CustomFn = Box<dyn Fn(&mut PipelineCtx) -> Result<()>>;

pub enum Step {
    /// verifier circom(s), circuit circom (None: `ctx.circom_file` as is), `circom` compile, copy `.dat`.
    CircomPrologue {
        verifier: Option<VerifierGen>,
        circom_template: Option<&'static str>,
        opt: CircomOpt,
    },
    /// Background build of the compiled circuit's witness library.
    WitnessLib,
    Plonk2Pil {
        kind: &'static str,
        opts: PlonkOptions,
    },
    WriteFixed {
        source: ConstSource,
    },
    CompilePil,
    /// pil_info (or `reuse` of `ctx.reuse_pil_info`), starkinfo/verifierinfo/expressionsinfo JSON, `.bin`,
    /// plonk `.const`; `starkinfo_only` (recursive1) writes just the starkinfo the const tree reads.
    StarkTail {
        reuse: bool,
        starkinfo_only: bool,
    },
    /// `.const` -> const tree -> verkey; a missing `.const` fails when `required`.
    ConstTree {
        required: bool,
    },
    WriteVerifierRs,
    Custom {
        name: &'static str,
        run: CustomFn,
    },
}

impl Step {
    pub fn run(&self, ctx: &mut PipelineCtx) -> Result<()> {
        match self {
            Step::CircomPrologue { verifier, circom_template, opt } => {
                circom_prologue(ctx, verifier.as_ref(), *circom_template, *opt)
            }
            Step::WitnessLib => {
                let w = ctx.witness.as_ref().context("no witness tracker")?;
                let stem = ctx.compiled.as_deref().context("witness library before the circom compile")?;
                w.run_witness_library_generation(
                    path_str(&ctx.build_dir)?,
                    path_str(&ctx.files_dir)?,
                    stem,
                    &ctx.name,
                    &ctx.tools.circom_helpers_dir,
                );
                Ok(())
            }
            Step::Plonk2Pil { kind, opts } => {
                let stem = ctx.compiled.as_deref().context("plonk2pil before the circom compile")?;
                let r1cs = ctx.build_dir.join("build").join(format!("{stem}.r1cs"));
                let data = fs::read(&r1cs).with_context(|| format!("Failed to read R1CS: {}", r1cs.display()))?;
                let r = plonk2pil::plonk2pil(&data, kind, opts)
                    .with_context(|| format!("plonk2pil failed for {}", ctx.circuit))?;
                ctx.plonk = Some(r);
                Ok(())
            }
            Step::WriteFixed { source: ConstSource::Plonk } => write_plonk_fixed(ctx),
            Step::WriteFixed { source: ConstSource::Pilout } => write_basic_const(ctx, None),
            Step::WriteFixed { source: ConstSource::UserFixed } => {
                let fixed = ctx.fixed_dir.as_ref().map(|d| d.join(format!("{}.fixed", ctx.name)));
                write_basic_const(ctx, fixed)
            }
            Step::CompilePil => {
                let build = ctx.build_dir.join("build");
                let pil = ctx.build_dir.join("pil").join(format!("{}.pil", ctx.circuit));
                let pilout = build.join(format!("{}.pilout", ctx.circuit));
                compile_pil(
                    path_str(&pil)?,
                    path_str(&pilout)?,
                    &ctx.tools.std_pil_path,
                    &ctx.tools.recurser_pil_path,
                )?;
                let proxy = PilOutProxy::new(path_str(&pilout)?)
                    .map_err(|e| anyhow::anyhow!("Failed to load pilout {}: {e}", pilout.display()))?;
                if proxy.pilout.air_groups.is_empty() || proxy.pilout.air_groups[0].airs.is_empty() {
                    bail!("{} pilout has no AIR groups: {}", ctx.circuit, pilout.display());
                }
                ctx.pilout = Some(Arc::new(proxy.pilout));
                ctx.air = (0, 0);
                Ok(())
            }
            Step::StarkTail { reuse, starkinfo_only } => stark_tail(ctx, *reuse, *starkinfo_only),
            Step::ConstTree { required } => const_tree(ctx, *required),
            Step::WriteVerifierRs => {
                let (si, vi, _) = ctx.pil_info()?;
                crate::output::verifier::write_verifier_rust_file(
                    path_str(&ctx.file("verifier.rs"))?,
                    &StarkInfo::from_json(si)?,
                    &VerifierInfo::from_json(vi)?,
                    true,
                    ctx.hash(),
                )
            }
            Step::Custom { run, .. } => run(ctx),
        }
    }
}

pub(super) fn path_str(p: &Path) -> Result<&str> {
    p.to_str().with_context(|| format!("non-UTF-8 path {}", p.display()))
}

/// The argument list of the `circom` invocation (no process spawned).
pub fn circom_args(opt: CircomOpt, prime: &str, libs: &[String], circom_file: &Path, out_dir: &Path) -> Vec<String> {
    let mut a: Vec<String> =
        [opt.flag(), "--r1cs", "--prime", prime, "--c", "--verbose"].iter().map(|s| s.to_string()).collect();
    for l in libs {
        a.extend(["-l".to_string(), l.clone()]);
    }
    a.extend([circom_file.to_string_lossy().into_owned(), "-o".into(), out_dir.to_string_lossy().into_owned()]);
    a
}

/// The error of a failed `circom` run; names the circuit.
pub fn circom_failure(name: &str, stderr: &str) -> anyhow::Error {
    anyhow::anyhow!("Circom compilation failed for {name}: {stderr}")
}

/// File names (relative to the files dir) the STARK tail + const tree write for circuit `name`.
#[cfg(test)]
pub fn tail_output_files(name: &str, _reuse: bool) -> Vec<String> {
    ["starkinfo.json", "verifierinfo.json", "expressionsinfo.json", "bin", "const", "verkey.json", "verkey.bin"]
        .iter()
        .map(|e| format!("{name}.{e}"))
        .collect()
}

fn circom_prologue(
    ctx: &mut PipelineCtx,
    verifier: Option<&VerifierGen>,
    template: Option<&'static str>,
    opt: CircomOpt,
) -> Result<()> {
    let circom_dir = ctx.build_dir.join("circom");
    let build = ctx.build_dir.join("build");
    for d in [&circom_dir, &build, &ctx.build_dir.join("pil"), &ctx.files_dir] {
        fs::create_dir_all(d)?;
    }
    if let Some(gen) = verifier {
        let mut gen = gen.clone();
        // A batched circuit switches its slots off through `enable`.
        gen.opts_mut().enable_input |= ctx.batch_size > 1;
        for v in &ctx.verifiers {
            let src = gen
                .generate(&v.const_root, &v.stark_info, &v.verifier_info)
                .with_context(|| format!("verifier {} of {}", v.file, ctx.circuit))?;
            fs::write(circom_dir.join(&v.file), src)?;
        }
    }
    let circom_file = match template {
        Some(template_name) => {
            let input = ctx.circuit_input.as_ref().context("circuit circom without its input")?;
            let mut options = input.opts.clone();
            options.batch_size = ctx.batch_size.max(1);
            let verifier_filenames: Vec<String> = ctx.verifiers.iter().map(|v| v.file.clone()).collect();
            let src = gen_circom_circuit(&GenCircomCircuitInput {
                template_name,
                stark_infos: &input.stark_infos,
                vadcop_info: &input.vadcop_info,
                verifier_filenames: &verifier_filenames,
                basic_vk: &input.basic_vk,
                agg_vk: &input.agg_vk,
                publics: &input.publics,
                options: &options,
            })
            .with_context(|| format!("gen_circom failed for {}", ctx.circuit))?;
            let p = circom_dir.join(format!("{}.circom", ctx.circuit));
            fs::write(&p, src)?;
            p
        }
        None => ctx.circom_file.clone().context("no circom file to compile")?,
    };
    let stem = circom_file.file_stem().and_then(|s| s.to_str()).context("circom file has no stem")?.to_string();

    tracing::info!("Compiling {stem}...");
    let out = std::process::Command::new(&ctx.tools.circom_exec)
        .args(circom_args(opt, "goldilocks", &ctx.tools.libs, &circom_file, &build))
        .output()
        .with_context(|| format!("Failed to execute circom for {stem}"))?;
    if !out.status.success() {
        return Err(circom_failure(&stem, &String::from_utf8_lossy(&out.stderr)));
    }
    let dat = build.join(format!("{stem}_cpp")).join(format!("{stem}.dat"));
    if dat.exists() {
        fs::copy(&dat, ctx.file("dat"))?;
    }
    ctx.compiled = Some(stem);
    Ok(())
}

/// fixed.bin and `.pil` in `build/` and `pil/`, the exec buffer in the files dir.
fn write_plonk_fixed(ctx: &mut PipelineCtx) -> Result<()> {
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
    let exec: Vec<u8> = p.exec.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(ctx.file("exec"), exec)?;
    Ok(())
}

/// A basic air's `.const`: `fixed_file` when it exists, else the pilout's inline columns. An air
/// with no fixed columns has none; one whose columns neither source supplies is an error.
fn write_basic_const(ctx: &mut PipelineCtx, fixed_file: Option<std::path::PathBuf>) -> Result<()> {
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

fn write_bin(ctx: &PipelineCtx, si: &StarkInfo, ei: &ExpressionsInfo) -> Result<()> {
    crate::io::bin_file::write_expressions_bin_file(path_str(&ctx.file("bin"))?, si, ei)
}

fn stark_tail(ctx: &mut PipelineCtx, reuse: bool, starkinfo_only: bool) -> Result<()> {
    fs::create_dir_all(&ctx.files_dir)?;
    let pilout = ctx.pilout()?;
    let (ag, ai) = ctx.air;
    let air = &pilout.air_groups[ag].airs[ai];
    let starkinfo_path = ctx.file("starkinfo.json");

    if reuse {
        let (si, vi, ei) = ctx.reuse_pil_info.clone().context("StarkTail { reuse } without a setup to reuse")?;
        let si_loaded = StarkInfo::from_json(&si)?;
        tracing::info!("{} low-degree test (reused): {}", ctx.label, si_loaded.stark_struct.low_degree_test.describe());
        // The reused setup assumes every air's circuit has the same shape.
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
        ctx.n_bits = Some(si_loaded.stark_struct.n_bits);
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
                crate::types::stark_struct::generate_stark_struct(&ctx.settings, n_bits, ctx.hash())
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
        ctx.n_bits = Some(stark_struct.n_bits);
        ctx.pil_info = Some((si, serde_json::to_value(vi_code)?, serde_json::to_value(ei_code)?));
        ctx.stats = Some((r.summary, r.im_pols_info));
    }

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

fn const_tree(ctx: &mut PipelineCtx, required: bool) -> Result<()> {
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
    let bin: Vec<u8> = root.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(ctx.file("verkey.bin"), bin)?;
    ctx.const_root = Some(root);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circom_args_o1_and_o2_differ_only_in_the_opt_flag() {
        let libs = vec!["/l1".to_string(), "/l2".to_string()];
        let a = circom_args(CircomOpt::O1, "goldilocks", &libs, Path::new("c.circom"), Path::new("/o"));
        let b = circom_args(CircomOpt::O2, "goldilocks", &libs, Path::new("c.circom"), Path::new("/o"));
        assert!(a.contains(&"--O1".to_string()) && !a.contains(&"--O2".to_string()), "{a:?}");
        assert!(b.contains(&"--O2".to_string()) && !b.contains(&"--O1".to_string()), "{b:?}");
        assert_eq!(
            a.iter().filter(|s| !s.starts_with("--O")).collect::<Vec<_>>(),
            b.iter().filter(|s| !s.starts_with("--O")).collect::<Vec<_>>()
        );
    }

    #[test]
    fn circom_args_keep_the_flags_every_copy_passes_today() {
        let libs = vec!["/l1".to_string(), "/l2".to_string()];
        let a = circom_args(CircomOpt::O1, "goldilocks", &libs, Path::new("c.circom"), Path::new("/o"));
        for f in ["--r1cs", "--c", "--prime", "goldilocks", "-l", "/l1", "/l2", "c.circom", "-o", "/o"] {
            assert!(a.contains(&f.to_string()), "missing {f}: {a:?}");
        }
    }

    #[test]
    fn compile_failure_message_names_the_circuit() {
        let m = circom_failure("vadcop_final", "boom on line 3").to_string();
        assert!(m.contains("Circom compilation failed for vadcop_final") && m.contains("boom on line 3"), "{m}");
    }

    #[test]
    fn stark_tail_writes_the_same_files_as_today() {
        let files = tail_output_files("Compressor", false);
        for ext in
            ["starkinfo.json", "verifierinfo.json", "expressionsinfo.json", "const", "verkey.json", "verkey.bin", "bin"]
        {
            assert!(files.contains(&format!("Compressor.{ext}")), "missing Compressor.{ext}: {files:?}");
        }
    }

    #[test]
    fn reuse_and_fresh_tail_write_identical_files() {
        let mut fresh = tail_output_files("Circ", false);
        let mut reused = tail_output_files("Circ", true);
        fresh.sort();
        reused.sort();
        assert_eq!(fresh, reused);
    }
}
