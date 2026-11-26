//! The circom steps: the verifier and main circoms, their compile, the witness library.

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use pil2_stark_recurser::stark2circom::{gen_circom_circuit, CircuitTemplate, GenCircomCircuitInput};

use super::steps::{path_str, CircomOpt, Prime, VerifierGen};
use super::PipelineCtx;

/// The argument list of the `circom` invocation (no process spawned).
pub fn circom_args(opt: CircomOpt, prime: Prime, libs: &[String], circom_file: &Path, out_dir: &Path) -> Vec<String> {
    let field = match prime {
        Prime::Goldilocks => &["--prime", "goldilocks"][..],
        Prime::Bn128 => &["--prime", "bn128", "--inspect", "--wasm"][..],
    };
    let mut a: Vec<String> =
        [opt.flag(), "--r1cs"].iter().chain(field).chain(&["--c", "--verbose"]).map(|s| s.to_string()).collect();
    for l in libs {
        a.extend(["-l".to_string(), l.clone()]);
    }
    a.extend([circom_file.to_string_lossy().into_owned(), "-o".into(), out_dir.to_string_lossy().into_owned()]);
    a
}

pub(super) fn circom_prologue(
    ctx: &mut PipelineCtx,
    verifier: Option<&VerifierGen>,
    template: Option<CircuitTemplate>,
    opt: CircomOpt,
    prime: Prime,
) -> Result<()> {
    let circom_dir = ctx.build_dir.join("circom");
    let build = ctx.build_dir.join("build");
    for d in [&circom_dir, &build, &ctx.build_dir.join("pil"), &ctx.files_dir] {
        fs::create_dir_all(d)?;
    }
    // Each verifier's circom goes to its file, its data to the main.
    let mut keys = Vec::new();
    if let Some(gen) = verifier {
        let mut gen = gen.clone();
        // A batched circuit switches its slots off through `enable`.
        gen.opts_mut().enable_input |= ctx.batch_size > 1;
        for v in &ctx.verifiers {
            let key = gen
                .generate(&v.const_root, &v.stark_info, &v.verifier_info)
                .with_context(|| format!("verifier {} of {}", v.file, ctx.circuit))?;
            fs::write(circom_dir.join(&v.file), key.circom)?;
            keys.push(key.data);
        }
    }
    let circom_file = match template {
        Some(template) => {
            let input = ctx.circuit_input.as_ref().context("circuit circom without its input")?;
            let mut options = input.opts.clone();
            options.batch_size = ctx.batch_size.max(1);
            let verifier_filenames: Vec<String> = ctx.verifiers.iter().map(|v| v.file.clone()).collect();
            let src = gen_circom_circuit(&GenCircomCircuitInput {
                template,
                stark_infos: &input.stark_infos,
                vadcop_info: &input.vadcop_info,
                verifier_filenames: &verifier_filenames,
                keys: &keys,
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
        .args(circom_args(opt, prime, &ctx.tools.libs, &circom_file, &build))
        .output()
        .with_context(|| format!("Failed to execute circom for {stem}"))?;
    if !out.status.success() {
        bail!("Circom compilation failed for {stem}: {}", String::from_utf8_lossy(&out.stderr));
    }
    let dat = build.join(format!("{stem}_cpp")).join(format!("{stem}.dat"));
    if dat.exists() {
        fs::copy(&dat, ctx.file("dat"))?;
    }
    ctx.compiled = Some(stem);
    Ok(())
}

pub(super) fn witness_lib(ctx: &mut PipelineCtx) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circom_args_carry_the_level_the_field_and_the_libs() {
        let libs = vec!["/l1".to_string(), "/l2".to_string()];
        let a = circom_args(CircomOpt::O1, Prime::Goldilocks, &libs, Path::new("c.circom"), Path::new("/o"));
        let b = circom_args(CircomOpt::O2, Prime::Goldilocks, &libs, Path::new("c.circom"), Path::new("/o"));
        assert!(a.contains(&"--O1".to_string()) && !a.contains(&"--O2".to_string()), "{a:?}");
        assert!(b.contains(&"--O2".to_string()) && !b.contains(&"--O1".to_string()), "{b:?}");
        for f in ["--r1cs", "--c", "--prime", "goldilocks", "-l", "/l1", "/l2", "c.circom", "-o", "/o"] {
            assert!(a.contains(&f.to_string()), "missing {f}: {a:?}");
        }
        let f = circom_args(CircomOpt::O1, Prime::Bn128, &libs, Path::new("c.circom"), Path::new("/o"));
        for x in ["bn128", "--inspect", "--wasm"] {
            assert!(f.contains(&x.to_string()), "missing {x}: {f:?}");
        }
    }
}
