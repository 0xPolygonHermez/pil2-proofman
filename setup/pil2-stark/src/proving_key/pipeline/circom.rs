//! The circom steps: the verifier and main circoms, their compile, the witness library.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use pil2_stark_recurser::stark2circom::check::witness::Circom;
use pil2_stark_recurser::stark2circom::check::data::unhashed_publics;
use pil2_stark_recurser::stark2circom::check::{check_circuit, Circuit, Field, Key, Vadcop};
use pil2_stark_recurser::stark2circom::{gen_circom_circuit, CircuitTemplate, GenCircomCircuitInput};

use super::steps::{path_str, Prime, VerifierGen};
use super::PipelineCtx;

/// The argument list of the `circom` invocation (no process spawned). Always `--O1`: `--O2` gives a
/// different r1cs from one run to the next for the same source, so the keys would not be reproducible.
pub fn circom_args(prime: Prime, libs: &[String], circom_file: &Path, out_dir: &Path) -> Vec<String> {
    let field = match prime {
        Prime::Goldilocks => &["--prime", "goldilocks"][..],
        Prime::Bn128 => &["--prime", "bn128", "--inspect", "--wasm"][..],
    };
    let mut a: Vec<String> =
        ["--O1", "--r1cs"].iter().chain(field).chain(&["--c", "--verbose"]).map(|s| s.to_string()).collect();
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
    prime: Prime,
) -> Result<()> {
    let circom_dir = ctx.build_dir.join("circom");
    let build = ctx.build_dir.join("build");
    for d in [&circom_dir, &build, &ctx.build_dir.join("pil"), &ctx.files_dir] {
        fs::create_dir_all(d)?;
    }
    // Each verifier's circom goes to its file, its data to the main.
    let mut keys = Vec::new();
    let gen = verifier.map(|gen| {
        let mut gen = gen.clone();
        // A batched circuit switches its slots off through `enable`.
        gen.opts_mut().enable_input |= ctx.batch_size > 1;
        gen
    });
    if let Some(gen) = &gen {
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
    if let (Some(gen), Some(template)) = (&gen, template) {
        check_templates(ctx, gen, template, prime, &circom_file)
            .with_context(|| format!("the generated circom of {} does not match its keys", ctx.circuit))?;
    }

    tracing::info!("Compiling {stem}...");
    let out = std::process::Command::new(&ctx.tools.circom_exec)
        .args(circom_args(prime, &ctx.tools.libs, &circom_file, &build))
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

/// The trusted check of what the prologue generated: each key's constraint template and data, and
/// vadcop_final's global constraints, against the starkinfo, verifierinfo and globalInfo they come from.
fn check_templates(
    ctx: &PipelineCtx,
    gen: &VerifierGen,
    template: CircuitTemplate,
    prime: Prime,
    circom_file: &Path,
) -> Result<()> {
    let opts = match gen {
        VerifierGen::Gl(o) | VerifierGen::Bn128Blake3(o) => o,
    };
    let circom_dir = ctx.build_dir.join("circom");
    let main = fs::read_to_string(circom_file)?;
    let files: Vec<PathBuf> = ctx.verifiers.iter().map(|v| circom_dir.join(&v.file)).collect();
    let keys = (ctx.verifiers.iter().zip(&files))
        .map(|(v, file)| Key {
            verifier_file: file,
            stark_info: &v.stark_info,
            verifier_info: &v.verifier_info,
            const_root: (!opts.verkey_input).then_some(&v.const_root),
            enable_input: opts.enable_input,
            input_challenges: opts.input_challenges,
        })
        .collect::<Vec<_>>();
    // the mains that carry globalInfo's data (the others' vadcop data is empty)
    let with_global_info = matches!(
        template,
        CircuitTemplate::Compressor
            | CircuitTemplate::Recursive1
            | CircuitTemplate::Recursive2
            | CircuitTemplate::VadcopFinal
    );
    let global_info = match &ctx.circuit_input {
        Some(input) if with_global_info => &input.vadcop_info,
        _ => &serde_json::Value::Null,
    };
    let input = ctx.circuit_input.as_ref().context("circuit circom without its input")?;
    let vadcop = match template {
        CircuitTemplate::Compressor | CircuitTemplate::Recursive1 => Vadcop::Air,
        CircuitTemplate::Recursive2 => {
            let airgroup = input.opts.airgroup_id.context("recursive2 without its airgroup")?;
            Vadcop::Airgroup { airgroup: airgroup as u64, recursive1: recursive1_keys(ctx, global_info, airgroup)? }
        }
        CircuitTemplate::VadcopFinal => {
            let n = global_info["air_groups"].as_array().map_or(0, Vec::len);
            Vadcop::Final {
                recursive1: (0..n).map(|ag| recursive1_keys(ctx, global_info, ag)).collect::<Result<_>>()?,
                recursive2: (0..n)
                    .map(|ag| verkey(&airgroup_dir(ctx, global_info, ag)?.join("recursive2/recursive2.verkey.json")))
                    .collect::<Result<_>>()?,
            }
        }
        CircuitTemplate::Final => {
            Vadcop::Snark { publics_info: input.publics.first().unwrap_or(&serde_json::Value::Null) }
        }
        _ => Vadcop::None,
    };
    if let Vadcop::Snark { publics_info } = &vadcop {
        let unhashed = unhashed_publics(publics_info);
        if !unhashed.is_empty() {
            tracing::warn!(
                "publics {unhashed:?} are in no publics_info definition: the final SNARK does not bind them"
            );
        }
    }
    let circuit = Circuit {
        field: match prime {
            Prime::Goldilocks => Field::Goldilocks,
            Prime::Bn128 => Field::Bn128,
        },
        main: &main,
        keys,
        global_info,
        vadcop,
    };
    let start = std::time::Instant::now();
    let circom = Circom { exec: &ctx.tools.circom_exec, libs: &ctx.tools.libs };
    let times = check_circuit(&circom, &ctx.build_dir.join("check").join(&ctx.circuit), &circuit)?;
    let each: Vec<String> = times.iter().map(|(what, t)| format!("{what} {:.1}s", t.as_secs_f64())).collect();
    tracing::info!(
        "Checked the generated circom of {} in {:.1}s ({})",
        ctx.circuit,
        start.elapsed().as_secs_f64(),
        each.join(", ")
    );
    Ok(())
}

/// `provingKey/<global name>/<airgroup name>` under the build dir.
fn airgroup_dir(ctx: &PipelineCtx, global_info: &serde_json::Value, airgroup: usize) -> Result<PathBuf> {
    let name = global_info["air_groups"][airgroup].as_str().with_context(|| format!("no airgroup {airgroup}"))?;
    let global = global_info["name"].as_str().unwrap_or("pilout");
    Ok(ctx.build_dir.join("provingKey").join(global).join(name))
}

fn verkey(path: &Path) -> Result<[u64; 4]> {
    let json = fs::read_to_string(path).with_context(|| path.display().to_string())?;
    crate::commands::common::parse_const_root(&serde_json::from_str(&json)?).with_context(|| path.display().to_string())
}

/// The recursive1 keys the setup wrote for `airgroup`'s airs, in globalInfo's order.
fn recursive1_keys(ctx: &PipelineCtx, global_info: &serde_json::Value, airgroup: usize) -> Result<Vec<[u64; 4]>> {
    let dir = airgroup_dir(ctx, global_info, airgroup)?;
    (global_info["airs"][airgroup].as_array().into_iter().flatten())
        .map(|air| {
            let name = air["name"].as_str().context("an air without a name")?;
            verkey(&dir.join("airs").join(name).join("recursive1/recursive1.verkey.json"))
        })
        .collect()
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
    fn circom_args_carry_o1_the_field_and_the_libs() {
        let libs = vec!["/l1".to_string(), "/l2".to_string()];
        let a = circom_args(Prime::Goldilocks, &libs, Path::new("c.circom"), Path::new("/o"));
        assert!(a.contains(&"--O1".to_string()) && !a.contains(&"--O2".to_string()), "{a:?}");
        for f in ["--r1cs", "--c", "--prime", "goldilocks", "-l", "/l1", "/l2", "c.circom", "-o", "/o"] {
            assert!(a.contains(&f.to_string()), "missing {f}: {a:?}");
        }
        let f = circom_args(Prime::Bn128, &libs, Path::new("c.circom"), Path::new("/o"));
        for x in ["bn128", "--inspect", "--wasm"] {
            assert!(f.contains(&x.to_string()), "missing {x}: {f:?}");
        }
    }

    /// The template check on fibonacci-square's airs as the setup builds them, all in one main: its pilout from
    /// `FIBONACCI_PILOUT`, else `examples/fibonacci-square/pil/build.pilout`; skipped without one.
    #[test]
    fn fibonacci_keys_pass_the_template_check() {
        use pil2_stark_recurser::stark2circom::{gen_stark_verifier, key_dispatch, StarkVerifierOptions};

        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let pilout = std::env::var("FIBONACCI_PILOUT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("examples/fibonacci-square/pil/build.pilout"));
        let circom = root.join("setup/circom/circom");
        if !pilout.is_file() || !circom.is_file() {
            eprintln!("skipped: no {} or {}", pilout.display(), circom.display());
            return;
        }
        let p = pil2_pilout::pilout_proxy::PilOutProxy::new(pilout.to_str().unwrap()).unwrap().pilout;
        let opts = StarkVerifierOptions { verkey_input: true, ..Default::default() };
        let (mut keys, mut datas) = (Vec::new(), Vec::new());
        for (ag, group) in p.air_groups.iter().enumerate() {
            for (air, a) in group.airs.iter().enumerate() {
                let n_bits = a.num_rows.unwrap_or(0).trailing_zeros() as usize;
                let settings = crate::types::stark_struct::StarkSettings::default();
                let ss = crate::types::stark_struct::generate_stark_struct(
                    &settings,
                    n_bits,
                    proofman_common::hash_family::HashFamily::Blake3,
                );
                let r =
                    crate::pil::stark_setup::stark_setup(&p, ag, air, &ss, &Default::default(), &Default::default())
                        .unwrap();
                let si = serde_json::to_value(&r.starkinfo).unwrap();
                let vi: serde_json::Value =
                    serde_json::from_str(&crate::output::json::to_json_string(&r.pil_code.verifier_info).unwrap())
                        .unwrap();
                let v = gen_stark_verifier(None, &si, &vi, &opts).unwrap();
                keys.push((format!("air{ag}_{air}.verifier.circom"), si, vi, v.circom));
                datas.push(v.data);
            }
        }
        let dir = std::env::temp_dir().join(format!("fibonacci_template_check_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let main = key_dispatch(&datas, true);
        for (file, _, _, circom) in &keys {
            fs::write(dir.join(file), circom).unwrap();
        }
        let files: Vec<PathBuf> = keys.iter().map(|k| dir.join(&k.0)).collect();
        let circuit = Circuit {
            field: Field::Goldilocks,
            main: &main,
            keys: (keys.iter().zip(&files))
                .map(|((_, si, vi, _), f)| Key {
                    verifier_file: f,
                    stark_info: si,
                    verifier_info: vi,
                    const_root: None,
                    enable_input: false,
                    input_challenges: false,
                })
                .collect(),
            global_info: &serde_json::Value::Null,
            vadcop: Vadcop::None,
        };
        let lib = root.join("setup/stark-recurser/stark2circom/circom_verifier");
        let libs = ["helper_circuits", "circuits.gl"].map(|l| lib.join(l).to_string_lossy().into_owned());
        let exec = circom.to_string_lossy();
        let r = check_circuit(&Circom { exec: &exec, libs: &libs }, &dir.join("work"), &circuit);
        let _ = fs::remove_dir_all(&dir);
        let times = r.unwrap_or_else(|e| panic!("{e:#}"));
        assert_eq!(times.len(), keys.len());
        eprintln!("{times:?}");
    }
}
