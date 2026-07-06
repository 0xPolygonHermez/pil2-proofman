use clap::Parser;
use colored::Colorize;
use std::path::PathBuf;

use fields::{Field, Goldilocks, PrimeField64};
use proofman_common::{initialize_logger, GlobalInfo, ProofType};
use proofman_multilinear::{derive_global_challenges_for, verify_air, AirIr, Ext, MlProof};
use proofman_util::{timer_start_info, timer_stop_and_log_info};

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct VerifyMultilinearCmd {
    /// Multilinear proof file(s) (`.mlproof.bin`, as written by `prove --multilinear`).
    /// Pass the COMPLETE proof set of a run: the stage challenges are re-derived
    /// from every instance's stage-1 commitment, and the cross-instance global
    /// (bus) constraints are checked over the set.
    #[clap(short = 'p', long, num_args(1..), required = true)]
    pub proof: Vec<PathBuf>,

    /// Proving key folder (resolves each AIR's `.mlinfo.bin` and the global constraints)
    #[clap(short = 'k', long)]
    pub proving_key: PathBuf,

    /// Public inputs, as a JSON array of decimal field elements (overrides the
    /// publics carried in the proofs)
    #[clap(short = 'i', long)]
    pub public_inputs: Option<PathBuf>,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8,
}

type CliError = Box<dyn std::error::Error + Send + Sync>;

impl VerifyMultilinearCmd {
    pub fn run(&self) -> Result<(), CliError> {
        println!("{} VerifyMultilinear", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(self.verbose.into(), None);

        let global_info = GlobalInfo::new(&self.proving_key)?;

        // --- Load the proof set with their AIR IRs.
        struct Loaded {
            path: PathBuf,
            proof: MlProof,
            ir: AirIr,
        }
        let mut set: Vec<Loaded> = Vec::with_capacity(self.proof.len());
        for path in &self.proof {
            let proof = MlProof::load(path)?;
            let setup_path =
                global_info.get_air_setup_path(proof.airgroup_id as usize, proof.air_id as usize, &ProofType::Basic);
            let ir = AirIr::load(&setup_path.with_extension("mlinfo.bin"))?;
            set.push(Loaded { path: path.clone(), proof, ir });
        }

        // The instance order defines the challenge derivation; ids must be unique.
        set.sort_by_key(|l| l.proof.global_instance_id);
        if set.windows(2).any(|w| w[0].proof.global_instance_id == w[1].proof.global_instance_id) {
            return Err("duplicate global instance ids in the proof set".into());
        }

        // --- Publics: identical across the set (they are global inputs).
        let publics: Vec<Goldilocks> = match &self.public_inputs {
            Some(path) => {
                let data = std::fs::read_to_string(path)?;
                let values: Vec<serde_json::Value> = serde_json::from_str(&data)?;
                values
                    .iter()
                    .map(|v| {
                        let u = match v {
                            serde_json::Value::String(s) => s.parse::<u64>(),
                            serde_json::Value::Number(n) => Ok(n.as_u64().unwrap_or_default()),
                            _ => Ok(0),
                        }
                        .map_err(|e| format!("invalid public input {v}: {e}"))?;
                        Ok::<_, String>(Goldilocks::from_u64(u))
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
            None => set[0].proof.publics.iter().map(|v| Goldilocks::from_u64(v.as_canonical_u64())).collect(),
        };
        if set.iter().any(|l| l.proof.publics != set[0].proof.publics) {
            return Err("proofs carry different public inputs".into());
        }

        // --- Re-derive the global stage challenges from all stage-1 roots.
        let challenge_stages: Vec<u8> =
            set.iter().map(|l| l.ir.challenge_stages.clone()).max_by_key(|v| v.len()).unwrap_or_default();
        let max_n_stages = set.iter().map(|l| l.ir.n_stages()).max().unwrap_or(1);
        let stage1_roots: Vec<[Goldilocks; 4]> = set.iter().map(|l| l.proof.stage_roots[0]).collect();
        let expected = derive_global_challenges_for(&challenge_stages, max_n_stages, &stage1_roots);

        timer_start_info!(VERIFY_MULTILINEAR);
        for l in &set {
            let expected_air = &expected[..l.ir.challenge_stages.len().min(expected.len())];
            match verify_air(&l.ir, &l.proof, &publics, None, Some(expected_air)) {
                Ok(()) => {
                    println!(
                        "{}: {} {}",
                        l.path.display(),
                        format!("{: >2}", "\u{2713}").bright_green().bold(),
                        "Multilinear proof was verified".bright_green().bold()
                    );
                }
                Err(e) => {
                    println!(
                        "{}: {} {}",
                        l.path.display(),
                        format!("{: >2}", "\u{2717}").bright_red().bold(),
                        format!("Multilinear proof failed: {e}").bright_red().bold()
                    );
                    return Err(format!("proof {} did not verify: {e}", l.path.display()).into());
                }
            }
        }

        // --- Cross-instance global (bus) constraints over aggregated airgroup values.
        let agg = aggregate_airgroup_values(
            &global_info,
            set.iter().map(|l| (l.proof.airgroup_id as usize, l.proof.airgroup_values.as_slice())),
        )?;
        check_global_constraints(&self.proving_key, &agg, &publics)?;
        println!(
            "{} {}",
            format!("{: >12}", "Global").bright_green().bold(),
            "Cross-instance global constraints were verified".bright_green().bold()
        );
        timer_stop_and_log_info!(VERIFY_MULTILINEAR);

        Ok(())
    }
}

/// Aggregate each airgroup value over the proof set (aggType 0 = add, 1 = mul).
fn aggregate_airgroup_values<'a>(
    global_info: &GlobalInfo,
    values: impl Iterator<Item = (usize, &'a [Ext])>,
) -> Result<Vec<Vec<Ext>>, CliError> {
    let mut agg: Vec<Vec<Ext>> = global_info
        .agg_types
        .iter()
        .map(|group| group.iter().map(|t| if t.agg_type == 0 { Ext::zero() } else { Ext::one() }).collect::<Vec<_>>())
        .collect();

    for (airgroup_id, vals) in values {
        let types =
            global_info.agg_types.get(airgroup_id).ok_or_else(|| format!("airgroup {airgroup_id} out of range"))?;
        if vals.len() != types.len() {
            return Err(format!(
                "airgroup {airgroup_id}: proof carries {} airgroup values, globalInfo declares {}",
                vals.len(),
                types.len()
            )
            .into());
        }
        for (k, v) in vals.iter().enumerate() {
            if types[k].agg_type == 0 {
                agg[airgroup_id][k] += *v;
            } else {
                agg[airgroup_id][k] *= *v;
            }
        }
    }
    Ok(agg)
}

/// Evaluate `pilout.globalConstraints.json` over the aggregated airgroup
/// values and the publics; every constraint must evaluate to zero.
fn check_global_constraints(
    proving_key: &std::path::Path,
    agg: &[Vec<Ext>],
    publics: &[Goldilocks],
) -> Result<(), CliError> {
    let path = proving_key.join("pilout.globalConstraints.json");
    if !path.exists() {
        return Ok(()); // no global constraints emitted for this pilout
    }
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let constraints = match json.get("constraints").and_then(|c| c.as_array()) {
        Some(c) => c,
        None => return Ok(()),
    };

    let operand = |src: &serde_json::Value, tmps: &Vec<Ext>| -> Result<Ext, CliError> {
        let ty = src.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match ty {
            "tmp" => {
                let id = src["id"].as_u64().unwrap_or(0) as usize;
                Ok(tmps[id])
            }
            "airgroupvalue" => {
                let g = src["airgroupId"].as_u64().unwrap_or(0) as usize;
                let id = src["id"].as_u64().unwrap_or(0) as usize;
                agg.get(g)
                    .and_then(|v| v.get(id))
                    .copied()
                    .ok_or_else(|| format!("airgroupvalue ({g},{id}) out of range").into())
            }
            "public" => {
                let id = src["id"].as_u64().unwrap_or(0) as usize;
                let v = publics.get(id).copied().ok_or_else(|| format!("public {id} out of range"))?;
                Ok(Ext::from_array(&[v, Goldilocks::ZERO, Goldilocks::ZERO]))
            }
            "number" => {
                let raw = src["value"]
                    .as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| src["value"].as_u64().map(|v| v.to_string()).unwrap_or_else(|| "0".into()));
                let v = raw
                    .parse::<u128>()
                    .map(|v| (v % Goldilocks::ORDER_U64 as u128) as u64)
                    .map_err(|e| format!("bad number '{raw}': {e}"))?;
                Ok(Ext::from_array(&[Goldilocks::from_u64(v), Goldilocks::ZERO, Goldilocks::ZERO]))
            }
            other => Err(format!("unsupported global-constraint operand '{other}'").into()),
        }
    };

    for (idx, c) in constraints.iter().enumerate() {
        let n_tmps = c.get("tmpUsed").and_then(|t| t.as_u64()).unwrap_or(0) as usize;
        let mut tmps = vec![Ext::zero(); n_tmps.max(1)];
        let mut last_dest = 0usize;
        for instr in c.get("code").and_then(|v| v.as_array()).unwrap_or(&Vec::new()) {
            let op = instr.get("op").and_then(|o| o.as_str()).unwrap_or("");
            let srcs = instr.get("src").and_then(|s| s.as_array()).ok_or("missing src")?;
            let a = operand(&srcs[0], &tmps)?;
            let value = match op {
                "add" => a + operand(&srcs[1], &tmps)?,
                "sub" => a - operand(&srcs[1], &tmps)?,
                "mul" => a * operand(&srcs[1], &tmps)?,
                "copy" => a,
                other => return Err(format!("unsupported global-constraint op '{other}'").into()),
            };
            let dest = instr["dest"]["id"].as_u64().unwrap_or(0) as usize;
            if dest >= tmps.len() {
                tmps.resize(dest + 1, Ext::zero());
            }
            tmps[dest] = value;
            last_dest = dest;
        }
        let result = tmps[last_dest];
        if !result.is_zero() {
            let line = c.get("line").and_then(|l| l.as_str()).unwrap_or("?");
            return Err(format!("global constraint {idx} does not hold ({line}): value {result}").into());
        }
    }
    Ok(())
}
