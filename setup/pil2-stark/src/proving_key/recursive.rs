//! Port of `generateRecursiveSetup.js`: generate recursive verifier circuits,
//! compile them, and produce the proving key artifacts for recursive1,
//! recursive2, and compressor templates.
//!
//! The function drives:
//! 1. Circom verifier generation (pil2circom)
//! 2. Recursive circom generation (gencircom)
//! 3. Circom compilation to R1CS + C++
//! 4. R1CS-to-PIL conversion (plonk2pil)
//! 5. PIL compilation (pil2com JS compiler)
//! 6. starkSetup (pil_info)
//! 7. Constant tree computation (bctree)
//! 8. Binary file generation

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

/// Sentinel error returned when recursive1 detects that a compressor is required.
///
/// This is returned instead of a generic error so the caller can distinguish
/// "compressor needed" from a real failure and auto-retry.
#[derive(Debug)]
pub struct NeedsCompressorError {
    /// The n_bits value from plonk2pil that exceeded the 17-bit threshold.
    pub n_bits: usize,
}

impl std::fmt::Display for NeedsCompressorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Recursive1 circuit exceeds 17-bit threshold (n_bits={}); a compressor is needed", self.n_bits)
    }
}

impl std::error::Error for NeedsCompressorError {}

/// Sentinel error returned when a has-compressor recursive1 packs BELOW the shared
/// 2^(THRESHOLD-1) domain. The caller catches this, bumps the compressor's nQueries
/// (which enlarges recursive1's verifier), re-runs the compressor, and retries — so
/// every recursive1 in an airgroup lands at the same nBits and shares one setup.
/// Bailed BEFORE the const-tree build so the caller can resize before any const file
/// is written against a mismatched (reused) starkInfo.
#[derive(Debug)]
pub struct RecursiveTooSmallError {
    /// recursive1's n_bits from plonk2pil (below the threshold).
    pub n_bits: usize,
    /// recursive1's n_used (rows before padding) — drives the compressor nQueries bump.
    pub n_used: usize,
}

impl std::fmt::Display for RecursiveTooSmallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Recursive1 packs to 2^{} (n_used={}) below the shared recursive domain; \
             compressor nQueries must grow",
            self.n_bits, self.n_used
        )
    }
}

impl std::error::Error for RecursiveTooSmallError {}
use serde_json::Value;

use pilout::pilout_proxy::PilOutProxy;
use stark_recurser::plonk2pil::r1cs_types::PlonkOptions;
use stark_recurser::plonk2pil::{self, PlonkResult};

use crate::proving_key::bctree;
use crate::io::fixed_cols;
use crate::output::witness_gen::WitnessTracker;

/// Which recursive template to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecursiveTemplate {
    Compressor,
    Recursive1,
    Recursive2,
}

impl RecursiveTemplate {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Compressor => "compressor",
            Self::Recursive1 => "recursive1",
            Self::Recursive2 => "recursive2",
        }
    }

    pub fn tera_template(&self) -> &'static str {
        match self {
            Self::Compressor => "vadcop/compressor.circom.tera",
            Self::Recursive1 => "vadcop/recursive1.circom.tera",
            Self::Recursive2 => "vadcop/recursive2.circom.tera",
        }
    }

    pub fn ejs_template(&self) -> &'static str {
        match self {
            Self::Compressor => "src/vadcop/templates/compressor.circom.ejs",
            Self::Recursive1 => "src/vadcop/templates/recursive1.circom.ejs",
            Self::Recursive2 => "src/vadcop/templates/recursive2.circom.ejs",
        }
    }
}

/// Configuration for the recursive setup.
pub struct RecursiveSetupConfig<'a> {
    pub build_dir: &'a str,
    pub template: RecursiveTemplate,
    pub airgroup_name: &'a str,
    pub airgroup_id: usize,
    pub air_id: usize,
    pub air_name: &'a str,
    pub global_info: &'a Value,
    pub const_root: &'a [String; 4],
    pub verification_keys: &'a [Vec<Vec<String>>],
    pub stark_info: &'a Value,
    pub verifier_info: &'a Value,
    pub stark_struct: Option<&'a Value>,
    pub has_compressor: bool,
    pub hash: &'a str,

    /// When `Some`, the path to the original air's `{air_name}.starkinfo.json` on disk.
    /// Used by the A2 nQueries adjustment: if the recursive1 circuit is smaller than
    /// 2^RECURSIVE_BITS_THRESHOLD, the starkInfo is updated with `minimumQueriesRequired`
    /// and written back to this path so the change is persisted for re-runs.
    pub stark_info_path: Option<&'a std::path::Path>,

    /// When true, skip the (expensive) witness-library generation for this run and instead
    /// return its parameters in `RecursiveSetupResult::witness_lib_params`. Used by the
    /// compressor→recursive1 resize loop: intermediate compressor attempts that will be
    /// superseded by a nQueries bump would otherwise generate a witness lib that is thrown
    /// away. The caller generates the winning compressor's witness lib exactly once after
    /// the loop converges.
    pub defer_witness_lib: bool,

    /// Optional pre-computed pil_info result to reuse instead of running starkSetup again.
    /// When provided, the pil_info computation (starkInfo / verifierInfo / expressionsInfo)
    /// is skipped and these values are used directly — this mirrors the JS behaviour where
    /// `setupAggregation_` is passed in for the 2nd+ airs and for recursive2.
    ///
    /// Format: `(stark_info, verifier_info, expressions_info)` as JSON Values.
    pub existing_pil_info: Option<(Value, Value, Value)>,

    // Tool paths
    pub circom_exec: &'a str,
    pub circuits_gl_path: &'a str,
    pub recurser_circuits_path: &'a str,
    pub std_pil_path: &'a str,
    pub recurser_pil_path: &'a str,
    pub circom_helpers_dir: &'a str,
}

/// Result of a recursive setup step.
pub struct RecursiveSetupResult {
    /// The 4-element constant root.
    pub const_root: [u64; 4],
    /// Generated PIL source string.
    pub pil_str: String,
    /// The starkInfo JSON for this recursive circuit.
    pub stark_info: Option<Value>,
    /// The verifierInfo JSON for this recursive circuit.
    pub verifier_info: Option<Value>,
    /// The expressionsInfo JSON for this recursive circuit.
    pub expressions_info: Option<Value>,
    /// n_bits from plonk2pil (log2 of circuit rows).  Exposed so the caller
    /// can validate size without re-reading the starkInfo JSON.
    pub n_bits: usize,
    /// n_used from plonk2pil (rows actually used before power-of-2 padding).  Exposed
    /// so the caller can compute how much a compressor's nQueries must grow to fill a
    /// has-compressor recursive1 up to the shared 2^(THRESHOLD-1) target.
    pub n_used: usize,
    /// Set only when the config requested `defer_witness_lib`: the `(name_filename,
    /// files_dir)` needed to generate the witness library later, once this run is known
    /// to be the final (non-superseded) one. `None` when the witness lib was generated inline.
    pub witness_lib_params: Option<(String, String)>,
}

/// Run the recursive setup for a single air/template combination.
///
/// Ports `genRecursiveSetup()` from `generateRecursiveSetup.js`.
pub fn gen_recursive_setup(
    config: &RecursiveSetupConfig<'_>,
    witness_tracker: &WitnessTracker,
) -> Result<RecursiveSetupResult> {
    let template = config.template;
    let template_str = template.as_str();

    // Determine naming and paths based on template
    let (verifier_name, name_filename, files_dir, input_challenges, verkey_input, enable_input) =
        resolve_names_and_paths(config)?;

    let airgroup_pil_name = match template {
        RecursiveTemplate::Compressor => {
            format!("{}_{}_{}", config.airgroup_name, config.air_name, template_str)
        }
        RecursiveTemplate::Recursive1 => {
            format!("{}_{}_{}", config.airgroup_name, config.air_name, template_str)
        }
        RecursiveTemplate::Recursive2 => {
            // JS uses the recursive1 airgroup_pil_name as the recursive2 name:
            // "{airgroup_name}_{air_name}_recursive1"
            format!("{}_{}_{}", config.airgroup_name, config.air_name, "recursive1")
        }
    };

    // Create directories
    let circom_dir = PathBuf::from(config.build_dir).join("circom");
    let build_dir_path = PathBuf::from(config.build_dir).join("build");
    let pil_dir = PathBuf::from(config.build_dir).join("pil");
    fs::create_dir_all(&circom_dir)?;
    fs::create_dir_all(&build_dir_path)?;
    fs::create_dir_all(&pil_dir)?;
    fs::create_dir_all(&files_dir)?;

    // Prepare inputs that may change if an A2 nQueries adjustment is needed.
    let const_root_circuit: [String; 4] = if config.const_root.iter().all(|s| s.is_empty()) {
        ["0".to_string(), "0".to_string(), "0".to_string(), "0".to_string()]
    } else {
        config.const_root.clone()
    };
    let pil2circom_opts = crate::io::recurser::Pil2CircomOptions {
        skip_main: true,
        verkey_input,
        enable_input,
        input_challenges,
        hash: config.hash.to_string(),
    };
    let verifier_path = circom_dir.join(&verifier_name);
    let verifier_filenames = vec![verifier_name.clone()];
    let circom_out_path = circom_dir.join(format!("{}.circom", name_filename));
    let r1cs_path = build_dir_path.join(format!("{}.r1cs", name_filename));
    let dat_src = build_dir_path.join(format!("{}_cpp", name_filename)).join(format!("{}.dat", name_filename));
    let dat_dst = files_dir.join(format!("{}.dat", template_str));

    // For recursive2, the plonk2pil airgroup name (used in pilout/fixed.bin) is
    // hardcoded to "Recursive2" (matching JS), while airgroup_pil_name is still
    // "FiboCPU_FibonacciSquare_recursive1" for the starkinfo.json name field.
    let plonk_airgroup_name = match template {
        RecursiveTemplate::Recursive2 => "Recursive2".to_string(),
        _ => airgroup_pil_name.clone(),
    };
    let mut plonk_opts = PlonkOptions {
        airgroup_name: Some(plonk_airgroup_name),
        max_constraint_degree: None,
        hash_id: config.hash.to_string(),
        merge_copies: true,
    };
    if template == RecursiveTemplate::Compressor {
        plonk_opts.max_constraint_degree = Some(5);
    }
    let type_compressor = match template {
        RecursiveTemplate::Compressor => "compressor",
        _ => "aggregation",
    };

    // Macro-like helper: run pil2circom + gen_circom + compile + copy .dat + plonk2pil
    // for a given (possibly adjusted) stark_info.  We call this once; if the A2
    // nQueries adjustment is needed we call the relevant sub-steps again.
    let run_circom_and_plonk = |effective_si: &serde_json::Value| -> Result<PlonkResult> {
        // Generate verifier circom via pil2circom
        // Each template step generates its own verifier. For recursive1 with compressor,
        // the verifier is generated from the compressor's starkInfo/verifierInfo (passed
        // via config), and written to {air}_compressor.verifier.circom.
        let verifier_circom =
            crate::io::recurser::pil2circom(&const_root_circuit, effective_si, config.verifier_info, &pil2circom_opts)
                .context("pil2circom failed in recursive setup")?;
        fs::write(&verifier_path, &verifier_circom)?;

        // Generate recursive circom via gen_circom
        let gen_opts = crate::io::recurser::GenCircomOptions {
            airgroup_id: Some(config.airgroup_id as u64),
            has_compressor: config.has_compressor,
            ..Default::default()
        };
        let gen_input = crate::io::recurser::GenCircomInput {
            template_name: template.ejs_template(),
            stark_infos: std::slice::from_ref(effective_si),
            vadcop_info: config.global_info,
            verifier_filenames: &verifier_filenames,
            basic_verification_keys: config.verification_keys,
            agg_verification_keys: &[],
            publics: &[],
            options: &gen_opts,
        };
        let circom_str = crate::io::recurser::gen_circom(&gen_input).context("gen_circom failed in recursive setup")?;
        fs::write(&circom_out_path, &circom_str)?;

        // Compile circom
        tracing::info!("Compiling {}...", name_filename);
        let compile_status = std::process::Command::new(config.circom_exec)
            .args([
                "--O2",
                "--r1cs",
                "--prime",
                "goldilocks",
                "--c",
                "--verbose",
                "-l",
                config.recurser_circuits_path,
                "-l",
                config.circuits_gl_path,
            ])
            .arg(circom_out_path.to_str().unwrap())
            .arg("-o")
            .arg(build_dir_path.to_str().unwrap())
            .output()
            .context("Failed to execute circom compiler")?;
        if !compile_status.status.success() {
            let stderr = String::from_utf8_lossy(&compile_status.stderr);
            bail!("Circom compilation failed for {}: {}", name_filename, stderr);
        }

        // Copy .dat file
        tracing::info!("Copying circom files...");
        if dat_src.exists() {
            fs::copy(&dat_src, &dat_dst)?;
        }

        // plonk2pil: convert R1CS to PIL
        let r1cs_data =
            fs::read(&r1cs_path).with_context(|| format!("Failed to read R1CS file: {}", r1cs_path.display()))?;
        plonk2pil::plonk2pil(&r1cs_data, type_compressor, &plonk_opts).context("plonk2pil failed in recursive setup")
    };

    // First pass: compile and run plonk2pil with the original stark_info.
    let mut plonk_result = run_circom_and_plonk(config.stark_info)?;

    // Threshold check and A2 nQueries adjustment (recursive1 only, no compressor).
    const RECURSIVE_BITS_THRESHOLD: usize = 17;
    if template == RecursiveTemplate::Recursive1 && !config.has_compressor {
        if plonk_result.n_bits > RECURSIVE_BITS_THRESHOLD {
            // For recursive1: bail early (before the expensive pil_info steps) if the
            // circuit exceeds the 17-bit threshold.  The caller catches NeedsCompressorError
            // and auto-retries with has_compressor = true.
            tracing::warn!(
                "Recursive1 for air '{}' has n_bits={} > {} — compressor needed",
                config.air_name,
                plonk_result.n_bits,
                RECURSIVE_BITS_THRESHOLD
            );
            return Err(anyhow::Error::new(NeedsCompressorError { n_bits: plonk_result.n_bits }));
        }

        if plonk_result.n_bits < RECURSIVE_BITS_THRESHOLD {
            // A2: small circuit — adjust nQueries in the input starkInfo so the verifier
            // circuit fills close to 2^(THRESHOLD-1) rows, matching JS isCompressorNeeded.
            //
            // JS formula:
            //   nRowsPerFri = NUsed / starkInfo.starkStruct.nQueries
            //   minimumQueriesRequired = ceil((2^(recursiveBits-1) + 2^12) / nRowsPerFri)
            let current_n_queries = config
                .stark_info
                .get("starkStruct")
                .and_then(|s| s.get("nQueries"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            if current_n_queries > 0 {
                // Use integer ceil to avoid f64 precision loss:
                // ceil(numer / nRowsPerFri) = ceil(numer * nQueries / NUsed)
                //                          = (numer * nQueries + NUsed - 1) / NUsed
                let numer = (1u64 << (RECURSIVE_BITS_THRESHOLD - 1)) + (1u64 << 12);
                let n_used = plonk_result.n_used as u64;
                let min_queries = (numer * current_n_queries).div_ceil(n_used);
                tracing::info!(
                    "Air '{}' recursive1: n_bits={}, n_used={}, nQueries={}, \
                     minimumQueriesRequired={}",
                    config.air_name,
                    plonk_result.n_bits,
                    plonk_result.n_used,
                    current_n_queries,
                    min_queries
                );
                if min_queries > current_n_queries {
                    tracing::info!(
                        "A2: adjusting nQueries for air '{}' recursive1: {} → {}",
                        config.air_name,
                        current_n_queries,
                        min_queries
                    );
                    // Build adjusted copy of stark_info with updated nQueries.
                    let mut adjusted_si = config.stark_info.clone();
                    if let Some(ss) = adjusted_si.get_mut("starkStruct") {
                        if let Some(obj) = ss.as_object_mut() {
                            obj.insert("nQueries".to_string(), serde_json::json!(min_queries));
                        }
                    }
                    // Persist the adjusted starkInfo so subsequent re-runs pick it up.
                    if let Some(si_path) = config.stark_info_path {
                        fs::write(si_path, crate::output::json::to_json_string(&adjusted_si)?)?;
                        tracing::info!("A2: wrote adjusted starkInfo (nQueries={}) to {:?}", min_queries, si_path);
                    }
                    // Re-compile with the adjusted starkInfo.
                    tracing::info!(
                        "A2: recompiling recursive1 for air '{}' with nQueries={}",
                        config.air_name,
                        min_queries
                    );
                    plonk_result = run_circom_and_plonk(&adjusted_si)?;
                }
            }
        }
    }

    // Has-compressor recursive1: the A2 nQueries knob above is the compressor's, not
    // this circuit's starkInfo, so we can't fix the size here. Instead bail early (before
    // the const-tree build, which would otherwise crash with a size mismatch against the
    // reused/shared starkInfo) and let the caller bump the compressor's nQueries and retry.
    // All recursive1 in an airgroup share one setup, so they must all reach 2^(THRESHOLD-1).
    if template == RecursiveTemplate::Recursive1
        && config.has_compressor
        && plonk_result.n_bits < RECURSIVE_BITS_THRESHOLD
    {
        tracing::warn!(
            "Recursive1 for air '{}' (has compressor) packs to n_bits={} < {} (n_used={}); \
             requesting a compressor nQueries bump",
            config.air_name,
            plonk_result.n_bits,
            RECURSIVE_BITS_THRESHOLD,
            plonk_result.n_used
        );
        return Err(anyhow::Error::new(RecursiveTooSmallError {
            n_bits: plonk_result.n_bits,
            n_used: plonk_result.n_used,
        }));
    }

    // Generate witness library (background) — done AFTER the threshold / A2 check
    // so the compiled output is from the final (possibly adjusted) circom. When
    // `defer_witness_lib` is set (compressor inside the resize loop), skip it here and
    // hand the params back so the caller generates it once for the winning attempt.
    let witness_lib_params = if config.defer_witness_lib {
        Some((name_filename.clone(), files_dir.to_string_lossy().into_owned()))
    } else {
        witness_tracker.run_witness_library_generation(
            config.build_dir,
            files_dir.to_str().unwrap_or(""),
            &name_filename,
            template_str,
            config.circom_helpers_dir,
        );
        None
    };

    // Write fixed polynomials binary
    let fixed_bin_path = build_dir_path.join(format!("{}.fixed.bin", name_filename));
    let fixed_info: Vec<(String, Vec<u32>, Vec<u64>)> =
        plonk_result.fixed_pols.iter().map(|fp| (fp.name.clone(), vec![fp.index as u32], fp.values.clone())).collect();
    fixed_cols::write_fixed_pols_bin(
        fixed_bin_path.to_str().unwrap(),
        &plonk_result.airgroup_name,
        &plonk_result.air_name,
        1u64 << plonk_result.n_bits,
        &fixed_info,
    )?;

    // Write PIL source
    let pil_path = pil_dir.join(format!("{}.pil", name_filename));
    fs::write(&pil_path, &plonk_result.pil_str)?;

    // Write exec buffer
    let exec_path = files_dir.join(format!("{}.exec", template_str));
    let exec_bytes: Vec<u8> = plonk_result.exec.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(&exec_path, &exec_bytes)?;

    // Compile PIL via JS pil2com (pil2-compiler npm package)
    let pilout_path = build_dir_path.join(format!("{}.pilout", name_filename));
    compile_pil(
        pil_path.to_str().unwrap(),
        pilout_path.to_str().unwrap(),
        config.std_pil_path,
        config.recurser_pil_path,
    )?;

    // Const file writing is deferred until after pil_info, which determines
    // the true nConstants (may be larger than plonk2pil's fixedPols count).
    // The plonk2pil fixed columns are kept for later use.
    let const_path = files_dir.join(format!("{}.const", template_str));
    let plonk_n_rows = 1usize << plonk_result.n_bits;
    let plonk_n_fixed = plonk_result.fixed_pols.len();

    // Run real starkSetup via pil_info on the compiled pilout, or reuse an
    // existing result.  The JS equivalent is the `setupAggregation_` parameter:
    // when if it is passed (non-null), starkSetup is skipped and the existing
    // starkInfo/verifierInfo/expressionsInfo are reused as-is.  This matters
    // for the 2nd+ airs in an airgroup and for recursive2 — they all share the
    // same circuit structure and therefore the same setup.
    tracing::info!("Running starkSetup for recursive circuit...");
    let starkinfo_path = files_dir.join(format!("{}.starkinfo.json", template_str));

    let (setup_stark_info, setup_verifier_info, setup_expressions_info) =
        if let Some((existing_si, existing_vi, existing_ei)) = config.existing_pil_info.as_ref() {
            // Reuse the provided setup — skip pil_info.
            tracing::info!("Reusing existing starkSetup for {} (skipping pil_info)", template_str);

            // We still need to write the JSON and binary files for this air's
            // directory (the const tree needs a starkinfo.json on disk).
            let stark_info_loaded = crate::types::stark_info::StarkInfo::from_json(existing_si)?;

            // Write const file: load pilout to get inline selector polynomial values
            {
                let proxy = PilOutProxy::new(pilout_path.to_str().unwrap_or(""))
                    .map_err(|e| anyhow::anyhow!("Failed to load pilout for const file: {}", e))?;
                if proxy.pilout.air_groups.is_empty() || proxy.pilout.air_groups[0].airs.is_empty() {
                    bail!("Pilout has no AIR groups: {}", pilout_path.display());
                }
                let air = &proxy.pilout.air_groups[0].airs[0];
                let plonk_values =
                    fixed_cols::reorder_plonk_pols_for_pilout(&plonk_result.fixed_pols, &proxy.pilout.symbols, 0, 0);
                fixed_cols::write_const_file(const_path.to_str().unwrap(), air, &plonk_values)?;
                tracing::info!(
                    "Wrote {} const file (reused setup): {} cols, {} rows",
                    template_str,
                    air.fixed_cols.len(),
                    air.num_rows.unwrap_or(0)
                );
            }

            // Write starkinfo so bctree (C FFI) can find it on disk; deleted
            // after bctree for recursive1 to match JS temp-file behaviour.
            fs::write(&starkinfo_path, serde_json::to_string_pretty(existing_si)?)?;

            // JS: verifierinfo/expressionsinfo JSON and bin files are only written
            // for compressor and recursive2, not for recursive1.
            if template != RecursiveTemplate::Recursive1 {
                fs::write(
                    files_dir.join(format!("{}.verifierinfo.json", template_str)),
                    serde_json::to_string_pretty(existing_vi)?,
                )?;
                fs::write(
                    files_dir.join(format!("{}.expressionsinfo.json", template_str)),
                    serde_json::to_string_pretty(existing_ei)?,
                )?;

                let vi_loaded = crate::types::stark_info::VerifierInfo::from_json(existing_vi)?;
                let ei_loaded = crate::types::stark_info::ExpressionsInfo::from_json(existing_ei)?;
                crate::io::bin_file::write_expressions_bin_file(
                    files_dir.join(format!("{}.bin", template_str)).to_str().unwrap(),
                    &stark_info_loaded,
                    &ei_loaded,
                )?;
                crate::io::bin_file::write_verifier_expressions_bin_file(
                    files_dir.join(format!("{}.verifier.bin", template_str)).to_str().unwrap(),
                    &stark_info_loaded,
                    &vi_loaded,
                )?;
            }

            (Some(existing_si.clone()), Some(existing_vi.clone()), Some(existing_ei.clone()))
        } else {
            // Load the compiled pilout and run the real pil_info pipeline
            let pilout_path_str = pilout_path.to_str().unwrap_or("");
            if !Path::new(pilout_path_str).exists() {
                bail!("Pilout not found at {}. Cannot run starkSetup for recursive circuit.", pilout_path_str);
            }

            let proxy =
                PilOutProxy::new(pilout_path_str).map_err(|e| anyhow::anyhow!("Failed to load pilout: {}", e))?;
            let pilout = &proxy.pilout;
            if pilout.air_groups.is_empty() || pilout.air_groups[0].airs.is_empty() {
                bail!("Compiled pilout has no AIR groups: {}", pilout_path_str);
            }
            let air = &pilout.air_groups[0].airs[0];
            let num_rows_air = air.num_rows.unwrap_or(0) as usize;
            let n_bits_air = if num_rows_air > 0 { (num_rows_air as f64).log2() as usize } else { plonk_result.n_bits };

            // Generate stark struct for this recursive circuit.
            // JS uses { blowupFactor: 3, lastLevelVerification: 1 } for recursive1/2.
            // Compressor uses blowupFactor: 2 and the default lastLevelVerification: 2.
            let make_recursive_settings = || {
                let (blowup, last_level) = if template == RecursiveTemplate::Compressor {
                    (2, None) // compressor: default lastLevelVerification (2)
                } else {
                    (3, None) // recursive1/2: JS uses lastLevelVerification: 1
                };
                crate::types::stark_struct::StarkSettings {
                    blowup_factor: Some(blowup),
                    folding_factor: Some(3),
                    final_degree: Some(5),
                    last_level_verification: last_level,
                    ..Default::default()
                }
            };
            let stark_struct = if let Some(ss_val) = config.stark_struct {
                serde_json::from_value::<crate::types::stark_struct::StarkStruct>(ss_val.clone()).unwrap_or_else(|_| {
                    crate::types::stark_struct::generate_stark_struct(&make_recursive_settings(), n_bits_air)
                })
            } else {
                crate::types::stark_struct::generate_stark_struct(&make_recursive_settings(), n_bits_air)
            };

            // Run pil_info to get real starkinfo/expressionsinfo/verifierinfo
            let pil_info_result = crate::pil::info::pil_info(pilout, 0, 0, &stark_struct, &Default::default());

            // Build JSON representations using the same helpers as the non-recursive path
            let opening_points = crate::output::stark_info::collect_opening_points(&pil_info_result.setup);
            let folding_factors = crate::output::stark_info::compute_folding_factors(&stark_struct);
            let ev_map_len = pil_info_result.pil_code.ev_map.len();
            let field_size = crate::types::security::goldilocks_cube_field_size();
            let fri_params = crate::types::security::FRISecurityParams {
                field_size,
                dimension: 1u64 << stark_struct.n_bits,
                rate: 1.0 / (1u64 << (stark_struct.n_bits_ext - stark_struct.n_bits)) as f64,
                n_opening_points: opening_points.len() as u64,
                n_functions: ev_map_len.max(1) as u64,
                folding_factors: folding_factors.clone(),
                max_grinding_bits: stark_struct.pow_bits as u64,
                use_max_grinding_bits: true,
                tree_arity: stark_struct.merkle_tree_arity as u64,
                target_security_bits: 128,
            };
            let mut fri_security = crate::types::security::get_optimal_fri_query_params("JBR", &fri_params);

            // An explicit starkStruct override (config.stark_struct) may request MORE queries
            // than the security-optimal count — this is how the caller sizes a has-compressor
            // recursive1 up to the shared domain (more compressor queries → bigger recursive1
            // verifier). Honor it, but never go BELOW the security floor, so soundness only ever
            // strengthens. get_optimal_fri_query_params otherwise discards the override entirely.
            let override_q = stark_struct.n_queries as u64;
            if config.stark_struct.is_some() && override_q > fri_security.n_queries {
                tracing::info!(
                    "Honoring nQueries override for {}: {} → {} (security floor {})",
                    template_str,
                    fri_security.n_queries,
                    override_q,
                    fri_security.n_queries
                );
                fri_security.n_queries = override_q;
            }

            let starkinfo_output = crate::output::stark_info::build_starkinfo_output(
                &pil_info_result.setup,
                &stark_struct,
                &pil_info_result.pil_code,
                &opening_points,
                &fri_security,
                config.airgroup_id,
                config.air_id,
                &airgroup_pil_name,
                pil_info_result.c_exp_id,
                pil_info_result.fri_exp_id,
                pil_info_result.q_deg,
            );
            let verifier_info_ref = &pil_info_result.pil_code.verifier_info;
            let expressions_info_ref = &pil_info_result.pil_code.expressions_info;
            let si_json = serde_json::to_value(&starkinfo_output)?;

            // Write starkinfo so bctree (C FFI) can find it on disk; deleted
            // after bctree for recursive1 to match JS temp-file behaviour.
            fs::write(&starkinfo_path, crate::output::json::to_json_string(&starkinfo_output)?)?;

            // JS: verifierinfo/expressionsinfo JSON and bin files are only written
            // for compressor and recursive2, not for recursive1.
            let stark_info_loaded = crate::types::stark_info::StarkInfo::from_json(&si_json)?;

            if template != RecursiveTemplate::Recursive1 {
                fs::write(
                    files_dir.join(format!("{}.verifierinfo.json", template_str)),
                    crate::output::json::to_json_string(verifier_info_ref)?,
                )?;
                fs::write(
                    files_dir.join(format!("{}.expressionsinfo.json", template_str)),
                    crate::output::json::to_json_string(expressions_info_ref)?,
                )?;

                let expressions_loaded = crate::types::stark_info::ExpressionsInfo::from(expressions_info_ref);
                crate::io::bin_file::write_expressions_bin_file(
                    files_dir.join(format!("{}.bin", template_str)).to_str().unwrap(),
                    &stark_info_loaded,
                    &expressions_loaded,
                )?;

                let verifier_loaded = crate::types::stark_info::VerifierInfo::from(verifier_info_ref);
                crate::io::bin_file::write_verifier_expressions_bin_file(
                    files_dir.join(format!("{}.verifier.bin", template_str)).to_str().unwrap(),
                    &stark_info_loaded,
                    &verifier_loaded,
                )?;
            }

            // Write const file: use air already loaded above (for pil_info) which has
            // inline selector polynomial values for columns beyond plonk2pil's output.
            {
                let plonk_values =
                    fixed_cols::reorder_plonk_pols_for_pilout(&plonk_result.fixed_pols, &pilout.symbols, 0, 0);
                fixed_cols::write_const_file(const_path.to_str().unwrap(), air, &plonk_values)?;
                tracing::info!(
                    "Wrote {} const file: {} cols ({} from plonk + {} from pilout), {} rows",
                    template_str,
                    air.fixed_cols.len(),
                    plonk_n_fixed,
                    air.fixed_cols.len().saturating_sub(plonk_n_fixed),
                    plonk_n_rows
                );
            }

            (
                Some(si_json),
                Some(serde_json::to_value(verifier_info_ref)?),
                Some(serde_json::to_value(expressions_info_ref)?),
            )
        };

    // Compute const tree and verkey
    let verkey_json_path = files_dir.join(format!("{}.verkey.json", template_str));
    let const_root = if const_path.exists() && starkinfo_path.exists() {
        let root = bctree::compute_const_tree(
            const_path.to_str().unwrap(),
            starkinfo_path.to_str().unwrap(),
            verkey_json_path.to_str().unwrap(),
        );

        // Write verkey.bin
        let mut verkey_bin = Vec::with_capacity(32);
        for &val in root.iter() {
            verkey_bin.extend_from_slice(&val.to_le_bytes());
        }
        fs::write(files_dir.join(format!("{}.verkey.bin", template_str)), &verkey_bin)?;

        root
    } else {
        bail!(
            "Cannot compute const tree: const file ({}) or starkinfo ({}) missing",
            const_path.display(),
            starkinfo_path.display()
        );
    };

    // JS writes starkinfo to a temp file for bctree and only persists it to
    // filesDir when template != "recursive1".  We wrote to the permanent path
    // so the C FFI bctree call could find it; clean it up now for recursive1.
    if template == RecursiveTemplate::Recursive1 {
        let _ = fs::remove_file(&starkinfo_path);
    }

    // JSON files already written by the real pil_info pipeline above

    // For recursive2, write vks.json and verifier.rs
    if template == RecursiveTemplate::Recursive2 {
        // JS stores rootCRecursives1 as a flat 2-D array: [[air0_vk], [air1_vk], ...]
        // where each air_vk = [v0,v1,v2,v3].  config.verification_keys is
        // &[Vec<Vec<String>>] = &[ag_vkeys], so we unwrap the outer slice level.
        let r1_vks = config.verification_keys.first().cloned().unwrap_or_default();
        let root_c_recursive2: Vec<serde_json::Value> =
            const_root.iter().map(|&v| serde_json::Value::from(v)).collect();
        let r1_vks_numeric: Vec<Vec<serde_json::Value>> = r1_vks
            .iter()
            .map(|air_vk| {
                air_vk
                    .iter()
                    .map(|v| {
                        v.parse::<u64>()
                            .map(serde_json::Value::from)
                            .unwrap_or_else(|_| serde_json::Value::String(v.clone()))
                    })
                    .collect()
            })
            .collect();
        let vks = serde_json::json!({
            "rootCRecursives1": r1_vks_numeric,
            "rootCRecursive2": root_c_recursive2,
        });
        fs::write(files_dir.join(format!("{}.vks.json", template_str)), serde_json::to_string_pretty(&vks)?)?;

        // Write verifier Rust file using the real starkinfo and verifierinfo
        if let (Some(ref si_val), Some(ref vi_val)) = (&setup_stark_info, &setup_verifier_info) {
            let si_loaded = crate::types::stark_info::StarkInfo::from_json(si_val)?;
            let vi_loaded = crate::types::stark_info::VerifierInfo::from_json(vi_val)?;
            crate::output::verifier::write_verifier_rust_file(
                files_dir.join(format!("{}.verifier.rs", template_str)).to_str().unwrap(),
                &si_loaded,
                &vi_loaded,
                true, // recursive2 uses VadcopFinalProof
                config.hash,
            )?;
        }
    }

    Ok(RecursiveSetupResult {
        const_root,
        pil_str: plonk_result.pil_str,
        stark_info: setup_stark_info,
        verifier_info: setup_verifier_info,
        expressions_info: setup_expressions_info,
        n_bits: plonk_result.n_bits,
        n_used: plonk_result.n_used,
        witness_lib_params,
    })
}

/// Resolve names and output paths based on the template type.
fn resolve_names_and_paths(
    config: &RecursiveSetupConfig<'_>,
) -> Result<(
    String,  // verifier_name
    String,  // name_filename
    PathBuf, // files_dir
    bool,    // input_challenges
    bool,    // verkey_input
    bool,    // enable_input
)> {
    let template = config.template;
    let build_dir = PathBuf::from(config.build_dir);

    match template {
        RecursiveTemplate::Compressor => {
            let verifier_name = format!("{}.verifier.circom", config.air_name);
            let name_filename = format!("{}_{}", config.air_name, template.as_str());
            let files_dir = build_dir
                .join("provingKey")
                .join(get_global_name(config.global_info))
                .join(config.airgroup_name)
                .join("airs")
                .join(config.air_name)
                .join(template.as_str());
            Ok((verifier_name, name_filename, files_dir, true, false, false))
        }
        RecursiveTemplate::Recursive1 if !config.has_compressor => {
            let verifier_name = format!("{}.verifier.circom", config.air_name);
            let name_filename = format!("{}_{}", config.air_name, template.as_str());
            let files_dir = build_dir
                .join("provingKey")
                .join(get_global_name(config.global_info))
                .join(config.airgroup_name)
                .join("airs")
                .join(config.air_name)
                .join(template.as_str());
            Ok((verifier_name, name_filename, files_dir, true, false, false))
        }
        RecursiveTemplate::Recursive1 => {
            // With compressor
            let verifier_name = format!("{}_compressor.verifier.circom", config.air_name);
            let name_filename = format!("{}_{}", config.air_name, template.as_str());
            let files_dir = build_dir
                .join("provingKey")
                .join(get_global_name(config.global_info))
                .join(config.airgroup_name)
                .join("airs")
                .join(config.air_name)
                .join("recursive1");
            Ok((verifier_name, name_filename, files_dir, false, false, false))
        }
        RecursiveTemplate::Recursive2 => {
            let verifier_name = format!("{}_recursive2.verifier.circom", config.airgroup_name);
            let name_filename = format!("{}_{}", config.airgroup_name, template.as_str());
            let files_dir = build_dir
                .join("provingKey")
                .join(get_global_name(config.global_info))
                .join(config.airgroup_name)
                .join(template.as_str());

            // enableInput is true when there are multiple airgroups or multiple airs
            let n_airgroups =
                config.global_info.get("air_groups").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(1);
            let n_airs_first = config
                .global_info
                .get("airs")
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(1);
            let enable_input = n_airgroups > 1 || n_airs_first > 1;

            Ok((verifier_name, name_filename, files_dir, false, true, enable_input))
        }
    }
}

/// Extract the global name from vadcopInfo.
fn get_global_name(global_info: &Value) -> String {
    global_info.get("name").and_then(|v| v.as_str()).unwrap_or("pilout").to_string()
}

/// Compile PIL source by spawning `pil2com` (JS compiler) as a subprocess.
/// Install pil2com: `npm install` (reads package.json in the repo root).
///
/// Thin wrapper around [`crate::commands::compile_pil::run_compile_pil`] so
/// that the recursive setup pipeline and the public CLI command share a
/// single implementation.
pub fn compile_pil(pil_path: &str, output_path: &str, std_pil_path: &str, recurser_pil_path: &str) -> Result<()> {
    use crate::commands::compile_pil::{run_compile_pil, CompilePilOptions};
    let opts = CompilePilOptions {
        pil_path: pil_path.to_string(),
        output_path: output_path.to_string(),
        include_paths: vec![std_pil_path.to_string(), recurser_pil_path.to_string()],
        fixed_dir: None,
        fixed_to_file: false,
        no_proto_fixed_data: false,
    };
    run_compile_pil(&opts)
}

/// Find the `pil2com` JS binary, checking (in order):
///   1. `PIL2C_EXEC` environment variable
///   2. npm install in the proofman repo itself (compile-time baked root) — covers
///      consumers that use proofman as a path/git dependency from another workspace
///   3. Local npm install: `node_modules/.bin/pil2com` (relative to cwd)
///   4. Walk up from the executable's location to find `node_modules/.bin/pil2com`
///   5. Global install: `pil2com` on PATH
///
/// Install via: `npm install`  (reads package.json in the repo root)
pub(crate) fn resolve_pil2com_exec() -> Option<String> {
    if let Ok(path) = std::env::var("PIL2C_EXEC") {
        if Path::new(&path).is_file() {
            return Some(path);
        }
    }
    // npm install in the proofman repo root (this crate sits at <root>/setup/pil2-stark)
    const PROOFMAN_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let baked = Path::new(PROOFMAN_ROOT).join("node_modules/.bin/pil2com");
    if baked.is_file() {
        if let Ok(abs) = baked.canonicalize() {
            return abs.to_str().map(|s| s.to_string());
        }
    }
    // Local npm install (preferred)
    let local_npm = Path::new("node_modules/.bin/pil2com");
    if local_npm.is_file() {
        if let Ok(abs) = local_npm.canonicalize() {
            return abs.to_str().map(|s| s.to_string());
        }
    }
    // Walk up from the executable's location
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent();
        while let Some(d) = dir {
            let candidate = d.join("node_modules/.bin/pil2com");
            if candidate.is_file() {
                if let Ok(abs) = candidate.canonicalize() {
                    return abs.to_str().map(|s| s.to_string());
                }
            }
            dir = d.parent();
        }
    }
    // Global install on PATH
    which::which("pil2com").ok().map(|p| p.to_string_lossy().into_owned())
}

/// [`resolve_pil2com_exec`], but self-bootstrapping: when pil2com is missing,
/// run `npm install` in the proofman repo root (where `package.json` lives)
/// and resolve again. Setup owns making its own tooling available — callers
/// (SDK, worker, CLI, tests) shouldn't each carry an npm bootstrap.
pub(crate) fn ensure_pil2com_exec() -> Option<String> {
    if let Some(path) = resolve_pil2com_exec() {
        return Some(path);
    }
    const PROOFMAN_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let root = Path::new(PROOFMAN_ROOT);
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    if !root.join("package.json").is_file() {
        return None;
    }
    tracing::info!("pil2com not found; running `npm install` in {}", root.display());
    match std::process::Command::new("npm").arg("install").current_dir(&root).status() {
        Ok(status) if status.success() => resolve_pil2com_exec(),
        Ok(status) => {
            tracing::warn!("`npm install` in {} exited with {status}", root.display());
            None
        }
        Err(e) => {
            tracing::warn!("failed to run `npm install` in {}: {e}", root.display());
            None
        }
    }
}

/// Resolve a path inside `node_modules/<package>/<sub_path>`, with an env-var override.
/// Search order:
///   1. Env var override
///   2. `<proofman-repo-root>/node_modules/<package>/<sub_path>` (compile-time baked)
///   3. CWD-relative `node_modules/<package>/<sub_path>`
///   4. Walk up from the running executable
fn find_node_module_subpath(env_var: &str, package: &str, sub_path: &str) -> Option<String> {
    if let Ok(v) = std::env::var(env_var) {
        if !v.is_empty() {
            return Some(v);
        }
    }
    // Compile-time path to the proofman repo root.
    const PROOFMAN_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let baked = std::path::Path::new(PROOFMAN_ROOT).join("node_modules").join(package).join(sub_path);
    if baked.is_dir() {
        if let Ok(abs) = baked.canonicalize() {
            return Some(abs.to_string_lossy().into_owned());
        }
    }
    let rel = PathBuf::from("node_modules").join(package).join(sub_path);
    if rel.is_dir() {
        if let Ok(abs) = rel.canonicalize() {
            return Some(abs.to_string_lossy().into_owned());
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent();
        while let Some(d) = dir {
            let candidate = d.join("node_modules").join(package).join(sub_path);
            if candidate.is_dir() {
                if let Ok(abs) = candidate.canonicalize() {
                    return Some(abs.to_string_lossy().into_owned());
                }
            }
            dir = d.parent();
        }
    }
    None
}

/// [`find_node_module_subpath`], but self-bootstrapping: when the package is
/// missing, install the Node deps (see [`crate::proving_key::node_deps`]) and
/// look again. Falls back to the literal relative path so callers keep getting
/// a usable-for-error-messages string even when bootstrap is impossible.
pub(crate) fn ensure_node_module_subpath(env_var: &str, package: &str, sub_path: &str) -> String {
    if let Some(p) = find_node_module_subpath(env_var, package, sub_path) {
        return p;
    }
    let probe = format!("{package}/{sub_path}");
    if let Some(root) = crate::proving_key::node_deps::ensure_node_deps(&probe) {
        let candidate = root.join("node_modules").join(package).join(sub_path);
        if let Ok(abs) = candidate.canonicalize() {
            return abs.to_string_lossy().into_owned();
        }
    }
    format!("node_modules/{package}/{sub_path}")
}
