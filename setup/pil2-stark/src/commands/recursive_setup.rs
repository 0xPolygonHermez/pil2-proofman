//! Recursive proving-key setup: compressor → recursive1 → recursive2 → final → compressed-final.
//!
//! Called from `setup_cmd::run_setup` when `--recursive` is set.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use pilout::pilout as pb;

use crate::proving_key::recursive::{RecursiveSetupConfig, RecursiveTemplate};
use crate::commands::setup::SetupOptions;
use crate::types::stark_struct::StarkStructsConfig;
use crate::output::witness_gen::WitnessTracker;

/// Run the recursive setup pipeline after non-recursive AIR setup.
///
/// For each airgroup/air:
///   1. Check whether a compressor is needed
///   2. If so, run compressor recursive setup
///   3. Run recursive1 (air[0] serial, air[1..N] parallel)
///   4. Run recursive2 immediately after all airs in the airgroup
///
/// Then globally:
///   5. Run final setup
///   6. Run compressed final setup
pub(crate) fn run_recursive_setup(
    pilout: &pb::PilOut,
    pilout_name: &str,
    opts: &SetupOptions,
    settings_map: &StarkStructsConfig,
    global_info: serde_json::Value,
) -> Result<std::collections::HashSet<String>> {
    use crate::proving_key::compressed_final;
    use crate::proving_key::final_setup;

    let build_dir = &opts.build_dir;

    let circuits_gl_path =
        resolve_path_env("CIRCUITS_GL_PATH", "setup/stark-recurser/stark2circom/circom_verifier/circuits.gl");
    let recurser_circuits_path =
        resolve_path_env("RECURSER_CIRCUITS_PATH", "setup/stark-recurser/stark2circom/circom_verifier/helper_circuits");
    // compressed_final still compiles with --prime goldilocks; the BN128 transition
    // happens later in the SNARK setup. Pointing this at circuits.bn128 would shadow
    // circuits.gl/merkle.circom and pull in comparators.circom (circomlib), which
    // isn't on the include path — see JS generateCompressedFinalSetup.js, which uses
    // recursion/helpers/circuits (the goldilocks-side helpers).
    let recurser_circuits_compressed_final_path = resolve_path_env(
        "RECURSER_CIRCUITS_COMPRESSED_FINAL_PATH",
        "setup/stark-recurser/stark2circom/circom_verifier/helper_circuits",
    );
    let std_pil_path = resolve_path_env("STD_PIL_PATH", "pil2-components/lib/std/pil");
    let recurser_pil_path = resolve_path_env("RECURSER_PIL_PATH", "setup/stark-recurser/plonk2pil/pil");
    let circom_helpers_dir = resolve_path_env("CIRCOM_HELPERS_DIR", "setup/circom");
    let goldilocks_src_dir = resolve_path_env("GOLDILOCKS_SRC_DIR", "pil2-stark/src/goldilocks/src");
    let circom_exec = resolve_circom_exec(&circom_helpers_dir);
    let witness_tracker = WitnessTracker::with_goldilocks_src(&goldilocks_src_dir);

    let proving_key_dir = Path::new(build_dir).join("provingKey");

    let global_constraints_path = proving_key_dir.join("pilout.globalConstraints.json");
    let global_constraints: serde_json::Value = if global_constraints_path.exists() {
        serde_json::from_str(&fs::read_to_string(&global_constraints_path)?)?
    } else {
        anyhow::bail!("globalConstraints.json not found at {:?}, cannot run recursive setup", global_constraints_path);
    };

    // Per-airgroup, per-air: run recursive1 then recursive2.
    //
    // Within each airgroup:
    //   - air[0] runs first (serial) to produce existing_pil_info (skips pil_info on reuse)
    //   - air[1..N] run in parallel bounded by opts.recursive_jobs
    //
    // The r1 and r2 loops are merged: r2 starts immediately after all r1s for
    // that airgroup finish — no need to wait for all airgroups.
    let recursive_jobs = opts.recursive_jobs.max(1);
    let mut recursive1_vkeys: Vec<Vec<Vec<String>>> = Vec::new();

    let mut ag_existing_pil_info: Vec<Option<(serde_json::Value, serde_json::Value, serde_json::Value)>> =
        vec![None; pilout.air_groups.len()];

    // Mutex protecting writes to starkstructs.json during the rare auto-compressor retry.
    let persist_mutex = std::sync::Mutex::new(());

    // Tracks which air names needed a compressor (auto-detected via NeedsCompressorError or
    // already set in settings). Returned to the caller so globalInfo can always be re-written
    // with the correct hasCompressor flags regardless of whether a starkstructs path was given.
    let mut airs_with_compressor: std::collections::HashSet<String> = Default::default();

    for (ag_idx, airgroup) in pilout.air_groups.iter().enumerate() {
        let airgroup_name = airgroup.name.clone().unwrap_or_else(|| format!("airgroup_{}", ag_idx));

        // Collect all valid airs for this airgroup.
        struct AirItem {
            air_idx: usize,
            air_name: String,
            stark_info: serde_json::Value,
            verifier_info: serde_json::Value,
            const_root_strings: [String; 4],
            has_compressor: bool,
            /// Path to the original air's starkinfo.json — used to persist A2 nQueries
            /// adjustments made inside gen_recursive_setup back to disk.
            si_path: PathBuf,
        }

        let mut air_items: Vec<AirItem> = Vec::new();
        for (air_idx, air) in airgroup.airs.iter().enumerate() {
            let air_name = air.name.clone().unwrap_or_else(|| format!("air_{}", air_idx));
            let num_rows = air.num_rows.unwrap_or(0) as usize;
            if num_rows == 0 {
                continue;
            }

            let files_dir = PathBuf::from(build_dir)
                .join("provingKey")
                .join(pilout_name)
                .join(&airgroup_name)
                .join("airs")
                .join(&air_name)
                .join("air");

            let si_path = files_dir.join(format!("{}.starkinfo.json", air_name));
            let vi_path = files_dir.join(format!("{}.verifierinfo.json", air_name));
            let vk_path = files_dir.join(format!("{}.verkey.json", air_name));

            if !si_path.exists() || !vi_path.exists() {
                anyhow::bail!(
                    "Recursive setup failed for air '{}': starkinfo/verifierinfo not found at {:?}",
                    air_name,
                    files_dir
                );
            }

            let stark_info: serde_json::Value = serde_json::from_str(&fs::read_to_string(&si_path)?)?;
            let verifier_info: serde_json::Value = serde_json::from_str(&fs::read_to_string(&vi_path)?)?;
            let const_root_strings =
                parse_verkey_json(&vk_path).with_context(|| format!("Failed to load verkey for air '{}'", air_name))?;

            let has_compressor = settings_map.has_compressor(&airgroup_name, &air_name);
            if has_compressor {
                tracing::info!("Air '{}': hasCompressor=true from settings", air_name);
            }

            air_items.push(AirItem {
                air_idx,
                air_name,
                stark_info,
                verifier_info,
                const_root_strings,
                has_compressor,
                si_path,
            });
        }

        if air_items.is_empty() {
            recursive1_vkeys.push(vec![]);
            continue;
        }

        // Helper closure: runs compressor (if needed) then recursive1 for one air.
        // Returns (air_idx, vk_strings, Option<existing_pil_info>, has_compressor, r1_n_bits).
        // existing_pil_info is Some only when the input `existing` was None (first air).
        // has_compressor may be upgraded to true if NeedsCompressorError fires at runtime.
        // r1_n_bits is the recursive1 circuit size; the caller compares it against
        // recursive2's n_bits to catch starkStruct mismatches.
        #[allow(clippy::type_complexity)]
        let run_one_air = |item: &AirItem,
                           existing: Option<(serde_json::Value, serde_json::Value, serde_json::Value)>|
         -> Result<(
            usize,
            Vec<String>,
            Option<(serde_json::Value, serde_json::Value, serde_json::Value)>,
            bool,
            usize,
        )> {
            let mut has_compressor = item.has_compressor;
            let mut compressor_result: Option<crate::proving_key::recursive::RecursiveSetupResult> = None;

            // Compressor→recursive1 with two retry triggers, resolved in one loop:
            //   NeedsCompressorError   (recursive1 too BIG, no compressor) → enable a compressor.
            //   RecursiveTooSmallError (has-compressor recursive1 too SMALL) → bump the compressor's
            //     nQueries (enlarging recursive1's verifier) and recompress, so recursive1 fills the
            //     shared 2^(THRESHOLD-1) domain. Queries only ever INCREASE → soundness never weakens.
            let mut compressor_ss_override: Option<serde_json::Value> = None; // bumped starkStruct
                                                                              // recursive1's n_used is affine in the compressor's nQueries (n_used = base + k*nQueries;
                                                                              // base is large & query-independent), so a proportional guess undershoots. After one
                                                                              // measured point we fit the slope from two points and solve for the exact nQueries.
            let mut prev_point: Option<(u64, u64)> = None; // (nQueries, n_used)
            const MAX_R1_ATTEMPTS: usize = 6;
            let r1_result = (|| -> Result<crate::proving_key::recursive::RecursiveSetupResult> {
                for attempt in 0..MAX_R1_ATTEMPTS {
                    // (Re)run the compressor if this air has one.
                    if has_compressor {
                        tracing::info!(
                            "Running compressor setup for air '{}'{}",
                            item.air_name,
                            if compressor_ss_override.is_some() { " (resized)" } else { "" }
                        );
                        let cfg = RecursiveSetupConfig {
                            build_dir,
                            hash: &opts.hash,
                            template: RecursiveTemplate::Compressor,
                            airgroup_name: &airgroup_name,
                            airgroup_id: ag_idx,
                            air_id: item.air_idx,
                            air_name: &item.air_name,
                            global_info: &global_info,
                            const_root: &item.const_root_strings,
                            verification_keys: &[],
                            stark_info: &item.stark_info,
                            verifier_info: &item.verifier_info,
                            stark_struct: compressor_ss_override.as_ref(),
                            has_compressor: false,
                            stark_info_path: None,
                            // Defer the compressor witness-lib gen: the resize loop may
                            // supersede this attempt with a nQueries bump. We generate the
                            // winning compressor's witness lib exactly once after the loop.
                            defer_witness_lib: true,
                            existing_pil_info: None,
                            circom_exec: &circom_exec,
                            circuits_gl_path: &circuits_gl_path,
                            recurser_circuits_path: &recurser_circuits_path,
                            std_pil_path: &std_pil_path,
                            recurser_pil_path: &recurser_pil_path,
                            circom_helpers_dir: &circom_helpers_dir,
                        };
                        compressor_result = Some(
                            crate::proving_key::recursive::gen_recursive_setup(&cfg, &witness_tracker)
                                .with_context(|| format!("Compressor setup failed for air '{}'", item.air_name))?,
                        );
                        tracing::info!("Compressor setup complete for air '{}'", item.air_name);
                    }

                    // Build r1 inputs from the compressor result (if any) or directly from item.
                    let r1_const_root: [String; 4] = if let Some(ref cr) = compressor_result {
                        let s: Vec<String> = cr.const_root.iter().map(|v| v.to_string()).collect();
                        [s[0].clone(), s[1].clone(), s[2].clone(), s[3].clone()]
                    } else {
                        item.const_root_strings.clone()
                    };
                    let r1_si =
                        compressor_result.as_ref().and_then(|cr| cr.stark_info.as_ref()).unwrap_or(&item.stark_info);
                    let r1_vi = compressor_result
                        .as_ref()
                        .and_then(|cr| cr.verifier_info.as_ref())
                        .unwrap_or(&item.verifier_info);

                    let r1_cfg = RecursiveSetupConfig {
                        build_dir,
                        hash: &opts.hash,
                        template: RecursiveTemplate::Recursive1,
                        airgroup_name: &airgroup_name,
                        airgroup_id: ag_idx,
                        air_id: item.air_idx,
                        air_name: &item.air_name,
                        global_info: &global_info,
                        const_root: &r1_const_root,
                        verification_keys: &[],
                        stark_info: r1_si,
                        verifier_info: r1_vi,
                        stark_struct: None,
                        has_compressor,
                        // A2: the original air's starkinfo path is only the right nQueries knob
                        // for the NO-compressor path; with a compressor the knob is the
                        // compressor's nQueries (handled via RecursiveTooSmallError below).
                        stark_info_path: if compressor_result.is_none() { Some(item.si_path.as_path()) } else { None },
                        defer_witness_lib: false, // recursive1 that bails too-small skips its own gen anyway
                        existing_pil_info: existing.clone(),
                        circom_exec: &circom_exec,
                        circuits_gl_path: &circuits_gl_path,
                        recurser_circuits_path: &recurser_circuits_path,
                        std_pil_path: &std_pil_path,
                        recurser_pil_path: &recurser_pil_path,
                        circom_helpers_dir: &circom_helpers_dir,
                    };

                    tracing::info!("Running recursive1 for air '{}'", item.air_name);
                    match crate::proving_key::recursive::gen_recursive_setup(&r1_cfg, &witness_tracker) {
                        Ok(r) => return Ok(r),
                        Err(e) if e.is::<crate::proving_key::recursive::NeedsCompressorError>() => {
                            let n_bits =
                                e.downcast_ref::<crate::proving_key::recursive::NeedsCompressorError>().unwrap().n_bits;
                            tracing::warn!(
                                "Air '{}' needs compressor (n_bits={} > 17); retrying with compressor",
                                item.air_name,
                                n_bits
                            );
                            if let Some(ref sp) = opts.stark_structs_path {
                                let _lock = persist_mutex.lock().unwrap();
                                if let Err(we) = persist_has_compressor(sp, &item.air_name) {
                                    tracing::warn!("Could not update starkstructs.json: {}", we);
                                } else {
                                    tracing::info!(
                                        "Updated starkstructs.json: hasCompressor=true for air '{}'",
                                        item.air_name
                                    );
                                }
                            }
                            has_compressor = true;
                            // loop: recompress + recursive1 with the now-enabled compressor.
                        }
                        Err(e) if e.is::<crate::proving_key::recursive::RecursiveTooSmallError>() => {
                            let too_small =
                                e.downcast_ref::<crate::proving_key::recursive::RecursiveTooSmallError>().unwrap();
                            // Bump the compressor's nQueries so recursive1 fills to 2^(THRESHOLD-1).
                            // recursive1's n_used scales ~linearly with the compressor's nQueries;
                            // target NUsed = 2^(THRESHOLD-1)+2^12 (identical to the non-compressor A2).
                            const RECURSIVE_BITS_THRESHOLD: usize = 17; // must match gen_recursive_setup
                            const TARGET_ROWS: u64 = (1u64 << (RECURSIVE_BITS_THRESHOLD - 1)) + (1u64 << 12);
                            let comp_ss = compressor_result
                                .as_ref()
                                .and_then(|cr| cr.stark_info.as_ref())
                                .and_then(|si| si.get("starkStruct"))
                                .cloned();
                            let Some(comp_ss) = comp_ss else {
                                return Err(anyhow::anyhow!(
                                    "Air '{}' recursive1 too small (n_bits={}) but compressor starkStruct is missing; \
                                     cannot resize",
                                    item.air_name,
                                    too_small.n_bits
                                ));
                            };
                            let cur_q = comp_ss.get("nQueries").and_then(|v| v.as_u64()).unwrap_or(0);
                            let cur_used = (too_small.n_used as u64).max(1);
                            if cur_q == 0 {
                                return Err(anyhow::anyhow!(
                                    "Air '{}' recursive1 too small (n_bits={}) but compressor nQueries is 0; \
                                     cannot resize",
                                    item.air_name,
                                    too_small.n_bits
                                ));
                            }
                            // recursive1's n_used is AFFINE in the compressor's nQueries:
                            // n_used = base + k*q, where base is the large query-independent cost of
                            // verifying the compressor's structure. A pure-proportional guess
                            // (base=0) undershoots. Once we have a second measured point, fit the
                            // slope k from the two points and solve for the exact q (secant step).
                            let want_q = match prev_point {
                                Some((pq, pu)) if cur_q != pq => {
                                    // k = Δused / Δq (per-query row cost); q = cur_q + (TARGET-cur_used)/k.
                                    let (hi_q, hi_u, lo_q, lo_u) =
                                        if cur_q > pq { (cur_q, cur_used, pq, pu) } else { (pq, pu, cur_q, cur_used) };
                                    let d_used = hi_u.saturating_sub(lo_u).max(1);
                                    let d_q = hi_q - lo_q;
                                    // want = cur_q + ceil((TARGET - cur_used) * d_q / d_used)
                                    let deficit = TARGET_ROWS.saturating_sub(cur_used);
                                    cur_q + (deficit * d_q).div_ceil(d_used)
                                }
                                // First bump (or degenerate): proportional guess, which under the
                                // affine model is a lower bound — the secant corrects it next round.
                                _ => (TARGET_ROWS * cur_q).div_ceil(cur_used),
                            };
                            // Always make real progress even if the model rounds flat.
                            let want_q = want_q.max(cur_q + 1);
                            prev_point = Some((cur_q, cur_used));
                            tracing::info!(
                                "Air '{}' recursive1 packs to 2^{} (n_used={}) below 2^{}; bumping compressor \
                                 nQueries {} → {} and recompressing (attempt {}/{})",
                                item.air_name,
                                too_small.n_bits,
                                too_small.n_used,
                                RECURSIVE_BITS_THRESHOLD,
                                cur_q,
                                want_q,
                                attempt + 1,
                                MAX_R1_ATTEMPTS
                            );
                            let mut bumped = comp_ss;
                            if let Some(obj) = bumped.as_object_mut() {
                                obj.insert("nQueries".to_string(), serde_json::json!(want_q));
                            }
                            compressor_ss_override = Some(bumped);
                            // loop: recompress with the bumped starkStruct + rerun recursive1.
                        }
                        Err(e) => {
                            return Err(e.context(format!("Recursive1 setup failed for air '{}'", item.air_name)));
                        }
                    }
                }
                Err(anyhow::anyhow!(
                    "Air '{}' recursive1 did not converge to the shared domain after {} attempts",
                    item.air_name,
                    MAX_R1_ATTEMPTS
                ))
            })()?;

            tracing::info!("Recursive1 setup complete for air '{}'", item.air_name);

            // The compressor deferred its witness-library generation inside the resize loop
            // (so superseded attempts didn't waste one). Now that `compressor_result` holds
            // the winning compressor, generate its witness lib exactly once.
            if let Some((name_filename, files_dir)) =
                compressor_result.as_ref().and_then(|cr| cr.witness_lib_params.clone())
            {
                witness_tracker.run_witness_library_generation(
                    build_dir,
                    &files_dir,
                    &name_filename,
                    "compressor",
                    &circom_helpers_dir,
                );
            }

            let vk_str: Vec<String> = r1_result.const_root.iter().map(|v| v.to_string()).collect();
            let produced_pil_info = if existing.is_none() {
                if let (Some(si), Some(vi), Some(ei)) =
                    (r1_result.stark_info.clone(), r1_result.verifier_info.clone(), r1_result.expressions_info.clone())
                {
                    Some((si, vi, ei))
                } else {
                    None
                }
            } else {
                None
            };

            Ok((item.air_idx, vk_str, produced_pil_info, has_compressor, r1_result.n_bits))
        };

        // --- air[0]: run serially to produce existing_pil_info ---
        let (_, first_vk, first_pil_info, first_hc, first_air_r1_n_bits) =
            run_one_air(&air_items[0], ag_existing_pil_info[ag_idx].clone())?;
        if first_hc {
            airs_with_compressor.insert(air_items[0].air_name.clone());
        }

        let mut ag_vkeys: Vec<Vec<String>> = vec![first_vk];
        if let Some(pil_info) = first_pil_info {
            ag_existing_pil_info[ag_idx] = Some(pil_info);
        }

        // --- air[1..N]: run in parallel (bounded by recursive_jobs) ---
        if air_items.len() > 1 {
            let existing_for_rest = ag_existing_pil_info[ag_idx].clone();

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(recursive_jobs)
                .build()
                .context("Failed to build recursive-jobs rayon pool")?;

            #[allow(clippy::type_complexity)]
            let parallel_results: Vec<Result<(usize, String, Vec<String>, bool)>> = pool.install(|| {
                use rayon::prelude::*;
                air_items[1..]
                    .par_iter()
                    .map(|item| {
                        let (air_idx, vk, _, hc, _) = run_one_air(item, existing_for_rest.clone())?;
                        Ok((air_idx, item.air_name.clone(), vk, hc))
                    })
                    .collect()
            });

            let mut indexed: Vec<(usize, String, Vec<String>, bool)> =
                parallel_results.into_iter().collect::<Result<_>>()?;
            indexed.sort_by_key(|(idx, _, _, _)| *idx);
            for (_, air_name, vk, hc) in indexed {
                ag_vkeys.push(vk);
                if hc {
                    airs_with_compressor.insert(air_name);
                }
            }
        }

        recursive1_vkeys.push(ag_vkeys.clone());

        // --- recursive2 for this airgroup (immediately after all r1s) ---
        {
            let first_air = airgroup.airs.first();
            let first_air_name = first_air.and_then(|a| a.name.clone()).unwrap_or_else(|| "air_0".to_string());

            // Use the in-memory starkinfo/verifierinfo produced by recursive1 for the
            // first air (ag_existing_pil_info holds the result from the fresh run).
            // These files are not persisted to disk for recursive1 (they're temp-only
            // for bctree) so we cannot read from disk here.
            let (r2_stark_info, r2_verifier_info) = match ag_existing_pil_info[ag_idx].as_ref() {
                Some((si, vi, _)) => (si.clone(), vi.clone()),
                None => anyhow::bail!(
                    "Recursive2 requires recursive1 starkinfo/verifierinfo for airgroup '{}' air '{}' \
                     but it was not produced (missing in-memory pil_info)",
                    airgroup_name,
                    first_air_name,
                ),
            };

            let const_root_strings: [String; 4] = ["0".into(), "0".into(), "0".into(), "0".into()];
            let vkeys_nested: Vec<Vec<Vec<String>>> = vec![ag_vkeys.clone()];

            tracing::info!("Running recursive2 for airgroup '{}'", airgroup_name);
            let r2_config = RecursiveSetupConfig {
                build_dir,
                hash: &opts.hash,
                template: RecursiveTemplate::Recursive2,
                airgroup_name: &airgroup_name,
                airgroup_id: ag_idx,
                air_id: 0,
                air_name: &first_air_name,
                global_info: &global_info,
                const_root: &const_root_strings,
                verification_keys: &vkeys_nested,
                stark_info: &r2_stark_info,
                verifier_info: &r2_verifier_info,
                stark_struct: None,
                has_compressor: false,
                stark_info_path: None,
                defer_witness_lib: false,
                existing_pil_info: None, // recursive2 always computes its own starkSetup
                circom_exec: &circom_exec,
                circuits_gl_path: &circuits_gl_path,
                recurser_circuits_path: &recurser_circuits_path,
                std_pil_path: &std_pil_path,
                recurser_pil_path: &recurser_pil_path,
                circom_helpers_dir: &circom_helpers_dir,
            };

            let r2_result = crate::proving_key::recursive::gen_recursive_setup(&r2_config, &witness_tracker)
                .with_context(|| format!("Recursive2 setup failed for airgroup '{}'", airgroup_name))?;

            if r2_result.n_bits != first_air_r1_n_bits {
                anyhow::bail!(
                    "Recursive2 n_bits ({}) does not match recursive1 n_bits ({}) for airgroup '{}' \
                     (first air '{}'). The recursive2 circuit must be sized identically to recursive1; \
                     a mismatch usually means the recursive2 starkStruct is inconsistent with recursive1's, \
                     or recursive2's circom expands to a different row count than expected.",
                    r2_result.n_bits,
                    first_air_r1_n_bits,
                    airgroup_name,
                    first_air_name,
                );
            }

            tracing::info!("Recursive2 setup complete for airgroup '{}' (n_bits={})", airgroup_name, r2_result.n_bits);
        }
    }

    // Run final setup
    tracing::info!("Running final setup...");
    let final_config = final_setup::FinalSetupConfig {
        build_dir,
        hash: &opts.hash,
        global_info: &global_info,
        global_constraints: &global_constraints,
        circom_exec: &circom_exec,
        circuits_gl_path: &circuits_gl_path,
        recurser_circuits_path: &recurser_circuits_path,
        std_pil_path: &std_pil_path,
        recurser_pil_path: &recurser_pil_path,
        circom_helpers_dir: &circom_helpers_dir,
    };

    let final_result = final_setup::gen_final_setup(&final_config, &witness_tracker).context("Final setup failed")?;
    tracing::info!("Final setup complete");

    // Run compressed final setup
    {
        let fr = &final_result;
        tracing::info!("Running compressed final setup...");
        let const_root_str: [String; 4] = [
            fr.const_root[0].to_string(),
            fr.const_root[1].to_string(),
            fr.const_root[2].to_string(),
            fr.const_root[3].to_string(),
        ];

        let compressed_config = compressed_final::CompressedFinalConfig {
            build_dir,
            hash: &opts.hash,
            name: pilout_name,
            const_root: &const_root_str,
            verification_keys: &[],
            stark_info: &fr.stark_info,
            verifier_info: &fr.verifier_info,
            circom_exec: &circom_exec,
            circuits_gl_path: &circuits_gl_path,
            recurser_circuits_path: &recurser_circuits_compressed_final_path,
            std_pil_path: &std_pil_path,
            recurser_pil_path: &recurser_pil_path,
            circom_helpers_dir: &circom_helpers_dir,
        };

        compressed_final::gen_compressed_final_setup(&compressed_config, &witness_tracker)
            .context("Compressed final setup failed")?;
        tracing::info!("Compressed final setup complete");
    }

    // Wait for all witness library builds
    witness_tracker.await_all()?;

    tracing::info!("Recursive setup complete");
    Ok(airs_with_compressor)
}

/// Parse a verkey.json file and return exactly 4 u64 limb strings.
pub(crate) fn parse_verkey_json(path: &Path) -> Result<[String; 4]> {
    if !path.exists() {
        anyhow::bail!("verkey.json not found: {:?}", path);
    }
    let content = fs::read_to_string(path).with_context(|| format!("Failed to read verkey file: {:?}", path))?;
    let vk: Vec<serde_json::Value> =
        serde_json::from_str(&content).with_context(|| format!("Failed to parse verkey JSON: {:?}", path))?;
    if vk.len() != 4 {
        anyhow::bail!("verkey.json has {} entries, expected exactly 4: {:?}", vk.len(), path);
    }
    let mut limbs = [String::new(), String::new(), String::new(), String::new()];
    for i in 0..4 {
        limbs[i] = vk[i]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("verkey.json limb {} is not a valid u64: {:?} in {:?}", i, vk[i], path))?
            .to_string();
    }
    Ok(limbs)
}

/// Update (or insert) `"hasCompressor": true` for `air_name` in a starkstructs JSON file.
///
/// The file is rewritten atomically (write to temp then rename).
pub(crate) fn persist_has_compressor(path: &str, air_name: &str) -> anyhow::Result<()> {
    use std::io::Write;

    let content = fs::read_to_string(path).with_context(|| format!("Cannot read starkstructs file: {}", path))?;
    let mut map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&content).with_context(|| format!("Cannot parse starkstructs JSON: {}", path))?;

    let entry = map.entry(air_name.to_string()).or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if let serde_json::Value::Object(ref mut obj) = entry {
        obj.insert("hasCompressor".to_string(), serde_json::Value::Bool(true));
    }

    let updated = serde_json::to_string_pretty(&serde_json::Value::Object(map))?;

    let tmp_path = format!("{}.tmp", path);
    {
        let mut f =
            std::fs::File::create(&tmp_path).with_context(|| format!("Cannot create temp file: {}", tmp_path))?;
        f.write_all(updated.as_bytes())?;
    }
    fs::rename(&tmp_path, path).with_context(|| format!("Cannot rename {} -> {}", tmp_path, path))?;
    Ok(())
}

/// Resolve the circom executable path from env, well-known locations, or PATH.
pub fn resolve_circom_exec(circom_helpers_dir: &str) -> String {
    // Preferred binary name depends on the OS (mirrors JS logic)
    let bin_name = if cfg!(target_os = "macos") { "circom_mac" } else { "circom" };

    // Look inside circom_helpers_dir first
    let in_helpers = Path::new(circom_helpers_dir).join(bin_name);
    if in_helpers.is_file() {
        if let Ok(abs) = in_helpers.canonicalize() {
            return abs.to_string_lossy().to_string();
        }
        return in_helpers.to_string_lossy().to_string();
    }

    // Fall back to PATH
    bin_name.to_string()
}

/// Resolve a path from an environment variable; if not set, search for `fallback` by
/// checking (in order):
///   1. `<proofman-repo-root>/fallback`, where the root is captured at compile time
///      from `CARGO_MANIFEST_DIR` (this crate sits at `<root>/setup/pil2-stark`).
///      Works regardless of where the binary is invoked from, including when
///      proofman is consumed as a git dep from another workspace.
///   2. `fallback` relative to the current working directory
///   3. `fallback` relative to each ancestor directory of the running executable
///   4. `fallback` as a literal string (last resort / relative-path pass-through)
pub fn resolve_path_env(env_var: &str, fallback: &str) -> String {
    if let Ok(v) = std::env::var(env_var) {
        if !v.is_empty() {
            return v;
        }
    }
    // Compile-time path to the proofman repo root.
    const PROOFMAN_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let baked = std::path::Path::new(PROOFMAN_ROOT).join(fallback);
    if baked.exists() {
        if let Ok(abs) = baked.canonicalize() {
            return abs.to_string_lossy().into_owned();
        }
    }
    // CWD-relative
    let cwd_rel = std::path::Path::new(fallback);
    if cwd_rel.exists() {
        if let Ok(abs) = cwd_rel.canonicalize() {
            return abs.to_string_lossy().into_owned();
        }
    }
    // Walk up from the executable
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent();
        while let Some(d) = dir {
            let candidate = d.join(fallback);
            if candidate.exists() {
                if let Ok(abs) = candidate.canonicalize() {
                    return abs.to_string_lossy().into_owned();
                }
            }
            dir = d.parent();
        }
    }
    fallback.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::setup::SetupOptions;
    use prost::Message;

    #[test]
    fn test_recursive_fails_on_missing_global_constraints() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let pilout_path = base.join("pil/zisk.pilout");
        if !pilout_path.exists() {
            eprintln!("Skipping: zisk.pilout not found");
            return;
        }

        let tmp = std::env::temp_dir().join(format!("pil2_recursive_failfast_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let build_dir = tmp.join("build");
        let pk_dir = build_dir.join("provingKey");
        std::fs::create_dir_all(&pk_dir).unwrap();
        std::fs::write(pk_dir.join("pilout.globalInfo.json"), r#"{"name":"zisk","nPublics":68,"hash":"Poseidon2"}"#)
            .unwrap();

        let pilout = pb::PilOut::decode(std::fs::read(&pilout_path).unwrap().as_slice()).unwrap();
        let opts = SetupOptions {
            airout_path: pilout_path.to_str().unwrap().to_string(),
            build_dir: build_dir.to_str().unwrap().to_string(),
            fixed_dir: None,
            stark_structs_path: None,
            recursive: true,
            recursive_jobs: 1,
            setup_jobs: 1,
            stats_output_path: None,
            hash: "Poseidon2".to_string(),
            gen_exps: false,
            exps_arch: "auto".to_string(),
            exps_cap: 40000,
            exps_chunk: None,
            exps_stark_src: None,
        };
        let result = run_recursive_setup(&pilout, "zisk", &opts, &StarkStructsConfig::default(), serde_json::json!({}));
        assert!(result.is_err());
        assert!(format!("{:#}", result.unwrap_err()).contains("globalConstraints.json not found"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_recursive_fails_on_missing_verkey() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let pilout_path = base.join("pil/zisk.pilout");
        if !pilout_path.exists() {
            eprintln!("Skipping: zisk.pilout not found");
            return;
        }

        let tmp = std::env::temp_dir().join(format!("pil2_verkey_failfast_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let build_dir = tmp.join("build");
        let pk_dir = build_dir.join("provingKey").join("zisk").join("Zisk").join("airs").join("Dma").join("air");
        std::fs::create_dir_all(&pk_dir).unwrap();
        let ginfo = build_dir.join("provingKey");
        std::fs::write(ginfo.join("pilout.globalInfo.json"),
            r#"{"name":"zisk","nPublics":68,"numProofValues":[8],"proofValuesMap":[],"publicsMap":[],"airGroupsInfo":[{"airGroupId":0,"nAirs":35}],"aggTypes":[[]],"hash":"Poseidon2"}"#
        ).unwrap();
        std::fs::write(ginfo.join("pilout.globalConstraints.json"), r#"{"constraints":[],"hints":[]}"#).unwrap();
        std::fs::write(pk_dir.join("Dma.starkinfo.json"), r#"{"nStages":2}"#).unwrap();
        std::fs::write(pk_dir.join("Dma.verifierinfo.json"), r#"{}"#).unwrap();

        let pilout = pb::PilOut::decode(std::fs::read(&pilout_path).unwrap().as_slice()).unwrap();
        let opts = SetupOptions {
            airout_path: pilout_path.to_str().unwrap().to_string(),
            build_dir: build_dir.to_str().unwrap().to_string(),
            fixed_dir: None,
            stark_structs_path: None,
            recursive: true,
            recursive_jobs: 1,
            setup_jobs: 1,
            stats_output_path: None,
            hash: "Poseidon2".to_string(),
            gen_exps: false,
            exps_arch: "auto".to_string(),
            exps_cap: 40000,
            exps_chunk: None,
            exps_stark_src: None,
        };
        let result = run_recursive_setup(&pilout, "zisk", &opts, &StarkStructsConfig::default(), serde_json::json!({}));
        assert!(result.is_err());
        let msg = format!("{:#}", result.unwrap_err());
        assert!(msg.contains("verkey") || msg.contains("not found"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_recursive2_fails_without_recursive1_artifacts() {
        let pilout = pb::PilOut {
            name: Some("test".to_string()),
            air_groups: vec![pb::AirGroup {
                name: Some("TestGroup".to_string()),
                airs: vec![pb::Air { name: Some("TestAir".to_string()), num_rows: Some(4), ..Default::default() }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let tmp = std::env::temp_dir().join(format!("pil2_r2_prereq_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let build_dir = tmp.join("build");
        let pk_dir = build_dir.join("provingKey");
        std::fs::create_dir_all(&pk_dir).unwrap();
        std::fs::write(pk_dir.join("pilout.globalInfo.json"), r#"{"name":"test","nPublics":0,"hash":"Poseidon2"}"#)
            .unwrap();
        std::fs::write(pk_dir.join("pilout.globalConstraints.json"), r#"{"constraints":[],"hints":[]}"#).unwrap();

        let pilout_path = tmp.join("test.pilout");
        let mut buf = Vec::new();
        pilout.encode(&mut buf).unwrap();
        std::fs::write(&pilout_path, &buf).unwrap();

        let opts = SetupOptions {
            airout_path: pilout_path.to_str().unwrap().to_string(),
            build_dir: build_dir.to_str().unwrap().to_string(),
            fixed_dir: None,
            stark_structs_path: None,
            recursive: true,
            recursive_jobs: 1,
            setup_jobs: 1,
            stats_output_path: None,
            hash: "Poseidon2".to_string(),
            gen_exps: false,
            exps_arch: "auto".to_string(),
            exps_cap: 40000,
            exps_chunk: None,
            exps_stark_src: None,
        };
        let result = run_recursive_setup(&pilout, "test", &opts, &StarkStructsConfig::default(), serde_json::json!({}));
        assert!(result.is_err());
        let msg = format!("{:#}", result.unwrap_err());
        assert!(msg.contains("starkinfo/verifierinfo not found"), "unexpected error: {}", msg);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_parse_verkey_json_valid() {
        let tmp = std::env::temp_dir().join(format!("pil2_vk_valid_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let p = tmp.join("vk.json");
        std::fs::write(&p, "[1,2,3,4]").unwrap();
        assert_eq!(parse_verkey_json(&p).unwrap(), ["1", "2", "3", "4"]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_parse_verkey_json_rejects_5_entries() {
        let tmp = std::env::temp_dir().join(format!("pil2_vk_5_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let p = tmp.join("vk.json");
        std::fs::write(&p, "[1,2,3,4,5]").unwrap();
        assert!(parse_verkey_json(&p).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_parse_verkey_json_rejects_short_array() {
        let tmp = std::env::temp_dir().join(format!("pil2_vk_short_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let p = tmp.join("vk.json");
        std::fs::write(&p, "[1,2]").unwrap();
        assert!(parse_verkey_json(&p).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_parse_verkey_json_rejects_non_numeric() {
        let tmp = std::env::temp_dir().join(format!("pil2_vk_nan_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let p = tmp.join("vk.json");
        std::fs::write(&p, r#"[1, "bad", 3, 4]"#).unwrap();
        assert!(parse_verkey_json(&p).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_parse_verkey_json_rejects_non_array() {
        let tmp = std::env::temp_dir().join(format!("pil2_vk_obj_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let p = tmp.join("vk.json");
        std::fs::write(&p, r#"{"a":1}"#).unwrap();
        assert!(parse_verkey_json(&p).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_parse_verkey_json_rejects_missing() {
        assert!(parse_verkey_json(std::path::Path::new("/tmp/nonexistent_verkey.json")).is_err());
    }
}
