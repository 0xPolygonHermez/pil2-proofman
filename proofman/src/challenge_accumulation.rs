use proofman_curves::{EcGFp5, EcMasFp5, curve::EllipticCurve};
use proofman_common::{CurveType, ProofCtx};
use proofman_fields::{hash_state, new_transcript, ExtensionField, GoldilocksQuinticExtension, PrimeField64};

/// Max transcript width across families
const W_MAX: usize = 16;
use std::ops::Add;
use proofman_util::{timer_start_debug, timer_stop_and_log_debug};
use std::sync::Mutex;

use crate::ContributionsInfo;
use rayon::prelude::*;

/// Debug dump of per-instance root contributions plus the shared publics and proof
/// values. The caller gates this on the `print` flag of `calculate_internal_contributions`
/// (driven by the `DEBUG_CHALLENGES` switch in proofman.rs), so this function itself is an
/// unconditional dump. Runs on CPU and GPU (roots/values are host-side by this point).
fn print_challenges<F: PrimeField64>(pctx: &ProofCtx<F>, roots_contributions: &[[F; 4]]) {
    let fmt = |v: &[F]| v.iter().map(|x| x.as_canonical_u64().to_string()).collect::<Vec<_>>().join(", ");

    tracing::info!("··· Publics: [{}]", fmt(&pctx.get_publics()));
    tracing::info!("··· Proof values: [{}]", fmt(&pctx.get_proof_values()));

    let my_instances = pctx.dctx_get_process_instances();
    for instance_id in my_instances.iter() {
        let root_contribution = roots_contributions[*instance_id];
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(*instance_id).unwrap();
        tracing::info!(
            "··· Instance {} [{}:{}]: Root contribution: [{}, {}, {}, {}]",
            instance_id,
            airgroup_id,
            air_id,
            root_contribution[0],
            root_contribution[1],
            root_contribution[2],
            root_contribution[3]
        );
    }
}

pub fn calculate_internal_contributions<F>(
    pctx: &ProofCtx<F>,
    roots_contributions: &[[F; 4]],
    values_contributions: &[Mutex<Vec<F>>],
    print: bool,
) -> Vec<u64>
where
    F: PrimeField64,
    GoldilocksQuinticExtension: ExtensionField<F>,
{
    timer_start_debug!(CALCULATE_INTERNAL_CONTRIBUTION);
    if print {
        print_challenges(pctx, roots_contributions);
    }
    let my_instances = pctx.dctx_get_process_instances();

    let contributions_size = match pctx.global_info.curve {
        CurveType::None => pctx.global_info.lattice_size.unwrap(),
        _ => 10,
    };

    let mut values = vec![vec![F::ZERO; contributions_size]; my_instances.len()];

    values.par_iter_mut().zip(my_instances.par_iter()).for_each(|(values_row, instance_id)| {
        let root_contribution = roots_contributions[*instance_id];

        let mut values_to_hash =
            values_contributions[*instance_id].lock().expect("Missing values_contribution").clone();
        values_to_hash[4..8].copy_from_slice(&root_contribution[..4]);

        let mut hash = new_transcript::<F>(&pctx.global_info.hash);
        hash.put(&values_to_hash);
        // The chain width, not the digest width: see TranscriptDyn::get_chain_state. For a
        // sponge these are the same; for BLAKE3 the chain is a whole XOF block.
        let contribution = hash.get_chain_state();

        if pctx.global_info.curve != CurveType::None {
            for (i, v) in contribution.iter().enumerate().take(10) {
                values_row[i] = *v;
            }
        } else {
            let w = contribution.len();
            assert!(
                w > 0 && w <= W_MAX && contributions_size >= w && contributions_size % w == 0,
                "contributions_size ({contributions_size}) must be a non-zero multiple of the transcript width ({w}, max {W_MAX})"
            );
            for (i, v) in contribution.iter().enumerate().take(w) {
                values_row[i] = *v;
            }
            let n_hashes = contributions_size / w - 1;
            let mut state = [F::ZERO; W_MAX];
            for j in 0..n_hashes {
                let base = j * w;
                state[..w].copy_from_slice(&values_row[base..base + w]);
                hash_state(&pctx.global_info.hash, &mut state[..w]);
                values_row[(j + 1) * w..(j + 2) * w].copy_from_slice(&state[..w]);
            }
        }
    });

    let partial_contribution = add_contributions(pctx, &values);

    let partial_contribution_u64: Vec<u64> = partial_contribution.iter().map(|&x| x.as_canonical_u64()).collect();

    timer_stop_and_log_debug!(CALCULATE_INTERNAL_CONTRIBUTION);

    partial_contribution_u64
}

pub fn calculate_global_challenge<F>(pctx: &ProofCtx<F>, all_partial_contributions_u64: &[ContributionsInfo]) -> [F; 3]
where
    F: PrimeField64,
    GoldilocksQuinticExtension: ExtensionField<F>,
{
    let mut transcript = new_transcript::<F>(&pctx.global_info.hash);

    transcript.put(&pctx.get_publics());

    let proof_values_stage = pctx.get_proof_values_by_stage(1);
    if !proof_values_stage.is_empty() {
        transcript.put(&proof_values_stage);
    }

    let all_partial_contributions: Vec<Vec<F>> = all_partial_contributions_u64
        .iter()
        .map(|arr| arr.challenge.iter().map(|&x| F::from_u64(x)).collect())
        .collect();

    let value = aggregate_contributions(pctx, &all_partial_contributions);
    transcript.put(&value);

    let mut global_challenge = [F::ZERO; 3];
    transcript.get_field(&mut global_challenge);

    global_challenge
}

pub fn add_contributions<F>(pctx: &ProofCtx<F>, values: &[Vec<F>]) -> Vec<F>
where
    F: PrimeField64,
    GoldilocksQuinticExtension: ExtensionField<F>,
{
    match pctx.global_info.curve {
        CurveType::EcGFp5 => {
            let mut result = EcGFp5::hash_to_curve(
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][0..5]),
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][5..10]),
            );

            for value in values.iter().skip(1) {
                let curve_point = EcGFp5::hash_to_curve(
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[0..5]),
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[5..10]),
                );
                result = result.add(&curve_point);
            }

            let mut curve_point_values = vec![F::ZERO; 10];
            curve_point_values[0..5].copy_from_slice(result.x().as_basis_coefficients_slice());
            curve_point_values[5..10].copy_from_slice(result.y().as_basis_coefficients_slice());
            curve_point_values
        }

        CurveType::EcMasFp5 => {
            let mut result = EcMasFp5::hash_to_curve(
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][0..5]),
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][5..10]),
            );

            for value in values.iter().skip(1) {
                let curve_point = EcMasFp5::hash_to_curve(
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[0..5]),
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[5..10]),
                );
                result = result.add(&curve_point);
            }

            let mut curve_point_values = vec![F::ZERO; 10];
            curve_point_values[0..5].copy_from_slice(result.x().as_basis_coefficients_slice());
            curve_point_values[5..10].copy_from_slice(result.y().as_basis_coefficients_slice());
            curve_point_values
        }

        CurveType::None => {
            let mut result = vec![F::ZERO; pctx.global_info.lattice_size.unwrap()];
            for value in values.iter() {
                for (i, v) in value.iter().enumerate() {
                    result[i] = result[i].add(*v);
                }
            }
            result
        }
    }
}

pub fn aggregate_contributions<F>(pctx: &ProofCtx<F>, values: &[Vec<F>]) -> Vec<F>
where
    F: PrimeField64,
    GoldilocksQuinticExtension: ExtensionField<F>,
{
    match pctx.global_info.curve {
        CurveType::EcGFp5 => {
            let mut result = EcGFp5::new(
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][0..5]),
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][5..10]),
            );

            for value in values.iter().skip(1) {
                let curve_point = EcGFp5::new(
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[0..5]),
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[5..10]),
                );
                result = result.add(&curve_point);
            }

            let mut curve_point_values = vec![F::ZERO; 10];
            curve_point_values[0..5].copy_from_slice(result.x().as_basis_coefficients_slice());
            curve_point_values[5..10].copy_from_slice(result.y().as_basis_coefficients_slice());
            curve_point_values
        }

        CurveType::EcMasFp5 => {
            let mut result = EcMasFp5::new(
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][0..5]),
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][5..10]),
            );

            for value in values.iter().skip(1) {
                let curve_point = EcMasFp5::new(
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[0..5]),
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[5..10]),
                );
                result = result.add(&curve_point);
            }

            let mut curve_point_values = vec![F::ZERO; 10];
            curve_point_values[0..5].copy_from_slice(result.x().as_basis_coefficients_slice());
            curve_point_values[5..10].copy_from_slice(result.y().as_basis_coefficients_slice());
            curve_point_values
        }

        CurveType::None => {
            let mut result = vec![F::ZERO; pctx.global_info.lattice_size.unwrap()];
            for value in values.iter() {
                for (i, v) in value.iter().enumerate() {
                    result[i] = result[i].add(*v);
                }
            }
            result
        }
    }
}
