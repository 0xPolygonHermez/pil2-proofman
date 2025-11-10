use curves::{EcGFp5, EcMasFp5, curve::EllipticCurve};
use proofman_common::{CurveType, ProofCtx};
use fields::{poseidon2_hash, ExtensionField, GoldilocksQuinticExtension, PrimeField64};
use std::ops::Add;
use proofman_starks_lib_c::{calculate_hash_c};
use transcript::FFITranscript;
use proofman_util::{timer_start_info, timer_stop_and_log_info};
use std::sync::Mutex;

use std::ffi::c_void;

use crate::ContributionsInfo;
use rayon::prelude::*;

fn print_challenges<F: PrimeField64>(pctx: &ProofCtx<F>, roots_contributions: &[[F; 4]]) {
    let my_instances = pctx.dctx_get_process_instances();

    for instance_id in my_instances.iter() {
        let root_contribution = roots_contributions[*instance_id];
        tracing::info!(
            "··· Instance {}: Root contribution: [{}, {}, {}, {}]",
            instance_id,
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
) -> Vec<u64>
where
    F: PrimeField64,
    GoldilocksQuinticExtension: ExtensionField<F>,
{
    print_challenges(pctx, roots_contributions);

    timer_start_info!(CALCULATE_INTERNAL_CONTRIBUTION);
    let my_instances = pctx.dctx_get_process_instances();

    let contributions_size = match pctx.global_info.curve {
        CurveType::None => pctx.global_info.lattice_size.unwrap(),
        _ => 10,
    };

    let mut values = vec![vec![F::ZERO; contributions_size]; my_instances.len()];

    values.par_iter_mut().zip(my_instances.par_iter()).for_each(|(values_row, instance_id)| {
        let mut contribution = vec![F::ZERO; 12];
        let root_contribution = roots_contributions[*instance_id];

        let mut values_to_hash =
            values_contributions[*instance_id].lock().expect("Missing values_contribution").clone();
        values_to_hash[4..8].copy_from_slice(&root_contribution[..4]);

        calculate_hash_c(
            contribution.as_mut_ptr() as *mut u8,
            values_to_hash.as_mut_ptr() as *mut u8,
            values_to_hash.len() as u64,
            12,
        );

        if pctx.global_info.curve != CurveType::None {
            for (i, v) in contribution.iter().enumerate().take(10) {
                values_row[i] = *v;
            }
        } else {
            for (i, v) in contribution.iter().enumerate().take(12) {
                values_row[i] = *v;
            }
            let n_hashes = contributions_size / 12 - 1;
            for j in 0..n_hashes {
                let input_slice = &mut values_row[(j * 12)..((j + 1) * 12)];
                let output = poseidon2_hash(input_slice);
                values_row[((j + 1) * 12)..((j + 2) * 12)].copy_from_slice(&output[..12]);
            }
        }
    });

    let partial_contribution = add_contributions(pctx, &values);

    let partial_contribution_u64: Vec<u64> = partial_contribution.iter().map(|&x| x.as_canonical_u64()).collect();

    timer_stop_and_log_info!(CALCULATE_INTERNAL_CONTRIBUTION);

    partial_contribution_u64
}

pub fn calculate_global_challenge<F>(pctx: &ProofCtx<F>, all_partial_contributions_u64: &[ContributionsInfo])
where
    F: PrimeField64,
    GoldilocksQuinticExtension: ExtensionField<F>,
{
    timer_start_info!(CALCULATE_GLOBAL_CHALLENGE);

    let transcript = FFITranscript::new(2, true);

    transcript.add_elements(pctx.get_publics_ptr(), pctx.global_info.n_publics);

    let proof_values_stage = pctx.get_proof_values_by_stage(1);
    if !proof_values_stage.is_empty() {
        transcript.add_elements(proof_values_stage.as_ptr() as *mut u8, proof_values_stage.len());
    }

    let all_partial_contributions: Vec<Vec<F>> = all_partial_contributions_u64
        .iter()
        .map(|arr| arr.challenge.iter().map(|&x| F::from_u64(x)).collect())
        .collect();

    let value = aggregate_contributions(pctx, &all_partial_contributions);
    transcript.add_elements(value.as_ptr() as *mut u8, value.len());

    let global_challenge = [F::ZERO; 3];
    transcript.get_challenge(&global_challenge[0] as *const F as *mut c_void);

    tracing::info!("··· Global challenge: [{}, {}, {}]", global_challenge[0], global_challenge[1], global_challenge[2]);
    pctx.set_global_challenge(2, &global_challenge);

    timer_stop_and_log_info!(CALCULATE_GLOBAL_CHALLENGE);
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
