use serde::Serialize;
use std::fs;
use std::path::Path;
use std::io::{BufWriter, Write};
use rayon::prelude::*;

use fields::PrimeField64;
use proofman_common::{ProofCtx, ProofmanResult, skip_prover_instance};
use proofman_hints::format_hint_field_output_vec;

use crate::HintMetadata;

#[derive(Debug, Serialize)]
struct DebugDataMetadata {
    hint_id: usize,
    name_piop: String,
    name_exprs: String,
}

#[derive(Debug, Serialize)]
struct DebugDataRow {
    hint_id: usize,
    row: usize,
    value: String,
}

/// Store debug data as NDJSON + metadata (plain, no compression)
pub fn store_debug_data<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    instance_id: usize,
    hint_metadatas: &[HintMetadata<F>],
    num_rows: usize,
    is_prod: bool,
) -> ProofmanResult<()> {
    let (skip, _instance_info) = skip_prover_instance(pctx, instance_id)?;

    if skip {
        tracing::info!(
            "Skipping debug visualization data writing for instance_id {} (airgroup_id {}, air_id {})",
            instance_id,
            airgroup_id,
            air_id
        );
        return Ok(());
    }

    let tmp_dir = Path::new("tmp");
    if !tmp_dir.exists() {
        fs::create_dir_all(tmp_dir)?;
    }

    // --- Write metadata.json ---
    let metadata_filename = format!(
        "debug_data_{}_{}_{}_{}_metadata.json",
        if is_prod { "prod" } else { "sum" },
        airgroup_id,
        air_id,
        instance_id
    );
    let metadata_path = tmp_dir.join(metadata_filename);
    let metadata_file = fs::File::create(&metadata_path)?;
    let mut metadata_writer = BufWriter::new(metadata_file);

    let metadata_vec: Vec<DebugDataMetadata> = hint_metadatas
        .iter()
        .map(|h| DebugDataMetadata {
            hint_id: h.hint_id,
            name_piop: h.name_piop.clone(),
            name_exprs: h.name_exprs.join(", "),
        })
        .collect();

    serde_json::to_writer_pretty(&mut metadata_writer, &metadata_vec)?;
    metadata_writer.flush()?;

    // --- Write rows.ndjson ---
    let rows_filename = format!(
        "debug_data_{}_{}_{}_{}_rows.ndjson",
        if is_prod { "prod" } else { "sum" },
        airgroup_id,
        air_id,
        instance_id
    );
    let rows_path = tmp_dir.join(rows_filename);
    let rows_file = fs::File::create(&rows_path)?;
    let mut rows_writer = BufWriter::new(rows_file);

    // Process each hint sequentially (safe for writing), but row creation can be parallel
    for hint in hint_metadatas.iter() {
        let rows: Vec<DebugDataRow> = if hint.deg_expr.is_zero() && hint.deg_mul.is_zero() {
            vec![DebugDataRow {
                hint_id: hint.hint_id,
                row: 0,
                value: format_hint_field_output_vec(&hint.expressions.get(0)),
            }]
        } else {
            (0..num_rows)
                .into_par_iter()
                .map(|j| DebugDataRow {
                    hint_id: hint.hint_id,
                    row: j,
                    value: format_hint_field_output_vec(&hint.expressions.get(j)),
                })
                .collect()
        };

        // Write rows sequentially to NDJSON file
        for row in rows {
            serde_json::to_writer(&mut rows_writer, &row)?;
            rows_writer.write_all(b"\n")?;
        }
    }

    rows_writer.flush()?;

    tracing::info!("Debug visualization data written to: {:?} (+metadata {:?})", rows_path, metadata_path);

    Ok(())
}
