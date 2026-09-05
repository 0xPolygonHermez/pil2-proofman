//! Rust port of the EJS `getExpressionsChunks()` function from
//! `circuits.gl/stark_verifier.circom.ejs`.
//!
//! Groups a flat list of FPM instructions into chunks of at most 999 each,
//! tracking which `tmp` signals cross chunk boundaries (the "inputs" a chunk
//! needs from a previous chunk and the "outputs" it makes available to later
//! chunks).
//!
//! This exactly mirrors the JS split so that:
//!   - Each `VerifyEvaluationsChunks{i}` / `CalculateDeepPolChunks{i}` template
//!     receives the right `signal input tmp_*` / `signal output tmp_*` declarations.
//!   - Cross-chunk wiring (the `(out1, out2, ...) <== ChunkI()(inputs...)` call
//!     sites) uses the correct sorted id lists.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

// ── Public types ──────────────────────────────────────────────────────────────

/// Metadata about a single `tmp_N` signal across the whole code.
#[derive(Debug, Clone)]
pub struct TmpInfo {
    /// Index of the last instruction that references this tmp (as dest or src).
    pub last_pos: usize,
    /// Dimension: 1 (base field) or 3 (extension field).
    pub dim: u64,
}

/// A single chunk of instructions with its cross-chunk dependency info.
#[derive(Debug)]
pub struct Chunk {
    /// The FPM instructions belonging to this chunk (slice indices into the
    /// original code array — stored as owned `Value` objects).
    pub code: Vec<Value>,
    /// Sorted list of `tmp` ids that are defined in a *previous* chunk and
    /// consumed (for the first time from this chunk's perspective) here.
    pub inputs: Vec<u64>,
    /// Sorted list of `tmp` ids that are defined in this chunk and still live
    /// at chunk-end (i.e., they will be consumed by a later chunk).
    pub outputs: Vec<u64>,
}

/// Result of `get_expressions_chunks`.
pub struct ChunkedCode {
    pub chunks: Vec<Chunk>,
    /// Full per-tmp metadata keyed by tmp id.
    pub tmps: HashMap<u64, TmpInfo>,
}

// ── Constants ────────────────────────────────────────────────────────────────

/// Flush threshold: chunks contain at most `MIN_CHUNK_SIZE - 1 = 999`
/// instructions (EJS condition: `currentChunk.length + 1 >= MIN_CHUNK_SIZE`).
const MIN_CHUNK_SIZE: usize = 1000;

// ── Implementation ────────────────────────────────────────────────────────────

/// Split `code` into chunks and return cross-chunk dependency information.
///
/// Panics if any instruction has `dest.type != "tmp"` or `dest.dim` ∉ {1, 3}.
pub fn get_expressions_chunks(code: &[Value]) -> ChunkedCode {
    // ── Pass 1: build tmps map ────────────────────────────────────────────────
    // tmps[id] = {lastPos, dim}
    // For each instruction: record dest; then update lastPos for any tmp srcs.
    let mut tmps: HashMap<u64, TmpInfo> = HashMap::new();

    for (i, inst) in code.iter().enumerate() {
        let dest = &inst["dest"];
        assert_eq!(dest["type"].as_str(), Some("tmp"), "instruction {i}: dest.type must be 'tmp'");
        let dim = dest["dim"].as_u64().expect("dest.dim must be integer");
        assert!(dim == 1 || dim == 3, "instruction {i}: dest.dim must be 1 or 3, got {dim}");
        let dest_id = dest["id"].as_u64().expect("dest.id must be integer");
        tmps.insert(dest_id, TmpInfo { last_pos: i, dim });

        for s in 0..2usize {
            if let Some(src) = inst["src"].get(s) {
                if src["type"].as_str() == Some("tmp") {
                    let src_id = src["id"].as_u64().expect("src.id must be integer");
                    if let Some(info) = tmps.get_mut(&src_id) {
                        info.last_pos = i;
                    }
                }
            }
        }
    }

    // ── Pass 2: build chunks ──────────────────────────────────────────────────
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut current_chunk: Vec<Value> = Vec::new();

    // All tmps currently defined and not yet consumed anywhere.
    let mut live_tmps: HashSet<u64> = HashSet::new();
    // Tmps that were live at the end of a previous chunk (available as inputs).
    let mut previous_live_tmps: HashSet<u64> = HashSet::new();

    // Cross-chunk tracking for the current chunk being built.
    let mut inputs: HashSet<u64> = HashSet::new();
    let mut outputs: HashSet<u64> = HashSet::new();

    for (i, inst) in code.iter().enumerate() {
        current_chunk.push(inst.clone());

        // Dest: this tmp is now live and is an output of the current chunk.
        let dest_id = inst["dest"]["id"].as_u64().unwrap();
        live_tmps.insert(dest_id);
        outputs.insert(dest_id);

        // Sources: check consumption and cross-chunk provenance.
        for s in 0..2usize {
            if let Some(src) = inst["src"].get(s) {
                if src["type"].as_str() == Some("tmp") {
                    let src_id = src["id"].as_u64().unwrap();
                    let is_last_use = tmps.get(&src_id).is_some_and(|t| t.last_pos == i);

                    if is_last_use {
                        // This tmp is fully consumed — remove from live tracking.
                        live_tmps.remove(&src_id);
                        outputs.remove(&src_id);
                    }

                    if previous_live_tmps.contains(&src_id) {
                        // This src comes from a previous chunk → it's a cross-chunk input.
                        inputs.insert(src_id);
                        if is_last_use {
                            previous_live_tmps.remove(&src_id);
                        }
                    }
                }
            }
        }

        // Flush when current chunk reaches 999 instructions (EJS: length+1 >= 1000).
        if current_chunk.len() + 1 >= MIN_CHUNK_SIZE {
            let mut sorted_inputs: Vec<u64> = inputs.iter().copied().collect();
            let mut sorted_outputs: Vec<u64> = outputs.iter().copied().collect();
            sorted_inputs.sort_unstable();
            sorted_outputs.sort_unstable();

            chunks.push(Chunk { code: current_chunk, inputs: sorted_inputs, outputs: sorted_outputs });

            current_chunk = Vec::new();

            // All currently-live tmps become available as potential cross-chunk
            // inputs for subsequent chunks.
            previous_live_tmps.extend(live_tmps.drain());
            outputs.clear();
            inputs.clear();
        }
    }

    // Flush remaining instructions as the final chunk (may be empty).
    if !current_chunk.is_empty() {
        let mut sorted_inputs: Vec<u64> = inputs.iter().copied().collect();
        let mut sorted_outputs: Vec<u64> = outputs.iter().copied().collect();
        sorted_inputs.sort_unstable();
        sorted_outputs.sort_unstable();

        chunks.push(Chunk { code: current_chunk, inputs: sorted_inputs, outputs: sorted_outputs });
    }

    ChunkedCode { chunks, tmps }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Build a simple add instruction: `tmp[dest_id: dim] = tmp[a_id] + tmp[b_id]`.
    fn add_tmps(dest_id: u64, dest_dim: u64, a_id: u64, a_dim: u64, b_id: u64, b_dim: u64) -> Value {
        json!({
            "op": "add",
            "dest": {"type": "tmp", "id": dest_id, "dim": dest_dim},
            "src": [
                {"type": "tmp", "id": a_id, "dim": a_dim},
                {"type": "tmp", "id": b_id, "dim": b_dim}
            ]
        })
    }

    /// Build an add of two evals → tmp.
    fn add_evals(dest_id: u64) -> Value {
        json!({
            "op": "add",
            "dest": {"type": "tmp", "id": dest_id, "dim": 3},
            "src": [
                {"type": "eval", "id": 0, "dim": 3},
                {"type": "eval", "id": 1, "dim": 3}
            ]
        })
    }

    // ── basic: small code, single chunk ──────────────────────────────────────

    #[test]
    fn single_chunk_no_cross() {
        // 3 instructions, all evals → tmps; no cross-chunk deps.
        let code: Vec<Value> = (0..3).map(add_evals).collect();
        let result = get_expressions_chunks(&code);
        assert_eq!(result.chunks.len(), 1);
        let c = &result.chunks[0];
        assert_eq!(c.code.len(), 3);
        assert!(c.inputs.is_empty());
        // All 3 tmps are outputs (they live at chunk-end since nothing consumes them).
        assert_eq!(c.outputs.len(), 3);
        assert_eq!(result.tmps.len(), 3);
    }

    // ── consumed within chunk: no outputs ────────────────────────────────────

    #[test]
    fn tmp_consumed_within_chunk_not_in_outputs() {
        // tmp[0] = eval + eval;  tmp[1] = tmp[0] + tmp[0];
        // tmp[0] is last-used at instruction 1, so it should NOT be in outputs.
        let code = vec![add_evals(0), add_tmps(1, 3, 0, 3, 0, 3)];
        let result = get_expressions_chunks(&code);
        assert_eq!(result.chunks.len(), 1);
        let c = &result.chunks[0];
        // Only tmp[1] survives to end; tmp[0] is fully consumed.
        assert_eq!(c.outputs, vec![1]);
        assert!(c.inputs.is_empty());
    }

    // ── two chunks: cross-chunk input tracking ────────────────────────────────

    #[test]
    fn two_chunks_cross_input() {
        // Build exactly 999 evals→tmp instructions (chunk 0), then 1 instruction
        // that uses tmp[0] from chunk 0.
        let mut code: Vec<Value> = (0..999u64).map(add_evals).collect();
        // Instruction 999: tmp[999] = tmp[0] + tmp[1]  (both from chunk 0)
        code.push(add_tmps(999, 3, 0, 3, 1, 3));

        let result = get_expressions_chunks(&code);
        assert_eq!(result.chunks.len(), 2, "expected 2 chunks");

        let c0 = &result.chunks[0];
        assert_eq!(c0.code.len(), 999);
        // tmp[0] and tmp[1] are still live at chunk-0 end (last used in chunk 1).
        assert!(c0.outputs.contains(&0), "tmp[0] should be in chunk-0 outputs");
        assert!(c0.outputs.contains(&1), "tmp[1] should be in chunk-0 outputs");

        let c1 = &result.chunks[1];
        assert_eq!(c1.code.len(), 1);
        // Both tmp[0] and tmp[1] are cross-chunk inputs for chunk 1.
        assert!(c1.inputs.contains(&0), "tmp[0] should be input of chunk 1");
        assert!(c1.inputs.contains(&1), "tmp[1] should be input of chunk 1");
        // tmp[999] is produced in chunk 1 and nothing consumes it → it's an output.
        assert_eq!(c1.outputs, vec![999]);
    }

    // ── three chunks: input consumed then gone from previous_live ─────────────

    #[test]
    fn cross_chunk_input_not_repeated() {
        // chunk 0: 999 evals→tmps (0..998), tmp[0] last used in chunk 1 only.
        // chunk 1: adds two more evals→tmps (999,1000), consumes tmp[0].
        // chunk 2: final instruction using tmp[999].
        let mut code: Vec<Value> = (0..999u64).map(add_evals).collect();
        // chunk 1 start — will consume tmp[0] once (last use), then fill to 999.
        code.push(add_tmps(999, 3, 0, 3, 5, 3)); // uses tmp[0] (last use) and tmp[5]
        for id in 1000..1998u64 {
            code.push(add_evals(id));
        }
        // chunk 2: uses tmp[999]
        code.push(add_tmps(1998, 3, 999, 3, 1000, 3));

        let result = get_expressions_chunks(&code);
        assert_eq!(result.chunks.len(), 3);

        let c1 = &result.chunks[1];
        // tmp[0] and tmp[5] are inputs of chunk 1 (came from chunk 0).
        assert!(c1.inputs.contains(&0));
        assert!(c1.inputs.contains(&5));

        let c2 = &result.chunks[2];
        // tmp[0] should NOT appear in chunk 2's inputs (it was last used in chunk 1).
        assert!(!c2.inputs.contains(&0), "tmp[0] should not be in chunk-2 inputs");
        // tmp[999] should be in chunk 2's inputs.
        assert!(c2.inputs.contains(&999));
    }

    // ── tmps metadata ────────────────────────────────────────────────────────

    #[test]
    fn tmps_last_pos_correct() {
        // tmp[0] defined at 0, last used at instruction 2.
        let code = vec![
            add_evals(0),
            add_evals(1),
            add_tmps(2, 3, 0, 3, 1, 3), // tmp[0] and tmp[1] last used here
        ];
        let result = get_expressions_chunks(&code);
        assert_eq!(result.tmps[&0].last_pos, 2);
        assert_eq!(result.tmps[&1].last_pos, 2);
        assert_eq!(result.tmps[&2].last_pos, 2); // tmp[2] defined here, never consumed
        assert_eq!(result.tmps[&0].dim, 3);
    }
}
