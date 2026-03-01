//! Native Rust port of the JS `writeGlobalData` function from `setup_cmd.js`.
//!
//! Produces three files under `{buildDir}/provingKey/`:
//!   - `pilout.globalInfo.json`
//!   - `pilout.globalConstraints.json`
//!   - `pilout.globalConstraints.bin`  (iden3 "chps" binary format)
//!
//! The binary file replicates `writeGlobalConstraintsBinFile` from
//! `pil2-stark/chelpers/globalConstraintsBinFile.js`, including the
//! `getParserArgs` constraint compiler (global mode only).

use std::collections::HashMap;
use std::os::raw::c_void;
use std::path::Path;
use anyhow::{anyhow, Result};
use serde::Serialize as _;
use serde_json::Value;
use proofman_starks_lib_c::*;
use crate::SetupConfig;

const FIELD_EXTENSION: u64 = 3;

// ---------------------------------------------------------------------------
// Thin Rust wrapper around the C++ BinFileUtils::BinFileWriter
// ---------------------------------------------------------------------------

/// Wraps the C++ `BinFileUtils::BinFileWriter` via FFI.
/// The file is opened on construction and closed (flushed) on drop.
struct BinWriter(*mut c_void);

impl BinWriter {
    fn new(path: &Path, magic: &str, version: u32, n_sections: u32) -> Self {
        let path_str = path.to_string_lossy();
        Self(binfile_writer_new_c(&path_str, magic, version, n_sections))
    }

    fn write_u8(&mut self, v: u8) {
        binfile_writer_write_u8_c(self.0, v);
    }

    fn write_u16_le(&mut self, v: u16) {
        binfile_writer_write_u16_c(self.0, v);
    }

    fn write_u32_le(&mut self, v: u32) {
        binfile_writer_write_u32_c(self.0, v);
    }

    fn write_u64_le(&mut self, v: u64) {
        binfile_writer_write_u64_c(self.0, v);
    }

    fn write_str_nul(&mut self, s: &str) {
        binfile_writer_write_string_c(self.0, s);
    }

    #[allow(dead_code)]
    fn write_bytes(&mut self, b: &[u8]) {
        binfile_writer_write_bytes_c(self.0, b.as_ptr() as *const c_void, b.len() as u64);
    }

    fn start_section(&mut self, id: u32) {
        binfile_writer_start_section_c(self.0, id);
    }

    fn end_section(&mut self) {
        binfile_writer_end_section_c(self.0);
    }
}

impl Drop for BinWriter {
    fn drop(&mut self) {
        binfile_writer_free_c(self.0);
    }
}

// ---------------------------------------------------------------------------
// Constraint compiler output
// ---------------------------------------------------------------------------

struct ExpsInfo {
    n_temp1: u32,
    n_temp3: u32,
    /// Per-instruction operation-table indices (u8 each, written as u8 in binary).
    ops: Vec<u8>,
    /// Flat args array: [opType, destArgs…, src0Args…, src1Args…] per instruction.
    /// Each element is written as u16 LE in the binary.
    args: Vec<u16>,
    dest_dim: u64,
    dest_id: u64,
}

// ---------------------------------------------------------------------------
// Temporal-variable register allocator (mirrors JS getIdMaps / temporalsSubsets)
// ---------------------------------------------------------------------------

/// Greedy interval graph coloring: pack non-intersecting live ranges into
/// minimal subsets (register slots).  Mirrors JS `temporalsSubsets`.
fn temporals_subsets(segments: &mut Vec<(usize, usize, u64)>) -> Vec<Vec<(usize, usize, u64)>> {
    // Sort by live-range end (mirrors JS: `segments.sort((a, b) => a[1] - b[1])`)
    segments.sort_by_key(|s| s.1);

    let mut subsets: Vec<Vec<(usize, usize, u64)>> = Vec::new();

    for &seg in segments.iter() {
        let mut best: Option<usize> = None;
        let mut min_dist = usize::MAX;

        for (i, subset) in subsets.iter().enumerate() {
            let last = *subset.last().unwrap();
            // isIntersecting: start2 < end1 && start1 < end2
            if seg.0 < last.1 && last.0 < seg.1 {
                continue;
            }
            let dist = if last.1 > seg.0 { last.1 - seg.0 } else { seg.0 - last.1 };
            if dist < min_dist {
                min_dist = dist;
                best = Some(i);
            }
        }

        if let Some(idx) = best {
            subsets[idx].push(seg);
        } else {
            subsets.push(vec![seg]);
        }
    }

    subsets
}

/// Compute compact slot IDs for tmp variables using interval-graph coloring.
/// Returns `(id1d, id3d, count1d, count3d)`.
fn get_id_maps(code: &[Value]) -> (HashMap<u64, u64>, HashMap<u64, u64>, u32, u32) {
    let mut ini1d: HashMap<u64, usize> = HashMap::new();
    let mut end1d: HashMap<u64, usize> = HashMap::new();
    let mut ini3d: HashMap<u64, usize> = HashMap::new();
    let mut end3d: HashMap<u64, usize> = HashMap::new();

    for (j, r) in code.iter().enumerate() {
        // destination
        if r["dest"]["type"].as_str() == Some("tmp") {
            let id = r["dest"]["id"].as_u64().unwrap_or(0);
            let dim = r["dest"]["dim"].as_u64().unwrap_or(1);
            if dim == 1 {
                ini1d.entry(id).or_insert(j);
                end1d.insert(id, j);
            } else {
                ini3d.entry(id).or_insert(j);
                end3d.insert(id, j);
            }
        }
        // sources
        if let Some(srcs) = r["src"].as_array() {
            for src in srcs {
                if src["type"].as_str() == Some("tmp") {
                    let id = src["id"].as_u64().unwrap_or(0);
                    let dim = src["dim"].as_u64().unwrap_or(1);
                    if dim == 1 {
                        ini1d.entry(id).or_insert(j);
                        end1d.insert(id, j);
                    } else {
                        ini3d.entry(id).or_insert(j);
                        end3d.insert(id, j);
                    }
                }
            }
        }
    }

    let mut segs1d: Vec<(usize, usize, u64)> = ini1d.keys().map(|&id| (ini1d[&id], end1d[&id], id)).collect();
    let mut segs3d: Vec<(usize, usize, u64)> = ini3d.keys().map(|&id| (ini3d[&id], end3d[&id], id)).collect();

    let subsets1d = temporals_subsets(&mut segs1d);
    let subsets3d = temporals_subsets(&mut segs3d);

    let mut id1d: HashMap<u64, u64> = HashMap::new();
    let mut count1d = 0u32;
    for subset in &subsets1d {
        for &(_, _, orig) in subset {
            id1d.insert(orig, count1d as u64);
        }
        count1d += 1;
    }

    let mut id3d: HashMap<u64, u64> = HashMap::new();
    let mut count3d = 0u32;
    for subset in &subsets3d {
        for &(_, _, orig) in subset {
            id3d.insert(orig, count3d as u64);
        }
        count3d += 1;
    }

    (id1d, id3d, count1d, count3d)
}

// ---------------------------------------------------------------------------
// Constraint compiler (global mode)
// ---------------------------------------------------------------------------

/// Sorting priority key from the JS `operationsMap`, used to order sources.
fn op_sort_key(src: &Value) -> u32 {
    let t = src["type"].as_str().unwrap_or("");
    let dim = src["dim"].as_u64().unwrap_or(1);
    match t {
        "cm" => {
            if dim == 1 {
                0
            } else {
                6
            }
        }
        "airvalue" | "proofvalue" | "tmp" | "custom" => match (t, dim) {
            ("tmp", 1) => 1,
            ("tmp", _) => 7,
            ("airvalue", 1) => 4,
            ("airvalue", _) => 8,
            ("proofvalue", 1) => 5,
            ("proofvalue", _) => 10,
            ("custom", 1) => 0,
            ("custom", _) => 6,
            _ => 0,
        },
        "Zi" | "const" => 0,
        "xDivXSubXi" => 6,
        "public" => 2,
        "number" => 3,
        "airgroupvalue" => 9,
        "challenge" => 11,
        "eval" => 12,
        _ => 0,
    }
}

/// Map (dest_dim, src0_dim, src1_dim) to the global operations-table index (0–2).
fn find_op_index(dest_dim: u64, src0_dim: u64, src1_dim: Option<u64>) -> Result<usize> {
    match (dest_dim, src0_dim, src1_dim) {
        (1, 1, Some(1)) => Ok(0),
        (3, 3, Some(1)) => Ok(1),
        (3, 3, Some(3)) => Ok(2),
        _ => Err(anyhow!(
            "Unsupported global operation: dest=dim{dest_dim}, src0=dim{src0_dim}, src1={:?}",
            src1_dim.map(|d| format!("dim{d}"))
        )),
    }
}

/// Push args for a destination reference (always `tmp` in global constraints).
fn push_dest_args(args: &mut Vec<u16>, dest: &Value, id1d: &HashMap<u64, u64>, id3d: &HashMap<u64, u64>) -> Result<()> {
    let t = dest["type"].as_str().unwrap_or("");
    if t != "tmp" {
        return Err(anyhow!("Destination must be `tmp` in global constraint, got: {t}"));
    }
    let dim = dest["dim"].as_u64().unwrap_or(1);
    let id = dest["id"].as_u64().unwrap_or(0);
    if dim == 1 {
        let slot = *id1d.get(&id).ok_or_else(|| anyhow!("tmp1 dest id {id} not in id_maps"))?;
        args.push(slot as u16);
    } else {
        let slot = *id3d.get(&id).ok_or_else(|| anyhow!("tmp3 dest id {id} not in id_maps"))?;
        args.push((FIELD_EXTENSION * slot) as u16);
    }
    Ok(())
}

/// Push args for a source reference in global mode.
///
/// Buffer indices (global mode):
///   0 → tmp1     1 → public    2 → number
///   3 → proofvalue  4 → tmp3  5 → airgroupvalue  6 → challenge
fn push_src_args(
    args: &mut Vec<u16>,
    src: &Value,
    numbers: &mut Vec<String>,
    id1d: &HashMap<u64, u64>,
    id3d: &HashMap<u64, u64>,
    global_info: &Value,
) -> Result<()> {
    let t = src["type"].as_str().unwrap_or("");
    let dim = src["dim"].as_u64().unwrap_or(1);

    match t {
        "tmp" => {
            let id = src["id"].as_u64().unwrap_or(0);
            if dim == 1 {
                args.push(0);
                let slot = *id1d.get(&id).ok_or_else(|| anyhow!("tmp1 src id {id} not in id_maps"))?;
                args.push(slot as u16);
            } else {
                args.push(4);
                let slot = *id3d.get(&id).ok_or_else(|| anyhow!("tmp3 src id {id} not in id_maps"))?;
                args.push((FIELD_EXTENSION * slot) as u16);
            }
        }
        "number" => {
            // Mirror JS: BigInt(r.value); if < 0: += 0xFFFFFFFF00000001n (GL prime)
            let raw: i128 = if let Some(n) = src["value"].as_i64() {
                n as i128
            } else if let Some(n) = src["value"].as_u64() {
                n as i128
            } else if let Some(s) = src["value"].as_str() {
                s.parse::<i128>().unwrap_or(0)
            } else {
                0
            };
            const GL_PRIME: i128 = 0xFFFF_FFFF_0000_0001;
            let num: u64 = if raw < 0 { (raw + GL_PRIME) as u64 } else { raw as u64 };
            let num_str = num.to_string();
            if !numbers.contains(&num_str) {
                numbers.push(num_str.clone());
            }
            let idx = numbers.iter().position(|s| *s == num_str).unwrap();
            args.push(2);
            args.push(idx as u16);
        }
        "public" => {
            let id = src["id"].as_u64().unwrap_or(0);
            args.push(1);
            args.push(id as u16);
        }
        "proofvalue" => {
            let id = src["id"].as_u64().unwrap_or(0);
            let pvm = global_info["proofValuesMap"]
                .as_array()
                .ok_or_else(|| anyhow!("missing proofValuesMap in globalInfo"))?;
            let mut pos: u64 = 0;
            for i in 0..id as usize {
                let stage = pvm[i]["stage"].as_u64().unwrap_or(0);
                pos += if stage == 1 { 1 } else { FIELD_EXTENSION };
            }
            args.push(3);
            args.push(pos as u16);
        }
        "airgroupvalue" => {
            let airgroup_id = src["airgroupId"].as_u64().unwrap_or(0);
            let id = src["id"].as_u64().unwrap_or(0);
            let agg = global_info["aggTypes"].as_array().ok_or_else(|| anyhow!("missing aggTypes in globalInfo"))?;
            let mut offset: u64 = 0;
            for i in 0..airgroup_id as usize {
                let len = agg[i].as_array().map(|a| a.len() as u64).unwrap_or(0);
                offset += FIELD_EXTENSION * len;
            }
            args.push(5);
            args.push((offset + FIELD_EXTENSION * id) as u16);
        }
        "challenge" => {
            let id = src["id"].as_u64().unwrap_or(0);
            args.push(6);
            args.push((FIELD_EXTENSION * id) as u16);
        }
        _ => return Err(anyhow!("Unsupported source type in global constraint: {t}")),
    }
    Ok(())
}

/// Port of JS `getParserArgs` in global mode (`global=true, verify=false`).
///
/// Compiles a single constraint's `code` array into ops/args arrays,
/// performing temporal-variable register allocation.
fn get_parser_args_global(code: &[Value], numbers: &mut Vec<String>, global_info: &Value) -> Result<ExpsInfo> {
    let (id1d, id3d, count1d, count3d) = get_id_maps(code);

    let mut ops: Vec<u8> = Vec::new();
    let mut args: Vec<u16> = Vec::new();

    for r in code {
        let op_str = r["op"].as_str().unwrap_or("add");
        let dest_dim = r["dest"]["dim"].as_u64().unwrap_or(1);

        // Collect and sort sources (mirrors JS `getOperation` sort + sub_swap detection).
        let srcs_raw = r["src"].as_array().map(|v| v.as_slice()).unwrap_or(&[]);
        let mut srcs: Vec<&Value> = srcs_raw.iter().collect();
        let mut op_name = op_str.to_string();

        if srcs.len() == 2 {
            let dim0 = srcs[0]["dim"].as_u64().unwrap_or(1);
            let dim1 = srcs[1]["dim"].as_u64().unwrap_or(1);
            let key0 = op_sort_key(srcs[0]);
            let key1 = op_sort_key(srcs[1]);

            // swap = b.dim - a.dim  (or keyA - keyB for same dim) — mirrors JS
            let swap: i64 = if dim0 != dim1 { dim1 as i64 - dim0 as i64 } else { key0 as i64 - key1 as i64 };

            if swap > 0 {
                // b has higher priority → move to front
                srcs.swap(0, 1);
            } else if swap < 0 && op_str == "sub" {
                // a stays at front (higher priority), but sub is not commutative:
                // execution engine computes src1-src0, so we need sub_swap for src0-src1.
                op_name = "sub_swap".to_string();
            }
        }

        // First arg: operation type (add=0, sub=1, mul=2, sub_swap=3)
        let op_type: u16 = match op_name.as_str() {
            "add" => 0,
            "sub" => 1,
            "mul" => 2,
            "sub_swap" => 3,
            _ => return Err(anyhow!("Unknown operation: {op_name}")),
        };
        args.push(op_type);

        // Destination args
        push_dest_args(&mut args, &r["dest"], &id1d, &id3d)?;

        // Source args (sorted order)
        for src in &srcs {
            push_src_args(&mut args, src, numbers, &id1d, &id3d, global_info)?;
        }

        // ops[j] = index into the global operations table
        let src0_dim = srcs.first().map(|s| s["dim"].as_u64().unwrap_or(1)).unwrap_or(1);
        let src1_dim = srcs.get(1).map(|s| s["dim"].as_u64().unwrap_or(1));
        ops.push(find_op_index(dest_dim, src0_dim, src1_dim)? as u8);
    }

    // Destination info comes from the last instruction
    let last = code.last().ok_or_else(|| anyhow!("Constraint has empty code array"))?;
    let dest_dim = last["dest"]["dim"].as_u64().unwrap_or(1);
    let dest_orig = last["dest"]["id"].as_u64().unwrap_or(0);
    let dest_id = if dest_dim == 1 {
        *id1d.get(&dest_orig).ok_or_else(|| anyhow!("tmp1 dest id {dest_orig} not found"))?
    } else {
        *id3d.get(&dest_orig).ok_or_else(|| anyhow!("tmp3 dest id {dest_orig} not found"))?
    };

    Ok(ExpsInfo { n_temp1: count1d, n_temp3: count3d, ops, args, dest_dim, dest_id })
}

// ---------------------------------------------------------------------------
// Binary file writer: "chps" format
// ---------------------------------------------------------------------------

fn write_constraints_section(w: &mut BinWriter, infos: &[ExpsInfo], lines: &[String], numbers: &[String]) {
    // Flatten all per-constraint ops/args and record per-constraint offsets
    let mut ops_all: Vec<u8> = Vec::new();
    let mut args_all: Vec<u16> = Vec::new();
    let mut ops_offsets: Vec<u32> = Vec::new();
    let mut args_offsets: Vec<u32> = Vec::new();

    for info in infos {
        ops_offsets.push(ops_all.len() as u32);
        args_offsets.push(args_all.len() as u32);
        ops_all.extend_from_slice(&info.ops);
        args_all.extend_from_slice(&info.args);
    }

    w.start_section(1);

    w.write_u32_le(ops_all.len() as u32);
    w.write_u32_le(args_all.len() as u32);
    w.write_u32_le(numbers.len() as u32);
    w.write_u32_le(infos.len() as u32);

    for (i, info) in infos.iter().enumerate() {
        w.write_u32_le(info.dest_dim as u32);
        w.write_u32_le(info.dest_id as u32);
        w.write_u32_le(info.n_temp1);
        w.write_u32_le(info.n_temp3);
        w.write_u32_le(info.ops.len() as u32);
        w.write_u32_le(ops_offsets[i]);
        w.write_u32_le(info.args.len() as u32);
        w.write_u32_le(args_offsets[i]);
        w.write_str_nul(&lines[i]);
    }

    for &op in &ops_all {
        w.write_u8(op);
    }
    for &arg in &args_all {
        w.write_u16_le(arg);
    }
    for num_str in numbers {
        let v: u64 = num_str.parse().unwrap_or(0);
        w.write_u64_le(v);
    }

    w.end_section();
}

fn write_hints_section(w: &mut BinWriter, hints: &[Value]) -> Result<()> {
    w.start_section(2);
    w.write_u32_le(hints.len() as u32);

    for hint in hints {
        let name = hint["name"].as_str().unwrap_or("");
        w.write_str_nul(name);

        let fields = hint["fields"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
        w.write_u32_le(fields.len() as u32);

        for field in fields {
            let fname = field["name"].as_str().unwrap_or("");
            w.write_str_nul(fname);

            let values = field["values"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
            w.write_u32_le(values.len() as u32);

            for value in values {
                let op = value["op"].as_str().unwrap_or("");
                w.write_str_nul(op);

                match op {
                    "number" => {
                        let v: u64 = if let Some(n) = value["value"].as_u64() {
                            n
                        } else if let Some(n) = value["value"].as_i64() {
                            n as u64
                        } else if let Some(s) = value["value"].as_str() {
                            s.parse().unwrap_or(0)
                        } else {
                            0
                        };
                        w.write_u64_le(v);
                    }
                    "string" => {
                        let s = value["string"].as_str().unwrap_or("");
                        w.write_str_nul(s);
                    }
                    "airgroupvalue" => {
                        let ag = value["airgroupId"].as_u64().unwrap_or(0) as u32;
                        let id = value["id"].as_u64().unwrap_or(0) as u32;
                        w.write_u32_le(ag);
                        w.write_u32_le(id);
                    }
                    "tmp" | "public" | "proofvalue" => {
                        let id = value["id"].as_u64().unwrap_or(0) as u32;
                        w.write_u32_le(id);
                    }
                    _ => return Err(anyhow!("Unknown hint value op: {op}")),
                }

                let pos = value["pos"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
                w.write_u32_le(pos.len() as u32);
                for p in pos {
                    w.write_u32_le(p.as_u64().unwrap_or(0) as u32);
                }
            }
        }
    }

    w.end_section();
    Ok(())
}

/// Port of `writeGlobalConstraintsBinFile`: compile constraints and emit
/// the iden3 "chps" binary file.
fn write_global_constraints_bin(global_info: &Value, global_constraints: &Value, path: &Path) -> Result<()> {
    let constraints = global_constraints["constraints"]
        .as_array()
        .ok_or_else(|| anyhow!("globalConstraints missing 'constraints' array"))?;
    let hints = global_constraints["hints"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);

    // Compile each constraint
    let mut numbers: Vec<String> = Vec::new();
    let mut infos: Vec<ExpsInfo> = Vec::with_capacity(constraints.len());
    let mut lines: Vec<String> = Vec::with_capacity(constraints.len());

    for c in constraints {
        let code = c["code"].as_array().ok_or_else(|| anyhow!("constraint missing 'code' array"))?;
        let info = get_parser_args_global(code, &mut numbers, global_info)?;
        infos.push(info);
        lines.push(c["line"].as_str().unwrap_or("").to_string());
    }

    // Write binary — the C++ BinFileWriter opens the file on construction,
    // writes sections directly to disk, and closes it on drop.
    let mut w = BinWriter::new(path, "chps", 1, 2);
    write_constraints_section(&mut w, &infos, &lines, &numbers);
    write_hints_section(&mut w, hints)?;

    tracing::info!("> Writing the global constraints file finished");
    Ok(())
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

/// Serialize a JSON value with 1-space indentation (mirrors `JSON.stringify(v, null, 1)`).
fn to_json_indent1(v: &Value) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let fmt = serde_json::ser::PrettyFormatter::with_indent(b" ");
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, fmt);
    v.serialize(&mut ser)?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Native Rust replacement for the JS `writeGlobalData` call.
///
/// `global_data` must have `globalInfo` and `globalConstraints` fields,
/// exactly as returned by `generate_circuits` via the Node bridge.
pub fn write_global_data(setup_config: &SetupConfig, global_data: &Value) -> Result<()> {
    let global_info = &global_data["globalInfo"];
    let global_constraints = &global_data["globalConstraints"];

    let build_dir = &setup_config.builddir;

    let pk = build_dir.join("provingKey");

    // pilout.globalInfo.json
    let gi_path = pk.join("pilout.globalInfo.json");
    let gi_bytes = to_json_indent1(global_info)?;
    std::fs::write(&gi_path, gi_bytes).map_err(|e| anyhow!("Failed to write {}: {e}", gi_path.display()))?;

    // pilout.globalConstraints.json
    let gc_path = pk.join("pilout.globalConstraints.json");
    let gc_bytes = to_json_indent1(global_constraints)?;
    std::fs::write(&gc_path, gc_bytes).map_err(|e| anyhow!("Failed to write {}: {e}", gc_path.display()))?;

    // pilout.globalConstraints.bin
    let bin_path = pk.join("pilout.globalConstraints.bin");
    write_global_constraints_bin(global_info, global_constraints, &bin_path)?;

    Ok(())
}
