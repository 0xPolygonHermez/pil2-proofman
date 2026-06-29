//! Operand resolution (starkinfo/expressionsinfo -> typed IR), tmp liveness,
//! and the interval-coloring scratch-slot allocator for the chunked path.
//!
//! `color()` breaks ties on `(def_chunk, temp_id)` so the scratch-slot
//! assignment is fully deterministic, independent of hash-map iteration order.

use crate::model::{ExpressionsInfo, StarkInfo};
use std::collections::{HashMap, HashSet};

/// A resolved source operand. `dim()` is the field-extension degree (1 = base
/// Goldilocks, 3 = cubic).
#[derive(Debug, Clone)]
pub enum Operand {
    Tmp { id: u64, dim: u64 },
    Num(u64),
    Cm { stage: u64, pos: u64, dim: u64, stride: i64 },
    Const { id: u64, stride: i64 },
    Zi,
    Ch { base: u64 },
    Av { pos: u64, dim: u64 },
    Agv { pos: u64, dim: u64 },
    Pub { id: u64 },
}

impl Operand {
    pub fn dim(&self) -> u64 {
        match self {
            Operand::Tmp { dim, .. } | Operand::Cm { dim, .. } | Operand::Av { dim, .. } | Operand::Agv { dim, .. } => {
                *dim
            }
            Operand::Ch { .. } => 3,
            Operand::Num(_) | Operand::Const { .. } | Operand::Zi | Operand::Pub { .. } => 1,
        }
    }
    /// `(id, dim)` if this is a tmp operand.
    pub fn as_tmp(&self) -> Option<(u64, u64)> {
        if let Operand::Tmp { id, dim } = self {
            Some((*id, *dim))
        } else {
            None
        }
    }
}

/// One IR step: `dst = op(a, b)`. `idx` is the global op index, used to name
/// loaded operand temps `a{idx}` / `b{idx}`.
#[derive(Debug)]
pub struct Instr {
    pub op: String,
    pub a: Operand,
    pub b: Operand,
    pub dst_is_tmp: bool,
    pub dst_id: Option<u64>,
    pub ddim: u64,
    pub idx: usize,
}

/// The resolved expression: straight-line IR + the column counts and constant
/// count needed to compute buffer offsets during emission.
pub struct Ir {
    pub instrs: Vec<Instr>,
    pub ncols: HashMap<u64, u64>,
    pub n_constants: u64,
}

/// Operand type the generator does not support (e.g. `custom`). The driver
/// treats this as "skip this AIR -> keep it on the bytecode interpreter".
#[derive(Debug)]
pub struct UnhandledOperand(pub String);
impl std::fmt::Display for UnhandledOperand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unhandled operand {}", self.0)
    }
}
impl std::error::Error for UnhandledOperand {}

/// Resolve the AIR's Q (cExp) expression into typed IR. Returns
/// `UnhandledOperand` (as an `anyhow` error) if it references an operand type
/// the generator can't emit.
pub fn build_ir(stark_info: &StarkInfo, expr_info: &ExpressionsInfo) -> anyhow::Result<Ir> {
    let opening = &stark_info.opening_points;
    let blowup = stark_info.blowup();
    let n_constants = stark_info.n_constants;

    // ncols[stage] = number of committed columns at that stage, for stages 1..=nStages+1.
    let mut ncols = HashMap::new();
    for stage in 1..=(stark_info.n_stages + 1) {
        let n = stark_info.map_sections_n.get(&format!("cm{stage}")).copied().unwrap_or(0);
        ncols.insert(stage, n);
    }

    let code = &expr_info
        .expressions_code
        .iter()
        .find(|e| e.exp_id == stark_info.c_exp_id)
        .ok_or_else(|| anyhow::anyhow!("cExpId {} not found in expressionsCode", stark_info.c_exp_id))?
        .code;

    // stride(prime) = openingPoints[index(prime)] * blowup (membership-validated).
    let stride = |prime: i64| -> anyhow::Result<i64> {
        let pos = opening
            .iter()
            .position(|&x| x == prime)
            .ok_or_else(|| anyhow::anyhow!("opening point {prime} not in openingPoints"))?;
        Ok(opening[pos] * blowup)
    };
    // av_pos / agv_pos: position of value `idx` in the flattened air(group)Values
    // buffer; stage-1 values occupy 1 slot, others 3.
    let pos_of = |idx: u64, map: &[crate::model::ValueMapEntry]| -> u64 {
        (0..idx as usize).map(|j| if map[j].stage.unwrap_or(1) != 1 { 3 } else { 1 }).sum()
    };

    let operand = |src: &crate::model::Src| -> anyhow::Result<Operand> {
        let dim = src.dim.unwrap_or(1);
        Ok(match src.op_type.as_str() {
            "tmp" => Operand::Tmp { id: src.id.unwrap(), dim },
            "number" => Operand::Num(src.number_value()?),
            "cm" => {
                let cm = &stark_info.cm_pols_map[src.id.unwrap() as usize];
                Operand::Cm { stage: cm.stage, pos: cm.stage_pos, dim: cm.dim, stride: stride(src.prime.unwrap_or(0))? }
            }
            "const" => Operand::Const { id: src.id.unwrap(), stride: stride(src.prime.unwrap_or(0))? },
            "Zi" => Operand::Zi,
            "challenge" => Operand::Ch { base: 3 * src.id.unwrap() },
            "airvalue" => Operand::Av { pos: pos_of(src.id.unwrap(), &stark_info.air_values_map), dim },
            "airgroupvalue" => Operand::Agv { pos: pos_of(src.id.unwrap(), &stark_info.airgroup_values_map), dim },
            "public" => Operand::Pub { id: src.id.unwrap() },
            other => return Err(anyhow::Error::new(UnhandledOperand(other.to_string()))),
        })
    };

    let mut instrs = Vec::with_capacity(code.len());
    for (idx, step) in code.iter().enumerate() {
        instrs.push(Instr {
            op: step.op.clone(),
            a: operand(&step.src[0])?,
            b: operand(&step.src[1])?,
            dst_is_tmp: step.dest.dest_type == "tmp",
            dst_id: step.dest.id,
            ddim: step.dest.dim,
            idx,
        });
    }
    Ok(Ir { instrs, ncols, n_constants })
}

/// Liveness + chunking + scratch-slot coloring for one chunk size.
pub struct ChunkPlan {
    pub chunk: usize,
    pub n_chunks: usize,
    pub out_dim: u64,
    pub def_idx: HashMap<u64, usize>,
    pub dim_of: HashMap<u64, u64>,
    pub cut_temps: HashSet<u64>,
    pub total_slots: u64,
    slot_index: HashMap<u64, u64>,
}

impl ChunkPlan {
    pub fn chunk_of(&self, op_idx: usize) -> usize {
        op_idx / self.chunk
    }
    pub fn slot_index(&self, t: u64) -> u64 {
        self.slot_index[&t]
    }
}

/// Build the liveness/chunk/coloring plan for `ir` at the given chunk size.
/// `sym` is only used in the SSA-violation error message.
pub fn plan_chunks(ir: &Ir, chunk_req: usize, sym: &str) -> anyhow::Result<ChunkPlan> {
    let n_ops = ir.instrs.len();

    // tmp liveness. SSA is assumed (each temp written once); fail loud otherwise,
    // since the chunk coloring would silently be wrong.
    let mut def_idx: HashMap<u64, usize> = HashMap::new();
    let mut last_use: HashMap<u64, usize> = HashMap::new();
    let mut dim_of: HashMap<u64, u64> = HashMap::new();
    for (i, instr) in ir.instrs.iter().enumerate() {
        if instr.dst_is_tmp {
            let id = instr.dst_id.expect("tmp dest without id");
            if def_idx.contains_key(&id) {
                anyhow::bail!("{sym}: temp t{id} written twice -- IR is not SSA, chunk liveness would be wrong");
            }
            def_idx.insert(id, i);
            dim_of.insert(id, instr.ddim);
        }
        for opnd in [&instr.a, &instr.b] {
            if let Some((tid, tdim)) = opnd.as_tmp() {
                last_use.insert(tid, i);
                dim_of.insert(tid, tdim);
            }
        }
    }

    let chunk = chunk_req.clamp(1, n_ops.max(1));
    let n_chunks = n_ops.div_ceil(chunk);
    let chunk_of = |op_idx: usize| op_idx / chunk;

    let mut out_dim = 3u64;
    for instr in &ir.instrs {
        if !instr.dst_is_tmp {
            out_dim = instr.ddim;
        }
    }

    // temps whose live range crosses a chunk boundary must be materialized to scratch.
    let cut_temps: HashSet<u64> = def_idx
        .keys()
        .copied()
        .filter(|t| last_use.get(t).is_some_and(|&lu| chunk_of(lu) > chunk_of(def_idx[t])))
        .collect();

    // linear-scan slot allocation, deterministic tie-break on (def_chunk, id).
    let color = |temps: &[u64], width: u64| -> (HashMap<u64, u64>, u64) {
        let mut sorted: Vec<u64> = temps.to_vec();
        sorted.sort_by_key(|&t| (chunk_of(def_idx[&t]), t));
        let mut slot_of: HashMap<u64, u64> = HashMap::new();
        let mut active: Vec<(usize, u64)> = Vec::new(); // (end_chunk, base), in insertion order
        let mut free_bases: Vec<u64> = Vec::new(); // LIFO
        let mut next_base: u64 = 0;
        for t in sorted {
            let def_chunk = chunk_of(def_idx[&t]);
            let end_chunk = chunk_of(last_use[&t]);
            // expire: free slots of temps that ended before t starts (preserve active order)
            let mut still = Vec::with_capacity(active.len());
            for &(end, base) in &active {
                if end < def_chunk {
                    free_bases.push(base);
                } else {
                    still.push((end, base));
                }
            }
            active = still;
            let base = free_bases.pop().unwrap_or_else(|| {
                let b = next_base;
                next_base += width;
                b
            });
            slot_of.insert(t, base);
            active.push((end_chunk, base));
        }
        (slot_of, next_base)
    };

    let temps1: Vec<u64> = cut_temps.iter().copied().filter(|t| dim_of[t] == 1).collect();
    let temps3: Vec<u64> = cut_temps.iter().copied().filter(|t| dim_of[t] == 3).collect();
    let (slot1, n_slots1) = color(&temps1, 1);
    let (slot3, n_slots3) = color(&temps3, 3);
    let base3 = n_slots1;
    let total_slots = n_slots1 + n_slots3;

    let mut slot_index: HashMap<u64, u64> = HashMap::new();
    for &t in &cut_temps {
        let s = if dim_of[&t] == 1 { slot1[&t] } else { base3 + slot3[&t] };
        slot_index.insert(t, s);
    }

    Ok(ChunkPlan { chunk, n_chunks, out_dim, def_idx, dim_of, cut_temps, total_slots, slot_index })
}
