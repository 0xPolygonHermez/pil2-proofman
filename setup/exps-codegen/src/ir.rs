//! Operand resolution (starkinfo/expressionsinfo -> typed IR), tmp liveness,
//! and the interval-coloring scratch-slot allocator for the chunked path.
//!
//! `color()` breaks ties on `(def_chunk, temp_id)` so the scratch-slot
//! assignment is fully deterministic, independent of hash-map iteration order.

use crate::model::{ExpressionsInfo, StarkInfo};
use std::collections::{HashMap, HashSet};

/// A resolved source operand. `dim()` is the field-extension degree (1 = base
/// Goldilocks, 3 = cubic).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Operand {
    Tmp {
        id: u64,
        dim: u64,
    },
    Num(u64),
    Cm {
        stage: u64,
        pos: u64,
        dim: u64,
        stride: i64,
    },
    Const {
        id: u64,
        stride: i64,
    },
    Custom {
        off: u64,
        pos: u64,
        dim: u64,
        ncols: u64,
        stride: i64,
    },
    Zi,
    Ch {
        base: u64,
    },
    Av {
        pos: u64,
        dim: u64,
    },
    Agv {
        pos: u64,
        dim: u64,
    },
    Pub {
        id: u64,
    },
    /// `ch[base..base+3]^j` (j >= 2), read from the challenge-powers table the
    /// launcher computes once per call at the head of scratch (see `opt.rs`).
    Pow {
        base: u64,
        j: u64,
    },
    /// Word `idx` of the hoisted-invariants table that follows the powers in
    /// the same per-launch region: a subexpression built only from challenges,
    /// powers, publics and air values, computed once per launch instead of
    /// once per row (see `Ir::tab`).
    Tab {
        idx: u64,
        dim: u64,
    },
}

impl Operand {
    pub fn dim(&self) -> u64 {
        match self {
            Operand::Tmp { dim, .. }
            | Operand::Cm { dim, .. }
            | Operand::Custom { dim, .. }
            | Operand::Av { dim, .. }
            | Operand::Agv { dim, .. }
            | Operand::Tab { dim, .. } => *dim,
            Operand::Ch { .. } | Operand::Pow { .. } => 3,
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
    pub n_bits: u64,
    /// Challenge-powers table the kernels read, one region per challenge and
    /// sorted by base: `(ch_base, n)` means `ch[ch_base..]^j` for j in 0..n
    /// (3 elems each). The regions sit end to end at the head of scratch, so a
    /// challenge's region starts at `pow_offset(ch_base)`.
    pub pow: Vec<(u64, u64)>,
    /// Row-invariant program run once per launch (thread 0 of the table
    /// kernel, after the powers): straight-line ops over invariant leaves and
    /// its own tmps. `tab_out` exports tmps to table words (`Operand::Tab`):
    /// (tmp id, word index, dim); `tab_words` is the exported size.
    pub tab: Vec<Instr>,
    pub tab_out: Vec<(u64, u64, u64)>,
    pub tab_words: u64,
    /// Standalone (non-Q) expression kernels have no scratch table: their
    /// challenge powers are computed once per thread into registers in the
    /// kernel prologue instead, and `Operand::Pow` reads `pwreg_<base>_<j>`.
    pub pow_in_regs: bool,
}

impl Ir {
    /// Words of the powers table (3 per power, summed over the challenges).
    pub fn pow_words(&self) -> u64 {
        self.pow.iter().map(|&(_, n)| 3 * n).sum()
    }
    /// Word offset of `base`'s region inside the powers table.
    pub fn pow_offset(&self, base: u64) -> u64 {
        self.pow.iter().take_while(|&&(b, _)| b != base).map(|&(_, n)| 3 * n).sum()
    }
    /// Total per-launch table words carved off the head of scratch.
    pub fn table_words(&self) -> u64 {
        self.pow_words() + self.tab_words
    }

    pub fn uses_zi(&self) -> bool {
        self.instrs.iter().any(|i| matches!(i.a, Operand::Zi) || matches!(i.b, Operand::Zi))
    }

    /// Output dimension of the expression: the final instruction's value (for
    /// non-Q expressions the result is left in the last tmp; for Q it is the
    /// explicit output store). None for an empty body or when a non-tmp store
    /// occurs before the end (non-canonical shape).
    pub fn out_dim(&self) -> Option<u64> {
        let last = self.instrs.last()?;
        let early_out = self.instrs[..self.instrs.len() - 1].iter().any(|i| !i.dst_is_tmp);
        if early_out {
            None
        } else {
            Some(last.ddim)
        }
    }
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

/// Merkle node words for a tree of `height` leaves -- mirrors MerkleTreeGL::getNumNodes / stark_info's getNumNodesMT.
fn num_nodes_mt(height: u64, arity: u64) -> u64 {
    let mut num = height;
    let mut lvl = height;
    while lvl > 1 {
        num += (arity - (lvl % arity)) % arity;
        let next = lvl.div_ceil(arity);
        num += next;
        lvl = next;
    }
    num * 4
}

/// Element offset of custom commit `cid`'s section (small or extended domain)
/// inside `pCustomCommitsFixed`. Mirrors stark_info.cpp setMapOffsets: per
/// commit, the small section (width*N), then the extended section
/// (width*NExt) followed by its Merkle tree nodes.
fn custom_section_offset(stark_info: &StarkInfo, cid: usize, extended: bool) -> anyhow::Result<u64> {
    let n = 1u64 << stark_info.stark_struct.n_bits;
    let next = 1u64 << stark_info.stark_struct.n_bits_ext;
    let arity = stark_info.stark_struct.merkle_tree_arity;
    let mut off = 0u64;
    for (i, cc) in stark_info.custom_commits.iter().enumerate() {
        let w0 = *cc
            .stage_widths
            .first()
            .ok_or_else(|| anyhow::anyhow!("custom commit {} ({i}) has empty stageWidths", cc.name))?;
        if w0 == 0 {
            continue;
        }
        if i == cid && !extended {
            return Ok(off);
        }
        off += w0 * n;
        if i == cid {
            return Ok(off);
        }
        off += w0 * next + num_nodes_mt(next, arity);
    }
    anyhow::bail!("custom commit {cid} has no section (stageWidths[0] == 0?)")
}

/// Resolve the AIR's Q (cExp) expression into typed IR. Returns
/// `UnhandledOperand` (as an `anyhow` error) if it references an operand type
/// the generator can't emit.
pub fn build_ir(stark_info: &StarkInfo, expr_info: &ExpressionsInfo) -> anyhow::Result<Ir> {
    build_ir_expr(stark_info, expr_info, stark_info.c_exp_id, true)
}

/// Resolve any expression by id. `extended = true` evaluates over the extended
/// domain (Q: operand strides are openingPoint * blowup); `extended = false`
/// over the trace domain (hint/witness expressions: strides are the raw
/// opening points, matching the evaluator's non-extended `nextStrides`).
pub fn build_ir_expr(
    stark_info: &StarkInfo,
    expr_info: &ExpressionsInfo,
    exp_id: i64,
    extended: bool,
) -> anyhow::Result<Ir> {
    let opening = &stark_info.opening_points;
    let blowup = if extended { stark_info.blowup() } else { 1 };
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
        .find(|e| e.exp_id == exp_id)
        .ok_or_else(|| anyhow::anyhow!("expId {exp_id} not found in expressionsCode"))?
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
            "custom" => {
                let cid = src.commit_id.ok_or_else(|| anyhow::anyhow!("custom operand without commitId"))? as usize;
                let pol = &stark_info.custom_commits_map[cid][src.id.unwrap() as usize];
                let name = &stark_info.custom_commits[cid].name;
                let ncols = *stark_info
                    .map_sections_n
                    .get(&format!("{name}0"))
                    .ok_or_else(|| anyhow::anyhow!("mapSectionsN missing section {name}0"))?;
                Operand::Custom {
                    off: custom_section_offset(stark_info, cid, extended)?,
                    pos: pol.stage_pos,
                    dim: pol.dim,
                    ncols,
                    stride: stride(src.prime.unwrap_or(0))?,
                }
            }
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
        // Only the binary field ops are modelled (by the emitter, the optimizer
        // and the host evaluator alike); anything else leaves the AIR on the
        // interpreter instead of failing deep inside a pass.
        if !matches!(step.op.as_str(), "add" | "sub" | "mul") || step.src.len() != 2 {
            return Err(anyhow::Error::new(UnhandledOperand(format!(
                "op {} with {} operands",
                step.op,
                step.src.len()
            ))));
        }
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
    Ok(Ir {
        instrs,
        ncols,
        n_constants,
        n_bits: stark_info.stark_struct.n_bits,
        pow: Vec::new(),
        tab: Vec::new(),
        tab_out: Vec::new(),
        tab_words: 0,
        pow_in_regs: false,
    })
}

/// Liveness + chunking + scratch-slot coloring for one chunk size.
pub struct ChunkPlan {
    pub n_chunks: usize,
    /// Chunk i covers ops `bounds[i]..bounds[i+1]`; len = n_chunks + 1.
    pub bounds: Vec<usize>,
    pub out_dim: u64,
    pub def_idx: HashMap<u64, usize>,
    pub dim_of: HashMap<u64, u64>,
    pub cut_temps: HashSet<u64>,
    pub total_slots: u64,
    slot_index: HashMap<u64, u64>,
}

impl ChunkPlan {
    pub fn chunk_of(&self, op_idx: usize) -> usize {
        chunk_of_bounds(&self.bounds, op_idx)
    }
    pub fn range(&self, chunk_idx: usize) -> (usize, usize) {
        (self.bounds[chunk_idx], self.bounds[chunk_idx + 1])
    }
    pub fn slot_index(&self, t: u64) -> u64 {
        self.slot_index[&t]
    }
}

/// Index of the chunk containing `op_idx` for sorted `bounds` (bounds[0] = 0).
pub fn chunk_of_bounds(bounds: &[usize], op_idx: usize) -> usize {
    // number of starts <= op_idx, minus one
    bounds.partition_point(|&b| b <= op_idx) - 1
}

/// Choose chunk boundaries minimizing the scratch words crossing them.
/// `cross[p]` = words of temps defined before op p and last used at or after
/// it; a cut at p costs cross[p] + CUT_PENALTY (a kernel launch per wave and
/// the load/store pair around it), chunks are at most `max_chunk` ops.
/// DP over positions, O(n_ops * max_chunk).
fn adaptive_bounds(
    n_ops: usize,
    max_chunk: usize,
    def_idx: &HashMap<u64, usize>,
    last_use: &HashMap<u64, usize>,
    dim_of: &HashMap<u64, u64>,
) -> Vec<usize> {
    const CUT_PENALTY: u64 = 2;
    // difference array: temp t crosses every boundary p with def < p <= last_use
    let mut diff = vec![0i64; n_ops + 2];
    for (&t, &d) in def_idx {
        if let Some(&lu) = last_use.get(&t) {
            if lu > d {
                let w = dim_of.get(&t).copied().unwrap_or(1) as i64;
                diff[d + 1] += w;
                diff[lu + 1] -= w;
            }
        }
    }
    let mut cross = vec![0u64; n_ops + 1];
    let mut acc = 0i64;
    for p in 0..=n_ops {
        acc += diff[p];
        cross[p] = acc.max(0) as u64;
    }
    // best[p] = min cost of cutting ops [0, p) into chunks (cut at p included, except p = n_ops)
    let inf = u64::MAX / 4;
    let mut best = vec![inf; n_ops + 1];
    let mut prev = vec![usize::MAX; n_ops + 1];
    best[0] = 0;
    for p in 1..=n_ops {
        let cut_cost = if p == n_ops { 0 } else { cross[p] + CUT_PENALTY };
        let lo = p.saturating_sub(max_chunk);
        let mut bq = inf;
        let mut bi = usize::MAX;
        for (off, &b) in best[lo..p].iter().enumerate() {
            if b < bq {
                bq = b;
                bi = lo + off;
            }
        }
        best[p] = bq + cut_cost;
        prev[p] = bi;
    }
    let mut bounds = vec![n_ops];
    let mut p = n_ops;
    while p > 0 {
        p = prev[p];
        bounds.push(p);
    }
    bounds.reverse();
    bounds
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
    // Cut points. Fixed every `chunk` ops, or (default) placed where few values
    // are live: every temp alive across a boundary costs a scratch slot word per
    // row and shrinks the launcher grid, so boundaries belong at the live-range
    // minima, subject to each chunk staying within the spill-free size.
    let adaptive = std::env::var("EXPS_ADAPTIVE_CHUNKS").map_or(true, |v| v != "0");
    let bounds: Vec<usize> = if adaptive && n_ops > chunk {
        adaptive_bounds(n_ops, chunk, &def_idx, &last_use, &dim_of)
    } else {
        let mut b: Vec<usize> = (0..n_ops).step_by(chunk).collect();
        b.push(n_ops);
        b
    };
    let n_chunks = bounds.len() - 1;
    let chunk_of = |op_idx: usize| chunk_of_bounds(&bounds, op_idx);

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

    Ok(ChunkPlan { n_chunks, bounds, out_dim, def_idx, dim_of, cut_temps, total_slots, slot_index })
}
