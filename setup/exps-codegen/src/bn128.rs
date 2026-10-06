//! The pilfflonk target: pilfflonk's `<air>.bin` (bytecode over BN128's scalar field, see
//! setup/pilfflonk/src/bytecode.rs) -> the shared IR -> the same optimizer (`opt.rs`, with
//! `Field::Bn128` number folding) -> CUDA over sppark's `BN128GPUScalarField` (Montgomery form,
//! fully reduced, so any exact rewrite gives the interpreter's bytes).
//!
//! The generated `<air>.exps.so` takes the interpreter's own `ExpressionLaunch`
//! (pil2-stark/src/pilfflonk/pilfflonk_expressions_kernels.hpp): the same operand tables, read
//! with indices fixed at generation time. Exports (C ABI):
//! * `pfexps_abi()` -> `PFEXPS_ABI`;
//! * `pfexps_covers(expId)` -> 1 for a generated expression;
//! * `pfexps_scratch_bytes()` -> device bytes `pfexps_launch` needs at `scratch`;
//! * `pfexps_launch(expId, launch, scratch)` -> 0 once queued on the legacy default stream.

use crate::field::{bn128_r, Bn128Numbers, Field};
use crate::ir::{Instr, Ir, Operand};
use anyhow::{bail, Context, Result};
use num_bigint::BigUint;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub const PFEXPS_ABI: u32 = 1;
/// Threads per block of the generated row kernels.
const BLK: u64 = 128;

/// The slice of `<air>.pilfflonkinfo.json` the generator reads.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PilfflonkInfo {
    pub name: String,
    pub n_bits: u64,
    pub n_stages: u64,
    pub c_exp_id: u64,
}

/// One expression of section 1 of `<air>.bin`, its ops as `(op, dest, a, b)`, each operand
/// `(type, arg1, arg2)`.
pub struct BinExpr {
    pub exp_id: u32,
    pub ops: Vec<(u32, u32, [u32; 3], [u32; 3])>,
}

/// Section 1 of `<air>.bin` (format revision 3).
pub struct Bin {
    pub n_stages: u32,
    pub exprs: Vec<BinExpr>,
    pub numbers: Vec<BigUint>,
}

struct Cursor<'a> {
    d: &'a [u8],
    p: usize,
}
impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let s = self.d.get(self.p..self.p + n).context("truncated .bin")?;
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn cstr(&mut self) -> Result<()> {
        let n = self.d[self.p..].iter().position(|&b| b == 0).context("unterminated string in .bin")?;
        self.p += n + 1;
        Ok(())
    }
}

const BIN_VERSION: u32 = 0x7066_0003;

pub fn read_bin(path: &Path) -> Result<Bin> {
    let d = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut c = Cursor { d: &d, p: 0 };
    if c.take(4)? != b"chps" {
        bail!("{}: not a chps file", path.display());
    }
    if c.u32()? != BIN_VERSION {
        bail!("{}: not a pilfflonk .bin of revision 3", path.display());
    }
    let n_sections = c.u32()?;
    let mut s1 = None;
    for _ in 0..n_sections {
        let id = c.u32()?;
        let size = c.u64()? as usize;
        let start = c.p;
        c.take(size)?;
        if id == 1 {
            s1 = Some(start);
        }
    }
    c.p = s1.context("no section 1")?;
    let (_version, n8) = (c.u32()?, c.u32()?);
    if n8 != 32 || BigUint::from_bytes_le(c.take(32)?) != *bn128_r() {
        bail!("{}: not over BN128's scalar field", path.display());
    }
    let n_stages = c.u32()?;
    let _maxima = (c.u32()?, c.u32()?, c.u32()?);
    let (n_ops, n_args, n_numbers, n_exprs) = (c.u32()?, c.u32()?, c.u32()?, c.u32()?);
    let mut heads = Vec::with_capacity(n_exprs as usize);
    for _ in 0..n_exprs {
        let f: Vec<u32> = (0..8).map(|_| c.u32()).collect::<Result<_>>()?;
        c.cstr()?;
        heads.push(f); // expId destId stage nTemp nOps opsOffset nArgs argsOffset
    }
    c.take(n_ops as usize)?;
    let args: Vec<u32> = (0..n_args).map(|_| c.u32()).collect::<Result<_>>()?;
    let numbers: Vec<BigUint> =
        (0..n_numbers).map(|_| Ok(BigUint::from_bytes_le(c.take(32)?))).collect::<Result<_>>()?;
    let mut exprs = Vec::new();
    for h in heads {
        let (n, off) = (h[4] as usize, h[7] as usize);
        let a = args.get(off..off + 8 * n).context("expression args out of range")?;
        let ops = a.chunks(8).map(|o| (o[0], o[1], [o[2], o[3], o[4]], [o[5], o[6], o[7]])).collect();
        exprs.push(BinExpr { exp_id: h[0], ops });
    }
    Ok(Bin { n_stages, exprs, numbers })
}

/// Expression `exp_id` of `bin` as SSA IR over BN128 (the bytecode reuses temporaries; this
/// renames them). Column operands keep the opening-point *index* in `stride`: shifts depend on
/// the domain and are read at run time. The last op is the output.
pub fn build_ir(bin: &Bin, exp_id: u32, n_bits: u64, nums: &std::sync::Arc<Bn128Numbers>) -> Result<Ir> {
    let e = bin.exprs.iter().find(|e| e.exp_id == exp_id).with_context(|| format!("no expression {exp_id}"))?;
    let ns = bin.n_stages;
    let (zi, tmp) = (ns + 2, ns + 4);
    let mut cur: HashMap<u32, u64> = HashMap::new(); // bytecode temporary -> SSA id
    let mut instrs = Vec::with_capacity(e.ops.len());
    for (k, &(op, dest, a, b)) in e.ops.iter().enumerate() {
        let operand = |[ty, a1, a2]: [u32; 3]| -> Result<Operand> {
            Ok(match ty {
                0 => Operand::Const { id: a1 as u64, stride: a2 as i64 },
                s if s <= ns + 1 => Operand::Cm { stage: s as u64, pos: a1 as u64, dim: 1, stride: a2 as i64 },
                t if t == zi && a1 == 1 => Operand::Zi,
                t if t == zi && a1 > 1 => Operand::Zb { boundary: a1 as u64 - 1 },
                t if t == tmp => Operand::Tmp { id: *cur.get(&a1).context("read of an unwritten temporary")?, dim: 1 },
                t if t == tmp + 2 => Operand::Pub { id: a1 as u64 },
                t if t == tmp + 3 => {
                    Operand::Num(nums.intern(bin.numbers.get(a1 as usize).context("number out of range")?.clone()))
                }
                t if t == tmp + 4 => Operand::Av { pos: a1 as u64, dim: 1 },
                t if t == tmp + 6 => Operand::Agv { pos: a1 as u64, dim: 1 },
                t if t == tmp + 7 => Operand::Ch { base: a1 as u64, dim: 1 },
                other => {
                    return Err(anyhow::Error::new(crate::ir::UnhandledOperand(format!("bn128 operand type {other}"))))
                }
            })
        };
        let (oa, ob) = (operand(a)?, operand(b)?);
        let (name, oa, ob) = match op {
            0 => ("add", oa, ob),
            1 => ("sub", oa, ob),
            2 => ("mul", oa, ob),
            3 => ("sub", ob, oa),
            other => bail!("op {other}"),
        };
        let last = k + 1 == e.ops.len();
        let id = k as u64;
        instrs.push(Instr {
            op: name.to_string(),
            a: oa,
            b: ob,
            dst_is_tmp: !last,
            dst_id: (!last).then_some(id),
            ddim: 1,
            idx: k,
        });
        cur.insert(dest, id);
    }
    Ok(Ir {
        instrs,
        ncols: HashMap::new(),
        n_constants: 0,
        n_bits,
        pow: Vec::new(),
        tab: Vec::new(),
        tab_out: Vec::new(),
        tab_words: 0,
        pow_in_regs: false,
        field: Field::Bn128(nums.clone()),
    })
}

/// `v` (canonical) in Montgomery form, eight little-endian 32-bit limbs.
fn mont_limbs(v: &BigUint) -> [u32; 8] {
    let m = (v << 256u32) % bn128_r();
    let mut out = [0u32; 8];
    for (i, d) in m.to_u32_digits().into_iter().enumerate() {
        out[i] = d;
    }
    out
}

// The scalar tables, in OperandTables::scalars order (publics … challenges).
const SC_PUB: usize = 0;
const SC_AV: usize = 2;
const SC_AGV: usize = 4;
const SC_CH: usize = 5;

/// The operand reads of one kernel, resolved once in its prologue.
#[derive(Default)]
struct Prologue {
    types: HashSet<u64>,     // column types whose columnStart is read
    shifts: HashSet<i64>,    // opening indices
    zerofiers: HashSet<u64>, // boundaries
    scalars: HashSet<usize>, // scalar tables
}

impl Prologue {
    fn note(&mut self, o: &Operand) {
        match o {
            Operand::Const { stride, .. } => {
                self.types.insert(0);
                self.shifts.insert(*stride);
            }
            Operand::Cm { stage, stride, .. } => {
                self.types.insert(*stage);
                self.shifts.insert(*stride);
            }
            Operand::Zi => {
                self.zerofiers.insert(0);
            }
            Operand::Zb { boundary } => {
                self.zerofiers.insert(*boundary);
            }
            Operand::Pub { .. } => {
                self.scalars.insert(SC_PUB);
            }
            Operand::Av { .. } => {
                self.scalars.insert(SC_AV);
            }
            Operand::Agv { .. } => {
                self.scalars.insert(SC_AGV);
            }
            Operand::Ch { .. } => {
                self.scalars.insert(SC_CH);
            }
            _ => {}
        }
    }
    fn lines(&self, rows: bool) -> String {
        let mut l = Vec::new();
        let sorted = |s: &HashSet<u64>| {
            let mut v: Vec<u64> = s.iter().copied().collect();
            v.sort();
            v
        };
        if rows {
            l.push("  const uint64_t mask = t.mask;".to_string());
            for ty in sorted(&self.types) {
                l.push(format!("  const uint32_t cs{ty} = t.columnStart[{ty}];"));
            }
            let mut sh: Vec<i64> = self.shifts.iter().copied().collect();
            sh.sort();
            for k in sh {
                l.push(format!("  const uint64_t sh{k} = t.shifts[{k}];"));
            }
            for b in sorted(&self.zerofiers) {
                l.push(format!(
                    "  const Element* __restrict__ zi{b} = (const Element*)t.zerofiers[{b}]; const uint64_t zm{b} = t.zerofierMasks[{b}];"
                ));
            }
        }
        let mut sc: Vec<usize> = self.scalars.iter().copied().collect();
        sc.sort();
        for s in sc {
            l.push(format!("  const Element* __restrict__ sc{s} = (const Element*)t.scalars[{s}];"));
        }
        l.join("\n")
    }
}

/// An element from its Montgomery limbs.
fn num_expr(nums: &Bn128Numbers, h: u64) -> String {
    let l = mont_limbs(&nums.value(h));
    format!("pf_el({}u,{}u,{}u,{}u,{}u,{}u,{}u,{}u)", l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7])
}

/// The expression reading a non-tmp operand.
fn load(o: &Operand, ir: &Ir, nums: &Bn128Numbers) -> String {
    let col = |ty: u64, pos: u64, k: i64| format!("((const Element*)t.columns[cs{ty} + {pos}])[(row + sh{k}) & mask]");
    match o {
        Operand::Const { id, stride } => col(0, *id, *stride),
        Operand::Cm { stage, pos, stride, .. } => col(*stage, *pos, *stride),
        Operand::Zi => "zi0[row & zm0]".to_string(),
        Operand::Zb { boundary } => format!("zi{boundary}[row & zm{boundary}]"),
        Operand::Pub { id } => format!("sc{SC_PUB}[{id}]"),
        Operand::Av { pos, .. } => format!("sc{SC_AV}[{pos}]"),
        Operand::Agv { pos, .. } => format!("sc{SC_AGV}[{pos}]"),
        Operand::Ch { base, .. } => format!("sc{SC_CH}[{base}]"),
        Operand::Num(h) => num_expr(nums, *h),
        Operand::Pow { base, j, .. } => format!("pw[{}]", ir.pow_offset(*base) + j),
        Operand::Tab { idx, .. } => format!("pw[{}]", ir.pow_words() + idx),
        Operand::Tmp { .. } | Operand::Custom { .. } => unreachable!("not a bn128 load"),
    }
}

fn fr_op(op: &str) -> &'static str {
    match op {
        "add" => "Fr::add",
        "sub" => "Fr::sub",
        "mul" => "pf_mul",
        other => panic!("unexpected op {other}"),
    }
}

/// Straight-line body: tmps are `t<id>`, the output `qq`.
fn body(instrs: &[Instr], ir: &Ir, nums: &Bn128Numbers, indent: &str) -> String {
    let mut l = Vec::with_capacity(instrs.len());
    let opnd = |o: &Operand| match o.as_tmp() {
        Some((id, _)) => format!("t{id}"),
        None => load(o, ir, nums),
    };
    for i in instrs {
        let dst = if i.dst_is_tmp { format!("const Element t{}", i.dst_id.unwrap()) } else { "qq".to_string() };
        l.push(format!("{indent}{dst} = {}({}, {});", fr_op(&i.op), opnd(&i.a), opnd(&i.b)));
    }
    l.join("\n")
}

const HEADER: &str = r#"#include "pilfflonk/pilfflonk_cuda.cuh"
#include "pilfflonk/pilfflonk_expressions_kernels.hpp"
#include <cstdio>
using PilFflonk::ExpressionLaunch;
using PilFflonk::OperandTables;
static_assert(sizeof(Element) == PilFflonk::OPERAND_BYTES, "an operand is one element");
__device__ __forceinline__ Element pf_el(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4,
                                         uint32_t a5, uint32_t a6, uint32_t a7) {
  Element r; r[0]=a0; r[1]=a1; r[2]=a2; r[3]=a3; r[4]=a4; r[5]=a5; r[6]=a6; r[7]=a7; return r;
}
// Out of line: inlined, cicc took >14 min on Wrap's Q (1589 ops); called, 27 s and 2x faster than
// the interpreter.
__device__ __noinline__ Element pf_mul(const Element a, const Element b) { return Fr::mul(a, b); }
"#;

/// The TU of one expression: its table kernel (challenge powers + hoisted invariants, one
/// thread), its row kernel and a host launcher `pf_launch_e<expId>`.
fn emit_expr(ir: &Ir, exp_id: u32, nums: &Bn128Numbers) -> (String, u64) {
    let mut rows = Prologue::default();
    for i in &ir.instrs {
        rows.note(&i.a);
        rows.note(&i.b);
    }
    let mut tabp = Prologue::default();
    for i in &ir.tab {
        tabp.note(&i.a);
        tabp.note(&i.b);
    }
    if !ir.pow.is_empty() {
        tabp.scalars.insert(SC_CH);
    }
    let words = ir.table_words();
    let mut s = String::new();
    if words > 0 {
        let mut pw = Vec::new();
        for &(base, n) in &ir.pow {
            let o = ir.pow_offset(base);
            pw.push(format!("  pw[{o}] = Fr::one();"));
            if n > 1 {
                pw.push(format!("  pw[{}] = sc{SC_CH}[{base}];", o + 1));
            }
            if n > 2 {
                pw.push(format!(
                    "  for (uint32_t j = 2; j < {n}u; ++j) pw[{o} + j] = pf_mul(pw[{o} + j - 1], sc{SC_CH}[{base}]);"
                ));
            }
        }
        let mut tb = body(&ir.tab, ir, nums, "  ");
        for &(tmp, idx, _) in &ir.tab_out {
            tb.push_str(&format!("\n  pw[{}] = t{tmp};", ir.pow_words() + idx));
        }
        s.push_str(&format!(
            "__global__ void pf_tab_e{exp_id}(const OperandTables t, Element* __restrict__ pw) {{\n{}\n{}\n{}\n}}\n",
            tabp.lines(false),
            pw.join("\n"),
            tb
        ));
    }
    s.push_str(&format!(
        "__global__ void __launch_bounds__({BLK}) pf_rows_e{exp_id}(const OperandTables t, uint64_t size, Element* __restrict__ dest, uint64_t stride, const Element* __restrict__ pw) {{
{}
  for (uint64_t row = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; row < size; row += uint64_t(gridDim.x) * blockDim.x) {{
    Element qq;
{}
    dest[row * stride] = qq;
  }}
}}
static int pf_launch_e{exp_id}(const ExpressionLaunch* l, void* scratch) {{
  Element* pw = static_cast<Element*>(scratch);
{}
  const uint64_t blocks = std::min<uint64_t>((l->size + {BLK} - 1) / {BLK}, uint64_t(1) << 20);
  pf_rows_e{exp_id}<<<uint32_t(blocks), {BLK}>>>(l->tables, l->size, static_cast<Element*>(l->dest), l->stride, pw);
  return int(cudaGetLastError());
}}
",
        rows.lines(true),
        body(&ir.instrs, ir, nums, "    "),
        if words > 0 { format!("  pf_tab_e{exp_id}<<<1, 1>>>(l->tables, pw);") } else { String::new() },
    ));
    (s, words * 32)
}

/// The whole `.cu` of an AIR: every expression's TU and the C ABI.
pub fn emit_tu(items: &[(u32, Ir)], nums: &Bn128Numbers) -> String {
    let mut s = String::from(HEADER);
    let mut scratch = 0u64;
    for (e, ir) in items {
        let (t, b) = emit_expr(ir, *e, nums);
        s.push_str(&t);
        scratch = scratch.max(b);
    }
    let cases: String =
        items.iter().map(|(e, _)| format!("  case {e}u: return pf_launch_e{e}(l, scratch);\n")).collect();
    let covers: String = items.iter().map(|(e, _)| format!("  case {e}u: return 1;\n")).collect();
    s.push_str(&format!(
        r#"extern "C" unsigned pfexps_abi() {{ return {PFEXPS_ABI}u; }}
extern "C" int pfexps_covers(unsigned expId) {{
  switch (expId) {{
{covers}  default: return 0;
  }}
}}
extern "C" unsigned long long pfexps_scratch_bytes() {{ return {scratch}ull; }}
extern "C" int pfexps_launch(unsigned expId, const ExpressionLaunch* l, void* scratch) {{
  if (l->size == 0) return 0;
  switch (expId) {{
{cases}  default: return -1;
  }}
}}
"#
    ));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn montgomery_one_is_ffiasm_one() {
        // R mod r: ffiasm's Fr one in Montgomery form
        let l = mont_limbs(&BigUint::from(1u32));
        assert_eq!(&l[..2], &[0x4fff_fffb, 0xac96_341c]);
        assert_eq!(l[7], 0x0e0a_77c1);
    }

    /// A Horner chain over a challenge with a repeated subterm: SSA renaming of reused
    /// temporaries, sub_swap, number folding and the rewrites stay exact over BN128.
    #[test]
    fn optimizer_is_exact_over_bn128() {
        // nStages 2: tmp = 6, numbers = 9, challenges = 13
        let (tmp, num, ch) = (6u32, 9u32, 13u32);
        let c = |p: u32| [1u32, p, 1];
        let t = |id: u32| [tmp, id, 0];
        let mut ops = vec![(2, 0, c(0), c(1)), (0, 0, t(0), [num, 0, 0]), (0, 1, t(0), [num, 1, 0])];
        for k in 0..6u32 {
            ops.push((2, 2, c(k), c(1)));
            ops.push((3, 2, t(0), t(2))); // t2 = t2 - t0
            ops.push((2, 1, t(1), [ch, 0, 0]));
            ops.push((0, 1, t(1), t(2)));
        }
        let bin = Bin {
            n_stages: 2,
            exprs: vec![BinExpr { exp_id: 5, ops }],
            numbers: vec![BigUint::from(7u32), bn128_r() - 1u32],
        };
        let nums = Bn128Numbers::new();
        let ir = build_ir(&bin, 5, 4, &nums).unwrap();
        let (o, _) = crate::opt::optimize(&ir);
        crate::check::equivalent(&ir, &o, 3).unwrap();
        assert!(o.instrs.len() < ir.instrs.len());
        let cu = emit_tu(&[(5, o)], &nums);
        assert!(cu.contains("pf_rows_e5") && cu.contains("pfexps_launch"));
    }
}
