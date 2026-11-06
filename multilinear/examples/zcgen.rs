//! Prototype: emit CUDA for one AIR's zerocheck, straight from its `.mlinfo.bin` IR, the way
//! exps-codegen emits the univariate quotient. Two device functions:
//!   zc_eval_base  univariate-skip round: every leaf (column, row offset) is lifted inline from the
//!                 base trace at `row + offset` over the block's 2^l rows (no materialized tables),
//!                 and the IR runs in the base field
//!   zc_eval_ext   a later round: leaves are interpolated at X from their folded extension tables
//! Both return sum_i w_i * C_i over the EveryRow constraints. Usage:
//!   cargo run --release -p proofman-multilinear --example zcgen -- <air.mlinfo.bin> <out.cuh>
use std::collections::HashMap;
use std::fmt::Write;

use proofman_multilinear::{global_col, global_const_col, referenced_leaves, AirIr, Boundary, Op, Operand, SrcKind};

#[derive(Clone, Copy, PartialEq)]
enum Ty {
    B,
    E,
}

struct Gen<'a> {
    ir: &'a AirIr,
    ext_mode: bool,
    leaf_ix: HashMap<(SrcKind, u32, i32), usize>,
    leaf_col: Vec<(usize, i32)>,
    emitted: Vec<bool>,
    temp_ty: Vec<Ty>,
    out: String,
}

impl Gen<'_> {
    fn leaf(&mut self, o: &Operand) -> (String, Ty) {
        let key = (o.kind, o.idx, o.row_offset);
        let j = *self.leaf_ix.get(&key).unwrap_or_else(|| panic!("unreferenced leaf {key:?}"));
        if !self.emitted[j] {
            self.emitted[j] = true;
            let (c, off) = self.leaf_col[j];
            if self.ext_mode {
                writeln!(self.out, "    E L{j} = ZC_INTERP({j});").unwrap();
            } else {
                writeln!(self.out, "    gl64_t L{j} = ZC_LIFT({c}, {off});").unwrap();
            }
        }
        (format!("L{j}"), if self.ext_mode { Ty::E } else { Ty::B })
    }

    fn operand(&mut self, o: &Operand) -> (String, Ty) {
        match o.kind {
            SrcKind::Witness { .. } | SrcKind::Const => self.leaf(o),
            SrcKind::Custom { .. } => panic!("custom commits are not supported by this prototype"),
            SrcKind::Public => (format!("pubs[{}]", o.idx), Ty::B),
            SrcKind::Challenge => (format!("chal[{}]", o.idx), Ty::E),
            SrcKind::AirValue => (format!("airv[{}]", o.idx), Ty::E),
            SrcKind::AirGroupValue => (format!("agv[{}]", o.idx), Ty::E),
            SrcKind::ProofValue => (format!("pv[{}]", o.idx), Ty::E),
            SrcKind::Number => (format!("gl64_t({}ul)", self.ir.numbers[o.idx as usize]), Ty::B),
            SrcKind::Temp => (format!("t{}", o.idx), self.temp_ty[o.idx as usize]),
        }
    }

    /// One `__noinline__` function per `chunk` live instructions, so nvcc optimizes small pieces.
    /// Temps used past their chunk live in per-thread slot arrays (slots reused after the last use);
    /// leaves are re-lifted in every chunk that reads them; roots accumulate where they are defined.
    fn function(mut self, name: &str, sig: &str, args: &str, live: &[bool], chunk: usize) -> String {
        let ir = self.ir;
        let instrs: Vec<_> = ir.instrs.iter().filter(|ins| live[ins.dst as usize]).collect();
        let n_chunks = instrs.len().div_ceil(chunk).max(1);
        let nt = ir.n_temps as usize + 1;
        let leaf_ty = if self.ext_mode { Ty::E } else { Ty::B };
        let mut def = vec![usize::MAX; nt];
        let mut last = vec![0usize; nt];
        for (k, ins) in instrs.iter().enumerate() {
            let c = k / chunk;
            def[ins.dst as usize] = c;
            last[ins.dst as usize] = c;
            let ta = match ins.a.kind {
                SrcKind::Temp => self.temp_ty[ins.a.idx as usize],
                SrcKind::Witness { .. } | SrcKind::Const => leaf_ty,
                SrcKind::Public | SrcKind::Number => Ty::B,
                _ => Ty::E,
            };
            let tb = if ins.op == Op::Neg {
                Ty::B
            } else {
                match ins.b.kind {
                    SrcKind::Temp => self.temp_ty[ins.b.idx as usize],
                    SrcKind::Witness { .. } | SrcKind::Const => leaf_ty,
                    SrcKind::Public | SrcKind::Number => Ty::B,
                    _ => Ty::E,
                }
            };
            self.temp_ty[ins.dst as usize] = if ta == Ty::E || tb == Ty::E { Ty::E } else { Ty::B };
            for o in [&ins.a, &ins.b] {
                if o.kind == SrcKind::Temp && ins.op != Op::Neg || (o.kind == SrcKind::Temp && std::ptr::eq(o, &ins.a)) {
                    last[o.idx as usize] = last[o.idx as usize].max(c);
                }
            }
        }
        let roots: Vec<Operand> =
            ir.constraints.iter().filter(|c| c.boundary == Boundary::EveryRow).map(|c| c.root).collect();
        // slot allocation for temps that cross a chunk boundary
        let mut slot = vec![usize::MAX; nt];
        let (mut free_b, mut free_e): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        let (mut nb, mut ne) = (0usize, 0usize);
        let mut release: Vec<Vec<usize>> = vec![Vec::new(); n_chunks + 1];
        let mut out = String::new();
        let mut calls = String::new();
        for c in 0..n_chunks {
            for &t in &release[c] {
                if self.temp_ty[t] == Ty::E { free_e.push(slot[t]) } else { free_b.push(slot[t]) }
            }
            self.out = String::new();
            self.emitted = vec![false; self.emitted.len()];
            if c == 0 {
                for (i, r) in roots.iter().enumerate().filter(|(_, r)| r.kind != SrcKind::Temp) {
                    let (v, t) = self.operand(r);
                    let term = if t == Ty::E { format!("emul(w[{i}], {v})") } else { format!("emulb(w[{i}], {v})") };
                    writeln!(self.out, "    acc = eadd(acc, {term});").unwrap();
                }
            }
            for ins in &instrs[c * chunk..((c + 1) * chunk).min(instrs.len())] {
                let (a, ta) = self.operand_in(&ins.a, c, &def, &slot);
                let (b, tb) = if ins.op == Op::Neg { (String::new(), Ty::B) } else { self.operand_in(&ins.b, c, &def, &slot) };
                let ty = self.temp_ty[ins.dst as usize];
                let e = |s: &str, t: Ty| if t == Ty::E { s.to_string() } else { format!("EB({s})") };
                let rhs = match (ins.op, ta, tb) {
                    (Op::Neg, Ty::B, _) => format!("gl64_t(0ul) - {a}"),
                    (Op::Neg, Ty::E, _) => format!("esub(EZ, {a})"),
                    (Op::Add, Ty::B, Ty::B) => format!("{a} + {b}"),
                    (Op::Sub, Ty::B, Ty::B) => format!("{a} - {b}"),
                    (Op::Mul, Ty::B, Ty::B) => format!("{a} * {b}"),
                    (Op::Add, _, _) => format!("eadd({}, {})", e(&a, ta), e(&b, tb)),
                    (Op::Sub, _, _) => format!("esub({}, {})", e(&a, ta), e(&b, tb)),
                    (Op::Mul, Ty::E, Ty::B) => format!("emulb({a}, {b})"),
                    (Op::Mul, Ty::B, Ty::E) => format!("emulb({b}, {a})"),
                    (Op::Mul, Ty::E, Ty::E) => format!("emul({a}, {b})"),
                };
                let d = ins.dst as usize;
                let decl = if ty == Ty::E { "E" } else { "gl64_t" };
                writeln!(self.out, "    {decl} t{d} = {rhs};").unwrap();
                if last[d] > c {
                    let sl = if ty == Ty::E {
                        free_e.pop().unwrap_or_else(|| { ne += 1; ne - 1 })
                    } else {
                        free_b.pop().unwrap_or_else(|| { nb += 1; nb - 1 })
                    };
                    slot[d] = sl;
                    release[last[d] + 1].push(d);
                    writeln!(self.out, "    {}[{sl}] = t{d};", if ty == Ty::E { "XE" } else { "XB" }).unwrap();
                }
                for (i, r) in roots.iter().enumerate().filter(|(_, r)| r.kind == SrcKind::Temp && r.idx as usize == d) {
                    let term = if ty == Ty::E { format!("emul(w[{i}], t{d})") } else { format!("emulb(w[{i}], t{d})") };
                    let _ = r;
                    writeln!(self.out, "    acc = eadd(acc, {term});").unwrap();
                }
            }
            writeln!(out, "__device__ __noinline__ void {name}_c{c}({sig}, gl64_t* __restrict__ XB, E* __restrict__ XE, E& acc) {{\n{}}}", self.out).unwrap();
            writeln!(calls, "    {name}_c{c}({args}, XB, XE, acc);").unwrap();
        }
        format!(
            "{out}__device__ __forceinline__ E {name}({sig}) {{\n    gl64_t XB[{}]; E XE[{}]; E acc = EZ;\n{calls}    return acc;\n}}\n// {name}: {n_chunks} chunks, slots {nb} base + {ne} ext\n",
            nb.max(1),
            ne.max(1)
        )
    }

    fn operand_in(&mut self, o: &Operand, c: usize, def: &[usize], slot: &[usize]) -> (String, Ty) {
        if o.kind == SrcKind::Temp && def[o.idx as usize] != c {
            let t = self.temp_ty[o.idx as usize];
            let arr = if t == Ty::E { "XE" } else { "XB" };
            return (format!("{arr}[{}]", slot[o.idx as usize]), t);
        }
        self.operand(o)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ir = AirIr::load(std::path::Path::new(&args[1])).expect("load mlinfo");
    let mut leaves: Vec<(SrcKind, u32, i32)> = referenced_leaves(&ir).into_iter().collect();
    leaves.sort_by_key(|&(k, i, o)| (format!("{k:?}"), i, o));
    let leaf_ix: HashMap<_, _> = leaves.iter().enumerate().map(|(j, &l)| (l, j)).collect();
    let leaf_col: Vec<(usize, i32)> = leaves
        .iter()
        .map(|&(k, i, o)| match k {
            SrcKind::Witness { stage } => (global_col(&ir, stage, i), o),
            SrcKind::Const => (global_const_col(&ir, i), o),
            other => panic!("leaf kind {other:?}"),
        })
        .collect();
    // Only the EveryRow constraints' cones: the IR also carries the bus terms, which GKR evaluates.
    let mut live = vec![false; ir.n_temps as usize + 1];
    for c in ir.constraints.iter().filter(|c| c.boundary == Boundary::EveryRow) {
        if c.root.kind == SrcKind::Temp {
            live[c.root.idx as usize] = true;
        }
    }
    for ins in ir.instrs.iter().rev() {
        if live[ins.dst as usize] {
            for o in [&ins.a, &ins.b] {
                if o.kind == SrcKind::Temp {
                    live[o.idx as usize] = true;
                }
            }
        }
    }
    let n_live = ir.instrs.iter().filter(|ins| live[ins.dst as usize]).count();
    let n_cols = ir.total_cols();
    let n_roots = ir.constraints.iter().filter(|c| c.boundary == Boundary::EveryRow).count();
    let mk = |ext_mode| Gen {
        ir: &ir,
        ext_mode,
        leaf_ix: leaf_ix.clone(),
        leaf_col: leaf_col.clone(),
        emitted: vec![false; leaves.len()],
        temp_ty: vec![Ty::B; ir.n_temps as usize + 1],
        out: String::new(),
    };
    let common = "const gl64_t* __restrict__ pubs, const E* __restrict__ chal, const E* __restrict__ airv, \
                  const E* __restrict__ agv, const E* __restrict__ pv, const E* __restrict__ w";
    let mut src = String::new();
    writeln!(src, "// generated by zcgen from {} ({} instrs, {} leaves, {} roots)", ir.name, ir.instrs.len(), leaves.len(), n_roots).unwrap();
    writeln!(src, "#define ZC_N_COLS {n_cols}\n#define ZC_N_LEAVES {}\n#define ZC_N_ROOTS {n_roots}\n#define ZC_N_BITS {}\n#define ZC_DEG {}",
             leaves.len(), ir.n_bits, ir.max_constraint_degree).unwrap();
    let chunk: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1500);
    let common_args = "pubs, chal, airv, agv, pv, w";
    src += &mk(false).function(
        "zc_eval_base",
        &format!("const gl64_t* __restrict__ const* cols, long row0, long mask, const gl64_t* __restrict__ lag, {common}"),
        &format!("cols, row0, mask, lag, {common_args}"),
        &live,
        chunk,
    );
    src += &mk(true).function(
        "zc_eval_ext",
        &format!("const E* __restrict__ const* tabs, long i, gl64_t X, {common}"),
        &format!("tabs, i, X, {common_args}"),
        &live,
        chunk,
    );
    std::fs::write(&args[2], src).expect("write");
    eprintln!("{}: {n_live} of {} instrs live for the zerocheck", ir.name, ir.instrs.len());
    eprintln!("{}: {} instrs, {} leaves, {} cols, {} roots -> {}", ir.name, ir.instrs.len(), leaves.len(), n_cols, n_roots, args[2]);
}
