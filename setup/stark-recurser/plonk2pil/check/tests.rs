//! Synthetic airs: a plonk addition and multiplication over 3 connected columns, a public, and one
//! custom-gate use placed as [`gates_of`] says. The untampered air is accepted; each tampering is
//! rejected for the reason it names.

use std::collections::BTreeMap;

use num_bigint::BigUint;
use pil_info::cfg::{FieldCfg, Opening};
use pil_info::core::ir::{Boundary, ColType, Column, Constraint, ConstraintSystem, Expr, Hint, HintField, HintValue};
use proofman_common::exec_format::{ExecAddition, ExecFile, ExecLayout};
use proofman_fields::{Bn128, Field, Goldilocks};

use super::fixed::FixedSource;
use super::gates::{fill_values, gates_of, mul_of, Param};
use super::pil::fe;
use super::{check_plonk, CheckError, CheckInput, Template};
use crate::plonk2pil::field::{PlonkField, R1csPrime};
use crate::plonk2pil::r1cs::types::{CustomGate, CustomGateUse, R1csConstraint, R1csFile, R1csHeader};

fn k<F: PlonkField>(v: u64) -> F {
    fe(&BigUint::from(v))
}

fn wit(c: usize, off: i64) -> Expr {
    Expr::Col { col: ColType::Witness, commit_id: None, id: c as u32, offset: off }
}

/// Wires: 1 the public, 2 and 3 private, 4.. a gate's signals, then the addition `w2 + w3`.
const CONN: usize = 3;

/// A synthetic air under construction, its fixed columns by `NAME` or `NAME[i]`.
#[derive(Clone)]
struct Air<F> {
    n: usize,
    width: usize,
    fixed: Vec<(String, Vec<F>)>,
    map: Vec<Vec<u32>>,
    constraints: Vec<Expr>,
    template: Template,
    gate: Option<(CustomGate<F>, Vec<CustomGateUse>)>,
    n_signals: usize,
    /// The pilout's publics (the r1cs has 1).
    publics: usize,
    /// The addition hint's coefficients.
    add: [u64; 2],
    /// A cell `(col, row)` cut out of its copy cycle in `S`.
    cut: Option<(usize, usize)>,
    /// An exec map entry `(row, col, wire)` that `S` does not follow.
    lie: Option<(usize, usize, u32)>,
}

impl<F: PlonkField> Air<F> {
    fn new(n: usize, width: usize, file: &str, params: &[(&str, u64)]) -> Self {
        let params = params.iter().map(|&(k, v)| (k.to_string(), v)).collect::<BTreeMap<_, _>>();
        let template = Template { file: file.into(), params };
        let (fixed, map) = (Vec::new(), vec![vec![0; width]; n]);
        Air {
            n,
            width,
            fixed,
            map,
            constraints: vec![],
            template,
            gate: None,
            n_signals: 0,
            publics: 1,
            add: [1, 1],
            cut: None,
            lie: None,
        }
    }

    fn id(&mut self, name: &str) -> usize {
        match self.fixed.iter().position(|c| c.0 == name) {
            Some(i) => i,
            None => {
                self.fixed.push((name.to_string(), vec![F::ZERO; self.n]));
                self.fixed.len() - 1
            }
        }
    }

    fn set(&mut self, name: &str, row: usize, v: F) {
        let (i, n) = (self.id(name), self.n);
        self.fixed[i].1[row % n] = v;
    }

    fn col(&mut self, name: &str) -> Expr {
        Expr::Col { col: ColType::Fixed, commit_id: None, id: self.id(name) as u32, offset: 0 }
    }

    /// The plonk rows and the public's row from `at` (after any gate: the addition wire follows its signals).
    fn plonk(mut self, at: usize) -> Self {
        let q = (0..5).map(|i| self.col(&format!("C[{i}]"))).collect::<Vec<_>>();
        let (plonk, pubrow) = (self.col("PLONK"), self.col("PUBLICS_ROW"));
        let mul = |a: Expr, b: Expr| Expr::Mul(Box::new(a), Box::new(b));
        let add = |a: Expr, b: Expr| Expr::Add(Box::new(a), Box::new(b));
        let body = [mul(mul(q[0].clone(), wit(0, 0)), wit(1, 0)), mul(q[1].clone(), wit(0, 0))]
            .into_iter()
            .chain([mul(q[2].clone(), wit(1, 0)), mul(q[3].clone(), wit(2, 0)), q[4].clone()])
            .reduce(add)
            .unwrap();
        self.constraints.push(mul(plonk, body));
        self.constraints.push(mul(pubrow, Expr::Sub(Box::new(wit(0, 0)), Box::new(Expr::Public(0)))));
        let addw = self.addition_wire();
        self.map[at][..3].copy_from_slice(&[2, 3, addw]);
        self.map[at + 1][..3].copy_from_slice(&[addw, 3, 1]);
        self.map[at + 2][0] = 1;
        for (row, c, v) in
            [(at, 1, F::ONE), (at, 2, F::ONE), (at, 3, -F::ONE), (at + 1, 0, F::ONE), (at + 1, 3, -F::ONE)]
        {
            self.set(&format!("C[{c}]"), row, v);
        }
        self.set("PLONK", at, F::ONE);
        self.set("PLONK", at + 1, F::ONE);
        self.set("PUBLICS_ROW", at + 2, F::ONE);
        self
    }

    fn addition_wire(&self) -> u32 {
        4 + self.n_signals as u32
    }

    /// One use of `name` at `base`, as the template places it, with `params`.
    fn gate(mut self, name: &str, base: usize, params: Vec<F>) -> Self {
        let g = gates_of(&self.template).unwrap().into_iter().find(|g| g.name == name).unwrap();
        self.n_signals = g.n_signals;
        for c in &g.shapes[0] {
            self.map[(base as i64 + c.off) as usize % self.n][c.col as usize] = 4 + c.sig as u32;
        }
        self.set(g.anchor, base, F::ONE);
        for (c, offs) in &g.band {
            for &o in offs {
                self.set(c, base + o as usize, F::ONE);
            }
        }
        if let Param::Flags(rows) = &g.param {
            for &o in rows {
                self.set("FLAGS", base + o as usize, params[0]);
            }
        }
        for (c, vals) in fill_values::<F>(&g, base).unwrap() {
            vals.iter().enumerate().for_each(|(i, v)| self.set(&c, base + i, *v));
        }
        if g.name.starts_with("Blake3Compress") && self.template.file == "blake3_bn128/wrap" {
            mul_of::<F>().into_iter().enumerate().for_each(|(i, v)| self.set("MSG_MUL", base + i, v));
        }
        // a constraint that reads the anchor and every bound column (tier C)
        let mut e = self.col(g.anchor);
        let mut cols: Vec<u32> = g.shapes[0].iter().map(|c| c.col).collect();
        cols.sort();
        cols.dedup();
        for c in cols {
            e = Expr::Mul(Box::new(e), Box::new(wit(c as usize, 0)));
        }
        self.constraints.push(e);
        let template_name = name.split('/').next().unwrap().to_string();
        let signals = (0..g.n_signals as u64).map(|s| 4 + s).collect();
        self.gate = Some((CustomGate { template_name, parameters: params }, vec![CustomGateUse { id: 0, signals }]));
        self
    }

    /// The artefacts, with each witness cell's copies linked by `S`.
    fn check(&self) -> Result<super::CheckReport, CheckError> {
        let (n, bits) = (self.n, self.n.trailing_zeros());
        let w = F::ROOTS_OF_UNITY[bits as usize];
        let ids: Vec<F> = (0..n).scan(F::ONE, |x, _| Some(std::mem::replace(x, *x * w))).collect();
        let kj: Vec<F> = (0..CONN).scan(F::ONE, |x, _| Some(std::mem::replace(x, *x * F::K))).collect();
        let mut cycles: BTreeMap<u32, Vec<(usize, usize)>> = BTreeMap::new();
        for (r, row) in self.map.iter().enumerate() {
            for (c, &wire) in row.iter().enumerate().take(CONN).filter(|(_, &w)| w != 0) {
                if self.cut != Some((c, r)) {
                    cycles.entry(wire).or_default().push((c, r));
                }
            }
        }
        let mut s: Vec<Vec<F>> = kj.iter().map(|&k| ids.iter().map(|&x| k * x).collect()).collect();
        for cyc in cycles.values() {
            for (i, &(c, r)) in cyc.iter().enumerate() {
                let (c2, r2) = cyc[(i + 1) % cyc.len()];
                s[c][r] = kj[c2] * ids[r2];
            }
        }
        let mut fixed = self.fixed.clone();
        fixed.push(("ID".into(), ids));
        fixed.extend(s.into_iter().enumerate().map(|(j, v)| (format!("S[{j}]"), v)));
        let id_of = |name: &str| fixed.iter().position(|c| c.0 == name).unwrap() as u32;
        let fcol = |name: &str| Expr::Col { col: ColType::Fixed, commit_id: None, id: id_of(name), offset: 0 };
        let entry = |name: &str, value: HintValue| HintField { name: Some(name.into()), value };
        let ex = |e: Expr| HintField { name: None, value: HintValue::Expr(e) };
        let konst = |v: BigUint| HintValue::Expr(Expr::Const(v));
        let mut hints = Vec::new();
        for (j, kj) in kj.iter().enumerate() {
            let id = Expr::Mul(Box::new(Expr::Const(kj.as_canonical_biguint())), Box::new(fcol("ID")));
            for (ty, rhs) in [(0u32, id), (1, fcol(&format!("S[{j}]")))] {
                let fields = vec![
                    entry("name_piop", HintValue::Str("Connection".into())),
                    entry("deg_sel", konst(BigUint::ZERO)),
                    entry("opids", HintValue::Array(vec![ex(Expr::Const(BigUint::from(7u32)))])),
                    entry("type_piop", konst(BigUint::from(ty))),
                    entry("expressions", HintValue::Array(vec![ex(wit(j, 0)), ex(rhs)])),
                ];
                hints.push(Hint { name: "gprod_debug_data".into(), fields });
            }
        }
        let column = |name: &str, id: usize, stage| {
            let (base, idx) = match name.strip_suffix(']').and_then(|n| n.split_once('[')) {
                Some((b, i)) => (b, Some(vec![i.parse().unwrap()])),
                None => (name, None),
            };
            Column { name: format!("Air.{base}"), stage, dim: 1, id: id as u32, lengths: idx }
        };
        let (field, opening) = match F::PRIME {
            R1csPrime::Goldilocks => (FieldCfg::goldilocks(), Opening::Deep),
            R1csPrime::Bn128 => (FieldCfg::bn128(), Opening::Shplonk),
        };
        let every = |e: Expr| Constraint { e, boundary: Boundary::EveryRow, label: String::new() };
        let cs = ConstraintSystem {
            name: String::new(),
            field,
            opening,
            n_stages: 2,
            n_rows_bits: bits,
            exprs: vec![],
            constraints: self.constraints.iter().cloned().map(every).collect(),
            witness: (0..self.width).map(|c| column(&format!("a[{c}]"), c, 1)).collect(),
            fixed: fixed.iter().enumerate().map(|(i, c)| column(&c.0, i, 0)).collect(),
            custom: vec![],
            challenges: vec![],
            publics: vec![Default::default(); self.publics],
            proof_values: vec![],
            air_values: vec![],
            airgroup_values: vec![],
            hints,
            custom_commits: vec![],
        };
        let n_vars = self.addition_wire();
        let lc = |t: &[(u32, u64)]| t.iter().map(|&(w, c)| (w, k::<F>(c))).collect();
        let (gates, uses) = self.gate.clone().map_or((vec![], vec![]), |(g, u)| (vec![g], u));
        let r1cs = R1csFile {
            header: R1csHeader {
                n8: 0,
                prime_bytes: vec![],
                n_vars,
                n_outputs: 1,
                n_pub_inputs: 0,
                n_prv_inputs: 2,
                n_labels: 0,
                n_constraints: 1,
                use_custom_gates: !gates.is_empty(),
            },
            // (w2 + w3)·w3 = w1
            constraints: vec![R1csConstraint { a: lc(&[(2, 1), (3, 1)]), b: lc(&[(3, 1)]), c: lc(&[(1, 1)]) }],
            wire_to_label: vec![],
            custom_gates: gates,
            custom_gates_uses: uses,
        };
        let mut map = self.map.concat();
        if let Some((r, c, w)) = self.lie {
            map[r * self.width + c] = w;
        }
        let exec = ExecFile {
            layout: ExecLayout::new::<F>(n_vars as usize, 1, n, self.width),
            additions: vec![ExecAddition { wires: [2, 3], coeffs: self.add.map(k) }],
            map,
            band_aux: 0,
            bands: vec![],
        };
        let fixed = Cols(fixed.into_iter().map(|c| c.1).collect());
        check_plonk(&CheckInput { r1cs: &r1cs, cs: &cs, fixed: &fixed, exec: &exec, template: &self.template })
    }
}

struct Cols<F>(Vec<Vec<F>>);

impl<F: PlonkField> FixedSource<F> for Cols<F> {
    fn column(&self, id: u32) -> Result<Vec<F>, CheckError> {
        Ok(self.0[id as usize].clone())
    }
}

/// `air` with `f` applied, checked.
fn tampered<F: PlonkField>(air: &Air<F>, f: impl FnOnce(&mut Air<F>)) -> CheckError {
    let mut t = air.clone();
    f(&mut t);
    t.check().expect_err("tampered artefacts are rejected")
}

fn plonk_air() -> Air<Goldilocks> {
    Air::new(8, CONN, "poseidon2/aggregator", &[]).plonk(0)
}

#[test]
fn the_plonk_air_is_accepted_and_its_tamperings_rejected() {
    let air = plonk_air();
    let rep = air.check().unwrap();
    assert_eq!((rep.publics, rep.a.additions), (1, 1));
    let g = |v| k::<Goldilocks>(v);
    assert!(matches!(tampered(&air, |a| a.set("C[0]", 1, g(2))), CheckError::NotImplied { .. }));
    assert!(matches!(tampered(&air, |a| a.set("PLONK", 1, g(0))), CheckError::NotImplied { .. }));
    assert!(matches!(tampered(&air, |a| a.set("PUBLICS_ROW", 2, g(0))), CheckError::PublicUnbound { public: 0, .. }));
    // a pilout public the r1cs does not have would be unbound
    assert!(matches!(tampered(&air, |a| a.publics = 2), CheckError::Artefact(_)));
    // exec: w2 also claimed at a cell of another class
    assert!(matches!(tampered(&air, |a| a.lie = Some((3, 0, 2))), CheckError::WireSplit { wire: 2 }));
}

#[test]
fn a_cut_copy_or_a_wrong_addition_hint_is_rejected() {
    let air = plonk_air();
    // w3's second cell cut out of its cycle: the exec map puts w3 in two classes
    assert!(matches!(tampered(&air, |a| a.cut = Some((1, 1))), CheckError::WireSplit { wire: 3 }));
    // the hint says w2 + 2·w3, the plonk row enforces w2 + w3: the addition is never defined
    assert!(matches!(tampered(&air, |a| a.add = [1, 2]), CheckError::NotImplied { .. }));
}

#[test]
fn a_poseidon_band_is_located_and_its_anchor_checked() {
    let air = Air::<Goldilocks>::new(16, 16, "poseidon2/aggregator", &[]).gate("Poseidon2_16", 0, vec![]).plonk(8);
    assert_eq!(air.check().unwrap().gates.bands, 1);
    let e = tampered(&air, |a| {
        a.set("POSEIDONSPONGE", 0, Goldilocks::ZERO);
        a.set("POSEIDONSPONGE", 1, Goldilocks::ONE);
    });
    assert!(matches!(e, CheckError::GateNotFound { .. }), "{e}");
    let e = tampered(&air, |a| a.set("POSEIDON_FINAL", 4, Goldilocks::ZERO));
    assert!(matches!(e, CheckError::FixedPattern { .. }), "{e}");
}

#[test]
fn a_blake3_block_is_located_and_its_flags_checked() {
    let flags = vec![k::<Goldilocks>(3), Goldilocks::ZERO];
    let air = Air::new(64, 18, "blake3/aggregator", &[("LANES", 1)]).gate("Blake3Compress/chunk", 0, flags).plonk(58);
    assert_eq!(air.check().unwrap().gates.uses, 1);
    let e = tampered(&air, |a| a.set("FLAGS", 10, k(2)));
    assert!(matches!(e, CheckError::GateParam { .. }), "{e}");
    // a use with fewer signals than its template would bind fewer cells
    let e = tampered(&air, |a| _ = a.gate.as_mut().unwrap().1[0].signals.pop());
    assert!(matches!(e, CheckError::GateParam { .. }), "{e}");
}

#[test]
fn two_uses_sharing_an_unbound_signal_are_rejected() {
    // EvPol4 binds 21 of its 27 signals; a second use on the same cells shares the other 6
    let air = Air::<Goldilocks>::new(8, 21, "poseidon2/aggregator", &[]).gate("EvPol4", 0, vec![]).plonk(4);
    assert_eq!(air.check().unwrap().gates.uses, 1);
    let e = tampered(&air, |a| {
        let uses = &mut a.gate.as_mut().unwrap().1;
        uses.push(uses[0].clone());
    });
    assert!(matches!(&e, CheckError::GateParam { why, .. } if why.contains("shared")), "{e}");
}

#[test]
fn range_check_is_exactly_one_on_located_num2bytes_rows() {
    let mut air = Air::<Bn128>::new(16, 6, "poseidon_bn128/wrap", &[]).gate("Num2Bytes", 4, vec![k(80)]).plonk(8);
    let mut w = Bn128::ONE;
    for j in 0..5 {
        air.set(&format!("K[{j}]"), 4, w);
        w *= k(1 << 16);
    }
    assert_eq!(air.check().unwrap().gates.uses, 1);
    // RANGE_CHECK is the lookup's selector: −1 on a spare row would cancel a real row's term
    for (row, v) in [(12, -Bn128::ONE), (4, k(2))] {
        let e = tampered(&air, |a| a.set("RANGE_CHECK", row, v));
        assert!(matches!(&e, CheckError::FixedPattern { column, .. } if column == "RANGE_CHECK"), "{row}: {e}");
    }
}

#[test]
fn the_poseidon_wrap_checks_k_and_the_range_table() {
    let mut air = Air::<Bn128>::new(128, 5, "poseidon_bn128/wrap", &[]).gate("PoseidonT", 8, vec![k(5)]).plonk(100);
    for r in 0..128 {
        air.set("RANGE", r, k(r as u64));
    }
    assert_eq!(air.check().unwrap().gates.bands, 1);
    let e = tampered(&air, |a| a.set("K[2]", 15, k(1)));
    assert!(matches!(&e, CheckError::FixedPattern { column, .. } if column == "K[2]"), "{e}");
    let e = tampered(&air, |a| a.set("RANGE", 3, k(4)));
    assert!(matches!(&e, CheckError::FixedPattern { column, .. } if column == "RANGE"), "{e}");
}

#[test]
fn the_blake3_wrap_checks_its_message_bus_and_iv() {
    let params = vec![k::<Bn128>(4), Bn128::ONE];
    let air = Air::new(128, 13, "blake3_bn128/wrap", &[]).gate("Blake3Compress/parent", 64, params).plonk(0);
    assert_eq!(air.check().unwrap().gates.bands, 1);
    for (col, row) in [("MSG_KEY_X", 64 + 9), ("MSG_MUL", 10), ("FF_IV[0]", 64 + 61)] {
        let e = tampered(&air, |a| a.set(col, row, k(99)));
        assert!(matches!(&e, CheckError::FixedPattern { column, .. } if column == col), "{col}: {e}");
    }
}

#[test]
fn sigma_is_blake3s_schedule() {
    assert_eq!(super::gates::sigma(), proofman_common::exec_format::BLAKE3_SIGMA);
}
