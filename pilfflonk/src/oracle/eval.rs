//! The evaluation of a pilout's expressions, once for the three ways the oracle needs them: on
//! every row of `H`, at one point, and as polynomials. Only the columns change between the three
//! ([`Algebra`]); the scalars and the operations are the same.

use std::collections::{BTreeMap, BTreeSet};

use pil2_pilout::pilout::{self as pb, expression, operand};

use crate::error::{invalid, PilfflonkResult};

use super::fr::Fr;
use super::poly::{self, Barycentric};
use super::{ColumnRef, Values};

/// The columns of an instance, each as its `N` values over `H`.
pub(super) struct Columns<'a> {
    pub n: usize,
    pub fixed: &'a [Vec<Fr>],
    /// Expanded to `N` rows.
    pub periodic: &'a [Vec<Fr>],
    pub values: &'a Values,
    /// The im pols, by the index of their expression: its values on `H`.
    pub im: &'a BTreeMap<usize, Vec<Fr>>,
}

impl Columns<'_> {
    pub fn rows(&self, col: ColumnRef) -> PilfflonkResult<&[Fr]> {
        let column = match col {
            ColumnRef::Fixed(idx) => self.fixed.get(idx),
            ColumnRef::Periodic(idx) => self.periodic.get(idx),
            ColumnRef::Witness { stage, idx } => {
                stage.checked_sub(1).and_then(|s| self.values.witness.get(s)).and_then(|cols| cols.get(idx))
            }
            ColumnRef::Im(idx) => self.im.get(&idx),
        };
        match column {
            Some(values) if values.len() == self.n => Ok(values),
            Some(values) => invalid!("column {col:?} has {} values, not one per row ({})", values.len(), self.n),
            None => invalid!("there are no values for column {col:?}"),
        }
    }
}

/// What the columns become: row values, values at a point, or polynomials.
pub(super) trait Algebra {
    type V: Clone;

    fn constant(&self, c: &Fr) -> Self::V;

    /// Column `col` read at row offset `offset`.
    fn column(&mut self, col: ColumnRef, offset: i32) -> PilfflonkResult<Self::V>;

    fn add(&self, a: &Self::V, b: &Self::V) -> Self::V;

    fn sub(&self, a: &Self::V, b: &Self::V) -> Self::V;

    fn mul(&self, a: &Self::V, b: &Self::V) -> Self::V;

    fn neg(&self, a: &Self::V) -> Self::V;
}

/// Every row of `H`: `v[i]` is the value at row `i`, and a column at offset `s` is read at row
/// `(i + s) mod N`, cyclically.
pub(super) struct Rows<'a> {
    pub columns: Columns<'a>,
}

impl Algebra for Rows<'_> {
    type V = Vec<Fr>;

    fn constant(&self, c: &Fr) -> Vec<Fr> {
        vec![c.clone(); self.columns.n]
    }

    fn column(&mut self, col: ColumnRef, offset: i32) -> PilfflonkResult<Vec<Fr>> {
        let values = self.columns.rows(col)?;
        let n = values.len() as i64;
        Ok((0..n).map(|row| values[(row + i64::from(offset)).rem_euclid(n) as usize].clone()).collect())
    }

    fn add(&self, a: &Vec<Fr>, b: &Vec<Fr>) -> Vec<Fr> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    fn sub(&self, a: &Vec<Fr>, b: &Vec<Fr>) -> Vec<Fr> {
        a.iter().zip(b).map(|(x, y)| x - y).collect()
    }

    fn mul(&self, a: &Vec<Fr>, b: &Vec<Fr>) -> Vec<Fr> {
        a.iter().zip(b).map(|(x, y)| x * y).collect()
    }

    fn neg(&self, a: &Vec<Fr>) -> Vec<Fr> {
        a.iter().map(|x| -x).collect()
    }
}

/// One point `z`: a column at offset `s` is its polynomial at `z·ω^s`, by barycentric
/// interpolation over `H`.
pub(super) struct Point<'a> {
    pub columns: Columns<'a>,
    pub omega: Fr,
    pub z: Fr,
    pub weights: BTreeMap<i32, Barycentric>,
}

impl Algebra for Point<'_> {
    type V = Fr;

    fn constant(&self, c: &Fr) -> Fr {
        c.clone()
    }

    fn column(&mut self, col: ColumnRef, offset: i32) -> PilfflonkResult<Fr> {
        let n = self.columns.n;
        if !self.weights.contains_key(&offset) {
            let shift = self.omega.pow_u64(i64::from(offset).rem_euclid(n as i64) as u64);
            self.weights.insert(offset, Barycentric::new(&(&self.z * &shift), &self.omega, n)?);
        }
        let values = self.columns.rows(col)?;
        match self.weights.get(&offset) {
            Some(weights) => weights.evaluate(values),
            None => invalid!("no barycentric weights at offset {offset}"),
        }
    }

    fn add(&self, a: &Fr, b: &Fr) -> Fr {
        a + b
    }

    fn sub(&self, a: &Fr, b: &Fr) -> Fr {
        a - b
    }

    fn mul(&self, a: &Fr, b: &Fr) -> Fr {
        a * b
    }

    fn neg(&self, a: &Fr) -> Fr {
        -a
    }
}

/// One point `z`, the columns given at it: a column at offset `s` is the value `values` has for it,
/// as a verifier reads the evaluations of a proof.
pub(super) struct Given<'a> {
    pub values: &'a BTreeMap<(ColumnRef, i32), Fr>,
}

impl Algebra for Given<'_> {
    type V = Fr;

    fn constant(&self, c: &Fr) -> Fr {
        c.clone()
    }

    fn column(&mut self, col: ColumnRef, offset: i32) -> PilfflonkResult<Fr> {
        match self.values.get(&(col, offset)) {
            Some(v) => Ok(v.clone()),
            None => invalid!("no value is given for column {col:?} at offset {offset}"),
        }
    }

    fn add(&self, a: &Fr, b: &Fr) -> Fr {
        a + b
    }

    fn sub(&self, a: &Fr, b: &Fr) -> Fr {
        a - b
    }

    fn mul(&self, a: &Fr, b: &Fr) -> Fr {
        a * b
    }

    fn neg(&self, a: &Fr) -> Fr {
        -a
    }
}

/// Polynomials in coefficient form: a column at offset `s` is `p(ω^s·X)`, `p` its interpolant
/// over `H`.
pub(super) struct Coefficients<'a> {
    pub columns: Columns<'a>,
    pub omega: Fr,
    pub interpolants: BTreeMap<ColumnRef, Vec<Fr>>,
}

impl Algebra for Coefficients<'_> {
    type V = Vec<Fr>;

    fn constant(&self, c: &Fr) -> Vec<Fr> {
        poly::trim(vec![c.clone()])
    }

    fn column(&mut self, col: ColumnRef, offset: i32) -> PilfflonkResult<Vec<Fr>> {
        if !self.interpolants.contains_key(&col) {
            let p = poly::interpolate(self.columns.rows(col)?, &self.omega)?;
            self.interpolants.insert(col, p);
        }
        let n = self.columns.n as i64;
        let shift = self.omega.pow_u64(i64::from(offset).rem_euclid(n) as u64);
        match self.interpolants.get(&col) {
            Some(p) => Ok(poly::compose_scaled(p, &shift)),
            None => invalid!("no interpolant of column {col:?}"),
        }
    }

    fn add(&self, a: &Vec<Fr>, b: &Vec<Fr>) -> Vec<Fr> {
        poly::add(a, b)
    }

    fn sub(&self, a: &Vec<Fr>, b: &Vec<Fr>) -> Vec<Fr> {
        poly::sub(a, b)
    }

    fn mul(&self, a: &Vec<Fr>, b: &Vec<Fr>) -> Vec<Fr> {
        poly::mul(a, b)
    }

    fn neg(&self, a: &Vec<Fr>) -> Vec<Fr> {
        poly::neg(a)
    }
}

/// The expressions of an AIR over an algebra, each computed once.
pub(super) struct Evaluator<'a, A: Algebra> {
    pub algebra: A,
    expressions: &'a [pb::Expression],
    values: &'a Values,
    /// The expressions read as their im pol's column wherever an operand refers to them.
    substituted: BTreeSet<usize>,
    memo: Vec<Option<A::V>>,
    visiting: Vec<bool>,
}

impl<'a, A: Algebra> Evaluator<'a, A> {
    pub fn new(
        algebra: A,
        expressions: &'a [pb::Expression],
        values: &'a Values,
        substituted: BTreeSet<usize>,
    ) -> Self {
        let n = expressions.len();
        Self { algebra, expressions, values, substituted, memo: vec![None; n], visiting: vec![false; n] }
    }

    /// Expression `idx` as an operand refers to it: its im pol's column if it is one.
    pub fn expression(&mut self, idx: usize) -> PilfflonkResult<A::V> {
        if self.substituted.contains(&idx) {
            return self.algebra.column(ColumnRef::Im(idx), 0);
        }
        if let Some(Some(v)) = self.memo.get(idx) {
            return Ok(v.clone());
        }
        let v = self.body(idx)?;
        self.memo[idx] = Some(v.clone());
        Ok(v)
    }

    /// The operation of expression `idx` itself, even if it is an im pol.
    pub fn body(&mut self, idx: usize) -> PilfflonkResult<A::V> {
        let Some(e) = self.expressions.get(idx) else {
            return invalid!("expression {idx} is not in the AIR, which has {}", self.expressions.len());
        };
        if self.visiting[idx] {
            return invalid!("expression {idx} refers to itself");
        }
        self.visiting[idx] = true;
        let v = self.operation(idx, e);
        self.visiting[idx] = false;
        v
    }

    fn operation(&mut self, idx: usize, e: &pb::Expression) -> PilfflonkResult<A::V> {
        use expression::Operation;
        match &e.operation {
            Some(Operation::Add(op)) => {
                let (a, b) = (self.operand(&op.lhs)?, self.operand(&op.rhs)?);
                Ok(self.algebra.add(&a, &b))
            }
            Some(Operation::Sub(op)) => {
                let (a, b) = (self.operand(&op.lhs)?, self.operand(&op.rhs)?);
                Ok(self.algebra.sub(&a, &b))
            }
            Some(Operation::Mul(op)) => {
                let (a, b) = (self.operand(&op.lhs)?, self.operand(&op.rhs)?);
                Ok(self.algebra.mul(&a, &b))
            }
            Some(Operation::Neg(op)) => {
                let a = self.operand(&op.value)?;
                Ok(self.algebra.neg(&a))
            }
            None => invalid!("expression {idx} has no operation"),
        }
    }

    fn scalar(&self, value: Option<&Option<Fr>>, what: String) -> PilfflonkResult<A::V> {
        match value {
            Some(Some(v)) => Ok(self.algebra.constant(v)),
            _ => invalid!("there is no value for {what}"),
        }
    }

    fn operand(&mut self, operand: &Option<pb::Operand>) -> PilfflonkResult<A::V> {
        use operand::Operand;
        let values = self.values;
        let at = |list: &[Fr], i: u32| list.get(i as usize).cloned();
        match operand.as_ref().and_then(|o| o.operand.as_ref()) {
            Some(Operand::Constant(c)) => Ok(self.algebra.constant(&Fr::from_pilout_bytes(&c.value)?)),
            Some(Operand::Challenge(c)) => {
                let value =
                    (c.stage as usize).checked_sub(1).and_then(|s| values.challenges.get(s)).and_then(|l| at(l, c.idx));
                self.scalar(Some(&value), format!("challenge {} of stage {}", c.idx, c.stage))
            }
            Some(Operand::ProofValue(p)) => self
                .scalar(values.proof_values.get(p.idx as usize), format!("proof value {} (stage {})", p.idx, p.stage)),
            Some(Operand::AirGroupValue(a)) => {
                self.scalar(values.airgroup_values.get(a.idx as usize), format!("airgroup value {}", a.idx))
            }
            Some(Operand::AirValue(a)) => {
                self.scalar(values.air_values.get(a.idx as usize), format!("air value {}", a.idx))
            }
            Some(Operand::PublicValue(p)) => {
                let value = at(&values.publics, p.idx);
                self.scalar(Some(&value), format!("public {}", p.idx))
            }
            Some(Operand::FixedCol(c)) => self.algebra.column(ColumnRef::Fixed(c.idx as usize), c.row_offset),
            Some(Operand::PeriodicCol(c)) => self.algebra.column(ColumnRef::Periodic(c.idx as usize), c.row_offset),
            Some(Operand::WitnessCol(c)) => self
                .algebra
                .column(ColumnRef::Witness { stage: c.stage as usize, idx: c.col_idx as usize }, c.row_offset),
            Some(Operand::Expression(e)) => self.expression(e.idx as usize),
            Some(Operand::CustomCol(_)) => invalid!("the AIR has a custom commit, which pilfflonk refuses (P5)"),
            None => invalid!("an operand is empty"),
        }
    }
}
