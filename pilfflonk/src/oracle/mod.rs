//! The Rust test oracle (plan M14, R8): a reference evaluation of an AIR's constraints and of its
//! constraint polynomial `Q` (spec A.1), from the pilout alone, with `num-bigint`.
//!
//! It reads the pilout's protobuf directly (`pil2-pilout`) and none of `pil-info`, the setup or
//! the C++ core, so that it stays an independent reference for them: the prover's bytecode (M17,
//! M18), the JS verifier's `qVerifier` (M26) and `pilfflonk check` (M25) are checked against it.
//! It is for tests only: the feature `oracle` is off by default and only this crate's tests turn
//! it on.
//!
//! # Use
//!
//! ```text
//! let oracle = AirOracle::new(&pilout, airgroup_id, air_id)?;
//! let mut values = oracle.values(&witness_source, instance)?;   // stage 1, publics, …
//! values.challenges = …;                                        // what the witness does not give
//! oracle.check(&values)?;                                       // [] for a witness that satisfies the AIR
//! oracle.column_at(&values, ColumnRef::Witness { stage: 1, idx: 0 }, 1, &xi)?;   // l1(ξ·ω)
//! oracle.q_at(&values, &im_pols, &std_vc, &xi)?;                // Q(ξ), as the verifier computes it
//! oracle.q_polynomial(&values, &im_pols, &std_vc)?;             // Q itself, by exact division
//! oracle.fill_hint_columns(&mut values, 2)?;                    // stage 2, from the std's hints
//! ```
//!
//! # What it computes
//!
//! - **Rows.** Row `i` of a trace of `N = 2^nBits` rows is the point `ω^i`, with `ω = ω_N` the
//!   generator of `H` that ffiasm and ffjavascript use ([`omega`]). A column read at offset `s`
//!   on row `i` is its value at row `(i + s) mod N`: rows are cyclic.
//! - **Constraints.** `check` evaluates every constraint of the pilout on every row of its domain
//!   and returns those that are not 0. Domains (A.1): `everyRow` is every row, `firstRow` row 0,
//!   `lastRow` row `N − 1`, and `everyFrame { offsetMin, offsetMax }` every row but the first
//!   `offsetMin` and the last `offsetMax`, as the STARK's zerofier excludes them
//!   (`pil2-stark/src/starkpil/setup_ctx.hpp`, `buildFrameZerofierInv`) and `Boundary::EveryFrame`
//!   says. The compiler of `develop-0.14.0` only emits `everyRow` (spec §3.4); the other three
//!   are exercised with pilouts built in code.
//! - **Points.** A column at offset `s` at a point `z` is its interpolant over `H` at `z·ω^s`, by
//!   barycentric interpolation.
//! - **`Q`.** `Q(X) = Σ_{i<n} std_vc^(n−1−i)·c_i(X)/Z_{D_i}(X)` over the `n` terms of the fold: the
//!   pilout's constraints in its order, then, if the setup chose im pols, one term
//!   `im_k − e_k` for each of them in the order given (the STARK's fold,
//!   `pil-info/src/pil/{constraint_poly,im_polynomials}.rs`). With im pols, every operand that
//!   refers to the expression of an im pol reads its column instead, the interpolant of the
//!   expression's values on `H`; the expression `e_k` of its own term does not.
//!   - `q_at` computes it at a point `z ∉ H`, from the barycentric evaluations and `Z_D(z)` in the
//!     closed forms of A.1, as a verifier does;
//!   - `q_polynomial` computes it as a polynomial: each numerator in coefficient form, divided by
//!     `Z_D = Π_{j ∈ D}(X − ω^j)`. It is exact, and so `Q` a polynomial, if and only if every
//!     constraint holds on every row of its domain; `QPolynomial` keeps the remainders.
//!
//! - **The std's prover hints** (plan M30). The columns of stage 2 and above are not the witness's:
//!   the std's `gsum_col` and `gprod_col` hints give them, from the pilout's hints and nothing else.
//!   Each hint's `reference` column is, row after row, the running sum (`gsum_col`) or product
//!   (`gprod_col`) of `numerator_air/denominator_air`, each row's quotient with its own inversion:
//!   a naive sequential reference for the prover's, which inverts in a batch (`hint_columns`).
//!   `result` and the direct fields update an airgroup value, which v1 has none of (D2): the
//!   column does not depend on them. The other prover hints (`im_col`, `im_airval`) it does not
//!   compute.
//!
//! The im pols are given as the indices of their expressions in the pilout (`expId` of the
//! `imPol` entries of `cmPolsMap`, in their order there), which assumes that the setup numbers the
//! pilout's expressions as the pilout does.
//!
//! **Indices** are the pilout's: witness column `idx` of stage `s` is
//! `Values::witness[s − 1][idx]`, challenge `idx` of stage `s` is `Values::challenges[s − 1][idx]`,
//! and proof values, air values and airgroup values are indexed as their operands index them,
//! which is also the order of their symbols (proof values) or of `airValues` and `airGroupValues`.
//!
//! **Cost.** Quadratic in `N`, with big integers, recursive over the expressions: for fixtures of a
//! few hundred rows (the Fibonacci's `q_polynomial`, `N = 256`, takes seconds in a debug build).

mod eval;
mod fr;
mod poly;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, constraint, SymbolType};

use crate::error::{invalid, PilfflonkError, PilfflonkResult};
use crate::field::r;
use crate::global_info::MAX_NBITS;
use crate::witness::{AirInstanceRef, AirShape, WitnessShape, WitnessSource};

use eval::{Algebra, Coefficients, Columns, Evaluator, Given, Point, Rows};
pub use fr::{omega, Fr};

/// A column of an AIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ColumnRef {
    Fixed(usize),
    Periodic(usize),
    Witness {
        stage: usize,
        idx: usize,
    },
    /// The im pol of expression `idx`: the expression's values on `H`.
    Im(usize),
}

/// The domain of a constraint: the rows it holds on (see the module).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    EveryRow,
    FirstRow,
    LastRow,
    EveryFrame { offset_min: usize, offset_max: usize },
}

impl Domain {
    /// Its rows, of a trace of `n` rows.
    pub fn rows(&self, n: usize) -> Range<usize> {
        match *self {
            Domain::EveryRow => 0..n,
            Domain::FirstRow => 0..1,
            Domain::LastRow => n.saturating_sub(1)..n,
            Domain::EveryFrame { offset_min, offset_max } => offset_min..n.saturating_sub(offset_max),
        }
    }

    /// `Z_D(z)` as A.1 writes it, for a trace of `2^n_bits` rows: `z^N − 1`, `z − 1`,
    /// `z − ω^(N−1)`, and `(z^N − 1)/Π_j (z − ω^j)` over the rows `j` an `everyFrame` excludes.
    pub fn zerofier_at(&self, z: &Fr, n_bits: u32) -> PilfflonkResult<Fr> {
        let w = omega(n_bits)?;
        let n = 1u64 << n_bits;
        let z_h = &z.pow_u64(n) - &Fr::one();
        Ok(match *self {
            Domain::EveryRow => z_h,
            Domain::FirstRow => z - &Fr::one(),
            Domain::LastRow => z - &w.pow_u64(n - 1),
            Domain::EveryFrame { offset_min, offset_max } => {
                let excluded = (0..offset_min as u64).chain(n.saturating_sub(offset_max as u64)..n);
                let product = excluded.fold(Fr::one(), |acc, j| &acc * &(z - &w.pow_u64(j)));
                &z_h * &product.inv()?
            }
        })
    }

    /// `Z_D` as a polynomial, `Π_{j ∈ D}(X − ω^j)`: built from the rows, not from the closed
    /// forms of `zerofier_at`, so that the two check each other.
    fn zerofier(&self, n_bits: u32) -> PilfflonkResult<Vec<Fr>> {
        let w = omega(n_bits)?;
        let roots: Vec<Fr> = self.rows(1 << n_bits).map(|j| w.pow_u64(j as u64)).collect();
        Ok(poly::from_roots(&roots))
    }
}

/// A constraint of the pilout: its expression must be 0 on the rows of its domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    pub expression: usize,
    pub domain: Domain,
    pub debug_line: String,
}

/// What an AIR's expressions refer to, but its fixed columns: indices as in the module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Values {
    /// `witness[s − 1][idx]`: witness column `idx` of stage `s`, row by row.
    pub witness: Vec<Vec<Vec<Fr>>>,
    pub publics: Vec<Fr>,
    /// `challenges[s − 1][idx]`: challenge `idx` of stage `s`.
    pub challenges: Vec<Vec<Fr>>,
    /// `None` where no value is given.
    pub proof_values: Vec<Option<Fr>>,
    pub air_values: Vec<Option<Fr>>,
    pub airgroup_values: Vec<Option<Fr>>,
}

/// A constraint that does not hold: its index in the pilout, the row, and the value there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    pub constraint: usize,
    pub row: usize,
    pub value: Fr,
}

/// `Q` as `q_polynomial` computes it: the sum of the quotients of each term of the fold by its
/// zerofier, and the terms whose division is not exact, with their remainders. A term is its
/// index in the fold: the pilout's constraints, then the im pols.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QPolynomial {
    /// Lowest degree first, without zero coefficients of highest degree.
    pub coefficients: Vec<Fr>,
    pub remainders: Vec<(usize, Vec<Fr>)>,
}

impl QPolynomial {
    /// Whether every division was exact: `Q` is a polynomial.
    pub fn is_exact(&self) -> bool {
        self.remainders.is_empty()
    }

    pub fn evaluate(&self, z: &Fr) -> Fr {
        poly::evaluate(&self.coefficients, z)
    }
}

/// A term of the fold of `Q`.
#[derive(Clone, Copy, Debug)]
enum Term {
    /// Constraint `i` of the pilout.
    Constraint(usize),
    /// The im pol of expression `idx`: `im − e`.
    Im(usize),
}

/// A `gsum_col` or `gprod_col` hint of the AIR (see the module): its column, `reference`, and the
/// operands of `numerator_air` and `denominator_air`.
#[derive(Clone, Debug, PartialEq)]
pub struct BusHint {
    /// `gprod_col`: a running product; `gsum_col`: a running sum.
    pub prod: bool,
    /// The witness column of the reference: `(stage, idx)`.
    pub stage: usize,
    pub idx: usize,
    pub numerator: Option<pb::Operand>,
    pub denominator: Option<pb::Operand>,
}

/// The oracle of one AIR of a pilout (see the module).
#[derive(Clone, Debug)]
pub struct AirOracle {
    air: AirInstanceRef,
    n_bits: u32,
    n: usize,
    omega: Fr,
    expressions: Vec<pb::Expression>,
    constraints: Vec<Constraint>,
    fixed: Vec<Vec<Fr>>,
    periodic: Vec<Vec<Fr>>,
    stage_widths: Vec<u32>,
    air_value_stages: Vec<u32>,
    n_airgroup_values: usize,
    n_publics: usize,
    proof_value_stages: Vec<u32>,
    num_challenges: Vec<u32>,
    bus_hints: Vec<BusHint>,
}

/// The number of rows of an AIR, `2^nBits`.
fn n_bits_of(air: &pb::Air) -> PilfflonkResult<u32> {
    let rows = air.num_rows.unwrap_or(0);
    if !rows.is_power_of_two() || u64::from(rows.trailing_zeros()) > MAX_NBITS {
        return invalid!("air {:?} has {rows} rows, not a power of two of at most 2^{MAX_NBITS}", air.name);
    }
    Ok(rows.trailing_zeros())
}

/// The stage of each proof value, by index: the proof value symbols, each array spread over
/// consecutive indices from its `id`.
fn proof_value_stages(pilout: &pb::PilOut) -> PilfflonkResult<Vec<u32>> {
    let mut entries = BTreeMap::new();
    for s in pilout.symbols.iter().filter(|s| s.r#type == SymbolType::ProofValue as i32) {
        let count: u32 = s.lengths.iter().product();
        for offset in 0..count {
            if entries.insert(s.id + offset, s.stage.unwrap_or(1)).is_some() {
                return invalid!("two proof value symbols have index {}", s.id + offset);
            }
        }
    }
    if entries.keys().enumerate().any(|(i, id)| *id != i as u32) {
        return invalid!("the proof value symbols do not number the indices 0 … {}", entries.len().saturating_sub(1));
    }
    Ok(entries.into_values().collect())
}

/// The shape of the witness of every AIR of a pilout (see `crate::witness`), independent of the
/// setup: the pilout's stage-1 witness columns, air values and proof values, and its publics.
pub fn witness_shape(pilout: &pb::PilOut) -> PilfflonkResult<WitnessShape> {
    let mut airs = Vec::new();
    for (airgroup_id, airgroup) in pilout.air_groups.iter().enumerate() {
        for (air_id, air) in airgroup.airs.iter().enumerate() {
            airs.push(AirShape {
                airgroup_id: airgroup_id as u64,
                air_id: air_id as u64,
                n_bits: u64::from(n_bits_of(air)?),
                n_cols: air.stage_widths.first().copied().unwrap_or(0) as usize,
                n_air_values: air.air_values.iter().filter(|v| v.stage == 1).count(),
            });
        }
    }
    let n_proof_values = proof_value_stages(pilout)?.iter().filter(|s| **s == 1).count();
    WitnessShape::new(airs, pilout.num_public_values as usize, n_proof_values)
}

/// The `gsum_col` and `gprod_col` hints of air `air_id` of airgroup `airgroup_id`, whose `expressions`
/// they refer to (see the module).
fn bus_hints(
    pilout: &pb::PilOut,
    airgroup_id: usize,
    air_id: usize,
    expressions: &[pb::Expression],
) -> PilfflonkResult<Vec<BusHint>> {
    use pil2_pilout::pilout::{expression, hint_field, operand};
    let mut hints = Vec::new();
    for hint in &pilout.hints {
        let of_air = hint.air_group_id == Some(airgroup_id as u32) && hint.air_id == Some(air_id as u32);
        let prod = match hint.name.as_str() {
            "gsum_col" => false,
            "gprod_col" => true,
            _ => continue,
        };
        if !of_air {
            continue;
        }
        // Its fields are those of the array of its first field (the compiler's layout of a hint).
        let fields = match hint.hint_fields.first().and_then(|f| f.value.as_ref()) {
            Some(hint_field::Value::HintFieldArray(array)) => &array.hint_fields,
            _ => return invalid!("hint {} has no array of fields", hint.name),
        };
        let field = |name: &str| -> PilfflonkResult<Option<pb::Operand>> {
            match fields.iter().find(|f| f.name.as_deref() == Some(name)).and_then(|f| f.value.as_ref()) {
                Some(hint_field::Value::Operand(o)) => Ok(Some(o.clone())),
                _ => invalid!("hint {} has no operand {name}", hint.name),
            }
        };
        // The compiler writes the column of `reference` as the expression `column + 0`.
        let reference = field("reference")?;
        let witness_col = |o: &Option<pb::Operand>| match o.as_ref().and_then(|o| o.operand.as_ref()) {
            Some(operand::Operand::WitnessCol(c)) => Some(*c),
            _ => None,
        };
        let is_zero = |o: &Option<pb::Operand>| match o.as_ref().and_then(|o| o.operand.as_ref()) {
            Some(operand::Operand::Constant(c)) => c.value.iter().all(|&b| b == 0),
            _ => false,
        };
        let column = match reference.as_ref().and_then(|o| o.operand.as_ref()) {
            Some(operand::Operand::Expression(e)) => {
                match expressions.get(e.idx as usize).and_then(|x| x.operation.as_ref()) {
                    Some(expression::Operation::Add(add)) if is_zero(&add.rhs) => witness_col(&add.lhs),
                    _ => None,
                }
            }
            _ => witness_col(&reference),
        };
        let Some(column) = column.filter(|c| c.row_offset == 0 && c.stage >= 2) else {
            return invalid!("the reference of hint {} is not a column of stage 2 or above at its own row", hint.name);
        };
        hints.push(BusHint {
            prod,
            stage: column.stage as usize,
            idx: column.col_idx as usize,
            numerator: field("numerator_air")?,
            denominator: field("denominator_air")?,
        });
    }
    Ok(hints)
}

fn read_column(values: &[Vec<u8>], what: &str) -> PilfflonkResult<Vec<Fr>> {
    values
        .iter()
        .map(|v| Fr::from_pilout_bytes(v))
        .collect::<PilfflonkResult<_>>()
        .map_err(|e| PilfflonkError::InvalidFormat(format!("{what}: {e}")))
}

impl AirOracle {
    /// The oracle of air `air_id` of airgroup `airgroup_id`. Refuses a pilout that is not over
    /// BN254, fixed columns without their values (a pilout compiled with fixed columns to a
    /// file), and constraints without an expression.
    pub fn new(pilout: &pb::PilOut, airgroup_id: usize, air_id: usize) -> PilfflonkResult<Self> {
        if BigUint::from_bytes_be(&pilout.base_field) != *r() {
            return invalid!("the pilout is not over BN254: its base field is not r");
        }
        let Some(airgroup) = pilout.air_groups.get(airgroup_id) else {
            return invalid!("the pilout has no airgroup {airgroup_id}");
        };
        let Some(air) = airgroup.airs.get(air_id) else {
            return invalid!("airgroup {airgroup_id} has no air {air_id}");
        };
        let n_bits = n_bits_of(air)?;
        let n = 1usize << n_bits;

        let mut fixed = Vec::with_capacity(air.fixed_cols.len());
        for (i, col) in air.fixed_cols.iter().enumerate() {
            if col.values.len() != n {
                return invalid!(
                    "fixed column {i} has {} values, not one per row ({n}): was the pilout compiled with its fixed \
                     columns to a file?",
                    col.values.len()
                );
            }
            fixed.push(read_column(&col.values, &format!("fixed column {i}"))?);
        }
        let mut periodic = Vec::with_capacity(air.periodic_cols.len());
        for (i, col) in air.periodic_cols.iter().enumerate() {
            let cycle = read_column(&col.values, &format!("periodic column {i}"))?;
            if cycle.is_empty() || !n.is_multiple_of(cycle.len()) {
                return invalid!("periodic column {i} has a cycle of {} rows, which does not divide {n}", cycle.len());
            }
            periodic.push((0..n).map(|row| cycle[row % cycle.len()].clone()).collect());
        }

        let mut constraints = Vec::with_capacity(air.constraints.len());
        for (i, c) in air.constraints.iter().enumerate() {
            use constraint::Constraint as C;
            let (expression, domain, debug_line) = match &c.constraint {
                Some(C::EveryRow(c)) => (&c.expression_idx, Domain::EveryRow, &c.debug_line),
                Some(C::FirstRow(c)) => (&c.expression_idx, Domain::FirstRow, &c.debug_line),
                Some(C::LastRow(c)) => (&c.expression_idx, Domain::LastRow, &c.debug_line),
                Some(C::EveryFrame(c)) => {
                    let (offset_min, offset_max) = (c.offset_min as usize, c.offset_max as usize);
                    if offset_min + offset_max >= n {
                        return invalid!(
                            "constraint {i} is everyFrame {{{offset_min}, {offset_max}}}: no row of {n} is left"
                        );
                    }
                    (&c.expression_idx, Domain::EveryFrame { offset_min, offset_max }, &c.debug_line)
                }
                None => return invalid!("constraint {i} is empty"),
            };
            let Some(expression) = expression.as_ref().map(|e| e.idx as usize).filter(|e| *e < air.expressions.len())
            else {
                return invalid!("constraint {i} has no expression of the AIR");
            };
            constraints.push(Constraint { expression, domain, debug_line: debug_line.clone().unwrap_or_default() });
        }

        Ok(Self {
            air: AirInstanceRef { airgroup_id: airgroup_id as u64, air_id: air_id as u64 },
            n_bits,
            n,
            omega: omega(n_bits)?,
            expressions: air.expressions.clone(),
            constraints,
            fixed,
            periodic,
            stage_widths: air.stage_widths.clone(),
            air_value_stages: air.air_values.iter().map(|v| v.stage).collect(),
            n_airgroup_values: airgroup.air_group_values.len(),
            n_publics: pilout.num_public_values as usize,
            proof_value_stages: proof_value_stages(pilout)?,
            num_challenges: pilout.num_challenges.clone(),
            bus_hints: bus_hints(pilout, airgroup_id, air_id, &air.expressions)?,
        })
    }

    /// The `gsum_col` and `gprod_col` hints of the AIR, in the pilout's order.
    pub fn bus_hints(&self) -> &[BusHint] {
        &self.bus_hints
    }

    /// The columns of stage `stage` the std's hints give (see the module), by their `idx`: for each
    /// hint of that stage, row after row, `acc_i = acc_{i−1} ∘ numerator_i/denominator_i` from the
    /// identity (0 for a sum, 1 for a product), each quotient with an inversion of its own. They read
    /// `values`, but not its columns of stage `stage` or after, which may be empty. A denominator that
    /// is 0 on a row is an error.
    pub fn hint_columns(&self, values: &Values, stage: usize) -> PilfflonkResult<BTreeMap<usize, Vec<Fr>>> {
        let none = BTreeMap::new();
        let mut rows = self.rows(values, &none);
        let mut columns = BTreeMap::new();
        for hint in self.bus_hints.iter().filter(|h| h.stage == stage) {
            let numerator = rows.operand(&hint.numerator)?;
            let denominator = rows.operand(&hint.denominator)?;
            let mut acc = if hint.prod { Fr::one() } else { Fr::zero() };
            let mut column = Vec::with_capacity(self.n);
            for (row, (num, den)) in numerator.iter().zip(&denominator).enumerate() {
                let Ok(inverse) = den.inv() else {
                    return invalid!(
                        "the denominator of the hint of column {} of stage {stage} is 0 at row {row}",
                        hint.idx
                    );
                };
                let term = num * &inverse;
                acc = if hint.prod { &acc * &term } else { &acc + &term };
                column.push(acc.clone());
            }
            if columns.insert(hint.idx, column).is_some() {
                return invalid!("two hints give column {} of stage {stage}", hint.idx);
            }
        }
        Ok(columns)
    }

    /// Sets the columns of stage `stage` of `values` to those the std's hints give
    /// ([`hint_columns`](Self::hint_columns)): every column of the stage must be one.
    pub fn fill_hint_columns(&self, values: &mut Values, stage: usize) -> PilfflonkResult<()> {
        let width = stage.checked_sub(1).and_then(|s| self.stage_widths.get(s)).copied().unwrap_or(0) as usize;
        let mut columns = self.hint_columns(values, stage)?;
        let mut filled = Vec::with_capacity(width);
        for idx in 0..width {
            match columns.remove(&idx) {
                Some(column) => filled.push(column),
                None => return invalid!("no gsum_col or gprod_col hint gives column {idx} of stage {stage}"),
            }
        }
        if let Some(idx) = columns.keys().next() {
            return invalid!("a hint gives column {idx} of stage {stage}, which has {width}");
        }
        if values.witness.len() < stage {
            values.witness.resize(stage, Vec::new());
        }
        values.witness[stage - 1] = filled;
        Ok(())
    }

    pub fn n_bits(&self) -> u32 {
        self.n_bits
    }

    pub fn n_rows(&self) -> usize {
        self.n
    }

    /// `ω_N`: row `i` is `ω^i`.
    pub fn omega(&self) -> &Fr {
        &self.omega
    }

    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Fixed column `idx`, row by row, as the pilout has it.
    pub fn fixed(&self, idx: usize) -> Option<&[Fr]> {
        self.fixed.get(idx).map(Vec::as_slice)
    }

    /// The values of instance `instance` of a witness source, which must be of this AIR: its
    /// stage-1 columns and air values, the publics and the stage-1 proof values. The rest (the
    /// challenges, stages 2 and later, the airgroup values) is empty, or `None`, for the caller to
    /// fill.
    pub fn values<S: WitnessSource + ?Sized>(&self, source: &S, instance: usize) -> PilfflonkResult<Values> {
        match source.instances().get(instance) {
            Some(air) if *air == self.air => {}
            Some(air) => {
                return invalid!(
                    "instance {instance} is of air {}/{}, and this oracle is of air {}/{}",
                    air.airgroup_id,
                    air.air_id,
                    self.air.airgroup_id,
                    self.air.air_id
                );
            }
            None => return invalid!("the witness has no instance {instance}"),
        }
        let stage1 = source.stage1(instance)?;
        let width = self.stage_widths.first().copied().unwrap_or(0) as usize;
        if (stage1.n_rows(), stage1.n_cols()) != (self.n, width) {
            return invalid!(
                "the witness has {} rows and {} stage-1 columns, and the AIR {} and {width}",
                stage1.n_rows(),
                stage1.n_cols(),
                self.n
            );
        }
        let columns = (0..width)
            .map(|c| stage1.column(c).map(|col| col.iter().map(Fr::from).collect()).unwrap_or_default())
            .collect();
        let mut witness = vec![Vec::new(); self.stage_widths.len().max(1)];
        witness[0] = columns;

        let publics: Vec<Fr> = source.publics()?.iter().map(Fr::from).collect();
        if publics.len() != self.n_publics {
            return invalid!("the witness has {} publics, and the pilout {}", publics.len(), self.n_publics);
        }
        let air_values = stage_1_values(&self.air_value_stages, stage1.air_values().iter().map(Fr::from), "air")?;
        let proof_values =
            stage_1_values(&self.proof_value_stages, source.proof_values()?.iter().map(Fr::from), "proof")?;
        Ok(Values {
            witness,
            publics,
            challenges: self.num_challenges.iter().map(|_| Vec::new()).collect(),
            proof_values,
            air_values,
            airgroup_values: vec![None; self.n_airgroup_values],
        })
    }

    fn columns<'a>(&'a self, values: &'a Values, im: &'a BTreeMap<usize, Vec<Fr>>) -> Columns<'a> {
        Columns { n: self.n, fixed: &self.fixed, periodic: &self.periodic, values, im }
    }

    fn rows<'a>(&'a self, values: &'a Values, im: &'a BTreeMap<usize, Vec<Fr>>) -> Evaluator<'a, Rows<'a>> {
        Evaluator::new(Rows { columns: self.columns(values, im) }, &self.expressions, values, BTreeSet::new())
    }

    /// The values of expression `idx` on every row.
    pub fn expression_rows(&self, values: &Values, idx: usize) -> PilfflonkResult<Vec<Fr>> {
        let none = BTreeMap::new();
        self.rows(values, &none).expression(idx)
    }

    /// The expression of every constraint on every row, in the pilout's order: `[constraint][row]`,
    /// on all the rows, those out of its domain too.
    pub fn numerators(&self, values: &Values) -> PilfflonkResult<Vec<Vec<Fr>>> {
        let none = BTreeMap::new();
        let mut rows = self.rows(values, &none);
        self.constraints.iter().map(|c| rows.expression(c.expression)).collect()
    }

    /// The constraints that do not hold, on the rows of their domains where they do not, by
    /// constraint and then by row: empty if and only if the witness satisfies the AIR.
    pub fn check(&self, values: &Values) -> PilfflonkResult<Vec<Failure>> {
        let mut failures = Vec::new();
        for (constraint, (c, rows)) in self.constraints.iter().zip(self.numerators(values)?).enumerate() {
            for row in c.domain.rows(self.n) {
                if !rows[row].is_zero() {
                    failures.push(Failure { constraint, row, value: rows[row].clone() });
                }
            }
        }
        Ok(failures)
    }

    /// The values on `H` of the expressions of the im pols.
    fn im_rows(&self, values: &Values, im_pols: &[usize]) -> PilfflonkResult<BTreeMap<usize, Vec<Fr>>> {
        let mut seen = BTreeSet::new();
        let none = BTreeMap::new();
        let mut rows = self.rows(values, &none);
        let mut out = BTreeMap::new();
        for &idx in im_pols {
            if !seen.insert(idx) || idx >= self.expressions.len() {
                return invalid!("im pol {idx} is twice in the list, or not an expression of the AIR");
            }
            out.insert(idx, rows.expression(idx)?);
        }
        Ok(out)
    }

    /// Column `col` at `z·ω^offset`, by barycentric interpolation over `H`.
    pub fn column_at(&self, values: &Values, col: ColumnRef, offset: i32, z: &Fr) -> PilfflonkResult<Fr> {
        let im = match col {
            ColumnRef::Im(idx) => self.im_rows(values, &[idx])?,
            _ => BTreeMap::new(),
        };
        let mut point = Point {
            columns: self.columns(values, &im),
            omega: self.omega.clone(),
            z: z.clone(),
            weights: BTreeMap::new(),
        };
        point.column(col, offset)
    }

    /// The terms of the fold (see the module).
    fn terms(&self, im_pols: &[usize]) -> Vec<(Term, Domain)> {
        let constraints = self.constraints.iter().enumerate().map(|(i, c)| (Term::Constraint(i), c.domain));
        constraints.chain(im_pols.iter().map(|&idx| (Term::Im(idx), Domain::EveryRow))).collect()
    }

    fn numerator<A: Algebra>(&self, ev: &mut Evaluator<'_, A>, term: Term) -> PilfflonkResult<A::V> {
        match term {
            Term::Constraint(i) => ev.expression(self.constraints[i].expression),
            Term::Im(idx) => {
                let im = ev.algebra.column(ColumnRef::Im(idx), 0)?;
                let e = ev.body(idx)?;
                Ok(ev.algebra.sub(&im, &e))
            }
        }
    }

    /// `Q(z)` for `z ∉ H` (see the module): the fold of the numerators at `z`, each divided by
    /// `Z_D(z)`.
    pub fn q_at(&self, values: &Values, im_pols: &[usize], std_vc: &Fr, z: &Fr) -> PilfflonkResult<Fr> {
        if z.pow_u64(self.n as u64) == Fr::one() {
            return invalid!("Q is not defined on H, and {z} is in H");
        }
        let im = self.im_rows(values, im_pols)?;
        let point = Point {
            columns: self.columns(values, &im),
            omega: self.omega.clone(),
            z: z.clone(),
            weights: BTreeMap::new(),
        };
        let mut ev = Evaluator::new(point, &self.expressions, values, im.keys().copied().collect());
        let mut acc = Fr::zero();
        for (term, domain) in self.terms(im_pols) {
            let numerator = self.numerator(&mut ev, term)?;
            let quotient = &numerator * &domain.zerofier_at(z, self.n_bits)?.inv()?;
            acc = &(&acc * std_vc) + &quotient;
        }
        Ok(acc)
    }

    /// `Q(z)` for `z ∉ H` from the columns' values at `z·ω^s` (`values[(column, s)]`, the im pols
    /// as [`ColumnRef::Im`]), as a verifier computes it from the evaluations of a proof: the fold of
    /// `q_at`, with each column read from `values` instead of interpolated over `H`. The scalars
    /// (publics, challenges, …) are those of `scalars`; its columns are not read.
    ///
    /// With a proof's evaluations, which are those of blinded polynomials (A.3), it is the `Q(ξ)`
    /// the verifier derives and the prover's `Q` must take at `ξ`, and not `q_at`'s.
    pub fn q_from_evaluations(
        &self,
        scalars: &Values,
        values: &BTreeMap<(ColumnRef, i32), Fr>,
        im_pols: &[usize],
        std_vc: &Fr,
        z: &Fr,
    ) -> PilfflonkResult<Fr> {
        if z.pow_u64(self.n as u64) == Fr::one() {
            return invalid!("Q is not defined on H, and {z} is in H");
        }
        let mut ev = Evaluator::new(Given { values }, &self.expressions, scalars, im_pols.iter().copied().collect());
        let mut acc = Fr::zero();
        for (term, domain) in self.terms(im_pols) {
            let numerator = self.numerator(&mut ev, term)?;
            let quotient = &numerator * &domain.zerofier_at(z, self.n_bits)?.inv()?;
            acc = &(&acc * std_vc) + &quotient;
        }
        Ok(acc)
    }

    /// `Q` as a polynomial (see the module).
    pub fn q_polynomial(&self, values: &Values, im_pols: &[usize], std_vc: &Fr) -> PilfflonkResult<QPolynomial> {
        let im = self.im_rows(values, im_pols)?;
        let coefficients = Coefficients {
            columns: self.columns(values, &im),
            omega: self.omega.clone(),
            interpolants: BTreeMap::new(),
        };
        let mut ev = Evaluator::new(coefficients, &self.expressions, values, im.keys().copied().collect());
        let mut zerofiers: Vec<(Domain, Vec<Fr>)> = Vec::new();
        let mut acc = Vec::new();
        let mut remainders = Vec::new();
        for (i, (term, domain)) in self.terms(im_pols).into_iter().enumerate() {
            let numerator = self.numerator(&mut ev, term)?;
            if !zerofiers.iter().any(|(d, _)| *d == domain) {
                zerofiers.push((domain, domain.zerofier(self.n_bits)?));
            }
            let zerofier = zerofiers.iter().find(|(d, _)| *d == domain).map(|(_, z)| z.as_slice()).unwrap_or_default();
            let (quotient, remainder) = poly::div_rem(&numerator, zerofier)?;
            if !remainder.is_empty() {
                remainders.push((i, remainder));
            }
            acc = poly::add(&poly::scale(&acc, std_vc), &quotient);
        }
        Ok(QPolynomial { coefficients: acc, remainders })
    }
}

/// Values indexed as the pilout indexes them, the stage-1 ones taken from `given` in order and the
/// others `None`.
fn stage_1_values(stages: &[u32], given: impl Iterator<Item = Fr>, what: &str) -> PilfflonkResult<Vec<Option<Fr>>> {
    let given: Vec<Fr> = given.collect();
    let n_stage_1 = stages.iter().filter(|s| **s == 1).count();
    if given.len() != n_stage_1 {
        return invalid!("the witness has {} {what} values, and the pilout {n_stage_1} of stage 1", given.len());
    }
    let mut given = given.into_iter();
    Ok(stages.iter().map(|s| if *s == 1 { given.next() } else { None }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_closed_forms_of_the_zerofiers_are_the_products_over_their_rows() {
        let z = Fr::from_u64(987654321);
        let domains = [
            Domain::EveryRow,
            Domain::FirstRow,
            Domain::LastRow,
            Domain::EveryFrame { offset_min: 0, offset_max: 0 },
            Domain::EveryFrame { offset_min: 2, offset_max: 1 },
            Domain::EveryFrame { offset_min: 0, offset_max: 3 },
        ];
        for domain in domains {
            let product = poly::evaluate(&domain.zerofier(3).unwrap(), &z);
            assert_eq!(domain.zerofier_at(&z, 3).unwrap(), product, "{domain:?}");
        }
        assert_eq!(Domain::EveryFrame { offset_min: 2, offset_max: 1 }.rows(8), 2..7);
        assert_eq!(Domain::LastRow.rows(8), 7..8);
        let w = omega(3).unwrap();
        assert!(Domain::FirstRow.zerofier_at(&Fr::one(), 3).unwrap().is_zero());
        assert!(Domain::LastRow.zerofier_at(&w.pow_u64(7), 3).unwrap().is_zero());
        assert!(!Domain::LastRow.zerofier_at(&w.pow_u64(6), 3).unwrap().is_zero());
    }
}
