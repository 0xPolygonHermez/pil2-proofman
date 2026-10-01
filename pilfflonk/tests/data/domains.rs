//! Synthetic pilouts for the constraint domains (pilfflonk/docs/README.md#fixtures), built in code
//! with `prost`: the compiler of `develop-0.14.0` emits only `everyRow`, so `firstRow`, `lastRow`
//! and `everyFrame` are reachable no other way. And their witness generators, over BN254's `Fr`
//! with `num-bigint`.
//!
//! Each AIR has `N = 2^4` rows, a fixed column `K = [1, 2, …, N]`, one witness column `x<j>` per
//! rule `j` of it (and a public `p<j>` for a `firstRow` or `lastRow` rule), and a witness column `y`
//! after them. Its constraints are the rules', in their order, and then `y − x0·K` on every row:
//!
//! | Rule | Domain | Constraint | Offsets of `x<j>` |
//! |---|---|---|---|
//! | `FirstRow` | `firstRow` | `x·x − p` | 0 |
//! | `LastRow` | `lastRow` | `x·x − p` | 0 |
//! | `Next { min, max }` | `everyFrame {min, max}` | `x' − x·x − K` | 0, 1 |
//! | `Prev { min, max }` | `everyFrame {min, max}` | `x − 'x·'x − K` | −1, 0 |
//!
//! An `everyFrame {offsetMin, offsetMax}` holds on the rows `offsetMin ≤ i < N − offsetMax`: it
//! excludes the first `offsetMin` rows and the last `offsetMax`, as the STARK's zerofier
//! (`buildFrameZerofierInv`), the zerofiers of the constraint polynomial
//! (pilfflonk/docs/protocol.md#constraint-polynomial) and the oracle say. The pilout's comment on
//! the field ("frame size is defined as offsetMax − offsetMin + 1") describes something else;
//! nothing here follows it.
//!
//! The constraints of the rules have degree 2, so with the `δ = 1` of their domains
//! (pilfflonk/docs/protocol.md#degree-search) the constraint polynomial has degree 3: the search
//! chooses `qDeg = 2` and no im pols by default (an im pol per rule would cost more), and with
//! `--max-constraint-degree 2` one im pol per rule and `qDeg = 1`. On `everyRow` the same
//! constraints would give `qDeg = 1`.
//!
//! The witness satisfies each rule on the rows of its domain and breaks it on every other row, so
//! that a zerofier that vanishes on a row too many or too few makes `Q` no polynomial: `x<j>` is
//! `1000 + 100·j + 7·i + 2` on row `i` where its rule does not set it, and `y = x0·K`.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/domains.rs"] mod domains;`.

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, constraint, expression, operand, SymbolType};
use proofman_pilfflonk::{AirInstanceRef, FrBytes, InstanceWitness, Stage1Witness, Witness, BN254_R};

/// The AIRs' rows: `N = 2^4`.
pub const N_BITS: u32 = 4;
const N: usize = 1 << N_BITS;

/// A constraint of an AIR, on its own column (see the module).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    FirstRow,
    LastRow,
    Next { min: u32, max: u32 },
    Prev { min: u32, max: u32 },
}

impl Rule {
    /// The rows of its domain, `first..last`.
    pub fn rows(self) -> std::ops::Range<usize> {
        match self {
            Rule::FirstRow => 0..1,
            Rule::LastRow => N - 1..N,
            Rule::Next { min, max } | Rule::Prev { min, max } => min as usize..N - max as usize,
        }
    }
}

/// The synthetic AIRs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Air {
    /// Two `firstRow` rules, which share their boundary.
    FirstRow,
    /// Two `lastRow` rules.
    LastRow,
    /// Six `everyFrame` rules of as many `{offsetMin, offsetMax}`, reading the next row or the
    /// previous one, across the wrap too (`{2, 0}` at row N − 1 reads row 0).
    Frames,
    /// One rule of each domain; the two `everyFrame {1, 2}` share their boundary.
    All,
}

impl Air {
    pub fn name(self) -> &'static str {
        match self {
            Air::FirstRow => "FirstRow",
            Air::LastRow => "LastRow",
            Air::Frames => "Frames",
            Air::All => "Domains",
        }
    }

    pub fn rules(self) -> Vec<Rule> {
        match self {
            Air::FirstRow => vec![Rule::FirstRow, Rule::FirstRow],
            Air::LastRow => vec![Rule::LastRow, Rule::LastRow],
            Air::Frames => vec![
                Rule::Next { min: 1, max: 2 },
                Rule::Next { min: 0, max: 3 },
                Rule::Next { min: 2, max: 0 },
                Rule::Next { min: 3, max: 1 },
                Rule::Prev { min: 1, max: 0 },
                Rule::Prev { min: 2, max: 2 },
            ],
            Air::All => {
                vec![Rule::FirstRow, Rule::LastRow, Rule::Next { min: 1, max: 2 }, Rule::Prev { min: 1, max: 2 }]
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The pilout
// ---------------------------------------------------------------------------------------------

fn op(o: operand::Operand) -> Option<pb::Operand> {
    Some(pb::Operand { operand: Some(o) })
}

fn witness_col(col_idx: u32, row_offset: i32) -> Option<pb::Operand> {
    op(operand::Operand::WitnessCol(operand::WitnessCol { stage: 1, col_idx, row_offset }))
}

fn k() -> Option<pb::Operand> {
    op(operand::Operand::FixedCol(operand::FixedCol { idx: 0, row_offset: 0 }))
}

fn public(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::PublicValue(operand::PublicValue { idx }))
}

/// The expressions of the AIR, each operation pushed and referred to by the operand it returns.
#[derive(Default)]
struct Exprs(Vec<pb::Expression>);

impl Exprs {
    fn push(&mut self, operation: expression::Operation) -> Option<pb::Operand> {
        self.0.push(pb::Expression { operation: Some(operation) });
        op(operand::Operand::Expression(operand::Expression { idx: self.0.len() as u32 - 1 }))
    }

    fn sub(&mut self, lhs: Option<pb::Operand>, rhs: Option<pb::Operand>) -> Option<pb::Operand> {
        self.push(expression::Operation::Sub(expression::Sub { lhs, rhs }))
    }

    fn mul(&mut self, lhs: Option<pb::Operand>, rhs: Option<pb::Operand>) -> Option<pb::Operand> {
        self.push(expression::Operation::Mul(expression::Mul { lhs, rhs }))
    }

    fn last(&self) -> Option<operand::Expression> {
        Some(operand::Expression { idx: self.0.len() as u32 - 1 })
    }
}

fn symbol(name: &str, kind: SymbolType, id: u32, stage: Option<u32>, air: bool) -> pb::Symbol {
    pb::Symbol {
        name: name.to_string(),
        air_group_id: air.then_some(0),
        air_id: air.then_some(0),
        r#type: kind as i32,
        id,
        stage,
        dim: 1,
        lengths: vec![],
        commit_id: None,
        debug_line: None,
    }
}

/// The pilout of `air`, over BN254 (see the module).
pub fn pilout(air: Air) -> pb::PilOut {
    use constraint::Constraint as C;
    let rules = air.rules();
    let y = rules.len() as u32;
    let mut e = Exprs::default();
    let mut constraints = Vec::new();
    let mut symbols = Vec::new();
    let mut n_publics = 0;
    for (j, rule) in rules.iter().enumerate() {
        let x = |row_offset| witness_col(j as u32, row_offset);
        let line = |text: &str| Some(format!("{}: {text}", air.name()));
        symbols.push(symbol(&format!("x{j}"), SymbolType::WitnessCol, j as u32, Some(1), true));
        constraints.push(match *rule {
            Rule::FirstRow | Rule::LastRow => {
                let square = e.mul(x(0), x(0));
                e.sub(square, public(n_publics));
                symbols.push(symbol(&format!("p{j}"), SymbolType::PublicValue, n_publics, None, false));
                n_publics += 1;
                let text = format!("x{j}*x{j} - p{j}");
                if *rule == Rule::FirstRow {
                    C::FirstRow(constraint::FirstRow { expression_idx: e.last(), debug_line: line(&text) })
                } else {
                    C::LastRow(constraint::LastRow { expression_idx: e.last(), debug_line: line(&text) })
                }
            }
            Rule::Next { min, max } | Rule::Prev { min, max } => {
                let (at, before) = if matches!(rule, Rule::Next { .. }) { (1, 0) } else { (0, -1) };
                let square = e.mul(x(before), x(before));
                let step = e.sub(x(at), square);
                e.sub(step, k());
                let text = if at == 1 { format!("x{j}' - x{j}*x{j} - K") } else { format!("x{j} - 'x{j}*'x{j} - K") };
                C::EveryFrame(constraint::EveryFrame {
                    expression_idx: e.last(),
                    offset_min: min,
                    offset_max: max,
                    debug_line: line(&text),
                })
            }
        });
    }
    let x0_k = e.mul(witness_col(0, 0), k());
    e.sub(witness_col(y, 0), x0_k);
    let line = Some(format!("{}: y - x0*K", air.name()));
    constraints.push(C::EveryRow(constraint::EveryRow { expression_idx: e.last(), debug_line: line }));
    symbols.push(symbol("y", SymbolType::WitnessCol, y, Some(1), true));
    symbols.push(symbol(&format!("{}.K", air.name()), SymbolType::FixedCol, 0, Some(0), true));

    let air_pb = pb::Air {
        name: Some(air.name().into()),
        num_rows: Some(N as u32),
        fixed_cols: vec![pb::FixedCol { values: (0..N).map(|i| k_at(i).to_bytes_be()).collect() }],
        stage_widths: vec![y + 1],
        expressions: e.0,
        constraints: constraints.into_iter().map(|c| pb::Constraint { constraint: Some(c) }).collect(),
        ..Default::default()
    };
    pb::PilOut {
        name: Some("domains".into()),
        base_field: r().to_bytes_be(),
        air_groups: vec![pb::AirGroup { name: Some(air.name().into()), air_group_values: vec![], airs: vec![air_pb] }],
        num_challenges: vec![0],
        num_proof_values: vec![0],
        num_public_values: n_publics,
        symbols,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------------------------
// The witness
// ---------------------------------------------------------------------------------------------

fn r() -> BigUint {
    BigUint::parse_bytes(BN254_R.as_bytes(), 10).expect("r in decimal")
}

fn k_at(row: usize) -> BigUint {
    BigUint::from(row + 1)
}

/// Where a witness breaks a rule: at the first row of its domain, or at the last.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    First,
    Last,
}

/// The witness of `air`: one instance of air 0 of airgroup 0, with the columns `[x0, …, y]` and
/// the publics `p<j>` in the order of their rules.
pub fn witness(air: Air) -> Witness {
    generate(air, None)
}

/// The witness of `air` with rule `rule` broken at the `edge` row of its domain, and nowhere else;
/// and that row. A `firstRow` or `lastRow` rule is broken by its public, an `everyFrame` rule by the
/// cell that its edge row reads and the rows out of its domain around it only: `x[min]` or
/// `x[N − max]` for `Next`, `x[min − 1]` or `x[N − max − 1]` for `Prev`. `y` follows `x0`.
pub fn broken(air: Air, rule: usize, edge: Edge) -> (Witness, usize) {
    let r = air.rules()[rule].rows();
    let row = if edge == Edge::First { r.start } else { r.end - 1 };
    (generate(air, Some((rule, edge))), row)
}

fn generate(air: Air, broken: Option<(usize, Edge)>) -> Witness {
    let r = r();
    let rules = air.rules();
    let mut columns: Vec<Vec<BigUint>> = Vec::new();
    let mut publics: Vec<BigUint> = Vec::new();
    for (j, rule) in rules.iter().enumerate() {
        let mut x: Vec<BigUint> = (0..N).map(|i| BigUint::from(1000 + 100 * j + 7 * i + 2)).collect();
        let edge = broken.filter(|(b, _)| *b == j).map(|(_, edge)| edge);
        match *rule {
            Rule::FirstRow | Rule::LastRow => {
                let row = rule.rows().start;
                let mut p = (&x[row] * &x[row]) % &r;
                if edge.is_some() {
                    p = (p + 1u32) % &r;
                }
                publics.push(p);
            }
            Rule::Next { min, max } => {
                for i in rule.rows() {
                    x[(i + 1) % N] = (&x[i] * &x[i] + k_at(i)) % &r;
                }
                match edge {
                    Some(Edge::First) => x[min as usize] += 1u32,
                    Some(Edge::Last) => x[(N - max as usize) % N] += 1u32,
                    None => {}
                }
            }
            Rule::Prev { min, max } => {
                for i in rule.rows() {
                    x[i] = (&x[i - 1] * &x[i - 1] + k_at(i)) % &r;
                }
                match edge {
                    Some(Edge::First) => x[min as usize - 1] += 1u32,
                    Some(Edge::Last) => x[N - max as usize - 1] += 1u32,
                    None => {}
                }
            }
        }
        for v in x.iter_mut() {
            *v %= &r;
        }
        columns.push(x);
    }
    let y: Vec<BigUint> = (0..N).map(|i| (&columns[0][i] * k_at(i)) % &r).collect();
    columns.push(y);

    let fr = |v: &BigUint| FrBytes::from_decimal(&v.to_str_radix(10)).expect("a value below r");
    let columns: Vec<Vec<FrBytes>> = columns.iter().map(|c| c.iter().map(fr).collect()).collect();
    let stage1 = Stage1Witness::from_columns(N, &columns, vec![]).expect("the columns of N rows");
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 }],
        publics: publics.iter().map(fr).collect(),
        proof_values: vec![],
    }
}
