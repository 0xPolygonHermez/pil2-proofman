//! Synthetic `everyRow` pilouts (pilfflonk/docs/README.md#fixtures), built in code with `prost`, and
//! their witness generators, over BN128's `Fr` with `num-bigint`. pilfflonk has no other kind of
//! constraint: a domain constraint is refused at setup.
//!
//! Each AIR has `N = 2^4` rows, a fixed column `K = [1, 2, …, N]`, one witness column `x<j>` per
//! rule `j` of it (and a public `p<j>` for a `Square` rule), and a witness column `y` after them.
//! Its constraints are the rules', in their order, and then `y − x0·K`, all on every row:
//!
//! | Rule | Constraint | `x<j>` on row `i` |
//! |---|---|---|
//! | `Square` | `x·x − p` | `1000 + 100·j + 7·i + 2`, which `p = x·x` makes constant |
//! | `Scaled` | `x − K` | `K[i]` |
//!
//! so `y = x0·K`.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/domains.rs"] mod domains;`.

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, constraint, expression, operand, SymbolType};
use proofman_pilfflonk::{AirInstanceRef, FrBytes, InstanceWitness, Stage1Witness, Witness, BN128_R};

/// The AIRs' rows: `N = 2^4`.
pub const N_BITS: u32 = 4;
const N: usize = 1 << N_BITS;

/// A constraint of an AIR, on its own column (see the module).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    Square,
    Scaled,
}

/// The synthetic AIRs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Air {
    /// Two `Square` rules: two publics.
    Squares,
    /// Three `Scaled` rules: no publics.
    Scaled,
}

impl Air {
    pub fn name(self) -> &'static str {
        match self {
            Air::Squares => "Squares",
            Air::Scaled => "Scaled",
        }
    }

    pub fn rules(self) -> Vec<Rule> {
        match self {
            Air::Squares => vec![Rule::Square, Rule::Square],
            Air::Scaled => vec![Rule::Scaled, Rule::Scaled, Rule::Scaled],
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

/// The pilout of `air`, over BN128 (see the module).
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
            Rule::Square => {
                let square = e.mul(x(0), x(0));
                e.sub(square, public(n_publics));
                symbols.push(symbol(&format!("p{j}"), SymbolType::PublicValue, n_publics, None, false));
                n_publics += 1;
                C::AllRows(constraint::AllRows {
                    expression_idx: e.last(),
                    debug_line: line(&format!("x{j}*x{j} - p{j}")),
                })
            }
            Rule::Scaled => {
                e.sub(x(0), k());
                C::AllRows(constraint::AllRows { expression_idx: e.last(), debug_line: line(&format!("x{j} - K")) })
            }
        });
    }
    let x0_k = e.mul(witness_col(0, 0), k());
    e.sub(witness_col(y, 0), x0_k);
    let line = Some(format!("{}: y - x0*K", air.name()));
    constraints.push(C::AllRows(constraint::AllRows { expression_idx: e.last(), debug_line: line }));
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
    BigUint::parse_bytes(BN128_R.as_bytes(), 10).expect("r in decimal")
}

fn k_at(row: usize) -> BigUint {
    BigUint::from(row + 1)
}

/// The witness of `air`: one instance of air 0 of airgroup 0, with the columns `[x0, …, y]` and
/// the publics `p<j>` in the order of their rules.
pub fn witness(air: Air) -> Witness {
    let r = r();
    let mut columns: Vec<Vec<BigUint>> = Vec::new();
    let mut publics: Vec<BigUint> = Vec::new();
    for (j, rule) in air.rules().iter().enumerate() {
        let x: Vec<BigUint> = match rule {
            Rule::Square => {
                let x = BigUint::from(1000 + 100 * j + 2);
                publics.push((&x * &x) % &r);
                vec![x; N]
            }
            Rule::Scaled => (0..N).map(k_at).collect(),
        };
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
