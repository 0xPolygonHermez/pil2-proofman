//! Trusted checker: the setup's artefacts implement exactly the r1cs.
//!
//! Inputs are artefacts only — the r1cs, the compiled pilout (constraints, symbols, hints, its own
//! fixed columns), plonk2pil's fixed columns and its exec (a hint) — never the generator's
//! structures. Pure: no I/O; [`load`] adapts the artefacts.
//!
//! - **C** ([`pil`]): the plonk slots, public bindings and connection columns are read off the pilout.
//! - **B** ([`fixed`], [`gates`]): copy classes from `S`; the equation each slot enforces per row;
//!   every custom-gate use on cells of its signals' classes, anchor firing, parameters in place,
//!   band selectors exactly the located bands'.
//! - **A** ([`r1cs_plonk`]): every r1cs constraint implied by an enforced equation.
//!
//! Trusted: the PIL circuit libraries (a gate's constraints implement it when its anchor and band
//! selectors fire), the std's connection bus, `blake3Tables`, [`gates::gates_of`] (which cell a
//! template binds each signal of a circom custom gate to), the pil2 compiler, and the writers of
//! the checked in-memory artefacts to disk (`WriteFixed`, the `.const`).

pub mod fixed;
pub mod gates;
pub mod load;
pub mod pil;
pub mod poly;
pub mod r1cs_plonk;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use pil_info::core::ir::ConstraintSystem;
use proofman_common::exec_format::ExecFile;

use crate::plonk2pil::field::PlonkField;
use crate::plonk2pil::r1cs::types::R1csFile;
use fixed::{Columns, FixedSource};

/// The air template a pil instantiates, and its parameters (from the PIL text).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Template {
    pub file: String,
    pub params: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckError {
    /// An artefact (pilout, fixed column, exec hint, template) lacks the structure checked against.
    Artefact(String),
    /// `S_col[row]` is not an identity, or repeats one.
    NotAPermutation {
        col: usize,
        row: usize,
    },
    /// The exec map puts `wire` in two copy classes.
    WireSplit {
        wire: u32,
    },
    NotImplied {
        r1cs: u32,
        why: String,
    },
    PublicUnbound {
        public: u32,
        why: String,
    },
    GateNotFound {
        gate: u32,
        name: String,
        why: String,
    },
    GateParam {
        gate: u32,
        why: String,
    },
    SelectorPattern {
        column: String,
        row: usize,
        why: String,
    },
}

impl fmt::Display for CheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckError::Artefact(s) => write!(f, "{s}"),
            CheckError::NotAPermutation { col, row } => write!(f, "S[{col}] at row {row} is not a fresh identity"),
            CheckError::WireSplit { wire } => write!(f, "wire {wire} is placed in two copy classes"),
            CheckError::NotImplied { r1cs, why } => write!(f, "r1cs constraint {r1cs} is not enforced: {why}"),
            CheckError::PublicUnbound { public, why } => write!(f, "public {public} is not bound: {why}"),
            CheckError::GateNotFound { gate, name, why } => {
                write!(f, "custom gate use {gate} ({name}) is not placed: {why}")
            }
            CheckError::GateParam { gate, why } => write!(f, "custom gate use {gate}: {why}"),
            CheckError::SelectorPattern { column, row, why } => write!(f, "column {column} at row {row}: {why}"),
        }
    }
}

impl std::error::Error for CheckError {}

/// The artefacts of one air.
pub struct CheckInput<'a, F> {
    pub r1cs: &'a R1csFile<F>,
    pub cs: &'a ConstraintSystem,
    pub fixed: &'a dyn FixedSource<F>,
    pub exec: &'a ExecFile<F>,
    pub template: &'a Template,
}

#[derive(Debug, Clone)]
pub struct CheckReport {
    pub n_rows: usize,
    pub slots: usize,
    pub copy_classes: usize,
    pub publics: usize,
    pub a: r1cs_plonk::TierA,
    pub gates: gates::TierGates,
    pub secs: f64,
}

impl fmt::Display for CheckReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (a, g) = (&self.a, &self.gates);
        write!(
            f,
            "{} rows, {} slots, {} classes; {} r1cs constraints by {} equations ({} unique, {} additions); \
             {} gate uses in {} bands; {} publics; {:.2}s",
            self.n_rows,
            self.slots,
            self.copy_classes,
            a.constraints,
            a.equations,
            a.unique,
            a.additions,
            g.uses,
            g.bands,
            self.publics,
            self.secs
        )
    }
}

/// Tiers C, B and A over one air's artefacts.
pub fn check_plonk<F: PlonkField>(inp: &CheckInput<F>) -> Result<CheckReport, CheckError> {
    let (r1cs, cs, exec) = (inp.r1cs, inp.cs, inp.exec);
    let t = std::time::Instant::now();
    let air = pil::analyse::<F>(cs)?;
    let n = air.n_rows;
    // cells `wit·N + row` stay below `FREE`
    if (cs.witness.len() as u64) * (n as u64) >= u64::from(r1cs_plonk::FREE) {
        return Err(CheckError::Artefact(format!(
            "{} witness columns of {n} rows: cell ids overflow",
            cs.witness.len()
        )));
    }
    let mut cols = Columns::new(inp.fixed, n);
    let width = (0..).take_while(|&c| pil::wit_id(cs, "a", c).is_some()).count();
    let a_cols: Vec<u32> = (0..width as u32).map(|c| pil::wit_id(cs, "a", c).expect("counted")).collect();
    let cls = fixed::classes(&air, &mut cols, cs.witness.len())?;
    let n_wires = r1cs.header.n_vars as usize + exec.additions.len();
    let label = fixed::labels(exec, &cls, &a_cols, n_wires)?;
    let vars = r1cs_plonk::wire_vars(r1cs, &label);
    let enf = fixed::Enforced::new(&air, &mut cols)?;
    let a = r1cs_plonk::check(r1cs, exec, &label, &vars, &enf, &mut cols, &cls)?;
    let publics = check_publics(r1cs, &air, &vars, &cls, &mut cols)?;
    let gates = gates::check(r1cs, cs, &air, inp.template, &a_cols, &vars, &cls, &mut cols)?;
    Ok(CheckReport {
        n_rows: n,
        slots: air.slots.len(),
        copy_classes: cls.n_classes,
        publics,
        a,
        gates,
        secs: t.elapsed().as_secs_f64(),
    })
}

/// Each r1cs public `i` (wire `i + 1`) is bound to `publics[i]` at a cell of its class.
fn check_publics<F: PlonkField>(
    r1cs: &R1csFile<F>,
    air: &pil::PilAir<F>,
    vars: &r1cs_plonk::Vars,
    cls: &fixed::Classes,
    cols: &mut Columns<F>,
) -> Result<usize, CheckError> {
    let n_pub = r1cs.header.n_outputs + r1cs.header.n_pub_inputs;
    let mut fired: HashMap<&pil::FixedPoly<F>, Vec<usize>> = HashMap::new();
    for b in &air.publics {
        if !fired.contains_key(&b.sel) {
            let s = cols.eval(&b.sel)?;
            fired.insert(&b.sel, (0..s.len()).filter(|&r| !s[r].is_zero()).collect());
        }
    }
    for i in 0..n_pub {
        let v = vars.of(i + 1);
        let bound = air
            .publics
            .iter()
            .filter(|b| b.public == i)
            .any(|b| fired[&b.sel].iter().any(|&r| cls.class(b.cell.0, fixed::wrap(r, b.cell.1, cls.n)) == v));
        if !bound {
            return Err(CheckError::PublicUnbound {
                public: i,
                why: format!("no binding reaches the class of wire {}", i + 1),
            });
        }
    }
    Ok(n_pub as usize)
}
