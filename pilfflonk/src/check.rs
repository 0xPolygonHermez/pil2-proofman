//! `pilfflonk check` (spec §4.4, "Depuració"; plan M25): the witness of an instance checked row by
//! row against the constraints of its AIR, without proving anything. It is the pilfflonk
//! counterpart of the STARK's `verify-constraints`.
//!
//! The constraints are those of section 2 of the AIR's `<air>.bin` (spec A.6): the pilout's, in its
//! order, then one per intermediate polynomial, `im − e`, which holds by construction since the
//! prover computes `im` from `e`. The C++ core computes the intermediate polynomials of stage 1 as
//! the prover does, then each constraint's numerator on the trace with the bytecode (M17), and
//! reports the rows `firstRow ≤ i < lastRow` where it is not 0. Stages ≥ 2 are not supported yet
//! (plan M30): their columns come from the std's prover hints, and a constraint of one is refused.

use crate::error::PilfflonkResult;
use crate::field::FrBytes;
use crate::prover::{native, ProvingKey, WitnessInstance};
use crate::witness::{AirInstanceRef, WitnessSource};

/// The failed rows [`check`] keeps of each constraint by default: the STARK's
/// `DEFAULT_N_PRINT_CONSTRAINTS`.
pub const DEFAULT_MAX_ROWS: usize = 10;

/// How to check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckOptions {
    /// The failed rows kept of each constraint, the first ones; they are all counted.
    pub max_rows: usize,
}

impl Default for CheckOptions {
    fn default() -> Self {
        Self { max_rows: DEFAULT_MAX_ROWS }
    }
}

/// A row where a constraint does not hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FailedRow {
    pub row: u64,
    /// The constraint's numerator there, not 0.
    pub value: FrBytes,
}

/// What [`check`] found of one constraint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintCheck {
    /// Its index in the `<air>.bin`: the pilout's constraints, then the im pols'.
    pub index: usize,
    pub stage: u64,
    /// It holds on the rows `first_row ≤ i < last_row`.
    pub first_row: u64,
    pub last_row: u64,
    /// The constraint of an intermediate polynomial.
    pub im_pol: bool,
    /// The PIL it comes from.
    pub line: String,
    /// The rows of its domain where it does not hold.
    pub n_failed_rows: u64,
    /// The first of them, at most [`CheckOptions::max_rows`], in increasing order.
    pub failed_rows: Vec<FailedRow>,
}

impl ConstraintCheck {
    pub fn holds(&self) -> bool {
        self.n_failed_rows == 0
    }
}

/// What [`check`] found: every constraint of the instance's AIR, in the order of its `<air>.bin`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckReport {
    pub air: AirInstanceRef,
    pub air_name: String,
    pub constraints: Vec<ConstraintCheck>,
}

impl CheckReport {
    /// Whether every constraint holds on every row of its domain.
    pub fn holds(&self) -> bool {
        self.constraints.iter().all(ConstraintCheck::holds)
    }

    /// The constraints that do not hold.
    pub fn failures(&self) -> impl Iterator<Item = &ConstraintCheck> {
        self.constraints.iter().filter(|c| !c.holds())
    }
}

/// Checks the one instance of `witness` against its AIR's constraints (see [the module](self)).
/// A witness that does not satisfy them is a report whose [`holds`](CheckReport::holds) is false,
/// not an error.
pub fn check(pk: &ProvingKey, witness: &impl WitnessSource, options: &CheckOptions) -> PilfflonkResult<CheckReport> {
    let read = WitnessInstance::read(witness)?;
    let air = read.air;
    let info = pk.air(air)?;
    let constraints =
        pk.ctx().constraints(air.airgroup_id, air.air_id).map_err(native("reading the constraints of the AIR"))?;
    // Nothing of it is committed: the blinding is never drawn.
    let mut instance = read.instance(pk, None)?;
    // One per constraint.
    let checks =
        instance.check(constraints.len(), options.max_rows).map_err(native("checking the witness row by row"))?;
    let constraints = constraints
        .into_iter()
        .zip(checks)
        .enumerate()
        .map(|(index, (constraint, found))| {
            let failed_rows = found
                .rows
                .into_iter()
                .map(|(row, value)| Ok(FailedRow { row, value: FrBytes::from_le_bytes(value)? }))
                .collect::<PilfflonkResult<_>>()?;
            Ok(ConstraintCheck {
                index,
                stage: constraint.stage,
                first_row: constraint.first_row,
                last_row: constraint.last_row,
                im_pol: constraint.im_pol,
                line: constraint.line,
                n_failed_rows: found.n_failed,
                failed_rows,
            })
        })
        .collect::<PilfflonkResult<_>>()?;
    Ok(CheckReport { air, air_name: info.name.clone(), constraints })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constraint(index: usize, n_failed_rows: u64) -> ConstraintCheck {
        ConstraintCheck {
            index,
            stage: 1,
            first_row: 0,
            last_row: 8,
            im_pol: false,
            line: format!("c{index}"),
            n_failed_rows,
            failed_rows: (0..n_failed_rows.min(2)).map(|row| FailedRow { row, value: FrBytes::from_u64(1) }).collect(),
        }
    }

    #[test]
    fn a_report_holds_only_if_every_constraint_does() {
        let air = AirInstanceRef { airgroup_id: 0, air_id: 0 };
        let mut report =
            CheckReport { air, air_name: "A".into(), constraints: vec![constraint(0, 0), constraint(1, 0)] };
        assert!(report.holds());
        assert_eq!(report.failures().count(), 0);
        report.constraints.push(constraint(2, 5));
        assert!(!report.holds());
        assert_eq!(report.failures().map(|c| c.index).collect::<Vec<_>>(), [2]);
        let empty = CheckReport { air, air_name: "A".into(), constraints: vec![] };
        assert!(empty.holds(), "an AIR without constraints");
    }

    #[test]
    fn by_default_it_keeps_the_starks_ten_rows() {
        assert_eq!(CheckOptions::default().max_rows, 10);
    }
}
