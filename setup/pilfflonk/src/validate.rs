//! Reading a pilout for pilfflonk: what the setup refuses before any pass runs (spec §4.2.1).
//!
//! [`validate`] checks a whole pilout at once, but for two of §4.2.1's cases, which need what
//! only the passes and the layout know (`crate::layout`):
//!
//! - the extended domain must fit in the 2-adicity of BN254: [`check_extended_domain`], called
//!   with the `nBitsExt` of A.1 (`layout::Degrees`);
//! - the ptau must hold as many powers `[τ^i]₁` as the largest `degree` of the layout:
//!   [`crate::keys::write_srs`] asks the C++ reader for exactly that many, and the reader refuses a
//!   ptau with fewer before it reads a point.
//!
//! And the fixed values `≥ r` are refused where they are decoded, by
//! [`crate::fixed::FixedColumns::from_air`]; [`validate`] checks every other constant.

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, expression, global_expression, global_operand, operand};
use pil_info::FieldCfg;
use proofman_pilfflonk::global_info::MAX_NBITS;

use crate::error::SetupError;

/// The witness and debug hints of spec §3.4: the std's witness computation and debugging read
/// them, the prover does not. The setup ignores them.
pub const WITNESS_AND_DEBUG_HINTS: [&str; 12] = [
    "gsum_debug_data",
    "gsum_debug_data_global",
    "gprod_debug_data",
    "gprod_debug_data_global",
    "range_def",
    "specified_ranges",
    "specified_ranges_data",
    "virtual_table_data",
    "virtual_table_data_global",
    "std_sum_users",
    "std_prod_users",
    "std_rc_users",
];

/// The prover hints of spec §3.4, those of the std's buses. Fase 1 supports none of them (plan
/// R2): the setup refuses them, and so every column of stage 2 or above.
pub const PROVER_HINTS: [&str; 4] = ["gsum_col", "gprod_col", "im_col", "im_airval"];

/// The AIR of a pilout that passed [`validate`]: its only one.
#[derive(Clone, Copy, Debug)]
pub struct ValidAir<'a> {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub air: &'a pb::Air,
}

/// Goldilocks, `2^64 − 2^32 + 1`: the field of a pilout compiled without a `prime`.
const GOLDILOCKS: u64 = 0xFFFF_FFFF_0000_0001;

/// `r`, the modulus of `Fr`.
fn r() -> BigUint {
    FieldCfg::bn254().modulus().clone()
}

/// A name for an AIR in the errors: its name, or its position if it has none.
pub(crate) fn air_label(air: &pb::Air, airgroup_id: usize, air_id: usize) -> String {
    air.name.clone().unwrap_or_else(|| format!("{air_id} of airgroup {airgroup_id}"))
}

/// Checks `pilout` against spec §4.2.1, but for what [the module](self) leaves to others, and
/// returns its AIR. In order: the base field; what v1 leaves out (D2): more than one AIR, air
/// values, airgroup values, proof values and global constraints; the AIR's number of rows; custom
/// commits, periodic columns and public tables; the hints; the columns of stage 2 or above; and
/// the constants of the expressions.
pub fn validate(pilout: &pb::PilOut) -> Result<ValidAir<'_>, SetupError> {
    check_base_field(pilout)?;
    let valid = only_air(pilout)?;
    let air = valid.air;
    let label = air_label(air, valid.airgroup_id, valid.air_id);
    check_values(pilout, air, &label)?;

    let num_rows = air.num_rows.unwrap_or(0);
    if !num_rows.is_power_of_two() || u64::from(num_rows.trailing_zeros()) > MAX_NBITS {
        return Err(SetupError::NumRows { air: label, num_rows });
    }
    if !air.custom_commits.is_empty() {
        return Err(SetupError::CustomCommits { air: label, n: air.custom_commits.len() });
    }
    if !air.periodic_cols.is_empty() {
        return Err(SetupError::PeriodicColumns { air: label, n: air.periodic_cols.len() });
    }
    if !pilout.public_tables.is_empty() {
        return Err(SetupError::PublicTables { n: pilout.public_tables.len() });
    }
    check_hints(pilout)?;
    // Every column of stage 2 or above must be produced by a prover hint, and Fase 1 supports
    // none: the hints are checked first, so that a pilout with the std's buses is told about the
    // hint rather than its columns.
    if let Some((i, &n_columns)) = air.stage_widths.iter().enumerate().skip(1).find(|(_, &w)| w > 0) {
        return Err(SetupError::StageWithoutHint { air: label, stage: i + 1, n_columns });
    }
    check_constants(pilout, air, &label)?;
    Ok(valid)
}

/// Checks that the extended domain of A.1, of `2^n_bits_ext` points, fits in the 2-adicity of
/// BN254 (spec §4.2.1). The command calls it with the `nBitsExt` of `layout::Degrees`.
pub fn check_extended_domain(n_bits_ext: u64) -> Result<(), SetupError> {
    if n_bits_ext > MAX_NBITS {
        return Err(SetupError::ExtendedDomain { n_bits_ext });
    }
    Ok(())
}

fn check_base_field(pilout: &pb::PilOut) -> Result<(), SetupError> {
    let base_field = BigUint::from_bytes_be(&pilout.base_field);
    if base_field == r() {
        Ok(())
    } else if base_field == BigUint::from(GOLDILOCKS) {
        Err(SetupError::GoldilocksPilout)
    } else {
        Err(SetupError::NotBn254 { base_field: base_field.to_str_radix(10) })
    }
}

/// What v1 leaves out besides other AIRs (spec §4.2.1, D2): air values, airgroup values, proof
/// values and global constraints. The values are counted where the pilout declares them and by
/// their symbols, so that either one is enough to refuse them.
fn check_values(pilout: &pb::PilOut, air: &pb::Air, label: &str) -> Result<(), SetupError> {
    let symbols = |kind: pb::SymbolType| pilout.symbols.iter().filter(|s| s.r#type == kind as i32).count();
    let air_values = air.air_values.len().max(symbols(pb::SymbolType::AirValue));
    if air_values > 0 {
        return Err(SetupError::AirValues { air: label.to_string(), n: air_values });
    }
    let airgroup_values = pilout.air_groups.iter().map(|ag| ag.air_group_values.len()).sum::<usize>();
    let airgroup_values = airgroup_values.max(symbols(pb::SymbolType::AirGroupValue));
    if airgroup_values > 0 {
        return Err(SetupError::AirgroupValues { n: airgroup_values });
    }
    let proof_values = pilout.num_proof_values.iter().map(|&n| n as usize).sum::<usize>();
    let proof_values = proof_values.max(symbols(pb::SymbolType::ProofValue));
    if proof_values > 0 {
        return Err(SetupError::ProofValues { n: proof_values });
    }
    if !pilout.constraints.is_empty() {
        return Err(SetupError::GlobalConstraints { n: pilout.constraints.len() });
    }
    Ok(())
}

fn only_air(pilout: &pb::PilOut) -> Result<ValidAir<'_>, SetupError> {
    let mut airs = pilout.air_groups.iter().enumerate().flat_map(|(airgroup_id, ag)| {
        ag.airs.iter().enumerate().map(move |(air_id, air)| ValidAir { airgroup_id, air_id, air })
    });
    match (airs.next(), airs.next()) {
        (Some(air), None) => Ok(air),
        _ => Err(SetupError::AirCount { n_airs: pilout.air_groups.iter().map(|ag| ag.airs.len()).sum() }),
    }
}

fn check_hints(pilout: &pb::PilOut) -> Result<(), SetupError> {
    for hint in &pilout.hints {
        let name = hint.name.as_str();
        if WITNESS_AND_DEBUG_HINTS.contains(&name) {
            continue;
        }
        let location = match (hint.air_group_id, hint.air_id) {
            (Some(ag), Some(a)) => {
                let air = pilout.air_groups.get(ag as usize).and_then(|g| g.airs.get(a as usize));
                match air {
                    Some(air) => format!("air {}", air_label(air, ag as usize, a as usize)),
                    None => return Err(SetupError::InvalidPilout(format!("hint `{name}` is of no air"))),
                }
            }
            _ => "the pilout".to_string(),
        };
        return Err(if PROVER_HINTS.contains(&name) {
            SetupError::UnsupportedProverHint { name: name.to_string(), location }
        } else {
            SetupError::UnknownHint { name: name.to_string(), location }
        });
    }
    Ok(())
}

/// The operands of an expression of an AIR.
fn operands(e: &pb::Expression) -> [Option<&pb::Operand>; 2] {
    use expression::Operation as Op;
    match &e.operation {
        Some(Op::Add(o)) => [o.lhs.as_ref(), o.rhs.as_ref()],
        Some(Op::Sub(o)) => [o.lhs.as_ref(), o.rhs.as_ref()],
        Some(Op::Mul(o)) => [o.lhs.as_ref(), o.rhs.as_ref()],
        Some(Op::Neg(o)) => [o.value.as_ref(), None],
        None => [None, None],
    }
}

/// The operands of a global expression.
fn global_operands(e: &pb::GlobalExpression) -> [Option<&pb::GlobalOperand>; 2] {
    use global_expression::Operation as Op;
    match &e.operation {
        Some(Op::Add(o)) => [o.lhs.as_ref(), o.rhs.as_ref()],
        Some(Op::Sub(o)) => [o.lhs.as_ref(), o.rhs.as_ref()],
        Some(Op::Mul(o)) => [o.lhs.as_ref(), o.rhs.as_ref()],
        Some(Op::Neg(o)) => [o.value.as_ref(), None],
        None => [None, None],
    }
}

/// The constants of the AIR's expressions and of the global ones must be below `r`: a pilout
/// over BN254 has none that is not, and reducing one would hide a compiler bug (spec §4.1, C4).
/// The hints are not looked into: the setup refuses the prover's and ignores the others.
fn check_constants(pilout: &pb::PilOut, air: &pb::Air, label: &str) -> Result<(), SetupError> {
    let r = r();
    let refuse =
        |location: String, value: BigUint| SetupError::ConstantNotBelowR { location, value: value.to_str_radix(10) };
    for (i, e) in air.expressions.iter().enumerate() {
        for op in operands(e).into_iter().flatten() {
            if let Some(operand::Operand::Constant(c)) = &op.operand {
                let value = BigUint::from_bytes_be(&c.value);
                if value >= r {
                    return Err(refuse(format!("a constant of expression {i} of air {label}"), value));
                }
            }
        }
    }
    for (i, e) in pilout.expressions.iter().enumerate() {
        for op in global_operands(e).into_iter().flatten() {
            if let Some(global_operand::Operand::Constant(c)) = &op.operand {
                let value = BigUint::from_bytes_be(&c.value);
                if value >= r {
                    return Err(refuse(format!("a constant of global expression {i}"), value));
                }
            }
        }
    }
    Ok(())
}
