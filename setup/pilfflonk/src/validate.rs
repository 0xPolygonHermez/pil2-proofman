//! Reading a pilout for pilfflonk: what the setup refuses before any pass runs
//! (pilfflonk/docs/README.md#what-the-setup-refuses).
//!
//! [`validate`] checks a whole pilout at once, but for three of those cases, which need what only
//! the passes and the layout know (`crate::layout`):
//!
//! - the prover hints the setup supports, `im_col`, `gsum_col` and `gprod_col`, must produce every
//!   column of stage 2 or above, each once, from what the prover computes before them:
//!   [`check_prover_hints`], on the hints as the passes process them (the ones `<air>.bin` holds,
//!   `crate::bytecode`); [`validate`] only checks their names, and that a stage with columns has
//!   some hint;
//! - the extended domain must fit in the 2-adicity of BN128: [`check_extended_domain`], called
//!   with the `nBitsExt` of `layout::Degrees` (pilfflonk/docs/protocol.md#degrees);
//! - the ptau must hold as many powers `[τ^i]₁` as the largest `degree` of the layout:
//!   [`crate::keys::write_srs`] asks the C++ reader for exactly that many, and the reader refuses a
//!   ptau with fewer before it reads a point.
//!
//! And the fixed values `≥ r` are refused where they are decoded, by
//! [`crate::fixed::FixedColumns::from_air`]; [`validate`] checks every other constant.

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, expression, global_expression, global_operand, operand};
use pil_info::pil::gen_code::{ProcessedHint, ProcessedHintField};
use pil_info::{FieldCfg, PilInfoResult};
use proofman_pilfflonk::global_info::MAX_NBITS;

use crate::error::SetupError;

/// The witness and debug hints (pilfflonk/docs/README.md#what-the-setup-refuses): the std's
/// witness computation and debugging read them, the prover does not. The setup ignores them.
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

/// The prover hints, those of the std's buses.
pub const PROVER_HINTS: [&str; 4] = ["gsum_col", "gprod_col", "im_col", "im_airval"];

/// The prover hints the setup supports (pilfflonk/docs/protocol.md#hint-columns): the intermediate
/// columns of the terms of the std's buses (`im_col`), and their running product (`gprod_col`) and
/// sum (`gsum_col`), which produce the columns of stage 2. In the order the prover computes the
/// hints of a stage in, the STARK's (`pil2-stark/src/starkpil/gen_proof.hpp`): the `im_col` ones
/// (`calculateImHints`), then the `gprod_col` ones and the `gsum_col` ones (`calculateWitnessSTD`),
/// each in the pilout's order (`getHintIdsByName`). The other one, `im_airval`, computes an air
/// value, which v1 has none of (pilfflonk/docs/README.md#scope).
pub const SUPPORTED_PROVER_HINTS: [&str; 3] = ["im_col", "gprod_col", "gsum_col"];

/// The AIR of a pilout that passed [`validate`]: its only one.
#[derive(Clone, Copy, Debug)]
pub struct ValidAir<'a> {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub air: &'a pb::Air,
}

/// `r`, the modulus of `Fr`.
fn r() -> BigUint {
    FieldCfg::bn128().modulus().clone()
}

/// A name for an AIR in the errors: its name, or its position if it has none.
pub(crate) fn air_label(air: &pb::Air, airgroup_id: usize, air_id: usize) -> String {
    air.name.clone().unwrap_or_else(|| format!("{air_id} of airgroup {airgroup_id}"))
}

/// Checks `pilout` against what the setup refuses
/// (pilfflonk/docs/README.md#what-the-setup-refuses), but for what [the module](self) leaves to
/// others, and returns its AIR. In order: the base field; more than one AIR; the hints, before the
/// values that `im_airval` brings, so that it is told about; what else v1 leaves out
/// (pilfflonk/docs/README.md#scope): air values, airgroup values, proof values and global
/// constraints; the AIR's number of rows; custom commits, periodic columns and public tables; the
/// columns of stage 2 or above; and the constants of the expressions.
pub fn validate(pilout: &pb::PilOut) -> Result<ValidAir<'_>, SetupError> {
    check_base_field(pilout)?;
    let valid = only_air(pilout)?;
    let air = valid.air;
    let label = air_label(air, valid.airgroup_id, valid.air_id);
    let n_prover_hints = check_hints(pilout)?;
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
    // Every column of stage 2 or above must be produced by a supported prover hint: without any,
    // none is. Which column each one produces, [`check_prover_hints`] checks on the passes' hints.
    if n_prover_hints == 0 {
        if let Some((i, &n_columns)) = air.stage_widths.iter().enumerate().skip(1).find(|(_, &w)| w > 0) {
            return Err(SetupError::StageWithoutHint { air: label, stage: i + 1, n_columns });
        }
    }
    check_constants(pilout, air, &label)?;
    Ok(valid)
}

/// Checks that the extended domain, of `2^n_bits_ext` points, fits in the 2-adicity of BN128
/// (pilfflonk/docs/protocol.md#degrees). The command calls it with the `nBitsExt` of
/// `layout::Degrees`.
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
    } else if base_field == *FieldCfg::goldilocks().modulus() {
        Err(SetupError::GoldilocksPilout)
    } else {
        Err(SetupError::NotBn128 { base_field: base_field.to_str_radix(10) })
    }
}

/// What v1 leaves out besides other AIRs (pilfflonk/docs/README.md#scope): air values, airgroup
/// values, proof values and global constraints. The values are counted where the pilout declares
/// them and by their symbols, so that either one is enough to refuse them.
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

/// The hints of the pilout by name (pilfflonk/docs/README.md#what-the-setup-refuses): the witness
/// and debug ones are ignored, `im_col`, `gsum_col` and `gprod_col` must be of the AIR, and the
/// others are refused, `witness_bits` among them: the packed trace rows it asks for are not
/// accepted yet. Returns the number of `im_col`, `gsum_col` and `gprod_col`.
fn check_hints(pilout: &pb::PilOut) -> Result<usize, SetupError> {
    let mut n_supported = 0;
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
        match name {
            _ if SUPPORTED_PROVER_HINTS.contains(&name) => {
                if hint.air_id.is_none() {
                    return Err(SetupError::InvalidPilout(format!(
                        "the pilout has the prover hint `{name}` of no air: it produces a column of an AIR"
                    )));
                }
                n_supported += 1;
            }
            "im_airval" => return Err(SetupError::ImAirvalHint { location }),
            "witness_bits" => return Err(SetupError::PackedTrace { location }),
            _ => return Err(SetupError::UnknownHint { name: name.to_string(), location }),
        }
    }
    Ok(n_supported)
}

/// The value of a field of a processed hint that holds one value, not an array.
fn single<'a>(hint: &'a ProcessedHint, field: &str, label: &str) -> Result<Option<&'a ProcessedHintField>, SetupError> {
    let Some(entry) = hint.fields.iter().find(|f| f.name == field) else {
        return Ok(None);
    };
    match entry.values.as_slice() {
        [value] if value.pos.is_empty() => Ok(Some(value)),
        _ => Err(SetupError::ProverHint {
            hint: hint.name.clone(),
            air: label.to_string(),
            reason: format!("its field `{field}` holds an array, and the std's holds one value"),
        }),
    }
}

/// A supported prover hint, as [`check_prover_hints`] reads it: where the prover computes it, and
/// what it reads.
struct ProverHint<'a> {
    hint: &'a ProcessedHint,
    /// Its kind's place in [`SUPPORTED_PROVER_HINTS`], the order of the STARK.
    order: usize,
    /// Its reference: the column's `cmPolsMap` index, and its stage.
    column: usize,
    stage: usize,
    /// The columns (`cmPolsMap` indices) its numerator and its denominator read, by field.
    reads: [(&'static str, Vec<usize>); 2],
}

/// Checks the prover hints of the passes' result (the `im_col`, `gsum_col` and `gprod_col` of the
/// AIR: pilfflonk/docs/README.md#what-the-setup-refuses) against what the prover computes of them
/// (pilfflonk/docs/protocol.md#hint-columns), as the STARK's `calculateImHints` does with
/// `multiplyHintFields` and `calculateWitnessSTD` with `accMulHintFields`
/// (`pil2-stark/src/starkpil/hints.cpp`):
///
/// - `reference` is a column of stage 2 or above, read at its own row, and not an im pol: the
///   quotient `numerator/denominator` of an `im_col` goes there, and the running sum or product of
///   `numerator_air/denominator_air` of a `gsum_col` or `gprod_col`;
/// - the numerator and the denominator are each an expression, a column at an opening point or a
///   number (the operands the STARK's `addHintField` takes, air values aside: v1 has none), and
///   read only columns the prover computes before the hint: fixed ones, those of the stages before
///   the reference's, and of its stage those of the hints before it in the STARK's order
///   ([`SUPPORTED_PROVER_HINTS`]): an `im_col` may read the `im_col` columns before it, which the
///   std's product bus chains, and a `gsum_col` or `gprod_col` those of the `im_col` hints, but
///   none reads an im pol of its stage, which the prover computes last;
/// - an `im_col` is of an AIR with a `gsum_col` or a `gprod_col`: the STARK's `calculateImHints`
///   computes none otherwise;
/// - `result`, if a `gsum_col` or `gprod_col` has it, is a number: it is the airgroup value the
///   column's last row updates otherwise, with `numerator_direct/denominator_direct`, and v1 has
///   none. The std writes a number there in `STD_MODE_ONE_INSTANCE`, and the prover ignores the
///   three fields, as `calculateWitnessSTD` does when the AIR has no airgroup value;
/// - every column of stage 2 or above but the im pols is the reference of one hint exactly.
///
/// `label` is what the errors call the AIR. [`validate`] has checked the hints' names.
pub fn check_prover_hints(result: &PilInfoResult, label: &str) -> Result<(), SetupError> {
    let setup = &result.setup;
    let cm_pols = &setup.cm_pols_map;
    let code = &result.pil_code.expressions_info.expressions_code;
    let all = &result.pil_code.expressions_info.hints_info;
    let refuse = |hint: &ProcessedHint, reason: String| SetupError::ProverHint {
        hint: hint.name.clone(),
        air: label.to_string(),
        reason,
    };
    let has_bus = all.iter().any(|h| h.name == "gsum_col" || h.name == "gprod_col");

    let mut hints = Vec::new();
    let mut produced = vec![0usize; cm_pols.len()];
    for hint in all.iter() {
        let Some(order) = SUPPORTED_PROVER_HINTS.iter().position(|name| *name == hint.name) else {
            continue;
        };
        let refuse = |reason: String| refuse(hint, reason);
        let field = |name: &str| -> Result<&ProcessedHintField, SetupError> {
            single(hint, name, label)?.ok_or_else(|| refuse(format!("it has no field `{name}`")))
        };
        let im_col = hint.name == "im_col";
        if im_col && !has_bus {
            return Err(refuse(
                "the AIR has no gsum_col or gprod_col, and the STARK's calculateImHints computes im_col only in one \
                 that has"
                    .to_string(),
            ));
        }

        let reference = field("reference")?;
        let (column, stage) =
            match (reference.op.as_str(), reference.id.and_then(|id| cm_pols.get(id).map(|p| (id, p)))) {
                ("cm", Some((id, p))) if p.stage.unwrap_or(0) >= 2 && !p.im_pol && reference.row_offset == Some(0) => {
                    (id, p.stage.unwrap_or(0))
                }
                (op, _) => {
                    return Err(refuse(format!(
                        "its reference is {op} {:?}, not a column of stage 2 or above at its own row",
                        reference.id
                    )))
                }
            };
        produced[column] += 1;

        let fields = if im_col { ["numerator", "denominator"] } else { ["numerator_air", "denominator_air"] };
        let mut reads = fields.map(|name| (name, Vec::new()));
        for (name, read) in reads.iter_mut() {
            let value = field(name)?;
            // The columns the operand reads: its own, or those its expression's code does.
            *read = match value.op.as_str() {
                "number" | "const" => vec![],
                "cm" => value.id.into_iter().collect(),
                "tmp" => match code.iter().find(|e| Some(e.exp_id) == value.id) {
                    Some(e) => {
                        e.code.iter().flat_map(|c| &c.src).filter(|r| r.ref_type == "cm").map(|r| r.id).collect()
                    }
                    None => return Err(refuse(format!("its {name} is expression {:?}, which has no code", value.id))),
                },
                op => {
                    return Err(refuse(format!(
                        "its {name} is a {op}, and the prover takes an expression, a column or a number there"
                    )))
                }
            };
            if matches!(value.op.as_str(), "cm" | "const") && value.row_offset_index.is_none_or(|i| i < 0) {
                return Err(refuse(format!("its {name} reads a column at an offset that is not an opening point")));
            }
        }
        if !im_col {
            if let Some(result) = single(hint, "result", label)? {
                if result.op != "number" {
                    return Err(refuse(format!(
                        "its result is a {}: it updates an airgroup value, which pilfflonk does not support \
                         (pilfflonk/docs/README.md#scope); compile the std with set_std_mode(STD_MODE_ONE_INSTANCE)",
                        result.op
                    )));
                }
            }
        }
        hints.push(ProverHint { hint, order, column, stage, reads });
    }

    // The order the prover computes them in: stage by stage, and in each the STARK's. A column is there
    // before a hint if it is of an earlier stage, or a column of its stage an earlier hint gives.
    hints.sort_by_key(|h| (h.stage, h.order));
    let mut computed = vec![false; cm_pols.len()];
    for h in &hints {
        for (name, reads) in &h.reads {
            let before = |id: usize| computed[id] || cm_pols[id].stage.unwrap_or(0) < h.stage;
            if let Some(&id) = reads.iter().find(|&&id| id >= cm_pols.len() || !before(id)) {
                return Err(refuse(
                    h.hint,
                    format!(
                        "its {name} reads cm {id}, which is not computed before it: of stage {}, the prover computes \
                         the columns of the im_col hints in their order, then the gprod_col ones and the gsum_col \
                         ones, and the im pols last",
                        h.stage
                    ),
                ));
            }
        }
        computed[h.column] = true;
    }

    for (id, p) in cm_pols.iter().enumerate() {
        let stage = p.stage.unwrap_or(0);
        if stage < 2 || stage > setup.n_stages || p.im_pol || produced[id] == 1 {
            continue;
        }
        if produced[id] > 1 {
            return Err(SetupError::ProverHint {
                hint: "im_col/gsum_col/gprod_col".to_string(),
                air: label.to_string(),
                reason: format!("{} hints produce column {}", produced[id], p.name),
            });
        }
        let n_columns = cm_pols
            .iter()
            .enumerate()
            .filter(|(i, q)| q.stage == Some(stage) && !q.im_pol && produced[*i] == 0)
            .count() as u32;
        return Err(SetupError::StageWithoutHint { air: label.to_string(), stage, n_columns });
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
/// over BN128 has none that is not, and reducing one would hide a compiler bug
/// (pilfflonk/docs/README.md#compile-pil).
/// The expressions the hints refer to are the AIR's; the numbers of the prover hints' own fields,
/// the bytecode checks as it encodes them (`crate::bytecode`).
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
