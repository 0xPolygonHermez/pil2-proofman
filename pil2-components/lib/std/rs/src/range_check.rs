use proofman_fields::PrimeField64;

use proofman_common::{ProofCtx, ProofmanError, ProofmanResult, SetupCtx};
use proofman_hints::{
    get_hint_field_constant, get_hint_field_gc_constant_a, get_hint_ids_by_name, HintFieldOptions, HintFieldValue,
};

use crate::{
    extract_field_element_as_usize, get_global_hint_field_constant_as, get_hint_field_constant_as,
    get_hint_field_constant_as_field, validate_binary_field,
};

/// The PIL range-check type vocabulary; only `SpecifiedRanges` carries a non-zero bias.
#[derive(Debug)]
enum StdRangeType {
    U8Air,
    U16Air,
    U8AirDouble,
    U16AirDouble,
    SpecifiedRanges,
}

struct HintCache {
    opid: u64,
    min: i64,
    rc_type: StdRangeType,
    is_virtual: bool,
}

/// Every virtual range table this proof defines, as (table id, bias): the row a lookup addresses is
/// `value + bias`, which is -min for a specified range and 0 for a predefined one.
///
/// Returned rather than registered -- the FFI call belongs to the host binary, see
/// `ProofCtx::prover_owned_tables`.
pub fn collect_prover_owned_ranges<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
) -> ProofmanResult<Vec<(u64, i64)>> {
    let Some(std_rc_users) = get_hint_ids_by_name(sctx.get_global_bin(), "std_rc_users").first().copied() else {
        return Ok(Vec::new());
    };

    let num_users = get_global_hint_field_constant_as::<usize, F>(sctx, std_rc_users, "num_users")?;
    let airgroup_ids = get_hint_field_gc_constant_a(sctx, std_rc_users, "airgroup_ids", false)?;
    let air_ids = get_hint_field_gc_constant_a(sctx, std_rc_users, "air_ids", false)?;

    let mut owned: Vec<(u64, i64)> = Vec::new();
    for i in 0..num_users {
        let airgroup_id = extract_field_element_as_usize(&airgroup_ids.values[i], "airgroup_id")?;
        let air_id = extract_field_element_as_usize(&air_ids.values[i], "air_id")?;
        let setup = sctx.get_setup(airgroup_id, air_id)?;

        for hint in get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "range_def") {
            let hint_data = parse_range_hint(sctx, pctx, airgroup_id, air_id, hint)?;
            if !hint_data.is_virtual {
                return Err(ProofmanError::StdError(format!(
                    "Range check {} is not virtual; every range table is expected to be virtual",
                    hint_data.opid
                )));
            }
            let bias = match hint_data.rc_type {
                StdRangeType::SpecifiedRanges => -hint_data.min,
                _ => 0,
            };
            owned.push((hint_data.opid, bias));
        }
    }

    owned.sort_unstable();
    owned.dedup();
    Ok(owned)
}

fn parse_range_hint<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    pctx: &ProofCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    hint: u64,
) -> ProofmanResult<HintCache> {
    let options = HintFieldOptions::default();
    let setup = sctx.get_setup(airgroup_id, air_id)?;
    let hint = hint as usize;

    let num = |name: &str| {
        get_hint_field_constant_as::<u64, F>(pctx, setup, airgroup_id, air_id, hint, name, options.clone())
    };
    let flag = |name: &str, label: &str| -> ProofmanResult<bool> {
        let v = get_hint_field_constant_as_field::<F>(pctx, setup, airgroup_id, air_id, hint, name, options.clone())?;
        validate_binary_field(v, label)
    };

    let opid = num("opid")?;
    // Read for validation only: a malformed flag must still be an error here.
    flag("predefined", "Predefined")?;
    let (min_val, min_neg) = (num("min")?, flag("min_neg", "Min neg")?);
    let (max_val, max_neg) = (num("max")?, flag("max_neg", "Max neg")?);
    let is_virtual = flag("is_virtual", "Is virtual")?;

    let HintFieldValue::String(rc_type_str) =
        get_hint_field_constant::<F>(pctx, setup, airgroup_id, air_id, hint, "type", options.clone())?
    else {
        return Err(ProofmanError::StdError("Type hint must be a string".to_string()));
    };

    // A negative bound reaches the trace as a field element near p.
    let signed = |val: u64, neg: bool| if neg { val as i128 - F::ORDER_U64 as i128 } else { val as i128 };
    let (min, max) = (signed(min_val, min_neg), signed(max_val, max_neg));
    if min > i64::MAX as i128 || max > i64::MAX as i128 {
        return Err(ProofmanError::StdError("Min/Max value is too large".to_string()));
    }

    let rc_type = match rc_type_str.as_str() {
        "U8" => StdRangeType::U8Air,
        "U16" => StdRangeType::U16Air,
        "U8Double" => StdRangeType::U8AirDouble,
        "U16Double" => StdRangeType::U16AirDouble,
        "Specified" => StdRangeType::SpecifiedRanges,
        _ => return Err(ProofmanError::StdError(format!("Invalid range check type: {rc_type_str}"))),
    };

    Ok(HintCache { opid, min: min as i64, rc_type, is_virtual })
}
