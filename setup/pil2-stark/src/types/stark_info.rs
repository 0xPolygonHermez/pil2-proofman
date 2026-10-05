use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::collections::HashMap;

use crate::types::stark_struct::{StarkStep, StarkStruct};
use crate::types::{goldilocks_u64, GOLDILOCKS_MODULUS};

pub use pil_info::types::code::{CodeOperation, CodeType, OpType};

/// Polynomial map entry, shared by cmPolsMap, constPolsMap, challengesMap, etc.
#[derive(Debug, Clone, Default)]
pub struct PolMap {
    pub stage: u64,
    pub name: String,
    pub dim: u64,
    pub im_pol: bool,
    pub stage_pos: u64,
    pub stage_id: u64,
    pub commit_id: u64,
    pub exp_id: u64,
    pub pols_map_id: u64,
}

/// Evaluation map entry.
#[derive(Debug, Clone)]
pub struct EvMap {
    pub ev_type: String,
    pub id: u64,
    pub prime: i64,
    pub commit_id: u64,
    pub opening_pos: u64,
}

/// Custom commit info.
#[derive(Debug, Clone)]
pub struct CustomCommit {
    pub name: String,
}

/// Boundary descriptor.
#[derive(Debug, Clone)]
pub struct Boundary {
    pub name: String,
    pub offset_min: Option<u64>,
    pub offset_max: Option<u64>,
}

/// Mirrors the C++ StarkInfo loaded from starkinfo.json.
#[derive(Debug, Clone, Default)]
pub struct StarkInfo {
    pub stark_struct: StarkStruct,
    pub n_stages: u64,
    pub n_constants: u64,
    pub n_publics: u64,

    pub custom_commits: Vec<CustomCommit>,
    pub cm_pols_map: Vec<PolMap>,
    pub const_pols_map: Vec<PolMap>,
    pub challenges_map: Vec<PolMap>,
    pub airgroup_values_map: Vec<PolMap>,
    pub air_values_map: Vec<PolMap>,
    pub proof_values_map: Vec<PolMap>,

    pub ev_map: Vec<EvMap>,
    pub opening_points: Vec<i64>,
    pub boundaries: Vec<Boundary>,

    pub q_deg: u64,
    pub q_dim: u64,
    pub fri_exp_id: u64,
    pub c_exp_id: u64,

    pub map_sections_n: HashMap<String, u64>,
}

/// Expression code block as loaded from expressionsInfo JSON.
#[derive(Debug, Clone)]
pub struct ExpCode {
    pub exp_id: u64,
    pub stage: u64,
    pub tmp_used: u64,
    pub code: Vec<CodeOperation>,
    pub line: String,
    pub boundary: String,
    pub offset_min: u64,
    pub offset_max: u64,
    pub im_pol: u64,
}

/// Hint field value as loaded from hints JSON.
#[derive(Debug, Clone)]
pub struct HintFieldValue {
    pub op: String,
    pub id: u64,
    pub commit_id: u64,
    pub row_offset_index: u64,
    pub dim: u64,
    pub value: u64,
    pub string_value: String,
    pub airgroup_id: u64,
    pub pos: Vec<u64>,
}

/// Hint field containing a name and values.
#[derive(Debug, Clone)]
pub struct HintField {
    pub name: String,
    pub values: Vec<HintFieldValue>,
}

/// A single hint with a name and fields.
#[derive(Debug, Clone)]
pub struct Hint {
    pub name: String,
    pub fields: Vec<HintField>,
}

/// Container for all expressions info loaded from the JSON file.
#[derive(Debug, Clone)]
pub struct ExpressionsInfo {
    pub expressions_code: Vec<ExpCode>,
    pub constraints: Vec<ExpCode>,
    pub hints_info: Vec<Hint>,
}

/// Verifier info loaded from the JSON file.
#[derive(Debug, Clone)]
pub struct VerifierInfo {
    pub q_verifier: ExpCode,
    pub query_verifier: ExpCode,
}

/// Global constraints info loaded from JSON.
#[derive(Debug, Clone)]
pub struct GlobalConstraintsInfo {
    pub constraints: Vec<ExpCode>,
    pub hints: Vec<Hint>,
}

// Parsing helpers for loading from serde_json::Value

fn get_u64(v: &Value, key: &str) -> u64 {
    v.get(key).and_then(|x| x.as_u64()).unwrap_or(0)
}

fn get_i64(v: &Value, key: &str) -> i64 {
    v.get(key).and_then(|x| x.as_i64()).unwrap_or(0)
}

fn get_str<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(|x| x.as_str()).unwrap_or("")
}

fn get_bool(v: &Value, key: &str) -> bool {
    v.get(key).and_then(|x| x.as_bool()).unwrap_or(false)
}

/// The `"value"` of `v`, a string ([`goldilocks_u64`]) or a JSON number, as a canonical Goldilocks
/// element; 0 when `v` has none. Anything else is an error, not a 0.
fn get_goldilocks(v: &Value) -> Result<u64> {
    match v.get("value") {
        None => Ok(0),
        Some(Value::String(s)) => goldilocks_u64(s),
        Some(val) => match val.as_u64() {
            Some(n) if n < GOLDILOCKS_MODULUS => Ok(n),
            _ => {
                Err(anyhow!("the value {val} is not a Goldilocks element (an integer below p = {GOLDILOCKS_MODULUS})"))
            }
        },
    }
}

fn parse_code_type(v: &Value) -> Result<CodeType> {
    let type_str = get_str(v, "type");
    let op_type = OpType::parse(type_str)?;
    let value = get_goldilocks(v)?;
    Ok(CodeType {
        op_type,
        id: get_u64(v, "id"),
        prime: get_i64(v, "prime"),
        dim: get_u64(v, "dim"),
        value,
        commit_id: get_u64(v, "commitId"),
        boundary_id: get_u64(v, "boundaryId"),
        airgroup_id: get_u64(v, "airgroupId"),
    })
}

fn parse_code_operation(v: &Value) -> Result<CodeOperation> {
    let op = get_str(v, "op").to_string();
    let dest = parse_code_type(v.get("dest").unwrap_or(&Value::Null))?;
    let src: Vec<CodeType> = match v.get("src").and_then(|s| s.as_array()) {
        Some(arr) => arr.iter().map(parse_code_type).collect::<Result<_>>()?,
        None => Vec::new(),
    };
    Ok(CodeOperation { op, dest, src })
}

fn parse_pol_map(v: &Value) -> PolMap {
    PolMap {
        stage: get_u64(v, "stage"),
        name: get_str(v, "name").to_string(),
        dim: get_u64(v, "dim"),
        im_pol: get_bool(v, "imPol"),
        stage_pos: get_u64(v, "stagePos"),
        stage_id: get_u64(v, "stageId"),
        commit_id: get_u64(v, "commitId"),
        exp_id: get_u64(v, "expId"),
        pols_map_id: get_u64(v, "polsMapId"),
    }
}

fn parse_pol_map_array(v: &Value, key: &str) -> Vec<PolMap> {
    v.get(key).and_then(|a| a.as_array()).map(|arr| arr.iter().map(parse_pol_map).collect()).unwrap_or_default()
}

impl StarkInfo {
    /// Parse a StarkInfo from a serde_json::Value (the parsed starkinfo.json).
    pub fn from_json(j: &Value) -> Result<Self> {
        let ss = j.get("starkStruct").unwrap_or(&Value::Null);
        let steps: Vec<StarkStep> = ss
            .get("steps")
            .and_then(|s| s.as_array())
            .map(|arr| arr.iter().map(|step| StarkStep { n_bits: get_u64(step, "nBits") as usize }).collect())
            .unwrap_or_default();

        let stark_struct = StarkStruct {
            n_bits: get_u64(ss, "nBits") as usize,
            n_bits_ext: get_u64(ss, "nBitsExt") as usize,
            n_queries: get_u64(ss, "nQueries") as usize,
            hash_commits: get_bool(ss, "hashCommits"),
            last_level_verification: get_u64(ss, "lastLevelVerification") as usize,
            merkle_tree_arity: get_u64(ss, "merkleTreeArity") as usize,
            transcript_arity: get_u64(ss, "transcriptArity") as usize,
            merkle_tree_custom: get_bool(ss, "merkleTreeCustom"),
            verification_hash_type: get_str(ss, "verificationHashType").to_string(),
            steps,
            pow_bits: get_u64(ss, "powBits") as usize,
        };

        let custom_commits: Vec<CustomCommit> = j
            .get("customCommits")
            .and_then(|a| a.as_array())
            .map(|arr| arr.iter().map(|c| CustomCommit { name: get_str(c, "name").to_string() }).collect())
            .unwrap_or_default();

        let ev_map: Vec<EvMap> = j
            .get("evMap")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|e| EvMap {
                        ev_type: get_str(e, "type").to_string(),
                        id: get_u64(e, "id"),
                        prime: get_i64(e, "prime"),
                        commit_id: get_u64(e, "commitId"),
                        opening_pos: get_u64(e, "openingPos"),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let opening_points: Vec<i64> = j
            .get("openingPoints")
            .and_then(|a| a.as_array())
            .map(|arr| arr.iter().map(|v| v.as_i64().unwrap_or(0)).collect())
            .unwrap_or_default();

        let boundaries: Vec<Boundary> = j
            .get("boundaries")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|b| Boundary {
                        name: get_str(b, "name").to_string(),
                        offset_min: b.get("offsetMin").and_then(|v| v.as_u64()),
                        offset_max: b.get("offsetMax").and_then(|v| v.as_u64()),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut map_sections_n: HashMap<String, u64> = HashMap::new();
        if let Some(msn) = j.get("mapSectionsN").and_then(|v| v.as_object()) {
            for (k, v) in msn {
                if let Some(val) = v.as_u64() {
                    map_sections_n.insert(k.clone(), val);
                }
            }
        }

        Ok(StarkInfo {
            stark_struct,
            n_stages: get_u64(j, "nStages"),
            n_constants: get_u64(j, "nConstants"),
            n_publics: get_u64(j, "nPublics"),
            custom_commits,
            cm_pols_map: parse_pol_map_array(j, "cmPolsMap"),
            const_pols_map: parse_pol_map_array(j, "constPolsMap"),
            challenges_map: parse_pol_map_array(j, "challengesMap"),
            airgroup_values_map: parse_pol_map_array(j, "airgroupValuesMap"),
            air_values_map: parse_pol_map_array(j, "airValuesMap"),
            proof_values_map: parse_pol_map_array(j, "proofValuesMap"),
            ev_map,
            opening_points,
            boundaries,
            q_deg: get_u64(j, "qDeg"),
            q_dim: get_u64(j, "qDim"),
            fri_exp_id: get_u64(j, "friExpId"),
            c_exp_id: get_u64(j, "cExpId"),
            map_sections_n,
        })
    }
}

fn parse_exp_code(v: &Value) -> Result<ExpCode> {
    let code: Vec<CodeOperation> = match v.get("code").and_then(|c| c.as_array()) {
        Some(arr) => arr.iter().map(parse_code_operation).collect::<Result<_>>()?,
        None => Vec::new(),
    };

    Ok(ExpCode {
        exp_id: get_u64(v, "expId"),
        stage: get_u64(v, "stage"),
        tmp_used: get_u64(v, "tmpUsed"),
        code,
        line: get_str(v, "line").to_string(),
        boundary: get_str(v, "boundary").to_string(),
        offset_min: get_u64(v, "offsetMin"),
        offset_max: get_u64(v, "offsetMax"),
        im_pol: get_u64(v, "imPol"),
    })
}

fn parse_hint_field_value(v: &Value) -> Result<HintFieldValue> {
    let value = get_goldilocks(v)?;
    Ok(HintFieldValue {
        op: get_str(v, "op").to_string(),
        id: get_u64(v, "id"),
        commit_id: get_u64(v, "commitId"),
        row_offset_index: get_u64(v, "rowOffsetIndex"),
        dim: get_u64(v, "dim"),
        value,
        string_value: get_str(v, "string").to_string(),
        airgroup_id: get_u64(v, "airgroupId"),
        pos: v
            .get("pos")
            .and_then(|p| p.as_array())
            .map(|arr| arr.iter().map(|v| v.as_u64().unwrap_or(0)).collect())
            .unwrap_or_default(),
    })
}

/// The elements of array `key` of `v`, each parsed by `parse`; none when `v` has no such array.
fn parse_array<T>(v: &Value, key: &str, parse: impl Fn(&Value) -> Result<T>) -> Result<Vec<T>> {
    match v.get(key).and_then(|a| a.as_array()) {
        Some(arr) => arr.iter().map(parse).collect(),
        None => Ok(Vec::new()),
    }
}

fn parse_hint_field(v: &Value) -> Result<HintField> {
    Ok(HintField { name: get_str(v, "name").to_string(), values: parse_array(v, "values", parse_hint_field_value)? })
}

fn parse_hint(v: &Value) -> Result<Hint> {
    let name = get_str(v, "name").to_string();
    let fields = parse_array(v, "fields", parse_hint_field).with_context(|| format!("hint {name}"))?;
    Ok(Hint { name, fields })
}

fn parse_hints(v: &Value, key: &str) -> Result<Vec<Hint>> {
    parse_array(v, key, parse_hint)
}

impl ExpressionsInfo {
    /// Parse from the expressionsInfo JSON.
    pub fn from_json(j: &Value) -> Result<Self> {
        let expressions_code = parse_array(j, "expressionsCode", parse_exp_code)?;
        let constraints = parse_array(j, "constraints", parse_exp_code)?;
        let hints_info = parse_hints(j, "hintsInfo")?;

        Ok(ExpressionsInfo { expressions_code, constraints, hints_info })
    }
}

impl VerifierInfo {
    /// Parse from the verifierInfo JSON.
    pub fn from_json(j: &Value) -> Result<Self> {
        let q_verifier = parse_exp_code(j.get("qVerifier").unwrap_or(&Value::Null))?;
        let query_verifier = parse_exp_code(j.get("queryVerifier").unwrap_or(&Value::Null))?;
        Ok(VerifierInfo { q_verifier, query_verifier })
    }
}

impl GlobalConstraintsInfo {
    /// Parse from global constraints JSON.
    pub fn from_json(j: &Value) -> Result<Self> {
        let constraints = parse_array(j, "constraints", parse_exp_code)?;
        let hints = parse_hints(j, "hints")?;

        Ok(GlobalConstraintsInfo { constraints, hints })
    }
}

// ---------------------------------------------------------------------------
// Direct conversions from generate_pil_code types (avoids JSON round-trip)
//
// The passes keep numbers as decimal strings; here, where the STARK writers take them, they are
// narrowed to the u64 of a Goldilocks element, and one that is not is an error (goldilocks_u64).
// ---------------------------------------------------------------------------

fn code_ref_to_code_type(r: &crate::types::output::CodeRef) -> Result<CodeType> {
    let value = match r.value {
        Some(ref v) => goldilocks_u64(v)?,
        None => 0,
    };
    Ok(CodeType {
        op_type: OpType::parse(&r.ref_type)?,
        id: r.id as u64,
        prime: r.prime.unwrap_or(0),
        dim: r.dim as u64,
        value,
        commit_id: r.commit_id.unwrap_or(0) as u64,
        boundary_id: r.boundary_id.unwrap_or(0) as u64,
        airgroup_id: r.airgroup_id.unwrap_or(0) as u64,
    })
}

fn code_entry_to_operation(e: &crate::types::output::CodeEntry) -> Result<CodeOperation> {
    Ok(CodeOperation {
        op: e.op.clone(),
        dest: code_ref_to_code_type(&e.dest)?,
        src: e.src.iter().map(code_ref_to_code_type).collect::<Result<_>>()?,
    })
}

fn code_to_operations(code: &[crate::types::output::CodeEntry]) -> Result<Vec<CodeOperation>> {
    code.iter().map(code_entry_to_operation).collect()
}

fn expression_entry_to_exp_code(e: &crate::pil::gen_code::ExpressionCodeEntry) -> Result<ExpCode> {
    Ok(ExpCode {
        exp_id: e.exp_id as u64,
        stage: e.stage as u64,
        tmp_used: e.tmp_used as u64,
        code: code_to_operations(&e.code).with_context(|| format!("the code of expression {}", e.exp_id))?,
        line: e.line.clone(),
        boundary: String::new(),
        offset_min: 0,
        offset_max: 0,
        im_pol: 0,
    })
}

fn constraint_entry_to_exp_code(c: &crate::pil::gen_code::ConstraintCodeEntry) -> Result<ExpCode> {
    let line = c.line.clone().unwrap_or_default();
    Ok(ExpCode {
        exp_id: 0,
        stage: c.stage as u64,
        tmp_used: c.tmp_used as u64,
        code: code_to_operations(&c.code).with_context(|| format!("the code of constraint {line}"))?,
        line,
        boundary: c.boundary.clone(),
        offset_min: c.offset_min.unwrap_or(0) as u64,
        offset_max: c.offset_max.unwrap_or(0) as u64,
        im_pol: c.im_pol as u64,
    })
}

fn processed_hint_field_to_value(v: &crate::pil::gen_code::ProcessedHintField) -> Result<HintFieldValue> {
    let (value, string_value) = match v.value {
        Some(ref s) if v.op == "string" => (0u64, s.clone()),
        Some(ref s) => (goldilocks_u64(s)?, String::new()),
        None => (0, String::new()),
    };
    Ok(HintFieldValue {
        op: v.op.clone(),
        id: v.id.unwrap_or(0) as u64,
        commit_id: v.commit_id.unwrap_or(0) as u64,
        row_offset_index: v.row_offset_index.unwrap_or(0) as u64,
        dim: v.dim.unwrap_or(0) as u64,
        value,
        string_value,
        airgroup_id: v.airgroup_id.unwrap_or(0) as u64,
        pos: v.pos.iter().map(|&p| p as u64).collect(),
    })
}

/// Fails on a number that is not a Goldilocks element, and on an operand of an unknown type.
impl TryFrom<&crate::pil::gen_code::ExpressionsInfo> for ExpressionsInfo {
    type Error = anyhow::Error;

    fn try_from(ei: &crate::pil::gen_code::ExpressionsInfo) -> Result<Self> {
        let hints_info = ei
            .hints_info
            .iter()
            .map(|h| {
                let fields = h
                    .fields
                    .iter()
                    .map(|f| {
                        let values = f.values.iter().map(processed_hint_field_to_value).collect::<Result<_>>()?;
                        Ok(HintField { name: f.name.clone(), values })
                    })
                    .collect::<Result<_>>()
                    .with_context(|| format!("hint {}", h.name))?;
                Ok(Hint { name: h.name.clone(), fields })
            })
            .collect::<Result<_>>()?;
        Ok(ExpressionsInfo {
            expressions_code: ei.expressions_code.iter().map(expression_entry_to_exp_code).collect::<Result<_>>()?,
            constraints: ei.constraints.iter().map(constraint_entry_to_exp_code).collect::<Result<_>>()?,
            hints_info,
        })
    }
}

/// Fails on the verifier code of a backend that does not open with FRI: the STARK verifier needs
/// the `queryVerifier`.
impl TryFrom<&crate::pil::gen_code::VerifierInfo> for VerifierInfo {
    type Error = anyhow::Error;

    fn try_from(vi: &crate::pil::gen_code::VerifierInfo) -> Result<Self> {
        let query_verifier = vi
            .query_verifier
            .as_ref()
            .ok_or_else(|| anyhow!("the STARK verifier needs a queryVerifier, which only the FRI opening has"))?;
        Ok(VerifierInfo {
            q_verifier: expression_entry_to_exp_code(&vi.q_verifier).context("the qVerifier")?,
            query_verifier: expression_entry_to_exp_code(query_verifier).context("the queryVerifier")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pil::gen_code::{self, ExpressionCodeEntry, ProcessedHint, ProcessedHintField, ProcessedHintFieldEntry};
    use crate::types::output::{CodeEntry, CodeRef};

    /// BN128's `r − 1`: a constant of a pilout over BN128, which `parse().unwrap_or(0)` wrote as 0.
    const R_MINUS_ONE: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495616";

    fn code_ref(ref_type: &str, value: Option<&str>) -> CodeRef {
        CodeRef {
            ref_type: ref_type.to_string(),
            id: 0,
            dim: 1,
            prime: None,
            value: value.map(str::to_string),
            stage: None,
            stage_id: None,
            commit_id: None,
            opening: None,
            boundary_id: None,
            airgroup_id: None,
            exp_id: None,
        }
    }

    /// The passes' code of expression 3, `tmp0 = number + tmp0`, and a hint holding `hint_number`.
    fn passes_info(number: &str, hint_number: &str) -> gen_code::ExpressionsInfo {
        let add = CodeEntry {
            op: "add".to_string(),
            dest: code_ref("tmp", None),
            src: vec![code_ref("number", Some(number)), code_ref("tmp", None)],
        };
        let field = ProcessedHintField {
            op: "number".to_string(),
            id: None,
            dim: Some(1),
            pos: Vec::new(),
            stage: Some(0),
            stage_id: None,
            value: Some(hint_number.to_string()),
            row_offset: None,
            row_offset_index: None,
            commit_id: None,
            airgroup_id: None,
        };
        gen_code::ExpressionsInfo {
            hints_info: vec![ProcessedHint {
                name: "h".to_string(),
                fields: vec![ProcessedHintFieldEntry { name: "f".to_string(), values: vec![field] }],
            }],
            expressions_code: vec![ExpressionCodeEntry {
                tmp_used: 1,
                code: vec![add],
                exp_id: 3,
                stage: 1,
                dest: None,
                line: String::new(),
            }],
            constraints: Vec::new(),
        }
    }

    fn not_goldilocks(value: &str) -> String {
        format!("the number {value} is not a Goldilocks element (an integer below p = {GOLDILOCKS_MODULUS})")
    }

    #[test]
    fn numbers_below_p_are_narrowed_as_is() {
        let ei = ExpressionsInfo::try_from(&passes_info("18446744069414584320", "7")).unwrap();
        assert_eq!(ei.expressions_code[0].code[0].src[0].value, GOLDILOCKS_MODULUS - 1);
        assert_eq!(ei.hints_info[0].fields[0].values[0].value, 7);
    }

    /// The STARK writers refuse a number that is not a Goldilocks element, rather than truncate it
    /// to 0 (wider than 64 bits) or let it through non-canonical (between p and 2^64).
    #[test]
    fn numbers_that_are_not_goldilocks_elements_are_refused() {
        for wide in [R_MINUS_ONE, "18446744073709551616", "18446744073709551615", "18446744069414584321"] {
            let err = ExpressionsInfo::try_from(&passes_info(wide, "0")).unwrap_err();
            assert_eq!(format!("{err:#}"), format!("the code of expression 3: {}", not_goldilocks(wide)));

            let err = ExpressionsInfo::try_from(&passes_info("0", wide)).unwrap_err();
            assert_eq!(format!("{err:#}"), format!("hint h: {}", not_goldilocks(wide)));
        }
    }

    /// End to end: the passes keep a constant wider than Goldilocks as it is, and the STARK writers
    /// refuse it where they narrow it, instead of writing 0.
    #[test]
    fn a_wide_constant_of_the_pilout_is_refused_by_the_stark_writers() {
        use pil2_pilout::pilout::{self as pb, constraint, expression, operand, SymbolType};

        // 2^64 + 5, big-endian.
        let wide = vec![0x01, 0, 0, 0, 0, 0, 0, 0, 0x05];
        let operand = |o| Some(pb::Operand { operand: Some(o) });
        let a = operand(operand::Operand::WitnessCol(operand::WitnessCol { stage: 1, col_idx: 0, row_offset: 0 }));
        let c = operand(operand::Operand::Constant(operand::Constant { value: wide }));
        let air = pb::Air {
            name: Some("Wide".to_string()),
            num_rows: Some(16),
            stage_widths: vec![1],
            expressions: vec![pb::Expression {
                operation: Some(expression::Operation::Sub(expression::Sub { lhs: a, rhs: c })),
            }],
            constraints: vec![pb::Constraint {
                constraint: Some(constraint::Constraint::EveryRow(constraint::EveryRow {
                    expression_idx: Some(operand::Expression { idx: 0 }),
                    debug_line: None,
                })),
            }],
            ..Default::default()
        };
        let pilout = pb::PilOut {
            air_groups: vec![pb::AirGroup { airs: vec![air], ..Default::default() }],
            num_challenges: vec![0],
            symbols: vec![pb::Symbol {
                name: "Wide.a".to_string(),
                air_group_id: Some(0),
                air_id: Some(0),
                r#type: SymbolType::WitnessCol as i32,
                stage: Some(1),
                ..Default::default()
            }],
            ..Default::default()
        };

        let result = pil_info::run(&pilout, 0, 0, &pil_info::PilInfoCfg::goldilocks(1), &Default::default()).unwrap();
        let err = ExpressionsInfo::try_from(&result.pil_code.expressions_info).unwrap_err();
        assert!(format!("{err:#}").ends_with(&not_goldilocks("18446744073709551621")), "{err:#}");
    }

    /// The same refusal when the code is read back from its JSON (the recursive setups).
    #[test]
    fn numbers_of_the_json_that_are_not_goldilocks_elements_are_refused() {
        let json = |value: &str| {
            serde_json::json!({
                "expressionsCode": [{
                    "code": [{
                        "op": "add",
                        "dest": {"type": "tmp", "id": 0, "dim": 1},
                        "src": [{"type": "number", "value": value, "dim": 1}, {"type": "tmp", "id": 0, "dim": 1}],
                    }],
                }],
            })
        };
        let ei = ExpressionsInfo::from_json(&json("42")).unwrap();
        assert_eq!(ei.expressions_code[0].code[0].src[0].value, 42);

        let err = ExpressionsInfo::from_json(&json(R_MINUS_ONE)).unwrap_err();
        assert_eq!(err.to_string(), not_goldilocks(R_MINUS_ONE));

        let hints = serde_json::json!({"constraints": [], "hints": [{
            "name": "h", "fields": [{"name": "f", "values": [{"op": "number", "value": R_MINUS_ONE, "pos": []}]}],
        }]});
        let err = GlobalConstraintsInfo::from_json(&hints).unwrap_err();
        assert_eq!(format!("{err:#}"), format!("hint h: {}", not_goldilocks(R_MINUS_ONE)));
    }
}
