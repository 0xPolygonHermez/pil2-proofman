use anyhow::{bail, Result};
use serde_json::Value;
use std::collections::HashMap;

use crate::types::stark_struct::{LowDegreeTest, StarkStruct};

/// Operand type enum mirroring the C++ opType.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OpType {
    Const,
    Cm,
    #[default]
    Tmp,
    Public,
    Airgroupvalue,
    Challenge,
    Number,
    StringVal,
    Airvalue,
    Proofvalue,
    Custom,
    X,
    Zi,
    Eval,
    XDivXSubXi,
    Q,
    F,
}

impl OpType {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "const" => Ok(OpType::Const),
            "cm" => Ok(OpType::Cm),
            "tmp" => Ok(OpType::Tmp),
            "public" => Ok(OpType::Public),
            "airgroupvalue" => Ok(OpType::Airgroupvalue),
            "challenge" => Ok(OpType::Challenge),
            "number" => Ok(OpType::Number),
            "string" => Ok(OpType::StringVal),
            "airvalue" => Ok(OpType::Airvalue),
            "proofvalue" => Ok(OpType::Proofvalue),
            "custom" => Ok(OpType::Custom),
            "x" => Ok(OpType::X),
            "Zi" => Ok(OpType::Zi),
            "eval" => Ok(OpType::Eval),
            "xDivXSubXi" => Ok(OpType::XDivXSubXi),
            "q" => Ok(OpType::Q),
            "f" => Ok(OpType::F),
            _ => bail!("Unknown opType string: {}", s),
        }
    }

    pub fn to_str(self) -> &'static str {
        match self {
            OpType::Const => "const",
            OpType::Cm => "cm",
            OpType::Tmp => "tmp",
            OpType::Public => "public",
            OpType::Airgroupvalue => "airgroupvalue",
            OpType::Challenge => "challenge",
            OpType::Number => "number",
            OpType::StringVal => "string",
            OpType::Airvalue => "airvalue",
            OpType::Proofvalue => "proofvalue",
            OpType::Custom => "custom",
            OpType::X => "x",
            OpType::Zi => "Zi",
            OpType::Eval => "eval",
            OpType::XDivXSubXi => "xDivXSubXi",
            OpType::Q => "q",
            OpType::F => "f",
        }
    }
}

/// A single operand reference within a code operation.
#[derive(Debug, Clone, Default)]
pub struct CodeType {
    pub op_type: OpType,
    pub id: u64,
    pub prime: i64,
    pub dim: u64,
    pub value: u64,
    pub commit_id: u64,
    pub boundary_id: u64,
    pub airgroup_id: u64,
}

/// A single code instruction: an operation with dest and sources.
#[derive(Debug, Clone)]
pub struct CodeOperation {
    pub op: String,
    pub dest: CodeType,
    pub src: Vec<CodeType>,
}

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

fn parse_code_type(v: &Value) -> Result<CodeType> {
    let type_str = get_str(v, "type");
    let op_type = OpType::parse(type_str)?;
    let value = if let Some(val) = v.get("value") {
        if let Some(s) = val.as_str() {
            if s.starts_with("0x") || s.starts_with("0X") {
                u64::from_str_radix(&s[2..], 16).unwrap_or(0)
            } else {
                s.parse::<u64>().unwrap_or(0)
            }
        } else {
            val.as_u64().unwrap_or(0)
        }
    } else {
        0
    };
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
    let src: Vec<CodeType> = v
        .get("src")
        .and_then(|s| s.as_array())
        .map(|arr| arr.iter().map(|s| parse_code_type(s).unwrap_or_default()).collect())
        .unwrap_or_default();
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
        let low_degree_test: LowDegreeTest = serde_json::from_value(ss.clone()).unwrap_or_default();

        let stark_struct = StarkStruct {
            n_bits: get_u64(ss, "nBits") as usize,
            n_bits_ext: get_u64(ss, "nBitsExt") as usize,
            hash_commits: get_bool(ss, "hashCommits"),
            last_level_verification: get_u64(ss, "lastLevelVerification") as usize,
            merkle_tree_arity: get_u64(ss, "merkleTreeArity") as usize,
            transcript_arity: get_u64(ss, "transcriptArity") as usize,
            merkle_tree_custom: get_bool(ss, "merkleTreeCustom"),
            verification_hash_type: get_str(ss, "verificationHashType").to_string(),
            low_degree_test,
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
    let code: Vec<CodeOperation> = v
        .get("code")
        .and_then(|c| c.as_array())
        .map(|arr| arr.iter().map(|c| parse_code_operation(c).unwrap()).collect())
        .unwrap_or_default();

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

fn parse_hint_field_value(v: &Value) -> HintFieldValue {
    let value = if let Some(val) = v.get("value") {
        if let Some(s) = val.as_str() {
            s.parse::<u64>().unwrap_or(0)
        } else {
            val.as_u64().unwrap_or(0)
        }
    } else {
        0
    };
    HintFieldValue {
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
    }
}

fn parse_hint_field(v: &Value) -> HintField {
    HintField {
        name: get_str(v, "name").to_string(),
        values: v
            .get("values")
            .and_then(|a| a.as_array())
            .map(|arr| arr.iter().map(parse_hint_field_value).collect())
            .unwrap_or_default(),
    }
}

fn parse_hint(v: &Value) -> Hint {
    Hint {
        name: get_str(v, "name").to_string(),
        fields: v
            .get("fields")
            .and_then(|a| a.as_array())
            .map(|arr| arr.iter().map(parse_hint_field).collect())
            .unwrap_or_default(),
    }
}

fn parse_hints(v: &Value, key: &str) -> Vec<Hint> {
    v.get(key).and_then(|a| a.as_array()).map(|arr| arr.iter().map(parse_hint).collect()).unwrap_or_default()
}

impl ExpressionsInfo {
    /// Parse from the expressionsInfo JSON.
    pub fn from_json(j: &Value) -> Result<Self> {
        let expressions_code: Vec<ExpCode> = j
            .get("expressionsCode")
            .and_then(|a| a.as_array())
            .map(|arr| arr.iter().map(|e| parse_exp_code(e).unwrap()).collect())
            .unwrap_or_default();

        let constraints: Vec<ExpCode> = j
            .get("constraints")
            .and_then(|a| a.as_array())
            .map(|arr| arr.iter().map(|c| parse_exp_code(c).unwrap()).collect())
            .unwrap_or_default();

        let hints_info = parse_hints(j, "hintsInfo");

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
        let constraints: Vec<ExpCode> = j
            .get("constraints")
            .and_then(|a| a.as_array())
            .map(|arr| arr.iter().map(|c| parse_exp_code(c).unwrap()).collect())
            .unwrap_or_default();

        let hints = parse_hints(j, "hints");

        Ok(GlobalConstraintsInfo { constraints, hints })
    }
}

// ---------------------------------------------------------------------------
// Direct From conversions from generate_pil_code types (avoids JSON round-trip)
// ---------------------------------------------------------------------------

fn code_ref_to_code_type(r: &crate::types::output::CodeRef) -> CodeType {
    let value = if let Some(ref v) = r.value {
        if v.starts_with("0x") || v.starts_with("0X") {
            u64::from_str_radix(&v[2..], 16).unwrap_or(0)
        } else {
            v.parse::<u64>().unwrap_or(0)
        }
    } else {
        0
    };
    CodeType {
        op_type: OpType::parse(&r.ref_type).unwrap_or_default(),
        id: r.id as u64,
        prime: r.prime.unwrap_or(0),
        dim: r.dim as u64,
        value,
        commit_id: r.commit_id.unwrap_or(0) as u64,
        boundary_id: r.boundary_id.unwrap_or(0) as u64,
        airgroup_id: r.airgroup_id.unwrap_or(0) as u64,
    }
}

fn code_entry_to_operation(e: &crate::types::output::CodeEntry) -> CodeOperation {
    CodeOperation {
        op: e.op.clone(),
        dest: code_ref_to_code_type(&e.dest),
        src: e.src.iter().map(code_ref_to_code_type).collect(),
    }
}

fn expression_entry_to_exp_code(e: &crate::pil::gen_code::ExpressionCodeEntry) -> ExpCode {
    ExpCode {
        exp_id: e.exp_id as u64,
        stage: e.stage as u64,
        tmp_used: e.tmp_used as u64,
        code: e.code.iter().map(code_entry_to_operation).collect(),
        line: e.line.clone(),
        boundary: String::new(),
        offset_min: 0,
        offset_max: 0,
        im_pol: 0,
    }
}

fn constraint_entry_to_exp_code(c: &crate::pil::gen_code::ConstraintCodeEntry) -> ExpCode {
    ExpCode {
        exp_id: 0,
        stage: c.stage as u64,
        tmp_used: c.tmp_used as u64,
        code: c.code.iter().map(code_entry_to_operation).collect(),
        line: c.line.clone().unwrap_or_default(),
        boundary: c.boundary.clone(),
        offset_min: c.offset_min.unwrap_or(0) as u64,
        offset_max: c.offset_max.unwrap_or(0) as u64,
        im_pol: c.im_pol as u64,
    }
}

fn processed_hint_field_to_value(v: &crate::pil::gen_code::ProcessedHintField) -> HintFieldValue {
    let (value, string_value) = if let Some(ref s) = v.value {
        if v.op == "string" {
            (0u64, s.clone())
        } else if s.starts_with("0x") || s.starts_with("0X") {
            (u64::from_str_radix(&s[2..], 16).unwrap_or(0), String::new())
        } else {
            (s.parse::<u64>().unwrap_or(0), String::new())
        }
    } else {
        (0, String::new())
    };
    HintFieldValue {
        op: v.op.clone(),
        id: v.id.unwrap_or(0) as u64,
        commit_id: v.commit_id.unwrap_or(0) as u64,
        row_offset_index: v.row_offset_index.unwrap_or(0) as u64,
        dim: v.dim.unwrap_or(0) as u64,
        value,
        string_value,
        airgroup_id: v.airgroup_id.unwrap_or(0) as u64,
        pos: v.pos.iter().map(|&p| p as u64).collect(),
    }
}

impl From<&crate::pil::gen_code::ExpressionsInfo> for ExpressionsInfo {
    fn from(ei: &crate::pil::gen_code::ExpressionsInfo) -> Self {
        ExpressionsInfo {
            expressions_code: ei.expressions_code.iter().map(expression_entry_to_exp_code).collect(),
            constraints: ei.constraints.iter().map(constraint_entry_to_exp_code).collect(),
            hints_info: ei
                .hints_info
                .iter()
                .map(|h| Hint {
                    name: h.name.clone(),
                    fields: h
                        .fields
                        .iter()
                        .map(|f| HintField {
                            name: f.name.clone(),
                            values: f.values.iter().map(processed_hint_field_to_value).collect(),
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

impl From<&crate::pil::gen_code::VerifierInfo> for VerifierInfo {
    fn from(vi: &crate::pil::gen_code::VerifierInfo) -> Self {
        VerifierInfo {
            q_verifier: expression_entry_to_exp_code(&vi.q_verifier),
            query_verifier: expression_entry_to_exp_code(&vi.query_verifier),
        }
    }
}
