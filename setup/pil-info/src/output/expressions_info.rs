//! JSON for `expressionsinfo.json` and `verifierinfo.json`, and the code and hint serializers
//! that `globalConstraints.json` shares.

use serde_json::json;

use crate::pil::gen_code::ProcessedHintField;
use crate::types::output::{CodeEntry, CodeRef};

/// Build the expressionsinfo JSON structure.
pub fn build_expressions_info_json(info: &crate::pil::gen_code::ExpressionsInfo) -> serde_json::Value {
    let expressions_code: Vec<serde_json::Value> = info
        .expressions_code
        .iter()
        .map(|e| {
            let mut obj = serde_json::Map::new();
            obj.insert("tmpUsed".to_string(), json!(e.tmp_used));
            obj.insert("code".to_string(), code_entries_to_json(&e.code));
            obj.insert("expId".to_string(), json!(e.exp_id));
            obj.insert("stage".to_string(), json!(e.stage));
            if let Some(ref dest) = e.dest {
                obj.insert(
                    "dest".to_string(),
                    json!({
                        "op": dest.op,
                        "stage": dest.stage,
                        "stageId": dest.stage_id,
                        "id": dest.id,
                    }),
                );
            }
            obj.insert("line".to_string(), json!(e.line));
            serde_json::Value::Object(obj)
        })
        .collect();

    let constraints: Vec<serde_json::Value> = info
        .constraints
        .iter()
        .map(|c| {
            let mut obj = serde_json::Map::new();
            obj.insert("tmpUsed".to_string(), json!(c.tmp_used));
            obj.insert("code".to_string(), code_entries_to_json(&c.code));
            obj.insert("boundary".to_string(), json!(c.boundary));
            if let Some(ref line) = c.line {
                obj.insert("line".to_string(), json!(line));
            }
            obj.insert("imPol".to_string(), json!(c.im_pol));
            obj.insert("stage".to_string(), json!(c.stage));
            if let Some(omin) = c.offset_min {
                obj.insert("offsetMin".to_string(), json!(omin));
            }
            if let Some(omax) = c.offset_max {
                obj.insert("offsetMax".to_string(), json!(omax));
            }
            serde_json::Value::Object(obj)
        })
        .collect();

    let hints_info: Vec<serde_json::Value> = info
        .hints_info
        .iter()
        .map(|h| {
            json!({
                "name": h.name,
                "fields": h.fields.iter().map(|f| {
                    json!({
                        "name": f.name,
                        "values": f.values.iter().map(|v| {
                            hint_value_to_json(v)
                        }).collect::<Vec<_>>(),
                    })
                }).collect::<Vec<serde_json::Value>>(),
            })
        })
        .collect();

    // Build with explicit ordering matching golden: hintsInfo, expressionsCode, constraints
    let mut result = serde_json::Map::new();
    result.insert("hintsInfo".to_string(), serde_json::Value::Array(hints_info));
    result.insert("expressionsCode".to_string(), serde_json::Value::Array(expressions_code));
    result.insert("constraints".to_string(), serde_json::Value::Array(constraints));
    serde_json::Value::Object(result)
}

/// Build the verifierinfo JSON structure.
pub fn build_verifier_info_json(info: &crate::pil::gen_code::VerifierInfo) -> serde_json::Value {
    // qVerifier: {tmpUsed, code, line: ""}
    let mut qv = serde_json::Map::new();
    qv.insert("tmpUsed".to_string(), json!(info.q_verifier.tmp_used));
    qv.insert("code".to_string(), code_entries_to_json(&info.q_verifier.code));
    qv.insert("line".to_string(), json!(""));

    let mut result = serde_json::Map::new();
    result.insert("qVerifier".to_string(), serde_json::Value::Object(qv));

    // queryVerifier: {tmpUsed, code, expId, stage, line}, for the FRI opening only
    if let Some(query_verifier) = &info.query_verifier {
        let mut qr = serde_json::Map::new();
        qr.insert("tmpUsed".to_string(), json!(query_verifier.tmp_used));
        qr.insert("code".to_string(), code_entries_to_json(&query_verifier.code));
        qr.insert("expId".to_string(), json!(query_verifier.exp_id));
        qr.insert("stage".to_string(), json!(query_verifier.stage));
        qr.insert("line".to_string(), json!(query_verifier.line));
        result.insert("queryVerifier".to_string(), serde_json::Value::Object(qr));
    }

    serde_json::Value::Object(result)
}

/// Serialize a hint field value to JSON matching golden field order per op type.
///
/// Field order per op type (matching JS object spread behavior):
///   string:         op, string, pos
///   number:         op, value, pos
///   tmp:            op, id, dim, pos
///   cm/custom/const: op, id, stageId, rowOffset, stage, dim, [commitId,] rowOffsetIndex, pos
///   challenge/public/airgroupvalue/airvalue/proofvalue:
///                   op, id, dim, stage, [stageId,] [airgroupId,] pos
pub(crate) fn hint_value_to_json(v: &ProcessedHintField) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("op".to_string(), json!(v.op));
    match v.op.as_str() {
        "string" => {
            if let Some(ref val) = v.value {
                obj.insert("string".to_string(), json!(val));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
        "number" => {
            if let Some(ref val) = v.value {
                obj.insert("value".to_string(), json!(val));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
        "tmp" => {
            if let Some(id) = v.id {
                obj.insert("id".to_string(), json!(id));
            }
            if let Some(dim) = v.dim {
                obj.insert("dim".to_string(), json!(dim));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
        "cm" | "custom" | "const" => {
            if let Some(id) = v.id {
                obj.insert("id".to_string(), json!(id));
            }
            if let Some(sid) = v.stage_id {
                obj.insert("stageId".to_string(), json!(sid));
            }
            if let Some(ro) = v.row_offset {
                obj.insert("rowOffset".to_string(), json!(ro));
            }
            if let Some(stage) = v.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
            if let Some(dim) = v.dim {
                obj.insert("dim".to_string(), json!(dim));
            }
            if let Some(cid) = v.commit_id {
                obj.insert("commitId".to_string(), json!(cid));
            }
            if let Some(roi) = v.row_offset_index {
                obj.insert("rowOffsetIndex".to_string(), json!(roi));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
        "challenge" => {
            if let Some(stage) = v.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
            if let Some(sid) = v.stage_id {
                obj.insert("stageId".to_string(), json!(sid));
            }
            if let Some(id) = v.id {
                obj.insert("id".to_string(), json!(id));
            }
            if let Some(dim) = v.dim {
                obj.insert("dim".to_string(), json!(dim));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
        "airgroupvalue" => {
            if let Some(id) = v.id {
                obj.insert("id".to_string(), json!(id));
            }
            if let Some(agid) = v.airgroup_id {
                obj.insert("airgroupId".to_string(), json!(agid));
            }
            if let Some(dim) = v.dim {
                obj.insert("dim".to_string(), json!(dim));
            }
            if let Some(stage) = v.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
        "public" => {
            if let Some(id) = v.id {
                obj.insert("id".to_string(), json!(id));
            }
            if let Some(stage) = v.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
        _ => {
            // airvalue, proofvalue
            if let Some(id) = v.id {
                obj.insert("id".to_string(), json!(id));
            }
            if let Some(stage) = v.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
            if let Some(dim) = v.dim {
                obj.insert("dim".to_string(), json!(dim));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
    }
    serde_json::Value::Object(obj)
}

pub fn code_entries_to_json(entries: &[CodeEntry]) -> serde_json::Value {
    let arr: Vec<serde_json::Value> = entries
        .iter()
        .map(|e| {
            json!({
                "op": e.op,
                "dest": code_ref_to_json(&e.dest),
                "src": e.src.iter().map(code_ref_to_json).collect::<Vec<_>>(),
            })
        })
        .collect();
    serde_json::Value::Array(arr)
}

/// Serialize a code ref to JSON matching the golden field order per type.
///
/// Each type has its own property order matching the JS object construction
/// order in codegen.js. The patterns are:
///   tmp (from binary op):  type, id, dim
///   tmp (from exp ref):    type, [expId,] id, prime, dim
///   cm/const/custom:       type, id, prime, dim, [commitId]
///   eval:                  type, id, dim
///   challenge:             type, id, stageId, dim, stage
///   public:                type, id, dim
///   proofvalue:            type, id, stage, dim
///   number:                type, value, dim
///   airgroupvalue/airvalue: type, id, stage, dim, [airgroupId]
///   xDivXSubXi:            type, id, opening, dim
///   Zi:                    type, boundaryId, dim
pub fn code_ref_to_json(r: &CodeRef) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".to_string(), json!(r.ref_type));
    match r.ref_type.as_str() {
        "tmp" => {
            // Three tmp patterns based on JS object construction order:
            //   1. Binary op dest:       type, id, dim (no prime, no expId)
            //   2. pilCodeGen wrapper:    type, prime, id, dim (prime present, no expId)
            //   3. evalExp exp ref:       type, expId, id, prime, dim
            if let Some(eid) = r.exp_id {
                obj.insert("expId".to_string(), json!(eid));
                obj.insert("id".to_string(), json!(r.id));
                if let Some(prime) = r.prime {
                    obj.insert("prime".to_string(), json!(prime));
                }
            } else if let Some(prime) = r.prime {
                obj.insert("prime".to_string(), json!(prime));
                obj.insert("id".to_string(), json!(r.id));
            } else {
                obj.insert("id".to_string(), json!(r.id));
            }
            obj.insert("dim".to_string(), json!(r.dim));
        }
        "cm" | "const" | "custom" => {
            // For ImPol cm refs, expId comes before id (matching JS object
            // property insertion order from evalExp where r starts as
            // {type: "exp", expId, id, ...} then fixCommitPol changes type/id)
            if let Some(eid) = r.exp_id {
                obj.insert("expId".to_string(), json!(eid));
            }
            obj.insert("id".to_string(), json!(r.id));
            obj.insert("prime".to_string(), json!(r.prime.unwrap_or(0)));
            obj.insert("dim".to_string(), json!(r.dim));
            if let Some(cid) = r.commit_id {
                obj.insert("commitId".to_string(), json!(cid));
            }
        }
        "number" => {
            if let Some(ref value) = r.value {
                obj.insert("value".to_string(), json!(value));
            }
            obj.insert("dim".to_string(), json!(r.dim));
        }
        "challenge" => {
            obj.insert("id".to_string(), json!(r.id));
            if let Some(sid) = r.stage_id {
                obj.insert("stageId".to_string(), json!(sid));
            }
            obj.insert("dim".to_string(), json!(r.dim));
            if let Some(stage) = r.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
        }
        "eval" => {
            if let Some(eid) = r.exp_id {
                obj.insert("expId".to_string(), json!(eid));
            }
            obj.insert("id".to_string(), json!(r.id));
            obj.insert("dim".to_string(), json!(r.dim));
            if let Some(cid) = r.commit_id {
                obj.insert("commitId".to_string(), json!(cid));
            }
        }
        "public" => {
            obj.insert("id".to_string(), json!(r.id));
            obj.insert("dim".to_string(), json!(r.dim));
        }
        "proofvalue" => {
            obj.insert("id".to_string(), json!(r.id));
            if let Some(stage) = r.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
            obj.insert("dim".to_string(), json!(r.dim));
        }
        "airgroupvalue" | "airvalue" => {
            obj.insert("id".to_string(), json!(r.id));
            if let Some(stage) = r.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
            obj.insert("dim".to_string(), json!(r.dim));
            if let Some(agid) = r.airgroup_id {
                obj.insert("airgroupId".to_string(), json!(agid));
            }
        }
        "xDivXSubXi" => {
            obj.insert("id".to_string(), json!(r.id));
            if let Some(opening) = r.opening {
                obj.insert("opening".to_string(), json!(opening));
            }
            obj.insert("dim".to_string(), json!(r.dim));
        }
        "Zi" => {
            if let Some(bid) = r.boundary_id {
                obj.insert("boundaryId".to_string(), json!(bid));
            }
            obj.insert("dim".to_string(), json!(r.dim));
        }
        _ => {
            obj.insert("id".to_string(), json!(r.id));
            obj.insert("dim".to_string(), json!(r.dim));
            if let Some(prime) = r.prime {
                obj.insert("prime".to_string(), json!(prime));
            }
            if let Some(ref value) = r.value {
                obj.insert("value".to_string(), json!(value));
            }
            if let Some(stage) = r.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
        }
    }
    serde_json::Value::Object(obj)
}
