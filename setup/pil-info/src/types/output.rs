//! Code blocks as the passes emit them, and as `expressionsinfo.json`, `verifierinfo.json` and
//! `globalConstraints.json` hold them.

use serde::{Deserialize, Serialize};

/// Code block used in expressionsinfo.json and verifierinfo.json
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeOutput {
    pub tmp_used: usize,
    pub code: Vec<CodeEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEntry {
    pub op: String,
    pub dest: CodeRef,
    pub src: Vec<CodeRef>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CodeRef {
    #[serde(rename = "type")]
    pub ref_type: String,
    pub id: usize,
    pub dim: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prime: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit_id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opening: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boundary_id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub airgroup_id: Option<usize>,
    /// Original expression id, preserved when an `exp` ref is converted to
    /// `tmp` via fixExpression (matches JS `expId` property).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exp_id: Option<usize>,
}

impl serde::Serialize for CodeRef {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(None)?;
        map.serialize_entry("type", &self.ref_type)?;
        match self.ref_type.as_str() {
            "tmp" => {
                if let Some(eid) = self.exp_id {
                    // Pattern 3: expId before id
                    map.serialize_entry("expId", &eid)?;
                    map.serialize_entry("id", &self.id)?;
                    if let Some(prime) = self.prime {
                        map.serialize_entry("prime", &prime)?;
                    }
                } else if let Some(prime) = self.prime {
                    // Pattern 2: prime before id
                    map.serialize_entry("prime", &prime)?;
                    map.serialize_entry("id", &self.id)?;
                } else {
                    // Pattern 1: just id
                    map.serialize_entry("id", &self.id)?;
                }
                map.serialize_entry("dim", &self.dim)?;
            }
            "cm" | "const" | "custom" => {
                if let Some(eid) = self.exp_id {
                    map.serialize_entry("expId", &eid)?;
                }
                map.serialize_entry("id", &self.id)?;
                map.serialize_entry("prime", &self.prime.unwrap_or(0))?;
                map.serialize_entry("dim", &self.dim)?;
                if let Some(cid) = self.commit_id {
                    map.serialize_entry("commitId", &cid)?;
                }
            }
            "number" => {
                if let Some(ref value) = self.value {
                    map.serialize_entry("value", value)?;
                }
                map.serialize_entry("dim", &self.dim)?;
            }
            "challenge" => {
                map.serialize_entry("id", &self.id)?;
                if let Some(sid) = self.stage_id {
                    map.serialize_entry("stageId", &sid)?;
                }
                map.serialize_entry("dim", &self.dim)?;
                if let Some(stage) = self.stage {
                    map.serialize_entry("stage", &stage)?;
                }
            }
            "eval" => {
                if let Some(eid) = self.exp_id {
                    map.serialize_entry("expId", &eid)?;
                }
                map.serialize_entry("id", &self.id)?;
                map.serialize_entry("dim", &self.dim)?;
                if let Some(cid) = self.commit_id {
                    map.serialize_entry("commitId", &cid)?;
                }
            }
            "public" => {
                map.serialize_entry("id", &self.id)?;
                map.serialize_entry("dim", &self.dim)?;
            }
            "proofvalue" => {
                map.serialize_entry("id", &self.id)?;
                if let Some(stage) = self.stage {
                    map.serialize_entry("stage", &stage)?;
                }
                map.serialize_entry("dim", &self.dim)?;
            }
            "airgroupvalue" | "airvalue" => {
                map.serialize_entry("id", &self.id)?;
                if let Some(stage) = self.stage {
                    map.serialize_entry("stage", &stage)?;
                }
                map.serialize_entry("dim", &self.dim)?;
                if let Some(agid) = self.airgroup_id {
                    map.serialize_entry("airgroupId", &agid)?;
                }
            }
            "xDivXSubXi" => {
                map.serialize_entry("id", &self.id)?;
                if let Some(opening) = self.opening {
                    map.serialize_entry("opening", &opening)?;
                }
                map.serialize_entry("dim", &self.dim)?;
            }
            "Zi" => {
                if let Some(bid) = self.boundary_id {
                    map.serialize_entry("boundaryId", &bid)?;
                }
                map.serialize_entry("dim", &self.dim)?;
            }
            _ => {
                map.serialize_entry("id", &self.id)?;
                map.serialize_entry("dim", &self.dim)?;
                if let Some(prime) = self.prime {
                    map.serialize_entry("prime", &prime)?;
                }
                if let Some(ref value) = self.value {
                    map.serialize_entry("value", value)?;
                }
                if let Some(stage) = self.stage {
                    map.serialize_entry("stage", &stage)?;
                }
            }
        }
        map.end()
    }
}
