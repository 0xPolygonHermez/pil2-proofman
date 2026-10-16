//! The typed form of a code block: operations over operand references, as the bytecode
//! writers consume them.

use anyhow::{bail, Result};

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
