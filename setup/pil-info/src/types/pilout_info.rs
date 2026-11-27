// The arena's hint and symbol types.

use crate::expr::expression::Expression;

// ---------------------------------------------------------------------------
// Intermediate result types
// ---------------------------------------------------------------------------

/// A constraint as the code generator reads it.
#[derive(Debug, Clone)]
pub struct ConstraintInfo {
    pub boundary: String,
    pub e: usize,
    pub line: Option<String>,
    pub offset_min: Option<u32>,
    pub offset_max: Option<u32>,
    pub stage: Option<usize>,
    pub im_pol: bool,
}

/// A symbol as the code generator reads it, projected from the layout.
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub name: String,
    pub sym_type: String,
    pub stage: Option<usize>,
    pub dim: usize,
    pub id: Option<usize>,
    pub pol_id: Option<usize>,
    pub stage_id: Option<usize>,
    pub air_id: Option<usize>,
    pub airgroup_id: Option<usize>,
    pub commit_id: Option<usize>,
    pub lengths: Option<Vec<usize>>,
    pub stage_pos: Option<usize>,
    pub im_pol: bool,
    pub exp_id: Option<usize>,
}

impl Default for SymbolInfo {
    fn default() -> Self {
        Self {
            name: String::new(),
            sym_type: String::new(),
            stage: None,
            dim: 1,
            id: None,
            pol_id: None,
            stage_id: None,
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        }
    }
}

/// A single hint field value (leaf or nested array).
#[derive(Debug, Clone)]
pub enum HintFieldValue {
    /// A leaf: an expression node (op="exp", op="string", etc.)
    Single(Box<Expression>),
    /// A nested array of hint field values
    Array(Box<Vec<HintFieldValue>>),
}

/// A named hint field with value(s) and optional dimension lengths.
#[derive(Debug, Clone)]
pub struct HintFieldEntry {
    pub name: String,
    pub values: Vec<HintFieldValue>,
    pub lengths: Option<Vec<usize>>,
}

/// A formatted hint.
#[derive(Debug, Clone)]
pub struct HintInfo {
    pub name: String,
    pub fields: Vec<HintFieldEntry>,
}

/// What pilfflonk reads of an air's setup: its maps, constraints and opening points.
#[derive(Debug)]
pub struct ShplonkAir {
    pub name: String,
    pub pil_power: u32,
    pub n_stages: usize,
    pub cm_pols_map: Vec<SymbolInfo>,
    pub const_pols_map: Vec<SymbolInfo>,
    pub challenges_map: Vec<SymbolInfo>,
    pub airgroup_values_map: Vec<SymbolInfo>,
    pub air_values_map: Vec<SymbolInfo>,
    pub constraints: Vec<ConstraintInfo>,
    /// The opening points, sorted numerically.
    pub opening_points: Vec<i64>,
}
