/// Index into the expression arena
pub type ExprId = usize;

/// A child of an expression node: either a reference to an arena entry
/// (used by pilout-parsed expressions) or an inline expression object
/// (used by constraint/DEEP polynomial builders to match JS semantics
/// where helper nodes are not pushed to the expressions array).
#[derive(Debug, Clone)]
pub enum ExprChild {
    /// Reference to an expression in the arena by index.
    Id(ExprId),
    /// Inline expression object (not stored in the arena).
    Inline(Box<Expression>),
}

impl ExprChild {
    /// Resolve this child to an `&Expression`, looking up arena entries
    /// in `expressions` when needed.
    pub fn resolve<'a>(&'a self, expressions: &'a [Expression]) -> &'a Expression {
        match self {
            ExprChild::Id(id) => &expressions[*id],
            ExprChild::Inline(expr) => expr,
        }
    }
}

/// A single expression node in the arena.
#[derive(Debug, Clone)]
pub struct Expression {
    pub op: String,
    pub values: Vec<ExprChild>,
    pub id: Option<usize>,
    pub dim: usize,
    pub stage: usize,
    pub exp_deg: i64,
    pub row_offset: Option<i64>,
    pub rows_offsets: Vec<i64>,
    pub value: Option<String>,
    pub boundary_id: Option<usize>,
    pub boundary: Option<String>,
    pub commit_id: Option<usize>,
    pub stage_id: Option<usize>,
    pub airgroup_id: Option<usize>,
    pub im_pol: bool,
    pub pol_id: Option<usize>,
    pub keep: Option<bool>,
    /// True once `add_info_expressions` has fully processed this node.
    /// Mirrors the JS pattern `if("expDeg" in exp) return;`.
    pub info_computed: bool,
    /// Cached pretty-printed representation of this expression.
    /// Set by `print_expression` and used by `generateExpressionsCode`.
    /// Mirrors the JS pattern where `exp.line` is set lazily.
    pub line: Option<String>,
}

impl Default for Expression {
    fn default() -> Self {
        Self {
            op: String::new(),
            values: Vec::new(),
            id: None,
            dim: 1,
            stage: 0,
            exp_deg: 0,
            row_offset: None,
            rows_offsets: Vec::new(),
            value: None,
            boundary_id: None,
            boundary: None,
            commit_id: None,
            stage_id: None,
            airgroup_id: None,
            im_pol: false,
            pol_id: None,
            keep: None,
            info_computed: false,
            line: None,
        }
    }
}
