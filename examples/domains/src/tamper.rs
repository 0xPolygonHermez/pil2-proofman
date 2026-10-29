//! `DOMAINS_TAMPER="<Air>:<col>:<row>:<value>"` overwrites one cell of the witness after it is filled.

pub struct Tamper {
    pub col: String,
    pub row: usize,
    pub value: u64,
}

/// The cell `DOMAINS_TAMPER` names, if it names `air`.
pub fn tamper_for(air: &str) -> Option<Tamper> {
    let spec = std::env::var("DOMAINS_TAMPER").ok()?;
    let parts: Vec<&str> = spec.split(':').collect();
    let [target, col, row, value] = parts[..] else {
        panic!("DOMAINS_TAMPER={spec:?}: expected <Air>:<col>:<row>:<value>");
    };
    if target != air {
        return None;
    }
    let row = row.parse().unwrap_or_else(|_| panic!("DOMAINS_TAMPER={spec:?}: bad row"));
    let value = value.parse().unwrap_or_else(|_| panic!("DOMAINS_TAMPER={spec:?}: bad value"));
    Some(Tamper { col: col.to_string(), row, value })
}

/// Applies `DOMAINS_TAMPER` to `$trace` of `$air`, whose columns are `$col`s.
#[macro_export]
macro_rules! tamper {
    ($trace:ident, $air:literal, $($col:ident),+) => {
        if let Some(t) = $crate::tamper::tamper_for($air) {
            match t.col.as_str() {
                $(stringify!($col) => $trace[t.row].$col = F::from_u64(t.value),)+
                other => panic!("DOMAINS_TAMPER: {} has no column {other:?}", $air),
            }
        }
    };
}
