//! The names of the JSON view of a proof (A.6, D7): pil-fflonk's (`pil-fflonk/src/shplonk.cpp:
//! 316-328`, `pil-stark/src/fflonk/helpers/fflonk_verify.js`), extended to signed offsets, to
//! array columns and to proofs of several instances.
//!
//! `polynomials`:
//! - `f<g>`: the commitment of the `f` at position `g` of the global order of A.5, in decimal.
//!   Only the non-fixed `f` are there; the fixed ones are in the vkey (under the same names in
//!   a proof of one AIR, the only one vkey format 1 describes).
//! - `W` and `Wp`: SHPLONK's `W` and `W'`.
//!
//! `evaluations`:
//! - `<column><suffix>`: the evaluation of a column at `ξ·ω^s`, where
//!   - `<column>` is the column's name in `cmPolsMap` or `constPolsMap`, followed by `[i]` for
//!     each entry `i` of its `lengths`: `Fibonacci.l1`, `Main.a[0]`, `Main.b[1][0]`. The setup
//!     gives the `k`-th im pol `lengths: [k]`, and the `k`-th of the columns of a map that share a
//!     name and have no `lengths` too (the std's `im_cluster`, plan M34b), so that each has a name
//!     of its own: `Fibonacci.ImPol[0]`, `im_cluster[1]`;
//!   - `<suffix>` is empty for `s = 0`, `w` for `s = 1`, and `w` followed by `s` in decimal, sign
//!     included, otherwise: `w2`, `w-1`, `w-2`. For `s >= 0` these are pil-fflonk's names.
//! - `Q<i>` for the piece `i` of a split `Q` ([`q_piece_name`]): its `Q_i(ξ)`, after the other
//!   evaluations of its instance, in the order of the layout (A.4 step 4).
//! - `<value>`: an air value, an airgroup value or a proof value, by its name in its map with the
//!   same `[i]` suffixes.
//! - `inv` and `invZh`, as pil-fflonk.
//!
//! **Scopes.** A proof of a single instance (the v1 case, D2) uses the names above as they are.
//! In a proof of more than one, every name that belongs to an AIR, an instance or an airgroup is
//! prefixed with where it belongs:
//! - `<ag>.<a>:` for the evaluations of the fixed columns of AIR `a` of airgroup `ag`;
//! - `<ag>.<a>.<t>:` for the other evaluations and the air values of instance `t` of that AIR,
//!   counted from 0 in canonical order;
//! - `<ag>:` for the airgroup values of airgroup `ag`.
//!
//! The proof values, `inv` and `invZh` are never prefixed. A PIL identifier never starts with a
//! digit nor holds a `:`, so a prefixed name does not equal an unprefixed one, nor one of another
//! scope.
//!
//! Within a scope pil-fflonk's names can collide (a column `a` at offset 1 and a column `aw` at
//! offset 0 are both `aw`): `ProofNames` refuses a proof whose names collide, whatever the reason,
//! rather than lose a value.

/// The name of a column, an air value, an airgroup value or a proof value in the JSON view: its
/// name in its map and one `[i]` per entry of its `lengths`.
pub fn column_name(name: &str, lengths: &[u64]) -> String {
    let mut out = name.to_string();
    for index in lengths {
        out.push_str(&format!("[{index}]"));
    }
    out
}

/// The suffix of the offset `s` (see the module).
pub fn offset_suffix(offset: i64) -> String {
    match offset {
        0 => String::new(),
        1 => "w".to_string(),
        s => format!("w{s}"),
    }
}

/// The name of the evaluation of `column` at `ξ·ω^offset`.
pub fn evaluation_name(column: &str, offset: i64) -> String {
    format!("{column}{}", offset_suffix(offset))
}

/// The name of the piece `i` of `Q` (A.1, A.6): its name in `cmPolsMap` and the layout, and of its
/// `Q_i(ξ)` in the proof if `Q` is split. `Q` whole is its one piece, `Q0`.
pub fn q_piece_name(i: u64) -> String {
    format!("Q{i}")
}

/// The name of the commitment of the `f` at position `global_index` of the global order (A.5).
pub fn commitment_name(global_index: u64) -> String {
    format!("f{global_index}")
}

pub const W: &str = "W";
pub const WP: &str = "Wp";
pub const INV: &str = "inv";
pub const INV_ZH: &str = "invZh";

/// Where a name belongs, for the prefixes of a proof of several instances. The proof values,
/// `inv` and `invZh` belong to the whole proof and have no scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scope {
    Air { airgroup: u64, air: u64 },
    Instance { airgroup: u64, air: u64, instance: u64 },
    Airgroup { airgroup: u64 },
}

impl Scope {
    /// The prefix of the scope in a proof of `n_instances` instances.
    pub(crate) fn prefix(self, n_instances: usize) -> String {
        if n_instances <= 1 {
            return String::new();
        }
        match self {
            Scope::Air { airgroup, air } => format!("{airgroup}.{air}:"),
            Scope::Instance { airgroup, air, instance } => format!("{airgroup}.{air}.{instance}:"),
            Scope::Airgroup { airgroup } => format!("{airgroup}:"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_are_named_as_pil_fflonk_names_them_and_signed_ones_with_their_sign() {
        assert_eq!(evaluation_name("Fibonacci.l1", 0), "Fibonacci.l1");
        assert_eq!(evaluation_name("Fibonacci.l1", 1), "Fibonacci.l1w");
        assert_eq!(evaluation_name("Fibonacci.l1", 2), "Fibonacci.l1w2");
        assert_eq!(evaluation_name("Fibonacci.l1", -1), "Fibonacci.l1w-1");
        assert_eq!(evaluation_name("Fibonacci.l1", -12), "Fibonacci.l1w-12");
    }

    #[test]
    fn the_pieces_of_q_are_q_and_their_index() {
        assert_eq!((q_piece_name(0), q_piece_name(1), q_piece_name(12)), ("Q0".into(), "Q1".into(), "Q12".into()));
    }

    #[test]
    fn array_columns_carry_their_indices() {
        assert_eq!(column_name("Main.a", &[]), "Main.a");
        assert_eq!(column_name("Main.a", &[0]), "Main.a[0]");
        assert_eq!(column_name("Main.b", &[1, 0]), "Main.b[1][0]");
        assert_eq!(evaluation_name(&column_name("Main.a", &[3]), -1), "Main.a[3]w-1");
    }

    #[test]
    fn scopes_only_prefix_proofs_of_several_instances() {
        let air = Scope::Air { airgroup: 0, air: 1 };
        let instance = Scope::Instance { airgroup: 0, air: 1, instance: 2 };
        let airgroup = Scope::Airgroup { airgroup: 3 };
        for scope in [air, instance, airgroup] {
            assert_eq!(scope.prefix(1), "");
        }
        assert_eq!(air.prefix(2), "0.1:");
        assert_eq!(instance.prefix(2), "0.1.2:");
        assert_eq!(airgroup.prefix(2), "3:");
    }
}
