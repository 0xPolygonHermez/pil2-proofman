#![no_std]

//! The constraint DAG, the one Rust source of its semantics: the format and its JSON codec, `validate` (canonical
//! form, reachability, constants in the field), the leaf resolvers (an air's `LowerCtx`, the global `GlobalTables`),
//! the lowering to straight-line code and the evaluation an air's verifier runs. The kernel builds DAGs
//! (`pil_info::core::dag::build`); C++ `constraint_dag.hpp` mirrors this crate.

extern crate alloc;

pub mod format;
pub mod global;
pub mod lower_code;

pub use format::*;
pub use global::*;
pub use lower_code::*;

/// Dimension of a value of `stage`: a stage-1 value is a base element, any other one an extension element.
pub fn value_dim(stage: u32, ext: usize) -> usize {
    if stage == 1 {
        1
    } else {
        ext
    }
}

/// The only verifier-challenge fold (generic Horner): `acc = c0; acc = vc*acc + c_i` over `roots` in order; `None`
/// when there are none.
pub fn fold_roots<T>(
    roots: impl IntoIterator<Item = T>,
    mut mul_vc: impl FnMut(T) -> T,
    mut add: impl FnMut(T, T) -> T,
) -> Option<T> {
    let mut it = roots.into_iter();
    let first = it.next()?;
    Some(it.fold(first, |acc, c| add(mul_vc(acc), c)))
}

#[cfg(test)]
mod tests {
    use super::fold_roots;
    use alloc::format;
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;

    #[test]
    fn fold_is_horner_in_root_order() {
        let mul = |s: String| format!("(vc*{s})");
        let add = |a: String, b: String| format!("({a} + {b})");
        let r = |v: &[&str]| fold_roots(v.iter().map(|s| s.to_string()).collect::<Vec<_>>(), mul, add);
        assert_eq!(r(&["c0", "c1", "c2"]).unwrap(), "((vc*((vc*c0) + c1)) + c2)");
        assert_eq!(r(&["c0"]).unwrap(), "c0");
        assert_eq!(r(&[]), None);
    }
}
