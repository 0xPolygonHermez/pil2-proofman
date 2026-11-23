//! The only verifier-challenge fold (generic Horner).

/// `acc = c0; acc = vc*acc + c_i` over `roots` in order; `None` when there are none.
pub fn fold_roots<T>(roots: Vec<T>, mut mul_vc: impl FnMut(T) -> T, mut add: impl FnMut(T, T) -> T) -> Option<T> {
    let mut it = roots.into_iter();
    let first = it.next()?;
    Some(it.fold(first, |acc, c| add(mul_vc(acc), c)))
}

#[cfg(test)]
mod tests {
    use super::fold_roots;

    #[test]
    fn fold_is_horner_in_root_order() {
        let mul = |s: String| format!("(vc*{s})");
        let add = |a: String, b: String| format!("({a} + {b})");
        let r = |v: &[&str]| fold_roots(v.iter().map(|s| s.to_string()).collect(), mul, add);
        assert_eq!(r(&["c0", "c1", "c2"]).unwrap(), "((vc*((vc*c0) + c1)) + c2)");
        assert_eq!(r(&["c0"]).unwrap(), "c0");
        assert_eq!(r(&[]), None);
    }
}
