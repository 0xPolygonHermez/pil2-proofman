//! The circuit type identifying every proof of the recursion chain (key selection and transcripts).

/// Marks an empty aggregation slot; `circuit_type_of` never returns it.
pub const NULL_CIRCUIT_TYPE: u64 = 0;

/// Szudzik `pair(airgroup_id, slot) + 1`; slot 0 is the recursive2, air `a` is `a + 1`. Mirrored by circom `circuitTypeOf`.
pub fn circuit_type_of(airgroup_id: u64, slot: u64) -> u64 {
    let pair =
        if airgroup_id < slot { slot * slot + airgroup_id } else { airgroup_id * airgroup_id + airgroup_id + slot };
    pair + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn circuit_type_of_reserves_null_and_is_injective() {
        assert_eq!(circuit_type_of(0, 0), 1, "recursive2 of airgroup 0");
        assert_eq!(circuit_type_of(0, 1), 2, "air 0 of airgroup 0");
        assert_eq!(circuit_type_of(1, 1), 4);
        let mut seen = HashSet::new();
        for g in 0..64 {
            for s in 0..64 {
                let t = circuit_type_of(g, s);
                assert_ne!(t, NULL_CIRCUIT_TYPE, "({g}, {s}) collides with null");
                assert!(seen.insert(t), "({g}, {s}) -> {t} is not unique");
            }
        }
    }
}
