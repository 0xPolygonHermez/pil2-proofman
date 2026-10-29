//! Constraint domains: normalization and the row/period/degree facts the constraint polynomial needs.

use crate::error::{PilInfoError, Result};

/// The most vanishing factors a domain's offsets may have.
pub const MAX_DOMAIN_FACTORS: usize = 32;

/// Rows `o + k*2^cycle_bits` for `o` in `offsets`, or all other rows if `complement`: as the pilout
/// declares it, or normalized by [`normalize`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Domain {
    pub cycle_bits: u32,
    pub offsets: Vec<u32>,
    pub complement: bool,
}

/// The rows a constraint holds on: every row (boundary 0) or a domain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Boundary {
    EveryRow,
    Domain(Domain),
}

impl Boundary {
    /// The boundary's `name` in the JSON outputs.
    pub fn name(&self) -> &'static str {
        match self {
            Boundary::EveryRow => "everyRow",
            Boundary::Domain(_) => "domain",
        }
    }
}

/// Reduces the cycle while the selected rows repeat within it, then keeps the side with fewer offsets;
/// a domain of every row is `EveryRow`.
pub fn normalize(raw: &Domain, n_bits: u32, name: &str) -> Result<Boundary> {
    let err = |m: String| PilInfoError::InvalidDomain(format!("{name}: {m}"));
    if raw.cycle_bits > n_bits {
        return Err(err(format!("cycle of {} rows exceeds N={}", 1u64 << raw.cycle_bits, 1u64 << n_bits)));
    }
    let mut cycle = raw.cycle_bits;
    let mut sel = vec![false; 1usize << cycle];
    for &o in &raw.offsets {
        if o >= 1 << cycle {
            return Err(err(format!("offset {o} is outside the cycle of {} rows", 1u64 << cycle)));
        }
        sel[o as usize] = true;
    }
    if raw.complement {
        sel.iter_mut().for_each(|b| *b = !*b);
    }
    while cycle > 0 {
        let half = sel.len() / 2;
        if sel[..half] != sel[half..] {
            break;
        }
        sel.truncate(half);
        cycle -= 1;
    }
    let count = sel.iter().filter(|b| **b).count();
    if count == 0 {
        return Err(err("the domain selects no row".to_string()));
    }
    if count == sel.len() {
        return Ok(Boundary::EveryRow);
    }
    let complement = count > sel.len() / 2;
    let offsets: Vec<u32> = (0..sel.len() as u32).filter(|&o| sel[o as usize] != complement).collect();
    Ok(Boundary::Domain(Domain { cycle_bits: cycle, offsets, complement }))
}

/// The STARK builds the domain's vanishing product: at most `MAX_DOMAIN_FACTORS` factors.
pub fn check_vanishing_factors(d: &Domain, name: &str) -> Result<()> {
    if d.offsets.len() > MAX_DOMAIN_FACTORS {
        return Err(PilInfoError::InvalidDomain(format!(
            "{name}: the domain needs {} vanishing factors, above the limit of {MAX_DOMAIN_FACTORS}; use a fixed selector column",
            d.offsets.len()
        )));
    }
    Ok(())
}

impl Domain {
    /// Trace rows the domain selects.
    pub fn rows(&self, n_bits: u32) -> u64 {
        let direct = self.offsets.len() as u64 * (1u64 << (n_bits - self.cycle_bits));
        if self.complement {
            (1u64 << n_bits) - direct
        } else {
            direct
        }
    }

    /// log2 of the period of the domain's zerofier over the extended domain.
    pub fn period_bits(&self, n_bits: u32, n_bits_ext: u32) -> u32 {
        n_bits_ext - n_bits + self.cycle_bits
    }

    /// 0 if the selected rows plus the constraint degree exceed N (the quotient absorbs the zerofier), else 1.
    pub fn zi_extra_degree(&self, n_bits: u32, d: i64) -> i64 {
        if self.rows(n_bits) as i64 + d > 1i64 << n_bits {
            0
        } else {
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dom(c: u32, o: &[u32], comp: bool) -> Domain {
        Domain { cycle_bits: c, offsets: o.to_vec(), complement: comp }
    }
    // a domain as the pilout declares it
    fn raw(c: u32, o: &[u32], comp: bool) -> Domain {
        dom(c, o, comp)
    }
    fn norm(r: Domain, n: u32) -> Boundary {
        normalize(&r, n, "T").unwrap()
    }

    #[test]
    fn limits_16() {
        let Boundary::Domain(d) = norm(raw(4, &[0, 15], false), 8) else { panic!() };
        assert_eq!(d, dom(4, &[0, 15], false));
        assert_eq!(d.rows(8), 32);
        assert_eq!(d.period_bits(8, 9), 5);
    }

    #[test]
    fn alternating_reduces() {
        assert_eq!(norm(raw(2, &[0, 2], false), 8), Boundary::Domain(dom(1, &[0], false)));
    }

    #[test]
    fn dense_flips() {
        assert_eq!(norm(raw(2, &[0, 1, 2], false), 8), Boundary::Domain(dom(2, &[3], true)));
    }

    #[test]
    fn whole_is_every_row() {
        assert_eq!(norm(raw(0, &[0], false), 8), Boundary::EveryRow);
    }

    #[test]
    fn complement_of_whole_errors() {
        let e = normalize(&raw(0, &[0], true), 8, "T").unwrap_err().to_string();
        assert!(e.contains("selects no row"), "{e}");
    }

    #[test]
    fn cycle_bigger_than_n_errors() {
        let e = normalize(&raw(4, &[0], false), 3, "A.BIG").unwrap_err().to_string();
        assert!(e.contains("A.BIG") && e.contains("cycle of 16 rows exceeds N=8"), "{e}");
    }

    #[test]
    fn offset_outside_the_cycle_errors() {
        let e = normalize(&raw(2, &[1, 4], false), 8, "A.OUT").unwrap_err().to_string();
        assert!(e.contains("A.OUT") && e.contains("offset 4 is outside the cycle of 4 rows"), "{e}");
    }

    #[test]
    fn too_many_factors_errors() {
        let offs: Vec<u32> = (0..33).map(|i| i * 7).collect();
        let Boundary::Domain(d) = norm(raw(8, &offs, false), 10) else { panic!() };
        let e = check_vanishing_factors(&d, "T").unwrap_err().to_string();
        assert!(e.contains("fixed selector column"), "{e}");
    }

    #[test]
    fn equivalent_domains_compare_equal() {
        assert_eq!(norm(raw(2, &[0, 2], false), 8), norm(raw(1, &[0], false), 8));
        assert_eq!(norm(raw(1, &[1], true), 8), norm(raw(1, &[0], false), 8));
    }

    #[test]
    fn degree_rule() {
        let Boundary::Domain(first) = norm(raw(8, &[0], false), 8) else { panic!() };
        assert_eq!(first.zi_extra_degree(8, 2), 1);
        let Boundary::Domain(not_first) = norm(raw(8, &[0], true), 8) else { panic!() };
        assert_eq!(not_first.rows(8), 255);
        assert_eq!(not_first.zi_extra_degree(8, 2), 0);
        assert_eq!(not_first.zi_extra_degree(8, 1), 1);
    }
}
