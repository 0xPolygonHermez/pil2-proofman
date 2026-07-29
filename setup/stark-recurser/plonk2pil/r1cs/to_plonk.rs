//! Convert R1CS constraints (A*B=C) to PLONK gate format.
//!
//! Each PLONK constraint is `[sl, sr, so, qM, qL, qR, qO, qC]` representing:
//!   qM*sl*sr + qL*sl + qR*sr + qO*so + qC = 0
//!
//! Each PLONK addition is `[sl, sr, coef_l, coef_r]` recording that a new
//! variable was introduced equal to `coef_l*sl + coef_r*sr`.

use std::collections::HashMap;

use super::types::{CustomGateUse, R1csFile};
use crate::plonk2pil::utils::{addp, mulp, neg};

/// A PLONK constraint: [sl, sr, so, qM, qL, qR, qO, qC].
pub type PlonkConstraint = [u64; 8];

/// A PLONK addition: [sl, sr, coef_l, coef_r].
pub type PlonkAddition = [u64; 4];

/// Constraint key: hex fingerprint of the 5 coefficient fields.
/// Used to merge Plonk constraints that share the same gate type.
pub fn ckey(c: &PlonkConstraint) -> String {
    format!("{:x},{:x},{:x},{:x},{:x}", c[3], c[4], c[5], c[6], c[7])
}

type LC = HashMap<u32, u64>;

fn join(lc1: &LC, k: u64, lc2: &LC) -> LC {
    let mut res: LC = HashMap::new();
    for (&s, &coeff) in lc1 {
        let val = mulp(k, coeff);
        let e = res.entry(s).or_insert(0);
        *e = addp(*e, val);
    }
    for (&s, &coeff) in lc2 {
        let val = neg(coeff);
        let e = res.entry(s).or_insert(0);
        *e = addp(*e, val);
    }
    normalize(&mut res);
    res
}

fn normalize(lc: &mut LC) {
    lc.retain(|_, v| *v != 0);
}

struct ReducedCoefs {
    k: u64,
    s: Vec<u32>,
    coefs: Vec<u64>,
}

fn reduce_coefs(
    lc: &LC,
    max_c: usize,
    plonk_n_vars: &mut u32,
    plonk_constraints: &mut Vec<PlonkConstraint>,
    plonk_additions: &mut Vec<PlonkAddition>,
) -> ReducedCoefs {
    let mut result_k: u64 = 0;
    let mut cs: Vec<(u32, u64)> = Vec::new();

    for (&s, &coeff) in lc {
        if s == 0 {
            result_k = addp(result_k, coeff);
        } else if coeff != 0 {
            cs.push((s, coeff));
        }
    }
    // JavaScript iterates integer-keyed object properties in ascending numeric order.
    // Sort here to match that deterministic ordering so reduce_coefs output is identical.
    cs.sort_by_key(|(s, _)| *s);

    while cs.len() > max_c {
        let c1 = cs.remove(0);
        let c2 = cs.remove(0);
        let sl = c1.0;
        let sr = c2.0;
        let so = *plonk_n_vars;
        *plonk_n_vars += 1;
        plonk_constraints.push([sl as u64, sr as u64, so as u64, 0, neg(c1.1), neg(c2.1), 1, 0]);
        plonk_additions.push([sl as u64, sr as u64, c1.1, c2.1]);
        cs.push((so, 1));
    }

    let mut s_vec = Vec::with_capacity(max_c);
    let mut coefs_vec = Vec::with_capacity(max_c);
    for &(s, c) in &cs {
        s_vec.push(s);
        coefs_vec.push(c);
    }
    while s_vec.len() < max_c {
        s_vec.push(0);
        coefs_vec.push(0);
    }
    ReducedCoefs { k: result_k, s: s_vec, coefs: coefs_vec }
}

fn get_lc_type(lc: &mut LC) -> String {
    let mut k: u64 = 0;
    let mut n: usize = 0;
    let keys: Vec<u32> = lc.keys().copied().collect();
    for s in keys {
        let coeff = lc[&s];
        if coeff == 0 {
            lc.remove(&s);
        } else if s == 0 {
            k = addp(k, coeff);
        } else {
            n += 1;
        }
    }
    if n > 0 {
        return n.to_string();
    }
    if k != 0 {
        return "k".to_string();
    }
    "0".to_string()
}

fn add_constraint_sum(
    lc: &LC,
    plonk_n_vars: &mut u32,
    plonk_constraints: &mut Vec<PlonkConstraint>,
    plonk_additions: &mut Vec<PlonkAddition>,
) {
    let c = reduce_coefs(lc, 3, plonk_n_vars, plonk_constraints, plonk_additions);
    plonk_constraints.push([c.s[0] as u64, c.s[1] as u64, c.s[2] as u64, 0, c.coefs[0], c.coefs[1], c.coefs[2], c.k]);
}

fn add_constraint_mul(
    lc_a: &LC,
    lc_b: &LC,
    lc_c: &LC,
    plonk_n_vars: &mut u32,
    plonk_constraints: &mut Vec<PlonkConstraint>,
    plonk_additions: &mut Vec<PlonkAddition>,
) {
    let a = reduce_coefs(lc_a, 1, plonk_n_vars, plonk_constraints, plonk_additions);
    let b = reduce_coefs(lc_b, 1, plonk_n_vars, plonk_constraints, plonk_additions);
    let c = reduce_coefs(lc_c, 1, plonk_n_vars, plonk_constraints, plonk_additions);
    plonk_constraints.push([
        a.s[0] as u64,
        b.s[0] as u64,
        c.s[0] as u64,
        mulp(a.coefs[0], b.coefs[0]),
        mulp(a.coefs[0], b.k),
        mulp(a.k, b.coefs[0]),
        neg(c.coefs[0]),
        addp(mulp(a.k, b.k), neg(c.k)),
    ]);
}

pub fn r1cs2plonk(r1cs: &R1csFile) -> (Vec<PlonkConstraint>, Vec<PlonkAddition>) {
    let mut plonk_constraints: Vec<PlonkConstraint> = Vec::new();
    let mut plonk_additions: Vec<PlonkAddition> = Vec::new();
    let mut plonk_n_vars = r1cs.header.n_vars;

    for (c_idx, constraint) in r1cs.constraints.iter().enumerate() {
        if c_idx % 100_000 == 0 {
            tracing::debug!("Processing constraints: {}/{}", c_idx, r1cs.header.n_constraints);
        }
        let mut lc_a: LC = constraint.a.iter().map(|(&k, &v)| (k, v)).collect();
        let mut lc_b: LC = constraint.b.iter().map(|(&k, &v)| (k, v)).collect();
        let mut lc_c: LC = constraint.c.iter().map(|(&k, &v)| (k, v)).collect();

        let lct_a = get_lc_type(&mut lc_a);
        let lct_b = get_lc_type(&mut lc_b);

        if lct_a == "0" || lct_b == "0" {
            normalize(&mut lc_c);
            add_constraint_sum(&lc_c, &mut plonk_n_vars, &mut plonk_constraints, &mut plonk_additions);
        } else if lct_a == "k" {
            let k_a = lc_a.get(&0).copied().unwrap_or(0);
            let lc_cc = join(&lc_b, k_a, &lc_c);
            add_constraint_sum(&lc_cc, &mut plonk_n_vars, &mut plonk_constraints, &mut plonk_additions);
        } else if lct_b == "k" {
            let k_b = lc_b.get(&0).copied().unwrap_or(0);
            let lc_cc = join(&lc_a, k_b, &lc_c);
            add_constraint_sum(&lc_cc, &mut plonk_n_vars, &mut plonk_constraints, &mut plonk_additions);
        } else {
            add_constraint_mul(&lc_a, &lc_b, &lc_c, &mut plonk_n_vars, &mut plonk_constraints, &mut plonk_additions);
        }
    }
    (plonk_constraints, plonk_additions)
}

// ─── Custom gates ─────────────────────────────────────────────────────────────

use proofman_common::hash_family::{lookup_gate, GateRole};

#[derive(Debug, Clone, Default)]
pub struct CustomGatesInfo {
    pub gate_ids: HashMap<GateRole, u32>,
    pub fft4_parameters: HashMap<u32, Vec<u64>>,
    pub n_per_role: HashMap<GateRole, usize>,
    pub n_plonk_rows: usize,
}

impl CustomGatesInfo {
    pub fn role_id(&self, role: GateRole) -> Option<u32> {
        self.gate_ids.get(&role).copied()
    }

    pub fn n(&self, role: GateRole) -> usize {
        self.n_per_role.get(&role).copied().unwrap_or(0)
    }
}

pub fn get_custom_gates_info(r1cs: &R1csFile) -> CustomGatesInfo {
    let mut info = CustomGatesInfo::default();
    let mut families_seen: Vec<&'static str> = Vec::new();

    for (i, gate) in r1cs.custom_gates.iter().enumerate() {
        let i = i as u32;
        let name = gate.template_name.as_str();
        let (role, owning_family) = lookup_gate(name).unwrap_or_else(|| panic!("Unknown custom gate: {name}"));
        match role {
            GateRole::Fft4 => {
                info.fft4_parameters.insert(i, gate.parameters.clone());
            }
            _ => {
                assert!(gate.parameters.is_empty(), "{name} expected to be parameter-less");
                if let Some(prev) = info.gate_ids.insert(role, i) {
                    panic!("duplicate role {role:?}: gates {prev} and {i}");
                }
            }
        }
        if let Some(fam_id) = owning_family {
            if !families_seen.contains(&fam_id) {
                families_seen.push(fam_id);
            }
        }
    }

    if families_seen.len() > 1 {
        panic!("r1cs mixes multiple hash families: {families_seen:?}");
    }

    let id_to_role: HashMap<u32, GateRole> = info.gate_ids.iter().map(|(role, id)| (*id, *role)).collect();
    for cgu in &r1cs.custom_gates_uses {
        let role = if let Some(role) = id_to_role.get(&cgu.id) {
            *role
        } else if info.fft4_parameters.contains_key(&cgu.id) {
            GateRole::Fft4
        } else {
            panic!("Custom gate not defined: {}", cgu.id);
        };
        *info.n_per_role.entry(role).or_insert(0) += 1;
    }
    info
}

pub fn filter_gate_uses(uses: &[CustomGateUse], id: Option<u32>) -> Vec<&CustomGateUse> {
    match id {
        Some(id) => uses.iter().filter(|cgu| cgu.id == id).collect(),
        None => Vec::new(),
    }
}

pub fn filter_fft4_gate_uses<'a>(
    uses: &'a [CustomGateUse],
    fft4_params: &HashMap<u32, Vec<u64>>,
) -> Vec<&'a CustomGateUse> {
    uses.iter().filter(|cgu| fft4_params.contains_key(&cgu.id)).collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plonk2pil::r1cs::types::*;
    use crate::plonk2pil::utils::GOLDILOCKS_P;

    fn make_lc(terms: &[(u32, u64)]) -> LinearCombination {
        terms.iter().copied().collect()
    }

    fn make_r1cs(constraints: Vec<R1csConstraint>, n_vars: u32) -> R1csFile {
        R1csFile {
            header: R1csHeader {
                n8: 8,
                prime_bytes: vec![],
                n_vars,
                n_outputs: 0,
                n_pub_inputs: 0,
                n_prv_inputs: 0,
                n_labels: 0,
                n_constraints: constraints.len() as u32,
                use_custom_gates: false,
            },
            constraints,
            wire_to_label: vec![],
            custom_gates: vec![],
            custom_gates_uses: vec![],
        }
    }

    #[test]
    fn test_simple_mul() {
        let r1cs =
            make_r1cs(vec![R1csConstraint { a: make_lc(&[(1, 1)]), b: make_lc(&[(2, 1)]), c: make_lc(&[(3, 1)]) }], 4);
        let (cs, adds) = r1cs2plonk(&r1cs);
        assert_eq!(cs.len(), 1);
        assert_eq!(adds.len(), 0);
        assert_eq!(cs[0][3], 1);
        assert_eq!(cs[0][6], GOLDILOCKS_P - 1);
    }

    #[test]
    fn test_zero_a_sum() {
        let r1cs = make_r1cs(
            vec![R1csConstraint { a: HashMap::new(), b: make_lc(&[(1, 1)]), c: make_lc(&[(2, 1), (3, 1)]) }],
            4,
        );
        let (cs, _) = r1cs2plonk(&r1cs);
        assert_eq!(cs[0][3], 0);
    }

    #[test]
    fn test_additions() {
        let r1cs = make_r1cs(
            vec![R1csConstraint {
                a: make_lc(&[(1, 1), (2, 1), (3, 1), (4, 1)]),
                b: make_lc(&[(5, 1)]),
                c: make_lc(&[(6, 1)]),
            }],
            7,
        );
        let (cs, adds) = r1cs2plonk(&r1cs);
        assert!(cs.len() > 1);
        assert!(!adds.is_empty());
    }
}
