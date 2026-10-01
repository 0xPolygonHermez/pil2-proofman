//! Convert R1CS constraints (A*B=C) to PLONK gate format, over the r1cs's field.
//!
//! Each PLONK constraint is a [`PlonkConstraint`]: wires `[sl, sr, so]` and coefficients
//! `[qM, qL, qR, qO, qC]`, representing
//!   qM*sl*sr + qL*sl + qR*sr + qO*so + qC = 0
//!
//! Each PLONK addition is a [`PlonkAddition`] recording that a new variable was introduced equal to
//! `coef_l*sl + coef_r*sr`. The `i`-th one is wire `n_vars + i`.

use std::collections::HashMap;

use proofman_fields::Field;

use super::types::{CustomGateUse, R1csFile};
use crate::plonk2pil::field::PlonkField;

/// Coefficients of a PLONK gate: `qM, qL, qR, qO, qC`.
pub const PLONK_COEFFS: usize = 5;

/// A PLONK constraint `qM*sl*sr + qL*sl + qR*sr + qO*so + qC = 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlonkConstraint<F> {
    /// `[sl, sr, so]`.
    pub wires: [u32; 3],
    /// `[qM, qL, qR, qO, qC]`.
    pub coeffs: [F; PLONK_COEFFS],
}

/// A PLONK addition: the wire it introduces is `coeffs[0]*wires[0] + coeffs[1]*wires[1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlonkAddition<F> {
    /// `[sl, sr]`.
    pub wires: [u32; 2],
    /// `[coef_l, coef_r]`.
    pub coeffs: [F; 2],
}

/// Constraint key: hex fingerprint of the 5 coefficient fields, `{:x}` of each canonical value
/// joined by commas. Used to merge Plonk constraints that share the same gate type; blake3 also
/// sorts on it, so the spelling is part of the row placement.
pub fn ckey<F: PlonkField>(c: &PlonkConstraint<F>) -> String {
    let mut key = String::new();
    for (i, q) in c.coeffs.iter().enumerate() {
        if i > 0 {
            key.push(',');
        }
        q.push_hex(&mut key);
    }
    key
}

type LC<F> = HashMap<u32, F>;

fn join<F: Field>(lc1: &LC<F>, k: F, lc2: &LC<F>) -> LC<F> {
    let mut res: LC<F> = HashMap::new();
    for (&s, &coeff) in lc1 {
        *res.entry(s).or_insert(F::ZERO) += k * coeff;
    }
    for (&s, &coeff) in lc2 {
        *res.entry(s).or_insert(F::ZERO) -= coeff;
    }
    normalize(&mut res);
    res
}

fn normalize<F: Field>(lc: &mut LC<F>) {
    lc.retain(|_, v| !v.is_zero());
}

struct ReducedCoefs<F> {
    k: F,
    s: Vec<u32>,
    coefs: Vec<F>,
}

fn reduce_coefs<F: Field>(
    lc: &LC<F>,
    max_c: usize,
    plonk_n_vars: &mut u32,
    plonk_constraints: &mut Vec<PlonkConstraint<F>>,
    plonk_additions: &mut Vec<PlonkAddition<F>>,
) -> ReducedCoefs<F> {
    let mut result_k = F::ZERO;
    let mut cs: Vec<(u32, F)> = Vec::new();

    for (&s, &coeff) in lc {
        if s == 0 {
            result_k += coeff;
        } else if !coeff.is_zero() {
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
        plonk_constraints
            .push(PlonkConstraint { wires: [sl, sr, so], coeffs: [F::ZERO, -c1.1, -c2.1, F::ONE, F::ZERO] });
        plonk_additions.push(PlonkAddition { wires: [sl, sr], coeffs: [c1.1, c2.1] });
        cs.push((so, F::ONE));
    }

    let mut s_vec = Vec::with_capacity(max_c);
    let mut coefs_vec = Vec::with_capacity(max_c);
    for &(s, c) in &cs {
        s_vec.push(s);
        coefs_vec.push(c);
    }
    while s_vec.len() < max_c {
        s_vec.push(0);
        coefs_vec.push(F::ZERO);
    }
    ReducedCoefs { k: result_k, s: s_vec, coefs: coefs_vec }
}

fn get_lc_type<F: Field>(lc: &mut LC<F>) -> String {
    let mut k = F::ZERO;
    let mut n: usize = 0;
    let keys: Vec<u32> = lc.keys().copied().collect();
    for s in keys {
        let coeff = lc[&s];
        if coeff.is_zero() {
            lc.remove(&s);
        } else if s == 0 {
            k += coeff;
        } else {
            n += 1;
        }
    }
    if n > 0 {
        return n.to_string();
    }
    if !k.is_zero() {
        return "k".to_string();
    }
    "0".to_string()
}

fn add_constraint_sum<F: Field>(
    lc: &LC<F>,
    plonk_n_vars: &mut u32,
    plonk_constraints: &mut Vec<PlonkConstraint<F>>,
    plonk_additions: &mut Vec<PlonkAddition<F>>,
) {
    let c = reduce_coefs(lc, 3, plonk_n_vars, plonk_constraints, plonk_additions);
    plonk_constraints.push(PlonkConstraint {
        wires: [c.s[0], c.s[1], c.s[2]],
        coeffs: [F::ZERO, c.coefs[0], c.coefs[1], c.coefs[2], c.k],
    });
}

fn add_constraint_mul<F: Field>(
    lc_a: &LC<F>,
    lc_b: &LC<F>,
    lc_c: &LC<F>,
    plonk_n_vars: &mut u32,
    plonk_constraints: &mut Vec<PlonkConstraint<F>>,
    plonk_additions: &mut Vec<PlonkAddition<F>>,
) {
    let a = reduce_coefs(lc_a, 1, plonk_n_vars, plonk_constraints, plonk_additions);
    let b = reduce_coefs(lc_b, 1, plonk_n_vars, plonk_constraints, plonk_additions);
    let c = reduce_coefs(lc_c, 1, plonk_n_vars, plonk_constraints, plonk_additions);
    plonk_constraints.push(PlonkConstraint {
        wires: [a.s[0], b.s[0], c.s[0]],
        coeffs: [a.coefs[0] * b.coefs[0], a.coefs[0] * b.k, a.k * b.coefs[0], -c.coefs[0], a.k * b.k - c.k],
    });
}

pub fn r1cs2plonk<F: Field>(r1cs: &R1csFile<F>) -> (Vec<PlonkConstraint<F>>, Vec<PlonkAddition<F>>) {
    let mut plonk_constraints: Vec<PlonkConstraint<F>> = Vec::new();
    let mut plonk_additions: Vec<PlonkAddition<F>> = Vec::new();
    let mut plonk_n_vars = r1cs.header.n_vars;

    for (c_idx, constraint) in r1cs.constraints.iter().enumerate() {
        if c_idx % 100_000 == 0 {
            tracing::debug!("Processing constraints: {}/{}", c_idx, r1cs.header.n_constraints);
        }
        let mut lc_a: LC<F> = constraint.a.iter().map(|(&k, &v)| (k, v)).collect();
        let mut lc_b: LC<F> = constraint.b.iter().map(|(&k, &v)| (k, v)).collect();
        let mut lc_c: LC<F> = constraint.c.iter().map(|(&k, &v)| (k, v)).collect();

        let lct_a = get_lc_type(&mut lc_a);
        let lct_b = get_lc_type(&mut lc_b);

        if lct_a == "0" || lct_b == "0" {
            normalize(&mut lc_c);
            add_constraint_sum(&lc_c, &mut plonk_n_vars, &mut plonk_constraints, &mut plonk_additions);
        } else if lct_a == "k" {
            let k_a = lc_a.get(&0).copied().unwrap_or(F::ZERO);
            let lc_cc = join(&lc_b, k_a, &lc_c);
            add_constraint_sum(&lc_cc, &mut plonk_n_vars, &mut plonk_constraints, &mut plonk_additions);
        } else if lct_b == "k" {
            let k_b = lc_b.get(&0).copied().unwrap_or(F::ZERO);
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
pub struct CustomGatesInfo<F> {
    pub gate_ids: HashMap<GateRole, u32>,
    pub fft4_parameters: HashMap<u32, Vec<F>>,
    /// `Blake3Compress(flags, isParent)` parameters, per gate id.
    ///
    /// Circom mints one gate id per distinct parameter pair -- about eight across a whole verifier
    /// -- so this role, like Fft4, has no single id and is exempt from the duplicate-role check.
    /// The setup needs the VALUES: `flags` becomes an air fixed column and `isParent` picks which of
    /// the two Compress block kinds a block belongs to, so neither takes a trace cell. Reading them
    /// here is what makes that sound -- the alternative was pattern-matching the r1cs constraint
    /// that pins each one.
    pub blake3_compress_parameters: HashMap<u32, Vec<F>>,
    /// `PoseidonT(t)` widths, per gate id: circom mints one gate id per width, so this role, like
    /// Fft4, has no single id. Which widths a family places is the family's to check.
    pub poseidon_t_widths: HashMap<u32, F>,
    /// `Num2Bytes(nBits)` bits, per gate id: one gate id per nBits, as for `PoseidonT`. Which
    /// widths a family places is the family's to check.
    pub range_check_bits: HashMap<u32, F>,
    pub n_per_role: HashMap<GateRole, usize>,
    pub n_plonk_rows: usize,
}

impl<F> CustomGatesInfo<F> {
    pub fn role_id(&self, role: GateRole) -> Option<u32> {
        self.gate_ids.get(&role).copied()
    }

    pub fn n(&self, role: GateRole) -> usize {
        self.n_per_role.get(&role).copied().unwrap_or(0)
    }
}

pub fn get_custom_gates_info<F: Field>(r1cs: &R1csFile<F>) -> CustomGatesInfo<F> {
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
            GateRole::Blake3Compress => {
                assert_eq!(
                    gate.parameters.len(),
                    2,
                    "Blake3Compress is Blake3Compress(flags, isParent); gate {i} has {} parameters",
                    gate.parameters.len()
                );
                let is_parent = gate.parameters[1];
                assert!(is_parent.is_zero() || is_parent.is_one(), "isParent must be 0 or 1, gate {i} has {is_parent}");
                info.blake3_compress_parameters.insert(i, gate.parameters.clone());
            }
            GateRole::PoseidonT => {
                assert_eq!(
                    gate.parameters.len(),
                    1,
                    "PoseidonT is PoseidonT(t); gate {i} has {} parameters",
                    gate.parameters.len()
                );
                info.poseidon_t_widths.insert(i, gate.parameters[0]);
            }
            GateRole::RangeCheck => {
                assert_eq!(
                    gate.parameters.len(),
                    1,
                    "Num2Bytes is Num2Bytes(nBits); gate {i} has {} parameters",
                    gate.parameters.len()
                );
                info.range_check_bits.insert(i, gate.parameters[0]);
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
        } else if info.blake3_compress_parameters.contains_key(&cgu.id) {
            GateRole::Blake3Compress
        } else if info.poseidon_t_widths.contains_key(&cgu.id) {
            GateRole::PoseidonT
        } else if info.range_check_bits.contains_key(&cgu.id) {
            GateRole::RangeCheck
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

pub fn filter_fft4_gate_uses<'a, F>(
    uses: &'a [CustomGateUse],
    fft4_params: &HashMap<u32, Vec<F>>,
) -> Vec<&'a CustomGateUse> {
    uses.iter().filter(|cgu| fft4_params.contains_key(&cgu.id)).collect()
}

/// One `Blake3Compress` use with the `(flags, isParent)` its gate id was minted for.
///
/// The packer needs both: `flags` goes into an air fixed column and `isParent` decides whether the
/// block joins the chunk band or the parent band. Neither is a trace cell, so neither can be read
/// off `cgu.signals` -- they live on the gate, not the use.
pub fn blake3_compress_gate_uses<'a, F: Copy>(
    uses: &'a [CustomGateUse],
    params: &HashMap<u32, Vec<F>>,
) -> Vec<(&'a CustomGateUse, F, F)> {
    uses.iter().filter_map(|cgu| params.get(&cgu.id).map(|p| (cgu, p[0], p[1]))).collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plonk2pil::r1cs::types::*;
    use proofman_fields::{Bn254, Goldilocks, QuotientMap};

    fn make_lc<F: QuotientMap<u64>>(terms: &[(u32, u64)]) -> LinearCombination<F> {
        terms.iter().map(|&(w, c)| (w, F::from_int(c))).collect()
    }

    fn make_r1cs<F>(constraints: Vec<R1csConstraint<F>>, n_vars: u32) -> R1csFile<F> {
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

    /// Extends `witness` (wire 0 is the constant one) with the wires the additions introduce, in
    /// their order, and says whether every gate holds on the result.
    fn gates_hold<F: Field>(cs: &[PlonkConstraint<F>], adds: &[PlonkAddition<F>], mut witness: Vec<F>) -> bool {
        for a in adds {
            let v = a.coeffs[0] * witness[a.wires[0] as usize] + a.coeffs[1] * witness[a.wires[1] as usize];
            witness.push(v);
        }
        cs.iter().all(|c| {
            let [l, r, o] = c.wires.map(|w| witness[w as usize]);
            let [q_m, q_l, q_r, q_o, q_c] = c.coeffs;
            (q_m * l * r + q_l * l + q_r * r + q_o * o + q_c).is_zero()
        })
    }

    fn simple_mul<F: Field + QuotientMap<u64>>() {
        let r1cs = make_r1cs(
            vec![R1csConstraint { a: make_lc::<F>(&[(1, 1)]), b: make_lc(&[(2, 1)]), c: make_lc(&[(3, 1)]) }],
            4,
        );
        let (cs, adds) = r1cs2plonk(&r1cs);
        assert_eq!(cs.len(), 1);
        assert_eq!(adds.len(), 0);
        assert_eq!(cs[0].wires, [1, 2, 3]);
        assert_eq!(cs[0].coeffs, [F::ONE, F::ZERO, F::ZERO, F::NEG_ONE, F::ZERO]);
    }

    #[test]
    fn test_simple_mul() {
        simple_mul::<Goldilocks>();
        simple_mul::<Bn254>();
    }

    fn zero_a_sum<F: Field + QuotientMap<u64>>() {
        let r1cs = make_r1cs(
            vec![R1csConstraint {
                a: LinearCombination::new(),
                b: make_lc::<F>(&[(1, 1)]),
                c: make_lc(&[(2, 1), (3, 1)]),
            }],
            4,
        );
        let (cs, _) = r1cs2plonk(&r1cs);
        assert!(cs[0].coeffs[0].is_zero());
    }

    #[test]
    fn test_zero_a_sum() {
        zero_a_sum::<Goldilocks>();
        zero_a_sum::<Bn254>();
    }

    /// `(w1 + 2·w2 + 3·w3 + 4·w4 + 5) · w5 = w6`, a product of a wide sum, and `7 · (w1 + w2) = w3 + w6`
    /// with a constant factor: the chains of additions and both sum paths hold on a satisfying
    /// assignment, and stop holding when it is not one.
    fn conversion_holds_on_a_witness<F: Field + QuotientMap<u64>>() {
        let r1cs = make_r1cs(
            vec![
                R1csConstraint {
                    a: make_lc::<F>(&[(1, 1), (2, 2), (3, 3), (4, 4), (0, 5)]),
                    b: make_lc(&[(5, 1)]),
                    c: make_lc(&[(6, 1)]),
                },
                R1csConstraint { a: make_lc(&[(0, 7)]), b: make_lc(&[(1, 1), (2, 1)]), c: make_lc(&[(3, 1), (7, 1)]) },
            ],
            8,
        );
        let (cs, adds) = r1cs2plonk(&r1cs);
        assert!(!adds.is_empty(), "a five-term sum must introduce additions");

        let w = |v: u64| F::from_int(v);
        let (w1, w2, w3, w4, w5) = (w(11), w(13), w(17), w(19), w(23));
        let w6 = (w1 + w2.double() + w3 * w(3) + w4 * w(4) + w(5)) * w5;
        let w7 = w(7) * (w1 + w2) - w3;
        let witness = vec![F::ONE, w1, w2, w3, w4, w5, w6, w7];
        assert!(gates_hold(&cs, &adds, witness.clone()));

        let mut wrong = witness;
        wrong[6] += F::ONE;
        assert!(!gates_hold(&cs, &adds, wrong));
    }

    #[test]
    fn conversion_holds_in_both_fields() {
        conversion_holds_on_a_witness::<Goldilocks>();
        conversion_holds_on_a_witness::<Bn254>();
    }

    /// `PoseidonT(t)` has a parameter, its width: each width is a gate id of the role, and the
    /// uses of every one of them are the role's.
    #[test]
    fn poseidon_t_is_a_role_with_its_width() {
        let mut r1cs = make_r1cs::<Bn254>(vec![], 4);
        r1cs.custom_gates = vec![
            CustomGate { template_name: "PoseidonT".into(), parameters: vec![Bn254::from_int(5u64)] },
            CustomGate { template_name: "PoseidonT".into(), parameters: vec![Bn254::from_int(3u64)] },
        ];
        r1cs.custom_gates_uses = vec![
            CustomGateUse { id: 0, signals: vec![1; 345] },
            CustomGateUse { id: 1, signals: vec![1; 207] },
            CustomGateUse { id: 0, signals: vec![1; 345] },
        ];
        let cgi = get_custom_gates_info(&r1cs);
        assert_eq!(cgi.poseidon_t_widths, HashMap::from([(0, Bn254::from_int(5u64)), (1, Bn254::from_int(3u64))]));
        assert_eq!(cgi.n(GateRole::PoseidonT), 3);
        assert_eq!(cgi.role_id(GateRole::PoseidonT), None, "no single id, as for FFT4");
    }

    /// `Num2Bytes(nBits)` has a parameter, its bits: each is a gate id of the role, beside the
    /// wrap's `PoseidonT`, and the uses of every one of them are the role's.
    #[test]
    fn num2bytes_is_a_role_with_its_bits() {
        let gate = |name: &str, p: u64| CustomGate { template_name: name.into(), parameters: vec![Bn254::from_int(p)] };
        let mut r1cs = make_r1cs::<Bn254>(vec![], 4);
        r1cs.custom_gates = vec![gate("Num2Bytes", 64), gate("PoseidonT", 5), gate("Num2Bytes", 70)];
        r1cs.custom_gates_uses = vec![
            CustomGateUse { id: 0, signals: vec![1; 5] },
            CustomGateUse { id: 2, signals: vec![1; 6] },
            CustomGateUse { id: 1, signals: vec![1; 345] },
            CustomGateUse { id: 0, signals: vec![1; 5] },
        ];
        let cgi = get_custom_gates_info(&r1cs);
        assert_eq!(cgi.range_check_bits, HashMap::from([(0, Bn254::from_int(64u64)), (2, Bn254::from_int(70u64))]));
        assert_eq!((cgi.n(GateRole::RangeCheck), cgi.n(GateRole::PoseidonT)), (3, 1));
        assert_eq!(cgi.role_id(GateRole::RangeCheck), None, "no single id, as for PoseidonT");
    }

    /// The key is `{:x}` of each canonical coefficient, as it always was for Goldilocks.
    #[test]
    fn ckey_spells_the_coefficients_in_hex() {
        let c = PlonkConstraint {
            wires: [1, 2, 3],
            coeffs: [
                Goldilocks::ZERO,
                Goldilocks::ONE,
                Goldilocks::new(0xab),
                Goldilocks::NEG_ONE,
                Goldilocks::new(16),
            ],
        };
        assert_eq!(ckey(&c), "0,1,ab,ffffffff00000000,10");
        let c = PlonkConstraint {
            wires: [1, 2, 3],
            coeffs: [Bn254::ZERO, Bn254::ONE, Bn254::ZERO, Bn254::ZERO, Bn254::NEG_ONE],
        };
        assert_eq!(ckey(&c), "0,1,0,0,30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000");
    }
}
