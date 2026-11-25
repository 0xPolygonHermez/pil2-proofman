//! Exact security check of a solved low-degree test. The solver (alpha scan, query counts,
//! grinding) is untrusted and may use floats; this re-derives the bits in exact arithmetic,
//! rounding against the prover, from the schedule the verifier will run.
//!
//! Only query-message grinding is credited, as the protocol only grinds there.

pub mod exact;
pub mod schedule;

use super::pcs::LowDegreeTest;
use crate::types::stark_struct::{LowDegreeTest as Ldt, StarkStruct};
use exact::{goldilocks_cubic_field_size, with_grinding, ExactRegime, Rational};
use schedule::Params;
use std::fmt;

/// The security target, in bits.
pub const TARGET_BITS: u32 = 128;

#[derive(Clone, Debug)]
pub struct SecurityReport {
    /// `FRI/JBR`, `STIR/JBR`, ...
    pub config: String,
    /// Bits under BCHKS25 Theorem 4.2 (Corollary 1.4 in the unique-decoding regime), the minimum over the phases.
    pub bchks25_bits: Rational,
    /// ⌊`bchks25_bits`⌋.
    pub min_bits_floor: u32,
    /// The phase that sets `bchks25_bits`.
    pub weakest_phase: String,
}

impl SecurityReport {
    /// `security: BCHKS25 X bits`.
    pub fn summary(&self) -> String {
        format!("security: BCHKS25 {} bits", self.bchks25_bits)
    }
}

#[derive(Debug)]
pub enum SecurityError {
    /// The schedule is inconsistent with itself or the stark struct.
    Schedule(String),
    /// The regime's gap is not admissible at some round's rate.
    Gap(String),
    /// The BCHKS25 bits are under the target.
    BelowTarget { config: String, bits: Rational, weakest_phase: String },
}

impl fmt::Display for SecurityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecurityError::Schedule(s) => write!(f, "inconsistent low-degree-test schedule: {s}"),
            SecurityError::Gap(s) => write!(f, "regime gap not admissible: {s}"),
            SecurityError::BelowTarget { config, bits, weakest_phase } => write!(
                f,
                "{config} reaches only {bits} bits under BCHKS25 at {weakest_phase}, the target is {TARGET_BITS}"
            ),
        }
    }
}

impl std::error::Error for SecurityError {}

/// The bits of every phase of the schedule (BCHKS25), `(label, bits)`.
fn phases(p: &Params, regime: &ExactRegime) -> Result<Vec<(String, Rational)>, SecurityError> {
    let q = goldilocks_cubic_field_size();
    let m = p.folding.len();
    let rounds = if p.stir { m } else { m + 1 };
    for i in 0..rounds {
        if !regime.admits(p.mu(i)) {
            return Err(SecurityError::Gap(format!("round {i}, rate 2^-{}", p.mu(i))));
        }
    }
    let bits = |e: Rational| e.security_bits();
    let mut out = Vec::new();
    // Batching of the DEEP polynomial on the base code.
    if p.batch > 1 {
        out.push(("batching".to_string(), bits(regime.error_powers(p.mu(0), 1 << p.log_degrees[0], p.batch, &q))));
    }
    if !p.stir {
        // FRI: one rate throughout.
        let mu = p.mu(0);
        for i in 0..m {
            let e = regime.error_powers(mu, 1 << p.log_degrees[i + 1], 1 << p.folding[i], &q);
            out.push((format!("fold(i={i})"), bits(e)));
        }
        let e = with_grinding(regime.one_minus_delta(mu).pow(p.queries[0]), p.grinding[0]);
        out.push(("query".to_string(), bits(e)));
        return Ok(out);
    }
    // STIR (numOodSamples = 1 is fixed by the protocol, stark_info.hpp). Terms inside a phase add;
    // phases are separate rounds and the report takes the min of their bits (RBR / Fiat-Shamir convention).
    let fold = |i: usize| regime.error_powers(p.mu(i), 1 << (p.log_degrees[i] - p.folding[i]), 1 << p.folding[i], &q);
    out.push(("fold(i=0)".to_string(), bits(fold(0))));
    for i in 1..m {
        let s = p.ood[i - 1];
        // OOD: (ℓ²/2)·(2^{mᵢ}/(|F| − |Lᵢ|))^s.
        let l = Rational::int(regime.list_size(p.mu(i)));
        let degree = Rational::pow2(p.log_degrees[i] as i64);
        let domain = Rational::pow2(p.log_domains[i] as i64);
        let ood = l.clone() * l / Rational::int(2) * (degree / (q.clone() - domain)).pow(s);
        out.push((format!("ood(i={i})"), bits(ood)));
        // Shift: the prover draws the combination and fold challenges before the proof of work,
        // so grinding only protects the query-miss term.
        let first = regime.one_minus_delta(p.mu(i - 1)).pow(p.queries[i - 1]);
        let second = regime.error_powers(p.mu(i), 1 << p.log_degrees[i], p.queries[i - 1] + s, &q);
        let e = with_grinding(first, p.grinding[i - 1]) + second + fold(i);
        out.push((format!("shift(i={i})"), bits(e)));
    }
    let e = with_grinding(regime.one_minus_delta(p.mu(m - 1)).pow(p.queries[m - 1]), p.grinding[m - 1]);
    out.push(("final".to_string(), bits(e)));
    Ok(out)
}

/// Re-derive the bits of validated `params` under `regime`. Report only: nothing is gated here.
fn report_params(params: &Params, regime: &ExactRegime) -> Result<SecurityReport, SecurityError> {
    let phases = phases(params, regime)?;
    let (weakest_phase, bits) = phases.into_iter().min_by(|a, b| a.1.cmp(&b.1)).expect("a query phase always exists");
    let min_bits_floor = bits.floor_u32();
    Ok(SecurityReport {
        config: format!("{}/{}", if params.stir { "STIR" } else { "FRI" }, regime.name()),
        bchks25_bits: bits,
        min_bits_floor,
        weakest_phase,
    })
}

/// Out-of-range counts saturate, which `schedule::validate` refuses.
fn u32s<T: Copy + TryInto<u32>>(v: &[T]) -> Vec<u32> {
    v.iter().map(|&x| x.try_into().unwrap_or(u32::MAX)).collect()
}

/// The schedule of a stark struct, with the counts of `queries`/`grinding`/`ood` (all of them
/// are the solver's, or a built key's).
fn params_of(ss: &StarkStruct, queries: Vec<u64>, grinding: Vec<u32>, ood: Vec<u64>, batch: u64) -> Params {
    let (stir, folding, log_degrees, log_domains) = match &ss.low_degree_test {
        Ldt::Fri(f) => (false, &f.folding_factors, &f.log_degrees, &f.log_domain_sizes),
        Ldt::Stir(s) => (true, &s.folding_factors, &s.log_degrees, &s.log_domain_sizes),
    };
    Params {
        stir,
        n_bits: ss.n_bits as u32,
        n_bits_ext: ss.n_bits_ext as u32,
        folding: u32s(folding),
        log_degrees: u32s(log_degrees),
        log_domains: u32s(log_domains),
        queries,
        grinding,
        ood,
        batch,
    }
}

/// The query and grinding counts a stark struct already carries.
fn struct_counts(ss: &StarkStruct) -> (Vec<u64>, Vec<u32>) {
    match &ss.low_degree_test {
        Ldt::Fri(f) => (vec![f.num_queries as u64], vec![f.grinding_bits_queries as u32]),
        Ldt::Stir(s) => (s.num_queries.iter().map(|&t| t as u64).collect(), u32s(&s.grinding_bits_queries)),
    }
}

/// The regime of a solved test, from its gap (exact value of the solver's f64).
fn regime_of(identifier: &str, gap: f64, base_mu: u32) -> Result<ExactRegime, SecurityError> {
    match identifier {
        "UDR" => {
            let rho = Rational::pow2(-(base_mu as i64));
            let alpha = Rational::from_f64(gap) * Rational::int(20) / (Rational::int(1) - rho) - Rational::int(1);
            Ok(ExactRegime::Udr { alpha })
        }
        "JBR" => Ok(ExactRegime::Jbr { eta: Rational::from_f64(gap) }),
        other => Err(SecurityError::Gap(format!("unknown regime {other:?}"))),
    }
}

/// The report of a built key's stark struct (counts already in it). `ood_samples` of STIR rounds 1.. is 1,
/// as the prover runs. No gating.
pub fn report_built(
    ss: &proofman_common::StarkStruct,
    batch_size: u64,
    regime: &str,
    proximity_gap: f64,
) -> Result<SecurityReport, SecurityError> {
    use proofman_common::LowDegreeTest as Built;
    let (stir, folding, log_degrees, log_domains, queries, grinding, ood) = match &ss.low_degree_test {
        Built::Fri(f) => (
            false,
            &f.folding_factors,
            &f.log_degrees,
            &f.log_domain_sizes,
            vec![f.num_queries],
            vec![f.grinding_bits_queries as u32],
            vec![],
        ),
        Built::Stir(s) => (
            true,
            &s.folding_factors,
            &s.log_degrees,
            &s.log_domain_sizes,
            s.num_queries.clone(),
            u32s(&s.grinding_bits_queries),
            vec![1; s.folding_factors.len().saturating_sub(1)],
        ),
    };
    let params = Params {
        stir,
        n_bits: ss.n_bits as u32,
        n_bits_ext: ss.n_bits_ext as u32,
        folding: u32s(folding),
        log_degrees: u32s(log_degrees),
        log_domains: u32s(log_domains),
        queries,
        grinding,
        ood,
        batch: batch_size,
    };
    schedule::validate(&params).map_err(SecurityError::Schedule)?;
    let regime = regime_of(regime, proximity_gap, params.mu(0))?;
    report_params(&params, &regime)
}

/// The report of a solved low-degree test: the solver's counts, the stark struct's geometry. No gating.
pub fn report_low_degree_test(
    ss: &StarkStruct,
    ldt: &LowDegreeTest,
    batch_size: u64,
) -> Result<SecurityReport, SecurityError> {
    let (queries, grinding, ood) = match ldt {
        LowDegreeTest::Fri(f) => {
            let sp = f.security_params();
            (vec![sp.n_queries], vec![sp.grinding_bits_query], vec![])
        }
        LowDegreeTest::Stir(s) => {
            let sp = s.security_params();
            (sp.num_queries.clone(), sp.grinding_bits_queries.clone(), sp.num_ood_samples.clone())
        }
    };
    let params = params_of(ss, queries, grinding, ood, batch_size);
    schedule::validate(&params).map_err(SecurityError::Schedule)?;
    // A stark struct that already carries counts must agree with the solver's.
    let (have_q, have_g) = struct_counts(ss);
    let populated = have_q.iter().any(|&t| t != 0);
    if populated && (have_q != params.queries || have_g != params.grinding) {
        return Err(SecurityError::Schedule(format!(
            "stark struct counts {have_q:?}/{have_g:?} differ from the solver's {:?}/{:?}",
            params.queries, params.grinding
        )));
    }
    let regime = regime_of(ldt.regime_identifier(), ldt.proximity_gap(), params.mu(0))?;
    report_params(&params, &regime)
}

/// Check a solved low-degree test: the exact BCHKS25 bits must reach [`TARGET_BITS`].
pub fn check_low_degree_test(
    ss: &StarkStruct,
    ldt: &LowDegreeTest,
    batch_size: u64,
) -> Result<SecurityReport, SecurityError> {
    let report = report_low_degree_test(ss, ldt, batch_size)?;
    if report.bchks25_bits >= Rational::int(TARGET_BITS as i64) {
        Ok(report)
    } else {
        Err(SecurityError::BelowTarget {
            config: report.config,
            bits: report.bchks25_bits,
            weakest_phase: report.weakest_phase,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::stark_struct::{generate_stark_struct, LowDegreeTestKind, StarkSettings};

    /// Grinding scales only the query-miss term of a STIR shift phase: the challenges of the
    /// degree correction and the fold are drawn before the proof of work.
    #[test]
    fn stir_shift_grinding_credits_only_the_query_term() {
        let s = StarkSettings {
            low_degree_test: Some(LowDegreeTestKind::Stir),
            initial_blowup_factor: Some(2),
            initial_folding_factor: Some(3),
            ..Default::default()
        };
        let ss = generate_stark_struct(&s, 20, "blake3");
        let m = match &ss.low_degree_test {
            Ldt::Stir(s) => s.folding_factors.len(),
            _ => unreachable!(),
        };
        let at = |g: u32| {
            let p = params_of(&ss, vec![60; m], vec![g; m], vec![1; m - 1], 1);
            let regime = ExactRegime::Jbr { eta: Rational::new(1, 300) };
            let bits = phases(&p, &regime).unwrap();
            bits.into_iter().find(|(l, _)| l == "shift(i=1)").unwrap().1
        };
        // Saturated by the ungrindable terms; crediting g to them too would add 100 more bits.
        let gain = at(200) - at(100);
        assert!(gain < Rational::int(1), "grinding credited {gain} more bits to the shift phase");
    }

    /// A key whose gap is not positive (it would overstate the query bits) or whose regime is unknown is refused.
    #[test]
    fn bad_regime_gap_is_refused() {
        let ss = generate_stark_struct(&StarkSettings::default(), 20, "blake3");
        let p = params_of(&ss, vec![60], vec![0], vec![], 1);
        for regime in [ExactRegime::Jbr { eta: Rational::int(0) }, ExactRegime::Udr { alpha: Rational::int(-2) }] {
            assert!(matches!(phases(&p, &regime), Err(SecurityError::Gap(_))));
        }
        assert!(matches!(regime_of("XYZ", 0.01, 1), Err(SecurityError::Gap(_))));
    }
}
