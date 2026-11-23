//! Structural consistency of a solved low-degree-test schedule: the degrees and domains step by
//! the folding factors, domain = degree + blowup, and the query and grinding counts are sane.

/// What the verifier runs: geometry from the stark struct, counts from the solver (or a built key).
#[derive(Clone, Debug)]
pub struct Params {
    pub stir: bool,
    pub n_bits: u32,
    pub n_bits_ext: u32,
    /// `kᵢ`, one per folding round.
    pub folding: Vec<u32>,
    /// `log₂ dᵢ`, rounds + 1 entries.
    pub log_degrees: Vec<u32>,
    /// `log₂|Lᵢ|`, rounds + 1 entries.
    pub log_domains: Vec<u32>,
    /// FRI: one entry (queries into f₀). STIR: one per round.
    pub queries: Vec<u64>,
    /// Grinding bits on the query message(s), as `queries`.
    pub grinding: Vec<u32>,
    /// STIR: out-of-domain samples of rounds 1..M-1.
    pub ood: Vec<u64>,
    pub batch: u64,
}

impl Params {
    /// log₂(1/ρᵢ) of round i's code.
    pub fn mu(&self, i: usize) -> u32 {
        self.log_domains[i] - self.log_degrees[i]
    }
}

pub fn validate(p: &Params) -> Result<(), String> {
    let m = p.folding.len();
    if p.stir && m == 0 {
        return Err("STIR needs at least one round".into());
    }
    if p.log_degrees.len() != m + 1 || p.log_domains.len() != m + 1 {
        return Err(format!(
            "{m} rounds need {} degrees and domains, got {} and {}",
            m + 1,
            p.log_degrees.len(),
            p.log_domains.len()
        ));
    }
    if p.batch == 0 {
        return Err("batch size must be at least 1".into());
    }
    if p.log_degrees[0] != p.n_bits || p.log_domains[0] != p.n_bits_ext {
        return Err(format!(
            "round 0 is degree 2^{} on domain 2^{}, the stark struct says nBits {} / nBitsExt {}",
            p.log_degrees[0], p.log_domains[0], p.n_bits, p.n_bits_ext
        ));
    }
    if p.n_bits_ext <= p.n_bits || p.n_bits_ext >= 32 {
        return Err(format!(
            "blowup must be positive and the domain below 2^32 (nBits {}, nBitsExt {})",
            p.n_bits, p.n_bits_ext
        ));
    }
    for (i, &k) in p.folding.iter().enumerate() {
        if k == 0 || k > p.log_degrees[i] {
            return Err(format!("round {i}: folding factor {k} outside 1..={}", p.log_degrees[i]));
        }
        if p.log_degrees[i + 1] != p.log_degrees[i] - k {
            return Err(format!(
                "round {i}: degree 2^{} does not fold by 2^{k} to 2^{}",
                p.log_degrees[i],
                p.log_degrees[i + 1]
            ));
        }
        // FRI keeps the rate: the domain folds like the degree. STIR halves the domain.
        let expect = if p.stir { p.log_domains[i].checked_sub(1) } else { p.log_domains[i].checked_sub(k) };
        if expect != Some(p.log_domains[i + 1]) {
            return Err(format!(
                "round {i}: domain 2^{} does not step to 2^{}",
                p.log_domains[i],
                p.log_domains[i + 1]
            ));
        }
        if p.log_domains[i + 1] <= p.log_degrees[i + 1] {
            return Err(format!(
                "round {}: domain 2^{} is not above degree 2^{}",
                i + 1,
                p.log_domains[i + 1],
                p.log_degrees[i + 1]
            ));
        }
    }
    let n_queries = if p.stir { m } else { 1 };
    if p.queries.len() != n_queries || p.grinding.len() != n_queries {
        return Err(format!(
            "expected {n_queries} query and grinding entries, got {} and {}",
            p.queries.len(),
            p.grinding.len()
        ));
    }
    if p.queries.contains(&0) {
        return Err("a round with zero queries".into());
    }
    if p.grinding.iter().any(|&g| g >= 64) {
        return Err("grinding of 64 bits or more does not fit the proof-of-work check".into());
    }
    if p.stir {
        if p.ood.len() != m - 1 {
            return Err(format!("expected {} out-of-domain sample counts, got {}", m - 1, p.ood.len()));
        }
        for i in 1..m {
            let g = p.queries[i - 1] + p.ood[i - 1];
            if g >= 1u64 << p.log_degrees[i] {
                return Err(format!("round {i}: quotients over {g} points but the degree is 2^{}", p.log_degrees[i]));
            }
        }
    }
    Ok(())
}
