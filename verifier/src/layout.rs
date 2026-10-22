// The flat proof's shape (`Proofs::proof2pointer`), shared by the verifiers' size checks and the codec.

use alloc::vec;
use alloc::vec::Vec;

#[allow(unused)]
use num_traits::Float;

use crate::stir::{StirParams, StirVerifierInfo};
use crate::verifier::VerifierInfo;

/// The low-degree test of a [`ProofLayout`].
#[derive(Clone, Debug)]
pub enum LowDegreeLayout {
    /// `steps`: log₂ of each FRI domain, f₀'s first.
    Fri {
        steps: Vec<u64>,
        n_queries: u64,
    },
    Stir(StirParams),
}

/// The geometry of a flat proof.
#[derive(Clone, Debug)]
pub struct ProofLayout {
    pub n_stages: u32,
    pub n_constants: u64,
    pub n_evals: u64,
    pub n_bits_ext: u64,
    /// Committed columns per stage tree, cm1 … cm{n_stages+1}.
    pub num_vals: Vec<u64>,
    pub arity: u64,
    pub last_level_verification: u64,
    pub hash_commits: bool,
    pub ldt: LowDegreeLayout,
}

impl VerifierInfo {
    pub fn proof_layout(&self) -> ProofLayout {
        ProofLayout {
            n_stages: self.n_stages,
            n_constants: self.n_constants,
            n_evals: self.n_evals,
            n_bits_ext: self.n_bits_ext,
            num_vals: self.num_vals.clone(),
            arity: self.arity,
            last_level_verification: self.last_level_verification,
            hash_commits: self.hash_commits,
            ldt: LowDegreeLayout::Fri { steps: self.fri_steps.clone(), n_queries: self.n_fri_queries },
        }
    }
}

impl StirVerifierInfo {
    pub fn proof_layout(&self) -> ProofLayout {
        ProofLayout {
            n_stages: self.n_stages,
            n_constants: self.n_constants,
            n_evals: self.n_evals,
            n_bits_ext: self.n_bits_ext,
            num_vals: self.num_vals.clone(),
            arity: self.stir.arity,
            last_level_verification: self.stir.last_level_verification,
            hash_commits: self.stir.hash_commits,
            ldt: LowDegreeLayout::Stir(self.stir.clone()),
        }
    }
}

pub(crate) enum Segment {
    Data(usize),
    /// `n_queries` leaves of `width` words out of `2^leaf_bits`, then their paths of `n_levels`
    /// levels.
    Tree {
        width: usize,
        n_queries: usize,
        n_levels: usize,
        leaf_bits: u64,
    },
}

impl Segment {
    fn words(&self, arity: u64) -> usize {
        match *self {
            Segment::Data(n) => n,
            Segment::Tree { width, n_queries, n_levels, .. } => n_queries * (width + n_levels * sibling_words(arity)),
        }
    }
}

/// Words one Merkle path level takes: the arity − 1 sibling digests.
pub(crate) fn sibling_words(arity: u64) -> usize {
    ((arity - 1) * 4) as usize
}

/// Merkle path levels a query publishes (the GL case of the C++ `merkleProofLevels`).
pub(crate) fn merkle_proof_levels(n_bits: u64, arity: u64, last_level_verification: u64) -> usize {
    if n_bits == 0 {
        return 0;
    }
    let levels = ((n_bits as f64) / (arity as f64).log2()).ceil() as u64;
    levels.saturating_sub(last_level_verification) as usize
}

fn last_level_words(arity: u64, last_level_verification: u64) -> usize {
    match last_level_verification {
        0 => 0,
        llv => (arity.pow(llv as u32) * 4) as usize,
    }
}

/// The STIR section: roots, each tree's cosets, paths and last level, then β, p, nonces, Âns, Sh.
fn stir_segments(params: &StirParams, segs: &mut Vec<Segment>) {
    let m = params.m();
    let last_level = last_level_words(params.arity, params.last_level_verification);
    segs.push(Segment::Data(4 * m));
    for i in 0..m {
        let k = params.folding_factors[i];
        let leaf_bits = params.log_domain_sizes[i] - k;
        segs.push(Segment::Tree {
            width: (1usize << k) * 3,
            n_queries: params.num_queries[i] as usize,
            n_levels: merkle_proof_levels(leaf_bits, params.arity, params.last_level_verification),
            leaf_bits,
        });
        segs.push(Segment::Data(last_level));
    }
    let hints: usize = (1..m).map(|i| 2 * (1 + params.num_queries[i - 1] as usize) * 3).sum();
    segs.push(Segment::Data(3 * (m - 1) + 3 * (1usize << params.log_degrees[m]) + m + hints));
}

/// Words of the STIR section (the GL case of the C++ `StarkInfo::stirProofSectionSize`).
pub fn stir_section_size_words(params: &StirParams) -> usize {
    let mut segs = Vec::new();
    stir_segments(params, &mut segs);
    segs.iter().map(|s| s.words(params.arity)).sum()
}

impl ProofLayout {
    /// The flat proof's segments; the first holds the public count and the publics.
    pub(crate) fn segments(&self, n_publics: usize) -> Vec<Segment> {
        let last_level = last_level_words(self.arity, self.last_level_verification);
        let tree = |width: u64, n_queries: u64, n_bits: u64| Segment::Tree {
            width: width as usize,
            n_queries: n_queries as usize,
            n_levels: merkle_proof_levels(n_bits, self.arity, self.last_level_verification),
            leaf_bits: n_bits,
        };

        let mut segs = vec![
            Segment::Data(1 + n_publics),
            Segment::Data(4 * (self.n_stages as usize + 1)),
            Segment::Data(3 * self.n_evals as usize),
        ];
        let t0 = match &self.ldt {
            LowDegreeLayout::Fri { n_queries, .. } => *n_queries,
            LowDegreeLayout::Stir(stir) => stir.num_queries[0],
        };
        let widths = core::iter::once(self.n_constants).chain(self.num_vals.iter().copied());
        for width in widths.take(self.n_stages as usize + 2) {
            segs.push(tree(width, t0, self.n_bits_ext));
            segs.push(Segment::Data(last_level));
        }
        match &self.ldt {
            LowDegreeLayout::Fri { steps, n_queries } => {
                let n = steps.len();
                segs.push(Segment::Data(4 * (n - 1)));
                for i in 1..n {
                    segs.push(tree((1 << (steps[i - 1] - steps[i])) * 3, *n_queries, steps[i]));
                    segs.push(Segment::Data(last_level));
                }
                // The final polynomial, then the grinding nonce.
                segs.push(Segment::Data(3 * (1usize << steps[n - 1]) + 1));
            }
            LowDegreeLayout::Stir(stir) => stir_segments(stir, &mut segs),
        }
        segs
    }

    /// The flat proof's size in u64 words, publics and their count excluded.
    pub fn flat_proof_words(&self) -> usize {
        self.segments(0).iter().skip(1).map(|s| s.words(self.arity)).sum()
    }

    /// The expected size in u64 words of the proof as a multiproof, publics and their count
    /// excluded: the flat proof without its siblings, the tag, and each tree's kept siblings for
    /// uniform query indices. A proof's own size varies with how its paths overlap; it never
    /// exceeds `flat_proof_words() + 1`.
    pub fn expected_multiproof_words(&self) -> f64 {
        let arity = self.arity as f64;
        // Expected distinct nodes among `q` uniform picks of `m`.
        let distinct = |m: f64, q: f64| m * (1.0 - (1.0 - 1.0 / m).powf(q));
        let mut words = 1.0;
        for seg in self.segments(0).iter().skip(1) {
            match *seg {
                Segment::Data(n) => words += n as f64,
                Segment::Tree { width, n_queries, n_levels, leaf_bits } => {
                    let q = n_queries as f64;
                    words += q * width as f64;
                    let mut nodes = (1u64 << leaf_bits) as f64;
                    for _ in 0..n_levels {
                        let parents = (nodes / arity).ceil();
                        words += 4.0 * (arity * distinct(parents, q) - distinct(nodes, q));
                        nodes = parents;
                    }
                }
            }
        }
        words
    }
}
