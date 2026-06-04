//! Goldilocks field utilities shared by all plonk2pil setup routines.

use std::collections::HashMap;

use super::r1cs_types::FixedPol;

/// Goldilocks prime p = 2^64 - 2^32 + 1.
pub const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

/// Generator table: `GOLDILOCKS_GEN[n]` is a primitive 2^n-th root of unity mod p.
pub const GOLDILOCKS_GEN: [u64; 33] = [
    1,
    18446744069414584320,
    281474976710656,
    18446744069397807105,
    17293822564807737345,
    70368744161280,
    549755813888,
    17870292113338400769,
    13797081185216407910,
    1803076106186727246,
    11353340290879379826,
    455906449640507599,
    17492915097719143606,
    1532612707718625687,
    16207902636198568418,
    17776499369601055404,
    6115771955107415310,
    12380578893860276750,
    9306717745644682924,
    18146160046829613826,
    3511170319078647661,
    17654865857378133588,
    5416168637041100469,
    16905767614792059275,
    9713644485405565297,
    5456943929260765144,
    17096174751763063430,
    1213594585890690845,
    6414415596519834757,
    16116352524544190054,
    9123114210336311365,
    4614640910117430873,
    1753635133440165772,
];

/// Coset shift constant K used for building permutation polynomials.
pub const GOLDILOCKS_K: u64 = 12275445934081160404;

/// (a * b) mod p
#[inline]
pub fn mulp(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % GOLDILOCKS_P as u128) as u64
}

/// (a + b) mod p
#[inline]
pub fn addp(a: u64, b: u64) -> u64 {
    ((a as u128 + b as u128) % GOLDILOCKS_P as u128) as u64
}

/// (-v) mod p  =  (p - v) mod p
#[inline]
pub fn neg(v: u64) -> u64 {
    if v == 0 {
        0
    } else {
        GOLDILOCKS_P - v
    }
}

/// log2(v): floor(log2(v)) for a u32.  Returns 0 for v == 0.
#[inline]
pub fn log2(v: u32) -> u32 {
    if v == 0 {
        0
    } else {
        31 - v.leading_zeros()
    }
}

/// Compute n coset shift factors: K, K^2, ..., K^n.
pub fn get_ks(n: usize) -> Vec<u64> {
    let mut ks = Vec::with_capacity(n);
    if n == 0 {
        return ks;
    }
    ks.push(GOLDILOCKS_K);
    for i in 1..n {
        let prev = ks[i - 1];
        ks.push(mulp(prev, GOLDILOCKS_K));
    }
    ks
}

/// Build S permutation polynomials, matching the JS reference implementation.
///
/// JS only stores the **first** occurrence of each signal in `lastSignal` and
/// never updates it; every subsequent occurrence is swapped with that first
/// position.  This creates cycles in reverse-visit order, as opposed to
/// always updating `last` which creates forward-order cycles.  We must match
/// JS exactly so the `.fixed.bin` output byte-matches the golden reference.
///
/// * `n_cols`  — number of S columns (e.g. 27 for aggregation, 36 for compressor)
/// * `n`       — total rows (power of 2)
/// * `n_bits`  — log2(n)
/// * `r`       — number of used rows (connections iterated over `0..r`)
/// * `s_map`   — `s_map[col][row]` = signal id (0 = unused)
pub fn build_s_polynomials(n_cols: usize, n: usize, n_bits: usize, r: usize, s_map: &[Vec<u32>]) -> Vec<Vec<u64>> {
    let ks = get_ks(n_cols - 1);
    let mut sv: Vec<Vec<u64>> = (0..n_cols).map(|_| vec![0u64; n]).collect();
    let mut w = 1u64;
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        sv[0][i] = w;
        for j in 1..n_cols {
            sv[j][i] = mulp(w, ks[j - 1]);
        }
        w = mulp(w, GOLDILOCKS_GEN[n_bits]);
    }
    // JS: lastSignal is only set on the *first* occurrence of each signal.
    // Every later occurrence swaps with that first-seen position (not the most
    // recently seen one).  We replicate this by only inserting into `last`
    // inside the `else` branch.
    let mut last: HashMap<u32, (usize, usize)> = HashMap::new();
    for i in 0..r {
        for j in 0..n_cols {
            let sig = s_map[j][i];
            if sig != 0 {
                if let Some(&(lc, lr)) = last.get(&sig) {
                    let t = sv[lc][lr];
                    sv[lc][lr] = sv[j][i];
                    sv[j][i] = t;
                } else {
                    last.insert(sig, (j, i));
                }
            }
        }
    }
    sv
}

/// Compute the permuted initialSt for a Poseidon1_16 gate, mirroring the
/// `CustPoseidon1_16` input-ordering in circom:
///   key=(0,0) → initialSt = in
///   key=(1,0) → in[0..3]↔in[4..7] (swap halves of low 8 cells)
///   key=(0,1) → in[0..3]→pos 8..11, in[4..7]→pos 0..3, in[8..11]→pos 4..7
///   key=(1,1) → in[0..3]→pos 12..15, in[4..7]→0..3, in[8..11]→4..7, in[12..15]→8..11
/// For sponge mode (no key) this is identity.
pub fn cust_poseidon1_initial_state(input: &[u64], key: Option<&[u64]>) -> Vec<u64> {
    let mut p = vec![0u64; 16];
    match key {
        None => p.copy_from_slice(input),
        Some(k) => {
            let (k0, k1) = (k[0], k[1]);
            if k0 == 0 && k1 == 0 {
                p.copy_from_slice(input);
            } else if k0 == 1 && k1 == 0 {
                p[0..4].copy_from_slice(&input[4..8]);
                p[4..8].copy_from_slice(&input[0..4]);
                p[8..16].copy_from_slice(&input[8..16]);
            } else if k0 == 0 && k1 == 1 {
                p[0..4].copy_from_slice(&input[4..8]);
                p[4..8].copy_from_slice(&input[8..12]);
                p[8..12].copy_from_slice(&input[0..4]);
                p[12..16].copy_from_slice(&input[12..16]);
            } else {
                p[0..4].copy_from_slice(&input[4..8]);
                p[4..8].copy_from_slice(&input[8..12]);
                p[8..12].copy_from_slice(&input[12..16]);
                p[12..16].copy_from_slice(&input[0..4]);
            }
        }
    }
    p
}

pub fn build_fixed_pols(airgroup_name: &str, cv: &[Vec<u64>], sv: &[Vec<u64>]) -> Vec<FixedPol> {
    let mut pols = Vec::with_capacity(cv.len() + sv.len());
    for (k, cv_values) in cv.iter().enumerate() {
        pols.push(FixedPol { name: format!("{}.C", airgroup_name), index: k, values: cv_values.clone() });
    }
    for (j, sv_values) in sv.iter().enumerate() {
        pols.push(FixedPol { name: format!("{}.S", airgroup_name), index: j, values: sv_values.clone() });
    }
    pols
}
