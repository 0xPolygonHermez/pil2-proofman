//! For tests only (feature `test-ptau`): a powers-of-tau file with `τ = 1`, written here, instead
//! of downloading one or running snarkjs (plan N13).
//!
//! With `τ = 1` every power `[τ^i]₁` is the generator `G` of G1, and `[τ]₂` the generator of G2,
//! so the commitment of a fixed `f = Σ_j p_j(X^k)·X^j` is `f(1)·G = (Σ_j p_j(1))·G`, and `p_j(1)`
//! is the first row of column `j` (the domain starts at `ω^0 = 1`): a test knows the points in
//! advance, with no curve arithmetic. The ptau holds only what pilfflonk reads, sections 1 to 3,
//! as the C++ test ptau does (`pil2-stark/test/pilfflonk/pilfflonk_test_ptau.hpp`, whose `τ` is a
//! full-width scalar).

use std::path::Path;

/// BN254's base field modulus `q`, little-endian, in hex: byte by byte as it is stored.
const Q_LE: &str = "47fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430";

/// The generator of G1, `(1, 2)`, in Montgomery form (`c·2^256 mod q`), little-endian: as a
/// snarkjs ptau stores it. The same constants as `provers/starks-lib-c/tests/pilfflonk_srs.rs`;
/// the C++ reader checks that `[1]₁` is the generator, so a wrong digit fails every test.
const G1_MONTGOMERY_LE: [&str; 2] = [
    "9d0d8fc58d435dd33d0bc7f528eb780a2c4679786fa36e662fdf079ac1770a0e",
    "3a1b1e8b1b87baa67b168eeb51d6f114588cf2f0de46ddcc5ebe0f3483ef141c",
];

/// The generator of G2, `x.c0, x.c1, y.c0, y.c1`, in the same form.
const G2_MONTGOMERY_LE: [&str; 4] = [
    "2620bc02d1b5838e72017b493519ebdcdf1a81974726b8fb3b5096af41385719",
    "40614ca87d73b4afc4d802585add4360862fa052fc50e9096b7bea3a83f0fe14",
    "f6e96b889dfa9d61789b9ef597d27ffefe7d1b23621a9eff06429eaeeb7efd28",
    "ee5618c7565b0964bb3c7d3222f957dc76103533be35f9558264fd93e6a0a40d",
];

/// The bytes 64 hex digits spell, in the order they are written.
fn bytes(hex: &str) -> Vec<u8> {
    (0..hex.len()).step_by(2).filter_map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok()).collect()
}

/// A binfile as snarkjs and rapidsnark write them: type, version, then each section as (id, size,
/// bytes).
fn binfile(file_type: &[u8; 4], sections: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let mut out = file_type.to_vec();
    out.extend(1u32.to_le_bytes());
    out.extend((sections.len() as u32).to_le_bytes());
    for (id, contents) in sections {
        out.extend(id.to_le_bytes());
        out.extend((contents.len() as u64).to_le_bytes());
        out.extend(contents);
    }
    out
}

/// A ptau with `τ = 1` and `n_g1` powers `[τ^i]₁`: `n_g1` copies of G1's generator and two of
/// G2's. Its power is the smallest a snarkjs ptau of that many powers has, `2^(power+1) − 1 ≥
/// n_g1`, as the C++ test ptau's (the reader relies on the sizes of the sections only).
pub fn tau_one_ptau(n_g1: usize) -> Vec<u8> {
    let power = (1..usize::BITS - 1).find(|&p| (1usize << (p + 1)) > n_g1).unwrap_or(usize::BITS - 1);
    let mut header = 32u32.to_le_bytes().to_vec();
    header.extend(bytes(Q_LE));
    header.extend(power.to_le_bytes());
    header.extend(power.to_le_bytes());
    let g1: Vec<u8> = G1_MONTGOMERY_LE.iter().flat_map(|c| bytes(c)).collect();
    let g2: Vec<u8> = G2_MONTGOMERY_LE.iter().flat_map(|c| bytes(c)).collect();
    binfile(b"ptau", &[(1, header), (2, g1.repeat(n_g1)), (3, g2.repeat(2))])
}

/// Writes [`tau_one_ptau`] at `path`.
pub fn write_tau_one_ptau(path: &Path, n_g1: usize) -> std::io::Result<()> {
    std::fs::write(path, tau_one_ptau(n_g1))
}
