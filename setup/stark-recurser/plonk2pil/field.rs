//! The prime fields plonk2pil converts an r1cs over.
//!
//! The arithmetic is `proofman-fields`'s, [`Goldilocks`] and [`Bn128`] through [`Field`].
//! [`PlonkField`] adds what plonk2pil needs on top of it, and [`R1csPrime`] names the primes an r1cs
//! header can carry.

use std::fmt;
use std::fmt::Write as _;

use num_bigint::BigUint;
use proofman_fields::{Bn128, Goldilocks, PrimeField, PrimeField64};

/// A prime field plonk2pil converts an r1cs over.
///
/// A trait of its own because what it adds is plonk2pil's, not the field's: the prime an r1cs names,
/// and the constants the PIL std builds its connection argument from.
pub trait PlonkField: PrimeField {
    /// The prime, as an r1cs header names it.
    const PRIME: R1csPrime;

    /// The coset shift of the std's connection argument (`Goldilocks_k`, `Bn128_k`): the identity
    /// permutation puts `K^j·w^i` in row `i` of column `j`.
    const K: Self;

    /// `ROOTS_OF_UNITY[i]` is a primitive `2^i`-th root of unity: the std's `Goldilocks_Gen` and
    /// `Bn128_Gen`, which the connection argument's `w` is taken from.
    const ROOTS_OF_UNITY: &'static [Self];

    /// Appends the canonical value in lowercase hexadecimal, without leading zeros: `ckey`'s
    /// spelling of a coefficient.
    fn push_hex(&self, out: &mut String) {
        write!(out, "{:x}", self.as_canonical_biguint()).expect("writing to a String does not fail");
    }
}

/// The primes plonk2pil converts an r1cs over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R1csPrime {
    /// `2^64 − 2^32 + 1`, the STARK recursion's.
    Goldilocks,
    /// The order of BN128's groups (circom's `bn128`).
    Bn128,
}

impl fmt::Display for R1csPrime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Goldilocks => "Goldilocks",
            Self::Bn128 => "BN128",
        })
    }
}

/// The prime of `F`.
pub fn modulus<F: PrimeField>() -> BigUint {
    F::NEG_ONE.as_canonical_biguint() + 1u32
}

/// The std's `Goldilocks_Gen` (`pil2-components/lib/std/pil/goldilocks.pil`). Not
/// `Goldilocks::W`, whose roots are others from the 2-adic tower: the S columns are written with
/// these, and the PIL checks them against these.
const GOLDILOCKS_GEN: [Goldilocks; 33] = [
    Goldilocks::new(1),
    Goldilocks::new(18446744069414584320),
    Goldilocks::new(281474976710656),
    Goldilocks::new(18446744069397807105),
    Goldilocks::new(17293822564807737345),
    Goldilocks::new(70368744161280),
    Goldilocks::new(549755813888),
    Goldilocks::new(17870292113338400769),
    Goldilocks::new(13797081185216407910),
    Goldilocks::new(1803076106186727246),
    Goldilocks::new(11353340290879379826),
    Goldilocks::new(455906449640507599),
    Goldilocks::new(17492915097719143606),
    Goldilocks::new(1532612707718625687),
    Goldilocks::new(16207902636198568418),
    Goldilocks::new(17776499369601055404),
    Goldilocks::new(6115771955107415310),
    Goldilocks::new(12380578893860276750),
    Goldilocks::new(9306717745644682924),
    Goldilocks::new(18146160046829613826),
    Goldilocks::new(3511170319078647661),
    Goldilocks::new(17654865857378133588),
    Goldilocks::new(5416168637041100469),
    Goldilocks::new(16905767614792059275),
    Goldilocks::new(9713644485405565297),
    Goldilocks::new(5456943929260765144),
    Goldilocks::new(17096174751763063430),
    Goldilocks::new(1213594585890690845),
    Goldilocks::new(6414415596519834757),
    Goldilocks::new(16116352524544190054),
    Goldilocks::new(9123114210336311365),
    Goldilocks::new(4614640910117430873),
    Goldilocks::new(1753635133440165772),
];

impl PlonkField for Goldilocks {
    const PRIME: R1csPrime = R1csPrime::Goldilocks;

    /// The std's `Goldilocks_k`, `7^(2^32)`.
    const K: Self = Goldilocks::new(12275445934081160404);

    const ROOTS_OF_UNITY: &'static [Self] = &GOLDILOCKS_GEN;

    /// The word's own hex, without the detour through a `BigUint`: `ckey` spells every
    /// coefficient of every constraint this way, more than once.
    fn push_hex(&self, out: &mut String) {
        write!(out, "{:x}", self.as_canonical_u64()).expect("writing to a String does not fail");
    }
}

impl PlonkField for Bn128 {
    const PRIME: R1csPrime = R1csPrime::Bn128;

    /// The std's `Bn128_k`, `5^(2^28)`.
    const K: Self =
        match Bn128::from_decimal("5266228460530200451425464971825753823072228272503274930591399474110020095489") {
            Some(k) => k,
            None => panic!("Bn128_k is an element of Fr"),
        };

    /// The std's `Bn128_Gen`, which `Bn128::W` is.
    const ROOTS_OF_UNITY: &'static [Self] = &Bn128::W;
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use proofman_fields::{Field, QuotientMap};

    use super::*;

    /// The value of `const int <name> = <value>;` in a file of the PIL std, and of an array
    /// `const int <name>[n] = [<v>, ...];` element by element, as decimal strings.
    fn std_constant(file: &str, name: &str) -> Vec<String> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../pil2-components/lib/std/pil").join(file);
        let src = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let decl = format!("const int {name}");
        let at = src.find(&decl).unwrap_or_else(|| panic!("{file} declares no {name}")) + decl.len();
        let value = &src[at..][src[at..].find('=').expect("an initializer") + 1..];
        let value = &value[..value.find(';').expect("a terminated declaration")];
        value.trim().trim_start_matches('[').trim_end_matches(']').split(',').map(|v| v.trim().to_string()).collect()
    }

    fn decimal<F: PrimeField>(x: &F) -> String {
        x.as_canonical_biguint().to_string()
    }

    /// The S columns have to be built from the very constants the PIL's connection argument reads,
    /// or no permutation checks. Read off the std, not restated.
    fn assert_matches_the_std<F: PlonkField>(file: &str, k: &str, gen: &str) {
        assert_eq!(vec![decimal(&F::K)], std_constant(file, k), "{} K against {file}'s {k}", F::PRIME);
        let roots: Vec<String> = F::ROOTS_OF_UNITY.iter().map(decimal).collect();
        assert_eq!(roots, std_constant(file, gen), "{} roots of unity against {file}'s {gen}", F::PRIME);
    }

    #[test]
    fn goldilocks_constants_are_the_stds() {
        assert_matches_the_std::<Goldilocks>("goldilocks.pil", "Goldilocks_k", "Goldilocks_Gen");
    }

    #[test]
    fn bn128_constants_are_the_stds() {
        assert_matches_the_std::<Bn128>("bn128.pil", "Bn128_k", "Bn128_Gen");
        assert_eq!(vec![modulus::<Bn128>().to_string()], std_constant("bn128.pil", "Bn128_r"));
    }

    /// `ROOTS_OF_UNITY[i]` has order exactly `2^i`, and `K` is outside every `<w>` the trace can
    /// use, which is what keeps the columns' cosets disjoint.
    fn assert_roots_and_k_are_sound<F: PlonkField>() {
        for (i, w) in F::ROOTS_OF_UNITY.iter().enumerate() {
            assert_eq!(w.exp_power_of_2(i), F::ONE, "{} root {i}", F::PRIME);
            if i > 0 {
                assert_ne!(w.exp_power_of_2(i - 1), F::ONE, "{} root {i} is not primitive", F::PRIME);
            }
        }
        let max_bits = F::ROOTS_OF_UNITY.len() - 1;
        assert_ne!(F::K.exp_power_of_2(max_bits), F::ONE, "{} K is in <w>", F::PRIME);
    }

    #[test]
    fn roots_and_k_are_sound() {
        assert_roots_and_k_are_sound::<Goldilocks>();
        assert_roots_and_k_are_sound::<Bn128>();
    }

    /// The Goldilocks spelling is the one `ckey` always had, `{:x}` of the canonical word; the
    /// generic one agrees with it wherever both apply.
    #[test]
    fn hex_is_canonical_lowercase_without_leading_zeros() {
        for v in [0u64, 1, 0xab, 0xFFFF_FFFF_0000_0000] {
            let mut fast = String::new();
            Goldilocks::new(v).push_hex(&mut fast);
            assert_eq!(fast, format!("{v:x}"));
            let mut generic = String::new();
            write!(generic, "{:x}", Goldilocks::new(v).as_canonical_biguint()).unwrap();
            assert_eq!(fast, generic);
            let mut wide = String::new();
            Bn128::from_int(v).push_hex(&mut wide);
            assert_eq!(wide, format!("{v:x}"));
        }
        let mut s = String::new();
        Bn128::NEG_ONE.push_hex(&mut s);
        assert_eq!(s, "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000");
    }
}
