//! The std over BN254 (pilfflonk/docs/README.md#compile-pil): the constants of
//! `pil2-components/lib/std/pil/bn254.pil` checked numerically, and
//! `tests/fixtures/std_bn254/connection.pil` compiled over BN254 and over Goldilocks, whose pilouts
//! must carry the roots of unity and the coset generator of that field.
//!
//! The `#[ignore]` tests compile the fixture with the compiler `PIL2C_EXEC` names, which must have
//! `--field` (the pinned one silently compiles over Goldilocks):
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p proofman-pilfflonk --features proofman-common/cpu-only \
//!     --test std_bn254 -- --include-ignored
//! ```

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, operand, SymbolType};
use pil2_pilout::pilout_proxy::PilOutProxy;
use proofman_fields::{Bn254, Field, PrimeField};
use proofman_pilfflonk::global_info::MAX_NBITS;
use proofman_pilfflonk::{oracle, BN254_R};

/// The standard 2^28-th root of unity of BN254, `5^((r − 1)/2^28)`.
const BN254_ROOT_28: &str = "19103219067921713944291392827692070036145651957329286315305642004821462161904";

/// `r − 1 = 2^28 · 3^2 · 13 · 29 · 983 · 11003 · 237073 · 405928799 · 1670836401704629 ·
/// 13818364434197438864469338081`, as (prime, exponent).
const BN254_R_MINUS_1_FACTORS: [(&str, u32); 10] = [
    ("2", 28),
    ("3", 2),
    ("13", 1),
    ("29", 1),
    ("983", 1),
    ("11003", 1),
    ("237073", 1),
    ("405928799", 1),
    ("1670836401704629", 1),
    ("13818364434197438864469338081", 1),
];

/// The airs of the fixture, in order: name, columns connected and opid of the connection.
const AIRS: [(&str, u64, u32); 3] = [("Offline", 3, 1), ("Online", 4, 2), ("Reused", 5, 3)];

fn big(decimal: &str) -> BigUint {
    BigUint::parse_bytes(decimal.as_bytes(), 10).expect("a decimal integer")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().expect("the repository root")
}

// ---------------------------------------------------------------------------------------------
// The constants of the std's field files
// ---------------------------------------------------------------------------------------------

/// The roots of unity and the coset generator one of the std's field files defines.
struct FieldConstants {
    prime: BigUint,
    /// `gen[i]`, of order `2^i`.
    gen: Vec<BigUint>,
    /// The coset generator.
    k: BigUint,
}

impl FieldConstants {
    /// `Goldilocks_Gen` and `Goldilocks_k`, over `2^64 − 2^32 + 1`.
    fn goldilocks() -> Self {
        let src = std_file("goldilocks.pil");
        Self {
            prime: BigUint::from(0xffff_ffff_0000_0001u64),
            gen: pil_int_array(&src, "Goldilocks_Gen"),
            k: pil_int(&src, "Goldilocks_k"),
        }
    }

    /// `Bn254_Gen` and `Bn254_k`, over `Bn254_r`.
    fn bn254() -> Self {
        let src = std_file("bn254.pil");
        Self { prime: pil_int(&src, "Bn254_r"), gen: pil_int_array(&src, "Bn254_Gen"), k: pil_int(&src, "Bn254_k") }
    }

    fn one(&self) -> BigUint {
        BigUint::from(1u32)
    }

    fn minus_one(&self) -> BigUint {
        &self.prime - 1u32
    }

    fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        a * b % &self.prime
    }

    fn pow(&self, base: &BigUint, exponent: &BigUint) -> BigUint {
        base.modpow(exponent, &self.prime)
    }

    fn pow_u64(&self, base: &BigUint, exponent: u64) -> BigUint {
        self.pow(base, &BigUint::from(exponent))
    }
}

fn std_file(name: &str) -> String {
    let path = repo_root().join("pil2-components/lib/std/pil").join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The initializer of `const int <name>` in `src`, from `=` to `;`.
fn pil_initializer<'a>(src: &'a str, name: &str) -> &'a str {
    let declaration = format!("const int {name}");
    let start = src
        .match_indices(&declaration)
        .map(|(i, _)| i + declaration.len())
        .find(|&i| matches!(src.as_bytes()[i], b' ' | b'['))
        .unwrap_or_else(|| panic!("no `{declaration}`"));
    let rest = &src[start..];
    let eq = rest.find('=').expect("an initializer");
    let end = rest.find(';').expect("a `;`");
    rest[eq + 1..end].trim()
}

/// `const int <name> = <decimal>;`.
fn pil_int(src: &str, name: &str) -> BigUint {
    big(pil_initializer(src, name))
}

/// `const int <name>[<n>] = [<decimal>, …];`, checking `<n>`.
fn pil_int_array(src: &str, name: &str) -> Vec<BigUint> {
    let list = pil_initializer(src, name);
    let list = list.strip_prefix('[').and_then(|l| l.strip_suffix(']')).expect("an array literal");
    let values: Vec<BigUint> = list.split(',').map(|v| big(v.trim())).collect();
    assert!(src.contains(&format!("const int {name}[{}]", values.len())), "{name} declares its length");
    values
}

/// Miller–Rabin with the first 20 primes as bases: an error probability below `4^−20`, and
/// deterministic below `3.3·10^24`.
fn is_probable_prime(n: &BigUint) -> bool {
    const BASES: [u32; 20] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71];
    let one = BigUint::from(1u32);
    if *n < BigUint::from(2u32) {
        return false;
    }
    if let Some(&p) = BASES.iter().find(|&&p| n % p == BigUint::from(0u32)) {
        return *n == BigUint::from(p);
    }
    let n_minus_1 = n - 1u32;
    let s = n_minus_1.trailing_zeros().expect("n > 1");
    let d = &n_minus_1 >> s;
    BASES.iter().all(|&a| {
        let mut x = BigUint::from(a).modpow(&d, n);
        if x == one || x == n_minus_1 {
            return true;
        }
        for _ in 1..s {
            x = &x * &x % n;
            if x == n_minus_1 {
                return true;
            }
        }
        false
    })
}

/// The distinct primes of `r − 1`, after checking the factorization.
fn bn254_r_minus_1_primes(f: &FieldConstants) -> Vec<BigUint> {
    let mut product = BigUint::from(1u32);
    for (p, e) in BN254_R_MINUS_1_FACTORS {
        let p = big(p);
        assert!(is_probable_prime(&p), "{p} is prime");
        product *= p.pow(e);
    }
    assert_eq!(product, f.minus_one(), "the factorization of r − 1");
    BN254_R_MINUS_1_FACTORS.iter().map(|(p, _)| big(p)).collect()
}

#[test]
fn bn254_r_is_the_scalar_field_of_bn254() {
    let f = FieldConstants::bn254();
    assert_eq!(f.prime, big(BN254_R));
    assert!(is_probable_prime(&f.prime));
}

#[test]
fn five_is_the_smallest_non_residue_and_generates_the_multiplicative_group() {
    let f = FieldConstants::bn254();
    let half = f.minus_one() >> 1u32;
    for residue in [2u32, 3, 4] {
        assert_eq!(f.pow(&BigUint::from(residue), &half), f.one(), "{residue} is a square");
    }
    let five = BigUint::from(5u32);
    assert_eq!(f.pow(&five, &half), f.minus_one(), "5 is not a square");
    for q in bn254_r_minus_1_primes(&f) {
        assert_ne!(f.pow(&five, &(f.minus_one() / &q)), f.one(), "5 has order r − 1, not a divisor of (r − 1)/{q}");
    }
}

#[test]
fn bn254_gen_are_the_roots_of_unity_of_order_a_power_of_two() {
    let f = FieldConstants::bn254();
    // r − 1 = 2^28 · odd: one generator per subgroup of order 2^0 … 2^28.
    assert_eq!(f.minus_one().trailing_zeros(), Some(MAX_NBITS));
    assert_eq!(f.gen.len() as u64, MAX_NBITS + 1);
    for (i, g) in f.gen.iter().enumerate() {
        assert!(*g < f.prime, "Bn254_Gen[{i}] is canonical");
        assert_eq!(*g, f.pow(&BigUint::from(5u32), &(f.minus_one() >> i)), "Bn254_Gen[{i}] = 5^((r − 1)/2^{i})");
        assert_eq!(f.pow_u64(g, 1 << i), f.one(), "Bn254_Gen[{i}]^(2^{i}) = 1");
        if i > 0 {
            assert_eq!(f.pow_u64(g, 1 << (i - 1)), f.minus_one(), "Bn254_Gen[{i}]^(2^{}) = r − 1", i - 1);
            assert_eq!(f.mul(g, g), f.gen[i - 1], "Bn254_Gen[{i}]^2 = Bn254_Gen[{}]", i - 1);
        }
        // The std's subgroups are the domains of pilfflonk.
        assert_eq!(g, oracle::omega(i as u32).unwrap().as_biguint(), "Bn254_Gen[{i}] is the oracle's ω");
    }
    assert_eq!(f.gen[0], f.one());
    assert_eq!(f.gen[28], big(BN254_ROOT_28));
}

#[test]
fn bn254_has_the_roots_of_unity_and_the_coset_generator_of_the_std() {
    let f = FieldConstants::bn254();
    assert_eq!(Bn254::TWO_ADICITY as u64, MAX_NBITS);
    assert_eq!(Bn254::W.len(), f.gen.len());
    for (i, (w, g)) in Bn254::W.iter().zip(&f.gen).enumerate() {
        assert_eq!(w.as_canonical_biguint(), *g, "Bn254::W[{i}] is Bn254_Gen[{i}]");
    }
    assert_eq!(Bn254::GENERATOR.as_canonical_biguint(), BigUint::from(5u32));
    assert_eq!(Bn254::GENERATOR.exp_power_of_2(Bn254::TWO_ADICITY).as_canonical_biguint(), f.k, "Bn254_k = 5^(2^28)");
}

#[test]
fn bn254_k_generates_cosets_disjoint_from_each_other_and_from_h() {
    let f = FieldConstants::bn254();
    let two_adic = BigUint::from(1u32) << MAX_NBITS;
    let m = f.minus_one() >> MAX_NBITS;
    assert!(m.bit(0) && m.bits() == 226, "m = (r − 1)/2^28 is odd and above 2^225");
    assert_eq!(f.k, f.pow(&BigUint::from(5u32), &two_adic), "Bn254_k = 5^(2^28)");

    // k has order exactly m, so k^d is in the subgroup H of order 2^28 (and in its subgroups, of
    // order N | 2^28) only for m | d: the cosets k^j·H, 0 ≤ j < m, are disjoint.
    assert_eq!(f.pow(&f.k, &m), f.one());
    for q in bn254_r_minus_1_primes(&f).into_iter().skip(1) {
        assert_ne!(f.pow(&f.k, &(&m / &q)), f.one(), "the order of k is not a divisor of m/{q}");
    }

    // And directly, for far more cosets than the std's default ARRAY_SIZE (750) of ks:
    // (k^d)^(2^28) ≠ 1, that is, k^d ∉ H.
    let k_to_the_2_adic = f.pow(&f.k, &two_adic);
    let mut acc = f.one();
    for d in 1..=1u32 << 16 {
        acc = f.mul(&acc, &k_to_the_2_adic);
        assert_ne!(acc, f.one(), "k^{d} is in H");
    }
}

// ---------------------------------------------------------------------------------------------
// The fixture compiled over each field (needs PIL2C_EXEC)
// ---------------------------------------------------------------------------------------------

const FIXTURE: &str = "pilfflonk/tests/fixtures/std_bn254/connection.pil";

/// Runs `PIL2C_EXEC` on the fixture from the repository root, with `--field <field>` if given.
fn run_pil2com(out: &Path, field: Option<&str>) -> Output {
    let compiler = std::env::var("PIL2C_EXEC")
        .expect("PIL2C_EXEC must name a pil2com that has `--field` (e.g. <pil2-compiler>/src/pil.js)");
    let mut cmd = Command::new(compiler);
    cmd.current_dir(repo_root()).arg(FIXTURE).arg("-I").arg("pil2-components/lib/std/pil").arg("-o").arg(out);
    if let Some(field) = field {
        cmd.arg("--field").arg(field);
    }
    cmd.output().expect("PIL2C_EXEC runs")
}

fn compile(name: &str, field: Option<&str>) -> pb::PilOut {
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("std_bn254.{name}.pilout"));
    let output = run_pil2com(&out, field);
    assert!(output.status.success(), "pil2com failed:\n{}", String::from_utf8_lossy(&output.stdout));
    PilOutProxy::new(out.to_str().expect("a UTF-8 path")).expect("a pilout").pilout
}

/// The values of the fixed column `name` of air `air_id` (one of `index`, if it is an array).
fn fixed_col(pilout: &pb::PilOut, air_id: u32, name: &str, index: u32) -> Vec<BigUint> {
    let symbol = pilout
        .symbols
        .iter()
        .find(|s| s.r#type == SymbolType::FixedCol as i32 && s.air_id == Some(air_id) && s.name == name)
        .unwrap_or_else(|| panic!("air {air_id} has no fixed column {name}"));
    assert!(index < symbol.lengths.first().copied().unwrap_or(1), "{name}[{index}] is in range");
    let air = &pilout.air_groups[0].airs[air_id as usize];
    air.fixed_cols[(symbol.id + index) as usize].values.iter().map(|v| BigUint::from_bytes_be(v)).collect()
}

/// Every constant in the expressions of air `air_id`.
fn constants(pilout: &pb::PilOut, air_id: u32) -> BTreeSet<BigUint> {
    use pb::expression::Operation;
    let operand = |o: &Option<pb::Operand>| match o.as_ref().and_then(|o| o.operand.as_ref()) {
        Some(operand::Operand::Constant(c)) => Some(BigUint::from_bytes_be(&c.value)),
        _ => None,
    };
    let air = &pilout.air_groups[0].airs[air_id as usize];
    air.expressions
        .iter()
        .flat_map(|e| match e.operation.as_ref().expect("an operation") {
            Operation::Add(op) => vec![operand(&op.lhs), operand(&op.rhs)],
            Operation::Sub(op) => vec![operand(&op.lhs), operand(&op.rhs)],
            Operation::Mul(op) => vec![operand(&op.lhs), operand(&op.rhs)],
            Operation::Neg(op) => vec![operand(&op.value)],
        })
        .flatten()
        .collect()
}

/// What the std must have put in the fixture's pilout over the field of `f`:
/// - `ID = [1, g, g^2, …]`, `g = gen[log2 N]`, the subgroup `H` of order `N` of each air;
/// - `k^j·ID` on the bus for the `j`-th column (constants `k^j` in the expressions, `j ≥ 1`), with
///   the `k^j` of each of the three places where std_connection.pil computes them;
/// - `CONN_2[j] = k^j·ID`, but for `a[0] ↔ b[1]`, and every `CONN_2` value distinct.
fn check_connections(pilout: &pb::PilOut, f: &FieldConstants) {
    assert_eq!(BigUint::from_bytes_be(&pilout.base_field), f.prime);
    let airs = &pilout.air_groups[0].airs;
    let names: Vec<&str> = airs.iter().map(|a| a.name.as_deref().unwrap()).collect();
    assert_eq!(names, AIRS.map(|(name, _, _)| name));

    for (air_id, (_, n_cols, opid)) in AIRS.into_iter().enumerate() {
        let air_id = air_id as u32;
        let n = airs[air_id as usize].num_rows.unwrap() as usize;
        let g = &f.gen[n.trailing_zeros() as usize];
        let id = fixed_col(pilout, air_id, "StdConnection.ID", 0);
        let powers: Vec<BigUint> = std::iter::successors(Some(f.one()), |x| Some(f.mul(x, g))).take(n).collect();
        assert_eq!(id, powers, "air {air_id}: ID = [1, g, g^2, …], connection #{opid}");
        assert_eq!(f.pow_u64(g, n as u64 / 2), f.minus_one(), "air {air_id}: g has order N = {n}");

        let constants = constants(pilout, air_id);
        for j in 1..n_cols {
            let k_j = f.pow_u64(&f.k, j);
            assert!(constants.contains(&k_j), "air {air_id}: k^{j} multiplies ID on the bus of connection #{opid}");
        }
    }

    let n = airs[1].num_rows.unwrap() as usize;
    let id = fixed_col(pilout, 1, "StdConnection.ID", 0);
    let conn: Vec<Vec<BigUint>> = (0..4).map(|j| fixed_col(pilout, 1, "StdConnection.CONN_2", j)).collect();
    let mut all = BTreeSet::new();
    for (j, column) in conn.iter().enumerate() {
        let k_j = f.pow_u64(&f.k, j as u64);
        for (row, value) in column.iter().enumerate() {
            let expected = match (j, row) {
                (0, 0) => f.mul(&f.k, &id[1]), // a[0] gets b[1]'s k·g
                (1, 1) => f.one(),             // b[1] gets a[0]'s 1
                _ => f.mul(&k_j, &id[row]),
            };
            assert_eq!(*value, expected, "CONN_2[{j}][{row}]");
            all.insert(value.clone());
        }
    }
    assert_eq!(all.len(), 4 * n, "the cosets k^j·H are disjoint");
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn over_bn254_the_connections_carry_bn254s_constants() {
    let pilout = compile("bn254", Some("bn254"));
    let f = FieldConstants::bn254();
    check_connections(&pilout, &f);

    let goldilocks = FieldConstants::goldilocks();
    for air_id in 0..AIRS.len() as u32 {
        assert!(!constants(&pilout, air_id).contains(&goldilocks.k), "air {air_id}: no Goldilocks_k");
    }
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn over_goldilocks_the_connections_still_carry_goldilocks_constants() {
    let pilout = compile("goldilocks", None);
    check_connections(&pilout, &FieldConstants::goldilocks());
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn over_another_field_nothing_is_compiled() {
    // The compiler knows only the fields the std has constants for, and refuses any other name
    // before it compiles: BN254's base field, a likely mistake for its scalar field, is not one.
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("std_bn254.fq.pilout");
    let _ = fs::remove_file(&out);
    let output = run_pil2com(&out, Some("bn254fq"));
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("unknown field \"bn254fq\""), "{stdout}");
    assert!(!out.exists());
}
