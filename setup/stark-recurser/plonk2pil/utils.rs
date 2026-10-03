//! Field utilities shared by all plonk2pil setup routines.

use std::collections::HashMap;

use proofman_fields::Field;

use super::field::PlonkField;
use super::r1cs::to_plonk::PlonkConstraint;
use super::r1cs_types::FixedPol;

/// log2(v): floor(log2(v)) for a u32.  Returns 0 for v == 0.
#[inline]
pub fn log2(v: u32) -> u32 {
    if v == 0 {
        0
    } else {
        31 - v.leading_zeros()
    }
}

/// Compute n coset shift factors of `F`'s connection argument: K, K^2, ..., K^n.
pub fn get_ks<F: PlonkField>(n: usize) -> Vec<F> {
    let mut ks = Vec::with_capacity(n);
    if n == 0 {
        return ks;
    }
    ks.push(F::K);
    for i in 1..n {
        let prev = ks[i - 1];
        ks.push(prev * F::K);
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
///
/// The identity permutation is `F`'s: [`PlonkField::K`] and [`PlonkField::ROOTS_OF_UNITY`], the
/// std's constants. Panics if `F` has no `2^n_bits`-th root of unity.
pub fn build_s_polynomials<F: PlonkField>(
    n_cols: usize,
    n: usize,
    n_bits: usize,
    r: usize,
    s_map: &[Vec<u32>],
) -> Vec<Vec<F>> {
    let gen = *F::ROOTS_OF_UNITY
        .get(n_bits)
        .unwrap_or_else(|| panic!("{} has no 2^{n_bits}-th root of unity: the air is too tall", F::PRIME));
    let ks = get_ks::<F>(n_cols - 1);
    let mut sv: Vec<Vec<F>> = (0..n_cols).map(|_| vec![F::ZERO; n]).collect();
    let mut w = F::ONE;
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        sv[0][i] = w;
        for j in 1..n_cols {
            sv[j][i] = w * ks[j - 1];
        }
        w *= gen;
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

/// Number of dedicated rows needed to bind every R1CS public signal into the
/// connection band.
pub fn public_rows(n_publics: u32, n_cols: usize) -> usize {
    (n_publics as usize).div_ceil(n_cols)
}

/// Append one connection-covered occurrence of each R1CS public signal.
///
/// Circom numbers the constant-one wire as 0 and public wires as
/// `1..=n_publics`. The matching PIL constraints bind these cells to
/// `publics[0..n_publics]`; the connection argument then binds every other
/// occurrence of the same signal in the Plonk/custom-gate trace.
pub fn bind_public_signals(s_map: &mut [Vec<u32>], first_row: usize, n_publics: u32, n_cols: usize) {
    assert!(n_cols > 0 && n_cols <= s_map.len());
    for public in 0..n_publics as usize {
        let col = public % n_cols;
        let row = first_row + public / n_cols;
        assert_eq!(s_map[col][row], 0, "public binding cell a[{col}] at row {row} is already occupied");
        s_map[col][row] = public as u32 + 1;
    }
}

/// The nine `constFFT` values of an `FFT4` gate from its parameters `[firstW, incW, scale, type]`:
/// slots 0..6 for a radix-4 step (`type` 4) and 6..9 for a radix-2 one (`type` 2), the others zero.
/// Every family's air reads them in this order; where each slot lives in `C[]` is the family's.
pub fn fft4_constants<F: Field>(params: &[F]) -> [F; 9] {
    let (first_w, inc_w, scale, fft_type) = (params[0], params[1], params[2], params[3]);
    let fw2 = first_w * first_w;
    let mut c = [F::ZERO; 9];
    if fft_type == F::TWO.double() {
        c[0] = scale;
        c[1] = scale * fw2;
        c[2] = scale * first_w;
        c[3] = scale * first_w * fw2;
        c[4] = scale * first_w * inc_w;
        c[5] = scale * first_w * fw2 * inc_w;
    } else if fft_type == F::TWO {
        c[6] = scale;
        c[7] = scale * first_w;
        c[8] = scale * first_w * inc_w;
    } else {
        panic!("Invalid FFT4 type: {fft_type}");
    }
    c
}

pub fn build_fixed_pols<F: Clone>(airgroup_name: &str, cv: &[Vec<F>], sv: &[Vec<F>]) -> Vec<FixedPol<F>> {
    let mut pols = Vec::with_capacity(cv.len() + sv.len());
    for (k, cv_values) in cv.iter().enumerate() {
        pols.push(FixedPol { name: format!("{}.C", airgroup_name), index: k, values: cv_values.clone() });
    }
    for (j, sv_values) in sv.iter().enumerate() {
        pols.push(FixedPol { name: format!("{}.S", airgroup_name), index: j, values: sv_values.clone() });
    }
    pols
}

/// Per-row bookkeeping for the plonk "piggyback" band, shared by both compressor setups.
///
/// The band is a fixed list of 3-cell plonk gates: gate `g` occupies `a[3g..3g+2]`. Which
/// gates actually fire on a row is decided by the gate's selector in `compressor.pil`;
/// which gates the setup is allowed to fill must be exactly that set, or a constraint gets
/// placed into a gate that is never evaluated (silently unenforced — unsound) or into cells
/// a custom gate already owns (a real overlap). [`PlonkBand`] makes both a hard error:
/// each row declares its allowed-gate mask via [`PlonkBand::allow`] (the mirror of the PIL
/// selectors), and every placement goes through [`PlonkBand::put`].
pub struct PlonkBand {
    allowed: Vec<u16>,
    written: Vec<u16>,
}

impl PlonkBand {
    pub fn new(n: usize) -> Self {
        Self { allowed: vec![0u16; n], written: vec![0u16; n] }
    }

    /// Declare the gates whose PIL selector fires on `row` (bit `g` = gate `g`).
    pub fn allow(&mut self, row: usize, mask: u16) {
        self.allowed[row] = mask;
    }

    /// Place plonk constraint `c` into gate `g` of `row`.
    pub fn put<F>(&mut self, s_map: &mut [Vec<u32>], row: usize, g: usize, c: &PlonkConstraint<F>) {
        assert!(
            self.allowed[row] & (1 << g) != 0,
            "plonk gate {g} placed at row {row}, but its PIL selector does not fire there \
             (allowed mask {:#012b}) — the constraint would be unenforced",
            self.allowed[row]
        );
        if self.written[row] & (1 << g) == 0 {
            // First write to this gate: its cells must still be free, i.e. the gate does not
            // overlap the cells the row's custom gate owns. Later writes to the same gate are
            // refinements of an already-placed constraint with the same selector constants.
            for j in 0..3 {
                assert_eq!(
                    s_map[3 * g + j][row],
                    0,
                    "plonk gate {g} at row {row} overlaps a custom-gate cell a[{}]",
                    3 * g + j
                );
            }
            self.written[row] |= 1 << g;
        }
        s_map[3 * g][row] = c.wires[0];
        s_map[3 * g + 1][row] = c.wires[1];
        s_map[3 * g + 2][row] = c.wires[2];
    }
}

#[cfg(test)]
mod tests {
    use super::{bind_public_signals, fft4_constants, public_rows};
    use proofman_fields::Goldilocks;

    /// `[firstW, incW, scale, type]` = `[3, 5, 7, type]`: small enough that every constant is the
    /// integer it is in any field.
    fn fft4(fft_type: u64) -> Vec<u64> {
        let p = [3, 5, 7, fft_type].map(Goldilocks::new);
        fft4_constants(&p).iter().map(proofman_fields::PrimeField64::as_canonical_u64).collect()
    }

    #[test]
    fn fft4_constants_fill_the_slots_of_their_type() {
        // scale, scale·w², scale·w, scale·w³, scale·w·inc, scale·w³·inc
        assert_eq!(fft4(4), [7, 63, 21, 189, 105, 945, 0, 0, 0]);
        // scale, scale·w, scale·w·inc, in the last three slots
        assert_eq!(fft4(2), [0, 0, 0, 0, 0, 0, 7, 21, 105]);
    }

    #[test]
    #[should_panic(expected = "Invalid FFT4 type: 3")]
    fn fft4_constants_refuse_another_type() {
        fft4(3);
    }

    #[test]
    fn public_signals_are_appended_in_connection_rows() {
        let mut s_map = vec![vec![0; 8]; 4];

        assert_eq!(public_rows(0, 4), 0);
        assert_eq!(public_rows(5, 4), 2);
        bind_public_signals(&mut s_map, 3, 5, 4);

        assert_eq!([s_map[0][3], s_map[1][3], s_map[2][3], s_map[3][3]], [1, 2, 3, 4]);
        assert_eq!(s_map[0][4], 5);
    }
}
