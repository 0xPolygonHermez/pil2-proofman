//! Field utilities shared by all plonk2pil setup routines.

use std::collections::HashMap;

use proofman_fields::Field;

use super::field::PlonkField;

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

/// Time-derived hex suffix for scratch file names.
pub fn rand_hex() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    format!("{:x}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64)
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
