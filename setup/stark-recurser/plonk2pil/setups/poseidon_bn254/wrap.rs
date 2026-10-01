//! The final SNARK wrap's setup, over BN254 (`pil/poseidon_bn254/wrap.pil`), in layout L1 (M45):
//! 9 wires, all of them in the std's connection; 3 PLONK gates a row on the row's one coefficient
//! set; and each `PoseidonT(5)` use a band of 69 rows, one round a row on a[0..4], where gate 2 is
//! the only PLONK gate on.
//!
//! The rows, in order: the bands, one per use in the r1cs's order; the PLONK rows; the public rows.
//! A band is the use's signals as they are, 5 a row: `in`, then `im[0..67]`, then `out`. The gate
//! exposes every round's state, so the map covers the whole band and there is no gate band to
//! expand: the trace is the witness gathered through the map, and nothing else.
//!
//! **PLONK placement.** The gates of a row share its coefficients, so the constraints are grouped
//! by coefficient set, as the compressor groups them by `ckey`, and a group of `n` takes `n / 3`
//! rows of 3. Its `n % 3` left over would take one more row; a band row has room for one
//! constraint of any set, on gate 2. So the band rows go first to the groups whose left over is 1
//! (one band row saves a row), then to those whose left over is 2 (two save one), then three at a
//! time to any group (three save one). On a group's last row, a gate with no constraint of its own
//! repeats the row's last one: a cell the map leaves out reads 0, and a gate on zeros reads `qC`.

use std::collections::HashMap;

use anyhow::{bail, ensure, Result};
use proofman_common::hash_family::{lookup_gate, GateRole, BN254_WRAP_FAMILY};
use proofman_fields::{Bn254, Field, QuotientMap};

use super::constants::{is_full_round, ROUNDS, ROUND_CONSTANTS, WIDTH};
use super::{gen_pil_str, PilTemplateParams};
use crate::plonk2pil::merge_copies::{apply_remap_to_s_map, r1cs2plonk_merged, verify_merge_soundness};
use crate::plonk2pil::r1cs::to_plonk::{get_custom_gates_info, PlonkConstraint, PLONK_COEFFS};
use crate::plonk2pil::r1cs::types::{CustomGateUse, FixedPol, PlonkOptions, R1csFile, SetupResult};
use crate::plonk2pil::utils::{bind_public_signals, build_fixed_pols, build_s_polynomials, public_rows, PlonkBand};

/// The witness columns `a[0..8]`, every one in the connection.
const N_WIRES: usize = 9;

/// PLONK gates a row: gate `g` on `a[3g..3g+2]`.
const GATES_PER_ROW: usize = 3;

/// The PLONK gate that is on in a band: gates 0 and 1 would read the round's lanes.
const BAND_GATE: usize = 2;

/// Rows of a band: the state before each round, and the output.
pub const BAND_ROWS: usize = ROUNDS + 1;

/// Signals of a `PoseidonT(5)` use: `in[5]`, `im[67][5]` and `out[5]`, a row of 5 each.
const BAND_SIGNALS: usize = WIDTH * BAND_ROWS;

/// `set_max_constraint_degree` of the PIL by default: the std's own default, at which M45 measured
/// L1. It sets how the std's product bus groups its terms; the AIR's own constraints are of degree
/// 6 whatever it is.
const MAX_CONSTRAINT_DEGREE: usize = 3;

/// The airgroup, and air, by default.
const AIRGROUP_NAME: &str = "Wrap";

/// The wrap's AIR for `r1cs` (see the module), or an error naming what it cannot place: a custom
/// gate other than `PoseidonT(5)`, or more rows than BN254 has domains for.
pub fn wrap(r1cs: &R1csFile<Bn254>, options: &PlonkOptions) -> Result<SetupResult<Bn254>> {
    let poseidon_uses = poseidon_t_uses(r1cs)?;
    let (plonk_constraints, plonk_additions, copy_merge) = r1cs2plonk_merged(r1cs, options.merge_copies);
    tracing::info!("Number of plonk constraints: {}", plonk_constraints.len());

    let n_band_rows = BAND_ROWS * poseidon_uses.len();
    let placement = PlonkPlacement::new(&plonk_constraints, n_band_rows);
    let n_publics = r1cs.header.n_outputs + r1cs.header.n_pub_inputs;
    let first_public_row = n_band_rows + placement.n_rows;
    let n_used = first_public_row + public_rows(n_publics, N_WIRES);

    // Never below the floor, as the other families: the pre-floor size is the circuit's own.
    let n_bits_natural = n_used.max(2).next_power_of_two().trailing_zeros() as usize;
    let n_bits = n_bits_natural.max(options.min_n_bits.unwrap_or(0));
    ensure!(
        n_bits <= Bn254::TWO_ADICITY,
        "plonk2pil: the wrap needs 2^{n_bits} rows ({n_used} used), and BN254 has domains of at most 2^{}",
        Bn254::TWO_ADICITY
    );
    let n = 1usize << n_bits;
    tracing::info!(
        "NUsed: {n_used} ({} PoseidonT bands of {BAND_ROWS} rows, {} PLONK rows), nBits: {n_bits}, N: {n}",
        poseidon_uses.len(),
        placement.n_rows
    );

    let mut s_map: Vec<Vec<u32>> = vec![vec![0u32; n]; N_WIRES];
    let mut coefficients: Vec<Vec<Bn254>> = vec![vec![Bn254::ZERO; n]; PLONK_COEFFS];
    let mut poseidon = PoseidonColumns::new(n);
    let mut band = PlonkBand::new(n);

    // ── PoseidonT bands ──────────────────────────────────────────────────────
    for (b, cgu) in poseidon_uses.iter().enumerate() {
        let first_row = BAND_ROWS * b;
        for (k, lanes) in cgu.signals.chunks_exact(WIDTH).enumerate() {
            for (column, &signal) in s_map.iter_mut().zip(lanes) {
                // Below n_vars, which is a u32: poseidon_t_uses checks it.
                column[first_row + k] = signal as u32;
            }
            band.allow(first_row + k, 1 << BAND_GATE);
        }
        poseidon.write_band(first_row);
    }

    // ── PLONK constraints ────────────────────────────────────────────────────
    for row in n_band_rows..first_public_row {
        band.allow(row, (1 << GATES_PER_ROW) - 1);
    }
    placement.place(&plonk_constraints, &mut band, &mut s_map, &mut coefficients, n_band_rows);

    // ── Publics ──────────────────────────────────────────────────────────────
    bind_public_signals(&mut s_map, first_public_row, n_publics, N_WIRES);
    let publics_row = (n_publics > 0).then(|| {
        let mut column = vec![Bn254::ZERO; n];
        column[first_public_row] = Bn254::ONE;
        column
    });

    // ── S polynomials ────────────────────────────────────────────────────────
    // The copy-merge remap on every placed cell, the bands' included, then the check that each
    // merged equality is still enforced: every wire is in the connection.
    apply_remap_to_s_map(&mut s_map, &copy_merge.remap);
    verify_merge_soundness(&s_map, &copy_merge.merged_reps, N_WIRES);
    let sv = build_s_polynomials::<Bn254>(N_WIRES, n, n_bits, n_used, &s_map);

    let airgroup_name = options.airgroup_name.clone().unwrap_or_else(|| AIRGROUP_NAME.to_string());
    let mut fixed_pols = build_fixed_pols(&airgroup_name, &coefficients, &sv);
    fixed_pols.extend(poseidon.into_fixed_pols(&airgroup_name));
    if let Some(values) = publics_row {
        fixed_pols.push(FixedPol { name: format!("{airgroup_name}.PUBLICS_ROW"), index: 0, values });
    }

    let pil_str = gen_pil_str(&PilTemplateParams {
        template_file: "poseidon_bn254/wrap",
        template_name: "Wrap",
        namespace_name: &airgroup_name,
        n_bits,
        n_publics,
        max_constraint_degree: options.max_constraint_degree.unwrap_or(MAX_CONSTRAINT_DEGREE),
    });

    Ok(SetupResult {
        fixed_pols,
        pil_str,
        n_bits,
        n_bits_natural,
        n_used,
        s_map,
        gate_bands: Vec::new(),
        plonk_additions,
        airgroup_name: airgroup_name.clone(),
        air_name: airgroup_name,
        band_aux: 0,
    })
}

/// The uses of `PoseidonT(5)`, in the r1cs's order, once what the wrap cannot place is refused: a
/// custom gate of another kind or width, a use of a gate the r1cs does not define, of another
/// number of signals, or of a signal it does not have.
fn poseidon_t_uses(r1cs: &R1csFile<Bn254>) -> Result<Vec<&CustomGateUse>> {
    let width = Bn254::from_int(WIDTH as u64);
    for (id, gate) in r1cs.custom_gates.iter().enumerate() {
        let name = &gate.template_name;
        if !matches!(lookup_gate(name), Some((GateRole::PoseidonT, _))) {
            bail!("plonk2pil: the {BN254_WRAP_FAMILY} wrap places PoseidonT({WIDTH}) only, and custom gate {id} is {name}");
        }
        if gate.parameters != [width] {
            let parameters: Vec<String> = gate.parameters.iter().map(|p| p.to_string()).collect();
            bail!(
                "plonk2pil: the {BN254_WRAP_FAMILY} wrap places PoseidonT({WIDTH}) only, and custom gate {id} is \
                 PoseidonT({})",
                parameters.join(", ")
            );
        }
    }
    let n_vars = u64::from(r1cs.header.n_vars);
    for (i, cgu) in r1cs.custom_gates_uses.iter().enumerate() {
        ensure!(
            (cgu.id as usize) < r1cs.custom_gates.len(),
            "plonk2pil: custom gate use {i} is of gate {}, and the r1cs defines {}",
            cgu.id,
            r1cs.custom_gates.len()
        );
        ensure!(
            cgu.signals.len() == BAND_SIGNALS,
            "plonk2pil: use {i} of PoseidonT({WIDTH}) has {} signals, not the {BAND_SIGNALS} of its in, im and out",
            cgu.signals.len()
        );
        if let Some(signal) = cgu.signals.iter().find(|&&s| s >= n_vars) {
            bail!("plonk2pil: use {i} of PoseidonT({WIDTH}) reads signal {signal}, and the r1cs has {n_vars}");
        }
    }
    // Every gate is PoseidonT(5) now, so the registry's reading of the uses cannot refuse them.
    let cgi = get_custom_gates_info(r1cs);
    debug_assert_eq!(cgi.n(GateRole::PoseidonT), r1cs.custom_gates_uses.len());
    Ok(r1cs.custom_gates_uses.iter().filter(|cgu| cgi.poseidon_t_widths.contains_key(&cgu.id)).collect())
}

/// Where the PLONK constraints go (see the module).
struct PlonkPlacement {
    /// The constraints of each coefficient set, by index, the sets in the order they first appear.
    groups: Vec<Vec<usize>>,
    /// How many of each group's last constraints go to band rows.
    in_bands: Vec<usize>,
    /// The PLONK rows the rest take.
    n_rows: usize,
}

impl PlonkPlacement {
    fn new(constraints: &[PlonkConstraint<Bn254>], band_rows: usize) -> Self {
        let mut group_of: HashMap<[Bn254; PLONK_COEFFS], usize> = HashMap::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (i, c) in constraints.iter().enumerate() {
            let g = *group_of.entry(c.coeffs).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(i);
        }

        let mut free = band_rows;
        let mut in_bands = vec![0; groups.len()];
        for left_over in [1, 2] {
            for (g, group) in groups.iter().enumerate() {
                if group.len() % GATES_PER_ROW == left_over && free >= left_over {
                    in_bands[g] = left_over;
                    free -= left_over;
                }
            }
        }
        for (g, group) in groups.iter().enumerate() {
            let rows = ((group.len() - in_bands[g]) / GATES_PER_ROW).min(free / GATES_PER_ROW);
            in_bands[g] += GATES_PER_ROW * rows;
            free -= GATES_PER_ROW * rows;
        }
        let n_rows = groups.iter().zip(&in_bands).map(|(group, &b)| (group.len() - b).div_ceil(GATES_PER_ROW)).sum();
        Self { groups, in_bands, n_rows }
    }

    /// Places the constraints: the band rows from row 0, the PLONK rows from `first_row`, with the
    /// row's coefficients in `coefficients` (`C[0..4]`).
    fn place(
        &self,
        constraints: &[PlonkConstraint<Bn254>],
        band: &mut PlonkBand,
        s_map: &mut [Vec<u32>],
        coefficients: &mut [Vec<Bn254>],
        first_row: usize,
    ) {
        let mut band_row = 0;
        let mut row = first_row;
        let mut set_coefficients = |row: usize, c: &PlonkConstraint<Bn254>| {
            for (column, &q) in coefficients.iter_mut().zip(&c.coeffs) {
                column[row] = q;
            }
        };
        for (group, &in_bands) in self.groups.iter().zip(&self.in_bands) {
            let (in_rows, in_band_rows) = group.split_at(group.len() - in_bands);
            for &i in in_band_rows {
                set_coefficients(band_row, &constraints[i]);
                band.put(s_map, band_row, BAND_GATE, &constraints[i]);
                band_row += 1;
            }
            for gates in in_rows.chunks(GATES_PER_ROW) {
                set_coefficients(row, &constraints[gates[0]]);
                for g in 0..GATES_PER_ROW {
                    band.put(s_map, row, g, &constraints[gates[g.min(gates.len() - 1)]]);
                }
                row += 1;
            }
        }
        assert_eq!(row, first_row + self.n_rows, "PLONK row count mismatch");
    }
}

/// The fixed columns of the PoseidonT bands: `POSEIDON_RC[5]`, `POSEIDON`, `POSEIDON_FULL_ROUND`
/// and `POSEIDON_PARTIAL_ROUND` of `wrap.pil`.
struct PoseidonColumns {
    round_constants: Vec<Vec<Bn254>>,
    band: Vec<Bn254>,
    full_round: Vec<Bn254>,
    partial_round: Vec<Bn254>,
}

impl PoseidonColumns {
    fn new(n: usize) -> Self {
        let zeros = || vec![Bn254::ZERO; n];
        Self {
            round_constants: (0..WIDTH).map(|_| zeros()).collect(),
            band: zeros(),
            full_round: zeros(),
            partial_round: zeros(),
        }
    }

    /// The band whose first row is `first_row`: on its row `r < ROUNDS`, the constants and the kind
    /// of round `r`; on its last row, the output, only `POSEIDON`.
    fn write_band(&mut self, first_row: usize) {
        for r in 0..BAND_ROWS {
            let row = first_row + r;
            self.band[row] = Bn254::ONE;
            if r == ROUNDS {
                continue;
            }
            for (j, column) in self.round_constants.iter_mut().enumerate() {
                column[row] = ROUND_CONSTANTS[WIDTH * r + j];
            }
            if is_full_round(r) {
                self.full_round[row] = Bn254::ONE;
            } else {
                self.partial_round[row] = Bn254::ONE;
            }
        }
    }

    fn into_fixed_pols(self, airgroup_name: &str) -> Vec<FixedPol<Bn254>> {
        let pol = |name: &str, index: usize, values: Vec<Bn254>| FixedPol {
            name: format!("{airgroup_name}.{name}"),
            index,
            values,
        };
        let mut pols: Vec<FixedPol<Bn254>> =
            self.round_constants.into_iter().enumerate().map(|(j, values)| pol("POSEIDON_RC", j, values)).collect();
        pols.push(pol("POSEIDON", 0, self.band));
        pols.push(pol("POSEIDON_FULL_ROUND", 0, self.full_round));
        pols.push(pol("POSEIDON_PARTIAL_ROUND", 0, self.partial_round));
        pols
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plonk2pil::r1cs::to_plonk::PlonkAddition;
    use crate::plonk2pil::r1cs::types::{CustomGate, LinearCombination, R1csConstraint, R1csHeader};

    fn q(v: i64) -> Bn254 {
        Bn254::from_int(v)
    }

    /// `l·r = o`, `l + r = o` or `2·(l + r) = o` as an r1cs constraint, by `kind` (0, 1 or 2): three
    /// coefficient sets once converted.
    fn r1cs_constraint(l: u32, r: u32, o: u32, kind: u32) -> R1csConstraint<Bn254> {
        let lc = |terms: &[(u32, i64)]| -> LinearCombination<Bn254> { terms.iter().map(|&(w, c)| (w, q(c))).collect() };
        match kind {
            0 => R1csConstraint { a: lc(&[(l, 1)]), b: lc(&[(r, 1)]), c: lc(&[(o, 1)]) },
            k => R1csConstraint { a: lc(&[(0, i64::from(k))]), b: lc(&[(l, 1), (r, 1)]), c: lc(&[(o, 1)]) },
        }
    }

    fn r1cs(
        n_vars: u32,
        n_publics: u32,
        constraints: Vec<R1csConstraint<Bn254>>,
        custom_gates: Vec<CustomGate<Bn254>>,
        custom_gates_uses: Vec<CustomGateUse>,
    ) -> R1csFile<Bn254> {
        R1csFile {
            header: R1csHeader {
                n8: 32,
                prime_bytes: vec![],
                n_vars,
                n_outputs: n_publics,
                n_pub_inputs: 0,
                n_prv_inputs: 0,
                n_labels: 0,
                n_constraints: constraints.len() as u32,
                use_custom_gates: !custom_gates.is_empty(),
            },
            constraints,
            wire_to_label: vec![],
            custom_gates,
            custom_gates_uses,
        }
    }

    fn poseidon_t(t: u64) -> CustomGate<Bn254> {
        CustomGate { template_name: "PoseidonT".into(), parameters: vec![Bn254::from_int(t)] }
    }

    /// A use of gate 0 on the signals `first..first + 345`.
    fn poseidon_use(first: u64) -> CustomGateUse {
        CustomGateUse { id: 0, signals: (first..first + BAND_SIGNALS as u64).collect() }
    }

    fn options() -> PlonkOptions {
        PlonkOptions { hash_id: BN254_WRAP_FAMILY.into(), ..Default::default() }
    }

    /// The witness extended with the additions, then gathered through the map as the prover's
    /// `getCommitedPols` does: a cell the map leaves out is 0.
    fn trace(res: &SetupResult<Bn254>, witness: &[Bn254]) -> Vec<Vec<Bn254>> {
        let mut w = witness.to_vec();
        for PlonkAddition { wires, coeffs } in &res.plonk_additions {
            w.push(coeffs[0] * w[wires[0] as usize] + coeffs[1] * w[wires[1] as usize]);
        }
        res.s_map
            .iter()
            .map(|col| col.iter().map(|&s| if s == 0 { Bn254::ZERO } else { w[s as usize] }).collect())
            .collect()
    }

    fn column<'a>(res: &'a SetupResult<Bn254>, name: &str, index: usize) -> &'a [Bn254] {
        let name = format!("{AIRGROUP_NAME}.{name}");
        &res.fixed_pols
            .iter()
            .find(|p| p.name == name && p.index == index)
            .unwrap_or_else(|| panic!("no {name}[{index}]"))
            .values
    }

    /// The rows where one of `wrap.pil`'s PLONK gates does not hold on `trace`.
    fn failing_gates(res: &SetupResult<Bn254>, trace: &[Vec<Bn254>]) -> Vec<(usize, usize)> {
        let c: Vec<&[Bn254]> = (0..PLONK_COEFFS).map(|k| column(res, "C", k)).collect();
        let in_band = column(res, "POSEIDON", 0);
        let n = 1 << res.n_bits;
        let mut failing = Vec::new();
        for row in 0..n {
            for g in 0..GATES_PER_ROW {
                if g != BAND_GATE && in_band[row] == Bn254::ONE {
                    continue;
                }
                let [l, r, o] = [0, 1, 2].map(|j| trace[3 * g + j][row]);
                if !(c[0][row] * l * r + c[1][row] * l + c[2][row] * r + c[3][row] * o + c[4][row]).is_zero() {
                    failing.push((row, g));
                }
            }
        }
        failing
    }

    /// 198 constraints of three coefficient sets, a band of 69 rows and a public, with a satisfying
    /// witness: w1 = 3, w2 = 5, then w[k] is w[k-1]·w[k-2], w[k-1] + w[k-2] or 2·(w[k-1] + w[k-2]).
    fn three_sets_and_a_band() -> (R1csFile<Bn254>, Vec<Bn254>) {
        let mut constraints = Vec::new();
        let mut witness = vec![Bn254::ONE, q(3), q(5)];
        for k in 3..201u32 {
            constraints.push(r1cs_constraint(k - 1, k - 2, k, k % 3));
            let (a, b) = (witness[k as usize - 1], witness[k as usize - 2]);
            witness.push(match k % 3 {
                0 => a * b,
                1 => a + b,
                _ => (a + b).double(),
            });
        }
        let first_band_signal = witness.len() as u64;
        let n_vars = witness.len() as u32 + BAND_SIGNALS as u32;
        // The band's signals are zeros: the PLONK gates do not read them.
        witness.resize(n_vars as usize, Bn254::ZERO);
        (r1cs(n_vars, 1, constraints, vec![poseidon_t(5)], vec![poseidon_use(first_band_signal)]), witness)
    }

    /// The band rows take 69 of the constraints, PLONK rows the rest, and every gate holds on a
    /// satisfying witness and not on a wrong one.
    #[test]
    fn the_plonk_gates_hold_on_the_witness_wherever_they_are_placed() {
        let (r1cs, witness) = three_sets_and_a_band();
        let res = wrap(&r1cs, &options()).unwrap();
        // qO is -1 in every constraint, so a row with one has qO set.
        let in_bands = (0..BAND_ROWS).filter(|&row| !column(&res, "C", 3)[row].is_zero()).count();
        assert_eq!(in_bands, BAND_ROWS, "every band row holds a constraint");
        assert_eq!(res.n_used, BAND_ROWS + (198 - BAND_ROWS) / 3 + 1, "the band, 43 PLONK rows, a public row");

        let trace = trace(&res, &witness);
        assert_eq!(failing_gates(&res, &trace), Vec::<(usize, usize)>::new());
        let mut wrong = witness.clone();
        wrong[100] += Bn254::ONE;
        assert!(!failing_gates(&res, &self::trace(&res, &wrong)).is_empty(), "a wrong wire fails a gate");
    }

    /// The same r1cs gives the same map and columns: the groups are in the constraints' order, a
    /// hash map only finds them.
    #[test]
    fn the_setup_is_deterministic() {
        let (r1cs, _) = three_sets_and_a_band();
        let [a, b] = [0, 1].map(|_| wrap(&r1cs, &options()).unwrap());
        assert_eq!(a.s_map, b.s_map);
        let columns = |res: &SetupResult<Bn254>| -> Vec<(String, usize, Vec<Bn254>)> {
            res.fixed_pols.iter().map(|p| (p.name.clone(), p.index, p.values.clone())).collect()
        };
        assert_eq!(columns(&a), columns(&b));
        assert_eq!((a.pil_str, a.plonk_additions), (b.pil_str, b.plonk_additions));
    }

    /// Left overs of 1 go to band rows first, then left overs of 2, then whole rows; the rest takes
    /// PLONK rows of 3.
    #[test]
    fn band_rows_take_left_overs_first() {
        let constraint = |set: i64| PlonkConstraint { wires: [1, 2, 3], coeffs: [q(set), q(0), q(0), q(-1), q(0)] };
        // Sets of 4, 2 and 6 constraints: left overs 1, 2 and 0.
        let constraints: Vec<_> = [(1, 4), (2, 2), (3, 6)].iter().flat_map(|&(s, n)| vec![constraint(s); n]).collect();
        // The PLONK rows of each set.
        let rows = |band_rows| {
            let placement = PlonkPlacement::new(&constraints, band_rows);
            let per_set = placement.groups.iter().zip(&placement.in_bands);
            let rows: Vec<usize> = per_set.map(|(group, &b)| (group.len() - b).div_ceil(GATES_PER_ROW)).collect();
            assert_eq!(rows.iter().sum::<usize>(), placement.n_rows);
            rows
        };
        assert_eq!(rows(0), [2, 1, 2]);
        assert_eq!(rows(1), [1, 1, 2], "the left over of 1");
        assert_eq!(rows(2), [1, 1, 2], "a left over of 2 needs both");
        assert_eq!(rows(3), [1, 0, 2], "then the left over of 2");
        assert_eq!(rows(6), [0, 0, 2], "then three at a time");
        assert_eq!(rows(100), [0, 0, 0], "every constraint in a band row");
    }

    /// A band is the use's signals, 5 a row, with its round's constants and kind; its last row is
    /// the output, of no round.
    #[test]
    fn a_band_maps_every_signal_with_its_rounds_constants() {
        let n_vars = 1 + 2 * BAND_SIGNALS as u32;
        let uses = vec![poseidon_use(1), poseidon_use(1 + BAND_SIGNALS as u64)];
        let res = wrap(&r1cs(n_vars, 0, vec![], vec![poseidon_t(5)], uses), &options()).unwrap();
        assert_eq!(res.n_used, 2 * BAND_ROWS);
        let (full, partial, in_band) = (
            column(&res, "POSEIDON_FULL_ROUND", 0),
            column(&res, "POSEIDON_PARTIAL_ROUND", 0),
            column(&res, "POSEIDON", 0),
        );
        for row in 0..1 << res.n_bits {
            let (b, r) = (row / BAND_ROWS, row % BAND_ROWS);
            if b >= 2 {
                assert!(in_band[row].is_zero() && full[row].is_zero() && partial[row].is_zero(), "row {row}");
                continue;
            }
            for j in 0..WIDTH {
                assert_eq!(res.s_map[j][row] as usize, 1 + b * BAND_SIGNALS + WIDTH * r + j, "row {row} lane {j}");
                let rc = column(&res, "POSEIDON_RC", j)[row];
                assert_eq!(rc, if r < ROUNDS { ROUND_CONSTANTS[WIDTH * r + j] } else { Bn254::ZERO }, "row {row}");
            }
            assert_eq!(in_band[row], Bn254::ONE);
            let kind = (full[row], partial[row]);
            let expected = match r {
                ROUNDS => (Bn254::ZERO, Bn254::ZERO),
                r if !(4..64).contains(&r) => (Bn254::ONE, Bn254::ZERO),
                _ => (Bn254::ZERO, Bn254::ONE),
            };
            assert_eq!(kind, expected, "row {row}: round {r}");
        }
        assert!(res.gate_bands.is_empty(), "no band is recomputed: the map covers it");
        assert!(!res.fixed_pols.iter().any(|p| p.name.ends_with(".PUBLICS_ROW")), "no public, no PUBLICS_ROW");
    }

    /// Each public is a cell of the rows after the gates, where PUBLICS_ROW points.
    #[test]
    fn the_publics_follow_the_gates() {
        let r1cs = r1cs(4, 2, vec![r1cs_constraint(1, 2, 3, 0)], vec![], vec![]);
        let res = wrap(&r1cs, &options()).unwrap();
        let publics_row = column(&res, "PUBLICS_ROW", 0);
        let first = publics_row.iter().position(|v| *v == Bn254::ONE).unwrap();
        assert_eq!(first, res.n_used - 1, "one public row, after the one PLONK row");
        assert_eq!((res.s_map[0][first], res.s_map[1][first]), (1, 2));
    }

    #[test]
    fn other_gates_widths_and_uses_are_refused() {
        let refusal = |gates, uses| wrap(&r1cs(400, 0, vec![], gates, uses), &options()).unwrap_err().to_string();

        let cmul = CustomGate { template_name: "CMul".into(), parameters: vec![] };
        assert!(refusal(vec![cmul], vec![]).contains("custom gate 0 is CMul"));
        let unknown = CustomGate { template_name: "Keccak".into(), parameters: vec![] };
        assert!(refusal(vec![unknown], vec![]).contains("custom gate 0 is Keccak"));
        assert!(refusal(vec![poseidon_t(3)], vec![]).contains("custom gate 0 is PoseidonT(3)"));

        let short = CustomGateUse { id: 0, signals: vec![1; 10] };
        assert!(refusal(vec![poseidon_t(5)], vec![short]).contains("has 10 signals, not the 345"));
        let of_no_gate = CustomGateUse { id: 1, ..poseidon_use(1) };
        assert!(refusal(vec![poseidon_t(5)], vec![of_no_gate]).contains("is of gate 1, and the r1cs defines 1"));
        assert!(refusal(vec![poseidon_t(5)], vec![poseidon_use(100)]).contains("reads signal 400"));
    }
}
