//! The final SNARK wrap's setup, over BN128 (`pil/poseidon_bn128/wrap.pil`), in layout L1 with
//! range checks: 9 wires, all of them in the std's connection; 3 PLONK gates a row on
//! the row's one coefficient set; each `PoseidonT(5)` use a band of 69 rows, one round a row on
//! a[0..4]; and each `Num2Bytes(nBits)` use a range-check row. On the rows of the custom gates, the
//! bands' and the range checks', gate 2 is the only PLONK gate on.
//!
//! The rows, in order: the bands, one per use in the r1cs's order; the range-check rows, one per
//! use in the r1cs's order; the PLONK rows; the public rows.
//! - A band is the use's signals as they are, 5 a row: `in`, then `im[0..67]`, then `out`. The
//!   gate exposes every round's state, so the map covers the whole band and nothing is expanded.
//! - A range-check row is the use's `in` on a[0] and its `⌈nBits/16⌉` chunks of 16 bits from a[1]
//!   ([`RANGE_CHECK_CHUNK_COLS`]), the cells past them left empty; its row constants `K` are the
//!   chunks' weights, `2^(16·k)` for the chunks and 0 past them. Each is a gate band of the exec
//!   ([`GateBandKind::PoseidonBn128WrapRangeCheck`], its number of chunks as the payload), so that
//!   the wrap's witness counts the table's multiplicity, `RANGE_MUL`, the stage-1 column after the
//!   wires, which the band section's aux word names. The table `RANGE` is `i mod 2^16` on row `i`,
//!   so an AIR with range checks has at least `2^16` rows.
//!
//! The trace is the witness gathered through the map, and, with range checks, `RANGE_MUL`.
//!
//! **PLONK placement.** The gates of a row share its coefficients, so the constraints are grouped
//! by coefficient set, as the compressor groups them by `ckey`, and a group of `n` takes `n / 3`
//! rows of 3. Its `n % 3` left over would take one more row; a row of a custom gate has room for
//! one constraint of any set, on gate 2. So the gate rows go first to the groups whose left over is
//! 1 (one gate row saves a row), then to those whose left over is 2 (two save one), then three at a
//! time to any group (three save one). On a group's last row, a gate with no constraint of its own
//! repeats the row's last one: a cell the map leaves out reads 0, and a gate on zeros reads `qC`.

use std::collections::HashMap;
use std::fmt;

use anyhow::{bail, ensure, Result};
use proofman_common::exec_format::{RANGE_CHECK_CHUNK_BITS, RANGE_CHECK_CHUNK_COLS};
use proofman_common::hash_family::{lookup_gate, GateRole, BN128_WRAP_FAMILY};
use proofman_fields::{Bn128, Field, PrimeField, QuotientMap};

use super::constants::{is_full_round, ROUNDS, ROUND_CONSTANTS, WIDTH};
use super::{gen_pil_str, PilTemplateParams};
use crate::plonk2pil::merge_copies::{apply_remap_to_s_map, r1cs2plonk_merged, verify_merge_soundness};
use crate::plonk2pil::r1cs::to_plonk::{get_custom_gates_info, PlonkConstraint, PLONK_COEFFS};
use crate::plonk2pil::r1cs::types::{
    CustomGate, CustomGateUse, FixedPol, GateBand, GateBandKind, PlonkOptions, R1csFile, SetupResult,
};
use crate::plonk2pil::utils::{bind_public_signals, build_fixed_pols, build_s_polynomials, public_rows, PlonkBand};

/// The witness columns `a[0..8]`, every one in the connection.
const N_WIRES: usize = 9;

/// PLONK gates a row: gate `g` on `a[3g..3g+2]`.
const GATES_PER_ROW: usize = 3;

/// The PLONK gate that is on in the rows of a custom gate: gates 0 and 1 would read its cells.
const BAND_GATE: usize = 2;

/// Rows of a band: the state before each round, and the output.
pub const BAND_ROWS: usize = ROUNDS + 1;

/// Signals of a `PoseidonT(5)` use: `in[5]`, `im[67][5]` and `out[5]`, a row of 5 each.
const BAND_SIGNALS: usize = WIDTH * BAND_ROWS;

/// Chunk cells of a range-check row.
const RANGE_CHECK_CHUNKS: usize = RANGE_CHECK_CHUNK_COLS.end - RANGE_CHECK_CHUNK_COLS.start;

/// The widest `Num2Bytes` a range-check row holds: 5 chunks of 16 bits.
pub const MAX_RANGE_CHECK_BITS: usize = RANGE_CHECK_CHUNKS * RANGE_CHECK_CHUNK_BITS as usize;

/// The stage-1 column of the range checks' multiplicity, `RANGE_MUL`: the one after the wires.
pub const RANGE_MUL_COLUMN: usize = N_WIRES;

// The chunk weights are the row constants `K`, a column per lane of the band, and the use's `in` is
// on a[0], before the chunks.
const _: () = assert!(RANGE_CHECK_CHUNKS <= WIDTH && RANGE_CHECK_CHUNK_COLS.start == 1);

/// `set_max_constraint_degree` of the PIL by default: the degree the std groups its buses' terms
/// to, the AIR's own constraints being of degree 6 whatever it is. With the range checks' sum bus,
/// the AIR cannot be set up at the std's default degree, 3, and pilfflonk's default `--extra-muls`
/// (2); degree 6 with [`EXTRA_MULS`] is the cheapest of the settings that can to verify (measured
/// on the fibonacci-square wrap: 337,557 gas against 356,761 at degree 3 and `--extra-muls` 4).
/// setup-snark also bounds pilfflonk's degree search with it (setup-pilfflonk's
/// `--max-constraint-degree`): the AIR's own constraints reach it, so `Q` needs no im pol.
pub const MAX_CONSTRAINT_DEGREE: usize = 6;

/// The `--extra-muls` the wrap's AIR is set up with by pilfflonk (pilfflonk/docs/protocol.md,
/// grouping rule 3), at the PIL's [`MAX_CONSTRAINT_DEGREE`]: setup-snark passes it, with a ptau of
/// at least `13·N + 12` powers for it.
pub const EXTRA_MULS: u64 = 3;

/// The airgroup, and air, by default.
const AIRGROUP_NAME: &str = "Wrap";

/// The wrap's AIR for `r1cs` (see the module), or an error naming what it cannot place: a custom
/// gate other than `PoseidonT(5)` and `Num2Bytes` of up to [`MAX_RANGE_CHECK_BITS`] bits, a use
/// that does not fit its gate, or more rows than BN128 has domains for.
pub fn wrap(r1cs: &R1csFile<Bn128>, options: &PlonkOptions) -> Result<SetupResult<Bn128>> {
    let uses = gate_uses(r1cs)?;
    let (plonk_constraints, plonk_additions, copy_merge) = r1cs2plonk_merged(r1cs, options.merge_copies);
    tracing::info!("Number of plonk constraints: {}", plonk_constraints.len());

    let n_band_rows = BAND_ROWS * uses.poseidon_t.len();
    let n_range_checks = uses.num2bytes.len();
    let n_gate_rows = n_band_rows + n_range_checks;
    let placement = PlonkPlacement::new(&plonk_constraints, n_gate_rows);
    let n_publics = r1cs.header.n_outputs + r1cs.header.n_pub_inputs;
    let first_public_row = n_gate_rows + placement.n_rows;
    let n_public_rows = public_rows(n_publics, N_WIRES);
    let n_used = first_public_row + n_public_rows;

    // Never below the floor, as the other families: the pre-floor size is the circuit's own, the
    // range checks' table included.
    let table_bits = if n_range_checks > 0 { RANGE_CHECK_CHUNK_BITS as usize } else { 0 };
    let n_bits_natural = (n_used.max(2).next_power_of_two().trailing_zeros() as usize).max(table_bits);
    let n_bits = n_bits_natural.max(options.min_n_bits.unwrap_or(0));
    ensure!(
        n_bits <= Bn128::TWO_ADICITY,
        "plonk2pil: the wrap needs 2^{n_bits} rows ({n_used} used), and BN128 has domains of at most 2^{}",
        Bn128::TWO_ADICITY
    );
    let n = 1usize << n_bits;
    tracing::info!(
        "NUsed: {n_used} ({} PoseidonT bands of {BAND_ROWS} rows, {n_range_checks} range-check rows, {} PLONK rows), \
         nBits: {n_bits}, N: {n}",
        uses.poseidon_t.len(),
        placement.n_rows
    );

    let mut s_map: Vec<Vec<u32>> = vec![vec![0u32; n]; N_WIRES];
    let mut coefficients: Vec<Vec<Bn128>> = vec![vec![Bn128::ZERO; n]; PLONK_COEFFS];
    let mut columns = GateColumns::new(n, n_range_checks > 0);
    let mut band = PlonkBand::new(n);

    // ── PoseidonT bands ──────────────────────────────────────────────────────
    for (b, cgu) in uses.poseidon_t.iter().enumerate() {
        let first_row = BAND_ROWS * b;
        for (k, lanes) in cgu.signals.chunks_exact(WIDTH).enumerate() {
            for (column, &signal) in s_map.iter_mut().zip(lanes) {
                // Below n_vars, which is a u32: gate_uses checks it.
                column[first_row + k] = signal as u32;
            }
            band.allow(first_row + k, 1 << BAND_GATE);
        }
        columns.write_band(first_row);
    }

    // ── Range checks ─────────────────────────────────────────────────────────
    let mut gate_bands = Vec::with_capacity(n_range_checks);
    for (i, &(cgu, n_chunks)) in uses.num2bytes.iter().enumerate() {
        let row = n_band_rows + i;
        // `in`, then the chunks: gate_uses checks there are 1 + n_chunks, below n_vars.
        let (&value, chunks) = cgu.signals.split_first().expect("a Num2Bytes use has its in");
        s_map[0][row] = value as u32;
        for (col, &chunk) in RANGE_CHECK_CHUNK_COLS.zip(chunks) {
            s_map[col][row] = chunk as u32;
        }
        band.allow(row, 1 << BAND_GATE);
        columns.write_range_check(row, n_chunks);
        // Below n, at most 2^28.
        gate_bands.push(GateBand {
            row: row as u32,
            kind: GateBandKind::PoseidonBn128WrapRangeCheck,
            payload: n_chunks as u64,
        });
    }

    // ── PLONK constraints ────────────────────────────────────────────────────
    for row in n_gate_rows..first_public_row {
        band.allow(row, (1 << GATES_PER_ROW) - 1);
    }
    placement.place(&plonk_constraints, &mut band, &mut s_map, &mut coefficients, n_gate_rows);

    // ── Publics ──────────────────────────────────────────────────────────────
    bind_public_signals(&mut s_map, first_public_row, n_publics, N_WIRES);
    let publics_rows: Vec<Vec<Bn128>> = (0..n_public_rows)
        .map(|k| {
            let mut column = vec![Bn128::ZERO; n];
            column[first_public_row + k] = Bn128::ONE;
            column
        })
        .collect();

    // ── S polynomials ────────────────────────────────────────────────────────
    // The copy-merge remap on every placed cell, the custom gates' included, then the check that
    // each merged equality is still enforced: every wire is in the connection.
    apply_remap_to_s_map(&mut s_map, &copy_merge.remap);
    verify_merge_soundness(&s_map, &copy_merge.merged_reps, N_WIRES);
    let sv = build_s_polynomials::<Bn128>(N_WIRES, n, n_bits, n_used, &s_map);

    let airgroup_name = options.airgroup_name.clone().unwrap_or_else(|| AIRGROUP_NAME.to_string());
    let mut fixed_pols = build_fixed_pols(&airgroup_name, &coefficients, &sv);
    fixed_pols.extend(columns.into_fixed_pols(&airgroup_name));
    fixed_pols.extend(publics_rows.into_iter().enumerate().map(|(k, values)| FixedPol {
        name: format!("{airgroup_name}.PUBLICS_ROW"),
        index: k,
        values,
    }));

    let pil_str = gen_pil_str(&PilTemplateParams {
        template_file: "poseidon_bn128/wrap",
        template_name: "Wrap",
        namespace_name: &airgroup_name,
        n_bits,
        n_publics,
        n_range_checks,
        max_constraint_degree: options.max_constraint_degree.unwrap_or(MAX_CONSTRAINT_DEGREE),
    });

    Ok(SetupResult {
        fixed_pols,
        pil_str,
        n_bits,
        n_bits_natural,
        n_used,
        s_map,
        band_aux: if gate_bands.is_empty() { 0 } else { RANGE_MUL_COLUMN as u64 },
        gate_bands,
        plonk_additions,
        airgroup_name: airgroup_name.clone(),
        air_name: airgroup_name,
    })
}

/// A custom gate the wrap places.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WrapGate {
    /// `PoseidonT(5)`: a band.
    PoseidonT,
    /// `Num2Bytes(n_bits)`: a range-check row.
    Num2Bytes { n_bits: usize },
}

impl WrapGate {
    /// Gate `id` of the r1cs, or why the wrap cannot place it: a gate of another kind, a
    /// `PoseidonT` of another width, or a `Num2Bytes` of no bits or of more than a row holds.
    fn of(id: usize, gate: &CustomGate<Bn128>) -> Result<Self> {
        let parameter = match gate.parameters.as_slice() {
            [p] => usize::try_from(&p.as_canonical_biguint()).ok(),
            _ => None,
        };
        match (lookup_gate(&gate.template_name), parameter) {
            (Some((GateRole::PoseidonT, _)), Some(WIDTH)) => Ok(Self::PoseidonT),
            (Some((GateRole::RangeCheck, _)), Some(n_bits)) if (1..=MAX_RANGE_CHECK_BITS).contains(&n_bits) => {
                Ok(Self::Num2Bytes { n_bits })
            }
            _ => {
                let parameters: Vec<String> = gate.parameters.iter().map(|p| p.to_string()).collect();
                let parameters =
                    if parameters.is_empty() { String::new() } else { format!("({})", parameters.join(", ")) };
                bail!(
                    "plonk2pil: the {BN128_WRAP_FAMILY} wrap places PoseidonT({WIDTH}) and Num2Bytes(nBits), \
                     0 < nBits <= {MAX_RANGE_CHECK_BITS}, only, and custom gate {id} is {}{parameters}",
                    gate.template_name
                )
            }
        }
    }

    /// The signals of a use: `PoseidonT`'s `in`, `im` and `out`; `Num2Bytes`'s `in` and chunks.
    fn n_signals(self) -> usize {
        match self {
            Self::PoseidonT => BAND_SIGNALS,
            Self::Num2Bytes { n_bits } => 1 + n_chunks(n_bits),
        }
    }

    fn signal_names(self) -> &'static str {
        match self {
            Self::PoseidonT => "in, im and out",
            Self::Num2Bytes { .. } => "in and out",
        }
    }
}

impl fmt::Display for WrapGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PoseidonT => write!(f, "PoseidonT({WIDTH})"),
            Self::Num2Bytes { n_bits } => write!(f, "Num2Bytes({n_bits})"),
        }
    }
}

/// The 16-bit chunks of `Num2Bytes(n_bits)`.
fn n_chunks(n_bits: usize) -> usize {
    n_bits.div_ceil(RANGE_CHECK_CHUNK_BITS as usize)
}

/// The uses of the r1cs's custom gates, each kind in the r1cs's order.
struct GateUses<'a> {
    poseidon_t: Vec<&'a CustomGateUse>,
    /// Each with its number of chunks.
    num2bytes: Vec<(&'a CustomGateUse, usize)>,
}

/// The uses of the r1cs's custom gates, once what the wrap cannot place is refused: a gate
/// [`WrapGate::of`] refuses, a use of a gate the r1cs does not define, of another number of
/// signals than its gate's, or of a signal the r1cs does not have.
fn gate_uses(r1cs: &R1csFile<Bn128>) -> Result<GateUses<'_>> {
    let gates: Vec<WrapGate> =
        r1cs.custom_gates.iter().enumerate().map(|(id, gate)| WrapGate::of(id, gate)).collect::<Result<_>>()?;
    let n_vars = u64::from(r1cs.header.n_vars);
    let mut uses = GateUses { poseidon_t: Vec::new(), num2bytes: Vec::new() };
    for (i, cgu) in r1cs.custom_gates_uses.iter().enumerate() {
        let Some(&gate) = gates.get(cgu.id as usize) else {
            bail!("plonk2pil: custom gate use {i} is of gate {}, and the r1cs defines {}", cgu.id, gates.len());
        };
        ensure!(
            cgu.signals.len() == gate.n_signals(),
            "plonk2pil: use {i} of {gate} has {} signals, not the {} of its {}",
            cgu.signals.len(),
            gate.n_signals(),
            gate.signal_names()
        );
        if let Some(signal) = cgu.signals.iter().find(|&&s| s >= n_vars) {
            bail!("plonk2pil: use {i} of {gate} reads signal {signal}, and the r1cs has {n_vars}");
        }
        match gate {
            WrapGate::PoseidonT => uses.poseidon_t.push(cgu),
            WrapGate::Num2Bytes { n_bits } => uses.num2bytes.push((cgu, n_chunks(n_bits))),
        }
    }
    // Every gate is the wrap's now, so the registry reads the uses alike, and cannot refuse them.
    debug_assert!({
        let cgi = get_custom_gates_info(r1cs);
        cgi.n(GateRole::PoseidonT) == uses.poseidon_t.len() && cgi.n(GateRole::RangeCheck) == uses.num2bytes.len()
    });
    Ok(uses)
}

/// Where the PLONK constraints go (see the module).
struct PlonkPlacement {
    /// The constraints of each coefficient set, by index, the sets in the order they first appear.
    groups: Vec<Vec<usize>>,
    /// How many of each group's last constraints go to the rows of the custom gates.
    in_gate_rows: Vec<usize>,
    /// The PLONK rows the rest take.
    n_rows: usize,
}

impl PlonkPlacement {
    /// The placement with `gate_rows` rows of custom gates, each with room for one constraint.
    fn new(constraints: &[PlonkConstraint<Bn128>], gate_rows: usize) -> Self {
        let mut group_of: HashMap<[Bn128; PLONK_COEFFS], usize> = HashMap::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (i, c) in constraints.iter().enumerate() {
            let g = *group_of.entry(c.coeffs).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(i);
        }

        let mut free = gate_rows;
        let mut in_gate_rows = vec![0; groups.len()];
        for left_over in [1, 2] {
            for (g, group) in groups.iter().enumerate() {
                if group.len() % GATES_PER_ROW == left_over && free >= left_over {
                    in_gate_rows[g] = left_over;
                    free -= left_over;
                }
            }
        }
        for (g, group) in groups.iter().enumerate() {
            let rows = ((group.len() - in_gate_rows[g]) / GATES_PER_ROW).min(free / GATES_PER_ROW);
            in_gate_rows[g] += GATES_PER_ROW * rows;
            free -= GATES_PER_ROW * rows;
        }
        let n_rows =
            groups.iter().zip(&in_gate_rows).map(|(group, &b)| (group.len() - b).div_ceil(GATES_PER_ROW)).sum();
        Self { groups, in_gate_rows, n_rows }
    }

    /// Places the constraints: the custom gates' rows from row 0, the PLONK rows from `first_row`,
    /// with the row's coefficients in `coefficients` (`C[0..4]`).
    fn place(
        &self,
        constraints: &[PlonkConstraint<Bn128>],
        band: &mut PlonkBand,
        s_map: &mut [Vec<u32>],
        coefficients: &mut [Vec<Bn128>],
        first_row: usize,
    ) {
        let mut gate_row = 0;
        let mut row = first_row;
        let mut set_coefficients = |row: usize, c: &PlonkConstraint<Bn128>| {
            for (column, &q) in coefficients.iter_mut().zip(&c.coeffs) {
                column[row] = q;
            }
        };
        for (group, &in_gate_rows) in self.groups.iter().zip(&self.in_gate_rows) {
            let (in_rows, in_custom_rows) = group.split_at(group.len() - in_gate_rows);
            for &i in in_custom_rows {
                set_coefficients(gate_row, &constraints[i]);
                band.put(s_map, gate_row, BAND_GATE, &constraints[i]);
                gate_row += 1;
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

/// The fixed columns of the custom gates' rows: `K[5]`, `POSEIDON`, `POSEIDON_FULL_ROUND` and
/// `POSEIDON_PARTIAL_ROUND` of `wrap.pil`, and, in an AIR with range checks, `RANGE_CHECK` and
/// `RANGE`.
struct GateColumns {
    /// `K`: a band row's round constants, a range-check row's chunk weights.
    row_constants: Vec<Vec<Bn128>>,
    band: Vec<Bn128>,
    full_round: Vec<Bn128>,
    partial_round: Vec<Bn128>,
    /// `RANGE_CHECK`, if the AIR has range checks. `RANGE` is the same in every one.
    range_check: Option<Vec<Bn128>>,
}

impl GateColumns {
    fn new(n: usize, range_checks: bool) -> Self {
        let zeros = || vec![Bn128::ZERO; n];
        Self {
            row_constants: (0..WIDTH).map(|_| zeros()).collect(),
            band: zeros(),
            full_round: zeros(),
            partial_round: zeros(),
            range_check: range_checks.then(zeros),
        }
    }

    /// The band whose first row is `first_row`: on its row `r < ROUNDS`, the constants and the kind
    /// of round `r`; on its last row, the output, only `POSEIDON`.
    fn write_band(&mut self, first_row: usize) {
        for r in 0..BAND_ROWS {
            let row = first_row + r;
            self.band[row] = Bn128::ONE;
            if r == ROUNDS {
                continue;
            }
            for (j, column) in self.row_constants.iter_mut().enumerate() {
                column[row] = ROUND_CONSTANTS[WIDTH * r + j];
            }
            if is_full_round(r) {
                self.full_round[row] = Bn128::ONE;
            } else {
                self.partial_round[row] = Bn128::ONE;
            }
        }
    }

    /// The range-check row `row` of `n_chunks` chunks: `RANGE_CHECK`, and the weights `2^(16·k)` of
    /// its chunks `k`, 0 past them.
    fn write_range_check(&mut self, row: usize, n_chunks: usize) {
        let selector = self.range_check.as_mut().expect("range-check rows are in an AIR with range checks");
        selector[row] = Bn128::ONE;
        let chunk_size = Bn128::from_int(1u64 << RANGE_CHECK_CHUNK_BITS);
        let mut weight = Bn128::ONE;
        for column in &mut self.row_constants[..n_chunks] {
            column[row] = weight;
            weight *= chunk_size;
        }
    }

    fn into_fixed_pols(self, airgroup_name: &str) -> Vec<FixedPol<Bn128>> {
        let pol = |name: &str, index: usize, values: Vec<Bn128>| FixedPol {
            name: format!("{airgroup_name}.{name}"),
            index,
            values,
        };
        let mut pols: Vec<FixedPol<Bn128>> =
            self.row_constants.into_iter().enumerate().map(|(j, values)| pol("K", j, values)).collect();
        pols.push(pol("POSEIDON", 0, self.band));
        pols.push(pol("POSEIDON_FULL_ROUND", 0, self.full_round));
        pols.push(pol("POSEIDON_PARTIAL_ROUND", 0, self.partial_round));
        if let Some(selector) = self.range_check {
            let table =
                (0..selector.len() as u64).map(|i| Bn128::from_int(i % (1 << RANGE_CHECK_CHUNK_BITS))).collect();
            pols.push(pol("RANGE_CHECK", 0, selector));
            pols.push(pol("RANGE", 0, table));
        }
        pols
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plonk2pil::r1cs::to_plonk::PlonkAddition;
    use crate::plonk2pil::r1cs::types::{LinearCombination, R1csConstraint, R1csHeader};

    fn q(v: i64) -> Bn128 {
        Bn128::from_int(v)
    }

    /// `l·r = o`, `l + r = o` or `2·(l + r) = o` as an r1cs constraint, by `kind` (0, 1 or 2): three
    /// coefficient sets once converted.
    fn r1cs_constraint(l: u32, r: u32, o: u32, kind: u32) -> R1csConstraint<Bn128> {
        let lc = |terms: &[(u32, i64)]| -> LinearCombination<Bn128> { terms.iter().map(|&(w, c)| (w, q(c))).collect() };
        match kind {
            0 => R1csConstraint { a: lc(&[(l, 1)]), b: lc(&[(r, 1)]), c: lc(&[(o, 1)]) },
            k => R1csConstraint { a: lc(&[(0, i64::from(k))]), b: lc(&[(l, 1), (r, 1)]), c: lc(&[(o, 1)]) },
        }
    }

    fn r1cs(
        n_vars: u32,
        n_publics: u32,
        constraints: Vec<R1csConstraint<Bn128>>,
        custom_gates: Vec<CustomGate<Bn128>>,
        custom_gates_uses: Vec<CustomGateUse>,
    ) -> R1csFile<Bn128> {
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

    fn poseidon_t(t: u64) -> CustomGate<Bn128> {
        CustomGate { template_name: "PoseidonT".into(), parameters: vec![Bn128::from_int(t)] }
    }

    fn num2bytes(n_bits: u64) -> CustomGate<Bn128> {
        CustomGate { template_name: "Num2Bytes".into(), parameters: vec![Bn128::from_int(n_bits)] }
    }

    /// A use of gate 0 on the signals `first..first + 345`.
    fn poseidon_use(first: u64) -> CustomGateUse {
        CustomGateUse { id: 0, signals: (first..first + BAND_SIGNALS as u64).collect() }
    }

    /// A use of gate `id`, `Num2Bytes(n_bits)`, of `value`: appends to `witness` its signals,
    /// `value` and its chunks of 16 bits, least significant first.
    fn num2bytes_use(witness: &mut Vec<Bn128>, id: u32, n_bits: usize, value: u128) -> CustomGateUse {
        let first = witness.len() as u64;
        witness.push(Bn128::from_int(value));
        witness.extend((0..n_chunks(n_bits)).map(|k| Bn128::from_int((value >> (16 * k)) & 0xffff)));
        CustomGateUse { id, signals: (first..witness.len() as u64).collect() }
    }

    fn options() -> PlonkOptions {
        PlonkOptions { hash_id: BN128_WRAP_FAMILY.into(), ..Default::default() }
    }

    /// The witness extended with the additions, then gathered through the map as the prover's
    /// `getCommitedPols` does: a cell the map leaves out is 0.
    fn trace(res: &SetupResult<Bn128>, witness: &[Bn128]) -> Vec<Vec<Bn128>> {
        let mut w = witness.to_vec();
        for PlonkAddition { wires, coeffs } in &res.plonk_additions {
            w.push(coeffs[0] * w[wires[0] as usize] + coeffs[1] * w[wires[1] as usize]);
        }
        res.s_map
            .iter()
            .map(|col| col.iter().map(|&s| if s == 0 { Bn128::ZERO } else { w[s as usize] }).collect())
            .collect()
    }

    fn find_column<'a>(res: &'a SetupResult<Bn128>, name: &str, index: usize) -> Option<&'a [Bn128]> {
        let name = format!("{AIRGROUP_NAME}.{name}");
        res.fixed_pols.iter().find(|p| p.name == name && p.index == index).map(|p| p.values.as_slice())
    }

    fn column<'a>(res: &'a SetupResult<Bn128>, name: &str, index: usize) -> &'a [Bn128] {
        find_column(res, name, index).unwrap_or_else(|| panic!("no {name}[{index}]"))
    }

    /// Whether PLONK gates 0 and 1 are off on each row: `POSEIDON + RANGE_CHECK`, the latter if the
    /// AIR has range checks.
    fn custom_gate_rows(res: &SetupResult<Bn128>) -> Vec<bool> {
        let in_band = column(res, "POSEIDON", 0);
        let range_check = find_column(res, "RANGE_CHECK", 0);
        (0..1 << res.n_bits)
            .map(|row| in_band[row].is_one() || range_check.is_some_and(|rc| rc[row].is_one()))
            .collect()
    }

    /// The rows where one of `wrap.pil`'s PLONK gates does not hold on `trace`.
    fn failing_gates(res: &SetupResult<Bn128>, trace: &[Vec<Bn128>]) -> Vec<(usize, usize)> {
        let c: Vec<&[Bn128]> = (0..PLONK_COEFFS).map(|k| column(res, "C", k)).collect();
        let custom = custom_gate_rows(res);
        let mut failing = Vec::new();
        for (row, &custom) in custom.iter().enumerate() {
            for g in 0..GATES_PER_ROW {
                if g != BAND_GATE && custom {
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

    /// The rows where `num2bytes.pil`'s recomposition does not hold on `trace`:
    /// `RANGE_CHECK·(a[0] − Σ K[k]·a[1 + k])`.
    fn failing_recompositions(res: &SetupResult<Bn128>, trace: &[Vec<Bn128>]) -> Vec<usize> {
        let selector = column(res, "RANGE_CHECK", 0);
        let weights: Vec<&[Bn128]> = (0..WIDTH).map(|k| column(res, "K", k)).collect();
        let recomposed = |row: usize| -> Bn128 {
            (0..RANGE_CHECK_CHUNKS)
                .fold(Bn128::ZERO, |sum, k| sum + weights[k][row] * trace[RANGE_CHECK_CHUNK_COLS.start + k][row])
        };
        (0..1 << res.n_bits).filter(|&row| !(selector[row] * (trace[0][row] - recomposed(row))).is_zero()).collect()
    }

    /// 198 constraints of three coefficient sets, with a satisfying witness: w1 = 3, w2 = 5, then
    /// w[k] is w[k-1]·w[k-2], w[k-1] + w[k-2] or 2·(w[k-1] + w[k-2]).
    fn three_sets() -> (Vec<R1csConstraint<Bn128>>, Vec<Bn128>) {
        let mut constraints = Vec::new();
        let mut witness = vec![Bn128::ONE, q(3), q(5)];
        for k in 3..201u32 {
            constraints.push(r1cs_constraint(k - 1, k - 2, k, k % 3));
            let (a, b) = (witness[k as usize - 1], witness[k as usize - 2]);
            witness.push(match k % 3 {
                0 => a * b,
                1 => a + b,
                _ => (a + b).double(),
            });
        }
        (constraints, witness)
    }

    /// [`three_sets`], a band of 69 rows and a public.
    fn three_sets_and_a_band() -> (R1csFile<Bn128>, Vec<Bn128>) {
        let (constraints, mut witness) = three_sets();
        let first_band_signal = witness.len() as u64;
        let n_vars = witness.len() as u32 + BAND_SIGNALS as u32;
        // The band's signals are zeros: the PLONK gates do not read them.
        witness.resize(n_vars as usize, Bn128::ZERO);
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
        wrong[100] += Bn128::ONE;
        assert!(!failing_gates(&res, &self::trace(&res, &wrong)).is_empty(), "a wrong wire fails a gate");
    }

    /// The same r1cs gives the same map and columns: the groups are in the constraints' order, a
    /// hash map only finds them.
    #[test]
    fn the_setup_is_deterministic() {
        let (r1cs, _) = three_sets_and_a_band();
        let [a, b] = [0, 1].map(|_| wrap(&r1cs, &options()).unwrap());
        assert_eq!(a.s_map, b.s_map);
        let columns = |res: &SetupResult<Bn128>| -> Vec<(String, usize, Vec<Bn128>)> {
            res.fixed_pols.iter().map(|p| (p.name.clone(), p.index, p.values.clone())).collect()
        };
        assert_eq!(columns(&a), columns(&b));
        assert_eq!((a.pil_str, a.plonk_additions), (b.pil_str, b.plonk_additions));
    }

    /// Left overs of 1 go to the custom gates' rows first, then left overs of 2, then whole rows;
    /// the rest takes PLONK rows of 3.
    #[test]
    fn gate_rows_take_left_overs_first() {
        let constraint = |set: i64| PlonkConstraint { wires: [1, 2, 3], coeffs: [q(set), q(0), q(0), q(-1), q(0)] };
        // Sets of 4, 2 and 6 constraints: left overs 1, 2 and 0.
        let constraints: Vec<_> = [(1, 4), (2, 2), (3, 6)].iter().flat_map(|&(s, n)| vec![constraint(s); n]).collect();
        // The PLONK rows of each set.
        let rows = |gate_rows| {
            let placement = PlonkPlacement::new(&constraints, gate_rows);
            let per_set = placement.groups.iter().zip(&placement.in_gate_rows);
            let rows: Vec<usize> = per_set.map(|(group, &b)| (group.len() - b).div_ceil(GATES_PER_ROW)).collect();
            assert_eq!(rows.iter().sum::<usize>(), placement.n_rows);
            rows
        };
        assert_eq!(rows(0), [2, 1, 2]);
        assert_eq!(rows(1), [1, 1, 2], "the left over of 1");
        assert_eq!(rows(2), [1, 1, 2], "a left over of 2 needs both");
        assert_eq!(rows(3), [1, 0, 2], "then the left over of 2");
        assert_eq!(rows(6), [0, 0, 2], "then three at a time");
        assert_eq!(rows(100), [0, 0, 0], "every constraint in a gate row");
    }

    /// A band is the use's signals, 5 a row, with its round's constants and kind; its last row is
    /// the output, of no round. With no Num2Bytes, the AIR has no range check.
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
                let rc = column(&res, "K", j)[row];
                assert_eq!(rc, if r < ROUNDS { ROUND_CONSTANTS[WIDTH * r + j] } else { Bn128::ZERO }, "row {row}");
            }
            assert_eq!(in_band[row], Bn128::ONE);
            let kind = (full[row], partial[row]);
            let expected = match r {
                ROUNDS => (Bn128::ZERO, Bn128::ZERO),
                r if !(4..64).contains(&r) => (Bn128::ONE, Bn128::ZERO),
                _ => (Bn128::ZERO, Bn128::ONE),
            };
            assert_eq!(kind, expected, "row {row}: round {r}");
        }
        assert!(res.gate_bands.is_empty() && res.band_aux == 0, "no band: the map covers the gate's");
        for name in ["PUBLICS_ROW", "RANGE_CHECK", "RANGE"] {
            assert!(find_column(&res, name, 0).is_none(), "no {name}");
        }
        assert!(res.pil_str.contains("nPublics: 0, nRangeChecks: 0"), "{}", res.pil_str);
    }

    /// A range-check row is its use's `in` and chunks, after the bands and in the r1cs's order,
    /// with the chunks' weights and the selector; the cells past the chunks are empty, of weight 0.
    /// The exec marks each with its number of chunks, and the table makes the AIR 2^16 rows.
    #[test]
    fn a_range_check_row_is_its_in_and_chunks_with_their_weights() {
        let mut witness = vec![Bn128::ONE];
        // Gate 1 is PoseidonT(5): its use's signals are zeros, which nothing here reads.
        let n_bits = [64, 0, 3, 80, 17];
        let values: [u128; 5] = [0x1234_5678_9abc_def0, 0, 5, (1 << 80) - 1, 0x1_0001];
        let mut uses = Vec::new();
        for (id, (&bits, &value)) in n_bits.iter().zip(&values).enumerate() {
            if id == 1 {
                uses.push(CustomGateUse { id: 1, ..poseidon_use(witness.len() as u64) });
                witness.resize(witness.len() + BAND_SIGNALS, Bn128::ZERO);
            } else {
                uses.push(num2bytes_use(&mut witness, id as u32, bits, value));
            }
        }
        let gates = vec![num2bytes(64), poseidon_t(5), num2bytes(3), num2bytes(80), num2bytes(17)];
        let res = wrap(&r1cs(witness.len() as u32, 0, vec![], gates, uses.clone()), &options()).unwrap();

        let rc_uses = [(&uses[0], 4), (&uses[2], 1), (&uses[3], 5), (&uses[4], 2)];
        assert_eq!(res.n_used, BAND_ROWS + rc_uses.len());
        assert_eq!((res.n_bits, res.n_bits_natural), (16, 16), "the table's 2^16 rows");
        assert!(res.pil_str.contains("nRangeChecks: 4"), "{}", res.pil_str);
        assert_eq!(res.band_aux, RANGE_MUL_COLUMN as u64, "the exec names RANGE_MUL's column");
        let bands: Vec<(u32, u64)> = res.gate_bands.iter().map(|b| (b.row, b.payload)).collect();
        assert_eq!(bands, [(69, 4), (70, 1), (71, 5), (72, 2)]);
        assert!(res.gate_bands.iter().all(|b| b.kind == GateBandKind::PoseidonBn128WrapRangeCheck));

        let (selector, table, in_band) =
            (column(&res, "RANGE_CHECK", 0), column(&res, "RANGE", 0), column(&res, "POSEIDON", 0));
        for (i, &(cgu, n_chunks)) in rc_uses.iter().enumerate() {
            let row = BAND_ROWS + i;
            assert_eq!((selector[row], in_band[row]), (Bn128::ONE, Bn128::ZERO), "row {row}");
            assert_eq!(res.s_map[0][row] as u64, cgu.signals[0], "in, row {row}");
            for k in 0..RANGE_CHECK_CHUNKS {
                let (cell, weight) = (res.s_map[1 + k][row] as u64, column(&res, "K", k)[row]);
                if k < n_chunks {
                    assert_eq!((cell, weight), (cgu.signals[1 + k], Bn128::from_int(1u128 << (16 * k))), "row {row}");
                } else {
                    assert_eq!((cell, weight), (0, Bn128::ZERO), "past the chunks, row {row}");
                }
            }
            assert!((6..N_WIRES).all(|col| res.s_map[col][row] == 0), "no PLONK constraint to place");
        }
        let rc_rows = BAND_ROWS..BAND_ROWS + rc_uses.len();
        assert!((0..1 << 16).all(|row| selector[row].is_one() == rc_rows.contains(&row)));
        assert!((0..1u64 << 16).all(|i| table[i as usize] == Bn128::from_int(i)), "RANGE is 0..2^16");
        assert_eq!(failing_recompositions(&res, &trace(&res, &witness)), Vec::<usize>::new());
    }

    /// The range-check rows take PLONK constraints on gate 2 as band rows do, and their gates and
    /// recompositions hold on a satisfying witness; a wrong chunk fails its row's recomposition.
    #[test]
    fn the_plonk_gates_and_range_checks_hold_on_the_witness() {
        const N_RANGE_CHECKS: usize = 30;
        let (constraints, mut witness) = three_sets();
        let uses: Vec<CustomGateUse> = (0..N_RANGE_CHECKS as u128)
            .map(|i| {
                num2bytes_use(&mut witness, 0, 64, 0x9e37_79b9_7f4a_7c15u128.wrapping_mul(i + 1) & u128::from(u64::MAX))
            })
            .collect();
        let r1cs = r1cs(witness.len() as u32, 1, constraints, vec![num2bytes(64)], uses);
        let res = wrap(&r1cs, &options()).unwrap();
        let in_gate_rows = (0..N_RANGE_CHECKS).filter(|&row| !column(&res, "C", 3)[row].is_zero()).count();
        assert_eq!(in_gate_rows, N_RANGE_CHECKS, "every range-check row holds a constraint");
        assert_eq!(res.n_used, N_RANGE_CHECKS + (198 - N_RANGE_CHECKS) / 3 + 1);

        let trace = trace(&res, &witness);
        assert_eq!(failing_gates(&res, &trace), Vec::<(usize, usize)>::new());
        assert_eq!(failing_recompositions(&res, &trace), Vec::<usize>::new());
        let mut wrong = trace.clone();
        wrong[2][7] += Bn128::ONE;
        assert_eq!(failing_recompositions(&res, &wrong), [7], "a wrong chunk");
    }

    /// Each public is a cell of the rows after the gates, 9 a row, and each public row has a
    /// selector of its own, `PUBLICS_ROW[k]`.
    #[test]
    fn the_publics_follow_the_gates_a_selector_a_row() {
        let res = wrap(&r1cs(4, 2, vec![r1cs_constraint(1, 2, 3, 0)], vec![], vec![]), &options()).unwrap();
        let publics_row = column(&res, "PUBLICS_ROW", 0);
        let first = publics_row.iter().position(|v| *v == Bn128::ONE).unwrap();
        assert_eq!(first, res.n_used - 1, "one public row, after the one PLONK row");
        assert_eq!((res.s_map[0][first], res.s_map[1][first]), (1, 2));
        assert!(find_column(&res, "PUBLICS_ROW", 1).is_none());

        let res = wrap(&r1cs(16, 14, vec![r1cs_constraint(1, 2, 15, 0)], vec![], vec![]), &options()).unwrap();
        let first = res.n_used - 2;
        for k in 0..2 {
            let ones: Vec<usize> =
                (0..1 << res.n_bits).filter(|&row| column(&res, "PUBLICS_ROW", k)[row].is_one()).collect();
            assert_eq!(ones, [first + k], "PUBLICS_ROW[{k}]");
        }
        for public in 0..14 {
            assert_eq!(res.s_map[public % N_WIRES][first + public / N_WIRES] as usize, public + 1, "public {public}");
        }
        assert!(res.pil_str.contains("nPublics: 14"), "{}", res.pil_str);
    }

    #[test]
    fn other_gates_widths_and_uses_are_refused() {
        let refusal = |gates, uses| wrap(&r1cs(400, 0, vec![], gates, uses), &options()).unwrap_err().to_string();

        let cmul = CustomGate { template_name: "CMul".into(), parameters: vec![] };
        assert!(refusal(vec![cmul], vec![]).ends_with("custom gate 0 is CMul"));
        let unknown = CustomGate { template_name: "Keccak".into(), parameters: vec![] };
        assert!(refusal(vec![unknown], vec![]).ends_with("custom gate 0 is Keccak"));
        assert!(refusal(vec![poseidon_t(3)], vec![]).ends_with("custom gate 0 is PoseidonT(3)"));
        assert!(refusal(vec![num2bytes(0)], vec![]).ends_with("custom gate 0 is Num2Bytes(0)"));
        assert!(refusal(vec![num2bytes(81)], vec![]).ends_with("custom gate 0 is Num2Bytes(81)"));
        let no_bits = CustomGate { template_name: "Num2Bytes".into(), parameters: vec![] };
        assert!(refusal(vec![no_bits], vec![]).ends_with("custom gate 0 is Num2Bytes"));
        let huge = CustomGate { template_name: "Num2Bytes".into(), parameters: vec![Bn128::NEG_ONE] };
        assert!(refusal(vec![huge], vec![]).contains("custom gate 0 is Num2Bytes(2188824287183927522224640574525"));

        let short = CustomGateUse { id: 0, signals: vec![1; 10] };
        assert!(refusal(vec![poseidon_t(5)], vec![short]).contains("has 10 signals, not the 345 of its in, im and out"));
        let short = CustomGateUse { id: 0, signals: vec![1; 4] };
        assert!(refusal(vec![num2bytes(65)], vec![short]).contains("use 0 of Num2Bytes(65) has 4 signals, not the 6"));
        let of_no_gate = CustomGateUse { id: 1, ..poseidon_use(1) };
        assert!(refusal(vec![poseidon_t(5)], vec![of_no_gate]).contains("is of gate 1, and the r1cs defines 1"));
        assert!(refusal(vec![poseidon_t(5)], vec![poseidon_use(100)]).contains("reads signal 400"));
        let past = CustomGateUse { id: 0, signals: vec![1, 2, 3, 400] };
        assert!(refusal(vec![num2bytes(48)], vec![past]).contains("use 0 of Num2Bytes(48) reads signal 400"));
    }
}
