//! The Poseidon1 BN128 wrap's layout: bands, range checks, plonk leftovers and publics.

use proofman_common::exec_format::{RANGE_CHECK_CHUNK_BITS, RANGE_CHECK_CHUNK_COLS};
use proofman_common::hash_family::HashFamily;
use proofman_fields::{Bn128, Field, QuotientMap};

use super::layouts::{BAND_ROWS, RANGE_MUL_COLUMN};
use super::place::plan_groups;
use crate::plonk2pil::consts::{ROUNDS, ROUND_CONSTANTS, WIDTH};
use super::{setup, SetupKind};
use crate::plonk2pil::r1cs::to_plonk::{PlonkAddition, PlonkConstraint, PLONK_COEFFS};
use crate::plonk2pil::r1cs::types::{
    CustomGate, CustomGateUse, GateBandKind, LinearCombination, PlonkOptions, R1csConstraint, R1csFile, R1csHeader,
    SetupResult,
};
use anyhow::Result;

const AIRGROUP_NAME: &str = "Wrap";
const N_WIRES: usize = 9;
const GATES_PER_ROW: usize = 3;
const BAND_GATE: usize = 2;
const BAND_SIGNALS: usize = WIDTH * BAND_ROWS;
const RANGE_CHECK_CHUNKS: usize = RANGE_CHECK_CHUNK_COLS.end - RANGE_CHECK_CHUNK_COLS.start;

fn n_chunks(n_bits: usize) -> usize {
    n_bits.div_ceil(RANGE_CHECK_CHUNK_BITS as usize)
}

fn wrap(r1cs: &R1csFile<Bn128>, options: &PlonkOptions) -> Result<SetupResult<Bn128>> {
    setup(r1cs, SetupKind::Wrap, options)
}

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
    PlonkOptions { hash: HashFamily::Poseidon1, ..Default::default() }
}

/// The witness extended with the additions, then gathered through the map as the prover's
/// `getWitnessTrace` does: a cell the map leaves out is 0.
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
    (0..1 << res.n_bits).map(|row| in_band[row].is_one() || range_check.is_some_and(|rc| rc[row].is_one())).collect()
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
        let (groups, in_gate_rows) = plan_groups(&constraints, GATES_PER_ROW, false, gate_rows);
        let per_set = groups.iter().zip(&in_gate_rows);
        per_set.map(|(group, &b)| (group.len() - b).div_ceil(GATES_PER_ROW)).collect::<Vec<usize>>()
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
