//! Every AIR plonk2pil builds, as data: one function per kind, the hash family a parameter.
//!
//! The row-kind masks MIRROR the plonk selectors of the PIL templates named here; `PlonkBand`
//! refuses any placement outside them.

use anyhow::{bail, ensure, Result};
use proofman_common::exec_format::{
    BLAKE3_WRAP_BLOCK_ROWS, BLAKE3_WRAP_RANGE_CHECK_CELLS, BLAKE3_WRAP_RANGE_CHECK_SLOTS, RANGE_CHECK_CHUNK_BITS,
    RANGE_CHECK_CHUNK_COLS,
};
use proofman_common::hash_family::{lookup_gate, GateRole, HashFamily};
use proofman_fields::PrimeField;

use super::blocks::{stage1_cols, BAND_COLS, C_COLS, CMUL_PER_ROW, EVPOL4_SIGNALS, PLONK_GATES_PER_ROW};
use crate::plonk2pil::consts::{ROUNDS, WIDTH};
use super::spec::{
    rows_of, AirLayout, Blocks, ConstFill, GateGroup, Gates, Init, Param, PlonkPolicy, RowKind, Run, Section, SetupKind,
};
use crate::plonk2pil::r1cs::types::{CustomGate, GateBandKind, R1csFile};

/// Rows of a `PoseidonT(5)` band: the state before each round, and the output.
pub const BAND_ROWS: usize = ROUNDS + 1;

/// The stage-1 column of the Poseidon BN128 wrap's range-check multiplicity, `RANGE_MUL`: the one
/// after its 9 wires.
pub const RANGE_MUL_COLUMN: usize = 9;

/// The widest `Num2Bytes` a wrap places: 5 chunks of 16 bits.
pub const MAX_RANGE_CHECK_BITS: usize = (BLAKE3_WRAP_RANGE_CHECK_CELLS - 1) * RANGE_CHECK_CHUNK_BITS as usize;

const _: () = assert!(RANGE_CHECK_CHUNK_COLS.start == 1 && RANGE_CHECK_CHUNK_COLS.end - 1 <= WIDTH);
const _: () = assert!(BLAKE3_WRAP_RANGE_CHECK_SLOTS * BLAKE3_WRAP_RANGE_CHECK_CELLS <= BAND_COLS);

/// What the pilfflonk setup of a BN128 wrap takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrapKnobs {
    /// The PIL's `set_max_constraint_degree` by default, and the bound of pilfflonk's degree search.
    pub max_constraint_degree: usize,
    /// pilfflonk's `--extra-muls`.
    pub extra_muls: u64,
}

/// The knobs of the BN128 wrap of `hash`: Poseidon1 (`PoseidonT`, degree 6 with 3 extra muls, the
/// cheapest that verifies, 337,557 gas on fibonacci-square) or blake3 (degree 5 and 4, the fewest a
/// 2^24 ptau holds).
pub fn wrap_knobs(hash: HashFamily) -> Result<WrapKnobs> {
    match hash {
        HashFamily::Poseidon1 => Ok(WrapKnobs { max_constraint_degree: 6, extra_muls: 3 }),
        HashFamily::Blake3 => Ok(WrapKnobs { max_constraint_degree: 5, extra_muls: 4 }),
        HashFamily::Poseidon2 => {
            bail!("plonk2pil: there is no BN128 wrap of {hash}: Poseidon1's wraps every Poseidon key")
        }
    }
}

fn group(coef: usize, gates: std::ops::Range<usize>) -> GateGroup {
    GateGroup { coef, gates }
}

fn run(signal: usize, len: usize, col: usize, row: usize) -> Run {
    Run { signal, len, col, row }
}

/// The GL layouts: Poseidon1/Poseidon2 compressor and aggregator, blake3's aggregator at `lanes`
/// (its compressor is the aggregator at a geometry of its own).
pub fn stark(kind: SetupKind, family: HashFamily, lanes: usize) -> AirLayout {
    match family {
        HashFamily::Blake3 => blake3_aggregator(kind, lanes),
        _ => poseidon(kind, family),
    }
}

/// The Poseidon1/Poseidon2 airs: `compressor.pil` (a[46]/S[30], 10-row hash) or `aggregator.pil`
/// (a[48]/S[24], 5-row hash). P1 and P2 differ only in the data marked so.
fn poseidon(kind: SetupKind, family: HashFamily) -> AirLayout {
    use GateBandKind as B;
    use GateRole as R;
    let p1 = family == HashFamily::Poseidon1;
    let compressor = kind == SetupKind::Compressor;
    // P1: `TreeSelector8` (30 signals), P2: `TreeSelector4` (17).
    let tree_signals = if p1 { 30 } else { 17 };
    let pil_params = vec![
        ("nPlonkRows", Param::PlonkRows),
        (if p1 { "nPoseidon1Compression" } else { "nPoseidonCompressor" }, Param::Uses(R::PoseidonCompression)),
        (if p1 { "nPoseidon1Sponge" } else { "nPoseidonSponge" }, Param::Uses(R::PoseidonSponge)),
        ("nCMulRows", Param::Rows(R::CMul)),
        ("nEvPol4", Param::Uses(R::EvPol4)),
        ("nFFT4", Param::Uses(R::Fft4)),
        (if p1 { "nTreeSelector8" } else { "nTreeSelector4" }, Param::Uses(R::TreeSelector)),
        ("nSelectValArity4", Param::Uses(R::SelectValArity4)),
    ];
    let cmul = |per_row: usize, row_kinds: Vec<(usize, usize)>| Section {
        per_row,
        stride: 9,
        row_kinds,
        ..Section::plain(Gates::Role(R::CMul), 9, 1, vec![run(0, 9, 0, 0)])
    };
    let evpol = |kind: usize| Section {
        row_kinds: vec![(0, kind)],
        ..Section::plain(Gates::Role(R::EvPol4), EVPOL4_SIGNALS, 1, vec![run(0, 21, 0, 0)])
    };
    let fft4 = Section { fill: ConstFill::Fft4, ..Section::plain(Gates::Role(R::Fft4), 24, 1, vec![run(0, 24, 0, 0)]) };

    if compressor {
        // Plonk band a[0..29] = 10 gates: q0 = gates 0..5, q1 = gates 6..9; the chain a[30..45] is off-band.
        let rk = |mask, groups| RowKind { mask, groups };
        let q01 = |end: usize| vec![group(0, 0..6), group(5, end..10)];
        let row_kinds = vec![
            rk(0x3FF, q01(6)),                // 0: inner full rounds
            rk(0x33F, q01(8)),                // 1: PR (a[18..23] anchors)
            rk(0x3C0, vec![group(5, 6..10)]), // 2: INIT, FINAL (+ P2's TreeSelector4)
            rk(0x380, vec![group(5, 7..10)]), // 3: resEVPOL on gate 6
            rk(0x300, vec![group(5, 8..10)]), // 4: SelectValueArity4
        ];
        let pure = rk(0x3FF, q01(6));
        let hash_rows: Vec<(usize, usize)> = (0..10).map(|o| (o, [2, 0, 0, 0, 0, 1, 0, 0, 0, 2][o])).collect();
        let hash = |role, n: usize, band, cells| Section {
            band: Some(band),
            row_kinds: hash_rows.clone(),
            ..Section::plain(Gates::Role(role), n, 10, cells)
        };
        let (comp, sponge) = if p1 {
            (B::Poseidon1CompressorCompression, B::Poseidon1CompressorSponge)
        } else {
            (B::Poseidon2CompressorCompression, B::Poseidon2CompressorSponge)
        };
        let tree = Section {
            row_kinds: if p1 { vec![] } else { vec![(0, 2)] },
            ..Section::plain(Gates::Role(R::TreeSelector), tree_signals, 1, vec![run(0, tree_signals, 0, 0)])
        };
        AirLayout {
            kind,
            family,
            committed: 46,
            conn_cols: 30,
            c_cols: 10,
            pil_file: if p1 { "poseidon1/compressor" } else { "poseidon2/compressor" },
            pil_template: "Compressor",
            pil_params,
            max_degree: 5,
            degree_fixed: true,
            default_name: "Compressor",
            stamped: true,
            blocks: Blocks::None,
            sections: vec![
                // in, key at a[16..17] of INIT, out at FINAL; the chain is the expander's.
                hash(R::PoseidonCompression, 34, comp, vec![run(0, 16, 0, 0), run(16, 2, 16, 0), run(18, 16, 0, 9)]),
                hash(R::PoseidonSponge, 32, sponge, vec![run(0, 16, 0, 0), run(16, 16, 0, 9)]),
                cmul(3, vec![]),
                evpol(3),
                fft4,
                tree,
                Section {
                    row_kinds: vec![(0, 4)],
                    ..Section::plain(Gates::Role(R::SelectValArity4), 22, 1, vec![run(0, 22, 0, 0)])
                },
            ],
            row_kinds,
            plonk: PlonkPolicy::Stream { pure },
            fixed: vec![("C", 10, Init::Zero), ("S", 30, Init::Zero)],
            publics_row_fixed: false,
            min_bits: 0,
            band_aux: 0,
        }
    } else {
        // Plonk band a[0..23] = 8 gates: q0 = gates 0,1, q1 = gates 2..7; the chains are a[16..47].
        let rk = |mask, groups| RowKind { mask, groups };
        let row_kinds = vec![
            rk(0x1F, vec![group(0, 0..2), group(5, 2..5)]), // 0: PR', FINAL'
            rk(0x07, vec![group(0, 0..2), group(5, 2..3)]), // 1: PR (a[9..14] anchors)
            rk(0xC0, vec![group(5, 6..8)]),                 // 2: CMul (+ P2's TreeSelector4)
            rk(0x80, vec![group(5, 7..8)]),                 // 3: EvPol4
        ];
        let hash = |role, n: usize, band, cells| Section {
            band: Some(band),
            row_kinds: vec![(1, 0), (2, 1), (3, 0)],
            ..Section::plain(Gates::Role(role), n, 5, cells)
        };
        let (comp, sponge) = if p1 {
            (B::Poseidon1AggregationCompression, B::Poseidon1AggregationSponge)
        } else {
            (B::Poseidon2AggregationCompression, B::Poseidon2AggregationSponge)
        };
        // key0 at a[15] of PR', key1 at a[15] of PR.
        let comp = hash(
            R::PoseidonCompression,
            34,
            comp,
            vec![run(0, 16, 0, 0), run(16, 1, 15, 1), run(17, 1, 15, 2), run(18, 16, 0, 4)],
        );
        let sponge = hash(R::PoseidonSponge, 32, sponge, vec![run(0, 16, 0, 0), run(16, 16, 0, 4)]);
        // P1 compresses first, P2 absorbs first: the order the PIL's selectors lay them out in.
        let hashes = if p1 { [comp, sponge] } else { [sponge, comp] };
        // P1's 30 signals split 24 on the gate row, 6 on the next.
        let tree = if p1 {
            Section::plain(Gates::Role(R::TreeSelector), 30, 2, vec![run(0, 24, 0, 0), run(24, 6, 0, 1)])
        } else {
            Section {
                row_kinds: vec![(0, 2)],
                ..Section::plain(Gates::Role(R::TreeSelector), 17, 1, vec![run(0, 17, 0, 0)])
            }
        };
        let mut sections = hashes.to_vec();
        sections.extend([
            cmul(2, vec![(0, 2)]),
            evpol(3),
            fft4,
            tree,
            Section::plain(Gates::Role(R::SelectValArity4), 22, 1, vec![run(0, 22, 0, 0)]),
        ]);
        AirLayout {
            kind,
            family,
            committed: 48,
            conn_cols: 24,
            c_cols: 10,
            pil_file: if p1 { "poseidon1/aggregator" } else { "poseidon2/aggregator" },
            pil_template: "Aggregator",
            pil_params,
            max_degree: 8,
            degree_fixed: false,
            default_name: "Compressor",
            stamped: true,
            blocks: Blocks::None,
            sections,
            row_kinds,
            plonk: PlonkPolicy::Stream { pure: rk(0xFF, vec![group(0, 0..2), group(5, 2..8)]) },
            fixed: vec![("C", 10, Init::Zero), ("S", 24, Init::Zero)],
            publics_row_fixed: false,
            min_bits: 0,
            band_aux: 0,
        }
    }
}

/// blake3's aggregator (`blake3/aggregator.pil`) at `lanes`: the 56-row blocks, then each band
/// circuit in whole runs of block interiors, plonk last, one coefficient set a row.
fn blake3_aggregator(kind: SetupKind, lanes: usize) -> AirLayout {
    use GateRole as R;
    AirLayout {
        kind,
        family: HashFamily::Blake3,
        committed: stage1_cols(lanes),
        conn_cols: BAND_COLS,
        c_cols: C_COLS,
        pil_file: "blake3/aggregator",
        pil_template: "Aggregator",
        pil_params: vec![
            ("nPublicsStart", Param::PublicsStart),
            ("nPlonkRows", Param::PlonkRows),
            ("nCMulRows", Param::Rows(R::CMul)),
            ("nEvPol4", Param::Uses(R::EvPol4)),
            ("nFFT4", Param::Uses(R::Fft4)),
            ("nTreeSelector4", Param::Uses(R::TreeSelector)),
            ("nSelValRows", Param::Uses(R::SelectValArity2)),
            ("nNodeBlocks", Param::NodeBlocks),
            ("nChunkBlocks", Param::ChunkBlocks),
            ("nParentBlocks", Param::ParentBlocks),
            ("LANES", Param::Lanes),
        ],
        max_degree: 5,
        degree_fixed: false,
        default_name: "Blake3Agg",
        stamped: true,
        blocks: Blocks::Blake3 { lanes },
        sections: vec![
            Section {
                per_row: CMUL_PER_ROW,
                stride: 9,
                ..Section::plain(Gates::Role(R::CMul), 9, 1, vec![run(0, 9, 0, 0)])
            },
            // The outputs on the next row, which the PIL reads through primes.
            Section::plain(Gates::Role(R::EvPol4), EVPOL4_SIGNALS, 2, rows_of(EVPOL4_SIGNALS, BAND_COLS)),
            Section {
                fill: ConstFill::Fft4,
                ..Section::plain(Gates::Role(R::Fft4), 24, 2, vec![run(0, 12, 0, 0), run(12, 12, 0, 1)])
            },
            Section::plain(Gates::Role(R::TreeSelector), 17, 1, vec![run(0, 17, 0, 0)]),
            Section::plain(Gates::Role(R::SelectValArity2), 13, 1, vec![run(0, 13, 0, 0)]),
        ],
        row_kinds: vec![],
        plonk: PlonkPolicy::GroupByKey { gates_per_row: PLONK_GATES_PER_ROW, sorted: true, selector: None },
        fixed: vec![("C", C_COLS, Init::Zero), ("S", BAND_COLS, Init::Zero), ("FLAGS", 1, Init::Zero)],
        publics_row_fixed: false,
        min_bits: 0,
        // LANES low, band width high: the expander reads where every lane column starts from them.
        band_aux: (lanes as u64) | ((BAND_COLS as u64) << 32),
    }
}

/// The BN128 wrap of `hash` for `r1cs`, or why it cannot place one of its gates.
pub fn wrap<F: PrimeField>(hash: HashFamily, r1cs: &R1csFile<F>) -> Result<AirLayout> {
    let knobs = wrap_knobs(hash)?;
    let chunks = admit_wrap(hash, r1cs)?;
    Ok(match hash {
        HashFamily::Blake3 => blake3_wrap(&chunks, knobs),
        _ => poseidon_wrap(!chunks.is_empty(), knobs),
    })
}

/// The Poseidon1 BN128 wrap (`poseidon_bn128/wrap.pil`, layout L1): 9 wires, all connected, 3 gates
/// a row on one coefficient set; a `PoseidonT(5)` use is a 69-row band of its signals 5 a row, a
/// `Num2Bytes` use a range-check row; on both, gate 2 is the only plonk gate on.
fn poseidon_wrap(range_checks: bool, knobs: WrapKnobs) -> AirLayout {
    let mut fixed = vec![("C", 5, Init::Zero), ("S", 9, Init::Zero), ("K", WIDTH, Init::Zero)];
    fixed.extend([
        ("POSEIDON", 1, Init::Zero),
        ("POSEIDON_FULL_ROUND", 1, Init::Zero),
        ("POSEIDON_PARTIAL_ROUND", 1, Init::Zero),
    ]);
    if range_checks {
        fixed.extend([("RANGE_CHECK", 1, Init::Zero), ("RANGE", 1, Init::RangeTable)]);
    }
    let band_signals = WIDTH * BAND_ROWS;
    AirLayout {
        kind: SetupKind::Wrap,
        family: HashFamily::Poseidon1,
        committed: 9,
        conn_cols: 9,
        c_cols: 5,
        pil_file: "poseidon_bn128/wrap",
        pil_template: "Wrap",
        pil_params: vec![("nRangeChecks", Param::Uses(GateRole::RangeCheck))],
        max_degree: knobs.max_constraint_degree,
        degree_fixed: false,
        default_name: "Wrap",
        stamped: false,
        blocks: Blocks::None,
        sections: vec![
            Section {
                fill: ConstFill::PoseidonT,
                row_kinds: (0..BAND_ROWS).map(|o| (o, 0)).collect(),
                ..Section::plain(
                    Gates::Role(GateRole::PoseidonT),
                    band_signals,
                    BAND_ROWS,
                    rows_of(band_signals, WIDTH),
                )
            },
            // `in` on a[0], its chunks from a[1].
            Section {
                n_signals: None,
                band: Some(GateBandKind::PoseidonBn128WrapRangeCheck),
                fill: ConstFill::RangeCheck { weights: "K" },
                row_kinds: vec![(0, 0)],
                ..Section::plain(
                    Gates::Role(GateRole::RangeCheck),
                    0,
                    1,
                    vec![run(0, RANGE_CHECK_CHUNK_COLS.end, 0, 0)],
                )
            },
        ],
        row_kinds: vec![RowKind { mask: 1 << 2, groups: vec![group(0, 2..3)] }],
        plonk: PlonkPolicy::GroupByKey { gates_per_row: 3, sorted: false, selector: None },
        fixed,
        publics_row_fixed: true,
        // The range table `RANGE` is 2^16 rows.
        min_bits: if range_checks { RANGE_CHECK_CHUNK_BITS as usize } else { 0 },
        band_aux: if range_checks { RANGE_MUL_COLUMN as u64 } else { 0 },
    }
}

/// The blake3 BN128 wrap (`blake3_bn128/wrap.pil`): a 64-row block per BLAKE3 compression, then on
/// the rows the blocks leave free, the range checks (three `Num2Bytes` of `chunks` chunks a row) and
/// the plonk rows (six constraints of one coefficient set). Every selector is a fixed column here.
fn blake3_wrap(chunks: &[usize], knobs: WrapKnobs) -> AirLayout {
    let sections = chunks
        .iter()
        .map(|&n| Section {
            per_row: BLAKE3_WRAP_RANGE_CHECK_SLOTS,
            stride: BLAKE3_WRAP_RANGE_CHECK_CELLS,
            pad: true,
            band: Some(GateBandKind::Blake3Bn128WrapRangeCheck),
            fill: ConstFill::RangeCheck { weights: "C" },
            ..Section::plain(Gates::RangeChunks(n), 1 + n, 1, vec![run(0, 1 + n, 0, 0)])
        })
        .collect();
    let one = |name| (name, 1, Init::Zero);
    let mut fixed = vec![("S", BAND_COLS, Init::Zero), ("C", C_COLS, Init::Zero), one("PLONK"), one("RANGE_CHECK")];
    fixed.extend([("G", 8, Init::Zero), one("HOLD"), one("INIT_NODE"), one("INIT_CHUNK"), one("INIT_PARENT")]);
    fixed.extend([one("IN_NODE"), one("IN_CHUNK"), one("IN_PARENT"), one("BLOCKLEN_ROW"), one("FLAGS")]);
    fixed.extend([one("MSG_KEY_X"), one("MSG_KEY_Y"), one("MSG_MUL"), ("FF", 8, Init::Zero), one("FF_NODE")]);
    fixed.extend([one("FF_CV"), ("FF_IV", 2, Init::Zero)]);
    AirLayout {
        kind: SetupKind::Wrap,
        family: HashFamily::Blake3,
        committed: BAND_COLS,
        conn_cols: BAND_COLS,
        c_cols: C_COLS,
        pil_file: "blake3_bn128/wrap",
        pil_template: "Wrap",
        pil_params: vec![],
        max_degree: knobs.max_constraint_degree,
        degree_fixed: false,
        default_name: "Wrap",
        stamped: false,
        blocks: Blocks::Blake3Wrap,
        sections,
        row_kinds: vec![],
        plonk: PlonkPolicy::GroupByKey { gates_per_row: PLONK_GATES_PER_ROW, sorted: true, selector: Some("PLONK") },
        fixed,
        publics_row_fixed: true,
        // blake3Tables' XOR table is 2^17 rows.
        min_bits: 17,
        band_aux: BLAKE3_WRAP_BLOCK_ROWS as u64,
    }
}

/// The gate's one parameter, if it has one that fits a `usize`.
fn parameter<F: PrimeField>(gate: &CustomGate<F>) -> Option<usize> {
    match gate.parameters.as_slice() {
        [p] => usize::try_from(&p.as_canonical_biguint()).ok(),
        _ => None,
    }
}

/// Checks every gate and use of `r1cs` is one `hash`'s wrap places; returns the `Num2Bytes` chunk
/// counts used, ascending.
fn admit_wrap<F: PrimeField>(hash: HashFamily, r1cs: &R1csFile<F>) -> Result<Vec<usize>> {
    let blake3 = hash == HashFamily::Blake3;
    // Per gate: the signals of a use and, for Num2Bytes, its chunks.
    let mut gates: Vec<(usize, Option<usize>, String)> = Vec::new();
    for (id, gate) in r1cs.custom_gates.iter().enumerate() {
        let p = parameter(gate);
        let label = match &gate.parameters[..] {
            [] => gate.template_name.clone(),
            ps => {
                format!("{}({})", gate.template_name, ps.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", "))
            }
        };
        let role = lookup_gate(&gate.template_name).map(|(role, _)| role);
        gates.push(match (role, p) {
            (Some(GateRole::Blake3Node | GateRole::Blake3Compress), _) if blake3 => (0, None, label),
            (Some(GateRole::PoseidonT), Some(WIDTH)) if !blake3 => (WIDTH * BAND_ROWS, None, label),
            (Some(GateRole::RangeCheck), Some(bits)) if (1..=MAX_RANGE_CHECK_BITS).contains(&bits) => {
                let chunks = bits.div_ceil(RANGE_CHECK_CHUNK_BITS as usize);
                (1 + chunks, Some(chunks), label)
            }
            _ => bail!(
                "plonk2pil: the {hash} BN128 wrap places {} and Num2Bytes(nBits), 0 < nBits <= {MAX_RANGE_CHECK_BITS}, \
                 only, and custom gate {id} is {label}",
                if blake3 { "Blake3Node, Blake3Compress".to_string() } else { format!("PoseidonT({WIDTH})") }
            ),
        });
    }
    let n_vars = u64::from(r1cs.header.n_vars);
    let mut chunks = std::collections::BTreeSet::new();
    for (i, cgu) in r1cs.custom_gates_uses.iter().enumerate() {
        let Some((n_signals, n_chunks, label)) = gates.get(cgu.id as usize) else {
            bail!("plonk2pil: custom gate use {i} is of gate {}, and the r1cs defines {}", cgu.id, gates.len());
        };
        if *n_signals > 0 {
            let what = if n_chunks.is_some() { "in and out" } else { "in, im and out" };
            ensure!(
                cgu.signals.len() == *n_signals,
                "plonk2pil: use {i} of {label} has {} signals, not the {n_signals} of its {what}",
                cgu.signals.len()
            );
        }
        if let Some(signal) = cgu.signals.iter().find(|&&s| s >= n_vars) {
            bail!("plonk2pil: use {i} of {label} reads signal {signal}, and the r1cs has {n_vars}");
        }
        if let Some(n) = n_chunks {
            ensure!(
                !blake3 || !cgu.signals.contains(&0),
                "plonk2pil: a Num2Bytes signal of use {i} is the constant wire 0"
            );
            chunks.insert(*n);
        }
    }
    Ok(chunks.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    /// Each P1/P2 pair differs only in the data the plan lists: names, the TreeSelector radix, the
    /// hash order and band kinds.
    #[test]
    fn p1_and_p2_layouts_differ_only_in_listed_fields() {
        for kind in [SetupKind::Compressor, SetupKind::Aggregation] {
            let (p1, p2) = (poseidon(kind, HashFamily::Poseidon1), poseidon(kind, HashFamily::Poseidon2));
            let masks = |l: &AirLayout| l.row_kinds.iter().map(|k| (k.mask, k.groups.clone())).collect::<Vec<_>>();
            assert_eq!(
                (p1.committed, p1.conn_cols, p1.c_cols, &p1.fixed),
                (p2.committed, p2.conn_cols, p2.c_cols, &p2.fixed)
            );
            assert_eq!(masks(&p1), masks(&p2), "{kind}: the same plonk band");
            let gates = |l: &AirLayout| l.sections.iter().map(|s| s.gates).collect::<HashSet<_>>();
            assert_eq!(gates(&p1), gates(&p2));
        }
    }

    /// Every mask covers exactly the gates its groups open: the guard and the placement agree.
    #[test]
    fn masks_are_the_union_of_their_groups() {
        let layouts = [
            poseidon(SetupKind::Compressor, HashFamily::Poseidon2),
            poseidon(SetupKind::Aggregation, HashFamily::Poseidon1),
            poseidon_wrap(true, wrap_knobs(HashFamily::Poseidon1).unwrap()),
        ];
        for l in &layouts {
            let pure = match &l.plonk {
                PlonkPolicy::Stream { pure } => Some(pure),
                PlonkPolicy::GroupByKey { .. } => None,
            };
            for k in l.row_kinds.iter().chain(pure) {
                let union = k.groups.iter().flat_map(|g| g.gates.clone()).fold(0u16, |m, g| m | 1 << g);
                assert_eq!(k.mask, union, "{} {}: {k:?}", l.pil_file, l.kind);
            }
        }
    }

    fn pil_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("plonk2pil/pil")
    }

    /// Names the `airtemplate <name> (...)` header of `file` declares.
    fn airtemplate_params(file: &str, template: &str) -> HashSet<String> {
        let src = std::fs::read_to_string(pil_dir().join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
        let head = src
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("airtemplate {template} ")))
            .unwrap_or_else(|| panic!("{file} has no `airtemplate {template}`"));
        let args = &head[head.find('(').unwrap() + 1..head.rfind(')').unwrap()];
        args.split(',').map(|a| a.trim().rsplit(' ').next().unwrap().to_string()).filter(|a| !a.is_empty()).collect()
    }

    /// Every argument the generated call passes is one the airtemplate declares: a renamed
    /// parameter is otherwise invisible until `compile-pil` runs on a real r1cs.
    #[test]
    fn generated_calls_match_their_airtemplates() {
        let knobs = |h| wrap_knobs(h).unwrap();
        let mut layouts = vec![blake3_aggregator(SetupKind::Aggregation, 4)];
        for family in [HashFamily::Poseidon1, HashFamily::Poseidon2] {
            layouts.extend([SetupKind::Compressor, SetupKind::Aggregation].map(|k| poseidon(k, family)));
        }
        layouts
            .extend([poseidon_wrap(true, knobs(HashFamily::Poseidon1)), blake3_wrap(&[4], knobs(HashFamily::Blake3))]);
        for l in &layouts {
            let params: Vec<(&str, usize)> = l.pil_params.iter().map(|&(n, _)| (n, 0)).collect();
            let pil = super::super::pil_str::gen_pil_str(l, "Recursion", 19, 0, 5, &params);
            let call = pil.rsplit_once('(').expect("a call").1.split_once(')').expect("terminated").0;
            let emitted: Vec<&str> = call.split(',').filter_map(|a| a.split_once(':').map(|(n, _)| n.trim())).collect();
            let declared = airtemplate_params(&format!("{}.pil", l.pil_file), l.pil_template);
            for arg in emitted {
                assert!(declared.contains(arg), "{}: emits `{arg}:`, not among {declared:?}", l.pil_file);
            }
        }
    }

    /// blake3 is an arity-2 family: its aggregator places the one-row SelectValueArity2 and the
    /// radix-4 TreeSelector (17 signals, one row of the 18-wide band; radix 8 needs two).
    #[test]
    fn blake3_is_an_arity_2_family_so_its_selectval_is_arity_2() {
        use proofman_common::hash_family::{has_forced_tree_geometry, merkle_tree_arity};
        assert_eq!(merkle_tree_arity("blake3"), 2);
        assert!(has_forced_tree_geometry("blake3"));
        let src = std::fs::read_to_string(pil_dir().join("blake3/aggregator.pil")).unwrap();
        assert!(src.contains("selectValueArity2(") && !src.contains("selectValueArity4("));
        assert!(src.contains("treeselector4(") && !src.contains("treeselector8("));
    }
}
