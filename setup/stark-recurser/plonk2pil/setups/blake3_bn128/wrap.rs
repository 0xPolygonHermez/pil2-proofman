//! The final SNARK wrap's setup of a blake3 proving key, over BN128 (`pil/blake3_bn128/wrap.pil`).
//!
//! The rows, in order: a block of [`BLAKE3_WRAP_BLOCK_ROWS`] rows per BLAKE3 compression, in
//! [`Blake3Blocks`]' order (Node, then chunk and parent by flags); the band rows past them that its
//! rows 8..55 do not hold; the public rows. A block's cells are its use's signals:
//!
//! | kind   | rows 0..7 (round 0, row t)             | row 0 too              | row 56 + k                                   |
//! |--------|----------------------------------------|------------------------|----------------------------------------------|
//! | Node   | a0 = in[t], a1 = in[(t+4)%8], a2 = key |                        | a0 = out[k], k < 4                           |
//! | chunk  | a0 = in[8+t]                           | a3..a10 = cv, a11, a12 | a0, a1 = out[2k], out[2k+1]; a2, a3 = cv[2k-8], cv[2k-7] for k >= 4 |
//! | parent | a0 = in[2t], a1 = in[2t+1]             | a11 = blockLen, a12    | a0, a1 = out[2k], out[2k+1]                  |
//!
//! Each block is a gate band of the exec ([`GateBandKind::Blake3Bn128WrapNode`] and the others, its
//! flags the payload), whose other columns the wrap's witness rebuilds. The band: range-check rows,
//! three `Num2Bytes` uses of the same number of chunks (weights in `C`), each a gate band
//! ([`GateBandKind::Blake3Bn128WrapRangeCheck`]) for the witness to count; then PLONK rows, six
//! constraints of one coefficient set. Every selector is a fixed column of this setup.

use std::collections::{BTreeMap, HashMap};

use anyhow::{bail, ensure, Result};
use proofman_common::exec_format::{
    BLAKE3_IV, BLAKE3_SIGMA, BLAKE3_WRAP_RANGE_CHECK_CELLS, BLAKE3_WRAP_BLOCK_ROWS, BLAKE3_WRAP_RANGE_CHECK_SLOTS,
    RANGE_CHECK_CHUNK_BITS,
};
use proofman_common::hash_family::{lookup_gate, GateRole, BLAKE3_BN128_WRAP_FAMILY};
use proofman_fields::{Bn128, Field, PrimeField, QuotientMap};

use super::{gen_pil_str, PilTemplateParams};
use crate::plonk2pil::merge_copies::{apply_remap_to_s_map, r1cs2plonk_merged, verify_merge_soundness};
use crate::plonk2pil::r1cs::to_plonk::{ckey, get_custom_gates_info, PlonkConstraint};
use crate::plonk2pil::r1cs::types::{CustomGateUse, FixedPol, GateBand, GateBandKind, PlonkOptions, R1csFile, SetupResult};
use crate::plonk2pil::setups::blake3::aggregation::{plan_plonk_rows, Blake3Blocks};
use crate::plonk2pil::setups::blake3::{compress_signal, BAND_COLS};
use crate::plonk2pil::utils::{bind_public_signals, build_fixed_pols, build_s_polynomials, public_rows};

/// PLONK gates a row: gate `g` on `a[3g..3g+2]`.
const PLONK_GATES_PER_ROW: usize = 6;

/// The widest `Num2Bytes` a use holds: 5 chunks of 16 bits.
pub const MAX_RANGE_CHECK_BITS: usize = (BLAKE3_WRAP_RANGE_CHECK_CELLS - 1) * RANGE_CHECK_CHUNK_BITS as usize;

/// The rows of a block the band has: those of the G steps after round 0.
const BAND_IN_BLOCK: std::ops::Range<usize> = 8..56;

/// `set_max_constraint_degree` of the PIL: the adds' checks reach 4, and the std groups its buses'
/// terms to it.
pub const MAX_CONSTRAINT_DEGREE: usize = 5;

/// The `--extra-muls` its pilfflonk setup takes: the fewest a 2^24 ptau holds (3 needs 2^25 + 31
/// powers). Measured on a blake3 key: 4 is 272,371 gas, 6 284,979, 8 299,364, at the same proof time.
pub const EXTRA_MULS: u64 = 4;

const AIRGROUP_NAME: &str = "Wrap";

const _: () = assert!(BLAKE3_WRAP_RANGE_CHECK_SLOTS * BLAKE3_WRAP_RANGE_CHECK_CELLS <= BAND_COLS);

/// The fixed columns of the AIR this setup computes, by name.
struct Fixed {
    n: usize,
    pols: BTreeMap<(&'static str, usize), Vec<Bn128>>,
}

impl Fixed {
    fn set(&mut self, name: &'static str, index: usize, row: usize, value: Bn128) {
        let n = self.n;
        self.pols.entry((name, index)).or_insert_with(|| vec![Bn128::ZERO; n])[row] = value;
    }

    fn one(&mut self, name: &'static str, index: usize, row: usize) {
        self.set(name, index, row, Bn128::ONE);
    }

    /// Every column of `names`, all-zero ones included, as the PIL declares them.
    fn into_pols(mut self, airgroup: &str, names: &[(&'static str, usize)]) -> Vec<FixedPol<Bn128>> {
        let n = self.n;
        names
            .iter()
            .flat_map(|&(name, count)| (0..count).map(move |i| (name, i)))
            .map(|(name, index)| FixedPol {
                name: format!("{airgroup}.{name}"),
                index,
                values: self.pols.remove(&(name, index)).unwrap_or_else(|| vec![Bn128::ZERO; n]),
            })
            .collect()
    }
}

/// The wrap's AIR for `r1cs` (see the module), or an error naming what it cannot place: a custom
/// gate other than blake3's and `Num2Bytes` of up to [`MAX_RANGE_CHECK_BITS`] bits, or more rows than
/// BN128 has domains for.
pub fn wrap(r1cs: &R1csFile<Bn128>, options: &PlonkOptions) -> Result<SetupResult<Bn128>> {
    let range_checks = range_check_uses(r1cs)?;
    let cgi = get_custom_gates_info(r1cs);
    let (plonk_constraints, plonk_additions, copy_merge) = r1cs2plonk_merged(r1cs, options.merge_copies);
    let blocks = Blake3Blocks::new(r1cs, &cgi, 1).uses();
    let n_blocks = blocks.len();

    let n_range_rows: usize =
        range_checks.values().map(|uses| uses.len().div_ceil(BLAKE3_WRAP_RANGE_CHECK_SLOTS)).sum();
    let n_plonk_rows = plan_plonk_rows(&plonk_constraints, 0, 1, 1, PLONK_GATES_PER_ROW).rows_needed;
    let n_band_rows = n_range_rows + n_plonk_rows;
    let blocks_end = n_blocks * BLAKE3_WRAP_BLOCK_ROWS;
    let band_rows: Vec<usize> = (0..n_blocks)
        .flat_map(|b| BAND_IN_BLOCK.map(move |t| b * BLAKE3_WRAP_BLOCK_ROWS + t))
        .chain(blocks_end..)
        .take(n_band_rows)
        .collect();
    let first_public_row = band_rows.last().map_or(blocks_end, |&r| r + 1).max(blocks_end);
    let n_publics = r1cs.header.n_outputs + r1cs.header.n_pub_inputs;
    let n_public_rows = public_rows(n_publics, BAND_COLS);
    let n_used = first_public_row + n_public_rows;
    // blake3Tables' XOR table is 2^17 rows.
    let n_bits_natural = (n_used.next_power_of_two().trailing_zeros() as usize).max(17);
    let n_bits = n_bits_natural.max(options.min_n_bits.unwrap_or(0));
    ensure!(
        n_bits <= Bn128::TWO_ADICITY,
        "plonk2pil: the wrap needs 2^{n_bits} rows ({n_used} used), and BN128 has domains of at most 2^{}",
        Bn128::TWO_ADICITY
    );
    let n = 1usize << n_bits;
    tracing::info!(
        "Plonk: {} constraints -> {n_plonk_rows} rows ({} ideal); range checks: {} uses -> {n_range_rows} rows; \
         BLAKE3: {n_blocks} blocks of {BLAKE3_WRAP_BLOCK_ROWS} rows; band past them: {} rows",
        plonk_constraints.len(),
        plonk_constraints.len().div_ceil(PLONK_GATES_PER_ROW),
        range_checks.values().map(Vec::len).sum::<usize>(),
        n_band_rows.saturating_sub(n_blocks * BAND_IN_BLOCK.len())
    );
    tracing::info!("NUsed: {n_used}, nBits: {n_bits}, N: {n}");

    let mut s_map: Vec<Vec<u32>> = (0..BAND_COLS).map(|_| vec![0u32; n]).collect();
    let mut fixed = Fixed { n, pols: BTreeMap::new() };
    let mut gate_bands = Vec::with_capacity(n_blocks + n_range_rows);
    let band_kind =
        [GateBandKind::Blake3Bn128WrapNode, GateBandKind::Blake3Bn128WrapChunk, GateBandKind::Blake3Bn128WrapParent];

    // ── The blocks ────────────────────────────────────────────────────────────
    for (b, &(kind, flags, cgu)) in blocks.iter().enumerate() {
        let base = b * BLAKE3_WRAP_BLOCK_ROWS;
        // Wire 0 is "not connected" in the s-maps: a gate cell there would be free.
        let sig = |i: usize| {
            let wire = cgu.signals[i] as u32;
            assert_ne!(wire, 0, "a blake3 gate's signal {i} is the constant wire 0");
            wire
        };
        let (node, chunk) = (kind == 0, kind == 1);
        if node {
            assert_eq!(cgu.signals.len(), 13, "Blake3Node is in[8] + key + out[4]");
        } else {
            assert_eq!(
                cgu.signals.len(),
                compress_signal::COUNT,
                "Blake3Compress is in[16] + blockLen + counterLo + out[16]"
            );
        }
        for t in 0..8 {
            let row = base + t;
            if node {
                s_map[0][row] = sig(t);
                s_map[1][row] = sig((t + 4) % 8);
                s_map[2][row] = sig(8);
            } else if chunk {
                s_map[0][row] = sig(8 + t);
            } else {
                s_map[0][row] = sig(2 * t);
                s_map[1][row] = sig(2 * t + 1);
            }
            fixed.one(["IN_NODE", "IN_CHUNK", "IN_PARENT"][kind as usize], 0, row);
        }
        if chunk {
            for j in 0..8 {
                s_map[3 + j][base] = sig(j);
            }
        }
        if !node {
            s_map[11][base] = sig(16);
            s_map[12][base] = sig(17);
            fixed.one("BLOCKLEN_ROW", 0, base + 2);
        }
        fixed.one(["INIT_NODE", "INIT_CHUNK", "INIT_PARENT"][kind as usize], 0, base);
        fixed.set("FLAGS", 0, base, Bn128::from_int(flags));

        for t in 0..56 {
            let (r, g) = (t / 8, t % 8);
            let row = base + t;
            fixed.one("G", g, row);
            let key = |w: usize| Bn128::from_int((16 * b + BLAKE3_SIGMA[r][w]) as u64);
            fixed.set("MSG_KEY_X", 0, row, key(2 * g));
            fixed.set("MSG_KEY_Y", 0, row, key(2 * g + 1));
            fixed.set("MSG_MUL", 0, row, if r == 0 { Bn128::from_int(6u64) } else { Bn128::NEG_ONE });
        }
        for t in 56..63 {
            fixed.one("HOLD", 0, base + t);
        }
        for k in 0..if node { 4 } else { 8 } {
            let row = base + 56 + k;
            fixed.one("FF", k, row);
            if node {
                s_map[0][row] = sig(9 + k);
                fixed.one("FF_NODE", 0, row);
                continue;
            }
            s_map[0][row] = sig(compress_signal::IN_CELLS + 2 * k);
            s_map[1][row] = sig(compress_signal::IN_CELLS + 2 * k + 1);
            if k >= 4 {
                if chunk {
                    s_map[2][row] = sig(2 * k - 8);
                    s_map[3][row] = sig(2 * k - 7);
                    fixed.one("FF_CV", 0, row);
                } else {
                    for h in 0..2 {
                        fixed.set("FF_IV", h, row, Bn128::from_int(u64::from(BLAKE3_IV[2 * k - 8 + h])));
                    }
                }
            }
        }
        gate_bands.push(GateBand { row: base as u32, kind: band_kind[kind as usize], payload: flags });
    }

    // ── The band: range checks, then PLONK ───────────────────────────────────
    let mut rows = band_rows.iter().copied();
    let chunk_size = Bn128::from_int(1u64 << RANGE_CHECK_CHUNK_BITS);
    for (&n_chunks, uses) in &range_checks {
        for row_uses in uses.chunks(BLAKE3_WRAP_RANGE_CHECK_SLOTS) {
            let row = rows.next().expect("the band rows were counted");
            fixed.one("RANGE_CHECK", 0, row);
            let mut weight = Bn128::ONE;
            for k in 0..n_chunks {
                fixed.set("C", k, row, weight);
                weight *= chunk_size;
            }
            // A row of fewer uses repeats its last: its copies are true by construction.
            for slot in 0..BLAKE3_WRAP_RANGE_CHECK_SLOTS {
                let cgu = row_uses[slot.min(row_uses.len() - 1)];
                for (j, &signal) in cgu.signals.iter().enumerate() {
                    assert_ne!(signal, 0, "a Num2Bytes signal is the constant wire 0");
                    s_map[BLAKE3_WRAP_RANGE_CHECK_CELLS * slot + j][row] = signal as u32;
                }
            }
            gate_bands.push(GateBand {
                row: row as u32,
                kind: GateBandKind::Blake3Bn128WrapRangeCheck,
                payload: n_chunks as u64,
            });
        }
    }
    // One coefficient set a row, as the blake3 recursion airs: a row of fewer than six constraints
    // repeats its last one.
    let mut by_key: HashMap<String, Vec<&PlonkConstraint<Bn128>>> = HashMap::new();
    for c in &plonk_constraints {
        by_key.entry(ckey(c)).or_default().push(c);
    }
    let mut keys: Vec<&String> = by_key.keys().collect();
    keys.sort_unstable();
    for k in keys {
        for group in by_key[k].chunks(PLONK_GATES_PER_ROW) {
            let row = rows.next().expect("the band rows were counted");
            fixed.one("PLONK", 0, row);
            for (j, &q) in group[0].coeffs.iter().enumerate() {
                fixed.set("C", j, row, q);
            }
            for gate in 0..PLONK_GATES_PER_ROW {
                let c = group[gate.min(group.len() - 1)];
                for (w, &wire) in c.wires.iter().enumerate() {
                    s_map[3 * gate + w][row] = wire;
                }
            }
        }
    }
    assert!(rows.next().is_none(), "every band row is placed");

    // ── Publics and S polynomials ─────────────────────────────────────────────
    bind_public_signals(&mut s_map, first_public_row, n_publics, BAND_COLS);
    for k in 0..n_public_rows {
        fixed.one("PUBLICS_ROW", k, first_public_row + k);
    }
    apply_remap_to_s_map(&mut s_map, &copy_merge.remap);
    verify_merge_soundness(&s_map, &copy_merge.merged_reps, BAND_COLS);
    let sv = build_s_polynomials::<Bn128>(BAND_COLS, n, n_bits, n_used, &s_map);

    let airgroup_name = options.airgroup_name.clone().unwrap_or_else(|| AIRGROUP_NAME.to_string());
    let mut fixed_pols = build_fixed_pols::<Bn128>(&airgroup_name, &[], &sv);
    let mut names = vec![
        ("C", 5),
        ("PLONK", 1),
        ("RANGE_CHECK", 1),
        ("G", 8),
        ("HOLD", 1),
        ("INIT_NODE", 1),
        ("INIT_CHUNK", 1),
        ("INIT_PARENT", 1),
        ("IN_NODE", 1),
        ("IN_CHUNK", 1),
        ("IN_PARENT", 1),
        ("BLOCKLEN_ROW", 1),
        ("FLAGS", 1),
        ("MSG_KEY_X", 1),
        ("MSG_KEY_Y", 1),
        ("MSG_MUL", 1),
        ("FF", 8),
        ("FF_NODE", 1),
        ("FF_CV", 1),
        ("FF_IV", 2),
    ];
    if n_public_rows > 0 {
        names.push(("PUBLICS_ROW", n_public_rows));
    }
    fixed_pols.extend(fixed.into_pols(&airgroup_name, &names));

    let pil_str = gen_pil_str(&PilTemplateParams {
        template_file: "blake3_bn128/wrap",
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
        gate_bands,
        plonk_additions,
        airgroup_name: airgroup_name.clone(),
        air_name: airgroup_name,
        band_aux: BLAKE3_WRAP_BLOCK_ROWS as u64,
    })
}

/// The `Num2Bytes` uses of `r1cs` by their number of chunks, in the r1cs's order, or why the wrap
/// cannot place a gate of it: one that is not blake3's nor a `Num2Bytes` of 1 to
/// [`MAX_RANGE_CHECK_BITS`] bits, or a use of other than its `in` and chunks.
fn range_check_uses(r1cs: &R1csFile<Bn128>) -> Result<BTreeMap<usize, Vec<&CustomGateUse>>> {
    let mut chunks_of = HashMap::new();
    for (id, gate) in r1cs.custom_gates.iter().enumerate() {
        let parameter = match gate.parameters.as_slice() {
            [p] => usize::try_from(&p.as_canonical_biguint()).ok(),
            _ => None,
        };
        match (lookup_gate(&gate.template_name), parameter) {
            (Some((GateRole::Blake3Node | GateRole::Blake3Compress, _)), _) => {}
            (Some((GateRole::RangeCheck, _)), Some(n_bits)) if (1..=MAX_RANGE_CHECK_BITS).contains(&n_bits) => {
                chunks_of.insert(id as u32, n_bits.div_ceil(RANGE_CHECK_CHUNK_BITS as usize));
            }
            _ => bail!(
                "plonk2pil: the {BLAKE3_BN128_WRAP_FAMILY} wrap places Blake3Node, Blake3Compress and \
                 Num2Bytes(nBits), 0 < nBits <= {MAX_RANGE_CHECK_BITS}, only, and custom gate {id} is {} with \
                 parameters {:?}",
                gate.template_name,
                gate.parameters.iter().map(|p| p.to_string()).collect::<Vec<_>>()
            ),
        }
    }
    let mut uses: BTreeMap<usize, Vec<&CustomGateUse>> = BTreeMap::new();
    for cgu in &r1cs.custom_gates_uses {
        if let Some(&n_chunks) = chunks_of.get(&cgu.id) {
            ensure!(
                cgu.signals.len() == 1 + n_chunks,
                "plonk2pil: a use of Num2Bytes of {n_chunks} chunks has {} signals",
                cgu.signals.len()
            );
            uses.entry(n_chunks).or_default().push(cgu);
        }
    }
    Ok(uses)
}
