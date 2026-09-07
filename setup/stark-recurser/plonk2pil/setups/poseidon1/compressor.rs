//! Compressor setup.
//! 46 committed pols, 30 S cols, 10 rows/Poseidon1, 3 CMul/row.
//! Chain slot a[30..45], disjoint from the plonk band a[0..29] = 10 gates, so the inner
//! full-round rows piggyback all 10; TreeSelector8's 30 signals fit one row.

use crate::plonk2pil::r1cs::to_plonk::{ckey, filter_fft4_gate_uses, filter_gate_uses, get_custom_gates_info};
use crate::plonk2pil::r1cs::types::{GateBand, GateBandKind, PlonkOptions, R1csFile, SetupResult};
use crate::plonk2pil::utils::{build_fixed_pols, build_s_polynomials, log2, mulp, PlonkBand};
use crate::plonk2pil::merge_copies::{apply_remap_to_s_map, r1cs2plonk_merged, verify_merge_soundness};
use super::{gen_pil_str, PilTemplateParams};
use proofman_common::hash_family::GateRole;
use std::collections::HashMap;

const COMMITTED_POLS: usize = 46;
const N_COLS: usize = 30; // S connection columns
const POSEIDON_ROWS: usize = 10;
const CMUL_PER_ROW: usize = 3;
const POSEIDON_WIDTH: usize = 16;

// Allowed-plonk-gate masks per row type (bit g = gate g on a[3g..3g+2]). These MIRROR the
// gate selectors in poseidon1/compressor.pil — keep the two in sync; `PlonkBand` asserts
// every placement lands in an allowed gate whose cells are still free.
const G_ALL: u16 = 0x3FF; // gates 0..9  — inner Poseidon rows + pure-plonk rows
const G_PR: u16 = 0x33F; // gates 0..5 + 8,9 — PR row (a[18..23] = overflow anchors)
const G_Q1: u16 = 0x3C0; // gates 6..9 — INIT / FINAL rows
const G_EVPOL: u16 = 0x380; // gates 7..9 — EvPol4 rows (gate 6 = resEVPOL a[18..20])
const G_SELVAL: u16 = 0x300; // gates 8,9 — SelectValueArity4 rows (a[0..21] used)

fn rand_hex() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    format!("{:x}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64)
}

// (row, n_used, max_used)
type PR = (usize, usize, usize);

pub fn compressor(r1cs: &R1csFile, options: &PlonkOptions) -> SetupResult {
    let (plonk_constraints, plonk_additions, copy_merge) = r1cs2plonk_merged(r1cs, options.merge_copies);
    tracing::info!("Number of plonk constraints: {}", plonk_constraints.len());

    let mut cgi = get_custom_gates_info(r1cs);
    let n_poseidon1_compression = cgi.n(GateRole::PoseidonCompression);
    let n_poseidon1_sponge = cgi.n(GateRole::PoseidonSponge);
    let n_total_poseidon = n_poseidon1_compression + n_poseidon1_sponge;
    let n_cmul_rows = cgi.n(GateRole::CMul).div_ceil(CMUL_PER_ROW);
    let n_poseidon_rows = n_total_poseidon * POSEIDON_ROWS;
    let n_fft4_rows = cgi.n(GateRole::Fft4);
    let n_ev_pol4_rows = cgi.n(GateRole::EvPol4);
    let n_tree_sel8_rows = cgi.n(GateRole::TreeSelector); // 30 signals = band width -> 1 row/gate
    let n_sel_val_arity4_rows = cgi.n(GateRole::SelectValArity4);

    // Per-gate row tiers for plonk piggyback. Band cells a[0..29] host 10 gates
    // (q0 = gates 0..5 on a[0..17], q1 = gates 6..9 on a[18..29]). The chain slot a[30..45]
    // is off-band, so it never steals a gate — a Poseidon row only loses the gates whose
    // cells it actually writes. Tiers:
    //   R1,R2,PR',R4,R26,R27,R28 (7 rows) — full band free → all 10 gates → ten tier.
    //   PR (1 row)            — q0 gates 0..5 + q1 gates 8,9 (a[18..23]=anchors) → pr tier.
    //   INIT, FINAL (2 rows)  — q1 gates 6..9 (a[0..17] = input+key / output) → four tier.
    //   EvPol4                — q1 gates 7..9 (gate 6 collides with resEVPOL a[18..20]) → three tier.
    //   SelectValueArity4            — q1 gates 8,9 (a[0..21] used) → two tier.
    //   pure-plonk row        — all 10 gates: q0 0..5 + q1 6..9.
    // TreeSelector8 fills the whole band (a[0..29]) and FFT4 needs cv[0..9] for its own
    // parameters, so neither piggybacks; CMul leaves only a[27..29] and is left alone.
    let ten_count = n_total_poseidon * 7; // R1, R2, R3/PR', R4, R26, R27, R28
    let pr_count = n_total_poseidon; // PR row (q0 0..5 + q1 8,9)
    let four_count = n_total_poseidon * 2; // INIT + FINAL rows (q1 gates 6..9)
    let three_count = n_ev_pol4_rows; // EvPol4: q1 gates 7..9
    let two_count = n_sel_val_arity4_rows; // SelectValueArity4: q1 gates 8,9

    cgi.n_plonk_rows = {
        let mut partial: HashMap<String, (usize, usize)> = HashMap::new(); // (n_used, max_used)
        let mut half: Vec<(usize, usize)> = Vec::new();
        let (mut ten, mut pr_row, mut four, mut three, mut two) =
            (ten_count, pr_count, four_count, three_count, two_count);
        let mut rows = 0usize;
        for c in &plonk_constraints {
            let k = ckey(c);
            if let Some(pr) = partial.get_mut(&k) {
                pr.0 += 1;
                if pr.0 == pr.1 {
                    partial.remove(&k);
                }
            } else if !half.is_empty() {
                let mut pr = half.remove(0);
                pr.0 += 1;
                if pr.0 < pr.1 {
                    partial.insert(k, pr);
                }
            } else if ten > 0 {
                ten -= 1;
                partial.insert(k, (1, 6)); // q0 gates 0..5
                half.push((6, 10)); // q1 gates 6..9
            } else if pr_row > 0 {
                pr_row -= 1;
                partial.insert(k, (1, 6)); // q0 gates 0..5
                half.push((8, 10)); // q1 gates 8,9 (6,7 hold the overflow anchors)
            } else if four > 0 {
                four -= 1;
                partial.insert(k, (7, 10)); // open fills gates 6..9; 3 more refine 7,8,9
            } else if three > 0 {
                three -= 1;
                partial.insert(k, (8, 10)); // open fills gates 7..9; 2 more refine 8,9
            } else if two > 0 {
                two -= 1;
                partial.insert(k, (9, 10)); // open fills gates 8,9; 1 more refines 9
            } else {
                partial.insert(k.clone(), (1, 6));
                half.push((6, 10)); // pure-plonk row: full q1 gates 6..9
                rows += 1;
            }
        }
        rows
    };

    let n_used = cgi.n_plonk_rows
        + n_cmul_rows
        + n_poseidon_rows
        + n_fft4_rows
        + n_ev_pol4_rows
        + n_tree_sel8_rows
        + n_sel_val_arity4_rows;

    let n_bits = if n_used <= 1 { 1 } else { log2((n_used - 1) as u32) as usize + 1 };
    // Never below the floor: an air reusing another air's starkSetup has to match its rows. The
    // pre-floor size is kept -- it is what decides whether the circuit itself is too big.
    let n_bits_natural = n_bits;
    let n_bits = n_bits.max(options.min_n_bits.unwrap_or(0));
    let n = 1usize << n_bits;
    let n_publics = r1cs.header.n_outputs + r1cs.header.n_pub_inputs;
    let airgroup_name = options.airgroup_name.clone().unwrap_or_else(|| format!("Compressor{}", rand_hex()));

    let pil_str = gen_pil_str(&PilTemplateParams {
        template_file: "poseidon1/compressor",
        template_name: "Compressor",
        namespace_name: &airgroup_name,
        n_bits,
        n_publics,
        max_constraint_degree: 5,
        n_plonk_rows: cgi.n_plonk_rows,
        n_poseidon1_compression,
        n_poseidon1_sponge,
        n_cmul_rows,
        n_ev_pol4: cgi.n(GateRole::EvPol4),
        n_fft4: cgi.n(GateRole::Fft4),
        n_tree_selector8: cgi.n(GateRole::TreeSelector),
        n_select_val_arity4: cgi.n(GateRole::SelectValArity4),
    });

    tracing::info!("NUsed: {}, nBits: {}, N: {}", n_used, n_bits, n);

    let mut s_map: Vec<Vec<u32>> = (0..COMMITTED_POLS).map(|_| vec![0u32; n]).collect();
    let mut cv: Vec<Vec<u64>> = (0..10).map(|_| vec![0u64; n]).collect();
    let mut band = PlonkBand::new(n);
    // Bands whose interiors the trace expander rebuilds; see GateBand.
    let mut gate_bands: Vec<GateBand> = Vec::new();

    // Extra-constraint row queues. Plonk band a[0..29] = 10 gates (q0 0..5, q1 6..9).
    //   ten_extra   : Poseidon rows R1,R2,PR',R4,R26,R27,R28 — all 10 gates.
    //   pr_extra    : PR row — q0 gates 0..5 + q1 gates 8,9 (a[18..23] = overflow anchors).
    //   four_extra  : INIT + FINAL rows — q1 gates 6..9 (a[0..17] = input+key / output).
    //   three_extra : EvPol4 rows — q1 gates 7..9 (gate 6 collides with resEVPOL).
    //   two_extra   : SelectValueArity4 rows — q1 gates 8,9.
    let mut ten_extra: Vec<usize> = Vec::new();
    let mut pr_extra: Vec<usize> = Vec::new();
    let mut four_extra: Vec<usize> = Vec::new();
    let mut three_extra: Vec<usize> = Vec::new();
    let mut two_extra: Vec<usize> = Vec::new();

    // CustPoseidon1 (compression) gates come first, then Poseidon1 (sponge) gates,
    // matching the fixed-col patterns in compressor.pil.
    let cust_poseidon1_uses = if n_poseidon1_compression > 0 {
        filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::PoseidonCompression))
    } else {
        Vec::new()
    };
    let poseidon1_uses = if n_poseidon1_sponge > 0 {
        filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::PoseidonSponge))
    } else {
        Vec::new()
    };
    let cmul_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::CMul));
    let fft4_uses = filter_fft4_gate_uses(&r1cs.custom_gates_uses, &cgi.fft4_parameters);
    let ev_pol4_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::EvPol4));
    let tree_sel8_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::TreeSelector));
    let sel_val1_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::SelectValArity4));

    let mut r = 0usize;

    // ── Poseidon1 — 10 rows per gate (compression then sponge) ───────────────
    // CustPoseidon1_16 (compression) signal layout: in[16] + key[2] + out[16] = 34.
    // Poseidon1_16   (sponge)      signal layout: in[16]         + out[16] = 32.
    //
    // The gate publishes only its boundary. The round snapshots filling rows 0..9 of the chain
    // slot and the anchor row are recomputed from it by the trace expander, so each application
    // below records a band instead of mapping those cells. See gate_bands.hpp.
    //
    // R4 (= circom im[4], round-3 P-matrix output) IS stored at row 4 (a normal round
    // writing to the next-row chain slot), so the PIL no longer needs a dedicated preMatP
    // to recompute it — it reads the stored R4 directly.
    //
    // Witness layout per gate — chain slot = a[30..45] (see the row map in compressor.pil):
    //   row 0 INIT input@a[0..15], key@a[16..17]; rows 0..4 chain = R0,R1,R2,R3,R4;
    //   row 5 PR = anchors[0..15]@chain + anchors[16..21]@a[18..23]; rows 6..8 = R26,R27,R28;
    //   row 9 FINAL output@a[0..15], chain = R29.
    let process_poseidon1 = |s: &[u64],
                             is_compression: bool,
                             s_map: &mut [Vec<u32>],
                             cv: &mut [Vec<u64>],
                             ten_extra: &mut Vec<usize>,
                             pr_extra: &mut Vec<usize>,
                             four_extra: &mut Vec<usize>,
                             band: &mut PlonkBand,
                             r: usize| {
        let key_off = if is_compression { 2 } else { 0 };
        let expected = POSEIDON_WIDTH + key_off + POSEIDON_WIDTH;
        assert_eq!(s.len(), expected, "unexpected Poseidon1 signal count");

        let input = &s[0..POSEIDON_WIDTH];
        let key = if is_compression { Some(&s[POSEIDON_WIDTH..POSEIDON_WIDTH + 2]) } else { None };
        let output = &s[POSEIDON_WIDTH + key_off..POSEIDON_WIDTH + key_off + POSEIDON_WIDTH];

        // Boundary only: input at the INIT row, output at FINAL. The chain slot and the
        // anchor row are left unmapped for the expander to fill.
        for i in 0..POSEIDON_WIDTH {
            s_map[i][r] = input[i] as u32;
            s_map[i][r + 9] = output[i] as u32;
        }

        // Key bits at INIT row cols 16..17 (compression only). At INIT plonk gate 5
        // (a[15..17]) doesn't fire (its selector is CHECK_PLONK, which excludes INIT),
        // so a[16..17] are free for the key.
        if let Some(k) = key {
            s_map[16][r] = k[0] as u32;
            s_map[17][r] = k[1] as u32;
        }

        for off in 0..POSEIDON_ROWS {
            for item in cv.iter_mut() {
                item[r + off] = 0;
            }
        }

        // Plonk piggyback queues. The chain is off-band, so R1,R2,PR',R4,R26,R27,R28 leave
        // the whole band free → all 10 gates (ten tier). PR keeps q0 0..5 + q1 8,9 (a[18..23]
        // hold the overflow anchors). INIT and FINAL expose q1 gates 6..9 only.
        four_extra.push(r); // INIT row (a[0..15] input, a[16..17] key)
        ten_extra.push(r + 1); // R1
        ten_extra.push(r + 2); // R2
        ten_extra.push(r + 3); // R3 / PR'
        ten_extra.push(r + 4); // R4 (stored)
        pr_extra.push(r + 5); // PR (a[18..23] anchors)
        ten_extra.push(r + 6); // R26
        ten_extra.push(r + 7); // R27
        ten_extra.push(r + 8); // R28
        four_extra.push(r + 9); // FINAL row (a[0..15] output)

        band.allow(r, G_Q1);
        band.allow(r + 9, G_Q1);
        for off in [1, 2, 3, 4, 6, 7, 8] {
            band.allow(r + off, G_ALL);
        }
        band.allow(r + 5, G_PR);
    };

    tracing::info!("Processing {} CustPoseidon1 (compression) gates...", cust_poseidon1_uses.len());
    for cgu in &cust_poseidon1_uses {
        process_poseidon1(
            &cgu.signals,
            true, // is_compression
            &mut s_map,
            &mut cv,
            &mut ten_extra,
            &mut pr_extra,
            &mut four_extra,
            &mut band,
            r,
        );
        gate_bands.push(GateBand { row: r as u32, kind: GateBandKind::Poseidon1CompressorCompression, payload: 0 });
        r += POSEIDON_ROWS;
    }
    assert_eq!(r, POSEIDON_ROWS * cust_poseidon1_uses.len());

    tracing::info!("Processing {} Poseidon1 (sponge) gates...", poseidon1_uses.len());
    for cgu in &poseidon1_uses {
        process_poseidon1(
            &cgu.signals,
            false, // is_compression
            &mut s_map,
            &mut cv,
            &mut ten_extra,
            &mut pr_extra,
            &mut four_extra,
            &mut band,
            r,
        );
        gate_bands.push(GateBand { row: r as u32, kind: GateBandKind::Poseidon1CompressorSponge, payload: 0 });
        r += POSEIDON_ROWS;
    }
    assert_eq!(r, POSEIDON_ROWS * (cust_poseidon1_uses.len() + poseidon1_uses.len()));

    // ── CMul (3/row) ──────────────────────────────────────────────────────────
    tracing::info!("Processing {} cmul gates...", cmul_uses.len());
    let mut cmul_row: i64 = -1;
    let mut cmul_used = 0usize;
    for cgu in &cmul_uses {
        assert_eq!(cgu.signals.len(), 9);
        if cmul_row >= 0 {
            let row = cmul_row as usize;
            for (i, item) in s_map[9 * cmul_used..].iter_mut().enumerate().take(9) {
                item[row] = cgu.signals[i] as u32;
            }
            cmul_used += 1;
            if cmul_used == CMUL_PER_ROW {
                cmul_row = -1;
                cmul_used = 0;
            }
        } else {
            for (i, item) in s_map.iter_mut().enumerate().take(9) {
                item[r] = cgu.signals[i] as u32;
            }
            for item in cv.iter_mut() {
                item[r] = 0;
            }
            cmul_row = r as i64;
            cmul_used = 1;
            r += 1;
        }
    }
    assert_eq!(r, n_poseidon_rows + n_cmul_rows);

    // ── EvPol4 ────────────────────────────────────────────────────────────────
    tracing::info!("Processing {} evPol4 gates...", ev_pol4_uses.len());
    for cgu in &ev_pol4_uses {
        for (i, item) in s_map.iter_mut().enumerate().take(21) {
            item[r] = cgu.signals[i] as u32;
        }
        for item in cv.iter_mut() {
            item[r] = 0;
        }
        three_extra.push(r);
        band.allow(r, G_EVPOL);
        r += 1;
    }

    // ── FFT4 (1 row) ──────────────────────────────────────────────────────────
    tracing::info!("Processing {} fft4 gates...", fft4_uses.len());
    for cgu in &fft4_uses {
        for (i, item) in s_map.iter_mut().enumerate().take(24) {
            item[r] = cgu.signals[i] as u32;
        }
        let p = cgi.fft4_parameters.get(&cgu.id).expect("FFT4 params");
        let (fft_type, scale, first_w, inc_w) = (p[3], p[2], p[0], p[1]);
        let fw2 = mulp(first_w, first_w);
        if fft_type == 4 {
            cv[0][r] = scale;
            cv[1][r] = mulp(scale, fw2);
            cv[2][r] = mulp(scale, first_w);
            cv[3][r] = mulp(mulp(scale, first_w), fw2);
            cv[4][r] = mulp(mulp(scale, first_w), inc_w);
            cv[5][r] = mulp(mulp(mulp(scale, first_w), fw2), inc_w);
            for item in cv.iter_mut().skip(6) {
                item[r] = 0;
            }
        } else if fft_type == 2 {
            for item in cv.iter_mut().take(6) {
                item[r] = 0;
            }
            cv[6][r] = scale;
            cv[7][r] = mulp(scale, first_w);
            cv[8][r] = mulp(mulp(scale, first_w), inc_w);
            cv[9][r] = 0;
        } else {
            panic!("Invalid FFT4 type: {}", fft_type);
        }
        r += 1;
    }

    // ── TreeSelector8 (1 row) ──────────────────────────────────────────────────
    // Signal layout: values[8][3] (a[0..23]) + keys[3] (a[24..26]) + out[3] (a[27..29]) =
    // 30 signals = exactly the connection-band width N_COLS, so the gate fits a single row
    // and every signal stays inside S. Band is full — no plonk piggyback at TreeSel rows.
    const TREE_SEL8_SIGNALS: usize = N_COLS;
    tracing::info!("Processing {} treeSelector8 gates...", tree_sel8_uses.len());
    for cgu in &tree_sel8_uses {
        assert_eq!(cgu.signals.len(), TREE_SEL8_SIGNALS);
        for (i, item) in s_map.iter_mut().enumerate().take(TREE_SEL8_SIGNALS) {
            item[r] = cgu.signals[i] as u32;
        }
        for item in cv.iter_mut() {
            item[r] = 0;
        }
        r += 1;
    }

    // ── SelectValueArity4 ────────────────────────────────────────────────────────────
    tracing::info!("Processing {} selectVal1 gates...", sel_val1_uses.len());
    for cgu in &sel_val1_uses {
        assert_eq!(cgu.signals.len(), 22);
        for (i, item) in s_map.iter_mut().enumerate().take(22) {
            item[r] = cgu.signals[i] as u32;
        }
        for item in cv.iter_mut() {
            item[r] = 0;
        }
        two_extra.push(r);
        band.allow(r, G_SELVAL);
        r += 1;
    }

    // ── Plonk constraints ─────────────────────────────────────────────────────
    tracing::info!("Placing {} plonk constraints...", plonk_constraints.len());
    let mut partial: HashMap<String, PR> = HashMap::new(); // (row, n_used, max_used)
    let mut half: Vec<PR> = Vec::new();
    let mut pure_plonk_rows: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut plonk_in_pure: usize = 0;
    let mut plonk_in_custom: usize = 0;

    for (idx, c) in plonk_constraints.iter().enumerate() {
        if idx % 10_000 == 0 {
            tracing::debug!("constraint {}/{}", idx, plonk_constraints.len());
        }
        let k = ckey(c);

        if let Some(pr) = partial.get_mut(&k) {
            let n = pr.1;
            let row = pr.0;
            band.put(&mut s_map, row, n, c);
            pr.1 += 1;
            if pr.1 == pr.2 {
                partial.remove(&k);
            }
            if pure_plonk_rows.contains(&row) {
                plonk_in_pure += 1;
            } else {
                plonk_in_custom += 1;
            }
        } else if !half.is_empty() {
            let mut pr = half.remove(0);
            let row = pr.0;
            cv[5][row] = c[3];
            cv[6][row] = c[4];
            cv[7][row] = c[5];
            cv[8][row] = c[6];
            cv[9][row] = c[7];
            for i in pr.1..pr.2 {
                band.put(&mut s_map, row, i, c);
            }
            pr.1 += 1;
            if pr.1 < pr.2 {
                partial.insert(k, pr); // skip reinsert of an exhausted (1-gate) half
            }
            if pure_plonk_rows.contains(&row) {
                plonk_in_pure += 1;
            } else {
                plonk_in_custom += 1;
            }
        } else if !ten_extra.is_empty() {
            let row = ten_extra.remove(0); // inner Poseidon row: all 10 gates
            cv[0][row] = c[3];
            cv[1][row] = c[4];
            cv[2][row] = c[5];
            cv[3][row] = c[6];
            cv[4][row] = c[7];
            for i in 0..6 {
                band.put(&mut s_map, row, i, c);
            }
            partial.insert(k.clone(), (row, 1, 6));
            half.push((row, 6, 10)); // q1 gates 6..9
            plonk_in_custom += 1;
        } else if !pr_extra.is_empty() {
            let row = pr_extra.remove(0); // PR: q0 gates 0..5 (q1 8,9 handed to `half`)
            cv[0][row] = c[3];
            cv[1][row] = c[4];
            cv[2][row] = c[5];
            cv[3][row] = c[6];
            cv[4][row] = c[7];
            for i in 0..6 {
                band.put(&mut s_map, row, i, c);
            }
            partial.insert(k.clone(), (row, 1, 6));
            half.push((row, 8, 10)); // q1 gates 8,9 (6,7 hold the overflow anchors)
            plonk_in_custom += 1;
        } else if !four_extra.is_empty() {
            let row = four_extra.remove(0); // INIT / FINAL: q1 gates 6..9
            cv[5][row] = c[3];
            cv[6][row] = c[4];
            cv[7][row] = c[5];
            cv[8][row] = c[6];
            cv[9][row] = c[7];
            for i in 6..10 {
                band.put(&mut s_map, row, i, c);
            }
            partial.insert(k, (row, 7, 10));
            plonk_in_custom += 1;
        } else if !three_extra.is_empty() {
            let row = three_extra.remove(0); // EvPol4: q1 gates 7..9
            cv[5][row] = c[3];
            cv[6][row] = c[4];
            cv[7][row] = c[5];
            cv[8][row] = c[6];
            cv[9][row] = c[7];
            for i in 7..10 {
                band.put(&mut s_map, row, i, c);
            }
            partial.insert(k, (row, 8, 10));
            plonk_in_custom += 1;
        } else if !two_extra.is_empty() {
            let row = two_extra.remove(0); // SelectValueArity4: q1 gates 8,9
            cv[5][row] = c[3];
            cv[6][row] = c[4];
            cv[7][row] = c[5];
            cv[8][row] = c[6];
            cv[9][row] = c[7];
            for i in 8..10 {
                // gate g -> a[3g..3g+2]; gates 8,9 = a[24..26], a[27..29].
                band.put(&mut s_map, row, i, c);
            }
            partial.insert(k, (row, 9, 10));
            plonk_in_custom += 1;
        } else {
            pure_plonk_rows.insert(r);
            plonk_in_pure += 1;
            band.allow(r, G_ALL);
            cv[0][r] = c[3];
            cv[1][r] = c[4];
            cv[2][r] = c[5];
            cv[3][r] = c[6];
            cv[4][r] = c[7];
            for i in 0..6 {
                band.put(&mut s_map, r, i, c);
            }
            partial.insert(k.clone(), (r, 1, 6));
            half.push((r, 6, 10)); // pure-plonk row: q1 gates 6..9
            r += 1;
        }
    }
    assert_eq!(r, n_used, "row count mismatch: {} != {}", r, n_used);

    tracing::info!(
        "Plonk placement: {} constraints in {} pure plonk rows, {} constraints piggybacked on custom-gate rows",
        plonk_in_pure,
        pure_plonk_rows.len(),
        plonk_in_custom,
    );

    // ── S polynomials ─────────────────────────────────────────────────────────
    // Apply copy-merge remap to every placed cell (incl. custom-gate I/O) so the
    // connection argument ties merged signals — the soundness-critical sweep,
    // then assert each merged equality is actually re-enforced in-band.
    apply_remap_to_s_map(&mut s_map, &copy_merge.remap);
    verify_merge_soundness(&s_map, &copy_merge.merged_reps, N_COLS);
    let sv = build_s_polynomials(N_COLS, n, n_bits, r, &s_map);
    let fixed_pols = build_fixed_pols(&airgroup_name, &cv, &sv);

    SetupResult {
        gate_bands,
        fixed_pols,
        pil_str,
        n_bits,
        n_bits_natural,
        n_used,
        s_map,
        plonk_additions,
        airgroup_name: airgroup_name.clone(),
        air_name: airgroup_name,
        band_aux: 0,
    }
}
