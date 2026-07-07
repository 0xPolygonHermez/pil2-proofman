//! Compressor setup.
//! 40 committed pols, 27 S cols, 10 rows/Poseidon1, 3 CMul/row.
//! Chain slot a[24..39] (overlaps band at a[24..26]); plonk band a[0..23] on Poseidon
//! rows = 8 gates; gate 8 (a[24..26]) only off-Poseidon; TreeSelector8 split over 2 rows.

use crate::plonk2pil::r1cs::to_plonk::{ckey, filter_fft4_gate_uses, filter_gate_uses, get_custom_gates_info};
use crate::plonk2pil::r1cs::types::{PlonkOptions, R1csFile, SetupResult};
use crate::plonk2pil::utils::{build_fixed_pols, build_s_polynomials, log2, mulp};
use crate::plonk2pil::merge_copies::{apply_remap_to_s_map, r1cs2plonk_merged, verify_merge_soundness};
use super::{gen_pil_str, PilTemplateParams};
use proofman_common::hash_family::GateRole;
use std::collections::HashMap;

const COMMITTED_POLS: usize = 40;
const N_COLS: usize = 27; // S connection columns
const POSEIDON_ROWS: usize = 10;
const COL_P: usize = 24; // first Poseidon chain column offset (width-16 slot a[24..39]; overlaps band a[24..26])
const CMUL_PER_ROW: usize = 3;
const POSEIDON_WIDTH: usize = 16;

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
    let n_tree_sel8_rows = 2 * cgi.n(GateRole::TreeSelector); // 2 rows per gate (27/3 split)
    let n_sel_val1_rows = cgi.n(GateRole::SelectVal1);

    // Per-gate row tiers for plonk piggyback. Band cells a[0..26] host 9 gates
    // (q0 = gates 0..5 on a[0..17], q1 = gates 6..8 on a[18..26]) — but the chain slot
    // a[24..39] overlaps gate 8 (a[24..26]) on POSEIDON rows, so Poseidon rows expose
    // only gates 0..7 (8 gates). Gate 8 survives on non-Poseidon rows (pure-plonk / cmul
    // / EvPol4 / SelectVal1). Tiers:
    //   R1,R2,PR',R4,R26,R27,R28 (7 rows) — gates 0..7 → eight tier.
    //   PR (1 row)            — gates 0..5 only (a[18..23]=anchors, a[24..26]=chain) → six tier.
    //   INIT, FINAL (2 rows)  — q1 gates 6,7 (gate 8 is chain here) → two_if tier.
    //   EvPol4                — q1 gates 7,8 (gate 6 collides with resEVPOL a[18..20]) → ev tier.
    //   SelectVal1            — q1 gate 8 → one tier.
    //   pure-plonk row        — full 9 gates (no chain): q0 0..5 + q1 6..8.
    // TreeSelector8 spans 2 rows (a[0..26] + a[0..2]') — no piggyback at TreeSel rows.
    let eight_count = n_total_poseidon * 7; // R1, R2, R3/PR', R4, R26, R27, R28
    let six_count = n_total_poseidon; // PR row (gates 0..5 only)
    let two_if_count = n_total_poseidon * 2; // INIT + FINAL rows (q1 gates 6,7)
    let ev_count = n_ev_pol4_rows; // EvPol4: q1 gates 7,8
    let one_count = n_sel_val1_rows; // SelectVal1: q1 gate 8

    cgi.n_plonk_rows = {
        let mut partial: HashMap<String, (usize, usize)> = HashMap::new(); // (n_used, max_used)
        let mut half: Vec<(usize, usize)> = Vec::new();
        let (mut eight, mut six, mut two_if, mut ev, mut one) =
            (eight_count, six_count, two_if_count, ev_count, one_count);
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
            } else if eight > 0 {
                eight -= 1;
                partial.insert(k, (1, 6)); // q0 gates 0..5
                half.push((6, 8)); // q1 gates 6,7
            } else if six > 0 {
                six -= 1;
                partial.insert(k, (1, 6)); // q0 gates 0..5 only (no q1: 6,7=anchors, 8=chain)
            } else if two_if > 0 {
                two_if -= 1;
                partial.insert(k, (7, 8)); // open fills gates 6,7; 1 more refines 7
            } else if ev > 0 {
                ev -= 1;
                partial.insert(k, (8, 9)); // open fills gates 7,8; 1 more refines 8
            } else if one > 0 {
                one -= 1; // single gate (gate 8); nothing to track after placing.
            } else {
                partial.insert(k.clone(), (1, 6));
                half.push((6, 9)); // pure-plonk row: full q1 gates 6,7,8 (no chain)
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
        + n_sel_val1_rows;

    let n_bits = if n_used <= 1 { 1 } else { log2((n_used - 1) as u32) as usize + 1 };
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
        n_select_val1: cgi.n(GateRole::SelectVal1),
    });

    tracing::info!("NUsed: {}, nBits: {}, N: {}", n_used, n_bits, n);

    let mut s_map: Vec<Vec<u32>> = (0..COMMITTED_POLS).map(|_| vec![0u32; n]).collect();
    let mut cv: Vec<Vec<u64>> = (0..10).map(|_| vec![0u64; n]).collect();

    // Extra-constraint row queues. Plonk band a[0..26] = 9 gates (q0 0..5, q1 6..8).
    //   eight_extra : Poseidon rows R1,R2,PR',R4,R26,R27,R28 — gates 0..7 (gate 8 = chain).
    //   six_extra   : PR row — gates 0..5 only (a[18..23]=anchors, a[24..26]=chain).
    //   two_if_extra: INIT + FINAL rows — q1 gates 6,7 (gate 8 = chain).
    //   ev_extra    : EvPol4 rows — q1 gates 7,8 (gate 6 collides with resEVPOL).
    //   one_extra   : SelectVal1 rows — q1 gate 8 only.
    let mut eight_extra: Vec<usize> = Vec::new();
    let mut six_extra: Vec<usize> = Vec::new();
    let mut two_if_extra: Vec<usize> = Vec::new();
    let mut ev_extra: Vec<usize> = Vec::new();
    let mut one_extra: Vec<usize> = Vec::new();

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
    let sel_val1_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::SelectVal1));

    let mut r = 0usize;

    // ── Poseidon1 — 10 rows per gate (compression then sponge) ───────────────
    // CustPoseidon1_16 (compression) signal layout: in[16] + key[2] + im[11][16] + out[16] = 210.
    // Poseidon1_16   (sponge)      signal layout: in[16]         + im[11][16] + out[16] = 208.
    //
    // R4 (= circom im[4], round-3 P-matrix output) IS stored at row 4 (a normal round
    // writing to the next-row chain slot), so the PIL no longer needs a dedicated preMatP
    // to recompute it — it reads the stored R4 directly.
    //
    // Witness layout per gate — chain slot = a[24..39] (see the row map in compressor.pil):
    //   row 0 INIT input@a[0..15], key@a[16..17]; rows 0..4 chain = R0,R1,R2,R3,R4;
    //   row 5 PR = anchors[0..15]@chain + anchors[16..21]@a[18..23]; rows 6..8 = R26,R27,R28;
    //   row 9 FINAL output@a[0..15], chain = R29.
    let process_poseidon1 = |s: &[u64],
                             is_compression: bool,
                             s_map: &mut [Vec<u32>],
                             cv: &mut [Vec<u64>],
                             eight_extra: &mut Vec<usize>,
                             six_extra: &mut Vec<usize>,
                             two_if_extra: &mut Vec<usize>,
                             r: usize| {
        let key_off = if is_compression { 2 } else { 0 };
        let expected = POSEIDON_WIDTH + key_off + 12 * POSEIDON_WIDTH + POSEIDON_WIDTH;
        assert_eq!(s.len(), expected, "unexpected Poseidon1 signal count");

        let input = &s[0..POSEIDON_WIDTH];
        let key = if is_compression { Some(&s[POSEIDON_WIDTH..POSEIDON_WIDTH + 2]) } else { None };
        let im_base = POSEIDON_WIDTH + key_off;
        let r0 = &s[im_base..im_base + POSEIDON_WIDTH]; // im[0]
        let r1 = &s[im_base + POSEIDON_WIDTH..im_base + 2 * POSEIDON_WIDTH]; // im[1]
        let r2 = &s[im_base + 2 * POSEIDON_WIDTH..im_base + 3 * POSEIDON_WIDTH]; // im[2]
        let r3 = &s[im_base + 3 * POSEIDON_WIDTH..im_base + 4 * POSEIDON_WIDTH]; // im[3]
        let r4 = &s[im_base + 4 * POSEIDON_WIDTH..im_base + 5 * POSEIDON_WIDTH]; // im[4]: R4 (post-P), stored at row 4
        let im1 = &s[im_base + 5 * POSEIDON_WIDTH..im_base + 6 * POSEIDON_WIDTH]; // im[5]: h1 anchors[0..10]
                                                                                  // im[6] = midState (intermediate, not used with single-chain layout)
        let im2 = &s[im_base + 7 * POSEIDON_WIDTH..im_base + 8 * POSEIDON_WIDTH]; // im[7]: h2 anchors[0..10]
        let r26 = &s[im_base + 8 * POSEIDON_WIDTH..im_base + 9 * POSEIDON_WIDTH]; // im[8]
        let r27 = &s[im_base + 9 * POSEIDON_WIDTH..im_base + 10 * POSEIDON_WIDTH]; // im[9]
        let r28 = &s[im_base + 10 * POSEIDON_WIDTH..im_base + 11 * POSEIDON_WIDTH]; // im[10]
        let r29 = &s[im_base + 11 * POSEIDON_WIDTH..im_base + 12 * POSEIDON_WIDTH]; // im[11]
        let output = &s[im_base + 12 * POSEIDON_WIDTH..im_base + 13 * POSEIDON_WIDTH];

        for i in 0..POSEIDON_WIDTH {
            s_map[i][r] = input[i] as u32;
            s_map[i + COL_P][r] = r0[i] as u32; // row 0 chain slot = R0 (= circom im[0], permuted input signal)
            s_map[i + COL_P][r + 1] = r1[i] as u32;
            s_map[i + COL_P][r + 2] = r2[i] as u32;
            s_map[i + COL_P][r + 3] = r3[i] as u32;
            s_map[i + COL_P][r + 4] = r4[i] as u32; // row 4 chain slot = R4 (stored)
                                                    // row 5 chain slot is anchors[0..15] (filled below).
            s_map[i + COL_P][r + 6] = r26[i] as u32;
            s_map[i + COL_P][r + 7] = r27[i] as u32;
            s_map[i + COL_P][r + 8] = r28[i] as u32;
            s_map[i + COL_P][r + 9] = r29[i] as u32;
            s_map[i][r + 9] = output[i] as u32;
        }

        // Partial-chain anchors (single 22-round chain, one flat array):
        //   anchors[0..15]  → row 5 (PR) chain slot a[24..39]
        //   anchors[16..21] → row 5 (PR) cols a[18..23] (overflow, PR plonk band, same row)
        // Source: anchors[0..10] = im1 (h1[0..10]); anchors[11..21] = im2 (h2[0..10]).
        // First 5 of im2 sit at the PR chain-slot tail (a[35..39]); last 6 go to a[18..23]@PR.
        for i in 0..11 {
            s_map[i + COL_P][r + 5] = im1[i] as u32; // anchors[0..10] = im_h1
        }
        for i in 0..5 {
            s_map[i + 11 + COL_P][r + 5] = im2[i] as u32; // anchors[11..15] = im_h2[0..4]
        }
        for i in 0..6 {
            s_map[i + 18][r + 5] = im2[i + 5] as u32; // anchors[16..21] = im_h2[5..10] @ PR row
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

        // Plonk piggyback queues. R1,R2,PR',R4,R26,R27,R28 fire gates 0..7 → eight tier
        // (gate 8 = a[24..26] holds chain). PR fires gates 0..5 only → six tier. INIT and
        // FINAL each pick up q1 gates 6,7 (gate 8 = chain) → two_if tier.
        two_if_extra.push(r); // INIT row (a[18..23] freed by moving anchors to PR)
        eight_extra.push(r + 1); // R1
        eight_extra.push(r + 2); // R2
        eight_extra.push(r + 3); // R3 / PR'
        eight_extra.push(r + 4); // R4 (stored)
        six_extra.push(r + 5); // PR (a[18..23] anchors, a[24..26] chain)
        eight_extra.push(r + 6); // R26
        eight_extra.push(r + 7); // R27
        eight_extra.push(r + 8); // R28
        two_if_extra.push(r + 9); // FINAL row
    };

    tracing::info!("Processing {} CustPoseidon1 (compression) gates...", cust_poseidon1_uses.len());
    for cgu in &cust_poseidon1_uses {
        process_poseidon1(
            &cgu.signals,
            true, // is_compression
            &mut s_map,
            &mut cv,
            &mut eight_extra,
            &mut six_extra,
            &mut two_if_extra,
            r,
        );
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
            &mut eight_extra,
            &mut six_extra,
            &mut two_if_extra,
            r,
        );
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
        ev_extra.push(r);
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

    // ── TreeSelector8 (2 rows) ──────────────────────────────────────────────────
    // TreeSelector8 signal layout: values[8][3] + keys[3] + out[3] = 30 signals.
    // Split at the connection-band width N_COLS so every signal stays inside S: the
    // first N_COLS signals on row r (a[0..N_COLS-1]), the remaining TREE_SEL8_SIGNALS -
    // N_COLS on row r+1 (a[0..]'). No plonk piggyback at TreeSel rows.
    const TREE_SEL8_SIGNALS: usize = 30;
    tracing::info!("Processing {} treeSelector8 gates...", tree_sel8_uses.len());
    for cgu in &tree_sel8_uses {
        assert_eq!(cgu.signals.len(), TREE_SEL8_SIGNALS);
        for (i, item) in s_map.iter_mut().enumerate().take(N_COLS) {
            item[r] = cgu.signals[i] as u32;
        }
        for (i, item) in s_map.iter_mut().enumerate().take(TREE_SEL8_SIGNALS - N_COLS) {
            item[r + 1] = cgu.signals[N_COLS + i] as u32;
        }
        for item in cv.iter_mut() {
            item[r] = 0;
            item[r + 1] = 0;
        }
        r += 2;
    }

    // ── SelectVal1 ────────────────────────────────────────────────────────────
    tracing::info!("Processing {} selectVal1 gates...", sel_val1_uses.len());
    for cgu in &sel_val1_uses {
        assert_eq!(cgu.signals.len(), 22);
        for (i, item) in s_map.iter_mut().enumerate().take(22) {
            item[r] = cgu.signals[i] as u32;
        }
        for item in cv.iter_mut() {
            item[r] = 0;
        }
        one_extra.push(r);
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
            s_map[n * 3][row] = c[0] as u32;
            s_map[n * 3 + 1][row] = c[1] as u32;
            s_map[n * 3 + 2][row] = c[2] as u32;
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
                s_map[3 * i][row] = c[0] as u32;
                s_map[3 * i + 1][row] = c[1] as u32;
                s_map[3 * i + 2][row] = c[2] as u32;
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
        } else if !eight_extra.is_empty() {
            let row = eight_extra.remove(0); // Poseidon row: gates 0..7 (gate 8 = chain)
            cv[0][row] = c[3];
            cv[1][row] = c[4];
            cv[2][row] = c[5];
            cv[3][row] = c[6];
            cv[4][row] = c[7];
            for i in 0..6 {
                s_map[3 * i][row] = c[0] as u32;
                s_map[3 * i + 1][row] = c[1] as u32;
                s_map[3 * i + 2][row] = c[2] as u32;
            }
            partial.insert(k.clone(), (row, 1, 6));
            half.push((row, 6, 8)); // q1 gates 6,7 (gate 8 is chain)
            plonk_in_custom += 1;
        } else if !six_extra.is_empty() {
            let row = six_extra.remove(0); // PR: q0 gates 0..5 only
            cv[0][row] = c[3];
            cv[1][row] = c[4];
            cv[2][row] = c[5];
            cv[3][row] = c[6];
            cv[4][row] = c[7];
            for i in 0..6 {
                s_map[3 * i][row] = c[0] as u32;
                s_map[3 * i + 1][row] = c[1] as u32;
                s_map[3 * i + 2][row] = c[2] as u32;
            }
            partial.insert(k.clone(), (row, 1, 6));
            // no q1 half: gates 6,7 hold anchors, gate 8 holds chain.
            plonk_in_custom += 1;
        } else if !two_if_extra.is_empty() {
            let row = two_if_extra.remove(0); // INIT / FINAL: q1 gates 6,7 (gate 8 = chain)
            cv[5][row] = c[3];
            cv[6][row] = c[4];
            cv[7][row] = c[5];
            cv[8][row] = c[6];
            cv[9][row] = c[7];
            for i in 6..8 {
                s_map[3 * i][row] = c[0] as u32;
                s_map[3 * i + 1][row] = c[1] as u32;
                s_map[3 * i + 2][row] = c[2] as u32;
            }
            partial.insert(k, (row, 7, 8));
            plonk_in_custom += 1;
        } else if !ev_extra.is_empty() {
            let row = ev_extra.remove(0); // EvPol4: q1 gates 7,8
            cv[5][row] = c[3];
            cv[6][row] = c[4];
            cv[7][row] = c[5];
            cv[8][row] = c[6];
            cv[9][row] = c[7];
            for i in 7..9 {
                s_map[3 * i][row] = c[0] as u32;
                s_map[3 * i + 1][row] = c[1] as u32;
                s_map[3 * i + 2][row] = c[2] as u32;
            }
            partial.insert(k, (row, 8, 9));
            plonk_in_custom += 1;
        } else if !one_extra.is_empty() {
            let row = one_extra.remove(0); // SelectVal1: q1 gate 8 only
            cv[5][row] = c[3];
            cv[6][row] = c[4];
            cv[7][row] = c[5];
            cv[8][row] = c[6];
            cv[9][row] = c[7];
            for i in 8..9 {
                // gate 8 = a[24..26], derived from the gate g -> a[3g..3g+2] convention.
                s_map[3 * i][row] = c[0] as u32;
                s_map[3 * i + 1][row] = c[1] as u32;
                s_map[3 * i + 2][row] = c[2] as u32;
            }
            // single gate — nothing left to track.
            plonk_in_custom += 1;
        } else {
            pure_plonk_rows.insert(r);
            plonk_in_pure += 1;
            cv[0][r] = c[3];
            cv[1][r] = c[4];
            cv[2][r] = c[5];
            cv[3][r] = c[6];
            cv[4][r] = c[7];
            for i in 0..6 {
                s_map[3 * i][r] = c[0] as u32;
                s_map[3 * i + 1][r] = c[1] as u32;
                s_map[3 * i + 2][r] = c[2] as u32;
            }
            partial.insert(k.clone(), (r, 1, 6));
            half.push((r, 6, 9)); // q1 gates 6,7,8
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
        fixed_pols,
        pil_str,
        n_bits,
        n_used,
        s_map,
        plonk_additions,
        airgroup_name: airgroup_name.clone(),
        air_name: airgroup_name,
    }
}
