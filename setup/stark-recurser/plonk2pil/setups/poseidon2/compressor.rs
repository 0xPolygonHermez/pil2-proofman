//! Compressor setup.
//! 46 committed pols, 30 S cols, 10 rows/Poseidon, 3 CMul/row.
//! Chain slot a[30..45], disjoint from the plonk band a[0..29] = 10 gates, so the inner
//! full-round rows piggyback all 10; overflow anchors on the PR row.

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
const POSEIDON_WIDTH: usize = 16;
const CMUL_PER_ROW: usize = 3;

// Allowed-plonk-gate masks per row type (bit g = gate g on a[3g..3g+2]). These MIRROR the
// gate selectors in poseidon2/compressor.pil — keep the two in sync; `PlonkBand` asserts
// every placement lands in an allowed gate whose cells are still free.
const G_ALL: u16 = 0x3FF; // gates 0..9  — inner Poseidon rows + pure-plonk rows
const G_PR: u16 = 0x33F; // gates 0..5 + 8,9 — PR row (a[18..23] = overflow anchors)
const G_Q1: u16 = 0x3C0; // gates 6..9 — INIT / FINAL / TreeSelector4 rows
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
    let n_poseidon_sponge = cgi.n(GateRole::PoseidonSponge);
    let n_poseidon_compression = cgi.n(GateRole::PoseidonCompression);
    let n_poseidon = n_poseidon_sponge + n_poseidon_compression;
    let n_cmul_rows = cgi.n(GateRole::CMul).div_ceil(CMUL_PER_ROW);
    let n_poseidon_rows = n_poseidon * POSEIDON_ROWS;
    let n_fft4_rows = cgi.n(GateRole::Fft4);
    let n_ev_pol4_rows = cgi.n(GateRole::EvPol4);
    let n_tree_sel4_rows = cgi.n(GateRole::TreeSelector);
    let n_sel_val_arity4_rows = cgi.n(GateRole::SelectValArity4);

    // Row-count tiers (used to pre-compute n_plonk_rows). Plonk band a[0..29] = 10 gates:
    // q0 = gates 0..5 (a[0..17]), q1 = gates 6..9 (a[18..29]). The chain slot a[30..45] is
    // off-band, so it never steals a gate — a row only loses the gates it actually writes.
    //   R1,R2,R3,R4,R26,R27,R28 (7 rows)      — full band free → all 10 gates → ten tier.
    //   PR (1 row)                            — q0 0..5 + q1 8,9 (a[18..23]=anchors) → pr tier.
    //   INIT(R0), FINAL, TreeSelector4        — q1 gates 6..9 → four tier.
    //   EvPol4                                — q1 gates 7..9 → three tier.
    //   SelectValueArity4                            — q1 gates 8,9 → two tier.
    // FFT4 needs cv[0..9] for its own parameters, so it never piggybacks; CMul leaves only
    // a[27..29] free and is left alone.
    let ten_count = n_poseidon * 7;
    let pr_count = n_poseidon;
    let four_count = n_poseidon * 2 + n_tree_sel4_rows;
    let three_count = n_ev_pol4_rows;
    let two_count = n_sel_val_arity4_rows;

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
                half.push((6, 10)); // pure-plonk row: q1 gates 6..9
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
        + n_tree_sel4_rows
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
        template_file: "poseidon2/compressor",
        template_name: "Compressor",
        namespace_name: &airgroup_name,
        n_bits,
        n_publics,
        max_constraint_degree: 5,
        n_plonk_rows: cgi.n_plonk_rows,
        n_poseidon_compressor: n_poseidon_compression,
        n_poseidon_sponge,
        n_cmul_rows,
        n_ev_pol4: cgi.n(GateRole::EvPol4),
        n_fft4: cgi.n(GateRole::Fft4),
        n_tree_selector4: cgi.n(GateRole::TreeSelector),
        n_select_val_arity4: cgi.n(GateRole::SelectValArity4),
    });

    tracing::info!("NUsed: {}, nBits: {}, N: {}", n_used, n_bits, n);

    let mut s_map: Vec<Vec<u32>> = (0..COMMITTED_POLS).map(|_| vec![0u32; n]).collect();
    let mut cv: Vec<Vec<u64>> = (0..10).map(|_| vec![0u64; n]).collect();
    let mut band = PlonkBand::new(n);
    // Bands whose interiors the trace expander rebuilds; see GateBand.
    let mut gate_bands: Vec<GateBand> = Vec::new();

    // Extra-constraint row queues. Band a[0..29] = 10 gates (q0 0..5, q1 6..9).
    //   ten_extra   : R1,R2,R3,R4,R26,R27,R28 — all 10 gates.
    //   pr_extra    : PR row — q0 gates 0..5 + q1 gates 8,9 (a[18..23] = overflow anchors).
    //   four_extra  : INIT(R0), FINAL, TreeSelector4 — q1 gates 6..9.
    //   three_extra : EvPol4 — q1 gates 7..9 (gate 6 collides with resEVPOL).
    //   two_extra   : SelectValueArity4 — q1 gates 8,9.
    let mut ten_extra: Vec<usize> = Vec::new();
    let mut pr_extra: Vec<usize> = Vec::new();
    let mut four_extra: Vec<usize> = Vec::new();
    let mut three_extra: Vec<usize> = Vec::new();
    let mut two_extra: Vec<usize> = Vec::new();

    let poseidon_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::PoseidonSponge));
    let poseidon_cust_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::PoseidonCompression));
    let cmul_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::CMul));
    let fft4_uses = filter_fft4_gate_uses(&r1cs.custom_gates_uses, &cgi.fft4_parameters);
    let ev_pol4_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::EvPol4));
    let tree_sel4_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::TreeSelector));
    let sel_val1_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::SelectValArity4));

    let mut r = 0usize;

    // ── Poseidon custom / compressor (10 rows) ────────────────────────────────
    tracing::info!("Processing {} poseidon custom gates...", poseidon_cust_uses.len());
    for cgu in &poseidon_cust_uses {
        assert_eq!(cgu.signals.len(), 2 * POSEIDON_WIDTH + 2, "unexpected Poseidon2 compression signal count");
        let s = &cgu.signals;
        let input = &s[0..POSEIDON_WIDTH];
        let output = &s[POSEIDON_WIDTH + 2..2 * POSEIDON_WIDTH + 2];

        // Boundary only; the chain slots and anchor row belong to the expander. See gate_bands.hpp.
        for i in 0..POSEIDON_WIDTH {
            s_map[i][r] = input[i] as u32;
            s_map[i][r + 9] = output[i] as u32;
        }
        // Key bits at a[16]/a[17] of the band's first row; the expander reads them back there.
        s_map[16][r] = s[POSEIDON_WIDTH] as u32;
        s_map[17][r] = s[POSEIDON_WIDTH + 1] as u32;

        for off in 0..10 {
            for item in cv.iter_mut() {
                item[r + off] = 0;
            }
        }
        four_extra.push(r); // INIT (R0)
        for off in 1..=4 {
            ten_extra.push(r + off); // R1, R2, R3, R4
        }
        pr_extra.push(r + 5); // PR
        for off in 6..=8 {
            ten_extra.push(r + off); // R26, R27, R28
        }
        four_extra.push(r + 9); // FINAL

        band.allow(r, G_Q1);
        band.allow(r + 9, G_Q1);
        for off in [1, 2, 3, 4, 6, 7, 8] {
            band.allow(r + off, G_ALL);
        }
        band.allow(r + 5, G_PR);
        gate_bands.push(GateBand { row: r as u32, kind: GateBandKind::Poseidon2CompressorCompression, payload: 0 });
        r += 10;
    }
    assert_eq!(r, 10 * poseidon_cust_uses.len());

    // ── Poseidon sponge (10 rows) ─────────────────────────────────────────────
    tracing::info!("Processing {} poseidon gates...", poseidon_uses.len());
    for cgu in &poseidon_uses {
        assert_eq!(cgu.signals.len(), 2 * POSEIDON_WIDTH, "unexpected Poseidon2 sponge signal count");
        let s = &cgu.signals;
        let input = &s[0..POSEIDON_WIDTH];
        let output = &s[POSEIDON_WIDTH..2 * POSEIDON_WIDTH];

        // Boundary only; the chain slots and anchor row belong to the expander. See gate_bands.hpp.
        for i in 0..16 {
            s_map[i][r] = input[i] as u32;
            s_map[i][r + 9] = output[i] as u32;
        }

        for off in 0..10 {
            for item in cv.iter_mut() {
                item[r + off] = 0;
            }
        }
        four_extra.push(r); // INIT (R0): a[0..15] input, a[16..17] key
        for off in 1..=4 {
            ten_extra.push(r + off); // R1, R2, R3, R4
        }
        pr_extra.push(r + 5); // PR (a[18..23] hold overflow anchors)
        for off in 6..=8 {
            ten_extra.push(r + off); // R26, R27, R28
        }
        four_extra.push(r + 9); // FINAL

        band.allow(r, G_Q1);
        band.allow(r + 9, G_Q1);
        for off in [1, 2, 3, 4, 6, 7, 8] {
            band.allow(r + off, G_ALL);
        }
        band.allow(r + 5, G_PR);
        gate_bands.push(GateBand { row: r as u32, kind: GateBandKind::Poseidon2CompressorSponge, payload: 0 });
        r += 10;
    }
    assert_eq!(r, 10 * poseidon_cust_uses.len() + 10 * poseidon_uses.len());

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

    // ── TreeSelector4 ─────────────────────────────────────────────────────────
    tracing::info!("Processing {} treeSelector4 gates...", tree_sel4_uses.len());
    for cgu in &tree_sel4_uses {
        assert_eq!(cgu.signals.len(), 17);
        for (i, item) in s_map.iter_mut().enumerate().take(17) {
            item[r] = cgu.signals[i] as u32;
        }
        for item in cv.iter_mut() {
            item[r] = 0;
        }
        four_extra.push(r);
        band.allow(r, G_Q1);
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

    for (idx, c) in plonk_constraints.iter().enumerate() {
        if idx % 10_000 == 0 {
            tracing::debug!("constraint {}/{}", idx, plonk_constraints.len());
        }
        let k = ckey(c);

        if let Some(pr) = partial.get_mut(&k) {
            let n = pr.1;
            band.put(&mut s_map, pr.0, n, c);
            pr.1 += 1;
            if pr.1 == pr.2 {
                partial.remove(&k);
            }
        } else if !half.is_empty() {
            let mut pr = half.remove(0);
            cv[5][pr.0] = c[3];
            cv[6][pr.0] = c[4];
            cv[7][pr.0] = c[5];
            cv[8][pr.0] = c[6];
            cv[9][pr.0] = c[7];
            for i in pr.1..pr.2 {
                band.put(&mut s_map, pr.0, i, c);
            }
            pr.1 += 1;
            if pr.1 < pr.2 {
                partial.insert(k, pr); // skip reinsert of an exhausted (1-gate) half
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
        } else if !four_extra.is_empty() {
            let row = four_extra.remove(0); // INIT / FINAL / TreeSelector4: q1 gates 6..9
            cv[5][row] = c[3];
            cv[6][row] = c[4];
            cv[7][row] = c[5];
            cv[8][row] = c[6];
            cv[9][row] = c[7];
            for i in 6..10 {
                band.put(&mut s_map, row, i, c);
            }
            partial.insert(k, (row, 7, 10));
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
        } else {
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
