//! Custom gates: where each PIL template binds a circom custom gate's signals (the trusted table),
//! and the check that every r1cs custom-gate use sits on cells of its signals' classes, at a row
//! whose anchor selector fires, with the gate's parameters in the fixed columns it reads them from.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use super::fixed::{wrap, Classes, Columns};
use super::pil::{fixed_id, Atom, PilAir};
use super::poly::ONE;
use super::r1cs_plonk::{Vars, FREE};
use super::{CheckError, Template};
use crate::plonk2pil::consts::{lift, ROUNDS, ROUND_CONSTANTS, WIDTH};
use crate::plonk2pil::field::PlonkField;
use crate::plonk2pil::r1cs::types::{CustomGateUse, R1csFile};
use pil_info::core::ir::ConstraintSystem;

/// Signal `sig` of the gate is `a[col]` at `base + off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub sig: usize,
    pub col: u32,
    pub off: i64,
}

/// Parameters a gate reads from fixed columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Param {
    None,
    /// `constFFT[i]` is `C[c]` at `base + off`.
    Fft4(Vec<(u32, i64)>),
    /// `flags` (Node: 11) in `FLAGS` at these offsets.
    Flags(Vec<i64>),
    /// `PoseidonT(t)` of this width.
    Width(u64),
    /// `Num2Bytes(nBits)`: weights `2^(16k)` then 0 in `col[k]`, k < 5, at `base`.
    Weights(&'static str),
}

/// One gate of one template: which circom gate (`Blake3Compress` split by `isParent`).
#[derive(Debug, Clone)]
pub struct Gate {
    pub name: &'static str,
    pub n_signals: usize,
    pub anchor: &'static str,
    /// Placements; a use matches one. Entries past a Num2Bytes use's signal count are skipped.
    pub shapes: Vec<Vec<Cell>>,
    /// Rows a band occupies (no two bands overlap), and its selectors' exact pattern.
    pub band_rows: usize,
    pub band: Vec<(&'static str, Vec<i64>)>,
    /// Rows shared with other bands (the blake3 wrap packs range checks inside its blocks).
    pub shares_rows: bool,
    pub param: Param,
    pub fill: Fill,
}

/// Constants a band writes, re-derived here: `K` of PoseidonT, the blake3 wrap's message keys,
/// multiplicities and parent IV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fill {
    None,
    PoseidonK,
    Blake3Wrap,
}

fn ident(n: usize) -> Vec<Cell> {
    (0..n).map(|i| Cell { sig: i, col: i as u32, off: 0 }).collect()
}

fn plain(name: &'static str, n: usize, anchor: &'static str) -> Gate {
    Gate {
        name,
        n_signals: n,
        anchor,
        shapes: vec![ident(n)],
        band_rows: 0,
        band: vec![],
        shares_rows: false,
        param: Param::None,
        fill: Fill::None,
    }
}

/// `per_row` CMuls of 9 cells side by side.
fn cmul(per_row: u32) -> Gate {
    let shapes =
        (0..per_row).map(|s| (0..9).map(|j| Cell { sig: j, col: 9 * s + j as u32, off: 0 }).collect()).collect();
    Gate { shapes, ..plain("CMul", 9, "CMUL") }
}

fn fft4_one_row() -> Gate {
    Gate { param: Param::Fft4((0..9).map(|i| (i, 0)).collect()), ..plain("FFT4", 24, "FFT4") }
}

/// A Poseidon band of `rows` rows: in at row 0, key (compression) at `key`, out at the last row.
fn poseidon(
    name: &'static str,
    anchor: &'static str,
    rows: usize,
    key: Option<[(u32, i64); 2]>,
    band: Vec<(&'static str, Vec<i64>)>,
) -> Gate {
    let last = rows as i64 - 1;
    let mut c: Vec<Cell> = (0..16).map(|i| Cell { sig: i, col: i as u32, off: 0 }).collect();
    let out = if let Some(k) = key {
        c.extend(k.iter().enumerate().map(|(j, &(col, off))| Cell { sig: 16 + j, col, off }));
        18
    } else {
        16
    };
    c.extend((0..16).map(|i| Cell { sig: out + i, col: i as u32, off: last }));
    let mut band = band;
    band.insert(0, (anchor, vec![0]));
    Gate { shapes: vec![c], band_rows: rows, band, n_signals: out + 16, ..plain(name, 0, anchor) }
}

/// A range-check row: `RANGE_CHECK` is the lookup's selector, so it is exactly 1 on located rows, 0 elsewhere.
/// `shares`: it may sit inside another band (only the blake3 wrap packs range checks in its blocks).
fn num2bytes(n: usize, shares: bool) -> Gate {
    Gate {
        band_rows: 1,
        band: vec![("RANGE_CHECK", vec![0])],
        shares_rows: shares,
        ..plain("Num2Bytes", n, "RANGE_CHECK")
    }
}

const CLOCKS: i64 = 56;

/// The gates of `t`'s template.
pub fn gates_of(t: &Template) -> Result<Vec<Gate>, CheckError> {
    let lanes = || {
        t.params
            .get("LANES")
            .copied()
            .ok_or_else(|| CheckError::Artefact("pilout: blake3 template without LANES".into()))
    };
    Ok(match t.file.as_str() {
        "poseidon2/compressor" | "poseidon1/compressor" | "poseidon2/aggregator" | "poseidon1/aggregator" => {
            let (p1, comp) = (t.file.starts_with("poseidon1"), t.file.ends_with("compressor"));
            let (cn, sn, comp_sel, sponge_sel, fin) = if p1 {
                ("CustPoseidon1_16", "Poseidon1_16", "POSEIDON1_COMPRESSION", "POSEIDON1_SPONGE", "POSEIDON1_FINAL")
            } else {
                ("CustPoseidon2_16", "Poseidon2_16", "POSEIDONCOMPRESSION", "POSEIDONSPONGE", "POSEIDON_FINAL")
            };
            // compressor: 10-row hash, key on the gate row, 3 CMuls a row; aggregator: 5 rows, key at a[15]+1/+2
            let (rows, key, n_cmul) = if comp { (10, [(16, 0), (17, 0)], 3) } else { (5, [(15, 1), (15, 2)], 2) };
            let mut tail = vec![(fin, vec![rows as i64 - 1])];
            if !p1 {
                tail.insert(0, ("POSEIDON_PARTIAL_ROUND", vec![rows as i64 / 2]));
            }
            let tree = match (p1, comp) {
                (false, _) => plain("TreeSelector4", 17, "TREESELECTOR4"),
                (true, true) => plain("TreeSelector8", 30, "TREESELECTOR8"),
                // vals on the gate row, key and result on the next
                (true, false) => {
                    let mut c = ident(24);
                    c.extend((0..6).map(|i| Cell { sig: 24 + i, col: i as u32, off: 1 }));
                    Gate { shapes: vec![c], ..plain("TreeSelector8", 30, "TREESELECTOR8") }
                }
            };
            vec![
                poseidon(cn, comp_sel, rows, Some(key), tail.clone()),
                poseidon(sn, sponge_sel, rows, None, tail),
                cmul(n_cmul),
                Gate { n_signals: 27, ..plain("EvPol4", 21, "EVPOL4") },
                fft4_one_row(),
                tree,
                plain("SelectValueArity4", 22, "SELECTVAL_ARITY4"),
            ]
        }
        "blake3/aggregator" => {
            let l = lanes()? as i64;
            let lane_shapes = |n_in: usize, n_out: usize| -> Vec<Vec<Cell>> {
                (0..l)
                    .map(|lane| {
                        let mut c: Vec<Cell> = (0..n_in).map(|j| Cell { sig: j, col: j as u32, off: lane }).collect();
                        c.extend((0..n_out).map(|j| Cell { sig: n_in + j, col: j as u32, off: CLOCKS - l + lane }));
                        c
                    })
                    .collect()
            };
            let block = |name, n_in, n_out, kind: &'static str| Gate {
                name,
                n_signals: n_in + n_out,
                anchor: "CLK_0",
                shapes: lane_shapes(n_in, n_out),
                band_rows: CLOCKS as usize,
                band: vec![("CLK_0", vec![0]), (kind, (0..CLOCKS).collect())],
                shares_rows: false,
                param: Param::Flags((0..CLOCKS).collect()),
                fill: Fill::None,
            };
            let ev: Vec<Cell> = (0..27).map(|j| Cell { sig: j, col: (j % 18) as u32, off: (j / 18) as i64 }).collect();
            let mut fft: Vec<Cell> = (0..12).map(|j| Cell { sig: j, col: j as u32, off: 0 }).collect();
            fft.extend((0..12).map(|j| Cell { sig: 12 + j, col: j as u32, off: 1 }));
            let fft_c = (0..9).map(|i| if i < 5 { (i, 0) } else { (i - 5, 1) }).collect();
            vec![
                block("Blake3Node", 9, 4, "BLAKE3_NODE"),
                block("Blake3Compress/chunk", 18, 16, "BLAKE3_CHUNK"),
                block("Blake3Compress/parent", 18, 16, "BLAKE3_PARENT"),
                cmul(2),
                Gate { shapes: vec![ev], ..plain("EvPol4", 27, "EVPOL4") },
                Gate { shapes: vec![fft], param: Param::Fft4(fft_c), ..plain("FFT4", 24, "FFT4") },
                plain("TreeSelector4", 17, "TREESELECTOR4"),
                plain("SelectValueArity2", 13, "SELECTVAL_ARITY2"),
            ]
        }
        "poseidon_bn128/wrap" => {
            const W: usize = WIDTH;
            const ROWS: usize = ROUNDS + 1;
            let c = (0..W * ROWS).map(|i| Cell { sig: i, col: (i % W) as u32, off: (i / W) as i64 }).collect();
            let full: Vec<i64> = (0..68).filter(|r| !(4..64).contains(r)).collect();
            let rc = (0..1 + W).map(|i| Cell { sig: i, col: i as u32, off: 0 }).collect();
            vec![
                Gate {
                    name: "PoseidonT",
                    n_signals: W * ROWS,
                    anchor: "POSEIDON",
                    shapes: vec![c],
                    band_rows: ROWS,
                    band: vec![
                        ("POSEIDON", (0..ROWS as i64).collect()),
                        ("POSEIDON_FULL_ROUND", full),
                        ("POSEIDON_PARTIAL_ROUND", (4..64).collect()),
                    ],
                    shares_rows: false,
                    param: Param::Width(W as u64),
                    fill: Fill::PoseidonK,
                },
                Gate { shapes: vec![rc], param: Param::Weights("K"), ..num2bytes(1 + W, false) },
            ]
        }
        "blake3_bn128/wrap" => {
            let at = |sig, col, off| Cell { sig, col, off };
            let mut node = Vec::new();
            for t in 0..8usize {
                node.extend([at(t, 0, t as i64), at((t + 4) % 8, 1, t as i64), at(8, 2, t as i64)]);
            }
            node.extend((0..4).map(|k| at(9 + k, 0, 56 + k as i64)));
            let (mut chunk, mut parent) = (Vec::new(), Vec::new());
            for t in 0..8usize {
                chunk.push(at(8 + t, 0, t as i64));
                parent.extend([at(2 * t, 0, t as i64), at(2 * t + 1, 1, t as i64)]);
            }
            chunk.extend((0..8).map(|j| at(j, 3 + j as u32, 0)));
            for c in [&mut chunk, &mut parent] {
                c.extend([at(16, 11, 0), at(17, 12, 0)]);
                for k in 0..8usize {
                    c.extend([at(18 + 2 * k, 0, 56 + k as i64), at(19 + 2 * k, 1, 56 + k as i64)]);
                }
            }
            chunk.extend((4..8usize).flat_map(|k| [at(2 * k - 8, 2, 56 + k as i64), at(2 * k - 7, 3, 56 + k as i64)]));
            const FF: [&str; 8] = ["FF[0]", "FF[1]", "FF[2]", "FF[3]", "FF[4]", "FF[5]", "FF[6]", "FF[7]"];
            const G: [&str; 8] = ["G[0]", "G[1]", "G[2]", "G[3]", "G[4]", "G[5]", "G[6]", "G[7]"];
            let block = |name,
                         n,
                         init: &'static str,
                         input: &'static str,
                         shape,
                         outs: usize,
                         extra: Vec<(&'static str, Vec<i64>)>| {
                let mut band = vec![(init, vec![0]), (input, (0..8).collect()), ("HOLD", (56..63).collect())];
                band.extend((0..8).map(|g| (G[g], (0..56).filter(|t| t % 8 == g as i64).collect())));
                band.extend((0..8).map(|k| (FF[k], if k < outs { vec![56 + k as i64] } else { vec![] })));
                band.extend(extra);
                Gate {
                    name,
                    n_signals: n,
                    anchor: init,
                    shapes: vec![shape],
                    band_rows: 64,
                    band,
                    shares_rows: false,
                    param: Param::Flags(vec![0]),
                    fill: Fill::Blake3Wrap,
                }
            };
            let slots =
                (0..3u32).map(|s| (0..6).map(|j| Cell { sig: j, col: 6 * s + j as u32, off: 0 }).collect()).collect();
            vec![
                block("Blake3Node", 13, "INIT_NODE", "IN_NODE", node, 4, vec![("FF_NODE", (56..60).collect())]),
                block(
                    "Blake3Compress/chunk",
                    34,
                    "INIT_CHUNK",
                    "IN_CHUNK",
                    chunk,
                    8,
                    vec![("BLOCKLEN_ROW", vec![2]), ("FF_CV", (60..64).collect())],
                ),
                block(
                    "Blake3Compress/parent",
                    34,
                    "INIT_PARENT",
                    "IN_PARENT",
                    parent,
                    8,
                    vec![("BLOCKLEN_ROW", vec![2])],
                ),
                Gate { shapes: slots, param: Param::Weights("C"), ..num2bytes(6, true) },
            ]
        }
        other => return Err(CheckError::Artefact(format!("no gate table for template {other}"))),
    })
}

fn small<F: PlonkField>(v: u64) -> F {
    super::pil::fe::<F>(&num_bigint::BigUint::from(v))
}

/// The fixed id of `NAME` or `NAME[i]`.
fn col_id(cs: &ConstraintSystem, name: &str) -> Option<u32> {
    match name.strip_suffix(']').and_then(|n| n.split_once('[')) {
        Some((base, i)) => fixed_id(cs, base, Some(i.parse().ok()?)),
        None => fixed_id(cs, name, None),
    }
}

/// `constFFT` of `FFT4(firstW, incW, scale, type)`, re-derived from fft.pil's use of it.
fn fft4_consts<F: PlonkField>(p: &[F]) -> Option<[F; 9]> {
    let [w, inc, scale, ty] = p[..] else { return None };
    let w2 = w * w;
    let mut c = [F::ZERO; 9];
    if ty == small::<F>(4) {
        c[..6].copy_from_slice(&[scale, scale * w2, scale * w, scale * w * w2, scale * w * inc, scale * w * w2 * inc]);
    } else if ty == small::<F>(2) {
        c[6..].copy_from_slice(&[scale, scale * w, scale * w * inc]);
    } else {
        return None;
    }
    Some(c)
}

/// The spec key of a use: its template name, `Blake3Compress` split by `isParent`.
fn key_of<F: PlonkField>(r1cs: &R1csFile<F>, u: &CustomGateUse) -> Result<String, CheckError> {
    let g = r1cs
        .custom_gates
        .get(u.id as usize)
        .ok_or_else(|| CheckError::Artefact(format!("exec: custom gate {} undefined", u.id)))?;
    Ok(match g.template_name.as_str() {
        "Blake3Compress" => match g.parameters.get(1) {
            Some(p) if p.is_one() => "Blake3Compress/parent".into(),
            Some(p) if p.is_zero() => "Blake3Compress/chunk".into(),
            _ => return Err(CheckError::GateParam { gate: u.id, why: "isParent is not a bit".into() }),
        },
        n => n.to_string(),
    })
}

/// What the gate check saw.
#[derive(Debug, Clone, Default)]
pub struct TierGates {
    pub uses: usize,
    pub bands: usize,
}

/// Locates every custom-gate use, checks its parameters, and the bands' selector patterns.
#[allow(clippy::too_many_arguments)]
pub fn check<F: PlonkField>(
    r1cs: &R1csFile<F>,
    cs: &ConstraintSystem,
    air: &PilAir<F>,
    t: &Template,
    a_cols: &[u32],
    vars: &Vars,
    cls: &Classes,
    cols: &mut Columns<F>,
) -> Result<TierGates, CheckError> {
    let specs: HashMap<&str, Gate> = gates_of(t)?.into_iter().map(|g| (g.name, g)).collect();
    let n = cls.n;
    let id = |name: &str, idx: Option<u32>| {
        fixed_id(cs, name, idx).ok_or_else(|| CheckError::Artefact(format!("pilout: no fixed column {name}{idx:?}")))
    };
    // tier C: each used gate's cells are read by a constraint that reads its anchor
    let mut with: HashMap<u32, HashSet<u32>> = HashMap::new();
    for (_, s) in &air.other {
        let wits: Vec<u32> = s.iter().filter_map(|a| if let Atom::Wit(w, _) = a { Some(*w) } else { None }).collect();
        for a in s.iter() {
            if let Atom::Fixed(f, _) = a {
                with.entry(*f).or_default().extend(&wits);
            }
        }
    }
    let mut rep = TierGates::default();
    let mut bases: BTreeMap<&str, BTreeSet<usize>> = BTreeMap::new();
    let mut referenced: HashSet<&str> = HashSet::new();
    let mut unbound: HashSet<u32> = HashSet::new();
    for (ui, u) in r1cs.custom_gates_uses.iter().enumerate() {
        let key = key_of(r1cs, u)?;
        let g = specs
            .get(key.as_str())
            .ok_or_else(|| CheckError::GateParam { gate: u.id, why: format!("{key} has no place in {}", t.file) })?;
        // every signal bound or free; Num2Bytes's count follows nBits, checked against its chunks below
        let fits = match g.param {
            Param::Weights(_) => u.signals.len() <= g.n_signals,
            _ => u.signals.len() == g.n_signals,
        };
        if !fits {
            return Err(CheckError::GateParam { gate: u.id, why: format!("{key} has {} signals", u.signals.len()) });
        }
        let anchor = id(g.anchor, None)?;
        if referenced.insert(g.name) {
            let sels: Vec<u32> =
                std::iter::once(g.anchor).chain(g.band.iter().map(|b| b.0)).filter_map(|c| col_id(cs, c)).collect();
            for c in g.shapes.iter().flatten() {
                let w = *a_cols
                    .get(c.col as usize)
                    .ok_or_else(|| CheckError::Artefact(format!("pilout: no a[{}]", c.col)))?;
                if !sels.iter().any(|f| with.get(f).is_some_and(|s| s.contains(&w))) {
                    return Err(CheckError::Artefact(format!(
                        "{key}: no constraint reads both one of its selectors and a[{}]",
                        c.col
                    )));
                }
            }
        }
        let params = &r1cs.custom_gates[u.id as usize].parameters;
        let sv: Vec<u32> = u
            .signals
            .iter()
            .map(|&s| if s < r1cs.header.n_vars as u64 { vars.of(s as u32) } else { u32::MAX })
            .collect();
        let free = |v: u32| v != ONE && v & FREE != 0;
        let mut found = None;
        'shapes: for shape in &g.shapes {
            let shape: Vec<&Cell> = shape.iter().filter(|c| c.sig < u.signals.len()).collect();
            // a signal the template does not bind must be in no other relation, so free to extract
            let bound: HashSet<usize> = shape.iter().map(|c| c.sig).collect();
            if let Some(k) = (0..sv.len()).find(|&k| {
                sv[k] == u32::MAX || (bound.contains(&k) && free(sv[k])) || (!bound.contains(&k) && !free(sv[k]))
            }) {
                return Err(CheckError::GateNotFound {
                    gate: ui as u32,
                    name: key,
                    why: format!(
                        "signal {k} (wire {}): bound {} but its wire is {}",
                        u.signals[k],
                        bound.contains(&k),
                        if free(sv[k]) { "in no class" } else { "in a class" }
                    ),
                });
            }
            let Some(pivot) = shape.iter().min_by_key(|c| cls.cells_of(sv[c.sig]).len()) else { continue };
            for cell in cls.cells_of(sv[pivot.sig]) {
                let (w, row) = (cell / n as u32, cell as usize % n);
                if w != a_cols[pivot.col as usize] {
                    continue;
                }
                let base = wrap(row, -pivot.off, n);
                if shape.iter().all(|c| cls.class(a_cols[c.col as usize], wrap(base, c.off, n)) == sv[c.sig])
                    && !cols.get(anchor)?[base].is_zero()
                {
                    found = Some((base, bound));
                    break 'shapes;
                }
            }
        }
        let (base, bound) = found.ok_or_else(|| CheckError::GateNotFound {
            gate: ui as u32,
            name: key.clone(),
            why: "no placement of its signals' classes".into(),
        })?;
        // an unbound signal is the gate's own output: no two uses may share it
        if let Some(k) = (0..sv.len()).find(|k| !bound.contains(k) && !unbound.insert(sv[*k])) {
            return Err(CheckError::GateParam { gate: ui as u32, why: format!("{key}: unbound signal {k} is shared") });
        }
        let bad = |why: String| CheckError::GateParam { gate: ui as u32, why: format!("{key} at row {base}: {why}") };
        match &g.param {
            Param::None => {}
            Param::Fft4(at) => {
                let c = fft4_consts(params).ok_or_else(|| bad("FFT4 parameters".into()))?;
                for (i, &(ci, off)) in at.iter().enumerate() {
                    if cols.get(id("C", Some(ci))?)?[wrap(base, off, n)] != c[i] {
                        return Err(bad(format!("constFFT[{i}] is not C[{ci}]")));
                    }
                }
            }
            Param::Flags(rows) => {
                let flags = if g.name == "Blake3Node" {
                    small::<F>(11)
                } else {
                    *params.first().ok_or_else(|| bad("no flags".into()))?
                };
                let col = id("FLAGS", None)?;
                if let Some(r) =
                    rows.iter().find(|&&o| cols.get(col).map(|c| c[wrap(base, o, n)] != flags).unwrap_or(true))
                {
                    return Err(bad(format!("FLAGS at +{r} is not {flags}")));
                }
            }
            Param::Width(w) => {
                if params.first().copied() != Some(small::<F>(*w)) {
                    return Err(bad(format!("width is not {w}")));
                }
            }
            Param::Weights(name) => {
                let bits = params.first().ok_or_else(|| bad("no nBits".into()))?.as_canonical_biguint();
                let chunks = u64::try_from(bits).map_err(|_| bad("nBits".into()))?.div_ceil(16) as usize;
                if u.signals.len() != 1 + chunks {
                    return Err(bad(format!("{} signals for {chunks} chunks", u.signals.len())));
                }
                let mut wgt = F::ONE;
                for k in 0..5u32 {
                    let want = if (k as usize) < chunks { wgt } else { F::ZERO };
                    if cols.get(id(name, Some(k))?)?[base] != want {
                        return Err(bad(format!("weight {name}[{k}]")));
                    }
                    wgt *= small::<F>(1 << 16);
                }
            }
        }
        rep.uses += 1;
        if g.band_rows > 0 {
            bases.entry(g.name).or_default().insert(base);
        }
    }
    // bands: disjoint (rows wrap), each band selector exactly the union of the located bands' patterns
    let mut taken = vec![false; n];
    let mut expect: BTreeMap<&str, BTreeSet<usize>> = BTreeMap::new();
    for g in specs.values() {
        for (c, _) in &g.band {
            expect.entry(c).or_default();
        }
    }
    for (name, bs) in &bases {
        let g = &specs[name];
        for &b in bs {
            let mut rows = (0..g.band_rows).map(|k| (b + k) % n);
            if let Some(r) = rows.find(|&r| !g.shares_rows && std::mem::replace(&mut taken[r], true)) {
                return Err(CheckError::FixedPattern {
                    column: "band".into(),
                    row: r,
                    why: "two bands overlap".into(),
                });
            }
            for (c, offs) in &g.band {
                expect.get_mut(c).expect("listed").extend(offs.iter().map(|&o| wrap(b, o, n)));
            }
            check_fill(cs, g, b, cols)?;
        }
        rep.bands += bs.len();
    }
    let mut cut = |c: &str, want: &dyn Fn(usize) -> F| -> Result<(), CheckError> {
        let Some(fid) = col_id(cs, c) else { return Err(CheckError::Artefact(format!("pilout: no fixed column {c}"))) };
        let col = cols.get(fid)?;
        match (0..n).find(|&r| col[r] != want(r)) {
            Some(r) => Err(CheckError::FixedPattern {
                column: c.to_string(),
                row: r,
                why: format!("{} where the bands want {}", col[r], want(r)),
            }),
            None => Ok(()),
        }
    };
    for (c, rows) in &expect {
        if col_id(cs, c).is_some() || !rows.is_empty() {
            cut(c, &|r| if rows.contains(&r) { F::ONE } else { F::ZERO })?;
        }
    }
    // the wrap's range table: RANGE[i] = i mod 2^16
    if t.file == "poseidon_bn128/wrap" && col_id(cs, "RANGE").is_some() {
        cut("RANGE", &|r| small::<F>(r as u64 % (1 << 16)))?;
    }
    // the blake3 wrap's message multiplicities are 0 off its blocks
    if t.file == "blake3_bn128/wrap" {
        let mut mul = vec![F::ZERO; n];
        for (name, bs) in &bases {
            for &b in bs.iter().filter(|_| specs[name].fill == Fill::Blake3Wrap) {
                for (k, m) in mul_of::<F>().into_iter().enumerate() {
                    mul[(b + k) % n] = m;
                }
            }
        }
        cut("MSG_MUL", &|r| mul[r])?;
    }
    Ok(rep)
}

/// BLAKE3's message schedule `σ_r` (`σ_{r+1}[i] = σ_r[P[i]]`) and IV, independent of the generator.
pub fn sigma() -> [[usize; 16]; 7] {
    const P: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
    let mut s = [[0; 16]; 7];
    s[0] = core::array::from_fn(|i| i);
    for r in 1..7 {
        s[r] = core::array::from_fn(|i| s[r - 1][P[i]]);
    }
    s
}
const IV: [u64; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

/// `MSG_MUL` on a block's rows: 6 on round 0, −1 on rounds 1..6, 0 on the feedforward.
pub fn mul_of<F: PlonkField>() -> Vec<F> {
    (0..64)
        .map(|t| {
            if t < 8 {
                small(6)
            } else if t < 56 {
                -F::ONE
            } else {
                F::ZERO
            }
        })
        .collect()
}

/// What a band's [`Fill`] writes: per column, its values on the band's rows from `b`.
pub fn fill_values<F: PlonkField>(g: &Gate, b: usize) -> Result<Vec<(String, Vec<F>)>, CheckError> {
    let mut want: Vec<(String, Vec<F>)> = Vec::new();
    match g.fill {
        Fill::None => {}
        Fill::PoseidonK => {
            let rc: Vec<F> = ROUND_CONSTANTS
                .iter()
                .map(lift)
                .collect::<Option<_>>()
                .ok_or_else(|| CheckError::Artefact("PoseidonT's constants are not in this field".into()))?;
            for j in 0..WIDTH {
                let k = (0..g.band_rows).map(|k| if k < ROUNDS { rc[WIDTH * k + j] } else { F::ZERO });
                want.push((format!("K[{j}]"), k.collect()));
            }
        }
        Fill::Blake3Wrap => {
            // keys 16·block + σ_r[word]: blocks are 64-row, so b / 64 is distinct per located block
            let (sig, block) = (sigma(), (b / 64) as u64);
            for (x, name) in ["MSG_KEY_X", "MSG_KEY_Y"].iter().enumerate() {
                let key = |t: usize| small::<F>(16 * block + sig[t / 8][2 * (t % 8) + x] as u64);
                want.push((name.to_string(), (0..64).map(|t| if t < 56 { key(t) } else { F::ZERO }).collect()));
            }
            for h in 0..2 {
                let iv = |t: usize| {
                    let parent = g.name == "Blake3Compress/parent" && t >= 60;
                    if parent {
                        small(IV[2 * (t - 56) - 8 + h])
                    } else {
                        F::ZERO
                    }
                };
                want.push((format!("FF_IV[{h}]"), (0..64).map(iv).collect()));
            }
        }
    }
    Ok(want)
}

/// The columns a band's [`Fill`] writes hold its constants on every row of the band at `b`.
fn check_fill<F: PlonkField>(
    cs: &ConstraintSystem,
    g: &Gate,
    b: usize,
    cols: &mut Columns<F>,
) -> Result<(), CheckError> {
    let n = cols.n();
    for (c, vals) in fill_values::<F>(g, b)? {
        let fid = col_id(cs, &c).ok_or_else(|| CheckError::Artefact(format!("pilout: no fixed column {c}")))?;
        let col = cols.get(fid)?;
        if let Some(k) = (0..vals.len()).find(|&k| col[(b + k) % n] != vals[k]) {
            return Err(CheckError::FixedPattern {
                column: c,
                row: (b + k) % n,
                why: format!("{} where {} at band row {k} wants {}", col[(b + k) % n], g.name, vals[k]),
            });
        }
    }
    Ok(())
}
