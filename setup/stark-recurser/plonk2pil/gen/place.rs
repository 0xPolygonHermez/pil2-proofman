//! The one placer: walks an [`AirLayout`]'s sections, then packs the plonk constraints by its
//! policy. Without a [`Sheet`] it is a dry run that only counts rows, so sizing and placement
//! cannot drift apart.

use std::collections::{BTreeMap, HashMap, VecDeque};

use proofman_common::hash_family::GateRole;
use proofman_fields::{Field, QuotientMap};

use super::blocks::{HashBlocks, RowAlloc, BLAKE3_CLOCKS};
use super::finish::Prepared;
use crate::plonk2pil::consts::{is_full_round, lift, ROUNDS, ROUND_CONSTANTS, WIDTH};
use super::spec::{AirLayout, Blocks, ConstFill, GateGroup, Gates, Init, PlonkPolicy, RowKind, Section};
use crate::plonk2pil::field::PlonkField;
use crate::plonk2pil::r1cs::to_plonk::{ckey, filter_fft4_gate_uses, filter_gate_uses, PlonkConstraint};
use crate::plonk2pil::r1cs::types::{CustomGateUse, GateBand};
use crate::plonk2pil::utils::fft4_constants;
use proofman_common::exec_format::{BLAKE3_WRAP_BLOCK_ROWS, RANGE_CHECK_CHUNK_BITS};

/// The blake3 wrap's band rows of a block: those of the G steps after round 0.
const WRAP_BAND_IN_BLOCK: std::ops::Range<usize> = 8..56;

/// The plonk gates each row offers, mirroring the PIL selectors: a placement on a gate whose
/// selector does not fire there, or on cells a custom gate owns, panics naming the row.
pub struct PlonkBand {
    allowed: Vec<u16>,
    written: Vec<u16>,
}

impl PlonkBand {
    pub fn new(n: usize) -> Self {
        Self { allowed: vec![0u16; n], written: vec![0u16; n] }
    }

    pub fn allow(&mut self, row: usize, mask: u16) {
        self.allowed[row] = mask;
    }

    /// Places `c` into gate `g` of `row`; a later write to the same gate refines it.
    pub fn put<F>(&mut self, s_map: &mut [Vec<u32>], row: usize, g: usize, c: &PlonkConstraint<F>) {
        assert!(
            self.allowed[row] & (1 << g) != 0,
            "plonk gate {g} placed at row {row}, but its PIL selector does not fire there \
             (allowed mask {:#012b}) — the constraint would be unenforced",
            self.allowed[row]
        );
        if self.written[row] & (1 << g) == 0 {
            for j in 0..3 {
                assert_eq!(
                    s_map[3 * g + j][row],
                    0,
                    "plonk gate {g} at row {row} overlaps a custom-gate cell a[{}]",
                    3 * g + j
                );
            }
            self.written[row] |= 1 << g;
        }
        for (j, &w) in c.wires.iter().enumerate() {
            s_map[3 * g + j][row] = w;
        }
    }
}

/// What the placement writes: the map, the coefficient and named fixed columns, the gate bands.
pub struct Sheet<F> {
    pub s_map: Vec<Vec<u32>>,
    pub cv: Vec<Vec<F>>,
    named: BTreeMap<(&'static str, usize), Vec<F>>,
    pub band: PlonkBand,
    pub gate_bands: Vec<GateBand>,
    /// `PoseidonT`'s round constants in `F`, once a band needs them.
    rc: Vec<F>,
    n: usize,
}

impl<F: Field> Sheet<F> {
    pub fn new(layout: &AirLayout, n: usize) -> Self {
        Self {
            s_map: vec![vec![0; n]; layout.committed],
            cv: vec![vec![F::ZERO; n]; layout.c_cols],
            named: BTreeMap::new(),
            band: PlonkBand::new(n),
            gate_bands: Vec::new(),
            rc: Vec::new(),
            n,
        }
    }

    /// Named column `name[index]` (`C` is the coefficients).
    pub fn col(&mut self, name: &'static str, index: usize) -> &mut [F] {
        if name == "C" {
            return &mut self.cv[index];
        }
        let n = self.n;
        self.named.entry((name, index)).or_insert_with(|| vec![F::ZERO; n])
    }

    pub fn set(&mut self, name: &'static str, index: usize, row: usize, value: F) {
        self.col(name, index)[row] = value;
    }

    /// Takes `name[index]` out, or its initial value if nothing set it.
    pub fn take(&mut self, name: &'static str, index: usize, init: Init) -> Vec<F>
    where
        F: QuotientMap<u64>,
    {
        if name == "C" {
            return std::mem::take(&mut self.cv[index]);
        }
        self.named.remove(&(name, index)).unwrap_or_else(|| match init {
            Init::Zero => vec![F::ZERO; self.n],
            Init::RangeTable => (0..self.n as u64).map(|i| F::from_int(i % (1 << RANGE_CHECK_CHUNK_BITS))).collect(),
        })
    }

    fn coeffs(&mut self, row: usize, coef: usize, c: &PlonkConstraint<F>) {
        for (j, &q) in c.coeffs.iter().enumerate() {
            self.cv[coef + j][row] = q;
        }
    }
}

/// Where the next row comes from.
enum Rows {
    Seq(usize),
    Interiors { alloc: RowAlloc, hash_blocks: usize },
    WrapBand { blocks: usize, taken: usize, last: Option<usize> },
}

impl Rows {
    fn open(&mut self, rows: usize, step: usize) {
        if let Rows::Interiors { alloc, .. } = self {
            alloc.open(rows, step);
        }
    }

    fn take(&mut self, step: usize) -> usize {
        match self {
            Rows::Seq(r) => {
                *r += step;
                *r - step
            }
            Rows::Interiors { alloc, .. } => alloc.take(),
            Rows::WrapBand { blocks, taken, last } => {
                let k = *taken;
                let per = WRAP_BAND_IN_BLOCK.len();
                let row = if k < *blocks * per {
                    (k / per) * BLAKE3_WRAP_BLOCK_ROWS + WRAP_BAND_IN_BLOCK.start + k % per
                } else {
                    *blocks * BLAKE3_WRAP_BLOCK_ROWS + (k - *blocks * per)
                };
                *taken += 1;
                *last = Some(row);
                row
            }
        }
    }

    /// The first row past everything placed: where the publics go.
    fn end(&self) -> usize {
        match self {
            Rows::Seq(r) => *r,
            Rows::Interiors { alloc, hash_blocks } => alloc.blocks_used().max(*hash_blocks) * BLAKE3_CLOCKS,
            Rows::WrapBand { blocks, last, .. } => {
                let end = *blocks * BLAKE3_WRAP_BLOCK_ROWS;
                last.map_or(end, |r| r + 1).max(end)
            }
        }
    }
}

/// What a placement found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placed {
    pub first_public_row: usize,
    /// Rows the plonk constraints took on their own (`nPlonkRows`).
    pub plonk_rows: usize,
    /// Blocks the band runs took (blake3).
    pub band_blocks: usize,
    /// Custom-gate uses placed.
    pub uses: usize,
}

/// The uses `gates` selects, in the r1cs's order.
pub fn select<'a, F: PlonkField>(prep: &Prepared<'a, F>, gates: Gates) -> Vec<&'a CustomGateUse> {
    let (uses, cgi) = (&prep.r1cs.custom_gates_uses, &prep.cgi);
    match gates {
        Gates::Role(GateRole::Fft4) => filter_fft4_gate_uses(uses, &cgi.fft4_parameters),
        Gates::Role(GateRole::RangeCheck) => uses.iter().filter(|u| cgi.range_check_bits.contains_key(&u.id)).collect(),
        Gates::Role(GateRole::PoseidonT) => uses.iter().filter(|u| cgi.poseidon_t_widths.contains_key(&u.id)).collect(),
        Gates::Role(role) => filter_gate_uses(uses, cgi.role_id(role)),
        Gates::RangeChunks(n) => {
            uses.iter().filter(|u| cgi.range_check_bits.contains_key(&u.id) && u.signals.len() == 1 + n).collect()
        }
    }
}

/// Places `layout` (writes into `sheet`, or only counts without one).
pub fn place<F: PlonkField + QuotientMap<u64>>(
    layout: &AirLayout,
    prep: &Prepared<'_, F>,
    blocks: &HashBlocks<'_, F>,
    mut sheet: Option<&mut Sheet<F>>,
) -> Placed {
    let mut rows = match layout.blocks {
        Blocks::None => Rows::Seq(0),
        Blocks::Blake3 { lanes } => Rows::Interiors { alloc: RowAlloc::new(lanes), hash_blocks: blocks.n_blocks() },
        Blocks::Blake3Wrap => Rows::WrapBand { blocks: blocks.n_blocks(), taken: 0, last: None },
    };
    if let Some(sh) = sheet.as_deref_mut() {
        blocks.place(sh);
    }
    let mut tiers: Vec<VecDeque<usize>> = vec![VecDeque::new(); layout.row_kinds.len()];
    let mut n_uses = blocks.n_uses();

    for s in &layout.sections {
        let uses = select(prep, s.gates);
        n_uses += uses.len();
        rows.open(uses.len().div_ceil(s.per_row), s.rows);
        for chunk in uses.chunks(s.per_row) {
            let r = rows.take(s.rows);
            for &(off, k) in &s.row_kinds {
                tiers[k].push_back(r + off);
            }
            if let Some(sh) = sheet.as_deref_mut() {
                write_section_row(layout, s, prep, chunk, r, sh);
            }
        }
    }

    let plonk_rows = match &layout.plonk {
        PlonkPolicy::Stream { pure } => stream(layout, pure, &prep.constraints, &mut tiers, &mut rows, sheet),
        PlonkPolicy::GroupByKey { gates_per_row, sorted, selector } => {
            group_by_key(layout, *gates_per_row, *sorted, *selector, &prep.constraints, &tiers, &mut rows, sheet)
        }
    };
    let band_blocks = match &rows {
        Rows::Interiors { alloc, .. } => alloc.blocks_used(),
        _ => 0,
    };
    Placed { first_public_row: rows.end(), plonk_rows, band_blocks, uses: n_uses }
}

fn write_section_row<F: PlonkField + QuotientMap<u64>>(
    layout: &AirLayout,
    s: &Section,
    prep: &Prepared<'_, F>,
    chunk: &[&CustomGateUse],
    r: usize,
    sh: &mut Sheet<F>,
) {
    for &(off, k) in &s.row_kinds {
        sh.band.allow(r + off, layout.row_kinds[k].mask);
    }
    let slots = if s.pad { s.per_row } else { chunk.len() };
    for slot in 0..slots {
        let cgu = chunk[slot.min(chunk.len() - 1)];
        if let Some(n) = s.n_signals {
            assert_eq!(cgu.signals.len(), n, "a {:?} use has {} signals, not {n}", s.gates, cgu.signals.len());
        }
        for run in &s.cells {
            let len = run.len.min(cgu.signals.len().saturating_sub(run.signal));
            for j in 0..len {
                sh.s_map[run.col + j + slot * s.stride][r + run.row] = cgu.signals[run.signal + j] as u32;
            }
        }
    }
    let mut payload = 0;
    match s.fill {
        ConstFill::None => {}
        ConstFill::Fft4 => {
            let p = prep.cgi.fft4_parameters.get(&chunk[0].id).expect("FFT4 params");
            for (i, &v) in fft4_constants(p).iter().enumerate() {
                sh.cv[i % layout.c_cols][r + i / layout.c_cols] = v;
            }
        }
        ConstFill::PoseidonT => {
            if sh.rc.is_empty() {
                sh.rc = ROUND_CONSTANTS.iter().map(lift).collect();
            }
            for k in 0..s.rows {
                let row = r + k;
                sh.set("POSEIDON", 0, row, F::ONE);
                if k == ROUNDS {
                    continue;
                }
                for j in 0..WIDTH {
                    let rc = sh.rc[WIDTH * k + j];
                    sh.set("K", j, row, rc);
                }
                sh.set(if is_full_round(k) { "POSEIDON_FULL_ROUND" } else { "POSEIDON_PARTIAL_ROUND" }, 0, row, F::ONE);
            }
        }
        ConstFill::RangeCheck { weights } => {
            let n_chunks = chunk[0].signals.len() - 1;
            sh.set("RANGE_CHECK", 0, r, F::ONE);
            let mut weight = F::ONE;
            for k in 0..n_chunks {
                sh.set(weights, k, r, weight);
                weight *= F::from_int(1u64 << RANGE_CHECK_CHUNK_BITS);
            }
            payload = n_chunks as u64;
        }
    }
    if let Some(kind) = s.band {
        sh.gate_bands.push(GateBand { row: r as u32, kind, payload });
    }
}

/// Per key, the open slot: (row, next gate, end).
type Partial = HashMap<String, (usize, usize, usize)>;

/// Opens coefficient group `g` of `row` with `c`; the rest of its gates stay open to `c`'s key.
fn open_group<F: PlonkField>(
    row: usize,
    g: &GateGroup,
    c: &PlonkConstraint<F>,
    k: String,
    partial: &mut Partial,
    sheet: &mut Option<&mut Sheet<F>>,
) {
    if let Some(sh) = sheet.as_deref_mut() {
        sh.coeffs(row, g.coef, c);
        for gate in g.gates.clone() {
            sh.band.put(&mut sh.s_map, row, gate, c);
        }
    }
    if g.gates.start + 1 < g.gates.end {
        partial.insert(k, (row, g.gates.start + 1, g.gates.end));
    }
}

/// [`PlonkPolicy::Stream`]; returns the new `pure` rows.
fn stream<F: PlonkField>(
    layout: &AirLayout,
    pure: &RowKind,
    constraints: &[PlonkConstraint<F>],
    tiers: &mut [VecDeque<usize>],
    rows: &mut Rows,
    mut sheet: Option<&mut Sheet<F>>,
) -> usize {
    let mut partial = Partial::new();
    let mut half: VecDeque<(usize, GateGroup)> = VecDeque::new();
    let mut n_pure = 0;
    for c in constraints {
        let k = ckey(c);
        if let Some(p) = partial.get_mut(&k) {
            if let Some(sh) = sheet.as_deref_mut() {
                sh.band.put(&mut sh.s_map, p.0, p.1, c);
            }
            p.1 += 1;
            if p.1 == p.2 {
                partial.remove(&k);
            }
            continue;
        }
        if let Some((row, g)) = half.pop_front() {
            open_group(row, &g, c, k, &mut partial, &mut sheet);
            continue;
        }
        let tier = tiers.iter_mut().zip(&layout.row_kinds).find_map(|(q, kind)| Some((q.pop_front()?, kind)));
        let (row, kind) = tier.unwrap_or_else(|| {
            let row = rows.take(1);
            if let Some(sh) = sheet.as_deref_mut() {
                sh.band.allow(row, pure.mask);
            }
            n_pure += 1;
            (row, pure)
        });
        open_group(row, &kind.groups[0], c, k, &mut partial, &mut sheet);
        half.extend(kind.groups[1..].iter().map(|g| (row, g.clone())));
    }
    n_pure
}

/// The leftover plan of [`PlonkPolicy::GroupByKey`]: the groups, in placement order, and how many
/// of each group's last constraints take a piggyback row's single gate.
pub fn plan_groups<F: PlonkField>(
    constraints: &[PlonkConstraint<F>],
    gates_per_row: usize,
    sorted: bool,
    piggyback: usize,
) -> (Vec<Vec<&PlonkConstraint<F>>>, Vec<usize>) {
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut keyed: Vec<(String, Vec<&PlonkConstraint<F>>)> = Vec::new();
    for c in constraints {
        let k = ckey(c);
        let g = *index.entry(k.clone()).or_insert_with(|| {
            keyed.push((k, Vec::new()));
            keyed.len() - 1
        });
        keyed[g].1.push(c);
    }
    if sorted {
        keyed.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    }
    let groups: Vec<_> = keyed.into_iter().map(|(_, g)| g).collect();

    // Leftovers of 1 first (one gate saves a row), then of 2, then whole rows.
    let mut free = piggyback;
    let mut in_gate_rows = vec![0; groups.len()];
    for left_over in [1, 2] {
        for (g, group) in groups.iter().enumerate() {
            if group.len() % gates_per_row == left_over && free >= left_over {
                in_gate_rows[g] = left_over;
                free -= left_over;
            }
        }
    }
    for (g, group) in groups.iter().enumerate() {
        let rows = ((group.len() - in_gate_rows[g]) / gates_per_row).min(free / gates_per_row);
        in_gate_rows[g] += gates_per_row * rows;
        free -= gates_per_row * rows;
    }
    (groups, in_gate_rows)
}

/// [`PlonkPolicy::GroupByKey`]; returns the dedicated rows.
#[allow(clippy::too_many_arguments)]
fn group_by_key<F: PlonkField>(
    layout: &AirLayout,
    gates_per_row: usize,
    sorted: bool,
    selector: Option<&'static str>,
    constraints: &[PlonkConstraint<F>],
    tiers: &[VecDeque<usize>],
    rows: &mut Rows,
    mut sheet: Option<&mut Sheet<F>>,
) -> usize {
    let mut piggyback =
        tiers.iter().zip(&layout.row_kinds).flat_map(|(q, k)| q.iter().map(move |&r| (r, &k.groups[0])));
    let n_piggyback = tiers.iter().map(VecDeque::len).sum();
    let (groups, in_gate_rows) = plan_groups(constraints, gates_per_row, sorted, n_piggyback);
    let n_rows: usize = groups.iter().zip(&in_gate_rows).map(|(g, &b)| (g.len() - b).div_ceil(gates_per_row)).sum();
    rows.open(n_rows, 1);
    let mask = (1u16 << gates_per_row) - 1;
    for (group, &in_gate) in groups.iter().zip(&in_gate_rows) {
        let (own, shared) = group.split_at(group.len() - in_gate);
        for &c in shared {
            let (row, g) = piggyback.next().expect("the piggyback rows were counted");
            assert_eq!(g.gates.len(), 1, "a key's leftovers take single-gate rows");
            if let Some(sh) = sheet.as_deref_mut() {
                sh.coeffs(row, g.coef, c);
                sh.band.put(&mut sh.s_map, row, g.gates.start, c);
            }
        }
        for gates in own.chunks(gates_per_row) {
            let row = rows.take(1);
            let Some(sh) = sheet.as_deref_mut() else { continue };
            sh.band.allow(row, mask);
            if let Some(name) = selector {
                sh.set(name, 0, row, F::ONE);
            }
            sh.coeffs(row, 0, gates[0]);
            // A short row repeats its last constraint: a gate on zeros reads `qC`.
            for g in 0..gates_per_row {
                sh.band.put(&mut sh.s_map, row, g, gates[g.min(gates.len() - 1)]);
            }
        }
    }
    n_rows
}
