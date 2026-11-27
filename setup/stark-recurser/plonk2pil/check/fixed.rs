//! Tier B: what the fixed columns make the PIL enforce — copy classes from the `S` permutation
//! (decoded against the identity `k·ID`, never from the exec map), and the plonk equations each slot
//! enforces on the rows its selector is non-zero.

use std::collections::HashMap;

use proofman_common::exec_format::ExecFile;

use super::pil::{FixedPoly, PilAir, Slot};
use super::poly::{mono, Mono, Poly};
use super::CheckError;
use crate::plonk2pil::field::PlonkField;

/// The fixed columns, by pilout fixed id.
pub trait FixedSource<F> {
    fn column(&self, id: u32) -> Result<Vec<F>, CheckError>;
}

/// Loaded columns, each once.
pub struct Columns<'a, F> {
    src: &'a dyn FixedSource<F>,
    n: usize,
    cache: HashMap<u32, Vec<F>>,
}

impl<'a, F: PlonkField> Columns<'a, F> {
    pub fn new(src: &'a dyn FixedSource<F>, n: usize) -> Self {
        Self { src, n, cache: HashMap::new() }
    }

    pub fn get(&mut self, id: u32) -> Result<&[F], CheckError> {
        if !self.cache.contains_key(&id) {
            let c = self.src.column(id)?;
            if c.len() != self.n {
                return Err(CheckError::Artefact(format!(
                    "fixed: fixed column {id} has {} rows, not {}",
                    c.len(),
                    self.n
                )));
            }
            self.cache.insert(id, c);
        }
        Ok(&self.cache[&id])
    }

    /// A column read without caching it (the `S` columns, each read once).
    pub fn take(&mut self, id: u32) -> Result<Vec<F>, CheckError> {
        match self.cache.remove(&id) {
            Some(c) => Ok(c),
            None => self.src.column(id),
        }
    }

    /// `p` on every row, rows wrapping.
    pub fn eval(&mut self, p: &FixedPoly<F>) -> Result<Vec<F>, CheckError> {
        let n = self.n;
        let mut out = vec![F::ZERO; n];
        for (m, c) in &p.0 {
            let mut term = vec![*c; n];
            for &(id, off) in m {
                let col = self.get(id)?;
                for (r, t) in term.iter_mut().enumerate() {
                    *t *= col[wrap(r, off, n)];
                }
            }
            for (o, t) in out.iter_mut().zip(term) {
                *o += t;
            }
        }
        Ok(out)
    }

    pub fn n(&self) -> usize {
        self.n
    }
}

pub fn wrap(r: usize, off: i64, n: usize) -> usize {
    (r as i64 + off).rem_euclid(n as i64) as usize
}

/// Copy classes of the witness cells: a cell is `wit·N + row`; a class is named by its least cell.
pub struct Classes {
    pub n: usize,
    /// Connection index of each witness column.
    conn_of: Vec<Option<u32>>,
    /// Class of each connection cell, by `conn·N + row`.
    class: Vec<u32>,
    /// Cells of each connection class, CSR by class order.
    member_at: HashMap<u32, (u32, u32)>,
    members: Vec<u32>,
    pub n_classes: usize,
}

impl Classes {
    pub fn class(&self, wit: u32, row: usize) -> u32 {
        match self.conn_of.get(wit as usize).copied().flatten() {
            Some(j) => self.class[j as usize * self.n + row],
            None => wit * self.n as u32 + row as u32,
        }
    }

    /// The cells of class `c`.
    pub fn cells_of(&self, c: u32) -> Vec<u32> {
        match self.member_at.get(&c) {
            Some(&(a, b)) => self.members[a as usize..b as usize].to_vec(),
            None => vec![c],
        }
    }
}

/// Decodes the permutation `σ` of the connection from the `S` columns: `S_j[i] = k_{j'}·ID[i']`
/// names the cell `(j', i')`. `S` must be a bijection onto the identities; classes are `σ`'s cycles.
pub fn classes<F: PlonkField>(air: &PilAir<F>, cols: &mut Columns<F>, n_wit: usize) -> Result<Classes, CheckError> {
    let n = cols.n();
    let k = air.conn.len();
    if k == 0 {
        return Err(CheckError::Artefact("pilout: no connection columns".into()));
    }
    let id = cols.get(air.conn[0].id_fixed)?.to_vec();
    // ID is the group of order N: ID[i] = ID[1]^i, ID[N-1]·ID[1] = 1, all distinct.
    let mut where_id: HashMap<F, u32> = HashMap::with_capacity(n);
    for (i, v) in id.iter().enumerate() {
        let expect = if i == 0 { F::ONE } else { id[i - 1] * id[1 % n] };
        if *v != expect || where_id.insert(*v, i as u32).is_some() {
            return Err(CheckError::Artefact(format!("pilout: ID is not the subgroup of order {n} at row {i}")));
        }
    }
    if n > 1 && id[n - 1] * id[1] != F::ONE {
        return Err(CheckError::Artefact("pilout: ID does not close a subgroup".into()));
    }
    // coset key k_j^N -> j
    let pow_n = |mut x: F| {
        let mut e = n;
        while e > 1 {
            x = x * x;
            e >>= 1;
        }
        x
    };
    let mut coset: HashMap<F, u32> = HashMap::new();
    for (j, c) in air.conn.iter().enumerate() {
        if c.k.is_zero() || coset.insert(pow_n(c.k), j as u32).is_some() {
            return Err(CheckError::Artefact(format!(
                "pilout: connection column {j}: its identity coset is not disjoint"
            )));
        }
    }
    let kinv: Vec<F> = air.conn.iter().map(|c| c.k.inverse()).collect();
    let mut sigma = vec![u32::MAX; k * n];
    let mut hit = vec![false; k * n];
    for (j, c) in air.conn.iter().enumerate() {
        let s = cols.take(c.s_fixed)?;
        if s.len() != n {
            return Err(CheckError::Artefact(format!("fixed: S column {j} has {} rows", s.len())));
        }
        for (i, v) in s.iter().enumerate() {
            // fast path: a fixed point of σ, as most cells are
            let target = if *v == c.k * id[i] {
                Some(j * n + i)
            } else {
                coset
                    .get(&pow_n(*v))
                    .and_then(|&jj| where_id.get(&(*v * kinv[jj as usize])).map(|&ii| jj as usize * n + ii as usize))
            };
            let Some(t) = target.filter(|t| !std::mem::replace(&mut hit[*t], true)) else {
                return Err(CheckError::NotAPermutation { col: j, row: i });
            };
            sigma[j * n + i] = t as u32;
        }
    }
    drop(hit);
    let mut conn_of = vec![None; n_wit];
    for (j, c) in air.conn.iter().enumerate() {
        conn_of[c.wit as usize] = Some(j as u32);
    }
    let cell = |x: u32| air.conn[x as usize / n].wit * n as u32 + (x as usize % n) as u32;
    // cycles: class = least cell; members gathered per cycle
    let mut class = vec![u32::MAX; k * n];
    let (mut members, mut member_at, mut n_classes) = (Vec::new(), HashMap::new(), 0usize);
    let mut cyc = Vec::new();
    for start in 0..k * n {
        if class[start] != u32::MAX {
            continue;
        }
        cyc.clear();
        let mut x = start as u32;
        loop {
            cyc.push(x);
            class[x as usize] = 0;
            x = sigma[x as usize];
            if x as usize == start {
                break;
            }
        }
        let rep = cyc.iter().map(|&x| cell(x)).min().expect("a cycle has a cell");
        for &x in &cyc {
            class[x as usize] = rep;
        }
        n_classes += 1;
        if cyc.len() > 1 {
            let a = members.len() as u32;
            members.extend(cyc.iter().map(|&x| cell(x)));
            member_at.insert(rep, (a, members.len() as u32));
        }
    }
    Ok(Classes { n, conn_of, class, member_at, members, n_classes })
}

/// Wire labels from the exec map (a hint): every cell holding wire `w` must be in one class.
pub fn labels<F: PlonkField>(
    exec: &ExecFile<F>,
    cls: &Classes,
    map_wit: &[u32],
    n_wires: usize,
) -> Result<Vec<u32>, CheckError> {
    let mut label = vec![u32::MAX; n_wires];
    let (rows, cols) = (exec.layout.map_rows(), exec.layout.map_cols());
    if cols > map_wit.len() || rows > cls.n {
        return Err(CheckError::Artefact(format!("exec: the exec map is {rows} x {cols}, past the air")));
    }
    for r in 0..rows {
        for (c, &wit) in map_wit.iter().enumerate().take(cols) {
            let w = exec.map_entry(r, c) as usize;
            if w == 0 {
                continue;
            }
            let class = cls.class(wit, r);
            let l = label
                .get_mut(w)
                .ok_or_else(|| CheckError::Artefact(format!("exec: wire {w} past the r1cs and its additions")))?;
            if *l == u32::MAX {
                *l = class;
            } else if *l != class {
                return Err(CheckError::WireSplit { wire: w as u32 });
            }
        }
    }
    Ok(label)
}

/// One enforced equation instance: slot `slot` on `row`.
pub type EqRef = (u32, u32);

/// The slots with their selectors evaluated, which the equations are decoded from.
pub struct Enforced<F> {
    pub slots: Vec<Slot<F>>,
    pub fired: Vec<Vec<u32>>,
}

impl<F: PlonkField> Enforced<F> {
    pub fn new(air: &PilAir<F>, cols: &mut Columns<F>) -> Result<Self, CheckError> {
        let mut fired = Vec::with_capacity(air.slots.len());
        for s in &air.slots {
            let sel = cols.eval(&s.sel)?;
            fired.push((0..sel.len()).filter(|&r| !sel[r].is_zero()).map(|r| r as u32).collect());
        }
        Ok(Self { slots: air.slots.clone(), fired })
    }

    pub fn refs(&self) -> impl Iterator<Item = EqRef> + '_ {
        self.fired.iter().enumerate().flat_map(|(s, rows)| rows.iter().map(move |&r| (s as u32, r)))
    }

    /// The equation `Σ_t Q_t(row)·w_t(row) = 0` over copy classes (selector divided out).
    pub fn equation(&self, (s, r): EqRef, cols: &mut Columns<F>, cls: &Classes) -> Result<Poly<F>, CheckError> {
        let n = cls.n;
        let r = r as usize;
        let mut terms: Vec<(Mono, F)> = Vec::with_capacity(4);
        for (cells, (qid, qoff)) in &self.slots[s as usize].parts {
            let q = cols.get(*qid)?[wrap(r, *qoff, n)];
            if q.is_zero() {
                continue;
            }
            let v: Vec<u32> = cells.iter().map(|&(w, o)| cls.class(w, wrap(r, o, n))).collect();
            let m = match v[..] {
                [] => mono(super::poly::ONE, super::poly::ONE),
                [a] => (a, super::poly::ONE),
                [a, b] => mono(a, b),
                _ => unreachable!("slots are of degree <= 2"),
            };
            terms.push((m, q));
        }
        Ok(Poly::new(terms))
    }
}
