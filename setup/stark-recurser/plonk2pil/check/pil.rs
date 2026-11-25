//! Tier C: what the compiled PIL enforces, read off the pilout's constraint DAG.
//!
//! Every everyRow constraint that expands to `s·Σ_t Q_t·w_t` (`s` a polynomial in fixed columns,
//! each `Q_t` one fixed column, each `w_t` a product of at most two witness cells) is a plonk *slot*;
//! `s·(cell − public_i)` binds a public; the std's `Connection` bus hints name the permuted columns.
//! Everything else is a custom gate's constraint, kept as the set of cells it reads.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use num_bigint::BigUint;
use pil_info::core::ir::{Boundary, ColType, ConstraintSystem, Expr, HintValue};

use super::CheckError;
use crate::plonk2pil::field::{modulus, n8, PlonkField};

/// A leaf of the expanded DAG: fixed or witness column at a row offset, or a public.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Atom {
    Fixed(u32, i64),
    Wit(u32, i64),
    Pub(u32),
}

/// `Σ coef·Π fixed(id, offset)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FixedPoly<F>(pub Vec<(Vec<(u32, i64)>, F)>);

/// `(w_t as witness cells (col, offset), Q_t as a fixed (col, offset))`.
pub type Part = (Vec<(u32, i64)>, (u32, i64));

/// `s · Σ_t Q_t·w_t = 0` on every row.
#[derive(Debug, Clone)]
pub struct Slot<F> {
    pub sel: FixedPoly<F>,
    /// `(w_t as witness cells (col, offset), Q_t as a fixed (col, offset))`; `w_t` empty is the constant.
    pub parts: Vec<Part>,
    pub constraint: usize,
}

/// `s · (cell − publics[public]) = 0` on every row.
#[derive(Debug, Clone)]
pub struct PublicBind<F> {
    pub sel: FixedPoly<F>,
    pub cell: (u32, i64),
    pub public: u32,
}

/// One permuted column: witness `wit` against identity `k·ID` and permutation `S`.
#[derive(Debug, Clone, Copy)]
pub struct ConnCol<F> {
    pub wit: u32,
    pub k: F,
    pub id_fixed: u32,
    pub s_fixed: u32,
}

/// The pilout's constraints, sorted by what they are.
#[derive(Debug, Clone)]
pub struct PilAir<F> {
    pub n_rows: usize,
    pub slots: Vec<Slot<F>>,
    pub publics: Vec<PublicBind<F>>,
    pub conn: Vec<ConnCol<F>>,
    /// The cells each remaining constraint reads, by constraint.
    pub other: Vec<(usize, Rc<BTreeSet<Atom>>)>,
}

/// An element of `F` from a pilout constant, reduced.
pub fn fe<F: PlonkField>(v: &BigUint) -> F {
    let mut bytes = (v % modulus::<F>()).to_bytes_le();
    bytes.resize(n8::<F>(), 0);
    F::from_canonical_le(&bytes).expect("reduced below the prime")
}

/// The pilout fixed id of `name` (its last `.` segment) at array index `index`.
pub fn fixed_id(cs: &ConstraintSystem, name: &str, index: Option<u32>) -> Option<u32> {
    cs.fixed.iter().position(|c| short(&c.name) == name && lengths_match(&c.lengths, index)).map(|i| i as u32)
}

/// The witness id of `a[c]`, the committed columns plonk2pil's exec map indexes.
pub fn wit_id(cs: &ConstraintSystem, name: &str, index: u32) -> Option<u32> {
    cs.witness
        .iter()
        .position(|c| c.stage == 1 && short(&c.name) == name && lengths_match(&c.lengths, Some(index)))
        .map(|i| i as u32)
}

fn short(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

fn lengths_match(l: &Option<Vec<u32>>, index: Option<u32>) -> bool {
    match (l, index) {
        (None, None) => true,
        (Some(v), Some(i)) => v.as_slice() == [i],
        _ => false,
    }
}

type Exp<F> = BTreeMap<Vec<Atom>, F>;

/// Expansion bound: a slot is ~50 terms; past this a constraint is a custom gate's.
const CAP: usize = 1024;
const MAX_DEG: usize = 6;

struct Expander<'a, F> {
    cs: &'a ConstraintSystem,
    memo: HashMap<u32, Option<Rc<Exp<F>>>>,
}

impl<F: PlonkField> Expander<'_, F> {
    fn expand(&mut self, e: &Expr) -> Option<Rc<Exp<F>>> {
        let leaf = |a: Atom| Some(Rc::new(BTreeMap::from([(vec![a], F::ONE)])));
        match e {
            Expr::Const(c) => Some(Rc::new(BTreeMap::from([(vec![], fe::<F>(c))]))),
            Expr::Col { col: ColType::Fixed, id, offset, .. } => leaf(Atom::Fixed(*id, *offset)),
            Expr::Col { col: ColType::Witness, id, offset, .. } => leaf(Atom::Wit(*id, *offset)),
            Expr::Public(i) => leaf(Atom::Pub(*i)),
            Expr::Ref(id) => {
                if let Some(m) = self.memo.get(id) {
                    return m.clone();
                }
                let r = self.expand(&self.cs.exprs.get(*id as usize)?.clone());
                self.memo.insert(*id, r.clone());
                r
            }
            Expr::Neg(a) => {
                let a = self.expand(a)?;
                Some(Rc::new(a.iter().map(|(m, c)| (m.clone(), -*c)).collect()))
            }
            Expr::Add(a, b) | Expr::Sub(a, b) => {
                let sign = if matches!(e, Expr::Sub(..)) { -F::ONE } else { F::ONE };
                let (a, b) = (self.expand(a)?, self.expand(b)?);
                let mut out = (*a).clone();
                for (m, c) in b.iter() {
                    *out.entry(m.clone()).or_insert(F::ZERO) += sign * *c;
                }
                out.retain(|_, c| !c.is_zero());
                (out.len() <= CAP).then(|| Rc::new(out))
            }
            Expr::Mul(a, b) => {
                let (a, b) = (self.expand(a)?, self.expand(b)?);
                if a.len() * b.len() > CAP * 4 {
                    return None;
                }
                let mut out: Exp<F> = BTreeMap::new();
                for (ma, ca) in a.iter() {
                    for (mb, cb) in b.iter() {
                        let mut m: Vec<Atom> = ma.iter().chain(mb).copied().collect();
                        if m.len() > MAX_DEG {
                            return None;
                        }
                        m.sort();
                        *out.entry(m).or_insert(F::ZERO) += *ca * *cb;
                    }
                }
                out.retain(|_, c| !c.is_zero());
                (out.len() <= CAP).then(|| Rc::new(out))
            }
            _ => None,
        }
    }
}

type Coef<F> = BTreeMap<Vec<(u32, i64)>, F>;

/// `p / a` if every monomial of `p` holds the fixed atom `a`.
fn divide<F: PlonkField>(p: &Coef<F>, a: (u32, i64)) -> Option<Coef<F>> {
    p.iter()
        .map(|(m, c)| {
            let i = m.iter().position(|x| *x == a)?;
            let mut m = m.clone();
            m.remove(i);
            Some((m, *c))
        })
        .collect()
}

fn atoms_of<F>(p: &Coef<F>) -> BTreeSet<(u32, i64)> {
    p.keys().flatten().copied().collect()
}

fn classify<F: PlonkField>(e: &Exp<F>, constraint: usize) -> Option<Result<Slot<F>, PublicBind<F>>> {
    // witness part (incl. publics) -> coefficient over fixed atoms
    let mut parts: BTreeMap<Vec<Atom>, Coef<F>> = BTreeMap::new();
    for (m, c) in e {
        let (w, f): (Vec<Atom>, Vec<Atom>) = m.iter().partition(|a| !matches!(a, Atom::Fixed(..)));
        let f = f.iter().map(|a| if let Atom::Fixed(i, o) = a { (*i, *o) } else { unreachable!() }).collect();
        parts.entry(w).or_default().insert(f, *c);
    }
    let publics: Vec<&Vec<Atom>> = parts.keys().filter(|w| w.iter().any(|a| matches!(a, Atom::Pub(_)))).collect();
    if !publics.is_empty() {
        let [p] = publics[..] else { return None };
        let (&[Atom::Pub(i)], 2) = (p.as_slice(), parts.len()) else { return None };
        let (x, s) = parts.iter().find(|(w, _)| *w != p)?;
        let &[Atom::Wit(col, off)] = x.as_slice() else { return None };
        let neg: Coef<F> = parts[p].iter().map(|(m, c)| (m.clone(), -*c)).collect();
        return (neg == *s && !s.is_empty())
            .then(|| Err(PublicBind { sel: FixedPoly(s.clone().into_iter().collect()), cell: (col, off), public: i }));
    }
    if parts.keys().any(|w| w.len() > 2) {
        return None;
    }
    let (_, first) = parts.iter().next()?;
    for a in atoms_of(first) {
        let Some(s) = divide(first, a) else { continue };
        let mut out = Vec::with_capacity(parts.len());
        for (w, coef) in &parts {
            let q = atoms_of(coef).into_iter().find(|&b| divide(coef, b).as_ref() == Some(&s));
            let Some(q) = q else { break };
            let cells = w.iter().map(|a| if let Atom::Wit(i, o) = a { (*i, *o) } else { unreachable!() }).collect();
            out.push((cells, q));
        }
        if out.len() == parts.len() {
            return Some(Ok(Slot { sel: FixedPoly(s.into_iter().collect()), parts: out, constraint }));
        }
    }
    None
}

fn atom_set(cs: &ConstraintSystem, e: &Expr, memo: &mut HashMap<u32, Rc<BTreeSet<Atom>>>) -> Rc<BTreeSet<Atom>> {
    match e {
        Expr::Col { col: ColType::Fixed, id, offset, .. } => Rc::new(BTreeSet::from([Atom::Fixed(*id, *offset)])),
        Expr::Col { col: ColType::Witness, id, offset, .. } => Rc::new(BTreeSet::from([Atom::Wit(*id, *offset)])),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
            let (a, b) = (atom_set(cs, a, memo), atom_set(cs, b, memo));
            Rc::new(a.union(&b).copied().collect())
        }
        Expr::Neg(a) => atom_set(cs, a, memo),
        Expr::Ref(id) => {
            if let Some(s) = memo.get(id) {
                return s.clone();
            }
            let s = cs.exprs.get(*id as usize).map_or_else(|| Rc::new(BTreeSet::new()), |x| atom_set(cs, x, memo));
            memo.insert(*id, s.clone());
            s
        }
        _ => Rc::new(BTreeSet::new()),
    }
}

fn named<'a>(fields: &'a [pil_info::core::ir::HintField], name: &str) -> Option<&'a HintValue> {
    fields.iter().find(|f| f.name.as_deref() == Some(name)).map(|f| &f.value)
}

fn konst(v: Option<&HintValue>) -> Option<&BigUint> {
    match v {
        Some(HintValue::Expr(Expr::Const(c))) => Some(c),
        _ => None,
    }
}

/// A column's identity terms `(k, ID)` and permutation columns.
type ConnTerms<F> = (Vec<(F, u32)>, Vec<u32>);

/// The permuted columns of the std's `Connection` bus, from its `gprod_debug_data` hints.
fn connection<F: PlonkField>(cs: &ConstraintSystem, expander: &mut Expander<F>) -> Result<Vec<ConnCol<F>>, CheckError> {
    let bad = |why: String| CheckError::Artefact(format!("pilout: connection: {why}"));
    // wit -> (assumes (k, ID), proves S)
    let mut cols: BTreeMap<u32, ConnTerms<F>> = BTreeMap::new();
    let mut opids = BTreeSet::new();
    for h in cs.hints.iter().filter(|h| h.name == "gprod_debug_data") {
        let f = h.entries();
        if !matches!(named(f, "name_piop"), Some(HintValue::Str(s)) if s == "Connection") {
            continue;
        }
        if konst(named(f, "deg_sel")).is_none_or(|d| *d != BigUint::ZERO) {
            return Err(bad("a selected connection term".into()));
        }
        if let Some(HintValue::Array(o)) = named(f, "opids") {
            opids.extend(o.iter().filter_map(|x| konst(Some(&x.value)).cloned()));
        }
        let ty = konst(named(f, "type_piop")).ok_or_else(|| bad("no type_piop".into()))?.clone();
        let Some(HintValue::Array(ex)) = named(f, "expressions") else { return Err(bad("no expressions".into())) };
        let ex: Vec<Rc<Exp<F>>> = ex
            .iter()
            .filter_map(|x| if let HintValue::Expr(e) = &x.value { Some(e) } else { None })
            .map(|e| expander.expand(e).ok_or_else(|| bad(format!("term {e:?} is not a polynomial"))))
            .collect::<Result<_, _>>()?;
        let [col, id] = &ex[..] else { return Err(bad(format!("{} expressions, not 2", ex.len()))) };
        // `k·atom`, the single monomial of `p`
        let single = |p: &Exp<F>| -> Option<(F, Atom)> {
            let [(m, c)] = p.iter().collect::<Vec<_>>()[..] else { return None };
            let [a] = m[..] else { return None };
            Some((*c, a))
        };
        let Some((k1, Atom::Wit(wit, 0))) = single(col).filter(|(k, _)| *k == F::ONE) else {
            return Err(bad(format!("permuted term {col:?} is not a witness column")));
        };
        let entry = cols.entry(wit).or_default();
        let _ = k1;
        match (ty == BigUint::ZERO, single(id)) {
            (true, Some((k, Atom::Fixed(f, 0)))) => entry.0.push((k, f)),
            (false, Some((k, Atom::Fixed(f, 0)))) if k == F::ONE => entry.1.push(f),
            _ => return Err(bad(format!("identity or permutation {id:?} is not k·(fixed column)"))),
        }
    }
    if opids.len() > 1 {
        return Err(bad(format!("{} connection opids; one is supported", opids.len())));
    }
    let mut out = Vec::new();
    let mut id_col = None;
    for (wit, (a, p)) in cols {
        let ([(k, idf)], [s]) = (a.as_slice(), p.as_slice()) else {
            return Err(bad(format!("witness {wit} has {} identities and {} permutations", a.len(), p.len())));
        };
        if *id_col.get_or_insert(*idf) != *idf {
            return Err(bad("two identity columns".into()));
        }
        out.push(ConnCol { wit, k: *k, id_fixed: *idf, s_fixed: *s });
    }
    Ok(out)
}

/// Tier C: classify every constraint of the air.
pub fn analyse<F: PlonkField>(cs: &ConstraintSystem) -> Result<PilAir<F>, CheckError> {
    let mut ex = Expander::<F> { cs, memo: HashMap::new() };
    let mut sets = HashMap::new();
    let (mut slots, mut publics, mut other) = (Vec::new(), Vec::new(), Vec::new());
    for (i, c) in cs.constraints.iter().enumerate() {
        let every = c.boundary == Boundary::EveryRow;
        match every.then(|| ex.expand(&c.e)).flatten().and_then(|e| classify(&e, i)) {
            Some(Ok(s)) => slots.push(s),
            Some(Err(p)) => publics.push(p),
            None => other.push((i, atom_set(cs, &c.e, &mut sets))),
        }
    }
    Ok(PilAir { n_rows: 1usize << cs.n_rows_bits, slots, publics, conn: connection(cs, &mut ex)?, other })
}
