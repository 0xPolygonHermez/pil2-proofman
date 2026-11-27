//! An air's constraints: the setup's `constraintDag` (verifierinfo.json) checked and resolved once by the
//! `constraint-dag` crate, then evaluated at xi for each proof: Horner(roots, std_vc) * Zi, as C++
//! `evalConstraintDag`. This verifier reads no proof, air or airgroup value: a DAG with one is refused.

use alloc::string::ToString;

use constraint_dag::{ConstraintDag, DagError, DagField, LowerCtx, LowerTables, Operand, ResolvedDag};
use proofman_fields::{CubicExtensionField, Field, Goldilocks};

type E3 = CubicExtensionField<Goldilocks>;

/// A leaf as this verifier reads it: a constant already in the field, or an index into the proof's inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leaf {
    Number(Goldilocks),
    Eval(u32),
    Challenge(u32),
    Public(u32),
}

/// An air's constraint DAG, checked and resolved; only `from_json` builds one, once per verifier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AirConstraints(ResolvedDag<Leaf>);

impl AirConstraints {
    /// The DAG of `json` (verifierinfo's `constraintDag`) validated over Goldilocks and resolved against `t`.
    pub fn from_json(json: &str, t: &LowerTables) -> Result<Self, DagError> {
        let dag = ConstraintDag::from_air_json(json, DagField::Goldilocks)?;
        let leaf = |node, op| match op {
            // a validated constant is a canonical decimal below the Goldilocks prime
            Operand::Number(c) => c
                .parse()
                .map(|c| Leaf::Number(Goldilocks::new(c)))
                .map_err(|_| DagError::Decode { what: "a constant that is not a Goldilocks element".to_string() }),
            Operand::Eval(k) => Ok(Leaf::Eval(k)),
            Operand::Challenge { id, .. } => Ok(Leaf::Challenge(id)),
            Operand::Public(k) => Ok(Leaf::Public(k)),
            _ => Err(DagError::Unresolved {
                node,
                what: "a proof, air or airgroup value: this verifier reads none".to_string(),
            }),
        };
        LowerCtx::new(t.clone())?.resolve_dag(&dag, leaf).map(AirConstraints)
    }

    /// Horner(roots, challenges[std_vc]) * zi; `None` when an input is shorter than a leaf reads.
    pub fn eval(&self, challenges: &[E3], evals: &[E3], publics: &[Goldilocks], zi: E3) -> Option<E3> {
        let embed = |g: Goldilocks| E3 { value: [g, Goldilocks::ZERO, Goldilocks::ZERO] };
        let acc = self.0.eval(|l| match *l {
            Leaf::Number(c) => Some(embed(c)),
            Leaf::Eval(k) => evals.get(k as usize).copied(),
            Leaf::Challenge(k) => challenges.get(k as usize).copied(),
            Leaf::Public(k) => publics.get(k as usize).copied().map(embed),
        })?;
        Some(acc * zi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::string::String;
    use constraint_dag::{ColRef, ColType};

    /// evMap: witness 0 at rows 0 and -1, fixed 0; challenges 2.0 (std_alpha) and 3.0 (std_vc); one public.
    fn tables() -> LowerTables {
        let ev = |col, offset| ColRef { col, commit_id: None, id: 0, offset };
        LowerTables {
            ev_map: alloc::vec![ev(ColType::Witness, 0), ev(ColType::Witness, -1), ev(ColType::Fixed, 0)],
            challenges: alloc::vec![(2, 0), (3, 0)],
            vc: 1,
            n_publics: 1,
            ext_dim: 3,
            ..Default::default()
        }
    }

    fn json(nodes: &str, roots: &str) -> String {
        format!(r#"{{"version":1,"field":"gl","extDim":3,"nodes":{nodes},"roots":{roots}}}"#)
    }

    fn e3(a: u64, b: u64, c: u64) -> E3 {
        E3 { value: [Goldilocks::new(a), Goldilocks::new(b), Goldilocks::new(c)] }
    }

    /// c0 = w0 * alpha - 7, c1 = w0' + fixed0 - public0: (vc * c0 + c1) * zi.
    #[test]
    fn horner_of_the_roots_times_zi() {
        let nodes = r#"[[5,0,0,0],[6,2,0],[2,0,1],[4,"7"],[1,2,3],[5,0,0,-1],[5,1,0,0],[0,5,6],[7,0],[1,7,8]]"#;
        let dag = AirConstraints::from_json(&json(nodes, r#"[[4,0,"c0"],[9,0,"c1"]]"#), &tables()).unwrap();
        let (evals, challenges) = ([e3(1, 2, 3), e3(4, 5, 6), e3(7, 8, 9)], [e3(10, 11, 12), e3(13, 14, 15)]);
        let (publics, zi) = ([Goldilocks::new(16)], e3(17, 18, 19));
        let embed = |v| e3(v, 0, 0);
        let c0 = evals[0] * challenges[0] - embed(7);
        let c1 = evals[1] + evals[2] - embed(16);
        assert_eq!(dag.eval(&challenges, &evals, &publics, zi), Some((challenges[1] * c0 + c1) * zi));
        assert_eq!(dag.eval(&challenges, &evals[..2], &publics, zi), None, "evals shorter than the DAG reads");
        assert_eq!(dag.eval(&challenges, &evals, &[], zi), None, "no publics");
    }

    /// Each refusal of the former verifier-local checks, now `constraint_dag`'s.
    #[test]
    fn refusals() {
        let refused = |nodes: &str, roots: &str| AirConstraints::from_json(&json(nodes, roots), &tables()).is_err();
        let r0 = r#"[[0,0,"r"]]"#;
        assert!(refused(r#"[[5,0,0,0],[5,0,0,-1],[0,1,0]]"#, r#"[[2,0,"r"]]"#), "commutative children unsorted");
        assert!(refused(r#"[[3,0]]"#, r0), "a self reference");
        assert!(refused(r#"[[5,0,0,0],[5,0,0,0],[0,0,1]]"#, r#"[[2,0,"r"]]"#), "a repeated node");
        assert!(refused(r#"[[5,0,0,0],[7,0]]"#, r0), "a node no root reaches");
        assert!(refused(r#"[[4,"07"]]"#, r0), "a constant not canonical");
        assert!(refused(r#"[[4,"18446744069414584321"]]"#, r0), "a constant out of range");
        assert!(refused(r#"[[7,0]]"#, r#"[[0,1,"r"]]"#), "a global root");
        assert!(refused(r#"[[10,0,1]]"#, r0), "an airgroup value naming its airgroup");
        for value in ["[[8,0]]", "[[9,0]]", "[[10,0]]"] {
            assert!(refused(value, r0), "a value: {value}");
        }
        assert!(refused(r#"[[5,0,1,0]]"#, r0), "a column the evMap does not open");
        assert!(refused(r#"[[6,2,1]]"#, r0), "an unknown challenge");
        assert!(refused(r#"[[7,1]]"#, r0), "a public out of range");
        assert!(refused(r#"[[11,0]]"#, r0), "unknown opcode");
        assert!(refused(r#"[[5,0,0]]"#, r0), "wrong number of fields");
        assert!(refused(r#"[[7,0]]"#, r#"[[1,0,"r"]]"#), "a root past the nodes");
        let mut t = tables();
        t.vc = 2;
        assert!(AirConstraints::from_json(&json("[[7,0]]", r0), &t).is_err(), "std_vc past the challenges");
        let other = r#"{"version":2,"field":"gl","extDim":3,"nodes":[[7,0]],"roots":[[0,0,"r"]]}"#;
        assert!(AirConstraints::from_json(other, &tables()).is_err(), "unsupported DAG version");
    }
}
