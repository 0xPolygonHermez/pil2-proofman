//! The constraint DAG: builder (canonical form, hash-consing, roots) and lowering to verifier code.

mod common;

use common::*;
use pil_info::core::dag::build::build_dag;
use pil_info::core::dag::format::{ColType, ConstraintDag, DagBoundary, DagError, DagNode};
use pil_info::core::dag::lower_code::{lower_to_code, Fold, LowerCtx, LowerTables};
use pil_info::core::im::{im_column_id, im_order, ImChoice};
use pil_info::core::ir::{ColRef, ConstraintSystem};
use pil_info::emit::verifier::expression_code;
use pil_info::pil::gen_code::ExpressionCodeEntry;
use pil_info::search::im::search_im;
use pil_info::DegreePolicy;

fn cm(id: u32, offset: i64) -> DagNode {
    DagNode::Col { col: ColType::Witness, commit_id: None, id, offset }
}

fn count(d: &ConstraintDag, f: impl Fn(&DagNode) -> bool) -> usize {
    d.nodes.iter().filter(|n| f(n)).count()
}

/// 4 witness columns a,b,c,d (ids 0..3), the given expressions, one every-row constraint per `roots` entry.
fn dag_of(exprs: Vec<pil2_pilout::pilout::Expression>, roots: &[u32], im: &[u32]) -> ConstraintDag {
    build_dag(&system(exprs, roots), &choice(im)).unwrap()
}

fn mul_count(d: &ConstraintDag) -> usize {
    count(d, |n| matches!(n, DagNode::Mul { .. }))
}

// ---- builder ---------------------------------------------------------------------------------

#[test]
fn shared_subexpression_is_one_node() {
    // 0: a*b used by two roots
    let e = vec![mul(witness(0, 0), witness(1, 0)), add(exp(0), witness(2, 0)), sub(exp(0), witness(3, 0))];
    let d = dag_of(e, &[1, 2], &[]);
    assert_eq!(mul_count(&d), 1);
    assert_eq!(d.nodes.len(), 7); // a b a*b c + d -
                                  // the same product written twice is still one node
    let e = vec![
        mul(witness(0, 0), witness(1, 0)),
        mul(witness(0, 0), witness(1, 0)),
        add(exp(0), witness(2, 0)),
        sub(exp(1), witness(3, 0)),
    ];
    let d = dag_of(e, &[2, 3], &[]);
    assert_eq!(mul_count(&d), 1);
    assert_eq!(d.nodes.len(), 7);
}

#[test]
fn intermediate_becomes_column() {
    // root e1 = e0*c with e0 = a*b promoted: the root reads the im column, not a*b
    let cs = system(chain(), &[1]);
    let im_col = DagNode::Col { col: ColType::Witness, commit_id: None, id: im_column_id(&cs, 0), offset: 0 };
    let d = build_dag(&cs, &choice(&[0])).unwrap();
    assert_eq!(d.roots.len(), 2);
    let DagNode::Mul { a, b } = d.nodes[d.roots[0].node as usize].clone() else { panic!("root 0 is a product") };
    assert!(d.nodes[a as usize] == im_col || d.nodes[b as usize] == im_col, "{d:?}");
    // a*b appears once, only inside the im root
    assert_eq!(mul_count(&d), 2);
    assert_eq!(count(&d, |n| *n == im_col), 1);
}

#[test]
fn roots_are_constraints_then_im_roots() {
    let cs = system(chain(), &[2, 1]);
    let im = choice(&[0, 1]);
    let d = build_dag(&cs, &im).unwrap();
    assert_eq!(d.roots.len(), 4);
    assert!(d.roots.iter().all(|r| r.boundary == DagBoundary::EveryRow));
    assert_eq!(d.roots[0].label, "constraint on expression 2");
    assert_eq!(d.roots[1].label, "constraint on expression 1");
    assert!(matches!(d.nodes[d.roots[0].node as usize], DagNode::Mul { .. }));
    // constraint 1 is on promoted expression 1 itself, so it reads the im column
    assert_eq!(d.nodes[d.roots[1].node as usize], cm(im_column_id(&cs, 1), 0));
    // `im_e - e`, in im order
    for k in 0..2 {
        let DagNode::Sub { a, b } = d.nodes[d.roots[2 + k].node as usize].clone() else { panic!("im root {k}") };
        let want = DagNode::Col { col: ColType::Witness, commit_id: None, id: im_column_id(&cs, k), offset: 0 };
        assert_eq!(d.nodes[a as usize], want, "im root {k}");
        assert!(matches!(d.nodes[b as usize], DagNode::Mul { .. }), "im root {k}");
    }
}

#[test]
fn hash_consing_sorts_commutative() {
    let (a, b) = (witness(0, 0), witness(1, 0));
    let e = vec![
        mul(a.clone(), b.clone()),
        mul(b.clone(), a.clone()),
        add(a.clone(), b.clone()),
        add(b.clone(), a.clone()),
        sub(a.clone(), b.clone()),
        sub(b, a),
    ];
    let d = dag_of(e, &[0, 1, 2, 3, 4, 5], &[]);
    let r: Vec<u32> = d.roots.iter().map(|r| r.node).collect();
    assert_eq!((r[0], r[2]), (r[1], r[3]), "a*b == b*a and a+b == b+a");
    assert_ne!(r[4], r[5], "sub is not reordered");
    assert_eq!(
        d.nodes,
        vec![
            cm(0, 0),
            cm(1, 0),
            DagNode::Mul { a: 0, b: 1 },
            DagNode::Add { a: 0, b: 1 },
            DagNode::Sub { a: 0, b: 1 },
            DagNode::Sub { a: 1, b: 0 },
        ]
    );
}

#[test]
fn nodes_in_post_order_first_appearance() {
    // 1: c + a*b (left operand first), 2: (a*b)*c reuses a*b and gets the next node
    let e = vec![mul(witness(0, 0), witness(1, 0)), add(witness(2, 0), exp(0)), mul(exp(0), witness(2, 0))];
    let d = dag_of(e, &[1, 2], &[]);
    assert_eq!(
        d.nodes,
        vec![
            cm(2, 0),
            cm(0, 0),
            cm(1, 0),
            DagNode::Mul { a: 1, b: 2 },
            DagNode::Add { a: 0, b: 3 },
            DagNode::Mul { a: 0, b: 3 },
        ]
    );
    assert_eq!(d.roots.iter().map(|r| r.node).collect::<Vec<_>>(), vec![4, 5]);
}

/// The searched promoted set, or none when the search is infeasible.
fn im_for(cs: &ConstraintSystem) -> ImChoice {
    match search_im(cs, &DegreePolicy::FromBlowup { blowup_bits: 3 }, 1) {
        Ok(set) => im_order(&set).unwrap(),
        Err(_) => choice(&[]),
    }
}

#[test]
fn build_is_deterministic() {
    for seed in 0..20 {
        let (p, cs) = random_pilout(seed);
        let im = im_for(&cs);
        let first = build_dag(&cs, &im).unwrap();
        let again = build_dag(&cs_of(&p), &im).unwrap();
        assert_eq!(first, again, "seed {seed}");
        assert_eq!(serde_json::to_string(&first).unwrap(), serde_json::to_string(&again).unwrap(), "seed {seed}");
    }
}

#[test]
fn every_dag_validates() {
    for seed in 0..20 {
        let (_, cs) = random_pilout(seed);
        for im in [choice(&[]), im_for(&cs)] {
            let d = build_dag(&cs, &im).unwrap();
            d.validate(&gl()).unwrap_or_else(|e| panic!("seed {seed}: {e:?}"));
            assert_eq!(d.roots.len(), cs.constraints.len() + im.exprs.len(), "seed {seed}");
        }
    }
}

#[test]
fn every_dag_validates_on_real_pilouts() {
    let mut airs = 0;
    for (f, p) in real_pilouts() {
        for (g, ag) in p.air_groups.iter().enumerate() {
            for a in 0..ag.airs.len() {
                let Some(run) = old_run_any(&p, g, a) else { continue };
                let d = build_dag(&run.cs, &run.im).unwrap_or_else(|e| panic!("{f:?} {g}/{a}: {e:?}"));
                d.validate(&gl()).unwrap_or_else(|e| panic!("{f:?} {g}/{a}: {e:?}"));
                assert_eq!(d.roots.len(), run.cs.constraints.len() + run.im.exprs.len());
                airs += 1;
            }
        }
    }
    assert!(airs > 0, "no air built");
}

/// 64 levels of `x_{k+1} = x_k + x_k`: inlining would be 2^64 nodes, the memoised walk is linear.
#[test]
fn doubling_chain_is_not_inlined() {
    let mut e = vec![mul(witness(0, 0), witness(1, 0))];
    for k in 0..64 {
        e.push(add(exp(k), exp(k)));
    }
    e.push(mul(exp(64), exp(0)));
    let cs = system(e, &[65]);
    for im in [choice(&[]), choice(&[32]), choice(&[64])] {
        let d = build_dag(&cs, &im).unwrap();
        assert!(d.nodes.len() < 200, "{} nodes", d.nodes.len());
        d.validate(&gl()).unwrap();
    }
}

// ---- lowering ---------------------------------------------------------------------------------

fn ctx() -> LowerCtx {
    LowerCtx::new(tables()).unwrap()
}

fn tables() -> LowerTables {
    let ev = |col, commit_id, id, offset| ColRef { col, commit_id, id, offset };
    LowerTables {
        ev_map: vec![
            ev(ColType::Witness, None, 0, 0),
            ev(ColType::Witness, None, 0, 1),
            ev(ColType::Witness, None, 3, 0),
            ev(ColType::Fixed, None, 1, 0),
            ev(ColType::Custom, Some(2), 0, 0),
        ],
        challenges: vec![(2, 0), (2, 1), (3, 0)],
        vc: 2,
        n_publics: 5,
        air_values_stage: vec![1, 2],
        proof_values_stage: vec![1, 2],
        airgroup_values_stage: vec![2],
        ext_dim: 3,
    }
}

fn dag(nodes: Vec<DagNode>, roots: &[u32]) -> ConstraintDag {
    let roots = roots
        .iter()
        .map(|&node| pil_info::core::dag::format::DagRoot {
            node,
            boundary: DagBoundary::EveryRow,
            label: format!("r{node}"),
        })
        .collect();
    ConstraintDag { version: 1, field: "gl".into(), ext_dim: 3, nodes, roots }
}

/// Lowered, then in the qVerifier JSON vocabulary.
fn lower(d: &ConstraintDag) -> ExpressionCodeEntry {
    expression_code(&lower_to_code(d, &ctx(), Fold::Stark).unwrap())
}

/// (type, id) of every operand, for compact assertions.
fn refs(e: &ExpressionCodeEntry, i: usize) -> Vec<(String, usize)> {
    e.code[i].src.iter().map(|r| (r.ref_type.clone(), r.id)).collect()
}
fn r(t: &str, id: usize) -> (String, usize) {
    (t.to_string(), id)
}

#[test]
fn one_instruction_per_op_node_then_fold() {
    // ops: 2 add, 4 mul, 5 neg (tmp0, tmp1, tmp2); roots are nodes 2 and 5
    let d = dag(
        vec![
            cm(0, 0),                    // 0
            cm(0, 1),                    // 1
            DagNode::Add { a: 0, b: 1 }, // 2
            cm(3, 0),                    // 3
            DagNode::Mul { a: 2, b: 3 }, // 4
            DagNode::Neg { a: 4 },       // 5
        ],
        &[2, 5],
    );
    d.validate(&gl()).unwrap();
    let e = lower(&d);
    let ops: Vec<&str> = e.code.iter().map(|c| c.op.as_str()).collect();
    assert_eq!(ops, ["add", "mul", "sub", "mul", "add", "mul"]);
    assert!(e.code.iter().enumerate().all(|(k, c)| c.dest.ref_type == "tmp" && c.dest.id == k));
    assert_eq!(e.tmp_used, 6);
    assert_eq!(refs(&e, 0), [r("eval", 0), r("eval", 1)]);
    assert_eq!(refs(&e, 1), [r("tmp", 0), r("eval", 2)]);
    assert_eq!(refs(&e, 3), [r("challenge", 2), r("tmp", 0)], "vc * c_0");
    assert_eq!(refs(&e, 4), [r("tmp", 3), r("tmp", 2)], "+ c_1");
    assert_eq!(refs(&e, 5), [r("tmp", 4), r("Zi", 0)], "times Zi");
}

#[test]
fn fold_is_horner_in_root_order() {
    // roots in the order 4, 2, 3: not node order
    let d = dag(
        vec![
            cm(0, 0),
            cm(0, 1),
            DagNode::Add { a: 0, b: 1 }, // 2 -> tmp0
            DagNode::Mul { a: 0, b: 1 }, // 3 -> tmp1
            DagNode::Sub { a: 0, b: 1 }, // 4 -> tmp2
        ],
        &[4, 2, 3],
    );
    let e = lower(&d);
    let tail: Vec<&str> = e.code[3..].iter().map(|c| c.op.as_str()).collect();
    assert_eq!(tail, ["mul", "add", "mul", "add", "mul"]);
    assert_eq!(refs(&e, 3), [r("challenge", 2), r("tmp", 2)]); // vc * c_0
    assert_eq!(refs(&e, 4), [r("tmp", 3), r("tmp", 0)]); // + c_1
    assert_eq!(refs(&e, 5), [r("challenge", 2), r("tmp", 4)]); // vc * acc
    assert_eq!(refs(&e, 6), [r("tmp", 5), r("tmp", 1)]); // + c_2
    assert_eq!(refs(&e, 7), [r("tmp", 6), r("Zi", 0)]);
    assert_eq!(e.tmp_used, 8);
}

#[test]
fn leaf_root_emits_copy() {
    let e = lower(&dag(vec![cm(0, 0)], &[0]));
    let copy = e.code.iter().find(|c| c.op == "copy").expect("a copy of the leaf into a temporary");
    assert_eq!(copy.dest.ref_type, "tmp");
    assert_eq!((copy.src[0].ref_type.as_str(), copy.src[0].id), ("eval", 0));
    assert!(e.code.iter().all(|c| c.dest.ref_type == "tmp"));
    let last = e.code.last().unwrap();
    assert_eq!((last.op.as_str(), last.src[1].ref_type.as_str()), ("mul", "Zi"));
}

#[test]
fn neg_lowers_to_sub_from_zero() {
    let e = lower(&dag(vec![cm(0, 0), DagNode::Neg { a: 0 }], &[1]));
    let first = &e.code[0];
    assert_eq!(first.op, "sub");
    assert_eq!((first.src[0].ref_type.as_str(), first.src[0].value.as_deref()), ("number", Some("0")));
    assert_eq!((first.src[1].ref_type.as_str(), first.src[1].id), ("eval", 0));
}

#[test]
fn col_resolves_to_ev_index() {
    let c = ctx();
    assert_eq!(c.ev_index(&ColRef { col: ColType::Witness, commit_id: None, id: 3, offset: 0 }), Some(2));
    assert_eq!(c.ev_index(&ColRef { col: ColType::Witness, commit_id: None, id: 0, offset: 1 }), Some(1));
    assert_eq!(c.ev_index(&ColRef { col: ColType::Fixed, commit_id: None, id: 1, offset: 0 }), Some(3));
    assert_eq!(c.ev_index(&ColRef { col: ColType::Custom, commit_id: Some(2), id: 0, offset: 0 }), Some(4));
    assert_eq!(c.ev_index(&ColRef { col: ColType::Witness, commit_id: None, id: 3, offset: 1 }), None);
    assert_eq!(c.ev_index(&ColRef { col: ColType::Custom, commit_id: Some(1), id: 0, offset: 0 }), None);
    assert_eq!(c.challenge_index(3, 0), Some(2));
    assert_eq!(c.challenge_index(2, 1), Some(1));
    assert_eq!(c.challenge_index(9, 9), None);
    // the lowered operand is the evaluation, with no prime left
    let cols = vec![
        cm(3, 0),
        DagNode::Col { col: ColType::Fixed, commit_id: None, id: 1, offset: 0 },
        DagNode::Col { col: ColType::Custom, commit_id: Some(2), id: 0, offset: 0 },
        DagNode::Add { a: 0, b: 1 },
        DagNode::Add { a: 2, b: 3 },
    ];
    let e = lower(&dag(cols, &[4]));
    assert_eq!(refs(&e, 0), [r("eval", 2), r("eval", 3)]);
    assert_eq!(refs(&e, 1), [r("eval", 4), r("tmp", 0)]);
    assert!(e.code[0].src.iter().all(|s| s.prime.is_none()));
}

#[test]
fn leaf_references_use_the_old_vocabulary() {
    let d = dag(
        vec![
            DagNode::Const { value: "12345".into() },     // 0
            DagNode::Challenge { stage: 2, stage_id: 1 }, // 1
            DagNode::Public { id: 4 },                    // 2
            DagNode::ProofValue { id: 1 },                // 3
            DagNode::AirValue { id: 0 },                  // 4
            DagNode::AirgroupValue { id: 0 },             // 5
            DagNode::Add { a: 0, b: 1 },                  // 6
            DagNode::Add { a: 2, b: 3 },                  // 7
            DagNode::Add { a: 4, b: 5 },                  // 8
        ],
        &[6, 7, 8],
    );
    let e = lower(&d);
    assert_eq!(refs(&e, 0), [r("number", 0), r("challenge", 1)]);
    assert_eq!(e.code[0].src[0].value.as_deref(), Some("12345"));
    assert_eq!(refs(&e, 1), [r("public", 4), r("proofvalue", 1)]);
    assert_eq!(refs(&e, 2), [r("airvalue", 0), r("airgroupvalue", 0)]);
}

#[test]
fn unresolved_col_is_error() {
    let unresolved = |d: ConstraintDag| match lower_to_code(&d, &ctx(), Fold::Stark) {
        Err(DagError::Unresolved { node, .. }) => node,
        other => panic!("expected Unresolved, got {other:?}"),
    };
    assert_eq!(unresolved(dag(vec![cm(9, 0)], &[0])), 0, "a column outside the evMap");
    assert_eq!(unresolved(dag(vec![cm(3, 1)], &[0])), 0, "a column at an offset outside the evMap");
    assert_eq!(
        unresolved(dag(
            vec![cm(0, 0), DagNode::Challenge { stage: 7, stage_id: 7 }, DagNode::Add { a: 0, b: 1 }],
            &[2]
        )),
        1
    );
    assert_eq!(unresolved(dag(vec![DagNode::AirValue { id: 5 }], &[0])), 0, "a value past its map");
    assert_eq!(unresolved(dag(vec![DagNode::ProofValue { id: 2 }], &[0])), 0);
    assert_eq!(unresolved(dag(vec![DagNode::AirgroupValue { id: 1 }], &[0])), 0);
    assert_eq!(unresolved(dag(vec![DagNode::Public { id: 5 }], &[0])), 0, "a public past nPublics");
}

#[test]
fn ctx_rejects_repeated_keys() {
    let mut t = tables();
    t.ev_map.push(t.ev_map[1]);
    assert!(matches!(LowerCtx::new(t), Err(DagError::DuplicateKey { .. })), "a repeated evMap entry");
    let mut t = tables();
    t.challenges.push((2, 1));
    assert!(matches!(LowerCtx::new(t), Err(DagError::DuplicateKey { .. })), "a repeated challenge");
}

#[test]
fn dims_follow_the_rule() {
    let c = ctx();
    let dims: Vec<usize> = [
        DagNode::Const { value: "1".into() },
        DagNode::Public { id: 0 },
        cm(0, 0),
        DagNode::Col { col: ColType::Fixed, commit_id: None, id: 1, offset: 0 },
        DagNode::Challenge { stage: 2, stage_id: 0 },
        DagNode::ProofValue { id: 0 }, // stage 1
        DagNode::ProofValue { id: 1 }, // stage 2
        DagNode::AirValue { id: 0 },
        DagNode::AirValue { id: 1 },
        DagNode::AirgroupValue { id: 0 }, // stage 2
    ]
    .iter()
    .map(|n| c.dim(n))
    .collect();
    assert_eq!(dims, [1, 1, 3, 3, 3, 1, 3, 1, 3, 3]);
    // an operation takes the larger of its operands
    let e = lower(&dag(
        vec![
            DagNode::Const { value: "5".into() },
            DagNode::Public { id: 0 },
            cm(0, 0),
            DagNode::Add { a: 0, b: 1 }, // base + base
            DagNode::Mul { a: 1, b: 2 }, // base * ext
        ],
        &[3, 4],
    ));
    assert_eq!(e.code[0].dest.dim, 1);
    assert!(e.code[0].src.iter().all(|s| s.dim == 1));
    assert_eq!(e.code[1].dest.dim, 3);
}
