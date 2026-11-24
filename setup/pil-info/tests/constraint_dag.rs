//! pilfflonk's `qVerifier` (Opening::Shplonk, BN128) is the lowered constraint DAG:
//! `expression_code(lower_to_code(dag, ctx, Fold::Shplonk{..}))`. Its value is guarded by pilfflonk's oracle tests.

use pil2_pilout::pilout as pb;
use pil_info::core::dag::build::{build_dag, dag_boundary};
use pil_info::core::dag::format::{ConstraintDag, DagBoundary};
use pil_info::core::dag::lower_code::{lower_to_code, Fold, LowerCtx};
use pil_info::core::from_pilout::from_pilout;
use pil_info::core::im::{im_order, ImChoice};
use pil_info::core::layout::{layout, LayoutParams};
use pil_info::core::openings::openings;
use pil_info::emit::verifier::{expression_code, lower_ctx};
use pil_info::search::im::search_im;
use pil_info::{PilInfoCfg, PilInfoResult};

mod common;
use common::*;

/// a*b + a' (everyRow), L1*(a - 1) (firstRow), b - a (lastRow), a' - b (everyFrame 0..1): four boundaries.
fn mixed_boundaries_pilout() -> pb::PilOut {
    pilout(
        &["a", "b"],
        &["L1"],
        vec![
            sub(witness(0, 0), constant("1")), // 0
            mul(fixed(0), exp(0)),             // 1
            mul(witness(0, 0), witness(1, 0)), // 2
            add(exp(2), witness(0, 1)),        // 3
            sub(witness(1, 0), witness(0, 0)), // 4
            sub(witness(0, 1), witness(1, 0)), // 5
        ],
        vec![first_row(1), every_row(3), last_row(4), every_frame(5, 0, 1)],
    )
}

fn cfg() -> PilInfoCfg {
    PilInfoCfg::bn128()
}

fn run(p: &pb::PilOut) -> PilInfoResult {
    pil_info::shplonk_info(p, 0, 0, &cfg(), &Default::default()).unwrap()
}

fn boundaries(r: &PilInfoResult) -> Vec<DagBoundary> {
    r.boundaries.iter().map(dag_boundary).collect()
}

/// The core's DAG and lowering context of the air, with the search's im choice.
fn dag_and_ctx(p: &pb::PilOut, r: &PilInfoResult) -> (ConstraintDag, LowerCtx) {
    let c = cfg();
    let cs = from_pilout(p, 0, 0, &c.field, c.opening).unwrap();
    let set = search_im(&cs, &c.degree_policy, 1).unwrap();
    let im: ImChoice = im_order(&cs, &set).unwrap();
    let params =
        LayoutParams { im_pols_stages: false, deep_challenges: false, q_deg: r.q_deg as u32, air_name: String::new() };
    let lay = layout(&cs, &im, &params).unwrap();
    let ops = openings(&cs, &im, &lay).unwrap();
    (build_dag(&cs, &im).unwrap(), lower_ctx(&cs, &lay, &ops).unwrap())
}

#[test]
fn shplonk_q_verifier_is_lowered_dag() {
    let p = mixed_boundaries_pilout();
    let r = run(&p);
    let b = boundaries(&r);
    assert_eq!(b.len(), 4);
    let want = expression_code(
        &{
            let (d, x) = dag_and_ctx(&p, &r);
            lower_to_code(&d, &x, Fold::Shplonk { boundaries: &b })
        }
        .unwrap(),
    );
    let got = r.pil_code.verifier_info.q_verifier.as_ref().expect("a qVerifier");
    assert_eq!(got.tmp_used, want.tmp_used);
    assert_eq!(serde_json::to_value(&got.code).unwrap(), serde_json::to_value(&want.code).unwrap());
}

// Dimension corruption: a context whose extension degree disagrees with the (BN128, ext 1) DAG.
#[test]
fn shplonk_refuses_dimension_mismatch() {
    let p = mixed_boundaries_pilout();
    let r = run(&p);
    let b = boundaries(&r);
    let (dag, ctx) = dag_and_ctx(&p, &r);
    let mut t = ctx.tables().clone();
    t.ext_dim = 3;
    let ctx = LowerCtx::new(t).unwrap();
    assert!(lower_to_code(&dag, &ctx, Fold::Shplonk { boundaries: &b }).is_err());
}

// A root whose boundary is not in the list has no Zi.
#[test]
fn shplonk_refuses_an_unlisted_boundary() {
    let p = mixed_boundaries_pilout();
    let r = run(&p);
    let only_every_row = [DagBoundary::EveryRow];
    let (dag, ctx) = dag_and_ctx(&p, &r);
    assert!(lower_to_code(&dag, &ctx, Fold::Shplonk { boundaries: &only_every_row }).is_err());
}
