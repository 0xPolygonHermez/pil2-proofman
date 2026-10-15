// Scratch repro: run bctree on the exact files the recursive setup died on.
//   BCT_CONST=... BCT_SI=... BCT_VK=... cargo test -p pil2-stark-setup --test bctree_repro -- --ignored --nocapture
#[test]
#[ignore]
fn bctree_repro() {
    let c = std::env::var("BCT_CONST").unwrap();
    let s = std::env::var("BCT_SI").unwrap();
    let v = std::env::var("BCT_VK").unwrap();
    let root = pil2_stark_setup::proving_key::bctree::compute_const_tree(&c, &s, &v);
    println!("root = {root:?}");
}
