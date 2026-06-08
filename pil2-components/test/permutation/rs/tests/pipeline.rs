//! Pipeline test for the `permutation` PIL.

#[test]
fn pipeline() {
    common::run_pipeline("permutation", "permutation.pil", "permutation").expect("permutation pipeline failed");
}
