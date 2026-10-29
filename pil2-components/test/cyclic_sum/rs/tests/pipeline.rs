//! Pipeline test for the `cyclic_sum` PIL.

#[test]
fn pipeline() {
    common::run_pipeline("cyclic_sum", "cyclic_sum.pil", "cyclic_sum").expect("cyclic_sum pipeline failed");
}
