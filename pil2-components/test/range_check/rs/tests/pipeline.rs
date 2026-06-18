//! Pipeline test for the range_check PIL.

#[test]
fn pipeline() {
    common::run_pipeline("range_check", "build.pil", "range_check").expect("range_check pipeline failed");
}
