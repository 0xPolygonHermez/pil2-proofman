//! Pipeline test for the `lookup` PIL.

#[test]
fn pipeline() {
    common::run_pipeline("lookup", "lookup.pil", "lookup").expect("lookup pipeline failed");
}
