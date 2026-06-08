//! Pipeline test for the `simple` PIL.

#[test]
fn pipeline() {
    common::run_pipeline("simple", "simple.pil", "simple").expect("simple pipeline failed");
}
