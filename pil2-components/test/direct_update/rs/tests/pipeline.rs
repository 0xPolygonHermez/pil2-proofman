//! Pipeline test for the `direct_update` PIL.

#[test]
fn pipeline() {
    common::run_pipeline("direct_update", "direct_update.pil", "direct_update").expect("direct_update pipeline failed");
}
