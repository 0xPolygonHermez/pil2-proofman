//! Pipeline test for the `one_instance` PIL.

#[test]
fn pipeline() {
    common::run_pipeline("one_instance", "one_instance.pil", "one_instance").expect("one_instance pipeline failed");
}
