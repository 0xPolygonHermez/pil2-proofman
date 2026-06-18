//! Pipeline test for the `connection` PIL.

#[test]
fn pipeline() {
    common::run_pipeline("connection", "connection.pil", "connection").expect("connection pipeline failed");
}
