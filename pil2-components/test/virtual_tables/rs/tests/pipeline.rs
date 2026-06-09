//! Pipeline test for the `virtual_tables` PIL.

#[test]
fn pipeline() {
    common::run_pipeline("virtual_tables", "virtual_tables.pil", "virtual_tables")
        .expect("virtual_tables pipeline failed");
}
