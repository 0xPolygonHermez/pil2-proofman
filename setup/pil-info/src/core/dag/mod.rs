//! Constraint DAG: the `constraint-dag` crate (format, validation, resolution, lowering) and the kernel's builder.

pub mod build;
pub use constraint_dag::{format, global, lower_code};
