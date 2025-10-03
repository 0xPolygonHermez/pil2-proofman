#[cfg(test)]
mod tests {
    use proofman_macros::{trace, packed_trace};
    use proofman_common as common;

    trace!(SampleTrace<F> { field1: F, field2: [F; 2] },  0, 0, 32 );

    packed_trace!(SimpleTest<F> { a: u32, b: ubit(8) }, 0, 0, 100);
}
