mod trace2;

#[cfg(test)]
mod tests {
    use proofman_macros::trace;
    use proofman_common as common;

    trace!(SampleTrace<F> { field1: F, field2: [F; 2] },  0, 0, 32 );

}
