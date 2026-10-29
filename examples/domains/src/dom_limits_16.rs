// s * s * (1 - s) === 0 domain LIMITS_16: s is boolean only on rows 0 and 15 of each 16.
crate::domain_component!(DomLimits16, DomLimits16Trace, "DomLimits16", [s], |t| {
    for i in 0..t.num_rows() {
        t[i].s = F::from_u64(match i % 16 {
            0 => 0,
            15 => 1,
            _ => 3,
        });
    }
});
