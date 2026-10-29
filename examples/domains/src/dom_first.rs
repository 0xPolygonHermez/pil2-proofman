// a === 1 domain FIRST: rows past 0 are free.
crate::domain_component!(DomFirst, DomFirstTrace, "DomFirst", [a], |t| {
    for i in 0..t.num_rows() {
        t[i].a = F::from_u64(i as u64 + 1);
    }
});
