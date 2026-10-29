// b === 0 domain LAST: b = (i + 1) mod N is 0 only on the last row.
crate::domain_component!(DomLast, DomLastTrace, "DomLast", [b], |t| {
    let n = t.num_rows();
    for i in 0..n {
        t[i].b = F::from_u64(((i + 1) % n) as u64);
    }
});
