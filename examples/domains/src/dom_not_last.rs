// a' === a + 1 domain !LAST: a = i + 5 breaks the wrap on row N-1, which is outside !LAST.
crate::domain_component!(DomNotLast, DomNotLastTrace, "DomNotLast", [a], |t| {
    for i in 0..t.num_rows() {
        t[i].a = F::from_u64(i as u64 + 5);
    }
});
