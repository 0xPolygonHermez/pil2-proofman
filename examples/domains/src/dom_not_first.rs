// (a - 'a - 1)^2 === 0 domain !FIRST: a = i + 5 breaks the wrap read by row 0, which is outside !FIRST.
crate::domain_component!(DomNotFirst, DomNotFirstTrace, "DomNotFirst", [a], |t| {
    for i in 0..t.num_rows() {
        t[i].a = F::from_u64(i as u64 + 5);
    }
});
