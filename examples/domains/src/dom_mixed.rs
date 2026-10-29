// x^3 === 8 on FIRST, x^3 === w on !FIRST (w[0] = 0 shows row 0 is excluded), e and f zero on even rows,
// v === x + w everywhere.
crate::domain_component!(DomMixed, DomMixedTrace, "DomMixed", [x, w, e, f, v], |t| {
    for i in 0..t.num_rows() {
        let x = if i == 0 { 2 } else { i as u64 + 3 };
        let w = if i == 0 { 0 } else { x * x * x };
        t[i].x = F::from_u64(x);
        t[i].w = F::from_u64(w);
        t[i].e = F::from_u64(if i % 2 == 0 { 0 } else { 1 });
        t[i].f = F::from_u64(if i % 2 == 0 { 0 } else { 2 });
        t[i].v = F::from_u64(x + w);
    }
});
