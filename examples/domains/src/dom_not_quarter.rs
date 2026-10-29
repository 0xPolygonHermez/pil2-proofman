// u === 7 domain !Q1 (Q1 = [1,0,0,0]): u is 7 except on rows 0 mod 4.
crate::domain_component!(DomNotQuarter, DomNotQuarterTrace, "DomNotQuarter", [u], |t| {
    for i in 0..t.num_rows() {
        t[i].u = F::from_u64(if i % 4 == 0 { i as u64 } else { 7 });
    }
});
