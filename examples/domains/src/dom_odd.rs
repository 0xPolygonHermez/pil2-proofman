// t === 0 domain ODD: the even rows hold 3.
crate::domain_component!(DomOdd, DomOddTrace, "DomOdd", [t], |tr| {
    for i in 0..tr.num_rows() {
        tr[i].t = F::from_u64(if i % 2 == 1 { 0 } else { 3 });
    }
});
