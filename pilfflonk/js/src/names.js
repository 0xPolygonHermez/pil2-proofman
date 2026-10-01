// The names of the JSON view of a proof (pilfflonk/docs/formats.md#proof-names), as
// pilfflonk/src/names.rs writes them for a proof of one instance, the only one vkey format 1
// describes (pilfflonk/docs/README.md#scope): no scope prefixes.
//
// polynomials: f<g> for the non-fixed f at position g of the global order (the fixed ones are the
// vkey's, under the same names; pilfflonk/docs/protocol.md#global-order), W and Wp.
// evaluations: <column><suffix> for a column at ξ·ω^s, <column> its name in the layout of the
// vkey and <suffix> "" for s = 0, "w" for s = 1 and "w" and s in decimal, sign included,
// otherwise; the pieces of a split Q by their column names; inv and invZh. The layout's names are
// the setup's, indices included (im pols and columns named alike, `im_cluster[1]`), and are taken
// as they are.

export const W = "W";
export const WP = "Wp";
export const INV = "inv";
export const INV_ZH = "invZh";

export function offsetSuffix(offset) {
    if (offset === 0) return "";
    if (offset === 1) return "w";
    return `w${offset}`;
}

export function evaluationName(column, offset) {
    return `${column}${offsetSuffix(offset)}`;
}

export function commitmentName(globalIndex) {
    return `f${globalIndex}`;
}
