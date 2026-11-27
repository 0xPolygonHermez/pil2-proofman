pragma circom 2.1.0;

/*
    A Merkle node's four children (4 words each) in tree order, for the keyed Poseidon gates: in holds the
    opened child first, then its three siblings; the opened child goes to position key[0] + 2 key[1].
*/
function nodeChildrenInOrder(in, key) {
    var position = key[0] + 2 * key[1];
    var children[16];
    var sibling = 1;
    for (var c = 0; c < 4; c++) {
        for (var w = 0; w < 4; w++) { children[4 * c + w] = c == position ? in[w] : in[4 * sibling + w]; }
        if (c != position) { sibling++; }
    }
    return children;
}
