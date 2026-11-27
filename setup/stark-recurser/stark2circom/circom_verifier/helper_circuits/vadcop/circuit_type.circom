pragma circom 2.1.0;

// The circuit type of every proof in the recursion chain: 0 is an empty slot, slot 0 is
// the airgroup's recursive2 and air `airId` is slot airId + 1.
// Mirrors proofman_common::circuit_type_of (Szudzik pairing + 1).
function circuitTypeOf(airgroupId, slot) {
    var pair;
    if (airgroupId < slot) {
        pair = slot * slot + airgroupId;
    } else {
        pair = airgroupId * airgroupId + airgroupId + slot;
    }
    return pair + 1;
}
