//! A compressed proof's wire form: a multiproof when its key says so, the flat proof otherwise.

use proofman_common::{hash_family, ProofmanError, ProofmanResult, StarkInfo};
use proofman_fields::{
    Blake3Transcript, Blake3_8, Goldilocks, Poseidon1_16, Poseidon1_8, Poseidon2_16, Poseidon2_8, Transcript,
};
use proofman_verifier::{expand_proof, is_multiproof, prune_proof, ProofLayout};

fn refuse(reason: &str) -> ProofmanError {
    ProofmanError::InvalidProof(reason.to_string())
}

/// The key's layout, refused where the codec cannot walk its proof.
fn codec_layout(stark_info: &StarkInfo) -> ProofmanResult<ProofLayout> {
    let empty = |m: &Option<Vec<_>>| m.as_ref().is_none_or(|v| v.is_empty());
    if !empty(&stark_info.airgroupvalues_map) || !empty(&stark_info.airvalues_map) {
        return Err(refuse("a multiproof carries no air or airgroup values"));
    }
    if !stark_info.custom_commits.is_empty() {
        return Err(refuse("a multiproof carries no custom commits"));
    }
    Ok(stark_info.proof_layout())
}

type CodecFn = fn(&[u64], &[u64], &ProofLayout) -> Option<Vec<u64>>;

/// `prune_proof` and `expand_proof` over the hashes the key's generated verifier uses
/// (`hash_family::rust_hash_type` / `rust_transcript_type`).
fn codec(hash: &str, stark_info: &StarkInfo) -> ProofmanResult<(CodecFn, CodecFn)> {
    type P1 = Transcript<Goldilocks, Poseidon1_16>;
    type P2 = Transcript<Goldilocks, Poseidon2_16>;
    type B3 = Blake3Transcript<Goldilocks>;
    let ss = &stark_info.stark_struct;
    let (mta, ta) = (ss.merkle_tree_arity, ss.transcript_arity);
    if !hash_family::has_multiproof_codec(hash, mta, ta) {
        return Err(ProofmanError::InvalidProof(format!(
            "no multiproof codec for {hash} at tree arity {mta}, transcript arity {ta}"
        )));
    }
    match hash {
        "blake3" => Ok((prune_proof::<B3>, expand_proof::<Blake3_8, Blake3_8, B3>)),
        "Poseidon1" => Ok((prune_proof::<P1>, expand_proof::<Poseidon1_8, Poseidon1_8, P1>)),
        "Poseidon2" => Ok((prune_proof::<P2>, expand_proof::<Poseidon2_8, Poseidon2_8, P2>)),
        fam => Err(ProofmanError::InvalidProof(format!("has_multiproof_codec lists {fam} but proofman has no types"))),
    }
}

/// The proof (`[n_publics, publics…, proof…]`) as it goes on the wire.
pub fn prune_compressed_proof(
    proof: &[u64],
    stark_info: &StarkInfo,
    verkey: &[u64],
    hash: &str,
) -> ProofmanResult<Vec<u64>> {
    if !stark_info.multiproof {
        return Ok(proof.to_vec());
    }
    let (prune, _) = codec(hash, stark_info)?;
    let pruned = prune(proof, verkey, &codec_layout(stark_info)?)
        .ok_or_else(|| refuse("compressed proof does not have the shape of its key"))?;
    // Never ship a wire proof that does not rebuild the prover's flat proof exactly.
    if expand_compressed_proof(&pruned, stark_info, verkey, hash)? != proof {
        return Err(refuse("pruned compressed proof does not expand back to itself"));
    }
    Ok(pruned)
}

/// The flat proof the verifiers read; a multiproof key takes only tagged proofs, others only flat.
pub fn expand_compressed_proof(
    proof: &[u64],
    stark_info: &StarkInfo,
    verkey: &[u64],
    hash: &str,
) -> ProofmanResult<Vec<u64>> {
    match (stark_info.multiproof, is_multiproof(proof)) {
        (false, false) => Ok(proof.to_vec()),
        (false, true) => Err(refuse("this key's compressed proofs are flat; this one is a multiproof")),
        (true, false) => Err(refuse("this key's compressed proofs are multiproofs; this one is flat")),
        (true, true) => {
            let (_, expand) = codec(hash, stark_info)?;
            expand(proof, verkey, &codec_layout(stark_info)?)
                .ok_or_else(|| refuse("compressed proof is not a well-formed multiproof"))
        }
    }
}
