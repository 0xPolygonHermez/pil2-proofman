//! The calldata of the Solidity verifier (spec §4.5, "Calldata"; plan M41): the arguments of its
//! `verifyProof` for a proof, which `proofman-cli pilfflonk calldata` prints as snarkjs's `zkey
//! export soliditycalldata` prints those of its FFLONK verifier.
//!
//! The verifier that `setup-pilfflonk --solidity` generates from the vkey (`pilfflonk-setup`,
//! `solidity.rs`) has the interface of snarkjs's `FflonkVerifier`,
//! `verifyProof(bytes32[W] calldata proof, uint256[P] calldata pubSignals)`, with `P = nPublic` (no
//! `pubSignals` if it is 0) and `W` the words of [`CalldataLayout`]:
//! - the proof's bytes (A.6, [`Proof::to_bytes`]) as 32-byte words: the commitments of the
//!   non-fixed `f`, `W`, `W'`, the evaluations, the `Q_i(ξ)` if `Q` is split, `inv` and `invZh`;
//! - one auxiliary inverse `1/(ξ − ω^j)` per `firstRow` (`j = 0`) or `lastRow` (`j = N − 1`)
//!   boundary, in the order of the boundaries. The `Zi` of those boundaries is `Z_H(ξ)/(ξ − ω^j)`
//!   (A.1), the only division of the verifier that neither `inv` nor `invZh` covers; the contract
//!   checks `(ξ − ω^j)·aux = 1` and `aux < r`, so that a proof has one calldata. They are of the
//!   calldata only: the proof's format does not change, and a vkey without those boundaries, as
//!   that of every compiled PIL2 program (§3.4), has none.
//!
//! `ξ = xiSeed^powerW`, and `xiSeed` is the transcript's (A.4): [`verifier_challenges`] replays the
//! transcript on the proof as the verifier does (`pilfflonk/js/src/challenges.js`,
//! `computeChallenges`), with the transcript of the C++ core, the prover's.
//!
//! [`Calldata::read`] takes the files `pilfflonk verify` takes, and [`Calldata::encode`] their
//! values; [`Calldata::to_solidity`] and [`Calldata::to_hex`] write the result.

use std::path::Path;

use proofman_fields::{Bn254, Field};
use proofman_starks_lib_c::{pilfflonk_keccak256_c, PilFflonkErrorKind};

use crate::error::{invalid, PilfflonkError, PilfflonkResult};
use crate::field::{FrBytes, G1Affine, FIELD_BYTES};
use crate::json::JsonFile;
use crate::names::{commitment_name, W};
use crate::pilfflonk_info::Boundary;
use crate::proof::{Proof, ProofNames, Publics};
use crate::prover::{native, Transcript};
use crate::vkey::Vkey;

/// A word of the calldata: 32 bytes, big-endian.
pub type Word = [u8; FIELD_BYTES];

/// Bytes of the selector of `verifyProof`, before its arguments.
pub const SELECTOR_BYTES: usize = 4;

/// Where the verifier finds each value in its `proof` argument, in words (see the module).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalldataLayout {
    /// The commitments of the non-fixed `f_i`, `x` and `y` each a word, from word 0.
    pub n_commitments: u64,
    /// The evaluations of the evMap, in the order of the proof: the fixed columns' first (A.4,
    /// step 4; A.6).
    pub n_evaluations: u64,
    /// The pieces `Q_i(ξ)` of a split `Q`, and 0 if it is whole.
    pub n_q_pieces: u64,
    /// The rows `j` of the auxiliary inverses `1/(ξ − ω^j)` after the proof: one per `firstRow`
    /// (`0`) and `lastRow` (`N − 1`) boundary, in the order of the boundaries.
    pub aux_rows: Vec<u64>,
}

impl CalldataLayout {
    pub fn of(vkey: &Vkey) -> Self {
        let n_fixed = vkey.layout.n_fixed() as u64;
        let n_rows = 1u64 << vkey.power;
        let q_stage = vkey.layout.0.last().map_or(0, |f| f.stage);
        let n_q = vkey.layout.0.iter().filter(|f| f.stage == q_stage).map(|f| f.k).sum::<u64>();
        let aux_rows = vkey
            .boundaries
            .iter()
            .filter_map(|b| match b {
                Boundary::FirstRow => Some(0),
                Boundary::LastRow => Some(n_rows - 1),
                Boundary::EveryRow | Boundary::EveryFrame { .. } => None,
            })
            .collect();
        CalldataLayout {
            n_commitments: vkey.layout.0.len() as u64 - n_fixed,
            n_evaluations: vkey.ev_map.len() as u64,
            n_q_pieces: if n_q > 1 { n_q } else { 0 },
            aux_rows,
        }
    }

    /// The words of the proof's bytes (A.6): the commitments, `W`, `W'`, the evaluations, the
    /// pieces of `Q`, `inv` and `invZh`. Format version 1 has no air, airgroup or proof values.
    pub fn proof_words(&self) -> u64 {
        2 * (self.n_commitments + 2) + self.n_evaluations + self.n_q_pieces + 2
    }

    /// The words of the `proof` argument: the proof's and the auxiliary inverses.
    pub fn words(&self) -> u64 {
        self.proof_words() + self.aux_rows.len() as u64
    }

    /// The word of the first scalar, the first evaluation (or, with none, the first piece of `Q`).
    pub fn first_scalar(&self) -> u64 {
        2 * (self.n_commitments + 2)
    }
}

/// The challenges of a proof as the verifier replays them (A.4; `challenges.js`,
/// `computeChallenges`). Those the prover also gives ([`ProofChallenges`](crate::ProofChallenges))
/// are the same for its proofs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierChallenges {
    /// `stages[s - 2]`: the challenges of stage `s`, for `2 ≤ s ≤ nStages`.
    pub stages: Vec<Vec<FrBytes>>,
    pub std_vc: FrBytes,
    pub xi_seed: FrBytes,
    /// SHPLONK's `α_S` and `y` (A.5).
    pub alpha: FrBytes,
    pub y: FrBytes,
}

/// Refuses a proof and publics that are not of the shape of `vkey`'s: exactly the values the vkey
/// names ([`ProofNames::of_vkey`]), and `nPublic` publics.
fn check_shape(vkey: &Vkey, proof: &Proof, publics: &[FrBytes]) -> PilfflonkResult<()> {
    let shape = ProofNames::of_vkey(vkey)?.shape();
    if proof.shape() != shape {
        return invalid!("the proof has the shape {:?}, and a proof of the vkey {shape:?}", proof.shape());
    }
    if publics.len() as u64 != vkey.n_public {
        return invalid!("publics: {} of them, and the vkey has nPublic = {}", publics.len(), vkey.n_public);
    }
    Ok(())
}

/// Absorbs `point`, the value `name` of the proof. The transcript refuses a point that is not on
/// the curve, the point at infinity and a point with a coordinate below 2^192 (A.4), as the JS
/// verifier's does: then the verifier rejects the proof, and its challenges cannot be replayed.
fn absorb_point(transcript: &mut Transcript, point: &G1Affine, name: &str) -> PilfflonkResult<()> {
    transcript.points(std::slice::from_ref(point)).map_err(|e| match e {
        PilfflonkError::Native { source, .. }
            if matches!(source.kind, PilFflonkErrorKind::InvalidPoint | PilFflonkErrorKind::NonCanonical) =>
        {
            PilfflonkError::InvalidFormat(format!(
                "{name} of the proof is not a point the transcript absorbs (A.4), so its challenges cannot be \
                 replayed and the verifier rejects it: {}",
                source.message
            ))
        }
        other => other,
    })
}

/// The challenges of `proof` and `publics`, a proof of `vkey`, as the verifier replays them (see
/// [`VerifierChallenges`]):
///
/// 1. absorb `digest mod r`, the number of instances of the AIR (1) and the publics;
/// 2. for each stage `s = 1 … nStages`: absorb the commitments of its `f`, in the global order of
///    A.5, and, if `s < nStages`, squeeze the `numChallenges[s]` challenges of stage `s + 1`;
/// 3. squeeze `std_vc`; absorb the commitments of `Q`; squeeze `xiSeed`;
/// 4. absorb the evaluations and the pieces of `Q`, in the order of the proof; squeeze `α_S`;
/// 5. absorb `[W]₁`; squeeze `y`.
///
/// Refuses a proof or publics not of the vkey's shape, and a proof with a point the transcript
/// refuses (A.4: off the curve, the point at infinity, or with a coordinate below 2^192), which it
/// names.
pub fn verifier_challenges(vkey: &Vkey, proof: &Proof, publics: &[FrBytes]) -> PilfflonkResult<VerifierChallenges> {
    check_shape(vkey, proof, publics)?;
    let layout = &vkey.layout.0;
    let n_fixed = vkey.layout.n_fixed();
    let q_stage = layout.last().map_or(0, |f| f.stage);
    let n_stages = q_stage.saturating_sub(1);

    // Step 1.
    let mut t = Transcript::new()?;
    t.scalars(&[vkey.digest.to_fr(), FrBytes::from_u64(1)])?;
    t.scalars(publics)?;
    let absorb_stage = |t: &mut Transcript, stage: u64| -> PilfflonkResult<()> {
        for (i, (point, f)) in proof.commitments.iter().zip(&layout[n_fixed..]).enumerate() {
            if f.stage == stage {
                absorb_point(t, point, &commitment_name((n_fixed + i) as u64))?;
            }
        }
        Ok(())
    };

    // Step 2.
    let mut stages = Vec::new();
    for s in 1..=n_stages {
        absorb_stage(&mut t, s)?;
        if s < n_stages {
            let count = vkey.num_challenges.get(s as usize).copied().unwrap_or(0);
            stages.push((0..count).map(|_| t.squeeze()).collect::<PilfflonkResult<Vec<_>>>()?);
        }
    }

    // Step 3.
    let std_vc = t.squeeze()?;
    absorb_stage(&mut t, q_stage)?;
    let xi_seed = t.squeeze()?;

    // Step 4: Proof::evaluations holds the pieces of Q after the evMap's, in the order of the layout.
    t.scalars(&proof.evaluations)?;
    let alpha = t.squeeze()?;

    // Step 5.
    absorb_point(&mut t, &proof.w, W)?;
    let y = t.squeeze()?;
    Ok(VerifierChallenges { stages, std_vc, xi_seed, alpha, y })
}

/// The auxiliary inverses of the calldata of a proof of `vkey` whose `xiSeed` is `xi_seed`:
/// `1/(ξ − ω^j)` for each row `j` of [`CalldataLayout::aux_rows`], with `ξ = xiSeed^powerW` and `ω`
/// the root of unity of the domain, `5^((r − 1)/N)` (`Bn254::W`, `shplonk.js`'s `rootOfUnity`).
/// Refuses a `ξ` that is a row of the domain: then `Z_H(ξ) = 0`, and the verifier rejects the proof
/// for its `invZh`.
pub fn auxiliary_inverses(vkey: &Vkey, xi_seed: &FrBytes) -> PilfflonkResult<Vec<FrBytes>> {
    let rows = CalldataLayout::of(vkey).aux_rows;
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let Some(omega) = usize::try_from(vkey.power).ok().and_then(|power| Bn254::W.get(power)) else {
        return invalid!("power {} is above the 2-adicity of r − 1, {}", vkey.power, Bn254::TWO_ADICITY);
    };
    let xi = Bn254::from(*xi_seed).exp_u64(vkey.power_w);
    rows.iter()
        .map(|&j| match (xi - omega.exp_u64(j)).try_inverse() {
            Some(inverse) => Ok(FrBytes::from(inverse)),
            None => invalid!(
                "ξ = ω^{j} is a row of the domain: Z_H(ξ) = 0, and the verifier rejects the proof for its invZh"
            ),
        })
        .collect()
}

/// The arguments of `verifyProof` for a proof (see the module).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Calldata {
    /// The words of `proof`: the proof's bytes, then the auxiliary inverses.
    pub proof: Vec<Word>,
    /// `pubSignals`: the publics, in the order of `publics.json`.
    pub pub_signals: Vec<FrBytes>,
}

impl Calldata {
    /// The calldata of the proof at `proof_path` with the publics at `publics_path`, for the vkey at
    /// `vkey_path`, the files `pilfflonk verify` takes: `pilfflonk.vkey.json`, the proof (its JSON
    /// view, `proof.json`, or its bytes in a file whose name ends in `.bin`: [`Proof::read`]) and
    /// `publics.json`. They are read as the JS verifier reads them: the vkey validated, its digest
    /// checked (the verifier is generated for no other vkey), the proof with exactly the values the
    /// vkey names ([`ProofNames::of_vkey`]) and `nPublic` publics; then [`Calldata::encode`].
    pub fn read(vkey_path: &Path, proof_path: &Path, publics_path: &Path) -> PilfflonkResult<Self> {
        let vkey = Vkey::read(vkey_path)?;
        vkey.check_digest().map_err(|e| e.in_file(vkey_path))?;
        let names = ProofNames::of_vkey(&vkey).map_err(|e| e.in_file(vkey_path))?;
        let proof = Proof::read(proof_path, &names)?;
        let Publics(publics) = Publics::read(publics_path)?;
        if publics.len() as u64 != vkey.n_public {
            let message = format!("{} publics, and the vkey has nPublic = {}", publics.len(), vkey.n_public);
            return Err(PilfflonkError::InvalidFormat(message).in_file(publics_path));
        }
        Self::encode(&vkey, &proof, &publics)
    }

    /// The calldata of `proof` and `publics`, a proof of `vkey`: the proof's bytes and the
    /// auxiliary inverses of its own `ξ` ([`verifier_challenges`], [`auxiliary_inverses`]), and the
    /// publics. Refuses a proof or publics not of the vkey's shape, a proof whose transcript cannot
    /// be replayed (a point the transcript refuses, A.4) and, if the calldata has auxiliary inverses,
    /// a `ξ` that is a row of the domain: the verifier rejects each.
    pub fn encode(vkey: &Vkey, proof: &Proof, publics: &[FrBytes]) -> PilfflonkResult<Self> {
        let challenges = verifier_challenges(vkey, proof, publics)?;
        let aux = auxiliary_inverses(vkey, &challenges.xi_seed)?;
        Self::with_auxiliary_inverses(vkey, proof, publics, &aux)
    }

    /// The calldata of `proof` and `publics`, a proof of `vkey`, with the auxiliary inverses `aux`,
    /// whatever they are: for tests of what the verifier does with calldata that [`encode`] does not
    /// give (a proof whose transcript cannot be replayed, a wrong inverse). Refuses a proof,
    /// publics or inverses not of the vkey's shape.
    ///
    /// [`encode`]: Calldata::encode
    pub fn with_auxiliary_inverses(
        vkey: &Vkey,
        proof: &Proof,
        publics: &[FrBytes],
        aux: &[FrBytes],
    ) -> PilfflonkResult<Self> {
        check_shape(vkey, proof, publics)?;
        let layout = CalldataLayout::of(vkey);
        if aux.len() != layout.aux_rows.len() {
            return invalid!(
                "{} auxiliary inverses, and the calldata of the vkey has {}",
                aux.len(),
                layout.aux_rows.len()
            );
        }
        let bytes = proof.to_bytes();
        let mut words: Vec<Word> = bytes
            .chunks_exact(FIELD_BYTES)
            .map(|chunk| {
                let mut word = [0u8; FIELD_BYTES];
                word.copy_from_slice(chunk);
                word
            })
            .collect();
        words.extend(aux.iter().map(FrBytes::to_be_bytes));
        if words.len() as u64 != layout.words() {
            return invalid!("the calldata has {} words, and the vkey's {}", words.len(), layout.words());
        }
        Ok(Calldata { proof: words, pub_signals: publics.to_vec() })
    }

    /// The signature of `verifyProof` of the verifier these arguments are for:
    /// `verifyProof(bytes32[W],uint256[P])`, or `verifyProof(bytes32[W])` if there are no publics.
    pub fn signature(&self) -> String {
        if self.pub_signals.is_empty() {
            format!("verifyProof(bytes32[{}])", self.proof.len())
        } else {
            format!("verifyProof(bytes32[{}],uint256[{}])", self.proof.len(), self.pub_signals.len())
        }
    }

    /// The ABI encoding of the call: the selector, the first 4 bytes of the Keccak-256 of
    /// [`signature`](Calldata::signature), and the arguments, both fixed-size arrays and so encoded
    /// in place: the words of `proof`, then those of `pubSignals`.
    pub fn to_abi_bytes(&self) -> PilfflonkResult<Vec<u8>> {
        let selector = selector(&self.signature())?;
        let mut bytes = Vec::with_capacity(SELECTOR_BYTES + FIELD_BYTES * (self.proof.len() + self.pub_signals.len()));
        bytes.extend_from_slice(&selector);
        for word in self.proof.iter().copied().chain(self.pub_signals.iter().map(FrBytes::to_be_bytes)) {
            bytes.extend_from_slice(&word);
        }
        Ok(bytes)
    }

    /// [`to_abi_bytes`](Calldata::to_abi_bytes) as `0x` and lowercase hexadecimal digits: the `data`
    /// of an `eth_call`, or of `cast call <address> --data <hex>`.
    pub fn to_hex(&self) -> PilfflonkResult<String> {
        Ok(format!("0x{}", hex(&self.to_abi_bytes()?)))
    }

    /// The arguments of `verifyProof` as snarkjs's `zkey export soliditycalldata` prints those of its
    /// FFLONK verifier (`fflonk_export_calldata.js`): each word `0x` and 64 hexadecimal digits,
    /// comma-separated, the words of `proof` in brackets and then, if there are publics, those of
    /// `pubSignals`: `[0x…,0x…],[0x…]`. snarkjs spaces some of its commas; here none is.
    pub fn to_solidity(&self) -> String {
        let array = |words: &mut dyn Iterator<Item = Word>| {
            format!("[{}]", words.map(|word| format!("0x{}", hex(&word))).collect::<Vec<_>>().join(","))
        };
        let proof = array(&mut self.proof.iter().copied());
        if self.pub_signals.is_empty() {
            proof
        } else {
            format!("{proof},{}", array(&mut self.pub_signals.iter().map(FrBytes::to_be_bytes)))
        }
    }
}

/// The selector of a function of this signature: the first 4 bytes of its Keccak-256, as Solidity
/// computes it.
pub fn selector(signature: &str) -> PilfflonkResult<[u8; SELECTOR_BYTES]> {
    let hash = pilfflonk_keccak256_c(signature.as_bytes()).map_err(native("hashing the signature of verifyProof"))?;
    let mut selector = [0u8; SELECTOR_BYTES];
    selector.copy_from_slice(&hash[..SELECTOR_BYTES]);
    Ok(selector)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_selector_is_solidity_s() {
        // ERC-20's transfer and balanceOf, as solc computes them.
        assert_eq!(selector("transfer(address,uint256)").unwrap(), [0xa9, 0x05, 0x9c, 0xbb]);
        assert_eq!(selector("balanceOf(address)").unwrap(), [0x70, 0xa0, 0x82, 0x31]);
    }

    fn word(value: u64) -> Word {
        FrBytes::from_u64(value).to_be_bytes()
    }

    #[test]
    fn the_calldata_is_written_as_snarkjs_writes_it_and_as_the_abi_encodes_it() {
        let calldata = Calldata { proof: vec![word(1), word(0xab)], pub_signals: vec![FrBytes::from_u64(2)] };
        let digits = |value: &str| format!("0x{value:0>64}");
        assert_eq!(calldata.to_solidity(), format!("[{},{}],[{}]", digits("1"), digits("ab"), digits("2")));
        assert_eq!(calldata.signature(), "verifyProof(bytes32[2],uint256[1])");
        let bytes = calldata.to_abi_bytes().unwrap();
        assert_eq!(bytes.len(), 4 + 3 * 32);
        assert_eq!(bytes[..4], selector("verifyProof(bytes32[2],uint256[1])").unwrap());
        assert_eq!((bytes[4 + 31], bytes[4 + 63], bytes[4 + 95]), (1, 0xab, 2));
        assert_eq!(calldata.to_hex().unwrap(), format!("0x{}", hex(&bytes)));

        // No publics: no pubSignals, which Solidity cannot declare as uint256[0].
        let calldata = Calldata { proof: vec![word(7)], pub_signals: vec![] };
        assert_eq!(calldata.to_solidity(), format!("[{}]", digits("7")));
        assert_eq!(calldata.signature(), "verifyProof(bytes32[1])");
        assert_eq!(calldata.to_abi_bytes().unwrap()[..4], selector("verifyProof(bytes32[1])").unwrap());
    }
}
