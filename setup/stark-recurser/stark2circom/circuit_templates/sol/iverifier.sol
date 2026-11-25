// SPDX-License-Identifier: AGPL-3.0
pragma solidity ^0.8.20;

/// @title __NAME__ Verifier Interface
/// @author SilentSig
/// @notice This contract is the interface for the __NAME__ Verifier.
/// @dev `verifySnarkProof` is the only __NAME__-binding entry point. The deployed
/// `__NAME__Verifier` also carries `verifyProof`, the inherited raw __SNARK_LABEL__ verifier -- public
/// because its assembly reads arguments at fixed calldata offsets -- which proves only
/// that some proof satisfies some digest, with no __NAME__ statement attached. Note the names
/// invert the usual convention: the raw one is the shorter.
interface I__NAME__Verifier {
    /// @notice Verifies a proof with given public values and vkey.
    /// @dev __PIN__ of the calling contract, never __PIN_VALUE__ taken
    /// from the submitted proof, or verification is self-keyed and authenticates
    /// nothing.__FOLD_TREE__
    __PROGRAM_VK_DOC__/// @param rootCVadcopFinal The recursion root the proof was wrapped under:
    /// `__NAME__Verifier.getRootCVadcopFinal()` for a leaf, the recurser's own verkey for
    /// an aggregate. It is not fixed by the verifier because those differ.
    /// @param publicValues The public values encoded as bytes.
    /// @param proofBytes The proof of the program execution the __NAME__ zkVM encoded as bytes.
    function verifySnarkProof(
        __PROGRAM_VK_PARAM__bytes32 rootCVadcopFinal,
        bytes calldata publicValues,
        bytes calldata proofBytes
    ) external view;
}
