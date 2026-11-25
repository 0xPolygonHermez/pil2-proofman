// SPDX-License-Identifier: AGPL-3.0
pragma solidity ^0.8.20;

import {I__NAME__Verifier} from "./I__NAME__Verifier.sol";
import {__SNARK__Verifier} from "__SNARK_SOURCE__";

/// @title __NAME__ Verifier
/// @author SilentSig
/// @notice This contracts implements a solidity verifier for __NAME__.
contract __NAME__Verifier is __SNARK__Verifier, I__NAME__Verifier {
    error InvalidProof();

    function VERSION() external pure returns (string memory) {
        return "__VERSION__";
    }

    /// @notice Root constant as bytes32 (pre-packed to match the original uint64[4] layout)
    /// @dev The root for a *leaf* proof. An aggregated proof is wrapped under its
    /// recurser's own verkey instead, so a caller verifying folds passes that one.
    /// Either way the value must be a constant of the calling contract -- see
    /// I__NAME__Verifier.
    function getRootCVadcopFinal() external pure returns (bytes32) {
        return bytes32(
            abi.encodePacked(
                uint64(__ROOT_C_0__),
                uint64(__ROOT_C_1__),
                uint64(__ROOT_C_2__),
                uint64(__ROOT_C_3__)));
    }

    uint256 internal constant _RFIELD =
        21888242871839275222246405745257275088548364400416034343698204186575808495617;

    /// @notice Hashes the public values into a field element inside BN254.
    function hashPublicValues(
        __PROGRAM_VK_PARAM__bytes32 rootCVadcopFinal,
        bytes calldata publicValues
    ) public pure returns (uint256) {
        return uint256(
            sha256(abi.encodePacked(__PACKED__))
        ) % _RFIELD;
    }

    /// @notice Verifies a proof with given public values and vkey.
    function verifySnarkProof(
        __PROGRAM_VK_PARAM__bytes32 rootCVadcopFinal,
        bytes calldata publicValues,
        bytes calldata proofBytes
    ) external view {
        uint256 publicValuesDigest = hashPublicValues(__PROGRAM_VK_ARG__rootCVadcopFinal, publicValues);

        __PROOF_DECODE__

        bool success = this.verifyProof(proofDecoded, [publicValuesDigest]);

        if (!success) {
            revert InvalidProof();
        }
    }
}
