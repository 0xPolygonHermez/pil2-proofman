// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.20;

// The generated verifier of one key (setup/pilfflonk/src/solidity.rs), which the tests
// (pilfflonk/tests/data/foundry.rs) write here with the cases of that key.
import {PilfflonkVerifier} from "../src/PilfflonkVerifier.sol";

// The cheatcodes the test uses, declared here: the project has no forge-std, and Foundry answers
// these calls at its cheatcode address whoever declares them.
interface Vm {
    function readFile(string calldata path) external view returns (string memory);
    function writeLine(string calldata path, string calldata data) external;
    function parseJsonUint(string calldata json, string calldata key) external pure returns (uint256);
    function parseJsonBytes(string calldata json, string calldata key) external pure returns (bytes memory);
    function parseJsonString(string calldata json, string calldata key) external pure returns (string memory);
    function toString(uint256 value) external pure returns (string memory);
}

/// @notice Runs `verifyProof` of the key's verifier on every case of cases/cases.json:
///         {"n": n, "cases": [{"label", "calldata", "expected"}]}, `calldata` the call's, as hex: the
///         selector of `verifyProof` and its two arguments ABI-encoded, the words of `bytes32[W]` and
///         of `uint256[P]`, as `proofman-cli pilfflonk calldata --format hex` writes it; and
///         `expected` what the call must do: "accept" (return true, what the JS verifier accepts),
///         "reject" (return false: every refusal, a malformed value included) or "revert" (calldata
///         shorter than the arguments, which the ABI decoder refuses). A case whose selector is not
///         solc's for `verifyProof` is an unexpected outcome. It writes one line "<i> <outcome> <gas>"
///         per case to cases/results.txt, the gas that of the call, and fails if an outcome is not the
///         expected one.
contract PilfflonkVerifierTest {
    Vm constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));

    string constant CASES = "cases/cases.json";
    string constant RESULTS = "cases/results.txt";

    function test_verifyProof_agrees_with_the_js_verifier() public {
        PilfflonkVerifier verifier = new PilfflonkVerifier();
        string memory json = vm.readFile(CASES);
        uint256 n = vm.parseJsonUint(json, ".n");
        string memory disagreements = "";
        for (uint256 i = 0; i < n; i++) {
            string memory c = string.concat(".cases[", vm.toString(i), "]");
            // The ABI encodes the fixed-size arrays in place: the calldata is the selector and the
            // words of both.
            bytes memory data = vm.parseJsonBytes(json, string.concat(c, ".calldata"));
            string memory label = vm.parseJsonString(json, string.concat(c, ".label"));
            if (data.length < 4 || bytes4(data) != PilfflonkVerifier.verifyProof.selector) {
                disagreements = string.concat(disagreements, " ", label, " (its selector is not verifyProof's)");
            }
            uint256 before = gasleft();
            (bool ok, bytes memory ret) = address(verifier).staticcall(data);
            uint256 used = before - gasleft();
            string memory outcome = "revert";
            if (ok) {
                // A call that returns is verifyProof's: a bool, 32 bytes.
                require(ret.length == 32, "verifyProof returned something else than a bool");
                outcome = abi.decode(ret, (bool)) ? "accept" : "reject";
            }
            vm.writeLine(RESULTS, string.concat(vm.toString(i), " ", outcome, " ", vm.toString(used)));
            string memory expected = vm.parseJsonString(json, string.concat(c, ".expected"));
            if (keccak256(bytes(outcome)) != keccak256(bytes(expected))) {
                disagreements = string.concat(disagreements, " ", label);
            }
        }
        require(bytes(disagreements).length == 0, string.concat("unexpected outcomes on:", disagreements));
    }
}
