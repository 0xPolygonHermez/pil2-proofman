// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity ^0.8.20;

// The Foundry test of the differential fuzzer of the Solidity verifier
// (pilfflonk/docs/verifier.md#differential-fuzzer; pilfflonk/tests/data/fuzz.rs). The fuzzer copies
// it to the test/ of a project of its own, with the generated verifier of one key as
// src/PilfflonkVerifier.sol and its probe as src/PilfflonkProbe.sol: a copy of the verifier that the
// fuzzer instruments (never the generator), which says which check refused a case and the gas left
// after each step.
import {PilfflonkVerifier} from "../src/PilfflonkVerifier.sol";
import {PilfflonkProbe} from "../src/PilfflonkProbe.sol";

// The cheatcodes the test uses, declared here: the project has no forge-std, and Foundry answers
// these calls at its cheatcode address whoever declares them.
interface Vm {
    function readFileBinary(string calldata path) external view returns (bytes memory);
    function writeLine(string calldata path, string calldata data) external;
    function envUint(string calldata name) external view returns (uint256);
    function toString(uint256 value) external pure returns (string memory);
}

/// @notice Calls `verifyProof` of the verifier and of its probe on each case cases/<i>.bin, for i
///         from PILFFLONK_FUZZ_FIRST, PILFFLONK_FUZZ_N of them (the environment's): the calldata
///         of the call, the selector of `verifyProof` and its arguments ABI-encoded, as
///         `proofman-cli pilfflonk calldata --format hex` writes it, or mutated. It writes one line
///         per case to cases/results.txt, "<i> <outcome> <gas> <probe> <probe gas> <words>":
///         - the outcome of the verifier's call: "accept" or "reject" (it returned true or false),
///           "revert", or "badreturn" (it returned something other than a bool);
///         - the gas of that call, measured as test/PilfflonkVerifier.t.sol measures it;
///         - the probe's call, "revert" or "return", its gas and the words it returned, in
///           decimal.
///         It checks nothing itself: the fuzzer compares the outcomes with the JS verifier's.
contract PilfflonkFuzzTest {
    Vm constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));

    string constant RESULTS = "cases/results.txt";

    function test_verifyProof_on_the_fuzz_cases() public {
        PilfflonkVerifier verifier = new PilfflonkVerifier();
        PilfflonkProbe probe = new PilfflonkProbe();
        uint256 first = vm.envUint("PILFFLONK_FUZZ_FIRST");
        uint256 n = vm.envUint("PILFFLONK_FUZZ_N");
        // Nothing of a case is used after it: the next one reuses its memory, so that the memory of
        // the test does not grow with the cases. The first case runs twice, and only the second
        // time is written: the first grows the memory to what a case needs, so that every call is
        // measured with the memory of the test already grown, as the first call of
        // test/PilfflonkVerifier.t.sol is.
        uint256 free;
        assembly {
            free := mload(0x40)
        }
        runCase(address(verifier), address(probe), first, false);
        for (uint256 i = first; i < first + n; i++) {
            assembly {
                mstore(0x40, free)
            }
            runCase(address(verifier), address(probe), i, true);
        }
    }

    function runCase(address verifier, address probe, uint256 i, bool write) internal {
        bytes memory data = vm.readFileBinary(string.concat("cases/", vm.toString(i), ".bin"));
        uint256 before = gasleft();
        (bool ok, bytes memory ret) = verifier.staticcall(data);
        uint256 used = before - gasleft();
        string memory line = string.concat(vm.toString(i), " ", outcome(ok, ret), " ", vm.toString(used));
        before = gasleft();
        (ok, ret) = probe.staticcall(data);
        used = before - gasleft();
        line = string.concat(line, ok ? " return " : " revert ", vm.toString(used));
        if (ok) {
            for (uint256 w = 0; w < ret.length / 32; w++) {
                line = string.concat(line, " ", vm.toString(wordAt(ret, w)));
            }
        }
        if (write) {
            vm.writeLine(RESULTS, line);
        }
    }

    /// What a call to `verifyProof` did: a call that returns must return a bool, 32 bytes that are
    /// 0 or 1.
    function outcome(bool ok, bytes memory ret) internal pure returns (string memory) {
        if (!ok) {
            return "revert";
        }
        if (ret.length != 32) {
            return "badreturn";
        }
        uint256 v = wordAt(ret, 0);
        if (v == 1) {
            return "accept";
        }
        return v == 0 ? "reject" : "badreturn";
    }

    function wordAt(bytes memory b, uint256 w) internal pure returns (uint256 v) {
        assembly {
            v := mload(add(add(b, 32), mul(w, 32)))
        }
    }
}
