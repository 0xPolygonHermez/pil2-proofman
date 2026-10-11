# Generated constraint code for the reference AIR

These are pil1's `chelpers` output for pil-fflonk's example AIR: the compiled
`publics`, `step2prev`, `step3prev`, `step3` and `step42ns` bodies the prover
calls per row.

They are **not library code**. They describe one specific AIR, and they exist
here so the moved prover can be run end to end against the vendored artifacts
in `../../tests/fixtures/reference/`.

pil2 expresses the same thing as uint16 bytecode rather than generated C++, and
evaluates it with the machinery in `../fr_expressions.hpp` and
`../fr_evaluator.hpp`. Making the prover work for a pil2 AIR means giving
`PilFflonkSteps` a second implementation backed by that evaluator — the class is
currently concrete, so that is the seam to open when there is a BN254 pil2 AIR
to drive it.
