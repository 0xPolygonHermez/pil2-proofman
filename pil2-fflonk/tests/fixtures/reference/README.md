# Reference artifacts from pil-fflonk

Inputs to pil-fflonk's C++ prover, copied from its `config/` and `runtime/`
before that repository was retired. They are the only surviving copies.

## Why these are kept

pil2-fflonk is a new implementation of the parts of pil-fflonk that had to
adapt to pil2-proofman, not a move of it. The opening scheme carries over
unchanged; the artifact layer, the constraint machinery and the orchestration
do not. Validating the new prover therefore means checking it against what the
old one produced, and these are what the old one consumed.

They cannot be regenerated:

* producing them needs pil1's setup (pilcom / pil-stark JS), which pil2 replaced;
* rebuilding the prover that consumed them needs the `ffiasm` commit
  `38003c1ae51d8cc1` recorded in pil-fflonk's submodules, and that commit no
  longer exists upstream -- `git fetch origin 38003c1ae51d8cc1` answers
  "couldn't find remote ref". `build-repair.patch` is the workaround.

## Contents

| File | What it is |
| --- | --- |
| `pilfflonk.zkey` | The pil1 proving key: the SRS (PTau), the `f_i` plan, omegas, `X_2`. |
| `pilfflonk.fflonkinfo.json` | The pil1 AIR description: `verifierCode`, `evMap`, `maxPolsOpenings`, `qDeg`. |
| `pilfflonk.const` | Constant (preprocessed) polynomial values. |
| `pilfflonk.commit` | The committed trace the reference proof was produced from. |
| `public.json` | The public inputs for that proof. |
| `build-repair.patch` | Repairs to pil-fflonk's build: repoints the includes at pil2-stark's vendored ffiasm/rapidsnark/XKCP, adds `-D__USE_ASSEMBLY__` and `fec.cpp`/`fnec.cpp`, and fixes a `.c`/`.hpp` include in the transcript. Without it the reference does not build at all. |

Note that the patch hardcodes `PIL2_STARK := /home/xavi/dev/pil2-proofman/pil2-stark/src`.
Point it at your own checkout before applying.

The proof produced from these, and the vectors logged alongside it, are the
files in the parent directory: `pilfflonk.proof.json`, `pilfflonk.vkey`,
`pilfflonk.shkey.json`, `pilfflonk.reference.json` and
`pilfflonk.verifierinfo.json` (a trimmed `fflonkinfo`).

Blinding is randomised, so the proof and its vectors correspond only to each
other. Re-running the reference against these inputs produces a *different*
valid proof -- it will not reproduce `pilfflonk.proof.json`.
