#!/usr/bin/env python3
"""Builds the GOLDEN_DIR cases of tests/plonk2pil_golden.rs from the r1cs of these runs, all under <scratch>:
  b-<H>/            proofman-setup setup -r -u <fixed-shared> --hash <H>  (+ setup-snark --publics-info ... for recursivef/final)
  rt-<H>-<kind>/    proofman-setup setup-recursive-test -c examples/test-recursive/test.circom -n test -t <kind> --hash <H>
usage: plonk2pil_golden_cases.py <scratch> <out-dir>   (H in blake3 Poseidon1 Poseidon2; the fib-square key's nBits below)"""
import json, os, shutil, sys

scratch, out = sys.argv[1:3]

def add(name, src, **meta):
    if not os.path.exists(src):
        print("skip (missing):", name, src)
        return
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    shutil.copyfile(src, os.path.join(d, "circuit.r1cs"))
    json.dump({"airgroup": None, "lanes": None, "min_n_bits": None, "max_degree": None, **meta}, open(os.path.join(d, "meta.json"), "w"))

for h in ("blake3", "Poseidon2", "Poseidon1"):
    b = f"{scratch}/b-{h}/build"
    pos = h != "blake3"
    # floors = the nBits the key's setup logged: recursive1/2, vadcop_final, vadcop_final_compressed
    rec, fin, cfin = (17, 16, 15) if pos else (19, 18, 18)
    lanes = None if pos else 4
    for air in ("FibonacciSquare", "Module", "SpecifiedRanges"):
        add(f"{h}-{air}_recursive1", f"{b}/{air}_recursive1.r1cs", family=h, kind="aggregation", min_n_bits=rec, lanes=lanes)
    add(f"{h}-recursive2", f"{b}/FiboCPU_recursive2.r1cs", family=h, kind="aggregation", min_n_bits=rec, lanes=lanes)
    add(f"{h}-vadcop_final", f"{b}/vadcop_final.r1cs", family=h, kind="aggregation", min_n_bits=fin, lanes=lanes)
    add(f"{h}-vadcop_final_compressed", f"{b}/vadcop_final_compressed.r1cs", family=h, kind="aggregation", min_n_bits=cfin, lanes=None if pos else 2)
    add(f"{h}-compressor", f"{b}/FibonacciSquare_compressor.r1cs", family=h, kind="compressor", max_degree=5)  # Poseidon only
    add(f"{h}-recursivef", f"{scratch}/b-{h}/build/recursivef.r1cs", family=h, kind="aggregation", lanes=None if pos else 2)
    add(f"{h}-wrap", f"{scratch}/b-{h}/build/final.r1cs", family="blake3" if h == "blake3" else "Poseidon1", kind="wrap")  # Poseidon1 wraps every Poseidon key
    for kind in ("compressor", "aggregation"):
        add(f"{h}-rt-{kind}", f"{scratch}/rt-{h}-{kind}/build/test.r1cs", family=h, kind=kind, max_degree=5 if kind == "compressor" else None, lanes=lanes)
