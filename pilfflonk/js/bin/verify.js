#!/usr/bin/env node
// Verifies a pilfflonk proof (pilfflonk/docs/verifier.md#js-verifier), with the arguments of
// `snarkjs fflonk verify`:
//
//     node pilfflonk/js/bin/verify.js <pilfflonk.vkey.json> <publics.json> <proof.json>
//
// Exit status: 0 if the proof verifies, 1 if it does not (a malformed input included), 2 if the
// arguments are wrong or a file cannot be read as JSON. Unlike pil-stark's main_verifier.js, which
// exits with 0 either way, only a proof that verifies gives 0. The steps are logged to stderr, and
// the verdict is the last line.

import { readFileSync } from "node:fs";
import { argv, exit, stderr } from "node:process";

import { verify } from "../src/verify.js";

const USAGE = "usage: verify.js <pilfflonk.vkey.json> <publics.json> <proof.json>";

const logger = {
    debug: () => {},
    info: (message) => stderr.write(`[INFO]  ${message}\n`),
    warn: (message) => stderr.write(`[WARN]  ${message}\n`),
    error: (message) => stderr.write(`[ERROR] ${message}\n`),
};

function readJson(path) {
    try {
        return JSON.parse(readFileSync(path, "utf8"));
    } catch (e) {
        stderr.write(`[ERROR] ${path}: ${e.message}\n`);
        exit(2);
    }
}

const args = argv.slice(2);
if (args.length !== 3) {
    stderr.write(`${USAGE}\n`);
    exit(2);
}
const [vkey, publics, proof] = args.map(readJson);
const verified = await verify(vkey, publics, proof, logger);
stderr.write(verified ? "OK: the proof verifies\n" : "INVALID: the proof does not verify\n");
exit(verified ? 0 : 1);
