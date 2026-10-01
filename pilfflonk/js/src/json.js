// The canonical JSON (pilfflonk/docs/formats.md#digest), over which the digest of the vkey is
// computed: what proofman_pilfflonk::json::canonical_json writes (pilfflonk/src/json.rs):
// - no whitespace;
// - object keys sorted by UTF-16 code units, the order of Array.prototype.sort() without a
//   comparator. The keys are written here in that order, one by one: JSON.stringify of an object
//   would list integer-like keys ("0", "10") first, in numeric order, whatever the sort;
// - strings escaped as JSON.stringify escapes them;
// - numbers only as integers of at most 2^53 - 1 in absolute value, which is what the Rust side
//   reads as integers; big integers and points are decimal strings already.
// Anything else -- another number, a string with a lone surrogate (which Rust cannot hold), a
// value JSON has not -- is refused with a PilFflonkInputError: no vkey Rust writes has one.

import { PilFflonkInputError } from "./elements.js";

// A UTF-16 code unit of a surrogate that is not part of a pair.
const LONE_SURROGATE = /\p{Surrogate}/u;

function fail(path, message) {
    throw new PilFflonkInputError(`canonical JSON of ${path}: ${message}`);
}

function writeString(s, path) {
    if (LONE_SURROGATE.test(s)) fail(path, "a string with a lone surrogate");
    return JSON.stringify(s);
}

function write(value, path, out) {
    if (value === null) {
        out.push("null");
    } else if (typeof value === "boolean") {
        out.push(value ? "true" : "false");
    } else if (typeof value === "number") {
        // -0 is not an integer to serde_json, which reads "-0" as a float.
        if (!Number.isSafeInteger(value) || Object.is(value, -0)) {
            fail(path, `${value} is not an integer of at most 2^53 - 1 in absolute value`);
        }
        out.push(String(value));
    } else if (typeof value === "string") {
        out.push(writeString(value, path));
    } else if (Array.isArray(value)) {
        out.push("[");
        value.forEach((item, i) => {
            if (i > 0) out.push(",");
            write(item, `${path}[${i}]`, out);
        });
        out.push("]");
    } else if (typeof value === "object" && Object.getPrototypeOf(value) === Object.prototype) {
        out.push("{");
        Object.keys(value)
            .sort()
            .forEach((key, i) => {
                if (i > 0) out.push(",");
                out.push(writeString(key, path), ":");
                write(value[key], `${path}.${key}`, out);
            });
        out.push("}");
    } else {
        fail(path, `a ${typeof value} is not a JSON value`);
    }
}

// The canonical JSON of a value as JSON.parse returns it.
export function canonicalJson(value) {
    const out = [];
    write(value, "$", out);
    return out.join("");
}
