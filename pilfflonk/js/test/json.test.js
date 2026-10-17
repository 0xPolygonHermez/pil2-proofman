// The canonical JSON and the digest of the vkey (pilfflonk/docs/formats.md#digest): the cases of
// pilfflonk/src/json.rs with the same expected text, the order of JavaScript's integer-like keys,
// and the sample vkey's pinned digest vector (setup/pilfflonk/tests/setup/digest.rs), which Rust
// computes with the C++ Keccak.

import assert from "node:assert/strict";
import { test } from "node:test";

import { PilFflonkInputError } from "../src/elements.js";
import { canonicalJson } from "../src/json.js";
import { vkeyDigest } from "../src/vkey.js";
import { G2_GENERATOR, SAMPLE_DIGEST, sampleVkey } from "./proofs.js";

test("canonical JSON has no whitespace and sorted keys (json.rs)", () => {
    const v = { b: [1, { d: null, c: true }], a: "x\ny", e: -3 };
    assert.equal(canonicalJson(v), '{"a":"x\\ny","b":[1,{"c":true,"d":null}],"e":-3}');
});

test("canonical JSON does not depend on the order of the keys (json.rs)", () => {
    const one = JSON.parse('{"z": 1, "a": {"y": [3, {"k": "v", "j": "w"}], "b": 2}, "m": "0"}');
    const other = JSON.parse('{"m": "0", "a": {"b": 2, "y": [3, {"j": "w", "k": "v"}]}, "z": 1}');
    assert.equal(canonicalJson(one), canonicalJson(other));
    assert.equal(canonicalJson(one), '{"a":{"b":2,"y":[3,{"j":"w","k":"v"}]},"m":"0","z":1}');
});

test("keys sort by UTF-16 code units, as sort() sorts them (json.rs)", () => {
    // U+FF61 is 0xFF61 in UTF-16 and U+1F600 is 0xD83D 0xDE00: the emoji goes first.
    const v = { "\u{ff61}": 1, "\u{1f600}": 2, b: 3, B: 4 };
    assert.equal(canonicalJson(v), '{"B":4,"b":3,"\u{1f600}":2,"\u{ff61}":1}');
});

test("integer-like keys are sorted as strings, not listed first as JSON.stringify lists them", () => {
    const v = JSON.parse('{"a": 3, "10": 1, "9": 2, "1": 4}');
    assert.equal(JSON.stringify(v), '{"1":4,"9":2,"10":1,"a":3}');
    assert.equal(canonicalJson(v), '{"1":4,"10":1,"9":2,"a":3}');
});

test("strings are escaped as JSON.stringify escapes them (json.rs)", () => {
    const v = ['"\\/', "\b\f\n\r\t", "\x01\x1f\x7f", "é "];
    assert.equal(canonicalJson(v), '["\\"\\\\/","\\b\\f\\n\\r\\t","\\u0001\\u001f\x7f","é "]');
});

test("only integers of at most 2^53 - 1 in absolute value are canonical (json.rs)", () => {
    assert.equal(canonicalJson(Number.MAX_SAFE_INTEGER), "9007199254740991");
    assert.equal(canonicalJson(-Number.MAX_SAFE_INTEGER), "-9007199254740991");
    for (const bad of [2 ** 53, 1.5, -0, Number.NaN, Infinity, [1.5], { a: 2 ** 60 }]) {
        assert.throws(() => canonicalJson(bad), PilFflonkInputError, String(bad));
    }
});

test("what JSON has not, or Rust cannot hold, has no canonical form", () => {
    for (const bad of [undefined, 1n, () => 1, new Date(0), "\ud800", { "\udc00": 1 }, [undefined]]) {
        assert.throws(() => canonicalJson(bad), PilFflonkInputError);
    }
    assert.equal(canonicalJson("\u{1f600}"), '"\u{1f600}"', "a surrogate pair is a character");
});

test("the digest of the sample vkey is its pinned vector", () => {
    const vkey = sampleVkey();
    assert.equal(vkeyDigest(vkey), SAMPLE_DIGEST);
    // The preimage starts as Rust's does: "pilfflonk-v1" and the first key, X_2.
    const { digest: _digest, ...rest } = vkey;
    assert.ok(canonicalJson(rest).startsWith('{"X_2":[['));
});

test("the digest depends neither on the digest nor on the order of the fields", () => {
    const vkey = sampleVkey();
    assert.equal(vkeyDigest({ ...vkey, digest: `0x${"0".repeat(64)}` }), SAMPLE_DIGEST);
    const reversed = Object.fromEntries(Object.entries(vkey).reverse());
    assert.equal(vkeyDigest(reversed), SAMPLE_DIGEST);
    assert.equal(vkeyDigest(JSON.parse(JSON.stringify(vkey, null, 1))), SAMPLE_DIGEST);
});

// The changes of digest.rs's the_digest_changes_with_every_field, and a few more.
test("changing any field changes the digest", () => {
    const changes = {
        nPublic: (v) => (v.nPublic = 3),
        power: (v) => (v.power = 4),
        powerW: (v) => (v.powerW = 4),
        "X_2.y.c1": (v) => (v.X_2 = [G2_GENERATOR[0], [G2_GENERATOR[1][0], "1"]]),
        numChallenges: (v) => (v.numChallenges = [1]),
        "evMap[1].prime": (v) => (v.evMap[1].prime = -2),
        "evMap[2].openingPos": (v) => (v.evMap[2].openingPos = 0),
        "layout[2].degree": (v) => (v.layout[2].degree = 22),
        boundaries: (v) => v.boundaries.push({ name: "firstRow" }),
        f0: (v) => (v.f0 = ["1", "21888242871839275222246405745257275088696311157297823662689037894645226208581"]),
        qDeg: (v) => (v.qDeg = 3),
        maxQDegree: (v) => (v.maxQDegree = 1),
        "qVerifier.tmpUsed": (v) => (v.qVerifier.tmpUsed = 2),
        protocol: (v) => (v.protocol = "pilfflonk2"),
        curve: (v) => (v.curve = "not-a-curve"),
        formatVersion: (v) => (v.formatVersion = 2),
        "layout[0].pols[0].name": (v) => (v.layout[0].pols[0].name = "L2"),
        "evMap[0].type": (v) => (v.evMap[0].type = "cm"),
        "qVerifier.code[0].src[0].id": (v) => (v.qVerifier.code[0].src[0].id = 2),
        "a new field": (v) => (v.f1 = ["1", "2"]),
    };
    const digests = new Map([[SAMPLE_DIGEST, "none"]]);
    for (const [name, change] of Object.entries(changes)) {
        const vkey = sampleVkey();
        change(vkey);
        const digest = vkeyDigest(vkey);
        assert.ok(!digests.has(digest), `${name} gives the digest of ${digests.get(digest)}`);
        digests.set(digest, name);
    }
});
