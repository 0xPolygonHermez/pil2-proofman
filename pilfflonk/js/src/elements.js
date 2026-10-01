// What the verifier reads as JSON -- integers as decimal strings, as snarkjs writes them -- turned
// into ffjavascript elements, with the checks of steps 1-3 of snarkjs' fflonk verifier
// (src/fflonk_verify.js; pilfflonk/docs/verifier.md#steps, steps 1-3):
// - a scalar is below r;
// - a G1 point is affine, [x, y] or [x, y, "1"] as snarkjs writes it, with coordinates below q and
//   on the curve, which puts it in the group: the cofactor of BN254's G1 is 1;
// - a G2 point is affine, [[x.c0, x.c1], [y.c0, y.c1]] (c0 + c1·u), on the twist and in the
//   r-torsion group.
// A value that fails throws a PilFflonkInputError naming it; the verifier rejects the proof.

import { utils } from "ffjavascript";

export class PilFflonkInputError extends Error {
    constructor(message) {
        super(message);
        this.name = "PilFflonkInputError";
    }
}

const DECIMAL = /^[0-9]+$/;
const CANONICAL_DECIMAL = /^(0|[1-9][0-9]*)$/;

// A non-negative integer from a decimal string, a bigint or a safe integer.
export function toBigInt(value, what) {
    if (typeof value === "bigint" && value >= 0n) return value;
    if (typeof value === "string" && DECIMAL.test(value)) return BigInt(value);
    if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
    throw new PilFflonkInputError(`${what}: ${String(value)} is not a non-negative decimal integer`);
}

// A non-negative integer as pilfflonk's files spell one (pilfflonk/src/field.rs): a decimal
// string of digits only, without sign or leading zeros. The Rust side reads nothing else, so a
// value has a single spelling: a proof, publics or a vkey the verifier accepts are ones Rust reads.
export function decimalFromObject(value, what) {
    if (typeof value === "string" && CANONICAL_DECIMAL.test(value)) return BigInt(value);
    throw new PilFflonkInputError(`${what}: ${show(value)} is not a decimal string without sign or leading zeros`);
}

// A value as an error message shows it: its JSON, bigints included, which JSON.stringify refuses.
export function show(value) {
    return JSON.stringify(value, (_, v) => (typeof v === "bigint" ? `${v}n` : v));
}

// A scalar below r, as an element of curve.Fr.
export function frFromObject(curve, value, what) {
    const v = toBigInt(value, what);
    if (v >= curve.r) {
        throw new PilFflonkInputError(`${what}: ${v} is not below r`);
    }
    return curve.Fr.e(v);
}

function fqFromObject(curve, value, what) {
    const v = toBigInt(value, what);
    if (v >= curve.q) {
        throw new PilFflonkInputError(`${what}: ${v} is not below q`);
    }
    return v;
}

// [x, y], or [x, y, z] with z = 1 as snarkjs' JSON writes affine points: "1" in G1, ["1", "0"] in G2.
function affineCoordinates(value, what, dimension) {
    if (!Array.isArray(value) || (value.length !== 2 && value.length !== 3)) {
        throw new PilFflonkInputError(`${what}: not an affine point [x, y]`);
    }
    if (value.length === 3) {
        const z = dimension === 1 ? [value[2]] : value[2];
        const isOne = (c, i) => toBigInt(c, `${what}.z`) === (i === 0 ? 1n : 0n);
        if (!Array.isArray(z) || z.length !== dimension || !z.every(isOne)) {
            throw new PilFflonkInputError(`${what}: z is not 1, the point is not affine`);
        }
    }
    return value.slice(0, 2);
}

// A G1 point, never the point at infinity unless `allowInfinity`: then (0, 0), the affine form
// ffiasm and the C API write it in (pilfflonk_transcript.hpp, encodeG1).
export function g1FromObject(curve, value, what, { allowInfinity = false } = {}) {
    const [xObject, yObject] = affineCoordinates(value, what, 1);
    const x = fqFromObject(curve, xObject, `${what}.x`);
    const y = fqFromObject(curve, yObject, `${what}.y`);
    if (x === 0n && y === 0n) {
        if (allowInfinity) return curve.G1.zeroAffine;
        throw new PilFflonkInputError(`${what}: the point at infinity`);
    }
    const point = curve.G1.fromObject([x, y]);
    if (!curve.G1.isValid(point)) {
        throw new PilFflonkInputError(`${what}: (${x}, ${y}) is not on the curve`);
    }
    return point;
}

// A G2 point: on the twist, in the r-torsion group, and not the point at infinity.
export function g2FromObject(curve, value, what) {
    const [xObject, yObject] = affineCoordinates(value, what, 2);
    const fq2 = (c, name) => {
        if (!Array.isArray(c) || c.length !== 2) {
            throw new PilFflonkInputError(`${what}.${name}: not an element [c0, c1] of Fq2`);
        }
        return [fqFromObject(curve, c[0], `${what}.${name}.c0`), fqFromObject(curve, c[1], `${what}.${name}.c1`)];
    };
    const x = fq2(xObject, "x");
    const y = fq2(yObject, "y");
    const G2 = curve.G2;
    const point = G2.fromObject([x, y]);
    if (G2.isZero(point)) {
        throw new PilFflonkInputError(`${what}: the point at infinity`);
    }
    if (!G2.isValid(point)) {
        throw new PilFflonkInputError(`${what}: not on the twist`);
    }
    if (!G2.isZero(G2.timesScalar(point, curve.r))) {
        throw new PilFflonkInputError(`${what}: not in the r-torsion group`);
    }
    return point;
}

// The inverses of frFromObject and g1FromObject: decimal strings, a G1 point affine as [x, y] and
// the point at infinity as (0, 0).
export function frToObject(curve, element) {
    return curve.Fr.toString(element, 10);
}

export function g1ToObject(curve, point) {
    if (curve.G1.isZero(point)) return ["0", "0"];
    const [x, y] = curve.G1.toObject(curve.G1.toAffine(point));
    return utils.stringifyBigInts([x, y]);
}
