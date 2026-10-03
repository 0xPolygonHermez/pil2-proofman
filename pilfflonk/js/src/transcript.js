// The Fiat-Shamir transcript of pilfflonk (pilfflonk/docs/protocol.md#transcript): rapidsnark's
// Keccak256Transcript as the C++ prover drives it (PilFflonk::Transcript,
// pil2-stark/src/pilfflonk/pilfflonk_transcript.cpp), written after snarkjs'
// src/Keccak256Transcript.js, which is the same transcript on the verifier side of the existing
// FFLONK:
// - addScalar: an element of Fr as 32 bytes, big-endian, canonical;
// - addPolCommitment: a G1 point as x‖y, affine, each coordinate 32 bytes big-endian;
// - getChallenge: keccak256 of everything added since the last reset, mod r;
// - squeeze: getChallenge, then reset() and addScalar(challenge), as FflonkProver does between
//   rounds (fflonk_prover.c.hpp:849-851).
//
// addPolCommitment refuses, with a PilFflonkInputError, the points the C++ transcript refuses: the
// point at infinity, which Keccak256Transcript does not encode as x‖y, and a point with a
// coordinate below 2^192, which ffiasm does not write as 32 big-endian bytes. The prover never
// absorbs either, so no proof it makes has one where the transcript absorbs a point.

import { Scalar } from "ffjavascript";
import { keccak_256 } from "@noble/hashes/sha3";

import { PilFflonkInputError } from "./elements.js";

const FR_BYTES = 32;
const FQ_BYTES = 32;
// The big-endian bytes of a coordinate below 2^192: its first 8.
const SHORT_BYTES = 8;

function isShortCoordinate(bytes, offset) {
    return bytes.subarray(offset, offset + SHORT_BYTES).every((b) => b === 0);
}

export class Keccak256Transcript {
    constructor(curve) {
        this.Fr = curve.Fr;
        this.G1 = curve.G1;
        this.r = curve.r;

        this.reset();
    }

    reset() {
        this.data = [];
    }

    isEmpty() {
        return this.data.length === 0;
    }

    addScalar(scalar) {
        if (!(scalar instanceof Uint8Array) || scalar.byteLength !== this.Fr.n8) {
            throw new TypeError("Keccak256Transcript: a scalar must be an element of curve.Fr");
        }
        const bytes = new Uint8Array(FR_BYTES);
        this.Fr.toRprBE(bytes, 0, scalar);
        this.data.push(bytes);
    }

    addPolCommitment(point) {
        const G1 = this.G1;
        const size = point instanceof Uint8Array ? point.byteLength : 0;
        if (size !== 2 * FQ_BYTES && size !== 3 * FQ_BYTES) {
            throw new TypeError("Keccak256Transcript: a commitment must be a point of curve.G1");
        }
        if (G1.isZero(point)) {
            throw new PilFflonkInputError(
                "Keccak256Transcript: the point at infinity is never absorbed (pilfflonk/docs/protocol.md#transcript)",
            );
        }
        if (!G1.isValid(point)) {
            throw new PilFflonkInputError("Keccak256Transcript: the point is not on the curve");
        }
        const bytes = new Uint8Array(2 * FQ_BYTES);
        G1.toRprUncompressed(bytes, 0, point);
        if (isShortCoordinate(bytes, 0) || isShortCoordinate(bytes, FQ_BYTES)) {
            throw new PilFflonkInputError(
                "Keccak256Transcript: a point with a coordinate below 2^192 is never absorbed " +
                    "(pilfflonk/docs/protocol.md#transcript)",
            );
        }
        this.data.push(bytes);
    }

    getChallenge() {
        if (this.isEmpty()) {
            throw new Error("Keccak256Transcript: No data to generate a transcript");
        }
        const length = this.data.reduce((n, bytes) => n + bytes.byteLength, 0);
        const buffer = new Uint8Array(length);
        let offset = 0;
        for (const bytes of this.data) {
            buffer.set(bytes, offset);
            offset += bytes.byteLength;
        }
        const value = Scalar.fromRprBE(keccak_256(buffer), 0, 32);
        return this.Fr.e(Scalar.mod(value, this.r));
    }

    squeeze() {
        const challenge = this.getChallenge();
        this.reset();
        this.addScalar(challenge);
        return challenge;
    }
}
