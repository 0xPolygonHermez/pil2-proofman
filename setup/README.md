# Setup

This directory contains the Rust-based proving-key generation pipeline for ProofMan, replacing the prior JavaScript-based setup toolchain. It is a Rust-native adaptation of the JavaScript implementation in [pil2-proofman-js](https://github.com/0xPolygonHermez/pil2-proofman-js) by PolygonHermez.

## Layout

- [circom/](circom/) — Vendored Circom compiler and embedded witness runtime used by the recursive setup.
- [final_snark_circom/](final_snark_circom/) — Final SNARK Circom templates and wrappers.
- [pil2-stark/](pil2-stark/) — Native PIL info, setup layout, global constraints, and expression / verifier helper binfile generation.
- [stark-recurser/](stark-recurser/) — Recursive PLONK layout, AIR schema builder, and recursive Circom rendering (recursive1/recursive2/final/compressed-final wrappers, FRI templates, transcripts, evaluation templates, VADCOP helpers).

## Acknowledgements

This setup pipeline was heavily inspired by the [Venus](https://github.com/) team's `pk-setup-rs` work, released in Venus v0.2.0. We are grateful to the Venus team for their foundational work, which made this Rust-native pipeline possible.
