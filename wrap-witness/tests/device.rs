//! The blake3 BN128 wrap's witness of `tests/fixtures/blake3_wrap.circom`, a Node of each `over`, a
//! Node with a free `d_inv`, a chunk and a parent Compress and a range check, wrapped by plonk2pil and
//! set up with the ptau of `τ = 1` (`tests/common`, `wrap_key`):
//!
//! - `the_witness_checks_and_binds_its_inverses`: pilfflonk's check holds on the host's witness, and
//!   fails with the `d_inv` of a digest word changed, or with its `over` bit flipped.
//! - `the_device_builds_the_hosts_witness`: the witness the GPU builds from its parts
//!   (`WrapWitness::device_witness`, `proofman_pilfflonk::prove_exec`) is the host's
//!   (`WrapWitness::witness_from_zkin`): the same columns, and the same proof with the same blinding
//!   seed. Without a GPU it says so and passes, as pil2-stark's GPU tests do.
//!
//! Both need `PIL2C_EXEC` and, as the other tests here, Node.js and snarkjs:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test --release -p pilfflonk-wrap-witness --test device \
//!     -- --ignored
//! ```

mod common;

#[path = "../../setup/stark-recurser/tests/common/wrap_key.rs"]
mod wrap_key;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use pil2_stark_recurser::plonk2pil::r1cs::types::PlonkOptions;
use pil2_stark_recurser::plonk2pil::{plonk2pil, PlonkResult};
use pilfflonk_wrap_witness::{WrapArtifacts, WrapWitness};
use proofman_common::exec_format::blake3_wrap_cols::{A, D_INV};
use proofman_common::exec_format::BLAKE3_WRAP_BLOCK_ROWS;
use proofman_common::hash_family::BLAKE3_BN128_WRAP_FAMILY;
use proofman_fields::{Bn128, Field, QuotientMap};
use proofman_pilfflonk::{
    check, gpu_available, prove, prove_exec, stage_columns, stage_columns_exec, CheckOptions, Device, FrBytes,
    ProveOptions, ProvingKey, Witness, WitnessSource,
};

use common::{build_final, circuits_bn128, missing_prerequisite, repo_root};
use wrap_key::{set_up_key_with_powers, Ptau};

/// The circuit's files, its wrap and its key, built once for the tests of this binary.
struct Circuit {
    dir: PathBuf,
    key: PathBuf,
    zkin: serde_json::Value,
}

impl Circuit {
    fn wrap(&self) -> WrapWitness {
        WrapWitness::load(&WrapArtifacts::with_stem(&self.dir.join("final/final"))).unwrap()
    }
}

/// The circuit, or `None` (and why, on stderr) without what the circuit's tools need.
fn circuit() -> Option<&'static Circuit> {
    static CIRCUIT: OnceLock<Option<Circuit>> = OnceLock::new();
    CIRCUIT
        .get_or_init(|| match missing_prerequisite() {
            Some(why) => {
                eprintln!("skipping the blake3 wrap's device tests: {why}");
                None
            }
            None => Some(build_circuit()),
        })
        .as_ref()
}

fn build_circuit() -> Circuit {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("wrap_witness_blake3");
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    // b3_field.circom from circuits.bn128 before circuits.gl's, and circomlib's bitify for it.
    let libraries = [
        circuits_bn128(),
        repo_root().join("setup/stark-recurser/stark2circom/circom_verifier/circuits.gl"),
        repo_root().join("setup/pil2-stark/node_modules/circomlib/circuits"),
    ];
    let r1cs = build_final(&dir, &fixtures.join("blake3_wrap.circom"), &libraries);
    let options = PlonkOptions { hash_id: BLAKE3_BN128_WRAP_FAMILY.into(), ..Default::default() };
    let res: PlonkResult<Bn128> = plonk2pil(&r1cs, "wrap", &options).unwrap();
    let exec: Vec<u8> = res.exec.iter().flat_map(|w| w.to_le_bytes()).collect();
    fs::write(dir.join("final/final.exec"), exec).unwrap();
    let key_dir = dir.join("key");
    fs::create_dir_all(&key_dir).unwrap();
    // The blake3 wrap's largest f, 32·N + 31.
    let key = set_up_key_with_powers(&repo_root(), &key_dir, &res, Ptau::TauOne, 33 << res.n_bits);
    let zkin = serde_json::from_slice(&fs::read(fixtures.join("blake3_wrap.json")).unwrap()).unwrap();
    Circuit { dir, key, zkin }
}

/// The digest rows of the Node blocks (rows 56..59 of each, which come first), as (row, over).
fn digest_rows(witness: &Witness, n_nodes: usize) -> Vec<(usize, bool)> {
    let cell = |row, col| Bn128::from(witness.instances[0].stage1.get(row, col).unwrap());
    (0..n_nodes)
        .flat_map(|b| (0..4).map(move |k| b * BLAKE3_WRAP_BLOCK_ROWS + 56 + k))
        .map(|row| (row, cell(row, A + 1) == Bn128::ONE))
        .collect()
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_witness_checks_and_binds_its_inverses() {
    let Some(circuit) = circuit() else { return };
    let pk = ProvingKey::load(&circuit.key).unwrap();
    let witness = circuit.wrap().witness_from_zkin(&pk.witness_shape().unwrap(), &circuit.zkin).unwrap();
    let report = check(&pk, &witness, &CheckOptions::default()).unwrap();
    assert!(report.holds(), "the check fails on the witness: {:?}", report.failures().collect::<Vec<_>>());

    let rows = digest_rows(&witness, 2);
    let over = rows.iter().find(|(_, over)| *over).expect("the fixture's Node word over p").0;
    let bound =
        rows.iter().find(|(row, over)| !over && witness.instances[0].stage1.get(*row, D_INV).unwrap() != FrBytes::ZERO);
    let bound = bound.expect("a word below p whose d_inv is bound").0;
    let changed = |row: usize, col: usize, value: Bn128| {
        let mut wrong = witness.clone();
        wrong.instances[0].stage1.set(row, col, value.into()).unwrap();
        check(&pk, &wrong, &CheckOptions::default()).unwrap().holds()
    };
    let d_inv = |row| Bn128::from(witness.instances[0].stage1.get(row, D_INV).unwrap());
    assert!(!changed(over, D_INV, d_inv(over) + Bn128::ONE), "a wrong 1/C0 passes");
    assert!(!changed(bound, D_INV, d_inv(bound) + Bn128::ONE), "a wrong 1/(C1 - (2^32 - 1)) passes");
    // The other packing of a word over p, `over` 0 with out = C: refused, as it is not below p.
    let out = Bn128::from(witness.instances[0].stage1.get(over, A).unwrap());
    let p = Bn128::from_int(0xFFFF_FFFF_0000_0001u64);
    let mut wrong = witness.clone();
    wrong.instances[0].stage1.set(over, A, (out + p).into()).unwrap();
    wrong.instances[0].stage1.set(over, A + 1, Bn128::ZERO.into()).unwrap();
    assert!(!check(&pk, &wrong, &CheckOptions::default()).unwrap().holds(), "an unreduced digest word passes");
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_device_builds_the_hosts_witness() {
    if !gpu_available() {
        eprintln!("skipping the blake3 wrap's device test: no GPU");
        return;
    }
    let Some(circuit) = circuit() else { return };
    let pk = ProvingKey::load_on(&circuit.key, Device::Gpu).unwrap();
    let shape = pk.witness_shape().unwrap();
    let wrap = circuit.wrap();
    let options = ProveOptions { insecure_blinding_seed: Some([1; 32]), ..Default::default() };

    let host = wrap.witness_from_zkin(&shape, &circuit.zkin).unwrap();
    let device = wrap.device_witness(&shape, &circuit.zkin).unwrap();
    pk.set_exec(device.air, &wrap.exec_static().expect("a blake3 wrap's exec")).unwrap();
    assert_eq!(device.publics, host.publics().unwrap());
    let host_columns = stage_columns(&pk, &host, &options).unwrap();
    let device_columns = stage_columns_exec(&pk, device.air, device.exec(), device.publics.clone(), &options).unwrap();
    for (p, (h, d)) in host_columns.columns[0].iter().zip(&device_columns.columns[0]).enumerate() {
        if let Some(row) = h.iter().zip(d).position(|(a, b)| a != b) {
            panic!("stage-1 column {p} differs first at row {row}: host {:?}, device {:?}", h[row], d[row]);
        }
    }
    assert_eq!(host_columns.columns, device_columns.columns, "the stage-2 columns");

    let host_proof = prove(&pk, &host, &options).unwrap();
    let device_proof = prove_exec(&pk, device.air, device.exec(), device.publics.clone(), &options).unwrap();
    assert_eq!(host_proof.proof.to_bytes(), device_proof.proof.to_bytes());
}
