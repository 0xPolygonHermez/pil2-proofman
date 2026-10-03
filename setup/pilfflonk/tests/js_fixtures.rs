//! Fixtures for the JS verifier, for what only Rust has: the ptau with `τ = 1` and `Q(ξ)` from the
//! oracle (pilfflonk/docs/README.md#tests). `pilfflonk/js/test/setup-fixtures.sh` runs them around
//! `proofman-setup`, into the directory `PILFFLONK_JS_FIXTURES` names:
//!
//! 1. `tau_one_ptau` writes `tau_one.ptau`, for `proofman-setup setup-pilfflonk`;
//! 2. the script compiles the Fibonacci fixture to `fibonacci.pilout` and sets it up in `build/`;
//! 3. `q_at_xi` writes `q_at_xi.json`: for a few points `ξ` and challenges `std_vc`, the
//!    evaluation at `ξ·ω^prime` of every entry of the evMap of `build/provingKey`, named as the
//!    proof names it (`ProofNames`), and the oracle's `Q(ξ)` from the same columns. The JS test
//!    evaluates the vkey's `qVerifier` on them and must obtain that `Q(ξ)`.
//!
//! Both are `#[ignore]`: they only write fixtures, and only when asked. Run without
//! `PILFFLONK_JS_FIXTURES` (as `--include-ignored` runs them), they say so and write nothing.
//!
//! ```text
//! PILFFLONK_JS_FIXTURES=<dir> cargo test -p pilfflonk-setup --features proofman-starks-lib-c/cpu-only \
//!     --test js_fixtures -- --ignored --exact tau_one_ptau
//! ```
//!
//! They are this crate's, not `proofman-pilfflonk`'s, because the ptau is (`test_ptau`): this crate
//! depends on `proofman-pilfflonk`, and a dev-dependency back on this one would be a cycle.

/// The Fibonacci's generator, `proofman-pilfflonk`'s test module, included as it is.
#[path = "../../../pilfflonk/tests/data/fibonacci.rs"]
mod fibonacci;

use std::fs;
use std::path::PathBuf;

use pil2_pilout::pilout_proxy::PilOutProxy;
use pilfflonk_setup::test_ptau::write_tau_one_ptau;
use proofman_pilfflonk::oracle::{self, AirOracle, ColumnRef, Fr, Values};
use proofman_pilfflonk::{
    AirFile, EvMapEntry, FileWitnessSource, JsonFile, PilfflonkGlobalInfo, PilfflonkInfo, PolType, ProofNames, Vkey,
};
use serde_json::{json, Map, Value};

/// The powers `[τ^i]₁` of the ptau: more than the Fibonacci's largest degree, 513 grouped, 261
/// unpacked.
const PTAU_G1: usize = 1024;

/// The fixture's size and inputs, as `pilfflonk/tests/fibonacci.rs`
/// (pilfflonk/docs/README.md#fixtures).
const N_BITS: u32 = 8;
const INPUTS: [u64; 2] = [1, 2];

/// The directory the fixtures go to, if they are asked for.
fn fixtures_dir() -> Option<PathBuf> {
    match std::env::var_os("PILFFLONK_JS_FIXTURES") {
        Some(dir) if !dir.is_empty() => Some(PathBuf::from(dir)),
        _ => {
            eprintln!(
                "PILFFLONK_JS_FIXTURES is not set: no fixture is written (see pilfflonk/js/test/setup-fixtures.sh)"
            );
            None
        }
    }
}

fn decimal(v: &Fr) -> String {
    v.to_bytes().to_decimal()
}

#[test]
#[ignore = "writes the JS verifier's fixtures into PILFFLONK_JS_FIXTURES"]
fn tau_one_ptau() {
    let Some(dir) = fixtures_dir() else { return };
    write_tau_one_ptau(&dir.join("tau_one.ptau"), PTAU_G1).unwrap();
}

/// The column of the oracle an evaluation is of: a fixed column by its index, which is the
/// pilout's, and a committed one by its stage and `stageId`, the pilout's index within the stage
/// (`pilfflonk/tests/fibonacci.rs`); an im pol by its `expId`, the index of its expression in the
/// pilout (the setup's numbering, which `setup/pil2-stark/tests/setup_pilfflonk.rs` checks).
fn column_of(info: &PilfflonkInfo, e: &EvMapEntry) -> ColumnRef {
    let pol = info.pol(e.pol_type, e.id).expect("an evMap entry of the pilfflonkinfo");
    match (e.pol_type, pol.exp_id) {
        (PolType::Const, _) => ColumnRef::Fixed(e.id as usize),
        (PolType::Cm, Some(exp_id)) if pol.im_pol => ColumnRef::Im(exp_id as usize),
        (PolType::Cm, _) => ColumnRef::Witness { stage: pol.stage as usize, idx: pol.stage_id as usize },
    }
}

/// The AIR the cases of `q_at_xi.json` are of.
struct Air<'a> {
    oracle: AirOracle,
    info: &'a PilfflonkInfo,
    /// Every entry of the evMap, with its name in the proof.
    names: Vec<(String, &'a EvMapEntry)>,
}

impl Air<'_> {
    /// One case of `q_at_xi.json`: the evaluations of `values` at `ξ` and `Q(ξ)` from them.
    fn case(&self, name: &str, values: &Values, xi: Fr, std_vc: Fr) -> Value {
        let im_pols: Vec<usize> = self
            .info
            .cm_pols_map
            .iter()
            .filter(|p| p.im_pol)
            .map(|p| p.exp_id.expect("an im pol has an expId") as usize)
            .collect();
        let mut evaluations = Map::new();
        for (name, e) in &self.names {
            let value = self.oracle.column_at(values, column_of(self.info, e), e.prime as i32, &xi).unwrap();
            evaluations.insert(name.clone(), json!(decimal(&value)));
        }
        let q = self.oracle.q_at(values, &im_pols, &std_vc, &xi).unwrap();
        json!({
            "name": name,
            "xi": decimal(&xi),
            "challenges": {"std_vc": decimal(&std_vc)},
            "publics": values.publics.iter().map(decimal).collect::<Vec<_>>(),
            "evaluations": evaluations,
            "q": decimal(&q),
        })
    }
}

#[test]
#[ignore = "writes the JS verifier's fixtures into PILFFLONK_JS_FIXTURES"]
fn q_at_xi() {
    let Some(dir) = fixtures_dir() else { return };
    let pilout = PilOutProxy::new(dir.join("fibonacci.pilout").to_str().expect("a UTF-8 path")).unwrap().pilout;
    let proving_key = dir.join("build").join("provingKey");
    let global_info = PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap();
    let info = PilfflonkInfo::read(&global_info.air_file(&proving_key, 0, 0, AirFile::PilfflonkInfo).unwrap()).unwrap();
    let vkey = Vkey::read(&global_info.vkey_path(&proving_key)).unwrap();
    assert_eq!(vkey.ev_map, info.ev_map, "the vkey has the evMap of the AIR");

    // The names of the evaluations in the order of the proof (pilfflonk/docs/formats.md#proof):
    // the fixed columns', then the others', each in the order of the evMap.
    let ordered = info.ev_map.iter().filter(|e| e.pol_type == PolType::Const);
    let ordered = ordered.chain(info.ev_map.iter().filter(|e| e.pol_type == PolType::Cm));
    let proof_names = ProofNames::new(&global_info, &[&info]).unwrap();
    assert_eq!(proof_names.evaluations().len(), info.ev_map.len());
    let names: Vec<(String, &EvMapEntry)> = proof_names.evaluations().iter().cloned().zip(ordered).collect();

    let air = Air { oracle: AirOracle::new(&pilout, 0, 0).unwrap(), info: &info, names };
    assert_eq!(air.oracle.n_bits(), N_BITS);
    let witness_dir = dir.join("witness");
    fibonacci::witness(N_BITS, INPUTS).write(&witness_dir, &oracle::witness_shape(&pilout).unwrap()).unwrap();
    let source = FileWitnessSource::open(&witness_dir, &oracle::witness_shape(&pilout).unwrap()).unwrap();
    let values = air.oracle.values(&source, 0).unwrap();
    fs::remove_dir_all(&witness_dir).unwrap();
    assert_eq!(air.oracle.check(&values).unwrap(), [], "the generator's witness satisfies the AIR");

    // A witness that does not satisfy the AIR: Q(ξ) is still what the fold of the evaluations
    // gives, which is all the verifier computes.
    let mut mutated = values.clone();
    mutated.witness[0][0][100] = &mutated.witness[0][0][100] + &Fr::one();
    assert!(!air.oracle.check(&mutated).unwrap().is_empty());

    // Points off H and challenges, fixed so that the fixture is deterministic: as good as random.
    let cases = [
        air.case("satisfied", &values, Fr::from_u64(7).pow_u64(1000), Fr::from_u64(0x5eed).pow_u64(77)),
        air.case("negative", &values, -&Fr::from_u64(123_456_789), Fr::from_u64(3).pow_u64(200)),
        air.case("mutated", &mutated, Fr::from_u64(11).pow_u64(999), Fr::from_u64(13).pow_u64(555)),
    ];
    let fixture = json!({
        "air": info.name,
        "nBits": info.n_bits,
        "challengesMap": info.challenges_map,
        "cases": cases,
    });
    fs::write(dir.join("q_at_xi.json"), serde_json::to_string_pretty(&fixture).unwrap()).unwrap();
}
