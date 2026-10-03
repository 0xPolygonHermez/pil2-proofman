//! The witness generator of the fixture `mixed_bus` (pilfflonk/docs/README.md#fixtures),
//! `tests/fixtures/mixed_bus/mixed_bus.pil`, over BN254's `Fr`: the columns of the Connection's
//! generator and then the Plookup's for `N = 2^8`, as `tests/data/all.rs` writes them. In the
//! pilout's order (stage 1, `colIdx` 0 to 7): `connection_a`, `connection_b`, `connection_c`,
//! `plookup_sel`, `plookup_a`, `plookup_b`, `plookup_cc` and `plookup_mul`; no public.
//!
//! Include it next to those two, as `#[path = ".../pilfflonk/tests/data/mixed_bus.rs"] mod
//! mixed_bus;`: it reads them as `super::connection` and `super::plookup`.

use proofman_pilfflonk::{AirInstanceRef, FrBytes, InstanceWitness, Stage1Witness, Witness};

use super::{connection, plookup};

/// The fixture's rows: `N = 2^8`.
pub const N_BITS: u32 = 8;

/// The witness of the fixture: one instance of air 0 of airgroup 0, and no public.
pub fn witness() -> Witness {
    generate(connection::columns(N_BITS), plookup::columns(N_BITS))
}

/// [`witness`], but with the Connection disconnected (`connection::disconnect`): the product bus
/// does not close, and the sum bus does.
pub fn witness_not_connected() -> Witness {
    let mut columns = connection::columns(N_BITS);
    connection::disconnect(&mut columns);
    generate(columns, plookup::columns(N_BITS))
}

/// [`witness`], but with the Plookup's wrong multiplicity (`plookup::move_a_multiplicity`): the sum
/// bus does not close, and the product bus does.
pub fn witness_with_a_wrong_multiplicity() -> Witness {
    let mut columns = plookup::columns(N_BITS);
    plookup::move_a_multiplicity(&mut columns[4]);
    generate(connection::columns(N_BITS), columns)
}

fn generate(connection: Vec<Vec<u64>>, plookup: Vec<Vec<u64>>) -> Witness {
    let n = 1usize << N_BITS;
    let columns: Vec<Vec<FrBytes>> =
        connection.iter().chain(&plookup).map(|col| col.iter().map(|&v| FrBytes::from_u64(v)).collect()).collect();
    let stage1 = Stage1Witness::from_columns(n, &columns, vec![]).expect("eight columns of n rows");
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 }],
        publics: vec![],
        proof_values: vec![],
    }
}
