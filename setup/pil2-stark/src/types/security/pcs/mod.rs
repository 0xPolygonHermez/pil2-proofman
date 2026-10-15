mod fri;
mod stir;
mod types;
mod whir;

pub use fri::{Fri, FriConfig, FriSecurityParams};
pub use stir::{Stir, StirConfig, StirSecurityParams};
pub use types::{Batching, Pcs};
pub use whir::{Whir, WhirConfig, WhirSecurityParams, whir_security_per_query};

/// The solved low-degree test of a proof: whichever PCS the stark struct selects.
#[derive(Clone, Debug)]
pub enum LowDegreeTest {
    Fri(Fri),
    Stir(Stir),
}

impl LowDegreeTest {
    fn pcs(&self) -> &dyn Pcs {
        match self {
            LowDegreeTest::Fri(fri) => fri,
            LowDegreeTest::Stir(stir) => stir,
        }
    }

    pub fn proximity_gap(&self) -> f64 {
        match self {
            LowDegreeTest::Fri(fri) => fri.proximity_gap(),
            LowDegreeTest::Stir(stir) => stir.proximity_gap(),
        }
    }

    pub fn proximity_parameter(&self) -> f64 {
        match self {
            LowDegreeTest::Fri(fri) => fri.proximity_parameter(),
            LowDegreeTest::Stir(stir) => stir.proximity_parameter(),
        }
    }

    /// The regime the security parameters were solved in.
    pub fn regime_identifier(&self) -> &'static str {
        match self {
            LowDegreeTest::Fri(fri) => fri.regime().identifier(),
            LowDegreeTest::Stir(stir) => stir.regime().identifier(),
        }
    }

    pub fn as_fri(&self) -> Option<&Fri> {
        match self {
            LowDegreeTest::Fri(fri) => Some(fri),
            _ => None,
        }
    }

    pub fn as_stir(&self) -> Option<&Stir> {
        match self {
            LowDegreeTest::Stir(stir) => Some(stir),
            _ => None,
        }
    }
}

impl Pcs for LowDegreeTest {
    fn identifier(&self) -> &'static str {
        self.pcs().identifier()
    }

    fn security_levels(&self) -> Vec<(String, u32)> {
        self.pcs().security_levels()
    }

    fn parameter_summary(&self) -> String {
        self.pcs().parameter_summary()
    }

    fn num_merkle_openings(&self) -> u64 {
        self.pcs().num_merkle_openings()
    }

    fn total_query_hashes(&self) -> f64 {
        self.pcs().total_query_hashes()
    }

    fn proof_size_bits(&self) -> u64 {
        self.pcs().proof_size_bits()
    }
}
