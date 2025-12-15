use std::str::FromStr;
use fields::PrimeField64;

#[derive(Debug, Clone, PartialEq, Default)]
pub enum ProofType {
    #[default]
    Basic = 0,
    Compressor,
    Recursive1,
    Recursive2,
    VadcopFinal,
    RecursiveF,
}

impl ProofType {
    pub fn as_usize(&self) -> usize {
        match self {
            ProofType::Basic => 0,
            ProofType::Compressor => 1,
            ProofType::Recursive1 => 2,
            ProofType::Recursive2 => 3,
            ProofType::VadcopFinal => 4,
            ProofType::RecursiveF => 5,
        }
    }
}

impl From<ProofType> for &'static str {
    fn from(p: ProofType) -> Self {
        match p {
            ProofType::Basic => "basic",
            ProofType::Compressor => "compressor",
            ProofType::Recursive1 => "recursive1",
            ProofType::Recursive2 => "recursive2",
            ProofType::VadcopFinal => "vadcop_final",
            ProofType::RecursiveF => "recursive_f",
        }
    }
}

impl FromStr for ProofType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "basic" => Ok(ProofType::Basic),
            "compressor" => Ok(ProofType::Compressor),
            "recursive1" => Ok(ProofType::Recursive1),
            "recursive2" => Ok(ProofType::Recursive2),
            "vadcop_final" => Ok(ProofType::VadcopFinal),
            "recursive_f" => Ok(ProofType::RecursiveF),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Proof<F: PrimeField64> {
    pub proof_type: ProofType,
    pub airgroup_id: usize,
    pub air_id: usize,
    pub global_idx: Option<usize>,
    pub proof: Vec<u64>,
    pub circom_witness: Vec<F>,
    pub n_cols: usize,
}

impl<F: PrimeField64> Proof<F> {
    pub fn new(
        proof_type: ProofType,
        airgroup_id: usize,
        air_id: usize,
        global_idx: Option<usize>,
        proof: Vec<u64>,
    ) -> Self {
        Self { proof_type, global_idx, airgroup_id, air_id, proof, circom_witness: Vec::new(), n_cols: 0 }
    }

    pub fn new_witness(
        proof_type: ProofType,
        airgroup_id: usize,
        air_id: usize,
        global_idx: Option<usize>,
        circom_witness: Vec<F>,
        n_cols: usize,
    ) -> Self {
        Self { proof_type, global_idx, airgroup_id, air_id, circom_witness, proof: Vec::new(), n_cols }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProverInfo {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub air_instance_id: usize,
}
