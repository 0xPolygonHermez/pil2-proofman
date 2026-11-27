use proofman_fields::PrimeField64;
use proofman_starks_lib_c::{get_n_constraints_c, get_constraints_lines_sizes_c, get_constraints_lines_c};

use crate::{ProofmanResult, SetupCtx};

#[derive(Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ConstraintRowInfo {
    pub row: u64,
    pub dim: u64,
    pub value: [u64; 3usize],
}

#[derive(Default, Debug, Clone)]
#[repr(C)]
pub struct ConstraintInfo {
    pub id: u64,
    pub stage: u64,
    pub im_pol: bool,
    pub n_rows: u64,
    pub skip: bool,
    pub n_print_constraints: u64,
    pub rows: Vec<ConstraintRowInfo>,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct ConstraintInfoC {
    pub id: u64,
    pub stage: u64,
    pub im_pol: bool,
    pub n_rows: u64,
    pub skip: bool,
    pub n_print_constraints: u64,
    pub rows: *mut ConstraintRowInfo,
}

impl ConstraintInfo {
    pub fn new(n_print_constraints: u64) -> Self {
        Self {
            rows: vec![ConstraintRowInfo::default(); n_print_constraints as usize],
            n_print_constraints,
            ..Default::default()
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct GlobalConstraintInfo {
    pub id: u64,
    pub dim: u64,
    pub valid: bool,
    pub skip: bool,
    pub value: [u64; 3usize],
}

pub fn get_constraints_lines_str<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    airgroup_id: usize,
    air_id: usize,
) -> ProofmanResult<Vec<String>> {
    let setup = sctx.get_setup(airgroup_id, air_id)?;

    let p_setup = (&setup.p_setup).into();
    let n_constraints = get_n_constraints_c(p_setup);

    let mut constraints_sizes = vec![0u64; n_constraints as usize];

    get_constraints_lines_sizes_c(p_setup, constraints_sizes.as_mut_ptr());

    let mut constraints_lines = vec![Vec::new(); n_constraints as usize];
    for i in 0..n_constraints as usize {
        constraints_lines[i] = vec![0u8; constraints_sizes[i] as usize];
    }

    get_constraints_lines_c(
        p_setup,
        constraints_lines.iter_mut().map(|v| v.as_mut_ptr()).collect::<Vec<_>>().as_mut_ptr(),
    );

    let mut constraints_lines_str = Vec::new();
    for constraint_line in constraints_lines {
        constraints_lines_str.push(std::str::from_utf8(&constraint_line).unwrap().to_string());
    }

    Ok(constraints_lines_str)
}
