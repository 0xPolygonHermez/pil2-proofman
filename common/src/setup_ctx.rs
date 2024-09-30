use std::cell::OnceCell;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use crate::GlobalInfo;
use crate::Setup;
use crate::WitnessPilout;

#[derive(Debug)]
pub struct SetupRepository {
    // We store the setup in two stages: a partial setup in the first cell and a full setup in the second cell.
    // This allows for loading only the partial setup when constant polynomials are not needed, improving performance.
    // In C++, same SetupCtx structure is used to store either the partial or full setup for each instance.
    // A full setup can be loaded in one or two steps: partial first, then full (which includes constant polynomial data).
    // Since the setup is referenced immutably in the repository, we use OnceCell for both the partial and full setups.
    setups: HashMap<(usize, usize), (OnceCell<Setup>, OnceCell<Setup>)>, // (partial setup, full setup)
}

unsafe impl Send for SetupRepository {}
unsafe impl Sync for SetupRepository {}

impl SetupRepository {
    pub fn new(pilout: WitnessPilout) -> Self {
        let mut setups = HashMap::new();

        // Initialize Hashmao for each airgroup_id, air_id
        pilout.air_groups().iter().enumerate().for_each(|(airgroup_id, air_group)| {
            air_group.airs().iter().enumerate().for_each(|(air_id, _)| {
                setups.insert((airgroup_id, air_id), (OnceCell::new(), OnceCell::new()));
            });
        });

        Self { setups }
    }
}
/// Air instance context for managing air instances (traces)
#[allow(dead_code)]
pub struct SetupCtx {
    global_info: GlobalInfo,
    proving_key_path: PathBuf,

    setup_repository: SetupRepository,
}

impl SetupCtx {
    pub fn new(pilout: WitnessPilout, proving_key_path: &Path) -> Self {
        SetupCtx {
            global_info: GlobalInfo::new(proving_key_path),
            proving_key_path: proving_key_path.to_path_buf(),
            setup_repository: SetupRepository::new(pilout),
        }
    }

    pub fn get_setup(&self, airgroup_id: usize, air_id: usize) -> Result<&Setup, String> {
        let setup = self
            .setup_repository
            .setups
            .get(&(airgroup_id, air_id))
            .ok_or_else(|| format!("Setup not found for airgroup_id: {}, Air_id: {}", airgroup_id, air_id))?;

        if let Some(setup_ref) = setup.1.get() {
            Ok(setup_ref)
        } else if let Some(setup_ref) = setup.0.get() {
            let mut new_setup = setup_ref.clone();
            new_setup.load_const_pols(&self.proving_key_path, &self.global_info);
            setup.1.set(new_setup).unwrap();

            Ok(setup.1.get().unwrap())
        } else {
            let new_setup = Setup::new(&self.proving_key_path, &self.global_info, airgroup_id, air_id);
            setup.1.set(new_setup).unwrap();

            Ok(setup.1.get().unwrap())
        }
    }

    pub fn get_partial_setup(&self, airgroup_id: usize, air_id: usize) -> Result<&Setup, String> {
        let setup = self
            .setup_repository
            .setups
            .get(&(airgroup_id, air_id))
            .ok_or_else(|| format!("Setup not found for airgroup_id: {}, Air_id: {}", airgroup_id, air_id))?;

        if setup.0.get().is_some() {
            Ok(setup.0.get().unwrap())
        } else if setup.1.get().is_some() {
            Ok(setup.1.get().unwrap())
        } else {
            let new_setup = Setup::new_partial(&self.proving_key_path, &self.global_info, airgroup_id, air_id);
            setup.0.set(new_setup).unwrap();

            Ok(setup.0.get().unwrap())
        }
    }
}
