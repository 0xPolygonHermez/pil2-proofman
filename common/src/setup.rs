use std::{os::raw::c_void, path::Path, ptr::null_mut};

use log::info;

use proofman_starks_lib_c::{const_pols_new_c, expressions_bin_new_c, stark_info_new_c};

use crate::GlobalInfo;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct SetupC {
    pub p_stark_info: *mut c_void,
    pub p_expressions_bin: *mut c_void,
    pub p_const_pols: *mut c_void,
}

unsafe impl Send for SetupC {}
unsafe impl Sync for SetupC {}

impl From<&SetupC> for *mut c_void {
    fn from(setup: &SetupC) -> *mut c_void {
        setup as *const SetupC as *mut c_void
    }
}

/// Air instance context for managing air instances (traces)
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Setup {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub p_setup: SetupC,
}

impl Setup {
    const MY_NAME: &'static str = "Setup";

    pub fn new(proving_key_path: &Path, global_info: &GlobalInfo, airgroup_id: usize, air_id: usize) -> Self {
        let air_setup_folder = proving_key_path.join(global_info.get_air_setup_path(airgroup_id, air_id));

        // Check path exists and is a folder
        if !air_setup_folder.exists() {
            panic!("Setup AIR folder not found at path: {:?}", air_setup_folder);
        }
        if !air_setup_folder.is_dir() {
            panic!("Setup AIR path is not a folder: {:?}", air_setup_folder);
        }

        let base_filename_path =
            air_setup_folder.join(global_info.get_air_name(airgroup_id, air_id)).display().to_string();

        info!("{}   : ··· Loading setup for AIR [{}:{}]: {:?}", Self::MY_NAME, airgroup_id, air_id, air_setup_folder);

        let stark_info_path = base_filename_path.clone() + ".starkinfo.json";
        let expressions_bin_path = base_filename_path.clone() + ".bin";
        let const_pols_path = base_filename_path.clone() + ".const";

        let p_stark_info = stark_info_new_c(stark_info_path.as_str());
        let p_expressions_bin = expressions_bin_new_c(expressions_bin_path.as_str());
        let p_const_pols = const_pols_new_c(const_pols_path.as_str(), p_stark_info);

        Self { air_id, airgroup_id, p_setup: SetupC { p_stark_info, p_expressions_bin, p_const_pols } }
    }

    pub fn new_partial(proving_key_path: &Path, global_info: &GlobalInfo, airgroup_id: usize, air_id: usize) -> Self {
        let air_setup_folder = proving_key_path.join(global_info.get_air_setup_path(airgroup_id, air_id));

        // Check path exists and is a folder
        if !air_setup_folder.exists() {
            panic!("Setup AIR folder not found at path: {:?}", air_setup_folder);
        }
        if !air_setup_folder.is_dir() {
            panic!("Setup AIR path is not a folder: {:?}", air_setup_folder);
        }

        let base_filename_path =
            air_setup_folder.join(global_info.get_air_name(airgroup_id, air_id)).display().to_string();

        info!("{}   : ··· Loading setup for AIR [{}:{}]: {:?}", Self::MY_NAME, airgroup_id, air_id, air_setup_folder);

        let stark_info_path = base_filename_path.clone() + ".starkinfo.json";
        let expressions_bin_path = base_filename_path.clone() + ".bin";

        let p_stark_info = stark_info_new_c(stark_info_path.as_str());
        let p_expressions_bin = expressions_bin_new_c(expressions_bin_path.as_str());

        Self { air_id, airgroup_id, p_setup: SetupC { p_stark_info, p_expressions_bin, p_const_pols: null_mut() } }
    }

    pub fn load_const_pols(&mut self, proving_key_path: &Path, global_info: &GlobalInfo) {
        if !self.p_setup.p_const_pols.is_null() {
            return;
        }
        assert!(!self.p_setup.p_stark_info.is_null());
        assert!(!self.p_setup.p_expressions_bin.is_null());

        let air_setup_folder = proving_key_path.join(global_info.get_air_setup_path(self.airgroup_id, self.air_id));

        // Check path exists and is a folder
        if !air_setup_folder.exists() {
            panic!("Setup AIR folder not found at path: {:?}", air_setup_folder);
        }
        if !air_setup_folder.is_dir() {
            panic!("Setup AIR path is not a folder: {:?}", air_setup_folder);
        }

        let base_filename_path =
            air_setup_folder.join(global_info.get_air_name(self.airgroup_id, self.air_id)).display().to_string();

        info!(
            "{}   : ··· Loading const pols for AIR [{}:{}]: {:?}",
            Self::MY_NAME,
            self.airgroup_id,
            self.air_id,
            air_setup_folder
        );

        let const_pols_path = base_filename_path.clone() + ".const";

        let p_const_pols = const_pols_new_c(const_pols_path.as_str(), self.p_setup.p_stark_info);

        self.p_setup.p_const_pols = p_const_pols;
    }
}
