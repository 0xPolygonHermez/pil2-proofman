use pilout::{
    pilout::{PilOut, Air},
    pilout_proxy::PilOutProxy,
};
use std::path::Path;
use std::error::Error;

#[derive(Debug)]
pub struct AirOut {
    pilout_proxy: PilOutProxy,
}

impl AirOut {
    /// Load from a Protobuf `.airout` file
    pub fn from_file(filename: &Path) -> Result<Self, Box<dyn Error>> {
        let pilout_proxy = PilOutProxy::new(filename.to_str().unwrap())?;
        Ok(Self { pilout_proxy })
    }

    /// Get a reference to the internal `PilOut`
    pub fn pilout(&self) -> &PilOut {
        &self.pilout_proxy.pilout
    }

    /// Expose existing methods from `PilOutProxy`
    pub fn get_airgroup_idx(&self, name: &str) -> Option<usize> {
        self.pilout_proxy.get_airgroup_idx(name)
    }

    pub fn get_air_idx(&self, airgroup_id: usize, name: &str) -> Option<usize> {
        self.pilout_proxy.get_air_idx(airgroup_id, name)
    }

    pub fn get_air(&self, airgroup_id: usize, air_id: usize) -> &Air {
        self.pilout_proxy.get_air(airgroup_id, air_id)
    }

    pub fn find_air(&self, air_group_name: &str, air_name: &str) -> Option<&Air> {
        self.pilout_proxy.find_air(air_group_name, air_name)
    }

    pub fn num_stages(&self) -> u32 {
        self.pilout_proxy.num_stages()
    }

    pub fn num_rows(&self, airgroup_id: usize, air_id: usize) -> usize {
        self.pilout_proxy.num_rows(airgroup_id, air_id)
    }

    pub fn print_pilout_info(&self) {
        self.pilout_proxy.print_pilout_info();
    }
}
