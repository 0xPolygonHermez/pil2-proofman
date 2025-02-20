use std::{collections::HashMap, sync::RwLock};
use rayon::prelude::*;

use p3_field::Field;

use crate::AirInstance;

pub struct AirInstancesRepository<F> {
    pub air_instances: RwLock<HashMap<usize, AirInstance<F>>>,
}

impl<F: Field> Default for AirInstancesRepository<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> AirInstancesRepository<F> {
    pub fn new() -> Self {
        AirInstancesRepository { air_instances: RwLock::new(HashMap::new()) }
    }

    pub fn add_air_instance(&self, air_instance: AirInstance<F>, global_idx: usize) {
        let mut air_instances = self.air_instances.write().unwrap();
        air_instances.insert(global_idx, air_instance);
    }

    pub fn free(&self) {
        let mut air_instances = self.air_instances.write().unwrap();
        air_instances.clear();
    }

    pub fn free_traces(&self) {
        let mut air_instances = self.air_instances.write().unwrap();
        air_instances.par_iter_mut().for_each(|(_, air_instance)| {
            air_instance.clear_trace();
            air_instance.clear_custom_commits_fixed_trace();
        });
    }

    pub fn find_airgroup_instances(&self, airgroup_id: usize) -> Vec<usize> {
        let air_instances = self.air_instances.read().unwrap();

        let mut indices = Vec::new();
        for (index, air_instance) in air_instances.iter() {
            if air_instance.airgroup_id == airgroup_id {
                indices.push(*index);
            }
        }
        indices
    }

    pub fn find_air_instances(&self, airgroup_id: usize, air_id: usize) -> Vec<usize> {
        let air_instances = self.air_instances.read().unwrap();

        let mut indices = Vec::new();
        for (index, air_instance) in air_instances.iter() {
            if air_instance.airgroup_id == airgroup_id && air_instance.air_id == air_id {
                indices.push(*index);
            }
        }

        indices
    }
}
