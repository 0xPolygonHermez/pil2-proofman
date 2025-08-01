use crate::pilout::PilOut;
use prost::{DecodeError, Message};

use std::fs::File;
use std::io::Read;
use std::ops::Deref;

#[derive(Debug, Default)]
pub struct PilOutProxy {
    pub pilout: PilOut,
}

impl PilOutProxy {
    pub fn new(pilout_filename: &str) -> Result<PilOutProxy, Box<dyn std::error::Error>> {
        let pilout = Self::load_pilout(pilout_filename)?;
        Ok(PilOutProxy { pilout })
    }

    fn load_pilout(pilout_filename: &str) -> Result<PilOut, DecodeError> {
        tracing::debug!("··· Loading pilout");

        // Open the file
        let mut file = File::open(pilout_filename).unwrap_or_else(|error| {
            panic!("Failed to open file {pilout_filename}: {error}");
        });

        // Read the file content into a Vec<u8>
        let mut file_content = Vec::new();
        if let Err(e) = file.read_to_end(&mut file_content) {
            panic!("Failed to read file content {pilout_filename}: {e}");
        }

        // Parse the protobuf message
        let result = PilOut::decode(file_content.as_slice());

        result
    }

    pub fn get_airgroup_idx(&self, name: &str) -> Option<usize> {
        self.pilout.air_groups.iter().position(|x| x.name.as_deref() == Some(name))
    }

    pub fn get_air_idx(&self, airgroup_id: usize, name: &str) -> Option<usize> {
        self.pilout.air_groups[airgroup_id].airs.iter().position(|x| x.name.as_deref() == Some(name))
    }

    pub fn get_air(&self, airgroup_id: usize, air_id: usize) -> &crate::pilout::Air {
        &self.pilout.air_groups[airgroup_id].airs[air_id]
    }

    pub fn find_air(&self, air_group_name: &str, air_name: &str) -> Option<&crate::pilout::Air> {
        let airgroup_id = self.get_airgroup_idx(air_group_name)?;
        let air_id = self.get_air_idx(airgroup_id, air_name)?;
        Some(&self.pilout.air_groups[airgroup_id].airs[air_id])
    }

    pub fn num_stages(&self) -> u32 {
        if self.pilout.num_challenges.is_empty() {
            1
        } else {
            self.pilout.num_challenges.len() as u32
        }
    }

    pub fn num_rows(&self, airgroup_id: usize, air_id: usize) -> usize {
        self.pilout.air_groups[airgroup_id].airs[air_id].num_rows.unwrap() as usize
    }

    pub fn name(&self, airgroup_id: usize, air_id: usize) -> &str {
        self.pilout.air_groups[airgroup_id].airs[air_id].name.as_ref().unwrap()
    }

    pub fn print_pilout_info(&self) {
        // Print PilOut airgroups and airs names and degrees
        tracing::trace!("··· '{}' PilOut info", self.name.as_ref().unwrap());

        let base_field: &Vec<u8> = self.pilout.base_field.as_ref();
        let mut hex_string = "0x".to_owned();
        for &byte in base_field {
            hex_string.push_str(&format!("{byte:02X}"));
        }
        tracing::trace!("    Base field: {}", hex_string);

        tracing::trace!("    Airgroups:");
        for (airgroup_index, airgroup) in self.pilout.air_groups.iter().enumerate() {
            tracing::trace!(
                "    + [{}] {} (airgroup values: {})",
                airgroup_index,
                airgroup.name.as_ref().unwrap(),
                airgroup.air_group_values.len()
            );

            for (air_index, air) in self.pilout.air_groups[airgroup_index].airs.iter().enumerate() {
                tracing::trace!(
                    "      [{}][{}] {} (rows: {}, fixed cols: {}, periodic cols: {}, stage widths: {}, expressions: {}, constraints: {})",
                    airgroup_index,
                    air_index,
                    air.name.as_ref().unwrap(),
                    air.num_rows.unwrap(),
                    air.fixed_cols.len(),
                    air.periodic_cols.len(),
                    air.stage_widths.len(),
                    air.expressions.len(),
                    air.constraints.len()
                );
            }
        }

        tracing::trace!("    Challenges: {}", self.pilout.num_challenges.len());
        for i in 0..self.pilout.num_challenges.len() {
            tracing::trace!("      stage {}: {}", i, self.pilout.num_challenges[i]);
        }

        tracing::trace!(
            "    #Proof values: {}, #Public values: {}, #Global expressions: {}, #Global constraints: {}",
            self.pilout.num_proof_values.len(),
            self.pilout.num_public_values,
            self.pilout.expressions.len(),
            self.pilout.constraints.len()
        );
        tracing::trace!("    #Hints: {}, #Symbols: {}", self.pilout.hints.len(), self.pilout.symbols.len());
        tracing::trace!("    Public tables: {}", self.pilout.public_tables.len());
    }
}

impl Deref for PilOutProxy {
    type Target = PilOut;

    fn deref(&self) -> &Self::Target {
        &self.pilout
    }
}
