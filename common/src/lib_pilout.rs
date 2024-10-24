#[derive(Debug)]
pub struct WitnessPilout {
    name: String,
    hash: Vec<u8>,
    num_stages: u32,
    air_groups: Vec<AirGroup>,
}

impl WitnessPilout {
    pub fn new(name: &str, num_stages: u32, hash: Vec<u8>) -> Self {
        WitnessPilout { name: name.to_string(), num_stages, air_groups: Vec::new(), hash }
    }

    pub fn add_air_group(&mut self, air_group_name: Option<&str>) -> &mut AirGroup {
        let airgroup_id = self.air_groups.len();
        let air_group = AirGroup::new(airgroup_id, air_group_name);
        self.air_groups.push(air_group);
        &mut self.air_groups[airgroup_id]
    }

    pub fn find_air(&self, airgroup_name: &str, air_name: &str) -> Option<&Air> {
        for airgroup in &self.air_groups {
            if let Some(name) = &airgroup.name {
                if name == airgroup_name {
                    for air in &airgroup.airs {
                        if let Some(name) = &air.name {
                            if name == air_name {
                                return Some(air);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    pub fn get_air_group(&self, airgroup_id: usize) -> &AirGroup {
        &self.air_groups[airgroup_id]
    }

    pub fn get_air(&self, airgroup_id: usize, air_id: usize) -> &Air {
        &self.air_groups[airgroup_id].airs[air_id]
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn hash(&self) -> &[u8] {
        &self.hash
    }

    pub fn num_stages(&self) -> u32 {
        self.num_stages
    }

    pub fn air_groups(&self) -> &[AirGroup] {
        &self.air_groups
    }
}

#[derive(Debug)]
pub struct AirGroup {
    airgroup_id: usize,
    name: Option<String>,
    airs: Vec<Air>,
}

impl AirGroup {
    pub fn new(airgroup_id: usize, name: Option<&str>) -> Self {
        AirGroup { airgroup_id, name: name.map(|s| s.to_string()), airs: Vec::new() }
    }

    pub fn add_air(&mut self, air_name: Option<&str>, num_rows: usize) -> &Air {
        let air_id = self.airs.len();
        let air = Air::new(self.airgroup_id, air_id, air_name, num_rows);
        self.airs.push(air);
        &self.airs[air_id]
    }

    pub fn airgroup_id(&self) -> usize {
        self.airgroup_id
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn airs(&self) -> &[Air] {
        &self.airs
    }
}

#[derive(Debug)]
pub struct Air {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub name: Option<String>,
    /// log2(n), where n is the number of rows
    num_rows: usize,
}

impl Air {
    pub fn new(airgroup_id: usize, air_id: usize, name: Option<&str>, num_rows: usize) -> Self {
        Air { airgroup_id, air_id, name: name.map(|s| s.to_string()), num_rows }
    }

    pub fn airgroup_id(&self) -> usize {
        self.airgroup_id
    }

    pub fn air_id(&self) -> usize {
        self.air_id
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }
}
