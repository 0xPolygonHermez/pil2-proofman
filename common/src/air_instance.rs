use crate::AirInstanceCtx;

#[derive(Debug)]
pub struct AirInstance {
    pub air_group_id: usize,
    pub air_id: usize,
    pub inputs_interval: Option<(usize, usize)>,
}

impl AirInstance {
    pub fn new(air_group_id: usize, air_id: usize, inputs_interval: Option<(usize, usize)>) -> Self {
        AirInstance { air_group_id, air_id, inputs_interval }
    }
}

impl From<&AirInstance> for AirInstanceCtx {
    fn from(air_instance: &AirInstance) -> Self {
        AirInstanceCtx::new(air_instance.air_group_id, air_instance.air_id)
    }
}
