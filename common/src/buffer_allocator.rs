use std::error::Error;

use crate::SetupCtx;

pub trait BufferAllocator: Send + Sync {
    // Returns the size of the buffer and the offsets for each stage
    fn get_buffer_info(
        &self,
        sctx: &SetupCtx,
        airgroup_id: usize,
        air_id: usize,
    ) -> Result<(u64, Vec<u64>), Box<dyn Error>>;
}
