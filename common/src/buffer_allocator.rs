use std::{error::Error, path::Path};

pub trait BufferAllocator: Send + Sync {
    // Returns the size of the buffer and the offsets for each stage
    fn get_buffer_info(&self, air_pk_folder: &Path) -> Result<(u64, Vec<u64>), Box<dyn Error>>;
}
