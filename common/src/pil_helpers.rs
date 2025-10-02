pub trait PilHelpers {
    fn get_packed_words(&self, airgroup_id: usize, air_id: usize) -> Option<u64>;

    fn get_unpack_info(&self, airgroup_id: usize, air_id: usize) -> Option<Vec<u64>>;
}