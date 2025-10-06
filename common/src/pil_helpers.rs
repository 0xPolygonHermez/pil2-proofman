pub trait PilHelpers {
    fn get_packed_words(&self, airgroup_id: usize, air_id: usize) -> Option<u64>;
}
