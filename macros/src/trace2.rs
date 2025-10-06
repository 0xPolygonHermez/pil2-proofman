// use rayon::prelude::*;

// pub trait TraceRow: Copy + Default + Send {
//     const ROW_SIZE: usize;
// }

// pub struct GenericTrace<
//     F,
//     R: TraceRow,
//     const NUM_ROWS: usize,
//     const AIRGROUP_ID: usize,
//     const AIR_ID: usize,
//     const COMMIT_ID: usize = 0,
// > {
//     pub buffer: Vec<R>,
//     pub shared_buffer: bool,
//     _phantom: std::marker::PhantomData<F>,
// }

// impl<
//         F: Default + Clone + Copy + Send,
//         R: TraceRow,
//         const NUM_ROWS: usize,
//         const AIRGROUP_ID: usize,
//         const AIR_ID: usize,
//         const COMMIT_ID: usize,
//     > GenericTrace<F, R, NUM_ROWS, AIRGROUP_ID, AIR_ID, COMMIT_ID>
// {
//     pub fn new() -> Self {
//         GenericTrace::with_capacity(NUM_ROWS)
//     }
//     pub fn new_zeroes() -> Self {
//         let num_rows = NUM_ROWS;

//         assert!(num_rows >= 2);
//         assert!(num_rows & (num_rows - 1) == 0);

//         let buffer: Vec<R> = vec![R::default(); num_rows];

//         Self { buffer, shared_buffer: false, _phantom: std::marker::PhantomData }
//     }

//     pub fn with_capacity(num_rows: usize) -> Self {
//         assert!(num_rows >= 2);
//         assert!(num_rows & (num_rows - 1) == 0);

//         let mut vec: Vec<std::mem::MaybeUninit<R>> = Vec::with_capacity(num_rows);
//         let buffer: Vec<R> = unsafe {
//             vec.set_len(num_rows);
//             std::mem::transmute(vec)
//         };

//         #[cfg(feature = "diagnostic")]
//         unsafe {
//             let mut ptr = buffer.as_mut_ptr() as *mut u64;
//             let expected_len = num_rows;
//             for _ in 0..expected_len * R::ROW_SIZE {
//                 ptr.write(u64::MAX - 1);
//                 ptr = ptr.add(1);
//             }
//         }

//         Self { buffer, shared_buffer: false, _phantom: std::marker::PhantomData }
//     }

//     pub fn new_from_vec_zeroes(mut buffer: Vec<F>) -> Self {
//         let row_size = R::ROW_SIZE;
//         let num_rows = NUM_ROWS;
//         let used_len = num_rows * row_size;

//         assert!(
//             buffer.len() >= used_len,
//             "Provided buffer too small: got {}, expected at least {}",
//             buffer.len(),
//             used_len
//         );

//         buffer[..used_len].par_iter_mut().for_each(|x| {
//             *x = <F>::default();
//         });

//         let ptr = buffer.as_mut_ptr();
//         std::mem::forget(buffer);
//         let buffer = unsafe { Vec::from_raw_parts(ptr as *mut R, num_rows, num_rows) };

//         Self { buffer, shared_buffer: true, _phantom: std::marker::PhantomData }
//     }
//     pub fn new_from_vec(mut buffer: Vec<F>) -> Self {
//         let row_size = R::ROW_SIZE;
//         let num_rows = NUM_ROWS;
//         let expected_len = num_rows * row_size;

//         assert!(buffer.len() >= expected_len, "Flat buffer too small");
//         assert!(num_rows >= 2);
//         assert!(num_rows & (num_rows - 1) == 0);

//         if cfg!(feature = "diagnostic") {
//             unsafe {
//                 let mut ptr = buffer.as_mut_ptr() as *mut u64;
//                 let expected_len = num_rows;
//                 for _ in 0..expected_len * R::ROW_SIZE {
//                     ptr.write(u64::MAX - 1);
//                     ptr = ptr.add(1);
//                 }
//             }
//         }

//         let ptr = buffer.as_mut_ptr();
//         std::mem::forget(buffer);
//         let buffer = unsafe { Vec::from_raw_parts(ptr as *mut R, num_rows, num_rows) };

//         Self { buffer, shared_buffer: true, _phantom: std::marker::PhantomData }
//     }

//     pub fn from_vec(mut buffer: Vec<F>) -> Self {
//         let row_size = R::ROW_SIZE;
//         let num_rows = NUM_ROWS;
//         let expected_len = num_rows * row_size;

//         assert!(buffer.len() >= expected_len, "Flat buffer too small");
//         assert!(num_rows >= 2);
//         assert!(num_rows & (num_rows - 1) == 0);

//         let ptr = buffer.as_mut_ptr();
//         std::mem::forget(buffer);
//         let buffer = unsafe { Vec::from_raw_parts(ptr as *mut R, num_rows, num_rows) };

//         Self { buffer, shared_buffer: true, _phantom: std::marker::PhantomData }
//     }

//     pub fn par_iter_mut_chunks(&mut self, n: usize) -> impl IndexedParallelIterator<Item = &mut [R]> {
//         assert!(n > 0 && (n & (n - 1)) == 0, "n must be a power of two");
//         assert!(n <= NUM_ROWS, "n must be less than or equal to NUM_ROWS");
//         let chunk_size = NUM_ROWS / n;
//         assert!(chunk_size > 0, "Chunk size must be greater than zero");
//         self.buffer.par_chunks_mut(chunk_size)
//     }

//     pub fn get_buffer(&mut self) -> Vec<F> {
//         let mut buffer = std::mem::take(&mut self.buffer);

//         let new_len = NUM_ROWS * R::ROW_SIZE;
//         let ptr = buffer.as_mut_ptr();
//         std::mem::forget(buffer);
//         unsafe { Vec::from_raw_parts(ptr as *mut F, new_len, new_len) }
//     }

//     pub fn is_shared_buffer(&self) -> bool {
//         self.shared_buffer
//     }

//     pub const fn num_rows(&self) -> usize {
//         NUM_ROWS
//     }

//     pub const fn airgroup_id(&self) -> usize {
//         AIRGROUP_ID
//     }

//     pub const fn air_id(&self) -> usize {
//         AIR_ID
//     }

//     pub const fn row_size(&self) -> usize {
//         R::ROW_SIZE
//     }

//     pub const fn num_cols(&self) -> usize {
//         R::ROW_SIZE
//     }

//     pub const fn commit_id(&self) -> Option<usize> {
//         // Return the commit ID if it's not zero
//         if COMMIT_ID == 0 {
//             None
//         } else {
//             Some(COMMIT_ID)
//         }
//     }
// }

// impl<
//         F: Default + Clone + Copy + Send,
//         R: TraceRow,
//         const NUM_ROWS: usize,
//         const AIRGROUP_ID: usize,
//         const AIR_ID: usize,
//         const COMMIT_ID: usize,
//     > std::ops::Index<usize> for GenericTrace<F, R, NUM_ROWS, AIRGROUP_ID, AIR_ID, COMMIT_ID>
// {
//     type Output = R;

//     fn index(&self, index: usize) -> &Self::Output {
//         &self.buffer[index]
//     }
// }

// impl<
//         F: Default + Clone + Copy + Send,
//         R: TraceRow,
//         const NUM_ROWS: usize,
//         const AIRGROUP_ID: usize,
//         const AIR_ID: usize,
//     > std::ops::IndexMut<usize> for GenericTrace<F, R, NUM_ROWS, AIRGROUP_ID, AIR_ID>
// {
//     fn index_mut(&mut self, index: usize) -> &mut Self::Output {
//         &mut self.buffer[index]
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     // A simple TraceRow implementation for tests
//     #[derive(Copy, Clone, Default)]
//     struct SampleTestRow;
//     impl TraceRow for SampleTestRow {
//         const ROW_SIZE: usize = 4;
//     }

//     // Helper to build a flat buffer of F (u64 here) with sequential numbers
//     fn make_buffer(len: usize) -> Vec<u64> {
//         (0..len as u64).collect()
//     }

//     type SampleTrace = GenericTrace<u64, SampleTestRow, 16, 2, 7>; // 16 rows, row_size 4 => flat 64 elements

//     #[test]
//     fn new_zeroes_initializes_rows() {
//         let t = SampleTrace::new_zeroes();

//         assert_eq!(t.num_rows(), 16);
//         assert_eq!(t.row_size(), SampleTestRow::ROW_SIZE);
//         assert!(!t.is_shared_buffer());
//         assert_eq!(t.buffer.len(), 16);
//     }

//     #[test]
//     fn with_capacity_uninitialized_layout() {
//         let t = SampleTrace::new();
//         assert_eq!(t.buffer.len(), 16);
//     }

//     #[test]
//     fn new_from_vec_zeroes_sets_all_zero_and_marks_shared() {
//         // Provide larger buffer than needed to ensure slicing works
//         let buf = vec![123u64; 16 * SampleTestRow::ROW_SIZE + 10];
//         let mut t = GenericTrace::<u64, SampleTestRow, 16, 0, 0>::new_from_vec_zeroes(buf.clone());
//         assert!(t.is_shared_buffer());
//         assert_eq!(t.buffer.len(), 16);
//         // Convert back to flat representation safely via get_buffer()
//         let flat = t.get_buffer();
//         assert_eq!(flat.len(), 16 * SampleTestRow::ROW_SIZE);
//         assert!(flat.iter().all(|&x| x == 0), "expected all zeroes after zero-initialization");
//     }

//     #[test]
//     fn new_from_vec_keeps_shared_flag() {
//         let flat_len = 16 * SampleTestRow::ROW_SIZE;
//         let buf = make_buffer(flat_len);
//         let t = GenericTrace::<u64, SampleTestRow, 16, 2, 7>::new_from_vec(buf);
//         assert!(t.is_shared_buffer());
//         assert_eq!(t.airgroup_id(), 2);
//         assert_eq!(t.air_id(), 7);
//         assert_eq!(t.commit_id(), None);
//     }

//     #[test]
//     fn from_vec_alias() {
//         let flat_len = 16 * SampleTestRow::ROW_SIZE;
//         let buf = make_buffer(flat_len);
//         let t = GenericTrace::<u64, SampleTestRow, 16, 0, 0>::from_vec(buf);
//         assert!(t.is_shared_buffer());
//     }

//     #[test]
//     fn par_iter_mut_chunks_power_of_two_partitions() {
//         let mut t = SampleTrace::new_zeroes();
//         // Write distinct sentinel per chunk using 4 chunks
//         t.par_iter_mut_chunks(4).enumerate().for_each(|(i, chunk)| {
//             for row in chunk.iter_mut() {
//                 *row = SampleTestRow;
//             }
//             // store sentinel by writing to first element of chunk if we could
//             // (We cannot access internal fields of row; this test ensures iteration doesn't panic.)
//             assert!(!chunk.is_empty());
//             assert!(i < 4);
//         });
//     }

//     #[test]
//     #[should_panic]
//     fn par_iter_mut_chunks_panics_non_power_of_two() {
//         let mut t = SampleTrace::new_zeroes();
//         let _ = t.par_iter_mut_chunks(3); // not power of two
//     }

//     #[test]
//     #[should_panic]
//     fn par_iter_mut_chunks_panics_too_large() {
//         let mut t = SampleTrace::new_zeroes();
//         let _ = t.par_iter_mut_chunks(32); // greater than NUM_ROWS
//     }

//     #[test]
//     fn get_buffer_converts_back_to_flat() {
//         let flat_len = 16 * SampleTestRow::ROW_SIZE;
//         let buf = make_buffer(flat_len);
//         let mut t = GenericTrace::<u64, SampleTestRow, 16, 0, 0>::new_from_vec(buf.clone());
//         let recovered = t.get_buffer();
//         assert_eq!(recovered.len(), flat_len);
//         // We can't guarantee ordering semantics without knowing row representation layout, but we can at least
//         // check capacity/length match expectation.
//         assert_eq!(recovered.capacity(), flat_len);
//     }
// }
