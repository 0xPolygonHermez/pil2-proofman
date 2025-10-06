mod trace2;

#[cfg(test)]
mod tests {
    use proofman_macros::trace;
    use proofman_common as common;

    trace!(SampleTrace<F> { field1: F, field2: [F; 2] },  0, 0, 32 );

    // #[test]
    // fn split_rows_works() {
    //     let trace = SampleTrace::<i32>::new();

    //     let sizes = [8u32, 3, 1, 2, 4, 5, 6, 3];
    //     assert_eq!(sizes.iter().sum::<u32>() as usize, trace.num_rows());

    //     let split_trace = trace.to_split_struct(&[8, 3, 1, 2, 4, 5, 6, 3]);
    //     assert_eq!(split_trace.chunks.len(), sizes.len());

    //     let trace = SampleTrace::<i32>::from_split_struct(split_trace);

    //     assert_eq!(SampleTrace::<i32>::NUM_ROWS, trace.num_rows());
    // }

    // #[test]
    // fn split_data_works() {
    //     let mut trace = SampleTrace::<i32>::new();

    //     for i in 0..SampleTrace::<i32>::NUM_ROWS {
    //         let val = i as i32 * 3;
    //         trace[i].field1 = val;
    //         trace[i].field2 = [val + 1, val + 2];
    //     }

    //     let sizes = [8u32, 3, 1, 2, 4, 5, 6, 3];
    //     assert_eq!(sizes.iter().sum::<u32>() as usize, trace.num_rows());

    //     let split_trace = trace.to_split_struct(&[8, 3, 1, 2, 4, 5, 6, 3]);

    //     let mut i: i32 = 0;
    //     for split in split_trace.chunks {
    //         for j in 0..split.len() {
    //             let val = i * 3;
    //             assert_eq!(val, split[j].field1);
    //             assert_eq!(val + 1, split[j].field2[0]);
    //             assert_eq!(val + 2, split[j].field2[1]);
    //             i += 1;
    //         }
    //     }
    // }
}
