#[macro_export]
macro_rules! trace {
    ($row_struct_name:ident, $trace_struct_name:ident<$generic:ident> {
        $( $field_name:ident : $field_type:ty ),* $(,)?
    }) => {
        // Define the row structure (Main0RowTrace)
        #[allow(dead_code)]
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $row_struct_name<$generic> {
            $( pub $field_name: $field_type ),*
        }

        impl<$generic: Copy> $row_struct_name<$generic> {
            // The size of each row in terms of the number of fields
             pub const ROW_SIZE: usize = 0 $(+ trace!(@count_elements $field_type))*;
        }

        // Define the trace structure (Main0Trace) that manages the row structure
        pub struct $trace_struct_name<$generic> {
            pub slice: Box<[$row_struct_name<$generic>]>,
            num_rows: usize,
        }

        impl<$generic: Default + Clone + Copy> $trace_struct_name<$generic> {
            // Constructor for creating a new buffer
            pub fn new(num_rows: usize) -> Self {
                // PRECONDITIONS
                // num_rows must be greater than or equal to 2
                assert!(num_rows >= 2);
                // num_rows must be a power of 2
                assert!(num_rows & (num_rows - 1) == 0);

                let buffer = Box::new(vec![$row_struct_name::<$generic>::default(); num_rows]);
                let slice = buffer.into_boxed_slice();

                $trace_struct_name { slice, num_rows }
            }

            // Constructor to map over an external buffer
            pub fn from_buffer<'a>(external_buffer: &'a [$generic], num_rows: usize, offset: usize) -> Result<Self, Box<dyn std::error::Error>> {
                // PRECONDITIONS
                // num_rows must be greater than or equal to 2
                assert!(num_rows >= 2);
                // num_rows must be a power of 2
                assert!(num_rows & (num_rows - 1) == 0);

                let start = offset;
                let end = start + num_rows * $row_struct_name::<$generic>::ROW_SIZE;

                if end > external_buffer.len() {
                    return Err("Buffer is too small to fit the trace".into());
                }

                let slice = unsafe {
                    std::slice::from_raw_parts(
                        external_buffer[start..end].as_ptr() as *const $row_struct_name<$generic>,
                        num_rows,
                    )
                };

                Ok($trace_struct_name {
                    slice: slice.into(),
                    num_rows,
                })
            }

            pub fn num_rows(&self) -> usize {
                self.num_rows
            }
        }

        // Implement Index trait for immutable access
        impl<F> std::ops::Index<usize> for $trace_struct_name<$generic> {
            type Output = $row_struct_name<$generic>;

            fn index(&self, index: usize) -> &Self::Output {
                &self.slice[index]
            }
        }

        // Implement IndexMut trait for mutable access
        impl<F> std::ops::IndexMut<usize> for $trace_struct_name<F> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.slice[index]
            }
        }
    };

    (@count_elements [$elem_type:ty; $len:expr]) => {
        $len
    };

    (@count_elements $elem_type:ty) => {
        1
    };
}

#[cfg(test)]
mod tests {
    // use rand::Rng;

    #[test]
    fn check() {
        const OFFSET: usize = 1;
        let num_rows = 8;

        trace!(TraceRow, Trace<F> { a: F, b: F });

        let buffer = vec![0usize; num_rows * TraceRow::<usize>::ROW_SIZE + OFFSET];
        let trace = Trace::from_buffer(&buffer, num_rows, OFFSET);
        let mut trace = trace.unwrap();

        // Set values
        for i in 0..num_rows {
            trace[i].a = i;
            trace[i].b = i * 10;
        }

        // Check values
        for i in 0..num_rows {
            assert_eq!(trace[i].a, i);
            assert_eq!(trace[i].b, i * 10);
        }
    }

        #[test]
        #[should_panic]
        fn test_errors_are_launched_when_num_rows_is_invalid_1() {
            let buffer = vec![0u8; 3];
            trace!(SimpleRow, Simple<F> { a: F });
            let _ = Simple::from_buffer(&buffer, 1, 0);
        }

        #[test]
        #[should_panic]
        fn test_errors_are_launched_when_num_rows_is_invalid_2() {
            let buffer = vec![0u8; 3];
            trace!(SimpleRow, Simple<F> { a: F });
            let _ = Simple::from_buffer(&buffer, 3, 0);
        }

        #[test]
        #[should_panic]
        fn test_errors_are_launched_when_num_rows_is_invalid_3() {
            trace!(SimpleRow, Simple<F> { a: F });
            let _ = Simple::<u8>::new(1);
        }

        #[test]
        #[should_panic]
        fn test_errors_are_launched_when_num_rows_is_invalid_4() {
            trace!(SimpleRow, Simple<F> { a: F });
            let _ = Simple::<u8>::new(3);
        }
}
