macro_rules! from_integer_types {
    ($($type:ty),* $(,)? ) => {
        $( paste::paste!{
            fn [<from_ $type>](int: $type) -> Self {
                Self::from_int(int)
            }
        }
        )*
    };
}

pub(crate) use from_integer_types;

pub trait QuotientMap<Int>: Sized {
    fn from_int(int: Int) -> Self;

    fn from_canonical_checked(int: Int) -> Option<Self>;

    /// # Safety
    unsafe fn from_canonical_unchecked(int: Int) -> Self;
}

#[macro_export]
macro_rules! quotient_map_small_internals {
    ($field:ty, $field_size:ty, $small_int:ty) => {
        #[doc = concat!("Convert a given `", stringify!($small_int), "` integer into an element of the `", stringify!($field), "` field.
        \n Due to the integer type, the input value is always canonical.")]
        #[inline]
        fn from_int(int: $small_int) -> Self {
            // Should be removed by the compiler.
            assert!(size_of::<$small_int>() < size_of::<$field_size>());
            unsafe {
                Self::from_canonical_unchecked(int as $field_size)
            }
        }

        #[doc = concat!("Convert a given `", stringify!($small_int), "` integer into an element of the `", stringify!($field), "` field.
        \n Due to the integer type, the input value is always canonical.")]
        #[inline]
        fn from_canonical_checked(int: $small_int) -> Option<Self> {
            // Should be removed by the compiler.
            assert!(size_of::<$small_int>() < size_of::<$field_size>());
            Some(unsafe {
                Self::from_canonical_unchecked(int as $field_size)
            })
        }

        #[doc = concat!("Convert a given `", stringify!($small_int), "` integer into an element of the `", stringify!($field), "` field.
        \n Due to the integer type, the input value is always canonical.")]
        #[inline]
        unsafe fn from_canonical_unchecked(int: $small_int) -> Self {
            // We use debug_assert to ensure this is removed by the compiler in release mode.
            debug_assert!(size_of::<$small_int>() < size_of::<$field_size>());
            unsafe {
                Self::from_canonical_unchecked(int as $field_size)
            }
        }
    };
}

#[macro_export]
macro_rules! quotient_map_small_int {
    ($field:ty, $field_size:ty, [$($small_int:ty),*] ) => {
        $(
        paste::paste!{
            impl QuotientMap<$small_int> for $field {
                $crate::quotient_map_small_internals!($field, $field_size, $small_int);
            }
        }
        )*
    };

    ($field:ty, $field_size:ty, $field_param:ty, [$($small_int:ty),*] ) => {
        $(
        paste::paste!{
            impl<FP: $field_param> QuotientMap<$small_int> for $field<FP> {
                $crate::quotient_map_small_internals!($field, $field_size, $small_int);
            }
        }
        )*
    };
}

macro_rules! impl_u_i_size {
    ($intsize:ty, $int8:ty, $int16:ty, $int32:ty, $int64:ty) => {
        impl<
                F: QuotientMap<$int8>
                    + QuotientMap<$int16>
                    + QuotientMap<$int32>
                    + QuotientMap<$int64>,
            > QuotientMap<$intsize> for F
        {
            #[doc = concat!("We use the `from_int` method of the primitive integer type identical to `", stringify!($intsize), "` on this machine")]
            fn from_int(int: $intsize) -> Self {
                match size_of::<$intsize>() {
                    1 => Self::from_int(int as $int8),
                    2 => Self::from_int(int as $int16),
                    4 => Self::from_int(int as $int32),
                    8 => Self::from_int(int as $int64),
                    _ => unreachable!(concat!(stringify!($intsize), "is not equivalent to any primitive integer types.")),
                }
            }

            #[doc = concat!("We use the `from_canonical_checked` method of the primitive integer type identical to `", stringify!($intsize), "` on this machine")]
            fn from_canonical_checked(int: $intsize) -> Option<Self> {
                match size_of::<$intsize>() {
                    1 => Self::from_canonical_checked(int as $int8),
                    2 => Self::from_canonical_checked(int as $int16),
                    4 => Self::from_canonical_checked(int as $int32),
                    8 => Self::from_canonical_checked(int as $int64),
                    _ => unreachable!(concat!(stringify!($intsize), " is not equivalent to any primitive integer types.")),
                }
            }

            #[doc = concat!("We use the `from_canonical_unchecked` method of the primitive integer type identical to `", stringify!($intsize), "` on this machine")]
            unsafe fn from_canonical_unchecked(int: $intsize) -> Self {
                unsafe {
                    match size_of::<$intsize>() {
                        1 => Self::from_canonical_unchecked(int as $int8),
                        2 => Self::from_canonical_unchecked(int as $int16),
                        4 => Self::from_canonical_unchecked(int as $int32),
                        8 => Self::from_canonical_unchecked(int as $int64),
                        _ => unreachable!(concat!(stringify!($intsize), " is not equivalent to any primitive integer types.")),
                    }
                }
            }
        }
    };
}

impl_u_i_size!(usize, u8, u16, u32, u64);
impl_u_i_size!(isize, i8, i16, i32, i64);
