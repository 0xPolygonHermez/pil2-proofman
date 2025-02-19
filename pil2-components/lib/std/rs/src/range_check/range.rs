use std::fmt::{Debug, Display};

use p3_field::PrimeField64;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct RangeData<F: PrimeField64> {
    pub min: u64,
    pub max: u64,
    pub min_neg: bool,
    pub max_neg: bool,
    pub predefined: bool,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField64> Display for RangeData<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let order = Self::ORDER;

        let min_128 = self.min as i128;
        let max_128 = self.max as i128;
        let min = if self.min_neg { min_128 - order } else { min_128 };
        let max = if self.max_neg { max_128 - order } else { max_128 };

        write!(f, "[{},{}]", min, max)
    }
}

impl<F: PrimeField64> PartialEq<(bool, i64, i64)> for RangeData<F> {
    fn eq(&self, other: &(bool, i64, i64)) -> bool {
        let order = Self::ORDER;

        let min_128 = self.min as i128;
        let max_128 = self.max as i128;
        let min = if self.min_neg { min_128 - order } else { min_128 };
        let max = if self.max_neg { max_128 - order } else { max_128 };

        let other_min = other.1 as i128;
        let other_max = other.2 as i128;

        self.predefined == other.0 && min == other_min && max == other_max
    }
}

impl<F: PrimeField64> RangeData<F> {
    pub const ORDER: i128 = F::ORDER_U64 as i128;

    pub fn new(min: u64, max: u64, min_neg: bool, max_neg: bool, predefined: bool) -> Self {
        Self { min, max, min_neg, max_neg, predefined, _phantom: std::marker::PhantomData }
    }

    pub fn contains(&self, value: i64) -> bool {
        let order = Self::ORDER;

        let value = value as i128;

        let min_128 = self.min as i128;
        let max_128 = self.max as i128;
        let min = if self.min_neg { min_128 - order } else { min_128 };
        let max = if self.max_neg { max_128 - order } else { max_128 };

        value >= min && value <= max
    }
}
