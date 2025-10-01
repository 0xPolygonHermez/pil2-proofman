use proofman_starks_lib_c::{
    get_hint_field_id_c, acc_hint_field_c, acc_mul_hint_fields_c, get_hint_field_c, get_hint_field_sizes_c,
    get_hint_field_values_c, get_hint_ids_by_name_c, mul_hint_fields_c, set_hint_field_c, update_airgroupvalue_c,
    n_hint_ids_by_name_c,
};

use std::collections::HashMap;
use std::ffi::c_void;

use fields::{CubicExtensionField, PrimeField64};
use proofman_common::{ProofCtx, SetupCtx, StepsParams};

use std::ops::{Add, Div, Mul, Sub, AddAssign, DivAssign, MulAssign, SubAssign};

use std::fmt::{Display, Debug, Formatter};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum HintFieldType {
    Field = 0,          // F
    FieldExtended = 1,  // [F; 3]
    Column = 2,         // Vec<F>
    ColumnExtended = 3, // Vec<[F;3]>
    String = 4,
}

#[repr(C)]
#[derive(Debug)]
pub struct HintFieldInfoC<F: PrimeField64> {
    size: u64,
    string_size: u64,
    offset: u8, // 1 or 3cd
    field_type: HintFieldType,
    values: *mut F,
    string_value: *mut u8,
    pub matrix_size: u64,
    pub pos: *mut u64,
    pub expression_line: *mut u8,
    expression_line_size: u64,
}

impl<F: PrimeField64> HintFieldInfoC<F> {
    pub fn from_hint_field_info_vec(hint_field_values: &mut [HintFieldInfo<F>]) -> Vec<HintFieldInfoC<F>> {
        hint_field_values
            .iter_mut()
            .map(|info| HintFieldInfoC {
                size: info.size,
                string_size: info.string_size,
                offset: info.offset,
                field_type: info.field_type,
                values: info.values.as_mut_ptr(),
                string_value: info.string_value.as_mut_ptr(),
                matrix_size: info.matrix_size,
                pos: info.pos.as_mut_ptr(),
                expression_line: info.expression_line.as_mut_ptr(),
                expression_line_size: info.expression_line_size,
            })
            .collect()
    }

    pub fn sync_to_hint_field_info(
        hint_field_values: &mut [HintFieldInfo<F>],
        hint_field_values_c: &Vec<HintFieldInfoC<F>>,
    ) {
        for (original, updated) in hint_field_values.iter_mut().zip(hint_field_values_c) {
            original.size = updated.size;
            original.string_size = updated.string_size;
            original.matrix_size = updated.matrix_size;
            original.expression_line_size = updated.expression_line_size;
            original.offset = updated.offset;
            original.field_type = updated.field_type;
        }
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct HintFieldInfo<F: PrimeField64> {
    size: u64,
    string_size: u64,
    offset: u8, // 1 or 3cd
    field_type: HintFieldType,
    values: Vec<F>,
    string_value: Vec<u8>,
    pub matrix_size: u64,
    pub pos: Vec<u64>,
    pub expression_line: Vec<u8>,
    expression_line_size: u64,
}

impl<F: PrimeField64> Default for HintFieldInfo<F> {
    fn default() -> Self {
        HintFieldInfo {
            size: 0,
            string_size: 0,
            offset: 0,
            field_type: HintFieldType::Field,
            values: Vec::new(),
            string_value: Vec::new(),
            matrix_size: 0,
            pos: Vec::new(),
            expression_line: Vec::new(),
            expression_line_size: 0,
        }
    }
}

impl<F: PrimeField64> HintFieldInfo<F> {
    pub fn init_buffers(&mut self) {
        if self.size > 0 {
            self.values = vec![F::ZERO; self.size as usize];
        }

        if self.matrix_size > 0 {
            self.pos = vec![0; self.matrix_size as usize];
        }

        if self.string_size > 0 {
            self.string_value = vec![0; self.string_size as usize];
        }

        if self.expression_line_size > 0 {
            self.expression_line = vec![0; self.expression_line_size as usize]
        }
    }
}

#[repr(C)]
pub struct HintFieldInfoValues<F: PrimeField64> {
    pub n_values: u64,
    pub hint_field_values: *mut HintFieldInfo<F>,
}

#[repr(C)]
#[derive(Clone, Default)]
pub struct HintFieldOptions {
    pub dest: bool,
    pub inverse: bool,
    pub print_expression: bool,
    pub initialize_zeros: bool,
    pub compilation_time: bool,
}

impl From<&HintFieldOptions> for *mut u8 {
    fn from(options: &HintFieldOptions) -> *mut u8 {
        options as *const HintFieldOptions as *mut u8
    }
}

impl HintFieldOptions {
    pub fn dest() -> Self {
        Self { dest: true, ..Default::default() }
    }

    pub fn dest_with_zeros() -> Self {
        Self { dest: true, initialize_zeros: true, ..Default::default() }
    }

    pub fn inverse() -> Self {
        Self { inverse: true, ..Default::default() }
    }

    pub fn compilation_time() -> Self {
        Self { compilation_time: true, ..Default::default() }
    }

    pub fn inverse_and_print_expression() -> Self {
        Self { inverse: true, print_expression: true, ..Default::default() }
    }

    pub fn print_expression() -> Self {
        Self { print_expression: true, ..Default::default() }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HintFieldValue<F: PrimeField64> {
    Field(F),
    FieldExtended(CubicExtensionField<F>),
    Column(Vec<F>),
    ColumnExtended(Vec<CubicExtensionField<F>>),
    String(String),
}

impl<F: PrimeField64> Display for HintFieldValue<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            HintFieldValue::Field(value) => write!(f, "{value}"),
            HintFieldValue::FieldExtended(ext_field) => write!(f, "{ext_field}"),
            HintFieldValue::Column(column) => {
                let formatted: Vec<String> = column.iter().map(|v| format!("{v}")).collect();
                write!(f, "[{}]", formatted.join(", "))
            }
            HintFieldValue::ColumnExtended(ext_column) => {
                let formatted: Vec<String> = ext_column.iter().map(|v| format!("{v}")).collect();
                write!(f, "[{}]", formatted.join(", "))
            }
            HintFieldValue::String(s) => write!(f, "{s}"),
        }
    }
}

pub struct HintFieldValues<F: PrimeField64> {
    pub values: HashMap<Vec<u64>, HintFieldValue<F>>,
}

impl<F: PrimeField64> HintFieldValues<F> {
    pub fn get(&self, index: usize) -> HashMap<Vec<u64>, HintFieldOutput<F>> {
        self.values.iter().map(|(key, value)| (key.clone(), value.get(index))).collect()
    }
}

#[derive(Clone, Debug)]
pub struct HintFieldValuesVec<F: PrimeField64> {
    pub values: Vec<HintFieldValue<F>>,
}

impl<F: PrimeField64> HintFieldValuesVec<F> {
    pub fn get(&self, index: usize) -> Vec<HintFieldOutput<F>> {
        self.values.iter().map(|value| value.get(index)).collect()
    }
}

impl<F: PrimeField64> Display for HintFieldValuesVec<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, value) in self.values.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{value}")?;
        }
        write!(f, "]")
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
// Define an enum to represent the possible return types
pub enum HintFieldOutput<F: PrimeField64> {
    Field(F),
    FieldExtended(CubicExtensionField<F>),
}

impl<F: PrimeField64> Display for HintFieldOutput<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            HintFieldOutput::Field(value) => write!(f, "{value}"),
            HintFieldOutput::FieldExtended(ext_field) => write!(f, "{ext_field}"),
        }
    }
}

pub fn format_hint_field_output_vec<F: PrimeField64>(vec: &[HintFieldOutput<F>]) -> String {
    format!("[{}]", vec.iter().map(|item| item.to_string()).collect::<Vec<String>>().join(", "))
}

impl<F: PrimeField64> HintFieldValue<F> {
    pub fn get(&self, index: usize) -> HintFieldOutput<F> {
        match self {
            HintFieldValue::Field(value) => HintFieldOutput::Field(*value),
            HintFieldValue::FieldExtended(value) => HintFieldOutput::FieldExtended(*value),
            HintFieldValue::Column(vec) => HintFieldOutput::Field(vec[index]),
            HintFieldValue::ColumnExtended(vec) => HintFieldOutput::FieldExtended(vec[index]),
            HintFieldValue::String(_str) => panic!(),
        }
    }

    pub fn set(&mut self, index: usize, output: HintFieldOutput<F>) {
        match (self, output) {
            (HintFieldValue::Field(val), HintFieldOutput::Field(new_val)) => {
                *val = new_val;
            }
            (HintFieldValue::FieldExtended(val), HintFieldOutput::FieldExtended(new_val)) => {
                *val = new_val;
            }
            (HintFieldValue::Column(vec), HintFieldOutput::Field(new_val)) => {
                vec[index] = new_val;
            }
            (HintFieldValue::ColumnExtended(vec), HintFieldOutput::FieldExtended(new_val)) => {
                vec[index] = new_val;
            }
            _ => panic!("Mismatched types in set method"),
        }
    }
}

impl<F: PrimeField64> Add<F> for HintFieldOutput<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self {
        match self {
            HintFieldOutput::Field(a) => HintFieldOutput::Field(a + rhs),
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a + rhs),
        }
    }
}

impl<F: PrimeField64> Add<CubicExtensionField<F>> for HintFieldOutput<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: CubicExtensionField<F>) -> Self {
        match self {
            HintFieldOutput::Field(a) => HintFieldOutput::FieldExtended(rhs + a),
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a + rhs),
        }
    }
}

impl<F: PrimeField64> Add for HintFieldOutput<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        match (self, rhs) {
            // Field * Field
            (HintFieldOutput::Field(a), HintFieldOutput::Field(b)) => HintFieldOutput::Field(a + b),

            // Field * FieldExtended
            (HintFieldOutput::Field(a), HintFieldOutput::FieldExtended(b)) => HintFieldOutput::FieldExtended(b + a),

            // FieldExtended * Field
            (HintFieldOutput::FieldExtended(a), HintFieldOutput::Field(b)) => HintFieldOutput::FieldExtended(a + b),

            // FieldExtended * FieldExtended
            (HintFieldOutput::FieldExtended(a), HintFieldOutput::FieldExtended(b)) => {
                HintFieldOutput::FieldExtended(a + b)
            }
        }
    }
}

impl<F: PrimeField64> AddAssign<F> for HintFieldOutput<F> {
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        *self = match *self {
            HintFieldOutput::Field(a) => HintFieldOutput::Field(a + rhs),
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a + rhs),
        }
    }
}

impl<F: PrimeField64> AddAssign<CubicExtensionField<F>> for HintFieldOutput<F> {
    #[inline]
    fn add_assign(&mut self, rhs: CubicExtensionField<F>) {
        *self = match *self {
            HintFieldOutput::Field(a) => HintFieldOutput::FieldExtended(rhs + a),
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a + rhs),
        }
    }
}

impl<F: PrimeField64> AddAssign<HintFieldOutput<F>> for HintFieldOutput<F> {
    #[inline]
    fn add_assign(&mut self, rhs: HintFieldOutput<F>) {
        match rhs {
            HintFieldOutput::Field(b) => match self {
                HintFieldOutput::Field(a) => *self = HintFieldOutput::Field(*a + b),
                HintFieldOutput::FieldExtended(a) => *self = HintFieldOutput::FieldExtended(*a + b),
            },
            HintFieldOutput::FieldExtended(b) => match self {
                HintFieldOutput::Field(a) => *self = HintFieldOutput::FieldExtended(b + *a),
                HintFieldOutput::FieldExtended(a) => *self = HintFieldOutput::FieldExtended(*a + b),
            },
        }
    }
}

impl<F: PrimeField64> Sub<F> for HintFieldOutput<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        match self {
            HintFieldOutput::Field(a) => HintFieldOutput::Field(a - rhs),
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a - rhs),
        }
    }
}

impl<F: PrimeField64> Sub<CubicExtensionField<F>> for HintFieldOutput<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: CubicExtensionField<F>) -> Self {
        match self {
            HintFieldOutput::Field(a) => {
                HintFieldOutput::FieldExtended(CubicExtensionField { value: [a, F::ZERO, F::ZERO] } - rhs)
            }
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a - rhs),
        }
    }
}

impl<F: PrimeField64> Sub for HintFieldOutput<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        match (self, rhs) {
            // Field * Field
            (HintFieldOutput::Field(a), HintFieldOutput::Field(b)) => HintFieldOutput::Field(a - b),

            // Field * FieldExtended
            (HintFieldOutput::Field(a), HintFieldOutput::FieldExtended(b)) => {
                HintFieldOutput::FieldExtended(CubicExtensionField { value: [a, F::ZERO, F::ZERO] } - b)
            }

            // FieldExtended * Field
            (HintFieldOutput::FieldExtended(a), HintFieldOutput::Field(b)) => HintFieldOutput::FieldExtended(a - b),

            // FieldExtended * FieldExtended
            (HintFieldOutput::FieldExtended(a), HintFieldOutput::FieldExtended(b)) => {
                HintFieldOutput::FieldExtended(a - b)
            }
        }
    }
}

impl<F: PrimeField64> SubAssign<F> for HintFieldOutput<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        *self = match *self {
            HintFieldOutput::Field(a) => HintFieldOutput::Field(a - rhs),
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a - rhs),
        }
    }
}

impl<F: PrimeField64> SubAssign<CubicExtensionField<F>> for HintFieldOutput<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: CubicExtensionField<F>) {
        *self = match *self {
            HintFieldOutput::Field(a) => {
                HintFieldOutput::FieldExtended(CubicExtensionField { value: [a, F::ZERO, F::ZERO] } - rhs)
            }
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a - rhs),
        }
    }
}

impl<F: PrimeField64> SubAssign<HintFieldOutput<F>> for HintFieldOutput<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: HintFieldOutput<F>) {
        match rhs {
            HintFieldOutput::Field(b) => match self {
                HintFieldOutput::Field(a) => *self = HintFieldOutput::Field(*a - b),
                HintFieldOutput::FieldExtended(a) => *self = HintFieldOutput::FieldExtended(*a - b),
            },
            HintFieldOutput::FieldExtended(b) => match self {
                HintFieldOutput::Field(a) => {
                    *self = HintFieldOutput::FieldExtended(CubicExtensionField { value: [*a, F::ZERO, F::ZERO] } - b)
                }
                HintFieldOutput::FieldExtended(a) => *self = HintFieldOutput::FieldExtended(*a - b),
            },
        }
    }
}

impl<F: PrimeField64> Mul<F> for HintFieldOutput<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self {
        match self {
            HintFieldOutput::Field(a) => HintFieldOutput::Field(a * rhs),
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a * rhs),
        }
    }
}

impl<F: PrimeField64> Mul<CubicExtensionField<F>> for HintFieldOutput<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: CubicExtensionField<F>) -> Self {
        match self {
            HintFieldOutput::Field(a) => HintFieldOutput::FieldExtended(rhs * a),
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a * rhs),
        }
    }
}

impl<F: PrimeField64> Mul for HintFieldOutput<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        match (self, rhs) {
            // Field * Field
            (HintFieldOutput::Field(a), HintFieldOutput::Field(b)) => HintFieldOutput::Field(a * b),

            // Field * FieldExtended
            (HintFieldOutput::Field(a), HintFieldOutput::FieldExtended(b)) => HintFieldOutput::FieldExtended(b * a),

            // FieldExtended * Field
            (HintFieldOutput::FieldExtended(a), HintFieldOutput::Field(b)) => HintFieldOutput::FieldExtended(a * b),

            // FieldExtended * FieldExtended
            (HintFieldOutput::FieldExtended(a), HintFieldOutput::FieldExtended(b)) => {
                HintFieldOutput::FieldExtended(a * b)
            }
        }
    }
}

impl<F: PrimeField64> MulAssign<F> for HintFieldOutput<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: F) {
        *self = match *self {
            HintFieldOutput::Field(a) => HintFieldOutput::Field(a * rhs),
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a * rhs),
        }
    }
}

impl<F: PrimeField64> MulAssign<CubicExtensionField<F>> for HintFieldOutput<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: CubicExtensionField<F>) {
        *self = match *self {
            HintFieldOutput::Field(a) => HintFieldOutput::FieldExtended(rhs * a),
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a * rhs),
        }
    }
}

impl<F: PrimeField64> MulAssign<HintFieldOutput<F>> for HintFieldOutput<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: HintFieldOutput<F>) {
        match rhs {
            HintFieldOutput::Field(b) => match self {
                HintFieldOutput::Field(a) => *self = HintFieldOutput::Field(*a * b),
                HintFieldOutput::FieldExtended(a) => *self = HintFieldOutput::FieldExtended(*a * b),
            },
            HintFieldOutput::FieldExtended(b) => match self {
                HintFieldOutput::Field(a) => *self = HintFieldOutput::FieldExtended(b * *a),
                HintFieldOutput::FieldExtended(a) => *self = HintFieldOutput::FieldExtended(*a * b),
            },
        }
    }
}

impl<F: PrimeField64> Div<F> for HintFieldOutput<F> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: F) -> Self {
        match self {
            HintFieldOutput::Field(a) => HintFieldOutput::Field(a / rhs),
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a * rhs.inverse()),
        }
    }
}

impl<F: PrimeField64> Div<CubicExtensionField<F>> for HintFieldOutput<F> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: CubicExtensionField<F>) -> Self {
        match self {
            HintFieldOutput::Field(a) => HintFieldOutput::FieldExtended(rhs.inverse() * a),
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a / rhs),
        }
    }
}

impl<F: PrimeField64> Div for HintFieldOutput<F> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        match (self, rhs) {
            // Field * Field
            (HintFieldOutput::Field(a), HintFieldOutput::Field(b)) => HintFieldOutput::Field(a / b),

            // Field * FieldExtended
            (HintFieldOutput::Field(a), HintFieldOutput::FieldExtended(b)) => {
                HintFieldOutput::FieldExtended(b.inverse() * a)
            }

            // FieldExtended * Field
            (HintFieldOutput::FieldExtended(a), HintFieldOutput::Field(b)) => {
                HintFieldOutput::FieldExtended(a * b.inverse())
            }

            // FieldExtended * FieldExtended
            (HintFieldOutput::FieldExtended(a), HintFieldOutput::FieldExtended(b)) => {
                HintFieldOutput::FieldExtended(a / b)
            }
        }
    }
}

impl<F: PrimeField64> DivAssign<F> for HintFieldOutput<F> {
    #[inline]
    fn div_assign(&mut self, rhs: F) {
        *self = match *self {
            HintFieldOutput::Field(a) => HintFieldOutput::Field(a / rhs),
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a * rhs.inverse()),
        }
    }
}

impl<F: PrimeField64> DivAssign<CubicExtensionField<F>> for HintFieldOutput<F> {
    #[inline]
    fn div_assign(&mut self, rhs: CubicExtensionField<F>) {
        *self = match *self {
            HintFieldOutput::Field(a) => HintFieldOutput::FieldExtended(rhs.inverse() * a),
            HintFieldOutput::FieldExtended(a) => HintFieldOutput::FieldExtended(a / rhs),
        }
    }
}

impl<F: PrimeField64> DivAssign<HintFieldOutput<F>> for HintFieldOutput<F> {
    #[inline]
    fn div_assign(&mut self, rhs: HintFieldOutput<F>) {
        match rhs {
            HintFieldOutput::Field(b) => match self {
                HintFieldOutput::Field(a) => *self = HintFieldOutput::Field(*a / b),
                HintFieldOutput::FieldExtended(a) => *self = HintFieldOutput::FieldExtended(*a * b.inverse()),
            },
            HintFieldOutput::FieldExtended(b) => match self {
                HintFieldOutput::Field(a) => *self = HintFieldOutput::FieldExtended(b.inverse() * *a),
                HintFieldOutput::FieldExtended(a) => *self = HintFieldOutput::FieldExtended(*a / b),
            },
        }
    }
}

impl<F: PrimeField64> HintFieldValue<F> {
    pub fn add(&mut self, index: usize, value: F) {
        match self {
            HintFieldValue::Field(v) => *v += value,
            HintFieldValue::FieldExtended(v) => *v += value,
            HintFieldValue::Column(vec) => vec[index] += value,
            HintFieldValue::ColumnExtended(vec) => vec[index] += value,
            HintFieldValue::String(_str) => panic!(),
        };
    }

    pub fn add_e(&mut self, index: usize, value: CubicExtensionField<F>) {
        match self {
            HintFieldValue::FieldExtended(v) => *v += value,
            HintFieldValue::ColumnExtended(vec) => vec[index] += value,
            _ => panic!(),
        };
    }
}

impl<F: PrimeField64> HintFieldValue<F> {
    pub fn sub(&mut self, index: usize, value: F) {
        match self {
            HintFieldValue::Field(v) => *v -= value,
            HintFieldValue::FieldExtended(v) => *v -= value,
            HintFieldValue::Column(vec) => vec[index] -= value,
            HintFieldValue::ColumnExtended(vec) => vec[index] -= value,
            HintFieldValue::String(_str) => panic!(),
        };
    }

    pub fn sub_e(&mut self, index: usize, value: CubicExtensionField<F>) {
        match self {
            HintFieldValue::FieldExtended(v) => *v -= value,
            HintFieldValue::ColumnExtended(vec) => vec[index] -= value,
            _ => panic!(),
        };
    }
}

impl<F: PrimeField64> HintFieldValue<F> {
    pub fn mul(&mut self, index: usize, value: F) {
        match self {
            HintFieldValue::Field(v) => *v *= value,
            HintFieldValue::FieldExtended(v) => *v *= value,
            HintFieldValue::Column(vec) => vec[index] *= value,
            HintFieldValue::ColumnExtended(vec) => vec[index] *= value,
            HintFieldValue::String(_str) => panic!(),
        };
    }

    pub fn mul_e(&mut self, index: usize, value: CubicExtensionField<F>) {
        match self {
            HintFieldValue::FieldExtended(v) => *v *= value,
            HintFieldValue::ColumnExtended(vec) => vec[index] *= value,
            _ => panic!(),
        };
    }
}

impl<F: PrimeField64> HintFieldValue<F> {
    pub fn div(&mut self, index: usize, value: F) {
        match self {
            HintFieldValue::Field(v) => *v *= value.inverse(),
            HintFieldValue::FieldExtended(v) => *v *= value.inverse(),
            HintFieldValue::Column(vec) => vec[index] *= value.inverse(),
            HintFieldValue::ColumnExtended(vec) => vec[index] *= value.inverse(),
            HintFieldValue::String(_str) => panic!(),
        };
    }

    pub fn div_e(&mut self, index: usize, value: CubicExtensionField<F>) {
        match self {
            HintFieldValue::FieldExtended(v) => *v *= value.inverse(),
            HintFieldValue::ColumnExtended(vec) => vec[index] *= value.inverse(),
            _ => panic!(),
        };
    }
}
pub struct HintCol;

impl HintCol {
    pub fn from_hint_field<F: PrimeField64>(hint_field: &HintFieldInfo<F>) -> HintFieldValue<F> {
        match hint_field.field_type {
            HintFieldType::Field => HintFieldValue::Field(hint_field.values[0]),
            HintFieldType::FieldExtended => {
                let array = [hint_field.values[0], hint_field.values[1], hint_field.values[2]];
                HintFieldValue::FieldExtended(CubicExtensionField { value: array })
            }
            HintFieldType::Column => HintFieldValue::Column(hint_field.values.to_vec()),
            HintFieldType::ColumnExtended => {
                let mut extended_vec = Vec::with_capacity(hint_field.size as usize / 3);
                for chunk in hint_field.values.chunks(3) {
                    extended_vec.push(CubicExtensionField { value: [chunk[0], chunk[1], chunk[2]] });
                }
                HintFieldValue::ColumnExtended(extended_vec)
            }
            HintFieldType::String => match std::str::from_utf8(&hint_field.string_value) {
                Ok(value) => HintFieldValue::String(value.to_string()),
                Err(_) => HintFieldValue::String(String::new()),
            },
        }
    }
}

pub fn get_hint_ids_by_name(p_expressions_bin: *mut std::os::raw::c_void, name: &str) -> Vec<u64> {
    let n_hints = n_hint_ids_by_name_c(p_expressions_bin, name);

    let mut hint_ids = vec![0; n_hints as usize];

    get_hint_ids_by_name_c(p_expressions_bin, hint_ids.as_mut_ptr(), name);

    hint_ids
}

#[allow(clippy::too_many_arguments)]
pub fn mul_hint_fields<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    pctx: &ProofCtx<F>,
    instance_id: usize,
    n_hints: u64,
    hint_ids: Vec<u64>,
    hint_field_dest: Vec<&str>,
    hint_field_name1: Vec<&str>,
    mut options1: Vec<HintFieldOptions>,
    hint_field_name2: Vec<&str>,
    mut options2: Vec<HintFieldOptions>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;

    let setup = sctx.get_setup(airgroup_id, air_id)?;

    let steps_params = pctx.get_air_instance_params(instance_id, false);

    let mut hint_options1: Vec<*mut u8> = options1.iter_mut().map(|s| s as *mut HintFieldOptions as *mut u8).collect();

    let mut hint_options2: Vec<*mut u8> = options2.iter_mut().map(|s| s as *mut HintFieldOptions as *mut u8).collect();

    mul_hint_fields_c(
        (&setup.p_setup).into(),
        (&steps_params).into(),
        n_hints,
        hint_ids.as_ptr() as *mut u64,
        hint_field_dest,
        hint_field_name1,
        hint_field_name2,
        hint_options1.as_mut_ptr(),
        hint_options2.as_mut_ptr(),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn acc_hint_field<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    pctx: &ProofCtx<F>,
    instance_id: usize,
    hint_id: usize,
    hint_field_dest: &str,
    hint_field_airgroupvalue: &str,
    hint_field_name: &str,
    add: bool,
) -> Result<(u64, u64), Box<dyn std::error::Error + Send + Sync>> {
    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;
    let setup = sctx.get_setup(airgroup_id, air_id)?;

    let steps_params = pctx.get_air_instance_params(instance_id, false);

    acc_hint_field_c(
        (&setup.p_setup).into(),
        (&steps_params).into(),
        hint_id as u64,
        hint_field_dest,
        hint_field_airgroupvalue,
        hint_field_name,
        add,
    );

    let dest_id = get_hint_field_id_c((&setup.p_setup).into(), hint_id as u64, hint_field_dest);
    let airgroup_value_id = get_hint_field_id_c((&setup.p_setup).into(), hint_id as u64, hint_field_airgroupvalue);

    Ok((dest_id, airgroup_value_id))
}

#[allow(clippy::too_many_arguments)]
pub fn acc_mul_hint_fields<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    pctx: &ProofCtx<F>,
    instance_id: usize,
    hint_id: usize,
    hint_field_dest: &str,
    hint_field_airgroupvalue: Option<&str>,
    hint_field_name1: &str,
    hint_field_name2: &str,
    options1: HintFieldOptions,
    options2: HintFieldOptions,
    add: bool,
) -> Result<(u64, u64), Box<dyn std::error::Error + Send + Sync>> {
    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;
    let setup = sctx.get_setup(airgroup_id, air_id)?;

    let steps_params = pctx.get_air_instance_params(instance_id, false);

    let field_airgroupvalue = hint_field_airgroupvalue.unwrap_or("");

    acc_mul_hint_fields_c(
        (&setup.p_setup).into(),
        (&steps_params).into(),
        hint_id as u64,
        hint_field_dest,
        field_airgroupvalue,
        hint_field_name1,
        hint_field_name2,
        (&options1).into(),
        (&options2).into(),
        add,
    );

    if let Some(hint_field_airgroupvalue) = hint_field_airgroupvalue {
        let dest_id = get_hint_field_id_c((&setup.p_setup).into(), hint_id as u64, hint_field_dest);
        let airgroup_value_id = get_hint_field_id_c((&setup.p_setup).into(), hint_id as u64, hint_field_airgroupvalue);

        Ok((dest_id, airgroup_value_id))
    } else {
        Ok((0, 0))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn update_airgroupvalue<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    pctx: &ProofCtx<F>,
    instance_id: usize,
    hint_id: usize,
    hint_field_airgroupvalue: Option<&str>,
    hint_field_name1: &str,
    hint_field_name2: &str,
    options1: HintFieldOptions,
    options2: HintFieldOptions,
    add: bool,
) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;
    let setup = sctx.get_setup(airgroup_id, air_id)?;

    let steps_params = pctx.get_air_instance_params(instance_id, false);

    let field_airgroupvalue = hint_field_airgroupvalue.unwrap_or("");

    Ok(update_airgroupvalue_c(
        (&setup.p_setup).into(),
        (&steps_params).into(),
        hint_id as u64,
        field_airgroupvalue,
        hint_field_name1,
        hint_field_name2,
        (&options1).into(),
        (&options2).into(),
        add,
    ))
}

#[allow(clippy::too_many_arguments)]
fn get_hint_f<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    pctx: Option<&ProofCtx<F>>,
    airgroup_id: usize,
    air_id: usize,
    instance_id: Option<usize>,
    hint_id: usize,
    hint_field_name: &str,
    options: HintFieldOptions,
) -> Result<Vec<HintFieldInfo<F>>, Box<dyn std::error::Error + Send + Sync>> {
    let setup = sctx.get_setup(airgroup_id, air_id)?;

    let steps_params = if let Some(instance_id) = instance_id {
        pctx.unwrap().get_air_instance_params(instance_id, false)
    } else {
        StepsParams::default()
    };

    let n_hints_values = get_hint_field_values_c((&setup.p_setup).into(), hint_id as u64, hint_field_name);

    let mut hint_field_values: Vec<HintFieldInfo<F>> = vec![HintFieldInfo::default(); n_hints_values as usize];

    let mut hint_field_values_c = HintFieldInfoC::from_hint_field_info_vec(&mut hint_field_values);
    let mut hint_field_values_c_ptr = hint_field_values_c.as_mut_ptr() as *mut c_void;

    get_hint_field_sizes_c(
        (&setup.p_setup).into(),
        hint_field_values_c_ptr,
        hint_id as u64,
        hint_field_name,
        (&options).into(),
    );

    HintFieldInfoC::sync_to_hint_field_info(&mut hint_field_values, &hint_field_values_c);

    for hint_field_value in hint_field_values.iter_mut() {
        hint_field_value.init_buffers();
    }

    hint_field_values_c = HintFieldInfoC::from_hint_field_info_vec(&mut hint_field_values);
    hint_field_values_c_ptr = hint_field_values_c.as_mut_ptr() as *mut c_void;

    get_hint_field_c(
        (&setup.p_setup).into(),
        (&steps_params).into(),
        hint_field_values_c_ptr,
        hint_id as u64,
        hint_field_name,
        (&options).into(),
    );

    Ok(hint_field_values)
}
pub fn get_hint_field<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    pctx: &ProofCtx<F>,
    instance_id: usize,
    hint_id: usize,
    hint_field_name: &str,
    options: HintFieldOptions,
) -> Result<HintFieldValue<F>, Box<dyn std::error::Error + Send + Sync>> {
    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;

    let hint_info = get_hint_f(
        sctx,
        Some(pctx),
        airgroup_id,
        air_id,
        Some(instance_id),
        hint_id,
        hint_field_name,
        options.clone(),
    )?;

    if hint_info[0].matrix_size != 0 {
        return Err(format!(
            "get_hint_field can only be called with single expressions, but {hint_field_name} is an array"
        )
        .into());
    }

    if options.print_expression {
        tracing::info!("HintsInf: {}", std::str::from_utf8(&hint_info[0].expression_line).unwrap());
    }

    Ok(HintCol::from_hint_field(&hint_info[0]))
}

pub fn get_hint_field_a<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    pctx: &ProofCtx<F>,
    instance_id: usize,
    hint_id: usize,
    hint_field_name: &str,
    options: HintFieldOptions,
) -> Result<HintFieldValuesVec<F>, Box<dyn std::error::Error + Send + Sync>> {
    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;

    let hint_infos = get_hint_f(
        sctx,
        Some(pctx),
        airgroup_id,
        air_id,
        Some(instance_id),
        hint_id,
        hint_field_name,
        options.clone(),
    )?;

    let mut hint_field_values = Vec::new();
    for (v, hint_info) in hint_infos.iter().enumerate() {
        if v == 0 && hint_info.matrix_size != 1 {
            return Err(("get_hint_field_m can only be called with an array of expressions!".to_string()).into());
        }
        if options.print_expression {
            tracing::info!("HintsInf: {}", std::str::from_utf8(&hint_info.expression_line).unwrap());
        }
        let hint_value = HintCol::from_hint_field(hint_info);
        hint_field_values.push(hint_value);
    }

    Ok(HintFieldValuesVec { values: hint_field_values })
}

pub fn get_hint_field_m<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    pctx: &ProofCtx<F>,
    instance_id: usize,
    hint_id: usize,
    hint_field_name: &str,
    options: HintFieldOptions,
) -> Result<HintFieldValues<F>, Box<dyn std::error::Error + Send + Sync>> {
    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;

    let hint_infos = get_hint_f(
        sctx,
        Some(pctx),
        airgroup_id,
        air_id,
        Some(instance_id),
        hint_id,
        hint_field_name,
        options.clone(),
    )?;

    let mut hint_field_values = HashMap::with_capacity(hint_infos.len() as usize);

    for (v, hint_info) in hint_infos.iter().enumerate() {
        if v == 0 && hint_info.matrix_size > 2 {
            return Err(("get_hint_field_m can only be called with a matrix of expressions!".to_string()).into());
        }
        let hint_value = HintCol::from_hint_field(hint_info);
        let mut pos = Vec::new();
        for p in 0..hint_info.matrix_size {
            pos.push(hint_info.pos[p as usize]);
        }
        if options.print_expression {
            tracing::info!("HintsInf: {}", std::str::from_utf8(&hint_info.expression_line).unwrap());
        }
        hint_field_values.insert(pos, hint_value);
    }

    Ok(HintFieldValues { values: hint_field_values })
}

pub fn get_hint_field_constant<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    hint_id: usize,
    hint_field_name: &str,
    mut options: HintFieldOptions,
) -> Result<HintFieldValue<F>, Box<dyn std::error::Error + Send + Sync>> {
    options.compilation_time = true;

    let hint_info = get_hint_f(sctx, None, airgroup_id, air_id, None, hint_id, hint_field_name, options.clone())?;

    if hint_info[0].matrix_size != 0 {
        return Err(format!(
            "get_hint_field can only be called with single expressions, but {hint_field_name} is an array"
        )
        .into());
    }

    if options.print_expression {
        tracing::info!("HintsInf: {}", std::str::from_utf8(&hint_info[0].expression_line).unwrap());
    }

    Ok(HintCol::from_hint_field(&hint_info[0]))
}

pub fn get_hint_field_constant_a<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    hint_id: usize,
    hint_field_name: &str,
    mut options: HintFieldOptions,
) -> Result<HintFieldValuesVec<F>, Box<dyn std::error::Error + Send + Sync>> {
    options.compilation_time = true;

    let hint_infos = get_hint_f(sctx, None, airgroup_id, air_id, None, hint_id, hint_field_name, options.clone())?;

    let mut hint_field_values = Vec::new();
    for (v, hint_info) in hint_infos.iter().enumerate() {
        if v == 0 && hint_info.matrix_size != 1 {
            return Err(("get_hint_field_m can only be called with an array of expressions!".to_string()).into());
        }
        if options.print_expression {
            tracing::info!("HintsInf: {}", std::str::from_utf8(&hint_info.expression_line).unwrap());
        }
        let hint_value = HintCol::from_hint_field(hint_info);
        hint_field_values.push(hint_value);
    }

    Ok(HintFieldValuesVec { values: hint_field_values })
}

pub fn get_hint_field_constant_m<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    hint_id: usize,
    hint_field_name: &str,
    mut options: HintFieldOptions,
) -> Result<HintFieldValues<F>, Box<dyn std::error::Error + Send + Sync>> {
    options.compilation_time = true;

    let hint_infos = get_hint_f(sctx, None, airgroup_id, air_id, None, hint_id, hint_field_name, options.clone())?;

    let mut hint_field_values = HashMap::with_capacity(hint_infos.len() as usize);

    for (v, hint_info) in hint_infos.iter().enumerate() {
        if v == 0 && hint_info.matrix_size > 2 {
            return Err(("get_hint_field_m can only be called with a matrix of expressions!".to_string()).into());
        }
        let hint_value = HintCol::from_hint_field(hint_info);
        let mut pos = Vec::new();
        for p in 0..hint_info.matrix_size {
            pos.push(hint_info.pos[p as usize]);
        }
        if options.print_expression {
            tracing::info!("HintsInf: {}", std::str::from_utf8(&hint_info.expression_line).unwrap());
        }
        hint_field_values.insert(pos, hint_value);
    }

    Ok(HintFieldValues { values: hint_field_values })
}

pub fn set_hint_field<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    instance_id: usize,
    hint_id: u64,
    hint_field_name: &str,
    values: &HintFieldValue<F>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;
    let setup = sctx.get_setup(airgroup_id, air_id)?;

    let steps_params = pctx.get_air_instance_params(instance_id, false);

    let values_ptr: *mut u8 = match values {
        HintFieldValue::Column(vec) => vec.as_ptr() as *mut u8,
        HintFieldValue::ColumnExtended(vec) => vec.as_ptr() as *mut u8,
        _ => return Err(("Only column and column extended are accepted".to_string()).into()),
    };

    set_hint_field_c((&setup.p_setup).into(), (&steps_params).into(), values_ptr, hint_id, hint_field_name);
    Ok(())
}

pub fn set_hint_field_val<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    instance_id: usize,
    hint_id: u64,
    hint_field_name: &str,
    value: HintFieldOutput<F>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;
    let setup = sctx.get_setup(airgroup_id, air_id)?;

    let steps_params = pctx.get_air_instance_params(instance_id, false);

    let mut value_array = Vec::new();

    match value {
        HintFieldOutput::Field(val) => {
            value_array.push(val);
        }
        HintFieldOutput::FieldExtended(val) => {
            value_array.push(val.value[0]);
            value_array.push(val.value[1]);
            value_array.push(val.value[2]);
        }
    };

    let values_ptr = value_array.as_ptr() as *mut u8;

    set_hint_field_c((&setup.p_setup).into(), (&steps_params).into(), values_ptr, hint_id, hint_field_name);
    Ok(())
}
