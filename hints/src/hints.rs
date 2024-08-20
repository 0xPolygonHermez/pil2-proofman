use p3_field::AbstractField;
use starks_lib_c::get_hint_field_c;

use ::std::os::raw::c_void;
use std::ops::{Index, IndexMut};    

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HintFieldType {
    Field = 0, // F
    FieldExtended = 1, // [F; 3]
    Column = 2, // Vec<F>
    ColumnExtended = 3, // Vec<[F;3]>
}

#[repr(C)]
#[allow(dead_code)]
pub struct HintFieldInfo<F> {
    size: u64,
    offset: u64,
    type_: HintFieldType,
    pub values: *mut F,
}


pub struct HintCol<F> {
    inner: Vec<F>,
    offset: usize,
    is_column: bool,
}


impl<F> HintCol<F> {
    pub fn from_hint_field(hint_field: &HintFieldInfo<F>) -> Self {
        HintCol {
            inner: unsafe { Vec::from_raw_parts(hint_field.values, hint_field.size as usize, hint_field.size as usize) },
            offset: hint_field.offset as usize,
            is_column: hint_field.type_ == HintFieldType::Column || hint_field.type_ == HintFieldType::ColumnExtended,
        }    
    }
}

impl<F> Index<usize> for HintCol<F> {
    type Output = [F];  // Return a slice of F

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.offset;
        let end = start + self.offset;

        if self.is_column {
            &self.inner[start..end]
        } else {
            &self.inner[0..self.offset]
        }
    }
}

impl<F> IndexMut<usize> for HintCol<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let start = index * self.offset;
        let end = start + self.offset;

        if self.is_column {
            &mut self.inner[start..end]
        } else {
            &mut self.inner[0..self.offset] 
        }
    }
}



pub fn get_hint_field<F> (p_chelpers_steps: *mut c_void, hint_id: u64, hint_field_name: &str, dest: bool) -> HintCol<F>
{
    let raw_ptr = get_hint_field_c(p_chelpers_steps, hint_id, hint_field_name, dest);

    let hint_field = unsafe { Box::from_raw(raw_ptr as *mut HintFieldInfo<F>) };

    HintCol {
        inner: unsafe { Vec::from_raw_parts(hint_field.values, hint_field.size as usize, hint_field.size as usize) },
        offset: hint_field.offset as usize,
        is_column: hint_field.type_ == HintFieldType::Column || hint_field.type_ == HintFieldType::ColumnExtended,
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_element_1() {
        let mut buffer = [0usize; 90];
        for i in 0..buffer.len() {
            buffer[i] = i + 144;
        }

        let hint_field: HintFieldInfo<usize> = HintFieldInfo::<usize> {
            size: 1,
            offset: 1,
            type_: HintFieldType::Field,
            values: buffer.as_mut_ptr(),
        };

        let hint_col = HintCol::<usize>::from_hint_field(&hint_field);

        assert_eq!(hint_col[0][0], 144);
    }
}