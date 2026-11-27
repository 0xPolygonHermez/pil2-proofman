use anyhow::Result;

use crate::io::bin_file_writer::BinFileWriter;
use crate::types::stark_info::{GlobalConstraintsInfo, Hint};

const GLOBAL_CONSTRAINTS_NSECTIONS: u32 = 1;
const GLOBAL_HINTS_SECTION: u32 = 1;

/// Writes the global constraints binary file: its one section is the global hints (the constraints are
/// globalInfo's `globalConstraintDag`).
pub fn write_global_constraints_bin_file(global_constraints_info: &GlobalConstraintsInfo, path: &str) -> Result<()> {
    let mut writer = BinFileWriter::new(path, "chps", 1, GLOBAL_CONSTRAINTS_NSECTIONS)?;
    write_global_hints_section(&mut writer, &global_constraints_info.hints, GLOBAL_HINTS_SECTION)?;

    println!("> Writing the global constraints file finished");
    println!("---------------------------------------------");

    writer.close()?;
    Ok(())
}

fn write_global_hints_section(writer: &mut BinFileWriter, hints_info: &[Hint], section: u32) -> Result<()> {
    writer.start_write_section(section)?;

    let n_hints = hints_info.len() as u32;
    writer.write_u32(n_hints)?;

    for hint in hints_info {
        writer.write_string(&hint.name)?;
        let n_fields = hint.fields.len() as u32;
        writer.write_u32(n_fields)?;

        for field in &hint.fields {
            writer.write_string(&field.name)?;
            let n_values = field.values.len() as u32;
            writer.write_u32(n_values)?;

            for value in &field.values {
                writer.write_string(&value.op)?;

                match value.op.as_str() {
                    "number" => {
                        writer.write_u64(value.value)?;
                    }
                    "string" => {
                        writer.write_string(&value.string_value)?;
                    }
                    "airgroupvalue" => {
                        writer.write_u32(value.airgroup_id as u32)?;
                        writer.write_u32(value.id as u32)?;
                    }
                    "tmp" | "public" | "proofvalue" => {
                        writer.write_u32(value.id as u32)?;
                    }
                    _ => {
                        anyhow::bail!("Unknown operand type in global hint: {}", value.op);
                    }
                }

                writer.write_u32(value.pos.len() as u32)?;
                for p in &value.pos {
                    writer.write_u32(*p as u32)?;
                }
            }
        }
    }

    writer.end_write_section()?;
    Ok(())
}
