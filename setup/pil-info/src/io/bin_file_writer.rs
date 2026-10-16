use anyhow::{bail, Result};
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

/// Low-level binary file writer that implements the iden3/pil2 binfile format.
///
/// File layout:
///   - 4-byte magic type (e.g. "chps")
///   - u32 LE version
///   - u32 LE number of sections
///   - For each section:
///       - u32 LE section ID
///       - u64 LE section size (filled in at endWriteSection)
///       - section payload bytes
pub struct BinFileWriter {
    file: File,
    n_sections: u32,
    sections_written: u32,
    section_start: Option<u64>,
}

impl BinFileWriter {
    /// Creates a new binary file with the given magic type, version, and number of sections.
    pub fn new(path: &str, file_type: &str, version: u32, n_sections: u32) -> Result<Self> {
        assert!(file_type.len() == 4, "File type must be exactly 4 bytes");
        let mut file = File::create(path)?;

        // Write magic type (4 bytes)
        file.write_all(file_type.as_bytes())?;

        // Write version (u32 LE)
        file.write_all(&version.to_le_bytes())?;

        // Write number of sections (u32 LE)
        file.write_all(&n_sections.to_le_bytes())?;

        Ok(BinFileWriter { file, n_sections, sections_written: 0, section_start: None })
    }

    /// Begins writing a new section with the given ID.
    pub fn start_write_section(&mut self, section_id: u32) -> Result<()> {
        if self.section_start.is_some() {
            bail!("Already writing a section");
        }
        let pos = self.file.stream_position()?;
        self.section_start = Some(pos);

        // Write section ID
        self.write_u32(section_id)?;
        // Write placeholder for section size (u64)
        self.write_u64(0)?;

        Ok(())
    }

    /// Ends the current section and patches the section size header.
    pub fn end_write_section(&mut self) -> Result<()> {
        let start = match self.section_start {
            Some(s) => s,
            None => bail!("Not writing a section"),
        };

        let current_pos = self.file.stream_position()?;
        // Section size = current - start - 12 (4 for section_id + 8 for size placeholder)
        let section_size = current_pos - start - 12;

        // Seek back and write the actual size
        self.file.seek(SeekFrom::Start(start + 4))?;
        self.file.write_all(&section_size.to_le_bytes())?;
        self.file.seek(SeekFrom::Start(current_pos))?;

        self.section_start = None;
        self.sections_written += 1;
        Ok(())
    }

    pub fn write_u8(&mut self, value: u8) -> Result<()> {
        self.file.write_all(&[value])?;
        Ok(())
    }

    pub fn write_u16(&mut self, value: u16) -> Result<()> {
        self.file.write_all(&value.to_le_bytes())?;
        Ok(())
    }

    pub fn write_u32(&mut self, value: u32) -> Result<()> {
        self.file.write_all(&value.to_le_bytes())?;
        Ok(())
    }

    pub fn write_u64(&mut self, value: u64) -> Result<()> {
        self.file.write_all(&value.to_le_bytes())?;
        Ok(())
    }

    /// Writes a null-terminated string.
    pub fn write_string(&mut self, s: &str) -> Result<()> {
        self.file.write_all(s.as_bytes())?;
        self.file.write_all(&[0u8])?;
        Ok(())
    }

    /// Writes raw bytes.
    pub fn write_bytes(&mut self, data: &[u8]) -> Result<()> {
        self.file.write_all(data)?;
        Ok(())
    }

    /// Closes the file and checks that all sections were written.
    pub fn close(self) -> Result<()> {
        if self.sections_written != self.n_sections {
            eprintln!("Warning: expected {} sections but only {} were written", self.n_sections, self.sections_written);
        }
        // File is flushed and closed on drop
        Ok(())
    }
}
