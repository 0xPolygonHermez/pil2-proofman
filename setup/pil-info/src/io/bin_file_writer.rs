use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

use crate::error::{BinFileError, Result};

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
    pub fn new(path: &str, file_type: &str, version: u32, n_sections: u32) -> Result<Self, BinFileError> {
        if file_type.len() != 4 {
            return Err(BinFileError::FileType(file_type.to_string()));
        }
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
    pub fn start_write_section(&mut self, section_id: u32) -> Result<(), BinFileError> {
        if self.section_start.is_some() {
            return Err(BinFileError::SectionOpen);
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
    pub fn end_write_section(&mut self) -> Result<(), BinFileError> {
        let Some(start) = self.section_start else {
            return Err(BinFileError::NoSectionOpen);
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

    pub fn write_u8(&mut self, value: u8) -> Result<(), BinFileError> {
        self.file.write_all(&[value])?;
        Ok(())
    }

    pub fn write_u16(&mut self, value: u16) -> Result<(), BinFileError> {
        self.file.write_all(&value.to_le_bytes())?;
        Ok(())
    }

    pub fn write_u32(&mut self, value: u32) -> Result<(), BinFileError> {
        self.file.write_all(&value.to_le_bytes())?;
        Ok(())
    }

    pub fn write_u64(&mut self, value: u64) -> Result<(), BinFileError> {
        self.file.write_all(&value.to_le_bytes())?;
        Ok(())
    }

    /// Writes a null-terminated string.
    pub fn write_string(&mut self, s: &str) -> Result<(), BinFileError> {
        self.file.write_all(s.as_bytes())?;
        self.file.write_all(&[0u8])?;
        Ok(())
    }

    /// Writes raw bytes.
    pub fn write_bytes(&mut self, data: &[u8]) -> Result<(), BinFileError> {
        self.file.write_all(data)?;
        Ok(())
    }

    /// Closes the file and checks that all sections were written.
    pub fn close(self) -> Result<(), BinFileError> {
        if self.sections_written != self.n_sections {
            eprintln!("Warning: expected {} sections but only {} were written", self.n_sections, self.sections_written);
        }
        // File is flushed and closed on drop
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(dir: &std::path::Path) -> String {
        dir.join("file.bin").to_str().map(str::to_string).unwrap_or_default()
    }

    #[test]
    fn a_section_is_written_with_its_size() {
        let dir = std::env::temp_dir().join(format!("pil-info-bin-file-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut writer = BinFileWriter::new(&path(&dir), "chps", 1, 1).unwrap();
        writer.start_write_section(1).unwrap();
        writer.write_u32(7).unwrap();
        writer.end_write_section().unwrap();
        writer.close().unwrap();
        let bytes = std::fs::read(path(&dir)).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(&bytes[..4], b"chps");
        assert_eq!(bytes[16..24], 4u64.to_le_bytes());
        assert_eq!(bytes[24..], 7u32.to_le_bytes());
    }

    /// Were an `assert!` and `anyhow` errors.
    #[test]
    fn misuse_of_the_writer_is_an_error() {
        let dir = std::env::temp_dir().join(format!("pil-info-bin-file-misuse-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file_type = BinFileWriter::new(&path(&dir), "chp", 1, 1).err();
        let mut writer = BinFileWriter::new(&path(&dir), "chps", 1, 1).unwrap();
        let no_section = writer.end_write_section().err();
        writer.start_write_section(1).unwrap();
        let open = writer.start_write_section(2).err();
        std::fs::remove_dir_all(&dir).unwrap();

        assert!(matches!(&file_type, Some(BinFileError::FileType(t)) if t == "chp"), "{file_type:?}");
        assert!(matches!(no_section, Some(BinFileError::NoSectionOpen)), "{no_section:?}");
        assert!(matches!(open, Some(BinFileError::SectionOpen)), "{open:?}");
        let missing = BinFileWriter::new(&path(&dir.join("missing")), "chps", 1, 1).err();
        assert!(matches!(missing, Some(BinFileError::Io(_))), "{missing:?}");
    }
}
