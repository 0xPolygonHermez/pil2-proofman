use serde::de::DeserializeOwned;
use std::fs::{self, File};
use std::io::Read;
use std::path::PathBuf;

pub fn load_from_json<T: Default + DeserializeOwned>(path: &Option<PathBuf>) -> T {
    if let Some(path) = &path {
        // Open the file if the path is Some
        let mut file = File::open(path).unwrap_or_else(|_| panic!("File not found at {path:?}"));

        if !file.metadata().unwrap().is_file() {
            panic!("{path:?} is not a valid file");
        }

        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap_or_else(|err| panic!("Failed to read file: {err}"));

        // Deserialize the contents into the expected struct
        serde_json::from_str(&contents).unwrap_or_else(|err| panic!("Failed to parse JSON: {err}"))
    } else {
        // Return the default value if path is None
        T::default()
    }
}

pub fn is_json_file(path: &Option<PathBuf>) -> bool {
    if let Some(path) = path {
        if let Ok(text) = fs::read_to_string(path) {
            return serde_json::from_str::<serde_json::Value>(&text).is_ok();
        }
    }
    false
}
