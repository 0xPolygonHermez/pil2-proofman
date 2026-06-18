use serde::ser::Error as _;
use serde::Serialize;

/// Serialize to JSON matching JS `JSON.stringify(obj, null, 1)` format:
/// single-space indentation.
pub fn to_json_string<T: Serialize>(value: &T) -> serde_json::Result<String> {
    let buf = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b" ");
    let mut ser = serde_json::Serializer::with_formatter(buf, formatter);
    value.serialize(&mut ser)?;
    String::from_utf8(ser.into_inner()).map_err(|e| serde_json::Error::custom(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_single_space_indent() {
        let val = json!({"a": 1, "b": [2, 3]});
        let s = to_json_string(&val).unwrap();
        // Verify single-space indentation
        assert!(s.contains(" \"a\""));
        assert!(!s.contains("  \"a\""));
    }
}
