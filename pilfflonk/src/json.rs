//! JSON as pilfflonk writes it: the files, deterministically, and the canonical form the digest of
//! the vkey is computed over (A.6).

use std::cmp::Ordering;
use std::fs;
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::ser::{Error as _, SerializeMap, SerializeSeq};
use serde::{Serialize, Serializer};
use serde_json::Value;

use crate::error::{invalid, PilfflonkError, PilfflonkResult};

/// A pilfflonk JSON file: its type fixes the fields and their order, and `validate` what the
/// types alone cannot say.
///
/// The text is `JSON.stringify(value, null, 1)`'s layout (one space of indentation, no final
/// newline), as the STARK setup writes its `provingKey/` files (`setup/pil2-stark/src/output/
/// json.rs`). It depends only on the value: no field is a `HashMap`, objects with free keys are
/// `BTreeMap`s, and the one free-form value, the vkey's `qVerifier`, is written with its keys
/// sorted, so the bytes do not depend on whether some crate in the build turns on serde_json's
/// `preserve_order`.
pub trait JsonFile: Serialize + DeserializeOwned {
    /// Checks what the types do not.
    fn validate(&self) -> PilfflonkResult<()>;

    fn from_json_str(text: &str) -> PilfflonkResult<Self> {
        let value: Self = serde_json::from_str(text)?;
        value.validate()?;
        Ok(value)
    }

    /// Validates, then serialises: nothing invalid is written.
    fn to_json_string(&self) -> PilfflonkResult<String> {
        self.validate()?;
        to_json_string(self)
    }

    fn read(path: &Path) -> PilfflonkResult<Self> {
        let text =
            fs::read_to_string(path).map_err(|source| PilfflonkError::Io { path: path.to_path_buf(), source })?;
        Self::from_json_str(&text).map_err(|e| e.in_file(path))
    }

    fn write(&self, path: &Path) -> PilfflonkResult<()> {
        let text = self.to_json_string().map_err(|e| e.in_file(path))?;
        fs::write(path, text).map_err(|source| PilfflonkError::Io { path: path.to_path_buf(), source })
    }
}

impl PilfflonkError {
    pub(crate) fn in_file(self, path: &Path) -> Self {
        PilfflonkError::InFile { path: path.to_path_buf(), source: Box::new(self) }
    }
}

/// `value` in the layout of the files, `JSON.stringify(value, null, 1)`'s (see [`JsonFile`]), with
/// no validation: for what the setup writes in the STARK's formats (`expressionsinfo.json`,
/// `verifierinfo.json`, `globalConstraints.json`), whose types are `pil-info`'s.
pub fn to_json_string<T: Serialize + ?Sized>(value: &T) -> PilfflonkResult<String> {
    let mut buffer = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b" ");
    let mut serializer = serde_json::Serializer::with_formatter(&mut buffer, formatter);
    value.serialize(&mut serializer)?;
    String::from_utf8(buffer).map_err(|e| PilfflonkError::InvalidFormat(e.to_string()))
}

/// JavaScript's order of strings, the order of `Array.prototype.sort()`: by UTF-16 code units.
/// It is Rust's order of `str` (by code points) except that a character above U+FFFF sorts before
/// one in U+E000..U+FFFF.
fn js_order(a: &str, b: &str) -> Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

/// The largest integer JavaScript represents exactly, `Number.MAX_SAFE_INTEGER`.
pub const MAX_SAFE_INTEGER: u64 = (1 << 53) - 1;

/// The canonical JSON of `value` (A.6), over which the digest of the vkey is computed:
///
/// - no whitespace;
/// - object keys sorted by UTF-16 code units, the order of JavaScript's default `sort()`; the
///   order serde_json keeps them in is not relied on (the workspace turns on `preserve_order` for
///   some crates);
/// - strings escaped as `JSON.stringify` escapes them, which is how serde_json does it: `\"`,
///   `\\`, `\b`, `\f`, `\n`, `\r`, `\t`, other control characters as `\u00xx` in lowercase, and
///   nothing else;
/// - numbers only as integers of at most `MAX_SAFE_INTEGER` in absolute value, which
///   `JSON.stringify` writes the same way; any other number is an error. Big integers and points
///   are decimal strings already: that is how this crate's types serialise them.
///
/// A JavaScript implementation must write the keys in this order itself rather than build an
/// object and stringify it: an engine lists integer-like keys (`"0"`, `"1"`) first whatever the
/// insertion order. No key of the vkey is integer-like.
pub fn canonical_json<T: Serialize + ?Sized>(value: &T) -> PilfflonkResult<String> {
    let value = serde_json::to_value(value)?;
    let mut out = String::new();
    write_canonical(&value, &mut out)?;
    Ok(out)
}

fn write_canonical(value: &Value, out: &mut String) -> PilfflonkResult<()> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            let safe = match (n.as_u64(), n.as_i64()) {
                (Some(u), _) => u <= MAX_SAFE_INTEGER,
                (None, Some(i)) => i.unsigned_abs() <= MAX_SAFE_INTEGER,
                (None, None) => false,
            };
            if !safe {
                return invalid!(
                    "{n} has no canonical JSON form: numbers must be integers of at most 2^53 - 1 in absolute value"
                );
            }
            out.push_str(&n.to_string());
        }
        Value::String(s) => out.push_str(&serde_json::to_string(s)?),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| js_order(a.0, b.0));
            out.push('{');
            for (i, (key, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key)?);
                out.push(':');
                write_canonical(item, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// Serialises a free-form value with every object's keys sorted as `canonical_json` sorts them,
/// whatever order serde_json's `Map` keeps: for `#[serde(serialize_with)]`.
pub(crate) fn serialize_sorted<S: Serializer>(value: &Value, serializer: S) -> Result<S::Ok, S::Error> {
    Sorted(value).serialize(serializer)
}

struct Sorted<'a>(&'a Value);

impl Serialize for Sorted<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            Value::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(&Sorted(item))?;
                }
                seq.end()
            }
            Value::Object(map) => {
                let mut entries: Vec<(&String, &Value)> = map.iter().collect();
                entries.sort_by(|a, b| js_order(a.0, b.0));
                let mut out = serializer.serialize_map(Some(entries.len()))?;
                for (key, item) in entries {
                    out.serialize_entry(key, &Sorted(item))?;
                }
                out.end()
            }
            scalar => scalar.serialize(serializer).map_err(S::Error::custom),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_json_has_no_whitespace_and_sorted_keys() {
        let v = json!({"b": [1, {"d": null, "c": true}], "a": "x\ny", "e": -3});
        assert_eq!(canonical_json(&v).unwrap(), r#"{"a":"x\ny","b":[1,{"c":true,"d":null}],"e":-3}"#);
    }

    #[test]
    fn canonical_json_does_not_depend_on_the_order_of_the_keys() {
        let one = r#"{"z": 1, "a": {"y": [3, {"k": "v", "j": "w"}], "b": 2}, "m": "0"}"#;
        let other = r#"{"m": "0", "a": {"b": 2, "y": [3, {"j": "w", "k": "v"}]}, "z": 1}"#;
        let one: Value = serde_json::from_str(one).unwrap();
        let other: Value = serde_json::from_str(other).unwrap();
        assert_eq!(canonical_json(&one).unwrap(), canonical_json(&other).unwrap());
        assert_eq!(canonical_json(&one).unwrap(), r#"{"a":{"b":2,"y":[3,{"j":"w","k":"v"}]},"m":"0","z":1}"#);
    }

    #[test]
    fn keys_sort_as_javascript_sorts_them() {
        // U+FF61 is 0xFF61 in UTF-16 and U+1F600 is 0xD83D 0xDE00: JavaScript puts the emoji
        // first, and an order by code points (or by UTF-8 bytes) the other one.
        let v = json!({"\u{ff61}": 1, "\u{1f600}": 2, "b": 3, "B": 4});
        assert_eq!(canonical_json(&v).unwrap(), "{\"B\":4,\"b\":3,\"\u{1f600}\":2,\"\u{ff61}\":1}");
    }

    #[test]
    fn strings_are_escaped_as_json_stringify_escapes_them() {
        let v = json!(["\"\\/", "\u{8}\u{c}\n\r\t", "\u{1}\u{1f}\u{7f}", "é\u{2028}"]);
        // JSON.stringify(["\"\\/", "\b\f\n\r\t", "\x01\x1f\x7f", "é "])
        assert_eq!(
            canonical_json(&v).unwrap(),
            "[\"\\\"\\\\/\",\"\\b\\f\\n\\r\\t\",\"\\u0001\\u001f\u{7f}\",\"é\u{2028}\"]"
        );
    }

    #[test]
    fn only_safe_integers_are_canonical() {
        assert_eq!(canonical_json(&json!(MAX_SAFE_INTEGER)).unwrap(), "9007199254740991");
        assert_eq!(canonical_json(&json!(-(MAX_SAFE_INTEGER as i64))).unwrap(), "-9007199254740991");
        assert!(canonical_json(&json!(MAX_SAFE_INTEGER + 1)).is_err());
        assert!(canonical_json(&json!(u64::MAX)).is_err());
        assert!(canonical_json(&json!(1.5)).is_err());
        assert!(canonical_json(&json!({"a": [1.0]})).is_err());
    }

    #[test]
    fn free_form_values_are_written_with_sorted_keys() {
        #[derive(Serialize)]
        struct Holder {
            #[serde(serialize_with = "serialize_sorted")]
            v: Value,
        }
        let mut inner = serde_json::Map::new();
        inner.insert("z".into(), json!(1));
        inner.insert("a".into(), json!([{"y": 1, "b": 2}]));
        let holder = Holder { v: Value::Object(inner) };
        assert_eq!(serde_json::to_string(&holder).unwrap(), r#"{"v":{"a":[{"b":2,"y":1}],"z":1}}"#);
    }
}
