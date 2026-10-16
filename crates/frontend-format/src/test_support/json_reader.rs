//! The shared vectors of the lossless JSON reader, run against any reader.
//!
//! The transport and this library each own a copy of the reader. Both run
//! these vectors, so a change to one copy fails a test until the other and
//! the vectors agree.

use std::fmt::Debug;

use serde::de::DeserializeOwned;
use serde_json::Value;

/// The vectors, described in the file.
pub const JSON_READER_CASES: &str = include_str!("../../tests/json_reader_cases.json");

/// Reads every vector with the reader `T`. `tagged` writes a value in the
/// vectors' tagged form, and `to_value` converts it to a serde_json value.
pub fn assert_json_reader_cases<T>(tagged: impl Fn(&T) -> Value, to_value: impl Fn(&T) -> Value)
where
    T: DeserializeOwned + PartialEq + Debug,
{
    let vectors: Value = serde_json::from_str(JSON_READER_CASES).unwrap();
    let cases = vectors["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let text = if let Some(text) = case["text"].as_str() {
            text.to_owned()
        } else if let Some(entries) = case["object_entries"].as_u64() {
            let entries: Vec<_> = (0..entries)
                .map(|index| format!("\"k{index}\":{index}"))
                .collect();
            format!("{{{}}}", entries.join(","))
        } else {
            let depth = case["nested_arrays"].as_u64().unwrap() as usize;
            format!("{}{}", "[".repeat(depth), "]".repeat(depth))
        };
        let read = serde_json::from_str::<T>(&text);
        if let Some(error) = case["error"].as_str() {
            let Err(refusal) = read else {
                panic!("{name}: read where it should be refused");
            };
            assert!(refusal.to_string().contains(error), "{name}: {refusal}");
            continue;
        }
        let read = read.unwrap_or_else(|refusal| panic!("{name}: {refusal}"));
        let tag = tagged(&read);
        if let Some(value) = case.get("value") {
            assert_eq!(&tag, value, "{name}");
        }
        if let Some(entries) = case["entries"].as_u64() {
            assert_eq!(
                tag["object"].as_array().map(Vec::len),
                Some(entries as usize),
                "{name}"
            );
        }
        if let Some(depth) = case["depth"].as_u64() {
            let mut level = &tag;
            let mut seen = 0;
            while let Some(items) = level["array"].as_array() {
                seen += 1;
                match items.first() {
                    Some(item) => level = item,
                    None => break,
                }
            }
            assert_eq!(seen, depth, "{name}");
        }
        let value = to_value(&read);
        assert_eq!(
            serde_json::from_value::<T>(value.clone()).unwrap(),
            read,
            "{name}: owned"
        );
        assert_eq!(T::deserialize(&value).unwrap(), read, "{name}: borrowed");
    }
}
