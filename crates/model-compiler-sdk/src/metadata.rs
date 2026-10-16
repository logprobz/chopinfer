//! Bounded JSON normalization without losing duplicate execution fields.
use crate::CompilerError;
use serde_json::Value;

/// The caller chooses the document's size bound. Serde retains its nesting
/// limit; unknown execution fields remain the typed importer's responsibility.
pub fn parse(bytes: &[u8], maximum: usize) -> Result<Value, CompilerError> {
    if bytes.len() > maximum {
        return Err(CompilerError::validation("metadata exceeds its byte bound"));
    }
    runtime_bundle::parse_unique_json(bytes).map_err(|error| {
        let message = error
            .to_string()
            .replace("duplicate JSON key:", "duplicate metadata field");
        CompilerError::validation(format!("invalid metadata: {message}"))
    })
}

/// The checkpoint document `name` in `root`, read through a confined
/// descriptor and [`parse`]d within `maximum` bytes.
pub fn read(root: &std::path::Path, name: &str, maximum: usize) -> Result<Value, CompilerError> {
    let bytes = runtime_weights::checkpoint::CheckpointFile::open(root, name)
        .and_then(|file| file.read_bounded(maximum as u64))
        .map_err(|e| CompilerError::validation(format!("{name}: {e}")))?;
    parse(&bytes, maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_preserves_exact_integers_and_typed_floats() {
        let value = parse(br#"{"integer":18446744073709551617,"large":1234567890123456789012345678901234567890,"float":1.00e-6}"#, 1024).unwrap();
        assert_eq!(value["integer"].to_string(), "18446744073709551617");
        assert_eq!(
            value["large"].to_string(),
            "1234567890123456789012345678901234567890"
        );
        assert_eq!(
            serde_json::from_value::<f64>(value["float"].clone()).unwrap(),
            1e-6
        );
        assert!(parse(b"1e400", 1024).is_err());
        let object = parse(br#"{"$serde_json::private::Number":"1"}"#, 1024).unwrap();
        assert!(object.is_object());
    }

    #[test]
    fn normalization_preserves_values_and_refuses_duplicate_fields() {
        let bytes = br#"{"a":[null,false,1,-3,0.25,"x"],"b":{"nested":true}}"#;
        assert_eq!(
            parse(bytes, 1024).unwrap(),
            serde_json::from_slice::<Value>(bytes).unwrap()
        );
        for bytes in [
            br#"{"n":1,"n":2}"#.as_slice(),
            br#"{"a":[{"n":1,"\u006e":2}]}"#,
        ] {
            assert!(
                parse(bytes, 1024)
                    .unwrap_err()
                    .to_string()
                    .contains("duplicate metadata field n")
            );
        }
        assert!(parse(bytes, 4).is_err());
    }
}
