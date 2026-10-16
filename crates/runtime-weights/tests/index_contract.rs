use runtime_weights::parse_safetensors_index;

#[test]
fn parses_valid_multi_shard_index() {
    let index = parse_safetensors_index(br#"{"metadata":{"total_size":7},"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#).unwrap();
    assert_eq!(index.total_size(), 7);
    assert_eq!(index.tensor_count(), 2);
    assert_eq!(
        index.shards().collect::<Vec<_>>(),
        vec![
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
    );
}

#[test]
fn rejects_malformed_json_and_duplicate_tensor_mapping() {
    assert!(parse_safetensors_index(b"{").is_err());
    let duplicate = br#"{"metadata":{"total_size":1},"weight_map":{"a":"one.safetensors","a":"two.safetensors"}}"#;
    let error = parse_safetensors_index(duplicate).unwrap_err();
    assert!(error.to_string().contains("duplicate JSON key 'a'"));
}

#[test]
fn rejects_traversal_and_absolute_shard_paths() {
    for path in [
        "../escape.safetensors",
        "/tmp/escape.safetensors",
        "nested/a.safetensors",
    ] {
        let input = format!(r#"{{"metadata":{{"total_size":1}},"weight_map":{{"a":"{path}"}}}}"#);
        let error = parse_safetensors_index(input.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("portable file name"), "{error}");
    }
}
