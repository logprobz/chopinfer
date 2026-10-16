mod support;

use std::io::Cursor;

use runtime_weights::{DType, parse_safetensors_header};
use support::safetensors;

#[test]
fn enumerates_without_reading_payload() {
    let bytes = safetensors(
        r#"{"a":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"b":{"dtype":"F32","shape":[1],"data_offsets":[12,16]}}"#,
        &[],
    );
    let declared_size = bytes.len() as u64 + 16;
    let header = parse_safetensors_header(&mut Cursor::new(bytes), declared_size).unwrap();
    assert_eq!(header.tensors().len(), 2);
    assert_eq!(header.tensor("a").unwrap().byte_len(), 12);
}

#[test]
fn accepts_actual_fp4_packed_storage_metadata() {
    let bytes = safetensors(
        r#"{"layers.0.ffn.experts.0.w1.weight":{"dtype":"I8","shape":[2048,2048],"data_offsets":[0,4194304]},"layers.0.ffn.experts.0.w1.scale":{"dtype":"F8_E8M0","shape":[2048,128],"data_offsets":[4194304,4456448]}}"#,
        &[],
    );
    let declared_size = bytes.len() as u64 + 4_456_448;
    let header = parse_safetensors_header(&mut Cursor::new(bytes), declared_size).unwrap();
    let packed = header.tensor("layers.0.ffn.experts.0.w1.weight").unwrap();
    assert_eq!(packed.dtype(), DType::I8);
    assert_eq!(packed.shape(), &[2048, 2048]);
    assert_eq!(packed.byte_len(), 4_194_304);
    assert_eq!(
        header
            .tensor("layers.0.ffn.experts.0.w1.scale")
            .unwrap()
            .dtype(),
        DType::F8E8M0
    );
}

#[test]
fn rejects_malformed_prefix_and_json() {
    let error = parse_safetensors_header(&mut Cursor::new(vec![0; 7]), 7).unwrap_err();
    assert!(error.to_string().contains("8-byte"));

    let mut bytes = 8_u64.to_le_bytes().to_vec();
    bytes.extend(b"not-json");
    let error = parse_safetensors_header(&mut Cursor::new(bytes), 16).unwrap_err();
    assert!(error.to_string().contains("JSON"));
}

#[test]
fn rejects_overlapping_offsets() {
    let bytes = safetensors(
        r#"{"a":{"dtype":"I8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"I8","shape":[4],"data_offsets":[2,6]}}"#,
        &[],
    );
    let declared_size = bytes.len() as u64 + 6;
    let error = parse_safetensors_header(&mut Cursor::new(bytes), declared_size).unwrap_err();
    assert!(error.to_string().contains("overlap"));
}

#[test]
fn rejects_shape_and_offset_overflow() {
    let bytes = safetensors(
        r#"{"huge":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#,
        &[],
    );
    let declared_size = bytes.len() as u64;
    let error = parse_safetensors_header(&mut Cursor::new(bytes), declared_size).unwrap_err();
    assert!(error.to_string().contains("overflow"));
}
