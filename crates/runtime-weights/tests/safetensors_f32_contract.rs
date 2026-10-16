use runtime_weights::decode_safetensors_f32;

#[test]
fn decodes_known_little_endian_f32_bytes() {
    let decoded = decode_safetensors_f32(&[0x00, 0x00, 0x80, 0x3f]).unwrap();

    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].to_bits(), 1.0_f32.to_bits());
}

#[test]
fn byte_swapping_changes_the_decoded_value() {
    let decoded = decode_safetensors_f32(&[0x3f, 0x80, 0x00, 0x00]).unwrap();

    assert_ne!(decoded[0].to_bits(), 1.0_f32.to_bits());
}

#[test]
fn rejects_wrong_width_data() {
    for bytes in [
        &[][..],
        &[0x00][..],
        &[0x00, 0x00][..],
        &[0x00, 0x00, 0x80][..],
    ] {
        if bytes.is_empty() {
            assert!(decode_safetensors_f32(bytes).unwrap().is_empty());
        } else {
            assert!(decode_safetensors_f32(bytes).is_err());
        }
    }
}
