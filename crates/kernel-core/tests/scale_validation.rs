use kernel_core::contract::DType;
use kernel_core::precision::packed::ScaleBytes;

#[test]
fn ue8m0_domain_matches_decoding_for_every_code_and_positivity_policy() {
    for code in 0..=u8::MAX {
        let bytes = [code];
        let scales = ScaleBytes {
            dtype: DType::Ue8M0,
            bytes: &bytes,
        };
        for positive in [false, true] {
            let decoded = scales.value(0).and_then(|value| {
                if positive && value == 0.0 {
                    Err("global scale must be positive")
                } else {
                    Ok(())
                }
            });
            assert_eq!(scales.validate(1, positive), decoded, "code {code}");
            assert_eq!(decoded.is_ok(), code != 255);
        }
    }
    let bytes: Vec<_> = (0..255).collect();
    let scales = ScaleBytes {
        dtype: DType::Ue8M0,
        bytes: &bytes,
    };
    for positive in [false, true] {
        assert_eq!(scales.validate(bytes.len(), positive), Ok(()));
        assert!(scales.validate(bytes.len() - 1, positive).is_err());
        assert!(scales.validate(bytes.len() + 1, positive).is_err());
        assert!(scales.validate(usize::MAX, positive).is_err());
        assert_eq!(
            ScaleBytes {
                dtype: DType::Ue8M0,
                bytes: &[]
            }
            .validate(0, positive),
            Ok(())
        );
    }
    assert_eq!(scales.value(0).unwrap().to_bits(), 1 << 22);
}
