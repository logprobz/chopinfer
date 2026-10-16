use model_compiler_sdk::native_contract::NativeModule;
use model_qwen38_contract::{contract::Shape, note};

#[test]
fn ordinary_cargo_native_contract_matches_shared_declarations() {
    let module = std::env::var_os("CHOPINFER_QWEN38_NATIVE_MODULE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .join("libbundle_qwen38.so")
        });
    let native =
        NativeModule::inspect(&module).expect("Cargo-built native module with embedded note");
    for shape in Shape::ALL {
        for profile in [note::profile(shape), note::prepared_profile(shape)] {
            let expected = serde_json::from_value(profile).unwrap();
            native.require(&expected).unwrap();
            let mut incompatible = expected.clone();
            incompatible.package_contract = "0".repeat(64);
            assert!(
                native
                    .require(&incompatible)
                    .unwrap_err()
                    .to_string()
                    .contains("package-declaration")
            );
            incompatible = expected;
            incompatible.weight_contract = "0".repeat(64);
            assert!(
                native
                    .require(&incompatible)
                    .unwrap_err()
                    .to_string()
                    .contains("weight-layout")
            );
        }
    }
}
