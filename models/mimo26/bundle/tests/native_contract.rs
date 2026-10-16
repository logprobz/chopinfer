use model_compiler_sdk::native_contract::NativeModule;

/// The Cargo-built module's note publishes the audited kernel contract
/// identity, which hashes each entry's declared arithmetic, as its
/// implementation contract.
#[test]
fn native_note_publishes_the_kernel_contract_identity() {
    let module = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .join("libbundle_mimo26.so");
    let native =
        NativeModule::inspect(&module).expect("Cargo-built native module with embedded note");
    assert!(native.contracts.profiles.is_empty());
    let identity = bundle_mimo26::kernel_contract_identity();
    native.require_implementation_contract(identity).unwrap();
    assert!(
        native
            .require_implementation_contract(&"0".repeat(64))
            .is_err()
    );
}
