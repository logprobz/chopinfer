//! The native contract note of the MiMo module.
use runtime_bundle::preparation::{ImplementationContract, NativeContracts, VERSION};

/// The note publishing the audited kernel contract identity `selection`,
/// which hashes every entry's wrapper and declared arithmetic, and the
/// dependency identity `dependencies` of the cubins that implement it as the
/// module's implementation contract. MiMo declares no preparation profiles.
pub fn elf_note(selection: &str, dependencies: &str) -> Vec<u8> {
    let contracts = NativeContracts {
        version: VERSION,
        profiles: Vec::new(),
        implementation_contract: Some(ImplementationContract {
            selection: selection.into(),
            dependencies: dependencies.into(),
            capabilities: None,
        }),
    };
    contracts.validate().expect("MiMo implementation contract");
    let description = serde_json::to_vec(&contracts).expect("note JSON");
    model_compiler_sdk::native_contract::encode_native_note(&description)
}

#[cfg(test)]
mod tests {
    #[test]
    fn note_decodes_to_its_implementation_contract() {
        let (selection, dependencies) = ("a".repeat(64), "b".repeat(64));
        let note = super::elf_note(&selection, &dependencies);
        let contracts = model_compiler_sdk::native_contract::decode_note(&note).unwrap();
        let contract = contracts.implementation_contract.unwrap();
        assert_eq!(
            (contract.selection, contract.dependencies),
            (selection, dependencies)
        );
        assert!(contracts.profiles.is_empty());
    }
}
