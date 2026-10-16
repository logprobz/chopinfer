//! The native contract note: the audited kernel contract identity, which
//! hashes every entry's wrapper and declared arithmetic, and the dependency
//! identity of the cubins that implement it.

runtime_bundle_sdk::embed_note!(include_bytes!(concat!(
    env!("OUT_DIR"),
    "/kernel-contract-note.bin"
)));

/// The audited kernel contract identity the note publishes as its
/// implementation contract's selection.
pub fn kernel_contract_identity() -> &'static str {
    include_str!(concat!(env!("OUT_DIR"), "/kernel-contract-identity.txt"))
}
