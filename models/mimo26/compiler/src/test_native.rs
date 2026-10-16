//! Object-format fixture for package tests; it cannot execute a model.
// The native fixture compile step every model shares, from the SDK's test
// support sources (outside its module tree).
#[path = "../../../../crates/model-compiler-sdk/test-support/native.rs"]
mod native;
use native::{compile_native_fixture, native_root_module};
use std::{path::PathBuf, sync::OnceLock};

/// The kernel contract identity the fixture's note publishes.
pub(crate) const SELECTION: &str =
    "f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1";

fn name() -> String {
    format!(
        "{}bundle_mimo26{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    )
}

/// The module under `CHOPINFER_PACKAGE_NATIVE_ROOT` when set, else the
/// object fixture.
pub(crate) fn module() -> PathBuf {
    // The SDK must verify this exact input. Invalid overrides never select a fixture.
    native_root_module(&name()).unwrap_or_else(fixture)
}

/// The object fixture, whose note publishes [`SELECTION`], whatever the
/// override.
pub(crate) fn fixture() -> PathBuf {
    static MODULE: OnceLock<PathBuf> = OnceLock::new();
    MODULE
        .get_or_init(|| {
            let note = model_mimo26_contract::note::elf_note(SELECTION, &"b".repeat(64));
            let source = format!(
                r#"
#[repr(C, align(4))]
struct Note<const N: usize>([u8; N]);
#[used]
#[unsafe(link_section = ".note.chopinfer.model")]
static NOTE: Note<{}> = Note({note:?});
#[unsafe(no_mangle)]
pub extern "C" fn chop_bundle_get_api_v1() -> *const core::ffi::c_void {{
    core::ptr::null()
}}
"#,
                note.len()
            );
            compile_native_fixture(
                "mimo-package-module",
                "mimo_package_fixture",
                &name(),
                &source,
            )
        })
        .clone()
}
