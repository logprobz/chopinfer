//! Object-format and package-validation fixture, never a Qwen execution module.
// The native fixture compile step every model shares, from the SDK's test
// support sources (outside its module tree).
#[path = "../../../../crates/model-compiler-sdk/test-support/native.rs"]
mod native;
use native::{compile_native_fixture, native_root_module};
use std::{path::PathBuf, sync::OnceLock};

pub(crate) fn module() -> PathBuf {
    if let Some(module) = native_root_module("libbundle_qwen38.so") {
        return module;
    }
    static MODULE: OnceLock<PathBuf> = OnceLock::new();
    MODULE
        .get_or_init(|| {
            let note = model_qwen38_contract::note::elf_note();
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
                "qwen-package-module",
                "qwen_package_fixture",
                "libbundle_qwen38.so",
                &source,
            )
        })
        .clone()
}
