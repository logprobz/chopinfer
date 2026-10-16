//! Native object fixture for package validation, with no model execution.
// The native fixture compile step every model shares, from the SDK's test
// support sources (outside its module tree).
#[path = "../../../../crates/model-compiler-sdk/test-support/native.rs"]
mod native;
use native::{compile_native_fixture, native_root_module};
use std::{path::PathBuf, sync::OnceLock};

pub(crate) fn module() -> PathBuf {
    native_root_module(super::NATIVE_MODULE).unwrap_or_else(fixture)
}

pub(crate) fn fixture() -> PathBuf {
    static MODULE: OnceLock<PathBuf> = OnceLock::new();
    MODULE
        .get_or_init(|| {
            compile_native_fixture(
                "glm-package-module",
                "glm_package_fixture",
                super::NATIVE_MODULE,
                r#"
#[unsafe(no_mangle)]
pub extern "C" fn chop_bundle_get_api_v1() -> *const core::ffi::c_void {
    core::ptr::null()
}
"#,
            )
        })
        .clone()
}
