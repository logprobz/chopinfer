//! The native fixture cache of the test support sources
//! (test-support/native.rs), which models' package tests include.

#[path = "../test-support/native.rs"]
mod native;

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use native::cached_native_fixture;

const CRATE: &str = "sdk_cache_fixture";
const FILE: &str = "libsdk_cache_fixture.so";
const SOURCE: &str = r#"
#[unsafe(no_mangle)]
pub extern "C" fn chop_bundle_get_api_v1() -> *const core::ffi::c_void {
    core::ptr::null()
}
"#;

#[test]
fn concurrent_callers_share_one_build_in_the_profile_directory() {
    // A label of this run alone: another run on the same target directory
    // never shares its entry, nor removes it.
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let label = format!("sdk-native-cache-{}-{stamp}", std::process::id());
    let modules: Vec<PathBuf> = std::thread::scope(|threads| {
        let callers: Vec<_> = (0..4)
            .map(|_| threads.spawn(|| cached_native_fixture(&label, CRATE, FILE, SOURCE)))
            .collect();
        callers
            .into_iter()
            .map(|caller| caller.join().unwrap())
            .collect()
    });
    let module = &modules[0];
    assert!(modules.iter().all(|other| other == module), "{modules:?}");
    assert!(std::fs::symlink_metadata(module).unwrap().is_file());

    // `<profile>/chopinfer-native-fixtures/<label>-<key>/<file>`, the
    // profile directory holding this test binary's `deps`.
    let executable = std::env::current_exe().unwrap();
    let deps = executable.parent().unwrap();
    assert_eq!(deps.file_name().unwrap(), "deps");
    let cache = deps.parent().unwrap().join("chopinfer-native-fixtures");
    let directory = module.parent().unwrap();
    let name = directory.file_name().unwrap().to_str().unwrap().to_owned();
    assert_eq!(directory.parent().unwrap(), cache);
    assert!(name.starts_with(&format!("{label}-")), "{name}");
    assert_eq!(module.file_name().unwrap(), FILE);
    let building: Vec<_> = std::fs::read_dir(&cache)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|entry| entry.contains(&label) && entry.ends_with(".building"))
        .collect();
    assert!(building.is_empty(), "{building:?}");

    // A later call reuses the published module.
    let built = std::fs::metadata(module).unwrap().modified().unwrap();
    let again = cached_native_fixture(&label, CRATE, FILE, SOURCE);
    assert_eq!(&again, module);
    assert_eq!(
        std::fs::metadata(&again).unwrap().modified().unwrap(),
        built
    );

    // This run's entry and lock go, so repeated runs leave nothing behind.
    std::fs::remove_dir_all(directory).unwrap();
    std::fs::remove_file(cache.join(format!("{name}.lock"))).unwrap();
}
