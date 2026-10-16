//! Loading a real V1 module, from bytes whose package file has moved on.
//!
//! Everything here compiles an actual shared object, hands the loader that
//! object's exact bytes, and then changes the file those bytes were read from.
//! What the loader maps has to stay what it was given, and the only way to know
//! that is to load a module the platform's dynamic loader really accepts.

#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use runtime_bundle_abi::{
    BUNDLE_ABI_VERSION_V1, BundleAbiError, BundleApiWithRequestRetirementV1, BundleLibraryV1,
    BundleLoaderError, BundleRawApiV1, HostServicesTableV1, UnsupportedHostServicesBackendV1,
};

/// A version no V1 loader implements, so a module advertising it is refused.
const REFUSED_ABI_VERSION: u32 = BUNDLE_ABI_VERSION_V1 + 6;

/// A module changed under the loader still loads as the bytes that verified.
///
/// The package file is attacked both ways a writer can attack it once the
/// digest is taken: its inode is rewritten through a handle held open across
/// the load, and its directory entry is replaced. Both changes put a module
/// there that this loader refuses, so a load that consulted the package path or
/// its inode for anything would fail instead of returning a library.
#[test]
fn a_real_module_loads_the_bytes_that_verified_after_its_package_file_changes() {
    let fixture = TempDirectory::new("runtime-bundle-abi-native-loader");
    let package = build_fixture_module(
        fixture.path(),
        "authentic",
        BUNDLE_ABI_VERSION_V1,
        FixtureEntryV1::Table,
    );
    let authenticated = fs::read(&package).expect("the fixture module reads");

    // A second real module, refused only for advertising a version this loader
    // does not implement. It is a valid, signed object either way, so the two
    // are told apart by what the loader does with them and nothing else.
    let refused = build_fixture_module(
        fixture.path(),
        "refused",
        REFUSED_ABI_VERSION,
        FixtureEntryV1::Table,
    );
    let refused_bytes = fs::read(&refused).expect("the refused fixture module reads");
    assert!(
        matches!(
            BundleLibraryV1::open_verified_image(&refused_bytes, "refused"),
            Err(BundleLoaderError::InvalidApi(BundleAbiError::AbiVersion {
                actual: REFUSED_ABI_VERSION,
                ..
            }))
        ),
        "the refused fixture module must be distinguishable through the loader"
    );

    // Rewrite the inode the package path named when it verified.
    let mut resident = OpenOptions::new()
        .write(true)
        .open(&package)
        .expect("the fixture module opens for writing");
    resident
        .write_all(&refused_bytes)
        .expect("the fixture module inode is writable");
    resident
        .set_len(u64::try_from(refused_bytes.len()).expect("the fixture module fits in u64"))
        .expect("the fixture module inode resizes");
    resident.flush().expect("the fixture module inode flushes");
    drop(resident);
    // And replace the directory entry, which is the other half of the race.
    let replacement = build_fixture_module(
        fixture.path(),
        "replacement",
        REFUSED_ABI_VERSION,
        FixtureEntryV1::Table,
    );
    fs::rename(&replacement, &package).expect("the fixture module entry is replaceable");
    assert_ne!(
        fs::read(&package).expect("the changed package path reads"),
        authenticated,
        "the package file must actually have changed"
    );

    // The loader is given the vector that verified and nothing else, so the
    // module it maps still advertises V1 and still answers the entry call.
    let library = BundleLibraryV1::open_verified_image(&authenticated, "authentic")
        .expect("the authenticated module loads");
    drop(library);
}

/// The entry the loader reports on is the one in the module it staged.
///
/// A null API pointer is not something a file can be inspected for. It is what
/// the loaded code returns when it is called, so reporting it means the staged
/// copy was mapped and its entry ran. Bytes that are no object file at all
/// never get that far.
#[test]
fn the_loader_calls_the_entry_in_the_module_it_staged() {
    let fixture = TempDirectory::new("runtime-bundle-abi-native-entry");
    let null_entry = build_fixture_module(
        fixture.path(),
        "null_entry",
        BUNDLE_ABI_VERSION_V1,
        FixtureEntryV1::Null,
    );
    let bytes = fs::read(&null_entry).expect("the fixture module reads");

    assert!(matches!(
        BundleLibraryV1::open_verified_image(&bytes, "null_entry"),
        Err(BundleLoaderError::NullApi { .. })
    ));
    assert!(matches!(
        BundleLibraryV1::open_verified_image(b"not an object file", "junk"),
        Err(BundleLoaderError::Open { .. })
    ));
}

/// The chat frontend export is optional: a module without it has none, and
/// one with it is reached through the checked table.
#[test]
fn the_frontend_export_is_found_only_in_modules_that_have_one() {
    let fixture = TempDirectory::new("runtime-bundle-abi-native-frontend");
    let plain = build_fixture_module(
        fixture.path(),
        "without_frontend",
        BUNDLE_ABI_VERSION_V1,
        FixtureEntryV1::Table,
    );
    let library = BundleLibraryV1::open_verified_image(
        &fs::read(plain).expect("the fixture module reads"),
        "without_frontend",
    )
    .expect("the module loads");
    assert!(library.frontend_v1().unwrap().is_none());

    let module = build_fixture_module(
        fixture.path(),
        "with_frontend",
        BUNDLE_ABI_VERSION_V1,
        FixtureEntryV1::Frontend,
    );
    let library = BundleLibraryV1::open_verified_image(
        &fs::read(module).expect("the fixture module reads"),
        "with_frontend",
    )
    .expect("the module loads");
    let (frontend, answer) = library
        .frontend_v1()
        .unwrap()
        .expect("the module exports a frontend")
        .open(b"component")
        .unwrap();
    assert_eq!(answer, b"component");
    assert_eq!(frontend.render(b"chat", 1 << 20).unwrap(), b"chat");
    assert_eq!(frontend.constraint(b"", 1 << 20).unwrap(), b"");
    let mut parser = frontend.open_parser(b"request").unwrap();
    assert_eq!(parser.push(b"text", true, 1 << 20).unwrap(), b"text");
    parser.close().unwrap();
    frontend.close().unwrap();
}

/// A frontend export and the handles opened from it keep their module
/// mapped: the library, and then the export, can be dropped first.
#[test]
fn frontend_handles_keep_their_module_after_the_library_is_dropped() {
    let fixture = TempDirectory::new("runtime-bundle-abi-native-frontend-lifetime");
    let module = build_fixture_module(
        fixture.path(),
        "frontend_lifetime",
        BUNDLE_ABI_VERSION_V1,
        FixtureEntryV1::Frontend,
    );
    let library = BundleLibraryV1::open_verified_image(
        &fs::read(module).expect("the fixture module reads"),
        "frontend_lifetime",
    )
    .expect("the module loads");
    let export = library
        .frontend_v1()
        .unwrap()
        .expect("the module exports a frontend");
    drop(library);
    let (frontend, _) = export.open(b"component").unwrap();
    let mut parser = frontend.open_parser(b"request").unwrap();
    drop(export);
    assert_eq!(frontend.render(b"chat", 1 << 20).unwrap(), b"chat");
    // The frontend is dropped before its parser, which keeps it open.
    drop(frontend);
    assert_eq!(parser.push(b"text", true, 1 << 20).unwrap(), b"text");
    parser.close().unwrap();
}

/// Retirement support requires both a complete advertised tail and a callback.
#[test]
fn retirement_extension_is_detected_in_real_modules() {
    let fixture = TempDirectory::new("runtime-bundle-abi-native-retirement");
    for (name, entry, supported) in [
        ("old_exact_size", FixtureEntryV1::Table, false),
        ("complete_retirement", FixtureEntryV1::Retirement, true),
        (
            "incomplete_retirement",
            FixtureEntryV1::IncompleteRetirement,
            false,
        ),
        ("null_retirement", FixtureEntryV1::NullRetirement, false),
    ] {
        let module = build_fixture_module(fixture.path(), name, BUNDLE_ABI_VERSION_V1, entry);
        let bytes = fs::read(module).expect("the fixture module reads");
        let library = BundleLibraryV1::open_verified_image(&bytes, name)
            .expect("the compatible V1 module loads");
        assert_eq!(library.supports_request_retirement(), supported, "{name}");
    }
}

#[test]
fn loaded_instance_forwards_retirement_and_pins_library_after_failure() {
    let fixture = TempDirectory::new("runtime-bundle-abi-native-retirement-forwarding");
    let module = build_fixture_module(
        fixture.path(),
        "retirement_forwarding",
        BUNDLE_ABI_VERSION_V1,
        FixtureEntryV1::Retirement,
    );
    let bytes = fs::read(module).expect("the fixture module reads");
    let library = BundleLibraryV1::open_verified_image(&bytes, "retirement_forwarding")
        .expect("the extended module loads");
    let host = || HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
    let mut instance = library.create(host(), &[]).expect("the fixture creates");
    let mut observer = library.create(host(), &[]).expect("the observer creates");
    drop(library);
    assert_eq!(observer.library_ref_count(), 2);
    assert!(instance.supports_request_retirement());
    instance
        .retire_request(0x1234_5678_9abc_def0)
        .expect("the complete request ID reaches the loaded callback");
    assert!(!instance.is_poisoned());
    assert!(instance.retire_request(42).is_err());
    assert!(instance.is_poisoned());
    assert!(matches!(
        instance.retire_request(0x1234_5678_9abc_def0),
        Err(BundleAbiError::Poisoned)
    ));
    assert!(matches!(instance.commit(), Err(BundleAbiError::Poisoned)));
    assert!(matches!(instance.cancel(), Err(BundleAbiError::Poisoned)));
    assert!(matches!(instance.destroy(), Err(BundleAbiError::Poisoned)));
    drop(instance);
    assert_eq!(
        observer.library_ref_count(),
        2,
        "poisoned code stays loaded"
    );
    observer
        .retire_request(0x1234_5678_9abc_def0)
        .expect("the retained library remains callable");
    observer.destroy().expect("the healthy observer destroys");
    drop(observer);

    // This entry returns null, unlike the pinned module's usable API table.
    // Loading authenticated new bytes must call this new entry in this process.
    let different = build_fixture_module(
        fixture.path(),
        "different_after_poison",
        BUNDLE_ABI_VERSION_V1,
        FixtureEntryV1::Null,
    );
    let different_bytes = fs::read(different).expect("the different fixture reads");
    assert!(
        matches!(
            BundleLibraryV1::open_verified_image(&different_bytes, "different_after_poison"),
            Err(BundleLoaderError::NullApi { .. })
        ),
        "the new authenticated entry must run instead of the pinned entry"
    );
}

/// What a fixture module's V1 entry hands back when the loader calls it.
#[derive(Clone, Copy)]
enum FixtureEntryV1 {
    Table,
    /// The V1 table, and a chat frontend table under its optional export.
    Frontend,
    Null,
    Retirement,
    IncompleteRetirement,
    NullRetirement,
}

/// Compiles one real V1 module: a shared object exporting the V1 entry symbol.
///
/// The advertised ABI version is a parameter so a test can build a module this
/// loader refuses. That is what makes "the package file changed and the load
/// still succeeded" say something: the refused module is what the changed file
/// holds by then.
fn build_fixture_module(
    directory: &Path,
    name: &str,
    abi_version: u32,
    entry: FixtureEntryV1,
) -> PathBuf {
    let source = directory.join(format!("{name}.rs"));
    let module = directory.join(format!("{name}.{}", fixture_module_extension()));
    let returned = match entry {
        FixtureEntryV1::Table | FixtureEntryV1::Frontend => "&raw const API",
        FixtureEntryV1::Null => "core::ptr::null()",
        FixtureEntryV1::Retirement
        | FixtureEntryV1::IncompleteRetirement
        | FixtureEntryV1::NullRetirement => "&raw const EXTENDED.base",
    };
    let extended_size = match entry {
        FixtureEntryV1::IncompleteRetirement => size_of::<BundleApiWithRequestRetirementV1>() - 1,
        _ => size_of::<BundleApiWithRequestRetirementV1>(),
    };
    let retirement_callback = match entry {
        FixtureEntryV1::NullRetirement => "None",
        _ => "Some(retire_request)",
    };
    fs::write(
        &source,
        format!(
            r#"
#[repr(C)]
pub struct ApiV1 {{
    struct_size: u32,
    abi_version: u32,
    // SAFETY-REVIEWED: the generated ABI test fixture uses typed callbacks in a local process
    create: Option<unsafe extern "C" fn(*const core::ffi::c_void, ByteView) -> CreateResult>,
    // SAFETY-REVIEWED: the generated ABI test fixture uses typed callbacks in a local process
    execute: Option<unsafe extern "C" fn(Handle, ByteView, ResultSink) -> Status>,
    commit: Option<extern "C" fn(Handle) -> Status>,
    cancel: Option<extern "C" fn(Handle) -> Status>,
    destroy: Option<extern "C" fn(Handle) -> Status>,
}}

#[repr(C)]
pub struct Handle {{ raw: u64 }}
#[repr(C)]
pub struct ByteView {{ data: *const u8, length: u64 }}
#[repr(C)]
pub struct CreateResult {{ struct_size: u32, status: Status, handle: Handle }}
#[repr(C)]
pub struct LogitsRow {{
    struct_size: u32, reserved: u32, request_id: u64,
    logit_count: u64, logit_bits: ByteView,
}}
#[repr(C)]
pub struct ResultSink {{
    struct_size: u32, reserved: u32, user_data: *mut core::ffi::c_void,
    maximum_rows: u64, maximum_logits_per_row: u64, maximum_total_logits: u64,
    // SAFETY-REVIEWED: the generated ABI test fixture uses typed callbacks in a local process
    emit_logits_row: Option<unsafe extern "C" fn(*mut core::ffi::c_void, LogitsRow) -> Status>,
}}

// SAFETY-REVIEWED: the generated ABI test fixture uses typed callbacks in a local process
unsafe extern "C" fn create(_: *const core::ffi::c_void, _: ByteView) -> CreateResult {{
    CreateResult {{
        struct_size: core::mem::size_of::<CreateResult>() as u32,
        status: Status(0), handle: Handle {{ raw: 0x100000001 }},
    }}
}}
// SAFETY-REVIEWED: the generated ABI test fixture uses typed callbacks in a local process
unsafe extern "C" fn execute(_: Handle, _: ByteView, _: ResultSink) -> Status {{ Status(0) }}
extern "C" fn commit(_: Handle) -> Status {{ Status(0) }}
extern "C" fn cancel(_: Handle) -> Status {{ Status(0) }}
extern "C" fn destroy(_: Handle) -> Status {{ Status(0) }}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Status(pub u32);

#[repr(C)]
pub struct Extended {{
    base: ApiV1,
    retire_request_v1: Option<extern "C" fn(Handle, u64) -> Status>,
}}

extern "C" fn retire_request(handle: Handle, request_id: u64) -> Status {{
    if handle.raw == 0x100000001 && request_id == 0x1234_5678_9abc_def0 {{
        Status(0)
    }} else {{
        Status(8)
    }}
}}

static EXTENDED: Extended = Extended {{
    base: ApiV1 {{
        struct_size: {extended_size},
        abi_version: {abi_version},
        create: Some(create),
        execute: Some(execute),
        commit: Some(commit),
        cancel: Some(cancel),
        destroy: Some(destroy),
    }},
    retire_request_v1: {retirement_callback},
}};

static API: ApiV1 = ApiV1 {{
    struct_size: {struct_size},
    abi_version: {abi_version},
    create: Some(create),
    execute: Some(execute),
    commit: Some(commit),
    cancel: Some(cancel),
    destroy: Some(destroy),
}};

#[unsafe(no_mangle)]
pub extern "C" fn chop_bundle_get_api_v1() -> *const ApiV1 {{
    {returned}
}}
"#,
            struct_size = size_of::<BundleRawApiV1>(),
        ),
    )
    .expect("the fixture module source is writable");
    if matches!(entry, FixtureEntryV1::Frontend) {
        let frontend = FRONTEND_FIXTURE_SOURCE.replace(
            "TABLE_SIZE",
            &size_of::<runtime_bundle_abi::frontend::BundleFrontendTableV1>().to_string(),
        );
        let mut file = OpenOptions::new()
            .append(true)
            .open(&source)
            .expect("the fixture module source reopens");
        file.write_all(frontend.as_bytes())
            .expect("the fixture frontend source is writable");
    }

    let compiler = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = Command::new(compiler)
        .args(["--edition=2024", "--crate-type=cdylib"])
        .arg(&source)
        .arg("-o")
        .arg(&module)
        .output()
        .expect("the fixture module compiler starts");
    assert!(
        output.status.success(),
        "the fixture module compiler failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    module
}

/// A frontend whose every answer echoes its input: open and parser_open
/// hand back the address of one static byte.
const FRONTEND_FIXTURE_SOURCE: &str = r#"
#[repr(C)]
pub struct FrontendSink {
    struct_size: u32,
    layout: u32,
    context: *mut core::ffi::c_void,
    // SAFETY-REVIEWED: the generated ABI test fixture uses typed callbacks in a local process
    emit: Option<unsafe extern "C" fn(*mut core::ffi::c_void, ByteView) -> Status>,
    // SAFETY-REVIEWED: the generated ABI test fixture uses typed callbacks in a local process
    emit_error: Option<unsafe extern "C" fn(*mut core::ffi::c_void, ByteView) -> Status>,
}

#[repr(C)]
pub struct FrontendTable {
    struct_size: u32,
    layout: u32,
    // SAFETY-REVIEWED: the generated ABI test fixture uses typed callbacks in a local process
    open: Option<unsafe extern "C" fn(ByteView, FrontendSink, *mut *const u8) -> Status>,
    close: Option<extern "C" fn(*const u8) -> Status>,
    // SAFETY-REVIEWED: the generated ABI test fixture uses typed callbacks in a local process
    render: Option<unsafe extern "C" fn(*const u8, ByteView, FrontendSink) -> Status>,
    // SAFETY-REVIEWED: the generated ABI test fixture uses typed callbacks in a local process
    constraint: Option<unsafe extern "C" fn(*const u8, ByteView, FrontendSink) -> Status>,
    // SAFETY-REVIEWED: the generated ABI test fixture uses typed callbacks in a local process
    parser_open: Option<unsafe extern "C" fn(*const u8, ByteView, FrontendSink, *mut *const u8) -> Status>,
    // SAFETY-REVIEWED: the generated ABI test fixture uses typed callbacks in a local process
    parser_push: Option<unsafe extern "C" fn(*const u8, ByteView, u32, FrontendSink) -> Status>,
    parser_close: Option<extern "C" fn(*const u8) -> Status>,
}

static OPENED: u8 = 0;

// SAFETY-REVIEWED: the generated ABI test fixture answers with the component and writes the host's output pointer
unsafe extern "C" fn frontend_open(component: ByteView, sink: FrontendSink, out: *mut *const u8) -> Status {
    // SAFETY-REVIEWED: the host lends its sink, the component and a writable output pointer for the call
    unsafe {
        (sink.emit.unwrap())(sink.context, component);
        *out = &raw const OPENED;
    }
    Status(0)
}
extern "C" fn frontend_close(_: *const u8) -> Status { Status(0) }
// SAFETY-REVIEWED: the generated ABI test fixture echoes through the host's sink
unsafe extern "C" fn frontend_echo(_: *const u8, request: ByteView, sink: FrontendSink) -> Status {
    // SAFETY-REVIEWED: the host lends its sink and request for the call
    unsafe { (sink.emit.unwrap())(sink.context, request) }
}
// SAFETY-REVIEWED: the generated ABI test fixture writes the host's output pointer
unsafe extern "C" fn frontend_parser_open(_: *const u8, _: ByteView, _: FrontendSink, out: *mut *const u8) -> Status {
    // SAFETY-REVIEWED: the host lends a writable output pointer for the call
    unsafe { *out = &raw const OPENED };
    Status(0)
}
// SAFETY-REVIEWED: the generated ABI test fixture echoes through the host's sink
unsafe extern "C" fn frontend_push(_: *const u8, text: ByteView, _: u32, sink: FrontendSink) -> Status {
    // SAFETY-REVIEWED: the host lends its sink and text for the call
    unsafe { (sink.emit.unwrap())(sink.context, text) }
}

static FRONTEND: FrontendTable = FrontendTable {
    struct_size: TABLE_SIZE,
    layout: 1,
    open: Some(frontend_open),
    close: Some(frontend_close),
    render: Some(frontend_echo),
    constraint: Some(frontend_echo),
    parser_open: Some(frontend_parser_open),
    parser_push: Some(frontend_push),
    parser_close: Some(frontend_close),
};

#[unsafe(no_mangle)]
pub extern "C" fn chop_bundle_frontend_api_v1() -> *const FrontendTable {
    &raw const FRONTEND
}
"#;

fn fixture_module_extension() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "dylib"
    }
    #[cfg(not(target_os = "macos"))]
    {
        "so"
    }
}

struct TempDirectory {
    path: PathBuf,
}

impl TempDirectory {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);

        let path = std::env::temp_dir().join(format!(
            "{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).expect("the fixture temporary directory is creatable");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
