//! CPU compilation and SASS audit of the shared snapshot copy.

use kernel_build::{BuildOptions, Toolchain, build};
use kernels_sm120::snapshot_copy::{SYMBOL, cubin_spec, entries};
use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

/// The entry compiles to a native SM120 cubin with no local memory or
/// stack (the view switch stays in registers), releases dependents and
/// waits before it reads, and moves aligned vectors in 128-bit accesses.
#[test]
fn the_snapshot_copy_compiles_with_pdl_and_vector_accesses_without_spills() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is required");
    assert!(!toolchain.skipped);
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("snapshot-copy-{}-{stamp}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let specs = [cubin_spec("snapshot_copy_compile_v1", &entries())];
    let built = build(
        &toolchain,
        &specs,
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("snapshot copy cubin builds and audits");
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].entries.len(), 1);
    let entry = &built[0].entries[0];
    assert_eq!(entry.symbol, SYMBOL);
    assert_eq!(entry.local_bytes, 0, "the snapshot copy spills");
    assert_eq!(entry.stack_bytes, 0, "the snapshot copy uses a stack");
    let sass = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&built[0].path)
        .output()
        .unwrap();
    assert!(sass.status.success());
    let text = String::from_utf8(sass.stdout).unwrap();
    let body = text
        .split_once(&format!("Function : {SYMBOL}\n"))
        .expect("entry SASS")
        .1;
    assert!(body.contains("ACQBULK"), "the copy lacks a PDL wait");
    assert!(body.contains("PREEXIT"), "the copy lacks a PDL release");
    assert!(body.contains("LDG.E.128"), "the copy lacks a vector load");
    assert!(body.contains("STG.E.128"), "the copy lacks a vector store");
    fs::remove_dir_all(root).unwrap();
}
