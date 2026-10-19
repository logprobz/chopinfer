//! Audit the row-wide query rotation candidate without opening CUDA.
use kernel_build::{BuildOptions, Toolchain};
use kernels_sm120::latent::query_parallel::{self, Entry};

#[test]
fn parallel_query_compiles_without_spills() {
    let toolchain = Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let out = std::env::temp_dir().join(format!("chopinfer-query-parallel-{stamp}"));
    std::fs::create_dir(&out).unwrap();
    let entry = Entry {
        capacity: 192,
        symbol: "query_parallel_test_v1".into(),
    };
    let spec = query_parallel::cubin_spec("query_parallel_test_sm120", &[entry]);
    let built = kernel_build::build(
        &toolchain,
        &[spec],
        &out,
        &out.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].entries.len(), 1);
    let entry = &built[0].entries[0];
    assert_eq!(entry.local_bytes, 0);
    assert_eq!(entry.stack_bytes, 0);
    assert!(entry.shared_static_bytes <= 101_376);
    std::fs::remove_dir_all(out).unwrap();
}
