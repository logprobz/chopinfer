//! Audit the clustered query RoPE and window write twin without opening CUDA.
use kernel_build::{BuildOptions, Toolchain};
use kernel_core::latent_attention::LatentKvFormat;
use kernels_sm120::latent::query_window_twin::{self, Entry};

#[test]
fn query_window_twin_compiles_without_spills() {
    let toolchain = Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let out = std::env::temp_dir().join(format!("chopinfer-query-window-twin-{stamp}"));
    std::fs::create_dir(&out).unwrap();
    let entries = [
        (LatentKvFormat::Fp8E4M3Ue8M0Block32, "native"),
        (LatentKvFormat::Bf16, "bf16"),
    ]
    .map(|(kv, tag)| Entry {
        kv,
        capacity: 32,
        symbol: format!("query_window_twin_{tag}_test_v1"),
    });
    let spec = query_window_twin::cubin_spec("query_window_twin_test_sm120", &entries).unwrap();
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
    assert_eq!(built[0].entries.len(), 2);
    for entry in &built[0].entries {
        assert_eq!(entry.local_bytes, 0);
        assert_eq!(entry.stack_bytes, 0);
        // The reported size includes the 1 KiB the toolchain reserves.
        assert!(
            entry.shared_static_bytes <= 1024 + 128,
            "{}",
            entry.shared_static_bytes
        );
    }
    std::fs::remove_dir_all(out).unwrap();
}
