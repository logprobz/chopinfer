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
    let sass = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&built[0].path)
        .output()
        .expect("inspect the twin's SASS");
    assert!(sass.status.success());
    let text = String::from_utf8_lossy(&sass.stdout);
    let functions = text.split("Function : ").skip(1).collect::<Vec<_>>();
    assert_eq!(functions.len(), 2);
    for function in functions {
        let name = function.lines().next().unwrap_or_default().to_owned();
        let opcodes: Vec<&str> = function
            .lines()
            .filter_map(|line| line.split_once("*/").map(|(_, rest)| rest.trim()))
            .filter(|op| !op.is_empty() && !op.starts_with("/*"))
            .collect();
        // Both halves arrive, and both wait.
        assert!(
            opcodes.iter().any(|op| op.starts_with("UCGABAR_ARV")),
            "{name}"
        );
        assert!(
            opcodes.iter().any(|op| op.starts_with("UCGABAR_WAIT")),
            "{name}"
        );
        // A cluster wait is warp-wide: one that a warp reconverges for
        // (WARPSYNC.ALL just before it) is a lane waiting alone while its
        // warp may wait at a CTA barrier for that lane.
        for (at, op) in opcodes.iter().enumerate() {
            if op.starts_with("UCGABAR_WAIT") {
                assert!(
                    !opcodes[at.saturating_sub(2)..at]
                        .iter()
                        .any(|op| op.starts_with("WARPSYNC")),
                    "{name}: a divergent cluster wait at instruction {at}"
                );
            }
        }
    }
    std::fs::remove_dir_all(out).unwrap();
}
