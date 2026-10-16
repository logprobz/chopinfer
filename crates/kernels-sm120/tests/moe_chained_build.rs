//! Compile and audit every served chained expert instance (the dense
//! projections in their schedules; the split's routed projections in every
//! schedule but tile pairs, which mark their live fragments with a
//! shared-memory `atomicOr`, routes, route sums and the sum that adds a
//! row) without loading a cubin: sm_120f, exact symbols, no spills, shared
//! memory within the SM120 limit, and the intended SASS.

use kernel_build::{BuildOptions, Toolchain};
use kernels_sm120::moe::chained;
use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn chained_instances_build_with_bf16_mma_and_no_float_atomics() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 toolchain");
    assert!(
        !toolchain.skipped,
        "chained MoE compile audit requires Linux and CUDA"
    );
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!(
        "chopinfer-moe-chained-build-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&out_dir).expect("create test output directory");
    let mut specs = chained::cubin_specs();
    specs.extend(chained::block_warp_cubin_specs());
    specs.extend(
        chained::split::cubin_specs()
            .expect("the split cubins")
            .into_iter()
            .filter(|spec| !spec.name.ends_with("_tp")),
    );
    let projections = [
        chained::catalog(),
        chained::block_warp_catalog(),
        chained::split::forms(),
    ]
    .concat();
    let built = kernel_build::build(
        &toolchain,
        &specs,
        &out_dir,
        &out_dir.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("compile and audit chained MoE cubins");
    assert_eq!(built.len(), specs.len());
    assert_eq!(
        built.iter().map(|c| c.entries.len()).sum::<usize>(),
        specs.iter().map(|s| s.entries.len()).sum::<usize>()
    );
    for cubin in &built {
        for entry in &cubin.entries {
            assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
            assert!(
                entry.shared_static_bytes <= 48 * 1024,
                "{} exceeds static shared memory",
                entry.symbol
            );
        }
        let output = Command::new(toolchain.home.join("bin/cuobjdump"))
            .arg("-sass")
            .arg(&cubin.path)
            .output()
            .expect("cuobjdump SASS");
        assert!(
            output.status.success(),
            "cuobjdump failed for {}",
            cubin.name
        );
        let sass = String::from_utf8(output.stdout).expect("SASS is UTF-8");
        for atomic in ["ATOM", "RED.E.", "REDG"] {
            assert!(!sass.contains(atomic), "{} uses {atomic}", cubin.name);
        }
        if cubin.name.ends_with("_route") {
            // Warp selection reduces ordered keys and minimum IDs.
            assert!(
                sass.contains("REDUX"),
                "{} lacks warp reductions",
                cubin.name
            );
            assert!(!sass.contains("HMMA"), "{} has an MMA", cubin.name);
            continue;
        }
        if cubin.name.ends_with("_sum_add") {
            assert!(!sass.contains("HMMA"), "{} has an MMA", cubin.name);
            continue;
        }
        let entries = projections
            .iter()
            .filter(|p| cubin.entries.iter().any(|e| e.symbol == p.symbol))
            .count();
        assert_eq!(entries, cubin.entries.len(), "{} entries", cubin.name);
        // BF16 operands, F32 accumulation; never TF32 or narrow MMA forms.
        assert!(
            sass.contains("HMMA.16816.F32.BF16"),
            "{} lacks BF16 MMA",
            cubin.name
        );
        for other in ["HMMA.1684", "TF32", "QMMA", "OMMA"] {
            assert!(!sass.contains(other), "{} emits {other}", cubin.name);
        }
        if cubin.name.ends_with("_bw4") || cubin.name.ends_with("_ws") {
            // Block warps copy by cp.async and hand chains over shared
            // memory, never through global scratch.
            assert!(sass.contains("LDGSTS"), "{} lacks cp.async", cubin.name);
        }
        if cubin.name.ends_with("_nvfp4") {
            assert!(
                sass.contains("HMUL2.BF16"),
                "{} lacks BF16 scale products",
                cubin.name
            );
        }
    }
    fs::remove_dir_all(out_dir).expect("remove hermetic test output");
}

#[test]
fn warp_streams_build_at_the_shortest_k_their_layouts_admit() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 toolchain");
    assert!(
        !toolchain.skipped,
        "warp-stream compile audit requires Linux and CUDA"
    );
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!(
        "chopinfer-moe-warp-streams-build-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&out_dir).expect("create test output directory");
    // Gate/up's two stages admit two rounds (K 1024), down's four stages
    // four rounds (K 2048); one round less is refused. The split serves
    // its decode gate/up in warp streams; its decode down's query (K 512)
    // takes the down's layout once K is long enough.
    let decode = |gate_up: bool| {
        chained::split::catalog()
            .into_iter()
            .find(|p| {
                p.weights() == chained::Weights::Fp8Block128
                    && p.row_tile == 16
                    && p.gate_up() == gate_up
            })
            .expect("the split's FP8 decode projections")
    };
    let mut shortest = Vec::new();
    for served in [decode(true), decode(false)] {
        let k = if served.gate_up() { 1024 } else { 2048 };
        for (k, admitted) in [(k - 512, false), (k, true)] {
            let mut query = served.query;
            query.gemm.k = k;
            let p = chained::Projection::new(query, served.row_tile, served.tokens_capacity)
                .expect("a chained projection")
                .warp_streams();
            assert_eq!(p.is_ok(), admitted, "{} at K {k}", served.symbol);
            shortest.extend(p);
        }
    }
    assert_eq!(shortest.len(), 2);
    let specs = chained::projection_cubin_specs("chop_moe_chain_shortest", &shortest);
    let built = kernel_build::build(
        &toolchain,
        &specs,
        &out_dir,
        &out_dir.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("compile the shortest admitted warp streams");
    assert_eq!(
        built.iter().map(|c| c.entries.len()).sum::<usize>(),
        shortest.len()
    );
    for entry in built.iter().flat_map(|c| &c.entries) {
        assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
        assert!(
            entry.shared_static_bytes <= 48 * 1024,
            "{} exceeds static shared memory",
            entry.symbol
        );
    }
    fs::remove_dir_all(out_dir).expect("remove hermetic test output");
}
