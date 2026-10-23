//! CPU checks of the quad-lane gate queues (`chained_lanes.cuh`) and the
//! column kernel's quad-lane load (`chained_experts.cuh`): the word-wide
//! code checks and the lane operand bytes on the host, and compiles of
//! DeepSeek V4.1 Flash's TP4 backbone gate queues and column gate/up beside
//! their row-tiled twins.
use kernel_build::{BuildOptions, CubinSpec, EntryParam, EntrySpec, Toolchain};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn output_dir(label: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("chopinfer-{label}-{}-{stamp}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    dir
}

#[test]
fn word_checks_and_lane_bytes_match_the_row_tiled_queues() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = output_dir("lanes-checks");
    let binary = dir.join("checks");
    let compiled = Command::new("c++")
        .args(["-O2", "-std=c++17", "-ffp-contract=off", "-I"])
        .arg(root.join("cuda"))
        .arg(root.join("tests/chained_lanes_checks.cpp"))
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let ran = Command::new(&binary).output().unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    assert!(
        ran.status.success(),
        "{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&ran.stdout),
        "LANES_CHECKS_COMPLETE words=1142784 groups=458752 operands=4096\n"
    );
}

/// The Flash TP4 backbone gate/up projection: 384 experts, top 6, 576
/// columns per half, K 5120.
const CONFIG: &str = "chop::moe::chained::quantized::Config<chop::moe::chained::Projection::gate_up,384,6,576,5120,64,64,chop::moe::clamp::SwigluClamp<true,1088421888u,1088421888u>>";
/// The same projection at the streamed module's 8-token capacity.
const CONFIG_T8: &str = "chop::moe::chained::quantized::Config<chop::moe::chained::Projection::gate_up,384,6,576,5120,64,8,chop::moe::clamp::SwigluClamp<true,1088421888u,1088421888u>>";

fn entry(symbol: &str, call: &str, min_blocks: u32) -> EntrySpec {
    let pointer = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    EntrySpec::new(
        symbol.into(),
        vec![
            pointer("input", "const uint8_t"),
            pointer("weight", "const uint8_t"),
            pointer("weight_scales", "const uint8_t"),
            pointer("scales", "const uint8_t"),
            pointer("offsets", "const int32_t"),
            pointer("sorted", "const int32_t"),
            pointer("padded", "const int32_t"),
            pointer("live", "const int32_t"),
            pointer("scratch", "void"),
        ],
        format!("{call}(input,weight,weight_scales,scales,offsets,sorted,padded,live,scratch);"),
        128,
        min_blocks,
    )
}

fn sass(cuobjdump: &Path, cubin: &Path, symbol: &str) -> String {
    let output = Command::new(cuobjdump)
        .args(["-sass", "-fun", symbol])
        .arg(cubin)
        .output()
        .unwrap();
    assert!(output.status.success(), "cuobjdump {symbol}");
    String::from_utf8(output.stdout).unwrap()
}

/// Each quad-lane queue builds without spills, with no more registers than
/// its row-tiled twin at the same launch bound, as many BF16 MMAs, and no
/// shuffles: the conversions read the lane's bytes in place.
#[test]
fn lane_queues_build_beside_their_row_tiled_twins() {
    let toolchain = Toolchain::detect().unwrap();
    assert!(
        !toolchain.skipped,
        "the quad-lane compile audit requires Linux and CUDA"
    );
    let pairs = [
        (
            "small",
            format!("chop::moe::streamed::compute<{CONFIG},8,true>"),
            format!("chop::moe::lanes::compute<{CONFIG},8>"),
            1,
        ),
        (
            "wide",
            format!("chop::moe::streamed::compute<{CONFIG},64,true>"),
            format!("chop::moe::lanes::compute<{CONFIG},64>"),
            1,
        ),
        (
            "resident8",
            format!("chop::moe::streamed::compute<{CONFIG},8,true>"),
            format!("chop::moe::lanes::compute<{CONFIG},8>"),
            8,
        ),
        (
            "absorb",
            format!("chop::moe::absorb::compute<{CONFIG},true>"),
            format!("chop::moe::lanes::absorbing<{CONFIG}>"),
            8,
        ),
        (
            "stream_t8_small",
            format!("chop::moe::streamed::compute<{CONFIG_T8},8,true>"),
            format!("chop::moe::lanes::compute<{CONFIG_T8},8>"),
            1,
        ),
    ];
    let spec = CubinSpec {
        name: "chained_lanes_audit".into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: Vec::new(),
        headers: vec!["#include \"moe/chained_lanes.cuh\"".into()],
        defines: Vec::new(),
        entries: pairs
            .iter()
            .flat_map(|(name, tiled, lanes, min_blocks)| {
                [
                    entry(&format!("tiles_{name}_v1"), tiled, *min_blocks),
                    entry(&format!("lanes_{name}_v1"), lanes, *min_blocks),
                ]
            })
            .collect(),
    };
    let dir = output_dir("lanes-build");
    let built = kernel_build::build(
        &toolchain,
        std::slice::from_ref(&spec),
        &dir,
        &dir.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    let cubin = &built[0];
    let report = |symbol: String| {
        cubin
            .entries
            .iter()
            .find(|e| e.symbol == symbol)
            .unwrap()
            .clone()
    };
    let cuobjdump = toolchain.home.join("bin/cuobjdump");
    for (name, ..) in &pairs {
        let (tiled, lanes) = (
            report(format!("tiles_{name}_v1")),
            report(format!("lanes_{name}_v1")),
        );
        assert_eq!(
            (lanes.local_bytes, lanes.stack_bytes),
            (0, 0),
            "{name} spills"
        );
        assert!(
            lanes.registers <= tiled.registers,
            "{name}: {} registers against {}",
            lanes.registers,
            tiled.registers
        );
        let (tiled, lanes) = (
            sass(&cuobjdump, &cubin.path, &tiled.symbol),
            sass(&cuobjdump, &cubin.path, &lanes.symbol),
        );
        assert!(tiled.contains("SHFL"), "{name}: the twin shuffles");
        assert!(!lanes.contains("SHFL"), "{name} shuffles");
        assert!(
            lanes.contains("F2FP.BF16.E2M1.UNPACK_B") && lanes.contains(".B3,"),
            "{name} converts without byte selectors"
        );
        assert_eq!(
            lanes.matches("HMMA.16816.F32.BF16").count(),
            tiled.matches("HMMA.16816.F32.BF16").count(),
            "{name} MMAs"
        );
        for other in ["TF32", "QMMA", "OMMA", "ATOM", "RED.E."] {
            assert!(!lanes.contains(other), "{name} emits {other}");
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

fn sass_counts(sass: &str, opcode: &str) -> usize {
    sass.matches(opcode).count()
}

/// The TP4 backbone column gate/up (8192 tokens, Verify above 64 rows and
/// native-off prefill) over quad-lane tiles builds beside its row-tiled
/// twin: no spills, no more registers, the same shared memory, and as many
/// MMAs, conversions and shuffles. Its shared tile is the twin's
/// (`chained_experts.cuh`), so the arithmetic is.
#[test]
fn column_lanes_build_beside_the_row_tiled_column() {
    let toolchain = Toolchain::detect().unwrap();
    assert!(
        !toolchain.skipped,
        "the quad-lane compile audit requires Linux and CUDA"
    );
    let config = |lanes: &str| {
        format!(
            "chop::moe::chained::quantized::Config<chop::moe::chained::Projection::gate_up,384,6,576,5120,64,8192,chop::moe::clamp::SwigluClamp<true,0x40e00000u,0x40e00000u>,true{lanes}>"
        )
    };
    let pointer = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let column = |symbol: &str, lanes: &str| {
        EntrySpec::new(
            symbol.into(),
            vec![
                pointer("input", "const uint8_t"),
                pointer("weight", "const uint8_t"),
                pointer("weight_scales", "const uint8_t"),
                pointer("activation_scales", "const uint8_t"),
                pointer("offsets", "const int32_t"),
                pointer("sorted_rows", "const int32_t"),
                pointer("padded_rows", "const int32_t"),
                pointer("live_tokens", "const int32_t"),
                pointer("route_weights", "const float"),
                pointer("output", "void"),
            ],
            format!(
                "chop::moe::chained::run<{}>(reinterpret_cast<const uint16_t*>(input),weight,nullptr,weight_scales,nullptr,offsets,sorted_rows,padded_rows,live_tokens,route_weights,output,activation_scales);",
                config(lanes)
            ),
            128,
            1,
        )
    };
    let spec = CubinSpec {
        name: "column_lanes_audit".into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: Vec::new(),
        headers: vec!["#include \"moe/chained_quantized.cuh\"".into()],
        defines: Vec::new(),
        entries: vec![
            column("column_tiles_v1", ""),
            column("column_lanes_v1", ",true"),
        ],
    };
    let dir = output_dir("column-lanes-build");
    let built = kernel_build::build(
        &toolchain,
        std::slice::from_ref(&spec),
        &dir,
        &dir.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    let cubin = &built[0];
    let report = |symbol: &str| cubin.entries.iter().find(|e| e.symbol == symbol).unwrap();
    let (tiled, lanes) = (report("column_tiles_v1"), report("column_lanes_v1"));
    assert_eq!((lanes.local_bytes, lanes.stack_bytes), (0, 0), "spills");
    assert!(
        lanes.registers <= tiled.registers,
        "{} registers against {}",
        lanes.registers,
        tiled.registers
    );
    assert_eq!(lanes.shared_static_bytes, tiled.shared_static_bytes);
    let cuobjdump = toolchain.home.join("bin/cuobjdump");
    let (tiled, lanes) = (
        sass(&cuobjdump, &cubin.path, "column_tiles_v1"),
        sass(&cuobjdump, &cubin.path, "column_lanes_v1"),
    );
    assert_eq!(
        sass_counts(&lanes, "HMMA.16816.F32.BF16"),
        sass_counts(&tiled, "HMMA.16816.F32.BF16")
    );
    assert_eq!(
        sass_counts(&lanes, "F2FP.BF16.E2M1.UNPACK_B"),
        sass_counts(&tiled, "F2FP.BF16.E2M1.UNPACK_B")
    );
    for other in ["SHFL", "TF32", "QMMA", "OMMA", "ATOM", "RED.E."] {
        assert_eq!(
            sass_counts(&lanes, other),
            sass_counts(&tiled, other),
            "{other}"
        );
    }
    println!(
        "column lanes {} registers, tiled {}",
        report("column_lanes_v1").registers,
        report("column_tiles_v1").registers
    );
    std::fs::remove_dir_all(dir).unwrap();
}
