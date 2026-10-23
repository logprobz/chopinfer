//! CPU checks of the native grouped expert tiles: the host emulation of
//! their operand layout, a compile of every lab tile for both projections
//! without spills, TF32 or float atomics, and the quad-lane builds of the
//! served gate/up tiles beside their row-tiled twins.
use kernel_build::{BuildOptions, CubinSpec, Toolchain};
use kernels_sm120::moe::native_grouped::{self, DOWN_TILES, GATE_UP_TILES, Shape, Tune, Variant};
use std::path::PathBuf;
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
fn operand_layout_matches_the_instruction_fragments() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = output_dir("native-layout");
    let binary = dir.join("layout");
    let compiled = Command::new("c++")
        .args(["-O2", "-std=c++17", "-ffp-contract=off", "-I"])
        .arg(root.join("cuda"))
        .arg(root.join("tests/native_grouped_layout.cpp"))
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
        "NATIVE_LAYOUT_COMPLETE groups=800 negative_control_rejected=1 untangled=256\n"
    );
}

/// Two expert geometries: full-width columns and a small one with a
/// partial column strip and an unaligned scale row.
fn shapes() -> Vec<(Shape, &'static [native_grouped::Tile])> {
    let clamp = Some((0x40e0_0000, 0x40e0_0000));
    vec![
        (
            Shape::new(384, 6, 5120, 576, true, 8192, clamp).unwrap(),
            &GATE_UP_TILES[..],
        ),
        (
            Shape::new(384, 6, 576, 5120, false, 8192, None).unwrap(),
            &DOWN_TILES[..],
        ),
        (
            Shape::new(128, 3, 2048, 80, true, 512, clamp).unwrap(),
            &GATE_UP_TILES[..1],
        ),
        (
            Shape::new(128, 3, 160, 2048, false, 512, None).unwrap(),
            &DOWN_TILES[..1],
        ),
    ]
}

#[test]
fn native_tiles_compile_without_spills_tf32_or_float_atomics() {
    let toolchain = Toolchain::detect().unwrap();
    assert!(
        !toolchain.skipped,
        "CUDA is required for the native tile compile gate"
    );
    let out = output_dir("native-tiles");
    let instances = shapes()
        .into_iter()
        .flat_map(|(shape, tiles)| tiles.iter().map(move |tile| (shape, *tile)))
        .collect::<Vec<_>>();
    let spec = native_grouped::cubin_spec("chop_moe_native_compile", &instances, true).unwrap();
    let expected = spec.entries.len();
    // Pack, serving, omit-first and (gate/up) raw entries, and the probe.
    let gate_up = instances.iter().filter(|(s, _)| s.gate_up).count();
    assert_eq!(expected, 3 * instances.len() + gate_up + 1);
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
    .expect("compile the native grouped tiles");
    assert_eq!(built[0].entries.len(), expected);
    for entry in &built[0].entries {
        assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
        println!(
            "native tile {} registers {} static shared {}",
            entry.symbol, entry.registers, entry.shared_static_bytes
        );
    }
    let sass = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&built[0].path)
        .output()
        .unwrap();
    assert!(sass.status.success());
    let sass = String::from_utf8(sass.stdout).unwrap();
    let functions = sass.split("Function : ").skip(1).collect::<Vec<_>>();
    assert_eq!(functions.len(), expected);
    for function in functions {
        let name = function.lines().next().unwrap();
        for forbidden in [
            "TF32",
            "HMMA",
            "RED.E.ADD.F32",
            "ATOM.E.ADD.F32",
            "REDG.E.ADD.F32",
        ] {
            assert!(
                !function.contains(forbidden),
                "{name}: forbidden {forbidden}"
            );
        }
        if name.contains("_tiles_") || name.starts_with(native_grouped::PROBE) {
            // The block-scaled instruction and one integer ticket per item.
            assert!(
                function.contains("OMMA") || function.contains("QMMA"),
                "{name}: no native MMA"
            );
            assert_eq!(
                function.contains("ATOMG"),
                name.contains("_tiles_"),
                "{name}: one ticket per item"
            );
        } else {
            assert!(!function.contains("MMA"), "{name}");
        }
    }
    std::fs::remove_dir_all(out).unwrap();
}

#[test]
fn fused_ceil_tiles_compile_without_spills() {
    let toolchain = Toolchain::detect().unwrap();
    assert!(
        !toolchain.skipped,
        "CUDA is required for the fused ceil compile gate"
    );
    let out = output_dir("native-ceil");
    let ceil = native_grouped::Ceil {
        floor_bits: 1e-4_f32.to_bits(),
    };
    let flash = Shape::new(
        384,
        6,
        5120,
        576,
        true,
        8192,
        Some((0x40e0_0000, 0x40e0_0000)),
    )
    .unwrap();
    let instances: Vec<_> = GATE_UP_TILES.iter().map(|t| (flash, *t)).collect();
    let spec =
        native_grouped::ceil_cubin_spec("chop_moe_native_ceil_compile", &ceil, &instances).unwrap();
    // One fused tile entry per tile and one shared fixup.
    assert_eq!(spec.entries.len(), GATE_UP_TILES.len() + 1);
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
    .expect("compile the fused ceil tiles");
    for entry in &built[0].entries {
        assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
        println!(
            "fused ceil {} registers {} static shared {}",
            entry.symbol, entry.registers, entry.shared_static_bytes
        );
    }
    std::fs::remove_dir_all(out).unwrap();
}

fn sass(cuobjdump: &std::path::Path, cubin: &std::path::Path, symbol: &str) -> String {
    let output = Command::new(cuobjdump)
        .args(["-sass", "-fun", symbol])
        .arg(cubin)
        .output()
        .unwrap();
    assert!(output.status.success(), "cuobjdump {symbol}");
    String::from_utf8(output.stdout).unwrap()
}

/// DeepSeek V4.1 Flash's served TP4 gate/up tiles (the fused ceil at the
/// first gate/up tile and the scanned loop at the wide tile), each built
/// over row tiles and over quad-lane tiles: every quad-lane build compiles
/// without spills at its twin's ABI and launch bound, with the same shared
/// memory and as many native MMAs, atomics and shuffles.
#[test]
fn lane_tiles_build_beside_their_row_tiled_twins() {
    let toolchain = Toolchain::detect().unwrap();
    assert!(
        !toolchain.skipped,
        "CUDA is required for the quad-lane tile compile gate"
    );
    let flash = Shape::new(
        384,
        6,
        5120,
        576,
        true,
        8192,
        Some((0x40e0_0000, 0x40e0_0000)),
    )
    .unwrap();
    let ceil = native_grouped::Ceil {
        floor_bits: 1e-4_f32.to_bits(),
    };
    let fused = ceil
        .entries(&flash, &GATE_UP_TILES[0])
        .unwrap()
        .into_iter()
        .find(|e| e.symbol == ceil.tiles_symbol(&flash, &GATE_UP_TILES[0]))
        .unwrap();
    let scanned = native_grouped::tune_entries(
        &flash,
        &GATE_UP_TILES[2],
        &[Tune::Scanned],
        Variant::Serving,
    )
    .unwrap()
    .remove(0);
    let mut entries = Vec::new();
    for served in [&fused, &scanned] {
        entries.push(native_grouped::row_tiles_entry(&flash, served).unwrap());
        entries.push(native_grouped::lanes_entry(&flash, served).unwrap());
    }
    let spec = CubinSpec {
        name: "chop_moe_native_lanes_compile".into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: Vec::new(),
        headers: vec!["#include \"moe/native_ceil.cuh\"".into()],
        defines: Vec::new(),
        entries: entries.clone(),
    };
    let out = output_dir("native-lanes");
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
    .expect("compile the quad-lane native tiles");
    let cuobjdump = toolchain.home.join("bin/cuobjdump");
    for pair in entries.chunks(2) {
        let (tiled, lanes) = (&pair[0], &pair[1]);
        assert_eq!(
            lanes.symbol,
            native_grouped::lanes_symbol(&tiled.symbol.replace("_row_tiles_v1", "_v1"))
        );
        assert_eq!(
            (&lanes.params, lanes.max_threads, lanes.min_blocks),
            (&tiled.params, tiled.max_threads, tiled.min_blocks)
        );
        let report = |symbol: &str| {
            built[0]
                .entries
                .iter()
                .find(|e| e.symbol == symbol)
                .unwrap()
        };
        let (t, l) = (report(&tiled.symbol), report(&lanes.symbol));
        assert_eq!(
            (l.local_bytes, l.stack_bytes),
            (0, 0),
            "{} spills",
            l.symbol
        );
        assert_eq!(l.shared_static_bytes, t.shared_static_bytes, "{}", l.symbol);
        println!(
            "native lanes {} registers {} (row tiles {})",
            l.symbol, l.registers, t.registers
        );
        let (ts, ls) = (
            sass(&cuobjdump, &built[0].path, &tiled.symbol),
            sass(&cuobjdump, &built[0].path, &lanes.symbol),
        );
        for opcode in ["OMMA", "QMMA", "ATOMG", "SHFL"] {
            assert_eq!(
                ls.matches(opcode).count(),
                ts.matches(opcode).count(),
                "{opcode}"
            );
        }
        for forbidden in ["TF32", "HMMA", "RED.E.ADD.F32", "ATOM.E.ADD.F32"] {
            assert!(!ls.contains(forbidden), "{}: {forbidden}", lanes.symbol);
        }
    }
    std::fs::remove_dir_all(out).unwrap();
}
