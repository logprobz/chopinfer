//! CPU checks of the native grouped expert tiles: the host emulation of
//! their operand layout, and a compile of every lab tile for both
//! projections without spills, TF32 or float atomics.
use kernel_build::{BuildOptions, Toolchain};
use kernels_sm120::moe::native_grouped::{self, DOWN_TILES, GATE_UP_TILES, Shape};
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
        "NATIVE_LAYOUT_COMPLETE groups=800 negative_control_rejected=1\n"
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
