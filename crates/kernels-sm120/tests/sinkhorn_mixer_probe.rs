use kernels_sm120::sinkhorn_hc::{Geometry, lagged, lagged_split, mixer_probe::Entry};
fn entry() -> Entry {
    Entry {
        reference: lagged_split::entry(
            lagged::LAB_PROFILE,
            Geometry {
                capacity: 32,
                ..lagged::LAB
            },
            false,
        ),
        prefix: "chop_hc_mixer_probe".into(),
    }
}
#[test]
fn finalizer_geometry_and_abi_are_exact() {
    let e = entry();
    let specs = e.specs().unwrap();
    assert_eq!(specs.len(), 6);
    assert!(specs.iter().all(|s| s.params.len() == 13));
    for rows in [1, 8] {
        assert_eq!(e.reference.plan(rows, rows).unwrap().grid, [20, rows, 1]);
        assert!(e.reference.plan(rows, rows + 1).is_err());
    }
}
#[test]
fn task_scheduling_and_shared_extent_are_explicit() {
    let source = include_str!("../cuda/sinkhorn_hc/finalizer_probe.cuh");
    for forbidden in ["blockIdx", "blockDim", "gridDim", "__shared__", "griddep_"] {
        assert!(!source.contains(forbidden), "{forbidden}");
    }
    assert_eq!((4096 + 24 * 161 + 256 + 8 + 32) * 4, 33024);
    assert!(
        include_str!("../cuda/sinkhorn_hc/finalizer_probe.cuh")
            .contains("if constexpr (Kind == 2) __syncwarp();")
    );
}
#[test]
fn finalizers_compile_without_spills_and_preserve_ordering_instructions() {
    let root = std::env::temp_dir().join(format!("hc-mixer-probes-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let tc = kernel_build::Toolchain::detect().unwrap();
    assert!(!tc.skipped);
    let built = kernel_build::build(
        &tc,
        &[entry().cubin_spec("hc_mixer_probe").unwrap()],
        &root.join("out"),
        &root.join("lock"),
        kernel_build::BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    for image in built {
        for usage in image.entries {
            assert_eq!(usage.shared_static_bytes, 16_704, "{}", usage.symbol);
            assert_eq!(
                (usage.local_bytes, usage.stack_bytes),
                (0, 0),
                "{}",
                usage.symbol
            );
            eprintln!(
                "MIXER_PROBE_RESOURCE {} regs={} shared={} local={} stack={}",
                usage.symbol,
                usage.registers,
                usage.shared_static_bytes,
                usage.local_bytes,
                usage.stack_bytes
            );
            let out = std::process::Command::new("cuobjdump")
                .args(["-sass", "-fun", &usage.symbol])
                .arg(&image.path)
                .output()
                .unwrap();
            assert!(out.status.success());
            let sass = String::from_utf8(out.stdout).unwrap();
            for op in ["LDL", "STL", "HMMA", "QMMA"] {
                assert!(!sass.contains(op), "{} {op}", usage.symbol);
            }
            assert!(sass.contains("ATOM"));
            assert!(sass.contains("ACQBULK"));
            assert!(sass.contains("FFMA"));
        }
    }
}
