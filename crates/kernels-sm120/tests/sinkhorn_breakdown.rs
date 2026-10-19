use kernels_sm120::sinkhorn_hc::{
    Geometry,
    breakdown::{self, Entry, Mode},
    lagged, lagged_split,
};
fn entries() -> Vec<Entry> {
    [
        (Mode::Full, false),
        (Mode::Full, true),
        (Mode::Loads, true),
        (Mode::Resident, true),
        (Mode::Sinkhorn, true),
        (Mode::Ticket, true),
    ]
    .into_iter()
    .map(|(mode, sampled)| Entry {
        reference: lagged_split::entry(
            lagged::LAB_PROFILE,
            Geometry {
                capacity: 32,
                ..lagged::LAB
            },
            false,
        ),
        prefix: "chop_hc_breakdown".into(),
        mode,
        sampled,
    })
    .collect()
}
#[test]
fn ablations_keep_reference_geometry_and_refuse_impossible_cases() {
    for e in entries() {
        for rows in [1, 8] {
            for live in [0, 1, rows] {
                assert_eq!(e.plan(rows, live).unwrap().grid, [20, rows, 1]);
            }
            assert_eq!(
                e.stamps_bytes(rows).unwrap(),
                u64::from(rows) * 20 * 32 * 12 * 8
            );
            assert!(e.plan(rows, rows + 1).is_err());
        }
        assert!(e.plan(0, 0).is_err());
        assert!(e.plan(33, 1).is_err());
        assert_eq!(e.spec().unwrap().params.len(), 16);
    }
    let mut bad = entries()[0].clone();
    bad.mode = Mode::Loads;
    assert!(bad.validate().is_err());
}
#[test]
fn byte_models_distinguish_unique_weights_from_eight_row_requests() {
    let e = &entries()[0];
    let one = breakdown::traffic(e, 1).unwrap();
    let eight = breakdown::traffic(e, 8).unwrap();
    assert_eq!(one.weights_unique, 1_966_080);
    assert_eq!(eight.weights_unique, one.weights_unique);
    assert_eq!(one.unique_input, 2_007_172);
    assert_eq!(eight.unique_input, 2_294_032);
    assert_eq!(eight.projection_weight_requests, 8 * one.weights_unique);
    assert_eq!(one.source_read_requests, 122_880);
    assert_eq!(one.partial_reads, 15_360);
    assert_eq!(one.output_writes, 10_336);
    assert_eq!(eight.ticket_atomics, 160);
}
#[test]
fn diagnostics_keep_dependency_operations_in_wrappers() {
    let source = include_str!("../cuda/sinkhorn_hc/breakdown.cuh");
    for forbidden in [
        "blockIdx",
        "blockDim",
        "gridDim",
        "__shared__",
        "ptx::griddep",
    ] {
        assert!(!source.contains(forbidden), "{forbidden}");
    }
    assert!(source.contains("chain_sums<M>(source,fn,chain,parts)"));
    assert!(source.contains("volatile float* packet"));
}
#[test]
fn ablations_compile_and_retain_the_intended_work_without_spills() {
    let root = std::env::temp_dir().join(format!("hc-breakdown-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let tc = kernel_build::Toolchain::detect().unwrap();
    assert!(!tc.skipped);
    let built = kernel_build::build(
        &tc,
        &[breakdown::cubin_spec("hc_breakdown", &entries()).unwrap()],
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
            assert_eq!(
                (usage.local_bytes, usage.stack_bytes),
                (0, 0),
                "{}",
                usage.symbol
            );
            eprintln!(
                "BREAKDOWN_RESOURCE {} regs={} shared={} local={} stack={}",
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
            for op in ["ATOM", "ACQBULK", "PREEXIT"] {
                assert!(sass.contains(op), "{} {op}", usage.symbol);
            }
            for op in ["HMMA", "QMMA", "LDL", "STL"] {
                assert!(!sass.contains(op), "{} {op}", usage.symbol);
            }
            if usage.symbol.contains("_loads_") || usage.symbol.contains("_ticket_") {
                for op in ["FFMA", "FADD", "FMUL", "MUFU"] {
                    assert!(!sass.contains(op), "{} {op}", usage.symbol);
                }
            } else {
                assert!(sass.contains("FFMA"), "{}", usage.symbol);
            }
            if usage.symbol.contains("_loads_") {
                assert!(sass.contains("LDG"));
                assert!(sass.contains("STS"));
            }
            if usage.symbol.contains("_resident_") {
                assert!(sass.matches("FFMA").count() > 60);
                assert!(sass.contains("LDS"));
            }
        }
    }
}
