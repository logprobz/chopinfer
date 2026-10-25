use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::contract::{Argument, Validate};
use kernel_core::sinkhorn_expand_mix::{
    SinkhornExpandMixQuery as Query, SinkhornRoundExpandMixBuffers as Buffers,
};
use kernels_sm120::{
    backend::Sm120,
    sinkhorn_hc::{
        lagged,
        round_expand_mix::{self as expand_mix, Entry},
    },
};

fn entry() -> Entry {
    Entry {
        query: Query {
            mix: lagged::LAB.mix,
        },
        capacity: 32,
        symbol: "chop_hc_round_expand_mix".into(),
        probe: false,
    }
}
fn buffers() -> Buffers<View> {
    let v = |handle| View {
        handle,
        bytes: 1 << 32,
    };
    Buffers {
        x: v(1),
        sum: v(2),
        rounded: v(16),
        post_in: v(3),
        comb_in: v(4),
        expanded: v(5),
        fn_: v(6),
        base: v(7),
        scale: v(8),
        pre_in: v(9),
        collapsed: v(10),
        pre: v(11),
        post: v(12),
        comb: v(13),
        live_rows: v(14),
    }
}
#[test]
fn fused_round_expand_mix_enforces_two_stage_lifetimes_and_full_row_extents() {
    let entry = entry();
    struct Empty;
    impl Backend for Empty {
        fn name(&self) -> &str {
            "empty"
        }
    }
    assert!(matches!(
        Empty.sinkhorn_round_expand_mix(&entry.query, Need::rows(1)),
        Err(kernel_core::backend::Refused::Unsupported)
    ));
    let backend = Sm120::new().with_sinkhorn_round_expand_mix(entry.clone());
    let op = backend
        .sinkhorn_round_expand_mix(&entry.query, Need::rows(32))
        .unwrap();
    assert_eq!(op.info().workspace, 32 * (24 * 160 + 256) * 4);
    for rows in [1, 2, 4, 8, 16, 32] {
        let mut b = buffers();
        b.post = b.post_in;
        b.comb = b.comb_in;
        let ws = View {
            handle: 15,
            bytes: entry.workspace(rows).unwrap(),
        };
        let launches = op.launches(&b, ws, Counts::rows(rows)).unwrap();
        assert_eq!(launches.len(), 2);
        assert_eq!(launches[0].plan.grid, [20, rows, 1]);
        assert_eq!(launches[1].plan.grid, [10, rows, 1]);
        assert!(launches.iter().all(|l| l.plan.pdl));
        assert_eq!(launches[0].arguments.0[7], Argument::DeviceBuffer(15));
        assert_eq!(launches[1].arguments.0[8], Argument::DeviceBuffer(15));
        for variant in 0..10 {
            let mut bad = b;
            let mut scratch = ws;
            match variant {
                0 => bad.expanded = bad.x,
                1 => bad.pre = bad.pre_in,
                2 => bad.collapsed = bad.expanded,
                3 => bad.fn_.bytes = 1,
                4 => bad.sum.bytes = 1,
                5 => bad.comb.bytes = 1,
                6 => scratch.handle = bad.x.handle,
                7 => scratch.bytes -= 1,
                8 => bad.rounded = bad.sum,
                _ => bad.rounded.bytes = 1,
            }
            assert!(
                op.launches(&bad, scratch, Counts::rows(rows)).is_err(),
                "variant {variant}"
            );
        }
    }
    assert!(
        backend
            .sinkhorn_round_expand_mix(&entry.query, Need::rows(33))
            .is_err()
    );
    let mut bad = entry.query;
    bad.mix.placement =
        kernel_core::sinkhorn_hyper_connection::HyperConnectionPlacement::PerSublayer;
    assert!(bad.validate().is_err());
    assert!(entry.plans(0).is_err());
}
#[test]
fn rms_lane_ownership_retains_each_original_fma_chain_once() {
    for groups in [1, 2, 4, 8, 20] {
        let mut owners = vec![0; 256];
        for group in 0..groups {
            for lane in 0..32 {
                for slot in (group * 32 + lane..256).step_by(groups * 32) {
                    owners[slot] += 1;
                }
            }
        }
        assert!(owners.iter().all(|&n| n == 1));
    }
    // Splitting a lane into independently accumulated chunks changes bits.
    let x = [
        1.0f32,
        f32::from_bits(0x3980_0000),
        f32::from_bits(0x3980_0000),
    ];
    let original = x.iter().fold(0.0f32, |sum, &x| x.mul_add(x, sum));
    let regrouped = x[0] * x[0] + x[1..].iter().fold(0.0f32, |sum, &x| x.mul_add(x, sum));
    assert_ne!(original.to_bits(), regrouped.to_bits());
}
#[test]
fn round_expand_mix_compiles_with_no_spills_stack_or_atomic_protocol() {
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let root = std::env::temp_dir().join(format!("round-expand-mix-build-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let spec = expand_mix::cubin_spec("expand_mix", &[entry()]).unwrap();
    let built = kernel_build::build(
        &toolchain,
        &[spec],
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
            assert_eq!((usage.local_bytes, usage.stack_bytes), (0, 0));
            let output = std::process::Command::new("/usr/local/cuda/bin/cuobjdump")
                .args(["-sass", "-fun", &usage.symbol])
                .arg(&image.path)
                .output()
                .unwrap();
            assert!(output.status.success());
            let sass = String::from_utf8(output.stdout).unwrap();
            for opcode in ["ACQBULK", "PREEXIT", "FFMA"] {
                assert!(sass.contains(opcode), "{opcode}");
            }
            for opcode in ["ATOM", "RED.", "HMMA", "QMMA"] {
                assert!(!sass.contains(opcode), "{opcode}");
            }
        }
    }
}

#[test]
fn tasks_use_virtual_ctas_and_only_entry_wrappers_own_grid_dependencies() {
    let body = include_str!("../cuda/sinkhorn_hc/round_expand_mix.cuh");
    for forbidden in [
        "blockIdx",
        "gridDim",
        "blockDim",
        "__shared__",
        "ptx::griddep",
    ] {
        assert!(!body.contains(forbidden), "{forbidden}");
    }
    assert!(body.contains("uint32_t tile, uint32_t row, unsigned char* arena"));
    let spec = expand_mix::cubin_spec("round_expand_mix", &[entry()]).unwrap();
    assert_eq!(spec.entries.len(), 2);
}

#[test]
fn measured_chain_buckets_keep_the_finisher_arguments_and_workspace_exact() {
    let reference = entry();
    let chain = kernels_sm120::sinkhorn_hc::round_chain_partials::Entry {
        reference: reference.clone(),
        prefix: "chain".into(),
    };
    let old = Sm120::new()
        .with_sinkhorn_round_expand_mix(reference.clone())
        .sinkhorn_round_expand_mix(&reference.query, Need::rows(32))
        .unwrap();
    let backend = Sm120::new()
        .with_sinkhorn_round_expand_mix(reference.clone())
        .with_sinkhorn_round_chain(chain.clone(), vec![(1, 8), (8, 4)]);
    let op = backend
        .sinkhorn_round_expand_mix(&reference.query, Need::rows(32))
        .unwrap();
    assert_eq!(op.info().implementation, old.info().implementation);
    assert_eq!(op.info().bytes, old.info().bytes);
    assert_eq!(op.info().workspace, old.info().workspace);
    for rows in [1, 2, 4, 8, 16, 32] {
        let mut b = buffers();
        b.post = b.post_in;
        b.comb = b.comb_in;
        let ws = View {
            handle: 15,
            bytes: reference.workspace(rows).unwrap(),
        };
        let expected = old.launches(&b, ws, Counts::rows(rows)).unwrap();
        let actual = op.launches(&b, ws, Counts::rows(rows)).unwrap();
        assert_eq!(actual.len(), 2);
        let route = op
            .route(Counts::rows(rows), &actual[0])
            .expect("selected chain producer");
        assert_eq!(
            route.class,
            if [1, 8].contains(&rows) {
                kernel_core::route::RouteClass::Fast
            } else {
                kernel_core::route::RouteClass::Baseline
            }
        );
        let finish = op
            .route(Counts::rows(rows), &actual[1])
            .expect("standalone finisher");
        assert_eq!(finish.class, kernel_core::route::RouteClass::Fast);
        assert_eq!(
            old.route(Counts::rows(rows), &expected[0]).unwrap().class,
            kernel_core::route::RouteClass::Fast
        );
        for i in 0..2 {
            assert_eq!(actual[i].arguments, expected[i].arguments);
            assert_eq!(
                actual[i].plan.workspace_bytes,
                expected[i].plan.workspace_bytes
            );
            assert!(actual[i].plan.pdl);
        }
        assert_eq!(actual[1].symbol, expected[1].symbol);
        assert_eq!(actual[1].plan, expected[1].plan);
        let (name, grid, block) = match rows {
            1 => (chain.symbols()[1].clone(), 160, 256),
            8 => (chain.symbols()[0].clone(), 160, 128),
            _ => (reference.symbols()[0].clone(), 20, 256),
        };
        assert_eq!(actual[0].symbol, name);
        assert_eq!(actual[0].plan.grid, [grid, rows, 1]);
        assert_eq!(actual[0].plan.block, [block, 1, 1]);
        for control in 0..4 {
            let mut bad = b;
            let mut scratch = ws;
            match control {
                0 => scratch.bytes -= 1,
                1 => scratch.handle = b.x.handle,
                2 => bad.rounded = b.sum,
                _ => bad.fn_.bytes = 1,
            };
            assert!(op.launches(&bad, scratch, Counts::rows(rows)).is_err());
        }
    }
    for buckets in [
        vec![],
        vec![(0, 8)],
        vec![(1, 16)],
        vec![(1, 8), (1, 4)],
        vec![(8, 4), (1, 8)],
        vec![(33, 4)],
    ] {
        assert!(
            Sm120::new()
                .with_sinkhorn_round_expand_mix(reference.clone())
                .with_sinkhorn_round_chain(chain.clone(), buckets)
                .sinkhorn_round_expand_mix(&reference.query, Need::rows(32))
                .is_err()
        );
    }
    assert!(
        backend
            .with_sinkhorn_round_chain(chain, vec![(1, 8)])
            .sinkhorn_round_expand_mix(&reference.query, Need::rows(32))
            .is_err()
    );
}
