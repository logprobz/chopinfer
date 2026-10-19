use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::contract::{Argument, Validate};
use kernel_core::sinkhorn_expand_mix::{
    SinkhornExpandMixBuffers as Buffers, SinkhornExpandMixQuery as Query,
};
use kernels_sm120::{
    backend::Sm120,
    sinkhorn_hc::{
        expand_mix::{self, Entry},
        lagged,
    },
};

fn entry() -> Entry {
    Entry {
        query: Query {
            mix: lagged::LAB.mix,
        },
        capacity: 32,
        symbol: "chop_hc_expand_mix".into(),
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
        y: v(2),
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
fn fused_expand_mix_enforces_two_stage_lifetimes_and_full_row_extents() {
    let entry = entry();
    struct Empty;
    impl Backend for Empty {
        fn name(&self) -> &str {
            "empty"
        }
    }
    assert!(matches!(
        Empty.sinkhorn_expand_mix(&entry.query, Need::rows(1)),
        Err(kernel_core::backend::Refused::Unsupported)
    ));
    let backend = Sm120::new().with_sinkhorn_expand_mix(entry.clone());
    let op = backend
        .sinkhorn_expand_mix(&entry.query, Need::rows(32))
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
        assert_eq!(launches[0].arguments.0[6], Argument::DeviceBuffer(15));
        assert_eq!(launches[1].arguments.0[8], Argument::DeviceBuffer(15));
        for variant in 0..8 {
            let mut bad = b;
            let mut scratch = ws;
            match variant {
                0 => bad.expanded = bad.x,
                1 => bad.pre = bad.pre_in,
                2 => bad.collapsed = bad.expanded,
                3 => bad.fn_.bytes = 1,
                4 => bad.y.bytes = 1,
                5 => bad.comb.bytes = 1,
                6 => scratch.handle = bad.x.handle,
                _ => scratch.bytes -= 1,
            }
            assert!(
                op.launches(&bad, scratch, Counts::rows(rows)).is_err(),
                "variant {variant}"
            );
        }
    }
    assert!(
        backend
            .sinkhorn_expand_mix(&entry.query, Need::rows(33))
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
fn expand_mix_compiles_with_no_spills_stack_or_atomic_protocol() {
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let root = std::env::temp_dir().join(format!("expand-mix-build-{}", std::process::id()));
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
