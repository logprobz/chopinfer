use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::contract::{Argument, DType, RmsNormKind};
use kernel_core::rmsnorm::{RmsNormNumerics, RmsNormRequirement};
use kernel_core::rmsnorm_quantize::{RmsNormQuantizeBuffers, RmsNormQuantizeQuery};
use kernels_sm120::{
    backend::Sm120,
    row::rmsnorm_quantize::{self, Entry},
};

fn query(input: DType) -> RmsNormQuantizeQuery {
    RmsNormQuantizeQuery {
        norm: RmsNormRequirement {
            input,
            output: input,
            ..RmsNormRequirement::bf16(
                1280,
                1e-20,
                RmsNormKind::Plain,
                false,
                RmsNormNumerics::TREE_256,
            )
        },
        amax_floor_bits: 1e-4_f32.to_bits(),
    }
}
fn buffers() -> RmsNormQuantizeBuffers<View> {
    let v = |handle| View {
        handle,
        bytes: 32 * 1280 * 4,
    };
    RmsNormQuantizeBuffers {
        input: v(1),
        weight: v(2),
        normed: v(3),
        widened: Some(v(4)),
        codes: v(5),
        scales: v(6),
        live_rows: v(7),
        status: v(8),
        status_row_offset: 2,
    }
}

#[test]
fn fused_norm_preserves_the_contract_and_refuses_aliases_and_short_roles() {
    for dtype in [DType::Bf16, DType::F32] {
        let query = query(dtype);
        let entry = Entry::new(query, 32, "fused_norm").unwrap();
        let op = Sm120::new()
            .with_rmsnorm_quantize(entry)
            .rmsnorm_quantize(&query, Need::rows(32))
            .unwrap();
        assert_eq!(op.info().workspace, 0);
        let mut b = buffers();
        if dtype == DType::F32 {
            b.widened = None;
        }
        let launch = |b: &RmsNormQuantizeBuffers<View>, rows| {
            op.launches(b, View::EMPTY, Counts::rows(rows))
        };
        for rows in [1, 2, 4, 8, 16, 32] {
            let calls = launch(&b, rows).unwrap();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].plan.block, [256, 1, 1]);
            assert!(calls[0].plan.pdl);
            assert_eq!(calls[0].plan.workspace_bytes, 0);
            assert_eq!(calls[0].arguments.0[8], Argument::U64(2));
        }
        assert!(launch(&b, 0).is_err());
        assert!(launch(&b, 33).is_err());
        for index in 0..8 {
            let mut short = b;
            let role = match index {
                0 => &mut short.input,
                1 => &mut short.weight,
                2 => &mut short.normed,
                3 if dtype == DType::Bf16 => short.widened.as_mut().unwrap(),
                3 => continue,
                4 => &mut short.codes,
                5 => &mut short.scales,
                6 => &mut short.live_rows,
                _ => &mut short.status,
            };
            role.bytes = 1;
            assert!(launch(&short, 32).is_err(), "role {index}");
        }
        for handle in 1..=7 {
            let mut aliased = b;
            aliased.status.handle = handle;
            if handle == 4 && dtype == DType::F32 {
                continue;
            }
            assert!(launch(&aliased, 1).is_err(), "alias {handle}");
        }
        let mut overflow = b;
        overflow.status_row_offset = u64::MAX;
        assert!(launch(&overflow, 1).is_err());
    }
}

#[test]
fn fusion_selection_is_order_independent_and_rejects_changed_rounding() {
    let q = query(DType::Bf16);
    let small = Entry::new(q, 8, "small").unwrap();
    let large = Entry::new(q, 32, "large").unwrap();
    for entries in [
        [small.clone(), large.clone()],
        [large.clone(), small.clone()],
    ] {
        let backend = entries
            .into_iter()
            .fold(Sm120::new(), Sm120::with_rmsnorm_quantize);
        assert_eq!(
            backend
                .rmsnorm_quantize(&q, Need::rows(8))
                .unwrap()
                .info()
                .symbols[0]
                .0,
            "small"
        );
        let mut wrong = q;
        wrong.norm.numerics = Some(RmsNormNumerics::HF_ORDERED);
        assert!(backend.rmsnorm_quantize(&wrong, Need::rows(8)).is_err());
        wrong = q;
        wrong.amax_floor_bits = 2e-4_f32.to_bits();
        assert!(backend.rmsnorm_quantize(&wrong, Need::rows(8)).is_err());
    }
    let tied = Sm120::new()
        .with_rmsnorm_quantize(small)
        .with_rmsnorm_quantize(Entry::new(q, 8, "other").unwrap());
    assert!(tied.rmsnorm_quantize(&q, Need::rows(8)).is_err());
}

#[test]
fn fused_norm_compiles_without_spills_or_stack_and_keeps_legacy_sources() {
    let root = std::env::temp_dir().join(format!("norm-quantize-build-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let specs = rmsnorm_quantize::cubin_spec(
        "norm_quantize",
        &[
            Entry::new(query(DType::Bf16), 32, "chop_bf16_norm_quantize_v1").unwrap(),
            Entry::new(query(DType::F32), 32, "chop_f32_norm_quantize_v1").unwrap(),
        ],
    )
    .unwrap();
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let built = kernel_build::build(
        &toolchain,
        &[specs],
        &root.join("out"),
        &root.join("lock"),
        kernel_build::BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    assert_eq!(built[0].entries.len(), 2);
    for entry in &built[0].entries {
        assert_eq!((entry.local_bytes, entry.stack_bytes), (0, 0));
    }
    let source = include_str!("../cuda/row/rmsnorm_quantize.cuh");
    assert!(source.contains("rmsnorm<Width,0,0,false>"));
    assert!(source.contains("rmsnorm_f32<Width>"));
    assert!(source.contains("ceil_epilogue<Width,F32,FloorBits>"));
    assert!(
        include_str!("../cuda/row/ceil_epilogue.cuh")
            .contains("block_quantize_parallel::quantize_group<0,F32,false,FloorBits>")
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn qualified_norm_tiles_preserve_abi_and_only_select_measured_buckets() {
    let q = query(DType::Bf16);
    let entry = Entry::new(q, 32, "norm_reference").unwrap();
    let tiles = kernels_sm120::row::rmsnorm_quantize_tiles::Entry {
        reference: entry.clone(),
        prefix: "norm_candidate".into(),
    };
    let backend = Sm120::new()
        .with_rmsnorm_quantize(entry.clone())
        .with_rmsnorm_quantize_tiles(tiles.clone(), vec![1, 8]);
    let op = backend.rmsnorm_quantize(&q, Need::rows(32)).unwrap();
    let reference = Sm120::new()
        .with_rmsnorm_quantize(entry.clone())
        .rmsnorm_quantize(&q, Need::rows(32))
        .unwrap();
    assert_eq!(op.info().implementation, reference.info().implementation);
    for rows in [1, 2, 4, 8, 16, 32] {
        let old = reference
            .launches(&buffers(), View::EMPTY, Counts::rows(rows))
            .unwrap();
        let new = op
            .launches(&buffers(), View::EMPTY, Counts::rows(rows))
            .unwrap();
        assert_eq!(old[0].arguments, new[0].arguments);
        assert_eq!(new[0].plan.workspace_bytes, 0);
        let selected = [1, 8].contains(&rows);
        assert_eq!(
            new[0].symbol,
            if selected {
                tiles.symbols()[1].clone()
            } else {
                entry.symbol.clone()
            }
        );
        assert_eq!(
            new[0].plan.grid,
            if selected { [5, rows, 1] } else { [rows, 1, 1] }
        );
        let mut bad = buffers();
        bad.normed.handle = bad.input.handle;
        assert!(op.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
        bad = buffers();
        bad.codes.bytes = 1;
        assert!(op.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
    }
    for buckets in [vec![], vec![0], vec![8, 1], vec![1, 1], vec![33]] {
        assert!(
            Sm120::new()
                .with_rmsnorm_quantize(entry.clone())
                .with_rmsnorm_quantize_tiles(tiles.clone(), buckets)
                .rmsnorm_quantize(&q, Need::rows(32))
                .is_err()
        );
    }
}

#[test]
fn cached_norm_selects_only_qualified_buckets_and_retains_alias_refusals() {
    let q = query(DType::Bf16);
    let entry = Entry::new(q, 32, "norm_reference").unwrap();
    let tiles = kernels_sm120::row::rmsnorm_quantize_tiles::Entry {
        reference: entry.clone(),
        prefix: "norm_tile".into(),
    };
    let cached = kernels_sm120::row::rmsnorm_quantize_cached::Entry {
        reference: entry.clone(),
        prefix: "norm_cache".into(),
        early_release: false,
    };
    let baseline = Sm120::new()
        .with_rmsnorm_quantize(entry.clone())
        .with_rmsnorm_quantize_tiles(tiles.clone(), vec![1, 8]);
    let old = baseline.rmsnorm_quantize(&q, Need::rows(32)).unwrap();
    let backend = baseline.with_rmsnorm_quantize_cached(cached.clone(), vec![(1, 32), (8, 32)]);
    let op = backend.rmsnorm_quantize(&q, Need::rows(32)).unwrap();
    assert_eq!(op.info().implementation, old.info().implementation);
    assert_eq!(op.info().bytes, old.info().bytes);
    assert_eq!(op.info().workspace, 0);
    for rows in [1, 2, 4, 8, 16, 32] {
        for widened in [false, true] {
            let mut b = buffers();
            if !widened {
                b.widened = None;
            }
            let reference = old.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap();
            let actual = op.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap();
            assert_eq!(actual.len(), 1);
            assert_eq!(actual[0].arguments, reference[0].arguments);
            let selected = [1, 8].contains(&rows);
            assert_eq!(
                actual[0].symbol,
                if selected {
                    cached.symbols()[2].clone()
                } else {
                    entry.symbol.clone()
                }
            );
            assert_eq!(
                actual[0].plan.grid,
                if selected { [1, rows, 1] } else { [rows, 1, 1] }
            );
            assert_eq!(
                actual[0].plan.block,
                [if selected { 1024 } else { 256 }, 1, 1]
            );
            assert!(actual[0].plan.pdl);
            assert_eq!(actual[0].plan.workspace_bytes, 0);
            let mut bad = b;
            bad.input = bad.normed;
            assert!(op.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
            bad = b;
            bad.codes.bytes = 1;
            assert!(op.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
            bad = b;
            bad.status_row_offset = u64::MAX;
            assert!(op.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
        }
    }
    for buckets in [
        vec![],
        vec![(0, 32)],
        vec![(1, 4)],
        vec![(1, 32), (1, 16)],
        vec![(8, 32), (1, 32)],
        vec![(33, 32)],
    ] {
        assert!(
            Sm120::new()
                .with_rmsnorm_quantize(entry.clone())
                .with_rmsnorm_quantize_cached(cached.clone(), buckets)
                .rmsnorm_quantize(&q, Need::rows(32))
                .is_err()
        );
    }
    assert!(
        backend
            .with_rmsnorm_quantize_cached(cached, vec![(1, 32)])
            .rmsnorm_quantize(&q, Need::rows(32))
            .is_err()
    );
}

#[test]
fn cached_f32_norm_keeps_the_bf16_spec_and_selects_only_qualified_buckets() {
    let cached = |input| kernels_sm120::row::rmsnorm_quantize_cached::Entry {
        reference: Entry::new(query(input), 32, "norm_reference").unwrap(),
        prefix: "norm_cache".into(),
        early_release: false,
    };
    let bf16 = cached(DType::Bf16).cubin_spec("bf16").unwrap();
    assert_eq!(
        bf16.headers,
        ["#include \"row/rmsnorm_quantize_cached.cuh\""]
    );
    let f32 = cached(DType::F32);
    let spec = f32.cubin_spec("f32").unwrap();
    assert_eq!(
        spec.headers,
        ["#include \"row/rmsnorm_quantize_cached_f32.cuh\""]
    );
    assert_eq!(spec.entries.len(), 3);
    for (entry, threads) in spec.entries.iter().zip([256, 512, 1024]) {
        assert_eq!(entry.params, f32.reference.spec().params);
        assert!(entry.body.contains(&format!("blockDim.x!={threads}")));
        assert!(entry.body.contains("run<1280,"));
    }
    for width in [1290, 2080] {
        let mut wide = f32.clone();
        wide.reference.query.norm.width = width;
        assert!(wide.validate().is_err(), "{width}");
    }
    let q = query(DType::F32);
    let entry = f32.reference.clone();
    let old = Sm120::new()
        .with_rmsnorm_quantize(entry.clone())
        .rmsnorm_quantize(&q, Need::rows(32))
        .unwrap();
    let op = Sm120::new()
        .with_rmsnorm_quantize(entry.clone())
        .with_rmsnorm_quantize_cached(f32.clone(), vec![(1, 8), (8, 16)])
        .rmsnorm_quantize(&q, Need::rows(32))
        .unwrap();
    assert_eq!(op.info().implementation, old.info().implementation);
    assert_eq!(op.info().bytes, old.info().bytes);
    let mut b = buffers();
    b.widened = None;
    for rows in [1, 2, 8, 32] {
        let reference = old.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap();
        let actual = op.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap();
        assert_eq!(actual[0].arguments, reference[0].arguments);
        let (symbol, block) = match rows {
            1 => (f32.symbols()[0].clone(), 256),
            8 => (f32.symbols()[1].clone(), 512),
            _ => (entry.symbol.clone(), 256),
        };
        assert_eq!(actual[0].symbol, symbol);
        assert_eq!(actual[0].plan.block, [block, 1, 1]);
        assert!(actual[0].plan.pdl);
        // F32 rows are never widened.
        assert!(
            op.launches(&buffers(), View::EMPTY, Counts::rows(rows))
                .is_err()
        );
    }
}

#[test]
fn cached_f32_norm_compiles_without_spills_or_stack() {
    let root = std::env::temp_dir().join(format!("norm-cached-f32-build-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let mut specs = Vec::new();
    for width in [1280, 2048] {
        let mut q = query(DType::F32);
        q.norm.width = width;
        specs.push(
            kernels_sm120::row::rmsnorm_quantize_cached::Entry {
                reference: Entry::new(q, 32, "norm_reference").unwrap(),
                prefix: format!("norm_cache_{width}"),
                early_release: false,
            }
            .cubin_spec(&format!("norm_cached_f32_{width}"))
            .unwrap(),
        );
    }
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let built = kernel_build::build(
        &toolchain,
        &specs,
        &root.join("out"),
        &root.join("lock"),
        kernel_build::BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    for module in &built {
        assert_eq!(module.entries.len(), 3);
        for entry in &module.entries {
            assert_eq!(
                (entry.local_bytes, entry.stack_bytes),
                (0, 0),
                "{}",
                entry.symbol
            );
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
