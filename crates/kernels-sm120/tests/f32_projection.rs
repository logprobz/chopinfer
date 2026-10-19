use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery, WeightFormat};
use kernel_core::precision::{
    buffers::{ProjectionBuffers, ProjectionScratch, QuantizedWeights},
    *,
};
use kernels_sm120::{
    backend::{F32ProjectionOffers, Sm120},
    f32_projection::{self, Entry},
};

fn entry(weight: WeightFormat) -> Entry {
    Entry::new(
        GemmQuery {
            m_capacity: 17,
            n: 19,
            k: 257,
            weight,
            activation: ActivationFormat::F32,
            output: DType::F32,
            epilogues: Epilogues::default(),
            grouped: None,
        },
        OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 17,
            graph: true,
        },
        "exact_test",
    )
    .unwrap()
}
fn operation(e: &Entry) -> OperationSpec {
    let q = e.query();
    let f32_weight = q.weight == WeightFormat::F32;
    OperationSpec {
        id: "test".into(),
        kind: OperationKind::Dense,
        target: kernel_core::contract::Target::Sm120f,
        weights: vec![WeightSpec {
            codes: TensorSpan {
                tensor: "weight".into(),
                dtype: if f32_weight { DType::F32 } else { DType::Bf16 },
                shape: vec![q.n.into(), q.k.into()],
                offset: 0,
                byte_length: e.weight_bytes(),
            },
            logical_shape: vec![q.n.into(), q.k.into()],
            format: q.weight,
            encoding: if f32_weight {
                CodeEncoding::F32LittleEndian
            } else {
                CodeEncoding::Bf16LittleEndian
            },
            layout: WeightLayout::row_major(),
            local_scale: None,
            global_scale: None,
            transforms: vec![],
        }],
        geometry: None,
        activations: vec![ActivationSpec::f32()],
        arithmetic: e.capability().arithmetic,
        epilogues: q.epilogues,
        uses: vec![e.usage()],
    }
}
fn backend(entries: Vec<Entry>) -> Sm120 {
    let available = entries.iter().flat_map(Entry::runtime_symbols).collect();
    Sm120::new()
        .with_f32_projections(F32ProjectionOffers::new(entries, available, "0".repeat(64)).unwrap())
}

#[test]
fn routed_prefill_preserves_decode_and_bounds_every_measured_bucket() {
    let seed = entry(WeightFormat::Bf16);
    let q = GemmQuery {
        m_capacity: 8192,
        ..seed.query()
    };
    let usage = OperationUse {
        capacity: 8192,
        ..seed.usage()
    };
    let old = Entry::new(q, usage, "routed_test").unwrap();
    let split = old.clone().with_split_decode();
    let routed = split
        .clone()
        .with_prefill_buckets(vec![(512, 32), (2048, 64), (8192, 64)])
        .unwrap();
    let old_spec = f32_projection::cubin_spec("unchanged", std::slice::from_ref(&split)).unwrap();
    let new_spec = f32_projection::cubin_spec("unchanged", std::slice::from_ref(&routed)).unwrap();
    assert_eq!(
        kernel_build::generate(&old_spec).unwrap().translation_unit,
        kernel_build::generate(&new_spec).unwrap().translation_unit
    );
    let prepared = backend(vec![routed.clone()])
        .projection(&operation(&routed), usage, Need::rows(8192))
        .unwrap()
        .handle;
    let mut b = buffers(&routed);
    b.scratch.weight_domain = Some(View {
        handle: 10,
        bytes: routed.weight_domain_bytes(),
    });
    let work = View {
        handle: 9,
        bytes: prepared.info().workspace,
    };
    assert!(
        work.bytes >= routed.prefill_workspace_bytes(8192),
        "prefill must fit the prepared scratch"
    );
    assert_eq!(prepared.initialize(&b).unwrap().len(), 1);
    assert_eq!(
        prepared.initialize(&b).unwrap()[0].symbol,
        split.stage_symbol("_weights")
    );
    for rows in [1, 32, 33, 256, 257, 511, 512, 513, 2048, 4096, 8192] {
        let launches = prepared.launches(&b, work, Counts::rows(rows)).unwrap();
        if let Some(&(_, tile)) = routed.prefill_buckets().iter().find(|&&(r, _)| r == rows) {
            let tiled = old.clone().with_prefill_tiles(tile).unwrap();
            assert_eq!(launches.len(), 2);
            assert_eq!(launches[1].symbol, tiled.symbol());
            assert!(prepared.info().symbols.contains(&(
                tiled.symbol().into(),
                tiled.prefill_plan(rows).unwrap().dynamic_shared_bytes
            )));
            assert!(
                prepared
                    .launches(
                        &b,
                        View {
                            bytes: tiled.prefill_workspace_bytes(rows) - 1,
                            ..work
                        },
                        Counts::rows(rows)
                    )
                    .is_err()
            );
        } else if rows > 256 {
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0].symbol, old.symbol());
        } else {
            assert_eq!(launches.len(), if rows <= 32 { 3 } else { 2 });
        }
    }
    for bad in [
        vec![(256, 32)],
        vec![(8193, 32)],
        vec![(512, 16)],
        vec![(512, 32), (512, 64)],
    ] {
        assert!(split.clone().with_prefill_buckets(bad).is_err());
    }
    let mut missing: std::collections::BTreeSet<_> = routed.runtime_symbols().into_iter().collect();
    missing.remove(old.clone().with_prefill_tiles(64).unwrap().symbol());
    let incomplete = Sm120::new().with_f32_projections(
        F32ProjectionOffers::new(vec![routed.clone()], missing, "0".repeat(64)).unwrap(),
    );
    assert!(
        incomplete
            .projection(&operation(&routed), usage, Need::rows(8192))
            .is_err()
    );
}

#[test]
fn prefill_tiles_bound_expansion_and_configure_both_shared_tiles() {
    for weight in [WeightFormat::Bf16, WeightFormat::F32] {
        for features in [32, 64] {
            let original = entry(weight);
            let q = GemmQuery {
                m_capacity: 8192,
                ..original.query()
            };
            let usage = OperationUse {
                capacity: 8192,
                ..original.usage()
            };
            let e = Entry::new(q, usage, "prefill_test")
                .unwrap()
                .with_prefill_tiles(features)
                .unwrap();
            let prepared = backend(vec![e.clone()])
                .projection(&operation(&e), usage, Need::rows(8192))
                .unwrap()
                .handle;
            let mut b = buffers(&e);
            b.scratch.weight_domain = Some(View {
                handle: 10,
                bytes: e.weight_domain_bytes(),
            });
            let workspace = View {
                handle: 9,
                bytes: prepared.info().workspace,
            };
            for rows in [
                512, 1, 31, 32, 33, 63, 64, 65, 511, 513, 2047, 2048, 2049, 8191, 8192,
            ] {
                let launches = prepared
                    .launches(&b, workspace, Counts::rows(rows))
                    .unwrap();
                assert_eq!(launches.len(), 2, "rows={rows}");
                assert_eq!(launches[1].plan, e.prefill_plan(rows).unwrap());
                assert_eq!(launches[1].plan.block[0], features * 8);
                assert!(
                    prepared
                        .info()
                        .symbols
                        .contains(&(e.symbol().into(), launches[1].plan.dynamic_shared_bytes))
                );
                let short = View {
                    bytes: e.prefill_workspace_bytes(rows) - 1,
                    ..workspace
                };
                assert!(prepared.launches(&b, short, Counts::rows(rows)).is_err());
            }
            assert_eq!(workspace.bytes, 8192 * 4 + 8192 * u64::from(q.k) * 6);
            assert_eq!(prepared.initialize(&b).unwrap().len(), 1);
            assert!(e.prefill_plan(8193).is_err());
            assert!(original.clone().with_prefill_tiles(16).is_err());
            assert!(original.with_split_decode().with_prefill_tiles(32).is_err());
        }
    }
}

#[test]
fn split_decode_selects_the_measured_path_for_every_row_through_256() {
    for weight in [WeightFormat::Bf16, WeightFormat::F32] {
        let original = entry(weight);
        let q = GemmQuery {
            m_capacity: 8192,
            ..original.query()
        };
        let usage = OperationUse {
            capacity: 8192,
            ..original.usage()
        };
        let e = Entry::new(q, usage, "split_test")
            .unwrap()
            .with_split_decode();
        let prepared = backend(vec![e.clone()])
            .projection(&operation(&e), usage, Need::rows(8192))
            .unwrap()
            .handle;
        let legacy = Entry::new(q, usage, "split_test").unwrap().with_decode();
        let legacy_op = backend(vec![legacy.clone()])
            .projection(&operation(&legacy), usage, Need::rows(8192))
            .unwrap()
            .handle;
        let mut b = buffers(&e);
        b.scratch.weight_domain = Some(View {
            handle: 10,
            bytes: e.weight_domain_bytes(),
        });
        let workspace = View {
            handle: 9,
            bytes: prepared.info().workspace,
        };
        assert_eq!(prepared.initialize(&b).unwrap().len(), 1);
        for rows in 1..=256 {
            let launches = prepared
                .launches(&b, workspace, Counts::rows(rows))
                .unwrap();
            assert_eq!(
                launches.len(),
                if rows <= 32 { 3 } else { 2 },
                "rows={rows}"
            );
            if rows <= 32 {
                assert_eq!(
                    launches,
                    legacy_op
                        .launches(&b, workspace, Counts::rows(rows))
                        .unwrap()
                );
            }
            assert_eq!(launches[1].plan.grid, [2, rows.div_ceil(8), 1]);
            let short = View {
                bytes: e.decode_workspace_bytes(rows) - 1,
                ..workspace
            };
            assert!(prepared.launches(&b, short, Counts::rows(rows)).is_err());
        }
        for rows in [257, 2048, 8192] {
            let launches = prepared
                .launches(&b, View::EMPTY, Counts::rows(rows))
                .unwrap();
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0].symbol, e.baseline_symbol());
        }
    }
}

#[test]
fn cached_weights_initialize_once_and_decode_preserves_every_chain_plane() {
    for weight in [WeightFormat::Bf16, WeightFormat::F32] {
        let mut q = entry(weight).query();
        q.m_capacity = 256;
        let usage = OperationUse {
            capacity: 256,
            ..entry(weight).usage()
        };
        let original = Entry::new(q, usage, "cached_test").unwrap();
        let e = original.clone().with_decode();
        let baseline = backend(vec![original.clone()])
            .projection(&operation(&original), usage, Need::rows(256))
            .unwrap()
            .handle;
        let prepared = backend(vec![e.clone()])
            .projection(&operation(&e), usage, Need::rows(256))
            .unwrap()
            .handle;
        let mut b = buffers(&e);
        let workspace = View {
            handle: 9,
            bytes: prepared.info().workspace,
        };
        assert!(prepared.initialize(&b).is_err());
        assert!(prepared.launches(&b, workspace, Counts::rows(1)).is_err());
        b.scratch.weight_domain = Some(View {
            handle: 10,
            bytes: e.weight_domain_bytes(),
        });
        let init = prepared.initialize(&b).unwrap();
        assert_eq!(init.len(), 1);
        assert_eq!(init[0].symbol, e.stage_symbol("_weights"));
        assert_eq!(init[0].arguments.0.len(), 2);
        for rows in [1, 2, 4, 8, 16, 32] {
            let launches = prepared
                .launches(&b, workspace, Counts::rows(rows))
                .unwrap();
            assert_eq!(launches.len(), 3);
            assert_eq!(launches[1].plan.grid, [2, rows.div_ceil(8), 1]);
            assert_eq!(launches[1].plan.block, [128, 1, 1]);
            assert_eq!(
                e.decode_workspace_bytes(rows),
                128 + u64::from(rows * q.n * 3 * 4)
            );
            assert!(
                launches
                    .iter()
                    .all(|l| l.plan.pdl && !l.symbol.contains("_weights_"))
            );
            let short = View {
                bytes: e.decode_workspace_bytes(rows) - 1,
                ..workspace
            };
            assert!(prepared.launches(&b, short, Counts::rows(rows)).is_err());
        }
        for rows in 33..=256 {
            let launches = prepared
                .launches(&b, workspace, Counts::rows(rows))
                .unwrap();
            assert_eq!(launches.len(), 1);
            assert_eq!(
                launches,
                baseline
                    .launches(&b, View::EMPTY, Counts::rows(rows))
                    .unwrap()
            );
        }
        b.scratch.weight_domain.as_mut().unwrap().bytes -= 1;
        assert!(prepared.initialize(&b).is_err());
        assert!(prepared.launches(&b, workspace, Counts::rows(1)).is_err());
        assert!(e.decode_plan(33).is_err());
        let spec = f32_projection::cubin_spec("cached_test", &[e]).unwrap();
        assert_eq!(spec.entries.len(), 5);
        assert!(
            spec.entries
                .iter()
                .all(|entry| entry.symbol.ends_with("_v1"))
        );
    }
}

#[test]
fn cached_decode_refuses_unlaunchable_splits_at_prepare() {
    let original = entry(WeightFormat::Bf16);
    let q = GemmQuery {
        k: 65_535 * 512 + 1,
        ..original.query()
    };
    let e = Entry::new(q, original.usage(), "too_many_splits")
        .unwrap()
        .with_decode();
    assert!(
        backend(vec![e.clone()])
            .projection(&operation(&e), e.usage(), Need::rows(1))
            .is_err()
    );
}

#[test]
fn cached_decode_requires_the_locked_baseline_in_its_inventory() {
    let e = entry(WeightFormat::Bf16).with_decode();
    let backend = Sm120::new().with_f32_projections(
        F32ProjectionOffers::new(
            vec![e.clone()],
            e.symbols().into_iter().collect(),
            "0".repeat(64),
        )
        .unwrap(),
    );
    assert!(
        backend
            .projection(&operation(&e), e.usage(), Need::rows(1))
            .is_err()
    );
}
fn buffers(e: &Entry) -> ProjectionBuffers<View> {
    let v = |handle, bytes| View { handle, bytes };
    let q = e.query();
    ProjectionBuffers {
        activation: v(1, u64::from(q.m_capacity) * u64::from(q.k) * 4),
        live_rows: v(2, 4),
        output: v(3, u64::from(q.m_capacity) * u64::from(q.n) * 4),
        residual: None,
        weights: QuantizedWeights {
            codes: v(4, e.weight_bytes()),
            local_scales: None,
            global_scales: None,
        },
        activation_global: None,
        routing: None,
        scratch: ProjectionScratch {
            weight_domain: None,
            quantizer_status: None,
            quantizer_status_row_offset: 0,
            partial: View::EMPTY,
            semaphores: View::EMPTY,
            unused: View::EMPTY,
            temporary: None,
            gathered: None,
            codes: None,
            scales: None,
            activation_global: None,
            row_amax: None,
            output_codes: None,
            output_scales: None,
        },
        row_offset: 0,
        shard_rows: 0,
    }
}
#[test]
fn f32_binding_covers_original_operands_without_scales_or_scratch() {
    for weight in [WeightFormat::Bf16, WeightFormat::F32] {
        let e = entry(weight);
        let op = operation(&e);
        let usage = e.usage();
        assert!(Sm120::new().projection(&op, usage, Need::rows(17)).is_err());
        let backend = backend(vec![e.clone()]);
        let prepared = backend.projection(&op, usage, Need::rows(17)).unwrap();
        assert_eq!(prepared.identity.stages, vec![e.symbol()]);
        assert_eq!(prepared.handle.info().workspace, 0);
        assert_eq!(
            prepared.handle.info().weight_layouts,
            vec![("weights".into(), WeightLayout::row_major())]
        );
        let counts = Counts {
            rows: 17,
            ..Counts::default()
        };
        let b = buffers(&e);
        let launches = prepared.handle.launches(&b, View::EMPTY, counts).unwrap();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].plan.grid, [2, 3, 1]);
        assert_eq!(launches[0].plan.block, [32, 1, 1]);
        assert!(launches[0].plan.pdl);
        assert_eq!(launches[0].arguments.0.len(), 5);
        for role in ["input", "output", "weight", "live"] {
            let mut bad = b;
            match role {
                "input" => bad.activation.bytes -= 1,
                "output" => bad.output.bytes -= 1,
                "weight" => bad.weights.codes.bytes -= 1,
                _ => bad.live_rows.bytes -= 1,
            }
            assert!(
                prepared.handle.launches(&bad, View::EMPTY, counts).is_err(),
                "{role}"
            );
        }
        let mut bad = b;
        bad.weights.local_scales = Some(View::EMPTY);
        assert!(prepared.handle.launches(&bad, View::EMPTY, counts).is_err());
        assert!(
            prepared
                .handle
                .launches(&b, View::EMPTY, Counts { rows: 18, ..counts })
                .is_err()
        );
        assert!(backend.projection(&op, usage, Need::rows(18)).is_err());
        let mut bad = op.clone();
        bad.activations[0] = ActivationSpec::bf16();
        assert!(backend.projection(&bad, usage, Need::rows(17)).is_err());
        let absent = Sm120::new().with_f32_projections(
            F32ProjectionOffers::new(vec![e.clone()], Default::default(), "0".repeat(64)).unwrap(),
        );
        assert!(absent.projection(&op, usage, Need::rows(17)).is_err());
        let mut other = e.query();
        other.activation = ActivationFormat::Bf16;
        assert!(Entry::new(other, usage, "invalid").is_err());
        let spec = f32_projection::cubin_spec("exact_test", &[e]).unwrap();
        assert_eq!(spec.headers, vec!["#include \"gemm/exact_f32.cuh\""]);
    }
}
#[test]
fn ambiguous_entries_refuse_and_duplicate_entries_collapse() {
    let e = entry(WeightFormat::Bf16);
    let op = operation(&e);
    let duplicate = backend(vec![e.clone(), e.clone()]);
    assert!(duplicate.projection(&op, e.usage(), Need::rows(17)).is_ok());
    let other = Entry::new(e.query(), e.usage(), "other_entry").unwrap();
    assert!(
        backend(vec![e.clone(), other])
            .projection(&op, e.usage(), Need::rows(17))
            .is_err()
    );
}

#[test]
fn conservative_tensor_guard_is_inside_the_exact_product_domain() {
    use kernel_core::precision::f32_gemm::{self, DotPath};
    // The CUDA interval endpoints, and low mantissa bits which require
    // all three components. Check every exponent pair in that interval.
    for a_exp in -30..=30 {
        for b_exp in -30..=30 {
            for mantissa in [0, 1, 0x8001, 0x7fffff] {
                let a = f32::from_bits(((a_exp + 127) as u32) << 23 | mantissa);
                let b = f32::from_bits(((b_exp + 127) as u32) << 23 | 0x7f0000);
                for sign in [1., -1.] {
                    assert_eq!(
                        f32_gemm::dot_path(&[a * sign], &[b], DType::Bf16).unwrap(),
                        DotPath::Bf16Expansion3
                    );
                }
            }
        }
    }
}

#[test]
fn dimensions_cannot_overflow_cuda_fold_indices() {
    let e = entry(WeightFormat::Bf16);
    let mut query = e.query();
    query.k = i32::MAX as u32;
    assert!(Entry::new(query, e.usage(), "overflow").is_err());
    query = e.query();
    query.n = u32::MAX;
    assert!(Entry::new(query, e.usage(), "overflow").is_err());
}
