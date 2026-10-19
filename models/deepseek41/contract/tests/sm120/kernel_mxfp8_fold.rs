#[path = "../../../bundle/src/sm120/mxfp8_fold.rs"]
#[allow(dead_code)]
mod registration;

use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::contract::{DType, Validate};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::{admission, geometry, projections};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn measured_prefetch_ties_keep_ticket_at_sixty_four_rows() {
    let backend = registration::register(Sm120::new(), &geometry::FLASH, 2).unwrap();
    for (n, k, paired) in [(16384, 1280, false), (1152, 5120, true)] {
        let entry = registration::ticket_entries(&geometry::FLASH, 2)
            .unwrap()
            .into_iter()
            .find(|e| (e.query.n, e.query.k, e.paired) == (n, k, paired))
            .unwrap();
        let info = if paired {
            backend
                .mxfp8_fold_gemm_pair(&entry.query, Need::rows(64))
                .unwrap()
                .info()
                .clone()
        } else {
            backend
                .mxfp8_fold_gemm(&entry.query, Need::rows(64))
                .unwrap()
                .info()
                .clone()
        };
        assert_eq!(
            info.symbols
                .iter()
                .map(|(s, _)| s.clone())
                .collect::<Vec<_>>(),
            entry.symbols(),
            "N={n} K={k} paired={paired}"
        );
    }
}

#[test]
fn exact_row_bounds_keep_compiled_fold_entries_unchanged() {
    let cubins = registration::ticket_cubins(geometry::GEOMETRIES)
        .unwrap()
        .into_iter()
        .chain(registration::prefetch_cubins(geometry::GEOMETRIES).unwrap())
        .collect::<Vec<_>>();
    for g in geometry::GEOMETRIES {
        for entry in registration::ticket_catalog(g)
            .unwrap()
            .into_iter()
            .chain(registration::prefetch_catalog(g).unwrap())
        {
            let (family, name) = if entry.prefetch_weights {
                (
                    registration::PREFETCH_FAMILY,
                    registration::prefetch_cubin_name(g),
                )
            } else {
                (
                    registration::TICKET_FAMILY,
                    registration::ticket_cubin_name(g),
                )
            };
            let (owner, cubin) = cubins.iter().find(|(_, s)| s.name == name).unwrap();
            assert_eq!(*owner, family);
            assert_eq!(
                cubin.entries.len(),
                registration::ticket_catalog(g).unwrap().len() * 2
            );
            assert!(
                entry
                    .symbols()
                    .iter()
                    .all(|symbol| cubin.entries.iter().any(|s| s.symbol == *symbol))
            );
            let specs = format!("{:?}", entry.specs());
            for rows in 1..=64 {
                let mut bounded = entry.clone();
                bounded.query = registration::routed_query(entry.query, rows);
                bounded.validate().unwrap();
                assert_eq!(format!("{:?}", bounded.specs()), specs);
            }
        }
        for entry in registration::decode_catalog(g).unwrap() {
            let specs = format!("{:?}", entry.specs());
            for rows in 1..=32 {
                let mut bounded = entry.clone();
                bounded.query = registration::routed_query(entry.query, rows);
                bounded.validate().unwrap();
                assert_eq!(format!("{:?}", bounded.specs()), specs);
            }
        }
    }
}

#[test]
fn measured_entries_bind_all_topologies_and_exact_row_buckets() {
    use registration::DecodePath;
    let mut records = BTreeMap::new();
    for g in geometry::GEOMETRIES {
        let prefetch = registration::prefetch_catalog(g).unwrap();
        let ticket = registration::ticket_catalog(g).unwrap();
        let split = registration::decode_catalog(g).unwrap();
        for &ranks in admission::SUPPORTED_RANKS {
            let backend = registration::register(Sm120::new(), g, ranks).unwrap();
            for entry in registration::prefetch_entries(g, ranks).unwrap() {
                for rows in 1..=64 {
                    let q = registration::routed_query(entry.query, rows);
                    let path = registration::decode_path(q, rows, entry.paired);
                    let (symbols, workspace, plans) = match path {
                        DecodePath::Split => {
                            let old = split
                                .iter()
                                .find(|old| {
                                    old.paired == entry.paired
                                        && registration::routed_query(old.query, rows) == q
                                })
                                .unwrap();
                            (
                                old.symbols().to_vec(),
                                old.workspace(rows).unwrap(),
                                old.plans(rows).unwrap().to_vec(),
                            )
                        }
                        _ => {
                            let catalog = if path == DecodePath::Prefetch {
                                &prefetch
                            } else {
                                &ticket
                            };
                            let old = catalog
                                .iter()
                                .find(|old| {
                                    old.paired == entry.paired
                                        && registration::routed_query(old.query, rows) == q
                                })
                                .unwrap();
                            (
                                old.symbols().to_vec(),
                                old.workspace(rows).unwrap(),
                                old.plans(rows).unwrap().to_vec(),
                            )
                        }
                    };
                    let b = entry.legacy().unwrap().bytes(rows);
                    let view = |handle, bytes| View { handle, bytes };
                    let buffers = Mxfp8FoldGemmBuffers {
                        activations: view(1, b.activations),
                        activation_scales: view(2, b.activation_scales),
                        weights: view(3, b.weights),
                        weight_scales: view(4, b.weight_scales),
                        output: view(5, b.output),
                        live_rows: view(6, 4),
                        status: view(7, b.status),
                        status_row_offset: 0,
                    };
                    let second = Mxfp8FoldGemmBuffers {
                        weights: view(8, b.weights),
                        weight_scales: view(9, b.weight_scales),
                        output: view(10, b.output),
                        ..buffers
                    };
                    let (info, launches) = if entry.paired {
                        let op = backend.mxfp8_fold_gemm_pair(&q, Need::rows(rows)).unwrap();
                        let launches = op
                            .launches(&[buffers, second], view(11, workspace), Counts::rows(rows))
                            .unwrap();
                        (op.info().clone(), launches)
                    } else {
                        let op = backend.mxfp8_fold_gemm(&q, Need::rows(rows)).unwrap();
                        let launches = op
                            .launches(&buffers, view(11, workspace), Counts::rows(rows))
                            .unwrap();
                        (op.info().clone(), launches)
                    };
                    assert_eq!(
                        info.symbols
                            .iter()
                            .map(|(s, _)| s.clone())
                            .collect::<Vec<_>>(),
                        symbols
                    );
                    assert_eq!(info.workspace, workspace);
                    assert_eq!(launches.iter().map(|l| l.plan).collect::<Vec<_>>(), plans);
                    let key = (g.id, q.n, q.k, q.groups, entry.paired, rows);
                    let record = serde_json::json!({
                        "n": q.n, "k": q.k, "groups": q.groups, "paired": entry.paired, "rows": rows,
                        "routing": match path { DecodePath::Split => "split-fold", DecodePath::Ticket => "ticket", DecodePath::Prefetch => "prefetch" },
                        "symbols": symbols, "workspace_bytes": workspace,
                    });
                    if let Some(old) = records.insert(key, record.clone()) {
                        assert_eq!(old, record);
                    }
                }
            }
        }
    }
    let flash = records
        .iter()
        .filter(|(k, _)| k.0 == "flash")
        .map(|(_, r)| r)
        .collect::<Vec<_>>();
    assert_eq!(flash.len(), 23 * 64);
    assert_eq!(
        flash.iter().filter(|r| r["routing"] == "prefetch").count(),
        623
    );
    assert_eq!(
        flash.iter().filter(|r| r["routing"] == "ticket").count(),
        791
    );
    assert_eq!(
        flash
            .iter()
            .filter(|r| r["routing"] == "split-fold")
            .count(),
        58
    );
    if let Some(path) = std::env::var_os("CHOPINFER_FOLD_ROUTING_OUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&flash).unwrap()).unwrap();
    }
}

#[test]
fn decode_uses_bind_every_bucket_and_pair_only_the_shared_operands() {
    use kernel_core::precision::ExecutionPhase;
    let specs = registration::decode_cubins(geometry::GEOMETRIES).unwrap();
    assert!(
        specs
            .iter()
            .all(|(family, _)| *family == registration::DECODE_FAMILY)
    );
    for g in geometry::GEOMETRIES {
        let spec = &specs
            .iter()
            .find(|(_, s)| s.name == registration::decode_cubin_name(g))
            .unwrap()
            .1;
        let catalog = registration::decode_catalog(g).unwrap();
        assert_eq!(spec.entries.len(), catalog.len() * 3);
        for &ranks in admission::SUPPORTED_RANKS {
            let backend = registration::decode_entries(g, ranks)
                .unwrap()
                .into_iter()
                .fold(Sm120::new(), Sm120::with_mxfp8_fold_decode);
            for r in registration::decode_registrations(g, ranks, 0).unwrap() {
                assert_eq!(r.usage.phase, ExecutionPhase::Decode);
                assert!(registration::DECODE_BUCKETS.contains(&r.usage.capacity));
                let q = r.entry.query;
                let op = if r.entry.paired {
                    assert_eq!(r.projection, "layer.shared.gate-up");
                    assert_eq!(r.operand, 0);
                    backend
                        .mxfp8_fold_gemm_pair(&q, Need::rows(r.usage.capacity))
                        .unwrap()
                        .info()
                        .clone()
                } else {
                    backend
                        .mxfp8_fold_gemm(&q, Need::rows(r.usage.capacity))
                        .unwrap()
                        .info()
                        .clone()
                };
                assert_eq!(op.workspace, r.entry.workspace(r.usage.capacity).unwrap());
                assert_eq!(op.symbols.len(), 3);
                assert!(
                    op.symbols
                        .iter()
                        .all(|(symbol, _)| spec.entries.iter().any(|e| e.symbol == *symbol))
                );
                if g.id == "flash" && ranks == 4 && r.projection == "layer.wkv" {
                    let grid = r.entry.plans(r.usage.capacity).unwrap()[1].grid;
                    assert!(grid.into_iter().product::<u32>() >= kernels_sm120::gemm::SM_COUNT);
                }
            }
        }
    }
}

#[test]
fn all_declared_fold_shapes_prepare_at_every_rank_count() {
    let mut prepared = 0;
    let mut geometries = BTreeSet::new();
    for g in geometry::GEOMETRIES {
        geometries.insert(g.id);
        let declared = projections::declaration(g)
            .unwrap()
            .expanded_profiles(&projections::geometry(g))
            .unwrap();
        let expected_uses: usize = declared
            .projections
            .iter()
            .filter(|p| p.format == Some(kernel_core::precision::format::FormatId::Fp8Block32))
            .map(|p| p.operands.len() * p.uses.len())
            .sum();
        let catalog = registration::catalog(g).unwrap();
        assert!(!catalog.is_empty());
        assert!(registration::cubin_name(g).ends_with("_v1"));
        for &ranks in admission::SUPPORTED_RANKS {
            // Verify the original catalog independently of decode overrides.
            let backend = registration::entries(g, ranks)
                .unwrap()
                .into_iter()
                .fold(Sm120::new(), Sm120::with_mxfp8_fold);
            let backend = registration::prefill_entries(g, ranks)
                .unwrap()
                .into_iter()
                .fold(backend, Sm120::with_mxfp8_prefill);
            for rank in 0..ranks {
                let entries = registration::registrations(g, ranks, rank).unwrap();
                assert_eq!(entries.len(), expected_uses);
                for id in [
                    "layer.wq_a",
                    "layer.wq_b",
                    "layer.wkv",
                    "layer.wo_a",
                    "layer.wo_b",
                    "layer.shared.gate-up",
                    "layer.shared.down",
                    "csa.indexer.wq_b",
                    "engram.wkv",
                    "dspark.main_proj",
                ] {
                    assert!(
                        entries.iter().any(|e| e.projection == id),
                        "{} TP{ranks} rank{rank}: {id}",
                        g.id
                    );
                }
                for r in entries {
                    let q = r.entry.query;
                    q.validate().unwrap();
                    assert_eq!(q.output, DType::F32);
                    assert_eq!(q.groups, r.source_groups.end - r.source_groups.start);
                    assert_eq!(r.scales_offset, r.codes_offset / 32);
                    let op = if r.usage.phase == kernel_core::precision::ExecutionPhase::Prefill {
                        backend.mxfp8_prefill_gemm(&q, Need::rows(r.usage.capacity))
                    } else {
                        backend.mxfp8_fold_gemm(&q, Need::rows(r.usage.capacity))
                    }
                    .unwrap();
                    let b = r.entry.bytes(q.m_capacity);
                    let v = |handle, bytes| View { handle, bytes };
                    let buffers = Mxfp8FoldGemmBuffers {
                        activations: v(1, b.activations),
                        activation_scales: v(2, b.activation_scales),
                        weights: v(3, b.weights),
                        weight_scales: v(4, b.weight_scales),
                        output: v(5, b.output),
                        live_rows: v(6, 4),
                        status: v(7, b.status),
                        status_row_offset: 0,
                    };
                    let launches = op
                        .launches(&buffers, View::EMPTY, Counts::rows(q.m_capacity))
                        .unwrap();
                    if r.usage.phase == kernel_core::precision::ExecutionPhase::Prefill {
                        assert_eq!(launches.len(), 2);
                        assert_eq!(
                            launches[1].symbol,
                            kernels_sm120::gemm::mxfp8_prefill::symbol(&r.entry)
                        );
                        assert_eq!(
                            launches[1].plan.grid,
                            [q.n.div_ceil(128), q.m_capacity.div_ceil(128), q.groups]
                        );
                    } else {
                        assert_eq!(launches.len(), 1);
                        assert!(catalog.iter().any(|e| e.symbol == launches[0].symbol));
                    }
                    assert_eq!(
                        op.info().bytes_of("weights"),
                        Some(u64::from(q.groups) * u64::from(q.n) * u64::from(q.k))
                    );
                    assert_eq!(
                        op.info().bytes_of("weight_scales"),
                        Some(u64::from(q.groups) * u64::from(q.n) * u64::from(q.k / 32))
                    );
                    assert_eq!(
                        op.info().layout_of("weights").unwrap().id,
                        if q.groups == 1 {
                            "row-major"
                        } else {
                            "row-major-groups"
                        }
                    );
                    if r.projection.ends_with("wo_a") {
                        assert_eq!(q.groups, g.shape.attention.o_groups / ranks);
                        assert_eq!(q.k, g.o_group_width());
                        assert_eq!(q.activation_row_stride, q.groups * q.k);
                        assert_eq!(r.source_groups.start, rank * q.groups);
                    }
                    if r.projection.ends_with("shared.gate-up") {
                        assert_eq!(q.n, g.shape.moe.intermediate / ranks);
                        assert_eq!(
                            r.codes_offset,
                            u64::from(r.operand) * u64::from(q.n) * u64::from(q.k)
                        );
                    }
                    prepared += 1;
                }
            }
        }
        assert!(registration::registrations(g, 3, 0).is_err());
        assert!(registration::registrations(g, 1, 1).is_err());
    }
    assert!(prepared > 100);
    assert!(geometries.contains("flash"));
    if std::env::var_os("CHOPINFER_DEEPSEEK41_GEOMETRY_DIRS").is_some() {
        assert!(geometries.contains("synthetic_small"));
    }
    println!("prepared {prepared} declared fold uses across {geometries:?}");
}

#[test]
fn shared_operands_and_grouped_records_match_the_declared_selections() {
    for g in geometry::GEOMETRIES {
        let keys = projections::geometry(g);
        let declaration = projections::declaration(g).unwrap();
        for &ranks in admission::SUPPORTED_RANKS {
            for rank in 0..ranks {
                for r in registration::registrations(g, ranks, rank).unwrap() {
                    let p = declaration
                        .projections
                        .iter()
                        .find(|p| p.id == r.projection)
                        .unwrap();
                    let selected = projection_spec::derive::rank_selection(
                        p,
                        &keys,
                        projection_spec::derive::Topology { ranks, rank },
                        r.operand,
                    )
                    .unwrap();
                    assert_eq!(u64::from(r.entry.query.n), selected.physical.rows);
                    assert_eq!(u64::from(r.entry.query.k), selected.physical.k);
                    assert_eq!(r.source_groups, selected.experts);
                    assert_eq!(
                        kernels_sm120::gemm::mxfp8_fold::CODES_LAYOUT,
                        "mxfp8_e4m3_rows_v1"
                    );
                    assert_eq!(
                        kernels_sm120::gemm::mxfp8_fold::SCALES_LAYOUT,
                        "mxfp8_ue8m0_scale_rows_v1"
                    );
                }
            }
        }
    }
}
