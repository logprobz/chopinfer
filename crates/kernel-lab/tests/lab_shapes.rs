//! CPU catalog and contract coverage for shapes awaiting device qualification.
use kernel_core::contract::{AttentionPhase, DType, Validate};
use kernel_lab::lab_shapes::{
    CATALOG, INDEXER_SHAPES, PROFILES, QUALIFICATION_SHAPES, ROW_PROFILES, SPARSE_PROFILES,
    WIDE_SPARSE_PROFILES, gated_config, sparse_geometry,
};

#[test]
fn lab_shapes_have_unique_profiles_and_never_enter_production_catalogs() {
    assert_eq!(CATALOG.name, "lab-shapes");
    assert_eq!(CATALOG.profiles, PROFILES);
    for model in kernel_lab::MODELS.iter().filter(|m| m.name != CATALOG.name) {
        let catalogs = kernels_sm120::Catalogs::new(&[*model]).unwrap();
        assert!(catalogs.profiles().iter().all(|p| !PROFILES.contains(p)));
        assert!(
            catalogs
                .cubins()
                .iter()
                .all(|c| !c.name.contains("lab_shapes"))
        );
    }
    let catalogs = kernels_sm120::Catalogs::new(&[&CATALOG]).unwrap();
    assert_eq!(catalogs.models().len(), 1);
    for cubin in catalogs
        .cubins()
        .iter()
        .filter(|c| c.name.contains("lab_shapes"))
    {
        assert!(!kernel_lab::cubin(&cubin.name).unwrap().is_empty());
        assert!(
            cubin
                .entries
                .iter()
                .all(|e| e.symbol.contains("lab_shapes"))
        );
    }
}

#[test]
fn catalog_compiles_supported_geometries() {
    let variants = (CATALOG.attention)();
    assert_eq!(variants.len(), 70);
    assert_eq!(
        (CATALOG.gated)().len(),
        9 + 3 * (QUALIFICATION_SHAPES.len() + kernel_lab::lab_shapes::NEW_ROW_SHAPES.len())
    );
    for profile in ROW_PROFILES.into_iter().filter(|&p| p == PROFILES[0]) {
        let geometry = gated_config(profile);
        for phase in [
            AttentionPhase::Prefill,
            AttentionPhase::Decode,
            AttentionPhase::Verify,
        ] {
            for dtype in [DType::Bf16, DType::Fp8E4M3] {
                let entry = variants
                    .iter()
                    .find(|v| {
                        v.profile == profile && v.query.phase == phase && v.query.kv_dtype == dtype
                    })
                    .unwrap();
                entry.instantiation().unwrap();
                assert_eq!(entry.query.qk_dim, geometry.head_dim);
                assert_eq!(entry.query.v_dim, geometry.head_dim);
            }
        }
    }
    for profile in ROW_PROFILES {
        for heads in [4, 8, 16] {
            let g = sparse_geometry(profile, heads);
            for entry in kernels_sm120::attention::sparse::indexer_entries(profile, g, 2) {
                entry.query.validate().unwrap();
            }
        }
    }
    for variant in &variants {
        variant.instantiation().unwrap();
    }
    for entry in (CATALOG.gated)() {
        entry.contract().unwrap();
    }
    let sparse = (CATALOG.sparse)();
    assert_eq!(sparse.len(), 174);
    for profile in SPARSE_PROFILES {
        let attention: Vec<_> = sparse.iter().filter(|e| e.profile == profile).collect();
        assert_eq!(attention.len(), 6);
        for entry in attention {
            entry.contract().unwrap();
            assert_eq!(entry.geometry.head_dim, 128);
        }
    }
    let two_tiles = sparse_geometry(SPARSE_PROFILES[1], 4);
    assert_eq!(two_tiles.q_heads / two_tiles.kv_heads, 16);
    for (profile, heads, rotary) in INDEXER_SHAPES {
        let entries: Vec<_> = sparse.iter().filter(|e| e.profile == profile).collect();
        assert_eq!(entries.len(), 6);
        for entry in entries {
            entry.contract().unwrap();
            assert_eq!(entry.geometry.index_heads, heads);
            assert_eq!(entry.geometry.rotary_dim, rotary);
        }
    }
    let catalogs = kernels_sm120::Catalogs::new(&[&CATALOG]).unwrap();
    let cubins = catalogs.cubins();
    let lab: Vec<_> = cubins
        .iter()
        .filter(|c| c.name.contains("lab_shapes") && !c.name.contains("lab_shapes_w7"))
        .collect();
    assert_eq!(lab.len(), 12);
    assert_eq!(lab.iter().map(|c| c.entries.len()).sum::<usize>(), 93);
}

#[test]
fn qualification_covers_tile_groups_row_warps_and_wide_sparse_instances() {
    let variants = (CATALOG.attention)();
    for group in 1..=8 {
        for dtype in [DType::Bf16, DType::Fp8E4M3] {
            for phase in [AttentionPhase::Prefill, AttentionPhase::Decode] {
                let entry = variants
                    .iter()
                    .find(|v| {
                        v.profile.tag().starts_with("lab_shapes_w7_")
                            && v.query.qk_dim == 128
                            && v.query.q_heads == group
                            && v.query.phase == phase
                            && v.query.kv_dtype == dtype
                    })
                    .unwrap();
                entry.instantiation().unwrap();
                assert_eq!(entry.gated_fold.is_some(), phase == AttentionPhase::Decode);
                assert!(!entry.invariance.claims().is_empty());
            }
        }
    }
    for heads in 1..=16 {
        assert!(
            QUALIFICATION_SHAPES
                .iter()
                .any(|s| s.head_dim == 128 && s.rotary == 32 && s.q_heads == heads)
        );
    }
    for heads in [1, 2, 3, 5, 7, 8, 10, 12, 14, 16] {
        assert!(
            QUALIFICATION_SHAPES
                .iter()
                .any(|s| s.head_dim == 256 && s.rotary == 32 && s.q_heads == heads)
        );
    }
    for (profile, heads) in WIDE_SPARSE_PROFILES.into_iter().zip([6, 12]) {
        let g = sparse_geometry(profile, 8);
        assert_eq!((g.q_heads, g.kv_heads, g.head_dim), (heads, 1, 128));
        assert_eq!(
            (g.index_heads, g.rotary_dim, g.top_blocks * g.block),
            (8, 32, 1024)
        );
        assert_eq!(g.theta, 5e6);
        assert_eq!(g.mrope, [6, 5, 5]);
    }
}

/// Serving admission reads this same table. Check template operands against
/// actual compiled entries, not a second list of nominally supported shapes.
#[test]
fn admission_table_has_every_compiled_attention_instance() {
    use kernel_core::contract::{GatedAttentionQuery, SparseAttentionQuery};
    use kernels_sm120::attention::{self, gated, qualification as q, sparse};
    use kernels_sm120::invariance::Rows;
    let catalogs = kernel_lab::catalogs();
    let rows = gated::catalog(catalogs);
    let dense = attention::catalog(catalogs);
    let sparse = sparse::catalog(catalogs);
    for shape in q::HEAD_SHAPES {
        for &(rotary, sections) in q::ROTARY_SECTIONS {
            for &heads in shape.rows {
                for dtype in [DType::Bf16, DType::Fp8E4M3] {
                    assert!(
                        rows.iter().any(|e| matches!(e.query,
                            GatedAttentionQuery::QkNormRopeKvWrite {
                                q_heads, kv_heads: 1, head_dim, rotary_dim, mrope,
                                page_size: 64, kv_dtype, ..
                            } if q_heads == heads && head_dim == shape.head_dim
                                && rotary_dim == rotary && mrope == sections && kv_dtype == dtype
                        )),
                        "rows d{} r{rotary} q{heads} {dtype:?}",
                        shape.head_dim
                    );
                }
                assert!(rows.iter().any(|e| matches!(e.query,
                    GatedAttentionQuery::OutputGate { heads: h, head_dim, .. }
                        if h == heads && head_dim == shape.head_dim)));
            }
            for &group in shape.dense_groups {
                for dtype in [DType::Bf16, DType::Fp8E4M3] {
                    for phase in [
                        AttentionPhase::Prefill,
                        AttentionPhase::Decode,
                        AttentionPhase::Verify,
                    ] {
                        assert!(
                            dense.iter().any(|e| {
                                let a = e.query;
                                a.q_heads == group
                                    && a.kv_heads == 1
                                    && a.qk_dim == shape.head_dim
                                    && a.v_dim == shape.head_dim
                                    && a.page_size == 64
                                    && a.phase == phase
                                    && a.kv_dtype == dtype
                                    && (phase != AttentionPhase::Decode
                                        || e.gated_fold.is_some_and(|f| {
                                            f.rotary == rotary && f.sections == sections
                                        }))
                            }),
                            "dense d{} r{rotary} g{group} {dtype:?} {phase:?}",
                            shape.head_dim
                        );
                    }
                }
            }
        }
        for &group in shape.sparse_groups {
            for &top in q::KEPT_BLOCKS {
                for dtype in [DType::Bf16, DType::Fp8E4M3] {
                    for phase in [
                        AttentionPhase::Prefill,
                        AttentionPhase::Decode,
                        AttentionPhase::Verify,
                    ] {
                        assert!(
                            sparse.iter().any(|e| {
                                let g = e.geometry;
                                g.head_dim == shape.head_dim
                                    && g.q_heads == group
                                    && g.kv_heads == 1
                                    && g.page_size == 64
                                    && g.block == q::BLOCK
                                    && g.top_blocks == top
                                    && matches!(e.query, SparseAttentionQuery::SparseAttention {
                                    phase: p, kv_dtype: d, .. } if p == phase && d == dtype)
                                    && (phase != AttentionPhase::Decode
                                        || matches!(e.invariance.rows,
                                    Rows::OneRow { max } if max >= q::EXACT_VERIFY_ROWS))
                            }),
                            "sparse d{} g{group} k{top} {dtype:?} {phase:?}",
                            shape.head_dim
                        );
                    }
                }
            }
        }
    }
    for &heads in q::INDEX_HEADS {
        for &(rotary, sections) in q::ROTARY_SECTIONS {
            for &top in q::KEPT_BLOCKS {
                for kind in [
                    sparse::Kind::Query,
                    sparse::Kind::KeyWrite,
                    sparse::Kind::TopK,
                    sparse::Kind::Scores {
                        rows: 1,
                        by_run: false,
                    },
                    sparse::Kind::Scores {
                        rows: 8,
                        by_run: true,
                    },
                    sparse::Kind::Scores {
                        rows: 64.min(256 / heads),
                        by_run: false,
                    },
                ] {
                    assert!(
                        sparse.iter().any(|e| {
                            let g = e.geometry;
                            e.kind == kind
                                && g.index_heads == heads
                                && g.index_dim == 128
                                && g.rotary_dim == rotary
                                && g.mrope == sections
                                && g.top_blocks == top
                                && g.block == q::BLOCK
                                && g.ring == 16
                                && g.page_size == 64
                        }),
                        "indexer i{heads} r{rotary} k{top} {kind:?}"
                    );
                }
            }
        }
    }
}
