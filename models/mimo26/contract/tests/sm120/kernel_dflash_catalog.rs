use kernel_core::contract::{DType, RuntimeCounts, StaticQuery, SwigluColumns, WeightFormat};
use kernels_sm120::{gemm, row};

#[test]
fn dflash_bf16_instances_cover_every_projection_and_capacity() {
    let catalog = gemm::catalog(model_mimo26_contract::sm120::catalogs());
    for (name, n, k, swiglu) in [
        ("q", 2048, 4096, false),
        ("qkv", 2560, 4096, false),
        ("context_kv", 2560, 4096, false),
        ("o", 4096, 2048, false),
        ("gate_up", 8192, 4096, true),
        ("down", 4096, 4096, false),
        ("fc", 4096, 5120, false),
    ] {
        for m in [1, 2, 4, 8, 16, 32] {
            let symbol = format!("chop_gemm_mimo_dflash_{name}_m{m}_v1");
            let variant = catalog.iter().find(|item| item.symbol == symbol).unwrap();
            assert_eq!(
                variant.profile,
                model_mimo26_contract::sm120::config::PROFILE
            );
            let StaticQuery::Gemm(query) = variant.key.query else {
                panic!("DFlash catalog entry is not GEMM");
            };
            assert_eq!((query.m_capacity, query.n, query.k), (m, n, k));
            assert_eq!(query.weight, WeightFormat::Bf16);
            assert_eq!(query.output, DType::Bf16);
            assert_eq!(
                query.epilogues.swiglu,
                swiglu.then_some(SwigluColumns::ConcatenatedHalves)
            );
            assert!(gemm::fragment_weight_enabled(&variant.key));
            let split_k = gemm::choose_split_k(&variant.key).unwrap();
            gemm::plan(
                &variant.key,
                RuntimeCounts {
                    m,
                    batch: m,
                    context: 0,
                    routed_rows: 0,
                    split_k,
                },
            )
            .unwrap();
        }
        let symbol = format!("chop_gemm_mimo_dflash_{name}_prefill_v1");
        let variant = catalog.iter().find(|item| item.symbol == symbol).unwrap();
        let StaticQuery::Gemm(query) = variant.key.query else {
            panic!("DFlash prefill entry is not GEMM");
        };
        assert_eq!((query.m_capacity, query.n, query.k), (8192, n, k));
        assert!(gemm::fragment_weight_enabled(&variant.key));
    }
}

#[test]
fn dflash_row_entries_use_token_head_grid_and_explicit_strides() {
    let catalog = row::catalog(model_mimo26_contract::sm120::catalogs());
    for (name, heads) in [("q", 16), ("k", 2)] {
        for operation in ["norm", "rope_pos32", "rope_pos64"] {
            let symbol = format!("chop_row_dflash_{name}_{operation}_v1");
            let entry = catalog.iter().find(|item| item.symbol == symbol).unwrap();
            assert_eq!(entry.profile, model_mimo26_contract::sm120::config::PROFILE);
            let plan = entry
                .plan(
                    2,
                    RuntimeCounts {
                        m: 2,
                        batch: 2,
                        context: 0,
                        routed_rows: 0,
                        split_k: 1,
                    },
                )
                .unwrap();
            assert_eq!(plan.grid, [2, heads, 1]);
            for stride in ["input_stride", "output_stride"] {
                assert!(entry.params.iter().any(|param| matches!(
                    param,
                    kernel_build::EntryParam::U32 { name } if name == stride
                )));
            }
        }
    }
    for name in ["argmax_partial", "argmax_final"] {
        let symbol = format!("chop_row_dflash_{name}_v1");
        catalog
            .iter()
            .find(|entry| entry.symbol == symbol)
            .unwrap()
            .contract()
            .unwrap();
    }
}
