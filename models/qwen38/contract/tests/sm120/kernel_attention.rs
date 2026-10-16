//! Qwen3.8's attention catalogs (`sm120::attention`): the QSA entries'
//! contracts, plans and splits, their exact verify rows, the cubin of each
//! profile, and the gated rows' derived TP4 geometry. CPU only.

use kernel_core::contract::{AttentionPhase, NumericalContract, SparseAttentionQuery};
use kernels_sm120::attention::sparse::{Entry, cubin_specs};
use kernels_sm120::invariance::{Batch, Invariance, Rows};
use model_qwen38_contract::sm120::attention::{GATED, Q_HEADS, SPARSE, sparse_catalog};
use model_qwen38_contract::sm120::catalogs;
use model_qwen38_contract::sm120::config::EXACT_VERIFY_ROWS;

/// The derived TP4 attention heads keep the value the TP4 cubins were
/// locked with.
#[test]
fn derived_tp4_heads_keep_the_locked_values() {
    assert_eq!(Q_HEADS, 6);
    assert_eq!(GATED.q_heads, 6);
    assert_eq!(SPARSE.q_heads, 6);
}

fn find(name: &str) -> Entry {
    sparse_catalog()
        .into_iter()
        .find(|entry| entry.name == name)
        .unwrap()
}

#[test]
fn catalog_entries_have_contracts_and_plans() {
    let entries = sparse_catalog();
    assert_eq!(
        entries.len(),
        model_qwen38_contract::sm120::geometry::MODELS.len() * (12 + 6)
    );
    assert_eq!(
        entries
            .iter()
            .filter(|entry| entry.profile == model_qwen38_contract::sm120::config::PROFILE)
            .count(),
        12
    );
    for entry in &entries {
        entry.contract().unwrap();
        entry.plan(entry.capacity(), 262_144).unwrap();
        assert!(entry.plan(entry.capacity() + 1, 1).is_err());
    }
    let decode = find("qwen38_attn_bf16_decode");
    assert_eq!(
        decode.query.numerical_contract_id().unwrap().as_str(),
        "sparse.attention-paged-kv-bf16.decode.v1"
    );
    assert_eq!(decode.splits(1), 64);
    assert_eq!(decode.splits(8), 64);
    assert_eq!(decode.splits(32), 16);
    // Wider decode buckets (CONC-QWEN): fewer splits fill the SMs.
    assert_eq!([64, 128, 256].map(|rows| decode.splits(rows)), [8, 4, 2]);
    assert_eq!(decode.plan(256, 262_144).unwrap().grid, [256, 1, 2]);
    assert_eq!(decode.merge_plan(256).unwrap().unwrap().grid, [256, 6, 1]);
    let scores = find("qwen38_block_scores_decode");
    assert_eq!(scores.capacity(), 256);
    assert_eq!(scores.plan(256, 262_144).unwrap().grid, [376, 1, 1]);
    assert_eq!(decode.plan(1, 262_144).unwrap().grid, [1, 1, 64]);
    assert_eq!(decode.merge_plan(1).unwrap().unwrap().grid, [1, 6, 1]);
    let prefill = find("qwen38_attn_fp8_prefill");
    assert_eq!([1, 64, 375, 8192].map(|rows| prefill.splits(rows)), [1; 4]);
    assert_eq!(prefill.merge_plan(8192).unwrap(), None);
    assert_eq!(prefill.workspace_bytes(8192), 0);
    assert_eq!(prefill.dynamic_shared_bytes(), 38_912);
    assert_eq!(decode.dynamic_shared_bytes(), 71_680);
    let scores = find("qwen38_block_scores_prefill");
    assert_eq!(scores.plan(2048, 262_144).unwrap().grid, [32, 64, 1]);
    assert_eq!(scores.dynamic_shared_bytes(), 98_304);
    assert_eq!(SPARSE.score_stride(2051), 512);
    assert_eq!(SPARSE.score_stride(2052), 576);
}

/// An exact verify step's rows take AR's split count, which splits, so
/// each decode attention entry's merge runs there and claims its rows.
#[test]
fn exact_verify_rows_split_as_ar() {
    let rows = Rows::OneRow {
        max: EXACT_VERIFY_ROWS,
    };
    let decode = sparse_catalog()
        .into_iter()
        .filter(|entry| {
            matches!(
                entry.query,
                SparseAttentionQuery::SparseAttention {
                    phase: AttentionPhase::Decode,
                    ..
                }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        decode.len(),
        model_qwen38_contract::sm120::PROFILES.len() * 2
    );
    for entry in &decode {
        for bucket in 1..=EXACT_VERIFY_ROWS {
            assert_eq!(entry.splits(bucket), 64, "{}", entry.name);
        }
        for (symbol, invariance) in entry.declared_symbols() {
            assert_eq!(invariance.rows, rows, "{symbol}");
            assert_eq!(invariance.batch, Batch::None, "{symbol}");
        }
    }
    for name in [
        "qwen38_index_query",
        "qwen38_index_key_write",
        "qwen38_block_topk",
        "qwen38_block_scores_decode",
    ] {
        assert_eq!(find(name).invariance.rows, rows, "{name}");
    }
    for name in ["qwen38_block_scores_verify", "qwen38_attn_bf16_verify"] {
        assert_eq!(find(name).invariance, Invariance::NONE, "{name}");
    }
}

#[test]
fn score_grids_bound_launch_cost_across_context_and_row_buckets() {
    let name = "qwen38_block_scores_decode";
    let entry = sparse_catalog()
        .into_iter()
        .find(|e| e.name == name)
        .unwrap();
    for rows in [1, 2, 16, 32, 33, 64, 128, 256] {
        for context in [1, 2048, 2052, 8192, 262_144] {
            let grid = entry.plan(rows, context).unwrap().grid;
            assert!(grid[0] <= 2 * 188);
            assert_eq!(grid[1], 1);
        }
    }
}

#[test]
fn cubin_names_its_traits() {
    let specs = cubin_specs(
        catalogs(),
        Some(model_qwen38_contract::sm120::config::PROFILE),
    );
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, "sparse_attention_qwen38_sm120");
    assert_eq!(specs[0].entries.len(), 18);
    assert!(specs[0].entries.iter().any(|entry| {
        entry
            .body
            .contains("sparse::Attention<6,1,256,64,4,512,true,16>")
    }));
    assert!(specs[0].entries.iter().any(|entry| {
        entry
            .body
            .contains("sparse::Indexer<4,128,64,11,10,4,16,64,512>")
    }));
    assert!(
        cubin_specs(
            catalogs(),
            Some(
                kernels_sm120::Model::new(
                    kernels_sm120::profile_names!("other-model", "other", 4, 2),
                    model_qwen38_contract::sm120::config::CONFIG
                )
                .at(4)
            )
        )
        .is_empty()
    );
}
