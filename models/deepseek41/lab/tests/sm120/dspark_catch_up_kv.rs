//! The drafter catch-up's KV projection (`wkv`, N 512, K 5120) moved from
//! the prefill-shaped fold entry to the decode path. Two exact comparisons
//! per row count make the chain: the prefill-shaped entry against the split
//! fold decode entry, and the split fold against the prefetch ticket that
//! serving selects.
use super::mxfp8_fold_decode as previous;
use bundle_deepseek41::sm120::mxfp8_fold as registration;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use kernels_sm120::gemm::{mxfp8_fold_decode, mxfp8_fold_ticket};
use model_deepseek41_contract::geometry::FLASH;
use serde_json::json;
use std::collections::BTreeMap;

/// Catch-up chunks at C1 K3 hold one to four rows, at C8 up to 32.
const ROWS: [u32; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32];

fn entries() -> (mxfp8_fold_decode::Entry, mxfp8_fold_ticket::Entry) {
    let decode = registration::decode_catalog(&FLASH)
        .unwrap()
        .into_iter()
        .find(|e| !e.paired && e.query.n == 512 && e.query.k == 5120 && e.query.groups == 1)
        .expect("the wkv split fold entry");
    let ticket = registration::prefetch_catalog(&FLASH)
        .unwrap()
        .into_iter()
        .find(|t| {
            !t.paired
                && Mxfp8FoldGemmQuery {
                    m_capacity: 32,
                    ..t.query
                } == decode.query
        })
        .expect("the wkv prefetch ticket");
    (decode, ticket)
}

#[test]
fn dspark_catch_up_kv_cpu_serves_the_prefetch_ticket_at_tp4() {
    let (decode, ticket) = entries();
    assert!(ticket.prefetch_weights);
    assert!(
        registration::prefetch_entries(&FLASH, 4)
            .unwrap()
            .iter()
            .any(|t| t.symbol == ticket.symbol)
    );
    assert_eq!(decode.query.m_capacity, 32);
}

#[test]
fn dspark_catch_up_kv_gpu_matches_the_prefill_shaped_fold() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("the catch-up KV comparison needs an idle GPU").unwrap();
        return;
    }
    let (decode, ticket) = entries();
    let result = kernel_harness::gpu_test(1 << 30, |session| {
        let mut records = Vec::new();
        for rows in ROWS {
            // compare_with fails on any differing output or status bit.
            let fold = previous::compare_with(session, &decode, rows, None, None, false, false)?;
            let weights = previous::Weights::new(session, &decode)?;
            let prefetch = previous::compare_with(
                session,
                &decode,
                rows,
                Some(&ticket),
                Some(&weights),
                false,
                false,
            )?;
            for (label, record) in [
                ("prefill_vs_split", &fold),
                ("split_vs_prefetch", &prefetch),
            ] {
                if record["bitwise"] != true {
                    return Err(format!("{label} rows={rows} not bitwise: {record}"));
                }
            }
            eprintln!("CATCHUP_KV_BITWISE rows={rows} PASS");
            records.push(
                json!({"rows": rows, "prefill_vs_split": fold, "split_vs_prefetch": prefetch}),
            );
        }
        session.receipt(
            "dspark-catch-up-kv",
            BTreeMap::from([("comparisons".into(), json!(records))]),
        )?;
        eprintln!("CATCHUP_KV_COMPLETE cases={}", records.len());
        Ok(())
    })
    .unwrap();
    assert!(result.is_some(), "the comparison ran");
}
