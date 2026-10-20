//! Real full-chunk query norm/ceil parity, admission, replay and paired timing.
mod artifacts;
#[path = "../fusion_checks.rs"]
mod fusion_checks;
mod replay;
mod served;

use bundle_deepseek41::sm120::{ceil_prefill, query_prefill as registration};
use kernel_harness::LabResult;
use kernels_sm120::row::{
    block_quantize,
    rmsnorm::{RmsNorm, RmsNormTarget},
    rmsnorm_f32_prefill::Entry,
};
use model_deepseek41_contract::{geometry::FLASH, serving::PREFILL_TOKENS};

const WIDTH: usize = 1280;
const OFFSET: usize = 3;
const NORM: &str = "chop_deepseek41_q_norm_f32_v1";
const ROLES: [&str; 4] = ["normed", "codes", "scales", "status"];

fn norm() -> RmsNorm {
    RmsNorm::new(
        registration::entries().unwrap()[0].reference.query.norm,
        PREFILL_TOKENS,
        RmsNormTarget::SM120F,
    )
    .unwrap()
}
fn quant() -> block_quantize::Entry {
    let query = registration::entries().unwrap()[0]
        .reference
        .query
        .quantizer();
    ceil_prefill::quantizers(&FLASH, 4, 8)
        .unwrap()
        .into_iter()
        .find(|e| e.query == query)
        .unwrap()
}
fn validate(rows: u32, live: u32, entry: &Entry) -> LabResult<()> {
    let mut n = norm().plan(rows).map_err(|e| e.to_string())?;
    n.grid[0] = rows;
    fusion_checks::validate_replay(
        rows,
        live,
        PREFILL_TOKENS,
        &[
            n,
            quant().plan(rows).map_err(|e| e.to_string())?,
            entry.plan(rows).map_err(|e| e.to_string())?,
        ],
    )
}

#[test]
fn reject_impossible_case_parameters_before_device_work() {
    for entry in registration::entries().unwrap() {
        for rows in [1, 6, 32, 64, 8191, 8192] {
            for live in fusion_checks::live_counts(rows) {
                validate(rows, live, &entry).unwrap();
            }
            assert!(validate(rows, rows + 1, &entry).is_err());
        }
        assert!(validate(0, 0, &entry).is_err());
        assert!(validate(8193, 8193, &entry).is_err());
        assert_eq!(entry.plan(8192).unwrap().workspace_bytes, 0);
    }
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_QUERY_PREFILL"]
fn stage_and_validate_artifacts_on_cpu() {
    artifacts::stage().unwrap();
    artifacts::verify().unwrap();
    let cases = artifacts::corpus().unwrap();
    assert_eq!(cases.len(), 2);
    println!("QUERY_PREFILL_DRY_RUN cases=2 modules=5 entries=9 PASS");
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_QUERY_PREFILL"]
fn validate_frozen_corpus_and_modules_on_cpu() {
    artifacts::verify().unwrap();
    assert_eq!(artifacts::corpus().unwrap().len(), 2);
    println!("QUERY_PREFILL_FROZEN cases=2 modules=5 entries=9 PASS");
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_QUERY_PREFILL"]
fn bitwise_real_8k_status_replays_and_paired_timing_on_gpu() {
    artifacts::verify().unwrap();
    assert!(kernel_harness::selected_ordinals().unwrap().is_some());
    let result =
        kernel_harness::gpu_test_with(artifacts::probes(), 1 << 30, |s| replay::run(s, false))
            .unwrap();
    assert!(result.is_some());
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_QUERY_PREFILL"]
fn memcheck_real_8k_on_gpu() {
    artifacts::verify().unwrap();
    assert!(kernel_harness::selected_ordinals().unwrap().is_some());
    let result =
        kernel_harness::gpu_test_with(artifacts::probes(), 1 << 30, |s| replay::run(s, true))
            .unwrap();
    assert!(result.is_some());
}
