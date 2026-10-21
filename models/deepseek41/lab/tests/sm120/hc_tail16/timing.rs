//! Paired cold/hot CUDA-event timing of the tail against the served
//! Decode-16 boundary, and of the served boundary against itself (A/A), at
//! each case's widening policy.
use super::{
    artifacts,
    cases::{self, ROWS},
    device::{self, Arm, Kernels, Output},
};
use kernel_harness::{LabResult, eviction::EvictionMode};
use serde_json::json;

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU, CHOPINFER_HC_TAIL16_ARTIFACTS and CHOPINFER_HC_GRAM_CORPUS"]
fn paired_cold_hot_tail_and_aa_on_gpu() {
    let cubins = artifacts::timing_cubins().unwrap();
    let result = kernel_harness::gpu_test_with(cubins, 1 << 30, |s| -> LabResult<()> {
        let (corpus, composites, sha) = cases::load()?;
        let kernels = Kernels::load(s)?;
        let mut records = Vec::new();
        let start = std::time::Instant::now();
        let map = (0..ROWS).collect::<Vec<_>>();
        for case in &composites {
            let input = cases::input(s, &corpus, case, &map)?;
            let outputs = [
                Output::new(s, ROWS)?,
                Output::new(s, ROWS)?,
                Output::new(s, ROWS)?,
            ];
            outputs[0].assert_disjoint(&outputs[1])?;
            outputs[1].assert_disjoint(&outputs[2])?;
            outputs[0].assert_disjoint(&outputs[2])?;
            let widen = cases::widening(&corpus, case);
            // Separate coefficient buffers keep every replay identical; the
            // aliased served form moves the same bytes.
            let mut graphs = Vec::new();
            for (o, arm) in outputs.iter().zip([Arm::Served, Arm::Served, Arm::Tail]) {
                device::reset(&s.stream, &input, o, 0, false)?;
                graphs.push(s.capture(|stream| {
                    device::launch(stream, &kernels, &input, o, arm, widen, false, None).map(|_| ())
                })?);
            }
            for (pair, candidate) in [("aa", 1), ("tail", 2)] {
                for hot in [false, true] {
                    let label = format!("{}:{pair}:hot={hot}", case.id);
                    let arms = [&graphs[0], &graphs[candidate]];
                    let time = kernel_harness::with_timing_label(&label, || {
                        kernel_harness::time_ab_prepared(
                            s,
                            artifacts::WARMUP,
                            artifacts::ROUNDS,
                            [EvictionMode::Read; 2],
                            |arm, stream| {
                                if hot {
                                    arms[arm].launch(stream).map_err(|e| e.to_string())?;
                                }
                                Ok(())
                            },
                            |arm, stream| arms[arm].launch(stream).map_err(|e| e.to_string()),
                        )
                    })?;
                    records.push(
                        json!({"case":case.id,"kind":case.kind,"pair":pair,"widen":widen,"hot":hot,
                        "rounds":artifacts::ROUNDS,"warmup":artifacts::WARMUP,
                        "reference_us":f64::from(time.arms[0].median_ms)*1000.,
                        "candidate_us":f64::from(time.arms[1].median_ms)*1000.,
                        "paired_ratio":time.paired_ratio,"reference_launches":3,
                        "candidate_launches":if pair == "aa" {3} else {2}}),
                    );
                }
            }
            println!(
                "HC_TAIL16_TIMING_CASE {}",
                json!({"case":case.id,"records":4,"elapsed_s":start.elapsed().as_secs_f64()})
            );
        }
        let value = json!({"schema":"chopinfer.hc-tail16-timing.v1","corpus_sha256":sha,"cases":composites.len(),"records":records,
            "scope":"Whole-boundary AB/BA CUDA event intervals of one graph per arm. Read eviction before each arm; hot adds an untimed replay. The served Decode-16 boundary is the reference; the A/A pair times it against a second capture of itself. Site widening policy with separate post/comb buffers."});
        std::fs::write(
            std::env::var_os("CHOPINFER_HC_TAIL16_TIMING_REPORT")
                .ok_or("timing report path required")?,
            serde_json::to_vec(&value).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        s.receipt(
            "hc_tail16_timing",
            std::collections::BTreeMap::from([("case_count".into(), json!(composites.len()))]),
        )?;
        println!(
            "HC_TAIL16_TIMING_RESULT cases={} records={} PASS",
            composites.len(),
            value["records"].as_array().unwrap().len()
        );
        Ok(())
    });
    assert!(result.is_ok(), "{result:?}");
    assert!(result.unwrap().is_some(), "timing cannot skip");
}
