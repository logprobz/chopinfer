//! Paired cold/hot CUDA-event timing of the served boundary against itself
//! and of each candidate against it, at the site's widening policy, on
//! every seventh site at 16 and 32 rows.
use super::{
    artifacts, cases,
    device::{self, Output},
};
use kernel_harness::{LabResult, eviction::EvictionMode};
use serde_json::json;

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU, CHOPINFER_HC_COLS_ARTIFACTS and CHOPINFER_HC_GRAM_CORPUS"]
fn paired_cold_hot_served_and_candidates_on_gpu() {
    let cubins = artifacts::timing_cubins().unwrap();
    let result = kernel_harness::gpu_test_with(cubins, 1 << 30, |s| -> LabResult<()> {
        let (corpus, all, sha) = artifacts::cases()?;
        let selected = artifacts::timing_cases(&all);
        let kernels = device::Kernels::load(s)?;
        let mut arms = vec!["served".to_owned()];
        arms.extend(device::variants());
        let mut records = Vec::new();
        let start = std::time::Instant::now();
        for case in &selected {
            let rows = case.size();
            let input = cases::input(s, &corpus, case, &(0..rows).collect::<Vec<_>>())?;
            let old = Output::new(s, rows)?;
            let new = Output::new(s, rows)?;
            old.assert_disjoint(&new)?;
            let widen = cases::widening(&corpus, case);
            // Separate coefficient buffers keep every replay identical; the
            // aliased served form moves the same bytes.
            device::reset(&s.stream, &input, &old, 0, false)?;
            device::reset(&s.stream, &input, &new, 0, false)?;
            let reference = s.capture(|stream| {
                device::launch(stream, &kernels, &input, &old, None, widen, false, None).map(|_| ())
            })?;
            for arm in &arms {
                let candidate = match arm.as_str() {
                    "served" => None,
                    name => Some(device::variant(name)?),
                };
                let graph = s.capture(|stream| {
                    device::launch(
                        stream,
                        &kernels,
                        &input,
                        &new,
                        candidate.as_ref(),
                        widen,
                        false,
                        None,
                    )
                    .map(|_| ())
                })?;
                for hot in [false, true] {
                    let label = format!("{}:{arm}:hot={hot}", case.id);
                    let graphs = [&reference, &graph];
                    let time = kernel_harness::with_timing_label(&label, || {
                        kernel_harness::time_ab_prepared(
                            s,
                            artifacts::WARMUP,
                            artifacts::ROUNDS,
                            [EvictionMode::Read; 2],
                            |arm, stream| {
                                if hot {
                                    graphs[arm].launch(stream).map_err(|e| e.to_string())?;
                                }
                                Ok(())
                            },
                            |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()),
                        )
                    })?;
                    records.push(
                        json!({"case":case.id,"kind":case.kind,"rows":rows,"site":case.site,
                        "variant":arm,"widen":widen,"hot":hot,
                        "rounds":artifacts::ROUNDS,"warmup":artifacts::WARMUP,
                        "reference_us":f64::from(time.arms[0].median_ms)*1000.,
                        "candidate_us":f64::from(time.arms[1].median_ms)*1000.,
                        "paired_ratio":time.paired_ratio}),
                    );
                }
            }
            println!(
                "HC_COLS_TIMING_CASE {}",
                json!({"case":case.id,"rows":rows,"records":records.len(),"elapsed_s":start.elapsed().as_secs_f64()})
            );
        }
        let value = json!({"schema":"chopinfer.hc-cols-timing.v1","corpus_sha256":sha,"cases":selected.len(),
            "arms":arms,"records":records,
            "scope":"Whole-boundary AB/BA CUDA event intervals of one graph per arm. Read eviction before each arm; hot adds an untimed replay. The served boundary is the reference at each row count; the served arm is its A/A control on separate outputs. Site widening policy with separate post/comb buffers."});
        std::fs::write(
            std::env::var_os("CHOPINFER_HC_COLS_TIMING_REPORT")
                .ok_or("timing report path required")?,
            serde_json::to_vec(&value).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        s.receipt(
            "hc_cols_timing",
            std::collections::BTreeMap::from([("case_count".into(), json!(selected.len()))]),
        )?;
        println!(
            "HC_COLS_TIMING_RESULT cases={} records={} PASS",
            selected.len(),
            value["records"].as_array().unwrap().len()
        );
        Ok(())
    });
    assert!(result.is_ok(), "{result:?}");
    assert!(result.unwrap().is_some(), "timing cannot skip");
}
