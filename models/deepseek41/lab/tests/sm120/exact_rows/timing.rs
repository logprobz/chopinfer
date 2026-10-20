use super::{
    artifacts,
    device::{self, Input, Kernels, Output, VARIANTS},
};
use kernel_harness::{LabResult, eviction::EvictionMode};
use serde_json::json;
#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU, CHOPINFER_HC_ROWS_ARTIFACTS and CHOPINFER_HC_GRAM_CORPUS"]
fn paired_cold_hot_current_path_on_gpu() {
    let cubins = artifacts::timing_cubins().unwrap();
    let result = kernel_harness::gpu_test_with(cubins, 1024 * 1024 * 1024, |s| -> LabResult<()> {
        let (cases, sha) = artifacts::cases()?;
        let kernels = Kernels::load(s)?;
        let mut records = Vec::new();
        for case in &cases {
            let input = Input::new(s, case, &(0..case.rows).collect::<Vec<_>>())?;
            let old = Output::new(s, case.rows)?;
            let output = Output::new(s, case.rows)?;
            old.assert_disjoint(&output)?;
            for widen in [false, true] {
                for variant in VARIANTS {
                    old.poison(&s.stream, 0)?;
                    output.poison(&s.stream, 0)?;
                    let reference = s.capture(|stream| {
                        device::launch(stream, &kernels, &input, &old, "reference", widen)
                    })?;
                    let candidate = s.capture(|stream| {
                        device::launch(stream, &kernels, &input, &output, variant, widen)
                    })?;
                    for hot in [false, true] {
                        let label = format!("{}:{variant}:widen={widen}:hot={hot}", case.id);
                        let graphs = [&reference, &candidate];
                        let time = kernel_harness::with_timing_label(&label, || {
                            kernel_harness::time_ab_prepared(
                                s,
                                2,
                                7,
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
                        records.push(json!({"case":case.id,"rows":case.rows,"variant":variant,"widen":widen,"hot":hot,"rounds":7,"warmup":2,"reference_us":f64::from(time.arms[0].median_ms)*1000.,"candidate_us":f64::from(time.arms[1].median_ms)*1000.,"paired_ratio":time.paired_ratio,"reference_launches":3,"candidate_launches":device::launches(variant)}));
                    }
                    device::read(&s.stream, &old)?;
                    device::read(&s.stream, &output)?;
                }
            }
            println!(
                "HC_ROWS_TIMING_CASE {}",
                json!({"case":case.id,"rows":case.rows,"records":24})
            );
        }
        let value = json!({"schema":"chopinfer.hc-exact-rows-timing.v1","corpus_sha256":sha,"cases":cases.len(),"records":records,"scope":"Whole-boundary AB/BA CUDA event intervals. Read eviction before each arm; hot adds untimed replay. Round-six AR baseline at each row count; 32-row operands are real Verify captures. Both widening policies are measured separately. No PDL component times are summed."});
        std::fs::write(
            std::env::var_os("CHOPINFER_HC_ROWS_TIMING_REPORT")
                .ok_or("timing report path required")?,
            serde_json::to_vec(&value).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        s.receipt(
            "hc_exact_rows_timing",
            std::collections::BTreeMap::from([("case_count".into(), json!(cases.len()))]),
        )?;
        println!(
            "HC_ROWS_TIMING_RESULT cases={} records={} PASS",
            cases.len(),
            value["records"].as_array().unwrap().len()
        );
        Ok(())
    });
    assert!(result.is_ok(), "{result:?}");
    assert!(result.unwrap().is_some(), "timing cannot skip");
}
