//! Paired cold/hot CUDA-event timing of each candidate against the served
//! pair, at the widening policy of every real C1 and C8 boundary.
use super::{
    artifacts,
    device::{self, Input, Kernels, Output, Tickets, VARIANTS},
};
use kernel_harness::{LabResult, eviction::EvictionMode};
use serde_json::json;

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU, CHOPINFER_HC_COMPLETION_ARTIFACTS and CHOPINFER_HC_GRAM_CORPUS"]
fn paired_cold_hot_served_configuration_on_gpu() {
    let cubins = artifacts::timing_cubins().unwrap();
    let result = kernel_harness::gpu_test_with(cubins, 1 << 30, |s| -> LabResult<()> {
        let (cases, sha) = artifacts::cases()?;
        let kernels = Kernels::load(s)?;
        let mut records = Vec::new();
        let start = std::time::Instant::now();
        for case in &cases {
            let rows = case.rows;
            let input = Input::new(s, case, &(0..rows).collect::<Vec<_>>())?;
            let old = Output::new(s, rows)?;
            let new = Output::new(s, rows)?;
            old.assert_disjoint(&new)?;
            let old_tickets = Tickets::new(s, rows)?;
            let new_tickets = Tickets::new(s, rows)?;
            let widen = case.widening();
            // Separate coefficient buffers keep every replay identical; the
            // aliased served form moves the same bytes.
            device::reset(&s.stream, &input, &old, 0, false)?;
            device::reset(&s.stream, &input, &new, 0, false)?;
            let reference = s.capture(|stream| {
                device::launch(
                    stream,
                    &kernels,
                    &input,
                    &old,
                    &old_tickets,
                    "served",
                    widen,
                    false,
                )
                .map(|_| ())
            })?;
            for variant in VARIANTS {
                let candidate = s.capture(|stream| {
                    device::launch(
                        stream,
                        &kernels,
                        &input,
                        &new,
                        &new_tickets,
                        variant,
                        widen,
                        false,
                    )
                    .map(|_| ())
                })?;
                for hot in [false, true] {
                    let label = format!("{}:{variant}:hot={hot}", case.id);
                    let graphs = [&reference, &candidate];
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
                    records.push(json!({"case":case.id,"rows":rows,"variant":variant,"widen":widen,"hot":hot,
                        "rounds":artifacts::ROUNDS,"warmup":artifacts::WARMUP,
                        "reference_us":f64::from(time.arms[0].median_ms)*1000.,
                        "candidate_us":f64::from(time.arms[1].median_ms)*1000.,
                        "paired_ratio":time.paired_ratio,"reference_launches":2,"candidate_launches":1}));
                }
                new_tickets.check(&s.stream)?;
            }
            old_tickets.check(&s.stream)?;
            println!(
                "HC_COMPLETION_TIMING_CASE {}",
                json!({"case":case.id,"rows":rows,"records":VARIANTS.len()*2,"elapsed_s":start.elapsed().as_secs_f64()})
            );
        }
        let value = json!({"schema":"chopinfer.hc-completion-timing.v1","corpus_sha256":sha,"cases":cases.len(),"records":records,
            "scope":"Whole-boundary AB/BA CUDA event intervals of one graph per arm. Read eviction before each arm; hot adds an untimed replay. The served pair is the reference at each row count. Site widening policy with separate post/comb buffers."});
        std::fs::write(
            std::env::var_os("CHOPINFER_HC_COMPLETION_TIMING_REPORT")
                .ok_or("timing report path required")?,
            serde_json::to_vec(&value).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        s.receipt(
            "hc_completion_timing",
            std::collections::BTreeMap::from([("case_count".into(), json!(cases.len()))]),
        )?;
        println!(
            "HC_COMPLETION_TIMING_RESULT cases={} records={} PASS",
            cases.len(),
            value["records"].as_array().unwrap().len()
        );
        Ok(())
    });
    assert!(result.is_ok(), "{result:?}");
    assert!(result.unwrap().is_some(), "timing cannot skip");
}
