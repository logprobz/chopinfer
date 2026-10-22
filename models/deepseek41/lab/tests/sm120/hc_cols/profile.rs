//! One eager launch of the served boundary and of every candidate on the
//! first case at 16 and 32 rows, in a recorded order, for Nsight Compute.
use super::{
    artifacts, cases,
    device::{self, Output},
};
use kernel_harness::LabResult;
use serde_json::json;

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU, CHOPINFER_HC_COLS_ARTIFACTS and CHOPINFER_HC_GRAM_CORPUS"]
fn eager_arms_for_nsight_compute_on_gpu() {
    let result = kernel_harness::gpu_test(1 << 30, |s| -> LabResult<()> {
        let (corpus, all, _) = artifacts::cases()?;
        let kernels = device::Kernels::load(s)?;
        let mut arms = vec!["served".to_owned()];
        arms.extend(device::variants());
        let mut order = Vec::new();
        for case in artifacts::profile_cases(&all) {
            let rows = case.size();
            let input = cases::input(s, &corpus, case, &(0..rows).collect::<Vec<_>>())?;
            let out = Output::new(s, rows)?;
            let widen = cases::widening(&corpus, case);
            for arm in &arms {
                let candidate = match arm.as_str() {
                    "served" => None,
                    name => Some(device::variant(name)?),
                };
                device::reset(&s.stream, &input, &out, 0, true)?;
                let symbols = device::launch(
                    &s.stream,
                    &kernels,
                    &input,
                    &out,
                    candidate.as_ref(),
                    widen,
                    true,
                    None,
                )?;
                s.stream.synchronize().map_err(|e| e.to_string())?;
                order.push(json!({"case":case.id,"rows":rows,"arm":arm,"symbols":symbols}));
            }
        }
        let value = json!({"schema":"chopinfer.hc-cols-profile.v1","order":order});
        std::fs::write(
            std::env::var_os("CHOPINFER_HC_COLS_PROFILE_REPORT")
                .ok_or("profile report path required")?,
            serde_json::to_vec(&value).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        println!("HC_COLS_PROFILE_RESULT launches={} PASS", order.len());
        Ok(())
    });
    assert!(result.is_ok(), "{result:?}");
    assert!(result.unwrap().is_some(), "profile cannot skip");
}
