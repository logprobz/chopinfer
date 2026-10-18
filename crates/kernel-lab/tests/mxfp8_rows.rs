//! Exact MXFP8 row decode, poisoned replays, invariance, and a device mutant.
//! Run with CHOPINFER_KERNEL_GPU and CHOPINFER_KERNEL_REQUIRE=1 for GPU evidence.

use kernel_core::backend::{Backend, Need};
use kernel_core::mxfp8_rows::Mxfp8Rows;
use kernel_lab::families::mxfp8_rows::{MUTANT, MUTANT_CUBIN};
use kernel_lab::kit::cases::mxfp8_rows::{self, CAPACITY, Inputs};
use kernel_lab::kit::{self, LabModules, Modules, OpResult, Outcome, Report};
use kernel_lab::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::mxfp8_decode::{CUBIN, SYMBOL};
use serde_json::json;
use std::collections::BTreeMap;

fn report(session: Option<&Session>) -> LabResult<Report> {
    kit::conform(
        &Sm120::new().with_mxfp8_rows(SYMBOL),
        &LabModules,
        session,
        &mxfp8_rows::family(),
    )
}

fn check(report: &Report, gpu: bool) {
    assert_eq!(report.ops.len(), 5);
    assert_eq!(report.prepared().len(), 5, "{report:#?}");
    assert!(report.failures().is_empty(), "{:#?}", report.failures());
    for op in &report.ops {
        assert!(op.required);
        let OpResult::Prepared { checks, .. } = &op.result else {
            panic!("the op did not prepare");
        };
        assert_eq!(
            checks
                .iter()
                .filter(|c| c.name.starts_with("property "))
                .count(),
            3
        );
        for name in ["declaration", "need", "conformance"] {
            let outcome = &checks.iter().find(|c| c.name == name).unwrap().outcome;
            if gpu || name != "conformance" {
                assert!(
                    matches!(outcome, Outcome::Passed(_)),
                    "{} {name}: {outcome:?}",
                    op.name
                );
            }
        }
    }
    if gpu {
        assert!(
            report.not_exercised().is_empty(),
            "{:#?}",
            report.not_exercised()
        );
    }
}

#[test]
fn mxfp8_row_cases_declare_and_resolve_their_locked_entries() {
    let specs = kernel_lab::families::mxfp8_rows::cubins();
    assert_eq!(specs.len(), 2);
    assert_eq!(kernel_lab::cubin_entries(CUBIN).unwrap(), [SYMBOL]);
    assert_eq!(kernel_lab::cubin_entries(MUTANT_CUBIN).unwrap(), [MUTANT]);
    check(&report(None).unwrap(), false);
}

#[test]
fn mxfp8_rows_match_the_oracle_and_properties_gpu() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let report = report(Some(session))?;
        session.receipt(
            "mxfp8-rows",
            BTreeMap::from([("report".into(), report.json())]),
        )?;
        check(&report, true);
        eprintln!("MXFP8 decode: 5 prepared cases, 15 property checks, all GPU checks exercised");
        Ok(())
    })
    .expect("MXFP8 row decode conformance");
}

#[test]
fn mxfp8_rows_reject_the_truncating_device_mutant_gpu() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let query = Mxfp8Rows { width: 128 };
        let need = Need::rows(CAPACITY);
        let op = Sm120::new()
            .with_mxfp8_rows(MUTANT)
            .decode_mxfp8_rows(&query, need)
            .map_err(|e| e.to_string())?;
        let case = mxfp8_rows::case(query, CAPACITY, Inputs::Random, op)?;
        case.validate()?;
        let modules = LabModules;
        let functions = modules.load(session)?;
        let probes = modules.probes(session)?;
        let checks = case.checks(
            need,
            Some(session),
            Some(functions.as_ref()),
            probes.as_ref(),
        );
        let result = &checks
            .iter()
            .find(|c| c.name == "conformance")
            .unwrap()
            .outcome;
        let Outcome::Failed(error) = result else {
            return Err(format!("truncating device mutant did not fail: {result:?}"));
        };
        // Accept only the constructed numerical witness, never an unrelated
        // launch, allocation, declaration, or module failure.
        if !error.contains("output byte 0 is 0x00, the reference 0x01") {
            return Err(format!("mutant failed outside its RNE witness: {error}"));
        }
        session.receipt(
            "mxfp8-rows-mutant",
            BTreeMap::from([
                ("rejected_device_mutants".into(), json!(1)),
                (
                    "witness".into(),
                    json!({"code": 3, "scale": 1, "rne": 1, "truncated": 0}),
                ),
                ("error".into(), json!(error)),
            ]),
        )?;
        eprintln!("MXFP8 decode: 1 truncating device mutant rejected at code 0x03, scale 1");
        Ok(())
    })
    .expect("the device mutant must fail at the RNE witness");
}
