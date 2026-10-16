//! The conformance kit's sharded argmax cases (`kit::cases::argmax`) on
//! SM120's shared entries: the one-rank partials and finals through the
//! kit, and the chain of partial, all-gather and final through its
//! multi-rank runner.
//!
//! The CPU tests prepare every op on SM120, at two and four ranks for the
//! chains: each is prepared, every case declares and every launched symbol
//! is in a lab cubin. The GPU tests run
//! every check, the one-rank cases on `CHOPINFER_KERNEL_GPU` and the chains
//! at the rank count `CHOPINFER_KERNEL_GPUS` names (two or four GPUs with
//! peer access), and skip without them.

use kernel_lab::kit::cases::argmax;
use kernel_lab::kit::{self, LabModules, OpResult, Outcome, Report};
use kernel_lab::{LabResult, gpu_test, gpu_test_multi};
use serde_json::json;
use std::collections::BTreeMap;

/// Every op is prepared and declares, every launched symbol is in a lab
/// cubin, and on a GPU every check passes or is not exercised, and every
/// conformance run passes.
fn check(report: &Report, gpu: bool) {
    let failures = report.failures();
    assert!(failures.is_empty(), "{}: {failures:#?}", report.backend);
    assert_eq!(report.prepared().len(), report.ops.len(), "{report:#?}");
    for op in &report.ops {
        let OpResult::Prepared { checks, .. } = &op.result else {
            unreachable!("every op is prepared");
        };
        let outcome = |name: &str| {
            &checks
                .iter()
                .find(|check| check.name == name)
                .unwrap_or_else(|| panic!("{}: no {name} check", op.name))
                .outcome
        };
        let Outcome::Passed(declared) = outcome("declaration") else {
            panic!("{}: {:?}", op.name, outcome("declaration"));
        };
        for symbol in declared["launches"].as_array().expect("launches") {
            let symbol = symbol.as_str().expect("a symbol");
            assert!(
                LabModules::holds(symbol),
                "{}: no lab cubin holds {symbol}",
                op.name
            );
        }
        if gpu {
            assert!(
                matches!(outcome("conformance"), Outcome::Passed(_)),
                "{}: {:?}",
                op.name,
                outcome("conformance")
            );
        }
    }
}

#[test]
fn argmax_cases_declare_on_sm120() {
    let required = argmax::required();
    let report = kit::conform(&argmax::sm120(), &LabModules, None, &required).expect("report");
    if std::env::var_os("KIT_DUMP").is_some() {
        eprintln!("{}", report.json());
    }
    assert_eq!(report.ops.len(), 14);
    check(&report, false);
    // The scattered final over row-major pairs is a row op: its row and
    // chunk checks wait only for a GPU, and it claims no batch check.
    let scattered = report
        .ops
        .iter()
        .filter(|op| op.name.contains("RowMajor, dense: true, scattered: true"))
        .collect::<Vec<_>>();
    assert_eq!(scattered.len(), 2);
    for op in scattered {
        let OpResult::Prepared { checks, .. } = &op.result else {
            unreachable!("every op is prepared");
        };
        let names = checks
            .iter()
            .filter(|check| check.name.starts_with("property"))
            .map(|check| (check.name.as_str(), &check.outcome))
            .collect::<Vec<_>>();
        assert_eq!(names.len(), 2, "{}: {names:?}", op.name);
        for (name, outcome) in names {
            assert!(
                matches!(outcome, Outcome::NotExercised(reason) if reason == "no GPU session"),
                "{} {name}: {outcome:?}",
                op.name
            );
        }
    }
    for ranks in argmax::RANKS {
        let chains = argmax::required_chains();
        let report = kit::collective::conform(&argmax::sm120(), &LabModules, ranks, None, &chains)
            .expect("chain report");
        if std::env::var_os("KIT_DUMP").is_some() {
            eprintln!("{}", report.json());
        }
        assert_eq!(report.ops.len(), 2);
        check(&report, false);
    }
}

#[test]
fn sm120_passes_the_argmax_cases_on_a_gpu() {
    gpu_test(1 << 30, |session| -> LabResult<()> {
        let report = kit::conform(
            &argmax::sm120(),
            &LabModules,
            Some(session),
            &argmax::required(),
        )?;
        session.receipt(
            "conformance-kit-argmax",
            BTreeMap::from([
                ("report".into(), report.json()),
                ("not_exercised".into(), json!(report.not_exercised())),
            ]),
        )?;
        check(&report, true);
        Ok(())
    })
    .unwrap_or_else(|error| panic!("SM120 passes the argmax cases: {error}"));
}

#[test]
fn sm120_passes_the_argmax_chains_on_a_gpu() {
    gpu_test_multi(1 << 30, |sessions| -> LabResult<()> {
        let ranks = sessions.len() as u32;
        let report = kit::collective::conform(
            &argmax::sm120(),
            &LabModules,
            ranks,
            Some(sessions),
            &argmax::required_chains(),
        )?;
        sessions[0].receipt(
            &format!("conformance-kit-argmax-chains-{ranks}-ranks"),
            BTreeMap::from([
                ("report".into(), report.json()),
                ("not_exercised".into(), json!(report.not_exercised())),
            ]),
        )?;
        check(&report, true);
        Ok(())
    })
    .unwrap_or_else(|error| panic!("SM120 passes the argmax chains: {error}"));
}
