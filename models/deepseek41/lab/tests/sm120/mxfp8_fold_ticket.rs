//! Exact comparisons at every row count, paired graph timing, and stage profiles.
use super::mxfp8_fold_decode::{self as previous, modules};
use bundle_deepseek41::sm120::mxfp8_fold as registration;
use kernel_core::backend::{Backend, Need};
use kernel_harness::kit::{self, Required, cases::mxfp8_fold as cases};
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::{admission::SUPPORTED_RANKS, geometry::FLASH};
use serde_json::json;
use std::collections::BTreeMap;

fn required() -> (Sm120, Vec<Required<'static>>) {
    let mut backend = Sm120::new();
    let mut required = Vec::new();
    for entry in registration::ticket_catalog(&FLASH)
        .unwrap()
        .into_iter()
        .filter(|e| !e.paired)
    {
        let q = entry.query;
        backend = backend.with_mxfp8_fold_ticket(entry);
        required.push(
            Required::new(
                format!("ticket {q:?}"),
                cases::need(64),
                move |backend, need| backend.mxfp8_fold_gemm(&q, need),
                move |op| cases::case(q, 64, op),
            )
            .required(),
        );
    }
    (backend, required)
}

#[test]
fn ticket_fold_dry_run() {
    let mut inventory = modules::check().unwrap();
    inventory["test"] = json!("mxfp8_fold_ticket::ticket_fold_gpu");
    println!("{inventory}");
    modules::timing_cubins()
        .unwrap()
        .bytes("probe_sm120")
        .unwrap();
    let (backend, required) = required();
    let report = kit::conform(&backend, &super::mxfp8_fold::ModelModules, None, &required).unwrap();
    previous::check(&report, required.len(), false);
    let tested = registration::ticket_catalog(&FLASH).unwrap();
    for &ranks in SUPPORTED_RANKS {
        for enabled in registration::ticket_entries(&FLASH, ranks).unwrap() {
            assert!(tested.iter().any(|e| e.symbol == enabled.symbol
                && e.query == enabled.query
                && e.paired == enabled.paired));
        }
    }
    for entry in &tested {
        let backend = Sm120::new().with_mxfp8_fold_ticket(entry.clone());
        for rows in 1..=64 {
            let info = if entry.paired {
                backend
                    .mxfp8_fold_gemm_pair(&entry.query, Need::rows(rows))
                    .unwrap()
                    .info()
                    .clone()
            } else {
                backend
                    .mxfp8_fold_gemm(&entry.query, Need::rows(rows))
                    .unwrap()
                    .info()
                    .clone()
            };
            assert_eq!(info.workspace, entry.workspace(rows).unwrap());
            assert_eq!(info.symbols.len(), 2);
            entry.plans(rows).unwrap();
        }
    }
    println!(
        "validated {} entries at every row count 1..64 across TP1/TP2/TP4; {} timed comparisons per GPU",
        tested.len(),
        tested.len() * 64
    );
}

#[test]
fn ticket_fold_gpu() {
    let cubins = modules::timing_cubins().unwrap();
    kernel_harness::gpu_test_multi_with(cubins, 3 << 30, |sessions| {
        let (backend, required) = required();
        let report = kit::conform(
            &backend, &super::mxfp8_fold::ModelModules, Some(&sessions[0]), &required,
        )?;
        previous::check(&report, required.len(), true);
        println!("{}", json!({"phase":"conformance-complete","cases":required.len(),"rows":64}));
        let old = registration::decode_catalog(&FLASH)?;
        let mut regressions = Vec::new();
        for (device, session) in sessions.iter().enumerate() {
            let mut records = Vec::new();
            println!("N K G paired rows baseline_us ticket_us floor_us floor_ratio paired_ratio");
            for entry in registration::ticket_catalog(&FLASH)? {
                let query = kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery {
                    m_capacity: 32, ..entry.query
                };
                let baseline = old.iter().find(|e| e.paired == entry.paired && e.query == query)
                    .ok_or("missing accepted split-fold baseline")?;
                let weights = previous::Weights::new(session, baseline)?;
                for rows in 1..=64 {
                    let mut record = previous::compare_with(
                        session, baseline, rows, Some(&entry), Some(&weights), true,
                        device == 0 && rows == 1,
                    )?;
                    record["device"] = json!(device);
                    println!("{} {} {} {} {} {:.3} {:.3} {:.3} {:.3} {:.5}",
                        entry.query.n, entry.query.k, entry.query.groups, entry.paired, rows,
                        record["baseline_us"].as_f64().unwrap(),
                        record["decode_us"].as_f64().unwrap(),
                        record["floor_us"].as_f64().unwrap(),
                        record["floor_ratio"].as_f64().unwrap(),
                        record["paired_ratio"].as_f64().unwrap());
                    println!("{}", record);
                    if record["no_regression"] != true {
                        regressions.push(json!({"device":device,"case":record}));
                    }
                    records.push(record);
                }
                if !entry.paired {
                    let mut record = previous::compare_with(
                        session, baseline, 128, None, Some(&weights), true, false,
                    )?;
                    record["device"] = json!(device);
                    record["phase"] = json!("preserved-prefill");
                    println!("{}", record);
                    records.push(record);
                }
            }
            session.receipt("mxfp8-fold-ticket", BTreeMap::from([
                ("comparisons".into(), json!(records)),
                ("conformance".into(), report.json()),
                ("power_cap_watts".into(), json!(250)),
                ("bandwidth_bytes_per_second".into(), json!(1_600_000_000_000_u64)),
                ("stage_profile_note".into(), json!("Per-stage predecessors run outside events; stage times need not sum to the graph time.")),
            ]))?;
        }
        if !regressions.is_empty() {
            return Err(format!("ticket regressions: {}", json!(regressions)));
        }
        Ok(())
    }).unwrap();
}
