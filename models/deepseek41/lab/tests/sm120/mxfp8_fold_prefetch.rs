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
    let mut backend = Sm120::new().with_packed_fold_weights();
    let mut required = Vec::new();
    for entry in registration::prefetch_catalog(&FLASH)
        .unwrap()
        .into_iter()
        .filter(|e| !e.paired)
    {
        let q = entry.query;
        backend = backend.with_mxfp8_fold_ticket(entry);
        required.push(
            Required::new(
                format!("prefetch {q:?}"),
                cases::need(64),
                move |backend, need| backend.mxfp8_fold_gemm(&q, need),
                move |op| cases::packed_case(q, 64, op),
            )
            .required(),
        );
    }
    (backend, required)
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_DEEPSEEK41_FOLD_CUBINS"]
fn prefetch_fold_dry_run() {
    let mut inventory = modules::check().unwrap();
    inventory["test"] = json!("mxfp8_fold_prefetch::prefetch_fold_gpu");
    println!("{inventory}");
    modules::timing_cubins()
        .unwrap()
        .bytes("probe_sm120")
        .unwrap();
    let (backend, required) = required();
    let report = kit::conform(&backend, &super::mxfp8_fold::ModelModules, None, &required).unwrap();
    previous::check(&report, required.len(), false);
    let tested = registration::prefetch_catalog(&FLASH).unwrap();
    for &ranks in SUPPORTED_RANKS {
        for enabled in registration::prefetch_entries(&FLASH, ranks).unwrap() {
            assert!(tested.iter().any(|e| e.symbol == enabled.symbol
                && e.query == enabled.query
                && e.paired == enabled.paired));
        }
    }
    let mut retained = Vec::new();
    for entry in &tested {
        let baseline = registration::decode_catalog(&FLASH)
            .unwrap()
            .into_iter()
            .find(|old| {
                old.paired == entry.paired
                    && old.query
                        == kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery {
                            m_capacity: 32,
                            ..entry.query
                        }
            })
            .unwrap();
        let backend = Sm120::new()
            .with_packed_fold_weights()
            .with_mxfp8_fold_ticket(entry.clone())
            .with_mxfp8_fold_decode(baseline.clone());
        for rows in 1..=64 {
            let query = registration::small_query(&FLASH, entry.query, rows);
            let info = if entry.paired {
                backend
                    .mxfp8_fold_gemm_pair(&query, Need::rows(rows))
                    .unwrap()
                    .info()
                    .clone()
            } else {
                backend
                    .mxfp8_fold_gemm(&query, Need::rows(rows))
                    .unwrap()
                    .info()
                    .clone()
            };
            if query.m_capacity == 32 {
                retained.push(json!({"n":query.n,"k":query.k,"groups":query.groups,"paired":entry.paired,"rows":rows}));
                assert_eq!(info.workspace, baseline.workspace(rows).unwrap());
                assert_eq!(
                    info.symbols
                        .iter()
                        .map(|(s, _)| s.clone())
                        .collect::<Vec<_>>(),
                    {
                        let mut symbols = baseline.symbols();
                        symbols[1] = kernels_sm120::gemm::mxfp8_fold_packed::symbol(&symbols[1]);
                        symbols
                    }
                );
                baseline.plans(rows).unwrap();
            } else {
                assert_eq!(info.workspace, entry.workspace(rows).unwrap());
                assert_eq!(info.symbols.len(), 2);
                entry.plans(rows).unwrap();
            }
        }
    }
    println!("{}", json!({"retained_baseline_cases":retained}));
    println!(
        "validated {} entries at every row count 1..64 across TP1/TP2/TP4; {} timed comparisons per GPU",
        tested.len(),
        tested.len() * 64
    );
}

fn compare_device(session: &kernel_harness::Session) -> kernel_harness::LabResult<()> {
    let (backend, required) = required();
    let report = kit::conform(
        &backend,
        &super::mxfp8_fold::ModelModules,
        Some(session),
        &required,
    )?;
    previous::check(&report, required.len(), true);
    println!(
        "{}",
        json!({"phase":"conformance-complete","cases":required.len(),"rows":64})
    );
    let old = registration::decode_catalog(&FLASH)?;
    let mut regressions = Vec::new();
    let device = session.ordinal;
    let mut records = Vec::new();
    println!("N K G paired rows baseline_us ticket_us floor_us floor_ratio paired_ratio");
    for entry in registration::prefetch_catalog(&FLASH)? {
        let query = kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery {
            m_capacity: 32,
            ..entry.query
        };
        let baseline = old
            .iter()
            .find(|e| e.paired == entry.paired && e.query == query)
            .ok_or("missing accepted split-fold baseline")?;
        let weights = previous::Weights::new(session, baseline)?;
        for rows in 1..=64 {
            let mut record = previous::compare_with(
                session,
                baseline,
                rows,
                Some(&entry),
                Some(&weights),
                true,
                device == 0 && rows == 1,
            )?;
            record["device"] = json!(device);
            println!(
                "{} {} {} {} {} {:.3} {:.3} {:.3} {:.3} {:.5}",
                entry.query.n,
                entry.query.k,
                entry.query.groups,
                entry.paired,
                rows,
                record["baseline_us"].as_f64().unwrap(),
                record["decode_us"].as_f64().unwrap(),
                record["floor_us"].as_f64().unwrap(),
                record["floor_ratio"].as_f64().unwrap(),
                record["paired_ratio"].as_f64().unwrap()
            );
            println!("{}", record);
            if record["no_regression"] != true {
                regressions.push(json!({"device":device,"case":record}));
            }
            records.push(record);
        }
        if !entry.paired {
            let mut record =
                previous::compare_with(session, baseline, 128, None, Some(&weights), true, false)?;
            record["device"] = json!(device);
            record["phase"] = json!("preserved-prefill");
            println!("{}", record);
            records.push(record);
        }
    }
    session.receipt("mxfp8-fold-prefetch", BTreeMap::from([
                ("comparisons".into(), json!(records)),
                ("conformance".into(), report.json()),
                ("power_cap_watts".into(), json!(250)),
                ("bandwidth_bytes_per_second".into(), json!(1_600_000_000_000_u64)),
                ("stage_profile_note".into(), json!("Per-stage predecessors run outside events; stage times need not sum to the graph time.")),
            ]))?;
    if !regressions.is_empty() {
        return Err(format!("prefetch regressions: {}", json!(regressions)));
    }
    Ok(())
}

/// The pair job runs one process per GPU so both full matrices fit its deadline.
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_DEEPSEEK41_FOLD_CUBINS"]
fn prefetch_fold_gpu() {
    if let Some(ordinals) = kernel_harness::selected_ordinals().unwrap() {
        assert_eq!(ordinals.len(), 1, "select one GPU per comparison process");
    }
    let cubins = modules::timing_cubins().unwrap();
    kernel_harness::gpu_test_with(cubins, 3 << 30, compare_device).unwrap();
}
