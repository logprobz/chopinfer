//! Direct ordered-fold qualification against the current serving route.
mod compare;
use super::mxfp8_fold_decode as previous;
use bundle_deepseek41::sm120::{mxfp8_fold as registration, mxfp8_fold_stream as catalog};
use kernel_core::backend::{Backend, Need};
use kernel_harness::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::{admission::SUPPORTED_RANKS, geometry::FLASH};
use serde_json::json;
use std::sync::OnceLock;
#[path = "../../sm120/fold_bandwidth.rs"]
#[allow(dead_code)]
mod family;

pub(super) fn cubin() -> LabResult<(&'static str, &'static [u8])> {
    static BYTES: OnceLock<LabResult<Vec<u8>>> = OnceLock::new();
    let data = BYTES
        .get_or_init(|| {
            let dir = previous::modules::directory()?;
            let bytes = std::fs::read(dir.join(format!("{}.cubin", family::CUBIN)))
                .map_err(|e| e.to_string())?;
            let lock: Vec<serde_json::Value> =
                serde_json::from_str(include_str!("../../kernels.lock.d/mxfp8_fold_direct.lock"))
                    .map_err(|e| e.to_string())?;
            let entry = lock
                .iter()
                .find(|r| r["name"] == family::CUBIN)
                .ok_or("missing direct fold lock")?;
            if entry["cubin_sha256"] != kernel_build::digest(&bytes) {
                return Err("direct fold cubin differs from lock".into());
            }
            for symbol in family::cubin()?.entries.iter().map(|e| &e.symbol) {
                if !entry["entries"]
                    .as_array()
                    .ok_or("missing lock exports")?
                    .iter()
                    .any(|e| e["symbol"] == *symbol)
                {
                    return Err(format!("direct fold lock omits {symbol}"));
                }
            }
            Ok(bytes)
        })
        .as_ref()
        .map_err(Clone::clone)?;
    Ok((family::CUBIN, data))
}

#[test]
fn direct_fold_cpu_matrix() {
    let entries = family::catalog().unwrap();
    let mut shapes = std::collections::BTreeSet::new();
    let mut ranks_by_shape =
        std::collections::BTreeMap::<_, std::collections::BTreeSet<u32>>::new();
    let mut matrix = std::collections::BTreeMap::new();
    for &ranks in SUPPORTED_RANKS {
        let baseline = catalog::register(
            registration::register(Sm120::new(), &FLASH, ranks).unwrap(),
            &FLASH,
            ranks,
        )
        .unwrap();
        for current in registration::ticket_entries(&FLASH, ranks).unwrap() {
            let variants: Vec<_> = entries
                .iter()
                .filter(|e| {
                    e.packed.ticket.query == current.query
                        && e.packed.ticket.paired == current.paired
                })
                .collect();
            assert!(!variants.is_empty());
            for e in variants {
                e.validate().unwrap();
                let q = current.query;
                shapes.insert((q.n, q.k, q.groups, current.paired));
                ranks_by_shape
                    .entry((q.n, q.k, q.groups, current.paired))
                    .or_default()
                    .insert(ranks);
                for rows in family::ROWS {
                    let query = registration::routed_query(q, rows);
                    let stream = catalog::qualified(&FLASH, query, rows, current.paired);
                    let symbols = if stream {
                        baseline
                            .mxfp8_fold_gemm_stream(&query, current.paired, Need::rows(rows))
                            .unwrap()
                            .info()
                            .symbols
                            .clone()
                    } else if current.paired {
                        baseline
                            .mxfp8_fold_gemm_pair(&query, Need::rows(rows))
                            .unwrap()
                            .info()
                            .symbols
                            .clone()
                    } else {
                        baseline
                            .mxfp8_fold_gemm(&query, Need::rows(rows))
                            .unwrap()
                            .info()
                            .symbols
                            .clone()
                    };
                    let plan = e.plan(rows).unwrap();
                    plan.validate_geometry().unwrap();
                    assert_eq!(plan.workspace_bytes, 0);
                    assert!(e.state_bytes(rows).unwrap() >= 1024);
                    assert!(plan.dynamic_shared_bytes <= 101376);
                    let key = (q.n, q.k, q.groups, current.paired, rows, e.schedule.name());
                    let value = json!({"n":q.n,"k":q.k,"groups":q.groups,"paired":current.paired,
                        "rows":rows,"variant":e.schedule.name(),"baseline_stream":stream,
                        "baseline_symbols":symbols.into_iter().map(|(s,_)|s)
                            .filter(|s| !s.ends_with("_prepare_v1")).collect::<Vec<_>>(),
                        "grid":plan.grid,"block":plan.block,"state_bytes":e.state_bytes(rows).unwrap(),
                        "stamp_bytes":e.stamp_bytes(rows).unwrap()});
                    if let Some(prior) = matrix.insert(key, value.clone()) {
                        assert_eq!(prior, value);
                    }
                }
            }
        }
    }
    assert_eq!(shapes.len(), 23);
    assert_eq!(matrix.len(), entries.len() * family::ROWS.len());
    for (&(n, k, g, p, _, _), value) in &mut matrix {
        value["ranks"] = json!(ranks_by_shape[&(n, k, g, p)]);
    }
    println!(
        "fold_bandwidth_cpu {}",
        json!({"cases":matrix.len(),"shapes":shapes.len(),
        "rows":family::ROWS,"topologies":SUPPORTED_RANKS,"required_ordinals":1,
        "matrix":matrix.into_values().collect::<Vec<_>>(),"serving_enabled":false})
    );
}

fn check_worker(ordinals: &[i32]) -> LabResult<()> {
    if ordinals.len() != 1 {
        return Err("direct fold requires exactly one ordinal per worker".into());
    }
    Ok(())
}

#[test]
fn direct_fold_rejects_missing_or_multiple_ordinals() {
    assert!(check_worker(&[]).is_err());
    assert!(check_worker(&[0, 1]).is_err());
    for device in [0, 1] {
        assert!(check_worker(&[device]).is_ok());
    }
}

#[test]
#[ignore = "requires the selected ordinal supplied by a job worker"]
fn direct_fold_worker_dry_run() {
    let ordinals = kernel_harness::selected_ordinals()
        .unwrap()
        .expect("worker ordinal");
    check_worker(&ordinals).unwrap();
    println!(
        "fold_bandwidth_worker {}",
        json!({"required_ordinals":1,"selected_ordinals":ordinals,
        "test":"mxfp8_fold_bandwidth::direct_fold_gpu","gpu_launched":false})
    );
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_DEEPSEEK41_FOLD_CUBINS"]
fn direct_fold_dry_run() {
    println!(
        "fold_bandwidth_modules {}",
        previous::modules::check().unwrap()
    );
    let (old, bytes) = super::mxfp8_fold_stream::cubin().unwrap();
    println!(
        "fold_bandwidth_module {}",
        json!({"name":old,"sha256":kernel_build::digest(bytes)})
    );
    let (name, bytes) = cubin().unwrap();
    println!(
        "fold_bandwidth_module {}",
        json!({"name":name,"sha256":kernel_build::digest(bytes),
        "symbols":family::cubin().unwrap().entries.into_iter().map(|e|e.symbol).collect::<Vec<_>>()})
    );
    let _ = previous::modules::timing_cubins().unwrap();
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_DEEPSEEK41_FOLD_CUBINS"]
fn direct_fold_gpu() {
    if let Some(ordinals) = kernel_harness::selected_ordinals().unwrap() {
        check_worker(&ordinals).unwrap();
    }
    let cubins = previous::modules::timing_cubins().unwrap();
    kernel_harness::gpu_test_with(cubins, 8 << 30, |session| {
        let old = registration::decode_catalog(&FLASH)?;
        let entries = family::catalog()?;
        for reference in &old {
            let variants: Vec<_> = entries
                .iter()
                .filter(|e| {
                    e.packed.ticket.paired == reference.paired
                        && e.packed.ticket.query == registration::routed_query(reference.query, 64)
                })
                .collect();
            let weights = previous::Weights::new(session, reference)?;
            for candidate in variants {
                for rows in family::ROWS {
                    let mut record = compare::case(
                        session,
                        reference,
                        rows,
                        candidate,
                        &weights,
                        true,
                        matches!(rows, 1 | 8 | 64),
                    )?;
                    record["device"] = json!(session.ordinal);
                    println!("fold_bandwidth_result {record}");
                }
            }
            if !reference.paired {
                let mut record = previous::compare_with(
                    session,
                    reference,
                    128,
                    None,
                    Some(&weights),
                    false,
                    false,
                )?;
                record["device"] = json!(session.ordinal);
                println!("fold_bandwidth_prefill {record}");
            }
        }
        let deferred = family::probe(true)?;
        let reference = old
            .iter()
            .find(|e| {
                !e.paired && registration::routed_query(e.query, 64) == deferred.packed.ticket.query
            })
            .ok_or("missing probe baseline")?;
        let weights = previous::Weights::new(session, reference)?;
        for rows in [1, 32] {
            let record =
                compare::case(session, reference, rows, &deferred, &weights, false, false)?;
            println!("fold_bandwidth_deferred {record}");
        }
        let defect = compare::case(
            session,
            reference,
            1,
            &family::probe(false)?,
            &weights,
            false,
            false,
        )
        .expect_err("omitted-chain negative control escaped");
        if !defect.starts_with("bitwise mismatch") {
            return Err(defect);
        }
        println!(
            "fold_bandwidth_control {}",
            json!({"omitted_chain_caught":true})
        );
        session.receipt(
            "mxfp8-fold-bandwidth",
            std::collections::BTreeMap::from([
                ("cases".into(), json!(entries.len() * family::ROWS.len())),
                ("serving_enabled".into(), json!(false)),
            ]),
        )?;
        Ok(())
    })
    .unwrap();
}
