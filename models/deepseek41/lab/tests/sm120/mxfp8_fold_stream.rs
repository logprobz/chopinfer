//! Qualification of one-launch folds against the accepted serving selection.
mod ahead;
mod sublayer;
mod worker;
use super::mxfp8_fold_decode::{self as previous, modules};
use bundle_deepseek41::sm120::mxfp8_fold as registration;
use bundle_deepseek41::sm120::mxfp8_fold_stream as catalog;
use kernel_core::backend::{Backend, Need};
use kernel_harness::LabResult;
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::{admission::SUPPORTED_RANKS, geometry::FLASH};
use serde_json::json;
use std::sync::OnceLock;
const ROWS: [u32; 7] = [1, 2, 4, 6, 8, 16, 32];

fn required_symbols() -> LabResult<Vec<String>> {
    Ok(catalog::catalog(&FLASH)?
        .into_iter()
        .chain([catalog::deferred_entry(&FLASH)?])
        .flat_map(|e| e.symbols())
        .chain([kernels_sm120::gemm::mxfp8_fold_stream::PREFETCH.into()])
        .collect())
}

pub(super) fn cubin() -> LabResult<(&'static str, &'static [u8])> {
    static BYTES: OnceLock<LabResult<Vec<u8>>> = OnceLock::new();
    let data = BYTES
        .get_or_init(|| {
            let name = catalog::cubin_name(&FLASH);
            let dir = std::env::var_os("CHOPINFER_DEEPSEEK41_FOLD_CUBINS")
                .ok_or("fold cubin directory is required")?;
            let data = std::fs::read(std::path::PathBuf::from(dir).join(format!("{name}.cubin")))
                .map_err(|e| e.to_string())?;
            let lock: Vec<serde_json::Value> = serde_json::from_str(include_str!(
                "../../../bundle/kernels.lock.d/mxfp8_fold_stream.lock"
            ))
            .map_err(|e| e.to_string())?;
            let expected = lock
                .iter()
                .find(|r| r["name"] == name)
                .ok_or("missing packed fold lock")?;
            if expected["cubin_sha256"] != kernel_build::digest(&data) {
                return Err("packed fold cubin differs from its lock".into());
            }
            for symbol in required_symbols()? {
                if !expected["entries"]
                    .as_array()
                    .ok_or("missing packed entries")?
                    .iter()
                    .any(|r| r["symbol"] == symbol)
                {
                    return Err(format!("packed lock omits {symbol}"));
                }
            }
            Ok(data)
        })
        .as_ref()
        .map_err(Clone::clone)?;
    Ok(("mxfp8_fold_stream_deepseek41_sm120_v1", data))
}

#[test]
fn stream_fold_dry_run() {
    println!("{}", modules::check().unwrap());
    let (name, bytes) = cubin().unwrap();
    println!("{}", sublayer::check_module().unwrap());
    let entries = catalog::catalog(&FLASH).unwrap();
    println!(
        "{}",
        json!({"module":name,"sha256":kernel_build::digest(bytes),"symbols":required_symbols().unwrap()})
    );
    let mut covered = std::collections::BTreeSet::new();
    for &ranks in SUPPORTED_RANKS {
        let baseline = registration::register(Sm120::new(), &FLASH, ranks).unwrap();
        for old in registration::ticket_entries(&FLASH, ranks).unwrap() {
            let candidate = entries
                .iter()
                .find(|e| e.ticket.query == old.query && e.ticket.paired == old.paired)
                .unwrap();
            let backend = Sm120::new().with_mxfp8_fold_stream(candidate.clone());
            for rows in ROWS {
                let q = registration::routed_query(old.query, rows);
                let info = if old.paired {
                    baseline
                        .mxfp8_fold_gemm_pair(&q, Need::rows(rows))
                        .unwrap()
                        .info()
                        .clone()
                } else {
                    baseline
                        .mxfp8_fold_gemm(&q, Need::rows(rows))
                        .unwrap()
                        .info()
                        .clone()
                };
                assert!(info.symbols.len() == 2 || info.symbols.len() == 3);
                let op = backend
                    .mxfp8_fold_gemm_stream(&old.query, old.paired, Need::rows(rows))
                    .unwrap();
                assert_eq!(
                    op.info().bytes_of("state"),
                    Some(candidate.state_bytes(rows).unwrap())
                );
                covered.insert((old.query.n, old.query.k, old.query.groups, old.paired, rows));
            }
        }
    }
    assert_eq!(covered.len(), 23 * ROWS.len());
    println!(
        "{}",
        json!({"matrix_cases":covered.len(),"matrix":covered,"rows":ROWS,"topologies":SUPPORTED_RANKS,"serving_candidate_buckets":0})
    );
}

#[test]
fn stream_fold_gpu() {
    let cubins = modules::timing_cubins().unwrap();
    worker::run(worker::FOLD, cubins, |session| {
        let old = registration::decode_catalog(&FLASH)?;
        let candidates = catalog::catalog(&FLASH)?;
        for candidate in &candidates {
            let entry = old
                .iter()
                .find(|e| {
                    e.paired == candidate.ticket.paired
                        && registration::routed_query(e.query, 64) == candidate.ticket.query
                })
                .ok_or("missing fold baseline")?;
            let weights = previous::Weights::new(session, entry)?;
            for rows in ROWS {
                let mut record =
                    previous::compare_stream(session, entry, rows, candidate, &weights, rows == 1)?;
                record["device"] = json!(session.ordinal);
                record["phase"] = json!("stream-comparison");
                record["packed_weight_bytes"] =
                    json!(candidate.packed_bytes() * if entry.paired { 2 } else { 1 });
                record["state_bytes"] =
                    json!(candidate.state_bytes(rows).map_err(|e| e.to_string())?);
                record["trace_candidate_replays"] = json!(15);
                record["trace_baseline_replays"] = json!(7);
                println!("{record}");
            }
            if !entry.paired {
                let mut record = previous::compare_with(
                    session,
                    entry,
                    128,
                    None,
                    Some(&weights),
                    false,
                    false,
                )?;
                record["phase"] = json!("preserved-prefill");
                record["device"] = json!(session.ordinal);
                println!("{record}");
            }
        }
        let deferred = catalog::deferred_entry(&FLASH)?;
        let baseline = old
            .iter()
            .find(|e| !e.paired && registration::routed_query(e.query, 64) == deferred.ticket.query)
            .ok_or("missing deferred baseline")?;
        let weights = previous::Weights::new(session, baseline)?;
        for rows in [1, 32] {
            let mut record =
                previous::compare_stream_deferred(session, baseline, rows, &deferred, &weights)?;
            record["phase"] = json!("deferred-admission");
            record["device"] = json!(session.ordinal);
            println!("{record}");
        }
        session.receipt(
            "mxfp8-fold-stream",
            std::collections::BTreeMap::from([("matrix_cases".into(), json!(23 * ROWS.len()))]),
        )?;
        Ok(())
    })
    .unwrap();
}
