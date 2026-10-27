//! Every served decode fold and row count against the template instance.
use super::packed_serving::{self, family::Entry as Old};
#[path = "../../../../../crates/kernel-lab/families/tests/support/dense_decode.rs"]
mod support;
use crate::dense_decode_reference as catalog;
use bundle_deepseek41::sm120::projection_reuse;
use kernel_build::{CubinSpec, EntrySpec};
use kernels_sm120::gemm::dense_decode::{self, Entry, Finish, Knobs};
use model_deepseek41_contract::geometry::FLASH;
use std::collections::{BTreeMap, BTreeSet};
const CUBIN: &str = "dense_decode_serving_twins";
const DIRECTORY: &str = "CHOPINFER_DENSE_DECODE_TWINS";

struct Case {
    old: Old,
    new: Entry,
    rows: Vec<u32>,
}
fn cases() -> Vec<Case> {
    let mut result = Vec::new();
    for case in packed_serving::family::cases() {
        let old = case.entry;
        let q = old.base().query;
        if !matches!(&old, Old::Ticket(_) | Old::Decode(_) | Old::Reuse(_)) {
            continue;
        }
        let mut schedules = BTreeMap::<Knobs, Vec<u32>>::new();
        for rows in 1..=q.m_capacity {
            let knobs = match &old {
                Old::Reuse(e) => {
                    let Some(row) = catalog::tuning_row(
                        &FLASH,
                        model_deepseek41_contract::tuning::Family::Reuse,
                        "reuse.fold.decode",
                        q,
                        false,
                        rows,
                    ) else {
                        continue;
                    };
                    let (knobs, folds) = Knobs::from_row(row).unwrap();
                    assert!(folds.is_some());
                    if knobs.tile_n != e.tile_n {
                        continue;
                    }
                    knobs
                }
                Old::Ticket(e) => {
                    let knobs = catalog::dense_knobs(&FLASH, q, rows, e.paired).unwrap();
                    if knobs.finish != Finish::Ticket || knobs.weight_prefetch != e.prefetch_weights
                    {
                        continue;
                    }
                    knobs
                }
                Old::Decode(e) => {
                    let knobs = catalog::dense_knobs(&FLASH, q, rows, e.paired).unwrap();
                    if knobs.finish != Finish::Merge {
                        continue;
                    }
                    knobs
                }
                _ => unreachable!(),
            };
            schedules.entry(knobs).or_default().push(rows);
        }
        for (knobs, rows) in schedules {
            let new = catalog::dense_entry(q, old.paired(), knobs).unwrap();
            result.push(Case {
                old: old.clone(),
                new,
                rows,
            });
        }
    }
    result
}

fn specs() -> (Vec<CubinSpec>, Vec<serde_json::Value>) {
    let mut entries = BTreeMap::<String, EntrySpec>::new();
    let mut pairs = Vec::new();
    for case in cases() {
        let mut old = case.old.specs();
        old[1] = case.old.packed_specs().remove(0);
        let new = case.new.packed_specs().unwrap();
        for (a, b) in old.into_iter().zip(new) {
            pairs.push(serde_json::json!({"old":a.symbol,"new":b.symbol,"rows":case.rows,"family":case.old.family()}));
            for entry in [a, b] {
                if let Some(previous) = entries.insert(entry.symbol.clone(), entry.clone()) {
                    assert_eq!(previous.body, entry.body);
                }
            }
        }
    }
    let mut spec = dense_decode::cubin_spec(CUBIN, &[cases().remove(0).new], true).unwrap();
    spec.headers.extend(
        [
            "mxfp8_fold_ticket",
            "mxfp8_fold_prefetch",
            "mxfp8_fold_decode",
            "mxfp8_fold_reuse",
        ]
        .map(|h| format!("#include \"gemm/{h}.cuh\"")),
    );
    spec.entries = entries.into_values().collect();
    (
        vec![
            crate::dense_decode_reference::frozen::configure(spec),
            support::producer_spec(),
        ],
        pairs,
    )
}

#[test]
fn served_twins_preserve_every_selected_grid_and_abi() {
    let cases = cases();
    let families = cases
        .iter()
        .map(|c| c.old.family())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        families,
        BTreeSet::from(["ticket", "prefetch", "decode", "reuse"])
    );
    let mut count = 0;
    for case in &cases {
        let new = case.new.packed_specs().unwrap();
        let mut old = case.old.specs();
        old[1] = case.old.packed_specs().remove(0);
        for (a, b) in old.iter().zip(&new) {
            assert_eq!(a.params, b.params);
            assert_eq!(a.max_threads, b.max_threads);
            assert_eq!(a.min_blocks, b.min_blocks);
        }
        for &rows in &case.rows {
            let ([_, calls], workspace) = packed_serving::launches(&case.old, rows).unwrap();
            let folds = if let Old::Reuse(_) = case.old {
                projection_reuse::choice(case.new.query, rows).unwrap().1
            } else {
                case.new.folds_per_cta(rows)
            };
            assert_eq!(workspace, case.new.workspace(rows).unwrap());
            let plans = case.new.plans(rows, folds).unwrap();
            assert_eq!(calls.len(), plans.len());
            for (call, plan) in calls.iter().zip(plans) {
                assert_eq!(call.plan, plan);
            }
            count += 1;
        }
    }
    let mut table_rows = 0;
    for family in [
        model_deepseek41_contract::tuning::Family::DenseFold,
        model_deepseek41_contract::tuning::Family::Reuse,
    ] {
        for row in model_deepseek41_contract::tuning::table(&FLASH, family)
            .unwrap()
            .rows()
            .iter()
            .filter(|r| r.schedule == "dense_decode")
        {
            let (knobs, _) = Knobs::from_row(row).unwrap();
            for rows in row.key.rows.unwrap_or([1, row.key.bucket])[0]
                ..=row.key.rows.unwrap_or([1, row.key.bucket])[1]
            {
                assert!(
                    cases.iter().any(|case| {
                        let q = case.new.query;
                        case.new.knobs == knobs
                            && case.rows.contains(&rows)
                            && row.key.shape.get("n") == Some(&q.n)
                            && row.key.shape.get("k") == Some(&q.k)
                            && row.key.shape.get("groups") == Some(&q.groups)
                            && row.key.shape.get("paired") == Some(&u32::from(case.new.paired))
                    }),
                    "uncovered dense decode row {} at {rows}",
                    row.key
                );
            }
            table_rows += 1;
        }
    }
    assert_eq!(table_rows, 72);
    assert!(count > 0);
    let (specs, pairs) = specs();
    for spec in &specs {
        kernel_build::generate(spec).unwrap();
    }
    println!(
        "dense_decode coverage entries={} symbols={} rows={count}",
        cases.len(),
        pairs.len()
    );
}

#[test]
#[ignore = "requires an external directory named by CHOPINFER_DENSE_DECODE_TWINS"]
fn build_served_twins() {
    let directory = std::path::PathBuf::from(std::env::var_os(DIRECTORY).expect(DIRECTORY));
    let (specs, pairs) = specs();
    support::build(&specs, &directory).unwrap();
    std::fs::write(
        directory.join("pairs.json"),
        serde_json::to_vec_pretty(&pairs).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "requires an external directory named by CHOPINFER_DENSE_DECODE_TWINS"]
fn verify_staged_served_twins() {
    let directory = std::path::PathBuf::from(std::env::var_os(DIRECTORY).expect(DIRECTORY));
    support::verify_staged(&directory).unwrap();
}

#[test]
fn served_twins_match_with_poisoned_replay_on_gpu() {
    kernel_harness::gpu_test(4 << 30, |session| {
        let directory = std::path::PathBuf::from(std::env::var_os(DIRECTORY).ok_or(DIRECTORY)?);
        let module = support::load(session, &directory, CUBIN)?;
        let probe = support::load(session, &directory, "dense_decode_pdl_probe")?;
        let producer = kernel_harness::poison::LiveProducer::with_function(
            session,
            support::producer(&probe)?,
        )?;
        let shard = std::env::var("CHOPINFER_DENSE_TWIN_SHARD")
            .unwrap_or_else(|_| "0".into())
            .parse::<usize>()
            .map_err(|e| e.to_string())?;
        let shards = std::env::var("CHOPINFER_DENSE_TWIN_SHARDS")
            .unwrap_or_else(|_| "1".into())
            .parse::<usize>()
            .map_err(|e| e.to_string())?;
        if shards == 0 || shard >= shards {
            return Err("invalid twin shard".into());
        }
        let cases = cases();
        let selected = cases
            .iter()
            .enumerate()
            .filter(|(i, _)| i % shards == shard)
            .collect::<Vec<_>>();
        let expected: usize = selected.iter().map(|(_, c)| c.rows.len()).sum();
        if expected == 0 {
            return Err("empty twin shard".into());
        }
        let start = std::time::Instant::now();
        let mut count = 0;
        for (index, case) in selected {
            let new_specs = case.new.packed_specs().map_err(|e| e.to_string())?;
            let maximum = *case.rows.last().unwrap();
            let mut images = packed_serving::images(
                &case.old,
                maximum,
                case.new.workspace(maximum).map_err(|e| e.to_string())?,
                index as u64 + 1,
            )?;
            for &rows in case.rows.iter().rev() {
                if start.elapsed().as_secs() >= 240 {
                    return Err("twin shard exceeded four minutes".into());
                }
                let ([_, old], workspace) = packed_serving::launches(&case.old, rows)?;
                let mut new = old.clone();
                let folds = if matches!(case.old, Old::Reuse(_)) {
                    projection_reuse::choice(case.new.query, rows).unwrap().1
                } else {
                    case.new.folds_per_cta(rows)
                };
                for ((call, spec), plan) in new
                    .iter_mut()
                    .zip(&new_specs)
                    .zip(case.new.plans(rows, folds).map_err(|e| e.to_string())?)
                {
                    call.symbol = spec.symbol.clone();
                    call.plan = plan;
                }
                let sizes = case.old.base().bytes(rows);
                for image in &mut images {
                    let bytes = match image.handle {
                        1 => sizes.activations,
                        2 => sizes.activation_scales,
                        5 | 10 => sizes.output,
                        7 => sizes.status + 12,
                        11 => workspace,
                        _ => continue,
                    };
                    image.bytes.truncate(bytes as usize);
                }
                images.iter_mut().find(|i| i.handle == 6).unwrap().bytes =
                    rows.to_le_bytes().to_vec();
                support::compare(
                    session,
                    &module,
                    &producer,
                    [&old, &new],
                    &images,
                    support::ReplayShape {
                        dtype: case.new.query.output,
                        rows,
                        header: case.new.header(rows).map_err(|e| e.to_string())? as usize,
                    },
                )
                .map_err(|e| format!("{} rows={rows}: {e}", case.old.key()))?;
                count += 1;
                println!("DENSE_TWIN_PASS {} rows={rows}", case.old.key());
            }
        }
        if count != expected {
            return Err(format!("executed {count}, expected {expected}"));
        }
        session.receipt(
            "dense-decode-twins",
            BTreeMap::from([
                ("shard".into(), serde_json::json!(shard)),
                ("shards".into(), serde_json::json!(shards)),
                ("executed_cases".into(), serde_json::json!(count)),
                ("expected_cases".into(), serde_json::json!(expected)),
                (
                    "phases".into(),
                    serde_json::json!(["eager", "relaunch", "graph", "pdl"]),
                ),
                ("workspace_bitwise".into(), serde_json::json!(true)),
            ]),
        )?;
        println!("DENSE_TWIN_COMPLETE shard={shard}/{shards} executed={count}");
        Ok(())
    })
    .unwrap();
}
