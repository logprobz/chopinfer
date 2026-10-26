//! Every served decode fold and row count against the template instance.
use super::packed_serving::{self, family::Entry as Old};
#[path = "../../../../../crates/kernel-lab/families/tests/support/dense_decode.rs"]
mod support;
use bundle_deepseek41::sm120::{mxfp8_fold as catalog, projection_reuse};
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
    packed_serving::family::cases()
        .into_iter()
        .filter_map(|case| {
            let old = case.entry;
            let q = old.base().query;
            let knobs = match &old {
                Old::Ticket(e) => Knobs {
                    tile_n: e.tile_n(),
                    row_groups: 1,
                    stages: 3,
                    weight_prefetch: e.prefetch_weights,
                    finish: Finish::Ticket,
                },
                Old::Decode(_) => Knobs::SPLIT,
                Old::Reuse(e) => Knobs {
                    tile_n: e.tile_n,
                    row_groups: 2,
                    stages: 3,
                    weight_prefetch: false,
                    finish: Finish::Ticket,
                },
                _ => return None,
            };
            let rows = (1..=q.m_capacity)
                .filter(|&rows| match &old {
                    Old::Ticket(e) => {
                        catalog::decode_path(&FLASH, q, rows, e.paired)
                            == if e.prefetch_weights {
                                catalog::DecodePath::Prefetch
                            } else {
                                catalog::DecodePath::Ticket
                            }
                    }
                    Old::Decode(e) => {
                        catalog::decode_path(&FLASH, q, rows, e.paired)
                            == catalog::DecodePath::Split
                    }
                    Old::Reuse(e) => {
                        projection_reuse::choice(q, rows).is_some_and(|(tile, _)| tile == e.tile_n)
                    }
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();
            // Catalog entries outside the selected rows remain compiled references.
            if rows.is_empty() {
                return None;
            }
            let new = Entry::new(
                q,
                &format!(
                    "chop_dense_decode_n{}_k{}_g{}_{}",
                    q.n,
                    q.k,
                    q.groups,
                    if old.paired() { "pair" } else { "single" }
                ),
                old.paired(),
                knobs,
            )
            .unwrap();
            Some(Case { old, new, rows })
        })
        .collect()
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
    (vec![spec, support::producer_spec()], pairs)
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
                for (call, spec) in new.iter_mut().zip(&new_specs) {
                    call.symbol = spec.symbol.clone();
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
