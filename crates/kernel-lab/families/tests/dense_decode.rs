//! Template twins cover padded groups, half folds and both output formats.
use kernel_lab_families::dense_decode_reference::ReferenceBackend as Sm120;
#[path = "support/packed_row_readers.rs"]
#[allow(dead_code, reason = "the shared fixtures also serve timing tests")]
mod fixtures;
#[path = "support/dense_decode.rs"]
mod support;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_lab_families::{
    mxfp8_fold_decode_packed as split, mxfp8_fold_reuse_packed as reuse,
    mxfp8_fold_ticket_packed as ticket,
};
use kernels_sm120::gemm::{
    dense_decode::{self, Entry, Finish, Knobs},
    mxfp8_fold_packed,
};
use std::collections::BTreeMap;
const CUBIN: &str = "dense_decode_synthetic_twins";
const DIRECTORY: &str = "CHOPINFER_DENSE_DECODE_SYNTHETIC_TWINS";
#[derive(Clone)]
enum Old {
    Split(kernel_lab_families::dense_decode_reference::mxfp8_fold_decode::Entry),
    Ticket(kernel_lab_families::dense_decode_reference::mxfp8_fold_ticket::Entry),
    Reuse(
        kernel_lab_families::dense_decode_reference::mxfp8_fold_reuse::Entry,
        u32,
    ),
}
struct Case {
    old: Old,
    new: Entry,
}
impl Case {
    fn specs(&self) -> Vec<EntrySpec> {
        let mut specs = match &self.old {
            Old::Split(e) => e.specs(),
            Old::Ticket(e) => e.specs(),
            Old::Reuse(e, _) => e.specs(),
        };
        specs[1] = match &self.old {
            Old::Split(e) => e.packed_spec().unwrap(),
            Old::Ticket(e) => e.packed_spec().unwrap(),
            Old::Reuse(e, _) => e.packed_spec().unwrap(),
        };
        specs
    }
    fn rows(&self) -> [u32; 4] {
        if matches!(self.old, Old::Reuse(..)) {
            [6, 8, 17, 32]
        } else {
            [1, 7, 17, self.new.query.m_capacity]
        }
    }
    fn folds(&self, rows: u32) -> u32 {
        if let Old::Reuse(_, folds) = self.old {
            folds
        } else {
            self.new.folds_per_cta(rows)
        }
    }
    fn calls(&self, rows: u32) -> Vec<Launch> {
        let backend = match &self.old {
            Old::Split(e) => Sm120::new().with_mxfp8_fold_decode(e.clone()),
            Old::Ticket(e) => Sm120::new().with_mxfp8_fold_ticket(e.clone()),
            Old::Reuse(e, _) => Sm120::new().with_mxfp8_fold_ticket(e.ticket.clone()),
        };
        let q = self.new.query;
        let buffers = fixtures::buffers(q, rows);
        let work = View {
            handle: 11,
            bytes: match &self.old {
                Old::Reuse(e, _) => e.ticket.workspace(rows).unwrap(),
                _ => self.new.workspace(rows).unwrap(),
            },
        };
        let mut calls = if self.new.paired {
            backend
                .mxfp8_fold_gemm_pair(&q, Need::rows(rows))
                .unwrap()
                .launches(&buffers, work, Counts::rows(rows))
                .unwrap()
        } else {
            backend
                .mxfp8_fold_gemm(&q, Need::rows(rows))
                .unwrap()
                .launches(&buffers[0], work, Counts::rows(rows))
                .unwrap()
        };
        if let Old::Reuse(entry, folds) = &self.old {
            for ((call, spec), plan) in calls
                .iter_mut()
                .zip(entry.specs())
                .zip(entry.plans(rows, *folds).unwrap())
            {
                call.symbol = spec.symbol;
                call.plan = plan;
            }
            calls[1].arguments.0[11] = kernel_core::contract::Argument::U32(*folds);
        }
        let symbol = match &self.old {
            Old::Split(e) => e.specs()[1].symbol.clone(),
            Old::Ticket(e) => e.specs()[1].symbol.clone(),
            Old::Reuse(e, _) => e.specs()[1].symbol.clone(),
        };
        mxfp8_fold_packed::launches(calls, |s| s == symbol)
    }
}
fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for old in split::entries() {
        let new = Entry::new(
            old.query,
            &format!("{}_template", old.symbol),
            old.paired,
            Knobs::SPLIT,
        )
        .unwrap();
        cases.push(Case {
            old: Old::Split(old),
            new,
        });
    }
    for old in ticket::entries() {
        for prefetch in [false, true] {
            let mut old = old.clone();
            if prefetch {
                old.symbol = format!("{}_prefetch_v1", old.symbol);
                old = old.with_weight_prefetch();
            }
            let knobs = Knobs {
                tile_n: old.tile_n(),
                row_groups: 1,
                stages: 3,
                weight_prefetch: prefetch,
                finish: Finish::Ticket,
            };
            let new = Entry::new(
                old.query,
                &format!("{}_template", old.symbol),
                old.paired,
                knobs,
            )
            .unwrap();
            cases.push(Case {
                old: Old::Ticket(old),
                new,
            });
        }
    }
    for old in reuse::entries() {
        for folds in [2, 4, 8] {
            let new = Entry::new(
                old.ticket.query,
                &format!("{}_template", old.symbol()),
                old.ticket.paired,
                Knobs {
                    tile_n: old.tile_n,
                    row_groups: 2,
                    ..Knobs::ticket(old.ticket.query.n)
                },
            )
            .unwrap();
            cases.push(Case {
                old: Old::Reuse(old.clone(), folds),
                new,
            });
        }
    }
    cases
}
fn specs() -> Vec<CubinSpec> {
    let cases = cases();
    let mut entries = BTreeMap::new();
    for case in &cases {
        for spec in case
            .specs()
            .into_iter()
            .chain(case.new.packed_specs().unwrap())
        {
            entries.insert(spec.symbol.clone(), spec);
        }
    }
    let mut spec = dense_decode::cubin_spec(CUBIN, &[cases[0].new.clone()], true).unwrap();
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
    vec![
        kernel_lab_families::dense_decode_reference::configure(spec),
        support::producer_spec(),
    ]
}
#[test]
fn synthetic_twins_preserve_launch_abis_and_grids() {
    let cases = cases();
    assert!(!cases.is_empty());
    for case in &cases {
        for (old, new) in case
            .specs()
            .into_iter()
            .zip(case.new.packed_specs().unwrap())
        {
            assert_eq!(old.params, new.params);
            assert_eq!(old.max_threads, new.max_threads);
            assert_eq!(old.min_blocks, new.min_blocks);
        }
        for rows in case.rows() {
            let plans = case.new.plans(rows, case.folds(rows)).unwrap();
            for (call, plan) in case.calls(rows).iter().zip(plans) {
                assert_eq!(call.plan, plan);
            }
        }
    }
    for spec in specs() {
        kernel_build::generate(&spec).unwrap();
    }
    println!("dense_decode synthetic entries={}", cases.len());
}
#[test]
#[ignore = "requires an external directory named by CHOPINFER_DENSE_DECODE_SYNTHETIC_TWINS"]
fn build_synthetic_twins() {
    let directory = std::path::PathBuf::from(std::env::var_os(DIRECTORY).expect(DIRECTORY));
    support::build(&specs(), &directory).unwrap();
}
#[test]
#[ignore = "requires an external directory named by CHOPINFER_DENSE_DECODE_SYNTHETIC_TWINS"]
fn verify_staged_synthetic_twins() {
    let directory = std::path::PathBuf::from(std::env::var_os(DIRECTORY).expect(DIRECTORY));
    support::verify_staged(&directory).unwrap();
}
#[test]
fn synthetic_twins_match_poisoned_replay_on_gpu() {
    kernel_harness::gpu_test(512 << 20, |session| {
        let directory = std::path::PathBuf::from(std::env::var_os(DIRECTORY).ok_or(DIRECTORY)?);
        let module = support::load(session, &directory, CUBIN)?;
        let probe = support::load(session, &directory, "dense_decode_pdl_probe")?;
        let producer = kernel_harness::poison::LiveProducer::with_function(
            session,
            support::producer(&probe)?,
        )?;
        let cases = cases();
        let expected: usize = cases
            .iter()
            .map(|case| {
                case.rows()
                    .into_iter()
                    .map(|rows| kernel_harness::poison::live_counts(rows).len())
                    .sum::<usize>()
            })
            .sum();
        let mut count = 0;
        for (index, case) in cases.iter().enumerate() {
            for rows in case.rows() {
                let old = case.calls(rows);
                let mut new = old.clone();
                for (call, spec) in new.iter_mut().zip(case.new.packed_specs().unwrap()) {
                    call.symbol = spec.symbol;
                }
                let workspace = case.new.workspace(rows).unwrap();
                let images = fixtures::images(
                    case.new.query,
                    rows,
                    case.new.paired,
                    workspace,
                    index as u64 + 1,
                )?;
                let mut images = images;
                for live in kernel_harness::poison::live_counts(rows) {
                    images.iter_mut().find(|i| i.handle == 6).unwrap().bytes =
                        live.to_le_bytes().to_vec();
                    support::compare(
                        session,
                        &module,
                        &producer,
                        [&old, &new],
                        &images,
                        support::ReplayShape {
                            dtype: case.new.query.output,
                            rows,
                            header: case.new.header(rows).unwrap() as usize,
                        },
                    )?;
                    count += 1;
                    println!(
                        "DENSE_SYNTHETIC_PASS {} rows={rows} live={live} folds={}",
                        case.new.symbol,
                        case.folds(rows)
                    );
                }
            }
        }
        if count == 0 || count != expected {
            return Err(format!("executed {count}, expected {expected}"));
        }
        session.receipt(
            "dense-decode-synthetic",
            BTreeMap::from([
                ("executed_cases".into(), serde_json::json!(count)),
                ("expected_cases".into(), serde_json::json!(expected)),
                (
                    "phases".into(),
                    serde_json::json!(["eager", "relaunch", "graph", "pdl"]),
                ),
                ("workspace_bitwise".into(), serde_json::json!(true)),
                ("partial_live_rows".into(), serde_json::json!(true)),
            ]),
        )?;
        println!("DENSE_SYNTHETIC_COMPLETE executed={count}");
        Ok(())
    })
    .unwrap();
}

#[test]
fn poisoned_twins_reject_missing_writes_on_gpu() {
    kernel_harness::gpu_test(512 << 20, |session| {
        let directory = std::path::PathBuf::from(std::env::var_os(DIRECTORY).ok_or(DIRECTORY)?);
        let module = support::load(session, &directory, CUBIN)?;
        let probe = support::load(session, &directory, "dense_decode_pdl_probe")?;
        let producer = kernel_harness::poison::LiveProducer::with_function(
            session,
            support::producer(&probe)?,
        )?;
        let case = cases().remove(0);
        let rows = 1;
        let calls = case.calls(rows);
        let images = fixtures::images(
            case.new.query,
            rows,
            case.new.paired,
            case.new.workspace(rows).unwrap(),
            7,
        )?;
        let error = support::compare(
            session,
            &module,
            &producer,
            [&calls, &[]],
            &images,
            support::ReplayShape {
                dtype: case.new.query.output,
                rows,
                header: case.new.header(rows).unwrap() as usize,
            },
        )
        .unwrap_err();
        if !error.contains("output5") || !error.contains("holds its pattern fill") {
            return Err(format!("unexpected mutant refusal: {error}"));
        }
        session.receipt(
            "dense-decode-mutant",
            BTreeMap::from([
                ("executed_cases".into(), serde_json::json!(1)),
                ("refusal".into(), serde_json::json!(error)),
            ]),
        )?;
        println!("DENSE_TWIN_MUTANT_REJECTED {error}");
        Ok(())
    })
    .unwrap();
}
