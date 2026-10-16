//! Selected attention partials (`selected_partials_shapes.rs`; README in
//! crates/kernels-sm120/cuda/selected_partials) at a TP4 rank of the
//! block-sparse attention: six query heads over one KV head of 256, 64-token
//! pages, blocks of 4 and 512 selected blocks, BF16 and FP8 KV:
//!
//! - every rank's partials of decode, verify and prefill rows, over random
//!   ascending selections (and every block of rows at most 512 complete
//!   blocks long), against the f64 reference
//!   (`context_parallel_oracle::selected_attention_partials`): the maximum
//!   within one score's bound, the denominator and numerator within `2^-22
//!   K S` of the reference rescaled to the kernel's maximum, and rows
//!   without a kept key exactly -inf, +0, +0;
//! - the 128-K accumulation rule: queries and keys spanning 2^-8..2^8 with
//!   random signs over 2051 selected keys, against Neumaier-compensated f64;
//! - every group rank of sizes 2 and 4, each over its own pages and the
//!   global selection, merged by `partial_merge{N}` of six heads of 256,
//!   against the f64 block-sparse attention of the full key sets
//!   (`sparse_attention_oracle::sparse_attention`) within the shared gate
//!   plus the merge's `exp2f` allowance;
//! - pages are shuffled, strided past decoy bytes that would poison any
//!   read outside a page, and rows past the live count stay untouched; every
//!   launch runs eager, relaunched, from a graph and behind a PDL producer
//!   on poisoned outputs (`kernel_lab::poison`).
//!
//! Run in release with
//! `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release --test selected_partials`.

#[path = "common/selected_partials.rs"]
mod fixture;
#[allow(dead_code)]
#[path = "../paged_partials_shapes.rs"]
mod merge_shapes;
#[path = "../selected_partials_shapes.rs"]
mod shapes;

use fixture::{
    Kv, Launch, Rng, Step, err, full_reference, gate_merged, gate_partial, merge, queries,
    rank_cache, reference, run, scales, selections,
};
use kernel_lab::gpu_test;
use kernels_sm120::selected_partials::Entry;

fn entry(kv: &str, phase: &str) -> Entry {
    shapes::lab_entries()
        .into_iter()
        .find(|entry| entry.name == format!("{}_{kv}_{phase}", shapes::PROFILE.tag()))
        .unwrap_or_else(|| panic!("no lab entry {kv} {phase}"))
}

/// Every group rank's partials at group sizes 1, 2 and 4 against the f64
/// reference, for decode, verify and prefill rows of BF16 and FP8 KV: dense
/// rows, rows across the 2051-token limit and long sparse rows, with rows
/// that see no key of a rank.
#[test]
fn partials_match_their_references() {
    gpu_test(4 << 30, |session| {
        let module = session.module(shapes::CUBIN)?;
        let mut rng = Rng(0x5e1e_0042);
        let mut failures = Vec::new();
        for (kv_name, fp8) in [("bf16", false), ("fp8", true)] {
            let cases = [
                (
                    "decode",
                    Step::decode(&[1, 7, 64, 2051, 2052, 2600, 5000, 9000]),
                    8,
                ),
                ("decode", Step::decode(&[50, 12_000]), 32),
                ("verify", Step::verify(&[4, 70, 2053, 3000], 4), 32),
                ("prefill", Step::prefill(0, 300), 300),
                ("prefill", Step::prefill(1900, 300), 512),
            ];
            for (phase, step, prepared) in cases {
                let entry = entry(kv_name, phase);
                let function = module.function(&entry.symbol()).map_err(err)?;
                let kv = Kv::new(&step.lengths, fp8, false, &mut rng);
                let q = queries(step.rows.len(), false, &mut rng);
                let selected = selections(&step, &mut rng);
                for size in [1, 2, 4] {
                    for rank in 0..size {
                        let label = format!(
                            "{} rows {} prepared {prepared} rank {rank} of {size}",
                            entry.name,
                            step.rows.len()
                        );
                        let cache = rank_cache(&kv, rank, size, true, &mut rng);
                        let operands = Launch {
                            step: &step,
                            q: &q,
                            selected: &selected,
                            cache: &cache,
                            scales: scales(fp8),
                            rank,
                            size,
                            prepared,
                        };
                        let result = run(session, &function, &entry, &operands, true, &label)
                            .and_then(|actual| {
                                gate_partial(&label, &actual, &reference(&entry, &operands)?)
                            });
                        if let Err(error) = result {
                            failures.push(error);
                        }
                    }
                }
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("\n"))
        }
    })
    .expect("selected attention partials match their references");
}

/// Queries and keys spanning 2^-8..2^8 with random signs: a score sums 256
/// products and a partial up to 2051 selected keys, against
/// Neumaier-compensated f64 within 2^-22 K S.
#[test]
fn partials_hold_the_accumulation_rule() {
    gpu_test(4 << 30, |session| {
        let module = session.module(shapes::CUBIN)?;
        let mut rng = Rng(0xacc0_5e1e);
        let entry = entry("bf16", "decode");
        let function = module.function(&entry.symbol()).map_err(err)?;
        let step = Step::decode(&[9000, 2051]);
        let kv = Kv::new(&step.lengths, false, true, &mut rng);
        let q = queries(step.rows.len(), true, &mut rng);
        let selected = selections(&step, &mut rng);
        // Scores of hundreds keep the weights from all vanishing.
        let scales = (2_f32.powi(-12), 1.0, 1.0);
        for size in [1, 2] {
            for rank in 0..size {
                let label = format!("accumulation rank {rank} of {size}");
                let cache = rank_cache(&kv, rank, size, true, &mut rng);
                let operands = Launch {
                    step: &step,
                    q: &q,
                    selected: &selected,
                    cache: &cache,
                    scales,
                    rank,
                    size,
                    prepared: 2,
                };
                let actual = run(session, &function, &entry, &operands, true, &label)?;
                gate_partial(&label, &actual, &reference(&entry, &operands)?)?;
            }
        }
        Ok(())
    })
    .expect("selected attention partials hold the 128-K accumulation rule");
}

/// Every group rank's partials over its own pages and the global selection,
/// merged by `partial_merge{N}`, equal the f64 block-sparse attention of the
/// full key sets within the shared gate plus the merge's `exp2f` allowance,
/// at group sizes 2 and 4.
#[test]
fn rank_partials_merge_into_the_selected_attention() {
    gpu_test(4 << 30, |session| {
        let module = session.module(shapes::CUBIN)?;
        let merges = session.module(merge_shapes::MERGE_CUBIN)?;
        let merge_entries = merge_shapes::merge_entries();
        let mut rng = Rng(0x3e26_5e1e);
        let mut failures = Vec::new();
        for (kv_name, fp8) in [("bf16", false), ("fp8", true)] {
            for (phase, step) in [
                ("decode", Step::decode(&[3, 129, 2051, 2052, 5000, 20_000])),
                ("verify", Step::verify(&[6, 2100, 9000], 4)),
                ("prefill", Step::prefill(1950, 260)),
            ] {
                let entry = entry(kv_name, phase);
                let function = module.function(&entry.symbol()).map_err(err)?;
                let kv = Kv::new(&step.lengths, fp8, false, &mut rng);
                let q = queries(step.rows.len(), false, &mut rng);
                let selected = selections(&step, &mut rng);
                let rows = step.rows.len();
                let full = full_reference(&step, &q, &selected, &kv, scales(fp8))?;
                for size in [2_usize, 4] {
                    let label = format!("{} rows {rows} merged over {size}", entry.name);
                    let parts = (0..size)
                        .map(|rank| {
                            let cache = rank_cache(&kv, rank, size, false, &mut rng);
                            let operands = Launch {
                                step: &step,
                                q: &q,
                                selected: &selected,
                                cache: &cache,
                                scales: scales(fp8),
                                rank,
                                size,
                                prepared: rows,
                            };
                            run(session, &function, &entry, &operands, false, &label)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let merged = merge(session, &merges, &merge_entries, &parts, rows)?;
                    if let Err(error) = gate_merged(&label, &merged, &parts, &full) {
                        failures.push(error);
                    }
                }
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("\n"))
        }
    })
    .expect("rank partials merge into the selected attention");
}
