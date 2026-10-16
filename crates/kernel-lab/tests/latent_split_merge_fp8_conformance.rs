//! Latent attention's split merge (`latent_shapes.rs`) over an FP8 E4M3
//! block-128 pool through the declarative conformance runner
//! (`kernel_lab::conformance`): the FP8 decode and verify entries and their
//! partial forms, each chained with its merge kernel ([`SplitMerge`]) and
//! checked as one launch, over the FP8 prefill cases' fixture
//! (`common/latent_fp8_case.rs`). Every decode and verify row takes 64
//! fixed splits of one or two units at every bucket
//! (`latent/fixed_splits.cuh`), so a row's keys spread over several splits
//! and most splits hold only holes. Each chain runs at four rows, where
//! every CTA takes one split, and at [`GROUPED`] rows (the four rows
//! repeated), where every CTA takes four consecutive splits.
//!
//! The keys are FP8 one-hot (`common/hot_fp8.rs`), which gives exact bits
//! for both stages: the split partials (`workspace`, F32
//! `[row][split][head][max, den, num]`, an output of the chain) are each
//! split's own one-hot partial over the keys of its units, and -inf, +0
//! and +0 for a split of holes; the merged outputs equal the one-split
//! prefill oracle's, since every split but the best one's has a maximum at
//! least the one-hot margin below, so its merge weight is +0, and the best
//! split's weight is 1. These check that the merge picks the best split;
//! the BF16 chains' twin keys (`latent_split_merge_conformance.rs`) check
//! that it sums tied splits, which reads no KV.
//!
//! The block tables rerun with their pages moved to the top of the pool:
//! the workspace and the outputs witness it (the old IDs over the moved
//! pool read other tokens' codes and scales). The last live key list reruns
//! all holes, which the merge folds to zeros, or -inf, +0 and +0.
//!
//! GPU: `CHOPINFER_KERNEL_GPU=<ordinal> CHOPINFER_KERNEL_REQUIRE=1 cargo test
//! -p kernel-lab --release --locked --offline --test
//! latent_split_merge_fp8_conformance -- --test-threads=1 --nocapture`

#[path = "common/hot_keys.rs"]
mod hot_keys;

#[path = "common/hot_fp8.rs"]
mod hot_fp8;

#[path = "common/latent_case.rs"]
mod latent_case;

#[path = "common/latent_fp8_case.rs"]
mod latent_fp8_case;

use std::collections::BTreeMap;
use std::ops::Range;

use hot_keys::f32_bytes;
use kernel_core::contract::DType;
use kernel_lab::conformance::{Case, Expected, Images, Rows, Write};
use kernel_lab::lab_entry::SplitMerge;
use kernel_lab::latent_shapes::cubin_name;
use kernel_lab::{LabResult, gpu_test};
use kernels_sm120::latent::Entry;
use latent_case::{CAPACITY, HEADS, LATENT, MAX_KEYS, POOL_PAGES, entry_of};
use latent_fp8_case::{inputs_of, outputs, row_partials};
use serde_json::json;

/// The chained entries: decode and verify, each normalized and partial.
const KERNELS: [&str; 4] = ["decode", "decode_partial", "verify", "verify_partial"];
/// One split partial: maximum, denominator and numerator.
const PARTIAL: usize = LATENT + 2;
/// Splits per row at every bucket.
const SPLITS: usize = 64;
/// Rows of the grouped chains, and their CTAs per row and head tile, each
/// taking four consecutive splits (`Entry::groups`).
const GROUPED: usize = 24;
const GROUPS: usize = 16;

fn entry(kernel: &str) -> Entry {
    entry_of("fp8", kernel)
}

/// The key list entries split `split` of `splits` owns: those of units
/// `[units s / splits, units (s + 1) / splits)` of `keys` entries
/// (`fixed_splits.cuh`).
fn split_entries(split: usize, splits: usize, keys: usize) -> Range<usize> {
    let units = MAX_KEYS.div_ceil(keys);
    let begin = units * split / splits * keys;
    let end = (units * (split + 1) / splits * keys).min(MAX_KEYS);
    begin..end
}

/// The split partials of every live row. A split holding no key takes the
/// row's partial over no entries.
fn workspace(images: &Images, live: u32, splits: usize, keys: usize) -> LabResult<Vec<u8>> {
    let mut out = vec![0.0; latent_case::capacity(images)? * splits * HEADS * PARTIAL];
    let lists = images.i32s("key_lists")?;
    for row in 0..live as usize {
        let empty = row_partials(images, row, 0..0, hot_keys::whole)?;
        for split in 0..splits {
            let entries = split_entries(split, splits, keys);
            let list = &lists[row * MAX_KEYS..][entries.clone()];
            let partial = if list.iter().any(|&token| token >= 0) {
                &row_partials(images, row, entries, hot_keys::whole)?
            } else {
                &empty
            };
            for head in 0..HEADS {
                let at = &mut out[((row * splits + split) * HEADS + head) * PARTIAL..][..PARTIAL];
                at[0] = partial.max[head];
                at[1] = partial.den[head];
                at[2..].copy_from_slice(&partial.num[head * LATENT..][..LATENT]);
            }
        }
    }
    Ok(f32_bytes(&out))
}

/// The chain of `entry` and `merge` at `capacity` rows.
fn case<'a>(
    entry: &'a Entry,
    merge: &'a SplitMerge<'a, Entry>,
    partial: bool,
    capacity: usize,
) -> Case<'a> {
    let splits = entry.splits();
    let unit = entry.unit_keys().expect("an attention entry") as usize;
    let case = inputs_of(
        Case::new(entry, capacity as u32).then(merge),
        splits,
        POOL_PAGES,
        &latent_case::repeated(&latent_case::rows(), capacity),
    )
    .output(
        "workspace",
        DType::F32,
        (Rows::Live, splits as usize * HEADS * PARTIAL),
        Write::LiveRows,
    )
    .oracle(move |images, live| {
        let workspace = workspace(images, live, splits as usize, unit)?;
        let mut expected = outputs(images, live, partial, hot_keys::whole)?;
        expected.push(("workspace".into(), Expected::Bits(workspace)));
        Ok(expected)
    });
    if partial {
        case.output("out_max", DType::F32, (Rows::Live, HEADS), Write::LiveRows)
            .output("out_den", DType::F32, (Rows::Live, HEADS), Write::LiveRows)
            .output(
                "out_num",
                DType::F32,
                (Rows::Live, HEADS * LATENT),
                Write::LiveRows,
            )
    } else {
        case.output(
            "out",
            DType::Bf16,
            (Rows::Live, HEADS * LATENT),
            Write::LiveRows,
        )
    }
}

/// Each chain folds 64 splits, several holding a row's keys, one per CTA
/// at four rows and four per CTA at [`GROUPED`] rows.
#[test]
fn fp8_latent_split_merge_cases_validate() {
    for kernel in KERNELS {
        let entry = entry(kernel);
        let merge = SplitMerge(&entry);
        assert_eq!(entry.splits(), SPLITS as u32, "{kernel}");
        let unit = entry.unit_keys().expect("an attention entry") as usize;
        // Every row's keys spread over more than one split.
        for (row, (_, list)) in latent_case::rows().iter().enumerate() {
            let holding = (0..SPLITS)
                .filter(|&split| {
                    split_entries(split, SPLITS, unit)
                        .any(|at| list.get(at).is_some_and(|&t| t >= 0))
                })
                .count();
            assert!(
                holding > 1,
                "{kernel} row {row}: {holding} splits hold keys"
            );
        }
        for (capacity, groups) in [(CAPACITY, SPLITS), (GROUPED, GROUPS)] {
            assert_eq!(entry.groups(capacity as u32), groups as u32, "{kernel}");
            let plan = entry.plan(capacity as u32).expect("a plan at the capacity");
            assert_eq!(plan.grid[2], groups as u32, "{kernel}");
            case(&entry, &merge, kernel.ends_with("_partial"), capacity)
                .validate()
                .unwrap_or_else(|e| panic!("{kernel} at {capacity}: {e}"));
        }
    }
}

#[test]
fn fp8_latent_split_merges_conform_gpu() {
    gpu_test(1 << 30, |session| {
        let module = session.module(&cubin_name())?;
        for kernel in KERNELS {
            let entry = entry(kernel);
            let merge = SplitMerge(&entry);
            for capacity in [CAPACITY, GROUPED] {
                let evidence = case(&entry, &merge, kernel.ends_with("_partial"), capacity)
                    .run(session, &module, Some(&module))
                    .map_err(|e| format!("fp8 {kernel} at {capacity}: {e}"))?;
                session.receipt(
                    "latent_split_merge_fp8_conformance",
                    BTreeMap::from([
                        ("kernel".into(), json!(kernel)),
                        ("capacity".into(), json!(capacity)),
                        ("evidence".into(), evidence),
                    ]),
                )?;
            }
        }
        Ok(())
    })
    .expect("fp8 latent split merges conform");
}
