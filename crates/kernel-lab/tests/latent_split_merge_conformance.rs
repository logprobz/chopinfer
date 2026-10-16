//! Latent attention's split merge (`latent_shapes.rs`) through the
//! declarative conformance runner (`kernel_lab::conformance`): the BF16
//! decode and verify entries and their partial forms, each chained with its
//! merge kernel ([`SplitMerge`]) and checked as one launch. Every decode
//! and verify row takes 64 fixed splits of one or two 32-key units at every
//! bucket (`latent/fixed_splits.cuh`), so a row's keys spread over several
//! splits and most splits hold only holes. Only the grid follows the bucket:
//! each chain runs at four rows, where every CTA takes one split, and at
//! [`GROUPED`] rows (the four rows repeated), where every CTA takes four
//! consecutive splits, so the partials also witness each split's own
//! softmax and copies that run across split boundaries.
//!
//! Two fixtures over the prefill cases' (`common/latent_case.rs`) give
//! exact bits for both stages. The split partials (`workspace`, F32
//! `[row][split][head][max, den, num]`) are an output of the chain in both:
//! each split's own one-hot partial over the keys of its units, and -inf,
//! +0 and +0 for a split of holes.
//!
//! - One-hot keys (`common/hot_keys.rs`): the merged outputs equal the
//!   one-split prefill oracle's, since every split but the best one's has
//!   a maximum at least the one-hot margin below, so its merge weight is
//!   +0, and the best split's weight is 1. These check that the merge picks
//!   the best split; a merge that copies only the greatest-max split passes
//!   them.
//! - Twin keys: tokens `2k` and `2k + 1` of a page score alike under every
//!   query head but hold different value rows ([`twin_row`]), and each row
//!   puts a token and its twin in different splits ([`twin_rows`]). Most
//!   heads' best keys then tie across two splits, each of weight exactly 1:
//!   the merged denominator is 2, the numerator the two value rows' sum and
//!   BF16 `o / l` their mean, all exact ([`merged`] folds the oracle's split
//!   partials as the merge does). A merge that copies one split's partial
//!   fails them ([`a_max_only_merge_fails_the_twin_keys`]).
//!
//! The block tables rerun with their pages moved to the top of the pool:
//! the workspace and the outputs witness it (the old IDs over the moved
//! pool read other tokens' latents). The last live key list reruns all
//! holes, which the merge folds to zeros, or -inf, +0 and +0.
//!
//! GPU: `CHOPINFER_KERNEL_GPU=<ordinal> CHOPINFER_KERNEL_REQUIRE=1 cargo test
//! -p kernel-lab --release --locked --offline --test
//! latent_split_merge_conformance -- --test-threads=1 --nocapture`

#[path = "common/hot_keys.rs"]
mod hot_keys;

#[path = "common/latent_case.rs"]
mod latent_case;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ops::Range;

use hot_keys::{MARGIN, bf16_bytes, f32_bytes, key_row};
use kernel_build::EntryParam;
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan, Refusal};
use kernel_lab::conformance::{Case, Expected, Images, Rows, Write};
use kernel_lab::families::pdl_misorder::{self, Stage};
use kernel_lab::lab_entry::{LabEntry, SplitMerge};
use kernel_lab::latent_shapes::cubin_name;
use kernel_lab::{LabResult, gpu_test};
use kernels_sm120::latent::Entry;
use latent_case::{
    CAPACITY, HEADS, LATENT, MAX_KEYS, POOL_PAGES, entry, inputs_of, outputs, repeated,
    row_partials,
};
use serde_json::json;

/// The chained entries: decode and verify, each normalized and partial.
const KERNELS: [&str; 4] = ["decode", "decode_partial", "verify", "verify_partial"];
/// One split partial: maximum, denominator and numerator.
const PARTIAL: usize = LATENT + 2;
/// Splits per row at every bucket, and keys per unit.
const SPLITS: usize = 64;
const UNIT: usize = 32;
/// Rows of the grouped chains, and their CTAs per row and head tile, each
/// taking four consecutive splits (`Entry::groups`).
const GROUPED: usize = 24;
const GROUPS: usize = 16;
/// The chains' capacities: one split per CTA, and four.
const CAPACITIES: [usize; 2] = [CAPACITY, GROUPED];

/// The keys of a chain's pool.
#[derive(Clone, Copy, Debug)]
enum Keys {
    OneHot,
    Twins,
}

const FIXTURES: [Keys; 2] = [Keys::OneHot, Keys::Twins];

/// The key list entries split `split` of `splits` owns: those of units
/// `[units s / splits, units (s + 1) / splits)` of `keys` entries
/// (`fixed_splits.cuh`).
fn split_entries(split: usize, splits: usize, keys: usize) -> Range<usize> {
    let units = MAX_KEYS.div_ceil(keys);
    let begin = units * split / splits * keys;
    let end = (units * (split + 1) / splits * keys).min(MAX_KEYS);
    begin..end
}

/// The split partials of every live row.
fn workspace(images: &Images, live: u32, splits: usize, keys: usize) -> LabResult<Vec<u8>> {
    partials_over(images, live, splits, |split| {
        split_entries(split, splits, keys)
    })
}

/// The partials of every live row's `splits` splits, split `s` over the
/// key list entries `entries(s)`. A split holding no key takes the row's
/// partial over no entries.
fn partials_over(
    images: &Images,
    live: u32,
    splits: usize,
    entries: impl Fn(usize) -> Range<usize>,
) -> LabResult<Vec<u8>> {
    let mut out = vec![0.0; latent_case::capacity(images)? * splits * HEADS * PARTIAL];
    let lists = images.i32s("key_lists")?;
    for row in 0..live as usize {
        let empty = row_partials(images, row, 0..0, hot_keys::whole)?;
        for split in 0..splits {
            let entries = entries(split);
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

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("4 bytes")))
        .collect()
}

/// A row and head's merged partial: maximum, denominator and numerator.
type Merged = (f32, f32, Vec<f32>);

/// The rows of the split partials `workspace`: the case's capacity.
fn rows_of(workspace: &[f32], splits: usize) -> usize {
    workspace.len() / (splits * HEADS * PARTIAL)
}

/// The chain's outputs at `live` of `capacity` rows from merged partials,
/// `fold(row, head)` for each live row: BF16 `o / l` (zeros when `l` is 0,
/// the merge's `__fdiv_rn`), or the F32 partials when `partial`.
fn merged_outputs(
    capacity: usize,
    live: u32,
    partial: bool,
    mut fold: impl FnMut(usize, usize) -> LabResult<Merged>,
) -> LabResult<Vec<(String, Vec<u8>)>> {
    let mut max = vec![0.0; capacity * HEADS];
    let mut den = vec![0.0; capacity * HEADS];
    let mut num = vec![0.0; capacity * HEADS * LATENT];
    for row in 0..live as usize {
        for head in 0..HEADS {
            let pair = row * HEADS + head;
            let (m, l, o) = fold(row, head)?;
            max[pair] = m;
            den[pair] = l;
            num[pair * LATENT..][..LATENT].copy_from_slice(&o);
        }
    }
    if partial {
        return Ok(vec![
            ("out_max".into(), f32_bytes(&max)),
            ("out_den".into(), f32_bytes(&den)),
            ("out_num".into(), f32_bytes(&num)),
        ]);
    }
    let out = num
        .iter()
        .enumerate()
        .map(|(at, &o)| {
            let l = den[at / LATENT];
            f32_to_bf16(if l == 0.0 { 0.0 } else { o / l })
        })
        .collect::<Vec<_>>();
    Ok(vec![("out".into(), bf16_bytes(&out))])
}

/// The split partial of `row`, `split` and `head` in `workspace`.
fn split_partial(
    workspace: &[f32],
    splits: usize,
    row: usize,
    split: usize,
    head: usize,
) -> &[f32] {
    &workspace[((row * splits + split) * HEADS + head) * PARTIAL..][..PARTIAL]
}

/// The merge kernel's fold of the split partials `workspace` (`attention.cuh`
/// `merge`): `M = max m_s`, weight `exp(m_s - M)` (+0 for an empty split),
/// `L` and `O` FMA sums from zero in split order. Exact only when every
/// weight is 1 (a split's maximum equals `M`) or +0 (it lies at least the
/// one-hot margin below `M`, or the split is empty); refused otherwise. Also
/// returns the splits of weight 1.
fn fold(workspace: &[f32], splits: usize, row: usize, head: usize) -> LabResult<(Merged, usize)> {
    let partial = |split| split_partial(workspace, splits, row, split, head);
    let top = (0..splits)
        .map(|split| partial(split)[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let (mut l, mut o, mut winners) = (0.0_f32, vec![0.0_f32; LATENT], 0);
    for split in 0..splits {
        let p = partial(split);
        let weight: f32 = if p[0] == f32::NEG_INFINITY {
            0.0
        } else if p[0] == top {
            winners += 1;
            1.0
        } else if f64::from(top) - f64::from(p[0]) >= MARGIN {
            0.0
        } else {
            return Err(format!(
                "row {row} head {head} split {split}'s maximum {} lies within {MARGIN} of {top}, \
                 so its merge weight is not exact",
                p[0]
            ));
        };
        l = weight.mul_add(p[1], l);
        for (o, &n) in o.iter_mut().zip(&p[2..]) {
            *o = weight.mul_add(n, *o);
        }
    }
    Ok(((top, l, o), winners))
}

/// The merged outputs at `live` of the split partials `workspace`.
fn merged(
    workspace: &[f32],
    live: u32,
    splits: usize,
    partial: bool,
) -> LabResult<Vec<(String, Expected)>> {
    let rows = rows_of(workspace, splits);
    Ok(merged_outputs(rows, live, partial, |row, head| {
        Ok(fold(workspace, splits, row, head)?.0)
    })?
    .into_iter()
    .map(|(name, bits)| (name, Expected::Bits(bits)))
    .collect())
}

/// A merge that copies each row and head's greatest-max split partial (the
/// first among ties), as the review's mutant does, at `live` of the split
/// partials `workspace`.
fn max_only(
    workspace: &[f32],
    live: u32,
    splits: usize,
    partial: bool,
) -> LabResult<Vec<(String, Vec<u8>)>> {
    let rows = rows_of(workspace, splits);
    merged_outputs(rows, live, partial, |row, head| {
        let p = |split| split_partial(workspace, splits, row, split, head);
        let best = (0..splits).fold(0, |best, split| {
            if p(split)[0] > p(best)[0] {
                split
            } else {
                best
            }
        });
        Ok((p(best)[0], p(best)[1], p(best)[2..].to_vec()))
    })
}

/// Token `code`'s latent row under the twin keys: its key row
/// ([`key_row`]) with dims 0 and 1 of the even code `code & !1`. Twin codes
/// `2k` and `2k + 1` lie on one page and score alike under the queries
/// (which zero every other dim), while their filler, and so their value
/// rows, differ; other codes still score at least the one-hot margin apart.
fn twin_row(code: usize, width: usize) -> Vec<u16> {
    let mut row = key_row(code, width);
    row[..2].copy_from_slice(&key_row(code & !1, width)[..2]);
    row
}

/// The twin keys' rows: each split takes either even or odd tokens of a
/// run, so no split holds a token and its twin (`t ^ 1`), and twins sit in
/// different splits. Items as in `latent_case::TABLES`: item 0's tokens
/// `0..64` lie on page 2, `64..128` on page 0 and `128..150` on page 3;
/// item 1's on page 1.
///
/// - Row 0: evens and odds of `0..64` in splits 0 and 1; every head's best
///   keys tie across them.
/// - Row 1: evens and odds of `64..128` in splits 5 and 40, evens of
///   `128..150` in split 63 (two units). Negative heads tie across splits 5
///   and 40; positive heads take split 63 alone.
/// - Row 2 (item 1): odds of `0..40` in split 10, evens reversed in 11.
/// - Row 3: evens and odds of `100..128` in splits 20 and 21, evens and odds
///   of `128..150` in splits 30 and 62: two winners and two +0 weights for
///   every head.
fn twin_rows() -> Vec<(usize, Vec<i32>)> {
    let tokens = |range: Range<i32>, parity: i32| range.filter(move |t| t % 2 == parity);
    let row = |item: usize, placed: Vec<(usize, Vec<i32>)>| {
        let mut list = vec![-1; MAX_KEYS];
        for (split, tokens) in placed {
            let entries = split_entries(split, SPLITS, UNIT);
            assert!(tokens.len() <= entries.len(), "split {split} overflows");
            list[entries.start..][..tokens.len()].copy_from_slice(&tokens);
        }
        (item, list)
    };
    vec![
        row(
            0,
            vec![
                (0, tokens(0..64, 0).collect()),
                (1, tokens(0..64, 1).collect()),
            ],
        ),
        row(
            0,
            vec![
                (5, tokens(64..128, 0).collect()),
                (40, tokens(64..128, 1).rev().collect()),
                (63, tokens(128..150, 0).collect()),
            ],
        ),
        row(
            1,
            vec![
                (10, tokens(0..40, 1).collect()),
                (11, tokens(0..40, 0).rev().collect()),
            ],
        ),
        row(
            0,
            vec![
                (20, tokens(100..128, 0).collect()),
                (21, tokens(100..128, 1).collect()),
                (30, tokens(128..150, 0).collect()),
                (62, tokens(128..150, 1).collect()),
            ],
        ),
    ]
}

/// The chain of `entry` and `merge` over `keys` at `capacity` rows.
fn case<'a>(
    entry: &'a Entry,
    merge: &'a SplitMerge<'a, Entry>,
    partial: bool,
    keys: Keys,
    capacity: usize,
) -> Case<'a> {
    chain(entry, entry, merge, partial, keys, capacity)
}

/// The chain of the stages `split` and `merge`, which launch as `entry` and
/// its split merge do, over `keys` at `capacity` rows.
fn chain<'a>(
    entry: &'a Entry,
    split: &'a dyn LabEntry,
    merge: &'a dyn LabEntry,
    partial: bool,
    keys: Keys,
    capacity: usize,
) -> Case<'a> {
    let splits = entry.splits();
    let unit = entry.unit_keys().expect("an attention entry") as usize;
    let chain = Case::new(split, capacity as u32).then(merge);
    let case = match keys {
        Keys::OneHot => inputs_of(
            chain,
            splits,
            POOL_PAGES,
            key_row,
            &repeated(&latent_case::rows(), capacity),
        ),
        Keys::Twins => inputs_of(
            chain,
            splits,
            POOL_PAGES,
            twin_row,
            &repeated(&twin_rows(), capacity),
        ),
    }
    .output(
        "workspace",
        DType::F32,
        (Rows::Live, splits as usize * HEADS * PARTIAL),
        Write::LiveRows,
    )
    .oracle(move |images, live| {
        let workspace = workspace(images, live, splits as usize, unit)?;
        let mut expected = match keys {
            Keys::OneHot => outputs(images, live, partial, hot_keys::whole)?,
            Keys::Twins => merged(&f32s(&workspace), live, splits as usize, partial)?,
        };
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
fn latent_split_merge_cases_validate() {
    for kernel in KERNELS {
        let entry = entry(kernel);
        let merge = SplitMerge(&entry);
        assert_eq!(entry.splits(), SPLITS as u32, "{kernel}");
        assert_eq!(entry.unit_keys(), Some(UNIT as u32), "{kernel}");
        for (capacity, groups) in [(CAPACITY, SPLITS), (GROUPED, GROUPS)] {
            assert_eq!(entry.groups(capacity as u32), groups as u32, "{kernel}");
            let plan = entry.plan(capacity as u32).expect("a plan at the capacity");
            assert_eq!(plan.grid[2], groups as u32, "{kernel}");
            for keys in FIXTURES {
                case(&entry, &merge, kernel.ends_with("_partial"), keys, capacity)
                    .validate()
                    .unwrap_or_else(|e| panic!("{kernel} {keys:?} at {capacity}: {e}"));
            }
        }
    }
}

/// The splits of each row of `rows` that hold a key.
fn holding(rows: &[(usize, Vec<i32>)]) -> Vec<usize> {
    rows.iter()
        .map(|(_, list)| {
            (0..SPLITS)
                .filter(|&split| {
                    let entries = split_entries(split, SPLITS, UNIT);
                    list.iter()
                        .take(entries.end)
                        .skip(entries.start)
                        .any(|&token| token >= 0)
                })
                .count()
        })
        .collect()
}

/// Rows spread their keys over several splits, so the merge folds real
/// partials beside the best one, and most splits hold only holes.
#[test]
fn rows_spread_their_keys_over_several_splits() {
    assert_eq!(holding(&latent_case::rows()), [5, 2, 2, 2]);
    assert_eq!(holding(&twin_rows()), [2, 3, 2, 4]);
}

/// No twin split holds a token and its twin, and every row puts some token
/// and its twin in different splits. At [`GROUPED`] rows, rows 0, 2 and 3
/// put twins in different splits of one CTA.
#[test]
fn twin_rows_keep_twins_in_different_splits() {
    let mut shared = Vec::new();
    for (row, (_, list)) in twin_rows().iter().enumerate() {
        let split_of = |token: i32| {
            (0..SPLITS).find(|&split| list[split_entries(split, SPLITS, UNIT)].contains(&token))
        };
        let mut apart = 0;
        for &token in list.iter().filter(|&&token| token >= 0) {
            let (own, twin) = (split_of(token), split_of(token ^ 1));
            assert_ne!(
                own, twin,
                "row {row} token {token} shares a split with its twin"
            );
            apart += usize::from(twin.is_some());
            let cta = |split: Option<usize>| split.map(|split| split / (SPLITS / GROUPS));
            if twin.is_some() && cta(own) == cta(twin) && !shared.contains(&row) {
                shared.push(row);
            }
        }
        assert!(apart > 0, "row {row} holds no twins");
    }
    assert_eq!(shared, [0, 2, 3]);
    // Twins tie in dims 0 and 1 and differ in their filler.
    for code in [0, 129, 190, 212] {
        let (even, odd) = (twin_row(code & !1, LATENT), twin_row(code | 1, LATENT));
        assert_eq!(even[..2], odd[..2]);
        assert_ne!(even[2..], odd[2..]);
    }
}

/// A merge that copies the greatest-max split's partial fails the twin
/// keys in every run with a live row, at both capacities, and the twin
/// oracle's fold has two splits of weight 1 at some head of every row with
/// keys.
#[test]
fn a_max_only_merge_fails_the_twin_keys() {
    for (kernel, capacity) in KERNELS.into_iter().flat_map(|k| CAPACITIES.map(|c| (k, c))) {
        let entry = entry(kernel);
        let merge = SplitMerge(&entry);
        let partial = kernel.ends_with("_partial");
        let twins = case(&entry, &merge, partial, Keys::Twins, capacity);
        // Per run: the rows whose fold has two splits of weight 1 at some
        // head, and the rows holding a key.
        let folds = RefCell::new(Vec::new());
        let refusing = twins
            .refusing_runs(|images, live| {
                let workspace = f32s(&workspace(images, live, SPLITS, UNIT)?);
                let (mut tied, mut keyed) = (0, 0);
                for row in 0..live as usize {
                    let mut winners = Vec::new();
                    for head in 0..HEADS {
                        let ((top, ..), count) = fold(&workspace, SPLITS, row, head)?;
                        winners.push((top, count));
                    }
                    tied += usize::from(winners.iter().any(|&(_, count)| count >= 2));
                    keyed += usize::from(winners.iter().any(|&(top, _)| top.is_finite()));
                }
                folds.borrow_mut().push((live, tied, keyed));
                max_only(&workspace, live, SPLITS, partial)
            })
            .unwrap_or_else(|e| panic!("{kernel} at {capacity}: {e}"));
        let folds = folds.into_inner();
        for &(live, tied, keyed) in &folds {
            assert_eq!(
                tied, keyed,
                "{kernel} at live {live}: a keyed row folds one split"
            );
        }
        assert!(
            folds.iter().any(|&(_, tied, _)| tied == capacity),
            "{kernel} at {capacity}"
        );
        let mut expected = kernel_lab::poison::live_counts(capacity as u32)
            .into_iter()
            .filter(|&live| live > 0)
            .map(|live| format!("live {live}"))
            .collect::<Vec<_>>();
        assert_eq!(expected.len(), 4);
        expected.extend(["high block_tables".into(), "holes key_lists".into()]);
        assert_eq!(refusing, expected, "{kernel} at {capacity}");
    }
}

/// A launch whose CTAs carry the online softmax across their splits
/// instead of restarting it at each one: a split's partial also covers the
/// keys of the CTA's earlier splits. It gives every split's own partial
/// when each CTA takes one split, so only the grouped chains refuse it, in
/// every run with a live row.
#[test]
fn a_softmax_carried_across_a_ctas_splits_fails_only_the_grouped_chains() {
    for (kernel, capacity) in KERNELS.into_iter().flat_map(|k| CAPACITIES.map(|c| (k, c))) {
        let entry = entry(kernel);
        let merge = SplitMerge(&entry);
        let per_cta = SPLITS / entry.groups(capacity as u32) as usize;
        let refusing = case(
            &entry,
            &merge,
            kernel.ends_with("_partial"),
            Keys::OneHot,
            capacity,
        )
        .refusing_runs(|images, live| {
            let carried = partials_over(images, live, SPLITS, |split| {
                let first = split / per_cta * per_cta;
                split_entries(first, SPLITS, UNIT).start..split_entries(split, SPLITS, UNIT).end
            })?;
            Ok(vec![("workspace".into(), carried)])
        })
        .unwrap_or_else(|e| panic!("{kernel} at {capacity}: {e}"));
        let expected: &[&str] = if capacity == GROUPED {
            &[
                "live 24",
                "live 1",
                "live 12",
                "live 23",
                "high block_tables",
                "holes key_lists",
            ]
        } else {
            &[]
        };
        assert_eq!(refusing, expected, "{kernel} at {capacity}");
    }
}

/// The merge model the twin oracle uses agrees with the one-split prefill
/// oracle on the one-hot keys in every run, at both capacities.
#[test]
fn the_merge_model_matches_the_one_split_oracle() {
    for (kernel, capacity) in KERNELS.into_iter().flat_map(|k| CAPACITIES.map(|c| (k, c))) {
        let entry = entry(kernel);
        let merge = SplitMerge(&entry);
        let partial = kernel.ends_with("_partial");
        let refusing = case(&entry, &merge, partial, Keys::OneHot, capacity)
            .refusing_runs(|images, live| {
                let workspace = f32s(&workspace(images, live, SPLITS, UNIT)?);
                merged_outputs(capacity, live, partial, |row, head| {
                    Ok(fold(&workspace, SPLITS, row, head)?.0)
                })
            })
            .unwrap_or_else(|e| panic!("{kernel} at {capacity}: {e}"));
        assert!(refusing.is_empty(), "{kernel} at {capacity}: {refusing:?}");
    }
}

#[test]
fn latent_split_merges_conform_gpu() {
    gpu_test(1 << 30, |session| {
        let module = session.module(&cubin_name())?;
        for kernel in KERNELS {
            let entry = entry(kernel);
            let merge = SplitMerge(&entry);
            for (keys, capacity) in FIXTURES
                .into_iter()
                .flat_map(|k| CAPACITIES.map(|c| (k, c)))
            {
                let evidence = case(&entry, &merge, kernel.ends_with("_partial"), keys, capacity)
                    .run(session, &module, Some(&module))
                    .map_err(|e| format!("{kernel} {keys:?} at {capacity}: {e}"))?;
                session.receipt(
                    "latent_split_merge_conformance",
                    BTreeMap::from([
                        ("kernel".into(), json!(kernel)),
                        ("keys".into(), json!(format!("{keys:?}"))),
                        ("capacity".into(), json!(capacity)),
                        ("evidence".into(), evidence),
                    ]),
                )?;
            }
        }
        Ok(())
    })
    .expect("latent split merges conform");
}

/// A stage launched under another symbol: a negative control's copy of a
/// served stage (`kernel_lab::families::pdl_misorder`).
struct Renamed<'a> {
    stage: &'a dyn LabEntry,
    symbol: String,
}

impl LabEntry for Renamed<'_> {
    fn symbol(&self) -> String {
        self.symbol.clone()
    }

    fn params(&self) -> Vec<EntryParam> {
        self.stage.params()
    }

    fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.stage.plan(prepared_rows)
    }

    fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        self.stage.pack_arguments(values)
    }
}

/// Runs of each negative control.
const CONTROL_RUNS: usize = 10;

/// The phase a run's `error` names (`<symbol> live <n>: [<producer>: ]
/// <buffer> <phase>: ...`), if any.
fn failed_phase(error: &str) -> Option<&'static str> {
    ["eager", "relaunch", "graph", "PDL"]
        .into_iter()
        .find(|phase| error.contains(&format!(" {phase}: ")))
}

/// Each control cubin holds the decode chain's two stages under the
/// control's symbols, with their served parameters.
#[test]
fn pdl_misorder_controls_hold_the_chain_stages() {
    let entry = entry(pdl_misorder::KERNEL);
    let merge = SplitMerge(&entry);
    for stage in Stage::ALL {
        let symbols = kernel_lab::cubin_entries(&stage.cubin_name()).expect("a built control");
        assert_eq!(
            symbols,
            [stage.symbol(&entry.symbol()), stage.symbol(&merge.symbol())],
            "{stage:?}"
        );
        let spec = stage.cubin();
        assert_eq!(spec.entries[0].params, entry.params(), "{stage:?}");
        assert_eq!(spec.entries[1].params, merge.params(), "{stage:?}");
    }
}

/// A GPU negative control of the runner's graph and PDL phases: the decode
/// chain with `stage`'s dependency wait dropped, run [`CONTROL_RUNS`] times.
/// Every run must fail as expected: in a phase that launches the stages
/// behind a predecessor (graph or PDL), never eagerly. A run stops at its
/// first failing phase, so a run the graph phase fails never reaches the
/// PDL phases behind the live-count and copy producers. The receipt records
/// each run's phase, whether any run reached a PDL phase, and the first
/// error as the expected failure.
fn misordered_chain_fails(stage: Stage) {
    gpu_test(1 << 30, |session| {
        let lab = session.module(&cubin_name())?;
        let module = session.module(&stage.cubin_name())?;
        let entry = entry(pdl_misorder::KERNEL);
        let merge = SplitMerge(&entry);
        let split = Renamed {
            stage: &entry,
            symbol: stage.symbol(&entry.symbol()),
        };
        let merged = Renamed {
            stage: &merge,
            symbol: stage.symbol(&merge.symbol()),
        };
        let case = chain(&entry, &split, &merged, false, Keys::OneHot, CAPACITY);
        let mut phases = Vec::new();
        let mut first = None;
        for _ in 0..CONTROL_RUNS {
            let phase = match case.run(session, &module, Some(&lab)) {
                Ok(_) => "passed",
                Err(error) => {
                    let phase = failed_phase(&error).unwrap_or("unnamed");
                    first.get_or_insert(error);
                    phase
                }
            };
            phases.push(phase);
        }
        let caught = phases
            .iter()
            .filter(|phase| ["graph", "PDL"].contains(phase))
            .count();
        let pdl_phase_reached = phases.contains(&"PDL");
        eprintln!("{stage:?} control: caught {caught} of {CONTROL_RUNS}: {phases:?}");
        if !pdl_phase_reached {
            eprintln!("{stage:?} control: no run reached the PDL phases");
        }
        eprintln!("{stage:?} control: first error: {first:?}");
        session.receipt(
            "pdl-negative-control",
            BTreeMap::from([
                (
                    "control".into(),
                    json!(format!("{stage:?} without its PDL wait")),
                ),
                ("kernel".into(), json!(pdl_misorder::KERNEL)),
                ("capacity".into(), json!(CAPACITY)),
                ("expected_failure".into(), json!(true)),
                ("runs".into(), json!(CONTROL_RUNS)),
                ("caught".into(), json!(caught)),
                ("phases".into(), json!(phases)),
                ("pdl_phase_reached".into(), json!(pdl_phase_reached)),
                ("first_error".into(), json!(first)),
            ]),
        )?;
        if caught == CONTROL_RUNS {
            Ok(())
        } else {
            Err(format!("caught {caught} of {CONTROL_RUNS}: {phases:?}"))
        }
    })
    .unwrap_or_else(|e| panic!("{stage:?} without its PDL wait: {e}"));
}

/// The split reads the live count and its inputs before its wait.
#[test]
fn a_split_read_before_its_pdl_wait_fails_the_chain_gpu() {
    misordered_chain_fails(Stage::Split);
}

/// The merge reads the split's partials and the live count before its
/// wait.
#[test]
fn a_merge_read_before_its_pdl_wait_fails_the_chain_gpu() {
    misordered_chain_fails(Stage::Merge);
}
