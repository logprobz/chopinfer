//! The context-parallel entries against their host references at the served
//! latent shape (`context_parallel_shapes.rs`; README in
//! crates/kernels-sm120/cuda/context_parallel).
//!
//! - Key localization is exact: every output equals
//!   `context_parallel_oracle::localize_keys`, which agrees with the host's
//!   `ContextParallelPagesV1` ownership, at group sizes 2 and 4.
//! - The partial merge equals `context_parallel_oracle::partial_merge_f32`
//!   bit for bit when every weight is exact (equal maxima, empty ranks), and
//!   otherwise holds the f64 merge of the same F32 partials within the
//!   shared gate (half a BF16 ulp plus `2^-22 K S`, S = `sum a_r |o_r| / L`)
//!   plus a separately stated `exp2f` allowance, for every group rank's
//!   heads in row-major and block-major (`MergeLayout::gathered`) segments.
//! - Rows past the live count stay untouched, and every launch runs eager,
//!   relaunched, from a graph and behind a PDL producer on poisoned outputs
//!   (`kernel_lab::poison`).
//!
//! Run in release with
//! `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release --test context_parallel`.

#[path = "../context_parallel_shapes.rs"]
mod shapes;

#[path = "common/support.rs"]
mod support;

use std::collections::BTreeMap;

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::context_parallel::ContextParallelQuery;
use kernel_core::context_parallel_oracle::{self as oracle, PartialsF32};
use kernel_core::contract::DType;
use kernel_core::latent_attention_oracle::{LatentPartials, merge_partials};
use kernel_core::oracle::{OracleOutput, numeric_bound};
use kernel_lab::poison::{Replay, WriteMap, replay_all};
use kernel_lab::{LabResult, Session, gpu_test, launch};
use kernels_sm120::context_parallel::{Entry, Kind, MergeLayout};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module};
use runtime_kv::ContextParallelPagesV1;
use serde_json::json;
use support::{Rng, download, upload};

const HEADS: usize = 16;
const WIDTH: usize = 512;
const PAGE: usize = 64;
/// The fill of key list outputs: no position, and not -1.
const LIST_FILL: [u8; 4] = [0xa5, 0x5a, 0xc3, 0x3c];

fn err(e: impl ToString) -> String {
    e.to_string()
}

fn entry(kind: Kind) -> Entry {
    shapes::lab_entries()
        .into_iter()
        .find(|entry| entry.kind == kind)
        .unwrap_or_else(|| panic!("no lab entry {kind:?}"))
}

fn module(session: &Session) -> LabResult<Module> {
    session.module(shapes::CUBIN)
}

/// One group's gathered partials: `ranks` segments of `rank_rows * 16 *
/// ranks` pairs each (`m`, `l`, then `o`). The same segments read as
/// row-major `[rank_rows][16 * ranks]` or block-major `[ranks][rank_rows]
/// [16]` (`MergeLayout::gathered`), the layout the per-block partial
/// attention launches write.
struct Gathered {
    ranks: usize,
    maximum: Vec<Vec<f32>>,
    denominator: Vec<Vec<f32>>,
    numerator: Vec<Vec<f32>>,
}

impl Gathered {
    /// `case`: "equal" gives every non-empty rank of a pair one maximum;
    /// "general" draws maxima in [-8, 8); some pairs have empty ranks.
    fn new(rng: &mut Rng, ranks: usize, rank_rows: usize, case: &str) -> Gathered {
        let input_heads = HEADS * ranks;
        let pairs = rank_rows * input_heads;
        let mut maximum = vec![vec![0.0; pairs]; ranks];
        let mut denominator = vec![vec![0.0; pairs]; ranks];
        let mut numerator = vec![vec![0.0; pairs * WIDTH]; ranks];
        for pair in 0..pairs {
            let shared = 4.0 * rng.unit();
            // Pair kinds: every rank holding keys, one empty rank, all empty.
            let kind = pair % 7;
            for rank in 0..ranks {
                let empty = kind == 6 || (kind == 5 && rank == pair % ranks);
                if empty {
                    maximum[rank][pair] = f32::NEG_INFINITY;
                    continue;
                }
                maximum[rank][pair] = if case == "equal" {
                    shared
                } else {
                    8.0 * rng.unit()
                };
                // A denominator of at least 1 (the maximum's own key).
                denominator[rank][pair] = 1.0 + 64.0 * (rng.unit() + 1.0);
                for column in 0..WIDTH {
                    numerator[rank][pair * WIDTH + column] = 16.0 * rng.unit();
                }
            }
        }
        Gathered {
            ranks,
            maximum,
            denominator,
            numerator,
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for rank in 0..self.ranks {
            for values in [
                &self.maximum[rank],
                &self.denominator[rank],
                &self.numerator[rank],
            ] {
                out.extend(values.iter().flat_map(|v| v.to_ne_bytes()));
            }
        }
        out
    }

    /// The pairs `layout` reads for rows `0..rows` as the contract's
    /// per-rank inputs `[rows, 16]`.
    fn owned(&self, layout: MergeLayout, rows: usize) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        (0..self.ranks)
            .map(|rank| {
                let pairs = (0..rows as u32)
                    .flat_map(|row| (0..HEADS as u32).map(move |h| layout.pair(row, h) as usize))
                    .collect::<Vec<_>>();
                (
                    pairs.iter().map(|&p| self.maximum[rank][p]).collect(),
                    pairs.iter().map(|&p| self.denominator[rank][p]).collect(),
                    pairs
                        .iter()
                        .flat_map(|&p| self.numerator[rank][p * WIDTH..][..WIDTH].iter().copied())
                        .collect(),
                )
            })
            .collect()
    }
}

/// The weight error the merge adds beyond the shared gate's sums, as a
/// multiple of the normalized `S = sum a_r |o_r| / L`. Each weight is
/// `exp2f((m_r - M) * log2e)`: `exp2f` is within 2 ulp (2^-22 relative),
/// and the F32 subtraction, the product and the F32 `log2e` each round once,
/// moving the exponent by at most `3 * 2^-24 |m_r - M| log2e`, which is
/// `3 * 2^-24 |m_r - M| / ln 2` in the exponent and `3 * 2^-24 |m_r - M|`
/// relative in the weight. `O` and `L` both carry the weight error, so the
/// quotient carries it twice: `2 * max_r (2^-22 + 3 * 2^-24 |m_r - M|) * S`.
/// The F32 FMA sums and the division are inside the shared gate's
/// `2^-22 K S` (two K-term chains of 2^-24 each, and one rounding).
fn exp2_allowance(maxima: &[f32], top: f32, s: f64) -> f64 {
    let worst = maxima
        .iter()
        .filter(|&&m| m != f32::NEG_INFINITY)
        .map(|&m| 2_f64.powi(-22) + 3.0 * 2_f64.powi(-24) * (f64::from(top) - f64::from(m)))
        .fold(0.0, f64::max);
    2.0 * worst * s
}

/// Checks a merged output against the f64 merge of the same F32 partials
/// (`latent_attention_oracle::merge_partials`, each rank's numerator one
/// term of `S = |o_r|`, so the merged reference carries `S = sum a_r |o_r| /
/// L` and K = the non-empty ranks): the shared gate (`numeric_bound`, half
/// a BF16 ulp plus `2^-22 K S`) plus [`exp2_allowance`].
fn check_f64(case: &str, out: &[u16], inputs: &[(Vec<f32>, Vec<f32>, Vec<f32>)]) -> LabResult<()> {
    let refusal = |e: kernel_core::contract::Refusal| e.0.to_string();
    let as_f64 = |values: &[f32]| values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let partials = inputs
        .iter()
        .map(|(m, l, o)| {
            Ok(LatentPartials {
                maximum: OracleOutput::new(as_f64(m), vec![0.0; m.len()], 1).map_err(refusal)?,
                denominator: OracleOutput::new(as_f64(l), vec![0.0; l.len()], 1)
                    .map_err(refusal)?,
                numerator: OracleOutput::new(
                    as_f64(o),
                    o.iter().map(|&v| f64::from(v).abs()).collect(),
                    1,
                )
                .map_err(refusal)?,
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    let reference = merge_partials(&partials).map_err(refusal)?;
    for (index, &bits) in out.iter().enumerate() {
        let pair = index / WIDTH;
        let maxima = inputs.iter().map(|r| r.0[pair]).collect::<Vec<_>>();
        let top = maxima.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let actual = f64::from(bf16_to_f32(bits));
        if top == f32::NEG_INFINITY {
            if bits != 0 {
                return Err(format!("{case} pair {pair}: an empty pair is not zero"));
            }
            continue;
        }
        let expected = reference.values[index];
        let bound = numeric_bound(&reference, index, DType::Bf16)
            + exp2_allowance(&maxima, top, reference.sum_abs_products[index]);
        if !actual.is_finite() || (actual - expected).abs() > bound {
            return Err(format!(
                "{case} pair {pair} column {}: {actual} vs {expected} (bound {bound})",
                index % WIDTH
            ));
        }
    }
    Ok(())
}

fn merge_launch(
    session: &Session,
    function: &Function,
    entry: &Entry,
    prepared: u32,
    args: (&DeviceBuffer, &DeviceBuffer, &DeviceBuffer, u32, u32, u32),
) -> LabResult<()> {
    let plan = entry.plan(prepared).map_err(|e| e.0.to_string())?;
    let (partials, live, out, rank_rows, input_heads, head_offset) = args;
    function
        .launch(
            &session.stream,
            launch::config(plan)?,
            &[
                KernelArgument::Buffer(partials),
                KernelArgument::Buffer(live),
                KernelArgument::Buffer(out),
                KernelArgument::U32(rank_rows),
                KernelArgument::U32(input_heads),
                KernelArgument::U32(head_offset),
            ],
        )
        .map_err(err)
}

/// The partial merge at group sizes 2 and 4, for every group rank's heads,
/// bitwise at exact weights and within the f64 gate otherwise.
#[test]
fn partial_merge_matches_its_references() {
    gpu_test(4 << 30, |session| {
        let module = module(session)?;
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for ranks in [2_u32, 4] {
            let entry = entry(Kind::PartialMerge { ranks });
            let function = module.function(&entry.symbol()).map_err(err)?;
            // (case, live rows, prepared rows, segment rows)
            for (case, live, prepared, rank_rows) in [
                ("equal", 1, 1, 1),
                ("equal", 29, 32, 35),
                ("general", 32, 32, 32),
                ("general", 6, 8, 8),
                ("general", 2048, 2048, 2048),
            ] {
                let gathered = Gathered::new(&mut rng, ranks as usize, rank_rows, case);
                let partials = upload(session, &gathered.bytes())?;
                let live_buffer = upload(session, &(live as u32).to_ne_bytes())?;
                let out_bytes = prepared * HEADS * WIDTH * 2;
                let out = session.allocate(out_bytes)?;
                let layouts = (0..ranks).flat_map(|owner| {
                    [
                        (
                            "row-major",
                            MergeLayout {
                                rank_rows: rank_rows as u32,
                                input_heads: HEADS as u32 * ranks,
                                head_offset: owner * HEADS as u32,
                            },
                        ),
                        (
                            "block-major",
                            MergeLayout::gathered(rank_rows as u32, HEADS as u32, ranks, owner),
                        ),
                    ]
                    .map(|(name, layout)| (owner, name, layout))
                });
                for (owner, layout_name, layout) in layouts {
                    entry
                        .check_merge(prepared as u32, layout)
                        .map_err(|e| e.0.to_string())?;
                    let launch = || {
                        merge_launch(
                            session,
                            &function,
                            &entry,
                            prepared as u32,
                            (
                                &partials,
                                &live_buffer,
                                &out,
                                layout.rank_rows,
                                layout.input_heads,
                                layout.head_offset,
                            ),
                        )
                    };
                    let replay = Replay::new(session).output(
                        "merged",
                        &out,
                        WriteMap::of(DType::Bf16).prefix(live * HEADS * WIDTH)?,
                    );
                    let mut receipt = BTreeMap::from([
                        ("entry".into(), json!(entry.name)),
                        ("case".into(), json!(case)),
                        ("live".into(), json!(live)),
                        ("prepared".into(), json!(prepared)),
                        ("layout".into(), json!(layout_name)),
                        ("owner".into(), json!(owner)),
                    ]);
                    replay_all(session, replay, &live_buffer, launch, &mut receipt)?;
                    session.receipt("context_parallel_partial_merge", receipt)?;
                    let merged = download(session, &out, live * HEADS * WIDTH * 2)?
                        .chunks_exact(2)
                        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
                        .collect::<Vec<_>>();
                    let inputs = gathered.owned(layout, live);
                    let label = format!(
                        "{} {case} {layout_name} live {live} owner {owner}",
                        entry.name
                    );
                    if case == "equal" {
                        let views = inputs
                            .iter()
                            .map(|(m, l, o)| PartialsF32 {
                                maximum: m,
                                denominator: l,
                                numerator: o,
                            })
                            .collect::<Vec<_>>();
                        let expected =
                            oracle::partial_merge_f32(&views).map_err(|e| e.0.to_string())?;
                        if let Some(index) = merged.iter().zip(&expected).position(|(a, e)| a != e)
                        {
                            return Err(format!("{label}: element {index} differs bitwise"));
                        }
                    }
                    check_f64(&label, &merged, &inputs)?;
                }
            }
        }
        Ok(())
    })
    .expect("the context-parallel partial merge holds its references");
}

/// Key list and appended position localization equal the exact reference
/// at group sizes 2 and 4 for every group rank, from position 0 to
/// 1,048,575 with holes.
#[test]
fn key_localization_is_exact() {
    gpu_test(1 << 30, |session| {
        let module = module(session)?;
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for entry in shapes::lab_entries()
            .into_iter()
            .filter(|entry| matches!(entry.kind, Kind::LocalizeKeys { .. }))
        {
            let Kind::LocalizeKeys { ranks } = entry.kind else {
                unreachable!("a localization entry");
            };
            let list = entry.geometry.list_width as usize;
            let function = module.function(&entry.symbol()).map_err(err)?;
            let ContextParallelQuery::LocalizeKeys { page, .. } = entry.query() else {
                unreachable!("a localization entry");
            };
            let pages =
                ContextParallelPagesV1::new(ranks as usize, page as usize, 4).map_err(err)?;
            for (live, prepared) in [(1_usize, 1_usize), (29, 32), (2048, 2048)] {
                let lists = (0..prepared * list)
                    .map(|i| match i % 11 {
                        0 => -1,
                        1 => (i % (4 * PAGE)) as i32,
                        2 if list == 1 => i as i32,
                        _ => (rng.next() % 1_048_576) as i32,
                    })
                    .collect::<Vec<i32>>();
                let input = upload(
                    session,
                    &lists
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )?;
                let live_buffer = upload(session, &(live as u32).to_ne_bytes())?;
                let out = session.allocate(prepared * list * 4)?;
                for rank in 0..ranks {
                    entry.check_rank(rank).map_err(|e| e.0.to_string())?;
                    let plan = entry.plan(prepared as u32).map_err(|e| e.0.to_string())?;
                    let launch = || {
                        function
                            .launch(
                                &session.stream,
                                launch::config(plan)?,
                                &[
                                    KernelArgument::Buffer(&input),
                                    KernelArgument::Buffer(&live_buffer),
                                    KernelArgument::Buffer(&out),
                                    KernelArgument::U32(rank),
                                ],
                            )
                            .map_err(err)
                    };
                    let replay = Replay::new(session).output_with(
                        "local_lists",
                        &out,
                        WriteMap::of(DType::I32).prefix(live * list)?,
                        &LIST_FILL,
                    );
                    let mut receipt = BTreeMap::from([
                        ("entry".into(), json!(entry.name)),
                        ("live".into(), json!(live)),
                        ("rank".into(), json!(rank)),
                    ]);
                    replay_all(session, replay, &live_buffer, launch, &mut receipt)?;
                    session.receipt("context_parallel_localize_keys", receipt)?;
                    let actual = download(session, &out, live * list * 4)?
                        .chunks_exact(4)
                        .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                        .collect::<Vec<_>>();
                    let expected =
                        oracle::localize_keys(entry.query(), rank, &lists[..live * list])
                            .map_err(|e| e.0.to_string())?;
                    if actual != expected {
                        return Err(format!(
                            "{} rank {rank} live {live}: lists differ",
                            entry.name
                        ));
                    }
                    // The reference agrees with the host's page ownership.
                    for (&global, &local) in lists.iter().zip(&expected).take(4096) {
                        if global >= 0 {
                            let owner = pages.owner(global as usize) == rank as usize;
                            if owner != (local >= 0)
                                || (owner
                                    && pages.rank_position(rank as usize, local as usize)
                                        != global as usize)
                            {
                                return Err(format!("position {global}: ownership differs"));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    })
    .expect("context-parallel key localization is exact");
}

/// The lab entries plan their served buckets on the host.
#[test]
fn lab_entries_plan_their_buckets() {
    let entries = shapes::lab_entries();
    assert_eq!(entries.len(), 6);
    for entry in &entries {
        entry.contract().unwrap();
        for rows in [1, 6, 32, 2048, 8192] {
            entry.plan(rows).unwrap();
        }
    }
    let merge = entry(Kind::PartialMerge { ranks: 4 });
    assert_eq!(merge.plan(32).unwrap().grid, [32, 16, 1]);
    assert_eq!(merge.segment_floats(32, 64), 32 * 64 * 514);
}

/// The merge gate is half a BF16 ulp plus the stated F32 and exp2 terms: it
/// refuses a BF16 value one ulp from the nearest (the review's case,
/// maxima 0 and 1, denominators 1, numerators 1.99) and accepts the nearest.
#[test]
fn the_merge_gate_refuses_a_second_nearest_bf16_value() {
    let rank = |m: f32| (vec![m], vec![1.0_f32], vec![1.99_f32; WIDTH]);
    let inputs = [rank(0.0), rank(1.0)];
    let nearest = f32_to_bf16(1.99);
    assert_eq!(bf16_to_f32(nearest), 1.992_187_5);
    check_f64("nearest", &vec![nearest; WIDTH], &inputs).unwrap();
    let below = f32_to_bf16(1.984_375);
    assert!(check_f64("one ulp below", &vec![below; WIDTH], &inputs).is_err());
    let above = f32_to_bf16(2.0);
    assert!(check_f64("one ulp above", &vec![above; WIDTH], &inputs).is_err());
}
