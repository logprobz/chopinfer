//! One context-parallel latent attention step on the host, in the order and
//! with the buffer layouts a bundle launches it, against attention over
//! every key on one rank, for several query rows.
//!
//! Heads stay split across tensor-parallel ranks (`HEADS` per rank); a group
//! of `N` consecutive ranks splits the keys by page
//! (`ContextParallelPagesV1`). Each rank:
//!
//! 1. all-gathers its group's queries: the output is rank-major
//!    `[N][rows][HEADS][WIDTH]` (`group::gather_arguments` slot order);
//! 2. localizes each row's merged key list (`localize_keys`);
//! 3. runs the heads-`HEADS` partial attention once per source block s,
//!    reading block s at `query_block_offset` as a row-major `[rows, HEADS,
//!    WIDTH]` query, over its own keys, and writes the block's `m`, `l`, `o`
//!    at `partial_block_offsets` of its segment (block-major);
//! 4. receives the partials of its own heads, either by all-gathering every
//!    segment (`MergeLayout::gathered`), by the row exchange of only its
//!    block (`MergeLayout::exchanged`) or by one all-gather per owner
//!    targeting it (`group::owner_gather_arguments`, the same compact
//!    layout);
//! 5. merges with the merge kernel's addressing (`MergeLayout::pair`) and
//!    arithmetic (`partial_merge_f32`).
//!
//! Collectives are emulated by their data movement: an all-gather places
//! member s's shard at slot s of every output, the exchange places source
//! s's block for the owner at slot s of the owner's output.

use kernel_core::codec::bf16_to_f32;
use kernel_core::context_parallel::ContextParallelQuery;
use kernel_core::context_parallel_oracle::{PartialsF32, localize_keys, partial_merge_f32};
use kernels_sm120::context_parallel::group::{GatherState, GroupRank, gather_arguments};
use kernels_sm120::context_parallel::{MergeLayout, partial_block_offsets, query_block_offset};
use runtime_kv::ContextParallelPagesV1;

const TP: usize = 4;
const HEADS: usize = 2;
const WIDTH: usize = 8;
const PAGE: usize = 4;

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// F32 partials of one query row of `WIDTH` over `keys` (local rows, -1
/// holes) of a local latent store `[rows, WIDTH]`: `(m, l, o)`.
fn partial(q: &[f32], keys: &[i32], store: &[f32]) -> (f32, f32, Vec<f32>) {
    let (mut m, mut l, mut o) = (f32::NEG_INFINITY, 0.0_f32, vec![0.0_f32; WIDTH]);
    for &key in keys.iter().filter(|&&key| key >= 0) {
        let k = &store[key as usize * WIDTH..][..WIDTH];
        let s = q
            .iter()
            .zip(k)
            .fold(0.0_f32, |acc, (a, b)| a.mul_add(*b, acc));
        let top = m.max(s);
        let scale = if m == f32::NEG_INFINITY {
            0.0
        } else {
            (m - top).exp()
        };
        let p = (s - top).exp();
        l = l.mul_add(scale, p);
        for (value, key) in o.iter_mut().zip(k) {
            *value = value.mul_add(scale, p * key);
        }
        m = top;
    }
    (m, l, o)
}

/// The merge kernel over `segments` (one per source, slot order) for live
/// rows `0..rows`: each (row, head) reads pair `layout.pair(row, head)`.
fn merge(segments: &[Vec<f32>], layout: MergeLayout, rows: usize) -> Vec<u16> {
    let pairs = layout.pairs() as usize;
    let mut out = Vec::new();
    for row in 0..rows as u32 {
        for head in 0..HEADS as u32 {
            let pair = layout.pair(row, head) as usize;
            let views = segments
                .iter()
                .map(|segment| PartialsF32 {
                    maximum: &segment[pair..pair + 1],
                    denominator: &segment[pairs + pair..pairs + pair + 1],
                    numerator: &segment[2 * pairs + pair * WIDTH..][..WIDTH],
                })
                .collect::<Vec<_>>();
            out.extend(partial_merge_f32(&views).unwrap());
        }
    }
    out
}

#[test]
fn a_group_step_equals_attention_over_every_key() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for (tokens, rows) in [(1_usize, 1_usize), (5, 2), (13, 3), (37, 5), (64, 4)] {
        let latent: Vec<f32> = (0..tokens * WIDTH).map(|_| rng.unit()).collect();
        // Distinct queries per row and head: `[rows][TP * HEADS][WIDTH]`.
        let queries: Vec<f32> = (0..rows * TP * HEADS * WIDTH)
            .map(|_| 2.0 * rng.unit())
            .collect();
        let query =
            |row: usize, head: usize| &queries[(row * TP * HEADS + head) * WIDTH..][..WIDTH];
        // Each row's merged key list: its own causal prefix, then holes.
        let visible = |row: usize| tokens - (rows - 1 - row).min(tokens - 1);
        let width = tokens + 3;
        let lists: Vec<i32> = (0..rows)
            .flat_map(|row| (0..width).map(move |i| if i < visible(row) { i as i32 } else { -1 }))
            .collect();
        // Attention over every visible key, in f64.
        let reference = |row: usize, head: usize, d: usize| {
            let q = query(row, head);
            let scores: Vec<f64> = (0..visible(row))
                .map(|t| {
                    q.iter()
                        .zip(&latent[t * WIDTH..][..WIDTH])
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum()
                })
                .collect();
            let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let l: f64 = scores.iter().map(|s| (s - top).exp()).sum();
            scores
                .iter()
                .enumerate()
                .map(|(t, s)| (s - top).exp() * f64::from(latent[t * WIDTH + d]))
                .sum::<f64>()
                / l
        };
        for size in [1_usize, 2, 4] {
            let pages = ContextParallelPagesV1::new(size, PAGE, TP).unwrap();
            let ranks: Vec<GroupRank> = (0..TP)
                .map(|t| GroupRank::of(t as u32, TP as u32, size as u32).unwrap())
                .collect();
            // Each rank's local store: its own pages, one per stripe.
            let stores: Vec<Vec<f32>> = ranks
                .iter()
                .map(|rank| {
                    let r = rank.rank as usize;
                    let held = pages.rank_tokens(r, tokens);
                    let mut store = vec![f32::NAN; held.max(1) * WIDTH];
                    for row in 0..held {
                        let t = pages.rank_position(r, row);
                        store[row * WIDTH..][..WIDTH]
                            .copy_from_slice(&latent[t * WIDTH..][..WIDTH]);
                    }
                    store
                })
                .collect();
            // Each rank's own queries, row-major `[rows][HEADS][WIDTH]`.
            let own_queries: Vec<Vec<f32>> = (0..TP)
                .map(|t| {
                    (0..rows)
                        .flat_map(|row| (0..HEADS).flat_map(move |h| query(row, t * HEADS + h)))
                        .copied()
                        .collect()
                })
                .collect();
            let states: Vec<GatherState> = (0..TP as u64)
                .map(|t| GatherState {
                    output: 0x1000 * (t + 1),
                    flags: 0x10 * (t + 1),
                    epoch: t,
                })
                .collect();
            // 1. Query all-gather: `[N][rows][HEADS][WIDTH]`.
            let gathered_queries: Vec<Vec<f32>> = ranks
                .iter()
                .map(|rank| {
                    gather_arguments(*rank, &states, 1, (rows * HEADS * WIDTH) as u64).unwrap();
                    rank.members()
                        .flat_map(|member| own_queries[member as usize].iter().copied())
                        .collect()
                })
                .collect();
            // 2. Localize, 3. one partial attention launch per source block.
            let segment_floats = size * rows * HEADS * (WIDTH + 2);
            let segments: Vec<Vec<f32>> = ranks
                .iter()
                .enumerate()
                .map(|(t, rank)| {
                    let keys = if size == 1 {
                        lists.clone()
                    } else {
                        let localize = ContextParallelQuery::LocalizeKeys {
                            ranks: size as u32,
                            page: PAGE as u32,
                            width: width as u32,
                            capacity: rows as u32,
                        };
                        localize_keys(localize, rank.rank, &lists).unwrap()
                    };
                    let mut segment = vec![f32::NAN; segment_floats];
                    for block in 0..size as u32 {
                        let q = query_block_offset(rows as u32, HEADS as u32, WIDTH as u32, block)
                            as usize;
                        let [m, l, o] = partial_block_offsets(
                            rows as u32,
                            HEADS as u32,
                            WIDTH as u32,
                            size as u32,
                            block,
                        )
                        .map(|offset| offset as usize);
                        for row in 0..rows {
                            for h in 0..HEADS {
                                let pair = row * HEADS + h;
                                let (pm, pl, po) = partial(
                                    &gathered_queries[t][q + pair * WIDTH..][..WIDTH],
                                    &keys[row * width..][..width],
                                    &stores[t],
                                );
                                segment[m + pair] = pm;
                                segment[l + pair] = pl;
                                segment[o + pair * WIDTH..][..WIDTH].copy_from_slice(&po);
                            }
                        }
                    }
                    segment
                })
                .collect();
            // 4. Partials to the owner, 5. merge the owner's heads.
            for (t, rank) in ranks.iter().enumerate() {
                let sources: Vec<Vec<f32>> = rank
                    .members()
                    .map(|member| segments[member as usize].clone())
                    .collect();
                let gathered = merge(
                    &sources,
                    MergeLayout::gathered(rows as u32, HEADS as u32, size as u32, rank.rank),
                    rows,
                );
                // The exchange delivers only the owner's block of each segment.
                let [m, l, o] = partial_block_offsets(
                    rows as u32,
                    HEADS as u32,
                    WIDTH as u32,
                    size as u32,
                    rank.rank,
                )
                .map(|offset| offset as usize);
                let pairs = rows * HEADS;
                let exchanged_segments: Vec<Vec<f32>> = sources
                    .iter()
                    .map(|segment| {
                        [
                            &segment[m..m + pairs],
                            &segment[l..l + pairs],
                            &segment[o..o + pairs * WIDTH],
                        ]
                        .concat()
                    })
                    .collect();
                let exchanged = merge(
                    &exchanged_segments,
                    MergeLayout::exchanged(rows as u32, HEADS as u32),
                    rows,
                );
                assert_eq!(gathered, exchanged, "size {size} rank {t}");
                // The owner gathers: one all-gather per owner o targeting
                // o, each member's input its destination-major send block
                // o, stored at `output[o] + source * block` (compact, the
                // exchanged view's layout).
                let block = rows * HEADS * (WIDTH + 2);
                let mut compact = vec![f32::NAN; size * block];
                for (source, segment) in sources.iter().enumerate() {
                    let send = [
                        &segment[m..m + pairs],
                        &segment[l..l + pairs],
                        &segment[o..o + pairs * WIDTH],
                    ]
                    .concat();
                    compact[source * block..][..block].copy_from_slice(&send);
                }
                let owned = merge(
                    &compact
                        .chunks_exact(block)
                        .map(<[f32]>::to_vec)
                        .collect::<Vec<_>>(),
                    MergeLayout::exchanged(rows as u32, HEADS as u32),
                    rows,
                );
                assert_eq!(owned, exchanged, "size {size} rank {t}");
                for (i, &bits) in gathered.iter().enumerate() {
                    let (row, head, d) = (i / (HEADS * WIDTH), i / WIDTH % HEADS, i % WIDTH);
                    let expected = reference(row, t * HEADS + head, d);
                    let actual = f64::from(bf16_to_f32(bits));
                    assert!(
                        (actual - expected).abs() <= expected.abs() * 2_f64.powi(-8) + 1e-5,
                        "tokens {tokens} rows {rows} size {size} rank {t} row {row} head {head}: \
                         {actual} vs {expected}"
                    );
                }
            }
        }
    }
}
