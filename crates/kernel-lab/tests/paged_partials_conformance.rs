//! Paged attention partials (`paged_partials_shapes.rs`) through the
//! declarative conformance runner (`kernel_lab::conformance`): BF16 verify
//! rows of three items and prefill rows of one, over shuffled pages of a
//! larger pool, with the block tables rerun on their pages moved to the top
//! of the pool. The keys are one-hot (`common/hot_keys.rs`), so the oracle
//! gives every partial's exact bits, and the maxima and numerators witness
//! the high-id rerun: the moved pages give the first run's bits, the old IDs
//! over the moved pool other keys' scores and values. A -1 page traps, so
//! the tables declare no hole row. The verify rows rerun over a pool of
//! [`hot_keys::WIDE_PAGES`] too, whose rerun moves their pages to IDs past
//! 16 bits and byte offsets past 2^31; the oracle shows that the IDs
//! narrowed to 16 bits change its outputs (`narrowing_checked`).
//!
//! The partial merge (`partial_merge{N}`, MERGE_CUBIN) folds such partials
//! across a group and reads no IDs, so its case has no high-id rerun: one
//! rank of each pair is best by a margin no exp survives, others are empty
//! or far below, and one pair has every rank empty, so the oracle gives its
//! BF16 output's exact bits.
//!
//! GPU: `CHOPINFER_KERNEL_GPU=<ordinal> CHOPINFER_KERNEL_REQUIRE=1 cargo test
//! -p kernel-lab --release --locked --offline --test paged_partials_conformance
//! -- --test-threads=1 --nocapture`

#[path = "../paged_partials_shapes.rs"]
mod shapes;

#[path = "common/hot_keys.rs"]
mod hot_keys;

use std::collections::BTreeMap;
use std::sync::atomic::Ordering::Relaxed;

use hot_keys::{
    HIGHEST, NARROWED, PagedShape, WIDE_PAGES, bf16_bytes, bf16_values, f32_bytes, hot, i32_bytes,
    key_row, narrowing_checked, partial_outputs, pool, query_row, value_row,
};
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{Argument, DType};
use kernel_lab::conformance::{Case, Expected, Holes, Images, Rows, Write};
use kernel_lab::{LabResult, gpu_test, latent_shapes};
use kernels_sm120::context_parallel::{self, Kind, MergeLayout};
use kernels_sm120::paged_partials::Entry;
use serde_json::json;

const HEADS: usize = (shapes::GEOMETRY.q_heads * shapes::GEOMETRY.kv_heads) as usize;
const DIM: usize = shapes::GEOMETRY.head_dim as usize;
const PAGE: usize = shapes::GEOMETRY.page as usize;
/// Pages of the pool, more than the items use, so every page can move up.
const POOL_PAGES: usize = 10;
/// A page and 512 decoy bytes.
const STRIDE: usize = PAGE * DIM * 2 + 512;
const SHAPE: PagedShape = PagedShape {
    heads: HEADS,
    width: DIM,
    page: PAGE,
    stride: STRIDE,
    scale: 0.0625,
};

/// One entry's rows over a paged cache.
struct Paged {
    label: &'static str,
    entry: Entry,
    /// Each row's item and position.
    rows: Vec<(usize, usize)>,
    /// Each item's pages, all low in the pool.
    tables: Vec<Vec<i32>>,
    /// Pages of the K and V pools.
    pool_pages: usize,
}

impl Paged {
    fn entry(phase: &str) -> Entry {
        let name = format!("{}_bf16_{phase}", shapes::PROFILE.tag());
        shapes::lab_entries()
            .into_iter()
            .find(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("no lab entry {name}"))
    }

    /// Verify rows: the last four tokens of a 154-token item, then one
    /// decode row each of items of 71 and 6 tokens.
    fn verify() -> Self {
        Self {
            label: "verify",
            entry: Self::entry("verify"),
            rows: vec![(0, 150), (0, 151), (0, 152), (0, 153), (1, 70), (2, 5)],
            tables: vec![vec![3, 1, 4], vec![0, 5], vec![2]],
            pool_pages: POOL_PAGES,
        }
    }

    /// The verify rows over a pool of [`WIDE_PAGES`].
    fn wide() -> Self {
        Self {
            label: "wide verify",
            pool_pages: WIDE_PAGES,
            ..Self::verify()
        }
    }

    /// Prefill rows: the last eight tokens of a 200-token item.
    fn prefill() -> Self {
        Self {
            label: "prefill",
            entry: Self::entry("prefill"),
            rows: (192..200).map(|position| (0, position)).collect(),
            tables: vec![vec![3, 0, 2, 1]],
            pool_pages: POOL_PAGES,
        }
    }

    fn max_pages(&self) -> usize {
        self.tables.iter().map(Vec::len).max().unwrap_or(0) + 1
    }

    fn case(&self) -> Case<'_> {
        let capacity = self.rows.len();
        let max_pages = self.max_pages();
        let mut tables = vec![-1; self.tables.len() * max_pages];
        for (item, pages) in self.tables.iter().enumerate() {
            tables[item * max_pages..][..pages.len()].copy_from_slice(pages);
        }
        // Heads alternate signs, and the signs alternate by row, so heads
        // pick both the largest and the smallest code they see.
        let q = (0..capacity)
            .flat_map(|row| (0..HEADS).flat_map(move |head| query_row(DIM, (row + head) % 2 == 1)))
            .collect::<Vec<_>>();
        let items = self.rows.iter().map(|&(item, _)| item as i32);
        let positions = self.rows.iter().map(|&(_, position)| position as i32);
        let stride = STRIDE as u64;
        let pages = self.pool_pages;
        Case::new(&self.entry, capacity as u32)
            .input("q", DType::Bf16, (Rows::Live, HEADS * DIM), bf16_bytes(&q))
            .produced("q")
            .input(
                "k_cache",
                DType::U8,
                (Rows::Fixed(pages), STRIDE),
                pool(pages, PAGE, DIM, STRIDE, key_row),
            )
            .input(
                "v_cache",
                DType::U8,
                (Rows::Fixed(pages), STRIDE),
                pool(pages, PAGE, DIM, STRIDE, value_row),
            )
            .ids(
                "block_tables",
                (Rows::Fixed(self.tables.len()), max_pages),
                &tables,
                &[("k_cache", STRIDE), ("v_cache", STRIDE)],
                Holes::Never,
            )
            .input(
                "row_batch",
                DType::I32,
                (Rows::Live, 1),
                i32_bytes(&items.collect::<Vec<_>>()),
            )
            .input(
                "positions",
                DType::I32,
                (Rows::Live, 1),
                i32_bytes(&positions.collect::<Vec<_>>()),
            )
            .output("out_max", DType::F32, (Rows::Live, HEADS), Write::LiveRows)
            .output("out_den", DType::F32, (Rows::Live, HEADS), Write::LiveRows)
            .output(
                "out_num",
                DType::F32,
                (Rows::Live, HEADS * DIM),
                Write::LiveRows,
            )
            .scalar("max_pages", Argument::U32(max_pages as u32))
            .scalar("k_page_stride_bytes", Argument::U64(stride))
            .scalar("v_page_stride_bytes", Argument::U64(stride))
            .scalar("scale_bits", Argument::U32(SHAPE.scale.to_bits()))
            .scalar("k_scale_bits", Argument::U32(1.0_f32.to_bits()))
            .scalar("v_scale_bits", Argument::U32(1.0_f32.to_bits()))
            .scalar("cp_rank", Argument::U32(0))
            .scalar("cp_size", Argument::U32(1))
            .oracle(move |images, live| {
                narrowing_checked(images, live, "block_tables", |images, live, read| {
                    // A row keeps every position up to its own.
                    partial_outputs(
                        images,
                        live,
                        capacity,
                        SHAPE,
                        max_pages,
                        read,
                        |_, _, position| Ok((0..=position).collect()),
                    )
                })
            })
    }
}

/// Rows of the merge case.
const MERGE_ROWS: usize = 6;

/// A partial merge of `ranks` segments of `MERGE_ROWS` rows exchanged to
/// their owner (`MergeLayout::exchanged`).
struct Merge {
    ranks: usize,
    entry: context_parallel::Entry,
}

impl Merge {
    fn new(ranks: usize) -> Self {
        let entry = shapes::merge_entries()
            .into_iter()
            .find(|entry| {
                entry.kind
                    == Kind::PartialMerge {
                        ranks: ranks as u32,
                    }
            })
            .unwrap_or_else(|| panic!("no partial merge of {ranks} ranks"));
        Self { ranks, entry }
    }

    /// Floats of one rank's segment: `m` and `l` `[rows, heads]`, then `o`
    /// `[rows, heads, width]`.
    fn segment() -> usize {
        MERGE_ROWS * HEADS * (DIM + 2)
    }

    /// Every rank's partials of pair (row, head): rank `(row + head) %
    /// ranks` is best with `l` 2, and the others are empty or 256 or more
    /// below it; pair (1, 2) has every rank empty.
    fn partials(&self) -> Vec<f32> {
        let pairs = MERGE_ROWS * HEADS;
        let mut partials = vec![0.0; self.ranks * Self::segment()];
        for rank in 0..self.ranks {
            let segment = &mut partials[rank * Self::segment()..][..Self::segment()];
            for pair in 0..pairs {
                let (row, head) = (pair / HEADS, pair % HEADS);
                let best = (row + head) % self.ranks;
                let top = 16.0 * (row as f32 - 2.0);
                let (m, l) = if pair == HEADS + 2 || (rank != best && (pair + rank) % 3 == 0) {
                    (f32::NEG_INFINITY, 0.0)
                } else if rank == best {
                    (top, 2.0)
                } else {
                    (top - 256.0 * (rank + 1) as f32, 3.0)
                };
                segment[pair] = m;
                segment[pairs + pair] = l;
                if l > 0.0 {
                    let value = bf16_values(&bf16_bytes(&value_row(pair * self.ranks + rank, DIM)));
                    for (o, value) in segment[2 * pairs + pair * DIM..][..DIM]
                        .iter_mut()
                        .zip(value)
                    {
                        *o = value as f32;
                    }
                }
            }
        }
        partials
    }

    /// The BF16 `O / L` of every live pair: the best rank's weight is 1 and
    /// every other's +0, so the sums are exact.
    fn oracle(ranks: usize, images: &Images, live: u32) -> LabResult<Vec<(String, Expected)>> {
        let partials = images.f32s("partials")?;
        let pairs = MERGE_ROWS * HEADS;
        let mut out = vec![0_u16; pairs * DIM];
        for pair in 0..live as usize * HEADS {
            let segment = |rank: usize| &partials[rank * Self::segment()..][..Self::segment()];
            let live_ranks = (0..ranks)
                .filter(|&rank| segment(rank)[pair] != f32::NEG_INFINITY)
                .collect::<Vec<_>>();
            let maxima = live_ranks
                .iter()
                .map(|&rank| f64::from(segment(rank)[pair]))
                .collect::<Vec<_>>();
            let Some(best) = hot(&maxima, 1.0)? else {
                continue;
            };
            let best = segment(live_ranks[best]);
            let l = best[pairs + pair];
            for (d, out) in out[pair * DIM..][..DIM].iter_mut().enumerate() {
                let o = best[2 * pairs + pair * DIM + d];
                *out = f32_to_bf16(if l == 0.0 { 0.0 } else { o / l });
            }
        }
        Ok(vec![("out".into(), Expected::Bits(bf16_bytes(&out)))])
    }

    fn case(&self) -> Case<'_> {
        let layout = MergeLayout::exchanged(MERGE_ROWS as u32, HEADS as u32);
        self.entry
            .check_merge(MERGE_ROWS as u32, layout)
            .expect("merge scalars");
        let ranks = self.ranks;
        Case::new(&self.entry, MERGE_ROWS as u32)
            .input(
                "partials",
                DType::F32,
                (Rows::Fixed(ranks), Self::segment()),
                f32_bytes(&self.partials()),
            )
            .produced("partials")
            .output(
                "out",
                DType::Bf16,
                (Rows::Live, HEADS * DIM),
                Write::LiveRows,
            )
            .scalar("rank_rows", Argument::U32(layout.rank_rows))
            .scalar("input_heads", Argument::U32(layout.input_heads))
            .scalar("head_offset", Argument::U32(layout.head_offset))
            .oracle(move |images, live| Self::oracle(ranks, images, live))
    }
}

fn merges() -> Vec<Merge> {
    [2, 4].into_iter().map(Merge::new).collect()
}

fn cases() -> Vec<Paged> {
    vec![Paged::verify(), Paged::prefill()]
}

#[test]
fn paged_partial_cases_validate() {
    for paged in cases() {
        paged
            .case()
            .validate()
            .unwrap_or_else(|e| panic!("{}: {e}", paged.label));
    }
}

/// The wide case's high-id rerun moves its pages to the top of the pool,
/// IDs 65630 to 65635, and the IDs narrowed to 16 bits change its outputs.
#[test]
fn wide_paged_partial_case_validates() {
    let paged = Paged::wide();
    paged.case().validate().expect("wide verify");
    assert!(
        NARROWED.load(Relaxed) > 0,
        "no ID past 16 bits was narrowed"
    );
    assert_eq!(HIGHEST.load(Relaxed), WIDE_PAGES as i32 - 1);
}

#[test]
fn partial_merge_cases_validate() {
    for merge in merges() {
        merge
            .case()
            .validate()
            .unwrap_or_else(|e| panic!("merge of {}: {e}", merge.ranks));
    }
}

#[test]
fn paged_partials_conform_gpu() {
    // The wide case's K and V pools, and 1 GiB for the rest.
    let need = 2 * WIDE_PAGES * STRIDE + (1 << 30);
    gpu_test(need as u64, |session| {
        let module = session.module(shapes::CUBIN)?;
        let lab = session.module(&latent_shapes::cubin_name())?;
        for paged in cases().into_iter().chain([Paged::wide()]) {
            let evidence = paged
                .case()
                .run(session, &module, Some(&lab))
                .map_err(|e| format!("{}: {e}", paged.label))?;
            session.receipt(
                "paged_partials_conformance",
                BTreeMap::from([
                    ("case".into(), json!(paged.label)),
                    ("evidence".into(), evidence),
                ]),
            )?;
        }
        let merges_module = session.module(shapes::MERGE_CUBIN)?;
        for merge in merges() {
            let evidence = merge
                .case()
                .run(session, &merges_module, Some(&lab))
                .map_err(|e| format!("merge of {}: {e}", merge.ranks))?;
            session.receipt(
                "paged_partials_conformance",
                BTreeMap::from([
                    ("case".into(), json!(format!("merge of {}", merge.ranks))),
                    ("evidence".into(), evidence),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("paged partials conform");
}
