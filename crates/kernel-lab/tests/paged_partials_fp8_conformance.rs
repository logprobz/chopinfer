//! Paged attention partials (`paged_partials_shapes.rs`) over FP8 E4M3 KV
//! through the declarative conformance runner (`kernel_lab::conformance`):
//! the FP8 verify and prefill entries over the rows and tables of the BF16
//! cases (`paged_partials_conformance.rs`), with E4M3 K and V pools of
//! 1-byte elements, so a token's row takes half the BF16 bytes. The keys
//! are FP8 one-hot (`common/hot_fp8.rs`) at K scale 2 and V scale 1/4, so
//! the oracle gives every partial's exact bits, and the maxima and
//! numerators witness the high-id rerun: the moved pages give the first
//! run's bits, the old IDs over the moved pool other keys' scores and
//! values. A -1 page traps, so the tables declare no hole row. The verify
//! rows rerun over a pool of [`hot_keys::WIDE_PAGES`] too, whose
//! rerun moves their pages to IDs past 16 bits; the stride keeps the BF16
//! cases' 33280 bytes (a 16 KiB page of codes, then decoys), so their byte
//! offsets pass 2^31 as well. The oracle shows that the IDs narrowed to 16
//! bits change its outputs (`narrowing_checked`).
//!
//! GPU: `CHOPINFER_KERNEL_GPU=<ordinal> CHOPINFER_KERNEL_REQUIRE=1 cargo test
//! -p kernel-lab --release --locked --offline --test
//! paged_partials_fp8_conformance -- --test-threads=1 --nocapture`

// The merge entries are the BF16 cases' (`paged_partials_conformance.rs`).
#[allow(dead_code)]
#[path = "../paged_partials_shapes.rs"]
mod shapes;

#[path = "common/hot_keys.rs"]
mod hot_keys;

#[path = "common/hot_fp8.rs"]
mod hot_fp8;

use std::collections::BTreeMap;
use std::sync::atomic::Ordering::Relaxed;

use hot_fp8::{K_SCALE, V_SCALE, key_row, partial_outputs, pool, query_row, value_row};
use hot_keys::{
    HIGHEST, NARROWED, PagedShape, WIDE_PAGES, bf16_bytes, i32_bytes, narrowing_checked,
};
use kernel_core::contract::{Argument, DType};
use kernel_lab::conformance::{Case, Holes, Rows, Write};
use kernel_lab::{gpu_test, latent_shapes};
use kernels_sm120::paged_partials::Entry;
use serde_json::json;

const HEADS: usize = (shapes::GEOMETRY.q_heads * shapes::GEOMETRY.kv_heads) as usize;
const DIM: usize = shapes::GEOMETRY.head_dim as usize;
const PAGE: usize = shapes::GEOMETRY.page as usize;
/// Pages of the pool, more than the items use, so every page can move up.
const POOL_PAGES: usize = 10;
/// The BF16 cases' stride: a page of E4M3 codes and 16896 decoy bytes.
const STRIDE: usize = PAGE * DIM * 2 + 512;
const SHAPE: PagedShape = PagedShape {
    heads: HEADS,
    width: DIM,
    page: PAGE,
    stride: STRIDE,
    scale: 0.0625,
};

/// One FP8 entry's rows over a paged cache.
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
        let name = format!("{}_fp8_{phase}", shapes::PROFILE.tag());
        shapes::lab_entries()
            .into_iter()
            .find(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("no lab entry {name}"))
    }

    /// Verify rows: the last four tokens of a 154-token item, then one
    /// decode row each of items of 71 and 6 tokens.
    fn verify() -> Self {
        Self {
            label: "fp8 verify",
            entry: Self::entry("verify"),
            rows: vec![(0, 150), (0, 151), (0, 152), (0, 153), (1, 70), (2, 5)],
            tables: vec![vec![3, 1, 4], vec![0, 5], vec![2]],
            pool_pages: POOL_PAGES,
        }
    }

    /// The verify rows over a pool of [`WIDE_PAGES`].
    fn wide() -> Self {
        Self {
            label: "wide fp8 verify",
            pool_pages: WIDE_PAGES,
            ..Self::verify()
        }
    }

    /// Prefill rows: the last eight tokens of a 200-token item.
    fn prefill() -> Self {
        Self {
            label: "fp8 prefill",
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
            .scalar("k_scale_bits", Argument::U32(K_SCALE.to_bits()))
            .scalar("v_scale_bits", Argument::U32(V_SCALE.to_bits()))
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

fn cases() -> Vec<Paged> {
    vec![Paged::verify(), Paged::prefill()]
}

#[test]
fn fp8_paged_partial_cases_validate() {
    for paged in cases() {
        assert_eq!(paged.entry.kv, DType::Fp8E4M3, "{}", paged.label);
        paged
            .case()
            .validate()
            .unwrap_or_else(|e| panic!("{}: {e}", paged.label));
    }
}

/// The wide case's high-id rerun moves its pages to the top of the pool,
/// IDs 65630 to 65635 at byte offsets past 2^31, and the IDs narrowed to 16
/// bits change its outputs.
#[test]
fn wide_fp8_paged_partial_case_validates() {
    let paged = Paged::wide();
    assert!(((WIDE_PAGES - 6) * STRIDE) as u64 > 1 << 31);
    paged.case().validate().expect("wide fp8 verify");
    assert!(
        NARROWED.load(Relaxed) > 0,
        "no ID past 16 bits was narrowed"
    );
    assert_eq!(HIGHEST.load(Relaxed), WIDE_PAGES as i32 - 1);
}

#[test]
fn fp8_paged_partials_conform_gpu() {
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
                "paged_partials_fp8_conformance",
                BTreeMap::from([
                    ("case".into(), json!(paged.label)),
                    ("evidence".into(), evidence),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("fp8 paged partials conform");
}
