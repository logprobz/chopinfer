//! The latent KV write (`latent::Entry` `kv_write_{bf16,fp8}`, kernel-core
//! `LatentAttentionQuery::KvWrite`) through the declarative conformance
//! runner (`kernel_lab::conformance`), at the lab's served width of 512 and
//! at width 128 over a wide pool.
//!
//! The pool is state (`common/kv_pool.rs`): pages of old contents and
//! decoy bytes, of which a launch may change only the slots of its live
//! rows, so untouched pages, decoys and the slots of rows past the live
//! count must keep their bytes. Six rows write three items' pages, one row
//! sits at a page's last slot and the next at the next page's first, and one
//! row has position -1 (a row the step pads), which writes nothing. The
//! positions are ID inputs without pools that rerun with the last live row
//! -1 (`Holes::LastLive`); the page tables index the pool and rerun with
//! their pages moved to the top of it. A -1 page lies outside the pool, so
//! the tables declare no hole row.
//!
//! The oracle is exact. BF16 rows are copies. FP8 rows take one F32 scale
//! per 128-element block, the block's largest magnitude divided by 448 (1
//! for an all-zero block), and codes `E4M3(x / scale)`. Each block's
//! elements are E4M3 values times a power of two 2^e (`kv_pool::e4m3_value`,
//! magnitudes at most 7.5 2^e) with one element of magnitude 448 2^e, so the
//! scale is exactly 2^e and every code is exact; one block of some rows is
//! all zeros. The oracle refuses a scale that is not a power of two or a
//! code E4M3 does not hold exactly. The wide cases run over a pool of
//! [`hot_keys::WIDE_PAGES`] pages at a stride of 33280 bytes, so the
//! high-id rerun's pages lie past 16 bits and their byte offsets past 2^31
//! (below 2^32, so an unsigned 32-bit offset is not told apart). The oracle
//! shows that the IDs narrowed to 16 bits change the pool, and
//! `wide_latent_kv_write_cases_validate` requires a writer that narrows
//! them to fail the high-id rerun's state check.
//!
//! GPU: `CHOPINFER_KERNEL_GPU=<ordinal> CHOPINFER_KERNEL_REQUIRE=1 cargo test
//! -p kernel-lab --release --locked --offline --test
//! latent_kv_write_conformance -- --test-threads=1 --nocapture`

#[path = "common/hot_keys.rs"]
mod hot_keys;

#[path = "common/hot_fp8.rs"]
mod hot_fp8;

#[path = "common/kv_pool.rs"]
mod kv_pool;

use std::collections::BTreeMap;
use std::sync::atomic::Ordering::Relaxed;

use hot_keys::{HIGHEST, NARROWED, WIDE_PAGES, bf16_bytes, i32_bytes, low16, narrowing_checked};
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::{Argument, DType};
use kernel_core::latent_attention::{LatentAttentionQuery, LatentKvFormat};
use kernel_lab::conformance::{Case, Expected, Holes, Images, Rows, Write};
use kernel_lab::poison::WriteMap;
use kernel_lab::{LabResult, gpu_test, latent_shapes};
use kernels_sm120::latent::{Entry, Query};
use kv_pool::{bf16_exact, e4m3_value, mix, old_pool, put, ranges, slots};
use serde_json::json;

const PAGE: usize = latent_shapes::GEOMETRY.page_size as usize;
/// Elements per FP8 scale.
const BLOCK: usize = 128;
/// Pages of the pool, more than the tables use, so every page can move up.
const POOL_PAGES: usize = 10;
/// Each row's item and position; -1 writes nothing.
const ROWS: [(i32, i32); 6] = [(0, 127), (0, 128), (1, -1), (1, 70), (2, 5), (0, 0)];
/// Each item's pages, all low in the pool.
const TABLES: [&[i32]; 3] = [&[3, 1, 4], &[0, 5], &[2]];
/// Table width: one more than the longest table, so every table ends in -1.
const MAX_PAGES: usize = 4;

/// One latent KV write entry over a pool.
struct Writer {
    label: &'static str,
    entry: Entry,
    latent: usize,
    fp8: bool,
    pool_pages: usize,
    stride: usize,
}

impl Writer {
    /// The lab entry `kv_write_{dtype}`, at the served width or, with
    /// `width`, its lab-only `_w{width}` form.
    fn new(label: &'static str, dtype: &str, width: Option<u32>) -> Self {
        let name = match width {
            Some(width) => format!("{}_kv_write_{dtype}_w{width}", latent_shapes::PROFILE.tag()),
            None => format!("{}_kv_write_{dtype}", latent_shapes::PROFILE.tag()),
        };
        let entry = latent_shapes::catalog()
            .into_iter()
            .find(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("no lab entry {name}"));
        let Query::Latent(LatentAttentionQuery::KvWrite {
            latent,
            page_size,
            kv,
            ..
        }) = entry.query
        else {
            panic!("{name} is not a latent KV write");
        };
        assert_eq!(page_size as usize, PAGE);
        let fp8 = kv == LatentKvFormat::Fp8E4M3Block128;
        // A page of the widest format and 1024 decoy bytes, or 33280 bytes
        // at width 128, as the latent attention cases' pools.
        let stride = if latent == 128 { 33280 } else { 66560 };
        assert!(kv.page_bytes(latent, page_size) as usize <= stride - 512);
        Self {
            label,
            entry,
            latent: latent as usize,
            fp8,
            pool_pages: POOL_PAGES,
            stride,
        }
    }

    /// Every count of `poison::live_counts`, or a wide case's
    /// (`kv_pool::wide_counts`).
    fn counts(&self) -> Vec<u32> {
        let capacity = ROWS.len() as u32;
        if self.pool_pages == WIDE_PAGES {
            kv_pool::wide_counts(capacity)
        } else {
            kernel_lab::poison::live_counts(capacity)
        }
    }

    fn wide(self, label: &'static str) -> Self {
        Self {
            label,
            pool_pages: WIDE_PAGES,
            ..self
        }
    }

    fn format(&self) -> LatentKvFormat {
        if self.fp8 {
            LatentKvFormat::Fp8E4M3Block128
        } else {
            LatentKvFormat::Bf16
        }
    }

    fn page_bytes(&self) -> usize {
        self.format().page_bytes(self.latent as u32, PAGE as u32) as usize
    }

    /// The pool's dtype: BF16 rows, or E4M3 codes and F32 scales as bytes.
    fn pool_dtype(&self) -> DType {
        if self.fp8 { DType::U8 } else { DType::Bf16 }
    }

    /// Each written slot's byte spans: the row, and for FP8 its scales.
    fn spans(&self, page: usize, slot: usize) -> Vec<(usize, usize)> {
        let base = page * self.stride;
        let blocks = self.latent / BLOCK;
        if self.fp8 {
            vec![
                (base + slot * self.latent, self.latent),
                (base + PAGE * self.latent + slot * blocks * 4, blocks * 4),
            ]
        } else {
            vec![(base + slot * self.latent * 2, self.latent * 2)]
        }
    }

    /// The pool after a launch at `live`, its tables read through `read`.
    fn pool_after(&self, images: &Images, live: u32, read: fn(i32) -> i32) -> LabResult<Vec<u8>> {
        let mut pool = images.bytes("pool")?.to_vec();
        let latent = images.u16s("latent")?;
        for s in slots(images, live, "positions", PAGE, MAX_PAGES, read)? {
            let row = &latent[s.row * self.latent..][..self.latent];
            let spans = self.spans(s.page, s.slot);
            if !self.fp8 {
                put(&mut pool, spans[0].0, &bf16_bytes(row))?;
                continue;
            }
            for (block, values) in row.chunks_exact(BLOCK).enumerate() {
                let (codes, scale) =
                    fp8_block(values).map_err(|e| format!("row {} block {block}: {e}", s.row))?;
                put(&mut pool, spans[0].0 + block * BLOCK, &codes)?;
                put(&mut pool, spans[1].0 + block * 4, &scale.to_le_bytes())?;
            }
        }
        Ok(pool)
    }

    fn case(&self) -> Case<'_> {
        let capacity = ROWS.len();
        let mut tables = vec![-1; TABLES.len() * MAX_PAGES];
        for (item, pages) in TABLES.iter().enumerate() {
            tables[item * MAX_PAGES..][..pages.len()].copy_from_slice(pages);
        }
        let latent = (0..capacity)
            .flat_map(|row| latent_row(row, self.latent))
            .collect::<Vec<_>>();
        let items = ROWS.map(|(item, _)| item);
        let positions = ROWS.map(|(_, position)| position);
        let element = if self.fp8 { 1 } else { 2 };
        let dtype = self.pool_dtype();
        Case::new(&self.entry, capacity as u32)
            .live_counts(self.counts())
            .input(
                "latent",
                DType::Bf16,
                (Rows::Live, self.latent),
                bf16_bytes(&latent),
            )
            .produced("latent")
            .input("row_batch", DType::I32, (Rows::Live, 1), i32_bytes(&items))
            .ids(
                "positions",
                (Rows::Live, 1),
                &positions,
                &[],
                Holes::LastLive,
            )
            .ids(
                "block_tables",
                (Rows::Fixed(TABLES.len()), MAX_PAGES),
                &tables,
                &[("pool", self.stride)],
                Holes::Never,
            )
            .state(
                "pool",
                dtype,
                (Rows::Fixed(self.pool_pages), self.stride / element),
                old_pool(self.pool_pages, self.page_bytes(), self.stride, 0),
                Write::Map(Box::new(move |images, live| {
                    let spans = slots(images, live, "positions", PAGE, MAX_PAGES, |id| id)?
                        .into_iter()
                        .flat_map(|s| self.spans(s.page, s.slot));
                    WriteMap::of(dtype).byte_ranges(ranges(spans))
                })),
                |_, _| Ok(Vec::new()),
            )
            .scalar("max_pages", Argument::U32(MAX_PAGES as u32))
            .scalar("page_stride_bytes", Argument::U64(self.stride as u64))
            .oracle(move |images, live| {
                narrowing_checked(images, live, "block_tables", |images, live, read| {
                    Ok(vec![(
                        "pool".into(),
                        Expected::Bits(self.pool_after(images, live, read)?),
                    )])
                })
            })
    }
}

/// Row `row`'s latent of `width` BF16: per 128-element block a power of two
/// 2^e (e from -4 to 4), one element of magnitude 448 2^e and the others
/// E4M3 values times 2^e; block b of a row with `(row + b) % 7 == 3` is all
/// zeros.
fn latent_row(row: usize, width: usize) -> Vec<u16> {
    (0..width / BLOCK)
        .flat_map(|block| {
            let h = mix((row * 64 + block) as u64);
            let zero = (row + block) % 7 == 3;
            let scale = 2f32.powi((h % 9) as i32 - 4);
            let peak = (h >> 8) as usize % BLOCK;
            (0..BLOCK).map(move |j| {
                let value = if zero {
                    0.0
                } else if j == peak {
                    if (h >> 20) & 1 == 0 { 448.0 } else { -448.0 }
                } else {
                    e4m3_value(mix(h ^ j as u64))
                };
                bf16_exact(value * scale).expect("latent values are BF16")
            })
        })
        .collect()
}

/// One block's E4M3 codes and F32 scale, as the kernel computes them: the
/// largest magnitude over 448 (1 for an all-zero block), and each code
/// `E4M3(x / scale)`. Refused unless the scale is a power of two and every
/// code is exact.
fn fp8_block(values: &[u16]) -> LabResult<(Vec<u8>, f32)> {
    let values = values.iter().map(|&bits| bf16_to_f32(bits));
    let amax = values.clone().fold(0.0f32, |amax, x| amax.max(x.abs()));
    let scale = if amax == 0.0 { 1.0 } else { amax / 448.0 };
    if scale.to_bits() & 0x007f_ffff != 0 || scale * 448.0 != amax && amax != 0.0 {
        return Err(format!(
            "scale {scale} of maximum {amax} is not an exact power of two"
        ));
    }
    let codes = values
        .map(|x| hot_fp8::e4m3(f64::from(x / scale)))
        .collect::<LabResult<Vec<_>>>()?;
    Ok((codes, scale))
}

fn cases() -> Vec<Writer> {
    vec![
        Writer::new("bf16", "bf16", None),
        Writer::new("fp8", "fp8", None),
    ]
}

fn wide_cases() -> Vec<Writer> {
    vec![
        Writer::new("bf16 w128", "bf16", Some(128)).wide("wide bf16 w128"),
        Writer::new("fp8 w128", "fp8", Some(128)).wide("wide fp8 w128"),
    ]
}

#[test]
fn latent_kv_write_cases_validate() {
    for writer in cases() {
        writer
            .case()
            .validate()
            .unwrap_or_else(|e| panic!("{}: {e}", writer.label));
    }
}

/// The wide cases' high-id reruns move their pages to IDs 65630 to 65635,
/// at byte offsets past 2^31, and the IDs narrowed to 16 bits change the
/// pool: a writer that narrows them fails the high-id rerun's state check
/// and only that. One test, so the wide pools are built one at a time.
#[test]
fn wide_latent_kv_write_cases_validate() {
    for writer in wide_cases() {
        assert!(((WIDE_PAGES - 6) * writer.stride) as u64 > 1 << 31);
        let case = writer.case();
        case.validate()
            .unwrap_or_else(|e| panic!("{}: {e}", writer.label));
        let narrowing = case
            .refusing_runs(|images, live| {
                Ok(vec![(
                    "pool".into(),
                    writer.pool_after(images, live, low16)?,
                )])
            })
            .unwrap_or_else(|e| panic!("{}: {e}", writer.label));
        assert_eq!(narrowing, ["high block_tables"], "{}", writer.label);
    }
    assert!(
        NARROWED.load(Relaxed) > 0,
        "no ID past 16 bits was narrowed"
    );
    assert_eq!(HIGHEST.load(Relaxed), WIDE_PAGES as i32 - 1);
}

/// The FP8 rows exercise the zero-block scale and scales that vary by row
/// and block, and the oracle refuses inexact blocks.
#[test]
fn fp8_blocks_are_exact_and_inexact_ones_refused() {
    let mut scales = std::collections::BTreeSet::new();
    let mut zero = false;
    for row in 0..ROWS.len() {
        let values = latent_row(row, 512);
        for block in values.chunks_exact(BLOCK) {
            let (codes, scale) = fp8_block(block).expect("an exact block");
            zero |= codes.iter().all(|&code| code == 0) && scale == 1.0;
            scales.insert(scale.to_bits());
        }
    }
    assert!(zero, "no all-zero block");
    assert!(scales.len() >= 6, "scales {scales:?}");
    // A maximum of 300 gives a scale 300 / 448, not a power of two.
    let mut block = latent_row(0, 128);
    let magnitude = |bits: &u16| bf16_to_f32(*bits).abs();
    let peak = (0..BLOCK)
        .max_by(|&a, &b| magnitude(&block[a]).total_cmp(&magnitude(&block[b])))
        .expect("a peak");
    let scale = bf16_to_f32(block[peak]).abs() / 448.0;
    block[peak] = bf16_exact(300.0 * scale).unwrap();
    assert!(fp8_block(&block).is_err());
    // A value of 17 2^e is no E4M3 code at the block's scale.
    let mut block = latent_row(0, 128);
    let other = (peak + 1) % BLOCK;
    block[other] = bf16_exact(17.0 * scale).unwrap();
    assert!(fp8_block(&block).unwrap_err().contains("not an E4M3 value"));
}

#[test]
fn latent_kv_writes_conform_gpu() {
    // A wide case's pool, 2.03 GiB, and 1 GiB for the rest; the cases run
    // one at a time and free their buffers.
    let need = WIDE_PAGES * 33280 + (1 << 30);
    gpu_test(need as u64, |session| {
        let module = session.module(&latent_shapes::cubin_name())?;
        for writer in cases().into_iter().chain(wide_cases()) {
            let evidence = writer
                .case()
                .run(session, &module, Some(&module))
                .map_err(|e| format!("{}: {e}", writer.label))?;
            session.receipt(
                "latent_kv_write_conformance",
                BTreeMap::from([
                    ("case".into(), json!(writer.label)),
                    ("evidence".into(), evidence),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("latent KV writes conform");
}
