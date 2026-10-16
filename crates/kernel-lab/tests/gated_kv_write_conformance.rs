//! The gated Q/K norm, M-RoPE and paged KV write through the declarative
//! conformance runner (`kernel_lab::conformance`): the served entries
//! `qk_norm_rope_kv_{bf16,fp8}` (`attention::gated`, kernel-core
//! `GatedAttentionQuery::QkNormRopeKvWrite`) of the lab shapes at heads of
//! 128 and 256, and the owner-routed `kv_write_{bf16,fp8}` of the
//! context-parallel lab profile (`owned_writes_shapes.rs`), which run the
//! same body over group-local positions and skip the K/V write of a row
//! another rank owns (local position -1).
//!
//! The K and V caches are state (`common/kv_pool.rs`): pages of old
//! contents and decoy bytes, of which a launch may change only the slots of
//! its live rows, so untouched pages, decoys and the slots of rows past the
//! live count must keep their bytes; Q is an output of live rows. Six rows
//! write three items' pages, one at a page's last slot and the next at the
//! next page's first. The page tables index both caches and rerun with
//! their pages moved to the top; a -1 page lies outside the pool, so they
//! declare no hole row. The served entries take every position as a cache
//! position (a negative one is outside their contract and may write inside
//! the pool), so their positions are plain inputs. The owner-routed rows include one
//! another rank owns, and their local positions rerun with the last live
//! row -1 (`Holes::LastLive`), which must keep that row's slot and still
//! write its Q.
//!
//! The oracle emulates the kernel's F32 arithmetic (`gated.cuh`
//! `inverse_rms`, `normalize`, `rotary::rotate`) on fixture operands that
//! keep every step exact: Q and K elements are +-8, so the sum of squares is
//! exact in any order and the mean square 64 absorbs the norm's 1e-6
//! epsilon (half an F32 step at 64 is 2^-18), leaving the inverse 1/8; the
//! norm weights make `1 + w` one of 3/4, 1, 5/4, 3/2 and 7/4; and the RoPE
//! table's cosines and sines are 0 and +-1, varying by position and pair,
//! so a rotation only moves and negates elements. Q and K are then exact in
//! BF16 and, divided by the K scale 2, in E4M3. V is copied in BF16, or its
//! E4M3 values times the V scale 1/4 are divided back exactly. The oracle's
//! checks are narrower than this domain: it refuses a sum of squares whose
//! terms are not integers or that reaches 2^24 (within those bounds every
//! reduction order is exact), an epsilon that changes the mean square, and
//! any value BF16 or E4M3 does not hold exactly. The wide cases run the
//! served entries at heads 128 and 256, whose page and byte addresses the
//! kernel computes in separate branches, over caches of
//! [`hot_keys::WIDE_PAGES`] pages at a stride of 33280 bytes, so the
//! high-id rerun's pages lie past 16 bits and their byte offsets past 2^31
//! (below 2^32, so an unsigned 32-bit offset is not told apart). The oracle
//! shows that the IDs narrowed to 16 bits change the caches, and
//! `wide_gated_kv_write_cases_validate` requires a writer that narrows them
//! to fail the high-id rerun's state checks of every wide case.
//!
//! GPU: `CHOPINFER_KERNEL_GPU=<ordinal> CHOPINFER_KERNEL_REQUIRE=1 cargo test
//! -p kernel-lab --release --locked --offline --test
//! gated_kv_write_conformance -- --test-threads=1 --nocapture`

#[allow(dead_code)]
#[path = "../owned_writes_shapes.rs"]
mod owned_shapes;

#[path = "common/hot_keys.rs"]
mod hot_keys;

#[path = "common/hot_fp8.rs"]
mod hot_fp8;

#[path = "common/kv_pool.rs"]
mod kv_pool;

use std::collections::BTreeMap;
use std::sync::atomic::Ordering::Relaxed;

use hot_fp8::{K_SCALE, V_SCALE};
use hot_keys::{
    HIGHEST, NARROWED, WIDE_PAGES, bf16_bytes, f32_bytes, i32_bytes, low16, narrowing_checked,
};
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::{Argument, DType, GatedAttentionQuery};
use kernel_lab::conformance::{Case, Expected, Holes, Images, Rows, Write};
use kernel_lab::lab_entry::LabEntry;
use kernel_lab::lab_shapes::{ROW_PROFILES, gated_config};
use kernel_lab::poison::WriteMap;
use kernel_lab::{LabResult, gpu_test, latent_shapes};
use kernels_sm120::Profile;
use kernels_sm120::attention::gated;
use kv_pool::{bf16_exact, e4m3_value, mix, old_pool, put, ranges, slots};
use serde_json::json;

/// Pages of each cache, more than the tables use, so every page can move up.
const POOL_PAGES: usize = 10;
/// Each item's pages, all low in the pool.
const TABLES: [&[i32]; 3] = [&[3, 1, 4], &[0, 5], &[2]];
/// Table width: one more than the longest table, so every table ends in -1.
const MAX_PAGES: usize = 4;
/// Each row's item and cache position.
const SERVED_ROWS: [(i32, i32); 6] = [(0, 127), (0, 128), (1, 64), (1, 70), (2, 5), (0, 0)];
/// Each row's item and local position; row 2's page belongs to another
/// rank.
const OWNED_ROWS: [(i32, i32); 6] = [(0, 127), (0, 128), (1, -1), (1, 70), (2, 5), (0, 0)];
/// Rows of the RoPE table; every RoPE position lies below it.
const ROPE_ROWS: usize = 8;
/// Bytes between page starts: every page and at least 512 decoy bytes (a
/// head-256 BF16 page takes 32768), so wide offsets pass 2^31.
const STRIDE: usize = 33280;

/// One gated KV write entry over two caches.
struct Writer {
    label: &'static str,
    entry: Box<dyn LabEntry>,
    config: gated::Config,
    fp8: bool,
    /// The positions parameter: `positions` (served) or `local_positions`
    /// (owner-routed, -1 for a row another rank owns).
    positions: &'static str,
    rows: [(i32, i32); 6],
    pool_pages: usize,
    stride: usize,
}

impl Writer {
    /// The served entry of lab profile `profile` for `dtype`.
    fn served(label: &'static str, profile: Profile, dtype: DType) -> Self {
        let entry = gated::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|entry| {
                entry.profile == profile
                    && matches!(entry.query,
                        GatedAttentionQuery::QkNormRopeKvWrite { kv_dtype, .. } if kv_dtype == dtype)
            })
            .unwrap_or_else(|| panic!("no gated KV write of {} {dtype:?}", profile.tag()));
        let config = gated_config(profile);
        Self::new(
            label,
            Box::new(entry),
            config,
            dtype,
            "positions",
            SERVED_ROWS,
        )
    }

    /// The owner-routed write of the context-parallel lab profile.
    fn owned(label: &'static str, dtype: DType) -> Self {
        let name = format!(
            "{}_kv_write_{}",
            owned_shapes::PROFILE.tag(),
            if dtype == DType::Fp8E4M3 {
                "fp8"
            } else {
                "bf16"
            }
        );
        let entry = owned_shapes::lab_entries()
            .into_iter()
            .find(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("no owner-routed write {name}"));
        let config = owned_shapes::GATED;
        Self::new(
            label,
            Box::new(entry),
            config,
            dtype,
            "local_positions",
            OWNED_ROWS,
        )
    }

    fn new(
        label: &'static str,
        entry: Box<dyn LabEntry>,
        config: gated::Config,
        dtype: DType,
        positions: &'static str,
        rows: [(i32, i32); 6],
    ) -> Self {
        let writer = Self {
            label,
            entry,
            config,
            fp8: dtype == DType::Fp8E4M3,
            positions,
            rows,
            pool_pages: POOL_PAGES,
            stride: STRIDE,
        };
        assert!(writer.page_bytes() <= STRIDE - 512, "{label}");
        writer
    }

    /// Every count of `poison::live_counts`, or a wide case's
    /// (`kv_pool::wide_counts`).
    fn counts(&self) -> Vec<u32> {
        let capacity = self.rows.len() as u32;
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

    fn dim(&self) -> usize {
        self.config.head_dim as usize
    }

    fn element(&self) -> usize {
        if self.fp8 { 1 } else { 2 }
    }

    /// Bytes of one cache page: `page` tokens of every KV head.
    fn page_bytes(&self) -> usize {
        self.config.page as usize * self.config.kv_heads as usize * self.dim() * self.element()
    }

    fn cache_dtype(&self) -> DType {
        if self.fp8 {
            DType::Fp8E4M3
        } else {
            DType::Bf16
        }
    }

    /// Elements of one packed row: each Q head's query and gate, then each
    /// KV head's key and value.
    fn width(&self) -> usize {
        self.config.row_width() as usize
    }

    /// KV head `head`'s bytes at `slot` of `page`.
    fn span(&self, page: usize, slot: usize, head: usize) -> (usize, usize) {
        let row = self.dim() * self.element();
        let within = slot * self.config.kv_heads as usize + head;
        (page * self.stride + within * row, row)
    }

    fn spans(&self, page: usize, slot: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.config.kv_heads as usize).map(move |head| self.span(page, slot, head))
    }

    /// Head `head`'s norm and RoPE of row `row`, as the kernel computes
    /// them; `weight` is the head's norm weights, `start` its first element
    /// in the packed row.
    fn normed(
        &self,
        images: &Images,
        row: usize,
        start: usize,
        weight: &str,
    ) -> LabResult<Vec<f32>> {
        let qkv = images.u16s("qkv")?;
        let x = qkv[row * self.width() + start..][..self.dim()]
            .iter()
            .map(|&bits| bf16_to_f32(bits))
            .collect::<Vec<_>>();
        let weight = images.u16s(weight)?;
        let rope = images.i32s("rope_positions")?;
        let table = images.f32s("cos_sin")?;
        let half = self.config.rotary as usize / 2;
        let [_, section_h, section_w] = self.config.sections.map(|s| s as usize);
        let angle = |pair: usize| -> LabResult<(f32, f32)> {
            let axis = if pair % 3 == 1 && pair < 3 * section_h {
                1
            } else if pair % 3 == 2 && pair < 3 * section_w {
                2
            } else {
                0
            };
            let position = usize::try_from(rope[row * 3 + axis])
                .map_err(|_| format!("row {row} has a negative RoPE position"))?;
            let at = (position * half + pair) * 2;
            let cs = table
                .get(at..at + 2)
                .ok_or_else(|| format!("row {row} RoPE position {position} lies past the table"))?;
            Ok((cs[0], cs[1]))
        };
        norm_rope(&x, &weight, self.config.eps, 2 * half, angle)
    }

    /// The caches and Q after a launch at `live`, the tables read through
    /// `read`.
    fn after(
        &self,
        images: &Images,
        live: u32,
        read: fn(i32) -> i32,
    ) -> LabResult<Vec<(String, Expected)>> {
        let (dim, width) = (self.dim(), self.width());
        let (q_heads, kv_heads) = (self.config.q_heads as usize, self.config.kv_heads as usize);
        let capacity = self.rows.len();
        let mut q = vec![0u16; capacity * q_heads * dim];
        for row in 0..live as usize {
            for head in 0..q_heads {
                let values = self.normed(images, row, head * 2 * dim, "q_norm")?;
                for (d, value) in values.into_iter().enumerate() {
                    q[(row * q_heads + head) * dim + d] = bf16_exact(value)
                        .map_err(|e| format!("Q row {row} head {head} dim {d}: {e}"))?;
                }
            }
        }
        let mut k_cache = images.bytes("k_cache")?.to_vec();
        let mut v_cache = images.bytes("v_cache")?.to_vec();
        let qkv = images.u16s("qkv")?;
        let page = self.config.page as usize;
        for s in slots(images, live, self.positions, page, MAX_PAGES, read)? {
            for head in 0..kv_heads {
                let key = q_heads * 2 * dim + head * dim;
                let k = self.normed(images, s.row, key, "k_norm")?;
                let value = q_heads * 2 * dim + kv_heads * dim + head * dim;
                let v = &qkv[s.row * width + value..][..dim];
                let at = self.span(s.page, s.slot, head).0;
                let context = |what: &str, d: usize, e: String| {
                    format!("{what} row {} head {head} dim {d}: {e}", s.row)
                };
                if self.fp8 {
                    let codes = |values: &mut dyn Iterator<Item = f32>, scale: f32, what: &str| {
                        values
                            .enumerate()
                            .map(|(d, x)| {
                                hot_fp8::e4m3(f64::from(x / scale)).map_err(|e| context(what, d, e))
                            })
                            .collect::<LabResult<Vec<u8>>>()
                    };
                    put(&mut k_cache, at, &codes(&mut k.into_iter(), K_SCALE, "K")?)?;
                    let v = &mut v.iter().map(|&bits| bf16_to_f32(bits));
                    put(&mut v_cache, at, &codes(v, V_SCALE, "V")?)?;
                } else {
                    let k = k
                        .into_iter()
                        .enumerate()
                        .map(|(d, x)| bf16_exact(x).map_err(|e| context("K", d, e)))
                        .collect::<LabResult<Vec<u16>>>()?;
                    put(&mut k_cache, at, &bf16_bytes(&k))?;
                    put(&mut v_cache, at, &bf16_bytes(v))?;
                }
            }
        }
        Ok(vec![
            ("q".into(), Expected::Bits(bf16_bytes(&q))),
            ("k_cache".into(), Expected::Bits(k_cache)),
            ("v_cache".into(), Expected::Bits(v_cache)),
        ])
    }

    fn case(&self) -> Case<'_> {
        let capacity = self.rows.len();
        let (dim, width) = (self.dim(), self.width());
        let (q_heads, kv_heads) = (self.config.q_heads as usize, self.config.kv_heads as usize);
        let mut tables = vec![-1; TABLES.len() * MAX_PAGES];
        for (item, pages) in TABLES.iter().enumerate() {
            tables[item * MAX_PAGES..][..pages.len()].copy_from_slice(pages);
        }
        let qkv = (0..capacity)
            .flat_map(|row| packed_row(row, q_heads, kv_heads, dim))
            .collect::<Vec<_>>();
        assert_eq!(qkv.len(), capacity * width);
        let rope = (0..capacity)
            .flat_map(|row| [row * 3, row * 5 + 1, row * 7 + 2].map(|p| (p % ROPE_ROWS) as i32))
            .collect::<Vec<_>>();
        let half = self.config.rotary as usize / 2;
        let table = (0..ROPE_ROWS * half)
            .flat_map(|at| {
                // Position `at / half`, pair `at % half`.
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]][(at / half + at % half) % 4]
            })
            .collect::<Vec<f32>>();
        let items = self.rows.map(|(item, _)| item);
        let positions = self.rows.map(|(_, position)| position);
        let dtype = self.cache_dtype();
        let element = self.element();
        let stride = self.stride;
        let pages = self.pool_pages;
        let cache_map = move |images: &Images, live: u32| {
            let spans = slots(
                images,
                live,
                self.positions,
                self.config.page as usize,
                MAX_PAGES,
                |id| id,
            )?
            .into_iter()
            .flat_map(|s| self.spans(s.page, s.slot).collect::<Vec<_>>());
            WriteMap::of(dtype).byte_ranges(ranges(spans))
        };
        let case = Case::new(self.entry.as_ref(), capacity as u32)
            .live_counts(self.counts())
            .input("qkv", DType::Bf16, (Rows::Live, width), bf16_bytes(&qkv))
            .produced("qkv")
            .input(
                "q_norm",
                DType::Bf16,
                (Rows::Fixed(1), dim),
                bf16_bytes(&norm_weights(dim, 0)),
            )
            .input(
                "k_norm",
                DType::Bf16,
                (Rows::Fixed(1), dim),
                bf16_bytes(&norm_weights(dim, 2)),
            )
            .input(
                "rope_positions",
                DType::I32,
                (Rows::Live, 3),
                i32_bytes(&rope),
            )
            .input(
                "cos_sin",
                DType::F32,
                (Rows::Fixed(ROPE_ROWS), half * 2),
                f32_bytes(&table),
            )
            .input("row_batch", DType::I32, (Rows::Live, 1), i32_bytes(&items));
        let case = if self.positions == "positions" {
            case.input(
                "positions",
                DType::I32,
                (Rows::Live, 1),
                i32_bytes(&positions),
            )
        } else {
            case.ids(
                self.positions,
                (Rows::Live, 1),
                &positions,
                &[],
                Holes::LastLive,
            )
        };
        case.ids(
            "block_tables",
            (Rows::Fixed(TABLES.len()), MAX_PAGES),
            &tables,
            &[("k_cache", stride), ("v_cache", stride)],
            Holes::Never,
        )
        .state(
            "k_cache",
            dtype,
            (Rows::Fixed(pages), stride / element),
            old_pool(pages, self.page_bytes(), stride, 0),
            Write::Map(Box::new(cache_map)),
            |_, _| Ok(Vec::new()),
        )
        .state(
            "v_cache",
            dtype,
            (Rows::Fixed(pages), stride / element),
            // Other old contents than the K cache's.
            old_pool(pages, self.page_bytes(), stride, 1),
            Write::Map(Box::new(cache_map)),
            |_, _| Ok(Vec::new()),
        )
        .output(
            "q",
            DType::Bf16,
            (Rows::Live, q_heads * dim),
            Write::LiveRows,
        )
        .scalar("max_pages", Argument::U32(MAX_PAGES as u32))
        .scalar("k_page_stride_bytes", Argument::U64(stride as u64))
        .scalar("v_page_stride_bytes", Argument::U64(stride as u64))
        .scalar("eps_bits", Argument::U32(self.config.eps.to_bits()))
        .scalar("k_scale_bits", Argument::U32(K_SCALE.to_bits()))
        .scalar("v_scale_bits", Argument::U32(V_SCALE.to_bits()))
        .oracle(move |images, live| {
            narrowing_checked(images, live, "block_tables", |images, live, read| {
                self.after(images, live, read)
            })
        })
    }
}

/// `x` normed and rotated as `gated.cuh` does: the sum of squares, the
/// inverse `1 / sqrt(sum / dim + eps)`, each element `(x * inverse) * (1 +
/// w)`, then NeoX pairs `(d, d + rotary / 2)` of the first `rotary`
/// elements rotated by `angle(d % (rotary / 2))`, `x c - y s` for the first
/// of a pair and `x c + y s` for the second (`rotary::rotate`). Refused
/// unless every square is an integer and their sum lies below 2^24, so the
/// sum is exact in every order the kernel's lanes may add it, and unless
/// `eps` leaves the mean square unchanged.
fn norm_rope(
    x: &[f32],
    weight: &[u16],
    eps: f32,
    rotary: usize,
    angle: impl Fn(usize) -> LabResult<(f32, f32)>,
) -> LabResult<Vec<f32>> {
    let dim = x.len();
    let squares = x.iter().map(|&v| f64::from(v) * f64::from(v));
    let exact = squares.clone().sum::<f64>();
    if squares.clone().any(|square| square.fract() != 0.0) || exact >= f64::from(1u32 << 24) {
        return Err(format!(
            "the sum of squares {exact} is not exact in every order"
        ));
    }
    let sum = x.iter().map(|v| v * v).sum::<f32>();
    let mean = sum * (1.0 / dim as f32);
    if mean + eps != mean {
        return Err(format!("epsilon {eps} changes the mean square {mean}"));
    }
    let inverse = 1.0 / (mean + eps).sqrt();
    let normed = x
        .iter()
        .zip(weight)
        .map(|(&v, &w)| (v * inverse) * (1.0 + bf16_to_f32(w)))
        .collect::<Vec<_>>();
    let half = rotary / 2;
    (0..dim)
        .map(|d| {
            if d >= rotary {
                return Ok(normed[d]);
            }
            let (c, s) = angle(d % half)?;
            Ok(if d < half {
                normed[d].mul_add(c, -(normed[d + half] * s))
            } else {
                normed[d].mul_add(c, normed[d - half] * s)
            })
        })
        .collect()
}

/// Norm weights of `dim` whose `1 + w` cycle through 3/4, 1, 5/4, 3/2 and
/// 7/4 from `offset`.
fn norm_weights(dim: usize, offset: usize) -> Vec<u16> {
    (0..dim)
        .map(|d| bf16_exact([-0.25, 0.0, 0.25, 0.5, 0.75][(d + offset) % 5]).expect("BF16"))
        .collect()
}

/// Row `row`'s packed BF16 projection: Q heads (query, then gate) and K
/// heads of +-8 with signs that differ by row and head, then V heads of
/// E4M3 values times the V scale.
fn packed_row(row: usize, q_heads: usize, kv_heads: usize, dim: usize) -> Vec<u16> {
    let signed = |head: usize| {
        (0..dim).map(move |d| {
            let bits = mix(((row * 64 + head) * 8 + d / 64) as u64);
            bf16_exact(if (bits >> (d % 64)) & 1 == 0 {
                8.0
            } else {
                -8.0
            })
            .expect("BF16")
        })
    };
    let values = |head: usize, scale: f32| {
        (0..dim).map(move |d| {
            let h = mix(((row * 64 + head) * 1024 + d) as u64);
            bf16_exact(e4m3_value(h) * scale).expect("BF16")
        })
    };
    let mut packed = Vec::new();
    for head in 0..q_heads {
        packed.extend(signed(head));
        packed.extend(values(head, 1.0));
    }
    for head in 0..kv_heads {
        packed.extend(signed(32 + head));
    }
    for head in 0..kv_heads {
        packed.extend(values(48 + head, V_SCALE));
    }
    packed
}

/// The served entries at heads of 128 and 256, BF16 and FP8, and the
/// owner-routed writes.
fn cases() -> Vec<Writer> {
    let (d128, d256) = (ROW_PROFILES[3], ROW_PROFILES[2]);
    vec![
        Writer::served("served d128 bf16", d128, DType::Bf16),
        Writer::served("served d128 fp8", d128, DType::Fp8E4M3),
        Writer::served("served d256 bf16", d256, DType::Bf16),
        Writer::served("served d256 fp8", d256, DType::Fp8E4M3),
        Writer::owned("owned d256 bf16", DType::Bf16),
        Writer::owned("owned d256 fp8", DType::Fp8E4M3),
    ]
}

/// The served entries at both heads over wide caches. The owner-routed
/// writes run the same body (`owned_writes.cuh` calls
/// `qk_norm_rope_kv`), whose head-128 and head-256 address branches these
/// cover.
fn wide_cases() -> Vec<Writer> {
    let (d128, d256) = (ROW_PROFILES[3], ROW_PROFILES[2]);
    vec![
        Writer::served("served d128 bf16", d128, DType::Bf16).wide("wide served d128 bf16"),
        Writer::served("served d128 fp8", d128, DType::Fp8E4M3).wide("wide served d128 fp8"),
        Writer::served("served d256 bf16", d256, DType::Bf16).wide("wide served d256 bf16"),
        Writer::served("served d256 fp8", d256, DType::Fp8E4M3).wide("wide served d256 fp8"),
    ]
}

/// The attention cubin that holds a served entry, or the owner-routed
/// writes' cubin.
fn cubin(writer: &Writer) -> String {
    match writer.positions {
        "positions" => {
            let profile = [ROW_PROFILES[2], ROW_PROFILES[3]]
                .into_iter()
                .find(|&profile| gated_config(profile) == writer.config)
                .expect("a lab profile");
            format!("attention_{}_sm120", profile.tag())
        }
        _ => owned_shapes::CUBIN.into(),
    }
}

#[test]
fn gated_kv_write_cases_validate() {
    for writer in cases() {
        let symbol = writer.entry.symbol();
        let entries = kernel_lab::cubin_entries(&cubin(&writer)).expect("a lab cubin");
        assert!(
            entries.contains(&symbol.as_str()),
            "{symbol} is not in {}",
            cubin(&writer)
        );
        writer
            .case()
            .validate()
            .unwrap_or_else(|e| panic!("{}: {e}", writer.label));
    }
}

/// The wide cases' high-id reruns move their pages to IDs 65630 to 65635,
/// at byte offsets past 2^31, and the IDs narrowed to 16 bits change the
/// caches: a writer that narrows them, at either head's address branch,
/// fails the high-id rerun's state checks and only those. One test, so the
/// wide caches are built one case at a time.
#[test]
fn wide_gated_kv_write_cases_validate() {
    for writer in wide_cases() {
        assert!(((WIDE_PAGES - 6) * writer.stride) as u64 > 1 << 31);
        let case = writer.case();
        case.validate()
            .unwrap_or_else(|e| panic!("{}: {e}", writer.label));
        let narrowing = case
            .refusing_runs(|images, live| {
                let mut states = writer.after(images, live, low16)?;
                states.retain(|(name, _)| name != "q");
                states
                    .into_iter()
                    .map(|(name, expected)| match expected {
                        Expected::Bits(bits) => Ok((name, bits)),
                        Expected::Values(_) | Expected::Judge(_) => {
                            Err(format!("{name} has no bits"))
                        }
                    })
                    .collect()
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

/// The fixture's rotations move and negate exact elements, and the oracle
/// refuses a sum not exact in every order, an epsilon that changes the
/// mean square, and a K value E4M3 does not hold.
#[test]
fn the_norm_emulation_is_exact_and_refuses_outside_its_domain() {
    let weight = norm_weights(128, 0);
    let x = packed_row(0, 1, 1, 128)[..128]
        .iter()
        .map(|&bits| bf16_to_f32(bits))
        .collect::<Vec<_>>();
    let quarter = |_| Ok((0.0, 1.0));
    let rotated = norm_rope(&x, &weight, 1e-6, 32, quarter).unwrap();
    for d in 0..16 {
        // x c - y s with c 0 and s 1: minus the mate.
        let mate = (x[d + 16] / 8.0) * (1.0 + bf16_to_f32(weight[d + 16]));
        assert_eq!(rotated[d], -mate);
    }
    for d in 32..128 {
        assert_eq!(rotated[d], (x[d] / 8.0) * (1.0 + bf16_to_f32(weight[d])));
    }
    let mut odd = x.clone();
    odd[3] = 8.0 + 1.0 / 1024.0;
    assert!(norm_rope(&odd, &weight, 1e-6, 32, quarter).is_err());
    // Squares of 2^-8 and 2^16 sum exactly in one order but not in every
    // order a warp may add them.
    let mut spread = vec![0.0; 128];
    (spread[0], spread[4], spread[8]) = (1.0 / 16.0, 1.0 / 16.0, 256.0);
    assert!(
        norm_rope(&spread, &[0; 128], 1e-6, 32, |_| Ok((1.0, 0.0)))
            .unwrap_err()
            .contains("not exact in every order")
    );
    assert!(
        norm_rope(&x, &weight, 1e-3, 32, quarter)
            .unwrap_err()
            .contains("changes the mean square")
    );
    // 1 + w of 1.0625 needs 4 mantissa bits, which E4M3 lacks.
    assert!(hot_fp8::e4m3(f64::from(1.0625f32 / K_SCALE)).is_err());
    for w in [-0.25f32, 0.0, 0.25, 0.5, 0.75] {
        hot_fp8::e4m3(f64::from((1.0 + w) / K_SCALE)).unwrap();
    }
}

#[test]
fn gated_kv_writes_conform_gpu() {
    // A wide case's two caches, 2 x 2.03 GiB, and 1 GiB for the rest; the
    // cases run one at a time and free their buffers.
    let need = 2 * WIDE_PAGES * STRIDE + (1 << 30);
    gpu_test(need as u64, |session| {
        let lab = session.module(&latent_shapes::cubin_name())?;
        for writer in cases().into_iter().chain(wide_cases()) {
            let module = session.module(&cubin(&writer))?;
            let evidence = writer
                .case()
                .run(session, &module, Some(&lab))
                .map_err(|e| format!("{}: {e}", writer.label))?;
            session.receipt(
                "gated_kv_write_conformance",
                BTreeMap::from([
                    ("case".into(), json!(writer.label)),
                    ("evidence".into(), evidence),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("gated KV writes conform");
}
