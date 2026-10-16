//! Cases of the attention family (`kernel_core::attention`): decode,
//! verify and prefill over paged caches and window rings, BF16 or FP8
//! E4M3, decode with folded NeoX RoPE and KV write, and verify on the
//! decode rows entries for one-row exactness; with the SM120 selection the
//! lab's catalogs give and the family's required op set. The gated fold is
//! not covered.
//!
//! A case row is one item: its query rows (one in decode), and its
//! per-item arrays. Items attend to one to four pages or rings of keys,
//! some ending on a page boundary; paged items' pages scatter across the
//! cache and ring items' rings run in reverse, and every ring is full, so
//! long contexts wrap it. Every page and token of a plane is distinct.
//! Block tables, slot ids and lengths are rows of the live count, so the
//! property checks move them with their items. With several query rows
//! per item, the batch check copies whole items and the chunk check, which
//! compares single query rows, is reported as not exercised
//! ([`ITEM_ROWS`]); one-row bits (exact verify rows) compare each verify
//! row with the decode op launched over it alone ([`decoded`]).
//! Paged prefill and verify keys include the step's rows (written before
//! attention); ring steps outside decode read the step's own K and V from
//! `k_new` and `v_new`.
//!
//! The device live count is a step input ([`STEP_INPUTS`]): the PDL
//! phases hold Q (a fold's packed rows) behind the producers, not the
//! count.
//!
//! The oracle is `oracle::attention` or `oracle::ring_attention` over the
//! values the cache stores (an FP8 code times its plane's scale), within
//! its bound. A fold's rotated K compares as the RoPE KV write cases' does
//! ([`super::rope_kv`]); its output's bound adds the most the logits can
//! move each output when the device's rotated Q and stored K differ from
//! the oracle's within that allowance and their rounding (`widen_by_fold`),
//! weighted by the softmax.

use crate::LabResult;
use crate::conformance::{Expected, Images, Rows, Write, admitted_range};
use crate::kit::decode::{Compared, DecodeLaunch, Verified};
use crate::kit::{Decoded, OpCase, Required, Spec, bf16_bytes, bf16_values};
use crate::poison::WriteMap;
use kernel_core::attention::{AttentionBuffers, AttentionFold, AttentionOp, FoldInputs, KvCache};
use kernel_core::backend::{Counts, Need, Prepared, View};
use kernel_core::codec::{SmallFloat, bf16_to_f32};
use kernel_core::contract::{
    AttentionLayout, AttentionPhase, AttentionQuery, DType, RotarySource, RotaryStyle, RowQuery,
};
use kernel_core::oracle::{self, OracleOutput, PagedAttention, RingAttention};
use kernel_core::properties::Invariance;
use kernels_sm120::attention::Variant;
use kernels_sm120::backend::AttentionEntries;

use super::rope_kv::{Rope, half_bf16};

/// The items one decode op serves: a capacity bucket of the SM120 row
/// kernels, as the row families'.
pub const CAPACITY: u32 = super::ROW_CAPACITY;

/// Why an attention launch may read the device live count before its PDL
/// wait: the host writes it with the step's lengths, tables and slots by
/// stream-ordered copies before the step, and SM120's decode, verify and
/// merge entries read those step inputs before the wait to plan their
/// splits. Q and a fold's packed rows come from producers and stay probed.
pub const STEP_INPUTS: &str = "the host writes the step's live count, lengths, tables and slots \
                               with stream-ordered copies before the step";

/// Why a case of multi-row items does not run the chunk check, and checks
/// one-row bits against decode launches.
pub const ITEM_ROWS: &str = "a case row is an item of several query rows, and the chunk check \
                             compares single query rows";

/// The items of a verify or prefill op.
pub const STEP_ITEMS: u32 = 4;

/// Query rows of each verify item, at most, and of each prefill item.
const VERIFY_ROWS: u32 = 4;
const PREFILL_ROWS: u32 = 8;

/// The FP8 K and V tensor scales, as the RoPE KV write cases store them.
const FP8_SCALES: [f32; 2] = [0.5, 0.25];

/// The theta and V multiplier of a folded RoPE: a power-of-two multiplier,
/// so the V the fold stores is exact.
const FOLD_THETA: f32 = 1e4;
const FOLD_V_MULTIPLIER: f32 = 0.5;

/// Item `item`'s keys at page size `page`: one to four pages, every fifth
/// item ending on a page boundary.
fn keys(item: usize, page: usize) -> usize {
    if item % 5 == 2 {
        (item % 4 + 1) * page
    } else {
        1 + (item * 61 + 7) % (4 * page)
    }
}

/// The cache page of an item's `at`th page in order of all items' pages
/// among `used` pages: a permutation by a stride coprime to `used`, so the
/// items' pages lie neither in order nor reversed.
fn scattered(at: usize, used: usize) -> usize {
    let gcd = |mut a: usize, mut b: usize| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let stride = (7..)
        .find(|&s| gcd(s, used) == 1)
        .expect("a coprime stride");
    (at * stride + 3) % used
}

/// A cache plane image of `elements`: BF16 values from the kit's
/// generator, or E4M3 codes hashed from `seed` and each element's index,
/// so no two pages or tokens hold the same codes.
fn plane(elements: usize, dtype: DType, seed: u64) -> Vec<u8> {
    match dtype {
        DType::Fp8E4M3 => (0..elements as u64)
            .map(|at| {
                let hash = kernel_core::ngram_oracle::splitmix64(seed << 40 ^ at);
                // Finite codes of magnitude at most 2: at most 0x40.
                (hash as u8 & 0x80) | (((hash >> 8) as u8 & 0x7f) % 0x41)
            })
            .collect(),
        _ => bf16_bytes(&bf16_values(seed, elements)),
    }
}

/// The values of a plane image: BF16 values, or E4M3 codes times `scale`.
fn plane_values(image: &[u8], dtype: DType, scale: f32) -> Vec<f64> {
    match dtype {
        DType::Fp8E4M3 => image
            .iter()
            .map(|&code| SmallFloat::E4M3.decode_f64(code) * f64::from(scale))
            .collect(),
        _ => image
            .chunks_exact(2)
            .map(|half| f64::from(bf16_to_f32(u16::from_le_bytes([half[0], half[1]]))))
            .collect(),
    }
}

/// `value` as the cache stores it, in its own units: BF16, or an E4M3 code
/// of `value / scale` decoded and scaled back.
fn stored(value: f64, dtype: DType, scale: f32) -> f64 {
    match dtype {
        DType::Fp8E4M3 => {
            kernel_core::kv_fp8::decode(kernel_core::kv_fp8::encode(value as f32, scale), scale)
        }
        _ => oracle::bf16_round(value),
    }
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The layout of one case: its items and their keys.
#[derive(Clone)]
struct Layout {
    query: AttentionQuery,
    fold: Option<AttentionFold>,
    /// Query rows of each item.
    rows: usize,
    /// Keys before each item's query rows.
    prefix: Vec<usize>,
    /// Pages or rings of the cache, the block table width, and each item's
    /// block table row (paged) or ring (ring).
    slots: usize,
    pages: usize,
    index: Vec<i32>,
}

impl Layout {
    fn new(op: &AttentionOp, items: usize) -> LabResult<Self> {
        let query = op.query;
        let decode = query.phase == AttentionPhase::Decode;
        let rows = if decode { 1 } else { op.item_rows as usize };
        let page = query.page_size as usize;
        let ring = query.layout == AttentionLayout::Ring;
        let totals = (0..items)
            .map(|item| keys(item, page).max(rows))
            .collect::<Vec<_>>();
        if totals
            .iter()
            .any(|&t| t > query.max_context_capacity as usize)
        {
            return Err("the case's contexts exceed the op's".into());
        }
        let prefix = totals.iter().map(|t| t - rows).collect::<Vec<_>>();
        let (slots, pages, index) = if ring {
            (
                items + 1,
                1,
                (0..items).rev().map(|item| item as i32).collect(),
            )
        } else {
            let pages = query.max_context_capacity.div_ceil(query.page_size) as usize;
            let used = totals.iter().map(|t| t.div_ceil(page)).sum::<usize>();
            // Each item's pages, scattered across the cache; unread
            // entries hold the spare top slot.
            let mut tables = vec![used as i32; items * pages];
            let mut next = 0;
            for (item, total) in totals.iter().enumerate() {
                for entry in &mut tables[item * pages..][..total.div_ceil(page)] {
                    *entry = scattered(next, used) as i32;
                    next += 1;
                }
            }
            (used + 1, pages, tables)
        };
        Ok(Self {
            query,
            fold: op.fold,
            rows,
            prefix,
            slots,
            pages,
            index,
        })
    }

    /// The layout of `items` one-row decode items over `verify`'s cache:
    /// item `k` is verify item `i = k / r` (cycling) at its row `j = k % r`,
    /// with its table or ring and `prefix + j` keys before it, so the
    /// decode case's planes, tables and context are the verify step's.
    fn decoding(verify: &Layout, items: usize) -> LabResult<Self> {
        let r = verify.rows;
        let n = verify.prefix.len();
        if items < n * r {
            return Err(format!(
                "{items} decode items cannot hold the verify step's {} rows",
                n * r
            ));
        }
        let item = |k: usize| (k / r) % n;
        let index = if verify.ring() {
            (0..items).map(|k| verify.index[item(k)]).collect()
        } else {
            (0..items)
                .flat_map(|k| verify.index[item(k) * verify.pages..][..verify.pages].to_vec())
                .collect()
        };
        Ok(Self {
            query: AttentionQuery {
                phase: AttentionPhase::Decode,
                max_batch_capacity: items as u32,
                ..verify.query
            },
            fold: None,
            rows: 1,
            prefix: (0..items).map(|k| verify.prefix[item(k)] + k % r).collect(),
            slots: verify.slots,
            pages: verify.pages,
            index,
        })
    }

    fn ring(&self) -> bool {
        self.query.layout == AttentionLayout::Ring
    }

    fn decode(&self) -> bool {
        self.query.phase == AttentionPhase::Decode
    }

    /// The elements of one slot of a plane of head width `width`.
    fn slot(&self, width: usize) -> usize {
        self.query.page_size as usize * self.query.kv_heads as usize * width
    }

    /// Where item `item`'s key at `position` lives: its slot and token.
    fn place(&self, index: &[i32], item: usize, position: usize) -> (usize, usize) {
        let page = self.query.page_size as usize;
        if self.ring() {
            (index[item] as usize, position % page)
        } else {
            (
                index[item * self.pages + position / page] as usize,
                position % page,
            )
        }
    }

    /// The packed width of a query row: the folded rows' Q, K and V, or Q.
    fn q_width(&self) -> usize {
        let query = self.query;
        let (qh, kvh) = (query.q_heads as usize, query.kv_heads as usize);
        let (qk, v) = (query.qk_dim as usize, query.v_dim as usize);
        match self.fold {
            Some(_) => qh * qk + kvh * (qk + v),
            None => qh * qk,
        }
    }

    fn index_path(&self) -> &'static str {
        if self.ring() {
            "slot_ids"
        } else {
            "block_tables"
        }
    }
}

/// The rotation of a NeoX fold over the query's head width.
fn fold_rope(query: &AttentionQuery, fold: AttentionFold) -> LabResult<Rope> {
    let AttentionFold::Neox {
        rotary,
        theta,
        source,
        ..
    } = fold
    else {
        return Err("the kit covers NeoX folds only".into());
    };
    Rope::of(RowQuery::Rope {
        head_dim: query.qk_dim,
        rotary_dim: rotary,
        theta,
        style: RotaryStyle::Neox,
        source,
        fused_paged_kv_write: query.layout == AttentionLayout::Paged,
    })
}

/// The case of attention `op` over `items` items, a case row each.
pub fn attention_case<'a>(
    op: AttentionOp,
    items: u32,
    prepared: Prepared<AttentionBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    case_over(Layout::new(&op, items as usize)?, op, items, prepared)
}

/// The case of attention `op` over `items` items laid out as `layout`.
fn case_over<'a>(
    layout: Layout,
    op: AttentionOp,
    items: u32,
    prepared: Prepared<AttentionBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let n = items as usize;
    let query = op.query;
    if layout.fold.is_some() && !layout.decode() {
        return Err("folds are decode ops".into());
    }
    let rope = layout
        .fold
        .map(|fold| fold_rope(&query, fold))
        .transpose()?;
    let (qh, kvh) = (query.q_heads as usize, query.kv_heads as usize);
    let (qk, v) = (query.qk_dim as usize, query.v_dim as usize);
    let r = layout.rows;
    let dtype = query.kv_dtype;
    let element = if dtype == DType::Fp8E4M3 { 1 } else { 2 };
    let scales = if dtype == DType::Fp8E4M3 {
        FP8_SCALES
    } else {
        [1.0; 2]
    };
    let totals = layout.prefix.iter().map(|p| p + r).collect::<Vec<_>>();
    // A fold rotates at the decode row's cache position; its table holds
    // every position of the step.
    let positions = totals.iter().copied().max().unwrap_or(1);
    let workspace = prepared.info().bytes_of("workspace").unwrap_or(0) as usize;
    let per_item = |values: Vec<i32>| Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&values));
    // A fold writes each live item's decode row into the planes; otherwise
    // they are inputs.
    let cache_plane = |width: usize, seed: u64| -> Spec<'a> {
        let image = plane(layout.slots * layout.slot(width), dtype, seed);
        let dims = (Rows::Fixed(layout.slots), layout.slot(width));
        if layout.fold.is_none() {
            return Spec::input(dtype, dims, image);
        }
        let layout = layout.clone();
        Spec::state(
            dtype,
            dims,
            image,
            Write::Map(Box::new(move |images, live| {
                let index = images.i32s(layout.index_path())?;
                let totals = images.i32s("seq_lens")?;
                let row = layout.query.kv_heads as usize * width;
                WriteMap::of(layout.query.kv_dtype).ranges((0..live as usize).map(|item| {
                    let (slot, token) = layout.place(&index, item, totals[item] as usize - 1);
                    let start = slot * layout.slot(width) + token * row;
                    start..start + row
                }))
            })),
        )
    };
    let non_decode = !layout.decode();
    let roles = AttentionBuffers {
        q: Spec::input(
            DType::Bf16,
            (Rows::Live, r * layout.q_width()),
            bf16_bytes(&bf16_values(21, n * r * layout.q_width())),
        ),
        cache: KvCache {
            k: cache_plane(qk, 22),
            v: cache_plane(v, 23),
            slots: layout.slots as u32,
            k_slot_stride: (layout.slot(qk) * element) as u64,
            v_slot_stride: (layout.slot(v) * element) as u64,
            k_scale: scales[0],
            v_scale: scales[1],
        },
        block_tables: (!layout.ring()).then(|| {
            Spec::input(
                DType::I32,
                (Rows::Live, layout.pages),
                i32_bytes(&layout.index),
            )
        }),
        slot_ids: layout.ring().then(|| per_item(layout.index.clone())),
        seq_lens: Some(per_item(totals.iter().map(|&t| t as i32).collect())),
        // Each item's first packed row follows from its place, so the
        // starts do not move with an item.
        q_start: non_decode.then(|| {
            Spec::input(
                DType::I32,
                (Rows::Fixed(n), 1),
                i32_bytes(&(0..n).map(|item| (item * r) as i32).collect::<Vec<_>>()),
            )
        }),
        q_len: non_decode.then(|| per_item(vec![r as i32; n])),
        prefix_len: non_decode.then(|| per_item(layout.prefix.iter().map(|&p| p as i32).collect())),
        live_batch: Spec::live(),
        k_new: (layout.ring() && non_decode).then(|| {
            Spec::input(
                DType::Bf16,
                (Rows::Live, r * kvh * qk),
                bf16_bytes(&bf16_values(25, n * r * kvh * qk)),
            )
        }),
        v_new: (layout.ring() && non_decode).then(|| {
            Spec::input(
                DType::Bf16,
                (Rows::Live, r * kvh * v),
                bf16_bytes(&bf16_values(26, n * r * kvh * v)),
            )
        }),
        sink: query.sink.then(|| {
            Spec::input(
                DType::Bf16,
                (Rows::Fixed(1), qh),
                bf16_bytes(&bf16_values(24, qh)),
            )
        }),
        rope: rope.map(|rope| FoldInputs {
            cos_sin: rope.table.then(|| {
                Spec::input(
                    DType::F32,
                    (Rows::Fixed(positions), rope.rotary),
                    rope.table_image(positions),
                )
            }),
            q_norm: None,
            k_norm: None,
            rope_positions: None,
        }),
        output: Spec::output(DType::Bf16, (Rows::Live, r * qh * v)),
        workspace: if workspace > 0 {
            Spec::scratch(workspace)
        } else {
            Spec::untouched(16)
        },
    };
    let oracle_layout = layout.clone();
    let oracle =
        move |images: &Images, live: u32| oracle(&oracle_layout, rope, scales, images, live);
    let context = positions as u32;
    let case = OpCase::with_counts(prepared, items, roles, move |items| {
        Counts::rows(items * r as u32)
            .with_items(items, context)
            .with_positions(context)
    })?
    .oracle(oracle)
    .live_count_before_wait(STEP_INPUTS);
    Ok(if r > 1 {
        // Copying whole items still checks that items ignore each other.
        case.item_rows(ITEM_ROWS)
    } else {
        case
    })
}

/// The decode op whose one-row launches a verify `op`'s exact rows equal:
/// the same query in decode, over [`CAPACITY`] items, unfolded, planning
/// each step for the bucket of its items, so a one-item launch plans as
/// the one-row reference does (bucket 1).
fn decode_op(op: &AttentionOp) -> AttentionOp {
    AttentionOp {
        query: AttentionQuery {
            phase: AttentionPhase::Decode,
            max_batch_capacity: CAPACITY,
            ..op.query
        },
        item_rows: 1,
        fold: None,
        plan_at_capacity: false,
    }
}

/// How verify `op`'s items of `items` map to launches of the decode op
/// `prepared` ([`decode_op`]) over one row alone, as the decode step of
/// that row would run (the step's context, bucket 1):
///
/// - the verify step runs the case's items, which must hold no more rows
///   than the claim covers;
/// - verify row `j` of item `i` is a decode item of Q row `i r + j`, item
///   `i`'s block table (paged) or ring (`slot_ids`), and `prefix + j + 1`
///   keys; paged keys are the verify step's cache, which holds the step's
///   rows; a ring holds item `i`'s step rows `0..=j` at their ring slots,
///   from `k_new` and `v_new` (the verify step reads them there), so
///   BF16 rings only;
/// - the decode output row equals verify output row `i r + j`.
fn decoded<'a>(
    op: AttentionOp,
    items: u32,
    prepared: Prepared<AttentionBuffers<View>>,
) -> LabResult<Decoded<'a>> {
    let verify = Layout::new(&op, items as usize)?;
    if verify.decode() || op.fold.is_some() || !op.query.causal {
        return Err(
            "only causal unfolded steps of several rows map to decode launches: a \
                    bidirectional row attends past itself"
                .into(),
        );
    }
    if verify.ring() && verify.query.kv_dtype != DType::Bf16 {
        return Err("a ring's step rows map to decode launches only for BF16 rings".into());
    }
    let layout = Layout::decoding(&verify, CAPACITY as usize)?;
    let case = case_over(layout, decode_op(&op), CAPACITY, prepared)?;
    let r = verify.rows;
    // Items hold no more rows than a claim covers: the oracle and the
    // output rows follow the case's fixed rows per item.
    let cut = move |images: &Images, max: u32| -> LabResult<Images> {
        if r > max as usize {
            return Err(format!(
                "the case's items hold {r} rows, more than the claim's {max}"
            ));
        }
        Ok(images.clone())
    };
    let launches = move |verified: &Verified<'_>, base: &Images| -> LabResult<Vec<DecodeLaunch>> {
        let images = verified.images;
        let query = verify.query;
        let (qh, kvh) = (query.q_heads as usize, query.kv_heads as usize);
        let (qk, v) = (query.qk_dim as usize, query.v_dim as usize);
        let q_row = verify.q_width() * 2;
        let out_row = qh * v * 2;
        let prefix = images.i32s("prefix_len")?;
        let lengths = images.i32s("q_len")?;
        let index = images.i32s(verify.index_path())?;
        let q = images.bytes("q")?;
        let mut launches = Vec::new();
        for item in 0..verified.live as usize {
            let prefix = prefix[item] as usize;
            for row in 0..lengths[item] as usize {
                let token = item * r + row;
                // The decode item's row 0 of each per-item input.
                let first = |path: &str, row: &[u8]| -> LabResult<Vec<u8>> {
                    let mut bytes = base.bytes(path)?.to_vec();
                    bytes[..row.len()].copy_from_slice(row);
                    Ok(bytes)
                };
                let mut decode = base
                    .with("q", first("q", &q[token * q_row..][..q_row])?)
                    .with("cache.k", images.bytes("cache.k")?.to_vec())
                    .with("cache.v", images.bytes("cache.v")?.to_vec())
                    .with(
                        "seq_lens",
                        first("seq_lens", &i32_bytes(&[(prefix + row + 1) as i32]))?,
                    );
                if query.sink {
                    decode = decode.with("sink", images.bytes("sink")?.to_vec());
                }
                if verify.ring() {
                    let mut planes = Vec::new();
                    for (plane, new, width) in [("cache.k", "k_new", qk), ("cache.v", "v_new", v)] {
                        let mut bytes = images.bytes(plane)?.to_vec();
                        let new = images.bytes(new)?;
                        let row_bytes = kvh * width * 2;
                        for step in 0..=row {
                            let (slot, at) = verify.place(&index, item, prefix + step);
                            let start = (slot * verify.slot(width)) * 2 + at * row_bytes;
                            bytes[start..][..row_bytes].copy_from_slice(
                                &new[(item * r + step) * row_bytes..][..row_bytes],
                            );
                        }
                        planes.push((plane, bytes));
                    }
                    for (plane, bytes) in planes {
                        decode = decode.with(plane, bytes);
                    }
                    decode =
                        decode.with("slot_ids", first("slot_ids", &i32_bytes(&[index[item]]))?);
                } else {
                    let table = i32_bytes(&index[item * verify.pages..][..verify.pages]);
                    decode = decode.with("block_tables", first("block_tables", &table)?);
                }
                launches.push(DecodeLaunch {
                    label: format!("item {item} row {row} (position {})", prefix + row),
                    images: decode,
                    compared: vec![Compared {
                        verify: ("output".into(), token * out_row..(token + 1) * out_row),
                        decode: ("output".into(), 0..out_row),
                    }],
                });
            }
        }
        Ok(launches)
    };
    Ok(Decoded::new(case, cut, launches))
}

/// The oracle's outputs at `live` items: the attention over the stored
/// planes and, for a fold, the planes it writes.
fn oracle(
    layout: &Layout,
    rope: Option<Rope>,
    scales: [f32; 2],
    images: &Images,
    live: u32,
) -> LabResult<Vec<(String, Expected)>> {
    let query = layout.query;
    let dtype = query.kv_dtype;
    let r = layout.rows;
    let widen = |path: &str| -> LabResult<Vec<f64>> {
        Ok(images
            .u16s(path)?
            .into_iter()
            .map(|bits| f64::from(bf16_to_f32(bits)))
            .collect())
    };
    let totals = images
        .i32s("seq_lens")?
        .into_iter()
        .map(|t| t as usize)
        .collect::<Vec<_>>();
    let n = totals.len();
    let prefix = totals.iter().map(|t| t - r).collect::<Vec<_>>();
    let index = images.i32s(layout.index_path())?;
    let mut keys = plane_values(images.bytes("cache.k")?, dtype, scales[0]);
    let mut values = plane_values(images.bytes("cache.v")?, dtype, scales[1]);
    let q_rows = widen("q")?;
    let mut results = Vec::new();
    // A fold rotates each live decode row's Q and K, scales its V and
    // stores K and V at its position before attending ([`fold`]).
    let (q, slack) = match rope {
        None => (q_rows, None),
        Some(rope) => {
            let at = totals.iter().map(|t| t - 1).collect::<Vec<_>>();
            let folded = fold(layout, rope, scales, images, live, &q_rows, &at, &index)?;
            keys = folded.keys;
            values = folded.values;
            results.extend(folded.planes);
            (folded.q, Some(folded.slack))
        }
    };
    let sink = if query.sink {
        Some(widen("sink")?)
    } else {
        None
    };
    let mut output = if layout.ring() {
        let k_new = (!layout.decode()).then(|| widen("k_new")).transpose()?;
        let v_new = (!layout.decode()).then(|| widen("v_new")).transpose()?;
        oracle::ring_attention(
            &query,
            &RingAttention {
                q: &q,
                keys: &keys,
                values: &values,
                slot_ids: &index.iter().map(|&s| s as usize).collect::<Vec<_>>(),
                query_lengths: &vec![r; n],
                prefix_lengths: &prefix,
                k_new: k_new.as_deref(),
                v_new: v_new.as_deref(),
                sink_logits: sink.as_deref(),
            },
        )
    } else {
        let page = query.page_size as usize;
        let page_tables = totals
            .iter()
            .enumerate()
            .map(|(item, total)| {
                index[item * layout.pages..][..total.div_ceil(page)]
                    .iter()
                    .map(|&slot| slot as usize)
                    .collect()
            })
            .collect::<Vec<Vec<usize>>>();
        oracle::attention(
            &query,
            &PagedAttention {
                q: &q,
                keys: &keys,
                values: &values,
                page_tables: &page_tables,
                query_lengths: &vec![r; n],
                query_starts: &prefix,
                key_lengths: &totals,
                sink_logits: sink.as_deref(),
            },
        )
    }
    .map_err(|e| e.0.to_string())?;
    if let Some((q_slack, k_slack)) = slack {
        let planes = Planes {
            q: &q,
            keys: &keys,
            values: &values,
        };
        widen_by_fold(
            layout,
            &index,
            &totals,
            planes,
            (&q_slack, &k_slack),
            sink.as_deref(),
            &mut output,
        );
    }
    results.push(("output".to_owned(), Expected::Values(output)));
    Ok(results)
}

/// How far a stored key the cache.k comparison admits may be from the
/// oracle's stored key `own`, in value units: the comparison admits the
/// BF16 values or E4M3 codes that round the F32 values within `allowance`
/// of the F64 reference `value`, FP8 in code units of `scale`
/// (`conformance::admitted_range`), so the slack takes the same range.
fn key_slack(value: f64, allowance: f64, own: f64, dtype: DType, scale: f32) -> LabResult<f64> {
    let units = if dtype == DType::Fp8E4M3 {
        f64::from(scale)
    } else {
        1.0
    };
    let (low, high) = admitted_range(dtype, value / units, allowance / units)?;
    Ok((low * units - own).abs().max((high * units - own).abs()))
}

/// A fold's rotated queries, the planes it attends over and stores, and
/// how far the device's rotated Q and stored K may be from the oracle's.
struct Folded {
    q: Vec<f64>,
    keys: Vec<f64>,
    values: Vec<f64>,
    /// The expected `cache.k` and `cache.v`.
    planes: Vec<(String, Expected)>,
    slack: (Vec<f64>, Vec<f64>),
}

/// The fold of the first `live` items' decode rows at positions `at`: Q
/// and K rotated, V scaled, and K and V stored at each row's position. The
/// attention reads each stored key and value as the cache holds it, an
/// FP8 code decoded or a BF16 value; `cache.k` compares within the
/// rotation's allowance in its own units, `cache.v` as exact bits.
#[allow(clippy::too_many_arguments)]
fn fold(
    layout: &Layout,
    rope: Rope,
    scales: [f32; 2],
    images: &Images,
    live: u32,
    q_rows: &[f64],
    at: &[usize],
    index: &[i32],
) -> LabResult<Folded> {
    let query = layout.query;
    let (qh, kvh) = (query.q_heads as usize, query.kv_heads as usize);
    let (qk, v) = (query.qk_dim as usize, query.v_dim as usize);
    let dtype = query.kv_dtype;
    let n = at.len();
    let mut keys = plane_values(images.bytes("cache.k")?, dtype, scales[0]);
    let mut values = plane_values(images.bytes("cache.v")?, dtype, scales[1]);
    let width = layout.q_width();
    let part = |from: usize, len: usize| {
        q_rows
            .chunks_exact(width)
            .flat_map(|row| row[from..from + len].to_vec())
            .collect::<Vec<_>>()
    };
    let (q, q_allow) = rope.apply(images, "rope.cos_sin", &part(0, qh * qk), qh, at)?;
    let (k, k_allow) = rope.apply(images, "rope.cos_sin", &part(qh * qk, kvh * qk), kvh, at)?;
    let new_v = part(qh * qk + kvh * qk, kvh * v)
        .into_iter()
        .map(|x| x * f64::from(FOLD_V_MULTIPLIER))
        .collect::<Vec<_>>();
    // FP8 planes compare in code units.
    let units = if dtype == DType::Fp8E4M3 {
        f64::from(scales[0])
    } else {
        1.0
    };
    let mut k_plane = keys.iter().map(|x| x / units).collect::<Vec<_>>();
    let mut k_bound = vec![0.0; k_plane.len()];
    let mut v_bits = images.bytes("cache.v")?.to_vec();
    let mut k_slack = vec![0.0; n * kvh * qk];
    for item in 0..live as usize {
        let (slot, token) = layout.place(index, item, at[item]);
        let row = kvh * qk;
        let start = slot * layout.slot(qk) + token * row;
        for e in 0..row {
            let value = k[item * row + e];
            keys[start + e] = stored(value, dtype, scales[0]);
            k_plane[start + e] = value / units;
            let allow = k_allow[item * row + e];
            k_bound[start + e] = allow / units;
            // The fold attends over the key it stores, which the cache.k
            // comparison holds to the values or codes that round its
            // allowance's ends.
            k_slack[item * row + e] = key_slack(value, allow, keys[start + e], dtype, scales[0])?;
        }
        let row = kvh * v;
        let start = slot * layout.slot(v) + token * row;
        for e in 0..row {
            let value = new_v[item * row + e];
            // The attention reads the value as the cache stores it.
            values[start + e] = stored(value, dtype, scales[1]);
            if dtype == DType::Fp8E4M3 {
                v_bits[start + e] = kernel_core::kv_fp8::encode(value as f32, scales[1]);
            } else {
                v_bits[(start + e) * 2..][..2].copy_from_slice(&bf16_bytes(&[
                    kernel_core::codec::f32_to_bf16(value as f32),
                ]));
            }
        }
    }
    let sums = k_bound.iter().map(|a| a * 2_f64.powi(22)).collect();
    let planes = vec![
        (
            "cache.k".to_owned(),
            Expected::Values(OracleOutput::new(k_plane, sums, 1).map_err(|e| e.0.to_string())?),
        ),
        ("cache.v".to_owned(), Expected::Bits(v_bits)),
    ];
    // The device's rotated Q is within its allowance of the oracle's, and
    // within half a BF16 step more if it rounds it.
    let q_slack = q_allow
        .iter()
        .zip(&q)
        .map(|(a, x)| a + half_bf16(x.abs() + a))
        .collect::<Vec<_>>();
    Ok(Folded {
        q,
        keys,
        values,
        planes,
        slack: (q_slack, k_slack),
    })
}

/// The rotated queries and the stored planes a fold attends over.
struct Planes<'p> {
    q: &'p [f64],
    keys: &'p [f64],
    values: &'p [f64],
}

/// Widens a folded decode's output bound by the effect of the rotated Q and
/// the current key differing from the oracle's by their slack. Key `i`'s
/// logit moves by at most `D_i = scale (sum |dq| |k_i| + sum (|q| + |dq|)
/// |dk|)`, the second sum for the current key only (a softcap only shrinks the
/// move). With `D` the largest, each weight `p_i` moves to within a factor
/// `exp(D_i + D)` of itself and a sink's within `exp(D)`, and since the
/// weights' changes sum to zero, each output `o` moves by at most
/// `sum_i p_i (exp(D_i + D) - 1) |v_i - o| + p_sink (exp(D) - 1) |o|`.
fn widen_by_fold(
    layout: &Layout,
    index: &[i32],
    totals: &[usize],
    planes: Planes<'_>,
    (q_slack, k_slack): (&[f64], &[f64]),
    sink: Option<&[f64]>,
    output: &mut OracleOutput,
) {
    let query = layout.query;
    let (qh, kvh) = (query.q_heads as usize, query.kv_heads as usize);
    let (qk, v) = (query.qk_dim as usize, query.v_dim as usize);
    let group = qh / kvh;
    let scale = f64::from(query.softmax_scale);
    let window = query
        .window_left
        .map_or(usize::MAX, |left| left as usize + 1);
    for (item, &total) in totals.iter().enumerate() {
        for head in 0..qh {
            let kv = head / group;
            let row = &planes.q[(item * qh + head) * qk..][..qk];
            let dq = &q_slack[(item * qh + head) * qk..][..qk];
            // Each attended key's logit, its largest move and its value.
            let mut keys = Vec::new();
            for position in total.saturating_sub(window)..total {
                let (slot, token) = layout.place(index, item, position);
                let key = &planes.keys[slot * layout.slot(qk) + (token * kvh + kv) * qk..][..qk];
                let mut logit = scale * row.iter().zip(key).map(|(q, k)| q * k).sum::<f64>();
                if let Some(cap) = query.softcap {
                    let cap = f64::from(cap);
                    logit = cap * (logit / cap).tanh();
                }
                let mut moved = dq.iter().zip(key).map(|(d, k)| d * k.abs()).sum::<f64>();
                if position + 1 == total {
                    let dk = &k_slack[(item * kvh + kv) * qk..][..qk];
                    // (q + dq)(k + dk) - q k = dq k + q dk + dq dk.
                    moved += row
                        .iter()
                        .zip(dq)
                        .zip(dk)
                        .map(|((q, dq), dk)| (q.abs() + dq) * dk)
                        .sum::<f64>();
                }
                let value = &planes.values[slot * layout.slot(v) + (token * kvh + kv) * v..][..v];
                keys.push((logit, scale * moved, value));
            }
            let sink = sink.map(|logits| logits[head]);
            let largest = keys.iter().map(|&(_, moved, _)| moved).fold(0.0, f64::max);
            let top = keys
                .iter()
                .map(|&(logit, ..)| logit)
                .chain(sink)
                .fold(f64::NEG_INFINITY, f64::max);
            let sink_weight = sink.map_or(0.0, |logit| (logit - top).exp());
            let total_weight = keys
                .iter()
                .map(|&(logit, ..)| (logit - top).exp())
                .sum::<f64>()
                + sink_weight;
            for d in 0..v {
                let o = keys
                    .iter()
                    .map(|&(logit, _, value)| (logit - top).exp() * value[d])
                    .sum::<f64>()
                    / total_weight;
                let moved = keys
                    .iter()
                    .map(|&(logit, moved, value)| {
                        (logit - top).exp() * (moved + largest).exp_m1() * (value[d] - o).abs()
                    })
                    .sum::<f64>()
                    + sink_weight * largest.exp_m1() * o.abs();
                let at = (item * qh + head) * v + d;
                let k = output.reduction_lengths[at] as f64;
                output.sum_abs_products[at] += moved / total_weight * 2_f64.powi(22) / k;
            }
        }
    }
}

/// The required attention `op` over `items` items at `need`; a verify
/// op's one-row bits map to the decode op of its query ([`decoded`]).
pub fn attention<'a>(name: String, op: AttentionOp, items: u32, need: Need) -> Required<'a> {
    let required = Required::new(
        name,
        need,
        move |backend, need| backend.attention(&op, need),
        move |prepared| attention_case(op, items, prepared),
    );
    if op.query.phase != AttentionPhase::Verify {
        return required;
    }
    required.with_decode(move |backend| {
        backend
            .attention(&decode_op(&op), Need::rows(CAPACITY))
            .map(|prepared| decoded(op, items, prepared))
    })
}

/// The attention variants of the lab's catalogs, one per query and layout
/// (two variants of one query are an SM120 tie): of equal ones, the one
/// with the most fold and exact rows entries, else the first. A variant
/// with those entries also serves the unfolded op through its phase
/// entries.
pub fn variants(catalogs: &kernels_sm120::Catalogs) -> Vec<Variant> {
    let key = |v: &Variant| {
        (
            AttentionQuery {
                max_batch_capacity: 0,
                ..v.query
            },
            v.layout,
        )
    };
    let features = |v: &Variant| {
        u8::from(v.rope_fold.is_some()) + u8::from(v.gated_fold.is_some()) + u8::from(v.exact_rows)
    };
    let mut chosen: Vec<Variant> = Vec::new();
    for variant in kernels_sm120::attention::catalog(catalogs) {
        match chosen.iter_mut().find(|other| key(other) == key(&variant)) {
            Some(held) if features(&variant) > features(held) => *held = variant,
            Some(_) => {}
            None => chosen.push(variant),
        }
    }
    chosen
}

/// SM120's attention entries for [`family`].
pub fn sm120_entries(variants: Vec<Variant>) -> AttentionEntries {
    AttentionEntries {
        variants,
        per_row_exact: false,
        rows_above: None,
    }
}

fn describe(op: &AttentionOp, items: u32) -> String {
    let query = op.query;
    let fold = match op.fold {
        None => String::new(),
        Some(AttentionFold::Neox { rotary, source, .. }) => {
            format!(" fold neox {rotary} {source:?}")
        }
        Some(AttentionFold::Gated { .. }) => " fold gated".into(),
    };
    format!(
        "attention {:?} {:?} q{} kv{} {}/{} page {} {:?} scale {}{}{}{}{}{} items {items} rows {}",
        query.phase,
        query.layout,
        query.q_heads,
        query.kv_heads,
        query.qk_dim,
        query.v_dim,
        query.page_size,
        query.kv_dtype,
        query.softmax_scale,
        if query.causal { "" } else { " bidirectional" },
        if query.sink { " sink" } else { "" },
        query
            .window_left
            .map_or_else(String::new, |window| format!(" window {window}")),
        query
            .softcap
            .map_or_else(String::new, |cap| format!(" softcap {cap}")),
        fold,
        op.item_rows,
    )
}

/// The family's required ops over `variants`: per decode variant an
/// unfolded op over [`CAPACITY`] items and, for a variant that folds NeoX
/// RoPE, a fold with computed angles and one with a table; per verify
/// variant an op over [`STEP_ITEMS`] items of up to four rows, and the
/// same op needing one-row bits where a decode variant of its query has
/// exact rows entries; per prefill variant an op over [`STEP_ITEMS`]
/// items of up to eight rows. Gated folds are not listed.
pub fn family<'a>(variants: &[Variant]) -> Vec<Required<'a>> {
    let mut ops = Vec::new();
    for variant in variants {
        let query = variant.query;
        let at = |items: u32| AttentionQuery {
            max_batch_capacity: items,
            ..query
        };
        match query.phase {
            AttentionPhase::Decode => {
                let mut folds = vec![None];
                if let Some(rotary) = variant.rope_fold {
                    for source in [RotarySource::Computed, RotarySource::CosSinTable] {
                        folds.push(Some(AttentionFold::Neox {
                            rotary,
                            theta: FOLD_THETA,
                            v_multiplier: FOLD_V_MULTIPLIER,
                            source,
                        }));
                    }
                }
                for fold in folds {
                    let op = AttentionOp {
                        query: at(CAPACITY),
                        item_rows: 1,
                        fold,
                        plan_at_capacity: false,
                    };
                    ops.push(attention(
                        describe(&op, CAPACITY),
                        op,
                        CAPACITY,
                        Need::rows(CAPACITY),
                    ));
                }
            }
            AttentionPhase::Verify => {
                let rows = variant.query_capacity.min(VERIFY_ROWS);
                let op = AttentionOp {
                    query: at(STEP_ITEMS),
                    item_rows: rows,
                    fold: None,
                    plan_at_capacity: false,
                };
                let need = Need::rows(STEP_ITEMS * rows);
                ops.push(attention(describe(&op, STEP_ITEMS), op, STEP_ITEMS, need));
                let decode = AttentionQuery {
                    phase: AttentionPhase::Decode,
                    max_batch_capacity: 0,
                    ..query
                };
                let exact = variants.iter().any(|other| {
                    other.has_rows()
                        && other.layout == variant.layout
                        && AttentionQuery {
                            max_batch_capacity: 0,
                            ..other.query
                        } == decode
                });
                if exact {
                    // Each verify row's bits are a one-row decode's.
                    let need = need.with_exact(Invariance {
                        rows: kernel_core::properties::Rows::OneRow { max: rows },
                        ..Invariance::NONE
                    });
                    ops.push(attention(
                        format!("{} exact rows", describe(&op, STEP_ITEMS)),
                        op,
                        STEP_ITEMS,
                        need,
                    ));
                }
            }
            AttentionPhase::Prefill => {
                let rows = PREFILL_ROWS.min(variant.query_capacity / STEP_ITEMS).max(1);
                let op = AttentionOp {
                    query: at(STEP_ITEMS),
                    item_rows: rows,
                    fold: None,
                    plan_at_capacity: false,
                };
                ops.push(attention(
                    describe(&op, STEP_ITEMS),
                    op,
                    STEP_ITEMS,
                    Need::rows(STEP_ITEMS * rows),
                ));
            }
        }
    }
    ops
}

/// Self-checks of the family's cases over a consumer's attention variants
/// (as [`variants`] selects them), for its tests: each panics naming the
/// first failure. kernel-lab runs them over the in-tree catalogs.
pub mod checks {
    use super::*;
    use crate::kit::cases::testing::{decode_gaps, prepared};
    use kernel_core::oracle::OracleOutput;

    /// Each exact verify row (paged and ring) maps to the decode launch of
    /// its own row: the decode oracle over a launch's images gives the
    /// verify oracle's row bit for bit (both round the same F64 values),
    /// and a launch of the next row's images does not; a claim of fewer
    /// rows than the items hold refuses to declare. Returns the layouts
    /// checked (paged, ring or both).
    pub fn exact_verify_rows_map_to_their_decode_launches(variants: &[Variant]) -> usize {
        let backend =
            kernels_sm120::backend::Sm120::new().with_attention(sm120_entries(variants.to_vec()));
        let mut layouts = std::collections::BTreeSet::new();
        for required in family(variants) {
            if !required.name.ends_with("exact rows") {
                continue;
            }
            let case = prepared(&backend, &required);
            layouts.insert(required.name.contains("Ring"));
            let gaps = decode_gaps(&case, VERIFY_ROWS, false, None);
            assert_eq!(gaps.len(), (STEP_ITEMS * VERIFY_ROWS) as usize);
            assert!(
                gaps.iter().all(|&gap| gap == 0.0),
                "{}: {gaps:?}",
                required.name
            );
            let shifted = decode_gaps(&case, VERIFY_ROWS, true, None);
            assert!(
                shifted.iter().all(|&gap| gap > 1e-3),
                "{}: {shifted:?}",
                required.name
            );
            // Items of four rows cannot show a claim of two: refused.
            let two = required.need.with_exact(Invariance::one_row(2));
            assert!(case.rows_mapping(two).is_err(), "{}", required.name);
        }
        layouts.len()
    }

    /// Every NeoX fold op of the decode `variants` with a fold, with its
    /// case on SM120.
    pub fn fold_cases(variants: &[Variant]) -> Vec<(AttentionOp, OpCase<'static>)> {
        let backend =
            kernels_sm120::backend::Sm120::new().with_attention(sm120_entries(variants.to_vec()));
        let mut cases = Vec::new();
        for variant in variants {
            let (Some(rotary), AttentionPhase::Decode) = (variant.rope_fold, variant.query.phase)
            else {
                continue;
            };
            for source in [RotarySource::Computed, RotarySource::CosSinTable] {
                let op = AttentionOp {
                    query: AttentionQuery {
                        max_batch_capacity: CAPACITY,
                        ..variant.query
                    },
                    item_rows: 1,
                    fold: Some(AttentionFold::Neox {
                        rotary,
                        theta: FOLD_THETA,
                        v_multiplier: FOLD_V_MULTIPLIER,
                        source,
                    }),
                    plan_at_capacity: false,
                };
                let prepared =
                    kernel_core::backend::Backend::attention(&backend, &op, Need::rows(CAPACITY))
                        .unwrap();
                cases.push((op, attention_case(op, CAPACITY, prepared).unwrap()));
            }
        }
        cases
    }

    /// A fold's inputs on a case's images at `live` items.
    fn folded(op: &AttentionOp, images: &Images, live: u32) -> (Layout, Folded) {
        let layout = Layout::new(op, CAPACITY as usize).unwrap();
        let rope = fold_rope(&op.query, op.fold.unwrap()).unwrap();
        let scales = if op.query.kv_dtype == DType::Fp8E4M3 {
            FP8_SCALES
        } else {
            [1.0; 2]
        };
        let at = images
            .i32s("seq_lens")
            .unwrap()
            .iter()
            .map(|&t| t as usize - 1)
            .collect::<Vec<_>>();
        let q_rows = images
            .u16s("q")
            .unwrap()
            .into_iter()
            .map(|bits| f64::from(bf16_to_f32(bits)))
            .collect::<Vec<_>>();
        let index = images.i32s(layout.index_path()).unwrap();
        let folded = fold(&layout, rope, scales, images, live, &q_rows, &at, &index).unwrap();
        (layout, folded)
    }

    /// A fold's output bound is tight enough to refuse an all-zero output
    /// at every live count, for both angle sources and KV dtypes: at least
    /// an eighth of each live item's outputs lie beyond their bound from
    /// zero (the old bound, twice the largest logit move times the largest
    /// value, admitted zero for every output of FP8 folds).
    pub fn a_fold_bound_refuses_zero_outputs(variants: &[Variant]) {
        for (op, case) in fold_cases(variants) {
            let layout = Layout::new(&op, CAPACITY as usize).unwrap();
            let rope = fold_rope(&op.query, op.fold.unwrap()).unwrap();
            let scales = if op.query.kv_dtype == DType::Fp8E4M3 {
                FP8_SCALES
            } else {
                [1.0; 2]
            };
            let images = case.case.images();
            let width = (op.query.q_heads * op.query.v_dim) as usize;
            for live in [1, 8, 15, 16] {
                let results = oracle(&layout, Some(rope), scales, &images, live).unwrap();
                let Some((_, Expected::Values(output))) =
                    results.iter().find(|(name, _)| name == "output")
                else {
                    panic!("the output compares as values");
                };
                let refused = |index: usize| !admitted(output, index, 0.0);
                for item in 0..live as usize {
                    let refusing = (item * width..(item + 1) * width)
                        .filter(|&index| refused(index))
                        .count();
                    assert!(
                        refusing * 8 >= width,
                        "{} live {live} item {item}: {refusing} of {width} outputs refuse zero",
                        describe(&op, CAPACITY)
                    );
                }
            }
        }
    }

    /// A rotated BF16 key just inside a binade may round one step past its
    /// boundary (GPU: the TP2 computed fold's key -0.49957 at element
    /// 411476 read back as -0.50390625), which the cache.k bound admits.
    /// `variants` hold the paged fold of 32 Q and 2 KV heads.
    pub fn a_fold_key_may_round_past_a_binade(variants: &[Variant]) {
        let (op, case) = fold_cases(variants)
            .into_iter()
            .find(|(op, _)| {
                let name = describe(op, CAPACITY);
                name.contains("Paged q32 kv2") && name.contains("Bf16") && name.contains("Computed")
            })
            .unwrap();
        let (_, folded) = folded(&op, &case.case.images(), CAPACITY);
        let Some((_, Expected::Values(keys))) =
            folded.planes.iter().find(|(name, _)| name == "cache.k")
        else {
            panic!("the fold's keys compare as values");
        };
        let at = 411_476;
        assert!(
            (keys.values[at] + 0.499_57).abs() < 1e-5,
            "{}",
            keys.values[at]
        );
        assert!(admitted(keys, at, -0.503_906_25));
    }

    /// A fold's plane maps follow the items the property checks move.
    pub fn fold_plane_maps_follow_moved_items(variants: &[Variant]) {
        for (_, case) in fold_cases(variants) {
            for plane in ["cache.k", "cache.v"] {
                super::super::testing::assert_maps_follow_rows(&case, plane, 8);
            }
        }
    }

    /// A fold attends over the values it stores: the current rows' V as
    /// the cache holds it (an FP8 code decoded), not as computed.
    pub fn a_fold_attends_over_the_values_it_stores(variants: &[Variant]) {
        for (op, case) in fold_cases(variants) {
            let images = case.case.images();
            let (_, folded) = folded(&op, &images, CAPACITY);
            let Some((_, Expected::Bits(bits))) =
                folded.planes.iter().find(|(name, _)| name == "cache.v")
            else {
                panic!("the fold stores V as bits");
            };
            let scale = if op.query.kv_dtype == DType::Fp8E4M3 {
                FP8_SCALES[1]
            } else {
                1.0
            };
            assert!(
                plane_values(bits, op.query.kv_dtype, scale) == folded.values,
                "{}",
                describe(&op, CAPACITY)
            );
        }
    }

    /// Whether the gate admits BF16 `value` for element `at` of `output`.
    fn admitted(output: &OracleOutput, at: usize, value: f64) -> bool {
        let one = OracleOutput {
            values: vec![output.values[at]],
            sum_abs_products: vec![output.sum_abs_products[at]],
            reduction_lengths: vec![output.reduction_lengths[at]],
        };
        crate::conformance::rounding_admissible(DType::Bf16, &[value as f32], &one).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::oracle::OracleOutput;

    /// The fold's widening covers Q and the current key moving together,
    /// cross term included: with Q = K = 0 and both moving by 0.1 the
    /// logit moves by their product, which two keys of values -1 and 1
    /// turn into an output move; random simultaneous moves within the
    /// slack stay within the widened bound.
    #[test]
    fn the_fold_bound_covers_simultaneous_q_and_key_moves() {
        let dim = 4;
        let query = AttentionQuery {
            q_heads: 1,
            kv_heads: 1,
            qk_dim: dim as u32,
            v_dim: 1,
            page_size: 2,
            kv_dtype: DType::Bf16,
            layout: AttentionLayout::Paged,
            causal: true,
            window_left: None,
            sink: false,
            softcap: None,
            softmax_scale: 1.0,
            phase: AttentionPhase::Decode,
            max_context_capacity: 2,
            max_batch_capacity: 1,
        };
        let layout = Layout {
            query,
            fold: None,
            rows: 1,
            prefix: vec![1],
            slots: 1,
            pages: 1,
            index: vec![0],
        };
        let values = [-1.0, 1.0];
        let slack = 0.1;
        let output_of = |q: &[f64], keys: &[f64]| {
            let logits = (0..2)
                .map(|key| (0..dim).map(|e| q[e] * keys[key * dim + e]).sum::<f64>())
                .collect::<Vec<_>>();
            let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = logits.iter().map(|l| (l - top).exp()).collect::<Vec<_>>();
            weights.iter().zip(values).map(|(w, v)| w * v).sum::<f64>()
                / weights.iter().sum::<f64>()
        };
        let mut state = 7_u64;
        let mut draw = |scale: f64| {
            state = kernel_core::ngram_oracle::splitmix64(state);
            ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * scale
        };
        for trial in 0..64 {
            // The first trial is the zero counterexample; the rest random.
            let (q, keys): (Vec<f64>, Vec<f64>) = if trial == 0 {
                (vec![0.0; dim], vec![0.0; 2 * dim])
            } else {
                (
                    (0..dim).map(|_| draw(1.0)).collect(),
                    (0..2 * dim).map(|_| draw(1.0)).collect(),
                )
            };
            let reference = output_of(&q, &keys);
            let mut output = OracleOutput {
                values: vec![reference],
                sum_abs_products: vec![0.0],
                reduction_lengths: vec![2],
            };
            widen_by_fold(
                &layout,
                &layout.index,
                &[2],
                Planes {
                    q: &q,
                    keys: &keys,
                    values: &values,
                },
                (&vec![slack; dim], &vec![slack; dim]),
                None,
                &mut output,
            );
            let bound = kernel_core::oracle::accumulation_bound(2, output.sum_abs_products[0]);
            // Q and the current key (key 1) move by the full slack, in the
            // signs that move the logit most, and at random.
            for signs in 0..3 {
                let mut sign = |x: f64| match signs {
                    0 => 1.0,
                    1 => x.signum(),
                    _ => draw(1.0).signum(),
                };
                let moved_q = q.iter().map(|&x| x + slack * sign(x)).collect::<Vec<_>>();
                let mut moved_keys = keys.clone();
                for e in 0..dim {
                    moved_keys[dim + e] += slack * sign(keys[dim + e]);
                }
                let error = (output_of(&moved_q, &moved_keys) - reference).abs();
                assert!(
                    error <= bound,
                    "trial {trial} signs {signs}: {error} > {bound}"
                );
            }
        }
    }

    /// The key slack takes the cache gate's range: a reference of
    /// 0.53124999 within 2e-8 at scale 0.5 holds no F32 code-unit value
    /// past the midpoint 1.0625, which rounds to the even code 1.0, so the
    /// gate admits only the oracle's 0.5 and the slack is zero (rounding the
    /// F64 ends admitted code 1.125, 0.0625 away); a BF16 reference of
    /// 0.4990234 within 0.00294 admits 0.50390625 past the binade,
    /// 0.005859375 from the oracle's 0.498046875, more than the allowance
    /// and a step at the reference.
    #[test]
    fn the_key_slack_rounds_the_gates_ends() {
        let (value, allowance, scale) = (0.531_249_99, 2e-8, 0.5_f32);
        let own = stored(value, DType::Fp8E4M3, scale);
        assert_eq!(own, 0.5);
        let slack = key_slack(value, allowance, own, DType::Fp8E4M3, scale).unwrap();
        assert_eq!(slack, 0.0);
        let code = |x: f64| kernel_core::kv_fp8::round_e4m3(x / f64::from(scale));
        assert_eq!(code(value + allowance), 1.125);
        let (value, allowance) = (0.499_023_4, 0.002_94);
        let own = stored(value, DType::Bf16, 1.0);
        assert_eq!(own, 0.498_046_875);
        let slack = key_slack(value, allowance, own, DType::Bf16, 1.0).unwrap();
        assert_eq!(slack, 0.005_859_375);
        assert!(slack > allowance + 2.0 * half_bf16(value));
        // A BF16 interval ending less than an F32 step past the midpoint
        // 1 + 2^-8 holds the midpoint at most, which rounds to 1.
        let mid = 1.0 + 2_f64.powi(-8);
        let (value, allowance) = (mid - 2_f64.powi(-23), 2_f64.powi(-23) + 2_f64.powi(-26));
        let own = stored(value, DType::Bf16, 1.0);
        assert_eq!(own, 1.0);
        let slack = key_slack(value, allowance, own, DType::Bf16, 1.0).unwrap();
        assert_eq!(slack, 0.0);
        // Every value or code the gate admits lies within the slack of the
        // oracle's.
        let mut state = 3_u64;
        for dtype in [DType::Bf16, DType::Fp8E4M3] {
            for _ in 0..1000 {
                state = kernel_core::ngram_oracle::splitmix64(state);
                let value = ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 6.0;
                let allowance = (state & 0xff) as f64 * 1e-4;
                let own = stored(value, dtype, scale);
                let slack = key_slack(value, allowance, own, dtype, scale).unwrap();
                let units = if dtype == DType::Fp8E4M3 {
                    f64::from(scale)
                } else {
                    1.0
                };
                let (low, high) = admitted_range(dtype, value / units, allowance / units).unwrap();
                for admitted in [low * units, high * units] {
                    assert!(
                        (admitted - own).abs() <= slack,
                        "{dtype:?} {value} {allowance}"
                    );
                }
            }
        }
    }

    /// Every page and every token of a plane differs from every other, so
    /// a launch that ignores the block tables or reads another token reads
    /// other values: the oracle over in-order tables differs from the
    /// case's scattered tables for every item past the bound.
    #[test]
    fn pages_and_tokens_are_distinct_and_tables_matter() {
        let (page, width, heads) = (64, 128, 2);
        for dtype in [DType::Bf16, DType::Fp8E4M3] {
            let element = if dtype == DType::Fp8E4M3 { 1 } else { 2 };
            let image = plane(9 * page * width * heads, dtype, 22);
            let tokens = image
                .chunks_exact(width * heads * element)
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(tokens.len(), 9 * page, "{dtype:?}");
        }
        let query = AttentionQuery {
            q_heads: 2,
            kv_heads: 1,
            qk_dim: 128,
            v_dim: 128,
            page_size: page as u32,
            kv_dtype: DType::Fp8E4M3,
            layout: AttentionLayout::Paged,
            causal: true,
            window_left: None,
            sink: false,
            softcap: None,
            softmax_scale: 0.088,
            phase: AttentionPhase::Decode,
            max_context_capacity: 4096,
            max_batch_capacity: 4,
        };
        let lens = (0..4).map(|item| keys(item, page)).collect::<Vec<_>>();
        let used = lens.iter().map(|len| len.div_ceil(page)).sum::<usize>();
        let keys = plane_values(
            &plane((used + 1) * page * 128, DType::Fp8E4M3, 22),
            DType::Fp8E4M3,
            0.5,
        );
        let values = plane_values(
            &plane((used + 1) * page * 128, DType::Fp8E4M3, 23),
            DType::Fp8E4M3,
            0.25,
        );
        let q = bf16_values(21, 4 * 2 * 128)
            .into_iter()
            .map(|bits| f64::from(bf16_to_f32(bits)))
            .collect::<Vec<_>>();
        let attend = |tables: &[Vec<usize>]| {
            oracle::attention(
                &query,
                &PagedAttention {
                    q: &q,
                    keys: &keys,
                    values: &values,
                    page_tables: tables,
                    query_lengths: &[1; 4],
                    query_starts: &lens.iter().map(|len| len - 1).collect::<Vec<_>>(),
                    key_lengths: &lens,
                    sink_logits: None,
                },
            )
            .unwrap()
        };
        let mut next = 0;
        let scattered = lens
            .iter()
            .map(|len| {
                (0..len.div_ceil(page))
                    .map(|_| {
                        next += 1;
                        scattered(next - 1, used)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut first = 0;
        let identity = lens
            .iter()
            .map(|len| {
                let pages = (first..first + len.div_ceil(page)).collect::<Vec<_>>();
                first += pages.len();
                pages
            })
            .collect::<Vec<_>>();
        let (case, ignored) = (attend(&scattered), attend(&identity));
        for item in 0..4 {
            let row = item * 2 * 128..(item + 1) * 2 * 128;
            assert!(
                row.clone().any(|at| {
                    let (low, high) = admits(&case, at);
                    let (other_low, other_high) = admits(&ignored, at);
                    high < other_low || other_high < low
                }),
                "item {item}"
            );
        }
    }

    /// The lowest and highest BF16 values the gate admits for element `at`.
    fn admits(output: &OracleOutput, at: usize) -> (f64, f64) {
        let bound = kernel_core::oracle::accumulation_bound(
            output.reduction_lengths[at],
            output.sum_abs_products[at],
        );
        admitted_range(DType::Bf16, output.values[at], bound).unwrap()
    }

    #[test]
    fn contexts_span_one_to_four_pages_and_end_on_a_page() {
        let lens = (0..16).map(|item| keys(item, 64)).collect::<Vec<_>>();
        assert!(lens.iter().all(|&len| (1..=256).contains(&len)));
        assert!(lens.iter().any(|&len| len % 64 == 0));
        assert!(lens.iter().any(|&len| len > 128));
        // FP8 images hold finite codes of magnitude at most 2.
        assert!(
            plane(4096, DType::Fp8E4M3, 3)
                .iter()
                .all(|&code| SmallFloat::E4M3.decode_f64(code).abs() <= 2.0)
        );
    }
}
