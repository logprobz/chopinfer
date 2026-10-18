//! The row invariance adapter: every `OpKind::Row` entry that claims a
//! property. Three families launch through it:
//!
//! - the lab's row engine cubin (`row::catalog` over every model):
//!   embeddings, MiMo's RMSNorm, combine and router, its SWA ring write and
//!   its RoPE entries with the paged GA KV write or the SWA prefill rows;
//! - the Qwen3.8 gated attention rows (`attention::gated`): the Q/K norm,
//!   M-RoPE and paged KV write, and the output gate;
//! - the Qwen3.8 rows of `row::qwen38`: hyper-connection rows, the PLE gate,
//!   the PLE convolution and the n-gram gather;
//! - the shared row copies (`row::copy`, the lab's `row_copy_sm120`): the
//!   stream broadcast and the column slice, launched as the SM120 backend
//!   prepares them at each launch's rows.
//!
//! Every row's inputs derive from its seed only. A chunk case gives prompt
//! row r seed r and position `BASE + r`, launches each chunk alone and keeps
//! the state the entry writes (the paged KV pool or window ring, the PLE
//! convolution state, the n-gram history) across the chunks; that state
//! follows the rows as one more unit. Other cases make each seed its own
//! item, owning one page or ring slot, whose bytes follow its rows. The PLE
//! convolution and the n-gram read earlier rows, so their units are
//! segments a launch isolates: nine zero rows before each convolution
//! segment (its whole window), and an end-of-sequence token after each
//! n-gram segment.
//!
//! Every compared launch starts with poisoned outputs, each with one row
//! of padding past its prepared capacity, poisoned output state (the PLE
//! convolution's `state_out`, the n-gram `history_out`) and, in the KV pool
//! or window ring, poisoned cache rows the launch writes; the rest of a
//! pool keeps its image. Afterwards every output's live rows must be
//! written (and finite for arithmetic outputs; IDs and row indices are
//! checked as raw words) with the rows past the live count untouched
//! (`written_rows`), the output state written in full, and each pool's
//! written rows (`Cache::maps`) written with every other byte unchanged
//! (`written_cache`).

use super::*;
use crate::invariance::{Adapter, Compared, Element, Units, written, written_over};
use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::contract::{
    ArgumentList, GatedAttentionQuery, LaunchPlan, NgramQuery, RecurrentPhase, RecurrentQuery,
    Refusal, SigmoidMeanHyperConnectionQuery as H, StaticQuery,
};
use kernel_core::mxfp8_rows::{Mxfp8Rows, Mxfp8RowsBuffers};
use kernel_core::ngram_oracle;
use kernel_core::row_copy::{
    ColumnSlice, GatherByIndex, GatherByIndexBuffers, RowCopyBuffers, StreamBroadcast,
};
use kernels_sm120::attention::gated;
use kernels_sm120::backend::Sm120;
use kernels_sm120::invariance::{Declared, OpKind};
use kernels_sm120::row::{Grid, smallest_bucket};
use kernels_sm120::row::{copy, gather, mxfp8_decode};
use model_qwen38_contract::sm120::row as qwen38;
use std::collections::BTreeSet;
use std::ops::Range;

/// Positions the RoPE tables cover; every launch's rows lie below.
const POSITIONS: usize = 8192;
/// The position of a chunk case's first prompt row.
const BASE: u32 = 1000;
/// Tokens of one MiMo SWA ring slot.
const RING: usize = 128;
/// Rows of the MiMo embedding test table.
const VOCAB: usize = 97;
/// Qwen3.8 PLE channels (four streams of 2,560) and the convolution's
/// window of earlier rows (kernel 4, dilation 3).
const WIDE: usize = 10_240;
const WINDOW: usize = 9;
const NGRAM_VOCAB: u64 = 248_064;
const EOS: u32 = 248_044;

/// A seeded 64-bit word for `salt` of `seed` (splitmix64), so row 0 and
/// nearby seeds get unrelated values.
fn mix(seed: u64, salt: u64) -> u64 {
    let mut z = seed ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// `width` BF16 values per seed.
fn rows_of(seeds: &[u64], width: usize, salt: u64) -> Vec<u16> {
    seeds
        .iter()
        .flat_map(|&seed| seeded_bf16(width, mix(seed, salt)))
        .collect()
}

/// A unit case's position for `seed`: past several pages and ring wraps.
fn position(seed: u64) -> i32 {
    37 + (seed % 6000) as i32
}

/// The F32 cos/sin table over `POSITIONS` positions and 32 pairs.
fn cos_sin(theta: f32) -> Vec<f32> {
    (0..POSITIONS)
        .flat_map(|position| {
            (0..32).flat_map(move |pair| {
                let (cos, sin) = kernel_core::hf_rope::cos_sin(position, pair, f64::from(theta));
                [cos as f32, sin as f32]
            })
        })
        .collect()
}

fn bytes_of(session: &Session, buffer: &DeviceBuffer) -> LabResult<Vec<u8>> {
    let mut out = vec![0; buffer.byte_len()];
    buffer
        .copy_to_bytes(&mut out, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(out)
}

/// An output a launch writes: its name, bytes per row and element type.
type Output = (&'static str, usize, Element);

/// Allocates every output poisoned over `rows` rows, the launch's prepared
/// capacity plus a padding row.
fn poison_outputs(f: &mut Fixture<'_>, outputs: &[Output], rows: usize) -> LabResult<()> {
    for &(name, width, _) in outputs {
        f.poisoned(name, rows * width)?;
    }
    Ok(())
}

/// Checks an output buffer after a launch that found it poisoned: each
/// element of its first `live` bytes (the live rows) is written and, for
/// arithmetic outputs, finite; every later byte (rows past the live count
/// up to the prepared capacity, and the padding row) still holds the
/// poison. Every row kernel here writes whole live rows and nothing past
/// `live_rows` (row.cuh, router.cuh, rope_kv.cuh, gated.cuh,
/// qwen38_ple.cuh: each returns at `row >= *live_rows`).
fn written_rows(label: &str, bytes: &[u8], live: usize, element: Element) -> Result<(), String> {
    written(
        label,
        bytes,
        &element.write_map()?.byte_prefix(live)?,
        element,
    )
}

/// Each live row's bytes of every output, in order, after checking each
/// output's writes (`written_rows`).
fn row_outputs(f: &Fixture<'_>, symbol: &str, outputs: &[Output], rows: usize) -> LabResult<Units> {
    let mut units = vec![Vec::new(); rows];
    for &(name, width, element) in outputs {
        let bytes = bytes_of(f.session, &f.buffers[name])?;
        written_rows(&format!("{symbol} {name}"), &bytes, rows * width, element)?;
        for (unit, row) in units.iter_mut().zip(bytes[..rows * width].chunks(width)) {
            unit.extend_from_slice(row);
        }
    }
    Ok(units)
}

/// Launches `symbol` with every parameter bound by name from `f`; a
/// missing buffer or scalar is an error.
fn launch_named(
    f: &Fixture<'_>,
    module: &Module,
    symbol: &str,
    params: &[EntryParam],
    pack: impl FnOnce(Vec<Argument>) -> Result<ArgumentList, Refusal>,
    plan: Result<LaunchPlan, Refusal>,
) -> LabResult<()> {
    let mut bindings = BTreeMap::new();
    let args = params
        .iter()
        .map(|param| match param {
            EntryParam::DevicePointer { name, .. } => {
                let index = f
                    .buffers
                    .keys()
                    .position(|n| n == name)
                    .ok_or_else(|| format!("{symbol}: no buffer {name}"))?;
                let handle = index as u64 + 1;
                bindings.insert(handle, &f.buffers[name]);
                Ok(Argument::DeviceBuffer(handle))
            }
            EntryParam::U32 { name } => f
                .scalars
                .get(name)
                .map(|&v| Argument::U32(v))
                .ok_or_else(|| format!("{symbol}: no scalar {name}")),
            EntryParam::U64 { name } => f
                .wide_scalars
                .get(name)
                .map(|&v| Argument::U64(v))
                .ok_or_else(|| format!("{symbol}: no scalar {name}")),
            _ => Err(format!("{symbol}: unexpected parameter kind")),
        })
        .collect::<LabResult<Vec<_>>>()?;
    let args = pack(args).map_err(|e| format!("{symbol}: {e}"))?;
    let plan = plan.map_err(|e| format!("{symbol}: {e}"))?;
    let function = module.function(symbol).map_err(|e| e.to_string())?;
    launch::launch(
        &f.session.context,
        &function,
        &f.session.stream,
        plan,
        &args,
        &Bindings { buffers: bindings },
    )?;
    kernel_lab::launched(symbol);
    Ok(())
}

/// A cache's K and V pools.
type Pools = (DeviceBuffer, DeviceBuffer);
/// The bytes of a cache's K and V pools.
type PoolBytes = (Vec<u8>, Vec<u8>);

/// Where a launch's items write their cache rows.
#[derive(Clone, Copy)]
enum Layout {
    /// One prompt over every page of its block table, or ring slot 0.
    Prompt,
    /// `n` items of one row each; item i owns page i (every entry of its
    /// block table) or ring slot i.
    Items(usize),
}

impl Layout {
    /// Each of `rows` rows' item.
    fn items(self, rows: usize) -> Vec<i32> {
        match self {
            Layout::Prompt => vec![0; rows],
            Layout::Items(_) => (0..rows as i32).collect(),
        }
    }
}

/// A KV cache a row entry writes: pages or ring slots of `tokens` tokens,
/// with `k_row` and `v_row` bytes per token of `element` (BF16, or FP8
/// E4M3 codes as raw bytes).
#[derive(Clone, Copy)]
struct Cache {
    tokens: usize,
    paged: bool,
    k_row: usize,
    v_row: usize,
    element: Element,
}

/// The cache element of `dtype`.
fn cache_element(dtype: DType) -> Element {
    if dtype == DType::Fp8E4M3 {
        Element::Raw(1)
    } else {
        Element::Bf16
    }
}

impl Cache {
    fn max_pages(self) -> usize {
        POSITIONS / self.tokens
    }

    fn regions(self, layout: Layout) -> usize {
        match layout {
            Layout::Prompt if self.paged => self.max_pages(),
            Layout::Prompt => 1,
            Layout::Items(n) => n,
        }
    }

    /// Zeroed K and V pools for `layout`.
    fn pools(self, session: &Session, layout: Layout) -> LabResult<Pools> {
        let regions = self.regions(layout);
        let zeroed = |bytes: usize| -> LabResult<DeviceBuffer> {
            let buffer = session.allocate(bytes)?;
            buffer
                .copy_from_bytes(&vec![0; bytes], &session.stream)
                .map_err(|e| e.to_string())?;
            Ok(buffer)
        };
        Ok((
            zeroed(regions * self.tokens * self.k_row)?,
            zeroed(regions * self.tokens * self.v_row)?,
        ))
    }

    fn block_tables(self, layout: Layout) -> Vec<i32> {
        let pages = self.max_pages();
        match layout {
            Layout::Prompt => (0..pages as i32).collect(),
            Layout::Items(n) => (0..n as i32)
                .flat_map(|item| std::iter::repeat_n(item, pages))
                .collect(),
        }
    }

    fn slot_ids(layout: Layout) -> Vec<i32> {
        match layout {
            Layout::Prompt => vec![0],
            Layout::Items(n) => (0..n as i32).collect(),
        }
    }

    /// Each region's K then V bytes.
    fn read(self, session: &Session, pools: &Pools, layout: Layout) -> LabResult<Units> {
        let (k, v) = (bytes_of(session, &pools.0)?, bytes_of(session, &pools.1)?);
        let (k_region, v_region) = (self.tokens * self.k_row, self.tokens * self.v_row);
        Ok((0..self.regions(layout))
            .map(|r| {
                [
                    &k[r * k_region..][..k_region],
                    &v[r * v_region..][..v_region],
                ]
                .concat()
            })
            .collect())
    }

    /// The K and V pool rows a launch writes: row r of `writers` (rows of
    /// `items` at `positions`) stores one token, every KV head of it, at
    /// pool token `page * tokens + position % tokens` with `page` its
    /// item's block table entry for `position / tokens`, or, in a ring, at
    /// `slot * tokens + position % tokens` with `slot` its item's slot ID
    /// (the README's page/slot, token, KV head, dimension layout;
    /// `rope_kv`, `ring_write` in rope_kv.cuh, `qk_norm_rope_kv` in
    /// gated.cuh). Rows sharing a token write it once in the map.
    fn maps(
        self,
        layout: Layout,
        items: &[i32],
        positions: &[i32],
        writers: &[usize],
    ) -> LabResult<(WriteMap, WriteMap)> {
        let (tables, slots) = (self.block_tables(layout), Self::slot_ids(layout));
        let tokens = writers
            .iter()
            .map(|&r| {
                let (item, position) = (items[r] as usize, positions[r] as usize);
                let region = if self.paged {
                    tables[item * self.max_pages() + position / self.tokens] as usize
                } else {
                    slots[item] as usize
                };
                region * self.tokens + position % self.tokens
            })
            .collect::<BTreeSet<_>>();
        let e = self.element.write_map()?;
        let rows = |row: usize| tokens.iter().map(move |&t| t * row..(t + 1) * row);
        Ok((
            e.byte_ranges(rows(self.k_row))?,
            e.byte_ranges(rows(self.v_row))?,
        ))
    }
}

/// Checks a K or V pool after a launch against its image `before` it, in
/// which the rows of `map` (`Cache::maps`) were poisoned: each of them is
/// written (finite for BF16) and every other byte kept its image. A launch
/// that writes no cache rows (the SWA prefill RoPE, which leaves the ring
/// to `ring_write`) must leave the whole pool unchanged.
fn written_cache(
    label: &str,
    before: &[u8],
    after: &[u8],
    map: &WriteMap,
    element: Element,
) -> Result<(), String> {
    if map.written_bytes() > 0 {
        return written_over(label, before, after, map, element);
    }
    if before.len() != after.len() {
        return Err(format!("{label}: the pool changed size"));
    }
    match before.iter().zip(after).position(|(a, b)| a != b) {
        Some(at) => Err(format!(
            "{label}: byte {at} changed from {:#04x} to {:#04x}; the launch writes no cache",
            before[at], after[at]
        )),
        None => Ok(()),
    }
}

/// Each item's last position among rows of `items` at `positions`: the
/// `last_positions` a ring write reads.
fn last_positions(items: &[i32], positions: &[i32]) -> Vec<i64> {
    let mut last = vec![0_i64; items.iter().max().map_or(0, |&i| i as usize + 1)];
    for (&item, &p) in items.iter().zip(positions) {
        last[item as usize] = last[item as usize].max(i64::from(p));
    }
    last
}

/// The rows a ring write stores: those within `RING` of their item's last
/// position (`ring_write` in rope_kv.cuh skips a row at
/// `position + ring_size <= last_positions[batch]`).
fn ring_tail(items: &[i32], positions: &[i32]) -> Vec<usize> {
    let last = last_positions(items, positions);
    (0..items.len())
        .filter(|&r| i64::from(positions[r]) + RING as i64 > last[items[r] as usize])
        .collect()
}

/// Poisons the rows of `maps` in `pools` and returns each pool's image.
fn poison_cache(
    session: &Session,
    pools: &Pools,
    maps: &(WriteMap, WriteMap),
) -> LabResult<PoolBytes> {
    poison::fill(session, &pools.0, maps.0.ranges(), &[POISON])?;
    poison::fill(session, &pools.1, maps.1.ranges(), &[POISON])?;
    Ok((bytes_of(session, &pools.0)?, bytes_of(session, &pools.1)?))
}

/// Checks both pools after a launch (`written_cache`).
fn check_cache(
    session: &Session,
    label: &str,
    cache: Cache,
    pools: &Pools,
    maps: &(WriteMap, WriteMap),
    before: &PoolBytes,
) -> LabResult<()> {
    let after = (bytes_of(session, &pools.0)?, bytes_of(session, &pools.1)?);
    written_cache(
        &format!("{label} K pool"),
        &before.0,
        &after.0,
        &maps.0,
        cache.element,
    )?;
    written_cache(
        &format!("{label} V pool"),
        &before.1,
        &after.1,
        &maps.1,
        cache.element,
    )?;
    Ok(())
}

/// One launch of row-local rows: each row's output bytes for rows seeded
/// by `seeds` at `positions`, writing the entry's cache in `pools`.
type Launch<'a> = dyn FnMut(&[u64], &[i32], Layout, Option<&Pools>) -> LabResult<Units> + 'a;

/// A chunk case of a row-local entry: prompt row r has seed r at position
/// `BASE + r`; the cache the chunks share follows the rows.
fn local_chunks(
    session: &Session,
    cache: Option<Cache>,
    ends: &[u32],
    launch: &mut Launch<'_>,
) -> LabResult<Units> {
    let pools = cache
        .map(|c| c.pools(session, Layout::Prompt))
        .transpose()?;
    let mut units = Vec::new();
    let mut start = 0;
    for &end in ends {
        let seeds = (start..end).map(u64::from).collect::<Vec<_>>();
        let positions = (start..end).map(|r| (BASE + r) as i32).collect::<Vec<_>>();
        // A ring write's rows output nothing but the ring.
        let rows = launch(&seeds, &positions, Layout::Prompt, pools.as_ref())?;
        units.extend(rows.into_iter().filter(|row| !row.is_empty()));
        start = end;
    }
    if let (Some(cache), Some(pools)) = (cache, &pools) {
        units.push(cache.read(session, pools, Layout::Prompt)?.concat());
    }
    Ok(units)
}

/// Each seed's row as its own item, followed by its item's page or slot.
fn local_units(
    session: &Session,
    cache: Option<Cache>,
    seeds: &[u64],
    launch: &mut Launch<'_>,
) -> LabResult<Units> {
    let layout = Layout::Items(seeds.len());
    let positions = seeds.iter().map(|&s| position(s)).collect::<Vec<_>>();
    let pools = cache.map(|c| c.pools(session, layout)).transpose()?;
    let mut units = launch(seeds, &positions, layout, pools.as_ref())?;
    if let (Some(cache), Some(pools)) = (cache, &pools) {
        for (unit, region) in units.iter_mut().zip(cache.read(session, pools, layout)?) {
            unit.extend(region);
        }
    }
    Ok(units)
}

/// A row engine entry's operation.
#[derive(Clone, Copy)]
enum Mimo {
    Embedding {
        width: usize,
    },
    Rms {
        width: usize,
    },
    Combine {
        width: usize,
        top_k: usize,
    },
    Router {
        experts: usize,
        top_k: usize,
    },
    Ring {
        kv_heads: usize,
        cache: Cache,
    },
    /// `writes`: the entry stores K and V in its cache (the paged GA
    /// write); the SWA prefill rows only emit `k_new` and `v_new`.
    Rope {
        q_heads: usize,
        kv_heads: usize,
        theta: f32,
        cache: Cache,
        writes: bool,
    },
}

impl Mimo {
    fn of(entry: &RowInstantiation) -> LabResult<Self> {
        let symbol = &entry.symbol;
        let refuse = || Err(format!("{symbol}: no row adapter for {:?}", entry.query));
        let pos32 = symbol.ends_with("_pos32_v1");
        Ok(match entry.query {
            RowQuery::EmbeddingGather { width, .. } => Mimo::Embedding {
                width: width as usize,
            },
            RowQuery::RmsNorm {
                width,
                fused_residual: false,
                output_quantization: None,
                ..
            } => Mimo::Rms {
                width: width as usize,
            },
            RowQuery::ExpertCombine { top_k, .. } => Mimo::Combine {
                width: entry.width as usize,
                top_k: top_k as usize,
            },
            RowQuery::Router { experts, top_k, .. } => Mimo::Router {
                experts: experts as usize,
                top_k: top_k as usize,
            },
            RowQuery::RingWrite {
                qk_dim: 192,
                v_dim: 128,
                kv_heads,
                ring_size: 128,
                kv_dtype,
            } if pos32 && !symbol.contains("single_sequence") => {
                let element = if kv_dtype == DType::Fp8E4M3 { 1 } else { 2 };
                let kv_heads = kv_heads as usize;
                Mimo::Ring {
                    kv_heads,
                    cache: Cache {
                        tokens: RING,
                        paged: false,
                        k_row: kv_heads * ROPE_QK * element,
                        v_row: kv_heads * ROPE_V * element,
                        element: cache_element(kv_dtype),
                    },
                }
            }
            RowQuery::Rope {
                head_dim: 192,
                rotary_dim: 64,
                theta,
                fused_paged_kv_write,
                ..
            } if pos32 => {
                let Grid::RowsHeads(groups) = entry.grid else {
                    return refuse();
                };
                let q_heads = groups as usize * 4;
                let kv_heads = (entry.width as usize - q_heads * ROPE_QK) / (ROPE_QK + ROPE_V);
                let dtype = if symbol.contains("_fp8_") {
                    DType::Fp8E4M3
                } else {
                    DType::Bf16
                };
                let element = cache_element(dtype).bytes();
                // The page size is the symbol's `_p{N}_` part.
                let page = symbol
                    .split('_')
                    .find_map(|part| part.strip_prefix('p')?.parse::<usize>().ok())
                    .ok_or_else(|| format!("{symbol}: no page size"))?;
                Mimo::Rope {
                    q_heads,
                    kv_heads,
                    theta,
                    cache: Cache {
                        tokens: if fused_paged_kv_write { page } else { RING },
                        paged: fused_paged_kv_write,
                        k_row: kv_heads * ROPE_QK * element,
                        v_row: kv_heads * ROPE_V * element,
                        element: cache_element(dtype),
                    },
                    // `rope_kv` stores the cache unless `Ring && Prefill`.
                    writes: fused_paged_kv_write || !symbol.contains("_prefill_"),
                }
            }
            _ => return refuse(),
        })
    }

    fn cache(self) -> Option<Cache> {
        match self {
            Mimo::Ring { cache, .. } | Mimo::Rope { cache, .. } => Some(cache),
            _ => None,
        }
    }

    /// The rows of `items` at `positions` that store their K and V in the
    /// cache: a ring write's tail (`ring_tail`), every row of a RoPE entry
    /// that writes the cache, else none.
    fn cache_writers(self, items: &[i32], positions: &[i32]) -> Vec<usize> {
        match self {
            Mimo::Ring { .. } => ring_tail(items, positions),
            Mimo::Rope { writes: true, .. } => (0..items.len()).collect(),
            _ => Vec::new(),
        }
    }
}

/// One row engine launch of `entry`; each row's outputs.
fn mimo_rows(
    session: &Session,
    entry: &RowInstantiation,
    op: Mimo,
    seeds: &[u64],
    positions: &[i32],
    layout: Layout,
    pools: Option<&Pools>,
) -> LabResult<Units> {
    let rows = seeds.len();
    let items = layout.items(rows);
    let mut f = Fixture::new(session, rows)?;
    let outputs = match op {
        Mimo::Embedding { width } => {
            f.bf16("table", &seeded_bf16(VOCAB * width, 0x9101))?;
            let ids = seeds
                .iter()
                .map(|&s| (mix(s, 1) % VOCAB as u64) as i32)
                .collect::<Vec<_>>();
            f.i32("ids", &ids)?;
            f.scalar("vocab_offset", 0);
            f.scalar("shard_vocab", VOCAB as u32);
            vec![("output", width * 2, Element::Bf16)]
        }
        Mimo::Rms { width } => {
            f.bf16("x", &rows_of(seeds, width, 2))?;
            f.bf16("weight", &seeded_bf16(width, 0x9103))?;
            vec![("y", width * 2, Element::Bf16)]
        }
        Mimo::Combine { width, top_k } => {
            f.bf16("routed", &rows_of(seeds, top_k * width, 3))?;
            let weights = seeds
                .iter()
                .flat_map(|&s| seeded_f32(top_k, mix(s, 4)))
                .collect::<Vec<_>>();
            f.f32("weights", &weights)?;
            vec![("output", width * 2, Element::Bf16)]
        }
        Mimo::Router { experts, top_k } => {
            let logits = seeds
                .iter()
                .flat_map(|&s| seeded_f32(experts, mix(s, 5)))
                .map(|x| x * 4.0)
                .collect::<Vec<_>>();
            f.f32("logits", &logits)?;
            let bias = seeded_f32(experts, 0x9105);
            f.f32("bias", &bias.iter().map(|x| x * 0.1).collect::<Vec<_>>())?;
            vec![
                ("ids", top_k * 4, Element::Raw(4)),
                ("weights", top_k * 4, Element::F32),
            ]
        }
        Mimo::Ring { kv_heads, .. } => {
            f.bf16("k_new", &rows_of(seeds, kv_heads * ROPE_QK, 6))?;
            f.bf16("v_new", &rows_of(seeds, kv_heads * ROPE_V, 7))?;
            f.bytes(
                "last_positions",
                &last_positions(&items, positions)
                    .iter()
                    .flat_map(|p| p.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )?;
            Vec::new()
        }
        Mimo::Rope {
            q_heads,
            kv_heads,
            theta,
            ..
        } => {
            let width = q_heads * ROPE_QK + kv_heads * (ROPE_QK + ROPE_V);
            f.bf16("qkv", &rows_of(seeds, width, 8))?;
            f.f32("cos_sin", &cos_sin(theta))?;
            f.scalar("theta_bits", theta.to_bits());
            f.scalar("v_multiplier_bits", ROPE_V_MULTIPLIER.to_bits());
            vec![
                ("q", q_heads * ROPE_QK * 2, Element::Bf16),
                ("k_new", kv_heads * ROPE_QK * 2, Element::Bf16),
                ("v_new", kv_heads * ROPE_V * 2, Element::Bf16),
            ]
        }
    };
    let mut written_cache_rows = None;
    if let Some(cache) = op.cache() {
        let pools = pools.ok_or_else(|| format!("{}: no cache pools", entry.symbol))?;
        let (k, v) = pools;
        f.buffers.insert("k_pool".into(), k.clone());
        f.buffers.insert("v_pool".into(), v.clone());
        f.i32("token_batch", &items)?;
        f.i32("positions", positions)?;
        f.i32("slot_ids", &Cache::slot_ids(layout))?;
        f.i32("block_tables", &cache.block_tables(layout))?;
        f.scalar("max_pages", cache.max_pages() as u32);
        f.scalar("ring_size", RING as u32);
        f.scalar64("k_page_stride_bytes", (cache.tokens * cache.k_row) as u64);
        f.scalar64("v_page_stride_bytes", (cache.tokens * cache.v_row) as u64);
        f.scalar("k_scale_bits", 1.0_f32.to_bits());
        f.scalar("v_scale_bits", 1.0_f32.to_bits());
        let writers = op.cache_writers(&items, positions);
        let maps = cache.maps(layout, &items, positions, &writers)?;
        let before = poison_cache(session, pools, &maps)?;
        written_cache_rows = Some((cache, pools, maps, before));
    }
    let bucket = smallest_bucket(rows as u32).ok_or("no row bucket")?;
    poison_outputs(&mut f, &outputs, bucket as usize + 1)?;
    f.launch(&module(session)?, entry, bucket, rows as u32)?;
    if let Some((cache, pools, maps, before)) = &written_cache_rows {
        check_cache(session, &entry.symbol, *cache, pools, maps, before)?;
    }
    row_outputs(&f, &entry.symbol, &outputs, rows)
}

/// The paged KV cache of a gated Q/K norm entry.
fn gated_cache(entry: &gated::Entry) -> Option<Cache> {
    let GatedAttentionQuery::QkNormRopeKvWrite {
        kv_dtype,
        kv_heads,
        head_dim,
        page_size,
        ..
    } = entry.query
    else {
        return None;
    };
    let element = cache_element(kv_dtype);
    let row = (kv_heads * head_dim) as usize * element.bytes();
    Some(Cache {
        tokens: page_size as usize,
        paged: true,
        k_row: row,
        v_row: row,
        element,
    })
}

/// One launch of a gated attention row entry; each row's outputs.
fn gated_rows(
    session: &Session,
    entry: &gated::Entry,
    seeds: &[u64],
    positions: &[i32],
    layout: Layout,
    pools: Option<&Pools>,
) -> LabResult<Units> {
    let rows = seeds.len();
    let d = match entry.query {
        GatedAttentionQuery::QkNormRopeKvWrite { head_dim, .. }
        | GatedAttentionQuery::OutputGate { head_dim, .. } => head_dim as usize,
    };
    let mut f = Fixture::new(session, rows)?;
    let items = layout.items(rows);
    let mut written_cache_rows = None;
    let outputs = match entry.query {
        GatedAttentionQuery::QkNormRopeKvWrite {
            q_heads,
            kv_heads,
            kv_dtype,
            rotary_dim,
            theta_bits,
            mrope,
            eps_bits,
            ..
        } => {
            let qh = q_heads as usize;
            let cache = gated_cache(entry).ok_or("a gated cache")?;
            let (k, v) = pools.ok_or_else(|| format!("{}: no cache pools", entry.symbol))?;
            f.bf16("qkv", &rows_of(seeds, (qh + kv_heads as usize) * 2 * d, 11))?;
            f.bf16("q_norm", &seeded_bf16(d, 0x9201))?;
            f.bf16("k_norm", &seeded_bf16(d, 0x9202))?;
            // Text rows with the H and W axes moved off the position.
            let rope = seeds
                .iter()
                .zip(positions)
                .flat_map(|(&s, &p)| [p, (p - 1 - (s % 3) as i32).max(0), p / 2])
                .collect::<Vec<_>>();
            f.i32("rope_positions", &rope)?;
            f.f32(
                "cos_sin",
                &(0..POSITIONS)
                    .flat_map(|position| {
                        kernel_core::gated_attention_oracle::mrope_cos_sin(
                            [position; 3],
                            rotary_dim as usize,
                            f64::from(f32::from_bits(theta_bits)),
                            mrope,
                        )
                        .into_iter()
                        .flat_map(|(c, s)| [c as f32, s as f32])
                    })
                    .collect::<Vec<_>>(),
            )?;
            f.i32("row_batch", &items)?;
            f.i32("positions", positions)?;
            f.i32("block_tables", &cache.block_tables(layout))?;
            f.buffers.insert("k_cache".into(), k.clone());
            f.buffers.insert("v_cache".into(), v.clone());
            f.scalar("max_pages", cache.max_pages() as u32);
            f.scalar64("k_page_stride_bytes", (cache.tokens * cache.k_row) as u64);
            f.scalar64("v_page_stride_bytes", (cache.tokens * cache.v_row) as u64);
            f.scalar("eps_bits", eps_bits);
            // Unequal FP8 scales, as the bundle may set them.
            let (k_scale, v_scale) = if kv_dtype == DType::Fp8E4M3 {
                (0.5_f32, 0.25_f32)
            } else {
                (1.0, 1.0)
            };
            f.scalar("k_scale_bits", k_scale.to_bits());
            f.scalar("v_scale_bits", v_scale.to_bits());
            // Every live row writes its K and V (`qk_norm_rope_kv`).
            let writers = (0..rows).collect::<Vec<_>>();
            let maps = cache.maps(layout, &items, positions, &writers)?;
            let before = poison_cache(session, pools.ok_or("gated pools")?, &maps)?;
            written_cache_rows = Some((cache, maps, before));
            vec![("q", qh * d * 2, Element::Bf16)]
        }
        GatedAttentionQuery::OutputGate { heads, .. } => {
            let h = heads as usize;
            f.bf16("attention", &rows_of(seeds, h * d, 12))?;
            f.bf16("qkv", &rows_of(seeds, (h + 1) * 2 * d, 11))?;
            vec![("out", h * d * 2, Element::Bf16)]
        }
    };
    // The plan's capacity is the live count; one padding row follows.
    poison_outputs(&mut f, &outputs, rows + 1)?;
    let module = session.module(&format!("attention_{}_sm120", entry.profile.tag()))?;
    launch_named(
        &f,
        &module,
        &entry.symbol,
        &entry.params,
        |args| entry.pack_arguments(args),
        entry.plan(rows as u32),
    )?;
    if let (Some((cache, maps, before)), Some(pools)) = (&written_cache_rows, pools) {
        check_cache(session, &entry.symbol, *cache, pools, maps, before)?;
    }
    row_outputs(&f, &entry.symbol, &outputs, rows)
}

/// A `row::qwen38` entry's operation.
#[derive(Clone, Copy)]
enum Qwen {
    /// BF16 `inputs` (name, elements per row) and a shared `weight` of that
    /// many elements, giving `output` elements per row; `conv` carries the
    /// PLE convolution state.
    Rows {
        inputs: &'static [(&'static str, usize)],
        weight: usize,
        output: usize,
        conv: bool,
    },
    /// The n-gram gather of `local` hash heads.
    Ngram { local: usize },
}

impl Qwen {
    fn of(entry: &kernels_sm120::row::hyper_connection::Entry) -> LabResult<Self> {
        fn rows(inputs: &'static [(&'static str, usize)], weight: usize, output: usize) -> Qwen {
            Qwen::Rows {
                inputs,
                weight,
                output,
                conv: false,
            }
        }
        Ok(match &entry.query {
            StaticQuery::HyperConnection(H::GroupedNorm { .. }) => rows(&[("x", WIDE)], WIDE, WIDE),
            StaticQuery::HyperConnection(H::Bottleneck { .. }) => rows(&[("x", 320)], 0, 320),
            StaticQuery::HyperConnection(H::Mix { .. }) => {
                rows(&[("up", WIDE), ("normed", WIDE)], 0, 2560)
            }
            StaticQuery::HyperConnection(H::Inject { .. }) => {
                rows(&[("x", WIDE), ("y", 2560), ("logits", 4)], 0, WIDE)
            }
            StaticQuery::Ngram(NgramQuery::PleGate { .. }) => {
                rows(&[("key", WIDE), ("query", WIDE), ("value", 2560)], 0, WIDE)
            }
            StaticQuery::Recurrent(RecurrentQuery::CausalConv1d {
                phase: RecurrentPhase::Prefill,
                ..
            }) => Qwen::Rows {
                inputs: &[("input", WIDE)],
                weight: 4 * WIDE,
                output: WIDE,
                conv: true,
            },
            StaticQuery::Ngram(NgramQuery::HashGather { local_heads, .. })
                if !entry.symbol.contains("decode") =>
            {
                Qwen::Ngram {
                    local: *local_heads as usize,
                }
            }
            other => return Err(format!("{}: no row adapter for {other:?}", entry.symbol)),
        })
    }

    /// The state a first chunk reads: the convolution's window, or the
    /// n-gram history (ending in end-of-sequence for isolated segments).
    fn initial(self, isolated: bool) -> Vec<u8> {
        match self {
            Qwen::Rows { conv: false, .. } => Vec::new(),
            Qwen::Rows { conv: true, .. } => seeded_bf16(WINDOW * WIDE, 0x9302)
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
            Qwen::Ngram { .. } => [19, if isolated { EOS } else { 23 }]
                .iter()
                .flat_map(|t: &u32| t.to_ne_bytes())
                .collect(),
        }
    }
}

/// One launch of a Qwen3.8 row entry over `rows` (a seeded row, or `None`
/// for a zero row or an end-of-sequence token) from `state`; each row's
/// outputs and the state it leaves.
fn qwen_rows(
    session: &Session,
    entry: &kernels_sm120::row::hyper_connection::Entry,
    op: Qwen,
    rows: &[Option<u64>],
    state: &[u8],
) -> LabResult<(Units, Vec<u8>)> {
    let n = rows.len();
    let mut f = Fixture::new(session, n)?;
    let (outputs, state_out) = match op {
        Qwen::Rows {
            inputs,
            weight,
            output,
            conv,
        } => {
            for (i, &(name, width)) in inputs.iter().enumerate() {
                let values = rows
                    .iter()
                    .flat_map(|row| match row {
                        Some(seed) => seeded_bf16(width, mix(*seed, 20 + i as u64)),
                        None => vec![0; width],
                    })
                    .collect::<Vec<_>>();
                f.bf16(name, &values)?;
            }
            if weight > 0 {
                f.bf16("weight", &seeded_bf16(weight, 0x9301))?;
            }
            if conv {
                f.bytes("state", state)?;
                f.poisoned("state_out", state.len())?;
            }
            (
                vec![("output", output * 2, Element::Bf16)],
                conv.then_some("state_out"),
            )
        }
        Qwen::Ngram { local } => {
            let tables = ngram_oracle::ngram_tables(NGRAM_VOCAB, 3, 8, 17, 0, 1, 1)
                .map_err(|e| e.to_string())?;
            // The last rank's heads, whose trigram heads read two tokens back.
            let rank = 16 / local - 1;
            let first = rank * local;
            let table_rows = tables.sizes[first..first + local].iter().sum::<u64>();
            let tokens = rows
                .iter()
                .map(|row| row.map_or(EOS, |seed| (mix(seed, 30) % NGRAM_VOCAB) as u32))
                .collect::<Vec<_>>();
            f.u32("tokens", &tokens)?;
            f.bytes("history", state)?;
            for (name, values) in [
                ("multipliers", &tables.multipliers),
                ("sizes", &tables.sizes),
                ("offsets", &tables.offsets),
            ] {
                f.bytes(
                    name,
                    &values
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )?;
            }
            // E4M3 codes of both signs, never NaN.
            let table = (0..table_rows as usize * 160)
                .map(|i| (i % 127) as u8 | if i % 2 == 0 { 0 } else { 128 })
                .collect::<Vec<_>>();
            f.bytes("table", &table)?;
            f.bf16("scale", &[f32_to_bf16(0.125)])?;
            f.poisoned("history_out", 8)?;
            f.scalar("rank", rank as u32);
            f.scalar64("table_start", tables.offsets[first]);
            f.scalar64("table_rows", table_rows);
            (
                vec![
                    ("output", local * 160 * 2, Element::Bf16),
                    // Exact U64 global table rows.
                    ("row_ids", local * 8, Element::Raw(8)),
                ],
                Some("history_out"),
            )
        }
    };
    // The plan's capacity is the live count; one padding row follows.
    poison_outputs(&mut f, &outputs, n + 1)?;
    let module = session.module(&format!("row_{}", entry.profile.tag()))?;
    launch_named(
        &f,
        &module,
        &entry.symbol,
        &entry.params,
        |args| entry.pack_arguments(args),
        entry.plan(n as u32, n as u32),
    )?;
    let state = match state_out {
        Some(name) => {
            let bytes = bytes_of(session, &f.buffers[name])?;
            written_state_out(op, &format!("{} {name}", entry.symbol), &bytes)?;
            bytes
        }
        None => Vec::new(),
    };
    Ok((row_outputs(&f, &entry.symbol, &outputs, n)?, state))
}

/// Checks a Qwen entry's output state after a launch that found it
/// poisoned: a live launch writes all of it (qwen38_ple.cuh: row 0 writes
/// every one of the nine BF16 convolution history rows; the last row
/// writes both U32 n-gram history IDs).
fn written_state_out(op: Qwen, label: &str, bytes: &[u8]) -> Result<(), String> {
    let element = match op {
        Qwen::Rows { .. } => Element::Bf16,
        Qwen::Ngram { .. } => Element::Raw(4),
    };
    written_rows(label, bytes, bytes.len(), element)
}

/// The rows of a Qwen unit case and each unit's rows: one seeded row per
/// unit for row-local entries, else segments of one to four rows, each
/// after nine zero rows (convolution) or before an end-of-sequence token
/// (n-gram, whose token row belongs to the segment).
fn segments(op: Qwen, seeds: &[u64]) -> (Vec<Option<u64>>, Vec<Range<usize>>) {
    let mut rows = Vec::new();
    let mut spans = Vec::new();
    for &seed in seeds {
        let own = (0..1 + seed % 4).map(|k| Some(mix(seed, k)));
        match op {
            Qwen::Rows { conv: false, .. } => {
                rows.push(Some(seed));
                spans.push(rows.len() - 1..rows.len());
            }
            Qwen::Rows { conv: true, .. } => {
                rows.extend([None; WINDOW]);
                let start = rows.len();
                rows.extend(own);
                spans.push(start..rows.len());
            }
            Qwen::Ngram { .. } => {
                let start = rows.len();
                rows.extend(own);
                rows.push(None);
                spans.push(start..rows.len());
            }
        }
    }
    (rows, spans)
}

/// An entry of one of the adapter's families.
enum Family {
    Mimo(RowInstantiation, Mimo),
    Gated(gated::Entry),
    Qwen(kernels_sm120::row::hyper_connection::Entry, Qwen),
    Copy(copy::Entry),
    Gather(String),
    Mxfp8(String),
}

fn family(symbol: &str) -> LabResult<Family> {
    if let Some(entry) = catalog().into_iter().find(|e| e.symbol == symbol) {
        let op = Mimo::of(&entry)?;
        return Ok(Family::Mimo(entry, op));
    }
    if let Some(entry) = gated::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|e| e.symbol == symbol)
    {
        return Ok(Family::Gated(entry));
    }
    if let Some(entry) = qwen38::hyper_connection_rows()
        .into_iter()
        .find(|e| e.symbol == symbol)
    {
        let op = Qwen::of(&entry)?;
        return Ok(Family::Qwen(entry, op));
    }
    if let Some(entry) = copy::entries().into_iter().find(|e| e.symbol == symbol) {
        return Ok(Family::Copy(entry));
    }
    if symbol == mxfp8_decode::SYMBOL {
        return Ok(Family::Mxfp8(symbol.into()));
    }
    if symbol == gather::SYMBOL {
        return Ok(Family::Gather(symbol.into()));
    }
    Err(format!("{symbol}: not in a row adapter catalog"))
}

#[test]
fn claimed_lab_gated_entries_resolve_to_rows() {
    let entries = (kernel_lab::lab_shapes::CATALOG.gated)();
    assert_eq!(
        entries.len(),
        9 + 3
            * (kernel_lab::lab_shapes::QUALIFICATION_SHAPES.len()
                + kernel_lab::lab_shapes::NEW_ROW_SHAPES.len())
    );
    let inventory = crate::invariance::inventory();
    for entry in entries {
        let declared = inventory.iter().find(|d| d.symbol == entry.symbol).unwrap();
        assert_eq!(declared.invariance, entry.invariance);
        assert_eq!(declared.invariance.claims().len(), 3);
        assert!(matches!(family(&entry.symbol).unwrap(), Family::Gated(_)));
    }
}

/// The shape each shared copy runs at: four streams of BF16 rows of 512,
/// and 256 columns from 512 of BF16 rows of 1024.
fn copy_op(pattern: copy::Pattern) -> CopyOp {
    match pattern {
        copy::Pattern::BroadcastStreams => CopyOp::Broadcast(StreamBroadcast {
            dtype: DType::Bf16,
            width: 512,
            streams: 4,
            alignment_bytes: 16,
        }),
        copy::Pattern::ColumnSlice => CopyOp::Slice(ColumnSlice {
            dtype: DType::Bf16,
            width: 256,
            stride: 1024,
            offset: 512,
            alignment_bytes: 16,
        }),
    }
}

#[derive(Clone, Copy)]
enum CopyOp {
    Broadcast(StreamBroadcast),
    Slice(ColumnSlice),
}

/// One launch of the shared copy `entry` over a source row per seed, as the
/// SM120 backend prepares it at those rows: each unit is one row's
/// destination bytes, written whole, with the poisoned padding row past
/// them untouched.
fn copy_rows(session: &Session, entry: &copy::Entry, seeds: &[u64]) -> LabResult<Units> {
    let rows = seeds.len();
    let backend = Sm120::new().with_row_copy(entry.clone());
    let need = Need::rows(rows as u32);
    let (prepared, source_width, destination_row) = match copy_op(entry.pattern) {
        CopyOp::Broadcast(op) => (
            backend.broadcast_streams(&op, need),
            op.width as usize,
            op.destination_row_bytes().map_err(|e| e.to_string())? as usize,
        ),
        CopyOp::Slice(op) => (
            backend.column_slice(&op, need),
            op.stride as usize,
            op.destination_row_bytes().map_err(|e| e.to_string())? as usize,
        ),
    };
    let prepared = prepared.map_err(|e| format!("{}: {e}", entry.symbol))?;
    let source_bytes = rows_of(seeds, source_width, 70)
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect::<Vec<_>>();
    let upload = |bytes: &[u8]| -> LabResult<DeviceBuffer> {
        let buffer = session.allocate(bytes.len())?;
        poison::write(session, &buffer, 0, bytes)?;
        Ok(buffer)
    };
    let source = upload(&source_bytes)?;
    let destination = upload(&vec![POISON; (rows + 1) * destination_row])?;
    let live = upload(&(rows as u32).to_le_bytes())?;
    let view = |handle, buffer: &DeviceBuffer| View {
        handle,
        bytes: buffer.byte_len() as u64,
    };
    let buffers = RowCopyBuffers {
        source: view(1, &source),
        destination: view(2, &destination),
        live_rows: view(3, &live),
    };
    let launches = prepared
        .launches(&buffers, View::EMPTY, Counts::rows(rows as u32))
        .map_err(|e| format!("{}: {e}", entry.symbol))?;
    let module = session.module(copy::LAB_CUBIN)?;
    let bindings = Bindings {
        buffers: BTreeMap::from([(1, &source), (2, &destination), (3, &live)]),
    };
    for one in &launches {
        let function = module.function(&one.symbol).map_err(|e| e.to_string())?;
        launch::launch(
            &session.context,
            &function,
            &session.stream,
            one.plan,
            &one.arguments,
            &bindings,
        )?;
        kernel_lab::launched(&one.symbol);
    }
    let bytes = bytes_of(session, &destination)?;
    written_rows(
        &format!("{} destination", entry.symbol),
        &bytes,
        rows * destination_row,
        Element::Bf16,
    )?;
    Ok(bytes[..rows * destination_row]
        .chunks(destination_row)
        .map(<[u8]>::to_vec)
        .collect())
}

/// The shape the shared gather runs at: BF16 rows of 512.
const GATHER: GatherByIndex = GatherByIndex {
    dtype: DType::Bf16,
    width: 512,
    alignment_bytes: 16,
};

/// One launch of the shared gather `symbol` over a source row per seed,
/// laid out in reverse so destination row `r` gathers the source row of
/// seed `r` from the other end, as the SM120 backend prepares it at those
/// rows: each unit is one row's destination bytes, written whole, with the
/// poisoned padding row past them untouched.
fn gather_rows(session: &Session, symbol: &str, seeds: &[u64]) -> LabResult<Units> {
    let rows = seeds.len();
    let backend = Sm120::new().with_row_gather(symbol);
    let prepared = backend
        .gather_by_index(&GATHER, Need::rows(rows as u32))
        .map_err(|e| format!("{symbol}: {e}"))?;
    let row = GATHER.row_bytes().map_err(|e| e.to_string())? as usize;
    let mut source_rows = rows_of(seeds, GATHER.width as usize, 70);
    source_rows.reverse();
    let source_bytes = source_rows
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect::<Vec<_>>();
    let indices = (0..rows as u32)
        .rev()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let upload = |bytes: &[u8]| -> LabResult<DeviceBuffer> {
        let buffer = session.allocate(bytes.len())?;
        poison::write(session, &buffer, 0, bytes)?;
        Ok(buffer)
    };
    let source = upload(&source_bytes)?;
    let row_indices = upload(&indices)?;
    let count = upload(&(rows as u32).to_le_bytes())?;
    let destination = upload(&vec![POISON; (rows + 1) * row])?;
    let view = |handle, buffer: &DeviceBuffer| View {
        handle,
        bytes: buffer.byte_len() as u64,
    };
    let buffers = GatherByIndexBuffers {
        source: view(1, &source),
        row_indices: view(2, &row_indices),
        count: view(3, &count),
        destination: view(4, &destination),
    };
    let launches = prepared
        .launches(&buffers, View::EMPTY, Counts::rows(rows as u32))
        .map_err(|e| format!("{symbol}: {e}"))?;
    let module = session.module(gather::CUBIN)?;
    let bindings = Bindings {
        buffers: BTreeMap::from([
            (1, &source),
            (2, &row_indices),
            (3, &count),
            (4, &destination),
        ]),
    };
    for one in &launches {
        let function = module.function(&one.symbol).map_err(|e| e.to_string())?;
        launch::launch(
            &session.context,
            &function,
            &session.stream,
            one.plan,
            &one.arguments,
            &bindings,
        )?;
        kernel_lab::launched(&one.symbol);
    }
    let bytes = bytes_of(session, &destination)?;
    written_rows(
        &format!("{symbol} destination"),
        &bytes,
        rows * row,
        Element::Bf16,
    )?;
    Ok(bytes[..rows * row]
        .chunks(row)
        .map(<[u8]>::to_vec)
        .collect())
}

/// Decode independently seeded rows, including non-finite values, checking
/// every output bit and the extra poisoned row against the oracle.
fn mxfp8_rows(session: &Session, symbol: &str, seeds: &[u64]) -> LabResult<Units> {
    let rows = seeds.len();
    let backend = Sm120::new().with_mxfp8_rows(symbol);
    let prepared = backend
        .decode_mxfp8_rows(&Mxfp8Rows { width: 128 }, Need::rows(rows as u32))
        .map_err(|e| format!("{symbol}: {e}"))?;
    let width = 128;
    let row = width * 2;
    let codes = seeds
        .iter()
        .flat_map(|&seed| {
            (0..width).map(move |col| seed.wrapping_mul(29).wrapping_add(col as u64 * 17) as u8)
        })
        .collect::<Vec<_>>();
    let scales = seeds
        .iter()
        .flat_map(|&seed| {
            (0..width / 32).map(move |col| seed.wrapping_mul(7).wrapping_add(col as u64 * 53) as u8)
        })
        .collect::<Vec<_>>();
    let upload = |bytes: &[u8]| -> LabResult<DeviceBuffer> {
        let buffer = session.allocate(bytes.len())?;
        poison::write(session, &buffer, 0, bytes)?;
        Ok(buffer)
    };
    let source = upload(&codes)?;
    let row_scales = upload(&scales)?;
    let count = upload(&(rows as u32).to_le_bytes())?;
    let destination = upload(&vec![POISON; (rows + 1) * row])?;
    let view = |handle, buffer: &DeviceBuffer| View {
        handle,
        bytes: buffer.byte_len() as u64,
    };
    let buffers = Mxfp8RowsBuffers {
        codes: view(1, &source),
        scales: view(2, &row_scales),
        live_rows: view(3, &count),
        output: view(4, &destination),
    };
    let launches = prepared
        .launches(&buffers, View::EMPTY, Counts::rows(rows as u32))
        .map_err(|e| format!("{symbol}: {e}"))?;
    let module = session.module(mxfp8_decode::CUBIN)?;
    let bindings = Bindings {
        buffers: BTreeMap::from([
            (1, &source),
            (2, &row_scales),
            (3, &count),
            (4, &destination),
        ]),
    };
    for one in &launches {
        let function = module.function(&one.symbol).map_err(|e| e.to_string())?;
        launch::launch(
            &session.context,
            &function,
            &session.stream,
            one.plan,
            &one.arguments,
            &bindings,
        )?;
        kernel_lab::launched(&one.symbol);
    }
    let bytes = bytes_of(session, &destination)?;
    let mut expected = vec![u16::from_le_bytes([POISON; 2]); (rows + 1) * width];
    kernel_core::mxfp8_rows::decode_rows(
        &Mxfp8Rows {
            width: width as u32,
        },
        &codes,
        &scales,
        rows as u32,
        &mut expected,
    )
    .map_err(|e| e.to_string())?;
    let expected = expected
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    if bytes != expected {
        return Err(format!(
            "{symbol}: decoded rows or untouched tail differ from the oracle"
        ));
    }
    Ok(bytes[..rows * row]
        .chunks(row)
        .map(<[u8]>::to_vec)
        .collect())
}

/// The shared gather entry resolves to the gather family and prepares
/// with its declared claims (no GPU).
#[test]
fn the_shared_gather_entry_resolves_to_the_gather() {
    let inventory = crate::invariance::inventory();
    let declared = inventory
        .iter()
        .find(|d| d.symbol == gather::SYMBOL)
        .unwrap();
    assert_eq!(declared.op, OpKind::Row);
    assert!(matches!(family(gather::SYMBOL).unwrap(), Family::Gather(_)));
    let prepared = Sm120::new()
        .with_row_gather(gather::SYMBOL)
        .gather_by_index(&GATHER, Need::rows(512))
        .unwrap();
    assert_eq!(prepared.info().guarantees, declared.invariance);
}

#[test]
fn the_mxfp8_entry_resolves_to_the_decoder() {
    let inventory = crate::invariance::inventory();
    let declared = inventory
        .iter()
        .find(|d| d.symbol == mxfp8_decode::SYMBOL)
        .unwrap();
    assert_eq!(declared.op, OpKind::Row);
    assert!(matches!(
        family(mxfp8_decode::SYMBOL).unwrap(),
        Family::Mxfp8(_)
    ));
    let prepared = Sm120::new()
        .with_mxfp8_rows(mxfp8_decode::SYMBOL)
        .decode_mxfp8_rows(&Mxfp8Rows { width: 128 }, Need::rows(512))
        .unwrap();
    assert_eq!(prepared.info().guarantees, declared.invariance);
}

/// Every shared copy entry resolves to the copy family, and the copy
/// shapes prepare (no GPU).
#[test]
fn shared_copy_entries_resolve_to_copies() {
    let inventory = crate::invariance::inventory();
    for entry in copy::entries() {
        let declared = inventory.iter().find(|d| d.symbol == entry.symbol).unwrap();
        assert_eq!(declared.op, OpKind::Row);
        assert!(matches!(family(&entry.symbol).unwrap(), Family::Copy(_)));
        let backend = Sm120::new().with_row_copy(entry.clone());
        let prepared = match copy_op(entry.pattern) {
            CopyOp::Broadcast(op) => backend.broadcast_streams(&op, Need::rows(512)),
            CopyOp::Slice(op) => backend.column_slice(&op, Need::rows(512)),
        };
        assert_eq!(prepared.unwrap().info().guarantees, declared.invariance);
    }
}

#[derive(Default)]
pub(crate) struct Row;

impl Adapter for Row {
    fn kind(&self) -> OpKind {
        OpKind::Row
    }

    fn chunks(
        &mut self,
        sessions: &[Session],
        entry: &Declared,
        _rows: u32,
        ends: &[u32],
    ) -> LabResult<Units> {
        let session = &sessions[0];
        match family(&entry.symbol)? {
            Family::Mimo(row, op) => local_chunks(session, op.cache(), ends, &mut |s, p, l, c| {
                mimo_rows(session, &row, op, s, p, l, c)
            }),
            Family::Gated(gated) => {
                local_chunks(session, gated_cache(&gated), ends, &mut |s, p, l, c| {
                    gated_rows(session, &gated, s, p, l, c)
                })
            }
            Family::Qwen(qwen, op) => {
                let mut state = op.initial(false);
                let mut units = Vec::new();
                let mut start = 0;
                for &end in ends {
                    let rows = (start..end).map(|r| Some(u64::from(r))).collect::<Vec<_>>();
                    let (outputs, next) = qwen_rows(session, &qwen, op, &rows, &state)?;
                    units.extend(outputs);
                    state = next;
                    start = end;
                }
                if !state.is_empty() {
                    units.push(state);
                }
                Ok(units)
            }
            Family::Copy(entry) => {
                let mut units = Vec::new();
                let mut start = 0;
                for &end in ends {
                    let seeds = (start..end).map(u64::from).collect::<Vec<_>>();
                    units.extend(copy_rows(session, &entry, &seeds)?);
                    start = end;
                }
                Ok(units)
            }
            Family::Gather(symbol) => {
                let mut units = Vec::new();
                let mut start = 0;
                for &end in ends {
                    let seeds = (start..end).map(u64::from).collect::<Vec<_>>();
                    units.extend(gather_rows(session, &symbol, &seeds)?);
                    start = end;
                }
                Ok(units)
            }
            Family::Mxfp8(symbol) => {
                let mut units = Vec::new();
                let mut start = 0;
                for &end in ends {
                    let seeds = (start..end).map(u64::from).collect::<Vec<_>>();
                    units.extend(mxfp8_rows(session, &symbol, &seeds)?);
                    start = end;
                }
                Ok(units)
            }
        }
    }

    fn units(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let session = &sessions[0];
        match family(&entry.symbol)? {
            Family::Mimo(row, op) => local_units(session, op.cache(), seeds, &mut |s, p, l, c| {
                mimo_rows(session, &row, op, s, p, l, c)
            }),
            Family::Gated(gated) => {
                local_units(session, gated_cache(&gated), seeds, &mut |s, p, l, c| {
                    gated_rows(session, &gated, s, p, l, c)
                })
            }
            Family::Qwen(qwen, op) => {
                let (rows, spans) = segments(op, seeds);
                let (outputs, _) = qwen_rows(session, &qwen, op, &rows, &op.initial(true))?;
                Ok(spans
                    .into_iter()
                    .map(|span| outputs[span].concat())
                    .collect())
            }
            Family::Copy(entry) => copy_rows(session, &entry, seeds),
            Family::Gather(symbol) => gather_rows(session, &symbol, seeds),
            Family::Mxfp8(symbol) => mxfp8_rows(session, &symbol, seeds),
        }
    }

    /// The router's top-k over rows of exact ties: each expert's score is
    /// one of six well-separated classes (three logits, two biases), so
    /// equal classes are bitwise equal and the selection must take each
    /// class's experts by ascending index, as the host orders them.
    fn ties(&mut self, sessions: &[Session], entry: &Declared) -> LabResult<Vec<Compared>> {
        let session = &sessions[0];
        let Family::Mimo(row, Mimo::Router { experts, top_k }) = family(&entry.symbol)? else {
            return Err(format!("{}: no tie adapter", entry.symbol));
        };
        const ROWS: usize = 16;
        let levels = [3.0_f32, 0.0, -3.0];
        let bias = (0..experts as u64)
            .map(|e| if mix(e, 40) % 2 == 0 { 0.25_f32 } else { 0.0 })
            .collect::<Vec<_>>();
        // Row 0 ties every logit; row r > 0 puts each expert on the top
        // logit with odds (r % 4) / 64, so the top-k ends inside the top
        // class or the next, and the others on the lower two.
        let logits = (0..ROWS)
            .map(|r| {
                (0..experts)
                    .map(|e| {
                        let draw = mix(r as u64, 41 + e as u64);
                        if r == 0 {
                            levels[1]
                        } else if draw % 64 < (r % 4) as u64 {
                            levels[0]
                        } else {
                            levels[1 + (draw % 2) as usize]
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut f = Fixture::new(session, ROWS)?;
        f.f32("logits", &logits.concat())?;
        f.f32("bias", &bias)?;
        let outputs = [
            ("ids", top_k * 4, Element::Raw(4)),
            ("weights", top_k * 4, Element::F32),
        ];
        let bucket = smallest_bucket(ROWS as u32).ok_or("no row bucket")?;
        poison_outputs(&mut f, &outputs, bucket as usize + 1)?;
        f.launch(&module(session)?, &row, bucket, ROWS as u32)?;
        // Checks both outputs' writes; the units are unused.
        row_outputs(&f, &row.symbol, &outputs, ROWS)?;
        let ids = f.read_i32("ids", ROWS * top_k)?;
        Ok(logits
            .iter()
            .enumerate()
            .map(|(r, row)| {
                let score =
                    |e: usize| 1.0 / (1.0 + (-f64::from(row[e])).exp()) + f64::from(bias[e]);
                let mut order = (0..experts).collect::<Vec<_>>();
                order.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(a.cmp(&b)));
                let bytes = |ids: &[i32]| {
                    ids.iter()
                        .flat_map(|id| id.to_ne_bytes())
                        .collect::<Vec<_>>()
                };
                Compared {
                    label: format!("tie row {r} ids"),
                    got: bytes(&ids[r * top_k..][..top_k]),
                    expected: bytes(&order[..top_k].iter().map(|&e| e as i32).collect::<Vec<_>>()),
                }
            })
            .collect())
    }
}

/// The adapter's write checks reject unwritten, non-finite and misplaced
/// writes and accept correct images (no GPU).
#[test]
fn row_write_checks_reject_unwritten_outputs() {
    let one = 0x3f80_u16.to_ne_bytes();
    let nan = 0x7fc0_u16.to_ne_bytes();
    // A BF16 output of two live rows of two elements and a padding row.
    let mut output = vec![POISON; 12];
    output[..8].copy_from_slice(&one.repeat(4));
    let rows = |bytes: &[u8]| written_rows("output", bytes, 8, Element::Bf16);
    assert!(rows(&output).is_ok());
    // Unwritten: every byte still the poison.
    assert!(rows(&[POISON; 12]).is_err());
    // A hole in a live row.
    let mut hole = output.clone();
    hole[6..8].fill(POISON);
    assert!(rows(&hole).is_err());
    // A live element written NaN, as from poisoned input.
    let mut not_finite = output.clone();
    not_finite[2..4].copy_from_slice(&nan);
    assert!(rows(&not_finite).is_err());
    // A write into the padding row.
    let mut padding = output.clone();
    padding[10..12].copy_from_slice(&one);
    assert!(rows(&padding).is_err());

    // Router IDs are raw words: any written value passes, the poison and
    // padding writes do not. Its F32 weights must be finite.
    let id_bits = 0x7fc0_0000_u32.to_ne_bytes();
    let mut ids = vec![POISON; 12];
    ids[..8].copy_from_slice(&id_bits.repeat(2));
    assert!(written_rows("ids", &ids, 8, Element::Raw(4)).is_ok());
    assert!(written_rows("ids", &ids, 8, Element::F32).is_err());
    let mut unwritten_id = ids.clone();
    unwritten_id[4..8].fill(POISON);
    assert!(written_rows("ids", &unwritten_id, 8, Element::Raw(4)).is_err());
    let mut padding_id = ids.clone();
    padding_id[8..12].fill(0);
    assert!(written_rows("ids", &padding_id, 8, Element::Raw(4)).is_err());
    // N-gram row IDs are U64 words; one half written is still a hole.
    let mut row_ids = vec![POISON; 24];
    row_ids[..16].fill(0x01);
    assert!(written_rows("row_ids", &row_ids, 16, Element::Raw(8)).is_ok());
    row_ids[8..16].fill(POISON);
    assert!(written_rows("row_ids", &row_ids, 16, Element::Raw(8)).is_err());

    // Output state is written in full: the BF16 convolution history and
    // the two U32 n-gram history IDs.
    let conv = Qwen::Rows {
        inputs: &[],
        weight: 0,
        output: 0,
        conv: true,
    };
    let state = one.repeat(6);
    assert!(written_state_out(conv, "state_out", &state).is_ok());
    assert!(written_state_out(conv, "state_out", &[POISON; 12]).is_err());
    let mut state_nan = state.clone();
    state_nan[4..6].copy_from_slice(&nan);
    assert!(written_state_out(conv, "state_out", &state_nan).is_err());
    let ngram = Qwen::Ngram { local: 4 };
    let mut history = [19_u32, EOS]
        .iter()
        .flat_map(|t| t.to_ne_bytes())
        .collect::<Vec<_>>();
    assert!(written_state_out(ngram, "history_out", &history).is_ok());
    history[4..].fill(POISON);
    assert!(written_state_out(ngram, "history_out", &history).is_err());

    // A ring write keeps only the last RING positions of each item.
    let items = [vec![0; 130], vec![1; 3]].concat();
    let positions = (0..130).chain(500..503).collect::<Vec<i32>>();
    let tail = ring_tail(&items, &positions);
    assert_eq!(tail, (2..133).collect::<Vec<_>>());
    let ring = Mimo::Ring {
        kv_heads: 1,
        cache: Cache {
            tokens: RING,
            paged: false,
            k_row: 4,
            v_row: 2,
            element: Element::Bf16,
        },
    };
    assert_eq!(ring.cache_writers(&items, &positions), tail);
    // A ring prompt writes slot 0 at the position modulo the ring.
    let (k, _) = Cache {
        tokens: RING,
        paged: false,
        k_row: 4,
        v_row: 2,
        element: Element::Bf16,
    }
    .maps(Layout::Prompt, &[0], &[130], &[0])
    .unwrap();
    assert_eq!(k.ranges(), std::slice::from_ref(&(8..12)));
    // SWA prefill RoPE rows store no cache.
    let prefill = Mimo::Rope {
        q_heads: 4,
        kv_heads: 1,
        theta: 1e4,
        cache: Cache {
            tokens: RING,
            paged: false,
            k_row: 4,
            v_row: 2,
            element: Element::Bf16,
        },
        writes: false,
    };
    assert!(prefill.cache_writers(&[0, 0], &[3, 4]).is_empty());

    // A paged cache of four-token pages, three items of one row each at
    // positions 5, 6 and 9, two BF16 elements per K row and one per V row;
    // only rows 0 and 2 store (a selective map): item 0's page 0 token 1
    // and item 2's page 2 token 1.
    let cache = Cache {
        tokens: 4,
        paged: true,
        k_row: 4,
        v_row: 2,
        element: Element::Bf16,
    };
    let (items, positions) = ([0, 1, 2], [5, 6, 9]);
    let maps = cache
        .maps(Layout::Items(3), &items, &positions, &[0, 2])
        .unwrap();
    assert_eq!(maps.0.ranges(), [4..8, 36..40]);
    assert_eq!(maps.1.ranges(), [2..4, 18..20]);
    // A prompt's page is the position's block table entry.
    let prompt = cache.maps(Layout::Prompt, &[0], &[70], &[0]).unwrap();
    assert_eq!(prompt.0.ranges(), std::slice::from_ref(&(280..284)));
    let mut before = vec![0x11_u8; 3 * 4 * 4];
    for range in maps.0.ranges() {
        before[range.clone()].fill(POISON);
    }
    let mut after = before.clone();
    for range in maps.0.ranges() {
        after[range.clone()].copy_from_slice(&one.repeat(2));
    }
    let pool = |after: &[u8]| written_cache("K pool", &before, after, &maps.0, Element::Bf16);
    assert!(pool(&after).is_ok());
    // Unwritten: the poisoned rows left as they were.
    assert!(pool(&before).is_err());
    // A hole in a stored row.
    let mut hole = after.clone();
    hole[38..40].fill(POISON);
    assert!(pool(&hole).is_err());
    // A stored element written NaN.
    let mut not_finite = after.clone();
    not_finite[4..6].copy_from_slice(&nan);
    assert!(pool(&not_finite).is_err());
    // A write to row 1's token (item 1, page 1 token 2), which this launch
    // does not store, or to another byte of the pool.
    let mut stray = after.clone();
    stray[24..28].copy_from_slice(&one.repeat(2));
    assert!(pool(&stray).is_err());
    let mut changed = after.clone();
    changed[0] = 0x22;
    assert!(pool(&changed).is_err());
    // A launch that stores nothing must leave the pool unchanged.
    let none = WriteMap::of(DType::Bf16).none().unwrap();
    assert!(written_cache("ring", &before, &before, &none, Element::Bf16).is_ok());
    assert!(written_cache("ring", &before, &changed, &none, Element::Bf16).is_err());
    // FP8 codes are raw bytes: a code left at the poison is unwritten.
    let fp8 = WriteMap::of(DType::U8).rows(2, [1]).unwrap();
    let fp8_before = [0x11, 0x11, POISON, POISON];
    assert!(
        written_cache(
            "fp8",
            &fp8_before,
            &[0x11, 0x11, 0x38, 0xb8],
            &fp8,
            Element::Raw(1)
        )
        .is_ok()
    );
    assert!(
        written_cache(
            "fp8",
            &fp8_before,
            &[0x11, 0x11, 0x38, POISON],
            &fp8,
            Element::Raw(1)
        )
        .is_err()
    );
}
