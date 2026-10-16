//! RoPE and KV writes through the row engine entries a model instantiates:
//! `rope_kv` and `ring_write` (`row/rope_kv.cuh`) and the RoPE alone
//! (`dflash_rope`, `row/dflash.cuh`). The backend knows each template, so
//! it serves an op with the selection's entry that instantiates the
//! template for the op, and refuses shapes a template cannot compute.
//! Entries that instantiate it and would launch differently are a tie the
//! op does not resolve (a ring op ignores the page size, so ring entries
//! of two page sizes tie, as do a ring entry's two `max_pages` fills). Each
//! prepared op launches its entry once per step at the smallest capacity
//! bucket that holds the step's rows, as the model's row builders plan it:
//! rows at or past the device live count are not written, so every bucket
//! writes equal bits.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{
    Argument, DType, LaunchPlan, Refusal, RotarySource, RotaryStyle, RowQuery, RuntimeCounts,
    Validate,
};
use kernel_core::properties::Invariance;
use kernel_core::rope_kv::{
    KvCache, KvWrite, RingWriteBuffers, RingWriteOp, RopeBuffers, RopeKvBuffers, RopeKvOp, RopeOp,
};

use super::rows::row_launch;
use super::select::{Candidate, choose};
use super::{MAX_CAPACITY, check_plan};
use crate::row::builders::{
    HEAD_DIM as ROPE_HEAD, RING_WRITE_ARGUMENTS, ROPE_ARGUMENTS, ROPE_KV_ARGUMENTS,
    ROPE_KV_HEAD_GROUP, ROPE_KV_THREADS, ROPE_ROTARY, ROPE_THETA,
};
use crate::row::{Grid, RowInstantiation, smallest_bucket};

/// A RoPE and KV write entry of a model's selection, with the value SM120
/// passes a ring `rope_kv` entry's `max_pages` parameter. The ring branch
/// of the template never reads that parameter; the model supplies the
/// value its launches pass today (its block table width), so a ring
/// launch's arguments stay what they are. Other entries ignore it.
#[derive(Clone, Debug)]
pub struct RopeKvEntry {
    pub entry: RowInstantiation,
    pub ring_max_pages: u32,
}

impl RopeKvEntry {
    /// An entry whose unread ring `max_pages` argument is zero.
    pub fn new(entry: RowInstantiation) -> Self {
        Self {
            entry,
            ring_max_pages: 0,
        }
    }

    pub fn with_ring_max_pages(self, ring_max_pages: u32) -> Self {
        Self {
            ring_max_pages,
            ..self
        }
    }
}

/// The RoPE alone of `op` through the selection's `dflash_rope` entry for it.
pub(super) fn rope(
    entries: &[RopeKvEntry],
    op: &RopeOp,
    need: Need,
) -> Result<Prepared<RopeBuffers<View>>, Refused> {
    op.validate()?;
    let RowQuery::Rope {
        head_dim,
        rotary_dim,
        theta,
        style,
        source,
        ..
    } = op.rope
    else {
        unreachable!("a validated RoPE op holds a RoPE query");
    };
    let table = source == RotarySource::CosSinTable;
    if head_dim != ROPE_HEAD
        || rotary_dim != ROPE_ROTARY
        || style != RotaryStyle::Neox
        || !table && theta != ROPE_THETA
    {
        return Err(Refused::Rule(
            "the SM120 RoPE alone rotates 64 of 128 dimensions NeoX, computing at theta 1e4".into(),
        ));
    }
    let heads = op.heads;
    let pos64 = op.positions == DType::I64;
    let body = if table {
        format!(
            "chop::row::dflash_rope<{heads},{pos64},true>{}",
            ROPE_ARGUMENTS[1]
        )
    } else {
        format!(
            "chop::row::dflash_rope<{heads},{pos64}>{}",
            ROPE_ARGUMENTS[0]
        )
    };
    let entry = choose(
        "RoPE",
        entries
            .iter()
            .map(|selected| &selected.entry)
            .filter(|entry| {
                entry.query == op.rope
                    && entry.body == body
                    && entry.grid == Grid::RowsHeads(heads)
                    && entry.threads == head_dim
            })
            .map(candidate),
        need,
    )?;
    let width = u64::from(heads) * u64::from(head_dim);
    Ok(Box::new(Rope {
        row: RowOp::new(entry.entry, entry.guarantees, need)?,
        width: u32::try_from(width).map_err(|_| Refusal("RoPE row width overflows u32"))?,
        pairs: rotary_dim / 2,
        table,
        position_bytes: position_bytes(op.positions),
    }))
}

/// The RoPE KV write of `op` through the selection's `rope_kv` entry for it.
pub(super) fn rope_kv_write(
    entries: &[RopeKvEntry],
    op: &RopeKvOp,
    need: Need,
) -> Result<Prepared<RopeKvBuffers<View>>, Refused> {
    op.validate()?;
    let RowQuery::Rope {
        head_dim,
        rotary_dim,
        style,
        source,
        ..
    } = op.rope
    else {
        unreachable!("a validated RoPE KV write holds a RoPE query");
    };
    let (q_heads, kv_heads, v_dim) = (op.q_heads, op.kv_heads, op.v_dim);
    if !q_heads.is_multiple_of(ROPE_KV_HEAD_GROUP) || kv_heads > q_heads {
        return Err(Refused::Rule(
            "SM120 RoPE KV writes rotate Q heads in groups of four and at most as many KV heads"
                .into(),
        ));
    }
    let (ring, store) = match op.cache {
        KvWrite::Paged { .. } => (false, true),
        KvWrite::Ring { store, .. } => (true, store),
    };
    let prefix = format!(
        "chop::row::rope_kv<{q_heads},{kv_heads},{head_dim},{v_dim},{rotary_dim},{},{},{ring},{},{},{},",
        style == RotaryStyle::Neox,
        source == RotarySource::CosSinTable,
        !store,
        op.cache.kv_dtype() == DType::Fp8E4M3,
        op.positions == DType::I64,
    );
    let served = entries.iter().filter(|RopeKvEntry { entry, .. }| {
        // The page size is the template's last parameter; a ring
        // ignores it.
        let page = entry
            .body
            .strip_prefix(&prefix)
            .and_then(|rest| rest.strip_suffix(ROPE_KV_ARGUMENTS))
            .and_then(|rest| rest.strip_suffix('>'));
        let page_served = page.is_some_and(|page| match op.cache {
            KvWrite::Paged { page_size, .. } => page == page_size.to_string(),
            KvWrite::Ring { .. } => page.parse::<u32>().is_ok(),
        });
        entry.query == op.rope
            && page_served
            && entry.grid == Grid::RowsHeads(q_heads / ROPE_KV_HEAD_GROUP)
            && entry.threads == ROPE_KV_THREADS
    });
    // A ring launch passes the entry's `max_pages` fill; a paged one the
    // block table width.
    let selected = choose(
        "RoPE KV write",
        served.map(|selected| {
            let entry = &selected.entry;
            let launches = rope_kv_launch(selected, ring);
            Candidate::new(selected, &entry.symbol, launches, entry.invariance)
        }),
        need,
    )?;
    Ok(Box::new(RopeKv {
        row: RowOp::new(&selected.entry.entry, selected.guarantees, need)?,
        op: *op,
        ring_max_pages: selected.entry.ring_max_pages,
    }))
}

/// The ring write of `op` through the selection's `ring_write` entry for it.
pub(super) fn ring_write(
    entries: &[RopeKvEntry],
    op: &RingWriteOp,
    need: Need,
) -> Result<Prepared<RingWriteBuffers<View>>, Refused> {
    op.validate()?;
    let RowQuery::RingWrite {
        qk_dim,
        v_dim,
        kv_heads,
        ring_size,
        kv_dtype,
    } = op.query
    else {
        unreachable!("a validated ring write holds a ring write query");
    };
    let (fp8, pos64) = (kv_dtype == DType::Fp8E4M3, op.positions == DType::I64);
    let (body, grid) = if op.single_sequence {
        (
            format!(
                "chop::row::ring_write<{kv_heads},{qk_dim},{v_dim},{fp8},{pos64},true>{RING_WRITE_ARGUMENTS}"
            ),
            Grid::RingTail {
                ring: ring_size,
                heads: kv_heads,
            },
        )
    } else {
        (
            format!(
                "chop::row::ring_write<{kv_heads},{qk_dim},{v_dim},{fp8},{pos64}>{RING_WRITE_ARGUMENTS}"
            ),
            Grid::RowsHeads(kv_heads),
        )
    };
    let entry = choose(
        "ring write",
        entries
            .iter()
            .map(|selected| &selected.entry)
            .filter(|entry| entry.query == op.query && entry.body == body && entry.grid == grid)
            .map(candidate),
        need,
    )?;
    Ok(Box::new(RingWrite {
        row: RowOp::new(entry.entry, entry.guarantees, need)?,
        op: *op,
    }))
}

/// What a `rope_kv` entry launches: its row launch, and on a ring its
/// `max_pages` fill.
fn rope_kv_launch(selected: &RopeKvEntry, ring: bool) -> String {
    let RopeKvEntry {
        entry,          // its row launch
        ring_max_pages, // a ring launch's argument; a paged launch passes the table width
    } = selected;
    if !ring {
        return row_launch(entry);
    }
    // A ring ignores the page size, the template's last parameter.
    let body = entry
        .body
        .strip_suffix(ROPE_KV_ARGUMENTS)
        .and_then(|rest| rest.strip_suffix('>'))
        .and_then(|rest| rest.rsplit_once(','))
        .map_or_else(
            || entry.body.clone(),
            |(head, _page)| format!("{head},_>{ROPE_KV_ARGUMENTS}"),
        );
    let entry = RowInstantiation {
        body,
        ..entry.clone()
    };
    format!("{} max_pages {ring_max_pages}", row_launch(&entry))
}

/// An entry that matches an op. Only a ring `rope_kv` launch reads the
/// `max_pages` fill, so entries of other ops differing in it are one.
fn candidate(entry: &RowInstantiation) -> Candidate<&RowInstantiation> {
    Candidate::new(entry, &entry.symbol, row_launch(entry), entry.invariance)
}

fn position_bytes(positions: DType) -> u64 {
    if positions == DType::I64 { 8 } else { 4 }
}

fn kv_bytes(dtype: DType) -> u64 {
    if dtype == DType::Fp8E4M3 { 1 } else { 2 }
}

fn buffer(view: View) -> Argument {
    Argument::DeviceBuffer(view.handle)
}

fn optional(view: Option<View>) -> Argument {
    view.map_or(Argument::NullDevicePointer, buffer)
}

/// `a * b * c` bytes, refused rather than wrapped.
fn bytes(a: u64, b: u64, c: u64) -> Result<u64, Refusal> {
    a.checked_mul(b)
        .and_then(|n| n.checked_mul(c))
        .ok_or(Refusal("RoPE or KV write bytes overflow"))
}

/// One row engine entry with its plan at each capacity bucket up to the
/// prepared capacity, and each row count's bucket, chosen when the op is
/// prepared.
struct RowOp {
    entry: RowInstantiation,
    /// The plans of the buckets, smallest first.
    plans: Vec<LaunchPlan>,
    /// The index in `plans` of each row count from one to the capacity.
    bucket_of: Vec<u8>,
    info: OpInfo,
}

impl RowOp {
    /// `entry`, claiming `guarantees` (what its registrations declare).
    fn new(entry: &RowInstantiation, guarantees: Invariance, need: Need) -> Result<Self, Refused> {
        let capacity = need.capacity;
        if capacity == 0 || capacity > MAX_CAPACITY {
            return Err(Refused::Rule(format!(
                "an SM120 RoPE or KV write serves 1 to {MAX_CAPACITY} rows"
            )));
        }
        if smallest_bucket(capacity).is_none() {
            return Err(Refused::Rule(
                "the row capacity exceeds every SM120 row bucket".into(),
            ));
        }
        let mut plans = Vec::new();
        let mut planned = None;
        let mut bucket_of = Vec::with_capacity(capacity as usize);
        for rows in 1..=capacity {
            let bucket = smallest_bucket(rows).ok_or(Refusal("no row bucket holds the rows"))?;
            if planned != Some(bucket) {
                let plan = entry.plan(bucket, counts(bucket))?;
                check_plan(&plan)?;
                plans.push(plan);
                planned = Some(bucket);
            }
            bucket_of.push(u8::try_from(plans.len() - 1).expect("few row buckets"));
        }
        let digest = kernel_build::digest(format!("{entry:?}").as_bytes());
        let info = OpInfo::new(
            format!("sm120.row.v1[{}]#{}", entry.symbol, &digest[..16]),
            guarantees,
            vec![(entry.symbol.clone(), 0)],
        );
        need.check(&info)?;
        Ok(Self {
            entry: entry.clone(),
            plans,
            bucket_of,
            info,
        })
    }

    /// The rows of `counts` the step's buffers hold, refused outside the
    /// prepared capacity.
    fn rows(&self, counts: Counts) -> Result<u64, Refusal> {
        self.grid_rows(counts)?;
        Ok(u64::from(counts.rows))
    }

    /// The rows a launch grids and records: the step's rows, or the bucket
    /// it pads them to (`Counts::padded_rows`). Rows past the device live
    /// count are not written, so every grid that holds them writes equal
    /// bits.
    fn grid_rows(&self, counts: Counts) -> Result<u32, Refusal> {
        let grid = counts.rows.max(counts.padded_rows);
        if counts.rows == 0 || grid as usize > self.bucket_of.len() {
            return Err(Refusal("row count outside the prepared capacity"));
        }
        Ok(grid)
    }

    fn launch(&self, counts: Counts, arguments: Vec<Argument>) -> Result<Vec<Launch>, Refusal> {
        let grid = self.grid_rows(counts)?;
        let bucket = self.bucket_of[grid as usize - 1];
        Ok(vec![Launch {
            symbol: self.entry.symbol.clone(),
            rows: grid,
            plan: self.plans[usize::from(bucket)],
            arguments: self.entry.pack_arguments(arguments)?,
            collective: false,
            persistent: false,
        }])
    }
}

fn counts(rows: u32) -> RuntimeCounts {
    RuntimeCounts {
        m: rows,
        batch: 1,
        context: 1,
        routed_rows: 0,
        split_k: 1,
    }
}

/// Checks a cache's planes hold `slots` slots of `k_slot` and `v_slot`
/// bytes at their strides; a ring's slots are packed, as its entries index
/// them. An FP8 cache needs finite positive scales.
fn check_cache(
    cache: &KvCache<View>,
    [k_slot, v_slot]: [u64; 2],
    packed: bool,
    dtype: DType,
) -> Result<(), Refusal> {
    if cache.slots == 0 {
        return Err(Refusal("a KV cache holds at least one slot"));
    }
    for (plane, stride, slot) in [
        (cache.k, cache.k_slot_stride, k_slot),
        (cache.v, cache.v_slot_stride, v_slot),
    ] {
        if if packed {
            stride != slot
        } else {
            stride < slot
        } {
            return Err(Refusal(
                "a KV cache slot stride does not fit the entry's slot layout",
            ));
        }
        let end = u64::from(cache.slots - 1)
            .checked_mul(stride)
            .and_then(|start| start.checked_add(slot))
            .ok_or(Refusal("KV cache bytes overflow"))?;
        plane.covers(end)?;
    }
    if dtype == DType::Fp8E4M3
        && [cache.k_scale, cache.v_scale]
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(Refusal("an FP8 KV cache needs finite positive scales"));
    }
    Ok(())
}

/// `dflash_rope`: one block per row and head, one thread per dimension.
struct Rope {
    row: RowOp,
    /// Elements of one row.
    width: u32,
    pairs: u32,
    table: bool,
    position_bytes: u64,
}

impl PreparedOp<RopeBuffers<View>> for Rope {
    fn info(&self) -> &OpInfo {
        &self.row.info
    }

    fn launches(
        &self,
        buffers: &RopeBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.row.rows(counts)?;
        let row_bytes = bytes(rows, u64::from(self.width), 2)?;
        buffers.input.covers(row_bytes)?;
        buffers.output.covers(row_bytes)?;
        buffers.positions.covers(rows * self.position_bytes)?;
        buffers.live_rows.covers(4)?;
        let mut arguments = vec![
            buffer(buffers.input),
            buffer(buffers.output),
            buffer(buffers.positions),
            buffer(buffers.live_rows),
            Argument::U32(self.width),
            Argument::U32(self.width),
        ];
        if self.table {
            // The host bounds positions by the table's rows.
            let table = buffers
                .cos_sin
                .ok_or(Refusal("a table-backed RoPE needs its cos and sin table"))?;
            table.covers(u64::from(counts.positions.max(1)) * u64::from(self.pairs) * 8)?;
            arguments.push(buffer(table));
        }
        self.row.launch(counts, arguments)
    }
}

/// `rope_kv`: a block per row and Q head below 512 rows, a block per row
/// and four Q heads from 512 on; the first KV heads' blocks write K and V.
struct RopeKv {
    row: RowOp,
    op: RopeKvOp,
    /// The unread `max_pages` argument of a ring entry (see [`RopeKvEntry`]).
    ring_max_pages: u32,
}

impl PreparedOp<RopeKvBuffers<View>> for RopeKv {
    fn info(&self) -> &OpInfo {
        &self.row.info
    }

    fn launches(
        &self,
        buffers: &RopeKvBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.row.rows(counts)?;
        let RowQuery::Rope {
            head_dim,
            rotary_dim,
            theta,
            source,
            ..
        } = self.op.rope
        else {
            unreachable!("a prepared RoPE KV write holds a RoPE query");
        };
        let (qk, v) = (u64::from(head_dim), u64::from(self.op.v_dim));
        let (q_heads, kv_heads) = (u64::from(self.op.q_heads), u64::from(self.op.kv_heads));
        let packed = q_heads * qk + kv_heads * (qk + v);
        buffers.qkv.covers(bytes(rows, packed, 2)?)?;
        buffers.q.covers(bytes(rows, q_heads * qk, 2)?)?;
        if let Some(k_new) = buffers.k_new {
            k_new.covers(bytes(rows, kv_heads * qk, 2)?)?;
        }
        if let Some(v_new) = buffers.v_new {
            v_new.covers(bytes(rows, kv_heads * v, 2)?)?;
        }
        buffers.token_batch.covers(rows * 4)?;
        buffers
            .positions
            .covers(rows * position_bytes(self.op.positions))?;
        buffers.live_rows.covers(4)?;
        if source == RotarySource::CosSinTable {
            // The host bounds positions by the table's rows.
            buffers
                .cos_sin
                .ok_or(Refusal("a table-backed RoPE needs its cos and sin table"))?
                .covers(u64::from(counts.positions.max(1)) * u64::from(rotary_dim / 2) * 8)?;
        }
        let dtype = self.op.cache.kv_dtype();
        let cache = &buffers.cache;
        let (block_tables, slot_ids, max_pages, ring_size, strides) = match self.op.cache {
            KvWrite::Paged {
                page_size,
                max_context,
                ..
            } => {
                let pages = max_context.div_ceil(page_size);
                let page = u64::from(page_size) * kv_heads * kv_bytes(dtype);
                check_cache(cache, [page * qk, page * v], false, dtype)?;
                // The host bounds each row's item by the block table's rows.
                let tables = buffers
                    .block_tables
                    .ok_or(Refusal("a paged KV write needs block tables"))?;
                tables.covers(u64::from(pages) * 4)?;
                (
                    buffer(tables),
                    Argument::NullDevicePointer,
                    pages,
                    0,
                    [cache.k_slot_stride, cache.v_slot_stride],
                )
            }
            KvWrite::Ring {
                ring_size, store, ..
            } => {
                if store {
                    let slot = u64::from(ring_size) * kv_heads * kv_bytes(dtype);
                    check_cache(cache, [slot * qk, slot * v], true, dtype)?;
                } else if buffers.k_new.is_none() || buffers.v_new.is_none() {
                    return Err(Refusal(
                        "a ring RoPE that does not store writes k_new and v_new",
                    ));
                }
                // The host bounds each row's item by the ring slots' rows.
                let slots = buffers
                    .slot_ids
                    .ok_or(Refusal("a ring KV write needs slot ids"))?;
                slots.covers(4)?;
                // The entry reads neither the block table width nor
                // strides of a ring; the width is the selection's fill.
                (
                    Argument::NullDevicePointer,
                    buffer(slots),
                    self.ring_max_pages,
                    ring_size,
                    [0, 0],
                )
            }
        };
        self.row.launch(
            counts,
            vec![
                buffer(buffers.qkv),
                buffer(buffers.q),
                optional(buffers.k_new),
                optional(buffers.v_new),
                buffer(cache.k),
                buffer(cache.v),
                block_tables,
                slot_ids,
                buffer(buffers.token_batch),
                buffer(buffers.positions),
                optional(buffers.cos_sin),
                buffer(buffers.live_rows),
                Argument::U32(max_pages),
                Argument::U32(ring_size),
                Argument::U64(strides[0]),
                Argument::U64(strides[1]),
                Argument::U32(theta.to_bits()),
                Argument::U32(self.op.value_scale.to_bits()),
                Argument::U32(cache.k_scale.to_bits()),
                Argument::U32(cache.v_scale.to_bits()),
            ],
        )
    }
}

/// `ring_write`: a block per row and KV head; the single-sequence entry
/// launches blocks for the last `ring_size` rows only.
struct RingWrite {
    row: RowOp,
    op: RingWriteOp,
}

impl PreparedOp<RingWriteBuffers<View>> for RingWrite {
    fn info(&self) -> &OpInfo {
        &self.row.info
    }

    fn launches(
        &self,
        buffers: &RingWriteBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.row.rows(counts)?;
        let RowQuery::RingWrite {
            qk_dim,
            v_dim,
            kv_heads,
            ring_size,
            kv_dtype,
        } = self.op.query
        else {
            unreachable!("a prepared ring write holds a ring write query");
        };
        let (qk, v, heads) = (u64::from(qk_dim), u64::from(v_dim), u64::from(kv_heads));
        buffers.k_new.covers(bytes(rows, heads * qk, 2)?)?;
        buffers.v_new.covers(bytes(rows, heads * v, 2)?)?;
        let slot = u64::from(ring_size) * heads * kv_bytes(kv_dtype);
        check_cache(&buffers.cache, [slot * qk, slot * v], true, kv_dtype)?;
        // The host bounds each row's item by the per-item arrays' rows.
        buffers.slot_ids.covers(4)?;
        buffers.last_positions.covers(8)?;
        buffers.token_batch.covers(rows * 4)?;
        buffers
            .positions
            .covers(rows * position_bytes(self.op.positions))?;
        buffers.live_rows.covers(4)?;
        self.row.launch(
            counts,
            vec![
                buffer(buffers.k_new),
                buffer(buffers.v_new),
                buffer(buffers.cache.k),
                buffer(buffers.cache.v),
                buffer(buffers.slot_ids),
                buffer(buffers.token_batch),
                buffer(buffers.positions),
                buffer(buffers.last_positions),
                buffer(buffers.live_rows),
                Argument::U32(ring_size),
                Argument::U32(buffers.cache.k_scale.to_bits()),
                Argument::U32(buffers.cache.v_scale.to_bits()),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Profile;
    use crate::backend::select::orders;
    use crate::backend::{Sm120, view};
    use crate::row::builders::{ring_write_entries, rope_entries};
    use crate::row::row_local;
    use kernel_build::EntryParam;
    use kernel_core::backend::Backend;
    use kernel_core::properties::Invariance;

    fn ptr(name: &str, ty: &str) -> EntryParam {
        EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: ty.into(),
        }
    }

    fn u32_arg(name: &str) -> EntryParam {
        EntryParam::U32 { name: name.into() }
    }

    fn u64_arg(name: &str) -> EntryParam {
        EntryParam::U64 { name: name.into() }
    }

    /// The shapes of one rank's rows: 16 Q heads, 192-wide Q and K heads,
    /// 128-wide V heads and NeoX RoPE over 64 dimensions; one KV head on
    /// paged layers, two on ring layers of 128 tokens.
    const QH: u32 = 16;
    const QK: u32 = 192;
    const V: u32 = 128;
    const ROTARY: u32 = 64;
    const RING: u32 = 128;
    const THETA: f32 = 5e6;
    const VALUE_SCALE: f32 = 0.707;

    fn rope_query(paged: bool) -> RowQuery {
        RowQuery::Rope {
            head_dim: QK,
            rotary_dim: ROTARY,
            theta: THETA,
            style: RotaryStyle::Neox,
            source: RotarySource::CosSinTable,
            fused_paged_kv_write: paged,
        }
    }

    /// A `rope_kv` entry as a model's row catalog instantiates it.
    fn rope_kv_entry(
        kvh: u32,
        ring: bool,
        prefill: bool,
        fp8: bool,
        page: u32,
    ) -> RowInstantiation {
        let mut entry = RowInstantiation::new(
            Profile::StdDenseMoe,
            &format!("rope_test_{kvh}_{ring}_{prefill}_{fp8}_p{page}_pos32"),
            rope_query(!ring),
            QH * QK + kvh * (QK + V),
            (Grid::RowsHeads(QH / 4), 256),
            vec![
                ptr("qkv", "const uint16_t"),
                ptr("q", "uint16_t"),
                ptr("k_new", "uint16_t"),
                ptr("v_new", "uint16_t"),
                ptr("k_pool", "void"),
                ptr("v_pool", "void"),
                ptr("block_tables", "const int32_t"),
                ptr("slot_ids", "const int32_t"),
                ptr("token_batch", "const int32_t"),
                ptr("positions", "const void"),
                ptr("cos_sin", "const float"),
                ptr("live_tokens", "const uint32_t"),
                u32_arg("max_pages"),
                u32_arg("ring_size"),
                u64_arg("k_page_stride_bytes"),
                u64_arg("v_page_stride_bytes"),
                u32_arg("theta_bits"),
                u32_arg("v_multiplier_bits"),
                u32_arg("k_scale_bits"),
                u32_arg("v_scale_bits"),
            ],
            format!(
                "chop::row::rope_kv<{QH},{kvh},{QK},{V},{ROTARY},true,true,{ring},{prefill},{fp8},false,{page}>(qkv,q,k_new,v_new,k_pool,v_pool,block_tables,slot_ids,token_batch,positions,cos_sin,live_tokens,max_pages,ring_size,k_page_stride_bytes,v_page_stride_bytes,__uint_as_float(theta_bits),__uint_as_float(v_multiplier_bits),__uint_as_float(k_scale_bits),__uint_as_float(v_scale_bits));"
            ),
        );
        entry.invariance = row_local();
        entry
    }

    /// A ring write as the shared builder makes it: BF16 or FP8, or the
    /// BF16 single-sequence tail.
    fn ring_write_entry(fp8: bool, tail: bool) -> RowInstantiation {
        let entries = ring_write_entries(Profile::StdDenseMoe, "test", 2, QK, V, RING).unwrap();
        let mut entry = match (fp8, tail) {
            (false, false) => entries[0].clone(),
            (true, false) => entries[2].clone(),
            (false, true) => entries[4].clone(),
            (true, true) => unreachable!("the single-sequence tail writes BF16"),
        };
        entry.invariance = row_local().with_anchors(kernel_core::properties::Anchors::None);
        entry
    }

    /// The RoPE alone as the shared builder makes it, with 32-bit positions.
    fn rope_entry(heads: u32, table: bool) -> RowInstantiation {
        // Named as the drafter's entries are: a `rope_` prefix would take
        // the planner's grouped RoPE KV grid.
        rope_entries(
            Profile::StdDenseMoe,
            &format!("dflash_test{heads}_q"),
            heads,
            128,
            64,
            1e4,
        )
        .unwrap()[usize::from(table)]
        .clone()
    }

    /// The block table width the model passes ring entries today: its
    /// context over its page size.
    const RING_MAX_PAGES: u32 = 262_144 / 64;

    fn backend() -> Sm120 {
        let mut entries = vec![
            rope_kv_entry(1, false, false, false, 16),
            rope_kv_entry(1, false, false, false, 64),
            rope_kv_entry(1, false, false, true, 64),
            rope_kv_entry(2, true, false, false, 32),
            rope_kv_entry(2, true, true, false, 32),
            ring_write_entry(false, false),
            ring_write_entry(true, false),
            ring_write_entry(false, true),
        ];
        for heads in [4, 8] {
            entries.extend([rope_entry(heads, false), rope_entry(heads, true)]);
        }
        Sm120::new().with_rope_kv(
            entries
                .into_iter()
                .map(|entry| RopeKvEntry::new(entry).with_ring_max_pages(RING_MAX_PAGES))
                .collect(),
        )
    }

    fn paged_op(fp8: bool) -> RopeKvOp {
        RopeKvOp {
            rope: rope_query(true),
            q_heads: QH,
            kv_heads: 1,
            v_dim: V,
            value_scale: VALUE_SCALE,
            positions: DType::I32,
            cache: KvWrite::Paged {
                page_size: 64,
                max_context: 65_536,
                kv_dtype: if fp8 { DType::Fp8E4M3 } else { DType::Bf16 },
            },
        }
    }

    fn ring_op(store: bool) -> RopeKvOp {
        RopeKvOp {
            rope: rope_query(false),
            kv_heads: 2,
            cache: KvWrite::Ring {
                ring_size: RING,
                kv_dtype: DType::Bf16,
                store,
            },
            ..paged_op(false)
        }
    }

    const ROWS: u64 = 8192;
    const PAGES: u32 = 100;
    /// Pages interleave two layers' K and V planes.
    const K_PAGE_STRIDE: u64 = 2 * 64 * (QK as u64 + V as u64) * 2;
    const V_PAGE_STRIDE: u64 = K_PAGE_STRIDE;

    fn cache(paged: bool, kvh: u64, element: u64, scales: [f32; 2]) -> KvCache<View> {
        let (slots, k_stride, v_stride) = if paged {
            (PAGES, K_PAGE_STRIDE, V_PAGE_STRIDE)
        } else {
            let slot = u64::from(RING) * kvh * element;
            (4, slot * u64::from(QK), slot * u64::from(V))
        };
        KvCache {
            k: view(5, u64::from(slots) * k_stride),
            v: view(6, u64::from(slots) * v_stride),
            slots,
            k_slot_stride: k_stride,
            v_slot_stride: v_stride,
            k_scale: scales[0],
            v_scale: scales[1],
        }
    }

    fn rope_kv_buffers(
        paged: bool,
        kvh: u64,
        element: u64,
        scales: [f32; 2],
    ) -> RopeKvBuffers<View> {
        let (qh, qk, v) = (u64::from(QH), u64::from(QK), u64::from(V));
        RopeKvBuffers {
            qkv: view(1, ROWS * (qh * qk + kvh * (qk + v)) * 2),
            q: view(2, ROWS * qh * qk * 2),
            k_new: Some(view(3, ROWS * kvh * qk * 2)),
            v_new: Some(view(4, ROWS * kvh * v * 2)),
            cache: cache(paged, kvh, element, scales),
            block_tables: paged.then(|| view(7, 32 * 1024 * 4)),
            slot_ids: (!paged).then(|| view(8, 32 * 4)),
            token_batch: view(9, ROWS * 4),
            positions: view(10, ROWS * 4),
            cos_sin: Some(view(11, 65_536 * 32 * 8)),
            live_rows: view(12, 4),
        }
    }

    /// What the model's builder launches for `rows` rows: the entry
    /// planned at the smallest bucket that holds them, with `arguments`.
    fn legacy(entry: &RowInstantiation, rows: u32, arguments: Vec<Argument>) -> Launch {
        let bucket = smallest_bucket(rows).unwrap();
        Launch {
            symbol: entry.symbol.clone(),
            rows,
            plan: entry.plan(bucket, counts(rows)).unwrap(),
            arguments: entry.pack_arguments(arguments).unwrap(),
            collective: false,
            persistent: false,
        }
    }

    const ROW_COUNTS: [u32; 7] = [1, 3, 32, 33, 511, 512, 8192];

    #[test]
    fn paged_rope_kv_writes_launch_what_the_builder_plans() {
        let backend = backend();
        for (fp8, scales) in [(false, [1.0, 1.0]), (true, [1.25, 0.5])] {
            let op = paged_op(fp8);
            let prepared = backend.rope_kv_write(&op, Need::rows(8192)).unwrap();
            let entry = rope_kv_entry(1, false, false, fp8, 64);
            assert_eq!(prepared.info().guarantees, row_local());
            assert!(
                prepared
                    .info()
                    .implementation
                    .starts_with(&format!("sm120.row.v1[{}]#", entry.symbol))
            );
            assert_eq!(prepared.info().symbols, [(entry.symbol.clone(), 0)]);
            let buffers = rope_kv_buffers(true, 1, if fp8 { 1 } else { 2 }, scales);
            for rows in ROW_COUNTS {
                let launches = prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                let arguments = vec![
                    Argument::DeviceBuffer(1),
                    Argument::DeviceBuffer(2),
                    Argument::DeviceBuffer(3),
                    Argument::DeviceBuffer(4),
                    Argument::DeviceBuffer(5),
                    Argument::DeviceBuffer(6),
                    Argument::DeviceBuffer(7),
                    Argument::NullDevicePointer,
                    Argument::DeviceBuffer(9),
                    Argument::DeviceBuffer(10),
                    Argument::DeviceBuffer(11),
                    Argument::DeviceBuffer(12),
                    Argument::U32(65_536 / 64),
                    Argument::U32(0),
                    Argument::U64(K_PAGE_STRIDE),
                    Argument::U64(V_PAGE_STRIDE),
                    Argument::U32(THETA.to_bits()),
                    Argument::U32(VALUE_SCALE.to_bits()),
                    Argument::U32(scales[0].to_bits()),
                    Argument::U32(scales[1].to_bits()),
                ];
                assert_eq!(launches, [legacy(&entry, rows, arguments)], "{rows} rows");
            }
        }
        // The page size picks the entry.
        let small = RopeKvOp {
            cache: KvWrite::Paged {
                page_size: 16,
                max_context: 65_536,
                kv_dtype: DType::Bf16,
            },
            ..paged_op(false)
        };
        let prepared = backend.rope_kv_write(&small, Need::rows(4)).unwrap();
        assert_eq!(
            prepared.info().symbols[0].0,
            rope_kv_entry(1, false, false, false, 16).symbol
        );
    }

    #[test]
    fn ring_rope_and_ring_writes_launch_what_the_builder_plans() {
        let backend = backend();
        for store in [true, false] {
            let prepared = backend
                .rope_kv_write(&ring_op(store), Need::rows(8192))
                .unwrap();
            let entry = rope_kv_entry(2, true, !store, false, 32);
            let buffers = rope_kv_buffers(false, 2, 2, [1.0, 1.0]);
            for rows in ROW_COUNTS {
                let launches = prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                let arguments = vec![
                    Argument::DeviceBuffer(1),
                    Argument::DeviceBuffer(2),
                    Argument::DeviceBuffer(3),
                    Argument::DeviceBuffer(4),
                    Argument::DeviceBuffer(5),
                    Argument::DeviceBuffer(6),
                    Argument::NullDevicePointer,
                    Argument::DeviceBuffer(8),
                    Argument::DeviceBuffer(9),
                    Argument::DeviceBuffer(10),
                    Argument::DeviceBuffer(11),
                    Argument::DeviceBuffer(12),
                    // Unread by a ring entry; the model's value today.
                    Argument::U32(RING_MAX_PAGES),
                    Argument::U32(RING),
                    Argument::U64(0),
                    Argument::U64(0),
                    Argument::U32(THETA.to_bits()),
                    Argument::U32(VALUE_SCALE.to_bits()),
                    Argument::U32(1.0_f32.to_bits()),
                    Argument::U32(1.0_f32.to_bits()),
                ];
                assert_eq!(launches, [legacy(&entry, rows, arguments)], "{rows} rows");
            }
        }
        for (fp8, tail) in [(false, false), (true, false), (false, true)] {
            let entry = ring_write_entry(fp8, tail);
            let op = RingWriteOp {
                query: entry.query,
                positions: DType::I32,
                single_sequence: tail,
            };
            let prepared = backend.ring_write(&op, Need::rows(8192)).unwrap();
            assert_eq!(prepared.info().guarantees, entry.invariance);
            let scales = if fp8 { [2.0_f32, 4.0] } else { [1.0, 1.0] };
            let (qk, v) = (u64::from(QK), u64::from(V));
            let buffers = RingWriteBuffers {
                k_new: view(1, ROWS * 2 * qk * 2),
                v_new: view(2, ROWS * 2 * v * 2),
                cache: cache(false, 2, if fp8 { 1 } else { 2 }, scales),
                slot_ids: view(7, 4 * 4),
                token_batch: view(8, ROWS * 4),
                positions: view(9, ROWS * 4),
                last_positions: view(10, 4 * 8),
                live_rows: view(11, 4),
            };
            for rows in ROW_COUNTS {
                let launches = prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                let arguments = vec![
                    Argument::DeviceBuffer(1),
                    Argument::DeviceBuffer(2),
                    Argument::DeviceBuffer(5),
                    Argument::DeviceBuffer(6),
                    Argument::DeviceBuffer(7),
                    Argument::DeviceBuffer(8),
                    Argument::DeviceBuffer(9),
                    Argument::DeviceBuffer(10),
                    Argument::DeviceBuffer(11),
                    Argument::U32(RING),
                    Argument::U32(scales[0].to_bits()),
                    Argument::U32(scales[1].to_bits()),
                ];
                assert_eq!(launches, [legacy(&entry, rows, arguments)], "{rows} rows");
            }
        }
    }

    #[test]
    fn rope_alone_launches_what_the_builder_plans() {
        let backend = backend();
        for (heads, table) in [(4, false), (8, true)] {
            let entry = rope_entry(heads, table);
            let op = RopeOp {
                rope: entry.query,
                heads,
                positions: DType::I32,
            };
            let prepared = backend.rope(&op, Need::rows(32)).unwrap();
            let width = u64::from(heads) * 128;
            let buffers = RopeBuffers {
                input: view(1, 32 * width * 2),
                output: view(2, 32 * width * 2),
                positions: view(3, 32 * 4),
                cos_sin: Some(view(4, 4096 * 32 * 8)),
                live_rows: view(5, 4),
            };
            for rows in [1, 5, 32] {
                let launches = prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                let mut arguments = vec![
                    Argument::DeviceBuffer(1),
                    Argument::DeviceBuffer(2),
                    Argument::DeviceBuffer(3),
                    Argument::DeviceBuffer(5),
                    Argument::U32(width as u32),
                    Argument::U32(width as u32),
                ];
                if table {
                    arguments.push(Argument::DeviceBuffer(4));
                }
                assert_eq!(launches, [legacy(&entry, rows, arguments)], "{rows} rows");
                // A block per row of the bucket and head.
                assert_eq!(
                    launches[0].plan.grid,
                    [smallest_bucket(rows).unwrap(), heads, 1]
                );
            }
        }
    }

    /// A step that grids a padded bucket while its buffers hold fewer rows
    /// passes those rows as `Counts::rows` and the bucket as
    /// `Counts::padded_rows`: buffers are checked at the rows, and each
    /// launch grids and reports the bucket, as one of the bucket's rows.
    #[test]
    fn padded_rows_grid_the_bucket_over_the_rows_the_buffers_hold() {
        let backend = backend();
        let (qh, qk, v) = (u64::from(QH), u64::from(QK), u64::from(V));
        let (rows, bucket) = (600_u32, 1024_u32);
        let held = u64::from(rows);
        let padded = Counts {
            padded_rows: bucket,
            ..Counts::rows(rows)
        };
        // The paged RoPE KV write.
        let prepared = backend
            .rope_kv_write(&paged_op(false), Need::rows(8192))
            .unwrap();
        let full = rope_kv_buffers(true, 1, 2, [1.0, 1.0]);
        let short = RopeKvBuffers {
            qkv: view(1, held * (qh * qk + qk + v) * 2),
            q: view(2, held * qh * qk * 2),
            k_new: Some(view(3, held * qk * 2)),
            v_new: Some(view(4, held * v * 2)),
            token_batch: view(9, held * 4),
            positions: view(10, held * 4),
            ..full
        };
        assert_eq!(
            prepared.launches(&short, View::EMPTY, padded).unwrap(),
            prepared
                .launches(&full, View::EMPTY, Counts::rows(bucket))
                .unwrap()
        );
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(bucket))
                .is_err()
        );
        // The ring write.
        let entry = ring_write_entry(false, false);
        let op = RingWriteOp {
            query: entry.query,
            positions: DType::I32,
            single_sequence: false,
        };
        let prepared = backend.ring_write(&op, Need::rows(8192)).unwrap();
        let ring = |rows: u64| RingWriteBuffers {
            k_new: view(1, rows * 2 * qk * 2),
            v_new: view(2, rows * 2 * v * 2),
            cache: cache(false, 2, 2, [1.0, 1.0]),
            slot_ids: view(7, 4 * 4),
            token_batch: view(8, rows * 4),
            positions: view(9, rows * 4),
            last_positions: view(10, 4 * 8),
            live_rows: view(11, 4),
        };
        assert_eq!(
            prepared.launches(&ring(held), View::EMPTY, padded).unwrap(),
            prepared
                .launches(&ring(u64::from(bucket)), View::EMPTY, Counts::rows(bucket))
                .unwrap()
        );
        assert!(
            prepared
                .launches(&ring(held), View::EMPTY, Counts::rows(bucket))
                .is_err()
        );
        // The RoPE alone.
        let entry = rope_entry(8, true);
        let op = RopeOp {
            rope: entry.query,
            heads: 8,
            positions: DType::I32,
        };
        let prepared = backend.rope(&op, Need::rows(8192)).unwrap();
        let rope = |rows: u64| RopeBuffers {
            input: view(1, rows * 8 * 128 * 2),
            output: view(2, rows * 8 * 128 * 2),
            positions: view(3, rows * 4),
            cos_sin: Some(view(4, 4096 * 32 * 8)),
            live_rows: view(5, 4),
        };
        let launches = prepared.launches(&rope(held), View::EMPTY, padded).unwrap();
        assert_eq!(
            launches,
            prepared
                .launches(&rope(u64::from(bucket)), View::EMPTY, Counts::rows(bucket))
                .unwrap()
        );
        assert_eq!(launches[0].rows, bucket);
        assert_eq!(launches[0].plan.grid, [bucket, 8, 1]);
        // A bucket past the prepared capacity is refused.
        let past = Counts {
            padded_rows: 8193,
            ..Counts::rows(rows)
        };
        assert!(prepared.launches(&rope(held), View::EMPTY, past).is_err());
    }

    #[test]
    fn prepare_refuses_unserved_over_limit_and_inexact_ops() {
        let backend = backend();
        let rule = |result: Result<_, Refused>, text: &str| match result {
            Err(Refused::Rule(rule)) => assert!(rule.contains(text), "{rule}"),
            Err(other) => panic!("{other}"),
            Ok(_) => panic!("prepared"),
        };
        // An op no entry serves: an FP8 ring, an I64 position, an unknown
        // page size.
        let fp8_ring = RopeKvOp {
            cache: KvWrite::Ring {
                ring_size: RING,
                kv_dtype: DType::Fp8E4M3,
                store: true,
            },
            ..ring_op(true)
        };
        rule(
            backend.rope_kv_write(&fp8_ring, Need::rows(4)).map(drop),
            "no SM120",
        );
        rule(
            backend
                .rope_kv_write(
                    &RopeKvOp {
                        positions: DType::I64,
                        ..paged_op(false)
                    },
                    Need::rows(4),
                )
                .map(drop),
            "no SM120",
        );
        rule(
            backend
                .rope_kv_write(
                    &RopeKvOp {
                        cache: KvWrite::Paged {
                            page_size: 128,
                            max_context: 65_536,
                            kv_dtype: DType::Bf16,
                        },
                        ..paged_op(false)
                    },
                    Need::rows(4),
                )
                .map(drop),
            "no SM120",
        );
        // Shapes SM120's templates cannot compute.
        rule(
            backend
                .rope_kv_write(
                    &RopeKvOp {
                        q_heads: 6,
                        ..paged_op(false)
                    },
                    Need::rows(4),
                )
                .map(drop),
            "groups of four",
        );
        let alone = RopeOp {
            rope: rope_entry(4, false).query,
            heads: 4,
            positions: DType::I32,
        };
        let RowQuery::Rope {
            head_dim,
            rotary_dim,
            style,
            source,
            fused_paged_kv_write,
            ..
        } = alone.rope
        else {
            unreachable!()
        };
        let other_theta = RopeOp {
            rope: RowQuery::Rope {
                head_dim,
                rotary_dim,
                theta: 5e6,
                style,
                source,
                fused_paged_kv_write,
            },
            ..alone
        };
        rule(
            backend.rope(&other_theta, Need::rows(4)).map(drop),
            "theta 1e4",
        );
        let gptj = RopeOp {
            rope: RowQuery::Rope {
                head_dim,
                rotary_dim,
                theta: 1e4,
                style: RotaryStyle::GptJ,
                source,
                fused_paged_kv_write,
            },
            ..alone
        };
        rule(backend.rope(&gptj, Need::rows(4)).map(drop), "NeoX");
        // Capacities past the buckets and MAX_CAPACITY, refused before any
        // table is built.
        rule(backend.rope(&alone, Need::rows(0)).map(drop), "serves 1 to");
        rule(
            backend.rope(&alone, Need::rows(8193)).map(drop),
            "every SM120 row bucket",
        );
        rule(
            backend.rope(&alone, Need::rows(u32::MAX)).map(drop),
            "serves 1 to",
        );
        assert!(backend.rope(&alone, Need::rows(8192)).is_ok());
        // Exactness past what the entry declares.
        assert!(matches!(
            backend.rope(&alone, Need::rows(4).with_exact(Invariance::one_row(4))),
            Err(Refused::Inexact { .. })
        ));
        assert!(
            backend
                .rope_kv_write(
                    &paged_op(false),
                    Need::rows(32).with_exact(Invariance::any_rows(32))
                )
                .is_ok()
        );
        // A family the selection lacks.
        assert!(matches!(
            Sm120::new().rope_kv_write(&paged_op(false), Need::rows(4)),
            Err(Refused::Rule(_))
        ));
    }

    #[test]
    fn launches_check_their_buffers() {
        let backend = backend();
        let paged = backend
            .rope_kv_write(&paged_op(true), Need::rows(64))
            .unwrap();
        let good = rope_kv_buffers(true, 1, 1, [1.0, 1.0]);
        assert!(paged.launches(&good, View::EMPTY, Counts::rows(64)).is_ok());
        assert!(paged.launches(&good, View::EMPTY, Counts::rows(0)).is_err());
        assert!(
            paged
                .launches(&good, View::EMPTY, Counts::rows(65))
                .is_err()
        );
        let short_q = RopeKvBuffers {
            q: view(2, 64 * u64::from(QH * QK) * 2 - 2),
            ..good
        };
        assert!(
            paged
                .launches(&short_q, View::EMPTY, Counts::rows(64))
                .is_err()
        );
        assert!(
            paged
                .launches(&short_q, View::EMPTY, Counts::rows(63))
                .is_ok()
        );
        let no_tables = RopeKvBuffers {
            block_tables: None,
            ..good
        };
        assert!(
            paged
                .launches(&no_tables, View::EMPTY, Counts::rows(1))
                .is_err()
        );
        let short_tables = RopeKvBuffers {
            block_tables: Some(view(7, 1023 * 4)),
            ..good
        };
        assert!(
            paged
                .launches(&short_tables, View::EMPTY, Counts::rows(1))
                .is_err()
        );
        for scales in [[0.0, 1.0], [1.0, f32::INFINITY]] {
            let unscaled = RopeKvBuffers {
                cache: cache(true, 1, 1, scales),
                ..good
            };
            assert!(
                paged
                    .launches(&unscaled, View::EMPTY, Counts::rows(1))
                    .is_err()
            );
        }
        let short_pool = RopeKvBuffers {
            cache: KvCache {
                v: view(6, u64::from(PAGES - 1) * V_PAGE_STRIDE),
                ..good.cache
            },
            ..good
        };
        assert!(
            paged
                .launches(&short_pool, View::EMPTY, Counts::rows(1))
                .is_err()
        );
        let no_table = RopeKvBuffers {
            cos_sin: None,
            ..good
        };
        assert!(
            paged
                .launches(&no_table, View::EMPTY, Counts::rows(1))
                .is_err()
        );

        // A ring's slots are packed as its entries index them.
        let ring = backend
            .rope_kv_write(&ring_op(true), Need::rows(4))
            .unwrap();
        let good = rope_kv_buffers(false, 2, 2, [1.0, 1.0]);
        assert!(ring.launches(&good, View::EMPTY, Counts::rows(4)).is_ok());
        let strided = RopeKvBuffers {
            cache: KvCache {
                k_slot_stride: good.cache.k_slot_stride * 2,
                ..good.cache
            },
            ..good
        };
        assert!(
            ring.launches(&strided, View::EMPTY, Counts::rows(4))
                .is_err()
        );
        let no_slots = RopeKvBuffers {
            slot_ids: None,
            ..good
        };
        assert!(
            ring.launches(&no_slots, View::EMPTY, Counts::rows(4))
                .is_err()
        );
        // A ring RoPE that does not store needs its K and V rows.
        let prefill = backend
            .rope_kv_write(&ring_op(false), Need::rows(4))
            .unwrap();
        let no_v = RopeKvBuffers {
            v_new: None,
            ..good
        };
        assert!(
            prefill
                .launches(&no_v, View::EMPTY, Counts::rows(4))
                .is_err()
        );
        assert!(ring.launches(&no_v, View::EMPTY, Counts::rows(4)).is_ok());
    }

    #[test]
    fn entries_prepare_alike_in_every_order() {
        let ring = rope_kv_entry(2, true, true, false, 32);
        let paged = rope_kv_entry(1, false, false, false, 64);
        let alone = rope_entry(4, true);
        let write = ring_write_entry(false, false);
        let with = |entry: &RowInstantiation, pages| {
            RopeKvEntry::new(entry.clone()).with_ring_max_pages(pages)
        };
        let renamed = |entry: &RowInstantiation| {
            let mut twin = entry.clone();
            twin.symbol = format!("{}_twin", entry.symbol);
            twin
        };
        type Symbols = Result<Vec<String>, Refused>;
        let symbols = |info: &OpInfo| info.symbols.iter().map(|(s, _)| s.clone()).collect();
        let prepare = |order: &[RopeKvEntry], which: u8| -> Symbols {
            let backend = Sm120::new().with_rope_kv(order.to_vec());
            let need = Need::rows(4);
            match which {
                0 => backend
                    .rope_kv_write(&ring_op(false), need)
                    .map(|op| symbols(op.info())),
                1 => backend
                    .rope_kv_write(&paged_op(false), need)
                    .map(|op| symbols(op.info())),
                2 => backend
                    .rope(
                        &RopeOp {
                            rope: alone.query,
                            heads: 4,
                            positions: DType::I32,
                        },
                        need,
                    )
                    .map(|op| symbols(op.info())),
                _ => backend
                    .ring_write(
                        &RingWriteOp {
                            query: write.query,
                            positions: DType::I32,
                            single_sequence: false,
                        },
                        need,
                    )
                    .map(|op| symbols(op.info())),
            }
        };
        let in_every_order = |registered: &[RopeKvEntry], which| {
            let results = orders(registered)
                .iter()
                .map(|order| prepare(order, which))
                .collect::<Vec<_>>();
            assert!(results.iter().all(|result| *result == results[0]));
            results[0].clone()
        };
        let tied = |result: Symbols, a: &str, b: &str| {
            let mut names = [a, b];
            names.sort_unstable();
            matches!(result, Err(Refused::Rule(rule)) if rule.contains(&names.join(" and ")))
        };
        // The fill only a ring launch reads: two fills of a ring entry
        // tie, and of the other entries are one entry.
        let registered = [
            with(&ring, 0),
            with(&ring, RING_MAX_PAGES),
            with(&paged, 0),
            with(&paged, RING_MAX_PAGES),
            with(&alone, 0),
            with(&alone, RING_MAX_PAGES),
            with(&write, 0),
            with(&write, RING_MAX_PAGES),
        ];
        let registered = &registered[..];
        assert!(tied(
            in_every_order(&registered[..2], 0),
            &ring.symbol,
            &ring.symbol
        ));
        assert_eq!(
            in_every_order(&registered[2..4], 1),
            Ok(vec![paged.symbol.clone()])
        );
        assert_eq!(
            in_every_order(&registered[4..6], 2),
            Ok(vec![alone.symbol.clone()])
        );
        assert_eq!(
            in_every_order(&registered[6..], 3),
            Ok(vec![write.symbol.clone()])
        );
        // A ring op ignores the page size, so ring entries of two page
        // sizes tie; entries of two symbols tie for every op.
        let other_page = rope_kv_entry(2, true, true, false, 64);
        let pages = [with(&ring, 0), with(&other_page, 0)];
        assert!(tied(
            in_every_order(&pages, 0),
            &ring.symbol,
            &other_page.symbol
        ));
        for (index, entry) in [(1, &paged), (2, &alone), (3, &write)] {
            let twin = renamed(entry);
            let registered = [with(entry, 0), with(&twin, 0)];
            assert!(tied(
                in_every_order(&registered, index),
                &entry.symbol,
                &twin.symbol
            ));
        }
    }

    /// The RoPE KV keys hold what their entries launch, every field changed
    /// alone: the symbol and ABI, and a ring launch's `max_pages` fill,
    /// which a paged launch never reads; a ring ignores the page size in
    /// its body. The query, grid, threads and (for a paged op) the body's
    /// page size select the op.
    #[test]
    fn launch_keys_hold_what_rope_kv_entries_launch() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let other = Profile::synthetic(crate::ProfileNames {
            name: "rope-kv-other",
            tag: "rope_kv_other",
        });
        let mut names = field_names!(RopeKvEntry {
            entry,
            ring_max_pages
        });
        names.retain(|name| *name != "entry");
        names.extend(field_names!(RowInstantiation {
            profile,
            symbol,
            query,
            width,
            grid,
            threads,
            params,
            body,
            invariance
        }));
        // The body's page size: 64 for a paged entry, 32 for a ring's.
        let repage = |e: &mut RopeKvEntry| {
            let page = if e.entry.body.contains(",64>") {
                ",64>"
            } else {
                ",32>"
            };
            e.entry.body = e.entry.body.replace(page, ",16>");
        };
        let fields = |ring: bool| -> Vec<Field<RopeKvEntry>> {
            vec![
                (
                    "profile",
                    Class::Inert,
                    Box::new(move |e| e.entry.profile = other),
                ),
                (
                    "symbol",
                    Class::Launch,
                    Box::new(|e| e.entry.symbol = "chop_row_other_v1".into()),
                ),
                (
                    "query",
                    Class::Selects,
                    Box::new(|e| e.entry.query = rope_entry(4, false).query),
                ),
                ("width", Class::Inert, Box::new(|e| e.entry.width *= 2)),
                (
                    "grid",
                    Class::Selects,
                    Box::new(|e| e.entry.grid = Grid::Rows),
                ),
                (
                    "threads",
                    Class::Selects,
                    Box::new(|e| e.entry.threads *= 2),
                ),
                (
                    "params",
                    Class::Launch,
                    Box::new(|e| e.entry.params.swap(0, 1)),
                ),
                (
                    "body",
                    if ring { Class::Unread } else { Class::Selects },
                    Box::new(repage),
                ),
                (
                    "invariance",
                    Class::Inert,
                    Box::new(|e| e.entry.invariance = Invariance::NONE),
                ),
                (
                    "ring_max_pages",
                    if ring { Class::Launch } else { Class::Inert },
                    Box::new(|e| e.ring_max_pages = 7),
                ),
            ]
        };
        let code = |e: &RopeKvEntry| format!("{:?} {}", e.entry.params, e.entry.body);
        for (ring, entry, op, buffers) in [
            (
                false,
                rope_kv_entry(1, false, false, false, 64),
                paged_op(false),
                rope_kv_buffers(true, 1, 2, [1.0, 1.0]),
            ),
            (
                true,
                rope_kv_entry(2, true, false, false, 32),
                ring_op(true),
                rope_kv_buffers(false, 2, 2, [1.0, 1.0]),
            ),
        ] {
            let base = RopeKvEntry::new(entry).with_ring_max_pages(RING_MAX_PAGES);
            keys_follow_launches(
                &base,
                names.clone(),
                fields(ring),
                |e| rope_kv_launch(e, ring),
                |e| {
                    let prepared = Sm120::new()
                        .with_rope_kv(vec![e.clone()])
                        .rope_kv_write(&op, Need::rows(32))?;
                    let launches = [1, 32].map(|rows| {
                        prepared
                            .launches(&buffers, View::EMPTY, Counts::rows(rows))
                            .unwrap()
                    });
                    Ok((code(e), launches))
                },
            );
        }
    }
}
