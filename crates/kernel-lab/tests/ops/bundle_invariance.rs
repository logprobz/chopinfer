//! The bundle invariance adapter: every `OpKind::Bundle` entry, the kernels
//! the model bundles own (`bundle_mimo26::sm120::declared_kernels` and
//! `model_qwen38_contract::owned::declared`). Each case runs the bundle's
//! own source and records the bundle's symbol:
//!
//! - MiMo's ring backups (`bundle_symbol(tp, "ring_backup")` for a ring of
//!   32 slots and its `_s{slots}` siblings for every other slot count, per
//!   topology) from the bundle's embedded `{tag}_ring` and `{tag}_ring_slots`
//!   cubins (`bundle_mimo26::embedded_cubin`): the bundle's entries
//!   themselves, each over a ring of its own slots.
//! - Qwen's PLE residual, PLE stage, PLE checkpoints,
//!   verify PLE convolution and TP4 verify n-gram from kernel-lab's
//!   `probe_sm120` cubin, and the TP2 verify n-gram from
//!   `probe_qwen38_tp2`. kernel-lab's build compiles
//!   the bundle's headers (`qwen38_bundle.cuh`, `qwen38_speculation.cuh`)
//!   into `chop_probe_*` entries with the bodies, parameters and launch
//!   bounds of the bundle's (`models/qwen38/bundle/src/sm120/build.rs`) and the rank
//!   shape's speculation defines, a superset of the bundle cubins' defines.
//!   The only template argument is the n-gram's hash heads, which those
//!   defines set: TP4's in `probe_sm120` as in `qwen38_speculation`, the
//!   rank's in `probe_{tag}` as in `{tag}_speculation`.
//! - Qwen's PLE anchor, which no lab cubin holds, from a cubin this adapter
//!   builds once per process with `kernel_build` from the bundle's
//!   speculation header with the bundle's TP4 entry spec and symbol.
//!
//! - The registered n-gram hash entries the catalogs declare
//!   (`ModelCatalog::ngram_hashes`), from the cubins their bundles embed
//!   (`CHOPINFER_KERNEL_OWNED_CUBINS`), with the form's tables on every
//!   rank (each rank's heads, of both hash orders) and each item's history
//!   in a PLE entry at the served stride.
//!
//! The decode and prefill rows the cases compare against or run around the
//! bundle's entries come from the lab's `row_qwen38` cubin (and
//! `row_qwen38_tp2` for those ranks' n-gram decode).
//!
//! Units, per entry:
//!
//! - Ring backup: items, each its backed-up ring entries (every layer's K
//!   and V of each position it touches) at the decode, verify and prefill
//!   spans in turn. Items own distinct slots up to the ring's slots, then
//!   the items sharing a slot touch disjoint parts of its ring; rows cases
//!   reach 128 and 256 items where the ring holds them. Alone, every other
//!   working slot holds another slot's cells, and a batch case's second
//!   launch gives every slot but its even items' another slot's cells, so a
//!   backup reading the wrong slot differs (the negative control
//!   `a_backup_reading_another_slot_fails_the_ring_cases_gpu`). Its chunk
//!   case backs up a 128-row prompt that wraps the ring, one launch per
//!   chunk at the prefill span; each row is its entries.
//! - PLE residual: rows.
//! - PLE stage: items, each its staged entry. Its chunk case prefills a
//!   prompt as the bundle runs a PLE prefill item: each chunk stages its
//!   slot's committed entry (fresh at row 0), runs the PLE convolution and
//!   n-gram prefill rows on it and commits their output state to the slot.
//!   Each row's outputs and the final entry are units.
//! - Verify n-gram and PLE convolution: items of one to eight rows in one
//!   verify launch, each its rows' outputs and its output state; alone,
//!   each item's rows run as one-row decode steps of one item through the
//!   decode entries (`chop_row_{tag}_ngram_decode_v1`,
//!   `chop_row_qwen38_ple_conv_items_decode_v1`), the PLE entry carried
//!   from step to step: AR's rows. Item 0 of an R-unit launch has R rows
//!   (at most eight), so a rows case of R reaches R verify rows; a batch
//!   case's other items change their rows and row counts.
//! - PLE checkpoints: as the verify rows, each item its entries after each
//!   row, which must be the PLE entries those decode steps leave. Its chunk
//!   case runs eight rows of one item in chunks, each staged from the
//!   previous chunk's last checkpoint.
//! - Hash prefill: rows, each its ids on every rank; its chunk case
//!   prefills a prompt in chunks, each after the history the previous chunk
//!   left, and the last row's unit adds the history after the prompt.
//! - Hash verify: items of one to eight rows in one verify launch, each its
//!   rows' ids and its history after them on every rank; the rows case adds
//!   item 0 verified alone (one item of four rows at four units); alone,
//!   each item's rows run as one-row decode steps through the form's decode
//!   entry, the history carried from step to step.
//! - PLE anchor: its chunk case prefills a prompt as the PLE stage's does,
//!   the last chunk anchored at the granule multiple before the prompt's
//!   last 37 rows; the anchor entry is the unit. Its anchor case is
//!   `Bundle::anchors`.
//!
//! Every buffer a compared launch writes starts poisoned, and is checked
//! right after the launch against a write map taken from the CUDA source
//! (`mimo_ring.cuh`, `qwen38_bundle.cuh`, `qwen38_speculation.cuh`,
//! `row/qwen38_ple.cuh`), each map in a named function below that the CPU
//! test `bundle_write_checks_reject_unwritten_outputs` exercises: the live
//! rows, entries or ring cells are written (finite for arithmetic outputs,
//! no longer the poison for copies, IDs and staged entries) and every other
//! byte (planned rows past the live ones, verify capacity past an item's
//! rows, entries the launch does not own) is untouched. Output states the
//! n-gram and convolution rows write in place (a decode step's or a prefill
//! chunk's entry) start as the staged entry with the parts those rows
//! write poisoned, and are checked after each launch with `written_over`.

use crate::invariance::{
    Adapter, Compared, Element, Units, inventory, row_counts, written, written_over,
};
use bundle_mimo26::{
    contract::MAX_VERIFY_ROWS,
    topology::{self as mimo, Tp},
};
use kernel_build::{BuildOptions, CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::DType;
use kernel_core::ngram_oracle;
use kernel_lab::launch::simple_config;
use kernel_lab::poison::{POISON, WriteMap, read};
use kernel_lab::{LabResult, Session};
use kernels_sm120::Profile;
use kernels_sm120::invariance::{Declared, OpKind};
use kernels_sm120::row::hyper_connection::Entry as RowEntry;
use kernels_sm120::row::ngram_hash::NgramHashForm;

use model_qwen38_contract::sm120::row::hyper_connection_rows as row_catalog;
use model_qwen38_contract::{contract, sm120::owned};
use runtime_cuda::{DeviceBuffer, KernelArgument, LaunchConfig, Module};
use std::path::PathBuf;
use std::sync::OnceLock;

const HIDDEN: usize = 2560;
const STREAMS: usize = 4;
const WIDE: usize = HIDDEN * STREAMS;
const WIDE_BYTES: usize = WIDE * 2;
/// One PLE state entry: the convolution history of nine rows, then the two
/// n-gram history tokens and padding.
const CONV_ROWS: usize = 9;
const CONV_BYTES: usize = contract::PLE_CONV_BYTES as usize;
const ENTRY: usize = contract::PLE_ENTRY_BYTES as usize;
const _: () = assert!(CONV_BYTES == CONV_ROWS * WIDE_BYTES);
/// Bytes one block of the bundle's state copies moves (`kChunk`).
const COPY_CHUNK: usize = 16 * 1024;
/// Rows the verify grids cover per item (`kVerifyRows`).
const VERIFY_ROWS: usize = contract::MAX_VERIFY_ROWS as usize;
/// Channel blocks of the PLE convolution grids.
const CONV_GROUPS: u32 = (WIDE / 256) as u32;
const EOS: u32 = 248_044;
const NGRAM_VOCAB: u64 = 248_064;
/// Prompt rows of a prefill chunk case, and the rows the anchor chunk case
/// adds past its anchor.
const PROMPT: u32 = 512;
const ANCHOR_TAIL: u32 = 37;
/// Entries of a staged pool, and the slot a prefill case commits to.
const POOL_SLOTS: usize = 64;
const PREFILL_SLOT: usize = 5;

/// A seeded 64-bit word for `salt` of `seed` (splitmix64).
fn mix(seed: u64, salt: u64) -> u64 {
    let mut z = seed ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// `count` BF16 values in [-4, 4) for `seed`, as bytes.
fn bf16_bytes(count: usize, seed: u64) -> Vec<u8> {
    (0..count as u64)
        .flat_map(|i| {
            let unit = (mix(seed, i) >> 40) as f32 / (1_u64 << 24) as f32;
            (((8.0 * unit - 4.0).to_bits() >> 16) as u16).to_ne_bytes()
        })
        .collect()
}

fn noise(len: usize, seed: u64) -> Vec<u8> {
    (0..len.div_ceil(8) as u64)
        .flat_map(|i| mix(seed, i).to_ne_bytes())
        .take(len)
        .collect()
}

/// A token for `seed`: one in thirteen ends a document.
fn token(seed: u64) -> u32 {
    let draw = mix(seed, 7);
    if draw % 13 == 0 {
        EOS
    } else {
        ((draw >> 8) % NGRAM_VOCAB) as u32
    }
}

fn u32s(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn i32s(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(16))?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

/// A buffer its launch must write: every byte starts as poison.
fn poisoned(session: &Session, len: usize) -> LabResult<DeviceBuffer> {
    upload(session, &vec![POISON; len])
}

fn download(
    session: &Session,
    buffer: &DeviceBuffer,
    offset: usize,
    len: usize,
) -> LabResult<Vec<u8>> {
    let mut out = vec![0; len];
    buffer
        .copy_to_bytes_at_async(offset, &mut out, &session.stream)
        .map_err(|e| e.to_string())?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    Ok(out)
}

/// A kernel argument: a buffer, a buffer from a byte offset, or a scalar.
#[derive(Clone, Copy)]
enum Arg<'a> {
    Buf(&'a DeviceBuffer),
    At(&'a DeviceBuffer, usize),
    U32(u32),
    U64(u64),
}

impl<'a> Arg<'a> {
    fn kernel(self) -> KernelArgument<'a> {
        match self {
            Arg::Buf(buffer) => KernelArgument::Buffer(buffer),
            Arg::At(buffer, offset) => KernelArgument::Range {
                buffer,
                offset,
                byte_len: buffer.byte_len() - offset,
            },
            Arg::U32(value) => KernelArgument::U32(value),
            Arg::U64(value) => KernelArgument::U64(value),
        }
    }
}

fn launch(
    session: &Session,
    module: &Module,
    symbol: &str,
    grid: (u32, u32, u32),
    threads: u32,
    args: &[Arg<'_>],
) -> LabResult<()> {
    let args = args.iter().map(|arg| arg.kernel()).collect::<Vec<_>>();
    module
        .function(symbol)
        .map_err(|e| format!("{symbol}: {e}"))?
        .launch(
            &session.stream,
            simple_config(grid, (threads, 1, 1), 0, LaunchConfig::NO_CLUSTER, true)?,
            &args,
        )
        .map_err(|e| format!("{symbol}: {e}"))
}

fn row_entry(symbol: &str) -> LabResult<RowEntry> {
    row_catalog()
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .ok_or_else(|| format!("{symbol}: not in the Qwen3.8 row catalog"))
}

/// Launches a `row::qwen38` entry of `module` over `rows` rows (or items),
/// its parameters bound by name.
fn launch_row(
    session: &Session,
    module: &Module,
    entry: &RowEntry,
    rows: u32,
    bind: &[(&str, Arg<'_>)],
) -> LabResult<()> {
    let symbol = &entry.symbol;
    let plan = entry
        .plan(rows, rows)
        .map_err(|e| format!("{symbol}: {e}"))?;
    let args = entry
        .params
        .iter()
        .map(|param| {
            let name = match param {
                EntryParam::DevicePointer { name, .. }
                | EntryParam::U32 { name }
                | EntryParam::U64 { name } => name,
                _ => return Err(format!("{symbol}: unexpected parameter kind")),
            };
            bind.iter()
                .find(|(bound, _)| bound == name)
                .map(|&(_, arg)| arg)
                .ok_or_else(|| format!("{symbol}: no argument {name}"))
        })
        .collect::<LabResult<Vec<_>>>()?;
    let [x, y, z] = plan.grid;
    launch(session, module, symbol, (x, y, z), plan.block[0], &args)?;
    kernel_lab::launched(symbol);
    Ok(())
}

// MiMo's ring backup.

const RING_TOKENS: u64 = 128;
/// The slots of the unsuffixed backup entry's ring; every other slot
/// count's entry names its own (`_s{slots}`).
const RING_SLOTS: u64 = 32;
const RING_LAYERS: u64 = 39;
/// Recent positions one block row of the backup covers, as the bundle's
/// steps launch it: a decode step's one, a verify step's
/// `MAX_VERIFY_ROWS` and a prefill's window. Every rows and batch case runs
/// each; a chunk case runs the prefill's.
const RING_SPANS: [u32; 3] = [1, MAX_VERIFY_ROWS, RING_PREFILL_SPAN];
const RING_PREFILL_SPAN: u32 = RING_TOKENS as u32;
/// The chunk case's slot (modulo the ring's slots) and first position, 104
/// positions into the ring so that the prompt wraps it.
const RING_SLOT: i32 = 5;
const RING_BASE: i32 = 1000;

/// A rank's SWA ring at its topology (`mimo_ring.cuh`): a working half of
/// [layer][K plane | V plane], each plane [slot][position][entry], and a
/// shadow half of the same layout that the backup writes.
#[derive(Clone, Copy)]
struct Ring {
    tp: Tp,
    slots: u64,
    k_entry: u64,
    v_entry: u64,
}

impl Ring {
    fn new(tp: Tp, slots: u64) -> Self {
        let heads = tp.kv_heads(false);
        Self {
            tp,
            slots,
            k_entry: heads * 192 * 2,
            v_entry: heads * 128 * 2,
        }
    }

    fn k_layer(self) -> u64 {
        self.slots * RING_TOKENS * self.k_entry
    }

    fn layer(self) -> u64 {
        self.k_layer() + self.slots * RING_TOKENS * self.v_entry
    }

    fn working(self) -> u64 {
        RING_LAYERS * self.layer()
    }

    /// Every layer's K plane, then its V plane: (offset in a half, bytes of
    /// one slot's cells).
    fn planes(self) -> Vec<(usize, usize)> {
        (0..RING_LAYERS)
            .flat_map(|layer| {
                let base = layer * self.layer();
                [
                    (base, RING_TOKENS * self.k_entry),
                    (base + self.k_layer(), RING_TOKENS * self.v_entry),
                ]
            })
            .map(|(at, len)| (at as usize, len as usize))
            .collect()
    }

    /// `slot`'s shadow byte ranges: per layer its K cells, then its V cells.
    fn slot_ranges(self, slot: i32) -> Vec<(usize, usize)> {
        let working = self.working() as usize;
        self.planes()
            .into_iter()
            .map(|(plane, len)| (working + plane + slot as usize * len, len))
            .collect()
    }

    /// The shadow byte ranges, from the shadow's first byte, of `slot`'s
    /// cell of `position`: every layer's K entry, then its V entry.
    fn cell_ranges(self, slot: i32, position: i32) -> Vec<std::ops::Range<usize>> {
        let cell = position as u64 % RING_TOKENS;
        self.planes()
            .into_iter()
            .map(|(plane, len)| {
                let entry = len / RING_TOKENS as usize;
                let at = plane + slot as usize * len + cell as usize * entry;
                at..at + entry
            })
            .collect()
    }

    /// The byte ranges of `position`'s cell in a slot's image (its
    /// `slot_ranges` in order): every layer's K entry, then its V entry.
    fn slot_cell_ranges(self, position: i32) -> Vec<std::ops::Range<usize>> {
        let cell = position as u64 % RING_TOKENS;
        let mut at = 0;
        self.planes()
            .into_iter()
            .map(|(_, len)| {
                let entry = len / RING_TOKENS as usize;
                let start = at + cell as usize * entry;
                at += len;
                start..start + entry
            })
            .collect()
    }

    /// The bytes of one slot's image.
    fn slot_bytes(self) -> usize {
        (RING_LAYERS * RING_TOKENS * (self.k_entry + self.v_entry)) as usize
    }

    /// The shadow entries of `position` in `slot` from the shadow's image:
    /// every layer's K, then V.
    fn cell(self, shadow: &[u8], slot: i32, position: i32) -> Vec<u8> {
        self.cell_ranges(slot, position)
            .into_iter()
            .flat_map(|range| shadow[range].to_vec())
            .collect()
    }

    /// The shadow cells a backup of `items` at `span` writes (`backup` in
    /// `mimo_ring.cuh`): for each live item, every layer's K and V entry of
    /// each position it touches (`ring_touched`), in 16-byte vectors. Every
    /// other cell of the shadow, other slots' and the item's own untouched
    /// positions, stays as it was.
    fn shadow_written(
        self,
        label: &str,
        before: &[u8],
        after: &[u8],
        items: &[RingItem],
        span: u32,
    ) -> Result<(), String> {
        let map = WriteMap::of_records::<16>().byte_ranges(items.iter().flat_map(|&item| {
            ring_touched(item, span).flat_map(move |position| self.cell_ranges(item.0, position))
        }))?;
        written_over(label, before, after, &map, Element::Raw(16))
    }

    /// As `shadow_written` for one item over its slot's image, which was all
    /// poison before the launch.
    fn slot_written(
        self,
        label: &str,
        after: &[u8],
        item: RingItem,
        span: u32,
    ) -> Result<(), String> {
        let map = WriteMap::of_records::<16>().byte_ranges(
            ring_touched(item, span).flat_map(|position| self.slot_cell_ranges(position)),
        )?;
        written_over(
            label,
            &vec![POISON; after.len()],
            after,
            &map,
            Element::Raw(16),
        )
    }
}

/// A backup item as the bundle packs it: (slot, q_len, prefix_len).
type RingItem = (i32, i32, i32);

/// Item `index` of a launch of `count` items for `seed` in a ring of `slots`
/// slots. Up to `slots` items own distinct slots and touch up to 200
/// positions (the last 128 of them) anywhere; past that, the items sharing a
/// slot (`i`, `i + slots`, ...) split its ring into as many parts, each item
/// touching up to a part's positions in its own part.
fn ring_item(index: usize, count: usize, slots: u64, seed: u64) -> LabResult<RingItem> {
    let slots = slots as usize;
    let parts = count.div_ceil(slots) as u64;
    if parts > RING_TOKENS {
        return Err(format!(
            "{count} ring items, at most {} fit disjoint cells",
            RING_TOKENS as usize * slots
        ));
    }
    let slot = ((7 * (index % slots) + 3) % slots) as i32;
    let (q_len, prefix) = if parts == 1 {
        (1 + mix(seed, 90) % 200, mix(seed, 91) % 5000)
    } else {
        let (part, width) = ((index / slots) as u64, RING_TOKENS / parts);
        (
            1 + mix(seed, 90) % width,
            RING_TOKENS * (mix(seed, 91) % 40) + width * part,
        )
    };
    Ok((slot, q_len as i32, prefix as i32))
}

/// The positions an item's backup at `span` saves: its last
/// min(q_len, span, 128).
fn ring_touched((_, q_len, prefix): RingItem, span: u32) -> std::ops::Range<i32> {
    let end = prefix + q_len;
    end - q_len.min(span.min(RING_TOKENS as u32) as i32)..end
}

/// A rank's ring state on the device with the bundle's backup entry: the
/// working half a pattern of its own offsets, the shadow poisoned, and a
/// pristine copy of the working pattern after them, from which
/// `scramble` rewrites slots.
struct RingState<'s> {
    session: &'s Session,
    ring: Ring,
    state: DeviceBuffer,
    module: Module,
    symbol: String,
    /// Emulates a backup that reads the next slot's working cells: after
    /// each launch, every saved cell is overwritten with that slot's (the
    /// negative control).
    reads_next_slot: bool,
}

impl<'s> RingState<'s> {
    fn new(
        session: &'s Session,
        ring: Ring,
        symbol: &str,
        reads_next_slot: bool,
    ) -> LabResult<Self> {
        let working = ring.working() as usize;
        let state = session.allocate(3 * working)?;
        let mut pattern = vec![0_u8; working];
        for (index, word) in pattern.chunks_exact_mut(4).enumerate() {
            word.copy_from_slice(&(index as u32).wrapping_mul(2_654_435_761).to_le_bytes());
        }
        state
            .copy_from_bytes_at(0, &pattern, &session.stream)
            .map_err(|e| e.to_string())?;
        drop(pattern);
        state
            .copy_within_async(0, 2 * working, working, &session.stream)
            .map_err(|e| e.to_string())?;
        state
            .copy_from_bytes_at(working, &vec![POISON; working], &session.stream)
            .map_err(|e| e.to_string())?;
        // The unsuffixed entry's cubin, or the one of every other slot count.
        let cubin = if ring.slots == RING_SLOTS {
            ""
        } else {
            "_slots"
        };
        let name = format!("{}_ring{cubin}", mimo::kernel_tag(ring.tp));
        let cubin = bundle_mimo26::embedded_cubin(&name)
            .ok_or_else(|| format!("bundle-mimo26 embeds no {name}"))?;
        Ok(Self {
            session,
            ring,
            state,
            module: session.load_cubin(cubin)?,
            symbol: symbol.into(),
            reads_next_slot,
        })
    }

    /// Copies `len` bytes at `from` of the state onto `to`.
    fn copy(&self, from: usize, to: usize, len: usize) -> LabResult<()> {
        self.state
            .copy_within_async(from, to, len, &self.session.stream)
            .map_err(|e| e.to_string())
    }

    /// Gives every working slot but `kept` the pristine cells of the next
    /// slot (the last slot the first's), and every `kept` slot its own: a
    /// backup that reads a slot other than its item's then saves other bytes
    /// than over the pristine ring.
    fn scramble(&self, kept: &[i32]) -> LabResult<()> {
        let (pristine, slots) = (2 * self.ring.working() as usize, self.ring.slots as usize);
        for (plane, len) in self.ring.planes() {
            if slots > 1 {
                self.copy(pristine + plane + len, plane, (slots - 1) * len)?;
                self.copy(pristine + plane, plane + (slots - 1) * len, len)?;
            }
            for &slot in kept {
                let at = plane + slot as usize * len;
                self.copy(pristine + at, at, len)?;
            }
        }
        Ok(())
    }

    /// Backs up `items` at `span` in one launch over their power-of-two
    /// bucket.
    fn backup(&self, items: &[RingItem], span: u32) -> LabResult<()> {
        let session = self.session;
        let column = |f: fn(&RingItem) -> i32| {
            upload(session, &i32s(&items.iter().map(f).collect::<Vec<_>>()))
        };
        let (slots, q_len, prefix) = (column(|i| i.0)?, column(|i| i.1)?, column(|i| i.2)?);
        let live = upload(session, &u32s(&[items.len() as u32]))?;
        let planned = items.len().next_power_of_two() as u32;
        launch(
            session,
            &self.module,
            &self.symbol,
            (planned * span, RING_LAYERS as u32, 1),
            mimo::ring_threads(self.ring.tp),
            &[
                Arg::Buf(&self.state),
                Arg::Buf(&slots),
                Arg::Buf(&q_len),
                Arg::Buf(&prefix),
                Arg::Buf(&live),
                Arg::U32(span),
            ],
        )?;
        kernel_lab::launched(&self.symbol);
        if self.reads_next_slot {
            let (working, slots) = (self.ring.working() as usize, self.ring.slots as i32);
            for &item in items {
                let next = (item.0 + 1) % slots;
                for position in ring_touched(item, span) {
                    let cells = self.ring.cell_ranges(next, position);
                    for (from, to) in cells
                        .into_iter()
                        .zip(self.ring.cell_ranges(item.0, position))
                    {
                        self.copy(from.start, working + to.start, from.len())?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Backs up `items` at `span` in one launch and checks the shadow
    /// against its image `before` the launch (`Ring::shadow_written`); the
    /// shadow's image after it.
    fn backup_checked(&self, items: &[RingItem], before: &[u8], span: u32) -> LabResult<Vec<u8>> {
        self.backup(items, span)?;
        let working = self.ring.working() as usize;
        let after = download(self.session, &self.state, working, working)?;
        self.ring.shadow_written(
            &format!("{} shadow at span {span}", self.symbol),
            before,
            &after,
            items,
            span,
        )?;
        Ok(after)
    }

    /// Poisons `slot`'s shadow again.
    fn reset_slot(&self, slot: i32) -> LabResult<()> {
        for (at, len) in self.ring.slot_ranges(slot) {
            self.state
                .copy_from_bytes_at(at, &vec![POISON; len], &self.session.stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Poisons the whole shadow again.
    fn reset_shadow(&self) -> LabResult<()> {
        let working = self.ring.working() as usize;
        self.state
            .copy_from_bytes_at(working, &self.poisoned_shadow(), &self.session.stream)
            .map_err(|e| e.to_string())
    }

    /// An item's unit: the shadow entries of each position it touches at
    /// `span`.
    fn unit(&self, item: RingItem, shadow: &[u8], span: u32) -> Vec<u8> {
        ring_touched(item, span)
            .flat_map(|position| self.ring.cell(shadow, item.0, position))
            .collect()
    }

    /// The shadow's image before any backup: all poison.
    fn poisoned_shadow(&self) -> Vec<u8> {
        vec![POISON; self.ring.working() as usize]
    }
}

/// The shadow a ring's items alone are backed up into: the device's
/// (`RingState`), or a host image in the CPU tests.
trait AloneShadow {
    fn ring(&self) -> Ring;
    /// Backs up `item` alone at `span`.
    fn launch(&self, item: RingItem, span: u32) -> LabResult<()>;
    /// `slot`'s image (its `Ring::slot_ranges` in order).
    fn slot_image(&self, slot: i32) -> LabResult<Vec<u8>>;
    fn reset_slot(&self, slot: i32) -> LabResult<()>;
    /// The whole shadow.
    fn image(&self) -> LabResult<Vec<u8>>;
}

impl AloneShadow for RingState<'_> {
    fn ring(&self) -> Ring {
        self.ring
    }

    fn launch(&self, item: RingItem, span: u32) -> LabResult<()> {
        self.backup(&[item], span)
    }

    fn slot_image(&self, slot: i32) -> LabResult<Vec<u8>> {
        let mut image = Vec::with_capacity(self.ring.slot_bytes());
        for (at, len) in self.ring.slot_ranges(slot) {
            image.extend(download(self.session, &self.state, at, len)?);
        }
        Ok(image)
    }

    fn reset_slot(&self, slot: i32) -> LabResult<()> {
        RingState::reset_slot(self, slot)
    }

    fn image(&self) -> LabResult<Vec<u8>> {
        let working = self.ring.working() as usize;
        download(self.session, &self.state, working, working)
    }
}

/// Fails at the first byte of `image` that is not poison.
fn all_poison(label: &str, image: &[u8]) -> LabResult<()> {
    match image.iter().position(|&byte| byte != POISON) {
        Some(at) => Err(format!("{label}: byte {at} written")),
        None => Ok(()),
    }
}

/// Each of `items` backed up alone at `span` into a poisoned shadow; their
/// units. Every launch first finds its slot all poison (no earlier launch
/// wrote it, before this launch could overwrite or reset the bytes), then
/// writes exactly its cells there (`Ring::slot_written`), and its slot is
/// poisoned again; after the last, the whole shadow must be poison (no
/// launch wrote a slot no later launch checked). A launch's write outside
/// its cells therefore fails whatever the later launches do.
fn ring_alone_span(
    shadow: &impl AloneShadow,
    label: &str,
    items: &[RingItem],
    span: u32,
    before: &mut dyn FnMut(RingItem) -> LabResult<()>,
) -> LabResult<Units> {
    let ring = shadow.ring();
    let units = items
        .iter()
        .map(|&item| {
            let slot = item.0;
            let label = format!("{label} slot {slot} at span {span}");
            all_poison(
                &format!("{label} before its launch"),
                &shadow.slot_image(slot)?,
            )?;
            before(item)?;
            shadow.launch(item, span)?;
            let image = shadow.slot_image(slot)?;
            ring.slot_written(&label, &image, item, span)?;
            let unit = ring_touched(item, span)
                .flat_map(|position| {
                    ring.slot_cell_ranges(position)
                        .into_iter()
                        .flat_map(|range| image[range].to_vec())
                        .collect::<Vec<_>>()
                })
                .collect();
            shadow.reset_slot(slot)?;
            Ok(unit)
        })
        .collect::<LabResult<Units>>()?;
    all_poison(
        &format!("{label} shadow after its items alone at span {span}"),
        &shadow.image()?,
    )?;
    Ok(units)
}

fn ring_items(ring: Ring, seeds: &[u64]) -> LabResult<Vec<RingItem>> {
    seeds
        .iter()
        .enumerate()
        .map(|(index, &seed)| ring_item(index, seeds.len(), ring.slots, seed))
        .collect()
}

/// Each item's unit: its saved entries after one launch of all items at
/// each span in turn (`RING_SPANS`), each launch over a poisoned shadow.
/// With `scramble`, every working slot but the even items' holds another
/// slot's pristine cells (`RingState::scramble`), so the even items of a
/// batch case's two launches read the same slots while their neighbors
/// differ.
fn ring_launches(
    session: &Session,
    ring: Ring,
    symbol: &str,
    seeds: &[u64],
    scramble: bool,
    reads_next_slot: bool,
) -> LabResult<Units> {
    let items = ring_items(ring, seeds)?;
    let state = RingState::new(session, ring, symbol, reads_next_slot)?;
    if scramble {
        let kept = items
            .iter()
            .step_by(2)
            .map(|item| item.0)
            .collect::<Vec<_>>();
        state.scramble(&kept)?;
    }
    let poisoned = state.poisoned_shadow();
    let mut units = vec![Vec::new(); items.len()];
    for span in RING_SPANS {
        let shadow = state.backup_checked(&items, &poisoned, span)?;
        for (unit, &item) in units.iter_mut().zip(&items) {
            unit.extend(state.unit(item, &shadow, span));
        }
        state.reset_shadow()?;
    }
    Ok(units)
}

/// Each item of the same layout backed up in a launch of its own at each
/// span, every other working slot holding another slot's pristine cells
/// (`ring_alone_span`).
fn ring_alone(
    session: &Session,
    ring: Ring,
    symbol: &str,
    seeds: &[u64],
    reads_next_slot: bool,
) -> LabResult<Units> {
    let items = ring_items(ring, seeds)?;
    let state = RingState::new(session, ring, symbol, reads_next_slot)?;
    let mut units = vec![Vec::new(); items.len()];
    for span in RING_SPANS {
        let spanned = ring_alone_span(&state, symbol, &items, span, &mut |item| {
            state.scramble(&[item.0])
        })?;
        for (unit, spanned) in units.iter_mut().zip(spanned) {
            unit.extend(spanned);
        }
    }
    Ok(units)
}

/// A prompt at positions `RING_BASE..` backed up one chunk per launch at the
/// prefill span, as its prefill steps do; each row's saved entries.
fn ring_chunks(
    session: &Session,
    ring: Ring,
    symbol: &str,
    rows: u32,
    ends: &[u32],
) -> LabResult<Units> {
    if rows > RING_TOKENS as u32 {
        return Err(format!(
            "a ring chunk case of {rows} rows overwrites its own cells"
        ));
    }
    let state = RingState::new(session, ring, symbol, false)?;
    let slot = RING_SLOT % ring.slots as i32;
    // Each chunk writes cells no earlier chunk wrote (the prompt fits the
    // ring), so each launch finds its own cells poisoned.
    let mut shadow = state.poisoned_shadow();
    let mut start = 0;
    for &end in ends {
        let item = (slot, (end - start) as i32, RING_BASE + start as i32);
        shadow = state.backup_checked(&[item], &shadow, RING_PREFILL_SPAN)?;
        start = end;
    }
    Ok((0..rows as i32)
        .map(|row| state.ring.cell(&shadow, slot, RING_BASE + row))
        .collect())
}

// Qwen3.8's owned kernels.

/// The n-gram hash tables of one rank (the last, whose heads read two
/// tokens back) and its argument values.
struct Ngram {
    profile: Profile,
    local: usize,
    rank: u32,
    start: u64,
    rows: u64,
    buffers: [DeviceBuffer; 5],
}

impl Ngram {
    fn new(session: &Session, profile: Profile) -> LabResult<Self> {
        let local = model_qwen38_contract::sm120::config::profile_shape(profile)
            .ok_or_else(|| format!("{profile:?} is not a Qwen3.8 profile"))?
            .ngram_heads() as usize;
        let tables = ngram_oracle::ngram_tables(NGRAM_VOCAB, 3, 8, 17, 0, 1, 1)
            .map_err(|e| e.to_string())?;
        let rank = 16 / local - 1;
        let first = rank * local;
        let rows = tables.sizes[first..first + local].iter().sum::<u64>();
        let words = |values: &[u64]| {
            values
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        // E4M3 codes of both signs, never NaN.
        let table = (0..rows as usize * 160)
            .map(|i| (i % 127) as u8 | if i % 2 == 0 { 0 } else { 128 })
            .collect::<Vec<_>>();
        Ok(Self {
            profile,
            local,
            rank: rank as u32,
            start: tables.offsets[first],
            rows,
            buffers: [
                upload(session, &words(&tables.multipliers))?,
                upload(session, &words(&tables.sizes))?,
                upload(session, &words(&tables.offsets))?,
                upload(session, &table)?,
                // BF16 0.125.
                upload(session, &0x3e00_u16.to_ne_bytes())?,
            ],
        })
    }

    /// Bytes of one row's output, then of its row ids.
    fn output_bytes(&self) -> usize {
        self.local * 160 * 2
    }

    fn id_bytes(&self) -> usize {
        self.local * 8
    }

    /// The table arguments every n-gram entry takes, by name.
    fn bind(&self) -> Vec<(&'static str, Arg<'_>)> {
        let [multipliers, sizes, offsets, table, scale] = &self.buffers;
        vec![
            ("multipliers", Arg::Buf(multipliers)),
            ("sizes", Arg::Buf(sizes)),
            ("offsets", Arg::Buf(offsets)),
            ("table", Arg::Buf(table)),
            ("scale", Arg::Buf(scale)),
            ("rank", Arg::U32(self.rank)),
            ("table_start", Arg::U64(self.start)),
            ("table_rows", Arg::U64(self.rows)),
        ]
    }
}

/// The cubins and fixed operands of the Qwen3.8 cases at the n-gram rank of
/// `profile`: the probe holding that rank's bundle rows, TP4's row cubin and
/// the rank's.
struct Qwen<'s> {
    session: &'s Session,
    probe: Module,
    owned: Module,
    row: Module,
    ngram_row: Module,
    ngram: Ngram,
    weight: DeviceBuffer,
}

impl<'s> Qwen<'s> {
    fn new(session: &'s Session, profile: Profile) -> LabResult<Self> {
        let common = model_qwen38_contract::sm120::config::PROFILE;
        let probe = if profile == common {
            "probe_sm120".to_owned()
        } else {
            format!("probe_{}", profile.tag())
        };
        Ok(Self {
            session,
            probe: session.module(&probe)?,
            owned: session
                .context
                .load_module(&crate::owned_cubins::load(&format!(
                    "{}_speculation",
                    common.tag()
                ))?)
                .map_err(|e| e.to_string())?,
            row: session.module(&format!("row_{}", common.tag()))?,
            ngram_row: session.module(&format!("row_{}", profile.tag()))?,
            ngram: Ngram::new(session, profile)?,
            weight: upload(session, &bf16_bytes(4 * WIDE, 0x9301))?,
        })
    }

    /// Launches the probe's entry of the bundle's `symbol`, recording the
    /// bundle's symbol.
    fn launch_owned(&self, symbol: &str, grid: (u32, u32, u32), args: &[Arg<'_>]) -> LabResult<()> {
        if symbol == owned::PLE_CHECKPOINTS_INDEXED {
            launch(self.session, &self.owned, symbol, grid, 256, args)?;
        } else {
            let probe = symbol.replacen("chop_", "chop_probe_", 1);
            launch(self.session, &self.probe, &probe, grid, 256, args)?;
        }
        kernel_lab::launched(symbol);
        Ok(())
    }

    fn ngram_entry(&self, name: &str) -> LabResult<RowEntry> {
        row_entry(&format!("chop_row_{}_{name}_v1", self.ngram.profile.tag()))
    }
}

/// A row-local entry (the PLE residual), each row's
/// outputs from the rows seeded by `seeds`, planned over their power-of-two
/// bucket.
fn local_rows(q: &Qwen<'_>, symbol: &str, seeds: &[u64]) -> LabResult<Units> {
    let session = q.session;
    let rows = seeds.len();
    let planned = rows.next_power_of_two();
    let live = upload(session, &u32s(&[rows as u32]))?;
    let seeded = |width: usize, salt: u64| -> LabResult<DeviceBuffer> {
        let values = seeds
            .iter()
            .flat_map(|&seed| bf16_bytes(width, mix(seed, salt)))
            .collect::<Vec<_>>();
        upload(session, &values)
    };
    let out = poisoned(session, planned * WIDE_BYTES)?;
    let grid = (planned as u32, STREAMS as u32, 1);
    let (x, gated, conv) = (seeded(WIDE, 81)?, seeded(WIDE, 82)?, seeded(WIDE, 83)?);
    q.launch_owned(
        symbol,
        grid,
        &[
            Arg::Buf(&x),
            Arg::Buf(&gated),
            Arg::Buf(&conv),
            Arg::Buf(&out),
            Arg::Buf(&live),
        ],
    )?;
    let out = read(session, &out)?;
    local_written(symbol, &out, rows)?;
    Ok(out[..rows * WIDE_BYTES]
        .chunks(WIDE_BYTES)
        .map(<[u8]>::to_vec)
        .collect())
}

/// The rows a row-local entry writes (`ple_residual` in
/// `qwen38_bundle.cuh`): every stream of each row below the live count, as
/// finite sums; the planned rows past it stay poisoned.
fn local_written(symbol: &str, out: &[u8], rows: usize) -> Result<(), String> {
    let map = WriteMap::of(DType::Bf16).byte_prefix(rows * WIDE_BYTES)?;
    written(&format!("{symbol} output"), out, &map, Element::Bf16)
}

/// Each prompt row r (seed r) of row-local chunks ending at `ends`.
fn local_chunks(q: &Qwen<'_>, symbol: &str, ends: &[u32]) -> LabResult<Units> {
    let mut units = Vec::new();
    let mut start = 0;
    for &end in ends {
        let seeds = (start..end).map(u64::from).collect::<Vec<_>>();
        units.extend(local_rows(q, symbol, &seeds)?);
        start = end;
    }
    Ok(units)
}

/// PLE staging of an item per seed (its slot and prefix length from the
/// seed, a quarter of them fresh) from a pool of `POOL_SLOTS` entries over
/// the items' power-of-two bucket; each item's staged entry.
fn stage_units(q: &Qwen<'_>, seeds: &[u64]) -> LabResult<Units> {
    let session = q.session;
    let items = seeds.len();
    let planned = items.next_power_of_two();
    let pool = upload(session, &noise(POOL_SLOTS * ENTRY, 0x9002))?;
    let slots = seeds
        .iter()
        .map(|&seed| (mix(seed, 70) % POOL_SLOTS as u64) as i32)
        .collect::<Vec<_>>();
    let prefix = seeds
        .iter()
        .map(|&seed| match mix(seed, 71) {
            draw if draw % 4 == 0 => 0,
            draw => 1 + (draw >> 8) as i32 % 5000,
        })
        .collect::<Vec<_>>();
    let (slots, prefix) = (
        upload(session, &i32s(&slots))?,
        upload(session, &i32s(&prefix))?,
    );
    let live = upload(session, &u32s(&[items as u32]))?;
    let staging = poisoned(session, planned * ENTRY)?;
    q.launch_owned(
        owned::PLE_STAGE,
        (planned as u32, ENTRY.div_ceil(COPY_CHUNK) as u32, 1),
        &[
            Arg::Buf(&pool),
            Arg::Buf(&staging),
            Arg::Buf(&slots),
            Arg::Buf(&prefix),
            Arg::Buf(&live),
        ],
    )?;
    let staging = read(session, &staging)?;
    stage_written(owned::PLE_STAGE, &staging, items)?;
    Ok(staging[..items * ENTRY]
        .chunks(ENTRY)
        .map(<[u8]>::to_vec)
        .collect())
}

/// The entries the PLE stage writes (`ple_stage` in `qwen38_bundle.cuh`):
/// each live item's whole entry, copied from its slot or fresh, in 16-byte
/// vectors; the planned items' entries past the live ones stay poisoned.
fn stage_written(label: &str, staging: &[u8], items: usize) -> Result<(), String> {
    let map = WriteMap::of_records::<16>().byte_prefix(items * ENTRY)?;
    written(&format!("{label} staging"), staging, &map, Element::Raw(16))
}

/// The output rows the n-gram rows write (`ngram_row` in
/// `row/qwen38_ple.cuh`, through the verify, decode and prefill entries):
/// for each of the `rows` live rows, every local head's 160 BF16 outputs
/// (finite: E4M3 codes times a scale) and its table row id; the capacity
/// past the live rows stays poisoned.
fn ngram_written(
    label: &str,
    output: &[u8],
    row_ids: &[u8],
    rows: usize,
    local: usize,
) -> Result<(), String> {
    let map = WriteMap::of(DType::Bf16).prefix(rows * local * 160)?;
    written(&format!("{label} output"), output, &map, Element::Bf16)?;
    let map = WriteMap::of(DType::I64).prefix(rows * local)?;
    written(&format!("{label} row ids"), row_ids, &map, Element::Raw(8))
}

/// The output rows the PLE convolution writes (`ple_conv_row` in
/// `row/qwen38_ple.cuh`): every channel of each of the `rows` live rows,
/// finite; the capacity past them stays poisoned.
fn conv_written(label: &str, output: &[u8], rows: usize) -> Result<(), String> {
    let map = WriteMap::of(DType::Bf16).byte_prefix(rows * WIDE_BYTES)?;
    written(&format!("{label} output"), output, &map, Element::Bf16)
}

/// The output state the n-gram rows write into `items` PLE entries in
/// place (`ngram_row`, its last row): each entry's two n-gram history
/// tokens; the convolution history and the padding keep their image.
fn history_written(label: &str, before: &[u8], after: &[u8], items: usize) -> Result<(), String> {
    let map = WriteMap::of(DType::I32)
        .byte_ranges((0..items).map(|z| z * ENTRY + CONV_BYTES..z * ENTRY + CONV_BYTES + 8))?;
    written_over(
        &format!("{label} history"),
        before,
        after,
        &map,
        Element::Raw(4),
    )
}

/// The output state the PLE convolution writes into `items` PLE entries in
/// place (`ple_conv_row`, its row 0): each entry's nine-row convolution
/// history, copied BF16 rows; the n-gram history and the padding keep their
/// image.
fn conv_state_written(
    label: &str,
    before: &[u8],
    after: &[u8],
    items: usize,
) -> Result<(), String> {
    let map = WriteMap::of(DType::Bf16)
        .byte_ranges((0..items).map(|z| z * ENTRY..z * ENTRY + CONV_BYTES))?;
    written_over(
        &format!("{label} convolution history"),
        before,
        after,
        &map,
        Element::Raw(2),
    )
}

/// The image of an output entry the n-gram and convolution rows write in
/// place, before they run: the `staged` entry with the parts they write
/// without reading (the convolution history, the two history tokens)
/// poisoned; the padding keeps the staged bytes, as the bundle's does.
fn entry_before(staged: &[u8]) -> Vec<u8> {
    let mut entry = staged.to_vec();
    entry[..CONV_BYTES + 8].fill(POISON);
    entry
}

/// The checkpoints a verify launch writes (`ple_checkpoints` in
/// `qwen38_speculation.cuh`): entry z * kVerifyRows + r, whole, for each
/// row r below item z's q_len; the entries past an item's rows stay
/// poisoned.
fn checkpoints_written(label: &str, checkpoints: &[u8], q_len: &[i32]) -> Result<(), String> {
    let entries = q_len
        .iter()
        .enumerate()
        .flat_map(|(z, &rows)| (0..rows.max(0) as usize).map(move |r| z * VERIFY_ROWS + r));
    let map =
        WriteMap::of_records::<16>().byte_ranges(entries.map(|e| e * ENTRY..(e + 1) * ENTRY))?;
    written(
        &format!("{label} checkpoints"),
        checkpoints,
        &map,
        Element::Raw(16),
    )
}

/// The spare entries the PLE anchor writes (`ple_anchor` in
/// `qwen38_speculation.cuh`): entry `slot`, whole; every other spare stays
/// poisoned.
fn anchor_written(label: &str, spares: &[u8], slot: usize) -> Result<(), String> {
    let map = WriteMap::of_records::<16>()
        .byte_ranges(std::iter::once(slot * ENTRY..(slot + 1) * ENTRY))?;
    written(&format!("{label} spares"), spares, &map, Element::Raw(16))
}

/// One verify item: its rows' tokens and PLE convolution input, and its
/// staged PLE entry.
struct Item {
    tokens: Vec<u32>,
    input: Vec<u8>,
    staged: Vec<u8>,
}

/// A staged PLE entry for `seed`: a seeded convolution history, two n-gram
/// history tokens and seeded padding.
fn staged_entry(seed: u64) -> Vec<u8> {
    let mut entry = bf16_bytes(CONV_ROWS * WIDE, mix(seed, 40));
    entry.extend(u32s(&[token(mix(seed, 41)), token(mix(seed, 42))]));
    entry.extend(noise(ENTRY - CONV_BYTES - 8, mix(seed, 43)));
    entry
}

/// `rows` rows of seeded tokens and convolution input for `seed`.
fn seeded_rows(rows: usize, seed: u64) -> (Vec<u32>, Vec<u8>) {
    let tokens = (0..rows as u64).map(|r| token(mix(seed, 50 + r))).collect();
    let input = (0..rows as u64)
        .flat_map(|r| bf16_bytes(WIDE, mix(seed, 1000 + r)))
        .collect();
    (tokens, input)
}

/// An item per seed: item 0 of `seeds.len()` items has that many rows (at
/// most eight), the others one to that many from their seed.
fn verify_items(seeds: &[u64]) -> Vec<Item> {
    let most = seeds.len().min(VERIFY_ROWS);
    seeds
        .iter()
        .enumerate()
        .map(|(index, &seed)| {
            let rows = if index == 0 {
                most
            } else {
                1 + (mix(seed, 44) % most as u64) as usize
            };
            let (tokens, input) = seeded_rows(rows, seed);
            Item {
                tokens,
                input,
                staged: staged_entry(seed),
            }
        })
        .collect()
}

/// A verify entry of the bundle.
#[derive(Clone, Copy)]
enum Verify {
    Ngram,
    Conv,
    Checkpoints,
}

/// One item's results: per row its outputs, then its output state.
struct ItemOut {
    rows: Vec<Vec<u8>>,
    state: Vec<u8>,
}

impl ItemOut {
    fn unit(self) -> Vec<u8> {
        let mut unit = self.rows.concat();
        unit.extend(self.state);
        unit
    }
}

/// One verify launch of `items` through the bundle's `symbol` of `kind`, as
/// the bundle lowers it: item z's rows start at cu_rows[z], its staged entry
/// and output state at entry z. Per item: the n-gram's output and row ids
/// per row and its output history; the convolution's output per row and its
/// output history; or the checkpoint after each row. Output rows have the
/// verify capacity, kVerifyRows per item, as the bundle's; every written
/// buffer is checked after the launch.
fn verify_launch(
    q: &Qwen<'_>,
    kind: Verify,
    symbol: &str,
    items: &[Item],
) -> LabResult<Vec<ItemOut>> {
    let session = q.session;
    let n = items.len();
    let mut cu = Vec::new();
    let mut total = 0;
    for item in items {
        cu.push(total as i32);
        total += item.tokens.len();
    }
    let q_len = items
        .iter()
        .map(|item| item.tokens.len() as i32)
        .collect::<Vec<_>>();
    let tokens = upload(
        session,
        &u32s(
            &items
                .iter()
                .flat_map(|i| i.tokens.clone())
                .collect::<Vec<_>>(),
        ),
    )?;
    let input = upload(
        session,
        &items
            .iter()
            .map(|i| i.input.as_slice())
            .collect::<Vec<_>>()
            .concat(),
    )?;
    let staged = upload(
        session,
        &items
            .iter()
            .map(|i| i.staged.as_slice())
            .collect::<Vec<_>>()
            .concat(),
    )?;
    let staged_out = poisoned(session, n * ENTRY)?;
    let cu_rows = upload(session, &i32s(&cu))?;
    let q_len_buffer = upload(session, &i32s(&q_len))?;
    let live = upload(session, &u32s(&[n as u32]))?;
    let capacity = n * VERIFY_ROWS;
    let rows_of = |bytes: &[u8], width: usize, z: usize| -> Vec<Vec<u8>> {
        (0..q_len[z] as usize)
            .map(|r| bytes[(cu[z] as usize + r) * width..][..width].to_vec())
            .collect()
    };
    match kind {
        Verify::Ngram => {
            let ngram = &q.ngram;
            let (ob, ib) = (ngram.output_bytes(), ngram.id_bytes());
            let output = poisoned(session, capacity * ob)?;
            let row_ids = poisoned(session, capacity * ib)?;
            let table = ngram.bind();
            let at = |name: &str| table.iter().find(|(n, _)| *n == name).unwrap().1;
            q.launch_owned(
                symbol,
                (VERIFY_ROWS as u32, ngram.local as u32, n as u32),
                &[
                    Arg::Buf(&tokens),
                    Arg::At(&staged, CONV_BYTES),
                    at("multipliers"),
                    at("sizes"),
                    at("offsets"),
                    at("table"),
                    at("scale"),
                    Arg::Buf(&output),
                    Arg::Buf(&row_ids),
                    Arg::At(&staged_out, CONV_BYTES),
                    Arg::Buf(&cu_rows),
                    Arg::Buf(&q_len_buffer),
                    Arg::Buf(&live),
                    at("rank"),
                    at("table_start"),
                    at("table_rows"),
                    Arg::U64((ENTRY / 4) as u64),
                ],
            )?;
            let (output, row_ids) = (read(session, &output)?, read(session, &row_ids)?);
            ngram_written(symbol, &output, &row_ids, total, ngram.local)?;
            let states = read(session, &staged_out)?;
            history_written(symbol, &vec![POISON; n * ENTRY], &states, n)?;
            Ok((0..n)
                .map(|z| ItemOut {
                    rows: rows_of(&output, ob, z)
                        .into_iter()
                        .zip(rows_of(&row_ids, ib, z))
                        .map(|(out, ids)| [out, ids].concat())
                        .collect(),
                    state: states[z * ENTRY + CONV_BYTES..][..8].to_vec(),
                })
                .collect())
        }
        Verify::Conv => {
            let output = poisoned(session, capacity * WIDE_BYTES)?;
            q.launch_owned(
                symbol,
                (VERIFY_ROWS as u32, CONV_GROUPS, n as u32),
                &[
                    Arg::Buf(&input),
                    Arg::Buf(&q.weight),
                    Arg::Buf(&staged),
                    Arg::Buf(&output),
                    Arg::Buf(&staged_out),
                    Arg::Buf(&cu_rows),
                    Arg::Buf(&q_len_buffer),
                    Arg::Buf(&live),
                    Arg::U64((ENTRY / 2) as u64),
                ],
            )?;
            let output = read(session, &output)?;
            conv_written(symbol, &output, total)?;
            let states = read(session, &staged_out)?;
            conv_state_written(symbol, &vec![POISON; n * ENTRY], &states, n)?;
            Ok((0..n)
                .map(|z| ItemOut {
                    rows: rows_of(&output, WIDE_BYTES, z),
                    state: states[z * ENTRY..][..CONV_BYTES].to_vec(),
                })
                .collect())
        }
        Verify::Checkpoints => {
            let checkpoints = poisoned(session, capacity * ENTRY)?;
            let checkpoint_ids = upload(session, &i32s(&(0..capacity as i32).collect::<Vec<_>>()))?;
            let mut arguments = vec![
                Arg::Buf(&input),
                Arg::Buf(&tokens),
                Arg::Buf(&staged),
                Arg::Buf(&checkpoints),
                Arg::Buf(&cu_rows),
                Arg::Buf(&q_len_buffer),
                Arg::Buf(&live),
            ];
            if symbol == owned::PLE_CHECKPOINTS_INDEXED {
                arguments.push(Arg::Buf(&checkpoint_ids));
                arguments.push(Arg::U64(capacity as u64));
            }
            q.launch_owned(
                symbol,
                (
                    VERIFY_ROWS as u32,
                    ENTRY.div_ceil(COPY_CHUNK) as u32,
                    n as u32,
                ),
                &arguments,
            )?;
            let all = read(session, &checkpoints)?;
            checkpoints_written(symbol, &all, &q_len)?;
            Ok((0..n)
                .map(|z| ItemOut {
                    rows: (0..q_len[z] as usize)
                        .map(|r| all[(z * VERIFY_ROWS + r) * ENTRY..][..ENTRY].to_vec())
                        .collect(),
                    state: Vec::new(),
                })
                .collect())
        }
    }
}

/// One row of an item run as a one-row decode step: the n-gram's output
/// and row ids, the convolution's output, and the PLE entry after the row.
struct Decoded {
    ngram: Vec<u8>,
    conv: Vec<u8>,
    entry: Vec<u8>,
}

/// `item`'s rows as one-row decode steps of one item, AR's rows: each step
/// stages the PLE entry the previous one left and runs the decode n-gram
/// and PLE convolution on it. What the steps do not write (the padding
/// after the two history tokens) keeps the staged bytes, as the
/// checkpoints keep them; the rest of each step's output entry starts
/// poisoned (`entry_before`), and each launch's writes are checked.
fn decode_rows(q: &Qwen<'_>, item: &Item) -> LabResult<Vec<Decoded>> {
    let session = q.session;
    let ngram_entry = q.ngram_entry("ngram_decode")?;
    let conv_entry = row_entry("chop_row_qwen38_ple_conv_items_decode_v1")?;
    let (ob, ib) = (q.ngram.output_bytes(), q.ngram.id_bytes());
    let q_len = upload(session, &u32s(&[1]))?;
    let mut entry = item.staged.clone();
    let mut rows = Vec::new();
    for (r, &token) in item.tokens.iter().enumerate() {
        let tokens = upload(session, &u32s(&[token]))?;
        let input = upload(session, &item.input[r * WIDE_BYTES..][..WIDE_BYTES])?;
        let staged = upload(session, &entry)?;
        let before = entry_before(&entry);
        let out = upload(session, &before)?;
        let (ngram_out, ids, conv_out) = (
            poisoned(session, ob)?,
            poisoned(session, ib)?,
            poisoned(session, WIDE_BYTES)?,
        );
        let mut bind = q.ngram.bind();
        bind.extend([
            ("tokens", Arg::Buf(&tokens)),
            ("history", Arg::At(&staged, CONV_BYTES)),
            ("output", Arg::Buf(&ngram_out)),
            ("row_ids", Arg::Buf(&ids)),
            ("history_out", Arg::At(&out, CONV_BYTES)),
            ("q_len", Arg::Buf(&q_len)),
            ("entry", Arg::U64((ENTRY / 4) as u64)),
        ]);
        launch_row(session, &q.ngram_row, &ngram_entry, 1, &bind)?;
        let label = format!("{} row {r}", ngram_entry.symbol);
        let (ngram, row_ids) = (read(session, &ngram_out)?, read(session, &ids)?);
        ngram_written(&label, &ngram, &row_ids, 1, q.ngram.local)?;
        let after_ngram = read(session, &out)?;
        history_written(&label, &before, &after_ngram, 1)?;
        launch_row(
            session,
            &q.row,
            &conv_entry,
            1,
            &[
                ("input", Arg::Buf(&input)),
                ("weight", Arg::Buf(&q.weight)),
                ("state", Arg::Buf(&staged)),
                ("output", Arg::Buf(&conv_out)),
                ("state_out", Arg::Buf(&out)),
                ("q_len", Arg::Buf(&q_len)),
                ("entry", Arg::U64((ENTRY / 2) as u64)),
            ],
        )?;
        let label = format!("{} row {r}", conv_entry.symbol);
        let conv = read(session, &conv_out)?;
        conv_written(&label, &conv, 1)?;
        entry = read(session, &out)?;
        conv_state_written(&label, &after_ngram, &entry, 1)?;
        rows.push(Decoded {
            ngram: [ngram, row_ids].concat(),
            conv,
            entry: entry.clone(),
        });
    }
    Ok(rows)
}

fn verify_units(q: &Qwen<'_>, kind: Verify, symbol: &str, seeds: &[u64]) -> LabResult<Units> {
    Ok(verify_launch(q, kind, symbol, &verify_items(seeds))?
        .into_iter()
        .map(ItemOut::unit)
        .collect())
}

/// Each item of `verify_units` through one-row decode steps, its unit laid
/// out as the verify launch's.
fn verify_alone(q: &Qwen<'_>, kind: Verify, seeds: &[u64]) -> LabResult<Units> {
    verify_items(seeds)
        .iter()
        .map(|item| {
            let rows = decode_rows(q, item)?;
            let last = &rows.last().ok_or("an item without rows")?.entry;
            Ok(match kind {
                Verify::Ngram => ItemOut {
                    rows: rows.iter().map(|row| row.ngram.clone()).collect(),
                    state: last[CONV_BYTES..][..8].to_vec(),
                },
                Verify::Conv => ItemOut {
                    rows: rows.iter().map(|row| row.conv.clone()).collect(),
                    state: last[..CONV_BYTES].to_vec(),
                },
                Verify::Checkpoints => ItemOut {
                    rows: rows.iter().map(|row| row.entry.clone()).collect(),
                    state: Vec::new(),
                },
            }
            .unit())
        })
        .collect()
}

/// One item's `rows` rows checkpointed in chunks ending at `ends`, each
/// chunk staged from the previous chunk's last checkpoint; each row's
/// checkpoint.
fn checkpoint_chunks(q: &Qwen<'_>, symbol: &str, rows: u32, ends: &[u32]) -> LabResult<Units> {
    const SEED: u64 = 0xc4ec;
    let (tokens, input) = seeded_rows(rows as usize, SEED);
    let mut staged = staged_entry(SEED);
    let mut units = Vec::new();
    let mut start = 0;
    for &end in ends {
        let (s, e) = (start as usize, end as usize);
        let chunk = Item {
            tokens: tokens[s..e].to_vec(),
            input: input[s * WIDE_BYTES..e * WIDE_BYTES].to_vec(),
            staged,
        };
        let out = verify_launch(q, Verify::Checkpoints, symbol, &[chunk])?.remove(0);
        staged = out.rows.last().ok_or("an empty chunk")?.clone();
        units.extend(out.rows);
        start = end;
    }
    Ok(units)
}

/// A prompt's tokens and PLE convolution input rows.
struct Prompt {
    tokens: Vec<u32>,
    input: Vec<u8>,
}

fn prompt(rows: u32, seed: u64) -> Prompt {
    let (tokens, input) = seeded_rows(rows as usize, seed);
    Prompt { tokens, input }
}

/// A prefill item's results: each row's convolution output, n-gram output
/// and row ids; the entry committed after its last chunk; and, when
/// anchored, the two spare entries of the anchor launch (the anchor writes
/// spare 1).
struct Prefilled {
    rows: Units,
    entry: Vec<u8>,
    spares: Vec<Vec<u8>>,
}

/// The PLE anchor's cubin: the bundle's speculation header with the
/// bundle's TP4 entry spec and symbol (`speculation_spec` in
/// `models/qwen38/bundle/src/sm120/build.rs`), built once per process.
fn anchor_module(session: &Session) -> LabResult<Module> {
    static BUILT: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    let cubin = BUILT
        .get_or_init(build_anchor)
        .as_ref()
        .map_err(Clone::clone)?;
    session.load_cubin(cubin)
}

fn build_anchor() -> Result<Vec<u8>, String> {
    let shape = model_qwen38_contract::sm120::config::profile_shape(
        model_qwen38_contract::sm120::config::PROFILE,
    )
    .ok_or("Qwen3.8's TP4 shapes")?;
    let ptr = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let spec = CubinSpec {
        name: "bundle_qwen38_ple_anchor".into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        include_dirs: vec![
            PathBuf::from("models/qwen38/bundle/cuda"),
            PathBuf::from("crates/kernels-sm120/cuda"),
        ],
        headers: vec!["#include \"qwen38_speculation.cuh\"".into()],
        defines: shape.speculation_defines(),
        entries: vec![EntrySpec::new(
            owned::PLE_ANCHOR.into(),
            vec![
                ptr("input", "const uint16_t"),
                ptr("tokens", "const uint32_t"),
                ptr("staged", "const uint8_t"),
                ptr("anchors", "uint8_t"),
                ptr("cu_rows", "const int32_t"),
                ptr("prefix_len", "const int32_t"),
                ptr("anchor_slots", "const int32_t"),
                ptr("anchor_ends", "const int32_t"),
                ptr("live_items", "const uint32_t"),
            ],
            "chop::qwen38_bundle::ple_anchor(input, tokens, staged, anchors, cu_rows, prefix_len, anchor_slots, anchor_ends, live_items);".into(),
            256,
            1,
        )],
    };
    let out = std::env::temp_dir().join(format!(
        "chopinfer-bundle-invariance-{}",
        std::process::id()
    ));
    let toolchain = kernel_build::Toolchain::detect().map_err(|e| e.to_string())?;
    let built = kernel_build::build(
        &toolchain,
        &[spec],
        &out,
        &out.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .map_err(|e| format!("building {}: {e}", owned::PLE_ANCHOR))?;
    let cubin = built.first().ok_or("no anchor cubin")?;
    std::fs::read(&cubin.path).map_err(|e| format!("{}: {e}", cubin.path.display()))
}

/// `prompt` prefilled in chunks ending at `ends` as the bundle runs one PLE
/// prefill item: each chunk stages its slot's committed entry (fresh at row
/// 0) with the bundle's PLE stage, runs TP4's PLE convolution and n-gram
/// prefill rows on it and commits their output state to the slot. With
/// `anchor`, the chunk that holds it runs the PLE anchor after its rows,
/// into spare 1 of two; the chunk's outputs are read after the anchor ran.
/// Each launch's writes are checked right after it: the staged entry, the
/// rows' outputs, the output entry (`entry_before`) and the spares.
fn prefill(
    q: &Qwen<'_>,
    prompt: &Prompt,
    ends: &[u32],
    anchor: Option<u32>,
) -> LabResult<Prefilled> {
    let session = q.session;
    let conv = row_entry("chop_row_qwen38_ple_conv_prefill_v1")?;
    let ngram = q.ngram_entry("ngram")?;
    let anchor_cubin = anchor.map(|_| anchor_module(session)).transpose()?;
    let (ob, ib) = (q.ngram.output_bytes(), q.ngram.id_bytes());
    let pool = upload(session, &noise(POOL_SLOTS * ENTRY, 0x9003))?;
    let slot_ids = upload(session, &i32s(&[PREFILL_SLOT as i32]))?;
    let live = upload(session, &u32s(&[1]))?;
    let mut prefilled = Prefilled {
        rows: Vec::new(),
        entry: Vec::new(),
        spares: Vec::new(),
    };
    let mut start = 0;
    for &end in ends {
        let (s, e) = (start as usize, end as usize);
        let rows = e - s;
        let prefix_len = upload(session, &i32s(&[start as i32]))?;
        let staging = poisoned(session, ENTRY)?;
        q.launch_owned(
            owned::PLE_STAGE,
            (1, ENTRY.div_ceil(COPY_CHUNK) as u32, 1),
            &[
                Arg::Buf(&pool),
                Arg::Buf(&staging),
                Arg::Buf(&slot_ids),
                Arg::Buf(&prefix_len),
                Arg::Buf(&live),
            ],
        )?;
        let staged = read(session, &staging)?;
        stage_written(owned::PLE_STAGE, &staged, 1)?;
        let before = entry_before(&staged);
        let out = upload(session, &before)?;
        let tokens = upload(session, &u32s(&prompt.tokens[s..e]))?;
        let input = upload(session, &prompt.input[s * WIDE_BYTES..e * WIDE_BYTES])?;
        let live_rows = upload(session, &u32s(&[rows as u32]))?;
        let conv_out = poisoned(session, rows * WIDE_BYTES)?;
        let (ngram_out, ids) = (poisoned(session, rows * ob)?, poisoned(session, rows * ib)?);
        launch_row(
            session,
            &q.row,
            &conv,
            rows as u32,
            &[
                ("input", Arg::Buf(&input)),
                ("weight", Arg::Buf(&q.weight)),
                ("state", Arg::Buf(&staging)),
                ("output", Arg::Buf(&conv_out)),
                ("state_out", Arg::Buf(&out)),
                ("live_rows", Arg::Buf(&live_rows)),
            ],
        )?;
        let label = format!("{} rows {s}..{e}", conv.symbol);
        conv_written(&label, &read(session, &conv_out)?, rows)?;
        let after_conv = read(session, &out)?;
        conv_state_written(&label, &before, &after_conv, 1)?;
        let mut bind = q.ngram.bind();
        bind.extend([
            ("tokens", Arg::Buf(&tokens)),
            ("history", Arg::At(&staging, CONV_BYTES)),
            ("output", Arg::Buf(&ngram_out)),
            ("row_ids", Arg::Buf(&ids)),
            ("history_out", Arg::At(&out, CONV_BYTES)),
            ("live_rows", Arg::Buf(&live_rows)),
        ]);
        launch_row(session, &q.ngram_row, &ngram, rows as u32, &bind)?;
        let label = format!("{} rows {s}..{e}", ngram.symbol);
        let (output, row_ids) = (read(session, &ngram_out)?, read(session, &ids)?);
        ngram_written(&label, &output, &row_ids, rows, q.ngram.local)?;
        history_written(&label, &after_conv, &read(session, &out)?, 1)?;
        if let (Some(at), Some(module)) = (anchor, &anchor_cubin)
            && start < at
            && at <= end
        {
            // The anchor writes spare 1 of two.
            let spare = 1;
            let spares = poisoned(session, 2 * ENTRY)?;
            let column = |value: i32| upload(session, &i32s(&[value]));
            let (cu_rows, slots, anchor_ends) =
                (column(0)?, column(spare as i32)?, column(at as i32)?);
            launch(
                session,
                module,
                owned::PLE_ANCHOR,
                (ENTRY.div_ceil(COPY_CHUNK) as u32, 1, 1),
                256,
                &[
                    Arg::Buf(&input),
                    Arg::Buf(&tokens),
                    Arg::Buf(&staging),
                    Arg::Buf(&spares),
                    Arg::Buf(&cu_rows),
                    Arg::Buf(&prefix_len),
                    Arg::Buf(&slots),
                    Arg::Buf(&anchor_ends),
                    Arg::Buf(&live),
                ],
            )?;
            kernel_lab::launched(owned::PLE_ANCHOR);
            let both = read(session, &spares)?;
            anchor_written(owned::PLE_ANCHOR, &both, spare)?;
            prefilled.spares = both.chunks(ENTRY).map(<[u8]>::to_vec).collect();
        }
        let conv_rows = download(session, &conv_out, 0, rows * WIDE_BYTES)?;
        let ngram_rows = download(session, &ngram_out, 0, rows * ob)?;
        let id_rows = download(session, &ids, 0, rows * ib)?;
        prefilled.rows.extend((0..rows).map(|r| {
            [
                &conv_rows[r * WIDE_BYTES..][..WIDE_BYTES],
                &ngram_rows[r * ob..][..ob],
                &id_rows[r * ib..][..ib],
            ]
            .concat()
        }));
        let committed = download(session, &out, 0, ENTRY)?;
        pool.copy_from_bytes_at(PREFILL_SLOT * ENTRY, &committed, &session.stream)
            .map_err(|e| e.to_string())?;
        start = end;
    }
    prefilled.entry = download(session, &pool, PREFILL_SLOT * ENTRY, ENTRY)?;
    Ok(prefilled)
}

/// The PLE stage's chunk case: each row's outputs, then the committed
/// entry.
fn stage_chunks(q: &Qwen<'_>, rows: u32, ends: &[u32]) -> LabResult<Units> {
    let prefilled = prefill(q, &prompt(rows, 0x57a6), ends, None)?;
    let mut units = prefilled.rows;
    units.push(prefilled.entry);
    Ok(units)
}

/// The PLE anchor's chunk case: a prompt `ANCHOR_TAIL` rows past `rows`,
/// its last chunk (ending at the prompt's end) anchored at row `rows`, a
/// granule multiple. The anchor entry is the unit.
fn anchor_chunks(q: &Qwen<'_>, rows: u32, ends: &[u32]) -> LabResult<Units> {
    let total = rows + ANCHOR_TAIL;
    let mut ends = ends.to_vec();
    *ends.last_mut().ok_or("no chunk")? = total;
    let prefilled = prefill(q, &prompt(total, 0xa7c5), &ends, Some(rows))?;
    let spare = prefilled
        .spares
        .get(1)
        .ok_or_else(|| format!("no chunk of {ends:?} held the anchor {rows}"))?;
    Ok(vec![spare.clone()])
}

/// The PLE anchor against a prefill truncated at the anchor: for each
/// (chunk ends, anchor), the spare entry the anchored step writes must be
/// the entry a prefill whose last chunk ends at the anchor commits (its
/// convolution history, n-gram history and the staged padding); the other
/// spare must stay untouched; and the step's rows and committed entry must
/// be those of the same step without the anchor.
fn anchor_cases(q: &Qwen<'_>) -> LabResult<Vec<Compared>> {
    let granule = contract::PREFILL_GRANULE;
    let cases: [(&[u32], u32, &str); 6] = [
        (&[200], 2 * granule, "a first chunk"),
        (&[128, 300], 4 * granule, "a continued chunk"),
        (&[64, 134], 2 * granule, "a granule into a continued chunk"),
        (
            &[64, 100],
            69,
            "five rows into a chunk, inside the staged history",
        ),
        (&[64, 100], 65, "one row into a chunk"),
        (&[64, 100], 100, "a chunk's last row"),
    ];
    let mut compared = Vec::new();
    for (ends, at, what) in cases {
        let prompt = prompt(*ends.last().ok_or("no chunk")?, 0xa7c4 ^ u64::from(at));
        let anchored = prefill(q, &prompt, ends, Some(at))?;
        let plain = prefill(q, &prompt, ends, None)?;
        let truncated_ends = ends
            .iter()
            .copied()
            .filter(|&end| end < at)
            .chain([at])
            .collect::<Vec<_>>();
        let truncated = prefill(q, &prompt, &truncated_ends, None)?;
        let [other, spare] = <[Vec<u8>; 2]>::try_from(anchored.spares)
            .map_err(|_| format!("{what}: the anchor did not run"))?;
        let label = |part: &str| format!("{what} (anchor {at}, chunks {ends:?}): {part}");
        compared.extend([
            Compared {
                label: label("the anchor entry against the prefill to the anchor"),
                got: spare,
                expected: truncated.entry,
            },
            Compared {
                label: label("the other spare entry"),
                got: other,
                expected: vec![POISON; ENTRY],
            },
            Compared {
                label: label("the step's rows"),
                got: anchored.rows.concat(),
                expected: plain.rows.concat(),
            },
            Compared {
                label: label("the committed entry"),
                got: anchored.entry,
                expected: plain.entry,
            },
        ]);
    }
    Ok(compared)
}

/// The registered n-gram hash entries the catalogs declare
/// (`ModelCatalog::ngram_hashes`), whose bundles embed them in their own
/// cubins (`NgramHashForm::cubin`, loaded through
/// `CHOPINFER_KERNEL_OWNED_CUBINS`).
fn hash_forms() -> &'static [NgramHashForm] {
    static FORMS: OnceLock<Vec<NgramHashForm>> = OnceLock::new();
    FORMS.get_or_init(|| {
        kernel_lab::catalogs()
            .models()
            .iter()
            .flat_map(|model| (model.ngram_hashes)())
            .collect()
    })
}

/// One rank's table shard: the first global row of its heads and its
/// rows.
#[derive(Clone, Copy)]
struct Shard {
    rank: u32,
    start: u64,
    rows: u64,
}

/// One hash form's entries and fixed operands: the form's tables, and
/// every rank's shard of them. Each item's history sits in a PLE entry at
/// the served stride (`ENTRY`), after its convolution history
/// (`CONV_BYTES`), as serving binds it.
struct Hash<'s> {
    session: &'s Session,
    form: &'static NgramHashForm,
    module: Module,
    local: usize,
    shards: Vec<Shard>,
    /// The multipliers, every head's table size and offset.
    tables: [DeviceBuffer; 3],
}

impl<'s> Hash<'s> {
    fn new(session: &'s Session, form: &'static NgramHashForm) -> LabResult<Self> {
        let t = form.table;
        let tables = ngram_oracle::ngram_tables(
            t.vocab,
            t.orders,
            t.heads_per_order,
            t.base,
            t.seed,
            t.layer_index,
            t.alignment,
        )
        .map_err(|e| e.to_string())?;
        let local = form.local_heads as usize;
        // A rank owns the table rows of its heads; every rank's heads are
        // covered, the two-token heads of the first order and the
        // three-token heads of the second.
        let shards = (0..form.ranks)
            .map(|rank| {
                let first = rank as usize * local;
                let start = tables.offsets[first];
                let end = if rank + 1 == form.ranks {
                    tables.rows
                } else {
                    tables.offsets[first + local]
                };
                Shard {
                    rank,
                    start,
                    rows: end - start,
                }
            })
            .collect();
        let words = |values: &[u64]| {
            values
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        Ok(Self {
            session,
            form,
            module: session
                .context
                .load_module(&crate::owned_cubins::load(&form.cubin)?)
                .map_err(|e| e.to_string())?,
            local,
            shards,
            tables: [
                upload(session, &words(&tables.multipliers))?,
                upload(session, &words(&tables.sizes))?,
                upload(session, &words(&tables.offsets))?,
            ],
        })
    }

    /// A token of the form's vocabulary for `seed`: one in thirteen ends a
    /// document.
    fn token(&self, seed: u64) -> u32 {
        let draw = mix(seed, 7);
        if draw % 13 == 0 {
            self.form.eos
        } else {
            ((draw >> 8) % self.form.table.vocab) as u32
        }
    }

    /// `rows` seeded tokens and a seeded two-token history.
    fn item(&self, rows: usize, seed: u64) -> (Vec<u32>, [u32; 2]) {
        let tokens = (0..rows as u64)
            .map(|r| self.token(mix(seed, 50 + r)))
            .collect();
        (
            tokens,
            [self.token(mix(seed, 41)), self.token(mix(seed, 42))],
        )
    }

    /// PLE entries holding `histories`, one per item: seeded bytes around
    /// each item's two history tokens.
    fn entries(histories: &[[u32; 2]]) -> Vec<u8> {
        histories
            .iter()
            .enumerate()
            .flat_map(|(z, history)| {
                let mut entry = noise(ENTRY, mix(z as u64, 45));
                entry[CONV_BYTES..CONV_BYTES + 8].copy_from_slice(&u32s(history));
                entry
            })
            .collect()
    }

    /// Launches `symbol` on `shard` with the tables bound after the tokens
    /// and the entries' history, `ids` and the output entries' history,
    /// then `middle` (the item tables), the shard and `tail`; records the
    /// symbol.
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        symbol: &str,
        shard: Shard,
        grid: (u32, u32, u32),
        tokens: &DeviceBuffer,
        entries: &DeviceBuffer,
        ids: &DeviceBuffer,
        entries_out: &DeviceBuffer,
        middle: &[Arg<'_>],
        tail: &[Arg<'_>],
    ) -> LabResult<()> {
        let [multipliers, sizes, offsets] = &self.tables;
        let mut args = vec![
            Arg::Buf(tokens),
            Arg::At(entries, CONV_BYTES),
            Arg::Buf(multipliers),
            Arg::Buf(sizes),
            Arg::Buf(offsets),
            Arg::Buf(ids),
            Arg::At(entries_out, CONV_BYTES),
        ];
        args.extend_from_slice(middle);
        args.extend([
            Arg::U32(shard.rank),
            Arg::U64(shard.start),
            Arg::U64(shard.rows),
        ]);
        args.extend_from_slice(tail);
        launch(self.session, &self.module, symbol, grid, 256, &args)?;
        kernel_lab::launched(symbol);
        Ok(())
    }

    /// The ids and history a launch over `items` items wrote: the `rows`
    /// live rows' ids, and each item's two history tokens in its entry;
    /// the ids past them and the rest of every entry stay poisoned.
    fn written(
        &self,
        label: &str,
        ids: &[u8],
        entries: &[u8],
        rows: usize,
        items: usize,
    ) -> LabResult<()> {
        let map = WriteMap::of(DType::I64).prefix(rows * self.local)?;
        written(&format!("{label} ids"), ids, &map, Element::Raw(8))?;
        let map = WriteMap::of(DType::I32)
            .byte_ranges((0..items).map(|z| z * ENTRY + CONV_BYTES..z * ENTRY + CONV_BYTES + 8))?;
        written(&format!("{label} history"), entries, &map, Element::Raw(4))
    }

    /// The two history tokens of entry `z`.
    fn history(entries: &[u8], z: usize) -> [u32; 2] {
        let at = z * ENTRY / 4 + CONV_BYTES / 4;
        [word(entries, at), word(entries, at + 1)]
    }

    /// One prefill launch on `shard` over `tokens` after `history`: each
    /// row's ids and the history after the rows.
    fn prefill(
        &self,
        shard: Shard,
        tokens: &[u32],
        history: [u32; 2],
    ) -> LabResult<(Vec<Vec<u8>>, [u32; 2])> {
        let session = self.session;
        let rows = tokens.len();
        let ids = poisoned(session, rows * self.local * 8)?;
        let entries_out = poisoned(session, ENTRY)?;
        let q_len = upload(session, &u32s(&[rows as u32]))?;
        self.launch(
            &self.form.prefill,
            shard,
            (rows as u32, self.local as u32, 1),
            &upload(session, &u32s(tokens))?,
            &upload(session, &Self::entries(&[history]))?,
            &ids,
            &entries_out,
            &[Arg::Buf(&q_len)],
            &[],
        )?;
        let (ids, out) = (read(session, &ids)?, read(session, &entries_out)?);
        self.written(&self.form.prefill, &ids, &out, rows, 1)?;
        let row = self.local * 8;
        Ok((
            ids[..rows * row].chunks(row).map(<[u8]>::to_vec).collect(),
            Self::history(&out, 0),
        ))
    }

    /// One verify launch on `shard` over `items` (tokens, history): per
    /// item, its rows' ids then its history after them.
    fn verify(&self, shard: Shard, items: &[(Vec<u32>, [u32; 2])]) -> LabResult<Units> {
        let session = self.session;
        let n = items.len();
        let mut cu = Vec::new();
        let mut total = 0;
        for (tokens, _) in items {
            cu.push(total as i32);
            total += tokens.len();
        }
        let q_len = items
            .iter()
            .map(|(tokens, _)| tokens.len() as u32)
            .collect::<Vec<_>>();
        let ids = poisoned(session, n * VERIFY_ROWS * self.local * 8)?;
        let entries_out = poisoned(session, n * ENTRY)?;
        let (cu_rows, q_len_buffer, live) = (
            upload(session, &i32s(&cu))?,
            upload(session, &u32s(&q_len))?,
            upload(session, &u32s(&[n as u32]))?,
        );
        self.launch(
            &self.form.verify,
            shard,
            (VERIFY_ROWS as u32, self.local as u32, n as u32),
            &upload(
                session,
                &u32s(
                    &items
                        .iter()
                        .flat_map(|(tokens, _)| tokens.clone())
                        .collect::<Vec<_>>(),
                ),
            )?,
            &upload(
                session,
                &Self::entries(&items.iter().map(|(_, h)| *h).collect::<Vec<_>>()),
            )?,
            &ids,
            &entries_out,
            &[Arg::Buf(&cu_rows), Arg::Buf(&q_len_buffer), Arg::Buf(&live)],
            &[Arg::U64((ENTRY / 4) as u64)],
        )?;
        let (ids, out) = (read(session, &ids)?, read(session, &entries_out)?);
        self.written(&self.form.verify, &ids, &out, total, n)?;
        let row = self.local * 8;
        Ok((0..n)
            .map(|z| {
                let mut unit = ids[cu[z] as usize * row..][..q_len[z] as usize * row].to_vec();
                unit.extend(u32s(&Self::history(&out, z)));
                unit
            })
            .collect())
    }

    /// `tokens` after `history` on `shard` as one-row decode steps of one
    /// item, AR's rows, the history carried from step to step: the rows'
    /// ids, then the history after them.
    fn decode(&self, shard: Shard, tokens: &[u32], mut history: [u32; 2]) -> LabResult<Vec<u8>> {
        let session = self.session;
        let q_len = upload(session, &u32s(&[1]))?;
        let mut unit = Vec::new();
        for (r, &token) in tokens.iter().enumerate() {
            let ids = poisoned(session, self.local * 8)?;
            let entries_out = poisoned(session, ENTRY)?;
            self.launch(
                &self.form.decode,
                shard,
                (1, self.local as u32, 1),
                &upload(session, &u32s(&[token]))?,
                &upload(session, &Self::entries(&[history]))?,
                &ids,
                &entries_out,
                &[Arg::Buf(&q_len)],
                &[Arg::U64((ENTRY / 4) as u64)],
            )?;
            let (ids, out) = (read(session, &ids)?, read(session, &entries_out)?);
            self.written(&format!("{} row {r}", self.form.decode), &ids, &out, 1, 1)?;
            unit.extend(&ids[..self.local * 8]);
            history = Self::history(&out, 0);
        }
        unit.extend(u32s(&history));
        Ok(unit)
    }

    /// An item per seed, as `verify_items` draws their rows.
    fn verify_items(&self, seeds: &[u64]) -> Vec<(Vec<u32>, [u32; 2])> {
        let most = seeds.len().min(VERIFY_ROWS);
        seeds
            .iter()
            .enumerate()
            .map(|(index, &seed)| {
                let rows = if index == 0 {
                    most
                } else {
                    1 + (mix(seed, 44) % most as u64) as usize
                };
                self.item(rows, seed)
            })
            .collect()
    }

    /// Each item's units on every rank: unit `i` is item `i`'s unit of
    /// rank 0, then rank 1's, and so on.
    fn every_rank(&self, units: impl Fn(Shard) -> LabResult<Units>) -> LabResult<Units> {
        let mut all: Option<Units> = None;
        for &shard in &self.shards {
            let units = units(shard)?;
            match all.as_mut() {
                None => all = Some(units),
                Some(all) => {
                    if all.len() != units.len() {
                        return Err(format!("{}: ranks of different units", self.form.cubin));
                    }
                    for (unit, more) in all.iter_mut().zip(units) {
                        unit.extend(more);
                    }
                }
            }
        }
        all.ok_or_else(|| format!("{}: no rank", self.form.cubin))
    }

    /// The verify launch of the items `seeds` draw, on every rank.
    fn verify_units(&self, seeds: &[u64]) -> LabResult<Units> {
        let items = self.verify_items(seeds);
        self.every_rank(|shard| self.verify(shard, &items))
    }

    /// The rows case: the verify launch of the items `seeds` draw, then
    /// item 0 verified alone (one item of up to eight rows: Verify(4) at
    /// four seeds), on every rank.
    fn verify_rows(&self, seeds: &[u64]) -> LabResult<Units> {
        let items = self.verify_items(seeds);
        self.every_rank(|shard| {
            let mut units = self.verify(shard, &items)?;
            units.extend(self.verify(shard, &items[..1])?);
            Ok(units)
        })
    }

    /// The rows case's units through one-row decode steps.
    fn verify_alone(&self, seeds: &[u64]) -> LabResult<Units> {
        let items = self.verify_items(seeds);
        self.every_rank(|shard| {
            items
                .iter()
                .chain(&items[..1])
                .map(|(tokens, history)| self.decode(shard, tokens, *history))
                .collect()
        })
    }
}

/// Word `index` of `bytes`.
fn word(bytes: &[u8], index: usize) -> u32 {
    u32::from_ne_bytes(bytes[index * 4..][..4].try_into().unwrap())
}

/// One prompt of `rows` rows prefilled in chunks ending at `ends`, each
/// chunk after the history the previous one left, on every rank: each
/// row's ids, the last row's followed by the history after the prompt.
fn hash_chunks(hash: &Hash<'_>, rows: u32, ends: &[u32]) -> LabResult<Units> {
    const SEED: u64 = 0x4a54;
    let (tokens, first) = hash.item(rows as usize, SEED);
    hash.every_rank(|shard| {
        let mut history = first;
        let mut units = Vec::new();
        let mut start = 0;
        for &end in ends {
            let (ids, after) =
                hash.prefill(shard, &tokens[start as usize..end as usize], history)?;
            units.extend(ids);
            history = after;
            start = end;
        }
        units
            .last_mut()
            .ok_or("an empty prompt")?
            .extend(u32s(&history));
        Ok(units)
    })
}

/// The ring slots of backup entry `symbol` of the topology whose 32-slot
/// entry is `unsuffixed`: 32 for that entry, `slots` for its `_s{slots}`
/// sibling, `None` for any other symbol.
fn ring_slots(unsuffixed: &str, symbol: &str) -> Option<u64> {
    if symbol == unsuffixed {
        return Some(RING_SLOTS);
    }
    symbol
        .strip_prefix(unsuffixed.strip_suffix("_v1")?)?
        .strip_prefix("_s")?
        .strip_suffix("_v1")?
        .parse()
        .ok()
        .filter(|&slots| slots > 0)
}

/// What a bundle entry is.
#[derive(Clone, Copy)]
enum Kind {
    Ring(Ring),
    Local,
    Stage,
    Verify(Verify, Profile),
    Anchor,
    /// A registered hash entry: its form, and whether it is the verify
    /// entry (else the prefill entry).
    Hash(usize, bool),
}

fn kind(symbol: &str) -> LabResult<Kind> {
    if let Some(ring) = Tp::SUPPORTED.into_iter().find_map(|tp| {
        ring_slots(&mimo::bundle_symbol(tp, "ring_backup"), symbol)
            .map(|slots| Ring::new(tp, slots))
    }) {
        return Ok(Kind::Ring(ring));
    }
    if let Some((index, verify)) = hash_forms().iter().enumerate().find_map(|(index, form)| {
        (form.verify == symbol || form.prefill == symbol).then_some((index, form.verify == symbol))
    }) {
        return Ok(Kind::Hash(index, verify));
    }
    let tp4 = model_qwen38_contract::sm120::config::PROFILE;
    Ok(match symbol {
        owned::PLE_RESIDUAL => Kind::Local,
        owned::PLE_STAGE => Kind::Stage,
        owned::PLE_CHECKPOINTS | owned::PLE_CHECKPOINTS_INDEXED => {
            Kind::Verify(Verify::Checkpoints, tp4)
        }
        owned::PLE_CONV_VERIFY => Kind::Verify(Verify::Conv, tp4),
        owned::PLE_ANCHOR => Kind::Anchor,
        _ => {
            let profile = model_qwen38_contract::sm120::PROFILES
                .into_iter()
                .find(|&profile| owned::per_rank(owned::NGRAM_VERIFY, profile.tag()) == symbol)
                .ok_or_else(|| format!("{symbol}: no bundle adapter"))?;
            Kind::Verify(Verify::Ngram, profile)
        }
    })
}

#[derive(Default)]
pub(crate) struct Bundle {
    /// Ring batch launches so far: the harness launches each batch case
    /// twice, and the second of each pair scrambles the other slots.
    ring_batches: u64,
}

impl Adapter for Bundle {
    fn kind(&self) -> OpKind {
        OpKind::Bundle
    }

    fn prompt_rows(&self, entry: &Declared) -> u32 {
        match kind(&entry.symbol) {
            Ok(Kind::Ring(_)) => RING_TOKENS as u32,
            Ok(Kind::Verify(..) | Kind::Hash(_, true)) => VERIFY_ROWS as u32,
            _ => PROMPT,
        }
    }

    fn chunks(
        &mut self,
        sessions: &[Session],
        entry: &Declared,
        rows: u32,
        ends: &[u32],
    ) -> LabResult<Units> {
        let session = &sessions[0];
        let symbol = &entry.symbol;
        match kind(symbol)? {
            Kind::Ring(ring) => ring_chunks(session, ring, symbol, rows, ends),
            Kind::Local => local_chunks(
                &Qwen::new(session, model_qwen38_contract::sm120::config::PROFILE)?,
                symbol,
                ends,
            ),
            Kind::Stage => stage_chunks(
                &Qwen::new(session, model_qwen38_contract::sm120::config::PROFILE)?,
                rows,
                ends,
            ),
            Kind::Anchor => anchor_chunks(
                &Qwen::new(session, model_qwen38_contract::sm120::config::PROFILE)?,
                rows,
                ends,
            ),
            Kind::Verify(Verify::Checkpoints, profile) => {
                checkpoint_chunks(&Qwen::new(session, profile)?, symbol, rows, ends)
            }
            Kind::Hash(form, false) => {
                hash_chunks(&Hash::new(session, &hash_forms()[form])?, rows, ends)
            }
            Kind::Verify(..) | Kind::Hash(_, true) => {
                Err(format!("{symbol}: verify rows have no chunk case"))
            }
        }
    }

    fn units(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let session = &sessions[0];
        let symbol = &entry.symbol;
        match kind(symbol)? {
            Kind::Ring(ring) => ring_launches(session, ring, symbol, seeds, false, false),
            Kind::Local => local_rows(
                &Qwen::new(session, model_qwen38_contract::sm120::config::PROFILE)?,
                symbol,
                seeds,
            ),
            Kind::Stage => stage_units(
                &Qwen::new(session, model_qwen38_contract::sm120::config::PROFILE)?,
                seeds,
            ),
            Kind::Verify(verify, profile) => {
                verify_units(&Qwen::new(session, profile)?, verify, symbol, seeds)
            }
            Kind::Anchor => Err(format!("{symbol}: the anchor has no unit case")),
            Kind::Hash(form, true) => Hash::new(session, &hash_forms()[form])?.verify_units(seeds),
            Kind::Hash(_, false) => Err(format!("{symbol}: the prefill hash has no unit case")),
        }
    }

    /// A ring's rows cases add 128 and 256 items where its slots hold them.
    fn row_counts(&self, entry: &Declared, max: u32) -> Vec<u32> {
        let mut counts = row_counts(max);
        if let Ok(Kind::Ring(ring)) = kind(&entry.symbol) {
            counts.extend(
                [128, 256]
                    .into_iter()
                    .filter(|&n| n <= max && u64::from(n) <= ring.slots),
            );
        }
        counts
    }

    fn batch(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        match kind(&entry.symbol)? {
            Kind::Ring(ring) => {
                let scramble = self.ring_batches % 2 == 1;
                self.ring_batches += 1;
                ring_launches(&sessions[0], ring, &entry.symbol, seeds, scramble, false)
            }
            _ => self.units(sessions, entry, seeds),
        }
    }

    fn rows(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        match kind(&entry.symbol)? {
            Kind::Hash(form, true) => {
                Hash::new(&sessions[0], &hash_forms()[form])?.verify_rows(seeds)
            }
            _ => self.units(sessions, entry, seeds),
        }
    }

    fn alone(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let session = &sessions[0];
        match kind(&entry.symbol)? {
            Kind::Ring(ring) => ring_alone(session, ring, &entry.symbol, seeds, false),
            Kind::Verify(verify, profile) => {
                verify_alone(&Qwen::new(session, profile)?, verify, seeds)
            }
            Kind::Hash(form, true) => Hash::new(session, &hash_forms()[form])?.verify_alone(seeds),
            _ => seeds
                .iter()
                .map(|&seed| {
                    self.units(sessions, entry, &[seed])?
                        .pop()
                        .ok_or_else(|| format!("{}: no unit", entry.symbol))
                })
                .collect(),
        }
    }

    fn anchors(&mut self, sessions: &[Session], entry: &Declared) -> LabResult<Vec<Compared>> {
        match kind(&entry.symbol)? {
            Kind::Anchor => anchor_cases(&Qwen::new(
                &sessions[0],
                model_qwen38_contract::sm120::config::PROFILE,
            )?),
            _ => Err(format!("{}: no anchor adapter", entry.symbol)),
        }
    }
}

/// Every bundle entry the catalogs and bundles declare has a case here (no
/// GPU), the registered hash entries among them.
#[test]
fn every_bundle_declaration_has_a_case() {
    let bundle = inventory()
        .into_iter()
        .filter(|declared| declared.op == OpKind::Bundle)
        .collect::<Vec<_>>();
    for declared in &bundle {
        assert!(kind(&declared.symbol).is_ok(), "{}", declared.symbol);
    }
    for form in hash_forms() {
        for symbol in [&form.prefill, &form.verify] {
            assert!(
                matches!(kind(symbol), Ok(Kind::Hash(..))),
                "{symbol}: a hash case"
            );
        }
    }
}

/// The write checks the adapter runs after every launch, on synthetic bytes
/// (no GPU): each rejects an unwritten buffer, a hole in its written region
/// and a write outside it (rows past the live ones, another item's or spare
/// entry, padding, an untouched ring cell), rejects NaN in arithmetic
/// outputs, and accepts a correct image.
#[test]
fn bundle_write_checks_reject_unwritten_outputs() {
    let patched = |image: &[u8], at: usize, bytes: &[u8]| {
        let mut image = image.to_vec();
        image[at..at + bytes.len()].copy_from_slice(bytes);
        image
    };
    let one = 0x3f80_u16.to_ne_bytes();
    let nan = 0x7fc0_u16.to_ne_bytes();
    let hole = [POISON; 16];

    // The row-local PLE residual: three live rows of four planned.
    let rows = 3;
    let good = patched(&vec![POISON; 4 * WIDE_BYTES], 0, &one.repeat(rows * WIDE));
    let check = |out: &[u8]| local_written(owned::PLE_RESIDUAL, out, rows);
    assert!(check(&good).is_ok());
    assert!(check(&vec![POISON; good.len()]).is_err());
    assert!(check(&patched(&good, WIDE_BYTES + 6, &hole[..2])).is_err());
    assert!(check(&patched(&good, rows * WIDE_BYTES, &one)).is_err());
    assert!(check(&patched(&good, 4, &nan)).is_err());

    // The PLE stage: two live items of four planned.
    let good = patched(&vec![POISON; 4 * ENTRY], 0, &vec![0; 2 * ENTRY]);
    let check = |staging: &[u8]| stage_written("stage", staging, 2);
    assert!(check(&good).is_ok());
    assert!(check(&vec![POISON; good.len()]).is_err());
    assert!(check(&patched(&good, ENTRY + 4096, &hole)).is_err());
    assert!(check(&patched(&good, 2 * ENTRY, &[0])).is_err());

    // N-gram rows of four local heads: three live rows of a capacity of five.
    let local = 4;
    let (ob, ib) = (local * 320, local * 8);
    let output = patched(&vec![POISON; 5 * ob], 0, &one.repeat(3 * ob / 2));
    let ids = patched(
        &vec![POISON; 5 * ib],
        0,
        &7_u64.to_ne_bytes().repeat(3 * local),
    );
    let check = |output: &[u8], ids: &[u8]| ngram_written("ngram", output, ids, 3, local);
    assert!(check(&output, &ids).is_ok());
    assert!(check(&vec![POISON; output.len()], &ids).is_err());
    assert!(check(&output, &vec![POISON; ids.len()]).is_err());
    assert!(check(&patched(&output, ob + 2, &hole[..2]), &ids).is_err());
    assert!(check(&output, &patched(&ids, ib + 8, &hole[..8])).is_err());
    assert!(check(&patched(&output, 2 * ob, &nan), &ids).is_err());
    assert!(check(&patched(&output, 3 * ob, &one), &ids).is_err());
    assert!(check(&output, &patched(&ids, 3 * ib, &[0; 8])).is_err());

    // Convolution rows: two live rows of a capacity of four.
    let output = patched(&vec![POISON; 4 * WIDE_BYTES], 0, &one.repeat(2 * WIDE));
    let check = |output: &[u8]| conv_written("conv", output, 2);
    assert!(check(&output).is_ok());
    assert!(check(&vec![POISON; output.len()]).is_err());
    assert!(check(&patched(&output, WIDE_BYTES, &hole[..2])).is_err());
    assert!(check(&patched(&output, 6, &nan)).is_err());
    assert!(check(&patched(&output, 2 * WIDE_BYTES, &one)).is_err());

    // Verify output states of three items, all poisoned before the launch.
    let items = 3;
    let poisoned = vec![POISON; items * ENTRY];
    let tokens = u32s(&[EOS, 17]);
    let (mut history, mut conv_state) = (poisoned.clone(), poisoned.clone());
    for z in 0..items {
        history = patched(&history, z * ENTRY + CONV_BYTES, &tokens);
        conv_state = patched(&conv_state, z * ENTRY, &one.repeat(CONV_BYTES / 2));
    }
    let check = |after: &[u8]| history_written("history", &poisoned, after, items);
    assert!(check(&history).is_ok());
    assert!(check(&poisoned).is_err());
    assert!(check(&patched(&history, ENTRY + CONV_BYTES + 4, &hole[..4])).is_err());
    assert!(check(&patched(&history, CONV_BYTES + 8, &[0])).is_err());
    assert!(check(&patched(&history, 0, &one)).is_err());
    let check = |after: &[u8]| conv_state_written("conv state", &poisoned, after, items);
    assert!(check(&conv_state).is_ok());
    assert!(check(&poisoned).is_err());
    assert!(check(&patched(&conv_state, 3 * ENTRY - 128 - 2, &hole[..2])).is_err());
    assert!(check(&patched(&conv_state, CONV_BYTES, &[0; 4])).is_err());

    // A decode step's or prefill chunk's output entry: the staged padding
    // kept, the n-gram history then the convolution history written.
    let staged = staged_entry(0x5eed);
    let before = entry_before(&staged);
    let after_ngram = patched(&before, CONV_BYTES, &tokens);
    let after = patched(&after_ngram, 0, &staged[..CONV_BYTES]);
    assert!(history_written("entry", &before, &after_ngram, 1).is_ok());
    assert!(conv_state_written("entry", &after_ngram, &after, 1).is_ok());
    assert!(history_written("entry", &before, &before, 1).is_err());
    assert!(conv_state_written("entry", &after_ngram, &after_ngram, 1).is_err());
    let padding = CONV_BYTES + 8;
    let clobbered = patched(&after_ngram, padding, &[!staged[padding]]);
    assert!(history_written("entry", &before, &clobbered, 1).is_err());
    // An output entry that already held the history proves nothing.
    let unpoisoned = patched(&before, CONV_BYTES, &staged[CONV_BYTES..padding]);
    assert!(history_written("entry", &unpoisoned, &after_ngram, 1).is_err());

    // Checkpoints of items of two rows and one row.
    let q_len = [2, 1];
    let mut checkpoints = vec![POISON; 2 * VERIFY_ROWS * ENTRY];
    for entry in [0, 1, VERIFY_ROWS] {
        checkpoints = patched(&checkpoints, entry * ENTRY, &staged);
    }
    let check = |checkpoints: &[u8]| checkpoints_written("verify", checkpoints, &q_len);
    assert!(check(&checkpoints).is_ok());
    assert!(check(&vec![POISON; checkpoints.len()]).is_err());
    assert!(check(&patched(&checkpoints, VERIFY_ROWS * ENTRY + 4096, &hole)).is_err());
    assert!(check(&patched(&checkpoints, 2 * ENTRY, &staged[..16])).is_err());
    assert!(
        check(&patched(
            &checkpoints,
            (VERIFY_ROWS + 1) * ENTRY,
            &staged[..16]
        ))
        .is_err()
    );

    // The anchor's spares: spare 1 written, spare 0 untouched.
    let spares = patched(&vec![POISON; 2 * ENTRY], ENTRY, &staged);
    assert!(anchor_written("anchor", &spares, 1).is_ok());
    assert!(anchor_written("anchor", &vec![POISON; spares.len()], 1).is_err());
    assert!(anchor_written("anchor", &patched(&spares, 0, &staged[..16]), 1).is_err());
    assert!(anchor_written("anchor", &patched(&spares, 2 * ENTRY - 16, &hole), 1).is_err());
    assert!(anchor_written("anchor", &spares, 0).is_err());

    // A ring backup of two items on a ring of small entries: slot 3's
    // positions 126 and 127, and slot 10's position 0.
    let ring = Ring {
        tp: Tp::TP2,
        slots: RING_SLOTS,
        k_entry: 16,
        v_entry: 32,
    };
    let items: [RingItem; 2] = [(3, 2, 126), (10, 1, 0)];
    let poisoned = vec![POISON; ring.working() as usize];
    let mut shadow = poisoned.clone();
    for item in items {
        for position in ring_touched(item, RING_PREFILL_SPAN) {
            for range in ring.cell_ranges(item.0, position) {
                shadow[range].fill(0x5a);
            }
        }
    }
    let check =
        |after: &[u8]| ring.shadow_written("ring", &poisoned, after, &items, RING_PREFILL_SPAN);
    assert!(check(&shadow).is_ok());
    assert!(check(&poisoned).is_err());
    // Layer 7's V entry of slot 10's position 0 left unwritten.
    let v = ring.cell_ranges(10, 0)[2 * 7 + 1].clone();
    assert!(check(&patched(&shadow, v.start, &hole)).is_err());
    // Slot 3's position 125, which no item touches, and another slot's.
    let untouched = ring.cell_ranges(3, 125)[0].start;
    assert!(check(&patched(&shadow, untouched, &[0])).is_err());
    let other = ring.cell_ranges(4, 0)[5].start;
    assert!(check(&patched(&shadow, other, &[0])).is_err());
    // Slot 10 alone through its image: the same cells, and a hole fails.
    let image = ring
        .slot_ranges(10)
        .into_iter()
        .flat_map(|(at, len)| shadow[at - ring.working() as usize..][..len].to_vec())
        .collect::<Vec<_>>();
    assert_eq!(image.len(), ring.slot_bytes());
    let cell = ring
        .slot_cell_ranges(0)
        .into_iter()
        .flat_map(|range| image[range].to_vec())
        .collect::<Vec<_>>();
    assert_eq!(cell, ring.cell(&shadow, 10, 0));
    assert!(
        ring.slot_written("slot", &image, items[1], RING_PREFILL_SPAN)
            .is_ok()
    );
    let v = ring.slot_cell_ranges(0)[2 * 7 + 1].clone();
    assert!(
        ring.slot_written("slot", &patched(&image, v.start, &hole), items[1], 1)
            .is_err()
    );
    // A decode span saves the last position only, a verify span the last 8.
    assert_eq!(ring_touched((3, 2, 126), 1), 127..128);
    assert_eq!(ring_touched((3, 20, 126), MAX_VERIFY_ROWS), 138..146);
}

/// Every bundle declaration has a case, every ring backup entry's slot
/// count comes from its symbol, and the items of a launch touch disjoint
/// ring cells at every slot count, up to the window's positions per slot
/// (no GPU).
#[test]
fn ring_cases_fit_every_slot_count() {
    for declared in crate::invariance::inventory() {
        if declared.op == OpKind::Bundle {
            assert!(kind(&declared.symbol).is_ok(), "{}", declared.symbol);
        }
    }
    let unsuffixed = "chop_x_ring_backup_v1";
    assert_eq!(ring_slots(unsuffixed, unsuffixed), Some(RING_SLOTS));
    assert_eq!(ring_slots(unsuffixed, "chop_x_ring_backup_s8_v1"), Some(8));
    for other in [
        "chop_x_ring_backup_s0_v1",
        "chop_x_ring_backup_sx_v1",
        "chop_x_ring_restore_s8_v1",
        "chop_y_ring_backup_s8_v1",
    ] {
        assert_eq!(ring_slots(unsuffixed, other), None, "{other}");
    }
    for slots in [1, 2, 8, 32, 256] {
        for count in [1, 2, 5, 64, 128] {
            let items = (0..count)
                .map(|index| ring_item(index, count, slots, mix(index as u64, 7)))
                .collect::<LabResult<Vec<_>>>()
                .unwrap();
            let mut cells = std::collections::BTreeSet::new();
            for item in items {
                assert!((0..slots as i32).contains(&item.0));
                for position in ring_touched(item, RING_PREFILL_SPAN) {
                    let cell = (item.0, position.rem_euclid(RING_TOKENS as i32));
                    assert!(cells.insert(cell), "{slots} slots, {count} items: {cell:?}");
                }
            }
        }
    }
    assert!(ring_item(0, 129, 1, 0).is_err());
}

/// A host shadow for the alone routine: each launch writes its item's
/// cells and, for `stray`'s item, 16 bytes at `stray`'s offset too.
struct HostShadow {
    ring: Ring,
    shadow: std::cell::RefCell<Vec<u8>>,
    stray: Option<(RingItem, usize)>,
}

impl AloneShadow for HostShadow {
    fn ring(&self) -> Ring {
        self.ring
    }

    fn launch(&self, item: RingItem, span: u32) -> LabResult<()> {
        let mut shadow = self.shadow.borrow_mut();
        for position in ring_touched(item, span) {
            for range in self.ring.cell_ranges(item.0, position) {
                shadow[range].fill(position as u8 | 1);
            }
        }
        if let Some((faulty, at)) = self.stray
            && faulty == item
        {
            shadow[at..at + 16].fill(0x22);
        }
        Ok(())
    }

    fn slot_image(&self, slot: i32) -> LabResult<Vec<u8>> {
        let (shadow, working) = (self.shadow.borrow(), self.ring.working() as usize);
        Ok(self
            .ring
            .slot_ranges(slot)
            .into_iter()
            .flat_map(|(at, len)| shadow[at - working..][..len].to_vec())
            .collect())
    }

    fn reset_slot(&self, slot: i32) -> LabResult<()> {
        let (mut shadow, working) = (self.shadow.borrow_mut(), self.ring.working() as usize);
        for (at, len) in self.ring.slot_ranges(slot) {
            shadow[at - working..][..len].fill(POISON);
        }
        Ok(())
    }

    fn image(&self) -> LabResult<Vec<u8>> {
        Ok(self.shadow.borrow().clone())
    }
}

/// An alone launch that also writes a cell of another slot fails even when
/// a later launch writes that cell and resets its slot (no GPU): the
/// reviewer's `_s8` case, where item 0 (slot 3) also writes 16 bytes of
/// slot 2's position 2891, which item 1 then backs up, at every span.
#[test]
fn an_alone_write_into_a_later_items_cell_is_refused() {
    let ring = Ring {
        tp: Tp::TP2,
        slots: 8,
        k_entry: 16,
        v_entry: 32,
    };
    let items: [RingItem; 2] = [(3, 194, 2091), (2, 1, 2891)];
    let stray = ring.cell_ranges(2, 2891)[0].start;
    for span in RING_SPANS {
        let run = |stray| {
            let host = HostShadow {
                ring,
                shadow: std::cell::RefCell::new(vec![POISON; ring.working() as usize]),
                stray,
            };
            ring_alone_span(&host, "host", &items, span, &mut |_| Ok(()))
        };
        let units = run(None).unwrap();
        assert_eq!(units.len(), 2);
        let Err(error) = run(Some((items[0], stray))) else {
            panic!("span {span}: a write into slot 2's cell before its launch passed");
        };
        assert!(
            error.contains("slot 2") && error.contains("before its launch"),
            "{error}"
        );
    }
    // A stray into a slot no later item launches fails the whole shadow.
    let other = ring.cell_ranges(6, 0)[0].start;
    let host = HostShadow {
        ring,
        shadow: std::cell::RefCell::new(vec![POISON; ring.working() as usize]),
        stray: Some((items[0], other)),
    };
    let Err(error) = ring_alone_span(&host, "host", &items, 1, &mut |_| Ok(())) else {
        panic!("a write into slot 6 passed");
    };
    assert!(error.contains("after its items alone"), "{error}");
}

/// The negative control of the ring cases: a backup that reads the next
/// slot's working cells (the bundle's entry with every saved cell then
/// overwritten from that slot) writes every cell the write map expects, so
/// only the comparisons can refuse it. The batch case's scrambled launch and
/// the rows case's items alone must each differ for it, and compare equal
/// for the entry itself.
#[test]
fn a_backup_reading_another_slot_fails_the_ring_cases_gpu() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let (ring, symbol) = inventory()
            .into_iter()
            .find_map(|declared| match kind(&declared.symbol) {
                Ok(Kind::Ring(ring)) if ring.slots == 8 && ring.tp.ranks() == 4 => {
                    Some((ring, declared.symbol))
                }
                _ => None,
            })
            .ok_or("no 8-slot ring backup at four ranks")?;
        let seeds = [11, 12, 13, 14];
        let even = |units: &Units| units.iter().step_by(2).cloned().collect::<Units>();
        for wrong in [false, true] {
            let kept = ring_launches(session, ring, &symbol, &seeds, false, wrong)?;
            let scrambled = ring_launches(session, ring, &symbol, &seeds, true, wrong)?;
            let alone = ring_alone(session, ring, &symbol, &seeds, wrong)?;
            let batch = even(&kept) == even(&scrambled);
            let rows = kept == alone;
            eprintln!(
                "{symbol} reading the next slot {wrong}: batch equal {batch}, rows equal {rows}"
            );
            if (batch, rows) != (!wrong, !wrong) {
                return Err(format!(
                    "{symbol} reading the next slot {wrong}: batch equal {batch}, rows equal {rows}"
                ));
            }
        }
        Ok(())
    })
    .expect("ring negative control");
}
