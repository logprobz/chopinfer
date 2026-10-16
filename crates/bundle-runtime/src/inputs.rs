//! Validated device input arrays for one mixed append step.

use std::collections::{BTreeMap, BTreeSet};

use runtime_bundle_abi::BundleStatusV1;
use runtime_bundle_sdk::{BatchRequestV1, BatchWireVersionV1};

use crate::geometry::{BundleGeometry, Items, ModelLimits};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputError {
    InvalidBatch,
    InvalidAppend,
    InvalidToken,
    /// The item's request has no sequence slot.
    MissingSlot,
    /// A verify item this instance was not created to run: speculation is
    /// off, the item is bundle-drafted, or it exceeds the enabled limits.
    UnsupportedVerify,
}

impl std::fmt::Display for InputError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidBatch => "invalid batch",
            Self::InvalidAppend => "invalid KV append",
            Self::InvalidToken => "input token outside the vocabulary",
            Self::MissingSlot => "request without a sequence slot",
            Self::UnsupportedVerify => "verify outside the enabled limits",
        })
    }
}

impl InputError {
    /// The status a step refused with this error reports: a request without
    /// a slot is not found, a verify the instance does not run is
    /// unsupported, anything else an invalid argument.
    pub fn status(self) -> BundleStatusV1 {
        match self {
            Self::MissingSlot => BundleStatusV1::NotFound,
            Self::UnsupportedVerify => BundleStatusV1::Unsupported,
            _ => BundleStatusV1::InvalidArgument,
        }
    }
}

/// A batch's size set against the limits packing checks.
///
/// A bundle has no channel to tell its host why it refused a batch, so it
/// logs this with the refusal: the numbers show which limit was crossed.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchShape {
    pub items: usize,
    pub prefill_items: usize,
    pub prefill_tokens: u64,
    pub largest_prefill: u64,
    pub decode_items: usize,
    pub verify_items: usize,
    /// Decode-phase rows: one per decode item, every row of a verify item.
    pub decode_rows: u64,
    pub largest_verify: u64,
    pub media_encodes: usize,
    pub media_rows: u64,
    pub injected_rows: u64,
}

impl BatchShape {
    #[must_use]
    pub fn of(request: &BatchRequestV1) -> Self {
        let media = request.media();
        let mut shape = Self {
            items: request.items().len(),
            media_encodes: media.encodes.len(),
            media_rows: media
                .encodes
                .iter()
                .map(|encode| u64::from(encode.output_rows))
                .sum(),
            injected_rows: media
                .injects
                .iter()
                .map(|inject| u64::from(inject.rows))
                .sum(),
            ..Self::default()
        };
        for item in request.items() {
            let tokens = u64::from(item.phase_tokens());
            if item.is_verify() {
                shape.verify_items += 1;
                shape.largest_verify = shape.largest_verify.max(tokens);
                shape.decode_rows += tokens;
            } else if item.is_decode() {
                shape.decode_items += 1;
                shape.decode_rows += tokens;
            } else {
                shape.prefill_items += 1;
                shape.largest_prefill = shape.largest_prefill.max(tokens);
                shape.prefill_tokens += tokens;
            }
        }
        shape
    }

    /// The shape with `G`'s limits and the verify rows the host enabled.
    #[must_use]
    pub fn describe<G: BundleGeometry>(&self, verify: VerifyLimits) -> String {
        self.describe_within::<G>(verify, Items::package::<G>())
    }

    /// [`Self::describe`] for an instance serving `items`.
    #[must_use]
    pub fn describe_within<G: BundleGeometry>(&self, verify: VerifyLimits, items: Items) -> String {
        let live = if self.verify_items > 0 {
            u32::try_from(self.decode_rows).unwrap_or(u32::MAX)
        } else {
            u32::try_from(self.decode_items).unwrap_or(u32::MAX)
        };
        // Verify steps bucket their rows by the verify tokens, decode steps
        // their items by the served count.
        let bucket = if self.verify_items > 0 {
            crate::graphs::verify_bucket_within::<G>(live, items)
        } else {
            items.bucket(live)
        };
        let bucket = match bucket {
            Some(bucket) => bucket.to_string(),
            None if live == 0 => "none".to_owned(),
            None => "over the largest".to_owned(),
        };
        format!(
            "{} items (at most {}); prefill: {} items, {} tokens (at most {}), largest {}; \
             decode: {} items, {} verify items, {} rows (verify at most {} rows, {} per item), \
             largest verify {}, graph bucket {}; media: {} encodes, {} rows, {} injected rows",
            self.items,
            items.get(),
            self.prefill_items,
            self.prefill_tokens,
            G::PREFILL_TOKENS,
            self.largest_prefill,
            self.decode_items,
            self.verify_items,
            self.decode_rows,
            verify.rows_per_step,
            verify.rows_per_item,
            self.largest_verify,
            bucket,
            self.media_encodes,
            self.media_rows,
            self.injected_rows,
        )
    }
}

/// The verify rows the host enabled at create. Zero rows disables verify.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifyLimits {
    /// Most rows one verify item carries: its drafts plus one.
    pub rows_per_item: u32,
    /// Most decode-phase rows one step carries.
    pub rows_per_step: u32,
    /// This instance loaded a supported drafter.
    pub bundle_drafter: bool,
}

impl VerifyLimits {
    pub const NONE: Self = Self {
        rows_per_item: 0,
        rows_per_step: 0,
        bundle_drafter: false,
    };

    /// Clamps the host's enabled limits to what this bundle runs.
    pub fn enabled<G: BundleGeometry>(
        maximum_draft_tokens: u32,
        maximum_verify_tokens: u32,
    ) -> Self {
        Self::enabled_within::<G>(
            maximum_draft_tokens,
            maximum_verify_tokens,
            Items::package::<G>(),
        )
    }

    /// Clamps the host's enabled limits to what an instance serving `items`
    /// runs.
    pub fn enabled_within<G: BundleGeometry>(
        maximum_draft_tokens: u32,
        maximum_verify_tokens: u32,
        items: Items,
    ) -> Self {
        if maximum_draft_tokens == 0 || maximum_verify_tokens == 0 {
            return Self::NONE;
        }
        Self {
            rows_per_item: maximum_draft_tokens
                .saturating_add(1)
                .min(G::MAX_VERIFY_ROWS),
            rows_per_step: maximum_verify_tokens.min(crate::graphs::verify_tokens::<G>(items)),
            bundle_drafter: false,
        }
    }
}

/// One phase is packed independently because attention has separate decode and
/// prefill entry points. `item_order` restores the original output row order.
#[derive(Debug, Default)]
pub struct PhaseInputs {
    pub decode: bool,
    /// The decode phase runs as verify rows: a step with any verify item
    /// carries its one-token decode items as one-row verify items.
    pub verify: bool,
    /// Items whose every row emits logits: the step's verify items.
    pub verify_rows: Vec<bool>,
    pub item_order: Vec<usize>,
    pub request_ids: Vec<u64>,
    pub emits_row: Vec<bool>,
    pub prompt_rows: Vec<bool>,
    /// Prompt-ending prefill items that keep their state at the last prefill
    /// granule multiple inside their rows (`BatchItemV1::anchor`).
    pub anchors: Vec<bool>,
    pub token_ids: Vec<u32>,
    pub positions: Vec<i32>,
    /// Explicit (T, H, W) RoPE positions, one per token, when any item of
    /// the phase carries them; items without them repeat their 1D position.
    /// Empty when no item does.
    pub position_axes: Vec<[i32; 3]>,
    pub token_batch: Vec<i32>,
    /// Each item's page table, unpadded; rows of the device table are
    /// `ModelLimits::max_pages` entries apart and padded with -1.
    pub block_tables: Vec<Vec<i32>>,
    pub slot_ids: Vec<i32>,
    pub q_start: Vec<i32>,
    pub q_len: Vec<i32>,
    pub prefix_len: Vec<i32>,
    pub seq_lens: Vec<i32>,
    pub last_positions: Vec<i64>,
}

impl PhaseInputs {
    pub fn live_batch(&self) -> u32 {
        self.item_order.len() as u32
    }

    pub fn live_tokens(&self) -> u32 {
        self.token_ids.len() as u32
    }

    /// A tap step's `items.json` (`runtime_bundle_sdk::taps`): the program,
    /// its rows and each item's request, rows, prefix, sequence length and
    /// token IDs, so a request's rows can be found in a packed prefill, and
    /// the copied rows `[first, end)` with the first kept position.
    pub fn tap_items(
        &self,
        program: &str,
        rows: u32,
        (first, end): (u32, u32),
        from: Option<i32>,
    ) -> serde_json::Value {
        let items = (0..self.q_start.len())
            .map(|item| {
                let start = self.q_start[item].max(0) as usize;
                let len = self.q_len[item].max(0) as usize;
                serde_json::json!({
                    "request": self.request_ids.get(item),
                    "slot": self.slot_ids.get(item),
                    "start": start,
                    "rows": len,
                    "prefix": self.prefix_len.get(item),
                    "seq_len": self.seq_lens.get(item),
                    "tokens": self.token_ids.get(start..start + len),
                    "positions": self.positions.get(start..start + len),
                    "position_axes": if self.position_axes.is_empty() {None} else {self.position_axes.get(start..start + len)},
                    "item_order": self.item_order.get(item),
                    "verify_rows": self.verify_rows.get(item),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "program": program,
            "rows": rows,
            "dump_rows": [first, end],
            "dump_from": from,
            "live_tokens": self.live_tokens(),
            "items": items,
            "decode":self.decode,"verify":self.verify,
            "positions":self.positions,"position_axes":self.position_axes,
            "token_ids":self.token_ids,"token_batch":self.token_batch,
            "item_order":self.item_order,"last_positions":self.last_positions,
            "row_map":(0..rows as usize).map(|row| serde_json::json!({"row":row,
                "item":self.token_batch.get(row),"position":self.positions.get(row),
                "token":self.token_ids.get(row),"padding":row >= self.token_ids.len()})).collect::<Vec<_>>(),
        })
    }

    pub fn logit_rows(&self) -> Vec<u32> {
        self.q_start
            .iter()
            .zip(&self.q_len)
            .zip(&self.emits_row)
            .zip(&self.prompt_rows)
            .enumerate()
            .flat_map(|(index, (((&start, &length), &emit), &prompt))| {
                let start = start as u32;
                let length = length as u32;
                if prompt || self.verify_rows.get(index).copied().unwrap_or(false) {
                    (start..start + length).collect::<Vec<_>>()
                } else if emit {
                    vec![start + length - 1]
                } else {
                    Vec::new()
                }
            })
            .collect()
    }

    /// Splits a prefill phase at token `at` into two phases a pipelined
    /// prefill runs one after the other. An item that crosses `at` becomes a
    /// first piece that ends there and emits nothing, and a second piece that
    /// continues from it with the first piece as extra prefix. Rows keep their
    /// order: the second half's token `t` is this phase's token `at + t`.
    /// Returns `None` when either half would be empty, or the phase scores
    /// prompt rows, whose views need one piece per item, or keeps an
    /// anchor, which one item's rows must hold.
    pub fn split_at(&self, at: u32) -> Option<(Self, Self)> {
        let at = at as i32;
        if self.decode
            || at <= 0
            || at >= self.live_tokens() as i32
            || self.prompt_rows.iter().any(|&row| row)
            || self.anchors.iter().any(|&anchor| anchor)
        {
            return None;
        }
        self.split_at_with_rows(at as u32, &vec![None; self.item_order.len()])
    }

    /// Validate a contiguous split without allocating its row views.
    pub fn can_split_at_with_rows(&self, at: u32, anchor_ends: &[Option<i32>]) -> bool {
        let Ok(at) = i32::try_from(at) else {
            return false;
        };
        if self.decode
            || self.verify
            || at <= 0
            || at >= self.live_tokens() as i32
            || anchor_ends.len() != self.item_order.len()
        {
            return false;
        }
        for (item, &end) in anchor_ends.iter().enumerate() {
            if self.anchors.get(item).copied().unwrap_or(false) != end.is_some()
                || end.is_some_and(|end| {
                    end <= self.prefix_len[item] || end > self.prefix_len[item] + self.q_len[item]
                })
            {
                return false;
            }
        }
        true
    }

    /// Split contiguous prefill rows while retaining prompt scoring. An
    /// anchor belongs to the piece ending at or after its absolute frontier;
    /// callers supply that frontier for every anchored item. The legacy
    /// split keeps its prompt-row and anchor refusal policy.
    pub fn split_at_with_rows(&self, at: u32, anchor_ends: &[Option<i32>]) -> Option<(Self, Self)> {
        if !self.can_split_at_with_rows(at, anchor_ends) {
            return None;
        }
        let at = at as i32;
        let mut halves = (Self::default(), Self::default());
        for (row, &anchor_end) in anchor_ends.iter().enumerate() {
            let (start, length) = (self.q_start[row], self.q_len[row]);
            // Tokens of this item before `at`; the rest go to the second half.
            let first = (at - start).clamp(0, length);
            for (half, (piece_start, piece_length)) in
                [(0, (start, first)), (1, (start + first, length - first))]
            {
                if piece_length == 0 {
                    continue;
                }
                let target = if half == 0 {
                    &mut halves.0
                } else {
                    &mut halves.1
                };
                let base = if half == 0 { 0 } else { at };
                let batch = target.item_order.len() as i32;
                let prefix = self.prefix_len[row] + (piece_start - start);
                let last = half == 1 || first == length;
                target.item_order.push(self.item_order[row]);
                target.request_ids.push(self.request_ids[row]);
                target.emits_row.push(last && self.emits_row[row]);
                target.prompt_rows.push(self.prompt_rows[row]);
                target.anchors.push(
                    anchor_end.is_some_and(|end| prefix < end && end <= prefix + piece_length),
                );
                target.slot_ids.push(self.slot_ids[row]);
                target.q_start.push(piece_start - base);
                target.q_len.push(piece_length);
                target.prefix_len.push(prefix);
                target.seq_lens.push(prefix + piece_length);
                target
                    .last_positions
                    .push(i64::from(prefix + piece_length) - 1);
                target.block_tables.push(self.block_tables[row].clone());
                for token in piece_start..piece_start + piece_length {
                    target.token_ids.push(self.token_ids[token as usize]);
                    target.positions.push(self.positions[token as usize]);
                    if let Some(axes) = self.position_axes.get(token as usize) {
                        target.position_axes.push(*axes);
                    }
                    target.token_batch.push(batch);
                }
            }
        }
        (halves.0.live_tokens() > 0 && halves.1.live_tokens() > 0).then_some(halves)
    }

    fn push(
        &mut self,
        item_index: usize,
        item: &runtime_bundle_sdk::BatchItemV1,
        append: &runtime_bundle_sdk::BatchAppendV1,
        slot: u32,
    ) {
        let batch = self.item_order.len() as i32;
        self.item_order.push(item_index);
        self.request_ids.push(item.request_id());
        self.emits_row.push(item.expects_logits());
        self.prompt_rows.push(item.prompt_rows());
        self.anchors.push(item.anchor());
        self.verify_rows.push(item.is_verify());
        self.slot_ids.push(slot as i32);
        self.q_start.push(self.token_ids.len() as i32);
        self.q_len.push(item.phase_tokens() as i32);
        self.prefix_len.push(append.previous_tokens() as i32);
        self.seq_lens.push(append.new_tokens() as i32);
        self.last_positions.push(append.new_tokens() as i64 - 1);
        self.block_tables.push(
            append
                .page_table()
                .iter()
                .map(|page| *page as i32)
                .collect(),
        );
        let explicit = item.position_axes();
        if !explicit.is_empty() && self.position_axes.is_empty() {
            // Earlier items of the phase take their implied positions.
            self.position_axes = self.positions.iter().map(|&p| [p; 3]).collect();
        }
        for offset in 0..item.phase_tokens() as usize {
            self.token_ids
                .push(item.input_tokens().get(offset).copied().unwrap_or(0));
            let position = (append.previous_tokens() + offset as u64) as i32;
            self.positions.push(position);
            if let Some(axes) = explicit.get(offset) {
                self.position_axes.push(*axes);
            } else if !self.position_axes.is_empty() {
                self.position_axes.push([position; 3]);
            }
            self.token_batch.push(batch);
        }
    }
}

#[derive(Debug)]
pub struct PackedStep {
    pub prefill: PhaseInputs,
    pub decode: PhaseInputs,
}

impl PackedStep {
    /// [`Self::pack_striped`] on the state slots the host placed each item
    /// in (`BatchRequestV1::state_slots`), refused with
    /// [`InputError::status`]. A batch without slots is refused, and so is
    /// a decode of a request with nothing committed: decode reads the
    /// state its slot committed, and a fresh request starts with prefill.
    pub fn pack_on_host_slots<G: BundleGeometry>(
        limits: ModelLimits,
        request: &BatchRequestV1,
        page_capacity: u64,
        verify: VerifyLimits,
        served: Items,
        group: u64,
    ) -> Result<Self, BundleStatusV1> {
        let slots = request
            .state_slots()
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let by_request = request
            .items()
            .iter()
            .zip(slots)
            .map(|(item, &slot)| (item.request_id(), slot))
            .collect::<BTreeMap<_, _>>();
        let step = Self::pack_striped::<G>(
            limits,
            request,
            page_capacity,
            verify,
            served,
            group,
            |request| {
                by_request
                    .get(&request)
                    .and_then(|&slot| u32::try_from(slot).ok())
            },
        )
        .map_err(InputError::status)?;
        if step.decode.prefix_len.contains(&0) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        Ok(step)
    }

    /// Per decode item, the drafts `request` asks the bundle to write: every
    /// row but the first of a bundle-drafted item, none of another.
    pub fn bundle_drafts(&self, request: &BatchRequestV1) -> Vec<u32> {
        self.decode
            .item_order
            .iter()
            .zip(&self.decode.q_len)
            .map(|(&wire, &rows)| {
                if request.items()[wire].bundle_drafts() {
                    rows as u32 - 1
                } else {
                    0
                }
            })
            .collect()
    }

    /// Packs scheduled items in their wire order within each phase. The caller
    /// supplies its committed SWA slot mapping and opens KV transactions only
    /// after this validation succeeds.
    pub fn pack<G: BundleGeometry>(
        limits: ModelLimits,
        request: &BatchRequestV1,
        page_capacity: u64,
        slot_for: impl Fn(u64) -> Option<u32>,
    ) -> Result<Self, InputError> {
        Self::pack_with_verify::<G>(limits, request, page_capacity, VerifyLimits::NONE, slot_for)
    }

    /// Packs a step that may hold verify items within `verify`. Verify items
    /// carry host drafts: their input tokens are the bonus input and the
    /// drafts, and every row emits logits. They join the decode phase, which
    /// then runs as verify rows for all its items.
    pub fn pack_with_verify<G: BundleGeometry>(
        limits: ModelLimits,
        request: &BatchRequestV1,
        page_capacity: u64,
        verify: VerifyLimits,
        slot_for: impl Fn(u64) -> Option<u32>,
    ) -> Result<Self, InputError> {
        Self::pack_within::<G>(
            limits,
            request,
            page_capacity,
            verify,
            Items::package::<G>(),
            slot_for,
        )
    }

    /// [`Self::pack_with_verify`] for an instance serving `served` items: a
    /// step holds at most that many items, on slots below it.
    pub fn pack_within<G: BundleGeometry>(
        limits: ModelLimits,
        request: &BatchRequestV1,
        page_capacity: u64,
        verify: VerifyLimits,
        served: Items,
        slot_for: impl Fn(u64) -> Option<u32>,
    ) -> Result<Self, InputError> {
        Self::pack_striped::<G>(limits, request, page_capacity, verify, served, 1, slot_for)
    }

    /// [`Self::pack_within`] for a context-parallel group of `group` ranks,
    /// whose page tables list one local page per stripe of `group` physical
    /// pages (`runtime_kv::ContextParallelPagesV1`): an item of `end` tokens
    /// carries `end.div_ceil(group * page_tokens)` entries. A group of 1 is
    /// [`Self::pack_within`].
    pub fn pack_striped<G: BundleGeometry>(
        limits: ModelLimits,
        request: &BatchRequestV1,
        page_capacity: u64,
        verify: VerifyLimits,
        served: Items,
        group: u64,
        slot_for: impl Fn(u64) -> Option<u32>,
    ) -> Result<Self, InputError> {
        assert_eq!(limits.page_tokens(), G::PAGE_TOKENS);
        assert!(group > 0, "a context-parallel group has a rank");
        let items = request.items();
        let appends = request.appends();
        if items.is_empty() || items.len() != appends.len() || items.len() > served.count() {
            return Err(InputError::InvalidBatch);
        }
        let mut seen = BTreeSet::new();
        let mut prefill = PhaseInputs::default();
        let mut decode = PhaseInputs {
            decode: true,
            ..PhaseInputs::default()
        };
        for (index, (item, append)) in items.iter().zip(appends).enumerate() {
            let count = item.phase_tokens() as usize;
            let expected_inputs = if item.bundle_drafts() { 1 } else { count };
            if !seen.insert(item.request_id())
                || count == 0
                || item.input_tokens().len() != expected_inputs
                || (item.is_decode() && count != 1)
                || (!item.is_decode() && count > G::PREFILL_TOKENS as usize)
                || (item.is_verify() && (count < 2 || !item.expects_logits()))
                || !(item.position_axes().is_empty() || item.position_axes().len() == count)
            {
                return Err(InputError::InvalidBatch);
            }
            if item.is_verify()
                && (item.bundle_drafts() && !verify.bundle_drafter
                    || count > verify.rows_per_item as usize)
            {
                return Err(InputError::UnsupportedVerify);
            }
            let Some(end) = append.previous_tokens().checked_add(count as u64) else {
                return Err(InputError::InvalidAppend);
            };
            if append.additional_tokens() != count as u64
                || append.new_tokens() != end
                || end > limits.context_tokens()
                || append.page_table().len() != end.div_ceil(G::PAGE_TOKENS * group) as usize
                || append
                    .page_table()
                    .iter()
                    .any(|&page| page >= page_capacity)
                || (request.version() == BatchWireVersionV1::V1 && append.copy().is_some())
            {
                return Err(InputError::InvalidAppend);
            }
            if item
                .input_tokens()
                .iter()
                .any(|&token| token >= limits.vocab())
            {
                return Err(InputError::InvalidToken);
            }
            let slot = slot_for(item.request_id()).ok_or(InputError::MissingSlot)?;
            if !served.contains(slot) {
                return Err(InputError::MissingSlot);
            }
            if item.is_decode() || item.is_verify() {
                decode.push(index, item, append, slot);
            } else {
                prefill.push(index, item, append, slot);
            }
        }
        decode.verify = decode.verify_rows.contains(&true);
        if prefill.live_tokens() > G::PREFILL_TOKENS || decode.live_batch() > served.get() {
            return Err(InputError::InvalidBatch);
        }
        if decode.verify && decode.live_tokens() > verify.rows_per_step {
            return Err(InputError::UnsupportedVerify);
        }
        Ok(Self { prefill, decode })
    }

    /// Moves every decode item into the prefill phase as a one-token prefill
    /// item after the prefill items, so one forward serves the step and reads
    /// the weights once. Each item keeps its wire index, slot, context and
    /// page table, so its logits view and KV writes are those of the decode
    /// phase. Returns the rows moved, or zero, changing nothing, unless both
    /// phases hold items, the decode phase is plain decode (no verify rows),
    /// no item carries explicit position axes, and the prefill phase stays
    /// within `max_tokens` tokens and `max_logit_rows` logits rows.
    pub fn fold_decode(&mut self, max_tokens: u32, max_logit_rows: usize) -> u32 {
        let (prefill, decode) = (&mut self.prefill, &self.decode);
        let rows = decode.live_tokens();
        if prefill.live_batch() == 0
            || decode.live_batch() == 0
            || decode.verify
            || !prefill.position_axes.is_empty()
            || !decode.position_axes.is_empty()
            || prefill.live_tokens().saturating_add(rows) > max_tokens
            || prefill.logit_rows().len() + decode.logit_rows().len() > max_logit_rows
        {
            return 0;
        }
        for row in 0..decode.item_order.len() {
            let batch = prefill.item_order.len() as i32;
            let (start, length) = (decode.q_start[row], decode.q_len[row]);
            prefill.item_order.push(decode.item_order[row]);
            prefill.request_ids.push(decode.request_ids[row]);
            prefill.emits_row.push(decode.emits_row[row]);
            prefill.prompt_rows.push(false);
            prefill.anchors.push(false);
            prefill.verify_rows.push(false);
            prefill.slot_ids.push(decode.slot_ids[row]);
            prefill.q_start.push(prefill.token_ids.len() as i32);
            prefill.q_len.push(length);
            prefill.prefix_len.push(decode.prefix_len[row]);
            prefill.seq_lens.push(decode.seq_lens[row]);
            prefill.last_positions.push(decode.last_positions[row]);
            prefill.block_tables.push(decode.block_tables[row].clone());
            for token in start..start + length {
                prefill.token_ids.push(decode.token_ids[token as usize]);
                prefill.positions.push(decode.positions[token as usize]);
                prefill.token_batch.push(batch);
            }
        }
        self.decode = PhaseInputs {
            decode: true,
            ..PhaseInputs::default()
        };
        rows
    }
}

#[cfg(test)]
mod tests {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

    use super::*;
    use crate::geometry::tests::{LIMITS, Tp4};

    #[test]
    fn runtime_model_limits_bound_each_instance() {
        let request = decode_step(0..1);
        let wide = ModelLimits::new(43, 128, 64).unwrap();
        let small_vocab = ModelLimits::new(42, 128, 64).unwrap();
        let short_context = ModelLimits::new(43, 64, 64).unwrap();
        let pack = |limits| PackedStep::pack::<Tp4>(limits, &request, 64, |_| Some(0));
        assert!(pack(wide).is_ok());
        assert_eq!(pack(small_vocab).err(), Some(InputError::InvalidToken));
        assert_eq!(pack(short_context).err(), Some(InputError::InvalidAppend));
        assert!(pack(wide).is_ok());
    }

    const DECODE_BATCH: u32 = Tp4::DECODE_BATCH;
    const MAX_VERIFY_ROWS: u32 = Tp4::MAX_VERIFY_ROWS;

    /// One decode item per request id in `ids`, each on page 5 after 64
    /// tokens.
    fn decode_step(ids: impl Iterator<Item = u64>) -> BatchRequestV1 {
        let (items, appends) = ids
            .map(|id| {
                (
                    BatchItemV1::new(id, true, 1, true, vec![42]),
                    BatchAppendV1::new(64, 1, 65, vec![5, 6], None),
                )
            })
            .unzip();
        BatchRequestV1::new(BatchWireVersionV1::V2, items, appends)
    }

    /// Packing on the host's slots places each item on the slot the batch
    /// names, and refuses a batch without slots, an item whose slot is out
    /// of range, and a decode of a request with nothing committed.
    #[test]
    fn host_slots_place_items_and_refuse_fresh_decodes() {
        let pack = |request: BatchRequestV1| {
            PackedStep::pack_on_host_slots::<Tp4>(
                LIMITS,
                &request,
                64,
                VerifyLimits::NONE,
                Items::package::<Tp4>(),
                1,
            )
        };
        let invalid = Some(BundleStatusV1::InvalidArgument);
        assert_eq!(pack(decode_step(7..9)).err(), invalid);
        let step = pack(decode_step(7..9).with_state_slots(vec![3, 1])).unwrap();
        assert_eq!(step.decode.slot_ids, [3, 1]);
        assert_eq!(step.bundle_drafts(&decode_step(7..9)), [0, 0]);
        assert_eq!(
            pack(decode_step(7..9).with_state_slots(vec![3, u64::MAX])).err(),
            Some(BundleStatusV1::NotFound)
        );
        let fresh = || {
            BatchRequestV1::new(
                BatchWireVersionV1::V2,
                vec![BatchItemV1::new(7, true, 1, true, vec![42])],
                vec![BatchAppendV1::new(0, 1, 1, vec![5], None)],
            )
        };
        assert!(PackedStep::pack::<Tp4>(LIMITS, &fresh(), 64, |_| Some(0)).is_ok());
        assert_eq!(pack(fresh().with_state_slots(vec![0])).err(), invalid);
    }

    #[test]
    fn folded_decode_items_join_the_prefill_phase_as_one_token_items() {
        let items = vec![
            BatchItemV1::new(7, true, 1, true, vec![42]),
            BatchItemV1::new(8, false, 3, true, vec![1, 2, 3]),
            BatchItemV1::new(9, true, 1, true, vec![43]),
        ];
        let appends = vec![
            BatchAppendV1::new(64, 1, 65, vec![5, 6], None),
            BatchAppendV1::new(0, 3, 3, vec![7], None),
            BatchAppendV1::new(31, 1, 32, vec![8], None),
        ];
        let request = BatchRequestV1::new(BatchWireVersionV1::V2, items, appends);
        let pack =
            || PackedStep::pack::<Tp4>(LIMITS, &request, 64, |id| Some(id as u32 - 7)).unwrap();
        let unfolded = pack();
        let mut step = pack();
        assert_eq!(
            step.fold_decode(3, 512),
            0,
            "the phase would pass max_tokens"
        );
        assert_eq!(
            step.fold_decode(8192, 2),
            0,
            "the logits rows would pass the cap"
        );
        assert_eq!(step.fold_decode(8192, 512), 2);
        assert_eq!(step.decode.live_batch(), 0);
        assert!(step.decode.decode);
        let prefill = &step.prefill;
        assert_eq!(prefill.item_order, [1, 0, 2]);
        assert_eq!(prefill.request_ids, [8, 7, 9]);
        assert_eq!(prefill.token_ids, [1, 2, 3, 42, 43]);
        assert_eq!(prefill.positions, [0, 1, 2, 64, 31]);
        assert_eq!(prefill.token_batch, [0, 0, 0, 1, 2]);
        assert_eq!(prefill.q_start, [0, 3, 4]);
        assert_eq!(prefill.q_len, [3, 1, 1]);
        assert_eq!(prefill.prefix_len, [0, 64, 31]);
        assert_eq!(prefill.seq_lens, [3, 65, 32]);
        assert_eq!(prefill.last_positions, [2, 64, 31]);
        assert_eq!(prefill.slot_ids, [1, 0, 2]);
        assert_eq!(prefill.block_tables, [vec![7], vec![5, 6], vec![8]]);
        assert_eq!(prefill.logit_rows(), [2, 3, 4]);
        // Each folded item keeps its decode slot, context and pages.
        for (row, index) in [(1, 0), (2, 1)] {
            let decode = &unfolded.decode;
            assert_eq!(prefill.slot_ids[row], decode.slot_ids[index]);
            assert_eq!(prefill.seq_lens[row], decode.seq_lens[index]);
            assert_eq!(prefill.block_tables[row], decode.block_tables[index]);
        }
        // Nothing to fold into, or no decode items: unchanged.
        let decode_only =
            PackedStep::pack::<Tp4>(LIMITS, &decode_step(0..2), 64, |id| Some(id as u32)).unwrap();
        let mut again =
            PackedStep::pack::<Tp4>(LIMITS, &decode_step(0..2), 64, |id| Some(id as u32)).unwrap();
        assert_eq!(again.fold_decode(8192, 512), 0);
        assert_eq!(again.decode.request_ids, decode_only.decode.request_ids);
        assert_eq!(step.fold_decode(8192, 512), 0);
    }

    #[test]
    fn steps_hold_at_most_the_served_items_on_their_slots() {
        use crate::geometry::tests::{Max, every_item_count};
        for items in every_item_count() {
            let count = u64::from(items.get());
            let pack = |request: &BatchRequestV1, slot_for: &dyn Fn(u64) -> Option<u32>| {
                PackedStep::pack_within::<Max>(
                    LIMITS,
                    request,
                    32_768,
                    VerifyLimits::NONE,
                    items,
                    slot_for,
                )
            };
            let full = pack(&decode_step(0..count), &|id| Some(id as u32)).unwrap();
            assert_eq!(full.decode.live_batch(), items.get());
            assert_eq!(full.decode.slot_ids.last(), Some(&(items.get() as i32 - 1)));
            assert!(items.bucket(full.decode.live_batch()).is_some());
            assert_eq!(
                pack(&decode_step(0..count + 1), &|id| Some(id as u32)).err(),
                Some(InputError::InvalidBatch)
            );
            // A slot past the served ones is refused, whatever the package runs.
            assert_eq!(
                pack(&decode_step(0..1), &|_| Some(items.get())).err(),
                Some(InputError::MissingSlot)
            );
            let verify = VerifyLimits::enabled_within::<Max>(7, 1024, items);
            assert_eq!(verify.rows_per_step, 256);
            assert_eq!(
                VerifyLimits::enabled_within::<Tp4>(7, 1024, Items::within::<Tp4>(1).unwrap())
                    .rows_per_step,
                DECODE_BATCH
            );
            let shape = BatchShape::of(&decode_step(0..count + 1));
            assert!(
                shape
                    .describe_within::<Max>(verify, items)
                    .contains(&format!("{} items (at most {count})", count + 1)),
            );
        }
        // A verify step is bucketed by its rows: 8 items of 4 rows at 8
        // served items is the 32-row bucket, not over the largest.
        let verify_step = BatchShape {
            items: 8,
            verify_items: 8,
            decode_rows: 32,
            largest_verify: 4,
            ..BatchShape::default()
        };
        let described = verify_step.describe_within::<Max>(
            VerifyLimits::enabled_within::<Max>(7, 256, Items::new(8).unwrap()),
            Items::new(8).unwrap(),
        );
        assert!(described.contains("graph bucket 32;"), "{described}");
        // The package maximum packs as before the count was a runtime value.
        let step = decode_step(0..u64::from(DECODE_BATCH) + 1);
        assert_eq!(
            PackedStep::pack::<Tp4>(LIMITS, &step, 32_768, |id| Some(id as u32)).err(),
            Some(InputError::InvalidBatch)
        );
    }

    #[test]
    fn logit_rows_include_flagged_prompt_tokens_and_other_final_rows() {
        let inputs = PhaseInputs {
            q_start: vec![0, 2, 5],
            q_len: vec![2, 3, 1],
            emits_row: vec![true, false, true],
            prompt_rows: vec![false, true, false],
            ..PhaseInputs::default()
        };
        assert_eq!(inputs.logit_rows(), [1, 2, 3, 4, 5]);
    }

    #[test]
    fn mixed_step_packs_phase_local_descriptors_and_preserves_item_order() {
        let request = BatchRequestV1::new(
            BatchWireVersionV1::V2,
            vec![
                BatchItemV1::new(7, true, 1, true, vec![42]),
                BatchItemV1::new(8, false, 2, false, vec![1, 2]),
                BatchItemV1::new(9, true, 1, true, vec![3]),
            ],
            vec![
                BatchAppendV1::new(64, 1, 65, vec![5, 6], None),
                BatchAppendV1::new(0, 2, 2, vec![7], None),
                BatchAppendV1::new(2, 1, 3, vec![8], None),
            ],
        );
        let step = PackedStep::pack::<Tp4>(LIMITS, &request, 32_768, |id| Some((id - 7) as u32))
            .expect("mixed step");
        assert_eq!(step.prefill.item_order, [1]);
        assert_eq!(step.prefill.token_ids, [1, 2]);
        assert_eq!(step.prefill.positions, [0, 1]);
        assert_eq!(step.prefill.token_batch, [0, 0]);
        assert_eq!(step.prefill.slot_ids, [1]);
        assert_eq!(step.prefill.q_start, [0]);
        assert_eq!(step.prefill.q_len, [2]);
        assert_eq!(step.prefill.prefix_len, [0]);
        assert_eq!(step.decode.item_order, [0, 2]);
        assert_eq!(step.decode.positions, [64, 2]);
        assert_eq!(step.decode.token_batch, [0, 1]);
        assert_eq!(step.decode.slot_ids, [0, 2]);
        assert_eq!(step.decode.seq_lens, [65, 3]);
        assert_eq!(step.decode.block_tables, [vec![5, 6], vec![8]]);
    }

    #[test]
    fn split_with_rows_preserves_scoring_and_assigns_each_anchor_once() {
        let request = BatchRequestV1::new(
            BatchWireVersionV1::V2,
            vec![
                BatchItemV1::new(7, false, 3, true, vec![1, 2, 3]),
                BatchItemV1::new(8, false, 5, true, vec![4, 5, 6, 7, 8])
                    .with_prompt_rows(true)
                    .with_anchor(true),
            ],
            vec![
                BatchAppendV1::new(10, 3, 13, vec![5], None),
                BatchAppendV1::new(0, 5, 5, vec![7], None),
            ],
        );
        let step =
            PackedStep::pack::<Tp4>(LIMITS, &request, 32_768, |id| Some((id - 7) as u32)).unwrap();
        assert!(step.prefill.split_at(5).is_none());
        for (end, in_first) in [(1, true), (2, true), (4, false), (5, false)] {
            let (first, second) = step
                .prefill
                .split_at_with_rows(5, &[None, Some(end)])
                .unwrap();
            assert_eq!(first.logit_rows(), [2, 3, 4]);
            assert_eq!(second.logit_rows(), [0, 1, 2]);
            assert_eq!(first.anchors, [false, in_first]);
            assert_eq!(second.anchors, [!in_first]);
            assert_eq!(second.prefix_len, [2]);
            assert_eq!(second.positions, [2, 3, 4]);
            assert_eq!(
                first
                    .token_ids
                    .into_iter()
                    .chain(second.token_ids)
                    .collect::<Vec<_>>(),
                step.prefill.token_ids
            );
        }
        assert!(step.prefill.split_at_with_rows(5, &[None, None]).is_none());
        assert!(
            step.prefill
                .split_at_with_rows(5, &[None, Some(6)])
                .is_none()
        );
    }

    #[test]
    fn split_prefill_keeps_rows_and_continues_crossing_items() {
        let request = BatchRequestV1::new(
            BatchWireVersionV1::V2,
            vec![
                BatchItemV1::new(7, false, 3, true, vec![1, 2, 3]),
                BatchItemV1::new(8, false, 5, true, vec![4, 5, 6, 7, 8]),
            ],
            vec![
                BatchAppendV1::new(10, 3, 13, vec![5], None),
                BatchAppendV1::new(0, 5, 5, vec![7], None),
            ],
        );
        let step =
            PackedStep::pack::<Tp4>(LIMITS, &request, 32_768, |id| Some((id - 7) as u32)).unwrap();
        let (first, second) = step.prefill.split_at(5).unwrap();
        // Item 7 stays whole in the first half; item 8 is split 2 + 3.
        assert_eq!(first.item_order, [0, 1]);
        assert_eq!(first.token_ids, [1, 2, 3, 4, 5]);
        assert_eq!(first.q_start, [0, 3]);
        assert_eq!(first.q_len, [3, 2]);
        assert_eq!(first.prefix_len, [10, 0]);
        assert_eq!(first.seq_lens, [13, 2]);
        assert_eq!(first.last_positions, [12, 1]);
        assert_eq!(first.emits_row, [true, false]);
        assert_eq!(first.token_batch, [0, 0, 0, 1, 1]);
        assert_eq!(second.item_order, [1]);
        assert_eq!(second.token_ids, [6, 7, 8]);
        assert_eq!(second.positions, [2, 3, 4]);
        assert_eq!(second.q_start, [0]);
        assert_eq!(second.prefix_len, [2]);
        assert_eq!(second.seq_lens, [5]);
        assert_eq!(second.emits_row, [true]);
        assert_eq!(second.slot_ids, [1]);
        assert_eq!(second.block_tables, [vec![7]]);
        // Logit rows of the halves, shifted back, are the whole phase's.
        let mut rows = first.logit_rows();
        rows.extend(second.logit_rows().iter().map(|row| row + 5));
        assert_eq!(rows, step.prefill.logit_rows());
        assert!(step.prefill.split_at(0).is_none());
        assert!(step.prefill.split_at(8).is_none());
        assert!(step.decode.split_at(1).is_none());
    }

    #[test]
    fn explicit_position_axes_pack_per_token_and_follow_a_split() {
        let request = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![
                BatchItemV1::new(7, false, 2, true, vec![1, 2]),
                BatchItemV1::new(8, false, 3, true, vec![3, 4, 5]).with_position_axes(vec![
                    [2, 2, 2],
                    [2, 2, 3],
                    [5, 5, 5],
                ]),
                BatchItemV1::verify(9, 3, false, vec![6, 7, 8]).with_position_axes(vec![
                    [4, 4, 4],
                    [5, 5, 5],
                    [6, 6, 6],
                ]),
                BatchItemV1::new(10, true, 1, true, vec![9]),
            ],
            vec![
                BatchAppendV1::new(10, 2, 12, vec![5], None),
                BatchAppendV1::new(0, 3, 3, vec![7], None),
                BatchAppendV1::new(8, 3, 11, vec![8], None),
                BatchAppendV1::new(2, 1, 3, vec![9], None),
            ],
        );
        let step = PackedStep::pack_with_verify::<Tp4>(
            LIMITS,
            &request,
            32_768,
            VerifyLimits::enabled::<Tp4>(7, 32),
            |id| Some((id - 7) as u32),
        )
        .unwrap();
        // The item without positions repeats its 1D position on every axis.
        assert_eq!(
            step.prefill.position_axes,
            [[10, 10, 10], [11, 11, 11], [2, 2, 2], [2, 2, 3], [5, 5, 5]]
        );
        assert_eq!(
            step.decode.position_axes,
            [[4, 4, 4], [5, 5, 5], [6, 6, 6], [2, 2, 2]]
        );
        let (first, second) = step.prefill.split_at(3).unwrap();
        assert_eq!(first.position_axes, [[10, 10, 10], [11, 11, 11], [2, 2, 2]]);
        assert_eq!(second.position_axes, [[2, 2, 3], [5, 5, 5]]);
        // A step without explicit positions packs none.
        let plain = PackedStep::pack::<Tp4>(
            LIMITS,
            &BatchRequestV1::new(
                BatchWireVersionV1::V3,
                vec![BatchItemV1::new(7, false, 2, true, vec![1, 2])],
                vec![BatchAppendV1::new(0, 2, 2, vec![5], None)],
            ),
            32_768,
            |_| Some(0),
        )
        .unwrap();
        assert!(plain.prefill.position_axes.is_empty());
        // Positions that do not cover the item's rows are refused.
        let partial = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::new(7, false, 2, true, vec![1, 2]).with_position_axes(vec![[0; 3]])],
            vec![BatchAppendV1::new(0, 2, 2, vec![5], None)],
        );
        assert_eq!(
            PackedStep::pack::<Tp4>(LIMITS, &partial, 32_768, |_| Some(0)).err(),
            Some(InputError::InvalidBatch)
        );
    }

    #[test]
    fn malformed_append_and_unknown_slot_refuse_before_packing() {
        let item = BatchItemV1::new(1, true, 1, true, vec![0]);
        let append = BatchAppendV1::new(0, 1, 1, vec![32_768], None);
        let request = BatchRequestV1::new(BatchWireVersionV1::V2, vec![item], vec![append]);
        assert!(matches!(
            PackedStep::pack::<Tp4>(LIMITS, &request, 32_768, |_| Some(0)),
            Err(InputError::InvalidAppend)
        ));
        let request = BatchRequestV1::new(
            BatchWireVersionV1::V2,
            vec![BatchItemV1::new(1, true, 1, true, vec![0])],
            vec![BatchAppendV1::new(0, 1, 1, vec![0], None)],
        );
        assert!(matches!(
            PackedStep::pack::<Tp4>(LIMITS, &request, 32_768, |_| None),
            Err(InputError::MissingSlot)
        ));
    }

    /// A group of 2 lists one page per stripe of 128 tokens: 129 tokens take
    /// two entries, not the three 64-token pages a group of 1 lists.
    #[test]
    fn striped_page_tables_list_one_page_per_stripe() {
        let step = |pages: Vec<u64>| {
            BatchRequestV1::new(
                BatchWireVersionV1::V2,
                vec![BatchItemV1::new(1, true, 1, true, vec![0])],
                vec![BatchAppendV1::new(128, 1, 129, pages, None)],
            )
        };
        let pack = |request: &BatchRequestV1, group| {
            PackedStep::pack_striped::<Tp4>(
                LIMITS,
                request,
                64,
                VerifyLimits::NONE,
                Items::package::<Tp4>(),
                group,
                |_| Some(0),
            )
        };
        let striped = pack(&step(vec![4, 9]), 2).unwrap();
        assert_eq!(striped.decode.block_tables, [vec![4, 9]]);
        assert_eq!(
            pack(&step(vec![4, 9, 11]), 2).err(),
            Some(InputError::InvalidAppend)
        );
        assert_eq!(
            pack(&step(vec![4, 9]), 1).err(),
            Some(InputError::InvalidAppend)
        );
        pack(&step(vec![4, 9, 11]), 1).unwrap();
    }

    fn verify_step() -> BatchRequestV1 {
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![
                BatchItemV1::verify(7, 4, false, vec![42, 1, 2, 3]),
                BatchItemV1::new(8, false, 2, true, vec![5, 6]),
                BatchItemV1::new(9, true, 1, true, vec![4]),
            ],
            vec![
                BatchAppendV1::new(62, 4, 66, vec![5, 6], None),
                BatchAppendV1::new(0, 2, 2, vec![7], None),
                BatchAppendV1::new(2, 1, 3, vec![8], None),
            ],
        )
    }

    #[test]
    fn verify_items_and_decodes_share_a_verify_phase_that_emits_every_row() {
        let step = PackedStep::pack_with_verify::<Tp4>(
            LIMITS,
            &verify_step(),
            32_768,
            VerifyLimits::enabled::<Tp4>(7, 32),
            |id| Some((id - 7) as u32),
        )
        .expect("verify step");
        assert!(step.decode.verify && step.decode.decode);
        assert!(!step.prefill.verify);
        assert_eq!(step.decode.item_order, [0, 2]);
        assert_eq!(step.decode.verify_rows, [true, false]);
        assert_eq!(step.decode.token_ids, [42, 1, 2, 3, 4]);
        assert_eq!(step.decode.positions, [62, 63, 64, 65, 2]);
        assert_eq!(step.decode.token_batch, [0, 0, 0, 0, 1]);
        assert_eq!(step.decode.q_start, [0, 4]);
        assert_eq!(step.decode.q_len, [4, 1]);
        assert_eq!(step.decode.prefix_len, [62, 2]);
        assert_eq!(step.decode.seq_lens, [66, 3]);
        assert_eq!(step.decode.last_positions, [65, 2]);
        assert_eq!(step.decode.logit_rows(), [0, 1, 2, 3, 4]);
        assert_eq!(step.prefill.item_order, [1]);
    }

    #[test]
    fn a_wide_verify_step_carries_eight_rows_for_every_item() {
        use crate::geometry::tests::Wide;
        let items = (0..32_u64)
            .map(|id| BatchItemV1::verify(id, 8, false, vec![1; 8]))
            .collect();
        let appends = (0..32_u64)
            .map(|_| BatchAppendV1::new(62, 8, 70, vec![5, 6], None))
            .collect();
        let request = BatchRequestV1::new(BatchWireVersionV1::V3, items, appends);
        let pack = |limits| {
            PackedStep::pack_with_verify::<Wide>(LIMITS, &request, 32_768, limits, |id| {
                Some(id as u32)
            })
        };
        let step = pack(VerifyLimits::enabled::<Wide>(7, 256)).expect("256 verify rows");
        assert_eq!(step.decode.live_tokens(), 256);
        assert_eq!(step.decode.logit_rows().len(), 256);
        assert_eq!(
            pack(VerifyLimits::enabled::<Wide>(7, 255)).err(),
            Some(InputError::UnsupportedVerify)
        );
        assert_eq!(
            PackedStep::pack_with_verify::<Tp4>(
                LIMITS,
                &request,
                32_768,
                VerifyLimits::enabled::<Tp4>(7, 256),
                |id| Some(id as u32)
            )
            .err(),
            Some(InputError::UnsupportedVerify)
        );
    }

    #[test]
    fn verify_items_outside_the_enabled_limits_are_refused() {
        let pack = |request: &BatchRequestV1, limits| {
            PackedStep::pack_with_verify::<Tp4>(LIMITS, request, 32_768, limits, |id| {
                Some((id - 7) as u32)
            })
        };
        let step = verify_step();
        assert_eq!(
            pack(&step, VerifyLimits::NONE).err(),
            Some(InputError::UnsupportedVerify)
        );
        assert_eq!(
            pack(&step, VerifyLimits::enabled::<Tp4>(2, 32)).err(),
            Some(InputError::UnsupportedVerify)
        );
        assert_eq!(
            pack(&step, VerifyLimits::enabled::<Tp4>(7, 4)).err(),
            Some(InputError::UnsupportedVerify)
        );
        assert_eq!(
            VerifyLimits::enabled::<Tp4>(15, 64).rows_per_item,
            MAX_VERIFY_ROWS
        );
        assert_eq!(
            VerifyLimits::enabled::<Tp4>(15, 64).rows_per_step,
            DECODE_BATCH
        );
        // A geometry with wider verify steps admits their rows, not more.
        assert_eq!(
            VerifyLimits::enabled::<crate::geometry::tests::Wide>(7, 1024).rows_per_step,
            256
        );
        let drafted = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::verify(7, 4, true, vec![42])],
            vec![BatchAppendV1::new(62, 4, 66, vec![5, 6], None)],
        );
        assert_eq!(
            pack(&drafted, VerifyLimits::enabled::<Tp4>(7, 32)).err(),
            Some(InputError::UnsupportedVerify)
        );
        let single = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::verify(7, 1, false, vec![42])],
            vec![BatchAppendV1::new(62, 1, 63, vec![5], None)],
        );
        assert_eq!(
            pack(&single, VerifyLimits::enabled::<Tp4>(7, 32)).err(),
            Some(InputError::InvalidBatch)
        );
    }

    /// Prefill past the arena is refused, and the refusal's description
    /// names the tokens and the limit they crossed.
    #[test]
    fn a_batch_past_the_prefill_arena_is_refused_and_described() {
        let chunk = |id, tokens: u32| {
            (
                BatchItemV1::new(id, false, tokens, false, vec![1; tokens as usize]),
                BatchAppendV1::new(
                    0,
                    u64::from(tokens),
                    u64::from(tokens),
                    (0..u64::from(tokens).div_ceil(64)).collect(),
                    None,
                ),
            )
        };
        let (items, appends): (Vec<_>, Vec<_>) = [chunk(1, 2048), chunk(2, 2048), chunk(3, 2048)]
            .into_iter()
            .chain([chunk(4, 2048), chunk(5, 117)])
            .unzip();
        let request = BatchRequestV1::new(BatchWireVersionV1::V3, items, appends);
        let limits = VerifyLimits::enabled::<Tp4>(7, 32);
        let refused = PackedStep::pack_with_verify::<Tp4>(LIMITS, &request, 32_768, limits, |id| {
            Some(id as u32)
        });
        assert_eq!(refused.err(), Some(InputError::InvalidBatch));
        let shape = BatchShape::of(&request);
        assert_eq!((shape.items, shape.prefill_tokens), (5, 8309));
        let description = shape.describe::<Tp4>(limits);
        assert!(
            description
                .starts_with("5 items (at most 32); prefill: 5 items, 8309 tokens (at most 8192)"),
            "{description}"
        );
        assert!(description.contains("graph bucket none"), "{description}");
    }
}
