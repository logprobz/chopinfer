//! Pure allocation layouts consumed by package admission and native allocation.
use crate::geometry::V26_FLASH as GEOMETRY;

use crate::Tp;
use crate::backend::{Kernels, Selected, StateBytes};
use crate::serving::{DECODE_ENTRY_ROWS, PREFILL_TOKENS};
use runtime_bundle_abi::BundleAbiError;

pub const TOKENS: u64 = PREFILL_TOKENS as u64;
pub const ROUTES_FLASH: u64 = TOKENS * GEOMETRY.top_k() as u64;
pub const PADDED_ROUTES_FLASH: u64 = ROUTES_FLASH + crate::tp::EXPERTS * (64 - 1);
/// Token-indexed hidden buffers keep half a chunk of rows past their span, so
/// a pipelined prefill's second half can use a span of the full capacity
/// that starts at its first row.
pub const HIDDEN_SLACK_ROWS: u64 = TOKENS / 2;

pub use kernel_core::allocation::Allocation as BufferLayout;
use kernel_core::allocation::Traversal;

pub const HIDDEN_LAYOUT_FLASH: BufferLayout = match BufferLayout::new(
    (TOKENS + HIDDEN_SLACK_ROWS) * crate::tp::HIDDEN * 2,
    TOKENS * crate::tp::HIDDEN * 2,
) {
    Ok(layout) => layout,
    Err(_) => panic!("MiMo hidden allocation layout"),
};

/// One comm state per decode and verify token bucket, then the prefill
/// state. Buckets above 32 rows reduce through the two-shot protocol.
/// Policy: select prepared communication capacities.
pub const COMM_BUCKETS: [u32; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, PREFILL_TOKENS];

/// Inbox bytes of each decode comm state; the prefill state takes the rest.
const DECODE_INBOX_BYTES: u64 = 8 << 20;
/// Inbox bytes of each verify state above the decode batch: its two-shot
/// rows start at 8 MiB, above the push protocols' regions.
const WIDE_INBOX_BYTES: u64 = 16 << 20;

fn inbox_bytes(bucket: u32) -> u64 {
    if bucket <= DECODE_ENTRY_ROWS {
        DECODE_INBOX_BYTES
    } else {
        WIDE_INBOX_BYTES
    }
}

/// Flag bytes of one comm state's hidden all-reduces: four words per rank
/// and row, or more if the fused protocols of every row count up to
/// `capacity` need more.
pub fn hidden_flag_bytes(capacity: u32, tp: Tp) -> Result<u64, BundleAbiError> {
    let per_row = 4 * u64::from(tp.ranks()) * u64::from(capacity) * 4;
    Ok(per_row.max(fused_state(capacity, tp)?.flags))
}

/// Epoch bytes of one comm state's hidden all-reduces: one word per row,
/// raised as the flags are.
pub fn hidden_epoch_bytes(capacity: u32, tp: Tp) -> Result<u64, BundleAbiError> {
    let per_row = u64::from(capacity) * 4;
    Ok(per_row.max(fused_state(capacity, tp)?.epoch))
}

fn fused_state(capacity: u32, tp: Tp) -> Result<StateBytes, BundleAbiError> {
    Selected::all_reduce_state(tp.ranks(), capacity)
        .map_err(|_| BundleAbiError::InvalidRange("MiMo comm state for the rank count"))
}

/// A rank's comm inbox. The decode and verify states take a fixed 96 MiB and
/// the prefill state the rest: what its fused protocols need at the rank
/// count, rounded up to MiB.
pub fn comm_inbox_bytes(tp: Tp) -> u64 {
    let fixed = COMM_BUCKETS[..COMM_BUCKETS.len() - 1]
        .iter()
        .map(|&bucket| inbox_bytes(bucket))
        .sum::<u64>();
    let prefill = Selected::all_reduce_inbox_bytes(tp.ranks(), PREFILL_TOKENS)
        .expect("MiMo prefill comm state");
    fixed + prefill.next_multiple_of(1 << 20)
}

/// KV heads per rank the attention fold counters cover: the most any
/// topology places on a rank, including copied KV heads.
#[cfg(test)]
pub const FOLD_COUNTER_KV_HEADS: u64 = max_rank_kv_heads();

#[cfg(test)]
const fn max_rank_kv_heads() -> u64 {
    let mut maximum = 1;
    let mut index = 0;
    while index < Tp::SUPPORTED.len() {
        let ranks = Tp::SUPPORTED[index].ranks() as u64;
        let ga = crate::tp::GA_KV_HEADS.div_ceil(ranks);
        let swa = crate::tp::SWA_KV_HEADS.div_ceil(ranks);
        if ga > maximum {
            maximum = ga;
        }
        if swa > maximum {
            maximum = swa;
        }
        index += 1;
    }
    maximum
}

kernel_core::scratch_record! {
    #[derive(Clone, Copy, Debug)]
    pub struct ExchangeBuffers {
        pub flags: value,
        pub epoch: value,
    }
}

kernel_core::scratch_record! {
    #[derive(Clone, Copy, Debug)]
    pub struct CommBuffers {
        pub hidden_flags: value,
        pub hidden_epoch: value,
        pub gather_rows_flags: value,
        pub gather_rows_epoch: value,
        /// The logits row exchange's flags and epoch, when it is enabled.
        pub exchange_rows: optional(ExchangeBuffers),
    }
}

impl CommBuffers<BufferLayout> {
    fn new(capacity: u32, exchange: bool, tp: Tp) -> Result<Self, BundleAbiError> {
        Ok(Self {
            hidden_flags: BufferLayout::bytes(hidden_flag_bytes(capacity, tp)?),
            hidden_epoch: BufferLayout::bytes(hidden_epoch_bytes(capacity, tp)?),
            gather_rows_flags: BufferLayout::bytes(Selected::GATHER_ROWS_STATE.flags),
            gather_rows_epoch: BufferLayout::bytes(Selected::GATHER_ROWS_STATE.epoch),
            exchange_rows: exchange.then_some(ExchangeBuffers {
                flags: BufferLayout::bytes(Selected::EXCHANGE_ROWS_STATE.flags),
                epoch: BufferLayout::bytes(Selected::EXCHANGE_ROWS_STATE.epoch),
            }),
        })
    }
}

kernel_core::scratch_record! {
    #[derive(Clone, Debug)]
    pub struct ScratchBuffers {
        pub qkv: value,
        pub q: value,
        pub k_new: value,
        pub v_new: value,
        pub attention: value,
        pub hidden_partial: value,
        pub dense_gate: value,
        pub router_logits: value,
        pub topk_ids: value,
        pub topk_weights: value,
        pub expert_counts: value,
        pub expert_offsets: value,
        pub sorted_rows: value,
        pub padded_rows: value,
        pub gathered: value,
        pub routed_gate: value,
        pub routed_down: value,
        pub combined: value,
        pub comm_inbox: value,
        pub comm_state: list(CommBuffers),
        pub active_experts: value,
        /// A constant expert count of 256 for prefill, which scans every expert;
        /// decode's fused router rewrites `active_experts` instead.
        pub all_experts: value,
        pub unit_scale: value,
        pub gemm_semaphores: value,
        /// One u32 per (token row, KV head) for attention CTAs that fold a row's
        /// partials themselves: each owner adds one after writing its partial,
        /// the last folds the row and stores zero again, so the counters need
        /// only the zeroing at create. Nothing else writes them.
        pub attention_fold_counters: value,
        /// The parallel decode route's arrival count. Zeroed at create; only the
        /// route writes it (it used to share the GEMM semaphores' first word).
        pub route_ticket: value,
    }
}

pub type ScratchLayout = ScratchBuffers<BufferLayout>;

impl ScratchLayout {
    pub fn new(tp: Tp, exchange_through: u32) -> Result<Self, BundleAbiError> {
        let geometry = tp.geometry;
        let routes = TOKENS * geometry.top_k() as u64;
        let padded_routes = routes + geometry.experts() as u64 * (64 - 1);
        let hidden = BufferLayout::rows(TOKENS, HIDDEN_SLACK_ROWS, geometry.hidden() as u64 * 2)
            .map_err(bundle_runtime::scratch::allocation_error)?;
        let fold_heads = Tp::SUPPORTED
            .iter()
            .flat_map(|tp| {
                let tp = Tp::of_geometry(geometry, tp.ranks()).expect("listed geometry");
                [tp.kv_heads(true), tp.kv_heads(false)]
            })
            .max()
            .unwrap();
        let q = [true, false]
            .map(|global| tp.q_heads_for(global) * tp.qk_dim(global))
            .into_iter()
            .max()
            .unwrap();
        let k = [true, false]
            .map(|global| tp.kv_heads(global) * tp.qk_dim(global))
            .into_iter()
            .max()
            .unwrap();
        let v = [true, false]
            .map(|global| tp.kv_heads(global) * tp.value_dim(global))
            .into_iter()
            .max()
            .unwrap();
        let attention = [true, false]
            .map(|global| tp.q_heads_for(global) * tp.value_dim(global))
            .into_iter()
            .max()
            .unwrap();
        Ok(Self {
            qkv: BufferLayout::bytes(TOKENS * tp.qkv_rows(true).max(tp.qkv_rows(false)) * 2),
            q: BufferLayout::bytes(TOKENS * q * 2),
            k_new: BufferLayout::bytes(TOKENS * k * 2),
            v_new: BufferLayout::bytes(TOKENS * v * 2),
            attention: BufferLayout::bytes(TOKENS * attention * 2),
            hidden_partial: hidden,
            dense_gate: BufferLayout::bytes(TOKENS * tp.dense_intermediate() * 2),
            router_logits: BufferLayout::bytes(TOKENS * geometry.experts() as u64 * 4),
            topk_ids: BufferLayout::bytes(TOKENS * geometry.top_k() as u64 * 4),
            topk_weights: BufferLayout::bytes(TOKENS * geometry.top_k() as u64 * 4),
            expert_counts: BufferLayout::bytes(geometry.experts() as u64 * 4),
            expert_offsets: BufferLayout::bytes((geometry.experts() as u64 + 1) * 4),
            sorted_rows: BufferLayout::bytes(padded_routes * 4),
            padded_rows: BufferLayout::bytes(4),
            gathered: BufferLayout::bytes(padded_routes * geometry.hidden() as u64 * 2),
            routed_gate: BufferLayout::bytes(routes * tp.expert_intermediate() * 2),
            routed_down: BufferLayout::bytes(routes * geometry.hidden() as u64 * 2),
            combined: hidden,
            comm_inbox: BufferLayout::bytes(comm_inbox_bytes(tp)),
            comm_state: COMM_BUCKETS
                .into_iter()
                .map(|capacity| CommBuffers::new(capacity, capacity <= exchange_through, tp))
                .collect::<Result<Vec<_>, _>>()?,
            active_experts: BufferLayout::bytes((geometry.experts() as u64 + 1) * 4),
            all_experts: BufferLayout::bytes((geometry.experts() as u64 + 1) * 4),
            unit_scale: BufferLayout::bytes(4),
            gemm_semaphores: BufferLayout::bytes(TOKENS * geometry.hidden() as u64 / 128 * 4),
            attention_fold_counters: BufferLayout::bytes(TOKENS * fold_heads * 4),
            route_ticket: BufferLayout::bytes(Selected::ROUTE_TICKET_BYTES),
        })
    }

    /// Ordered physical allocations, including hidden slack and optional exchanges.
    pub fn allocations(&self) -> Vec<BufferLayout> {
        let mut result = Vec::new();
        self.visit(&mut |layout| result.push(*layout));
        result
    }
}

/// Decode/verify and scored logits allocated by create, in their existing order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogitsLayout {
    pub local: u64,
    pub gathered_prefill: u64,
    pub gathered_decode: u64,
    pub local_scored: u64,
    pub gathered_scored: u64,
}

impl LogitsLayout {
    pub fn new(tp: Tp, logit_rows: u32, f32_decode: bool) -> Self {
        use crate::serving::{DECODE_LOGIT_BYTES, PREFILL_LOGIT_BYTES, SCORED_ROWS};
        let local = u64::from(crate::topology::vocab_per_rank(tp));
        let vocab = u64::from(tp.geometry.vocab());
        let rows = u64::from(logit_rows);
        let scored = u64::from(SCORED_ROWS.max(logit_rows));
        Self {
            local: rows * local * DECODE_LOGIT_BYTES,
            gathered_prefill: rows * vocab * DECODE_LOGIT_BYTES,
            gathered_decode: rows
                * vocab
                * if f32_decode {
                    PREFILL_LOGIT_BYTES
                } else {
                    DECODE_LOGIT_BYTES
                },
            local_scored: scored * local * PREFILL_LOGIT_BYTES,
            gathered_scored: scored * vocab * PREFILL_LOGIT_BYTES,
        }
    }

    pub fn total_bytes(self) -> u64 {
        self.local
            + self.gathered_prefill
            + self.gathered_decode
            + self.local_scored
            + self.gathered_scored
    }
}

/// Offset and visible bytes of a comm bucket within its rank inbox.
pub fn comm_inbox_range(bytes: u64, index: usize) -> (u64, u64) {
    let prefill = COMM_BUCKETS.len() - 1;
    let offset = COMM_BUCKETS[..index.min(prefill)]
        .iter()
        .map(|&bucket| inbox_bytes(bucket))
        .sum::<u64>();
    let length = if index < prefill {
        inbox_bytes(COMM_BUCKETS[index])
    } else {
        bytes.saturating_sub(offset)
    };
    (offset, length)
}

/// One physical allocation in the shared create buffer table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferRow {
    pub name: String,
    pub layout: BufferLayout,
}

/// Text and drafter allocations under the package's maximum verification policy.
/// Prefix pages and allocation padding are reserved separately by capacity admission.
pub fn working_rows(
    tp: Tp,
    sequences: u32,
    drafter: Option<crate::weight_plan::Drafter>,
) -> Vec<BufferRow> {
    use crate::serving::{DRAFT_ITEMS, VERIFY_TOKENS};
    use crate::weight_plan::Drafter;
    use bundle_runtime::upload::InputLayout;
    let items = bundle_runtime::geometry::Items::new(sequences).expect("served item count");
    let slots = items.capacity();
    let hidden = tp.geometry.hidden() as u64 * 2;
    let table = tp.model_limits().max_pages() as u64 * 4 * sequences as u64;
    let verify = if drafter == Some(Drafter::Mtp) {
        DECODE_ENTRY_ROWS
    } else {
        VERIFY_TOKENS.max(slots)
    };
    let logits = LogitsLayout::new(tp, verify.max(slots), true);
    let mut rows = Vec::new();
    let mut add = |name: &str, bytes| {
        rows.push(BufferRow {
            name: name.into(),
            layout: BufferLayout::bytes(bytes),
        })
    };
    for name in ["inputs.prefill", "inputs.half0", "inputs.half1"] {
        add(
            name,
            InputLayout::within(PREFILL_TOKENS, PREFILL_TOKENS, items).end,
        );
        add(&format!("{name}.block_tables"), table);
    }
    add("inputs.decode", InputLayout::within(slots, 1, items).end);
    add("inputs.decode.block_tables", table);
    add(
        "inputs.verify",
        InputLayout::within(VERIFY_TOKENS.max(slots), 1, items).end,
    );
    add("inputs.verify.block_tables", table);
    add(
        "restore_inputs",
        4 * (slots as u64 * 4).next_multiple_of(128) + 2048 + 64,
    );
    for name in ["activations", "normalized"] {
        add(name, (TOKENS + HIDDEN_SLACK_ROWS) * hidden);
    }
    for (name, bytes) in [
        ("logits.local", logits.local),
        ("logits.prefill", logits.gathered_prefill),
        ("logits.decode", logits.gathered_decode),
        ("logits.local_scored", logits.local_scored),
        ("logits.scored", logits.gathered_scored),
    ] {
        add(name, bytes);
    }
    let rope = tp.geometry.context_tokens() as u64 * tp.geometry.rotary() as u64 * 4;
    add("rope.ga", rope);
    add("rope.swa", rope);
    add("swa.state", crate::topology::ring_state_bytes(tp, slots));
    add(
        "workspace",
        Selected::workspace_bytes(tp, items).expect("geometry workspace plan"),
    );
    match drafter {
        Some(Drafter::Mtp) => {
            let batch = DRAFT_ITEMS as u64;
            add(
                "mtp.inputs",
                InputLayout::within(PREFILL_TOKENS, PREFILL_TOKENS, items).end,
            );
            add(
                "mtp.block_tables",
                tp.model_limits().max_pages() as u64 * 4 * batch,
            );
            let ring = batch
                * tp.kv_heads(false)
                * tp.geometry.window() as u64
                * (tp.geometry.swa_head_dim() + tp.geometry.swa_value_head_dim()) as u64
                * 2;
            for name in ["mtp.ring", "mtp.before", "mtp.caught_up"] {
                add(name, ring);
            }
            add(
                "mtp.committed",
                batch * retained_capacity(tp.geometry.window() as u64) * hidden,
            );
            add("mtp.target_rows", TOKENS * hidden);
            add("mtp.chain_hidden", batch * hidden);
            add("mtp.hidden_input", TOKENS * hidden);
            add("mtp.next_tokens", batch * 4);
            add("mtp.compact_logits", batch * tp.vocab() * 2);
            add("mtp.packed", TOKENS * tp.mtp_eh_k() * 2);
            add("mtp.scores", 2 * batch * tp.ranks() as u64 * 4);
            add("mtp.ids", 2 * batch * tp.ranks() as u64 * 4);
            add("mtp.token", batch * 4);
            add("mtp.live_rows", 4);
            add("mtp.width", 4);
        }
        Some(Drafter::Dflash) => {
            let d = tp.geometry.dflash();
            add(
                "dflash.cache",
                d.num_hidden_layers as u64
                    * 2
                    * slots as u64
                    * crate::state::DFLASH_CACHE_TOKENS
                    * tp.dflash_kv_heads()
                    * d.head_dim as u64
                    * 2,
            );
            add(
                "dflash.taps",
                (TOKENS + HIDDEN_SLACK_ROWS) * tp.dflash_fc_k() * 2,
            );
            let pairs = 2 * VERIFY_TOKENS as u64 * 4;
            add("dflash.pairs", pairs);
            add("dflash.gathered", pairs * tp.ranks() as u64);
            add("dflash.gather_flags", 2 * tp.ranks() as u64 * 4);
            add("dflash.gather_epoch", 4);
            add("dflash.vocab_width", 4);
            add(
                "dflash.history",
                sequences as u64
                    * retained_capacity(d.sliding_window as u64 - 1)
                    * tp.dflash_fc_k()
                    * 2,
            );
            add(
                "dflash.inputs",
                InputLayout::within(PREFILL_TOKENS, PREFILL_TOKENS, items).end,
            );
            add("dflash.block_tables", table);
        }
        None => {}
    }
    let scratch =
        ScratchLayout::new(tp, VERIFY_TOKENS.max(slots)).expect("supported scratch layout");
    let mut index = 0;
    scratch.visit_named(&mut |_, layout| {
        rows.push(BufferRow {
            name: format!("scratch.{index}"),
            layout: *layout,
        });
        index += 1;
    });
    for row in &mut rows {
        match row.name.as_str() {
            "activations" | "normalized" => {
                row.layout = BufferLayout::new(row.layout.allocation_bytes(), TOKENS * hidden)
                    .expect("hidden span")
            }
            "dflash.taps" => {
                row.layout =
                    BufferLayout::new(row.layout.allocation_bytes(), TOKENS * tp.dflash_fc_k() * 2)
                        .expect("tap span")
            }
            _ => {}
        }
    }
    rows
}

pub fn working_row_bytes(
    tp: Tp,
    sequences: u32,
    drafter: Option<crate::weight_plan::Drafter>,
) -> u64 {
    working_rows(tp, sequences, drafter)
        .iter()
        .map(|row| row.layout.allocation_bytes())
        .sum()
}

#[cfg(test)]
mod capacity_tests {
    use super::*;
    #[test]
    fn flash_capacity_reproduces_the_shipped_values() {
        for tp in Tp::SUPPORTED {
            for sequences in [1u32, 2, 4, 8, 16, 17, 32, 33, 64, 100, 128, 256] {
                let slots = sequences.next_power_of_two();
                for drafter in [
                    None,
                    Some(crate::weight_plan::Drafter::Mtp),
                    Some(crate::weight_plan::Drafter::Dflash),
                ] {
                    if drafter == Some(crate::weight_plan::Drafter::Mtp)
                        && sequences > crate::serving::DRAFT_ITEMS
                    {
                        continue;
                    }
                    let draft = match drafter {
                        None => 0,
                        Some(crate::weight_plan::Drafter::Mtp) => 64 << 20,
                        Some(crate::weight_plan::Drafter::Dflash) => {
                            ((512 << 20) + slots as u64 * (81 << 20)) / tp.ranks() as u64
                        }
                    };
                    let expected = (2816 << 20)
                        + (2520 << 20) / tp.ranks() as u64
                        + slots.saturating_sub(32) as u64 * (1 << 19)
                        + crate::topology::ring_state_bytes(tp, slots)
                        + draft;
                    assert_eq!(
                        crate::topology::working_bytes(tp, sequences, drafter),
                        expected
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four words per rank and row and one epoch word per row cover the
    /// fused protocols of every comm bucket at every rank count.
    #[test]
    fn hidden_comm_state_is_per_row_at_every_rank_count() {
        for tp in Tp::SUPPORTED {
            for bucket in COMM_BUCKETS {
                let rows = u64::from(bucket);
                assert_eq!(
                    hidden_flag_bytes(bucket, tp).unwrap(),
                    16 * u64::from(tp.ranks()) * rows
                );
                assert_eq!(hidden_epoch_bytes(bucket, tp).unwrap(), 4 * rows);
            }
        }
    }
}

#[cfg(test)]
mod fold_counter_tests {
    #[test]
    fn fold_counters_cover_all_served_attention_heads() {
        let maximum = crate::Tp::SUPPORTED
            .into_iter()
            .flat_map(|tp| [tp.kv_heads(true), tp.kv_heads(false)])
            .max()
            .unwrap();
        assert_eq!(super::FOLD_COUNTER_KV_HEADS, maximum);
    }
}

/// Rows past a retained window that an open step can write.
pub const RETAINED_SLACK_ROWS: u64 = crate::serving::MAX_VERIFY_ROWS as u64;

/// Retained rows per slot, including the rows an open step can write.
#[must_use]
pub const fn retained_capacity(window: u64) -> u64 {
    window + RETAINED_SLACK_ROWS
}
