//! Model-owned scratch allocations for one rank.
use crate::backend::{Kernels, Selected};
use crate::contract::*;
use kernel_core::allocation::{Allocation, Traversal};

#[derive(Clone, Copy, Debug)]
pub struct Served {
    pub slots: u32,
    pub items: u32,
    pub rows: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Speculation {
    pub verify_rows: u32,
    pub drafter: bool,
}

/// Planner, communication and prefix allocations supplied by the bundle.
#[derive(Clone, Copy, Debug, Default)]
pub struct Overhead {
    pub workspace: u64,
    pub hc_inbox: u64,
    pub hc_flags: u64,
    pub hc_epoch: u64,
    pub hc_decode_inbox: u64,
    pub hc_decode_flags: u64,
    pub hc_decode_epochs: u64,
    pub gathers: u64,
    pub prefix_pages: u64,
}

kernel_core::scratch_record! {
/// The MTP drafter's own buffers. Its input rows, input norms and fc
/// outputs borrow buffers the drafter layer does not use.
#[derive(Clone, Copy, Debug)]
pub struct DrafterScratch {
    /// Target X rows each slot keeps, `[slot][RETAINED_ROWS][4][2560]`.
    pub retained: value,
    /// A phase's target X rows until commit retains them: every decode or
    /// verify row, or each prefill item's last row (prefill, decode).
    pub staged: array(2),
    /// Per drafting item, its latest draft.
    pub drafts: value,
    /// Per anchor item, retained rows as a slot's: the X of the row before
    /// the anchor, at its position (`[ANCHOR_ITEMS][RETAINED_ROWS][4][2560]`).
    pub anchor_retained: value,
    /// This rank's best logit of each draft row, then every rank's.
    pub best: value,
    pub bests: value,
}
}

kernel_core::scratch_record! {
/// A phase's PLE staging: every item's state before the step, and after it.
#[derive(Clone, Copy, Debug)]
pub struct PleStaging {
    pub input: value,
    pub output: value,
}
}

kernel_core::scratch_record! {
/// One rank's allocation fields, in declaration order.
#[derive(Clone, Debug)]
pub struct Scratch {
    /// The four-stream residual, twice: the all-reduce writes the other.
    pub x: array(2),
    /// The grouped norm of X the next sublayer (or PLE) reads.
    pub xn: value,
    pub hc_down: value,
    pub hc_bottleneck: value,
    /// Inject logits, [rows, 4].
    pub hc_inject: value,
    /// The up projection; PLE borrows it for its key and conv input.
    pub hc_up: value,
    /// The sublayer input `mix(norm(X))`, H wide; after the mixer, the head
    /// input.
    pub u: value,
    /// This rank's partial of the sublayer output.
    pub partial: value,
    pub embed: value,
    pub lin_qkv: value,
    pub lin_z: value,
    pub lin_ab: value,
    pub lin_conv: value,
    pub lin_core: value,
    pub lin_normed: value,
    /// Packed q+gate, K and V rows.
    pub attn_qkv: value,
    pub attn_index: value,
    pub attn_q: value,
    pub index_q: value,
    pub attn_out: value,
    pub attn_gated: value,
    pub scores: value,
    pub selected: value,
    pub router_logits: value,
    pub topk_ids: value,
    pub topk_weights: value,
    pub expert_counts: value,
    pub expert_offsets: value,
    pub sorted_rows: value,
    pub padded_rows: value,
    pub active_experts: value,
    pub route_tiles: value,
    pub moe_intermediate: value,
    pub moe_routed: value,
    pub shared_intermediate: value,
    pub shared_out: value,
    /// The engine ABI's slots this program leaves unused.
    pub unused: value,
    pub ngram_local: value,
    pub ngram_rows: value,
    pub ngram: value,
    pub ple_value: value,
    pub ple_key_normed: value,
    pub ple_gated: value,
    pub ple_conv: value,
    /// Prefill and decode staging.
    pub ple_staging: records(PleStaging, 2),
    /// Rows the head reads: at most one per item, or every scored prompt
    /// row; sized for the fragment prefill head's row capacity.
    pub head_rows: value,
    /// This rank's F32 decode and verify logits shard, or the drafter's
    /// BF16 rows.
    pub local_logits: value,
    /// F32 logits of prefill rows, scored or not, bounded by SCORED_ROWS:
    /// this rank's shard, and the gathered full-vocabulary rows on rank 0.
    pub local_scored: value,
    pub gathered_scored: value,
    /// Full-vocabulary F32 decode and verify rows on rank 0.
    pub gathered_decode: value,
    pub gemm_semaphores: value,
    pub workspace: value,
    pub hc_inbox: value,
    pub hc_flags: value,
    pub hc_epoch: value,
    /// The decode all-reduce's own comm state.
    pub hc_decode_inbox: value,
    pub hc_decode_flags: value,
    pub hc_decode_epochs: value,
    /// Packed flag and epoch sets; runtime derives the gather subspans.
    pub gathers: value,
    /// F32 `[position][pair][cos, sin]` for the geometry's rotary dimensions.
    pub cos_sin: value,
    /// Linear state pools, [layer][entry][state].
    pub conv_pool: value,
    pub delta_pool: value,
    /// Committed PLE entries and the writable verify workspace.
    pub ple_pool: value,
    /// Per anchor item, the PLE state at its anchor (`[ANCHOR_ITEMS]` entries).
    pub anchor_ple: value,
    /// Per anchor item, the QSA raw-key ring its sequence would hold ending
    /// at its anchor: the live ring's layout with `ANCHOR_ITEMS` slots.
    pub anchor_ring: value,
    pub drafter: optional(DrafterScratch),
    /// A prefix snapshot's page IDs, uploaded for its copy kernel.
    pub prefix_pages: value,
    /// Dedicated last-CTA counter, reset by each completed parallel route.
    pub route_ticket: value,
    /// Temporary BF16 rows for explicit prepared projection layout copies.
    /// Legacy fusions need no additional allocation.
    pub precision_temporary: option,
}
}

/// Bytes the QSA score rows take at context-parallel size
/// `context_parallel`: every complete block of the context for each of a
/// selection piece's rows at size 1. Above 1 the group's head-sum selection
/// replaces QSA's scored one and no launch reads the rows, so the buffer
/// keeps one 256-byte placeholder.
pub fn score_bytes(shape: Shape, context_parallel: u32) -> u64 {
    if context_parallel > 1 {
        return 256;
    }
    u64::from(SCORE_ROWS)
        * (shape.context_tokens() / u64::from(shape.qsa_block())).next_multiple_of(64)
        * 4
}

/// Physical and visible bytes retain the current overlap backing spans, at
/// context-parallel size `context_parallel` (1 without groups).
pub fn allocations(
    shape: Shape,
    served: Served,
    speculation: Speculation,
    overhead: Overhead,
    overlap: bool,
    context_parallel: u32,
) -> Scratch<Allocation> {
    let p = u64::from(PREFILL_TOKENS);
    let b = u64::from(DRAFT_ITEMS);
    let wide = p * shape.wide() as u64 * BF16_BYTES;
    let hidden = p * shape.hidden() as u64 * BF16_BYTES;
    let routed = p * u64::from(shape.top_k());
    let sorted = u64::from(
        <Selected as Kernels>::expert_route_rows(routed as u32, shape.local_experts)
            .expect("MoE capacity"),
    );
    let local_experts = u64::from(shape.local_experts);
    let ple = u64::from(served.items) * shape.ple_entry_bytes();
    let rows = u64::from(served.rows);
    let pool = u64::from(served.slots)
        + if speculation.verify_rows > 1 {
            u64::from(served.slots.min(DRAFT_ITEMS)) * u64::from(speculation.verify_rows)
        } else {
            0
        }
        + u64::from(ANCHOR_ITEMS);
    let copies = if overlap { 2 } else { 1 };
    Scratch {
        x: [Allocation::bytes(wide * copies); 2],
        xn: Allocation::bytes((wide) * copies),
        hc_down: Allocation::bytes(p * shape.hc_rank() as u64 * BF16_BYTES),
        hc_bottleneck: Allocation::bytes(p * shape.hc_rank() as u64 * BF16_BYTES),
        hc_inject: Allocation::bytes((p * shape.streams() as u64 * BF16_BYTES) * copies),
        hc_up: Allocation::bytes(wide),
        u: Allocation::bytes((hidden) * copies),
        partial: Allocation::bytes((hidden) * copies),
        embed: Allocation::bytes(hidden),
        lin_qkv: Allocation::bytes(p * shape.linear_qkv as u64 * BF16_BYTES),
        lin_z: Allocation::bytes(p * shape.linear_value as u64 * BF16_BYTES),
        lin_ab: Allocation::bytes(p * shape.linear_ab_stride as u64 * BF16_BYTES),
        lin_conv: Allocation::bytes(p * shape.linear_qkv as u64 * BF16_BYTES),
        lin_core: Allocation::bytes(p * shape.linear_value as u64 * BF16_BYTES),
        lin_normed: Allocation::bytes(p * shape.linear_value as u64 * BF16_BYTES),
        attn_qkv: Allocation::bytes(p * shape.attn_row as u64 * BF16_BYTES),
        attn_index: Allocation::bytes(p * shape.index_row() as u64 * BF16_BYTES),
        attn_q: Allocation::bytes(p * shape.attn_value as u64 * BF16_BYTES),
        index_q: Allocation::bytes(
            p * (shape.index_heads() * shape.index_dim()) as u64 * BF16_BYTES,
        ),
        attn_out: Allocation::bytes(p * shape.attn_value as u64 * BF16_BYTES),
        attn_gated: Allocation::bytes(p * shape.attn_value as u64 * BF16_BYTES),
        scores: Allocation::bytes(score_bytes(shape, context_parallel)),
        selected: Allocation::bytes(p * shape.top_blocks() as u64 * 4),
        router_logits: Allocation::bytes(p * shape.experts() as u64 * 4),
        topk_ids: Allocation::bytes(routed * 4),
        topk_weights: Allocation::bytes(routed * 4),
        expert_counts: Allocation::bytes(local_experts * 4),
        expert_offsets: Allocation::bytes((local_experts + 1) * 4),
        sorted_rows: Allocation::bytes(sorted * 4),
        padded_rows: Allocation::bytes(4),
        active_experts: Allocation::bytes((local_experts + 1) * 4),
        route_tiles: Allocation::bytes(routed.div_ceil(256) * local_experts * 4 * 2),
        moe_intermediate: Allocation::bytes(
            routed * shape.expert_intermediate() as u64 * BF16_BYTES,
        ),
        moe_routed: Allocation::bytes(routed * shape.hidden() as u64 * BF16_BYTES),
        shared_intermediate: Allocation::bytes(p * shape.shared_intermediate as u64 * BF16_BYTES),
        shared_out: Allocation::bytes(hidden),
        unused: Allocation::bytes(256),
        ngram_local: Allocation::bytes(p * shape.ngram_local as u64 * BF16_BYTES),
        ngram_rows: Allocation::bytes(p * shape.ngram_local_heads as u64 * 8),
        ngram: Allocation::bytes(
            p * u64::from(shape.hidden().max(shape.ple_embed_dim())) * BF16_BYTES,
        ),
        ple_value: Allocation::bytes(hidden),
        ple_key_normed: Allocation::bytes(wide),
        ple_gated: Allocation::bytes(wide),
        ple_conv: Allocation::bytes(wide),
        ple_staging: [PleStaging {
            input: Allocation::bytes(ple),
            output: Allocation::bytes(ple),
        }; 2],
        head_rows: Allocation::bytes(hidden),
        local_logits: Allocation::bytes(b * shape.vocab_per_rank as u64 * 4),
        local_scored: Allocation::bytes(SCORED_ROWS as u64 * shape.vocab_per_rank as u64 * 4),
        gathered_scored: Allocation::bytes(SCORED_ROWS as u64 * shape.vocab() as u64 * 4),
        gathered_decode: Allocation::bytes(rows * shape.vocab() as u64 * 4),
        gemm_semaphores: Allocation::bytes(4096),
        workspace: Allocation::bytes(overhead.workspace),
        hc_inbox: Allocation::bytes(overhead.hc_inbox),
        hc_flags: Allocation::bytes(overhead.hc_flags),
        hc_epoch: Allocation::bytes(overhead.hc_epoch),
        hc_decode_inbox: Allocation::bytes(overhead.hc_decode_inbox),
        hc_decode_flags: Allocation::bytes(overhead.hc_decode_flags),
        hc_decode_epochs: Allocation::bytes(overhead.hc_decode_epochs),
        gathers: Allocation::bytes(overhead.gathers),
        cos_sin: Allocation::bytes(shape.context_tokens() * u64::from(shape.rotary() / 2) * 2 * 4),
        conv_pool: Allocation::bytes(shape.linear_layers() * pool * shape.conv_entry_bytes),
        delta_pool: Allocation::bytes(shape.linear_layers() * pool * shape.delta_entry_bytes),
        ple_pool: Allocation::bytes(
            (u64::from(served.slots)
                + if speculation.verify_rows > 0 {
                    u64::from(served.items) * u64::from(MAX_VERIFY_ROWS)
                } else {
                    0
                })
                * shape.ple_entry_bytes(),
        ),
        anchor_ple: Allocation::bytes(u64::from(ANCHOR_ITEMS) * shape.ple_entry_bytes()),
        anchor_ring: Allocation::bytes(
            shape
                .ring(shape.full_layers_mtp(), u64::from(ANCHOR_ITEMS))
                .bytes(),
        ),
        drafter: speculation.drafter.then(|| DrafterScratch {
            retained: Allocation::bytes(
                u64::from(served.slots)
                    * u64::from(MAX_VERIFY_ROWS)
                    * (u64::from(shape.wide()) * BF16_BYTES),
            ),
            staged: [
                Allocation::bytes(u64::from(served.items) * (u64::from(shape.wide()) * BF16_BYTES)),
                Allocation::bytes(u64::from(served.rows) * (u64::from(shape.wide()) * BF16_BYTES)),
            ],
            drafts: Allocation::bytes(b * 4),
            anchor_retained: Allocation::bytes(
                u64::from(ANCHOR_ITEMS)
                    * u64::from(MAX_VERIFY_ROWS)
                    * (u64::from(shape.wide()) * BF16_BYTES),
            ),
            best: Allocation::bytes(b * 16),
            bests: Allocation::bytes(b * 16 * u64::from(shape.ranks())),
        }),
        prefix_pages: Allocation::bytes(overhead.prefix_pages),
        route_ticket: Allocation::bytes(<Selected as Kernels>::ROUTE_TICKET_BYTES),
        precision_temporary: None,
    }
}

/// The existing diagnostic names for declaration paths.
pub fn allocation_name(path: &str) -> &str {
    match path {
        "x[0]" => "x0",
        "x[1]" => "x1",
        "ple_staging[0].input" => "ple_prefill_in",
        "ple_staging[0].output" => "ple_prefill_out",
        "ple_staging[1].input" => "ple_decode_in",
        "ple_staging[1].output" => "ple_decode_out",
        "drafter.staged[0]" => "staged_prefill",
        "drafter.staged[1]" => "staged_decode",
        _ => path.strip_prefix("drafter.").unwrap_or(path),
    }
}

/// Named allocation rows derived from the one scratch declaration.
pub fn rows(
    shape: Shape,
    served: Served,
    speculation: Speculation,
    overhead: Overhead,
) -> Vec<(String, u64)> {
    let mut rows = Vec::new();
    allocations(shape, served, speculation, overhead, false, 1).visit_named(
        &mut |name, allocation| {
            rows.push((
                allocation_name(name).to_owned(),
                allocation.allocation_bytes(),
            ));
        },
    );
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_row_holds_the_larger_embedding_width() {
        static GEOMETRY: crate::geometry::Geometry = {
            let mut g = crate::geometry::FLASH;
            g.shape.ple.embed_dim = g.shape.hidden * 2;
            g
        };
        let shape = Shape::new(&GEOMETRY, Ranks::TP4);
        let rows = rows(
            shape,
            Served {
                slots: 1,
                items: 1,
                rows: 1,
            },
            Speculation {
                verify_rows: 0,
                drafter: false,
            },
            Overhead::default(),
        );
        let bytes = rows.iter().find(|(name, _)| *name == "ngram").unwrap().1;
        assert_eq!(
            bytes,
            u64::from(PREFILL_TOKENS) * u64::from(shape.ple_embed_dim()) * BF16_BYTES
        );
    }
}
