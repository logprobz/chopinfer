//! Capacity planning by formula: the bytes one rank holds besides its KV
//! pool, and the KV tokens the serving pool then gets.
//!
//! The host sizes the serving KV pool from the device memory left after the
//! bundle's weights, workspaces and request state (docs/numerics.md#kv-offload-and-prefix-state):
//! create allocates the recurrent state before the pool
//! (docs/numerics.md#recurrent-state, `chop.recurrent-state.preallocated.v1`),
//! and the host subtracts a 1 GiB reserve and divides the rest by the page
//! bytes. The plan conservatively counts 90 % of that rest. Each slot
//! holds its committed KDA state; verify checkpoints share the total
//! verify-row budget. The indexer retains its fixed per-bucket-item tail
//! checkpoints, and the [`ANCHOR_ITEMS`] spares hold one state copy each.
//!
//! The weights are those of the checkpoint's quantization layout: NVFP4
//! routed experts leave about 30 GiB more per rank for state and KV pages
//! than block FP8. The text rank image holds no vision tensor; a package
//! serving images also holds the encoder's replicated image and buffers on
//! each rank ([`encoder_bytes`]), which create acquires before the KV pool.
//!
//! The latent cache is replicated across the tensor-parallel ranks: every
//! rank holds the whole latent and pooled keys of each page it owns. With
//! context parallelism of size N a logical page is a stripe of one page on
//! each of N ranks, so the token capacity scales with N. The recurrent state
//! is per rank (KDA heads split across ranks) and does not.
//!
//! Workspaces are named by the engine buffers that dominate them, from each
//! engine's planner formula, plus [`WORKSPACE_ALLOWANCE`] for the buffers the
//! bundle has not listed yet. The bundle program replaces the allowance with
//! its buffer list; until then capacities here are planning estimates, not
//! measurements.
use crate::backend::{Kernels, Selected};
use crate::catalog::checkpoint_catalog;
use crate::contract::ANCHOR_ITEMS;
use crate::geometry::{Geometry, LayerKind};
use crate::quantization::{FP8_BLOCK128, Layout};
use crate::weight_plan::{GroupBytes, plan};
use kernel_core::contract::{DType, RecurrentPhase};
use kernel_core::gated_pooling::GatedGroupPooling;
use kernel_core::latent_attention::LatentKvFormat;
use kernel_core::per_channel_decay_delta_rule::{
    DELTA_RULE_CHUNK, DecayGate, PerChannelDecayDeltaRuleQuery, QkvConvs,
};
use kernel_core::pooled_indexer::{IndexerKeyFormat, PooledIndexerQuery};

const GIB: u64 = 1 << 30;

/// Tokens of one KV page on one rank; also the prefill granule (KDA chunks
/// start at absolute multiples of 64).
pub const PAGE_TOKENS: u32 = DELTA_RULE_CHUNK;

/// Rows of one prefill launch (the MoE, latent attention, mHC and KDA
/// engines' prefill capacity).
pub const PREFILL_TOKENS: u32 = 8192;

/// Sequences one KDA prefill launch may hold; each adds up to two chunks of
/// workspace (a partial first and last chunk).
pub const PREFILL_SEQUENCES: u32 = 256;

/// Rows of one indexer prefill launch; a longer prefill runs several.
pub const INDEXER_PREFILL_ROWS: u32 = 2048;

/// Decode and verify rows of the latent attention engine's largest bucket.
pub const DECODE_ROWS: u32 = 256;

/// Most draft tokens a verify launch checks: verify runs `draft + 1` rows,
/// and the indexer and mHC verify entries hold 6 rows per sequence.
pub const MAX_DRAFT_TOKENS: u32 = 5;

/// Multiprocessors whose double the split rules of the latent attention and
/// indexer engines fill.
const SPLIT_TARGET: u64 = 2 * 188;

/// Most splits of one decode or verify row.
const MAX_SPLITS: u64 = 64;

/// Unnamed buffers until the bundle lists them: projection activations,
/// logits and sampling, communication buffers, prefix snapshot staging and
/// allocation padding. Hand-picked; the qwen38 bundle's measured unnamed
/// working set is in the same range.
pub const WORKSPACE_ALLOWANCE: u64 = 2 * GIB;

/// What the host leaves for the KV pool on a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Device {
    /// Device memory a rank may use: the device less what the driver and
    /// CUDA context keep.
    pub bytes: u64,
    /// Free memory the host keeps before sizing the KV pool.
    pub reserve: u64,
    /// Share of the rest the plan counts for the KV pool: a conservative
    /// planning margin, below the host's default of all of it.
    pub fraction_millionths: u32,
}

/// A conservative 94 GiB usable per 96 GB device, the qwen38 package
/// plan's default (the board reports 97,887 MiB; GLM's free memory after
/// the driver and CUDA context is not measured yet), with the host's
/// reserve and a 90 % planning share.
pub const DEVICE: Device = Device {
    bytes: model_compiler_sdk::device::DEVICE_BYTES,
    reserve: model_compiler_sdk::device::HOST_RESERVE_BYTES,
    fraction_millionths: 900_000,
};

/// One serving configuration to plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Serving {
    /// Tensor-parallel ranks.
    pub ranks: u32,
    /// Context-parallel group size, dividing `ranks`.
    pub context_parallel: u32,
    /// Sequences served at once (`--max-sequences`): one state slot each.
    pub slots: u32,
    /// MTP draft tokens per step; 0 serves without the drafter.
    pub draft_tokens: u32,
    pub kv: LatentKvFormat,
    /// Whether the package serves images (`--modalities` with image, or
    /// `auto` on a checkpoint with a vision tower): each rank then also
    /// holds [`encoder_bytes`].
    pub vision: bool,
}

/// Named workspace terms of one rank, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Workspace {
    /// KDA prefill chunk workspace: `(ceil(T / 64) + 2 * sequences) *
    /// heads * (4 * 64 * d + 64 * 64 + d) * 4`.
    pub delta_rule: u64,
    /// MoE down-projection output, F32 per route: `T * top_k * hidden * 4`.
    pub moe_routed: u64,
    /// MoE activated gate/up output of the rank's slice of every expert,
    /// BF16 per route: `T * top_k * intermediate / ranks * 2`.
    pub moe_activated: u64,
    /// The MoE sublayers' all-reduce state of F32 partials (a state of its
    /// own beside the BF16 partials', which the allowance holds), by the
    /// backend's formulas (`Kernels::moe_all_reduce_state_bytes`): the
    /// low-latency round entry's inbox and epochs, and the phased entries'
    /// inbox (the prefill chunk's F32 rows, `T * hidden * 4`, and the
    /// one-shot region), flags and epoch.
    pub moe_all_reduce: u64,
    /// Two hyper-connection residuals (the output never aliases the input):
    /// `2 * T * streams * hidden * 2`.
    pub hyper_connection: u64,
    /// Key lists `T * (top_k + pool - 1) * 4`, the absorbed queries and the
    /// latent outputs `2 * T * heads * latent * 2`.
    pub latent: u64,
    /// Split-row partials of decode and verify, at the largest bucket.
    pub latent_splits: u64,
    /// Streaming top-k candidates `rows * splits * kept * 8`, at most over
    /// the indexer's prefill rows.
    pub indexer: u64,
    /// Context parallel only: the group's buffers on one rank
    /// (`context_parallel::RankBytes`), dominated by the partial blocks it
    /// sends and receives, `size` blocks of 2048 rows each.
    pub context_parallel: u64,
    pub allowance: u64,
}

impl Workspace {
    pub fn total(&self) -> u64 {
        self.delta_rule
            + self.moe_routed
            + self.moe_activated
            + self.moe_all_reduce
            + self.hyper_connection
            + self.latent
            + self.latent_splits
            + self.indexer
            + self.context_parallel
            + self.allowance
    }
}

/// One rank's plan for a serving configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capacity {
    /// The text rank image: no vision tensor, MTP weights when drafting.
    pub weights: u64,
    /// With images, the encoder's image and buffers ([`encoder_bytes`]).
    pub encoder: u64,
    pub workspace: Workspace,
    /// One copy of a slot's recurrent state: KDA delta and conv state of
    /// every linear layer, and the indexer tail of every sparse layer.
    pub state_per_slot: u64,
    /// KDA checkpoint rows shared by all slots, zero without speculation.
    pub checkpoint_rows: u32,
    /// The anchor spares, one copy for each of the [`ANCHOR_ITEMS`]
    /// prompt-ending items a step keeps at its anchor:
    /// `ANCHOR_ITEMS * state_per_slot`.
    pub anchors: u64,
    /// The state create allocates before the KV pool:
    /// Committed KDA slots, packed verify checkpoints, fixed indexer tails
    /// and anchor spares. Retained drafter rows remain in the allowance.
    pub state: u64,
    /// Bytes of one rank's page.
    pub page_bytes: u64,
    /// The KV pool's budget on each rank.
    pub kv_bytes: u64,
    /// Pages per rank (stripes of the logical pool).
    pub pages: u64,
    /// Token capacity counted across the context-parallel group.
    pub kv_tokens: u64,
}

/// Weights of each rank's memory groups, computed once per geometry and
/// rank count.
#[derive(Clone, Debug)]
pub struct Planner<'g> {
    geometry: &'g Geometry,
    layout: &'static Layout,
    ranks: u32,
    weights: Vec<GroupBytes>,
}

fn state_queries(g: &Geometry, ranks: u32) -> [PerChannelDecayDeltaRuleQuery; 2] {
    let s = &g.shape.linear;
    let heads = s.heads / ranks;
    let phase = RecurrentPhase::Decode;
    [
        PerChannelDecayDeltaRuleQuery::DeltaRule {
            heads,
            head_dim: s.head_dim,
            gate: DecayGate::BoundedSigmoid {
                lower_bound_bits: (s.gate_lower_bound as f32).to_bits(),
            },
            phase,
            capacity: 1,
        },
        PerChannelDecayDeltaRuleQuery::ShortConv {
            heads,
            head_dim: s.head_dim,
            kernel: s.conv_taps,
            convs: QkvConvs::Separate,
            phase,
            capacity: 1,
        },
    ]
}

fn pooling(g: &Geometry) -> GatedGroupPooling {
    GatedGroupPooling {
        group: g.shape.indexer.pool,
        width: g.shape.indexer.dim,
        position_dtype: DType::Bf16,
    }
}

/// Splits of a decode or verify launch of `rows` row-units: enough to fill
/// twice the multiprocessors, a power of two, at most 64.
fn splits(units: u64) -> u64 {
    SPLIT_TARGET
        .div_ceil(units.max(1))
        .next_power_of_two()
        .min(MAX_SPLITS)
}

/// Sparse attention layers that hold KV: the main layers, and the MTP
/// layers when drafting.
pub fn kv_layers(g: &Geometry, drafting: bool) -> u32 {
    g.count_layers(LayerKind::Sparse) + if drafting { g.mtp_layers() } else { 0 }
}

/// Bytes of one rank's page: the latent rows and the pooled indexer keys of
/// `PAGE_TOKENS` tokens in every KV layer.
pub fn page_bytes(g: &Geometry, kv: LatentKvFormat, drafting: bool) -> u64 {
    let pooled = PooledIndexerQuery::KeyWrite {
        pooling: pooling(g),
        key_norm_eps_bits: 1e-6_f32.to_bits(),
        page_size: PAGE_TOKENS,
        key_format: IndexerKeyFormat::Bf16,
        phase: RecurrentPhase::Prefill,
        capacity: PREFILL_TOKENS,
    }
    .pooled_page_bytes()
    .expect("a key write has pooled page bytes");
    u64::from(kv_layers(g, drafting))
        * (kv.page_bytes(g.shape.latent.kv_rank, PAGE_TOKENS) + pooled)
}

/// One copy of a slot's recurrent state on one rank.
pub fn state_per_slot(g: &Geometry, ranks: u32, drafting: bool) -> u64 {
    let linear = state_queries(g, ranks)
        .iter()
        .map(PerChannelDecayDeltaRuleQuery::state_bytes)
        .sum::<u64>();
    u64::from(g.count_layers(LayerKind::Linear)) * linear
        + u64::from(kv_layers(g, drafting)) * pooling(g).tail_state_elements() * 2
}

/// Workspace terms of one rank. A context-parallel size above eight ranks
/// panics; [`Planner::plan`] refuses one that does not divide the ranks.
pub fn workspace(g: &Geometry, ranks: u32, context_parallel: u32) -> Workspace {
    let s = &g.shape;
    let t = u64::from(PREFILL_TOKENS);
    let routes = t * u64::from(s.moe.top_k);
    let linear_heads = u64::from(s.linear.heads / ranks);
    let d = u64::from(s.linear.head_dim);
    let chunk = u64::from(DELTA_RULE_CHUNK);
    let chunks = t.div_ceil(chunk) + 2 * u64::from(PREFILL_SEQUENCES);
    let latent_heads = u64::from(s.latent.heads / ranks);
    let latent = u64::from(s.latent.kv_rank);
    let list = u64::from(g.limits.index_topk + s.indexer.pool - 1);
    let kept = u64::from(g.kept_pools());
    let partial = latent_heads * (latent + 2) * 4;
    let latent_splits = (1..=u64::from(DECODE_ROWS))
        .map(|rows| {
            let units = (rows * latent_heads).div_ceil(16);
            let splits = splits(units);
            if splits > 1 {
                rows * splits * partial
            } else {
                0
            }
        })
        .max()
        .unwrap_or(0);
    let indexer = (1..=u64::from(INDEXER_PREFILL_ROWS))
        .map(|rows| rows * splits(rows) * kept * 8)
        .max()
        .unwrap_or(0);
    Workspace {
        delta_rule: chunks * linear_heads * (4 * chunk * d + chunk * chunk + d) * 4,
        moe_routed: routes * u64::from(s.hidden) * 4,
        moe_activated: routes * u64::from(s.moe.intermediate / ranks) * 2,
        moe_all_reduce: Selected::moe_all_reduce_state_bytes(ranks, s.hidden)
            .expect("a served rank count's F32 all-reduce state"),
        hyper_connection: 2 * t * u64::from(s.hc.streams) * u64::from(s.hidden) * 2,
        latent: t * list * 4 + 2 * t * latent_heads * latent * 2,
        latent_splits,
        indexer,
        context_parallel: if context_parallel > 1 {
            crate::context_parallel::rank_bytes(
                crate::context_parallel::Dims::planned(g, ranks),
                context_parallel,
            )
            .expect("a context-parallel group of at most eight ranks")
            .total()
        } else {
            0
        },
        allowance: WORKSPACE_ALLOWANCE,
    }
}

/// What a package serving images adds to each rank: the encoder's
/// replicated image (its declaration's stacked views) and every buffer
/// create allocates for it at the declared capacity, the media arena and the
/// prefill's source rows included.
pub fn encoder_bytes() -> u64 {
    crate::encoder::views::image_bytes().expect("the encoder image layout")
        + crate::encoder::buffer_bytes()
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

impl<'g> Planner<'g> {
    /// The planner of the block-FP8 checkpoint.
    pub fn new(geometry: &'g Geometry, ranks: u32) -> Result<Self, String> {
        Self::for_checkpoint(geometry, &FP8_BLOCK128, ranks)
    }

    /// The planner of a checkpoint stored in `layout`.
    pub fn for_checkpoint(
        geometry: &'g Geometry,
        layout: &'static Layout,
        ranks: u32,
    ) -> Result<Self, String> {
        let tensors = checkpoint_catalog(geometry, layout);
        let weights = plan(geometry, &tensors, ranks)?
            .into_iter()
            .map(|rank| rank.bytes)
            .collect();
        Ok(Self {
            geometry,
            layout,
            ranks,
            weights,
        })
    }

    /// The largest rank's weights: MTP weights only when drafting, vision
    /// weights only with the vision tower.
    pub fn weights(&self, drafting: bool, vision: bool) -> u64 {
        self.weights
            .iter()
            .map(|bytes| {
                bytes.total()
                    - if drafting { 0 } else { bytes.mtp }
                    - if vision { 0 } else { bytes.vision }
            })
            .max()
            .unwrap_or(0)
    }

    /// Plan `serving` on `device`, or say why it does not fit.
    pub fn plan(&self, serving: Serving, device: Device) -> Result<Capacity, String> {
        let g = self.geometry;
        if serving.ranks != self.ranks {
            return Err(format!(
                "planned for {} ranks, asked for {}",
                self.ranks, serving.ranks
            ));
        }
        let n = serving.context_parallel;
        if n == 0 || !self.ranks.is_multiple_of(n) {
            return Err(format!(
                "context-parallel size {n} does not divide {} ranks",
                self.ranks
            ));
        }
        if serving.slots == 0 {
            return Err("a plan needs at least one slot".into());
        }
        if serving.draft_tokens > MAX_DRAFT_TOKENS {
            return Err(format!(
                "{} draft tokens exceed the verify rows: at most {MAX_DRAFT_TOKENS}",
                serving.draft_tokens
            ));
        }
        if device.fraction_millionths == 0 || device.fraction_millionths > 1_000_000 {
            return Err("the KV memory fraction must be in (0, 1]".into());
        }
        let drafting = serving.draft_tokens > 0;
        let weights = self.weights(drafting, false);
        let encoder = if serving.vision { encoder_bytes() } else { 0 };
        let workspace = workspace(g, self.ranks, n);
        let state_per_slot = state_per_slot(g, self.ranks, drafting);
        let checkpoint_rows = if drafting {
            serving
                .slots
                .min(crate::contract::DRAFT_ITEMS)
                .saturating_mul(serving.draft_tokens + 1)
                .min(crate::contract::head_rows())
        } else {
            0
        };
        let tail = u64::from(kv_layers(g, drafting)) * pooling(g).tail_state_elements() * 2;
        let kda = state_per_slot - tail;
        let bucket = serving
            .slots
            .checked_next_power_of_two()
            .ok_or("state bucket overflows")?;
        let anchors = u64::from(ANCHOR_ITEMS) * state_per_slot;
        // The host's per-item tail reservation always includes all six
        // checkpoints, even when the instance does not verify.
        let state = (u64::from(serving.slots) + u64::from(checkpoint_rows)) * kda
            + u64::from(bucket) * u64::from(crate::contract::MAX_VERIFY_ROWS + 1) * tail
            + anchors;
        let used = weights + encoder + workspace.total() + device.reserve + state;
        let free = device.bytes.checked_sub(used).ok_or_else(|| {
            format!(
                "{} at TP{} does not fit {:.2} GiB: weights {:.2} GiB, encoder {:.2} GiB, workspace {:.2} GiB, reserve {:.2} GiB and state {:.2} GiB ({} slots, {checkpoint_rows} checkpoint rows and {ANCHOR_ITEMS} anchors) leave no KV pool",
                g.id,
                self.ranks,
                gib(device.bytes),
                gib(weights),
                gib(encoder),
                gib(workspace.total()),
                gib(device.reserve),
                gib(state),
                serving.slots,
            )
        })?;
        let kv_bytes =
            (u128::from(free) * u128::from(device.fraction_millionths) / 1_000_000) as u64;
        let page_bytes = page_bytes(g, serving.kv, drafting);
        let pages = kv_bytes / page_bytes;
        if pages == 0 {
            return Err(format!(
                "{} at TP{}: {kv_bytes} bytes hold no {page_bytes}-byte KV page",
                g.id, self.ranks
            ));
        }
        Ok(Capacity {
            weights,
            encoder,
            workspace,
            state_per_slot,
            checkpoint_rows,
            anchors,
            state,
            page_bytes,
            kv_bytes,
            pages,
            kv_tokens: pages * u64::from(PAGE_TOKENS) * u64::from(n),
        })
    }

    /// The capacity table: slots 1 to 64, MTP draft tokens 0, 3 and 5, FP8
    /// and BF16 latent KV, and the KV tokens at context-parallel sizes 1, 2
    /// and 4 of a text package and of a package serving images. The state,
    /// KV and page columns are the text package's at DCP 1. A configuration
    /// that does not fit shows `-`. The title names a layout other than
    /// block FP8.
    pub fn table(&self, device: Device) -> String {
        let sizes = [1, 2, 4];
        let name = if *self.layout == FP8_BLOCK128 {
            self.geometry.id.to_owned()
        } else {
            format!("{} {}", self.geometry.id, self.layout.id)
        };
        let mut out = format!(
            "{name} TP{}, {:.2} GiB per rank; weights {:.2} GiB (AR) / {:.2} GiB (MTP); encoder {:.2} GiB with images (image {:.2} GiB, buffers {:.2} GiB); workspace {:.2} GiB (DCP 1), context-parallel buffers {:.2} / {:.2} GiB (DCP 2 / 4)\n\n\
             | slots | MTP K | KV | checkpoint rows | state GiB | KV GiB | pages | tokens DCP 1 | tokens DCP 2 | tokens DCP 4 | images DCP 1 | images DCP 2 | images DCP 4 |\n\
             |---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
            self.ranks,
            gib(device.bytes),
            gib(self.weights(false, false)),
            gib(self.weights(true, false)),
            gib(encoder_bytes()),
            gib(crate::encoder::views::image_bytes().expect("the encoder image layout")),
            gib(crate::encoder::buffer_bytes()),
            gib(workspace(self.geometry, self.ranks, 1).total()),
            gib(workspace(self.geometry, self.ranks, 2).context_parallel),
            gib(workspace(self.geometry, self.ranks, 4).context_parallel),
        );
        for slots in [1, 2, 4, 8, 16, 32, 64] {
            for draft_tokens in [0, 3, 5] {
                for (name, kv) in [
                    ("fp8", LatentKvFormat::Fp8E4M3Block128),
                    ("bf16", LatentKvFormat::Bf16),
                ] {
                    let plans = |vision| {
                        sizes.map(|context_parallel| {
                            self.plan(
                                Serving {
                                    ranks: self.ranks,
                                    context_parallel,
                                    slots,
                                    draft_tokens,
                                    kv,
                                    vision,
                                },
                                device,
                            )
                        })
                    };
                    let tokens = |plans: &[Result<Capacity, String>]| {
                        plans
                            .iter()
                            .map(|plan| {
                                plan.as_ref()
                                    .map_or_else(|_| "-".to_owned(), |c| c.kv_tokens.to_string())
                            })
                            .collect::<Vec<_>>()
                            .join(" | ")
                    };
                    let (plans, images) = (plans(false), plans(true));
                    let tokens = format!("{} | {}", tokens(&plans), tokens(&images));
                    let (copies, state, kv_gib, pages) = match &plans[0] {
                        Ok(c) => (
                            c.checkpoint_rows.to_string(),
                            format!("{:.2}", gib(c.state)),
                            format!("{:.2}", gib(c.kv_bytes)),
                            c.pages.to_string(),
                        ),
                        Err(_) => ("-".into(), "-".into(), "-".into(), "-".into()),
                    };
                    out += &format!(
                        "| {slots} | {draft_tokens} | {name} | {copies} | {state} | {kv_gib} | {pages} | {tokens} |\n"
                    );
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{FLASH, GEOMETRIES};

    /// A text package's serving configuration.
    fn serving(slots: u32, draft_tokens: u32, kv: LatentKvFormat, n: u32) -> Serving {
        Serving {
            ranks: 4,
            context_parallel: n,
            slots,
            draft_tokens,
            kv,
            vision: false,
        }
    }

    /// The same configuration of a package serving images.
    fn images(slots: u32, draft_tokens: u32, kv: LatentKvFormat, n: u32) -> Serving {
        Serving {
            vision: true,
            ..serving(slots, draft_tokens, kv, n)
        }
    }

    const FP8: LatentKvFormat = LatentKvFormat::Fp8E4M3Block128;

    #[test]
    fn flash_state_and_page_bytes_follow_the_contracts() {
        // 34 KDA layers of delta (1,048,576 B) and conv (36,864 B) state at
        // 16 heads per rank, plus a 1,536 B indexer tail per sparse layer.
        assert_eq!(state_per_slot(&FLASH, 4, false), 36_904_960 + 11 * 1536);
        assert_eq!(state_per_slot(&FLASH, 4, true), 36_904_960 + 12 * 1536);
        // 528 B (FP8) or 1024 B (BF16) of latent and 64 B of pooled keys
        // per token, in 11 sparse layers, or 12 with the MTP layer.
        assert_eq!(page_bytes(&FLASH, FP8, false), 11 * 64 * (528 + 64));
        assert_eq!(page_bytes(&FLASH, FP8, true), 12 * 64 * (528 + 64));
        assert_eq!(
            page_bytes(&FLASH, LatentKvFormat::Bf16, true),
            12 * 64 * (1024 + 64)
        );
    }

    #[test]
    fn flash_workspace_terms() {
        let w = workspace(&FLASH, 4, 1);
        // 128 + 512 chunks x 16 heads x 147,968 B.
        assert_eq!(w.delta_rule, 640 * 16 * 147_968);
        assert_eq!(w.moe_routed, GIB);
        assert_eq!(w.moe_activated, GIB / 16);
        // The phased inbox of the prefill chunk's F32 rows (128 MiB) and its
        // one-shot region (16 MiB), the phased flags and epoch (33,804 B)
        // and the low-latency round entry's inbox and epochs (262,160 B).
        assert_eq!(w.moe_all_reduce, GIB / 8 + (16 << 20) + 33_804 + 262_160);
        assert_eq!(w.hyper_connection, GIB / 2);
        assert_eq!(w.latent, 8192 * 2051 * 4 + (GIB / 4));
        assert_eq!(w.latent_splits, 187 * 4 * 16 * 514 * 4);
        assert_eq!(w.indexer, 8 << 20);
        assert_eq!(w.context_parallel, 0);
        // The group buffers of `context_parallel::RankBytes`: 0.42 GiB at
        // size 2, 0.76 GiB at size 4 (0.25 GiB of it received blocks).
        assert_eq!(workspace(&FLASH, 4, 2).context_parallel, 446_034_672);
        assert_eq!(workspace(&FLASH, 4, 4).context_parallel, 816_387_400);
    }

    #[test]
    fn flash_weights_match_the_weight_plan() {
        let planner = Planner::new(&FLASH, 4).unwrap();
        assert_eq!(planner.weights(true, true), 83_370_801_144);
        assert_eq!(planner.weights(false, true), 83_370_801_144 - 1_942_986_368);
    }

    /// Verify rows share a bounded pool. Increasing draft depth beyond
    /// that pool changes neither state nor KV capacity.
    #[test]
    fn verify_checkpoints_share_the_total_row_budget() {
        let planner = Planner::new(&FLASH, 4).unwrap();
        let ar = planner.plan(serving(32, 0, FP8, 1), DEVICE).unwrap();
        let k3 = planner.plan(serving(32, 3, FP8, 1), DEVICE).unwrap();
        let k5 = planner.plan(serving(32, 5, FP8, 1), DEVICE).unwrap();
        assert_eq!(
            (ar.checkpoint_rows, k3.checkpoint_rows, k5.checkpoint_rows),
            (0, 32, 32)
        );
        assert_eq!(k3.anchors, 16 * k3.state_per_slot);
        assert_eq!(
            k3.state,
            (32 + 32 + 16) * 36_904_960 + (32 * 7 + 16) * 12 * 1536
        );
        assert_eq!(ar.state, (32 + 16) * 36_904_960 + (32 * 7 + 16) * 11 * 1536);
        assert_eq!(k3.state, k5.state);
        assert_eq!(k3.kv_tokens, k5.kv_tokens);
        assert!(ar.kv_tokens > k3.kv_tokens);
        let small = planner.plan(serving(2, 3, FP8, 1), DEVICE).unwrap();
        assert_eq!(small.checkpoint_rows, 8);
        assert!(
            planner
                .plan(serving(32, MAX_DRAFT_TOKENS + 1, FP8, 1), DEVICE)
                .is_err()
        );
    }

    #[test]
    fn kv_capacity_scales_with_the_context_parallel_size_and_state_does_not() {
        let planner = Planner::new(&FLASH, 4).unwrap();
        let one = planner.plan(serving(32, 3, FP8, 1), DEVICE).unwrap();
        let four = planner.plan(serving(32, 3, FP8, 4), DEVICE).unwrap();
        assert_eq!(one.state, four.state);
        assert!(four.workspace.total() > one.workspace.total());
        assert_eq!(four.kv_tokens, four.pages * 64 * 4);
        assert!(four.kv_tokens > 2 * one.kv_tokens);
        assert!(
            planner
                .plan(serving(32, 3, FP8, 3), DEVICE)
                .unwrap_err()
                .contains("does not divide 4 ranks")
        );
    }

    /// The bounded verify pool holds 32 KDA checkpoints instead of 128.
    /// The fixed indexer tail reservation is counted independently.
    #[test]
    fn flash_fp8_mtp_at_32_slots_matches_the_plan_estimate() {
        let planner = Planner::new(&FLASH, 4).unwrap();
        let one = planner.plan(serving(32, 3, FP8, 1), DEVICE).unwrap();
        let two = planner.plan(serving(32, 3, FP8, 2), DEVICE).unwrap();
        let four = planner.plan(serving(32, 3, FP8, 4), DEVICE).unwrap();
        assert_eq!(one.state_per_slot, 36_923_392);
        assert_eq!(one.state, 2_956_820_480);
        assert_eq!(one.workspace.total(), 5_860_325_916);
        assert_eq!(one.page_bytes, 64 * 7104);
        assert_eq!(one.encoder, 0);
        assert_eq!((one.pages, one.kv_tokens), (17_414, 1_114_496));
        assert_eq!((two.pages, two.kv_tokens), (16_531, 2_115_968));
        assert_eq!((four.pages, four.kv_tokens), (15_798, 4_044_288));
        assert_eq!(four.workspace.total() - one.workspace.total(), 816_387_400);
    }

    /// The context-parallel buffers fit for every supported table entry
    /// that leaves KV capacity at DCP 1.
    #[test]
    fn context_parallel_buffers_fit_at_94_gib() {
        let planner = Planner::new(&FLASH, 4).unwrap();
        for slots in [1, 2, 4, 8, 16, 32, 64] {
            for draft_tokens in [0, 3, 5] {
                for kv in [FP8, LatentKvFormat::Bf16] {
                    let plan = |n| planner.plan(serving(slots, draft_tokens, kv, n), DEVICE);
                    if plan(1).is_ok() {
                        for n in [2, 4] {
                            let c = plan(n).unwrap();
                            assert!(c.pages > 0, "{slots} {draft_tokens} {kv:?} DCP {n}");
                        }
                    }
                }
            }
        }
        let tight = |n| {
            planner
                .plan(serving(32, 5, FP8, n), DEVICE)
                .unwrap()
                .kv_tokens
        };
        assert_eq!(
            (tight(1), tight(2), tight(4)),
            (1_114_496, 2_115_968, 4_044_288)
        );
    }

    /// Images still subtract the encoder's image and buffers. A deeper
    /// draft does not enlarge an already full verify checkpoint pool.
    #[test]
    fn images_hold_the_encoder_image_and_buffers() {
        let planner = Planner::new(&FLASH, 4).unwrap();
        assert_eq!(
            encoder_bytes(),
            crate::encoder::views::image_bytes().unwrap() + crate::encoder::buffer_bytes()
        );
        let text = planner.plan(serving(32, 3, FP8, 1), DEVICE).unwrap();
        let image = planner.plan(images(32, 3, FP8, 1), DEVICE).unwrap();
        assert_eq!(image.encoder, encoder_bytes());
        assert_eq!((image.weights, image.state), (text.weights, text.state));
        let free = |c: &Capacity| {
            DEVICE.bytes - c.weights - c.encoder - c.workspace.total() - DEVICE.reserve - c.state
        };
        assert_eq!(free(&text) - free(&image), encoder_bytes());
        assert_eq!(
            image.kv_bytes,
            free(&image) / 10 * 9 + free(&image) % 10 * 9 / 10
        );
        assert_eq!((image.pages, image.kv_tokens), (10_382, 664_448));
        let deeper = planner.plan(images(32, 5, FP8, 1), DEVICE).unwrap();
        assert_eq!(deeper.kv_tokens, image.kv_tokens);
    }

    #[test]
    fn a_plan_that_leaves_no_kv_pool_is_refused_by_name() {
        let planner = Planner::new(&FLASH, 4).unwrap();
        let error = planner.plan(serving(1024, 5, FP8, 1), DEVICE).unwrap_err();
        assert!(
            error.starts_with("flash at TP4 does not fit 94.00 GiB:"),
            "{error}"
        );
    }

    #[test]
    fn every_geometry_plans_at_its_served_ranks() {
        for g in GEOMETRIES {
            for &ranks in crate::kernels::SERVED_RANKS {
                let planner = Planner::new(g, ranks).unwrap();
                let table = planner.table(DEVICE);
                assert_eq!(table.lines().count(), 4 + 7 * 3 * 2, "{table}");
            }
        }
    }
}
