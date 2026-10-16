//! Packed ring decode: the grouped-MoE template's decode schedule over the
//! packed row-major NVFP4 and MXFP8 banks (`cuda/moe/grouped_packed.cuh`).
//!
//! A packed grouped operation selects it with knobs whose `tile_k` is
//! [`PACKED_RING_TILE_K`]: `tile_n` is the CTA's output tile (16 columns per
//! strip), `warps` one per strip or, for gate/up with split parts, two,
//! and `stages` the per-warp ring depth in K64 steps. The knobs come from
//! `packed_ring` rows of a model's schedule table ([`packed_ring_knobs`]). The weight bank, the
//! scale tiles, the entry ABI and the tensor map stay the packed engine's,
//! and every accumulator adds the engine's K16 products in the same order,
//! so the outputs equal the engine's bit for bit.

use super::*;

/// The ring's K step, which marks packed ring knobs.
pub const PACKED_RING_TILE_K: u16 = 64;
/// Marks packed staged knobs (`packed_staged_decode`, one token): `warps`
/// is the consumer warp plus the producers, `stages` the producer ring
/// depth and `swizzle` the depth of the converted-step ring.
pub const PACKED_STAGED_TILE_K: u16 = 128;

/// Most warps a packed ring or staged CTA may have.
pub const PACKED_MAX_WARPS: u32 = 16;

/// Dynamic shared memory a CTA may opt into on SM120.
const SHARED_LIMIT: u32 = 101_376;

/// A packed ring schedule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedRing {
    /// 16-column output strips per CTA.
    pub strips: u16,
    /// Gate/up: the gate and up parts of a strip run on separate warps.
    pub split_parts: bool,
    /// Ring slots per warp, in K64 steps.
    pub depth: u8,
    /// L2 prefetch size of each weight copy in bytes (0, 64, 128 or 256).
    pub prefetch: u16,
}

impl PackedRing {
    /// Warps of the ring's CTA, computed without narrowing.
    pub fn warps(self) -> u32 {
        u32::from(self.strips) * if self.split_parts { 2 } else { 1 }
    }

    /// The knobs, or `None` when the strip count or CTA warps are out of
    /// range; nothing is narrowed or wrapped.
    pub fn checked_knobs(self) -> Option<CompileKnobs> {
        (self.strips > 0 && self.warps() <= PACKED_MAX_WARPS).then(|| self.knobs())
    }

    /// The knobs. Out-of-range counts saturate, so validation refuses them
    /// instead of reading a wrapped, legal-looking schedule.
    pub fn knobs(self) -> CompileKnobs {
        CompileKnobs {
            tile_m: 16,
            tile_n: self.strips.saturating_mul(16),
            tile_k: PACKED_RING_TILE_K,
            stages: self.depth,
            // The ring has no swizzle; the field carries the prefetch size.
            swizzle: u8::try_from(self.prefetch / 64).unwrap_or(u8::MAX),
            warps: u8::try_from(self.warps()).unwrap_or(u8::MAX),
            split_k_capacity: 1,
        }
    }

    pub fn of(knobs: CompileKnobs) -> Self {
        let strips = knobs.tile_n / 16;
        Self {
            strips,
            split_parts: u16::from(knobs.warps) == 2 * strips,
            depth: knobs.stages,
            prefetch: u16::from(knobs.swizzle) * 64,
        }
    }
}

/// A packed staged schedule (one token).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedStaged {
    /// Warps that convert operands; one more warp issues the MMAs.
    pub producers: u8,
    /// Producer ring slots, in K64 steps.
    pub depth: u8,
    /// Converted-step ring slots, a multiple of the producers.
    pub stages: u8,
}

impl PackedStaged {
    /// The knobs, or `None` when the producers and the consumer exceed the
    /// CTA's warps or there is no producer.
    pub fn checked_knobs(self) -> Option<CompileKnobs> {
        (self.producers > 0 && u32::from(self.producers) < PACKED_MAX_WARPS).then(|| self.knobs())
    }

    /// The knobs; an out-of-range producer count saturates, so validation
    /// refuses it.
    pub fn knobs(self) -> CompileKnobs {
        CompileKnobs {
            tile_m: 16,
            tile_n: 16,
            tile_k: PACKED_STAGED_TILE_K,
            stages: self.depth,
            swizzle: self.stages,
            warps: self.producers.saturating_add(1),
            split_k_capacity: 1,
        }
    }

    pub fn of(knobs: CompileKnobs) -> Self {
        Self {
            producers: knobs.warps.saturating_sub(1),
            depth: knobs.stages,
            stages: knobs.swizzle,
        }
    }
}

/// One K64 step's converted A fragments of one part.
const STAGED_PART_BYTES: u32 = 4 * 32 * 16;

fn staged_shared_bytes(weight: WeightFormat, gate_up: bool, k: u32, staged: PackedStaged) -> u32 {
    let parts = if gate_up { 2 } else { 1 };
    u32::from(staged.producers) * u32::from(staged.depth) * parts * weight_part_bytes(weight)
        + u32::from(staged.stages) * (parts * STAGED_PART_BYTES + 16)
        + k * 2
}

fn weight_part_bytes(weight: WeightFormat) -> u32 {
    if weight == WeightFormat::Nvfp4 {
        512 + 64
    } else {
        1024 + 32
    }
}

fn slot_bytes(weight: WeightFormat, parts: u32, tokens_capacity: u32) -> u32 {
    parts * weight_part_bytes(weight)
        + if tokens_capacity > 1 {
            tokens_capacity * (128 + 16)
        } else {
            0
        }
}

/// Dynamic shared memory of a ring: each warp's ring, then at one token the
/// CTA's single BF16 row.
fn shared_bytes(
    weight: WeightFormat,
    gate_up: bool,
    k: u32,
    tokens_capacity: u32,
    ring: PackedRing,
) -> u32 {
    let parts = if gate_up && !ring.split_parts { 2 } else { 1 };
    let warps = u32::from(ring.knobs().warps);
    warps * u32::from(ring.depth) * slot_bytes(weight, parts, tokens_capacity)
        + if tokens_capacity == 1 { k * 2 } else { 0 }
}

/// The packed ring of a decode projection from its row of the schedule
/// `table` (`template::packed_key`), or `None` where the
/// packed engine stays: other formats, activations or phases, and shapes
/// or buckets without a row. A row the build cannot use (see
/// `template::validate_packed_row`) is a build error, not a fallback.
pub fn packed_ring_knobs(
    table: &kernel_build::tuning::Table,
    shape: Shape,
    projection: Projection,
    phase: Phase,
    tokens_capacity: u32,
    weight: WeightFormat,
    activation: ActivationFormat,
) -> Option<CompileKnobs> {
    if phase != Phase::Decode
        || activation != ActivationFormat::Bf16
        || !matches!(weight, WeightFormat::Nvfp4 | WeightFormat::Mxfp8)
    {
        return None;
    }
    let key = template::packed_key(shape, projection, tokens_capacity, weight);
    let row = table.get(&key)?;
    Some(
        template::validate_packed_row(shape, projection, tokens_capacity, weight, row)
            .unwrap_or_else(|e| panic!("{e}")),
    )
}

/// Packed decode knobs that select the ring.
pub(super) fn selects_ring(
    query: &GemmQuery,
    phase: Phase,
    knobs: CompileKnobs,
    layout: WeightLayout,
) -> bool {
    layout == WeightLayout::Packed
        && phase == Phase::Decode
        && query.activation == ActivationFormat::Bf16
        && matches!(query.weight, WeightFormat::Nvfp4 | WeightFormat::Mxfp8)
        && matches!(knobs.tile_k, PACKED_RING_TILE_K | PACKED_STAGED_TILE_K)
}

fn validate(
    weight: WeightFormat,
    gate_up: bool,
    out_n: u32,
    k: u32,
    tokens_capacity: u32,
    knobs: CompileKnobs,
) -> Result<(), Refusal> {
    if knobs.tile_k == PACKED_STAGED_TILE_K {
        return validate_staged(weight, gate_up, out_n, k, tokens_capacity, knobs);
    }
    let ring = PackedRing::of(knobs);
    let warps = u16::from(knobs.warps);
    if knobs.tile_m != 16
        || knobs.tile_n == 0
        || knobs.tile_n % 16 != 0
        || out_n % u32::from(knobs.tile_n) != 0
        || k % 64 != 0
        || knobs.split_k_capacity != 1
        || !matches!(knobs.swizzle, 0 | 1 | 2 | 4)
        || knobs.stages == 0
        || u32::from(knobs.stages) > k / 64
        || !(warps == ring.strips || gate_up && warps == 2 * ring.strips)
        || u32::from(warps) > PACKED_MAX_WARPS
        || tokens_capacity == 0
        || tokens_capacity > 32
    {
        return Err(Refusal("unsupported packed ring decode schedule"));
    }
    if shared_bytes(weight, gate_up, k, tokens_capacity, ring) > SHARED_LIMIT {
        return Err(Refusal("packed ring decode exceeds SM120 shared memory"));
    }
    Ok(())
}

fn validate_staged(
    weight: WeightFormat,
    gate_up: bool,
    out_n: u32,
    k: u32,
    tokens_capacity: u32,
    knobs: CompileKnobs,
) -> Result<(), Refusal> {
    let staged = PackedStaged::of(knobs);
    let k_tiles = k / 64;
    let producers = u32::from(staged.producers);
    if knobs.tile_m != 16
        || knobs.tile_n != 16
        || out_n % 16 != 0
        || k % 64 != 0
        || knobs.split_k_capacity != 1
        || tokens_capacity != 1
        || producers == 0
        || u32::from(knobs.warps) > PACKED_MAX_WARPS
        || staged.depth == 0
        || u32::from(staged.depth) > k_tiles.div_ceil(producers)
        || staged.stages == 0
        || u32::from(staged.stages) > k_tiles
        || u32::from(staged.stages) % producers != 0
    {
        return Err(Refusal("unsupported packed staged decode schedule"));
    }
    if staged_shared_bytes(weight, gate_up, k, staged) > SHARED_LIMIT {
        return Err(Refusal("packed staged decode exceeds SM120 shared memory"));
    }
    Ok(())
}

impl<S: AsRef<str>> GroupedOperation<S> {
    /// The one-token staged schedule among the packed schedules.
    pub fn packed_staged(&self) -> bool {
        self.packed_ring() && self.knobs.tile_k == PACKED_STAGED_TILE_K
    }

    /// NVFP4 or MXFP8 decode with BF16 activations on the packed layout,
    /// run by the template's packed ring instead of the packed engine.
    pub fn packed_ring(&self) -> bool {
        selects_ring(&self.query, self.phase, self.knobs, self.layout)
    }

    pub(super) fn packed_ring_supported(&self) -> Result<(), Refusal> {
        if self.query.epilogues.output_quantization.is_some()
            || self.query.epilogues.swiglu == Some(SwigluColumns::Interleaved)
            || self.path != InputPath::A16Gather
        {
            return Err(Refusal(
                "packed ring decode takes BF16 rows and concatenated gate/up halves",
            ));
        }
        let gate_up = self.projection == Projection::GateUp;
        let out_n = if gate_up {
            self.query.n / 2
        } else {
            self.query.n
        };
        validate(
            self.query.weight,
            gate_up,
            out_n,
            self.query.k,
            self.tokens_capacity,
            self.knobs,
        )
    }

    /// One CTA per (active expert slot, output tile) of the bucket's most
    /// active experts; CTAs past the live items exit.
    pub(super) fn packed_ring_plan(&self) -> Result<LaunchPlan, Refusal> {
        self.packed_ring_supported()?;
        let group = self.query.grouped.expect("validated grouped GEMM");
        let gate_up = self.projection == Projection::GateUp;
        let out_n = if gate_up {
            self.query.n / 2
        } else {
            self.query.n
        };
        let ring = PackedRing::of(self.knobs);
        let dynamic_shared_bytes = if self.packed_staged() {
            staged_shared_bytes(
                self.query.weight,
                gate_up,
                self.query.k,
                PackedStaged::of(self.knobs),
            )
        } else {
            shared_bytes(
                self.query.weight,
                gate_up,
                self.query.k,
                self.tokens_capacity,
                ring,
            )
        };
        let plan = LaunchPlan {
            grid: [
                group.experts.min(self.query.m_capacity) * (out_n / u32::from(self.knobs.tile_n)),
                1,
                1,
            ],
            block: [u32::from(self.knobs.warps) * 32, 1, 1],
            dynamic_shared_bytes,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The entry body: a call of `packed_ring_decode` at the operation's
    /// shape, format and schedule.
    pub fn packed_ring_body(&self) -> String {
        let group = self.query.grouped.expect("grouped GEMM");
        let gate_up = self.projection == Projection::GateUp;
        let (hidden, intermediate) = if gate_up {
            (self.query.k, self.query.n / 2)
        } else {
            (self.query.n, self.query.k)
        };
        let ring = PackedRing::of(self.knobs);
        let expert_globals = self
            .packed_contract
            .is_some_and(|c| c.scales.weight_globals == GlobalScaleLayout::ExpertProjection);
        let weights = if self.query.weight == WeightFormat::Nvfp4 {
            "nvfp4"
        } else {
            "mxfp8"
        };
        if self.packed_staged() {
            let staged = PackedStaged::of(self.knobs);
            return format!(
                "chop::moe::grouped::packed_staged_decode<chop::moe::grouped::Weights::{weights}, chop::moe::grouped::Shape{{.experts = {}, .top_k = {}, .hidden = {hidden}, .intermediate = {intermediate}}}, {gate_up}, chop::moe::grouped::PackedStagedSchedule{{.producers = {}, .depth = {}, .stages = {}, .expert_globals = {expert_globals}}}>(activation, weight_bytes, weight_scales, weight_f32_scales, output, topk_weights, expert_offsets, sorted_rows, num_padded_rows, live_tokens, active_experts);",
                group.experts, group.top_k, staged.producers, staged.depth, staged.stages,
            );
        }
        format!(
            "chop::moe::grouped::packed_ring_decode<chop::moe::grouped::Weights::{}, chop::moe::grouped::Shape{{.experts = {}, .top_k = {}, .hidden = {hidden}, .intermediate = {intermediate}}}, {gate_up}, {}, chop::moe::grouped::PackedRingSchedule{{.strips = {}, .depth = {}, .split_parts = {}, .expert_globals = {expert_globals}{}}}>(activation, weight_bytes, weight_scales, weight_f32_scales, output, topk_weights, expert_offsets, sorted_rows, num_padded_rows, live_tokens, active_experts);",
            if self.query.weight == WeightFormat::Nvfp4 {
                "nvfp4"
            } else {
                "mxfp8"
            },
            group.experts,
            group.top_k,
            self.tokens_capacity,
            ring.strips,
            ring.depth,
            ring.split_parts,
            if ring.prefetch == 0 {
                String::new()
            } else {
                format!(", .prefetch = {}", ring.prefetch)
            },
        )
    }
}
