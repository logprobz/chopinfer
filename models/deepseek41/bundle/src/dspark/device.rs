//! Device storage for one drafter instance, allocated before the KV pool.
//! The target's scratch and collective buffers are reused only between
//! completed target invocations. Provisional features and draft KV have
//! their own storage and never alias committed request state.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use bundle_runtime::geometry::Items;
use bundle_runtime::graphs::GraphCache;
use bundle_runtime::resources::{Held, allocate_span};
use bundle_runtime::upload::{DeviceInputs, InputLayout, PhaseArena};
use kernel_core::allocation::Allocation;
use kernel_core::backend::{Need, Prepared as Op, View};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::moe::Experts;
use model_deepseek41_contract::own_ops::ring_gather::{RingGather, RingGatherBuffers};
use model_deepseek41_contract::own_ops::rope::MAX_POSITIONS;
use model_deepseek41_contract::serving::{DECODE_BATCH, DeepSeek, MAX_VERIFY_ROWS, PREFILL_TOKENS};
use model_deepseek41_contract::state::KvStorage;
use runtime_bundle_abi::{
    BundleAbiError, BundleContextHandleV1, BundleMemoryPoolHandleV1, BundleStreamHandleV1,
    DeviceBufferSpanV1 as Span, HostServicesV1,
};

use super::calibration::Context;
use super::execution::Runtime;
use super::features::Layout;
use super::staging::Staging;
use super::state::{BLOCK_ROWS, Config, State};
use crate::buffers::{RankBuffers, Scratch, StateSpans};
use crate::engram_rows::Destination;
use crate::lower::dspark::heads::Heads;
use crate::lower::dspark::{DraftMode, Prepared, Window};
use crate::ops::Ops;
use crate::shape::Shape;

/// Enough disjoint staging for a maximum prefill and the decode phase.
pub(crate) const TARGET_ROWS: u32 = PREFILL_TOKENS + DECODE_BATCH * MAX_VERIFY_ROWS;
pub(crate) const DECODE_BASE: u32 = PREFILL_TOKENS;
pub(crate) const COMMIT_ROWS: u32 = PREFILL_TOKENS;
const DRAFT_ROWS: u32 = DECODE_BATCH * BLOCK_ROWS;
static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);

pub(crate) struct Device {
    pub shape: Shape,
    pub config: Config,
    /// From the create record's `dspark_coupled_drafts` model option:
    /// sampled requests draft from the target sampler's noise, or every
    /// request drafts greedily, as the drafter did before coupling.
    pub mode: DraftMode,
    pub cycle: super::cycle::Cycle,
    pub prefetch: super::prefetch::Cache,
    pub results: super::results::Results,
    /// Allocated verify steps' ops and storage, when the host allocates
    /// (the create record's `verify_allocation` model option).
    pub allocation: Option<super::allocation::Allocation>,
    pub features: Layout,
    pub ops: Ops,
    pub prepared: Prepared,
    pub heads: Heads,
    pub ring_gathers: Vec<Op<RingGatherBuffers<View>>>,
    pub runtime: Option<Runtime>,
    pub staging: Option<Staging>,
    pub ranks: Vec<Rank>,
    pub stages: Vec<Vec<RankBuffers>>,
    /// Exact row counts over the fixed catch-up staging input.
    pub catch_up_graphs: BTreeMap<u32, GraphCache<DeepSeek>>,
    pub diagnostic: Context,
    pub verified: Vec<u64>,
    /// `CHOPINFER_STEP_TRACE=1` at create: each execute logs its phase times.
    pub step_trace: bool,
    /// Most drafting items per drafter forward, from the create record's
    /// `dspark_draft_chunk_items` model option.
    pub draft_chunk_items: usize,
    /// Opt-in until the packed commit passes state and served comparisons.
    pub batch_commit: bool,
    #[cfg(feature = "diagnostic-taps")]
    pub draft_check: Option<std::path::PathBuf>,
}

pub(crate) struct Rank {
    pub metadata: Span,
    pub inputs: DeviceInputs,
    pub status: super::status::Status,
    pub window: Window,
    pub item_slots: Span,
    pub source_slots: Span,
    pub first: Span,
    pub last: Span,
    pub keys: Span,
    pub taps: Span,
    pub packed: Span,
    pub feature_indices: Span,
    pub feature_counts: Vec<Span>,
    pub feature_codes: Span,
    pub feature_scales: Span,
    pub commit: Option<CommitStorage>,
    pub predecessor_ids: Span,
    pub draft_ids: Span,
    pub confidence: Span,
    pub chunk_output: Span,
    pub position_rows: Vec<Span>,
    pub block_logits: Span,
    pub position_logits: Span,
    pub embedding_partial: Span,
    pub embedding_wide: Span,
    pub embedding: Span,
    pub embedding_copy: Span,
    pub pairs: Span,
    pub gathered: Span,
    pub live_items: Span,
    pub live_width: Span,
    pub noise: Span,
    pub reduce_state: Vec<Span>,
    pub gather_state: Vec<Span>,
    /// The early Engram gathers' destination (`early_engram_bytes`).
    pub engram_early: Option<Span>,
}

pub(crate) struct CommitStorage {
    pub inputs: DeviceInputs,
    pub codes: Span,
    pub scales: Span,
}

pub(crate) fn batch_commit_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = BATCH_COMMIT_OVERRIDE.with(std::cell::Cell::get) {
        return enabled;
    }
    std::env::var("CHOPINFER_DSPARK_BATCHED_COMMIT").as_deref() == Ok("1")
}

#[cfg(test)]
thread_local! {
    static BATCH_COMMIT_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_batch_commit<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Reset(Option<bool>);
    impl Drop for Reset {
        fn drop(&mut self) {
            BATCH_COMMIT_OVERRIDE.with(|value| value.set(self.0));
        }
    }
    let _reset = Reset(BATCH_COMMIT_OVERRIDE.with(|value| value.replace(Some(enabled))));
    run()
}

pub(crate) fn slice(span: Span, offset: u64, bytes: u64) -> Result<Span, String> {
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end > span.length)
    {
        return Err("DSpark device span is too short".into());
    }
    let offset = span
        .offset
        .checked_add(offset)
        .ok_or("DSpark device address overflows")?;
    Ok(Span::new(span.buffer, offset, bytes))
}

impl Device {
    /// Refuses a drafter weight view stored in another layout than its ops
    /// read: the `mtp.*` dense projections and main_proj.
    pub(crate) fn check_weight_layouts(
        &self,
        index: &bundle_runtime::index::RankIndex,
    ) -> Result<(), String> {
        if let Some(dense) = &self.ops.dense {
            dense.check_weight_layouts(index, "mtp.")?;
        }
        let layout = self.prepared.main_weight_layout()?;
        match index.views.get("mtp.0.main_proj.weight") {
            Some(view) if view.layout_id != layout => Err(format!(
                "mtp.0.main_proj.weight is stored as {}, but its ops read {layout}",
                view.layout_id
            )),
            _ => Ok(()),
        }
    }

    pub fn new(
        kernels: &crate::Kernels,
        shape: Shape,
        items: Items,
        context_tokens: u64,
        draft_chunk_items: usize,
        mode: DraftMode,
        verify_allocation: bool,
    ) -> Result<Self, String> {
        let g = shape.geometry;
        if !(1..=DECODE_BATCH as usize).contains(&draft_chunk_items) {
            return Err("DSpark draft chunk items must be between 1 and 32".into());
        }
        let config = Config {
            slots: items.capacity(),
            stages: g.dspark_stages(),
            window: g.shape.attention.window,
            max_prefill_rows: PREFILL_TOKENS,
            context_tokens,
            // RoPE rows are computed on demand. The existing contract
            // includes a validated tail beyond the served context.
            position_capacity: u64::from(MAX_POSITIONS),
        };
        let cycle = super::cycle::Cycle::new(crate::ops::engram_hash::HashDevice::new(
            kernels.own.as_ref(),
            g,
            config.slots,
            DECODE_BATCH * MAX_VERIFY_ROWS,
        )?);
        let features = Layout::new(g, config)?;
        let ops = Ops::prepare_set(
            kernels,
            &shape,
            KvStorage::Native,
            Experts {
                count: g.shape.dspark.experts,
                top_k: g.shape.dspark.top_k,
            },
        );
        if let Some(missing) = ops.missing.first() {
            return Err(missing.clone());
        }
        let prepared = Prepared::prepare(kernels.backend.as_ref(), shape)
            .map_err(|e| format!("DSpark operations: {e:?}"))?;
        let heads = Heads::prepare(kernels.backend.as_ref(), kernels.own.as_ref(), shape, mode)
            .map_err(|e| format!("DSpark heads: {e:?}"))?;
        let ring_gathers = [1, 32]
            .into_iter()
            .map(|divisor| {
                let width = g.shape.attention.head_dim / divisor;
                kernels.own.dspark_ring_gather(
                    &RingGather {
                        bytes: config.window * width,
                        source_stride: config.window * width,
                        destination_stride: (config.window + BLOCK_ROWS) * width,
                        slots: config.slots,
                    },
                    Need::rows(DECODE_BATCH),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("DSpark ring gathering: {e:?}"))?;
        let runtime = Runtime::new(State::new(config)?, g.shape.vocab, config.window)?;
        let instance_id = NEXT_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| "DSpark instance counter overflows")?;
        Ok(Self {
            shape,
            config,
            mode,
            cycle,
            prefetch: super::prefetch::Cache::new(),
            results: super::results::Results::new(kernels.own.as_ref(), g.shape.vocab)?,
            allocation: verify_allocation
                .then(|| {
                    super::allocation::Allocation::new(
                        kernels.backend.as_ref(),
                        kernels.own.as_ref(),
                        g.shape.vocab,
                        (config.slots, shape.verify_rows),
                        features.row_bytes(),
                        g.shape.hidden,
                    )
                })
                .transpose()?,
            features,
            ops,
            prepared,
            heads,
            ring_gathers,
            runtime: Some(runtime),
            staging: Some(Staging::new(g)?),
            ranks: Vec::new(),
            stages: Vec::new(),
            catch_up_graphs: BTreeMap::new(),
            diagnostic: Context {
                instance_id,
                step_id: 0,
            },
            verified: Vec::new(),
            step_trace: std::env::var("CHOPINFER_STEP_TRACE").as_deref() == Ok("1"),
            draft_chunk_items,
            batch_commit: batch_commit_enabled(),
            #[cfg(feature = "diagnostic-taps")]
            draft_check: std::env::var_os("CHOPINFER_DSPARK_DRAFT_CHECK")
                .map(std::path::PathBuf::from),
        })
    }

    pub fn workspace(&self) -> u64 {
        self.ops
            .workspace()
            .max(self.prepared.workspace())
            .max(self.heads.workspace())
    }

    pub fn shared_bytes(&self) -> std::collections::BTreeMap<String, u32> {
        let mut bytes = self.ops.shared_bytes();
        for (symbol, shared) in self.prepared.symbols().chain(self.heads.symbols()) {
            let most = bytes.entry(symbol.clone()).or_default();
            *most = (*most).max(*shared);
        }
        for op in &self.ring_gathers {
            for (symbol, shared) in &op.info().symbols {
                let most = bytes.entry(symbol.clone()).or_default();
                *most = (*most).max(*shared);
            }
        }
        bytes
    }

    pub fn extend_state(
        &self,
        services: &HostServicesV1,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        extent: bundle_runtime::state::Extent,
        target: &mut StateSpans,
        held: &mut Vec<Held>,
    ) -> Result<(), BundleAbiError> {
        let state = model_deepseek41_contract::state::request_state(
            self.shape.geometry,
            KvStorage::Native,
            true,
        );
        for buffer in state.buffers.iter().filter(|buffer| buffer.drafter) {
            let span = allocate_span(services, pool, stream, buffer.bytes(extent), held)?;
            services.zero_device_buffer(stream, span)?;
            target.buffers.push((buffer.name, span));
        }
        Ok(())
    }

    pub fn bind_stages(&mut self, buffers: &[RankBuffers]) -> Result<(), String> {
        self.stages = (0..self.config.stages)
            .map(|stage| {
                buffers
                    .iter()
                    .map(|rank| crate::lower::dspark::stage_views(rank, self.shape.geometry, stage))
                    .collect()
            })
            .collect::<Result<_, _>>()?;
        Ok(())
    }

    pub fn reserve_engram(&self, scratch: &mut Scratch<Allocation>) {
        let g = self.shape.geometry;
        let bytes = u64::from(g.engram_layers())
            * u64::from(TARGET_ROWS)
            * u64::from(crate::ops::dense::engram_k(g));
        scratch.engram_codes = Allocation::bytes(bytes);
        scratch.engram_scales = Allocation::bytes(bytes / 32);
    }

    pub fn engram_destinations(&self, scratch: &Scratch<Span>) -> Result<Vec<Destination>, String> {
        let g = self.shape.geometry;
        let stride = u64::from(TARGET_ROWS) * u64::from(crate::ops::dense::engram_k(g));
        (0..g.engram_layers())
            .map(|layer| {
                Ok(Destination {
                    codes: slice(scratch.engram_codes, u64::from(layer) * stride, stride)?,
                    scales: slice(
                        scratch.engram_scales,
                        u64::from(layer) * stride / 32,
                        stride / 32,
                    )?,
                })
            })
            .collect()
    }

    pub fn allocate(
        &mut self,
        services: &HostServicesV1,
        (context, pool, stream): (
            BundleContextHandleV1,
            BundleMemoryPoolHandleV1,
            BundleStreamHandleV1,
        ),
        held: &mut Vec<Held>,
    ) -> Result<(), BundleAbiError> {
        let g = self.shape.geometry;
        if self.ranks.is_empty() {
            self.cycle.memory = Some(self.cycle.hash.allocate(services, pool, stream, held)?);
            if self.cycle.early_enabled {
                let side = services.create_stream(context, 0)?;
                held.push(Held::Stream(side));
                self.cycle.early_stream = Some(side);
                self.cycle.early_ids = Some(allocate_span(
                    services,
                    pool,
                    side,
                    self.cycle.hash.shape.output_bytes(),
                    held,
                )?);
            }
            self.cycle.diagnostics = Some(allocate_span(
                services,
                pool,
                stream,
                u64::from(DECODE_BATCH * BLOCK_ROWS)
                    .saturating_mul(8)
                    .max(super::cycle::BOUNDARY_BYTES),
                held,
            )?);
        }
        self.results.allocate(services, pool, stream, held)?;
        if let Some(allocation) = &mut self.allocation {
            allocation.allocate(services, pool, stream, held)?;
        }
        let items = u64::from(DECODE_BATCH);
        let rows = u64::from(DRAFT_ROWS);
        let hidden = u64::from(g.shape.hidden);
        let width = u64::from(self.shape.vocab_rows());
        let markov = u64::from(g.shape.dspark.markov_rank);
        let targets = g.shape.dspark.targets.len() as u64;
        let mut alloc = |bytes| allocate_span(services, pool, stream, bytes, held);
        if self.prefetch.enabled {
            let declaration =
                model_deepseek41_contract::state::request_state(g, KvStorage::Native, true);
            let ring = declaration
                .buffers
                .iter()
                .find(|buffer| buffer.name == "dspark.ring")
                .ok_or(BundleAbiError::InvalidRange("DSpark ring declaration"))?;
            let bytes =
                u64::from(self.config.slots) * ring.layers * ring.planes.iter().sum::<u64>();
            self.prefetch.ranks.push(super::prefetch::Memory {
                undo_ring: alloc(bytes)?,
                next_ring: alloc(bytes)?,
                draft_ids: alloc(rows * 4)?,
                confidence: alloc(rows * 4)?,
                status: super::status::Status::new(alloc(
                    u64::from(super::state::BATCH_PROJECTION_ROWS + 8 * BLOCK_ROWS) * 8,
                )?),
            });
        }
        let layout = InputLayout::new::<DeepSeek>(DRAFT_ROWS.max(self.config.window), DRAFT_ROWS);
        let cells = self.config.window + BLOCK_ROWS;
        let metadata_layout = super::upload::Layout::new(layout.end, items, rows, u64::from(cells))
            .map_err(|_| BundleAbiError::InvalidRange("DSpark metadata layout"))?;
        let metadata = alloc(metadata_layout.bytes)?;
        let plane = |(at, bytes)| {
            slice(metadata, at, bytes)
                .map_err(|_| BundleAbiError::InvalidRange("DSpark metadata plane"))
        };
        let arena = PhaseArena {
            arena: plane(metadata_layout.inputs)?,
            block_tables: alloc(4)?,
        };
        let inputs = DeviceInputs::within(arena, &layout);
        // Keep draft metadata offsets and uploads unchanged. Commit has a
        // separate row domain, allocated before the host sizes the KV pool.
        let commit_inputs = if self.batch_commit {
            let layout = InputLayout::new::<DeepSeek>(COMMIT_ROWS, 1);
            Some(DeviceInputs::within(
                PhaseArena {
                    arena: alloc(layout.end)?,
                    block_tables: arena.block_tables,
                },
                &layout,
            ))
        } else {
            None
        };
        let stride = u64::from(cells) * u64::from(g.shape.attention.head_dim);
        let window = Window {
            codes: alloc(items * stride)?,
            scales: alloc(items * stride / 32)?,
            sequences: DECODE_BATCH,
            cells,
            code_stride: stride,
            scale_stride: stride / 32,
        };
        let indices = self
            .prepared
            .feature_indices(PREFILL_TOKENS)
            .map_err(|_| BundleAbiError::InvalidRange("DSpark feature indices"))?;
        let feature_indices = alloc(indices.len() as u64 * 4)?;
        services.copy_host_to_device(
            stream,
            feature_indices,
            &indices
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        let feature_counts = (0..targets)
            .map(|_| alloc(4))
            .collect::<Result<Vec<_>, _>>()?;
        let position_rows = (0..BLOCK_ROWS)
            .map(|position| {
                let span = alloc(items * 4)?;
                let ids = (0..DECODE_BATCH)
                    .flat_map(|item| (item * BLOCK_ROWS + position).to_le_bytes())
                    .collect::<Vec<_>>();
                services.copy_host_to_device(stream, span, &ids)?;
                Ok(span)
            })
            .collect::<Result<Vec<_>, BundleAbiError>>()?;
        let (reduce, gather) = self.heads.state();
        let mut state = |sizes: &[u64]| {
            sizes
                .iter()
                .map(|&bytes| {
                    let span = alloc(bytes)?;
                    services.zero_device_buffer(stream, span)?;
                    Ok(span)
                })
                .collect::<Result<Vec<_>, BundleAbiError>>()
        };
        let reduce_state = state(reduce)?;
        let gather_state = state(gather)?;
        let pairs = alloc(items * 2 * 4)?;
        services.zero_device_buffer(stream, pairs)?;
        let live_width = alloc(4)?;
        services.copy_host_to_device(stream, live_width, &self.shape.vocab_rows().to_le_bytes())?;
        // Catch-up retains at most one raw window per slot. A target step
        // and its commit can each inspect all prefill/decode rows and stages.
        let status_rows = u64::from(self.config.slots)
            * self.config.history_capacity()
            * u64::from(self.config.stages)
            + u64::from(TARGET_ROWS)
                * (u64::from(self.config.stages) + g.shape.csa.index_sources.len() as u64 + 1)
            + rows;
        let feature_rows = if self.batch_commit {
            COMMIT_ROWS
        } else {
            self.config.window
        };
        let feature_codes;
        let feature_scales;
        self.ranks.push(Rank {
            metadata,
            status: super::status::Status::new(alloc(status_rows * 8)?),
            inputs,
            window,
            item_slots: plane(metadata_layout.item_slots)?,
            source_slots: plane(metadata_layout.source_slots)?,
            first: plane(metadata_layout.first)?,
            last: plane(metadata_layout.last)?,
            keys: plane(metadata_layout.keys)?,
            taps: alloc(u64::from(PREFILL_TOKENS) * targets * hidden * 2)?,
            packed: alloc(u64::from(TARGET_ROWS) * targets * hidden * 2)?,
            feature_indices,
            feature_counts,
            feature_codes: {
                feature_codes = alloc(u64::from(feature_rows) * targets * hidden)?;
                slice(
                    feature_codes,
                    0,
                    u64::from(self.config.window) * targets * hidden,
                )
                .map_err(|_| BundleAbiError::InvalidRange("DSpark feature codes"))?
            },
            feature_scales: {
                feature_scales = alloc(u64::from(feature_rows) * targets * hidden / 32)?;
                slice(
                    feature_scales,
                    0,
                    u64::from(self.config.window) * targets * hidden / 32,
                )
                .map_err(|_| BundleAbiError::InvalidRange("DSpark feature scales"))?
            },
            commit: commit_inputs.map(|inputs| CommitStorage {
                inputs,
                codes: feature_codes,
                scales: feature_scales,
            }),
            predecessor_ids: plane(metadata_layout.predecessor_ids)?,
            draft_ids: alloc(rows * 4)?,
            confidence: alloc(rows * 4)?,
            chunk_output: alloc(rows * 8)?,
            position_rows,
            block_logits: alloc(rows * width * 4)?,
            position_logits: alloc(items * width * 4)?,
            embedding_partial: alloc(items * markov * 2)?,
            embedding_wide: alloc(items * markov * 4)?,
            embedding: alloc(items * markov * 2)?,
            embedding_copy: alloc(items * markov * 2)?,
            pairs,
            gathered: alloc(items * 2 * 4 * u64::from(self.shape.ranks()))?,
            live_items: plane(metadata_layout.live_items)?,
            live_width,
            noise: plane(metadata_layout.noise)?,
            reduce_state,
            gather_state,
            engram_early: if self.cycle.early_enabled {
                Some(alloc(early_engram_bytes(g))?)
            } else {
                None
            },
        });
        Ok(())
    }
}

/// One rank's early Engram scratch: each layer's codes, then its scales, for
/// one row per decode item. No kernel reads it.
pub(crate) fn early_engram_bytes(g: &Geometry) -> u64 {
    let codes = u64::from(DECODE_BATCH) * u64::from(crate::ops::dense::engram_k(g));
    u64::from(g.engram_layers()) * (codes + codes / 32)
}
