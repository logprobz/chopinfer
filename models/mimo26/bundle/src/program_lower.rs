//! Bind one rank-program stage to the rank image and fixed device buffers.

use std::collections::BTreeMap;

use kernel_core::contract::{DType, Refusal};
use runtime_bundle_abi::DeviceBufferSpanV1;

pub use crate::upload::DeviceInputs;

use crate::ops::{
    AttentionLayer, AttentionPart, AttentionViews, CacheViews, LogitsBuffers, RopeTarget,
    RopeViews, RoutePart, RouteViews,
};
use crate::pipeline::{PreparedLaunch, Program};
use crate::program::Stage;
use crate::scratch::RankScratch;
use crate::selection::{MimoPhase as _, Phase};
use crate::topology::Tp;

/// A lowered launch with what the pipelined prefill reads of it
/// (`Target::adjust`): the facts of the op that made it.
pub struct Lowered {
    pub launch: PreparedLaunch,
    pub facts: crate::overlap::Facts,
}

impl From<PreparedLaunch> for Lowered {
    /// The one launch MiMo still builds without an op, the media
    /// injection's row override: its grid neither strides over the SMs nor
    /// runs a collective.
    fn from(launch: PreparedLaunch) -> Self {
        Self {
            facts: Default::default(),
            launch,
        }
    }
}

/// The launches of `lowered`, without their facts.
pub fn launches(lowered: Vec<Lowered>) -> Vec<PreparedLaunch> {
    lowered.into_iter().map(|lowered| lowered.launch).collect()
}

/// The program of `lowered`, with which of its launches are collectives.
pub fn program(lowered: Vec<Lowered>) -> Program {
    Program::of(
        lowered
            .into_iter()
            .map(|lowered| (lowered.launch, lowered.facts)),
    )
}

pub struct RankStage<'a> {
    pub services: Option<&'a runtime_bundle_abi::HostServicesV1>,
    /// The topology whose per-rank shapes and kernels the stage uses.
    pub tp: Tp,
    pub rank: u32,
    pub phase: Phase,
    pub live_tokens: u32,
    /// Items in the phase; only the prefill ring backup reads it.
    pub live_batch: u32,
    pub context: u32,
    pub ga_kv: DType,
    /// Per global layer, the K and V scales of an FP8 pool; empty is 1.0.
    pub ga_kv_scales: &'static [[f32; 2]],
    pub views: &'a BTreeMap<String, DeviceBufferSpanV1>,
    /// The rank's authenticated projections; launches need them.
    pub projections: Option<crate::projection::Rank<'a>>,
    pub inputs: DeviceInputs,
    pub scratch: &'a RankScratch,
    pub activation: DeviceBufferSpanV1,
    pub normalized: DeviceBufferSpanV1,
    pub workspace: DeviceBufferSpanV1,
    pub local_logits: DeviceBufferSpanV1,
    pub scored_prompt_logits: bool,
    pub logit_rows: &'a [u32],
    pub logits_gather: LogitsBuffers,
    /// Decode logits rows go to their owning ranks with the row exchange
    /// instead of all to rank 0; `logits_gather` then holds the exchange's
    /// flags and epoch.
    pub exchange_logits: bool,
    /// One span per rank, in rank order.
    pub hidden_inboxes: Vec<DeviceBufferSpanV1>,
    pub hidden_flags: Vec<DeviceBufferSpanV1>,
    pub hidden_epochs: Vec<DeviceBufferSpanV1>,
    pub hidden_gathered: Vec<DeviceBufferSpanV1>,
    pub ring_state: DeviceBufferSpanV1,
    /// The layout of `ring_state`: one ring slot per item of the largest
    /// graph bucket.
    pub ring: crate::ring::RingGeometry,
    /// The sequences the instance serves: attention entries are planned for
    /// its largest bucket.
    pub items: bundle_runtime::geometry::Items,
    /// The instance's environment switches, read when it was created.
    pub switches: crate::switches::Switches,
    /// The ops the instance prepared, and this phase's bucket among
    /// `scratch::COMM_BUCKETS`, which selects its ops.
    pub ops: &'a crate::ops::Ops,
    pub bucket: usize,
    pub kv_pool: DeviceBufferSpanV1,
}

impl RankStage<'_> {
    fn view(&self, name: &str) -> Result<DeviceBufferSpanV1, Refusal> {
        self.views.get(name).copied().ok_or_else(|| {
            if let Some(services) = self.services {
                let _ =
                    services.log(format!("MiMo rank {} missing view {name}", self.rank).as_bytes());
            }
            Refusal("MiMo rank image has no required view")
        })
    }

    /// The declared projection a stage runs. The MTP program runs the
    /// main layer's adapters over its own rank views.
    fn projection_id(&self, stage: Stage) -> Result<String, Refusal> {
        let mtp = self.views.contains_key("model.mtp.layers.0.eh_proj.weight");
        let attention = |layer| {
            if mtp {
                "mtp.attention-adapter"
            } else if self.tp.geometry.is_global(layer) {
                "ga"
            } else {
                "swa"
            }
        };
        Ok(match stage {
            Stage::Embedding => "embedding.bf16".into(),
            Stage::Qkv(layer) => format!("{}.qkv.{layer}.bf16", attention(layer)),
            Stage::AttentionOutput(layer) => format!("{}.output.{layer}.bf16", attention(layer)),
            Stage::DenseGateUp(layer) if mtp => format!("mtp.mlp-adapter.gate-up.{layer}.bf16"),
            Stage::DenseGateUp(layer) => format!("dense.mlp.gate-up.{layer}.bf16"),
            Stage::DenseDown(layer) if mtp => format!("mtp.mlp-adapter.down.{layer}.bf16"),
            Stage::DenseDown(layer) => format!("dense.down.{layer}.bf16"),
            Stage::RouterProjection(layer) => format!("router.{layer}.f32"),
            Stage::GroupedGateUp(layer) => format!("moe.experts.gate-up.{layer}.bf16"),
            Stage::GroupedDown(layer) => format!("moe.experts.down.{layer}.bf16"),
            Stage::LmHead if self.scored_prompt_logits => "head.f32".into(),
            Stage::LmHead => crate::projection::BF16_HEAD.into(),
            _ => return Err(Refusal("stage is not a declared projection")),
        })
    }

    /// Launch a declared projection through the authenticated store.
    pub(crate) fn bound_projection(
        &self,
        id: &str,
        phase: Phase,
        rows: u32,
        activation: DeviceBufferSpanV1,
        routing: Option<bundle_runtime::projection::Routing>,
        output: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, Refusal> {
        crate::projection::launch(
            self.projections.ok_or(Refusal(
                "MiMo projections need the authenticated precision store",
            ))?,
            id,
            crate::projection::Launch {
                phase,
                rows,
                activation,
                live_rows: self.inputs.live_tokens,
                routing,
                output,
                partial: self.workspace,
                semaphores: self.scratch.gemm_semaphores,
                unit_scale: self.scratch.unit_scale,
            },
        )
    }

    /// One dense projection stage at this step's rows.
    fn project(
        &self,
        stage: Stage,
        activation: DeviceBufferSpanV1,
        output: DeviceBufferSpanV1,
    ) -> Result<Lowered, Refusal> {
        let mut calls = self.bound_projection(
            &self.projection_id(stage)?,
            self.phase,
            self.live_tokens,
            activation,
            None,
            output,
        )?;
        if calls.len() != 1 {
            return Err(Refusal("dense projection stage count differs"));
        }
        Ok(calls.remove(0))
    }

    /// The views of a RoPE KV write or ring write of a global or
    /// sliding-window layer, with that layer kind's table.
    fn rope_views(&self, global: bool) -> Result<RopeViews, Refusal> {
        Ok(RopeViews {
            cos_sin: self
                .scratch
                .rope_tables
                .ok_or(Refusal("RoPE tables missing"))?[usize::from(!global)],
            qkv: self.scratch.qkv,
            q: self.scratch.q,
            k_new: self.scratch.k_new,
            v_new: self.scratch.v_new,
            token_batch: self.inputs.token_batch,
            positions: self.inputs.positions,
            live_tokens: self.inputs.live_tokens,
        })
    }

    /// Global layer `layer`'s pages of the KV pool.
    fn paged_cache(&self, layer: u32) -> Result<CacheViews, Refusal> {
        let pages =
            crate::ga_pool::layer_pages(self.kv_pool, layer, self.ga_kv == DType::Fp8E4M3, self.tp)
                .map_err(Refusal)?;
        if !self.tp.geometry.is_global(layer) || pages.page_capacity == 0 {
            return Err(Refusal("MiMo global-attention page geometry is invalid"));
        }
        Ok(CacheViews {
            k: pages.k,
            v: pages.v,
            slots: u32::try_from(pages.page_capacity)
                .map_err(|_| Refusal("MiMo global-attention pages overflow"))?,
            strides: [pages.k_page_stride_bytes, pages.v_page_stride_bytes],
            scale_bits: crate::ga_pool::scale_bits_for(self.tp, self.ga_kv_scales, layer),
        })
    }

    /// `part` of layer `layer`'s attention: global layers over their pages
    /// and block tables, sliding-window layers over the items' rings with
    /// their sink logits. A decode that folds RoPE and the KV write in reads
    /// the packed QKV rows and the layer's table.
    fn attention(&self, layer: u32, part: AttentionPart) -> Result<Option<Lowered>, Refusal> {
        let global = self.tp.geometry.is_global(layer);
        let fold = crate::attention_plan::folds_rope(self.phase, self.switches);
        let tables = self
            .scratch
            .rope_tables
            .ok_or(Refusal("RoPE tables missing"))?;
        let (cache, block_tables, slot_ids, sink) = if global {
            (
                self.paged_cache(layer)?,
                Some(self.inputs.block_tables),
                None,
                None,
            )
        } else {
            let (cache, slot_ids) = self.ring_cache(layer)?;
            let sink = self
                .views
                .get(&format!(
                    "model.layers.{layer}.self_attn.attention_sink_bias"
                ))
                .copied()
                .ok_or(Refusal("MiMo attention sink view is missing"))?;
            (cache, None, Some(slot_ids), Some(sink))
        };
        let views = AttentionViews {
            q: if fold {
                self.scratch.qkv
            } else {
                self.scratch.q
            },
            k_new: self.scratch.k_new,
            v_new: self.scratch.v_new,
            slot_ids,
            block_tables,
            seq_lens: self.inputs.seq_lens,
            q_start: self.inputs.q_start,
            q_len: self.inputs.q_len,
            prefix_len: self.inputs.prefix_len,
            live_batch: self.inputs.live_batch,
            sink,
            cos_sin: fold.then_some(tables[usize::from(!global)]),
            output: self.scratch.attention,
            workspace: self.workspace,
        };
        // A decode step's items are its bucket's rows.
        let items = match self.phase {
            Phase::Decode(rows) => rows,
            Phase::Verify(_) | Phase::Prefill => self.live_batch,
        };
        self.ops.attention(
            if global {
                AttentionLayer::Global
            } else {
                AttentionLayer::Sliding
            },
            part,
            self.phase,
            self.bucket,
            self.live_tokens,
            items,
            self.context,
            &views,
            &cache,
        )
    }

    /// Sliding-window layer `layer`'s rings, and the items' ring slots.
    fn ring_cache(&self, layer: u32) -> Result<(CacheViews, DeviceBufferSpanV1), Refusal> {
        let g = self.tp.geometry;
        if layer >= g.layers() || g.is_global(layer) {
            return Err(Refusal("MiMo sliding-window layer is invalid"));
        }
        let local = layer - g.global_ordinal(layer) as u32;
        let (k, v) = crate::ring::cache_views(self.ring_state, local, self.ring)
            .ok_or(Refusal("MiMo SWA state has no cache planes"))?;
        let window = u64::from(g.window());
        Ok((
            CacheViews {
                k,
                v,
                slots: self.ring.slots,
                strides: [window * self.ring.k_entry, window * self.ring.v_entry],
                scale_bits: [1.0_f32.to_bits(); 2],
            },
            self.inputs.slot_ids,
        ))
    }

    /// The views of this step's routes of MoE layer `layer`.
    fn route_views(&self, layer: u32) -> Result<RouteViews, Refusal> {
        Ok(RouteViews {
            router_logits: self.scratch.router_logits,
            bias: self.view(&format!(
                "model.layers.{layer}.mlp.gate.e_score_correction_bias"
            ))?,
            ids: self.scratch.topk_ids,
            weights: self.scratch.topk_weights,
            counts: self.scratch.expert_counts,
            offsets: self.scratch.expert_offsets,
            sorted_rows: self.scratch.sorted_rows,
            padded_rows: self.scratch.padded_rows,
            active_experts: self.scratch.active_experts,
            tile_tables: crate::ops::route_tile_tables(self.tp, self.workspace)?,
            live_tokens: self.inputs.live_tokens,
        })
    }

    /// A route stage of MoE layer `layer`, or its combine.
    fn moe_row(&self, stage: Stage, layer: u32) -> Result<Vec<Lowered>, Refusal> {
        let part = match stage {
            Stage::RouterTop8(_) => RoutePart::Router,
            Stage::PermuteCount(_) => RoutePart::Count,
            Stage::PermuteScan(_) => RoutePart::Scan,
            Stage::PermuteScatter(_) => RoutePart::Scatter,
            Stage::CombineExperts(_) => {
                return Ok(vec![self.ops.moe_combine(
                    self.phase,
                    self.bucket,
                    self.switches,
                    self.live_tokens,
                    self.scratch.routed_down,
                    self.scratch.topk_weights,
                    self.scratch.combined,
                    self.inputs.live_tokens,
                )?]);
            }
            _ => return Err(Refusal("not a MiMo MoE row stage")),
        };
        self.ops.moe_route(
            part,
            self.phase,
            self.bucket,
            self.switches,
            self.live_tokens,
            &self.route_views(layer)?,
        )
    }

    /// The route of a decode or verify bucket that does not route as
    /// prefill does: the projected route of the buckets the fused router
    /// serves, which projects the router logits itself, the decode route
    /// otherwise.
    fn moe_decode_route(&self, layer: u32) -> Result<Vec<Lowered>, Refusal> {
        if !self.ops.projects_router(self.phase, self.bucket) {
            return self.moe_row(Stage::RouterTop8(layer), layer);
        }
        Ok(vec![self.ops.moe_projected_route(
            self.phase,
            self.bucket,
            self.switches,
            self.live_tokens,
            self.normalized,
            self.view(&format!("model.layers.{layer}.mlp.gate.weight"))?,
            [self.workspace, self.scratch.gemm_semaphores],
            &self.route_views(layer)?,
        )?])
    }

    fn grouped(&self, stage: Stage) -> Result<Vec<Lowered>, Refusal> {
        let gate = matches!(stage, Stage::GroupedGateUp(_));
        self.bound_projection(
            &self.projection_id(stage)?,
            self.phase,
            self.live_tokens,
            if gate {
                self.normalized
            } else {
                self.scratch.routed_gate
            },
            Some(bundle_runtime::projection::Routing {
                weights: self.scratch.topk_weights,
                offsets: self.scratch.expert_offsets,
                sorted_rows: self.scratch.sorted_rows,
                padded_rows: self.scratch.padded_rows,
                active_experts: if self.phase.prefill_route() {
                    self.scratch.all_experts
                } else {
                    self.scratch.active_experts
                },
            }),
            if gate {
                self.scratch.routed_gate
            } else {
                self.scratch.routed_down
            },
        )
    }

    fn reduce_hidden(&self, stage: Stage) -> Result<Vec<Lowered>, Refusal> {
        let (input, norm_name) = match stage {
            Stage::AttentionReduceNorm(layer) => (
                self.scratch.hidden_partial,
                format!("model.layers.{layer}.post_attention_layernorm.weight"),
            ),
            Stage::MlpReduceNorm(layer) => (
                if layer == 0 {
                    self.scratch.hidden_partial
                } else {
                    self.scratch.combined
                },
                format!("model.layers.{}.input_layernorm.weight", layer + 1),
            ),
            Stage::FinalReduceNorm => (self.scratch.combined, "model.norm.weight".into()),
            _ => return Err(Refusal("MiMo stage is not hidden reduction")),
        };
        // Decode and verify fold the MoE combine into the MLP all-reduce, which then
        // reads the routed rows directly.
        let combine = self.folds_combine(stage);
        let input = if combine {
            self.scratch.routed_down
        } else {
            input
        };
        let route = if self.phase.exact_rows(self.switches) {
            crate::ops::Route::Exact
        } else {
            crate::ops::Route::Phase
        };
        self.hidden_reduce(route, input, self.view(&norm_name)?, combine)
    }

    /// This rank's hidden all-reduce of `input` into the residual, normalized
    /// with `norm_weight` into every rank's normalized rows; a `combine`
    /// input is each row's routed rows. Decode and verify reduce their
    /// bucket's rows, prefill its live rows.
    pub(crate) fn hidden_reduce(
        &self,
        route: crate::ops::Route,
        input: DeviceBufferSpanV1,
        norm_weight: DeviceBufferSpanV1,
        combine: bool,
    ) -> Result<Vec<Lowered>, Refusal> {
        let rows = match self.phase {
            Phase::Decode(_) | Phase::Verify(_) => self.phase.capacity()?,
            Phase::Prefill => self.live_tokens,
        };
        self.ops.hidden_reduce(
            self.rank,
            self.phase,
            self.bucket,
            route,
            rows,
            crate::ops::ReduceViews {
                input,
                residual: self.activation,
                norm_weight,
                output: self.normalized,
                live_rows: combine.then_some(self.inputs.live_tokens),
                outputs: &self.hidden_gathered,
                inboxes: &self.hidden_inboxes,
                flags: &self.hidden_flags,
                epochs: &self.hidden_epochs,
            },
        )
    }

    /// Decode and verify MoE layers of up to 32 rows sum their routed rows
    /// inside the MLP all-reduce, so their combine stage launches nothing.
    /// Wider graphs keep the combine row (`graph_folds_combine`).
    fn folds_combine(&self, stage: Stage) -> bool {
        self.ops.folds_combine(self.phase, self.bucket)
            && match stage {
                Stage::MlpReduceNorm(layer) | Stage::CombineExperts(layer) => layer > 0,
                Stage::FinalReduceNorm => true,
                _ => false,
            }
    }

    /// Prefill copies the rows that need logits into a compact buffer, so the
    /// vocabulary head runs over those rows instead of every prompt token.
    fn gather_head_rows(&self) -> Result<Vec<Lowered>, Refusal> {
        if self.phase != Phase::Prefill || self.logit_rows.is_empty() {
            return Ok(Vec::new());
        }
        Ok(vec![self.ops.head_rows(
            self.logit_rows.len() as u32,
            self.live_tokens,
            &crate::ops::EmbeddingViews {
                table: self.normalized,
                ids: self.inputs.row_indices,
                output: self.head_rows(),
                live_rows: self.inputs.logit_rows,
            },
        )?])
    }

    /// Where the gathered head rows live: `hidden_partial` is free once the
    /// last layer's reductions have read it.
    fn head_rows(&self) -> DeviceBufferSpanV1 {
        self.scratch.hidden_partial
    }

    /// Up to 32 unscored rows use the decode head of their bucket; more rows,
    /// or scored prompt rows, use the prefill head over the gathered count.
    fn prefill_head(&self) -> Result<Vec<Lowered>, Refusal> {
        let rows = self.logit_rows.len() as u32;
        if rows == 0 {
            return Ok(Vec::new());
        }
        let phase = if !self.scored_prompt_logits && rows <= crate::contract::DECODE_ENTRY_ROWS {
            Phase::Decode(rows)
        } else {
            Phase::Prefill
        };
        self.bound_projection(
            &self.projection_id(Stage::LmHead)?,
            phase,
            match phase {
                Phase::Decode(_) | Phase::Verify(_) => phase.capacity()?,
                Phase::Prefill => rows,
            },
            self.head_rows(),
            None,
            self.local_logits,
        )
    }

    /// Lower one rank stage to its engine launches.
    pub fn lower(&self, stage: Stage) -> Result<Vec<PreparedLaunch>, Refusal> {
        self.lower_with_facts(stage).map(launches)
    }

    /// [`Self::lower`], with each launch's facts for the pipelined prefill.
    pub fn lower_with_facts(&self, stage: Stage) -> Result<Vec<Lowered>, Refusal> {
        if stage == Stage::MediaInject {
            // The override kernel is qualified, but the encoder programs do
            // not yet bind their launches. Nonempty media is refused in preflight.
            return Ok(Vec::new());
        }
        // Decode and verify buckets that route their experts as prefill does
        // (`MimoPhase::prefill_route`) keep the router GEMM, top-8 and tiled
        // permutation stages; the others run the decode route.
        if self.phase.token_bucketed() && !self.phase.prefill_route() {
            match stage {
                Stage::RouterProjection(_) if self.ops.projects_router(self.phase, self.bucket) => {
                    return Ok(Vec::new());
                }
                Stage::RouterTop8(layer) => return self.moe_decode_route(layer),
                Stage::PermuteCount(_) | Stage::PermuteScan(_) | Stage::PermuteScatter(_) => {
                    return Ok(Vec::new());
                }
                _ => {}
            }
        }
        // Decode attention rotates Q and writes each row's current K and V
        // to the ring or pages itself, at every bucket; verify writes the
        // ring after attention, as prefill does.
        if self.phase.token_bucketed() {
            match stage {
                Stage::SlidingWindowWrite(_) if matches!(self.phase, Phase::Decode(_)) => {
                    return Ok(Vec::new());
                }
                Stage::RopeForSlidingWindow(_) | Stage::RopeAndPagedWrite(_)
                    if crate::attention_plan::folds_rope(self.phase, self.switches) =>
                {
                    return Ok(Vec::new());
                }
                _ => {}
            }
        }
        let launch: Lowered = match stage {
            Stage::MediaInject => unreachable!("handled before regular rank lowering"),
            Stage::RingBackup => self.ops.ring_backup(
                self.phase,
                self.bucket,
                self.live_batch,
                &crate::ops::RingBackupViews {
                    ring_state: self.ring_state,
                    slots: self.ring.slots,
                    slot_ids: self.inputs.slot_ids,
                    q_len: self.inputs.q_len,
                    prefix_len: self.inputs.prefix_len,
                    live_batch: self.inputs.live_batch,
                },
            )?,
            Stage::Embedding => self.project(stage, self.inputs.token_ids, self.activation)?,
            Stage::InputNorm => self.ops.hidden_norm(
                self.phase,
                self.bucket,
                self.switches,
                self.live_tokens,
                self.activation,
                self.view("model.layers.0.input_layernorm.weight")?,
                self.normalized,
                self.inputs.live_tokens,
            )?,
            Stage::Qkv(_) => self.project(stage, self.normalized, self.scratch.qkv)?,
            Stage::RopeForSlidingWindow(layer) => {
                let (cache, slot_ids) = self.ring_cache(layer)?;
                self.ops.rope_kv(
                    self.phase,
                    self.bucket,
                    self.switches,
                    self.live_tokens,
                    &self.rope_views(false)?,
                    &cache,
                    RopeTarget::Ring { slot_ids },
                )?
            }
            Stage::RopeAndPagedWrite(layer) => {
                let cache = self.paged_cache(layer)?;
                self.ops.rope_kv(
                    self.phase,
                    self.bucket,
                    self.switches,
                    self.live_tokens,
                    &self.rope_views(true)?,
                    &cache,
                    RopeTarget::Paged {
                        block_tables: self.inputs.block_tables,
                    },
                )?
            }
            Stage::SlidingWindowWrite(layer) => {
                let (cache, slot_ids) = self.ring_cache(layer)?;
                self.ops.ring_write(
                    self.phase,
                    self.bucket,
                    self.switches,
                    self.live_tokens,
                    &self.rope_views(false)?,
                    &cache,
                    slot_ids,
                    self.inputs.last_positions,
                )?
            }
            Stage::AttentionPartial(layer) | Stage::AttentionMerge(layer) => {
                let part = if matches!(stage, Stage::AttentionPartial(_)) {
                    AttentionPart::Partial
                } else {
                    AttentionPart::Merge
                };
                let Some(launch) = self.attention(layer, part)? else {
                    return Ok(Vec::new());
                };
                launch
            }
            Stage::AttentionOutput(_) => {
                self.project(stage, self.scratch.attention, self.scratch.hidden_partial)?
            }
            Stage::DenseGateUp(0) => {
                self.project(stage, self.normalized, self.scratch.dense_gate)?
            }
            Stage::DenseDown(0) => {
                self.project(stage, self.scratch.dense_gate, self.scratch.hidden_partial)?
            }
            Stage::RouterProjection(_) => {
                self.project(stage, self.normalized, self.scratch.router_logits)?
            }
            Stage::RouterTop8(layer)
            | Stage::PermuteCount(layer)
            | Stage::PermuteScan(layer)
            | Stage::PermuteScatter(layer)
            | Stage::CombineExperts(layer) => {
                if self.folds_combine(stage) {
                    return Ok(Vec::new());
                }
                return self.moe_row(stage, layer);
            }
            Stage::GatherHeadRows => return self.gather_head_rows(),
            Stage::LmHead if self.phase == Phase::Prefill => return self.prefill_head(),
            Stage::LmHead => self.project(stage, self.normalized, self.local_logits)?,
            Stage::GatherLogits => {
                if self.exchange_logits && self.phase == Phase::Prefill {
                    return Err(Refusal(
                        "MiMo logits exchange serves decode and verify only",
                    ));
                }
                // Prefill gathers its logits rows before the head, so they
                // sit at rows 0..count.
                let compact;
                let rows = if self.phase == Phase::Prefill {
                    compact = (0..self.logit_rows.len() as u32).collect::<Vec<_>>();
                    &compact
                } else {
                    self.logit_rows
                };
                return self.ops.gather_logits(
                    self.rank,
                    self.phase,
                    self.bucket,
                    self.exchange_logits,
                    if self.scored_prompt_logits {
                        DType::F32
                    } else {
                        DType::Bf16
                    },
                    rows,
                    &self.logits_gather,
                );
            }
            Stage::GroupedGateUp(_) | Stage::GroupedDown(_) => {
                return self.grouped(stage);
            }
            Stage::AttentionReduceNorm(_) | Stage::MlpReduceNorm(_) | Stage::FinalReduceNorm => {
                return self.reduce_hidden(stage);
            }
            Stage::DenseGateUp(_) | Stage::DenseDown(_) => {
                return Err(Refusal("MiMo stage has no launch adapter"));
            }
        };
        Ok(vec![launch])
    }
}
