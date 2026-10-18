//! Bind model buffers to retained declaration projection handles.
use super::*;
use bundle_runtime::projection as shared;
use model_qwen38_contract::precision::DenseRole;

pub(crate) fn scratch(s: &RankScratch) -> shared::Scratch {
    shared::Scratch {
        quantizer_status: None,
        quantizer_status_row_offset: 0,
        partial: s.workspace,
        semaphores: s.gemm_semaphores,
        unused: s.unused,
        temporary: s.precision_temporary,
        gathered: None,
        codes: None,
        scales: None,
        activation_global: None,
        row_amax: None,
        output_codes: None,
        output_scales: None,
    }
}

impl RankStage<'_> {
    /// The registered lookups of a package without a precision component.
    fn lookups(&self) -> Result<&crate::ops::Lookups, Refusal> {
        self.ops
            .ok_or(Refusal("Qwen3.8 ops are not prepared"))?
            .lookups(crate::ops::Key::of(self.phase))
    }

    fn projection_phase(&self, rows: u32) -> shared::Phase {
        shared::Phase {
            usage: crate::projections::usage(self.phase),
            rows,
            padded_rows: rows * self.shape.top_k(),
        }
    }

    fn projection_input(&self, activation: DeviceBufferSpanV1) -> shared::Input<'static> {
        shared::Input {
            activation,
            live_rows: self.live_tokens(),
            routing: None,
            residual: None,
            row_offset: 0,
            shard_rows: 0,
            side: &[],
            scalars: &[],
        }
    }

    pub(super) fn prepared_stage(&self, stage: Stage) -> Result<Option<Vec<Lowered>>, Refusal> {
        match stage {
            Stage::NgramGather => return Ok(Some(self.built(self.projection_ngram()?))),
            Stage::Embedding => return Ok(Some(self.built(self.projection_embedding()?))),
            _ => {}
        }
        let Some(precision) = self.precision else {
            return Ok(None);
        };
        let s = self.scratch();
        let (layer, ports): (u32, Vec<&(DenseRole, String)>) = match stage {
            Stage::LinearInProj(layer) | Stage::AttnInProj(layer) => {
                let ports = self.dense_ports(layer)?;
                let first = ports
                    .first()
                    .ok_or(Refusal("missing prepared input port"))?;
                let handle = precision.operation(self.rank, &first.1, self.phase)?;
                if matches!(handle.recipe(), shared::Recipe::Registered) {
                    return Ok(None);
                }
                (layer, ports.iter().take(4).collect())
            }
            Stage::LinearOut(layer) | Stage::AttnOut(layer) => (
                layer,
                self.dense_ports(layer)?
                    .iter()
                    .filter(|(role, _)| {
                        matches!(role, DenseRole::Output | DenseRole::AttentionOutput)
                    })
                    .collect(),
            ),
            Stage::SharedGateUp(layer) | Stage::SharedDown(layer) => {
                let wanted = if matches!(stage, Stage::SharedGateUp(_)) {
                    DenseRole::SharedGateUp
                } else {
                    DenseRole::SharedDown
                };
                let ports = self.dense_ports(layer)?;
                (
                    layer,
                    ports.iter().filter(|(role, _)| *role == wanted).collect(),
                )
            }
            Stage::ExpertsGateUp(layer) => {
                return self
                    .projection_experts(layer, GatedMlpProjection::GateUp)
                    .map(Some);
            }
            Stage::ExpertsDown(layer) => {
                return self
                    .projection_experts(layer, GatedMlpProjection::Down)
                    .map(Some);
            }
            _ => return Ok(None),
        };
        let mut launches = Vec::new();
        for (role, id) in ports {
            let handle = precision.operation(self.rank, id, self.phase)?;
            if matches!(handle.recipe(), shared::Recipe::Registered) {
                if launches.is_empty() {
                    return Ok(None);
                }
                return Err(Refusal(
                    "prepared input fusion mixes legacy and separate ports",
                ));
            }
            let (activation, output) = match *role {
                DenseRole::LinearQkv => (s.u, s.lin_qkv),
                DenseRole::LinearZ => (s.u, s.lin_z),
                DenseRole::LinearA | DenseRole::LinearB => (s.u, s.lin_ab),
                DenseRole::AttentionQ | DenseRole::AttentionK | DenseRole::AttentionV => {
                    (s.u, s.attn_qkv)
                }
                DenseRole::IndexQk => (s.u, s.attn_index),
                DenseRole::Output | DenseRole::AttentionOutput => (
                    if self.shape.is_full(layer) {
                        s.attn_gated
                    } else {
                        s.lin_normed
                    },
                    s.partial,
                ),
                DenseRole::SharedGateUp => (s.u, s.shared_intermediate),
                DenseRole::SharedDown => (s.shared_intermediate, s.shared_out),
            };
            launches.extend(self.stored(
                handle,
                self.projection_phase(self.rows),
                self.projection_input(activation),
                output,
            )?);
        }
        Ok(Some(launches))
    }

    /// A main layer's dense ports and operation IDs, from the declaration.
    fn dense_ports(&self, layer: u32) -> Result<&'static [(DenseRole, String)], Refusal> {
        Ok(model_qwen38_contract::precision::ports(self.shape)?.layer(layer))
    }

    pub(super) fn registered_dense(
        &self,
        id: &str,
        rows: u32,
        weights: &[DeviceBufferSpanV1],
        activation: DeviceBufferSpanV1,
        outputs: &[(DeviceBufferSpanV1, u32)],
    ) -> Result<PreparedLaunch, Refusal> {
        let handle = self
            .precision
            .ok_or(Refusal("missing prepared dense recipe"))?
            .operation(self.rank, id, self.phase)?;
        let mut side = weights.to_vec();
        side.extend(outputs.iter().map(|o| o.0));
        let mut scalars = vec![weights.len() as u64, outputs.len() as u64];
        scalars.extend(outputs.iter().map(|o| u64::from(o.1)));
        let mut input = self.projection_input(activation);
        input.side = &side;
        input.scalars = &scalars;
        let launches = shared::launch(
            handle,
            self.projection_phase(rows),
            input,
            outputs
                .first()
                .ok_or(Refusal("missing registered dense output"))?
                .0,
            &scratch(self.scratch()),
        )?;
        let [launch] = <[PreparedLaunch; 1]>::try_from(launches)
            .map_err(|_| Refusal("registered dense stage count"))?;
        Ok(launch)
    }

    fn projection_experts(
        &self,
        layer: u32,
        projection: GatedMlpProjection,
    ) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        let (name, activation, output) = match projection {
            GatedMlpProjection::GateUp => ("gate_up_proj", s.u, s.moe_intermediate),
            GatedMlpProjection::Down => ("down_proj", s.moe_intermediate, s.moe_routed),
        };
        let id = format!(
            "{}.mlp.experts.{name}",
            model_qwen38_contract::views::layer_stem_for(layer, self.shape)
        );
        let handle = self
            .precision
            .ok_or(Refusal("missing prepared experts"))?
            .operation(self.rank, &id, self.phase)?;
        let mut input = self.projection_input(activation);
        input.routing = Some(shared::Routing {
            weights: s.topk_weights,
            offsets: s.expert_offsets,
            sorted_rows: s.sorted_rows,
            padded_rows: s.padded_rows,
            active_experts: s.active_experts,
        });
        let weights = handle.weights();
        let side = [weights.codes, weights.local_scales.unwrap_or(s.unused)];
        if matches!(handle.recipe(), shared::Recipe::Registered) {
            input.side = &side;
        }
        self.stored(handle, self.projection_phase(self.rows), input, output)
    }

    /// A stored projection's launches, with the facts of their symbols: the
    /// generated precision entries were never shrunk beside the all-reduces,
    /// though their ops mark a launch that fills every SM persistent, so
    /// adopting the ops' persistence would change those grids.
    fn stored(
        &self,
        handle: &shared::Handle,
        phase: shared::Phase,
        input: shared::Input<'_>,
        output: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, Refusal> {
        Ok(self.built(shared::launch(
            handle,
            phase,
            input,
            output,
            &scratch(self.scratch()),
        )?))
    }

    /// The token embedding, prepared or the package's registered gather.
    fn projection_embedding(&self) -> Result<Vec<PreparedLaunch>, Refusal> {
        let s = self.scratch();
        let id = format!("{MODEL}.embed_tokens");
        let phase = self.projection_phase(self.rows);
        let mut input = self.projection_input(self.inputs.token_ids);
        let Some(precision) = self.precision else {
            let side = [self.view(&format!("{id}.weight"))?];
            input.side = &side;
            return self
                .lookups()?
                .embedding
                .launch(phase, input, s.embed, &scratch(s));
        };
        let handle = precision.operation(self.rank, &id, self.phase)?;
        let side = [handle.weights().codes];
        input.side = &side;
        shared::launch(handle, phase, input, s.embed, &scratch(s))
    }

    /// The n-gram hash and lookup, prepared or the package's registered
    /// FP8 lookup with its fused hash.
    fn projection_ngram(&self) -> Result<Vec<PreparedLaunch>, Refusal> {
        let id = crate::views::ngram_view_for(self.shape)
            .trim_end_matches(".weight")
            .to_owned();
        let handle = self
            .precision
            .map(|p| p.operation(self.rank, &id, self.phase))
            .transpose()?;
        let s = self.scratch();
        let staging = s.ple_staging[usize::from(self.decode())];
        let ple = format!(
            "{MODEL}.layers.{}.ple.ple_embedding",
            self.shape.ple_layer()
        );
        let mut side = vec![
            self.inputs.token_ids,
            at(staging.input, 1, self.shape.ple_conv_bytes()),
            self.view(&format!("{ple}.layer_multipliers"))?,
            self.view(&format!("{ple}.ngram_heads_vocab_sizes"))?,
            self.view(&format!("{ple}.ngram_heads_offsets"))?,
            at(staging.output, 1, self.shape.ple_conv_bytes()),
            self.inputs.q_len,
            self.side.cu_rows,
            self.inputs.live_batch,
            s.ngram_rows,
        ];
        match handle {
            Some(handle) if matches!(handle.recipe(), shared::Recipe::Registered) => {
                side.extend([
                    handle.weights().codes,
                    handle
                        .weights()
                        .local_scales
                        .ok_or(Refusal("missing lookup scale"))?,
                ]);
            }
            Some(_) => {}
            None => side.extend([
                self.view(&format!("{id}.weight"))?,
                self.view(&format!("{id}.weight_scale"))?,
            ]),
        }
        let mut scalars = vec![u64::from(self.rank), self.items.len() as u64];
        for item in self.items {
            scalars.extend([u64::from(item.start), u64::from(item.rows)]);
        }
        let phase = self.projection_phase(self.rows);
        let mut input = self.projection_input(s.ngram_rows);
        input.side = &side;
        input.scalars = &scalars;
        (input.row_offset, input.shard_rows) =
            crate::views::ngram_rows(self.rank, self.shape).map_err(Refusal)?;
        match handle {
            Some(handle) => shared::launch(handle, phase, input, s.ngram_local, &scratch(s)),
            None => self
                .lookups()?
                .ngram
                .launch(phase, input, s.ngram_local, &scratch(s)),
        }
    }
}
