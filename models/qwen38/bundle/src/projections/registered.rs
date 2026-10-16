//! Registered model adapters keep legacy fusion and hash ABI arguments.
//! Lowering without a prepared component launches the same adapters.
use super::*;
use crate::ops::{Bind, DenseOp, Key, exact, rows_need};
use crate::{lower::ItemRows, pipeline::PreparedLaunch};
use kernel_core::backend::{Backend, Counts, Need, Prepared, Refused, View};
use kernel_core::contract::{NgramQuery, RowQuery};
use kernel_core::gated_mlp::GatedMlpProjectionQuery;
use kernel_core::moe::RouteMetadata;
use kernel_core::moe_experts::{GatedMlpBuffers, GatedMlpScratch, PreparedExperts};
use kernel_core::precision::buffers::{ProjectionRouting, QuantizedWeights};
use kernel_core::precision::execution::{BoundUse, PrecisionSelection};
use kernel_core::properties::{Anchors, Invariance};
use kernel_core::rows::{EmbeddingBuffers, NgramBuffers};
use model_qwen38_contract::own_ops::dense::{DenseGroup, DenseGroupBuffers, Projection};
use model_qwen38_contract::own_ops::ngram::{Ngram, NgramBuffers as OwnNgramBuffers, NgramWrites};
use model_qwen38_contract::own_ops::{ItemForm, OwnOps};

/// Refuses an op whose kernels are not the authenticated stages.
fn check_stages(info: &kernel_core::backend::OpInfo, stages: &[String]) -> Result<(), Refused> {
    if !info.symbols.iter().map(|(symbol, _)| symbol).eq(stages) {
        return Err(Refused::Rule(format!(
            "Qwen3.8's prepared op launches {:?}, not the bound stages {stages:?}",
            info.symbols
        )));
    }
    Ok(())
}

/// The n-gram hash, alone before a shared lookup or fused with the legacy
/// FP8 lookup. Scalars are the rank, the item count, then each item's first
/// row and rows.
struct Hash {
    shape: Shape,
    legacy: bool,
    /// A legacy prefill's per-sequence hash gather, prepared for each rank.
    prefill: Vec<Prepared<NgramBuffers<View>>>,
    /// Every other form's own op (`OwnOps::ngram`), prepared for each rank.
    own: Vec<Prepared<OwnNgramBuffers<View>>>,
}
fn shifted(view: Span, bytes: u64) -> Result<Span, Refusal> {
    let length = view
        .length
        .checked_sub(bytes)
        .ok_or(Refusal("prefix offset exceeds its view"))?;
    Ok(Span::new(view.buffer, view.offset + bytes, length))
}
impl Hash {
    /// The hash `symbol` of a step of use `usage`, fused with the FP8
    /// lookup when `legacy`: a legacy prefill launches each rank's
    /// prepared hash gather from `backend`, every other form each rank's
    /// own op from `own`.
    fn new(
        shape: Shape,
        backend: &dyn Backend,
        own: &dyn OwnOps,
        symbol: String,
        legacy: bool,
        usage: OperationUse,
    ) -> Result<Self, Refused> {
        let (mut prefill, mut owned) = (Vec::new(), Vec::new());
        match usage.phase {
            ExecutionPhase::Prefill if legacy => {
                let g = shape.geometry;
                let query = NgramQuery::HashGather {
                    orders: g.ngram_orders(),
                    heads_per_order: g.ngram_heads_per_order(),
                    head_width: g.ngram_head_width(),
                    local_heads: shape.ngram_local_heads,
                    capacity: PREFILL_TOKENS,
                };
                for rank in 0..shape.ranks() {
                    let op = backend.ngram_hash_gather(
                        &query,
                        shape.ngram_eos(),
                        rank,
                        rows_need(Key::of_use(usage)),
                    )?;
                    check_stages(op.info(), std::slice::from_ref(&symbol))?;
                    prefill.push(op);
                }
            }
            phase => {
                // The registry keeps one launcher per implementation, which
                // serves every use of its phase: its ops take the phase's
                // most rows.
                let (items, capacity) = match phase {
                    ExecutionPhase::Prefill => (ItemForm::One, PREFILL_TOKENS),
                    ExecutionPhase::Decode => (ItemForm::Decode, DECODE_BATCH),
                    ExecutionPhase::Verify => (
                        ItemForm::Verify {
                            rows: MAX_VERIFY_ROWS,
                        },
                        MAX_VERIFY_TOKENS,
                    ),
                };
                // What serving relies on, as for the target's other ops: a
                // prefill item's rows keep their ids over chunks at the
                // manifest's granule, its history derived at anchors by the
                // PLE anchor; the exact verify steps among a verify
                // launcher's uses give their recorded verify rows one-row
                // bits.
                let exact = match phase {
                    ExecutionPhase::Prefill => {
                        exact(Key::Prefill, PREFILL_TOKENS, Anchors::Companion)
                    }
                    ExecutionPhase::Verify => exact(
                        Key::Verify(EXACT_VERIFY.rows),
                        MAX_VERIFY_ROWS,
                        Anchors::NotApplicable,
                    ),
                    ExecutionPhase::Decode => Invariance::NONE,
                };
                let need = Need::rows(capacity).with_exact(exact);
                let writes = if legacy {
                    NgramWrites::Gather
                } else {
                    NgramWrites::Hash
                };
                // The stage launches as the instance's geometry tags it.
                let stage = launch::tagged(&symbol, shape);
                for rank in 0..shape.ranks() {
                    let op = own.ngram(&Ngram::of(shape, rank, writes, items), need)?;
                    check_stages(op.info(), std::slice::from_ref(&stage))?;
                    owned.push(op);
                }
            }
        }
        Ok(Self {
            shape,
            legacy,
            prefill,
            own: owned,
        })
    }

    fn items(&self, input: shared::Input<'_>) -> Result<(u32, Vec<ItemRows>), Refusal> {
        let &[rank, count, ref rows @ ..] = input.scalars else {
            return Err(Refusal("missing prefix item rows"));
        };
        if rows.len() as u64 != 2 * count {
            return Err(Refusal("invalid prefix item rows"));
        }
        let narrow =
            |value: u64| u32::try_from(value).map_err(|_| Refusal("prefix row exceeds u32"));
        let items = rows
            .chunks_exact(2)
            .map(|pair| {
                Ok(ItemRows {
                    start: narrow(pair[0])?,
                    rows: narrow(pair[1])?,
                })
            })
            .collect::<Result<_, Refusal>>()?;
        // The legacy lookup appends its table and scale to the hash's inputs.
        if input.side.len() != if self.legacy { 12 } else { 10 } {
            return Err(Refusal("missing prefix side inputs"));
        }
        Ok((narrow(rank)?, items))
    }
}
impl shared::Launcher for Hash {
    fn repetitions(
        &self,
        phase: shared::Phase,
        input: shared::Input<'_>,
    ) -> Result<Vec<u32>, Refusal> {
        let count = if phase.usage.phase == ExecutionPhase::Prefill {
            self.items(input)?.1.len()
        } else {
            1
        };
        Ok(vec![
            count
                .try_into()
                .map_err(|_| Refusal("prefix item count exceeds u32"))?,
        ])
    }
    fn writes(
        &self,
        input: shared::Input<'_>,
        output: Span,
        _: &shared::Scratch,
    ) -> Result<Vec<Span>, Refusal> {
        let (_, items) = self.items(input)?;
        let mut written = if self.legacy { vec![output] } else { vec![] };
        let ids = u64::from(self.shape.ngram_local_heads) * 8;
        for (index, item) in items.iter().enumerate() {
            written.push(Span::new(
                input.activation.buffer,
                input.activation.offset + u64::from(item.start) * ids,
                u64::from(item.rows) * ids,
            ));
            written.push(Span::new(
                input.side[5].buffer,
                input.side[5].offset + index as u64 * self.shape.ple_entry_bytes(),
                8,
            ));
        }
        Ok(written)
    }
    fn launch(
        &self,
        phase: shared::Phase,
        input: shared::Input<'_>,
        output: Span,
        _: &shared::Scratch,
    ) -> Result<Vec<PreparedLaunch>, Refusal> {
        let (rank, items) = self.items(input)?;
        let entry = self.shape.ple_entry_bytes();
        if phase.usage.phase == ExecutionPhase::Prefill && self.legacy {
            let op = self
                .prefill
                .get(rank as usize)
                .ok_or(Refusal("no Qwen3.8 n-gram hash gather for the rank"))?;
            let mut launches = Vec::new();
            for (index, item) in items.iter().enumerate() {
                let start = u64::from(item.start);
                let mut bind = Bind::default();
                let buffers = NgramBuffers {
                    tokens: bind.view(shifted(input.side[0], start * 4)?),
                    history: bind.view(shifted(input.side[1], index as u64 * entry)?),
                    multipliers: bind.view(input.side[2]),
                    sizes: bind.view(input.side[3]),
                    offsets: bind.view(input.side[4]),
                    table: bind.view(input.side[10]),
                    scale: bind.view(input.side[11]),
                    output: bind.view(shifted(
                        output,
                        start * u64::from(self.shape.ngram_local) * BF16_BYTES,
                    )?),
                    row_ids: bind.view(shifted(
                        input.activation,
                        start * u64::from(self.shape.ngram_local_heads) * 8,
                    )?),
                    history_out: bind.view(shifted(input.side[5], index as u64 * entry)?),
                    live_rows: bind.view(Span::new(
                        input.side[6].buffer,
                        input.side[6].offset + index as u64 * 4,
                        4,
                    )),
                    table_start: input.row_offset,
                    table_rows: input.shard_rows,
                };
                launches.extend(bind.launch(op, &buffers, Counts::rows(item.rows))?);
            }
            return Ok(launches);
        }
        let op = self
            .own
            .get(rank as usize)
            .ok_or(Refusal("no Qwen3.8 n-gram op for the rank"))?;
        if phase.usage.phase == ExecutionPhase::Prefill {
            // One launch per item, its views offset to the item.
            let mut launches = Vec::new();
            for (index, item) in items.iter().enumerate() {
                let start = u64::from(item.start);
                let mut bind = Bind::default();
                let buffers = self.own_buffers(
                    &mut bind,
                    &input,
                    [
                        shifted(input.side[0], start * 4)?,
                        shifted(input.side[1], index as u64 * entry)?,
                        shifted(
                            input.activation,
                            start * u64::from(self.shape.ngram_local_heads) * 8,
                        )?,
                        shifted(input.side[5], index as u64 * entry)?,
                        shifted(
                            output,
                            start * u64::from(self.shape.ngram_local) * BF16_BYTES,
                        )?,
                        Span::new(
                            input.side[6].buffer,
                            input.side[6].offset + index as u64 * 4,
                            4,
                        ),
                    ],
                    false,
                );
                launches.extend(bind.launch(op, &buffers, Counts::rows(item.rows))?);
            }
            return Ok(launches);
        }
        let verify = phase.usage.phase == ExecutionPhase::Verify;
        let mut bind = Bind::default();
        let buffers = self.own_buffers(
            &mut bind,
            &input,
            [
                input.side[0],
                input.side[1],
                input.activation,
                input.side[5],
                output,
                input.side[6],
            ],
            verify,
        );
        let counts = if verify {
            Counts::rows(phase.rows).with_items(items.len() as u32, 0)
        } else {
            Counts::rows(phase.rows)
        };
        bind.launch(op, &buffers, counts)
    }
}

impl Hash {
    /// The own op's buffers over the tokens, histories, row ids, output
    /// histories, output rows and row counts `spans`, with the input's
    /// tables, a gather's table and scale, and a verify step's item
    /// tables.
    fn own_buffers(
        &self,
        bind: &mut Bind,
        input: &shared::Input<'_>,
        [tokens, history, ids, history_out, output, q_len]: [Span; 6],
        verify: bool,
    ) -> OwnNgramBuffers<View> {
        let gather = self.legacy;
        OwnNgramBuffers {
            tokens: bind.view(tokens),
            history: bind.view(history),
            multipliers: bind.view(input.side[2]),
            sizes: bind.view(input.side[3]),
            offsets: bind.view(input.side[4]),
            table: gather.then(|| bind.view(input.side[10])),
            scale: gather.then(|| bind.view(input.side[11])),
            output: gather.then(|| bind.view(output)),
            row_ids: bind.view(ids),
            history_out: bind.view(history_out),
            cu_rows: verify.then(|| bind.view(input.side[7])),
            q_len: bind.view(q_len),
            live_items: verify.then(|| bind.view(input.side[8])),
            table_start: input.row_offset,
            table_rows: input.shard_rows,
        }
    }
}

/// The FP8 fragment experts of a registered binding, through the expert
/// op prepared from the entry's contract at the binding's use.
struct Experts {
    op: Box<dyn PreparedExperts>,
    needs: shared::Needs,
}

impl Experts {
    /// The experts computing `contract` into `output_bytes` of output.
    fn new(
        backend: &dyn Backend,
        contract: &GatedMlpProjectionQuery,
        output_bytes: u64,
        binding: &BoundUse,
    ) -> Result<Self, Refused> {
        let usage = binding.usage;
        let op = backend.gated_mlp_projection(contract, usage, rows_need(Key::of_use(usage)))?;
        check_stages(op.info(), &binding.implementation.stages)?;
        Ok(Self {
            op,
            needs: shared::Needs {
                output_bytes,
                ..shared::Needs::default()
            },
        })
    }
}

/// Weight views and `(output, row stride)` pairs.
type DensePorts<'a> = (&'a [Span], Vec<(Span, u32)>);

/// A fused BF16 dense entry, through the dense group launching it at each
/// use bound to it. Scalars are the weight and output counts, then each
/// output's row stride; side spans are the weights, then the outputs.
struct Dense {
    /// Each use's group and its outputs' row strides.
    ops: Vec<(OperationUse, DenseOp, Vec<u32>)>,
    needs: shared::Needs,
}

/// The dense projections a precision port's registered entry can launch,
/// in the order a shared entry is matched (the MTP fc shares the linear
/// QKV entry at TP4, so the port's projection comes first).
const PORTED: [Projection; 11] = [
    Projection::LinearIn,
    Projection::LinearQkv,
    Projection::LinearZ,
    Projection::LinearAb,
    Projection::AttentionIn,
    Projection::AttentionQkv,
    Projection::IndexQk,
    Projection::LinearOutput,
    Projection::AttentionOutput,
    Projection::SharedGateUp,
    Projection::SharedDown,
];

impl Dense {
    /// The groups launching `symbol` at each of `uses`, prepared on `own`.
    fn new(
        shape: Shape,
        own: &dyn OwnOps,
        symbol: &str,
        uses: impl Iterator<Item = OperationUse>,
        needs: shared::Needs,
    ) -> Result<Self, Refused> {
        let ops = uses
            .map(|usage| {
                let need = rows_need(Key::of_use(usage));
                PORTED
                    .iter()
                    .find_map(|&projection| {
                        let group = DenseGroup::of(shape, projection);
                        let op = own.dense_group(&group, usage, need).ok()?;
                        (op.info().symbols[0].0 == symbol).then_some((usage, op, group.strides))
                    })
                    .ok_or_else(|| {
                        Refused::Rule(format!(
                            "Qwen3.8 has no dense group launching {symbol} at {usage:?}"
                        ))
                    })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { ops, needs })
    }
}

impl Dense {
    fn ports(input: shared::Input<'_>) -> Result<DensePorts<'_>, Refusal> {
        let &[weights, outputs, ref strides @ ..] = input.scalars else {
            return Err(Refusal("missing registered dense port counts"));
        };
        let weights =
            usize::try_from(weights).map_err(|_| Refusal("dense weight count exceeds usize"))?;
        let outputs =
            usize::try_from(outputs).map_err(|_| Refusal("dense output count exceeds usize"))?;
        if input.side.len() != weights + outputs
            || strides.len() != outputs
            || weights == 0
            || outputs == 0
        {
            return Err(Refusal("invalid registered dense ports"));
        }
        let (weights, outputs) = input.side.split_at(weights);
        let outputs = outputs
            .iter()
            .zip(strides)
            .map(|(&span, &stride)| {
                Ok((
                    span,
                    u32::try_from(stride).map_err(|_| Refusal("dense stride exceeds u32"))?,
                ))
            })
            .collect::<Result<_, Refusal>>()?;
        Ok((weights, outputs))
    }
}

impl shared::Launcher for Dense {
    fn needs(&self) -> shared::Needs {
        self.needs
    }

    fn writes(
        &self,
        input: shared::Input<'_>,
        _: Span,
        scratch: &shared::Scratch,
    ) -> Result<Vec<Span>, Refusal> {
        let (_, outputs) = Self::ports(input)?;
        let mut regions: Vec<Span> = Vec::new();
        for (span, _) in outputs {
            if let Some(region) = regions.iter_mut().find(|r| r.buffer == span.buffer) {
                let end = (region.offset + region.length).max(span.offset + span.length);
                region.offset = region.offset.min(span.offset);
                region.length = end - region.offset;
            } else {
                regions.push(span);
            }
        }
        if self.needs.workspace_bytes != 0 {
            regions.push(scratch.partial);
        }
        if self.needs.semaphore_bytes != 0 {
            regions.push(scratch.semaphores);
        }
        Ok(regions)
    }

    fn launch(
        &self,
        phase: shared::Phase,
        input: shared::Input<'_>,
        _: Span,
        scratch: &shared::Scratch,
    ) -> Result<Vec<PreparedLaunch>, Refusal> {
        let (weights, outputs) = Self::ports(input)?;
        let (_, op, strides) = self
            .ops
            .iter()
            .find(|(usage, ..)| *usage == phase.usage)
            .ok_or(Refusal("registered dense entry is not bound at this use"))?;
        if !outputs
            .iter()
            .map(|&(_, stride)| stride)
            .eq(strides.iter().copied())
        {
            return Err(Refusal("registered dense outputs are not the group's"));
        }
        let mut bind = Bind::default();
        let buffers = DenseGroupBuffers {
            weights: weights.iter().map(|&weight| bind.view(weight)).collect(),
            activation: bind.view(input.activation),
            outputs: outputs
                .iter()
                .map(|&(output, _)| bind.view(output))
                .collect(),
            bottleneck: None,
            partial: bind.view(scratch.partial),
            semaphores: bind.view(scratch.semaphores),
        };
        bind.launch(op, &buffers, Counts::rows(phase.rows))
    }
}

/// The token embedding gather. Its one side span is the table.
struct Embedding {
    op: Prepared<EmbeddingBuffers<View>>,
    vocab: u32,
}

impl Embedding {
    /// The embedding `symbol` of a step of use `usage`.
    fn new(
        shape: Shape,
        backend: &dyn Backend,
        symbol: &str,
        usage: OperationUse,
    ) -> Result<Self, Refused> {
        let vocab = shape.geometry.vocab();
        let op = backend.embedding_gather(
            &RowQuery::EmbeddingGather {
                vocab,
                width: shape.hidden(),
            },
            0,
            rows_need(Key::of_use(usage)),
        )?;
        check_stages(op.info(), &[symbol.to_owned()])?;
        Ok(Self { op, vocab })
    }
}
impl shared::Launcher for Embedding {
    fn writes(
        &self,
        _: shared::Input<'_>,
        output: Span,
        _: &shared::Scratch,
    ) -> Result<Vec<Span>, Refusal> {
        Ok(vec![output])
    }
    fn launch(
        &self,
        phase: shared::Phase,
        input: shared::Input<'_>,
        output: Span,
        _: &shared::Scratch,
    ) -> Result<Vec<PreparedLaunch>, Refusal> {
        let &[weight] = input.side else {
            return Err(Refusal("missing registered embedding table"));
        };
        let mut bind = Bind::default();
        let buffers = EmbeddingBuffers {
            table: bind.view(weight),
            ids: bind.view(input.activation),
            output: bind.view(output),
            live_rows: bind.view(input.live_rows),
            table_rows: self.vocab,
        };
        bind.launch(&self.op, &buffers, Counts::rows(phase.rows))
    }
}

/// Bindings served by a registered native implementation rather than a
/// generated offer: the fused BF16 dense entries, the FP8 fragment experts
/// and the unpacked lookups.
pub(super) fn registered(selection: &PrecisionSelection, binding: &BoundUse) -> bool {
    match selection.effective.kind {
        OperationKind::Dense => binding
            .implementation
            .implementation
            .starts_with("qwen38.legacy-dense."),
        OperationKind::Grouped { .. } => matches!(
            selection.effective.weights[0].format,
            WeightFormat::Fp8Block128x128(_)
        ),
        OperationKind::Lookup => selection.effective.weights[0].format != WeightFormat::Nvfp4,
    }
}

/// The registered stage of a binding, or `None` for a generated offer
/// without a registered prefix.
/// Lookups and experts prepare their ops on `backend` (the n-gram forms on
/// `own`) at the binding's use and check them against its stages; a dense
/// entry prepares on `own` the group launching it at each use of
/// `document` bound to it.
pub(super) fn launcher(
    shape: Shape,
    target: &dyn crate::target::Target,
    backend: &dyn Backend,
    own: &dyn OwnOps,
    document: &PrecisionDocument,
    selection: &PrecisionSelection,
    binding: &BoundUse,
) -> Result<Option<Arc<dyn shared::Launcher>>, String> {
    let symbol = &binding.implementation.stages[0];
    let format = selection.effective.weights[0].format;
    let usage = binding.usage;
    if !registered(selection, binding) {
        // A packed lookup's registered prefix is the hash alone.
        if selection.effective.kind != OperationKind::Lookup {
            return Ok(None);
        }
        let hash = Hash::new(shape, backend, own, symbol.clone(), false, usage)
            .map_err(|refused| refused.to_string())?;
        return Ok(Some(Arc::new(hash)));
    }
    Ok(Some(match selection.effective.kind {
        OperationKind::Dense => {
            let needs = target.registered_dense(symbol)?;
            let uses = bound(document, binding).map(|other| other.usage);
            Arc::new(Dense::new(shape, own, symbol, uses, needs).map_err(|r| r.to_string())?)
        }
        OperationKind::Grouped { .. } => {
            let (contract, output_bytes) = target.registered_experts(symbol)?;
            Arc::new(
                Experts::new(backend, &contract, output_bytes, binding)
                    .map_err(|r| r.to_string())?,
            )
        }
        OperationKind::Lookup if format == WeightFormat::Bf16 => {
            Arc::new(Embedding::new(shape, backend, symbol, usage).map_err(|r| r.to_string())?)
        }
        OperationKind::Lookup => Arc::new(
            Hash::new(shape, backend, own, symbol.clone(), true, usage)
                .map_err(|r| r.to_string())?,
        ),
    }))
}

/// Every use of `document` bound to `binding`'s implementation.
fn bound<'a>(
    document: &'a PrecisionDocument,
    binding: &'a BoundUse,
) -> impl Iterator<Item = &'a BoundUse> {
    document
        .operations
        .iter()
        .flat_map(|selection| &selection.bindings)
        .filter(|other| other.implementation == binding.implementation)
}

/// The registered lookup a package without a prepared component launches
/// at use `usage`: the token embedding or the FP8 n-gram lookup with its
/// fused hash, prepared on `backend` and `own` once at create. Its ops are
/// checked against the entry the contract's backend registers for the
/// lookup (`precision::legacy_lookup_capability`).
pub(crate) fn legacy_lookup(
    shape: Shape,
    backend: &dyn Backend,
    own: &dyn OwnOps,
    ngram: bool,
    usage: OperationUse,
) -> Result<Arc<dyn shared::Launcher>, Refused> {
    let capability =
        model_qwen38_contract::precision::legacy_lookup_capability(shape, ngram, usage)?;
    let [symbol] = capability.stages.as_slice() else {
        return Err(Refused::Rule(format!(
            "Qwen3.8's registered lookup has stages {:?}, not one entry",
            capability.stages
        )));
    };
    Ok(if ngram {
        Arc::new(Hash::new(shape, backend, own, symbol.clone(), true, usage)?)
    } else {
        Arc::new(Embedding::new(shape, backend, symbol, usage)?)
    })
}

impl shared::Launcher for Experts {
    fn needs(&self) -> shared::Needs {
        self.needs
    }
    fn writes(
        &self,
        _: shared::Input<'_>,
        output: Span,
        _: &shared::Scratch,
    ) -> Result<Vec<Span>, Refusal> {
        Ok(vec![output])
    }
    fn launch(
        &self,
        phase: shared::Phase,
        input: shared::Input<'_>,
        output: Span,
        scratch: &shared::Scratch,
    ) -> Result<Vec<PreparedLaunch>, Refusal> {
        let route = input
            .routing
            .ok_or(Refusal("registered experts require routing"))?;
        let &[codes, scales] = input.side else {
            return Err(Refusal("registered experts require packed weight views"));
        };
        let mut bind = Bind::default();
        let buffers = GatedMlpBuffers {
            activation: bind.view(input.activation),
            activation_scales: None,
            activation_global: None,
            live_tokens: bind.view(input.live_rows),
            output: bind.view(output),
            weights: QuantizedWeights {
                codes: bind.view(codes),
                local_scales: Some(bind.view(scales)),
                global_scales: None,
            },
            routing: Some(ProjectionRouting {
                weights: bind.view(route.weights),
                routes: RouteMetadata {
                    counts: None,
                    offsets: bind.view(route.offsets),
                    sorted_rows: bind.view(route.sorted_rows),
                    padded_rows: bind.view(route.padded_rows),
                    active_experts: Some(bind.view(route.active_experts)),
                },
            }),
            scratch: GatedMlpScratch {
                partial: None,
                output_codes: None,
                output_scales: None,
                unused: bind.view(scratch.unused),
            },
        };
        bind.launch_experts(self.op.as_ref(), &buffers, Counts::rows(phase.rows))
    }
}
