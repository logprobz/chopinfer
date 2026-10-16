//! Lowering of the latent KV write, the indexer selection and the latent
//! attention for a context-parallel group of 2 or 4 ranks
//! (`crate::context_parallel`). At size 1 none of these lowerings runs and
//! the stages lower as before.

use bundle_runtime::upload::DeviceInputs;
use kernel_core::backend::{Counts, Prepared, View};
use kernel_core::collective::{GatherBuffers, RankViews};
use kernel_core::context_parallel::{
    GroupTransferBuffers, LocalizeKeysBuffers, PartialMergeBuffers,
};
use kernel_core::latent_attention::{LatentAttentionBuffers, LatentPartial};
use kernel_core::pooled_indexer::{IndexerMergeBuffers, IndexerSelectBuffers};
use model_glm5next_contract::contract::latent_format;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Lowering, Value};
use crate::buffers::{Buffer, SCRATCH_ROWS};
use crate::context_parallel::{
    ContextParallel, GatherProtocol, RankState, block_floats, bucket, bucket_index, pair_words,
};
use crate::ops::cp::{CpOps, GroupTransfers};
use crate::ops::{Bind, Lowered};
use crate::program::{Phase, Stage};

const BF16: u64 = 2;
const F32: u64 = 4;
const I32: u64 = 4;

/// `span` from `offset` bytes for `length` bytes, which it must hold.
fn view(span: DeviceBufferSpanV1, offset: u64, length: u64) -> Result<DeviceBufferSpanV1, String> {
    if offset + length > span.length {
        return Err(format!(
            "a context-parallel view of {length} bytes at {offset} exceeds {span:?}"
        ));
    }
    Ok(DeviceBufferSpanV1::new(
        span.buffer,
        span.offset + offset,
        length,
    ))
}

/// `span` from `offset` bytes to its end.
fn from(span: DeviceBufferSpanV1, offset: u64) -> Result<DeviceBufferSpanV1, String> {
    view(span, offset, span.length.saturating_sub(offset))
}

/// The group stages through the instance's context-parallel ops
/// (`crate::ops::cp`), the indexer's per-rank selections and the latent
/// family's partial attention.
impl Lowering<'_> {
    /// The group size: 1 without context parallelism.
    pub(crate) fn cp_size(&self) -> u32 {
        self.context_parallel.map_or(1, |cp| cp.size)
    }

    /// The rank's place in its group: 0 without context parallelism.
    pub(crate) fn cp_rank(&self) -> u32 {
        self.rank % self.cp_size()
    }

    /// The rank's place in its group of `cp.size` consecutive ranks, and
    /// the group's ranks in group rank order.
    fn place(&self, cp: &ContextParallel) -> Result<(u32, std::ops::Range<u32>), String> {
        let ranks = self.peers.len() as u32;
        if cp.size == 0 || !ranks.is_multiple_of(cp.size) || self.rank >= ranks {
            return Err(format!(
                "a context-parallel group of {} does not divide {ranks} ranks",
                cp.size
            ));
        }
        let first = self.rank - self.rank % cp.size;
        Ok((self.rank % cp.size, first..first + cp.size))
    }

    fn own<'b>(&self, cp: &'b ContextParallel) -> Result<&'b RankState, String> {
        cp.ranks
            .get(self.rank as usize)
            .ok_or_else(|| format!("no context-parallel state of rank {}", self.rank))
    }

    /// The launches that replace `stage`'s in a group of 2 or more ranks,
    /// or `None` when the stage lowers as at size 1.
    pub(crate) fn context_parallel_stage(
        &self,
        phase: Phase,
        stage: Stage,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(cp) = self.context_parallel.filter(|cp| cp.size > 1) else {
            return Ok(None);
        };
        Ok(Some(match stage {
            Stage::LatentKvWrite(layer) => self.owner_kv_write(cp, phase, layer)?,
            Stage::IndexSelect(layer) => self.in_pieces(phase, |lowering, phase| {
                lowering.group_select(cp, phase, layer)
            })?,
            Stage::LatentAttention(layer) => self.in_pieces(phase, |lowering, phase| {
                lowering.group_attention(cp, phase, layer)
            })?,
            _ => return Ok(None),
        }))
    }

    /// The instance's context-parallel ops.
    fn cp_ops(&self) -> Result<&CpOps, String> {
        self.ops
            .cp
            .as_ref()
            .ok_or_else(|| "no context-parallel ops at group size 1".into())
    }

    /// The rank's localization of `rows` rows of `lists` into `out` by
    /// `ops` (one per group rank).
    fn localize(
        &self,
        cp: &ContextParallel,
        ops: &[Prepared<LocalizeKeysBuffers<View>>],
        rows: u32,
        lists: DeviceBufferSpanV1,
        out: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, String> {
        let op = of_rank(ops, self.place(cp)?.0)?;
        let mut bind = Bind::default();
        let buffers = LocalizeKeysBuffers {
            lists: bind.view(lists),
            live_rows: bind.view(self.inputs.live_tokens),
            output: bind.view(out),
        };
        bind.launch(op, &buffers, Counts::rows(rows))
            .map_err(|refusal| format!("context-parallel localization: {refusal}"))
    }

    /// Each appended row's position localized to the rank that owns its
    /// page (-1 elsewhere), then the latent KV write of the owned rows.
    fn owner_kv_write(
        &self,
        cp: &ContextParallel,
        phase: Phase,
        layer: u32,
    ) -> Result<Vec<Lowered>, String> {
        let own = self.own(cp)?;
        let rows = phase.rows();
        let positions = view(own.positions, 0, u64::from(rows) * I32)?;
        let mut launches = self.localize(
            cp,
            &self.cp_ops()?.positions,
            rows,
            self.inputs.positions,
            positions,
        )?;
        let inputs = DeviceInputs {
            positions,
            ..*self.inputs
        };
        let owned = Lowering {
            inputs: &inputs,
            ..*self
        };
        launches.push(owned.kv_write(phase, layer)?);
        Ok(launches)
    }

    /// The group transfer `op` (the rank's) of `input` into the members'
    /// `output`s over their `protocol`s.
    fn group_transfer(
        &self,
        cp: &ContextParallel,
        ops: &[Prepared<GroupTransferBuffers<View>>],
        rows: u32,
        input: DeviceBufferSpanV1,
        output: impl Fn(&RankState) -> Result<DeviceBufferSpanV1, String>,
        protocol: impl Fn(&RankState) -> Result<GatherProtocol, String>,
    ) -> Result<Vec<Lowered>, String> {
        let (rank, members) = self.place(cp)?;
        let op = of_rank(ops, rank)?;
        let mut bind = Bind::default();
        let buffers = GatherBuffers {
            input: bind.view(input),
            ranks: members
                .map(|member| {
                    let state = cp
                        .ranks
                        .get(member as usize)
                        .ok_or_else(|| format!("no context-parallel state of rank {member}"))?;
                    let protocol = protocol(state)?;
                    Ok(RankViews {
                        output: bind.view(output(state)?),
                        state: vec![bind.view(protocol.flags), bind.view(protocol.epoch)],
                    })
                })
                .collect::<Result<_, String>>()?,
        };
        bind.launch(op, &buffers, Counts::rows(rows))
            .map_err(|refusal| format!("context-parallel group transfer: {refusal}"))
    }

    /// The transfers of `rows` rows' bucket.
    fn transfers(&self, rows: u32) -> Result<&GroupTransfers, String> {
        self.cp_ops()?
            .buckets
            .get(bucket_index(rows)?)
            .ok_or_else(|| format!("no group transfers of {rows} rows"))
    }

    /// The indexer's selection over the rank's own pools as (score, pool)
    /// pairs, the group all-gather of the pairs, and their exact merge into
    /// every rank's key lists.
    fn group_select(
        &self,
        cp: &ContextParallel,
        phase: Phase,
        layer: u32,
    ) -> Result<Vec<Lowered>, String> {
        let (rank, _) = self.place(cp)?;
        let own = self.own(cp)?;
        let rows = phase.rows();
        let bucket = bucket(rows);
        let index = bucket_index(rows)?;
        let pairs = view(own.pairs, 0, pair_words(self.shape, bucket) * 4)?;
        let select = self
            .ops
            .indexer
            .select(rank)
            .ok_or("no indexer selection at the rank's place in its group")?;
        let mut bind = Bind::default();
        let buffers = IndexerSelectBuffers {
            q: bind.view(self.scratch(Buffer::IndexQuery)),
            head_weights: Some(bind.view(self.scratch(Buffer::IndexWeights))),
            pooled: self.pooled_pages(&mut bind, layer)?,
            row_batch: bind.view(self.inputs.token_batch),
            positions: bind.view(self.inputs.positions),
            live_rows: bind.view(self.inputs.live_tokens),
            workspace: bind.view(self.scratch(Buffer::Workspace)),
            output: bind.view(pairs),
        };
        let mut launches = bind
            .launch(select, &buffers, Counts::rows(rows))
            .map_err(|refusal| format!("context-parallel indexer selection: {refusal}"))?;
        launches.extend(self.group_transfer(
            cp,
            &self.transfers(rows)?.pairs,
            rows,
            pairs,
            |state| {
                state
                    .gathered_pairs
                    .get(index)
                    .copied()
                    .ok_or_else(|| format!("no gathered pairs of {bucket} rows"))
            },
            |state| {
                state
                    .pair_gathers
                    .get(index)
                    .copied()
                    .ok_or_else(|| format!("no pair gather state of {bucket} rows"))
            },
        )?);
        let gathered = own
            .gathered_pairs
            .get(index)
            .copied()
            .ok_or_else(|| format!("no gathered pairs of {bucket} rows"))?;
        let mut bind = Bind::default();
        let buffers = IndexerMergeBuffers {
            pairs: bind.view(gathered),
            rank_rows: bucket,
            positions: Some(bind.view(self.inputs.positions)),
            live_rows: bind.view(self.inputs.live_tokens),
            output: bind.view(self.scratch(Buffer::KeyLists)),
        };
        launches.extend(
            bind.launch(&self.cp_ops()?.merge, &buffers, Counts::rows(rows))
                .map_err(|refusal| format!("context-parallel selection merge: {refusal}"))?,
        );
        Ok(launches)
    }

    /// The latent attention of the rank's heads over the group's keys: key
    /// list localization, the group all-gather of the absorbed queries, the
    /// partial attention of each source block over the rank's own keys,
    /// the exchange of each block to the rank that owns its heads, the
    /// merge of the rank's blocks into `AttentionOut`, and the group
    /// barrier that ends the piece.
    fn group_attention(
        &self,
        cp: &ContextParallel,
        phase: Phase,
        layer: u32,
    ) -> Result<Vec<Lowered>, String> {
        let own = self.own(cp)?;
        let ops = self.cp_ops()?;
        let p = self.shape.profile;
        let (heads, width) = (p.latent_heads, p.kv_rank);
        let rows = phase.rows();
        let bucket = bucket(rows);
        let index = bucket_index(rows)?;
        let list_width = u64::from(self.shape.latent_keys());
        let first = u64::from(self.row_offset);
        let transfers = self.transfers(rows)?;
        // 1. The rank's own key lists, at the piece's rows, from the global
        // lists the size-1 attention reads: the step's selection, or a
        // draft step's shared lists (`Lowering::common`).
        let Some(Value::Span(global)) = self.common("key_lists") else {
            unreachable!("the key lists are a scratch span")
        };
        // The rank's own buffer holds its localized key lists.
        let lists = from(own.local, first * list_width * I32)?;
        let mut launches = self.localize(cp, &ops.lists, rows, global, lists)?;
        // 2. The gathered queries: a bucket of rows from the piece's first,
        // moved back to end inside the scratch; the piece's rows start
        // `shift` rows into each block.
        let start = self.row_offset.min(SCRATCH_ROWS - bucket);
        let shift = u64::from(self.row_offset - start);
        let row_bytes = u64::from(heads) * u64::from(width) * BF16;
        let queries = view(
            self.buffers.get(Buffer::QueryAbsorbed),
            u64::from(start) * row_bytes,
            u64::from(bucket) * row_bytes,
        )?;
        launches.extend(self.group_transfer(
            cp,
            &transfers.queries,
            rows,
            queries,
            |state| Ok(state.queries),
            |state| {
                state
                    .query_gathers
                    .get(index)
                    .copied()
                    .ok_or_else(|| format!("no query gather state of {bucket} rows"))
            },
        )?);
        // 3. One partial attention per source block, into the destination's
        // block of the send buffer.
        let attention = self
            .ops
            .latent
            .attention(latent_format(self.format), self.attention_phase(phase));
        let ordinal = self.sparse(layer)?;
        let block = block_floats(self.shape, bucket);
        let pairs = u64::from(bucket) * u64::from(heads) * F32;
        let source_block = u64::from(bucket) * row_bytes;
        for source in 0..cp.size {
            let q = from(
                own.queries,
                u64::from(source) * source_block + shift * row_bytes,
            )?;
            let send = view(own.send, u64::from(source) * block * F32, block * F32)?;
            let mut bind = Bind::default();
            let buffers = LatentAttentionBuffers {
                q: bind.view(q),
                cache: self.latent_cache(&mut bind, ordinal),
                block_tables: self.block_tables(&mut bind),
                row_batch: bind.view(self.inputs.token_batch),
                key_lists: bind.view(lists),
                budget: self.index_topk + p.index_pool - 1,
                output: None,
                partial: Some(LatentPartial {
                    max: bind.view(view(send, 0, pairs)?),
                    den: bind.view(view(send, pairs, pairs)?),
                    num: bind.view(from(send, 2 * pairs)?),
                }),
                workspace: bind.view(self.scratch(Buffer::Workspace)),
                live_rows: bind.view(self.inputs.live_tokens),
            };
            launches.extend(
                bind.launch(attention, &buffers, Counts::rows(rows))
                    .map_err(|refusal| format!("context-parallel partial attention: {refusal}"))?,
            );
        }
        // 4. After every partial attention: block o of each member's send
        // buffer to group rank o, compact source-major in o's receive
        // buffer, in one exchange over the bucket's state.
        let send = view(own.send, 0, u64::from(cp.size) * block * F32)?;
        launches.extend(self.group_transfer(
            cp,
            &transfers.partials,
            rows,
            send,
            |state| Ok(state.receive),
            |state| {
                state
                    .partial_gathers
                    .get(index)
                    .copied()
                    .ok_or_else(|| format!("no partial gather state of {bucket} rows"))
            },
        )?);
        // 5. The rank's own heads, merged over the group's blocks: each
        // block holds the bucket's rows of the rank's heads.
        let received = view(own.receive, 0, u64::from(cp.size) * block * F32)?;
        let out = from(self.buffers.get(Buffer::AttentionOut), first * row_bytes)?;
        let mut bind = Bind::default();
        let buffers = PartialMergeBuffers {
            partials: bind.view(received),
            rank_rows: bucket,
            input_heads: heads,
            head_offset: 0,
            live_rows: bind.view(self.inputs.live_tokens),
            output: bind.view(out),
        };
        launches.extend(
            bind.launch(&ops.partial_merge, &buffers, Counts::rows(rows))
                .map_err(|refusal| format!("context-parallel partial merge: {refusal}"))?,
        );
        // 6. The piece ends once every member has merged: no member's next
        // exchange or query gather overwrites a buffer a peer still reads.
        launches.extend(self.group_transfer(
            cp,
            &ops.barrier,
            rows,
            own.barrier_input,
            |state| Ok(state.barrier_output),
            |state| Ok(state.barrier),
        )?);
        Ok(launches)
    }
}

/// Group rank `rank`'s op of `ops`.
fn of_rank<T>(ops: &[T], rank: u32) -> Result<&T, String> {
    ops.get(rank as usize)
        .ok_or_else(|| format!("no context-parallel op of group rank {rank}"))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use bundle_runtime::dispatch;
    use bundle_runtime::fake_host::FakeDevice;
    use kernel_core::latent_attention::LatentKvFormat;
    use model_glm5next_contract::contract::{Shape, has_mtp_layer};
    use runtime_bundle_abi::{BundleStreamHandleV1, KernelArgumentV1};

    use super::*;
    use crate::journal_tests::{FORMATS, check_operands, extents, head_rows, inputs, rank_buffers};
    use crate::launch::drafter_records;
    use crate::program::DraftPass;

    /// The MTP drafter's passes at group sizes 2 and 4 lower on every rank,
    /// and the latent attention localizes the key lists the size-1
    /// attention reads: the extend pass's own selection, and a draft step's
    /// shared lists (`DraftShareSelection`), not the extend pass's stale
    /// `KeyLists` rows. Whether a catch-up pass takes the verify path (30
    /// or 32 rows) or the prefill path (36 rows), and in a draft step, each
    /// source block's partial attention is the prefill entry's one split
    /// per row with the same scalars, and no merge.
    #[test]
    fn drafter_passes_localize_the_lists_their_attention_reads() {
        let shape = Shape::TP4;
        for kv in FORMATS {
            let format = shape
                .kv_formats()
                .into_iter()
                .find(|format| latent_format(format) == kv && has_mtp_layer(format))
                .unwrap();
            for size in [2_u32, 4] {
                let ops = crate::ops::Ops::for_tests(shape, size);
                let device: FakeDevice = FakeDevice::default();
                let ranks = (0..shape.ranks())
                    .map(|rank| rank_buffers(&device, rank, &format))
                    .collect::<Vec<_>>();
                let dims = model_glm5next_contract::context_parallel::Dims::of(shape);
                let bytes =
                    model_glm5next_contract::context_parallel::rank_bytes(dims, size).unwrap();
                let cp = ContextParallel {
                    size,
                    ranks: (0..shape.ranks())
                        .map(|rank| {
                            let stream = BundleStreamHandleV1::from_parts(1, 3000 + rank);
                            RankState::allocate(&bytes, |bytes| {
                                let id = device.id();
                                Ok::<_, ()>(DeviceBufferSpanV1::new(
                                    device.allocate(id, stream, bytes, 256),
                                    0,
                                    bytes,
                                ))
                            })
                            .unwrap()
                        })
                        .collect(),
                };
                let peers = ranks
                    .iter()
                    .map(|buffers| buffers.comm.clone())
                    .collect::<Vec<_>>();
                let localize = crate::sm120::engines::Engines::localize_keys(shape, size);
                let lists_argument = localize
                    .params()
                    .iter()
                    .position(|param| {
                        matches!(param, kernel_build::EntryParam::DevicePointer { name, .. }
                            if name == "lists")
                    })
                    .unwrap();
                let mut attention = BTreeSet::new();
                for (pass, phase, read) in [
                    (
                        DraftPass::Extend,
                        Phase::Verify { rows: 30, items: 8 },
                        Buffer::KeyLists,
                    ),
                    (
                        DraftPass::Extend,
                        Phase::Verify { rows: 32, items: 8 },
                        Buffer::KeyLists,
                    ),
                    (
                        DraftPass::Extend,
                        Phase::Prefill { rows: 36 },
                        Buffer::KeyLists,
                    ),
                    (DraftPass::Step, Phase::Decode(8), Buffer::SharedKeyLists),
                ] {
                    let records = drafter_records(shape, kv, phase, pass).unwrap();
                    for (rank, buffers) in ranks.iter().enumerate() {
                        let rank = rank as u32;
                        let verify_tokens = inputs(rank, Phase::Verify { rows: 32, items: 8 });
                        let step_inputs = inputs(rank, phase);
                        let mut extents = extents(rank, buffers, &peers, phase);
                        for span in cp.ranks.iter().flat_map(RankState::spans) {
                            let end = extents.entry(span.buffer.slot()).or_default();
                            *end = (*end).max(span.offset + span.length);
                        }
                        let lowering = Lowering {
                            shape,
                            format: &format,
                            buffers,
                            inputs: &step_inputs,
                            ops: &ops,
                            rank,
                            index_topk: shape.profile.index_topk_max,
                            pieces: &[],
                            row_offset: 0,
                            peers: &peers,
                            head_rows: head_rows(phase),
                            context_parallel: Some(&cp),
                            draft: Some(pass),
                            verify_tokens: Some(verify_tokens.token_ids),
                            projections: None,
                        };
                        let mut localized = Vec::new();
                        for record in &records {
                            let Some(launches) = lowering.lower(phase, record.stage).unwrap()
                            else {
                                continue;
                            };
                            for launch in launches {
                                let arguments =
                                    dispatch::arguments(&launch.arguments, &launch.bindings)
                                        .unwrap();
                                check_operands(rank, &launch.symbol, &arguments, &extents);
                                if matches!(record.stage, Stage::LatentAttention(_))
                                    && launch.symbol == localize.symbol()
                                {
                                    localized.push(arguments[lists_argument].clone());
                                }
                                if launch.symbol.contains("_attn_") {
                                    let scalars = arguments
                                        .iter()
                                        .filter(|argument| {
                                            !matches!(
                                                argument,
                                                KernelArgumentV1::DeviceBuffer(_)
                                                    | KernelArgumentV1::NullDevicePointer
                                            )
                                        })
                                        .map(|argument| format!("{argument:?}"))
                                        .collect::<Vec<_>>();
                                    attention.insert((launch.symbol.clone(), scalars));
                                }
                            }
                        }
                        assert_eq!(
                            localized,
                            [KernelArgumentV1::DeviceBuffer(buffers.get(read))],
                            "{kv:?} size {size} {pass:?} rank {rank}"
                        );
                    }
                }
                assert_eq!(attention.len(), 1, "{kv:?} size {size}: {attention:?}");
                // FP8 pages run the entry's slice-widened form.
                let (symbol, _) = attention.first().unwrap();
                let form = if kv == LatentKvFormat::Fp8E4M3Block128 {
                    "_prefill_partial_slices_v1"
                } else {
                    "_prefill_partial_v1"
                };
                assert!(symbol.ends_with(form), "{symbol}");
            }
        }
    }
}
