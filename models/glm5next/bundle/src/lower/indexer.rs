//! Lowering of the pooled indexer's key write, selection and draft share
//! compaction through the instance's prepared ops (`crate::ops::indexer`).

use kernel_core::backend::{Counts, View};
use kernel_core::contract::RecurrentPhase;
use kernel_core::pooled_indexer::{
    IndexerKeyWriteBuffers, IndexerSelectBuffers, IndexerTailState, PooledKeyPages,
    SelectionCompactBuffers,
};
use model_glm5next_contract::contract::{DRAFT_ITEMS, latent_format};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Lowering, layer_view};
use crate::buffers::Buffer;
use crate::ops::{Bind, Lowered, one};
use crate::program::Phase;

impl Lowering<'_> {
    /// Layer `layer`'s pooled key pages: the KV pool's pages from the
    /// layer's pooled keys in page 0, through the step's block tables.
    pub(super) fn pooled_pages(
        &self,
        bind: &mut Bind,
        layer: u32,
    ) -> Result<PooledKeyPages<View>, String> {
        Ok(PooledKeyPages {
            keys: bind.view(self.pooled_keys(layer)?),
            pages: self.kv_pages(),
            page_stride: self.format.page_bytes,
            page_tables: self.block_tables(bind),
        })
    }

    /// Normalizes the raw key, writes the incomplete pool's tail state and
    /// each completed pool's key, at the rank's place in its group. A
    /// drafter run writes its slots' tails in place, whatever its phase
    /// (the prefill's write): every row of its extend pass is a committed
    /// token's (`crate::drafter`), so the state after its last row is the
    /// frontier's, and the target's verify checkpoints are not the
    /// drafter's.
    pub(crate) fn index_key_write(&self, phase: Phase, layer: u32) -> Result<Lowered, String> {
        let wanted = match (self.draft, phase.recurrent()) {
            (Some(_), RecurrentPhase::Verify { .. }) => RecurrentPhase::Prefill,
            (_, wanted) => wanted,
        };
        let op = self
            .ops
            .indexer
            .key_write(wanted, self.cp_rank())
            .ok_or("no indexer key write at the rank's place in its group")?;
        let ordinal = self
            .shape
            .sparse_ordinal(layer)
            .ok_or_else(|| format!("layer {layer} has no indexer"))?;
        let dim = self.shape.profile.indexer_dim;
        let tail = self.buffers.state.layer("indexer.tail", ordinal)?;
        let view = |name: &str| self.buffers.view(&layer_view(layer, name));
        let mut bind = Bind::default();
        let buffers = IndexerKeyWriteBuffers {
            key: bind.view(self.scratch(Buffer::IndexKey)),
            key_stride: dim,
            gate: bind.view(self.scratch(Buffer::IndexGate)),
            gate_stride: dim,
            norm_weight: bind.view(view("self_attn.indexer.k_norm.weight")?),
            norm_bias: bind.view(view("self_attn.indexer.k_norm.bias")?),
            position: bind.view(view("self_attn.indexer.index_kpool_compress_ape")?),
            row_batch: bind.view(self.inputs.token_batch),
            positions: bind.view(self.inputs.positions),
            live_rows: bind.view(self.inputs.live_tokens),
            tail: IndexerTailState {
                state: bind.view(tail),
                entries: (tail.length / self.shape.indexer_tail_bytes()) as u32,
                slot_ids: bind.view(self.inputs.slot_ids),
                checkpoint_ids: matches!(wanted, RecurrentPhase::Verify { .. })
                    .then(|| bind.view(self.scratch(Buffer::TailCheckpointIds))),
            },
            pooled: self.pooled_pages(&mut bind, layer)?,
        };
        one(bind
            .launch(op, &buffers, Counts::rows(phase.rows()))
            .map_err(|refusal| format!("indexer key write: {refusal}"))?)
    }

    /// Scores every visible pool and writes each row's key list: the
    /// selection over splits and their merge, or the one-split form, as the
    /// op runs the step's rows.
    pub(crate) fn index_select(&self, phase: Phase, layer: u32) -> Result<Vec<Lowered>, String> {
        if self.cp_size() > 1 {
            return Err("a context-parallel group selects per rank".into());
        }
        let op = self.ops.indexer.select(0).ok_or("no indexer selection")?;
        let mut bind = Bind::default();
        let buffers = IndexerSelectBuffers {
            q: bind.view(self.scratch(Buffer::IndexQuery)),
            head_weights: Some(bind.view(self.scratch(Buffer::IndexWeights))),
            pooled: self.pooled_pages(&mut bind, layer)?,
            row_batch: bind.view(self.inputs.token_batch),
            positions: bind.view(self.inputs.positions),
            live_rows: bind.view(self.inputs.live_tokens),
            workspace: bind.view(self.scratch(Buffer::Workspace)),
            output: bind.view(self.scratch(Buffer::KeyLists)),
        };
        bind.launch(op, &buffers, Counts::rows(phase.rows()))
            .map_err(|refusal| format!("indexer selection: {refusal}"))
    }

    /// A draft step's key lists: each item's row of the extend pass's, by
    /// `DraftShareRows` (`index_share_for_mtp_iteration`), through the
    /// selection compaction of every draft item in every phase.
    pub(crate) fn share_selection(&self) -> Result<Lowered, String> {
        let mut bind = Bind::default();
        let buffers = SelectionCompactBuffers {
            lists: bind.view(self.scratch(Buffer::KeyLists)),
            source: bind.view(self.scratch(Buffer::DraftShareRows)),
            live_rows: bind.view(self.inputs.live_tokens),
            output: bind.view(self.scratch(Buffer::SharedKeyLists)),
        };
        one(bind
            .launch(
                &self.ops.indexer.compact,
                &buffers,
                Counts::rows(DRAFT_ITEMS),
            )
            .map_err(|refusal| format!("indexer share compaction: {refusal}"))?)
    }

    /// This layer's pooled keys from page 0: after its latent rows.
    pub(super) fn pooled_keys(&self, layer: u32) -> Result<DeviceBufferSpanV1, String> {
        let ordinal = self
            .shape
            .sparse_ordinal(layer)
            .ok_or_else(|| format!("layer {layer} has no indexer"))?;
        let layer_span = self.buffers.kv_layer(self.shape, self.format, ordinal);
        let latent = self.shape.latent_page_bytes(latent_format(self.format));
        Ok(DeviceBufferSpanV1::new(
            layer_span.buffer,
            layer_span.offset + latent,
            layer_span.length - latent,
        ))
    }
}
