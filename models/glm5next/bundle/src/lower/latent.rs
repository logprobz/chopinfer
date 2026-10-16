//! Lowering of the latent KV write, the absorbed head GEMMs and the latent
//! attention with its split merge through the instance's prepared ops
//! (`crate::ops::latent`).

use kernel_core::attention::BlockTables;
use kernel_core::backend::{Counts, View};
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::{
    HeadGemmBuffers, HeadWeightLayout, LatentAttentionBuffers, LatentKvCache, LatentKvWriteBuffers,
};
use model_glm5next_contract::contract::latent_format;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{BF16, Lowering, view_name};
use crate::buffers::Buffer;
use crate::ops::{Bind, Lowered, one};
use crate::program::Phase;

impl Lowering<'_> {
    /// The step's block tables: every item's row of pages its span holds.
    pub(crate) fn block_tables(&self, bind: &mut Bind) -> BlockTables<View> {
        let max_pages = self.shape.model_limits().max_pages();
        let tables = self.inputs.block_tables;
        BlockTables {
            tables: bind.view(tables),
            sequences: (tables.length / (u64::from(max_pages) * 4)) as u32,
            max_pages,
        }
    }

    /// Sparse layer `ordinal`'s latent cache: the KV pool's pages from the
    /// layer's planes in page 0.
    pub(crate) fn latent_cache(&self, bind: &mut Bind, ordinal: u64) -> LatentKvCache<View> {
        LatentKvCache {
            pool: bind.view(self.buffers.kv_layer(self.shape, self.format, ordinal)),
            pages: self.kv_pages(),
            page_stride: self.format.page_bytes,
        }
    }

    /// The pages of the rank's KV pool.
    pub(crate) fn kv_pages(&self) -> u32 {
        (self.buffers.kv_pool.length / self.format.page_bytes) as u32
    }

    pub(super) fn sparse(&self, layer: u32) -> Result<u64, String> {
        self.shape
            .sparse_ordinal(layer)
            .ok_or_else(|| format!("layer {layer} has no latent KV"))
    }

    /// The latent KV write of `layer`: each row's latent (`Latent`) at its
    /// position (`-1` writes nothing) in its item's pages.
    pub(crate) fn kv_write(&self, phase: Phase, layer: u32) -> Result<Lowered, String> {
        let ordinal = self.sparse(layer)?;
        let mut bind = Bind::default();
        let buffers = LatentKvWriteBuffers {
            latent: bind.view(self.scratch(Buffer::Latent)),
            cache: self.latent_cache(&mut bind, ordinal),
            block_tables: self.block_tables(&mut bind),
            row_batch: bind.view(self.inputs.token_batch),
            positions: bind.view(self.inputs.positions),
            live_rows: bind.view(self.inputs.live_tokens),
        };
        let op = self.ops.latent.kv_write(latent_format(self.format));
        one(bind
            .launch(op, &buffers, Counts::rows(phase.rows()))
            .map_err(|refusal| format!("latent KV write: {refusal}"))?)
    }

    /// `q_abs = W_UK^T q_nope` (`KByN`) or `out = W_UV o` (`NByK`), each
    /// head's half of `kv_b_proj` (`[heads, nope + v, latent]` on the rank)
    /// at the head stride.
    pub(crate) fn head_gemm(
        &self,
        phase: Phase,
        layer: u32,
        layout: HeadWeightLayout,
    ) -> Result<Lowered, String> {
        let p = self.shape.profile;
        let weights = self.buffers.view(&view_name(layer, "kv_b_proj.weight"))?;
        let (x, weight, y) = match layout {
            HeadWeightLayout::KByN => (Buffer::QueryUp, weights, Buffer::QueryAbsorbed),
            HeadWeightLayout::NByK => {
                let key_half = u64::from(p.qk_nope * p.kv_rank) * BF16;
                (
                    Buffer::AttentionOut,
                    DeviceBufferSpanV1::new(
                        weights.buffer,
                        weights.offset + key_half,
                        weights.length - key_half,
                    ),
                    Buffer::ValueOut,
                )
            }
        };
        let mut bind = Bind::default();
        let buffers = HeadGemmBuffers {
            x: bind.view(self.scratch(x)),
            weight: bind.view(weight),
            head_stride: u64::from((p.qk_nope + p.v_head) * p.kv_rank),
            y: bind.view(self.scratch(y)),
            live_rows: bind.view(self.inputs.live_tokens),
        };
        one(bind
            .launch(
                self.ops.latent.head_gemm(layout),
                &buffers,
                Counts::rows(phase.rows()),
            )
            .map_err(|refusal| format!("head-batched GEMM: {refusal}"))?)
    }

    /// The attention over each row's key list (`Lowering::common`'s
    /// `key_lists`) in its splits, then the split merge of a decode or
    /// verify op, into `AttentionOut`.
    pub(crate) fn attention(&self, phase: Phase, layer: u32) -> Result<Vec<Lowered>, String> {
        let ordinal = self.sparse(layer)?;
        let Some(super::Value::Span(key_lists)) = self.common("key_lists") else {
            unreachable!("the key lists are a scratch span")
        };
        let mut bind = Bind::default();
        let buffers = LatentAttentionBuffers {
            q: bind.view(self.scratch(Buffer::QueryAbsorbed)),
            cache: self.latent_cache(&mut bind, ordinal),
            block_tables: self.block_tables(&mut bind),
            row_batch: bind.view(self.inputs.token_batch),
            key_lists: bind.view(key_lists),
            budget: self.index_topk + self.shape.profile.index_pool - 1,
            output: Some(bind.view(self.scratch(Buffer::AttentionOut))),
            partial: None,
            workspace: bind.view(self.scratch(Buffer::Workspace)),
            live_rows: bind.view(self.inputs.live_tokens),
        };
        let op = self
            .ops
            .latent
            .attention(latent_format(self.format), self.attention_phase(phase));
        bind.launch(op, &buffers, Counts::rows(phase.rows()))
            .map_err(|refusal| format!("latent attention: {refusal}"))
    }

    /// The latent attention's phase for `phase`, which fixes each target
    /// row's partition:
    /// a prefill-path drafter pass of at most the ops' drafter rows runs
    /// verify's fixed splits.
    pub(crate) fn attention_phase(&self, phase: Phase) -> AttentionPhase {
        let drafter_rows = self.ops.latent.drafter_rows(latent_format(self.format));
        match (self.draft, phase) {
            (Some(_), _) if self.context_parallel.is_some() => AttentionPhase::Prefill,
            (Some(_), Phase::Prefill { rows }) if rows <= drafter_rows => AttentionPhase::Verify,
            _ => phase.attention(),
        }
    }
}
