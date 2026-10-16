//! The attention family's ops and buffer roles: dense attention over a
//! paged KV cache or a per-item window ring, in prefill, decode and verify
//! steps, optionally folding RoPE and the current rows' KV write in. The
//! math is [`AttentionQuery`] with its oracles (`oracle::attention`,
//! `oracle::ring_attention`). Gated, sparse and latent attention are other
//! families.
//!
//! Every field of an [`AttentionOp`] is math or a layout the model shares
//! with its other ops (the cache's page size, KV dtype and context, which
//! fixes the block table's row): splits, tiles, CTA counts, merge widths
//! and which entries serve a verify step are the backend's choices. A model
//! that relies on verify rows equal to decode rows asks for one-row bits in
//! [`Need::exact`](crate::backend::Need).

use crate::contract::{AttentionPhase, AttentionQuery, Refusal, RotarySource, Validate};

/// One attention op a model prepares.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionOp {
    pub query: AttentionQuery,
    /// The most query rows of one item: one in decode, a verify item's
    /// tokens, and in prefill the step's packed tokens (one item may hold
    /// them all).
    pub item_rows: u32,
    /// RoPE and the KV write the op folds in before it attends.
    pub fold: Option<AttentionFold>,
    /// Whether every step plans as for the op's item capacity
    /// (`query.max_batch_capacity`), whatever items it serves, so a
    /// captured launch does not depend on the step's items. Otherwise a
    /// decode or verify step plans for the smallest bucket that holds its
    /// items. The per-item arrays hold the step's items either way.
    pub plan_at_capacity: bool,
}

/// RoPE and the current rows' KV write folded into an attention. The op
/// reads packed rows from `q` in place of query rows, rotates Q and K, and
/// writes each row's K and V to the cache (quantized by the cache's scales
/// for FP8) before it attends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AttentionFold {
    /// NeoX rotate-half RoPE over the first `rotary` dimensions of each Q
    /// and K head at base `theta`, with computed angles or an F32
    /// `[position][rotary / 2][cos, sin]` table (`rope.cos_sin`); V is
    /// written times `v_multiplier`. Rows are
    /// `[row][q_heads * qk | kv_heads * qk | kv_heads * v]`.
    Neox {
        rotary: u32,
        theta: f32,
        v_multiplier: f32,
        source: RotarySource,
    },
    /// The gated norm and M-RoPE. Rows are
    /// `[row][q_heads * 2 qk | kv_heads * qk | kv_heads * v]`, each head's
    /// query followed by its gate. Q and K get the per-head Gemma RMSNorm
    /// (`rope.q_norm`, `rope.k_norm`, `eps`), then NeoX RoPE over the first
    /// `rotary` dimensions with interleaved M-RoPE `sections` (T, H, W) at
    /// `rope.rope_positions`, from the `rope.cos_sin` table.
    Gated {
        rotary: u32,
        sections: [u32; 3],
        eps: f32,
    },
}

impl Validate for AttentionOp {
    fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        if self.item_rows == 0 || self.query.phase == AttentionPhase::Decode && self.item_rows != 1
        {
            return Err(Refusal(
                "attention items hold at least one query row, one in decode",
            ));
        }
        let rotary = match self.fold {
            None => return Ok(()),
            Some(AttentionFold::Neox {
                rotary,
                theta,
                v_multiplier,
                ..
            }) => {
                if !theta.is_finite() || theta <= 1.0 || !v_multiplier.is_finite() {
                    return Err(Refusal(
                        "folded RoPE needs a finite theta above 1 and V multiplier",
                    ));
                }
                rotary
            }
            Some(AttentionFold::Gated {
                rotary,
                sections,
                eps,
            }) => {
                if sections.iter().sum::<u32>() != rotary / 2 || !eps.is_finite() || eps <= 0.0 {
                    return Err(Refusal(
                        "gated fold needs M-RoPE sections of half its rotary width and a positive epsilon",
                    ));
                }
                rotary
            }
        };
        if rotary == 0 || !rotary.is_multiple_of(2) || rotary > self.query.qk_dim {
            return Err(Refusal(
                "folded RoPE rotates an even width within the QK head",
            ));
        }
        Ok(())
    }
}

crate::scratch_record! {
    /// An attention op's buffers by role. `q` holds the BF16 query rows
    /// `[row][q_heads * qk]` (a fold's packed rows instead) and `output` the
    /// BF16 rows `[row][q_heads * v]` the op writes. Per-item inputs are I32
    /// arrays: `block_tables` (paged: each item's row of
    /// `max_context_capacity` over `page_size` pages, rounded up), `slot_ids` (ring: each
    /// item's ring), `seq_lens` (decode: keys including the current token),
    /// `q_start`, `q_len` and `prefix_len` (prefill and verify: each item's
    /// first packed row, its rows and the cached keys before them), and
    /// `live_batch`, the device count of live items. `k_new` and `v_new`
    /// hold a ring step's own BF16 K and V rows outside decode, which the
    /// ring receives after attention. `sink` holds each Q head's BF16 sink
    /// logit. A fold reads its inputs from `rope`. `workspace` holds the
    /// partials the op sizes in its info.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct AttentionBuffers {
        pub q: value,
        pub cache: record(KvCache),
        pub block_tables: option,
        pub slot_ids: option,
        pub seq_lens: option,
        pub q_start: option,
        pub q_len: option,
        pub prefix_len: option,
        pub live_batch: value,
        pub k_new: option,
        pub v_new: option,
        pub sink: option,
        pub rope: optional(FoldInputs),
        pub output: value,
        pub workspace: value,
    }
}

crate::scratch_record! {
    /// The I32 page tables `[sequences, max_pages]` of the sequences a
    /// launch reads or writes: entry `[s][i]` is the page holding sequence
    /// s's tokens `[i * page_size, (i + 1) * page_size)`. Ops over a latent
    /// cache or pooled indexer keys share it.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct BlockTables {
        pub tables: value,
        pub sequences: scalar(u32),
        pub max_pages: scalar(u32),
    }
}

impl BlockTables<crate::backend::View> {
    /// Refuses tables without a sequence or a page, or a view that does not
    /// hold every row.
    pub fn check(&self) -> Result<(), Refusal> {
        if self.sequences == 0 || self.max_pages == 0 {
            return Err(Refusal("block tables hold at least one sequence and page"));
        }
        let bytes = u64::from(self.sequences)
            .checked_mul(u64::from(self.max_pages))
            .and_then(|entries| entries.checked_mul(4))
            .ok_or(Refusal("block table bytes overflow"))?;
        self.tables.covers(bytes)
    }
}

crate::scratch_record! {
    /// A KV cache's K and V planes: `slots` pages (paged) or rings (ring) of
    /// `page_size` or `ring_size` tokens, laid out `[slot][token][kv_head][dim]`
    /// in the cache's KV dtype, each slot `*_slot_stride` bytes after the
    /// last. An FP8 cache holds values divided by `k_scale` and `v_scale`; a
    /// BF16 cache ignores them. The RoPE and KV write family writes the same
    /// cache (`rope_kv`).
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct KvCache {
        pub k: value,
        pub v: value,
        pub slots: scalar(u32),
        pub k_slot_stride: scalar(u64),
        pub v_slot_stride: scalar(u64),
        pub k_scale: scalar(f32),
        pub v_scale: scalar(f32),
    }
}

crate::scratch_record! {
    /// A fold's inputs: a table-backed or gated fold's F32 `cos_sin` table
    /// of `Counts::positions` positions, and the gated fold's BF16 `[qk]` norm weights and I32
    /// `[row][3]` M-RoPE positions.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct FoldInputs {
        pub cos_sin: option,
        pub q_norm: option,
        pub k_norm: option,
        pub rope_positions: option,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::allocation::Traversal;
    use crate::contract::{AttentionLayout, DType};

    pub(crate) fn decode() -> AttentionOp {
        AttentionOp {
            query: AttentionQuery {
                q_heads: 16,
                kv_heads: 2,
                qk_dim: 128,
                v_dim: 128,
                page_size: 32,
                kv_dtype: DType::Bf16,
                layout: AttentionLayout::Paged,
                causal: true,
                window_left: None,
                sink: false,
                softcap: None,
                softmax_scale: 0.1,
                phase: AttentionPhase::Decode,
                max_context_capacity: 4096,
                max_batch_capacity: 32,
            },
            item_rows: 1,
            fold: None,
            plan_at_capacity: false,
        }
    }

    #[test]
    fn ops_check_their_math_only() {
        assert_eq!(decode().validate(), Ok(()));
        // Any head width and an FP8 ring are math; a backend refuses what
        // it lacks.
        let odd = AttentionQuery {
            qk_dim: 96,
            v_dim: 80,
            ..decode().query
        };
        assert_eq!(odd.validate(), Ok(()));
        assert!(AttentionQuery { v_dim: 0, ..odd }.validate().is_err());
        let ring = AttentionOp {
            query: AttentionQuery {
                kv_dtype: DType::Fp8E4M3,
                layout: AttentionLayout::Ring,
                window_left: Some(31),
                ..decode().query
            },
            ..decode()
        };
        assert_eq!(ring.validate(), Ok(()));
        let window_past_ring = AttentionQuery {
            window_left: Some(32),
            ..ring.query
        };
        assert!(
            AttentionOp {
                query: window_past_ring,
                ..ring
            }
            .validate()
            .is_err()
        );
        assert!(
            AttentionOp {
                item_rows: 2,
                ..decode()
            }
            .validate()
            .is_err()
        );
        let fold = |rotary, sections| AttentionOp {
            fold: Some(AttentionFold::Gated {
                rotary,
                sections,
                eps: 1e-6,
            }),
            ..decode()
        };
        assert_eq!(fold(64, [11, 11, 10]).validate(), Ok(()));
        assert!(fold(64, [11, 11, 11]).validate().is_err());
        assert!(fold(256, [43, 43, 42]).validate().is_err());
        let neox = |theta| AttentionOp {
            fold: Some(AttentionFold::Neox {
                rotary: 64,
                theta,
                v_multiplier: 1.0,
                source: RotarySource::Computed,
            }),
            ..decode()
        };
        assert_eq!(neox(10_000.0).validate(), Ok(()));
        assert!(neox(f32::NAN).validate().is_err());
    }

    #[test]
    fn roles_are_named_by_path() {
        let buffers = AttentionBuffers {
            q: 1,
            cache: KvCache {
                k: 2,
                v: 3,
                slots: 4,
                k_slot_stride: 64,
                v_slot_stride: 64,
                k_scale: 1.0,
                v_scale: 1.0,
            },
            block_tables: Some(5),
            slot_ids: None,
            seq_lens: Some(6),
            q_start: None,
            q_len: None,
            prefix_len: None,
            live_batch: 7,
            k_new: None,
            v_new: None,
            sink: None,
            rope: Some(FoldInputs {
                cos_sin: Some(8),
                q_norm: None,
                k_norm: None,
                rope_positions: None,
            }),
            output: 9,
            workspace: 10,
        };
        let mut named = Vec::new();
        buffers.visit_named(&mut |role, &view| named.push((role.to_owned(), view)));
        let roles: Vec<_> = named.iter().map(|(role, _)| role.as_str()).collect();
        assert_eq!(
            roles,
            [
                "q",
                "cache.k",
                "cache.v",
                "block_tables",
                "seq_lens",
                "live_batch",
                "rope.cos_sin",
                "output",
                "workspace"
            ]
        );
    }
}
