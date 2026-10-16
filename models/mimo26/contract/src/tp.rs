//! Per-rank MiMo shapes at each supported tensor-parallel rank count.
//!
//! Every split comes from `kernel_core::topology`, so the weight plan, the
//! packager and the bundle compute the same shapes. TP4 is the checkpoint's
//! own chunking and the shapes every TP4 image already has.
use crate::geometry::V26_FLASH as GEOMETRY;

use kernel_core::contract::DType;
use kernel_core::row_copy::ColumnSlice;
use kernel_core::topology::TensorParallel;
use std::ops::Range;

pub const HIDDEN: u64 = GEOMETRY.hidden() as u64;
pub const VOCAB: u64 = GEOMETRY.vocab() as u64;
pub const Q_HEADS: u64 = GEOMETRY.attention_q_heads() as u64;
pub const GA_KV_HEADS: u64 = GEOMETRY.attention_kv_heads() as u64;
pub const SWA_KV_HEADS: u64 = GEOMETRY.swa_kv_heads() as u64;
pub const QK_DIM: u64 = GEOMETRY.head_dim() as u64;
pub const V_DIM: u64 = GEOMETRY.value_head_dim() as u64;
pub const EXPERTS: u64 = GEOMETRY.experts() as u64;
pub const EXPERT_INTERMEDIATE: u64 = GEOMETRY.expert_intermediate() as u64;
pub const DENSE_INTERMEDIATE: u64 = GEOMETRY.dense_intermediate() as u64;
/// Checkpoint `qkv_proj` chunks: one Q|K|V group of 16 Q heads each.
/// Checkpoint data: independent of execution geometry.
pub const QKV_CHUNKS: u64 = 4;
const _: () = assert!(
    Q_HEADS % QKV_CHUNKS == 0 && GA_KV_HEADS % QKV_CHUNKS == 0 && SWA_KV_HEADS % QKV_CHUNKS == 0
);

/// Rows in the checkpoint's fused QKV layout, independent of serving ranks.
pub const fn checkpoint_qkv_rows(global: bool) -> u64 {
    let kv = if global { GA_KV_HEADS } else { SWA_KV_HEADS };
    Q_HEADS * QK_DIM + kv * (QK_DIM + V_DIM)
}

/// Rows in one independently quantized checkpoint Q|K|V chunk.
pub const fn checkpoint_qkv_chunk_rows(global: bool) -> u64 {
    checkpoint_qkv_rows(global) / QKV_CHUNKS
}

/// Scale rows in the checkpoint, restarting the block grid at each chunk.
pub const fn checkpoint_qkv_scale_rows(global: bool) -> u64 {
    QKV_CHUNKS * checkpoint_qkv_chunk_rows(global).div_ceil(128)
}
/// Row granularity of the QKV scales when the chunks are regrouped.
/// Checkpoint data: independent of execution geometry.
pub const QKV_SCALE_GROUP_ROWS: u64 = 64;
pub const BLOCK_SCALE_LAYOUT: &str = "fp8_f32_block_scale_128x128_v1";
pub const ROW_SCALE_LAYOUT: &str = "fp8_f32_row_scale_64x128_v1";
/// DFlash drafter: 64 Q and 8 KV heads of 128, `fc` over five 4096 taps,
/// MLP intermediate 16384.
pub const DFLASH_Q_HEADS: u64 = GEOMETRY.dflash().num_attention_heads as u64;
pub const DFLASH_KV_HEADS: u64 = GEOMETRY.dflash().num_key_value_heads as u64;
pub const DFLASH_HEAD_DIM: u64 = GEOMETRY.dflash().head_dim as u64;
pub const DFLASH_FC_K: u64 = GEOMETRY.dflash().dflash_config.target_layer_ids.len() as u64
    * GEOMETRY.dflash().target_hidden_size as u64;
pub const DFLASH_INTERMEDIATE: u64 = GEOMETRY.dflash().intermediate_size as u64;
/// MTP `eh_proj` input: the normalized embedding, then the hidden state.
pub const MTP_EH_K: u64 = 2 * HIDDEN;

/// A supported MiMo rank count and its per-rank shapes.
#[derive(Clone, Copy)]
pub struct Tp {
    pub geometry: &'static crate::geometry::Geometry,
    ranks: u32,
}

impl std::fmt::Debug for Tp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut fields = f.debug_struct("Tp");
        if self.geometry.id != GEOMETRY.id {
            fields.field("geometry", &self.geometry.id);
        }
        fields.field("ranks", &self.ranks).finish()
    }
}

impl PartialEq for Tp {
    fn eq(&self, other: &Self) -> bool {
        self.ranks == other.ranks && self.geometry.id == other.geometry.id
    }
}
impl Eq for Tp {}
impl std::hash::Hash for Tp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ranks.hash(state);
        self.geometry.id.hash(state);
    }
}

impl Tp {
    pub const TP2: Self = Self {
        geometry: &GEOMETRY,
        ranks: crate::supported_ranks::SUPPORTED_RANKS[0],
    };
    pub const TP4: Self = Self {
        geometry: &GEOMETRY,
        ranks: crate::supported_ranks::SUPPORTED_RANKS[1],
    };
    pub const SUPPORTED: [Self; 2] = [Self::TP2, Self::TP4];

    /// MiMo is served at 2 or 4 ranks. One rank does not fit, and the MTP
    /// `eh_proj` shard needs an even rank count.
    pub fn new(ranks: u32) -> Result<Self, String> {
        Self::SUPPORTED
            .into_iter()
            .find(|tp| tp.ranks == ranks)
            .ok_or_else(|| format!("MiMo supports 2 or 4 ranks, not {ranks}"))
    }

    /// Select an explicitly listed geometry at a supported rank count.
    pub fn of_geometry(
        geometry: &'static crate::geometry::Geometry,
        ranks: u32,
    ) -> Result<Self, String> {
        let mut shape = Self::new(ranks)?;
        if !crate::geometry::GEOMETRIES
            .iter()
            .any(|g| g.id == geometry.id)
        {
            return Err(format!(
                "unsupported geometry: {}; supported ids: {}",
                geometry.id,
                crate::geometry::GEOMETRIES
                    .iter()
                    .map(|g| g.id)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        shape.geometry = geometry;
        Ok(shape)
    }

    pub fn model_limits(self) -> bundle_runtime::ModelLimits {
        bundle_runtime::ModelLimits::new(
            self.geometry.vocab(),
            self.geometry.context_tokens() as u64,
            crate::serving::PAGE_TOKENS,
        )
        .expect("validated geometry model limits")
    }

    pub const fn ranks(self) -> u32 {
        self.ranks
    }

    pub const fn count(self) -> usize {
        self.ranks as usize
    }

    fn parallel(self) -> TensorParallel {
        TensorParallel::new(self.count()).expect("supported rank count")
    }

    /// One rank's share of an evenly split dimension.
    pub fn share(self, total: u64) -> u64 {
        self.parallel()
            .split(total as usize)
            .expect("MiMo dimensions divide by every supported rank count")
            .padded_len() as u64
    }

    /// The source range of `total` that `rank` holds.
    pub fn range(self, total: u64, rank: usize) -> Range<u64> {
        let shard = self
            .parallel()
            .split(total as usize)
            .and_then(|split| split.rank(rank))
            .expect("rank within the topology");
        shard.live.start as u64..shard.live.end as u64
    }

    /// KV heads `rank` holds (with copies when there are fewer heads than
    /// ranks).
    pub fn kv_head_range(self, total: u64, rank: usize) -> Range<u64> {
        let shard = self
            .parallel()
            .kv_heads(total as usize)
            .and_then(|split| split.rank(rank))
            .expect("rank within the topology");
        shard.live.start as u64..shard.live.end as u64
    }

    pub fn q_heads(self) -> u64 {
        self.share(self.geometry.attention_q_heads() as u64)
    }

    pub fn q_heads_for(self, global: bool) -> u64 {
        self.share(if global {
            self.geometry.attention_q_heads()
        } else {
            self.geometry.swa_q_heads()
        } as u64)
    }

    pub fn qk_dim(self, global: bool) -> u64 {
        (if global {
            self.geometry.head_dim()
        } else {
            self.geometry.swa_head_dim()
        }) as u64
    }

    pub fn value_dim(self, global: bool) -> u64 {
        (if global {
            self.geometry.value_head_dim()
        } else {
            self.geometry.swa_value_head_dim()
        }) as u64
    }

    pub fn checkpoint_rows(self, global: bool) -> u64 {
        self.q_heads_for(global) * self.ranks() as u64 * self.qk_dim(global)
            + if global {
                self.geometry.attention_kv_heads()
            } else {
                self.geometry.swa_kv_heads()
            } as u64
                * (self.qk_dim(global) + self.value_dim(global))
    }

    pub fn kv_heads(self, global: bool) -> u64 {
        let total = if global {
            self.geometry.attention_kv_heads() as u64
        } else {
            self.geometry.swa_kv_heads() as u64
        };
        self.kv_head_range(total, 0).end - self.kv_head_range(total, 0).start
    }

    pub fn qkv_rows(self, global: bool) -> u64 {
        self.q_heads_for(global) * self.qk_dim(global)
            + self.kv_heads(global) * (self.qk_dim(global) + self.value_dim(global))
    }

    /// TP4 keeps the checkpoint's per-chunk 128-row blocks; the regrouped
    /// layouts carry one scale row per 64 output rows.
    pub fn qkv_scale_rows(self, global: bool) -> u64 {
        if self.ranks() == 4 {
            self.qkv_rows(global).div_ceil(128)
        } else {
            self.qkv_rows(global) / QKV_SCALE_GROUP_ROWS
        }
    }

    pub fn qkv_scale_layout(self) -> &'static str {
        if self.ranks() == 4 {
            BLOCK_SCALE_LAYOUT
        } else {
            ROW_SCALE_LAYOUT
        }
    }

    pub fn o_proj_k(self) -> u64 {
        self.o_proj_k_for(true)
    }

    pub fn o_proj_k_for(self, global: bool) -> u64 {
        self.q_heads_for(global) * self.value_dim(global)
    }

    pub fn vocab(self) -> u64 {
        self.share(self.geometry.vocab() as u64)
    }

    pub fn expert_intermediate(self) -> u64 {
        self.share(self.geometry.expert_intermediate() as u64)
    }

    pub fn dense_intermediate(self) -> u64 {
        self.share(self.geometry.dense_intermediate() as u64)
    }

    pub fn mtp_eh_k(self) -> u64 {
        self.share(2 * self.geometry.hidden() as u64)
    }

    /// Source of `rank`'s `eh_proj` K shard: `(hidden, column offset)` in the
    /// 4096-wide normalized embedding (`hidden == false`) or hidden state.
    pub fn mtp_eh_source(self, rank: usize) -> (bool, u64) {
        let columns = self.range(2 * self.geometry.hidden() as u64, rank);
        (
            columns.start >= (self.geometry.hidden() as u64),
            columns.start % (self.geometry.hidden() as u64),
        )
    }

    /// `rank`'s `eh_proj` K shard of its normalized source rows: the BF16
    /// columns of `mtp_eh_source` from each hidden-wide row, whose views
    /// start at 16-byte alignment.
    pub fn mtp_eh_slice(self, rank: usize) -> ColumnSlice {
        ColumnSlice {
            dtype: DType::Bf16,
            width: self.mtp_eh_k() as u32,
            stride: self.geometry.hidden(),
            offset: self.mtp_eh_source(rank).1 as u32,
            alignment_bytes: 16,
        }
    }

    pub fn dflash_q_heads(self) -> u64 {
        self.share(self.geometry.dflash().num_attention_heads as u64)
    }

    pub fn dflash_kv_heads(self) -> u64 {
        self.share(self.geometry.dflash().num_key_value_heads as u64)
    }

    pub fn dflash_fc_k(self) -> u64 {
        self.share(self.geometry.tap_count() as u64 * self.geometry.hidden() as u64)
    }

    pub fn dflash_intermediate(self) -> u64 {
        self.share(self.geometry.dflash().intermediate_size as u64)
    }

    pub fn dflash_qkv_rows(self) -> u64 {
        (self.dflash_q_heads() + 2 * self.dflash_kv_heads())
            * (self.geometry.dflash().head_dim as u64)
    }

    /// Source rows of `rank`'s regrouped `qkv_proj`: its Q heads, then its K
    /// heads, then its V heads, each a run of whole heads in the checkpoint's
    /// chunked layout. Adjacent runs are merged, so TP4 is one run per rank.
    pub fn qkv_segments(self, global: bool, rank: usize) -> Vec<Range<u64>> {
        let kv_total = if global {
            self.geometry.attention_kv_heads() as u64
        } else {
            self.geometry.swa_kv_heads() as u64
        };
        let q_heads = self.q_heads_for(global) * self.ranks() as u64;
        let qk = self.qk_dim(global);
        let value = self.value_dim(global);
        let chunk_q = q_heads / QKV_CHUNKS;
        let chunk_kv = kv_total / QKV_CHUNKS;
        let chunk_rows = self.checkpoint_rows(global) / QKV_CHUNKS;
        let mut runs: Vec<Range<u64>> = Vec::new();
        let mut push = |run: Range<u64>| match runs.last_mut() {
            Some(last) if last.end == run.start => last.end = run.end,
            _ => runs.push(run),
        };
        for head in self.range(q_heads, rank) {
            let start = head / chunk_q * chunk_rows + head % chunk_q * qk;
            push(start..start + qk);
        }
        let kv = self.kv_head_range(kv_total, rank);
        for head in kv.clone() {
            let start = head / chunk_kv * chunk_rows + chunk_q * qk + head % chunk_kv * qk;
            push(start..start + qk);
        }
        for head in kv {
            let start = head / chunk_kv * chunk_rows
                + chunk_q * qk
                + chunk_kv * qk
                + head % chunk_kv * value;
            push(start..start + value);
        }
        runs
    }

    /// Rows of one checkpoint chunk, for the chunked FP8 scale grid.
    pub fn qkv_chunk_rows(global: bool) -> u64 {
        checkpoint_qkv_chunk_rows(global)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listed_topologies_are_canonical_and_expose_model_limits() {
        for geometry in crate::geometry::GEOMETRIES {
            for tp in Tp::SUPPORTED {
                let shape = Tp::of_geometry(geometry, tp.ranks()).unwrap();
                let id = crate::topology::topology_id(shape);
                assert_eq!(crate::topology::from_topology_id(id), Some(shape));
                assert_eq!(
                    shape.model_limits().max_pages(),
                    geometry
                        .context_tokens()
                        .div_ceil(crate::serving::PAGE_TOKENS as u32)
                );
                assert_eq!(
                    shape.vocab() * shape.ranks() as u64,
                    geometry.vocab() as u64
                );
            }
        }
        for id in [
            "mimo-v26-flash-tp02",
            "mimo-v26-flash-tp+2",
            "mimo-absent-tp2",
            "mimo-v26-flash-tp8",
        ] {
            assert_eq!(crate::topology::from_topology_id(id), None);
        }
    }

    #[test]
    fn tp8_is_not_supported() {
        assert_eq!(Tp::new(8), Err("MiMo supports 2 or 4 ranks, not 8".into()));
        assert_eq!(crate::sm120::config::profile(8), None);
        assert_eq!(
            crate::topology::from_topology_id("mimo-v26-flash-tp8"),
            None
        );
    }

    #[test]
    fn tp4_shapes_are_the_checkpoint_chunks() {
        let tp = Tp::TP4;
        assert_eq!(
            (tp.q_heads(), tp.kv_heads(true), tp.kv_heads(false)),
            (16, 1, 2)
        );
        assert_eq!((tp.qkv_rows(true), tp.qkv_rows(false)), (3392, 3712));
        assert_eq!(
            (tp.qkv_scale_rows(true), tp.qkv_scale_rows(false)),
            (27, 29)
        );
        assert_eq!(tp.qkv_scale_layout(), BLOCK_SCALE_LAYOUT);
        assert_eq!((tp.o_proj_k(), tp.vocab()), (2048, 38144));
        assert_eq!(
            (tp.expert_intermediate(), tp.dense_intermediate()),
            (512, 4096)
        );
        assert_eq!((tp.mtp_eh_k(), tp.dflash_fc_k()), (2048, 5120));
        assert_eq!((tp.dflash_q_heads(), tp.dflash_kv_heads()), (16, 2));
        assert_eq!(tp.dflash_qkv_rows(), 2560);
        for rank in 0..4 {
            for (global, rows) in [(true, 3392), (false, 3712)] {
                let start = rank as u64 * rows;
                let segments = tp.qkv_segments(global, rank);
                assert_eq!((segments.len(), &segments[0]), (1, &(start..start + rows)));
            }
        }
        let sources: Vec<_> = (0..4).map(|rank| tp.mtp_eh_source(rank)).collect();
        assert_eq!(
            sources,
            [(false, 0), (false, 2048), (true, 0), (true, 2048)]
        );
        let offsets: Vec<_> = (0..4).map(|rank| tp.mtp_eh_slice(rank).offset).collect();
        assert_eq!(offsets, [0, 2048, 0, 2048]);
        assert_eq!(
            (tp.mtp_eh_slice(3).width, tp.mtp_eh_slice(3).stride),
            (2048, 4096)
        );
    }

    #[test]
    fn tp2_shapes() {
        let tp = Tp::TP2;
        assert_eq!(
            (tp.q_heads(), tp.kv_heads(true), tp.kv_heads(false)),
            (32, 2, 4)
        );
        assert_eq!((tp.qkv_rows(true), tp.qkv_rows(false)), (6784, 7424));
        assert_eq!(
            (tp.qkv_scale_rows(true), tp.qkv_scale_rows(false)),
            (106, 116)
        );
        assert_eq!(
            (tp.o_proj_k(), tp.vocab(), tp.expert_intermediate()),
            (4096, 76288, 1024)
        );
        assert_eq!(
            (tp.mtp_eh_k(), tp.dflash_fc_k(), tp.dflash_qkv_rows()),
            (4096, 10240, 5120)
        );
        assert_eq!(tp.qkv_scale_layout(), ROW_SCALE_LAYOUT);
        assert_eq!(
            [tp.mtp_eh_source(0), tp.mtp_eh_source(1)],
            [(false, 0), (true, 0)]
        );
        for rank in 0..2 {
            let slice = tp.mtp_eh_slice(rank);
            assert_eq!((slice.width, slice.stride, slice.offset), (4096, 4096, 0));
        }
        assert!(Tp::new(1).is_err() && Tp::new(16).is_err());
    }

    #[test]
    fn regrouped_qkv_rows_are_whole_heads_in_q_k_v_order() {
        // TP2 GA rank 1: chunks 2 and 3.
        assert_eq!(
            Tp::TP2.qkv_segments(true, 1),
            [
                6784..6784 + 3072,
                10176..10176 + 3072,
                6784 + 3072..6784 + 3264,
                10176 + 3072..10176 + 3264,
                6784 + 3264..6784 + 3392,
                10176 + 3264..10176 + 3392,
            ]
        );
        for tp in Tp::SUPPORTED {
            for global in [true, false] {
                let chunk = Tp::qkv_chunk_rows(global);
                let mut q = 0;
                let mut all = Vec::new();
                for rank in 0..tp.count() {
                    let segments = tp.qkv_segments(global, rank);
                    let rows: u64 = segments.iter().map(|run| run.end - run.start).sum();
                    assert_eq!(rows, tp.qkv_rows(global));
                    for run in &segments {
                        assert_eq!(run.start % QKV_SCALE_GROUP_ROWS, 0);
                        assert_eq!(run.end % QKV_SCALE_GROUP_ROWS, 0);
                        all.extend(run.clone());
                        q += run.clone().filter(|row| row % chunk < 3072).count();
                    }
                }
                // Every Q row once; K and V rows once, or once per copy.
                assert_eq!(q, 64 * 192);
                all.sort_unstable();
                all.dedup();
                assert_eq!(all.len() as u64, 4 * chunk);
            }
        }
    }
}
