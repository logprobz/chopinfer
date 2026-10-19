//! Lowering of the KV sources' compressor stages through the instance's
//! prepared ops (`crate::ops::compressor`), over the serving pages' cache
//! planes and the items' F32 tails (`crate::compressor`). A ratio-2 source
//! pools completed pairs; a ratio-1 source normalizes every F32 row, and
//! each row is its own compressed position.

use bundle_runtime::inputs::PhaseInputs;
use bundle_runtime::state::Extent;
use kernel_core::contract::{RecurrentPhase, Refusal};
use model_deepseek41_contract::geometry::{Geometry, LayerRole};
use model_deepseek41_contract::serving::MAX_VERIFY_ROWS;
use model_deepseek41_contract::state;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Lowering, layer_view, rows};
use crate::buffers::StateSpans;
use crate::compressor::{SourceBinding, source_binding, tail_binding};
use crate::ops::{Lowered, NormViews, PlaneViews, PoolingViews, WriteViews};
use crate::program::Phase;

/// The request state buffer that holds the ratio-2 sources' tails.
const TAIL_BUFFER: &str = "compressor.tail";

impl Lowering<'_> {
    /// The cache planes of layer `layer`'s KV source in the serving pages.
    pub(super) fn source(&self, layer: u32) -> Result<SourceBinding, String> {
        source_binding(
            self.shape.geometry,
            layer,
            self.kv.storage,
            self.buffers.kv_pool,
            self.kv.pages,
            self.shape.geometry.context_tokens(),
        )
        .map_err(|refusal| format!("layer {layer}'s compressor cache: {refusal}"))?
        .ok_or_else(|| format!("layer {layer} has no compressed source"))
    }

    fn plane(&self, plane: crate::compressor::CachePlane) -> PlaneViews {
        PlaneViews {
            values: plane.values,
            scales: plane.scales,
            pages: plane.pages,
            page_stride: plane.page_stride,
            tables: self.inputs.block_tables,
            sequences: self.kv.sequences,
            max_pages: self.kv.max_pages,
        }
    }

    /// Layer `layer`'s compression: a ratio-2 source's gated pooling of its
    /// F32 key and gate rows with its item tails, or a ratio-1 source's
    /// RMSNorm of its BF16 rows.
    pub(super) fn compress(
        &self,
        phase: Phase,
        layer: u32,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let g = self.shape.geometry;
        if g.layer_role(layer) != LayerRole::Full {
            return Err(format!("layer {layer} compresses no KV"));
        }
        let scratch = &self.buffers.scratch;
        let norm_weight = || {
            self.buffers
                .view(&layer_view(layer, "attn.compressor.norm.weight"))
        };
        if g.compress_ratio(layer) == 1 {
            let Some(ops) = &self.ops.compressor else {
                return Ok(None);
            };
            let views = NormViews {
                x: rows(scratch.compressor_rows, phase.rows()),
                weight: norm_weight()?,
                y: rows(scratch.compressed, phase.rows()),
                live_rows: self.inputs.live_tokens,
                status: Some(self.carried(phase, scratch.row_status)?),
            };
            return ops
                .normalize(phase.rows(), &views)
                .map(Some)
                .map_err(|refusal| format!("ratio-1 compression of layer {layer}: {refusal}"));
        }
        let Some(ops) = &self.ops.compressor else {
            return Ok(None);
        };
        let tail = self
            .buffers
            .state
            .named(TAIL_BUFFER)
            .ok_or("the request state has no compressor tails")?;
        let tail = tail_binding(g, layer, self.kv.extent, tail)
            .map_err(|refusal| format!("layer {layer}'s tails: {refusal}"))?
            .ok_or_else(|| format!("layer {layer} keeps no tail"))?;
        let views = PoolingViews {
            key: scratch.compressor_key,
            gate: scratch.compressor_gate,
            norm_weight: norm_weight()?,
            positions: self.inputs.positions,
            row_batch: self.inputs.token_batch,
            live_rows: self.inputs.live_tokens,
            tail: tail.state,
            tail_entries: tail.entries,
            slot_ids: scratch.tail_ids,
            checkpoint_ids: matches!(phase, Phase::Verify { .. })
                .then_some(scratch.tail_checkpoints),
            output: scratch.compressed,
            output_positions: scratch.compressed_positions,
            workspace: self.buffers.workspace,
        };
        // Every item of the phase's layout, live or not, has a tail entry.
        let items = (self.inputs.slot_ids.length / 4) as u32;
        ops.pool(phase, items, &views)
            .map(Some)
            .map_err(|refusal| format!("pooling of layer {layer}: {refusal}"))
    }

    /// Each row's compressed position: the pooling's for a ratio-2 source,
    /// the token's own for a ratio-1 source.
    fn compressed_positions(&self, layer: u32) -> DeviceBufferSpanV1 {
        match self.shape.geometry.compress_ratio(layer) {
            1 => self.inputs.positions,
            _ => self.buffers.scratch.compressed_positions,
        }
    }

    /// Layer `layer`'s compressed latents (`index` false) or index keys
    /// written into its pages, or `None` for index keys in BF16 pages.
    pub(super) fn compressed_write(
        &self,
        phase: Phase,
        layer: u32,
        index: bool,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.compressor else {
            return Ok(None);
        };
        let source = self.source(layer)?;
        // BF16 pages' index keys wait for a BF16 writer contract.
        if index && !ops.writes_index_keys(source.ratio) {
            return Ok(None);
        }
        let scratch = &self.buffers.scratch;
        let views = WriteViews {
            rows: if index {
                scratch.index_key
            } else {
                scratch.compressed
            },
            norm_weight: match index {
                true => Some(
                    self.buffers
                        .view(&layer_view(layer, "attn.indexer.k_norm.weight"))?,
                ),
                false => None,
            },
            cos_sin: self.buffers.rope_compressed,
            positions: self.compressed_positions(layer),
            row_batch: self.inputs.token_batch,
            live_rows: self.inputs.live_tokens,
            plane: self.plane(if index { source.index } else { source.latent }),
            workspace: self.buffers.workspace,
            status: rows(scratch.row_status, phase.rows()),
        };
        let launches = match index {
            true => ops.write_index_keys(source.ratio, phase.rows(), &views),
            false => ops.write_latent(source.ratio, phase.rows(), &views),
        };
        launches
            .map(Some)
            .map_err(|refusal| format!("compressed write of layer {layer}: {refusal}"))
    }
}

/// One rank's tail entries of a step's items, which the step uploads
/// before its launches: each item's source entry, a verify item's
/// checkpoint entries (`MAX_VERIFY_ROWS` per item), and the exact entry
/// copies an off-grid prefill needs before its pooling, as (destination,
/// source).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TailUpload {
    pub ids: Vec<i32>,
    pub checkpoints: Vec<i32>,
    pub seeds: Vec<(DeviceBufferSpanV1, DeviceBufferSpanV1)>,
}

/// The tail entries of `inputs`' items in `phase` over a rank's request
/// state `state`, or `None` for a geometry without a ratio-2 source.
pub fn tail_upload(
    g: &Geometry,
    extent: Extent,
    inputs: &PhaseInputs,
    phase: Phase,
    state: &StateSpans,
) -> Result<Option<TailUpload>, String> {
    let layers: Vec<u32> = state::sources_of(g, 2).collect();
    if layers.is_empty() {
        return Ok(None);
    }
    let buffer = state
        .named(TAIL_BUFFER)
        .ok_or("the request state has no compressor tails")?;
    let recurrent = match phase {
        Phase::Prefill { .. } => RecurrentPhase::Prefill,
        Phase::Decode(_) => RecurrentPhase::Decode,
        Phase::Verify { .. } => RecurrentPhase::Verify {
            max_rows: MAX_VERIFY_ROWS,
        },
    };
    let error = |refusal: Refusal| format!("compressor tails: {refusal}");
    let mut upload = TailUpload::default();
    for (index, &layer) in layers.iter().enumerate() {
        let tail = tail_binding(g, layer, extent, buffer)
            .map_err(error)?
            .ok_or("a ratio-2 source keeps no tail")?;
        for item in 0..inputs.slot_ids.len() {
            let slot = u32::try_from(inputs.slot_ids[item])
                .map_err(|_| "an item has no state slot".to_owned())?;
            let first = u64::try_from(inputs.prefix_len[item])
                .map_err(|_| "an item's prefix is negative".to_owned())?;
            let rows = inputs.q_len.get(item).map_or(1, |&rows| rows as u32);
            let ids = tail.ids(slot, first, rows, recurrent).map_err(error)?;
            upload
                .seeds
                .extend(ids.seed_copy.map(|(from, to)| (to, from)));
            if index == 0 {
                upload.ids.push(ids.source);
                if let RecurrentPhase::Verify { max_rows } = recurrent {
                    let mut checkpoints = ids.checkpoints;
                    checkpoints.resize(max_rows as usize, -1);
                    upload.checkpoints.extend(checkpoints);
                }
            }
        }
    }
    Ok(Some(upload))
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_deepseek41_contract::geometry::FLASH;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    /// Flash's request state for two items, laid out from offset 0.
    fn state(extent: Extent) -> StateSpans {
        let state = state::request_state(&FLASH, state::KvStorage::Native, false);
        let bytes = state.buffers.iter().map(|b| b.bytes(extent)).sum();
        let view = DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, 1), 0, bytes);
        StateSpans::new(&state, extent, view).unwrap()
    }

    /// Every item names its slot's entries, the same in each source; a
    /// prefill that starts and ends inside a pair seeds its target entry in
    /// every ratio-2 source; a verify item's checkpoints fill its row of
    /// `MAX_VERIFY_ROWS`.
    #[test]
    fn tail_entries_follow_the_items_and_seed_every_source() {
        let extent = Extent {
            served: 2,
            bucket: 2,
            verify_rows: 0,
        };
        let spans = state(extent);
        let tails = spans.named(TAIL_BUFFER).unwrap();
        let prefill = PhaseInputs {
            slot_ids: vec![1, 0],
            prefix_len: vec![3, 0],
            q_len: vec![2, 64],
            ..PhaseInputs::default()
        };
        let phase = Phase::Prefill { rows: 66 };
        let upload = tail_upload(&FLASH, extent, &prefill, phase, &spans)
            .unwrap()
            .unwrap();
        let sources: Vec<u32> = state::sources_of(&FLASH, 2).collect();
        assert!(!sources.is_empty());
        let first = tail_binding(&FLASH, sources[0], extent, tails)
            .unwrap()
            .unwrap();
        let ids = first.ids(1, 3, 2, RecurrentPhase::Prefill).unwrap();
        let other = first.ids(0, 0, 64, RecurrentPhase::Prefill).unwrap();
        assert_eq!(upload.ids, [ids.source, other.source]);
        assert!(upload.checkpoints.is_empty());
        let mut seeds = Vec::new();
        for &layer in &sources {
            let tail = tail_binding(&FLASH, layer, extent, tails).unwrap().unwrap();
            let (from, to) = tail
                .ids(1, 3, 2, RecurrentPhase::Prefill)
                .unwrap()
                .seed_copy
                .unwrap();
            seeds.push((to, from));
        }
        assert_eq!(upload.seeds, seeds);
        let verify = PhaseInputs {
            slot_ids: vec![0],
            prefix_len: vec![10],
            q_len: vec![2],
            ..PhaseInputs::default()
        };
        let phase = Phase::Verify { rows: 2, items: 1 };
        let upload = tail_upload(&FLASH, extent, &verify, phase, &spans)
            .unwrap()
            .unwrap();
        let rows = MAX_VERIFY_ROWS as usize;
        assert_eq!(upload.checkpoints.len(), rows);
        assert!(upload.checkpoints[..rows].iter().all(|&id| id >= 0));
        assert!(upload.seeds.is_empty());
    }
}
