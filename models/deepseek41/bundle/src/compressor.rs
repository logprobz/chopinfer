//! Cache and F32-tail bindings for the compressor stages. Gated pooling
//! and key writes remain backend operations; the program owns dispatch,
//! workspace and publication. Resolve a source on every step, including a
//! step whose compressor returns no new latent.

use bundle_runtime::state::Extent;
use kernel_core::attention::BlockTables;
use kernel_core::compressed_kv::CompressedKvPages;
use kernel_core::contract::{RecurrentPhase, Refusal};
use kernel_core::pooled_indexer::PositionKeyPages;
use model_deepseek41_contract::geometry::{Geometry, LayerRole};
use model_deepseek41_contract::state::{self, KvStorage, TAIL_ROWS, Widths};
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

fn part(span: Span, offset: u64, length: u64) -> Result<Span, Refusal> {
    let end = offset
        .checked_add(length)
        .ok_or(Refusal("compressor span overflows"))?;
    if end > span.length {
        return Err(Refusal("compressor span exceeds its allocation"));
    }
    let offset = span
        .offset
        .checked_add(offset)
        .ok_or(Refusal("compressor offset overflows"))?;
    offset
        .checked_add(length)
        .ok_or(Refusal("compressor end overflows"))?;
    Ok(Span::new(span.buffer, offset, length))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CachePlane {
    pub values: Span,
    pub scales: Option<Span>,
    pub pages: u32,
    pub positions_per_page: u32,
    pub page_stride: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceBinding {
    pub layer: u32,
    pub ratio: u32,
    /// Visibility after the current step, even when no new group completed.
    pub completed_positions: u64,
    pub storage: KvStorage,
    pub latent: CachePlane,
    pub index: CachePlane,
}

impl SourceBinding {
    pub fn latent_pages(self, tables: BlockTables<Span>) -> CompressedKvPages<Span> {
        CompressedKvPages {
            values: self.latent.values,
            scales: self.latent.scales,
            pages: self.latent.pages,
            value_page_stride: self.latent.page_stride,
            scale_page_stride: self.latent.page_stride,
            page_tables: tables,
        }
    }

    /// The position indexer's native MXFP4 cache record. A BF16 cache needs
    /// a BF16 reader/writer contract and cannot masquerade as packed codes.
    pub fn native_index_pages(
        self,
        tables: BlockTables<Span>,
    ) -> Result<PositionKeyPages<Span>, Refusal> {
        let scales = self
            .index
            .scales
            .ok_or(Refusal("BF16 index keys need a BF16 cache operation"))?;
        Ok(PositionKeyPages {
            codes: self.index.values,
            scales,
            pages: self.index.pages,
            code_page_stride: self.index.page_stride,
            scale_page_stride: self.index.page_stride,
            page_tables: tables,
        })
    }
}

/// Resolve the latest source unconditionally, with this step's pool and
/// token end. No latent option or previous binding participates, so a graph
/// replay or restored slot cannot inherit another step's cache pointer.
pub fn source_binding(
    geometry: &Geometry,
    layer: u32,
    storage: KvStorage,
    pool: Span,
    pages: u32,
    token_end: u64,
) -> Result<Option<SourceBinding>, Refusal> {
    if layer >= geometry.layers() {
        return Err(Refusal("compressor layer is outside the backbone"));
    }
    if geometry.layer_role(layer) == LayerRole::Window {
        return Ok(None);
    }
    if pages == 0 || token_end > geometry.context_tokens() {
        return Err(Refusal("invalid compressor cache extent"));
    }
    let source = geometry
        .shape
        .csa
        .kv_sources
        .iter()
        .rev()
        .copied()
        .find(|&source| source <= layer)
        .ok_or(Refusal("compressed layer has no source"))?;
    let ratio = geometry.compress_ratio(source);
    let declaration = state::request_state(geometry, storage, false);
    let stride = declaration.page_bytes();
    let total = u64::from(pages)
        .checked_mul(stride)
        .ok_or(Refusal("compressor pool extent overflows"))?;
    part(pool, 0, total)?;
    let group = match ratio {
        1 => "csa2.ratio1",
        2 => "csa2.ratio2",
        _ => return Err(Refusal("unsupported compressor ratio")),
    };
    let local_layer = state::sources_of(geometry, ratio)
        .position(|l| l == source)
        .ok_or(Refusal("source absent from cache declaration"))? as u64;
    let positions = declaration.page_tokens / u64::from(ratio);
    let w = Widths::of(geometry);
    let plane = |stream: &str, width: u64, block: u64| -> Result<CachePlane, Refusal> {
        let layout = declaration
            .page_planes()
            .into_iter()
            .find(|p| p.layers == group && p.layer == local_layer && p.stream == stream)
            .ok_or(Refusal("compressor cache plane is missing"))?;
        let values = positions * width * if storage == KvStorage::Native { 1 } else { 4 } / 2;
        let scales = if storage == KvStorage::Native {
            positions * width / block
        } else {
            0
        };
        if values + scales != layout.bytes {
            return Err(Refusal(
                "compressor cache plane disagrees with state layout",
            ));
        }
        let extent = |bytes| {
            u64::from(pages - 1)
                .checked_mul(stride)
                .and_then(|n| n.checked_add(bytes))
                .ok_or(Refusal("compressor plane extent overflows"))
        };
        Ok(CachePlane {
            values: part(pool, layout.offset, extent(values)?)?,
            scales: if scales == 0 {
                None
            } else {
                Some(part(pool, layout.offset + values, extent(scales)?)?)
            },
            pages,
            positions_per_page: positions as u32,
            page_stride: stride,
        })
    };
    Ok(Some(SourceBinding {
        layer: source,
        ratio,
        completed_positions: token_end / u64::from(ratio),
        storage,
        latent: plane("latent", w.latent, 16)?,
        index: plane("index-key", w.index, 32)?,
    }))
}

#[derive(Clone, Copy, Debug)]
pub struct TailBinding {
    /// One source layer's `[slot][history row][key or score][channel]`.
    pub state: Span,
    pub entries: u32,
    pub entry_bytes: u64,
    slots: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TailIds {
    pub source: i32,
    pub checkpoints: Vec<i32>,
    /// Exact byte copy before pooling when an off-grid prefill reads its
    /// incoming tail from a different history cell than it finally writes.
    pub seed_copy: Option<(Span, Span)>,
}

/// The ratio-two source's flat F32 state pool. Ratio one allocates none.
pub fn tail_binding(
    geometry: &Geometry,
    layer: u32,
    extent: Extent,
    tail_buffer: Span,
) -> Result<Option<TailBinding>, Refusal> {
    if !geometry.shape.csa.kv_sources.contains(&layer) {
        return Err(Refusal("only KV sources own compressor state"));
    }
    if geometry.compress_ratio(layer) == 1 {
        return Ok(None);
    }
    let local = state::sources_of(geometry, 2)
        .position(|l| l == layer)
        .ok_or(Refusal("source has no ratio-two tail"))? as u64;
    let entries = u64::from(extent.bucket)
        .checked_mul(TAIL_ROWS)
        .ok_or(Refusal("compressor tail count overflows"))?;
    if extent.bucket == 0 || entries > i32::MAX as u64 {
        return Err(Refusal("compressor tail IDs exceed I32"));
    }
    let entry_bytes = 2 * u64::from(geometry.shape.attention.head_dim) * 4;
    let stride = entries
        .checked_mul(entry_bytes)
        .ok_or(Refusal("compressor tail bytes overflow"))?;
    Ok(Some(TailBinding {
        state: part(tail_buffer, local * stride, stride)?,
        entries: entries as u32,
        entry_bytes,
        slots: extent.bucket,
    }))
}

impl TailBinding {
    pub fn entry(self, id: i32) -> Result<Span, Refusal> {
        if id < 0 || id as u32 >= self.entries {
            return Err(Refusal("compressor tail ID is outside its pool"));
        }
        part(self.state, id as u64 * self.entry_bytes, self.entry_bytes)
    }

    pub fn ids(
        self,
        slot: u32,
        first: u64,
        rows: u32,
        phase: RecurrentPhase,
    ) -> Result<TailIds, Refusal> {
        if slot >= self.slots || rows == 0 {
            return Err(Refusal("invalid compressor tail item"));
        }
        let end = first
            .checked_add(u64::from(rows))
            .ok_or(Refusal("compressor token end overflows"))?;
        let id = |position: u64| (u64::from(slot) * TAIL_ROWS + position % TAIL_ROWS) as i32;
        let incoming = id(if first == 0 { TAIL_ROWS - 1 } else { first - 1 });
        match phase {
            RecurrentPhase::Decode => {
                if rows != 1 {
                    return Err(Refusal("decode tail binding needs one row per item"));
                }
                Ok(TailIds {
                    source: if first % 2 == 0 { id(first) } else { incoming },
                    checkpoints: vec![],
                    seed_copy: None,
                })
            }
            RecurrentPhase::Verify { max_rows } => {
                if rows > max_rows || max_rows == 0 || u64::from(max_rows) >= TAIL_ROWS {
                    return Err(Refusal("verify tail history has insufficient headroom"));
                }
                first
                    .checked_add(u64::from(max_rows))
                    .ok_or(Refusal("verify tail positions overflow"))?;
                let checkpoints = (0..max_rows)
                    .map(|row| id(first + u64::from(row)))
                    .collect();
                Ok(TailIds {
                    source: incoming,
                    checkpoints,
                    seed_copy: None,
                })
            }
            RecurrentPhase::Prefill => {
                let target = if end % 2 == 1 { id(end - 1) } else { incoming };
                let seed_copy = if first % 2 == 1 && end % 2 == 1 && incoming != target {
                    Some((self.entry(incoming)?, self.entry(target)?))
                } else {
                    None
                };
                Ok(TailIds {
                    source: target,
                    checkpoints: vec![],
                    seed_copy,
                })
            }
        }
    }
}

#[cfg(test)]
#[path = "compressor_tests.rs"]
mod tests;
