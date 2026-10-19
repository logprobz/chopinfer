//! Optional retirement snapshots for eager/graph parity, outside timed cohorts.

use super::Instance;
use crate::dspark::device::slice;
use bundle_runtime::snapshot::{Descriptor, Table};
use bundle_runtime::state::{Capture, Holder};
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;
use std::path::Path;

impl Instance {
    pub(crate) fn capture_dspark_state(&mut self, request: u64) -> Result<(), String> {
        let result = match self.active_dspark_state_capture() {
            Some(root) => self.capture_dspark_state_to(request, root),
            None => Ok(()),
        };
        self.state_capture_pages.remove(&request);
        result
    }

    pub(crate) fn active_dspark_state_capture(&self) -> Option<&Path> {
        self.state_capture_root
            .as_deref()
            .filter(|root| root.join("enabled").is_file())
    }

    pub(crate) fn capture_dspark_state_to(&self, request: u64, root: &Path) -> Result<(), String> {
        let Some(context) = self
            .dspark
            .as_ref()
            .and_then(|device| device.runtime.as_ref())
            .and_then(|runtime| runtime.state().context(request))
        else {
            return Ok(());
        };
        self.capture_complete_state_to(request, context.end, root)
    }

    pub(crate) fn capture_complete_state_to(
        &self,
        request: u64,
        end: u64,
        root: &Path,
    ) -> Result<(), String> {
        let slot = self
            .ledger
            .committed(request, end)
            .map_err(|e| format!("state capture ledger: {e:?}"))?;
        let context = self
            .dspark
            .as_ref()
            .and_then(|d| d.runtime.as_ref())
            .and_then(|r| r.state().context(request));
        let retained_start = context.map_or(
            end.saturating_sub(u64::from(self.config.shape.geometry.shape.attention.window)),
            |c| c.retained_start,
        );
        let projected = context.map_or_else(Vec::new, |c| c.projected().to_vec());
        let mut table = Table::new(&self.state, self.extent, Holder::Slot(slot))
            .map_err(|e| format!("DSpark state capture table: {e:?}"))?;
        // Prefix restore rebuilds the drafter rings from retained features.
        // Parity captures include those rings as well as the prefix payload.
        for (index, buffer) in self
            .state
            .buffers
            .iter()
            .enumerate()
            .filter(|(_, buffer)| buffer.drafter && buffer.capture == Capture::None)
        {
            let entries = buffer.entries(self.extent);
            for layer in 0..buffer.layers {
                let mut plane_start = layer * buffer.layer_stride(entries);
                for &length in &buffer.planes {
                    let at = table.header_at();
                    table.descriptors.push(Descriptor {
                        buffer: index,
                        spare: false,
                        offset: plane_start + u64::from(slot) * length,
                        length,
                        at,
                    });
                    plane_start += entries * length;
                }
            }
        }
        let directory = root.join(format!("request-{request}"));
        std::fs::create_dir(&directory)
            .map_err(|e| format!("DSpark state capture directory: {e}"))?;
        let page_ids = self
            .state_capture_pages
            .get(&request)
            .ok_or("DSpark state capture has no current block table")?;
        let mut paged_pieces = Vec::new();
        for (rank, buffers) in self.buffers.iter().enumerate() {
            let spans = table
                .descriptors
                .iter()
                .map(|part| {
                    if part.spare {
                        return Err("DSpark state capture has an unsupported spare".into());
                    }
                    slice(
                        buffers.state.buffers[part.buffer].1,
                        part.offset,
                        part.length,
                    )
                })
                .collect::<Result<Vec<_>, String>>()?;
            let bytes = self.read_capture_spans(rank, &spans)?;
            std::fs::write(directory.join(format!("rank-{rank}.bin")), bytes)
                .map_err(|e| format!("DSpark state capture write: {e}"))?;
            let mut paged = Vec::new();
            let mut metadata = Vec::new();
            let mut at = 0;
            let geometry = self.config.shape.geometry;
            for &layer in geometry.shape.csa.kv_sources.iter() {
                let binding = crate::compressor::source_binding(
                    geometry,
                    layer,
                    self.config.kv.storage,
                    buffers.kv_pool,
                    u32::try_from(self.kv_pages)
                        .map_err(|_| "DSpark capture page count exceeds U32")?,
                    end,
                )
                .map_err(|e| e.to_string())?
                .ok_or("DSpark state capture KV source is absent")?;
                for (kind, cache) in [("latent", binding.latent), ("index-key", binding.index)] {
                    for (plane, source) in [("codes", Some(cache.values)), ("scales", cache.scales)]
                    {
                        let Some(source) = source else {
                            continue;
                        };
                        for (logical, span) in
                            committed_pages(cache, source, page_ids, binding.completed_positions)?
                                .into_iter()
                                .enumerate()
                        {
                            metadata.push(
                                serde_json::json!({"layer":layer, "kind":kind, "plane":plane,
                                "logical_page":logical, "at":at, "bytes":span.length}),
                            );
                            at += span.length;
                            paged.push(span);
                        }
                    }
                }
            }
            let bytes = self.read_capture_spans(rank, &paged)?;
            std::fs::write(directory.join(format!("kv-rank-{rank}.bin")), bytes)
                .map_err(|e| format!("DSpark paged state capture write: {e}"))?;
            if rank == 0 {
                paged_pieces = metadata;
            } else if metadata != paged_pieces {
                return Err("DSpark paged capture differs across ranks".into());
            }
        }
        let pieces: Vec<_> = table.descriptors.iter().map(|part| {
            serde_json::json!({"buffer":self.state.buffers[part.buffer].name, "bytes":part.length, "at":part.at})
        }).collect();
        let metadata = serde_json::json!({
            "schema":"chopinfer.deepseek41.dspark-state-capture.v1",
            "end":end, "retained_start":retained_start,
            "projected":projected, "ranks":self.ranks.len(), "pieces":pieces, "paged_pieces":paged_pieces,
        });
        std::fs::write(
            directory.join("state.json"),
            serde_json::to_vec_pretty(&metadata).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("DSpark state capture metadata: {e}"))?;
        Ok(())
    }
    pub(crate) fn capture_profile_boundary(
        &self,
        request: u64,
        end: u64,
        kind: &str,
    ) -> Result<(), String> {
        if self.config.prefill.chunk == 0 {
            return Ok(());
        }
        let Some(root) = self.active_dspark_state_capture() else {
            return Ok(());
        };
        let directory = root.join(format!("{kind}-{end}"));
        std::fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        self.capture_complete_state_to(request, end, &directory)
    }

    pub(crate) fn capture_profile_chunks(&self) -> Result<(), String> {
        for boundary in &self.prefill_boundaries {
            self.capture_profile_boundary(boundary.request_id, boundary.chunk_end, "chunk")?;
        }
        Ok(())
    }

    pub(crate) fn capture_profile_logits(
        &self,
        views: &[runtime_bundle_abi::BundleLogitsViewV1],
    ) -> Result<(), String> {
        if self.config.prefill.chunk == 0 {
            return Ok(());
        }
        let Some(root) = self.active_dspark_state_capture() else {
            return Ok(());
        };
        let directory = root.join("logits");
        std::fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        for (index, view) in views.iter().enumerate() {
            let spans: Vec<_> = (0..view.row_count)
                .map(|row| {
                    Span::new(
                        view.buffer.buffer,
                        view.buffer.offset + u64::from(row) * view.row_stride_elements * 4,
                        u64::from(view.live_width) * 4,
                    )
                })
                .collect();
            let bytes = self.read_capture_spans(view.rank as usize, &spans)?;
            std::fs::write(
                directory.join(format!("step-{}-view-{index}.f32", self.steps.get())),
                bytes,
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn read_capture_spans(&self, rank: usize, spans: &[Span]) -> Result<Vec<u8>, String> {
        let stream = self.ranks[rank].2;
        self.services
            .synchronize_stream(stream)
            .map_err(|e| e.to_string())?;
        let length = spans
            .iter()
            .try_fold(0_usize, |sum, span| {
                sum.checked_add(usize::try_from(span.length).ok()?)
            })
            .ok_or("DSpark state capture is too large")?;
        let mut bytes = vec![0; length];
        let mut at = 0;
        let copied = (|| -> Result<(), String> {
            for &span in spans {
                let end = at + span.length as usize;
                self.services
                    .copy_device_to_host(stream, &mut bytes[at..end], span)
                    .map_err(|e| e.to_string())?;
                at = end;
            }
            Ok(())
        })();
        if let Err(error) = self.services.synchronize_stream(stream) {
            // Keep queued host destinations alive if completion is unknown.
            self.collectives.poison();
            std::mem::forget(bytes);
            return Err(format!("DSpark state capture drain: {error}"));
        }
        copied?;
        Ok(bytes)
    }
}

fn committed_pages(
    cache: crate::compressor::CachePlane,
    source: Span,
    pages: &[i32],
    positions: u64,
) -> Result<Vec<Span>, String> {
    let capacity = u64::from(cache.positions_per_page);
    if capacity == 0 || cache.pages == 0 {
        return Err("invalid DSpark paged capture capacity".into());
    }
    let prefix = u64::from(cache.pages - 1) * cache.page_stride;
    let plane_bytes = source
        .length
        .checked_sub(prefix)
        .ok_or("short DSpark cache plane")?;
    if plane_bytes == 0 || !plane_bytes.is_multiple_of(capacity) {
        return Err("DSpark cache plane has incomplete entries".into());
    }
    let width = plane_bytes / capacity;
    (0..positions.div_ceil(capacity))
        .map(|logical| {
            let page = *pages
                .get(logical as usize)
                .ok_or("DSpark capture block table is short")?;
            let page = u32::try_from(page).map_err(|_| "DSpark capture page is negative")?;
            if page >= cache.pages {
                return Err("DSpark capture page is outside the pool".into());
            }
            let live = (positions - logical * capacity).min(capacity);
            slice(source, u64::from(page) * cache.page_stride, live * width)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    #[test]
    fn committed_pages_exclude_rejected_rows_and_follow_physical_tables() {
        let values = Span::new(
            BundleDeviceBufferHandleV1::from_parts(1, 1),
            16,
            7 * 256 + 32 * 4,
        );
        let cache = crate::compressor::CachePlane {
            values,
            scales: None,
            pages: 8,
            positions_per_page: 32,
            page_stride: 256,
        };
        let spans = committed_pages(cache, values, &[7, 2, -1], 35).unwrap();
        assert_eq!(
            spans,
            vec![
                Span::new(values.buffer, 16 + 7 * 256, 128),
                Span::new(values.buffer, 16 + 2 * 256, 12)
            ]
        );
        assert!(committed_pages(cache, values, &[], 0).unwrap().is_empty());
        assert!(committed_pages(cache, values, &[7], 35).is_err());
        assert!(committed_pages(cache, values, &[-1], 1).is_err());
    }
}
