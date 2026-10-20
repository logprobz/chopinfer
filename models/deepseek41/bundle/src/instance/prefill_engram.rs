//! Prepared, per-layer Engram uploads for eager AR text prefills.
//! Enqueue before compute, then join each transfer at its first consumer.
mod plan;
pub(super) use plan::Plan;

use super::Instance;
use crate::engram_rows::Hashed;
use bundle_runtime::resources::{allocate_span, upload_bytes};
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::{
    BundleAbiError, BundleMappedGatherHandleV1, BundleStatusV1, BundleStreamHandleV1,
    DeviceBufferSpanV1 as Span, HostServicesV1, MappedGatherPlaneV1,
};
use std::cell::Cell;

const SWITCH: &str = "CHOPINFER_DEEPSEEK41_PREFILL_ENGRAM";
const PINNED_LIMIT: u64 = 256 * 1024 * 1024;

pub(super) fn enabled() -> Result<bool, BundleStatusV1> {
    #[cfg(test)]
    if let Some(value) = IN_TESTS.get() {
        return Ok(value);
    }
    match std::env::var(SWITCH).as_deref() {
        Ok("1") => Ok(true),
        Ok("0") | Err(std::env::VarError::NotPresent) => Ok(false),
        _ => Err(BundleStatusV1::InvalidArgument),
    }
}

#[cfg(test)]
thread_local! {
    static IN_TESTS: Cell<Option<bool>> = const { Cell::new(Some(false)) };
}

#[cfg(test)]
pub(crate) fn with_enabled<T>(value: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            IN_TESTS.set(self.0);
        }
    }
    let _restore = Restore(IN_TESTS.replace(Some(value)));
    run()
}

struct Layer {
    id: u32,
    gather: BundleMappedGatherHandleV1,
    indices: Span,
    destinations: Vec<Span>,
}

pub(super) struct Gathers {
    layers: Vec<Layer>,
    columns: usize,
    pending: Cell<bool>,
    failed: Cell<bool>,
    joined: Cell<usize>,
    sequence: Cell<u64>,
    rows: Cell<u32>,
    start: Cell<i32>,
}

impl Gathers {
    pub(super) fn idle(&self) -> bool {
        !self.pending.get() && !self.failed.get()
    }

    /// Copy immutable per-layer IDs before any consumer work is submitted.
    /// All input validation precedes the first upload or enqueue.
    pub(super) fn begin(
        &self,
        services: &HostServicesV1,
        producer: BundleStreamHandleV1,
        hashed: &Hashed,
        start: i32,
    ) -> Result<(), BundleAbiError> {
        if !self.idle() {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
        }
        let rows = hashed.rows();
        let width = self.columns * self.layers.len();
        if rows == 0 || rows > PREFILL_TOKENS as usize || hashed.ids().len() != rows * width {
            return Err(BundleAbiError::InvalidRange("prefill Engram hash extent"));
        }
        let inputs: Vec<Vec<u8>> = (0..self.layers.len())
            .map(|layer| {
                hashed
                    .ids()
                    .chunks_exact(width)
                    .flat_map(|row| &row[layer * self.columns..(layer + 1) * self.columns])
                    .flat_map(|word| word.to_le_bytes())
                    .collect()
            })
            .collect();
        self.pending.set(true);
        self.joined.set(0);
        self.rows.set(rows as u32);
        self.start.set(start);
        self.sequence.set(self.sequence.get().wrapping_add(1));
        let result: Result<(), BundleAbiError> = (|| {
            for (layer, bytes) in self.layers.iter().zip(&inputs) {
                upload_bytes(services, producer, layer.indices, bytes)?;
            }
            for layer in &self.layers {
                services.enqueue_mapped_gather(layer.gather, (rows * self.columns) as u64)?;
            }
            Ok(())
        })();
        if result.is_err() {
            self.failed.set(true);
        }
        result
    }

    fn join(
        &self,
        services: &HostServicesV1,
        streams: &[BundleStreamHandleV1],
        at: usize,
    ) -> Result<(), BundleAbiError> {
        if !self.pending.get() || self.failed.get() || self.joined.get() != at {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
        }
        let layer = self
            .layers
            .get(at)
            .ok_or(BundleAbiError::InvalidRange("prefill Engram join"))?;
        let result = streams
            .iter()
            .try_for_each(|&stream| services.wait_mapped_gather(layer.gather, stream));
        if result.is_err() {
            self.failed.set(true);
        }
        result?;
        self.joined.set(at + 1);
        Ok(())
    }

    /// Existing row-status readbacks have completed all consumer streams.
    /// A callback error still refuses the phase before any result is published.
    pub(super) fn finish(
        &self,
        services: &HostServicesV1,
        ranks: usize,
    ) -> Result<(), BundleAbiError> {
        if !self.pending.get() || self.failed.get() || self.joined.get() != self.layers.len() {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
        }
        let result: Result<(), BundleAbiError> = (|| {
            let mut layers = Vec::new();
            for layer in &self.layers {
                let stats = services.complete_mapped_gather(layer.gather)?.ok_or(
                    BundleAbiError::InvalidRange(
                        "prefill Engram upload incomplete at result boundary",
                    ),
                )?;
                layers.push(format!(
                    "{{\"layer\":{},\"host_ns\":{},\"exposed_ns\":{}}}",
                    layer.id, stats.host_nanoseconds, stats.exposed_nanoseconds
                ));
            }
            let _ = services.log(
                format!(
                    concat!(
                        "{{\"phase\":\"prefill-engram\",\"mode\":\"layer-split\",\"sequence\":{},",
                        "\"rows\":{},\"start\":{},\"ranks\":{},\"input_bytes\":{},",
                        "\"waits\":{},\"layers\":[{}]}}"
                    ),
                    self.sequence.get(),
                    self.rows.get(),
                    self.start.get(),
                    ranks,
                    u64::from(self.rows.get()) * self.columns as u64 * 4 * self.layers.len() as u64,
                    ranks * self.layers.len(),
                    layers.join(",")
                )
                .as_bytes(),
            );
            Ok(())
        })();
        if result.is_err() {
            self.failed.set(true);
        }
        result?;
        self.pending.set(false);
        Ok(())
    }

    pub(super) fn release(&mut self, services: &HostServicesV1) -> Result<(), BundleAbiError> {
        // The host fences unfinished callbacks and DMA before releasing their
        // pinned buffers. This precedes release of every referenced allocation.
        while let Some(layer) = self.layers.last() {
            services.release_mapped_gather(layer.gather)?;
            self.layers.pop();
        }
        Ok(())
    }
}

impl Instance {
    pub(super) fn prepare_prefill_engram(&mut self) -> Result<(), BundleAbiError> {
        if !enabled().map_err(BundleAbiError::Status)?
            || self.dspark.is_some()
            || !self.services.has_async_mapped_gather()
        {
            return Ok(());
        }
        let Some(tables) = self
            .engram
            .as_ref()
            .and_then(|e| e.tables())
            .map(<[_]>::to_vec)
        else {
            return Ok(());
        };
        if tables.is_empty() {
            return Ok(());
        }
        let image = self
            .image
            .ok_or(BundleAbiError::InvalidRange("prefill Engram image"))?;
        let shape = &self.config.shape;
        let e = &shape.geometry.shape.engram;
        let columns = ((e.max_ngram - 1) * e.heads) as usize;
        let bytes = u64::from(PREFILL_TOKENS) * columns as u64 * 4;
        let mut planes = vec![Vec::new(); tables.len()];
        for (rank, buffers) in self.buffers.iter().enumerate() {
            for (layer, table) in tables.iter().enumerate() {
                for (plane, staged) in [
                    (&table.codes, buffers.scratch.engram_codes),
                    (&table.scales, buffers.scratch.engram_scales),
                ] {
                    let destination =
                        crate::buffers::engram_staged(staged, shape, layer, PREFILL_TOKENS);
                    if destination.length
                        != u64::from(PREFILL_TOKENS) * columns as u64 * plane.row_bytes
                    {
                        return Err(BundleAbiError::InvalidRange(
                            "prefill Engram destination extent",
                        ));
                    }
                    planes[layer].push(MappedGatherPlaneV1 {
                        image,
                        rank: rank as u32,
                        view_id: plane.view.clone(),
                        row_bytes: plane.row_bytes,
                        index_offset: 0,
                        stream: self.ranks[rank].2,
                        destination,
                    });
                }
            }
        }
        if planes.iter().any(|p| {
            p.iter()
                .fold(bytes, |sum, p| sum.saturating_add(p.destination.length))
                > PINNED_LIMIT
        }) {
            return Err(BundleAbiError::InvalidRange(
                "prefill Engram pinned capacity",
            ));
        }
        // Allocate all ID spans before creating any gather, so reverse cleanup
        // releases every gather before the allocations its transfers reference.
        let (_, pool, stream) = self.ranks[0];
        let indices = (0..tables.len())
            .map(|_| allocate_span(&self.services, pool, stream, bytes, &mut self.held))
            .collect::<Result<Vec<_>, _>>()?;
        self.prefill_engram = Some(Gathers {
            layers: Vec::new(),
            columns,
            pending: Cell::new(false),
            failed: Cell::new(false),
            joined: Cell::new(0),
            sequence: Cell::new(0),
            rows: Cell::new(0),
            start: Cell::new(0),
        });
        for (index, planes) in planes.into_iter().enumerate() {
            let gather = self
                .services
                .create_mapped_gather(stream, indices[index], &planes)?;
            self.prefill_engram
                .as_mut()
                .expect("gather cleanup retained")
                .layers
                .push(Layer {
                    id: e.layers[index],
                    gather,
                    indices: indices[index],
                    destinations: planes.iter().map(|p| p.destination).collect(),
                });
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn indices(handle: runtime_bundle_abi::BundleHandleV1) -> Vec<(u32, Span)> {
    use bundle_runtime::entry::Bundle;
    let mut result = Vec::new();
    Instance::instances().with(|table| {
        let status = bundle_runtime::table::with_instance(table, handle, |instance| {
            if let Some(gathers) = &instance.prefill_engram {
                result = gathers.layers.iter().map(|l| (l.id, l.indices)).collect();
            }
            BundleStatusV1::Success
        });
        assert_eq!(status, BundleStatusV1::Success);
    });
    result
}
