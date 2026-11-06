//! Prepared, per-layer Engram uploads for eager text prefills.
//! Enqueue before compute, then join each transfer at its first consumer.
mod plan;
#[cfg(test)]
pub(super) use plan::Builder;
pub(super) use plan::{DmaPiece, Plan};

use super::{
    Instance,
    layer_gather::{self, LayerGather},
};
use crate::engram_rows::{Destination, Hashed};
use bundle_runtime::inputs::PhaseInputs;
use bundle_runtime::resources::upload_bytes;
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::{
    BundleAbiError, BundleStatusV1, BundleStreamHandleV1, DeviceBufferSpanV1 as Span,
    HostServicesV1,
};
use std::cell::Cell;

const PINNED_LIMIT: u64 = 256 * 1024 * 1024;

pub(super) struct Gathers {
    layers: Vec<LayerGather>,
    /// The ids and destinations of a mixed DSpark step's verify-row
    /// gathers, which begin once the last layer joined
    /// (`dspark_mixed_prepared_engram`); empty without the candidate.
    verify: Vec<Span>,
    columns: usize,
    pending: Cell<bool>,
    /// The pending gathers began before their step's planning
    /// (`dspark_early_prefill_engram`), and its target has not taken them.
    early: Cell<bool>,
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

    /// Whether a step's Engram plan may be made: no gather is pending, or
    /// only those this step began early, before any join.
    pub(super) fn plannable(&self) -> bool {
        self.idle() || (self.early.get() && !self.failed.get() && self.joined.get() == 0)
    }

    /// Whether the pending gathers began early.
    pub(super) fn begun_early(&self) -> bool {
        self.early.get()
    }

    /// Takes the gathers this step began early for its target, which
    /// hashed the same `rows` from position `start`; false when none
    /// began early.
    pub(super) fn take_early(&self, rows: usize, start: i32) -> Result<bool, BundleAbiError> {
        if !self.early.get() {
            return Ok(false);
        }
        if !self.pending.get()
            || self.failed.get()
            || self.joined.get() != 0
            || self.rows.get() as usize != rows
            || self.start.get() != start
        {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
        }
        self.early.set(false);
        Ok(true)
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
        self.early.set(false);
        self.joined.set(0);
        self.rows.set(rows as u32);
        self.start.set(start);
        self.sequence.set(self.sequence.get().wrapping_add(1));
        let result: Result<(), BundleAbiError> = (|| {
            for (layer, bytes) in self.layers.iter().zip(&inputs) {
                upload_bytes(services, producer, layer.indices, bytes)?;
            }
            for layer in &self.layers {
                layer.begin(
                    services,
                    (rows * self.columns) as u64,
                    crate::engram_rows::PreparedGather::Prefill,
                )?;
            }
            Ok(())
        })();
        if result.is_err() {
            self.failed.set(true);
        }
        result
    }

    pub(in crate::instance) fn join(
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
            .try_for_each(|&stream| layer.join(services, stream));
        if result.is_err() {
            self.failed.set(true);
        }
        result?;
        self.joined.set(at + 1);
        Ok(())
    }

    /// All ranks' destinations and immutable IDs still owned by pending
    /// reads: once every layer joined, a mixed step's verify-row gathers'.
    pub(in crate::instance) fn pending_spans(&self, joined: usize) -> Vec<Span> {
        if joined == self.layers.len() {
            return self.verify.clone();
        }
        self.layers[joined..]
            .iter()
            .flat_map(|layer| {
                std::iter::once(layer.indices).chain(layer.destinations.iter().copied())
            })
            .collect()
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
                let stats = layer.finish(services)?.ok_or(BundleAbiError::InvalidRange(
                    "prefill Engram upload incomplete at result boundary",
                ))?;
                layers.push(format!(
                    "{{\"layer\":{},\"host_ns\":{},\"exposed_ns\":{}}}",
                    layer.layer, stats.host_nanoseconds, stats.exposed_nanoseconds
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
        self.early.set(false);
        Ok(())
    }

    /// Settle a refused step before any later upload can reuse its rows.
    /// Host callback/handle failures remain terminal; they cannot be retried.
    fn settle(
        &self,
        services: &HostServicesV1,
        streams: &[BundleStreamHandleV1],
    ) -> Result<(), BundleAbiError> {
        if self.failed.get() {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
        }
        if !self.pending.get() {
            return Ok(());
        }
        let result = (|| {
            for layer in self.joined.get()..self.layers.len() {
                self.join(services, streams, layer)?;
            }
            for &stream in streams {
                services.synchronize_stream(stream)?;
            }
            self.finish(services, streams.len())
        })();
        if result.is_err() {
            self.failed.set(true);
        }
        result
    }

    pub(super) fn release(&mut self, services: &HostServicesV1) -> Result<(), BundleAbiError> {
        // The host fences unfinished callbacks and DMA before releasing their
        // pinned buffers. This precedes release of every referenced allocation.
        while let Some(layer) = self.layers.last() {
            layer.release(services)?;
            self.layers.pop();
        }
        Ok(())
    }
}

impl Instance {
    /// Joins prefill Engram layer `at` on every stream, then, after the
    /// last layer, begins a mixed DSpark step's verify-row gathers
    /// (`begin_mixed_dspark_engram`).
    pub(in crate::instance) fn join_prefill_engram(
        &self,
        streams: &[BundleStreamHandleV1],
        at: usize,
    ) -> Result<(), BundleAbiError> {
        let gathers = self
            .prefill_engram
            .as_ref()
            .ok_or(BundleAbiError::Status(BundleStatusV1::InvalidState))?;
        gathers.join(&self.services, streams, at)?;
        if at + 1 == gathers.layers.len() {
            self.begin_mixed_dspark_engram()?;
        }
        Ok(())
    }

    pub(super) fn settle_prefill_engram(&self) -> Result<(), BundleAbiError> {
        if let Some(gathers) = self
            .prefill_engram
            .as_ref()
            .filter(|gathers| !gathers.idle())
        {
            if self.collectives.is_poisoned() {
                return Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
            }
            let streams = self.ranks.iter().map(|rank| rank.2).collect::<Vec<_>>();
            gathers.settle(&self.services, &streams)?;
        }
        Ok(())
    }

    /// Whether a DSpark step's prefill hashes its rows and begins its
    /// prepared gathers before catch-up and drafting
    /// (`dspark_early_prefill_engram`).
    pub(super) fn admits_early_prefill_engram(
        &self,
        request: &[u8],
        prefill: &PhaseInputs,
    ) -> bool {
        self.config.execution.dspark_early_prefill_engram
            && self.takes_prepared_prefill(request, prefill)
    }

    /// Whether a DSpark step plans its prefill while its drafter runs
    /// (`dspark_early_prefill_plan`).
    pub(super) fn admits_early_prefill_plan(&self, request: &[u8], prefill: &PhaseInputs) -> bool {
        self.config.execution.dspark_early_prefill_plan
            && self.takes_prepared_prefill(request, prefill)
    }

    /// Whether a DSpark step's prefill takes the prepared gathers with ids
    /// staged before its target, exactly where its target would
    /// (`prefill_gather_reason`): a prefill reading no encoder rows on an
    /// instance with prepared gathers and no text taps.
    fn takes_prepared_prefill(&self, request: &[u8], prefill: &PhaseInputs) -> bool {
        let reads_media = || {
            let Ok(batch) = runtime_bundle_sdk::BatchCodecV1::decode(request) else {
                return true;
            };
            crate::encoders::ARENA
                .plan(batch.media(), prefill, batch.items().len())
                .ok()
                .is_none_or(|plan| plan.injects())
        };
        self.dspark.is_some()
            && self.prefill_engram.is_some()
            && !self.taps.text()
            && !prefill.request_ids.is_empty()
            && (self.media.is_none() || !reads_media())
    }

    /// Begins the prefill's prepared gathers of `hashed` ahead of its
    /// step's planning, which takes them (`Gathers::take_early`). The
    /// previous step's gathers settled at execute entry, and no work the
    /// step issues before the prefill's joins touches their ids or rows.
    pub(super) fn begin_early_prefill_engram(
        &self,
        prefill: &PhaseInputs,
        hashed: &Hashed,
    ) -> Result<(), BundleAbiError> {
        let gathers = self
            .prefill_engram
            .as_ref()
            .ok_or(BundleAbiError::Status(BundleStatusV1::InvalidState))?;
        gathers.begin(
            &self.services,
            self.ranks[0].2,
            hashed,
            prefill.positions.first().copied().unwrap_or(0),
        )?;
        gathers.early.set(true);
        Ok(())
    }

    /// The exact row domain read by the target, for both prepared gathers
    /// and their synchronous fallback. DSpark separates prefill and decode.
    pub(super) fn engram_destinations(
        &self,
        scratch: &crate::buffers::Scratch<Span>,
        rows: u32,
    ) -> Result<Vec<Destination>, BundleAbiError> {
        let shape = &self.config.shape;
        if let Some(device) = &self.dspark {
            let bytes = u64::from(rows) * u64::from(crate::ops::dense::engram_k(shape.geometry));
            device
                .engram_destinations(scratch)
                .and_then(|destinations| {
                    destinations
                        .into_iter()
                        .map(|destination| {
                            Ok(Destination {
                                codes: crate::dspark::device::slice(destination.codes, 0, bytes)?,
                                scales: crate::dspark::device::slice(
                                    destination.scales,
                                    0,
                                    bytes / 32,
                                )?,
                            })
                        })
                        .collect()
                })
                .map_err(|_| BundleAbiError::InvalidRange("DSpark prefill Engram destinations"))
        } else {
            Ok((0..shape.geometry.engram_layers() as usize)
                .map(|layer| Destination {
                    codes: crate::buffers::engram_staged(scratch.engram_codes, shape, layer, rows),
                    scales: crate::buffers::engram_staged(
                        scratch.engram_scales,
                        shape,
                        layer,
                        rows,
                    ),
                })
                .collect())
        }
    }

    pub(super) fn prepare_prefill_engram(&mut self) -> Result<(), BundleAbiError> {
        if self.prefill_engram_inputs.is_empty() {
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
            let destinations = self.engram_destinations(&buffers.scratch, PREFILL_TOKENS)?;
            for (layer, table) in tables.iter().enumerate() {
                for (plane, destination) in [
                    (&table.codes, destinations[layer].codes),
                    (&table.scales, destinations[layer].scales),
                ] {
                    if destination.length
                        != u64::from(PREFILL_TOKENS) * columns as u64 * plane.row_bytes
                    {
                        return Err(BundleAbiError::InvalidRange(
                            "prefill Engram destination extent",
                        ));
                    }
                    planes[layer].push(layer_gather::plane(
                        image,
                        rank as u32,
                        self.ranks[rank].2,
                        plane,
                        destination,
                    ));
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
        // ID spans were reserved before KV sizing. Gather cleanup still
        // precedes release of every allocation its transfers reference.
        let (_, _, stream) = self.ranks[0];
        let indices = &self.prefill_engram_inputs;
        if indices.len() != tables.len() || indices.iter().any(|span| span.length != bytes) {
            return Err(BundleAbiError::InvalidRange("prefill Engram ID storage"));
        }
        // A mixed step's verify rows gather through the DSpark cycle's
        // layers, so the prefill's audits keep its launches off them.
        let verify = self
            .dspark
            .as_ref()
            .filter(|_| self.config.execution.dspark_mixed_prepared_engram)
            .map_or_else(Vec::new, |device| {
                device
                    .cycle
                    .layers
                    .iter()
                    .flat_map(|layer| {
                        std::iter::once(layer.indices).chain(layer.destinations.iter().copied())
                    })
                    .collect()
            });
        self.prefill_engram = Some(Gathers {
            layers: Vec::new(),
            verify,
            columns,
            pending: Cell::new(false),
            early: Cell::new(false),
            failed: Cell::new(false),
            joined: Cell::new(0),
            sequence: Cell::new(0),
            rows: Cell::new(0),
            start: Cell::new(0),
        });
        for (index, planes) in planes.into_iter().enumerate() {
            let gather = LayerGather::create(
                &self.services,
                stream,
                e.layers[index],
                indices[index],
                &planes,
                || (),
            )?;
            self.prefill_engram
                .as_mut()
                .expect("gather cleanup retained")
                .layers
                .push(gather);
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
                result = gathers
                    .layers
                    .iter()
                    .map(|l| (l.layer, l.indices))
                    .collect();
            }
            BundleStatusV1::Success
        });
        assert_eq!(status, BundleStatusV1::Success);
    });
    result
}

#[cfg(test)]
pub(crate) fn refusal_state(handle: runtime_bundle_abi::BundleHandleV1) -> (bool, bool) {
    use bundle_runtime::entry::Bundle;
    let mut result = None;
    Instance::instances().with(|table| {
        let status = bundle_runtime::table::with_instance(table, handle, |instance| {
            result = Some((
                instance.prefill_engram.as_ref().is_none_or(Gathers::idle),
                instance
                    .dspark
                    .as_ref()
                    .and_then(|device| device.runtime.as_ref())
                    .is_some_and(|runtime| runtime.state().is_poisoned()),
            ));
            BundleStatusV1::Success
        });
        assert_eq!(status, BundleStatusV1::Success);
    });
    result.expect("live instance")
}

#[cfg(test)]
pub(crate) fn settle_refused(
    handle: runtime_bundle_abi::BundleHandleV1,
) -> Result<(), BundleAbiError> {
    use bundle_runtime::entry::Bundle;
    // Deliberately reach the local guard even when the ABI table's outer
    // poison check would reject the call first.
    Instance::instances().with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("live instance")
            .settle_prefill_engram()
    })
}
