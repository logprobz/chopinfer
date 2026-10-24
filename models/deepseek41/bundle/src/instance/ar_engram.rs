//! Per-layer Engram uploads for AR decode graphs. They are the default;
//! `CHOPINFER_DEEPSEEK41_AR_ENGRAM=0` keeps the synchronous route.
//!
//! The synchronous route has the engine thread gather every Engram layer's
//! table rows for every rank before the step's first launch. Here the host
//! hashes the same rows from the same histories, uploads the ids once, and
//! the prepared mapped gather (`create_mapped_gather`) copies each layer's
//! rows on its own streams while the graph prefix runs. Each rank joins a
//! layer's upload immediately before that layer's `EngramRows` consumer,
//! through the same layered target graphs DSpark uses
//! (`dspark::cycle::graphs`). The staged bytes, the kernels and their order
//! on every stream are unchanged; only when the copies run moves.
//!
//! The switch also serves a DSpark instance's decode steps without drafts:
//! their host-hashed ids go through the gathers DSpark's drafted steps use
//! (`instance::dspark::cycle`).

use super::Instance;
use crate::dspark::cycle::LayerGather;
use crate::engram_rows::Hashed;
use crate::indexer_workloads::Route as IndexerRoute;
use bundle_runtime::graphs::{GraphCache, GraphKey, GraphReservation};
use bundle_runtime::resources::{allocate_span, upload_bytes};
use runtime_bundle_abi::{
    BundleAbiError, BundleStatusV1, DeviceBufferSpanV1 as Span, MappedGatherPlaneV1,
    MappedGatherStatsV1,
};

const SWITCH: &str = "CHOPINFER_DEEPSEEK41_AR_ENGRAM";
/// Logs each step's per-layer host and exposed wait times when set to 1.
const TRACE: &str = "CHOPINFER_DEEPSEEK41_AR_ENGRAM_TRACE";

/// The switch's decision for its raw value: absent or `1` serves AR decode
/// through the per-layer route, `0` keeps the synchronous route.
fn decide(value: Option<&str>) -> Result<bool, BundleStatusV1> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(_) => Err(BundleStatusV1::InvalidArgument),
    }
}

pub(super) fn enabled() -> Result<bool, BundleStatusV1> {
    #[cfg(test)]
    return decide(IN_TESTS.get());
    #[cfg(not(test))]
    match std::env::var(SWITCH) {
        Ok(value) => decide(Some(&value)),
        Err(std::env::VarError::NotPresent) => decide(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(BundleStatusV1::InvalidArgument),
    }
}

pub(super) fn traced() -> bool {
    cfg!(test) || std::env::var_os(TRACE).is_some_and(|value| value == "1")
}

#[cfg(test)]
thread_local! {
    /// The switch's raw value in tests, `None` being unset. Tests keep the
    /// synchronous route unless they set it, so their pinned journals stay
    /// that route's.
    static IN_TESTS: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(Some("0")) };
}

#[cfg(test)]
pub(crate) fn with_enabled<T>(value: bool, run: impl FnOnce() -> T) -> T {
    with_switch(Some(if value { "1" } else { "0" }), run)
}

/// Runs `run` as if the switch held `value`.
#[cfg(test)]
pub(crate) fn with_switch<T>(value: Option<&'static str>, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<&'static str>);
    impl Drop for Restore {
        fn drop(&mut self) {
            IN_TESTS.set(self.0);
        }
    }
    let _restore = Restore(IN_TESTS.replace(value));
    run()
}

/// The prepared gathers of an instance serving AR decode through them.
pub(crate) struct State {
    /// Rank 0's ids: each layer's `capacity * columns` U32 slots in turn.
    indices: Span,
    pub(crate) layers: Vec<LayerGather>,
    /// Rows of the widest decode bucket, which every staging span holds.
    capacity: u32,
    /// Hash ids per row of one layer.
    columns: usize,
    /// Layers whose upload was enqueued and not yet completed.
    enqueued: usize,
    /// Leading layers every rank's stream joined; the launch joins them
    /// through a shared borrow.
    joined: std::cell::Cell<usize>,
    rows: u32,
    sequence: u64,
    trace: bool,
}

impl State {
    /// Whether decode graphs of `(bucket, items)` join their layers here:
    /// every decode bucket, one row included. A one-row step's synchronous
    /// gather faults each row in on the engine thread, one at a time; here
    /// the read pool overlaps them and the graph prefix hides them.
    pub(crate) fn serves(&self, (bucket, items): (u32, u32)) -> bool {
        items == 0 && bucket >= 1 && bucket <= self.capacity && !self.layers.is_empty()
    }

    fn idle(&self) -> bool {
        self.enqueued == 0
    }
}

/// One layer's upload times in a step's trace.
pub(super) fn layer_trace(layer: u32, stats: &MappedGatherStatsV1) -> String {
    format!(
        "{{\"layer\":{layer},\"host_ns\":{},\"exposed_ns\":{}}}",
        stats.host_nanoseconds, stats.exposed_nanoseconds
    )
}

/// A step's trace: its sequence, rows and each layer's upload times.
pub(super) fn step_trace(sequence: u64, rows: u32, layers: &[String]) -> String {
    format!(
        concat!(
            "{{\"schema\":\"chopinfer.deepseek41.ar-engram.v1\",\"sequence\":{},",
            "\"rows\":{},\"layers\":[{}]}}"
        ),
        sequence,
        rows,
        layers.join(",")
    )
}

/// `hashed`'s ids, `[row][layer][column]`, as each layer's U32 words in
/// turn, `capacity` rows per layer; rows past the live ones stay zero.
pub(super) fn layer_major(
    hashed: &Hashed,
    layers: usize,
    columns: usize,
    capacity: usize,
) -> Result<Vec<u8>, BundleAbiError> {
    let rows = hashed.rows();
    if rows == 0
        || rows > capacity
        || layers == 0
        || columns == 0
        || hashed.ids().len() != rows * layers * columns
    {
        return Err(BundleAbiError::InvalidRange("AR Engram hash extent"));
    }
    let mut bytes = vec![0; layers * capacity * columns * 4];
    for (row, ids) in hashed.ids().chunks_exact(layers * columns).enumerate() {
        for layer in 0..layers {
            let at = ((layer * capacity + row) * columns) * 4;
            for (column, id) in ids[layer * columns..(layer + 1) * columns]
                .iter()
                .enumerate()
            {
                bytes[at + column * 4..at + column * 4 + 4].copy_from_slice(&id.to_le_bytes());
            }
        }
    }
    Ok(bytes)
}

impl Instance {
    /// Prepares one gather per Engram layer whose planes are every rank's
    /// codes and scales staging, sized for the widest decode bucket, when
    /// the switch is on and the instance serves AR with mapped tables.
    pub(super) fn prepare_ar_engram(&mut self) -> Result<(), BundleAbiError> {
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
            .ok_or(BundleAbiError::InvalidRange("AR Engram image"))?;
        let shape = self.config.shape;
        let e = &shape.geometry.shape.engram;
        let columns = ((e.max_ngram - 1) * e.heads) as usize;
        let capacity = self.items().capacity();
        let bytes = u64::from(capacity) * columns as u64 * 4;
        // Ids live on rank 0's stream, the producer every gather reads.
        let (_, pool, stream) = self.ranks[0];
        let indices = allocate_span(
            &self.services,
            pool,
            stream,
            bytes * tables.len() as u64,
            &mut self.held,
        )?;
        let mut state = State {
            indices,
            layers: Vec::new(),
            capacity,
            columns,
            enqueued: 0,
            joined: std::cell::Cell::new(0),
            rows: 0,
            sequence: 0,
            trace: traced(),
        };
        for (layer, table) in tables.iter().enumerate() {
            let mut planes = Vec::new();
            for (rank, buffers) in self.buffers.iter().enumerate() {
                for (plane, staged) in [
                    (&table.codes, buffers.scratch.engram_codes),
                    (&table.scales, buffers.scratch.engram_scales),
                ] {
                    let destination =
                        crate::buffers::engram_staged(staged, &shape, layer, capacity);
                    if destination.length != u64::from(capacity) * columns as u64 * plane.row_bytes
                    {
                        return Err(BundleAbiError::InvalidRange("AR Engram destination extent"));
                    }
                    planes.push(MappedGatherPlaneV1 {
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
            let ids = Span::new(indices.buffer, indices.offset + layer as u64 * bytes, bytes);
            // A failed creation leaves the earlier gathers in the state, so
            // release still drains them before their allocations.
            let gather = match self.services.create_mapped_gather(stream, ids, &planes) {
                Ok(gather) => gather,
                Err(error) => {
                    self.ar_engram = Some(state);
                    return Err(error);
                }
            };
            let mut graphs = GraphCache::sharing(&self.graphs);
            graphs.label("ar-engram", None, layer as u32 + 1);
            state.layers.push(LayerGather {
                layer: e.layers[layer],
                gather,
                indices: ids,
                destinations: planes.iter().map(|plane| plane.destination).collect(),
                graphs,
            });
        }
        self.ar_engram = Some(state);
        self.log(&format!(
            "{SWITCH}: decode buckets 1..={capacity} join {} Engram layers at their consumers",
            tables.len()
        ));
        Ok(())
    }

    /// Releases the prepared gathers, newest first: the host fences their
    /// transfers before any allocation they reference is released.
    pub(super) fn release_ar_engram(&mut self) -> Result<(), BundleAbiError> {
        if let Some(state) = &mut self.ar_engram {
            while let Some(layer) = state.layers.last() {
                self.services.release_mapped_gather(layer.gather)?;
                state.layers.pop();
            }
        }
        Ok(())
    }

    /// Whether decode phase `graph` joins its Engram layers here.
    pub(super) fn ar_engram_serves(&self, graph: Option<(u32, u32)>) -> bool {
        graph.is_some_and(|key| {
            self.ar_engram
                .as_ref()
                .is_some_and(|state| state.serves(key))
        })
    }

    /// Admits every uncaptured piece of `key` in `route`'s bank at once, or
    /// none, so a run captures all of its ranks' pieces or runs eagerly.
    pub(super) fn reserve_ar_engram(
        &self,
        (bucket, items): (u32, u32),
        route: IndexerRoute,
    ) -> Option<GraphReservation> {
        let missing = (0..self.ranks.len())
            .map(|rank| {
                self.indexer_graph_missing(
                    super::indexer_graphs::Bank::of(route),
                    GraphKey {
                        rank: rank as u32,
                        bucket,
                        items,
                    },
                )
            })
            .sum();
        self.graphs.reserve_retaining(missing)
    }

    /// Uploads the decode phase's ids and enqueues every layer's gather. The
    /// previous step's gathers completed at its result boundary, or at this
    /// step's start after a failed step (`settle_ar_engram`).
    pub(super) fn begin_ar_engram(&mut self, hashed: &Hashed) -> Result<(), BundleAbiError> {
        let services = &self.services;
        let producer = self.ranks[0].2;
        let state = self
            .ar_engram
            .as_mut()
            .filter(|state| state.idle())
            .ok_or(BundleAbiError::Status(BundleStatusV1::InvalidState))?;
        let bytes = layer_major(
            hashed,
            state.layers.len(),
            state.columns,
            state.capacity as usize,
        )?;
        upload_bytes(services, producer, state.indices, &bytes)?;
        state.rows = hashed.rows() as u32;
        state.sequence = state.sequence.wrapping_add(1);
        let rows = (hashed.rows() * state.columns) as u64;
        for layer in &state.layers {
            // Count before enqueueing: a refused enqueue may still hold
            // transfers, which release fences.
            state.enqueued += 1;
            services.enqueue_mapped_gather(layer.gather, rows)?;
        }
        Ok(())
    }

    /// Joins Engram layer `layer` (its index among the layers) on every
    /// rank's stream, before the piece holding its consumer.
    pub(crate) fn wait_ar_rows(&self, layer: usize) -> Result<(), BundleAbiError> {
        let state = self
            .ar_engram
            .as_ref()
            .filter(|state| state.enqueued == state.layers.len() && state.joined.get() == layer)
            .ok_or(BundleAbiError::Status(BundleStatusV1::InvalidState))?;
        let gather = state
            .layers
            .get(layer)
            .ok_or(BundleAbiError::InvalidRange("AR Engram layer"))?
            .gather;
        for &(_, _, stream) in &self.ranks {
            self.services.wait_mapped_gather(gather, stream)?;
        }
        state.joined.set(layer + 1);
        Ok(())
    }

    /// Completes every layer's gather once the phase's streams finished: a
    /// copy still running or a failed gather refuses the step before any
    /// result is published.
    pub(super) fn finish_ar_engram(&mut self) -> Result<(), BundleAbiError> {
        let Some(state) = self.ar_engram.as_mut().filter(|state| !state.idle()) else {
            return Ok(());
        };
        let mut layers = Vec::new();
        for layer in &state.layers[..state.enqueued] {
            let stats = self.services.complete_mapped_gather(layer.gather)?.ok_or(
                BundleAbiError::InvalidRange("AR Engram upload incomplete at the result boundary"),
            )?;
            layers.push(layer_trace(layer.layer, &stats));
        }
        state.enqueued = 0;
        state.joined.set(0);
        if state.trace {
            let message = step_trace(state.sequence, state.rows, &layers);
            let _ = self.services.log(message.as_bytes());
        }
        Ok(())
    }

    /// Completes the gathers a failed step left enqueued, before the next
    /// step writes any staging row: every rank's stream joins the layers it
    /// had not joined, and the host waits for those streams. A successful
    /// step completed its gathers already, so this waits on nothing then.
    pub(super) fn settle_ar_engram(&mut self) -> Result<(), BundleAbiError> {
        let Some(state) = self.ar_engram.as_mut().filter(|state| !state.idle()) else {
            return Ok(());
        };
        for layer in &state.layers[state.joined.get().min(state.enqueued)..state.enqueued] {
            for &(_, _, stream) in &self.ranks {
                self.services.wait_mapped_gather(layer.gather, stream)?;
            }
        }
        for &(_, _, stream) in &self.ranks {
            self.services.synchronize_stream(stream)?;
        }
        for layer in &state.layers[..state.enqueued] {
            self.services
                .complete_mapped_gather(layer.gather)?
                .ok_or(BundleAbiError::Status(BundleStatusV1::InvalidState))?;
        }
        state.enqueued = 0;
        state.joined.set(0);
        Ok(())
    }
}

/// The instance's id span and Engram layers, when it prepared the gathers.
#[cfg(test)]
pub(crate) fn indices(handle: runtime_bundle_abi::BundleHandleV1) -> Option<(Span, Vec<u32>)> {
    use bundle_runtime::entry::Bundle;
    let mut result = None;
    Instance::instances().with(|table| {
        let status = bundle_runtime::table::with_instance(table, handle, |instance| {
            result = instance.ar_engram.as_ref().map(|state| {
                (
                    state.indices,
                    state.layers.iter().map(|layer| layer.layer).collect(),
                )
            });
            BundleStatusV1::Success
        });
        assert_eq!(status, BundleStatusV1::Success);
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram_rows::{EngramRows, Step};
    use model_deepseek41_contract::geometry::FLASH;
    use model_deepseek41_contract::own_ops::engram::HashTables;

    /// Each layer's words are the synchronous gather's ids of that layer,
    /// row after row, and capacity rows past the live ones are zero.
    #[test]
    fn layer_major_ids_are_each_layers_synchronous_gather_ids() {
        let hash = HashTables::of(&FLASH).unwrap();
        let columns = hash.columns();
        let layers = FLASH.shape.engram.layers.len();
        let rows = EngramRows::new(hash);
        for live in [1_u32, 2, 7, 8, 16, 31, 32] {
            let tokens: Vec<Vec<u32>> = (0..live)
                .map(|item| vec![item * 7919 % 129_280 + 3])
                .collect();
            let steps: Vec<Step<'_>> = tokens
                .iter()
                .enumerate()
                .map(|(item, tokens)| Step {
                    request: item as u64,
                    tokens,
                    text: None,
                })
                .collect();
            let hashed = rows.hash(&steps).unwrap();
            let bytes = layer_major(&hashed, layers, columns, 32).unwrap();
            assert_eq!(bytes.len(), layers * 32 * columns * 4);
            for layer in 0..layers {
                let wanted: Vec<u8> = hashed
                    .ids()
                    .chunks_exact(layers * columns)
                    .flat_map(|row| &row[layer * columns..(layer + 1) * columns])
                    .flat_map(|id| id.to_le_bytes())
                    .collect();
                let at = layer * 32 * columns * 4;
                assert_eq!(&bytes[at..at + wanted.len()], wanted);
                assert!(
                    bytes[at + wanted.len()..at + 32 * columns * 4]
                        .iter()
                        .all(|&byte| byte == 0)
                );
            }
        }
        let empty = rows.hash(&[]).unwrap();
        assert!(layer_major(&empty, layers, columns, 32).is_err());
        let tokens: Vec<Vec<u32>> = (0..33).map(|t| vec![t]).collect();
        let steps: Vec<Step<'_>> = tokens
            .iter()
            .enumerate()
            .map(|(item, tokens)| Step {
                request: item as u64,
                tokens,
                text: None,
            })
            .collect();
        let wide = rows.hash(&steps).unwrap();
        assert!(layer_major(&wide, layers, columns, 32).is_err());
        assert!(layer_major(&wide, layers, 0, 64).is_err());
    }
}
