//! An AR step's row and index status in one readback: every rank's row
//! status planes and rank 0's index sources, joined into one packet on rank
//! 0 after each rank's work, read back once and checked in the order and
//! with the messages of the separate reads (`check_index_status`, then
//! `check_row_status`). A one-row Flash TP4 step's separate reads
//! synchronize a stream 21 times and read four bytes back 16 times, each
//! read waiting for its copy; the packet waits once. Status bytes and the
//! steps they refuse are unchanged. Steps of more rows than a packet holds
//! keep the separate reads.
use runtime_bundle_abi::{BundleAbiError, BundleEventHandleV1, BundleStatusV1, DeviceBufferSpanV1};

use super::Run;
use crate::dspark::device::slice;
use crate::instance::Instance;

/// Whether AR steps read their status back in one packet; not until the
/// served pair has measured it.
pub(crate) const SERVED: bool = false;

/// Set to zero to keep the separate status reads.
pub(crate) const SWITCH: &str = "CHOPINFER_DEEPSEEK41_AR_STATUS_PACKET";

/// The live rows a packet holds: the decode batch.
pub(crate) const ROWS: u32 = model_deepseek41_contract::serving::DECODE_BATCH;

pub(crate) struct Packet {
    /// Rank 0's packet: the index sources, then each rank's two planes.
    buffer: DeviceBufferSpanV1,
    /// Each peer rank's work done, which rank 0's stream waits for.
    ready: Vec<BundleEventHandleV1>,
}

fn enabled() -> bool {
    #[cfg(test)]
    if let Some(forced) = IN_TESTS.get() {
        return forced;
    }
    SERVED && std::env::var(SWITCH).as_deref() != Ok("0")
}

#[cfg(test)]
thread_local! {
    /// Tests keep the separate reads unless they switch the packet on, so
    /// their pinned journals stay those reads' whatever [`SERVED`] says.
    static IN_TESTS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(Some(false)) };
}

/// Runs `run` with the packet switched on or off.
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

enum Part {
    Index(u32),
    Quantized(usize),
}

impl Instance {
    /// Allocates the packet on rank 0 and each peer rank's ready event,
    /// when the packet serves and the instance serves AR.
    pub(crate) fn prepare_ar_status(&mut self) -> Result<(), BundleAbiError> {
        if !enabled() || self.dspark.is_some() {
            return Ok(());
        }
        let sources = self.config.shape.geometry.shape.csa.index_sources.len() as u64;
        let planes = 2 * self.ranks.len() as u64;
        let bytes = (sources + planes) * u64::from(ROWS) * 4;
        let (_, pool, stream) = self.ranks[0];
        let buffer = bundle_runtime::resources::allocate_span(
            &self.services,
            pool,
            stream,
            bytes,
            &mut self.held,
        )?;
        let mut ready = Vec::with_capacity(self.ranks.len());
        for &(context, _, _) in self.ranks.iter().skip(1) {
            let event = self.services.create_event(context)?;
            self.held
                .push(bundle_runtime::resources::Held::Event(event));
            ready.push(event);
        }
        self.ar_status = Some(Packet { buffer, ready });
        self.log(&format!(
            "{SWITCH}: AR steps of 1..={ROWS} rows read {} status parts back at once",
            sources + planes
        ));
        Ok(())
    }

    /// Refuses `run`'s phase when an index selection or a quantizer flagged
    /// a live row, as `check_index_status` then `check_row_status` do: with
    /// the packet when it holds the phase's rows, else with those reads.
    pub(super) fn check_status(&self, run: &Run) -> Result<(), BundleAbiError> {
        let rows = run.inputs.live_tokens();
        match &self.ar_status {
            Some(packet) if rows > 0 && rows <= ROWS && self.dspark.is_none() => {
                self.check_packet(run, packet, rows)
            }
            _ => {
                self.check_index_status(run)?;
                self.check_row_status(run)
            }
        }
    }

    fn check_packet(&self, run: &Run, packet: &Packet, rows: u32) -> Result<(), BundleAbiError> {
        let host = |error: String| {
            self.log(&format!("status packet: {error}"));
            BundleAbiError::Status(BundleStatusV1::BundleError)
        };
        let (context, _, stream) = self.ranks[0];
        for (&ready, &(_, _, peer)) in packet.ready.iter().zip(self.ranks.iter().skip(1)) {
            self.services.record_event(ready, peer)?;
            self.services.wait_event(stream, ready)?;
        }
        let length = u64::from(rows) * 4;
        let mut parts = Vec::new();
        let mut at = 0;
        let sources = self.config.shape.geometry.shape.csa.index_sources;
        for (source, &layer) in sources.iter().enumerate() {
            let span =
                crate::buffers::index_status(self.buffers[0].scratch.index_status, source, rows);
            self.services.copy_device_to_device(
                stream,
                slice(packet.buffer, at, length).map_err(host)?,
                span,
            )?;
            parts.push(Part::Index(layer));
            at += length;
        }
        for (rank, buffers) in self.buffers.iter().enumerate() {
            for plane in [
                buffers.scratch.row_status,
                buffers.scratch.shared_row_status,
            ] {
                let span = crate::buffers::rows(plane, rows);
                let destination = slice(packet.buffer, at, length).map_err(host)?;
                if rank == 0 {
                    self.services
                        .copy_device_to_device(stream, destination, span)?;
                } else {
                    self.services.copy_peer_to_peer(
                        stream,
                        context,
                        destination,
                        self.ranks[rank].0,
                        span,
                    )?;
                }
                parts.push(Part::Quantized(rank));
                at += length;
            }
        }
        let mut bytes = vec![0_u8; at as usize];
        self.services.copy_device_to_host(
            stream,
            &mut bytes,
            slice(packet.buffer, 0, at).map_err(host)?,
        )?;
        for (part, words) in parts.iter().zip(bytes.chunks_exact(length as usize)) {
            let Some(row) = words.chunks_exact(4).position(|word| word != [0; 4]) else {
                continue;
            };
            let item = run.inputs.token_batch.get(row).copied().unwrap_or(-1);
            let request = usize::try_from(item)
                .ok()
                .and_then(|item| run.inputs.request_ids.get(item));
            self.log(&match part {
                Part::Index(layer) => format!(
                    "the index selection of layer {layer} met a non-finite score in row {row} \
                     (request {request:?}); the step is refused"
                ),
                Part::Quantized(rank) => format!(
                    "a quantized input on rank {rank} was invalid in row {row} \
                     (request {request:?}); the step is refused"
                ),
            });
            return Err(BundleAbiError::Status(BundleStatusV1::BundleError));
        }
        Ok(())
    }
}
