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

/// The live rows a packet holds: the decode batch.
pub(crate) const ROWS: u32 = model_deepseek41_contract::serving::DECODE_BATCH;

pub(crate) struct Packet {
    /// Rank 0's packet: the index sources, then each rank's two planes.
    buffer: DeviceBufferSpanV1,
    /// Each peer rank's work done, which rank 0's stream waits for.
    ready: Vec<BundleEventHandleV1>,
}

enum Part {
    Index(u32),
    Quantized(usize),
}

impl Instance {
    /// Binds the packet reserved before KV sizing and creates each peer
    /// rank's ready event when the packet serves AR.
    pub(crate) fn prepare_ar_status(&mut self) -> Result<(), BundleAbiError> {
        if self.dspark.is_some() {
            return Ok(());
        }
        let sources = self.config.shape.geometry.shape.csa.index_sources.len() as u64;
        let planes = 2 * self.ranks.len() as u64;
        let buffer = self
            .ar_status_buffer
            .ok_or(BundleAbiError::InvalidRange("AR reserved status packet"))?;
        let mut ready = Vec::with_capacity(self.ranks.len());
        for &(context, _, _) in self.ranks.iter().skip(1) {
            let event = self.services.create_event(context)?;
            self.held
                .push(bundle_runtime::resources::Held::Event(event));
            ready.push(event);
        }
        self.ar_status = Some(Packet { buffer, ready });
        self.log(&format!(
            "AR status packet: steps of 1..={ROWS} rows read {} status parts back at once",
            sources + planes
        ));
        Ok(())
    }

    /// Refuses `run`'s phase when an index selection or a quantizer flagged
    /// a live row, as `check_index_status` then `check_row_status` do: with
    /// the packet when it holds the phase's rows, else with those reads.
    pub(super) fn check_status(&self, run: &Run) -> Result<(), BundleAbiError> {
        let _range = bundle_runtime::profile_ranges::Range::new(
            &self.services,
            if run.phase.prefill() {
                "bundle.prefill.status"
            } else if run.inputs.verify {
                "bundle.verify.status"
            } else {
                "bundle.decode.status"
            },
        );
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
