//! Successor comparison controls and per-step dumps for a state-capture
//! build's GPU tests, read from the enabled capture directory
//! (`CHOPINFER_DSPARK_STATE_DIR`). A serving build has none of them.
//!
//! - `successors-off`: queue no successor, the inline arm of a comparison.
//! - `successor.json`: the acceptance a drafting verify step queues its
//!   successor from, in place of a device result: `{"version":1,
//!   "allocated":false, "items":[{"request":1, "accepted":2, "bonus":7}]}`,
//!   one entry an item in the step's order. `allocated` queues the
//!   five-head successor of an allocated step. Only a step whose items all
//!   draft decode rows reads it; such a step whose items it does not name
//!   fails.
//! - `successor-steps`: each drafted step writes `successor-step-{step}/`:
//!   every rank's saved status before each check of the step's execute,
//!   decode-only or mixed, appended in check order (`status-rank-R.bin`),
//!   its draft IDs then confidence (`drafts-rank-R.bin`), the drafter ring
//!   cells each item's context holds (`ring-rank-R.bin`), and `step.json`,
//!   with the status checks, those cells' positions, the computed heads and
//!   the successor outcome. A context holds, at each stage, the cells of the
//!   positions it projected within the last window, from its retained
//!   start: other cells hold an earlier occupant's bytes, which no draft
//!   reads. Every I/O error names its path.
use super::Instance;
use crate::dspark::device::slice;
use crate::dspark::execution::Input;
use crate::dspark::prefetch::Feedback;
use crate::dspark::results::SAMPLE_OFFSET;
use crate::dspark::state::{BLOCK_ROWS, Context, Kind};
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;
use runtime_bundle_abi::device_results::{
    BUNDLE_ACCEPTANCE_ALLOCATED_V1, BUNDLE_ACCEPTANCE_NO_FAILURE_V1, BUNDLE_ACCEPTANCE_REJECTED_V1,
    BundleAcceptanceInputV1, BundleAcceptanceResultV1, DeviceCycleItemV1, DeviceCycleMetadataV1,
};
use serde_json::{Value, json};
use std::io::Read;
use std::path::{Path, PathBuf};

/// The drafter ring cells a context holds on one rank, and each stage's
/// positions.
struct RingCells {
    spans: Vec<Span>,
    positions: Vec<(u64, u64)>,
}

/// The largest `successor.json` read.
const CONTROL_BYTES: u64 = 64 << 10;

/// `result`, its error naming `path`.
fn at<T>(path: &Path, result: std::io::Result<T>) -> Result<T, String> {
    result.map_err(|error| format!("{}: {error}", path.display()))
}

fn read_control(path: &Path) -> Result<Value, String> {
    let metadata = at(path, std::fs::symlink_metadata(path))?;
    if !metadata.is_file() || metadata.len() > CONTROL_BYTES {
        return Err(format!(
            "{}: successor control must be a regular file of at most 64 KiB",
            path.display()
        ));
    }
    let mut bytes = Vec::new();
    let read = at(path, std::fs::File::open(path))?
        .take(CONTROL_BYTES + 1)
        .read_to_end(&mut bytes);
    at(path, read)?;
    if bytes.len() as u64 > CONTROL_BYTES {
        return Err(format!(
            "{}: successor control grew past its bound",
            path.display()
        ));
    }
    serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

/// The feedback `control` names for the drafting decode step `inputs`.
fn feedback(control: &Value, inputs: &[Input], vocab: u32) -> Result<Feedback, String> {
    let object = control
        .as_object()
        .ok_or("successor feedback must be an object")?;
    let allocated = control["allocated"]
        .as_bool()
        .ok_or("successor feedback names no allocated flag")?;
    let items = control["items"]
        .as_array()
        .ok_or("successor feedback names no items")?;
    if object.len() != 3 || control["version"] != 1 || items.len() != inputs.len() {
        return Err("unsupported successor feedback or item count".into());
    }
    let epoch = 1;
    let mut metadata = DeviceCycleMetadataV1 {
        cycle_epoch: epoch,
        items: Vec::new(),
        termination_stop_ids: Vec::new(),
    };
    let mut results = Vec::new();
    let mut row = 0_u32;
    for (index, (item, input)) in items.iter().zip(inputs).enumerate() {
        let Kind::Decode { drafts } = input.rows.kind else {
            return Err("successor feedback needs a decode step".into());
        };
        let field = |name: &str| {
            item[name]
                .as_u64()
                .ok_or_else(|| format!("successor feedback item {index} names no {name}"))
        };
        let accepted = u32::try_from(field("accepted")?).map_err(|e| e.to_string())?;
        let bonus = u32::try_from(field("bonus")?).map_err(|e| e.to_string())?;
        if item.as_object().is_none_or(|item| item.len() != 3)
            || field("request")? != input.rows.request
            || !(1..=BLOCK_ROWS).contains(&drafts)
            || accepted > drafts
            || bonus >= vocab
        {
            return Err(format!(
                "successor feedback item {index} differs from the step's drafting item"
            ));
        }
        metadata.items.push(DeviceCycleItemV1 {
            request_id: input.rows.request,
            slot_generation: 1,
            acceptance: BundleAcceptanceInputV1 {
                cycle_epoch: epoch,
                remaining_tokens: 1 << 20,
                first_sample_row: row,
                draft_item_index: index as u32,
                draft_tokens: drafts,
                stop_offset: 0,
                stop_count: 0,
                reserved: 0,
            },
        });
        let mut flags = 0;
        if allocated {
            flags |= BUNDLE_ACCEPTANCE_ALLOCATED_V1;
        }
        if accepted < drafts {
            flags |= BUNDLE_ACCEPTANCE_REJECTED_V1;
        }
        let record = BundleAcceptanceResultV1 {
            cycle_epoch: epoch,
            accepted_drafts: accepted,
            emitted_rows: accepted + 1,
            flags,
            status: 0,
            failure_row: BUNDLE_ACCEPTANCE_NO_FAILURE_V1,
            verified_drafts: if allocated { drafts } else { 0 },
        };
        results.push((record, row + accepted, bonus));
        row += drafts + 1;
    }
    let mut packet = vec![0; SAMPLE_OFFSET as usize + row as usize * 256];
    for (index, (record, sample, bonus)) in results.into_iter().enumerate() {
        packet[index * 32..index * 32 + 32].copy_from_slice(&record.to_le_bytes());
        let at = SAMPLE_OFFSET as usize + sample as usize * 256;
        packet[at..at + 4].copy_from_slice(&bonus.to_le_bytes());
    }
    Ok(Feedback {
        allocated,
        metadata,
        inputs: inputs.to_vec(),
        packet,
    })
}

impl Instance {
    fn successor_control(&self, name: &str) -> Option<PathBuf> {
        self.active_dspark_state_capture()
            .map(|root| root.join(name))
            .filter(|path| path.is_file())
    }

    /// Whether the inline arm of a comparison runs: no successor queues.
    pub(super) fn successors_off(&self) -> bool {
        self.successor_control("successors-off").is_some()
    }

    /// The feedback `successor.json` names for this drafting decode step.
    pub(super) fn forced_successor_feedback(
        &self,
        inputs: &[Input],
    ) -> Result<Option<Feedback>, String> {
        let drafting = inputs
            .iter()
            .all(|input| matches!(input.rows.kind, Kind::Decode { drafts } if drafts > 0));
        let Some(path) = self
            .successor_control("successor.json")
            .filter(|_| drafting && !inputs.is_empty())
        else {
            return Ok(None);
        };
        let vocab = self.config.shape.geometry.shape.vocab;
        feedback(&read_control(&path)?, inputs, vocab).map(Some)
    }

    /// The step's dump directory, created, when `successor-steps` asks.
    fn successor_step_directory(&self) -> Result<Option<PathBuf>, String> {
        if self.successor_control("successor-steps").is_none() {
            return Ok(None);
        }
        let (Some(root), Some(device)) = (self.active_dspark_state_capture(), &self.dspark) else {
            return Ok(None);
        };
        let directory = root.join(format!("successor-step-{}", device.diagnostic.step_id));
        at(&directory, std::fs::create_dir_all(&directory))?;
        Ok(Some(directory))
    }

    /// Before a step's execute: a drafted step dumps each saved status
    /// check while it runs.
    pub(super) fn begin_successor_status(&self, inputs: &[Input]) {
        let step = match &self.dspark {
            Some(device)
                if self.successor_control("successor-steps").is_some()
                    && inputs.iter().any(|input| input.rows.kind.drafts() > 0) =>
            {
                device.diagnostic.step_id
            }
            _ => 0,
        };
        self.successor_status_step.set(step);
        self.successor_status_checks.set(0);
    }

    /// After a step's execute: later checks, a commit's among them, dump
    /// nothing.
    pub(super) fn end_successor_status(&self) {
        self.successor_status_step.set(0);
    }

    fn append(path: &Path, bytes: &[u8]) -> Result<(), String> {
        use std::io::Write as _;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path);
        at(path, at(path, file)?.write_all(bytes))
    }

    fn read_span(&self, rank: usize, span: Span, output: &mut Vec<u8>) -> Result<(), String> {
        if span.length == 0 {
            return Ok(());
        }
        let at = output.len();
        output.resize(at + span.length as usize, 0);
        self.services
            .copy_device_to_host(self.ranks[rank].2, &mut output[at..], span)
            .map_err(|error| error.to_string())
    }

    /// Before a drafted step's saved status is validated: each rank's bytes.
    pub(in crate::instance) fn dump_successor_status(&self) -> Result<(), String> {
        let Some(device) = &self.dspark else {
            return Ok(());
        };
        let step = device.diagnostic.step_id;
        if step == 0 || self.successor_status_step.get() != step {
            return Ok(());
        }
        let Some(directory) = self.successor_step_directory()? else {
            return Ok(());
        };
        for (rank, stored) in device.ranks.iter().enumerate() {
            let mut bytes = Vec::new();
            self.read_span(rank, stored.status.pending()?, &mut bytes)?;
            Self::append(&directory.join(format!("status-rank-{rank}.bin")), &bytes)?;
        }
        self.successor_status_checks
            .set(self.successor_status_checks.get() + 1);
        Ok(())
    }

    /// The drafter ring cells `context` holds on `rank`, in stage, plane
    /// and position order, and each stage's positions.
    fn context_ring_spans(&self, rank: usize, context: &Context) -> Result<RingCells, String> {
        let (index, buffer) = self
            .state
            .buffers
            .iter()
            .enumerate()
            .find(|(_, buffer)| buffer.name == "dspark.ring")
            .ok_or("DSpark ring is not declared")?;
        let window = u64::from(self.config.shape.geometry.shape.attention.window);
        let entries = buffer.entries(self.extent);
        let mut spans = Vec::new();
        let mut positions = Vec::new();
        for (stage, &end) in context.projected().iter().enumerate() {
            let start = end.saturating_sub(window).max(context.retained_start);
            positions.push((start, end));
            let mut plane_start = stage as u64 * buffer.layer_stride(entries);
            for &length in &buffer.planes {
                let cell = length / window;
                let base = plane_start + u64::from(context.slot) * length;
                let mut position = start;
                while position < end {
                    let rows = (end - position).min(window - position % window);
                    spans.push(slice(
                        self.buffers[rank].state.buffers[index].1,
                        base + position % window * cell,
                        rows * cell,
                    )?);
                    position += rows;
                }
                plane_start += entries * length;
            }
        }
        Ok(RingCells { spans, positions })
    }

    /// After a drafted step and its queue: each rank's draft outputs and
    /// ring planes, and the step's heads and successor outcome.
    pub(super) fn dump_successor_step(&self, inputs: &[Input]) -> Result<(), String> {
        let drafting = inputs
            .iter()
            .filter(|input| input.rows.kind.drafts() > 0)
            .count() as u64;
        if drafting == 0 {
            return Ok(());
        }
        let Some(directory) = self.successor_step_directory()? else {
            return Ok(());
        };
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let state = device
            .runtime
            .as_ref()
            .ok_or("DSpark lifecycle borrowed")?
            .state();
        let bytes = drafting * u64::from(BLOCK_ROWS) * 4;
        let mut rings = Vec::new();
        for (rank, stored) in device.ranks.iter().enumerate() {
            let mut drafts = Vec::new();
            for output in [stored.draft_ids, stored.confidence] {
                self.read_span(rank, slice(output, 0, bytes)?, &mut drafts)?;
            }
            let mut ring = Vec::new();
            for input in inputs {
                let Some(context) = state.context(input.rows.request) else {
                    continue;
                };
                let RingCells { spans, positions } = self.context_ring_spans(rank, context)?;
                for span in spans {
                    self.read_span(rank, span, &mut ring)?;
                }
                if rank == 0 {
                    rings.push(json!({"request": input.rows.request, "positions": positions}));
                }
            }
            for (name, bytes) in [("drafts", drafts), ("ring", ring)] {
                let path = directory.join(format!("{name}-rank-{rank}.bin"));
                at(&path, std::fs::write(&path, bytes))?;
            }
            // A step whose execute checked no status still lists its file.
            Self::append(&directory.join(format!("status-rank-{rank}.bin")), &[])?;
        }
        let items: Vec<_> = inputs
            .iter()
            .map(|input| {
                json!({"request": input.rows.request, "slot": input.rows.slot,
                    "start": input.rows.start, "kind": format!("{:?}", input.rows.kind)})
            })
            .collect();
        let record = json!({
            "schema": "chopinfer.deepseek41.successor-step.v1",
            "step": device.diagnostic.step_id, "items": items, "drafting": drafting,
            "computed_positions": device.computed_positions, "rings": rings,
            "status_checks": self.successor_status_checks.get(),
            "consumed": device.prefetch.outcome.get(),
            "queued": device.prefetch.plan.is_some(),
        });
        let path = directory.join("step.json");
        let bytes = serde_json::to_vec_pretty(&record).map_err(|error| error.to_string())?;
        at(&path, std::fs::write(&path, bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dspark::execution::Noise;
    use crate::dspark::state::Rows;

    fn decode(request: u64, drafts: u32) -> Input {
        Input {
            rows: Rows {
                request,
                slot: request as u32,
                start: 10,
                kind: Kind::Decode { drafts },
            },
            tokens: vec![7],
            noise: Noise::GREEDY,
        }
    }

    /// A forced feedback is the device result a drafting decode step of
    /// those items would report: it validates as one, and its records and
    /// bonus samples are the named ones.
    #[test]
    fn forced_feedback_reads_as_the_named_device_result() {
        let inputs = [decode(1, 3), decode(2, 5)];
        let control = json!({"version":1, "allocated":true, "items":[
            {"request":1, "accepted":2, "bonus":9}, {"request":2, "accepted":5, "bonus":11}]});
        let forced = feedback(&control, &inputs, 100).unwrap();
        crate::dspark::results::validate_metadata(&forced.metadata, &inputs, 100, 0).unwrap();
        assert!(forced.allocated);
        for (index, (accepted, sample, bonus)) in [(2_u32, 2_usize, 9_u32), (5, 4 + 5, 11)]
            .into_iter()
            .enumerate()
        {
            let record =
                BundleAcceptanceResultV1::from_le_bytes(&forced.packet[index * 32..][..32])
                    .unwrap();
            assert_eq!(record.accepted_drafts, accepted);
            assert_eq!(record.emitted_rows, accepted + 1);
            assert_ne!(record.flags & BUNDLE_ACCEPTANCE_ALLOCATED_V1, 0);
            let at = SAMPLE_OFFSET as usize + sample * 256;
            assert_eq!(forced.packet[at..at + 4], bonus.to_le_bytes());
        }
        assert_eq!(forced.packet.len(), SAMPLE_OFFSET as usize + (4 + 6) * 256);
        for bad in [
            json!({"version":2, "allocated":true, "items":control["items"]}),
            json!({"version":1, "allocated":true, "items":[control["items"][0]]}),
            json!({"version":1, "allocated":true, "items":[
                {"request":2, "accepted":2, "bonus":9}, control["items"][1]]}),
            json!({"version":1, "allocated":true, "items":[
                {"request":1, "accepted":4, "bonus":9}, control["items"][1]]}),
            json!({"version":1, "allocated":true, "items":[
                {"request":1, "accepted":2, "bonus":100}, control["items"][1]]}),
            json!({"version":1, "items":control["items"]}),
            json!({"version":1, "allocated":true, "items":control["items"], "extra":1}),
        ] {
            assert!(feedback(&bad, &inputs, 100).is_err(), "{bad}");
        }
        let undrafted = [decode(1, 3), decode(2, 0)];
        assert!(feedback(&control, &undrafted, 100).is_err());
    }
}
