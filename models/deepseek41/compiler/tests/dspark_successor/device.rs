//! Installed-package multi-item fixture over the state-capture controls.
//! Shared host APIs own every device call.
use super::*;
use runtime_bundle_sdk::{
    BatchAppendCopyV2, BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1,
};
use runtime_kv::{KvGeometry, KvPool, PreparedAppendId, SequenceHandle};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

type Backend = runtime_host::HostServicesV1Backend<runtime_host::cuda::CudaHostDeviceV1>;
/// Request pages; prefix snapshots take the pages above them.
const REQUEST_PAGES: usize = 256;
const DEVICE_PAGES: u64 = 768;
const QUOTA: usize = 512;
/// Drafting items and verify rows: C16 at depth five.
const SEQUENCES: u32 = 20;
const VERIFY_ROWS: u32 = 96;

/// One item of a step.
#[derive(Clone, Debug)]
pub enum Item {
    /// A prompt's last chunk of `tokens`.
    Prefill { tokens: Vec<u32> },
    /// A verify of `drafts` drafts after the input token `bonus`.
    Verify { bonus: u32, drafts: u32 },
}

/// A request's item of a step.
#[derive(Clone, Debug)]
pub struct Scheduled {
    pub request: u64,
    pub item: Item,
}

/// What a step returned to the host.
#[derive(Debug, Default, PartialEq)]
pub struct Capture {
    /// Each drafting request's draft tokens.
    pub drafts: BTreeMap<u64, Vec<u32>>,
    /// Every emitted F32 logits row, in item order.
    pub rows: Vec<Vec<u8>>,
}

/// Each verify item's greedy acceptance, as `successor.json` in the
/// capture directory `root` names it.
pub struct Greedy {
    pub root: PathBuf,
    pub allocated: bool,
}

struct Sink {
    backend: Rc<Backend>,
    vocabulary: u32,
    expected: usize,
    drafts: BTreeMap<u64, Vec<u32>>,
    rows: BTreeMap<u32, Vec<u8>>,
    /// (request, drafts, first row) of each item that emits rows.
    layout: Vec<(u64, u32, usize)>,
    greedy: Option<Greedy>,
    written: Result<()>,
}

impl Sink {
    /// With every row in, write the greedy acceptance the step's queue
    /// reads after its target: each item keeps its drafts while they are
    /// each row's argmax, and the next row's argmax is the bonus.
    fn write_greedy(&mut self) -> Result<()> {
        let Some(greedy) = &self.greedy else {
            return Ok(());
        };
        let mut items = Vec::new();
        for &(request, drafts, first) in &self.layout {
            let proposed = self
                .drafts
                .get(&request)
                .ok_or("verify item has no drafts")?;
            let mut accepted = 0;
            while accepted < drafts as usize
                && proposed.get(accepted)
                    == Some(&maximum(&self.rows[&((first + accepted) as u32)])?)
            {
                accepted += 1;
            }
            let bonus = maximum(&self.rows[&((first + accepted) as u32)])?;
            items.push((request, accepted as u32, bonus));
        }
        Ok(protocol::write_feedback(
            &greedy.root,
            greedy.allocated,
            &items,
        )?)
    }
}

impl LogitsRowSinkV1 for Sink {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(
            u64::from(VERIFY_ROWS) + u64::from(SEQUENCES),
            u64::from(self.vocabulary),
            (u64::from(VERIFY_ROWS) + u64::from(SEQUENCES)) * u64::from(self.vocabulary),
        )
    }
    fn emit_row(&mut self, _: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    fn emit_draft_tokens(&mut self, row: OwnedDraftTokensV1) -> HostServiceResultV1<()> {
        if !self.rows.is_empty() || self.drafts.insert(row.request_id, row.tokens).is_some() {
            return Err(BundleStatusV1::InvalidState);
        }
        Ok(())
    }
    fn emit_view(&mut self, view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        let raw = view.raw();
        if raw.element_type != BUNDLE_LOGITS_ELEMENT_V1_F32 || raw.live_width != self.vocabulary {
            return Err(BundleStatusV1::Unsupported);
        }
        let stream = self.backend.create_stream(raw.context, 0)?;
        let result = (|| {
            for row in 0..raw.row_count {
                let bytes = u64::from(raw.live_width) * 4;
                let offset = raw
                    .row_stride_elements
                    .checked_mul(u64::from(row))
                    .and_then(|n| n.checked_mul(4))
                    .ok_or(BundleStatusV1::InvalidArgument)?;
                if offset
                    .checked_add(bytes)
                    .is_none_or(|end| end > raw.buffer.length)
                {
                    return Err(BundleStatusV1::InvalidArgument);
                }
                let span =
                    DeviceBufferSpanV1::new(raw.buffer.buffer, raw.buffer.offset + offset, bytes);
                let mut data = vec![0; bytes as usize];
                self.backend.copy_device_to_host(stream, &mut data, span)?;
                self.backend.synchronize_stream(stream)?;
                if self.rows.insert(raw.first_item_index + row, data).is_some() {
                    return Err(BundleStatusV1::InvalidState);
                }
            }
            Ok(())
        })();
        let released = self.backend.release_stream(stream);
        result.and(released)?;
        if self.rows.len() == self.expected {
            self.written = self.write_greedy();
        }
        Ok(())
    }
}

fn service<T>(result: HostServiceResultV1<T>) -> Result<T> {
    result.map_err(|e| format!("host service: {e:?}").into())
}

/// The greedy token of an F32 logits row; ties take the smallest id.
pub fn maximum(row: &[u8]) -> Result<u32> {
    if row.is_empty() || !row.len().is_multiple_of(4) {
        return Err("invalid F32 logits row".into());
    }
    let mut best = (0, f32::NEG_INFINITY);
    for (token, word) in row.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes(word.try_into()?);
        if !value.is_finite() {
            return Err(format!("non-finite logit at token {token}").into());
        }
        if value > best.1 {
            best = (token as u32, value);
        }
    }
    Ok(best.0)
}

/// An open step: its prepared appends, and each verify item's handle.
struct Open {
    id: PreparedAppendId,
    verify: Vec<(u64, SequenceHandle)>,
}

pub struct Fixture {
    pub instance: LoadedBundleInstanceV1,
    backend: Rc<Backend>,
    pool: KvPool,
    vocabulary: u32,
    /// The capture directory the bundle reads (`CHOPINFER_DSPARK_STATE_DIR`).
    pub root: PathBuf,
    /// Each live request's sequence and state slot.
    requests: BTreeMap<u64, (SequenceHandle, u64)>,
    /// Preempted requests' sequences, pinned for prefix restores.
    preempted: BTreeMap<u64, SequenceHandle>,
    open: Option<Open>,
}

impl Fixture {
    pub fn new(config: &Config, options: &BTreeMap<String, String>) -> Result<Self> {
        let root = PathBuf::from(
            std::env::var_os("CHOPINFER_DSPARK_STATE_DIR")
                .ok_or("CHOPINFER_DSPARK_STATE_DIR must name an empty capture directory")?,
        );
        protocol::start(&root)?;
        let arguments = vec![
            "serve".into(),
            "--bundle".into(),
            config.bundle.to_string_lossy().into_owned(),
        ];
        let runtime_host::HostCommandV1::Serve {
            selection: runtime_host::preparation::ServeSelection::Bundle(selection),
            host,
            ..
        } = runtime_host::parse_host_command_v1(&arguments)?
        else {
            return Err("expected bundle selection".into());
        };
        runtime_host::direct::select_checkpoint_v1(&config.checkpoint)?;
        let loaded = runtime_host::load_selected_topology_bundle_v1(
            &selection,
            &host,
            "deepseek41-flash-tp4",
        )?;
        loaded.with_model_options(options)?;
        let spec = loaded
            .manifest()
            .speculation()
            .ok_or("package has no speculation")?;
        if !spec.bundle_drafter()
            || spec.drafter_kind() != Some("dspark")
            || spec.maximum_draft_tokens() != 5
        {
            return Err("package does not advertise DSpark's five-position block".into());
        }
        let vocabulary = model_deepseek41_contract::geometry::FLASH.shape.vocab;
        let mut speculation = BundleRawSpeculationV1::disabled();
        speculation.maximum_draft_tokens = 5;
        speculation.maximum_verify_tokens = VERIFY_ROWS;
        let (backend, instance) = loaded.create_instance_with_serving_options(
            runtime_host::cuda::CudaHostDeviceV1::new()?,
            "deepseek41-flash-tp4",
            vec![0, 1, 2, 3],
            Some(runtime_host::ServingKvPolicyV1 {
                sampling_vocabulary: Some(vocabulary as usize),
                maximum_context_tokens: Some(QUOTA),
                state_reserved_bytes: 0,
                utilization_millionths: 1_000_000,
                maximum_pages: Some(DEVICE_PAGES),
                ..Default::default()
            }),
            speculation,
            Some(SEQUENCES),
            None,
        )?;
        if backend.serving_kv_pages().is_none_or(|n| n < DEVICE_PAGES) {
            return Err("the fixture requires its reserved snapshot pages".into());
        }
        Ok(Self {
            backend,
            instance,
            vocabulary,
            pool: KvPool::new(KvGeometry::new(64, REQUEST_PAGES)?, SEQUENCES as usize)?,
            root,
            requests: BTreeMap::new(),
            preempted: BTreeMap::new(),
            open: None,
        })
    }

    /// A fresh request in state slot `slot`.
    pub fn admit(&mut self, request: u64, slot: u64) -> Result<()> {
        let handle = self.pool.create_sequence(QUOTA)?;
        if self.requests.insert(request, (handle, slot)).is_some() {
            return Err(format!("request {request} is already live").into());
        }
        Ok(())
    }

    /// Captures `request`'s first `tokens` tokens into the snapshot pages.
    pub fn capture(&mut self, request: u64, tokens: usize) -> Result<Vec<u64>> {
        let count = self.instance.describe_prefix_state()?;
        if REQUEST_PAGES as u64 + count > DEVICE_PAGES {
            return Err("snapshot exceeds reserved pages".into());
        }
        let pages = (REQUEST_PAGES as u64..REQUEST_PAGES as u64 + count).collect::<Vec<_>>();
        self.instance
            .capture_prefix_state(request, tokens as u64, &pages)?;
        Ok(pages)
    }

    /// Retires `request` from its state slot, keeping its KV pages for
    /// prefix restores.
    pub fn preempt(&mut self, request: u64) -> Result<()> {
        let (handle, _) = self
            .requests
            .remove(&request)
            .ok_or("preempted request is absent")?;
        self.instance.retire_request(request)?;
        self.preempted.insert(request, handle);
        Ok(())
    }

    /// Restores preempted `source`'s captured prefix `ids` for `request` in
    /// `slot`, sharing `source`'s KV pages.
    pub fn restore(
        &mut self,
        source: u64,
        request: u64,
        slot: u64,
        ids: &[u32],
        pages: &[u64],
    ) -> Result<()> {
        let handle = *self
            .preempted
            .get(&source)
            .ok_or("restore source is not preempted")?;
        let child = self.pool.fork(handle, QUOTA)?;
        self.instance
            .restore_prefix_state_in_slot(request, ids, pages, slot)?;
        self.requests.insert(request, (child, slot));
        Ok(())
    }

    /// Runs one step of `items`; `greedy` writes their greedy acceptance as
    /// `successor.json` before the step queues its successor.
    pub fn step(&mut self, items: &[Scheduled], greedy: Option<Greedy>) -> Result<Capture> {
        if self.open.is_some() {
            return Err("fixture step is already open".into());
        }
        let mut appends = Vec::new();
        let mut layout = Vec::new();
        let mut verify = Vec::new();
        let mut expected = 0;
        for scheduled in items {
            let (handle, _) = *self
                .requests
                .get(&scheduled.request)
                .ok_or("scheduled request is absent")?;
            let rows = match &scheduled.item {
                Item::Prefill { tokens } => {
                    expected += 1;
                    tokens.len()
                }
                Item::Verify { drafts, .. } => {
                    layout.push((scheduled.request, *drafts, expected));
                    verify.push((scheduled.request, handle));
                    expected += *drafts as usize + 1;
                    *drafts as usize + 1
                }
            };
            appends.push((handle, rows));
        }
        let plan = self.pool.prepare_append_batch(&appends)?;
        service(self.backend.prepare_serving_batch(&plan))?;
        let mut wire_items = Vec::new();
        let mut wire_appends = Vec::new();
        let mut slots = Vec::new();
        for (scheduled, append) in items.iter().zip(plan.appends()) {
            wire_items.push(match &scheduled.item {
                Item::Prefill { tokens } => BatchItemV1::new(
                    scheduled.request,
                    false,
                    tokens.len() as u32,
                    true,
                    tokens.clone(),
                ),
                Item::Verify { bonus, drafts } => {
                    BatchItemV1::verify(scheduled.request, drafts + 1, true, vec![*bonus])
                }
            });
            let copy = append.copy().map(|copy| {
                BatchAppendCopyV2::new(
                    copy.source().index() as u64,
                    copy.destination().index() as u64,
                    copy.initialized_tokens() as u64,
                )
            });
            wire_appends.push(BatchAppendV1::new(
                append.previous_tokens() as u64,
                append.additional_tokens() as u64,
                append.new_tokens() as u64,
                append
                    .block_table()
                    .iter()
                    .map(|page| page.index() as u64)
                    .collect(),
                copy,
            ));
            slots.push(self.requests[&scheduled.request].1);
        }
        let wire = BatchRequestV1::new(BatchWireVersionV1::V3, wire_items, wire_appends)
            .with_state_slots(slots)
            .encode()?;
        self.open = Some(Open {
            id: plan.id(),
            verify,
        });
        let mut sink = Sink {
            backend: Rc::clone(&self.backend),
            vocabulary: self.vocabulary,
            expected,
            drafts: BTreeMap::new(),
            rows: BTreeMap::new(),
            layout,
            greedy,
            written: Ok(()),
        };
        self.instance
            .execute(&wire, &mut sink)
            .map_err(|e| format!("execute: {e:?}, {:?}", self.instance.last_execute_refusal()))?;
        sink.written?;
        if sink.rows.keys().copied().collect::<Vec<_>>() != (0..expected as u32).collect::<Vec<_>>()
            || sink.drafts.len() != sink.layout.len()
        {
            return Err("fixture got incomplete or unordered outputs".into());
        }
        Ok(Capture {
            drafts: sink.drafts,
            rows: sink.rows.into_values().collect(),
        })
    }

    /// Commits the open step: each verify item keeps `kept` rows, every
    /// prefill all of its rows.
    pub fn commit(&mut self, kept: &BTreeMap<u64, u32>) -> Result<()> {
        let open = self.open.take().ok_or("fixture has no open step")?;
        let handles: Vec<_> = open
            .verify
            .iter()
            .map(|&(request, handle)| {
                Ok((
                    handle,
                    *kept.get(&request).ok_or("unkept verify item")? as usize,
                ))
            })
            .collect::<Result<_>>()?;
        let commit = self
            .pool
            .preflight_commit_prepared_kept(open.id, &handles, &[])?;
        if open.verify.is_empty() {
            self.instance.commit()?;
        } else {
            let records: Vec<_> = open
                .verify
                .iter()
                .map(|&(request, _)| BundleKeptRowsV1 {
                    struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
                    reserved: 0,
                    request_id: request,
                    kept_rows: kept[&request],
                    reserved_tail: 0,
                })
                .collect();
            self.instance.commit_accepted(&records)?;
        }
        self.pool.apply_prepared_commit(commit)?;
        Ok(())
    }

    /// Releases preempted `request`'s pinned sequence.
    pub fn release(&mut self, request: u64) -> Result<()> {
        let handle = self
            .preempted
            .remove(&request)
            .ok_or("released request is not preempted")?;
        self.pool.release(handle)?;
        Ok(())
    }

    /// Retires `request`, releasing its sequence.
    pub fn retire(&mut self, request: u64) -> Result<()> {
        let (handle, _) = self
            .requests
            .remove(&request)
            .ok_or("retired request is absent")?;
        self.instance.retire_request(request)?;
        self.pool.release(handle)?;
        Ok(())
    }

    /// Moves the one step dump the bundle wrote since the last call to
    /// `destination`, returning its `step.json`.
    pub fn take_dump(&self, destination: &Path) -> Result<Value> {
        Ok(protocol::take_dump(&self.root, destination)?)
    }

    pub fn finish(mut self) -> Result<()> {
        if self.open.is_some() {
            return Err("fixture ended with an open transaction".into());
        }
        for request in self.requests.keys().copied().collect::<Vec<_>>() {
            self.retire(request)?;
        }
        for (_, handle) in std::mem::take(&mut self.preempted) {
            self.pool.release(handle)?;
        }
        self.instance.destroy()?;
        if self.backend.live_resource_count() != 0 {
            return Err("device resources remain live".into());
        }
        Ok(())
    }
}
