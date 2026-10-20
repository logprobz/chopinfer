//! The single adapter from the engine's model transaction to the C ABI.
//!
//! The engine keeps its Rust `ModelExecutor` trait. Nothing exports it across
//! the boundary: this adapter encodes one prepared batch into bytes, runs the
//! bundle's `execute` callback, collects the rows the bundle emits, and maps the
//! engine's commit and rollback onto the ABI's `commit` and `cancel`.

mod device_results;
mod submission;
use device_results::{DeviceResults, cycle_metadata};
use runtime_bundle_abi::device_results::{
    BundleDeviceCompletionV1, BundleDeviceDraftsV1, DeviceCycleMetadataV1,
};
use runtime_engine::AcceptedPrefix;
pub use submission::PendingBundleExecutionV1;

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

pub use crate::sampling_dispatch::SamplingDispatch;
use crate::sampling_dispatch::gathered_ranges;
use runtime_bundle::BundleManifestV1;
use runtime_bundle_abi::device_results::{
    BundleDeviceSamplingRequestV1, BundleReadyDeviceViewV1, BundleReadyLogitsViewV1,
};
use runtime_bundle_abi::{
    BundleAbiError, BundleKeptRowsV1, BundleStatusV1, HostServiceResultV1, LoadedBundleInstanceV1,
    LogitsRowSinkV1, OwnedDraftTokensV1, OwnedLogitsRowV1, OwnedLogitsViewV1, ResultSinkLimitsV1,
};
use runtime_engine::profile::{PhaseTimer, StepPhase};
use runtime_engine::{
    DraftRow, DraftSource, ExecutorError, KeptRows, LogitsRow, ModelBatchItem, ModelExecutor,
    ModelTransaction, ModelTransactionState, RequestId, SampledRow,
};
use runtime_kv::{BlockId, PreparedAppendBatch};
use runtime_scheduler::BatchPhase;

/// Runtime override of a bundle's declared device-result default.
pub const DEVICE_RESULTS: &str = "CHOPINFER_DEVICE_RESULTS";

pub(crate) fn device_results_selected(manifest: &BundleManifestV1, preferred_kv: bool) -> bool {
    use runtime_bundle_abi::device_results::{
        DEVICE_RESULTS_CAPABILITY_V1, DEVICE_RESULTS_DEFAULT_CAPABILITY_V1,
        device_results_selected_v1,
    };
    let capabilities = manifest.resource_requirements().required_capabilities();
    let has = |name| {
        capabilities
            .iter()
            .any(|capability| capability.as_str() == name)
    };
    has(DEVICE_RESULTS_CAPABILITY_V1)
        && device_results_selected_v1(
            preferred_kv && has(DEVICE_RESULTS_DEFAULT_CAPABILITY_V1),
            std::env::var_os(DEVICE_RESULTS).as_deref(),
        )
}

/// The fixed prefix of one encoded batch request.
pub const BATCH_REQUEST_MAGIC_V1: &[u8; 12] = b"chopbatch-v1";

/// The fixed prefix of one capability-gated V2 batch request.
pub const BATCH_REQUEST_MAGIC_V2: &[u8; 12] = b"chopbatch-v2";
pub const BATCH_REQUEST_MAGIC_V3: &[u8; 12] = b"chopbatch-v3";

/// The prefix of a V3 payload followed by one state slot ID per item.
pub const BATCH_REQUEST_MAGIC_V4: &[u8; 12] = b"chopbatch-v4";

/// The prefix of a V3 or V4 record followed by each item's explicit positions.
pub const BATCH_REQUEST_MAGIC_V5: &[u8; 12] = b"chopbatch-v5";

/// The manifest capability that opts a bundle into V2 COW append records.
pub const BATCH_WIRE_V2_CAPABILITY_V1: &str = "chop.batch-wire.v2";
/// Compatibility name for the V2 prompt-row capability.
pub use runtime_bundle_abi::PROMPT_ROWS_CAPABILITY_V2 as PROMPT_ROWS_CAPABILITY_V1;
pub const SPECULATIVE_CAPABILITY_V1: &str = runtime_bundle_abi::SPECULATIVE_CAPABILITY_V1;
pub const MEDIA_INPUTS_CAPABILITY_V1: &str = "chop.media-inputs.v1";
pub const POSITION_AXES_CAPABILITY_V1: &str = runtime_bundle_abi::POSITION_AXES_CAPABILITY_V1;

/// The wire code for a scheduled prefill.
///
/// The engine's phase enum is a Rust type, so the wire carries its own declared
/// codes rather than any compiler-chosen discriminant.
pub const BATCH_PHASE_PREFILL_V1: u8 = 0;

/// The wire code for a scheduled decode.
pub const BATCH_PHASE_DECODE_V1: u8 = 1;
pub const BATCH_PHASE_VERIFY_V3: u8 = 2;

/// The record revision a verified bundle declared it can decode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchWireCapabilityV1 {
    V1,
    V2CopyOnWrite,
    V2PromptRows,
    V3Speculative,
    V3Media,
    V3MediaSpeculative,
}

impl BatchWireCapabilityV1 {
    const fn magic(self) -> &'static [u8; 12] {
        match self {
            Self::V1 => BATCH_REQUEST_MAGIC_V1,
            Self::V2CopyOnWrite | Self::V2PromptRows => BATCH_REQUEST_MAGIC_V2,
            Self::V3Speculative | Self::V3Media | Self::V3MediaSpeculative => {
                BATCH_REQUEST_MAGIC_V3
            }
        }
    }
}

/// Selects the wire revision from one verified manifest's requirements.
///
/// A manifest that declares anchor items takes a V3 wire, the only one that
/// carries the anchor flag, whether or not the host speculates.
#[must_use]
pub fn batch_wire_capability_v1(manifest: &BundleManifestV1) -> BatchWireCapabilityV1 {
    if manifest
        .resource_requirements()
        .required_capabilities()
        .iter()
        .any(|capability| capability.as_str() == MEDIA_INPUTS_CAPABILITY_V1)
    {
        BatchWireCapabilityV1::V3Media
    } else if manifest
        .batch_limits()
        .is_some_and(|limits| limits.anchor_items() > 0)
    {
        BatchWireCapabilityV1::V3Speculative
    } else if manifest
        .resource_requirements()
        .required_capabilities()
        .iter()
        .any(|capability| capability.as_str() == PROMPT_ROWS_CAPABILITY_V1)
    {
        BatchWireCapabilityV1::V2PromptRows
    } else if manifest.speculation().is_some()
        || manifest
            .resource_requirements()
            .required_capabilities()
            .iter()
            .any(|capability| capability.as_str() == BATCH_WIRE_V2_CAPABILITY_V1)
    {
        BatchWireCapabilityV1::V2CopyOnWrite
    } else {
        BatchWireCapabilityV1::V1
    }
}

/// Whether one verified manifest reads explicit 3-axis positions.
#[must_use]
pub fn position_axes_capability_v1(manifest: &BundleManifestV1) -> bool {
    manifest
        .resource_requirements()
        .required_capabilities()
        .iter()
        .any(|capability| capability.as_str() == POSITION_AXES_CAPABILITY_V1)
}

/// What the last finalized transaction did at the ABI boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TransactionOutcomeV1 {
    None,
    Committed,
    Cancelled,
    Failed(BundleAbiError),
}

/// One execution's collected rows, bounded before the bundle emits them.
///
/// The scheduled items are borrowed rather than copied: each carries its
/// request's whole token history for sampling, and a copy per step costs as
/// much as the engine's own.
struct CollectingSinkV1<'a> {
    limits: ResultSinkLimitsV1,
    rows: Vec<OwnedLogitsRowV1>,
    sampled: Vec<SampledRow>,
    prompt_view_rows: Vec<(usize, LogitsRow)>,
    expected: &'a [ModelBatchItem],
    sampled_indices: BTreeSet<usize>,
    drafts: BTreeMap<RequestId, Vec<u32>>,
    sampler: Option<Rc<dyn DeviceLogitsSamplerV1>>,
    /// Issue views without waiting and collect them after execute returns
    /// (`CHOPINFER_SAMPLING_DEFER_VIEWS`, on unless `0`), so views on
    /// different contexts sample concurrently.
    defer: bool,
    /// Issued views in emission order, with the rows each was issued for.
    pending: Vec<(OwnedLogitsViewV1, Vec<ModelBatchItem>)>,
    sampling_groups: BTreeMap<usize, SamplingGroup>,
    device: Option<DeviceResults>,
}

struct SamplingGroup {
    dispatch: SamplingDispatch,
    rows: Vec<Option<ModelBatchItem>>,
}

/// Whether the host defers view sampling; on unless set to `0`.
pub const DEFER_VIEWS: &str = "CHOPINFER_SAMPLING_DEFER_VIEWS";

fn defer_views_from_environment() -> bool {
    std::env::var_os(DEFER_VIEWS).is_none_or(|value| value != "0")
}

impl CollectingSinkV1<'_> {
    fn expand_rows(
        &self,
        expected: &[&ModelBatchItem],
        start: usize,
        end: usize,
    ) -> HostServiceResultV1<Vec<ModelBatchItem>> {
        let items = &expected[start..end];
        let mut expanded = Vec::with_capacity(items.len());
        for (index, item) in items.iter().enumerate() {
            if let Some(draft) = item.draft() {
                let drafts = match draft.source() {
                    DraftSource::Bundle => self
                        .drafts
                        .get(&item.request_id())
                        .ok_or_else(|| {
                            refused(
                                BundleStatusV1::InvalidState,
                                "a verify row before its bundle drafts",
                            )
                        })?
                        .as_slice(),
                    DraftSource::Host => item.input_tokens().get(1..).ok_or_else(|| {
                        refused(
                            BundleStatusV1::InvalidArgument,
                            "a host-drafted verify item without drafts",
                        )
                    })?,
                };
                let row = expected[..start + index]
                    .iter()
                    .rev()
                    .take_while(|other| other.request_id() == item.request_id())
                    .count();
                let work = item
                    .sampling()
                    .ok_or_else(|| {
                        refused(
                            BundleStatusV1::InvalidArgument,
                            "a verify row without sampling parameters",
                        )
                    })?
                    .for_verify_row(drafts, row);
                expanded.push((*item).clone().with_sampling(work));
            } else {
                expanded.push((*item).clone());
            }
        }
        Ok(expanded)
    }

    fn sample_partition(
        &mut self,
        sampler: &dyn DeviceLogitsSamplerV1,
        view: OwnedLogitsViewV1,
        expanded: Vec<ModelBatchItem>,
        dispatch: SamplingDispatch,
    ) -> HostServiceResultV1<()> {
        let raw = view.raw();
        let start = raw.first_item_index as usize;
        let end = start + expanded.len();
        let refs = expanded.iter().collect::<Vec<_>>();
        if self.defer {
            if self
                .pending
                .iter()
                .any(|(pending, _)| pending.raw().context == raw.context)
            {
                self.collect_pending()?;
            }
            if sampler.issue_view_with_dispatch(view, &refs, dispatch)? {
                self.sampled_indices.extend(start..end);
                self.pending.push((view, expanded));
                return Ok(());
            }
        }
        self.collect_pending()?;
        let sampled = sampler.sample_view_with_dispatch(view, &refs, dispatch)?;
        if sampled.len() != expanded.len()
            || sampled
                .iter()
                .zip(&expanded)
                .any(|(row, item)| row.request_id() != item.request_id())
        {
            return Err(BundleStatusV1::HostError);
        }
        self.sampled_indices.extend(start..end);
        self.sampled.extend(sampled);
        Ok(())
    }

    /// Collects every issued view in emission order. The bundle keeps an
    /// emitted view's buffer unchanged until its execute call returns and
    /// the host finalizes the transaction, and the next execute starts only
    /// after this.
    fn collect_pending(&mut self) -> HostServiceResultV1<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let sampler = self.sampler.clone().ok_or(BundleStatusV1::Unsupported)?;
        for (view, items) in std::mem::take(&mut self.pending) {
            let refs = items.iter().collect::<Vec<_>>();
            let sampled = sampler
                .collect_view(view, &refs)
                .map_err(|status| refused(status, "a view the device sampler did not collect"))?;
            if sampled.len() != items.len()
                || sampled
                    .iter()
                    .zip(&items)
                    .any(|(row, item)| row.request_id() != item.request_id())
            {
                return Err(refused(
                    BundleStatusV1::HostError,
                    "collected rows that do not match their view",
                ));
            }
            self.sampled.extend(sampled);
        }
        Ok(())
    }

    fn pending_rows(&self) -> usize {
        self.pending.iter().map(|(_, items)| items.len()).sum()
    }
}

/// Keep the device sampler's vocabulary-sized workspace independent of batch size.
use crate::sampling_workspace::SAMPLER_VIEW_ROWS;

fn sampling_subview(
    view: OwnedLogitsViewV1,
    first: usize,
    rows: usize,
) -> HostServiceResultV1<OwnedLogitsViewV1> {
    let mut raw = view.raw();
    let invalid = BundleStatusV1::InvalidArgument;
    if rows == 0
        || rows > SAMPLER_VIEW_ROWS
        || first.checked_add(rows).ok_or(invalid)? > raw.row_count as usize
    {
        return Err(invalid);
    }
    let bytes = match raw.element_type {
        runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32 => 4,
        runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_BF16 => 2,
        _ => return Err(invalid),
    };
    if raw.row_stride_elements < u64::from(raw.live_width) {
        return Err(invalid);
    }
    let stride = raw.row_stride_elements.checked_mul(bytes).ok_or(invalid)?;
    let offset = (first as u64).checked_mul(stride).ok_or(invalid)?;
    let length = (rows as u64 - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(u64::from(raw.live_width) * bytes))
        .ok_or(invalid)?;
    if offset.checked_add(length).ok_or(invalid)? > raw.buffer.length {
        return Err(invalid);
    }
    raw.buffer.offset = raw.buffer.offset.checked_add(offset).ok_or(invalid)?;
    raw.buffer.length = length;
    raw.first_item_index = raw
        .first_item_index
        .checked_add(first as u32)
        .ok_or(invalid)?;
    raw.row_count = rows as u32;
    Ok(OwnedLogitsViewV1::new(raw))
}

/// One ordered V2 logits-view row and the token it predicts, if known.
pub struct DevicePromptRowV1<'a> {
    pub item: &'a ModelBatchItem,
    pub target: Option<u32>,
    pub prompt: bool,
    pub output: bool,
}

/// Samples one borrowed device view while its bundle callback is live.
pub trait DeviceLogitsSamplerV1 {
    fn supports_device_results(&self) -> bool {
        false
    }
    /// Fences all private sampler/readback work for this owner before an
    /// abandoned bundle transaction can rewrite or release its storage.
    fn drain_device_results(
        &self,
        _context: runtime_bundle_abi::BundleContextHandleV1,
        _rank: u32,
    ) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    fn retain_device_drafts(
        &self,
        _drafts: BundleDeviceDraftsV1,
    ) -> HostServiceResultV1<Box<dyn std::any::Any>> {
        Err(BundleStatusV1::Unsupported)
    }
    fn cancel_device_readback(
        &self,
        _view: BundleReadyDeviceViewV1,
        _ticket: u64,
    ) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    fn issue_ready_logits(
        &self,
        _view: BundleReadyLogitsViewV1,
        _items: &[&ModelBatchItem],
        _dispatch: SamplingDispatch,
    ) -> HostServiceResultV1<u64> {
        Err(BundleStatusV1::Unsupported)
    }

    /// Enqueue the ordinary sampler into a caller-owned result span. The
    /// callback returns after submission and records the supplied done event.
    fn sample_device_logits(
        &self,
        _request: BundleDeviceSamplingRequestV1,
        _items: &[&ModelBatchItem],
        _dispatch: SamplingDispatch,
    ) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    fn enqueue_device_readback(&self, _view: BundleReadyDeviceViewV1) -> HostServiceResultV1<u64> {
        Err(BundleStatusV1::Unsupported)
    }

    fn collect_device_readback(
        &self,
        _view: BundleReadyDeviceViewV1,
        _ticket: u64,
    ) -> HostServiceResultV1<Vec<u8>> {
        Err(BundleStatusV1::Unsupported)
    }

    /// Classify a complete gathered group after any sampler wrapper transforms its rows.
    fn sampling_dispatch(
        &self,
        items: &[&ModelBatchItem],
    ) -> HostServiceResultV1<SamplingDispatch> {
        SamplingDispatch::for_items(items)
    }

    /// Sample a partition with the original gathered view's dispatch.
    fn sample_view_with_dispatch(
        &self,
        view: OwnedLogitsViewV1,
        items: &[&ModelBatchItem],
        _dispatch: SamplingDispatch,
    ) -> HostServiceResultV1<Vec<SampledRow>> {
        self.sample_view(view, items)
    }

    fn issue_view_with_dispatch(
        &self,
        view: OwnedLogitsViewV1,
        items: &[&ModelBatchItem],
        _dispatch: SamplingDispatch,
    ) -> HostServiceResultV1<bool> {
        self.issue_view(view, items)
    }

    fn score_view_with_dispatch(
        &self,
        view: OwnedLogitsViewV1,
        rows: &[DevicePromptRowV1<'_>],
        _dispatch: SamplingDispatch,
    ) -> HostServiceResultV1<(Vec<SampledRow>, Vec<SampledRow>)> {
        self.score_view(view, rows)
    }

    fn sample_view(
        &self,
        view: OwnedLogitsViewV1,
        items: &[&ModelBatchItem],
    ) -> HostServiceResultV1<Vec<SampledRow>>;

    fn score_view(
        &self,
        _view: OwnedLogitsViewV1,
        _rows: &[DevicePromptRowV1<'_>],
    ) -> HostServiceResultV1<(Vec<SampledRow>, Vec<SampledRow>)> {
        Err(BundleStatusV1::Unsupported)
    }

    /// Starts sampling a view without waiting for it, so views on different
    /// contexts sample concurrently; `collect_view` on the same context
    /// returns the rows. Returns false, starting nothing, when this sampler
    /// cannot defer the view; the caller then uses `sample_view`.
    fn issue_view(
        &self,
        _view: OwnedLogitsViewV1,
        _items: &[&ModelBatchItem],
    ) -> HostServiceResultV1<bool> {
        Ok(false)
    }

    /// Returns the rows of the view `issue_view` started last on `view`'s
    /// context; `items` are the rows it received.
    fn collect_view(
        &self,
        _view: OwnedLogitsViewV1,
        _items: &[&ModelBatchItem],
    ) -> HostServiceResultV1<Vec<SampledRow>> {
        Err(BundleStatusV1::Unsupported)
    }
}

/// Both forms of one bundle execute result, still pending finalization.
pub struct BatchEmissionsV1 {
    rows: Vec<OwnedLogitsRowV1>,
    sampled: Vec<SampledRow>,
    prompt_view_rows: Vec<(usize, LogitsRow)>,
    drafts: Vec<OwnedDraftTokensV1>,
    accepted_prefixes: Vec<AcceptedPrefix>,
}

impl BatchEmissionsV1 {
    /// Builds a host-memory result for an executor that emits full rows and
    /// optional bundle-owned drafts. The adapter validates both at execution.
    pub fn from_rows_and_drafts(
        rows: Vec<OwnedLogitsRowV1>,
        drafts: Vec<OwnedDraftTokensV1>,
    ) -> Self {
        Self {
            rows,
            drafts,
            accepted_prefixes: Vec::new(),
            sampled: Vec::new(),
            prompt_view_rows: Vec::new(),
        }
    }
}

/// Reports why the result sink refused an emission and returns `status`.
///
/// The bundle sees only the status and may return it from execute as its own,
/// so this line is the one record of which host check refused the batch.
fn refused(status: BundleStatusV1, emission: &str) -> BundleStatusV1 {
    eprintln!(
        "bundle result sink refused {emission} with status {} ({})",
        status.0,
        status.name().unwrap_or("undefined")
    );
    status
}

fn view_outside(first: usize, rows: u32) -> BundleStatusV1 {
    let emission = format!(
        "a view of rows {first}..{} outside the batch's unsampled rows",
        first.saturating_add(rows as usize)
    );
    refused(BundleStatusV1::InvalidArgument, &emission)
}

impl LogitsRowSinkV1 for CollectingSinkV1<'_> {
    fn device_results_enabled(&self) -> bool {
        self.device.is_some()
    }
    fn device_cycle_metadata(&self) -> HostServiceResultV1<DeviceCycleMetadataV1> {
        self.device_metadata()
    }
    fn declare_input_dependency(&mut self, dependency: u32) -> HostServiceResultV1<()> {
        self.device_dependency(dependency)
    }
    fn emit_device_drafts(&mut self, drafts: BundleDeviceDraftsV1) -> HostServiceResultV1<()> {
        self.device_drafts(drafts)
    }
    fn sample_device_logits(
        &mut self,
        request: BundleDeviceSamplingRequestV1,
    ) -> HostServiceResultV1<()> {
        self.device_sample(request)
    }
    fn emit_ready_logits_view(&mut self, view: BundleReadyLogitsViewV1) -> HostServiceResultV1<()> {
        self.device_ready(view)
    }
    fn emit_device_completion(
        &mut self,
        completion: BundleDeviceCompletionV1,
    ) -> HostServiceResultV1<()> {
        self.device_completion(completion)
    }
    fn limits(&self) -> ResultSinkLimitsV1 {
        self.limits
    }

    fn emit_row(&mut self, row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        if (self.rows.len() + self.sampled_indices.len()) as u64 >= self.limits.maximum_rows {
            return Err(refused(
                BundleStatusV1::ResourceExhausted,
                "a logits row past the batch's rows",
            ));
        }
        self.rows.push(row);
        Ok(())
    }

    fn emit_draft_tokens(&mut self, drafts: OwnedDraftTokensV1) -> HostServiceResultV1<()> {
        let id = RequestId::new(drafts.request_id);
        let expected = self
            .expected
            .iter()
            .find(|item| item.request_id() == id)
            .and_then(ModelBatchItem::draft)
            .filter(|draft| draft.source() == DraftSource::Bundle)
            .ok_or_else(|| {
                refused(
                    BundleStatusV1::InvalidArgument,
                    "drafts for a request with no bundle-drafted verify item",
                )
            })?;
        if drafts.tokens.len() != expected.tokens() {
            return Err(refused(
                BundleStatusV1::InvalidArgument,
                "a draft count other than the verify item's",
            ));
        }
        if self.drafts.insert(id, drafts.tokens).is_some() {
            return Err(refused(
                BundleStatusV1::InvalidArgument,
                "a second set of drafts for one request",
            ));
        }
        Ok(())
    }

    fn emit_view(&mut self, view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        let sampler = self
            .sampler
            .clone()
            .ok_or_else(|| refused(BundleStatusV1::Unsupported, "a view without a sampler"))?;
        let raw = view.raw();
        if u64::from(raw.live_width) != self.limits.maximum_logits_per_row {
            return Err(refused(
                BundleStatusV1::InvalidArgument,
                "a view whose width is not the vocabulary",
            ));
        }
        if self
            .expected
            .iter()
            .any(|item| !item.prompt_targets().is_empty())
        {
            self.collect_pending()?;
            let mut owned = Vec::new();
            for item in self.expected {
                for (index, target) in item.prompt_targets().iter().copied().enumerate() {
                    owned.push((
                        item.clone(),
                        target,
                        true,
                        item.expects_logits() && index + 1 == item.prompt_targets().len(),
                    ));
                }
                if item.expects_logits() && item.prompt_targets().is_empty() {
                    let draft_tokens = match item.draft().map(|draft| draft.source()) {
                        Some(DraftSource::Bundle) => {
                            self.drafts.get(&item.request_id()).map(Vec::as_slice)
                        }
                        Some(DraftSource::Host) => {
                            Some(item.input_tokens().get(1..).ok_or_else(|| {
                                refused(
                                    BundleStatusV1::InvalidArgument,
                                    "a host-drafted verify item without drafts",
                                )
                            })?)
                        }
                        None => None,
                    };
                    for row in 0..item.draft().map_or(1, |draft| draft.tokens() + 1) {
                        let expanded = if let Some(drafts) = draft_tokens {
                            item.clone().with_sampling(
                                item.sampling()
                                    .ok_or_else(|| {
                                        refused(
                                            BundleStatusV1::InvalidArgument,
                                            "a verify row without sampling parameters",
                                        )
                                    })?
                                    .for_verify_row(drafts, row),
                            )
                        } else {
                            item.clone()
                        };
                        owned.push((expanded, None, false, true));
                    }
                }
            }
            let expected = owned
                .iter()
                .map(|(item, target, prompt, output)| DevicePromptRowV1 {
                    item,
                    target: *target,
                    prompt: *prompt,
                    output: *output,
                })
                .collect::<Vec<_>>();
            let start = raw.first_item_index as usize;
            let end = start
                .checked_add(raw.row_count as usize)
                .ok_or_else(|| view_outside(start, raw.row_count))?;
            let rows = expected
                .get(start..end)
                .ok_or_else(|| view_outside(start, raw.row_count))?;
            if rows.iter().any(|row| {
                row.item.draft().is_some_and(|draft| {
                    draft.source() == DraftSource::Bundle
                        && !self.drafts.contains_key(&row.item.request_id())
                })
            }) {
                return Err(refused(
                    BundleStatusV1::InvalidState,
                    "a verify row before its bundle drafts",
                ));
            }
            if (self.rows.len() + self.sampled_indices.len() + rows.len()) as u64
                > self.limits.maximum_rows
                || (start..end).any(|index| self.sampled_indices.contains(&index))
            {
                return Err(view_outside(start, raw.row_count));
            }
            let (scored, generated) = if rows.iter().all(|row| !row.prompt) {
                let mut scored = Vec::with_capacity(rows.len());
                let mut generated = Vec::new();
                for group in gathered_ranges(self.expected) {
                    let first = start.max(group.start);
                    let last = end.min(group.end);
                    if first >= last {
                        continue;
                    }
                    let dispatch = sampler.sampling_dispatch(
                        &expected[group]
                            .iter()
                            .map(|row| row.item)
                            .collect::<Vec<_>>(),
                    )?;
                    let part = if first == start && last == end {
                        view
                    } else {
                        sampling_subview(view, first - start, last - first)?
                    };
                    let (scores, outputs) =
                        sampler.score_view_with_dispatch(part, &expected[first..last], dispatch)?;
                    scored.extend(scores);
                    generated.extend(outputs);
                }
                (scored, generated)
            } else if rows.len() <= SAMPLER_VIEW_ROWS {
                sampler.score_view(view, rows).map_err(|status| {
                    refused(status, "a prompt-scored view the device sampler refused")
                })?
            } else {
                let mut scored = Vec::with_capacity(rows.len());
                let mut generated = Vec::new();
                for (index, chunk) in rows.chunks(SAMPLER_VIEW_ROWS).enumerate() {
                    let part = sampling_subview(view, index * SAMPLER_VIEW_ROWS, chunk.len())?;
                    let (scores, outputs) = sampler.score_view(part, chunk).map_err(|status| {
                        refused(status, "a prompt-scored view the device sampler refused")
                    })?;
                    scored.extend(scores);
                    generated.extend(outputs);
                }
                (scored, generated)
            };
            if scored.len() != rows.len()
                || scored.iter().zip(rows).any(|(score, row)| {
                    score.request_id() != row.item.request_id()
                        || row.target.is_some_and(|target| score.token_id() != target)
                })
            {
                return Err(BundleStatusV1::HostError);
            }
            for (index, (row, score)) in rows.iter().zip(scored).enumerate() {
                if row.prompt {
                    let scored_row = if let Some(target) = row.target {
                        LogitsRow::scored_prompt(
                            row.item.request_id(),
                            target,
                            score.logprob(),
                            score.top_logprobs().to_vec(),
                        )
                    } else {
                        LogitsRow::unscored_prompt(row.item.request_id())
                    };
                    self.prompt_view_rows.push((start + index, scored_row));
                }
                if !row.prompt {
                    self.sampled.push(score);
                }
            }
            self.sampled.extend(generated);
            self.sampled_indices.extend(start..end);
            return Ok(());
        }
        let expected = self
            .expected
            .iter()
            .filter(|item| item.expects_logits())
            .flat_map(|item| {
                std::iter::repeat_n(
                    item,
                    item.phase().appended_tokens().min(1)
                        + item.draft().map_or(0, |draft| draft.tokens()),
                )
            })
            .collect::<Vec<_>>();
        let start = raw.first_item_index as usize;
        let end = start
            .checked_add(raw.row_count as usize)
            .ok_or_else(|| view_outside(start, raw.row_count))?;
        let items = expected
            .get(start..end)
            .ok_or_else(|| view_outside(start, raw.row_count))?;
        if (self.rows.len() + self.sampled.len() + self.pending_rows() + items.len()) as u64
            > self.limits.maximum_rows
            || (start..end).any(|index| self.sampled_indices.contains(&index))
        {
            return Err(view_outside(start, raw.row_count));
        }
        for group in gathered_ranges(self.expected) {
            let first = start.max(group.start);
            let last = end.min(group.end);
            if first >= last {
                continue;
            }
            if !self.sampling_groups.contains_key(&group.start) {
                let expanded = self.expand_rows(&expected, group.start, group.end)?;
                let dispatch = sampler.sampling_dispatch(&expanded.iter().collect::<Vec<_>>())?;
                self.sampling_groups.insert(
                    group.start,
                    SamplingGroup {
                        dispatch,
                        rows: expanded.into_iter().map(Some).collect(),
                    },
                );
            }
            let cached = self
                .sampling_groups
                .get_mut(&group.start)
                .expect("gathered group exists");
            let dispatch = cached.dispatch;
            let part = if first == start && last == end {
                view
            } else {
                sampling_subview(view, first - start, last - first)?
            };
            let partition = cached.rows[first - group.start..last - group.start]
                .iter_mut()
                .map(|item| item.take().ok_or(BundleStatusV1::InvalidState))
                .collect::<Result<Vec<_>, _>>()?;
            self.sample_partition(sampler.as_ref(), part, partition, dispatch)?;
        }
        Ok(())
    }
}

/// The uncommitted bundle execution this adapter drives.
///
/// The adapter needs exactly these operations, and a loaded instance provides
/// them. Naming them here also lets a test drive every failure path between a
/// successful `execute` and its finalizer without a device.
pub trait BundleExecutionV1 {
    fn prefill_chunk_tokens(&self) -> Result<u32, BundleAbiError> {
        Ok(0)
    }

    fn supports_device_results(&self) -> bool {
        false
    }
    fn execute_with_device_sampling(
        &mut self,
        request: &[u8],
        limits: ResultSinkLimitsV1,
        items: &[ModelBatchItem],
        sampler: Option<Rc<dyn DeviceLogitsSamplerV1>>,
        _metadata: Option<DeviceCycleMetadataV1>,
    ) -> Result<BatchEmissionsV1, BundleAbiError> {
        self.execute_with_sampling(request, limits, items, sampler)
    }
    /// Submits one batch. The guard owns collection and cancellation until
    /// it returns the still-uncommitted emissions. Legacy implementations
    /// collect synchronously before returning a ready guard.
    fn submit_with_device_sampling<'a>(
        &'a mut self,
        request: &[u8],
        limits: ResultSinkLimitsV1,
        items: &'a [ModelBatchItem],
        sampler: Option<Rc<dyn DeviceLogitsSamplerV1>>,
        metadata: Option<DeviceCycleMetadataV1>,
    ) -> Result<PendingBundleExecutionV1<'a, Self>, BundleAbiError>
    where
        Self: Sized,
    {
        let rows = self.execute_with_device_sampling(request, limits, items, sampler, metadata)?;
        Ok(PendingBundleExecutionV1::ready(self, rows))
    }

    /// Evidence from the most recent failed execute: no host work or results,
    /// no pending transaction, and a usable instance. Unknown is not recoverable.
    fn last_execute_refusal(&self) -> Option<&str> {
        None
    }

    fn retire_request(&mut self, _request_id: u64) -> Result<(), BundleAbiError> {
        Ok(())
    }

    fn supports_request_retirement(&self) -> bool {
        false
    }

    fn supports_speculation(&self) -> bool {
        false
    }

    /// Snapshots a request's committed bundle state into serving-pool pages.
    fn capture_prefix_state(
        &mut self,
        _request_id: u64,
        _tokens: u64,
        _pages: &[u64],
    ) -> Result<(), BundleAbiError> {
        Err(BundleAbiError::MissingApiCallback(
            "capture_prefix_state_v1",
        ))
    }

    /// Creates an unseen request's bundle state from a snapshot.
    fn restore_prefix_state(
        &mut self,
        _request_id: u64,
        _token_ids: &[u32],
        _pages: &[u64],
    ) -> Result<(), BundleAbiError> {
        Err(BundleAbiError::MissingApiCallback(
            "restore_prefix_state_v1",
        ))
    }

    fn supports_prefix_state_slot(&self) -> bool {
        false
    }

    /// Creates an unseen request's bundle state from a snapshot, in the
    /// state slot the host assigned it.
    fn restore_prefix_state_in_slot(
        &mut self,
        _request_id: u64,
        _token_ids: &[u32],
        _pages: &[u64],
        _state_slot: u64,
    ) -> Result<(), BundleAbiError> {
        Err(BundleAbiError::MissingApiCallback(
            "restore_prefix_state_in_slot_v1",
        ))
    }

    /// Runs one encoded batch and returns the rows the bundle emitted.
    ///
    /// On success the execution is pending until `commit` or `cancel` runs. On
    /// failure it is not pending.
    fn execute_batch(
        &mut self,
        request: &[u8],
        limits: ResultSinkLimitsV1,
    ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError>;

    fn execute_with_sampling(
        &mut self,
        request: &[u8],
        limits: ResultSinkLimitsV1,
        _items: &[ModelBatchItem],
        _sampler: Option<Rc<dyn DeviceLogitsSamplerV1>>,
    ) -> Result<BatchEmissionsV1, BundleAbiError> {
        self.execute_batch(request, limits)
            .map(|rows| BatchEmissionsV1 {
                accepted_prefixes: Vec::new(),
                rows,
                sampled: Vec::new(),
                prompt_view_rows: Vec::new(),
                drafts: Vec::new(),
            })
    }

    fn commit(&mut self) -> Result<(), BundleAbiError>;

    fn commit_accepted(&mut self, _records: &[BundleKeptRowsV1]) -> Result<(), BundleAbiError> {
        Err(BundleAbiError::MissingApiCallback("commit_accepted_v1"))
    }

    fn cancel(&mut self) -> Result<(), BundleAbiError>;

    /// Releases the instance and every resource it borrowed.
    ///
    /// Only a caller that finalized its execution may destroy, and a destroyed
    /// instance runs nothing afterwards. The engine never reaches this: it
    /// finalizes transactions and lets the executor own the instance. A
    /// conformance run does, because giving everything back is the thing it is
    /// there to observe.
    fn destroy(&mut self) -> Result<(), BundleAbiError>;

    fn is_poisoned(&self) -> bool;
}

impl BundleExecutionV1 for LoadedBundleInstanceV1 {
    fn prefill_chunk_tokens(&self) -> Result<u32, BundleAbiError> {
        Self::prefill_policy(self).map(|policy| policy.chunk_tokens)
    }

    fn supports_device_results(&self) -> bool {
        Self::supports_device_results(self)
    }
    fn last_execute_refusal(&self) -> Option<&str> {
        Self::last_execute_refusal(self)
    }

    fn retire_request(&mut self, request_id: u64) -> Result<(), BundleAbiError> {
        Self::retire_request(self, request_id)
    }

    fn supports_request_retirement(&self) -> bool {
        Self::supports_request_retirement(self)
    }

    fn supports_speculation(&self) -> bool {
        Self::supports_speculation(self)
    }

    fn capture_prefix_state(
        &mut self,
        request_id: u64,
        tokens: u64,
        pages: &[u64],
    ) -> Result<(), BundleAbiError> {
        Self::capture_prefix_state(self, request_id, tokens, pages)
    }

    fn restore_prefix_state(
        &mut self,
        request_id: u64,
        token_ids: &[u32],
        pages: &[u64],
    ) -> Result<(), BundleAbiError> {
        Self::restore_prefix_state(self, request_id, token_ids, pages)
    }

    fn supports_prefix_state_slot(&self) -> bool {
        Self::supports_prefix_state_slot(self)
    }

    fn restore_prefix_state_in_slot(
        &mut self,
        request_id: u64,
        token_ids: &[u32],
        pages: &[u64],
        state_slot: u64,
    ) -> Result<(), BundleAbiError> {
        Self::restore_prefix_state_in_slot(self, request_id, token_ids, pages, state_slot)
    }

    fn execute_batch(
        &mut self,
        request: &[u8],
        limits: ResultSinkLimitsV1,
    ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
        let mut sink = CollectingSinkV1 {
            limits,
            rows: Vec::new(),
            sampled: Vec::new(),
            prompt_view_rows: Vec::new(),
            expected: &[],
            sampled_indices: BTreeSet::new(),
            drafts: BTreeMap::new(),
            sampler: None,
            defer: false,
            pending: Vec::new(),
            sampling_groups: BTreeMap::new(),
            device: None,
        };
        Self::execute(self, request, &mut sink)?;
        Ok(sink.rows)
    }

    fn execute_with_sampling(
        &mut self,
        request: &[u8],
        limits: ResultSinkLimitsV1,
        items: &[ModelBatchItem],
        sampler: Option<Rc<dyn DeviceLogitsSamplerV1>>,
    ) -> Result<BatchEmissionsV1, BundleAbiError> {
        self.execute_with_device_sampling(request, limits, items, sampler, None)
    }

    fn execute_with_device_sampling(
        &mut self,
        request: &[u8],
        limits: ResultSinkLimitsV1,
        items: &[ModelBatchItem],
        sampler: Option<Rc<dyn DeviceLogitsSamplerV1>>,
        metadata: Option<DeviceCycleMetadataV1>,
    ) -> Result<BatchEmissionsV1, BundleAbiError> {
        self.submit_with_device_sampling(request, limits, items, sampler, metadata)?
            .collect()
    }

    fn submit_with_device_sampling<'a>(
        &'a mut self,
        request: &[u8],
        limits: ResultSinkLimitsV1,
        items: &'a [ModelBatchItem],
        sampler: Option<Rc<dyn DeviceLogitsSamplerV1>>,
        metadata: Option<DeviceCycleMetadataV1>,
    ) -> Result<PendingBundleExecutionV1<'a, Self>, BundleAbiError> {
        let deferred = metadata.is_some();
        let device = metadata
            .map(|metadata| {
                DeviceResults::new(
                    metadata,
                    items,
                    sampler
                        .as_ref()
                        .ok_or(BundleStatusV1::Unsupported)?
                        .as_ref(),
                    limits,
                )
            })
            .transpose()
            .map_err(BundleAbiError::Status)?;
        let mut sink = CollectingSinkV1 {
            limits,
            rows: Vec::new(),
            sampled: Vec::new(),
            prompt_view_rows: Vec::new(),
            expected: items,
            sampled_indices: BTreeSet::new(),
            drafts: BTreeMap::new(),
            sampler,
            defer: defer_views_from_environment(),
            pending: Vec::new(),
            sampling_groups: BTreeMap::new(),
            device,
        };
        {
            let _phase = PhaseTimer::traced(StepPhase::BundleExecute);
            let executed = if request.starts_with(runtime_bundle_sdk::prefill::MAGIC) {
                if sink.device.is_some() {
                    return Err(BundleAbiError::Status(BundleStatusV1::Unsupported));
                }
                Self::execute_prefill(self, request, &mut sink)
            } else {
                Self::execute(self, request, &mut sink)
            };
            if let Err(error) = executed {
                // A failed callback may have issued private-stream consumers.
                // Fence them before dropping any borrowed result ownership.
                if sink.discard_device().is_err() {
                    self.poison_pending_execution();
                }
                return Err(error);
            }
        }
        if deferred {
            Ok(PendingBundleExecutionV1::submitted(
                self,
                sink,
                Self::poison_pending_execution,
            ))
        } else {
            // Keep legacy collection on the stack and at its original boundary.
            let _phase = PhaseTimer::traced(StepPhase::ResultCollect);
            let rows = match sink.finish_emissions() {
                Ok(rows) => rows,
                Err(status) => {
                    Self::cancel(self)?;
                    return Err(BundleAbiError::Status(status));
                }
            };
            Ok(PendingBundleExecutionV1::ready(self, rows))
        }
    }

    fn commit(&mut self) -> Result<(), BundleAbiError> {
        Self::commit(self)
    }

    fn commit_accepted(&mut self, records: &[BundleKeptRowsV1]) -> Result<(), BundleAbiError> {
        Self::commit_accepted(self, records)
    }

    fn cancel(&mut self) -> Result<(), BundleAbiError> {
        Self::cancel(self)
    }

    fn destroy(&mut self) -> Result<(), BundleAbiError> {
        Self::destroy(self)
    }

    fn is_poisoned(&self) -> bool {
        Self::is_poisoned(self)
    }
}

/// An optional export is negotiated independently of the legacy wire version.
/// Its prefill envelope keeps the existing result sink and execution schedule.
fn negotiated_prefill_request(
    instance: &impl BundleExecutionV1,
    request: Vec<u8>,
    items: &[ModelBatchItem],
) -> Result<Vec<u8>, ExecutorError> {
    if !items.iter().any(|item| item.prefill_boundary().is_some()) {
        return Ok(request);
    }
    let chunk = instance
        .prefill_chunk_tokens()
        .map_err(|e| executor_error_v1(format!("bundle prefill declaration failed: {e}")))?;
    if chunk == 0 {
        return Ok(request);
    }
    let batch = runtime_bundle_sdk::BatchCodecV1::decode(&request)
        .map_err(|e| executor_error_v1(e.to_string()))?;
    if batch.appends().len() != items.len() {
        return Err(executor_error_v1(
            "prefill append count mismatch".to_owned(),
        ));
    }
    let boundaries = items
        .iter()
        .zip(batch.appends())
        .filter_map(|(item, append)| {
            item.prefill_boundary().map(|contract| {
                if contract.chunk_tokens != chunk as usize {
                    return Err(executor_error_v1("prefill declaration changed".to_owned()));
                }
                let chunk_end = append
                    .previous_tokens()
                    .checked_add(u64::from(item.phase().appended_tokens() as u32))
                    .ok_or_else(|| executor_error_v1("prefill boundary overflow".to_owned()))?;
                Ok(runtime_bundle_sdk::prefill::Boundary {
                    request_id: item.request_id().get(),
                    chunk_tokens: chunk,
                    prompt_tokens: contract.prompt_tokens as u64,
                    chunk_end,
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    runtime_bundle_sdk::prefill::encode(&request, &boundaries)
        .map_err(|e| executor_error_v1(format!("invalid guaranteed prefill: {e}")))
}

/// The finalizer the engine drives for one uncommitted bundle execution.
pub struct BundleTransactionStateV1<'a, E: BundleExecutionV1 = LoadedBundleInstanceV1> {
    instance: Option<&'a mut E>,
    outcome: Rc<RefCell<TransactionOutcomeV1>>,
}

impl<E: BundleExecutionV1> BundleTransactionStateV1<'_, E> {
    fn finalize(&mut self, commit: bool, kept: Option<&[KeptRows]>) -> Result<(), ExecutorError> {
        let Some(instance) = self.instance.take() else {
            return Ok(());
        };
        let result = if let Some(kept) = kept {
            let records = kept
                .iter()
                .map(|entry| BundleKeptRowsV1 {
                    struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
                    reserved: 0,
                    request_id: entry.request_id.get(),
                    kept_rows: entry.kept_rows as u32,
                    reserved_tail: 0,
                })
                .collect::<Vec<_>>();
            instance.commit_accepted(&records)
        } else if commit {
            instance.commit()
        } else {
            instance.cancel()
        };
        let (outcome, result) = match result {
            Ok(()) if commit => (TransactionOutcomeV1::Committed, Ok(())),
            Ok(()) => (TransactionOutcomeV1::Cancelled, Ok(())),
            Err(error) => {
                let action = if commit { "commit" } else { "cancel" };
                let failure = executor_error_v1(format!("bundle {action} failed: {error}"));
                (TransactionOutcomeV1::Failed(error), Err(failure))
            }
        };
        *self.outcome.borrow_mut() = outcome;
        result
    }
}

impl<E: BundleExecutionV1> ModelTransactionState for BundleTransactionStateV1<'_, E> {
    fn commit(&mut self) -> Result<(), ExecutorError> {
        self.finalize(true, None)
    }

    fn commit_kept(&mut self, kept: &[KeptRows]) -> Result<(), ExecutorError> {
        self.finalize(true, Some(kept))
    }

    fn rollback(&mut self) -> Result<(), ExecutorError> {
        self.finalize(false, None)
    }
}

/// Runs one loaded bundle instance as the engine's model executor.
pub struct BundleModelExecutorV1<E: BundleExecutionV1 = LoadedBundleInstanceV1> {
    instance: E,
    vocabulary: usize,
    maximum_rows: u64,
    batch_wire: BatchWireCapabilityV1,
    outcome: Rc<RefCell<TransactionOutcomeV1>>,
    view_sampler: Option<Rc<dyn DeviceLogitsSamplerV1>>,
    row_buckets: Vec<usize>,
    step_buckets: Vec<usize>,
    state_slots: Option<crate::recurrent::StateSlotPoolV1>,
    in_place_state: bool,
    position_axes: bool,
    offload: Option<Rc<dyn crate::offload::KvOffloadExecutorV1>>,
    offload_info: Option<runtime_engine::KvOffloadInfo>,
}

impl<E: BundleExecutionV1> BundleModelExecutorV1<E> {
    /// Binds one loaded instance to the engine's vocabulary and batch bound.
    #[must_use]
    pub fn new(instance: E, vocabulary: usize, maximum_rows: u64) -> Self {
        Self::new_with_batch_wire(
            instance,
            vocabulary,
            maximum_rows,
            BatchWireCapabilityV1::V1,
        )
    }

    /// Binds one instance to the batch revision its verified manifest requires.
    #[must_use]
    pub fn new_with_batch_wire(
        instance: E,
        vocabulary: usize,
        maximum_rows: u64,
        batch_wire: BatchWireCapabilityV1,
    ) -> Self {
        Self {
            instance,
            vocabulary,
            maximum_rows,
            batch_wire,
            outcome: Rc::new(RefCell::new(TransactionOutcomeV1::None)),
            view_sampler: None,
            row_buckets: Vec::new(),
            step_buckets: Vec::new(),
            state_slots: None,
            in_place_state: false,
            position_axes: false,
            offload: None,
            offload_info: None,
        }
    }

    #[must_use]
    pub fn with_kv_offload(
        mut self,
        offload: Rc<dyn crate::offload::KvOffloadExecutorV1>,
        info: runtime_engine::KvOffloadInfo,
    ) -> Self {
        self.offload = Some(offload);
        self.offload_info = Some(info);
        self
    }

    /// Lets items with explicit positions through as V5 records. Only a
    /// bundle that declares `chop.position-axes.v1` may enable it.
    #[must_use]
    pub fn with_position_axes(mut self, enabled: bool) -> Self {
        self.position_axes = enabled;
        self
    }

    #[must_use]
    pub fn with_in_place_state(mut self, enabled: bool) -> Self {
        self.in_place_state = enabled;
        self
    }

    #[must_use]
    pub fn with_state_slots(mut self, pool: crate::recurrent::StateSlotPoolV1) -> Self {
        self.state_slots = Some(pool);
        self
    }

    #[must_use]
    pub fn with_view_sampler(mut self, sampler: Rc<dyn DeviceLogitsSamplerV1>) -> Self {
        self.view_sampler = Some(sampler);
        self
    }

    /// Debug teacher forcing: wraps the device sampler so rows that continue
    /// `reference` report its tokens. `None` when there is no device sampler.
    #[must_use]
    pub fn with_reference_forcing(
        mut self,
        reference: Rc<crate::reference_forcing::ReferenceTokensV1>,
    ) -> Option<Self> {
        let inner = self.view_sampler.take()?;
        self.view_sampler = Some(Rc::new(
            crate::reference_forcing::ReferenceForcingSamplerV1::new(inner, reference),
        ));
        Some(self)
    }

    #[must_use]
    pub fn with_row_buckets(mut self, buckets: &[u32]) -> Self {
        self.row_buckets = buckets.iter().map(|bucket| *bucket as usize).collect();
        self
    }

    /// The row counts the instance's captured decode and verify graphs run,
    /// in increasing order; a step pads to the smallest that holds it.
    #[must_use]
    pub fn with_step_buckets(mut self, buckets: &[usize]) -> Self {
        self.step_buckets = buckets.to_vec();
        self.step_buckets.sort_unstable();
        self
    }

    /// Decode and verify use the configured execution buckets, including eager
    /// replays. Prefill/mixed batches can run several programs with different
    /// capacities, and an unconfigured executor has no capacity to report.
    fn executed_row_capacity(&self, items: &[ModelBatchItem]) -> Option<usize> {
        if items.is_empty()
            || items
                .iter()
                .any(|item| matches!(item.phase(), BatchPhase::Prefill { .. }))
        {
            return None;
        }
        let rows: usize = items
            .iter()
            .map(|item| item.phase().appended_tokens())
            .sum();
        self.step_buckets
            .iter()
            .copied()
            .find(|&bucket| bucket >= rows)
    }

    /// Reports whether the loaded execution exposes the retirement callback.
    #[must_use]
    pub fn supports_request_retirement(&self) -> bool {
        self.instance.supports_request_retirement()
    }

    #[must_use]
    pub fn supports_speculation(&self) -> bool {
        self.instance.supports_speculation()
    }

    /// Reports how the last transaction ended at the ABI boundary.
    ///
    /// A failed commit or cancel is also returned to the engine, which stops
    /// executing. An execution this adapter cancelled itself, because it could
    /// not use the rows the bundle emitted, is recorded here the same way.
    #[must_use]
    pub fn last_outcome(&self) -> TransactionOutcomeV1 {
        self.outcome.borrow().clone()
    }

    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.instance.is_poisoned()
    }

    /// Cancels an execution whose rows this adapter refused, then reports why.
    ///
    /// `ModelExecutor` requires that a returned error leave no pending model
    /// state, and a successful `execute` leaves the instance pending. A failed
    /// cancel poisons the instance, and the returned error says so rather than
    /// letting the caller believe the bundle is still usable.
    fn cancel_refused_execution_v1(&mut self, refusal: ExecutorError) -> ExecutorError {
        match self.instance.cancel() {
            Ok(()) => {
                *self.outcome.borrow_mut() = TransactionOutcomeV1::Cancelled;
                refusal
            }
            Err(error) => {
                *self.outcome.borrow_mut() = TransactionOutcomeV1::Failed(error.clone());
                executor_error_v1(format!(
                    "{}; cancelling that execution also failed: {error}",
                    refusal.message()
                ))
            }
        }
    }
}

impl<E: BundleExecutionV1> ModelExecutor for BundleModelExecutorV1<E> {
    fn rollback_preserves_state_v1(&self) -> bool {
        !self.in_place_state
    }

    fn kv_transfer_drain_timeout_v1(&self) -> std::time::Duration {
        self.offload
            .as_ref()
            .map_or(std::time::Duration::from_secs(5), |q| q.drain_timeout())
    }

    fn kv_offload_info_v1(&self) -> Option<runtime_engine::KvOffloadInfo> {
        self.offload_info
            .filter(|_| self.offload.as_ref().is_some_and(|o| o.available()))
    }

    fn offload_pages_v1(
        &mut self,
        plan: runtime_kv::KvTransferPlan,
    ) -> runtime_engine::KvTransferSubmitResult {
        if plan.direction() != runtime_kv::KvTransferDirection::Out {
            return Err((
                executor_error_v1("wrong KV transfer direction".into()),
                plan,
            ));
        }
        let Some(offload) = &self.offload else {
            return Err((executor_error_v1("KV offload is unsupported".into()), plan));
        };
        offload.submit(plan).map_err(|(e, plan)| {
            (
                executor_error_v1(format!("KV copy submission failed: {e:?}")),
                plan,
            )
        })
    }
    fn reload_pages_v1(
        &mut self,
        plan: runtime_kv::KvTransferPlan,
    ) -> runtime_engine::KvTransferSubmitResult {
        if plan.direction() != runtime_kv::KvTransferDirection::In {
            return Err((
                executor_error_v1("wrong KV transfer direction".into()),
                plan,
            ));
        }
        let Some(offload) = &self.offload else {
            return Err((executor_error_v1("KV reload is unsupported".into()), plan));
        };
        offload.submit(plan).map_err(|(e, plan)| {
            (
                executor_error_v1(format!("KV copy submission failed: {e:?}")),
                plan,
            )
        })
    }
    fn poll_kv_transfer_v1(
        &mut self,
        id: runtime_kv::KvTransferId,
    ) -> Result<runtime_engine::KvTransferPoll, ExecutorError> {
        self.offload
            .as_ref()
            .ok_or_else(|| executor_error_v1("KV transfers are unsupported".into()))?
            .poll(id)
            .map_err(|e| executor_error_v1(format!("KV copy poll failed: {e:?}")))
    }
    fn wait_kv_transfer_v1(
        &mut self,
        id: runtime_kv::KvTransferId,
    ) -> Result<runtime_engine::KvTransferPoll, ExecutorError> {
        self.offload
            .as_ref()
            .ok_or_else(|| executor_error_v1("KV transfers are unsupported".into()))?
            .wait(id)
            .map_err(|e| executor_error_v1(format!("KV copy wait failed: {e:?}")))
    }
    fn drain_kv_transfers_v1(&mut self) -> Result<(), ExecutorError> {
        if let Some(offload) = &self.offload {
            offload.drain().map_err(|e| {
                ExecutorError::fatal(format!("KV copy drain failed: {e:?}")).expect("bounded error")
            })?;
        }
        Ok(())
    }

    fn round_draft_tokens_v1(&self, proposed: usize) -> usize {
        if self.row_buckets.is_empty() || proposed == 0 {
            return proposed;
        }
        self.row_buckets
            .iter()
            .copied()
            .filter(|rows| *rows <= proposed + 1)
            .max()
            .unwrap_or(1)
            .saturating_sub(1)
    }

    fn step_rows_v1(&self, rows: usize) -> usize {
        self.step_buckets
            .iter()
            .copied()
            .find(|&bucket| bucket >= rows)
            .unwrap_or(rows)
    }

    fn device_results_v1(&self) -> bool {
        self.instance.supports_device_results()
            && self
                .view_sampler
                .as_ref()
                .is_some_and(|sampler| sampler.supports_device_results())
    }

    fn admit_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
        if let Some(pool) = &mut self.state_slots {
            pool.allocate(id).map_err(executor_error_v1)?;
        }
        Ok(())
    }

    fn retire_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
        // A slot the bundle did not release stays owned, so no later request
        // can be handed bytes that another request may still reach.
        self.instance
            .retire_request(id.get())
            .map_err(|error| executor_error_v1(error.to_string()))?;
        if let Some(pool) = &mut self.state_slots {
            pool.retire(id);
        }
        Ok(())
    }

    fn capture_prefix_state_v1(
        &mut self,
        id: RequestId,
        tokens: usize,
        pages: &[BlockId],
    ) -> Result<(), ExecutorError> {
        let pages = pages
            .iter()
            .map(|page| page.index() as u64)
            .collect::<Vec<_>>();
        let result = self
            .instance
            .capture_prefix_state(id.get(), tokens as u64, &pages);
        self.prefix_state_result(result, "capture")
    }

    fn restore_prefix_state_v1(
        &mut self,
        id: RequestId,
        prefix: &[u32],
        pages: &[BlockId],
    ) -> Result<(), ExecutorError> {
        let pages = pages
            .iter()
            .map(|page| page.index() as u64)
            .collect::<Vec<_>>();
        // A bundle that keeps request state in slots restores into the one
        // admission assigned, which every batch of the request carries.
        let result = match &self.state_slots {
            Some(pool) if self.instance.supports_prefix_state_slot() => {
                let slot = pool.slot(id).ok_or_else(|| {
                    executor_error_v1("request has no admitted state slot".to_owned())
                })?;
                self.instance
                    .restore_prefix_state_in_slot(id.get(), prefix, &pages, slot)
            }
            _ => self.instance.restore_prefix_state(id.get(), prefix, &pages),
        };
        self.prefix_state_result(result, "restore")
    }

    type TransactionState<'a>
        = BundleTransactionStateV1<'a, E>
    where
        Self: 'a;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        let request = match (&self.state_slots, self.batch_wire) {
            // Slot IDs wrap a V3 payload, so request state implies V3 records.
            (Some(pool), _) => wrap_state_slots_v1(
                encode_batch_request_with_capability_v1(
                    items,
                    kv_plan,
                    BatchWireCapabilityV1::V3MediaSpeculative,
                )?,
                items,
                pool,
            )?,
            (None, BatchWireCapabilityV1::V1) => encode_batch_request_v1(items, kv_plan)?,
            (None, wire) => encode_batch_request_with_capability_v1(items, kv_plan, wire)?,
        };
        let request = wrap_position_axes_v1(request, items, self.position_axes)?;
        let expected = items
            .iter()
            .flat_map(|item| {
                let prompt = std::iter::repeat_n(item.request_id(), item.prompt_targets().len());
                let output = (item.expects_logits() && item.prompt_targets().is_empty())
                    .then(|| {
                        std::iter::repeat_n(
                            item.request_id(),
                            item.draft().map_or(1, |draft| draft.tokens() + 1),
                        )
                    })
                    .into_iter()
                    .flatten();
                prompt.chain(output)
            })
            .collect::<Vec<_>>();
        let maximum_logits_per_row = self.vocabulary as u64;
        let limits = ResultSinkLimitsV1::new(
            self.maximum_rows.max(expected.len() as u64),
            maximum_logits_per_row,
            self.maximum_rows
                .max(expected.len() as u64)
                .saturating_mul(maximum_logits_per_row),
        );
        // The V1 failure contract leaves no pending transaction. The loader
        // separately observes whether this call reached host work or results.
        #[allow(
            clippy::needless_late_init,
            reason = "declare before emitted buffers so their destruction is timed"
        )]
        let _convert_phase;
        let request = negotiated_prefill_request(&self.instance, request, items)?;
        // The negotiated-prefill ABI has no device-result sink. Preserve its
        // schedule and collection boundary; decode/verify retain device results.
        let metadata = (!request.starts_with(runtime_bundle_sdk::prefill::MAGIC)
            && self.device_results_v1())
        .then(|| cycle_metadata(items, self.state_slots.as_ref()))
        .flatten();
        let emitted = self
            .instance
            .execute_with_device_sampling(
                &request,
                limits,
                items,
                self.view_sampler.clone(),
                metadata,
            )
            .map_err(|error| {
                if error == BundleAbiError::Status(BundleStatusV1::InvalidArgument)
                    && !self.instance.is_poisoned()
                    && let Some(check) = self.instance.last_execute_refusal()
                {
                    return ExecutorError::batch_refused(bounded_executor_message_v1(format!(
                        "bundle batch refused: {check}; {error}; batch of {}",
                        batch_shape_v1(items)
                    )))
                    .expect("bounded nonempty refusal message");
                }
                executor_error_v1(format!(
                    "bundle execute failed: {error}; batch of {}",
                    batch_shape_v1(items)
                ))
            })?;
        _convert_phase = PhaseTimer::traced(StepPhase::ResultConvert);
        // From here the bundle is pending, so every rejection cancels first.
        // Each request's sampled rows are counted once rather than per row.
        let mut sampled_counts = BTreeMap::new();
        for sample in &emitted.sampled {
            *sampled_counts.entry(sample.request_id()).or_insert(0_usize) += 1;
        }
        let mut items_by_request: BTreeMap<RequestId, Vec<&ModelBatchItem>> = BTreeMap::new();
        for item in items {
            items_by_request
                .entry(item.request_id())
                .or_default()
                .push(item);
        }
        if sampled_counts.iter().any(|(id, count)| {
            !items_by_request.get(id).is_some_and(|items| {
                items.iter().any(|item| {
                    item.expects_logits()
                        && *count == item.draft().map_or(1, |draft| draft.tokens() + 1)
                })
            })
        }) {
            return Err(self.cancel_refused_execution_v1(executor_error_v1(
                "bundle returned an unexpected or duplicate sampled output".to_owned(),
            )));
        }
        let mut device_rows = BTreeMap::new();
        for (index, row) in emitted.prompt_view_rows {
            if index >= expected.len()
                || row.request_id() != expected[index]
                || device_rows.insert(index, row).is_some()
            {
                return Err(self.cancel_refused_execution_v1(executor_error_v1(
                    "bundle returned an unexpected device prompt row".to_owned(),
                )));
            }
        }
        let slots = items
            .iter()
            .flat_map(|item| {
                let prompt =
                    std::iter::repeat_n((item.request_id(), true), item.prompt_targets().len());
                let output = (item.expects_logits() && item.prompt_targets().is_empty())
                    .then(|| {
                        std::iter::repeat_n(
                            (item.request_id(), false),
                            item.draft().map_or(1, |draft| draft.tokens() + 1),
                        )
                    })
                    .into_iter()
                    .flatten();
                prompt.chain(output)
            })
            .collect::<Vec<_>>();
        let expected_host = slots
            .iter()
            .enumerate()
            .filter_map(|(index, (id, prompt))| {
                (!device_rows.contains_key(&index) && (*prompt || !sampled_counts.contains_key(id)))
                    .then_some(*id)
            })
            .collect::<Vec<_>>();
        let decoded = match decode_rows_v1(emitted.rows, &expected_host, self.vocabulary) {
            Ok(rows) => rows,
            Err(refusal) => return Err(self.cancel_refused_execution_v1(refusal)),
        };
        let mut decoded = decoded.into_iter();
        let mut prompt_rows = Vec::new();
        let mut rows = Vec::new();
        let mut slot = 0;
        for item in items {
            for _ in item.prompt_targets() {
                prompt_rows.push(
                    device_rows
                        .remove(&slot)
                        .unwrap_or_else(|| decoded.next().expect("validated row count")),
                );
                slot += 1;
            }
            if item.expects_logits() {
                if item.prompt_targets().is_empty() {
                    let sampled = sampled_counts.contains_key(&item.request_id());
                    for _ in 0..item.draft().map_or(1, |draft| draft.tokens() + 1) {
                        if !sampled {
                            rows.push(
                                device_rows.remove(&slot).unwrap_or_else(|| {
                                    decoded.next().expect("validated row count")
                                }),
                            );
                        }
                        slot += 1;
                    }
                } else if !sampled_counts.contains_key(&item.request_id()) {
                    rows.push(prompt_rows.last().expect("prefill has a final row").clone());
                }
            }
        }
        if !device_rows.is_empty() {
            return Err(self.cancel_refused_execution_v1(executor_error_v1(
                "unconsumed device prompt rows".to_owned(),
            )));
        }
        let mut drafts_by_request: BTreeMap<u64, Vec<&OwnedDraftTokensV1>> = BTreeMap::new();
        for row in &emitted.drafts {
            drafts_by_request
                .entry(row.request_id)
                .or_default()
                .push(row);
        }
        let mut drafts = Vec::new();
        for item in items {
            let emitted_for_item = drafts_by_request
                .get(&item.request_id().get())
                .map_or(&[][..], Vec::as_slice);
            match item.draft().map(|draft| draft.source()) {
                Some(DraftSource::Bundle)
                    if emitted_for_item.len() == 1
                        && emitted_for_item[0].tokens.len()
                            == item.draft().expect("verify draft").tokens() =>
                {
                    drafts.push(DraftRow::new(
                        item.request_id(),
                        emitted_for_item[0].tokens.clone(),
                    ));
                }
                Some(DraftSource::Bundle)
                    if emitted_for_item.is_empty()
                        && emitted
                            .accepted_prefixes
                            .iter()
                            .any(|prefix| prefix.request_id == item.request_id()) => {}
                Some(DraftSource::Bundle) => {
                    return Err(self.cancel_refused_execution_v1(executor_error_v1(
                        "bundle omitted or mis-sized verify drafts".to_owned(),
                    )));
                }
                Some(DraftSource::Host) if !emitted_for_item.is_empty() => {
                    return Err(self.cancel_refused_execution_v1(executor_error_v1(
                        "bundle returned unexpected drafts".to_owned(),
                    )));
                }
                _ => {}
            }
        }
        if emitted.drafts.len() != drafts.len() {
            return Err(self.cancel_refused_execution_v1(executor_error_v1(
                "bundle returned drafts for an unscheduled item".to_owned(),
            )));
        }
        let row_capacity = self.executed_row_capacity(items);
        *self.outcome.borrow_mut() = TransactionOutcomeV1::None;
        Ok(ModelTransaction::new(
            rows,
            BundleTransactionStateV1 {
                instance: Some(&mut self.instance),
                outcome: Rc::clone(&self.outcome),
            },
        )
        .with_prompt_rows(prompt_rows)
        .with_sampled(emitted.sampled)
        .with_accepted_prefixes(emitted.accepted_prefixes)
        .with_drafts(drafts)
        .with_row_capacity(row_capacity))
    }
}

impl<E: BundleExecutionV1> BundleModelExecutorV1<E> {
    /// A prefix-state failure that poisoned the instance may leave device
    /// work on the pages, so the engine must keep them: that one is fatal.
    fn prefix_state_result(
        &self,
        result: Result<(), BundleAbiError>,
        call: &str,
    ) -> Result<(), ExecutorError> {
        result.map_err(|error| {
            let message = format!("prefix state {call} failed: {error}");
            if self.instance.is_poisoned() {
                ExecutorError::fatal(message).unwrap_or_else(|_| {
                    ExecutorError::fatal("prefix state call failed on a poisoned bundle")
                        .expect("the fixed executor error message is valid")
                })
            } else {
                executor_error_v1(message)
            }
        })
    }
}

/// What a batch asked the bundle to run. Execute returns only a status, so
/// a failure names the batch in the model-neutral terms of its items.
fn batch_shape_v1(items: &[ModelBatchItem]) -> String {
    let (mut prefill_items, mut prefill_tokens, mut decode_items) = (0, 0, 0);
    let (mut verify_items, mut verify_rows, mut prompt_rows) = (0, 0, 0);
    let (mut encodes, mut encoded_rows, mut injected_rows) = (0, 0_u64, 0_u64);
    for item in items {
        match item.phase() {
            BatchPhase::Prefill { tokens } => {
                prefill_items += 1;
                prefill_tokens += tokens;
            }
            BatchPhase::Decode => decode_items += 1,
            BatchPhase::Verify { draft_tokens } => {
                verify_items += 1;
                verify_rows += draft_tokens + 1;
            }
        }
        prompt_rows += item.prompt_targets().len();
        encodes += item.media_encodes().len();
        encoded_rows += item
            .media_encodes()
            .iter()
            .map(|encode| u64::from(encode.output_rows))
            .sum::<u64>();
        injected_rows += item
            .media_injects()
            .iter()
            .map(|inject| u64::from(inject.rows))
            .sum::<u64>();
    }
    format!(
        "{} item(s): {prefill_items} prefill with {prefill_tokens} tokens and {prompt_rows} \
         scored prompt rows, {decode_items} decode, {verify_items} verify with {verify_rows} \
         rows, {encodes} media encodes with {encoded_rows} rows, {injected_rows} injected rows",
        items.len()
    )
}

fn bounded_executor_message_v1(mut message: String) -> String {
    let mut end = message.len().min(1024);
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message.truncate(end);
    message
}

fn executor_error_v1(message: String) -> ExecutorError {
    ExecutorError::new(message).unwrap_or_else(|_| {
        ExecutorError::new("bundle executor reported an unusable error message")
            .expect("the fixed executor error message is valid")
    })
}

/// Encodes one prepared batch in the host's own model-neutral wire form.
///
/// Nothing compiler-chosen crosses the boundary: every field is fixed width and
/// little endian, and the phase byte has its own declared codes.
pub(crate) fn encode_batch_request_v1(
    items: &[ModelBatchItem],
    kv_plan: &PreparedAppendBatch,
) -> Result<Vec<u8>, ExecutorError> {
    encode_batch_request_with_capability_v1(items, kv_plan, BatchWireCapabilityV1::V1)
}

/// Appends each item's admitted slot ID after the enclosed V3 payload.
fn wrap_state_slots_v1(
    payload: Vec<u8>,
    items: &[ModelBatchItem],
    pool: &crate::recurrent::StateSlotPoolV1,
) -> Result<Vec<u8>, ExecutorError> {
    let too_large = || executor_error_v1("state batch does not fit the wire".to_owned());
    let length = u32::try_from(payload.len()).map_err(|_| too_large())?;
    let count = u32::try_from(items.len()).map_err(|_| too_large())?;
    let mut request = BATCH_REQUEST_MAGIC_V4.to_vec();
    request.extend_from_slice(&length.to_le_bytes());
    request.extend_from_slice(&payload);
    request.extend_from_slice(&count.to_le_bytes());
    for item in items {
        let slot = pool
            .slot(item.request_id())
            .ok_or_else(|| executor_error_v1("request has no admitted state slot".to_owned()))?;
        request.extend_from_slice(&slot.to_le_bytes());
    }
    Ok(request)
}

/// Wraps a V3 or V4 record with each item's explicit positions: a row count,
/// zero or the item's phase rows, then that many (T, H, W) i32 triples. A
/// step with no explicit positions keeps its record unchanged.
fn wrap_position_axes_v1(
    payload: Vec<u8>,
    items: &[ModelBatchItem],
    enabled: bool,
) -> Result<Vec<u8>, ExecutorError> {
    if items.iter().all(|item| item.position_axes().is_empty()) {
        return Ok(payload);
    }
    if !enabled {
        return Err(executor_error_v1(format!(
            "explicit positions require the {POSITION_AXES_CAPABILITY_V1} capability"
        )));
    }
    if !payload.starts_with(BATCH_REQUEST_MAGIC_V3) && !payload.starts_with(BATCH_REQUEST_MAGIC_V4)
    {
        return Err(executor_error_v1(
            "explicit positions require batch wire V3".to_owned(),
        ));
    }
    let length = u32::try_from(payload.len())
        .map_err(|_| executor_error_v1("position batch does not fit the wire".to_owned()))?;
    let mut request = BATCH_REQUEST_MAGIC_V5.to_vec();
    request.extend_from_slice(&length.to_le_bytes());
    request.extend_from_slice(&payload);
    write_wire_count_v1(&mut request, items.len(), "position item")?;
    for item in items {
        let rows = match item.phase() {
            BatchPhase::Prefill { tokens } => tokens,
            BatchPhase::Decode => 1,
            BatchPhase::Verify { draft_tokens } => draft_tokens + 1,
        };
        let axes = item.position_axes();
        if !axes.is_empty() && axes.len() != rows {
            return Err(executor_error_v1(
                "explicit positions must cover every row of their item".to_owned(),
            ));
        }
        write_wire_count_v1(&mut request, axes.len(), "position row")?;
        for axis in axes.iter().flatten() {
            request.extend_from_slice(&axis.to_le_bytes());
        }
    }
    Ok(request)
}

pub(crate) fn encode_batch_request_with_capability_v1(
    items: &[ModelBatchItem],
    kv_plan: &PreparedAppendBatch,
    batch_wire: BatchWireCapabilityV1,
) -> Result<Vec<u8>, ExecutorError> {
    let mut request = Vec::new();
    request.extend_from_slice(batch_wire.magic());
    let item_count = u32::try_from(items.len())
        .map_err(|_| executor_error_v1("batch item count does not fit the wire".to_owned()))?;
    request.extend_from_slice(&item_count.to_le_bytes());
    for item in items {
        request.extend_from_slice(&item.request_id().get().to_le_bytes());
        let (phase, phase_tokens) = match item.phase() {
            BatchPhase::Prefill { tokens } => {
                let tokens = u32::try_from(tokens).map_err(|_| {
                    executor_error_v1("prefill token count does not fit the wire".to_owned())
                })?;
                (BATCH_PHASE_PREFILL_V1, tokens)
            }
            BatchPhase::Decode => (BATCH_PHASE_DECODE_V1, 1),
            BatchPhase::Verify { draft_tokens }
                if matches!(
                    batch_wire,
                    BatchWireCapabilityV1::V3Speculative
                        | BatchWireCapabilityV1::V3MediaSpeculative
                ) =>
            {
                let draft = item
                    .draft()
                    .ok_or_else(|| executor_error_v1("verify item has no draft work".to_owned()))?;
                let rows = draft_tokens
                    .checked_add(1)
                    .ok_or_else(|| executor_error_v1("verify row count overflowed".to_owned()))?;
                let expected_inputs = if draft.source() == DraftSource::Bundle {
                    1
                } else {
                    rows
                };
                if draft_tokens == 0
                    || draft.tokens() != draft_tokens
                    || !item.expects_logits()
                    || !item.prompt_targets().is_empty()
                    || item.input_tokens().len() != expected_inputs
                {
                    return Err(executor_error_v1("invalid verify item shape".to_owned()));
                }
                (
                    BATCH_PHASE_VERIFY_V3,
                    u32::try_from(rows).map_err(|_| {
                        executor_error_v1("verify row count does not fit the wire".to_owned())
                    })?,
                )
            }
            BatchPhase::Verify { .. } => {
                return Err(executor_error_v1(
                    "verify items require batch wire V3".to_owned(),
                ));
            }
        };
        request.push(phase);
        request.extend_from_slice(&phase_tokens.to_le_bytes());
        if !item.prompt_targets().is_empty()
            && !matches!(
                batch_wire,
                BatchWireCapabilityV1::V2PromptRows
                    | BatchWireCapabilityV1::V3Speculative
                    | BatchWireCapabilityV1::V3Media
                    | BatchWireCapabilityV1::V3MediaSpeculative
            )
        {
            return Err(executor_error_v1(
                "prompt rows require the chop.prompt-rows.v2 capability".to_owned(),
            ));
        }
        if item.anchor()
            && !matches!(
                batch_wire,
                BatchWireCapabilityV1::V3Speculative
                    | BatchWireCapabilityV1::V3Media
                    | BatchWireCapabilityV1::V3MediaSpeculative
            )
        {
            return Err(executor_error_v1(
                "anchor items require batch wire V3".to_owned(),
            ));
        }
        request.push(
            u8::from(item.expects_logits())
                | (u8::from(!item.prompt_targets().is_empty()) << 1)
                | (u8::from(
                    item.draft()
                        .is_some_and(|draft| draft.source() == DraftSource::Bundle),
                ) << 2)
                | (u8::from(item.anchor()) << 3),
        );
        let token_count = u32::try_from(item.input_tokens().len())
            .map_err(|_| executor_error_v1("input token count does not fit the wire".to_owned()))?;
        request.extend_from_slice(&token_count.to_le_bytes());
        for token in item.input_tokens() {
            request.extend_from_slice(&token.to_le_bytes());
        }
    }
    let append_count = u32::try_from(kv_plan.appends().len())
        .map_err(|_| executor_error_v1("KV append count does not fit the wire".to_owned()))?;
    request.extend_from_slice(&append_count.to_le_bytes());
    for append in kv_plan.appends() {
        request.extend_from_slice(&(append.previous_tokens() as u64).to_le_bytes());
        request.extend_from_slice(&(append.additional_tokens() as u64).to_le_bytes());
        request.extend_from_slice(&(append.new_tokens() as u64).to_le_bytes());
        let block_count = u32::try_from(append.block_table().len()).map_err(|_| {
            executor_error_v1("KV page table length does not fit the wire".to_owned())
        })?;
        request.extend_from_slice(&block_count.to_le_bytes());
        for block in append.block_table() {
            request.extend_from_slice(&(block.index() as u64).to_le_bytes());
        }
        match (batch_wire, append.copy()) {
            (BatchWireCapabilityV1::V1, Some(_)) => {
                return Err(executor_error_v1(
                    "a COW append requires the chop.batch-wire.v2 capability".to_owned(),
                ));
            }
            (BatchWireCapabilityV1::V1, None) => {}
            (
                BatchWireCapabilityV1::V2CopyOnWrite
                | BatchWireCapabilityV1::V2PromptRows
                | BatchWireCapabilityV1::V3Speculative
                | BatchWireCapabilityV1::V3Media
                | BatchWireCapabilityV1::V3MediaSpeculative,
                copy,
            ) => {
                request.push(u8::from(copy.is_some()));
                if let Some(copy) = copy {
                    request.extend_from_slice(&(copy.source().index() as u64).to_le_bytes());
                    request.extend_from_slice(&(copy.destination().index() as u64).to_le_bytes());
                    request.extend_from_slice(&(copy.initialized_tokens() as u64).to_le_bytes());
                }
            }
        }
    }
    if matches!(
        batch_wire,
        BatchWireCapabilityV1::V3Speculative
            | BatchWireCapabilityV1::V3Media
            | BatchWireCapabilityV1::V3MediaSpeculative
    ) {
        let mut payload = Vec::new();
        let encodes = items
            .iter()
            .flat_map(ModelBatchItem::media_encodes)
            .collect::<Vec<_>>();
        let injects = items
            .iter()
            .enumerate()
            .flat_map(|(index, item)| {
                item.media_injects()
                    .iter()
                    .map(move |inject| (index, inject))
            })
            .collect::<Vec<_>>();
        if !matches!(
            batch_wire,
            BatchWireCapabilityV1::V3Media | BatchWireCapabilityV1::V3MediaSpeculative
        ) && (!encodes.is_empty() || !injects.is_empty())
        {
            return Err(executor_error_v1(
                "media records require the chop.media-inputs.v1 capability".to_owned(),
            ));
        }
        write_wire_count_v1(&mut request, encodes.len(), "media encode")?;
        for encode in encodes {
            request.extend_from_slice(&u32::from(encode.encoder).to_le_bytes());
            for dimension in encode.geometry {
                request.extend_from_slice(&dimension.to_le_bytes());
            }
            request.extend_from_slice(&encode.output_rows.to_le_bytes());
            write_wire_count_v1(&mut request, encode.pages.len(), "media page")?;
            for page in &encode.pages {
                request.extend_from_slice(&page.to_le_bytes());
            }
            request.extend_from_slice(
                &u64::try_from(payload.len())
                    .map_err(|_| executor_error_v1("media payload offset overflowed".to_owned()))?
                    .to_le_bytes(),
            );
            request.extend_from_slice(
                &u64::try_from(encode.payload.len())
                    .map_err(|_| executor_error_v1("media payload length overflowed".to_owned()))?
                    .to_le_bytes(),
            );
            payload.extend_from_slice(&encode.payload);
        }
        write_wire_count_v1(&mut request, injects.len(), "media inject")?;
        for (index, inject) in injects {
            request.extend_from_slice(
                &u32::try_from(index)
                    .map_err(|_| executor_error_v1("media item index overflowed".to_owned()))?
                    .to_le_bytes(),
            );
            request.extend_from_slice(&inject.item_row.to_le_bytes());
            request.extend_from_slice(&inject.rows.to_le_bytes());
            request.extend_from_slice(&inject.arena_row.to_le_bytes());
        }
        request.extend_from_slice(
            &u64::try_from(payload.len())
                .map_err(|_| executor_error_v1("media payload length overflowed".to_owned()))?
                .to_le_bytes(),
        );
        request.extend_from_slice(&payload);
    } else if items
        .iter()
        .any(|item| !item.media_encodes().is_empty() || !item.media_injects().is_empty())
    {
        return Err(executor_error_v1(
            "media records require batch wire V3".to_owned(),
        ));
    }
    Ok(request)
}

fn write_wire_count_v1(out: &mut Vec<u8>, count: usize, name: &str) -> Result<(), ExecutorError> {
    if count > 1 << 20 {
        return Err(executor_error_v1(format!(
            "{name} count exceeds the wire limit"
        )));
    }
    out.extend_from_slice(
        &u32::try_from(count)
            .map_err(|_| executor_error_v1(format!("{name} count does not fit the wire")))?
            .to_le_bytes(),
    );
    Ok(())
}

fn decode_rows_v1(
    rows: Vec<OwnedLogitsRowV1>,
    expected: &[RequestId],
    vocabulary: usize,
) -> Result<Vec<LogitsRow>, ExecutorError> {
    if rows.len() != expected.len() {
        return Err(executor_error_v1(format!(
            "bundle returned {} logits rows for {} scheduled outputs",
            rows.len(),
            expected.len()
        )));
    }
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(rows.len())
        .map_err(|_| executor_error_v1("logits row allocation failed".to_owned()))?;
    for (row, request_id) in rows.into_iter().zip(expected) {
        if row.request_id() != request_id.get() {
            return Err(executor_error_v1(format!(
                "bundle returned a row for request {} where request {} was scheduled",
                row.request_id(),
                request_id.get()
            )));
        }
        let bits = row.logit_bits();
        if bits.len() != vocabulary * 4 {
            return Err(executor_error_v1(format!(
                "bundle returned {} logits for a vocabulary of {vocabulary}",
                bits.len() / 4
            )));
        }
        let mut logits = Vec::new();
        logits
            .try_reserve_exact(vocabulary)
            .map_err(|_| executor_error_v1("logits allocation failed".to_owned()))?;
        logits.extend(
            bits.chunks_exact(4)
                .map(|lane| f32::from_le_bytes([lane[0], lane[1], lane[2], lane[3]])),
        );
        decoded.push(
            LogitsRow::new(*request_id, logits)
                .map_err(|error| executor_error_v1(error.message().to_owned()))?,
        );
    }
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    mod negotiated_prefill;

    use std::collections::{BTreeMap, BTreeSet};
    use std::rc::Rc;

    use runtime_bundle_abi::{
        BundleAbiError, BundleLogitsViewV1, BundleStatusV1, HostServiceResultV1, LogitsRowSinkV1,
        OwnedDraftTokensV1, OwnedLogitsRowV1, OwnedLogitsViewV1, ResultSinkLimitsV1,
    };
    use runtime_engine::{
        DraftSource, DraftWork, ExecutorError, KeptRows, ModelBatchItem, ModelExecutor,
        ModelTransactionState, RequestId, SampledRow, SamplingWork,
    };
    use runtime_kv::{KvGeometry, KvPool, PreparedAppendBatch};
    use runtime_sampling::{SamplingParameters, SamplingState};
    use runtime_scheduler::BatchPhase;

    use super::{
        BatchEmissionsV1, BundleExecutionV1, BundleModelExecutorV1, BundleTransactionStateV1,
        CollectingSinkV1, DeviceLogitsSamplerV1, DevicePromptRowV1, SAMPLER_VIEW_ROWS,
        TransactionOutcomeV1, sampling_subview,
    };

    struct StubDeviceSampler;

    struct OracleDeviceSampler(Vec<f32>);

    impl DeviceLogitsSamplerV1 for OracleDeviceSampler {
        fn sample_view(
            &self,
            _view: OwnedLogitsViewV1,
            items: &[&ModelBatchItem],
        ) -> HostServiceResultV1<Vec<SampledRow>> {
            items
                .iter()
                .map(|item| {
                    let work = item.sampling().ok_or(BundleStatusV1::InvalidArgument)?;
                    let mut state = work.state;
                    let sampled = runtime_sampling::sample(
                        &work.parameters,
                        &self.0,
                        work.history.context(work.emitted_tokens, &work.stop_ids),
                        &mut state,
                    )
                    .map_err(|_| BundleStatusV1::HostError)?;
                    Ok(SampledRow::new(
                        item.request_id(),
                        sampled.token_id(),
                        sampled.logprob(),
                        sampled.top_logprobs().to_vec(),
                    ))
                })
                .collect()
        }
    }

    /// The CPU oracle on per-context logits, deferring like the device
    /// sampler: one issued view per context, collected with its rows.
    struct DeferringOracleSampler {
        logits: BTreeMap<u64, Vec<f32>>,
        pending: std::cell::RefCell<BTreeMap<u64, Vec<RequestId>>>,
        log: std::cell::RefCell<Vec<(&'static str, u64)>>,
    }

    impl DeferringOracleSampler {
        fn rows(&self, context: u64, items: &[&ModelBatchItem]) -> Vec<SampledRow> {
            items
                .iter()
                .map(|item| {
                    let work = item.sampling().unwrap();
                    let mut state = work.state;
                    let mut context_rows =
                        work.history.context(work.emitted_tokens, &work.stop_ids);
                    if let Some(mask) = work.constraint.as_ref().and_then(|c| c.mask()) {
                        context_rows = context_rows.with_allowed_tokens(mask);
                    }
                    let sampled = runtime_sampling::sample(
                        &work.parameters,
                        &self.logits[&context],
                        context_rows,
                        &mut state,
                    )
                    .unwrap();
                    SampledRow::new(
                        item.request_id(),
                        sampled.token_id(),
                        sampled.logprob(),
                        sampled.top_logprobs().to_vec(),
                    )
                })
                .collect()
        }
    }

    impl DeviceLogitsSamplerV1 for DeferringOracleSampler {
        fn sample_view(
            &self,
            view: OwnedLogitsViewV1,
            items: &[&ModelBatchItem],
        ) -> HostServiceResultV1<Vec<SampledRow>> {
            let context = view.raw().context.raw;
            self.log.borrow_mut().push(("sample", context));
            Ok(self.rows(context, items))
        }

        fn issue_view(
            &self,
            view: OwnedLogitsViewV1,
            items: &[&ModelBatchItem],
        ) -> HostServiceResultV1<bool> {
            let context = view.raw().context.raw;
            let ids = items.iter().map(|item| item.request_id()).collect();
            if self.pending.borrow_mut().insert(context, ids).is_some() {
                return Err(BundleStatusV1::InvalidState);
            }
            self.log.borrow_mut().push(("issue", context));
            Ok(true)
        }

        fn collect_view(
            &self,
            view: OwnedLogitsViewV1,
            items: &[&ModelBatchItem],
        ) -> HostServiceResultV1<Vec<SampledRow>> {
            let context = view.raw().context.raw;
            let ids = self
                .pending
                .borrow_mut()
                .remove(&context)
                .ok_or(BundleStatusV1::InvalidState)?;
            assert!(
                ids.into_iter()
                    .eq(items.iter().map(|item| item.request_id()))
            );
            self.log.borrow_mut().push(("collect", context));
            Ok(self.rows(context, items))
        }
    }

    #[test]
    fn large_views_are_sampled_in_bounded_ordered_slices() {
        struct Bounded(std::cell::RefCell<Vec<(u32, u32, u64)>>);
        impl DeviceLogitsSamplerV1 for Bounded {
            fn sample_view(
                &self,
                view: OwnedLogitsViewV1,
                items: &[&ModelBatchItem],
            ) -> HostServiceResultV1<Vec<SampledRow>> {
                let raw = view.raw();
                assert!(items.len() <= SAMPLER_VIEW_ROWS);
                assert_eq!(items.len(), raw.row_count as usize);
                self.0
                    .borrow_mut()
                    .push((raw.first_item_index, raw.row_count, raw.buffer.offset));
                Ok(items
                    .iter()
                    .map(|item| SampledRow::new(item.request_id(), 0, 0.0, Vec::new()))
                    .collect())
            }
        }
        for defer in [false, true] {
            let items: Vec<_> = (0..257)
                .map(|id| {
                    ModelBatchItem::new(RequestId::new(id + 1), BatchPhase::Decode, vec![0], true)
                        .with_sampling(super::sampling_tests::work(0.0))
                })
                .collect();
            let sampler = Rc::new(Bounded(std::cell::RefCell::new(Vec::new())));
            let mut sink = CollectingSinkV1 {
                limits: ResultSinkLimitsV1::new(257, 3, 771),
                rows: Vec::new(),
                sampled: Vec::new(),
                prompt_view_rows: Vec::new(),
                expected: &items,
                sampled_indices: BTreeSet::new(),
                drafts: BTreeMap::new(),
                sampler: Some(sampler.clone()),
                defer,
                pending: Vec::new(),
                sampling_groups: BTreeMap::new(),
                device: None,
            };
            let mut raw = BundleLogitsViewV1::empty();
            raw.element_type = runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32;
            raw.live_width = 3;
            raw.row_stride_elements = 4;
            raw.row_count = 257;
            raw.buffer.offset = 16;
            raw.buffer.length = 257 * 16;
            sink.emit_view(OwnedLogitsViewV1::new(raw)).unwrap();
            assert_eq!(
                *sampler.0.borrow(),
                vec![
                    (0, 64, 16),
                    (64, 64, 1040),
                    (128, 64, 2064),
                    (192, 64, 3088),
                    (256, 1, 4112)
                ]
            );
            assert!(
                sink.sampled
                    .iter()
                    .map(SampledRow::request_id)
                    .eq(items.iter().map(ModelBatchItem::request_id))
            );
            raw.buffer.length = 1;
            assert!(sampling_subview(OwnedLogitsViewV1::new(raw), 64, 64).is_err());
            raw.buffer.length = u64::MAX;
            raw.buffer.offset = u64::MAX;
            assert!(sampling_subview(OwnedLogitsViewV1::new(raw), 64, 64).is_err());
        }
    }

    #[test]
    fn deferred_views_sample_like_synchronous_views_in_emission_order() {
        let width = 32;
        let vocabulary = std::sync::Arc::new(
            runtime_grammar::TokenVocabulary::new(
                width,
                (0..width)
                    .map(|token| runtime_grammar::VocabularyToken {
                        bytes: if token + 1 == width {
                            Vec::new()
                        } else {
                            vec![b'a' + (token % 26) as u8]
                        },
                        special: false,
                    })
                    .collect(),
            )
            .unwrap(),
        );
        let grammar = std::sync::Arc::new(
            runtime_grammar::CompiledGrammar::new(
                &runtime_grammar::regex_grammar("d|j").unwrap(),
                vocabulary,
            )
            .unwrap(),
        );
        let constraint = runtime_engine::RowConstraint::new(runtime_grammar::ConstraintState::new(
            grammar,
            None,
            vec![width as u32 - 1],
        ));
        let parameters = [
            SamplingParameters::new(1.0)
                .unwrap()
                .with_top_p(0.95)
                .unwrap(),
            SamplingParameters::new(0.0)
                .unwrap()
                .with_top_logprobs(5)
                .unwrap(),
            SamplingParameters::new(0.8)
                .unwrap()
                .with_top_k(4)
                .unwrap()
                .with_top_logprobs(2)
                .unwrap(),
            SamplingParameters::new(1.0)
                .unwrap()
                .with_top_logprobs(1)
                .unwrap(),
            SamplingParameters::new(0.7)
                .unwrap()
                .with_penalties(0.2, 0.1, 1.1)
                .unwrap(),
        ];
        let items = parameters
            .into_iter()
            .enumerate()
            .map(|(index, parameters)| {
                ModelBatchItem::new(
                    RequestId::new(index as u64 + 1),
                    BatchPhase::Decode,
                    vec![1],
                    true,
                )
                .with_sampling(SamplingWork {
                    parameters,
                    state: SamplingState::new(index as u64 * 7 + 3),
                    history: vec![1, 2, 2].into(),
                    emitted_tokens: 1,
                    stop_ids: vec![width as u32 - 1],
                    constraint: (index == 3).then(|| constraint.clone()),
                })
            })
            .collect::<Vec<_>>();
        let logits = |context: u64| {
            (0..width)
                .map(|token| ((token as u64 * 13 + context * 5) % 23) as f32 / 4.0)
                .collect::<Vec<_>>()
        };
        let sampler = || {
            Rc::new(DeferringOracleSampler {
                logits: (1..=2).map(|context| (context, logits(context))).collect(),
                pending: std::cell::RefCell::new(BTreeMap::new()),
                log: std::cell::RefCell::new(Vec::new()),
            })
        };
        // Views of rows 0..2 and 4..5 on context 1, rows 2..4 on context 2.
        let views = [(1_u64, 0_u32, 2_u32), (2, 2, 2), (1, 4, 1)].map(|(context, first, rows)| {
            let mut raw = BundleLogitsViewV1::empty();
            raw.context = runtime_bundle_abi::BundleContextHandleV1::from_raw(context);
            raw.live_width = width as u32;
            raw.first_item_index = first;
            raw.row_count = rows;
            OwnedLogitsViewV1::new(raw)
        });
        let mut results = Vec::new();
        for defer in [false, true] {
            let sampler = sampler();
            let mut sink = CollectingSinkV1 {
                limits: ResultSinkLimitsV1::new(8, width as u64, 8 * width as u64),
                rows: Vec::new(),
                sampled: Vec::new(),
                prompt_view_rows: Vec::new(),
                expected: &items,
                sampled_indices: BTreeSet::new(),
                drafts: BTreeMap::new(),
                sampler: Some(sampler.clone()),
                defer,
                pending: Vec::new(),
                sampling_groups: BTreeMap::new(),
                device: None,
            };
            for view in views {
                sink.emit_view(view).unwrap();
            }
            // A repeated view is refused whether or not it was collected.
            assert_eq!(
                sink.emit_view(views[1]),
                Err(BundleStatusV1::InvalidArgument)
            );
            sink.collect_pending().unwrap();
            let log = sampler.log.borrow().clone();
            if defer {
                // Context 1's second view waits for both earlier views.
                assert_eq!(
                    log,
                    [
                        ("issue", 1),
                        ("issue", 2),
                        ("collect", 1),
                        ("collect", 2),
                        ("issue", 1),
                        ("collect", 1),
                    ]
                );
            } else {
                assert_eq!(log, [("sample", 1), ("sample", 2), ("sample", 1)]);
            }
            results.push(sink.sampled);
        }
        assert_eq!(results[0], results[1]);
        assert_eq!(
            results[0]
                .iter()
                .map(|row| row.request_id().get())
                .collect::<Vec<_>>(),
            [1, 2, 3, 4, 5]
        );
        assert_eq!(results[0][1].top_logprobs().len(), 5);
        // The constrained row picks a token its mask admits: "d" (3, 29) or "j".
        let token = results[0][3].token_id();
        let mask = constraint.mask().unwrap();
        assert_ne!(mask[token as usize / 32] & (1 << (token % 32)), 0);
        assert!(matches!(token, 3 | 9 | 29));
    }

    #[test]
    fn gathered_and_owned_verify_views_sample_bitwise_identically() {
        struct EncodedOracle(DeferringOracleSampler);
        impl EncodedOracle {
            fn encode(&self, items: &[&ModelBatchItem]) {
                #[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
                crate::sampling::assert_verify_metadata_for_test(items);
                #[cfg(not(all(target_os = "linux", feature = "linux-cuda-host")))]
                let _ = items;
            }
        }
        impl DeviceLogitsSamplerV1 for EncodedOracle {
            fn sample_view(
                &self,
                view: OwnedLogitsViewV1,
                items: &[&ModelBatchItem],
            ) -> HostServiceResultV1<Vec<SampledRow>> {
                self.encode(items);
                self.0.sample_view(view, items)
            }
            fn issue_view(
                &self,
                view: OwnedLogitsViewV1,
                items: &[&ModelBatchItem],
            ) -> HostServiceResultV1<bool> {
                self.encode(items);
                self.0.issue_view(view, items)
            }
            fn collect_view(
                &self,
                view: OwnedLogitsViewV1,
                items: &[&ModelBatchItem],
            ) -> HostServiceResultV1<Vec<SampledRow>> {
                self.0.collect_view(view, items)
            }
        }
        let width = 32;
        let vocabulary = std::sync::Arc::new(
            runtime_grammar::TokenVocabulary::new(
                width,
                (0..width)
                    .map(|token| runtime_grammar::VocabularyToken {
                        bytes: vec![b'a' + (token % 26) as u8],
                        special: false,
                    })
                    .collect(),
            )
            .unwrap(),
        );
        let grammar = std::sync::Arc::new(
            runtime_grammar::CompiledGrammar::new(
                &runtime_grammar::regex_grammar("[dj]+").unwrap(),
                vocabulary,
            )
            .unwrap(),
        );
        let logits = (0..width)
            .map(|token| ((token * 13) % 23) as f32 / 4.0)
            .collect::<Vec<_>>();
        for greedy in [false, true] {
            let items = [2, 5, 1]
                .into_iter()
                .enumerate()
                .map(|(index, drafts)| {
                    let parameters = SamplingParameters::new(if greedy { 0.0 } else { 0.8 })
                        .unwrap()
                        .with_top_p(0.95)
                        .unwrap()
                        .with_min_p(0.05)
                        .unwrap()
                        .with_logit_bias(vec![runtime_sampling::LogitBias::new(9, 0.5).unwrap()])
                        .unwrap()
                        .with_top_k(8)
                        .unwrap()
                        .with_penalties(0.2, 0.1, 1.1)
                        .unwrap()
                        .with_top_logprobs(5)
                        .unwrap();
                    ModelBatchItem::new(
                        RequestId::new(index as u64 + 1),
                        BatchPhase::Verify {
                            draft_tokens: drafts,
                        },
                        vec![3],
                        true,
                    )
                    .with_draft(DraftWork::new(drafts, DraftSource::Bundle))
                    .with_sampling(SamplingWork {
                        parameters,
                        state: SamplingState::new(71 + index as u64),
                        history: vec![3, 3, 9, 9].into(),
                        emitted_tokens: 2,
                        stop_ids: vec![31],
                        constraint: Some(runtime_engine::RowConstraint::new(
                            runtime_grammar::ConstraintState::new(grammar.clone(), None, vec![31]),
                        )),
                    })
                })
                .collect::<Vec<_>>();
            let mut results = Vec::new();
            for (owned, defer) in [(false, false), (true, false), (true, true)] {
                let sampler = Rc::new(EncodedOracle(DeferringOracleSampler {
                    logits: (1..=4).map(|rank| (rank, logits.clone())).collect(),
                    pending: Default::default(),
                    log: Default::default(),
                }));
                let mut sink = CollectingSinkV1 {
                    limits: ResultSinkLimitsV1::new(11, width as u64, 11 * width as u64),
                    rows: Vec::new(),
                    sampled: Vec::new(),
                    prompt_view_rows: Vec::new(),
                    expected: &items,
                    sampled_indices: BTreeSet::new(),
                    drafts: BTreeMap::new(),
                    sampler: Some(sampler.clone()),
                    defer,
                    pending: Vec::new(),
                    sampling_groups: BTreeMap::new(),
                    device: None,
                };
                for (index, drafts) in [2, 5, 1].into_iter().enumerate() {
                    sink.emit_draft_tokens(OwnedDraftTokensV1 {
                        request_id: index as u64 + 1,
                        tokens: vec![3; drafts],
                    })
                    .unwrap();
                }
                let shape = if owned {
                    vec![(1, 0, 4), (2, 4, 4), (3, 8, 3)]
                } else {
                    vec![(1, 0, 11)]
                };
                for (context, first, rows) in shape {
                    let mut raw = BundleLogitsViewV1::empty();
                    raw.context = runtime_bundle_abi::BundleContextHandleV1::from_raw(context);
                    raw.live_width = width as u32;
                    raw.first_item_index = first;
                    raw.row_count = rows;
                    sink.emit_view(OwnedLogitsViewV1::new(raw)).unwrap();
                }
                sink.collect_pending().unwrap();
                assert!(sampler.0.pending.borrow().is_empty());
                assert_eq!(sink.sampled.len(), 11);
                results.push(sink.sampled);
            }
            for actual in &results[1..] {
                for (expected, actual) in results[0].iter().zip(actual) {
                    assert_eq!(expected.request_id(), actual.request_id());
                    assert_eq!(expected.token_id(), actual.token_id());
                    assert_eq!(expected.logprob().to_bits(), actual.logprob().to_bits());
                    assert_eq!(actual.top_logprobs().len(), 5);
                    for (expected, actual) in
                        expected.top_logprobs().iter().zip(actual.top_logprobs())
                    {
                        assert_eq!(expected.token_id(), actual.token_id());
                        assert_eq!(expected.logprob().to_bits(), actual.logprob().to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn minimum_tokens_follows_gathered_and_owned_verify_rows() {
        for sequences in [1, 16] {
            for drafts in [0, 2] {
                let items = (0..sequences)
                    .map(|row| {
                        let emitted = row % 3;
                        let item = ModelBatchItem::new(
                            RequestId::new(row as u64 + 1),
                            if drafts == 0 {
                                BatchPhase::Decode
                            } else {
                                BatchPhase::Verify {
                                    draft_tokens: drafts,
                                }
                            },
                            vec![3],
                            true,
                        )
                        .with_sampling(SamplingWork {
                            parameters: SamplingParameters::new(0.0)
                                .unwrap()
                                .with_minimum_new_tokens(3)
                                .unwrap(),
                            state: SamplingState::new(0),
                            history: vec![3; emitted + 1].into(),
                            emitted_tokens: emitted,
                            stop_ids: vec![1, 2],
                            constraint: None,
                        });
                        if drafts == 0 {
                            item
                        } else {
                            item.with_draft(DraftWork::new(drafts, DraftSource::Bundle))
                        }
                    })
                    .collect::<Vec<_>>();
                let count = sequences * (drafts + 1);
                for (owned, defer) in [(false, false), (true, false), (true, true)] {
                    let sampler = Rc::new(DeferringOracleSampler {
                        logits: (1..=4)
                            .map(|rank| (rank, vec![0.0, 10.0, 9.0, 8.0]))
                            .collect(),
                        pending: Default::default(),
                        log: Default::default(),
                    });
                    let mut sink = CollectingSinkV1 {
                        limits: ResultSinkLimitsV1::new(count as u64, 4, (count * 4) as u64),
                        rows: Vec::new(),
                        sampled: Vec::new(),
                        prompt_view_rows: Vec::new(),
                        expected: &items,
                        sampled_indices: BTreeSet::new(),
                        drafts: BTreeMap::new(),
                        sampler: Some(sampler),
                        defer,
                        pending: Vec::new(),
                        sampling_groups: BTreeMap::new(),
                        device: None,
                    };
                    if drafts != 0 {
                        for row in 0..sequences {
                            sink.emit_draft_tokens(OwnedDraftTokensV1 {
                                request_id: row as u64 + 1,
                                tokens: vec![3; drafts],
                            })
                            .unwrap();
                        }
                    }
                    let chunk = if owned { count.div_ceil(4) } else { count };
                    for (rank, first) in (0..count).step_by(chunk).enumerate() {
                        let mut raw = BundleLogitsViewV1::empty();
                        raw.context =
                            runtime_bundle_abi::BundleContextHandleV1::from_raw(rank as u64 + 1);
                        raw.live_width = 4;
                        raw.first_item_index = first as u32;
                        raw.row_count = (count - first).min(chunk) as u32;
                        sink.emit_view(OwnedLogitsViewV1::new(raw)).unwrap();
                    }
                    sink.collect_pending().unwrap();
                    assert_eq!(sink.sampled.len(), count);
                    for (index, sample) in sink.sampled.iter().enumerate() {
                        let item = index / (drafts + 1);
                        let position = index % (drafts + 1);
                        assert_eq!(sample.request_id(), items[item].request_id());
                        assert_eq!(
                            sample.token_id(),
                            if item % 3 + position < 3 { 3 } else { 1 }
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bundle_drafts_precede_views_and_every_verify_row_matches_cpu_sampling() {
        let work = SamplingWork {
            parameters: SamplingParameters::new(0.7)
                .unwrap()
                .with_penalties(0.2, 0.1, 1.1)
                .unwrap(),
            state: SamplingState::new(19),
            history: vec![2, 2, 3].into(),
            emitted_tokens: 1,
            stop_ids: vec![15],
            constraint: None,
        };
        let item = ModelBatchItem::new(
            RequestId::new(7),
            BatchPhase::Verify { draft_tokens: 2 },
            vec![3],
            true,
        )
        .with_draft(DraftWork::new(2, DraftSource::Bundle))
        .with_sampling(work.clone());
        let logits = (0..16)
            .map(|token| (token as f32 - 7.0) / 3.0)
            .collect::<Vec<_>>();
        let mut sink = CollectingSinkV1 {
            limits: ResultSinkLimitsV1::new(3, 16, 48),
            rows: Vec::new(),
            sampled: Vec::new(),
            prompt_view_rows: Vec::new(),
            expected: &[item],
            sampled_indices: BTreeSet::new(),
            drafts: BTreeMap::new(),
            sampler: Some(Rc::new(OracleDeviceSampler(logits.clone()))),
            defer: false,
            pending: Vec::new(),
            sampling_groups: BTreeMap::new(),
            device: None,
        };
        let mut raw = BundleLogitsViewV1::empty();
        raw.live_width = 16;
        raw.row_count = 3;
        assert_eq!(
            sink.emit_view(OwnedLogitsViewV1::new(raw)),
            Err(BundleStatusV1::InvalidState)
        );
        sink.emit_draft_tokens(OwnedDraftTokensV1 {
            request_id: 7,
            tokens: vec![4, 5],
        })
        .unwrap();
        sink.emit_view(OwnedLogitsViewV1::new(raw)).unwrap();
        for (row, actual) in sink.sampled.iter().enumerate() {
            let derived = work.for_verify_row(&[4, 5], row);
            let mut state = derived.state;
            let expected = runtime_sampling::sample(
                &derived.parameters,
                &logits,
                derived
                    .history
                    .context(derived.emitted_tokens, &derived.stop_ids),
                &mut state,
            )
            .unwrap();
            assert_eq!(actual.token_id(), expected.token_id());
            assert_eq!(actual.logprob(), expected.logprob());
        }
    }

    impl DeviceLogitsSamplerV1 for StubDeviceSampler {
        fn score_view(
            &self,
            _view: OwnedLogitsViewV1,
            rows: &[DevicePromptRowV1<'_>],
        ) -> HostServiceResultV1<(Vec<SampledRow>, Vec<SampledRow>)> {
            let scored = rows
                .iter()
                .map(|row| {
                    SampledRow::new(
                        row.item.request_id(),
                        row.target.unwrap_or(1),
                        -0.25,
                        Vec::new(),
                    )
                })
                .collect();
            let generated = rows
                .iter()
                .filter(|row| row.prompt && row.output)
                .map(|row| SampledRow::new(row.item.request_id(), 1, -0.25, Vec::new()))
                .collect();
            Ok((scored, generated))
        }

        fn sample_view(
            &self,
            _view: OwnedLogitsViewV1,
            items: &[&ModelBatchItem],
        ) -> HostServiceResultV1<Vec<SampledRow>> {
            Ok(items
                .iter()
                .map(|item| SampledRow::new(item.request_id(), 1, -0.25, Vec::new()))
                .collect())
        }
    }

    #[test]
    fn reference_forcing_emits_the_reference_on_every_verify_row() {
        let item = ModelBatchItem::new(
            RequestId::new(7),
            BatchPhase::Verify { draft_tokens: 2 },
            vec![1, 2, 3],
            true,
        )
        .with_draft(DraftWork::new(2, DraftSource::Host))
        .with_sampling(SamplingWork {
            parameters: SamplingParameters::new(0.0).unwrap(),
            state: SamplingState::new(0),
            history: vec![0, 1].into(),
            emitted_tokens: 1,
            stop_ids: Vec::new(),
            constraint: None,
        });
        let executor = BundleModelExecutorV1::new(FakeExecutionV1::emitting(Vec::new()), 16, 3)
            .with_view_sampler(Rc::new(StubDeviceSampler))
            .with_reference_forcing(Rc::new(crate::reference_forcing::ReferenceTokensV1::new(
                vec![vec![0, 1, 2, 3, 4]],
            )))
            .expect("a device sampler is present");
        assert!(
            BundleModelExecutorV1::new(FakeExecutionV1::emitting(Vec::new()), 16, 3)
                .with_reference_forcing(Rc::new(crate::reference_forcing::ReferenceTokensV1::new(
                    vec![vec![0]]
                ),))
                .is_none()
        );
        let mut sink = CollectingSinkV1 {
            limits: ResultSinkLimitsV1::new(3, 16, 48),
            rows: Vec::new(),
            sampled: Vec::new(),
            prompt_view_rows: Vec::new(),
            expected: std::slice::from_ref(&item),
            sampled_indices: BTreeSet::new(),
            drafts: BTreeMap::new(),
            sampler: executor.view_sampler.clone(),
            defer: false,
            pending: Vec::new(),
            sampling_groups: BTreeMap::new(),
            device: None,
        };
        let mut raw = BundleLogitsViewV1::empty();
        raw.live_width = 16;
        raw.row_count = 3;
        sink.emit_view(OwnedLogitsViewV1::new(raw)).unwrap();
        assert_eq!(
            sink.sampled
                .iter()
                .map(SampledRow::token_id)
                .collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn device_prompt_view_covers_every_prefill_row_and_samples_final_output() {
        let mut sink = CollectingSinkV1 {
            limits: ResultSinkLimitsV1::new(2, 3, 6),
            rows: Vec::new(),
            sampled: Vec::new(),
            prompt_view_rows: Vec::new(),
            expected: &[ModelBatchItem::new(
                RequestId::new(3),
                BatchPhase::Prefill { tokens: 2 },
                vec![1, 2],
                true,
            )
            .with_prompt_targets(vec![None, Some(2)], 0)],
            sampled_indices: BTreeSet::new(),
            drafts: BTreeMap::new(),
            sampler: Some(Rc::new(StubDeviceSampler)),
            defer: false,
            pending: Vec::new(),
            sampling_groups: BTreeMap::new(),
            device: None,
        };
        let mut raw = BundleLogitsViewV1::empty();
        raw.live_width = 3;
        raw.row_count = 2;
        sink.emit_view(OwnedLogitsViewV1::new(raw)).unwrap();
        assert_eq!(sink.prompt_view_rows.len(), 2);
        assert!(sink.prompt_view_rows[0].1.prompt_score().is_none());
        assert_eq!(sink.prompt_view_rows[1].1.prompt_score().unwrap().0, 2);
        assert_eq!(sink.sampled.len(), 1);
        assert_eq!(sink.sampled[0].token_id(), 1);
    }

    #[test]
    fn mixed_prompt_view_can_arrive_before_unrelated_verify_drafts() {
        let prompt = ModelBatchItem::new(
            RequestId::new(3),
            BatchPhase::Prefill { tokens: 2 },
            vec![1, 2],
            true,
        )
        .with_prompt_targets(vec![None, Some(2)], 0);
        let verify = ModelBatchItem::new(
            RequestId::new(4),
            BatchPhase::Verify { draft_tokens: 2 },
            vec![1],
            true,
        )
        .with_draft(DraftWork::new(2, DraftSource::Bundle))
        .with_sampling(SamplingWork {
            parameters: SamplingParameters::new(0.0).unwrap(),
            state: SamplingState::new(0),
            history: vec![1].into(),
            emitted_tokens: 0,
            stop_ids: Vec::new(),
            constraint: None,
        });
        let mut sink = CollectingSinkV1 {
            limits: ResultSinkLimitsV1::new(5, 3, 15),
            rows: Vec::new(),
            sampled: Vec::new(),
            prompt_view_rows: Vec::new(),
            expected: &[prompt, verify],
            sampled_indices: BTreeSet::new(),
            drafts: BTreeMap::new(),
            sampler: Some(Rc::new(StubDeviceSampler)),
            defer: false,
            pending: Vec::new(),
            sampling_groups: BTreeMap::new(),
            device: None,
        };
        let mut view = BundleLogitsViewV1::empty();
        view.live_width = 3;
        view.row_count = 2;
        sink.emit_view(OwnedLogitsViewV1::new(view)).unwrap();
        view.first_item_index = 2;
        view.row_count = 3;
        assert_eq!(
            sink.emit_view(OwnedLogitsViewV1::new(view)),
            Err(BundleStatusV1::InvalidState)
        );
        sink.emit_draft_tokens(OwnedDraftTokensV1 {
            request_id: 4,
            tokens: vec![1, 2],
        })
        .unwrap();
        sink.emit_view(OwnedLogitsViewV1::new(view)).unwrap();
        assert_eq!(sink.prompt_view_rows.len(), 2);
        assert_eq!(sink.sampled.len(), 4);
    }

    #[test]
    fn device_view_sink_checks_live_width_order_and_duplicate_coverage() {
        let mut sink = CollectingSinkV1 {
            limits: ResultSinkLimitsV1::new(2, 3, 6),
            rows: Vec::new(),
            sampled: Vec::new(),
            prompt_view_rows: Vec::new(),
            expected: &[
                ModelBatchItem::new(RequestId::new(3), BatchPhase::Decode, vec![1], true)
                    .with_sampling(super::sampling_tests::work(0.0)),
                ModelBatchItem::new(RequestId::new(4), BatchPhase::Decode, vec![1], true)
                    .with_sampling(super::sampling_tests::work(0.0)),
            ],
            sampled_indices: BTreeSet::new(),
            drafts: BTreeMap::new(),
            sampler: Some(Rc::new(StubDeviceSampler)),
            defer: false,
            pending: Vec::new(),
            sampling_groups: BTreeMap::new(),
            device: None,
        };
        let mut raw = BundleLogitsViewV1::empty();
        raw.live_width = 2;
        raw.row_count = 2;
        assert_eq!(
            sink.emit_view(OwnedLogitsViewV1::new(raw)),
            Err(BundleStatusV1::InvalidArgument)
        );
        raw.live_width = 3;
        sink.emit_view(OwnedLogitsViewV1::new(raw)).unwrap();
        assert_eq!(sink.sampled.len(), 2);
        assert_eq!(sink.sampled[1].request_id(), RequestId::new(4));
        assert_eq!(
            sink.emit_view(OwnedLogitsViewV1::new(raw)),
            Err(BundleStatusV1::InvalidArgument)
        );
    }

    const VOCABULARY: usize = 3;
    const REQUEST: u64 = 11;

    /// What one fake execution did, in call order.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum CallV1 {
        Execute,
        Commit,
        CommitAccepted(u64, u32),
        Cancel,
        Destroy,
        Retire(u64),
        /// A restore's request and the state slot it named, if any.
        Restore(u64, Option<u64>),
    }

    /// A bundle execution that emits exactly the rows a test hands it.
    struct FakeExecutionV1 {
        prefill_chunk: u32,
        envelopes: Vec<Vec<u8>>,
        device_results: bool,
        device_metadata: Vec<bool>,
        captured: Vec<(u64, u64)>,
        emitted: Vec<OwnedLogitsRowV1>,
        execute_error: Option<BundleAbiError>,
        commit_error: Option<BundleAbiError>,
        cancel_error: Option<BundleAbiError>,
        calls: Vec<CallV1>,
        poisoned: bool,
        pre_device_refusal: bool,
        dynamic_rows: bool,
        /// A prefix-state refusal and whether it poisons the instance.
        prefix_error: Option<(BundleAbiError, bool)>,
        state_slot_restore: bool,
        /// Samples every row on the "device": token 1, except that the
        /// rows of these requests have no surviving token.
        device_failing: Option<BTreeSet<u64>>,
    }

    impl FakeExecutionV1 {
        fn emitting(emitted: Vec<OwnedLogitsRowV1>) -> Self {
            Self {
                prefill_chunk: 0,
                envelopes: Vec::new(),
                device_results: false,
                device_metadata: Vec::new(),
                captured: Vec::new(),
                emitted,
                execute_error: None,
                commit_error: None,
                cancel_error: None,
                calls: Vec::new(),
                poisoned: false,
                pre_device_refusal: false,
                dynamic_rows: false,
                prefix_error: None,
                state_slot_restore: false,
                device_failing: None,
            }
        }

        fn prefix_state(&mut self) -> Result<(), BundleAbiError> {
            match self.prefix_error.clone() {
                Some((error, poisons)) => {
                    self.poisoned |= poisons;
                    Err(error)
                }
                None => Ok(()),
            }
        }

        fn is_pending(&self) -> bool {
            self.calls.last() == Some(&CallV1::Execute)
        }
    }

    impl BundleExecutionV1 for FakeExecutionV1 {
        fn prefill_chunk_tokens(&self) -> Result<u32, BundleAbiError> {
            Ok(self.prefill_chunk)
        }

        fn supports_device_results(&self) -> bool {
            self.device_results
        }

        fn execute_with_device_sampling(
            &mut self,
            request: &[u8],
            limits: ResultSinkLimitsV1,
            items: &[ModelBatchItem],
            sampler: Option<Rc<dyn DeviceLogitsSamplerV1>>,
            metadata: Option<super::DeviceCycleMetadataV1>,
        ) -> Result<BatchEmissionsV1, BundleAbiError> {
            self.device_metadata.push(metadata.is_some());
            // The existing prefill ABI does not carry a device-result sink.
            if metadata.is_some() && request.starts_with(runtime_bundle_sdk::prefill::MAGIC) {
                return Err(BundleAbiError::Status(BundleStatusV1::Unsupported));
            }
            self.execute_with_sampling(request, limits, items, sampler)
        }

        fn last_execute_refusal(&self) -> Option<&str> {
            self.pre_device_refusal
                .then_some("packing check: item token budget exceeded")
        }

        fn retire_request(&mut self, request_id: u64) -> Result<(), BundleAbiError> {
            assert!(!self.is_pending());
            self.calls.push(CallV1::Retire(request_id));
            Ok(())
        }

        fn supports_request_retirement(&self) -> bool {
            true
        }

        fn capture_prefix_state(
            &mut self,
            request_id: u64,
            tokens: u64,
            _pages: &[u64],
        ) -> Result<(), BundleAbiError> {
            self.captured.push((request_id, tokens));
            self.prefix_state()
        }

        fn restore_prefix_state(
            &mut self,
            request_id: u64,
            _token_ids: &[u32],
            _pages: &[u64],
        ) -> Result<(), BundleAbiError> {
            self.calls.push(CallV1::Restore(request_id, None));
            self.prefix_state()
        }

        fn supports_prefix_state_slot(&self) -> bool {
            self.state_slot_restore
        }

        fn restore_prefix_state_in_slot(
            &mut self,
            request_id: u64,
            _token_ids: &[u32],
            _pages: &[u64],
            state_slot: u64,
        ) -> Result<(), BundleAbiError> {
            self.calls
                .push(CallV1::Restore(request_id, Some(state_slot)));
            self.prefix_state()
        }

        fn execute_batch(
            &mut self,
            request: &[u8],
            limits: ResultSinkLimitsV1,
        ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
            if let Some(error) = self.execute_error.clone() {
                // A refused execute leaves nothing pending, exactly as the ABI
                // wrapper leaves a live instance.
                return Err(error);
            }
            if self.emitted.len() as u64 > limits.maximum_rows {
                return Err(BundleAbiError::ExecuteResultCapacity);
            }
            if request.starts_with(runtime_bundle_sdk::prefill::MAGIC) {
                assert_ne!(self.prefill_chunk, 0);
                runtime_bundle_sdk::prefill::decode(request).unwrap();
                self.envelopes.push(request.to_vec());
            }
            self.calls.push(CallV1::Execute);
            Ok(self.emitted.clone())
        }

        fn execute_with_sampling(
            &mut self,
            request: &[u8],
            limits: ResultSinkLimitsV1,
            items: &[ModelBatchItem],
            _sampler: Option<Rc<dyn DeviceLogitsSamplerV1>>,
        ) -> Result<BatchEmissionsV1, BundleAbiError> {
            let result = self.execute_batch(request, limits);
            if self.dynamic_rows {
                self.execute_error = None;
            }
            let mut rows = result?;
            if let Some(failing) = &self.device_failing {
                let sampled = items
                    .iter()
                    .filter(|item| item.expects_logits())
                    .flat_map(|item| {
                        let id = item.request_id();
                        let count = item.draft().map_or(1, |draft| draft.tokens() + 1);
                        std::iter::repeat_n(
                            if failing.contains(&id.get()) {
                                SampledRow::failed(
                                    id,
                                    runtime_sampling::SamplingError::AllTokensMasked,
                                )
                            } else {
                                SampledRow::new(id, 1, -0.25, Vec::new())
                            },
                            count,
                        )
                    })
                    .collect();
                return Ok(BatchEmissionsV1 {
                    accepted_prefixes: Vec::new(),
                    rows: Vec::new(),
                    sampled,
                    prompt_view_rows: Vec::new(),
                    drafts: Vec::new(),
                });
            }
            if self.dynamic_rows {
                rows = items
                    .iter()
                    .filter(|item| item.expects_logits())
                    .map(|item| row(item.request_id().get(), &[0.0, 10.0, 0.0]))
                    .collect();
            }
            Ok(BatchEmissionsV1 {
                accepted_prefixes: Vec::new(),
                rows,
                sampled: Vec::new(),
                prompt_view_rows: Vec::new(),
                drafts: Vec::new(),
            })
        }

        fn commit(&mut self) -> Result<(), BundleAbiError> {
            self.calls.push(CallV1::Commit);
            if let Some(error) = self.commit_error.clone() {
                self.poisoned = true;
                return Err(error);
            }
            Ok(())
        }

        fn commit_accepted(
            &mut self,
            records: &[runtime_bundle_abi::BundleKeptRowsV1],
        ) -> Result<(), BundleAbiError> {
            if records.len() != 1 {
                return Err(BundleAbiError::InvalidKeptRows);
            }
            self.calls.push(CallV1::CommitAccepted(
                records[0].request_id,
                records[0].kept_rows,
            ));
            Ok(())
        }

        fn cancel(&mut self) -> Result<(), BundleAbiError> {
            self.calls.push(CallV1::Cancel);
            if let Some(error) = self.cancel_error.clone() {
                self.poisoned = true;
                return Err(error);
            }
            Ok(())
        }

        fn destroy(&mut self) -> Result<(), BundleAbiError> {
            self.calls.push(CallV1::Destroy);
            Ok(())
        }

        fn is_poisoned(&self) -> bool {
            self.poisoned
        }
    }

    #[test]
    fn guaranteed_profile_reuses_complete_response_state_on_the_next_turn() {
        use runtime_engine::{Engine, EngineConfig, EngineRequest};
        use runtime_model_api::ModelVocabulary;
        use runtime_scheduler::SchedulerConfig;
        let mut instance = FakeExecutionV1::emitting(Vec::new());
        instance.prefill_chunk = 2048;
        instance.dynamic_rows = true;
        let vocabulary = ModelVocabulary::new(VOCABULARY).unwrap();
        let geometry = KvGeometry::new(64, 1024).unwrap();
        let config = EngineConfig::new(
            SchedulerConfig::new(1, 4, 8192, 2048, 1).unwrap(),
            geometry,
            vocabulary,
            geometry.token_capacity(),
            4,
        )
        .unwrap()
        .with_guaranteed_prefill_chunks(2048)
        .unwrap()
        .with_prefix_cache(1);
        let mut engine = Engine::new(
            config,
            BundleModelExecutorV1::new_with_batch_wire(
                instance,
                VOCABULARY,
                1,
                super::BatchWireCapabilityV1::V3Speculative,
            ),
        )
        .unwrap();
        let request = |id, tokens, max| {
            EngineRequest::new(
                RequestId::new(id),
                tokens,
                max,
                SamplingParameters::new(0.0).unwrap(),
                id,
                Vec::new(),
                vocabulary,
            )
            .unwrap()
        };
        engine.submit(request(1, vec![0; 65], 3)).unwrap();
        for _ in 0..3 {
            engine.step().unwrap();
        }
        assert_eq!(engine.executor().instance.captured, [(1, 65), (1, 67)]);
        let mut next = vec![0; 65];
        next.extend([1, 1, 1, 0]);
        engine.submit(request(2, next, 1)).unwrap();
        engine.step().unwrap();
        let instance = &engine.executor().instance;
        assert!(instance.calls.contains(&CallV1::Restore(2, None)));
        let envelope =
            runtime_bundle_sdk::prefill::decode(instance.envelopes.last().unwrap()).unwrap();
        assert_eq!(envelope.batch.appends()[0].previous_tokens(), 67);
        assert_eq!(envelope.batch.items()[0].phase_tokens(), 2);
    }

    #[test]
    fn engine_supplies_each_guaranteed_chunk_end_and_final_boundary() {
        use runtime_engine::{Engine, EngineConfig, EngineRequest};
        use runtime_model_api::ModelVocabulary;
        use runtime_scheduler::SchedulerConfig;
        for chunk in [2048, 8192] {
            let mut instance = FakeExecutionV1::emitting(Vec::new());
            instance.prefill_chunk = chunk as u32;
            instance.dynamic_rows = true;
            let vocabulary = ModelVocabulary::new(VOCABULARY).unwrap();
            let geometry = KvGeometry::new(64, 1024).unwrap();
            let config = EngineConfig::new(
                SchedulerConfig::new(1, 1, 8192, 2048, 1).unwrap(),
                geometry,
                vocabulary,
                geometry.token_capacity(),
                1,
            )
            .unwrap()
            .with_guaranteed_prefill_chunks(chunk)
            .unwrap();
            let mut engine =
                Engine::new(config, BundleModelExecutorV1::new(instance, VOCABULARY, 1)).unwrap();
            let prompt = chunk * 2 + 17;
            engine
                .submit(
                    EngineRequest::new(
                        RequestId::new(1),
                        vec![0; prompt],
                        1,
                        SamplingParameters::new(0.0).unwrap(),
                        1,
                        Vec::new(),
                        vocabulary,
                    )
                    .unwrap(),
                )
                .unwrap();
            for _ in 0..3 {
                engine.step().unwrap();
            }
            let envelopes = &engine.executor().instance.envelopes;
            assert_eq!(envelopes.len(), 3);
            for (encoded, end) in envelopes.iter().zip([chunk, chunk * 2, prompt]) {
                let envelope = runtime_bundle_sdk::prefill::decode(encoded).unwrap();
                assert_eq!(envelope.boundaries.len(), 1);
                assert_eq!(envelope.boundaries[0].chunk_end, end as u64);
                assert_eq!(envelope.boundaries[0].prompt_tokens, prompt as u64);
                assert_eq!(envelope.batch.items()[0].expects_logits(), end == prompt);
            }
        }
    }

    #[test]
    fn prefill_envelope_requires_both_a_declaration_and_scheduled_boundaries() {
        use runtime_engine::PrefillBoundary;
        for declared in [false, true] {
            for scheduled in [false, true] {
                let mut instance = FakeExecutionV1::emitting(vec![row(REQUEST, &[1.0, 2.0, 3.0])]);
                instance.prefill_chunk = if declared { 2048 } else { 0 };
                let mut executor = BundleModelExecutorV1::new(instance, VOCABULARY, 8);
                let mut items = items();
                if scheduled {
                    items[0] = items[0].clone().with_prefill_boundary(PrefillBoundary {
                        chunk_tokens: 2048,
                        prompt_tokens: 2,
                    });
                }
                drop(executor.execute(&items, &kv_plan()).unwrap());
                assert_eq!(
                    executor.instance.envelopes.len(),
                    usize::from(declared && scheduled)
                );
                if declared && scheduled {
                    let envelope =
                        runtime_bundle_sdk::prefill::decode(&executor.instance.envelopes[0])
                            .unwrap();
                    assert_eq!(envelope.boundaries[0].chunk_end, 2);
                    assert_eq!(envelope.boundaries[0].prompt_tokens, 2);
                    assert_eq!(
                        envelope.original,
                        super::encode_batch_request_v1(&items, &kv_plan()).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn verify_finalization_forwards_kept_rows_to_the_abi_tail() {
        let mut instance = FakeExecutionV1::emitting(Vec::new());
        let outcome = Rc::new(std::cell::RefCell::new(TransactionOutcomeV1::None));
        let mut state = BundleTransactionStateV1 {
            instance: Some(&mut instance),
            outcome: Rc::clone(&outcome),
        };
        state
            .commit_kept(&[KeptRows {
                request_id: RequestId::new(17),
                kept_rows: 2,
            }])
            .unwrap();
        assert_eq!(instance.calls, [CallV1::CommitAccepted(17, 2)]);
        assert_eq!(*outcome.borrow(), TransactionOutcomeV1::Committed);
    }

    #[test]
    fn only_a_prefix_state_failure_that_poisons_the_bundle_is_fatal() {
        for poisons in [false, true] {
            let mut instance = FakeExecutionV1::emitting(Vec::new());
            let status = if poisons {
                BundleStatusV1::InternalError
            } else {
                BundleStatusV1::InvalidState
            };
            instance.prefix_error = Some((BundleAbiError::Status(status), poisons));
            let mut executor = BundleModelExecutorV1::new(instance, 3, 8);
            let pages = [];
            let capture = executor
                .capture_prefix_state_v1(RequestId::new(REQUEST), 2, &pages)
                .unwrap_err();
            let restore = executor
                .restore_prefix_state_v1(RequestId::new(REQUEST), &[5, 6], &pages)
                .unwrap_err();
            assert_eq!(capture.is_fatal(), poisons);
            assert_eq!(restore.is_fatal(), poisons);
        }
    }

    #[test]
    fn a_failed_accepted_row_commit_is_returned_and_recorded() {
        let mut instance = FakeExecutionV1::emitting(Vec::new());
        let outcome = Rc::new(std::cell::RefCell::new(TransactionOutcomeV1::None));
        let mut state = BundleTransactionStateV1 {
            instance: Some(&mut instance),
            outcome: Rc::clone(&outcome),
        };
        let kept = [17, 18].map(|id| KeptRows {
            request_id: RequestId::new(id),
            kept_rows: 1,
        });
        let error = state.commit_kept(&kept).unwrap_err();
        assert!(error.message().contains("bundle commit failed"), "{error}");
        assert_eq!(
            *outcome.borrow(),
            TransactionOutcomeV1::Failed(BundleAbiError::InvalidKeptRows)
        );
    }

    /// The scheduler commits before the bundle does, so a failed bundle
    /// commit leaves nothing safe to publish: no token reaches a client, a
    /// request that just reached its length limit does not finish, and the
    /// engine executes nothing more.
    #[test]
    fn a_failed_bundle_commit_publishes_nothing_and_stops_the_engine() {
        use runtime_engine::{Engine, EngineConfig, EngineError, EngineEventKind, EngineRequest};
        use runtime_model_api::ModelVocabulary;
        use runtime_scheduler::SchedulerConfig;

        let mut instance =
            FakeExecutionV1::emitting(vec![row(1, &[0.0, 5.0, 0.0]), row(2, &[0.0, 0.0, 5.0])]);
        instance.commit_error = Some(BundleAbiError::Poisoned);
        let vocabulary = ModelVocabulary::new(VOCABULARY).unwrap();
        let geometry = KvGeometry::new(4, 8).unwrap();
        let mut engine = Engine::new(
            EngineConfig::new(
                SchedulerConfig::new(2, 4, 8, 8, 1).unwrap(),
                geometry,
                vocabulary,
                geometry.token_capacity(),
                2,
            )
            .unwrap(),
            BundleModelExecutorV1::new(instance, VOCABULARY, 4),
        )
        .unwrap();
        for (id, maximum) in [(1, 1), (2, 4)] {
            let request = EngineRequest::new(
                RequestId::new(id),
                vec![0],
                maximum,
                SamplingParameters::new(0.0).unwrap(),
                id,
                Vec::new(),
                vocabulary,
            )
            .unwrap();
            engine.submit(request).unwrap();
        }

        assert!(matches!(engine.step(), Err(EngineError::Executor(_))));
        let published = engine.drain_events();
        assert!(
            published.iter().all(|event| !matches!(
                event.kind(),
                EngineEventKind::Token { .. } | EngineEventKind::Finished { .. }
            )),
            "{published:?}"
        );
        assert_eq!(engine.step(), Err(EngineError::ModelStateFailed));
        assert!(matches!(
            engine.executor().last_outcome(),
            TransactionOutcomeV1::Failed(BundleAbiError::Poisoned)
        ));
    }

    /// A device-sampled row with no surviving token fails only its request.
    /// With in-place state the bundle commits the step, never cancelling it
    /// first, and the failed request retires; with rollback-preserving
    /// state the bundle rolls back and the sibling retries. Either way the
    /// sibling publishes all its tokens and finishes.
    #[test]
    fn a_device_row_that_samples_nothing_fails_only_its_request() {
        use runtime_engine::{Engine, EngineConfig, EngineError, EngineEventKind, EngineRequest};
        use runtime_model_api::ModelVocabulary;
        use runtime_sampling::SamplingError;
        use runtime_scheduler::SchedulerConfig;

        for in_place in [true, false] {
            let mut instance = FakeExecutionV1::emitting(Vec::new());
            instance.device_failing = Some(BTreeSet::from([1]));
            let vocabulary = ModelVocabulary::new(VOCABULARY).unwrap();
            let geometry = KvGeometry::new(4, 8).unwrap();
            let mut engine = Engine::new(
                EngineConfig::new(
                    SchedulerConfig::new(2, 4, 8, 8, 1).unwrap(),
                    geometry,
                    vocabulary,
                    geometry.token_capacity(),
                    2,
                )
                .unwrap(),
                BundleModelExecutorV1::new(instance, VOCABULARY, 4).with_in_place_state(in_place),
            )
            .unwrap();
            for (id, maximum) in [(1, 3), (2, 2)] {
                let request = EngineRequest::new(
                    RequestId::new(id),
                    vec![0],
                    maximum,
                    SamplingParameters::new(0.0).unwrap(),
                    id,
                    Vec::new(),
                    vocabulary,
                )
                .unwrap();
                engine.submit(request).unwrap();
            }
            assert_eq!(
                engine.step(),
                Err(EngineError::Sampling(SamplingError::AllTokensMasked)),
                "in place {in_place}"
            );
            engine.run_until_idle(8).unwrap();
            engine.audit().unwrap();
            let events = engine.drain_events();
            let kinds = |id: u64| {
                events
                    .iter()
                    .filter(|event| event.request_id() == RequestId::new(id))
                    .map(|event| event.kind().clone())
                    .collect::<Vec<_>>()
            };
            assert!(
                matches!(kinds(1)[..], [EngineEventKind::Error { .. }]),
                "in place {in_place}: {:?}",
                kinds(1)
            );
            assert!(
                matches!(
                    kinds(2)[..],
                    [
                        EngineEventKind::Token { token_id: 1, .. },
                        EngineEventKind::Token { token_id: 1, .. },
                        EngineEventKind::Finished { .. }
                    ]
                ),
                "in place {in_place}: {:?}",
                kinds(2)
            );
            let calls = &engine.executor().instance.calls;
            assert_eq!(
                calls[..2],
                [
                    CallV1::Execute,
                    if in_place {
                        CallV1::Commit
                    } else {
                        CallV1::Cancel
                    }
                ],
                "in place {in_place}"
            );
            assert!(calls.contains(&CallV1::Retire(1)) && calls.contains(&CallV1::Retire(2)));
            assert_eq!(engine.kv_stats().used_blocks(), 0);
        }
    }

    #[test]
    fn proposed_depth_rounds_down_to_a_declared_row_bucket() {
        let executor = BundleModelExecutorV1::new(FakeExecutionV1::emitting(Vec::new()), 3, 8)
            .with_row_buckets(&[1, 2, 4, 8]);
        assert_eq!(executor.round_draft_tokens_v1(0), 0);
        assert_eq!(executor.round_draft_tokens_v1(2), 1);
        assert_eq!(executor.round_draft_tokens_v1(5), 3);
        assert_eq!(executor.round_draft_tokens_v1(7), 7);
        let executor = executor.with_row_buckets(&[2, 3, 4, 8]);
        assert_eq!(executor.round_draft_tokens_v1(2), 2);
        assert_eq!(executor.round_draft_tokens_v1(3), 3);
    }

    #[test]
    fn steps_run_the_smallest_declared_bucket_that_holds_them() {
        let executor = BundleModelExecutorV1::new(FakeExecutionV1::emitting(Vec::new()), 3, 8);
        assert_eq!(executor.step_rows_v1(5), 5);
        let executor = executor.with_step_buckets(&[64, 1, 32, 16]);
        assert_eq!(executor.step_rows_v1(1), 1);
        assert_eq!(executor.step_rows_v1(17), 32);
        assert_eq!(executor.step_rows_v1(33), 64);
        assert_eq!(executor.step_rows_v1(65), 65);
    }

    #[test]
    fn executed_capacity_covers_realized_verify_and_ar_rows_without_guessing() {
        let item = |phase| ModelBatchItem::new(RequestId::new(1), phase, vec![0], true);
        let executor = BundleModelExecutorV1::new(FakeExecutionV1::emitting(Vec::new()), 3, 256);
        let mut items = vec![
            item(BatchPhase::Verify { draft_tokens: 2 }),
            item(BatchPhase::Decode),
        ];
        assert_eq!(executor.executed_row_capacity(&items), None);
        let executor = executor.with_step_buckets(&[1, 2, 4, 8, 16, 32, 64, 128, 256]);
        assert_eq!(executor.executed_row_capacity(&items), Some(4));
        items = (0..16)
            .map(|_| item(BatchPhase::Verify { draft_tokens: 5 }))
            .collect();
        assert_eq!(executor.executed_row_capacity(&items), Some(128));
        items.extend((0..16).map(|_| item(BatchPhase::Verify { draft_tokens: 5 })));
        assert_eq!(executor.executed_row_capacity(&items), Some(256));
        items.push(item(BatchPhase::Prefill { tokens: 1 }));
        assert_eq!(executor.executed_row_capacity(&items), None);
        assert_eq!(executor.executed_row_capacity(&[]), None);
        let too_many = vec![item(BatchPhase::Verify { draft_tokens: 256 })];
        assert_eq!(executor.executed_row_capacity(&too_many), None);
    }

    #[test]
    fn executed_capacity_survives_the_bundle_transaction_adapter() {
        for verify in [false, true] {
            let items = if verify {
                vec![
                    ModelBatchItem::new(
                        RequestId::new(REQUEST),
                        BatchPhase::Verify { draft_tokens: 1 },
                        vec![1, 2],
                        true,
                    )
                    .with_draft(runtime_engine::DraftWork::new(1, DraftSource::Host)),
                ]
            } else {
                items()
            };
            let emitted = (0..if verify { 2 } else { 1 })
                .map(|_| row(REQUEST, &[1.0, 2.0, 3.0]))
                .collect();
            let mut executor = BundleModelExecutorV1::new_with_batch_wire(
                FakeExecutionV1::emitting(emitted),
                VOCABULARY,
                8,
                super::BatchWireCapabilityV1::V3Speculative,
            )
            .with_step_buckets(&[1, 4, 8]);
            let transaction = executor.execute(&items, &kv_plan()).unwrap();
            assert_eq!(transaction.row_capacity(), verify.then_some(4));
        }
    }

    fn row(request_id: u64, logits: &[f32]) -> OwnedLogitsRowV1 {
        OwnedLogitsRowV1::new(
            request_id,
            logits
                .iter()
                .flat_map(|logit| logit.to_le_bytes())
                .collect(),
        )
    }

    fn items() -> Vec<ModelBatchItem> {
        vec![ModelBatchItem::new(
            RequestId::new(REQUEST),
            BatchPhase::Prefill { tokens: 2 },
            vec![5, 6],
            true,
        )]
    }

    fn kv_plan() -> PreparedAppendBatch {
        let geometry = KvGeometry::new(4, 8).expect("the test KV geometry is valid");
        let mut pool = KvPool::new(geometry, 2).expect("the test KV pool is valid");
        let sequence = pool.create_sequence(8).expect("the test sequence fits");
        pool.prepare_append_batch(&[(sequence, 2)])
            .expect("the test append fits")
    }

    /// Runs one execution that the adapter is expected to refuse.
    fn refused(
        emitted: Vec<OwnedLogitsRowV1>,
    ) -> (ExecutorError, BundleModelExecutorV1<FakeExecutionV1>) {
        let mut executor =
            BundleModelExecutorV1::new(FakeExecutionV1::emitting(emitted), VOCABULARY, 4);
        let error = executor
            .execute(&items(), &kv_plan())
            .err()
            .expect("the adapter refuses this execution");
        (error, executor)
    }

    fn assert_cancelled(executor: &BundleModelExecutorV1<FakeExecutionV1>) {
        assert_eq!(executor.last_outcome(), TransactionOutcomeV1::Cancelled);
        assert_eq!(
            executor.instance.calls,
            vec![CallV1::Execute, CallV1::Cancel]
        );
        assert!(!executor.instance.is_pending());
        assert!(!executor.is_poisoned());
    }

    #[test]
    fn a_state_slot_restore_names_the_slot_admission_assigned() {
        for state_slot_restore in [false, true] {
            let mut instance = FakeExecutionV1::emitting(Vec::new());
            instance.state_slot_restore = state_slot_restore;
            let mut executor = BundleModelExecutorV1::new(instance, VOCABULARY, 4)
                .with_state_slots(crate::recurrent::StateSlotPoolV1::with_capacity(2));
            executor.admit_request_v1(RequestId::new(40)).unwrap();
            executor.admit_request_v1(RequestId::new(41)).unwrap();
            // Unadmitted requests have no slot to restore into.
            let unadmitted = executor.restore_prefix_state_v1(RequestId::new(42), &[1], &[]);
            assert_eq!(unadmitted.is_err(), state_slot_restore);
            executor
                .restore_prefix_state_v1(RequestId::new(41), &[1, 2], &[])
                .unwrap();
            let expected = if state_slot_restore {
                vec![CallV1::Restore(41, Some(1))]
            } else {
                vec![CallV1::Restore(42, None), CallV1::Restore(41, None)]
            };
            assert_eq!(executor.instance.calls, expected);
        }
    }

    #[test]
    fn retirement_reaches_the_bundle_after_rollback() {
        let mut executor = BundleModelExecutorV1::new(
            FakeExecutionV1::emitting(vec![row(REQUEST, &[1.0, 2.0, 3.0])]),
            VOCABULARY,
            4,
        );
        assert!(executor.supports_request_retirement());
        drop(executor.execute(&items(), &kv_plan()).unwrap());
        executor.retire_request_v1(RequestId::new(REQUEST)).unwrap();
        assert_eq!(
            executor.instance.calls,
            vec![CallV1::Execute, CallV1::Cancel, CallV1::Retire(REQUEST),]
        );
    }

    #[test]
    fn accepted_rows_leave_the_execution_pending_until_the_engine_finalizes_it() {
        let mut executor = BundleModelExecutorV1::new(
            FakeExecutionV1::emitting(vec![row(REQUEST, &[1.0, 2.0, 3.0])]),
            VOCABULARY,
            4,
        );

        let transaction = executor
            .execute(&items(), &kv_plan())
            .expect("the adapter accepts this execution");
        assert_eq!(transaction.logits().len(), 1);
        drop(transaction);

        assert_eq!(executor.last_outcome(), TransactionOutcomeV1::Cancelled);
    }

    #[test]
    fn a_wrong_row_count_cancels_before_the_error_returns() {
        let (error, executor) = refused(vec![
            row(REQUEST, &[1.0, 2.0, 3.0]),
            row(REQUEST, &[1.0, 2.0, 3.0]),
        ]);

        assert!(error.message().contains("2 logits rows for 1"));
        assert_cancelled(&executor);
    }

    #[test]
    fn a_wrong_request_id_cancels_before_the_error_returns() {
        let (error, executor) = refused(vec![row(REQUEST + 1, &[1.0, 2.0, 3.0])]);

        assert!(error.message().contains("row for request 12"));
        assert_cancelled(&executor);
    }

    #[test]
    fn a_wrong_logits_byte_length_cancels_before_the_error_returns() {
        let (error, executor) = refused(vec![row(REQUEST, &[1.0, 2.0])]);

        assert!(error.message().contains("2 logits for a vocabulary of 3"));
        assert_cancelled(&executor);
    }

    #[test]
    fn an_unusable_logit_value_cancels_before_the_error_returns() {
        let (error, executor) = refused(vec![row(REQUEST, &[1.0, f32::NAN, 3.0])]);

        assert!(error.message().contains("non-finite logit"));
        assert_cancelled(&executor);
    }

    #[test]
    fn a_failed_cancel_reports_both_failures_and_poisons_the_instance() {
        let mut executor = BundleModelExecutorV1::new(
            FakeExecutionV1 {
                cancel_error: Some(BundleAbiError::Poisoned),
                ..FakeExecutionV1::emitting(vec![row(REQUEST, &[1.0, 2.0])])
            },
            VOCABULARY,
            4,
        );

        let error = executor
            .execute(&items(), &kv_plan())
            .err()
            .expect("the adapter refuses this execution");

        assert!(error.message().contains("2 logits for a vocabulary of 3"));
        assert!(
            error
                .message()
                .contains("cancelling that execution also failed")
        );
        assert_eq!(
            executor.last_outcome(),
            TransactionOutcomeV1::Failed(BundleAbiError::Poisoned)
        );
        assert!(executor.is_poisoned());
    }

    #[test]
    fn a_bundle_refusal_retires_exactly_its_batch_and_reuses_pages_and_slots() {
        use runtime_engine::{Engine, EngineConfig, EngineEventKind, EngineRequest};
        use runtime_model_api::ModelVocabulary;
        use runtime_scheduler::SchedulerConfig;
        let instance = FakeExecutionV1 {
            execute_error: Some(BundleAbiError::Status(BundleStatusV1::InvalidArgument)),
            pre_device_refusal: true,
            dynamic_rows: true,
            ..FakeExecutionV1::emitting(Vec::new())
        };
        let executor = BundleModelExecutorV1::new(instance, 3, 3)
            .with_state_slots(crate::recurrent::StateSlotPoolV1::for_test(3));
        let vocabulary = ModelVocabulary::new(3).unwrap();
        let config = EngineConfig::new(
            SchedulerConfig::new(3, 6, 8, 4, 1).unwrap(),
            KvGeometry::new(4, 12).unwrap(),
            vocabulary,
            16,
            6,
        )
        .unwrap();
        let mut engine = Engine::new(config, executor).unwrap();
        let request = |id| {
            EngineRequest::new(
                RequestId::new(id),
                vec![0; 4],
                1,
                SamplingParameters::new(0.0).unwrap(),
                id,
                Vec::new(),
                vocabulary,
            )
            .unwrap()
        };
        for id in 1..=3 {
            engine.submit(request(id)).unwrap();
        }
        engine.step().unwrap();
        engine.audit().unwrap();
        let events = engine.drain_events();
        assert_eq!(events.len(), 2);
        for (event, id) in events.iter().zip(1..=2) {
            assert_eq!(event.request_id(), RequestId::new(id));
            assert!(
                matches!(event.kind(), EngineEventKind::Error { message } if message.contains("packing check"))
            );
            assert_eq!(
                engine
                    .executor()
                    .state_slots
                    .as_ref()
                    .unwrap()
                    .slot(RequestId::new(id)),
                None
            );
        }
        assert!(
            engine
                .executor()
                .state_slots
                .as_ref()
                .unwrap()
                .slot(RequestId::new(3))
                .is_some()
        );
        assert_eq!(
            engine.executor().instance.calls,
            vec![CallV1::Retire(1), CallV1::Retire(2)]
        );
        for id in 4..=5 {
            engine.submit(request(id)).unwrap();
        }
        engine.run_until_idle(10).unwrap();
        engine.audit().unwrap();
        assert_eq!(engine.kv_stats().used_blocks(), 0);
        for id in 1..=5 {
            assert_eq!(
                engine
                    .executor()
                    .state_slots
                    .as_ref()
                    .unwrap()
                    .slot(RequestId::new(id)),
                None
            );
        }
        let finished: Vec<_> = engine
            .drain_events()
            .into_iter()
            .filter_map(|event| {
                matches!(event.kind(), EngineEventKind::Finished { .. })
                    .then_some(event.request_id())
            })
            .collect();
        assert_eq!(
            finished,
            vec![RequestId::new(3), RequestId::new(4), RequestId::new(5)]
        );
        let metrics = engine.metrics().prometheus_compatibility("fixture");
        assert!(
            metrics.contains(
                "vllm:batch_refused_requests_total{model_name=\"fixture\",engine=\"0\"} 2"
            )
        );
    }

    #[test]
    fn only_a_proven_clean_invalid_argument_refusal_can_recover() {
        for (pre_device, poisoned, status) in [
            (true, false, BundleStatusV1::InvalidArgument),
            (false, false, BundleStatusV1::InvalidArgument),
            (true, true, BundleStatusV1::InvalidArgument),
            (true, false, BundleStatusV1::InvalidState),
        ] {
            let mut executor = BundleModelExecutorV1::new(
                FakeExecutionV1 {
                    execute_error: Some(BundleAbiError::Status(status)),
                    pre_device_refusal: pre_device,
                    poisoned,
                    ..FakeExecutionV1::emitting(Vec::new())
                },
                VOCABULARY,
                4,
            );
            let error = executor.execute(&items(), &kv_plan()).err().unwrap();
            let safe = pre_device && !poisoned && status == BundleStatusV1::InvalidArgument;
            assert_eq!(error.is_batch_refused(), safe);
            if safe {
                assert!(error.message().contains("packing check"));
            }
            assert!(executor.instance.calls.is_empty());
        }
    }

    #[test]
    fn a_refused_execute_leaves_nothing_to_cancel() {
        let mut executor = BundleModelExecutorV1::new(
            FakeExecutionV1 {
                execute_error: Some(BundleAbiError::ExecuteResultCapacity),
                ..FakeExecutionV1::emitting(Vec::new())
            },
            VOCABULARY,
            4,
        );

        let error = executor
            .execute(&items(), &kv_plan())
            .err()
            .expect("the adapter refuses this execution");

        assert!(error.message().contains("bundle execute failed"));
        assert!(executor.instance.calls.is_empty());
        assert_eq!(executor.last_outcome(), TransactionOutcomeV1::None);
    }

    /// Execute returns only a status, so the failure names the status and
    /// the batch the bundle refused.
    #[test]
    fn a_refused_execute_names_its_status_and_the_batch_shape() {
        let mut executor = BundleModelExecutorV1::new(
            FakeExecutionV1 {
                execute_error: Some(BundleAbiError::Status(BundleStatusV1::InvalidArgument)),
                ..FakeExecutionV1::emitting(Vec::new())
            },
            VOCABULARY,
            4,
        );

        let error = executor
            .execute(&items(), &kv_plan())
            .err()
            .expect("the bundle refuses this execution");

        assert_eq!(
            error.message(),
            "bundle execute failed: bundle ABI status 1 (invalid argument); batch of 1 item(s): \
             1 prefill with 2 tokens and 0 scored prompt rows, 0 decode, 0 verify with 0 rows, \
             0 media encodes with 0 rows, 0 injected rows"
        );
    }

    #[test]
    fn more_rows_than_the_batch_bound_never_reach_the_adapter() {
        let mut executor = BundleModelExecutorV1::new(
            FakeExecutionV1::emitting(vec![
                row(REQUEST, &[1.0, 2.0, 3.0]),
                row(REQUEST, &[1.0, 2.0, 3.0]),
            ]),
            VOCABULARY,
            1,
        );

        let error = executor
            .execute(&items(), &kv_plan())
            .err()
            .expect("the adapter refuses this execution");

        assert!(error.message().contains("bundle execute failed"));
        assert!(executor.instance.calls.is_empty());
    }

    #[test]
    fn an_anchor_item_carries_its_flag_on_the_v3_wire_only() {
        let items = items()
            .into_iter()
            .map(|item| item.with_anchor(true))
            .collect::<Vec<_>>();
        let encoded = super::encode_batch_request_with_capability_v1(
            &items,
            &kv_plan(),
            super::BatchWireCapabilityV1::V3Speculative,
        )
        .expect("a V3 wire carries the anchor flag");
        let decoded = runtime_bundle_sdk::BatchCodecV1::decode(&encoded)
            .expect("the SDK reads the anchor flag");
        assert!(decoded.items()[0].anchor());
        let error = super::encode_batch_request_with_capability_v1(
            &items,
            &kv_plan(),
            super::BatchWireCapabilityV1::V2PromptRows,
        )
        .expect_err("a V2 wire has no anchor flag");
        assert!(
            error
                .message()
                .contains("anchor items require batch wire V3")
        );
    }
}

#[cfg(test)]
#[path = "executor_sampling_tests.rs"]
mod sampling_tests;

#[cfg(test)]
#[path = "fixed_batch/mod.rs"]
mod fixed_batch;
