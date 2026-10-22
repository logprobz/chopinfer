//! Installed-package lifecycle fixture. Shared host APIs own every device call.
use super::*;
use runtime_bundle_sdk::{
    BatchAppendCopyV2, BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1,
};
use runtime_kv::{KvGeometry, KvPool, PreparedAppendId, SequenceHandle};
use std::collections::BTreeMap;
use std::rc::Rc;

type Backend = runtime_host::HostServicesV1Backend<runtime_host::cuda::CudaHostDeviceV1>;
const REQUEST_PAGES: usize = 128;
const DEVICE_PAGES: u64 = 512;
const QUOTA: usize = 512;

pub struct Fixture {
    pub instance: LoadedBundleInstanceV1,
    backend: Rc<Backend>,
    pool: KvPool,
    vocabulary: u32,
    open: Option<(PreparedAppendId, SequenceHandle, u64, bool)>,
}

#[derive(Debug)]
pub struct Capture {
    pub drafts: Vec<u32>,
    pub rows: Vec<Vec<u8>>,
}

struct Sink {
    backend: Rc<Backend>,
    vocabulary: u32,
    drafts: Vec<u32>,
    rows: BTreeMap<u32, Vec<u8>>,
}
impl LogitsRowSinkV1 for Sink {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(
            32,
            u64::from(self.vocabulary),
            32 * u64::from(self.vocabulary),
        )
    }
    fn emit_row(&mut self, _: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    fn emit_draft_tokens(&mut self, row: OwnedDraftTokensV1) -> HostServiceResultV1<()> {
        if !self.rows.is_empty() || !self.drafts.is_empty() {
            return Err(BundleStatusV1::InvalidState);
        }
        self.drafts = row.tokens;
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
        result.and(released)
    }
}

fn service<T>(result: HostServiceResultV1<T>) -> Result<T> {
    result.map_err(|e| format!("host service: {e:?}").into())
}

impl Fixture {
    pub fn new(config: &Config) -> Result<Self> {
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
        speculation.maximum_verify_tokens = 32;
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
            }),
            speculation,
            Some(4),
            None,
        )?;
        if backend.serving_kv_pages().is_none_or(|n| n < DEVICE_PAGES) {
            return Err("the fixture requires its reserved snapshot pages".into());
        }
        Ok(Self {
            backend,
            instance,
            vocabulary,
            pool: KvPool::new(KvGeometry::new(64, REQUEST_PAGES)?, 4)?,
            open: None,
        })
    }

    pub fn sequence(&mut self) -> Result<SequenceHandle> {
        Ok(self.pool.create_sequence(QUOTA)?)
    }
    pub fn restore(
        &mut self,
        source: SequenceHandle,
        request: u64,
        slot: u64,
        ids: &[u32],
        pages: &[u64],
    ) -> Result<SequenceHandle> {
        let child = self.pool.fork(source, QUOTA)?;
        self.instance
            .restore_prefix_state_in_slot(request, ids, pages, slot)?;
        Ok(child)
    }
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
    pub fn execute(
        &mut self,
        handle: SequenceHandle,
        request: u64,
        slot: u64,
        tokens: &[u32],
        prefill: bool,
        drafts: u32,
    ) -> Result<Capture> {
        if self.open.is_some() {
            return Err("fixture step is already open".into());
        }
        let rows = if drafts > 0 {
            drafts as usize + 1
        } else {
            tokens.len()
        };
        let plan = self.pool.prepare_append_batch(&[(handle, rows)])?;
        service(self.backend.prepare_serving_batch(&plan))?;
        let append = &plan.appends()[0];
        let item = if drafts > 0 {
            BatchItemV1::verify(request, rows as u32, true, tokens.to_vec())
        } else {
            BatchItemV1::new(request, !prefill, rows as u32, true, tokens.to_vec())
        };
        let copy = append.copy().map(|copy| {
            BatchAppendCopyV2::new(
                copy.source().index() as u64,
                copy.destination().index() as u64,
                copy.initialized_tokens() as u64,
            )
        });
        let wire = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![item],
            vec![BatchAppendV1::new(
                append.previous_tokens() as u64,
                append.additional_tokens() as u64,
                append.new_tokens() as u64,
                append
                    .block_table()
                    .iter()
                    .map(|page| page.index() as u64)
                    .collect(),
                copy,
            )],
        )
        .with_state_slots(vec![slot])
        .encode()?;
        self.open = Some((plan.id(), handle, request, drafts > 0));
        let mut sink = Sink {
            backend: Rc::clone(&self.backend),
            vocabulary: self.vocabulary,
            drafts: Vec::new(),
            rows: BTreeMap::new(),
        };
        self.instance.execute(&wire, &mut sink).map_err(|e| {
            format!(
                "execute request {request}: {e:?}, {:?}",
                self.instance.last_execute_refusal()
            )
        })?;
        let expected = if prefill { 1 } else { rows };
        if sink.rows.keys().copied().collect::<Vec<_>>() != (0..expected as u32).collect::<Vec<_>>()
            || sink.drafts.len() != drafts as usize
        {
            return Err("fixture got incomplete or unordered outputs".into());
        }
        Ok(Capture {
            drafts: sink.drafts,
            rows: sink.rows.into_values().collect(),
        })
    }
    pub fn commit(&mut self, count: usize) -> Result<()> {
        let (id, handle, request, verify) = self.open.ok_or("fixture has no open step")?;
        let kept = if verify {
            vec![(handle, count)]
        } else {
            Vec::new()
        };
        let commit = self.pool.preflight_commit_prepared_kept(id, &kept, &[])?;
        if verify {
            self.instance.commit_accepted(&[BundleKeptRowsV1 {
                struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
                reserved: 0,
                request_id: request,
                kept_rows: count as u32,
                reserved_tail: 0,
            }])?
        } else {
            self.instance.commit()?
        }
        self.pool.apply_prepared_commit(commit)?;
        self.open = None;
        Ok(())
    }
    pub fn cancel(&mut self) -> Result<()> {
        let (id, _, _, _) = self.open.ok_or("fixture has no open step")?;
        self.instance.cancel()?;
        self.pool.rollback_prepared(id)?;
        self.open = None;
        Ok(())
    }
    pub fn retire(&mut self, request: u64, handle: SequenceHandle) -> Result<()> {
        self.instance.retire_request(request)?;
        self.pool.release(handle)?;
        Ok(())
    }
    pub fn finish(mut self, source: SequenceHandle) -> Result<()> {
        if self.open.is_some() {
            return Err("fixture ended with an open transaction".into());
        }
        self.pool.release(source)?;
        self.instance.destroy()?;
        if self.backend.live_resource_count() != 0 {
            return Err("device resources remain live".into());
        }
        Ok(())
    }
}
