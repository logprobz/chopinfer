//! One bounded allocated verify cycle, using serving's metadata and acceptance.
use super::*;
use runtime_bundle_abi::device_results::*;
use runtime_engine::{OutputCycle, StepAllocation};

#[derive(Clone, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AllocationConfig {
    pub width: usize,
    #[serde(default)]
    pub capacity: Option<usize>,
    pub confidence: crate::confidence::Calibration,
}

impl AllocationConfig {
    pub(super) fn capacity(&self, trace: &Trace) -> Result<usize> {
        let minimum = trace
            .steps
            .iter()
            .map(|step| step.rows())
            .max()
            .ok_or("empty allocation trace")?;
        let capacity = self.capacity.unwrap_or(minimum);
        if capacity < minimum {
            return Err("allocation replay capacity is below its reserved rows".into());
        }
        Ok(capacity)
    }

    fn final_step<'a>(&self, trace: &'a Trace) -> Result<&'a super::super::trace::Step> {
        let final_step = trace.steps.last().ok_or("empty allocation trace")?;
        if trace.version != transition::VERSION
            || !trace.lifecycle.is_empty()
            || trace.steps[..trace.steps.len() - 1]
                .iter()
                .any(|s| s.items.iter().any(|i| !i.prefill))
            || !matches!(
                final_step.mode,
                super::super::trace::Mode::Greedy
                    | super::super::trace::Mode::Seeded
                    | super::super::trace::Mode::Top
            )
            || final_step.items.iter().any(|i| i.prefill || i.drafts == 0)
            || self.width >= final_step.rows()
            || self.width == 0
        {
            return Err("allocation replay requires prefills followed by one final drafted step with fewer computed than reserved rows".into());
        }
        Ok(final_step)
    }

    pub(super) fn resolve(
        &self,
        trace: &Trace,
        manifest: &runtime_bundle::BundleManifestV1,
    ) -> Result<StepAllocation> {
        let final_step = self.final_step(trace)?;
        let allocation = self.confidence.check()?.resolve(manifest, true)?;
        let minimum: usize = final_step
            .items
            .iter()
            .map(|item| 1 + item.drafts.min(allocation.floor()))
            .sum();
        if self.width < minimum
            || final_step
                .items
                .iter()
                .any(|item| item.drafts > allocation.positions())
        {
            return Err(
                "allocation replay width or calibration does not cover its reservations".into(),
            );
        }
        Ok(StepAllocation {
            width: self.width,
            allocation,
        })
    }
}

pub(super) fn cycle_items(items: Vec<ModelBatchItem>, epoch: u64) -> Vec<ModelBatchItem> {
    items
        .into_iter()
        .map(|item| {
            item.with_output_cycle(OutputCycle {
                epoch,
                remaining_tokens: 256,
                termination_stop_ids: vec![],
            })
        })
        .collect()
}

pub(super) fn accepted_step(
    step: &super::super::trace::Step,
    accepted: &[runtime_engine::AcceptedPrefix],
    width: usize,
) -> Result<super::super::trace::Step> {
    if accepted.iter().map(|a| a.drafted + 1).sum::<usize>() != width {
        return Err("allocated acceptance does not fill the computed width".into());
    }
    if accepted.len() != step.items.len() {
        return Err("missing allocated acceptance".into());
    }
    let mut result = step.clone();
    let mut seen = BTreeSet::new();
    for acceptance in accepted {
        let id = acceptance.request_id.get();
        let item = result
            .items
            .iter_mut()
            .find(|item| item.slot as u64 + 1 == id)
            .ok_or("unexpected allocated acceptance")?;
        if !seen.insert(id)
            || acceptance.empty_row.is_some()
            || acceptance.accepted > acceptance.drafted
            || acceptance.drafted > item.drafts
            || acceptance.terminal
            || acceptance.emitted != acceptance.accepted + 1
            || item.prefill
        {
            return Err("invalid allocated acceptance".into());
        }
        item.kept = 1 + acceptance.accepted;
    }
    Ok(result)
}

pub(super) fn execute_replay(
    instance: &mut impl BundleExecutionV1,
    request: &[u8],
    limits: ResultSinkLimitsV1,
    items: &[ModelBatchItem],
    sampler: Option<Rc<dyn DeviceLogitsSamplerV1>>,
    metadata: Option<DeviceCycleMetadataV1>,
) -> std::result::Result<crate::executor::BatchEmissionsV1, runtime_bundle_abi::BundleAbiError> {
    if metadata.is_some() {
        instance.execute_with_device_sampling(request, limits, items, sampler, metadata)
    } else {
        instance.execute_with_sampling(request, limits, items, sampler)
    }
}

impl RecordingSampler {
    pub(super) fn capture_drafts(&self, drafts: BundleDeviceDraftsV1) -> HostServiceResultV1<()> {
        let bytes = self.read_ready(drafts.proposals)?;
        let tokens = (0..drafts.requested_tokens as usize)
            .map(|i| {
                let at = i * drafts.token_stride_bytes as usize;
                let word = bytes
                    .get(at..at + 4)
                    .ok_or(BundleStatusV1::InvalidArgument)?;
                Ok(u32::from_le_bytes(word.try_into().unwrap()))
            })
            .collect::<HostServiceResultV1<Vec<_>>>()?;
        self.device_drafts
            .borrow_mut()
            .push((drafts.request_id, tokens));
        Ok(())
    }
    fn read_ready(&self, view: BundleReadyDeviceViewV1) -> HostServiceResultV1<Vec<u8>> {
        view.validate(1)?;
        let mut bytes = Vec::new();
        let mut offset = 0;
        // Diagnostic logits can exceed the completion transport's packet bound.
        // Each chunk keeps the producer event and is collected before slot reuse.
        while offset < view.buffer.length {
            let length = (view.buffer.length - offset)
                .min(crate::sampling_readback::MAX_PACKET_BYTES as u64);
            let chunk = BundleReadyDeviceViewV1 {
                buffer: runtime_bundle_abi::BundleDeviceBufferViewV1 {
                    offset: view.buffer.offset + offset,
                    length,
                    ..view.buffer
                },
                ..view
            };
            let ticket = self.backend.enqueue_device_readback(chunk)?;
            bytes.extend(self.backend.collect_device_readback(chunk, ticket)?);
            offset += length;
        }
        Ok(bytes)
    }

    pub(super) fn capture_ready(
        &self,
        source: BundleReadyLogitsViewV1,
        first: u32,
        items: &[&ModelBatchItem],
        dispatch: SamplingDispatch,
        map: Option<runtime_bundle_abi::BundleDeviceBufferViewV1>,
    ) -> HostServiceResultV1<()> {
        if let Some(budget) = &self.budget {
            budget
                .borrow_mut()
                .reserve(
                    source.logits.element_type,
                    source.logits.live_width,
                    items.len(),
                )
                .map_err(refusal)?;
        }
        let ready = |buffer| BundleReadyDeviceViewV1 {
            context: source.logits.context,
            rank: source.logits.rank,
            reserved: 0,
            buffer,
            ready: source.ready,
        };
        let bytes = self.read_ready(ready(source.logits.buffer))?;
        let indices = map.map(|view| self.read_ready(ready(view))).transpose()?;
        let element = match source.logits.element_type {
            runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32 => 4,
            runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_BF16 => 2,
            _ => return Err(BundleStatusV1::InvalidArgument),
        };
        let width = source.logits.live_width as usize * element;
        let stride = source.logits.row_stride_elements as usize * element;
        for (row, item) in items.iter().enumerate() {
            let physical = if let Some(indices) = &indices {
                let word = indices
                    .get(row * 4..row * 4 + 4)
                    .ok_or(BundleStatusV1::InvalidArgument)?;
                u32::from_le_bytes(word.try_into().unwrap()) as usize
            } else {
                row
            };
            if physical >= source.logits.row_count as usize {
                return Err(BundleStatusV1::InvalidArgument);
            }
            let values = bytes
                .get(physical * stride..physical * stride + width)
                .ok_or(BundleStatusV1::InvalidArgument)?;
            self.capture
                .borrow_mut()
                .insert(
                    first as usize + row,
                    CapturedRow {
                        rank: source.logits.rank,
                        element_type: source.logits.element_type,
                        width: source.logits.live_width,
                        bytes: values.to_vec(),
                        metadata: metadata(item, dispatch),
                        sample: None,
                    },
                )
                .map_err(refusal)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocation_replay_refuses_unrecorded_continuations_and_history_dependent_sampling() {
        let config = AllocationConfig {
            width: 16,
            capacity: Some(64),
            confidence: crate::confidence::Calibration {
                contract: "fixture.confidence.v1".into(),
                floor: 1,
                greedy: vec![[1.0, 0.0]; 5],
                sampled: vec![[1.0, 0.0]; 5],
            },
        };
        let mut trace = Trace::construct_fixed_depth(vec![vec![7; 16]; 4], 1024, 5);
        trace.version = transition::VERSION;
        trace.steps.truncate(2);
        for step in &mut trace.steps {
            for item in &mut step.items {
                item.history.clear();
            }
        }
        trace.validate().unwrap();
        assert!(config.final_step(&trace).is_ok());
        assert_eq!(config.capacity(&trace).unwrap(), 64);
        let mut smaller = config.clone();
        smaller.capacity = Some(23);
        assert!(smaller.capacity(&trace).is_err());
        smaller.capacity = None;
        assert_eq!(smaller.capacity(&trace).unwrap(), 24);
        let mut bad = trace.clone();
        bad.steps.push(bad.steps[1].clone());
        assert!(config.final_step(&bad).is_err());
        for mode in [
            super::super::super::trace::Mode::Grammar,
            super::super::super::trace::Mode::Filtered,
            super::super::super::trace::Mode::Mixed,
        ] {
            bad = trace.clone();
            bad.steps[1].mode = mode;
            assert!(config.final_step(&bad).is_err());
        }
        for width in [0, 24, 25] {
            let mut bad = config.clone();
            bad.width = width;
            assert!(bad.final_step(&trace).is_err());
        }
    }

    #[test]
    fn device_acceptance_replaces_prescribed_kept_counts_without_changing_inputs() {
        let trace = Trace::construct_fixed_depth(vec![vec![7; 16]; 4], 1024, 5);
        let step = trace
            .steps
            .iter()
            .find(|s| s.items.iter().any(|i| i.drafts > 0))
            .unwrap();
        let accepted = step
            .items
            .iter()
            .enumerate()
            .map(|(i, item)| runtime_engine::AcceptedPrefix {
                request_id: RequestId::new(item.slot as u64 + 1),
                cycle_epoch: 1,
                drafted: 3,
                accepted: i % 3,
                emitted: i % 3 + 1,
                terminal: false,
                rejected: true,
                empty_row: None,
            })
            .collect::<Vec<_>>();
        let actual = accepted_step(step, &accepted, 16).unwrap();
        for (index, item) in actual.items.iter().enumerate() {
            assert_eq!(item.kept, index % 3 + 1);
            assert_eq!(item.inputs, step.items[index].inputs);
            assert_eq!(item.history, step.items[index].history);
        }
        assert!(accepted_step(step, &accepted[..3], 16).is_err());
        let mut duplicate = accepted.clone();
        duplicate[1] = duplicate[0];
        assert!(accepted_step(step, &duplicate, 16).is_err());
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;
    use crate::executor::BatchEmissionsV1;
    use runtime_bundle_abi::{BundleAbiError, OwnedLogitsRowV1};

    #[derive(Default)]
    struct Host {
        calls: Vec<(Vec<u8>, Option<DeviceCycleMetadataV1>)>,
    }
    impl BundleExecutionV1 for Host {
        fn execute_batch(
            &mut self,
            request: &[u8],
            _: ResultSinkLimitsV1,
        ) -> std::result::Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
            self.calls.push((request.to_vec(), None));
            Ok(vec![OwnedLogitsRowV1::new(1, vec![1, 2, 3, 4])])
        }
        fn execute_with_device_sampling(
            &mut self,
            request: &[u8],
            limits: ResultSinkLimitsV1,
            items: &[ModelBatchItem],
            sampler: Option<Rc<dyn DeviceLogitsSamplerV1>>,
            metadata: Option<DeviceCycleMetadataV1>,
        ) -> std::result::Result<BatchEmissionsV1, BundleAbiError> {
            let rows = self.execute_with_sampling(request, limits, items, sampler)?;
            self.calls.last_mut().unwrap().1 = metadata;
            Ok(rows)
        }
        fn commit(&mut self) -> std::result::Result<(), BundleAbiError> {
            Ok(())
        }
        fn cancel(&mut self) -> std::result::Result<(), BundleAbiError> {
            Ok(())
        }
        fn destroy(&mut self) -> std::result::Result<(), BundleAbiError> {
            Ok(())
        }
        fn is_poisoned(&self) -> bool {
            false
        }
    }

    #[test]
    fn default_replay_keeps_legacy_wire_calls_and_outputs() {
        let trace = Trace::construct_fixed_depth(vec![vec![7; 16]], 1024, 1);
        let items = trace.steps[3].model_items_at_temperature(None, 1.0);
        let mut kv =
            runtime_kv::KvPool::new(runtime_kv::KvGeometry::new(4, 32).unwrap(), 1).unwrap();
        let handle = kv.create_sequence(64).unwrap();
        let plan = kv.prepare_append_batch(&[(handle, 2)]).unwrap();
        let wire = BatchWireCapabilityV1::V3MediaSpeculative;
        for sampling in [false, true] {
            let mut original = Vec::new();
            crate::executor::write_draft_sampling_v1(
                &mut original,
                &items,
                sampling,
                |out, trailing| write_batch_request_v1(out, &items, &plan, wire, trailing),
            )
            .unwrap();
            let mut current = Vec::new();
            write_replay_request(&mut current, &items, &plan, wire, None, sampling).unwrap();
            assert_eq!(current, original);
            let limits = ResultSinkLimitsV1::new(2, 1024, 2048);
            let mut host = Host::default();
            let expected = host
                .execute_with_sampling(&original, limits, &items, None)
                .unwrap();
            let actual = execute_replay(&mut host, &current, limits, &items, None, None).unwrap();
            assert_eq!(actual.rows, expected.rows);
            assert_eq!(host.calls[0], host.calls[1]);
            assert!(host.calls[1].1.is_none());
            let items = cycle_items(items.clone(), 2);
            let metadata = crate::executor::cycle_metadata(&items, None).unwrap();
            execute_replay(
                &mut host,
                &current,
                limits,
                &items,
                None,
                Some(metadata.clone()),
            )
            .unwrap();
            assert_eq!(host.calls[2].1.as_ref(), Some(&metadata));
        }
    }
}

#[cfg(all(test, target_os = "linux", feature = "cuda-fake-driver-tests"))]
mod readback_tests {
    use super::*;
    use crate::services::{HostInstanceConfigV1, HostTopologyV1};
    use runtime_bundle_abi::{BundleDeviceBufferViewV1, DeviceBufferSpanV1, HostServicesBackendV1};
    use runtime_cuda::fake_driver;

    #[test]
    fn recorder_reads_large_logits_through_bounded_transport() {
        fake_driver::reset();
        let backend = Rc::new(Backend::new(
            crate::cuda::CudaHostDeviceV1::new().unwrap(),
            HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).unwrap()),
        ));
        let context = backend.select_context(0, 0).unwrap();
        let pool = backend.select_memory_pool(context).unwrap();
        let stream = backend.create_stream(context, 0).unwrap();
        let bytes = (0..3 * (1 << 20) + 37)
            .map(|i| (i % 251) as u8)
            .collect::<Vec<_>>();
        let buffer = backend
            .allocate_device_buffer(pool, stream, bytes.len() as u64, 8)
            .unwrap();
        backend
            .copy_host_to_device(
                stream,
                DeviceBufferSpanV1::new(buffer, 0, bytes.len() as u64),
                &bytes,
            )
            .unwrap();
        let ready = backend.create_event(context).unwrap();
        backend.record_event(ready, stream).unwrap();
        let view = BundleReadyDeviceViewV1 {
            context,
            rank: 0,
            reserved: 0,
            buffer: BundleDeviceBufferViewV1 {
                buffer,
                offset: 19,
                length: (bytes.len() - 19) as u64,
                ..BundleDeviceBufferViewV1::empty()
            },
            ready,
        };
        // Exercise the same transport refusal as a real device before recording.
        assert_eq!(
            backend.enqueue_device_readback(view),
            Err(BundleStatusV1::InvalidArgument)
        );
        let recorder = RecordingSampler {
            backend,
            capture: RefCell::default(),
            budget: None,
            device_results: true,
            device_drafts: RefCell::default(),
        };
        assert_eq!(recorder.read_ready(view).unwrap(), bytes[19..]);
        let small = BundleReadyDeviceViewV1 {
            buffer: BundleDeviceBufferViewV1 {
                length: 20,
                ..view.buffer
            },
            ..view
        };
        assert_eq!(recorder.read_ready(small).unwrap(), bytes[19..39]);
        assert_eq!(fake_driver::ordering_violations(), 0);
    }
}
