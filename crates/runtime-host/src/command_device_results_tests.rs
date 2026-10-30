use super::tests::RetirementExecution;
use super::*;
use crate::executor::{BatchEmissionsV1, DeviceLogitsSamplerV1};
use runtime_bundle_abi::device_results::DeviceCycleMetadataV1;
use runtime_bundle_abi::{
    BundleAbiError, HostServiceResultV1, OwnedLogitsRowV1, OwnedLogitsViewV1, ResultSinkLimitsV1,
};
use std::cell::RefCell;

struct Execution {
    inner: RetirementExecution,
    enabled: bool,
    cycles: Rc<RefCell<Vec<Option<DeviceCycleMetadataV1>>>>,
}
impl BundleExecutionV1 for Execution {
    fn supports_device_results(&self) -> bool {
        self.enabled
    }
    fn execute_with_device_sampling(
        &mut self,
        request: &[u8],
        limits: ResultSinkLimitsV1,
        items: &[ModelBatchItem],
        sampler: Option<Rc<dyn DeviceLogitsSamplerV1>>,
        metadata: Option<DeviceCycleMetadataV1>,
    ) -> Result<BatchEmissionsV1, BundleAbiError> {
        self.cycles.borrow_mut().push(metadata);
        self.execute_with_sampling(request, limits, items, sampler)
    }
    fn execute_batch(
        &mut self,
        request: &[u8],
        limits: ResultSinkLimitsV1,
    ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
        self.inner.execute_batch(request, limits)
    }
    fn commit(&mut self) -> Result<(), BundleAbiError> {
        self.inner.commit()
    }
    fn cancel(&mut self) -> Result<(), BundleAbiError> {
        self.inner.cancel()
    }
    fn destroy(&mut self) -> Result<(), BundleAbiError> {
        self.inner.destroy()
    }
    fn is_poisoned(&self) -> bool {
        self.inner.is_poisoned()
    }
}
struct Sampler(bool);
impl DeviceLogitsSamplerV1 for Sampler {
    fn supports_device_results(&self) -> bool {
        self.0
    }
    fn sample_view(
        &self,
        _: OwnedLogitsViewV1,
        _: &[&ModelBatchItem],
    ) -> HostServiceResultV1<Vec<runtime_engine::SampledRow>> {
        panic!("the fixture returns host rows")
    }
}

#[test]
fn served_executor_preserves_cycle_metadata_only_when_both_ends_support_it() {
    for enabled in [false, true] {
        for sampling in [None, Some(false), Some(true)] {
            let cycles = Rc::new(RefCell::new(Vec::new()));
            let mut executor = BundleModelExecutorV1::new(
                Execution {
                    inner: RetirementExecution {
                        trace: Rc::new(RefCell::new(Vec::new())),
                        fail_retirement: false,
                    },
                    enabled,
                    cycles: Rc::clone(&cycles),
                },
                3,
                1,
            );
            if let Some(supported) = sampling {
                executor = executor.with_view_sampler(Rc::new(Sampler(supported)));
            }
            let executor = ServedExecutorV1 {
                executor,
                _services: Rc::new(()),
                pool_control: None,
                reference: None,
            };
            let geometry = KvGeometry::new(4, 4).unwrap();
            let vocabulary = ModelVocabulary::new(3).unwrap();
            let config = EngineConfig::new(
                SchedulerConfig::new(1, 2, 4, 4, 1).unwrap(),
                geometry,
                vocabulary,
                geometry.token_capacity(),
                1,
            )
            .unwrap();
            let mut engine = Engine::new(config, executor).unwrap();
            engine
                .submit(
                    runtime_engine::EngineRequest::new(
                        RequestId::new(7),
                        vec![0],
                        3,
                        runtime_sampling::SamplingParameters::new(0.0).unwrap(),
                        1,
                        Vec::new(),
                        vocabulary,
                    )
                    .unwrap(),
                )
                .unwrap();
            for _ in 0..4 {
                engine.step().unwrap();
            }
            let cycles = cycles.borrow();
            assert_eq!(cycles.len(), 3);
            let expected = enabled && sampling == Some(true);
            for (index, cycle) in cycles.iter().enumerate() {
                assert_eq!(
                    cycle.is_some(),
                    expected,
                    "bundle={enabled}, sampler={sampling:?}"
                );
                if let Some(cycle) = cycle {
                    assert_eq!(cycle.cycle_epoch, index as u64 + 1);
                    assert_eq!(cycle.items.len(), 1);
                    assert_eq!(cycle.items[0].request_id, 7);
                    assert_eq!(cycle.items[0].acceptance.remaining_tokens, 3 - index as u64);
                }
            }
        }
    }
}

/// A bundle that verifies allocated steps only through mapped device
/// sampling, as a served speculative package may: it refuses an allocated
/// step without it with the status a live server stopped on. Every step it
/// runs returns host rows that favour token 1, and drafts of token 1.
struct MappedAllocation {
    /// Per execute: whether the step was allocated, and whether it had
    /// mapped sampling.
    steps: Rc<RefCell<Vec<(bool, bool)>>>,
}
impl BundleExecutionV1 for MappedAllocation {
    fn supports_device_results(&self) -> bool {
        true
    }
    fn supports_speculation(&self) -> bool {
        true
    }
    fn execute_with_device_sampling(
        &mut self,
        request: &[u8],
        _: ResultSinkLimitsV1,
        items: &[ModelBatchItem],
        sampler: Option<Rc<dyn DeviceLogitsSamplerV1>>,
        metadata: Option<DeviceCycleMetadataV1>,
    ) -> Result<BatchEmissionsV1, BundleAbiError> {
        let batch = runtime_bundle_sdk::BatchCodecV1::decode(request).unwrap();
        let allocation = batch.verify_allocation();
        let mapped = metadata.is_some() && sampler.is_some_and(|s| s.supports_mapped_sampling());
        self.steps.borrow_mut().push((allocation.is_some(), mapped));
        if allocation.is_some() && !mapped {
            return Err(BundleAbiError::Status(BundleStatusV1::Unsupported));
        }
        let decoding = |item: &&ModelBatchItem| {
            matches!(item.phase(), BatchPhase::Decode | BatchPhase::Verify { .. })
        };
        let mut verified = allocation.map(|allocation| {
            let reserved = items
                .iter()
                .filter(decoding)
                .map(|item| item.draft().map_or(0, |draft| draft.tokens() as u32))
                .collect::<Vec<_>>();
            let scores = vec![[0.0_f32; 8]; reserved.len()];
            let scores = scores.iter().map(|s| &s[..]).collect::<Vec<_>>();
            runtime_bundle_sdk::allocate_verify_drafts_v1(
                allocation,
                &reserved,
                &scores,
                allocation.kinds(),
            )
            .into_iter()
        });
        let mut rows = Vec::new();
        let mut drafts = Vec::new();
        for item in items {
            let verified = verified
                .as_mut()
                .filter(|_| decoding(&item))
                .and_then(Iterator::next);
            if let Some(draft) = item.draft() {
                drafts.push(runtime_bundle_abi::OwnedDraftTokensV1 {
                    request_id: item.request_id().get(),
                    tokens: vec![1; verified.map_or(draft.tokens(), |k| k as usize)],
                });
            }
            if item.expects_logits() {
                for _ in 0..item.draft().map_or(1, |draft| draft.tokens() + 1) {
                    rows.push(OwnedLogitsRowV1::new(
                        item.request_id().get(),
                        [0.0_f32, 1.0, 0.0, 0.0]
                            .into_iter()
                            .flat_map(f32::to_le_bytes)
                            .collect(),
                    ));
                }
            }
        }
        Ok(BatchEmissionsV1::from_rows_and_drafts(rows, drafts))
    }
    fn execute_batch(
        &mut self,
        _: &[u8],
        _: ResultSinkLimitsV1,
    ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
        unreachable!("every step takes the device sampling entry point")
    }
    fn commit(&mut self) -> Result<(), BundleAbiError> {
        Ok(())
    }
    fn commit_accepted(
        &mut self,
        _: &[runtime_bundle_abi::BundleKeptRowsV1],
    ) -> Result<(), BundleAbiError> {
        Ok(())
    }
    fn cancel(&mut self) -> Result<(), BundleAbiError> {
        Ok(())
    }
    fn destroy(&mut self) -> Result<(), BundleAbiError> {
        Ok(())
    }
    fn is_poisoned(&self) -> bool {
        false
    }
}

/// A device sampler that takes mapped sampling; the fixture's rows are
/// host rows, so it never samples.
struct MappedSampler;
impl DeviceLogitsSamplerV1 for MappedSampler {
    fn supports_device_results(&self) -> bool {
        true
    }
    fn supports_mapped_sampling(&self) -> bool {
        true
    }
    fn sample_view(
        &self,
        _: OwnedLogitsViewV1,
        _: &[&ModelBatchItem],
    ) -> HostServiceResultV1<Vec<runtime_engine::SampledRow>> {
        panic!("the fixture returns host rows")
    }
}

/// Serving with confidence allocates a verify step only when its sampling
/// takes the device's mapped results. A step with a request whose
/// presence, frequency or repetition penalty reads its history gets host
/// values, so it is not allocated: before, it reached the bundle allocated
/// without mapped sampling, the bundle refused it as Unsupported, and the
/// failed engine step stopped the generation worker. Requests without
/// penalties keep their allocated steps.
#[test]
fn served_allocation_reaches_the_bundle_only_with_mapped_sampling() {
    let mut cases = 0;
    for penalties in [
        None,
        Some((0.5, 0.0, 1.0)),
        Some((0.0, 0.5, 1.0)),
        Some((0.0, 0.0, 1.2)),
    ] {
        let steps = Rc::new(RefCell::new(Vec::new()));
        let executor = BundleModelExecutorV1::new_with_batch_wire(
            MappedAllocation {
                steps: Rc::clone(&steps),
            },
            4,
            12,
            crate::executor::BatchWireCapabilityV1::V3Speculative,
        )
        .with_verify_allocation(true)
        .with_view_sampler(Rc::new(MappedSampler));
        let executor = ServedExecutorV1 {
            executor,
            _services: Rc::new(()),
            pool_control: None,
            reference: None,
        };
        let vocabulary = ModelVocabulary::new(4).unwrap();
        let config = EngineConfig::new(
            SchedulerConfig::new(2, 8, 64, 4, 1).unwrap(),
            KvGeometry::new(4, 64).unwrap(),
            vocabulary,
            64,
            32,
        )
        .unwrap()
        .with_speculation(
            SpeculationConfig::new(
                5,
                12,
                SpeculationMode::Balanced(DraftDepths::parse("1:5").unwrap()),
                DraftSource::Bundle,
            )
            .unwrap()
            .with_verify_widths(VerifyWidths::parse("6,8").unwrap())
            .with_verify_allocation(Some(
                runtime_engine::VerifyAllocation::new(1, &[(1.0, 0.0); 5], &[(1.0, 0.0); 5])
                    .unwrap(),
            )),
        );
        let mut engine = Engine::new(config, executor).unwrap();
        for id in [1, 2] {
            let mut sampling = runtime_sampling::SamplingParameters::new(0.0).unwrap();
            if id == 2
                && let Some((presence, frequency, repetition)) = penalties
            {
                sampling = sampling
                    .with_penalties(presence, frequency, repetition)
                    .unwrap();
            }
            engine
                .submit(
                    runtime_engine::EngineRequest::new(
                        RequestId::new(id),
                        vec![0, 2],
                        16,
                        sampling,
                        id,
                        Vec::new(),
                        vocabulary,
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        let mut finished = 0;
        for _ in 0..16 {
            if engine.step().unwrap() == runtime_engine::StepStatus::Idle {
                break;
            }
            finished += engine
                .drain_events()
                .into_iter()
                .filter(|event| {
                    matches!(
                        event.kind(),
                        runtime_engine::EngineEventKind::Finished { .. }
                    )
                })
                .count();
        }
        assert_eq!(finished, 2, "{penalties:?}");
        let steps = steps.borrow();
        assert!(
            !steps
                .iter()
                .any(|&(allocated, mapped)| allocated && !mapped),
            "{penalties:?}: {steps:?}"
        );
        let allocated = steps.iter().filter(|&&(allocated, _)| allocated).count();
        if penalties.is_none() {
            assert!(allocated > 0, "{steps:?}");
        } else {
            assert_eq!(allocated, 0, "{penalties:?}: {steps:?}");
        }
        cases += 1;
    }
    assert_eq!(cases, 4);
}
