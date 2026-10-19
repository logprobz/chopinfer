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
