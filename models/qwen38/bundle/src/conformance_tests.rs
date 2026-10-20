//! Shared conformance scenarios over the existing CPU host.
use super::*;
use bundle_runtime::fake_host::conformance::{self, Adapter, CancellationPolicy, ObservedHost};

struct Limits;
impl SinkLimits for Limits {
    fn limits() -> ResultSinkLimitsV1 {
        ViewSink::default().limits()
    }
}
struct Model(Rc<FakeHost>, Rc<ObservedHost<FakeHost>>);

impl Adapter for Model {
    type Startup = Served;
    type Observer = lifecycle::Recorder;
    type Limits = Limits;
    type Host = ObservedHost<FakeHost>;

    fn new() -> Self {
        let host = Rc::new(FakeHost::at(Shape::TP2));
        host.served.set(Some(1));
        host.speculation.set((7, 32));
        Self(Rc::clone(&host), Rc::new(ObservedHost::new(host)))
    }
    fn host(&self) -> Rc<Self::Host> {
        Rc::clone(&self.1)
    }
    fn device(&self) -> &FakeDevice<lifecycle::Recorder> {
        &self.0.device
    }
    fn api(&self) -> BundleApiV1 {
        api()
    }
    fn configuration(&self) -> Vec<u8> {
        configuration_served(Shape::TP2, self.0.layout(), &[], 1)
    }
    fn request(&self, request: u64, start: u64, tokens: &[u32], verify: bool) -> Vec<u8> {
        if !verify {
            return step(&[(request, 0, start, tokens)]).encode().unwrap();
        }
        let rows = tokens.len() as u32;
        let end = start + u64::from(rows);
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::verify(request, rows, false, tokens.to_vec())],
            vec![BatchAppendV1::new(
                start,
                u64::from(rows),
                end,
                (0..end.div_ceil(PAGE_TOKENS)).collect(),
                None,
            )],
        )
        .with_state_slots(vec![0])
        .encode()
        .unwrap()
    }
    fn raw_commit(&self, handle: BundleHandleV1, records: &[BundleKeptRowsV1]) -> BundleStatusV1 {
        bundle_runtime::entry::commit_accepted::<crate::instance::Instance>(handle, records)
    }
    fn assert_released(&self) {
        self.1.accounting.assert_released();
        for id in self.0.buffer_order.borrow().iter() {
            assert!(
                !self.0.extents.borrow().contains_key(id),
                "allocated buffer still live"
            );
        }
    }
    fn release_count(&self) -> usize {
        self.1.accounting.releases()
    }
    fn callback_count(&self) -> usize {
        self.1.accounting.calls()
    }
    fn cancellation_policy(&self) -> CancellationPolicy {
        CancellationPolicy::InvalidateTouched
    }
    fn arm_cancel_failure(&self) {
        self.0.fail_restore_rank.set(Some(0));
    }
    fn assert_cancel_failure_consumed(&self) {
        assert_eq!(self.0.fail_restore_rank.get(), None);
    }
    fn assert_poisoned_prefix_calls(&self, instance: &mut BundleInstanceV1) {
        assert_eq!(
            instance.describe_prefix_state(),
            Err(BundleAbiError::Poisoned)
        );
        assert_eq!(
            instance.capture_prefix_state(7, 3, &[100]),
            Err(BundleAbiError::Poisoned)
        );
        assert_eq!(
            instance.restore_prefix_state(8, &[1, 2, 3], &[100]),
            Err(BundleAbiError::Poisoned)
        );
        assert_eq!(
            instance.restore_prefix_state_in_slot(8, &[1, 2, 3], &[100], 0),
            Err(BundleAbiError::Poisoned)
        );
    }
}

conformance::suite! {
    startup: Served {
        tp4_ar: "tp4-ar",
        tp4_mtp: "tp4-mtp",
        tp4_ar_dcp2: "tp4-ar-dcp2",
        tp4_mtp_dcp2: "tp4-mtp-dcp2",
        tp2_ar: "tp2-ar",
        tp2_mtp: "tp2-mtp",
    }
    allocation_release: Model => AllocationRelease;
    malformed_commit: Model => MalformedCommit;
    poisoned_client: Model => PoisonedClient;
    cancellation_policy: Model => Cancellation;
    retire_and_slot_reuse: Model => SlotReuse;
}

/// One serving configuration per profile in profiles.json, over a host
/// serving its sequences, with the MTP drafter when the profile drafts and
/// the vision encoder when it serves images. A context-parallel profile's
/// instance is created with create's refusal switch lifted on the test's
/// thread, so the conformance suite holds whether `context_parallel::SERVED`
/// is on or off.
struct Served(
    Rc<FakeHost>,
    Rc<ObservedHost<FakeHost>>,
    conformance::Serving,
);

impl conformance::Startup for Served {
    type Limits = speculation::SpeculativeLimits;

    fn profiles() -> &'static str {
        include_str!("../../profiles.json")
    }
    fn servings() -> Vec<conformance::Serving> {
        let serving = |profile: &str, shape: Shape, sequences: u32, drafts: u32| {
            conformance::Serving {
                profile: profile.to_owned(),
                tensor_parallel: shape.ranks(),
                context_parallel: if profile.ends_with("-dcp2") { 2 } else { 1 },
                sequences,
                drafter: if drafts > 0 { "mtp" } else { "none" }.to_owned(),
                draft_tokens: drafts,
                prefix_cache: true,
                model_options: Default::default(),
                // The served checkpoints store the vision tower
                // (compiler `has_vision_tower`), so `auto` serves images.
                provides: vec!["text".to_owned(), "image".to_owned()],
                selection: None,
                // The package's declaration (compiler `package.rs`) without
                // wide rows: the MTP drafter drafts in the bundle.
                package: Some(conformance::PackageSpeculation {
                    maximum_draft_tokens: model_qwen38_contract::contract::MAX_DRAFT_TOKENS,
                    bundle_drafter: drafts > 0,
                    maximum_verify_tokens: MAX_VERIFY_TOKENS,
                    row_buckets: model_qwen38_contract::contract::VERIFY_ROW_BUCKETS.to_vec(),
                    maximum_draft_items: Some(model_qwen38_contract::contract::DRAFT_ITEMS),
                    narrow_rows: true,
                    exact_graphs: false,
                }),
            }
        };
        vec![
            serving("tp4-ar", Shape::TP4, 32, 0),
            serving("tp4-mtp", Shape::TP4, 32, 3),
            serving("tp4-ar-dcp2", Shape::TP4, 32, 0),
            serving("tp4-mtp-dcp2", Shape::TP4, 32, 3),
            serving("tp2-ar", Shape::TP2, 32, 0),
            serving("tp2-mtp", Shape::TP2, 32, 3),
        ]
    }
    fn new(serving: &conformance::Serving) -> Self {
        let shape = [Shape::TP2, Shape::TP4]
            .into_iter()
            .find(|shape| shape.ranks() == serving.tensor_parallel)
            .expect("a Qwen topology");
        crate::context_parallel::serve_in_tests(serving.context_parallel > 1);
        let host = Rc::new(FakeHost::at(shape));
        host.served.set(Some(serving.sequences));
        // The package places the n-gram table by the capacity rule alone.
        host.ngram_host.set(ngram_in_host_memory(shape));
        if serving.draft_tokens > 0 {
            host.drafter.set(true);
            host.speculation
                .set((serving.draft_tokens, serving.verify_rows()));
        }
        Self(
            Rc::clone(&host),
            Rc::new(ObservedHost::new(host)),
            serving.clone(),
        )
    }
    fn create(&self) -> Result<BundleInstanceV1, BundleAbiError> {
        let host = &self.0;
        let encoder = if self.2.serves_images() {
            super::encoder::encoder_roles(host.shape().ranks())
        } else {
            Vec::new()
        };
        let mut record = configuration_served(host.shape(), host.layout(), &encoder, host.served());
        if self.2.context_parallel > 1 {
            record.extend_from_slice(runtime_bundle_sdk::context_parallel::TAG);
            record.extend_from_slice(&self.2.context_parallel.to_le_bytes());
        }
        let instance = api().create(HostServicesTableV1::new(Rc::clone(&self.1)), &record)?;
        host.model(instance.handle());
        Ok(instance)
    }
    /// Four KV pages per slot, so every served slot fits the fake pool
    /// (the shared helpers give each slot 64), each a stripe of the
    /// context-parallel group's pages.
    fn step(&self, items: &[conformance::Item]) -> Vec<u8> {
        let size = u64::from(self.2.context_parallel);
        speculation::batch(
            items
                .iter()
                .map(|item| {
                    let rows = item.tokens.len() as u32;
                    let end = item.start + u64::from(rows);
                    let pages = (0..end.div_ceil(PAGE_TOKENS * size))
                        .map(|page| item.slot * 4 + page)
                        .collect();
                    let batch = match item.phase {
                        conformance::Phase::Verify => {
                            BatchItemV1::verify(item.request, rows, true, item.tokens[..1].to_vec())
                        }
                        phase => BatchItemV1::new(
                            item.request,
                            phase == conformance::Phase::Decode,
                            rows,
                            true,
                            item.tokens.clone(),
                        ),
                    };
                    (
                        batch,
                        BatchAppendV1::new(item.start, u64::from(rows), end, pages, None),
                        item.slot,
                    )
                })
                .collect(),
        )
        .encode()
        .unwrap()
    }
    /// The fake drafter's tokens, so every draft is right.
    fn token(&self, position: u64) -> u32 {
        speculation::truth(position)
    }
    fn assert_released(&self) {
        self.1.accounting.assert_released();
        for id in self.0.buffer_order.borrow().iter() {
            assert!(
                !self.0.extents.borrow().contains_key(id),
                "{}: allocated buffer still live",
                self.2.profile
            );
        }
    }
}

/// Every startup host places the n-gram table where the served package
/// does (compiler `package.rs`), with or without the drafter.
#[test]
fn startup_hosts_place_the_ngram_table_as_the_package_does() {
    use conformance::Startup;
    for serving in Served::servings() {
        let served = Served::new(&serving);
        assert_eq!(
            served.0.ngram_host.get(),
            ngram_in_host_memory(served.0.shape()),
            "{}: n-gram placement",
            serving.profile
        );
    }
}
