//! Shared conformance scenarios over the existing CPU host.
use super::*;
use bundle_runtime::fake_host::NoObserver;
use bundle_runtime::fake_host::conformance::{self, Adapter, CancellationPolicy, ObservedHost};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

struct Model(Rc<FakeHost>, Rc<ObservedHost<FakeHost>>);

impl Adapter for Model {
    type Startup = Served;
    type Observer = NoObserver;
    type Limits = Limits;
    type Host = ObservedHost<FakeHost>;

    fn new() -> Self {
        let host = Rc::new(FakeHost::at(Tp::TP2));
        host.served.set(1);
        host.speculation.set(Some((7, 32)));
        Self(Rc::clone(&host), Rc::new(ObservedHost::new(host)))
    }
    fn host(&self) -> Rc<Self::Host> {
        Rc::clone(&self.1)
    }
    fn device(&self) -> &FakeDevice {
        &self.0.device
    }
    fn api(&self) -> BundleApiV1 {
        BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap()
    }
    fn configuration(&self) -> Vec<u8> {
        configuration_serving(Tp::TP2, self.0.kv(), false, 1)
    }
    fn request(&self, request: u64, start: u64, tokens: &[u32], verify: bool) -> Vec<u8> {
        let rows = tokens.len() as u32;
        let item = if verify {
            BatchItemV1::verify(request, rows, false, tokens.to_vec())
        } else {
            BatchItemV1::new(request, start > 0 && rows == 1, rows, true, tokens.to_vec())
        };
        let end = start + u64::from(rows);
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![item],
            vec![BatchAppendV1::new(
                start,
                u64::from(rows),
                end,
                (0..end.div_ceil(PAGE_TOKENS)).collect(),
                None,
            )],
        )
        .encode()
        .unwrap()
    }
    fn raw_commit(&self, handle: BundleHandleV1, records: &[BundleKeptRowsV1]) -> BundleStatusV1 {
        bundle_runtime::entry::commit_accepted::<crate::instance::Instance>(handle, records)
    }
    fn assert_released(&self) {
        self.1.accounting.assert_released();
    }
    fn release_count(&self) -> usize {
        self.1.accounting.releases()
    }
    fn callback_count(&self) -> usize {
        self.1.accounting.calls()
    }
    fn cancellation_policy(&self) -> CancellationPolicy {
        CancellationPolicy::PreserveCommitted
    }
    fn arm_cancel_failure(&self) {
        self.0.fail_copy.set(Some(0));
    }
    fn assert_cancel_failure_consumed(&self) {
        assert_eq!(self.0.fail_copy.get(), None);
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
    }
}

conformance::suite! {
    startup: Served {
        tp4_ar: "tp4-ar",
        tp4_mtp: "tp4-mtp",
        tp4_dflash: "tp4-dflash",
        tp2_ar: "tp2-ar",
    }
    allocation_release: Model => AllocationRelease;
    malformed_commit: Model => MalformedCommit;
    poisoned_client: Model => PoisonedClient;
    cancellation_policy: Model => Cancellation;
    retire_and_slot_reuse: Model => SlotReuse;
}

/// One serving configuration per profile in profiles.json, over a host
/// serving its sequences, with the profile's MTP or DFlash drafter and the
/// encoders of the host's default modalities.
struct Served(
    Rc<FakeHost>,
    Rc<ObservedHost<FakeHost>>,
    conformance::Serving,
);

impl conformance::Startup for Served {
    type Observer = bundle_runtime::fake_host::NoObserver;

    fn envelope(serving: &conformance::Serving) -> conformance::Envelope {
        let mut envelope =
            conformance::Envelope::served(serving, model_mimo26_contract::serving::PREFILL_TOKENS);
        envelope.chunk_offsets = vec![8192, 16384, 32768, 40960];
        envelope.probes = vec![446, 789, 1803, 3658, 4542, 7754];
        envelope
    }
    fn route_tags(&self) -> Vec<kernel_core::route::RouteTag> {
        crate::sm120::backend(self.0.tp(), crate::switches::Switches::configured(), true)
            .expect("registered kernels")
            .route_tags()
    }
    fn finish_route_case(&self) {
        // Test-only row diagnostics have no consumer in an envelope sweep.
        drop(crate::pipeline::take_recorded());
        assert!(
            self.0.host_copy_offsets.borrow().is_empty(),
            "route sweep retained host_copy_offsets"
        );
        assert!(
            self.0.latest_uploads.borrow().is_empty(),
            "route sweep retained latest_uploads"
        );
        assert!(
            self.0.host_copies.borrow().is_empty(),
            "route sweep retained host_copies"
        );
        assert!(
            self.0.device_copies.borrow().is_empty(),
            "route sweep retained device_copies"
        );
        assert!(
            self.0.copies.borrow().is_empty(),
            "route sweep retained copies"
        );
        assert!(
            self.0.stream_launches.borrow().is_empty(),
            "route sweep retained stream_launches"
        );
        assert!(
            self.0.event_ops.borrow().is_empty(),
            "route sweep retained event_ops"
        );
        assert!(
            self.0.launch_records.borrow().is_empty(),
            "route sweep retained launch_records"
        );
        assert!(
            self.0.launch_symbols.borrow().is_empty(),
            "route sweep retained launch_symbols"
        );
        assert!(
            self.0.graph_launches.borrow().is_empty(),
            "route sweep retained graph_launches"
        );
        assert!(
            self.0.gather_launches.borrow().is_empty(),
            "route sweep retained gather_launches"
        );
        assert!(
            self.0.retain_launches.borrow().is_empty(),
            "route sweep retained retain_launches"
        );
        assert!(
            self.0.prefix_copies.borrow().is_empty(),
            "route sweep retained prefix_copies"
        );
        assert!(
            self.0.dflash_copy_outputs.borrow().is_empty(),
            "route sweep retained dflash_copy_outputs"
        );
        self.0.device.clear_route_journal();
    }
    fn route_device(&self) -> &bundle_runtime::fake_host::FakeDevice<Self::Observer> {
        &self.0.device
    }
    type Limits = Limits;

    fn profiles() -> &'static str {
        include_str!("../../profiles.json")
    }
    fn servings() -> Vec<conformance::Serving> {
        use model_mimo26_contract::serving::{
            BUNDLE_DRAFTS, DECODE_ENTRY_ROWS, MAX_VERIFY_ROWS, VERIFY_ROW_BUCKETS, VERIFY_TOKENS,
        };
        let provides: Vec<String> = if model_mimo26_contract::geometry::V26_FLASH.encoders {
            ["text", "image", "video", "audio"]
                .map(str::to_owned)
                .to_vec()
        } else {
            vec!["text".to_owned()]
        };
        let serving = |profile: &str, tp: Tp, sequences: u32, drafter: &str, drafts: u32| {
            conformance::Serving {
                profile: profile.to_owned(),
                tensor_parallel: tp.ranks(),
                context_parallel: 1,
                sequences,
                drafter: drafter.to_owned(),
                draft_tokens: drafts,
                prefix_cache: true,
                model_options: Default::default(),
                // The checkpoint stores both encoder towers and the
                // geometry has their kernels (compiler `served_modalities`),
                // so `auto` serves text and image: the package names its
                // encoder set (`weight_plan::ENCODER_SET_ROLE`).
                provides: provides.clone(),
                selection: None,
                // The package's declaration (compiler `mimo26_package`):
                // MTP drafts within the decode kernels' rows, DFlash drafts
                // every slot to full depth.
                package: Some(conformance::PackageSpeculation {
                    maximum_draft_tokens: MAX_VERIFY_ROWS - 1,
                    bundle_drafter: drafts > 0 && BUNDLE_DRAFTS,
                    maximum_verify_tokens: match drafter {
                        "mtp" => DECODE_ENTRY_ROWS,
                        _ => VERIFY_TOKENS,
                    },
                    row_buckets: VERIFY_ROW_BUCKETS.to_vec(),
                    maximum_draft_items: None,
                    narrow_rows: false,
                    exact_graphs: false,
                }),
            }
        };
        vec![
            serving("tp4-ar", Tp::TP4, 32, "none", 0),
            serving("tp4-mtp", Tp::TP4, 32, "mtp", 3),
            serving("tp4-dflash", Tp::TP4, 32, "dflash", 7),
            serving("tp2-ar", Tp::TP2, 32, "none", 0),
        ]
    }
    fn new(serving: &conformance::Serving) -> Self {
        let tp = [Tp::TP2, Tp::TP4]
            .into_iter()
            .find(|tp| tp.ranks() == serving.tensor_parallel)
            .expect("a MiMo topology");
        let host = Rc::new(FakeHost::at(tp));
        host.served.set(serving.sequences);
        if serving.draft_tokens > 0 {
            host.speculation
                .set(Some((serving.draft_tokens, serving.verify_rows())));
            host.dflash.set(serving.drafter == "dflash");
        }
        Self(
            Rc::clone(&host),
            Rc::new(ObservedHost::new(host)),
            serving.clone(),
        )
    }
    fn create(&self) -> Result<BundleInstanceV1, BundleAbiError> {
        let host = &self.0;
        if host.device.symbols_only.get() {
            host.pool_pages
                .set(Some(1024 * u64::from(self.2.sequences)));
        }
        // The package's encoders serve the media of the served modalities.
        let modalities = self.2.modalities();
        let encoders = model_mimo26_contract::encoders::Encoders::named(
            modalities
                .into_iter()
                .filter(|&modality| modality != "text"),
        )
        .unwrap_or_else(|error| panic!("{}: {error}", self.2.profile));
        host.served_encoders.set(Some(encoders));
        let ordinals = (0..host.tp().ranks() as i32).collect::<Vec<_>>();
        BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)?.create(
            HostServicesTableV1::new(Rc::clone(&self.1)),
            &configuration_on_serving_encoders(
                host.tp(),
                host.kv(),
                self.2.draft_tokens > 0,
                &ordinals,
                host.served(),
                encoders,
            ),
        )
    }
    /// Two KV pages per slot, so every served slot fits the fake pool. A
    /// step with verify items reads back one token per row, a decoding
    /// item's row included.
    fn step(&self, items: &[conformance::Item]) -> Vec<u8> {
        let verifies = items
            .iter()
            .any(|item| item.phase == conformance::Phase::Verify);
        *self.0.readback_tokens.borrow_mut() = items
            .iter()
            .filter(|_| verifies)
            .flat_map(|item| item.tokens.iter().copied())
            .collect();
        let (batch, appends): (Vec<_>, Vec<_>) = items
            .iter()
            .map(|item| {
                let rows = item.tokens.len() as u32;
                let end = item.start + u64::from(rows);
                let pages = (0..end.div_ceil(PAGE_TOKENS))
                    .map(|page| {
                        item.slot
                            * (if self.0.device.symbols_only.get() {
                                1024
                            } else {
                                2
                            })
                            + page
                    })
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
                )
            })
            .unzip();
        BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
            .encode()
            .unwrap()
    }
    fn token(&self, position: u64) -> u32 {
        40 + position as u32
    }
    fn assert_released(&self) {
        self.1.accounting.assert_released();
    }
}

#[test]
fn route_memory_reuses_state_across_varying_shapes() {
    conformance::route_guard::memory_regression::<Served>();
}
