//! Shared conformance scenarios over the glm5next CPU host, one served item
//! with the precision component bound, in each configuration that serves:
//! autoregressive (no speculation), the MTP drafter, and verify rows the
//! host drafts, each over FP8 and BF16 latent KV.
//!
//! Without speculation the instance takes no verify rows, so the scenarios
//! that open a verify step (malformed commits and two of the poisoned-client
//! methods) do not apply; the autoregressive configurations run the other
//! three, and a verify item is refused without opening a step.
//!
//! The allocation scenario repeats a create per allocation (about five
//! hundred creates per configuration), so it runs over FP8 KV only, autoregressive and
//! with the MTP drafter: BF16 KV and host-drafted verify rows acquire the
//! same resources at other sizes.
use super::*;
use bundle_runtime::fake_host::conformance::{self, Adapter, CancellationPolicy, ObservedHost};
use bundle_runtime::fake_host::{NoObserver, RecordingSink, SinkLimits};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

/// A served configuration: its KV format (an index into
/// `Shape::kv_formats`) and the host's (drafts, verify tokens).
struct Case {
    format: usize,
    speculation: (u32, u32),
}

const AR_FP8: usize = 0;
const AR_BF16: usize = 1;
const MTP_FP8: usize = 2;
const MTP_BF16: usize = 3;
const DRAFTS_FP8: usize = 4;
const DRAFTS_BF16: usize = 5;

const CASES: [Case; 6] = [
    Case {
        format: 0,
        speculation: (0, 0),
    },
    Case {
        format: 2,
        speculation: (0, 0),
    },
    Case {
        format: 1,
        speculation: (3, 32),
    },
    Case {
        format: 3,
        speculation: (3, 32),
    },
    Case {
        format: 0,
        speculation: (3, 32),
    },
    Case {
        format: 2,
        speculation: (3, 32),
    },
];

pub(super) struct Limits;
impl SinkLimits for Limits {
    fn limits() -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(256, 1 << 20, 256 << 20)
    }
}

/// The host, its observed wrapper, and the handle of the instance the
/// scenario created (cancellation's fault needs it).
struct Model<const CASE: usize> {
    host: Rc<Host>,
    observed: Rc<ObservedHost<Host>>,
    handle: Cell<Option<BundleHandleV1>>,
    /// Whether the next cancel finds its step committed behind the
    /// client's back.
    cancel_armed: Cell<bool>,
}

impl<const CASE: usize> Model<CASE> {
    fn format(&self) -> KvFormat {
        Shape::TP4.kv_formats()[CASES[CASE].format]
    }

    /// Whether the MTP drafter drafts the verify rows.
    fn drafting(&self) -> bool {
        has_mtp_layer(&self.format()) && CASES[CASE].speculation.0 > 0
    }
}

impl<const CASE: usize> Adapter for Model<CASE> {
    type Startup = Served;
    type Observer = NoObserver;
    type Limits = Limits;
    type Host = ObservedHost<Host>;

    fn new() -> Self {
        let case = &CASES[CASE];
        let host = FakeHost::with_format(case.speculation, Shape::TP4.kv_formats()[case.format]);
        host.items.set(1);
        Self {
            observed: Rc::new(ObservedHost::new(Rc::clone(&host))),
            host,
            handle: Cell::new(None),
            cancel_armed: Cell::new(false),
        }
    }
    fn host(&self) -> Rc<Self::Host> {
        Rc::clone(&self.observed)
    }
    fn device(&self) -> &FakeDevice {
        &self.host.device
    }
    fn api(&self) -> BundleApiV1 {
        api()
    }
    fn configuration(&self) -> Vec<u8> {
        let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
        crate::config::tests::record_naming(1, &options(2048), self.format(), &roles)
    }
    fn create(&self) -> Result<BundleInstanceV1, BundleAbiError> {
        let instance = self
            .api()
            .create(HostServicesTableV1::new(self.host()), &self.configuration())?;
        self.handle.set(Some(instance.handle()));
        Ok(instance)
    }
    fn request(&self, request: u64, start: u64, tokens: &[u32], verify: bool) -> Vec<u8> {
        let rows = tokens.len() as u32;
        let item = match (verify, self.drafting()) {
            // The drafter drafts; the host sends the bonus token alone.
            (true, true) => BatchItemV1::verify(request, rows, true, tokens[..1].to_vec()),
            (true, false) => BatchItemV1::verify(request, rows, false, tokens.to_vec()),
            (false, _) => {
                BatchItemV1::new(request, start > 0 && rows == 1, rows, true, tokens.to_vec())
            }
        };
        let end = start + u64::from(rows);
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![item],
            vec![BatchAppendV1::new(
                start,
                u64::from(rows),
                end,
                (0..end.div_ceil(64)).collect(),
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
        self.observed.accounting.assert_released();
        for id in self.host.device.buffer_order.borrow().iter() {
            assert!(
                !self.host.extents.borrow().contains_key(id),
                "allocated buffer {id} still live"
            );
        }
    }
    fn release_count(&self) -> usize {
        self.observed.accounting.releases()
    }
    fn callback_count(&self) -> usize {
        self.observed.accounting.calls()
    }
    /// Decode, verify and prefill steps advance the KDA state and the
    /// indexer tail in place, so cancel retires the requests they touched.
    fn cancellation_policy(&self) -> CancellationPolicy {
        CancellationPolicy::InvalidateTouched
    }
    /// Cancel issues no device work, so no host fault can fail it: the
    /// fault commits the open step through the raw entry, behind the
    /// client, and the bundle then refuses a cancel with no step to close.
    fn arm_cancel_failure(&self) {
        let handle = self.handle.get().expect("a created instance");
        assert_eq!(
            bundle_runtime::entry::commit::<crate::instance::Instance>(handle),
            BundleStatusV1::Success
        );
        self.cancel_armed.set(true);
    }
    fn assert_cancel_failure_consumed(&self) {
        assert!(self.cancel_armed.take());
        let handle = self.handle.get().expect("a created instance");
        assert_eq!(
            crate::instance::ledger_state(handle, 0),
            (Some((7, 5)), false),
            "the refused cancel left the committed continuation"
        );
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

// One startup test per profile, then one test per scenario of a
// configuration.
conformance::suite! {
    startup: Served {
        tp4_mtp: "tp4-mtp",
        tp4_mtp_k5: "tp4-mtp-k5",
        tp4_ar: "tp4-ar",
        tp4_mtp_dcp2: "tp4-mtp-dcp2",
        tp4_mtp_dcp4: "tp4-mtp-dcp4",
    }
    mod ar_fp8 {
        allocation_release: Model<AR_FP8> => AllocationRelease;
        cancellation_policy: Model<AR_FP8> => Cancellation;
        retire_and_slot_reuse: Model<AR_FP8> => SlotReuse;
    }
    mod ar_bf16 {
        cancellation_policy: Model<AR_BF16> => Cancellation;
        retire_and_slot_reuse: Model<AR_BF16> => SlotReuse;
    }
    mod mtp_fp8 {
        allocation_release: Model<MTP_FP8> => AllocationRelease;
        malformed_commit: Model<MTP_FP8> => MalformedCommit;
        poisoned_client: Model<MTP_FP8> => PoisonedClient;
        cancellation_policy: Model<MTP_FP8> => Cancellation;
        retire_and_slot_reuse: Model<MTP_FP8> => SlotReuse;
    }
    mod mtp_bf16 {
        malformed_commit: Model<MTP_BF16> => MalformedCommit;
        poisoned_client: Model<MTP_BF16> => PoisonedClient;
        cancellation_policy: Model<MTP_BF16> => Cancellation;
        retire_and_slot_reuse: Model<MTP_BF16> => SlotReuse;
    }
    mod drafts_fp8 {
        malformed_commit: Model<DRAFTS_FP8> => MalformedCommit;
        poisoned_client: Model<DRAFTS_FP8> => PoisonedClient;
        cancellation_policy: Model<DRAFTS_FP8> => Cancellation;
        retire_and_slot_reuse: Model<DRAFTS_FP8> => SlotReuse;
    }
    mod drafts_bf16 {
        malformed_commit: Model<DRAFTS_BF16> => MalformedCommit;
        poisoned_client: Model<DRAFTS_BF16> => PoisonedClient;
        cancellation_policy: Model<DRAFTS_BF16> => Cancellation;
        retire_and_slot_reuse: Model<DRAFTS_BF16> => SlotReuse;
    }
}

/// Without speculation a verify item is refused before a step opens: the
/// client stays usable and the committed prefill continues.
#[test]
fn an_autoregressive_instance_refuses_verify_items() {
    refused::<AR_FP8>();
    refused::<AR_BF16>();
}

fn refused<const CASE: usize>() {
    let model = Model::<CASE>::new();
    let mut instance = model.create().expect("create");
    let mut sink = RecordingSink::<Limits>::default();
    instance
        .execute(&model.request(7, 0, &[1, 2, 3], false), &mut sink)
        .expect("prefill");
    instance.commit().expect("commit prefill");
    let verify = model.request(7, 3, &[4, 5, 6], true);
    assert!(instance.execute(&verify, &mut sink).is_err());
    assert!(!instance.is_poisoned());
    assert_eq!(
        crate::instance::ledger_state(instance.handle(), 0),
        (Some((7, 3)), false)
    );
    instance
        .execute(&model.request(7, 3, &[4], false), &mut sink)
        .expect("decode");
    instance.commit().expect("commit decode");
    instance.destroy().expect("destroy");
    model.assert_released();
}

/// One serving configuration per profile in profiles.json, over a host
/// serving its sequences on the FP8 KV format `auto` selects (with the MTP
/// layer when the profile drafts), in its checkpoint layout and
/// context-parallel size, with the vision encoder when it serves images.
struct Served {
    host: Rc<Host>,
    observed: Rc<ObservedHost<Host>>,
    serving: conformance::Serving,
}

impl conformance::Startup for Served {
    type Observer = bundle_runtime::fake_host::NoObserver;

    fn envelope(serving: &conformance::Serving) -> conformance::Envelope {
        let mut envelope = conformance::Envelope::served(
            serving,
            model_glm5next_contract::contract::PREFILL_TOKENS,
        );
        envelope.chunk_offsets = vec![8192, 16384, 32768, 40960];
        envelope.probes = vec![446, 789, 1803, 3658, 4542, 7754];
        envelope
    }
    fn route_tags(&self) -> Vec<kernel_core::route::RouteTag> {
        crate::sm120::backend(
            crate::sm120::engines::Engines::shared(
                Shape::TP4,
                &model_glm5next_contract::quantization::FP8_BLOCK128,
            )
            .expect("registered kernels"),
            Shape::TP4,
        )
        .0
        .route_tags()
    }
    fn route_threads(tier: conformance::route_guard::Tier) -> usize {
        if tier == conformance::route_guard::Tier::Quick {
            4
        } else {
            8
        }
    }
    fn finish_route_case(&self) {
        assert!(
            self.host.upload_log.borrow().is_empty(),
            "route sweep retained upload_log"
        );
        // No request or readback remains pending. Resident kernels and graph
        // commands stay live, but this shape's upload snapshots can go.
        self.host.uploaded.borrow_mut().clear();
        assert!(
            self.host.launch_log.borrow().is_empty(),
            "route sweep retained launch_log"
        );
        assert!(
            self.host.capture_log.borrow().is_empty(),
            "route sweep retained capture_log"
        );
        self.host.device.clear_route_journal();
    }
    fn route_device(&self) -> &bundle_runtime::fake_host::FakeDevice<Self::Observer> {
        &self.host.device
    }
    type Limits = Limits;

    fn profiles() -> &'static str {
        include_str!("../../profiles.json")
    }
    fn servings() -> Vec<conformance::Serving> {
        // No profile names a checkpoint layout; each starts on the block-FP8
        // package the default host serves. The mixed layouts' packages are
        // stepped by `fake_host_tests::nvfp4`.
        let serving = |profile: &str, drafts: u32, context_parallel: u32| {
            conformance::Serving {
                profile: profile.to_owned(),
                tensor_parallel: Shape::TP4.ranks(),
                context_parallel,
                sequences: DECODE_BATCH,
                drafter: if drafts > 0 { "mtp" } else { "none" }.to_owned(),
                draft_tokens: drafts,
                prefix_cache: true,
                model_options: Default::default(),
                // The contract declares the vision encoder
                // (`encoder::declarations`), whose tower every admitted
                // checkpoint stores, so `auto` serves images.
                provides: vec!["text".to_owned(), "image".to_owned()],
                selection: None,
                // The default package's declaration (compiler `package.rs`):
                // the MTP drafter at the contract's maximum serves every
                // profile's draft count.
                package: Some(conformance::PackageSpeculation {
                    maximum_draft_tokens: model_glm5next_contract::contract::MAX_DRAFT_TOKENS,
                    bundle_drafter: true,
                    maximum_verify_tokens: model_glm5next_contract::contract::head_rows(),
                    row_buckets: (2..=model_glm5next_contract::contract::MAX_DRAFT_TOKENS + 1)
                        .collect(),
                    maximum_draft_items: Some(model_glm5next_contract::contract::DRAFT_ITEMS),
                    narrow_rows: false,
                    exact_graphs: false,
                }),
            }
        };
        vec![
            serving("tp4-ar", 0, 1),
            serving("tp4-mtp", 3, 1),
            serving("tp4-mtp-k5", 5, 1),
            serving("tp4-mtp-dcp2", 3, 2),
            serving("tp4-mtp-dcp4", 3, 4),
        ]
    }
    fn new(serving: &conformance::Serving) -> Self {
        let format = Shape::TP4.kv_formats()[usize::from(serving.draft_tokens > 0)];
        let speculation = (serving.draft_tokens, serving.verify_rows());
        let host = FakeHost::with_format(speculation, format);
        host.items.set(serving.sequences);
        Self {
            observed: Rc::new(ObservedHost::new(Rc::clone(&host))),
            host,
            serving: serving.clone(),
        }
    }
    fn create(&self) -> Result<BundleInstanceV1, BundleAbiError> {
        let mut tail = Vec::new();
        if self.serving.context_parallel > 1 {
            tail.extend(runtime_bundle_sdk::context_parallel::TAG);
            tail.extend_from_slice(&self.serving.context_parallel.to_le_bytes());
        }
        tail.extend(crate::config::tests::options(2048));
        let mut roles = vec![crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
        if self.serving.serves_images() {
            roles.extend(crate::config::tests::encoder_roles());
        }
        let record = crate::config::tests::record_naming(
            self.serving.sequences,
            &tail,
            self.host.format,
            &roles,
        );
        let result = api().create(HostServicesTableV1::new(Rc::clone(&self.observed)), &record);
        if result.is_err() && self.host.device.symbols_only.get() {
            eprintln!("route create diagnostics: {:?}", self.host.logs.borrow());
        }
        result
    }
    /// Four KV pages per slot, each a stripe of the context-parallel group's
    /// pages; the drafter drafts from the bonus token alone.
    fn step(&self, items: &[conformance::Item]) -> Vec<u8> {
        let size = u64::from(self.serving.context_parallel);
        let mut batch = Vec::new();
        let mut appends = Vec::new();
        let mut slots = Vec::new();
        for item in items {
            let rows = item.tokens.len() as u32;
            let end = item.start + u64::from(rows);
            let pages = (0..end.div_ceil(64 * size))
                .map(|stripe| {
                    item.slot
                        * (if self.host.device.symbols_only.get() {
                            1024
                        } else {
                            4
                        })
                        + stripe
                })
                .collect();
            batch.push(match item.phase {
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
            });
            appends.push(BatchAppendV1::new(
                item.start,
                u64::from(rows),
                end,
                pages,
                None,
            ));
            slots.push(item.slot);
        }
        BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
            .with_state_slots(slots)
            .encode()
            .unwrap()
    }
    fn token(&self, position: u64) -> u32 {
        position as u32 + 1
    }
    fn assert_released(&self) {
        self.observed.accounting.assert_released();
        for id in self.host.device.buffer_order.borrow().iter() {
            assert!(
                !self.host.extents.borrow().contains_key(id),
                "{}: allocated buffer {id} still live",
                self.serving.profile
            );
        }
    }
}

#[test]
fn route_memory_reuses_state_across_varying_shapes() {
    conformance::route_guard::memory_regression::<Served>();
}
