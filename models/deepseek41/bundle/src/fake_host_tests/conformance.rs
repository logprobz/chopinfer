//! Shared conformance scenarios over the deepseek41 fake host, one served
//! item over the native KV layout, at every topology Flash serves: without
//! speculation, and with host-supplied verify rows retained by DSpark.
//!
//! Without speculation the instance takes no verify rows, so the scenarios
//! that open a verify step (malformed commits and two of the poisoned-client
//! methods) run on the host-drafted configurations alone. The allocation
//! scenario repeats a create per allocation, so it runs without
//! speculation: verify rows acquire the same resources at other sizes.
use super::*;
use bundle_runtime::fake_host::conformance::{
    self, Adapter, CancellationPolicy, ObservedHost, Phase,
};
use bundle_runtime::fake_host::{RecordingSink, SinkLimits};
use model_deepseek41_contract::serving::DECODE_BATCH;
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

/// A served configuration: its rank count and the host's (drafts, verify
/// tokens), none without speculation.
struct Case {
    ranks: Ranks,
    speculation: Option<(u32, u32)>,
}

/// Host-drafted verify rows: three drafts, and the bonus token.
const DRAFTS: Option<(u32, u32)> = Some((3, 16));

const AR_TP1: usize = 0;
const AR_TP2: usize = 1;
const AR_TP4: usize = 2;
const DRAFTS_TP1: usize = 3;
const DRAFTS_TP2: usize = 4;
const DRAFTS_TP4: usize = 5;

const CASES: [Case; 6] = [
    Case {
        ranks: Ranks::TP1,
        speculation: None,
    },
    Case {
        ranks: Ranks::TP2,
        speculation: None,
    },
    Case {
        ranks: Ranks::TP4,
        speculation: None,
    },
    Case {
        ranks: Ranks::TP1,
        speculation: DRAFTS,
    },
    Case {
        ranks: Ranks::TP2,
        speculation: DRAFTS,
    },
    Case {
        ranks: Ranks::TP4,
        speculation: DRAFTS,
    },
];

pub(super) struct Limits;
impl SinkLimits for Limits {
    fn limits() -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(256, 1 << 20, 256 << 20)
    }
}

/// The encoded step of `items`, `(request, slot, start, tokens, phase)`,
/// each slot's KV pages eight from `slot * 8`. A bundle-drafted verify
/// item supplies only its bonus token; host-drafted items supply every row.
fn step(items: &[(u64, u64, u64, &[u32], Phase)], bundle_drafts: bool) -> Vec<u8> {
    let mut batch = Vec::new();
    let mut appends = Vec::new();
    let mut slots = Vec::new();
    for &(request, slot, start, tokens, phase) in items {
        let rows = tokens.len() as u32;
        let end = start + u64::from(rows);
        batch.push(match phase {
            Phase::Verify => BatchItemV1::verify(
                request,
                rows,
                bundle_drafts,
                if bundle_drafts { &tokens[..1] } else { tokens }.to_vec(),
            ),
            phase => BatchItemV1::new(request, phase == Phase::Decode, rows, true, tokens.to_vec()),
        });
        appends.push(BatchAppendV1::new(
            start,
            u64::from(rows),
            end,
            (0..end.div_ceil(64)).map(|page| slot * 8 + page).collect(),
            None,
        ));
        slots.push(slot);
    }
    BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap()
}

/// Checks that every acquisition through `observed` was released and that
/// no buffer `host` allocated is still live.
fn assert_host_released<B>(observed: &ObservedHost<B>, host: &FakeHost, label: &str) {
    observed.accounting.assert_released();
    for id in host.device.buffer_order.borrow().iter() {
        assert!(
            !host.device.extents.borrow().contains_key(id),
            "{label}: allocated buffer {id} still live"
        );
    }
}

/// The host, its observed wrapper, and the handle of the instance the
/// scenario created (cancellation's fault needs it).
struct Model<const CASE: usize> {
    host: Rc<Host>,
    observed: Rc<ObservedHost<dspark::host::MappedHost>>,
    handle: Cell<Option<BundleHandleV1>>,
    /// Whether the next cancel finds its step committed behind the
    /// client's back.
    cancel_armed: Cell<bool>,
}

impl<const CASE: usize> Adapter for Model<CASE> {
    type Startup = Served;
    type Observer = NoObserver;
    type Limits = Limits;
    type Host = ObservedHost<dspark::host::MappedHost>;

    fn new() -> Self {
        let case = &CASES[CASE];
        let host = FakeHost::new(Shape::new(&FLASH, case.ranks).unwrap(), KV_LAYOUTS[0]);
        host.speculation.set(case.speculation);
        host.drafter.set(case.speculation.is_some());
        Self {
            observed: Rc::new(ObservedHost::new(Rc::new(dspark::host::MappedHost::new(
                Rc::clone(&host),
            )))),
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
        crate::config::tests::record(&self.host.shape, self.host.kv, 1, false)
    }
    fn create(&self) -> Result<BundleInstanceV1, BundleAbiError> {
        let instance = self
            .api()
            .create(HostServicesTableV1::new(self.host()), &self.configuration())?;
        self.handle.set(Some(instance.handle()));
        Ok(instance)
    }
    fn request(&self, request: u64, start: u64, tokens: &[u32], verify: bool) -> Vec<u8> {
        let phase = match (verify, tokens.len()) {
            (true, _) => Phase::Verify,
            (false, 1) if start > 0 => Phase::Decode,
            (false, _) => Phase::Prefill,
        };
        step(&[(request, 0, start, tokens, phase)], false)
    }
    fn raw_commit(&self, handle: BundleHandleV1, records: &[BundleKeptRowsV1]) -> BundleStatusV1 {
        bundle_runtime::entry::commit_accepted::<crate::instance::Instance>(handle, records)
    }
    fn assert_released(&self) {
        assert_host_released(&self.observed, &self.host, "scenario");
    }
    fn release_count(&self) -> usize {
        self.observed.accounting.releases()
    }
    fn callback_count(&self) -> usize {
        self.observed.accounting.calls()
    }
    /// Steps advance the rings, the compressor tails and the Engram history
    /// in place, so cancel retires the requests they touched.
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
// configuration. Allocation sweeps partition every failure index across
// bounded tests: repeated operation preparation exceeds the time limit of
// one test, while every shard keeps fresh instances and all release checks.
conformance::suite! {
    startup: Served {
        tp4_ar: "tp4-ar",
        tp4_dspark: "tp4-dspark",
        tp4_ar_full_prefill: "tp4-ar-full-prefill",
        tp4_dspark_full_prefill: "tp4-dspark-full-prefill",
        tp4_ar_upper_skip_2048: "tp4-ar-upper-skip-2048",
        tp4_dspark_upper_skip_2048: "tp4-dspark-upper-skip-2048",
    }
    mod ar_tp1 {
        allocation_release: Model<AR_TP1> => AllocationReleaseShards;
        cancellation_policy: Model<AR_TP1> => Cancellation;
        retire_and_slot_reuse: Model<AR_TP1> => SlotReuse;
    }
    mod ar_tp2 {
        allocation_release: Model<AR_TP2> => AllocationReleaseShards;
        cancellation_policy: Model<AR_TP2> => Cancellation;
        retire_and_slot_reuse: Model<AR_TP2> => SlotReuse;
    }
    mod ar_tp4 {
        allocation_release: Model<AR_TP4> => AllocationReleaseShards;
        cancellation_policy: Model<AR_TP4> => Cancellation;
        retire_and_slot_reuse: Model<AR_TP4> => SlotReuse;
    }
    mod drafts_tp1 {
        malformed_commit: Model<DRAFTS_TP1> => MalformedCommit;
        poisoned_client: Model<DRAFTS_TP1> => PoisonedClient;
        cancellation_policy: Model<DRAFTS_TP1> => Cancellation;
        retire_and_slot_reuse: Model<DRAFTS_TP1> => SlotReuse;
    }
    mod drafts_tp2 {
        malformed_commit: Model<DRAFTS_TP2> => MalformedCommit;
        poisoned_client: Model<DRAFTS_TP2> => PoisonedClient;
        cancellation_policy: Model<DRAFTS_TP2> => Cancellation;
        retire_and_slot_reuse: Model<DRAFTS_TP2> => SlotReuse;
    }
    mod drafts_tp4 {
        malformed_commit: Model<DRAFTS_TP4> => MalformedCommit;
        poisoned_client: Model<DRAFTS_TP4> => PoisonedClient;
        cancellation_policy: Model<DRAFTS_TP4> => Cancellation;
        retire_and_slot_reuse: Model<DRAFTS_TP4> => SlotReuse;
    }
}

/// Without speculation a verify item is refused before a step opens: the
/// client stays usable and the committed prefill continues.
#[test]
fn an_autoregressive_instance_refuses_verify_items() {
    let model = Model::<AR_TP2>::new();
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

/// One serving configuration per profile in profiles.json: the Flash
/// topology of its rank count, its sequences, over the native KV layout
/// `auto` selects, with the host's speculation when the profile drafts.
struct Served {
    host: Rc<Host>,
    observed: Rc<ObservedHost<dspark::host::MappedHost>>,
    serving: conformance::Serving,
}

impl conformance::Startup for Served {
    type Limits = Limits;

    fn profiles() -> &'static str {
        include_str!("../../../profiles.json")
    }
    fn servings() -> Vec<conformance::Serving> {
        let serving = |profile: &str, draft_tokens: u32| {
            conformance::Serving {
            profile: profile.to_owned(),
            tensor_parallel: Ranks::TP4.get(),
            context_parallel: 1,
            sequences: DECODE_BATCH,
            drafter: if draft_tokens > 0 { "dspark" } else { "none" }.to_owned(),
            draft_tokens,
            prefix_cache: true,
            model_options: Default::default(),
            // The package serves text alone: no encoder loads.
            provides: vec!["text".to_owned()],
            selection: None,
            package: Some(conformance::PackageSpeculation {
                maximum_draft_tokens: model_deepseek41_contract::serving::MAX_DRAFT_TOKENS,
                bundle_drafter: true,
                maximum_verify_tokens: <model_deepseek41_contract::serving::DeepSeek as bundle_runtime::BundleGeometry>::VERIFY_TOKENS,
                row_buckets: (2..=model_deepseek41_contract::serving::MAX_VERIFY_ROWS).collect(),
                maximum_draft_items: None,
                narrow_rows: true,
                exact_graphs: true,
            }),
        }
        };
        let mut configurations = Vec::new();
        for (mode, drafts) in [("ar", 0), ("dspark", 5)] {
            // The default profiles skip at guaranteed 8192-token chunk ends.
            for (profile, chunk) in [
                (format!("tp4-{mode}"), 8192),
                (format!("tp4-{mode}-upper-skip-2048"), 2048),
            ] {
                let mut entry = serving(&profile, drafts);
                entry.model_options = BTreeMap::from([
                    ("prefill_chunk_tokens".into(), chunk.to_string()),
                    ("prefill_upper_skip".into(), "true".into()),
                ]);
                configurations.push(entry);
            }
            let mut full = serving(&format!("tp4-{mode}-full-prefill"), drafts);
            full.model_options = BTreeMap::from([
                ("prefill_chunk_tokens".into(), "0".into()),
                ("prefill_upper_skip".into(), "false".into()),
            ]);
            configurations.push(full);
        }
        // The DSpark profiles cap balanced chunks at twelve drafting items.
        for entry in &mut configurations {
            if entry.draft_tokens > 0 {
                entry.model_options.insert(
                    model_deepseek41_contract::dspark_options::DRAFT_CHUNK_ITEMS.into(),
                    "12".into(),
                );
            }
        }
        configurations
    }
    fn new(serving: &conformance::Serving) -> Self {
        let ranks = Ranks::new(serving.tensor_parallel).expect("a supported rank count");
        let host = FakeHost::new(Shape::new(&FLASH, ranks).unwrap(), KV_LAYOUTS[0]);
        if serving.draft_tokens > 0 {
            host.drafter.set(true);
            host.speculation
                .set(Some((serving.draft_tokens, serving.verify_rows())));
        }
        Self {
            observed: Rc::new(ObservedHost::new(Rc::new(dspark::host::MappedHost::new(
                Rc::clone(&host),
            )))),
            host,
            serving: serving.clone(),
        }
    }
    fn create(&self) -> Result<BundleInstanceV1, BundleAbiError> {
        let mut record = crate::config::tests::record(
            &self.host.shape,
            self.host.kv,
            self.serving.sequences,
            false,
        );
        if !self.serving.model_options.is_empty() {
            let tag = runtime_bundle_abi::MODEL_OPTIONS_TAG_V1;
            let at = record.windows(tag.len()).position(|w| w == tag).unwrap() + tag.len();
            let mut options: serde_json::Value = serde_json::from_slice(&record[at + 4..]).unwrap();
            for (key, value) in &self.serving.model_options {
                options[key] = serde_json::from_str(value).unwrap();
            }
            let options = serde_json::to_vec(&options).unwrap();
            record.truncate(at);
            record.extend_from_slice(&(options.len() as u32).to_le_bytes());
            record.extend_from_slice(&options);
        }
        api().create(HostServicesTableV1::new(Rc::clone(&self.observed)), &record)
    }
    fn step(&self, items: &[conformance::Item]) -> Vec<u8> {
        let items: Vec<_> = items
            .iter()
            .map(|item| {
                (
                    item.request,
                    item.slot,
                    item.start,
                    item.tokens.as_slice(),
                    item.phase,
                )
            })
            .collect();
        step(&items, self.serving.bundle_drafts())
    }
    fn token(&self, position: u64) -> u32 {
        position as u32 + 1
    }
    fn assert_released(&self) {
        assert_host_released(&self.observed, &self.host, &self.serving.profile);
    }
}
