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

mod allocated;
mod anchored;
mod drafter;
mod snapshot_copy;
mod successor;
mod templated;
mod wide;

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
fn step_with_stride(
    items: &[(u64, u64, u64, &[u32], Phase)],
    bundle_drafts: bool,
    stride: u64,
    prompt_ends: Option<&[u64]>,
) -> Vec<u8> {
    let mut batch = Vec::new();
    let mut appends = Vec::new();
    let mut slots = Vec::new();
    for (index, &(request, slot, start, tokens, phase)) in items.iter().enumerate() {
        let rows = tokens.len() as u32;
        let end = start + u64::from(rows);
        batch.push(match phase {
            Phase::Verify => BatchItemV1::verify(
                request,
                rows,
                bundle_drafts,
                if bundle_drafts { &tokens[..1] } else { tokens }.to_vec(),
            ),
            phase => BatchItemV1::new(
                request,
                phase == Phase::Decode,
                rows,
                prompt_ends.is_none_or(|ends| end == ends[index]),
                tokens.to_vec(),
            ),
        });
        appends.push(BatchAppendV1::new(
            start,
            u64::from(rows),
            end,
            (0..end.div_ceil(64))
                .map(|page| slot * stride + page)
                .collect(),
            None,
        ));
        slots.push(slot);
    }
    BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap()
}

pub(super) fn step(items: &[(u64, u64, u64, &[u32], Phase)], bundle_drafts: bool) -> Vec<u8> {
    step_with_stride(items, bundle_drafts, 8, None)
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
    allocated_epoch: Cell<u64>,
}

#[test]
fn adopted_ar_buffers_precede_serving_kv_sizing() {
    use conformance::Startup;
    let mut late = Vec::new();
    for serving in Served::servings() {
        let fixture = Served::new(&serving);
        fixture.host.drafter.set(true);
        let mut instance = fixture.create().unwrap();
        let logs = fixture.host.logs.borrow();
        let memory: Vec<serde_json::Value> = logs
            .iter()
            .filter_map(|line| line.strip_prefix("startup memory: "))
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(memory.len(), serving.tensor_parallel as usize);
        let pages = logs
            .iter()
            .find(|line| line.contains(" KV pages per rank, "))
            .unwrap();
        eprintln!("{}: {pages}; memory={memory:?}", serving.profile);
        for record in &memory {
            let bytes = record["bytes"]["ar_engram"].as_u64().unwrap_or(0);
            let expected = if serving.draft_tokens == 0 && record["rank"] == 0 {
                let engram = &FLASH.shape.engram;
                let ids = u64::from(DECODE_BATCH * (engram.max_ngram - 1) * engram.heads)
                    * engram.layers.len() as u64
                    * 4;
                let status = (FLASH.shape.csa.index_sources.len() as u64
                    + 2 * u64::from(serving.tensor_parallel))
                    * u64::from(DECODE_BATCH)
                    * 4;
                ids + status
            } else {
                0
            };
            assert_eq!(bytes, expected, "{}: {record}", serving.profile);
            if record["after_kv"] != 0 {
                late.push((serving.profile.clone(), record.clone()));
            }
        }
        drop(logs);
        instance.destroy().unwrap();
        assert_eq!(fixture.host.acquired.get(), fixture.host.released.get());
    }
    assert!(
        late.is_empty(),
        "AR buffers allocated after KV sizing: {late:?}"
    );
}

impl conformance::Startup for Served {
    type Observer = bundle_runtime::fake_host::NoObserver;

    fn envelope(serving: &conformance::Serving) -> conformance::Envelope {
        let mut envelope = conformance::Envelope::served(
            serving,
            model_deepseek41_contract::serving::PREFILL_TOKENS,
        );
        if let Some(chunk) = serving
            .model_options
            .get("prefill_chunk_tokens")
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&n| n > 0)
        {
            envelope.prefill_tokens = chunk.min(envelope.prefill_tokens);
            // A grid profile declares scheduler cuts: sweep chunks ending
            // before their prompt ends inside either grid. A small first
            // cut, a floor-sized cut, a cut after an earlier cut, a cut
            // start reaching the grid boundary, and packed items of odd
            // lengths, two cut and one ending its prompt.
            let cut =
                |mode: &str, offset, lengths: &[u32], remaining: &[u32]| conformance::CutCase {
                    mode: mode.into(),
                    offset,
                    lengths: lengths.to_vec(),
                    remaining: remaining.to_vec(),
                };
            envelope.cuts = vec![
                cut("cut-small", 0, &[64], &[1436]),
                cut("cut-floor", 0, &[512], &[988]),
                cut("cut-repeat", 7168, &[512], &[8]),
                cut("cut-grid", 7680, &[512], &[808]),
                cut("cut-packed", 576, &[448, 64, 37], &[476, 860, 0]),
            ];
        }
        envelope.chunk_offsets = vec![8192, 16384, 32768, 40960];
        envelope.probes = vec![
            65, 85, 127, 129, 191, 446, 789, 831, 853, 1557, 1599, 1803, 3658, 4542, 7754,
        ];
        if serving.profile == "tp4-dspark" {
            envelope.probes.extend([191, 192, 193, 255, 256, 257, 258]);
            envelope.probes.sort_unstable();
            envelope.probes.dedup();
            // Probe retained inner rows directly, including packed upper bounds.
            // Prompt length alone does not name the main projection width.
            envelope
                .cuts
                .extend(
                    super::dspark_main_compact::BOUNDARIES
                        .iter()
                        .map(|&(rows, lengths)| conformance::CutCase {
                            mode: format!("main-inner-{rows}"),
                            offset: 0,
                            lengths: lengths.to_vec(),
                            remaining: vec![0; lengths.len()],
                        }),
                );
        }
        if serving.draft_tokens > 0 {
            // Exercise direct 9-12-item drafts and balanced 9+8, 12+12 and
            // 9+8+8 partitions. The existing 32-item wave runs 11+11+10.
            envelope.decode_buckets.extend(
                [9, 10, 11, 12, 17, 24, 25]
                    .into_iter()
                    .filter(|&items| items <= serving.sequences),
            );
            // Uniform depth-three heads and both neighboring full-path bounds.
            for items in 1..=9_usize {
                if items <= serving.sequences as usize
                    && 4 * items <= serving.verify_rows() as usize
                {
                    envelope
                        .verify
                        .push(runtime_bundle_sdk::warmup::VerifyShape {
                            row_bucket: 4 * items,
                            item_bucket: items,
                            drafts: vec![3; items],
                        });
                }
            }
            envelope.decode_buckets.sort_unstable();
            envelope.decode_buckets.dedup();
        }
        envelope
    }
    fn route_tags(&self) -> Vec<kernel_core::route::RouteTag> {
        let shape = self.host.shape.with_verify_rows(self.serving.verify_rows());
        let mut tags = crate::Kernels::route_tags(&shape);
        tags.extend(crate::engram_rows::route_tags());
        tags
    }
    fn check_allocated_route_shapes(
        serving: &conformance::Serving,
        shapes: &[runtime_bundle_sdk::warmup::AllocatedShape],
    ) {
        if serving.profile == "tp4-dspark" {
            let identities: Vec<_> = shapes
                .iter()
                .map(|shape| (shape.items, shape.width, shape.reserved_rows))
                .collect();
            assert_eq!(
                identities,
                [
                    (2, 8, 12),
                    (3, 16, 18),
                    (4, 16, 24),
                    (5, 16, 30),
                    (6, 32, 36),
                    (7, 32, 42),
                    (8, 32, 48),
                    (9, 32, 54),
                    (10, 32, 60),
                    (11, 32, 55),
                    (13, 32, 52),
                    (14, 32, 56),
                ],
                "tp4-dspark allocated route identities"
            );
        }
    }
    /// Each worker's instance keeps up to 64 upper-skip plan templates of
    /// 21 to 31 MiB, so five workers keep the longer tiers within the 16 GiB
    /// sweep budget.
    fn route_threads(tier: conformance::route_guard::Tier) -> usize {
        if tier == conformance::route_guard::Tier::Quick {
            4
        } else {
            5
        }
    }
    fn route_device(&self) -> &bundle_runtime::fake_host::FakeDevice<Self::Observer> {
        &self.host.device
    }
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
            verify: Default::default(),
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
            let mut entry = serving(&format!("tp4-{mode}"), drafts);
            entry.model_options = BTreeMap::from([
                ("prefill_chunk_tokens".into(), "8192".into()),
                ("prefill_upper_skip".into(), "true".into()),
            ]);
            configurations.push(entry);
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
            allocated_epoch: Cell::new(0),
        }
    }
    fn create(&self) -> Result<BundleInstanceV1, BundleAbiError> {
        self.create_with_prefill(false)
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
        step_with_stride(
            &items,
            self.serving.bundle_drafts(),
            if self.host.device.symbols_only.get() {
                1024
            } else {
                8
            },
            None,
        )
    }
    fn execute_route_step(
        &self,
        instance: &mut BundleInstanceV1,
        items: &[conformance::Item],
        prompt_ends: &[u64],
        sink: &mut RecordingSink<Self::Limits>,
    ) -> Result<(), BundleAbiError> {
        let chunk = self
            .serving
            .model_options
            .get("prefill_chunk_tokens")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);
        if items.iter().any(|i| i.phase != Phase::Prefill) {
            return instance.execute(&self.step(items), sink);
        }
        let tuples: Vec<_> = items
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
        let batch = step_with_stride(
            &tuples,
            self.serving.bundle_drafts(),
            1024,
            Some(prompt_ends),
        );
        if chunk == 0 {
            return instance.execute(&batch, sink);
        }
        let boundaries: Vec<_> = items
            .iter()
            .zip(prompt_ends)
            .map(
                |(item, &prompt_tokens)| runtime_bundle_sdk::prefill::Boundary {
                    request_id: item.request,
                    prompt_tokens,
                    chunk_end: item.start + item.tokens.len() as u64,
                    chunk_tokens: chunk,
                },
            )
            .collect();
        // Grid and prompt ends take the grid entry; earlier ends are
        // scheduler cuts.
        if let Ok(encoded) = runtime_bundle_sdk::prefill::encode(&batch, &boundaries) {
            return instance.execute_prefill(&encoded, sink);
        }
        let encoded = runtime_bundle_sdk::prefill::encode_with_cuts(&batch, &boundaries)
            .expect("served prefill cuts");
        instance.execute_prefill_cuts(&encoded, sink)
    }

    fn execute_allocated_step(
        &self,
        instance: &mut BundleInstanceV1,
        items: &[conformance::Item],
        allocation: runtime_bundle_sdk::VerifyAllocationV1,
    ) -> Result<(), BundleAbiError> {
        allocated::execute(self, instance, items, allocation)
    }

    fn token(&self, position: u64) -> u32 {
        position as u32 + 1
    }
    fn assert_released(&self) {
        assert_host_released(&self.observed, &self.host, &self.serving.profile);
    }
}

#[test]
fn route_envelopes_cover_balanced_drafter_partitions() {
    use conformance::Startup;
    let mut profiles = 0;
    for serving in Served::servings().iter().filter(|s| s.draft_tokens > 0) {
        profiles += 1;
        assert_eq!(
            serving.model_options[model_deepseek41_contract::dspark_options::DRAFT_CHUNK_ITEMS],
            "12"
        );
        let envelope = Served::envelope(serving);
        for items in [9, 10, 11, 12, 17, 24, 25, 32] {
            assert!(
                envelope.decode_buckets.contains(&items),
                "{}: missing balanced drafter witness with {items} items",
                serving.profile
            );
        }
    }
    assert_eq!(profiles, 1);
}

#[test]
fn route_records_cover_legacy_and_negotiated_prefill() {
    use conformance::Startup;
    bundle_runtime::routes::with_logging(|| {
        for profile in ["tp4-ar", "tp4-dspark"] {
            let serving = Served::servings()
                .into_iter()
                .find(|s| s.profile == profile)
                .unwrap();
            let adapter = Served::new(&serving);
            adapter.host.device.symbols_only.set(true);
            let mut instance = adapter.create().unwrap();
            for (request, negotiated) in [(701, false), (702, true)] {
                let items = [conformance::Item {
                    request,
                    slot: 0,
                    start: 0,
                    tokens: (1..=789).collect(),
                    phase: Phase::Prefill,
                }];
                let from = adapter.host.logs.borrow().len();
                let mut sink = RecordingSink::<Limits>::default();
                if negotiated {
                    adapter
                        .execute_route_step(&mut instance, &items, &[789], &mut sink)
                        .unwrap();
                } else {
                    instance.execute(&adapter.step(&items), &mut sink).unwrap();
                }
                instance.commit().unwrap();
                let logs = adapter.host.logs.borrow();
                let records: Vec<serde_json::Value> = logs[from..]
                    .iter()
                    .filter_map(|s| serde_json::from_str(s).ok())
                    .filter(|v: &serde_json::Value| v["schema"] == "chopinfer.route.v1")
                    .collect();
                assert_eq!(
                    records.len(),
                    2,
                    "execute and commit, negotiated={negotiated}"
                );
                if serving.draft_tokens > 0 {
                    assert!(
                        !records[1]["entries"].as_object().unwrap().is_empty(),
                        "batched prefill commit launches must be counted"
                    );
                }
                let route = &records[0];
                assert!(route["fallback_launches"].as_u64().unwrap() > 0);
                assert_eq!(
                    route["fallback_launches"].as_u64().unwrap(),
                    route["families"]
                        .as_object()
                        .unwrap()
                        .values()
                        .map(|n| n.as_u64().unwrap())
                        .sum::<u64>()
                );
                drop(logs);
                if serving.draft_tokens > 0 {
                    let from = adapter.host.logs.borrow().len();
                    let rows = serving.draft_tokens + 1;
                    let verify = [conformance::Item {
                        request,
                        slot: 0,
                        start: 789,
                        tokens: (790..790 + rows).collect(),
                        phase: Phase::Verify,
                    }];
                    instance
                        .execute(
                            &adapter.step(&verify),
                            &mut RecordingSink::<Limits>::default(),
                        )
                        .unwrap();
                    instance
                        .commit_accepted(&[BundleKeptRowsV1 {
                            struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
                            reserved: 0,
                            request_id: request,
                            kept_rows: rows,
                            reserved_tail: 0,
                        }])
                        .unwrap();
                    let records = adapter.host.logs.borrow()[from..]
                        .iter()
                        .filter_map(|s| serde_json::from_str::<serde_json::Value>(s).ok())
                        .filter(|v| v["schema"] == "chopinfer.route.v1")
                        .count();
                    assert_eq!(records, 2, "verify execute and accepted commit");
                }
                instance.retire_request(request).unwrap();
            }
            instance.destroy().unwrap();
            adapter.assert_released();
        }
    });
}

#[test]
fn engram_route_counts_cover_prepared_prefill_and_decode() {
    use conformance::Startup;
    for profile in ["tp4-ar", "tp4-dspark"] {
        bundle_runtime::routes::with_logging(|| {
            let serving = Served::servings()
                .into_iter()
                .find(|s| s.profile == profile)
                .unwrap();
            let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
            host.drafter.set(true);
            if serving.draft_tokens > 0 {
                host.speculation
                    .set(Some((serving.draft_tokens, serving.verify_rows())));
            }
            let mut instance = api()
                .create(
                    HostServicesTableV1::new(Rc::new(dspark::host::MappedHost::new(Rc::clone(
                        &host,
                    )))),
                    &crate::config::tests::record(&host.shape, host.kv, serving.sequences, false),
                )
                .unwrap();
            for (phase, start, tokens, fast) in [
                (Phase::Prefill, 0, vec![1, 2, 3], "prefill"),
                (
                    Phase::Decode,
                    3,
                    vec![4],
                    if serving.draft_tokens > 0 {
                        "dspark_undrafted"
                    } else {
                        "ar_decode"
                    },
                ),
            ] {
                let before = host.mapped_gather_calls.get();
                let from = host.logs.borrow().len();
                let batch = step_with_stride(
                    &[(701, 0, start, tokens.as_slice(), phase)],
                    serving.bundle_drafts(),
                    8,
                    None,
                );
                instance
                    .execute(&batch, &mut RecordingSink::<Limits>::default())
                    .unwrap();
                instance.commit().unwrap();
                let logs = host.logs.borrow();
                let records: Vec<serde_json::Value> = logs[from..]
                    .iter()
                    .filter_map(|line| serde_json::from_str(line).ok())
                    .filter(|value: &serde_json::Value| value["schema"] == "chopinfer.route.v1")
                    .collect();
                assert_eq!(records.len(), 2);
                let synchronous: u64 = records
                    .iter()
                    .flat_map(|record| record["entries"].as_object().unwrap().values())
                    .filter(|entry| entry["symbol"] == "host.gather_mapped_rows")
                    .map(|entry| entry["launches"].as_u64().unwrap())
                    .sum();
                assert_eq!(synchronous, 0);
                assert_eq!(host.mapped_gather_calls.get() - before, synchronous);
                let entry = format!("engram/prepared/{fast}");
                let calls = records[0]["entries"][&entry]["launches"]
                    .as_u64()
                    .unwrap_or_else(|| panic!("{profile}: missing {entry}"));
                assert_eq!(calls, u64::from(host.shape.geometry.engram_layers()));
            }
            instance.retire_request(701).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        });
    }
}

/// Check selected reasons against independent calls for a real execute and commit.
pub(super) fn assert_sync_engram(
    host: &FakeHost,
    instance: &mut BundleInstanceV1,
    batch: &[u8],
    reasons: &[&str],
) {
    let before = host.mapped_gather_calls.get();
    let from = host.logs.borrow().len();
    instance
        .execute(batch, &mut RecordingSink::<Limits>::default())
        .unwrap_or_else(|error| panic!("{error:?}: {:?}", host.logs.borrow()));
    use runtime_bundle_sdk::BatchCodecV1;
    let decoded = BatchCodecV1::decode(batch).unwrap();
    if decoded.items().iter().any(|item| item.is_verify()) {
        let kept = decoded
            .items()
            .iter()
            .map(|item| BundleKeptRowsV1 {
                struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
                reserved: 0,
                request_id: item.request_id(),
                kept_rows: item.phase_tokens(),
                reserved_tail: 0,
            })
            .collect::<Vec<_>>();
        instance.commit_accepted(&kept).unwrap();
    } else {
        instance.commit().unwrap();
    }
    let mut selected = BTreeMap::<String, u64>::new();
    for record in host.logs.borrow()[from..]
        .iter()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|value| value["schema"] == "chopinfer.route.v1")
    {
        for (entry, value) in record["entries"].as_object().unwrap() {
            if value["symbol"] == bundle_runtime::routes::MAPPED_GATHER_SYMBOL {
                *selected.entry(entry.clone()).or_default() += value["launches"].as_u64().unwrap();
            }
        }
    }
    let per_phase =
        2 * u64::from(host.shape.ranks()) * u64::from(host.shape.geometry.engram_layers());
    let expected = reasons
        .iter()
        .map(|reason| (format!("engram/sync/{reason}"), per_phase))
        .collect();
    assert_eq!(selected, expected);
    assert_eq!(
        host.mapped_gather_calls.get() - before,
        selected.values().sum::<u64>()
    );
}

fn dspark_instance(host: &Rc<Host>) -> BundleInstanceV1 {
    host.drafter.set(true);
    host.speculation.set(Some((3, 32)));
    let record = crate::config::tests::record(&host.shape, host.kv, 4, false);
    api()
        .create(
            HostServicesTableV1::new(Rc::new(dspark::host::MappedHost::new(Rc::clone(host)))),
            &record,
        )
        .unwrap()
}

#[test]
fn prefill_engram_prepares_ar_and_dspark_full_schedules() {
    bundle_runtime::routes::with_logging(|| {
        for dspark in [false, true] {
            let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
            // Expose mapped Engram tables for both AR and DSpark.
            host.drafter.set(true);
            let mut instance = if dspark {
                dspark_instance(&host)
            } else {
                api()
                    .create(
                        HostServicesTableV1::new(Rc::new(dspark::host::MappedHost::new(
                            Rc::clone(&host),
                        ))),
                        &crate::config::tests::record(&host.shape, host.kv, 4, false),
                    )
                    .unwrap()
            };
            let tokens = vec![7; 1024];
            let batch = step_with_stride(&[(701, 0, 0, &tokens, Phase::Prefill)], dspark, 16, None);
            assert_sync_engram(&host, &mut instance, &batch, &[]);
            assert!(
                host.journal
                    .borrow()
                    .iter()
                    .any(|line| line.starts_with("async mapped ids "))
            );
            instance.retire_request(701).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    });
}

#[cfg(feature = "diagnostic-taps")]
#[test]
fn prefill_engram_keeps_taps_reason() {
    if std::env::var_os(super::TAPS_CHILD).is_none() {
        let directory =
            std::env::temp_dir().join(format!("engram-reason-taps-{}", std::process::id()));
        super::in_child(
            "fake_host_tests::conformance::prefill_engram_keeps_taps_reason",
            &directory,
            &[("CHOPINFER_TAPS_SITES", "L02.x.hc.attn.pre")],
        );
        std::fs::remove_dir_all(directory).unwrap();
        return;
    }
    bundle_runtime::routes::with_logging(|| {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        let mut instance = dspark_instance(&host);
        let batch = step_with_stride(&[(701, 0, 0, &[1, 2, 3], Phase::Prefill)], true, 8, None);
        assert_sync_engram(&host, &mut instance, &batch, &["taps"]);
        instance.retire_request(701).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    });
}

/// A three-row prefill is outside the prepared verify-row gathers'
/// admitted range, so the mixed step's verify rows keep the synchronous
/// gather (`fake_host_tests::dspark::mixed_engram`).
#[test]
fn engram_reasons_distinguish_mixed_and_host_targets() {
    bundle_runtime::routes::with_logging(|| {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        let mut instance = dspark_instance(&host);
        let first = step_with_stride(&[(701, 0, 0, &[1, 2, 3], Phase::Prefill)], false, 8, None);
        assert_sync_engram(&host, &mut instance, &first, &[]);
        let mixed = step_with_stride(
            &[
                (702, 1, 0, &[1, 2, 3], Phase::Prefill),
                (701, 0, 3, &[4], Phase::Decode),
            ],
            false,
            8,
            None,
        );
        assert_sync_engram(&host, &mut instance, &mixed, &["mixed_target"]);
        let verify = step_with_stride(&[(701, 0, 4, &[5, 6, 7, 8], Phase::Verify)], false, 8, None);
        assert_sync_engram(&host, &mut instance, &verify, &["host_target"]);
        for request in [701, 702] {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    });
}

impl Served {
    fn create_with_prefill(&self, prefill: bool) -> Result<BundleInstanceV1, BundleAbiError> {
        if self.host.device.symbols_only.get() {
            // The mapped index also serves AR. Speculation is negotiated separately.
            self.host.drafter.set(true);
            self.host
                .kv_pages
                .set(1024 * u64::from(self.serving.sequences));
        }
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
        let mut api = if self.host.device.symbols_only.get() || prefill {
            api()
                .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
                .expect("prefill API")
                .with_prefill_cuts_api(crate::instance::prefill::PREFILL_CUTS_API)
                .expect("prefill cuts API")
        } else {
            api()
        };
        if prefill {
            api = api
                .with_prefill_anchors_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_ANCHORS_API_V2)
                .expect("prefill anchors API");
        }
        api.create(HostServicesTableV1::new(Rc::clone(&self.observed)), &record)
    }
}
