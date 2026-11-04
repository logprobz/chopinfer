//! The RAM prefix tier as `serve` runs it: the startup line the host logs,
//! and an engine served on host-memory device services, whose copy queue
//! moves a bundle's prefix snapshots between the serving KV pool and the
//! pinned offload arena.
use super::*;
use crate::fake::FakeHostDeviceV1;
use crate::offload::HostArenaPlanV1;
use crate::services::{
    DeclaredKvContractV1, HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1,
    ServingKvPolicyV1,
};
use runtime_bundle_abi::{
    BundleAbiError, BundleStatusV1, BundleStreamHandleV1, DeviceBufferSpanV1,
    HostServicesBackendV1, OwnedLogitsRowV1, ResultSinkLimitsV1,
};
use runtime_engine::{
    EngineEventKind, EngineRequest, PrefixRamStatusV1, PrefixRamUnavailableV1, RequestId,
};
use runtime_kv::{DeclaredPageLayoutV1, KvGeometry, RankPlacementV1, StateLifetimeV1};
use runtime_model_api::ModelVocabulary;
use runtime_sampling::SamplingParameters;
use runtime_scheduler::{OptimisticAdmission, SchedulerConfig};
use std::cell::RefCell;
use std::collections::BTreeMap;

const GIB: u64 = 1024 * 1024 * 1024;

#[test]
fn the_startup_line_gives_the_bounds_or_why_the_tier_is_off() {
    // GLM tp4-mtp: 454,656-byte pages per rank, a 64 GiB arena over four
    // ranks and a pool of 14,249 pages per rank.
    let plan = HostArenaPlanV1::new(64 * GIB, 256 * GIB, 454_656, 4).unwrap();
    let tier = prefix_ram_config_v1(false, 14_249);
    let line = |status| prefix_ram_startup_line_v1(status, Some(plan), 64, tier, true);
    assert_eq!(
        line(PrefixRamStatusV1::Active { forced: false }),
        "prefix RAM tier: on; 68718526464 pinned bytes across 4 ranks in the KV offload arena \
         it shares with swaps, 37786 pages and 2418304 tokens per rank, no staging buffer; \
         copies in flight at most 1073741824 bytes per rank, demotions at most 4 chunks \
         (32735232 bytes) per rank; promotions at most 890 pages per step"
    );
    assert_eq!(
        line(PrefixRamStatusV1::Active { forced: true }),
        "prefix RAM tier: on in its debug exactness mode (--debug-prefix-ram), where every \
         public capture is demoted and every hit is a RAM hit; 68718526464 pinned bytes across \
         4 ranks in the KV offload arena it shares with swaps, 37786 pages and 2418304 tokens \
         per rank, no staging buffer; copies in flight at most 1073741824 bytes per rank, \
         demotions at most 4 chunks (32735232 bytes) per rank; promotions at most 890 pages per \
         step"
    );
    for (reason, expected) in [
        (
            PrefixRamUnavailableV1::NoPrefixCache,
            "prefix RAM tier: off, because prefix caching is off",
        ),
        (
            PrefixRamUnavailableV1::NoOffload,
            "prefix RAM tier: off, because the executor offers no KV offload",
        ),
        (
            PrefixRamUnavailableV1::NoCancel,
            "prefix RAM tier: off, because the executor cannot cancel a demotion",
        ),
    ] {
        assert_eq!(line(PrefixRamStatusV1::Unavailable(reason)), expected);
    }
    // A bundle without the offload capability gets no arena to copy
    // through.
    assert_eq!(
        prefix_ram_startup_line_v1(
            PrefixRamStatusV1::Unavailable(PrefixRamUnavailableV1::NoOffload),
            None,
            64,
            tier,
            false,
        ),
        "prefix RAM tier: off, because the bundle does not declare chop.kv-offload.v1"
    );
}

const RANKS: usize = 2;
const LAYOUT_ID: &[u8] = b"prefix-ram-layout";
const PAGE_TOKENS: usize = 8;
const VOCABULARY: usize = 16;

/// Logits that depend on every token of the history.
fn logits_after(context: &[u32]) -> Vec<u8> {
    let favoured = context.iter().fold(7_usize, |hash, token| {
        hash.wrapping_mul(31).wrapping_add(*token as usize)
    }) % VOCABULARY;
    (0..VOCABULARY)
        .flat_map(|token| f32::to_le_bytes(if token == favoured { 8.0 } else { 0.0 }))
        .collect()
}

/// A bundle whose request state is its committed tokens. Its prefix
/// capture writes that state into the anchor's snapshot page on every rank,
/// each rank's bytes its own, and its restore reads the page back and
/// checks it, so a RAM hit restores only what the copies moved.
struct SnapshotBundle {
    services: Rc<HostServicesV1Backend<FakeHostDeviceV1>>,
    /// Each rank's stream and serving KV pool.
    ranks: Vec<(BundleStreamHandleV1, DeviceBufferSpanV1)>,
    page_bytes: u64,
    committed: BTreeMap<u64, Vec<u32>>,
    pending: Vec<(u64, Vec<u32>)>,
    /// The prefix each restore read.
    restores: Rc<RefCell<Vec<Vec<u32>>>>,
}

impl SnapshotBundle {
    fn page(&self, rank: usize, page: u64) -> DeviceBufferSpanV1 {
        let pool = self.ranks[rank].1;
        DeviceBufferSpanV1::new(
            pool.buffer,
            pool.offset + page * self.page_bytes,
            self.page_bytes,
        )
    }

    /// The snapshot bytes of `tokens` on `rank`.
    fn image(&self, rank: usize, tokens: &[u32]) -> Vec<u8> {
        let mut bytes: Vec<u8> = std::iter::once(tokens.len() as u32)
            .chain(tokens.iter().copied())
            .flat_map(|word| (word ^ ((rank as u32 + 1) * 0x0101_0101)).to_le_bytes())
            .collect();
        assert!(bytes.len() <= self.page_bytes as usize);
        bytes.resize(self.page_bytes as usize, rank as u8 + 1);
        bytes
    }
}

fn failed(status: BundleStatusV1) -> BundleAbiError {
    BundleAbiError::Status(status)
}

impl BundleExecutionV1 for SnapshotBundle {
    fn supports_request_retirement(&self) -> bool {
        true
    }

    fn retire_request(&mut self, request_id: u64) -> Result<(), BundleAbiError> {
        self.committed.remove(&request_id);
        Ok(())
    }

    fn capture_prefix_state(
        &mut self,
        request_id: u64,
        tokens: u64,
        pages: &[u64],
    ) -> Result<(), BundleAbiError> {
        let state = self.committed[&request_id].clone();
        assert_eq!(
            state.len() as u64,
            tokens,
            "a capture names the committed length"
        );
        for rank in 0..RANKS {
            let stream = self.ranks[rank].0;
            self.services
                .copy_host_to_device(stream, self.page(rank, pages[0]), &self.image(rank, &state))
                .map_err(failed)?;
            self.services.synchronize_stream(stream).map_err(failed)?;
        }
        Ok(())
    }

    fn restore_prefix_state(
        &mut self,
        request_id: u64,
        token_ids: &[u32],
        pages: &[u64],
    ) -> Result<(), BundleAbiError> {
        for rank in 0..RANKS {
            let stream = self.ranks[rank].0;
            let mut bytes = vec![0; self.page_bytes as usize];
            self.services
                .copy_device_to_host(stream, &mut bytes, self.page(rank, pages[0]))
                .map_err(failed)?;
            self.services.synchronize_stream(stream).map_err(failed)?;
            assert_eq!(
                bytes,
                self.image(rank, token_ids),
                "rank {rank} restores the bytes its capture wrote"
            );
        }
        self.committed.insert(request_id, token_ids.to_vec());
        self.restores.borrow_mut().push(token_ids.to_vec());
        Ok(())
    }

    fn execute_batch(
        &mut self,
        request: &[u8],
        _limits: ResultSinkLimitsV1,
    ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
        let request = runtime_bundle_sdk::BatchRequestV1::decode(request)
            .map_err(|_| failed(BundleStatusV1::InvalidArgument))?;
        let mut rows = Vec::new();
        for (item, append) in request.items().iter().zip(request.appends()) {
            let id = item.request_id();
            let mut context = self.committed.get(&id).cloned().unwrap_or_default();
            assert_eq!(
                context.len() as u64,
                append.previous_tokens(),
                "state matches KV"
            );
            context.extend_from_slice(item.input_tokens());
            if item.expects_logits() {
                rows.push(OwnedLogitsRowV1::new(id, logits_after(&context)));
            }
            self.pending.push((id, item.input_tokens().to_vec()));
        }
        Ok(rows)
    }

    fn commit(&mut self) -> Result<(), BundleAbiError> {
        for (id, inputs) in self.pending.drain(..) {
            self.committed.entry(id).or_default().extend(inputs);
        }
        Ok(())
    }

    fn cancel(&mut self) -> Result<(), BundleAbiError> {
        self.pending.clear();
        Ok(())
    }

    fn destroy(&mut self) -> Result<(), BundleAbiError> {
        Ok(())
    }

    fn is_poisoned(&self) -> bool {
        false
    }
}

struct Served {
    engine: Engine<ServedExecutorV1<SnapshotBundle>>,
    /// The startup line of a requested tier.
    line: Option<String>,
    /// The prefix each restore read.
    restores: Rc<RefCell<Vec<Vec<u32>>>>,
    /// Bytes one host page holds across every rank, without the arena zero.
    page_bytes: u64,
}

/// An engine served as `ServedEngineV1::open` builds one: on host-memory
/// device services with a serving KV pool of `pool_pages` and, with
/// `offload`, the pinned arena, with the tier `prefix_ram` asks for
/// configured over the pool.
fn served(pool_pages: u64, offload: bool, prefix_ram: Option<bool>) -> Served {
    let layout = DeclaredPageLayoutV1::new(
        64,
        2,
        PAGE_TOKENS as u32,
        64,
        RANKS as u8,
        RankPlacementV1::Sharded,
        StateLifetimeV1::Request,
        64,
    )
    .unwrap();
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
        .with_kv_contract(DeclaredKvContractV1::new(LAYOUT_ID.to_vec(), layout))
        .with_serving_kv(ServingKvPolicyV1 {
            maximum_pages: Some(pool_pages),
            ..Default::default()
        });
    let services = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let mut streams = Vec::new();
    for rank in 0..RANKS {
        let context = services.select_context(rank as u32, rank as i32).unwrap();
        services.select_memory_pool(context).unwrap();
        streams.push(services.create_stream(context, 0).unwrap());
    }
    let pages = services.acquire_serving_kv(LAYOUT_ID).unwrap();
    assert_eq!(pages, pool_pages);
    let pool_control: Rc<dyn ServingPoolControlV1> = services.clone();
    let tier =
        prefix_ram.map(|forced| prefix_ram_config_v1(forced, pool_control.pages().unwrap_or(0)));
    let config = EngineConfig::new(
        SchedulerConfig::new(2, 8, 32, 2 * PAGE_TOKENS, 1).unwrap(),
        KvGeometry::new(PAGE_TOKENS, pool_pages as usize).unwrap(),
        ModelVocabulary::new(VOCABULARY).unwrap(),
        24 * PAGE_TOKENS,
        2,
    )
    .unwrap()
    .with_kv_page_capacity(pages as usize)
    .unwrap()
    .with_admission(AdmissionPolicy::Optimistic(OptimisticAdmission::DEFAULT))
    .with_prefix_cache(1);
    let config = match tier {
        Some(tier) => config.with_prefix_ram(tier),
        None => config,
    };
    // As serving, the arena is pinned once the pool is acquired.
    let plan = offload.then(|| {
        services
            .initialize_kv_offload(
                crate::offload::KvOffloadOptionsV1 {
                    budget_bytes: 2 * pool_pages * 1024 * RANKS as u64,
                    minimum_tokens: 1,
                    bytes_per_second: None,
                },
                32 * GIB,
            )
            .unwrap()
            .unwrap()
    });
    let page_bytes = plan.map_or(1024, |plan| plan.page_bytes as u64);
    let ranks = (0..RANKS)
        .map(|rank| {
            let pool = services.resolve_serving_kv_pool(rank as u32).unwrap();
            assert_eq!(pool.offset, 0);
            assert!(pool.length >= pool_pages * page_bytes);
            (streams[rank], pool)
        })
        .collect();
    let restores = Rc::new(RefCell::new(Vec::new()));
    let bundle = SnapshotBundle {
        services: services.clone(),
        ranks,
        page_bytes,
        committed: BTreeMap::new(),
        pending: Vec::new(),
        restores: Rc::clone(&restores),
    };
    let executor = BundleModelExecutorV1::new(bundle, VOCABULARY, 2);
    let executor = match plan {
        Some(plan) => executor.with_kv_offload(
            services.clone(),
            runtime_engine::KvOffloadInfo {
                host_page_capacity: plan.pages_per_rank,
                minimum_tokens: 1,
                snapshot_pages: 1,
            },
        ),
        None => executor,
    };
    let engine = Engine::new_with_runtime_metrics(
        config,
        ServedExecutorV1 {
            executor,
            _services: services,
            pool_control: Some(pool_control),
            reference: None,
        },
        Arc::new(runtime_engine::SystemMonotonicClock::new()),
        EngineMetrics::default(),
        KvMetrics::default(),
    )
    .unwrap();
    let line = tier.map(|tier| {
        prefix_ram_startup_line_v1(engine.prefix_ram_status(), plan, PAGE_TOKENS, tier, offload)
    });
    Served {
        engine,
        line,
        restores,
        page_bytes: plan.map_or(0, |plan| (plan.page_bytes * plan.rank_count) as u64),
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Output {
    cached: Option<usize>,
    tokens: Vec<u32>,
    finished: bool,
}

/// Serves each request in turn, letting copies settle between them, and
/// returns what each produced.
fn run<E: runtime_engine::ModelExecutor>(
    engine: &mut Engine<E>,
    requests: &[(u64, &[u32])],
) -> BTreeMap<u64, Output> {
    let mut outputs: BTreeMap<u64, Output> = BTreeMap::new();
    for (id, prompt) in requests {
        engine
            .submit(
                EngineRequest::new(
                    RequestId::new(*id),
                    prompt.to_vec(),
                    3,
                    SamplingParameters::new(0.0).unwrap(),
                    *id,
                    Vec::new(),
                    ModelVocabulary::new(VOCABULARY).unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let mut steps = 0;
        while !outputs.get(id).is_some_and(|output| output.finished) {
            steps += 1;
            assert!(steps < 200, "request {id} did not finish");
            engine.step().unwrap();
            engine.audit().unwrap();
            for event in engine.drain_events() {
                let output = outputs.entry(event.request_id().get()).or_default();
                match event.kind() {
                    EngineEventKind::PromptCached { tokens } => output.cached = Some(*tokens),
                    EngineEventKind::Token { token_id, .. } => output.tokens.push(*token_id),
                    EngineEventKind::Finished { .. } => output.finished = true,
                    _ => {}
                }
            }
        }
        let mut polls = 0;
        while engine.poll_idle_kv_transfers().unwrap() {
            polls += 1;
            assert!(polls < 10_000, "copies stayed pending");
        }
        engine.audit().unwrap();
    }
    outputs
}

/// The engine's metrics as served.
fn prometheus<E: runtime_engine::ModelExecutor>(engine: &Engine<E>) -> String {
    let mut text = engine.metrics().snapshot().prometheus();
    text.push_str(&engine.kv_metrics().snapshot().prometheus());
    text
}

/// The value of one sample, `name` with its labels.
fn metric(text: &str, sample: &str) -> u64 {
    text.lines()
        .find_map(|line| line.strip_prefix(sample)?.strip_prefix(' '))
        .unwrap_or_else(|| panic!("missing {sample}"))
        .parse()
        .unwrap()
}

/// A prompt of `length` tokens.
fn prompt(salt: u32, length: usize) -> Vec<u32> {
    (0..length as u32)
        .map(|position| (salt * 7 + position * 3) % VOCABULARY as u32)
        .collect()
}

#[test]
fn a_fake_host_serve_runs_the_tier_and_its_forced_mode_hits_ram() {
    let a = prompt(1, 3 * PAGE_TOKENS);
    let mut b = a.clone();
    b.extend(prompt(2, PAGE_TOKENS));
    let mut c = a.clone();
    c.extend(prompt(3, 2 * PAGE_TOKENS));
    let requests: [(u64, &[u32]); 3] = [(1, &a), (2, &b), (3, &c)];
    // Without the tier, B and C resume A's anchor on the GPU.
    let Served {
        engine: mut gpu,
        line,
        ..
    } = served(32, true, None);
    assert_eq!(line, None);
    assert_eq!(gpu.prefix_ram_status(), PrefixRamStatusV1::Off);
    let expected = run(&mut gpu, &requests);
    let hits = gpu.kv_metrics().snapshot().prefix_cache_hits;
    assert!(hits >= 2, "{expected:?}");
    for forced in [false, true] {
        let Served {
            mut engine,
            line,
            restores,
            page_bytes,
        } = served(32, true, Some(forced));
        assert_eq!(
            engine.prefix_ram_status(),
            PrefixRamStatusV1::Active { forced }
        );
        let line = line.unwrap();
        assert!(line.starts_with("prefix RAM tier: on"), "{line}");
        assert_eq!(line.contains("--debug-prefix-ram"), forced, "{line}");
        assert!(
            line.ends_with("promotions at most 2 pages per step"),
            "{line}"
        );
        let outputs = run(&mut engine, &requests);
        assert_eq!(outputs, expected, "forced {forced}");
        let kv = engine.kv_metrics().snapshot();
        assert_eq!(kv.prefix_cache_hits, hits, "forced {forced}");
        let text = prometheus(&engine);
        // Bytes count whole host pages across every rank.
        let out = metric(
            &text,
            "chop_kv_offload_bytes_total{direction=\"out\",kind=\"prefix\"}",
        );
        let held = metric(&text, "chop_prefix_ram_bytes");
        for bytes in [out, held] {
            assert_eq!(bytes % page_bytes, 0, "forced {forced}");
        }
        if forced {
            // Every hit restored from a promoted copy of A's demoted
            // anchor, whose bytes the copy queue moved on every rank, and
            // forced mode keeps every host entry.
            assert!(out > 0 && held > 0, "{out} {held}");
            assert_eq!(kv.prefix_cache_ram_hits, hits);
            assert_eq!(
                metric(&text, "chop_prefix_ram_promotions_total{result=\"ok\"}"),
                hits
            );
            assert!(metric(&text, "chop_prefix_ram_demotions_total{result=\"ok\"}") > 0);
            assert_eq!(restores.borrow().len() as u64, hits);
            assert!(restores.borrow().iter().all(|prefix| a.starts_with(prefix)));
        } else {
            // A pool this size evicts nothing: every hit stays on the GPU.
            assert_eq!(kv.prefix_cache_ram_hits, 0);
        }
    }
}

#[test]
fn a_fake_host_serve_without_the_offload_arena_runs_without_the_tier() {
    let Served {
        mut engine,
        line,
        page_bytes,
        ..
    } = served(32, false, Some(false));
    assert_eq!(page_bytes, 0);
    assert_eq!(
        engine.prefix_ram_status(),
        PrefixRamStatusV1::Unavailable(PrefixRamUnavailableV1::NoOffload)
    );
    assert_eq!(
        line.unwrap(),
        "prefix RAM tier: off, because the bundle does not declare chop.kv-offload.v1"
    );
    let a = prompt(1, 3 * PAGE_TOKENS);
    let outputs = run(&mut engine, &[(1, &a), (2, &a)]);
    assert!(outputs.values().all(|output| output.finished));
    assert_eq!(engine.kv_metrics().snapshot().prefix_cache_ram_hits, 0);
}

#[test]
fn a_fake_host_serve_restores_an_anchor_the_gpu_evicted_from_ram() {
    // A's anchor is demoted ahead of its eviction by a filler that needs
    // most of the 24-page pool; B extends A's prompt.
    let a = prompt(1, 4 * PAGE_TOKENS);
    let filler = prompt(5, 20 * PAGE_TOKENS);
    let mut b = a.clone();
    b.extend(prompt(2, PAGE_TOKENS));
    let requests: [(u64, &[u32]); 3] = [(1, &a), (2, &filler), (3, &b)];
    let tokens = |outputs: &BTreeMap<u64, Output>| {
        outputs
            .iter()
            .map(|(id, output)| (*id, output.tokens.clone()))
            .collect::<Vec<_>>()
    };
    let mut off = served(24, true, None).engine;
    let cold = run(&mut off, &requests);
    assert_eq!(cold[&3].cached, None, "the filler evicted A on the GPU");
    let Served {
        mut engine,
        restores,
        page_bytes,
        ..
    } = served(24, true, Some(false));
    let outputs = run(&mut engine, &requests);
    assert_eq!(tokens(&outputs), tokens(&cold));
    assert_eq!(outputs[&3].cached, Some(a.len()));
    let kv = engine.kv_metrics().snapshot();
    assert_eq!((kv.prefix_cache_hits, kv.prefix_cache_ram_hits), (1, 1));
    assert_eq!(*restores.borrow(), std::slice::from_ref(&a));
    let text = prometheus(&engine);
    assert!(metric(&text, "chop_prefix_ram_demotions_total{result=\"ok\"}") >= 1);
    assert_eq!(
        metric(&text, "chop_prefix_ram_promotions_total{result=\"ok\"}"),
        1
    );
    // The promotion copied A's four KV pages and its snapshot page in on
    // both ranks.
    assert_eq!(
        metric(
            &text,
            "chop_kv_offload_bytes_total{direction=\"in\",kind=\"prefix\"}"
        ),
        5 * page_bytes
    );
    let out = metric(
        &text,
        "chop_kv_offload_bytes_total{direction=\"out\",kind=\"prefix\"}",
    );
    assert!(out >= 5 * page_bytes && out % page_bytes == 0, "{out}");
}
